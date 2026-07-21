"""Train and evaluate quality-weighted selective residual correction.

The ``fit`` command uses grouped out-of-fold (OOF) predictions for both the
residual model and its apply/reject gate. The ``evaluate`` command never tunes
on its input set, so it is suitable for the untouched RPM-10K external test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from experiments.datasets import (
    SYNCG_PINNED_COMMIT,
    SYNCG_SAMPLE_IDS_SHA256,
    SYNCG_TRAIN_ROWS,
    syncg_sample_ids_sha256,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANGLE_DIR = PROJECT_DIR / "utils" / "angleDetect"
if str(ANGLE_DIR) not in sys.path:
    sys.path.insert(0, str(ANGLE_DIR))

from residual_calibrator import (  # noqa: E402
    SELECTIVE_FEATURE_COLUMNS,
    make_features,
)


METHOD_LABELS = {
    "transformer": "Original Transformer",
    "geometry_v1": "Geometry-v1",
    "geometry_v2": "Geometry-v2",
    "mean_fusion": "Mean Fusion",
    "weighted_fusion": "Quality-weighted Fusion",
    "residual_ungated": "Residual without Gate",
    "ours": "Ours",
}

FEATURE_SETS = {
    "full": list(SELECTIVE_FEATURE_COLUMNS),
    "geometry": [
        "p_geom",
        "p_geom_v2",
        "p_fusion",
        "v1_v2_progress_delta",
        "endNum",
        "v1_confidence",
        "v2_confidence",
        "fusion_weight_v1",
        "fusion_weight_v2",
        "startAngle",
        "endAngle",
        "disAngle",
    ],
    "no_mask": [
        column
        for column in SELECTIVE_FEATURE_COLUMNS
        if not column.startswith(("mask_", "seg_"))
    ],
    "no_ellipse": [
        column for column in SELECTIVE_FEATURE_COLUMNS if not column.startswith("ellipse_")
    ],
    "disagreement": [
        "p_geom",
        "p_geom_v2",
        "p_fusion",
        "v1_v2_progress_delta",
        "v1_confidence",
        "v2_confidence",
        "fusion_weight_v1",
        "fusion_weight_v2",
    ],
}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            key = (row.get("dataset"), row.get("split"), row.get("sample_id"))
            if key in seen:
                raise ValueError(f"{path}:{line_number} duplicates sample key {key}")
            seen.add(key)
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _prediction_cache_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_name(path.name + ".meta.json")
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata if isinstance(metadata, dict) else None


def _prediction_cache_signature(path: Path) -> dict[str, Any] | None:
    metadata = _prediction_cache_metadata(path)
    if metadata is None:
        return None
    signature = metadata.get("signature")
    return signature if isinstance(signature, dict) else None


def _validate_formal_syncg_training_cache(
    path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = _prediction_cache_metadata(path)
    if metadata is None:
        raise ValueError(
            "formal SyncG fitting requires prediction-cache metadata; "
            "re-run experiments.collect_predictions or pass "
            "--allow-other-training-data for a diagnostic"
        )
    protocol = metadata.get("manifest_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            "prediction cache has no embedded SyncG manifest protocol; "
            "rebuild the manifest/cache with the formal pipeline"
        )
    expected = {
        "protocol": "syncg_official_split_v1",
        "dataset": "SyncG",
        "split": "train",
        "reference_huggingface_commit": SYNCG_PINNED_COMMIT,
        "release_identity_verified": True,
        "strict_release": True,
        "expected_rows": SYNCG_TRAIN_ROWS,
        "emitted_rows": SYNCG_TRAIN_ROWS,
        "sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
        "expected_sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                "prediction cache is not the pinned formal SyncG train split: "
                f"{key}={protocol.get(key)!r}, expected {expected_value!r}"
            )
    if len(rows) != SYNCG_TRAIN_ROWS:
        raise ValueError(
            f"formal SyncG cache has {len(rows)} rows; "
            f"expected {SYNCG_TRAIN_ROWS}"
        )
    actual_ids_sha256 = syncg_sample_ids_sha256(
        str(row.get("sample_id") or "") for row in rows
    )
    if actual_ids_sha256 != SYNCG_SAMPLE_IDS_SHA256["train"]:
        raise ValueError(
            "prediction-cache sample identifiers do not match the pinned "
            "SyncG train split"
        )
    signature = metadata.get("signature")
    if (
        not isinstance(signature, dict)
        or not signature.get("manifest_sha256")
        or not signature.get("manifest_protocol_sha256")
    ):
        raise ValueError(
            "prediction cache is missing manifest/protocol content hashes"
        )
    return protocol


def _front_end_signature(signature: dict[str, Any] | None) -> dict[str, Any] | None:
    if signature is None:
        return None
    # The manifest and hardware may differ between train/test; the actual
    # front-end weights, model source, and inference protocol must not.  The
    # controlled input degradation is intentionally a test-set property.  The
    # collector only serializes predictions, so its own hash is excluded while
    # hashes for every model component remain checked.
    normalized = {
        key: value
        for key, value in signature.items()
        if key not in {
            "manifest_sha256",
            "manifest_protocol_sha256",
            "device",
            "input_degradation",
            "input_degradation_source_sha256",
        }
    }
    source_hashes = normalized.get("source_sha256")
    if isinstance(source_hashes, dict):
        normalized["source_sha256"] = {
            key: value
            for key, value in source_hashes.items()
            if key != "collector"
        }
    return normalized


def _method_prediction(row: dict[str, Any], method: str) -> float | None:
    method_payload = (row.get("methods") or {}).get(method) or {}
    if not method_payload.get("status"):
        return None
    return _finite(method_payload.get("prediction"))


def _base_valid_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    valid = []
    for row in rows:
        ground_truth = _finite(row.get("ground_truth"))
        scale_start = _finite(row.get("scale_start"))
        scale_end = _finite(row.get("scale_end"))
        base = _method_prediction(row, "weighted_fusion")
        if None in (ground_truth, scale_start, scale_end, base):
            continue
        if abs(scale_end - scale_start) <= 1e-12:
            continue
        valid.append(row)
    return valid, len(rows) - len(valid)


def _feature_matrix(rows: list[dict[str, Any]], columns: list[str]) -> np.ndarray:
    return np.vstack(
        [make_features(row.get("features") or {}, columns) for row in rows]
    ).astype(np.float32)


def _arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    ground_truth = np.asarray([float(row["ground_truth"]) for row in rows], dtype=np.float64)
    scale_start = np.asarray([float(row["scale_start"]) for row in rows], dtype=np.float64)
    scale_end = np.asarray([float(row["scale_end"]) for row in rows], dtype=np.float64)
    base = np.asarray(
        [_method_prediction(row, "weighted_fusion") for row in rows],
        dtype=np.float64,
    )
    span = scale_end - scale_start
    return ground_truth, scale_start, scale_end, base, span


def _groups(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            str(row.get("group_id") or row.get("meter_id") or row.get("sample_id"))
            for row in rows
        ],
        dtype=object,
    )


def _regressor(seed: int, trees: int, min_samples_leaf: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
        max_features=1.0,
        n_jobs=-1,
        random_state=seed,
    )


def _classifier(seed: int, trees: int, min_samples_leaf: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _predict_regressor(
    model: ExtraTreesRegressor,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(model.predict(x), dtype=np.float64)
    estimators = list(getattr(model, "estimators_", []))
    if not estimators:
        return prediction, np.zeros_like(prediction)
    tree_prediction = np.vstack(
        [np.asarray(estimator.predict(x), dtype=np.float64) for estimator in estimators]
    )
    return prediction, np.std(tree_prediction, axis=0)


def _fit_gate(
    x: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    trees: int,
    min_samples_leaf: int,
):
    classes = np.unique(target)
    if classes.size < 2:
        model = DummyClassifier(strategy="constant", constant=int(classes[0]))
    else:
        model = _classifier(seed, trees, min_samples_leaf)
    model.fit(x, target)
    return model


def _predict_positive_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        if len(classes) == 1:
            return np.full(
                x.shape[0],
                1.0 if classes[0] in (1, True, "1") else 0.0,
                dtype=np.float64,
            )
        return probabilities[:, -1]
    return np.asarray(model.predict(x), dtype=np.float64)


def _gate_matrix(
    features: np.ndarray,
    residual: np.ndarray,
    residual_std: np.ndarray,
) -> np.ndarray:
    return np.column_stack((features, residual, np.abs(residual), residual_std)).astype(
        np.float32
    )


def _group_folds(groups: np.ndarray, requested_folds: int) -> GroupKFold:
    unique_groups = np.unique(groups)
    n_splits = min(max(2, requested_folds), unique_groups.size)
    if unique_groups.size < 2:
        raise ValueError(
            "grouped OOF needs at least two group_id values; fix the manifest grouping"
        )
    return GroupKFold(n_splits=n_splits)


def _clip_to_scale(
    prediction: np.ndarray,
    scale_start: np.ndarray,
    scale_end: np.ndarray,
) -> np.ndarray:
    low = np.minimum(scale_start, scale_end)
    high = np.maximum(scale_start, scale_end)
    return np.clip(prediction, low, high)


def _normalized_abs_error(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    span: np.ndarray,
) -> np.ndarray:
    return np.abs(prediction - ground_truth) / np.maximum(np.abs(span), 1e-12)


def _residual_clip_value(
    target_raw: np.ndarray,
    quantile: float,
    maximum: float,
) -> float:
    """Choose a normalized residual cap from the current training partition."""
    values = np.asarray(target_raw, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot select residual clip from an empty partition")
    return min(
        float(maximum),
        max(1e-6, float(np.quantile(np.abs(values), quantile))),
    )


def _select_threshold(
    probability: np.ndarray,
    residual_std: np.ndarray,
    corrected_prediction: np.ndarray,
    base_prediction: np.ndarray,
    ground_truth: np.ndarray,
    span: np.ndarray,
    max_residual_std: float,
    min_coverage: float,
) -> tuple[float, dict[str, Any]]:
    base_error = _normalized_abs_error(base_prediction, ground_truth, span)
    corrected_error = _normalized_abs_error(corrected_prediction, ground_truth, span)
    if not np.any(corrected_error < base_error):
        return 1.000001, {
            "reason": "no_oof_improvement",
            "coverage": 0.0,
            "nmae": float(np.mean(base_error)),
            "negative_transfer_rate": None,
        }

    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 1.0, 201),
                probability,
                np.asarray([1.000001]),
            )
        )
    )
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in candidates:
        applied = (probability >= threshold) & (residual_std <= max_residual_std)
        coverage = float(np.mean(applied))
        if coverage + 1e-12 < min_coverage:
            continue
        system_prediction = np.where(applied, corrected_prediction, base_prediction)
        system_error = _normalized_abs_error(system_prediction, ground_truth, span)
        if np.any(applied):
            negative_transfer = float(np.mean(corrected_error[applied] > base_error[applied] + 1e-12))
        else:
            negative_transfer = 0.0
        objective = (float(np.mean(system_error)), negative_transfer, -coverage)
        summary = {
            "reason": "minimum_oof_system_nmae",
            "coverage": coverage,
            "nmae": objective[0],
            "negative_transfer_rate": negative_transfer,
        }
        if best is None or objective < best[0]:
            best = (objective, float(threshold), summary)

    if best is None:
        # The requested minimum coverage can be infeasible after the std gate.
        return _select_threshold(
            probability,
            residual_std,
            corrected_prediction,
            base_prediction,
            ground_truth,
            span,
            max_residual_std,
            min_coverage=0.0,
        )
    return best[1], best[2]


def _ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            selected = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if not np.any(selected):
            continue
        error += float(np.mean(selected)) * abs(
            float(np.mean(probability[selected])) - float(np.mean(target[selected]))
        )
    return error


def _gate_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "brier": float(brier_score_loss(target, probability)),
        "ece_10_bins": _ece(probability, target, bins=10),
        "positive_rate": float(np.mean(target)),
    }
    if np.unique(target).size >= 2:
        result["auroc"] = float(roc_auc_score(target, probability))
    else:
        result["auroc"] = None
    return result


def _bootstrap_group_ci(
    errors: np.ndarray,
    group_values: np.ndarray,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if iterations <= 0 or errors.size == 0:
        return None
    unique_groups = np.unique(group_values)
    if unique_groups.size < 2:
        return None
    indices = {group: np.flatnonzero(group_values == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sampled_errors = np.concatenate([errors[indices[group]] for group in sampled_groups])
        estimates.append(float(np.mean(sampled_errors)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def _method_metrics(
    all_rows: list[dict[str, Any]],
    predictions: list[float | None],
    *,
    seed: int,
    bootstrap_iterations: int,
    base_predictions: list[float | None] | None = None,
    applied: list[bool] | None = None,
) -> dict[str, Any]:
    total = len(all_rows)
    eligible_indices = [
        index
        for index, row in enumerate(all_rows)
        if _finite(row.get("ground_truth")) is not None
        and _finite(row.get("scale_start")) is not None
        and _finite(row.get("scale_end")) is not None
        and abs(float(row["scale_end"]) - float(row["scale_start"])) > 1e-12
    ]
    valid_indices = [
        index for index in eligible_indices if predictions[index] is not None
    ]
    if not eligible_indices:
        return {
            "samples": total,
            "eligible": 0,
            "successful": 0,
            "coverage": 0.0,
            "mae": None,
            "rmse": None,
            "nmae": None,
            "successful_nmae": None,
            "nrmse": None,
            "acc_1pct": None,
            "acc_2pct": None,
            "range_nmae_capped_10": None,
            "relative_mae_capped_100": None,
            "acc_relative_5pct": None,
            "dialbench_ref_successful": None,
            "dialbench_rel_successful": None,
            "dialbench_acc_epsilon_e2e": None,
            "dialbench_acc_theta_e2e": None,
            "relative_samples": 0,
        }

    eligible_ground_truth = np.asarray(
        [float(all_rows[index]["ground_truth"]) for index in eligible_indices],
        dtype=np.float64,
    )
    eligible_position = {
        global_index: position
        for position, global_index in enumerate(eligible_indices)
    }
    normalized_error_e2e = np.ones(len(eligible_indices), dtype=np.float64)
    normalized_error_success = np.asarray([], dtype=np.float64)
    absolute_error = np.asarray([], dtype=np.float64)
    error = np.asarray([], dtype=np.float64)
    if valid_indices:
        prediction = np.asarray(
            [predictions[index] for index in valid_indices],
            dtype=np.float64,
        )
        ground_truth = np.asarray(
            [float(all_rows[index]["ground_truth"]) for index in valid_indices],
            dtype=np.float64,
        )
        span = np.asarray(
            [
                float(all_rows[index]["scale_end"]) - float(all_rows[index]["scale_start"])
                for index in valid_indices
            ],
            dtype=np.float64,
        )
        error = prediction - ground_truth
        absolute_error = np.abs(error)
        normalized_error_success = (
            absolute_error / np.maximum(np.abs(span), 1e-12)
        )
        for global_index, value in zip(valid_indices, normalized_error_success):
            normalized_error_e2e[eligible_position[global_index]] = float(value)

    # Accuracy denominators include every evaluable sample; a front-end failure
    # is never silently removed from the headline score.
    success_at_1pct = int(np.sum(normalized_error_success <= 0.01))
    success_at_2pct = int(np.sum(normalized_error_success <= 0.02))
    relative_eligible_positions = np.flatnonzero(
        np.abs(eligible_ground_truth) > 1e-12
    )
    relative_error_e2e = np.full(
        relative_eligible_positions.size,
        100.0,
        dtype=np.float64,
    )
    relative_position = {
        eligible_indices[position]: relative_index
        for relative_index, position in enumerate(relative_eligible_positions)
    }
    relative_successful = 0
    relative_error_successful: list[float] = []
    for global_index in valid_indices:
        relative_index = relative_position.get(global_index)
        if relative_index is None:
            continue
        value = abs(
            float(predictions[global_index])
            - float(all_rows[global_index]["ground_truth"])
        ) / abs(float(all_rows[global_index]["ground_truth"]))
        relative_error_e2e[relative_index] = value
        relative_successful += 1
        relative_error_successful.append(float(value))

    meter_groups = np.asarray(
        [
            str(
                all_rows[index].get("meter_id")
                or all_rows[index].get("group_id")
                or "unknown"
            )
            for index in eligible_indices
        ],
        dtype=object,
    )
    bootstrap_groups = np.asarray(
        [
            str(all_rows[index].get("group_id") or all_rows[index].get("meter_id") or index)
            for index in eligible_indices
        ],
        dtype=object,
    )
    meter_nmae = [
        float(np.mean(normalized_error_e2e[meter_groups == group]))
        for group in np.unique(meter_groups)
    ]
    range_error_e2e = np.full(len(eligible_indices), 10.0, dtype=np.float64)
    for global_index, value in zip(valid_indices, normalized_error_success):
        range_error_e2e[eligible_position[global_index]] = float(
            np.clip(value, 0.0, 10.0)
        )
    result: dict[str, Any] = {
        "samples": total,
        "eligible": len(eligible_indices),
        "successful": len(valid_indices),
        "coverage": float(len(valid_indices) / max(total, 1)),
        # Absolute-unit errors remain success-only because meter ranges vary.
        "mae": float(np.mean(absolute_error)) if absolute_error.size else None,
        "rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
        # Headline normalized metrics are end-to-end. Missing predictions get
        # the worst possible in-range normalized error (1.0).
        "nmae": float(np.mean(normalized_error_e2e)),
        "nmae_failure_penalty": 1.0,
        "successful_nmae": (
            float(np.mean(normalized_error_success))
            if normalized_error_success.size
            else None
        ),
        "nrmse": float(np.sqrt(np.mean(normalized_error_e2e ** 2))),
        "acc_1pct": float(success_at_1pct / len(eligible_indices)),
        "acc_2pct": float(success_at_2pct / len(eligible_indices)),
        "successful_acc_1pct": (
            float(success_at_1pct / len(valid_indices)) if valid_indices else None
        ),
        "successful_acc_2pct": (
            float(success_at_2pct / len(valid_indices)) if valid_indices else None
        ),
        # DialBench/RPM-10K-style capped errors use their cap as the explicit
        # failure penalty, instead of dropping samples without a scalar output.
        "range_nmae_capped_10": float(np.mean(range_error_e2e)),
        "relative_mae_capped_100": (
            float(np.mean(np.clip(relative_error_e2e, 0.0, 100.0)))
            if relative_error_e2e.size
            else None
        ),
        "acc_relative_5pct": (
            float(np.mean(relative_error_e2e < 0.05))
            if relative_error_e2e.size
            else None
        ),
        # Official DialBench formulas, kept explicitly separate from the
        # capped robustness diagnostics above. Ref/Rel averages are defined
        # only for successful scalar outputs; the two accuracies use the full
        # evaluable denominator and therefore count a missing output as wrong.
        "dialbench_ref_successful": (
            float(np.mean(normalized_error_success))
            if normalized_error_success.size
            else None
        ),
        "dialbench_rel_successful": (
            float(np.mean(relative_error_successful))
            if relative_error_successful
            else None
        ),
        "dialbench_acc_epsilon_e2e": float(
            success_at_1pct / len(eligible_indices)
        ),
        "dialbench_acc_theta_e2e": (
            float(np.mean(relative_error_e2e < 0.05))
            if relative_error_e2e.size
            else None
        ),
        "relative_samples": int(relative_error_e2e.size),
        "relative_successful": int(relative_successful),
        "macro_meter_nmae": float(np.mean(meter_nmae)),
        "nmae_group_bootstrap_95ci": _bootstrap_group_ci(
            normalized_error_e2e,
            bootstrap_groups,
            bootstrap_iterations,
            seed,
        ),
    }

    if base_predictions is not None and applied is not None:
        comparable = [
            index
            for index in valid_indices
            if base_predictions[index] is not None and bool(applied[index])
        ]
        result["correction_coverage"] = float(
            sum(bool(value) for value in applied) / max(total, 1)
        )
        if comparable:
            corrected_error = np.asarray(
                [
                    abs(float(predictions[index]) - float(all_rows[index]["ground_truth"]))
                    for index in comparable
                ],
                dtype=np.float64,
            )
            base_error = np.asarray(
                [
                    abs(float(base_predictions[index]) - float(all_rows[index]["ground_truth"]))
                    for index in comparable
                ],
                dtype=np.float64,
            )
            result["negative_transfer_rate"] = float(
                np.mean(corrected_error > base_error + 1e-12)
            )
            result["correction_improvement_rate"] = float(
                np.mean(corrected_error < base_error - 1e-12)
            )
        else:
            result["negative_transfer_rate"] = None
            result["correction_improvement_rate"] = None
    return result


def _risk_coverage_rows(
    probability: np.ndarray,
    residual_std: np.ndarray,
    corrected_prediction: np.ndarray,
    base_prediction: np.ndarray,
    ground_truth: np.ndarray,
    span: np.ndarray,
    max_residual_std: float,
) -> list[dict[str, Any]]:
    base_error = _normalized_abs_error(base_prediction, ground_truth, span)
    corrected_error = _normalized_abs_error(corrected_prediction, ground_truth, span)
    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        applied = (probability >= threshold) & (residual_std <= max_residual_std)
        system_prediction = np.where(applied, corrected_prediction, base_prediction)
        system_error = _normalized_abs_error(system_prediction, ground_truth, span)
        selected_risk = float(np.mean(corrected_error[applied])) if np.any(applied) else None
        negative_transfer = (
            float(np.mean(corrected_error[applied] > base_error[applied] + 1e-12))
            if np.any(applied)
            else None
        )
        rows.append(
            {
                "threshold": float(threshold),
                "correction_coverage": float(np.mean(applied)),
                "selected_nmae": selected_risk,
                "system_nmae": float(np.mean(system_error)),
                "negative_transfer_rate": negative_transfer,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_table(
    all_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    ungated: np.ndarray,
    ours: np.ndarray,
    applied: np.ndarray,
    probability: np.ndarray,
    residual: np.ndarray,
    residual_std: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, list[float | None]], list[bool]]:
    by_key = {
        (row.get("dataset"), row.get("split"), row.get("sample_id")): index
        for index, row in enumerate(valid_rows)
    }
    method_predictions: dict[str, list[float | None]] = {
        method: [] for method in METHOD_LABELS
    }
    applied_all: list[bool] = []
    output_rows = []
    for row in all_rows:
        key = (row.get("dataset"), row.get("split"), row.get("sample_id"))
        valid_index = by_key.get(key)
        predictions = {
            method: _method_prediction(row, method)
            for method in (
                "transformer",
                "geometry_v1",
                "geometry_v2",
                "mean_fusion",
                "weighted_fusion",
            )
        }
        if valid_index is None:
            predictions["residual_ungated"] = None
            predictions["ours"] = None
            apply_value = False
            gate_value = None
            residual_value = None
            std_value = None
        else:
            predictions["residual_ungated"] = float(ungated[valid_index])
            predictions["ours"] = float(ours[valid_index])
            apply_value = bool(applied[valid_index])
            gate_value = float(probability[valid_index])
            residual_value = float(residual[valid_index])
            std_value = float(residual_std[valid_index])
        for method in METHOD_LABELS:
            method_predictions[method].append(predictions[method])
        applied_all.append(apply_value)
        output_rows.append(
            {
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "sample_id": row.get("sample_id"),
                "group_id": row.get("group_id"),
                "meter_id": row.get("meter_id"),
                "ground_truth": row.get("ground_truth"),
                "scale_start": row.get("scale_start"),
                "scale_end": row.get("scale_end"),
                "metadata": row.get("metadata"),
                "degradation": row.get("degradation"),
                "predictions": predictions,
                "gate_probability": gate_value,
                "residual_normalized": residual_value,
                "residual_std_normalized": std_value,
                "correction_applied": apply_value,
            }
        )
    return output_rows, method_predictions, applied_all


def _summarize(
    all_rows: list[dict[str, Any]],
    method_predictions: dict[str, list[float | None]],
    applied: list[bool],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    base = method_predictions["weighted_fusion"]
    metrics = {}
    for index, (method, label) in enumerate(METHOD_LABELS.items()):
        kwargs: dict[str, Any] = {}
        if method == "residual_ungated":
            kwargs.update(
                base_predictions=base,
                applied=[value is not None for value in method_predictions[method]],
            )
        elif method == "ours":
            kwargs.update(base_predictions=base, applied=applied)
        metrics[label] = _method_metrics(
            all_rows,
            method_predictions[method],
            seed=seed + index,
            bootstrap_iterations=bootstrap_iterations,
            **kwargs,
        )
    return metrics


def _subgroup_summaries(
    all_rows: list[dict[str, Any]],
    method_predictions: dict[str, list[float | None]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Report frozen per-meter and per-condition robustness without tuning."""

    memberships: dict[str, dict[str, list[int]]] = {
        "meter_type": {},
        "environment_condition": {},
    }
    for index, row in enumerate(all_rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        meter_type = str(
            metadata.get("meter_type") or row.get("meter_id") or ""
        ).strip()
        if meter_type:
            memberships["meter_type"].setdefault(meter_type, []).append(index)

        conditions = metadata.get("environment_conditions")
        if isinstance(conditions, str):
            conditions = conditions.split(",")
        if isinstance(conditions, list):
            for condition in sorted(
                {
                    str(value).strip()
                    for value in conditions
                    if str(value).strip()
                }
            ):
                memberships["environment_condition"].setdefault(
                    condition, []
                ).append(index)

    summaries: dict[str, Any] = {}
    for dimension, groups in memberships.items():
        dimension_summary = {}
        for group_number, (group, indices) in enumerate(sorted(groups.items())):
            subset_rows = [all_rows[index] for index in indices]
            dimension_summary[group] = {
                label: _method_metrics(
                    subset_rows,
                    [method_predictions[method][index] for index in indices],
                    seed=seed + group_number * len(METHOD_LABELS) + method_number,
                    bootstrap_iterations=0,
                )
                for method_number, (method, label) in enumerate(METHOD_LABELS.items())
            }
        if dimension_summary:
            summaries[dimension] = dimension_summary
    return summaries


def _paired_comparison(
    rows: list[dict[str, Any]],
    candidate: list[float | None],
    baseline: list[float | None],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    indices = [
        index
        for index in range(len(rows))
        if _finite(rows[index].get("ground_truth")) is not None
        and _finite(rows[index].get("scale_start")) is not None
        and _finite(rows[index].get("scale_end")) is not None
        and abs(float(rows[index]["scale_end"]) - float(rows[index]["scale_start"])) > 1e-12
    ]
    if not indices:
        return {
            "paired_samples": 0,
            "common_successes": 0,
            "delta_nmae": None,
            "delta_nmae_group_bootstrap_95ci": None,
            "candidate_better_rate": None,
        }
    delta = []
    groups = []
    common_successes = 0
    for index in indices:
        ground_truth = float(rows[index]["ground_truth"])
        span = abs(float(rows[index]["scale_end"]) - float(rows[index]["scale_start"]))
        candidate_error = (
            1.0
            if candidate[index] is None
            else abs(float(candidate[index]) - ground_truth) / span
        )
        baseline_error = (
            1.0
            if baseline[index] is None
            else abs(float(baseline[index]) - ground_truth) / span
        )
        common_successes += int(
            candidate[index] is not None and baseline[index] is not None
        )
        delta.append(candidate_error - baseline_error)
        groups.append(
            str(rows[index].get("group_id") or rows[index].get("meter_id") or index)
        )
    delta_array = np.asarray(delta, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    return {
        "paired_samples": len(indices),
        "common_successes": common_successes,
        "failure_penalty_nmae": 1.0,
        "delta_nmae": float(np.mean(delta_array)),
        "delta_nmae_group_bootstrap_95ci": _bootstrap_group_ci(
            delta_array,
            group_array,
            bootstrap_iterations,
            seed,
        ),
        "candidate_better_rate": float(np.mean(delta_array < -1e-12)),
    }


def _paired_comparisons(
    rows: list[dict[str, Any]],
    method_predictions: dict[str, list[float | None]],
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    pairs = (
        ("Ours vs Original Transformer", "ours", "transformer"),
        ("Quality-weighted Fusion vs Mean Fusion", "weighted_fusion", "mean_fusion"),
        ("Ours vs Quality-weighted Fusion", "ours", "weighted_fusion"),
        ("Ours vs Residual without Gate", "ours", "residual_ungated"),
    )
    return {
        label: _paired_comparison(
            rows,
            method_predictions[candidate],
            method_predictions[baseline],
            seed=seed + index,
            bootstrap_iterations=bootstrap_iterations,
        )
        for index, (label, candidate, baseline) in enumerate(pairs)
    }


def _fit(args: argparse.Namespace) -> None:
    all_rows = _read_jsonl(args.train_predictions)
    if not args.allow_other_training_data:
        datasets = {str(row.get("dataset") or "").strip().lower() for row in all_rows}
        splits = {str(row.get("split") or "").strip().lower() for row in all_rows}
        if datasets != {"syncg"} or splits != {"train"}:
            raise ValueError(
                "fit is locked to dataset=SyncG, split=train; "
                "pass --allow-other-training-data only for a clearly labelled non-paper run"
            )
        _validate_formal_syncg_training_cache(
            args.train_predictions,
            all_rows,
        )
    rows, skipped = _base_valid_rows(all_rows)
    if len(rows) < 10:
        raise ValueError(f"only {len(rows)} valid weighted-fusion samples; need at least 10")
    columns = FEATURE_SETS[args.feature_set]
    x = _feature_matrix(rows, columns)
    ground_truth, scale_start, scale_end, base, span = _arrays(rows)
    groups = _groups(rows)

    target_raw = (ground_truth - base) / span
    clip_value = _residual_clip_value(
        target_raw,
        args.residual_clip_quantile,
        args.max_residual_clip,
    )
    target = np.clip(target_raw, -clip_value, clip_value)
    splitter = _group_folds(groups, args.folds)
    if np.unique(groups).size < 3:
        raise ValueError(
            "nested grouped OOF for the learned gate needs at least three "
            "group_id values"
        )
    outer_splits = list(splitter.split(x, target_raw, groups))

    oof_residual = np.zeros(len(rows), dtype=np.float64)
    oof_std = np.zeros(len(rows), dtype=np.float64)
    gate_probability_oof = np.zeros(len(rows), dtype=np.float64)
    outer_clip_values: list[float] = []
    inner_clip_values: list[float] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, 1):
        # The residual prediction and uncertainty for the outer test groups use
        # only outer-train groups. The target clipping value is fitted on that
        # same partition, rather than leaking labels from the outer test groups.
        outer_clip = _residual_clip_value(
            target_raw[outer_train],
            args.residual_clip_quantile,
            args.max_residual_clip,
        )
        outer_clip_values.append(outer_clip)
        outer_model = _regressor(
            args.seed + outer_fold,
            args.trees,
            args.residual_min_samples_leaf,
        )
        outer_model.fit(
            x[outer_train],
            np.clip(
                target_raw[outer_train],
                -outer_clip,
                outer_clip,
            ),
        )
        outer_residual, outer_std = _predict_regressor(
            outer_model,
            x[outer_test],
        )
        outer_residual = np.clip(outer_residual, -outer_clip, outer_clip)
        oof_residual[outer_test] = outer_residual
        oof_std[outer_test] = outer_std

        # Gate training targets must not be built from residual models that saw
        # an outer-test group. Cross-fit the residual model again inside the
        # outer training partition, then fit the gate only on those inner-OOF
        # predictions and targets.
        inner_groups = groups[outer_train]
        inner_splitter = _group_folds(inner_groups, args.folds)
        inner_residual = np.zeros(len(outer_train), dtype=np.float64)
        inner_std = np.zeros(len(outer_train), dtype=np.float64)
        for inner_fold, (inner_train, inner_test) in enumerate(
            inner_splitter.split(
                x[outer_train],
                target_raw[outer_train],
                inner_groups,
            ),
            1,
        ):
            inner_model = _regressor(
                args.seed + outer_fold * 100 + inner_fold,
                args.trees,
                args.residual_min_samples_leaf,
            )
            inner_train_indices = outer_train[inner_train]
            inner_clip = _residual_clip_value(
                target_raw[inner_train_indices],
                args.residual_clip_quantile,
                args.max_residual_clip,
            )
            inner_clip_values.append(inner_clip)
            inner_model.fit(
                x[outer_train][inner_train],
                np.clip(
                    target_raw[inner_train_indices],
                    -inner_clip,
                    inner_clip,
                ),
            )
            prediction, prediction_std = _predict_regressor(
                inner_model,
                x[outer_train][inner_test],
            )
            inner_residual[inner_test] = np.clip(
                prediction,
                -inner_clip,
                inner_clip,
            )
            inner_std[inner_test] = prediction_std

        inner_corrected = _clip_to_scale(
            base[outer_train] + inner_residual * span[outer_train],
            scale_start[outer_train],
            scale_end[outer_train],
        )
        inner_base_error = _normalized_abs_error(
            base[outer_train],
            ground_truth[outer_train],
            span[outer_train],
        )
        inner_corrected_error = _normalized_abs_error(
            inner_corrected,
            ground_truth[outer_train],
            span[outer_train],
        )
        inner_gate_target = (
            inner_corrected_error + args.improvement_margin < inner_base_error
        ).astype(np.int64)
        inner_gate_x = _gate_matrix(
            x[outer_train],
            inner_residual,
            inner_std,
        )
        outer_gate = _fit_gate(
            inner_gate_x,
            inner_gate_target,
            seed=args.seed + 1000 + outer_fold,
            trees=args.gate_trees,
            min_samples_leaf=args.gate_min_samples_leaf,
        )
        gate_probability_oof[outer_test] = _predict_positive_probability(
            outer_gate,
            _gate_matrix(x[outer_test], outer_residual, outer_std),
        )

    corrected_oof = _clip_to_scale(
        base + oof_residual * span,
        scale_start,
        scale_end,
    )
    base_error = _normalized_abs_error(base, ground_truth, span)
    corrected_error = _normalized_abs_error(corrected_oof, ground_truth, span)
    gate_target = (corrected_error + args.improvement_margin < base_error).astype(np.int64)
    gate_x = _gate_matrix(x, oof_residual, oof_std)

    max_residual_std = float(np.quantile(oof_std, args.std_quantile))
    threshold, threshold_summary = _select_threshold(
        gate_probability_oof,
        oof_std,
        corrected_oof,
        base,
        ground_truth,
        span,
        max_residual_std,
        args.min_correction_coverage,
    )
    applied_oof = (gate_probability_oof >= threshold) & (oof_std <= max_residual_std)
    ours_oof = np.where(applied_oof, corrected_oof, base)

    final_model = _regressor(args.seed, args.trees, args.residual_min_samples_leaf)
    final_model.fit(x, target)
    final_gate = _fit_gate(
        gate_x,
        gate_target,
        seed=args.seed + 1000,
        trees=args.gate_trees,
        min_samples_leaf=args.gate_min_samples_leaf,
    )
    package = {
        "model": final_model,
        "model_type": "ExtraTreesRegressor",
        "gate_model": final_gate,
        "gate_model_type": type(final_gate).__name__,
        "feature_columns": columns,
        "feature_set": args.feature_set,
        "residual_unit": "normalized_range",
        "residual_clip": clip_value,
        "min_abs_residual": None,
        "max_abs_residual": None,
        "max_residual_std": max_residual_std,
        "boundary_margin": 0.0,
        "gate_apply_threshold": threshold,
        "backend_name": "geometry_fusion_weighted_calibrated",
        "training": {
            "source": str(args.train_predictions.resolve()),
            "prediction_cache_signature": _prediction_cache_signature(
                args.train_predictions
            ),
            "samples_total": len(all_rows),
            "samples_used": len(rows),
            "samples_skipped": skipped,
            "groups": int(np.unique(groups).size),
            "folds": splitter.n_splits,
            "gate_oof_protocol": "nested_grouped_cross_fit",
            "seed": args.seed,
            "improvement_margin_normalized": args.improvement_margin,
            "residual_clip_quantile": args.residual_clip_quantile,
            "oof_residual_clip_protocol": "fit_within_each_training_partition",
            "outer_residual_clip_range": [
                float(min(outer_clip_values)),
                float(max(outer_clip_values)),
            ],
            "inner_residual_clip_range": [
                float(min(inner_clip_values)),
                float(max(inner_clip_values)),
            ],
            "std_quantile": args.std_quantile,
            "threshold_selection": threshold_summary,
            "gate_positive_rate": float(np.mean(gate_target)),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = args.output_dir / "calibrator.joblib"
    joblib.dump(package, calibrator_path)
    output_rows, method_predictions, applied_all = _prediction_table(
        all_rows,
        rows,
        corrected_oof,
        ours_oof,
        applied_oof,
        gate_probability_oof,
        oof_residual,
        oof_std,
    )
    _write_jsonl(args.output_dir / "oof_predictions.jsonl", output_rows)
    metrics = _summarize(
        all_rows,
        method_predictions,
        applied_all,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    summary = {
        "protocol": "grouped_oof_training_diagnostics",
        "metrics": metrics,
        "subgroups": _subgroup_summaries(
            all_rows,
            method_predictions,
            seed=args.seed + 1_000,
        ),
        "paired_comparisons": _paired_comparisons(
            all_rows,
            method_predictions,
            seed=args.seed + 100,
            bootstrap_iterations=args.bootstrap_iterations,
        ),
        "gate": _gate_metrics(gate_probability_oof, gate_target),
        "threshold": threshold,
        "max_residual_std": max_residual_std,
        "residual_clip": clip_value,
        "oof_residual_clip_protocol": "fit_within_each_training_partition",
        "outer_residual_clip_range": [
            float(min(outer_clip_values)),
            float(max(outer_clip_values)),
        ],
        "inner_residual_clip_range": [
            float(min(inner_clip_values)),
            float(max(inner_clip_values)),
        ],
        "feature_set": args.feature_set,
        "feature_columns": columns,
        "samples_total": len(all_rows),
        "samples_used": len(rows),
        "samples_skipped": skipped,
        "groups": int(np.unique(groups).size),
        "gate_oof_protocol": "nested_grouped_cross_fit",
        "calibrator": str(calibrator_path.resolve()),
    }
    _write_json(args.output_dir / "training_summary.json", summary)
    _write_csv(
        args.output_dir / "oof_risk_coverage.csv",
        _risk_coverage_rows(
            gate_probability_oof,
            oof_std,
            corrected_oof,
            base,
            ground_truth,
            span,
            max_residual_std,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    package = joblib.load(args.calibrator)
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("invalid calibrator package")
    if package.get("residual_unit") != "normalized_range":
        raise ValueError(
            "this evaluator expects a normalized_range package produced by the fit command"
        )
    training_signature = (package.get("training") or {}).get(
        "prediction_cache_signature"
    )
    evaluation_signature = _prediction_cache_signature(args.predictions)
    front_end_verified = False
    if training_signature is not None and evaluation_signature is not None:
        front_end_verified = (
            _front_end_signature(training_signature)
            == _front_end_signature(evaluation_signature)
        )
        if not front_end_verified and not args.allow_front_end_mismatch:
            raise ValueError(
                "front-end signature mismatch between training and evaluation caches; "
                "rerun collection consistently or pass --allow-front-end-mismatch "
                "for a clearly labelled diagnostic"
            )
    all_rows = _read_jsonl(args.predictions)
    rows, skipped = _base_valid_rows(all_rows)
    if not rows:
        raise ValueError("no valid weighted-fusion predictions to evaluate")
    columns = list(package.get("feature_columns") or SELECTIVE_FEATURE_COLUMNS)
    x = _feature_matrix(rows, columns)
    ground_truth, scale_start, scale_end, base, span = _arrays(rows)

    residual, residual_std = _predict_regressor(package["model"], x)
    clip_value = float(package.get("residual_clip", 0.08))
    residual = np.clip(residual, -abs(clip_value), abs(clip_value))
    corrected = _clip_to_scale(base + residual * span, scale_start, scale_end)
    gate_x = _gate_matrix(x, residual, residual_std)
    gate_model = package.get("gate_model")
    if gate_model is None:
        probability = np.ones(len(rows), dtype=np.float64)
    else:
        probability = _predict_positive_probability(gate_model, gate_x)
    threshold = float(package.get("gate_apply_threshold", 0.5))
    max_residual_std = package.get("max_residual_std")
    if max_residual_std is None:
        max_residual_std = float("inf")
    else:
        max_residual_std = float(max_residual_std)
    applied = (probability >= threshold) & (residual_std <= max_residual_std)
    ours = np.where(applied, corrected, base)

    output_rows, method_predictions, applied_all = _prediction_table(
        all_rows,
        rows,
        corrected,
        ours,
        applied,
        probability,
        residual,
        residual_std,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "predictions.jsonl", output_rows)
    metrics = _summarize(
        all_rows,
        method_predictions,
        applied_all,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    base_error = _normalized_abs_error(base, ground_truth, span)
    corrected_error = _normalized_abs_error(corrected, ground_truth, span)
    improvement_margin = float(
        (package.get("training") or {}).get("improvement_margin_normalized", 0.0)
    )
    gate_target = (
        corrected_error + improvement_margin < base_error
    ).astype(np.int64)
    summary = {
        "protocol": "frozen_model_evaluation",
        "source_predictions": str(args.predictions.resolve()),
        "prediction_cache_signature": _prediction_cache_signature(args.predictions),
        "front_end_signature_verified": front_end_verified,
        "calibrator": str(args.calibrator.resolve()),
        "metrics": metrics,
        "subgroups": _subgroup_summaries(
            all_rows,
            method_predictions,
            seed=args.seed + 1_000,
        ),
        "paired_comparisons": _paired_comparisons(
            all_rows,
            method_predictions,
            seed=args.seed + 100,
            bootstrap_iterations=args.bootstrap_iterations,
        ),
        "gate": _gate_metrics(probability, gate_target),
        "threshold": threshold,
        "max_residual_std": None if math.isinf(max_residual_std) else max_residual_std,
        "residual_clip": clip_value,
        "improvement_margin_normalized": improvement_margin,
        "feature_set": package.get("feature_set"),
        "samples_total": len(all_rows),
        "samples_used": len(rows),
        "samples_skipped": skipped,
    }
    _write_json(args.output_dir / "metrics.json", summary)
    _write_csv(
        args.output_dir / "risk_coverage.csv",
        _risk_coverage_rows(
            probability,
            residual_std,
            corrected,
            base,
            ground_truth,
            span,
            max_residual_std,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit grouped-OOF residual and gate models")
    fit.add_argument("--train-predictions", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--feature-set", choices=tuple(FEATURE_SETS), default="full")
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--trees", type=int, default=300)
    fit.add_argument("--gate-trees", type=int, default=300)
    fit.add_argument("--residual-min-samples-leaf", type=int, default=3)
    fit.add_argument("--gate-min-samples-leaf", type=int, default=5)
    fit.add_argument("--seed", type=int, default=20260720)
    fit.add_argument("--improvement-margin", type=float, default=0.0005)
    fit.add_argument("--min-correction-coverage", type=float, default=0.10)
    fit.add_argument("--std-quantile", type=float, default=0.95)
    fit.add_argument("--residual-clip-quantile", type=float, default=0.995)
    fit.add_argument("--max-residual-clip", type=float, default=0.15)
    fit.add_argument("--bootstrap-iterations", type=int, default=500)
    fit.add_argument(
        "--allow-other-training-data",
        action="store_true",
        help="override the SyncG/train protocol guard for non-paper diagnostics",
    )
    fit.set_defaults(func=_fit)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a frozen SyncG-trained package without tuning",
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--calibrator", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, default=20260720)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=500)
    evaluate.add_argument(
        "--allow-front-end-mismatch",
        action="store_true",
        help="permit inconsistent weights/source/protocol for a non-paper diagnostic",
    )
    evaluate.set_defaults(func=_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probability_args = (
        "std_quantile",
        "residual_clip_quantile",
        "min_correction_coverage",
    )
    for name in probability_args:
        if hasattr(args, name):
            value = float(getattr(args, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    args.func(args)


if __name__ == "__main__":
    main()
