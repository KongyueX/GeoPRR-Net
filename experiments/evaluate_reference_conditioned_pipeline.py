"""Evaluate the frozen reference-conditioned calibrator and selective router."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from experiments.calibrated_progress_router import (
    FEATURE_NAMES as ROUTER_FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix as router_feature_matrix,
)
from experiments.evaluate_progress_calibrator import (
    _base_prediction,
    _flat_prediction,
    _metrics,
    _paired_group_bootstrap,
)
from experiments.evaluate_uncertainty_fusion import _validate_vector_evaluation
from experiments.progress_calibrator import (
    FEATURE_NAMES as CALIBRATOR_FEATURE_NAMES,
    extract_progress_features,
    feature_matrix as calibrator_feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float, normalized_error, rows_by_id
from experiments.reference_conditioned_progress_calibrator import (
    REFERENCE_BRANCHES,
    REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
    normalize_reference_branch,
    predict_reference_conditioned_residual,
)
from experiments.reference_conditioned_router import (
    REFERENCE_CONDITIONED_ROUTER_PROTOCOL,
    deterministic_router_prediction,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


EVALUATION_PROTOCOL = "frozen_reference_conditioned_pipeline_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--legacy-router-predictions", type=Path)
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=Path(
            "artifacts/runs/reference_conditioned_progress_calibrator_syncg/"
            "model/reference_conditioned_calibrator.joblib"
        ),
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=Path(
            "artifacts/runs/reference_conditioned_router_syncg/"
            "model/reference_conditioned_router.joblib"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_calibrator(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if (
        artifact.get("protocol") != REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL
        or artifact.get("train_only_certified") is not True
        or artifact.get("strict_nested_oof") is not True
        or tuple(artifact.get("feature_names") or ()) != CALIBRATOR_FEATURE_NAMES
        or tuple(artifact.get("reference_branches") or ()) != REFERENCE_BRANCHES
        or artifact.get("test_sets_used") != []
    ):
        raise ValueError("reference-conditioned calibrator artifact audit failed")
    sources = artifact.get("source_sha256") or {}
    source_paths = {
        "features": PROJECT_DIR / "experiments" / "progress_calibrator.py",
        "reference_policy": (
            PROJECT_DIR / "experiments" / "reference_conditioned_progress_calibrator.py"
        ),
    }
    for name, source_path in source_paths.items():
        if sources.get(name) != sha256_file(source_path):
            raise ValueError(f"calibrator {name} source changed after fitting")
    return artifact


def _validate_router(path: Path, calibrator_path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if (
        artifact.get("protocol") != REFERENCE_CONDITIONED_ROUTER_PROTOCOL
        or artifact.get("train_only_certified") is not True
        or artifact.get("nested_threshold_selection") is not True
        or tuple(artifact.get("feature_names") or ()) != ROUTER_FEATURE_NAMES
        or artifact.get("test_sets_used") != []
        or artifact.get("calibrator_sha256") != sha256_file(calibrator_path)
    ):
        raise ValueError("reference-conditioned router artifact audit failed")
    sources = artifact.get("source_sha256") or {}
    source_paths = {
        "features": PROJECT_DIR / "experiments" / "calibrated_progress_router.py",
        "quality_features": PROJECT_DIR / "experiments" / "uncertainty_fusion.py",
        "policy": PROJECT_DIR / "experiments" / "quality_router.py",
        "router_protocol": (
            PROJECT_DIR / "experiments" / "reference_conditioned_router.py"
        ),
    }
    for name, source_path in source_paths.items():
        if sources.get(name) != sha256_file(source_path):
            raise ValueError(f"router {name} source changed after fitting")
    return artifact


def _same_identifier_set(
    mappings: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    anchor: str,
) -> list[str]:
    identifiers = list(mappings[anchor])
    expected = set(identifiers)
    for name, mapping in mappings.items():
        if set(mapping) != expected:
            raise ValueError(
                f"{name} IDs differ from {anchor}: "
                f"missing={len(expected - set(mapping))}, "
                f"extra={len(set(mapping) - expected)}"
            )
    return identifiers


def _route(
    base: float | None,
    calibrated: float | None,
    score: float,
    threshold: float,
) -> tuple[float | None, str]:
    if base is None:
        return (
            calibrated,
            "reference_conditioned_hard_fallback"
            if calibrated is not None
            else "failure",
        )
    if calibrated is not None and math.isfinite(score) and score > threshold:
        return calibrated, "reference_conditioned_quality_switch"
    return base, "base"


def _oracle(
    row: Mapping[str, Any],
    first: float | None,
    second: float | None,
) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return (
        second
        if normalized_error(row, second) < normalized_error(row, first)
        else first
    )


def main() -> None:
    args = parse_args()
    for name in (
        "raw_predictions",
        "base_predictions",
        "vector_predictions",
        "reference_predictions",
        "calibrator",
        "router",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.legacy_router_predictions is not None:
        args.legacy_router_predictions = args.legacy_router_predictions.resolve()
    required = (
        args.raw_predictions,
        args.base_predictions,
        args.vector_predictions,
        args.reference_predictions,
        args.calibrator,
        args.router,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.legacy_router_predictions is not None
        and not args.legacy_router_predictions.is_file()
    ):
        raise FileNotFoundError(args.legacy_router_predictions)
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")

    calibrator = _validate_calibrator(args.calibrator)
    router = _validate_router(args.router, args.calibrator)
    vector_audit = _validate_vector_evaluation(
        args.vector_predictions,
        args.condition,
    )
    mappings: dict[str, dict[str, dict[str, Any]]] = {
        "raw": rows_by_id(args.raw_predictions),
        "base": rows_by_id(args.base_predictions),
        "vector": rows_by_id(args.vector_predictions),
        "reference": rows_by_id(args.reference_predictions),
    }
    if args.legacy_router_predictions is not None:
        mappings["legacy_router"] = rows_by_id(args.legacy_router_predictions)
    identifiers = _same_identifier_set(mappings, anchor="vector")
    rows = [mappings["vector"][sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [
            str(row.get("group_id") or row.get("meter_id") or row.get("sample_id"))
            for row in rows
        ],
        dtype=object,
    )
    branches = [
        normalize_reference_branch(mappings["reference"][sample_id])
        for sample_id in identifiers
    ]
    calibration_matrix = calibrator_feature_matrix(
        [
            extract_progress_features(
                mappings["vector"][sample_id],
                reference_row=mappings["reference"][sample_id],
            )
            for sample_id in identifiers
        ]
    )
    residual = predict_reference_conditioned_residual(
        calibrator,
        calibration_matrix,
        branches,
    )

    raw_vector_values: list[float | None] = []
    calibrated_values: list[float | None] = []
    corrected_progress: list[float | None] = []
    calibration_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw_value = _flat_prediction(row)
        raw_progress = finite_float(row.get("progress"))
        progress = (
            float(
                np.clip(
                    raw_progress + residual["applied_residual"][index],
                    0.0,
                    1.0,
                )
            )
            if raw_value is not None and raw_progress is not None
            else None
        )
        calibrated = reading_from_progress(
            progress,
            row.get("scale_start"),
            row.get("scale_end"),
        )
        raw_vector_values.append(raw_value)
        corrected_progress.append(progress)
        calibrated_values.append(calibrated)
        calibration_rows.append(
            {
                "progress": progress,
                "raw_progress": raw_progress,
                "predicted_progress_residual": float(residual["model_residual"][index]),
                "applied_progress_residual": float(residual["applied_residual"][index]),
                "progress_ensemble_std": float(residual["ensemble_std"][index]),
            }
        )

    base_values = [
        _base_prediction(mappings["base"][sample_id]) for sample_id in identifiers
    ]
    router_matrix = router_feature_matrix(
        [
            extract_calibrated_router_features(
                raw_row=mappings["raw"][sample_id],
                base_row=mappings["base"][sample_id],
                vector_row=mappings["vector"][sample_id],
                reference_row=mappings["reference"][sample_id],
                calibration_row=calibration_rows[index],
            )
            for index, sample_id in enumerate(identifiers)
        ]
    )
    joint = np.asarray(
        [
            base is not None and calibrated is not None
            for base, calibrated in zip(base_values, calibrated_values)
        ]
    )
    router_scores = np.full(len(rows), math.nan, dtype=np.float64)
    if np.any(joint):
        router_scores[joint] = deterministic_router_prediction(
            router["estimator"],
            router_matrix[joint],
        )
    router_threshold = float(router["threshold"])
    routed_values: list[float | None] = []
    routes: list[str] = []
    for base, calibrated, score in zip(
        base_values,
        calibrated_values,
        router_scores,
    ):
        prediction, route = _route(
            base,
            calibrated,
            float(score),
            router_threshold,
        )
        routed_values.append(prediction)
        routes.append(route)

    vdn_values = [
        _flat_prediction(mappings["reference"][sample_id]) for sample_id in identifiers
    ]
    hard_values = [
        base if base is not None else calibrated
        for base, calibrated in zip(base_values, calibrated_values)
    ]
    oracle_values = [
        _oracle(row, base, calibrated)
        for row, base, calibrated in zip(
            rows,
            base_values,
            calibrated_values,
        )
    ]
    predictions: dict[str, Sequence[float | None]] = {
        "base_mask": base_values,
        "raw_probabilistic_vector": raw_vector_values,
        "reference_conditioned_vector": calibrated_values,
        "hard_fallback": hard_values,
        "reference_conditioned_router": routed_values,
        "vdn": vdn_values,
        "oracle": oracle_values,
    }
    if "legacy_router" in mappings:
        predictions["legacy_router"] = [
            _flat_prediction(mappings["legacy_router"][sample_id])
            for sample_id in identifiers
        ]

    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    successful: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name], successful[name] = _metrics(rows, values)
    switched = np.asarray(
        [route == "reference_conditioned_quality_switch" for route in routes]
    )
    calibration_success = successful["reference_conditioned_vector"]
    calibration_positive = calibration_success & (
        errors["reference_conditioned_vector"] < errors["raw_probabilistic_vector"]
    )
    calibration_negative = calibration_success & (
        errors["reference_conditioned_vector"] > errors["raw_probabilistic_vector"]
    )
    routing_positive = switched & (
        errors["reference_conditioned_vector"] < errors["base_mask"]
    )
    routing_negative = switched & (
        errors["reference_conditioned_vector"] > errors["base_mask"]
    )

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    if (output_predictions.exists() or output_summary.exists()) and not args.overwrite:
        raise FileExistsError(
            "reference-conditioned evaluation exists; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(identifiers):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "group_id": rows[index].get("group_id"),
                        "dataset": rows[index].get("dataset"),
                        "split": rows[index].get("split"),
                        "ground_truth": rows[index].get("ground_truth"),
                        "scale_start": rows[index].get("scale_start"),
                        "scale_end": rows[index].get("scale_end"),
                        "condition": args.condition,
                        "status": routed_values[index] is not None,
                        "prediction": routed_values[index],
                        "route": routes[index],
                        "reference_branch": branches[index],
                        "base_prediction": base_values[index],
                        "raw_vector_prediction": raw_vector_values[index],
                        "reference_conditioned_prediction": calibrated_values[index],
                        "raw_progress": finite_float(rows[index].get("progress")),
                        "corrected_progress": corrected_progress[index],
                        "model_progress_residual": float(
                            residual["model_residual"][index]
                        ),
                        "applied_progress_residual": float(
                            residual["applied_residual"][index]
                        ),
                        "progress_ensemble_std": float(residual["ensemble_std"][index]),
                        "correction_clip": float(residual["correction_clip"][index]),
                        "deadband": float(residual["deadband"][index]),
                        "calibration_applied": bool(
                            residual["applied_residual"][index] != 0.0
                        ),
                        "used_branch_model": bool(residual["used_branch_model"][index]),
                        "router_score": finite_float(router_scores[index]),
                        "router_threshold": router_threshold,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    comparison_targets = {
        "base_mask": "router_vs_base_mask",
        "reference_conditioned_vector": ("router_vs_reference_conditioned_vector"),
        "raw_probabilistic_vector": "router_vs_raw_probabilistic_vector",
        "vdn": "router_vs_vdn",
        "legacy_router": "router_vs_legacy_router",
    }
    comparisons: dict[str, Any] = {}
    for offset, (name, comparison_name) in enumerate(comparison_targets.items()):
        if name not in errors:
            continue
        comparisons[comparison_name] = _paired_group_bootstrap(
            errors["reference_conditioned_router"],
            errors[name],
            groups,
            seed=args.seed + offset,
            iterations=args.bootstrap_iterations,
        )

    per_branch: dict[str, Any] = {}
    branch_array = np.asarray(branches, dtype=object)
    for branch in REFERENCE_BRANCHES:
        selected = branch_array == branch
        if not np.any(selected):
            continue
        branch_metrics: dict[str, Any] = {}
        for name, values in predictions.items():
            subset_rows = [row for row, keep in zip(rows, selected) if keep]
            subset_values = [value for value, keep in zip(values, selected) if keep]
            branch_metrics[name], _, _ = _metrics(
                subset_rows,
                subset_values,
            )
        per_branch[branch] = {
            "samples": int(np.sum(selected)),
            "metrics": branch_metrics,
            "calibration_apply_rate": float(
                np.mean(residual["applied_residual"][selected] != 0.0)
            ),
        }

    valid_uncertainty = calibration_success & np.isfinite(residual["ensemble_std"])
    uncertainty_correlation = (
        float(
            np.corrcoef(
                residual["ensemble_std"][valid_uncertainty],
                errors["reference_conditioned_vector"][valid_uncertainty],
            )[0, 1]
        )
        if int(np.sum(valid_uncertainty)) > 1
        else math.nan
    )
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "condition": args.condition,
        "samples": len(rows),
        "groups": int(len(np.unique(groups))),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "per_reference_branch": per_branch,
        "calibration": {
            "applied": int(np.sum(residual["applied_residual"] != 0.0)),
            "abstained": int(np.sum(residual["applied_residual"] == 0.0)),
            "positive_transfers": int(np.sum(calibration_positive)),
            "negative_transfers": int(np.sum(calibration_negative)),
            "mean_ensemble_std": float(
                np.mean(residual["ensemble_std"][valid_uncertainty])
            ),
            "ensemble_std_error_correlation": uncertainty_correlation,
        },
        "routing": {
            "threshold": router_threshold,
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(np.sum(routing_positive)),
            "negative_transfers": int(np.sum(routing_negative)),
        },
        "artifacts": {
            "calibrator": str(args.calibrator),
            "calibrator_sha256": sha256_file(args.calibrator),
            "router": str(args.router),
            "router_sha256": sha256_file(args.router),
            "training_oof_pairs_sha256": calibrator.get("training_oof_pairs_sha256"),
        },
        "inputs": {
            "raw": str(args.raw_predictions),
            "raw_sha256": sha256_file(args.raw_predictions),
            "base": str(args.base_predictions),
            "base_sha256": sha256_file(args.base_predictions),
            "vector": str(args.vector_predictions),
            "vector_sha256": sha256_file(args.vector_predictions),
            "vector_audit": vector_audit,
            "reference_vdn": str(args.reference_predictions),
            "reference_vdn_sha256": sha256_file(args.reference_predictions),
            "legacy_router": (
                str(args.legacy_router_predictions)
                if args.legacy_router_predictions
                else None
            ),
            "legacy_router_sha256": (
                sha256_file(args.legacy_router_predictions)
                if args.legacy_router_predictions
                else None
            ),
        },
        "test_labels_used_for_selection": 0,
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(output_summary)


if __name__ == "__main__":
    main()
