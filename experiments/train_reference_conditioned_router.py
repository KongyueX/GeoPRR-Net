"""Train a nested grouped router for mask and reference-conditioned readings."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

from experiments.calibrated_progress_router import (
    FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix,
)
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.reference_conditioned_progress_calibrator import (
    REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
)
from experiments.reference_conditioned_router import (
    REFERENCE_CONDITIONED_ROUTER_PROTOCOL,
    deterministic_router_prediction,
)
from experiments.train_quality_router import (
    _build_model,
    _choose_threshold,
    _paired_group_bootstrap,
)
from experiments.train_reference_conditioned_progress_calibrator import (
    TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_file,
    sha256_source_file,
)


TRAINING_PROTOCOL = "syncg_nested_reference_conditioned_router_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-pairs",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/probabilistic_oof_clean.jsonl"
        ),
    )
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        default=Path(
            "artifacts/runs/reference_conditioned_progress_calibrator_syncg/"
            "model/strict_nested_oof_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=Path(
            "artifacts/runs/reference_conditioned_progress_calibrator_syncg/"
            "model/training_summary.json"
        ),
    )
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=Path(
            "artifacts/runs/reference_conditioned_progress_calibrator_syncg/"
            "model/reference_conditioned_calibrator.joblib"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/reference_conditioned_router_syncg/model"),
    )
    parser.add_argument("--baseline-router-diagnostics", type=Path)
    parser.add_argument("--baseline-router-summary", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--expected-oof-protocol",
        default=UNCERTAINTY_FUSION_OOF_PROTOCOL,
        help=(
            "Exact protocol required in both the OOF metadata signature and "
            "OOF summary. Defaults to the legacy v1 contract; formal FADR v2 "
            "launchers must pass their frozen v2 protocol explicitly."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


def _validate_input_oof_contract(
    oof_pairs: Path,
    *,
    expected_protocol: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(expected_protocol, str) or not expected_protocol.strip():
        raise ValueError("expected OOF protocol must be a non-empty string")
    metadata_path = _metadata_path(oof_pairs)
    summary_path = _summary_path(oof_pairs)
    for path in (oof_pairs, metadata_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (metadata.get("signature") or {}).get("protocol") != expected_protocol:
        raise ValueError("input has the wrong probabilistic OOF protocol")
    if (
        summary.get("protocol") != expected_protocol
        or summary.get("status") != "complete"
        or summary.get("output_sha256") != sha256_file(oof_pairs)
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("probabilistic OOF collection failed its train-only audit")
    return metadata, summary


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _nested_prediction(row: dict[str, Any], name: str) -> float | None:
    nested = row.get(name)
    return finite_float(nested.get("prediction")) if isinstance(nested, dict) else None


def _metrics(errors: np.ndarray, successful: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(errors.size),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "acc_5pct": float(np.mean(errors <= 0.05)),
    }


def _cross_fitted_scores(
    args: argparse.Namespace,
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    splitter = GroupKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=seed,
    )
    scores = np.full(len(matrix), math.nan, dtype=np.float64)
    summaries: list[dict[str, int]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, groups),
        start=1,
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("inner reference-router group leakage")
        model = _build_model(args, seed=seed + fold)
        model.fit(matrix[train_index], target[train_index])
        scores[validation_index] = deterministic_router_prediction(
            model,
            matrix[validation_index],
        )
        summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("inner reference-router scores are incomplete")
    return scores, summaries


def _load_baseline_router(
    diagnostics_path: Path,
    summary_path: Path,
    *,
    input_path: Path,
    identifiers: list[str],
) -> tuple[list[float | None], dict[str, Any]]:
    from experiments.train_calibrated_progress_router import (
        TRAINING_PROTOCOL as BASELINE_ROUTER_TRAINING_PROTOCOL,
    )

    if not diagnostics_path.is_file():
        raise FileNotFoundError(diagnostics_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("protocol") != BASELINE_ROUTER_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("baseline router failed its train-only audit")
    rows = read_jsonl(diagnostics_path)
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(identifiers):
        raise ValueError("baseline router IDs differ from OOF input")
    values = [
        finite_float(by_id[sample_id].get("prediction")) for sample_id in identifiers
    ]
    return values, {
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "protocol": summary.get("protocol"),
    }


def main() -> None:
    args = parse_args()
    for name in (
        "oof_pairs",
        "calibration_diagnostics",
        "calibration_summary",
        "calibrator",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("baseline_router_diagnostics", "baseline_router_summary"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (args.baseline_router_diagnostics is None) != (
        args.baseline_router_summary is None
    ):
        raise ValueError(
            "--baseline-router-diagnostics and --baseline-router-summary "
            "must be supplied together"
        )
    if (
        args.folds < 3
        or args.inner_folds < 3
        or args.trees <= 0
        or args.min_samples_leaf <= 0
        or args.bootstrap_iterations <= 0
        or not isinstance(args.expected_oof_protocol, str)
        or not args.expected_oof_protocol.strip()
    ):
        raise ValueError("invalid reference-router training parameters")
    required = (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
        args.calibration_diagnostics,
        args.calibration_summary,
        args.calibrator,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_input_oof_contract(
        args.oof_pairs,
        expected_protocol=args.expected_oof_protocol,
    )
    calibration_summary = json.loads(
        args.calibration_summary.read_text(encoding="utf-8")
    )
    calibrator_artifact = joblib.load(args.calibrator)
    if (
        calibration_summary.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or calibration_summary.get("status") != "complete"
        or calibration_summary.get("input_sha256") != sha256_file(args.oof_pairs)
        or calibration_summary.get("diagnostics_sha256")
        != sha256_file(args.calibration_diagnostics)
        or calibration_summary.get("strict_nested_oof") is not True
        or int(calibration_summary.get("group_leakage_count", -1)) != 0
        or int(calibration_summary.get("test_samples_used", -1)) != 0
        or (
            calibration_summary.get("input_oof_protocol")
            not in (None, args.expected_oof_protocol)
        )
        or (
            calibration_summary.get("input_oof_protocol") is None
            and args.expected_oof_protocol != UNCERTAINTY_FUSION_OOF_PROTOCOL
        )
    ):
        raise ValueError(
            "reference-conditioned calibration failed its train-only audit"
        )
    if (
        calibrator_artifact.get("protocol") != REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL
        or calibrator_artifact.get("train_only_certified") is not True
        or calibrator_artifact.get("test_sets_used") != []
        or calibration_summary.get("model_sha256") != sha256_file(args.calibrator)
        or (
            calibrator_artifact.get("input_oof_protocol")
            not in (None, args.expected_oof_protocol)
        )
        or (
            calibrator_artifact.get("input_oof_protocol") is None
            and args.expected_oof_protocol != UNCERTAINTY_FUSION_OOF_PROTOCOL
        )
    ):
        raise ValueError("reference-conditioned calibrator artifact audit failed")

    rows = read_jsonl(args.oof_pairs)
    calibration_rows = read_jsonl(args.calibration_diagnostics)
    calibration_by_id = {str(row.get("sample_id")): row for row in calibration_rows}
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(
        calibration_by_id
    ):
        raise ValueError("calibration diagnostics IDs differ from OOF input")
    if any(
        row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows
    ):
        raise ValueError("reference-conditioned router accepts only SyncG/train")

    base = [_nested_prediction(row, "base") for row in rows]
    calibrated = [
        finite_float(calibration_by_id[sample_id].get("corrected_prediction_oof"))
        for sample_id in identifiers
    ]
    base_success = np.asarray([value is not None for value in base])
    calibrated_success = np.asarray([value is not None for value in calibrated])
    joint = base_success & calibrated_success
    if int(np.sum(joint)) < 1000:
        raise ValueError("too few joint-success calibrated OOF rows")
    groups_all = np.asarray(
        [str(row.get("group_id")) for row in rows],
        dtype=object,
    )
    feature_rows = [
        extract_calibrated_router_features(
            raw_row=row,
            base_row=row,
            vector_row=row,
            reference_row=None,
            calibration_row=calibration_by_id[sample_id],
        )
        for row, sample_id in zip(rows, identifiers)
    ]
    matrix_all = feature_matrix(feature_rows)
    matrix = matrix_all[joint]
    groups = groups_all[joint]
    base_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, base)],
        dtype=np.float64,
    )
    calibrated_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, calibrated)],
        dtype=np.float64,
    )
    target = np.clip(
        base_error[joint] - calibrated_error[joint],
        -1.0,
        1.0,
    )
    joint_indices = np.flatnonzero(joint)

    splitter = GroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    score = np.full(len(rows), math.nan, dtype=np.float64)
    threshold_oof = np.full(len(rows), math.nan, dtype=np.float64)
    fold_id = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    nested_thresholds: dict[str, float] = {}
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, groups),
        start=1,
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("outer reference-router group leakage")
        inner_score, inner_summaries = _cross_fitted_scores(
            args,
            matrix[train_index],
            target[train_index],
            groups[train_index],
            seed=args.seed + fold * 10_000,
        )
        threshold, selection = _choose_threshold(
            inner_score,
            base_error[joint][train_index],
            calibrated_error[joint][train_index],
        )
        model = _build_model(args, seed=args.seed + fold * 1_000)
        model.fit(matrix[train_index], target[train_index])
        destination = joint_indices[validation_index]
        score[destination] = deterministic_router_prediction(
            model,
            matrix[validation_index],
        )
        threshold_oof[destination] = threshold
        fold_id[destination] = fold
        nested_thresholds[str(fold)] = float(threshold)
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
                "inner_folds": inner_summaries,
                "inner_threshold_selection": {
                    "threshold": float(threshold),
                    **selection,
                },
            }
        )
    if (
        not np.isfinite(score[joint]).all()
        or not np.isfinite(threshold_oof[joint]).all()
        or np.any(fold_id[joint] < 1)
    ):
        raise RuntimeError("nested reference-router OOF scores are incomplete")

    final_threshold, threshold_diagnostic = _choose_threshold(
        score[joint],
        base_error[joint],
        calibrated_error[joint],
    )
    use_calibrated = np.zeros(len(rows), dtype=bool)
    use_calibrated[joint] = score[joint] > threshold_oof[joint]
    use_calibrated[~base_success & calibrated_success] = True

    routed: list[float | None] = []
    routes: list[str] = []
    for base_value, calibrated_value, switch in zip(
        base,
        calibrated,
        use_calibrated,
    ):
        if base_value is None:
            routed.append(calibrated_value)
            routes.append(
                "calibrated_hard_fallback"
                if calibrated_value is not None
                else "failure"
            )
        elif switch and calibrated_value is not None:
            routed.append(calibrated_value)
            routes.append("calibrated_quality_switch")
        else:
            routed.append(base_value)
            routes.append("base")
    routed_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, routed)],
        dtype=np.float64,
    )
    routed_success = np.asarray([value is not None for value in routed])
    hard = [
        first if first is not None else second
        for first, second in zip(base, calibrated)
    ]
    hard_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, hard)],
        dtype=np.float64,
    )
    hard_success = base_success | calibrated_success
    oracle_error = np.minimum(base_error, calibrated_error)

    final_model = _build_model(args, seed=args.seed)
    final_model.fit(matrix, target)
    model_path = args.output_dir / "reference_conditioned_router.joblib"
    diagnostics_path = args.output_dir / "strict_nested_oof_routing.jsonl"
    summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    source_hashes = {
        "features": sha256_source_file(
            PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
        ),
        "quality_features": sha256_source_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ),
        "policy": sha256_source_file(
            PROJECT_DIR / "experiments" / "quality_router.py"
        ),
        "router_protocol": sha256_source_file(
            PROJECT_DIR / "experiments" / "reference_conditioned_router.py"
        ),
        "trainer": sha256_source_file(Path(__file__).resolve()),
        "shared_training_helpers": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_quality_router.py"
        ),
    }
    training_parameters = {
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "trees": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    artifact = {
        "protocol": REFERENCE_CONDITIONED_ROUTER_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "nested_threshold_selection": True,
        "feature_names": list(FEATURE_NAMES),
        "threshold": float(final_threshold),
        "estimator": final_model,
        "input_oof_protocol": args.expected_oof_protocol,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
        "calibration_diagnostics_sha256": sha256_file(args.calibration_diagnostics),
        "calibrator_sha256": sha256_file(args.calibrator),
        "failure_policy": (
            "base failure -> reference-conditioned vector; joint failure -> failure"
        ),
        "test_sets_used": [],
        "seed": args.seed,
        "training_parameters": training_parameters,
        "source_sha256": source_hashes,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
    }
    _atomic_joblib(model_path, artifact)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row.get("sample_id"),
                        "group_id": row.get("group_id"),
                        "router_fold": (int(fold_id[index]) if joint[index] else None),
                        "router_score_oof": finite_float(score[index]),
                        "nested_threshold": finite_float(threshold_oof[index]),
                        "route": routes[index],
                        "prediction": routed[index],
                        "base_prediction": base[index],
                        "reference_conditioned_prediction": calibrated[index],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    regressor = final_model.named_steps["regressor"]
    transformed_names = final_model.named_steps["imputer"].get_feature_names_out(
        FEATURE_NAMES
    )
    importance = [
        {"feature": str(name), "importance": float(value)}
        for name, value in sorted(
            zip(transformed_names, regressor.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    switched = np.asarray([route == "calibrated_quality_switch" for route in routes])
    comparisons: dict[str, Any] = {
        "router_vs_base_mask": _paired_group_bootstrap(
            routed_error,
            base_error,
            groups_all,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ),
        "router_vs_reference_conditioned_vector": _paired_group_bootstrap(
            routed_error,
            calibrated_error,
            groups_all,
            seed=args.seed + 1,
            iterations=args.bootstrap_iterations,
        ),
        "router_vs_hard_fallback": _paired_group_bootstrap(
            routed_error,
            hard_error,
            groups_all,
            seed=args.seed + 2,
            iterations=args.bootstrap_iterations,
        ),
    }
    baseline_metrics = None
    baseline_audit = None
    if (
        args.baseline_router_diagnostics is not None
        and args.baseline_router_summary is not None
    ):
        baseline, baseline_audit = _load_baseline_router(
            args.baseline_router_diagnostics,
            args.baseline_router_summary,
            input_path=args.oof_pairs,
            identifiers=identifiers,
        )
        baseline_error = np.asarray(
            [normalized_error(row, value) for row, value in zip(rows, baseline)],
            dtype=np.float64,
        )
        baseline_success = np.asarray([value is not None for value in baseline])
        baseline_metrics = _metrics(baseline_error, baseline_success)
        comparisons["router_vs_global_calibrator_router_v1"] = _paired_group_bootstrap(
            routed_error,
            baseline_error,
            groups_all,
            seed=args.seed + 3,
            iterations=args.bootstrap_iterations,
        )

    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "seed": args.seed,
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "input_oof_protocol": args.expected_oof_protocol,
        "calibration_diagnostics": str(args.calibration_diagnostics),
        "calibration_diagnostics_sha256": sha256_file(args.calibration_diagnostics),
        "calibration_summary": str(args.calibration_summary),
        "calibration_summary_sha256": sha256_file(args.calibration_summary),
        "calibrator": str(args.calibrator),
        "calibrator_sha256": sha256_file(args.calibrator),
        "samples": len(rows),
        "joint_training_samples": int(np.sum(joint)),
        "groups": len(set(groups_all.tolist())),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "training_parameters": training_parameters,
        "nested_threshold_selection": True,
        "fold_summaries": fold_summaries,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": importance,
        "threshold": {
            "final": float(final_threshold),
            "selection_diagnostic": threshold_diagnostic,
            "nested_outer_fold": nested_thresholds,
        },
        "metrics": {
            "base_mask": _metrics(base_error, base_success),
            "reference_conditioned_vector": _metrics(
                calibrated_error,
                calibrated_success,
            ),
            "hard_fallback": _metrics(hard_error, hard_success),
            "reference_conditioned_router_nested_oof": _metrics(
                routed_error,
                routed_success,
            ),
            "global_calibrator_router_v1": baseline_metrics,
            "oracle": _metrics(oracle_error, hard_success),
        },
        "paired_comparisons": comparisons,
        "routing": {
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(
                np.sum(switched & (calibrated_error < base_error))
            ),
            "negative_transfers": int(
                np.sum(switched & (calibrated_error > base_error))
            ),
        },
        "baseline_audit": baseline_audit,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "source_sha256": source_hashes,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
