"""Train a grouped-OOF router between mask and calibrated-vector readings."""
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
    CALIBRATED_PROGRESS_ROUTER_PROTOCOL,
    FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix,
)
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.train_progress_calibrator import TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL
from experiments.train_quality_router import (
    _build_model,
    _choose_threshold,
    _metrics,
    _paired_group_bootstrap,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


TRAINING_PROTOCOL = "syncg_grouped_oof_calibrated_progress_router_v1"


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
            "artifacts/runs/progress_calibrator_syncg/model/nested_oof_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=Path(
            "artifacts/runs/progress_calibrator_syncg/model/training_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/calibrated_progress_router_syncg/model"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


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


def main() -> None:
    args = parse_args()
    for name in (
        "oof_pairs",
        "calibration_diagnostics",
        "calibration_summary",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.folds < 3 or args.trees <= 0 or args.min_samples_leaf <= 0:
        raise ValueError("invalid calibrated-router training parameters")
    required = (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
        args.calibration_diagnostics,
        args.calibration_summary,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(_metadata_path(args.oof_pairs).read_text(encoding="utf-8"))
    input_summary = json.loads(_summary_path(args.oof_pairs).read_text(encoding="utf-8"))
    calibration_summary = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    if (metadata.get("signature") or {}).get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("input has the wrong probabilistic OOF protocol")
    if (
        input_summary.get("status") != "complete"
        or input_summary.get("output_sha256") != sha256_file(args.oof_pairs)
        or int(input_summary.get("group_leakage_count", -1)) != 0
        or int(input_summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("probabilistic OOF collection failed its train-only audit")
    if (
        calibration_summary.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or calibration_summary.get("status") != "complete"
        or calibration_summary.get("input_sha256") != sha256_file(args.oof_pairs)
        or calibration_summary.get("diagnostics_sha256")
        != sha256_file(args.calibration_diagnostics)
        or int(calibration_summary.get("group_leakage_count", -1)) != 0
        or int(calibration_summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("progress calibration failed its train-only audit")

    rows = read_jsonl(args.oof_pairs)
    calibration_rows = read_jsonl(args.calibration_diagnostics)
    calibration_by_id = {
        str(row.get("sample_id")): row for row in calibration_rows
    }
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(calibration_by_id):
        raise ValueError("calibration diagnostics IDs differ from OOF input")
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("calibrated router accepts only SyncG/train rows")

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
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
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
    target = np.clip(base_error[joint] - calibrated_error[joint], -1.0, 1.0)
    joint_indices = np.flatnonzero(joint)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    score = np.full(len(rows), math.nan, dtype=np.float64)
    fold_id = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("calibrated-router group leakage")
        model = _build_model(args, seed=args.seed + fold)
        model.fit(matrix[train_index], target[train_index])
        destination = joint_indices[validation_index]
        score[destination] = model.predict(matrix[validation_index])
        fold_id[destination] = fold
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(score[joint]).all() or np.any(fold_id[joint] < 1):
        raise RuntimeError("calibrated-router OOF scores are incomplete")

    final_threshold, threshold_diagnostic = _choose_threshold(
        score[joint], base_error[joint], calibrated_error[joint]
    )
    nested_thresholds: dict[str, float] = {}
    use_calibrated = np.zeros(len(rows), dtype=bool)
    for fold in range(1, args.folds + 1):
        validation = joint & (fold_id == fold)
        threshold_train = joint & (fold_id != fold)
        threshold, _ = _choose_threshold(
            score[threshold_train],
            base_error[threshold_train],
            calibrated_error[threshold_train],
        )
        nested_thresholds[str(fold)] = float(threshold)
        use_calibrated[validation] = score[validation] > threshold
    use_calibrated[~base_success & calibrated_success] = True

    routed: list[float | None] = []
    routes: list[str] = []
    for base_value, calibrated_value, switch in zip(base, calibrated, use_calibrated):
        if base_value is None:
            routed.append(calibrated_value)
            routes.append("calibrated_hard_fallback" if calibrated_value is not None else "failure")
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
    hard = [first if first is not None else second for first, second in zip(base, calibrated)]
    hard_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, hard)],
        dtype=np.float64,
    )
    hard_success = base_success | calibrated_success
    oracle_error = np.minimum(base_error, calibrated_error)

    final_model = _build_model(args, seed=args.seed)
    final_model.fit(matrix, target)
    model_path = args.output_dir / "calibrated_progress_router.joblib"
    diagnostics_path = args.output_dir / "nested_oof_routing.jsonl"
    summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    source_hashes = {
        "features": sha256_file(
            PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
        ),
        "quality_features": sha256_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ),
        "policy": sha256_file(PROJECT_DIR / "experiments" / "quality_router.py"),
        "trainer": sha256_file(Path(__file__).resolve()),
        "shared_training_helpers": sha256_file(
            PROJECT_DIR / "experiments" / "train_quality_router.py"
        ),
    }
    artifact = {
        "protocol": CALIBRATED_PROGRESS_ROUTER_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "feature_names": list(FEATURE_NAMES),
        "threshold": float(final_threshold),
        "estimator": final_model,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
        "calibration_diagnostics_sha256": sha256_file(args.calibration_diagnostics),
        "failure_policy": "base failure -> calibrated vector; joint failure -> failure",
        "test_sets_used": [],
        "seed": args.seed,
        "source_sha256": source_hashes,
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
                        "router_fold": int(fold_id[index]) if joint[index] else None,
                        "router_score_oof": finite_float(score[index]),
                        "nested_threshold": (
                            nested_thresholds.get(str(fold_id[index])) if joint[index] else None
                        ),
                        "route": routes[index],
                        "prediction": routed[index],
                        "base_prediction": base[index],
                        "calibrated_prediction": calibrated[index],
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
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "calibration_diagnostics": str(args.calibration_diagnostics),
        "calibration_diagnostics_sha256": sha256_file(args.calibration_diagnostics),
        "samples": len(rows),
        "joint_training_samples": int(np.sum(joint)),
        "groups": len(set(groups_all.tolist())),
        "folds": args.folds,
        "fold_summaries": fold_summaries,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": importance,
        "threshold": {
            "final": float(final_threshold),
            "selection_diagnostic": threshold_diagnostic,
            "nested_leave_fold_out": nested_thresholds,
        },
        "metrics": {
            "base_mask": _metrics(base_error, base_success),
            "calibrated_vector": _metrics(calibrated_error, calibrated_success),
            "hard_fallback": _metrics(hard_error, hard_success),
            "calibrated_router_nested_oof": _metrics(routed_error, routed_success),
            "oracle": _metrics(oracle_error, hard_success),
        },
        "paired_vs_hard_fallback": _paired_group_bootstrap(
            routed_error, hard_error, groups_all, seed=args.seed
        ),
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
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "source_sha256": source_hashes,
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
