"""Fit a grouped-OOF perspective-aware progress residual calibrator."""
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from experiments.progress_calibrator import (
    FEATURE_NAMES,
    PROGRESS_CALIBRATOR_PROTOCOL,
    apply_progress_correction,
    ensemble_prediction,
    extract_progress_features,
    feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


TRAINING_PROTOCOL = "syncg_grouped_oof_progress_calibrator_v1"
CLIP_CANDIDATES = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


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
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/progress_calibrator_syncg/model"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--max-features", type=float, default=0.80)
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


def _build_model(args: argparse.Namespace, *, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=args.trees,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=args.max_features,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _vector_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("vector")
    return value if isinstance(value, dict) else {}


def _vector_progress(row: dict[str, Any]) -> float | None:
    payload = _vector_payload(row)
    return finite_float(payload.get("progress")) if payload.get("status") is True else None


def _vector_prediction(row: dict[str, Any]) -> float | None:
    payload = _vector_payload(row)
    return finite_float(payload.get("prediction")) if payload.get("status") is True else None


def _target_progress(row: dict[str, Any]) -> float | None:
    ground_truth = finite_float(row.get("ground_truth"))
    start = finite_float(row.get("scale_start"))
    end = finite_float(row.get("scale_end"))
    if ground_truth is None or start is None or end is None or abs(end - start) <= 1e-12:
        return None
    return float((ground_truth - start) / (end - start))


def _metrics(errors: np.ndarray, successful: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": len(errors),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "acc_5pct": float(np.mean(errors <= 0.05)),
    }


def _paired_group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int = 5000,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected])
        deltas[iteration] = float(
            np.mean(candidate[selected_indices]) - np.mean(baseline[selected_indices])
        )
    return {
        "delta_nmae": float(np.mean(candidate) - np.mean(baseline)),
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "groups": int(len(unique)),
        "iterations": iterations,
    }


def _choose_clip(
    raw_progress: np.ndarray,
    target_progress: np.ndarray,
    residual: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    diagnostics: list[dict[str, float]] = []
    for candidate in CLIP_CANDIDATES:
        corrected = np.clip(
            raw_progress + np.clip(residual, -candidate, candidate), 0.0, 1.0
        )
        errors = np.minimum(np.abs(corrected - target_progress), 1.0)
        diagnostics.append(
            {
                "correction_clip": candidate,
                "nmae": float(np.mean(errors)),
                "acc_2pct": float(np.mean(errors <= 0.02)),
            }
        )
    selected = min(
        diagnostics,
        key=lambda item: (item["nmae"], -item["acc_2pct"], item["correction_clip"]),
    )
    return float(selected["correction_clip"]), diagnostics


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.folds < 3 or args.trees <= 0 or args.min_samples_leaf <= 0:
        raise ValueError("invalid progress-calibrator training parameters")
    for path in (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(_metadata_path(args.oof_pairs).read_text(encoding="utf-8"))
    input_summary = json.loads(_summary_path(args.oof_pairs).read_text(encoding="utf-8"))
    if (metadata.get("signature") or {}).get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("input has the wrong probabilistic OOF protocol")
    if (
        input_summary.get("status") != "complete"
        or input_summary.get("output_sha256") != sha256_file(args.oof_pairs)
        or int(input_summary.get("group_leakage_count", -1)) != 0
        or int(input_summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("input probabilistic OOF collection failed its train-only audit")

    rows = read_jsonl(args.oof_pairs)
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("progress calibrator accepts only SyncG/train rows")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("progress-calibrator input contains duplicate IDs")
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    raw_progress_all = np.asarray(
        [
            value if (value := _vector_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    target_progress_all = np.asarray(
        [
            value if (value := _target_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    successful = np.isfinite(raw_progress_all) & np.isfinite(target_progress_all)
    if int(np.sum(successful)) < 1000:
        raise ValueError("too few successful vector OOF rows")
    matrix_all = feature_matrix([extract_progress_features(row) for row in rows])
    matrix = matrix_all[successful]
    raw_progress = raw_progress_all[successful]
    target_progress = target_progress_all[successful]
    target_residual = np.clip(target_progress - raw_progress, -0.5, 0.5)
    groups = groups_all[successful]
    successful_indices = np.flatnonzero(successful)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    residual_oof = np.full(len(rows), math.nan, dtype=np.float64)
    ensemble_std_oof = np.full(len(rows), math.nan, dtype=np.float64)
    fold_ids = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target_residual, groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("progress-calibrator group leakage")
        model = _build_model(args, seed=args.seed + fold)
        model.fit(matrix[train_index], target_residual[train_index])
        residual, ensemble_std = ensemble_prediction(model, matrix[validation_index])
        destination = successful_indices[validation_index]
        residual_oof[destination] = residual
        ensemble_std_oof[destination] = ensemble_std
        fold_ids[destination] = fold
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
    if not np.isfinite(residual_oof[successful]).all() or np.any(fold_ids[successful] < 1):
        raise RuntimeError("progress-calibrator OOF predictions are incomplete")

    final_clip, clip_diagnostic = _choose_clip(
        raw_progress,
        target_progress,
        residual_oof[successful],
    )
    nested_clips: dict[str, float] = {}
    corrected_progress_all = np.full(len(rows), math.nan, dtype=np.float64)
    for fold in range(1, args.folds + 1):
        validation = successful & (fold_ids == fold)
        selection = successful & (fold_ids != fold)
        clip, _ = _choose_clip(
            raw_progress_all[selection],
            target_progress_all[selection],
            residual_oof[selection],
        )
        nested_clips[str(fold)] = clip
        corrected_progress_all[validation] = np.clip(
            raw_progress_all[validation]
            + np.clip(residual_oof[validation], -clip, clip),
            0.0,
            1.0,
        )

    raw_values = [_vector_prediction(row) for row in rows]
    corrected_values = [
        reading_from_progress(progress, row.get("scale_start"), row.get("scale_end"))
        if np.isfinite(progress)
        else None
        for row, progress in zip(rows, corrected_progress_all)
    ]
    raw_errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, raw_values)],
        dtype=np.float64,
    )
    corrected_errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, corrected_values)],
        dtype=np.float64,
    )
    corrected_success = np.asarray([value is not None for value in corrected_values])
    oracle_errors = np.minimum(raw_errors, corrected_errors)
    positive = successful & (corrected_errors < raw_errors)
    negative = successful & (corrected_errors > raw_errors)

    final_model = _build_model(args, seed=args.seed)
    final_model.fit(matrix, target_residual)
    model_path = args.output_dir / "progress_calibrator.joblib"
    diagnostics_path = args.output_dir / "nested_oof_predictions.jsonl"
    summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    source_hashes = {
        "features_and_policy": sha256_file(
            PROJECT_DIR / "experiments" / "progress_calibrator.py"
        ),
        "trainer": sha256_file(Path(__file__).resolve()),
    }
    artifact = {
        "protocol": PROGRESS_CALIBRATOR_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "feature_names": list(FEATURE_NAMES),
        "correction_clip": final_clip,
        "clip_candidates": list(CLIP_CANDIDATES),
        "estimator": final_model,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
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
                        "held_out_seed": row.get("held_out_seed"),
                        "router_fold": int(fold_ids[index]) if successful[index] else None,
                        "raw_progress": finite_float(raw_progress_all[index]),
                        "target_progress": finite_float(target_progress_all[index]),
                        "predicted_residual_oof": finite_float(residual_oof[index]),
                        "ensemble_std_oof": finite_float(ensemble_std_oof[index]),
                        "nested_correction_clip": (
                            nested_clips.get(str(fold_ids[index])) if successful[index] else None
                        ),
                        "corrected_progress_oof": finite_float(
                            corrected_progress_all[index]
                        ),
                        "raw_prediction": raw_values[index],
                        "corrected_prediction_oof": corrected_values[index],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    correlation = float(
        np.corrcoef(
            ensemble_std_oof[successful],
            np.abs(target_residual),
        )[0, 1]
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
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "successful_training_samples": int(np.sum(successful)),
        "groups": len(set(groups_all.tolist())),
        "folds": args.folds,
        "fold_summaries": fold_summaries,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": importance,
        "clip_selection": {
            "final": final_clip,
            "nested_leave_fold_out": nested_clips,
            "diagnostic": clip_diagnostic,
        },
        "metrics": {
            "raw_probabilistic_vector": _metrics(raw_errors, successful),
            "calibrated_vector_nested_oof": _metrics(
                corrected_errors, corrected_success
            ),
            "per_sample_oracle": _metrics(oracle_errors, successful),
        },
        "paired_vs_raw_vector": _paired_group_bootstrap(
            corrected_errors,
            raw_errors,
            groups_all,
            seed=args.seed,
        ),
        "transfers": {
            "positive": int(np.sum(positive)),
            "negative": int(np.sum(negative)),
            "ties": int(np.sum(successful & (corrected_errors == raw_errors))),
        },
        "uncertainty": {
            "ensemble_std_target_abs_residual_correlation": correlation,
            "mean_ensemble_std": float(np.mean(ensemble_std_oof[successful])),
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
