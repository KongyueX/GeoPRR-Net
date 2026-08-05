"""Fit a strictly nested, reference-conditioned safe progress calibrator."""

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
    extract_progress_features,
    feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.reference_conditioned_progress_calibrator import (
    DEFAULT_REFERENCE_BRANCH,
    REFERENCE_BRANCHES,
    REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
    apply_safe_residual_array,
    deterministic_ensemble_prediction,
    normalize_reference_branch,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_file,
    sha256_source_file,
)


TRAINING_PROTOCOL = "syncg_strict_nested_reference_safe_calibrator_v2"
CLIP_CANDIDATES = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEADBAND_CANDIDATES = (
    0.0,
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
)


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
        default=Path(
            "artifacts/runs/reference_conditioned_progress_calibrator_syncg/model"
        ),
    )
    parser.add_argument("--baseline-diagnostics", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--max-features", type=float, default=0.80)
    parser.add_argument("--min-branch-samples", type=int, default=100)
    parser.add_argument("--min-branch-groups", type=int, default=10)
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
        raise ValueError(
            "input probabilistic OOF collection failed its train-only audit"
        )
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
    return (
        finite_float(payload.get("progress")) if payload.get("status") is True else None
    )


def _vector_prediction(row: dict[str, Any]) -> float | None:
    payload = _vector_payload(row)
    return (
        finite_float(payload.get("prediction"))
        if payload.get("status") is True
        else None
    )


def _target_progress(row: dict[str, Any]) -> float | None:
    ground_truth = finite_float(row.get("ground_truth"))
    start = finite_float(row.get("scale_start"))
    end = finite_float(row.get("scale_end"))
    if (
        ground_truth is None
        or start is None
        or end is None
        or abs(end - start) <= 1e-12
    ):
        return None
    return float((ground_truth - start) / (end - start))


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


def _paired_group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
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


def _fit_bundle(
    args: argparse.Namespace,
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    branches: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    fallback = _build_model(args, seed=seed)
    fallback.fit(matrix, target)
    estimators: dict[str, Pipeline] = {}
    skipped: dict[str, dict[str, int]] = {}
    for offset, branch in enumerate(REFERENCE_BRANCHES, start=1):
        selected = branches == branch
        samples = int(np.sum(selected))
        branch_groups = len(set(groups[selected].tolist()))
        if samples < args.min_branch_samples or branch_groups < args.min_branch_groups:
            skipped[branch] = {"samples": samples, "groups": branch_groups}
            continue
        estimator = _build_model(args, seed=seed + offset)
        estimator.fit(matrix[selected], target[selected])
        estimators[branch] = estimator
    return {
        "fallback_estimator": fallback,
        "branch_estimators": estimators,
        "skipped_branches": skipped,
    }


def _predict_bundle(
    bundle: dict[str, Any],
    matrix: np.ndarray,
    branches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.full(len(matrix), math.nan, dtype=np.float64)
    std = np.full(len(matrix), math.nan, dtype=np.float64)
    branch_model = np.zeros(len(matrix), dtype=bool)
    estimators = bundle["branch_estimators"]
    for branch in sorted(set(branches.tolist())):
        selected = branches == branch
        estimator = estimators.get(branch)
        if estimator is None:
            estimator = bundle["fallback_estimator"]
        else:
            branch_model[selected] = True
        predicted, disagreement = deterministic_ensemble_prediction(
            estimator,
            matrix[selected],
        )
        mean[selected] = predicted
        std[selected] = disagreement
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("reference-conditioned residual prediction is incomplete")
    fallback_mean, _ = deterministic_ensemble_prediction(
        bundle["fallback_estimator"],
        matrix,
    )
    if not np.isfinite(fallback_mean).all():
        raise RuntimeError("fallback residual prediction is incomplete")
    return mean, std, branch_model, fallback_mean


def _choose_policy(
    raw_progress: np.ndarray,
    target_progress: np.ndarray,
    model_residual: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    diagnostics: list[dict[str, float]] = []
    for correction_clip in CLIP_CANDIDATES:
        for deadband in DEADBAND_CANDIDATES:
            applied = apply_safe_residual_array(
                model_residual,
                correction_clip=correction_clip,
                deadband=deadband,
            )
            corrected = np.clip(raw_progress + applied, 0.0, 1.0)
            errors = np.abs(corrected - target_progress)
            diagnostics.append(
                {
                    "correction_clip": correction_clip,
                    "deadband": deadband,
                    "nmae": float(np.mean(errors)),
                    "acc_1pct": float(np.mean(errors <= 0.01)),
                    "acc_2pct": float(np.mean(errors <= 0.02)),
                    "acc_5pct": float(np.mean(errors <= 0.05)),
                    "apply_rate": float(np.mean(applied != 0.0)),
                }
            )
    selected = min(
        diagnostics,
        key=lambda item: (
            item["nmae"],
            -item["acc_2pct"],
            -item["acc_1pct"],
            item["correction_clip"],
            item["deadband"],
        ),
    )
    policy = {
        "correction_clip": float(selected["correction_clip"]),
        "deadband": float(selected["deadband"]),
    }
    return policy, diagnostics


def _choose_branch_policies(
    raw_progress: np.ndarray,
    target_progress: np.ndarray,
    model_residual: np.ndarray,
    branches: np.ndarray,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float],
    dict[str, list[dict[str, float]]],
]:
    fallback, fallback_diagnostic = _choose_policy(
        raw_progress,
        target_progress,
        model_residual,
    )
    policies: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, list[dict[str, float]]] = {"fallback": fallback_diagnostic}
    for branch in REFERENCE_BRANCHES:
        selected = branches == branch
        if not np.any(selected):
            policies[branch] = dict(fallback)
            diagnostics[branch] = []
            continue
        policy, diagnostic = _choose_policy(
            raw_progress[selected],
            target_progress[selected],
            model_residual[selected],
        )
        policies[branch] = policy
        diagnostics[branch] = diagnostic
    return policies, fallback, diagnostics


def _apply_branch_policies(
    raw_progress: np.ndarray,
    model_residual: np.ndarray,
    branches: np.ndarray,
    policies: dict[str, dict[str, float]],
    fallback: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    applied = np.full(len(raw_progress), math.nan, dtype=np.float64)
    clips = np.full(len(raw_progress), math.nan, dtype=np.float64)
    deadbands = np.full(len(raw_progress), math.nan, dtype=np.float64)
    for branch in sorted(set(branches.tolist())):
        selected = branches == branch
        policy = policies.get(branch, fallback)
        clips[selected] = float(policy["correction_clip"])
        deadbands[selected] = float(policy["deadband"])
        applied[selected] = apply_safe_residual_array(
            model_residual[selected],
            correction_clip=float(policy["correction_clip"]),
            deadband=float(policy["deadband"]),
        )
    corrected = np.clip(raw_progress + applied, 0.0, 1.0)
    return corrected, applied, clips, deadbands


def _cross_fitted_residual(
    args: argparse.Namespace,
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    branches: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    unique_groups = len(set(groups.tolist()))
    if unique_groups < folds:
        raise ValueError(f"cannot make {folds} folds from only {unique_groups} groups")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    residual = np.full(len(matrix), math.nan, dtype=np.float64)
    fallback_residual = np.full(len(matrix), math.nan, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, groups),
        start=1,
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError("inner progress-calibrator group leakage")
        bundle = _fit_bundle(
            args,
            matrix[train_index],
            target[train_index],
            groups[train_index],
            branches[train_index],
            seed=seed + fold * 100,
        )
        predicted, _, _, fallback_predicted = _predict_bundle(
            bundle,
            matrix[validation_index],
            branches[validation_index],
        )
        residual[validation_index] = predicted
        fallback_residual[validation_index] = fallback_predicted
        summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
                "skipped_branches": bundle["skipped_branches"],
            }
        )
    if not np.isfinite(residual).all() or not np.isfinite(fallback_residual).all():
        raise RuntimeError("inner cross-fitted residuals are incomplete")
    return residual, fallback_residual, summaries


def _feature_importance(estimator: Pipeline) -> list[dict[str, float | str]]:
    regressor = estimator.named_steps["regressor"]
    names = estimator.named_steps["imputer"].get_feature_names_out(FEATURE_NAMES)
    return [
        {"feature": str(name), "importance": float(value)}
        for name, value in sorted(
            zip(names, regressor.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _load_baseline(
    diagnostics_path: Path,
    summary_path: Path,
    *,
    input_path: Path,
    identifiers: list[str],
) -> tuple[list[float | None], dict[str, Any]]:
    from experiments.train_progress_calibrator import (
        TRAINING_PROTOCOL as BASELINE_TRAINING_PROTOCOL,
    )

    if not diagnostics_path.is_file():
        raise FileNotFoundError(diagnostics_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("protocol") != BASELINE_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("baseline calibration failed its train-only audit")
    rows = read_jsonl(diagnostics_path)
    by_id = {str(row.get("sample_id")): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(identifiers):
        raise ValueError("baseline calibration IDs differ from OOF input")
    values = [
        finite_float(by_id[sample_id].get("corrected_prediction_oof"))
        for sample_id in identifiers
    ]
    audit = {
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "protocol": summary.get("protocol"),
    }
    return values, audit


def main() -> None:
    args = parse_args()
    for name in ("oof_pairs", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for name in ("baseline_diagnostics", "baseline_summary"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (args.baseline_diagnostics is None) != (args.baseline_summary is None):
        raise ValueError(
            "--baseline-diagnostics and --baseline-summary must be supplied together"
        )
    if (
        args.folds < 3
        or args.inner_folds < 3
        or args.trees <= 0
        or args.min_samples_leaf <= 0
        or args.min_branch_samples <= 0
        or args.min_branch_groups < 3
        or args.bootstrap_iterations <= 0
        or not isinstance(args.expected_oof_protocol, str)
        or not args.expected_oof_protocol.strip()
    ):
        raise ValueError("invalid reference-conditioned training parameters")
    required = (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_input_oof_contract(
        args.oof_pairs,
        expected_protocol=args.expected_oof_protocol,
    )

    rows = read_jsonl(args.oof_pairs)
    if any(
        row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows
    ):
        raise ValueError("reference-conditioned calibration accepts only SyncG/train")
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("reference-conditioned input contains duplicate IDs")
    groups_all = np.asarray(
        [str(row.get("group_id")) for row in rows],
        dtype=object,
    )
    branches_all = np.asarray(
        [normalize_reference_branch(row) for row in rows],
        dtype=object,
    )
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
    branches = branches_all[successful]
    successful_indices = np.flatnonzero(successful)

    splitter = GroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    model_residual_oof = np.full(len(rows), math.nan, dtype=np.float64)
    fallback_model_residual_oof = np.full(
        len(rows),
        math.nan,
        dtype=np.float64,
    )
    applied_residual_oof = np.full(len(rows), math.nan, dtype=np.float64)
    corrected_progress_oof = np.full(len(rows), math.nan, dtype=np.float64)
    global_progress_oof = np.full(len(rows), math.nan, dtype=np.float64)
    branch_no_deadband_progress_oof = np.full(
        len(rows),
        math.nan,
        dtype=np.float64,
    )
    shared_deadband_progress_oof = np.full(
        len(rows),
        math.nan,
        dtype=np.float64,
    )
    ensemble_std_oof = np.full(len(rows), math.nan, dtype=np.float64)
    correction_clip_oof = np.full(len(rows), math.nan, dtype=np.float64)
    deadband_oof = np.full(len(rows), math.nan, dtype=np.float64)
    branch_model_oof = np.zeros(len(rows), dtype=bool)
    fold_ids = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    nested_policies: dict[str, Any] = {}
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target_residual, groups),
        start=1,
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("outer progress-calibrator group leakage")
        inner_residual, inner_fallback_residual, inner_summaries = (
            _cross_fitted_residual(
                args,
                matrix[train_index],
                target_residual[train_index],
                groups[train_index],
                branches[train_index],
                folds=args.inner_folds,
                seed=args.seed + fold * 10_000,
            )
        )
        branch_policies, shared_branch_policy, _ = _choose_branch_policies(
            raw_progress[train_index],
            target_progress[train_index],
            inner_residual,
            branches[train_index],
        )
        global_fallback_policy, _ = _choose_policy(
            raw_progress[train_index],
            target_progress[train_index],
            inner_fallback_residual,
        )
        nested_policies[str(fold)] = {
            "branch_policies": branch_policies,
            "shared_branch_policy": shared_branch_policy,
            "global_fallback_policy": global_fallback_policy,
        }
        bundle = _fit_bundle(
            args,
            matrix[train_index],
            target_residual[train_index],
            groups[train_index],
            branches[train_index],
            seed=args.seed + fold * 1_000,
        )
        (
            model_residual,
            ensemble_std,
            used_branch_model,
            fallback_model_residual,
        ) = _predict_bundle(
            bundle,
            matrix[validation_index],
            branches[validation_index],
        )
        corrected, applied, clips, deadbands = _apply_branch_policies(
            raw_progress[validation_index],
            model_residual,
            branches[validation_index],
            branch_policies,
            global_fallback_policy,
        )
        global_corrected, _, _, _ = _apply_branch_policies(
            raw_progress[validation_index],
            fallback_model_residual,
            branches[validation_index],
            {},
            global_fallback_policy,
        )
        no_deadband_policies = {
            branch: {
                "correction_clip": policy["correction_clip"],
                "deadband": 0.0,
            }
            for branch, policy in branch_policies.items()
        }
        no_deadband_corrected, _, _, _ = _apply_branch_policies(
            raw_progress[validation_index],
            model_residual,
            branches[validation_index],
            no_deadband_policies,
            {
                "correction_clip": shared_branch_policy["correction_clip"],
                "deadband": 0.0,
            },
        )
        shared_deadband_corrected, _, _, _ = _apply_branch_policies(
            raw_progress[validation_index],
            model_residual,
            branches[validation_index],
            {},
            shared_branch_policy,
        )
        destination = successful_indices[validation_index]
        model_residual_oof[destination] = model_residual
        fallback_model_residual_oof[destination] = fallback_model_residual
        applied_residual_oof[destination] = applied
        corrected_progress_oof[destination] = corrected
        global_progress_oof[destination] = global_corrected
        branch_no_deadband_progress_oof[destination] = no_deadband_corrected
        shared_deadband_progress_oof[destination] = shared_deadband_corrected
        ensemble_std_oof[destination] = ensemble_std
        correction_clip_oof[destination] = clips
        deadband_oof[destination] = deadbands
        branch_model_oof[destination] = used_branch_model
        fold_ids[destination] = fold
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
                "inner_folds": inner_summaries,
                "skipped_branches": bundle["skipped_branches"],
            }
        )
    if (
        not np.isfinite(model_residual_oof[successful]).all()
        or not np.isfinite(fallback_model_residual_oof[successful]).all()
        or not np.isfinite(applied_residual_oof[successful]).all()
        or not np.isfinite(corrected_progress_oof[successful]).all()
        or not np.isfinite(global_progress_oof[successful]).all()
        or not np.isfinite(branch_no_deadband_progress_oof[successful]).all()
        or not np.isfinite(shared_deadband_progress_oof[successful]).all()
        or np.any(fold_ids[successful] < 1)
    ):
        raise RuntimeError("strictly nested calibration predictions are incomplete")

    final_branch_policies, final_shared_policy, policy_diagnostics = (
        _choose_branch_policies(
            raw_progress,
            target_progress,
            model_residual_oof[successful],
            branches,
        )
    )
    final_fallback_policy, fallback_policy_diagnostic = _choose_policy(
        raw_progress,
        target_progress,
        fallback_model_residual_oof[successful],
    )
    final_bundle = _fit_bundle(
        args,
        matrix,
        target_residual,
        groups,
        branches,
        seed=args.seed,
    )
    raw_values = [_vector_prediction(row) for row in rows]
    corrected_values = [
        reading_from_progress(progress, row.get("scale_start"), row.get("scale_end"))
        if np.isfinite(progress)
        else None
        for row, progress in zip(rows, corrected_progress_oof)
    ]
    global_values = [
        reading_from_progress(progress, row.get("scale_start"), row.get("scale_end"))
        if np.isfinite(progress)
        else None
        for row, progress in zip(rows, global_progress_oof)
    ]
    branch_no_deadband_values = [
        reading_from_progress(progress, row.get("scale_start"), row.get("scale_end"))
        if np.isfinite(progress)
        else None
        for row, progress in zip(rows, branch_no_deadband_progress_oof)
    ]
    shared_deadband_values = [
        reading_from_progress(progress, row.get("scale_start"), row.get("scale_end"))
        if np.isfinite(progress)
        else None
        for row, progress in zip(rows, shared_deadband_progress_oof)
    ]
    raw_errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, raw_values)],
        dtype=np.float64,
    )
    corrected_errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, corrected_values)],
        dtype=np.float64,
    )
    global_errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, global_values)],
        dtype=np.float64,
    )
    branch_no_deadband_errors = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, branch_no_deadband_values)
        ],
        dtype=np.float64,
    )
    shared_deadband_errors = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, shared_deadband_values)
        ],
        dtype=np.float64,
    )
    corrected_success = np.asarray([value is not None for value in corrected_values])
    global_success = np.asarray([value is not None for value in global_values])
    branch_no_deadband_success = np.asarray(
        [value is not None for value in branch_no_deadband_values]
    )
    shared_deadband_success = np.asarray(
        [value is not None for value in shared_deadband_values]
    )
    oracle_errors = np.minimum(raw_errors, corrected_errors)
    positive = successful & (corrected_errors < raw_errors)
    negative = successful & (corrected_errors > raw_errors)

    model_path = args.output_dir / "reference_conditioned_calibrator.joblib"
    diagnostics_path = args.output_dir / "strict_nested_oof_predictions.jsonl"
    summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    source_hashes = {
        "features": sha256_source_file(
            PROJECT_DIR / "experiments" / "progress_calibrator.py"
        ),
        "reference_policy": sha256_source_file(
            PROJECT_DIR / "experiments" / "reference_conditioned_progress_calibrator.py"
        ),
        "trainer": sha256_source_file(Path(__file__).resolve()),
    }
    training_parameters = {
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "trees": args.trees,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "min_branch_samples": args.min_branch_samples,
        "min_branch_groups": args.min_branch_groups,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    artifact = {
        "protocol": REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "strict_nested_oof": True,
        "feature_names": list(FEATURE_NAMES),
        "reference_branches": list(REFERENCE_BRANCHES),
        "default_reference_branch": DEFAULT_REFERENCE_BRANCH,
        "branch_policies": final_branch_policies,
        "fallback_policy": final_fallback_policy,
        "shared_branch_policy": final_shared_policy,
        "clip_candidates": list(CLIP_CANDIDATES),
        "deadband_candidates": list(DEADBAND_CANDIDATES),
        "fallback_estimator": final_bundle["fallback_estimator"],
        "branch_estimators": final_bundle["branch_estimators"],
        "input_oof_protocol": args.expected_oof_protocol,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
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
                        "held_out_seed": row.get("held_out_seed"),
                        "calibrator_fold": (
                            int(fold_ids[index]) if successful[index] else None
                        ),
                        "reference_branch": branches_all[index],
                        "used_branch_model_oof": (
                            bool(branch_model_oof[index]) if successful[index] else None
                        ),
                        "raw_progress": finite_float(raw_progress_all[index]),
                        "target_progress": finite_float(target_progress_all[index]),
                        "model_residual_oof": finite_float(model_residual_oof[index]),
                        "fallback_model_residual_oof": finite_float(
                            fallback_model_residual_oof[index]
                        ),
                        "predicted_residual_oof": finite_float(
                            model_residual_oof[index]
                        ),
                        "applied_residual_oof": finite_float(
                            applied_residual_oof[index]
                        ),
                        "ensemble_std_oof": finite_float(ensemble_std_oof[index]),
                        "nested_correction_clip": finite_float(
                            correction_clip_oof[index]
                        ),
                        "nested_deadband": finite_float(deadband_oof[index]),
                        "calibration_applied_oof": (
                            bool(applied_residual_oof[index] != 0.0)
                            if successful[index]
                            else None
                        ),
                        "corrected_progress_oof": finite_float(
                            corrected_progress_oof[index]
                        ),
                        "global_progress_oof": finite_float(global_progress_oof[index]),
                        "branch_no_deadband_progress_oof": finite_float(
                            branch_no_deadband_progress_oof[index]
                        ),
                        "shared_deadband_progress_oof": finite_float(
                            shared_deadband_progress_oof[index]
                        ),
                        "raw_prediction": raw_values[index],
                        "corrected_prediction_oof": corrected_values[index],
                        "global_prediction_oof": global_values[index],
                        "branch_no_deadband_prediction_oof": (
                            branch_no_deadband_values[index]
                        ),
                        "shared_deadband_prediction_oof": (
                            shared_deadband_values[index]
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    per_branch: dict[str, Any] = {}
    for branch in REFERENCE_BRANCHES:
        selected = branches_all == branch
        branch_success = successful & selected
        per_branch[branch] = {
            "samples": int(np.sum(selected)),
            "successful": int(np.sum(branch_success)),
            "groups": len(set(groups_all[selected].tolist())),
            "raw": _metrics(raw_errors[selected], successful[selected]),
            "calibrated": _metrics(
                corrected_errors[selected],
                corrected_success[selected],
            ),
            "apply_rate_on_success": float(
                np.mean(applied_residual_oof[branch_success] != 0.0)
            ),
            "positive_transfers": int(np.sum(positive & selected)),
            "negative_transfers": int(np.sum(negative & selected)),
        }

    comparisons: dict[str, Any] = {
        "calibrated_vs_raw_vector": _paired_group_bootstrap(
            corrected_errors,
            raw_errors,
            groups_all,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ),
        "full_vs_strict_global_residual": _paired_group_bootstrap(
            corrected_errors,
            global_errors,
            groups_all,
            seed=args.seed + 10,
            iterations=args.bootstrap_iterations,
        ),
        "full_vs_branch_without_deadband": _paired_group_bootstrap(
            corrected_errors,
            branch_no_deadband_errors,
            groups_all,
            seed=args.seed + 11,
            iterations=args.bootstrap_iterations,
        ),
        "full_vs_branch_shared_deadband": _paired_group_bootstrap(
            corrected_errors,
            shared_deadband_errors,
            groups_all,
            seed=args.seed + 12,
            iterations=args.bootstrap_iterations,
        ),
    }
    baseline_audit = None
    baseline_metrics = None
    if args.baseline_diagnostics is not None and args.baseline_summary is not None:
        baseline_values, baseline_audit = _load_baseline(
            args.baseline_diagnostics,
            args.baseline_summary,
            input_path=args.oof_pairs,
            identifiers=identifiers,
        )
        baseline_errors = np.asarray(
            [normalized_error(row, value) for row, value in zip(rows, baseline_values)],
            dtype=np.float64,
        )
        baseline_success = np.asarray([value is not None for value in baseline_values])
        baseline_metrics = _metrics(baseline_errors, baseline_success)
        comparisons["reference_conditioned_vs_global_calibrator"] = (
            _paired_group_bootstrap(
                corrected_errors,
                baseline_errors,
                groups_all,
                seed=args.seed + 1,
                iterations=args.bootstrap_iterations,
            )
        )

    valid_uncertainty = successful & np.isfinite(ensemble_std_oof)
    uncertainty_correlation = float(
        np.corrcoef(
            ensemble_std_oof[valid_uncertainty],
            np.abs(
                target_progress_all[valid_uncertainty]
                - raw_progress_all[valid_uncertainty]
            ),
        )[0, 1]
    )
    importance = {
        "fallback": _feature_importance(final_bundle["fallback_estimator"]),
        "branches": {
            branch: _feature_importance(estimator)
            for branch, estimator in final_bundle["branch_estimators"].items()
        },
    }
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "seed": args.seed,
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "input_oof_protocol": args.expected_oof_protocol,
        "samples": len(rows),
        "successful_training_samples": int(np.sum(successful)),
        "groups": len(set(groups_all.tolist())),
        "reference_branch_counts": dict(sorted(Counter(branches_all).items())),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "training_parameters": training_parameters,
        "strict_nested_oof": True,
        "fold_summaries": fold_summaries,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": importance,
        "policies": {
            "final_branch_policies": final_branch_policies,
            "final_fallback_policy": final_fallback_policy,
            "final_shared_branch_policy": final_shared_policy,
            "nested_outer_fold_policies": nested_policies,
            "branch_selection_diagnostic": policy_diagnostics,
            "fallback_selection_diagnostic": fallback_policy_diagnostic,
        },
        "metrics": {
            "raw_probabilistic_vector": _metrics(raw_errors, successful),
            "global_progress_calibrator_v1": baseline_metrics,
            "strict_global_residual_nested_oof": _metrics(
                global_errors,
                global_success,
            ),
            "branch_residual_without_deadband_nested_oof": _metrics(
                branch_no_deadband_errors,
                branch_no_deadband_success,
            ),
            "branch_residual_shared_deadband_nested_oof": _metrics(
                shared_deadband_errors,
                shared_deadband_success,
            ),
            "reference_conditioned_nested_oof": _metrics(
                corrected_errors,
                corrected_success,
            ),
            "per_sample_oracle_raw_vs_reference_conditioned": _metrics(
                oracle_errors,
                successful,
            ),
        },
        "per_reference_branch": per_branch,
        "paired_comparisons": comparisons,
        "transfers": {
            "positive": int(np.sum(positive)),
            "negative": int(np.sum(negative)),
            "ties": int(np.sum(successful & (corrected_errors == raw_errors))),
            "calibration_applied": int(
                np.sum(successful & (applied_residual_oof != 0.0))
            ),
            "calibration_abstained": int(
                np.sum(successful & (applied_residual_oof == 0.0))
            ),
        },
        "uncertainty": {
            "ensemble_std_target_abs_residual_correlation": uncertainty_correlation,
            "mean_ensemble_std": float(np.mean(ensemble_std_oof[valid_uncertainty])),
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
