"""Train leakage-free joint outer-group FADR stacking evidence.

For every outer validation group, this runner rebuilds the calibrator,
calibrator policy, router features/targets, router model, and router threshold
from the complementary outer-training groups only.  Standalone calibrator OOF
artifacts are deliberately not accepted as inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablate_fadr_router_features import (
    _grouped_splits,
    _metrics,
    _route,
    _validate_preflight,
)
from experiments.calibrated_progress_router import (
    FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix as router_feature_matrix,
)
from experiments.fadr_feature_sets import FADR_ROUTER_FEATURE_SETS
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    FADR_JOINT_LINEAGE_PROTOCOL,
    FADR_JOINT_TRAINING_PROTOCOL,
    FADR_SEEDS,
    assert_train_only_path,
    sha256_file,
    sha256_strings,
    strict_jsonl_load,
)
from experiments.progress_calibrator import (
    extract_progress_features,
    feature_matrix as calibrator_feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float, normalized_error
from experiments.reference_conditioned_progress_calibrator import (
    normalize_reference_branch,
)
from experiments.reference_conditioned_router import (
    deterministic_router_prediction,
)
from experiments.strict_json import (
    STRICT_JSON_PROTOCOL,
    strict_json_source_sha256,
)
from experiments.train_quality_router import (
    _choose_threshold,
    _paired_group_bootstrap,
)
from experiments.train_reference_conditioned_progress_calibrator import (
    _apply_branch_policies,
    _choose_branch_policies,
    _choose_policy,
    _fit_bundle,
    _predict_bundle,
    _target_progress,
    _vector_prediction,
    _vector_progress,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_source_file,
)


DIAGNOSTICS_FILENAME = "joint_oof_predictions.jsonl"
LINEAGE_FILENAME = "joint_lineage.json"
SUMMARY_FILENAME = "joint_summary.json"
MIN_CALIBRATOR_SAMPLES = 1000
MIN_ROUTER_JOINT_SAMPLES = 1000


def joint_source_identity() -> dict[str, str]:
    return {
        "joint_trainer": sha256_source_file(Path(__file__).resolve()),
        "protocol": sha256_source_file(
            PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
        ),
        "feature_sets": sha256_source_file(
            PROJECT_DIR / "experiments" / "fadr_feature_sets.py"
        ),
        "calibrator_features": sha256_source_file(
            PROJECT_DIR / "experiments" / "progress_calibrator.py"
        ),
        "calibrator_policy": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "reference_conditioned_progress_calibrator.py"
        ),
        "calibrator_training_helpers": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "train_reference_conditioned_progress_calibrator.py"
        ),
        "router_features": sha256_source_file(
            PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
        ),
        "router_policy": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_quality_router.py"
        ),
        "shared_joint_helpers": sha256_source_file(
            PROJECT_DIR / "experiments" / "ablate_fadr_router_features.py"
        ),
        "strict_json": strict_json_source_sha256(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-pairs", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--calibrator-folds", type=int, default=5)
    parser.add_argument("--router-inner-folds", type=int, default=5)
    parser.add_argument("--calibrator-trees", type=int, default=600)
    parser.add_argument("--calibrator-max-depth", type=int, default=8)
    parser.add_argument("--calibrator-min-samples-leaf", type=int, default=12)
    parser.add_argument("--calibrator-max-features", type=float, default=0.80)
    parser.add_argument("--min-branch-samples", type=int, default=100)
    parser.add_argument("--min-branch-groups", type=int, default=10)
    parser.add_argument("--router-trees", type=int, default=600)
    parser.add_argument("--router-max-depth", type=int, default=12)
    parser.add_argument("--router-min-samples-leaf", type=int, default=8)
    parser.add_argument("--router-max-features", type=float, default=0.70)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, choices=FADR_SEEDS, required=True)
    return parser.parse_args()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identity(
    indices: Sequence[int] | np.ndarray,
    identifiers: Sequence[str],
    groups: np.ndarray,
) -> dict[str, Any]:
    numeric = np.asarray(indices, dtype=np.int64)
    sample_ids = sorted(identifiers[index] for index in numeric.tolist())
    group_ids = sorted(set(groups[numeric].tolist()))
    return {
        "samples": len(sample_ids),
        "groups": len(group_ids),
        "sample_ids_sha256": sha256_strings(sample_ids),
        "group_ids_sha256": sha256_strings(group_ids),
    }


def _build_router_model(args: argparse.Namespace, *, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=args.router_trees,
                    max_depth=args.router_max_depth,
                    min_samples_leaf=args.router_min_samples_leaf,
                    max_features=args.router_max_features,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _calibrator_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        trees=args.calibrator_trees,
        max_depth=args.calibrator_max_depth,
        min_samples_leaf=args.calibrator_min_samples_leaf,
        max_features=args.calibrator_max_features,
        min_branch_samples=args.min_branch_samples,
        min_branch_groups=args.min_branch_groups,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.outer_folds < 3
        or args.calibrator_folds < 3
        or args.router_inner_folds < 3
        or args.calibrator_trees <= 0
        or args.router_trees <= 0
        or args.calibrator_min_samples_leaf <= 0
        or args.router_min_samples_leaf <= 0
        or args.min_branch_samples <= 0
        or args.min_branch_groups < 3
        or args.bootstrap_iterations <= 0
        or not 0.0 < args.calibrator_max_features <= 1.0
        or not 0.0 < args.router_max_features <= 1.0
    ):
        raise ValueError("invalid joint FADR training parameters")


def _calibrator_for_outer_fold(
    *,
    args: argparse.Namespace,
    outer_fold: int,
    outer_train: np.ndarray,
    outer_validation: np.ndarray,
    identifiers: Sequence[str],
    groups: np.ndarray,
    branches: np.ndarray,
    progress_matrix: np.ndarray,
    raw_progress: np.ndarray,
    target_progress: np.ndarray,
    vector_available: np.ndarray,
    calibrator_trainable: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    train_global = outer_train[calibrator_trainable[outer_train]]
    validation_global = outer_validation[vector_available[outer_validation]]
    if len(train_global) < MIN_CALIBRATOR_SAMPLES:
        raise ValueError(
            f"outer fold {outer_fold}: too few calibrator-training samples"
        )
    train_groups = groups[train_global]
    target_residual = np.clip(
        target_progress[train_global] - raw_progress[train_global],
        -0.5,
        0.5,
    )
    split_seed = args.seed + outer_fold * 100_000 + 10_000
    splits = _grouped_splits(
        train_groups,
        target_residual,
        folds=args.calibrator_folds,
        seed=split_seed,
        label=f"joint outer {outer_fold} calibrator cross-fit",
    )
    cal_args = _calibrator_args(args)
    residual = np.full(len(train_global), math.nan, dtype=np.float64)
    fallback_residual = np.full(len(train_global), math.nan, dtype=np.float64)
    ensemble_std = np.full(len(train_global), math.nan, dtype=np.float64)
    used_branch = np.zeros(len(train_global), dtype=bool)
    calibrator_folds: list[dict[str, Any]] = []
    for calibrator_fold, (fit_local, validation_local) in enumerate(
        splits,
        start=1,
    ):
        fit_global = train_global[fit_local]
        held_global = train_global[validation_local]
        model_seed = (
            args.seed
            + outer_fold * 100_000
            + calibrator_fold * 100
        )
        bundle = _fit_bundle(
            cal_args,
            progress_matrix[fit_global],
            target_residual[fit_local],
            groups[fit_global],
            branches[fit_global],
            seed=model_seed,
        )
        (
            predicted,
            disagreement,
            branch_model,
            fallback_predicted,
        ) = _predict_bundle(
            bundle,
            progress_matrix[held_global],
            branches[held_global],
        )
        residual[validation_local] = predicted
        fallback_residual[validation_local] = fallback_predicted
        ensemble_std[validation_local] = disagreement
        used_branch[validation_local] = branch_model
        fold_record = {
            "fold": calibrator_fold,
            "seed": model_seed,
            "train": _identity(
                fit_global,
                identifiers,
                groups,
            ),
            "validation": _identity(
                held_global,
                identifiers,
                groups,
            ),
            "group_overlap": 0,
            "outer_validation_group_overlap": 0,
            "skipped_branches": bundle["skipped_branches"],
        }
        calibrator_folds.append(fold_record)
    if (
        not np.isfinite(residual).all()
        or not np.isfinite(fallback_residual).all()
        or not np.isfinite(ensemble_std).all()
    ):
        raise RuntimeError(
            f"outer fold {outer_fold}: calibrator cross-fit is incomplete"
        )

    branch_policies, shared_policy, _ = _choose_branch_policies(
        raw_progress[train_global],
        target_progress[train_global],
        residual,
        branches[train_global],
    )
    fallback_policy, _ = _choose_policy(
        raw_progress[train_global],
        target_progress[train_global],
        fallback_residual,
    )
    corrected_train, applied_train, clips_train, deadbands_train = (
        _apply_branch_policies(
            raw_progress[train_global],
            residual,
            branches[train_global],
            branch_policies,
            fallback_policy,
        )
    )

    final_seed = args.seed + outer_fold * 100_000 + 50_000
    final_bundle = _fit_bundle(
        cal_args,
        progress_matrix[train_global],
        target_residual,
        groups[train_global],
        branches[train_global],
        seed=final_seed,
    )
    if len(validation_global):
        (
            residual_validation,
            std_validation,
            branch_validation,
            _,
        ) = _predict_bundle(
            final_bundle,
            progress_matrix[validation_global],
            branches[validation_global],
        )
        (
            corrected_validation,
            applied_validation,
            clips_validation,
            deadbands_validation,
        ) = _apply_branch_policies(
            raw_progress[validation_global],
            residual_validation,
            branches[validation_global],
            branch_policies,
            fallback_policy,
        )
    else:
        residual_validation = np.empty(0, dtype=np.float64)
        std_validation = np.empty(0, dtype=np.float64)
        branch_validation = np.empty(0, dtype=bool)
        corrected_validation = np.empty(0, dtype=np.float64)
        applied_validation = np.empty(0, dtype=np.float64)
        clips_validation = np.empty(0, dtype=np.float64)
        deadbands_validation = np.empty(0, dtype=np.float64)

    calibration_rows: dict[int, dict[str, Any]] = {}

    def add_rows(
        global_indices: np.ndarray,
        *,
        corrected: np.ndarray,
        predicted_residual: np.ndarray,
        applied: np.ndarray,
        disagreement: np.ndarray,
        clips: np.ndarray,
        deadbands: np.ndarray,
        branch_model: np.ndarray,
        role: str,
    ) -> None:
        for local, global_index in enumerate(global_indices.tolist()):
            progress = float(corrected[local])
            calibration_rows[global_index] = {
                "sample_id": identifiers[global_index],
                "group_id": str(groups[global_index]),
                "joint_outer_fold": outer_fold,
                "joint_role": role,
                "corrected_progress_oof": progress,
                "corrected_prediction_oof": reading_from_progress(
                    progress,
                    rows[global_index].get("scale_start"),
                    rows[global_index].get("scale_end"),
                ),
                "predicted_residual_oof": float(predicted_residual[local]),
                "applied_residual_oof": float(applied[local]),
                "raw_progress": float(raw_progress[global_index]),
                "ensemble_std_oof": float(disagreement[local]),
                "correction_clip": float(clips[local]),
                "deadband": float(deadbands[local]),
                "used_branch_model": bool(branch_model[local]),
            }

    add_rows(
        train_global,
        corrected=corrected_train,
        predicted_residual=residual,
        applied=applied_train,
        disagreement=ensemble_std,
        clips=clips_train,
        deadbands=deadbands_train,
        branch_model=used_branch,
        role="cross_fitted_outer_training_candidate",
    )
    add_rows(
        validation_global,
        corrected=corrected_validation,
        predicted_residual=residual_validation,
        applied=applied_validation,
        disagreement=std_validation,
        clips=clips_validation,
        deadbands=deadbands_validation,
        branch_model=branch_validation,
        role="external_outer_validation_candidate",
    )
    lineage = {
        "cross_fit_seed": split_seed,
        "trainable_outer_training": _identity(
            train_global,
            identifiers,
            groups,
        ),
        "external_outer_validation": _identity(
            validation_global,
            identifiers,
            groups,
        ),
        "cross_fitted_training_candidates": calibrator_folds,
        "policy_selection": {
            "scope": _identity(train_global, identifiers, groups),
            "branch_policies": branch_policies,
            "shared_policy": shared_policy,
            "fallback_policy": fallback_policy,
            "outer_validation_group_overlap": 0,
        },
        "final_fit": {
            "seed": final_seed,
            "scope": _identity(train_global, identifiers, groups),
            "outer_validation_group_overlap": 0,
            "skipped_branches": final_bundle["skipped_branches"],
        },
    }
    return calibration_rows, lineage


def _baseline_prediction(row: Mapping[str, Any]) -> float | None:
    payload = row.get("base")
    if not isinstance(payload, Mapping) or payload.get("status") is not True:
        return None
    return finite_float(payload.get("prediction"))


def run_joint_fadr(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run joint outer-group stacking; exposed for synthetic tests."""

    _validate_args(args)
    oof_pairs = args.oof_pairs.resolve()
    input_preflight = args.input_preflight.resolve()
    assert_train_only_path(oof_pairs, label="joint FADR OOF input")
    assert_train_only_path(input_preflight, label="joint FADR input preflight")
    preflight = _validate_preflight(
        input_preflight,
        oof_pairs=oof_pairs,
        seed=args.seed,
    )
    rows = strict_jsonl_load(oof_pairs)
    if any(
        row.get("dataset") != "SyncG" or row.get("split") != "train"
        for row in rows
    ):
        raise ValueError("joint FADR accepts only SyncG/train")
    identifiers = [str(row.get("sample_id")) for row in rows]
    if (
        any(not sample_id for sample_id in identifiers)
        or len(identifiers) != len(set(identifiers))
    ):
        raise ValueError("joint FADR input has invalid or duplicate sample IDs")
    groups = np.asarray(
        [str(row.get("group_id")) for row in rows],
        dtype=object,
    )
    if any(not group for group in groups.tolist()):
        raise ValueError("joint FADR input has invalid physical groups")
    branches = np.asarray(
        [normalize_reference_branch(row) for row in rows],
        dtype=object,
    )
    raw_progress = np.asarray(
        [
            value if (value := _vector_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    target_progress = np.asarray(
        [
            value if (value := _target_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    vector_available = np.isfinite(raw_progress)
    calibrator_trainable = vector_available & np.isfinite(target_progress)
    progress_matrix = calibrator_feature_matrix(
        [extract_progress_features(row) for row in rows]
    )
    base_values = [_baseline_prediction(row) for row in rows]
    base_success = np.asarray([value is not None for value in base_values])
    base_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, base_values)
        ],
        dtype=np.float64,
    )
    outer_splits = _grouped_splits(
        groups,
        np.zeros(len(rows), dtype=np.float64),
        folds=args.outer_folds,
        seed=args.seed,
        label="joint FADR outer",
    )
    full_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    calibrated_values: list[float | None] = [None] * len(rows)
    calibration_diagnostic: list[dict[str, Any] | None] = [None] * len(rows)
    variant_scores = {
        name: np.full(len(rows), math.nan, dtype=np.float64)
        for name in FADR_ROUTER_FEATURE_SETS
    }
    variant_thresholds = {
        name: np.full(len(rows), math.nan, dtype=np.float64)
        for name in FADR_ROUTER_FEATURE_SETS
    }
    outer_fold_ids = np.full(len(rows), -1, dtype=np.int64)
    fold_lineage_ids: list[str | None] = [None] * len(rows)
    outer_lineage: list[dict[str, Any]] = []

    for outer_fold, (outer_train, outer_validation) in enumerate(
        outer_splits,
        start=1,
    ):
        train_group_set = set(groups[outer_train].tolist())
        validation_group_set = set(groups[outer_validation].tolist())
        if train_group_set & validation_group_set:
            raise RuntimeError(f"joint outer fold {outer_fold} leaks groups")
        calibration_rows, calibrator_lineage = _calibrator_for_outer_fold(
            args=args,
            outer_fold=outer_fold,
            outer_train=outer_train,
            outer_validation=outer_validation,
            identifiers=identifiers,
            groups=groups,
            branches=branches,
            progress_matrix=progress_matrix,
            raw_progress=raw_progress,
            target_progress=target_progress,
            vector_available=vector_available,
            calibrator_trainable=calibrator_trainable,
            rows=rows,
        )
        for global_index in outer_validation.tolist():
            calibration = calibration_rows.get(global_index)
            calibration_diagnostic[global_index] = calibration
            if calibration is not None:
                calibrated_values[global_index] = finite_float(
                    calibration.get("corrected_prediction_oof")
                )

        feature_indices = np.concatenate([outer_train, outer_validation])
        feature_rows = [
            extract_calibrated_router_features(
                raw_row=rows[global_index],
                base_row=rows[global_index],
                vector_row=rows[global_index],
                reference_row=None,
                calibration_row=calibration_rows.get(global_index, {}),
            )
            for global_index in feature_indices.tolist()
        ]
        matrix_combined = router_feature_matrix(feature_rows)
        matrix_by_global = {
            global_index: matrix_combined[local]
            for local, global_index in enumerate(feature_indices.tolist())
        }
        train_joint = np.asarray(
            [
                global_index
                for global_index in outer_train.tolist()
                if base_values[global_index] is not None
                and calibration_rows.get(global_index) is not None
                and np.isfinite(target_progress[global_index])
            ],
            dtype=np.int64,
        )
        validation_joint = np.asarray(
            [
                global_index
                for global_index in outer_validation.tolist()
                if base_values[global_index] is not None
                and calibration_rows.get(global_index) is not None
            ],
            dtype=np.int64,
        )
        if len(train_joint) < MIN_ROUTER_JOINT_SAMPLES:
            raise ValueError(
                f"outer fold {outer_fold}: too few joint router-training samples"
            )
        train_calibrated = [
            finite_float(
                calibration_rows[global_index].get(
                    "corrected_prediction_oof"
                )
            )
            for global_index in train_joint.tolist()
        ]
        if any(value is None for value in train_calibrated):
            raise RuntimeError("joint calibrator training candidate is missing")
        train_base_error = np.asarray(
            [
                normalized_error(rows[index], base_values[index])
                for index in train_joint.tolist()
            ],
            dtype=np.float64,
        )
        train_calibrated_error = np.asarray(
            [
                normalized_error(rows[index], value)
                for index, value in zip(
                    train_joint.tolist(),
                    train_calibrated,
                )
            ],
            dtype=np.float64,
        )
        router_target = np.clip(
            train_base_error - train_calibrated_error,
            -1.0,
            1.0,
        )
        router_groups = groups[train_joint]
        router_split_seed = args.seed + outer_fold * 100_000 + 60_000
        router_splits = _grouped_splits(
            router_groups,
            router_target,
            folds=args.router_inner_folds,
            seed=router_split_seed,
            label=f"joint outer {outer_fold} router inner",
        )
        router_inner_lineage = [
            {
                "fold": inner_fold,
                "train": _identity(
                    train_joint[fit_local],
                    identifiers,
                    groups,
                ),
                "validation": _identity(
                    train_joint[held_local],
                    identifiers,
                    groups,
                ),
                "group_overlap": 0,
                "outer_validation_group_overlap": 0,
            }
            for inner_fold, (fit_local, held_local) in enumerate(
                router_splits,
                start=1,
            )
        ]
        matrix_train_full = np.stack(
            [matrix_by_global[index] for index in train_joint.tolist()]
        )
        matrix_validation_full = (
            np.stack(
                [
                    matrix_by_global[index]
                    for index in validation_joint.tolist()
                ]
            )
            if len(validation_joint)
            else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        )
        variant_lineage: dict[str, Any] = {}
        for variant, feature_names in FADR_ROUTER_FEATURE_SETS.items():
            columns = [full_index[name] for name in feature_names]
            matrix_train = matrix_train_full[:, columns]
            matrix_validation = matrix_validation_full[:, columns]
            inner_score = np.full(len(train_joint), math.nan, dtype=np.float64)
            for inner_fold, (fit_local, held_local) in enumerate(
                router_splits,
                start=1,
            ):
                model_seed = (
                    args.seed
                    + outer_fold * 100_000
                    + 70_000
                    + inner_fold
                )
                model = _build_router_model(args, seed=model_seed)
                model.fit(matrix_train[fit_local], router_target[fit_local])
                inner_score[held_local] = deterministic_router_prediction(
                    model,
                    matrix_train[held_local],
                )
            if not np.isfinite(inner_score).all():
                raise RuntimeError(
                    f"outer fold {outer_fold}, {variant}: inner scores incomplete"
                )
            threshold, threshold_selection = _choose_threshold(
                inner_score,
                train_base_error,
                train_calibrated_error,
            )
            final_router_seed = (
                args.seed + outer_fold * 100_000 + 80_000
            )
            final_model = _build_router_model(
                args,
                seed=final_router_seed,
            )
            final_model.fit(matrix_train, router_target)
            if len(validation_joint):
                scores = deterministic_router_prediction(
                    final_model,
                    matrix_validation,
                )
                variant_scores[variant][validation_joint] = scores
                variant_thresholds[variant][validation_joint] = threshold
            variant_lineage[variant] = {
                "feature_names": list(feature_names),
                "feature_count": len(feature_names),
                "inner_score_seed_rule": (
                    "fadr_seed + outer_fold*100000 + 70000 + inner_fold"
                ),
                "final_model_seed": final_router_seed,
                "threshold": float(threshold),
                "threshold_selection": threshold_selection,
                "training": _identity(
                    train_joint,
                    identifiers,
                    groups,
                ),
                "validation": _identity(
                    validation_joint,
                    identifiers,
                    groups,
                ),
                "outer_validation_group_overlap": 0,
            }
        fold_record: dict[str, Any] = {
            "fold": outer_fold,
            "outer_train": _identity(
                outer_train,
                identifiers,
                groups,
            ),
            "outer_validation": _identity(
                outer_validation,
                identifiers,
                groups,
            ),
            "group_overlap": 0,
            "calibrator": calibrator_lineage,
            "router": {
                "inner_split_seed": router_split_seed,
                "training_joint": _identity(
                    train_joint,
                    identifiers,
                    groups,
                ),
                "validation_joint": _identity(
                    validation_joint,
                    identifiers,
                    groups,
                ),
                "inner_folds": router_inner_lineage,
                "variants": variant_lineage,
            },
            "outer_validation_exclusion": {
                "calibrator_model_fits": True,
                "calibrator_clip_deadband_selection": True,
                "router_model_fits": True,
                "router_threshold_selection": True,
                "router_feature_construction_for_training": True,
                "router_target_construction": True,
                "bootstrap_prediction_tuning": True,
            },
            "bootstrap_prediction_tuning": {
                "performed": False,
                "note": (
                    "bootstrap is post-hoc physical-group inference only and "
                    "does not affect any outer prediction"
                ),
            },
        }
        fold_record["fold_lineage_id"] = _canonical_sha256(fold_record)
        outer_lineage.append(fold_record)
        for index in outer_validation.tolist():
            outer_fold_ids[index] = outer_fold
            fold_lineage_ids[index] = fold_record["fold_lineage_id"]

    if np.any(outer_fold_ids < 1) or any(value is None for value in fold_lineage_ids):
        raise RuntimeError("joint outer assignment is incomplete")
    calibrated_success = np.asarray(
        [value is not None for value in calibrated_values]
    )
    calibrated_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, calibrated_values)
        ],
        dtype=np.float64,
    )
    hard_values = [
        base if base is not None else calibrated
        for base, calibrated in zip(base_values, calibrated_values)
    ]
    hard_success = base_success | calibrated_success
    hard_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, hard_values)
        ],
        dtype=np.float64,
    )
    variant_values: dict[str, list[float | None]] = {}
    variant_routes: dict[str, list[str]] = {}
    variant_errors: dict[str, np.ndarray] = {}
    variants_summary: dict[str, Any] = {}
    for variant in FADR_ROUTER_FEATURE_SETS:
        values: list[float | None] = []
        routes: list[str] = []
        for index in range(len(rows)):
            score = finite_float(variant_scores[variant][index])
            threshold = finite_float(variant_thresholds[variant][index])
            prediction, route = _route(
                base_values[index],
                calibrated_values[index],
                score,
                threshold,
            )
            values.append(prediction)
            routes.append(route)
        errors = np.asarray(
            [
                normalized_error(row, value)
                for row, value in zip(rows, values)
            ],
            dtype=np.float64,
        )
        successful = np.asarray([value is not None for value in values])
        switched = np.asarray(
            [route == "calibrated_quality_switch" for route in routes]
        )
        comparisons = {
            "router_vs_base_mask": _paired_group_bootstrap(
                errors,
                base_error,
                groups,
                seed=args.seed,
                iterations=args.bootstrap_iterations,
            ),
            "router_vs_reference_conditioned_vector": _paired_group_bootstrap(
                errors,
                calibrated_error,
                groups,
                seed=args.seed + 1,
                iterations=args.bootstrap_iterations,
            ),
            "router_vs_hard_fallback": _paired_group_bootstrap(
                errors,
                hard_error,
                groups,
                seed=args.seed + 2,
                iterations=args.bootstrap_iterations,
            ),
        }
        variant_values[variant] = values
        variant_routes[variant] = routes
        variant_errors[variant] = errors
        variants_summary[variant] = {
            "feature_names": list(FADR_ROUTER_FEATURE_SETS[variant]),
            "feature_count": len(FADR_ROUTER_FEATURE_SETS[variant]),
            "metrics": _metrics(errors, successful),
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
            "paired_comparisons": comparisons,
        }
    for variant in FADR_ROUTER_FEATURE_SETS:
        if variant == "full":
            continue
        variants_summary[variant]["paired_comparisons"][
            "router_vs_full"
        ] = _paired_group_bootstrap(
            variant_errors[variant],
            variant_errors["full"],
            groups,
            seed=args.seed + 3,
            iterations=args.bootstrap_iterations,
        )

    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        variants = {}
        for variant in FADR_ROUTER_FEATURE_SETS:
            variants[variant] = {
                "router_score_oof": finite_float(
                    variant_scores[variant][index]
                ),
                "nested_threshold": finite_float(
                    variant_thresholds[variant][index]
                ),
                "route": variant_routes[variant][index],
                "prediction": variant_values[variant][index],
                "normalized_error": float(variant_errors[variant][index]),
            }
        diagnostics.append(
            {
                "sample_id": identifiers[index],
                "group_id": str(groups[index]),
                "held_out_seed": row.get("held_out_seed"),
                "fadr_seed": args.seed,
                "joint_outer_fold": int(outer_fold_ids[index]),
                "fold_lineage_id": fold_lineage_ids[index],
                "base_prediction": base_values[index],
                "reference_conditioned_prediction": calibrated_values[index],
                "calibrator": calibration_diagnostic[index],
                "variants": variants,
            }
        )
    source_identity = joint_source_identity()
    parameters = {
        "outer_folds": args.outer_folds,
        "calibrator_folds": args.calibrator_folds,
        "router_inner_folds": args.router_inner_folds,
        "calibrator_trees": args.calibrator_trees,
        "calibrator_max_depth": args.calibrator_max_depth,
        "calibrator_min_samples_leaf": args.calibrator_min_samples_leaf,
        "calibrator_max_features": args.calibrator_max_features,
        "min_branch_samples": args.min_branch_samples,
        "min_branch_groups": args.min_branch_groups,
        "router_trees": args.router_trees,
        "router_max_depth": args.router_max_depth,
        "router_min_samples_leaf": args.router_min_samples_leaf,
        "router_max_features": args.router_max_features,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    lineage = {
        "schema_version": 1,
        "protocol": FADR_JOINT_LINEAGE_PROTOCOL,
        "status": "complete",
        "seed": args.seed,
        "scope": "SyncG/train joint outer-group stacking only",
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_oof_protocol": EXPECTED_OOF_PROTOCOL,
        "input_preflight": str(input_preflight),
        "input_preflight_sha256": sha256_file(input_preflight),
        "input_authorization_sha256": preflight[
            "input_authorization_sha256"
        ],
        "samples": len(rows),
        "groups": len(set(groups.tolist())),
        "outer_fold_assignment": (
            "all physical groups are assigned once before any calibrator or "
            "router fit; each fold rebuilds the complete stack"
        ),
        "standalone_calibrator_oof_used": False,
        "standalone_router_oof_used": False,
        "combined_fadr_oof_authorized": True,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "public_test_field_evaluation_authorized": False,
        "parameters": parameters,
        "feature_sets": {
            name: list(features)
            for name, features in FADR_ROUTER_FEATURE_SETS.items()
        },
        "outer_folds": outer_lineage,
        "source_identity": source_identity,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "strict_json_protocol": STRICT_JSON_PROTOCOL,
    }
    summary = {
        "schema_version": 1,
        "protocol": FADR_JOINT_TRAINING_PROTOCOL,
        "status": "complete",
        "seed": args.seed,
        "scope": "SyncG/train joint outer-group stacking only",
        "joint_outer_group_nested": True,
        "standalone_calibrator_oof_used": False,
        "standalone_router_oof_used": False,
        "combined_fadr_oof_authorized": True,
        "samples": len(rows),
        "groups": len(set(groups.tolist())),
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_oof_protocol": EXPECTED_OOF_PROTOCOL,
        "input_preflight": str(input_preflight),
        "input_preflight_sha256": sha256_file(input_preflight),
        "input_authorization_sha256": preflight[
            "input_authorization_sha256"
        ],
        "parameters": parameters,
        "feature_sets": lineage["feature_sets"],
        "baselines": {
            "base_mask": _metrics(base_error, base_success),
            "reference_conditioned_vector": _metrics(
                calibrated_error,
                calibrated_success,
            ),
            "hard_fallback": _metrics(hard_error, hard_success),
        },
        "variants": variants_summary,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "public_test_field_evaluation_authorized": False,
        "full_is_preregistered_primary": True,
        "no_reference_variant_interpretation": (
            "router evidence ablation only; the fold-specific calibrated "
            "candidate remains present"
        ),
        "bootstrap_role": (
            "post-hoc paired physical-group inference only; never used to "
            "tune an outer prediction"
        ),
        "source_identity": source_identity,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "strict_json_protocol": STRICT_JSON_PROTOCOL,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    return diagnostics, lineage, summary


def _serialize_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _serialize_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{path} already exists; refusing to overwrite"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.input_preflight = args.input_preflight.resolve()
    args.output_dir = args.output_dir.resolve()
    assert_train_only_path(args.output_dir, label="joint FADR output")
    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists; refusing to reuse a joint namespace"
        )
    diagnostics, lineage, summary = run_joint_fadr(args)
    lineage_path = args.output_dir / LINEAGE_FILENAME
    diagnostics_path = args.output_dir / DIAGNOSTICS_FILENAME
    summary_path = args.output_dir / SUMMARY_FILENAME
    created: list[Path] = []
    try:
        _write_no_clobber(lineage_path, _serialize_json(lineage))
        created.append(lineage_path)
        _write_no_clobber(
            diagnostics_path,
            _serialize_jsonl(diagnostics),
        )
        created.append(diagnostics_path)
        summary.update(
            {
                "lineage": str(lineage_path),
                "lineage_sha256": sha256_file(lineage_path),
                "diagnostics": str(diagnostics_path),
                "diagnostics_sha256": sha256_file(diagnostics_path),
            }
        )
        _write_no_clobber(summary_path, _serialize_json(summary))
        created.append(summary_path)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(summary_path)


if __name__ == "__main__":
    main()
