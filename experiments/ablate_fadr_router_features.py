"""Strict nested grouped-OOF router feature ablations for FADR v2.

Only the selected router columns differ among variants.  Candidate readings,
outer/inner folds, estimator hyperparameters, threshold selection, failure
penalty, and the hard-fallback policy are shared exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments.calibrated_progress_router import (
    FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix,
)
from experiments.evaluate_reference_conditioned_pipeline import _validate_calibrator
from experiments.fadr_feature_sets import (
    FADR_ROUTER_FEATURE_ABLATION_PROTOCOL,
    FADR_ROUTER_FEATURE_SETS,
)
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    FADR_INPUT_PREFLIGHT_PROTOCOL,
    FADR_PRIMARY_SEED,
    FADR_SEEDS,
    assert_train_only_path,
    sha256_file,
    sha256_strings,
    strict_json_load,
    strict_jsonl_load,
)
from experiments.quality_router import finite_float, normalized_error
from experiments.reference_conditioned_progress_calibrator import (
    REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
)
from experiments.reference_conditioned_router import deterministic_router_prediction
from experiments.strict_json import strict_json_source_sha256
from experiments.train_quality_router import (
    _build_model,
    _choose_threshold,
    _paired_group_bootstrap,
)
from experiments.train_reference_conditioned_progress_calibrator import (
    TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_source_file,
)


DIAGNOSTICS_FILENAME = "strict_nested_feature_ablation_diagnostics.jsonl"
SUMMARY_FILENAME = "feature_ablation_summary.json"
MIN_JOINT_SAMPLES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-pairs", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--calibration-diagnostics", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True, choices=FADR_SEEDS)
    return parser.parse_args()


def _prediction(row: Mapping[str, Any], name: str) -> float | None:
    nested = row.get(name)
    return finite_float(nested.get("prediction")) if isinstance(nested, Mapping) else None


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


def _grouped_splits(
    groups: np.ndarray,
    target: np.ndarray,
    *,
    folds: int,
    seed: int,
    label: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = len(set(groups.tolist()))
    if unique_groups < folds:
        raise ValueError(f"{label}: cannot make {folds} folds from {unique_groups} groups")
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(
        splitter.split(np.zeros((len(groups), 1), dtype=np.float64), target, groups)
    )
    for train_index, validation_index in splits:
        if set(groups[train_index].tolist()) & set(groups[validation_index].tolist()):
            raise RuntimeError(f"{label}: grouped split leakage")
    return splits


def _route(
    base: float | None,
    calibrated: float | None,
    score: float | None,
    threshold: float | None,
) -> tuple[float | None, str]:
    if base is None:
        return (
            (calibrated, "calibrated_hard_fallback")
            if calibrated is not None
            else (None, "failure")
        )
    if (
        calibrated is not None
        and score is not None
        and threshold is not None
        and score > threshold
    ):
        return calibrated, "calibrated_quality_switch"
    return base, "base"


def _validate_preflight(
    path: Path,
    *,
    oof_pairs: Path,
    seed: int,
) -> dict[str, Any]:
    preflight = strict_json_load(path)
    oof_identity = (preflight.get("inputs") or {}).get("oof_pairs") or {}
    if (
        preflight.get("schema_version") != 1
        or preflight.get("protocol") != FADR_INPUT_PREFLIGHT_PROTOCOL
        or preflight.get("status") != "verified"
        or preflight.get("train_only_certified") is not True
        or preflight.get("group_leakage_count") != 0
        or preflight.get("test_samples_used") != 0
        or preflight.get("public_samples_used") != 0
        or preflight.get("field_samples_used") != 0
        or preflight.get("public_test_field_evaluation_authorized") is not False
        or preflight.get("fadr_seeds") != list(FADR_SEEDS)
        or preflight.get("primary_fadr_seed") != FADR_PRIMARY_SEED
        or preflight.get("oof_protocol") != EXPECTED_OOF_PROTOCOL
        or seed not in preflight.get("fadr_seeds", [])
        or oof_identity.get("path") != str(oof_pairs)
        or oof_identity.get("sha256") != sha256_file(oof_pairs)
    ):
        raise ValueError("FADR input preflight audit failed")
    sources = preflight.get("source_identity") or {}
    expected_sources = {
        "preflight": sha256_file(
            PROJECT_DIR / "experiments" / "preflight_fadr_multiseed.py"
        ),
        "protocol": sha256_file(
            PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
        ),
        "strict_json": strict_json_source_sha256(),
    }
    if sources != expected_sources:
        raise ValueError("FADR input preflight source identity drifted")
    return preflight


def _validate_calibration_inputs(
    *,
    oof_pairs: Path,
    rows: list[dict[str, Any]],
    diagnostics_path: Path,
    summary_path: Path,
    calibrator_path: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary = strict_json_load(summary_path)
    diagnostics = strict_jsonl_load(diagnostics_path)
    artifact = _validate_calibrator(calibrator_path)
    if (
        summary.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("seed") != seed
        or summary.get("input_sha256") != sha256_file(oof_pairs)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("model_sha256") != sha256_file(calibrator_path)
        or summary.get("strict_nested_oof") is not True
        or summary.get("group_leakage_count") != 0
        or summary.get("test_samples_used") != 0
        or artifact.get("protocol") != REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL
        or artifact.get("training_protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or artifact.get("seed") != seed
        or artifact.get("training_oof_pairs_sha256") != sha256_file(oof_pairs)
        or summary.get("input_oof_protocol") != EXPECTED_OOF_PROTOCOL
        or artifact.get("input_oof_protocol") != EXPECTED_OOF_PROTOCOL
        or artifact.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL
        or artifact.get("source_sha256") != summary.get("source_sha256")
    ):
        raise ValueError("calibrator/diagnostics binding audit failed")
    identifiers = [row.get("sample_id") for row in rows]
    by_id = {row.get("sample_id"): row for row in diagnostics}
    if (
        any(not isinstance(sample_id, str) or not sample_id for sample_id in identifiers)
        or len(identifiers) != len(set(identifiers))
        or len(by_id) != len(diagnostics)
        or set(by_id) != set(identifiers)
    ):
        raise ValueError("calibration diagnostics identifiers differ from OOF input")
    for raw in rows:
        diagnostic = by_id[raw["sample_id"]]
        if diagnostic.get("group_id") != raw.get("group_id"):
            raise ValueError(f"{raw['sample_id']}: calibration group identity mismatch")
    return diagnostics, by_id


def run_feature_ablation(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the five frozen variants; exposed for synthetic tests."""

    oof_pairs = args.oof_pairs.resolve()
    input_preflight = args.input_preflight.resolve()
    calibration_diagnostics = args.calibration_diagnostics.resolve()
    calibration_summary = args.calibration_summary.resolve()
    calibrator = args.calibrator.resolve()
    for label, path in {
        "OOF pairs": oof_pairs,
        "input preflight": input_preflight,
        "calibration diagnostics": calibration_diagnostics,
        "calibration summary": calibration_summary,
        "calibrator": calibrator,
    }.items():
        assert_train_only_path(path, label=label)
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.folds < 3
        or args.inner_folds < 3
        or args.trees <= 0
        or args.min_samples_leaf <= 0
        or args.bootstrap_iterations <= 0
        or not 0.0 < args.max_features <= 1.0
    ):
        raise ValueError("invalid FADR feature-ablation parameters")
    preflight = _validate_preflight(
        input_preflight,
        oof_pairs=oof_pairs,
        seed=args.seed,
    )
    rows = strict_jsonl_load(oof_pairs)
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("FADR feature ablation accepts only SyncG/train")
    _, calibration_by_id = _validate_calibration_inputs(
        oof_pairs=oof_pairs,
        rows=rows,
        diagnostics_path=calibration_diagnostics,
        summary_path=calibration_summary,
        calibrator_path=calibrator,
        seed=args.seed,
    )

    identifiers = [str(row["sample_id"]) for row in rows]
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    base = [_prediction(row, "base") for row in rows]
    calibrated = [
        finite_float(calibration_by_id[sample_id].get("corrected_prediction_oof"))
        for sample_id in identifiers
    ]
    base_success = np.asarray([value is not None for value in base], dtype=bool)
    calibrated_success = np.asarray(
        [value is not None for value in calibrated],
        dtype=bool,
    )
    joint = base_success & calibrated_success
    if int(np.sum(joint)) < MIN_JOINT_SAMPLES:
        raise ValueError(
            "too few joint-success rows for the frozen FADR feature ablation"
        )
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
    full_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    matrices = {
        variant: matrix_all[:, [full_index[name] for name in names]][joint]
        for variant, names in FADR_ROUTER_FEATURE_SETS.items()
    }
    outer_splits = _grouped_splits(
        groups,
        target,
        folds=args.folds,
        seed=args.seed,
        label="outer FADR feature ablation",
    )
    states: dict[str, dict[str, Any]] = {
        variant: {
            "score": np.full(len(rows), math.nan, dtype=np.float64),
            "threshold": np.full(len(rows), math.nan, dtype=np.float64),
            "nested_thresholds": {},
        }
        for variant in FADR_ROUTER_FEATURE_SETS
    }
    fold_id = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(outer_splits, start=1):
        train_groups = groups[train_index]
        validation_groups = groups[validation_index]
        inner_splits = _grouped_splits(
            train_groups,
            target[train_index],
            folds=args.inner_folds,
            seed=args.seed + fold * 10_000,
            label=f"inner FADR feature ablation outer fold {fold}",
        )
        destination = joint_indices[validation_index]
        fold_id[destination] = fold
        for variant, matrix in matrices.items():
            inner_score = np.full(len(train_index), math.nan, dtype=np.float64)
            for inner_fold, (
                inner_train_index,
                inner_validation_index,
            ) in enumerate(inner_splits, start=1):
                model = _build_model(
                    args,
                    seed=args.seed + fold * 10_000 + inner_fold,
                )
                model.fit(
                    matrix[train_index][inner_train_index],
                    target[train_index][inner_train_index],
                )
                inner_score[inner_validation_index] = deterministic_router_prediction(
                    model,
                    matrix[train_index][inner_validation_index],
                )
            if not np.isfinite(inner_score).all():
                raise RuntimeError(f"{variant}: inner scores are incomplete")
            threshold, _ = _choose_threshold(
                inner_score,
                base_error[joint][train_index],
                calibrated_error[joint][train_index],
            )
            model = _build_model(args, seed=args.seed + fold * 1_000)
            model.fit(matrix[train_index], target[train_index])
            states[variant]["score"][destination] = deterministic_router_prediction(
                model,
                matrix[validation_index],
            )
            states[variant]["threshold"][destination] = threshold
            states[variant]["nested_thresholds"][str(fold)] = float(threshold)
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(set(train_groups.tolist())),
                "validation_groups": len(set(validation_groups.tolist())),
                "train_group_ids_sha256": sha256_strings(
                    sorted(set(train_groups.tolist()))
                ),
                "validation_group_ids_sha256": sha256_strings(
                    sorted(set(validation_groups.tolist()))
                ),
                "group_overlap": 0,
                "inner_folds": [
                    {
                        "fold": inner_fold,
                        "train_groups": len(
                            set(train_groups[inner_train].tolist())
                        ),
                        "validation_groups": len(
                            set(train_groups[inner_validation].tolist())
                        ),
                        "train_group_ids_sha256": sha256_strings(
                            sorted(set(train_groups[inner_train].tolist()))
                        ),
                        "validation_group_ids_sha256": sha256_strings(
                            sorted(set(train_groups[inner_validation].tolist()))
                        ),
                        "group_overlap": 0,
                    }
                    for inner_fold, (
                        inner_train,
                        inner_validation,
                    ) in enumerate(inner_splits, start=1)
                ],
            }
        )
    if np.any(fold_id[joint] < 1):
        raise RuntimeError("FADR feature-ablation outer folds are incomplete")

    hard = [
        base_value if base_value is not None else calibrated_value
        for base_value, calibrated_value in zip(base, calibrated)
    ]
    hard_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, hard)],
        dtype=np.float64,
    )
    hard_success = base_success | calibrated_success
    routed_by_variant: dict[str, list[float | None]] = {}
    route_by_variant: dict[str, list[str]] = {}
    error_by_variant: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    for variant, names in FADR_ROUTER_FEATURE_SETS.items():
        state = states[variant]
        if (
            not np.isfinite(state["score"][joint]).all()
            or not np.isfinite(state["threshold"][joint]).all()
        ):
            raise RuntimeError(f"{variant}: nested OOF scores are incomplete")
        routed: list[float | None] = []
        routes: list[str] = []
        for index, (base_value, calibrated_value) in enumerate(
            zip(base, calibrated)
        ):
            score = finite_float(state["score"][index])
            threshold = finite_float(state["threshold"][index])
            prediction, route = _route(
                base_value,
                calibrated_value,
                score,
                threshold,
            )
            routed.append(prediction)
            routes.append(route)
        routed_error = np.asarray(
            [normalized_error(row, value) for row, value in zip(rows, routed)],
            dtype=np.float64,
        )
        routed_success = np.asarray([value is not None for value in routed], dtype=bool)
        switched = np.asarray(
            [route == "calibrated_quality_switch" for route in routes],
            dtype=bool,
        )
        routed_by_variant[variant] = routed
        route_by_variant[variant] = routes
        error_by_variant[variant] = routed_error
        results[variant] = {
            "feature_names": list(names),
            "feature_count": len(names),
            "metrics": _metrics(routed_error, routed_success),
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
            "nested_thresholds": state["nested_thresholds"],
            "paired_comparisons": {
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
            },
        }
    for variant in FADR_ROUTER_FEATURE_SETS:
        if variant == "full":
            continue
        results[variant]["paired_comparisons"]["router_vs_full"] = (
            _paired_group_bootstrap(
                error_by_variant[variant],
                error_by_variant["full"],
                groups_all,
                seed=args.seed + 3,
                iterations=args.bootstrap_iterations,
            )
        )

    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        diagnostics.append(
            {
                "sample_id": row.get("sample_id"),
                "group_id": row.get("group_id"),
                "held_out_seed": row.get("held_out_seed"),
                "fadr_seed": args.seed,
                "router_fold": int(fold_id[index]) if joint[index] else None,
                "base_prediction": base[index],
                "reference_conditioned_prediction": calibrated[index],
                "variants": {
                    variant: {
                        "router_score_oof": finite_float(
                            states[variant]["score"][index]
                        ),
                        "nested_threshold": finite_float(
                            states[variant]["threshold"][index]
                        ),
                        "route": route_by_variant[variant][index],
                        "prediction": routed_by_variant[variant][index],
                        "normalized_error": float(error_by_variant[variant][index]),
                    }
                    for variant in FADR_ROUTER_FEATURE_SETS
                },
            }
        )

    source_hashes = {
        "trainer": sha256_source_file(Path(__file__).resolve()),
        "feature_sets": sha256_source_file(
            PROJECT_DIR / "experiments" / "fadr_feature_sets.py"
        ),
        "features": sha256_source_file(
            PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
        ),
        "quality_features": sha256_source_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ),
        "policy": sha256_source_file(
            PROJECT_DIR / "experiments" / "quality_router.py"
        ),
        "shared_training_helpers": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_quality_router.py"
        ),
    }
    summary = {
        "schema_version": 1,
        "protocol": FADR_ROUTER_FEATURE_ABLATION_PROTOCOL,
        "status": "complete",
        "scope": "SyncG/train strict nested grouped OOF only",
        "interpretation": {
            "without_reference_conditioned_router_evidence": (
                "router evidence ablation only; the reference-conditioned "
                "candidate reading is unchanged"
            )
        },
        "seed": args.seed,
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_oof_protocol": EXPECTED_OOF_PROTOCOL,
        "input_preflight": str(input_preflight),
        "input_preflight_sha256": sha256_file(input_preflight),
        "calibration_diagnostics": str(calibration_diagnostics),
        "calibration_diagnostics_sha256": sha256_file(calibration_diagnostics),
        "calibration_summary": str(calibration_summary),
        "calibration_summary_sha256": sha256_file(calibration_summary),
        "calibrator": str(calibrator),
        "calibrator_sha256": sha256_file(calibrator),
        "samples": len(rows),
        "groups": len(set(groups_all.tolist())),
        "joint_training_samples": int(np.sum(joint)),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "fold_summaries": fold_summaries,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "hard_fallback_policy": (
            "base failure -> reference-conditioned vector; joint failure -> failure"
        ),
        "failure_penalty_nmae": 1.0,
        "shared_hyperparameters": {
            "trees": args.trees,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "baselines": {
            "base_mask": _metrics(base_error, base_success),
            "reference_conditioned_vector": _metrics(
                calibrated_error,
                calibrated_success,
            ),
            "hard_fallback": _metrics(hard_error, hard_success),
        },
        "feature_sets": {
            name: list(features) for name, features in FADR_ROUTER_FEATURE_SETS.items()
        },
        "variants": results,
        "source_sha256": source_hashes,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "input_authorization_sha256": preflight["input_authorization_sha256"],
    }
    return diagnostics, summary


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
            raise FileExistsError(f"{path} already exists; refusing to overwrite") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    raise RuntimeError(
        "standalone calibrator OOF + independently folded router ablation is "
        "retired and is not valid combined FADR evidence; run "
        "experiments.train_joint_nested_fadr instead"
    )
    output_dir = args.output_dir.resolve()
    assert_train_only_path(output_dir, label="FADR feature-ablation output")
    diagnostics_path = output_dir / DIAGNOSTICS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    for path in (diagnostics_path, summary_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite")
    diagnostics, summary = run_feature_ablation(args)
    diagnostics_payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in diagnostics
    ).encode("utf-8")
    _write_no_clobber(diagnostics_path, diagnostics_payload)
    summary["diagnostics"] = str(diagnostics_path)
    summary["diagnostics_sha256"] = sha256_file(diagnostics_path)
    summary_payload = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _write_no_clobber(summary_path, summary_payload)
    except Exception:
        diagnostics_path.unlink(missing_ok=True)
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
