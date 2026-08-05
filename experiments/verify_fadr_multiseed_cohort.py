"""Verify and aggregate the exact three-seed, five-feature-set FADR v2 cohort."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.ablate_fadr_router_features import (
    DIAGNOSTICS_FILENAME,
    SUMMARY_FILENAME,
    _grouped_splits,
)
from experiments.fadr_feature_sets import (
    FADR_ROUTER_FEATURE_ABLATION_PROTOCOL,
    FADR_ROUTER_FEATURE_SETS,
)
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    FADR_INPUT_PREFLIGHT_PROTOCOL,
    FADR_JOINT_LINEAGE_PROTOCOL,
    FADR_JOINT_TRAINING_PROTOCOL,
    FADR_JOINT_VERIFICATION_PROTOCOL,
    FADR_MULTI_SEED_COHORT_PROTOCOL,
    FADR_PRIMARY_SEED,
    FADR_SEEDS,
    FADR_UDSF_HANDOFF_PROTOCOL,
    assert_train_only_path,
    sha256_file,
    sha256_strings,
    strict_json_load,
    strict_jsonl_load,
)
from experiments.quality_router import finite_float, normalized_error
from experiments.strict_json import (
    STRICT_JSON_PROTOCOL,
    strict_json_source_sha256,
)
from experiments.train_quality_router import _paired_group_bootstrap
from experiments.train_joint_nested_fadr import (
    DIAGNOSTICS_FILENAME as JOINT_DIAGNOSTICS_FILENAME,
    LINEAGE_FILENAME as JOINT_LINEAGE_FILENAME,
    SUMMARY_FILENAME as JOINT_SUMMARY_FILENAME,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_source_file,
)
from experiments.verify_reference_conditioned_training import (
    VERIFICATION_PROTOCOL,
    verify_training_pair,
)
from experiments.verify_joint_nested_fadr import verify_joint_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-pairs", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def _optional_float_equal(first: Any, second: Any) -> bool:
    left = finite_float(first)
    right = finite_float(second)
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _metrics(
    errors: np.ndarray,
    successful: np.ndarray,
) -> dict[str, float | int]:
    return {
        "samples": int(errors.size),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "acc_5pct": float(np.mean(errors <= 0.05)),
    }


def _assert_metrics(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} metric key set drifted")
    for name, value in actual.items():
        reference = expected[name]
        if isinstance(value, int):
            if type(reference) is not int or reference != value:
                raise ValueError(f"{label}.{name} mismatch")
        elif not _optional_float_equal(value, reference):
            raise ValueError(f"{label}.{name} mismatch")


def _route(
    *,
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


def _distribution(values: Sequence[float | int]) -> dict[str, float]:
    if len(values) != len(FADR_SEEDS):
        raise ValueError("three values are required for a FADR seed distribution")
    numeric = [float(value) for value in values]
    minimum = min(numeric)
    maximum = max(numeric)
    return {
        "mean": float(statistics.fmean(numeric)),
        "sample_std": float(statistics.stdev(numeric)),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "range": float(maximum - minimum),
    }


def _paired_group_bootstrap_from_seed_averaged_errors(
    candidate_by_seed: Sequence[np.ndarray],
    baseline_by_seed: Sequence[np.ndarray],
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    """Average replicate errors per sample before physical-group resampling."""

    if len(candidate_by_seed) != len(FADR_SEEDS) or len(baseline_by_seed) != len(
        FADR_SEEDS
    ):
        raise ValueError("paired cohort bootstrap requires exactly three seed arrays")
    candidate = np.mean(np.stack(candidate_by_seed, axis=0), axis=0)
    baseline = np.mean(np.stack(baseline_by_seed, axis=0), axis=0)
    if candidate.shape != baseline.shape or candidate.shape != groups.shape:
        raise ValueError("paired cohort bootstrap arrays have different shapes")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("paired cohort bootstrap received non-finite errors")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("paired cohort bootstrap requires at least two physical groups")
    indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected])
        deltas[iteration] = float(
            np.mean(candidate[selected_indices])
            - np.mean(baseline[selected_indices])
        )
    return {
        "delta_nmae": float(np.mean(candidate) - np.mean(baseline)),
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "iterations": iterations,
        "physical_groups": int(len(unique_groups)),
        "replicate_handling": (
            "average the three FADR-seed errors within each sample, then "
            "resample physical groups; seeds are not bootstrap units"
        ),
    }


def _validate_stored_seed_verification(
    stored_path: Path,
    recomputed: Mapping[str, Any],
    *,
    preflight_path: Path,
    expected_seed: int,
) -> dict[str, Any]:
    stored = strict_json_load(stored_path)
    if (
        stored.get("schema_version") != 2
        or stored.get("protocol") != VERIFICATION_PROTOCOL
        or stored.get("status") != "verified"
        or stored.get("seed") != expected_seed
        or stored.get("input_preflight") != str(preflight_path)
        or stored.get("input_preflight_sha256") != sha256_file(preflight_path)
        or stored.get("source_identity")
        != {
            "verifier": sha256_source_file(
                PROJECT_DIR
                / "experiments"
                / "verify_reference_conditioned_training.py"
            )
        }
    ):
        raise ValueError(f"seed {expected_seed}: stored verification audit failed")
    for name, value in recomputed.items():
        if name == "created_utc":
            continue
        if stored.get(name) != value:
            raise ValueError(
                f"seed {expected_seed}: stored verification differs at {name}"
            )
    return stored


def _validate_ablation_source(summary: Mapping[str, Any]) -> None:
    expected = {
        "trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "ablate_fadr_router_features.py"
        ),
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
    if (
        summary.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL
        or summary.get("source_sha256") != expected
    ):
        raise ValueError("FADR feature-ablation source identity drifted")


def _validate_ablation(
    *,
    seed: int,
    seed_root: Path,
    rows: list[dict[str, Any]],
    oof_pairs: Path,
    input_preflight: Path,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    ablation_root = seed_root / "feature_ablation"
    diagnostics_path = ablation_root / DIAGNOSTICS_FILENAME
    summary_path = ablation_root / SUMMARY_FILENAME
    router_diagnostics_path = (
        seed_root / "router" / "model" / "strict_nested_oof_routing.jsonl"
    )
    for path in (diagnostics_path, summary_path, router_diagnostics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = strict_json_load(summary_path)
    diagnostics = strict_jsonl_load(diagnostics_path)
    primary_diagnostics = strict_jsonl_load(router_diagnostics_path)
    calibrator = verification["calibrator"]
    router = verification["router"]
    expected_shared = {
        name: router["training_parameters"][name]
        for name in (
            "trees",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "bootstrap_iterations",
        )
    }
    if (
        summary.get("schema_version") != 1
        or summary.get("protocol") != FADR_ROUTER_FEATURE_ABLATION_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("scope") != "SyncG/train strict nested grouped OOF only"
        or summary.get("seed") != seed
        or summary.get("input") != str(oof_pairs)
        or summary.get("input_sha256") != sha256_file(oof_pairs)
        or summary.get("input_preflight") != str(input_preflight)
        or summary.get("input_preflight_sha256") != sha256_file(input_preflight)
        or summary.get("calibrator_sha256") != calibrator["model_sha256"]
        or summary.get("calibration_diagnostics_sha256")
        != calibrator["diagnostics_sha256"]
        or summary.get("calibration_summary_sha256")
        != calibrator["summary_sha256"]
        or summary.get("diagnostics") != str(diagnostics_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("samples") != len(rows)
        or summary.get("groups")
        != len(set(str(row.get("group_id")) for row in rows))
        or summary.get("folds") != router["training_parameters"]["folds"]
        or summary.get("inner_folds")
        != router["training_parameters"]["inner_folds"]
        or summary.get("shared_hyperparameters") != expected_shared
        or summary.get("group_leakage_count") != 0
        or summary.get("test_samples_used") != 0
        or summary.get("public_samples_used") != 0
        or summary.get("field_samples_used") != 0
        or summary.get("failure_penalty_nmae") != 1.0
        or summary.get("feature_sets")
        != {
            name: list(features)
            for name, features in FADR_ROUTER_FEATURE_SETS.items()
        }
        or set(summary.get("variants") or {}) != set(FADR_ROUTER_FEATURE_SETS)
    ):
        raise ValueError(f"seed {seed}: feature-ablation summary binding failed")
    interpretation = summary.get("interpretation") or {}
    if interpretation.get("without_reference_conditioned_router_evidence") != (
        "router evidence ablation only; the reference-conditioned "
        "candidate reading is unchanged"
    ):
        raise ValueError("no-reference feature ablation interpretation drifted")
    _validate_ablation_source(summary)

    identifiers = [row.get("sample_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("OOF input contains invalid identifiers")
    by_id = {row.get("sample_id"): row for row in diagnostics}
    primary_by_id = {row.get("sample_id"): row for row in primary_diagnostics}
    if (
        len(by_id) != len(diagnostics)
        or len(primary_by_id) != len(primary_diagnostics)
        or set(by_id) != set(identifiers)
        or set(primary_by_id) != set(identifiers)
    ):
        raise ValueError(f"seed {seed}: feature/primary diagnostic IDs drifted")

    groups = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    base_values: list[float | None] = []
    calibrated_values: list[float | None] = []
    hard_values: list[float | None] = []
    variant_values: dict[str, list[float | None]] = {
        name: [] for name in FADR_ROUTER_FEATURE_SETS
    }
    variant_errors: dict[str, list[float]] = {
        name: [] for name in FADR_ROUTER_FEATURE_SETS
    }
    variant_routes: dict[str, list[str]] = {
        name: [] for name in FADR_ROUTER_FEATURE_SETS
    }
    group_folds: dict[str, set[int]] = defaultdict(set)
    thresholds_by_variant_fold: dict[str, dict[int, set[float]]] = {
        name: defaultdict(set) for name in FADR_ROUTER_FEATURE_SETS
    }
    for raw in rows:
        sample_id = raw["sample_id"]
        diagnostic = by_id[sample_id]
        primary = primary_by_id[sample_id]
        if (
            diagnostic.get("group_id") != raw.get("group_id")
            or diagnostic.get("held_out_seed") != raw.get("held_out_seed")
            or diagnostic.get("fadr_seed") != seed
        ):
            raise ValueError(f"seed {seed}, {sample_id}: diagnostic identity mismatch")
        base = finite_float(diagnostic.get("base_prediction"))
        calibrated = finite_float(
            diagnostic.get("reference_conditioned_prediction")
        )
        if not _optional_float_equal(base, primary.get("base_prediction")):
            raise ValueError(f"seed {seed}, {sample_id}: base candidate drifted")
        if not _optional_float_equal(
            calibrated,
            primary.get("reference_conditioned_prediction"),
        ):
            raise ValueError(f"seed {seed}, {sample_id}: calibrated candidate drifted")
        joint = base is not None and calibrated is not None
        fold = diagnostic.get("router_fold")
        if joint:
            if type(fold) is not int or fold < 1:
                raise ValueError(f"seed {seed}, {sample_id}: missing router fold")
            group_folds[str(raw.get("group_id"))].add(fold)
        elif fold is not None:
            raise ValueError(f"seed {seed}, {sample_id}: non-joint row has a fold")
        if fold != primary.get("router_fold"):
            raise ValueError(f"seed {seed}, {sample_id}: full/primary fold mismatch")
        variants = diagnostic.get("variants")
        if not isinstance(variants, Mapping) or tuple(variants) != tuple(
            sorted(FADR_ROUTER_FEATURE_SETS)
        ):
            # JSON is serialized with sorted keys; exact membership is the
            # scientific invariant, while deterministic sorted order is the
            # artifact-format invariant.
            if not isinstance(variants, Mapping) or set(variants) != set(
                FADR_ROUTER_FEATURE_SETS
            ):
                raise ValueError(
                    f"seed {seed}, {sample_id}: feature variants drifted"
                )
        for variant in FADR_ROUTER_FEATURE_SETS:
            variant_row = variants[variant]
            if not isinstance(variant_row, Mapping):
                raise ValueError(f"seed {seed}, {sample_id}: malformed {variant}")
            score = finite_float(variant_row.get("router_score_oof"))
            threshold = finite_float(variant_row.get("nested_threshold"))
            if joint and (score is None or threshold is None):
                raise ValueError(
                    f"seed {seed}, {sample_id}: incomplete nested {variant} score"
                )
            if not joint and (score is not None or threshold is not None):
                raise ValueError(
                    f"seed {seed}, {sample_id}: non-joint {variant} has a score"
                )
            prediction, route = _route(
                base=base,
                calibrated=calibrated,
                score=score,
                threshold=threshold,
            )
            if (
                variant_row.get("route") != route
                or not _optional_float_equal(
                    variant_row.get("prediction"),
                    prediction,
                )
            ):
                raise ValueError(
                    f"seed {seed}, {sample_id}: {variant} route/prediction mismatch"
                )
            error = normalized_error(raw, prediction)
            if not _optional_float_equal(
                variant_row.get("normalized_error"),
                error,
            ):
                raise ValueError(
                    f"seed {seed}, {sample_id}: {variant} error mismatch"
                )
            if joint:
                assert fold is not None
                assert threshold is not None
                thresholds_by_variant_fold[variant][fold].add(threshold)
            variant_values[variant].append(prediction)
            variant_errors[variant].append(error)
            variant_routes[variant].append(route)
        full = variants["full"]
        for name in ("router_score_oof", "nested_threshold", "prediction"):
            if not _optional_float_equal(full.get(name), primary.get(name)):
                primary_name = (
                    "router_score_oof"
                    if name == "router_score_oof"
                    else "nested_threshold"
                    if name == "nested_threshold"
                    else "prediction"
                )
                if not _optional_float_equal(full.get(name), primary.get(primary_name)):
                    raise ValueError(
                        f"seed {seed}, {sample_id}: full ablation differs from primary {name}"
                    )
        if full.get("route") != primary.get("route"):
            raise ValueError(
                f"seed {seed}, {sample_id}: full ablation route differs from primary"
            )
        base_values.append(base)
        calibrated_values.append(calibrated)
        hard_values.append(base if base is not None else calibrated)

    leaking = {
        group: sorted(folds)
        for group, folds in group_folds.items()
        if len(folds) != 1
    }
    if leaking:
        raise ValueError(f"seed {seed}: physical group spans router folds")
    expected_fold_ids = set(
        range(1, int(router["training_parameters"]["folds"]) + 1)
    )
    if set(fold for folds in group_folds.values() for fold in folds) != expected_fold_ids:
        raise ValueError(f"seed {seed}: not all router folds are represented")
    for variant, fold_thresholds in thresholds_by_variant_fold.items():
        if set(fold_thresholds) != expected_fold_ids or any(
            len(values) != 1 for values in fold_thresholds.values()
        ):
            raise ValueError(f"seed {seed}: {variant} threshold/fold drifted")
        expected_thresholds = {
            str(fold): next(iter(values))
            for fold, values in fold_thresholds.items()
        }
        actual_thresholds = summary["variants"][variant]["nested_thresholds"]
        if set(actual_thresholds) != set(expected_thresholds) or any(
            not _optional_float_equal(actual_thresholds[name], value)
            for name, value in expected_thresholds.items()
        ):
            raise ValueError(f"seed {seed}: {variant} summary thresholds drifted")

    base_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, base_values)],
        dtype=np.float64,
    )
    calibrated_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, calibrated_values)
        ],
        dtype=np.float64,
    )
    hard_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, hard_values)],
        dtype=np.float64,
    )
    base_success = np.asarray([value is not None for value in base_values])
    calibrated_success = np.asarray(
        [value is not None for value in calibrated_values]
    )
    hard_success = base_success | calibrated_success
    joint = base_success & calibrated_success
    joint_groups = groups[joint]
    target = np.clip(
        base_error[joint] - calibrated_error[joint],
        -1.0,
        1.0,
    )
    outer_splits = _grouped_splits(
        joint_groups,
        target,
        folds=int(summary["folds"]),
        seed=seed,
        label=f"seed {seed} cohort outer-fold audit",
    )
    expected_fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        outer_splits,
        start=1,
    ):
        train_groups = joint_groups[train_index]
        validation_groups = joint_groups[validation_index]
        inner_splits = _grouped_splits(
            train_groups,
            target[train_index],
            folds=int(summary["inner_folds"]),
            seed=seed + fold * 10_000,
            label=f"seed {seed} cohort inner-fold audit {fold}",
        )
        expected_fold_summaries.append(
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
                            sorted(
                                set(train_groups[inner_validation].tolist())
                            )
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
    if summary.get("fold_summaries") != expected_fold_summaries:
        raise ValueError(f"seed {seed}: outer/inner grouped-fold evidence drifted")
    baseline_metrics = {
        "base_mask": _metrics(base_error, base_success),
        "reference_conditioned_vector": _metrics(
            calibrated_error,
            calibrated_success,
        ),
        "hard_fallback": _metrics(hard_error, hard_success),
    }
    for name, metrics in baseline_metrics.items():
        _assert_metrics(metrics, summary["baselines"][name], label=f"{seed}.{name}")

    variant_arrays: dict[str, np.ndarray] = {}
    for variant in FADR_ROUTER_FEATURE_SETS:
        errors = np.asarray(variant_errors[variant], dtype=np.float64)
        values = variant_values[variant]
        successful = np.asarray([value is not None for value in values])
        metrics = _metrics(errors, successful)
        _assert_metrics(
            metrics,
            summary["variants"][variant]["metrics"],
            label=f"{seed}.{variant}",
        )
        routes = variant_routes[variant]
        switched = np.asarray(
            [route == "calibrated_quality_switch" for route in routes]
        )
        routing = {
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(
                np.sum(switched & (calibrated_error < base_error))
            ),
            "negative_transfers": int(
                np.sum(switched & (calibrated_error > base_error))
            ),
        }
        if routing != summary["variants"][variant]["routing"]:
            raise ValueError(f"seed {seed}: {variant} routing summary drifted")
        iterations = int(
            summary["shared_hyperparameters"]["bootstrap_iterations"]
        )
        expected_comparisons = {
            "router_vs_base_mask": _paired_group_bootstrap(
                errors,
                base_error,
                groups,
                seed=seed,
                iterations=iterations,
            ),
            "router_vs_reference_conditioned_vector": _paired_group_bootstrap(
                errors,
                calibrated_error,
                groups,
                seed=seed + 1,
                iterations=iterations,
            ),
            "router_vs_hard_fallback": _paired_group_bootstrap(
                errors,
                hard_error,
                groups,
                seed=seed + 2,
                iterations=iterations,
            ),
        }
        if variant != "full":
            expected_comparisons["router_vs_full"] = _paired_group_bootstrap(
                errors,
                np.asarray(variant_errors["full"], dtype=np.float64),
                groups,
                seed=seed + 3,
                iterations=iterations,
            )
        if expected_comparisons != summary["variants"][variant][
            "paired_comparisons"
        ]:
            raise ValueError(f"seed {seed}: {variant} paired comparison drifted")
        variant_arrays[variant] = errors
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "metrics": {
            variant: summary["variants"][variant]["metrics"]
            for variant in FADR_ROUTER_FEATURE_SETS
        },
        "routing": {
            variant: summary["variants"][variant]["routing"]
            for variant in FADR_ROUTER_FEATURE_SETS
        },
        "paired_comparisons": {
            variant: summary["variants"][variant]["paired_comparisons"]
            for variant in FADR_ROUTER_FEATURE_SETS
        },
        "errors": variant_arrays,
        "base_error": base_error,
        "calibrated_error": calibrated_error,
        "hard_error": hard_error,
        "successful": {
            variant: np.asarray(
                [value is not None for value in variant_values[variant]]
            )
            for variant in FADR_ROUTER_FEATURE_SETS
        },
        "groups": groups,
    }


def _legacy_build_cohort(
    *,
    oof_pairs: Path,
    input_preflight: Path,
    run_root: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    raise RuntimeError(
        "legacy standalone-calibrator plus independently folded router cohort "
        "is invalid combined FADR evidence"
    )
    oof_pairs = oof_pairs.resolve()
    input_preflight = input_preflight.resolve()
    run_root = run_root.resolve()
    for label, path in {
        "OOF pairs": oof_pairs,
        "input preflight": input_preflight,
        "FADR run root": run_root,
    }.items():
        assert_train_only_path(path, label=label)
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rows = strict_jsonl_load(oof_pairs)
    preflight = strict_json_load(input_preflight)
    if (
        preflight.get("protocol") != FADR_INPUT_PREFLIGHT_PROTOCOL
        or preflight.get("status") != "verified"
        or preflight.get("fadr_seeds") != list(FADR_SEEDS)
        or preflight.get("primary_fadr_seed") != FADR_PRIMARY_SEED
        or preflight.get("oof_protocol") != EXPECTED_OOF_PROTOCOL
        or ((preflight.get("inputs") or {}).get("oof_pairs") or {}).get("path")
        != str(oof_pairs)
        or ((preflight.get("inputs") or {}).get("oof_pairs") or {}).get("sha256")
        != sha256_file(oof_pairs)
    ):
        raise ValueError("FADR cohort input preflight audit failed")

    seed_reports: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    common_training_identity: dict[str, Any] | None = None
    for seed in FADR_SEEDS:
        seed_root = run_root / f"seed_{seed}"
        calibrator_root = seed_root / "calibrator"
        router_root = seed_root / "router"
        verification_path = seed_root / "verification.json"
        recomputed = verify_training_pair(
            oof_pairs=oof_pairs,
            calibrator_root=calibrator_root,
            router_root=router_root,
            expected_seed=seed,
            input_preflight=input_preflight,
            expected_oof_protocol=EXPECTED_OOF_PROTOCOL,
        )
        stored = _validate_stored_seed_verification(
            verification_path,
            recomputed,
            preflight_path=input_preflight,
            expected_seed=seed,
        )
        ablation = _validate_ablation(
            seed=seed,
            seed_root=seed_root,
            rows=rows,
            oof_pairs=oof_pairs,
            input_preflight=input_preflight,
            verification=stored,
        )
        identity = {
            "input_sha256": stored["input_sha256"],
            "input_preflight_sha256": stored["input_preflight_sha256"],
            "input_authorization_sha256": stored["input_authorization_sha256"],
            "samples": stored["samples"],
            "groups": stored["groups"],
            "calibrator_training_parameters": stored["calibrator"][
                "training_parameters"
            ],
            "router_training_parameters": stored["router"]["training_parameters"],
            "calibrator_training_protocol": stored["calibrator"][
                "training_protocol"
            ],
            "router_training_protocol": stored["router"]["training_protocol"],
            "calibrator_source_sha256": stored["calibrator"]["source_sha256"],
            "router_source_sha256": stored["router"]["source_sha256"],
        }
        if common_training_identity is None:
            common_training_identity = identity
        elif identity != common_training_identity:
            raise ValueError("three FADR seeds do not share one training identity")
        seed_reports.append(
            {
                "seed": seed,
                "verification": str(verification_path),
                "verification_sha256": sha256_file(verification_path),
                "calibrator": stored["calibrator"],
                "router": stored["router"],
                "feature_ablation": {
                    name: value
                    for name, value in ablation.items()
                    if name
                    not in {
                        "errors",
                        "base_error",
                        "calibrated_error",
                        "hard_error",
                        "successful",
                        "groups",
                    }
                },
            }
        )
        ablations.append(ablation)

    assert common_training_identity is not None
    if any(
        not np.array_equal(ablation["groups"], ablations[0]["groups"])
        for ablation in ablations[1:]
    ):
        raise ValueError("three FADR seeds have different sample/group ordering")
    groups = ablations[0]["groups"]
    aggregate_variants: dict[str, Any] = {}
    for variant_index, variant in enumerate(FADR_ROUTER_FEATURE_SETS):
        metric_distributions = {
            metric: _distribution(
                [
                    seed_report["feature_ablation"]["metrics"][variant][metric]
                    for seed_report in seed_reports
                ]
            )
            for metric in (
                "nmae",
                "coverage",
                "acc_1pct",
                "acc_2pct",
                "acc_5pct",
            )
        }
        routing_distributions = {
            metric: _distribution(
                [
                    seed_report["feature_ablation"]["routing"][variant][metric]
                    for seed_report in seed_reports
                ]
            )
            for metric in (
                "quality_switches",
                "positive_transfers",
                "negative_transfers",
            )
        }
        mean_errors = np.mean(
            np.stack([ablation["errors"][variant] for ablation in ablations]),
            axis=0,
        )
        aggregate_variants[variant] = {
            "feature_names": list(FADR_ROUTER_FEATURE_SETS[variant]),
            "feature_count": len(FADR_ROUTER_FEATURE_SETS[variant]),
            "per_seed_metric_distribution": metric_distributions,
            "per_seed_routing_distribution": routing_distributions,
            "seed_averaged_sample_error": {
                "nmae": float(np.mean(mean_errors)),
                "acc_1pct": float(np.mean(mean_errors <= 0.01)),
                "acc_2pct": float(np.mean(mean_errors <= 0.02)),
                "acc_5pct": float(np.mean(mean_errors <= 0.05)),
            },
            "paired_group_bootstrap": {
                "router_vs_base_mask": (
                    _paired_group_bootstrap_from_seed_averaged_errors(
                        [ablation["errors"][variant] for ablation in ablations],
                        [ablation["base_error"] for ablation in ablations],
                        groups,
                        seed=bootstrap_seed + variant_index * 10,
                        iterations=bootstrap_iterations,
                    )
                ),
                "router_vs_reference_conditioned_vector": (
                    _paired_group_bootstrap_from_seed_averaged_errors(
                        [ablation["errors"][variant] for ablation in ablations],
                        [
                            ablation["calibrated_error"]
                            for ablation in ablations
                        ],
                        groups,
                        seed=bootstrap_seed + variant_index * 10 + 1,
                        iterations=bootstrap_iterations,
                    )
                ),
                "router_vs_hard_fallback": (
                    _paired_group_bootstrap_from_seed_averaged_errors(
                        [ablation["errors"][variant] for ablation in ablations],
                        [ablation["hard_error"] for ablation in ablations],
                        groups,
                        seed=bootstrap_seed + variant_index * 10 + 2,
                        iterations=bootstrap_iterations,
                    )
                ),
            },
        }
        if variant != "full":
            aggregate_variants[variant]["paired_group_bootstrap"][
                "router_vs_full"
            ] = _paired_group_bootstrap_from_seed_averaged_errors(
                [ablation["errors"][variant] for ablation in ablations],
                [ablation["errors"]["full"] for ablation in ablations],
                groups,
                seed=bootstrap_seed + variant_index * 10 + 3,
                iterations=bootstrap_iterations,
            )

    return {
        "schema_version": 1,
        "protocol": FADR_MULTI_SEED_COHORT_PROTOCOL,
        "status": "verified",
        "scope": "SyncG/train strict nested grouped OOF only",
        "seeds": list(FADR_SEEDS),
        "feature_set_order": list(FADR_ROUTER_FEATURE_SETS),
        "samples": len(rows),
        "groups": len(set(groups.tolist())),
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_preflight": str(input_preflight),
        "input_preflight_sha256": sha256_file(input_preflight),
        "input_authorization_sha256": preflight["input_authorization_sha256"],
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "public_test_field_evaluation_authorized": False,
        "all_three_seeds_complete": True,
        "all_five_feature_sets_complete": True,
        "full_is_preregistered_primary": True,
        "production_selection": {
            "primary_seed": FADR_PRIMARY_SEED,
            "primary_feature_set": "full",
            "rule": (
                "use seed 20260722 with the full feature set for the formal "
                "production/public configuration; seeds 20260723 and 20260724 "
                "are stability replicates only and cannot replace the primary "
                "after inspecting results"
            ),
            "post_hoc_best_seed_selection_allowed": False,
            "seed_ensemble_authorized": False,
            "ensemble_note": (
                "no seed ensemble is preregistered because the input table "
                "already contains one authoritative held-out direction "
                "prediction per physical group; any future deterministic "
                "FADR-seed ensemble requires a separately frozen protocol"
            ),
        },
        "oof_coverage": dict(preflight["rows"]["coverage"]),
        "seed_semantics": {
            "direction_seed_layer": {
                "seeds": preflight["direction_seeds"],
                "role": (
                    "fixed PEPD cross-fitted OOF input: each physical group is "
                    "predicted by one authoritative held-out direction model"
                ),
                "held_out_sample_counts": preflight["rows"][
                    "held_out_seed_counts"
                ],
                "replicate_interpretation": (
                    "not an independent replicate axis in this FADR cohort; "
                    "the three direction seeds are already embedded in one "
                    "fixed cross-fitted OOF table"
                ),
            },
            "fadr_seed_layer": {
                "seeds": list(FADR_SEEDS),
                "role": (
                    "calibrator/router outer-inner grouped split and tree-model "
                    "randomness only; no end-to-end vision model is retrained"
                ),
            },
            "joint_interpretation": (
                "direction seeds and FADR seeds are not multiplied into nine "
                "independent samples; physical groups remain the inferential units"
            ),
        },
        "no_reference_variant_interpretation": (
            "router evidence ablation only; the reference-conditioned "
            "candidate reading remains present"
        ),
        "replicate_statistics": {
            "scalar_summary": "mean, sample SD, minimum, maximum, and range",
            "paired_interval_unit": "physical group",
            "seed_handling": (
                "errors are averaged within each sample across the exact three "
                "FADR calibrator/router seeds before paired physical-group "
                "bootstrap; neither FADR seeds nor the embedded PEPD direction "
                "seeds are treated as independent samples"
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "common_training_identity": common_training_identity,
        "per_seed": seed_reports,
        "feature_ablation_aggregate": aggregate_variants,
        "source_identity": {
            "aggregator": sha256_source_file(Path(__file__).resolve()),
            "single_seed_verifier": sha256_source_file(
                PROJECT_DIR
                / "experiments"
                / "verify_reference_conditioned_training.py"
            ),
            "ablation": sha256_source_file(
                PROJECT_DIR / "experiments" / "ablate_fadr_router_features.py"
            ),
            "feature_sets": sha256_source_file(
                PROJECT_DIR / "experiments" / "fadr_feature_sets.py"
            ),
            "protocol": sha256_source_file(
                PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
            ),
            "strict_json": strict_json_source_sha256(),
        },
        "strict_json_protocol": STRICT_JSON_PROTOCOL,
    }


def build_cohort(
    *,
    oof_pairs: Path,
    input_preflight: Path,
    run_root: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate only verified joint outer-group FADR evidence."""

    oof_pairs = oof_pairs.resolve()
    input_preflight = input_preflight.resolve()
    run_root = run_root.resolve()
    for label, path in {
        "OOF pairs": oof_pairs,
        "input preflight": input_preflight,
        "FADR run root": run_root,
    }.items():
        assert_train_only_path(path, label=label)
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rows = strict_jsonl_load(oof_pairs)
    identifiers = [str(row.get("sample_id")) for row in rows]
    groups = np.asarray(
        [str(row.get("group_id")) for row in rows],
        dtype=object,
    )
    if (
        len(identifiers) != len(set(identifiers))
        or any(not sample_id for sample_id in identifiers)
        or any(not group for group in groups.tolist())
    ):
        raise ValueError("FADR cohort input identities are invalid")
    preflight = strict_json_load(input_preflight)
    if (
        preflight.get("protocol") != FADR_INPUT_PREFLIGHT_PROTOCOL
        or preflight.get("status") != "verified"
        or preflight.get("fadr_seeds") != list(FADR_SEEDS)
        or preflight.get("primary_fadr_seed") != FADR_PRIMARY_SEED
        or preflight.get("oof_protocol") != EXPECTED_OOF_PROTOCOL
        or ((preflight.get("inputs") or {}).get("oof_pairs") or {}).get("path")
        != str(oof_pairs)
        or ((preflight.get("inputs") or {}).get("oof_pairs") or {}).get(
            "sha256"
        )
        != sha256_file(oof_pairs)
    ):
        raise ValueError("FADR cohort input preflight audit failed")

    seed_reports: list[dict[str, Any]] = []
    seed_arrays: list[dict[str, Any]] = []
    common_identity: dict[str, Any] | None = None
    for seed in FADR_SEEDS:
        seed_root = run_root / f"seed_{seed}"
        component_path = seed_root / "component_verification.json"
        joint_root = seed_root / "joint"
        joint_verification_path = seed_root / "joint_verification.json"
        component_recomputed = verify_training_pair(
            oof_pairs=oof_pairs,
            calibrator_root=seed_root / "calibrator",
            router_root=seed_root / "router",
            expected_seed=seed,
            input_preflight=input_preflight,
            expected_oof_protocol=EXPECTED_OOF_PROTOCOL,
        )
        component = _validate_stored_seed_verification(
            component_path,
            component_recomputed,
            preflight_path=input_preflight,
            expected_seed=seed,
        )
        if (
            component.get("combined_fadr_oof_authorized") is not False
            or component.get("joint_outer_group_lineage_present") is not False
            or "component evidence" not in str(component.get("evidence_role"))
        ):
            raise ValueError(
                f"seed {seed}: standalone component evidence role is unsafe"
            )
        joint_recomputed = verify_joint_run(
            oof_pairs=oof_pairs,
            input_preflight=input_preflight,
            joint_root=joint_root,
            expected_seed=seed,
        )
        joint_stored = strict_json_load(joint_verification_path)
        if joint_stored != joint_recomputed:
            raise ValueError(
                f"seed {seed}: stored joint verification differs from recomputation"
            )
        if (
            joint_stored.get("protocol") != FADR_JOINT_VERIFICATION_PROTOCOL
            or joint_stored.get("status") != "verified"
            or joint_stored.get("joint_outer_group_nested") is not True
            or joint_stored.get("combined_fadr_oof_authorized") is not True
            or joint_stored.get("standalone_calibrator_oof_used") is not False
            or joint_stored.get("standalone_router_oof_used") is not False
        ):
            raise ValueError(f"seed {seed}: joint verification is not authorized")
        summary_path = joint_root / JOINT_SUMMARY_FILENAME
        lineage_path = joint_root / JOINT_LINEAGE_FILENAME
        diagnostics_path = joint_root / JOINT_DIAGNOSTICS_FILENAME
        summary = strict_json_load(summary_path)
        lineage = strict_json_load(lineage_path)
        diagnostics = strict_jsonl_load(diagnostics_path)
        diagnostic_by_id = {
            str(row.get("sample_id")): row for row in diagnostics
        }
        if (
            summary.get("protocol") != FADR_JOINT_TRAINING_PROTOCOL
            or lineage.get("protocol") != FADR_JOINT_LINEAGE_PROTOCOL
            or len(diagnostic_by_id) != len(diagnostics)
            or set(diagnostic_by_id) != set(identifiers)
        ):
            raise ValueError(f"seed {seed}: joint artifact identity drifted")
        identity = {
            "input_oof_protocol": joint_stored["input_oof_protocol"],
            "input_sha256": joint_stored["input_sha256"],
            "input_preflight_sha256": joint_stored[
                "input_preflight_sha256"
            ],
            "input_authorization_sha256": joint_stored[
                "input_authorization_sha256"
            ],
            "samples": joint_stored["samples"],
            "groups": joint_stored["groups"],
            "parameters": summary["parameters"],
            "feature_sets": summary["feature_sets"],
            "joint_trainer_source_sha256": summary["source_identity"][
                "joint_trainer"
            ],
            "joint_verifier_source_sha256": joint_stored[
                "source_identity"
            ]["verifier"],
        }
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError(
                "three FADR seeds do not share one joint training identity"
            )
        base_values = [
            finite_float(diagnostic_by_id[sample_id].get("base_prediction"))
            for sample_id in identifiers
        ]
        calibrated_values = [
            finite_float(
                diagnostic_by_id[sample_id].get(
                    "reference_conditioned_prediction"
                )
            )
            for sample_id in identifiers
        ]
        base_error = np.asarray(
            [
                normalized_error(row, value)
                for row, value in zip(rows, base_values)
            ],
            dtype=np.float64,
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
        hard_error = np.asarray(
            [
                normalized_error(row, value)
                for row, value in zip(rows, hard_values)
            ],
            dtype=np.float64,
        )
        errors = {
            variant: np.asarray(
                [
                    float(
                        diagnostic_by_id[sample_id]["variants"][variant][
                            "normalized_error"
                        ]
                    )
                    for sample_id in identifiers
                ],
                dtype=np.float64,
            )
            for variant in FADR_ROUTER_FEATURE_SETS
        }
        successful = {
            variant: np.asarray(
                [
                    finite_float(
                        diagnostic_by_id[sample_id]["variants"][variant][
                            "prediction"
                        ]
                    )
                    is not None
                    for sample_id in identifiers
                ]
            )
            for variant in FADR_ROUTER_FEATURE_SETS
        }
        seed_arrays.append(
            {
                "errors": errors,
                "successful": successful,
                "base_error": base_error,
                "calibrated_error": calibrated_error,
                "hard_error": hard_error,
            }
        )
        seed_reports.append(
            {
                "seed": seed,
                "component_verification": str(component_path),
                "component_verification_sha256": sha256_file(component_path),
                "component_evidence_role": component["evidence_role"],
                "component_combined_fadr_oof_authorized": False,
                "calibrator": component["calibrator"],
                "router": component["router"],
                "joint_verification": str(joint_verification_path),
                "joint_verification_sha256": sha256_file(
                    joint_verification_path
                ),
                "joint_summary": str(summary_path),
                "joint_summary_sha256": sha256_file(summary_path),
                "joint_lineage": str(lineage_path),
                "joint_lineage_sha256": sha256_file(lineage_path),
                "joint_diagnostics": str(diagnostics_path),
                "joint_diagnostics_sha256": sha256_file(diagnostics_path),
                "metrics": joint_stored["metrics"],
                "routing": {
                    variant: summary["variants"][variant]["routing"]
                    for variant in FADR_ROUTER_FEATURE_SETS
                },
                "paired_comparisons": {
                    variant: summary["variants"][variant][
                        "paired_comparisons"
                    ]
                    for variant in FADR_ROUTER_FEATURE_SETS
                },
            }
        )

    assert common_identity is not None
    aggregate_variants: dict[str, Any] = {}
    for variant_index, variant in enumerate(FADR_ROUTER_FEATURE_SETS):
        metric_distributions = {
            metric: _distribution(
                [
                    report["metrics"][variant][metric]
                    for report in seed_reports
                ]
            )
            for metric in (
                "nmae",
                "coverage",
                "acc_1pct",
                "acc_2pct",
                "acc_5pct",
            )
        }
        routing_distributions = {
            metric: _distribution(
                [
                    report["routing"][variant][metric]
                    for report in seed_reports
                ]
            )
            for metric in (
                "quality_switches",
                "positive_transfers",
                "negative_transfers",
            )
        }
        mean_errors = np.mean(
            np.stack(
                [arrays["errors"][variant] for arrays in seed_arrays]
            ),
            axis=0,
        )
        comparisons = {
            "router_vs_base_mask": (
                _paired_group_bootstrap_from_seed_averaged_errors(
                    [arrays["errors"][variant] for arrays in seed_arrays],
                    [arrays["base_error"] for arrays in seed_arrays],
                    groups,
                    seed=bootstrap_seed + variant_index * 10,
                    iterations=bootstrap_iterations,
                )
            ),
            "router_vs_reference_conditioned_vector": (
                _paired_group_bootstrap_from_seed_averaged_errors(
                    [arrays["errors"][variant] for arrays in seed_arrays],
                    [
                        arrays["calibrated_error"]
                        for arrays in seed_arrays
                    ],
                    groups,
                    seed=bootstrap_seed + variant_index * 10 + 1,
                    iterations=bootstrap_iterations,
                )
            ),
            "router_vs_hard_fallback": (
                _paired_group_bootstrap_from_seed_averaged_errors(
                    [arrays["errors"][variant] for arrays in seed_arrays],
                    [arrays["hard_error"] for arrays in seed_arrays],
                    groups,
                    seed=bootstrap_seed + variant_index * 10 + 2,
                    iterations=bootstrap_iterations,
                )
            ),
        }
        if variant != "full":
            comparisons["router_vs_full"] = (
                _paired_group_bootstrap_from_seed_averaged_errors(
                    [arrays["errors"][variant] for arrays in seed_arrays],
                    [arrays["errors"]["full"] for arrays in seed_arrays],
                    groups,
                    seed=bootstrap_seed + variant_index * 10 + 3,
                    iterations=bootstrap_iterations,
                )
            )
        aggregate_variants[variant] = {
            "feature_names": list(FADR_ROUTER_FEATURE_SETS[variant]),
            "feature_count": len(FADR_ROUTER_FEATURE_SETS[variant]),
            "per_seed_metric_distribution": metric_distributions,
            "per_seed_routing_distribution": routing_distributions,
            "seed_averaged_sample_error": {
                "nmae": float(np.mean(mean_errors)),
                "acc_1pct": float(np.mean(mean_errors <= 0.01)),
                "acc_2pct": float(np.mean(mean_errors <= 0.02)),
                "acc_5pct": float(np.mean(mean_errors <= 0.05)),
            },
            "paired_group_bootstrap": comparisons,
        }

    lineage_by_seed = {
        str(report["seed"]): {
            "summary": report["joint_summary"],
            "summary_sha256": report["joint_summary_sha256"],
            "lineage": report["joint_lineage"],
            "lineage_sha256": report["joint_lineage_sha256"],
            "verification": report["joint_verification"],
            "verification_sha256": report[
                "joint_verification_sha256"
            ],
        }
        for report in seed_reports
    }
    return {
        "schema_version": 2,
        "protocol": FADR_MULTI_SEED_COHORT_PROTOCOL,
        "status": "verified",
        "scope": "SyncG/train joint outer-group stacking only",
        "seeds": list(FADR_SEEDS),
        "feature_set_order": list(FADR_ROUTER_FEATURE_SETS),
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
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "public_test_field_evaluation_authorized": False,
        "joint_outer_group_nested": True,
        "legacy_sequential_combined_evidence_authorized": False,
        "standalone_component_oof_role": (
            "component evidence and final full-data fitting only"
        ),
        "all_three_seeds_complete": True,
        "all_five_feature_sets_complete": True,
        "full_is_preregistered_primary": True,
        "production_selection": {
            "primary_seed": FADR_PRIMARY_SEED,
            "primary_feature_set": "full",
            "post_hoc_best_seed_selection_allowed": False,
            "seed_ensemble_authorized": False,
        },
        "oof_coverage": dict(preflight["rows"]["coverage"]),
        "seed_semantics": {
            "direction_seed_layer": {
                "seeds": preflight["direction_seeds"],
                "role": "fixed PEPD cross-fitted OOF input",
                "replicate_interpretation": (
                    "embedded assignment layer, not a FADR replicate axis"
                ),
            },
            "fadr_seed_layer": {
                "seeds": list(FADR_SEEDS),
                "role": (
                    "joint outer folds, calibrator/router tree randomness, "
                    "and within-outer calibration/router folds"
                ),
            },
            "joint_interpretation": (
                "direction and FADR seeds are not multiplied into nine "
                "independent samples; physical groups are inferential units"
            ),
        },
        "replicate_statistics": {
            "scalar_summary": (
                "per-seed value plus mean, sample SD, minimum, maximum, range"
            ),
            "paired_interval_unit": "physical group",
            "seed_handling": (
                "average exact three FADR-seed errors within sample, then "
                "paired-bootstrap physical groups"
            ),
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
        },
        "common_joint_training_identity": common_identity,
        "per_seed": seed_reports,
        "feature_ablation_aggregate": aggregate_variants,
        "udsf_handoff": {
            "protocol": FADR_UDSF_HANDOFF_PROTOCOL,
            "authorized": True,
            "required_joint_training_protocol": FADR_JOINT_TRAINING_PROTOCOL,
            "required_joint_lineage_protocol": FADR_JOINT_LINEAGE_PROTOCOL,
            "required_joint_verification_protocol": (
                FADR_JOINT_VERIFICATION_PROTOCOL
            ),
            "seed_lineage": lineage_by_seed,
            "legacy_component_oof_allowed": False,
            "direct_repartition_of_global_joint_oof_allowed": False,
            "requires_context_specific_refit": True,
            "context_requirement": (
                "each UDSF outer split/context must rerun this joint trainer "
                "so every context-held-out physical group is excluded from "
                "PEPD/base/FADR fits, policies, features, targets, and "
                "thresholds; global joint OOF rows cannot be re-split"
            ),
        },
        "source_identity": {
            "aggregator": sha256_source_file(Path(__file__).resolve()),
            "joint_trainer": sha256_source_file(
                PROJECT_DIR
                / "experiments"
                / "train_joint_nested_fadr.py"
            ),
            "joint_verifier": sha256_source_file(
                PROJECT_DIR
                / "experiments"
                / "verify_joint_nested_fadr.py"
            ),
            "component_verifier": sha256_source_file(
                PROJECT_DIR
                / "experiments"
                / "verify_reference_conditioned_training.py"
            ),
            "feature_sets": sha256_source_file(
                PROJECT_DIR / "experiments" / "fadr_feature_sets.py"
            ),
            "protocol": sha256_source_file(
                PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
            ),
            "strict_json": strict_json_source_sha256(),
        },
        "strict_json_protocol": STRICT_JSON_PROTOCOL,
    }


def _markdown(cohort: Mapping[str, Any]) -> str:
    lines = [
        "# FADR v2 three-seed train-only cohort",
        "",
        f"- Status: `{cohort['status']}`",
        "- Scope: SyncG/train joint outer-group stacking only",
        "- Public/test/field authorization: **false**",
        (
            "- Combined evidence: every outer fold rebuilds calibrator fits, "
            "clip/deadband policy, router features/targets, router fits, and "
            "thresholds without its validation groups."
        ),
        (
            "- Legacy sequential calibrator OOF + independently folded router "
            "OOF: **not authorized** as combined FADR evidence."
        ),
        (
            "- Direction seeds: fixed cross-fitted PEPD OOF assignment; they "
            "are not three additional FADR replicates."
        ),
        (
            "- FADR seeds: calibrator/router grouped splits and tree randomness "
            "only; no vision backbone retraining."
        ),
        (
            "- Replicates: average three seed errors within each sample, then "
            "paired-bootstrap physical groups."
        ),
        (
            "- Formal production/public primary: `seed 20260722 + full`; the "
            "other seeds are stability replicates, not post-hoc alternatives."
        ),
        (
            "- OOF coverage: union of three nominal 10% grouped-validation "
            "splits, not complete K-fold coverage of the training manifest."
        ),
        (
            "- UDSF: global FADR joint OOF cannot be re-split; every UDSF "
            "outer context requires a context-specific joint refit."
        ),
        "",
        "| Variant | Features | NMAE mean ± sample SD | Range | Δ vs hard (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in cohort["feature_set_order"]:
        result = cohort["feature_ablation_aggregate"][variant]
        nmae = result["per_seed_metric_distribution"]["nmae"]
        comparison = result["paired_group_bootstrap"]["router_vs_hard_fallback"]
        interval = comparison["group_bootstrap_95ci"]
        lines.append(
            f"| {variant} | {result['feature_count']} | "
            f"{nmae['mean']:.6f} ± {nmae['sample_std']:.6f} | "
            f"{nmae['range']:.6f} | {comparison['delta_nmae']:+.6f} "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            (
                "`without_reference_conditioned_router_evidence` is a router "
                "evidence ablation only; it does not remove the calibrated "
                "candidate reading."
            ),
            "",
        ]
    )
    return "\n".join(lines)


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
    output = args.output.resolve()
    markdown = output.with_suffix(".md")
    assert_train_only_path(output, label="FADR cohort output")
    for path in (output, markdown):
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite")
    cohort = build_cohort(
        oof_pairs=args.oof_pairs,
        input_preflight=args.input_preflight,
        run_root=args.run_root,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    json_payload = (
        json.dumps(
            cohort,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_payload = _markdown(cohort).encode("utf-8")
    _write_no_clobber(output, json_payload)
    try:
        _write_no_clobber(markdown, markdown_payload)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            cohort,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
