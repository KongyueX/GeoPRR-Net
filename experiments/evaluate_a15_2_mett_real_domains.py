"""Evaluate one frozen A15.2-METT checkpoint on the four real ROI domains.

The evaluator regenerates the established six robustness conditions without
altering the source manifests and reuses the already-produced three-seed
EfficientNet-B0 Raw/SARN-v2 prediction ledgers.  It performs no training,
adaptation, calibration, routing, or model selection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from experiments import evaluate_paper_syncg_only_factorial as factorial
from experiments import evaluate_paper_syncg_only_lightweight_baselines as lightweight
from experiments import score_db_gar18_factorial_3seed as factorial_score
from experiments.evaluate_a15_2_mett_syncg import posterior_batch_diagnostics
from experiments.evaluate_a15_2_syncg_scene_holdout import (
    CONDITIONS,
    PROJECTIVE_CONDITIONS,
    _SharedPixelDataset,
)
from experiments.a15_fteb import frozen_twin_endpoint_forward
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.run_cagh_v5_plain_paper_batch import load_manifest
from experiments.train_a15_2_mett import (
    METT_VARIANTS,
    load_mett_models,
    publication_model_identity,
    validate_mett_variant_metadata,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import _loader


PROTOCOL: Final[str] = "a15_2_mett_real_domain_frozen_evaluation_v1"
DEFAULT_CHECKPOINT: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/"
    "a15_2_mett_seed20262022_epoch5.pt"
)
DEFAULT_PREDICTION_ROOT: Final[Path] = lightweight.DEFAULT_ROOT
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
BOOTSTRAP_SEED: Final[int] = 20260818
FAILURE_ERROR: Final[float] = 1.0
# The external prediction jobs evaluate the six robustness conditions for one
# sample together.  EfficientNet's CUDA FP32 kernels are batch-shape sensitive:
# replaying the same pixels at batch 64 produced a 6.80e-4 endpoint delta on
# field_gauge_roi_test_b, while batch 6 reproduced the ledger within 1.20e-7.
DEFAULT_REAL_DOMAIN_BATCH_SIZE: Final[int] = len(CONDITIONS)
# The same source weights and byte-paired inputs differ by at most 2.36e-4 in
# the established SyncG replay.  A 5e-4 FP32 tolerance leaves headroom for the
# four-domain loaders while still detecting a replaced source state or a
# conversion/preprocessing drift.  Paths, seeds, tensor shapes, and ordinary
# forward tests all remain valid in those concrete failure cases, so this check
# belongs at the formal external-comparison boundary.
ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE: Final[float] = 5.0e-4
ARCHITECTURE: Final[str] = "efficientnet_b0"
SEEDS: Final[tuple[int, ...]] = lightweight.SEEDS
REAL_DATASET_KEYS: Final[tuple[str, ...]] = (
    "field_gauge_roi_test_a",
    "field_gauge_roi_test_b",
    "field_gauge_external_roi",
    "rf100",
)
CANDIDATE_METHODS: Final[tuple[str, ...]] = (
    "raw_anchor",
    "sarn_endpoint",
    "tangent_base",
    "mett",
)
POSTERIOR_METHODS: Final[tuple[str, ...]] = (
    "raw_anchor",
    "sarn_endpoint",
    "mett",
)
INTERVAL_LEVELS: Final[tuple[int, ...]] = (50, 80, 90, 95)


class METTRealDomainEvaluationError(RuntimeError):
    """The frozen real-domain evaluation inputs are inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise METTRealDomainEvaluationError(message)


def full_denominator_metrics(
    errors: Sequence[float],
    passed: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Score every row, with upstream failures already represented by error 1."""

    values = np.asarray(tuple(float(value) for value in errors), dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0, "metric rows are empty")
    _require(
        bool(np.isfinite(values).all())
        and bool((values >= 0.0).all())
        and bool((values <= FAILURE_ERROR).all()),
        "full-denominator errors are invalid",
    )
    if passed is None:
        success = np.ones(values.shape, dtype=np.float64)
    else:
        _require(len(passed) == len(values), "metric pass vector is misaligned")
        success = np.asarray(tuple(bool(value) for value in passed), dtype=np.float64)
    return {
        "rows": int(values.size),
        "coverage": float(success.mean()),
        "nmae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_absolute_error": float(np.quantile(values, 0.95)),
        "p99_absolute_error": float(np.quantile(values, 0.99)),
        "maximum_absolute_error": float(values.max()),
        "acc_at_1_percent": float(np.mean(values <= 0.01)),
        "acc_at_2_percent": float(np.mean(values <= 0.02)),
        "acc_at_5_percent": float(np.mean(values <= 0.05)),
    }


def group_macro_metrics(
    errors: Sequence[float],
    passed: Sequence[bool],
    groups: Sequence[str],
) -> dict[str, Any]:
    _require(
        len(errors) == len(passed) == len(groups) and bool(groups),
        "group-macro vectors are misaligned",
    )
    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_rows.setdefault(str(group), []).append(index)
    _require(len(group_rows) >= 2, "group-macro scoring requires two groups")
    per_group = {
        group: full_denominator_metrics(
            [errors[index] for index in indices],
            [passed[index] for index in indices],
        )
        for group, indices in sorted(group_rows.items())
    }
    metric_names = tuple(next(iter(per_group.values())))
    macro = {
        name: statistics.fmean(float(row[name]) for row in per_group.values())
        for name in metric_names
        if name != "rows"
    }
    return {
        "groups": len(per_group),
        "macro": macro,
        "per_group": per_group,
    }


def paired_group_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    groups: Sequence[str],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    comparator_label: str = "comparator",
) -> dict[str, Any]:
    """Resample complete physical groups and retain their original row counts."""

    _require(replicates >= 1, "bootstrap replicates must be positive")
    _require(
        len(candidate_errors) == len(comparator_errors) == len(groups)
        and bool(groups),
        "paired-bootstrap vectors are misaligned",
    )
    effects = np.asarray(candidate_errors, dtype=np.float64) - np.asarray(
        comparator_errors, dtype=np.float64
    )
    _require(bool(np.isfinite(effects).all()), "paired effects are non-finite")
    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_rows.setdefault(str(group), []).append(index)
    group_ids = sorted(group_rows)
    _require(len(group_ids) >= 2, "paired bootstrap requires two groups")
    group_sums = {group: float(effects[indices].sum()) for group, indices in group_rows.items()}
    group_counts = {group: len(indices) for group, indices in group_rows.items()}
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _replicate in range(int(replicates)):
        numerator = 0.0
        denominator = 0
        for _group in group_ids:
            selected = rng.choice(group_ids)
            numerator += group_sums[selected]
            denominator += group_counts[selected]
        draws.append(numerator / float(denominator))
    low, high = np.quantile(np.asarray(draws), (0.025, 0.975)).tolist()
    point = float(effects.mean())
    return {
        "metric": "full_denominator_nmae",
        "delta_definition": f"mett_minus_{comparator_label}",
        "delta_nmae": point,
        "paired_group_bootstrap_ci95": {"low": float(low), "high": float(high)},
        "candidate_better": point < 0.0,
        "superiority_ci95": float(high) < 0.0,
        "draw_fraction_candidate_better": statistics.fmean(
            float(value < 0.0) for value in draws
        ),
        "rows": len(effects),
        "group_clusters": len(group_ids),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def _record(prediction: float | None, target: float, *, passed: bool) -> dict[str, Any]:
    if not passed:
        return {
            "status": "fail",
            "prediction": None,
            "absolute_error": FAILURE_ERROR,
        }
    _require(prediction is not None, "passing prediction is missing")
    value = float(prediction)
    _require(
        math.isfinite(value) and 0.0 <= value <= 1.0,
        "passing prediction is invalid",
    )
    return {
        "status": "pass",
        "prediction": value,
        "absolute_error": abs(value - float(target)),
    }


def _candidate_vectors(
    rows: Sequence[Mapping[str, Any]], method: str
) -> tuple[list[float], list[bool]]:
    errors = [float(row["candidate"][method]["absolute_error"]) for row in rows]
    passed = [str(row["candidate"][method]["status"]) == "pass" for row in rows]
    return errors, passed


def _baseline_vectors(
    rows: Sequence[Mapping[str, Any]], *, variant: str, seed: int
) -> tuple[list[float], list[bool]]:
    errors = [
        float(row["efficientnet_b0"][variant][str(seed)]["absolute_error"])
        for row in rows
    ]
    passed = [
        str(row["efficientnet_b0"][variant][str(seed)]["status"]) == "pass"
        for row in rows
    ]
    return errors, passed


def _posterior_summary(
    rows: Sequence[Mapping[str, Any]], method: str
) -> dict[str, Any] | None:
    diagnostics = [
        row["candidate"][method].get("posterior")
        for row in rows
        if row["candidate"][method].get("posterior") is not None
    ]
    if not diagnostics:
        return None
    result: dict[str, Any] = {
        "rows": len(diagnostics),
        "mean_crps": statistics.fmean(float(row["crps"]) for row in diagnostics),
        "mean_two_hot_nll": statistics.fmean(
            float(row["two_hot_nll"]) for row in diagnostics
        ),
        "mean_variance": statistics.fmean(
            float(row["variance"]) for row in diagnostics
        ),
    }
    calibration_errors: list[float] = []
    for level in INTERVAL_LEVELS:
        coverage = statistics.fmean(
            float(bool(row[f"coverage_{level}"])) for row in diagnostics
        )
        result[f"empirical_coverage_{level}"] = coverage
        result[f"mean_interval_width_{level}"] = statistics.fmean(
            float(row[f"width_{level}"]) for row in diagnostics
        )
        calibration_errors.append(abs(coverage - level / 100.0))
    result["mean_absolute_interval_calibration_error"] = statistics.fmean(
        calibration_errors
    )
    pit = np.asarray([float(row["pit"]) for row in diagnostics], dtype=np.float64)
    result["pit_cdf_calibration_mae"] = float(
        np.mean(
            [
                abs(float(np.mean(pit <= threshold)) - threshold)
                for threshold in np.linspace(0.1, 0.9, 9)
            ]
        )
    )
    return result


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_seed: int,
    bootstrap_replicates: int,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build six-condition, pooled, macro-group, and paired summaries."""

    _require(bool(rows), "real-domain result rows are empty")
    _require(source_seed in SEEDS, "METT source seed is outside the baseline roster")
    condition_sets: dict[str, tuple[str, ...]] = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": tuple(PROJECTIVE_CONDITIONS),
        "all_conditions": tuple(CONDITIONS),
    }
    summary: dict[str, Any] = {}
    for condition_index, (label, selected) in enumerate(condition_sets.items()):
        subset = [row for row in rows if str(row["condition"]) in selected]
        _require(bool(subset), f"empty condition summary: {label}")
        groups = [str(row["group_id"]) for row in subset]
        candidate: dict[str, Any] = {}
        for method in CANDIDATE_METHODS:
            errors, passed = _candidate_vectors(subset, method)
            candidate[method] = {
                "full_denominator": full_denominator_metrics(errors, passed),
                "group_macro": group_macro_metrics(errors, passed, groups),
            }
            posterior = _posterior_summary(subset, method)
            if posterior is not None:
                candidate[method]["posterior"] = posterior

        baselines: dict[str, Any] = {}
        seed_mean_error_by_variant: dict[str, list[float]] = {}
        for variant in ("raw", "sarn_v2"):
            per_seed: dict[str, Any] = {}
            per_seed_errors: list[list[float]] = []
            per_seed_passed: list[list[bool]] = []
            for seed in SEEDS:
                errors, passed = _baseline_vectors(subset, variant=variant, seed=seed)
                per_seed_errors.append(errors)
                per_seed_passed.append(passed)
                per_seed[str(seed)] = {
                    "full_denominator": full_denominator_metrics(errors, passed),
                    "group_macro": group_macro_metrics(errors, passed, groups),
                }
            metric_names = tuple(per_seed[str(SEEDS[0])]["full_denominator"])
            mean_across_seeds = {
                name: statistics.fmean(
                    float(per_seed[str(seed)]["full_denominator"][name])
                    for seed in SEEDS
                )
                for name in metric_names
            }
            rowwise_seed_mean_errors = [
                statistics.fmean(seed_errors[index] for seed_errors in per_seed_errors)
                for index in range(len(subset))
            ]
            rowwise_all_seed_passed = [
                all(seed_passed[index] for seed_passed in per_seed_passed)
                for index in range(len(subset))
            ]
            seed_mean_error_by_variant[variant] = rowwise_seed_mean_errors
            baselines[variant] = {
                "per_seed": per_seed,
                "mean_across_seeds": mean_across_seeds,
                "rowwise_seed_mean_error": full_denominator_metrics(
                    rowwise_seed_mean_errors, rowwise_all_seed_passed
                ),
                "rowwise_seed_mean_error_group_macro": group_macro_metrics(
                    rowwise_seed_mean_errors,
                    rowwise_all_seed_passed,
                    groups,
                ),
                "rowwise_seed_mean_coverage_semantics": (
                    "a row is covered only when all three seed predictions pass"
                ),
            }

        matched_errors, matched_passed = _baseline_vectors(
            subset, variant="sarn_v2", seed=source_seed
        )
        mett_errors, mett_passed = _candidate_vectors(subset, "mett")
        endpoint_errors, _endpoint_passed = _candidate_vectors(
            subset, "sarn_endpoint"
        )
        comparisons = {
            "matched_same_seed_sarn_v2": paired_group_bootstrap(
                mett_errors,
                matched_errors,
                groups,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + condition_index * 10,
                comparator_label=f"sarn_v2_seed_{source_seed}",
            ),
            "efficientnet_b0_raw_three_seed_mean": paired_group_bootstrap(
                mett_errors,
                seed_mean_error_by_variant["raw"],
                groups,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + condition_index * 10 + 1,
                comparator_label="efficientnet_b0_raw_three_seed_mean",
            ),
            "efficientnet_b0_sarn_v2_three_seed_mean": paired_group_bootstrap(
                mett_errors,
                seed_mean_error_by_variant["sarn_v2"],
                groups,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + condition_index * 10 + 2,
                comparator_label="efficientnet_b0_sarn_v2_three_seed_mean",
            ),
        }
        comparison_fractions = {
            "mett_improvement_fraction_vs_internal_sarn_endpoint": statistics.fmean(
                float(candidate_error < endpoint_error)
                for candidate_error, endpoint_error in zip(
                    mett_errors, endpoint_errors, strict=True
                )
            ),
            "mett_negative_transfer_fraction_vs_internal_sarn_endpoint": statistics.fmean(
                float(candidate_error > endpoint_error)
                for candidate_error, endpoint_error in zip(
                    mett_errors, endpoint_errors, strict=True
                )
            ),
            "mett_improvement_fraction_vs_matched_same_seed_sarn": statistics.fmean(
                float(candidate_error < endpoint_error)
                for candidate_error, endpoint_error in zip(
                    mett_errors, matched_errors, strict=True
                )
            ),
            "mett_negative_transfer_fraction_vs_matched_same_seed_sarn": statistics.fmean(
                float(candidate_error > endpoint_error)
                for candidate_error, endpoint_error in zip(
                    mett_errors, matched_errors, strict=True
                )
            ),
            "mett_coverage": statistics.fmean(float(value) for value in mett_passed),
            "matched_sarn_coverage": statistics.fmean(
                float(value) for value in matched_passed
            ),
        }
        summary[label] = {
            "conditions": list(selected),
            "rows": len(subset),
            "groups": len(set(groups)),
            "candidate": candidate,
            "efficientnet_b0_baselines": baselines,
            "matched_source_seed": source_seed,
            "comparisons": comparisons,
            "comparison_fractions": comparison_fractions,
        }
    return summary


def _load_baseline_runs(
    dataset: factorial.Dataset,
    *,
    prediction_root: Path,
) -> tuple[
    tuple[Any, ...],
    Mapping[str, Any],
    dict[str, dict[int, factorial_score.PredictionRun]],
    dict[str, Any],
]:
    sample_ids = factorial_score.load_validation_ids(dataset.roster)
    _require(
        len(sample_ids) == dataset.expected_samples,
        f"{dataset.slug}: sample count differs",
    )
    targets_tuple = factorial_score._load_targets_with_group_source(
        dataset.labels,
        sample_ids,
        group_source=dataset.group_source,
    )
    _require(
        len({target.group_id for target in targets_tuple}) == dataset.expected_groups,
        f"{dataset.slug}: group count differs",
    )
    targets = {target.sample_id: target for target in targets_tuple}
    manifest_by_id, manifest_audit = factorial_score._load_plain_manifest(
        dataset.manifest,
        sample_ids=sample_ids,
    )
    runs: dict[str, dict[int, factorial_score.PredictionRun]] = {
        "raw": {},
        "sarn_v2": {},
    }
    inputs: dict[str, Any] = {"prediction_root": str(Path(prediction_root).resolve())}
    for variant in runs:
        inputs[variant] = {}
        for seed in SEEDS:
            artifacts = lightweight.prediction_artifacts(
                prediction_root,
                architecture=ARCHITECTURE,
                seed=seed,
                dataset=dataset,
                variant=variant,
            )
            binding: dict[str, str] = {
                "path": str(artifacts["predictions"]),
                "method": lightweight.method_id(
                    ARCHITECTURE, seed, variant=variant
                ),
                "protocol": (
                    lightweight.RAW_PREDICTION_PROTOCOL
                    if variant == "raw"
                    else lightweight.sarn_v2.PROTOCOL
                ),
            }
            if variant == "sarn_v2":
                binding["sidecar"] = str(artifacts["sidecar"])
            run = factorial_score._load_prediction_run(
                binding,
                label=f"{dataset.slug}.{variant}.{seed}",
                base_dir=Path(prediction_root).resolve(),
                require_sidecar=variant == "sarn_v2",
                robustness_seed=factorial.ROBUSTNESS_SEED,
                targets=targets,
                conditions=CONDITIONS,
                manifest_by_id=manifest_by_id,
            )
            runs[variant][seed] = run
            inputs[variant][str(seed)] = {
                "predictions": str(run.path),
                "predictions_sha256": run.sha256,
                "sidecar": str(run.sidecar_path) if run.sidecar_path else None,
                "sidecar_sha256": run.sidecar_sha256,
                "failure_codes": dict(run.failure_codes),
            }
    return targets_tuple, targets, runs, {"manifest": manifest_audit, **inputs}


def _external_record(
    run: factorial_score.PredictionRun,
    *,
    key: tuple[str, str],
    target: float,
) -> dict[str, Any]:
    value = run.rows[key]
    result = _record(
        value.normalized_progress,
        target,
        passed=bool(value.passed),
    )
    if not value.passed:
        result["failure_code"] = value.failure_code
    return result


def _validate_batch_pixel_identity(
    raw_batch: Mapping[str, Any],
    *,
    condition: str,
    runs: Mapping[str, Mapping[int, factorial_score.PredictionRun]],
) -> None:
    """Keep paired external comparisons on exactly the same realized pixels.

    Concrete failure prevented: an OpenCV/SARN implementation or runtime
    change can regenerate different pixels under the same
    ``(sample_id, condition)`` key, making a numerically valid paired test
    compare METT and EfficientNet on different inputs.  Git/version metadata
    identifies code but not the realized image; primary/unique keys, types,
    and transactions only protect row structure; ordinary sampled tests do not
    establish equality for every evaluation row.  The ledgers already contain
    the two pixel digests, so equality is measured here at the point of use.
    """
    ids = tuple(str(value) for value in raw_batch["sample_id"])
    pre_hashes = tuple(str(value) for value in raw_batch["condition_pixel_sha256"])
    post_hashes = tuple(str(value) for value in raw_batch["sarn_pixel_sha256"])
    _require(
        len(ids) == len(pre_hashes) == len(post_hashes),
        "batch pixel identities are misaligned",
    )
    for sample_id, pre_hash, post_hash in zip(
        ids, pre_hashes, post_hashes, strict=True
    ):
        key = (sample_id, condition)
        for variant in ("raw", "sarn_v2"):
            for seed in SEEDS:
                run = runs[variant][seed]
                _require(
                    run.rows[key].condition_pixel_sha256 == pre_hash,
                    f"{sample_id}/{condition}: regenerated input pixels differ",
                )
                if variant == "sarn_v2":
                    _require(run.sidecar_hashes is not None, "SARN sidecar hashes missing")
                    sidecar_pre, sidecar_post = run.sidecar_hashes[key]
                    _require(
                        sidecar_pre == pre_hash and sidecar_post == post_hash,
                        f"{sample_id}/{condition}: regenerated SARN pixels differ",
                    )


def _evaluate_dataset(
    dataset: factorial.Dataset,
    *,
    anchor: torch.nn.Module,
    correction: torch.nn.Module,
    source_seed: int,
    prediction_root: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
    bootstrap_replicates: int,
    include_posterior_diagnostics: bool,
    dataset_index: int,
) -> dict[str, Any]:
    targets_tuple, targets, runs, baseline_inputs = _load_baseline_runs(
        dataset,
        prediction_root=prediction_root,
    )
    target_for_dataset = {
        target.sample_id: (float(target.normalized_target), str(target.group_id))
        for target in targets_tuple
    }
    manifest_rows = load_manifest(dataset.manifest)
    manifest_by_id = {row.sample_id: row for row in manifest_rows}
    _require(
        tuple(manifest_by_id) == tuple(target.sample_id for target in targets_tuple),
        f"{dataset.slug}: model manifest order differs from roster",
    )
    ordered_manifest = tuple(
        manifest_by_id[target.sample_id] for target in targets_tuple
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            shared_dataset = _SharedPixelDataset(
                ordered_manifest,
                target_for_dataset,
                condition=condition,
            )
            loader = _loader(
                shared_dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=BOOTSTRAP_SEED + dataset_index * 100 + condition_index,
                cuda=device.type == "cuda",
            )
            for raw_batch in loader:
                _validate_batch_pixel_identity(raw_batch, condition=condition, runs=runs)
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                groups = tuple(str(value) for value in raw_batch["scene_stem"])
                target_tensor = raw_batch["target"].float()
                try:
                    original = raw_batch["original_view"].to(device)
                    sarn = raw_batch["sarn_view"].to(device)
                    support = raw_batch["sarn_support_mask"].to(device)
                    active = raw_batch["sarn_active"].to(device).bool()
                    homography = raw_batch["raw_to_sarn_homography"].to(device)
                    effective_active = active & (condition != "clean")
                    endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                    prediction = forward_a15_correction(
                        correction,
                        {
                            "raw_posterior": endpoints["raw_posterior"],
                            "sarn_posterior": endpoints["sarn_posterior"],
                            "raw_mean": endpoints["raw_mean"],
                            "sarn_mean": endpoints["sarn_mean"],
                            "raw_features": endpoints["raw_features"],
                            "sarn_features": endpoints["sarn_features"],
                            "sarn_support_mask": support,
                            "sarn_active": effective_active,
                            "raw_to_sarn_homography": homography,
                        },
                        endpoint_null=False,
                    )
                    values = {
                        "raw_anchor": prediction["raw_anchor_mean"].float().cpu(),
                        "sarn_endpoint": prediction["sarn_endpoint_mean"].float().cpu(),
                        "tangent_base": prediction["tangent_base_mean"].float().cpu(),
                        "mett": prediction["mean"].float().cpu(),
                    }
                    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
                    if include_posterior_diagnostics:
                        posterior_values = {
                            "raw_anchor": prediction["raw_anchor_posterior"],
                            "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                            "mett": prediction["progress_posterior"],
                        }
                        diagnostics = {
                            method: posterior_batch_diagnostics(value, target_tensor)
                            for method, value in posterior_values.items()
                        }
                    relation = prediction["relation_available"].bool().cpu()
                    inference_failure: str | None = None
                except Exception as exc:  # full-denominator failure accounting
                    values = {}
                    diagnostics = {}
                    relation = torch.zeros(len(ids), dtype=torch.bool)
                    inference_failure = f"model_exception:{type(exc).__name__}"

                for row_index, (sample_id, group_id) in enumerate(
                    zip(ids, groups, strict=True)
                ):
                    target = float(target_tensor[row_index])
                    expected = float(targets[sample_id].normalized_target)
                    _require(
                        abs(target - expected) <= 1.0e-6,
                        f"{sample_id}: regenerated target differs",
                    )
                    candidate: dict[str, Any] = {}
                    for method in CANDIDATE_METHODS:
                        value = (
                            float(values[method][row_index])
                            if inference_failure is None
                            else None
                        )
                        candidate[method] = _record(
                            value,
                            expected,
                            passed=inference_failure is None,
                        )
                        if inference_failure is not None:
                            candidate[method]["failure_code"] = inference_failure
                        if method in diagnostics:
                            candidate[method]["posterior"] = {
                                name: (
                                    bool(tensor[row_index])
                                    if tensor.dtype == torch.bool
                                    else float(tensor[row_index])
                                )
                                for name, tensor in diagnostics[method].items()
                            }
                    key = (sample_id, condition)
                    external = {
                        variant: {
                            str(seed): _external_record(
                                runs[variant][seed], key=key, target=expected
                            )
                            for seed in SEEDS
                        }
                        for variant in ("raw", "sarn_v2")
                    }
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "condition": condition,
                            "normalized_target": expected,
                            "relation_available": bool(relation[row_index]),
                            "candidate": candidate,
                            "efficientnet_b0": external,
                        }
                    )

    _require(
        len(rows) == dataset.expected_samples * len(CONDITIONS),
        f"{dataset.slug}: Cartesian output count differs",
    )
    summary = summarize_rows(
        rows,
        source_seed=source_seed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=BOOTSTRAP_SEED + dataset_index * 1000,
    )
    raw_deltas: list[float] = []
    sarn_deltas: list[float] = []
    raw_status_mismatches = 0
    sarn_status_mismatches = 0
    for row in rows:
        raw_internal = row["candidate"]["raw_anchor"]
        sarn_internal = row["candidate"]["sarn_endpoint"]
        raw_external = row["efficientnet_b0"]["raw"][str(source_seed)]
        sarn_external = row["efficientnet_b0"]["sarn_v2"][str(source_seed)]
        if raw_internal["status"] != raw_external["status"]:
            raw_status_mismatches += 1
        if sarn_internal["status"] != sarn_external["status"]:
            sarn_status_mismatches += 1
        if raw_internal["status"] == raw_external["status"] == "pass":
            raw_deltas.append(
                abs(float(raw_internal["prediction"]) - float(raw_external["prediction"]))
            )
        if sarn_internal["status"] == sarn_external["status"] == "pass":
            sarn_deltas.append(
                abs(
                    float(sarn_internal["prediction"])
                    - float(sarn_external["prediction"])
                )
            )
    endpoint_audit = {
        "source_seed": source_seed,
        "absolute_prediction_tolerance": ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE,
        "raw_compared_rows": len(raw_deltas),
        "raw_status_mismatch_rows": raw_status_mismatches,
        "raw_max_absolute_prediction_delta": max(raw_deltas, default=None),
        "sarn_compared_rows": len(sarn_deltas),
        "sarn_status_mismatch_rows": sarn_status_mismatches,
        "sarn_max_absolute_prediction_delta": max(sarn_deltas, default=None),
    }
    _require(
        raw_status_mismatches == 0
        and sarn_status_mismatches == 0
        and len(raw_deltas) == len(rows)
        and len(sarn_deltas) == len(rows),
        f"{dataset.slug}: endpoint replay status or compared-row roster differs",
    )
    _require(
        max(raw_deltas) <= ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE
        and max(sarn_deltas) <= ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE,
        (
            f"{dataset.slug}: self-contained anchor replay exceeds the paired external "
            f"tolerance {ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE}; "
            f"raw_max={max(raw_deltas)}, sarn_max={max(sarn_deltas)}"
        ),
    )
    endpoint_audit["within_tolerance"] = True
    return {
        "dataset": {
            "slug": dataset.slug,
            "paper_name": dataset.paper_name,
            "samples": dataset.expected_samples,
            "groups": dataset.expected_groups,
            "group_unit": dataset.group_unit,
            "labels": str(dataset.labels.resolve()),
            "roster": str(dataset.roster.resolve()),
            "manifest": str(dataset.manifest.resolve()),
        },
        "baseline_inputs": baseline_inputs,
        "endpoint_replay_audit": endpoint_audit,
        "summary": summary,
        "per_sample_condition": rows,
    }


def evaluate_real_domains(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dataset_keys: Sequence[str] = REAL_DATASET_KEYS,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = DEFAULT_REAL_DOMAIN_BATCH_SIZE,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    include_posterior_diagnostics: bool = False,
    expected_variant: str = "full",
) -> dict[str, Any]:
    selected = tuple(str(value) for value in dataset_keys)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(REAL_DATASET_KEYS),
        "real-domain dataset selection is invalid",
    )
    _require(workers >= 0 and batch_size >= 1, "loader sizes are invalid")
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"output already exists: {output}")
    checkpoint = Path(checkpoint_path).resolve()
    _require(checkpoint.is_file(), f"METT checkpoint is missing: {checkpoint}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.manual_seed(BOOTSTRAP_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    anchor, correction, model_metadata = load_mett_models(checkpoint, device=device)
    validated_variant = validate_mett_variant_metadata(
        model_metadata, expected_variant=expected_variant
    )
    model_metadata = {
        **model_metadata,
        "validated_mett_variant": validated_variant,
    }
    if expected_variant == "full":
        model_metadata["validated_full_mett"] = validated_variant
    anchor.eval()
    correction.eval()
    source_seed = int(model_metadata["source_anchor"]["source_seed"])
    _require(source_seed in SEEDS, "checkpoint source seed lacks matched baselines")

    datasets: dict[str, Any] = {}
    for dataset_index, key in enumerate(selected):
        dataset = factorial.DATASETS[key]
        result = _evaluate_dataset(
            dataset,
            anchor=anchor,
            correction=correction,
            source_seed=source_seed,
            prediction_root=prediction_root,
            device=device,
            workers=workers,
            batch_size=batch_size,
            bootstrap_replicates=bootstrap_replicates,
            include_posterior_diagnostics=include_posterior_diagnostics,
            dataset_index=dataset_index,
        )
        datasets[key] = result
        all_conditions = result["summary"]["all_conditions"]
        print(
            json.dumps(
                {
                    "dataset": key,
                    "mett_nmae": all_conditions["candidate"]["mett"]
                    ["full_denominator"]["nmae"],
                    "matched_sarn_nmae": all_conditions["efficientnet_b0_baselines"]
                    ["sarn_v2"]["per_seed"][str(source_seed)]["full_denominator"]
                    ["nmae"],
                    "mett_minus_matched_sarn": all_conditions["comparisons"]
                    ["matched_same_seed_sarn_v2"]["delta_nmae"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "frozen_checkpoint": True,
            "training_or_adaptation_during_evaluation": False,
            "model_selection_during_evaluation": False,
            "sample_router_or_prediction_fusion": False,
            "source_manifests_rewritten": False,
            "failure_error": FAILURE_ERROR,
            "inference_precision": "float32",
            "batch_size": batch_size,
            "workers": workers,
            "posterior_diagnostics": bool(include_posterior_diagnostics),
            "experiment_variant": expected_variant,
        },
        "publication_model": publication_model_identity(expected_variant),
        "model": model_metadata,
        "source_seed": source_seed,
        "conditions": list(CONDITIONS),
        "bootstrap": {
            "method": "paired complete-group resampling",
            "replicates": bootstrap_replicates,
            "base_seed": BOOTSTRAP_SEED,
        },
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "complete",
        "output": str(output),
        "source_seed": source_seed,
        "datasets": list(datasets),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=REAL_DATASET_KEYS,
        help="repeat to select domains; defaults to all four real ROI domains",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experiment-variant",
        choices=tuple(METT_VARIANTS),
        default="full",
        help="terminal METT checkpoint identity required for this evaluation",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_REAL_DOMAIN_BATCH_SIZE
    )
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument(
        "--posterior-diagnostics",
        action="store_true",
        help="also retain CRPS, two-hot NLL, PIT, variance, and interval coverage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_real_domains(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        dataset_keys=tuple(args.dataset) if args.dataset else REAL_DATASET_KEYS,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap,
        include_posterior_diagnostics=args.posterior_diagnostics,
        expected_variant=args.experiment_variant,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_CHECKPOINT",
    "PROTOCOL",
    "REAL_DATASET_KEYS",
    "build_argument_parser",
    "evaluate_real_domains",
    "full_denominator_metrics",
    "group_macro_metrics",
    "paired_group_bootstrap",
    "summarize_rows",
]
