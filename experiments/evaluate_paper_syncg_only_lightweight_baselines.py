"""Canonical paper evaluation for the SyncG-only lightweight regressors.

The command evaluates MobileNetV3-Large and EfficientNet-B0 behind the same
SARN-v2 preprocessing used by the paper comparison.  Its supplemental Raw
path runs the same frozen checkpoints without SARN and performs a paired
three-seed preprocessing comparison.  It also assembles a single same-protocol
summary with Direct-ResNet18 and PG-SIAM after all four methods have produced
three-seed predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments import evaluate_paper_syncg_only_factorial as factorial_evaluation
from experiments import evaluate_paper_syncg_only_pgsiam as pgsiam_evaluation
from experiments import score_cagh_v5_plain_paper_batch as common_score
from experiments import support_aware_roi_normalization_v2 as sarn_v2
from experiments import syncg_lightweight_regression_baselines as lightweight
from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.summarize_support_normalized_cbam_pilot import (
    CONDITIONS,
    FAILURE_ERROR,
    load_predictions,
    paired_group_bootstrap,
)


DEFAULT_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1")
MODEL_DIRECTORY: Final[str] = "lightweight_baselines"
ARCHITECTURES: Final[tuple[str, ...]] = lightweight.ARCHITECTURES
SEEDS: Final[tuple[int, ...]] = factorial_evaluation.SEEDS
VARIANTS: Final[tuple[str, ...]] = ("raw", "sarn_v2")
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
SARN_NOOP_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
)
SUMMARY_PROTOCOL: Final[str] = "paper_syncg_only_competition_summary_v1"
RAW_PREDICTION_PROTOCOL: Final[str] = (
    "paper_syncg_only_lightweight_raw_prediction_v1"
)
RAW_VS_SARN_PROTOCOL: Final[str] = (
    "paper_syncg_only_lightweight_raw_vs_sarn_v2_paired_comparison_v1"
)


class LightweightPaperEvaluationError(RuntimeError):
    """Invalid lightweight paper evaluation input."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LightweightPaperEvaluationError(message)


def paper_name(architecture: str) -> str:
    _require(architecture in lightweight.BACKBONE_SPECS, "unknown architecture")
    return lightweight.BACKBONE_SPECS[architecture].paper_name


def checkpoint_path(root: Path, *, architecture: str, seed: int) -> Path:
    return (
        Path(root)
        / MODEL_DIRECTORY
        / architecture
        / f"seed_{seed}"
        / "terminal.pt"
    )


def method_id(
    architecture: str,
    seed: int,
    *,
    variant: str = "sarn_v2",
) -> str:
    _require(variant in VARIANTS, f"unknown prediction variant: {variant}")
    base = f"{paper_name(architecture)}_seed_{seed}"
    return base if variant == "raw" else f"SARN-v2+{base}"


def prediction_artifacts(
    root: Path,
    *,
    architecture: str,
    seed: int,
    dataset: factorial_evaluation.Dataset,
    variant: str = "sarn_v2",
) -> dict[str, Path]:
    _require(variant in VARIANTS, f"unknown prediction variant: {variant}")
    base = (
        Path(root)
        / MODEL_DIRECTORY
        / "evaluation"
        / variant
        / dataset.slug
        / architecture
        / f"seed_{seed}"
    )
    artifacts = {"predictions": base / "predictions.jsonl"}
    if variant == "sarn_v2":
        artifacts.update(
            {
                "sidecar": base / "normalization.jsonl",
                "summary": base / "summary.json",
            }
        )
    return artifacts


def _load_raw_predictor(
    checkpoint: Path,
    *,
    architecture: str,
    seed: int,
    device_name: str,
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source_method, predictor = lightweight.load_checkpoint_predictor(
        checkpoint,
        device_name=device_name,
    )
    _require(
        source_method == lightweight.method_id(architecture, seed),
        "lightweight checkpoint architecture or seed mismatch",
    )
    return method_id(architecture, seed, variant="raw"), predictor


def run_raw_prediction(
    *,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    architecture: str,
    seed: int,
    device_name: str,
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    """Run the frozen lightweight regressor on unnormalized degraded pixels."""

    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = factorial_evaluation.load_manifest(manifest)
    method, predictor = _load_raw_predictor(
        checkpoint,
        architecture=architecture,
        seed=seed,
        device_name=device_name,
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = factorial_evaluation.load_canonical_roi(source)
            images: list[np.ndarray] = []
            condition_pixels: list[str] = []
            for condition in selected:
                image, _metadata = factorial_evaluation.robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=factorial_evaluation.ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                images.append(image)
                condition_pixels.append(
                    factorial_evaluation.canonical_roi_pixel_sha256(image)
                )
            try:
                values: list[float | None] = [
                    float(value) for value in predictor(images)
                ]
                _require(len(values) == len(images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None
                        and math.isfinite(value)
                        and 0.0 <= value <= 1.0
                        for value in values
                    ),
                    "prediction outside [0,1]",
                )
                failures: list[str | None] = [None] * len(images)
            except Exception as exc:
                values = [None] * len(images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(images)
            for condition, pixel_identity, progress, failure in zip(
                selected,
                condition_pixels,
                values,
                failures,
                strict=True,
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": RAW_PREDICTION_PROTOCOL,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": factorial_evaluation.ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": pixel_identity,
                }
                _require(
                    set(row) == set(factorial_evaluation.OUTPUT_KEYS),
                    "prediction output schema drift",
                )
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return {
        "status": "complete",
        "variant": "raw",
        "method": method,
        "rows": count,
        "output": str(target),
    }


def run_prediction_job(
    *,
    root: Path,
    architecture: str,
    seed: int,
    dataset: factorial_evaluation.Dataset,
    device_name: str,
    variant: str = "sarn_v2",
) -> Mapping[str, Any]:
    _require(variant in VARIANTS, f"unknown prediction variant: {variant}")
    artifacts = prediction_artifacts(
        root,
        architecture=architecture,
        seed=seed,
        dataset=dataset,
        variant=variant,
    )
    checkpoint = checkpoint_path(
        root,
        architecture=architecture,
        seed=seed,
    )
    if variant == "raw":
        return run_raw_prediction(
            checkpoint=checkpoint,
            manifest=dataset.manifest,
            output=artifacts["predictions"],
            architecture=architecture,
            seed=seed,
            device_name=device_name,
            conditions=CONDITIONS,
        )

    def source_loader(
        checkpoint: Path,
        *,
        device_name: str,
    ) -> tuple[str, str, Callable[[Sequence[np.ndarray]], list[float]]]:
        source_method, predictor = lightweight.load_checkpoint_predictor(
            checkpoint,
            device_name=device_name,
        )
        _require(
            source_method == lightweight.method_id(architecture, seed),
            "lightweight checkpoint architecture or seed mismatch",
        )
        return (
            paper_name(architecture),
            method_id(architecture, seed, variant="sarn_v2"),
            predictor,
        )

    original_loader = sarn_v2.load_sarn_v2_predictor
    sarn_v2.load_sarn_v2_predictor = source_loader
    try:
        return sarn_v2.run_prediction(
            checkpoint_path=checkpoint,
            manifest_path=dataset.manifest,
            output_path=artifacts["predictions"],
            sidecar_path=artifacts["sidecar"],
            summary_path=artifacts["summary"],
            device_name=device_name,
            conditions=CONDITIONS,
        )
    finally:
        sarn_v2.load_sarn_v2_predictor = original_loader


@dataclass(frozen=True, slots=True)
class CompetitionMethod:
    paper_label: str
    seed_methods: tuple[str, ...]
    prediction_paths: tuple[Path, ...]


def competition_methods(
    root: Path,
    *,
    dataset: factorial_evaluation.Dataset,
) -> tuple[CompetitionMethod, ...]:
    factorial_root = Path(root) / "factorial"
    direct_methods = tuple(
        factorial_evaluation.method_id(
            cell="00",
            seed=seed,
            variant="sarn_v2",
        )
        for seed in SEEDS
    )
    direct_paths = tuple(
        factorial_evaluation.prediction_artifacts(
            factorial_root,
            cell="00",
            seed=seed,
            dataset=dataset,
            variant="sarn_v2",
        )["predictions"]
        for seed in SEEDS
    )
    methods: list[CompetitionMethod] = [
        CompetitionMethod("Direct-ResNet18", direct_methods, direct_paths)
    ]
    for architecture in ARCHITECTURES:
        methods.append(
            CompetitionMethod(
                paper_name(architecture),
                tuple(method_id(architecture, seed) for seed in SEEDS),
                tuple(
                    prediction_artifacts(
                        root,
                        architecture=architecture,
                        seed=seed,
                        dataset=dataset,
                    )["predictions"]
                    for seed in SEEDS
                ),
            )
        )
    methods.append(
        CompetitionMethod(
            "PG-SIAM",
            tuple(pgsiam_evaluation.method_id(seed) for seed in SEEDS),
            tuple(
                pgsiam_evaluation.prediction_artifacts(
                    root,
                    seed=seed,
                    dataset=dataset,
                )["predictions"]
                for seed in SEEDS
            ),
        )
    )
    return tuple(methods)


def _metric_mean_and_sd(
    values: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    names = tuple(values[0])
    means = {
        name: statistics.fmean(value[name] for value in values)
        for name in names
    }
    sample_sds = {
        name: statistics.stdev(value[name] for value in values)
        for name in names
    }
    return means, sample_sds


def _error_vectors(
    *,
    prediction_rows: Mapping[str, Mapping[tuple[str, str], tuple[float, bool]]],
    seed_methods: Sequence[str],
    targets: Mapping[str, tuple[float, str]],
    conditions: Sequence[str],
) -> dict[str, Any]:
    aggregate_errors: list[float] = []
    aggregate_pass_fraction: list[float] = []
    groups: list[str] = []
    per_seed_errors: list[list[float]] = [[] for _ in seed_methods]
    per_seed_passed: list[list[bool]] = [[] for _ in seed_methods]
    for sample_id, (target, group_id) in targets.items():
        for condition in conditions:
            values = [
                prediction_rows[method][(sample_id, condition)]
                for method in seed_methods
            ]
            sample_seed_errors: list[float] = []
            sample_seed_passed: list[bool] = []
            for index, (value, seed_passed) in enumerate(values):
                error = abs(value - target) if seed_passed else FAILURE_ERROR
                per_seed_errors[index].append(error)
                per_seed_passed[index].append(seed_passed)
                sample_seed_errors.append(error)
                sample_seed_passed.append(seed_passed)
            aggregate_errors.append(statistics.fmean(sample_seed_errors))
            aggregate_pass_fraction.append(
                statistics.fmean(int(passed) for passed in sample_seed_passed)
            )
            groups.append(group_id)
    return {
        "errors": aggregate_errors,
        "pass_fraction": aggregate_pass_fraction,
        "groups": groups,
        "per_seed_errors": per_seed_errors,
        "per_seed_passed": per_seed_passed,
    }


def _group_bootstrap_seed_mean_ci(
    *,
    per_seed_errors: Sequence[Sequence[float]],
    per_seed_passed: Sequence[Sequence[bool]],
    groups: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap complete groups, scoring each seed before averaging metrics."""

    _require(
        len(per_seed_errors) == len(per_seed_passed) == len(SEEDS),
        "bootstrap seed roster mismatch",
    )
    _require(replicates >= 1 and bool(groups), "bootstrap configuration is invalid")
    for errors, passed in zip(per_seed_errors, per_seed_passed, strict=True):
        _require(
            len(errors) == len(passed) == len(groups),
            "bootstrap seed arrays are misaligned",
        )
    group_ids = sorted(set(groups))
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    row_group_indices = np.asarray(
        [group_index[group_id] for group_id in groups], dtype=np.int64
    )
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {
        name: [] for name in common_score.METRIC_NAMES
    }

    # The original implementation materialized and sorted every resampled row
    # list in Python.  Group multiplicities are sufficient statistics for the
    # same bootstrap draw.  Processing them in NumPy batches preserves the
    # exact ``random.Random`` group-choice stream while avoiding millions of
    # Python-level list operations.
    batch_size = 512
    completed = 0
    seed_error_arrays = [np.asarray(values, dtype=np.float64) for values in per_seed_errors]
    seed_pass_arrays = [np.asarray(values, dtype=np.float64) for values in per_seed_passed]
    seed_orders = [np.argsort(values, kind="stable") for values in seed_error_arrays]
    quantile_probabilities = {
        "p50": 0.50,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    while completed < replicates:
        batch = min(batch_size, replicates - completed)
        choices = np.fromiter(
            (rng.randrange(len(group_ids)) for _ in range(batch * len(group_ids))),
            dtype=np.int64,
            count=batch * len(group_ids),
        ).reshape(batch, len(group_ids))
        group_counts = np.zeros((batch, len(group_ids)), dtype=np.int32)
        np.add.at(
            group_counts,
            (np.repeat(np.arange(batch), len(group_ids)), choices.reshape(-1)),
            1,
        )
        row_weights = group_counts[:, row_group_indices]
        totals = row_weights.sum(axis=1, dtype=np.int64).astype(np.float64)
        batch_metrics = {
            name: np.zeros(batch, dtype=np.float64)
            for name in common_score.METRIC_NAMES
        }
        for errors, passed, order in zip(
            seed_error_arrays, seed_pass_arrays, seed_orders, strict=True
        ):
            batch_metrics["nmae"] += (row_weights @ errors) / totals
            batch_metrics["coverage"] += (row_weights @ passed) / totals
            batch_metrics["acc_at_1pct"] += (
                row_weights @ (errors <= 0.01).astype(np.float64)
            ) / totals
            batch_metrics["acc_at_2pct"] += (
                row_weights @ (errors <= 0.02).astype(np.float64)
            ) / totals
            batch_metrics["acc_at_5pct"] += (
                row_weights @ (errors <= 0.05).astype(np.float64)
            ) / totals

            sorted_errors = errors[order]
            cumulative_weights = np.cumsum(row_weights[:, order], axis=1)
            for name, probability in quantile_probabilities.items():
                positions = (totals - 1.0) * probability
                lower = np.floor(positions).astype(np.int64)
                upper = np.ceil(positions).astype(np.int64)
                lower_index = np.argmax(
                    cumulative_weights > lower[:, np.newaxis], axis=1
                )
                upper_index = np.argmax(
                    cumulative_weights > upper[:, np.newaxis], axis=1
                )
                fraction = positions - lower
                batch_metrics[name] += (
                    sorted_errors[lower_index] * (1.0 - fraction)
                    + sorted_errors[upper_index] * fraction
                )

        divisor = float(len(seed_error_arrays))
        for name in common_score.METRIC_NAMES:
            draws[name].extend((batch_metrics[name] / divisor).tolist())
        completed += batch
    return {
        name: {
            "low": common_score._quantile(values, 0.025),
            "high": common_score._quantile(values, 0.975),
        }
        for name, values in draws.items()
    }


def _paired_group_bootstrap_fast(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired group bootstrap with the same draws as the scalar reference."""

    _require(
        len(candidate_errors) == len(comparator_errors) == len(groups)
        and bool(groups),
        "paired bootstrap arrays are misaligned",
    )
    group_ids = sorted(set(groups))
    _require(len(group_ids) >= 2, "paired bootstrap needs at least two groups")
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    deltas = np.asarray(candidate_errors, dtype=np.float64) - np.asarray(
        comparator_errors, dtype=np.float64
    )
    group_sums = np.zeros(len(group_ids), dtype=np.float64)
    group_sizes = np.zeros(len(group_ids), dtype=np.int64)
    for row_index, group_id in enumerate(groups):
        index = group_index[group_id]
        group_sums[index] += deltas[row_index]
        group_sizes[index] += 1

    rng = random.Random(seed)
    draws: list[float] = []
    batch_size = 4096
    completed = 0
    while completed < replicates:
        batch = min(batch_size, replicates - completed)
        choices = np.fromiter(
            (rng.randrange(len(group_ids)) for _ in range(batch * len(group_ids))),
            dtype=np.int64,
            count=batch * len(group_ids),
        ).reshape(batch, len(group_ids))
        counts = np.zeros((batch, len(group_ids)), dtype=np.int32)
        np.add.at(
            counts,
            (np.repeat(np.arange(batch), len(group_ids)), choices.reshape(-1)),
            1,
        )
        values = (counts @ group_sums) / (counts @ group_sizes)
        draws.extend(values.tolist())
        completed += batch
    return {
        "delta_nmae_candidate_minus_comparator": float(np.mean(deltas)),
        "paired_group_bootstrap_ci95": {
            "low": common_score._quantile(draws, 0.025),
            "high": common_score._quantile(draws, 0.975),
        },
        "groups": len(group_ids),
        "replicates": replicates,
        "seed": seed,
    }


def _score_cell(
    *,
    vectors: Mapping[str, Any],
    seed_methods: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_seed = [
        {
            "seed": int(seed),
            "method": method,
            "metrics": common_score._metrics(errors, passed),
        }
        for seed, method, errors, passed in zip(
            SEEDS,
            seed_methods,
            vectors["per_seed_errors"],
            vectors["per_seed_passed"],
            strict=True,
        )
    ]
    metric_mean, metric_sd = _metric_mean_and_sd(
        [row["metrics"] for row in per_seed]
    )
    return {
        "evaluation_rows": len(vectors["errors"]),
        "groups": len(set(vectors["groups"])),
        "aggregation": (
            "score each seed with failure error 1.0, then average metrics across "
            "the three training seeds"
        ),
        "metrics": metric_mean,
        "paired_group_bootstrap_ci95": _group_bootstrap_seed_mean_ci(
            per_seed_errors=vectors["per_seed_errors"],
            per_seed_passed=vectors["per_seed_passed"],
            groups=vectors["groups"],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "per_seed": per_seed,
        "metric_mean_across_seeds": metric_mean,
        "metric_sample_sd_across_seeds": metric_sd,
    }


def _raw_sarn_paired_effect(
    *,
    raw_errors: Sequence[float],
    sarn_errors: Sequence[float],
    groups: Sequence[str],
    group_unit: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    effect = _paired_group_bootstrap_fast(
        sarn_errors,
        raw_errors,
        groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    point = float(effect["delta_nmae_candidate_minus_comparator"])
    ci = effect["paired_group_bootstrap_ci95"]
    return {
        "metric": "full_denominator_nmae",
        "lower_is_better": True,
        "delta_definition": "sarn_v2_minus_raw",
        "failure_error": FAILURE_ERROR,
        "delta_nmae_sarn_v2_minus_raw": point,
        "paired_complete_group_bootstrap_ci95": ci,
        "evaluation_rows": len(raw_errors),
        "group_clusters": int(effect["groups"]),
        "group_unit": group_unit,
        "replicates": int(effect["replicates"]),
        "seed": int(effect["seed"]),
        "sarn_v2_better": point < 0.0,
        "superiority_ci95": float(ci["high"]) < 0.0,
    }


def compare_raw_sarn_dataset(
    *,
    root: Path,
    architecture: str,
    dataset: factorial_evaluation.Dataset,
    bootstrap_replicates: int,
    architecture_index: int,
    dataset_index: int,
) -> dict[str, Any]:
    """Audit and score Raw versus existing SARN-v2 predictions for one dataset."""

    _require(architecture in ARCHITECTURES, "unknown lightweight architecture")
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    sample_ids = factorial_evaluation.factorial_score.load_validation_ids(
        dataset.roster
    )
    _require(
        len(sample_ids) == dataset.expected_samples,
        f"{dataset.paper_name}: sample count drift",
    )
    targets_tuple = (
        factorial_evaluation.factorial_score._load_targets_with_group_source(
            dataset.labels,
            sample_ids,
            group_source=dataset.group_source,
        )
    )
    targets = {target.sample_id: target for target in targets_tuple}
    _require(
        len({target.group_id for target in targets_tuple})
        == dataset.expected_groups,
        f"{dataset.paper_name}: group count drift",
    )
    manifest_by_id, _manifest_audit = (
        factorial_evaluation.factorial_score._load_plain_manifest(
            dataset.manifest,
            sample_ids=sample_ids,
        )
    )

    raw_runs: dict[int, Any] = {}
    sarn_runs: dict[int, Any] = {}
    sarn_actions: dict[int, dict[tuple[str, str], bool]] = {}
    input_files: dict[str, Any] = {}
    for seed in SEEDS:
        raw_artifacts = prediction_artifacts(
            root,
            architecture=architecture,
            seed=seed,
            dataset=dataset,
            variant="raw",
        )
        sarn_artifacts = prediction_artifacts(
            root,
            architecture=architecture,
            seed=seed,
            dataset=dataset,
            variant="sarn_v2",
        )
        raw_runs[seed] = factorial_evaluation.factorial_score._load_prediction_run(
            {
                "path": str(raw_artifacts["predictions"]),
                "method": method_id(architecture, seed, variant="raw"),
                "protocol": RAW_PREDICTION_PROTOCOL,
            },
            label=f"{paper_name(architecture)}.{dataset.paper_name}.raw.{seed}",
            base_dir=Path(root).resolve(),
            require_sidecar=False,
            robustness_seed=factorial_evaluation.ROBUSTNESS_SEED,
            targets=targets,
            conditions=CONDITIONS,
            manifest_by_id=manifest_by_id,
        )
        sarn_runs[seed] = factorial_evaluation.factorial_score._load_prediction_run(
            {
                "path": str(sarn_artifacts["predictions"]),
                "method": method_id(architecture, seed, variant="sarn_v2"),
                "protocol": sarn_v2.PROTOCOL,
                "sidecar": str(sarn_artifacts["sidecar"]),
            },
            label=f"{paper_name(architecture)}.{dataset.paper_name}.sarn_v2.{seed}",
            base_dir=Path(root).resolve(),
            require_sidecar=True,
            robustness_seed=factorial_evaluation.ROBUSTNESS_SEED,
            targets=targets,
            conditions=CONDITIONS,
            manifest_by_id=manifest_by_id,
        )
        sarn_actions[seed] = factorial_evaluation._load_sarn_actions(
            sarn_artifacts["sidecar"],
            sample_ids=sample_ids,
        )
        input_files[str(seed)] = {
            "raw_predictions": str(raw_artifacts["predictions"].resolve()),
            "sarn_v2_predictions": str(sarn_artifacts["predictions"].resolve()),
            "sarn_v2_sidecar": str(sarn_artifacts["sidecar"].resolve()),
        }

    paired_rows = 0
    noop_rows = 0
    projective_changed_rows = 0
    projective_applied_rows = 0
    for sample_id in sample_ids:
        for condition in CONDITIONS:
            raw_hashes: set[str] = set()
            sarn_pre_hashes: set[str] = set()
            sarn_post_hashes: set[str] = set()
            for seed in SEEDS:
                raw_value = raw_runs[seed].rows[(sample_id, condition)]
                sarn_value = sarn_runs[seed].rows[(sample_id, condition)]
                _require(
                    raw_value.condition_pixel_sha256
                    == sarn_value.condition_pixel_sha256,
                    f"{paper_name(architecture)}/{dataset.paper_name}/{sample_id}/"
                    f"{condition}/{seed}: Raw and SARN-v2 input pixels differ",
                )
                _require(
                    sarn_runs[seed].sidecar_hashes is not None,
                    "SARN-v2 sidecar hashes are missing",
                )
                pre_hash, post_hash = sarn_runs[seed].sidecar_hashes[
                    (sample_id, condition)
                ]
                _require(
                    pre_hash == raw_value.condition_pixel_sha256,
                    f"{paper_name(architecture)}/{dataset.paper_name}/{sample_id}/"
                    f"{condition}/{seed}: SARN-v2 pre-normalization pixels differ "
                    "from Raw",
                )
                raw_hashes.add(raw_value.condition_pixel_sha256)
                sarn_pre_hashes.add(pre_hash)
                sarn_post_hashes.add(post_hash)
                applied = sarn_actions[seed][(sample_id, condition)]
                if condition in SARN_NOOP_CONDITIONS:
                    _require(
                        not applied and pre_hash == post_hash,
                        f"{paper_name(architecture)}/{dataset.paper_name}/"
                        f"{sample_id}/{condition}/{seed}: SARN-v2 was not a "
                        "strict no-op",
                    )
                    _require(
                        raw_value.passed == sarn_value.passed
                        and raw_value.normalized_progress
                        == sarn_value.normalized_progress
                        and raw_value.failure_code == sarn_value.failure_code,
                        f"{paper_name(architecture)}/{dataset.paper_name}/"
                        f"{sample_id}/{condition}/{seed}: no-op Raw/SARN-v2 "
                        "prediction differs",
                    )
                    noop_rows += 1
                else:
                    projective_changed_rows += int(pre_hash != post_hash)
                    projective_applied_rows += int(applied)
                paired_rows += 1
            _require(
                len(raw_hashes) == len(sarn_pre_hashes) == 1,
                f"{paper_name(architecture)}/{dataset.paper_name}/{sample_id}/"
                f"{condition}: degraded input pixels differ across training seeds",
            )
            _require(
                len(sarn_post_hashes) == 1,
                f"{paper_name(architecture)}/{dataset.paper_name}/{sample_id}/"
                f"{condition}: SARN-v2 output pixels differ across training seeds",
            )

    condition_sets = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": PROJECTIVE_CONDITIONS,
    }
    comparisons: dict[str, Any] = {}
    for condition_index, (label, selected_conditions) in enumerate(
        condition_sets.items()
    ):
        raw_vectors = factorial_evaluation._comparison_vectors(
            raw_runs,
            targets=targets_tuple,
            conditions=selected_conditions,
        )
        sarn_vectors = factorial_evaluation._comparison_vectors(
            sarn_runs,
            targets=targets_tuple,
            conditions=selected_conditions,
        )
        _require(
            raw_vectors["groups"] == sarn_vectors["groups"]
            and raw_vectors["sample_condition_keys"]
            == sarn_vectors["sample_condition_keys"],
            f"{paper_name(architecture)}/{dataset.paper_name}/{label}: paired "
            "vectors are misaligned",
        )
        comparisons[label] = {
            "conditions": list(selected_conditions),
            "evaluation_rows": len(raw_vectors["errors"]),
            "groups": dataset.expected_groups,
            "raw": {
                "metrics": raw_vectors["metrics"],
                "per_seed": raw_vectors["per_seed"],
            },
            "sarn_v2": {
                "metrics": sarn_vectors["metrics"],
                "per_seed": sarn_vectors["per_seed"],
            },
            "paired_effect": _raw_sarn_paired_effect(
                raw_errors=raw_vectors["errors"],
                sarn_errors=sarn_vectors["errors"],
                groups=raw_vectors["groups"],
                group_unit=dataset.group_unit,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=(
                    20260817
                    + architecture_index * len(factorial_evaluation.DATASETS)
                    * len(condition_sets)
                    + dataset_index * len(condition_sets)
                    + condition_index
                ),
            ),
        }

    return {
        "architecture": architecture,
        "paper_method": paper_name(architecture),
        "dataset": dataset.paper_name,
        "samples": dataset.expected_samples,
        "groups": dataset.expected_groups,
        "group_unit": dataset.group_unit,
        "group_source": dataset.group_source,
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "audit": {
            "same_sample_condition_seed_pre_normalization_pixels_raw_and_sarn_v2": True,
            "same_degraded_pixels_across_training_seeds": True,
            "same_sarn_v2_output_pixels_across_training_seeds": True,
            "clean_and_blur_sarn_v2_strict_noop": True,
            "clean_and_blur_raw_sarn_v2_predictions_identical": True,
            "paired_seed_sample_condition_rows": paired_rows,
            "clean_and_blur_noop_rows": noop_rows,
            "projective_rows": dataset.expected_samples
            * len(PROJECTIVE_CONDITIONS)
            * len(SEEDS),
            "projective_normalization_applied_rows": projective_applied_rows,
            "projective_changed_pixel_rows": projective_changed_rows,
        },
        "input_files": input_files,
        "comparisons": comparisons,
    }


def compare_preprocessing(
    *,
    root: Path,
    bootstrap_replicates: int,
    output: Path | None = None,
) -> dict[str, Any]:
    """Write the complete two-backbone, five-dataset Raw-vs-SARN result."""

    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    architecture_results: dict[str, Any] = {}
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        datasets = {
            dataset.paper_name: compare_raw_sarn_dataset(
                root=root,
                architecture=architecture,
                dataset=dataset,
                bootstrap_replicates=bootstrap_replicates,
                architecture_index=architecture_index,
                dataset_index=dataset_index,
            )
            for dataset_index, dataset in enumerate(
                factorial_evaluation.DATASETS.values()
            )
        }
        architecture_results[paper_name(architecture)] = {
            "architecture": architecture,
            "raw_method": paper_name(architecture),
            "sarn_v2_method": f"SARN-v2+{paper_name(architecture)}",
            "datasets": datasets,
        }

    payload = {
        "schema_version": 1,
        "protocol": RAW_VS_SARN_PROTOCOL,
        "status": "complete",
        "training_data": "SyncG scene-disjoint fit only",
        "field_data_role": "test only",
        "architectures": architecture_results,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "scoring_policy": {
            "failure_error": FAILURE_ERROR,
            "seed_handling": (
                "compute failure-penalized error independently for each training "
                "seed, then average the three errors within each sample-condition"
            ),
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "unit": "complete dataset-declared group; SyncG uses scene",
                "ci": "two-sided percentile 95%",
                "pairing": (
                    "SARN-v2 minus Raw on identical sample-condition inputs; "
                    "training seeds are paired before complete-group resampling"
                ),
            },
        },
    }
    target = (
        Path(output)
        if output is not None
        else Path(root)
        / MODEL_DIRECTORY
        / "evaluation"
        / "results"
        / "raw_vs_sarn_v2.json"
    ).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_json_bytes(payload))
    return {
        "status": "complete",
        "output": str(target),
        "architectures": list(architecture_results),
        "datasets": [
            dataset.paper_name for dataset in factorial_evaluation.DATASETS.values()
        ],
        "bootstrap_replicates": bootstrap_replicates,
    }


def summarize_dataset(
    *,
    root: Path,
    dataset: factorial_evaluation.Dataset,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    targets = pgsiam_evaluation._targets(dataset)
    roster = competition_methods(root, dataset=dataset)
    prediction_paths = tuple(
        path for method in roster for path in method.prediction_paths
    )
    prediction_rows, bindings = load_predictions(
        prediction_paths,
        sample_ids=set(targets),
    )
    expected_methods = {
        method for entry in roster for method in entry.seed_methods
    }
    _require(set(prediction_rows) == expected_methods, "competition method roster mismatch")

    method_vectors: dict[str, dict[str, dict[str, Any]]] = {}
    method_results: dict[str, dict[str, Any]] = {}
    condition_sets = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": PROJECTIVE_CONDITIONS,
    }
    for method_index, entry in enumerate(roster):
        method_vectors[entry.paper_label] = {}
        method_results[entry.paper_label] = {
            "paper_label": entry.paper_label,
            "seed_methods": list(entry.seed_methods),
            "conditions": {},
        }
        for condition_index, (label, conditions) in enumerate(condition_sets.items()):
            vectors = _error_vectors(
                prediction_rows=prediction_rows,
                seed_methods=entry.seed_methods,
                targets=targets,
                conditions=conditions,
            )
            method_vectors[entry.paper_label][label] = vectors
            method_results[entry.paper_label]["conditions"][label] = _score_cell(
                vectors=vectors,
                seed_methods=entry.seed_methods,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=20260815 + method_index * len(condition_sets) + condition_index,
            )

    pg_vectors = method_vectors["PG-SIAM"]
    paired_vs_pg: dict[str, dict[str, Any]] = {}
    for method_index, entry in enumerate(roster):
        if entry.paper_label == "PG-SIAM":
            continue
        paired_vs_pg[entry.paper_label] = {}
        for condition_index, label in enumerate(condition_sets):
            vectors = method_vectors[entry.paper_label][label]
            reference = pg_vectors[label]
            _require(
                vectors["groups"] == reference["groups"],
                "paired competition groups are misaligned",
            )
            paired_vs_pg[entry.paper_label][label] = _paired_group_bootstrap_fast(
                vectors["errors"],
                reference["errors"],
                vectors["groups"],
                replicates=bootstrap_replicates,
                seed=20260915 + method_index * len(condition_sets) + condition_index,
            )

    return {
        "dataset": dataset.paper_name,
        "samples": len(targets),
        "groups": len({group for _target, group in targets.values()}),
        "group_unit": dataset.group_unit,
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "methods": method_results,
        "paired_method_minus_pg_siam": paired_vs_pg,
        "prediction_bindings": bindings,
    }


def score_all(
    *,
    root: Path,
    datasets: Sequence[factorial_evaluation.Dataset],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    results = {
        dataset.paper_name: summarize_dataset(
            root=root,
            dataset=dataset,
            bootstrap_replicates=bootstrap_replicates,
        )
        for dataset in datasets
    }
    output = (
        Path(root)
        / MODEL_DIRECTORY
        / "evaluation"
        / "results"
        / "competition_all_datasets.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "status": "complete",
        "training_data": "SyncG scene-disjoint fit only",
        "field_data_role": "test only",
        "preprocessing": (
            "SARN-v2 for Direct-ResNet18, MobileNetV3-Large, and EfficientNet-B0; "
            "PG-SIAM uses the identical SARN-v2 View-A plus its conservative dual view"
        ),
        "seeds": list(SEEDS),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "unit": "dataset group; SyncG uses scene",
            "seed_handling": (
                "score each training seed first; average failure-penalized per-sample "
                "errors before paired group resampling"
            ),
            "ci": "two-sided percentile 95%",
        },
        "datasets": results,
    }
    output.write_bytes(_canonical_json_bytes(payload))
    return {
        "status": "complete",
        "output": str(output.resolve()),
        "datasets": list(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prediction.add_argument(
        "--variant",
        choices=("raw", "sarn_v2", "both"),
        default="sarn_v2",
    )
    prediction.add_argument("--architecture", action="append", choices=ARCHITECTURES)
    prediction.add_argument(
        "--dataset",
        action="append",
        choices=tuple(factorial_evaluation.DATASETS),
    )
    prediction.add_argument("--seed", action="append", type=int, choices=SEEDS)
    prediction.add_argument("--device", default="cuda:0")

    score = commands.add_parser("score")
    score.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    score.add_argument(
        "--dataset",
        action="append",
        choices=tuple(factorial_evaluation.DATASETS),
    )
    score.add_argument("--bootstrap-replicates", type=int, default=20_000)

    comparison = commands.add_parser("compare-preprocessing")
    comparison.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    comparison.add_argument("--bootstrap-replicates", type=int, default=20_000)
    comparison.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-preprocessing":
        print(
            json.dumps(
                compare_preprocessing(
                    root=args.root,
                    bootstrap_replicates=args.bootstrap_replicates,
                    output=args.output,
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    datasets = (
        tuple(factorial_evaluation.DATASETS[name] for name in args.dataset)
        if args.dataset
        else tuple(factorial_evaluation.DATASETS.values())
    )
    if args.command == "score":
        print(
            json.dumps(
                score_all(
                    root=args.root,
                    datasets=datasets,
                    bootstrap_replicates=args.bootstrap_replicates,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    architectures = tuple(args.architecture) if args.architecture else ARCHITECTURES
    seeds = tuple(args.seed) if args.seed else SEEDS
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    for variant in variants:
        for architecture in architectures:
            for seed in seeds:
                for dataset in datasets:
                    print(
                        json.dumps(
                            {
                                "status": "starting",
                                "variant": variant,
                                "dataset": dataset.paper_name,
                                "method": method_id(
                                    architecture,
                                    seed,
                                    variant=variant,
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    result = run_prediction_job(
                        root=args.root,
                        architecture=architecture,
                        seed=seed,
                        dataset=dataset,
                        device_name=args.device,
                        variant=variant,
                    )
                    print(
                        json.dumps(result, ensure_ascii=False, sort_keys=True),
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_ROOT",
    "MODEL_DIRECTORY",
    "RAW_PREDICTION_PROTOCOL",
    "RAW_VS_SARN_PROTOCOL",
    "SARN_NOOP_CONDITIONS",
    "SUMMARY_PROTOCOL",
    "VARIANTS",
    "checkpoint_path",
    "compare_preprocessing",
    "compare_raw_sarn_dataset",
    "competition_methods",
    "main",
    "method_id",
    "prediction_artifacts",
    "run_raw_prediction",
    "run_prediction_job",
    "score_all",
    "summarize_dataset",
]
