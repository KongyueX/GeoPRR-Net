"""Evaluate three A15.2-METT seeds on the clean natural-repeat cohort.

The cohort and stability units are inherited from the paper supplementary
natural-repeat protocol: an eligible unit is an exact
``(physical group_id, ground_truth)`` pair with at least two distinct source
photographs, and multiple materialized ROIs from one source photograph are
averaged before within-unit spread is measured.

For every checkpoint this evaluator runs the fixed Raw anchor, SARN endpoint,
and METT path on every retained image.  It performs no training, adaptation,
prediction-dependent routing, model selection, or cross-seed ensembling.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.a15_fteb import frozen_twin_endpoint_forward
from experiments.evaluate_a15_2_syncg_scene_holdout import _SharedPixelDataset
from experiments.evaluate_paper_syncg_only_factorial import SEEDS
from experiments.evaluate_paper_syncg_only_natural_repeat_stability import (
    DEFAULT_ASSETS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_ROOT,
    RepeatAssets,
    _canonical_cohort_summary,
    _paired_method_grouped_values,
    cluster_bootstrap_mean,
    cohort_manifest_path,
)
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.run_cagh_v5_plain_paper_batch import load_manifest
from experiments.score_natural_repeat_stability import (
    CohortRow,
    PredictionValue,
    load_xm2_repeat_cohort,
    score_one_method,
)
from experiments.train_a15_2_mett import (
    METT_VARIANTS,
    PROTOCOL as METT_TRAINING_PROTOCOL,
    load_mett_models,
    publication_model_identity,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
)


PROTOCOL: Final[str] = "a15_2_mett_natural_repeat_stability_v1"
METHODS: Final[tuple[str, ...]] = (
    "raw_anchor",
    "sarn_endpoint",
    "mett",
)
METHOD_LABELS: Final[dict[str, str]] = {
    "raw_anchor": "Raw anchor",
    "sarn_endpoint": "SARN endpoint",
    "mett": "A15.2-METT",
}
EXPECTED_SEEDS: Final[tuple[int, ...]] = tuple(int(seed) for seed in SEEDS)
FULL_VARIANT: Final[dict[str, bool]] = METT_VARIANTS["full"]
MATCHED_CONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "relation_channels",
    "token_dim",
    "attention_heads",
    "decoder_layers",
    "memory_grid_size",
    "progress_bins",
)


class METTNaturalRepeatError(ValueError):
    """The checkpoint roster, cohort, or predictions cannot be paired."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise METTNaturalRepeatError(message)


def validate_checkpoint_count(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Return three distinct resolved checkpoint paths."""

    checkpoints = tuple(Path(path).resolve() for path in paths)
    _require(len(checkpoints) == 3, "exactly three METT checkpoints are required")
    _require(
        len(set(checkpoints)) == 3,
        "the three METT checkpoint paths must be distinct",
    )
    return checkpoints


def _checkpoint_identity(
    path: Path, *, expected_variant: str = "full"
) -> dict[str, Any]:
    """Read one explicit matched-training identity from a METT artifact."""

    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    expected_flags = METT_VARIANTS[expected_variant]
    source = Path(path).resolve()
    _require(source.is_file(), f"METT checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("protocol") == METT_TRAINING_PROTOCOL,
        f"not an A15.2-METT checkpoint: {source}",
    )
    construction = checkpoint.get("construction")
    _require(
        isinstance(construction, Mapping) and int(checkpoint.get("epochs", -1)) == 5,
        f"METT checkpoint is not a terminal run: {source}",
    )
    variant_source = checkpoint.get("experiment_variant")
    if not isinstance(variant_source, Mapping):
        variant_source = construction
    else:
        for name, default in FULL_VARIANT.items():
            if name in construction:
                _require(
                    bool(variant_source.get(name, default))
                    == bool(construction[name]),
                    f"METT checkpoint variant and construction differ: {source}/{name}",
                )
    variant = {
        name: bool(variant_source.get(name, default))
        for name, default in FULL_VARIANT.items()
    }
    _require(
        variant == expected_flags,
        f"METT checkpoint is not the requested {expected_variant} variant: {source}",
    )
    parameter_counts = checkpoint.get("parameter_counts")
    training_seeds = checkpoint.get("seeds")
    _require(
        all(name in construction for name in MATCHED_CONSTRUCTION_FIELDS)
        and isinstance(parameter_counts, Mapping)
        and all(name in parameter_counts for name in ("anchor", "correction", "trainable"))
        and isinstance(training_seeds, Mapping)
        and all(name in training_seeds for name in ("initialization", "sample_order")),
        f"METT checkpoint matched identity is incomplete: {source}",
    )
    anchor = checkpoint.get("source_anchor")
    _require(
        isinstance(anchor, Mapping),
        f"METT checkpoint source-anchor metadata is missing: {source}",
    )
    seed = anchor.get("source_seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        f"METT checkpoint source seed is missing: {source}",
    )
    return {
        "source_seed": int(seed),
        "variant_label": expected_variant,
        "construction": {
            name: int(construction[name]) for name in MATCHED_CONSTRUCTION_FIELDS
        },
        "parameter_counts": {
            name: int(parameter_counts[name])
            for name in ("anchor", "correction", "trainable")
        },
        "training_seeds": {
            name: int(training_seeds[name])
            for name in ("initialization", "sample_order")
        },
    }


def checkpoint_source_seed(path: Path, *, expected_variant: str = "full") -> int:
    """Read the frozen anchor training seed from one METT variant."""

    return int(
        _checkpoint_identity(path, expected_variant=expected_variant)["source_seed"]
    )


def checkpoint_roster(
    paths: Sequence[Path], *, expected_variant: str = "full"
) -> dict[int, Path]:
    """Map exactly the three matched anchor seeds to their checkpoints."""

    checkpoints = validate_checkpoint_count(paths)
    result: dict[int, Path] = {}
    matched_identity: dict[str, Any] | None = None
    training_seed_pairs: set[tuple[int, int]] = set()
    for path in checkpoints:
        identity = _checkpoint_identity(path, expected_variant=expected_variant)
        seed = int(identity["source_seed"])
        _require(seed not in result, f"duplicate METT anchor seed: {seed}")
        comparison_identity = {
            name: identity[name]
            for name in ("construction", "parameter_counts")
        }
        if matched_identity is None:
            matched_identity = comparison_identity
        else:
            _require(
                comparison_identity == matched_identity,
                f"matched METT checkpoint identity differs: {path}",
            )
        training_pair = (
            int(identity["training_seeds"]["initialization"]),
            int(identity["training_seeds"]["sample_order"]),
        )
        _require(training_pair not in training_seed_pairs, "METT training seed pair repeats")
        training_seed_pairs.add(training_pair)
        result[seed] = path
    _require(
        set(result) == set(EXPECTED_SEEDS),
        f"METT anchor seed roster differs: {sorted(result)}",
    )
    return result


def _seed_summary(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    _require(bool(numbers), "three-seed summary is empty")
    _require(
        all(math.isfinite(value) for value in numbers),
        "three-seed summary contains a non-finite value",
    )
    return {
        "mean": float(statistics.fmean(numbers)),
        "sample_sd": (
            float(statistics.stdev(numbers)) if len(numbers) >= 2 else 0.0
        ),
    }


def _validate_predictions(
    cohort: Sequence[CohortRow],
    predictions: Mapping[
        str, Mapping[int, Mapping[str, PredictionValue]]
    ],
    *,
    seeds: Sequence[int],
) -> None:
    cohort_ids = {row.sample_id for row in cohort}
    _require(bool(cohort_ids), "natural-repeat cohort is empty")
    _require(set(predictions) == set(METHODS), "prediction method roster differs")
    expected_seeds = {int(seed) for seed in seeds}
    for method in METHODS:
        _require(
            set(predictions[method]) == expected_seeds,
            f"prediction seed roster differs for {method}",
        )
        for seed in seeds:
            _require(
                set(predictions[method][int(seed)]) == cohort_ids,
                f"prediction sample roster differs for {method}/{seed}",
            )


def summarize_predictions(
    cohort: Sequence[CohortRow],
    predictions: Mapping[
        str, Mapping[int, Mapping[str, PredictionValue]]
    ],
    *,
    seeds: Sequence[int] = EXPECTED_SEEDS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    candidate_label: str = METHOD_LABELS["mett"],
) -> tuple[
    dict[str, Any],
    dict[str, dict[int, dict[tuple[str, str], dict[str, float]]]],
]:
    """Score all methods and build paired physical-group bootstrap effects."""

    selected_seeds = tuple(int(seed) for seed in seeds)
    _require(
        len(selected_seeds) == 3 and len(set(selected_seeds)) == 3,
        "summary requires exactly three distinct seeds",
    )
    _validate_predictions(cohort, predictions, seeds=selected_seeds)

    method_labels = {**METHOD_LABELS, "mett": str(candidate_label)}
    scores: dict[str, dict[int, dict[str, Any]]] = {}
    units: dict[
        str, dict[int, dict[tuple[str, str], dict[str, float]]]
    ] = {}
    methods: dict[str, Any] = {}
    for method in METHODS:
        scores[method] = {}
        units[method] = {}
        for seed in selected_seeds:
            summary, detail = score_one_method(
                cohort, predictions[method][seed]
            )
            scores[method][seed] = summary
            units[method][seed] = detail
        methods[method] = {
            "label": method_labels[method],
            "per_seed": {
                str(seed): scores[method][seed] for seed in selected_seeds
            },
            "three_seed": {
                "full_denominator_nmae": _seed_summary(
                    [
                        scores[method][seed]["full_denominator_nmae"]
                        for seed in selected_seeds
                    ]
                ),
                "mean_within_unit_prediction_sd_population": _seed_summary(
                    [
                        scores[method][seed][
                            "within_unit_prediction_sd_population"
                        ]["mean"]
                        for seed in selected_seeds
                    ]
                ),
                "mean_within_unit_prediction_range": _seed_summary(
                    [
                        scores[method][seed][
                            "within_unit_prediction_range"
                        ]["mean"]
                        for seed in selected_seeds
                    ]
                ),
            },
        }

    comparisons: dict[str, Any] = {}
    for pair_index, reference in enumerate(("raw_anchor", "sarn_endpoint")):
        differences = _paired_method_grouped_values(
            cohort=cohort,
            candidate_predictions=predictions["mett"],
            reference_predictions=predictions[reference],
            candidate_units=units["mett"],
            reference_units=units[reference],
            seeds=selected_seeds,
        )
        comparison_metrics: dict[str, Any] = {}
        for metric_index, (metric, grouped_values) in enumerate(
            differences.items()
        ):
            comparison_metrics[metric] = cluster_bootstrap_mean(
                grouped_values,
                replicates=bootstrap_replicates,
                seed=(
                    int(bootstrap_seed)
                    + 100
                    + pair_index * 10
                    + metric_index
                ),
            )
        comparisons[f"mett_minus_{reference}"] = {
            "candidate": method_labels["mett"],
            "reference": method_labels[reference],
            "effect_direction": (
                f"{method_labels['mett']} minus {method_labels[reference]}; "
                f"negative favors {method_labels['mett']}"
            ),
            "physical_group_cluster_bootstrap_95ci": comparison_metrics,
        }

    return {
        "methods": methods,
        "paired_comparisons": comparisons,
    }, units


def _prediction_evidence(
    cohort: Sequence[CohortRow],
    predictions: Mapping[
        str, Mapping[int, Mapping[str, PredictionValue]]
    ],
    runtime_evidence: Mapping[int, Mapping[str, Mapping[str, bool]]],
    *,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort_row in cohort:
        seed_rows: dict[str, Any] = {}
        for seed in seeds:
            method_rows: dict[str, Any] = {}
            for method in METHODS:
                value = predictions[method][int(seed)][cohort_row.sample_id]
                method_rows[method] = {
                    "status": "pass" if value.passed else "fail",
                    "prediction": value.value,
                    "absolute_error": (
                        abs(float(value.value) - cohort_row.target)
                        if value.passed and value.value is not None
                        else 1.0
                    ),
                }
            seed_rows[str(seed)] = {
                **runtime_evidence[int(seed)][cohort_row.sample_id],
                "methods": method_rows,
            }
        rows.append(
            {
                "sample_id": cohort_row.sample_id,
                "original_sample_id": cohort_row.original_sample_id,
                "physical_group_id": cohort_row.group_id,
                "ground_truth_key": cohort_row.ground_truth_key,
                "source_image": cohort_row.source_image,
                "normalized_target": cohort_row.target,
                "seeds": seed_rows,
            }
        )
    return rows


def _unit_evidence(
    cohort: Sequence[CohortRow],
    units: Mapping[
        str, Mapping[int, Mapping[tuple[str, str], Mapping[str, float]]]
    ],
    *,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    by_unit: dict[tuple[str, str], list[CohortRow]] = {}
    for row in cohort:
        by_unit.setdefault(row.unit_key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (group_id, truth_key), cohort_rows in sorted(by_unit.items()):
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            per_seed[str(seed)] = {
                method: units[method][int(seed)].get((group_id, truth_key))
                for method in METHODS
            }
        rows.append(
            {
                "physical_group_id": group_id,
                "ground_truth_key": truth_key,
                "sample_ids": sorted(row.sample_id for row in cohort_rows),
                "distinct_source_images": len(
                    {row.source_image for row in cohort_rows}
                ),
                "same_source_image_rule": (
                    "average predictions before within-unit spread"
                ),
                "seeds": per_seed,
            }
        )
    return rows


def _run_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    dataset: _SharedPixelDataset,
    device: torch.device,
    workers: int,
    batch_size: int,
    use_amp: bool,
) -> tuple[
    dict[str, dict[str, PredictionValue]],
    dict[str, dict[str, bool]],
    dict[str, Any],
]:
    anchor, correction, metadata = load_mett_models(checkpoint, device=device)
    observed_seed = int(metadata["source_anchor"]["source_seed"])
    _require(observed_seed == int(seed), "loaded METT source seed differs")
    anchor.eval()
    correction.eval()
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        seed=20260818 + int(seed),
        cuda=device.type == "cuda",
    )
    predictions: dict[str, dict[str, PredictionValue]] = {
        method: {} for method in METHODS
    }
    evidence: dict[str, dict[str, bool]] = {}
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode():
        for batch in loader:
            original = batch["original_view"].to(
                device, non_blocking=device.type == "cuda"
            )
            sarn = batch["sarn_view"].to(
                device, non_blocking=device.type == "cuda"
            )
            support = batch["sarn_support_mask"].to(
                device, non_blocking=device.type == "cuda"
            )
            sarn_active = batch["sarn_active"].to(
                device, non_blocking=device.type == "cuda"
            ).bool()
            homography = batch["raw_to_sarn_homography"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                output = forward_a15_correction(
                    correction,
                    {
                        "raw_posterior": endpoints["raw_posterior"],
                        "sarn_posterior": endpoints["sarn_posterior"],
                        "raw_mean": endpoints["raw_mean"],
                        "sarn_mean": endpoints["sarn_mean"],
                        "raw_features": endpoints["raw_features"],
                        "sarn_features": endpoints["sarn_features"],
                        "sarn_support_mask": support,
                        "sarn_active": sarn_active,
                        "raw_to_sarn_homography": homography,
                    },
                    endpoint_null=False,
                )
            values = {
                "raw_anchor": output["raw_anchor_mean"].detach().float().cpu(),
                "sarn_endpoint": (
                    output["sarn_endpoint_mean"].detach().float().cpu()
                ),
                "mett": output["mean"].detach().float().cpu(),
            }
            active_cpu = sarn_active.detach().cpu()
            relation_cpu = output["relation_available"].detach().bool().cpu()
            sample_ids = tuple(str(value) for value in batch["sample_id"])
            for row_index, sample_id in enumerate(sample_ids):
                _require(sample_id not in evidence, f"duplicate sample: {sample_id}")
                evidence[sample_id] = {
                    "sarn_applied": bool(active_cpu[row_index]),
                    "relation_available": bool(relation_cpu[row_index]),
                }
                for method, tensor in values.items():
                    value = float(tensor[row_index])
                    _require(
                        math.isfinite(value) and 0.0 <= value <= 1.0,
                        f"invalid {method} prediction: {seed}/{sample_id}",
                    )
                    predictions[method][sample_id] = PredictionValue(True, value)
    _require(
        len(evidence) == len(dataset),
        f"METT prediction count differs for seed {seed}",
    )
    return predictions, evidence, metadata


def evaluate_natural_repeat_stability(
    *,
    checkpoints: Sequence[Path],
    output_path: Path,
    manifest_path: Path | None = None,
    assets: RepeatAssets = DEFAULT_ASSETS,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_variant: str = "full",
) -> dict[str, Any]:
    """Run and score three matched METT checkpoints on the clean cohort."""

    _require(workers >= 0 and batch_size >= 1, "invalid loader configuration")
    _require(bootstrap_replicates >= 100, "bootstrap needs at least 100 replicates")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"natural-repeat output exists: {output}")
    started = time.perf_counter()
    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    checkpoints_by_seed = checkpoint_roster(
        checkpoints, expected_variant=expected_variant
    )
    cohort, cohort_summary = load_xm2_repeat_cohort(
        provenance_path=assets.provenance,
        labels_path=assets.labels,
        xm2_manifest_path=assets.physical_capture_manifest,
    )
    manifest = Path(
        manifest_path or cohort_manifest_path(DEFAULT_ROOT)
    ).resolve()
    _require(manifest.is_file(), f"cohort input manifest is missing: {manifest}")
    manifest_rows = load_manifest(manifest)
    manifest_by_id = {row.sample_id: row for row in manifest_rows}
    cohort_ids = {row.sample_id for row in cohort}
    _require(
        len(manifest_by_id) == len(manifest_rows)
        and set(manifest_by_id) == cohort_ids,
        "cohort input manifest and retained cohort rosters differ",
    )
    ordered_manifest = tuple(
        manifest_by_id[row.sample_id] for row in cohort
    )
    targets = {
        row.sample_id: (row.target, row.group_id) for row in cohort
    }
    dataset = _SharedPixelDataset(
        ordered_manifest,
        targets,
        condition="clean",
    )

    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.manual_seed(20260818)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260818)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    predictions: dict[str, dict[int, dict[str, PredictionValue]]] = {
        method: {} for method in METHODS
    }
    runtime_evidence: dict[int, dict[str, dict[str, bool]]] = {}
    model_metadata: dict[str, Any] = {}
    seed_elapsed_seconds: dict[str, float] = {}
    for seed in EXPECTED_SEEDS:
        seed_started = time.perf_counter()
        seed_predictions, evidence, metadata = _run_checkpoint(
            checkpoint=checkpoints_by_seed[seed],
            seed=seed,
            dataset=dataset,
            device=device,
            workers=workers,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        for method in METHODS:
            predictions[method][seed] = seed_predictions[method]
        runtime_evidence[seed] = evidence
        model_metadata[str(seed)] = metadata
        seed_elapsed_seconds[str(seed)] = float(
            time.perf_counter() - seed_started
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    paper_identity = publication_model_identity(expected_variant)
    summary, units = summarize_predictions(
        cohort,
        predictions,
        seeds=EXPECTED_SEEDS,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        candidate_label=paper_identity["display_name"],
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "condition": "clean natural repeated captures",
            "terminal_mett_variant_required": expected_variant,
            "training_or_adaptation_during_evaluation": False,
            "prediction_dependent_routing_or_selection": False,
            "cross_seed_ensemble": False,
            "same_source_image_rule": (
                "average predictions before within-unit spread"
            ),
            "inference_precision": (
                "bfloat16_or_float16_autocast"
                if device.type == "cuda" and use_amp
                else "float32"
            ),
        },
        "publication_model": paper_identity,
        "cohort": _canonical_cohort_summary(cohort_summary),
        "models": model_metadata,
        "summary": summary,
        "per_unit": _unit_evidence(
            cohort, units, seeds=EXPECTED_SEEDS
        ),
        "per_sample": _prediction_evidence(
            cohort,
            predictions,
            runtime_evidence,
            seeds=EXPECTED_SEEDS,
        ),
        "inputs": {
            "manifest": str(manifest),
            "provenance": str(Path(assets.provenance).resolve()),
            "labels": str(Path(assets.labels).resolve()),
            "physical_capture_manifest": str(
                Path(assets.physical_capture_manifest).resolve()
            ),
            "checkpoints": {
                str(seed): str(checkpoints_by_seed[seed])
                for seed in EXPECTED_SEEDS
            },
        },
        "timing": {
            "per_seed_seconds": seed_elapsed_seconds,
            "total_seconds": float(time.perf_counter() - started),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "output": str(output),
        "retained_samples": len(cohort),
        "retained_units": cohort_summary["retained_exact_reading_units"],
        "summary": summary,
        "timing": result["timing"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Repeat exactly three times, once per matched METT anchor seed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=cohort_manifest_path(DEFAULT_ROOT),
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_ASSETS.provenance)
    parser.add_argument("--labels", type=Path, default=DEFAULT_ASSETS.labels)
    parser.add_argument(
        "--physical-capture-manifest",
        type=Path,
        default=DEFAULT_ASSETS.physical_capture_manifest,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experiment-variant", choices=tuple(METT_VARIANTS), default="full"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    checkpoints = validate_checkpoint_count(args.checkpoint)
    result = evaluate_natural_repeat_stability(
        checkpoints=checkpoints,
        output_path=args.output,
        manifest_path=args.manifest,
        assets=RepeatAssets(
            provenance=args.provenance,
            labels=args.labels,
            physical_capture_manifest=args.physical_capture_manifest,
            source_input_manifest=DEFAULT_ASSETS.source_input_manifest,
        ),
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        expected_variant=args.experiment_variant,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_SEEDS",
    "METHODS",
    "METTNaturalRepeatError",
    "PROTOCOL",
    "build_argument_parser",
    "checkpoint_roster",
    "checkpoint_source_seed",
    "evaluate_natural_repeat_stability",
    "main",
    "summarize_predictions",
    "validate_checkpoint_count",
]
