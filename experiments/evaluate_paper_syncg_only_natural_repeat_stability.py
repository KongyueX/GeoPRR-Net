"""Evaluate clean natural repeated-capture stability for the SyncG-only paper models.

This is a supplementary, test-only evaluation.  The repeated-capture cohort is
selected without consulting model outputs: samples are grouped by physical
instrument and exact ground-truth reading, and a unit is retained only when it
contains at least two distinct source photographs.  Multiple materialized ROIs
from one source photograph are averaged before within-unit spread is measured.

The runner evaluates three paper-facing methods and three training seeds:

* ``Direct-ResNet18``
* ``SARN-v2+GeoAttn-ResNet18``
* ``PG-SIAM``

It reports full-denominator NMAE, within-unit population SD and range, training
seed mean/sample-SD, and physical-instrument cluster-bootstrap confidence
intervals.  It performs no training, adaptation, calibration, or model
selection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from experiments import evaluate_projective_geometry_guided_siam as pg_evaluation
from experiments import train_projective_geometry_guided_siam_syncg as pg_training
from experiments.evaluate_paper_syncg_only_factorial import (
    SEEDS,
    checkpoint_path as factorial_checkpoint_path,
    run_sarn_prediction,
)
from experiments.evaluate_paper_syncg_only_pgsiam import (
    checkpoint_path as pgsiam_checkpoint_path,
)
from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.run_cagh_v5_plain_paper_batch import load_manifest
from experiments.score_natural_repeat_stability import (
    CohortRow,
    PredictionValue,
    load_xm2_repeat_cohort,
    score_one_method,
)


PROTOCOL: Final[str] = "paper_syncg_only_natural_repeat_stability_v1"
DEFAULT_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1")
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260815
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


class PaperNaturalRepeatError(RuntimeError):
    """An input cannot support the supplementary repeated-capture analysis."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperNaturalRepeatError(message)


@dataclass(frozen=True, slots=True)
class RepeatAssets:
    provenance: Path
    labels: Path
    physical_capture_manifest: Path
    source_input_manifest: Path


DEFAULT_ASSETS: Final[RepeatAssets] = RepeatAssets(
    provenance=Path("C:/pointer_read/unified_real_photo_progress_v1/provenance.jsonl"),
    labels=Path("C:/pointer_read/unified_real_photo_progress_v1/labels.jsonl"),
    physical_capture_manifest=(
        REPOSITORY_ROOT
        / "artifacts/manifests/field_holdout_xiangmu2_2026.jsonl"
    ),
    source_input_manifest=Path(
        "C:/pointer_read/unified_real_photo_progress_v1/input_manifest.jsonl"
    ),
)


@dataclass(frozen=True, slots=True)
class MethodSpec:
    slug: str
    paper_label: str


METHODS: Final[dict[str, MethodSpec]] = {
    "direct_resnet18": MethodSpec("direct_resnet18", "Direct-ResNet18"),
    "geoattn_sarn": MethodSpec(
        "geoattn_sarn", "SARN-v2+GeoAttn-ResNet18"
    ),
    "pgsiam": MethodSpec("pgsiam", "PG-SIAM"),
}


def prediction_method_id(method: str, seed: int) -> str:
    """Return the method identity written by the selected prediction pipeline."""

    _require(method in METHODS, f"unknown method: {method}")
    label = METHODS[method].paper_label
    if method == "direct_resnet18":
        label = f"SARN-v2+{label}"
    return f"{label}_seed_{int(seed)}"


def supplementary_root(root: Path) -> Path:
    return Path(root) / "supplementary" / "natural_repeat_stability"


def cohort_manifest_path(root: Path) -> Path:
    return supplementary_root(root) / "cohort" / "input_manifest.jsonl"


def result_path(root: Path) -> Path:
    return supplementary_root(root) / "results" / "clean_stability.json"


def prediction_artifacts(
    root: Path,
    *,
    method: str,
    seed: int,
) -> dict[str, Path]:
    _require(method in METHODS, f"unknown method: {method}")
    base = (
        supplementary_root(root)
        / "predictions"
        / METHODS[method].slug
        / f"seed_{seed}"
    )
    artifacts = {"predictions": base / "predictions.jsonl"}
    if method in {"direct_resnet18", "geoattn_sarn"}:
        artifacts.update(
            {
                "sidecar": base / "normalization.jsonl",
                "summary": base / "summary.json",
            }
        )
    elif method == "pgsiam":
        artifacts.update(
            {
                "diagnostics": base / "diagnostics.jsonl",
                "summary": base / "summary.json",
            }
        )
    return artifacts


def _cohort(
    assets: RepeatAssets,
) -> tuple[tuple[CohortRow, ...], dict[str, Any]]:
    return load_xm2_repeat_cohort(
        provenance_path=assets.provenance,
        labels_path=assets.labels,
        xm2_manifest_path=assets.physical_capture_manifest,
    )


def _canonical_cohort_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Rename legacy source-specific fields for the paper-facing result."""

    return {
        "source_joined_samples": int(summary["joined_xm2_samples"]),
        "source_joined_physical_groups": int(summary["joined_physical_groups"]),
        "source_joined_exact_reading_units": int(
            summary["joined_exact_reading_units"]
        ),
        "retained_samples": int(summary["retained_samples"]),
        "retained_physical_groups": int(summary["retained_physical_groups"]),
        "retained_exact_reading_units": int(
            summary["retained_exact_reading_units"]
        ),
        "retained_distinct_source_images": int(
            summary["retained_distinct_source_images"]
        ),
        "unit_rule": str(summary["unit_rule"]),
        "same_source_image_rule": str(summary["same_source_image_rule"]),
    }


def prepare_cohort_manifest(
    *,
    root: Path,
    assets: RepeatAssets = DEFAULT_ASSETS,
) -> dict[str, Any]:
    """Materialize the model-independent retained cohort with resolvable paths."""

    cohort, cohort_summary = _cohort(assets)
    source_rows = {row.sample_id: row for row in load_manifest(assets.source_input_manifest)}
    retained_ids = {row.sample_id for row in cohort}
    missing = retained_ids - set(source_rows)
    _require(not missing, f"source input manifest misses {len(missing)} cohort samples")

    output = cohort_manifest_path(root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for sample_id in sorted(retained_ids):
            row = source_rows[sample_id]
            stream.write(
                _canonical_json_bytes(
                    {
                        "sample_id": row.sample_id,
                        "roi_path": str(row.roi_path),
                        "roi_png_sha256": row.roi_png_sha256,
                        "roi_pixel_sha256": row.roi_pixel_sha256,
                    }
                )
                + b"\n"
            )
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "manifest": str(output),
        "cohort": _canonical_cohort_summary(cohort_summary),
    }


def run_prediction_job(
    *,
    root: Path,
    method: str,
    seed: int,
    device_name: str,
    manifest: Path | None = None,
) -> Mapping[str, Any]:
    """Run one clean-only seed without opening labels."""

    _require(method in METHODS, f"unknown method: {method}")
    _require(seed in SEEDS, f"unexpected training seed: {seed}")
    input_manifest = Path(manifest or cohort_manifest_path(root))
    artifacts = prediction_artifacts(root, method=method, seed=seed)

    if method == "direct_resnet18":
        return run_sarn_prediction(
            checkpoint=factorial_checkpoint_path(
                Path(root) / "factorial", cell="00", seed=seed
            ),
            manifest=input_manifest,
            output=artifacts["predictions"],
            sidecar=artifacts["sidecar"],
            summary=artifacts["summary"],
            cell="00",
            seed=seed,
            device_name=device_name,
            conditions=("clean",),
        )
    if method == "geoattn_sarn":
        return run_sarn_prediction(
            checkpoint=factorial_checkpoint_path(
                Path(root) / "factorial", cell="11", seed=seed
            ),
            manifest=input_manifest,
            output=artifacts["predictions"],
            sidecar=artifacts["sidecar"],
            summary=artifacts["summary"],
            cell="11",
            seed=seed,
            device_name=device_name,
            conditions=("clean",),
        )

    def checkpoint_loader(checkpoint: Path, *, device_name: str):
        _source_method, model, metadata = pg_training.load_checkpoint_model(
            checkpoint, device_name=device_name
        )
        _require(int(metadata.get("seed", -1)) == seed, "PG-SIAM seed mismatch")
        return prediction_method_id(method, seed), model, metadata

    return pg_evaluation.run_prediction(
        checkpoint_path=pgsiam_checkpoint_path(root, seed=seed),
        manifest_path=input_manifest,
        output_path=artifacts["predictions"],
        diagnostics_path=artifacts["diagnostics"],
        summary_path=artifacts["summary"],
        device_name=device_name,
        conditions=("clean",),
        checkpoint_loader=checkpoint_loader,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"prediction JSONL does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PaperNaturalRepeatError(
                    f"invalid prediction JSON at {source}:{line_number}"
                ) from exc
            _require(
                isinstance(value, Mapping),
                f"prediction row is not an object at {source}:{line_number}",
            )
            rows.append(dict(value))
    _require(bool(rows), f"prediction JSONL is empty: {source}")
    return rows


def load_clean_prediction_file(
    path: Path,
    *,
    expected_method: str,
    cohort_ids: set[str],
) -> tuple[dict[str, PredictionValue], dict[str, str]]:
    values: dict[str, PredictionValue] = {}
    pixel_hashes: dict[str, str] = {}
    for index, row in enumerate(_read_jsonl(path), 1):
        if row.get("condition") != "clean":
            continue
        method = str(row.get("method") or "").strip()
        _require(
            method == expected_method,
            f"prediction method mismatch at {path}:{index}: {method}",
        )
        sample_id = str(row.get("sample_id") or "").strip()
        if sample_id not in cohort_ids:
            continue
        _require(sample_id not in values, f"duplicate clean prediction: {sample_id}")
        pixel_hash = str(row.get("condition_pixel_sha256") or "").strip()
        _require(
            len(pixel_hash) == 64
            and all(character in "0123456789abcdef" for character in pixel_hash),
            f"invalid clean condition pixel hash: {expected_method}/{sample_id}",
        )
        if str(row.get("status") or "").lower() == "pass":
            raw = row.get("normalized_progress")
            _require(
                isinstance(raw, (int, float)) and not isinstance(raw, bool),
                f"prediction is not numeric: {expected_method}/{sample_id}",
            )
            value = float(raw)
            _require(
                math.isfinite(value) and 0.0 <= value <= 1.0,
                f"prediction outside [0,1]: {expected_method}/{sample_id}",
            )
            values[sample_id] = PredictionValue(True, value)
        else:
            values[sample_id] = PredictionValue(False, None)
        pixel_hashes[sample_id] = pixel_hash
    missing = cohort_ids - set(values)
    _require(not missing, f"predictions miss {len(missing)} retained samples")
    return values, pixel_hashes


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile input is empty")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_mean(
    grouped_values: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a mean while resampling whole physical instruments."""

    _require(replicates >= 100, "bootstrap requires at least 100 replicates")
    groups = sorted(grouped_values)
    _require(len(groups) >= 2, "cluster bootstrap requires at least two groups")
    _require(
        all(bool(grouped_values[group]) for group in groups),
        "cluster bootstrap contains an empty group",
    )
    group_totals = {
        group: float(math.fsum(float(value) for value in grouped_values[group]))
        for group in groups
    }
    group_counts = {group: len(grouped_values[group]) for group in groups}
    point = float(
        math.fsum(group_totals.values()) / sum(group_counts.values())
    )
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(replicates):
        sampled_groups = [groups[rng.randrange(len(groups))] for _ in groups]
        total = math.fsum(group_totals[group] for group in sampled_groups)
        count = sum(group_counts[group] for group in sampled_groups)
        draws.append(float(total / count))
    return {
        "point_estimate": point,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "replicates": int(replicates),
        "seed": int(seed),
        "cluster_unit": "physical instrument group_id",
        "physical_groups": len(groups),
    }


def _seed_summary(values: Sequence[float]) -> dict[str, float]:
    floats = [float(value) for value in values]
    _require(bool(floats), "seed summary is empty")
    return {
        "mean": float(statistics.fmean(floats)),
        "sample_sd": float(statistics.stdev(floats)) if len(floats) >= 2 else 0.0,
    }


def _seed_error(
    prediction: PredictionValue,
    *,
    target: float,
) -> float:
    if prediction.passed and prediction.value is not None:
        return abs(float(prediction.value) - float(target))
    return 1.0


def _method_grouped_values(
    *,
    cohort: Sequence[CohortRow],
    seed_predictions: Mapping[int, Mapping[str, PredictionValue]],
    seed_units: Mapping[int, Mapping[tuple[str, str], Mapping[str, float]]],
    seeds: Sequence[int],
) -> dict[str, dict[str, list[float]]]:
    nmae: dict[str, list[float]] = defaultdict(list)
    for row in cohort:
        nmae[row.group_id].append(
            statistics.fmean(
                _seed_error(seed_predictions[int(seed)][row.sample_id], target=row.target)
                for seed in seeds
            )
        )

    complete_units = set.intersection(
        *(set(seed_units[int(seed)]) for seed in seeds)
    )
    sd_values: dict[str, list[float]] = defaultdict(list)
    ranges: dict[str, list[float]] = defaultdict(list)
    for group_id, truth_key in sorted(complete_units):
        sd_values[group_id].append(
            statistics.fmean(
                float(
                    seed_units[int(seed)][(group_id, truth_key)][
                        "prediction_sd_population"
                    ]
                )
                for seed in seeds
            )
        )
        ranges[group_id].append(
            statistics.fmean(
                float(
                    seed_units[int(seed)][(group_id, truth_key)][
                        "prediction_range"
                    ]
                )
                for seed in seeds
            )
        )
    return {
        "full_denominator_nmae": dict(nmae),
        "mean_within_unit_prediction_sd_population": dict(sd_values),
        "mean_within_unit_prediction_range": dict(ranges),
    }


def _paired_method_grouped_values(
    *,
    cohort: Sequence[CohortRow],
    candidate_predictions: Mapping[int, Mapping[str, PredictionValue]],
    reference_predictions: Mapping[int, Mapping[str, PredictionValue]],
    candidate_units: Mapping[
        int, Mapping[tuple[str, str], Mapping[str, float]]
    ],
    reference_units: Mapping[
        int, Mapping[tuple[str, str], Mapping[str, float]]
    ],
    seeds: Sequence[int],
) -> dict[str, dict[str, list[float]]]:
    """Build candidate-minus-reference effects on exactly paired samples/units."""

    nmae: dict[str, list[float]] = defaultdict(list)
    for row in cohort:
        candidate_error = statistics.fmean(
            _seed_error(
                candidate_predictions[int(seed)][row.sample_id], target=row.target
            )
            for seed in seeds
        )
        reference_error = statistics.fmean(
            _seed_error(
                reference_predictions[int(seed)][row.sample_id], target=row.target
            )
            for seed in seeds
        )
        nmae[row.group_id].append(candidate_error - reference_error)

    paired_units = set.intersection(
        *(
            [set(candidate_units[int(seed)]) for seed in seeds]
            + [set(reference_units[int(seed)]) for seed in seeds]
        )
    )
    sd_values: dict[str, list[float]] = defaultdict(list)
    ranges: dict[str, list[float]] = defaultdict(list)
    for group_id, truth_key in sorted(paired_units):
        key = (group_id, truth_key)
        candidate_sd = statistics.fmean(
            float(candidate_units[int(seed)][key]["prediction_sd_population"])
            for seed in seeds
        )
        reference_sd = statistics.fmean(
            float(reference_units[int(seed)][key]["prediction_sd_population"])
            for seed in seeds
        )
        candidate_range = statistics.fmean(
            float(candidate_units[int(seed)][key]["prediction_range"])
            for seed in seeds
        )
        reference_range = statistics.fmean(
            float(reference_units[int(seed)][key]["prediction_range"])
            for seed in seeds
        )
        sd_values[group_id].append(candidate_sd - reference_sd)
        ranges[group_id].append(candidate_range - reference_range)
    return {
        "full_denominator_nmae": dict(nmae),
        "mean_within_unit_prediction_sd_population": dict(sd_values),
        "mean_within_unit_prediction_range": dict(ranges),
    }


def score_stability(
    *,
    root: Path,
    assets: RepeatAssets = DEFAULT_ASSETS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    cohort, cohort_summary = _cohort(assets)
    cohort_ids = {row.sample_id for row in cohort}

    predictions: dict[str, dict[int, dict[str, PredictionValue]]] = {}
    condition_pixel_hashes: dict[str, dict[int, dict[str, str]]] = {}
    scores: dict[str, dict[int, dict[str, Any]]] = {}
    units: dict[
        str, dict[int, dict[tuple[str, str], dict[str, float]]]
    ] = {}
    for method, spec in METHODS.items():
        predictions[method] = {}
        condition_pixel_hashes[method] = {}
        scores[method] = {}
        units[method] = {}
        for seed in SEEDS:
            path = prediction_artifacts(
                root, method=method, seed=seed
            )["predictions"]
            expected_method = prediction_method_id(method, seed)
            values, pixel_hashes = load_clean_prediction_file(
                path,
                expected_method=expected_method,
                cohort_ids=cohort_ids,
            )
            predictions[method][seed] = values
            condition_pixel_hashes[method][seed] = pixel_hashes
            summary, details = score_one_method(cohort, values)
            scores[method][seed] = summary
            units[method][seed] = details

    for sample_id in sorted(cohort_ids):
        observed = {
            condition_pixel_hashes[method][seed][sample_id]
            for method in METHODS
            for seed in SEEDS
        }
        _require(
            len(observed) == 1,
            f"paired clean input pixel mismatch across methods/seeds: {sample_id}",
        )

    methods: dict[str, Any] = {}
    grouped_by_method: dict[str, dict[str, dict[str, list[float]]]] = {}
    for method, spec in METHODS.items():
        per_seed = {
            str(seed): {
                "method": prediction_method_id(method, seed),
                **scores[method][seed],
            }
            for seed in SEEDS
        }
        grouped = _method_grouped_values(
            cohort=cohort,
            seed_predictions=predictions[method],
            seed_units=units[method],
            seeds=SEEDS,
        )
        grouped_by_method[method] = grouped
        three_seed_metrics = {
            "full_denominator_nmae": _seed_summary(
                [scores[method][seed]["full_denominator_nmae"] for seed in SEEDS]
            ),
            "mean_within_unit_prediction_sd_population": _seed_summary(
                [
                    scores[method][seed][
                        "within_unit_prediction_sd_population"
                    ]["mean"]
                    for seed in SEEDS
                ]
            ),
            "mean_within_unit_prediction_range": _seed_summary(
                [
                    scores[method][seed]["within_unit_prediction_range"]["mean"]
                    for seed in SEEDS
                ]
            ),
        }
        bootstrap = {
            metric: cluster_bootstrap_mean(
                values,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + offset,
            )
            for offset, (metric, values) in enumerate(grouped.items())
        }
        methods[spec.paper_label] = {
            "paper_label": spec.paper_label,
            "per_seed": per_seed,
            "three_seed": {
                "training_seed_summary": three_seed_metrics,
                "physical_group_cluster_bootstrap_95ci": bootstrap,
                "complete_stability_units_all_seeds": sum(
                    len(values) for values in grouped[
                        "mean_within_unit_prediction_sd_population"
                    ].values()
                ),
            },
        }

    comparisons: dict[str, Any] = {}
    comparison_pairs = (
        ("geoattn_sarn", "direct_resnet18"),
        ("pgsiam", "direct_resnet18"),
        ("pgsiam", "geoattn_sarn"),
    )
    for pair_offset, (candidate, reference) in enumerate(comparison_pairs):
        candidate_label = METHODS[candidate].paper_label
        reference_label = METHODS[reference].paper_label
        paired_values = _paired_method_grouped_values(
            cohort=cohort,
            candidate_predictions=predictions[candidate],
            reference_predictions=predictions[reference],
            candidate_units=units[candidate],
            reference_units=units[reference],
            seeds=SEEDS,
        )
        metric_results: dict[str, Any] = {}
        for metric_offset, (metric, differences) in enumerate(
            paired_values.items()
        ):
            metric_results[metric] = cluster_bootstrap_mean(
                differences,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + 100 + pair_offset * 10 + metric_offset,
            )
        comparisons[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate_label,
            "reference": reference_label,
            "effect_direction": (
                f"{candidate_label} minus {reference_label}; negative favors "
                f"{candidate_label}"
            ),
            "metrics": metric_results,
        }

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "evaluation_name": "natural repeated-capture stability",
        "paper_role": "supplementary material",
        "condition": "clean",
        "training_data": "SyncG fit only",
        "field_data_role": "test only",
        "paired_input_pixel_identity": True,
        "explicit_non_claims": [
            "not an angle or perspective-severity stratification",
            "not a claim that every repeated source image changes viewpoint",
        ],
        "cohort": _canonical_cohort_summary(cohort_summary),
        "methods": methods,
        "paired_physical_group_cluster_bootstrap": comparisons,
        "inputs": {
            "provenance": str(Path(assets.provenance).resolve()),
            "labels": str(Path(assets.labels).resolve()),
            "physical_capture_manifest": str(
                Path(assets.physical_capture_manifest).resolve()
            ),
            "cohort_input_manifest": str(cohort_manifest_path(root).resolve()),
            "predictions": {
                METHODS[method].paper_label: [
                    str(
                        prediction_artifacts(root, method=method, seed=seed)[
                            "predictions"
                        ].resolve()
                    )
                    for seed in SEEDS
                ]
                for method in METHODS
            },
        },
    }


def write_score(
    *,
    root: Path,
    assets: RepeatAssets = DEFAULT_ASSETS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    result = score_stability(
        root=root,
        assets=assets,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output = result_path(root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(result) + b"\n")
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "output": str(output),
        "samples": result["cohort"]["retained_samples"],
        "physical_groups": result["cohort"]["retained_physical_groups"],
        "exact_reading_units": result["cohort"][
            "retained_exact_reading_units"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=DEFAULT_ROOT)

    predict = commands.add_parser("predict")
    predict.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    predict.add_argument("--method", action="append", choices=tuple(METHODS))
    predict.add_argument("--seed", action="append", type=int, choices=SEEDS)
    predict.add_argument("--device", default="cuda:0")

    score = commands.add_parser("score")
    score.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    score.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    score.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        print(
            json.dumps(
                prepare_cohort_manifest(root=args.root),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "score":
        print(
            json.dumps(
                write_score(
                    root=args.root,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    manifest = cohort_manifest_path(args.root).resolve()
    _require(
        manifest.is_file(),
        f"cohort manifest does not exist; run the prepare command first: {manifest}",
    )
    methods = tuple(args.method) if args.method else tuple(METHODS)
    seeds = tuple(args.seed) if args.seed else SEEDS
    for method in methods:
        for seed in seeds:
            print(
                json.dumps(
                    {
                        "status": "starting",
                        "method": prediction_method_id(method, seed),
                        "condition": "clean",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            result = run_prediction_job(
                root=args.root,
                method=method,
                seed=seed,
                device_name=args.device,
                manifest=manifest,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ASSETS",
    "DEFAULT_ROOT",
    "METHODS",
    "PROTOCOL",
    "RepeatAssets",
    "cluster_bootstrap_mean",
    "cohort_manifest_path",
    "main",
    "prediction_artifacts",
    "prediction_method_id",
    "prepare_cohort_manifest",
    "result_path",
    "run_prediction_job",
    "score_stability",
    "write_score",
]
