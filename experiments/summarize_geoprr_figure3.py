"""Validate and publish the GeoPRR-Net Figure 3 experiment ledgers.

The script combines the three existing factorial cells with the new
geometry-off/fixed-routing inference cell, computes the prespecified
difference-in-differences interaction, and summarizes the paired perspective
scan.  Confidence intervals use 20,000 resamples of the 14 SyncG scene
clusters; no image or high-angle fallback row is removed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO

import numpy as np
import torch

from experiments.evaluate_a15_2_syncg_scene_holdout import CONDITIONS
from experiments.evaluate_geoprr_geometry_routing_factorial import (
    PROTOCOL as JOINT_PROTOCOL,
)
from experiments.evaluate_geoprr_perspective_scan import (
    ANGLES,
    FULL_MODEL,
    MODELS,
    NO_GEOMETRY_MODEL,
    PROTOCOL as PERSPECTIVE_PROTOCOL,
    RAW_EFFICIENTNET_MODEL,
    SEEDS,
)
from experiments.unified_pointer_reader import (
    FIXED_ROUTING,
    FULL,
    NO_GEOMETRY_FIXED_ROUTING,
    NO_GEOMETRY_FUSION,
    PROTOCOL as TRAINING_PROTOCOL,
    adaptive_routing_enabled,
)


PROTOCOL: Final[str] = "geoprr_figure3_public_results_v1"
BOOTSTRAP_REPLICATES: Final[int] = 20_000
BOOTSTRAP_SEED: Final[int] = 20_260_830
EXPECTED_SAMPLES: Final[int] = 1_558
EXPECTED_SCENES: Final[int] = 14
EXPECTED_FACTORIAL_ROWS_PER_SEED: Final[int] = EXPECTED_SAMPLES * len(CONDITIONS)
EXPECTED_PERSPECTIVE_ROWS: Final[int] = (
    EXPECTED_SAMPLES * len(ANGLES) * len(SEEDS) * len(MODELS)
)


@dataclass(frozen=True, slots=True)
class FactorialCell:
    variant: str
    geometry_on: bool
    adaptive_routing_on: bool
    source: str


FACTORIAL_CELLS: Final[tuple[FactorialCell, ...]] = (
    FactorialCell(FULL, True, True, "existing"),
    FactorialCell(FIXED_ROUTING, True, False, "existing"),
    FactorialCell(NO_GEOMETRY_FUSION, False, True, "existing"),
    FactorialCell(NO_GEOMETRY_FIXED_ROUTING, False, False, "joint"),
)
CELL_BY_VARIANT: Final[dict[str, FactorialCell]] = {
    cell.variant: cell for cell in FACTORIAL_CELLS
}

FACTORIAL_FIELDS: Final[tuple[str, ...]] = (
    "seed",
    "scene_id",
    "image_id",
    "condition",
    "target",
    "prediction",
    "absolute_error",
    "geometry_on",
    "adaptive_routing_on",
    "variant",
)
PERSPECTIVE_FIELDS: Final[tuple[str, ...]] = (
    "model",
    "seed",
    "scene_id",
    "image_id",
    "angle",
    "target",
    "prediction",
    "absolute_error",
    "availability",
    "fallback",
    "valid_support_fraction",
    "fallback_reason",
    "perspective_axis",
    "perspective_sign",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    result = float(statistics.fmean(values))
    _require(math.isfinite(result), "mean is non-finite")
    return result


def _sample_sd(values: Sequence[float]) -> float:
    _require(bool(values), "cannot compute SD of an empty sequence")
    result = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    _require(math.isfinite(result), "SD is non-finite")
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot compute a percentile of an empty sequence")
    result = float(np.quantile(np.asarray(values, dtype=np.float64), probability))
    _require(math.isfinite(result), "percentile is non-finite")
    return result


def _bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    _require(text in {"true", "false", "1", "0"}, f"{label} is not boolean")
    return text in {"true", "1"}


@contextmanager
def _deterministic_gzip_text(path: Path) -> Iterable[TextIO]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.open("wb")
    compressed = gzip.GzipFile(
        filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
    )
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.close()


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).resolve().write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _cluster_bootstrap_mean_ci(
    rows: Sequence[tuple[str, float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a row-weighted mean CI after resampling whole scene clusters."""

    _require(replicates >= 1, "bootstrap replicate count must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene, value in rows:
        scalar = float(value)
        _require(math.isfinite(scalar), "bootstrap value is non-finite")
        grouped[str(scene)].append(scalar)
    scenes = tuple(sorted(grouped))
    _require(len(scenes) == EXPECTED_SCENES, "bootstrap scene count differs")
    sums = np.asarray([sum(grouped[scene]) for scene in scenes], dtype=np.float64)
    sizes = np.asarray([len(grouped[scene]) for scene in scenes], dtype=np.float64)
    _require(bool((sizes > 0).all()), "bootstrap contains an empty scene")
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(scenes), size=(int(replicates), len(scenes)))
    estimates = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    point = float(sums.sum() / sizes.sum())
    return {
        "point_estimate": point,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "replicates": int(replicates),
        "clusters": len(scenes),
        "cluster_unit": "scene_id",
        "bootstrap_seed": int(seed),
        "row_weighted": True,
    }


def _factorial_path(existing_root: Path, joint_root: Path, seed: int, cell: FactorialCell) -> Path:
    if cell.source == "joint":
        return Path(joint_root) / f"seed_{seed}/{cell.variant}/syncg.json"
    return Path(existing_root) / f"seed_{seed}/{cell.variant}/syncg.json"


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def _audit_factorial_checkpoints(existing_root: Path) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        payloads: dict[str, Mapping[str, Any]] = {}
        for variant in (FULL, FIXED_ROUTING, NO_GEOMETRY_FUSION):
            path = Path(existing_root) / f"seed_{seed}/{variant}/ema.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _require(isinstance(payload, Mapping), "factorial checkpoint is malformed")
            _require(payload.get("protocol") == TRAINING_PROTOCOL, "training protocol differs")
            _require(payload.get("variant") == variant, "checkpoint variant differs")
            _require(int(payload.get("seed")) == seed, "checkpoint seed differs")
            payloads[variant] = payload

        full = payloads[FULL]
        fixed = payloads[FIXED_ROUTING]
        no_geometry = payloads[NO_GEOMETRY_FUSION]
        training = fixed.get("training")
        _require(isinstance(training, Mapping), "fixed training metadata is missing")
        source_checkpoints = {
            str(payloads[variant].get("source_r2mt_checkpoint"))
            for variant in payloads
        }
        source_metadata_equal = all(
            payloads[variant].get("source_r2mt_metadata")
            == full.get("source_r2mt_metadata")
            for variant in payloads
        )
        construction_equal = all(
            payloads[variant].get("construction") == full.get("construction")
            for variant in payloads
        )
        polar_equal = all(
            _state_equal(
                full["polar_expert_state"],
                payloads[variant]["polar_expert_state"],
            )
            for variant in payloads
        )
        router_states_distinct = not _state_equal(
            full["regret_router_state"], no_geometry["regret_router_state"]
        )
        report = {
            "same_source_expert_checkpoint": len(source_checkpoints) == 1,
            "same_source_expert_metadata": source_metadata_equal,
            "same_model_construction": construction_equal,
            "same_polar_expert_state": polar_equal,
            "full_and_no_geometry_use_distinct_trained_router_states": (
                router_states_distinct
            ),
            "fixed_checkpoint_router_training": training.get("router_training"),
            "joint_cell_uses_fixed_checkpoint": True,
            "joint_specific_checkpoint_or_parameters": False,
            "joint_specific_training_or_adaptation": False,
            "joint_executes_adaptive_router": adaptive_routing_enabled(
                NO_GEOMETRY_FIXED_ROUTING
            ),
        }
        _require(all(report[key] for key in (
            "same_source_expert_checkpoint",
            "same_source_expert_metadata",
            "same_model_construction",
            "same_polar_expert_state",
            "full_and_no_geometry_use_distinct_trained_router_states",
            "joint_cell_uses_fixed_checkpoint",
        )), f"factorial checkpoint audit failed for seed {seed}")
        _require(
            report["fixed_checkpoint_router_training"] is False,
            "fixed checkpoint unexpectedly trained the router",
        )
        _require(
            report["joint_executes_adaptive_router"] is False,
            "joint cell unexpectedly executes adaptive routing",
        )
        per_seed[str(seed)] = report
    return {
        "status": "pass",
        "conclusion": (
            "The missing geometry-off/fixed-routing cell is a pure inference "
            "intervention on each existing fixed-routing checkpoint. No "
            "joint-specific trained parameters are loaded."
        ),
        "per_seed": per_seed,
    }


def _summarize_factorial(
    *,
    existing_root: Path,
    joint_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    errors: dict[str, dict[int, dict[tuple[str, str, str], float]]] = {
        cell.variant: {seed: {} for seed in SEEDS} for cell in FACTORIAL_CELLS
    }
    per_condition: dict[str, dict[int, dict[str, list[float]]]] = {
        cell.variant: {
            seed: {condition: [] for condition in CONDITIONS} for seed in SEEDS
        }
        for cell in FACTORIAL_CELLS
    }
    canonical_roster: set[tuple[str, str, str, float]] | None = None
    ledger_path = output_dir / "geometry_routing_per_sample.csv.gz"
    with _deterministic_gzip_text(ledger_path) as stream:
        writer = csv.DictWriter(stream, fieldnames=FACTORIAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        written = 0
        for cell in FACTORIAL_CELLS:
            for seed in SEEDS:
                source = _factorial_path(existing_root, joint_root, seed, cell).resolve()
                payload = json.loads(source.read_text(encoding="utf-8-sig"))
                _require(payload.get("status") == "complete", "factorial run is incomplete")
                expected_protocol = JOINT_PROTOCOL if cell.source == "joint" else None
                if expected_protocol is not None:
                    _require(payload.get("protocol") == expected_protocol, "joint protocol differs")
                    scope = payload.get("scope")
                    _require(isinstance(scope, Mapping), "joint scope is missing")
                    _require(scope.get("joint_factorial_inference_only") is True, "joint is not inference-only")
                    _require(scope.get("joint_specific_training_or_adaptation") is False, "joint adaptation was reported")
                    _require(scope.get("joint_specific_parameters_loaded") is False, "joint parameters were reported")
                _require(payload.get("architecture_variant") == cell.variant, "factorial variant differs")
                rows = payload.get("per_sample_condition")
                _require(isinstance(rows, list), "factorial rows are missing")
                _require(len(rows) == EXPECTED_FACTORIAL_ROWS_PER_SEED, "factorial row count differs")
                roster: set[tuple[str, str, str, float]] = set()
                for row in rows:
                    _require(isinstance(row, Mapping), "factorial row is malformed")
                    image_id = str(row["sample_id"])
                    scene = str(row["scene_stem"])
                    condition = str(row["condition"])
                    target = float(row["normalized_target"])
                    candidate = row.get("mett")
                    _require(isinstance(candidate, Mapping), "factorial prediction is missing")
                    prediction = float(candidate["prediction"])
                    absolute_error = float(candidate["absolute_error"])
                    _require(condition in CONDITIONS, "factorial condition differs")
                    _require(
                        all(math.isfinite(value) for value in (target, prediction, absolute_error)),
                        "factorial value is non-finite",
                    )
                    _require(0.0 <= target <= 1.0 and 0.0 <= prediction <= 1.0, "factorial value is out of range")
                    _require(
                        abs(absolute_error - abs(prediction - target)) <= 1e-12,
                        "factorial absolute error differs",
                    )
                    key = (scene, image_id, condition)
                    _require(key not in errors[cell.variant][seed], "factorial key repeats")
                    errors[cell.variant][seed][key] = absolute_error
                    per_condition[cell.variant][seed][condition].append(absolute_error)
                    roster.add((scene, image_id, condition, target))
                    writer.writerow(
                        {
                            "seed": seed,
                            "scene_id": scene,
                            "image_id": image_id,
                            "condition": condition,
                            "target": target,
                            "prediction": prediction,
                            "absolute_error": absolute_error,
                            "geometry_on": cell.geometry_on,
                            "adaptive_routing_on": cell.adaptive_routing_on,
                            "variant": cell.variant,
                        }
                    )
                    written += 1
                _require(len(roster) == EXPECTED_FACTORIAL_ROWS_PER_SEED, "factorial roster repeats")
                _require(len({row[0] for row in roster}) == EXPECTED_SCENES, "factorial scene count differs")
                for condition in CONDITIONS:
                    _require(
                        len(per_condition[cell.variant][seed][condition]) == EXPECTED_SAMPLES,
                        "factorial condition denominator differs",
                    )
                if canonical_roster is None:
                    canonical_roster = roster
                else:
                    _require(roster == canonical_roster, "factorial image/perturbation roster differs")
        expected_written = len(FACTORIAL_CELLS) * len(SEEDS) * EXPECTED_FACTORIAL_ROWS_PER_SEED
        _require(written == expected_written, "factorial Cartesian row count differs")

    seed_metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {}
    for cell in FACTORIAL_CELLS:
        seed_nmae: list[float] = []
        seed_acc: list[float] = []
        pooled: list[float] = []
        per_seed_json: dict[str, Any] = {}
        for seed in SEEDS:
            values = list(errors[cell.variant][seed].values())
            nmae = _mean(values)
            acc = _mean([float(value <= 0.02) for value in values])
            seed_nmae.append(nmae)
            seed_acc.append(acc)
            pooled.extend(values)
            seed_metric_rows.append(
                {
                    "variant": cell.variant,
                    "geometry_on": cell.geometry_on,
                    "adaptive_routing_on": cell.adaptive_routing_on,
                    "seed": seed,
                    "rows": len(values),
                    "nmae": nmae,
                    "acc_at_2_percent": acc,
                }
            )
            per_seed_json[str(seed)] = {"nmae": nmae, "acc_at_2_percent": acc}
        aggregate = {
            "pooled_nmae": _mean(pooled),
            "pooled_acc_at_2_percent": _mean([float(value <= 0.02) for value in pooled]),
            "seed_nmae_mean": _mean(seed_nmae),
            "seed_nmae_sd": _sample_sd(seed_nmae),
            "seed_acc_at_2_percent_mean": _mean(seed_acc),
            "seed_acc_at_2_percent_sd": _sample_sd(seed_acc),
        }
        summary_rows.append(
            {
                "variant": cell.variant,
                "geometry_on": cell.geometry_on,
                "adaptive_routing_on": cell.adaptive_routing_on,
                "rows_per_seed": EXPECTED_FACTORIAL_ROWS_PER_SEED,
                "pooled_rows": len(pooled),
                **aggregate,
            }
        )
        summary_json[cell.variant] = {**aggregate, "per_seed": per_seed_json}
        for condition in CONDITIONS:
            condition_seed_nmae: list[float] = []
            condition_seed_acc: list[float] = []
            condition_pooled: list[float] = []
            for seed in SEEDS:
                values = per_condition[cell.variant][seed][condition]
                condition_seed_nmae.append(_mean(values))
                condition_seed_acc.append(_mean([float(value <= 0.02) for value in values]))
                condition_pooled.extend(values)
            condition_rows.append(
                {
                    "variant": cell.variant,
                    "geometry_on": cell.geometry_on,
                    "adaptive_routing_on": cell.adaptive_routing_on,
                    "condition": condition,
                    "rows_per_seed": EXPECTED_SAMPLES,
                    "pooled_nmae": _mean(condition_pooled),
                    "seed_nmae_mean": _mean(condition_seed_nmae),
                    "seed_nmae_sd": _sample_sd(condition_seed_nmae),
                    "pooled_acc_at_2_percent": _mean([float(value <= 0.02) for value in condition_pooled]),
                    "seed_acc_at_2_percent_mean": _mean(condition_seed_acc),
                    "seed_acc_at_2_percent_sd": _sample_sd(condition_seed_acc),
                }
            )

    interaction_per_seed: dict[str, float] = {}
    averaged_rows: list[tuple[str, float]] = []
    reference_keys = set(errors[FULL][SEEDS[0]])
    for variant in CELL_BY_VARIANT:
        for seed in SEEDS:
            _require(set(errors[variant][seed]) == reference_keys, "factorial interaction roster differs")
    per_seed_deltas: dict[int, dict[tuple[str, str, str], float]] = {}
    for seed in SEEDS:
        deltas = {
            key: (
                errors[FULL][seed][key]
                - errors[FIXED_ROUTING][seed][key]
                - errors[NO_GEOMETRY_FUSION][seed][key]
                + errors[NO_GEOMETRY_FIXED_ROUTING][seed][key]
            )
            for key in reference_keys
        }
        per_seed_deltas[seed] = deltas
        interaction_per_seed[str(seed)] = _mean(list(deltas.values()))
    for key in sorted(reference_keys):
        scene = key[0]
        averaged_rows.append((scene, _mean([per_seed_deltas[seed][key] for seed in SEEDS])))
    bootstrap = _cluster_bootstrap_mean_ci(averaged_rows)
    seed_values = list(interaction_per_seed.values())
    interaction = {
        "definition": (
            "I = (E[geometry on, adaptive] - E[geometry on, fixed]) - "
            "(E[geometry off, adaptive] - E[geometry off, fixed])"
        ),
        "error_metric": "absolute error on normalized full-scale target (NMAE)",
        "sign_convention": "positive means the adaptive-routing gain is larger when geometry is off",
        "per_seed": interaction_per_seed,
        "seed_mean": _mean(seed_values),
        "seed_sd": _sample_sd(seed_values),
        "scene_cluster_bootstrap": bootstrap,
        "seed_handling_in_bootstrap": "paired row differences averaged over the three seeds before scene resampling",
    }
    _require(
        abs(interaction["seed_mean"] - bootstrap["point_estimate"]) <= 1e-15,
        "interaction point estimates differ",
    )
    _write_csv(
        output_dir / "geometry_routing_seed_metrics.csv",
        (
            "variant", "geometry_on", "adaptive_routing_on", "seed", "rows", "nmae", "acc_at_2_percent"
        ),
        seed_metric_rows,
    )
    _write_csv(
        output_dir / "geometry_routing_summary.csv",
        tuple(summary_rows[0]),
        summary_rows,
    )
    _write_csv(
        output_dir / "geometry_routing_conditions.csv",
        tuple(condition_rows[0]),
        condition_rows,
    )
    _write_json(output_dir / "geometry_routing_interaction.json", interaction)
    return {
        "rows": written,
        "samples": EXPECTED_SAMPLES,
        "scenes": EXPECTED_SCENES,
        "conditions": list(CONDITIONS),
        "cells": summary_json,
        "interaction": interaction,
    }


def _summarize_perspective(
    *,
    ledger_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = json.loads(Path(metadata_path).resolve().read_text(encoding="utf-8-sig"))
    _require(metadata.get("protocol") == PERSPECTIVE_PROTOCOL, "perspective protocol differs")
    _require(metadata.get("status") == "complete", "perspective run is incomplete")
    errors: dict[str, dict[int, dict[int, dict[tuple[str, str], float]]]] = {
        model: {seed: {angle: {} for angle in ANGLES} for seed in SEEDS}
        for model in MODELS
    }
    input_diagnostics: dict[int, dict[tuple[str, str], tuple[bool, bool, float, str, str, int]]] = {
        angle: {} for angle in ANGLES
    }
    targets: dict[tuple[str, str], float] = {}
    public_path = output_dir / "perspective_scan_per_sample.csv.gz"
    source = Path(ledger_path).resolve()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as input_stream, _deterministic_gzip_text(public_path) as output_stream:
        reader = csv.DictReader(input_stream)
        _require(tuple(reader.fieldnames or ()) == PERSPECTIVE_FIELDS, "perspective fields differ")
        writer = csv.DictWriter(output_stream, fieldnames=PERSPECTIVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        written = 0
        for row in reader:
            model = str(row["model"])
            seed = int(row["seed"])
            angle = int(row["angle"])
            scene = str(row["scene_id"])
            image_id = str(row["image_id"])
            target = float(row["target"])
            prediction = float(row["prediction"])
            absolute_error = float(row["absolute_error"])
            availability = _bool(row["availability"], label="availability")
            fallback = _bool(row["fallback"], label="fallback")
            support_fraction = float(row["valid_support_fraction"])
            _require(model in MODELS and seed in SEEDS and angle in ANGLES, "perspective factor differs")
            _require(
                all(math.isfinite(value) for value in (target, prediction, absolute_error, support_fraction)),
                "perspective value is non-finite",
            )
            _require(availability != fallback, "perspective availability and fallback differ")
            _require(0.0 < support_fraction <= 1.0, "perspective support fraction differs")
            _require(abs(absolute_error - abs(prediction - target)) <= 1e-12, "perspective absolute error differs")
            key = (scene, image_id)
            _require(key not in errors[model][seed][angle], "perspective prediction key repeats")
            errors[model][seed][angle][key] = absolute_error
            diagnostic = (
                availability,
                fallback,
                support_fraction,
                str(row["fallback_reason"]),
                str(row["perspective_axis"]),
                int(row["perspective_sign"]),
            )
            if key in input_diagnostics[angle]:
                _require(input_diagnostics[angle][key] == diagnostic, "perspective pixels differ across models or seeds")
            else:
                input_diagnostics[angle][key] = diagnostic
            if key in targets:
                _require(abs(targets[key] - target) <= 1e-12, "perspective target differs")
            else:
                targets[key] = target
            writer.writerow(row)
            written += 1
    _require(written == EXPECTED_PERSPECTIVE_ROWS, "perspective Cartesian row count differs")
    reference_roster: set[tuple[str, str]] | None = None
    for model in MODELS:
        for seed in SEEDS:
            for angle in ANGLES:
                roster = set(errors[model][seed][angle])
                _require(len(roster) == EXPECTED_SAMPLES, "perspective denominator differs")
                _require(len({key[0] for key in roster}) == EXPECTED_SCENES, "perspective scene count differs")
                if reference_roster is None:
                    reference_roster = roster
                else:
                    _require(roster == reference_roster, "perspective roster differs")
    for angle in ANGLES:
        _require(len(input_diagnostics[angle]) == EXPECTED_SAMPLES, "perspective input diagnostics differ")

    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, dict[str, Any]] = {model: {} for model in MODELS}
    for model in MODELS:
        for angle in ANGLES:
            seed_nmae: list[float] = []
            seed_acc: list[float] = []
            seed_p95: list[float] = []
            pooled: list[float] = []
            per_seed_json: dict[str, Any] = {}
            for seed in SEEDS:
                values = list(errors[model][seed][angle].values())
                nmae = _mean(values)
                acc = _mean([float(value <= 0.02) for value in values])
                p95 = _percentile(values, 0.95)
                seed_nmae.append(nmae)
                seed_acc.append(acc)
                seed_p95.append(p95)
                pooled.extend(values)
                seed_rows.append(
                    {
                        "model": model,
                        "angle": angle,
                        "seed": seed,
                        "rows": len(values),
                        "nmae": nmae,
                        "acc_at_2_percent": acc,
                        "p95_absolute_error": p95,
                    }
                )
                per_seed_json[str(seed)] = {
                    "nmae": nmae,
                    "acc_at_2_percent": acc,
                    "p95_absolute_error": p95,
                }
            diagnostics = list(input_diagnostics[angle].values())
            availability_rate = _mean([float(value[0]) for value in diagnostics])
            fallback_rate = _mean([float(value[1]) for value in diagnostics])
            mean_support = _mean([float(value[2]) for value in diagnostics])
            aggregate = {
                "pooled_nmae": _mean(pooled),
                "seed_nmae_mean": _mean(seed_nmae),
                "seed_nmae_sd": _sample_sd(seed_nmae),
                "pooled_acc_at_2_percent": _mean([float(value <= 0.02) for value in pooled]),
                "seed_acc_at_2_percent_mean": _mean(seed_acc),
                "seed_acc_at_2_percent_sd": _sample_sd(seed_acc),
                "pooled_p95_absolute_error": _percentile(pooled, 0.95),
                "seed_p95_absolute_error_mean": _mean(seed_p95),
                "seed_p95_absolute_error_sd": _sample_sd(seed_p95),
                "support_normalization_availability_rate": availability_rate,
                "identity_fallback_rate": fallback_rate,
                "mean_valid_support_fraction": mean_support,
            }
            summary_rows.append(
                {
                    "model": model,
                    "angle": angle,
                    "rows_per_seed": EXPECTED_SAMPLES,
                    "pooled_rows": len(pooled),
                    **aggregate,
                }
            )
            summary_json[model][str(angle)] = {**aggregate, "per_seed": per_seed_json}

    paired_rows: list[dict[str, Any]] = []
    paired_json: dict[str, Any] = {}
    _require(reference_roster is not None, "perspective roster is missing")
    for angle in ANGLES:
        per_seed_deltas: dict[int, dict[tuple[str, str], float]] = {}
        seed_means: list[float] = []
        per_seed_json: dict[str, float] = {}
        for seed in SEEDS:
            delta = {
                key: errors[FULL_MODEL][seed][angle][key] - errors[RAW_EFFICIENTNET_MODEL][seed][angle][key]
                for key in reference_roster
            }
            per_seed_deltas[seed] = delta
            value = _mean(list(delta.values()))
            seed_means.append(value)
            per_seed_json[str(seed)] = value
        averaged = [
            (key[0], _mean([per_seed_deltas[seed][key] for seed in SEEDS]))
            for key in sorted(reference_roster)
        ]
        bootstrap = _cluster_bootstrap_mean_ci(averaged)
        _require(abs(_mean(seed_means) - bootstrap["point_estimate"]) <= 1e-15, "paired point estimates differ")
        result = {
            "paired_difference": "Full GeoPRR absolute error minus Raw EfficientNet-B0 absolute error",
            "per_seed": per_seed_json,
            "seed_mean": _mean(seed_means),
            "seed_sd": _sample_sd(seed_means),
            "scene_cluster_bootstrap": bootstrap,
        }
        paired_json[str(angle)] = result
        paired_rows.append(
            {
                "angle": angle,
                "difference_direction": "Full GeoPRR minus Raw EfficientNet-B0",
                "seed_mean": result["seed_mean"],
                "seed_sd": result["seed_sd"],
                "ci_95_lower": bootstrap["ci_95_lower"],
                "ci_95_upper": bootstrap["ci_95_upper"],
                "bootstrap_replicates": bootstrap["replicates"],
                "scene_clusters": bootstrap["clusters"],
                **{f"seed_{seed}": per_seed_json[str(seed)] for seed in SEEDS},
            }
        )
    _write_csv(
        output_dir / "perspective_scan_seed_metrics.csv",
        ("model", "angle", "seed", "rows", "nmae", "acc_at_2_percent", "p95_absolute_error"),
        seed_rows,
    )
    _write_csv(output_dir / "perspective_scan_summary.csv", tuple(summary_rows[0]), summary_rows)
    _write_csv(output_dir / "perspective_full_vs_raw.csv", tuple(paired_rows[0]), paired_rows)
    return {
        "rows": written,
        "samples": EXPECTED_SAMPLES,
        "scenes": EXPECTED_SCENES,
        "angles": list(ANGLES),
        "models": list(MODELS),
        "summary": summary_json,
        "full_vs_raw_paired_difference": paired_json,
    }


def _write_readme(output_dir: Path) -> None:
    text = """# GeoPRR-Net Figure 3 experiment data

This directory contains the complete sample-level ledgers and derived statistics for the Geometry x Routing factorial (Figure 3a) and perspective/fallback scan (Figure 3d).

- `geometry_routing_per_sample.csv.gz`: four cells x three seeds x 9,348 rows (1,558 images x six conditions).
- `geometry_routing_seed_metrics.csv`: NMAE and Acc@2% for every cell and seed.
- `geometry_routing_summary.csv`: pooled metrics and seed mean +/- sample SD.
- `geometry_routing_conditions.csv`: six-condition NMAE and Acc@2% summaries.
- `geometry_routing_interaction.json`: prespecified difference-in-differences interaction and the 20,000-resample, 14-scene cluster-bootstrap 95% CI.
- `perspective_scan_per_sample.csv.gz`: three models x three seeds x 1,558 images x six angles; no high-angle row is filtered.
- `perspective_scan_seed_metrics.csv`: per-angle, per-model, per-seed metrics.
- `perspective_scan_summary.csv`: NMAE, Acc@2%, P95 error, support-normalization availability, identity fallback, and support fraction.
- `perspective_full_vs_raw.csv`: paired Full GeoPRR minus Raw EfficientNet-B0 error differences with scene-bootstrap 95% CIs.
- `figure3_experiment_summary.json`: machine-readable consolidated results and intervention audit.

NMAE is mean absolute error on the normalized full-scale target. Acc@2% counts absolute error <= 0.02. SD is the sample SD over seeds. Bootstrap resampling treats each of the 14 scenes as a cluster and retains the complete image/condition denominator. The 25- and 45-degree pixels use the same deterministic axis/sign assignment and exact projective transform as the existing formal SyncG conditions.

`valid_support_fraction` is the geometric fraction of the projectively warped source plane remaining inside the canvas. It is reported independently of the all-ones effective model mask used when identity fallback is active.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def summarize_figure3(
    *,
    existing_root: Path,
    joint_root: Path,
    perspective_ledger: Path,
    perspective_metadata: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"output directory already exists: {output}")
    output.mkdir(parents=True)
    audit = _audit_factorial_checkpoints(existing_root)
    factorial = _summarize_factorial(
        existing_root=existing_root,
        joint_root=joint_root,
        output_dir=output,
    )
    perspective = _summarize_perspective(
        ledger_path=perspective_ledger,
        metadata_path=perspective_metadata,
        output_dir=output,
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "formal_syncg_images": EXPECTED_SAMPLES,
            "formal_syncg_scenes": EXPECTED_SCENES,
            "seeds": list(SEEDS),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "all_high_angle_rows_retained": True,
            "training_or_adaptation_during_evaluation": False,
        },
        "factorial_checkpoint_intervention_audit": audit,
        "geometry_routing_factorial": factorial,
        "perspective_fallback_scan": perspective,
    }
    _write_json(output / "figure3_experiment_summary.json", result)
    _write_readme(output)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=Path("artifacts/runs/unified_pointer_reader"),
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=Path("artifacts/runs/geoprr_figure3_geometry_routing"),
    )
    parser.add_argument("--perspective-ledger", type=Path, required=True)
    parser.add_argument("--perspective-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = summarize_figure3(
        existing_root=args.existing_root,
        joint_root=args.joint_root,
        perspective_ledger=args.perspective_ledger,
        perspective_metadata=args.perspective_metadata,
        output_dir=args.output_dir,
    )
    interaction = result["geometry_routing_factorial"]["interaction"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output_dir).resolve()),
                "factorial_rows": result["geometry_routing_factorial"]["rows"],
                "perspective_rows": result["perspective_fallback_scan"]["rows"],
                "interaction": interaction["seed_mean"],
                "interaction_ci_95": [
                    interaction["scene_cluster_bootstrap"]["ci_95_lower"],
                    interaction["scene_cluster_bootstrap"]["ci_95_upper"],
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "PROTOCOL",
    "_cluster_bootstrap_mean_ci",
    "summarize_figure3",
]
