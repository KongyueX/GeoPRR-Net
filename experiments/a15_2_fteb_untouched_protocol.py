"""Fixed, descriptive protocol for the one A15.2 Fold-B confirmation.

This module contains ordinary constants and paired statistics only.  It does
not open the physical Fold-B manifest, load a checkpoint, run a model, select
a result, or authorize another experiment.  The final method is the already
trained terminal A15.2 correction attached to its terminal A11 anchor.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from experiments.a10_pccot_protocol import (
    EVALUATION_CONDITIONS,
    FOLD_B_SAMPLES,
    FOLD_B_SCENES,
    PROJECTIVE_CONDITIONS,
)
from experiments.a15_fteb_inner_scene_probe_protocol import (
    paired_prediction_summary,
)
from experiments.prepare_a10_fold_b_manifest import PROTOCOL as PREPARE_PROTOCOL


PROTOCOL: Final[str] = "syncg_a15_2_terminal_fold_b_untouched_once_v1"
FINAL_METHOD: Final[str] = "a15_2_final"
METHOD_Q0: Final[str] = "q0"
METHOD_QS: Final[str] = "q_sarn_fail_closed"
METHOD_FIXED_GEOMETRIC: Final[str] = "fixed_geometric"
METHOD_A15_2: Final[str] = FINAL_METHOD
METHOD_A11_FULL: Final[str] = "terminal_a11_full_scort"
METHOD_ORDER: Final[tuple[str, ...]] = (
    METHOD_Q0,
    METHOD_QS,
    METHOD_FIXED_GEOMETRIC,
    METHOD_A15_2,
    METHOD_A11_FULL,
)
BOOTSTRAP_SEED: Final[int] = 20_262_224
BOOTSTRAP_ITERATIONS: Final[int] = 10_000
BOOTSTRAP_CONFIDENCE: Final[float] = 0.95
CVaR_TAIL_FRACTION: Final[float] = 0.25

# These contrasts are reported regardless of their signs.  They are not
# thresholds and are never used to choose, retry, or advance a model.
BOOTSTRAP_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (METHOD_A15_2, METHOD_Q0),
    (METHOD_A15_2, METHOD_FIXED_GEOMETRIC),
    (METHOD_A15_2, METHOD_A11_FULL),
)

FINAL_CONFIRMATION_SPEC: Final[dict[str, Any]] = {
    "protocol": PROTOCOL,
    "prepare_protocol": PREPARE_PROTOCOL,
    "physical_fold_b_samples": FOLD_B_SAMPLES,
    "physical_fold_b_scenes": FOLD_B_SCENES,
    "conditions": list(EVALUATION_CONDITIONS),
    "projective_conditions": list(PROJECTIVE_CONDITIONS),
    "method_order": list(METHOD_ORDER),
    "final_method_fixed_before_fold_b_open": FINAL_METHOD,
    "terminal_checkpoint_selection_only": True,
    "result_based_model_selection": False,
    "threshold_gate": False,
    "retry": False,
    "automatic_advancement": False,
    "bootstrap": {
        "unit": "scene",
        "seed": BOOTSTRAP_SEED,
        "iterations": BOOTSTRAP_ITERATIONS,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval": "percentile",
    },
}


class A152UntouchedProtocolError(ValueError):
    """The supplied per-row evidence does not satisfy the fixed protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152UntouchedProtocolError(message)


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute NMAE, CVaR25, and paired W/T/L from serialized rows."""

    values = tuple(rows)
    _require(bool(values), "A15.2 confirmation rows are empty")
    targets: list[float] = []
    predictions = {method: [] for method in METHOD_ORDER}
    for row in values:
        _require(isinstance(row, Mapping), "A15.2 confirmation row is malformed")
        target = float(row.get("normalized_target", float("nan")))
        means = row.get("mean")
        _require(
            math.isfinite(target)
            and isinstance(means, Mapping)
            and set(means) == set(METHOD_ORDER),
            "A15.2 confirmation target or method vector differs",
        )
        targets.append(target)
        for method in METHOD_ORDER:
            value = float(means[method])
            _require(
                math.isfinite(value),
                f"A15.2 confirmation prediction is non-finite: {method}",
            )
            predictions[method].append(value)
        recorded_errors = row.get("absolute_error")
        if recorded_errors is not None:
            _require(
                isinstance(recorded_errors, Mapping)
                and set(recorded_errors) == set(METHOD_ORDER)
                and all(
                    float(recorded_errors[method])
                    == abs(float(means[method]) - target)
                    for method in METHOD_ORDER
                ),
                "A15.2 serialized per-row absolute errors do not recompute",
            )

    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        errors = [
            abs(value - target)
            for value, target in zip(predictions[method], targets, strict=True)
        ]
        tail_count = int(math.ceil(CVaR_TAIL_FRACTION * len(errors)))
        methods[method] = {
            "samples": len(errors),
            "nmae": float(sum(errors) / len(errors)),
            "cvar25": float(
                sum(sorted(errors, reverse=True)[:tail_count]) / tail_count
            ),
            "cvar_tail_fraction": CVaR_TAIL_FRACTION,
            "cvar_tail_count": tail_count,
        }
    versus_q0 = {
        method: paired_prediction_summary(
            candidate_mean=predictions[method],
            reference_mean=predictions[METHOD_Q0],
            target=targets,
        )
        for method in METHOD_ORDER
    }
    final_contrasts = {
        f"{METHOD_A15_2}_minus_{reference}": paired_prediction_summary(
            candidate_mean=predictions[METHOD_A15_2],
            reference_mean=predictions[reference],
            target=targets,
        )
        for reference in (METHOD_Q0, METHOD_FIXED_GEOMETRIC, METHOD_A11_FULL)
    }
    return {
        "rows": len(values),
        "method_order": list(METHOD_ORDER),
        "methods": methods,
        "versus_q0": versus_q0,
        "final_contrasts": final_contrasts,
    }


def _percentile_interval(values: np.ndarray) -> list[float]:
    alpha = 1.0 - BOOTSTRAP_CONFIDENCE
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


def scene_block_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample complete scene blocks for the three fixed final contrasts."""

    values = tuple(rows)
    _require(
        type(iterations) is int and iterations > 0,
        "A15.2 scene bootstrap iterations must be a positive exact integer",
    )
    _require(type(seed) is int, "A15.2 scene bootstrap seed must be an exact integer")
    _require(bool(values), "A15.2 scene bootstrap rows are empty")
    scenes = tuple(str(row.get("scene_stem", "")) for row in values)
    _require(all(scenes), "A15.2 scene bootstrap has an empty scene identity")
    scene_order = tuple(sorted(set(scenes)))
    _require(len(scene_order) >= 2, "A15.2 scene bootstrap needs at least two scenes")
    blocks = tuple(
        np.asarray(
            [index for index, observed in enumerate(scenes) if observed == scene],
            dtype=np.int64,
        )
        for scene in scene_order
    )
    _require(
        all(block.size > 0 for block in blocks)
        and sum(int(block.size) for block in blocks) == len(values),
        "A15.2 scene bootstrap block partition differs",
    )

    target = np.asarray(
        [float(row["normalized_target"]) for row in values], dtype=np.float64
    )
    predictions = {
        method: np.asarray(
            [float(row["mean"][method]) for row in values], dtype=np.float64
        )
        for method in METHOD_ORDER
    }
    _require(
        np.isfinite(target).all()
        and all(np.isfinite(vector).all() for vector in predictions.values()),
        "A15.2 scene bootstrap input is non-finite",
    )
    errors = {
        method: np.abs(predictions[method] - target) for method in METHOD_ORDER
    }
    point = summarize_prediction_rows(values)
    contrast_names = tuple(
        f"{candidate}_minus_{reference}"
        for candidate, reference in BOOTSTRAP_CONTRASTS
    )
    distributions = {
        name: {
            "nmae": np.empty(iterations, dtype=np.float64),
            "cvar25": np.empty(iterations, dtype=np.float64),
            "net_paired_win_minus_loss": np.empty(iterations, dtype=np.float64),
        }
        for name in contrast_names
    }
    rng = np.random.default_rng(seed)
    scene_count = len(scene_order)
    for iteration in range(iterations):
        draw = rng.integers(0, scene_count, size=scene_count)
        indices = np.concatenate([blocks[int(index)] for index in draw])
        tail_count = int(math.ceil(CVaR_TAIL_FRACTION * len(indices)))
        for candidate, reference in BOOTSTRAP_CONTRASTS:
            name = f"{candidate}_minus_{reference}"
            candidate_error = errors[candidate][indices]
            reference_error = errors[reference][indices]
            delta = candidate_error - reference_error
            distributions[name]["nmae"][iteration] = float(
                candidate_error.mean() - reference_error.mean()
            )
            candidate_tail = np.partition(
                candidate_error, len(candidate_error) - tail_count
            )[-tail_count:]
            reference_tail = np.partition(
                reference_error, len(reference_error) - tail_count
            )[-tail_count:]
            distributions[name]["cvar25"][iteration] = float(
                candidate_tail.mean() - reference_tail.mean()
            )
            wins = int((delta < -1.0e-12).sum())
            losses = int((delta > 1.0e-12).sum())
            distributions[name]["net_paired_win_minus_loss"][iteration] = (
                wins - losses
            ) / len(indices)

    contrasts: dict[str, Any] = {}
    for candidate, reference in BOOTSTRAP_CONTRASTS:
        name = f"{candidate}_minus_{reference}"
        paired = point["final_contrasts"][name]
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "direction": "candidate_minus_reference_lower_nmae_and_cvar_are_better",
            "point_estimate": {
                "nmae_delta": float(
                    paired["nmae_delta_candidate_minus_reference"]
                ),
                "cvar25_delta": float(
                    paired["cvar25_delta_candidate_minus_reference"]
                ),
                "net_paired_win_minus_loss": float(
                    paired["net_paired_win_minus_loss"]
                ),
            },
            "scene_percentile_95ci": {
                metric: _percentile_interval(distribution)
                for metric, distribution in distributions[name].items()
            },
        }
    return {
        "seed": seed,
        "iterations": iterations,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval": "percentile",
        "cluster_unit": "scene",
        "clusters": scene_count,
        "cluster_roster": list(scene_order),
        "rows": len(values),
        "complete_scene_blocks_resampled": True,
        "shared_scene_draws_across_contrasts": True,
        "hypothesis_test_or_automatic_gate_used": False,
        "contrasts": contrasts,
    }


__all__ = [
    "A152UntouchedProtocolError",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_CONTRASTS",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "FINAL_CONFIRMATION_SPEC",
    "FINAL_METHOD",
    "METHOD_A11_FULL",
    "METHOD_A15_2",
    "METHOD_FIXED_GEOMETRIC",
    "METHOD_ORDER",
    "METHOD_Q0",
    "METHOD_QS",
    "PROTOCOL",
    "scene_block_bootstrap",
    "summarize_prediction_rows",
]
