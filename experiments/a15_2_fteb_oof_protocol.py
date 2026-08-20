"""Target-blind five-fold scene-wise OOF protocol for A15.2.

Only the 6,616-row, 60-scene A13 correction-train population is eligible.
The five 12-scene evaluation folds are fixed by a local seeded shuffle before
targets or endpoint predictions are inspected.  Every scene contributes the
17th--24th relation-complete candidates in the unchanged A15 candidate order,
so no physical candidate used by the A15 or A15.1 evaluation roles is reused.

Each fold trains one fresh fixed-quarter A15.2 full model on 48 scenes and
evaluates it on the other 12.  q0, direct q_sarn, and the fixed geometric
endpoint are analytic references; an endpoint-null arm is not retrained.
References are descriptive and never control execution or advancement.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from experiments.a15_fteb_inner_scene_probe_protocol import (
    ACCESS_EXPECTATION,
    CORRECTION_TRAIN_SCENE_STEMS,
    DESCRIPTIVE_REFERENCE as A15_DESCRIPTIVE_REFERENCE,
    EVAL_CACHE_ROWS,
    EVAL_PHYSICAL_SAMPLES,
    METHOD_DIRECT_QS,
    METHOD_FIXED_GEOMETRIC,
    METHOD_Q0,
    PHYSICAL_BATCH_SIZE,
    PHYSICAL_PRESENTATIONS,
    PHYSICAL_SAMPLES_PER_SCENE,
    PIXEL_CONDITION_SEED,
    PROJECTIVE_CONDITIONS,
    ROW_BATCH_SIZE,
    SAMPLE_SELECTION_SEED,
    SHARED_INITIALIZATION_SEED,
    SOURCE_SAMPLES,
    SOURCE_SCENES,
    TRAIN_PHYSICAL_SAMPLES,
    TRAIN_STEPS,
    paired_prediction_summary,
    schedule_presentation_counts,
)


PROTOCOL: Final[str] = "syncg_a15_2_fteb_scene_oof5_fixed_quarter_v1"
SCIENTIFIC_QUESTION: Final[str] = (
    "Across target-blind five-fold scene-wise OOF predictions on all 60 "
    "correction-train scenes, does a fixed 0.25 learned-residual A15.2 retain "
    "the A15 tail benefit while improving on both q0 and fixed geometry?"
)
DEVELOPMENT_SCOPE: Final[str] = (
    "A13_physical_correction_train_6616_60scenes_only;_correction_dev_"
    "Core_audit_FoldA_FoldB_formal_and_field_are_not_inputs"
)

OOF_FOLDS: Final[int] = 5
OOF_SCENE_SPLIT_SEED: Final[int] = 20_262_218
BATCH_SCHEDULE_SEED: Final[int] = 20_262_219
RELATION_COMPLETE_START: Final[int] = 16
RELATION_COMPLETE_RANKS: Final[str] = "17-24"

METHOD_A15_2_LEARNED: Final[str] = "a15_2_fixed_quarter_full"
METHOD_ORDER: Final[tuple[str, ...]] = (
    METHOD_Q0,
    METHOD_DIRECT_QS,
    METHOD_FIXED_GEOMETRIC,
    METHOD_A15_2_LEARNED,
)

DESCRIPTIVE_REFERENCE: Final[dict[str, Any]] = dict(
    A15_DESCRIPTIVE_REFERENCE
)


class A152OOFProtocolError(ValueError):
    """The A15.2 scene folds, schedule, or OOF metric input is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152OOFProtocolError(message)


@dataclass(frozen=True, slots=True)
class OOFSceneFold:
    fold_index: int
    train_scenes: tuple[str, ...]
    eval_scenes: tuple[str, ...]
    schedule_seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fold_schedule_seed(fold_index: int) -> int:
    index = int(fold_index)
    _require(0 <= index < OOF_FOLDS, "A15.2 fold index is out of range")
    # The positional training cache differs by scene fold, but the optimizer
    # presentation generator is intentionally identical across folds so fold
    # identity is not confounded with a second schedule stream.
    return BATCH_SCHEDULE_SEED


def fixed_scene_folds() -> tuple[OOFSceneFold, ...]:
    """Return the fixed target- and prediction-blind five scene folds."""

    scenes = list(CORRECTION_TRAIN_SCENE_STEMS)
    random.Random(OOF_SCENE_SPLIT_SEED).shuffle(scenes)
    folds: list[OOFSceneFold] = []
    for fold_index in range(OOF_FOLDS):
        start = fold_index * 12
        eval_scenes = tuple(sorted(scenes[start : start + 12]))
        eval_set = set(eval_scenes)
        train_scenes = tuple(
            sorted(scene for scene in scenes if scene not in eval_set)
        )
        folds.append(
            OOFSceneFold(
                fold_index=fold_index,
                train_scenes=train_scenes,
                eval_scenes=eval_scenes,
                schedule_seed=fold_schedule_seed(fold_index),
            )
        )
    return tuple(folds)


SCENE_FOLDS: Final[tuple[OOFSceneFold, ...]] = fixed_scene_folds()
RELATION_COMPLETE_START_BY_SCENE: Final[dict[str, int]] = {
    scene: RELATION_COMPLETE_START for scene in CORRECTION_TRAIN_SCENE_STEMS
}


def cache_materialization_partition() -> dict[str, tuple[str, ...]]:
    """Return one 48/12 roster used only to cache all 60 scenes once."""

    first = SCENE_FOLDS[0]
    return {
        "inner_train": first.train_scenes,
        "inner_eval": first.eval_scenes,
    }


def fixed_fold_physical_batch_schedule(
    fold_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the predeclared 200x8 physical schedule for one fold."""

    rng = random.Random(fold_schedule_seed(fold_index))
    flat: list[int] = []
    while len(flat) < PHYSICAL_PRESENTATIONS:
        cycle = list(range(TRAIN_PHYSICAL_SAMPLES))
        rng.shuffle(cycle)
        flat.extend(cycle)
    flat = flat[:PHYSICAL_PRESENTATIONS]
    schedule = tuple(
        tuple(flat[offset : offset + PHYSICAL_BATCH_SIZE])
        for offset in range(0, PHYSICAL_PRESENTATIONS, PHYSICAL_BATCH_SIZE)
    )
    _require(len(schedule) == TRAIN_STEPS, "A15.2 schedule step count differs")
    _require(
        all(
            len(batch) == PHYSICAL_BATCH_SIZE
            and len(set(batch)) == PHYSICAL_BATCH_SIZE
            for batch in schedule
        ),
        "A15.2 schedule batch differs",
    )
    return schedule


def _finite_vector(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    _require(bool(result), f"A15.2 {label} is empty")
    _require(
        all(value == value and abs(value) != float("inf") for value in result),
        f"A15.2 {label} contains a non-finite value",
    )
    return result


def _partition_metric_report(
    *, predictions: Mapping[str, Sequence[float]], target: Sequence[float]
) -> dict[str, Any]:
    _require(set(predictions) == set(METHOD_ORDER), "A15.2 methods differ")
    truth = _finite_vector(target, label="target")
    values = {
        method: _finite_vector(predictions[method], label=f"{method} prediction")
        for method in METHOD_ORDER
    }
    _require(
        all(len(value) == len(truth) for value in values.values()),
        "A15.2 prediction length differs",
    )
    return {
        "rows": len(truth),
        "versus_q0": {
            method: paired_prediction_summary(
                candidate_mean=values[method],
                reference_mean=values[METHOD_Q0],
                target=truth,
            )
            for method in METHOD_ORDER
        },
        "learned_comparisons": {
            "learned_minus_fixed_geometric": paired_prediction_summary(
                candidate_mean=values[METHOD_A15_2_LEARNED],
                reference_mean=values[METHOD_FIXED_GEOMETRIC],
                target=truth,
            )
        },
    }


def oof_metric_report(
    *,
    predictions: Mapping[str, Sequence[float]],
    target: Sequence[float],
    condition_names: Sequence[str],
) -> dict[str, Any]:
    """Report four methods pooled and by condition for a fold or all OOF rows."""

    truth = _finite_vector(target, label="OOF target")
    conditions = tuple(str(value) for value in condition_names)
    _require(
        len(truth) == len(conditions)
        and len(truth) in {EVAL_CACHE_ROWS, SOURCE_SCENES * PHYSICAL_SAMPLES_PER_SCENE * 3},
        "A15.2 metrics require one fold or all five OOF folds",
    )
    _require(
        set(conditions) == set(PROJECTIVE_CONDITIONS)
        and len(set(conditions.count(name) for name in PROJECTIVE_CONDITIONS)) == 1,
        "A15.2 condition roster/counts differ",
    )
    values = {
        method: _finite_vector(predictions[method], label=f"{method} prediction")
        for method in METHOD_ORDER
        if method in predictions
    }
    _require(set(values) == set(METHOD_ORDER), "A15.2 methods differ")
    pooled = _partition_metric_report(predictions=values, target=truth)
    per_condition: dict[str, Any] = {}
    for condition in PROJECTIVE_CONDITIONS:
        indices = tuple(
            index for index, observed in enumerate(conditions) if observed == condition
        )
        per_condition[condition] = _partition_metric_report(
            predictions={
                method: tuple(row[index] for index in indices)
                for method, row in values.items()
            },
            target=tuple(truth[index] for index in indices),
        )
    return {
        "condition_rows": len(truth),
        "physical_samples": len(truth) // len(PROJECTIVE_CONDITIONS),
        "method_order": METHOD_ORDER,
        "pooled": pooled,
        "per_condition": per_condition,
        "descriptive_reference": dict(DESCRIPTIVE_REFERENCE),
        "automatic_gate_used": False,
    }


def descriptive_reference_observations(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the six reference comparisons without producing a run decision."""

    pooled = metrics["pooled"]
    versus_q0 = pooled["versus_q0"][METHOD_A15_2_LEARNED]
    versus_fixed = pooled["learned_comparisons"][
        "learned_minus_fixed_geometric"
    ]
    per_condition_deltas = {
        condition: metrics["per_condition"][condition]["versus_q0"]
        [METHOD_A15_2_LEARNED]["nmae_delta_candidate_minus_reference"]
        for condition in PROJECTIVE_CONDITIONS
    }
    observations = {
        "learned_projective_nmae_delta_minus_q0_maximum": {
            "observed": versus_q0["nmae_delta_candidate_minus_reference"],
            "reference": DESCRIPTIVE_REFERENCE[
                "learned_projective_nmae_delta_minus_q0_maximum"
            ],
            "relation": "less_than_or_equal",
        },
        "learned_projective_cvar25_delta_minus_q0_maximum": {
            "observed": versus_q0["cvar25_delta_candidate_minus_reference"],
            "reference": DESCRIPTIVE_REFERENCE[
                "learned_projective_cvar25_delta_minus_q0_maximum"
            ],
            "relation": "less_than_or_equal",
        },
        "learned_projective_net_win_minus_loss_minimum": {
            "observed": versus_q0["net_paired_win_minus_loss"],
            "reference": DESCRIPTIVE_REFERENCE[
                "learned_projective_net_win_minus_loss_minimum"
            ],
            "relation": "greater_than_or_equal",
        },
        "each_condition_nmae_delta_minus_q0_maximum": {
            "observed": per_condition_deltas,
            "observed_maximum": max(per_condition_deltas.values()),
            "reference": DESCRIPTIVE_REFERENCE[
                "each_condition_nmae_delta_minus_q0_maximum"
            ],
            "relation": "all_less_than_or_equal",
        },
        "learned_nmae_delta_minus_fixed_geometric_maximum": {
            "observed": versus_fixed["nmae_delta_candidate_minus_reference"],
            "reference": DESCRIPTIVE_REFERENCE[
                "learned_nmae_delta_minus_fixed_geometric_maximum"
            ],
            "relation": "less_than_or_equal",
        },
        "learned_cvar25_delta_minus_fixed_geometric_maximum": {
            "observed": versus_fixed["cvar25_delta_candidate_minus_reference"],
            "reference": DESCRIPTIVE_REFERENCE[
                "learned_cvar25_delta_minus_fixed_geometric_maximum"
            ],
            "relation": "less_than_or_equal",
        },
    }
    return {
        "descriptive_only": True,
        "automatic_execution_selection_or_advancement_control": False,
        "observations": observations,
    }


_require((SOURCE_SAMPLES, SOURCE_SCENES, OOF_FOLDS) == (6_616, 60, 5), "A15.2 source differs")
_require(
    all(
        len(fold.train_scenes) == 48
        and len(fold.eval_scenes) == 12
        and not (set(fold.train_scenes) & set(fold.eval_scenes))
        and set(fold.train_scenes) | set(fold.eval_scenes)
        == set(CORRECTION_TRAIN_SCENE_STEMS)
        for fold in SCENE_FOLDS
    ),
    "A15.2 fold partition differs",
)
_require(
    sorted(scene for fold in SCENE_FOLDS for scene in fold.eval_scenes)
    == sorted(CORRECTION_TRAIN_SCENE_STEMS),
    "A15.2 each scene must be evaluated exactly once",
)
_require(
    set(RELATION_COMPLETE_START_BY_SCENE) == set(CORRECTION_TRAIN_SCENE_STEMS)
    and set(RELATION_COMPLETE_START_BY_SCENE.values()) == {16},
    "A15.2 relation-complete rank policy differs",
)


__all__ = [
    "ACCESS_EXPECTATION",
    "A152OOFProtocolError",
    "BATCH_SCHEDULE_SEED",
    "DEVELOPMENT_SCOPE",
    "DESCRIPTIVE_REFERENCE",
    "METHOD_A15_2_LEARNED",
    "METHOD_DIRECT_QS",
    "METHOD_FIXED_GEOMETRIC",
    "METHOD_ORDER",
    "METHOD_Q0",
    "OOF_FOLDS",
    "OOF_SCENE_SPLIT_SEED",
    "OOFSceneFold",
    "PHYSICAL_BATCH_SIZE",
    "PIXEL_CONDITION_SEED",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "RELATION_COMPLETE_RANKS",
    "RELATION_COMPLETE_START",
    "RELATION_COMPLETE_START_BY_SCENE",
    "ROW_BATCH_SIZE",
    "SAMPLE_SELECTION_SEED",
    "SCENE_FOLDS",
    "SCIENTIFIC_QUESTION",
    "SHARED_INITIALIZATION_SEED",
    "SOURCE_SAMPLES",
    "SOURCE_SCENES",
    "TRAIN_PHYSICAL_SAMPLES",
    "TRAIN_STEPS",
    "cache_materialization_partition",
    "descriptive_reference_observations",
    "fixed_fold_physical_batch_schedule",
    "fixed_scene_folds",
    "fold_schedule_seed",
    "oof_metric_report",
    "schedule_presentation_counts",
]
