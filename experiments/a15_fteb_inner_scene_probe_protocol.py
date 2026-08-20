"""Target-blind inner-scene protocol for the A15 FTEB matched probe.

Only the 6,616-row, 60-scene A13 correction-train population is eligible.
The scene split and per-scene candidate orders use local seeded RNG streams;
neither targets nor any model prediction participates.  A physical sample is
selected only when the same cached Raw/SARN pair is relation-available under
all three projective conditions.  Training and evaluation therefore use the
same eight physical samples per scene and the same three-condition expansion.

The learned full and endpoint-null arms start from the same correction state
and consume the same 200 physical-group batches.  Endpoint-null changes only
the posterior endpoint supplied to FTEB (q_sarn := q0); aligned SARN features,
homography, support, topology, optimizer, and sample order remain unchanged.
All reported references are descriptive and do not gate another experiment.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from experiments.a13_correction_dev_protocol import (
    CORRECTION_TRAIN_SAMPLES,
    CORRECTION_TRAIN_SCENES,
)
from experiments.a15_fteb import PROBABILITY_EPSILON


PROTOCOL: Final[str] = "syncg_a15_fteb_inner_scene48_12_matched_probe_v1"
SCIENTIFIC_QUESTION: Final[str] = (
    "On correction-train-only unseen scenes, does the learned per-bin FTEB "
    "improve on both frozen q0 and its untrained fixed-geometric endpoint "
    "anchor, and is that improvement attributable to the q_sarn posterior "
    "endpoint rather than only aligned SARN image features?"
)
DEVELOPMENT_SCOPE: Final[str] = (
    "A13_physical_correction_train_6616_60scenes_only;_correction_dev_"
    "Core_audit_FoldA_FoldB_formal_and_field_are_not_inputs"
)

SOURCE_SAMPLES: Final[int] = CORRECTION_TRAIN_SAMPLES
CORRECTION_TRAIN_SCENE_STEMS: Final[tuple[str, ...]] = tuple(
    sorted(Path(str(scene)).stem for scene in CORRECTION_TRAIN_SCENES)
)
SOURCE_SCENES: Final[int] = len(CORRECTION_TRAIN_SCENE_STEMS)
INNER_TRAIN_SCENE_COUNT: Final[int] = 48
INNER_EVAL_SCENE_COUNT: Final[int] = 12
PHYSICAL_SAMPLES_PER_SCENE: Final[int] = 8

PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
CONDITIONS_PER_PHYSICAL: Final[int] = len(PROJECTIVE_CONDITIONS)

TRAIN_PHYSICAL_SAMPLES: Final[int] = (
    INNER_TRAIN_SCENE_COUNT * PHYSICAL_SAMPLES_PER_SCENE
)
EVAL_PHYSICAL_SAMPLES: Final[int] = (
    INNER_EVAL_SCENE_COUNT * PHYSICAL_SAMPLES_PER_SCENE
)
TRAIN_CACHE_ROWS: Final[int] = TRAIN_PHYSICAL_SAMPLES * CONDITIONS_PER_PHYSICAL
EVAL_CACHE_ROWS: Final[int] = EVAL_PHYSICAL_SAMPLES * CONDITIONS_PER_PHYSICAL

PHYSICAL_BATCH_SIZE: Final[int] = 8
ROW_BATCH_SIZE: Final[int] = PHYSICAL_BATCH_SIZE * CONDITIONS_PER_PHYSICAL
TRAIN_STEPS: Final[int] = 200
PHYSICAL_PRESENTATIONS: Final[int] = TRAIN_STEPS * PHYSICAL_BATCH_SIZE
ROW_PRESENTATIONS: Final[int] = TRAIN_STEPS * ROW_BATCH_SIZE

# Independent local RNG streams, fixed before any A15 probe result.
SCENE_SPLIT_SEED: Final[int] = 20_262_211
SAMPLE_SELECTION_SEED: Final[int] = 20_262_212
PIXEL_CONDITION_SEED: Final[int] = 20_262_213
BATCH_SCHEDULE_SEED: Final[int] = 20_262_214
SHARED_INITIALIZATION_SEED: Final[int] = 20_262_215

METHOD_Q0: Final[str] = "q0"
METHOD_DIRECT_QS: Final[str] = "direct_q_sarn"
METHOD_FIXED_GEOMETRIC: Final[str] = "fixed_geometric"
METHOD_A15_LEARNED: Final[str] = "a15_learned_full"
METHOD_ENDPOINT_NULL: Final[str] = "a15_endpoint_null"
METHOD_ORDER: Final[tuple[str, ...]] = (
    METHOD_Q0,
    METHOD_DIRECT_QS,
    METHOD_FIXED_GEOMETRIC,
    METHOD_A15_LEARNED,
    METHOD_ENDPOINT_NULL,
)
PAIR_TIE_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12
CVAR_TAIL_FRACTION: Final[float] = 0.25


class A15InnerSceneProbeProtocolError(ValueError):
    """The A15 inner-scene roster, schedule, or metric input is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A15InnerSceneProbeProtocolError(message)


def _fixed_scene_rosters() -> tuple[tuple[str, ...], tuple[str, ...]]:
    scenes = list(CORRECTION_TRAIN_SCENE_STEMS)
    _require(
        len(scenes) == SOURCE_SCENES and len(set(scenes)) == SOURCE_SCENES,
        "A15 source scene roster differs from correction-train",
    )
    random.Random(SCENE_SPLIT_SEED).shuffle(scenes)
    inner_eval = tuple(sorted(scenes[:INNER_EVAL_SCENE_COUNT]))
    inner_train = tuple(sorted(scenes[INNER_EVAL_SCENE_COUNT:]))
    return inner_train, inner_eval


INNER_TRAIN_SCENES, INNER_EVAL_SCENES = _fixed_scene_rosters()


ACCESS_EXPECTATION: Final[dict[str, bool]] = {
    "correction_train_manifest_access": True,
    "terminal_a11_train_checkpoint_access": True,
    "correction_dev_manifest_access": False,
    "core_audit_manifest_access": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}

ENDPOINT_CACHE_SEMANTICS: Final[dict[str, Any]] = {
    "terminal_a11_mode": "eval_no_grad_frozen",
    "shared_anchor_weights": True,
    "raw_and_sarn_forward_calls": "separate_not_concatenated",
    "canonical_q0": "raw_only_forward_bits",
    "cached_endpoint_tensors": (
        "q0",
        "q_sarn",
        "raw_stride8",
        "raw_stride16",
        "sarn_stride8",
        "sarn_stride16",
        "support",
        "raw_to_sarn_homography",
    ),
    "fixed_geometric_recomputed_on_inner_eval": True,
    "prior_concat_endpoint_analysis_used_as_inner_eval_result": False,
}

MATCHED_ARM_SEMANTICS: Final[dict[str, Any]] = {
    "arms": (METHOD_A15_LEARNED, METHOD_ENDPOINT_NULL),
    "same_initial_correction_state": True,
    "same_optimizer_and_200_batch_sequence": True,
    "full_posterior_endpoint": "q_sarn",
    "endpoint_null_posterior_endpoint": "q0",
    "endpoint_null_retains": (
        "aligned_sarn_stride8",
        "aligned_sarn_stride16",
        "homography",
        "support",
        "relation_topology",
    ),
    "treatment_difference": "posterior_endpoint_q_sarn_versus_q0",
}

FIXED_GEOMETRIC_SEMANTICS: Final[dict[str, Any]] = {
    "helper": "experiments.a15_fteb.fixed_geometric_natural_parameter_base",
    "dtype": "float32",
    "clamp_min": "torch.float32_finfo_tiny",
    "clamp_min_value": PROBABILITY_EPSILON,
    "formula": "softmax(0.5*(log(clamp(q0))+log(clamp(q_sarn))))",
}

DESCRIPTIVE_REFERENCE: Final[dict[str, Any]] = {
    "learned_projective_nmae_delta_minus_q0_maximum": -0.001,
    "learned_projective_cvar25_delta_minus_q0_maximum": -0.001,
    "learned_projective_net_win_minus_loss_minimum": 0.03,
    "each_condition_nmae_delta_minus_q0_maximum": 0.0002,
    "learned_nmae_delta_minus_fixed_geometric_maximum": 0.0,
    "learned_cvar25_delta_minus_fixed_geometric_maximum": 0.0,
    "automatic_execution_selection_or_advancement_control": False,
}


@dataclass(frozen=True, slots=True)
class PhysicalSampleRef:
    source_index: int
    sample_id: str
    scene_stem: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def fixed_scene_partition() -> dict[str, tuple[str, ...]]:
    """Return the target- and output-blind 48/12 scene partition."""

    return {
        "inner_train": INNER_TRAIN_SCENES,
        "inner_eval": INNER_EVAL_SCENES,
    }


def per_scene_candidate_order(
    *,
    sample_ids: Sequence[str],
    scene_stems: Sequence[str],
) -> dict[str, tuple[PhysicalSampleRef, ...]]:
    """Create target-blind candidate orders from IDs and scene metadata only."""

    ids = tuple(str(value) for value in sample_ids)
    scenes = tuple(Path(str(value)).stem for value in scene_stems)
    _require(
        len(ids) == len(scenes) == SOURCE_SAMPLES,
        "A15 candidate inventory must contain 6616 physical rows",
    )
    _require(all(ids) and len(ids) == len(set(ids)), "A15 sample IDs differ")
    _require(
        set(scenes) == set(CORRECTION_TRAIN_SCENE_STEMS),
        "A15 candidate scene roster differs from correction-train",
    )
    by_scene: dict[str, list[PhysicalSampleRef]] = defaultdict(list)
    for source_index, (sample_id, scene_stem) in enumerate(
        zip(ids, scenes, strict=True)
    ):
        by_scene[scene_stem].append(
            PhysicalSampleRef(
                source_index=source_index,
                sample_id=sample_id,
                scene_stem=scene_stem,
            )
        )
    _require(
        all(len(by_scene[scene]) >= PHYSICAL_SAMPLES_PER_SCENE for scene in scenes),
        "A15 source scene has fewer than eight physical rows",
    )
    rng = random.Random(SAMPLE_SELECTION_SEED)
    ordered: dict[str, tuple[PhysicalSampleRef, ...]] = {}
    for scene in sorted(by_scene):
        candidates = list(by_scene[scene])
        rng.shuffle(candidates)
        ordered[scene] = tuple(candidates)
    return ordered


def select_relation_complete_physical_samples(
    candidate_order: Mapping[str, Sequence[PhysicalSampleRef]],
    relation_available: Mapping[tuple[int, str], bool],
) -> dict[str, tuple[PhysicalSampleRef, ...]]:
    """Take eight samples/scene available under every projective condition.

    ``relation_available`` is the sole model-derived selection input.  The
    caller must not pass targets, losses, predictions, or error ranks.
    """

    _require(
        set(candidate_order) == set(CORRECTION_TRAIN_SCENE_STEMS),
        "A15 candidate-order scene roster differs",
    )
    selected_by_scene: dict[str, tuple[PhysicalSampleRef, ...]] = {}
    for scene in sorted(candidate_order):
        selected: list[PhysicalSampleRef] = []
        seen_indices: set[int] = set()
        for candidate in candidate_order[scene]:
            _require(
                candidate.scene_stem == scene
                and candidate.source_index not in seen_indices,
                "A15 candidate scene/index differs",
            )
            seen_indices.add(candidate.source_index)
            keys = tuple(
                (candidate.source_index, condition)
                for condition in PROJECTIVE_CONDITIONS
            )
            _require(
                all(key in relation_available for key in keys),
                "A15 relation availability is incomplete for a physical candidate",
            )
            if all(bool(relation_available[key]) for key in keys):
                selected.append(candidate)
                if len(selected) == PHYSICAL_SAMPLES_PER_SCENE:
                    break
        _require(
            len(selected) == PHYSICAL_SAMPLES_PER_SCENE,
            f"A15 scene has fewer than eight all-condition relation rows: {scene}",
        )
        selected_by_scene[scene] = tuple(selected)

    train = tuple(
        candidate
        for scene in INNER_TRAIN_SCENES
        for candidate in selected_by_scene[scene]
    )
    inner_eval = tuple(
        candidate
        for scene in INNER_EVAL_SCENES
        for candidate in selected_by_scene[scene]
    )
    _require(len(train) == TRAIN_PHYSICAL_SAMPLES, "A15 train selection differs")
    _require(len(inner_eval) == EVAL_PHYSICAL_SAMPLES, "A15 eval selection differs")
    _require(
        not ({row.source_index for row in train} & {row.source_index for row in inner_eval}),
        "A15 train/eval physical rows overlap",
    )
    return {"inner_train": train, "inner_eval": inner_eval}


def expand_physical_indices_to_cache_rows(
    physical_indices: Sequence[int],
) -> tuple[int, ...]:
    """Expand physical-major cache indices in fixed three-condition order."""

    indices = tuple(int(value) for value in physical_indices)
    _require(
        len(indices) == len(set(indices)),
        "A15 physical batch repeats a sample",
    )
    _require(
        all(0 <= value < TRAIN_PHYSICAL_SAMPLES for value in indices),
        "A15 physical schedule index is outside train cache",
    )
    return tuple(
        physical * CONDITIONS_PER_PHYSICAL + condition_index
        for physical in indices
        for condition_index in range(CONDITIONS_PER_PHYSICAL)
    )


def fixed_physical_batch_schedule() -> tuple[tuple[int, ...], ...]:
    """Return the shared 200x8 physical-group training order."""

    rng = random.Random(BATCH_SCHEDULE_SEED)
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
    _require(len(schedule) == TRAIN_STEPS, "A15 schedule step count differs")
    _require(
        all(
            len(batch) == PHYSICAL_BATCH_SIZE and len(set(batch)) == len(batch)
            for batch in schedule
        ),
        "A15 physical batch width or uniqueness differs",
    )
    return schedule


def schedule_presentation_counts(
    schedule: Sequence[Sequence[int]],
) -> dict[str, dict[int, int]]:
    """Count physical and expanded-row presentations in a complete schedule."""

    batches = tuple(tuple(int(index) for index in batch) for batch in schedule)
    _require(len(batches) == TRAIN_STEPS, "A15 schedule must contain 200 steps")
    physical_counts = {index: 0 for index in range(TRAIN_PHYSICAL_SAMPLES)}
    row_counts = {index: 0 for index in range(TRAIN_CACHE_ROWS)}
    for batch in batches:
        _require(
            len(batch) == PHYSICAL_BATCH_SIZE and len(set(batch)) == len(batch),
            "A15 schedule batch differs",
        )
        for index in batch:
            _require(index in physical_counts, "A15 schedule index is out of range")
            physical_counts[index] += 1
        for row_index in expand_physical_indices_to_cache_rows(batch):
            row_counts[row_index] += 1
    _require(
        sum(physical_counts.values()) == PHYSICAL_PRESENTATIONS
        and sum(row_counts.values()) == ROW_PRESENTATIONS,
        "A15 presentation count differs",
    )
    return {"physical": physical_counts, "rows": row_counts}


def _finite_vector(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    _require(bool(result), f"A15 {label} is empty")
    _require(
        all(math.isfinite(value) for value in result),
        f"A15 {label} contains a non-finite value",
    )
    return result


def paired_prediction_summary(
    *,
    candidate_mean: Sequence[float],
    reference_mean: Sequence[float],
    target: Sequence[float],
) -> dict[str, Any]:
    """Summarize normalized error, independent CVaR tails, and paired W/T/L."""

    candidate = _finite_vector(candidate_mean, label="candidate prediction")
    reference = _finite_vector(reference_mean, label="reference prediction")
    truth = _finite_vector(target, label="target")
    _require(
        len(candidate) == len(reference) == len(truth),
        "A15 paired metric lengths differ",
    )
    candidate_error = tuple(abs(value - expected) for value, expected in zip(candidate, truth, strict=True))
    reference_error = tuple(abs(value - expected) for value, expected in zip(reference, truth, strict=True))
    wins = ties = losses = 0
    for candidate_value, reference_value in zip(candidate_error, reference_error, strict=True):
        delta = candidate_value - reference_value
        if delta < -PAIR_TIE_ABSOLUTE_TOLERANCE:
            wins += 1
        elif delta > PAIR_TIE_ABSOLUTE_TOLERANCE:
            losses += 1
        else:
            ties += 1
    samples = len(candidate_error)
    tail_count = int(math.ceil(CVAR_TAIL_FRACTION * samples))
    candidate_nmae = sum(candidate_error) / samples
    reference_nmae = sum(reference_error) / samples
    candidate_cvar = sum(sorted(candidate_error, reverse=True)[:tail_count]) / tail_count
    reference_cvar = sum(sorted(reference_error, reverse=True)[:tail_count]) / tail_count
    return {
        "samples": samples,
        "candidate_nmae": candidate_nmae,
        "reference_nmae": reference_nmae,
        "nmae_delta_candidate_minus_reference": candidate_nmae - reference_nmae,
        "cvar_tail_fraction": CVAR_TAIL_FRACTION,
        "cvar_tail_count_each": tail_count,
        "candidate_cvar25": candidate_cvar,
        "reference_cvar25": reference_cvar,
        "cvar25_delta_candidate_minus_reference": candidate_cvar - reference_cvar,
        "paired_wins": wins,
        "paired_ties": ties,
        "paired_losses": losses,
        "net_paired_win_minus_loss": (wins - losses) / samples,
        "tie_absolute_tolerance": PAIR_TIE_ABSOLUTE_TOLERANCE,
    }


def _partition_metric_report(
    *,
    predictions: Mapping[str, Sequence[float]],
    target: Sequence[float],
) -> dict[str, Any]:
    _require(set(predictions) == set(METHOD_ORDER), "A15 prediction methods differ")
    q0 = predictions[METHOD_Q0]
    versus_q0 = {
        method: paired_prediction_summary(
            candidate_mean=predictions[method],
            reference_mean=q0,
            target=target,
        )
        for method in METHOD_ORDER
    }
    learned_comparisons = {
        "learned_minus_fixed_geometric": paired_prediction_summary(
            candidate_mean=predictions[METHOD_A15_LEARNED],
            reference_mean=predictions[METHOD_FIXED_GEOMETRIC],
            target=target,
        ),
        "learned_minus_endpoint_null": paired_prediction_summary(
            candidate_mean=predictions[METHOD_A15_LEARNED],
            reference_mean=predictions[METHOD_ENDPOINT_NULL],
            target=target,
        ),
    }
    return {
        "rows": len(tuple(target)),
        "versus_q0": versus_q0,
        "learned_comparisons": learned_comparisons,
    }


def inner_eval_metric_report(
    *,
    predictions: Mapping[str, Sequence[float]],
    target: Sequence[float],
    condition_names: Sequence[str],
) -> dict[str, Any]:
    """Report all five methods pooled and by projective condition."""

    truth = _finite_vector(target, label="inner-eval target")
    conditions = tuple(str(value) for value in condition_names)
    _require(
        len(truth) == len(conditions) == EVAL_CACHE_ROWS,
        "A15 inner-eval metric input must contain 288 condition rows",
    )
    _require(
        set(conditions) == set(PROJECTIVE_CONDITIONS)
        and all(conditions.count(name) == EVAL_PHYSICAL_SAMPLES for name in PROJECTIVE_CONDITIONS),
        "A15 inner-eval condition roster/counts differ",
    )
    values = {
        method: _finite_vector(predictions[method], label=f"{method} prediction")
        for method in METHOD_ORDER
        if method in predictions
    }
    _require(set(values) == set(METHOD_ORDER), "A15 prediction methods differ")
    _require(
        all(len(row) == EVAL_CACHE_ROWS for row in values.values()),
        "A15 inner-eval prediction length differs",
    )
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
        "evaluation_scope": "inner_eval_only",
        "physical_samples": EVAL_PHYSICAL_SAMPLES,
        "condition_rows": EVAL_CACHE_ROWS,
        "method_order": METHOD_ORDER,
        "pooled": pooled,
        "per_condition": per_condition,
        "descriptive_reference": dict(DESCRIPTIVE_REFERENCE),
        "automatic_gate_used": False,
    }


_require(SOURCE_SAMPLES == 6_616 and SOURCE_SCENES == 60, "A15 source differs")
_require(
    len(INNER_TRAIN_SCENES) == INNER_TRAIN_SCENE_COUNT
    and len(INNER_EVAL_SCENES) == INNER_EVAL_SCENE_COUNT
    and not (set(INNER_TRAIN_SCENES) & set(INNER_EVAL_SCENES))
    and set(INNER_TRAIN_SCENES) | set(INNER_EVAL_SCENES)
    == set(CORRECTION_TRAIN_SCENE_STEMS),
    "A15 fixed scene split differs",
)
_require(
    (TRAIN_PHYSICAL_SAMPLES, EVAL_PHYSICAL_SAMPLES) == (384, 96)
    and (TRAIN_CACHE_ROWS, EVAL_CACHE_ROWS) == (1_152, 288),
    "A15 cache size differs",
)
_require(
    (TRAIN_STEPS, PHYSICAL_BATCH_SIZE, ROW_BATCH_SIZE, ROW_PRESENTATIONS)
    == (200, 8, 24, 4_800),
    "A15 training schedule differs",
)
_require(
    DESCRIPTIVE_REFERENCE["automatic_execution_selection_or_advancement_control"]
    is False,
    "A15 descriptive references cannot control execution",
)


__all__ = [
    "ACCESS_EXPECTATION",
    "A15InnerSceneProbeProtocolError",
    "BATCH_SCHEDULE_SEED",
    "CONDITIONS_PER_PHYSICAL",
    "CORRECTION_TRAIN_SCENE_STEMS",
    "DESCRIPTIVE_REFERENCE",
    "DEVELOPMENT_SCOPE",
    "ENDPOINT_CACHE_SEMANTICS",
    "EVAL_CACHE_ROWS",
    "EVAL_PHYSICAL_SAMPLES",
    "FIXED_GEOMETRIC_SEMANTICS",
    "INNER_EVAL_SCENES",
    "INNER_EVAL_SCENE_COUNT",
    "INNER_TRAIN_SCENES",
    "INNER_TRAIN_SCENE_COUNT",
    "MATCHED_ARM_SEMANTICS",
    "METHOD_A15_LEARNED",
    "METHOD_DIRECT_QS",
    "METHOD_ENDPOINT_NULL",
    "METHOD_FIXED_GEOMETRIC",
    "METHOD_ORDER",
    "METHOD_Q0",
    "PAIR_TIE_ABSOLUTE_TOLERANCE",
    "PHYSICAL_BATCH_SIZE",
    "PHYSICAL_PRESENTATIONS",
    "PHYSICAL_SAMPLES_PER_SCENE",
    "PIXEL_CONDITION_SEED",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "PhysicalSampleRef",
    "ROW_BATCH_SIZE",
    "ROW_PRESENTATIONS",
    "SAMPLE_SELECTION_SEED",
    "SCENE_SPLIT_SEED",
    "SCIENTIFIC_QUESTION",
    "SHARED_INITIALIZATION_SEED",
    "SOURCE_SAMPLES",
    "SOURCE_SCENES",
    "TRAIN_CACHE_ROWS",
    "TRAIN_PHYSICAL_SAMPLES",
    "TRAIN_STEPS",
    "expand_physical_indices_to_cache_rows",
    "fixed_physical_batch_schedule",
    "fixed_scene_partition",
    "inner_eval_metric_report",
    "paired_prediction_summary",
    "per_scene_candidate_order",
    "schedule_presentation_counts",
    "select_relation_complete_physical_samples",
]
