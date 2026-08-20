"""Predeclared train-only protocol for the A12 fixed-q0 causal probe.

The probe is deliberately small and diagnostic.  It asks whether a fresh
correction mechanism can learn on 128 relation-active rows when every arm
sees the same terminal A11 Raw posterior, pixels, conditions, and batch
order.  Candidate selection consumes only a relation-availability boolean;
neither the target nor any prediction error is an input to selection.

The descriptive values in :data:`DESCRIPTIVE_REFERENCE` are recorded for
interpretation only.  They are not an automatic gate and this module contains
no code that authorizes or suppresses another experiment.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final


PROTOCOL: Final[str] = "syncg_a12_fixed_q0_causal_probe128_step300_v1"
SCIENTIFIC_QUESTION: Final[str] = (
    "With one terminal A11 Raw q0 frozen in eval mode, distinguish correction-"
    "architecture learnability from the effect of q0-relative risk terms."
)
DEVELOPMENT_SCOPE: Final[str] = (
    "A11_physical_Core_train_7939_only;_Core_audit_FoldB_formal_and_field_are_"
    "not_inputs"
)
SOURCE_TRAIN_SAMPLES: Final[int] = 7_939
PROBE_SAMPLES: Final[int] = 128
PROBE_BATCH_SIZE: Final[int] = 16
PROBE_STEPS: Final[int] = 300
PROBE_PRESENTATIONS: Final[int] = PROBE_BATCH_SIZE * PROBE_STEPS
PROJECTIVE_CONDITION_CYCLE: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)

# Separate fixed seeds make selection, pixel/condition materialization,
# initialization, and optimization order independently auditable.  None may
# be changed after observing probe results.
SELECTION_SEED: Final[int] = 20_262_101
PIXEL_CONDITION_SEED: Final[int] = 20_262_102
BATCH_SCHEDULE_SEED: Final[int] = 20_262_103
SCORT_INITIALIZATION_SEED: Final[int] = 20_262_104
LDRT_INITIALIZATION_SEED: Final[int] = 20_262_105

ARM_SCORT_FULL: Final[str] = "scort_full"
ARM_SCORT_FINAL_ONLY: Final[str] = "scort_final_only"
ARM_LDRT_FULL: Final[str] = "ldrt_full"
ARM_LDRT_FINAL_ONLY: Final[str] = "ldrt_final_only"
ARM_ORDER: Final[tuple[str, ...]] = (
    ARM_SCORT_FULL,
    ARM_SCORT_FINAL_ONLY,
    ARM_LDRT_FULL,
    ARM_LDRT_FINAL_ONLY,
)

REGRET_SOFTPLUS_TEMPERATURE: Final[float] = 0.002
CVAR_TAIL_FRACTION: Final[float] = 0.25
CVAR_ERROR_NORMALIZATION: Final[float] = 0.05
FINAL_READ_WEIGHT: Final[float] = 1.0
REGRET_WEIGHT: Final[float] = 0.10
CVAR_WEIGHT: Final[float] = 0.25
SCORT_ANGLE_WEIGHT: Final[float] = 0.001
PAIR_TIE_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12


@dataclass(frozen=True, slots=True)
class ProbeArmSpec:
    architecture: str
    objective: str
    risk_scope: str
    initialization_pair: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


ARM_SPECS: Final[dict[str, ProbeArmSpec]] = {
    ARM_SCORT_FULL: ProbeArmSpec(
        architecture="SCORT_eight_layer_sparse_Givens_transport",
        objective="A11_final_read_plus_layer_regret_CVaR_and_angle",
        risk_scope="all_eight_intermediate_layers_relative_to_frozen_q0",
        initialization_pair="scort_fresh_pair",
    ),
    ARM_SCORT_FINAL_ONLY: ProbeArmSpec(
        architecture="SCORT_eight_layer_sparse_Givens_transport",
        objective="final_posterior_read_only",
        risk_scope="none",
        initialization_pair="scort_fresh_pair",
    ),
    ARM_LDRT_FULL: ProbeArmSpec(
        architecture="LDRT_centered_128_bin_log_density_residual_transport",
        objective="final_read_plus_final_regret_and_final_CVaR",
        risk_scope="final_output_only_relative_to_frozen_q0",
        initialization_pair="ldrt_fresh_pair",
    ),
    ARM_LDRT_FINAL_ONLY: ProbeArmSpec(
        architecture="LDRT_centered_128_bin_log_density_residual_transport",
        objective="final_posterior_read_only",
        risk_scope="none",
        initialization_pair="ldrt_fresh_pair",
    ),
}

LOSS_COMPARABILITY_NOTE: Final[str] = (
    "SCORT-full retains the original eight-layer regret/CVaR and angle terms, "
    "whereas LDRT-full has one final-output regret/CVaR analogue and no angle "
    "term.  The 2x2 probe is therefore a minimal causal diagnostic, not a "
    "claim that the two full objectives are algebraically isomorphic."
)

DESCRIPTIVE_REFERENCE: Final[dict[str, Any]] = {
    "arm": ARM_LDRT_FINAL_ONLY,
    "active_nmae_delta_final_minus_q0_maximum": -0.005,
    "active_net_paired_win_minus_loss_minimum": 0.50,
    "interpretation_if_either_reference_is_missed": (
        "the_fixed_q0_probe_does_not_support_LDRT_train_set_learnability"
    ),
    "train_probe_success_does_not_establish_generalization": True,
    "automatic_execution_or_advancement_control": False,
}

FORBIDDEN_DATA_NAMESPACES: Final[tuple[str, ...]] = (
    "core_audit",
    "audit_manifest",
    "fold_a",
    "fold_b",
    "formal",
    "holdout",
    "field",
    "test_photo",
)


class A12ProbeProtocolError(ValueError):
    """The fixed A12 probe roster, schedule, or metric input is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12ProbeProtocolError(message)


def candidate_permutation(sample_count: int = SOURCE_TRAIN_SAMPLES) -> tuple[int, ...]:
    """Return the predeclared target-blind candidate order.

    ``random.Random`` is local to the call, so global RNG state and model
    initialization cannot alter the roster.
    """

    count = int(sample_count)
    _require(count >= PROBE_SAMPLES, "A12 candidate population is too small")
    indices = list(range(count))
    random.Random(SELECTION_SEED).shuffle(indices)
    return tuple(indices)


def candidate_condition(candidate_position: int) -> str:
    """Bind a projective condition before relation availability is observed."""

    position = int(candidate_position)
    _require(position >= 0, "A12 candidate position is negative")
    return PROJECTIVE_CONDITION_CYCLE[position % len(PROJECTIVE_CONDITION_CYCLE)]


def candidate_pairs(sample_count: int = SOURCE_TRAIN_SAMPLES) -> tuple[tuple[int, str], ...]:
    """Return the fixed target-blind stream of sample-condition pairs."""

    return tuple(
        (sample_index, candidate_condition(position))
        for position, sample_index in enumerate(candidate_permutation(sample_count))
    )


def select_first_relation_active(
    candidate_indices: Sequence[int],
    relation_active_by_index: Sequence[bool],
    *,
    count: int = PROBE_SAMPLES,
) -> tuple[int, ...]:
    """Take the first active candidates without accepting targets or errors."""

    requested = int(count)
    _require(requested == PROBE_SAMPLES, "A12 probe sample count is fixed at 128")
    candidates = tuple(int(index) for index in candidate_indices)
    active = tuple(bool(value) for value in relation_active_by_index)
    _require(bool(candidates), "A12 candidate order is empty")
    _require(len(set(candidates)) == len(candidates), "candidate indices repeat")
    _require(
        min(candidates) >= 0 and max(candidates) < len(active),
        "candidate index is outside the availability vector",
    )
    selected = tuple(index for index in candidates if active[index])[:requested]
    _require(len(selected) == requested, "fewer than 128 relation-active rows exist")
    return selected


def select_first_relation_active_pairs(
    candidates: Sequence[tuple[int, str]],
    relation_active_by_index: Sequence[bool],
    *,
    count: int = PROBE_SAMPLES,
) -> tuple[tuple[int, str], ...]:
    """Preserve each candidate-position condition when selecting active pairs."""

    pairs = tuple((int(index), str(condition)) for index, condition in candidates)
    selected_indices = select_first_relation_active(
        tuple(index for index, _condition in pairs),
        relation_active_by_index,
        count=count,
    )
    selected_set = set(selected_indices)
    selected_pairs = tuple(pair for pair in pairs if pair[0] in selected_set)[:count]
    _require(
        tuple(index for index, _condition in selected_pairs) == selected_indices,
        "A12 selected pair order differs from selected indices",
    )
    _require(
        all(condition in PROJECTIVE_CONDITION_CYCLE for _index, condition in selected_pairs),
        "A12 selected pair condition is outside the projective cycle",
    )
    return selected_pairs


def fixed_batch_schedule() -> tuple[tuple[int, ...], ...]:
    """Generate the one sample-order schedule consumed by all four arms."""

    rng = random.Random(BATCH_SCHEDULE_SEED)
    flat: list[int] = []
    while len(flat) < PROBE_PRESENTATIONS:
        cycle = list(range(PROBE_SAMPLES))
        rng.shuffle(cycle)
        flat.extend(cycle)
    flat = flat[:PROBE_PRESENTATIONS]
    schedule = tuple(
        tuple(flat[offset : offset + PROBE_BATCH_SIZE])
        for offset in range(0, PROBE_PRESENTATIONS, PROBE_BATCH_SIZE)
    )
    _require(len(schedule) == PROBE_STEPS, "A12 schedule step count differs")
    _require(
        all(len(batch) == PROBE_BATCH_SIZE for batch in schedule),
        "A12 schedule batch width differs",
    )
    return schedule


def schedule_presentation_counts(
    schedule: Sequence[Sequence[int]],
) -> dict[int, int]:
    counts = {index: 0 for index in range(PROBE_SAMPLES)}
    batches = tuple(tuple(int(index) for index in batch) for batch in schedule)
    _require(len(batches) == PROBE_STEPS, "A12 schedule must contain 300 steps")
    for batch in batches:
        _require(len(batch) == PROBE_BATCH_SIZE, "A12 batch size must be 16")
        _require(len(set(batch)) == len(batch), "A12 batch repeats a probe row")
        for index in batch:
            _require(index in counts, "A12 schedule index is outside [0,127]")
            counts[index] += 1
    _require(sum(counts.values()) == PROBE_PRESENTATIONS, "presentation count differs")
    return counts


def paired_active_metrics(
    *,
    final_mean: Sequence[float],
    q0_mean: Sequence[float],
    target: Sequence[float],
    active: Sequence[bool],
) -> dict[str, Any]:
    """Compute active-row delta NMAE and strict paired win/tie/loss counts."""

    final = tuple(float(value) for value in final_mean)
    anchor = tuple(float(value) for value in q0_mean)
    truth = tuple(float(value) for value in target)
    mask = tuple(bool(value) for value in active)
    _require(
        len(final) == len(anchor) == len(truth) == len(mask) and bool(final),
        "A12 paired metric lengths differ or are empty",
    )
    _require(
        all(math.isfinite(value) for values in (final, anchor, truth) for value in values),
        "A12 paired metric contains a non-finite value",
    )
    differences: list[float] = []
    wins = ties = losses = 0
    for prediction, baseline, expected, enabled in zip(
        final, anchor, truth, mask, strict=True
    ):
        if not enabled:
            continue
        final_error = abs(prediction - expected)
        q0_error = abs(baseline - expected)
        delta = final_error - q0_error
        differences.append(delta)
        if delta < -PAIR_TIE_ABSOLUTE_TOLERANCE:
            wins += 1
        elif delta > PAIR_TIE_ABSOLUTE_TOLERANCE:
            losses += 1
        else:
            ties += 1
    _require(bool(differences), "A12 paired metric has no active rows")
    active_rows = len(differences)
    return {
        "active_rows": active_rows,
        "active_nmae_delta_final_minus_q0": sum(differences) / active_rows,
        "paired_wins": wins,
        "paired_ties": ties,
        "paired_losses": losses,
        "active_net_paired_win_minus_loss": (wins - losses) / active_rows,
        "tie_absolute_tolerance": PAIR_TIE_ABSOLUTE_TOLERANCE,
    }


def arm_protocol_record() -> dict[str, Mapping[str, str]]:
    return {name: ARM_SPECS[name].as_dict() for name in ARM_ORDER}


_require(tuple(ARM_SPECS) == ARM_ORDER, "A12 arm declaration order differs")
_require(PROBE_PRESENTATIONS == 4_800, "A12 presentation count differs")
_require(
    DESCRIPTIVE_REFERENCE["automatic_execution_or_advancement_control"] is False,
    "A12 descriptive reference cannot be an execution gate",
)


__all__ = [
    "ARM_LDRT_FINAL_ONLY",
    "ARM_LDRT_FULL",
    "ARM_ORDER",
    "ARM_SCORT_FINAL_ONLY",
    "ARM_SCORT_FULL",
    "ARM_SPECS",
    "A12ProbeProtocolError",
    "BATCH_SCHEDULE_SEED",
    "CVAR_ERROR_NORMALIZATION",
    "CVAR_TAIL_FRACTION",
    "CVAR_WEIGHT",
    "DESCRIPTIVE_REFERENCE",
    "DEVELOPMENT_SCOPE",
    "FINAL_READ_WEIGHT",
    "FORBIDDEN_DATA_NAMESPACES",
    "LDRT_INITIALIZATION_SEED",
    "LOSS_COMPARABILITY_NOTE",
    "PAIR_TIE_ABSOLUTE_TOLERANCE",
    "PIXEL_CONDITION_SEED",
    "PROBE_BATCH_SIZE",
    "PROBE_PRESENTATIONS",
    "PROBE_SAMPLES",
    "PROBE_STEPS",
    "PROJECTIVE_CONDITION_CYCLE",
    "PROTOCOL",
    "ProbeArmSpec",
    "REGRET_SOFTPLUS_TEMPERATURE",
    "REGRET_WEIGHT",
    "SCIENTIFIC_QUESTION",
    "SCORT_ANGLE_WEIGHT",
    "SCORT_INITIALIZATION_SEED",
    "SELECTION_SEED",
    "SOURCE_TRAIN_SAMPLES",
    "arm_protocol_record",
    "candidate_condition",
    "candidate_pairs",
    "candidate_permutation",
    "fixed_batch_schedule",
    "paired_active_metrics",
    "schedule_presentation_counts",
    "select_first_relation_active",
    "select_first_relation_active_pairs",
]
