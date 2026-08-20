"""Fixed scientific protocol for the A10 PCCOT matched comparison.

This module contains only ordinary experiment configuration and descriptive
statistics.  It does not construct a model, open a manifest, execute training,
or decide whether a later experiment may run.  A10 and its matched Raw
posterior parent are two independently parameterized ImageNet EfficientNet-B0
systems trained on the same physical Core rows and the same augmented Raw-view
pixels, order, and schedule.  A10 alone consumes the SARN-derived view and is
therefore neither input- nor compute-matched to the independent Raw parent.
"""
from __future__ import annotations

import math
from itertools import combinations
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final


PROTOCOL: Final[str] = "syncg_a10_pccot_matched_core5_fold_b_once_v1"
DEFAULT_SEED: Final[int] = 20262020
TERMINAL_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 3.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
CORE_SAMPLES: Final[int] = 9_825
CORE_SCENES: Final[int] = 89
FOLD_B_SAMPLES: Final[int] = 1_508
FOLD_B_SCENES: Final[int] = 14
STEPS_PER_EPOCH: Final[int] = math.ceil(CORE_SAMPLES / DEFAULT_BATCH_SIZE)
OPTIMIZER_STEPS_PER_SYSTEM: Final[int] = TERMINAL_EPOCHS * STEPS_PER_EPOCH
TOTAL_INDEPENDENT_OPTIMIZER_UPDATES: Final[int] = 2 * OPTIMIZER_STEPS_PER_SYSTEM
# Backward-compatible internal name: this is per system, not the sum of the two.
TOTAL_OPTIMIZER_STEPS: Final[int] = OPTIMIZER_STEPS_PER_SYSTEM
TRAIN_CONDITION_MIX: Final[tuple[tuple[str, float], ...]] = (
    ("clean", 0.30),
    ("perspective_moderate", 0.25),
    ("perspective_severe", 0.25),
    ("combined_severe", 0.20),
)
EVALUATION_CONDITIONS: Final[tuple[str, ...]] = tuple(
    condition for condition, _probability in TRAIN_CONDITION_MIX
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = tuple(
    condition for condition in EVALUATION_CONDITIONS if condition != "clean"
)
POSTERIOR_BRANCHES: Final[tuple[str, ...]] = (
    "raw_appearance",
    "sarn_appearance",
    "raw_conic_ordinal",
    "raw_ordered_tick_ordinal",
    "projective_relation_appearance",
)
STRUCTURE_AUX_BRANCHES: Final[tuple[str, ...]] = (
    "raw_conic_ordinal",
    "raw_ordered_tick_ordinal",
)
DIRECT_RELATION_BRANCH: Final[str] = "projective_relation_appearance"
COMPARISON_SCOPE: Final[str] = (
    "independent_full_systems_matched_raw_pixels_sample_order_epochs_optimizer_"
    "and_seed_not_compute_matched_not_identical_raw_learner_loss_exposure"
)
SARN_UNAVAILABLE_FALLBACK: Final[str] = (
    "a10_final_output_exact_a10_raw_appearance_q0_"
    "not_independent_raw_parent_weights_or_output"
)
INTERNAL_REGRET_REFERENCE: Final[str] = (
    "a10_internal_raw_branch_q0_mean_detached_never_independent_parent_prediction"
)

# These reproduce the established epoch-4 fixed condition pixels without
# coupling A10 to a previous model checkpoint or evaluation artifact.
EVALUATION_DATASET_SEED_OFFSET: Final[int] = 31_337
EVALUATION_PIXEL_TOTAL_EPOCHS: Final[int] = 5
EVALUATION_PIXEL_EPOCH: Final[int] = 4
EVALUATION_LOADER_SEED_OFFSET: Final[int] = 10_000
EVALUATION_CONDITION_SEED_STRIDE: Final[int] = 1_009
PAIRED_TIE_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12

PREDECLARED_REFERENCE: Final[dict[str, Any]] = {
    "projective_pooled_nmae_delta_a10_minus_raw_parent_maximum": -0.001,
    "projective_pooled_net_paired_win_margin_minimum": 0.03,
    "each_projective_condition_a10_nmae_strictly_below_raw_parent": True,
    "clean_nmae_delta_a10_minus_raw_parent_maximum": 0.0002,
    "monotonic_violation_total": 0,
    "posterior_branch_valid_rate_reported": True,
    "posterior_branch_order": list(POSTERIOR_BRANCHES),
    "comparison_scope": COMPARISON_SCOPE,
    "parameter_count_flops_and_latency_reported_for_both_systems": True,
    "sarn_unavailable_fallback_scope": SARN_UNAVAILABLE_FALLBACK,
    "internal_raw_branch_regret_reference": INTERNAL_REGRET_REFERENCE,
    "automatic_execution_or_advancement_control": False,
}


class A10ProtocolError(ValueError):
    """The requested A10 run or paired statistic violates the fixed protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A10ProtocolError(message)


@dataclass(frozen=True, slots=True)
class A10TrainingSpec:
    seed: int = DEFAULT_SEED
    epochs: int = TERMINAL_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    scheduler: str = "cosine"
    checkpoint_selection: str = "terminal_epoch_5_no_validation_selection"

    def __post_init__(self) -> None:
        _require(self.seed == DEFAULT_SEED, "A10 uses the fixed seed 20262020")
        _require(self.epochs == TERMINAL_EPOCHS, "A10 uses five terminal epochs")
        _require(self.batch_size == DEFAULT_BATCH_SIZE, "A10 uses batch size 8")
        _require(
            self.learning_rate == DEFAULT_LEARNING_RATE,
            "A10 uses AdamW learning rate 3e-4",
        )
        _require(
            self.weight_decay == DEFAULT_WEIGHT_DECAY,
            "A10 uses AdamW weight decay 1e-4",
        )
        _require(self.scheduler == "cosine", "A10 uses cosine scheduling")

    @property
    def steps_per_epoch(self) -> int:
        return STEPS_PER_EPOCH

    @property
    def total_optimizer_steps(self) -> int:
        return OPTIMIZER_STEPS_PER_SYSTEM

    def as_metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "optimizer": "AdamW",
            "steps_per_epoch": self.steps_per_epoch,
            "optimizer_steps_per_system": self.total_optimizer_steps,
            "total_independent_optimizer_updates": (
                TOTAL_INDEPENDENT_OPTIMIZER_UPDATES
            ),
            "train_condition_mix": dict(TRAIN_CONDITION_MIX),
        }


@dataclass(frozen=True, slots=True)
class A10ConditionEvaluationSpec:
    condition: str
    dataset_seed: int
    dataset_total_epochs: int
    transform_epoch: int
    loader_seed: int


def condition_evaluation_specs(
    seed: int = DEFAULT_SEED,
) -> tuple[A10ConditionEvaluationSpec, ...]:
    """Return the sole fixed-pixel Fold-B replay specification."""

    _require(int(seed) == DEFAULT_SEED, "Fold-B replay uses seed 20262020")
    return tuple(
        A10ConditionEvaluationSpec(
            condition=condition,
            dataset_seed=DEFAULT_SEED + EVALUATION_DATASET_SEED_OFFSET,
            dataset_total_epochs=EVALUATION_PIXEL_TOTAL_EPOCHS,
            transform_epoch=EVALUATION_PIXEL_EPOCH,
            loader_seed=(
                DEFAULT_SEED
                + EVALUATION_LOADER_SEED_OFFSET
                + EVALUATION_PIXEL_EPOCH
                + index * EVALUATION_CONDITION_SEED_STRIDE
            ),
        )
        for index, condition in enumerate(EVALUATION_CONDITIONS)
    )


def validate_matched_training_metadata(
    a10_training: Mapping[str, Any],
    raw_parent_training: Mapping[str, Any],
) -> None:
    """Check ordinary recorded settings for a genuinely matched comparison.

    This is report validation, not an execution gate.  Model independence must
    additionally be established from object/state identity by the trainer.
    """

    required_equal = (
        "seed",
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "steps_per_epoch",
        "optimizer_steps_per_system",
        "train_condition_mix",
        "raw_view_augmentation",
        "core_manifest",
        "core_samples",
        "sample_order_per_epoch",
        "encoder_initialization",
        "raw_preprocessing",
        "raw_posterior_head_spec",
        "amp",
        "gradient_clipping",
        "ema",
    )
    missing = [
        key
        for key in required_equal
        if key not in a10_training or key not in raw_parent_training
    ]
    _require(not missing, f"matched metadata is incomplete: {missing}")
    differing = [
        key
        for key in required_equal
        if a10_training[key] != raw_parent_training[key]
    ]
    _require(not differing, f"A10 and Raw parent settings differ: {differing}")
    expected = A10TrainingSpec().as_metadata()
    for key in (
        "seed",
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "steps_per_epoch",
        "optimizer_steps_per_system",
        "train_condition_mix",
    ):
        _require(
            a10_training[key] == expected[key],
            f"recorded A10 setting differs from the fixed protocol: {key}",
        )
    _require(
        a10_training.get("comparison_scope") == COMPARISON_SCOPE
        and raw_parent_training.get("comparison_scope") == COMPARISON_SCOPE,
        "matched metadata does not disclose the full-system comparison scope",
    )
    _require(
        bool(a10_training.get("loss_exposure"))
        and bool(raw_parent_training.get("loss_exposure")),
        "both systems must disclose their different loss exposure",
    )
    for name, metadata in (
        ("A10", a10_training),
        ("Raw parent", raw_parent_training),
    ):
        _require(
            isinstance(metadata.get("consumed_views"), Sequence)
            and bool(metadata["consumed_views"]),
            f"{name} consumed views are not recorded",
        )
        _require(
            isinstance(metadata.get("per_step_forwarded_image_count"), int)
            and int(metadata["per_step_forwarded_image_count"]) >= 1,
            f"{name} per-step forwarded image count is not recorded",
        )
        _require(
            all(
                key in metadata
                for key in (
                    "parameter_count",
                    "flops_measurement",
                    "latency_measurement",
                )
            ),
            f"{name} efficiency measurement fields are absent",
        )
    _require(
        "original_view" in a10_training["consumed_views"]
        and "sarn_view" in a10_training["consumed_views"],
        "A10 consumed-view disclosure omits Raw or SARN",
    )
    _require(
        tuple(raw_parent_training["consumed_views"]) == ("original_view",),
        "Raw parent must consume only the augmented Raw view",
    )
    _require(
        int(a10_training["per_step_forwarded_image_count"])
        == 2 * int(a10_training["batch_size"])
        and int(raw_parent_training["per_step_forwarded_image_count"])
        == int(raw_parent_training["batch_size"]),
        "per-step forwarded image counts do not disclose A10 Raw+SARN versus parent Raw",
    )
    _require(
        a10_training.get("internal_raw_branch_regret_reference")
        == INTERNAL_REGRET_REFERENCE,
        "A10 internal Raw-branch regret reference is not disclosed",
    )
    _require(
        a10_training.get("independent_parent_predictions_consumed_during_training")
        is False,
        "A10 training must not consume independent Raw-parent predictions",
    )
    step0 = a10_training.get("step0_common_subnetwork_evidence")
    _require(
        isinstance(step0, Mapping)
        and step0.get("tensor_names_equal") is True
        and step0.get("tensor_values_equal") is True
        and step0.get("parameter_and_buffer_storage_disjoint") is True,
        "step-0 common encoder/head equality and storage independence are absent",
    )
    independence = a10_training.get("training_independence_evidence")
    _require(
        isinstance(independence, Mapping)
        and independence.get("distinct_model_objects") is True
        and independence.get("parameter_storage_disjoint") is True
        and independence.get("distinct_optimizer_objects") is True
        and independence.get("optimizer_parameter_sets_disjoint") is True
        and independence.get("each_optimizer_covers_only_its_model") is True,
        "independent model/optimizer training evidence is absent",
    )


def paired_win_tie_loss(
    a10_absolute_error: Sequence[float],
    raw_parent_absolute_error: Sequence[float],
    *,
    tie_absolute_tolerance: float = PAIRED_TIE_ABSOLUTE_TOLERANCE,
) -> dict[str, float | int]:
    """Summarize per-sample A10 versus independent Raw-parent outcomes."""

    candidate = tuple(float(value) for value in a10_absolute_error)
    parent = tuple(float(value) for value in raw_parent_absolute_error)
    _require(bool(candidate), "paired error vectors are empty")
    _require(len(candidate) == len(parent), "paired error lengths differ")
    _require(
        math.isfinite(float(tie_absolute_tolerance))
        and float(tie_absolute_tolerance) >= 0.0,
        "paired tie tolerance is invalid",
    )
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in candidate + parent),
        "paired errors must be finite and non-negative",
    )
    tolerance = float(tie_absolute_tolerance)
    wins = ties = losses = 0
    for a10_error, parent_error in zip(candidate, parent, strict=True):
        difference = a10_error - parent_error
        if abs(difference) <= tolerance:
            ties += 1
        elif difference < 0.0:
            wins += 1
        else:
            losses += 1
    samples = len(candidate)
    return {
        "samples": samples,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / samples,
        "tie_rate": ties / samples,
        "loss_rate": losses / samples,
        "net_paired_win_margin": (wins - losses) / samples,
        "mean_absolute_error_a10": sum(candidate) / samples,
        "mean_absolute_error_raw_parent": sum(parent) / samples,
        "nmae_delta_a10_minus_raw_parent": (
            sum(candidate) - sum(parent)
        )
        / samples,
        "tie_absolute_tolerance": tolerance,
    }


def validate_read_loss_coverage(
    *,
    read_valid: Sequence[bool],
    final_read_loss_applied: Sequence[bool],
    final_internal_raw_branch_regret_applied: Sequence[bool],
    branch_source_available: Sequence[Sequence[bool]],
    branch_read_loss_applied: Sequence[Sequence[bool]],
    structure_valid: Sequence[bool],
    structure_aux_loss_applied: Sequence[bool],
    expected_branches: int = len(POSTERIOR_BRANCHES),
) -> dict[str, int]:
    """Audit read/structure masks without assuming model-output field names.

    Final Wasserstein-median reading and its regret against the detached A10
    q0 Raw-appearance mean must cover every read-valid output row.  The
    independent matched parent is never part of this loss.  Branch objectives
    use the pre-substitution posteriors: q0/q2/q3 cover all read-valid rows,
    q1 additionally requires SARN availability, and q4 additionally requires
    relation availability.  A missing branch copied from q0 participates only
    in final consensus and is not charged a duplicate q0 branch loss.  Only
    the separate structure auxiliary may use structure validity.
    """

    read = tuple(bool(value) for value in read_valid)
    final = tuple(bool(value) for value in final_read_loss_applied)
    internal_regret = tuple(
        bool(value) for value in final_internal_raw_branch_regret_applied
    )
    structure = tuple(bool(value) for value in structure_valid)
    structure_aux = tuple(bool(value) for value in structure_aux_loss_applied)
    branches = tuple(tuple(bool(value) for value in row) for row in branch_read_loss_applied)
    availability = tuple(
        tuple(bool(value) for value in row) for row in branch_source_available
    )
    _require(bool(read), "loss-coverage audit is empty")
    _require(
        len(final)
        == len(internal_regret)
        == len(structure)
        == len(structure_aux)
        == len(read),
        "loss-coverage vector lengths differ",
    )
    _require(
        len(branches) == len(availability) == len(read),
        "branch loss-coverage row count differs",
    )
    _require(
        all(
            len(row) == int(expected_branches)
            for row in branches + availability
        ),
        "branch loss-coverage width differs",
    )
    _require(final == read, "final read loss does not cover every read-valid row")
    _require(
        internal_regret == read,
        "internal Raw-branch regret does not cover every read-valid row",
    )
    for row_index, (is_read_valid, source_row, row) in enumerate(
        zip(read, availability, branches, strict=True)
    ):
        if is_read_valid:
            _require(
                source_row[0] and source_row[2] and source_row[3],
                f"always-defined Raw branch is unavailable at row {row_index}",
            )
        expected = tuple(is_read_valid and available for available in source_row)
        _require(row == expected, f"branch read loss coverage differs at row {row_index}")
    _require(
        all(not applied or valid for applied, valid in zip(structure_aux, structure, strict=True)),
        "structure auxiliary escaped the structure-valid selector",
    )
    return {
        "rows": len(read),
        "read_valid_rows": sum(read),
        "final_read_rows": sum(final),
        "internal_raw_branch_regret_rows": sum(internal_regret),
        "branch_read_rows": sum(sum(row) for row in branches),
        "structure_valid_rows": sum(structure),
        "structure_aux_rows": sum(structure_aux),
    }


def posterior_branch_valid_rates(
    branch_available: Sequence[Sequence[bool]],
) -> dict[str, dict[str, float | int]]:
    """Report source availability for every fixed PCCOT vote."""

    rows = tuple(tuple(bool(value) for value in row) for row in branch_available)
    _require(bool(rows), "posterior branch availability is empty")
    _require(
        all(len(row) == len(POSTERIOR_BRANCHES) for row in rows),
        "posterior branch availability width differs",
    )
    samples = len(rows)
    result: dict[str, dict[str, float | int]] = {}
    for branch_index, name in enumerate(POSTERIOR_BRANCHES):
        count = sum(row[branch_index] for row in rows)
        result[name] = {
            "available_rows": count,
            "samples": samples,
            "valid_rate": count / samples,
        }
    return result


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = 0.5 * (cursor + end - 1) + 1.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return tuple(ranks)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    _require(len(left) == len(right), "correlation vector lengths differ")
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 0.0:
        return None
    return sum(
        first * second
        for first, second in zip(left_centered, right_centered, strict=True)
    ) / denominator


def posterior_branch_error_diagnostics(
    *,
    raw_branch_means: Sequence[Sequence[float]],
    branch_source_available: Sequence[Sequence[bool]],
    target: Sequence[float],
    zero_error_tolerance: float = PAIRED_TIE_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Describe failure dependence of pre-substitution projective branch errors.

    The caller pools only the three projective Fold-B conditions before this
    function.  Correlations are descriptive mechanism evidence; within-batch
    disagreement is never interpreted as statistical independence.
    """

    means = tuple(tuple(float(value) for value in row) for row in raw_branch_means)
    availability = tuple(
        tuple(bool(value) for value in row) for row in branch_source_available
    )
    targets = tuple(float(value) for value in target)
    _require(bool(means), "branch-error diagnostic is empty")
    _require(
        len(means) == len(availability) == len(targets),
        "branch-error diagnostic row counts differ",
    )
    _require(
        all(
            len(row) == len(POSTERIOR_BRANCHES)
            for row in means + availability
        ),
        "branch-error diagnostic width differs",
    )
    _require(
        all(math.isfinite(value) for row in means for value in row)
        and all(math.isfinite(value) for value in targets),
        "branch-error diagnostic is non-finite",
    )
    tolerance = float(zero_error_tolerance)
    _require(
        math.isfinite(tolerance) and tolerance >= 0.0,
        "branch-error zero tolerance is invalid",
    )
    signed_errors = tuple(
        tuple(value - target_value for value in row)
        for row, target_value in zip(means, targets, strict=True)
    )
    pairwise: dict[str, dict[str, float | int | None]] = {}
    for first, second in combinations(range(len(POSTERIOR_BRANCHES)), 2):
        paired = tuple(
            (row[first], row[second])
            for row, available in zip(signed_errors, availability, strict=True)
            if available[first] and available[second]
        )
        left = tuple(row[0] for row in paired)
        right = tuple(row[1] for row in paired)
        same_direction = sum(
            (first_error > tolerance and second_error > tolerance)
            or (first_error < -tolerance and second_error < -tolerance)
            for first_error, second_error in paired
        )
        pairwise[
            f"{POSTERIOR_BRANCHES[first]}__{POSTERIOR_BRANCHES[second]}"
        ] = {
            "coavailable_rows": len(paired),
            "signed_error_pearson": _pearson(left, right),
            "signed_error_spearman": (
                _pearson(_average_ranks(left), _average_ranks(right))
                if paired
                else None
            ),
            "same_direction_error_rows": same_direction,
            "same_direction_error_rate": (
                same_direction / len(paired) if paired else None
            ),
        }
    three_or_more = 0
    for row, available in zip(signed_errors, availability, strict=True):
        positive = sum(
            is_available and error > tolerance
            for error, is_available in zip(row, available, strict=True)
        )
        negative = sum(
            is_available and error < -tolerance
            for error, is_available in zip(row, available, strict=True)
        )
        three_or_more += max(positive, negative) >= 3
    return {
        "samples": len(means),
        "posterior_branches": list(POSTERIOR_BRANCHES),
        "input_semantics": "pre_substitution_raw_branch_posteriors",
        "projective_conditions_pooled_before_calculation": True,
        "pairwise_signed_error": pairwise,
        "three_or_more_same_direction_error_rows": three_or_more,
        "three_or_more_same_direction_error_rate": three_or_more / len(means),
        "batch_disagreement_claims_independence": False,
        "zero_error_tolerance": tolerance,
    }


def independent_parameter_storage_evidence(
    a10_model: Any,
    raw_parent_model: Any,
) -> dict[str, Any]:
    """Describe object/parameter independence without controlling execution."""

    _require(hasattr(a10_model, "parameters"), "A10 model has no parameters()")
    _require(
        hasattr(raw_parent_model, "parameters"),
        "Raw parent model has no parameters()",
    )
    a10_parameters = tuple(a10_model.parameters())
    parent_parameters = tuple(raw_parent_model.parameters())
    a10_storage = {
        (parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
        for parameter in a10_parameters
    }
    parent_storage = {
        (parameter.untyped_storage().data_ptr(), parameter.untyped_storage().nbytes())
        for parameter in parent_parameters
    }
    shared = a10_storage & parent_storage
    return {
        "distinct_model_objects": a10_model is not raw_parent_model,
        "a10_parameter_tensors": len(a10_parameters),
        "raw_parent_parameter_tensors": len(parent_parameters),
        "shared_parameter_storages": len(shared),
        "parameter_storage_disjoint": not shared,
    }


def step0_common_subnetwork_evidence(
    a10_common_subnetwork: Any,
    raw_parent_common_subnetwork: Any,
) -> dict[str, Any]:
    """Compare equal-valued but storage-independent common initialization."""

    for label, module in (
        ("A10 common subnetwork", a10_common_subnetwork),
        ("Raw-parent common subnetwork", raw_parent_common_subnetwork),
    ):
        _require(hasattr(module, "state_dict"), f"{label} has no state_dict()")
        _require(hasattr(module, "parameters"), f"{label} has no parameters()")
        _require(hasattr(module, "buffers"), f"{label} has no buffers()")
    a10_state = a10_common_subnetwork.state_dict()
    parent_state = raw_parent_common_subnetwork.state_dict()
    names_equal = tuple(a10_state) == tuple(parent_state)
    values_equal = names_equal and all(
        value.shape == parent_state[name].shape
        and value.dtype == parent_state[name].dtype
        and bool(value.detach().cpu().equal(parent_state[name].detach().cpu()))
        for name, value in a10_state.items()
    )

    def storage_keys(module: Any) -> set[tuple[int, int]]:
        tensors = tuple(module.parameters()) + tuple(module.buffers())
        return {
            (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            for tensor in tensors
        }

    a10_storage = storage_keys(a10_common_subnetwork)
    parent_storage = storage_keys(raw_parent_common_subnetwork)
    shared = a10_storage & parent_storage
    return {
        "tensor_names_equal": names_equal,
        "tensor_values_equal": values_equal,
        "a10_state_tensors": len(a10_state),
        "raw_parent_state_tensors": len(parent_state),
        "shared_parameter_and_buffer_storages": len(shared),
        "parameter_and_buffer_storage_disjoint": not shared,
    }


def independent_training_objects_evidence(
    a10_model: Any,
    raw_parent_model: Any,
    a10_optimizer: Any,
    raw_parent_optimizer: Any,
) -> dict[str, Any]:
    """Describe whether models and optimizers remain completely independent."""

    for label, model in (("A10", a10_model), ("Raw parent", raw_parent_model)):
        _require(hasattr(model, "parameters"), f"{label} model has no parameters()")
    for label, optimizer in (
        ("A10", a10_optimizer),
        ("Raw parent", raw_parent_optimizer),
    ):
        _require(
            hasattr(optimizer, "param_groups"),
            f"{label} optimizer has no param_groups",
        )

    a10_parameters = tuple(a10_model.parameters())
    parent_parameters = tuple(raw_parent_model.parameters())
    a10_parameter_ids = {id(parameter) for parameter in a10_parameters}
    parent_parameter_ids = {id(parameter) for parameter in parent_parameters}

    def optimizer_parameter_ids(optimizer: Any) -> set[int]:
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

    a10_optimizer_ids = optimizer_parameter_ids(a10_optimizer)
    parent_optimizer_ids = optimizer_parameter_ids(raw_parent_optimizer)
    storage = independent_parameter_storage_evidence(a10_model, raw_parent_model)
    return {
        "distinct_model_objects": a10_model is not raw_parent_model,
        "parameter_storage_disjoint": storage["parameter_storage_disjoint"],
        "distinct_optimizer_objects": a10_optimizer is not raw_parent_optimizer,
        "optimizer_parameter_sets_disjoint": not (
            a10_optimizer_ids & parent_optimizer_ids
        ),
        "a10_optimizer_covers_only_a10": a10_optimizer_ids == a10_parameter_ids,
        "raw_parent_optimizer_covers_only_parent": (
            parent_optimizer_ids == parent_parameter_ids
        ),
        "each_optimizer_covers_only_its_model": (
            a10_optimizer_ids == a10_parameter_ids
            and parent_optimizer_ids == parent_parameter_ids
        ),
    }


__all__ = [
    "A10ConditionEvaluationSpec",
    "A10ProtocolError",
    "A10TrainingSpec",
    "CORE_SAMPLES",
    "CORE_SCENES",
    "COMPARISON_SCOPE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SEED",
    "DEFAULT_WEIGHT_DECAY",
    "EVALUATION_CONDITIONS",
    "INTERNAL_REGRET_REFERENCE",
    "DIRECT_RELATION_BRANCH",
    "FOLD_B_SAMPLES",
    "FOLD_B_SCENES",
    "PAIRED_TIE_ABSOLUTE_TOLERANCE",
    "OPTIMIZER_STEPS_PER_SYSTEM",
    "PREDECLARED_REFERENCE",
    "POSTERIOR_BRANCHES",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "SARN_UNAVAILABLE_FALLBACK",
    "STEPS_PER_EPOCH",
    "STRUCTURE_AUX_BRANCHES",
    "TERMINAL_EPOCHS",
    "TOTAL_OPTIMIZER_STEPS",
    "TOTAL_INDEPENDENT_OPTIMIZER_UPDATES",
    "TRAIN_CONDITION_MIX",
    "condition_evaluation_specs",
    "independent_parameter_storage_evidence",
    "independent_training_objects_evidence",
    "paired_win_tie_loss",
    "posterior_branch_error_diagnostics",
    "posterior_branch_valid_rates",
    "step0_common_subnetwork_evidence",
    "validate_read_loss_coverage",
    "validate_matched_training_metadata",
]
