"""Descriptive A15.2 frozen-twin-endpoint correction-development replay.

The CLI accepts only the 1,323-row physical correction-dev manifest, the
terminal A11 Raw anchor, the terminal five-epoch A15.2 correction, and one
create-once JSON output.  The complete A15.2 artifact is validated before the
development manifest is opened.  No correction-train pixels, prior audit,
Fold, formal, or field input is accepted.

For every batch the frozen A11 reader is called once through
``frozen_twin_endpoint_forward`` (one separate Raw call and one separate SARN
call).  q0, relation-available q_sarn, the fixed geometric endpoint, and the
A15.2 posterior all derive from those same endpoint tensors.  Clean rows are
explicitly disabled, and every relation-unavailable row must return exact q0.
The report is descriptive and contains no gate or automatic model choice.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

import experiments.evaluate_a13_correction_dev as a13_evaluation
from experiments.a10_pccot_protocol import (
    DEFAULT_SEED as EVALUATION_SEED,
    condition_evaluation_specs,
)
from experiments.a13_correction_dev_protocol import (
    CORRECTION_DEV_SAMPLES,
    CORRECTION_TRAIN_SAMPLES,
    CORRECTION_TRAIN_SCENES,
    CORRECTION_TRAIN_SCENE_COUNT,
    DEVELOPMENT_INTERPRETATION,
    EVALUATION_CONDITIONS,
    EVALUATION_PIXEL_EPOCH,
    EVALUATION_PIXEL_TOTAL_EPOCHS,
    PROJECTIVE_CONDITIONS,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
)
from experiments.a15_1_fteb_targets import LOSS_DESIGN_METADATA, LOSS_WEIGHTS
from experiments.a15_2_fteb import (
    A15_2_ARCHITECTURE,
    A15_2_LEARNED_RESIDUAL_SCALE,
    A152FTEBCorrection,
)
from experiments.a15_2_fteb_oof_protocol import PROTOCOL as SELECTION_PROTOCOL
from experiments.a15_fteb import (
    BRIDGE_LAYERS,
    fixed_geometric_natural_parameter_base,
    frozen_twin_endpoint_forward,
    fteb_parameter_counts,
)
from experiments.a15_fteb_inner_scene_probe_protocol import (
    paired_prediction_summary,
)
from experiments.evaluate_a11_scort_core_audit import (
    verify_clean_epoch4_exact_replay,
)
from experiments.prepare_a13_correction_scene_split import (
    PROTOCOL as DATA_PROTOCOL,
    load_a13_correction_dev_manifest,
)
from experiments.run_a12_fixed_q0_causal_probe import (
    TERMINAL_A11_ACCESS_SCHEMA,
    load_frozen_terminal_a11,
)
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.train_a15_2_fteb_correction_only import (
    CONDITIONS_PER_PHYSICAL,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_TOTAL_OPTIMIZER_STEPS,
    INITIALIZATION_SEED,
    LEARNING_RATE,
    PHYSICAL_BATCH_SIZE,
    PIXEL_AUGMENTATION_SEED,
    PROTOCOL as TRAINING_PROTOCOL,
    ROW_BATCH_SIZE,
    SAMPLE_ORDER_SEED,
    SCALAR_LOSS_COMPONENTS,
    SEMANTIC_GROUP_NAMES,
    SYSTEM,
    TERMINAL_EPOCHS,
    WEIGHT_DECAY,
)
from experiments.train_a11_scort_syncg import PROTOCOL as A11_TRAINING_PROTOCOL
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
)


PROTOCOL: Final[str] = (
    "a15_2_fteb_correction_dev_seen_q0_same_twin_endpoint_epoch4_v1"
)
EVALUATION_BATCH_SIZE: Final[int] = 24
EXPECTED_CORRECTION_PARAMETER_COUNT: Final[int] = 169_724
EXPECTED_TRAIN_ACCESS_FLAGS: Final[dict[str, bool]] = {
    "correction_train_manifest_access": True,
    "terminal_a11_checkpoint_access": True,
    "correction_dev_manifest_access": False,
    "core_audit_manifest_access": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}
METHOD_Q0: Final[str] = "q0"
METHOD_QS: Final[str] = "q_sarn"
METHOD_FIXED_GEOMETRIC: Final[str] = "fixed_geometric"
METHOD_A15_2: Final[str] = "a15_2"
METHOD_ORDER: Final[tuple[str, ...]] = (
    METHOD_Q0,
    METHOD_QS,
    METHOD_FIXED_GEOMETRIC,
    METHOD_A15_2,
)
EXPECTED_CONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "relation_channels",
    "token_dim",
    "attention_heads",
    "decoder_layers",
    "memory_grid_size",
    "progress_bins",
)
EXPECTED_CHECKPOINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "selection_protocol",
        "data_protocol",
        "system",
        "architecture",
        "learned_residual_scale",
        "source_terminal_a11",
        "construction",
        "parameter_counts",
        "optimizer_partition_evidence",
        "terminal_optimizer_and_scaler_state",
        "training",
        "physical_correction_train",
        "loss_design",
        "loss_weights",
        "semantic_gradient_groups",
        "history",
        "access_flags",
        "anchor_unchanged_evidence",
        "fresh_strict_load",
        "terminal_fallback_evidence",
        "deterministic_algorithms_enabled",
        "automatic_execution_or_advancement_control",
        "model_state",
    }
)
EXPECTED_TRAINING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "initialization_seed",
        "sample_order_seed",
        "sample_order_seed_rule",
        "pixel_augmentation_seed",
        "pixel_augmentation_epoch_rule",
        "epochs",
        "terminal_checkpoint_selection",
        "physical_batch_size",
        "conditions_per_physical",
        "row_batch_size_full",
        "last_row_batch_size",
        "physical_samples_per_epoch",
        "condition_rows_per_epoch",
        "steps_per_epoch",
        "optimizer_steps",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "scheduler_steps",
        "condition_counts_terminal_total",
        "projective_conditions",
        "clean_presentations",
        "gradient_clipping",
        "ema",
        "amp",
        "validation_manifest",
        "intermediate_checkpoint_selection",
    }
)
EXPECTED_HISTORY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "epoch",
        "sample_order_seed",
        "pixel_augmentation_seed",
        "pixel_epoch",
        "learning_rate",
        "metrics",
    }
)
EXPECTED_HISTORY_METRIC_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "physical_samples",
        "condition_rows",
        "steps",
        "optimizer_steps",
        "source_index_order",
        "condition_counts",
        "active_rows",
        "inactive_rows",
        "active_fraction",
        "cross_condition_active_groups",
        "loss_components_active_row_weighted",
        "loss_denominator",
        "cross_condition_denominator",
        "gradient_finite_every_step",
        "gradient_nonzero_seen",
        "optimizer_state_finite_every_step",
        "correction_state_finite_every_step",
        "anchor_runtime_contract_every_step",
        "anchor_state_bit_exact_at_epoch_end",
        "q0_output_identity_every_step",
        "inactive_exact_q0_every_step",
        "posterior_mass_cdf_every_step",
        "mixed_active_inactive_seen",
        "initial_active_exact_fixed_geometric",
        "autocast_precision",
        "gradient_clipping",
        "step_trace",
    }
)
EXPECTED_FRESH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "missing_keys",
        "unexpected_keys",
        "strict_load_clean",
        "state_bit_exact",
        "same_input_reported_outputs_bit_exact",
        "fresh_state_finite",
        "fresh_posterior_mass_cdf",
    }
)
EXPECTED_FALLBACK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "relation_available_all_false",
        "posterior_cdf_layers_and_moments_exact_q0",
        "posterior_mass_cdf",
    }
)


class A152CorrectionDevEvaluationError(ValueError):
    """An A15.2 terminal artifact, dev roster, output, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152CorrectionDevEvaluationError(message)


def _exact_int(value: Any, *, label: str) -> int:
    _require(type(value) is int, f"{label} must be an exact integer")
    return value


def _schema_equal(observed: Any, expected: Any) -> bool:
    """Compare serialized values without bool/int or tuple/list coercion."""

    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(_schema_equal(observed[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _schema_equal(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    return type(observed) is type(expected) and observed == expected


def _exact_bool_mapping(
    value: Any,
    expected: Mapping[str, bool],
    *,
    label: str,
) -> dict[str, bool]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    observed = dict(value)
    _require(set(observed) == set(expected), f"{label} keys differ")
    _require(
        all(observed[name] is bool(expected[name]) for name in expected),
        f"{label} values differ",
    )
    return dict(expected)


def _finite_tensor_state(value: Any, *, label: str) -> dict[str, torch.Tensor]:
    _require(isinstance(value, Mapping) and bool(value), f"{label} is missing")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        _require(isinstance(name, str) and bool(name), f"{label} key is malformed")
        _require(isinstance(tensor, torch.Tensor), f"{label} tensor is malformed")
        _require(
            not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
            f"{label} contains a non-finite tensor",
        )
        state[name] = tensor
    return state


def _validate_state_schema(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    _require(set(state) == set(reference), "A15.2 model-state keys differ")
    for name, expected in reference.items():
        observed = state[name]
        _require(
            observed.shape == expected.shape
            and observed.dtype == expected.dtype
            and observed.layout == expected.layout
            and observed.device == expected.device,
            f"A15.2 model-state tensor schema differs: {name}",
        )


def _finite_optimizer_evidence(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("finite") is not True:
        return False
    optimizer = value.get("optimizer")
    scaler = value.get("scaler")
    return (
        isinstance(optimizer, Mapping)
        and optimizer.get("finite") is True
        and optimizer.get("adam_moving_averages_present") is True
        and isinstance(scaler, Mapping)
        and scaler.get("finite") is True
    )


def _valid_posterior_evidence(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "final_cdf_exact_from_posterior",
            "layer_cdf_exact_from_posterior",
            "maximum_absolute_layer_mass_error",
            "minimum_layer_probability",
            "finite",
            "valid",
        }
        and value.get("valid") is True
        and value.get("final_cdf_exact_from_posterior") is True
        and value.get("layer_cdf_exact_from_posterior") is True
        and value.get("finite") is True
        and type(value.get("maximum_absolute_layer_mass_error")) is float
        and float(value["maximum_absolute_layer_mass_error"]) <= 1.0e-6
        and type(value.get("minimum_layer_probability")) is float
        and float(value["minimum_layer_probability"]) >= 0.0
    )


def _checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    _require(
        source.is_file(),
        f"terminal A15.2 correction checkpoint does not exist: {source}",
    )
    value = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(value, Mapping),
        "terminal A15.2 correction checkpoint is not a mapping",
    )
    return value


def _strict_training_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    training = checkpoint.get("training")
    _require(isinstance(training, Mapping), "A15.2 training metadata is missing")
    _require(
        set(training) == EXPECTED_TRAINING_FIELDS,
        "A15.2 training metadata fields differ",
    )
    expected_condition_counts = {
        condition: CORRECTION_TRAIN_SAMPLES * TERMINAL_EPOCHS
        for condition in PROJECTIVE_CONDITIONS
    }
    expected_values = {
        "initialization_seed": INITIALIZATION_SEED,
        "sample_order_seed": SAMPLE_ORDER_SEED,
        "sample_order_seed_rule": "base_plus_zero_based_epoch",
        "pixel_augmentation_seed": PIXEL_AUGMENTATION_SEED,
        "pixel_augmentation_epoch_rule": (
            "same_seed_epoch_index_across_three_fixed_conditions"
        ),
        "epochs": TERMINAL_EPOCHS,
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "conditions_per_physical": CONDITIONS_PER_PHYSICAL,
        "row_batch_size_full": ROW_BATCH_SIZE,
        "last_row_batch_size": PHYSICAL_BATCH_SIZE * CONDITIONS_PER_PHYSICAL,
        "physical_samples_per_epoch": CORRECTION_TRAIN_SAMPLES,
        "condition_rows_per_epoch": (
            CORRECTION_TRAIN_SAMPLES * CONDITIONS_PER_PHYSICAL
        ),
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps": EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "optimizer": "AdamW_A15.2_correction_only",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR_Tmax5_epoch_end_step",
        "scheduler_steps": TERMINAL_EPOCHS,
        "condition_counts_terminal_total": expected_condition_counts,
        "projective_conditions": list(PROJECTIVE_CONDITIONS),
        "clean_presentations": 0,
        "gradient_clipping": None,
        "ema": None,
        "amp": "bfloat16_if_supported_else_float16_cuda",
        "validation_manifest": None,
        "intermediate_checkpoint_selection": None,
    }
    differing = [
        name
        for name, expected in expected_values.items()
        if not _schema_equal(training.get(name), expected)
    ]
    _require(
        not differing,
        f"A15.2 fixed five-epoch/4135-step training metadata differs: {differing}",
    )
    return {
        "epochs": TERMINAL_EPOCHS,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps": EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "physical_samples_per_epoch": CORRECTION_TRAIN_SAMPLES,
        "condition_rows_per_epoch": (
            CORRECTION_TRAIN_SAMPLES * CONDITIONS_PER_PHYSICAL
        ),
        "condition_counts_terminal_total": expected_condition_counts,
        "validation_manifest": None,
        "intermediate_checkpoint_selection": None,
    }


def _strict_history(checkpoint: Mapping[str, Any]) -> None:
    history = checkpoint.get("history")
    _require(
        isinstance(history, Sequence)
        and not isinstance(history, (str, bytes))
        and len(history) == TERMINAL_EPOCHS,
        "A15.2 history is not five complete epochs",
    )
    expected_condition_counts = {
        condition: CORRECTION_TRAIN_SAMPLES for condition in PROJECTIVE_CONDITIONS
    }
    for epoch_index, row in enumerate(history):
        _require(isinstance(row, Mapping), "A15.2 history row is malformed")
        _require(set(row) == EXPECTED_HISTORY_FIELDS, "A15.2 history fields differ")
        _require(
            _exact_int(row.get("epoch"), label="A15.2 history epoch")
            == epoch_index + 1
            and _exact_int(
                row.get("sample_order_seed"), label="A15.2 history order seed"
            )
            == SAMPLE_ORDER_SEED + epoch_index
            and _exact_int(
                row.get("pixel_augmentation_seed"),
                label="A15.2 history pixel seed",
            )
            == PIXEL_AUGMENTATION_SEED
            and _exact_int(
                row.get("pixel_epoch"), label="A15.2 history pixel epoch"
            )
            == epoch_index,
            "A15.2 history epoch/seed schedule differs",
        )
        learning_rate = row.get("learning_rate")
        _require(
            type(learning_rate) is float
            and math.isfinite(learning_rate)
            and 0.0 <= learning_rate <= LEARNING_RATE,
            "A15.2 history learning rate is invalid",
        )
        metrics = row.get("metrics")
        _require(isinstance(metrics, Mapping), "A15.2 history metrics are missing")
        _require(
            set(metrics) == EXPECTED_HISTORY_METRIC_FIELDS,
            "A15.2 history metric fields differ",
        )
        _require(
            _exact_int(
                metrics.get("physical_samples"),
                label="A15.2 history physical samples",
            )
            == CORRECTION_TRAIN_SAMPLES
            and _exact_int(
                metrics.get("condition_rows"),
                label="A15.2 history condition rows",
            )
            == CORRECTION_TRAIN_SAMPLES * CONDITIONS_PER_PHYSICAL
            and _exact_int(metrics.get("steps"), label="A15.2 history steps")
            == EXPECTED_STEPS_PER_EPOCH
            and _exact_int(
                metrics.get("optimizer_steps"),
                label="A15.2 history optimizer steps",
            )
            == EXPECTED_STEPS_PER_EPOCH,
            "A15.2 history row/step coverage differs",
        )
        order = metrics.get("source_index_order")
        _require(
            isinstance(order, list)
            and len(order) == CORRECTION_TRAIN_SAMPLES
            and all(type(index) is int for index in order)
            and sorted(order) == list(range(CORRECTION_TRAIN_SAMPLES)),
            "A15.2 history physical sample order is not exact-once",
        )
        _require(
            _schema_equal(metrics.get("condition_counts"), expected_condition_counts),
            "A15.2 history synchronized condition counts differ",
        )
        active_rows = _exact_int(
            metrics.get("active_rows"), label="A15.2 history active rows"
        )
        inactive_rows = _exact_int(
            metrics.get("inactive_rows"), label="A15.2 history inactive rows"
        )
        _require(
            active_rows >= 0
            and inactive_rows >= 0
            and active_rows + inactive_rows
            == CORRECTION_TRAIN_SAMPLES * CONDITIONS_PER_PHYSICAL,
            "A15.2 history active/inactive coverage differs",
        )
        _require(
            type(metrics.get("active_fraction")) is float
            and math.isclose(
                float(metrics["active_fraction"]),
                active_rows
                / float(CORRECTION_TRAIN_SAMPLES * CONDITIONS_PER_PHYSICAL),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            "A15.2 history active fraction differs",
        )
        loss_components = metrics.get("loss_components_active_row_weighted")
        _require(
            isinstance(loss_components, Mapping)
            and set(loss_components) == set(SCALAR_LOSS_COMPONENTS)
            and all(
                type(value) is float and math.isfinite(value)
                for value in loss_components.values()
            ),
            "A15.2 history A15.1 loss components differ",
        )
        _require(
            metrics.get("loss_denominator")
            == "read_valid_and_relation_available_rows_only"
            and metrics.get("cross_condition_denominator")
            == "complete_three_row_all_active_physical_groups_only"
            and metrics.get("gradient_clipping") is None,
            "A15.2 history A15.1 denominator/clipping semantics differ",
        )
        for name in ("gradient_finite_every_step", "gradient_nonzero_seen"):
            evidence = metrics.get(name)
            _require(
                isinstance(evidence, Mapping)
                and set(evidence) == set(SEMANTIC_GROUP_NAMES)
                and all(value is True for value in evidence.values()),
                f"A15.2 history semantic gradient evidence differs: {name}",
            )
        for name in (
            "optimizer_state_finite_every_step",
            "correction_state_finite_every_step",
            "anchor_runtime_contract_every_step",
            "anchor_state_bit_exact_at_epoch_end",
            "q0_output_identity_every_step",
            "inactive_exact_q0_every_step",
            "posterior_mass_cdf_every_step",
        ):
            _require(metrics.get(name) is True, f"A15.2 history evidence differs: {name}")
        _require(
            type(metrics.get("mixed_active_inactive_seen")) is bool
            and type(metrics.get("initial_active_exact_fixed_geometric")) is bool,
            "A15.2 history active-row diagnostic types differ",
        )
        trace = metrics.get("step_trace")
        _require(
            isinstance(trace, list)
            and len(trace) == EXPECTED_STEPS_PER_EPOCH
            and all(
                isinstance(item, Mapping)
                and _exact_int(item.get("step"), label="A15.2 trace step")
                == index + 1
                for index, item in enumerate(trace)
            ),
            "A15.2 history step trace does not cover 827 optimizer steps",
        )


def _strict_terminal_evidence(checkpoint: Mapping[str, Any]) -> None:
    fresh = checkpoint.get("fresh_strict_load")
    _require(
        isinstance(fresh, Mapping)
        and set(fresh) == EXPECTED_FRESH_FIELDS
        and fresh.get("missing_keys") == []
        and fresh.get("unexpected_keys") == []
        and fresh.get("strict_load_clean") is True
        and fresh.get("state_bit_exact") is True
        and fresh.get("same_input_reported_outputs_bit_exact") is True
        and fresh.get("fresh_state_finite") is True
        and _valid_posterior_evidence(fresh.get("fresh_posterior_mass_cdf")),
        "A15.2 recorded fresh strict-load evidence differs",
    )
    fallback = checkpoint.get("terminal_fallback_evidence")
    _require(
        isinstance(fallback, Mapping)
        and set(fallback) == EXPECTED_FALLBACK_FIELDS
        and fallback.get("relation_available_all_false") is True
        and fallback.get("posterior_cdf_layers_and_moments_exact_q0") is True
        and _valid_posterior_evidence(fallback.get("posterior_mass_cdf")),
        "A15.2 recorded terminal exact-q0 fallback evidence differs",
    )


def load_terminal_a15_2_correction_and_anchor(
    correction_checkpoint_path: Path,
    terminal_a11_checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[Any, A152FTEBCorrection, dict[str, Any]]:
    """Validate the complete terminal artifact, then strict-load frozen A11."""

    correction_path = Path(correction_checkpoint_path).resolve()
    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    _require(correction_path != a11_path, "A11 and A15.2 checkpoint paths must differ")
    checkpoint = _checkpoint_mapping(correction_path)
    _require(
        set(checkpoint) == EXPECTED_CHECKPOINT_FIELDS,
        "A15.2 terminal checkpoint fields differ",
    )
    _require(
        _exact_int(checkpoint.get("schema_version"), label="A15.2 schema version")
        == 1,
        "A15.2 terminal checkpoint schema differs",
    )
    _require(
        checkpoint.get("protocol") == TRAINING_PROTOCOL
        and checkpoint.get("selection_protocol") == SELECTION_PROTOCOL
        and checkpoint.get("data_protocol") == DATA_PROTOCOL
        and checkpoint.get("system") == SYSTEM,
        "A15.2 training/selection/data protocol or system differs",
    )
    _require(
        checkpoint.get("architecture") == A15_2_ARCHITECTURE
        and type(checkpoint.get("learned_residual_scale")) is float
        and checkpoint.get("learned_residual_scale")
        == A15_2_LEARNED_RESIDUAL_SCALE,
        "A15.2 architecture or fixed 0.25 residual scale differs",
    )
    _require(
        checkpoint.get("deterministic_algorithms_enabled") is True
        and checkpoint.get("automatic_execution_or_advancement_control") is False,
        "A15.2 determinism or no-automatic-control evidence differs",
    )
    validated_access = _exact_bool_mapping(
        checkpoint.get("access_flags"),
        EXPECTED_TRAIN_ACCESS_FLAGS,
        label="A15.2 training access flags",
    )

    source = checkpoint.get("source_terminal_a11")
    _require(isinstance(source, Mapping), "A15.2 source terminal A11 evidence is missing")
    _require(
        set(source)
        == {
            "source",
            "strict_load",
            "eval_mode",
            "all_parameters_requires_grad_false",
            "training_protocol",
            "training_epochs",
            "validation_manifest",
            "access_flags",
            "state_bit_exact_terminal",
            "eval_frozen_gradient_free_every_step",
        },
        "A15.2 nested terminal A11 evidence fields differ",
    )
    _require(
        isinstance(source.get("source"), str)
        and Path(str(source["source"])).resolve() == a11_path
        and source.get("strict_load") is True
        and source.get("eval_mode") is True
        and source.get("all_parameters_requires_grad_false") is True
        and source.get("training_protocol") == A11_TRAINING_PROTOCOL
        and _exact_int(
            source.get("training_epochs"), label="A15.2 nested A11 epochs"
        )
        == 5
        and source.get("validation_manifest") is None
        and source.get("state_bit_exact_terminal") is True
        and source.get("eval_frozen_gradient_free_every_step") is True,
        "A15.2 nested terminal A11 path/frozen evidence differs",
    )
    source_access = _exact_bool_mapping(
        source.get("access_flags"),
        TERMINAL_A11_ACCESS_SCHEMA,
        label="A15.2 nested terminal A11 access flags",
    )

    construction_value = checkpoint.get("construction")
    _require(isinstance(construction_value, Mapping), "A15.2 construction is missing")
    _require(
        set(construction_value) == set(EXPECTED_CONSTRUCTION_FIELDS),
        "A15.2 construction fields differ",
    )
    construction = {
        name: _exact_int(
            construction_value[name], label=f"A15.2 construction value {name}"
        )
        for name in EXPECTED_CONSTRUCTION_FIELDS
    }
    _require(
        construction["progress_bins"] == 128
        and all(value >= 1 for value in construction.values())
        and construction["token_dim"] % construction["attention_heads"] == 0,
        "A15.2 construction dimensions differ",
    )

    training_metadata = _strict_training_metadata(checkpoint)
    _strict_history(checkpoint)
    _strict_terminal_evidence(checkpoint)
    _require(
        checkpoint.get("semantic_gradient_groups") == list(SEMANTIC_GROUP_NAMES),
        "A15.2 semantic gradient groups differ",
    )
    expected_loss_design = {
        **dict(LOSS_DESIGN_METADATA),
        "loss_function": "a15_1_fteb_loss",
        "inactive_rows_excluded_from_every_correction_denominator": True,
        "cross_condition_only_complete_all_active_three_row_groups": True,
    }
    _require(
        _schema_equal(checkpoint.get("loss_design"), expected_loss_design)
        and _schema_equal(checkpoint.get("loss_weights"), LOSS_WEIGHTS.as_dict()),
        "A15.2 terminal objective is not the fixed A15.1 loss",
    )
    _require(
        _finite_optimizer_evidence(checkpoint.get("terminal_optimizer_and_scaler_state")),
        "A15.2 terminal AdamW/scaler state evidence differs",
    )

    physical = checkpoint.get("physical_correction_train")
    _require(isinstance(physical, Mapping), "A15.2 physical correction-train evidence is missing")
    _require(
        set(physical) == {"manifest", "samples", "scenes", "scene_roster", "sample_ids"},
        "A15.2 physical correction-train fields differ",
    )
    train_ids = physical.get("sample_ids")
    expected_scene_stems = sorted(Path(scene).stem for scene in CORRECTION_TRAIN_SCENES)
    _require(
        isinstance(physical.get("manifest"), str)
        and bool(physical.get("manifest"))
        and _exact_int(
            physical.get("samples"), label="A15.2 correction-train samples"
        )
        == CORRECTION_TRAIN_SAMPLES
        and _exact_int(
            physical.get("scenes"), label="A15.2 correction-train scenes"
        )
        == CORRECTION_TRAIN_SCENE_COUNT
        and physical.get("scene_roster") == expected_scene_stems
        and isinstance(train_ids, list)
        and len(train_ids) == CORRECTION_TRAIN_SAMPLES
        and all(isinstance(value, str) and bool(value) for value in train_ids)
        and len(set(train_ids)) == CORRECTION_TRAIN_SAMPLES,
        "A15.2 physical correction-train roster/count evidence differs",
    )
    unchanged = checkpoint.get("anchor_unchanged_evidence")
    _require(
        isinstance(unchanged, Mapping)
        and set(unchanged)
        == {
            "state_bit_exact_terminal",
            "state_bit_exact_each_epoch",
            "runtime_contract_every_step",
            "q0_output_identity_every_step",
        }
        and unchanged.get("state_bit_exact_terminal") is True
        and unchanged.get("state_bit_exact_each_epoch") == [True] * TERMINAL_EPOCHS
        and unchanged.get("runtime_contract_every_step") == [True] * TERMINAL_EPOCHS
        and unchanged.get("q0_output_identity_every_step") == [True] * TERMINAL_EPOCHS,
        "A15.2 terminal anchor-unchanged evidence differs",
    )

    correction = A152FTEBCorrection(**construction)
    _require(
        correction.learned_residual_scale == A15_2_LEARNED_RESIDUAL_SCALE
        and correction.use_relation_memory is True,
        "fresh A15.2 construction is not fixed-quarter full relation memory",
    )
    state = _finite_tensor_state(checkpoint.get("model_state"), label="A15.2 model state")
    _validate_state_schema(state, correction.state_dict())
    try:
        incompatibility = correction.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise A152CorrectionDevEvaluationError(
            "A15.2 model state does not strict-load into a fresh module"
        ) from exc
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "A15.2 model-state strict load has incompatible keys",
    )
    counts = fteb_parameter_counts(correction)
    _require(
        int(counts["total"]) == EXPECTED_CORRECTION_PARAMETER_COUNT
        and int(counts["correction"]) == EXPECTED_CORRECTION_PARAMETER_COUNT
        and int(counts["component_sum"]) == EXPECTED_CORRECTION_PARAMETER_COUNT
        and int(counts["trainable"]) == EXPECTED_CORRECTION_PARAMETER_COUNT,
        "fresh A15.2 model is not the 169724-parameter correction",
    )
    _require(
        _schema_equal(checkpoint.get("parameter_counts"), counts),
        "A15.2 parameter-count evidence differs",
    )
    partition = checkpoint.get("optimizer_partition_evidence")
    _require(
        isinstance(partition, Mapping)
        and set(partition)
        == {
            "optimizer_exactly_correction_parameters",
            "optimizer_excludes_terminal_a11",
            "all_correction_parameters_trainable",
            "all_terminal_a11_parameters_frozen",
            "correction_parameter_count",
            "optimizer_parameter_count",
        }
        and partition.get("optimizer_exactly_correction_parameters") is True
        and partition.get("optimizer_excludes_terminal_a11") is True
        and partition.get("all_correction_parameters_trainable") is True
        and partition.get("all_terminal_a11_parameters_frozen") is True
        and _exact_int(
            partition.get("correction_parameter_count"),
            label="A15.2 partition correction count",
        )
        == EXPECTED_CORRECTION_PARAMETER_COUNT
        and _exact_int(
            partition.get("optimizer_parameter_count"),
            label="A15.2 partition optimizer count",
        )
        == EXPECTED_CORRECTION_PARAMETER_COUNT,
        "A15.2 correction-only optimizer partition differs",
    )

    # Only after the entire correction artifact is validated do we open A11.
    anchor, a11_evidence, a11_construction = load_frozen_terminal_a11(
        a11_path, device=device
    )
    _require(
        _schema_equal(a11_construction, construction),
        "terminal A11 and A15.2 construction differ",
    )
    correction.eval().to(device)
    _require(
        all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in correction.state_dict().values()
        ),
        "fresh A15.2 correction state is non-finite",
    )
    metadata = {
        "terminal_a11_checkpoint": str(a11_path),
        "terminal_a15_2_correction_checkpoint": str(correction_path),
        "training_protocol": TRAINING_PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "system": SYSTEM,
        "architecture": A15_2_ARCHITECTURE,
        "learned_residual_scale": A15_2_LEARNED_RESIDUAL_SCALE,
        "training_access_flags": validated_access,
        "source_terminal_a11_access_flags": source_access,
        "source_terminal_a11_fresh_strict_load": bool(a11_evidence["strict_load"]),
        "training": training_metadata,
        "physical_correction_train_samples": CORRECTION_TRAIN_SAMPLES,
        "physical_correction_train_scenes": CORRECTION_TRAIN_SCENE_COUNT,
        "physical_correction_train_sample_ids": list(train_ids),
        "anchor_unchanged": True,
        "terminal_correction_adam_and_scaler_finite": True,
        "terminal_fallback_exact_q0": True,
        "fresh_correction_strict_load": True,
        "parameter_counts": counts,
    }
    return anchor, correction, metadata


def build_a15_2_dev_evaluation_dataset(
    samples: Sequence[Any],
    *,
    seed: int,
    total_epochs: int,
    condition: str,
) -> Any:
    """Reuse A13/A14's fixed physical-dev pixel construction exactly."""

    try:
        return a13_evaluation.build_a13_dev_evaluation_dataset(
            samples,
            seed=seed,
            total_epochs=total_epochs,
            condition=condition,
        )
    except ValueError as exc:
        raise A152CorrectionDevEvaluationError(str(exc)) from exc


def _tensor_output(
    output: Mapping[str, Any], field: str, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    value = output.get(field)
    _require(isinstance(value, torch.Tensor), f"A15.2 output tensor is missing: {field}")
    if shape is not None:
        _require(value.shape == shape, f"A15.2 output shape differs: {field}")
    return value


def _exact_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _require(left.shape == right.shape, "A15.2 exact comparison shapes differ")
    if left.ndim == 1:
        return left == right
    return (left == right).reshape(left.shape[0], -1).all(dim=1)


def _fallback_exact_rows(output: Mapping[str, Any]) -> torch.Tensor:
    q0 = _tensor_output(output, "raw_anchor_posterior")
    batch, bins = q0.shape
    q0_cdf = _tensor_output(output, "raw_anchor_cdf", (batch, bins))
    q0_mean = _tensor_output(output, "raw_anchor_mean", (batch,))
    q0_variance = _tensor_output(output, "raw_anchor_variance", (batch,))
    layers = _tensor_output(
        output, "layer_posteriors", (batch, BRIDGE_LAYERS, bins)
    )
    layer_cdfs = _tensor_output(
        output, "layer_cdfs", (batch, BRIDGE_LAYERS, bins)
    )
    layer_means = _tensor_output(output, "layer_means", (batch, BRIDGE_LAYERS))
    layer_variances = _tensor_output(
        output, "layer_variances", (batch, BRIDGE_LAYERS)
    )
    columns = (
        _exact_rows(_tensor_output(output, "sarn_endpoint_posterior", q0.shape), q0),
        _exact_rows(_tensor_output(output, "sarn_endpoint_cdf", q0.shape), q0_cdf),
        _exact_rows(_tensor_output(output, "sarn_endpoint_mean", (batch,)), q0_mean),
        _exact_rows(_tensor_output(output, "geometric_base", q0.shape), q0),
        _exact_rows(_tensor_output(output, "geometric_base_cdf", q0.shape), q0_cdf),
        _exact_rows(_tensor_output(output, "geometric_base_mean", (batch,)), q0_mean),
        _exact_rows(_tensor_output(output, "progress_posterior", q0.shape), q0),
        _exact_rows(_tensor_output(output, "progress_cdf", q0.shape), q0_cdf),
        _exact_rows(_tensor_output(output, "mean", (batch,)), q0_mean),
        _exact_rows(_tensor_output(output, "variance", (batch,)), q0_variance),
        _exact_rows(layers, q0[:, None].expand_as(layers)),
        _exact_rows(layer_cdfs, q0_cdf[:, None].expand_as(layer_cdfs)),
        _exact_rows(layer_means, q0_mean[:, None].expand_as(layer_means)),
        _exact_rows(
            layer_variances, q0_variance[:, None].expand_as(layer_variances)
        ),
        ~_tensor_output(
            output, "correction_active", (batch, BRIDGE_LAYERS)
        ).any(dim=1),
    )
    return torch.stack(columns, dim=1).all(dim=1)


def _posterior_row_diagnostics(posterior: torch.Tensor) -> list[dict[str, Any]]:
    value = posterior.detach().float().cpu()
    _require(
        value.ndim == 2 and value.shape[1] == 128 and bool(torch.isfinite(value).all()),
        "A15.2 posterior diagnostics input differs",
    )
    cdf = value.cumsum(dim=1)
    return [
        {
            "absolute_mass_error": float(abs(value[index].sum().item() - 1.0)),
            "negative_probability_count": int((value[index] < -1.0e-6).sum()),
            "cdf_monotonic_violation_count": int(
                ((cdf[index, 1:] - cdf[index, :-1]) < -1.0e-6).sum()
            ),
            "absolute_cdf_terminal_error": float(abs(cdf[index, -1].item() - 1.0)),
        }
        for index in range(value.shape[0])
    ]


def _layer_row_diagnostics(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    layers_device = _tensor_output(output, "layer_posteriors").detach().float()
    layer_cdfs_device = _tensor_output(output, "layer_cdfs").detach().float()
    _require(
        layers_device.ndim == 3
        and layers_device.shape[1:] == (BRIDGE_LAYERS, 128)
        and layer_cdfs_device.shape == layers_device.shape
        and layer_cdfs_device.device == layers_device.device
        and bool(torch.isfinite(layers_device).all())
        and bool(torch.isfinite(layer_cdfs_device).all()),
        "A15.2 layer posterior/CDF diagnostics differ",
    )
    # CDF serialization happened on the model device.  Recompute with the
    # same per-layer Bx128 operator shape there before asking for bit identity;
    # CPU and CUDA cumsum may differ by an ULP even after identical fp32 values
    # are copied between devices.  Numeric diagnostics can move to CPU only
    # after the source-device reduction is complete.
    calculated_device = torch.stack(
        [
            layers_device[:, index].cumsum(dim=1)
            for index in range(BRIDGE_LAYERS)
        ],
        dim=1,
    )
    serialized_exact_rows = (
        layer_cdfs_device == calculated_device
    ).reshape(layers_device.shape[0], -1).all(dim=1)
    layers = layers_device.cpu()
    calculated = calculated_device.cpu()
    serialized_exact_rows_cpu = serialized_exact_rows.cpu()
    result = []
    for index in range(layers.shape[0]):
        mass_error = torch.abs(layers[index].sum(dim=1) - 1.0)
        cdf_steps = calculated[index, :, 1:] - calculated[index, :, :-1]
        result.append(
            {
                "layers": BRIDGE_LAYERS,
                "maximum_absolute_mass_error": float(mass_error.max()),
                "negative_probability_count": int((layers[index] < -1.0e-6).sum()),
                "cdf_monotonic_violation_count": int((cdf_steps < -1.0e-6).sum()),
                "maximum_absolute_cdf_terminal_error": float(
                    torch.abs(calculated[index, :, -1] - 1.0).max()
                ),
                "serialized_cdfs_exact_from_posteriors": bool(
                    serialized_exact_rows_cpu[index]
                ),
            }
        )
    return result


def _row_integrity_failure(
    method_diagnostics: Mapping[str, Mapping[str, Any]],
    layer_diagnostics: Mapping[str, Any],
) -> dict[str, Any] | None:
    checks = (
        ("absolute_mass_error", lambda value: float(value) <= 1.0e-6, 1.0e-6),
        ("negative_probability_count", lambda value: int(value) == 0, 0),
        ("cdf_monotonic_violation_count", lambda value: int(value) == 0, 0),
        (
            "absolute_cdf_terminal_error",
            lambda value: float(value) <= 1.0e-6,
            1.0e-6,
        ),
    )
    for method in METHOD_ORDER:
        values = method_diagnostics[method]
        for field, valid, expected in checks:
            actual = values[field]
            if not valid(actual):
                return {
                    "method": method,
                    "field": field,
                    "actual": actual,
                    "expected_maximum_or_exact": expected,
                }
    layer_checks = (
        (
            "maximum_absolute_mass_error",
            lambda value: float(value) <= 1.0e-6,
            1.0e-6,
        ),
        ("negative_probability_count", lambda value: int(value) == 0, 0),
        ("cdf_monotonic_violation_count", lambda value: int(value) == 0, 0),
        (
            "maximum_absolute_cdf_terminal_error",
            lambda value: float(value) <= 1.0e-6,
            1.0e-6,
        ),
        (
            "serialized_cdfs_exact_from_posteriors",
            lambda value: value is True,
            True,
        ),
    )
    for field, valid, expected in layer_checks:
        actual = layer_diagnostics[field]
        if not valid(actual):
            return {
                "method": "a15_2_layer_path",
                "field": field,
                "actual": actual,
                "expected_maximum_or_exact": expected,
            }
    return None


def _method_summary(
    *, predictions: Mapping[str, Sequence[float]], target: Sequence[float]
) -> dict[str, Any]:
    _require(set(predictions) == set(METHOD_ORDER), "A15.2 metric methods differ")
    truth = tuple(float(value) for value in target)
    _require(bool(truth) and all(math.isfinite(value) for value in truth), "A15.2 targets differ")
    values = {method: tuple(float(value) for value in predictions[method]) for method in METHOD_ORDER}
    _require(
        all(len(row) == len(truth) and all(math.isfinite(value) for value in row) for row in values.values()),
        "A15.2 prediction vectors differ",
    )
    metrics: dict[str, Any] = {}
    for method in METHOD_ORDER:
        errors = [abs(value - expected) for value, expected in zip(values[method], truth, strict=True)]
        tail_count = int(math.ceil(0.25 * len(errors)))
        metrics[method] = {
            "samples": len(errors),
            "nmae": sum(errors) / len(errors),
            "cvar25": sum(sorted(errors, reverse=True)[:tail_count]) / tail_count,
            "cvar_tail_fraction": 0.25,
            "cvar_tail_count": tail_count,
        }
    versus_q0 = {
        method: paired_prediction_summary(
            candidate_mean=values[method],
            reference_mean=values[METHOD_Q0],
            target=truth,
        )
        for method in METHOD_ORDER
    }
    return {
        "rows": len(truth),
        "method_order": list(METHOD_ORDER),
        "methods": metrics,
        "versus_q0": versus_q0,
        "a15_2_minus_fixed_geometric": paired_prediction_summary(
            candidate_mean=values[METHOD_A15_2],
            reference_mean=values[METHOD_FIXED_GEOMETRIC],
            target=truth,
        ),
    }


def evaluate_same_twin_endpoint_loader(
    anchor: Any,
    correction: A152FTEBCorrection,
    loader: Any,
    *,
    device: torch.device,
    expected_sample_ids: Sequence[str] | None = None,
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Evaluate all four methods from one Raw/SARN twin-endpoint call per batch."""

    anchor.eval()
    correction.eval()
    rows: list[dict[str, Any]] = []
    twin_endpoint_forward_batches = 0
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            _require(isinstance(raw_batch, Mapping), "A15.2 dev batch is not a mapping")
            original = raw_batch["original_view"].to(device)
            sarn = raw_batch["sarn_view"].to(device)
            support = raw_batch["sarn_support_mask"].to(device)
            dataset_active = raw_batch["sarn_active"].to(device).bool()
            homography = raw_batch["raw_to_sarn_homography"].to(device)
            target = raw_batch["target"].to(device).float()
            batch = int(target.shape[0])
            ids = raw_batch.get("sample_id")
            scenes = raw_batch.get("scene_stem")
            names = raw_batch.get("condition_name")
            _require(
                isinstance(ids, Sequence)
                and not isinstance(ids, (str, bytes))
                and len(ids) == batch
                and isinstance(scenes, Sequence)
                and not isinstance(scenes, (str, bytes))
                and len(scenes) == batch
                and isinstance(names, Sequence)
                and not isinstance(names, (str, bytes))
                and len(names) == batch,
                "A15.2 dev IDs/scenes/conditions are batch-misaligned",
            )
            if expected_condition is not None:
                _require(
                    all(str(name) == expected_condition for name in names),
                    "A15.2 dev condition pixels differ from the requested arm",
                )
            clean_rows = torch.tensor(
                [str(name) == "clean" for name in names],
                device=device,
                dtype=torch.bool,
            )
            effective_active = dataset_active & ~clean_rows
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                twin_endpoint_forward_batches += 1
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
                        "sarn_active": effective_active,
                        "raw_to_sarn_homography": homography,
                    },
                    endpoint_null=False,
                )

            q0 = _tensor_output(output, "raw_anchor_posterior", (batch, 128)).float()
            proposed_qs = endpoints["sarn_posterior"].float()
            relation = _tensor_output(output, "relation_available", (batch,))
            _require(relation.dtype == torch.bool, "A15.2 relation availability must be boolean")
            _require(
                torch.equal(q0, endpoints["raw_posterior"].float())
                and torch.equal(
                    _tensor_output(output, "proposed_sarn_endpoint_posterior", (batch, 128)).float(),
                    proposed_qs,
                ),
                "A15.2 output does not preserve the same twin endpoints",
            )
            qs = _tensor_output(output, "sarn_endpoint_posterior", (batch, 128)).float()
            fixed = _tensor_output(output, "geometric_base", (batch, 128)).float()
            learned = _tensor_output(output, "progress_posterior", (batch, 128)).float()
            proposed_fixed = fixed_geometric_natural_parameter_base(q0, proposed_qs)[
                "geometric_base"
            ]
            expected_fixed = torch.where(relation[:, None], proposed_fixed, q0)
            _require(
                torch.equal(fixed, expected_fixed),
                "A15.2 fixed geometry is not recomputed from the same endpoints",
            )
            q0_mean = _tensor_output(output, "raw_anchor_mean", (batch,)).float()
            qs_mean = _tensor_output(output, "sarn_endpoint_mean", (batch,)).float()
            fixed_mean = _tensor_output(output, "geometric_base_mean", (batch,)).float()
            learned_mean = _tensor_output(output, "mean", (batch,)).float()
            posteriors = {
                METHOD_Q0: q0,
                METHOD_QS: qs,
                METHOD_FIXED_GEOMETRIC: fixed,
                METHOD_A15_2: learned,
            }
            means = {
                METHOD_Q0: q0_mean,
                METHOD_QS: qs_mean,
                METHOD_FIXED_GEOMETRIC: fixed_mean,
                METHOD_A15_2: learned_mean,
            }
            diagnostics = {
                method: _posterior_row_diagnostics(value)
                for method, value in posteriors.items()
            }
            layer_diagnostics = _layer_row_diagnostics(output)
            fallback_exact = _fallback_exact_rows(output)
            unavailable = ~relation
            _require(
                bool(fallback_exact[unavailable].all()),
                "A15.2 relation-unavailable row is not exact q0",
            )
            _require(
                not bool(relation[clean_rows].any())
                and bool(fallback_exact[clean_rows].all()),
                "A15.2 clean row is not exact q0",
            )
            explicit_cdfs = {
                METHOD_Q0: _tensor_output(output, "raw_anchor_cdf", (batch, 128)),
                METHOD_QS: _tensor_output(output, "sarn_endpoint_cdf", (batch, 128)),
                METHOD_FIXED_GEOMETRIC: _tensor_output(output, "geometric_base_cdf", (batch, 128)),
                METHOD_A15_2: _tensor_output(output, "progress_cdf", (batch, 128)),
            }
            _require(
                all(
                    torch.equal(explicit_cdfs[method].float(), posteriors[method].cumsum(dim=1))
                    for method in METHOD_ORDER
                ),
                "A15.2 serialized method CDF differs from its same-forward posterior",
            )
            for index in range(batch):
                row_target = float(target[index])
                row_means = {
                    method: float(means[method][index]) for method in METHOD_ORDER
                }
                row_diagnostics = {
                    method: diagnostics[method][index] for method in METHOD_ORDER
                }
                integrity_failure = _row_integrity_failure(
                    row_diagnostics, layer_diagnostics[index]
                )
                _require(
                    integrity_failure is None,
                    "A15.2 per-row posterior mass/CDF integrity differs: "
                    + json.dumps(
                        {
                            "condition": str(names[index]),
                            "batch_index": batch_index,
                            "batch_row_index": index,
                            "sample_id": str(ids[index]),
                            "scene_stem": str(scenes[index]),
                            "failure": integrity_failure,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                )
                rows.append(
                    {
                        "row_index": len(rows),
                        "sample_id": str(ids[index]),
                        "scene_stem": str(scenes[index]),
                        "condition": str(names[index]),
                        "normalized_target": row_target,
                        "mean": row_means,
                        "absolute_error": {
                            method: abs(value - row_target)
                            for method, value in row_means.items()
                        },
                        "relation_available": bool(relation[index]),
                        "dataset_sarn_active": bool(dataset_active[index]),
                        "effective_sarn_active": bool(effective_active[index]),
                        "fallback": {
                            "clean_forced_off": bool(clean_rows[index]),
                            "relation_unavailable": bool(unavailable[index]),
                            "q_sarn_fixed_geometric_a15_2_path_exact_q0": bool(
                                fallback_exact[index]
                            ),
                        },
                        "posterior_mass_cdf": row_diagnostics,
                        "a15_2_layer_mass_cdf": layer_diagnostics[index],
                    }
                )

    _require(bool(rows), "A15.2 correction-dev evaluation produced no rows")
    ids = tuple(str(row["sample_id"]) for row in rows)
    _require(len(ids) == len(set(ids)), "A15.2 dev sample IDs are duplicated")
    if expected_sample_ids is not None:
        _require(
            ids == tuple(str(value) for value in expected_sample_ids),
            "A15.2 dev roster/order differs",
        )
    predictions = {
        method: [float(row["mean"][method]) for row in rows]
        for method in METHOD_ORDER
    }
    metrics = _method_summary(
        predictions=predictions,
        target=[float(row["normalized_target"]) for row in rows],
    )
    return {
        "rows": len(rows),
        "same_twin_endpoint_forward_for_all_methods": True,
        "twin_endpoint_forward_batches": twin_endpoint_forward_batches,
        "raw_and_sarn_anchor_calls_per_twin_forward": 2,
        "relation_available_rows": sum(bool(row["relation_available"]) for row in rows),
        "relation_unavailable_rows": sum(not bool(row["relation_available"]) for row in rows),
        "clean_forced_off_rows": sum(bool(row["fallback"]["clean_forced_off"]) for row in rows),
        "relation_unavailable_exact_q0_rows": sum(
            bool(row["fallback"]["relation_unavailable"])
            and bool(row["fallback"]["q_sarn_fixed_geometric_a15_2_path_exact_q0"])
            for row in rows
        ),
        "posterior_mass_cdf_all_valid": True,
        "metrics": metrics,
        "per_sample": rows,
    }


def pool_condition_results(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool all four conditions and the three projective conditions safely."""

    _require(
        set(conditions) == set(EVALUATION_CONDITIONS),
        "A15.2 condition result set differs",
    )
    typed: dict[str, list[Mapping[str, Any]]] = {}
    for condition in EVALUATION_CONDITIONS:
        rows = conditions[condition].get("per_sample")
        _require(
            isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
            f"A15.2 per-sample evidence is missing: {condition}",
        )
        typed[condition] = list(rows)
    reference_ids = tuple(str(row["sample_id"]) for row in typed["clean"])
    reference_targets = tuple(float(row["normalized_target"]) for row in typed["clean"])
    for condition, rows in typed.items():
        _require(
            tuple(str(row["sample_id"]) for row in rows) == reference_ids
            and tuple(float(row["normalized_target"]) for row in rows)
            == reference_targets,
            f"A15.2 condition roster/order/targets differ: {condition}",
        )

    def pool(names: Sequence[str]) -> dict[str, Any]:
        rows = [row for name in names for row in typed[name]]
        predictions = {
            method: [float(row["mean"][method]) for row in rows]
            for method in METHOD_ORDER
        }
        return {
            "conditions": list(names),
            "physical_correction_dev_samples": len(reference_ids),
            **_method_summary(
                predictions=predictions,
                target=[float(row["normalized_target"]) for row in rows],
            ),
        }

    return {
        "all_conditions": pool(EVALUATION_CONDITIONS),
        "projective_conditions": pool(PROJECTIVE_CONDITIONS),
    }


def _configure_reproducibility(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def evaluate_a15_2_correction_dev_once(
    *,
    dev_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    terminal_a15_2_correction_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Run one create-once A15.2 correction-dev replay without a gate."""

    _require(int(workers) >= 0, "A15.2 dev workers must be non-negative")
    _require(
        1 <= int(batch_size) <= EVALUATION_BATCH_SIZE,
        "A15.2 dev batch size differs",
    )
    dev_manifest = Path(dev_manifest_path).resolve()
    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    a15_2_path = Path(terminal_a15_2_correction_checkpoint_path).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A15.2 dev output already exists: {output}")
    _require(
        len({dev_manifest, a11_path, a15_2_path, output}) == 4,
        "A15.2 dev manifest/checkpoints/output paths must all differ",
    )
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "A15.2 dev device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(EVALUATION_SEED, device)

    # Terminal validation deliberately precedes any development-manifest read.
    anchor, correction, checkpoint_metadata = load_terminal_a15_2_correction_and_anchor(
        a15_2_path, a11_path, device=device
    )
    dev_samples = tuple(load_a13_correction_dev_manifest(dev_manifest))
    _require(
        len(dev_samples) == CORRECTION_DEV_SAMPLES,
        "A15.2 correction-dev physical sample count differs",
    )
    expected_ids = tuple(sample.sample_id for sample in dev_samples)
    train_ids = set(checkpoint_metadata.pop("physical_correction_train_sample_ids"))
    _require(
        not (train_ids & set(expected_ids)),
        "A15.2 correction train/dev sample IDs overlap",
    )
    _require(
        not (
            set(Path(scene).stem for scene in CORRECTION_TRAIN_SCENES)
            & {str(sample.scene_stem) for sample in dev_samples}
        ),
        "A15.2 correction train/dev scenes overlap",
    )

    specs = condition_evaluation_specs(EVALUATION_SEED)
    _require(
        tuple(spec.condition for spec in specs) == EVALUATION_CONDITIONS
        and all(
            spec.dataset_total_epochs == EVALUATION_PIXEL_TOTAL_EPOCHS
            for spec in specs
        )
        and all(spec.transform_epoch == EVALUATION_PIXEL_EPOCH for spec in specs),
        "A15.2 fixed four-condition/epoch specification differs",
    )
    conditions: dict[str, dict[str, Any]] = {}
    clean_replay: dict[str, Any] | None = None
    for spec in specs:
        dataset = build_a15_2_dev_evaluation_dataset(
            dev_samples,
            seed=spec.dataset_seed,
            total_epochs=spec.dataset_total_epochs,
            condition=spec.condition,
        )
        dataset.set_epoch(spec.transform_epoch)
        if spec.condition == "clean":
            clean_replay = verify_clean_epoch4_exact_replay(
                dataset, expected_sample_ids=expected_ids
            )
        loader = _loader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            workers=int(workers),
            seed=spec.loader_seed,
            cuda=device.type == "cuda",
        )
        conditions[spec.condition] = evaluate_same_twin_endpoint_loader(
            anchor,
            correction,
            loader,
            device=device,
            expected_sample_ids=expected_ids,
            expected_condition=spec.condition,
        )
    _require(clean_replay is not None, "A15.2 clean exact-replay evidence is missing")
    clean = conditions["clean"]
    _require(
        clean["relation_available_rows"] == 0
        and clean["clean_forced_off_rows"] == CORRECTION_DEV_SAMPLES
        and clean["relation_unavailable_exact_q0_rows"] == CORRECTION_DEV_SAMPLES,
        "A15.2 clean condition is not all-row exact q0",
    )
    pooled = pool_condition_results(conditions)
    projective = pooled["projective_conditions"]["a15_2_minus_fixed_geometric"]
    projective_vs_q0 = pooled["projective_conditions"]["versus_q0"][METHOD_A15_2]
    observations = {
        "projective_a15_2_nmae_delta_minus_q0": projective_vs_q0[
            "nmae_delta_candidate_minus_reference"
        ],
        "projective_a15_2_cvar25_delta_minus_q0": projective_vs_q0[
            "cvar25_delta_candidate_minus_reference"
        ],
        "projective_a15_2_net_paired_win_minus_loss_vs_q0": projective_vs_q0[
            "net_paired_win_minus_loss"
        ],
        "projective_a15_2_nmae_delta_minus_fixed_geometric": projective[
            "nmae_delta_candidate_minus_reference"
        ],
        "projective_a15_2_cvar25_delta_minus_fixed_geometric": projective[
            "cvar25_delta_candidate_minus_reference"
        ],
        "clean_all_four_methods_exact_q0": True,
        "all_relation_unavailable_rows_exact_q0": all(
            int(result["relation_unavailable_exact_q0_rows"])
            == int(result["relation_unavailable_rows"])
            for result in conditions.values()
        ),
        "posterior_mass_cdf_all_valid": all(
            bool(result["posterior_mass_cdf_all_valid"])
            for result in conditions.values()
        ),
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "scope": {
            "physical_correction_dev_only": True,
            "q0_has_seen_every_dev_scene": True,
            "correction_scene_disjoint_from_correction_train": True,
            "system_heldout_interpretation": False,
            "development_interpretation": DEVELOPMENT_INTERPRETATION,
            "correction_train_content_access": False,
            "previous_core_audit_content_access": False,
            "fold_a_content_access": False,
            "fold_b_content_access": False,
            "formal_holdout_content_access": False,
            "field_photo_content_access": False,
            "automatic_model_selection_or_advancement": False,
        },
        "comparison": {
            "single_terminal_a11_q0": True,
            "single_terminal_a15_2_correction": True,
            "one_twin_endpoint_forward_per_batch": True,
            "raw_and_sarn_anchor_forward_calls_separate": True,
            "same_endpoints_for_q0_q_sarn_fixed_geometric_and_a15_2": True,
            "fixed_geometric_recomputed_from_same_endpoints": True,
            "clean_forced_to_exact_q0": True,
            "relation_unavailable_forced_to_exact_q0": True,
        },
        "checkpoint": checkpoint_metadata,
        "data": {
            "physical_correction_dev_manifest": str(dev_manifest),
            "physical_correction_dev_samples": len(dev_samples),
            "physical_correction_dev_scenes": sorted(
                {str(sample.scene_stem) for sample in dev_samples}
            ),
            "conditions": list(EVALUATION_CONDITIONS),
            "pixel_total_epochs": EVALUATION_PIXEL_TOTAL_EPOCHS,
            "transform_epoch": EVALUATION_PIXEL_EPOCH,
            "condition_sample_rosters_exact_manifest_order": True,
        },
        "clean_exact_replay": clean_replay,
        "conditions": conditions,
        "pooled": pooled,
        "observations": observations,
    }
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise A152CorrectionDevEvaluationError(
            f"A15.2 dev output already exists: {output}"
        ) from exc
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--terminal-a11-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--terminal-a15-2-correction-checkpoint", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_a15_2_correction_dev_once(
        dev_manifest_path=args.dev_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        terminal_a15_2_correction_checkpoint_path=(
            args.terminal_a15_2_correction_checkpoint
        ),
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "pooled": result["pooled"],
                "observations": result["observations"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A152CorrectionDevEvaluationError",
    "EVALUATION_BATCH_SIZE",
    "EXPECTED_CORRECTION_PARAMETER_COUNT",
    "METHOD_A15_2",
    "METHOD_FIXED_GEOMETRIC",
    "METHOD_ORDER",
    "METHOD_Q0",
    "METHOD_QS",
    "PROTOCOL",
    "build_a15_2_dev_evaluation_dataset",
    "build_argument_parser",
    "evaluate_a15_2_correction_dev_once",
    "evaluate_same_twin_endpoint_loader",
    "load_terminal_a15_2_correction_and_anchor",
    "pool_condition_results",
]
