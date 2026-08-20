"""Fresh terminal A11 SCORT replay on the physical Core-audit roster only.

The evaluator accepts one physical Core-audit manifest, one terminal A11
checkpoint, and one create-once JSON output.  It has no input path for Core
training rows, Fold-A, Fold-B, a formal holdout, or field photographs.  The
four A10 pixel-condition implementations are reused at their fixed zero-based
transform epoch 4.  Final and q0 predictions always come from the same A11
forward; a separate SARN-unavailable control forward checks the model's exact
q0 fallback contract.

The protocol's Core screen is descriptive.  Its predeclared reference is
serialized beside observations but is never evaluated as a pass/fail gate.
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

from experiments.a10_pccot_protocol import (
    condition_evaluation_specs,
)
from experiments.a11_scort import (
    A11_ARCHITECTURE,
    A11SCORTImageModel,
    image_model_parameter_counts,
)
from experiments.a11_scort_protocol import (
    CORE_TRAIN_SAMPLES,
    CORE_TRAIN_SCENE_COUNT,
    CORE_SCREEN_REFERENCE,
    CORRECTION_LAYERS,
    EVALUATION_CONDITIONS,
    PROGRESS_BINS,
    PROJECTIVE_CONDITIONS,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
)
from experiments.a11_scort_targets import LOSS_DESIGN_METADATA, LOSS_WEIGHTS
from experiments.prepare_a11_core_scene_split import (
    PROTOCOL as MANIFEST_PROTOCOL,
    load_a11_audit_manifest,
    validate_a11_audit_samples,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _loader,
)
from experiments.train_a11_scort_syncg import (
    BATCH_SIZE as TRAIN_BATCH_SIZE,
    DEFAULT_SEED,
    FIXED_TRAIN_CONDITION_MIX,
    LEARNING_RATE,
    PROTOCOL as TRAINING_PROTOCOL,
    STEPS_PER_EPOCH,
    TERMINAL_EPOCHS,
    TOTAL_DATA_STEPS as OPTIMIZER_STEPS_EACH,
    WEIGHT_DECAY,
)
from experiments.resnet18_direct_progress import IMAGE_SIZE


PROTOCOL: Final[str] = "a11_scort_terminal_physical_core_audit_same_forward_v1"
EVALUATION_PIXEL_TOTAL_EPOCHS: Final[int] = 5
EVALUATION_PIXEL_EPOCH: Final[int] = 4
EVALUATION_BATCH_SIZE: Final[int] = 32
PAIRED_TIE_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12
POSTERIOR_MASS_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-5
CDF_MONOTONIC_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-7
CVaR_TAIL_FRACTION: Final[float] = 0.25

CHECKPOINT_CONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "imagenet_pretrained",
    "relation_channels",
    "token_dim",
    "attention_heads",
    "decoder_layers",
    "memory_grid_size",
    "progress_bins",
)
EXPECTED_CHECKPOINT_ACCESS_FLAGS: Final[dict[str, bool]] = {
    "train_manifest_access": True,
    "audit_manifest_access": False,
    "core_audit_predictions_generated": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}
MODEL_INPUT_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "original_view",
    "sarn_view",
    "sarn_support_mask",
    "sarn_active",
    "raw_to_sarn_homography",
    "target",
)


class A11CoreAuditEvaluationError(ValueError):
    """The audit roster, terminal artifact, model output, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A11CoreAuditEvaluationError(message)


def _finite_nonnegative_errors(
    values: Sequence[float], *, label: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    _require(bool(result), f"{label} error vector is empty")
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in result),
        f"{label} errors must be finite and non-negative",
    )
    return result


def _optimizer_and_scaler_state_evidence_is_finite(value: Any) -> bool:
    """Validate the trainer's numeric-state summary without loading state itself."""

    if not isinstance(value, Mapping) or value.get("finite") is not True:
        return False
    optimizer = value.get("optimizer")
    scaler = value.get("scaler")
    if not isinstance(optimizer, Mapping) or not isinstance(scaler, Mapping):
        return False
    count_fields = (
        "parameter_state_count",
        "exp_avg_tensor_count",
        "exp_avg_sq_tensor_count",
    )
    return bool(
        optimizer.get("finite") is True
        and optimizer.get("adam_moving_averages_present") is True
        and all(
            isinstance(optimizer.get(field), int)
            and not isinstance(optimizer.get(field), bool)
            and int(optimizer[field]) > 0
            for field in count_fields
        )
        and scaler.get("finite") is True
    )


def _require_two_optimizer_state_evidence(value: Any, *, label: str) -> None:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"raw", "correction"}
        and all(
            _optimizer_and_scaler_state_evidence_is_finite(value.get(arm))
            for arm in ("raw", "correction")
        ),
        f"{label} optimizer/scaler finite-state evidence differs",
    )


def paired_win_tie_loss(
    final_absolute_error: Sequence[float],
    q0_absolute_error: Sequence[float],
    *,
    tie_absolute_tolerance: float = PAIRED_TIE_ABSOLUTE_TOLERANCE,
) -> dict[str, float | int]:
    """Count paired Full-versus-q0 outcomes without controlling execution."""

    final = _finite_nonnegative_errors(final_absolute_error, label="final")
    q0 = _finite_nonnegative_errors(q0_absolute_error, label="q0")
    _require(len(final) == len(q0), "paired final/q0 error lengths differ")
    tolerance = float(tie_absolute_tolerance)
    _require(
        math.isfinite(tolerance) and tolerance >= 0.0,
        "paired tie tolerance is invalid",
    )
    wins = ties = losses = 0
    for final_error, q0_error in zip(final, q0, strict=True):
        difference = final_error - q0_error
        if abs(difference) <= tolerance:
            ties += 1
        elif difference < 0.0:
            wins += 1
        else:
            losses += 1
    samples = len(final)
    return {
        "samples": samples,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / samples,
        "tie_rate": ties / samples,
        "loss_rate": losses / samples,
        "net_wins_minus_losses": wins - losses,
        "net_paired_win_margin": (wins - losses) / samples,
        "tie_absolute_tolerance": tolerance,
    }


def cvar25_summary(
    final_absolute_error: Sequence[float],
    q0_absolute_error: Sequence[float],
) -> dict[str, float | int]:
    """Compute separate worst-ceil(25%) tails for final and q0 errors."""

    final = _finite_nonnegative_errors(final_absolute_error, label="final")
    q0 = _finite_nonnegative_errors(q0_absolute_error, label="q0")
    _require(len(final) == len(q0), "CVaR final/q0 error lengths differ")
    tail_count = int(math.ceil(CVaR_TAIL_FRACTION * len(final)))
    final_tail = sorted(final, reverse=True)[:tail_count]
    q0_tail = sorted(q0, reverse=True)[:tail_count]
    final_cvar = sum(final_tail) / tail_count
    q0_cvar = sum(q0_tail) / tail_count
    return {
        "tail_fraction": CVaR_TAIL_FRACTION,
        "tail_count_each": tail_count,
        "final_absolute_error_cvar25": final_cvar,
        "q0_absolute_error_cvar25": q0_cvar,
        "cvar25_delta_final_minus_q0": final_cvar - q0_cvar,
    }


def summarize_error_vectors(
    final_absolute_error: Sequence[float],
    q0_absolute_error: Sequence[float],
) -> dict[str, Any]:
    final = _finite_nonnegative_errors(final_absolute_error, label="final")
    q0 = _finite_nonnegative_errors(q0_absolute_error, label="q0")
    _require(len(final) == len(q0), "summary final/q0 error lengths differ")
    samples = len(final)
    final_nmae = sum(final) / samples
    q0_nmae = sum(q0) / samples
    return {
        "samples": samples,
        "final_nmae": final_nmae,
        "q0_nmae": q0_nmae,
        "nmae_delta_final_minus_q0": final_nmae - q0_nmae,
        "paired_win_tie_loss": paired_win_tie_loss(final, q0),
        "cvar25": cvar25_summary(final, q0),
    }


def _checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"terminal checkpoint does not exist: {source}")
    value = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), "terminal checkpoint is not a mapping")
    _require(int(value.get("schema_version", -1)) == 1, "checkpoint schema differs")
    _require(value.get("system") == "a11_scort_full", "checkpoint system differs")
    _require(
        value.get("architecture") == A11_ARCHITECTURE,
        "checkpoint architecture differs",
    )
    _require(
        value.get("protocol") == TRAINING_PROTOCOL
        and value.get("scientific_protocol") == SCIENTIFIC_PROTOCOL
        and value.get("manifest_protocol") == MANIFEST_PROTOCOL,
        "checkpoint training/scientific/manifest protocol differs",
    )
    _require(
        value.get("deterministic_algorithms_enabled") is True,
        "checkpoint lacks deterministic-algorithm evidence",
    )

    construction = value.get("construction")
    _require(isinstance(construction, Mapping), "checkpoint construction is missing")
    _require(
        set(construction) == set(CHECKPOINT_CONSTRUCTION_FIELDS),
        "checkpoint construction fields differ",
    )
    _require(
        construction["imagenet_pretrained"] is True,
        "checkpoint must record the fixed ImageNet initialization",
    )
    integer_fields = CHECKPOINT_CONSTRUCTION_FIELDS[1:]
    _require(
        all(
            isinstance(construction[field], int)
            and not isinstance(construction[field], bool)
            and int(construction[field]) >= 1
            for field in integer_fields
        ),
        "checkpoint construction dimensions are invalid",
    )
    _require(
        int(construction["progress_bins"]) == PROGRESS_BINS,
        "checkpoint progress-bin construction differs",
    )
    _require(
        int(construction["token_dim"]) % int(construction["attention_heads"]) == 0,
        "checkpoint token width and attention heads are incompatible",
    )

    training = value.get("training")
    _require(isinstance(training, Mapping), "checkpoint training metadata is missing")
    _require(
        int(training.get("epochs", -1)) == TERMINAL_EPOCHS
        and int(training.get("optimizer_steps_each", -1))
        == OPTIMIZER_STEPS_EACH,
        "checkpoint is not the complete five-epoch terminal artifact",
    )
    expected_training = {
        "seed": DEFAULT_SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": TRAIN_BATCH_SIZE,
        "samples_per_epoch": CORE_TRAIN_SAMPLES,
        "scenes": CORE_TRAIN_SCENE_COUNT,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_data_batches": OPTIMIZER_STEPS_EACH,
        "two_optimizer_step_applications": 2 * OPTIMIZER_STEPS_EACH,
        "optimizer": "two_disjoint_AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "two_disjoint_CosineAnnealingLR_Tmax5",
        "condition_mix": dict(FIXED_TRAIN_CONDITION_MIX),
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "validation_manifest": None,
    }
    differing_training = [
        field
        for field, expected in expected_training.items()
        if training.get(field) != expected
    ]
    _require(
        not differing_training,
        f"checkpoint fixed training metadata differs: {differing_training}",
    )
    optimizer_partition = value.get("optimizer_partition_evidence")
    _require(
        isinstance(optimizer_partition, Mapping)
        and all(
            optimizer_partition.get(field) is True
            for field in (
                "distinct_optimizer_objects",
                "optimizer_parameter_sets_disjoint",
                "raw_optimizer_exact_parameter_set",
                "correction_optimizer_exact_parameter_set",
                "all_trainable_parameters_covered",
            )
        ),
        "checkpoint two-optimizer partition evidence differs",
    )
    scheduler_partition = value.get("scheduler_partition_evidence")
    _require(
        isinstance(scheduler_partition, Mapping)
        and all(
            scheduler_partition.get(field) is True
            for field in (
                "distinct_scheduler_objects",
                "raw_scheduler_owns_raw_optimizer",
                "correction_scheduler_owns_correction_optimizer",
            )
        ),
        "checkpoint two-scheduler partition evidence differs",
    )
    _require_two_optimizer_state_evidence(
        value.get("terminal_optimizer_and_scaler_state"),
        label="checkpoint terminal",
    )
    history = value.get("history")
    _require(
        isinstance(history, Sequence)
        and not isinstance(history, (str, bytes))
        and len(history) == TERMINAL_EPOCHS,
        "checkpoint history is not five complete epochs",
    )
    for expected_epoch, record in enumerate(history, 1):
        _require(
            isinstance(record, Mapping)
            and int(record.get("epoch", -1)) == expected_epoch,
            "checkpoint history epoch order differs",
        )
        metrics = record.get("metrics")
        _require(
            isinstance(metrics, Mapping)
            and int(metrics.get("steps", -1)) == STEPS_PER_EPOCH
            and int(metrics.get("samples", -1)) == CORE_TRAIN_SAMPLES
            and int(metrics.get("raw_optimizer_steps", -1)) == STEPS_PER_EPOCH
            and int(metrics.get("correction_optimizer_steps", -1))
            == STEPS_PER_EPOCH,
            "checkpoint history coverage/optimizer steps differ",
        )
        finite_every_step = metrics.get(
            "optimizer_and_scaler_state_finite_every_step"
        )
        _require(
            isinstance(finite_every_step, Mapping)
            and set(finite_every_step) == {"raw", "correction"}
            and all(finite_every_step.get(arm) is True for arm in finite_every_step),
            "checkpoint history optimizer/scaler every-step evidence differs",
        )
        _require_two_optimizer_state_evidence(
            metrics.get("terminal_optimizer_and_scaler_state"),
            label=f"checkpoint history epoch {expected_epoch} terminal",
        )

    access_flags = value.get("access_flags")
    _require(isinstance(access_flags, Mapping), "checkpoint access flags are missing")
    _require(
        set(access_flags) == set(EXPECTED_CHECKPOINT_ACCESS_FLAGS)
        and all(
            access_flags.get(field) is expected
            for field, expected in EXPECTED_CHECKPOINT_ACCESS_FLAGS.items()
        ),
        "checkpoint records audit or forbidden-partition content access",
    )
    _require(
        value.get("loss_design") == LOSS_DESIGN_METADATA
        and value.get("loss_weights") == LOSS_WEIGHTS.as_dict(),
        "checkpoint fixed loss design or weights differ",
    )
    _require(
        isinstance(value.get("parameter_counts"), Mapping),
        "checkpoint parameter counts are missing",
    )
    model_state = value.get("model_state")
    _require(
        isinstance(model_state, Mapping) and bool(model_state),
        "checkpoint model state is missing",
    )
    _require(
        all(isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in model_state.items()),
        "checkpoint model state entries are malformed",
    )
    _require(
        all(
            not (tensor.is_floating_point() or tensor.is_complex())
            or bool(torch.isfinite(tensor).all())
            for tensor in model_state.values()
        ),
        "checkpoint model state contains a non-finite tensor",
    )
    return value


def load_terminal_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[A11SCORTImageModel, dict[str, Any]]:
    """Strict-load one fresh A11 model without opening any data manifest."""

    checkpoint = _checkpoint_mapping(checkpoint_path)
    construction = checkpoint["construction"]
    try:
        model = A11SCORTImageModel(
            # Loading a complete state never needs a network-backed download.
            imagenet_pretrained=False,
            relation_channels=int(construction["relation_channels"]),
            token_dim=int(construction["token_dim"]),
            attention_heads=int(construction["attention_heads"]),
            decoder_layers=int(construction["decoder_layers"]),
            memory_grid_size=int(construction["memory_grid_size"]),
            progress_bins=int(construction["progress_bins"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise A11CoreAuditEvaluationError(
            "terminal checkpoint does not strict-load into fresh A11 SCORT"
        ) from exc
    model.eval()
    actual_counts = image_model_parameter_counts(model)
    _require(
        dict(checkpoint["parameter_counts"]) == actual_counts,
        "fresh A11 parameter counts differ from the terminal checkpoint",
    )
    _require(
        all(
            not (tensor.is_floating_point() or tensor.is_complex())
            or bool(torch.isfinite(tensor).all())
            for tensor in model.state_dict().values()
        ),
        "fresh A11 model state is non-finite",
    )
    metadata = {
        "terminal_checkpoint": str(Path(checkpoint_path).resolve()),
        "system": checkpoint["system"],
        "architecture": checkpoint["architecture"],
        "construction": dict(construction),
        "parameter_counts": actual_counts,
        "training": {
            "epochs": int(checkpoint["training"]["epochs"]),
            "optimizer_steps_each": int(
                checkpoint["training"]["optimizer_steps_each"]
            ),
            "terminal_history_epoch": int(checkpoint["history"][-1]["epoch"]),
        },
        "pre_evaluation_access_flags": {
            field: bool(checkpoint["access_flags"][field])
            for field in EXPECTED_CHECKPOINT_ACCESS_FLAGS
        },
        "terminal_optimizer_and_scaler_state_finite": {
            arm: bool(checkpoint["terminal_optimizer_and_scaler_state"][arm]["finite"])
            for arm in ("raw", "correction")
        },
        "fresh_strict_load": True,
        "imagenet_download_during_fresh_load": False,
    }
    return model, metadata


def build_a11_audit_evaluation_dataset(
    samples: Sequence[Any],
    *,
    seed: int,
    total_epochs: int,
    condition: str,
) -> SyncGSupportGeometryMultiViewDataset:
    """Build one A10-compatible fixed condition from the audit roster only."""

    values = validate_a11_audit_samples(samples)
    _require(
        str(condition) in EVALUATION_CONDITIONS,
        f"unknown A11 Core-audit condition: {condition}",
    )
    _require(
        int(total_epochs) == EVALUATION_PIXEL_TOTAL_EPOCHS,
        "A11 Core-audit pixel total-epoch count differs",
    )
    return SyncGSupportGeometryMultiViewDataset(
        values,
        training=False,
        seed=int(seed),
        total_epochs=int(total_epochs),
        condition=str(condition),
    )


def verify_clean_epoch4_exact_replay(
    dataset: Any,
    *,
    expected_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Measure exact repeatability of every clean model input at epoch 4."""

    expected = tuple(str(value) for value in expected_sample_ids)
    _require(bool(expected), "clean exact replay roster is empty")
    _require(len(dataset) == len(expected), "clean exact replay roster length differs")
    exact_rows = 0
    clean_condition_rows = 0
    for index, expected_id in enumerate(expected):
        first = dataset[index]
        second = dataset[index]
        _require(
            isinstance(first, Mapping) and isinstance(second, Mapping),
            "clean replay dataset item is not a mapping",
        )
        first_id = first.get("sample_id")
        second_id = second.get("sample_id")
        _require(
            first_id == second_id == expected_id,
            "clean replay sample roster or order differs",
        )
        if first.get("condition_name") == second.get("condition_name") == "clean":
            clean_condition_rows += 1
        tensor_exact = True
        for field in MODEL_INPUT_TENSOR_FIELDS:
            left = first.get(field)
            right = second.get(field)
            tensor_exact = (
                tensor_exact
                and isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and bool(torch.equal(left, right))
            )
        if tensor_exact and first.get("condition_name") == second.get(
            "condition_name"
        ):
            exact_rows += 1
    recorded_epoch = getattr(dataset, "epoch", None)
    epoch_is_four = recorded_epoch is not None and int(recorded_epoch) == 4
    return {
        "samples": len(expected),
        "fixed_transform_epoch": EVALUATION_PIXEL_EPOCH,
        "dataset_recorded_epoch": (
            int(recorded_epoch) if recorded_epoch is not None else None
        ),
        "epoch_is_exactly_four": epoch_is_four,
        "condition_name_clean_rows": clean_condition_rows,
        "repeat_model_input_tensor_exact_rows": exact_rows,
        "model_input_tensor_fields": list(MODEL_INPUT_TENSOR_FIELDS),
        "all_exact": (
            epoch_is_four
            and clean_condition_rows == len(expected)
            and exact_rows == len(expected)
        ),
    }


def posterior_mass_and_cdf_diagnostics(
    final_posterior: torch.Tensor,
    q0_posterior: torch.Tensor,
    layer_posteriors: torch.Tensor,
) -> dict[str, torch.Tensor | int | float]:
    """Count unit-mass distribution and CDF-monotonicity violations per row."""

    _require(
        final_posterior.ndim == q0_posterior.ndim == 2
        and final_posterior.shape == q0_posterior.shape
        and final_posterior.shape[1] == PROGRESS_BINS,
        "final/q0 posterior shapes differ",
    )
    batch = final_posterior.shape[0]
    _require(
        layer_posteriors.shape == (batch, CORRECTION_LAYERS, PROGRESS_BINS),
        "layer posterior shape differs",
    )
    stack = torch.cat(
        (
            final_posterior.float()[:, None],
            q0_posterior.float()[:, None],
            layer_posteriors.float(),
        ),
        dim=1,
    )
    _require(stack.is_floating_point(), "posterior evidence must be floating")
    finite = torch.isfinite(stack).all(dim=2)
    nonnegative = (stack >= 0.0).all(dim=2)
    unit_mass = (
        torch.abs(stack.sum(dim=2) - 1.0)
        <= POSTERIOR_MASS_ABSOLUTE_TOLERANCE
    )
    mass_violation_count = (~(finite & nonnegative & unit_mass)).sum(dim=1)
    cdf = torch.cumsum(stack, dim=2)
    cdf_violation_count = (
        cdf[:, :, 1:]
        < cdf[:, :, :-1] - CDF_MONOTONIC_ABSOLUTE_TOLERANCE
    ).sum(dim=(1, 2)) + (~finite).sum(dim=1)
    return {
        "posterior_distributions_checked_per_sample": CORRECTION_LAYERS + 2,
        "mass_absolute_tolerance": POSTERIOR_MASS_ABSOLUTE_TOLERANCE,
        "cdf_monotonic_absolute_tolerance": CDF_MONOTONIC_ABSOLUTE_TOLERANCE,
        "mass_violation_count_per_sample": mass_violation_count,
        "cdf_monotonic_violation_count_per_sample": cdf_violation_count,
    }


def _tensor_output(
    output: Mapping[str, Any],
    field: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    value = output.get(field)
    _require(isinstance(value, torch.Tensor), f"A11 output tensor is missing: {field}")
    if shape is not None:
        _require(value.shape == shape, f"A11 output shape differs: {field}")
    return value


def _all_exact_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _require(left.shape == right.shape, "exact-comparison tensor shapes differ")
    if left.ndim == 1:
        return left == right
    return (left == right).reshape(left.shape[0], -1).all(dim=1)


def _angle_summary(
    angles: torch.Tensor,
    layer_angle_residuals: torch.Tensor,
) -> list[dict[str, Any]]:
    _require(
        angles.ndim == 2
        and layer_angle_residuals.shape
        == (angles.shape[0], CORRECTION_LAYERS),
        "A11 angle summary shapes differ",
    )
    _require(
        bool(torch.isfinite(angles).all())
        and bool(torch.isfinite(layer_angle_residuals).all()),
        "A11 angle evidence is non-finite",
    )
    values: list[dict[str, Any]] = []
    for row_angles, row_residuals in zip(
        angles.float().cpu(), layer_angle_residuals.float().cpu(), strict=True
    ):
        values.append(
            {
                "edge_count": int(row_angles.numel()),
                "mean_angle": float(row_angles.mean()),
                "mean_absolute_angle": float(row_angles.abs().mean()),
                "maximum_absolute_angle": float(row_angles.abs().max()),
                "root_mean_square_angle": float(
                    torch.sqrt(row_angles.square().mean())
                ),
                "layer_mean_normalized_squared_angle": [
                    float(value) for value in row_residuals.tolist()
                ],
            }
        )
    return values


def evaluate_same_forward_loader(
    model: Any,
    loader: Any,
    *,
    device: torch.device,
    expected_sample_ids: Sequence[str] | None = None,
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Collect final and q0 evidence from one Full forward per batch."""

    model.eval()
    rows: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    sarn_off_counters = {
        "checked_rows": 0,
        "progress_posterior_exact_q0_rows": 0,
        "mean_exact_q0_rows": 0,
        "variance_exact_q0_rows": 0,
        "all_layer_posteriors_exact_q0_rows": 0,
        "all_layer_means_exact_q0_rows": 0,
        "all_layer_variances_exact_q0_rows": 0,
        "correction_inactive_rows": 0,
        "relation_unavailable_rows": 0,
        "all_contract_exact_rows": 0,
    }
    with torch.inference_mode():
        for raw_batch in loader:
            _require(isinstance(raw_batch, Mapping), "audit batch is not a mapping")
            original = raw_batch["original_view"].to(device)
            sarn = raw_batch["sarn_view"].to(device)
            support = raw_batch["sarn_support_mask"].to(device)
            sarn_active = raw_batch["sarn_active"].to(device).bool()
            homography = raw_batch["raw_to_sarn_homography"].to(device)
            target = raw_batch["target"].to(device).float()
            batch = int(target.shape[0])
            _require(
                target.shape == (original.shape[0],) and batch >= 1,
                "audit target batch shape differs",
            )
            batch_ids = raw_batch.get("sample_id")
            _require(
                isinstance(batch_ids, Sequence)
                and not isinstance(batch_ids, (str, bytes))
                and len(batch_ids) == batch
                and all(isinstance(value, str) and value for value in batch_ids),
                "audit sample IDs are missing or batch-misaligned",
            )
            condition_names = raw_batch.get("condition_name")
            _require(
                isinstance(condition_names, Sequence)
                and not isinstance(condition_names, (str, bytes))
                and len(condition_names) == batch,
                "audit condition names are missing or batch-misaligned",
            )
            if expected_condition is not None:
                _require(
                    all(str(value) == expected_condition for value in condition_names),
                    "audit condition pixels differ from the requested arm",
                )

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                full_output = model(
                    original,
                    sarn,
                    support,
                    sarn_active=sarn_active,
                    raw_to_sarn_homography=homography,
                )
                sarn_off_output = model(
                    original,
                    sarn,
                    support,
                    sarn_active=torch.zeros_like(sarn_active),
                    raw_to_sarn_homography=homography,
                )
            _require(
                isinstance(full_output, Mapping)
                and isinstance(sarn_off_output, Mapping),
                "A11 forward output is not a mapping",
            )

            final_posterior = _tensor_output(
                full_output, "progress_posterior", (batch, PROGRESS_BINS)
            ).float()
            q0_posterior = _tensor_output(
                full_output, "raw_anchor_posterior", (batch, PROGRESS_BINS)
            ).float()
            layer_posteriors = _tensor_output(
                full_output,
                "layer_posteriors",
                (batch, CORRECTION_LAYERS, PROGRESS_BINS),
            ).float()
            final_mean = _tensor_output(full_output, "mean", (batch,)).float()
            q0_mean = _tensor_output(
                full_output, "raw_anchor_mean", (batch,)
            ).float()
            layer_means = _tensor_output(
                full_output, "layer_means", (batch, CORRECTION_LAYERS)
            ).float()
            relation_available = _tensor_output(
                full_output, "relation_available", (batch,)
            )
            _require(
                relation_available.dtype == torch.bool,
                "A11 relation availability must be boolean",
            )
            angles = _tensor_output(full_output, "transport_angles").float()
            _require(
                angles.ndim == 2 and angles.shape[0] == batch and angles.shape[1] >= 1,
                "A11 transport-angle shape differs",
            )
            angle_residuals = _tensor_output(
                full_output,
                "layer_angle_residuals",
                (batch, CORRECTION_LAYERS),
            ).float()
            _require(
                bool(torch.isfinite(final_mean).all())
                and bool(torch.isfinite(q0_mean).all())
                and bool(torch.isfinite(layer_means).all())
                and bool(torch.isfinite(target).all()),
                "A11 read evidence is non-finite",
            )
            diagnostics = posterior_mass_and_cdf_diagnostics(
                final_posterior, q0_posterior, layer_posteriors
            )
            angle_summaries = _angle_summary(angles, angle_residuals)
            final_error = torch.abs(final_mean - target)
            q0_error = torch.abs(q0_mean - target)

            off_final = _tensor_output(
                sarn_off_output, "progress_posterior", (batch, PROGRESS_BINS)
            )
            off_q0 = _tensor_output(
                sarn_off_output, "raw_anchor_posterior", (batch, PROGRESS_BINS)
            )
            off_mean = _tensor_output(sarn_off_output, "mean", (batch,))
            off_q0_mean = _tensor_output(
                sarn_off_output, "raw_anchor_mean", (batch,)
            )
            off_variance = _tensor_output(sarn_off_output, "variance", (batch,))
            off_q0_variance = _tensor_output(
                sarn_off_output, "raw_anchor_variance", (batch,)
            )
            off_layers = _tensor_output(
                sarn_off_output,
                "layer_posteriors",
                (batch, CORRECTION_LAYERS, PROGRESS_BINS),
            )
            off_layer_means = _tensor_output(
                sarn_off_output,
                "layer_means",
                (batch, CORRECTION_LAYERS),
            )
            off_layer_variances = _tensor_output(
                sarn_off_output,
                "layer_variances",
                (batch, CORRECTION_LAYERS),
            )
            off_correction = _tensor_output(
                sarn_off_output,
                "correction_active",
                (batch, CORRECTION_LAYERS),
            )
            off_relation = _tensor_output(
                sarn_off_output, "relation_available", (batch,)
            )
            _require(
                off_correction.dtype == torch.bool
                and off_relation.dtype == torch.bool,
                "SARN-off activity evidence must be boolean",
            )
            off_exact_columns = {
                "progress_posterior_exact_q0_rows": _all_exact_rows(
                    off_final, off_q0
                ),
                "mean_exact_q0_rows": _all_exact_rows(off_mean, off_q0_mean),
                "variance_exact_q0_rows": _all_exact_rows(
                    off_variance, off_q0_variance
                ),
                "all_layer_posteriors_exact_q0_rows": _all_exact_rows(
                    off_layers,
                    off_q0[:, None].expand(-1, CORRECTION_LAYERS, -1),
                ),
                "all_layer_means_exact_q0_rows": _all_exact_rows(
                    off_layer_means,
                    off_q0_mean[:, None].expand(-1, CORRECTION_LAYERS),
                ),
                "all_layer_variances_exact_q0_rows": _all_exact_rows(
                    off_layer_variances,
                    off_q0_variance[:, None].expand(-1, CORRECTION_LAYERS),
                ),
                "correction_inactive_rows": ~off_correction.any(dim=1),
                "relation_unavailable_rows": ~off_relation,
            }
            off_all_exact = torch.stack(tuple(off_exact_columns.values()), dim=1).all(
                dim=1
            )
            sarn_off_counters["checked_rows"] += batch
            for name, values in off_exact_columns.items():
                sarn_off_counters[name] += int(values.sum())
            sarn_off_counters["all_contract_exact_rows"] += int(off_all_exact.sum())

            mass_counts = diagnostics["mass_violation_count_per_sample"]
            cdf_counts = diagnostics["cdf_monotonic_violation_count_per_sample"]
            _require(
                isinstance(mass_counts, torch.Tensor)
                and isinstance(cdf_counts, torch.Tensor),
                "posterior diagnostic rows are missing",
            )
            for index in range(batch):
                rows.append(
                    {
                        "sample_id": str(batch_ids[index]),
                        "condition": str(condition_names[index]),
                        "normalized_target": float(target[index]),
                        "final_mean": float(final_mean[index]),
                        "q0_mean": float(q0_mean[index]),
                        "final_absolute_error": float(final_error[index]),
                        "q0_absolute_error": float(q0_error[index]),
                        "relation_available": bool(relation_available[index]),
                        "sarn_active": bool(sarn_active[index]),
                        "layer_means": [
                            float(value)
                            for value in layer_means[index].cpu().tolist()
                        ],
                        "angle_summary": angle_summaries[index],
                        "posterior_mass_violation_count": int(mass_counts[index]),
                        "cdf_monotonic_violation_count": int(cdf_counts[index]),
                        "sarn_off_exact_q0": bool(off_all_exact[index]),
                    }
                )

    _require(bool(rows), "A11 Core-audit evaluation produced no samples")
    sample_ids = tuple(str(row["sample_id"]) for row in rows)
    _require(len(set(sample_ids)) == len(sample_ids), "audit sample IDs are duplicated")
    if expected_sample_ids is not None:
        expected = tuple(str(value) for value in expected_sample_ids)
        _require(sample_ids == expected, "audit sample roster or order differs")
    final_errors = tuple(float(row["final_absolute_error"]) for row in rows)
    q0_errors = tuple(float(row["q0_absolute_error"]) for row in rows)
    summary = summarize_error_vectors(final_errors, q0_errors)
    checked_rows = int(sarn_off_counters["checked_rows"])
    sarn_off_counters["all_exact"] = (
        int(sarn_off_counters["all_contract_exact_rows"]) == checked_rows
    )
    return {
        **summary,
        "relation_available_rows": sum(bool(row["relation_available"]) for row in rows),
        "relation_available_rate": sum(
            bool(row["relation_available"]) for row in rows
        )
        / len(rows),
        "posterior_distributions_checked_per_sample": CORRECTION_LAYERS + 2,
        "posterior_mass_violation_total": sum(
            int(row["posterior_mass_violation_count"]) for row in rows
        ),
        "cdf_monotonic_violation_total": sum(
            int(row["cdf_monotonic_violation_count"]) for row in rows
        ),
        "sarn_off_exact_q0": sarn_off_counters,
        "same_forward_final_and_q0": True,
        "per_sample": rows,
    }


def pool_projective_results(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool three projective arms after exact roster/target comparisons."""

    for name in PROJECTIVE_CONDITIONS:
        _require(name in conditions, f"missing projective condition: {name}")
    rows_by_condition = [conditions[name].get("per_sample") for name in PROJECTIVE_CONDITIONS]
    _require(
        all(isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
            for rows in rows_by_condition),
        "projective per-sample evidence is missing",
    )
    typed_rows = [list(rows) for rows in rows_by_condition]
    reference_ids = tuple(str(row["sample_id"]) for row in typed_rows[0])
    reference_targets = tuple(float(row["normalized_target"]) for row in typed_rows[0])
    for rows in typed_rows[1:]:
        _require(
            tuple(str(row["sample_id"]) for row in rows) == reference_ids,
            "projective condition sample rosters or order differ",
        )
        _require(
            tuple(float(row["normalized_target"]) for row in rows)
            == reference_targets,
            "projective condition normalized targets differ",
        )
    pooled_rows = [row for rows in typed_rows for row in rows]
    final_errors = [float(row["final_absolute_error"]) for row in pooled_rows]
    q0_errors = [float(row["q0_absolute_error"]) for row in pooled_rows]
    summary = summarize_error_vectors(final_errors, q0_errors)
    return {
        **summary,
        "physical_audit_samples": len(reference_ids),
        "conditions": list(PROJECTIVE_CONDITIONS),
        "posterior_mass_violation_total": sum(
            int(row["posterior_mass_violation_count"]) for row in pooled_rows
        ),
        "cdf_monotonic_violation_total": sum(
            int(row["cdf_monotonic_violation_count"]) for row in pooled_rows
        ),
        "relation_available_rows": sum(
            bool(row["relation_available"]) for row in pooled_rows
        ),
        "sarn_off_exact_q0_rows": sum(
            bool(row["sarn_off_exact_q0"]) for row in pooled_rows
        ),
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


def evaluate_a11_core_audit_once(
    *,
    audit_manifest_path: Path,
    terminal_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the Core-audit replay and create exactly one descriptive JSON."""

    _require(int(workers) >= 0, "Core-audit workers must be non-negative")
    _require(
        1 <= int(batch_size) <= EVALUATION_BATCH_SIZE,
        f"Core-audit batch size must be in [1, {EVALUATION_BATCH_SIZE}]",
    )
    audit_manifest = Path(audit_manifest_path).resolve()
    checkpoint_path = Path(terminal_checkpoint_path).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), f"Core-audit output already exists: {output}")
    _require(
        len({audit_manifest, checkpoint_path, output}) == 3,
        "audit manifest, terminal checkpoint, and output paths must differ",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(DEFAULT_SEED, device)

    model, checkpoint_metadata = load_terminal_checkpoint(
        checkpoint_path, device=device
    )
    audit_samples = load_a11_audit_manifest(audit_manifest)
    expected_sample_ids = tuple(sample.sample_id for sample in audit_samples)
    _require(
        len(set(expected_sample_ids)) == len(expected_sample_ids),
        "physical audit sample IDs are duplicated",
    )
    specs = condition_evaluation_specs(DEFAULT_SEED)
    _require(
        tuple(spec.condition for spec in specs) == EVALUATION_CONDITIONS
        and all(spec.dataset_total_epochs == EVALUATION_PIXEL_TOTAL_EPOCHS for spec in specs)
        and all(spec.transform_epoch == EVALUATION_PIXEL_EPOCH for spec in specs),
        "reused A10 condition replay specification differs from A11 protocol",
    )

    condition_results: dict[str, dict[str, Any]] = {}
    clean_exact: dict[str, Any] | None = None
    for spec in specs:
        dataset = build_a11_audit_evaluation_dataset(
            audit_samples,
            seed=spec.dataset_seed,
            total_epochs=spec.dataset_total_epochs,
            condition=spec.condition,
        )
        dataset.set_epoch(spec.transform_epoch)
        if spec.condition == "clean":
            clean_exact = verify_clean_epoch4_exact_replay(
                dataset, expected_sample_ids=expected_sample_ids
            )
        loader = _loader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            workers=int(workers),
            seed=spec.loader_seed,
            cuda=device.type == "cuda",
        )
        condition_results[spec.condition] = evaluate_same_forward_loader(
            model,
            loader,
            device=device,
            expected_sample_ids=expected_sample_ids,
            expected_condition=spec.condition,
        )
    _require(clean_exact is not None, "clean exact replay evidence is missing")
    projective = pool_projective_results(condition_results)
    all_condition_mass_violations = sum(
        int(metrics["posterior_mass_violation_total"])
        for metrics in condition_results.values()
    )
    all_condition_cdf_violations = sum(
        int(metrics["cdf_monotonic_violation_total"])
        for metrics in condition_results.values()
    )
    sarn_off_exact = all(
        bool(metrics["sarn_off_exact_q0"]["all_exact"])
        for metrics in condition_results.values()
    )
    observations = {
        "audit_pooled_nmae_delta_full_minus_q0": projective[
            "nmae_delta_final_minus_q0"
        ],
        "audit_net_paired_win_minus_loss": projective["paired_win_tie_loss"][
            "net_paired_win_margin"
        ],
        "audit_cvar25_delta_full_minus_q0": projective["cvar25"][
            "cvar25_delta_final_minus_q0"
        ],
        "projective_condition_nmae": {
            name: {
                "final": condition_results[name]["final_nmae"],
                "q0": condition_results[name]["q0_nmae"],
                "delta_final_minus_q0": condition_results[name][
                    "nmae_delta_final_minus_q0"
                ],
            }
            for name in PROJECTIVE_CONDITIONS
        },
        "each_projective_condition_full_nmae_strictly_below_q0": all(
            float(condition_results[name]["nmae_delta_final_minus_q0"]) < 0.0
            for name in PROJECTIVE_CONDITIONS
        ),
        "clean_nmae_delta_full_minus_q0": condition_results["clean"][
            "nmae_delta_final_minus_q0"
        ],
        "clean_epoch4_deterministic_input_replay_exact": bool(
            clean_exact["all_exact"]
        ),
        "sarn_off_exact_q0": sarn_off_exact,
        "posterior_mass_violation_total": all_condition_mass_violations,
        "monotonic_violation_total": all_condition_cdf_violations,
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "scope": {
            "physical_core_audit_only": True,
            "training_partition_content_access": False,
            "fold_a_content_access": False,
            "fold_b_content_access": False,
            "formal_holdout_content_access": False,
            "field_photo_content_access": False,
            "automatic_model_selection_or_advancement": False,
        },
        "comparison": {
            "single_terminal_model": True,
            "same_forward_final_and_q0": True,
            "same_sample_ids_and_condition_pixels": True,
            "sarn_off_control_is_a_separate_forward": True,
        },
        "checkpoint": checkpoint_metadata,
        "data": {
            "physical_audit_manifest": str(audit_manifest),
            "physical_audit_samples": len(audit_samples),
            "physical_audit_scenes": sorted(
                {str(sample.scene_stem) for sample in audit_samples}
            ),
            "conditions": list(EVALUATION_CONDITIONS),
            "pixel_total_epochs": EVALUATION_PIXEL_TOTAL_EPOCHS,
            "transform_epoch": EVALUATION_PIXEL_EPOCH,
            "condition_sample_rosters_exact_manifest_order": True,
            "a10_dataset_and_condition_augmentation_reused": True,
        },
        "clean_exact_replay": clean_exact,
        "conditions": condition_results,
        "projective_pooled": projective,
        "predeclared_reference": dict(CORE_SCREEN_REFERENCE),
        "observations": observations,
    }
    payload = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise A11CoreAuditEvaluationError(
            f"Core-audit output already exists: {output}"
        ) from exc
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one terminal A11 checkpoint on the physical Core-audit "
            "manifest and create one descriptive JSON."
        )
    )
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--terminal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_a11_core_audit_once(
        audit_manifest_path=args.audit_manifest,
        terminal_checkpoint_path=args.terminal_checkpoint,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "projective_pooled": result["projective_pooled"],
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
    "A11CoreAuditEvaluationError",
    "EVALUATION_BATCH_SIZE",
    "PROTOCOL",
    "build_a11_audit_evaluation_dataset",
    "build_argument_parser",
    "cvar25_summary",
    "evaluate_a11_core_audit_once",
    "evaluate_same_forward_loader",
    "load_terminal_checkpoint",
    "paired_win_tie_loss",
    "pool_projective_results",
    "posterior_mass_and_cdf_diagnostics",
    "summarize_error_vectors",
    "verify_clean_epoch4_exact_replay",
]
