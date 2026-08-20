"""Descriptive A13 frozen-q0 correction-development replay.

The CLI accepts exactly one physical correction-dev manifest, the terminal A11
q0 checkpoint, the terminal A13 correction checkpoint, and a create-once JSON
output.  Both checkpoints are fully validated and fresh strict-loaded *before*
the development manifest is opened.  There is no input for correction-train,
the previous Core audit, Fold-B, formal data, or field photographs.

The terminal A11 Raw anchor already saw all twelve development scenes.  The
result therefore describes cross-scene generalization of the newly trained
correction conditional on a seen q0; it is not an end-to-end system holdout.
References are serialized beside observations and never act as an automatic
gate or model-selection rule.
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

from experiments.a10_pccot_protocol import condition_evaluation_specs
from experiments.a11_scort_protocol import CORRECTION_LAYERS, PROGRESS_BINS
from experiments.a11_scort_targets import LOSS_DESIGN_METADATA, LOSS_WEIGHTS
from experiments.a13_correction_dev_protocol import (
    CORRECTION_DEV_SAMPLES,
    CORRECTION_DEV_SCENE_COUNT,
    CORRECTION_TRAIN_SAMPLES,
    CORRECTION_TRAIN_SCENES,
    CORRECTION_TRAIN_SCENE_COUNT,
    DESCRIPTIVE_REFERENCE,
    DEVELOPMENT_INTERPRETATION,
    EVALUATION_CONDITIONS,
    EVALUATION_PIXEL_EPOCH,
    EVALUATION_PIXEL_TOTAL_EPOCHS,
    PROJECTIVE_CONDITIONS,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
)
from experiments.evaluate_a11_scort_core_audit import (
    cvar25_summary,
    paired_win_tie_loss,
    posterior_mass_and_cdf_diagnostics,
    summarize_error_vectors,
    verify_clean_epoch4_exact_replay,
)
from experiments.prepare_a13_correction_scene_split import (
    PROTOCOL as DATA_PROTOCOL,
    load_a13_correction_dev_manifest,
    validate_a13_correction_dev_samples,
)
from experiments.run_a12_fixed_q0_causal_probe import (
    A12SCORTCorrection,
    TERMINAL_A11_ACCESS_SCHEMA,
    forward_correction,
    load_frozen_terminal_a11,
    scort_parameter_counts,
)
from experiments.train_a13_scort_s2_correction_only import (
    BATCH_SIZE as TRAIN_BATCH_SIZE,
    CONDITION_PHASE_STRIDE,
    CONDITION_RATIO,
    CONDITION_ROSTER,
    DEFAULT_SEED,
    LEARNING_RATE,
    PROTOCOL as TRAINING_PROTOCOL,
    SYSTEM,
    TERMINAL_EPOCHS,
    WEIGHT_DECAY,
    expected_condition_counts,
    frozen_raw_forward,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _loader,
)


PROTOCOL: Final[str] = "a13_correction_dev_seen_q0_same_forward_epoch4_v1"
EVALUATION_BATCH_SIZE: Final[int] = 32
PAIRED_TIE_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12
EXPECTED_ACCESS_FLAGS: Final[dict[str, bool]] = {
    "correction_train_manifest_access": True,
    "correction_dev_manifest_access": False,
    "terminal_a11_checkpoint_access": True,
    "core_audit_manifest_access": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}
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
        "scientific_protocol",
        "data_protocol",
        "system",
        "source_terminal_a11",
        "construction",
        "parameter_counts",
        "optimizer_partition_evidence",
        "terminal_optimizer_and_scaler_state",
        "training",
        "physical_correction_train",
        "loss_design",
        "loss_weights",
        "history",
        "access_flags",
        "anchor_unchanged",
        "fresh_strict_load",
        "terminal_sarn_off_exact_q0",
        "deterministic_algorithms_enabled",
        "correction_state",
    }
)


class A13CorrectionDevEvaluationError(ValueError):
    """An A13 artifact, development roster, output, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A13CorrectionDevEvaluationError(message)


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


def _checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"terminal A13 correction checkpoint does not exist: {source}")
    value = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), "terminal A13 correction checkpoint is not a mapping")
    return value


def _strict_training_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    training = checkpoint.get("training")
    _require(isinstance(training, Mapping), "A13 training metadata is missing")
    steps_per_epoch = math.ceil(CORRECTION_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
    expected_total_steps = steps_per_epoch * TERMINAL_EPOCHS
    expected_counts_each = [
        expected_condition_counts(CORRECTION_TRAIN_SAMPLES, epoch)
        for epoch in range(TERMINAL_EPOCHS)
    ]
    expected_total_counts = {
        name: sum(counts[name] for counts in expected_counts_each)
        for name in PROJECTIVE_CONDITIONS
    }
    expected_values = {
        "seed": DEFAULT_SEED,
        "epochs": TERMINAL_EPOCHS,
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "batch_size": TRAIN_BATCH_SIZE,
        "samples_per_epoch": CORRECTION_TRAIN_SAMPLES,
        "scenes": CORRECTION_TRAIN_SCENE_COUNT,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": expected_total_steps,
        "optimizer": "AdamW_correction_only",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR_Tmax5",
        "condition_roster_14": list(CONDITION_ROSTER),
        "condition_ratio": dict(CONDITION_RATIO),
        "condition_phase_stride": CONDITION_PHASE_STRIDE,
        "condition_counts_each_epoch": expected_counts_each,
        "condition_counts_terminal_total": expected_total_counts,
        "projective_only": True,
        "clean_presentations": 0,
        "gradient_clipping": None,
        "ema": None,
        "validation_manifest": None,
    }
    differing = [
        name for name, expected in expected_values.items() if training.get(name) != expected
    ]
    _require(not differing, f"A13 fixed training metadata differs: {differing}")
    manifest = training.get("correction_train_manifest")
    _require(
        isinstance(manifest, str) and bool(manifest),
        "A13 correction-train manifest record is missing",
    )
    _require(
        training.get("amp") == "bfloat16_if_supported_else_float16_cuda",
        "A13 precision metadata differs",
    )
    return {
        "epochs": TERMINAL_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": expected_total_steps,
        "condition_counts_each_epoch": expected_counts_each,
        "condition_counts_terminal_total": expected_total_counts,
        "projective_only": True,
        "clean_presentations": 0,
        "validation_manifest": None,
    }


def _strict_history(checkpoint: Mapping[str, Any]) -> None:
    history = checkpoint.get("history")
    _require(
        isinstance(history, Sequence)
        and not isinstance(history, (str, bytes))
        and len(history) == TERMINAL_EPOCHS,
        "A13 history is not five complete epochs",
    )
    steps_per_epoch = math.ceil(CORRECTION_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
    for epoch_index, item in enumerate(history):
        _require(isinstance(item, Mapping), "A13 history row is malformed")
        _require(int(item.get("epoch", -1)) == epoch_index + 1, "A13 history epoch order differs")
        metrics = item.get("metrics")
        _require(isinstance(metrics, Mapping), "A13 history metrics are missing")
        _require(
            int(metrics.get("samples", -1)) == CORRECTION_TRAIN_SAMPLES
            and int(metrics.get("steps", -1)) == steps_per_epoch
            and int(metrics.get("optimizer_steps", -1)) == steps_per_epoch,
            "A13 history row/step coverage differs",
        )
        expected_counts = expected_condition_counts(CORRECTION_TRAIN_SAMPLES, epoch_index)
        _require(
            item.get("expected_condition_counts") == expected_counts
            and metrics.get("condition_counts") == expected_counts,
            "A13 history projective-only condition counts differ",
        )
        _require(
            metrics.get("optimizer_state_finite_every_step") is True
            and metrics.get("correction_state_finite_every_step") is True
            and metrics.get("anchor_contract_every_step") is True
            and metrics.get("anchor_state_bit_exact_at_epoch_end") is True
            and metrics.get("q0_output_identity_every_step") is True
            and metrics.get("inactive_exact_q0_every_step") is True,
            "A13 history finite/frozen/fallback evidence differs",
        )
        _require(
            _finite_optimizer_evidence(metrics.get("terminal_optimizer_and_scaler_state")),
            "A13 history Adam/scaler evidence differs",
        )


def load_terminal_a13_correction_and_anchor(
    correction_checkpoint_path: Path,
    terminal_a11_checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[Any, A12SCORTCorrection, dict[str, Any]]:
    """Validate both terminal artifacts and return fresh strict-loaded modules."""

    correction_path = Path(correction_checkpoint_path).resolve()
    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    _require(correction_path != a11_path, "A11 and A13 checkpoint paths must differ")
    checkpoint = _checkpoint_mapping(correction_path)
    _require(
        set(checkpoint) == EXPECTED_CHECKPOINT_FIELDS,
        "A13 terminal checkpoint fields differ",
    )
    _require(int(checkpoint.get("schema_version", -1)) == 1, "A13 checkpoint schema differs")
    _require(
        checkpoint.get("protocol") == TRAINING_PROTOCOL
        and checkpoint.get("scientific_protocol") == SCIENTIFIC_PROTOCOL
        and checkpoint.get("data_protocol") == DATA_PROTOCOL
        and checkpoint.get("system") == SYSTEM,
        "A13 training/scientific/data protocol or system differs",
    )
    _require(
        checkpoint.get("deterministic_algorithms_enabled") is True,
        "A13 deterministic-algorithm evidence differs",
    )
    validated_access = _exact_bool_mapping(
        checkpoint.get("access_flags"), EXPECTED_ACCESS_FLAGS, label="A13 access flags"
    )
    source = checkpoint.get("source_terminal_a11")
    _require(isinstance(source, Mapping), "A13 source terminal A11 evidence is missing")
    _require(
        Path(str(source.get("source", ""))).resolve() == a11_path,
        "A13 source terminal A11 path differs from the supplied q0 artifact",
    )
    source_access = _exact_bool_mapping(
        source.get("access_flags"),
        TERMINAL_A11_ACCESS_SCHEMA,
        label="A13 nested terminal A11 exact-seven access flags",
    )
    _require(
        source.get("strict_load") is True
        and source.get("eval_mode") is True
        and source.get("all_parameters_requires_grad_false") is True
        and int(source.get("training_epochs", -1)) == 5
        and source.get("validation_manifest") is None
        and source.get("raw_anchor_modules_used") == ["raw_encoder", "raw_posterior_head"]
        and source.get("raw_anchor_state_bit_exact_terminal") is True
        and source.get("raw_anchor_eval_and_frozen_every_step") is True
        and source.get("q0_output_identity_every_step") is True,
        "A13 nested terminal A11 strict/frozen evidence differs",
    )

    construction_value = checkpoint.get("construction")
    _require(isinstance(construction_value, Mapping), "A13 correction construction is missing")
    _require(
        set(construction_value) == set(EXPECTED_CONSTRUCTION_FIELDS),
        "A13 correction construction fields differ",
    )
    construction = {name: int(construction_value[name]) for name in EXPECTED_CONSTRUCTION_FIELDS}
    _require(
        construction["progress_bins"] == PROGRESS_BINS
        and all(value >= 1 for value in construction.values())
        and construction["token_dim"] % construction["attention_heads"] == 0,
        "A13 correction construction dimensions differ",
    )
    training_metadata = _strict_training_metadata(checkpoint)
    _strict_history(checkpoint)

    physical = checkpoint.get("physical_correction_train")
    _require(isinstance(physical, Mapping), "A13 physical correction-train metadata is missing")
    expected_scene_stems = sorted(Path(scene).stem for scene in CORRECTION_TRAIN_SCENES)
    train_ids = physical.get("sample_ids")
    _require(
        int(physical.get("samples", -1)) == CORRECTION_TRAIN_SAMPLES
        and int(physical.get("scenes", -1)) == CORRECTION_TRAIN_SCENE_COUNT
        and physical.get("scene_roster") == expected_scene_stems
        and isinstance(train_ids, Sequence)
        and not isinstance(train_ids, (str, bytes))
        and len(train_ids) == CORRECTION_TRAIN_SAMPLES
        and len(set(str(value) for value in train_ids)) == CORRECTION_TRAIN_SAMPLES,
        "A13 physical correction-train roster/count evidence differs",
    )
    anchor_unchanged = checkpoint.get("anchor_unchanged")
    _require(isinstance(anchor_unchanged, Mapping), "A13 anchor-unchanged evidence is missing")
    _require(
        anchor_unchanged.get("raw_anchor_state_bit_exact_terminal") is True
        and anchor_unchanged.get("raw_anchor_state_bit_exact_each_epoch") == [True] * 5
        and anchor_unchanged.get("anchor_contract_every_step") == [True] * 5
        and anchor_unchanged.get("q0_output_identity_every_step") == [True] * 5,
        "A13 terminal anchor-unchanged evidence differs",
    )
    _require(
        _finite_optimizer_evidence(checkpoint.get("terminal_optimizer_and_scaler_state")),
        "A13 terminal correction Adam/scaler evidence differs",
    )
    _require(
        checkpoint.get("terminal_sarn_off_exact_q0") is True,
        "A13 terminal SARN-off exact-q0 evidence differs",
    )
    strict_evidence = checkpoint.get("fresh_strict_load")
    _require(
        isinstance(strict_evidence, Mapping)
        and strict_evidence.get("strict_load_clean") is True
        and strict_evidence.get("state_bit_exact") is True
        and strict_evidence.get("same_input_reported_outputs_bit_exact") is True
        and strict_evidence.get("fresh_state_finite") is True
        and strict_evidence.get("missing_keys") == []
        and strict_evidence.get("unexpected_keys") == [],
        "A13 recorded fresh strict-load evidence differs",
    )
    expected_loss_design = {
        **dict(LOSS_DESIGN_METADATA),
        "a13_final_read_mask": "read_valid_and_relation_available",
        "a13_all_correction_terms_normalized_on": (
            "read_valid_and_relation_available_rows_only"
        ),
        "inactive_exact_q0_rows_excluded_from_every_correction_denominator": True,
    }
    _require(
        checkpoint.get("loss_design") == expected_loss_design
        and checkpoint.get("loss_weights") == LOSS_WEIGHTS.as_dict(),
        "A13 correction loss design or weights differ",
    )

    anchor, a11_evidence, a11_construction = load_frozen_terminal_a11(
        a11_path, device=device
    )
    _require(a11_construction == construction, "A11 q0 and A13 correction construction differ")
    correction = A12SCORTCorrection(**construction)
    state = _finite_tensor_state(checkpoint.get("correction_state"), label="A13 correction state")
    try:
        incompatibility = correction.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise A13CorrectionDevEvaluationError(
            "A13 correction state does not strict-load into a fresh module"
        ) from exc
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "A13 correction strict load has incompatible keys",
    )
    actual_counts = scort_parameter_counts(correction)
    _require(
        checkpoint.get("parameter_counts") == actual_counts,
        "A13 fresh correction parameter counts differ",
    )
    partition = checkpoint.get("optimizer_partition_evidence")
    _require(
        isinstance(partition, Mapping)
        and partition.get("optimizer_exactly_correction_parameters") is True
        and partition.get("optimizer_excludes_terminal_a11") is True
        and partition.get("all_correction_parameters_trainable") is True
        and partition.get("all_terminal_a11_parameters_frozen") is True
        and int(partition.get("correction_parameter_count", -1))
        == int(actual_counts["total"])
        and int(partition.get("optimizer_parameter_count", -1))
        == int(actual_counts["total"]),
        "A13 correction-only optimizer partition evidence differs",
    )
    correction.eval().to(device)
    _require(
        all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in correction.state_dict().values()
        ),
        "fresh A13 correction state is non-finite",
    )
    metadata = {
        "terminal_a11_checkpoint": str(a11_path),
        "terminal_a13_correction_checkpoint": str(correction_path),
        "training_protocol": TRAINING_PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "system": SYSTEM,
        "access_flags": validated_access,
        "source_terminal_a11_access_flags": source_access,
        "source_terminal_a11_fresh_strict_load": bool(a11_evidence["strict_load"]),
        "training": training_metadata,
        "physical_correction_train_samples": CORRECTION_TRAIN_SAMPLES,
        "physical_correction_train_scenes": CORRECTION_TRAIN_SCENE_COUNT,
        "physical_correction_train_sample_ids": [str(value) for value in train_ids],
        "anchor_unchanged": True,
        "terminal_correction_adam_and_scaler_finite": True,
        "fresh_correction_strict_load": True,
        "parameter_counts": actual_counts,
    }
    return anchor, correction, metadata


def build_a13_dev_evaluation_dataset(
    samples: Sequence[Any],
    *,
    seed: int,
    total_epochs: int,
    condition: str,
) -> SyncGSupportGeometryMultiViewDataset:
    values = validate_a13_correction_dev_samples(samples)
    _require(str(condition) in EVALUATION_CONDITIONS, f"unknown A13 dev condition: {condition}")
    _require(
        int(total_epochs) == EVALUATION_PIXEL_TOTAL_EPOCHS,
        "A13 dev pixel total-epoch count differs",
    )
    return SyncGSupportGeometryMultiViewDataset(
        values,
        training=False,
        seed=int(seed),
        total_epochs=int(total_epochs),
        condition=str(condition),
    )


def _tensor_output(
    output: Mapping[str, Any], field: str, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    value = output.get(field)
    _require(isinstance(value, torch.Tensor), f"A13 output tensor is missing: {field}")
    if shape is not None:
        _require(value.shape == shape, f"A13 output shape differs: {field}")
    return value


def _exact_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _require(left.shape == right.shape, "A13 exact comparison shapes differ")
    if left.ndim == 1:
        return left == right
    return (left == right).reshape(left.shape[0], -1).all(dim=1)


def _q0_contract_exact_rows(output: Mapping[str, Any]) -> torch.Tensor:
    q0 = _tensor_output(output, "raw_anchor_posterior")
    batch = q0.shape[0]
    q0_mean = _tensor_output(output, "raw_anchor_mean", (batch,))
    q0_variance = _tensor_output(output, "raw_anchor_variance", (batch,))
    columns = (
        _exact_rows(_tensor_output(output, "progress_posterior", q0.shape), q0),
        _exact_rows(_tensor_output(output, "mean", (batch,)), q0_mean),
        _exact_rows(_tensor_output(output, "variance", (batch,)), q0_variance),
        _exact_rows(
            _tensor_output(
                output, "layer_posteriors", (batch, CORRECTION_LAYERS, PROGRESS_BINS)
            ),
            q0[:, None].expand(-1, CORRECTION_LAYERS, -1),
        ),
        _exact_rows(
            _tensor_output(output, "layer_means", (batch, CORRECTION_LAYERS)),
            q0_mean[:, None].expand(-1, CORRECTION_LAYERS),
        ),
        _exact_rows(
            _tensor_output(output, "layer_variances", (batch, CORRECTION_LAYERS)),
            q0_variance[:, None].expand(-1, CORRECTION_LAYERS),
        ),
        ~_tensor_output(output, "correction_active", (batch, CORRECTION_LAYERS)).any(dim=1),
        ~_tensor_output(output, "relation_available", (batch,)),
    )
    return torch.stack(columns, dim=1).all(dim=1)


def evaluate_same_forward_loader(
    anchor: Any,
    correction: A12SCORTCorrection,
    loader: Any,
    *,
    device: torch.device,
    expected_sample_ids: Sequence[str] | None = None,
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Save q0/final from one correction forward plus a SARN-off control."""

    anchor.eval()
    correction.eval()
    rows: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode():
        for raw_batch in loader:
            _require(isinstance(raw_batch, Mapping), "A13 dev batch is not a mapping")
            original = raw_batch["original_view"].to(device)
            sarn = raw_batch["sarn_view"].to(device)
            support = raw_batch["sarn_support_mask"].to(device)
            active = raw_batch["sarn_active"].to(device).bool()
            homography = raw_batch["raw_to_sarn_homography"].to(device)
            target = raw_batch["target"].to(device).float()
            batch = int(target.shape[0])
            ids = raw_batch.get("sample_id")
            names = raw_batch.get("condition_name")
            _require(
                isinstance(ids, Sequence)
                and not isinstance(ids, (str, bytes))
                and len(ids) == batch
                and isinstance(names, Sequence)
                and not isinstance(names, (str, bytes))
                and len(names) == batch,
                "A13 dev IDs/conditions are batch-misaligned",
            )
            if expected_condition is not None:
                _require(
                    all(str(name) == expected_condition for name in names),
                    "A13 dev condition pixels differ from requested arm",
                )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                frozen = frozen_raw_forward(anchor, original)
                common = {
                    **frozen,
                    "sarn_view": sarn,
                    "sarn_support_mask": support,
                    "raw_to_sarn_homography": homography,
                }
                output = forward_correction(
                    correction, {**common, "sarn_active": active}
                )
                off_output = forward_correction(
                    correction,
                    {**common, "sarn_active": torch.zeros_like(active)},
                )
            final = _tensor_output(output, "progress_posterior", (batch, PROGRESS_BINS)).float()
            q0 = _tensor_output(output, "raw_anchor_posterior", (batch, PROGRESS_BINS)).float()
            layers = _tensor_output(
                output, "layer_posteriors", (batch, CORRECTION_LAYERS, PROGRESS_BINS)
            ).float()
            final_mean = _tensor_output(output, "mean", (batch,)).float()
            q0_mean = _tensor_output(output, "raw_anchor_mean", (batch,)).float()
            relation = _tensor_output(output, "relation_available", (batch,))
            _require(relation.dtype == torch.bool, "A13 relation availability must be boolean")
            diagnostics = posterior_mass_and_cdf_diagnostics(final, q0, layers)
            mass = diagnostics["mass_violation_count_per_sample"]
            cdf = diagnostics["cdf_monotonic_violation_count_per_sample"]
            _require(isinstance(mass, torch.Tensor) and isinstance(cdf, torch.Tensor), "A13 diagnostics missing")
            final_exact_q0 = (
                _exact_rows(final, q0)
                & _exact_rows(final_mean, q0_mean)
                & _exact_rows(
                    _tensor_output(output, "variance", (batch,)),
                    _tensor_output(output, "raw_anchor_variance", (batch,)),
                )
            )
            off_exact = _q0_contract_exact_rows(off_output)
            for index in range(batch):
                rows.append(
                    {
                        "sample_id": str(ids[index]),
                        "condition": str(names[index]),
                        "normalized_target": float(target[index]),
                        "final_mean": float(final_mean[index]),
                        "q0_mean": float(q0_mean[index]),
                        "final_absolute_error": float(torch.abs(final_mean[index] - target[index])),
                        "q0_absolute_error": float(torch.abs(q0_mean[index] - target[index])),
                        "relation_available": bool(relation[index]),
                        "sarn_active": bool(active[index]),
                        "final_posterior_and_moments_exact_q0": bool(final_exact_q0[index]),
                        "posterior_mass_violation_count": int(mass[index]),
                        "cdf_monotonic_violation_count": int(cdf[index]),
                        "sarn_off_exact_q0": bool(off_exact[index]),
                    }
                )
    _require(bool(rows), "A13 correction-dev evaluation produced no rows")
    ids = tuple(str(row["sample_id"]) for row in rows)
    _require(len(ids) == len(set(ids)), "A13 dev sample IDs are duplicated")
    if expected_sample_ids is not None:
        _require(ids == tuple(str(value) for value in expected_sample_ids), "A13 dev roster/order differs")
    final_errors = tuple(float(row["final_absolute_error"]) for row in rows)
    q0_errors = tuple(float(row["q0_absolute_error"]) for row in rows)
    summary = summarize_error_vectors(final_errors, q0_errors)
    return {
        **summary,
        "relation_available_rows": sum(bool(row["relation_available"]) for row in rows),
        "relation_available_rate": sum(bool(row["relation_available"]) for row in rows) / len(rows),
        "final_posterior_and_moments_exact_q0_rows": sum(
            bool(row["final_posterior_and_moments_exact_q0"]) for row in rows
        ),
        "posterior_distributions_checked_per_sample": CORRECTION_LAYERS + 2,
        "posterior_mass_violation_total": sum(int(row["posterior_mass_violation_count"]) for row in rows),
        "cdf_monotonic_violation_total": sum(int(row["cdf_monotonic_violation_count"]) for row in rows),
        "sarn_off_exact_q0_rows": sum(bool(row["sarn_off_exact_q0"]) for row in rows),
        "sarn_off_exact_q0_all": all(bool(row["sarn_off_exact_q0"]) for row in rows),
        "same_forward_final_and_q0": True,
        "per_sample": rows,
    }


def pool_projective_results(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    typed: list[list[Mapping[str, Any]]] = []
    for name in PROJECTIVE_CONDITIONS:
        _require(name in conditions, f"missing A13 projective condition: {name}")
        rows = conditions[name].get("per_sample")
        _require(
            isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
            "A13 projective per-sample evidence is missing",
        )
        typed.append(list(rows))
    reference_ids = tuple(str(row["sample_id"]) for row in typed[0])
    reference_targets = tuple(float(row["normalized_target"]) for row in typed[0])
    for rows in typed[1:]:
        _require(
            tuple(str(row["sample_id"]) for row in rows) == reference_ids
            and tuple(float(row["normalized_target"]) for row in rows) == reference_targets,
            "A13 projective condition rosters/order/targets differ",
        )
    pooled = [row for rows in typed for row in rows]
    final_errors = [float(row["final_absolute_error"]) for row in pooled]
    q0_errors = [float(row["q0_absolute_error"]) for row in pooled]
    return {
        **summarize_error_vectors(final_errors, q0_errors),
        "physical_correction_dev_samples": len(reference_ids),
        "conditions": list(PROJECTIVE_CONDITIONS),
        "posterior_mass_violation_total": sum(int(row["posterior_mass_violation_count"]) for row in pooled),
        "cdf_monotonic_violation_total": sum(int(row["cdf_monotonic_violation_count"]) for row in pooled),
        "sarn_off_exact_q0_rows": sum(bool(row["sarn_off_exact_q0"]) for row in pooled),
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


def evaluate_a13_correction_dev_once(
    *,
    dev_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    terminal_a13_correction_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Run one create-once correction-development replay without a gate."""

    _require(int(workers) >= 0, "A13 dev workers must be non-negative")
    _require(1 <= int(batch_size) <= EVALUATION_BATCH_SIZE, "A13 dev batch size differs")
    dev_manifest = Path(dev_manifest_path).resolve()
    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    a13_path = Path(terminal_a13_correction_checkpoint_path).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A13 dev output already exists: {output}")
    _require(
        len({dev_manifest, a11_path, a13_path, output}) == 4,
        "A13 dev manifest/checkpoints/output paths must all differ",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(DEFAULT_SEED, device)

    # Artifact validation intentionally precedes any development-manifest read.
    anchor, correction, checkpoint_metadata = load_terminal_a13_correction_and_anchor(
        a13_path, a11_path, device=device
    )
    dev_samples = load_a13_correction_dev_manifest(dev_manifest)
    expected_ids = tuple(sample.sample_id for sample in dev_samples)
    train_ids = set(checkpoint_metadata.pop("physical_correction_train_sample_ids"))
    _require(not (train_ids & set(expected_ids)), "A13 correction train/dev sample IDs overlap")
    _require(
        not (
            set(Path(scene).stem for scene in CORRECTION_TRAIN_SCENES)
            & {str(sample.scene_stem) for sample in dev_samples}
        ),
        "A13 correction train/dev scenes overlap",
    )

    specs = condition_evaluation_specs(DEFAULT_SEED)
    _require(
        tuple(spec.condition for spec in specs) == EVALUATION_CONDITIONS
        and all(spec.dataset_total_epochs == EVALUATION_PIXEL_TOTAL_EPOCHS for spec in specs)
        and all(spec.transform_epoch == EVALUATION_PIXEL_EPOCH for spec in specs),
        "A13 fixed condition/epoch specification differs",
    )
    conditions: dict[str, dict[str, Any]] = {}
    clean_replay: dict[str, Any] | None = None
    for spec in specs:
        dataset = build_a13_dev_evaluation_dataset(
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
        conditions[spec.condition] = evaluate_same_forward_loader(
            anchor,
            correction,
            loader,
            device=device,
            expected_sample_ids=expected_ids,
            expected_condition=spec.condition,
        )
    _require(clean_replay is not None, "A13 clean exact-replay evidence is missing")
    projective = pool_projective_results(conditions)
    mass_total = sum(int(value["posterior_mass_violation_total"]) for value in conditions.values())
    cdf_total = sum(int(value["cdf_monotonic_violation_total"]) for value in conditions.values())
    clean = conditions["clean"]
    observations = {
        "projective_pooled_nmae_delta_final_minus_q0": projective["nmae_delta_final_minus_q0"],
        "projective_net_paired_win_minus_loss": projective["paired_win_tie_loss"]["net_paired_win_margin"],
        "projective_cvar25_delta_final_minus_q0": projective["cvar25"]["cvar25_delta_final_minus_q0"],
        "projective_condition_nmae": {
            name: {
                "final": conditions[name]["final_nmae"],
                "q0": conditions[name]["q0_nmae"],
                "delta_final_minus_q0": conditions[name]["nmae_delta_final_minus_q0"],
            }
            for name in PROJECTIVE_CONDITIONS
        },
        "clean_nmae_delta_final_minus_q0": clean["nmae_delta_final_minus_q0"],
        "clean_final_posterior_and_moments_bit_exact_q0": (
            clean["final_posterior_and_moments_exact_q0_rows"] == CORRECTION_DEV_SAMPLES
        ),
        "clean_epoch4_deterministic_input_replay_exact": bool(clean_replay["all_exact"]),
        "sarn_off_exact_q0": all(bool(value["sarn_off_exact_q0_all"]) for value in conditions.values()),
        "posterior_mass_violation_total": mass_total,
        "cdf_monotonic_violation_total": cdf_total,
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
            "fold_b_content_access": False,
            "formal_holdout_content_access": False,
            "field_photo_content_access": False,
            "automatic_model_selection_or_advancement": False,
        },
        "comparison": {
            "single_terminal_a11_q0": True,
            "single_terminal_a13_correction": True,
            "same_forward_final_and_q0": True,
            "same_sample_ids_and_condition_pixels": True,
            "sarn_off_control_is_separate_forward": True,
        },
        "checkpoint": checkpoint_metadata,
        "data": {
            "physical_correction_dev_manifest": str(dev_manifest),
            "physical_correction_dev_samples": len(dev_samples),
            "physical_correction_dev_scenes": sorted({str(sample.scene_stem) for sample in dev_samples}),
            "conditions": list(EVALUATION_CONDITIONS),
            "pixel_total_epochs": EVALUATION_PIXEL_TOTAL_EPOCHS,
            "transform_epoch": EVALUATION_PIXEL_EPOCH,
            "condition_sample_rosters_exact_manifest_order": True,
        },
        "clean_exact_replay": clean_replay,
        "conditions": conditions,
        "projective_pooled": projective,
        "predeclared_reference": dict(DESCRIPTIVE_REFERENCE),
        "observations": observations,
    }
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise A13CorrectionDevEvaluationError(f"A13 dev output already exists: {output}") from exc
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--terminal-a11-checkpoint", type=Path, required=True)
    parser.add_argument("--terminal-a13-correction-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_a13_correction_dev_once(
        dev_manifest_path=args.dev_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        terminal_a13_correction_checkpoint_path=args.terminal_a13_correction_checkpoint,
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
    "A13CorrectionDevEvaluationError",
    "EVALUATION_BATCH_SIZE",
    "PROTOCOL",
    "build_a13_dev_evaluation_dataset",
    "build_argument_parser",
    "evaluate_a13_correction_dev_once",
    "evaluate_same_forward_loader",
    "load_terminal_a13_correction_and_anchor",
    "pool_projective_results",
]
