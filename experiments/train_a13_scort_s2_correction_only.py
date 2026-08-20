"""Five-stagewise epochs of fresh SCORT correction on physical A13 train only.

The terminal A11 Raw posterior model is strict-loaded once, frozen permanently,
and kept in evaluation mode.  A fresh :class:`A12SCORTCorrection` is the only
trainable module.  The correction objective is the unchanged A11 full
layer-regret/CVaR/angle objective, except that the final read mask is intersected
with relation availability so unavailable exact-q0 rows cannot dilute any
correction term.

Only the correction-train manifest and terminal A11 checkpoint are accepted.
There is no development, audit, Fold-B, formal, field, validation-selection, or
intermediate-checkpoint input.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.a11_scort import A11SCORTImageModel
from experiments.a11_scort_targets import (
    LOSS_DESIGN_METADATA,
    LOSS_WEIGHTS,
    a11_correction_loss,
    build_a11_targets,
)
from experiments.a13_correction_dev_protocol import (
    CORRECTION_TRAIN_SCENE_COUNT,
    CORRECTION_TRAIN_SCENES,
    CORRECTION_TRAIN_SAMPLES,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
    PROJECTIVE_CONDITIONS,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    PROTOCOL as DATA_PROTOCOL,
    load_a13_correction_train_manifest,
)
from experiments.resnet18_direct_progress import DirectSample
from experiments.run_a12_fixed_q0_causal_probe import (
    A12SCORTCorrection,
    forward_correction,
    load_frozen_terminal_a11,
    module_states_bit_exact,
    posterior_integrity_evidence,
    scort_parameter_counts,
    semantic_parameter_groups,
)
from experiments.train_a11_scort_syncg import (
    optimizer_and_scaler_state_finite_evidence,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _configure_reproducibility,
    _loader,
)


PROTOCOL: Final[str] = (
    "syncg_a13_scort_s2_stagewise_correction_only_terminal5_v1"
)
SYSTEM: Final[str] = "a13_scort_s2_correction_only"
DEFAULT_SEED: Final[int] = 20_262_020
TERMINAL_EPOCHS: Final[int] = 5
BATCH_SIZE: Final[int] = 8
LEARNING_RATE: Final[float] = 3.0e-4
WEIGHT_DECAY: Final[float] = 1.0e-4
CONDITION_ROSTER: Final[tuple[str, ...]] = (
    *("perspective_moderate",) * 5,
    *("perspective_severe",) * 5,
    *("combined_severe",) * 4,
)
CONDITION_RATIO: Final[dict[str, int]] = {
    "perspective_moderate": 5,
    "perspective_severe": 5,
    "combined_severe": 4,
}
CONDITION_PHASE_STRIDE: Final[int] = 5
RAW_ANCHOR_MODULE_NAMES: Final[tuple[str, ...]] = (
    "raw_encoder",
    "raw_posterior_head",
)
CORRECTION_GROUP_NAMES: Final[tuple[str, ...]] = (
    "sarn_encoder",
    "relation_encoder",
    "progress_decoder",
    "orthogonal_transport",
)


class A13TrainingError(ValueError):
    """An A13 input, frozen-anchor contract, loss, or state is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A13TrainingError(message)


def configure_a13_reproducibility(seed: int, device: torch.device) -> None:
    _require(int(seed) == DEFAULT_SEED, "A13 uses fixed seed 20262020")
    _configure_reproducibility(int(seed), device)
    torch.use_deterministic_algorithms(True, warn_only=False)


def condition_for_position(position: int, epoch: int) -> str:
    """Return one target-blind member of the fixed 5:5:4 projective roster."""

    _require(position >= 0, "A13 condition position must be non-negative")
    _require(0 <= int(epoch) < TERMINAL_EPOCHS, "A13 condition epoch differs")
    phase = (DEFAULT_SEED + int(epoch) * CONDITION_PHASE_STRIDE) % len(
        CONDITION_ROSTER
    )
    return CONDITION_ROSTER[(int(position) + phase) % len(CONDITION_ROSTER)]


def expected_condition_counts(samples: int, epoch: int) -> dict[str, int]:
    _require(int(samples) > 0, "A13 condition count requires samples")
    counts = Counter(condition_for_position(index, epoch) for index in range(samples))
    return {name: int(counts.get(name, 0)) for name in PROJECTIVE_CONDITIONS}


class A13ProjectiveOnlyDataset(Dataset[dict[str, Any]]):
    """Physical views with a target-blind fixed 14-slot projective roster.

    Existing dataset semantics remain untouched.  Three existing fixed-condition
    datasets materialize the same physical sample under moderate, severe, or
    combined projective shift.  This adapter selects among them by source order
    and epoch only; target values and model outputs are not selection inputs.
    """

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        seed: int = DEFAULT_SEED,
        total_epochs: int = TERMINAL_EPOCHS,
    ) -> None:
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self.epoch = 0
        _require(bool(self.samples), "A13 correction-train samples are empty")
        _require(self.seed == DEFAULT_SEED, "A13 dataset seed differs")
        _require(
            self.total_epochs == TERMINAL_EPOCHS,
            "A13 dataset terminal epoch count differs",
        )
        _require(
            tuple(PROJECTIVE_CONDITIONS)
            == (
                "perspective_moderate",
                "perspective_severe",
                "combined_severe",
            ),
            "A13 projective condition names differ",
        )
        self._datasets = {
            (epoch, condition): SyncGSupportGeometryMultiViewDataset(
                self.samples,
                training=False,
                seed=self.seed + epoch * 3_000_049,
                total_epochs=1,
                condition=condition,
            )
            for epoch in range(self.total_epochs)
            for condition in PROJECTIVE_CONDITIONS
        }

    def set_epoch(self, epoch: int) -> None:
        _require(0 <= int(epoch) < self.total_epochs, "A13 dataset epoch differs")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        _require(0 <= position < len(self.samples), "A13 dataset index differs")
        condition = condition_for_position(position, self.epoch)
        item = self._datasets[(self.epoch, condition)][position]
        _require(item["condition_name"] == condition, "A13 condition materialization differs")
        return item


def build_a13_train_dataset(
    samples: Sequence[DirectSample],
) -> A13ProjectiveOnlyDataset:
    values = tuple(samples)
    if CORRECTION_TRAIN_SAMPLES > 0:
        _require(
            len(values) == CORRECTION_TRAIN_SAMPLES,
            "A13 correction-train sample count differs",
        )
    if CORRECTION_TRAIN_SCENES:
        _require(
            {sample.scene_stem for sample in values}
            == {Path(scene).stem for scene in CORRECTION_TRAIN_SCENES},
            "A13 correction-train scene roster differs",
        )
    return A13ProjectiveOnlyDataset(values)


def _device_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _device_value(item, device) for key, item in value.items()}
    return value


def _state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _raw_anchor_state_cpu(model: A11SCORTImageModel) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for module_name in RAW_ANCHOR_MODULE_NAMES:
        module = getattr(model, module_name)
        for name, value in module.state_dict().items():
            result[f"{module_name}.{name}"] = value.detach().cpu().clone()
    return result


def _module_state_finite(module: nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in module.state_dict().values()
    )


def _anchor_contract(model: A11SCORTImageModel) -> dict[str, bool]:
    raw_parameters = tuple(
        parameter
        for name in RAW_ANCHOR_MODULE_NAMES
        for parameter in getattr(model, name).parameters()
    )
    return {
        "eval_mode": not model.training,
        "all_parameters_requires_grad_false": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "raw_gradients_absent": all(
            parameter.grad is None for parameter in raw_parameters
        ),
        "raw_state_finite": all(
            _module_state_finite(getattr(model, name))
            for name in RAW_ANCHOR_MODULE_NAMES
        ),
    }


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = posterior.float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    grid = torch.linspace(
        0.0,
        1.0,
        probability.shape[1],
        dtype=torch.float32,
        device=probability.device,
    )
    mean = (probability * grid[None]).sum(dim=1)
    variance = (probability * (grid[None] - mean[:, None]).square()).sum(dim=1)
    return mean, variance


def frozen_raw_forward(
    model: A11SCORTImageModel,
    original_view: torch.Tensor,
) -> dict[str, Any]:
    """Compute terminal q0 without entering the obsolete A11 correction path."""

    model.eval()
    with torch.no_grad():
        features = model.raw_encoder(original_view)
        logits = model.raw_posterior_head(features["representation"])
        posterior = torch.softmax(logits.float(), dim=1)
        mean, variance = _posterior_moments(posterior)
    return {
        "raw_posterior": posterior,
        "raw_mean": mean,
        "raw_variance": variance,
        "raw_features": {
            "stride8": features["stride8"],
            "stride16": features["stride16"],
        },
    }


def active_only_a11_correction_loss(
    output: Mapping[str, Any],
    raw_batch: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Apply the full A11 objective with every term normalized on active rows."""

    targets = build_a11_targets(raw_batch)
    output_device = output["mean"].device
    targets = {
        name: value.to(output_device) if isinstance(value, torch.Tensor) else value
        for name, value in targets.items()
    }
    relation_active = output["relation_available"].detach().bool()
    targets["read_valid"] = targets["read_valid"] & relation_active
    loss, components = a11_correction_loss(output, targets)
    coverage = components["coverage"]
    active = targets["read_valid"]
    expected_layers = active[:, None].expand_as(output["correction_active"])
    _require(
        torch.equal(coverage["final_read_applied"], active),
        "A13 final read denominator includes an inactive row",
    )
    _require(
        torch.equal(coverage["layer_regret_applied"], expected_layers)
        and torch.equal(coverage["angle_regularizer_applied"], expected_layers),
        "A13 layer risk denominator differs from relation-active rows",
    )
    _require(
        torch.equal(coverage["cvar_eligible"], active),
        "A13 CVaR eligibility differs from relation-active rows",
    )
    return loss, components, targets


def _autocast_settings(device: torch.device) -> tuple[bool, torch.dtype, str]:
    if device.type != "cuda":
        return False, torch.float32, "float32"
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bfloat16"
    return True, torch.float16, "float16"


def _gradient_summary(parameters: Sequence[nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    absolute_sum = (
        sum(float(gradient.detach().abs().double().sum().cpu()) for gradient in gradients)
        if finite
        else 0.0
    )
    return {
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero": absolute_sum > 0.0,
        "absolute_sum": absolute_sum,
    }


def _semantic_groups(
    model: A12SCORTCorrection,
) -> dict[str, tuple[nn.Parameter, ...]]:
    groups = dict(semantic_parameter_groups(model))
    groups["angle_output"] = tuple(
        model.orthogonal_transport.angle_output.parameters()
    )
    return groups


def correction_optimizer_partition_evidence(
    anchor: A11SCORTImageModel,
    correction: A12SCORTCorrection,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    correction_ids = {id(parameter) for parameter in correction.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    anchor_ids = {id(parameter) for parameter in anchor.parameters()}
    return {
        "optimizer_exactly_correction_parameters": optimizer_ids == correction_ids,
        "optimizer_excludes_terminal_a11": optimizer_ids.isdisjoint(anchor_ids),
        "all_correction_parameters_trainable": all(
            parameter.requires_grad for parameter in correction.parameters()
        ),
        "all_terminal_a11_parameters_frozen": all(
            not parameter.requires_grad for parameter in anchor.parameters()
        ),
        "correction_parameter_count": sum(
            parameter.numel() for parameter in correction.parameters()
        ),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
    }


def _exact_q0_rows(
    output: Mapping[str, Any], rows: torch.Tensor
) -> bool:
    selected = rows.bool()
    if not bool(selected.any()):
        return True
    q0 = output["raw_anchor_posterior"][selected]
    q0_mean = output["raw_anchor_mean"][selected]
    q0_variance = output["raw_anchor_variance"][selected]
    return (
        torch.equal(output["progress_posterior"][selected], q0)
        and torch.equal(output["mean"][selected], q0_mean)
        and torch.equal(output["variance"][selected], q0_variance)
        and torch.equal(
            output["layer_posteriors"][selected],
            q0[:, None].expand_as(output["layer_posteriors"][selected]),
        )
        and torch.equal(
            output["layer_means"][selected],
            q0_mean[:, None].expand_as(output["layer_means"][selected]),
        )
        and torch.equal(
            output["layer_variances"][selected],
            q0_variance[:, None].expand_as(output["layer_variances"][selected]),
        )
    )


def _reported_output_finite(output: Mapping[str, Any]) -> bool:
    return all(
        isinstance(output.get(name), torch.Tensor)
        and bool(torch.isfinite(output[name]).all())
        for name in (
            "progress_posterior",
            "mean",
            "variance",
            "raw_anchor_posterior",
            "raw_anchor_mean",
            "raw_anchor_variance",
            "layer_posteriors",
            "layer_means",
            "layer_variances",
            "transport_angles",
            "layer_angle_residuals",
        )
    )


def run_a13_correction_epoch(
    anchor: A11SCORTImageModel,
    correction: A12SCORTCorrection,
    loader: Any,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None = None,
    anchor_state_reference: Mapping[str, torch.Tensor] | None = None,
    max_steps: int | None = None,
    record_step_trace: bool = False,
) -> dict[str, Any]:
    """Run one correction-only epoch while the terminal Raw anchor stays fixed."""

    training = optimizer is not None
    anchor.eval()
    correction.train(training)
    _require(all(_anchor_contract(anchor).values()), "A13 frozen anchor contract differs")
    semantic = _semantic_groups(correction)
    autocast_enabled, autocast_dtype, precision = _autocast_settings(device)
    total_rows = active_rows_total = steps = optimizer_steps = 0
    final_error_sum = q0_error_sum = 0.0
    loss_active_weighted_sum = 0.0
    component_active_weighted_sums = {
        "final_read": 0.0,
        "layer_softplus_regret": 0.0,
        "relative_cvar25": 0.0,
        "normalized_angle": 0.0,
    }
    condition_counts: Counter[str] = Counter()
    gradient_finite_every_step = {name: True for name in semantic}
    gradient_nonzero_seen = {name: False for name in semantic}
    optimizer_state_finite_every_step = True
    model_state_finite_every_step = True
    anchor_contract_every_step = True
    q0_identity_every_step = True
    inactive_exact_q0_every_step = True
    active_rows_seen = False
    initial_active_exact_q0: bool | None = None
    terminal_optimizer_state: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    last_replay_batch: dict[str, Any] | None = None

    for raw_batch in loader:
        if max_steps is not None and steps >= int(max_steps):
            break
        names = raw_batch.get("condition_name", [])
        if isinstance(names, str):
            names = [names]
        _require(
            bool(names)
            and all(str(name) in PROJECTIVE_CONDITIONS for name in names),
            "A13 observed a clean or unknown training condition",
        )
        batch = _device_value(raw_batch, device)
        batch_size = int(batch["target"].shape[0])
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        anchor.eval()
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                frozen = frozen_raw_forward(anchor, batch["original_view"])
                correction_batch = {
                    **frozen,
                    "sarn_view": batch["sarn_view"],
                    "sarn_support_mask": batch["sarn_support_mask"],
                    "sarn_active": batch["sarn_active"],
                    "raw_to_sarn_homography": batch["raw_to_sarn_homography"],
                }
                output = forward_correction(correction, correction_batch)
                loss, components, targets = active_only_a11_correction_loss(
                    output, raw_batch
                )
        _require(bool(torch.isfinite(loss)), "A13 correction loss is non-finite")
        _require(_reported_output_finite(output), "A13 reported output is non-finite")
        active = output["relation_available"].detach().bool()
        inactive = ~active
        active_count = int(active.sum().cpu())
        active_rows_seen = active_rows_seen or active_count > 0
        if initial_active_exact_q0 is None and active_count > 0:
            initial_active_exact_q0 = _exact_q0_rows(output, active)
        q0_identity = (
            torch.equal(output["raw_anchor_posterior"], frozen["raw_posterior"])
            and torch.equal(output["raw_anchor_mean"], frozen["raw_mean"])
            and torch.equal(output["raw_anchor_variance"], frozen["raw_variance"])
        )
        q0_identity_every_step = q0_identity_every_step and q0_identity
        inactive_exact = _exact_q0_rows(output, inactive)
        inactive_exact_q0_every_step = (
            inactive_exact_q0_every_step and inactive_exact
        )
        step_gradient: dict[str, Any] = {}
        parameter_updated: dict[str, bool] = {}
        if training:
            assert optimizer is not None
            before = (
                {
                    name: tuple(parameter.detach().clone() for parameter in parameters)
                    for name, parameters in semantic.items()
                }
                if record_step_trace
                else {}
            )
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            step_gradient = {
                name: _gradient_summary(parameters)
                for name, parameters in semantic.items()
            }
            _require(
                all(summary["finite"] for summary in step_gradient.values()),
                "A13 correction gradient is absent or non-finite",
            )
            for name, summary in step_gradient.items():
                gradient_finite_every_step[name] = (
                    gradient_finite_every_step[name] and bool(summary["finite"])
                )
                gradient_nonzero_seen[name] = (
                    gradient_nonzero_seen[name] or bool(summary["nonzero"])
                )
            _require(
                all(parameter.grad is None for parameter in anchor.parameters()),
                "A13 correction loss reached the terminal A11 anchor",
            )
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_steps += 1
            terminal_optimizer_state = optimizer_and_scaler_state_finite_evidence(
                optimizer, scaler
            )
            _require(
                bool(terminal_optimizer_state["finite"]),
                "A13 AdamW or scaler state is non-finite",
            )
            optimizer_state_finite_every_step = (
                optimizer_state_finite_every_step
                and bool(terminal_optimizer_state["finite"])
            )
            if record_step_trace:
                parameter_updated = {
                    name: any(
                        not torch.equal(previous, current.detach())
                        for previous, current in zip(
                            before[name], parameters, strict=True
                        )
                    )
                    for name, parameters in semantic.items()
                }
        state_finite = _module_state_finite(correction)
        _require(state_finite, "A13 correction state is non-finite")
        model_state_finite_every_step = model_state_finite_every_step and state_finite
        contract = _anchor_contract(anchor)
        _require(all(contract.values()), "A13 frozen anchor changed mode/state/gradient")
        anchor_contract_every_step = anchor_contract_every_step and all(contract.values())
        integrity = posterior_integrity_evidence(output["progress_posterior"])
        _require(
            integrity["negative_probability_row_count"] == 0
            and integrity["mass_violation_row_count"] == 0
            and integrity["cdf_nonmonotone_row_count"] == 0
            and integrity["cdf_terminal_violation_row_count"] == 0,
            "A13 posterior integrity differs",
        )
        read = targets["read"].to(device)
        if active_count > 0:
            final_error_sum += float(
                torch.abs(output["mean"].detach().float() - read)
                .masked_select(active)
                .sum()
                .cpu()
            )
            q0_error_sum += float(
                torch.abs(output["raw_anchor_mean"].detach().float() - read)
                .masked_select(active)
                .sum()
                .cpu()
            )
            loss_active_weighted_sum += float(loss.detach().cpu()) * active_count
            for name in component_active_weighted_sums:
                component_active_weighted_sums[name] += (
                    float(components[name].detach().cpu()) * active_count
                )
        condition_counts.update(str(name) for name in names)
        total_rows += batch_size
        active_rows_total += active_count
        steps += 1
        last_replay_batch = {
            key: value.detach().clone()
            for key, value in correction_batch.items()
            if isinstance(value, torch.Tensor)
        }
        last_replay_batch["raw_features"] = {
            name: value.detach().clone()
            for name, value in correction_batch["raw_features"].items()
        }
        if record_step_trace:
            trace.append(
                {
                    "step": steps,
                    "loss": float(loss.detach().cpu()),
                    "active_rows": active_count,
                    "gradient": step_gradient,
                    "parameter_updated": parameter_updated,
                    "q0_identity": q0_identity,
                    "inactive_exact_q0": inactive_exact,
                    "anchor_contract": contract,
                    "optimizer_state": terminal_optimizer_state,
                }
            )

    _require(total_rows > 0 and steps > 0, "A13 epoch produced no rows")
    _require(active_rows_seen and active_rows_total > 0, "A13 epoch has no active rows")
    anchor_state_exact = (
        True
        if anchor_state_reference is None
        else module_states_bit_exact(
            dict(anchor_state_reference), _raw_anchor_state_cpu(anchor)
        )
    )
    _require(anchor_state_exact, "A13 terminal Raw anchor state changed")
    denominator = float(active_rows_total)
    return {
        "samples": total_rows,
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "active_rows": active_rows_total,
        "inactive_rows": total_rows - active_rows_total,
        "condition_counts": dict(sorted(condition_counts.items())),
        "correction_loss_active_row_weighted": loss_active_weighted_sum / denominator,
        "final_nmae_active": final_error_sum / denominator,
        "q0_nmae_active": q0_error_sum / denominator,
        "nmae_delta_final_minus_q0_active": (
            final_error_sum - q0_error_sum
        ) / denominator,
        "correction_components_active_row_weighted": {
            name: value / denominator
            for name, value in component_active_weighted_sums.items()
        },
        "loss_denominator": "relation_available_and_read_valid_rows_only",
        "risk_denominator": "relation_available_and_read_valid_rows_only",
        "batch_cvar_scope": "top_ceil_25_percent_of_active_rows_in_each_batch",
        "gradient_finite_every_step": gradient_finite_every_step,
        "gradient_nonzero_seen": gradient_nonzero_seen,
        "optimizer_state_finite_every_step": optimizer_state_finite_every_step,
        "correction_state_finite_every_step": model_state_finite_every_step,
        "anchor_contract_every_step": anchor_contract_every_step,
        "anchor_state_bit_exact_at_epoch_end": anchor_state_exact,
        "q0_output_identity_every_step": q0_identity_every_step,
        "inactive_exact_q0_every_step": inactive_exact_q0_every_step,
        "initial_active_exact_q0": bool(initial_active_exact_q0),
        "terminal_optimizer_and_scaler_state": terminal_optimizer_state,
        "step_trace": trace,
        "autocast_precision": precision,
        "gradient_clipping": None,
        "_last_replay_batch": last_replay_batch,
    }


def _fresh_strict_load_evidence(
    correction: A12SCORTCorrection,
    construction: Mapping[str, int],
    replay_batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    correction.eval()
    with torch.no_grad():
        expected = forward_correction(correction, replay_batch)
    terminal_state = _state_cpu(correction)
    fresh = A12SCORTCorrection(**dict(construction)).to(device)
    incompatibility = fresh.load_state_dict(terminal_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        observed = forward_correction(fresh, replay_batch)
    keys = (
        "progress_posterior",
        "mean",
        "variance",
        "raw_anchor_posterior",
        "raw_anchor_mean",
        "raw_anchor_variance",
        "layer_posteriors",
        "layer_means",
        "layer_variances",
        "transport_angles",
    )
    return {
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
        "strict_load_clean": (
            not incompatibility.missing_keys and not incompatibility.unexpected_keys
        ),
        "state_bit_exact": module_states_bit_exact(
            terminal_state, _state_cpu(fresh)
        ),
        "same_input_reported_outputs_bit_exact": all(
            torch.equal(expected[name], observed[name]) for name in keys
        ),
        "fresh_state_finite": _module_state_finite(fresh),
    }


def _load_correction_train_manifest(path: Path) -> tuple[DirectSample, ...]:
    return tuple(load_a13_correction_train_manifest(Path(path).resolve()))


def train_a13_correction_only(
    *,
    correction_train_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
) -> dict[str, Any]:
    """Train only a fresh correction for exactly five terminal epochs."""

    _require(workers >= 0, "A13 workers must be non-negative")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A13 terminal output already exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a13_reproducibility(DEFAULT_SEED, device)
    samples = _load_correction_train_manifest(correction_train_manifest_path)
    dataset = build_a13_train_dataset(samples)
    anchor, source_evidence, construction = load_frozen_terminal_a11(
        terminal_a11_checkpoint_path, device=device
    )
    anchor.eval()
    anchor_initial_state = _raw_anchor_state_cpu(anchor)
    torch.manual_seed(DEFAULT_SEED)
    correction = A12SCORTCorrection(**construction).to(device)
    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    partition = correction_optimizer_partition_evidence(anchor, correction, optimizer)
    _require(
        all(
            bool(partition[name])
            for name in (
                "optimizer_exactly_correction_parameters",
                "optimizer_excludes_terminal_a11",
                "all_correction_parameters_trainable",
                "all_terminal_a11_parameters_frozen",
            )
        ),
        "A13 correction-only optimizer partition differs",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TERMINAL_EPOCHS
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    history: list[dict[str, Any]] = []
    total_steps = 0
    replay_batch: Mapping[str, Any] | None = None
    condition_totals: Counter[str] = Counter()
    for epoch in range(TERMINAL_EPOCHS):
        dataset.set_epoch(epoch)
        loader = _loader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            workers=workers,
            seed=DEFAULT_SEED + epoch,
            cuda=device.type == "cuda",
        )
        metrics = run_a13_correction_epoch(
            anchor,
            correction,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            anchor_state_reference=anchor_initial_state,
        )
        replay_batch = metrics.pop("_last_replay_batch")
        expected_counts = expected_condition_counts(len(samples), epoch)
        _require(
            metrics["samples"] == len(samples)
            and metrics["steps"] == math.ceil(len(samples) / BATCH_SIZE),
            "A13 epoch does not cover correction-train once",
        )
        _require(
            metrics["condition_counts"] == expected_counts
            and "clean" not in metrics["condition_counts"],
            "A13 epoch condition roster differs from fixed 5:5:4 slots",
        )
        _require(
            all(metrics["gradient_finite_every_step"].values())
            and all(metrics["gradient_nonzero_seen"].values()),
            "A13 epoch semantic gradient evidence differs",
        )
        _require(
            metrics["optimizer_state_finite_every_step"]
            and metrics["correction_state_finite_every_step"]
            and metrics["anchor_contract_every_step"]
            and metrics["anchor_state_bit_exact_at_epoch_end"]
            and metrics["q0_output_identity_every_step"]
            and metrics["inactive_exact_q0_every_step"],
            "A13 epoch finite/frozen/fallback evidence differs",
        )
        total_steps += int(metrics["optimizer_steps"])
        condition_totals.update(metrics["condition_counts"])
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "expected_condition_counts": expected_counts,
                "metrics": metrics,
            }
        )
        scheduler.step()
    expected_total_steps = TERMINAL_EPOCHS * math.ceil(len(samples) / BATCH_SIZE)
    _require(total_steps == expected_total_steps, "A13 optimizer step count differs")
    _require(replay_batch is not None, "A13 has no terminal replay batch")
    _require(_module_state_finite(correction), "A13 terminal correction is non-finite")
    anchor_terminal_state = _raw_anchor_state_cpu(anchor)
    anchor_exact = module_states_bit_exact(anchor_initial_state, anchor_terminal_state)
    _require(anchor_exact, "A13 terminal Raw anchor state is not bit-exact")
    terminal_optimizer_state = optimizer_and_scaler_state_finite_evidence(
        optimizer, scaler
    )
    _require(
        bool(terminal_optimizer_state["finite"]),
        "A13 terminal AdamW/scaler state is non-finite",
    )
    fresh = _fresh_strict_load_evidence(
        correction, construction, replay_batch, device=device
    )
    _require(
        fresh["strict_load_clean"]
        and fresh["state_bit_exact"]
        and fresh["same_input_reported_outputs_bit_exact"]
        and fresh["fresh_state_finite"],
        "A13 fresh strict-load replay differs",
    )
    fallback_batch = dict(replay_batch)
    fallback_batch["sarn_active"] = torch.zeros_like(
        replay_batch["sarn_active"], dtype=torch.bool
    )
    correction.eval()
    with torch.no_grad():
        fallback_output = forward_correction(correction, fallback_batch)
    terminal_sarn_off_exact_q0 = (
        not bool(fallback_output["relation_available"].any())
        and _exact_q0_rows(
            fallback_output,
            torch.ones_like(fallback_output["relation_available"], dtype=torch.bool),
        )
    )
    _require(
        terminal_sarn_off_exact_q0,
        "A13 terminal SARN-off fallback is not exact q0",
    )
    scene_roster = tuple(sorted({sample.scene_stem for sample in samples}))
    if CORRECTION_TRAIN_SCENE_COUNT > 0:
        _require(
            len(scene_roster) == CORRECTION_TRAIN_SCENE_COUNT,
            "A13 correction-train scene count differs",
        )
    access_flags = {
        "correction_train_manifest_access": True,
        "correction_dev_manifest_access": False,
        "terminal_a11_checkpoint_access": True,
        "core_audit_manifest_access": False,
        "fold_a_content_access": False,
        "fold_b_content_access": False,
        "formal_holdout_content_access": False,
        "field_photo_content_access": False,
    }
    source_terminal = {
        **source_evidence,
        "source": str(Path(terminal_a11_checkpoint_path).resolve()),
        "raw_anchor_modules_used": list(RAW_ANCHOR_MODULE_NAMES),
        "raw_anchor_state_bit_exact_terminal": anchor_exact,
        "raw_anchor_eval_and_frozen_every_step": all(
            item["metrics"]["anchor_contract_every_step"] for item in history
        ),
        "q0_output_identity_every_step": all(
            item["metrics"]["q0_output_identity_every_step"] for item in history
        ),
    }
    training = {
        "seed": DEFAULT_SEED,
        "epochs": TERMINAL_EPOCHS,
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "batch_size": BATCH_SIZE,
        "samples_per_epoch": len(samples),
        "scenes": len(scene_roster),
        "steps_per_epoch": math.ceil(len(samples) / BATCH_SIZE),
        "optimizer_steps": total_steps,
        "optimizer": "AdamW_correction_only",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR_Tmax5",
        "condition_roster_14": list(CONDITION_ROSTER),
        "condition_ratio": dict(CONDITION_RATIO),
        "condition_phase_stride": CONDITION_PHASE_STRIDE,
        "condition_counts_each_epoch": [
            item["metrics"]["condition_counts"] for item in history
        ],
        "condition_counts_terminal_total": dict(sorted(condition_totals.items())),
        "projective_only": True,
        "clean_presentations": 0,
        "gradient_clipping": None,
        "ema": None,
        "amp": "bfloat16_if_supported_else_float16_cuda",
        "correction_train_manifest": str(
            Path(correction_train_manifest_path).resolve()
        ),
        "validation_manifest": None,
    }
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "system": SYSTEM,
        "source_terminal_a11": source_terminal,
        "construction": dict(construction),
        "parameter_counts": scort_parameter_counts(correction),
        "optimizer_partition_evidence": partition,
        "terminal_optimizer_and_scaler_state": terminal_optimizer_state,
        "training": training,
        "physical_correction_train": {
            "manifest": str(Path(correction_train_manifest_path).resolve()),
            "samples": len(samples),
            "scenes": len(scene_roster),
            "scene_roster": list(scene_roster),
            "sample_ids": [sample.sample_id for sample in samples],
        },
        "loss_design": {
            **dict(LOSS_DESIGN_METADATA),
            "a13_final_read_mask": "read_valid_and_relation_available",
            "a13_all_correction_terms_normalized_on": (
                "read_valid_and_relation_available_rows_only"
            ),
            "inactive_exact_q0_rows_excluded_from_every_correction_denominator": True,
        },
        "loss_weights": LOSS_WEIGHTS.as_dict(),
        "history": history,
        "access_flags": access_flags,
        "anchor_unchanged": {
            "raw_anchor_state_bit_exact_terminal": anchor_exact,
            "raw_anchor_state_bit_exact_each_epoch": [
                item["metrics"]["anchor_state_bit_exact_at_epoch_end"]
                for item in history
            ],
            "anchor_contract_every_step": [
                item["metrics"]["anchor_contract_every_step"] for item in history
            ],
            "q0_output_identity_every_step": [
                item["metrics"]["q0_output_identity_every_step"] for item in history
            ],
        },
        "fresh_strict_load": fresh,
        "terminal_sarn_off_exact_q0": terminal_sarn_off_exact_q0,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "correction_state": _state_cpu(correction),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)
    return {
        "status": "complete",
        "terminal_checkpoint": str(output),
        "optimizer_steps": total_steps,
        "samples": len(samples),
        "scenes": len(scene_roster),
        "condition_counts": training["condition_counts_terminal_total"],
        "anchor_state_bit_exact": anchor_exact,
        "fresh_strict_load": fresh,
        "terminal_sarn_off_exact_q0": terminal_sarn_off_exact_q0,
        "access_flags": access_flags,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument("--terminal-a11-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_a13_correction_only(
        correction_train_manifest_path=args.correction_train_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A13ProjectiveOnlyDataset",
    "A13TrainingError",
    "BATCH_SIZE",
    "CONDITION_RATIO",
    "CONDITION_ROSTER",
    "DEFAULT_SEED",
    "LEARNING_RATE",
    "PROTOCOL",
    "SYSTEM",
    "TERMINAL_EPOCHS",
    "WEIGHT_DECAY",
    "active_only_a11_correction_loss",
    "build_a13_train_dataset",
    "build_argument_parser",
    "condition_for_position",
    "configure_a13_reproducibility",
    "correction_optimizer_partition_evidence",
    "expected_condition_counts",
    "frozen_raw_forward",
    "run_a13_correction_epoch",
    "train_a13_correction_only",
]
