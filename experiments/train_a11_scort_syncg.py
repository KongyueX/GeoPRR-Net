"""Five terminal epochs of A11 SCORT on the physical Core-train split only.

The Raw anchor and SCORT correction parameters have disjoint AdamW optimizers
and cosine schedulers.  One forward produces both losses, but q0 is stepped
only from :func:`raw_anchor_loss`; Raw gradients are cleared before the
correction backward, whose references and inputs are detached by contract.
No audit/validation manifest argument exists and no checkpoint is selected by
validation.
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

from experiments.a11_scort import (
    A11_ARCHITECTURE,
    A11SCORTImageModel,
    image_model_parameter_counts,
)
from experiments.a11_scort_protocol import (
    CORE_TRAIN_SAMPLES,
    CORE_TRAIN_SCENE_COUNT,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
)
from experiments.a11_scort_targets import (
    LOSS_DESIGN_METADATA,
    LOSS_WEIGHTS,
    a11_correction_loss,
    build_a11_targets,
    raw_anchor_loss,
)
from experiments.prepare_a11_core_scene_split import (
    DEFAULT_TRAIN_MANIFEST,
    PROTOCOL as MANIFEST_PROTOCOL,
    load_a11_train_manifest,
)
from experiments.resnet18_direct_progress import IMAGE_SIZE
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    TRAIN_CONDITION_MIX,
    SyncGSupportGeometryMultiViewDataset,
    _configure_reproducibility,
    _loader,
)


PROTOCOL: Final[str] = "syncg_a11_scort_core_train72_terminal5_v1"
DEFAULT_SEED: Final[int] = 20262020
TERMINAL_EPOCHS: Final[int] = 5
BATCH_SIZE: Final[int] = 8
LEARNING_RATE: Final[float] = 3.0e-4
WEIGHT_DECAY: Final[float] = 1.0e-4
FIXED_TRAIN_CONDITION_MIX: Final[tuple[tuple[str, float], ...]] = (
    ("clean", 0.30),
    ("perspective_moderate", 0.25),
    ("perspective_severe", 0.25),
    ("combined_severe", 0.20),
)
STEPS_PER_EPOCH: Final[int] = math.ceil(CORE_TRAIN_SAMPLES / BATCH_SIZE)
TOTAL_DATA_STEPS: Final[int] = TERMINAL_EPOCHS * STEPS_PER_EPOCH
RAW_PARAMETER_PREFIXES: Final[tuple[str, ...]] = (
    "raw_encoder.",
    "raw_posterior_head.",
)
CORRECTION_MODULE_NAMES: Final[tuple[str, ...]] = (
    "sarn_encoder",
    "relation_encoder",
    "progress_decoder",
    "orthogonal_transport",
)
CORRECTION_GRADIENT_GROUP_NAMES: Final[tuple[str, ...]] = (
    *CORRECTION_MODULE_NAMES,
    "angle_output",
)


class A11TrainingError(ValueError):
    """A11 training input, gradient, optimizer, or terminal state is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A11TrainingError(message)


def configure_a11_reproducibility(seed: int, device: torch.device) -> None:
    _require(int(seed) == DEFAULT_SEED, "A11 uses fixed seed 20262020")
    _configure_reproducibility(int(seed), device)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _device_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _device_value(item, device) for key, item in value.items()}
    return value


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _module_state_finite(model: torch.nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
    )


def _parameter_ids(parameters: Sequence[torch.nn.Parameter]) -> set[int]:
    return {id(parameter) for parameter in parameters}


def a11_parameter_groups(
    model: A11SCORTImageModel,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.nn.Parameter, ...]]:
    raw: list[torch.nn.Parameter] = []
    correction: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(RAW_PARAMETER_PREFIXES):
            raw.append(parameter)
        else:
            correction.append(parameter)
    raw_values = tuple(raw)
    correction_values = tuple(correction)
    all_ids = _parameter_ids(tuple(model.parameters()))
    raw_ids = _parameter_ids(raw_values)
    correction_ids = _parameter_ids(correction_values)
    _require(bool(raw_values) and bool(correction_values), "A11 parameter group is empty")
    _require(raw_ids.isdisjoint(correction_ids), "A11 parameter groups overlap")
    _require(raw_ids | correction_ids == all_ids, "A11 parameter groups do not cover model")
    return raw_values, correction_values


def optimizer_partition_evidence(
    model: A11SCORTImageModel,
    raw_optimizer: torch.optim.Optimizer,
    correction_optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    raw_parameters, correction_parameters = a11_parameter_groups(model)
    expected_raw = _parameter_ids(raw_parameters)
    expected_correction = _parameter_ids(correction_parameters)
    actual_raw = {
        id(parameter)
        for group in raw_optimizer.param_groups
        for parameter in group["params"]
    }
    actual_correction = {
        id(parameter)
        for group in correction_optimizer.param_groups
        for parameter in group["params"]
    }
    return {
        "distinct_optimizer_objects": raw_optimizer is not correction_optimizer,
        "optimizer_parameter_sets_disjoint": actual_raw.isdisjoint(actual_correction),
        "raw_optimizer_exact_parameter_set": actual_raw == expected_raw,
        "correction_optimizer_exact_parameter_set": (
            actual_correction == expected_correction
        ),
        "all_trainable_parameters_covered": (
            actual_raw | actual_correction
            == _parameter_ids(tuple(model.parameters()))
        ),
        "raw_parameter_count": sum(parameter.numel() for parameter in raw_parameters),
        "correction_parameter_count": sum(
            parameter.numel() for parameter in correction_parameters
        ),
    }


def model_construction_metadata(model: A11SCORTImageModel) -> dict[str, Any]:
    return {
        "imagenet_pretrained": bool(model.raw_encoder.imagenet_pretrained),
        "relation_channels": int(model.relation_channels),
        "token_dim": int(model.token_dim),
        "attention_heads": int(model.attention_heads),
        "decoder_layers": int(model.decoder_layers),
        "memory_grid_size": int(model.memory_grid_size),
        "progress_bins": int(model.progress_bins),
    }


def build_a11_train_dataset(
    samples: Sequence[Any],
    *,
    seed: int = DEFAULT_SEED,
    total_epochs: int = TERMINAL_EPOCHS,
) -> SyncGSupportGeometryMultiViewDataset:
    _require(len(samples) == CORE_TRAIN_SAMPLES, "A11 train sample count differs")
    return SyncGSupportGeometryMultiViewDataset(
        samples,
        training=True,
        seed=int(seed),
        total_epochs=int(total_epochs),
        image_size=IMAGE_SIZE,
    )


def _autocast_settings(device: torch.device) -> tuple[bool, torch.dtype, str]:
    if device.type != "cuda":
        return False, torch.float32, "float32"
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, "bfloat16"
    return True, torch.float16, "float16"


def _gradient_summary(parameters: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for gradient in gradients:
        by_device.setdefault(gradient.device, []).append(gradient)
    finite = bool(gradients)
    nonzero_sum = 0.0
    for values in by_device.values():
        device_finite = torch.stack(
            [torch.isfinite(value).all() for value in values]
        ).all()
        finite = finite and bool(device_finite)
        if finite:
            device_sum = torch.stack(
                [value.detach().abs().double().sum() for value in values]
            ).sum()
            nonzero_sum += float(device_sum.cpu())
    if not finite:
        nonzero_sum = 0.0
    return {
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero": nonzero_sum > 0.0,
        "absolute_sum": nonzero_sum,
    }


def _all_gradients_zero_or_absent(parameters: Sequence[torch.nn.Parameter]) -> bool:
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            by_device.setdefault(parameter.grad.device, []).append(parameter.grad)
    return all(
        not bool(
            torch.stack([(gradient != 0.0).any() for gradient in gradients]).any()
        )
        for gradients in by_device.values()
    )


def _backward_unscale_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def _finite_state_value_summary(value: Any) -> dict[str, Any]:
    """Recursively count and check numeric optimizer/scaler state values."""

    tensor_count = floating_tensor_count = numeric_scalar_count = 0
    finite = True
    floating_tensors: list[torch.Tensor] = []

    def visit(item: Any) -> None:
        nonlocal tensor_count, floating_tensor_count, numeric_scalar_count, finite
        if isinstance(item, torch.Tensor):
            tensor_count += 1
            if item.is_floating_point() or item.is_complex():
                floating_tensor_count += 1
                floating_tensors.append(item)
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, float):
            numeric_scalar_count += 1
            finite = finite and math.isfinite(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            numeric_scalar_count += 1

    visit(value)
    tensors_by_device: dict[torch.device, list[torch.Tensor]] = {}
    for tensor in floating_tensors:
        tensors_by_device.setdefault(tensor.device, []).append(tensor)
    for tensors in tensors_by_device.values():
        device_finite = torch.stack(
            [torch.isfinite(tensor).all() for tensor in tensors]
        ).all()
        finite = finite and bool(device_finite)
    return {
        "finite": finite,
        "tensor_count": tensor_count,
        "floating_tensor_count": floating_tensor_count,
        "numeric_scalar_count": numeric_scalar_count,
    }


def optimizer_state_finite_evidence(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Report finite Adam-family moving-average state after an optimizer step."""

    state = optimizer.state_dict().get("state", {})
    summary = _finite_state_value_summary(state)
    values = tuple(state.values()) if isinstance(state, Mapping) else ()
    exp_avg_count = sum(
        isinstance(item, Mapping) and isinstance(item.get("exp_avg"), torch.Tensor)
        for item in values
    )
    exp_avg_sq_count = sum(
        isinstance(item, Mapping) and isinstance(item.get("exp_avg_sq"), torch.Tensor)
        for item in values
    )
    return {
        **summary,
        "parameter_state_count": len(values),
        "exp_avg_tensor_count": int(exp_avg_count),
        "exp_avg_sq_tensor_count": int(exp_avg_sq_count),
        "adam_moving_averages_present": bool(exp_avg_count and exp_avg_sq_count),
    }


def scaler_state_finite_evidence(
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    """Report finite AMP scaler state; disabled/no scaler has no numeric state."""

    state = {} if scaler is None else scaler.state_dict()
    return {
        **_finite_state_value_summary(state),
        "present": scaler is not None,
        "enabled": bool(scaler is not None and scaler.is_enabled()),
    }


def optimizer_and_scaler_state_finite_evidence(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    optimizer_evidence = optimizer_state_finite_evidence(optimizer)
    scaler_evidence = scaler_state_finite_evidence(scaler)
    return {
        "optimizer": optimizer_evidence,
        "scaler": scaler_evidence,
        "finite": bool(
            optimizer_evidence["finite"]
            and optimizer_evidence["adam_moving_averages_present"]
            and scaler_evidence["finite"]
        ),
    }


def _semantic_parameter_groups(
    model: A11SCORTImageModel,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    values = {
        "raw_encoder": tuple(model.raw_encoder.parameters()),
        "raw_posterior_head": tuple(model.raw_posterior_head.parameters()),
        **{
            name: tuple(getattr(model, name).parameters())
            for name in CORRECTION_MODULE_NAMES
        },
    }
    values["angle_output"] = tuple(
        model.orthogonal_transport.angle_output.parameters()
    )
    return values


def run_a11_epoch(
    model: A11SCORTImageModel,
    loader: Any,
    *,
    device: torch.device,
    raw_optimizer: torch.optim.Optimizer | None,
    correction_optimizer: torch.optim.Optimizer | None,
    raw_scaler: torch.amp.GradScaler | None = None,
    correction_scaler: torch.amp.GradScaler | None = None,
    max_steps: int | None = None,
    record_step_gradient_trace: bool = False,
    record_step_parameter_updates: bool = False,
) -> dict[str, Any]:
    """Run one train-only batch stream with two isolated optimizer steps."""

    training = raw_optimizer is not None or correction_optimizer is not None
    _require(
        (raw_optimizer is None) == (correction_optimizer is None),
        "raw/correction optimizers must both be present or absent",
    )
    model.train(training)
    raw_parameters, correction_parameters = a11_parameter_groups(model)
    semantic = _semantic_parameter_groups(model)
    autocast_enabled, autocast_dtype, precision = _autocast_settings(device)
    total_samples = steps = raw_steps = correction_steps = 0
    raw_loss_sum = correction_loss_sum = 0.0
    final_error_sum = q0_error_sum = 0.0
    relation_available_rows = 0
    condition_counts: Counter[str] = Counter()
    gradient_nonzero_seen = {name: False for name in semantic}
    gradient_finite_every_step = {name: True for name in semantic}
    raw_isolation_every_step = True
    correction_isolation_every_step = True
    optimizer_and_scaler_finite_every_step = {"raw": True, "correction": True}
    terminal_state_evidence: dict[str, Any] = {}
    gradient_trace: list[dict[str, Any]] = []
    parameter_update_trace: list[dict[str, bool]] = []
    component_sums = {
        "final_read": 0.0,
        "layer_softplus_regret": 0.0,
        "relative_cvar25": 0.0,
        "normalized_angle": 0.0,
    }
    batch_sizes: list[int] = []

    for raw_batch in loader:
        if max_steps is not None and steps >= int(max_steps):
            break
        batch = _device_value(raw_batch, device)
        targets = _device_value(build_a11_targets(raw_batch), device)
        read = targets["read"]
        read_valid = targets["read_valid"]
        _require(read.ndim == 1 and bool(read_valid.all()), "physical train read is invalid")
        batch_size = int(read.shape[0])
        if training:
            assert raw_optimizer is not None and correction_optimizer is not None
            raw_optimizer.zero_grad(set_to_none=True)
            correction_optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = model(
                    batch["original_view"],
                    batch["sarn_view"],
                    batch["sarn_support_mask"],
                    sarn_active=batch["sarn_active"],
                    raw_to_sarn_homography=batch["raw_to_sarn_homography"],
                )
                raw_loss, _raw_components = raw_anchor_loss(output, targets)
                correction_loss, correction_components = a11_correction_loss(
                    output, targets
                )
        _require(
            bool(torch.isfinite(raw_loss)) and bool(torch.isfinite(correction_loss)),
            "A11 loss is non-finite",
        )
        step_gradient: dict[str, Any] = {}
        if training:
            assert raw_optimizer is not None and correction_optimizer is not None
            parameters_before = (
                {
                    name: tuple(
                        parameter.detach().clone() for parameter in parameters
                    )
                    for name, parameters in semantic.items()
                }
                if record_step_parameter_updates
                else {}
            )
            _backward_unscale_step(raw_loss, raw_optimizer, raw_scaler)
            raw_group_summary = {
                name: _gradient_summary(semantic[name])
                for name in ("raw_encoder", "raw_posterior_head")
            }
            _require(
                all(value["finite"] for value in raw_group_summary.values()),
                "Raw-anchor gradient is absent or non-finite",
            )
            raw_isolated = _all_gradients_zero_or_absent(correction_parameters)
            _require(raw_isolated, "q0 read loss reached correction parameters")
            raw_isolation_every_step = raw_isolation_every_step and raw_isolated
            _optimizer_step(raw_optimizer, raw_scaler)
            raw_state_evidence = optimizer_and_scaler_state_finite_evidence(
                raw_optimizer, raw_scaler
            )
            _require(
                bool(raw_state_evidence["finite"]),
                "Raw-anchor optimizer or scaler state is non-finite",
            )
            optimizer_and_scaler_finite_every_step["raw"] = (
                optimizer_and_scaler_finite_every_step["raw"]
                and bool(raw_state_evidence["finite"])
            )
            raw_steps += 1
            raw_optimizer.zero_grad(set_to_none=True)
            correction_optimizer.zero_grad(set_to_none=True)

            _backward_unscale_step(
                correction_loss, correction_optimizer, correction_scaler
            )
            correction_group_summary = {
                name: _gradient_summary(semantic[name])
                for name in CORRECTION_GRADIENT_GROUP_NAMES
            }
            _require(
                all(value["finite"] for value in correction_group_summary.values()),
                "correction gradient is absent or non-finite",
            )
            correction_isolated = _all_gradients_zero_or_absent(raw_parameters)
            _require(correction_isolated, "correction loss reached Raw parameters")
            correction_isolation_every_step = (
                correction_isolation_every_step and correction_isolated
            )
            _optimizer_step(correction_optimizer, correction_scaler)
            correction_state_evidence = optimizer_and_scaler_state_finite_evidence(
                correction_optimizer, correction_scaler
            )
            _require(
                bool(correction_state_evidence["finite"]),
                "correction optimizer or scaler state is non-finite",
            )
            optimizer_and_scaler_finite_every_step["correction"] = (
                optimizer_and_scaler_finite_every_step["correction"]
                and bool(correction_state_evidence["finite"])
            )
            terminal_state_evidence = {
                "raw": raw_state_evidence,
                "correction": correction_state_evidence,
            }
            correction_steps += 1
            step_gradient = {**raw_group_summary, **correction_group_summary}
            for name, summary in step_gradient.items():
                gradient_finite_every_step[name] = (
                    gradient_finite_every_step[name] and bool(summary["finite"])
                )
                gradient_nonzero_seen[name] = (
                    gradient_nonzero_seen[name] or bool(summary["nonzero"])
                )
            if record_step_gradient_trace:
                gradient_trace.append(step_gradient)
            if record_step_parameter_updates:
                parameter_update_trace.append(
                    {
                        name: any(
                            not torch.equal(previous, current.detach())
                            for previous, current in zip(
                                parameters_before[name], parameters, strict=True
                            )
                        )
                        for name, parameters in semantic.items()
                    }
                )

        final_error = torch.abs(output["mean"].detach().float() - read)
        q0_error = torch.abs(output["raw_anchor_mean"].detach().float() - read)
        final_error_sum += float(final_error.sum().cpu())
        q0_error_sum += float(q0_error.sum().cpu())
        raw_loss_sum += float(raw_loss.detach().cpu()) * batch_size
        correction_loss_sum += float(correction_loss.detach().cpu()) * batch_size
        for name in component_sums:
            component_sums[name] += (
                float(correction_components[name].detach().cpu()) * batch_size
            )
        relation_available_rows += int(output["relation_available"].detach().sum().cpu())
        names = raw_batch.get("condition_name", [])
        if isinstance(names, str):
            names = [names]
        condition_counts.update(str(name) for name in names)
        total_samples += batch_size
        batch_sizes.append(batch_size)
        steps += 1
    _require(total_samples > 0, "A11 epoch produced no samples")
    return {
        "samples": total_samples,
        "steps": steps,
        "batch_sizes": batch_sizes,
        "raw_optimizer_steps": raw_steps,
        "correction_optimizer_steps": correction_steps,
        "raw_anchor_loss": raw_loss_sum / total_samples,
        "correction_loss": correction_loss_sum / total_samples,
        "final_nmae": final_error_sum / total_samples,
        "q0_nmae": q0_error_sum / total_samples,
        "nmae_delta_final_minus_q0": (final_error_sum - q0_error_sum) / total_samples,
        "relation_available_rows": relation_available_rows,
        "condition_counts": dict(sorted(condition_counts.items())),
        "correction_components": {
            name: value / total_samples for name, value in component_sums.items()
        },
        "gradient_nonzero_seen": gradient_nonzero_seen,
        "gradient_finite_every_step": gradient_finite_every_step,
        "raw_loss_correction_gradient_isolated_every_step": raw_isolation_every_step,
        "correction_loss_raw_gradient_isolated_every_step": (
            correction_isolation_every_step
        ),
        "optimizer_and_scaler_state_finite_every_step": (
            optimizer_and_scaler_finite_every_step
        ),
        "terminal_optimizer_and_scaler_state": terminal_state_evidence,
        "gradient_trace": gradient_trace,
        "parameter_update_trace": parameter_update_trace,
        "autocast_precision": precision,
        "gradient_clipping": None,
    }


def train_a11_core_train_only(
    *,
    train_manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
) -> dict[str, Any]:
    """Train Full A11 for exactly five epochs and save only the terminal state."""

    _require(workers >= 0, "A11 workers must be non-negative")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"terminal output already exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a11_reproducibility(DEFAULT_SEED, device)
    _require(
        tuple(TRAIN_CONDITION_MIX) == FIXED_TRAIN_CONDITION_MIX,
        "A11 dataset condition mixture differs from 30/25/25/20",
    )
    samples = load_a11_train_manifest(train_manifest_path)
    dataset = build_a11_train_dataset(samples)
    torch.manual_seed(DEFAULT_SEED)
    model = A11SCORTImageModel(imagenet_pretrained=True).to(device)
    raw_parameters, correction_parameters = a11_parameter_groups(model)
    raw_optimizer = torch.optim.AdamW(
        raw_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    correction_optimizer = torch.optim.AdamW(
        correction_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    partition_evidence = optimizer_partition_evidence(
        model, raw_optimizer, correction_optimizer
    )
    _require(
        all(
            bool(partition_evidence[key])
            for key in (
                "distinct_optimizer_objects",
                "optimizer_parameter_sets_disjoint",
                "raw_optimizer_exact_parameter_set",
                "correction_optimizer_exact_parameter_set",
                "all_trainable_parameters_covered",
            )
        ),
        "A11 optimizer partition differs",
    )
    raw_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        raw_optimizer, T_max=TERMINAL_EPOCHS
    )
    correction_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        correction_optimizer, T_max=TERMINAL_EPOCHS
    )
    scheduler_partition_evidence = {
        "distinct_scheduler_objects": raw_scheduler is not correction_scheduler,
        "raw_scheduler_owns_raw_optimizer": raw_scheduler.optimizer is raw_optimizer,
        "correction_scheduler_owns_correction_optimizer": (
            correction_scheduler.optimizer is correction_optimizer
        ),
    }
    _require(
        all(scheduler_partition_evidence.values()),
        "A11 scheduler partition differs",
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    raw_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    correction_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    history: list[dict[str, Any]] = []
    raw_steps = correction_steps = 0
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
        metrics = run_a11_epoch(
            model,
            loader,
            device=device,
            raw_optimizer=raw_optimizer,
            correction_optimizer=correction_optimizer,
            raw_scaler=raw_scaler,
            correction_scaler=correction_scaler,
        )
        _require(
            metrics["steps"] == STEPS_PER_EPOCH
            and metrics["samples"] == CORE_TRAIN_SAMPLES,
            "A11 epoch does not cover the complete physical train split",
        )
        _require(
            sum(metrics["condition_counts"].values()) == CORE_TRAIN_SAMPLES
            and set(metrics["condition_counts"])
            == {name for name, _probability in FIXED_TRAIN_CONDITION_MIX},
            "A11 epoch condition observations differ",
        )
        _require(
            all(metrics["gradient_finite_every_step"].values())
            and all(metrics["gradient_nonzero_seen"].values()),
            "A11 epoch gradient evidence differs",
        )
        _require(
            metrics["raw_loss_correction_gradient_isolated_every_step"]
            and metrics["correction_loss_raw_gradient_isolated_every_step"],
            "A11 optimizer gradient isolation differs",
        )
        _require(
            all(metrics["optimizer_and_scaler_state_finite_every_step"].values()),
            "A11 optimizer/scaler state evidence differs",
        )
        _require(_module_state_finite(model), "A11 epoch state is non-finite")
        raw_steps += int(metrics["raw_optimizer_steps"])
        correction_steps += int(metrics["correction_optimizer_steps"])
        history.append(
            {
                "epoch": epoch + 1,
                "raw_learning_rate": raw_optimizer.param_groups[0]["lr"],
                "correction_learning_rate": correction_optimizer.param_groups[0]["lr"],
                "metrics": metrics,
            }
        )
        raw_scheduler.step()
        correction_scheduler.step()
    _require(
        raw_steps == correction_steps == TOTAL_DATA_STEPS,
        "A11 did not complete exactly 4,965 steps per optimizer",
    )
    _require(_module_state_finite(model), "A11 terminal state is non-finite")
    terminal_optimizer_state = {
        "raw": optimizer_and_scaler_state_finite_evidence(raw_optimizer, raw_scaler),
        "correction": optimizer_and_scaler_state_finite_evidence(
            correction_optimizer, correction_scaler
        ),
    }
    _require(
        all(value["finite"] for value in terminal_optimizer_state.values()),
        "A11 terminal optimizer/scaler state is non-finite",
    )
    access_flags = {
        "train_manifest_access": True,
        "audit_manifest_access": False,
        "core_audit_predictions_generated": False,
        "fold_a_content_access": False,
        "fold_b_content_access": False,
        "formal_holdout_content_access": False,
        "field_photo_content_access": False,
    }
    training = {
        "seed": DEFAULT_SEED,
        "epochs": TERMINAL_EPOCHS,
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "samples_per_epoch": CORE_TRAIN_SAMPLES,
        "scenes": CORE_TRAIN_SCENE_COUNT,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps_each": TOTAL_DATA_STEPS,
        "total_data_batches": TOTAL_DATA_STEPS,
        "two_optimizer_step_applications": 2 * TOTAL_DATA_STEPS,
        "optimizer": "two_disjoint_AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "two_disjoint_CosineAnnealingLR_Tmax5",
        "condition_mix": dict(FIXED_TRAIN_CONDITION_MIX),
        "gradient_clipping": None,
        "ema": None,
        "amp": "bfloat16_if_supported_else_float16_cuda",
        "train_manifest": str(Path(train_manifest_path).resolve()),
        "validation_manifest": None,
    }
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "manifest_protocol": MANIFEST_PROTOCOL,
        "system": "a11_scort_full",
        "architecture": A11_ARCHITECTURE,
        "construction": model_construction_metadata(model),
        "parameter_counts": image_model_parameter_counts(model),
        "optimizer_partition_evidence": partition_evidence,
        "scheduler_partition_evidence": scheduler_partition_evidence,
        "terminal_optimizer_and_scaler_state": terminal_optimizer_state,
        "training": training,
        "loss_design": dict(LOSS_DESIGN_METADATA),
        "loss_weights": LOSS_WEIGHTS.as_dict(),
        "history": history,
        "access_flags": access_flags,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "model_state": _model_state_cpu(model),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)
    return {
        "status": "complete",
        "terminal_checkpoint": str(output),
        "raw_optimizer_steps": raw_steps,
        "correction_optimizer_steps": correction_steps,
        "parameter_counts": checkpoint["parameter_counts"],
        "access_flags": access_flags,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_a11_core_train_only(
        train_manifest_path=args.train_manifest,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A11TrainingError",
    "BATCH_SIZE",
    "DEFAULT_SEED",
    "FIXED_TRAIN_CONDITION_MIX",
    "LEARNING_RATE",
    "PROTOCOL",
    "STEPS_PER_EPOCH",
    "TERMINAL_EPOCHS",
    "TOTAL_DATA_STEPS",
    "WEIGHT_DECAY",
    "a11_parameter_groups",
    "build_a11_train_dataset",
    "configure_a11_reproducibility",
    "model_construction_metadata",
    "optimizer_partition_evidence",
    "optimizer_and_scaler_state_finite_evidence",
    "optimizer_state_finite_evidence",
    "run_a11_epoch",
    "scaler_state_finite_evidence",
    "train_a11_core_train_only",
]
