"""Targets and fixed float32 objectives for A11 SCORT.

The Raw q0 anchor has one objective: its own posterior read loss.  Every q0
quantity used as a correction reference is detached.  The model-side contract
additionally detaches correction inputs and Raw features before they enter the
SCORT path; this loss module never substitutes an independent parent output.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Final

import torch
import torch.nn.functional as F

from experiments.a11_scort_protocol import CORRECTION_LAYERS, PROGRESS_BINS


SMOOTH_L1_BETA: Final[float] = 0.05
UNIFORM_REGRESSION_BASELINE: Final[float] = (
    0.25 - 0.5 * SMOOTH_L1_BETA + SMOOTH_L1_BETA**2 / 3.0
)
REGRET_SOFTPLUS_TEMPERATURE: Final[float] = 0.002
CVAR_TAIL_FRACTION: Final[float] = 0.25
CVAR_ERROR_NORMALIZATION: Final[float] = 0.05
EPSILON: Final[float] = 1.0e-7


class A11TargetError(ValueError):
    """An A11 target, output, or loss mask is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A11TargetError(message)


def _tensor(value: Any, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _output_tensor(outputs: Mapping[str, Any], names: tuple[str, ...]) -> torch.Tensor:
    for name in names:
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    raise A11TargetError(f"A11 output is missing: {' / '.join(names)}")


@dataclass(frozen=True, slots=True)
class A11LossWeights:
    raw_anchor_read: float = 1.0
    final_read: float = 1.0
    layer_softplus_regret: float = 0.10
    relative_cvar25: float = 0.25
    normalized_angle: float = 0.001

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require(
                math.isfinite(float(value)) and float(value) >= 0.0,
                f"A11 loss weight is invalid: {name}",
            )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


LOSS_WEIGHTS: Final[A11LossWeights] = A11LossWeights()
NO_REGRET_LOSS_WEIGHTS: Final[A11LossWeights] = replace(
    LOSS_WEIGHTS,
    layer_softplus_regret=0.0,
    relative_cvar25=0.0,
)
LOSS_DESIGN_METADATA: Final[dict[str, Any]] = {
    "weights": LOSS_WEIGHTS.as_dict(),
    "no_regret_ablation_weights": NO_REGRET_LOSS_WEIGHTS.as_dict(),
    "layer_regret": (
        "mean_over_active_layers_per_sample_then_mean_over_active_samples_of_"
        "softplus((abs(layer_mean-y)-detach(abs(q0_mean-y)))/temperature)"
    ),
    "regret_softplus_temperature": REGRET_SOFTPLUS_TEMPERATURE,
    "regret_margin": 0.0,
    "cvar": (
        "top_ceil_25_percent_mean_of_per_sample_max_over_active_layers_of_"
        "relu(abs(layer_mean-y)-detach(abs(q0_mean-y)))/0.05"
    ),
    "cvar_tail_fraction": CVAR_TAIL_FRACTION,
    "cvar_error_normalization": CVAR_ERROR_NORMALIZATION,
    "angle": "mean_of_model_reported_normalized_squared_angle_residuals",
    "q0_gradient_scope": "raw_anchor_posterior_read_loss_only",
}


def build_a11_targets(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Build only read supervision; SCORT has no A9 geometry dependency."""

    _require(isinstance(batch, Mapping), "A11 batch must be a mapping")
    target = _tensor(batch.get("target"), "target").detach().float()
    _require(target.ndim == 1, "A11 target must be [B]")
    read_valid = torch.isfinite(target) & (target >= 0.0) & (target <= 1.0)
    safe_read = torch.where(
        read_valid,
        target.clamp(0.0, 1.0),
        torch.zeros_like(target),
    )
    return {"read": safe_read, "read_valid": read_valid}


def _linear_progress_target(target: torch.Tensor, bins: int) -> torch.Tensor:
    _require(target.ndim == 1 and bins >= 8, "progress target/bins are invalid")
    position = target.float().clamp(0.0, 1.0) * float(bins - 1)
    lower = torch.floor(position).long()
    upper = (lower + 1).clamp(max=bins - 1)
    upper_weight = position - lower.float()
    result = torch.zeros(
        target.shape[0], bins, device=target.device, dtype=torch.float32
    )
    result.scatter_add_(1, lower[:, None], (1.0 - upper_weight)[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


def _posterior_read_per_row(
    posterior: torch.Tensor,
    mean: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    _require(
        posterior.ndim == 2
        and mean.shape == target.shape == (posterior.shape[0],),
        "posterior read shapes differ",
    )
    probability = posterior.float().clamp_min(EPSILON)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        EPSILON
    )
    distribution_target = _linear_progress_target(target, posterior.shape[1])
    posterior_ce = -(
        distribution_target * torch.log(probability)
    ).sum(dim=1) / math.log(float(posterior.shape[1]))
    mean_loss = F.smooth_l1_loss(
        mean.float(), target.float(), beta=SMOOTH_L1_BETA, reduction="none"
    ) / UNIFORM_REGRESSION_BASELINE
    return 0.5 * posterior_ce + 0.5 * mean_loss


def _masked_mean_or_graph_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchor: torch.Tensor,
) -> torch.Tensor:
    _require(values.ndim == 1 and values.shape == mask.shape, "masked shapes differ")
    if bool(mask.any()):
        return values.masked_select(mask).mean()
    return anchor.float().sum() * 0.0


def _active_layer_sample_mean(
    values: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equalize samples after averaging the active layers within each sample."""

    _require(
        values.ndim == 2 and values.shape == active.shape,
        "active layer value/mask shapes differ",
    )
    active = active.bool()
    count = active.sum(dim=1)
    row_active = count > 0
    per_sample = (
        torch.where(active, values.float(), torch.zeros_like(values.float())).sum(dim=1)
        / count.clamp_min(1).float()
    )
    aggregate = _masked_mean_or_graph_zero(
        per_sample, row_active, anchor=values
    )
    return aggregate, per_sample, row_active


def relative_layer_cvar25(
    layer_relative_error: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """CVaR25 of each active sample's worst positive layer regret.

    The q0 subtraction must already have used a detached q0 error.  Empty
    active sets return a graph-connected zero and a tail count of zero.
    """

    _require(
        layer_relative_error.ndim == 2
        and layer_relative_error.shape == active.shape,
        "relative CVaR layer shapes differ",
    )
    active = active.bool()
    row_active = active.any(dim=1)
    positive = torch.where(
        active,
        F.relu(layer_relative_error.float()) / CVAR_ERROR_NORMALIZATION,
        torch.zeros_like(layer_relative_error.float()),
    )
    per_sample = positive.max(dim=1).values
    active_values = per_sample.masked_select(row_active)
    if active_values.numel() == 0:
        return layer_relative_error.float().sum() * 0.0, per_sample, 0
    count = max(1, int(math.ceil(CVAR_TAIL_FRACTION * active_values.numel())))
    cvar = torch.topk(active_values, k=count, largest=True).values.mean()
    return cvar, per_sample, count


def _raw_anchor_loss_fp32(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    posterior = _output_tensor(
        outputs, ("raw_anchor_posterior", "raw_posterior")
    ).float()
    mean = _output_tensor(outputs, ("raw_anchor_mean", "raw_mean")).float()
    target = _tensor(targets.get("read"), "read").to(mean.device).float()
    read_valid = _tensor(targets.get("read_valid"), "read_valid").to(
        mean.device
    ).bool()
    batch = mean.shape[0]
    _require(
        posterior.shape == (batch, PROGRESS_BINS)
        and target.shape == read_valid.shape == (batch,),
        "q0 posterior/read target shapes differ",
    )
    per_row = _posterior_read_per_row(posterior, mean, target)
    read = _masked_mean_or_graph_zero(per_row, read_valid, anchor=posterior)
    return read, {
        "raw_anchor_read": read,
        "raw_anchor_read_per_row": per_row,
        "read_valid": read_valid,
    }


def raw_anchor_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The sole optimization objective allowed to update the Raw q0 anchor."""

    reference = _output_tensor(outputs, ("raw_anchor_mean", "raw_mean"))
    with torch.autocast(device_type=reference.device.type, enabled=False):
        return _raw_anchor_loss_fp32(outputs, targets)


def _correction_loss_fp32(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    weights: A11LossWeights,
) -> tuple[torch.Tensor, dict[str, Any]]:
    posterior = _tensor(outputs.get("progress_posterior"), "progress_posterior").float()
    mean = _tensor(outputs.get("mean"), "mean").float()
    q0_mean = _output_tensor(outputs, ("raw_anchor_mean", "raw_mean")).float()
    layer_posteriors = _tensor(
        outputs.get("layer_posteriors"), "layer_posteriors"
    ).float()
    layer_means = _tensor(outputs.get("layer_means"), "layer_means").float()
    angle_residuals = _tensor(
        outputs.get("layer_angle_residuals"), "layer_angle_residuals"
    ).float()
    correction_active = _tensor(
        outputs.get("correction_active"), "correction_active"
    ).to(mean.device).bool()
    target = _tensor(targets.get("read"), "read").to(mean.device).float()
    read_valid = _tensor(targets.get("read_valid"), "read_valid").to(
        mean.device
    ).bool()
    batch = mean.shape[0]
    _require(
        posterior.shape == (batch, PROGRESS_BINS)
        and q0_mean.shape == target.shape == read_valid.shape == (batch,),
        "A11 final/q0 read shapes differ",
    )
    _require(
        layer_posteriors.shape == (batch, CORRECTION_LAYERS, PROGRESS_BINS)
        and layer_means.shape == (batch, CORRECTION_LAYERS)
        and angle_residuals.shape == (batch, CORRECTION_LAYERS)
        and correction_active.shape == (batch, CORRECTION_LAYERS),
        "A11 layer output shapes differ",
    )

    final_per_row = _posterior_read_per_row(posterior, mean, target)
    final_read = _masked_mean_or_graph_zero(
        final_per_row, read_valid, anchor=posterior
    )

    # q0 is a reference, never a correction target with a gradient edge.
    q0_absolute_error = torch.abs(q0_mean.detach() - target)
    layer_relative_error = (
        torch.abs(layer_means - target[:, None]) - q0_absolute_error[:, None]
    )
    active = correction_active & read_valid[:, None]
    softplus_regret_cells = F.softplus(
        layer_relative_error / REGRET_SOFTPLUS_TEMPERATURE
    )
    layer_regret, layer_regret_per_sample, row_active = _active_layer_sample_mean(
        softplus_regret_cells, active
    )
    cvar25, cvar_risk_per_sample, cvar_tail_count = relative_layer_cvar25(
        layer_relative_error, active
    )
    normalized_angle, angle_per_sample, angle_row_active = _active_layer_sample_mean(
        angle_residuals, active
    )
    _require(
        torch.equal(row_active, angle_row_active),
        "regret and angle active rows differ",
    )

    correction_total = (
        weights.final_read * final_read
        + weights.layer_softplus_regret * layer_regret
        + weights.relative_cvar25 * cvar25
        + weights.normalized_angle * normalized_angle
    )
    return correction_total, {
        "correction_total": correction_total,
        "final_read": final_read,
        "final_read_per_row": final_per_row,
        "layer_softplus_regret": layer_regret,
        "layer_softplus_regret_per_sample": layer_regret_per_sample,
        "layer_relative_error": layer_relative_error,
        "relative_cvar25": cvar25,
        "cvar_risk_per_sample": cvar_risk_per_sample,
        "cvar_tail_count": cvar_tail_count,
        "normalized_angle": normalized_angle,
        "normalized_angle_per_sample": angle_per_sample,
        "coverage": {
            "read_valid": read_valid,
            "final_read_applied": read_valid.clone(),
            "correction_active": correction_active,
            "layer_regret_applied": active,
            "cvar_eligible": row_active,
            "angle_regularizer_applied": active.clone(),
        },
    }


def a11_correction_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    weights: A11LossWeights = LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """SCORT-only objective; all q0 references inside it are detached."""

    reference = _tensor(outputs.get("mean"), "mean")
    with torch.autocast(device_type=reference.device.type, enabled=False):
        return _correction_loss_fp32(outputs, targets, weights=weights)


def a11_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    weights: A11LossWeights = LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Joint q0 + correction loss with a recorded, fixed coefficient vector."""

    reference = _tensor(outputs.get("mean"), "mean")
    with torch.autocast(device_type=reference.device.type, enabled=False):
        raw_read, raw_components = _raw_anchor_loss_fp32(outputs, targets)
        correction, correction_components = _correction_loss_fp32(
            outputs, targets, weights=weights
        )
        total = weights.raw_anchor_read * raw_read + correction
    coverage = dict(correction_components["coverage"])
    coverage["raw_anchor_read_applied"] = raw_components["read_valid"].clone()
    return total, {
        "total": total,
        "raw_anchor_read": raw_read,
        **correction_components,
        "coverage": coverage,
        "weights": weights.as_dict(),
    }


__all__ = [
    "A11LossWeights",
    "A11TargetError",
    "CVAR_ERROR_NORMALIZATION",
    "CVAR_TAIL_FRACTION",
    "LOSS_DESIGN_METADATA",
    "LOSS_WEIGHTS",
    "NO_REGRET_LOSS_WEIGHTS",
    "REGRET_SOFTPLUS_TEMPERATURE",
    "SMOOTH_L1_BETA",
    "a11_correction_loss",
    "a11_loss",
    "build_a11_targets",
    "raw_anchor_loss",
    "relative_layer_cvar25",
]
