"""Fixed float32 supervision for the A15 FTEB correction.

The frozen Raw and SARN endpoint distributions are references, not trainable
parents.  A15 learns from the final read, a dense eight-step CDF path, risk
relative to the fixed geometric endpoint, and agreement between the three
projective views of each physical sample.  Endpoint-only SARN scores are
reported as diagnostics and deliberately do not enter the optimization loss.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F

from experiments.a11_scort_targets import (
    EPSILON,
    SMOOTH_L1_BETA,
    UNIFORM_REGRESSION_BASELINE,
)
from experiments.a15_fteb import BRIDGE_LAYERS, DEFAULT_PROGRESS_BINS


CDF_ERROR_NORMALIZATION: Final[float] = 0.05
GEOMETRIC_REGRET_TEMPERATURE: Final[float] = 0.003
CVAR_TAIL_FRACTION: Final[float] = 0.25


class A15TargetError(ValueError):
    """An A15 target, model output, or physical grouping is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A15TargetError(message)


def _tensor(value: Any, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


@dataclass(frozen=True, slots=True)
class A15LossWeights:
    final_read: float = 1.0
    dense_cdf_path: float = 0.10
    geometric_softplus_regret: float = 0.10
    absolute_final_cvar25: float = 0.25
    cross_condition_w1: float = 0.05
    learned_delta_energy: float = 1.0e-4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require(
                math.isfinite(float(value)) and float(value) >= 0.0,
                f"A15 loss weight is invalid: {name}",
            )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


LOSS_WEIGHTS: Final[A15LossWeights] = A15LossWeights()
LOSS_DESIGN_METADATA: Final[dict[str, Any]] = {
    "weights": LOSS_WEIGHTS.as_dict(),
    "active_rows": "read_valid_and_relation_available",
    "final_read": "A11_normalized_twohot_CE_plus_smooth_L1_formula",
    "dense_cdf_path": (
        "mean_absolute_CDF_error_over_active_rows_eight_layers_and_bins_"
        "against_linear_q0_to_twohot_path_divided_by_0.05"
    ),
    "geometric_softplus_regret": (
        "0.003_times_softplus((abs(final-y)-detach(abs(geometric-y)))/0.003)_"
        "divided_by_0.05_on_active_rows"
    ),
    "absolute_final_cvar25": (
        "top_ceil_25_percent_active_absolute_final_error_mean_divided_by_0.05"
    ),
    "cross_condition_w1": (
        "three_pair_mean_within_each_complete_active_three_view_physical_group_"
        "then_group_mean_divided_by_0.05"
    ),
    "learned_delta_energy": "active_row_mean_only",
    "sarn_endpoint_diagnostics": (
        "normalized_twohot_CE_and_target_CDF_error_are_metrics_only"
    ),
    "cdf_error_normalization": CDF_ERROR_NORMALIZATION,
    "geometric_regret_temperature": GEOMETRIC_REGRET_TEMPERATURE,
    "cvar_tail_fraction": CVAR_TAIL_FRACTION,
}


def _linear_twohot(read: torch.Tensor, bins: int) -> torch.Tensor:
    _require(read.ndim == 1 and bins >= 8, "A15 read/bins are invalid")
    position = read.float().clamp(0.0, 1.0) * float(bins - 1)
    lower = torch.floor(position).long()
    upper = (lower + 1).clamp(max=bins - 1)
    upper_weight = position - lower.float()
    result = torch.zeros(
        read.shape[0], bins, device=read.device, dtype=torch.float32
    )
    result.scatter_add_(1, lower[:, None], (1.0 - upper_weight)[:, None])
    result.scatter_add_(1, upper[:, None], upper_weight[:, None])
    return result


def build_a15_targets(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Build detached read, two-hot posterior, and CDF supervision."""

    _require(isinstance(batch, Mapping), "A15 batch must be a mapping")
    target = _tensor(batch.get("target"), "target").detach().float()
    _require(target.ndim == 1, "A15 target must be [B]")
    read_valid = torch.isfinite(target) & (target >= 0.0) & (target <= 1.0)
    read = torch.where(
        read_valid,
        target.clamp(0.0, 1.0),
        torch.zeros_like(target),
    )
    twohot = _linear_twohot(read, DEFAULT_PROGRESS_BINS)
    return {
        "read": read,
        "read_valid": read_valid,
        "twohot": twohot,
        "cdf": twohot.cumsum(dim=1),
    }


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


def _normalized_posterior_ce_per_row(
    posterior: torch.Tensor,
    twohot: torch.Tensor,
) -> torch.Tensor:
    _require(
        posterior.shape == twohot.shape and posterior.ndim == 2,
        "posterior/twohot shapes differ",
    )
    probability = posterior.float().clamp_min(EPSILON)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        EPSILON
    )
    return -(twohot * torch.log(probability)).sum(dim=1) / math.log(
        float(posterior.shape[1])
    )


def _a11_read_per_row(
    posterior: torch.Tensor,
    mean: torch.Tensor,
    read: torch.Tensor,
    twohot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(
        mean.shape == read.shape == (posterior.shape[0],),
        "A15 final posterior/read shapes differ",
    )
    posterior_ce = _normalized_posterior_ce_per_row(posterior, twohot)
    mean_smooth_l1 = F.smooth_l1_loss(
        mean.float(), read.float(), beta=SMOOTH_L1_BETA, reduction="none"
    ) / UNIFORM_REGRESSION_BASELINE
    return 0.5 * posterior_ce + 0.5 * mean_smooth_l1, posterior_ce, mean_smooth_l1


def _physical_groups(
    physical_group_ids: Sequence[Any] | torch.Tensor,
    *,
    batch: int,
) -> tuple[tuple[int, int, int], ...]:
    if isinstance(physical_group_ids, torch.Tensor):
        _require(
            physical_group_ids.ndim == 1
            and physical_group_ids.shape[0] == batch,
            "physical_group_ids tensor must be [B]",
        )
        identifiers = tuple(physical_group_ids.detach().cpu().tolist())
    else:
        _require(
            isinstance(physical_group_ids, Sequence)
            and not isinstance(physical_group_ids, (str, bytes)),
            "physical_group_ids must be a one-dimensional tensor or sequence",
        )
        identifiers = tuple(physical_group_ids)
        _require(len(identifiers) == batch, "physical_group_ids length differs")

    grouped: dict[Any, list[int]] = {}
    try:
        for row, identifier in enumerate(identifiers):
            grouped.setdefault(identifier, []).append(row)
    except TypeError as error:
        raise A15TargetError("physical_group_ids must be hashable") from error
    _require(bool(grouped), "physical_group_ids must not be empty")
    _require(
        all(len(rows) == 3 for rows in grouped.values()),
        "each physical group must contain exactly three condition rows",
    )
    return tuple(tuple(rows) for rows in grouped.values())  # type: ignore[return-value]


def _cross_condition_w1(
    final_cdf: torch.Tensor,
    active: torch.Tensor,
    physical_group_ids: Sequence[Any] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = _physical_groups(physical_group_ids, batch=final_cdf.shape[0])
    group_values: list[torch.Tensor] = []
    group_active_values: list[torch.Tensor] = []
    denominator = float(final_cdf.shape[1] - 1)
    for rows in groups:
        index = torch.tensor(rows, device=final_cdf.device, dtype=torch.long)
        values = final_cdf.index_select(0, index)
        pair_values = torch.stack(
            (
                torch.abs(values[0] - values[1]).sum() / denominator,
                torch.abs(values[0] - values[2]).sum() / denominator,
                torch.abs(values[1] - values[2]).sum() / denominator,
            )
        )
        group_values.append(pair_values.mean())
        group_active_values.append(active.index_select(0, index).all())
    per_group = torch.stack(group_values)
    group_active = torch.stack(group_active_values).bool()
    value = _masked_mean_or_graph_zero(
        per_group / CDF_ERROR_NORMALIZATION,
        group_active,
        anchor=final_cdf,
    )
    return value, per_group, group_active


def _a15_fteb_loss_fp32(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    physical_group_ids: Sequence[Any] | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    final_posterior = _tensor(
        outputs.get("progress_posterior"), "progress_posterior"
    ).float()
    final_cdf = _tensor(outputs.get("progress_cdf"), "progress_cdf").float()
    final_mean = _tensor(outputs.get("mean"), "mean").float()
    layer_cdfs = _tensor(outputs.get("layer_cdfs"), "layer_cdfs").float()
    q0_cdf = _tensor(outputs.get("raw_anchor_cdf"), "raw_anchor_cdf").detach().float()
    geometric_mean = _tensor(
        outputs.get("geometric_base_mean"), "geometric_base_mean"
    ).detach().float()
    relation_available = _tensor(
        outputs.get("relation_available"), "relation_available"
    ).to(final_mean.device).bool()
    learned_delta_energy = _tensor(
        outputs.get("learned_delta_energy"), "learned_delta_energy"
    ).float()
    sarn_posterior = _tensor(
        outputs.get("sarn_endpoint_posterior"), "sarn_endpoint_posterior"
    ).detach().float()
    sarn_cdf = _tensor(
        outputs.get("sarn_endpoint_cdf"), "sarn_endpoint_cdf"
    ).detach().float()

    read = _tensor(targets.get("read"), "read").to(final_mean.device).float()
    read_valid = _tensor(targets.get("read_valid"), "read_valid").to(
        final_mean.device
    ).bool()
    twohot = _tensor(targets.get("twohot"), "twohot").to(
        final_mean.device
    ).float()
    target_cdf = _tensor(targets.get("cdf"), "cdf").to(final_mean.device).float()

    batch = final_mean.shape[0]
    expected_posterior = (batch, DEFAULT_PROGRESS_BINS)
    _require(
        final_posterior.shape
        == final_cdf.shape
        == q0_cdf.shape
        == sarn_posterior.shape
        == sarn_cdf.shape
        == twohot.shape
        == target_cdf.shape
        == expected_posterior,
        "A15 posterior/CDF target shapes differ",
    )
    _require(
        layer_cdfs.shape
        == (batch, BRIDGE_LAYERS, DEFAULT_PROGRESS_BINS),
        "A15 layer CDF shape differs",
    )
    _require(
        final_mean.shape
        == geometric_mean.shape
        == relation_available.shape
        == learned_delta_energy.shape
        == read.shape
        == read_valid.shape
        == (batch,),
        "A15 row output/target shapes differ",
    )
    active = read_valid & relation_available

    final_per_row, final_ce_per_row, final_mean_per_row = _a11_read_per_row(
        final_posterior, final_mean, read, twohot
    )
    final_read = _masked_mean_or_graph_zero(
        final_per_row, active, anchor=final_posterior
    )

    layer_times = (
        torch.arange(
            1,
            BRIDGE_LAYERS + 1,
            device=final_mean.device,
            dtype=torch.float32,
        )
        / float(BRIDGE_LAYERS)
    )
    detached_q0_cdf = q0_cdf.detach()
    target_path = detached_q0_cdf[:, None] + layer_times[None, :, None] * (
        target_cdf[:, None] - detached_q0_cdf[:, None]
    )
    dense_cdf_per_row = torch.abs(layer_cdfs - target_path).mean(dim=(1, 2))
    dense_cdf_path = _masked_mean_or_graph_zero(
        dense_cdf_per_row / CDF_ERROR_NORMALIZATION,
        active,
        anchor=layer_cdfs,
    )

    final_absolute_error = torch.abs(final_mean - read)
    geometric_absolute_error = torch.abs(geometric_mean.detach() - read)
    geometric_regret_per_row = (
        GEOMETRIC_REGRET_TEMPERATURE
        * F.softplus(
            (final_absolute_error - geometric_absolute_error.detach())
            / GEOMETRIC_REGRET_TEMPERATURE
        )
        / CDF_ERROR_NORMALIZATION
    )
    geometric_softplus_regret = _masked_mean_or_graph_zero(
        geometric_regret_per_row, active, anchor=final_mean
    )

    active_absolute_errors = final_absolute_error.masked_select(active)
    if active_absolute_errors.numel() == 0:
        absolute_final_cvar25 = final_mean.sum() * 0.0
        cvar_tail_count = 0
    else:
        cvar_tail_count = max(
            1,
            int(math.ceil(CVAR_TAIL_FRACTION * active_absolute_errors.numel())),
        )
        absolute_final_cvar25 = (
            torch.topk(
                active_absolute_errors,
                k=cvar_tail_count,
                largest=True,
            ).values.mean()
            / CDF_ERROR_NORMALIZATION
        )

    cross_condition_w1, cross_condition_w1_per_group, cross_condition_group_active = (
        _cross_condition_w1(final_cdf, active, physical_group_ids)
    )
    delta_energy = _masked_mean_or_graph_zero(
        learned_delta_energy, active, anchor=learned_delta_energy
    )

    sarn_ce_per_row = _normalized_posterior_ce_per_row(sarn_posterior, twohot)
    sarn_endpoint_ce_metric = _masked_mean_or_graph_zero(
        sarn_ce_per_row, active, anchor=sarn_posterior
    )
    sarn_cdf_per_row = torch.abs(sarn_cdf - target_cdf).mean(dim=1)
    sarn_endpoint_cdf_metric = _masked_mean_or_graph_zero(
        sarn_cdf_per_row / CDF_ERROR_NORMALIZATION,
        active,
        anchor=sarn_cdf,
    )

    weighted = {
        "final_read": LOSS_WEIGHTS.final_read * final_read,
        "dense_cdf_path": LOSS_WEIGHTS.dense_cdf_path * dense_cdf_path,
        "geometric_softplus_regret": (
            LOSS_WEIGHTS.geometric_softplus_regret * geometric_softplus_regret
        ),
        "absolute_final_cvar25": (
            LOSS_WEIGHTS.absolute_final_cvar25 * absolute_final_cvar25
        ),
        "cross_condition_w1": (
            LOSS_WEIGHTS.cross_condition_w1 * cross_condition_w1
        ),
        "learned_delta_energy": (
            LOSS_WEIGHTS.learned_delta_energy * delta_energy
        ),
    }
    total = sum(weighted.values(), final_posterior.sum() * 0.0)
    return total, {
        "total": total,
        "final_read": final_read,
        "dense_cdf_path": dense_cdf_path,
        "geometric_softplus_regret": geometric_softplus_regret,
        "absolute_final_cvar25": absolute_final_cvar25,
        "cross_condition_w1": cross_condition_w1,
        "learned_delta_energy": delta_energy,
        "sarn_endpoint_ce_metric": sarn_endpoint_ce_metric,
        "sarn_endpoint_cdf_metric": sarn_endpoint_cdf_metric,
        "weighted": weighted,
        "final_read_per_row": final_per_row,
        "final_posterior_ce_per_row": final_ce_per_row,
        "final_mean_smooth_l1_per_row": final_mean_per_row,
        "dense_cdf_path_per_row": dense_cdf_per_row,
        "geometric_regret_per_row": geometric_regret_per_row,
        "final_absolute_error_per_row": final_absolute_error,
        "cvar_tail_count": cvar_tail_count,
        "cross_condition_w1_per_group": cross_condition_w1_per_group,
        "cross_condition_active_group_count": int(
            cross_condition_group_active.sum().item()
        ),
        "coverage": {
            "read_valid": read_valid,
            "relation_available": relation_available,
            "active": active,
            "cross_condition_group_active": cross_condition_group_active,
        },
    }


def a15_fteb_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    physical_group_ids: Sequence[Any] | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate the fixed A15 objective in float32."""

    _require(isinstance(outputs, Mapping), "A15 outputs must be a mapping")
    _require(isinstance(targets, Mapping), "A15 targets must be a mapping")
    reference = _tensor(outputs.get("mean"), "mean")
    with torch.autocast(device_type=reference.device.type, enabled=False):
        return _a15_fteb_loss_fp32(outputs, targets, physical_group_ids)


__all__ = [
    "A15LossWeights",
    "A15TargetError",
    "CDF_ERROR_NORMALIZATION",
    "CVAR_TAIL_FRACTION",
    "GEOMETRIC_REGRET_TEMPERATURE",
    "LOSS_DESIGN_METADATA",
    "LOSS_WEIGHTS",
    "a15_fteb_loss",
    "build_a15_targets",
]
