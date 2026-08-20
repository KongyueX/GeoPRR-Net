"""A15.1 objective: A15 with one-sided geometric excess-MAE regret.

All target construction, final-read, dense-path, CVaR, cross-condition W1,
delta-energy, and endpoint diagnostics are evaluated by the A15 objective.
A15.1 changes only the geometric-reference term: it penalizes the amount by
which the learned final absolute error exceeds the detached fixed-geometric
absolute error.  Improvements over the fixed endpoint have zero regret.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F

from experiments.a15_fteb_targets import (
    CDF_ERROR_NORMALIZATION,
    LOSS_WEIGHTS as A15_LOSS_WEIGHTS,
    a15_fteb_loss,
    build_a15_targets as _build_a15_targets,
)


@dataclass(frozen=True, slots=True)
class A151LossWeights:
    """A15 weights with only the geometric-regret coefficient replaced."""

    final_read: float = A15_LOSS_WEIGHTS.final_read
    dense_cdf_path: float = A15_LOSS_WEIGHTS.dense_cdf_path
    geometric_excess_mae: float = 1.0
    absolute_final_cvar25: float = A15_LOSS_WEIGHTS.absolute_final_cvar25
    cross_condition_w1: float = A15_LOSS_WEIGHTS.cross_condition_w1
    learned_delta_energy: float = A15_LOSS_WEIGHTS.learned_delta_energy

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


LOSS_WEIGHTS: Final[A151LossWeights] = A151LossWeights()
LOSS_DESIGN_METADATA: Final[dict[str, Any]] = {
    "variant": "A15.1_FTEB_one_sided_geometric_excess_MAE",
    "weights": LOSS_WEIGHTS.as_dict(),
    "unchanged_from_a15": (
        "targets_final_read_dense_cdf_path_absolute_final_cvar25_"
        "cross_condition_w1_learned_delta_energy_and_sarn_diagnostics"
    ),
    "geometric_excess_mae": (
        "active_mean_relu(abs(final-y)-detach(abs(geometric-y)))_divided_by_0.05"
    ),
    "geometric_excess_mae_weight_rationale": (
        "For the fixed B=24 probe batch, weight 1.0 gives one active-row hinge "
        "marginal 1/(24*0.05)=0.833333; the existing weight-0.25 CVaR25 over "
        "top 6 rows gives 0.25/(6*0.05)=0.833333. The coefficient is derived "
        "from this marginal balance, not from an evaluation sweep."
    ),
    "sarn_endpoint_diagnostics": "metrics_only_not_in_total",
    "path_field_energy": "not_in_total",
}


def build_a15_1_targets(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Reuse the unchanged A15 detached float32 targets."""

    return _build_a15_targets(batch)


def a15_1_fteb_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    physical_group_ids: Sequence[Any] | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate A15.1 while reusing every unchanged A15 loss component."""

    _, a15_components = a15_fteb_loss(outputs, targets, physical_group_ids)
    reference = outputs["mean"]

    with torch.autocast(device_type=reference.device.type, enabled=False):
        final_mean = reference.float()
        geometric_mean = outputs["geometric_base_mean"].detach().float()
        read = targets["read"].to(final_mean.device).float()
        active = a15_components["coverage"]["active"]

        final_absolute_error = torch.abs(final_mean - read)
        geometric_absolute_error = torch.abs(geometric_mean - read).detach()
        geometric_excess_mae_per_row = F.relu(
            final_absolute_error - geometric_absolute_error
        ) / CDF_ERROR_NORMALIZATION
        if bool(active.any()):
            geometric_excess_mae = geometric_excess_mae_per_row.masked_select(
                active
            ).mean()
        else:
            geometric_excess_mae = final_mean.sum() * 0.0

        a15_weighted = a15_components["weighted"]
        weighted = {
            "final_read": a15_weighted["final_read"],
            "dense_cdf_path": a15_weighted["dense_cdf_path"],
            "geometric_excess_mae": (
                LOSS_WEIGHTS.geometric_excess_mae * geometric_excess_mae
            ),
            "absolute_final_cvar25": a15_weighted["absolute_final_cvar25"],
            "cross_condition_w1": a15_weighted["cross_condition_w1"],
            "learned_delta_energy": a15_weighted["learned_delta_energy"],
        }
        total = sum(weighted.values(), final_mean.sum() * 0.0)

        components = dict(a15_components)
        components.pop("geometric_softplus_regret", None)
        components.pop("geometric_regret_per_row", None)
        components.update(
            {
                "total": total,
                "geometric_excess_mae": geometric_excess_mae,
                "geometric_excess_mae_per_row": geometric_excess_mae_per_row,
                "geometric_absolute_error_per_row": geometric_absolute_error,
                "weighted": weighted,
            }
        )
        return total, components


__all__ = [
    "A151LossWeights",
    "CDF_ERROR_NORMALIZATION",
    "LOSS_DESIGN_METADATA",
    "LOSS_WEIGHTS",
    "a15_1_fteb_loss",
    "build_a15_1_targets",
]
