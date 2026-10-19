"""Source-supervised readout constraints for the U and Q fields."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def decoupled_readout_losses(output: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                            *, support_threshold: float = 0.15) -> dict[str, torch.Tensor]:
    """Supervise each field's readout while holding the other fields at GT."""
    field = output["field_probs"].float()
    point = output["point_probs"].float()
    mask = batch["field_mask"].float()
    gt_weights = batch["point_target"].float() * mask
    q_weights = point * mask
    weights = torch.stack((gt_weights.flatten(1), q_weights.flatten(1)), dim=1)
    mass = weights.sum(-1)

    # Project both weighted GT U distributions without a dense B,K,H,W label.
    positions = batch["field_target"].float().clamp(0, 1).flatten(1) * (field.shape[1] - 1)
    lower = positions.floor().long()[:, None].expand_as(weights)
    upper = (lower + 1).clamp(max=field.shape[1] - 1)
    fraction = positions[:, None] - lower
    projected = field.new_zeros((field.shape[0], 2, field.shape[1]))
    projected.scatter_add_(2, lower, weights * (1 - fraction))
    projected.scatter_add_(2, upper, weights * fraction)
    projected = projected / mass.clamp_min(1e-8)[..., None]
    target, q_distribution = projected[:, 0].detach(), projected[:, 1]

    with torch.autocast(device_type=field.device.type, enabled=False):
        u_distribution = torch.bmm(field.flatten(2), gt_weights.flatten(2).transpose(1, 2)).squeeze(-1)
    u_distribution = u_distribution / mass[:, :1].detach().clamp_min(1e-8)
    target_cdf = target.cumsum(1)[:, :-1]
    return {
        "q_physical_cdf": (q_distribution.cumsum(1)[:, :-1] - target_cdf).abs().mean(),
        "u_local_cdf": (u_distribution.cumsum(1)[:, :-1] - target_cdf).abs().mean(),
        "q_gt_support": F.relu(support_threshold - mass[:, 1]).mean(),
    }
