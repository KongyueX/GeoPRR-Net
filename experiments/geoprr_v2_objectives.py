"""Source-supervised readout constraints for the U and Q fields."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _gt_bin_mass(target: torch.Tensor, weights: torch.Tensor, bins: int) -> torch.Tensor:
    positions = target.float().clamp(0, 1).flatten(1) * (bins - 1)
    lower = positions.floor().long()[:, None].expand_as(weights)
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = positions[:, None] - lower
    projected = weights.new_zeros((*weights.shape[:-1], bins))
    projected.scatter_add_(2, lower, weights * (1 - fraction))
    projected.scatter_add_(2, upper, weights * fraction)
    return projected


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
    projected = _gt_bin_mass(batch["field_target"], weights, field.shape[1])
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


def validate_state_composition_pairs(batch: Mapping[str, Any]) -> None:
    if any(kind != "state" for kind in batch["pair_kind"]):
        raise ValueError("state_composition_v1 requires state-intervention pairs")
    if not torch.equal(batch["roi_to_canonical"], batch["pair_roi_to_canonical"]):
        raise ValueError("state_composition_v1 requires identical camera and ROI maps")


def state_composition_losses(output: Mapping[str, torch.Tensor], partner: Mapping[str, torch.Tensor],
                             batch: Mapping[str, Any], *, support_threshold: float = 0.15
                             ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Match fixed-camera cross-state bin mass on every GT-valid edge."""
    validate_state_composition_pairs(batch)
    eligible = batch["pair_valid"].bool().reshape(-1)
    if not eligible.any():
        return {}, {"eligible": 0, "field_valid": 0, "readout_supervised": 0}
    field_a, field_b = output["field_probs"].float(), partner["field_probs"].float()
    mask_a, mask_b = output["support_probs"].float(), partner["support_probs"].float()
    gt_a, gt_b = batch["field_mask"].float(), batch["pair_field_mask"].float()
    overlap = gt_a * gt_b
    weights = mask_a * partner["point_probs"].float() * mask_b * overlap
    mass = weights.flatten(1).sum(1)
    with torch.autocast(device_type=field_a.device.type, enabled=False):
        numerator = torch.bmm(field_a.flatten(2), weights.flatten(2).transpose(1, 2)).squeeze(-1)
    gt_weights = gt_a * batch["pair_point_target"].float() * gt_b * overlap
    gt_mass = gt_weights.flatten(1).sum(1).detach()
    if (gt_mass[eligible] <= 0).any():
        raise ValueError("GT-valid state edges require positive GT composition mass")
    gt_numerator = _gt_bin_mass(batch["field_target"], gt_weights.flatten(1)[:, None], field_a.shape[1])[:, 0].detach()
    scale = gt_mass[eligible]
    cdf = (numerator[eligible].cumsum(1)[:, :-1] - gt_numerator[eligible].cumsum(1)[:, :-1]).abs().mean(1)
    centers = torch.linspace(0, 1, field_a.shape[1], device=field_a.device).view(1, -1, 1, 1)
    field_difference = ((field_a * centers).sum(1, keepdim=True) - (field_b * centers).sum(1, keepdim=True)).abs()
    field_weight = overlap * eligible[:, None, None, None]
    losses = {
        "state_reading_cdf": (cdf / scale).mean(),
        "state_mass_l1": ((mass[eligible] - scale).abs() / scale).mean(),
        "state_field_l1": (field_difference * field_weight).sum() / field_weight.sum().clamp_min(1),
        "state_support_l1": (mask_a[eligible] - mask_b[eligible]).abs().mean(),
    }
    return losses, {"eligible": int(eligible.sum()), "field_valid": int((eligible & (mass >= support_threshold)).sum()),
                    "readout_supervised": int(eligible.sum())}
