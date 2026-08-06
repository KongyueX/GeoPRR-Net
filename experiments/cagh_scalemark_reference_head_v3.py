"""Dense-tick-conditioned ScaleMark reference head used by the frozen v3 probe."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.cagh_scalemark_reference_head import _ResidualBlock


class DenseTickReferenceOutputs(NamedTuple):
    tick_logits: torch.Tensor
    tick_probability: torch.Tensor
    tick_density_mass_error: torch.Tensor
    endpoint_logits: torch.Tensor
    endpoint_probability: torch.Tensor
    start_xy: torch.Tensor
    end_xy: torch.Tensor
    endpoint_peak: torch.Tensor
    endpoint_entropy: torch.Tensor
    endpoint_separation: torch.Tensor
    posterior_mass_error: torch.Tensor


class DenseTickConditionedReferenceHead(nn.Module):
    """Predict the complete ordered scale-mark support before its endpoints."""

    def __init__(self, *, hidden_channels: int = 64, residual_blocks: int = 2) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        if hidden < 16 or hidden % 8:
            raise ValueError("hidden_channels must be a multiple of 8 and at least 16")
        self.c2_projection = nn.Sequential(
            nn.Conv2d(64, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.GELU()
        )
        self.c5_projection = nn.Sequential(
            nn.Conv2d(512, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden + 2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.shared_residual = nn.Sequential(
            *[_ResidualBlock(hidden) for _ in range(int(residual_blocks))]
        )
        self.tick_head = nn.Conv2d(hidden, 1, 1)
        self.endpoint_conditioning = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            _ResidualBlock(hidden),
        )
        self.endpoint_head = nn.Conv2d(hidden, 2, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _soft_coordinates(probability: torch.Tensor) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, probability.shape[2], device=probability.device)
        x = torch.linspace(0.0, 1.0, probability.shape[3], device=probability.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (
                torch.sum(probability * xx[None, None], dim=(2, 3)),
                torch.sum(probability * yy[None, None], dim=(2, 3)),
            ),
            dim=2,
        )

    def forward(self, c2: torch.Tensor, c5: torch.Tensor) -> DenseTickReferenceOutputs:
        if c2.ndim != 4 or c2.shape[1:] != (64, 64, 64):
            raise ValueError("c2 must have shape [B,64,64,64]")
        if c5.ndim != 4 or c5.shape[1:] != (512, 8, 8):
            raise ValueError("c5 must have shape [B,512,8,8]")
        local = self.c2_projection(c2.float())
        context = F.interpolate(
            self.c5_projection(c5.float()), size=(64, 64), mode="bilinear", align_corners=False
        )
        coordinate = torch.linspace(-1.0, 1.0, 64, device=local.device)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        coords = torch.stack((xx, yy), dim=0)[None].expand(local.shape[0], -1, -1, -1)
        shared = self.shared_residual(self.fusion(torch.cat((local, context, coords), dim=1)))
        tick_logits = self.tick_head(shared)
        tick_probability = torch.sigmoid(tick_logits.float())
        tick_density = tick_probability / tick_probability.sum((2, 3), keepdim=True).clamp_min(1e-12)
        tick_mass_error = torch.abs(tick_density.sum((2, 3)) - 1.0)
        endpoint_features = self.endpoint_conditioning(
            torch.cat((shared, tick_probability.to(shared.dtype)), dim=1)
        )
        endpoint_logits = self.endpoint_head(endpoint_features)
        endpoint_probability = torch.softmax(endpoint_logits.flatten(2).float(), dim=2).reshape_as(endpoint_logits)
        xy = self._soft_coordinates(endpoint_probability)
        flat = endpoint_probability.flatten(2)
        entropy = -torch.sum(flat * torch.log(flat.clamp_min(1e-12)), dim=2) / torch.log(
            torch.tensor(float(flat.shape[2]), device=flat.device)
        )
        return DenseTickReferenceOutputs(
            tick_logits=tick_logits,
            tick_probability=tick_probability,
            tick_density_mass_error=tick_mass_error,
            endpoint_logits=endpoint_logits,
            endpoint_probability=endpoint_probability,
            start_xy=xy[:, 0],
            end_xy=xy[:, 1],
            endpoint_peak=flat.amax(2),
            endpoint_entropy=entropy,
            endpoint_separation=torch.linalg.vector_norm(xy[:, 1] - xy[:, 0], dim=1),
            posterior_mass_error=torch.abs(endpoint_probability.sum((2, 3)) - 1.0),
        )


def dense_tick_heatmaps(
    tick_xy: torch.Tensor,
    tick_valid: torch.Tensor,
    *,
    size: int = 64,
    sigma_pixels: float = 1.25,
) -> torch.Tensor:
    """Create a likelihood union with one preserved unit peak per annotated tick."""
    xy = torch.as_tensor(tick_xy, dtype=torch.float32)
    valid = torch.as_tensor(tick_valid, device=xy.device, dtype=torch.bool)
    if xy.ndim != 3 or xy.shape[2] != 2 or valid.shape != xy.shape[:2]:
        raise ValueError("tick_xy/tick_valid must have shapes [B,M,2]/[B,M]")
    if not bool(valid.any(dim=1).all()) or int(size) < 16 or float(sigma_pixels) <= 0.0:
        raise ValueError("every sample needs at least one tick and a valid grid/sigma")
    coordinate = torch.arange(int(size), device=xy.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    center = xy.clamp(0.0, 1.0) * float(size - 1)
    squared = (
        (xx[None, None] - center[:, :, 0, None, None]).square()
        + (yy[None, None] - center[:, :, 1, None, None]).square()
    )
    gaussian = torch.exp(-0.5 * squared / float(sigma_pixels) ** 2)
    gaussian = torch.where(valid[:, :, None, None], gaussian, torch.zeros_like(gaussian))
    return gaussian.amax(dim=1)


def dense_tick_loss(
    outputs: DenseTickReferenceOutputs,
    target: torch.Tensor,
    *,
    group_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = torch.as_tensor(target, device=outputs.tick_logits.device, dtype=torch.float32)
    if target.shape != outputs.tick_logits[:, 0].shape or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("tick target must match [B,64,64] and remain in [0,1]")
    weight = torch.as_tensor(group_weight, device=target.device, dtype=torch.float32).reshape(-1)
    if weight.shape != target.shape[:1] or bool((weight <= 0).any()):
        raise ValueError("group_weight must be positive with shape [B]")
    logits = outputs.tick_logits[:, 0].float()
    positive = -(target * F.logsigmoid(logits)).sum((1, 2)) / target.sum((1, 2)).clamp_min(1e-6)
    negative_target = 1.0 - target
    negative = -(negative_target * F.logsigmoid(-logits)).sum((1, 2)) / negative_target.sum((1, 2)).clamp_min(1e-6)
    bce_row = 0.5 * (positive + negative)
    probability = outputs.tick_probability[:, 0]
    dice_row = 1.0 - (2.0 * (probability * target).sum((1, 2)) + 1e-6) / (
        probability.sum((1, 2)) + target.sum((1, 2)) + 1e-6
    )
    weighted = lambda value: torch.sum(value * weight) / weight.sum().clamp_min(1e-8)
    bce, dice = weighted(bce_row), weighted(dice_row)
    return bce + dice, {"tick_balanced_bce": bce.detach(), "tick_dice_loss": dice.detach()}
