"""Small ordered ScaleMark endpoint head for frozen PEPD multiscale features."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class ScaleMarkReferenceOutputs(NamedTuple):
    endpoint_logits: torch.Tensor
    endpoint_probability: torch.Tensor
    start_xy: torch.Tensor
    end_xy: torch.Tensor
    endpoint_peak: torch.Tensor
    endpoint_entropy: torch.Tensor
    endpoint_separation: torch.Tensor
    posterior_mass_error: torch.Tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value + self.layers(value))


class ScaleMarkReferenceHead(nn.Module):
    """Predict ordered start/end ScaleMark points on PEPD's 64×64 crop grid."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        coordconv: bool = False,
        residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        if hidden < 16 or hidden % 8:
            raise ValueError("hidden_channels must be a multiple of 8 and at least 16")
        self.c2_projection = nn.Sequential(
            nn.Conv2d(64, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.c5_projection = nn.Sequential(
            nn.Conv2d(512, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.coordconv = bool(coordconv)
        fusion_inputs = 2 * hidden + (2 if self.coordconv else 0)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_inputs, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            *[_ResidualBlock(hidden) for _ in range(int(residual_blocks))]
        )
        self.output = nn.Conv2d(hidden, 2, 1)
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
        batch, endpoints, height, width = probability.shape
        y = torch.linspace(0.0, 1.0, height, device=probability.device)
        x = torch.linspace(0.0, 1.0, width, device=probability.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (
                torch.sum(probability * xx[None, None], dim=(2, 3)),
                torch.sum(probability * yy[None, None], dim=(2, 3)),
            ),
            dim=2,
        ).reshape(batch, endpoints, 2)

    def forward(self, c2: torch.Tensor, c5: torch.Tensor) -> ScaleMarkReferenceOutputs:
        if c2.ndim != 4 or c2.shape[1:] != (64, 64, 64):
            raise ValueError("c2 must have shape [B,64,64,64]")
        if c5.ndim != 4 or c5.shape[1:] != (512, 8, 8):
            raise ValueError("c5 must have shape [B,512,8,8]")
        local = self.c2_projection(c2.float())
        global_context = F.interpolate(
            self.c5_projection(c5.float()),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        fused = torch.cat((local, global_context), dim=1)
        if self.coordconv:
            coordinate = torch.linspace(-1.0, 1.0, 64, device=fused.device)
            yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
            coords = torch.stack((xx, yy), dim=0)[None].expand(fused.shape[0], -1, -1, -1)
            fused = torch.cat((fused, coords), dim=1)
        logits = self.output(self.residual(self.fusion(fused)))
        probability = torch.softmax(logits.flatten(2).float(), dim=2).reshape_as(logits)
        xy = self._soft_coordinates(probability)
        peak = probability.flatten(2).amax(2)
        entropy = -torch.sum(
            probability.flatten(2)
            * torch.log(probability.flatten(2).clamp_min(1e-12)),
            dim=2,
        ) / torch.log(torch.tensor(4096.0, device=probability.device))
        mass_error = torch.abs(probability.sum((2, 3)) - 1.0)
        return ScaleMarkReferenceOutputs(
            endpoint_logits=logits,
            endpoint_probability=probability,
            start_xy=xy[:, 0],
            end_xy=xy[:, 1],
            endpoint_peak=peak,
            endpoint_entropy=entropy,
            endpoint_separation=torch.linalg.vector_norm(xy[:, 1] - xy[:, 0], dim=1),
            posterior_mass_error=mass_error,
        )


def gaussian_endpoint_targets(
    endpoint_xy: torch.Tensor, *, size: int = 64, sigma_pixels: float = 1.5
) -> torch.Tensor:
    xy = torch.as_tensor(endpoint_xy, dtype=torch.float32)
    if xy.ndim != 3 or xy.shape[1:] != (2, 2):
        raise ValueError("endpoint_xy must have shape [B,2,2]")
    if int(size) < 16 or float(sigma_pixels) <= 0.0:
        raise ValueError("invalid endpoint target grid/sigma")
    coordinate = torch.arange(int(size), device=xy.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    center = xy.clamp(0.0, 1.0) * float(size - 1)
    squared = (
        (xx[None, None] - center[:, :, 0, None, None]).square()
        + (yy[None, None] - center[:, :, 1, None, None]).square()
    )
    target = torch.exp(-0.5 * squared / float(sigma_pixels) ** 2)
    return target / target.sum((2, 3), keepdim=True).clamp_min(1e-12)


def scalemark_reference_loss(
    outputs: ScaleMarkReferenceOutputs,
    target_endpoint_xy: torch.Tensor,
    *,
    group_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_xy = torch.as_tensor(
        target_endpoint_xy,
        device=outputs.endpoint_logits.device,
        dtype=torch.float32,
    )
    target = gaussian_endpoint_targets(target_xy).to(outputs.endpoint_logits.device)
    log_probability = F.log_softmax(outputs.endpoint_logits.flatten(2).float(), dim=2)
    heatmap_row = -(target.flatten(2) * log_probability).sum(2).mean(1)
    predicted_xy = torch.stack((outputs.start_xy, outputs.end_xy), dim=1)
    coordinate_row = F.smooth_l1_loss(
        predicted_xy, target_xy, reduction="none", beta=1.0 / 64.0
    ).mean((1, 2))
    target_separation = torch.linalg.vector_norm(target_xy[:, 1] - target_xy[:, 0], dim=1)
    separation_row = F.smooth_l1_loss(
        outputs.endpoint_separation,
        target_separation,
        reduction="none",
        beta=1.0 / 64.0,
    )
    if group_weight is None:
        weight = torch.ones_like(heatmap_row)
    else:
        weight = torch.as_tensor(
            group_weight, device=heatmap_row.device, dtype=torch.float32
        ).reshape(-1)
        if weight.shape != heatmap_row.shape or bool((weight <= 0.0).any()):
            raise ValueError("group_weight must be positive with shape [B]")
    def weighted(value: torch.Tensor) -> torch.Tensor:
        return torch.sum(value * weight) / weight.sum().clamp_min(1e-8)
    heatmap_loss = weighted(heatmap_row)
    coordinate_loss = weighted(coordinate_row)
    separation_loss = weighted(separation_row)
    loss = heatmap_loss + 8.0 * coordinate_loss + 2.0 * separation_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("ScaleMark reference loss is non-finite")
    return loss, {
        "endpoint_soft_ce": heatmap_loss.detach(),
        "endpoint_coordinate_smooth_l1": coordinate_loss.detach(),
        "endpoint_separation_smooth_l1": separation_loss.detach(),
    }
