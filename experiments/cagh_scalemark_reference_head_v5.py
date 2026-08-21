"""Reliability-aware multi-scale ScaleMark reference head for CAGH v5.

The PEPD feature extractor remains frozen.  This module is a drop-in upgrade
for the small ScaleMark reference head: ``forward(c2, c5)`` keeps every v3
output field, while the module names used by the staged v4 trainer
(``c2_projection``, ``c5_projection``, ``fusion``, ``shared_residual``,
``tick_head``, ``endpoint_conditioning`` and ``endpoint_head``) are preserved.

V5 adds three deliberately small mechanisms:

* a fine/coarse dense-tick pyramid instead of a single 64x64 tick branch;
* two independently parameterized endpoint posteriors whose geometric mean is
  used for the deployable ordered endpoints; and
* label-free reliability telemetry derived from endpoint agreement, a robust
  weighted circle fit to the dense ticks, and tick support along the ordered
  start-to-end arc.

The telemetry is continuous and is intended for a soft gate calibrated only on
public development data.  No hard deployment threshold is embedded here.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.cagh_scalemark_reference_head import (
    _ResidualBlock,
    gaussian_endpoint_targets,
    scalemark_reference_loss,
)
from experiments.cagh_scalemark_reference_head_v3 import dense_tick_loss


CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL = (
    "cagh_scalemark_reference_multiscale_reliability_v5"
)


class DenseTickReferenceOutputsV5(NamedTuple):
    # V3-compatible prefix.  Existing losses and inference code access these
    # fields by name, so adding telemetry after this prefix is non-breaking.
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

    # Multi-scale tick evidence.
    fine_tick_logits: torch.Tensor
    coarse_tick_logits: torch.Tensor
    tick_scale_disagreement: torch.Tensor

    # Independent endpoint evidence and agreement telemetry.
    direct_endpoint_logits: torch.Tensor
    auxiliary_endpoint_logits: torch.Tensor
    direct_endpoint_probability: torch.Tensor
    auxiliary_endpoint_probability: torch.Tensor
    direct_start_xy: torch.Tensor
    direct_end_xy: torch.Tensor
    auxiliary_start_xy: torch.Tensor
    auxiliary_end_xy: torch.Tensor
    endpoint_coordinate_disagreement: torch.Tensor
    endpoint_js_divergence: torch.Tensor
    endpoint_consistency_reliability: torch.Tensor

    # Dense-tick circle/arc telemetry.  All distances are fractions of the
    # 64x64 normalized crop and all reliability values are in [0, 1].
    tick_circle_center_xy: torch.Tensor
    tick_radius_mean: torch.Tensor
    tick_radius_cv: torch.Tensor
    endpoint_radius_relative_error: torch.Tensor
    radius_reliability: torch.Tensor
    ordered_arc_fraction: torch.Tensor
    ordered_arc_tick_mass: torch.Tensor
    endpoint_tick_support: torch.Tensor
    arc_length_reliability: torch.Tensor

    # Composite telemetry for a later public-development soft gate.
    gate_confidence: torch.Tensor
    uncertainty_score: torch.Tensor
    telemetry_valid: torch.Tensor


class _MultiScaleDenseTickHead(nn.Module):
    """Fuse local 64x64 evidence with native 8x8 PEPD context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fine = nn.Conv2d(channels, 1, 1)
        self.coarse = nn.Sequential(
            _ResidualBlock(channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(
        self, fine_features: torch.Tensor, coarse_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine = self.fine(fine_features)
        coarse_native = self.coarse(coarse_features)
        coarse = F.interpolate(
            coarse_native,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fuse(torch.cat((fine, coarse), dim=1))
        return fused, fine, coarse


class _ConsistentEndpointHead(nn.Module):
    """Two endpoint branches with different evidence paths."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.direct = nn.Conv2d(channels, 2, 1)
        self.tick_conditioned = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            _ResidualBlock(channels),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(
        self,
        endpoint_features: torch.Tensor,
        shared_features: torch.Tensor,
        tick_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direct = self.direct(endpoint_features)
        auxiliary = self.tick_conditioned(
            torch.cat((shared_features, tick_probability.to(shared_features.dtype)), dim=1)
        )
        return direct, auxiliary


class DenseTickConditionedReferenceHeadV5(nn.Module):
    """Multi-scale reference head with continuous uncertainty telemetry."""

    def __init__(self, *, hidden_channels: int = 64, residual_blocks: int = 2) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        if hidden < 16 or hidden % 8:
            raise ValueError("hidden_channels must be a multiple of 8 and at least 16")
        if int(residual_blocks) < 1:
            raise ValueError("V5 requires at least one shared residual block")
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
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden + 2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.shared_residual = nn.Sequential(
            *[_ResidualBlock(hidden) for _ in range(int(residual_blocks))]
        )
        # Keep this name so the existing staged trainer unfreezes the complete
        # fine/coarse/fusion tick pyramid during stage A.
        self.tick_head = _MultiScaleDenseTickHead(hidden)
        self.endpoint_conditioning = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            _ResidualBlock(hidden),
        )
        # Likewise, both endpoint branches are contained under endpoint_head
        # and are therefore trainable under the existing stage-B selector.
        self.endpoint_head = _ConsistentEndpointHead(hidden)

        axis = torch.linspace(0.0, 1.0, 64, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "_grid_xy",
            torch.stack((xx, yy), dim=2).reshape(-1, 2),
            persistent=False,
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _probability(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.flatten(2).float(), dim=2).reshape_as(logits)

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

    def _tick_circle(
        self,
        tick_probability: torch.Tensor,
        endpoints: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Fit a differentiable circle and measure ordered-arc support."""

        batch = int(tick_probability.shape[0])
        grid = self._grid_xy.to(device=tick_probability.device, dtype=torch.float32)
        # Fourth-power sharpening suppresses the low, spatially extensive
        # sigmoid background without a non-differentiable top-k threshold.
        weight = tick_probability[:, 0].float().flatten(1).clamp_min(1e-6).pow(4)
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)
        design = torch.cat(
            (grid, torch.ones((grid.shape[0], 1), device=grid.device)), dim=1
        )
        squared_radius = grid.square().sum(1)
        normal = torch.einsum("bn,ni,nj->bij", weight, design, design)
        ridge = torch.eye(3, device=grid.device, dtype=torch.float32)[None] * 1e-4
        right = -torch.einsum("bn,ni,n->bi", weight, design, squared_radius)
        solution = torch.linalg.solve(normal + ridge, right.unsqueeze(2)).squeeze(2)
        center = (-0.5 * solution[:, :2]).clamp(-0.25, 1.25)

        offset = grid[None] - center[:, None]
        radius = torch.linalg.vector_norm(offset, dim=2).clamp_min(1e-6)
        radius_mean = torch.sum(weight * radius, dim=1).clamp_min(1e-6)
        radius_std = torch.sqrt(
            torch.sum(weight * (radius - radius_mean[:, None]).square(), dim=1)
            .clamp_min(0.0)
            + 1e-12
        )
        radius_cv = radius_std / radius_mean
        endpoint_radius = torch.linalg.vector_norm(
            endpoints.float() - center[:, None], dim=2
        )
        endpoint_radius_error = torch.mean(
            torch.abs(endpoint_radius - radius_mean[:, None]) / radius_mean[:, None],
            dim=1,
        )
        radius_reliability = torch.exp(
            -2.5 * radius_cv.clamp_max(4.0)
            - 2.5 * endpoint_radius_error.clamp_max(4.0)
        ).clamp(0.0, 1.0)

        def angle(value: torch.Tensor) -> torch.Tensor:
            delta = value - center[:, None]
            return torch.remainder(
                torch.atan2(delta[..., 0], -delta[..., 1]) - math.pi,
                2.0 * math.pi,
            )

        endpoint_angle = angle(endpoints.float())
        ordered_arc = torch.remainder(
            endpoint_angle[:, 1] - endpoint_angle[:, 0], 2.0 * math.pi
        )
        pixel_angle = angle(grid[None].expand(batch, -1, -1))
        phase = torch.remainder(
            pixel_angle - endpoint_angle[:, 0, None], 2.0 * math.pi
        )
        # A soft three-degree boundary leaves this telemetry differentiable if
        # it is included as a low-weight public-training regularizer.
        inside = torch.sigmoid(
            (ordered_arc[:, None] - phase) / math.radians(3.0)
        )
        arc_tick_mass = torch.sum(weight * inside, dim=1).clamp(0.0, 1.0)

        # Exact align_corners=True bilinear sampling on the registered 64x64
        # grid.  Expressing it as triangular basis weights avoids CUDA's
        # non-deterministic grid_sample backward while retaining gradients to
        # both the tick map and the predicted endpoint coordinates.
        endpoint_grid = endpoints.float().clamp(0.0, 1.0)
        delta = torch.abs(
            endpoint_grid[:, :, None, :] - grid[None, None, :, :]
        )
        basis = (
            F.relu(1.0 - delta[..., 0] * 63.0)
            * F.relu(1.0 - delta[..., 1] * 63.0)
        )
        endpoint_support = torch.sum(
            tick_probability[:, 0].float().flatten(1)[:, None, :] * basis,
            dim=2,
        )
        peak = tick_probability.flatten(2).amax(2)[:, 0].clamp_min(1e-6)
        endpoint_support = (
            endpoint_support.mean(1) / peak
        ).clamp(0.0, 1.0)
        lower = torch.sigmoid(
            (ordered_arc - math.radians(10.0)) / math.radians(3.0)
        )
        upper = torch.sigmoid(
            (math.radians(350.0) - ordered_arc) / math.radians(3.0)
        )
        arc_reliability = (
            arc_tick_mass
            * torch.sqrt(endpoint_support.clamp_min(1e-6))
            * lower
            * upper
        ).clamp(0.0, 1.0)
        return (
            center,
            radius_mean,
            radius_cv,
            endpoint_radius_error,
            radius_reliability,
            ordered_arc / (2.0 * math.pi),
            arc_tick_mass,
            endpoint_support,
            arc_reliability,
        )

    def forward(self, c2: torch.Tensor, c5: torch.Tensor) -> DenseTickReferenceOutputsV5:
        if c2.ndim != 4 or c2.shape[1:] != (64, 64, 64):
            raise ValueError("c2 must have shape [B,64,64,64]")
        if c5.ndim != 4 or c5.shape[1:] != (512, 8, 8):
            raise ValueError("c5 must have shape [B,512,8,8]")
        local = self.c2_projection(c2.float())
        coarse_native = self.c5_projection(c5.float())
        context = F.interpolate(
            coarse_native,
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        coordinate = torch.linspace(-1.0, 1.0, 64, device=local.device)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        coords = torch.stack((xx, yy), dim=0)[None].expand(
            local.shape[0], -1, -1, -1
        )
        shared = self.shared_residual(
            self.fusion(torch.cat((local, context, coords), dim=1))
        )

        tick_logits, fine_tick_logits, coarse_tick_logits = self.tick_head(
            shared, coarse_native
        )
        tick_probability = torch.sigmoid(tick_logits.float())
        tick_density = tick_probability / tick_probability.sum(
            (2, 3), keepdim=True
        ).clamp_min(1e-12)
        tick_mass_error = torch.abs(tick_density.sum((2, 3)) - 1.0)
        tick_scale_disagreement = torch.mean(
            torch.abs(
                torch.sigmoid(fine_tick_logits.float())
                - torch.sigmoid(coarse_tick_logits.float())
            ),
            dim=(1, 2, 3),
        )

        endpoint_features = self.endpoint_conditioning(
            torch.cat((shared, tick_probability.to(shared.dtype)), dim=1)
        )
        direct_logits, auxiliary_logits = self.endpoint_head(
            endpoint_features, shared, tick_probability
        )
        # Averaging logits is a normalized geometric mean of the independent
        # posteriors and makes both branches trainable under the existing loss.
        endpoint_logits = 0.5 * (direct_logits + auxiliary_logits)
        probability = self._probability(endpoint_logits)
        direct_probability = self._probability(direct_logits)
        auxiliary_probability = self._probability(auxiliary_logits)
        xy = self._soft_coordinates(probability)
        direct_xy = self._soft_coordinates(direct_probability)
        auxiliary_xy = self._soft_coordinates(auxiliary_probability)
        flat = probability.flatten(2)
        entropy = -torch.sum(flat * torch.log(flat.clamp_min(1e-12)), dim=2) / math.log(
            float(flat.shape[2])
        )

        coordinate_disagreement = torch.linalg.vector_norm(
            direct_xy - auxiliary_xy, dim=2
        ).mean(1) / math.sqrt(2.0)
        direct_flat = direct_probability.flatten(2).clamp_min(1e-12)
        auxiliary_flat = auxiliary_probability.flatten(2).clamp_min(1e-12)
        mixture = 0.5 * (direct_flat + auxiliary_flat)
        js = 0.5 * (
            torch.sum(direct_flat * (torch.log(direct_flat) - torch.log(mixture)), dim=2)
            + torch.sum(
                auxiliary_flat * (torch.log(auxiliary_flat) - torch.log(mixture)),
                dim=2,
            )
        ).mean(1) / math.log(2.0)
        endpoint_consistency = (
            torch.exp(-8.0 * coordinate_disagreement.clamp_max(2.0))
            * (1.0 - js.clamp(0.0, 1.0))
            * (1.0 - entropy.mean(1).clamp(0.0, 1.0)).sqrt()
        ).clamp(0.0, 1.0)

        circle = self._tick_circle(tick_probability, xy)
        (
            circle_center,
            radius_mean,
            radius_cv,
            endpoint_radius_error,
            radius_reliability,
            arc_fraction,
            arc_tick_mass,
            endpoint_tick_support,
            arc_reliability,
        ) = circle
        tick_scale_reliability = torch.exp(
            -4.0 * tick_scale_disagreement.clamp_max(2.0)
        )
        components = torch.stack(
            (
                endpoint_consistency,
                radius_reliability,
                arc_reliability,
                tick_scale_reliability,
            ),
            dim=1,
        ).clamp_min(1e-6)
        gate_confidence = torch.exp(torch.mean(torch.log(components), dim=1)).clamp(
            0.0, 1.0
        )
        uncertainty = 1.0 - gate_confidence
        telemetry_valid = (
            torch.isfinite(components).all(1)
            & torch.isfinite(circle_center).all(1)
            & torch.isfinite(radius_mean)
            & (radius_mean > 1e-5)
        )

        return DenseTickReferenceOutputsV5(
            tick_logits=tick_logits,
            tick_probability=tick_probability,
            tick_density_mass_error=tick_mass_error,
            endpoint_logits=endpoint_logits,
            endpoint_probability=probability,
            start_xy=xy[:, 0],
            end_xy=xy[:, 1],
            endpoint_peak=flat.amax(2),
            endpoint_entropy=entropy,
            endpoint_separation=torch.linalg.vector_norm(xy[:, 1] - xy[:, 0], dim=1),
            posterior_mass_error=torch.abs(probability.sum((2, 3)) - 1.0),
            fine_tick_logits=fine_tick_logits,
            coarse_tick_logits=coarse_tick_logits,
            tick_scale_disagreement=tick_scale_disagreement,
            direct_endpoint_logits=direct_logits,
            auxiliary_endpoint_logits=auxiliary_logits,
            direct_endpoint_probability=direct_probability,
            auxiliary_endpoint_probability=auxiliary_probability,
            direct_start_xy=direct_xy[:, 0],
            direct_end_xy=direct_xy[:, 1],
            auxiliary_start_xy=auxiliary_xy[:, 0],
            auxiliary_end_xy=auxiliary_xy[:, 1],
            endpoint_coordinate_disagreement=coordinate_disagreement,
            endpoint_js_divergence=js,
            endpoint_consistency_reliability=endpoint_consistency,
            tick_circle_center_xy=circle_center,
            tick_radius_mean=radius_mean,
            tick_radius_cv=radius_cv,
            endpoint_radius_relative_error=endpoint_radius_error,
            radius_reliability=radius_reliability,
            ordered_arc_fraction=arc_fraction,
            ordered_arc_tick_mass=arc_tick_mass,
            endpoint_tick_support=endpoint_tick_support,
            arc_length_reliability=arc_reliability,
            gate_confidence=gate_confidence,
            uncertainty_score=uncertainty,
            telemetry_valid=telemetry_valid,
        )


# Alias and factory keep the one-line import swap small for existing trainers.
DenseTickConditionedReferenceHead = DenseTickConditionedReferenceHeadV5


def build_head() -> DenseTickConditionedReferenceHeadV5:
    return DenseTickConditionedReferenceHeadV5(hidden_channels=64, residual_blocks=2)


def _weighted(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / weight.sum().clamp_min(1e-8)


def _balanced_tick_branch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits[:, 0].float()
    positive = -(target * F.logsigmoid(logits)).sum((1, 2)) / target.sum(
        (1, 2)
    ).clamp_min(1e-6)
    negative_target = 1.0 - target
    negative = -(negative_target * F.logsigmoid(-logits)).sum(
        (1, 2)
    ) / negative_target.sum((1, 2)).clamp_min(1e-6)
    probability = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (probability * target).sum((1, 2)) + 1e-6) / (
        probability.sum((1, 2)) + target.sum((1, 2)) + 1e-6
    )
    return _weighted(0.5 * (positive + negative) + dice, weight)


def multiscale_dense_tick_loss(
    outputs: DenseTickReferenceOutputsV5,
    target: torch.Tensor,
    *,
    group_weight: torch.Tensor,
    branch_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V3 tick loss plus explicit supervision of both pyramid scales."""

    fused, components = dense_tick_loss(
        outputs, target, group_weight=group_weight
    )
    target = torch.as_tensor(
        target, device=outputs.tick_logits.device, dtype=torch.float32
    )
    weight = torch.as_tensor(
        group_weight, device=target.device, dtype=torch.float32
    ).reshape(-1)
    fine = _balanced_tick_branch_loss(outputs.fine_tick_logits, target, weight)
    coarse = _balanced_tick_branch_loss(outputs.coarse_tick_logits, target, weight)
    total = fused + float(branch_weight) * 0.5 * (fine + coarse)
    return total, {
        **components,
        "tick_fine_auxiliary_loss": fine.detach(),
        "tick_coarse_auxiliary_loss": coarse.detach(),
        "tick_scale_disagreement": _weighted(
            outputs.tick_scale_disagreement, weight
        ).detach(),
    }


def reference_consistency_loss(
    outputs: DenseTickReferenceOutputsV5,
    target_endpoint_xy: torch.Tensor,
    *,
    group_weight: torch.Tensor,
    auxiliary_weight: float = 0.20,
    consistency_weight: float = 0.10,
    radius_weight: float = 0.02,
    arc_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Endpoint supervision with small reliability-aware regularizers.

    The radius and arc terms are intentionally weak: they prevent geometrically
    implausible tail solutions without overruling the supervised endpoint and
    dense-tick evidence under mild perspective distortion.
    """

    base, components = scalemark_reference_loss(
        outputs, target_endpoint_xy, group_weight=group_weight
    )
    target_xy = torch.as_tensor(
        target_endpoint_xy,
        device=outputs.endpoint_logits.device,
        dtype=torch.float32,
    )
    target = gaussian_endpoint_targets(target_xy).to(outputs.endpoint_logits.device)
    weight = torch.as_tensor(
        group_weight, device=target.device, dtype=torch.float32
    ).reshape(-1)
    branch_rows = []
    for logits, start, end in (
        (
            outputs.direct_endpoint_logits,
            outputs.direct_start_xy,
            outputs.direct_end_xy,
        ),
        (
            outputs.auxiliary_endpoint_logits,
            outputs.auxiliary_start_xy,
            outputs.auxiliary_end_xy,
        ),
    ):
        soft_ce = -(
            target.flatten(2)
            * F.log_softmax(logits.flatten(2).float(), dim=2)
        ).sum(2).mean(1)
        coordinates = torch.stack((start, end), dim=1)
        coordinate = F.smooth_l1_loss(
            coordinates, target_xy, reduction="none", beta=1.0 / 64.0
        ).mean((1, 2))
        branch_rows.append(soft_ce + 8.0 * coordinate)
    auxiliary = _weighted(0.5 * (branch_rows[0] + branch_rows[1]), weight)
    consistency = _weighted(
        outputs.endpoint_coordinate_disagreement
        + outputs.endpoint_js_divergence,
        weight,
    )
    radius_penalty = _weighted(1.0 - outputs.radius_reliability, weight)
    arc_penalty = _weighted(1.0 - outputs.arc_length_reliability, weight)
    loss = (
        base
        + float(auxiliary_weight) * auxiliary
        + float(consistency_weight) * consistency
        + float(radius_weight) * radius_penalty
        + float(arc_weight) * arc_penalty
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("CAGH v5 reference consistency loss is non-finite")
    return loss, {
        **components,
        "endpoint_auxiliary_supervision": auxiliary.detach(),
        "endpoint_branch_consistency": consistency.detach(),
        "radius_reliability_penalty": radius_penalty.detach(),
        "arc_length_reliability_penalty": arc_penalty.detach(),
        "gate_confidence_mean": _weighted(outputs.gate_confidence, weight).detach(),
    }


def reference_geometry(
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    pivot: torch.Tensor,
    inverse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V4-compatible reference geometry without importing a training entry."""

    start_vector = torch.einsum(
        "bij,bj->bi", inverse.float(), start_xy.float() - pivot.float()
    )
    end_vector = torch.einsum(
        "bij,bj->bi", inverse.float(), end_xy.float() - pivot.float()
    )
    radii = torch.stack(
        (
            torch.linalg.vector_norm(start_vector, dim=1),
            torch.linalg.vector_norm(end_vector, dim=1),
        ),
        dim=1,
    )
    start = torch.remainder(
        torch.atan2(start_vector[:, 0], -start_vector[:, 1]) - math.pi,
        2.0 * math.pi,
    )
    end = torch.remainder(
        torch.atan2(end_vector[:, 0], -end_vector[:, 1]) - math.pi,
        2.0 * math.pi,
    )
    arc = torch.remainder(end - start, 2.0 * math.pi)
    valid = (
        (radii > 0.02).all(1)
        & (arc > math.radians(10.0))
        & (arc < math.radians(350.0))
        & torch.isfinite(arc)
    )
    return start, arc, radii, valid


def geometry_loss(
    outputs: DenseTickReferenceOutputsV5,
    endpoints: torch.Tensor,
    gt_start: torch.Tensor,
    gt_range: torch.Tensor,
    pivot: torch.Tensor,
    inverse: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for the staged v4 ``geometry_loss`` function."""

    endpoint_loss, _ = reference_consistency_loss(
        outputs, endpoints, group_weight=weight
    )
    start, arc, radii, _ = reference_geometry(
        outputs.start_xy, outputs.end_xy, pivot, inverse
    )
    angle = 1.0 - torch.cos(start - gt_start) + F.smooth_l1_loss(
        arc / (2.0 * math.pi),
        gt_range / (2.0 * math.pi),
        reduction="none",
        beta=0.02,
    )
    arc_margin = (
        F.relu(math.radians(10.0) - arc) / math.radians(10.0)
    ).square() + (
        F.relu(arc - math.radians(350.0)) / math.radians(10.0)
    ).square()
    radius_margin = (F.relu(0.02 - radii) / 0.02).square().mean(1)
    return (
        endpoint_loss
        + 0.25 * _weighted(angle, weight)
        + 0.5 * _weighted(arc_margin + radius_margin, weight)
    )


__all__ = [
    "CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL",
    "DenseTickConditionedReferenceHead",
    "DenseTickConditionedReferenceHeadV5",
    "DenseTickReferenceOutputsV5",
    "build_head",
    "geometry_loss",
    "multiscale_dense_tick_loss",
    "reference_consistency_loss",
    "reference_geometry",
]
