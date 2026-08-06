"""Minimal shared-spatial mask--geometry posterior head for UHPF probing."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class SharedMaskGeometryOutputs(NamedTuple):
    mask_logits: torch.Tensor
    mask_probability: torch.Tensor
    pivot_xy: torch.Tensor
    geometry_axis: torch.Tensor
    geometry_axis_score: torch.Tensor
    geometry_support: torch.Tensor
    geometry_log_probability: torch.Tensor
    geometry_expected_progress: torch.Tensor
    visual_log_probability: torch.Tensor
    visual_expected_progress: torch.Tensor
    final_log_probability: torch.Tensor
    final_expected_progress: torch.Tensor
    geometry_weight: torch.Tensor
    geometry_available: torch.Tensor
    valid: torch.Tensor
    posterior_mass_error: torch.Tensor


class SharedMaskGeometryPosteriorHead(nn.Module):
    """Turn frozen PEPD pivot-decoder features into a physical posterior.

    Only this small head is trained.  The pointer axis is obtained from soft
    mask moments around PEPD's frozen pivot, then oriented by the frozen visual
    direction.  Reference geometry maps that axis to the same progress grid as
    the visual posterior.
    """

    def __init__(self, *, progress_bins: int = 72) -> None:
        super().__init__()
        if int(progress_bins) < 16:
            raise ValueError("progress_bins must be at least 16")
        self.progress_bins = int(progress_bins)
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self.geometry_concentration = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.fusion_logit = nn.Parameter(torch.tensor(-1.38629436))
        self.register_buffer(
            "progress_grid", torch.linspace(0.0, 1.0, self.progress_bins)
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.mask_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.geometry_concentration.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.geometry_concentration[-1].bias, 2.0)

    @staticmethod
    def trainable_parameter_count() -> int:
        # Kept as an auditable ceiling rather than a hard-coded architecture
        # claim in the paper summary.
        model = SharedMaskGeometryPosteriorHead()
        return sum(parameter.numel() for parameter in model.parameters())

    @staticmethod
    def _coordinates(
        size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinate = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        return xx, yy

    def geometry_from_mask(
        self,
        mask_probability: torch.Tensor,
        pivot_logits: torch.Tensor,
        visual_direction: torch.Tensor,
        reference_start_angle: torch.Tensor,
        reference_range_angle: torch.Tensor,
        crop_affine: torch.Tensor,
        reference_available: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if mask_probability.ndim != 4 or mask_probability.shape[1] != 1:
            raise ValueError("mask_probability must have shape [B,1,H,W]")
        batch, _, height, width = mask_probability.shape
        if height != width or pivot_logits.shape != mask_probability.shape:
            raise ValueError("mask and pivot logits must share a square grid")
        if visual_direction.shape != (batch, 2):
            raise ValueError("visual_direction must have shape [B,2]")
        dtype, device = mask_probability.dtype, mask_probability.device
        xx, yy = self._coordinates(height, device=device, dtype=dtype)
        pivot_probability = torch.softmax(pivot_logits.flatten(1).float(), dim=1)
        pivot_x = torch.sum(pivot_probability * xx.flatten()[None, :], dim=1)
        pivot_y = torch.sum(pivot_probability * yy.flatten()[None, :], dim=1)
        pivot = torch.stack((pivot_x, pivot_y), dim=1).to(dtype)

        weight = mask_probability[:, 0].clamp(0.0, 1.0)
        weight_sum = weight.sum(dim=(1, 2)).clamp_min(1e-8)
        dx = xx[None, :, :] - pivot_x[:, None, None]
        dy = yy[None, :, :] - pivot_y[:, None, None]
        cxx = (weight * dx.square()).sum((1, 2)) / weight_sum
        cyy = (weight * dy.square()).sum((1, 2)) / weight_sum
        cxy = (weight * dx * dy).sum((1, 2)) / weight_sum
        angle = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy)
        axis = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
        visual = F.normalize(visual_direction.float(), dim=1, eps=1e-8)
        orientation = torch.where(
            torch.sum(axis * visual.detach(), dim=1, keepdim=True) >= 0.0,
            torch.ones((batch, 1), device=device, dtype=axis.dtype),
            -torch.ones((batch, 1), device=device, dtype=axis.dtype),
        )
        axis = F.normalize(axis * orientation, dim=1, eps=1e-8)
        trace = (cxx + cyy).clamp_min(1e-8)
        gap = torch.sqrt((cxx - cyy).square() + 4.0 * cxy.square())
        axis_score = (gap / trace).clamp(0.0, 1.0)
        support = (weight_sum / float(height * width)).clamp(0.0, 1.0)
        entropy = -(
            weight.clamp(1e-6, 1.0 - 1e-6)
            * torch.log(weight.clamp(1e-6, 1.0 - 1e-6))
            + (1.0 - weight).clamp(1e-6, 1.0 - 1e-6)
            * torch.log((1.0 - weight).clamp(1e-6, 1.0 - 1e-6))
        ).mean((1, 2)) / math.log(2.0)
        quality = torch.stack((axis_score, support, entropy), dim=1)
        concentration = 1.0 + 99.0 * torch.sigmoid(
            self.geometry_concentration(quality.float())[:, 0]
        )

        start = torch.as_tensor(
            reference_start_angle, device=device, dtype=torch.float32
        ).reshape(batch)
        angle_range = torch.as_tensor(
            reference_range_angle, device=device, dtype=torch.float32
        ).reshape(batch)
        affine = torch.as_tensor(crop_affine, device=device, dtype=torch.float32)
        if affine.shape != (batch, 2, 3):
            raise ValueError("crop_affine must have shape [B,2,3]")
        available = (
            torch.as_tensor(reference_available, device=device).bool().reshape(batch)
            & torch.isfinite(start)
            & torch.isfinite(angle_range)
            & (angle_range.abs() > 1e-8)
            & torch.isfinite(axis).all(1)
            & (support > 1e-5)
        )
        safe_start = torch.where(torch.isfinite(start), start, torch.zeros_like(start))
        safe_range = torch.where(
            torch.isfinite(angle_range), angle_range, torch.zeros_like(angle_range)
        )
        theta = safe_start[:, None] + safe_range[:, None] * self.progress_grid[
            None, :
        ].to(device=device)
        original = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=2)
        crop = torch.einsum("bij,bkj->bki", affine[:, :, :2], original)
        crop = F.normalize(crop, dim=2, eps=1e-8)
        similarity = torch.sum(crop * axis[:, None, :], dim=2)
        geometry_log = F.log_softmax(concentration[:, None] * similarity, dim=1)
        uniform = torch.full_like(
            geometry_log, -math.log(float(self.progress_bins))
        )
        geometry_log = torch.where(available[:, None], geometry_log, uniform)
        geometry_expected = torch.sum(
            torch.exp(geometry_log)
            * self.progress_grid[None, :].to(device=device),
            dim=1,
        )
        return (
            geometry_log,
            geometry_expected,
            pivot,
            axis,
            axis_score,
            support,
            available,
        )

    def forward(
        self,
        pivot_decoder_features: torch.Tensor,
        pivot_logits: torch.Tensor,
        visual_direction: torch.Tensor,
        visual_log_probability: torch.Tensor,
        reference_start_angle: torch.Tensor,
        reference_range_angle: torch.Tensor,
        crop_affine: torch.Tensor,
        reference_available: torch.Tensor,
        *,
        mask_probability_override: torch.Tensor | None = None,
    ) -> SharedMaskGeometryOutputs:
        if pivot_decoder_features.ndim != 4 or pivot_decoder_features.shape[1] != 64:
            raise ValueError("pivot_decoder_features must have shape [B,64,H,W]")
        mask_logits = self.mask_head(pivot_decoder_features.float())
        mask_probability = torch.sigmoid(mask_logits)
        geometry_mask = (
            mask_probability
            if mask_probability_override is None
            else torch.as_tensor(
                mask_probability_override,
                device=mask_probability.device,
                dtype=mask_probability.dtype,
            )
        )
        (
            geometry_log,
            geometry_expected,
            pivot,
            axis,
            axis_score,
            support,
            geometry_available,
        ) = self.geometry_from_mask(
            geometry_mask,
            pivot_logits,
            visual_direction,
            reference_start_angle,
            reference_range_angle,
            crop_affine,
            reference_available,
        )
        visual_log = torch.as_tensor(
            visual_log_probability,
            device=mask_probability.device,
            dtype=torch.float32,
        )
        if visual_log.shape != geometry_log.shape:
            raise ValueError("visual and geometry posteriors must share shape")
        visual_probability = torch.exp(visual_log)
        geometry_probability = torch.exp(geometry_log)
        global_weight = 0.5 * torch.sigmoid(self.fusion_logit)
        effective_weight = torch.where(
            geometry_available,
            global_weight.expand_as(geometry_expected),
            torch.zeros_like(geometry_expected),
        )
        final_probability = (
            (1.0 - effective_weight[:, None]) * visual_probability
            + effective_weight[:, None] * geometry_probability
        )
        final_probability = final_probability / final_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        final_log = torch.log(final_probability.clamp_min(1e-12))
        grid = self.progress_grid.to(device=final_log.device)
        visual_expected = torch.sum(visual_probability * grid[None, :], dim=1)
        final_expected = torch.sum(final_probability * grid[None, :], dim=1)
        mass_error = torch.abs(final_probability.sum(1) - 1.0)
        valid = torch.as_tensor(reference_available, device=final_log.device).bool()
        return SharedMaskGeometryOutputs(
            mask_logits=mask_logits,
            mask_probability=mask_probability,
            pivot_xy=pivot,
            geometry_axis=axis,
            geometry_axis_score=axis_score,
            geometry_support=support,
            geometry_log_probability=geometry_log,
            geometry_expected_progress=geometry_expected,
            visual_log_probability=visual_log,
            visual_expected_progress=visual_expected,
            final_log_probability=final_log,
            final_expected_progress=final_expected,
            geometry_weight=effective_weight,
            geometry_available=geometry_available,
            valid=valid,
            posterior_mass_error=mass_error,
        )
