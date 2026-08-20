"""Solver-anchored reliability-gated CAGH-V5 corrective model.

This versioned model leaves the frozen V2 implementation untouched.  One
encoder pass produces the PEPD, reference, keypoint-solver, and Base-style mask
evidence.  The center-referenced keypoint solver posterior is the anchor.  PEPD
probabilistic direction and mask-moment geometry can only enter through two
reliability-gated relative-RMS residuals whose bounded signed scalar gains are
initialized to exactly zero.

The runtime callback accepts pixels and preprocessing transforms only.  It
does not accept ground truth, numeric ranges, ScaleMark labels, readings,
errors, sample identities, or bounding boxes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.cagh_net import CAGHMultiscaleFeatures, CAGHOutputs
from experiments.cagh_scalemark_reference_head_v5 import DenseTickReferenceOutputsV5
from experiments.cagh_v5_unified_model import (
    CAGHV5UnifiedModel,
    projective_reference_geometry,
)
from experiments.geopepd_progress import _affine_linear
from experiments.probabilistic_pivot_direction import ProbabilisticDirectionPrediction


PROTOCOL = "cagh_v5_solver_gated_residual_v1"
FULL_GATED = "full_gated"
NO_PEPD_RESIDUAL = "no_pepd_residual"
NO_MASK_RESIDUAL = "no_mask_residual"
SOLVER_CORE = "solver_core"
ARM_NAMES = (FULL_GATED, NO_PEPD_RESIDUAL, NO_MASK_RESIDUAL, SOLVER_CORE)


class SolverGatedResidualOutputs(NamedTuple):
    arm: str
    progress_log_probability: torch.Tensor
    expected_progress: torch.Tensor
    valid: torch.Tensor
    pointer_valid: torch.Tensor
    reference_valid: torch.Tensor
    geometry_solver_valid: torch.Tensor
    fusion_solver_valid: torch.Tensor
    features: CAGHMultiscaleFeatures
    pepd: ProbabilisticDirectionPrediction
    reference: DenseTickReferenceOutputsV5
    core: CAGHOutputs
    reference_start_angle: torch.Tensor
    reference_range_angle: torch.Tensor
    reference_radii: torch.Tensor
    core_angle_evidence: torch.Tensor
    pepd_angle_evidence: torch.Tensor
    mask_angle_evidence: torch.Tensor
    pepd_reliability_gate: torch.Tensor
    mask_reliability_gate: torch.Tensor
    pepd_residual_scale: torch.Tensor
    mask_residual_scale: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _center_and_scale(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(value, dtype=torch.float32)
    _require(value.ndim == 2 and value.shape[1] >= 16, "evidence must be [B,K]")
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    centered = value - value.mean(dim=1, keepdim=True)
    scale = torch.sqrt(centered.square().mean(dim=1, keepdim=True) + 1e-6)
    return centered / scale, scale


def _progress_directions(
    start: torch.Tensor,
    arc: torch.Tensor,
    crop_affine: torch.Tensor,
    *,
    bins: int,
) -> torch.Tensor:
    """Reproduce the frozen CAGH crop-space direction convention exactly."""

    start_value = torch.as_tensor(start, dtype=torch.float32)
    arc_value = torch.as_tensor(arc, device=start_value.device, dtype=torch.float32)
    _require(start_value.ndim == 1 and arc_value.shape == start_value.shape,
             "reference geometry must be [B]")
    affine = _affine_linear(
        crop_affine,
        batch_size=start_value.shape[0],
        device=start_value.device,
        dtype=torch.float32,
    )
    grid = torch.linspace(0.0, 1.0, int(bins), device=start_value.device)
    safe_start = torch.nan_to_num(start_value, nan=0.0, posinf=0.0, neginf=0.0)
    safe_arc = torch.nan_to_num(arc_value, nan=0.0, posinf=0.0, neginf=0.0)
    safe_affine = torch.nan_to_num(affine, nan=0.0, posinf=0.0, neginf=0.0)
    theta = safe_start[:, None] + safe_arc[:, None] * grid[None]
    original = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=2)
    return F.normalize(
        torch.einsum("bij,bkj->bki", safe_affine, original),
        dim=2,
        eps=1e-8,
    )


def _pepd_quality(pepd: ProbabilisticDirectionPrediction) -> torch.Tensor:
    entropy = torch.nan_to_num(pepd.angle_entropy.float(), nan=1.0)
    entropy_confidence = (1.0 - entropy).clamp(0.0, 1.0)
    resultant = torch.nan_to_num(
        pepd.bin_resultant_length.float(), nan=0.0
    ).clamp(0.0, 1.0)
    log_variance = torch.nan_to_num(
        pepd.log_variance.float(), nan=2.0, posinf=2.0, neginf=-9.0
    ).clamp(-9.0, 2.0)
    variance_confidence = torch.sigmoid(-log_variance)
    return (
        pepd.valid.float()
        * entropy_confidence
        * resultant
        * variance_confidence
    ).clamp(0.0, 1.0)


class CAGHV5SolverGatedResidual(nn.Module):
    """Two-parameter corrective fusion around the differentiable solver."""

    def __init__(self, *, progress_bins: int = 72, dropout: float = 0.10) -> None:
        super().__init__()
        self.parent = CAGHV5UnifiedModel(
            progress_bins=int(progress_bins), dropout=float(dropout)
        )
        self.pepd_residual_gain = nn.Parameter(torch.tensor(0.0))
        self.mask_residual_gain = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "progress_grid", torch.linspace(0.0, 1.0, int(progress_bins))
        )
        self.configure_trainable_arm(FULL_GATED)
        self.parent.eval()

    def load_parent_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.parent.load_state_dict(state_dict, strict=True)
        self.parent.eval()

    @staticmethod
    def arm_uses_pepd_residual(arm: str) -> bool:
        _require(arm in ARM_NAMES, f"unknown solver-gated arm: {arm}")
        return arm in (FULL_GATED, NO_MASK_RESIDUAL)

    @staticmethod
    def arm_uses_mask_residual(arm: str) -> bool:
        _require(arm in ARM_NAMES, f"unknown solver-gated arm: {arm}")
        return arm in (FULL_GATED, NO_PEPD_RESIDUAL)

    def configure_trainable_arm(self, arm: str) -> tuple[str, ...]:
        _require(arm in ARM_NAMES, f"unknown solver-gated arm: {arm}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.pepd_residual_gain.requires_grad_(self.arm_uses_pepd_residual(arm))
        self.mask_residual_gain.requires_grad_(self.arm_uses_mask_residual(arm))
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    def train(self, mode: bool = True) -> "CAGHV5SolverGatedResidual":
        # The parent is a frozen terminal model.  Only two scalar gains train.
        nn.Module.train(self, bool(mode))
        self.parent.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        final_to_isotropic: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        arm: str = FULL_GATED,
    ) -> SolverGatedResidualOutputs:
        _require(arm in ARM_NAMES, f"unknown solver-gated arm: {arm}")
        features = self.parent.cagh.forward_multiscale_features(image)
        pepd = self.parent._decode_pepd(features)
        batch = int(image.shape[0])
        device = image.device

        reference = self.parent.reference_head(features.c2, features.c5)
        # A constant center makes the solver anchor independent of decoded PEPD
        # pivot and direction.  The PEPD claim in this version is deliberately
        # limited to the probabilistic direction residual.
        safe_pivot = torch.full(
            (batch, 2), 0.5, device=device, dtype=torch.float32
        )
        start, arc, radii, reference_valid = projective_reference_geometry(
            reference.start_xy,
            reference.end_xy,
            safe_pivot,
            final_to_isotropic,
        )

        # Both legacy evidence switches are false.  Thus PEPD direction, mask
        # support weighting, the fixed mask prior, and their old interaction
        # inputs cannot reach the core final posterior.
        core = self.parent.cagh.forward_from_multiscale_features(
            features,
            start,
            arc,
            crop_affine,
            reference_valid,
            use_pepd_evidence=False,
            use_mask_geometry_evidence=False,
        )
        directions = _progress_directions(
            start, arc, crop_affine, bins=self.progress_grid.numel()
        )

        safe_pepd_direction = torch.nan_to_num(
            pepd.direction.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        pepd_evidence = torch.sum(
            directions * safe_pepd_direction[:, None, :], dim=2
        )
        raw_mask_axis = core.mask_geometry_axis.float()
        mask_axis = torch.nan_to_num(
            raw_mask_axis, nan=0.0, posinf=0.0, neginf=0.0
        )
        mask_evidence = torch.sum(directions * mask_axis[:, None, :], dim=2)
        raw_core_evidence = core.keypoint_angle_evidence.float()
        core_finite = torch.isfinite(raw_core_evidence).all(1)
        core_evidence = torch.nan_to_num(
            raw_core_evidence, nan=0.0, posinf=0.0, neginf=0.0
        )

        standardized_core, core_scale = _center_and_scale(core_evidence)
        standardized_pepd, _ = _center_and_scale(pepd_evidence)
        standardized_mask, _ = _center_and_scale(mask_evidence)
        pepd_residual = core_scale * (standardized_pepd - standardized_core)
        mask_residual = core_scale * (standardized_mask - standardized_core)

        reference_confidence = torch.nan_to_num(
            reference.gate_confidence.float(), nan=0.0
        ).clamp(0.0, 1.0)
        reference_confidence = (
            reference_confidence
            * reference.telemetry_valid.float()
            * reference_valid.float()
        )
        pepd_alignment = (
            0.5
            + 0.5
            * torch.sum(
                safe_pepd_direction
                * torch.nan_to_num(
                    core.keypoint_direction.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dim=1,
            )
        ).clamp(0.0, 1.0)
        pepd_gate = torch.nan_to_num(
            reference_confidence * _pepd_quality(pepd) * pepd_alignment
        ).clamp(0.0, 1.0)

        mask_quality = (
            torch.nan_to_num(core.mask_geometry_axis_score.float(), nan=0.0).clamp(
                0.0, 1.0
            )
            * torch.clamp(
                torch.nan_to_num(core.mask_geometry_support.float(), nan=0.0) / 0.05,
                0.0,
                1.0,
            )
        )
        mask_alignment = torch.abs(
            torch.sum(
                mask_axis
                * torch.nan_to_num(
                    core.keypoint_direction.float(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dim=1,
            )
        ).clamp(0.0, 1.0)
        mask_finite = (
            torch.isfinite(raw_mask_axis).all(1)
            & torch.isfinite(core.mask_geometry_axis_score)
            & torch.isfinite(core.mask_geometry_support)
            & (core.mask_geometry_support > 1e-5)
        )
        mask_gate = torch.nan_to_num(
            reference_confidence
            * mask_quality
            * mask_alignment
            * mask_finite.float()
        ).clamp(0.0, 1.0)

        pepd_scale = (
            torch.tanh(self.pepd_residual_gain.float())
            if self.arm_uses_pepd_residual(arm)
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        mask_scale = (
            torch.tanh(self.mask_residual_gain.float())
            if self.arm_uses_mask_residual(arm)
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        final_logits = core_evidence
        if self.arm_uses_pepd_residual(arm):
            safe_pepd_residual = torch.nan_to_num(
                pepd_gate[:, None].detach() * pepd_residual.detach()
            )
            final_logits = final_logits + pepd_scale * safe_pepd_residual
        if self.arm_uses_mask_residual(arm):
            safe_mask_residual = torch.nan_to_num(
                mask_gate[:, None].detach() * mask_residual.detach()
            )
            final_logits = final_logits + mask_scale * safe_mask_residual
        pre_fallback_log_probability = F.log_softmax(final_logits, dim=1)
        pre_fallback_finite = (
            torch.isfinite(final_logits).all(1)
            & torch.isfinite(pre_fallback_log_probability).all(1)
        )
        uniform = torch.full_like(
            pre_fallback_log_probability,
            -math.log(float(pre_fallback_log_probability.shape[1])),
        )
        geometry_valid = core.keypoint_solver_valid & reference_valid & core_finite
        log_probability = torch.where(
            (geometry_valid & pre_fallback_finite)[:, None],
            pre_fallback_log_probability,
            uniform,
        )
        expected = torch.sum(
            torch.exp(log_probability)
            * self.progress_grid.to(device=device, dtype=torch.float32)[None],
            dim=1,
        )
        posterior_finite = (
            torch.isfinite(log_probability).all(1) & torch.isfinite(expected)
        )
        fusion_solver_valid = geometry_valid & pre_fallback_finite & posterior_finite
        pointer_delta = core.tip_xy.float() - core.tail_xy.float()
        pointer_valid = (
            torch.isfinite(core.keypoint_direction).all(1)
            & torch.isfinite(pointer_delta).all(1)
            & (torch.linalg.vector_norm(torch.nan_to_num(pointer_delta), dim=1) > 1e-7)
        )
        valid = pointer_valid & reference_valid & fusion_solver_valid
        return SolverGatedResidualOutputs(
            arm=arm,
            progress_log_probability=log_probability,
            expected_progress=expected,
            valid=valid,
            pointer_valid=pointer_valid,
            reference_valid=reference_valid,
            geometry_solver_valid=core.keypoint_solver_valid,
            fusion_solver_valid=fusion_solver_valid,
            features=features,
            pepd=pepd,
            reference=reference,
            core=core,
            reference_start_angle=start,
            reference_range_angle=arc,
            reference_radii=radii,
            core_angle_evidence=core_evidence,
            pepd_angle_evidence=pepd_evidence,
            mask_angle_evidence=mask_evidence,
            pepd_reliability_gate=pepd_gate,
            mask_reliability_gate=mask_gate,
            pepd_residual_scale=pepd_scale.expand(batch),
            mask_residual_scale=mask_scale.expand(batch),
        )


def solver_gated_progress_loss(
    outputs: SolverGatedResidualOutputs,
    *,
    target_progress: torch.Tensor,
    group_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Only the final reading objective trains the two corrective scalars."""

    target = torch.as_tensor(
        target_progress,
        device=outputs.expected_progress.device,
        dtype=torch.float32,
    ).reshape(-1)
    weight = torch.as_tensor(
        group_weight,
        device=outputs.expected_progress.device,
        dtype=torch.float32,
    ).reshape(-1)
    _require(target.shape == outputs.expected_progress.shape, "target shape drift")
    _require(weight.shape == target.shape, "group weight shape drift")
    _require(bool(torch.isfinite(weight).all()) and bool((weight > 0.0).all()),
             "group weights must be finite and positive")
    selected = (
        outputs.valid
        & torch.isfinite(target)
        & (target >= 0.0)
        & (target <= 1.0)
    )
    safe = torch.nan_to_num(target, nan=0.5).clamp(0.0, 1.0)
    grid = torch.linspace(
        0.0,
        1.0,
        outputs.progress_log_probability.shape[1],
        device=target.device,
    )
    sigma = 1.25 / float(outputs.progress_log_probability.shape[1] - 1)
    soft = torch.exp(-0.5 * ((grid[None] - safe[:, None]) / sigma).square())
    soft = soft / soft.sum(1, keepdim=True).clamp_min(1e-8)

    def weighted(row: torch.Tensor) -> torch.Tensor:
        if not bool(selected.any()):
            return outputs.expected_progress.sum() * 0.0
        return (
            (row[selected] * weight[selected]).sum()
            / weight[selected].sum().clamp_min(1e-8)
        )

    ce = weighted(-(soft * outputs.progress_log_probability).sum(1))
    expected = weighted(
        F.smooth_l1_loss(
            outputs.expected_progress, safe, beta=0.02, reduction="none"
        )
    )
    loss = ce + 2.0 * expected
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("solver-gated progress loss is non-finite")
    return loss, {
        "final_soft_ce": ce.detach(),
        "final_expected_smooth_l1": expected.detach(),
        "combined": loss.detach(),
    }


__all__ = [
    "ARM_NAMES",
    "CAGHV5SolverGatedResidual",
    "FULL_GATED",
    "NO_MASK_RESIDUAL",
    "NO_PEPD_RESIDUAL",
    "PROTOCOL",
    "SOLVER_CORE",
    "SolverGatedResidualOutputs",
    "solver_gated_progress_loss",
]
