"""Internally unified CAGH-V5 model and mechanism-faithful ablations.

One encoder pass feeds the signed probabilistic PEPD direction path, the CAGH
mask/keypoint/dense-geometry path, and the V5 dense-tick reference head.  The
predicted ordered reference endpoints are converted to angles inside ``forward``
and consumed by the differentiable CAGH solver.  No ground-truth angle, numeric
range, ScaleMark, target, or reading is accepted by the model callback.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.cagh_net import (
    CAGHMultiscaleFeatures,
    CAGHNet,
    CAGHOutputs,
    cagh_multitask_loss,
)
from experiments.cagh_scalemark_reference_head_v5 import (
    DenseTickConditionedReferenceHeadV5,
    DenseTickReferenceOutputsV5,
    multiscale_dense_tick_loss,
    reference_consistency_loss,
)
from experiments.probabilistic_pivot_direction import (
    ProbabilisticDirectionPrediction,
    decode_probabilistic_pivot_direction,
)


FULL_UNIFIED = "full_unified"
NO_MASK_GEOMETRY = "no_mask_geometry"
NO_PEPD_DIRECTION = "no_pepd_direction"
NO_REFERENCE_SOLVER = "no_reference_solver"
NO_ROBUSTNESS_AUGMENTATION = "no_robustness_augmentation"
ARM_NAMES = (
    FULL_UNIFIED,
    NO_MASK_GEOMETRY,
    NO_PEPD_DIRECTION,
    NO_REFERENCE_SOLVER,
    NO_ROBUSTNESS_AUGMENTATION,
)


class CAGHV5UnifiedOutputs(NamedTuple):
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
    reference: DenseTickReferenceOutputsV5 | None
    cagh: CAGHOutputs | None
    reference_start_angle: torch.Tensor
    reference_range_angle: torch.Tensor
    reference_radii: torch.Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unproject(points: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points.float(), torch.ones_like(points[..., :1])), dim=-1)
    transformed = torch.einsum("bij,bkj->bki", homography.float(), homogeneous)
    denominator = transformed[..., 2:]
    signed_epsilon = torch.where(
        denominator >= 0.0,
        torch.full_like(denominator, 1e-8),
        torch.full_like(denominator, -1e-8),
    )
    denominator = torch.where(denominator.abs() >= 1e-8, denominator, signed_epsilon)
    return transformed[..., :2] / denominator


def projective_reference_geometry(
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    pivot_xy: torch.Tensor,
    final_to_isotropic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert predicted points to ordered angles without external references."""

    start = torch.as_tensor(start_xy, dtype=torch.float32)
    end = torch.as_tensor(end_xy, device=start.device, dtype=torch.float32)
    pivot = torch.as_tensor(pivot_xy, device=start.device, dtype=torch.float32)
    matrix = torch.as_tensor(
        final_to_isotropic, device=start.device, dtype=torch.float32
    )
    _require(start.ndim == 2 and start.shape[1] == 2, "start_xy must be [B,2]")
    _require(end.shape == start.shape and pivot.shape == start.shape, "point shape drift")
    _require(matrix.shape == (start.shape[0], 3, 3), "final_to_isotropic must be [B,3,3]")
    points = torch.stack((start, end, pivot), dim=1)
    isotropic = _unproject(points, matrix)
    start_vector = isotropic[:, 0] - isotropic[:, 2]
    end_vector = isotropic[:, 1] - isotropic[:, 2]
    radii = torch.stack(
        (
            torch.linalg.vector_norm(start_vector, dim=1),
            torch.linalg.vector_norm(end_vector, dim=1),
        ),
        dim=1,
    )
    start_angle = torch.remainder(
        torch.atan2(start_vector[:, 0], -start_vector[:, 1]) - math.pi,
        2.0 * math.pi,
    )
    end_angle = torch.remainder(
        torch.atan2(end_vector[:, 0], -end_vector[:, 1]) - math.pi,
        2.0 * math.pi,
    )
    arc = torch.remainder(end_angle - start_angle, 2.0 * math.pi)
    valid = (
        torch.isfinite(isotropic).all(dim=(1, 2))
        & torch.isfinite(start_angle)
        & torch.isfinite(arc)
        & (radii > 0.02).all(dim=1)
        & (arc > math.radians(10.0))
        & (arc < math.radians(350.0))
    )
    return start_angle, arc, radii, valid


class CAGHV5UnifiedModel(nn.Module):
    """One deployable image-to-progress model with internal reference solving."""

    def __init__(
        self,
        *,
        progress_bins: int = 72,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.cagh = CAGHNet(
            progress_bins=progress_bins,
            imagenet_pretrained=False,
            dropout=dropout,
        )
        self.reference_head = DenseTickConditionedReferenceHeadV5(
            hidden_channels=64, residual_blocks=2
        )
        self.direct_progress_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, int(progress_bins)),
        )
        self.register_buffer(
            "progress_grid", torch.linspace(0.0, 1.0, int(progress_bins))
        )
        self._initialize_direct_head()

    def _initialize_direct_head(self) -> None:
        for module in self.direct_progress_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def load_pepd_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self.cagh.load_pepd_state_dict(state_dict)

    @staticmethod
    def arm_uses_pepd(arm: str) -> bool:
        _require(arm in ARM_NAMES, f"unknown V5 arm: {arm}")
        return arm not in (NO_PEPD_DIRECTION, NO_REFERENCE_SOLVER)

    @staticmethod
    def arm_uses_mask_geometry(arm: str) -> bool:
        _require(arm in ARM_NAMES, f"unknown V5 arm: {arm}")
        return arm not in (NO_MASK_GEOMETRY, NO_REFERENCE_SOLVER)

    @staticmethod
    def arm_uses_reference_solver(arm: str) -> bool:
        _require(arm in ARM_NAMES, f"unknown V5 arm: {arm}")
        return arm != NO_REFERENCE_SOLVER

    def configure_trainable_arm(self, arm: str) -> tuple[str, ...]:
        """Freeze the leakage-safe terminal PEPD and train only new V5 modules."""

        _require(arm in ARM_NAMES, f"unknown V5 arm: {arm}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        modules: list[nn.Module] = []
        if arm == NO_REFERENCE_SOLVER:
            modules.append(self.direct_progress_head)
        else:
            modules.append(self.reference_head)
            modules.extend(
                getattr(self.cagh, name)
                for name in self.cagh._NEW_MODULE_NAMES
                if not (arm == NO_MASK_GEOMETRY and name == "mask_support_head")
            )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    def _decode_pepd(
        self, features: CAGHMultiscaleFeatures
    ) -> ProbabilisticDirectionPrediction:
        pooled = self.cagh.direction_features(features.c5)
        return decode_probabilistic_pivot_direction(
            self.cagh.pivot_head(features.c5),
            self.cagh.vector_head(pooled),
            self.cagh.angle_head(pooled),
            self.cagh.log_variance_head(pooled),
        )

    def forward(
        self,
        image: torch.Tensor,
        final_to_isotropic: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        arm: str = FULL_UNIFIED,
    ) -> CAGHV5UnifiedOutputs:
        """Predict normalized progress from pixels and preprocessing transforms only."""

        _require(arm in ARM_NAMES, f"unknown V5 arm: {arm}")
        features = self.cagh.forward_multiscale_features(image)
        pepd = self._decode_pepd(features)
        batch = image.shape[0]
        device = image.device

        if arm == NO_REFERENCE_SOLVER:
            # Spatial mean is the exact 1x1 adaptive-average result and keeps
            # strict CUDA determinism available during the control training.
            pooled = features.c5.float().mean(dim=(2, 3))
            logits = self.direct_progress_head(pooled).float()
            log_probability = F.log_softmax(logits, dim=1)
            expected = torch.sum(
                torch.exp(log_probability)
                * self.progress_grid.to(device=device, dtype=torch.float32)[None],
                dim=1,
            )
            valid = torch.isfinite(log_probability).all(1) & torch.isfinite(expected)
            nan = torch.full((batch,), float("nan"), device=device)
            return CAGHV5UnifiedOutputs(
                arm=arm,
                progress_log_probability=log_probability,
                expected_progress=expected,
                valid=valid,
                pointer_valid=valid,
                reference_valid=torch.zeros(batch, dtype=torch.bool, device=device),
                geometry_solver_valid=torch.zeros(
                    batch, dtype=torch.bool, device=device
                ),
                fusion_solver_valid=valid,
                features=features,
                pepd=pepd,
                reference=None,
                cagh=None,
                reference_start_angle=nan,
                reference_range_angle=nan,
                reference_radii=torch.full((batch, 2), float("nan"), device=device),
            )

        reference = self.reference_head(features.c2, features.c5)
        pivot = (
            pepd.pivot_xy / 63.0
            if self.arm_uses_pepd(arm)
            else torch.full((batch, 2), 0.5, device=device, dtype=torch.float32)
        )
        start, arc, radii, reference_valid = projective_reference_geometry(
            reference.start_xy,
            reference.end_xy,
            pivot,
            final_to_isotropic,
        )
        cagh = self.cagh.forward_from_multiscale_features(
            features,
            start,
            arc,
            crop_affine,
            reference_valid,
            use_pepd_evidence=self.arm_uses_pepd(arm),
            use_mask_geometry_evidence=self.arm_uses_mask_geometry(arm),
        )
        pointer_valid = (
            pepd.valid
            if self.arm_uses_pepd(arm)
            else (
                torch.isfinite(cagh.keypoint_direction).all(1)
                & (torch.linalg.vector_norm(cagh.tip_xy - cagh.tail_xy, dim=1) > 1e-7)
            )
        )
        geometry_solver_valid = cagh.keypoint_solver_valid
        fusion_solver_valid = cagh.valid & torch.isfinite(cagh.expected_progress)
        valid = pointer_valid & reference_valid & fusion_solver_valid
        return CAGHV5UnifiedOutputs(
            arm=arm,
            progress_log_probability=cagh.progress_log_probability,
            expected_progress=cagh.expected_progress,
            valid=valid,
            pointer_valid=pointer_valid,
            reference_valid=reference_valid,
            geometry_solver_valid=geometry_solver_valid,
            fusion_solver_valid=fusion_solver_valid,
            features=features,
            pepd=pepd,
            reference=reference,
            cagh=cagh,
            reference_start_angle=start,
            reference_range_angle=arc,
            reference_radii=radii,
        )


def pointer_line_mask(
    tail_xy: torch.Tensor,
    tip_xy: torch.Tensor,
    *,
    size: int = 64,
    sigma: float = 0.012,
) -> torch.Tensor:
    """Generate a soft differentiable Base-style pointer-line supervision mask."""

    tail = torch.as_tensor(tail_xy, dtype=torch.float32)
    tip = torch.as_tensor(tip_xy, device=tail.device, dtype=torch.float32)
    _require(tail.ndim == 2 and tail.shape[1] == 2 and tip.shape == tail.shape,
             "pointer endpoints must be [B,2]")
    axis = torch.linspace(0.0, 1.0, int(size), device=tail.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    point = torch.stack((xx, yy), dim=2)[None]
    delta = tip - tail
    squared_length = delta.square().sum(1).clamp_min(1e-8)
    phase = torch.sum((point - tail[:, None, None]) * delta[:, None, None], dim=3)
    phase = (phase / squared_length[:, None, None]).clamp(0.0, 1.0)
    projection = tail[:, None, None] + phase[..., None] * delta[:, None, None]
    squared_distance = (point - projection).square().sum(3)
    mask = torch.exp(-0.5 * squared_distance / float(sigma) ** 2)
    return mask[:, None].clamp(0.0, 1.0)


def _weighted(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / weight.sum().clamp_min(1e-8)


def _progress_loss(
    log_probability: torch.Tensor,
    expected: torch.Tensor,
    target_progress: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(
        target_progress, device=expected.device, dtype=torch.float32
    ).reshape(-1)
    selected = valid & torch.isfinite(target) & (target >= 0.0) & (target <= 1.0)
    safe = torch.nan_to_num(target, nan=0.5).clamp(0.0, 1.0)
    grid = torch.linspace(0.0, 1.0, log_probability.shape[1], device=target.device)
    sigma = 1.25 / float(log_probability.shape[1] - 1)
    soft = torch.exp(-0.5 * ((grid[None] - safe[:, None]) / sigma).square())
    soft = soft / soft.sum(1, keepdim=True).clamp_min(1e-8)
    if bool(selected.any()):
        ce = _weighted(-(soft * log_probability).sum(1)[selected], weight[selected])
        expected_loss = _weighted(
            F.smooth_l1_loss(expected, safe, beta=0.02, reduction="none")[selected],
            weight[selected],
        )
    else:
        ce = log_probability.sum() * 0.0
        expected_loss = expected.sum() * 0.0
    return ce + 2.0 * expected_loss, ce, expected_loss


def cagh_v5_unified_loss(
    outputs: CAGHV5UnifiedOutputs,
    *,
    target_progress: torch.Tensor,
    target_tip_xy: torch.Tensor,
    target_tail_xy: torch.Tensor,
    target_endpoints: torch.Tensor,
    target_tick_heatmap: torch.Tensor,
    target_start_angle: torch.Tensor,
    target_range_angle: torch.Tensor,
    group_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint loss whose final-reading term reaches every enabled V5 branch."""

    device = outputs.expected_progress.device
    weight = torch.as_tensor(group_weight, device=device, dtype=torch.float32).reshape(-1)
    _require(weight.shape == outputs.expected_progress.shape, "group weight shape drift")
    _require(bool(torch.isfinite(weight).all()) and bool((weight > 0.0).all()),
             "group weights must be finite and positive")

    if outputs.arm == NO_REFERENCE_SOLVER:
        progress, ce, expected = _progress_loss(
            outputs.progress_log_probability,
            outputs.expected_progress,
            target_progress,
            outputs.valid,
            weight,
        )
        return progress, {
            "final_progress": progress.detach(),
            "final_soft_ce": ce.detach(),
            "final_expected_smooth_l1": expected.detach(),
        }

    assert outputs.reference is not None and outputs.cagh is not None
    target_tip = torch.as_tensor(target_tip_xy, device=device, dtype=torch.float32)
    target_tail = torch.as_tensor(target_tail_xy, device=device, dtype=torch.float32)
    mask = pointer_line_mask(target_tail, target_tip)
    use_mask_geometry = CAGHV5UnifiedModel.arm_uses_mask_geometry(outputs.arm)
    cagh_loss, cagh_parts = cagh_multitask_loss(
        outputs.cagh,
        target_progress,
        target_tip,
        target_tail,
        target_mask_probability=mask if use_mask_geometry else None,
        group_weight=weight,
        use_mask_geometry=use_mask_geometry,
    )

    tick_loss, tick_parts = multiscale_dense_tick_loss(
        outputs.reference,
        torch.as_tensor(target_tick_heatmap, device=device, dtype=torch.float32),
        group_weight=weight,
    )
    endpoint_loss, endpoint_parts = reference_consistency_loss(
        outputs.reference,
        torch.as_tensor(target_endpoints, device=device, dtype=torch.float32),
        group_weight=weight,
    )
    target_start = torch.as_tensor(target_start_angle, device=device, dtype=torch.float32)
    target_range = torch.as_tensor(target_range_angle, device=device, dtype=torch.float32)
    angle_row = 1.0 - torch.cos(outputs.reference_start_angle - target_start)
    angle_row = angle_row + F.smooth_l1_loss(
        outputs.reference_range_angle / (2.0 * math.pi),
        target_range / (2.0 * math.pi),
        reduction="none",
        beta=0.02,
    )
    reference_geometry_loss = _weighted(angle_row, weight)

    pepd_loss = outputs.expected_progress.sum() * 0.0
    if CAGHV5UnifiedModel.arm_uses_pepd(outputs.arm):
        pepd_loss, _, _ = _progress_loss(
            outputs.cagh.base_pepd_progress_log_probability,
            outputs.cagh.base_pepd_expected_progress,
            target_progress,
            outputs.reference_valid,
            weight,
        )

    loss = (
        cagh_loss
        + 0.50 * tick_loss
        + endpoint_loss
        + 0.25 * reference_geometry_loss
        + 0.10 * pepd_loss
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("unified CAGH-V5 loss is non-finite")
    parts = {
        **{f"cagh_{name}": value for name, value in cagh_parts.items()},
        **{f"tick_{name}": value for name, value in tick_parts.items()},
        **{f"reference_{name}": value for name, value in endpoint_parts.items()},
        "reference_geometry_angle_arc": reference_geometry_loss.detach(),
        "pepd_probability_progress": pepd_loss.detach(),
        "combined": loss.detach(),
    }
    return loss, parts


__all__ = [
    "ARM_NAMES",
    "CAGHV5UnifiedModel",
    "CAGHV5UnifiedOutputs",
    "FULL_UNIFIED",
    "NO_MASK_GEOMETRY",
    "NO_PEPD_DIRECTION",
    "NO_REFERENCE_SOLVER",
    "NO_ROBUSTNESS_AUGMENTATION",
    "cagh_v5_unified_loss",
    "pointer_line_mask",
    "projective_reference_geometry",
]
