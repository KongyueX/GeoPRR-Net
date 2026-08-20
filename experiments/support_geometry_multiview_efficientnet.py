"""Next-generation EfficientNet-B0 core for robust pointer regression.

The prototype has two separable architectural modules:

1. Support--Geometry Conditioned Attention (SGCA).  A shared EfficientNet-B0
   encoder processes the original ROI, the SARN-v2 normalized ROI, and an
   optional quadrilateral-rectified ROI.  SGCA is inserted after
   ``features[5]`` (112 channels at 16x16 for a 256x256 input), before stages
   6--8.  The SARN support mask participates in masked pooling and spatial
   modulation.  A geometry head predicts pivot, direction, and two reference
   points; that prediction is embedded and used to modulate the middle
   channels and spatial responses seen by all remaining backbone stages.
2. Constrained probabilistic multi-view fusion.  Every view predicts a mean and
   variance.  Learned, uncertainty-aware weights form a convex combination
   while retaining a fixed minimum weight on the original view.  Total output
   variance follows the law of total variance.  The first-stage default uses
   only original and SARN views; rectification is an explicit later ablation.

No condition name, dataset identity, label, or degradation metadata enters the
model.  Availability comes only from the preprocessing result.  When that
result marks SARN as a no-op/unavailable, or the view is numerically invalid,
the output is selected bit-exactly from the original-view prediction and
uncertainty.

This file intentionally contains only the network core and ablation surface;
it does not add a training runner or modify any frozen paper model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


ARCHITECTURE: Final[str] = "EfficientNet-B0-SGCA-Probabilistic-MultiView"
A5_ARCHITECTURE: Final[str] = (
    "EfficientNet-B0-Geometry-Consistency-Reliability-Fusion"
)
VIEW_NAMES: Final[tuple[str, ...]] = ("original", "sarn_v2", "rectified")
EFFICIENTNET_B0_FEATURES: Final[int] = 1280
EFFICIENTNET_B0_MIDDLE_FEATURES: Final[int] = 112
EFFICIENTNET_B0_SGCA_STAGE: Final[int] = 5
GEOMETRY_VALUES: Final[int] = 8
GEOMETRY_CONSISTENCY_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "pivot_distance",
    "direction_disagreement",
    "reference_mean_distance",
    "reference_max_distance",
    "support_fraction",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ArchitectureAblation:
    """Runtime switches for causal, same-weight module ablations."""

    use_support_mask: bool = True
    use_geometry_attention: bool = True
    use_multiview_fusion: bool = True
    use_learned_fusion: bool = True
    use_uncertainty_in_fusion: bool = True
    use_rectified_view: bool = True

    def __post_init__(self) -> None:
        if self.use_uncertainty_in_fusion and not self.use_multiview_fusion:
            raise ValueError(
                "uncertainty-aware fusion requires multi-view fusion"
            )
        if self.use_learned_fusion and not self.use_multiview_fusion:
            raise ValueError("learned fusion requires multi-view fusion")
        if self.use_uncertainty_in_fusion and not self.use_learned_fusion:
            raise ValueError(
                "uncertainty-aware fusion requires learned fusion"
            )


FULL_ARCHITECTURE: Final[ArchitectureAblation] = ArchitectureAblation()
DEFAULT_TWO_VIEW_ARCHITECTURE: Final[ArchitectureAblation] = ArchitectureAblation(
    use_rectified_view=False,
)
CAUSAL_A1_RAW: Final[ArchitectureAblation] = ArchitectureAblation(
    use_support_mask=False,
    use_geometry_attention=False,
    use_multiview_fusion=False,
    use_learned_fusion=False,
    use_uncertainty_in_fusion=False,
    use_rectified_view=False,
)
CAUSAL_A2_UNIFORM_TWO_VIEW: Final[ArchitectureAblation] = ArchitectureAblation(
    use_support_mask=False,
    use_geometry_attention=False,
    use_learned_fusion=False,
    use_uncertainty_in_fusion=False,
    use_rectified_view=False,
)
CAUSAL_A3_UNCERTAINTY_TWO_VIEW: Final[ArchitectureAblation] = ArchitectureAblation(
    use_support_mask=False,
    use_geometry_attention=False,
    use_rectified_view=False,
)
CAUSAL_A4_SGCA_TWO_VIEW: Final[ArchitectureAblation] = (
    DEFAULT_TWO_VIEW_ARCHITECTURE
)


def causal_ladder_ablations() -> dict[str, ArchitectureAblation]:
    """Return the nested A1--A4 causal architecture ladder."""

    return {
        "a1_raw": CAUSAL_A1_RAW,
        "a2_uniform_two_view": CAUSAL_A2_UNIFORM_TWO_VIEW,
        "a3_uncertainty_two_view": CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
        "a4_sgca_two_view": CAUSAL_A4_SGCA_TWO_VIEW,
    }


def canonical_ablations() -> dict[str, ArchitectureAblation]:
    """Return the intended same-weight ablation roster."""

    return {
        "original_only": ArchitectureAblation(
            use_support_mask=False,
            use_geometry_attention=False,
            use_multiview_fusion=False,
            use_learned_fusion=False,
            use_uncertainty_in_fusion=False,
            use_rectified_view=False,
        ),
        "without_support_mask": ArchitectureAblation(
            use_support_mask=False,
            use_rectified_view=False,
        ),
        "without_geometry_attention": ArchitectureAblation(
            use_geometry_attention=False,
            use_rectified_view=False,
        ),
        "without_multiview_fusion": ArchitectureAblation(
            use_multiview_fusion=False,
            use_learned_fusion=False,
            use_uncertainty_in_fusion=False,
            use_rectified_view=False,
        ),
        "uniform_convex_fusion": ArchitectureAblation(
            use_learned_fusion=False,
            use_uncertainty_in_fusion=False,
            use_rectified_view=False,
        ),
        "deterministic_convex_fusion": ArchitectureAblation(
            use_uncertainty_in_fusion=False,
            use_rectified_view=False,
        ),
        "two_view_without_rectification": ArchitectureAblation(
            use_rectified_view=False,
        ),
        "full": FULL_ARCHITECTURE,
    }


def _masked_average(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _require(features.ndim == 4, "features must be BCHW")
    _require(
        mask.shape == (features.shape[0], 1, features.shape[2], features.shape[3]),
        "support mask and feature shapes differ",
    )
    denominator = mask.sum(dim=(2, 3)).clamp_min(
        torch.finfo(features.dtype).eps
    )
    return (features * mask).sum(dim=(2, 3)) / denominator


class SharedEfficientNetB0Encoder(nn.Module):
    """One EfficientNet split around torchvision ``features[5]`` for SGCA."""

    def __init__(self, features: nn.Sequential) -> None:
        super().__init__()
        modules = tuple(features.children())
        _require(len(modules) == 9, "unexpected EfficientNet-B0 stage roster")
        split = EFFICIENTNET_B0_SGCA_STAGE + 1
        self.early = nn.Sequential(*modules[:split])
        self.late = nn.Sequential(*modules[split:])

    def encode_early(self, images: torch.Tensor) -> torch.Tensor:
        middle = self.early(images)
        _require(
            middle.ndim == 4
            and middle.shape[1] == EFFICIENTNET_B0_MIDDLE_FEATURES,
            "EfficientNet-B0 middle feature shape drifted",
        )
        return middle

    def encode_late(self, middle: torch.Tensor) -> torch.Tensor:
        final = self.late(middle)
        _require(
            final.ndim == 4 and final.shape[1] == EFFICIENTNET_B0_FEATURES,
            "EfficientNet-B0 final feature shape drifted",
        )
        return final

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_late(self.encode_early(images))


class SupportGeometryConditionedAttention(nn.Module):
    """Masked pooling plus geometry-conditioned middle feature modulation."""

    def __init__(
        self,
        channels: int,
        *,
        geometry_embedding_dim: int = 96,
        spatial_hidden_channels: int = 24,
        maximum_residual: float = 0.30,
    ) -> None:
        super().__init__()
        _require(channels >= 1, "attention channels must be positive")
        _require(geometry_embedding_dim >= 8, "geometry embedding is too small")
        _require(spatial_hidden_channels >= 4, "spatial attention is too small")
        _require(
            0.0 < float(maximum_residual) < 1.0,
            "attention residual must be in (0,1)",
        )
        self.channels = int(channels)
        self.geometry_embedding_dim = int(geometry_embedding_dim)
        self.maximum_residual = float(maximum_residual)

        hidden = max(64, min(256, self.channels // 4))
        self.geometry_head = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, GEOMETRY_VALUES),
        )
        self.geometry_embedding = nn.Sequential(
            nn.Linear(GEOMETRY_VALUES + 1, geometry_embedding_dim),
            nn.SiLU(),
            nn.Linear(geometry_embedding_dim, geometry_embedding_dim),
            nn.SiLU(),
        )
        self.channel_modulation = nn.Linear(geometry_embedding_dim, channels)
        self.spatial_feature = nn.Conv2d(
            channels, spatial_hidden_channels, kernel_size=1, bias=False
        )
        self.spatial_basis = nn.Conv2d(
            6, spatial_hidden_channels, kernel_size=1, bias=False
        )
        self.spatial_film = nn.Linear(
            geometry_embedding_dim, 2 * spatial_hidden_channels
        )
        self.spatial_output = nn.Conv2d(
            spatial_hidden_channels, 1, kernel_size=3, padding=1
        )

        # The module begins as an identity residual while the support-masked
        # pooling and auxiliary geometry prediction remain active.
        nn.init.zeros_(self.channel_modulation.weight)
        nn.init.zeros_(self.channel_modulation.bias)
        nn.init.zeros_(self.spatial_output.weight)
        nn.init.zeros_(self.spatial_output.bias)

    @staticmethod
    def _unit_direction(raw: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        default = F.one_hot(
            torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device),
            num_classes=2,
        ).to(dtype=raw.dtype)
        # Add, rather than select, the fallback so a collapsed direction still
        # receives a usable rotational gradient from direction supervision.
        recoverable = raw + (norm <= 1.0e-6).to(raw.dtype) * 1.0e-6 * default
        return F.normalize(
            recoverable,
            dim=1,
            eps=torch.finfo(raw.dtype).eps,
        )

    @staticmethod
    def _geometry_basis(
        *,
        pivot: torch.Tensor,
        direction_sin_cos: torch.Tensor,
        references: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _one, height, width = support_mask.shape
        y = torch.linspace(
            0.0, 1.0, height, dtype=support_mask.dtype, device=support_mask.device
        )
        x = torch.linspace(
            0.0, 1.0, width, dtype=support_mask.dtype, device=support_mask.device
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dx = xx[None] - pivot[:, 0, None, None]
        dy = yy[None] - pivot[:, 1, None, None]
        # Geometry labels use the established clock convention [sin(theta),
        # cos(theta)], where theta=0 points upward.  Image y increases downward,
        # so the corresponding Cartesian image vector is [sin(theta), -cos(theta)].
        direction_x = direction_sin_cos[:, 0, None, None]
        direction_y = -direction_sin_cos[:, 1, None, None]
        axial = dx * direction_x + dy * direction_y
        cross = -dx * direction_y + dy * direction_x
        radial = torch.sqrt(dx.square() + dy.square() + 1.0e-8)

        reference_dx = xx[None, None] - references[:, :, 0, None, None]
        reference_dy = yy[None, None] - references[:, :, 1, None, None]
        reference_distance = torch.sqrt(
            reference_dx.square() + reference_dy.square() + 1.0e-8
        )
        nearest_reference = torch.amin(reference_distance, dim=1)
        return torch.stack(
            (
                support_mask[:, 0],
                axial,
                torch.abs(cross),
                radial,
                nearest_reference,
                xx.expand(batch, -1, -1),
            ),
            dim=1,
        )

    def forward(
        self,
        features: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        use_support_mask: bool = True,
        use_geometry_attention: bool = True,
        detach_geometry_from_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        _require(
            features.ndim == 4 and features.shape[1] == self.channels,
            "attention feature shape mismatch",
        )
        _require(bool(torch.isfinite(features).all()), "features are non-finite")
        _require(
            support_mask.shape
            == (features.shape[0], 1, features.shape[2], features.shape[3]),
            "attention support-mask shape mismatch",
        )
        _require(bool(torch.isfinite(support_mask).all()), "support mask is non-finite")
        mask = torch.clamp(
            support_mask.to(dtype=features.dtype), 0.0, 1.0
        )
        if not use_support_mask:
            mask = torch.ones_like(mask)

        pooled_before = _masked_average(features, mask)
        # A3G/A5 use the geometry predictor as an auxiliary side head.  Its
        # labels may be much easier than the reading objective, so allowing
        # that loss to dominate the shared early encoder defeats the intended
        # A3-controlled comparison.  Detaching only this input preserves the
        # exact forward values and still trains every geometry-head parameter.
        geometry_source = (
            pooled_before.detach()
            if detach_geometry_from_features
            else pooled_before
        )
        raw_geometry = self.geometry_head(geometry_source)
        pivot = torch.sigmoid(raw_geometry[:, 0:2])
        direction = self._unit_direction(raw_geometry[:, 2:4])
        references = torch.sigmoid(raw_geometry[:, 4:8]).reshape(-1, 2, 2)
        geometry_values = torch.cat(
            (pivot, direction, references.flatten(1)), dim=1
        )
        support_fraction = mask.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        embedding = self.geometry_embedding(
            torch.cat((geometry_values, support_fraction), dim=1)
        )

        if use_geometry_attention:
            branch_radius = 0.5 * self.maximum_residual
            channel_delta = branch_radius * torch.tanh(
                self.channel_modulation(embedding)
            )[:, :, None, None]
            basis = self._geometry_basis(
                pivot=pivot,
                direction_sin_cos=direction,
                references=references,
                support_mask=mask,
            )
            spatial = self.spatial_feature(features) + self.spatial_basis(basis)
            gamma, beta = self.spatial_film(embedding).chunk(2, dim=1)
            spatial = F.silu(
                spatial * (1.0 + 0.25 * torch.tanh(gamma)[:, :, None, None])
                + 0.25 * torch.tanh(beta)[:, :, None, None]
            )
            spatial_delta = (
                branch_radius
                * torch.tanh(self.spatial_output(spatial))
                * mask
            )
            attention_residual = channel_delta + spatial_delta
            attended = features * (1.0 + attention_residual)
        else:
            attention_residual = torch.zeros_like(features)
            attended = features

        pooled_after = _masked_average(attended, mask)
        return {
            "features": attended,
            "pooled": pooled_after,
            "pooled_before_attention": pooled_before,
            "pivot": pivot,
            "direction_sin_cos": direction,
            # ``references`` follows the established Bx4 supervision shape;
            # ``reference_points`` retains Bx2x2 for spatial diagnostics.
            "references": references.flatten(1),
            "reference_points": references,
            "geometry_embedding": embedding,
            "support_fraction": support_fraction[:, 0],
            "attention_residual": attention_residual,
            "effective_support_mask": mask,
        }


class GeometryConsistencyReliabilityFusion(nn.Module):
    """Map SARN geometry to raw coordinates and amend only its fusion logit."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        maximum_logit_correction: float = 2.0,
        maximum_condition_number: float = 1.0e5,
    ) -> None:
        super().__init__()
        _require(hidden_dim >= 4, "geometry-consistency MLP is too small")
        _require(
            0.0 < float(maximum_logit_correction) <= 8.0,
            "geometry-consistency logit bound is invalid",
        )
        _require(
            float(maximum_condition_number) >= 10.0,
            "homography condition-number bound is invalid",
        )
        self.maximum_logit_correction = float(maximum_logit_correction)
        self.maximum_condition_number = float(maximum_condition_number)
        self.correction_mlp = nn.Sequential(
            nn.Linear(len(GEOMETRY_CONSISTENCY_FEATURE_NAMES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Exact A3 equivalence at initialization.  The last layer learns on the
        # first update.  Geometry itself remains supervised only by its explicit
        # auxiliary targets, never by a route that it could learn to manipulate.
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

    @staticmethod
    def _validate_geometry_shapes(
        raw_pivot: torch.Tensor,
        raw_direction_sin_cos: torch.Tensor,
        raw_references: torch.Tensor,
        sarn_pivot: torch.Tensor,
        sarn_direction_sin_cos: torch.Tensor,
        sarn_references: torch.Tensor,
        support_fraction: torch.Tensor,
        sarn_active: torch.Tensor,
    ) -> int:
        _require(raw_pivot.ndim == 2 and raw_pivot.shape[1] == 2, "raw pivot must be Bx2")
        batch = raw_pivot.shape[0]
        _require(
            raw_direction_sin_cos.shape
            == sarn_pivot.shape
            == sarn_direction_sin_cos.shape
            == (batch, 2),
            "pivot/direction geometry shapes differ",
        )
        _require(
            raw_references.shape == sarn_references.shape == (batch, 4),
            "reference geometry must be Bx4",
        )
        _require(support_fraction.shape == (batch,), "support fraction must be B")
        _require(
            sarn_active.shape == (batch,) and sarn_active.dtype == torch.bool,
            "SARN activity must be boolean B",
        )
        tensors = (
            raw_pivot,
            raw_direction_sin_cos,
            raw_references,
            sarn_pivot,
            sarn_direction_sin_cos,
            sarn_references,
            support_fraction,
            sarn_active,
        )
        _require(
            all(tensor.device == raw_pivot.device for tensor in tensors),
            "geometry-consistency inputs must share a device",
        )
        _require(
            all(
                tensor.is_floating_point()
                for tensor in tensors
                if tensor is not sarn_active
            ),
            "geometry-consistency values must be floating point",
        )
        return batch

    @staticmethod
    def _homography_batch(
        raw_to_sarn_homography: torch.Tensor | None,
        *,
        batch: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        identity = torch.eye(
            3, dtype=reference.dtype, device=reference.device
        ).expand(batch, -1, -1)
        if raw_to_sarn_homography is None:
            return identity, torch.zeros(
                batch, dtype=torch.bool, device=reference.device
            )
        value = raw_to_sarn_homography
        _require(value.is_floating_point(), "raw-to-SARN homography must be floating point")
        _require(
            value.device == reference.device,
            "raw-to-SARN homography must share the geometry device",
        )
        if value.shape == (3, 3):
            _require(batch == 1, "an unbatched homography requires batch size one")
            value = value.unsqueeze(0)
        _require(
            value.shape == (batch, 3, 3),
            "raw-to-SARN homography must be Bx3x3",
        )
        return value.to(dtype=reference.dtype), torch.ones(
            batch, dtype=torch.bool, device=reference.device
        )

    @staticmethod
    def _project_points(
        inverse_homography: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        homogeneous = torch.cat(
            (points, torch.ones_like(points[..., :1])), dim=2
        )
        mapped_homogeneous = torch.einsum(
            "bij,bnj->bni", inverse_homography, homogeneous
        )
        denominator = mapped_homogeneous[..., 2]
        valid = (
            torch.isfinite(mapped_homogeneous).all(dim=(1, 2))
            & torch.all(torch.abs(denominator) > 1.0e-6, dim=1)
        )
        safe_denominator = torch.where(
            torch.abs(denominator) > 1.0e-6,
            denominator,
            torch.ones_like(denominator),
        )
        mapped = mapped_homogeneous[..., :2] / safe_denominator[..., None]
        valid = valid & torch.isfinite(mapped).all(dim=(1, 2))
        mapped = torch.nan_to_num(mapped, nan=0.0, posinf=4.0, neginf=-4.0)
        return mapped, valid

    @staticmethod
    def _map_direction_with_inverse_jacobian(
        inverse_homography: torch.Tensor,
        sarn_pivot: torch.Tensor,
        sarn_direction_sin_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = sarn_pivot[:, 0]
        y = sarn_pivot[:, 1]
        a, b, c = inverse_homography[:, 0].unbind(dim=1)
        d, e, f = inverse_homography[:, 1].unbind(dim=1)
        g, h, i = inverse_homography[:, 2].unbind(dim=1)
        denominator = g * x + h * y + i
        x_numerator = a * x + b * y + c
        y_numerator = d * x + e * y + f
        safe_denominator = torch.where(
            torch.abs(denominator) > 1.0e-6,
            denominator,
            torch.ones_like(denominator),
        )
        denominator_squared = safe_denominator.square()
        jacobian = torch.stack(
            (
                (a * safe_denominator - g * x_numerator) / denominator_squared,
                (b * safe_denominator - h * x_numerator) / denominator_squared,
                (d * safe_denominator - g * y_numerator) / denominator_squared,
                (e * safe_denominator - h * y_numerator) / denominator_squared,
            ),
            dim=1,
        ).reshape(-1, 2, 2)
        sarn_direction_xy = torch.stack(
            (
                sarn_direction_sin_cos[:, 0],
                -sarn_direction_sin_cos[:, 1],
            ),
            dim=1,
        )
        mapped_xy = torch.einsum("bij,bj->bi", jacobian, sarn_direction_xy)
        norm = torch.linalg.vector_norm(mapped_xy, dim=1, keepdim=True)
        valid = (
            torch.isfinite(jacobian).all(dim=(1, 2))
            & torch.isfinite(mapped_xy).all(dim=1)
            & (torch.abs(denominator) > 1.0e-6)
            & (norm[:, 0] > 1.0e-6)
        )
        safe_xy = mapped_xy / norm.clamp_min(1.0e-6)
        mapped_direction = torch.stack((safe_xy[:, 0], -safe_xy[:, 1]), dim=1)
        mapped_direction = torch.nan_to_num(mapped_direction, nan=0.0)
        return mapped_direction, valid

    def map_sarn_geometry_to_raw(
        self,
        *,
        sarn_pivot: torch.Tensor,
        sarn_direction_sin_cos: torch.Tensor,
        sarn_references: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        _require(
            sarn_pivot.ndim == 2 and sarn_pivot.shape[1] == 2,
            "SARN pivot must be Bx2",
        )
        batch = sarn_pivot.shape[0]
        _require(
            sarn_direction_sin_cos.shape == (batch, 2)
            and sarn_references.shape == (batch, 4),
            "SARN geometry shapes differ",
        )
        _require(
            sarn_pivot.is_floating_point()
            and sarn_direction_sin_cos.is_floating_point()
            and sarn_references.is_floating_point(),
            "SARN geometry must be floating point",
        )
        _require(
            sarn_direction_sin_cos.device
            == sarn_references.device
            == sarn_pivot.device,
            "SARN geometry must share a device",
        )
        calculation_dtype = (
            torch.float64 if sarn_pivot.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=sarn_pivot.device.type, enabled=False):
            pivot = sarn_pivot.to(dtype=calculation_dtype)
            direction = sarn_direction_sin_cos.to(dtype=calculation_dtype)
            references = sarn_references.to(dtype=calculation_dtype)
            homography, supplied = self._homography_batch(
                raw_to_sarn_homography,
                batch=batch,
                reference=pivot,
            )
            matrix_finite = torch.isfinite(homography).all(dim=(1, 2))
            identity = torch.eye(
                3, dtype=calculation_dtype, device=pivot.device
            ).expand(batch, -1, -1)
            finite_safe = torch.where(
                matrix_finite[:, None, None], homography, identity
            )
            matrix_norm = torch.linalg.matrix_norm(
                finite_safe, ord="fro", dim=(1, 2)
            )
            normalized = finite_safe / matrix_norm.clamp_min(1.0e-12)[:, None, None]
            singular_values = torch.linalg.svdvals(normalized)
            condition_number = singular_values[:, 0] / singular_values[:, -1].clamp_min(
                1.0e-12
            )
            matrix_valid = (
                supplied
                & matrix_finite
                & (matrix_norm > 1.0e-8)
                & (singular_values[:, -1] > 1.0e-7)
                & torch.isfinite(condition_number)
                & (condition_number <= self.maximum_condition_number)
            )
            safe_homography = torch.where(
                matrix_valid[:, None, None], homography, identity
            )
            inverse = torch.linalg.inv(safe_homography)
            points = torch.cat(
                (pivot[:, None], references.reshape(batch, 2, 2)), dim=1
            )
            mapped_points, points_valid = self._project_points(inverse, points)
            mapped_direction, direction_valid = (
                self._map_direction_with_inverse_jacobian(
                    inverse, pivot, direction
                )
            )
            transform_valid = matrix_valid & points_valid & direction_valid
        output_dtype = sarn_pivot.dtype
        return {
            "pivot": mapped_points[:, 0].to(dtype=output_dtype),
            "direction_sin_cos": mapped_direction.to(dtype=output_dtype),
            "references": mapped_points[:, 1:].flatten(1).to(dtype=output_dtype),
            "transform_valid": transform_valid,
            "homography_condition_number": condition_number.to(dtype=output_dtype),
        }

    def forward(
        self,
        *,
        raw_pivot: torch.Tensor,
        raw_direction_sin_cos: torch.Tensor,
        raw_references: torch.Tensor,
        sarn_pivot: torch.Tensor,
        sarn_direction_sin_cos: torch.Tensor,
        sarn_references: torch.Tensor,
        support_fraction: torch.Tensor,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor | None,
    ) -> dict[str, Any]:
        batch = self._validate_geometry_shapes(
            raw_pivot,
            raw_direction_sin_cos,
            raw_references,
            sarn_pivot,
            sarn_direction_sin_cos,
            sarn_references,
            support_fraction,
            sarn_active,
        )
        geometry_finite = torch.stack(
            (
                torch.isfinite(raw_pivot).all(dim=1),
                torch.isfinite(raw_direction_sin_cos).all(dim=1),
                torch.isfinite(raw_references).all(dim=1),
                torch.isfinite(sarn_pivot).all(dim=1),
                torch.isfinite(sarn_direction_sin_cos).all(dim=1),
                torch.isfinite(sarn_references).all(dim=1),
                torch.isfinite(support_fraction),
            ),
            dim=1,
        ).all(dim=1)
        safe_sarn_pivot = torch.nan_to_num(sarn_pivot, nan=0.5)
        safe_sarn_direction = torch.nan_to_num(
            sarn_direction_sin_cos, nan=0.0
        )
        safe_sarn_references = torch.nan_to_num(sarn_references, nan=0.5)
        mapped = self.map_sarn_geometry_to_raw(
            sarn_pivot=safe_sarn_pivot,
            sarn_direction_sin_cos=safe_sarn_direction,
            sarn_references=safe_sarn_references,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        valid = sarn_active & geometry_finite & mapped["transform_valid"]
        mapped_pivot = mapped["pivot"]
        mapped_direction = F.normalize(
            mapped["direction_sin_cos"], dim=1, eps=1.0e-6
        )
        mapped_references = mapped["references"].reshape(batch, 2, 2)
        raw_direction = F.normalize(
            torch.nan_to_num(raw_direction_sin_cos, nan=0.0),
            dim=1,
            eps=1.0e-6,
        )
        raw_reference_points = torch.nan_to_num(
            raw_references, nan=0.5
        ).reshape(batch, 2, 2)
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        pivot_distance = (
            torch.linalg.vector_norm(
                torch.nan_to_num(raw_pivot, nan=0.5) - mapped_pivot, dim=1
            )
            * inverse_sqrt_two
        ).clamp(0.0, 1.0)
        direction_disagreement = (
            0.5 * (1.0 - torch.sum(raw_direction * mapped_direction, dim=1))
        ).clamp(0.0, 1.0)
        reference_distances = (
            torch.linalg.vector_norm(
                raw_reference_points - mapped_references, dim=2
            )
            * inverse_sqrt_two
        ).clamp(0.0, 1.0)
        safe_support_fraction = torch.nan_to_num(
            support_fraction, nan=0.0
        ).clamp(0.0, 1.0)
        features = torch.stack(
            (
                pivot_distance,
                direction_disagreement,
                reference_distances.mean(dim=1),
                torch.amax(reference_distances, dim=1),
                safe_support_fraction,
            ),
            dim=1,
        )
        features = torch.where(valid[:, None], features, torch.zeros_like(features))
        routing_features = features.detach()
        raw_correction = self.correction_mlp(routing_features).squeeze(1)
        correction = self.maximum_logit_correction * torch.tanh(raw_correction)
        correction = torch.where(valid, correction, torch.zeros_like(correction))
        return {
            "feature_names": GEOMETRY_CONSISTENCY_FEATURE_NAMES,
            "features": features,
            "transform_valid": valid,
            "sarn_logit_correction": correction,
            "mapped_sarn_pivot": mapped_pivot,
            "mapped_sarn_direction_sin_cos": mapped_direction,
            "mapped_sarn_references": mapped["references"],
            "homography_condition_number": mapped[
                "homography_condition_number"
            ],
        }


class ConstrainedUncertaintyFusion(nn.Module):
    """Convex multi-view fusion with a protected original-view contribution."""

    def __init__(
        self,
        representation_dim: int,
        *,
        minimum_original_weight: float = 0.25,
    ) -> None:
        super().__init__()
        _require(representation_dim >= 1, "fusion representation is empty")
        _require(
            0.0 <= float(minimum_original_weight) < 1.0,
            "minimum original-view weight must be in [0,1)",
        )
        self.minimum_original_weight = float(minimum_original_weight)
        self.score = nn.Linear(representation_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        # Begin close to deterministic score fusion.  A large arbitrary
        # variance penalty before calibration would structurally prefer one
        # randomly initialized uncertainty head.
        self.raw_uncertainty_strength = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        representations: torch.Tensor,
        view_means: torch.Tensor,
        view_variances: torch.Tensor,
        available: torch.Tensor,
        *,
        use_uncertainty: bool,
        enabled: bool,
        use_learned_scores: bool = True,
        force_original: torch.Tensor | None = None,
        sarn_logit_adjustment: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _require(representations.ndim == 3, "representations must be BVD")
        batch, views, _features = representations.shape
        _require(
            view_means.shape == view_variances.shape == (batch, views),
            "view distribution shapes differ",
        )
        _require(
            available.shape == (batch, views) and available.dtype == torch.bool,
            "view availability must be boolean BV",
        )
        _require(bool(torch.all(available[:, 0])), "original view must be available")
        _require(
            bool(torch.isfinite(view_means).all())
            and bool(torch.isfinite(view_variances).all())
            and bool(torch.all(view_variances > 0.0)),
            "view distributions are invalid",
        )
        if force_original is None:
            force_original = torch.zeros(
                batch, dtype=torch.bool, device=representations.device
            )
        _require(
            force_original.shape == (batch,) and force_original.dtype == torch.bool,
            "force-original selector must be boolean B",
        )
        if sarn_logit_adjustment is not None:
            _require(views >= 2, "SARN logit adjustment requires a second view")
            _require(
                use_learned_scores,
                "SARN logit adjustment requires learned fusion scores",
            )
            _require(
                sarn_logit_adjustment.shape == (batch,)
                and sarn_logit_adjustment.device == representations.device
                and sarn_logit_adjustment.is_floating_point()
                and bool(torch.isfinite(sarn_logit_adjustment).all()),
                "SARN logit adjustment must be finite device-matched B",
            )

        original_only = F.one_hot(
            torch.zeros(batch, dtype=torch.long, device=representations.device),
            num_classes=views,
        ).to(dtype=representations.dtype)
        if use_learned_scores:
            logits = self.score(representations).squeeze(-1)
            if use_uncertainty:
                strength = F.softplus(self.raw_uncertainty_strength)
                logits = logits - strength * torch.log(view_variances)
            if sarn_logit_adjustment is not None:
                sarn_selector = F.one_hot(
                    torch.ones(batch, dtype=torch.long, device=logits.device),
                    num_classes=views,
                ).to(dtype=logits.dtype)
                logits = logits + sarn_selector * sarn_logit_adjustment[:, None]
            maximum = torch.amax(
                torch.where(available, logits, torch.full_like(logits, -torch.inf)),
                dim=1,
                keepdim=True,
            )
            unnormalized = torch.exp(logits - maximum) * available.to(logits.dtype)
            base_weights = unnormalized / unnormalized.sum(dim=1, keepdim=True)
            weights = (
                (1.0 - self.minimum_original_weight) * base_weights
                + self.minimum_original_weight * original_only
            )
        else:
            weights = available.to(dtype=representations.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
            _require(
                bool(torch.all(weights[:, 0] >= self.minimum_original_weight)),
                "uniform fusion conflicts with the minimum original-view weight",
            )
        if not enabled:
            force_original = torch.ones_like(force_original)
        weights = torch.where(force_original[:, None], original_only, weights)

        mean = torch.sum(weights * view_means, dim=1)
        variance = torch.sum(
            weights
            * (view_variances + (view_means - mean[:, None]).square()),
            dim=1,
        )
        # Select the already-computed original values to make the fail-closed
        # branch exact rather than merely algebraically equivalent.
        mean = torch.where(force_original, view_means[:, 0], mean)
        variance = torch.where(force_original, view_variances[:, 0], variance)
        return {
            "mean": mean,
            "variance": variance,
            "standard_deviation": torch.sqrt(variance),
            "weights": weights,
            "uncertainty_strength": (
                F.softplus(self.raw_uncertainty_strength).expand(batch)
                if use_uncertainty and use_learned_scores
                else torch.zeros(batch, dtype=mean.dtype, device=mean.device)
            ),
        }


class EfficientNetB0SupportGeometryMultiView(nn.Module):
    """Shared EfficientNet-B0 SGCA encoder with probabilistic view fusion."""

    def __init__(
        self,
        *,
        imagenet_pretrained: bool = True,
        representation_dim: int = 192,
        geometry_embedding_dim: int = 96,
        minimum_original_weight: float = 0.25,
        minimum_variance: float = 1.0e-4,
        maximum_variance: float = 0.25,
        default_ablation: ArchitectureAblation = DEFAULT_TWO_VIEW_ARCHITECTURE,
    ) -> None:
        super().__init__()
        _require(representation_dim >= 16, "representation is too small")
        _require(
            0.0 < float(minimum_variance) < float(maximum_variance),
            "variance bounds are invalid",
        )
        _require(
            isinstance(default_ablation, ArchitectureAblation),
            "default ablation is invalid",
        )
        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1
            if imagenet_pretrained
            else None
        )
        backbone = efficientnet_b0(weights=weights)
        self.encoder = SharedEfficientNetB0Encoder(backbone.features)
        self.middle_channels = EFFICIENTNET_B0_MIDDLE_FEATURES
        self.feature_channels = EFFICIENTNET_B0_FEATURES
        self.minimum_variance = float(minimum_variance)
        self.maximum_variance = float(maximum_variance)
        self.default_ablation = default_ablation

        self.support_geometry_attention = SupportGeometryConditionedAttention(
            self.middle_channels,
            geometry_embedding_dim=geometry_embedding_dim,
        )
        self.representation = nn.Sequential(
            nn.Linear(
                self.feature_channels + geometry_embedding_dim,
                representation_dim,
            ),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(representation_dim, representation_dim),
            nn.SiLU(),
        )
        # SGCA begins as a safe residual extension of the image representation.
        # These columns learn immediately, but random geometry embeddings do not
        # impose an output offset before any geometry supervision is observed.
        nn.init.zeros_(
            self.representation[0].weight[:, self.feature_channels :]
        )
        self.mean_head = nn.Linear(representation_dim, 1)
        self.variance_head = nn.Linear(representation_dim, 1)
        self.fusion = ConstrainedUncertaintyFusion(
            representation_dim,
            minimum_original_weight=minimum_original_weight,
        )

    @staticmethod
    def _validate_image(image: torch.Tensor, *, label: str) -> None:
        _require(
            image.ndim == 4 and image.shape[1] == 3,
            f"{label} must be Bx3xHxW",
        )
        _require(image.is_floating_point(), f"{label} must be floating point")
        _require(min(image.shape[-2:]) >= 32, f"{label} is too small")

    @staticmethod
    def _validate_selector(
        selector: torch.Tensor, *, batch: int, device: torch.device, label: str
    ) -> None:
        _require(
            selector.shape == (batch,)
            and selector.dtype == torch.bool
            and selector.device == device,
            f"{label} must be device-matched boolean B",
        )

    @staticmethod
    def _mask_and_validity(
        mask: torch.Tensor | None,
        *,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _channels, height, width = reference.shape
        if mask is None:
            value = torch.ones(
                batch,
                1,
                height,
                width,
                dtype=reference.dtype,
                device=reference.device,
            )
            return value, torch.ones(
                batch, dtype=torch.bool, device=reference.device
            )
        _require(
            mask.shape == (batch, 1, height, width)
            and mask.device == reference.device,
            "support mask must be device-matched Bx1xHxW",
        )
        value = mask.to(dtype=reference.dtype)
        finite = torch.isfinite(value).all(dim=(1, 2, 3))
        safe = torch.where(finite[:, None, None, None], value, torch.ones_like(value))
        safe = torch.clamp(safe, 0.0, 1.0)
        positive = safe.sum(dim=(1, 2, 3)) > torch.finfo(safe.dtype).eps
        valid = finite & positive
        safe = torch.where(valid[:, None, None, None], safe, torch.ones_like(safe))
        return safe, valid

    def forward(
        self,
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        rectified_view: torch.Tensor | None = None,
        rectified_support_mask: torch.Tensor | None = None,
        rectified_active: torch.Tensor | None = None,
        ablation: ArchitectureAblation | None = None,
        raw_to_sarn_homography: torch.Tensor | None = None,
        detach_geometry_from_encoder: bool = False,
        _geometry_consistency_fusion: (
            GeometryConsistencyReliabilityFusion | None
        ) = None,
        _return_view_representations: bool = False,
        _return_middle_features: bool = False,
    ) -> dict[str, Any]:
        settings = self.default_ablation if ablation is None else ablation
        _require(isinstance(settings, ArchitectureAblation), "ablation is invalid")
        _require(
            isinstance(detach_geometry_from_encoder, bool),
            "geometry stop-gradient selector must be boolean",
        )
        _require(
            isinstance(_return_view_representations, bool),
            "representation-return selector must be boolean",
        )
        _require(
            isinstance(_return_middle_features, bool),
            "middle-feature-return selector must be boolean",
        )
        if detach_geometry_from_encoder:
            _require(
                settings == CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
                "geometry stop-gradient is reserved for the A3G/A5 two-view side head",
            )
        self._validate_image(original_view, label="original view")
        self._validate_image(sarn_view, label="SARN view")
        _require(
            sarn_view.shape == original_view.shape
            and sarn_view.dtype == original_view.dtype
            and sarn_view.device == original_view.device,
            "original and SARN views must share shape, dtype, and device",
        )
        batch = original_view.shape[0]
        self._validate_selector(
            sarn_active,
            batch=batch,
            device=original_view.device,
            label="SARN availability",
        )
        _require(
            bool(torch.isfinite(original_view).all()),
            "original view is non-finite; fail-closed prediction is impossible",
        )

        sarn_mask, sarn_mask_valid = self._mask_and_validity(
            sarn_support_mask,
            reference=sarn_view,
        )
        sarn_finite = torch.isfinite(sarn_view).all(dim=(1, 2, 3))
        sarn_available = sarn_active & sarn_finite & sarn_mask_valid
        safe_sarn = torch.where(
            sarn_available[:, None, None, None], sarn_view, original_view
        )
        sarn_mask = torch.where(
            sarn_available[:, None, None, None],
            sarn_mask,
            torch.ones_like(sarn_mask),
        )

        if rectified_view is None:
            _require(
                rectified_support_mask is None and rectified_active is None,
                "rectified metadata was supplied without a rectified view",
            )
            safe_rectified = original_view
            rectified_mask = torch.ones_like(sarn_mask)
            rectified_available = torch.zeros_like(sarn_available)
        else:
            self._validate_image(rectified_view, label="rectified view")
            _require(
                rectified_view.shape == original_view.shape
                and rectified_view.dtype == original_view.dtype
                and rectified_view.device == original_view.device,
                "rectified and original views must share shape, dtype, and device",
            )
            _require(rectified_active is not None, "rectified availability is missing")
            self._validate_selector(
                rectified_active,
                batch=batch,
                device=original_view.device,
                label="rectified availability",
            )
            rectified_mask, rectified_mask_valid = self._mask_and_validity(
                rectified_support_mask,
                reference=rectified_view,
            )
            rectified_finite = torch.isfinite(rectified_view).all(dim=(1, 2, 3))
            rectified_available = (
                rectified_active
                & sarn_available
                & rectified_finite
                & rectified_mask_valid
            )
            if not settings.use_rectified_view:
                rectified_available = torch.zeros_like(rectified_available)
            safe_rectified = torch.where(
                rectified_available[:, None, None, None],
                rectified_view,
                original_view,
            )
            rectified_mask = torch.where(
                rectified_available[:, None, None, None],
                rectified_mask,
                torch.ones_like(rectified_mask),
            )

        # Encode every real view exactly once in one shared-backbone call.  An
        # unavailable/no-op view reuses its row's original feature afterwards;
        # it must not duplicate raw images in the BatchNorm population.
        encode_sarn = sarn_available & settings.use_multiview_fusion
        encode_rectified = rectified_available & settings.use_multiview_fusion
        sarn_indices = torch.nonzero(encode_sarn, as_tuple=False).flatten()
        rectified_indices = torch.nonzero(
            encode_rectified, as_tuple=False
        ).flatten()
        encoder_inputs = [original_view]
        if sarn_indices.numel() > 0:
            encoder_inputs.append(safe_sarn.index_select(0, sarn_indices))
        if rectified_indices.numel() > 0:
            encoder_inputs.append(
                safe_rectified.index_select(0, rectified_indices)
            )
        encoded_middle = self.encoder.encode_early(
            torch.cat(encoder_inputs, dim=0)
        )

        def restore_view_rows(
            encoded_views: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            base_positions = torch.arange(batch, device=original_view.device)
            original_rows = encoded_views[:batch]
            offset = batch
            restored: list[torch.Tensor] = [original_rows]
            for indices in (sarn_indices, rectified_indices):
                count = int(indices.numel())
                if count == 0:
                    restored.append(original_rows)
                    continue
                encoded_positions = torch.arange(
                    offset,
                    offset + count,
                    dtype=torch.long,
                    device=original_view.device,
                )
                mapping = base_positions.scatter(0, indices, encoded_positions)
                restored.append(encoded_views.index_select(0, mapping))
                offset += count
            _require(
                offset == encoded_views.shape[0],
                "encoded view accounting drifted",
            )
            return restored[0], restored[1], restored[2]

        original_middle, sarn_middle, rectified_middle = restore_view_rows(
            encoded_middle
        )
        middle_features = torch.cat(
            (original_middle, sarn_middle, rectified_middle), dim=0
        )
        middle_height, middle_width = middle_features.shape[-2:]
        masks = torch.cat(
            (
                torch.ones_like(sarn_mask),
                sarn_mask,
                rectified_mask,
            ),
            dim=0,
        )
        masks = F.interpolate(
            masks,
            size=(middle_height, middle_width),
            mode="area",
        )
        attention = self.support_geometry_attention(
            middle_features,
            masks,
            use_support_mask=settings.use_support_mask,
            use_geometry_attention=settings.use_geometry_attention,
            detach_geometry_from_features=detach_geometry_from_encoder,
        )

        geometry_active = torch.cat(
            (sarn_available, encode_sarn, encode_rectified), dim=0
        )
        if not settings.use_geometry_attention:
            geometry_active = torch.zeros_like(geometry_active)
        active_scale = geometry_active.to(middle_features.dtype)[
            :, None, None, None
        ]
        effective_attention_residual = (
            attention["attention_residual"] * active_scale
        )
        attended_middle = middle_features * (1.0 + effective_attention_residual)
        geometry_embedding = attention["geometry_embedding"] * geometry_active.to(
            attention["geometry_embedding"].dtype
        )[:, None]

        original_attended = attended_middle[:batch]
        sarn_attended = attended_middle[batch : 2 * batch]
        rectified_attended = attended_middle[2 * batch :]
        late_inputs = [original_attended]
        if sarn_indices.numel() > 0:
            late_inputs.append(sarn_attended.index_select(0, sarn_indices))
        if rectified_indices.numel() > 0:
            late_inputs.append(
                rectified_attended.index_select(0, rectified_indices)
            )
        encoded_final = self.encoder.encode_late(torch.cat(late_inputs, dim=0))
        original_final, sarn_final, rectified_final = restore_view_rows(
            encoded_final
        )
        final_features = torch.cat(
            (original_final, sarn_final, rectified_final), dim=0
        )
        final_masks = F.interpolate(
            attention["effective_support_mask"],
            size=final_features.shape[-2:],
            mode="area",
        )
        final_pooled = _masked_average(final_features, final_masks)
        representation = self.representation(
            torch.cat((final_pooled, geometry_embedding), dim=1)
        )
        view_means = torch.sigmoid(self.mean_head(representation).squeeze(1))
        variance_fraction = torch.sigmoid(
            self.variance_head(representation).squeeze(1)
        )
        view_variances = self.minimum_variance + (
            self.maximum_variance - self.minimum_variance
        ) * variance_fraction

        def as_views(value: torch.Tensor) -> torch.Tensor:
            return torch.stack(value.split(batch, dim=0), dim=1)

        view_representations = as_views(representation)
        view_means = as_views(view_means)
        view_variances = as_views(view_variances)
        available = torch.stack(
            (
                torch.ones_like(sarn_available),
                sarn_available,
                rectified_available,
            ),
            dim=1,
        )
        force_original = ~sarn_available
        geometry = {
            "pivot": as_views(attention["pivot"]),
            "direction_sin_cos": as_views(attention["direction_sin_cos"]),
            "references": as_views(attention["references"]),
            "reference_points": as_views(attention["reference_points"]),
            "support_fraction": as_views(attention["support_fraction"]),
            "attention_active": as_views(geometry_active),
            "attention_residual": as_views(effective_attention_residual),
        }
        geometry_consistency: dict[str, Any] | None = None
        sarn_logit_adjustment: torch.Tensor | None = None
        if _geometry_consistency_fusion is not None:
            _require(
                isinstance(
                    _geometry_consistency_fusion,
                    GeometryConsistencyReliabilityFusion,
                ),
                "geometry-consistency fusion module is invalid",
            )
            _require(
                settings == CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
                "geometry-consistency fusion requires the A3 two-view base",
            )
            geometry_consistency = _geometry_consistency_fusion(
                raw_pivot=geometry["pivot"][:, 0],
                raw_direction_sin_cos=geometry["direction_sin_cos"][:, 0],
                raw_references=geometry["references"][:, 0],
                sarn_pivot=geometry["pivot"][:, 1],
                sarn_direction_sin_cos=geometry["direction_sin_cos"][:, 1],
                sarn_references=geometry["references"][:, 1],
                support_fraction=sarn_mask.mean(dim=(1, 2, 3)),
                sarn_active=sarn_available,
                raw_to_sarn_homography=raw_to_sarn_homography,
            )
            sarn_logit_adjustment = geometry_consistency[
                "sarn_logit_correction"
            ]
        fused = self.fusion(
            view_representations,
            view_means,
            view_variances,
            available,
            use_uncertainty=settings.use_uncertainty_in_fusion,
            enabled=settings.use_multiview_fusion,
            use_learned_scores=settings.use_learned_fusion,
            force_original=force_original,
            sarn_logit_adjustment=sarn_logit_adjustment,
        )
        result = {
            **fused,
            "view_names": VIEW_NAMES,
            "view_means": view_means,
            "view_variances": view_variances,
            "view_available": available,
            "primary_mean": view_means[:, 0],
            "primary_variance": view_variances[:, 0],
            "geometry": geometry,
            "ablation": settings,
        }
        if geometry_consistency is not None:
            result["geometry_consistency"] = geometry_consistency
        # A6 is trained after the A3 expert has finished.  This private opt-in
        # exposes the already-computed representation without changing any
        # A1--A5 default output key, parameter, buffer, or numerical path.
        if _return_view_representations:
            result["view_representations"] = view_representations
        # G1 is a post-hoc spatial probe.  It receives the pre-attention
        # 112-channel map only through this private, default-off observation
        # path, so all historical model outputs and computations stay intact.
        if _return_middle_features:
            result["view_middle_features"] = torch.stack(
                middle_features.split(batch, dim=0), dim=1
            )
        return result


class EfficientNetB0GeometryConsistencyReliability(nn.Module):
    """Optional A5 wrapper: A3 predictions plus geometry-consistency routing."""

    def __init__(
        self,
        base_model: EfficientNetB0SupportGeometryMultiView | None = None,
        *,
        imagenet_pretrained: bool = True,
        consistency_hidden_dim: int = 32,
        maximum_logit_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.base_model = (
            EfficientNetB0SupportGeometryMultiView(
                imagenet_pretrained=imagenet_pretrained,
                default_ablation=CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
            )
            if base_model is None
            else base_model
        )
        _require(
            isinstance(self.base_model, EfficientNetB0SupportGeometryMultiView),
            "A5 base model is invalid",
        )
        self.geometry_consistency_fusion = GeometryConsistencyReliabilityFusion(
            hidden_dim=consistency_hidden_dim,
            maximum_logit_correction=maximum_logit_correction,
        )

    def forward(
        self,
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        result = self.base_model(
            original_view,
            sarn_view,
            sarn_support_mask,
            sarn_active=sarn_active,
            ablation=CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
            raw_to_sarn_homography=raw_to_sarn_homography,
            detach_geometry_from_encoder=True,
            _geometry_consistency_fusion=self.geometry_consistency_fusion,
        )
        result["architecture"] = A5_ARCHITECTURE
        result["a5_base_ablation"] = CAUSAL_A3_UNCERTAINTY_TWO_VIEW
        return result


__all__ = [
    "A5_ARCHITECTURE",
    "ARCHITECTURE",
    "CAUSAL_A1_RAW",
    "CAUSAL_A2_UNIFORM_TWO_VIEW",
    "CAUSAL_A3_UNCERTAINTY_TWO_VIEW",
    "CAUSAL_A4_SGCA_TWO_VIEW",
    "DEFAULT_TWO_VIEW_ARCHITECTURE",
    "EFFICIENTNET_B0_FEATURES",
    "EFFICIENTNET_B0_MIDDLE_FEATURES",
    "EFFICIENTNET_B0_SGCA_STAGE",
    "GEOMETRY_CONSISTENCY_FEATURE_NAMES",
    "FULL_ARCHITECTURE",
    "GEOMETRY_VALUES",
    "VIEW_NAMES",
    "ArchitectureAblation",
    "ConstrainedUncertaintyFusion",
    "EfficientNetB0GeometryConsistencyReliability",
    "EfficientNetB0SupportGeometryMultiView",
    "GeometryConsistencyReliabilityFusion",
    "SharedEfficientNetB0Encoder",
    "SupportGeometryConditionedAttention",
    "causal_ladder_ablations",
    "canonical_ablations",
]
