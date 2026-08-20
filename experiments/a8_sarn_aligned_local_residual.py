"""Frozen-A3 probe for SARN-aligned local residual feature fusion.

This is an unvalidated internal research candidate.  It does not route scalar
endpoint predictions.  A completed A3 expert supplies shared EfficientNet-B0
middle features for Raw and SARN.  The SARN map is sampled onto the Raw
receptive-field lattice with the known image-space homography, relation content
is restricted to common valid support, and a small local block predicts a
bounded residual on the Raw middle feature map.  The frozen shared late encoder
and regression heads then produce one direct reading.

Only the local residual block is trainable.  The A3 expert remains in eval mode
with every parameter and buffer frozen.  A zero-initialized output projection
starts the spatial path at the Raw-primary feature map.  The published probe
prediction keeps A3's fused prediction as an anchor and adds only the
A3-roster-matched late-encoder response to the local feature residual.  Consequently zero
residual is exactly A3 without routing a scalar mixture weight.  A missing
SARN view selects Raw exactly; an unusable relation transform keeps A3 exactly.
"""
from __future__ import annotations

import math
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.support_geometry_multiview_efficientnet import (
    CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
    EFFICIENTNET_B0_FEATURES,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
    EfficientNetB0SupportGeometryMultiView,
    _masked_average,
)


A8_ARCHITECTURE: Final[str] = (
    "Frozen-A3-Anchored-SARN-Aligned-Local-Residual-Probe"
)
MIDDLE_FEATURE_STRIDE: Final[int] = 16
DEFAULT_MAXIMUM_RESIDUAL_RATIO: Final[float] = 0.25
RELATION_COMPONENTS: Final[tuple[str, ...]] = (
    "raw",
    "aligned_sarn",
    "absolute_difference",
    "product",
    "aligned_support",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ReceptiveFieldHomographyAligner(nn.Module):
    """Align image-normalized SARN evidence to A3 middle RF centres."""

    def __init__(
        self,
        *,
        feature_stride: int = MIDDLE_FEATURE_STRIDE,
        maximum_condition_number: float = 1.0e5,
    ) -> None:
        super().__init__()
        _require(feature_stride >= 1, "feature stride must be positive")
        _require(
            math.isfinite(maximum_condition_number)
            and maximum_condition_number >= 10.0,
            "homography condition-number bound is invalid",
        )
        self.feature_stride = int(feature_stride)
        self.maximum_condition_number = float(maximum_condition_number)

    def _raw_rf_lattice(
        self,
        batch: int,
        feature_hw: tuple[int, int],
        input_hw: tuple[int, int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        feature_height, feature_width = feature_hw
        input_height, input_width = input_hw
        _require(
            input_height >= 2
            and input_width >= 2
            and feature_height == math.ceil(input_height / self.feature_stride)
            and feature_width == math.ceil(input_width / self.feature_stride),
            "input and A3 middle-feature lattice sizes are inconsistent",
        )
        y = (
            torch.arange(feature_height, dtype=torch.float32, device=device)
            * self.feature_stride
            / float(input_height - 1)
        )
        x = (
            torch.arange(feature_width, dtype=torch.float32, device=device)
            * self.feature_stride
            / float(input_width - 1)
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack(
            (xx, yy, torch.ones_like(xx)), dim=-1
        ).reshape(1, feature_height * feature_width, 3).expand(batch, -1, -1)

    def sampling_grid(
        self,
        *,
        reference: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
        input_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        _require(
            reference.ndim == 4 and reference.is_floating_point(),
            "alignment reference must be floating-point BCHW",
        )
        batch, _channels, feature_height, feature_width = reference.shape
        _require(
            feature_height >= 2
            and feature_width >= 2
            and raw_to_sarn_homography.shape == (batch, 3, 3)
            and raw_to_sarn_homography.device == reference.device
            and raw_to_sarn_homography.is_floating_point(),
            "raw-to-SARN homography must be device-matched floating Bx3x3",
        )
        with torch.autocast(device_type=reference.device.type, enabled=False):
            homography = raw_to_sarn_homography.detach().float()
            matrix_finite = torch.isfinite(homography).all(dim=(1, 2))
            identity = torch.eye(
                3, dtype=torch.float32, device=reference.device
            ).expand(batch, -1, -1)
            finite_homography = torch.where(
                matrix_finite[:, None, None], homography, identity
            )
            norm = torch.linalg.matrix_norm(
                finite_homography, ord="fro", dim=(1, 2)
            )
            normalized = finite_homography / norm.clamp_min(1.0e-12)[
                :, None, None
            ]
            singular_values = torch.linalg.svdvals(normalized)
            condition_number = singular_values[:, 0] / singular_values[
                :, -1
            ].clamp_min(1.0e-12)
            matrix_valid = (
                matrix_finite
                & (norm > 1.0e-8)
                & (singular_values[:, -1] > 1.0e-7)
                & torch.isfinite(condition_number)
                & (condition_number <= self.maximum_condition_number)
            )
            safe_homography = torch.where(
                matrix_valid[:, None, None], homography, identity
            )
            raw_points = self._raw_rf_lattice(
                batch,
                (feature_height, feature_width),
                tuple(int(value) for value in input_hw),
                device=reference.device,
            )
            mapped_homogeneous = torch.einsum(
                "bij,bnj->bni", safe_homography, raw_points
            )
            denominator = mapped_homogeneous[..., 2]
            denominator_valid = torch.all(
                torch.abs(denominator) > 1.0e-7, dim=1
            )
            safe_denominator = torch.where(
                torch.abs(denominator) > 1.0e-7,
                denominator,
                torch.ones_like(denominator),
            )
            mapped_input = (
                mapped_homogeneous[..., :2] / safe_denominator[..., None]
            )
            mapped_finite = torch.isfinite(mapped_input).all(dim=(1, 2))
            transform_valid = matrix_valid & denominator_valid & mapped_finite
            input_height, input_width = (int(value) for value in input_hw)
            feature_x = (
                mapped_input[..., 0]
                * float(input_width - 1)
                / self.feature_stride
            )
            feature_y = (
                mapped_input[..., 1]
                * float(input_height - 1)
                / self.feature_stride
            )
            feature_normalized = torch.stack(
                (
                    feature_x / float(feature_width - 1),
                    feature_y / float(feature_height - 1),
                ),
                dim=2,
            )
            inside = (
                (feature_normalized[..., 0] >= 0.0)
                & (feature_normalized[..., 0] <= 1.0)
                & (feature_normalized[..., 1] >= 0.0)
                & (feature_normalized[..., 1] <= 1.0)
            )
            grid = (2.0 * feature_normalized - 1.0).reshape(
                batch, feature_height, feature_width, 2
            )
            identity_y = torch.linspace(
                -1.0,
                1.0,
                feature_height,
                dtype=torch.float32,
                device=reference.device,
            )
            identity_x = torch.linspace(
                -1.0,
                1.0,
                feature_width,
                dtype=torch.float32,
                device=reference.device,
            )
            identity_yy, identity_xx = torch.meshgrid(
                identity_y, identity_x, indexing="ij"
            )
            identity_grid = torch.stack(
                (identity_xx, identity_yy), dim=2
            ).expand(batch, -1, -1, -1)
            grid = torch.where(
                transform_valid[:, None, None, None], grid, identity_grid
            )
            inside = inside & transform_valid[:, None]
        return {
            "grid": grid,
            "inside": inside.reshape(
                batch, 1, feature_height, feature_width
            ),
            "transform_valid": transform_valid,
            "condition_number": condition_number,
        }

    def forward(
        self,
        *,
        sarn_features: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
        input_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        _require(
            sarn_features.ndim == 4
            and sarn_features.is_floating_point()
            and bool(torch.isfinite(sarn_features).all()),
            "SARN middle features must be finite floating BCHW",
        )
        batch, _channels, feature_height, feature_width = sarn_features.shape
        _require(
            sarn_support_mask.ndim == 4
            and sarn_support_mask.shape[:2] == (batch, 1)
            and sarn_support_mask.device == sarn_features.device,
            "SARN support mask must be device-matched Bx1xHxW",
        )
        geometry = self.sampling_grid(
            reference=sarn_features,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        support = torch.nan_to_num(
            sarn_support_mask.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        support = F.interpolate(
            support,
            size=(feature_height, feature_width),
            mode="area",
        )
        with torch.autocast(device_type=sarn_features.device.type, enabled=False):
            aligned_sarn = F.grid_sample(
                sarn_features.detach().float(),
                geometry["grid"],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            aligned_support = F.grid_sample(
                support,
                geometry["grid"],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).clamp(0.0, 1.0)
        common_support = aligned_support * geometry["inside"].to(
            aligned_support.dtype
        )
        support_valid = common_support.sum(dim=(1, 2, 3)) > 1.0e-6
        return {
            **geometry,
            "aligned_sarn": aligned_sarn,
            "aligned_support": aligned_support,
            "common_support": common_support,
            "support_valid": support_valid,
        }


class LocalResidualFusionBlock(nn.Module):
    """A depthwise-separable local relation block with bounded output."""

    def __init__(
        self,
        *,
        feature_channels: int = EFFICIENTNET_B0_MIDDLE_FEATURES,
        hidden_channels: int = 48,
        maximum_residual_ratio: float = DEFAULT_MAXIMUM_RESIDUAL_RATIO,
    ) -> None:
        super().__init__()
        _require(feature_channels >= 1, "feature channels are empty")
        _require(
            hidden_channels >= 8 and hidden_channels % 8 == 0,
            "hidden channels must be a positive multiple of eight",
        )
        _require(
            math.isfinite(maximum_residual_ratio)
            and maximum_residual_ratio > 0.0,
            "maximum relative residual is invalid",
        )
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.maximum_residual_ratio = float(maximum_residual_ratio)
        relation_channels = 4 * self.feature_channels + 1
        self.input_projection = nn.Sequential(
            nn.Conv2d(relation_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.local_relation = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.output_projection = nn.Conv2d(
            hidden_channels, self.feature_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        *,
        raw_features: torch.Tensor,
        aligned_sarn_features: torch.Tensor,
        common_support: torch.Tensor,
        residual_available: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        _require(
            raw_features.shape == aligned_sarn_features.shape
            and raw_features.ndim == 4
            and raw_features.shape[1] == self.feature_channels,
            "Raw/aligned-SARN middle feature shapes differ",
        )
        batch = raw_features.shape[0]
        _require(
            common_support.shape
            == (batch, 1, raw_features.shape[2], raw_features.shape[3])
            and residual_available.shape == (batch,)
            and residual_available.dtype == torch.bool,
            "local residual support/availability shapes differ",
        )
        raw = raw_features.detach().float()
        aligned = aligned_sarn_features.detach().float()
        support = common_support.detach().float().clamp(0.0, 1.0)
        _require(
            bool(torch.isfinite(raw).all())
            and bool(torch.isfinite(aligned).all())
            and bool(torch.isfinite(support).all()),
            "local residual evidence is non-finite",
        )
        active_support = support * residual_available.to(support.dtype)[
            :, None, None, None
        ]
        raw_supported = raw * active_support
        aligned_supported = aligned * active_support
        absolute_difference = torch.abs(aligned - raw) * active_support
        product = aligned * raw * active_support
        relation = torch.cat(
            (
                raw_supported,
                aligned_supported,
                absolute_difference,
                product,
                active_support,
            ),
            dim=1,
        )
        hidden = self.local_relation(self.input_projection(relation))
        raw_residual = self.output_projection(hidden)
        # A relative bound adapts to the frozen expert's local feature scale;
        # it is not a dataset- or condition-specific magnitude threshold.
        raw_local_rms = torch.sqrt(raw.square().mean(dim=1, keepdim=True))
        bounded_residual = (
            self.maximum_residual_ratio
            * raw_local_rms
            * torch.tanh(raw_residual.float())
            * active_support
        )
        return {
            "relation_component_names": RELATION_COMPONENTS,
            "relation": relation,
            "raw_residual": raw_residual,
            "raw_local_rms": raw_local_rms,
            "bounded_residual": bounded_residual,
            "active_support": active_support,
        }


class FrozenA3LocalResidualProbe(nn.Module):
    """Frozen A3 plus one trainable SARN-aligned local feature residual."""

    def __init__(
        self,
        base_model: EfficientNetB0SupportGeometryMultiView,
        *,
        residual_block: LocalResidualFusionBlock | None = None,
        aligner: ReceptiveFieldHomographyAligner | None = None,
    ) -> None:
        super().__init__()
        _require(
            isinstance(base_model, EfficientNetB0SupportGeometryMultiView),
            "A8 requires an EfficientNet-B0 A3 expert",
        )
        self.base_model = base_model
        self.residual_block = (
            LocalResidualFusionBlock()
            if residual_block is None
            else residual_block
        )
        self.aligner = (
            ReceptiveFieldHomographyAligner()
            if aligner is None
            else aligner
        )
        _require(
            self.residual_block.feature_channels
            == EFFICIENTNET_B0_MIDDLE_FEATURES,
            "A8 residual block channels differ from A3 middle features",
        )
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True) -> "FrozenA3LocalResidualProbe":
        super().train(mode)
        self.base_model.eval()
        return self

    def _regress_matched_middle(
        self,
        fused_middle: torch.Tensor,
        sarn_middle: torch.Tensor,
        sarn_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Regress fused Raw rows in A3's exact late-encoder batch roster.

        A3 sends ``[Raw B, available SARN A]`` through its frozen tail.  A8
        replaces only the first B rows by fused Raw maps and preserves the same
        SARN rows, shape, and order.  At zero residual the complete tail input
        is identical to A3, avoiding batch-shape-dependent CUDA rounding while
        gradients still flow through the fused Raw rows.
        """

        _require(
            fused_middle.shape == sarn_middle.shape
            and sarn_available.shape == (fused_middle.shape[0],)
            and sarn_available.dtype == torch.bool,
            "A8 matched late-roster evidence differs",
        )
        batch = fused_middle.shape[0]
        active_indices = torch.nonzero(sarn_available, as_tuple=False).flatten()
        # The outer expert and local block may use BF16, but the causal
        # correction is a difference of two nearby frozen-tail predictions.
        # Force both matched calls to FP32 so one BF16 ulp (~0.0039 near 0.5)
        # cannot masquerade as a learned residual effect.
        with torch.autocast(device_type=fused_middle.device.type, enabled=False):
            late_middle = torch.cat(
                (
                    fused_middle.float(),
                    sarn_middle.detach().float().index_select(0, active_indices),
                ),
                dim=0,
            )
            final_features = self.base_model.encoder.encode_late(late_middle)
            _require(
                final_features.shape[1] == EFFICIENTNET_B0_FEATURES,
                "A8 late feature channels differ from A3",
            )
            final_support = torch.ones(
                final_features.shape[0],
                1,
                final_features.shape[2],
                final_features.shape[3],
                dtype=final_features.dtype,
                device=final_features.device,
            )
            # Mirror A3's exact Raw pooling expression rather than replacing it
            # with Tensor.mean, whose reduction rounding need not be identical.
            pooled = _masked_average(final_features, final_support)
            geometry_dim = (
                int(self.base_model.representation[0].in_features)
                - EFFICIENTNET_B0_FEATURES
            )
            _require(
                geometry_dim >= 0,
                "A3 representation geometry width is invalid",
            )
            geometry = torch.zeros(
                pooled.shape[0],
                geometry_dim,
                dtype=pooled.dtype,
                device=pooled.device,
            )
            representation = self.base_model.representation(
                torch.cat((pooled, geometry), dim=1)
            )
            mean = torch.sigmoid(
                self.base_model.mean_head(representation).squeeze(1)
            )
            variance_fraction = torch.sigmoid(
                self.base_model.variance_head(representation).squeeze(1)
            )
            variance = self.base_model.minimum_variance + (
                self.base_model.maximum_variance
                - self.base_model.minimum_variance
            ) * variance_fraction
        return {
            "fused_mean": mean[:batch],
            "fused_variance": variance[:batch],
            "fused_representation": representation[:batch],
            "late_roster_rows": torch.tensor(
                final_features.shape[0],
                dtype=torch.int64,
                device=final_features.device,
            ),
        }

    def forward(
        self,
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        self.base_model.eval()
        with torch.no_grad():
            expert_outputs = self.base_model(
                original_view,
                sarn_view,
                sarn_support_mask,
                sarn_active=sarn_active,
                ablation=CAUSAL_A3_UNCERTAINTY_TWO_VIEW,
                raw_to_sarn_homography=raw_to_sarn_homography,
                _return_view_representations=True,
                _return_middle_features=True,
            )
            middle = expert_outputs.pop("view_middle_features").detach()
            # These weights decompose only the frozen A3 anchor.  A8's final
            # mean can leave the Raw/SARN endpoint hull, so exposing them as
            # unqualified top-level weights would be semantically misleading.
            a3_anchor_weights = expert_outputs.pop("weights").detach()
        batch = original_view.shape[0]
        _require(
            middle.shape[:3]
            == (batch, 3, EFFICIENTNET_B0_MIDDLE_FEATURES),
            "A3 middle feature roster is invalid",
        )
        raw_middle = middle[:, 0]
        sarn_middle = middle[:, 1]
        alignment = self.aligner(
            sarn_features=sarn_middle,
            sarn_support_mask=sarn_support_mask,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=tuple(int(value) for value in original_view.shape[-2:]),
        )
        sarn_available = expert_outputs["view_available"][:, 1]
        residual_available = (
            sarn_available
            & alignment["transform_valid"]
            & alignment["support_valid"]
        )
        residual_outputs = self.residual_block(
            raw_features=raw_middle,
            aligned_sarn_features=alignment["aligned_sarn"],
            common_support=alignment["common_support"],
            residual_available=residual_available,
        )
        residual = residual_outputs["bounded_residual"]
        fused_middle = raw_middle.float() + residual
        spatial = self._regress_matched_middle(
            fused_middle, sarn_middle, sarn_available
        )
        with torch.no_grad():
            spatial_reference = self._regress_matched_middle(
                raw_middle.float(), sarn_middle, sarn_available
            )
        raw_mean_correction = (
            spatial["fused_mean"] - spatial_reference["fused_mean"]
        )
        raw_variance_correction = (
            spatial["fused_variance"]
            - spatial_reference["fused_variance"]
        )
        exact_zero_residual = torch.all(
            residual.detach() == 0,
            dim=(1, 2, 3),
        )
        # Replaying a frozen CUDA tail can differ by a few ulps even for the
        # same input.  Cancel that detached numerical offset only on the exact
        # zero-residual path.  Forward correction is then bit-zero while its
        # derivative remains that of the spatial branch, so zero initialization
        # does not reproduce the dead-gradient ``where(A3, spatial)`` failure.
        mean_correction = raw_mean_correction - torch.where(
            exact_zero_residual,
            raw_mean_correction.detach(),
            torch.zeros_like(raw_mean_correction),
        )
        variance_correction = raw_variance_correction - torch.where(
            exact_zero_residual,
            raw_variance_correction.detach(),
            torch.zeros_like(raw_variance_correction),
        )
        anchored_mean = torch.clamp(
            expert_outputs["mean"].detach() + mean_correction,
            min=0.0,
            max=1.0,
        )
        anchored_variance = torch.clamp(
            expert_outputs["variance"].detach() + variance_correction,
            # Do not use dtype epsilon here: BF16 epsilon is ~7.8e-3 and would
            # alter a valid small A3 variance even when correction is zero.
            min=1.0e-8,
            max=1.0,
        )
        # View absence is the only case that discards A3 fusion.  A valid SARN
        # endpoint with an unusable relation transform remains exact A3 because
        # the residual/correction is zero, preserving established semantics.
        mean = torch.where(
            sarn_available, anchored_mean, expert_outputs["primary_mean"]
        )
        variance = torch.where(
            sarn_available,
            anchored_variance,
            expert_outputs["primary_variance"],
        )
        standard_deviation = torch.sqrt(variance)
        a3_standard_deviation = torch.sqrt(expert_outputs["variance"])
        raw_standard_deviation = torch.sqrt(expert_outputs["primary_variance"])
        standard_deviation = torch.where(
            sarn_available,
            torch.where(
                exact_zero_residual,
                a3_standard_deviation,
                standard_deviation,
            ),
            raw_standard_deviation,
        )
        return {
            **expert_outputs,
            "architecture": A8_ARCHITECTURE,
            "mean": mean,
            "variance": variance,
            "standard_deviation": standard_deviation,
            "a3_fused": {
                "mean": expert_outputs["mean"],
                "variance": expert_outputs["variance"],
                "anchor_weights": a3_anchor_weights,
            },
            "a3_anchor_weights": a3_anchor_weights,
            "raw_endpoint": {
                "mean": expert_outputs["primary_mean"],
                "variance": expert_outputs["primary_variance"],
                "representation": expert_outputs["view_representations"][:, 0],
            },
            "sarn_endpoint": {
                "mean": expert_outputs["view_means"][:, 1],
                "variance": expert_outputs["view_variances"][:, 1],
                "representation": expert_outputs["view_representations"][:, 1],
                "available": sarn_available,
            },
            "spatial_prediction": {
                "mean": spatial["fused_mean"],
                "variance": spatial["fused_variance"],
                "raw_reference_mean": spatial_reference["fused_mean"],
                "raw_reference_variance": spatial_reference["fused_variance"],
            },
            "anchor_correction": {
                "mean": mean_correction,
                "variance": variance_correction,
                "uncancelled_mean": raw_mean_correction,
                "uncancelled_variance": raw_variance_correction,
                "exact_zero_residual": exact_zero_residual,
                "anchored_mean": anchored_mean,
                "anchored_variance": anchored_variance,
            },
            "local_residual_probe": {
                "sarn_available": sarn_available,
                "residual_available": residual_available,
                "transform_valid": alignment["transform_valid"],
                "support_valid": alignment["support_valid"],
                "homography_condition_number": alignment["condition_number"],
                "aligned_support": alignment["aligned_support"],
                "common_support": alignment["common_support"],
                "aligned_sarn_middle": alignment["aligned_sarn"],
                "bounded_residual": residual,
                "raw_residual": residual_outputs["raw_residual"],
                "raw_local_rms": residual_outputs["raw_local_rms"],
                "active_support": residual_outputs["active_support"],
                "fused_middle": fused_middle,
                "fused_representation": spatial["fused_representation"],
                "raw_reference_representation": spatial_reference[
                    "fused_representation"
                ],
                "matched_late_roster_rows": spatial["late_roster_rows"],
            },
        }


def parameter_counts(model: FrozenA3LocalResidualProbe) -> dict[str, int]:
    _require(isinstance(model, FrozenA3LocalResidualProbe), "target is not A8")
    expert = sum(parameter.numel() for parameter in model.base_model.parameters())
    residual = sum(
        parameter.numel() for parameter in model.residual_block.parameters()
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "expert": int(expert),
        "residual_block": int(residual),
        "trainable": int(trainable),
    }


def estimated_probe_activation_bytes(
    *,
    batch_size: int,
    feature_height: int = 16,
    feature_width: int = 16,
    bytes_per_value: int = 4,
    hidden_channels: int = 48,
) -> int:
    """Approximate extra A8 relation/residual activation storage."""

    _require(
        batch_size >= 1
        and feature_height >= 1
        and feature_width >= 1
        and bytes_per_value >= 1
        and hidden_channels >= 1,
        "activation estimate inputs are invalid",
    )
    relation_channels = 4 * EFFICIENTNET_B0_MIDDLE_FEATURES + 1
    values = batch_size * feature_height * feature_width * (
        relation_channels
        + 2 * hidden_channels
        + 3 * EFFICIENTNET_B0_MIDDLE_FEATURES
        + 3
    )
    return int(values * bytes_per_value)


__all__ = [
    "A8_ARCHITECTURE",
    "DEFAULT_MAXIMUM_RESIDUAL_RATIO",
    "MIDDLE_FEATURE_STRIDE",
    "RELATION_COMPONENTS",
    "FrozenA3LocalResidualProbe",
    "LocalResidualFusionBlock",
    "ReceptiveFieldHomographyAligner",
    "estimated_probe_activation_bytes",
    "parameter_counts",
]
