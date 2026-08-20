"""Projective Geometry-Guided Support-Invariant Attention (PG-SIAM).

PG-SIAM is a small continuation of the frozen DB-GAR18 parent.  It consumes
two views of the same ROI:

``view_a``
    The SARN bounding-box-normalized view used by the frozen parent.
``view_b``
    The trusted SARN quadrilateral rectified to the full image rectangle.

Both views pass through one shared DB-GAR18 backbone.  Rectified features are
sampled back into view-A coordinates with the supplied input-pixel homography.
Only two zero-initialized diagonal adapters (256 + 512 = 768 parameters) can
mix the aligned view discrepancy at layers 3 and 4.  The old state and BatchNorm
statistics remain frozen.  Clean inputs, rejected geometry, numerical failure,
and zero initialization all return the parent prediction exactly.

The matrix contract is explicit: ``homography_a_to_b`` maps pixel-center
coordinates in ``view_a`` to pixel-center coordinates in ``view_b``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.geoattn_resnet18_progress import GeoAttnResNet18
from experiments.support_aware_roi_normalization_v2 import MIN_CONFIDENCE


PROTOCOL: Final[str] = "projective_geometry_guided_siam_db_gar18_v3"
ARCHITECTURE: Final[str] = "Projective-Geometry-Guided-SIAM-DB-GAR18"
METHOD_PREFIX: Final[str] = "pg_siam_db_gar18"
LAYER3_CHANNELS: Final[int] = 256
LAYER4_CHANNELS: Final[int] = 512
TRAINABLE_PARAMETERS: Final[int] = LAYER3_CHANNELS + LAYER4_CHANNELS
TRUST_REGION_RADIUS: Final[float] = 0.05
MIN_ALIGNED_FRACTION: Final[float] = 0.05
HOMOGRAPHY_EPSILON: Final[float] = 1e-7


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TwoViewFeatures:
    """Detached shared-backbone features for a corresponding two-view pair."""

    layer3_a: torch.Tensor
    layer4_a: torch.Tensor
    layer3_b: torch.Tensor
    layer4_b: torch.Tensor
    parent_progress: torch.Tensor
    rectified_progress: torch.Tensor


class ProjectiveGeometryGuidedSIAM(GeoAttnResNet18):
    """Shared DB-GAR18 with two homography-aligned diagonal adapters."""

    def __init__(
        self,
        *,
        imagenet_pretrained: bool = False,
        trust_region_radius: float = TRUST_REGION_RADIUS,
    ) -> None:
        super().__init__(imagenet_pretrained=imagenet_pretrained)
        _require(float(trust_region_radius) > 0.0, "trust-region radius must be positive")
        self.trust_region_radius = float(trust_region_radius)
        self.layer3_adapter_gain = nn.Parameter(torch.zeros(LAYER3_CHANNELS))
        self.layer4_adapter_gain = nn.Parameter(torch.zeros(LAYER4_CHANNELS))

    def train(self, mode: bool = True) -> "ProjectiveGeometryGuidedSIAM":
        """Keep all inherited BatchNorm statistics frozen during continuation."""

        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            # The auxiliary head is frozen too.  Keeping its Dropout disabled
            # makes its supervision a deterministic function of the two
            # adapter gains and preserves the parent-training-output contract.
            self.geometry_head.eval()
        return self

    def _to_layer3(self, image: torch.Tensor) -> torch.Tensor:
        value = self.backbone.conv1(image)
        value = self.backbone.bn1(value)
        value = self.backbone.relu(value)
        value = self.backbone.maxpool(value)
        value = self.backbone.layer1(value)
        value = self.backbone.layer2(value)
        return self.backbone.layer3(value)

    def _to_layer4(self, layer3: torch.Tensor) -> torch.Tensor:
        return self.backbone.layer4(layer3)

    def _progress_from_layer4(self, layer4: torch.Tensor) -> torch.Tensor:
        features = self.backbone.avgpool(layer4)
        features = torch.flatten(features, 1)
        features = self.backbone.fc(features)
        return torch.sigmoid(self.progress_head(features).squeeze(1))

    def _geometry_from_layer4(self, layer4: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.avgpool(layer4)
        features = torch.flatten(features, 1)
        features = self.backbone.fc(features)
        geometry = self.geometry_head(features)
        return {
            "pivot": torch.sigmoid(geometry[:, 0:2]),
            "direction_sin_cos": F.normalize(
                geometry[:, 2:4], dim=1, eps=1e-6
            ),
            "references": torch.sigmoid(geometry[:, 4:8]),
        }

    @staticmethod
    def _select_rows(
        active: torch.Tensor,
        candidate: torch.Tensor,
        parent: torch.Tensor,
    ) -> torch.Tensor:
        _require(candidate.shape == parent.shape, "candidate/parent shape mismatch")
        _require(
            active.ndim == 1 and active.shape[0] == candidate.shape[0],
            "row selector must be B",
        )
        shape = (active.shape[0],) + (1,) * (candidate.ndim - 1)
        return torch.where(active.reshape(shape), candidate, parent)

    def _package_pair_result(
        self,
        *,
        parent_progress: torch.Tensor,
        candidate_progress: torch.Tensor,
        trusted_progress: torch.Tensor,
        effective_gate: torch.Tensor,
        effective_active: torch.Tensor,
        parent_layer4: torch.Tensor,
        candidate_layer4: torch.Tensor | None,
        include_geometry: bool,
    ) -> dict[str, torch.Tensor]:
        result = {
            "parent_progress": parent_progress,
            "candidate_progress": candidate_progress,
            "trusted_progress": trusted_progress,
            "effective_gate": effective_gate,
        }
        if not include_geometry:
            return result

        with torch.no_grad():
            parent_geometry = self._geometry_from_layer4(parent_layer4)
        if candidate_layer4 is None:
            candidate_geometry = parent_geometry
        else:
            raw_candidate_geometry = self._geometry_from_layer4(candidate_layer4)
            candidate_geometry = {
                key: self._select_rows(
                    effective_active, raw_candidate_geometry[key], parent_geometry[key]
                )
                for key in parent_geometry
            }
        # Standard task keys make this result directly consumable by the
        # existing geometry objective.  Explicit aliases keep parent/candidate
        # provenance auditable for tests and mechanism analysis.
        result.update(
            {
                "progress": trusted_progress,
                **candidate_geometry,
                "candidate_pivot": candidate_geometry["pivot"],
                "candidate_direction_sin_cos": candidate_geometry[
                    "direction_sin_cos"
                ],
                "candidate_references": candidate_geometry["references"],
                "parent_pivot": parent_geometry["pivot"],
                "parent_direction_sin_cos": parent_geometry[
                    "direction_sin_cos"
                ],
                "parent_references": parent_geometry["references"],
            }
        )
        return result

    @staticmethod
    def reliability_gate(confidence: torch.Tensor) -> torch.Tensor:
        denominator = max(1.0 - float(MIN_CONFIDENCE), HOMOGRAPHY_EPSILON)
        return torch.clamp(
            (confidence - float(MIN_CONFIDENCE)) / denominator, 0.0, 1.0
        )

    @staticmethod
    def _homography_validity(homography: torch.Tensor) -> torch.Tensor:
        """Scale-invariant finite/non-singular check, evaluated per sample."""

        _require(
            homography.ndim == 3 and homography.shape[1:] == (3, 3),
            "homography must be Bx3x3",
        )
        matrix = homography.detach().to(dtype=torch.float32)
        finite = torch.isfinite(matrix).all(dim=(1, 2))
        safe = torch.where(
            finite[:, None, None], matrix, torch.eye(3, device=matrix.device)[None]
        )
        scale = torch.amax(torch.abs(safe), dim=(1, 2)).clamp_min(HOMOGRAPHY_EPSILON)
        normalized = safe / scale[:, None, None]
        determinant = torch.linalg.det(normalized)
        return finite & torch.isfinite(determinant) & (torch.abs(determinant) > HOMOGRAPHY_EPSILON)

    @staticmethod
    def align_feature_homography(
        feature_b: torch.Tensor,
        homography_a_to_b: torch.Tensor,
        *,
        input_a_hw: tuple[int, int],
        input_b_hw: tuple[int, int],
        output_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a B feature map at A lattice points.

        Returns ``(aligned_b_in_a, valid_mask, matrix_valid)``.  The validity
        mask is Bx1xHoutxWout.  Invalid matrices are replaced by identity only
        for safe computation and receive an all-zero validity mask.
        """

        _require(feature_b.ndim == 4, "feature map must be BCHW")
        batch = feature_b.shape[0]
        _require(
            homography_a_to_b.ndim == 3
            and homography_a_to_b.shape == (batch, 3, 3),
            "feature/homography batch mismatch",
        )
        input_a_height, input_a_width = (int(input_a_hw[0]), int(input_a_hw[1]))
        input_b_height, input_b_width = (int(input_b_hw[0]), int(input_b_hw[1]))
        output_height, output_width = (int(output_hw[0]), int(output_hw[1]))
        _require(
            min(
                input_a_height,
                input_a_width,
                input_b_height,
                input_b_width,
                output_height,
                output_width,
            )
            >= 2,
            "homography alignment dimensions must be at least two",
        )
        _require(
            feature_b.device == homography_a_to_b.device,
            "feature and homography devices differ",
        )

        matrix = homography_a_to_b.detach().to(dtype=torch.float32)
        matrix_valid = ProjectiveGeometryGuidedSIAM._homography_validity(matrix)
        identity = torch.eye(3, dtype=torch.float32, device=matrix.device)
        safe_matrix = torch.where(
            matrix_valid[:, None, None], matrix, identity[None]
        )

        # align_corners=True makes the feature-lattice endpoints correspond to
        # input pixel centers 0 and size-1, so one input-space H can be reused
        # without stage-specific matrix heuristics.
        y_a = torch.linspace(
            0.0,
            float(input_a_height - 1),
            output_height,
            dtype=torch.float32,
            device=feature_b.device,
        )
        x_a = torch.linspace(
            0.0,
            float(input_a_width - 1),
            output_width,
            dtype=torch.float32,
            device=feature_b.device,
        )
        yy, xx = torch.meshgrid(y_a, x_a, indexing="ij")
        points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
        mapped = torch.einsum("bij,hwj->bhwi", safe_matrix, points)
        denominator = mapped[..., 2]
        safe_denominator = torch.where(
            torch.abs(denominator) > HOMOGRAPHY_EPSILON,
            denominator,
            torch.ones_like(denominator),
        )
        x_b = mapped[..., 0] / safe_denominator
        y_b = mapped[..., 1] / safe_denominator
        grid_x = 2.0 * x_b / float(input_b_width - 1) - 1.0
        grid_y = 2.0 * y_b / float(input_b_height - 1) - 1.0
        finite_grid = torch.isfinite(grid_x) & torch.isfinite(grid_y)
        denominator_valid = torch.abs(denominator) > HOMOGRAPHY_EPSILON
        inside = (
            (grid_x >= -1.0)
            & (grid_x <= 1.0)
            & (grid_y >= -1.0)
            & (grid_y <= 1.0)
        )
        valid = finite_grid & denominator_valid & inside & matrix_valid[:, None, None]
        safe_grid_x = torch.where(finite_grid, grid_x, torch.full_like(grid_x, 2.0))
        safe_grid_y = torch.where(finite_grid, grid_y, torch.full_like(grid_y, 2.0))
        grid = torch.stack((safe_grid_x, safe_grid_y), dim=-1)
        sampled = F.grid_sample(
            feature_b,
            grid.to(dtype=feature_b.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Preserve exact values for the common identity control.  Homographies
        # are scale-equivalent, so normalize by H[2,2] when it is safe.
        h22 = safe_matrix[:, 2, 2]
        h22_safe = torch.where(
            torch.abs(h22) > HOMOGRAPHY_EPSILON, h22, torch.ones_like(h22)
        )
        normalized = safe_matrix / h22_safe[:, None, None]
        identity_rows = (
            matrix_valid
            & torch.eq(normalized, identity[None]).all(dim=(1, 2))
            & (feature_b.shape[-2:] == (output_height, output_width))
            & (input_a_hw == input_b_hw)
        )
        if bool(torch.any(identity_rows)):
            sampled = torch.where(
                identity_rows[:, None, None, None], feature_b, sampled
            )
        return sampled, valid[:, None].to(dtype=feature_b.dtype), matrix_valid

    @staticmethod
    def _diagonal_fusion(
        base_feature: torch.Tensor,
        reference_a: torch.Tensor,
        aligned_b: torch.Tensor,
        valid_mask: torch.Tensor,
        gate: torch.Tensor,
        adapter_gain: torch.Tensor,
    ) -> torch.Tensor:
        _require(
            base_feature.shape == reference_a.shape == aligned_b.shape,
            "aligned feature shape mismatch",
        )
        _require(
            valid_mask.shape
            == (
                base_feature.shape[0],
                1,
                base_feature.shape[2],
                base_feature.shape[3],
            ),
            "aligned validity-mask shape mismatch",
        )
        _require(
            gate.ndim == 1 and gate.shape[0] == base_feature.shape[0],
            "fusion gate must be B",
        )
        _require(
            adapter_gain.ndim == 1 and adapter_gain.shape[0] == base_feature.shape[1],
            "diagonal adapter width mismatch",
        )
        coefficient = (
            gate[:, None, None, None]
            * valid_mask
            * torch.tanh(adapter_gain)[None, :, None, None]
        )
        candidate = base_feature + coefficient * (aligned_b - reference_a)
        return torch.where(
            gate[:, None, None, None] > 0.0, candidate, base_feature
        )

    @staticmethod
    def symmetric_trust_region(
        parent: torch.Tensor,
        candidate: torch.Tensor,
        *,
        radius: float = TRUST_REGION_RADIUS,
    ) -> torch.Tensor:
        _require(parent.shape == candidate.shape, "trust-region shape mismatch")
        _require(float(radius) > 0.0, "trust-region radius must be positive")
        radius_value = torch.as_tensor(radius, dtype=parent.dtype, device=parent.device)
        return parent + radius_value * torch.tanh((candidate - parent) / radius_value)

    @staticmethod
    def _validate_views(view_a: torch.Tensor, view_b: torch.Tensor) -> None:
        _require(view_a.ndim == 4 and view_a.shape[1] == 3, "view A must be Bx3xHxW")
        _require(view_b.shape == view_a.shape, "two views must have identical BCHW shapes")
        _require(view_a.device == view_b.device, "two views must share a device")
        _require(view_a.dtype == view_b.dtype, "two views must share a dtype")
        _require(bool(torch.isfinite(view_a).all()), "view A contains non-finite values")

    def _extract_one_view(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer3 = self._to_layer3(image)
        layer4 = self._to_layer4(layer3)
        return layer3, layer4

    def extract_two_view_features(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> TwoViewFeatures:
        """Extract detached A/B stage maps with the shared frozen backbone."""

        self._validate_views(view_a, view_b)
        with torch.no_grad():
            layer3_a, layer4_a = self._extract_one_view(view_a)
            layer3_b, layer4_b = self._extract_one_view(view_b)
            parent = self._progress_from_layer4(layer4_a)
            rectified = self._progress_from_layer4(layer4_b)
        return TwoViewFeatures(
            layer3_a=layer3_a,
            layer4_a=layer4_a,
            layer3_b=layer3_b,
            layer4_b=layer4_b,
            parent_progress=parent,
            rectified_progress=rectified,
        )

    def extract_parent_teacher(self, image: torch.Tensor) -> torch.Tensor:
        """Return a detached frozen-parent prediction for teacher anchoring."""

        _require(image.ndim == 4 and image.shape[1] == 3, "teacher image must be Bx3xHxW")
        _require(bool(torch.isfinite(image).all()), "teacher image contains non-finite values")
        with torch.no_grad():
            layer3, layer4 = self._extract_one_view(image)
            del layer3
            return self._progress_from_layer4(layer4)

    def forward_parent_training(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return detached progress and geometry outputs of the frozen parent."""

        _require(image.ndim == 4 and image.shape[1] == 3, "teacher image must be Bx3xHxW")
        _require(bool(torch.isfinite(image).all()), "teacher image contains non-finite values")
        with torch.no_grad():
            _layer3, layer4 = self._extract_one_view(image)
            return {
                "progress": self._progress_from_layer4(layer4),
                **self._geometry_from_layer4(layer4),
            }

    def forward_parent(self, image: torch.Tensor) -> torch.Tensor:
        return self.extract_parent_teacher(image)

    def _forward_pair_impl(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        homography_a_to_b: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
        *,
        include_geometry: bool,
    ) -> dict[str, torch.Tensor]:
        """Shared implementation for inference and geometry-supervised training."""

        self._validate_views(view_a, view_b)
        batch = view_a.shape[0]
        _require(
            homography_a_to_b.shape == (batch, 3, 3),
            "homography must be Bx3x3",
        )
        _require(
            homography_a_to_b.device == view_a.device,
            "homography and views must share a device",
        )
        _require(
            confidence.ndim == 1
            and confidence.shape[0] == batch
            and confidence.device == view_a.device,
            "confidence must be device-matched B",
        )
        _require(
            active.ndim == 1
            and active.shape[0] == batch
            and active.device == view_a.device,
            "active must be device-matched B",
        )
        _require(active.dtype == torch.bool, "active must have boolean dtype")

        with torch.no_grad():
            layer3_a, layer4_a = self._extract_one_view(view_a)
            parent_progress = self._progress_from_layer4(layer4_a)

        raw_confidence = confidence.detach().to(dtype=layer3_a.dtype)
        finite_confidence = torch.isfinite(raw_confidence)
        safe_confidence = torch.where(
            finite_confidence,
            torch.clamp(raw_confidence, 0.0, 1.0),
            torch.zeros_like(raw_confidence),
        )
        requested_gate = self.reliability_gate(safe_confidence)
        matrix_valid = self._homography_validity(homography_a_to_b)
        finite_view_b = torch.isfinite(view_b).all(dim=(1, 2, 3))
        requested_active = (
            active.detach()
            & finite_confidence
            & finite_view_b
            & matrix_valid
            & (requested_gate > 0.0)
        )
        if not bool(torch.any(requested_active)):
            zeros = torch.zeros_like(parent_progress)
            return self._package_pair_result(
                parent_progress=parent_progress,
                candidate_progress=parent_progress,
                trusted_progress=parent_progress,
                effective_gate=zeros,
                effective_active=requested_active,
                parent_layer4=layer4_a,
                candidate_layer4=None,
                include_geometry=include_geometry,
            )

        safe_view_b = torch.where(
            finite_view_b[:, None, None, None], view_b, view_a
        )
        with torch.no_grad():
            layer3_b, layer4_b = self._extract_one_view(safe_view_b)
        input_a_hw = (int(view_a.shape[-2]), int(view_a.shape[-1]))
        input_b_hw = (int(view_b.shape[-2]), int(view_b.shape[-1]))
        aligned3, valid3, matrix_valid3 = self.align_feature_homography(
            layer3_b,
            homography_a_to_b,
            input_a_hw=input_a_hw,
            input_b_hw=input_b_hw,
            output_hw=(int(layer3_a.shape[-2]), int(layer3_a.shape[-1])),
        )
        aligned4, valid4, matrix_valid4 = self.align_feature_homography(
            layer4_b,
            homography_a_to_b,
            input_a_hw=input_a_hw,
            input_b_hw=input_b_hw,
            output_hw=(int(layer4_a.shape[-2]), int(layer4_a.shape[-1])),
        )
        valid_fraction3 = torch.mean(valid3.to(dtype=torch.float32), dim=(1, 2, 3))
        valid_fraction4 = torch.mean(valid4.to(dtype=torch.float32), dim=(1, 2, 3))
        effective_active = (
            requested_active
            & matrix_valid3
            & matrix_valid4
            & (valid_fraction3 >= MIN_ALIGNED_FRACTION)
            & (valid_fraction4 >= MIN_ALIGNED_FRACTION)
        )
        effective_gate = torch.where(
            effective_active, requested_gate, torch.zeros_like(requested_gate)
        )
        if not bool(torch.any(effective_active)):
            zeros = torch.zeros_like(parent_progress)
            return self._package_pair_result(
                parent_progress=parent_progress,
                candidate_progress=parent_progress,
                trusted_progress=parent_progress,
                effective_gate=zeros,
                effective_active=effective_active,
                parent_layer4=layer4_a,
                candidate_layer4=None,
                include_geometry=include_geometry,
            )

        fused3 = self._diagonal_fusion(
            layer3_a,
            layer3_a,
            aligned3,
            valid3,
            effective_gate,
            self.layer3_adapter_gain,
        )
        # This is the only inherited computation that needs an autograd graph:
        # its frozen operations propagate gradients back to the layer-3 gain.
        candidate4_pre = self._to_layer4(fused3)
        fused4 = self._diagonal_fusion(
            candidate4_pre,
            layer4_a,
            aligned4,
            valid4,
            effective_gate,
            self.layer4_adapter_gain,
        )
        raw_candidate = self._progress_from_layer4(fused4)
        candidate_progress = torch.where(
            effective_active, raw_candidate, parent_progress
        )
        bounded = self.symmetric_trust_region(
            parent_progress,
            candidate_progress,
            radius=self.trust_region_radius,
        )
        trusted_progress = torch.where(
            effective_active, bounded, parent_progress
        )
        return self._package_pair_result(
            parent_progress=parent_progress,
            candidate_progress=candidate_progress,
            trusted_progress=trusted_progress,
            effective_gate=effective_gate,
            effective_active=effective_active,
            parent_layer4=layer4_a,
            candidate_layer4=fused4,
            include_geometry=include_geometry,
        )

    def forward_pair(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        homography_a_to_b: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return parent, raw candidate, trusted prediction, and effective q."""

        return self._forward_pair_impl(
            view_a,
            view_b,
            homography_a_to_b,
            confidence,
            active,
            include_geometry=False,
        )

    def forward_pair_training(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        homography_a_to_b: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return trusted progress plus differentiable frozen-head geometry.

        The standard ``pivot``, ``direction_sin_cos``, and ``references`` keys
        are the candidate outputs selected row-wise against the exact parent.
        They can be passed directly to the existing synthetic auxiliary loss.
        """

        return self._forward_pair_impl(
            view_a,
            view_b,
            homography_a_to_b,
            confidence,
            active,
            include_geometry=True,
        )

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        homography_a_to_b: torch.Tensor,
        confidence: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_pair(
            view_a, view_b, homography_a_to_b, confidence, active
        )["trusted_progress"]


def load_db_gar_state_into_pg_siam_model(
    model: ProjectiveGeometryGuidedSIAM,
    state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load a DB-GAR18 parent while introducing exactly two zero vectors."""

    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    _require(not unexpected, f"unexpected DB-GAR parent keys: {unexpected}")
    _require(
        set(missing) == {"layer3_adapter_gain", "layer4_adapter_gain"},
        f"DB-GAR to PG-SIAM state mismatch: {missing}",
    )
    with torch.no_grad():
        model.layer3_adapter_gain.zero_()
        model.layer4_adapter_gain.zero_()
    return missing


def set_pg_siam_calibration_stage(
    model: ProjectiveGeometryGuidedSIAM,
) -> tuple[str, ...]:
    """Freeze all parent parameters and expose exactly 768 adapter values."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.layer3_adapter_gain.requires_grad_(True)
    model.layer4_adapter_gain.requires_grad_(True)
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    _require(
        set(names) == {"layer3_adapter_gain", "layer4_adapter_gain"},
        "PG-SIAM trainable parameter inventory drifted",
    )
    _require(
        sum(dict(model.named_parameters())[name].numel() for name in names)
        == TRAINABLE_PARAMETERS,
        "PG-SIAM must expose exactly 768 trainable parameters",
    )
    model.train(model.training)
    return names


__all__ = [
    "ARCHITECTURE",
    "METHOD_PREFIX",
    "PROTOCOL",
    "TRAINABLE_PARAMETERS",
    "TRUST_REGION_RADIUS",
    "ProjectiveGeometryGuidedSIAM",
    "TwoViewFeatures",
    "load_db_gar_state_into_pg_siam_model",
    "set_pg_siam_calibration_stage",
]
