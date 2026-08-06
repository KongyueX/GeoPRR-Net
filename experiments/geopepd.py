"""Unified geometry-aware probabilistic pointer-direction model.

GeoPEPD keeps the signed PEPD visual path intact and adds a calibrated
mask--geometry expert at the circular-posterior level.  The fusion is a
context-conditioned logarithmic opinion pool rather than a post-hoc mixture
of scalar meter readings.  Missing or deliberately dropped geometry has an
exact PEPD fallback.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.probabilistic_pivot_direction import (
    ProbabilisticPivotDirectionNet,
    decode_probabilistic_pivot_direction,
)


FINAL_METHOD_NAME = "GeoPEPD"

# Runtime-only quantities produced by the frozen mask/geometry front-end.
# Values are standardized inside the model.  A finite-value indicator is
# concatenated automatically, so unavailable individual measurements are not
# silently confused with a numerical zero.
GEOMETRY_FEATURE_NAMES = (
    "p_geom",
    "p_geom_v2",
    "p_fusion",
    "v1_v2_progress_delta",
    "v1_confidence",
    "v1_axis_score",
    "v1_support_ratio",
    "v1_direction_consistency",
    "v1_tip_support_ratio",
    "v2_confidence",
    "v2_axis_score",
    "v2_support_ratio",
    "v2_vote_concentration",
    "v2_side_separation",
    "mask_axis_threshold_ratio",
    "mask_candidate_ratio",
    "mask_center_distance_ratio",
    "mask_component_area_ratio",
    "seg_foreground_ratio",
    "seg_probability_max",
    "seg_probability_mean",
    "seg_probability_p99",
    "ellipse_ratio",
    "ellipse_area_ratio",
    "base_gate_probability",
    "base_residual_normalized",
    "base_residual_std_normalized",
    "base_correction_applied",
    "meter_confidence",
    "range_angle_fraction",
    "geometry_v1_available",
    "geometry_v2_available",
    "base_available",
    "branch_default_start_end",
    "branch_start_only",
    "branch_end_only",
    "branch_start_and_end",
)


class GeoPEPDOutputs(NamedTuple):
    """Outputs needed for training, diagnostics and legacy-compatible decode."""

    pivot_logits: torch.Tensor
    direction_raw: torch.Tensor
    visual_angle_logits: torch.Tensor
    geometry_angle_logits: torch.Tensor
    fused_angle_logits: torch.Tensor
    visual_log_variance_raw: torch.Tensor
    fused_log_variance_raw: torch.Tensor
    geometry_weight: torch.Tensor
    geometry_concentration: torch.Tensor
    geometry_available: torch.Tensor
    pooled_visual_features: torch.Tensor
    pooled_geometry_features: torch.Tensor

    def legacy_visual_tuple(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.pivot_logits,
            self.direction_raw,
            self.visual_angle_logits,
            self.visual_log_variance_raw,
        )

    def fused_tuple(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.pivot_logits,
            self.direction_raw,
            self.fused_angle_logits,
            self.fused_log_variance_raw,
        )


def _as_batch_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, dtype=torch.bool, device=device)
    if value.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(f"{name} must have shape [B] or [B, 1]")
    return value.reshape(batch_size).to(device=device, dtype=torch.bool)


def circular_mean_from_logits(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return angle, resultant length and normalized entropy for circular bins."""

    if logits.ndim != 2 or logits.shape[1] < 8:
        raise ValueError("logits must have shape [B, K] with K >= 8")
    probability = torch.softmax(logits.float(), dim=1)
    centers = torch.arange(
        logits.shape[1], device=logits.device, dtype=probability.dtype
    ) * (2.0 * math.pi / float(logits.shape[1]))
    cosine = torch.sum(probability * torch.cos(centers), dim=1)
    sine = torch.sum(probability * torch.sin(centers), dim=1)
    resultant = torch.sqrt(torch.clamp(cosine.square() + sine.square(), min=0.0))
    angle = torch.atan2(sine, cosine)
    entropy = -torch.sum(
        probability * torch.log(torch.clamp(probability, min=1e-12)), dim=1
    ) / math.log(float(logits.shape[1]))
    return angle, resultant, entropy


def von_mises_concentration_from_resultant(resultant: torch.Tensor) -> torch.Tensor:
    """Approximate the von-Mises concentration associated with mean resultant.

    The piecewise approximation is the standard inverse of ``A(kappa)`` used
    in directional statistics.  It lets the signed PEPD bin posterior retain
    its learned sharpness while centering the visual evidence on PEPD's actual
    vector-plus-bin decoded direction.
    """

    r = torch.clamp(resultant.float(), min=1e-4, max=0.995)
    low = 2.0 * r + r.pow(3) + (5.0 / 6.0) * r.pow(5)
    middle = -0.4 + 1.39 * r + 0.43 / torch.clamp(1.0 - r, min=1e-4)
    high = 1.0 / torch.clamp(r.pow(3) - 4.0 * r.square() + 3.0 * r, min=1e-4)
    concentration = torch.where(r < 0.53, low, torch.where(r < 0.85, middle, high))
    return torch.clamp(concentration, min=0.1, max=100.0)


class GeoPEPDCircularFusionNet(nn.Module):
    """PEPD plus a failure-aware mask--geometry circular expert.

    Given visual and geometry posteriors ``p_v`` and ``p_g``, the fused
    posterior is the normalized logarithmic opinion pool

        q(theta) proportional to p_v(theta) ** (1-a) * p_g(theta) ** a,

    where the reliability ``a`` is inferred from visual latent features,
    geometry quality, circular disagreement and PEPD uncertainty.  This is a
    distribution-level fusion; no final scalar reading is supplied as an
    input.  If geometry is unavailable, ``a`` is exactly zero.
    """

    _LEGACY_PREFIXES = (
        "encoder.",
        "pivot_head.",
        "direction_features.",
        "vector_head.",
        "angle_head.",
        "log_variance_head.",
    )

    def __init__(
        self,
        *,
        angle_bins: int = 72,
        geometry_feature_dim: int = len(GEOMETRY_FEATURE_NAMES),
        geometry_hidden_dim: int = 64,
        imagenet_pretrained: bool = False,
        initial_geometry_logit: float = -6.0,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if int(geometry_feature_dim) <= 0:
            raise ValueError("geometry_feature_dim must be positive")
        if int(geometry_hidden_dim) < 8:
            raise ValueError("geometry_hidden_dim must be at least 8")
        if not math.isfinite(float(initial_geometry_logit)):
            raise ValueError("initial_geometry_logit must be finite")

        base = ProbabilisticPivotDirectionNet(
            angle_bins=angle_bins,
            imagenet_pretrained=imagenet_pretrained,
        )
        self.angle_bins = int(base.angle_bins)
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.geometry_hidden_dim = int(geometry_hidden_dim)

        # Keep legacy names at the top level so a signed PEPD state dict loads
        # without key rewriting.
        self.encoder = base.encoder
        self.pivot_head = base.pivot_head
        self.direction_features = base.direction_features
        self.vector_head = base.vector_head
        self.angle_head = base.angle_head
        self.log_variance_head = base.log_variance_head

        self.register_buffer(
            "geometry_feature_mean",
            torch.zeros(self.geometry_feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "geometry_feature_std",
            torch.ones(self.geometry_feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "angle_centers",
            torch.arange(self.angle_bins, dtype=torch.float32)
            * (2.0 * math.pi / float(self.angle_bins)),
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(2 * self.geometry_feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, self.geometry_hidden_dim),
            nn.GELU(),
        )
        self.geometry_concentration_head = nn.Linear(self.geometry_hidden_dim, 1)
        self.geometry_residual_head = nn.Linear(
            self.geometry_hidden_dim, self.angle_bins
        )

        # Three disagreement coordinates + visual entropy/resultant/logvar +
        # log concentration + explicit availability = eight scalars.
        gate_input_dim = 256 + self.geometry_hidden_dim + 8
        self.reliability_context = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.geometry_reliability_head = nn.Linear(64, 1)
        self.variance_delta_head = nn.Linear(64, 1)
        self._initialize_new_heads(float(initial_geometry_logit))

    def _initialize_new_heads(self, initial_geometry_logit: float) -> None:
        for module in self.geometry_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.geometry_residual_head.weight)
        nn.init.zeros_(self.geometry_residual_head.bias)
        nn.init.zeros_(self.geometry_concentration_head.weight)
        # softplus(3.0) is a broad, non-degenerate initial circular expert.
        nn.init.constant_(self.geometry_concentration_head.bias, 3.0)
        for module in self.reliability_context.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.geometry_reliability_head.weight)
        nn.init.constant_(self.geometry_reliability_head.bias, initial_geometry_logit)
        nn.init.zeros_(self.variance_delta_head.weight)
        nn.init.zeros_(self.variance_delta_head.bias)

    @torch.no_grad()
    def set_geometry_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != (self.geometry_feature_dim,):
            raise ValueError("mean has the wrong geometry feature dimension")
        if std.shape != (self.geometry_feature_dim,):
            raise ValueError("std has the wrong geometry feature dimension")
        if not torch.isfinite(mean).all():
            raise ValueError("geometry mean must be finite")
        if not torch.isfinite(std).all() or bool((std <= 0.0).any()):
            raise ValueError("geometry std must be finite and positive")
        self.geometry_feature_mean.copy_(mean.float())
        self.geometry_feature_std.copy_(std.float())

    def load_pepd_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        """Load an unwrapped signed PEPD state without accepting partial legacy state."""

        own_state = self.state_dict()
        legacy_keys = {
            key
            for key in own_state
            if key.startswith(self._LEGACY_PREFIXES)
        }
        supplied = set(state_dict)
        missing = sorted(legacy_keys - supplied)
        unexpected = sorted(
            key for key in supplied if not key.startswith(self._LEGACY_PREFIXES)
        )
        shape_mismatch = sorted(
            key
            for key in legacy_keys & supplied
            if tuple(own_state[key].shape) != tuple(state_dict[key].shape)
        )
        if missing or unexpected or shape_mismatch:
            raise ValueError(
                "incompatible PEPD state: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"shape_mismatch={shape_mismatch[:5]}"
            )
        self.load_state_dict(dict(state_dict), strict=False)

    def freeze_pepd(self) -> None:
        """Freeze the complete signed visual path for stage-A fusion training."""

        for name, parameter in self.named_parameters():
            parameter.requires_grad = not name.startswith(self._LEGACY_PREFIXES)

    def unfreeze_direction_heads(self) -> None:
        """Unfreeze only the visual direction projection and heads for stage B."""

        self.freeze_pepd()
        for module in (
            self.direction_features,
            self.vector_head,
            self.angle_head,
            self.log_variance_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _geometry_features(self, raw: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(raw)
        safe = torch.where(finite, raw, self.geometry_feature_mean[None, :])
        standardized = (safe.float() - self.geometry_feature_mean[None, :]) / torch.clamp(
            self.geometry_feature_std[None, :], min=1e-6
        )
        standardized = torch.clamp(standardized, min=-8.0, max=8.0)
        return torch.cat((standardized, finite.float()), dim=1)

    def _fuse_visual_components(
        self,
        *,
        pivot_logits: torch.Tensor,
        pooled_visual: torch.Tensor,
        direction_raw: torch.Tensor,
        visual_logits: torch.Tensor,
        visual_log_variance: torch.Tensor,
        geometry_features: torch.Tensor | None,
        geometry_direction: torch.Tensor | None,
        geometry_available: torch.Tensor | None,
        geometry_drop_mask: torch.Tensor | None,
        visual_drop_mask: torch.Tensor | None,
    ) -> GeoPEPDOutputs:
        """Fuse already-computed visual components with geometry evidence."""

        if pooled_visual.ndim != 2 or pooled_visual.shape[1] != 256:
            raise ValueError("pooled_visual must have shape [B, 256]")
        batch_size = int(pooled_visual.shape[0])
        device = pooled_visual.device
        if direction_raw.shape != (batch_size, 2):
            raise ValueError("direction_raw must have shape [B, 2]")
        if visual_logits.shape != (batch_size, self.angle_bins):
            raise ValueError("visual_logits has the wrong circular-bin shape")
        if visual_log_variance.shape != (batch_size, 1):
            raise ValueError("visual_log_variance must have shape [B, 1]")
        if pivot_logits.ndim != 4 or pivot_logits.shape[0] != batch_size:
            raise ValueError("pivot_logits must have shape [B, 1, H, W]")
        if (geometry_features is None) != (geometry_direction is None):
            raise ValueError(
                "geometry_features and geometry_direction must both be supplied or omitted"
            )
        if geometry_features is None:
            geometry_features = torch.full(
                (batch_size, self.geometry_feature_dim),
                float("nan"),
                device=device,
                dtype=pooled_visual.dtype,
            )
            geometry_direction = torch.zeros(
                (batch_size, 2), device=device, dtype=pooled_visual.dtype
            )
            supplied_available = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else:
            if geometry_features.shape != (batch_size, self.geometry_feature_dim):
                raise ValueError(
                    "geometry_features must have shape [B, geometry_feature_dim]"
                )
            if geometry_direction is None or geometry_direction.shape != (batch_size, 2):
                raise ValueError("geometry_direction must have shape [B, 2]")
            supplied_available = _as_batch_mask(
                geometry_available,
                batch_size=batch_size,
                device=device,
                default=True,
                name="geometry_available",
            )

        geometry_drop = _as_batch_mask(
            geometry_drop_mask,
            batch_size=batch_size,
            device=device,
            default=False,
            name="geometry_drop_mask",
        )
        visual_drop = _as_batch_mask(
            visual_drop_mask,
            batch_size=batch_size,
            device=device,
            default=False,
            name="visual_drop_mask",
        )
        if bool((geometry_drop & visual_drop).any()):
            raise ValueError("geometry and visual experts cannot both be dropped")

        direction_norm = torch.linalg.vector_norm(geometry_direction.float(), dim=1)
        direction_finite = torch.isfinite(geometry_direction).all(dim=1)
        effective_available = (
            supplied_available
            & ~geometry_drop
            & direction_finite
            & (direction_norm > 1e-8)
        )
        if bool((visual_drop & ~effective_available).any()):
            raise ValueError("visual expert can be dropped only when geometry is available")

        encoded_geometry = self.geometry_encoder(
            self._geometry_features(geometry_features.to(device=device))
        )
        concentration = torch.clamp(
            F.softplus(self.geometry_concentration_head(encoded_geometry)[:, 0]),
            min=1e-3,
            max=100.0,
        )
        safe_direction = torch.nan_to_num(
            geometry_direction.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        geometry_angle = torch.atan2(safe_direction[:, 1], safe_direction[:, 0])
        circular_delta = self.angle_centers[None, :] - geometry_angle[:, None]
        geometry_logits = concentration[:, None] * torch.cos(circular_delta)
        geometry_logits = geometry_logits + self.geometry_residual_head(encoded_geometry)

        _, visual_resultant, visual_entropy = circular_mean_from_logits(visual_logits)
        visual_prediction = decode_probabilistic_pivot_direction(
            pivot_logits,
            direction_raw,
            visual_logits,
            visual_log_variance,
        )
        visual_angle = torch.atan2(
            visual_prediction.direction[:, 1], visual_prediction.direction[:, 0]
        )
        disagreement = visual_angle - geometry_angle
        disagreement_features = torch.stack(
            (
                torch.cos(disagreement),
                torch.abs(torch.sin(disagreement)),
                1.0 - torch.cos(disagreement),
                visual_entropy,
                visual_resultant,
                torch.clamp(visual_log_variance[:, 0].float(), -9.0, 2.0),
                torch.log1p(concentration),
                effective_available.float(),
            ),
            dim=1,
        )
        context = self.reliability_context(
            torch.cat(
                (pooled_visual.float(), encoded_geometry.float(), disagreement_features),
                dim=1,
            )
        )
        geometry_weight = torch.sigmoid(self.geometry_reliability_head(context)[:, 0])
        geometry_weight = torch.where(
            effective_available, geometry_weight, torch.zeros_like(geometry_weight)
        )
        geometry_weight = torch.where(
            visual_drop, torch.ones_like(geometry_weight), geometry_weight
        )

        visual_concentration = von_mises_concentration_from_resultant(
            visual_resultant
        )
        visual_evidence = visual_concentration[:, None] * torch.cos(
            self.angle_centers[None, :] - visual_angle[:, None]
        )
        visual_log_probability = F.log_softmax(visual_evidence, dim=1)
        geometry_log_probability = F.log_softmax(geometry_logits.float(), dim=1)
        fused_log_probability = (
            (1.0 - geometry_weight[:, None]) * visual_log_probability
            + geometry_weight[:, None] * geometry_log_probability
        )
        fused_log_probability = fused_log_probability - torch.logsumexp(
            fused_log_probability, dim=1, keepdim=True
        )
        # Preserve both the legacy decoded direction and its original 72-bin
        # posterior when the auxiliary branch is absent.
        fused_log_probability = torch.where(
            effective_available[:, None],
            fused_log_probability,
            F.log_softmax(visual_logits.float(), dim=1),
        )
        variance_delta = torch.clamp(
            self.variance_delta_head(context), min=-2.0, max=2.0
        )
        fused_log_variance = visual_log_variance.float() + (
            geometry_weight[:, None] * variance_delta
        )

        return GeoPEPDOutputs(
            pivot_logits=pivot_logits,
            direction_raw=direction_raw,
            visual_angle_logits=visual_logits,
            geometry_angle_logits=geometry_logits,
            fused_angle_logits=fused_log_probability,
            visual_log_variance_raw=visual_log_variance,
            fused_log_variance_raw=fused_log_variance,
            geometry_weight=geometry_weight,
            geometry_concentration=concentration,
            geometry_available=effective_available,
            pooled_visual_features=pooled_visual,
            pooled_geometry_features=encoded_geometry,
        )

    def forward_from_encoder_pooled(
        self,
        encoder_pooled: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        geometry_direction: torch.Tensor | None = None,
        geometry_available: torch.Tensor | None = None,
        *,
        geometry_drop_mask: torch.Tensor | None = None,
        visual_drop_mask: torch.Tensor | None = None,
    ) -> GeoPEPDOutputs:
        """Run the trainable direction/fusion path from cached encoder averages.

        This method is used only for the train-only head falsification stage.
        A 512-D tensor is the exact output of the signed encoder followed by
        adaptive average pooling.  It permits fast group-safe head training
        without mixing latent spaces from different PEPD checkpoints.
        """

        if encoder_pooled.ndim != 2 or encoder_pooled.shape[1] != 512:
            raise ValueError("encoder_pooled must have shape [B, 512]")
        modules = tuple(self.direction_features.children())
        if (
            len(modules) != 5
            or not isinstance(modules[0], nn.AdaptiveAvgPool2d)
            or not isinstance(modules[1], nn.Flatten)
            or not isinstance(modules[2], nn.Linear)
        ):
            raise RuntimeError("unexpected signed PEPD direction feature architecture")
        pooled_visual = encoder_pooled
        for module in modules[2:]:
            pooled_visual = module(pooled_visual)
        batch_size = int(encoder_pooled.shape[0])
        dummy_pivot = torch.zeros(
            (batch_size, 1, 1, 1),
            device=encoder_pooled.device,
            dtype=pooled_visual.dtype,
        )
        return self._fuse_visual_components(
            pivot_logits=dummy_pivot,
            pooled_visual=pooled_visual,
            direction_raw=self.vector_head(pooled_visual),
            visual_logits=self.angle_head(pooled_visual),
            visual_log_variance=self.log_variance_head(pooled_visual),
            geometry_features=geometry_features,
            geometry_direction=geometry_direction,
            geometry_available=geometry_available,
            geometry_drop_mask=geometry_drop_mask,
            visual_drop_mask=visual_drop_mask,
        )

    def forward(
        self,
        image: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        geometry_direction: torch.Tensor | None = None,
        geometry_available: torch.Tensor | None = None,
        *,
        geometry_drop_mask: torch.Tensor | None = None,
        visual_drop_mask: torch.Tensor | None = None,
    ) -> GeoPEPDOutputs:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        features = self.encoder(image)
        pooled_visual = self.direction_features(features)
        return self._fuse_visual_components(
            pivot_logits=self.pivot_head(features),
            pooled_visual=pooled_visual,
            direction_raw=self.vector_head(pooled_visual),
            visual_logits=self.angle_head(pooled_visual),
            visual_log_variance=self.log_variance_head(pooled_visual),
            geometry_features=geometry_features,
            geometry_direction=geometry_direction,
            geometry_available=geometry_available,
            geometry_drop_mask=geometry_drop_mask,
            visual_drop_mask=visual_drop_mask,
        )


def decode_geopepd(outputs: GeoPEPDOutputs):
    """Decode GeoPEPD without re-injecting a fixed-weight visual raw vector.

    The legacy PEPD decoder averages its raw-vector and bin-head directions.
    Repeating that operation after geometry fusion would pull every corrected
    prediction back toward the visual expert.  GeoPEPD therefore uses the
    circular mean of its fused posterior whenever geometry is present.  The
    exact legacy decoder remains the fallback when geometry is unavailable.
    """

    visual = decode_probabilistic_pivot_direction(*outputs.legacy_visual_tuple())
    fused_metadata = decode_probabilistic_pivot_direction(*outputs.fused_tuple())
    probability = torch.softmax(outputs.fused_angle_logits.float(), dim=1)
    centers = torch.arange(
        outputs.fused_angle_logits.shape[1],
        device=outputs.fused_angle_logits.device,
        dtype=probability.dtype,
    ) * (2.0 * math.pi / float(outputs.fused_angle_logits.shape[1]))
    posterior_vector = torch.stack(
        (
            torch.sum(probability * torch.cos(centers), dim=1),
            torch.sum(probability * torch.sin(centers), dim=1),
        ),
        dim=1,
    )
    posterior_norm = torch.linalg.vector_norm(posterior_vector, dim=1)
    posterior_direction = F.normalize(posterior_vector, dim=1, eps=1e-8)
    use_geometry = outputs.geometry_available[:, None]
    direction = torch.where(use_geometry, posterior_direction, visual.direction)
    geometry_valid = (
        torch.isfinite(posterior_direction).all(dim=1)
        & torch.isfinite(fused_metadata.pivot_peak)
        & torch.isfinite(fused_metadata.log_variance)
        & (posterior_norm > 1e-8)
    )
    valid = torch.where(outputs.geometry_available, geometry_valid, visual.valid)
    return fused_metadata._replace(direction=direction, valid=valid)
