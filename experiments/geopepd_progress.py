"""Progress-coordinate posterior fusion for the unified GeoPEPD model.

The first GeoPEPD prototype fused two circular *direction* posteriors and
only converted the result to a meter reading afterwards.  That coordinate is
ill posed for meters whose reference arc or crop transform differs.  This
module instead constructs a fixed posterior over normalized progress.  Each
progress hypothesis is projected through the reference arc and crop affine
before it is compared with PEPD's visual direction.  Mask--geometry evidence
is represented by a quality-conditioned bounded Gaussian on the same grid.

The final posterior is a context-conditioned logarithmic opinion pool.  No
scalar prediction is mixed after decoding, and unavailable geometry gives an
exact visual-posterior fallback.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.geopepd import (
    GEOMETRY_FEATURE_NAMES,
    von_mises_concentration_from_resultant,
)
from experiments.probabilistic_pivot_direction import ProbabilisticPivotDirectionNet


# Exact runtime-only schema used by the reference-conditioned visual transport.
# It contains the deployable visual/reference variables that drove the
# historical calibrator plus a compact set of raw MGC quality signals, while
# excluding every calibrated prediction, residual prediction, target,
# identity, group, Transformer, and FADR field.
TRANSPORT_RUNTIME_FEATURE_NAMES = (
    "base_progress",
    "vector_progress",
    "base_vector_progress_signed",
    "base_vector_progress_abs",
    "vector_angle_std_fraction",
    "vector_angle_bin_entropy",
    "vector_angle_bin_resultant_length",
    "vector_pivot_spatial_entropy",
    "vector_pivot_top2_margin",
    "vector_direction_raw_norm_log1p",
    "range_angle_fraction",
    "reference_start_and_end",
    "reference_start_only",
    "reference_end_only",
    "reference_default_start_end",
    "meter_bbox_log_aspect",
    "pivot_center_distance_fraction",
    "gate_probability",
    "residual_abs_normalized",
    "residual_std_normalized",
    "v1_confidence",
    "v2_vote_concentration",
    "mask_component_area_ratio",
    "seg_probability_p99",
)

TRANSPORT_BRANCH_NAMES = ("default", "start_end", "start_only", "end_only")


class GeoPEPDProgressOutputs(NamedTuple):
    """Finite model outputs used by training, evaluation, and diagnostics."""

    progress_log_probability: torch.Tensor
    visual_progress_log_probability: torch.Tensor
    raw_visual_progress_log_probability: torch.Tensor
    geometry_progress_log_probability: torch.Tensor
    expected_progress: torch.Tensor
    visual_expected_progress: torch.Tensor
    raw_visual_expected_progress: torch.Tensor
    mgc_progress: torch.Tensor
    visual_progress_residual: torch.Tensor
    visual_residual_mean: torch.Tensor
    visual_correction_probability: torch.Tensor
    visual_transport_scale: torch.Tensor
    transport_branch_index: torch.Tensor
    transport_available: torch.Tensor
    pre_fusion_visual_progress_log_probability: torch.Tensor
    geometry_center_progress: torch.Tensor
    geometry_progress_residual: torch.Tensor
    visual_reliability: torch.Tensor
    geometry_reliability: torch.Tensor
    visual_concentration: torch.Tensor
    geometry_scale: torch.Tensor
    valid: torch.Tensor
    geometry_available: torch.Tensor
    reference_available: torch.Tensor
    direction_raw: torch.Tensor
    visual_angle_logits: torch.Tensor
    visual_log_variance_raw: torch.Tensor
    pooled_visual_features: torch.Tensor
    pooled_geometry_features: torch.Tensor


def _batch_vector(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fill: float,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), fill, device=device, dtype=dtype)
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(f"{name} must have shape [B] or [B, 1]")
    return value.reshape(batch_size)


def _batch_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, device=device, dtype=torch.bool)
    value = torch.as_tensor(value, device=device)
    if value.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(f"{name} must have shape [B] or [B, 1]")
    return value.reshape(batch_size).bool()


def _affine_linear(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.eye(2, device=device, dtype=dtype).expand(batch_size, -1, -1)
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.shape == (batch_size, 2, 3):
        return value[:, :, :2]
    if value.shape == (batch_size, 2, 2):
        return value
    raise ValueError("crop_affine must have shape [B, 2, 3] or [B, 2, 2]")


def _legacy_visual_summary(
    direction_raw: torch.Tensor, angle_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce PEPD's vector-plus-bin direction without a pivot heatmap."""

    raw_direction = F.normalize(direction_raw.float(), dim=1, eps=1e-8)
    probability = torch.softmax(angle_logits.float(), dim=1)
    centers = torch.arange(
        angle_logits.shape[1], device=angle_logits.device, dtype=probability.dtype
    ) * (2.0 * math.pi / float(angle_logits.shape[1]))
    bin_vector = torch.stack(
        (
            torch.sum(probability * torch.cos(centers), dim=1),
            torch.sum(probability * torch.sin(centers), dim=1),
        ),
        dim=1,
    )
    resultant = torch.linalg.vector_norm(bin_vector, dim=1)
    direction = F.normalize(raw_direction + bin_vector, dim=1, eps=1e-8)
    return direction, resultant


def _transport_progress_probability(
    log_probability: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Differentiably translate a discrete progress posterior.

    A positive residual moves posterior mass toward larger progress.  Linear
    resampling is performed on probability rather than logits; mass outside
    ``[0, 1]`` is truncated and the result is normalized again.
    """

    if log_probability.ndim != 2:
        raise ValueError("log_probability must have shape [B, K]")
    if residual.shape != (log_probability.shape[0],):
        raise ValueError("residual must have shape [B]")
    batch_size, bins = log_probability.shape
    grid = torch.linspace(
        0.0, 1.0, bins, device=log_probability.device, dtype=log_probability.dtype
    )
    source = grid[None, :] - residual[:, None]
    sample_grid = torch.stack(
        (2.0 * source - 1.0, torch.zeros_like(source)), dim=2
    ).reshape(batch_size, 1, bins, 2)
    probability = torch.exp(log_probability).reshape(batch_size, 1, 1, bins)
    transported = F.grid_sample(
        probability,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0, 0, :]
    transported = torch.clamp(transported, min=1e-12)
    transported = transported / torch.clamp(transported.sum(dim=1, keepdim=True), min=1e-12)
    return torch.log(transported)


def _broaden_progress_probability(
    log_probability: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply a sample-conditioned bounded Gaussian uncertainty kernel."""

    if log_probability.ndim != 2:
        raise ValueError("log_probability must have shape [B, K]")
    if scale.shape != (log_probability.shape[0],):
        raise ValueError("scale must have shape [B]")
    bins = log_probability.shape[1]
    grid = torch.linspace(
        0.0, 1.0, bins, device=log_probability.device, dtype=log_probability.dtype
    )
    # [B, output progress, source progress].  Normalizing over output keeps
    # each source component a proper bounded distribution near the endpoints.
    delta = grid[None, :, None] - grid[None, None, :]
    kernel = torch.exp(-0.5 * (delta / scale[:, None, None].clamp_min(1e-4)).square())
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probability = torch.exp(log_probability)
    broadened = torch.einsum("bos,bs->bo", kernel, probability)
    broadened = broadened / broadened.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.log(broadened.clamp_min(1e-12))


class ReferenceConditionedTransportHead(nn.Module):
    """Shared reference trunk with four explicit branch-specific heads.

    Each head predicts a bounded correction mean, a soft probability that the
    correction should be applied, and a progress-domain uncertainty scale.
    The head consumes only features available at inference time.
    """

    def __init__(self, input_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) < 16:
            raise ValueError("invalid transport head dimensions")
        self.shared = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 64),
            nn.GELU(),
        )
        self.branch_heads = nn.ModuleDict(
            {name: nn.Linear(64, 3) for name in TRANSPORT_BRANCH_NAMES}
        )
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for head in self.branch_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            # Visual-preserving soft abstention and a near-one-bin initial
            # uncertainty.  Direct residual supervision can move both.
            with torch.no_grad():
                head.bias[1] = -2.2
                head.bias[2] = -4.5

    def forward(
        self, context: torch.Tensor, branch_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("transport context must have shape [B, D]")
        if branch_index.shape != (context.shape[0],):
            raise ValueError("branch_index must have shape [B]")
        if bool(((branch_index < 0) | (branch_index >= len(TRANSPORT_BRANCH_NAMES))).any()):
            raise ValueError("invalid transport branch index")
        hidden = self.shared(context.float())
        all_outputs = torch.stack(
            [self.branch_heads[name](hidden) for name in TRANSPORT_BRANCH_NAMES], dim=1
        )
        selected = all_outputs.gather(
            1, branch_index[:, None, None].expand(-1, 1, 3)
        )[:, 0, :]
        residual_mean = 0.40 * torch.tanh(selected[:, 0])
        correction_probability = torch.sigmoid(selected[:, 1])
        uncertainty_scale = torch.clamp(
            F.softplus(selected[:, 2]) + 0.003, min=0.005, max=0.20
        )
        return residual_mean, correction_probability, uncertainty_scale


class GeoPEPDProgressFusionNet(nn.Module):
    """Unify PEPD and calibrated mask geometry in progress coordinates.

    For progress grid point ``p``, the original-image pointer direction is

        ``[-sin(start + range*p), cos(start + range*p)]``.

    The linear part of the crop affine transports this direction into the
    visual network's coordinate system.  A von-Mises likelihood centered on
    PEPD's legacy decoded direction then defines visual evidence over
    progress.  MGC contributes a learned-width bounded Gaussian likelihood.
    Their log probabilities are pooled before a progress value is decoded.
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
        progress_bins: int = 72,
        angle_bins: int = 72,
        geometry_feature_dim: int = len(GEOMETRY_FEATURE_NAMES),
        geometry_hidden_dim: int = 64,
        imagenet_pretrained: bool = False,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if int(progress_bins) < 16:
            raise ValueError("progress_bins must be at least 16")
        if int(geometry_feature_dim) <= 0:
            raise ValueError("geometry_feature_dim must be positive")
        if int(geometry_hidden_dim) < 8:
            raise ValueError("geometry_hidden_dim must be at least 8")

        base = ProbabilisticPivotDirectionNet(
            angle_bins=angle_bins, imagenet_pretrained=imagenet_pretrained
        )
        self.progress_bins = int(progress_bins)
        self.angle_bins = int(base.angle_bins)
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.geometry_hidden_dim = int(geometry_hidden_dim)

        # Preserve signed PEPD key names for strict legacy checkpoint loading.
        self.encoder = base.encoder
        self.pivot_head = base.pivot_head
        self.direction_features = base.direction_features
        self.vector_head = base.vector_head
        self.angle_head = base.angle_head
        self.log_variance_head = base.log_variance_head

        self.register_buffer(
            "progress_grid", torch.linspace(0.0, 1.0, self.progress_bins)
        )
        self.register_buffer(
            "geometry_feature_mean",
            torch.zeros(self.geometry_feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "geometry_feature_std",
            torch.ones(self.geometry_feature_dim, dtype=torch.float32),
        )
        self.transport_runtime_feature_dim = len(TRANSPORT_RUNTIME_FEATURE_NAMES)
        self.register_buffer(
            "transport_runtime_feature_mean",
            torch.zeros(self.transport_runtime_feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "transport_runtime_feature_std",
            torch.ones(self.transport_runtime_feature_dim, dtype=torch.float32),
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(2 * self.geometry_feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, self.geometry_hidden_dim),
            nn.GELU(),
        )
        self.geometry_scale_head = nn.Linear(self.geometry_hidden_dim, 1)

        transport_runtime_hidden = 64
        self.transport_runtime_encoder = nn.Sequential(
            nn.Linear(2 * self.transport_runtime_feature_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(96, transport_runtime_hidden),
            nn.GELU(),
        )
        # Visual latent + explicit runtime token + mask/geometry quality token
        # + optional MGC conflict/availability + explicit circular reference
        # coordinates derived inside the model.
        self.reference_transport_head = ReferenceConditionedTransportHead(
            256 + transport_runtime_hidden + self.geometry_hidden_dim + 8,
            dropout=float(dropout),
        )

        # A bounded learned correction lets the joint posterior improve beyond
        # the convex hull of the two decoded point estimates.  Its inputs are
        # restricted to train-time-safe visual latent, geometry quality,
        # signed disagreement, and visual uncertainty.
        self.residual_context = nn.Sequential(
            nn.Linear(256 + self.geometry_hidden_dim + 5, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.geometry_progress_residual_head = nn.Linear(64, 1)

        # visual latent + geometry quality + circular/progress conflict stats.
        context_dim = 256 + self.geometry_hidden_dim + 8
        self.reliability_context = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.reliability_head = nn.Linear(64, 2)
        self._initialize_new_heads()

    def _initialize_new_heads(self) -> None:
        for module in self.geometry_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.geometry_scale_head.weight)
        # softplus(-3.2) + 0.005 ~= 0.045 normalized-progress standard deviation.
        nn.init.constant_(self.geometry_scale_head.bias, -3.2)
        for module in self.transport_runtime_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.residual_context.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.geometry_progress_residual_head.weight)
        nn.init.zeros_(self.geometry_progress_residual_head.bias)
        for module in self.reliability_context.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.reliability_head.weight)
        # Geometry is an incremental correction to the transported visual
        # anchor.  It starts visual-biased and is hard-capped below replacement.
        with torch.no_grad():
            self.reliability_head.bias.copy_(torch.tensor([1.0, -1.0]))

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

    @torch.no_grad()
    def set_transport_runtime_normalization(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != (self.transport_runtime_feature_dim,):
            raise ValueError("mean has the wrong transport runtime dimension")
        if std.shape != (self.transport_runtime_feature_dim,):
            raise ValueError("std has the wrong transport runtime dimension")
        if not torch.isfinite(mean).all():
            raise ValueError("transport runtime mean must be finite")
        if not torch.isfinite(std).all() or bool((std <= 0.0).any()):
            raise ValueError("transport runtime std must be finite and positive")
        self.transport_runtime_feature_mean.copy_(mean.float())
        self.transport_runtime_feature_std.copy_(std.float())

    def load_pepd_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        own_state = self.state_dict()
        legacy_keys = {
            key for key in own_state if key.startswith(self._LEGACY_PREFIXES)
        }
        supplied = set(state_dict)
        missing = sorted(legacy_keys - supplied)
        unexpected = sorted(
            key for key in supplied if not key.startswith(self._LEGACY_PREFIXES)
        )
        mismatched = sorted(
            key
            for key in legacy_keys & supplied
            if tuple(own_state[key].shape) != tuple(state_dict[key].shape)
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                "incompatible PEPD state: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"shape_mismatch={mismatched[:5]}"
            )
        self.load_state_dict(dict(state_dict), strict=False)

    def freeze_pepd(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not name.startswith(self._LEGACY_PREFIXES)

    def unfreeze_direction_heads(self) -> None:
        self.freeze_pepd()
        for module in (
            self.direction_features,
            self.vector_head,
            self.angle_head,
            self.log_variance_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _apply_visual_transport(
        self,
        raw_visual_log_probability: torch.Tensor,
        transport_context: torch.Tensor,
        transport_branch_index: torch.Tensor,
        transport_available: torch.Tensor,
        raw_visual_expected: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Mapping[str, torch.Tensor],
    ]:
        """Apply the v2 scalar reference-conditioned visual transport.

        This narrow hook keeps the established q-mixture and uncertainty
        broadening auditable while allowing later models to replace only the
        coordinate transport.  The empty diagnostics mapping is reserved for
        structural extensions such as endpoint-conditioned warps.
        """

        del raw_visual_expected  # Used by endpoint-conditioned subclasses.
        (
            visual_residual_mean,
            visual_correction_probability,
            visual_transport_scale,
        ) = self.reference_transport_head(
            transport_context, transport_branch_index
        )
        visual_residual_mean = torch.where(
            transport_available,
            visual_residual_mean,
            torch.zeros_like(visual_residual_mean),
        )
        visual_correction_probability = torch.where(
            transport_available,
            visual_correction_probability,
            torch.zeros_like(visual_correction_probability),
        )
        shifted_visual_log_probability = _transport_progress_probability(
            raw_visual_log_probability, visual_residual_mean
        )
        uncertain_shifted_visual_log_probability = _broaden_progress_probability(
            shifted_visual_log_probability, visual_transport_scale
        )
        raw_visual_probability = torch.exp(raw_visual_log_probability)
        shifted_visual_probability = torch.exp(
            uncertain_shifted_visual_log_probability
        )
        visual_probability = (
            (1.0 - visual_correction_probability[:, None])
            * raw_visual_probability
            + visual_correction_probability[:, None]
            * shifted_visual_probability
        )
        visual_probability = visual_probability / visual_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        visual_log_probability = torch.log(visual_probability.clamp_min(1e-12))
        visual_progress_residual = (
            visual_correction_probability * visual_residual_mean
        )
        return (
            visual_log_probability,
            visual_residual_mean,
            visual_correction_probability,
            visual_transport_scale,
            visual_progress_residual,
            {},
        )

    def _augment_progress_outputs(
        self,
        outputs: GeoPEPDProgressOutputs,
        transport_diagnostics: Mapping[str, torch.Tensor],
    ) -> GeoPEPDProgressOutputs:
        """Extension point for models exposing additional transport diagnostics."""

        del transport_diagnostics
        return outputs

    def _geometry_features(self, raw: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(raw)
        safe = torch.where(finite, raw, self.geometry_feature_mean[None, :])
        standardized = (safe.float() - self.geometry_feature_mean[None, :]) / torch.clamp(
            self.geometry_feature_std[None, :], min=1e-6
        )
        return torch.cat((torch.clamp(standardized, -8.0, 8.0), finite.float()), dim=1)

    def _transport_runtime_features(self, raw: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(raw)
        safe = torch.where(
            finite, raw, self.transport_runtime_feature_mean[None, :]
        )
        standardized = (
            safe.float() - self.transport_runtime_feature_mean[None, :]
        ) / torch.clamp(self.transport_runtime_feature_std[None, :], min=1e-6)
        return torch.cat(
            (torch.clamp(standardized, -8.0, 8.0), finite.float()), dim=1
        )

    def _direction_features_from_pooled(self, encoder_pooled: torch.Tensor) -> torch.Tensor:
        modules = tuple(self.direction_features.children())
        if (
            len(modules) != 5
            or not isinstance(modules[0], nn.AdaptiveAvgPool2d)
            or not isinstance(modules[1], nn.Flatten)
            or not isinstance(modules[2], nn.Linear)
        ):
            raise RuntimeError("unexpected signed PEPD direction feature architecture")
        result = encoder_pooled
        for module in modules[2:]:
            result = module(result)
        return result

    def forward_from_encoder_pooled(
        self,
        encoder_pooled: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        mgc_progress: torch.Tensor | None = None,
        reference_start_angle: torch.Tensor | None = None,
        reference_range_angle: torch.Tensor | None = None,
        crop_affine: torch.Tensor | None = None,
        geometry_available: torch.Tensor | None = None,
        reference_available: torch.Tensor | None = None,
        *,
        transport_runtime_features: torch.Tensor | None = None,
        visual_direction: torch.Tensor | None = None,
        visual_bin_resultant: torch.Tensor | None = None,
        geometry_drop_mask: torch.Tensor | None = None,
    ) -> GeoPEPDProgressOutputs:
        if encoder_pooled.ndim != 2 or encoder_pooled.shape[1] != 512:
            raise ValueError("encoder_pooled must have shape [B, 512]")
        pooled_visual = self._direction_features_from_pooled(encoder_pooled)
        direction_raw = self.vector_head(pooled_visual)
        visual_logits = self.angle_head(pooled_visual)
        visual_log_variance = self.log_variance_head(pooled_visual)
        return self._fuse_progress(
            pooled_visual=pooled_visual,
            direction_raw=direction_raw,
            visual_logits=visual_logits,
            visual_log_variance=visual_log_variance,
            geometry_features=geometry_features,
            mgc_progress=mgc_progress,
            reference_start_angle=reference_start_angle,
            reference_range_angle=reference_range_angle,
            crop_affine=crop_affine,
            geometry_available=geometry_available,
            reference_available=reference_available,
            transport_runtime_features=transport_runtime_features,
            visual_direction=visual_direction,
            visual_bin_resultant=visual_bin_resultant,
            geometry_drop_mask=geometry_drop_mask,
        )

    def _fuse_progress(
        self,
        *,
        pooled_visual: torch.Tensor,
        direction_raw: torch.Tensor,
        visual_logits: torch.Tensor,
        visual_log_variance: torch.Tensor,
        geometry_features: torch.Tensor | None,
        mgc_progress: torch.Tensor | None,
        reference_start_angle: torch.Tensor | None,
        reference_range_angle: torch.Tensor | None,
        crop_affine: torch.Tensor | None,
        geometry_available: torch.Tensor | None,
        reference_available: torch.Tensor | None,
        transport_runtime_features: torch.Tensor | None,
        visual_direction: torch.Tensor | None,
        visual_bin_resultant: torch.Tensor | None,
        geometry_drop_mask: torch.Tensor | None,
    ) -> GeoPEPDProgressOutputs:
        if pooled_visual.ndim != 2 or pooled_visual.shape[1] != 256:
            raise ValueError("pooled_visual must have shape [B, 256]")
        batch_size = pooled_visual.shape[0]
        device, dtype = pooled_visual.device, pooled_visual.dtype
        if direction_raw.shape != (batch_size, 2):
            raise ValueError("direction_raw must have shape [B, 2]")
        if visual_logits.shape != (batch_size, self.angle_bins):
            raise ValueError("visual_logits has the wrong angle-bin shape")
        if visual_log_variance.shape != (batch_size, 1):
            raise ValueError("visual_log_variance must have shape [B, 1]")

        if visual_direction is None or visual_bin_resultant is None:
            if visual_direction is not None or visual_bin_resultant is not None:
                raise ValueError(
                    "visual_direction and visual_bin_resultant must be supplied together"
                )
            visual_direction, visual_resultant = _legacy_visual_summary(
                direction_raw, visual_logits
            )
        else:
            visual_direction = torch.as_tensor(
                visual_direction, device=device, dtype=torch.float32
            )
            if visual_direction.shape != (batch_size, 2):
                raise ValueError("visual_direction must have shape [B, 2]")
            visual_direction = F.normalize(visual_direction, dim=1, eps=1e-8)
            visual_resultant = _batch_vector(
                visual_bin_resultant,
                batch_size=batch_size,
                device=device,
                dtype=torch.float32,
                fill=0.0,
                name="visual_bin_resultant",
            )

        start = _batch_vector(
            reference_start_angle,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            fill=0.0,
            name="reference_start_angle",
        )
        angle_range = _batch_vector(
            reference_range_angle,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            fill=0.0,
            name="reference_range_angle",
        )
        affine = _affine_linear(
            crop_affine,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        supplied_reference = _batch_mask(
            reference_available,
            batch_size=batch_size,
            device=device,
            default=reference_start_angle is not None and reference_range_angle is not None,
            name="reference_available",
        )
        affine_valid = torch.isfinite(affine).all(dim=(1, 2)) & (
            torch.linalg.det(torch.nan_to_num(affine)) .abs() > 1e-8
        )
        effective_reference = (
            supplied_reference
            & torch.isfinite(start)
            & torch.isfinite(angle_range)
            & (torch.abs(angle_range) > 1e-7)
            & affine_valid
        )
        # Availability masks cannot undo NaNs that have already passed through
        # a Linear layer: zero times NaN is still NaN in backward.  All
        # unavailable reference values therefore receive finite computational
        # placeholders before any trigonometry or learned context is formed.
        safe_start = torch.nan_to_num(start, nan=0.0, posinf=0.0, neginf=0.0)
        safe_angle_range = torch.nan_to_num(
            angle_range, nan=0.0, posinf=0.0, neginf=0.0
        )
        safe_affine = torch.nan_to_num(affine, nan=0.0, posinf=0.0, neginf=0.0)

        grid = self.progress_grid.to(device=device, dtype=torch.float32)
        theta = safe_start[:, None] + safe_angle_range[:, None] * grid[None, :]
        original_directions = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=2)
        crop_directions = torch.einsum(
            "bij,bkj->bki", safe_affine, original_directions
        )
        crop_directions = F.normalize(crop_directions, dim=2, eps=1e-8)
        visual_direction_finite = torch.isfinite(visual_direction).all(dim=1)
        visual_direction_nonzero = (
            torch.linalg.vector_norm(torch.nan_to_num(visual_direction.float()), dim=1)
            > 1e-8
        )
        visual_direction = torch.nan_to_num(visual_direction.float())
        visual_cosine = torch.sum(
            crop_directions * visual_direction[:, None, :], dim=2
        )
        safe_visual_resultant = torch.clamp(
            torch.nan_to_num(
                visual_resultant.float(), nan=1e-4, posinf=0.995, neginf=1e-4
            ),
            1e-4,
            0.995,
        )
        safe_visual_log_variance = torch.clamp(
            torch.nan_to_num(
                visual_log_variance[:, 0].float(), nan=2.0, posinf=2.0, neginf=-9.0
            ),
            -9.0,
            2.0,
        )
        visual_concentration = von_mises_concentration_from_resultant(
            safe_visual_resultant
        )
        visual_log_probability = F.log_softmax(
            visual_concentration[:, None] * visual_cosine, dim=1
        )

        if geometry_features is None:
            geometry_features = torch.full(
                (batch_size, self.geometry_feature_dim),
                float("nan"),
                device=device,
                dtype=dtype,
            )
            supplied_geometry = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            geometry_features = torch.as_tensor(
                geometry_features, device=device, dtype=dtype
            )
            if geometry_features.shape != (batch_size, self.geometry_feature_dim):
                raise ValueError(
                    "geometry_features must have shape [B, geometry_feature_dim]"
                )
            supplied_geometry = _batch_mask(
                geometry_available,
                batch_size=batch_size,
                device=device,
                default=mgc_progress is not None,
                name="geometry_available",
            )
        geometry_drop = _batch_mask(
            geometry_drop_mask,
            batch_size=batch_size,
            device=device,
            default=False,
            name="geometry_drop_mask",
        )
        mgc = _batch_vector(
            mgc_progress,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            fill=0.5,
            name="mgc_progress",
        )
        encoded_geometry = self.geometry_encoder(self._geometry_features(geometry_features))
        geometry_scale = torch.clamp(
            F.softplus(self.geometry_scale_head(encoded_geometry)[:, 0]) + 0.005,
            min=0.01,
            max=0.35,
        )
        safe_mgc = torch.clamp(torch.nan_to_num(mgc, nan=0.5), 0.0, 1.0)
        effective_geometry = (
            supplied_geometry
            & ~geometry_drop
            & effective_reference
            & torch.isfinite(mgc)
            & (mgc >= 0.0)
            & (mgc <= 1.0)
        )

        # ``mgc_progress`` is the raw mask--geometry ``base_progress``.  A
        # reference-conditioned FADR/calibrated progress is deliberately not
        # part of this model's interface.
        raw_visual_log_probability = visual_log_probability
        raw_visual_probability = torch.exp(raw_visual_log_probability)
        raw_visual_expected = torch.sum(
            raw_visual_probability * grid[None, :], dim=1
        )
        raw_visual_entropy = -torch.sum(
            raw_visual_probability * raw_visual_log_probability, dim=1
        ) / math.log(float(self.progress_bins))

        if transport_runtime_features is None:
            transport_runtime_features = torch.full(
                (batch_size, self.transport_runtime_feature_dim),
                float("nan"),
                device=device,
                dtype=dtype,
            )
        else:
            transport_runtime_features = torch.as_tensor(
                transport_runtime_features, device=device, dtype=dtype
            )
            if transport_runtime_features.shape != (
                batch_size,
                self.transport_runtime_feature_dim,
            ):
                raise ValueError(
                    "transport_runtime_features must have shape "
                    "[B, len(TRANSPORT_RUNTIME_FEATURE_NAMES)]"
                )
        branch_bits = torch.nan_to_num(
            transport_runtime_features[:, 11:15].float(), nan=0.0
        ) > 0.5
        if bool((branch_bits.sum(dim=1) > 1).any()):
            raise ValueError("transport reference branch bits must be one-hot")
        transport_branch_index = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        transport_branch_index = torch.where(
            branch_bits[:, 0],
            torch.ones_like(transport_branch_index),
            transport_branch_index,
        )
        transport_branch_index = torch.where(
            branch_bits[:, 1],
            torch.full_like(transport_branch_index, 2),
            transport_branch_index,
        )
        transport_branch_index = torch.where(
            branch_bits[:, 2],
            torch.full_like(transport_branch_index, 3),
            transport_branch_index,
        )
        encoded_transport_runtime = self.transport_runtime_encoder(
            self._transport_runtime_features(transport_runtime_features)
        )
        visual_valid = visual_direction_finite & visual_direction_nonzero
        transport_available = effective_reference & visual_valid
        signed_mgc_conflict = torch.where(
            effective_geometry,
            raw_visual_expected - safe_mgc,
            torch.zeros_like(raw_visual_expected),
        )
        transport_context = torch.cat(
            (
                pooled_visual.float(),
                encoded_transport_runtime.float(),
                encoded_geometry.float(),
                signed_mgc_conflict[:, None],
                effective_geometry.float()[:, None],
                visual_direction.float(),
                torch.sin(safe_start)[:, None],
                torch.cos(safe_start)[:, None],
                torch.sin(safe_angle_range)[:, None],
                torch.cos(safe_angle_range)[:, None],
            ),
            dim=1,
        )
        (
            visual_log_probability,
            visual_residual_mean,
            visual_correction_probability,
            visual_transport_scale,
            visual_progress_residual,
            transport_diagnostics,
        ) = self._apply_visual_transport(
            raw_visual_log_probability,
            transport_context,
            transport_branch_index,
            transport_available,
            raw_visual_expected,
        )
        visual_probability = torch.exp(visual_log_probability)
        visual_expected = torch.sum(visual_probability * grid[None, :], dim=1)
        visual_entropy = -torch.sum(
            visual_probability * visual_log_probability, dim=1
        ) / math.log(float(self.progress_bins))
        residual_scalars = torch.stack(
            (
                raw_visual_expected - safe_mgc,
                raw_visual_entropy,
                safe_visual_resultant,
                safe_visual_log_variance,
                effective_geometry.float(),
            ),
            dim=1,
        )
        residual_context = self.residual_context(
            torch.cat(
                (pooled_visual.float(), encoded_geometry.float(), residual_scalars), dim=1
            )
        )
        geometry_progress_residual = 0.10 * torch.tanh(
            self.geometry_progress_residual_head(residual_context)[:, 0]
        )
        geometry_progress_residual = torch.where(
            effective_geometry,
            geometry_progress_residual,
            torch.zeros_like(geometry_progress_residual),
        )
        geometry_center = torch.clamp(
            safe_mgc + geometry_progress_residual, min=0.0, max=1.0
        )
        geometry_logits = -0.5 * (
            (grid[None, :] - geometry_center[:, None]) / geometry_scale[:, None]
        ).square()
        geometry_log_probability = F.log_softmax(geometry_logits, dim=1)

        mgc_theta = safe_start + safe_angle_range * geometry_center
        mgc_original_direction = torch.stack(
            (-torch.sin(mgc_theta), torch.cos(mgc_theta)), dim=1
        )
        mgc_crop_direction = F.normalize(
            torch.bmm(safe_affine, mgc_original_direction[:, :, None])[:, :, 0],
            dim=1,
            eps=1e-8,
        )
        direction_conflict_cosine = torch.sum(
            mgc_crop_direction * visual_direction, dim=1
        ).clamp(-1.0, 1.0)
        direction_conflict_sine_abs = torch.abs(
            mgc_crop_direction[:, 0] * visual_direction[:, 1]
            - mgc_crop_direction[:, 1] * visual_direction[:, 0]
        )
        progress_conflict = torch.abs(visual_expected - geometry_center)
        context_scalars = torch.stack(
            (
                progress_conflict,
                direction_conflict_cosine,
                direction_conflict_sine_abs,
                visual_entropy,
                safe_visual_resultant,
                safe_visual_log_variance,
                geometry_scale,
                effective_geometry.float(),
            ),
            dim=1,
        )
        context = self.reliability_context(
            torch.cat((pooled_visual.float(), encoded_geometry.float(), context_scalars), dim=1)
        )
        reliability_logits = self.reliability_head(context)
        # Geometry can refine, but never replace, the calibrated visual anchor.
        # The 0.5 cap prevents the Stage-A collapse previously observed when a
        # runner initialized the raw geometry softmax near 0.94.
        geometry_gate = 0.5 * torch.sigmoid(
            reliability_logits[:, 1] - reliability_logits[:, 0]
        )
        geometry_reliability = torch.where(
            effective_geometry, geometry_gate, torch.zeros_like(geometry_gate)
        )
        visual_reliability = 1.0 - geometry_reliability
        pooled_log_probability = visual_log_probability + geometry_reliability[
            :, None
        ] * (
            geometry_log_probability - visual_log_probability
        )
        pooled_log_probability = pooled_log_probability - torch.logsumexp(
            pooled_log_probability, dim=1, keepdim=True
        )

        # A missing reference makes progress unidentified.  Keep all tensors
        # finite but expose invalidity explicitly to losses and evaluators.
        uniform_log_probability = torch.full_like(
            pooled_log_probability, -math.log(float(self.progress_bins))
        )
        pooled_log_probability = torch.where(
            effective_reference[:, None], pooled_log_probability, uniform_log_probability
        )
        visual_log_probability = torch.where(
            effective_reference[:, None], visual_log_probability, uniform_log_probability
        )
        raw_visual_log_probability = torch.where(
            effective_reference[:, None],
            raw_visual_log_probability,
            uniform_log_probability,
        )
        geometry_log_probability = torch.where(
            effective_reference[:, None], geometry_log_probability, uniform_log_probability
        )
        expected = torch.sum(torch.exp(pooled_log_probability) * grid[None, :], dim=1)
        visual_expected = torch.sum(
            torch.exp(visual_log_probability) * grid[None, :], dim=1
        )
        raw_visual_expected = torch.sum(
            torch.exp(raw_visual_log_probability) * grid[None, :], dim=1
        )
        visual_valid = visual_direction_finite & visual_direction_nonzero

        outputs = GeoPEPDProgressOutputs(
            progress_log_probability=pooled_log_probability,
            visual_progress_log_probability=visual_log_probability,
            raw_visual_progress_log_probability=raw_visual_log_probability,
            geometry_progress_log_probability=geometry_log_probability,
            expected_progress=expected,
            visual_expected_progress=visual_expected,
            raw_visual_expected_progress=raw_visual_expected,
            mgc_progress=safe_mgc,
            visual_progress_residual=visual_progress_residual,
            visual_residual_mean=visual_residual_mean,
            visual_correction_probability=visual_correction_probability,
            visual_transport_scale=visual_transport_scale,
            transport_branch_index=transport_branch_index,
            transport_available=transport_available,
            pre_fusion_visual_progress_log_probability=visual_log_probability,
            geometry_center_progress=geometry_center,
            geometry_progress_residual=geometry_progress_residual,
            visual_reliability=visual_reliability,
            geometry_reliability=geometry_reliability,
            visual_concentration=visual_concentration,
            geometry_scale=geometry_scale,
            valid=effective_reference & visual_valid,
            geometry_available=effective_geometry,
            reference_available=effective_reference,
            direction_raw=direction_raw,
            visual_angle_logits=visual_logits,
            visual_log_variance_raw=visual_log_variance,
            pooled_visual_features=pooled_visual,
            pooled_geometry_features=encoded_geometry,
        )
        return self._augment_progress_outputs(outputs, transport_diagnostics)

    def forward(
        self,
        image: torch.Tensor,
        geometry_features: torch.Tensor | None = None,
        mgc_progress: torch.Tensor | None = None,
        reference_start_angle: torch.Tensor | None = None,
        reference_range_angle: torch.Tensor | None = None,
        crop_affine: torch.Tensor | None = None,
        geometry_available: torch.Tensor | None = None,
        reference_available: torch.Tensor | None = None,
        *,
        transport_runtime_features: torch.Tensor | None = None,
        geometry_drop_mask: torch.Tensor | None = None,
    ) -> GeoPEPDProgressOutputs:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        features = self.encoder(image)
        encoder_pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.forward_from_encoder_pooled(
            encoder_pooled,
            geometry_features,
            mgc_progress,
            reference_start_angle,
            reference_range_angle,
            crop_affine,
            geometry_available,
            reference_available,
            transport_runtime_features=transport_runtime_features,
            geometry_drop_mask=geometry_drop_mask,
        )


def progress_posterior_nll(
    outputs: GeoPEPDProgressOutputs, target_progress: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation NLL on the fixed progress grid, valid rows only."""

    target = torch.as_tensor(
        target_progress,
        device=outputs.progress_log_probability.device,
        dtype=outputs.progress_log_probability.dtype,
    ).reshape(-1)
    if target.shape != outputs.expected_progress.shape:
        raise ValueError("target_progress must have shape [B] or [B, 1]")
    valid = outputs.valid & torch.isfinite(target) & (target >= 0.0) & (target <= 1.0)
    if not bool(valid.any()):
        return outputs.progress_log_probability.sum() * 0.0
    bins = outputs.progress_log_probability.shape[1]
    position = torch.clamp(target[valid], 0.0, 1.0) * float(bins - 1)
    lower = torch.floor(position).long()
    upper = torch.clamp(lower + 1, max=bins - 1)
    fraction = position - lower.float()
    selected = outputs.progress_log_probability[valid]
    log_lower = selected.gather(1, lower[:, None])[:, 0]
    log_upper = selected.gather(1, upper[:, None])[:, 0]
    log_probability = torch.logaddexp(
        log_lower + torch.log(torch.clamp(1.0 - fraction, min=1e-8)),
        log_upper + torch.log(torch.clamp(fraction, min=1e-8)),
    )
    return -log_probability.mean()
