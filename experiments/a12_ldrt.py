"""A12 SARN-conditioned log-density residual transport (LDRT).

LDRT is a correction-only model.  It consumes a frozen Raw progress
posterior and the corresponding frozen Raw stride-8/stride-16 features from
an A11 anchor.  An independent compact SARN encoder and homography-aligned
dual-scale relation encoder predict one residual for every progress bin.  The
residual is centered in the q0-weighted log-density tangent space and applied
as

    p1 = softmax(log(q0 + epsilon) + r_centered).

There is no expert router, scalar gate, convex blend, or posterior vote.  The
Raw posterior and Raw features are detached at the correction boundary.  A
missing SARN row, invalid homography, or empty aligned support returns q0 and
its moments exactly.

The residual output projection is initialized to zero.  At that point the
mathematical formula is an identity up to the explicit epsilon and floating
point roundoff.  A straight-through numerical canonicalization makes the
forward posterior and moments bit-exact q0 while preserving the first task
gradient into the residual output projection.  Any non-zero centered
residual follows the ordinary FP32 formula above.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
import torch.nn as nn

from experiments.a11_scort import (
    DEFAULT_PROGRESS_BINS,
    DEFAULT_RELATION_CHANNELS,
    DEFAULT_TOKEN_DIM,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
    RAW_STRIDE8_CHANNELS,
    SCORTCompactSARNEncoder,
    SCORTDualScaleRelationEncoder,
    _SCORTDifferentiableAligner,
)


A12_ARCHITECTURE: Final[str] = (
    "LDRT-SARN-Conditioned-Homography-Aligned-Log-Density-Residual-Transport"
)
DEFAULT_LOG_DENSITY_EPSILON: Final[float] = 1.0e-8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(
        posterior.ndim == 2 and posterior.is_floating_point(),
        "LDRT posterior must be floating BxK",
    )
    probability = posterior.float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    grid = torch.linspace(
        0.0,
        1.0,
        probability.shape[1],
        dtype=torch.float32,
        device=probability.device,
    )
    mean = (probability * grid[None]).sum(dim=1)
    variance = (
        probability * (grid[None] - mean[:, None]).square()
    ).sum(dim=1)
    return mean, variance


class LDRTResidualDecoder(nn.Module):
    """Decode dual-scale Raw/SARN relation memory into per-bin residuals."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        log_density_epsilon: float = DEFAULT_LOG_DENSITY_EPSILON,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "LDRT needs at least 16 progress bins")
        _require(
            token_dim >= 8
            and attention_heads >= 1
            and token_dim % attention_heads == 0,
            "LDRT token width/attention heads are incompatible",
        )
        _require(decoder_layers >= 1, "LDRT needs at least one decoder layer")
        _require(
            0.0 < log_density_epsilon < 1.0,
            "LDRT log-density epsilon must be between zero and one",
        )
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.log_density_epsilon = float(log_density_epsilon)
        self.progress_position_embedding = nn.Parameter(
            torch.zeros(1, self.progress_bins, self.token_dim)
        )
        nn.init.trunc_normal_(self.progress_position_embedding, std=0.02)
        self.posterior_embedding = nn.Sequential(
            nn.Linear(2, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=self.token_dim,
            nhead=attention_heads,
            dim_feedforward=2 * self.token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(self.token_dim)
        # A scalar bias would be unidentifiable after centering.  Omitting it
        # also makes every output-head parameter task-identifiable.
        self.residual_output = nn.Linear(self.token_dim, 1, bias=False)
        nn.init.zeros_(self.residual_output.weight)

    def forward(
        self,
        raw_posterior: torch.Tensor,
        relation_memory: torch.Tensor,
    ) -> torch.Tensor:
        _require(
            raw_posterior.ndim == 2
            and raw_posterior.shape[1] == self.progress_bins
            and raw_posterior.is_floating_point(),
            "LDRT Raw posterior has the wrong shape",
        )
        _require(
            relation_memory.ndim == 3
            and relation_memory.shape[0] == raw_posterior.shape[0]
            and relation_memory.shape[2] == self.token_dim
            and relation_memory.is_floating_point(),
            "LDRT relation memory has the wrong shape",
        )
        probability = raw_posterior.detach().float()
        posterior_features = torch.stack(
            (
                probability,
                torch.log(probability + self.log_density_epsilon),
            ),
            dim=2,
        )
        query = self.posterior_embedding(posterior_features)
        query = query + self.progress_position_embedding.to(query.dtype)
        decoded = self.decoder(tgt=query, memory=relation_memory)
        decoded = self.output_norm(decoded)
        return self.residual_output(decoded).squeeze(2)


class LDRTLogDensityTransport(nn.Module):
    """Apply a q0-centered residual as an FP32 log-density transport."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        log_density_epsilon: float = DEFAULT_LOG_DENSITY_EPSILON,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "LDRT needs at least 16 progress bins")
        _require(
            0.0 < log_density_epsilon < 1.0,
            "LDRT log-density epsilon must be between zero and one",
        )
        self.progress_bins = int(progress_bins)
        self.log_density_epsilon = float(log_density_epsilon)

    def forward(
        self,
        raw_posterior: torch.Tensor,
        uncentered_log_density_residual: torch.Tensor,
        *,
        correction_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            raw_posterior.ndim == 2
            and raw_posterior.shape[1] == self.progress_bins
            and raw_posterior.is_floating_point(),
            "LDRT Raw posterior must be floating BxK",
        )
        _require(
            uncentered_log_density_residual.shape == raw_posterior.shape
            and uncentered_log_density_residual.is_floating_point(),
            "LDRT log-density residual shape differs from q0",
        )
        _require(
            correction_available.shape == (raw_posterior.shape[0],)
            and correction_available.dtype == torch.bool
            and correction_available.device == raw_posterior.device,
            "LDRT correction availability has the wrong shape/device",
        )
        q0 = raw_posterior.detach().float()
        _require(
            bool(torch.isfinite(q0).all())
            and bool((q0 >= 0.0).all())
            and bool((q0.sum(dim=1) > 0.0).all()),
            "LDRT q0 is not a finite probability distribution",
        )
        mass = q0.sum(dim=1)
        _require(
            bool(
                torch.allclose(
                    mass,
                    torch.ones_like(mass),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
            ),
            "LDRT q0 mass differs from one",
        )
        with torch.autocast(device_type=q0.device.type, enabled=False):
            residual = uncentered_log_density_residual.float()
            _require(
                bool(torch.isfinite(residual).all()),
                "LDRT log-density residual is non-finite",
            )
            weighted_center = (q0 * residual).sum(dim=1, keepdim=True)
            centered_residual = residual - weighted_center
            raw_log_density = torch.log(q0 + self.log_density_epsilon)
            transport_logits = raw_log_density + centered_residual
            proposed_posterior = torch.softmax(transport_logits, dim=1)

            # At exactly zero centered residual the epsilon-regularized
            # softmax is only numerically, not bitwise, q0.  Canonicalize the
            # forward value while retaining the proposal derivative so the
            # zero-initialized residual head receives its first task gradient.
            residual_is_zero = (centered_residual == 0.0).all(dim=1)
            active_zero = correction_available & residual_is_zero
            straight_through_identity = (
                proposed_posterior - proposed_posterior.detach() + q0
            )
            posterior = torch.where(
                active_zero[:, None],
                straight_through_identity,
                proposed_posterior,
            )
            posterior = torch.where(
                correction_available[:, None], posterior, q0
            )
        return {
            "progress_posterior": posterior,
            "uncentered_log_density_residual": residual,
            "log_density_residual": centered_residual,
            "q0_weighted_residual_center": weighted_center.squeeze(1),
            "raw_log_density": raw_log_density,
            "transport_logits": transport_logits,
            "proposed_posterior": proposed_posterior,
            "correction_available": correction_available,
        }


class A12LDRTCorrection(nn.Module):
    """Independent LDRT correction over externally supplied frozen A11 q0."""

    def __init__(
        self,
        *,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = 4,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        log_density_epsilon: float = DEFAULT_LOG_DENSITY_EPSILON,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.memory_grid_size = int(memory_grid_size)
        self.log_density_epsilon = float(log_density_epsilon)
        self.sarn_encoder = SCORTCompactSARNEncoder()
        self.stride8_aligner = _SCORTDifferentiableAligner(feature_stride=8)
        self.stride16_aligner = _SCORTDifferentiableAligner(feature_stride=16)
        self.relation_encoder = SCORTDualScaleRelationEncoder(
            relation_channels=self.relation_channels,
            token_dim=self.token_dim,
            memory_grid_size=self.memory_grid_size,
        )
        self.residual_decoder = LDRTResidualDecoder(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
            attention_heads=self.attention_heads,
            decoder_layers=self.decoder_layers,
            log_density_epsilon=self.log_density_epsilon,
        )
        self.log_density_transport = LDRTLogDensityTransport(
            progress_bins=self.progress_bins,
            log_density_epsilon=self.log_density_epsilon,
        )

    def _validate_inputs(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require(
            raw_posterior.ndim == 2
            and raw_posterior.shape[1] == self.progress_bins
            and raw_posterior.is_floating_point(),
            "LDRT q0 has the wrong shape",
        )
        batch = raw_posterior.shape[0]
        _require(
            "stride8" in raw_encoder_features
            and "stride16" in raw_encoder_features,
            "LDRT Raw features need stride8 and stride16",
        )
        raw_stride8 = raw_encoder_features["stride8"]
        raw_stride16 = raw_encoder_features["stride16"]
        _require(
            raw_stride8.ndim == 4
            and raw_stride8.shape[0] == batch
            and raw_stride8.shape[1] == RAW_STRIDE8_CHANNELS
            and raw_stride16.ndim == 4
            and raw_stride16.shape[0] == batch
            and raw_stride16.shape[1] == EFFICIENTNET_B0_MIDDLE_FEATURES,
            "LDRT Raw stride feature shapes are incompatible with A11",
        )
        _require(
            sarn_view.ndim == 4
            and sarn_view.shape[0] == batch
            and sarn_view.shape[1] == 3
            and sarn_view.shape[2] >= 32
            and sarn_view.shape[3] >= 32
            and sarn_support_mask.shape
            == (batch, 1, sarn_view.shape[2], sarn_view.shape[3]),
            "LDRT SARN image/support shapes differ",
        )
        _require(
            sarn_active.shape == (batch,)
            and sarn_active.dtype == torch.bool
            and raw_to_sarn_homography.shape == (batch, 3, 3),
            "LDRT activity/homography shapes differ",
        )
        tensors = (
            raw_posterior,
            raw_stride8,
            raw_stride16,
            sarn_view,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        _require(
            all(value.device == raw_posterior.device for value in tensors),
            "LDRT inputs must share one device",
        )
        _require(
            raw_stride8.is_floating_point()
            and raw_stride16.is_floating_point()
            and sarn_view.is_floating_point()
            and sarn_support_mask.is_floating_point()
            and raw_to_sarn_homography.is_floating_point(),
            "LDRT features/images/support/homography must be floating",
        )
        _require(
            bool(torch.isfinite(raw_stride8).all())
            and bool(torch.isfinite(raw_stride16).all())
            and bool(torch.isfinite(sarn_view).all()),
            "LDRT feature/image input is non-finite",
        )
        return raw_stride8, raw_stride16

    def forward(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        raw_stride8, raw_stride16 = self._validate_inputs(
            raw_posterior,
            raw_encoder_features,
            sarn_view,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        # This is the explicit frozen-A11 boundary.  A correction loss cannot
        # update q0 or either Raw feature stage even when callers accidentally
        # pass tensors that require gradients.
        q0 = raw_posterior.detach().float()
        raw_stride8 = raw_stride8.detach()
        raw_stride16 = raw_stride16.detach()
        raw_mean, raw_variance = _posterior_moments(q0)

        sarn_features = self.sarn_encoder(sarn_view)
        input_hw = tuple(int(value) for value in sarn_view.shape[-2:])
        alignment8 = self.stride8_aligner(
            sarn_features=sarn_features["stride8"],
            sarn_support_mask=sarn_support_mask,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        alignment16 = self.stride16_aligner(
            sarn_features=sarn_features["stride16"],
            sarn_support_mask=sarn_support_mask,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        relation_available = (
            sarn_active
            & alignment8["transform_valid"]
            & alignment16["transform_valid"]
            & alignment8["support_valid"]
            & alignment16["support_valid"]
        )
        relation = self.relation_encoder(
            raw_stride8=raw_stride8,
            aligned_sarn_stride8=alignment8["aligned_sarn"],
            support_stride8=alignment8["common_support"],
            raw_stride16=raw_stride16,
            aligned_sarn_stride16=alignment16["aligned_sarn"],
            support_stride16=alignment16["common_support"],
            available=relation_available,
        )
        uncentered_residual = self.residual_decoder(q0, relation["memory"])
        transport = self.log_density_transport(
            q0,
            uncentered_residual,
            correction_available=relation_available,
        )
        posterior = transport["progress_posterior"]
        mean, variance = _posterior_moments(posterior)
        return {
            "architecture": A12_ARCHITECTURE,
            "progress_posterior": posterior,
            "mean": mean,
            "variance": variance,
            "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
            "raw_anchor_posterior": q0,
            "raw_anchor_mean": raw_mean,
            "raw_anchor_variance": raw_variance,
            "raw_anchor_standard_deviation": torch.sqrt(
                raw_variance.clamp_min(0.0)
            ),
            "raw_posterior": q0,
            "raw_mean": raw_mean,
            "raw_variance": raw_variance,
            "uncentered_log_density_residual": transport[
                "uncentered_log_density_residual"
            ],
            "log_density_residual": transport["log_density_residual"],
            "q0_weighted_residual_center": transport[
                "q0_weighted_residual_center"
            ],
            "raw_log_density": transport["raw_log_density"],
            "transport_logits": transport["transport_logits"],
            "proposed_posterior": transport["proposed_posterior"],
            "correction_active": relation_available,
            "relation_available": relation_available,
            "sarn_active": sarn_active,
            "sarn_encoder_features": sarn_features,
            "projective_relation": relation,
            "stride8_alignment": alignment8,
            "stride16_alignment": alignment16,
            "physical_outputs": {
                "progress_mean": mean,
                "progress_variance": variance,
            },
        }


def ldrt_parameter_counts(model: A12LDRTCorrection) -> dict[str, int]:
    _require(isinstance(model, A12LDRTCorrection), "target is not A12 LDRT")

    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    components = {
        "sarn_encoder": count(model.sarn_encoder),
        "relation_encoder": count(model.relation_encoder),
        "residual_decoder": count(model.residual_decoder),
        "log_density_transport": count(model.log_density_transport),
    }
    total = count(model)
    return {
        **components,
        "raw_anchor": 0,
        "correction": total,
        "aligners": 0,
        "total": total,
        "component_sum": int(sum(components.values())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "progress_bins": model.progress_bins,
    }


__all__ = [
    "A12_ARCHITECTURE",
    "DEFAULT_LOG_DENSITY_EPSILON",
    "A12LDRTCorrection",
    "LDRTLogDensityTransport",
    "LDRTResidualDecoder",
    "ldrt_parameter_counts",
]
