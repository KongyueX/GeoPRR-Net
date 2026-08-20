"""A15 frozen twin-endpoint per-bin natural-parameter bridge (FTEB).

FTEB keeps the terminal A11 Raw model as one frozen, shared semantic endpoint.
The anchor is evaluated separately on the Raw and SARN pixels, producing two
complete progress distributions and two pairs of stride features.  A small
trainable correction then consumes only detached anchor outputs:

* homography-aligned Raw/SARN features use the same projection at each scale;
* a target-free geometry/distribution token conditions 128 progress tokens;
* every progress bin predicts its own endpoint and free residual; and
* eight points on one natural-parameter path report the corrected posterior.

There is no sample-wise router, expert weight, posterior vote, or convex
blend.  The fixed 0.5 endpoint is the symmetric product-of-experts consensus
in natural-parameter space, not a learned Bx1 decision.  At construction the
learned per-bin residual is exactly zero.  Active rows therefore travel from
q0 to that fixed Raw/SARN geometric endpoint, while missing SARN, invalid
homography, or empty aligned support selects bit-exact q0 at the outermost
boundary of every layer.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.a11_scort import (
    DEFAULT_RELATION_CHANNELS,
    DEFAULT_TOKEN_DIM,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
    RAW_STRIDE8_CHANNELS,
    _SCORTDifferentiableAligner,
    deterministic_adaptive_average_pool2d,
)


A15_ARCHITECTURE: Final[str] = (
    "FTEB-Frozen-Twin-Endpoint-Homography-Conditioned-Per-Bin-"
    "Natural-Parameter-Bridge"
)
DEFAULT_PROGRESS_BINS: Final[int] = 128
DEFAULT_MEMORY_GRID_SIZE: Final[int] = 4
BRIDGE_LAYERS: Final[int] = 8
BIN_FEATURES: Final[int] = 9
GEOMETRY_FEATURES: Final[int] = 13
LOG_RATIO_LIMIT: Final[float] = 8.0
MAX_ENDPOINT_RESIDUAL_GAIN: Final[float] = 1.5
MAX_FREE_FIELD: Final[float] = 0.5
PROBABILITY_EPSILON: Final[float] = torch.finfo(torch.float32).tiny


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalization_groups(channels: int) -> int:
    _require(channels >= 1, "FTEB normalization width must be positive")
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise AssertionError("one group always divides a positive width")


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(
        posterior.ndim == 2 and posterior.is_floating_point(),
        "FTEB posterior must be floating BxK",
    )
    probability = posterior.float()
    probability = probability / probability.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    grid = torch.linspace(
        0.0,
        1.0,
        probability.shape[1],
        device=probability.device,
        dtype=torch.float32,
    )
    mean = (probability * grid[None]).sum(dim=1)
    variance = (
        probability * (grid[None] - mean[:, None]).square()
    ).sum(dim=1)
    return mean, variance


def fixed_geometric_natural_parameter_base(
    q0: torch.Tensor,
    q_sarn: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the measured FP32 geometric endpoint and its exact operands."""

    _require(
        q0.shape == q_sarn.shape
        and q0.ndim == 2
        and q0.is_floating_point()
        and q_sarn.is_floating_point(),
        "FTEB geometric endpoints must be matching floating BxK tensors",
    )
    raw = q0.detach().float()
    sarn = q_sarn.detach().float()
    log0 = torch.log(raw.clamp_min(PROBABILITY_EPSILON))
    log_sarn = torch.log(sarn.clamp_min(PROBABILITY_EPSILON))
    base_logits = 0.5 * (log0 + log_sarn)
    geometric_base = torch.softmax(base_logits, dim=1)
    return {
        "raw": raw,
        "sarn": sarn,
        "raw_log_density": log0,
        "sarn_log_density": log_sarn,
        "base_logits": base_logits,
        "geometric_base": geometric_base,
    }


def _validate_posterior(posterior: torch.Tensor, *, label: str) -> None:
    _require(
        posterior.ndim == 2 and posterior.is_floating_point(),
        f"{label} must be floating BxK",
    )
    value = posterior.detach().float()
    _require(
        bool(torch.isfinite(value).all())
        and bool((value >= 0.0).all())
        and bool((value.sum(dim=1) > 0.0).all()),
        f"{label} is not a finite probability distribution",
    )
    mass = value.sum(dim=1)
    _require(
        bool(
            torch.allclose(
                mass,
                torch.ones_like(mass),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        ),
        f"{label} mass differs from one",
    )


def _endpoint_from_anchor(
    anchor: nn.Module,
    images: torch.Tensor,
) -> dict[str, Any]:
    encoder = getattr(anchor, "raw_encoder", None)
    head = getattr(anchor, "raw_posterior_head", None)
    _require(
        isinstance(encoder, nn.Module) and isinstance(head, nn.Module),
        "FTEB anchor needs raw_encoder and raw_posterior_head modules",
    )
    features = encoder(images)
    _require(
        isinstance(features, Mapping)
        and all(name in features for name in ("stride8", "stride16", "representation")),
        "FTEB anchor encoder outputs are incomplete",
    )
    representation = features["representation"]
    _require(
        isinstance(representation, torch.Tensor),
        "FTEB anchor representation is not a tensor",
    )
    logits = head(representation)
    _require(
        isinstance(logits, torch.Tensor)
        and logits.ndim == 2
        and logits.shape[0] == images.shape[0]
        and bool(torch.isfinite(logits).all()),
        "FTEB anchor logits are malformed",
    )
    posterior = torch.softmax(logits.float(), dim=1)
    mean, variance = _posterior_moments(posterior)
    return {
        "logits": logits,
        "log_posterior": torch.log_softmax(logits.float(), dim=1),
        "posterior": posterior,
        "mean": mean,
        "variance": variance,
        "features": {
            "stride8": features["stride8"],
            "stride16": features["stride16"],
        },
    }


def frozen_twin_endpoint_forward(
    anchor: nn.Module,
    original_view: torch.Tensor,
    sarn_view: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate one frozen A11 anchor separately on Raw and SARN pixels.

    Separate calls are deliberate: the Raw call has the same batch shape and
    operator order as the established q0 forward, so adding the SARN endpoint
    cannot change q0 through a concatenated-batch kernel choice.  ``no_grad``
    and explicit detachment make this helper a hard correction boundary even
    if a caller accidentally leaves an anchor parameter trainable.
    """

    _require(isinstance(anchor, nn.Module), "FTEB anchor is not a module")
    _require(
        original_view.ndim == sarn_view.ndim == 4
        and original_view.shape == sarn_view.shape
        and original_view.is_floating_point()
        and sarn_view.is_floating_point(),
        "FTEB twin endpoint images must be matching floating BCHW tensors",
    )
    _require(
        original_view.device == sarn_view.device,
        "FTEB twin endpoint images must share one device",
    )
    anchor.eval()
    with torch.no_grad():
        raw = _endpoint_from_anchor(anchor, original_view)
        sarn = _endpoint_from_anchor(anchor, sarn_view)
    _require(
        raw["posterior"].shape == sarn["posterior"].shape,
        "FTEB Raw/SARN endpoint posterior shapes differ",
    )
    return {
        "raw_posterior": raw["posterior"].detach(),
        "raw_log_posterior": raw["log_posterior"].detach(),
        "raw_mean": raw["mean"].detach(),
        "raw_variance": raw["variance"].detach(),
        "raw_logits": raw["logits"].detach(),
        "raw_features": {
            name: value.detach() for name, value in raw["features"].items()
        },
        "sarn_posterior": sarn["posterior"].detach(),
        "sarn_log_posterior": sarn["log_posterior"].detach(),
        "sarn_mean": sarn["mean"].detach(),
        "sarn_variance": sarn["variance"].detach(),
        "sarn_logits": sarn["logits"].detach(),
        "sarn_features": {
            name: value.detach() for name, value in sarn["features"].items()
        },
    }


class _FTEBSharedRelationScale(nn.Module):
    """Encode one aligned scale with one projection shared by both views."""

    def __init__(
        self,
        *,
        input_channels: int,
        relation_channels: int,
        token_dim: int,
        memory_grid_size: int,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.shared_projection = nn.Sequential(
            nn.Conv2d(input_channels, relation_channels, kernel_size=1),
            nn.GroupNorm(_normalization_groups(relation_channels), relation_channels),
            nn.SiLU(),
        )
        relation_input_channels = 5 * relation_channels + 2
        self.relation_stem = nn.Sequential(
            nn.Conv2d(relation_input_channels, token_dim, kernel_size=1),
            nn.GroupNorm(_normalization_groups(token_dim), token_dim),
            nn.SiLU(),
            nn.Conv2d(
                token_dim,
                token_dim,
                kernel_size=3,
                padding=1,
                groups=token_dim,
            ),
            nn.Conv2d(token_dim, token_dim, kernel_size=1),
            nn.GroupNorm(_normalization_groups(token_dim), token_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        *,
        raw: torch.Tensor,
        aligned_sarn: torch.Tensor,
        common_support: torch.Tensor,
        available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = raw.shape[0]
        _require(
            raw.ndim == aligned_sarn.ndim == 4
            and raw.shape == aligned_sarn.shape
            and raw.shape[1] == self.input_channels,
            "FTEB shared-relation view shapes differ",
        )
        _require(
            common_support.shape == (batch, 1, raw.shape[2], raw.shape[3])
            and available.shape == (batch,)
            and available.dtype == torch.bool,
            "FTEB shared-relation support/availability shapes differ",
        )
        with torch.autocast(device_type=raw.device.type, enabled=False):
            support = torch.nan_to_num(
                common_support.detach().float(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(0.0, 1.0)
            support = support * available[:, None, None, None].to(torch.float32)
            raw_projected = self.shared_projection(raw.detach().float())
            sarn_projected = self.shared_projection(aligned_sarn.detach().float())
            cosine = F.cosine_similarity(
                raw_projected, sarn_projected, dim=1, eps=1.0e-6
            )[:, None]
            relation = torch.cat(
                (
                    raw_projected,
                    sarn_projected,
                    raw_projected - sarn_projected,
                    torch.abs(raw_projected - sarn_projected),
                    raw_projected * sarn_projected,
                    cosine,
                    support,
                ),
                dim=1,
            )
            encoded = self.relation_stem(relation * support) * support
            memory = deterministic_adaptive_average_pool2d(
                encoded, self.memory_grid_size
            ).flatten(2).transpose(1, 2)
        return {
            "raw_projected": raw_projected,
            "sarn_projected": sarn_projected,
            "relation": relation,
            "encoded": encoded,
            "memory": memory,
            "active_support": support,
        }


class FTEBSharedDualScaleRelationEncoder(nn.Module):
    """Build 32 default relation tokens from frozen twin A11 features."""

    def __init__(
        self,
        *,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        memory_grid_size: int = DEFAULT_MEMORY_GRID_SIZE,
    ) -> None:
        super().__init__()
        _require(
            relation_channels >= 8 and relation_channels % 8 == 0,
            "FTEB relation width must be a multiple of eight",
        )
        _require(
            token_dim >= 8 and token_dim % 8 == 0,
            "FTEB token width must be a multiple of eight",
        )
        _require(memory_grid_size >= 2, "FTEB memory grid is too small")
        self.token_dim = int(token_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.stride8 = _FTEBSharedRelationScale(
            input_channels=RAW_STRIDE8_CHANNELS,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.stride16 = _FTEBSharedRelationScale(
            input_channels=EFFICIENTNET_B0_MIDDLE_FEATURES,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        token_count = 2 * memory_grid_size * memory_grid_size
        self.memory_position_embedding = nn.Parameter(
            torch.zeros(1, token_count, token_dim)
        )
        nn.init.trunc_normal_(self.memory_position_embedding, std=0.02)

    def forward(
        self,
        *,
        raw_stride8: torch.Tensor,
        aligned_sarn_stride8: torch.Tensor,
        support_stride8: torch.Tensor,
        raw_stride16: torch.Tensor,
        aligned_sarn_stride16: torch.Tensor,
        support_stride16: torch.Tensor,
        available: torch.Tensor,
    ) -> dict[str, Any]:
        fine = self.stride8(
            raw=raw_stride8,
            aligned_sarn=aligned_sarn_stride8,
            common_support=support_stride8,
            available=available,
        )
        coarse = self.stride16(
            raw=raw_stride16,
            aligned_sarn=aligned_sarn_stride16,
            common_support=support_stride16,
            available=available,
        )
        memory = torch.cat((fine["memory"], coarse["memory"]), dim=1)
        memory = memory + self.memory_position_embedding.to(memory.dtype)
        memory = memory * available[:, None, None].to(memory.dtype)
        return {"memory": memory, "fine": fine, "coarse": coarse}


class FTEBGeometryTokenEncoder(nn.Module):
    """Encode target-free homography, support, and endpoint disagreement."""

    def __init__(self, *, token_dim: int = DEFAULT_TOKEN_DIM) -> None:
        super().__init__()
        _require(
            token_dim >= 8 and token_dim % 8 == 0,
            "FTEB geometry-token width must be a multiple of eight",
        )
        self.token_dim = int(token_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(GEOMETRY_FEATURES),
            nn.Linear(GEOMETRY_FEATURES, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        q0: torch.Tensor,
        q_sarn: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
        support_area8: torch.Tensor,
        support_area16: torch.Tensor,
        *,
        available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = q0.shape[0]
        _require(
            q0.shape == q_sarn.shape
            and raw_to_sarn_homography.shape == (batch, 3, 3)
            and support_area8.shape == support_area16.shape == available.shape == (batch,)
            and available.dtype == torch.bool,
            "FTEB geometry-token inputs differ",
        )
        with torch.autocast(device_type=q0.device.type, enabled=False):
            q0f = q0.detach().float()
            qsf = q_sarn.detach().float()
            homography = raw_to_sarn_homography.detach().float()
            finite_h = torch.isfinite(homography).all(dim=(1, 2))
            denominator = homography[:, 2, 2]
            valid_h = finite_h & torch.isfinite(denominator) & (denominator.abs() > 1.0e-8)
            safe_h = torch.nan_to_num(
                homography, nan=0.0, posinf=0.0, neginf=0.0
            )
            usable_h = available & valid_h
            identity_h = torch.eye(
                3, device=homography.device, dtype=torch.float32
            )[None].expand(batch, -1, -1)
            safe_h = torch.where(usable_h[:, None, None], safe_h, identity_h)
            safe_denominator = safe_h[:, 2, 2]
            normalized_h = safe_h / safe_denominator[:, None, None]
            h_features = torch.stack(
                (
                    normalized_h[:, 0, 0] - 1.0,
                    normalized_h[:, 0, 1],
                    normalized_h[:, 0, 2],
                    normalized_h[:, 1, 0],
                    normalized_h[:, 1, 1] - 1.0,
                    normalized_h[:, 1, 2],
                    normalized_h[:, 2, 0],
                    normalized_h[:, 2, 1],
                ),
                dim=1,
            )
            mean0, variance0 = _posterior_moments(q0f)
            mean_sarn, variance_sarn = _posterior_moments(qsf)
            mixture = 0.5 * (q0f + qsf)
            log0 = torch.log(q0f.clamp_min(PROBABILITY_EPSILON))
            logs = torch.log(qsf.clamp_min(PROBABILITY_EPSILON))
            logm = torch.log(mixture.clamp_min(PROBABILITY_EPSILON))
            js = 0.5 * (
                (q0f * (log0 - logm)).sum(dim=1)
                + (qsf * (logs - logm)).sum(dim=1)
            )
            distribution_features = torch.stack(
                (
                    mean_sarn - mean0,
                    torch.log(
                        (variance_sarn + PROBABILITY_EPSILON)
                        / (variance0 + PROBABILITY_EPSILON)
                    ),
                    js,
                ),
                dim=1,
            )
            features = torch.cat(
                (
                    h_features,
                    support_area8.detach().float()[:, None],
                    support_area16.detach().float()[:, None],
                    distribution_features,
                ),
                dim=1,
            )
            _require(
                features.shape == (batch, GEOMETRY_FEATURES),
                "FTEB geometry feature count differs",
            )
            features = torch.nan_to_num(
                features, nan=0.0, posinf=0.0, neginf=0.0
            )
            features = features * available[:, None].to(torch.float32)
            token = self.network(features)[:, None]
            token = token + self.position_embedding.to(token.dtype)
            token = token * available[:, None, None].to(token.dtype)
        return {"features": features, "token": token}


class FTEBBinDecoder(nn.Module):
    """Condition nine-feature endpoint tokens on aligned relation memory."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "FTEB decoder needs at least 16 bins")
        _require(
            token_dim >= 8
            and attention_heads >= 1
            and token_dim % attention_heads == 0,
            "FTEB token width/attention heads are incompatible",
        )
        _require(decoder_layers >= 1, "FTEB needs at least one decoder layer")
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.progress_position_embedding = nn.Parameter(
            torch.zeros(1, progress_bins, token_dim)
        )
        nn.init.trunc_normal_(self.progress_position_embedding, std=0.02)
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(BIN_FEATURES, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=attention_heads,
            dim_feedforward=2 * token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        q0: torch.Tensor,
        q_sarn: torch.Tensor,
        relation_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            q0.shape == q_sarn.shape
            == (relation_memory.shape[0], self.progress_bins)
            and relation_memory.ndim == 3
            and relation_memory.shape[2] == self.token_dim,
            "FTEB endpoint/relation-memory shapes differ",
        )
        with torch.autocast(device_type=q0.device.type, enabled=False):
            probability0 = q0.detach().float()
            probability_sarn = q_sarn.detach().float()
            log0 = torch.log(probability0.clamp_min(PROBABILITY_EPSILON))
            logs = torch.log(probability_sarn.clamp_min(PROBABILITY_EPSILON))
            cdf0 = probability0.cumsum(dim=1)
            cdfs = probability_sarn.cumsum(dim=1)
            grid = torch.linspace(
                0.0,
                1.0,
                self.progress_bins,
                device=q0.device,
                dtype=torch.float32,
            )[None].expand(q0.shape[0], -1)
            features = torch.stack(
                (
                    probability0,
                    probability_sarn,
                    log0,
                    logs,
                    cdf0,
                    cdfs,
                    probability_sarn - probability0,
                    cdfs - cdf0,
                    grid,
                ),
                dim=2,
            )
            query = self.endpoint_embedding(features)
            query = query + self.progress_position_embedding.to(query.dtype)
            decoded = self.decoder(tgt=query, memory=relation_memory.float())
            tokens = self.output_norm(decoded)
        return {"features": features, "tokens": tokens}


class FTEBNaturalParameterBridge(nn.Module):
    """Add a learned per-bin field around the fixed geometric endpoint."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        learned_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "FTEB bridge needs at least 16 bins")
        _require(token_dim >= 8, "FTEB bridge token width is too small")
        _require(
            math.isfinite(float(learned_residual_scale))
            and float(learned_residual_scale) >= 0.0,
            "FTEB learned residual scale must be finite and non-negative",
        )
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        # This is a fixed architecture constant, not a parameter or buffer.
        # Keeping it out of state_dict preserves strict loading of A15 artifacts.
        self.learned_residual_scale = float(learned_residual_scale)
        self.field_output = nn.Linear(token_dim, 2)
        nn.init.zeros_(self.field_output.weight)
        nn.init.zeros_(self.field_output.bias)
        self.register_buffer(
            "layer_times",
            torch.arange(1, BRIDGE_LAYERS + 1, dtype=torch.float32)
            / float(BRIDGE_LAYERS),
        )

    def forward(
        self,
        q0: torch.Tensor,
        q_sarn: torch.Tensor,
        progress_tokens: torch.Tensor,
        *,
        correction_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            q0.shape == q_sarn.shape
            == (progress_tokens.shape[0], self.progress_bins)
            and progress_tokens.shape
            == (q0.shape[0], self.progress_bins, self.token_dim),
            "FTEB bridge endpoint/token shapes differ",
        )
        _require(
            correction_available.shape == (q0.shape[0],)
            and correction_available.dtype == torch.bool
            and correction_available.device == q0.device,
            "FTEB bridge availability is malformed",
        )
        _validate_posterior(q0, label="FTEB q0")
        _validate_posterior(q_sarn, label="FTEB q_sarn")
        with torch.autocast(device_type=q0.device.type, enabled=False):
            endpoint = fixed_geometric_natural_parameter_base(q0, q_sarn)
            raw = endpoint["raw"]
            raw_log_density = endpoint["raw_log_density"]
            sarn_log_density = endpoint["sarn_log_density"]
            base_logits = endpoint["base_logits"]
            proposed_geometric_base = endpoint["geometric_base"]
            geometric_base = torch.where(
                correction_available[:, None], proposed_geometric_base, raw
            )
            field_logits = self.field_output(progress_tokens.float())
            gain = MAX_ENDPOINT_RESIDUAL_GAIN * torch.tanh(
                field_logits[:, :, 0]
            )
            free = MAX_FREE_FIELD * torch.tanh(field_logits[:, :, 1])
            endpoint_log_ratio = (sarn_log_density - raw_log_density).clamp(
                -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT
            )
            uncentered_delta = gain * endpoint_log_ratio + free
            delta_center = (raw * uncentered_delta).sum(dim=1, keepdim=True)
            centered_delta = uncentered_delta - delta_center
            effective_centered_delta = (
                self.learned_residual_scale * centered_delta
            )
            delta_zero = (effective_centered_delta == 0.0).all(dim=1)
            base_direction = base_logits - raw_log_density
            raw_cdf = raw.cumsum(dim=1)
            raw_mean, raw_variance = _posterior_moments(raw)
            geometric_cdf = geometric_base.cumsum(dim=1)
            geometric_mean, geometric_variance = _posterior_moments(
                geometric_base
            )
            layer_posteriors: list[torch.Tensor] = []
            layer_cdfs: list[torch.Tensor] = []
            layer_means: list[torch.Tensor] = []
            layer_variances: list[torch.Tensor] = []
            layer_fields: list[torch.Tensor] = []
            for layer_index, layer_time in enumerate(self.layer_times):
                proposed_layer_field = layer_time * (
                    base_direction + effective_centered_delta
                )
                if layer_index == BRIDGE_LAYERS - 1:
                    final_logits = base_logits + effective_centered_delta
                    proposed = torch.softmax(final_logits, dim=1)
                    exact_geometric_ste = (
                        proposed - proposed.detach() + proposed_geometric_base
                    )
                    proposed = torch.where(
                        delta_zero[:, None], exact_geometric_ste, proposed
                    )
                else:
                    proposed = torch.softmax(
                        raw_log_density + proposed_layer_field, dim=1
                    )
                posterior = torch.where(correction_available[:, None], proposed, raw)
                layer_field = torch.where(
                    correction_available[:, None],
                    proposed_layer_field,
                    torch.zeros_like(proposed_layer_field),
                )
                proposed_mean, proposed_variance = _posterior_moments(posterior)
                mean_identity = proposed_mean - proposed_mean.detach() + raw_mean
                variance_identity = (
                    proposed_variance - proposed_variance.detach() + raw_variance
                )
                mean = torch.where(
                    correction_available, proposed_mean, mean_identity
                )
                variance = torch.where(
                    correction_available, proposed_variance, variance_identity
                )
                layer_posteriors.append(posterior)
                layer_cdfs.append(posterior.cumsum(dim=1))
                layer_means.append(mean)
                layer_variances.append(variance)
                layer_fields.append(layer_field)
            posterior_stack = torch.stack(layer_posteriors, dim=1)
            cdf_stack = torch.stack(layer_cdfs, dim=1)
            mean_stack = torch.stack(layer_means, dim=1)
            variance_stack = torch.stack(layer_variances, dim=1)
            field_stack = torch.stack(layer_fields, dim=1)
            final = posterior_stack[:, -1]
            final_cdf = cdf_stack[:, -1]
            final_mean = mean_stack[:, -1]
            final_variance = variance_stack[:, -1]
        return {
            "progress_posterior": final,
            "progress_cdf": final_cdf,
            "mean": final_mean,
            "variance": final_variance,
            "layer_posteriors": posterior_stack,
            "layer_cdfs": cdf_stack,
            "layer_means": mean_stack,
            "layer_variances": variance_stack,
            "layer_fields": field_stack,
            "field_logits": field_logits.float(),
            "endpoint_gain": gain,
            "free_field": free,
            "endpoint_log_ratio": endpoint_log_ratio,
            "uncentered_delta": uncentered_delta,
            "delta_center": delta_center.squeeze(1),
            "centered_delta": centered_delta,
            "effective_centered_delta": effective_centered_delta,
            "learned_residual_scale": self.learned_residual_scale,
            "delta_zero": delta_zero,
            "base_direction": base_direction,
            "base_logits": base_logits,
            "proposed_geometric_base": proposed_geometric_base,
            "geometric_base": geometric_base,
            "geometric_base_cdf": geometric_cdf,
            "geometric_base_mean": geometric_mean,
            "geometric_base_variance": geometric_variance,
            "raw_log_density": raw_log_density,
            "raw_cdf": raw_cdf,
            "raw_mean": raw_mean,
            "raw_variance": raw_variance,
            "path_field_energy": (raw[:, None] * field_stack.square()).sum(dim=2),
            "learned_delta_energy": (
                raw
                * torch.where(
                    correction_available[:, None],
                    effective_centered_delta,
                    torch.zeros_like(effective_centered_delta),
                ).square()
            ).sum(dim=1),
        }


class A15FTEBCorrection(nn.Module):
    """Correction-only A15 core over externally supplied frozen twin endpoints."""

    def __init__(
        self,
        *,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = DEFAULT_MEMORY_GRID_SIZE,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        learned_residual_scale: float = 1.0,
        use_relation_memory: bool = True,
    ) -> None:
        super().__init__()
        _require(
            isinstance(use_relation_memory, bool),
            "FTEB relation-memory switch must be boolean",
        )
        self.progress_bins = int(progress_bins)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.memory_grid_size = int(memory_grid_size)
        self.learned_residual_scale = float(learned_residual_scale)
        # Fixed architectural treatment used only by a predeclared ablation.
        # It is deliberately neither learned nor serialized, so legacy A15
        # state keys and strict-load behavior remain unchanged.
        self.use_relation_memory = use_relation_memory
        self.stride8_aligner = _SCORTDifferentiableAligner(feature_stride=8)
        self.stride16_aligner = _SCORTDifferentiableAligner(feature_stride=16)
        self.shared_relation_encoder = FTEBSharedDualScaleRelationEncoder(
            relation_channels=self.relation_channels,
            token_dim=self.token_dim,
            memory_grid_size=self.memory_grid_size,
        )
        self.geometry_token_encoder = FTEBGeometryTokenEncoder(
            token_dim=self.token_dim
        )
        self.bin_decoder = FTEBBinDecoder(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
            attention_heads=self.attention_heads,
            decoder_layers=self.decoder_layers,
        )
        self.natural_parameter_bridge = FTEBNaturalParameterBridge(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
            learned_residual_scale=self.learned_residual_scale,
        )

    def _validate_inputs(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_posterior: torch.Tensor,
        sarn_encoder_features: Mapping[str, torch.Tensor],
        sarn_support_mask: torch.Tensor,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _require(
            raw_posterior.shape == sarn_posterior.shape
            and raw_posterior.ndim == 2
            and raw_posterior.shape[1] == self.progress_bins,
            "FTEB Raw/SARN endpoint posterior shapes differ",
        )
        _validate_posterior(raw_posterior, label="FTEB q0")
        _validate_posterior(sarn_posterior, label="FTEB q_sarn")
        batch = raw_posterior.shape[0]
        _require(
            all(name in raw_encoder_features for name in ("stride8", "stride16"))
            and all(name in sarn_encoder_features for name in ("stride8", "stride16")),
            "FTEB twin endpoint features need stride8 and stride16",
        )
        raw8 = raw_encoder_features["stride8"]
        raw16 = raw_encoder_features["stride16"]
        sarn8 = sarn_encoder_features["stride8"]
        sarn16 = sarn_encoder_features["stride16"]
        _require(
            raw8.ndim == sarn8.ndim == 4
            and raw8.shape == sarn8.shape
            and raw8.shape[:2] == (batch, RAW_STRIDE8_CHANNELS)
            and raw16.ndim == sarn16.ndim == 4
            and raw16.shape == sarn16.shape
            and raw16.shape[:2] == (batch, EFFICIENTNET_B0_MIDDLE_FEATURES),
            "FTEB frozen twin feature shapes differ",
        )
        _require(
            sarn_support_mask.ndim == 4
            and sarn_support_mask.shape[0] == batch
            and sarn_support_mask.shape[1] == 1
            and sarn_support_mask.shape[2] >= 32
            and sarn_support_mask.shape[3] >= 32
            and sarn_active.shape == (batch,)
            and sarn_active.dtype == torch.bool
            and raw_to_sarn_homography.shape == (batch, 3, 3),
            "FTEB support/activity/homography shapes differ",
        )
        tensors = (
            raw_posterior,
            sarn_posterior,
            raw8,
            raw16,
            sarn8,
            sarn16,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        _require(
            all(value.device == raw_posterior.device for value in tensors),
            "FTEB inputs must share one device",
        )
        _require(
            all(
                value.is_floating_point()
                for value in (
                    raw8,
                    raw16,
                    sarn8,
                    sarn16,
                    sarn_support_mask,
                    raw_to_sarn_homography,
                )
            ),
            "FTEB features/support/homography must be floating",
        )
        _require(
            all(
                bool(torch.isfinite(value).all())
                for value in (raw8, raw16, sarn8, sarn16)
            ),
            "FTEB frozen twin features are non-finite",
        )
        return raw8, raw16, sarn8, sarn16

    def forward(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_posterior: torch.Tensor,
        sarn_encoder_features: Mapping[str, torch.Tensor],
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        raw8, raw16, sarn8, sarn16 = self._validate_inputs(
            raw_posterior,
            raw_encoder_features,
            sarn_posterior,
            sarn_encoder_features,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        q0 = raw_posterior.detach().float()
        q_sarn = sarn_posterior.detach().float()
        input_hw = tuple(int(value) for value in sarn_support_mask.shape[-2:])
        alignment8 = self.stride8_aligner(
            sarn_features=sarn8.detach(),
            sarn_support_mask=sarn_support_mask.detach(),
            raw_to_sarn_homography=raw_to_sarn_homography.detach(),
            input_hw=input_hw,
        )
        alignment16 = self.stride16_aligner(
            sarn_features=sarn16.detach(),
            sarn_support_mask=sarn_support_mask.detach(),
            raw_to_sarn_homography=raw_to_sarn_homography.detach(),
            input_hw=input_hw,
        )
        relation_available = (
            sarn_active.detach()
            & alignment8["transform_valid"]
            & alignment16["transform_valid"]
            & alignment8["support_valid"]
            & alignment16["support_valid"]
        )
        relation = self.shared_relation_encoder(
            raw_stride8=raw8.detach(),
            aligned_sarn_stride8=alignment8["aligned_sarn"],
            support_stride8=alignment8["common_support"],
            raw_stride16=raw16.detach(),
            aligned_sarn_stride16=alignment16["aligned_sarn"],
            support_stride16=alignment16["common_support"],
            available=relation_available,
        )
        geometry = self.geometry_token_encoder(
            q0,
            q_sarn,
            raw_to_sarn_homography,
            alignment8["support_area"],
            alignment16["support_area"],
            available=relation_available,
        )
        relation_memory = (
            relation["memory"]
            if self.use_relation_memory
            else torch.zeros_like(relation["memory"])
        )
        memory = torch.cat((relation_memory, geometry["token"]), dim=1)
        decoded = self.bin_decoder(q0, q_sarn, memory)
        bridge = self.natural_parameter_bridge(
            q0,
            q_sarn,
            decoded["tokens"],
            correction_available=relation_available,
        )
        proposed_q_sarn_mean, proposed_q_sarn_variance = _posterior_moments(
            q_sarn
        )
        proposed_q_sarn_cdf = q_sarn.cumsum(dim=1)
        sarn_endpoint_posterior = torch.where(
            relation_available[:, None], q_sarn, q0
        )
        sarn_endpoint_cdf = torch.where(
            relation_available[:, None],
            proposed_q_sarn_cdf,
            bridge["raw_cdf"],
        )
        sarn_endpoint_mean = torch.where(
            relation_available,
            proposed_q_sarn_mean,
            bridge["raw_mean"],
        )
        sarn_endpoint_variance = torch.where(
            relation_available,
            proposed_q_sarn_variance,
            bridge["raw_variance"],
        )
        proposed_q_sarn_standard_deviation = torch.sqrt(
            proposed_q_sarn_variance.clamp_min(0.0)
        )
        sarn_endpoint_standard_deviation = torch.where(
            relation_available,
            proposed_q_sarn_standard_deviation,
            torch.sqrt(bridge["raw_variance"].clamp_min(0.0)),
        )
        correction_active = relation_available[:, None].expand(-1, BRIDGE_LAYERS)
        return {
            "architecture": A15_ARCHITECTURE,
            "progress_posterior": bridge["progress_posterior"],
            "progress_cdf": bridge["progress_cdf"],
            "mean": bridge["mean"],
            "variance": bridge["variance"],
            "standard_deviation": torch.sqrt(bridge["variance"].clamp_min(0.0)),
            "raw_anchor_posterior": q0,
            "raw_anchor_cdf": bridge["raw_cdf"],
            "raw_anchor_mean": bridge["raw_mean"],
            "raw_anchor_variance": bridge["raw_variance"],
            "raw_anchor_standard_deviation": torch.sqrt(
                bridge["raw_variance"].clamp_min(0.0)
            ),
            "sarn_endpoint_posterior": sarn_endpoint_posterior,
            "sarn_endpoint_cdf": sarn_endpoint_cdf,
            "sarn_endpoint_mean": sarn_endpoint_mean,
            "sarn_endpoint_variance": sarn_endpoint_variance,
            "sarn_endpoint_standard_deviation": (
                sarn_endpoint_standard_deviation
            ),
            "proposed_sarn_endpoint_posterior": q_sarn,
            "proposed_sarn_endpoint_cdf": proposed_q_sarn_cdf,
            "proposed_sarn_endpoint_mean": proposed_q_sarn_mean,
            "proposed_sarn_endpoint_variance": proposed_q_sarn_variance,
            "proposed_sarn_endpoint_standard_deviation": (
                proposed_q_sarn_standard_deviation
            ),
            "geometric_base": bridge["geometric_base"],
            "proposed_geometric_base": bridge["proposed_geometric_base"],
            "geometric_base_cdf": bridge["geometric_base_cdf"],
            "geometric_base_mean": bridge["geometric_base_mean"],
            "geometric_base_variance": bridge["geometric_base_variance"],
            "geometric_base_standard_deviation": torch.sqrt(
                bridge["geometric_base_variance"].clamp_min(0.0)
            ),
            "layer_posteriors": bridge["layer_posteriors"],
            "layer_cdfs": bridge["layer_cdfs"],
            "layer_means": bridge["layer_means"],
            "layer_variances": bridge["layer_variances"],
            "layer_fields": bridge["layer_fields"],
            "path_field_energy": bridge["path_field_energy"],
            "learned_delta_energy": bridge["learned_delta_energy"],
            "field_logits": bridge["field_logits"],
            "endpoint_gain": bridge["endpoint_gain"],
            "free_field": bridge["free_field"],
            "endpoint_log_ratio": bridge["endpoint_log_ratio"],
            "uncentered_delta": bridge["uncentered_delta"],
            "delta_center": bridge["delta_center"],
            "centered_delta": bridge["centered_delta"],
            "effective_centered_delta": bridge["effective_centered_delta"],
            "learned_residual_scale": bridge["learned_residual_scale"],
            "delta_zero": bridge["delta_zero"],
            "base_direction": bridge["base_direction"],
            "base_logits": bridge["base_logits"],
            "raw_log_density": bridge["raw_log_density"],
            "correction_active": correction_active,
            "relation_available": relation_available,
            "sarn_active": sarn_active,
            "shared_projective_relation": relation,
            "geometry_token": geometry,
            "bin_endpoint_features": decoded["features"],
            "progress_tokens": decoded["tokens"],
            "relation_memory": memory,
            "relation_memory_enabled": self.use_relation_memory,
            "stride8_alignment": alignment8,
            "stride16_alignment": alignment16,
            "physical_outputs": {
                "progress_mean": bridge["mean"],
                "progress_variance": bridge["variance"],
            },
        }


def fteb_parameter_counts(model: A15FTEBCorrection) -> dict[str, int]:
    """Return the executable A15 correction-core parameter decomposition."""

    _require(isinstance(model, A15FTEBCorrection), "target is not A15 FTEB")

    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    components = {
        "shared_relation_encoder": count(model.shared_relation_encoder),
        "geometry_token_encoder": count(model.geometry_token_encoder),
        "bin_decoder": count(model.bin_decoder),
        "natural_parameter_bridge": count(model.natural_parameter_bridge),
    }
    correction = int(sum(components.values()))
    return {
        **components,
        "aligners": 0,
        "frozen_twin_anchor": 0,
        "raw_anchor": 0,
        "correction": correction,
        "total": correction,
        "component_sum": correction,
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "progress_bins": model.progress_bins,
        "bridge_layers": BRIDGE_LAYERS,
        "relation_memory_tokens": 2 * model.memory_grid_size**2 + 1,
    }


__all__ = [
    "A15_ARCHITECTURE",
    "A15FTEBCorrection",
    "BIN_FEATURES",
    "BRIDGE_LAYERS",
    "FTEBBinDecoder",
    "FTEBGeometryTokenEncoder",
    "FTEBNaturalParameterBridge",
    "FTEBSharedDualScaleRelationEncoder",
    "GEOMETRY_FEATURES",
    "MAX_ENDPOINT_RESIDUAL_GAIN",
    "MAX_FREE_FIELD",
    "PROBABILITY_EPSILON",
    "fixed_geometric_natural_parameter_base",
    "frozen_twin_endpoint_forward",
    "fteb_parameter_counts",
]
