"""Hierarchical relation-moment network built from native ReMST blocks.

The publication architecture is a dual-view network rather than a named
classification backbone with a terminal correction head.  A shared mobile
inverted-bottleneck stem is interrupted at scales 8 and 16 by ReMST blocks.
Each block writes a bounded relation residual back into the feature stream and
performs one exact first-moment information projection of the current progress
posterior.  Missing geometry selects the unmodified Raw posterior at the outer
boundary.

The foundation initializer can import the convolutional stages and scalar readout
from the established terminal direct checkpoint.  That is weight provenance,
not the public architecture identity; after import the stages are owned once by
this model and the two relation-moment blocks are part of its forward graph.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from experiments.a11_scort import _SCORTDifferentiableAligner
from experiments.a15_2_mett import (
    DEFAULT_POSTERIOR_SCALE,
    MOMENT_SOLVER_LIMIT,
    MOMENT_SOLVER_STEPS,
    MomentExactEfficientNetB0Anchor,
    load_moment_exact_anchor_from_direct_checkpoint,
)
from experiments.a15_fteb import (
    BRIDGE_LAYERS,
    DEFAULT_PROGRESS_BINS,
    FTEBGeometryTokenEncoder,
    PROBABILITY_EPSILON,
    _FTEBSharedRelationScale,
    _posterior_moments,
)


REMST_BLOCK_NET_ARCHITECTURE: Final[str] = (
    "ReMSTNet-Hierarchical-Dual-View-Relation-Moment-Backbone-v1"
)
COORDINATED_REMST_NET_ARCHITECTURE: Final[str] = (
    "ReMSTNet-Cross-Scale-Coordinated-Relation-Moment-Backbone-v2"
)
ADAPTIVE_REMST_NET_ARCHITECTURE: Final[str] = (
    "ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3"
)
PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE: Final[str] = (
    "ReMSTNet-Fixed-Budget-Progress-Mixing-Relation-Moment-Ablation-v3"
)
ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE: Final[str] = (
    "ReMSTNet-Adaptive-Budget-No-Progress-Mixing-Relation-Moment-Ablation-v3"
)
SCALE8_CHANNELS: Final[int] = 40
SCALE16_CHANNELS: Final[int] = 112
CONTEXT_CHANNELS: Final[int] = 1280
DEFAULT_RELATION_CHANNELS: Final[int] = 48
DEFAULT_TOKEN_DIM: Final[int] = 64
DEFAULT_FEATURE_RESIDUAL_SCALE: Final[float] = 0.10
SCALE8_MOMENT_BUDGET: Final[float] = 0.00875
SCALE16_MOMENT_BUDGET: Final[float] = 0.00875
CONTEXT_MOMENT_BUDGET: Final[float] = 0.00750
TOTAL_MOMENT_BUDGET: Final[float] = (
    SCALE8_MOMENT_BUDGET + SCALE16_MOMENT_BUDGET + CONTEXT_MOMENT_BUDGET
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _moment_exact_transport(
    posterior: torch.Tensor,
    target_mean: torch.Tensor,
    progress_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """KL-project a posterior onto one exact discrete first moment."""

    _require(
        posterior.ndim == 2
        and target_mean.shape == (posterior.shape[0],)
        and progress_grid.shape == (posterior.shape[1],),
        "ReMST moment-transport shapes differ",
    )
    with torch.autocast(device_type=posterior.device.type, enabled=False):
        base = posterior.float()
        grid = progress_grid.float()[None]
        base_mean, _base_variance = _posterior_moments(base)
        target = target_mean.float().clamp(
            torch.finfo(torch.float32).eps,
            1.0 - torch.finfo(torch.float32).eps,
        )
        log_base = torch.log(base.clamp_min(PROBABILITY_EPSILON))
        natural_tilt = torch.zeros_like(target[:, None])
        for _ in range(MOMENT_SOLVER_STEPS):
            proposed = torch.softmax(log_base + natural_tilt * grid, dim=1)
            observed_mean, observed_variance = _posterior_moments(proposed)
            natural_tilt = (
                natural_tilt
                + (target[:, None] - observed_mean[:, None])
                / observed_variance[:, None].clamp_min(1.0e-10)
            ).clamp(-MOMENT_SOLVER_LIMIT, MOMENT_SOLVER_LIMIT)
        proposed = torch.softmax(log_base + natural_tilt * grid, dim=1)
        zero_shift = target == base_mean
        exact_base_ste = proposed - proposed.detach() + base
        result = torch.where(zero_shift[:, None], exact_base_ste, proposed)
        mean, variance = _posterior_moments(result)
    return {
        "posterior": result,
        "mean": mean,
        "variance": variance,
        "natural_tilt": natural_tilt.squeeze(1),
        "target_mean": target,
        "zero_shift": zero_shift,
        "absolute_moment_error": torch.abs(mean - target),
    }


def _natural_path(
    raw: torch.Tensor,
    final: torch.Tensor,
    *,
    available: torch.Tensor,
) -> torch.Tensor:
    """Return eight natural-geodesic samples from Raw to the final posterior."""

    with torch.autocast(device_type=raw.device.type, enabled=False):
        rawf = raw.detach().float()
        finalf = final.float()
        raw_log = torch.log(rawf.clamp_min(PROBABILITY_EPSILON))
        final_log = torch.log(finalf.clamp_min(PROBABILITY_EPSILON))
        values: list[torch.Tensor] = []
        for index in range(BRIDGE_LAYERS):
            time = float(index + 1) / float(BRIDGE_LAYERS)
            if index == BRIDGE_LAYERS - 1:
                proposed = finalf
            else:
                proposed = torch.softmax(
                    (1.0 - time) * raw_log + time * final_log,
                    dim=1,
                )
            values.append(
                torch.where(available[:, None], proposed, rawf)
            )
    return torch.stack(values, dim=1)


class ReMSTBlock(nn.Module):
    """Align relations, rewrite one feature scale, and predict one moment step."""

    def __init__(
        self,
        *,
        feature_channels: int,
        feature_stride: int,
        moment_budget: float,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        memory_grid_size: int = 4,
        feature_residual_scale: float = DEFAULT_FEATURE_RESIDUAL_SCALE,
    ) -> None:
        super().__init__()
        _require(feature_channels >= 8, "ReMST feature width is too small")
        _require(feature_stride in (8, 16), "ReMST feature stride is unsupported")
        _require(moment_budget > 0.0, "ReMST moment budget must be positive")
        _require(
            feature_residual_scale > 0.0,
            "ReMST feature residual scale must be positive",
        )
        self.feature_channels = int(feature_channels)
        self.feature_stride = int(feature_stride)
        self.moment_budget = float(moment_budget)
        self.feature_residual_scale = float(feature_residual_scale)
        self.aligner = _SCORTDifferentiableAligner(
            feature_stride=self.feature_stride
        )
        self.relation_encoder = _FTEBSharedRelationScale(
            input_channels=self.feature_channels,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.feature_projection = nn.Conv2d(
            token_dim, self.feature_channels, kernel_size=1
        )
        self.moment_projection = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.zeros_(self.moment_projection.weight)
        nn.init.zeros_(self.moment_projection.bias)

    def forward(
        self,
        raw_features: torch.Tensor,
        sarn_features: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = raw_features.shape[0]
        _require(
            raw_features.shape == sarn_features.shape
            and raw_features.ndim == 4
            and raw_features.shape[1] == self.feature_channels,
            "ReMST block feature shapes differ",
        )
        _require(
            support_mask.ndim == 4
            and support_mask.shape[0] == batch
            and support_mask.shape[1] == 1
            and sarn_active.shape == (batch,)
            and sarn_active.dtype == torch.bool
            and raw_to_sarn_homography.shape == (batch, 3, 3),
            "ReMST block support/activity/geometry shapes differ",
        )
        input_hw = tuple(int(value) for value in support_mask.shape[-2:])
        alignment = self.aligner(
            sarn_features=sarn_features.detach(),
            sarn_support_mask=support_mask.detach(),
            raw_to_sarn_homography=raw_to_sarn_homography.detach(),
            input_hw=input_hw,
        )
        available = (
            sarn_active.detach()
            & alignment["transform_valid"]
            & alignment["support_valid"]
        )
        relation = self.relation_encoder(
            raw=raw_features,
            aligned_sarn=alignment["aligned_sarn"],
            common_support=alignment["common_support"],
            available=available,
        )
        with torch.autocast(device_type=raw_features.device.type, enabled=False):
            encoded = relation["encoded"].float()
            support = relation["active_support"].float()
            denominator = support.sum(dim=(2, 3)).clamp_min(1.0)
            pooled = (encoded * support).sum(dim=(2, 3)) / denominator
            evidence = self.moment_projection(pooled).squeeze(1)
            moment_shift = self.moment_budget * torch.tanh(evidence)
            moment_shift = torch.where(
                available, moment_shift, torch.zeros_like(moment_shift)
            )
            feature_residual = self.feature_residual_scale * torch.tanh(
                self.feature_projection(encoded)
            )
            feature_residual = feature_residual * support
            feature_residual = torch.where(
                available[:, None, None, None],
                feature_residual,
                torch.zeros_like(feature_residual),
            )
        return {
            "available": available,
            "aligned_sarn": alignment["aligned_sarn"],
            "support": support,
            "support_area": alignment["support_area"],
            "relation_features": encoded,
            "relation_memory": relation["memory"],
            "pooled_relation": pooled,
            "moment_evidence": evidence,
            "moment_shift": moment_shift,
            "feature_residual": feature_residual,
        }


class _ProgressRelationCrossAttention(nn.Module):
    """One compact progress-query to relation-memory attention layer."""

    def __init__(
        self,
        *,
        token_dim: int,
        attention_heads: int,
        use_progress_mixing: bool = False,
    ) -> None:
        super().__init__()
        _require(
            token_dim % attention_heads == 0,
            "progress attention width and heads differ",
        )
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = self.token_dim // self.attention_heads
        self.query_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(token_dim)
        self.query_projection = nn.Linear(token_dim, token_dim)
        self.key_projection = nn.Linear(token_dim, token_dim)
        self.value_projection = nn.Linear(token_dim, token_dim)
        self.output_projection = nn.Linear(token_dim, token_dim)
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, 2 * token_dim),
            nn.GELU(),
            nn.Linear(2 * token_dim, token_dim),
        )
        self.progress_mixing_norm: nn.LayerNorm | None = None
        self.progress_depthwise: nn.Conv1d | None = None
        self.progress_pointwise: nn.Conv1d | None = None
        if use_progress_mixing:
            self.progress_mixing_norm = nn.LayerNorm(token_dim)
            self.progress_depthwise = nn.Conv1d(
                token_dim,
                token_dim,
                kernel_size=5,
                padding=2,
                groups=token_dim,
            )
            self.progress_pointwise = nn.Conv1d(
                token_dim, token_dim, kernel_size=1
            )

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        batch, bins, _channels = query.shape
        memory_tokens = memory.shape[1]
        q = self.query_projection(self.query_norm(query)).reshape(
            batch, bins, self.attention_heads, self.head_dim
        ).transpose(1, 2)
        normalized_memory = self.memory_norm(memory)
        k = self.key_projection(normalized_memory).reshape(
            batch, memory_tokens, self.attention_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value_projection(normalized_memory).reshape(
            batch, memory_tokens, self.attention_heads, self.head_dim
        ).transpose(1, 2)
        weights = torch.softmax(
            torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim),
            dim=3,
        )
        context = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, bins, self.token_dim
        )
        query = query + self.output_projection(context)
        query = query + self.feed_forward(self.feed_forward_norm(query))
        if (
            self.progress_mixing_norm is not None
            and self.progress_depthwise is not None
            and self.progress_pointwise is not None
        ):
            mixed = self.progress_mixing_norm(query).transpose(1, 2)
            mixed = self.progress_depthwise(mixed)
            mixed = torch.nn.functional.gelu(mixed)
            mixed = self.progress_pointwise(mixed).transpose(1, 2)
            query = query + mixed
        return query


class ProgressConditionedRelationDecoder(nn.Module):
    """Decode endpoint-bin queries with relation memory but no bin self-attention."""

    def __init__(
        self,
        *,
        progress_bins: int,
        token_dim: int,
        attention_heads: int,
        decoder_layers: int,
        use_progress_mixing: bool = False,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.progress_position_embedding = nn.Parameter(
            torch.zeros(1, progress_bins, token_dim)
        )
        nn.init.trunc_normal_(self.progress_position_embedding, std=0.02)
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(9, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.layers = nn.ModuleList(
            _ProgressRelationCrossAttention(
                token_dim=token_dim,
                attention_heads=attention_heads,
                use_progress_mixing=use_progress_mixing,
            )
            for _ in range(decoder_layers)
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, progress_bins, dtype=torch.float32),
        )

    def forward(
        self,
        raw_posterior: torch.Tensor,
        sarn_posterior: torch.Tensor,
        relation_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            raw_posterior.shape
            == sarn_posterior.shape
            == (relation_memory.shape[0], self.progress_bins)
            and relation_memory.ndim == 3
            and relation_memory.shape[2] == self.token_dim,
            "progress decoder endpoint/memory shapes differ",
        )
        raw = raw_posterior.detach().float()
        sarn = sarn_posterior.detach().float()
        log_raw = torch.log(raw.clamp_min(PROBABILITY_EPSILON))
        log_sarn = torch.log(sarn.clamp_min(PROBABILITY_EPSILON))
        cdf_raw = raw.cumsum(dim=1)
        cdf_sarn = sarn.cumsum(dim=1)
        grid = self.progress_grid.float()[None].expand(raw.shape[0], -1)
        features = torch.stack(
            (
                raw,
                sarn,
                log_raw,
                log_sarn,
                cdf_raw,
                cdf_sarn,
                sarn - raw,
                cdf_sarn - cdf_raw,
                grid,
            ),
            dim=2,
        )
        query = self.endpoint_embedding(features)
        query = query + self.progress_position_embedding.float()
        for layer in self.layers:
            query = layer(query, relation_memory.float())
        return {"features": features, "tokens": self.output_norm(query)}


class CrossScaleMomentCoordinator(nn.Module):
    """Predict one bounded total shift and allocate it across three stages.

    The total shift is progress-aware: 128 endpoint-bin queries attend to both
    relation memories, an explicit geometry token, and the feature-rewrite
    context token.  A non-negative simplex allocation then decomposes that one
    shift into the two ReMST blocks and the context stage, so hierarchical
    updates cannot cancel one another by construction.
    """

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = 4,
        use_progress_mixing: bool = False,
        learnable_budget_gain: bool = False,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "coordinator needs at least 16 bins")
        _require(
            token_dim >= 8
            and attention_heads >= 1
            and token_dim % attention_heads == 0,
            "coordinator token width and heads differ",
        )
        _require(decoder_layers >= 1, "coordinator needs a decoder layer")
        _require(memory_grid_size >= 2, "coordinator memory grid is too small")
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.memory_grid_size = int(memory_grid_size)
        self.use_progress_mixing = bool(use_progress_mixing)
        self.learnable_budget_gain = bool(learnable_budget_gain)
        self.geometry_encoder = FTEBGeometryTokenEncoder(token_dim=token_dim)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(CONTEXT_CHANNELS),
            nn.Linear(CONTEXT_CHANNELS, token_dim, bias=False),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim, bias=False),
        )
        memory_tokens = 2 * memory_grid_size * memory_grid_size + 2
        self.memory_position_embedding = nn.Parameter(
            torch.zeros(1, memory_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.memory_position_embedding, std=0.02)
        self.bin_decoder = ProgressConditionedRelationDecoder(
            progress_bins=progress_bins,
            token_dim=token_dim,
            attention_heads=attention_heads,
            decoder_layers=decoder_layers,
            use_progress_mixing=use_progress_mixing,
        )
        # Two progress-aware heads preserve the successful symmetric scalar
        # readout capacity while the stage allocation remains a distinct task.
        self.total_output_a = nn.Linear(token_dim, 1)
        self.total_output_b = nn.Linear(token_dim, 1)
        self.context_responsibility = nn.Sequential(
            nn.LayerNorm(4 * token_dim),
            nn.Linear(4 * token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, 1),
        )
        for module in (
            self.total_output_a,
            self.total_output_b,
            self.context_responsibility[-1],
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        if self.learnable_budget_gain:
            self.budget_gain_parameter = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("budget_gain_parameter", None)

    def forward(
        self,
        *,
        raw_posterior: torch.Tensor,
        sarn_posterior: torch.Tensor,
        relation_memory8: torch.Tensor,
        relation_memory16: torch.Tensor,
        pooled_relation8: torch.Tensor,
        pooled_relation16: torch.Tensor,
        local_evidence8: torch.Tensor,
        local_evidence16: torch.Tensor,
        adapted_representation: torch.Tensor,
        raw_representation: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
        support_area8: torch.Tensor,
        support_area16: torch.Tensor,
        available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = raw_posterior.shape[0]
        _require(
            raw_posterior.shape == sarn_posterior.shape
            == (batch, self.progress_bins)
            and relation_memory8.shape
            == relation_memory16.shape
            == (
                batch,
                self.memory_grid_size * self.memory_grid_size,
                self.token_dim,
            )
            and pooled_relation8.shape
            == pooled_relation16.shape
            == (batch, self.token_dim)
            and local_evidence8.shape
            == local_evidence16.shape
            == available.shape
            == (batch,)
            and adapted_representation.shape
            == raw_representation.shape
            == (batch, CONTEXT_CHANNELS),
            "coordinator input shapes differ",
        )
        geometry = self.geometry_encoder(
            raw_posterior,
            sarn_posterior,
            raw_to_sarn_homography,
            support_area8,
            support_area16,
            available=available,
        )
        with torch.autocast(
            device_type=raw_posterior.device.type, enabled=False
        ):
            context_delta = (
                adapted_representation.float()
                - raw_representation.detach().float()
            )
            context_token = self.context_projection(context_delta)[:, None]
            context_token = context_token * available[:, None, None].to(
                context_token.dtype
            )
            memory = torch.cat(
                (
                    relation_memory8.float(),
                    relation_memory16.float(),
                    geometry["token"].float(),
                    context_token.float(),
                ),
                dim=1,
            )
            memory = memory + self.memory_position_embedding.float()
            memory = memory * available[:, None, None].to(memory.dtype)
            decoded = self.bin_decoder(
                raw_posterior,
                sarn_posterior,
                memory,
            )
            logits_a = self.total_output_a(decoded["tokens"]).squeeze(2)
            logits_b = self.total_output_b(decoded["tokens"]).squeeze(2)
            endpoint = sarn_posterior.detach().float()
            progress_evidence = 0.5 * (
                (endpoint * logits_a).sum(dim=1)
                + (endpoint * logits_b).sum(dim=1)
            )
            responsibility_features = torch.cat(
                (
                    pooled_relation8.float(),
                    pooled_relation16.float(),
                    geometry["token"].squeeze(1).float(),
                    context_token.squeeze(1).float(),
                ),
                dim=1,
            )
            context_evidence = self.context_responsibility(
                responsibility_features
            ).squeeze(1)
            # Local block evidence and the progress-aware decoder jointly
            # determine one total correction.  The local heads therefore
            # receive final-read gradients, while their three-way allocation
            # cannot introduce an opposite-sign cancellation.
            bounded_local_evidence = torch.stack(
                (
                    torch.tanh(local_evidence8.float()),
                    torch.tanh(local_evidence16.float()),
                    torch.tanh(context_evidence),
                ),
                dim=1,
            )
            total_evidence = progress_evidence + bounded_local_evidence.mean(
                dim=1
            )
            budget_gain = torch.ones(
                (), device=total_evidence.device, dtype=total_evidence.dtype
            )
            if self.budget_gain_parameter is not None:
                budget_gain = 1.0 + 0.5 * torch.tanh(
                    self.budget_gain_parameter.float()
                )
            total_shift = (
                TOTAL_MOMENT_BUDGET
                * budget_gain
                * torch.tanh(total_evidence)
            )
            total_shift = torch.where(
                available, total_shift, torch.zeros_like(total_shift)
            )
            allocation_logits = torch.stack(
                (
                    bounded_local_evidence[:, 0],
                    bounded_local_evidence[:, 1],
                    bounded_local_evidence[:, 2],
                ),
                dim=1,
            )
            allocation = torch.softmax(allocation_logits, dim=1)
            allocation = torch.where(
                available[:, None],
                allocation,
                torch.full_like(allocation, 1.0 / 3.0),
            )
            stage_shifts = total_shift[:, None] * allocation
        return {
            "progress_evidence": progress_evidence,
            "total_evidence": total_evidence,
            "total_shift": total_shift,
            "budget_gain": budget_gain.expand(batch),
            "allocation": allocation,
            "stage_shifts": stage_shifts,
            "context_token": context_token.squeeze(1),
            "geometry_token": geometry["token"].squeeze(1),
            "progress_tokens": decoded["tokens"],
        }


class ReMSTBlockNet(nn.Module):
    """Native dual-view backbone with hierarchical relation-moment updates."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        posterior_scale: float = DEFAULT_POSTERIOR_SCALE,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        memory_grid_size: int = 4,
    ) -> None:
        super().__init__()
        _require(progress_bins == 128, "ReMSTNet pilot uses 128 progress bins")
        foundation = MomentExactEfficientNetB0Anchor(
            progress_bins=progress_bins,
            initial_scale=posterior_scale,
        )
        # Functional names intentionally describe the new architecture.  The
        # initializer below records where their starting weights came from.
        self.shared_scale8_encoder = foundation.raw_encoder.to_stride8
        self.shared_scale16_encoder = foundation.raw_encoder.to_stride16
        self.shared_context_encoder = foundation.raw_encoder.to_final
        self.moment_readout = foundation.raw_posterior_head
        self.progress_bins = int(progress_bins)
        self.posterior_scale = float(posterior_scale)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.scale8_block = ReMSTBlock(
            feature_channels=SCALE8_CHANNELS,
            feature_stride=8,
            moment_budget=SCALE8_MOMENT_BUDGET,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.scale16_block = ReMSTBlock(
            feature_channels=SCALE16_CHANNELS,
            feature_stride=16,
            moment_budget=SCALE16_MOMENT_BUDGET,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, self.progress_bins, dtype=torch.float32),
        )
        self._freeze_foundation()

    @property
    def construction(self) -> dict[str, int | float]:
        return {
            "progress_bins": self.progress_bins,
            "posterior_scale": self.posterior_scale,
            "relation_channels": self.relation_channels,
            "token_dim": self.token_dim,
            "memory_grid_size": self.memory_grid_size,
        }

    def _foundation_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.shared_scale8_encoder,
            self.shared_scale16_encoder,
            self.shared_context_encoder,
            self.moment_readout,
        )

    def _freeze_foundation(self) -> None:
        for module in self._foundation_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def train(self, mode: bool = True) -> "ReMSTBlockNet":
        super().train(mode)
        # Frozen normalization statistics are part of the imported starting
        # function.  ReMST blocks remain in the requested mode.
        for module in self._foundation_modules():
            module.eval()
        return self

    def import_foundation(
        self, source: MomentExactEfficientNetB0Anchor
    ) -> None:
        """Import one terminal scalar reader into the functional stem."""

        loads = (
            self.shared_scale8_encoder.load_state_dict(
                source.raw_encoder.to_stride8.state_dict(), strict=True
            ),
            self.shared_scale16_encoder.load_state_dict(
                source.raw_encoder.to_stride16.state_dict(), strict=True
            ),
            self.shared_context_encoder.load_state_dict(
                source.raw_encoder.to_final.state_dict(), strict=True
            ),
            self.moment_readout.load_state_dict(
                source.raw_posterior_head.state_dict(), strict=True
            ),
        )
        _require(
            all(not value.missing_keys and not value.unexpected_keys for value in loads),
            "ReMSTNet foundation import does not load strictly",
        )
        self._freeze_foundation()

    def _base_encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        scale8 = self.shared_scale8_encoder(images)
        scale16 = self.shared_scale16_encoder(scale8)
        context = self.shared_context_encoder(scale16)
        representation = context.mean(dim=(2, 3))
        lifted = self.moment_readout.posterior_parameters(representation)
        return {
            "scale8": scale8,
            "scale16": scale16,
            "context": context,
            "representation": representation,
            "posterior": lifted["posterior"],
            "mean": lifted["posterior_mean"],
            "variance": lifted["posterior_variance"],
        }

    def forward(
        self,
        raw_view: torch.Tensor,
        sarn_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        _require(
            raw_view.shape == sarn_view.shape
            and raw_view.ndim == 4
            and raw_view.shape[1] == 3,
            "ReMSTNet input views differ",
        )
        self._freeze_foundation()
        with torch.no_grad():
            raw = self._base_encode(raw_view)
            sarn = self._base_encode(sarn_view)

        block8 = self.scale8_block(
            raw["scale8"],
            sarn["scale8"],
            support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        adapted_scale8 = raw["scale8"].detach() + block8["feature_residual"]
        adapted_scale16 = self.shared_scale16_encoder(adapted_scale8)
        block16 = self.scale16_block(
            adapted_scale16,
            sarn["scale16"],
            support_mask,
            sarn_active=sarn_active & block8["available"],
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        adapted_scale16 = adapted_scale16 + block16["feature_residual"]
        adapted_context = self.shared_context_encoder(adapted_scale16)
        adapted_representation = adapted_context.mean(dim=(2, 3))

        with torch.autocast(device_type=raw_view.device.type, enabled=False):
            point = self.moment_readout.point_projection
            adapted_logit = torch.nn.functional.linear(
                adapted_representation.float(),
                point.weight.float(),
                None if point.bias is None else point.bias.float(),
            ).squeeze(1)
            raw_logit = torch.nn.functional.linear(
                raw["representation"].detach().float(),
                point.weight.detach().float(),
                None if point.bias is None else point.bias.detach().float(),
            ).squeeze(1)
            context_evidence = adapted_logit - raw_logit
            context_shift = CONTEXT_MOMENT_BUDGET * torch.tanh(
                context_evidence
            )

            relation_available = block8["available"] & block16["available"]
            shift8 = torch.where(
                relation_available,
                block8["moment_shift"],
                torch.zeros_like(block8["moment_shift"]),
            )
            shift16 = torch.where(
                relation_available,
                block16["moment_shift"],
                torch.zeros_like(block16["moment_shift"]),
            )
            context_shift = torch.where(
                relation_available,
                context_shift,
                torch.zeros_like(context_shift),
            )

            stage8 = _moment_exact_transport(
                sarn["posterior"],
                sarn["mean"] + shift8,
                self.progress_grid,
            )
            stage16 = _moment_exact_transport(
                stage8["posterior"],
                stage8["mean"] + shift16,
                self.progress_grid,
            )
            terminal = _moment_exact_transport(
                stage16["posterior"],
                stage16["mean"] + context_shift,
                self.progress_grid,
            )
            final_posterior = torch.where(
                relation_available[:, None],
                terminal["posterior"],
                raw["posterior"],
            )
            final_mean, final_variance = _posterior_moments(final_posterior)
            endpoint_posterior = torch.where(
                relation_available[:, None],
                sarn["posterior"],
                raw["posterior"],
            )
            endpoint_mean, endpoint_variance = _posterior_moments(
                endpoint_posterior
            )
            layers = _natural_path(
                raw["posterior"],
                final_posterior,
                available=relation_available,
            )
            layer_cdfs = layers.cumsum(dim=2)
            layer_means = (layers * self.progress_grid[None, None]).sum(dim=2)
            layer_variances = (
                layers
                * (self.progress_grid[None, None] - layer_means[:, :, None]).square()
            ).sum(dim=2)
            learned_delta_energy = (
                shift8.square() + shift16.square() + context_shift.square()
            )

        return {
            "architecture": REMST_BLOCK_NET_ARCHITECTURE,
            "progress_posterior": final_posterior,
            "progress_cdf": final_posterior.cumsum(dim=1),
            "mean": final_mean,
            "variance": final_variance,
            "standard_deviation": torch.sqrt(final_variance.clamp_min(0.0)),
            "raw_anchor_posterior": raw["posterior"],
            "raw_anchor_cdf": raw["posterior"].cumsum(dim=1),
            "raw_anchor_mean": raw["mean"],
            "raw_anchor_variance": raw["variance"],
            "sarn_endpoint_posterior": endpoint_posterior,
            "sarn_endpoint_cdf": endpoint_posterior.cumsum(dim=1),
            "sarn_endpoint_mean": endpoint_mean,
            "sarn_endpoint_variance": endpoint_variance,
            "proposed_sarn_endpoint_posterior": sarn["posterior"],
            "proposed_sarn_endpoint_mean": sarn["mean"],
            "proposed_sarn_endpoint_variance": sarn["variance"],
            "geometric_base": endpoint_posterior,
            "geometric_base_cdf": endpoint_posterior.cumsum(dim=1),
            "geometric_base_mean": endpoint_mean,
            "geometric_base_variance": endpoint_variance,
            "tangent_base": endpoint_posterior,
            "tangent_base_cdf": endpoint_posterior.cumsum(dim=1),
            "tangent_base_mean": endpoint_mean,
            "tangent_base_variance": endpoint_variance,
            "layer_posteriors": layers,
            "layer_cdfs": layer_cdfs,
            "layer_means": layer_means,
            "layer_variances": layer_variances,
            "relation_available": relation_available,
            "correction_active": relation_available[:, None].expand(
                -1, BRIDGE_LAYERS
            ),
            "learned_delta_energy": learned_delta_energy,
            "scale8_posterior": stage8["posterior"],
            "scale16_posterior": stage16["posterior"],
            "scale8_moment_shift": shift8,
            "scale16_moment_shift": shift16,
            "context_moment_shift": context_shift,
            "total_moment_shift": shift8 + shift16 + context_shift,
            "scale8_feature_residual_rms": block8["feature_residual"]
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt(),
            "scale16_feature_residual_rms": block16["feature_residual"]
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt(),
            "moment_absolute_error": terminal["absolute_moment_error"],
        }


class CoordinatedReMSTNet(ReMSTBlockNet):
    """ReMSTNet v2 with progress-aware cross-scale moment coordination."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        posterior_scale: float = DEFAULT_POSTERIOR_SCALE,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        memory_grid_size: int = 4,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        use_progress_mixing: bool = False,
        learnable_budget_gain: bool = False,
    ) -> None:
        super().__init__(
            progress_bins=progress_bins,
            posterior_scale=posterior_scale,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.use_progress_mixing = bool(use_progress_mixing)
        self.learnable_budget_gain = bool(learnable_budget_gain)
        self.moment_coordinator = CrossScaleMomentCoordinator(
            progress_bins=progress_bins,
            token_dim=token_dim,
            attention_heads=attention_heads,
            decoder_layers=decoder_layers,
            memory_grid_size=memory_grid_size,
            use_progress_mixing=use_progress_mixing,
            learnable_budget_gain=learnable_budget_gain,
        )

    @property
    def construction(self) -> dict[str, int | float | str | bool]:
        variant = {
            (False, False): "cross_scale_coordinated_v2",
            (True, False): "progress_mixing_fixed_budget_ablation",
            (False, True): "adaptive_budget_no_progress_mixing_ablation",
            (True, True): "adaptive_budget_progress_mixing_v3",
        }[(self.use_progress_mixing, self.learnable_budget_gain)]
        construction: dict[str, int | float | str | bool] = {
            **super().construction,
            "architecture_variant": variant,
            "attention_heads": self.attention_heads,
            "decoder_layers": self.decoder_layers,
            "use_progress_mixing": self.use_progress_mixing,
            "learnable_budget_gain": self.learnable_budget_gain,
        }
        return construction

    @property
    def architecture(self) -> str:
        return {
            (False, False): COORDINATED_REMST_NET_ARCHITECTURE,
            (True, False): PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
            (False, True): ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
            (True, True): ADAPTIVE_REMST_NET_ARCHITECTURE,
        }[(self.use_progress_mixing, self.learnable_budget_gain)]

    def forward(
        self,
        raw_view: torch.Tensor,
        sarn_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        _require(
            raw_view.shape == sarn_view.shape
            and raw_view.ndim == 4
            and raw_view.shape[1] == 3,
            "coordinated ReMSTNet input views differ",
        )
        self._freeze_foundation()
        with torch.no_grad():
            raw = self._base_encode(raw_view)
            sarn = self._base_encode(sarn_view)

        block8 = self.scale8_block(
            raw["scale8"],
            sarn["scale8"],
            support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        adapted_scale8 = raw["scale8"].detach() + block8["feature_residual"]
        adapted_scale16 = self.shared_scale16_encoder(adapted_scale8)
        block16 = self.scale16_block(
            adapted_scale16,
            sarn["scale16"],
            support_mask,
            sarn_active=sarn_active & block8["available"],
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        adapted_scale16 = adapted_scale16 + block16["feature_residual"]
        adapted_context = self.shared_context_encoder(adapted_scale16)
        adapted_representation = adapted_context.mean(dim=(2, 3))
        relation_available = block8["available"] & block16["available"]
        coordination = self.moment_coordinator(
            raw_posterior=raw["posterior"],
            sarn_posterior=sarn["posterior"],
            relation_memory8=block8["relation_memory"],
            relation_memory16=block16["relation_memory"],
            pooled_relation8=block8["pooled_relation"],
            pooled_relation16=block16["pooled_relation"],
            local_evidence8=block8["moment_evidence"],
            local_evidence16=block16["moment_evidence"],
            adapted_representation=adapted_representation,
            raw_representation=raw["representation"],
            raw_to_sarn_homography=raw_to_sarn_homography,
            support_area8=block8["support_area"],
            support_area16=block16["support_area"],
            available=relation_available,
        )

        with torch.autocast(device_type=raw_view.device.type, enabled=False):
            shift8 = coordination["stage_shifts"][:, 0]
            shift16 = coordination["stage_shifts"][:, 1]
            context_shift = coordination["stage_shifts"][:, 2]
            stage8 = _moment_exact_transport(
                sarn["posterior"],
                sarn["mean"] + shift8,
                self.progress_grid,
            )
            stage16 = _moment_exact_transport(
                stage8["posterior"],
                stage8["mean"] + shift16,
                self.progress_grid,
            )
            terminal = _moment_exact_transport(
                stage16["posterior"],
                stage16["mean"] + context_shift,
                self.progress_grid,
            )
            final_posterior = torch.where(
                relation_available[:, None],
                terminal["posterior"],
                raw["posterior"],
            )
            final_mean, final_variance = _posterior_moments(final_posterior)
            endpoint_posterior = torch.where(
                relation_available[:, None],
                sarn["posterior"],
                raw["posterior"],
            )
            endpoint_mean, endpoint_variance = _posterior_moments(
                endpoint_posterior
            )
            layers = _natural_path(
                raw["posterior"],
                final_posterior,
                available=relation_available,
            )
            layer_cdfs = layers.cumsum(dim=2)
            layer_means = (layers * self.progress_grid[None, None]).sum(dim=2)
            layer_variances = (
                layers
                * (self.progress_grid[None, None] - layer_means[:, :, None]).square()
            ).sum(dim=2)
            learned_delta_energy = coordination["stage_shifts"].square().sum(
                dim=1
            )

        return {
            "architecture": self.architecture,
            "progress_posterior": final_posterior,
            "progress_cdf": final_posterior.cumsum(dim=1),
            "mean": final_mean,
            "variance": final_variance,
            "standard_deviation": torch.sqrt(final_variance.clamp_min(0.0)),
            "raw_anchor_posterior": raw["posterior"],
            "raw_anchor_cdf": raw["posterior"].cumsum(dim=1),
            "raw_anchor_mean": raw["mean"],
            "raw_anchor_variance": raw["variance"],
            "sarn_endpoint_posterior": endpoint_posterior,
            "sarn_endpoint_cdf": endpoint_posterior.cumsum(dim=1),
            "sarn_endpoint_mean": endpoint_mean,
            "sarn_endpoint_variance": endpoint_variance,
            "proposed_sarn_endpoint_posterior": sarn["posterior"],
            "proposed_sarn_endpoint_mean": sarn["mean"],
            "proposed_sarn_endpoint_variance": sarn["variance"],
            "geometric_base": endpoint_posterior,
            "geometric_base_cdf": endpoint_posterior.cumsum(dim=1),
            "geometric_base_mean": endpoint_mean,
            "geometric_base_variance": endpoint_variance,
            "tangent_base": endpoint_posterior,
            "tangent_base_cdf": endpoint_posterior.cumsum(dim=1),
            "tangent_base_mean": endpoint_mean,
            "tangent_base_variance": endpoint_variance,
            "layer_posteriors": layers,
            "layer_cdfs": layer_cdfs,
            "layer_means": layer_means,
            "layer_variances": layer_variances,
            "relation_available": relation_available,
            "correction_active": relation_available[:, None].expand(
                -1, BRIDGE_LAYERS
            ),
            "learned_delta_energy": learned_delta_energy,
            "scale8_posterior": stage8["posterior"],
            "scale16_posterior": stage16["posterior"],
            "scale8_moment_shift": shift8,
            "scale16_moment_shift": shift16,
            "context_moment_shift": context_shift,
            "total_moment_shift": coordination["total_shift"],
            "stage_moment_allocation": coordination["allocation"],
            "coordinator_total_evidence": coordination["total_evidence"],
            "moment_budget_gain": coordination["budget_gain"],
            "context_relation_token_norm": coordination["context_token"]
            .square()
            .sum(dim=1)
            .sqrt(),
            "scale8_feature_residual_rms": block8["feature_residual"]
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt(),
            "scale16_feature_residual_rms": block16["feature_residual"]
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt(),
            "moment_absolute_error": terminal["absolute_moment_error"],
        }


def initialize_remst_block_net(
    direct_checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ReMSTBlockNet, dict[str, Any]]:
    """Create ReMSTNet and import the terminal scalar-reader foundation."""

    source, source_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path,
        device="cpu",
    )
    model = ReMSTBlockNet(
        progress_bins=int(source_metadata["progress_bins"]),
        posterior_scale=float(source_metadata["initial_posterior_scale"]),
    )
    model.import_foundation(source)
    del source
    return model.to(torch.device(device)), source_metadata


def initialize_coordinated_remst_net(
    direct_checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CoordinatedReMSTNet, dict[str, Any]]:
    """Create coordinated ReMSTNet and import the same scalar foundation."""

    source, source_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path,
        device="cpu",
    )
    model = CoordinatedReMSTNet(
        progress_bins=int(source_metadata["progress_bins"]),
        posterior_scale=float(source_metadata["initial_posterior_scale"]),
    )
    model.import_foundation(source)
    del source
    return model.to(torch.device(device)), source_metadata


def initialize_adaptive_remst_net(
    direct_checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CoordinatedReMSTNet, dict[str, Any]]:
    """Create v3 with local progress mixing and a learned bounded budget."""

    source, source_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path,
        device="cpu",
    )
    model = CoordinatedReMSTNet(
        progress_bins=int(source_metadata["progress_bins"]),
        posterior_scale=float(source_metadata["initial_posterior_scale"]),
        use_progress_mixing=True,
        learnable_budget_gain=True,
    )
    model.import_foundation(source)
    del source
    return model.to(torch.device(device)), source_metadata


def initialize_progress_mixing_ablation_remst_net(
    direct_checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CoordinatedReMSTNet, dict[str, Any]]:
    """Create the progress-mixing arm with the original fixed budget."""

    source, source_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path,
        device="cpu",
    )
    model = CoordinatedReMSTNet(
        progress_bins=int(source_metadata["progress_bins"]),
        posterior_scale=float(source_metadata["initial_posterior_scale"]),
        use_progress_mixing=True,
        learnable_budget_gain=False,
    )
    model.import_foundation(source)
    del source
    return model.to(torch.device(device)), source_metadata


def initialize_adaptive_budget_ablation_remst_net(
    direct_checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CoordinatedReMSTNet, dict[str, Any]]:
    """Create the adaptive-budget arm without local progress mixing."""

    source, source_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path,
        device="cpu",
    )
    model = CoordinatedReMSTNet(
        progress_bins=int(source_metadata["progress_bins"]),
        posterior_scale=float(source_metadata["initial_posterior_scale"]),
        use_progress_mixing=False,
        learnable_budget_gain=True,
    )
    model.import_foundation(source)
    del source
    return model.to(torch.device(device)), source_metadata


def remstnet_parameter_counts(model: ReMSTBlockNet) -> dict[str, int]:
    """Return unique foundation/block parameters for the executable model."""

    _require(isinstance(model, ReMSTBlockNet), "parameter target is not ReMSTNet")

    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    foundation = sum(count(module) for module in model._foundation_modules())
    scale8 = count(model.scale8_block)
    scale16 = count(model.scale16_block)
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    coordinator = (
        count(model.moment_coordinator)
        if isinstance(model, CoordinatedReMSTNet)
        else 0
    )
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    result = {
        "shared_foundation": foundation,
        "scale8_remst_block": scale8,
        "scale16_remst_block": scale16,
        "trainable": trainable,
        "total_unique": total,
        "component_sum": foundation + scale8 + scale16 + coordinator,
    }
    if isinstance(model, CoordinatedReMSTNet):
        result["cross_scale_moment_coordinator"] = coordinator
    return result


__all__ = [
    "ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE",
    "ADAPTIVE_REMST_NET_ARCHITECTURE",
    "COORDINATED_REMST_NET_ARCHITECTURE",
    "CONTEXT_MOMENT_BUDGET",
    "CoordinatedReMSTNet",
    "CrossScaleMomentCoordinator",
    "PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE",
    "REMST_BLOCK_NET_ARCHITECTURE",
    "ReMSTBlock",
    "ReMSTBlockNet",
    "SCALE16_MOMENT_BUDGET",
    "SCALE8_MOMENT_BUDGET",
    "TOTAL_MOMENT_BUDGET",
    "initialize_adaptive_budget_ablation_remst_net",
    "initialize_coordinated_remst_net",
    "initialize_adaptive_remst_net",
    "initialize_progress_mixing_ablation_remst_net",
    "initialize_remst_block_net",
    "remstnet_parameter_counts",
]
