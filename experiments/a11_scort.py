"""A11 SARN-conditioned orthogonal regret transport (SCORT).

SCORT is an end-to-end image model with a strictly Raw-only EfficientNet-B0
posterior anchor and a separate compact SARN encoder.  Homography-aligned
Raw/SARN relations condition a sequence of multiscale Givens rotations on the
square root of the Raw progress posterior.  Every rotation is orthogonal, so
squaring the transported amplitude yields a finite, non-negative unit-mass
posterior without expert routing, convex view weights, or posterior voting.

The correction path consumes detached Raw anchor posterior/features.  Its
losses therefore cannot update the Raw anchor; the anchor remains an exact
nested Raw-only comparator.  Missing SARN, an invalid homography, or empty
support returns the detached Raw posterior, mean, and variance exactly.

This module defines only the model core and numerical transport.  Training
targets, regret objectives, Core splits, and confirmation protocol live in
their own A11 modules.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from experiments.a8_sarn_aligned_local_residual import (
    ReceptiveFieldHomographyAligner,
)
from experiments.support_geometry_multiview_efficientnet import (
    EFFICIENTNET_B0_FEATURES,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
)


A11_ARCHITECTURE: Final[str] = (
    "SCORT-SARN-Conditioned-Homography-Aligned-Orthogonal-Regret-Transport"
)
DEFAULT_PROGRESS_BINS: Final[int] = 128
DEFAULT_RELATION_CHANNELS: Final[int] = 48
DEFAULT_TOKEN_DIM: Final[int] = 64
RAW_STRIDE8_CHANNELS: Final[int] = 40
SARN_STRIDE8_CHANNELS: Final[int] = 48
SARN_STRIDE16_CHANNELS: Final[int] = 96
TRANSPORT_DISTANCES: Final[tuple[int, ...]] = (1, 2, 4, 8)
TRANSPORT_PARITIES: Final[tuple[int, ...]] = (0, 1)
TRANSPORT_STAGE_COUNT: Final[int] = len(TRANSPORT_DISTANCES) * len(
    TRANSPORT_PARITIES
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(
        posterior.ndim == 2 and posterior.is_floating_point(),
        "SCORT posterior must be floating BxK",
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


def deterministic_adaptive_average_pool2d(
    value: torch.Tensor,
    output_size: int | tuple[int, int],
) -> torch.Tensor:
    """Adaptive-average semantics using only deterministic slice reductions.

    Divisible grids use a reshape/block-mean fast path.  General shapes use
    the same floor-start/ceil-end cells as adaptive average pooling, including
    overlapping cells when an output dimension is larger than its input.
    """

    _require(
        value.ndim == 4 and value.is_floating_point(),
        "deterministic average pooling expects floating BCHW",
    )
    if isinstance(output_size, int):
        output_height = output_width = int(output_size)
    else:
        _require(len(output_size) == 2, "pool output size must have two axes")
        output_height, output_width = (int(axis) for axis in output_size)
    _require(
        output_height >= 1 and output_width >= 1,
        "pool output dimensions must be positive",
    )
    batch, channels, height, width = value.shape
    _require(height >= 1 and width >= 1, "pool input spatial dimensions are empty")
    if output_height == output_width == 1:
        return value.mean(dim=(2, 3), keepdim=True)
    if height % output_height == 0 and width % output_width == 0:
        block_height = height // output_height
        block_width = width // output_width
        return value.reshape(
            batch,
            channels,
            output_height,
            block_height,
            output_width,
            block_width,
        ).mean(dim=(3, 5))

    rows: list[torch.Tensor] = []
    for output_y in range(output_height):
        start_y = output_y * height // output_height
        end_y = ((output_y + 1) * height + output_height - 1) // output_height
        columns: list[torch.Tensor] = []
        for output_x in range(output_width):
            start_x = output_x * width // output_width
            end_x = ((output_x + 1) * width + output_width - 1) // output_width
            columns.append(
                value[:, :, start_y:end_y, start_x:end_x].mean(dim=(2, 3))
            )
        rows.append(torch.stack(columns, dim=2))
    return torch.stack(rows, dim=2)


class _DeterministicGlobalAveragePool2d(nn.Module):
    """Drop-in state-free replacement for ``AdaptiveAvgPool2d(1)``."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return deterministic_adaptive_average_pool2d(value, 1)


def _replace_global_adaptive_average_pools(module: nn.Module) -> None:
    """Remove state-free global adaptive pools without changing parameters."""

    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            output_size = child.output_size
            _require(
                output_size == 1 or tuple(output_size) == (1, 1),
                "only global adaptive pools have a deterministic replacement",
            )
            setattr(module, name, _DeterministicGlobalAveragePool2d())
        else:
            _replace_global_adaptive_average_pools(child)


class SCORTRawEfficientNetB0Encoder(nn.Module):
    """Raw-only ImageNet EfficientNet-B0 split at strides 8 and 16."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1
            if imagenet_pretrained
            else None
        )
        backbone = efficientnet_b0(weights=weights)
        # Torchvision EfficientNet squeeze-excitation blocks contain
        # state-free AdaptiveAvgPool2d(1) modules.  Replace them as well so a
        # strict-deterministic CUDA backward never reaches that operator.
        _replace_global_adaptive_average_pools(backbone.features)
        stages = tuple(backbone.features.children())
        _require(len(stages) == 9, "unexpected EfficientNet-B0 feature stages")
        self.to_stride8 = nn.Sequential(*stages[:4])
        self.to_stride16 = nn.Sequential(*stages[4:6])
        self.to_final = nn.Sequential(*stages[6:])
        self.imagenet_pretrained = bool(imagenet_pretrained)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _require(
            images.ndim == 4
            and images.shape[1] == 3
            and images.shape[2] >= 32
            and images.shape[3] >= 32
            and images.is_floating_point(),
            "SCORT Raw images must be floating Bx3xHxW with H,W >= 32",
        )
        _require(bool(torch.isfinite(images).all()), "SCORT Raw image is non-finite")
        stride8 = self.to_stride8(images)
        stride16 = self.to_stride16(stride8)
        final = self.to_final(stride16)
        _require(
            stride8.shape[1] == RAW_STRIDE8_CHANNELS
            and stride16.shape[1] == EFFICIENTNET_B0_MIDDLE_FEATURES
            and final.shape[1] == EFFICIENTNET_B0_FEATURES,
            "SCORT Raw EfficientNet feature channels drifted",
        )
        representation = final.mean(dim=(2, 3))
        return {
            "stride8": stride8,
            "stride16": stride16,
            "final": final,
            "representation": representation,
        }


class SCORTRawPosteriorHead(nn.Module):
    def __init__(self, *, progress_bins: int = DEFAULT_PROGRESS_BINS) -> None:
        super().__init__()
        _require(progress_bins >= 16, "SCORT needs at least 16 progress bins")
        self.progress_bins = int(progress_bins)
        self.projection = nn.Linear(EFFICIENTNET_B0_FEATURES, self.progress_bins)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == EFFICIENTNET_B0_FEATURES,
            "SCORT Raw representation has the wrong shape",
        )
        return self.projection(representation)


class SCORTRawParent(nn.Module):
    """Standalone Raw-only parent with the same anchor topology as A11."""

    def __init__(
        self,
        *,
        imagenet_pretrained: bool = True,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.encoder = SCORTRawEfficientNetB0Encoder(
            imagenet_pretrained=imagenet_pretrained
        )
        self.posterior_head = SCORTRawPosteriorHead(
            progress_bins=self.progress_bins
        )

    def forward(self, original_view: torch.Tensor) -> dict[str, Any]:
        encoded = self.encoder(original_view)
        logits = self.posterior_head(encoded["representation"])
        posterior = torch.softmax(logits.float(), dim=1)
        mean, variance = _posterior_moments(posterior)
        return {
            "architecture": "SCORT-Matched-Raw-EfficientNetB0-Posterior",
            "logits": logits,
            "progress_posterior": posterior,
            "mean": mean,
            "variance": variance,
            "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
            "encoder_features": encoded,
        }

    @classmethod
    def from_scort(cls, model: "A11SCORTImageModel") -> "SCORTRawParent":
        _require(isinstance(model, A11SCORTImageModel), "source is not A11 SCORT")
        parent = cls(
            imagenet_pretrained=False,
            progress_bins=model.progress_bins,
        )
        parent.encoder.load_state_dict(copy.deepcopy(model.raw_encoder.state_dict()))
        parent.posterior_head.load_state_dict(
            copy.deepcopy(model.raw_posterior_head.state_dict())
        )
        parent.train(model.training)
        return parent


def _normalization_groups(channels: int) -> int:
    _require(channels >= 1, "normalization channels must be positive")
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise AssertionError("one group always divides a positive channel count")


class _ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_normalization_groups(output_channels), output_channels),
            nn.SiLU(),
        )


class _CompactInvertedResidual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        _require(stride in (1, 2), "compact SARN stride must be one or two")
        expanded = 2 * input_channels
        self.use_residual = stride == 1 and input_channels == output_channels
        self.block = nn.Sequential(
            _ConvNormActivation(
                input_channels, expanded, kernel_size=1, stride=1
            ),
            _ConvNormActivation(
                expanded,
                expanded,
                kernel_size=3,
                stride=stride,
                groups=expanded,
            ),
            nn.Conv2d(expanded, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_normalization_groups(output_channels), output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.block(value)
        if self.use_residual:
            output = output + value
        return F.silu(output)


class SCORTCompactSARNEncoder(nn.Module):
    """Independent 0.12M-parameter SARN-only encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = _ConvNormActivation(3, 24, stride=2)
        self.stage2 = nn.Sequential(
            _CompactInvertedResidual(24, 24, stride=1),
            _CompactInvertedResidual(24, 32, stride=2),
            _CompactInvertedResidual(32, 32, stride=1),
        )
        self.stage8 = nn.Sequential(
            _CompactInvertedResidual(32, SARN_STRIDE8_CHANNELS, stride=2),
            _CompactInvertedResidual(
                SARN_STRIDE8_CHANNELS, SARN_STRIDE8_CHANNELS, stride=1
            ),
        )
        self.stage16 = nn.Sequential(
            _CompactInvertedResidual(
                SARN_STRIDE8_CHANNELS, SARN_STRIDE16_CHANNELS, stride=2
            ),
            _CompactInvertedResidual(
                SARN_STRIDE16_CHANNELS, SARN_STRIDE16_CHANNELS, stride=1
            ),
            _CompactInvertedResidual(
                SARN_STRIDE16_CHANNELS, SARN_STRIDE16_CHANNELS, stride=1
            ),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _require(
            images.ndim == 4
            and images.shape[1] == 3
            and images.shape[2] >= 32
            and images.shape[3] >= 32
            and images.is_floating_point(),
            "SCORT SARN images must be floating Bx3xHxW with H,W >= 32",
        )
        _require(bool(torch.isfinite(images).all()), "SCORT SARN image is non-finite")
        value = self.stage2(self.stem(images))
        stride8 = self.stage8(value)
        stride16 = self.stage16(stride8)
        _require(
            stride8.shape[1] == SARN_STRIDE8_CHANNELS
            and stride16.shape[1] == SARN_STRIDE16_CHANNELS,
            "compact SARN feature channels drifted",
        )
        return {"stride8": stride8, "stride16": stride16}


class _SCORTDifferentiableAligner(nn.Module):
    def __init__(self, *, feature_stride: int) -> None:
        super().__init__()
        self.geometry = ReceptiveFieldHomographyAligner(
            feature_stride=feature_stride
        )

    def forward(
        self,
        *,
        sarn_features: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
        input_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        geometry = self.geometry.sampling_grid(
            reference=sarn_features,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        support = torch.nan_to_num(
            sarn_support_mask.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0.0, 1.0)
        support = deterministic_adaptive_average_pool2d(
            support, tuple(int(axis) for axis in sarn_features.shape[-2:])
        )
        with torch.autocast(device_type=sarn_features.device.type, enabled=False):
            aligned_sarn = F.grid_sample(
                sarn_features.float(),
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
        support_area = common_support.mean(dim=(1, 2, 3))
        support_valid = (
            torch.isfinite(common_support).all(dim=(1, 2, 3))
            & torch.isfinite(support_area)
            & (support_area > 0.0)
        )
        return {
            **geometry,
            "aligned_sarn": aligned_sarn,
            "aligned_support": aligned_support,
            "common_support": common_support,
            "support_area": support_area,
            "support_valid": support_valid,
        }


class _SCORTRelationScale(nn.Module):
    def __init__(
        self,
        *,
        raw_channels: int,
        sarn_channels: int,
        relation_channels: int,
        token_dim: int,
        memory_grid_size: int,
    ) -> None:
        super().__init__()
        self.raw_channels = int(raw_channels)
        self.sarn_channels = int(sarn_channels)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.raw_projection = nn.Sequential(
            nn.Conv2d(raw_channels, relation_channels, kernel_size=1),
            nn.GroupNorm(8, relation_channels),
            nn.SiLU(),
        )
        self.sarn_projection = nn.Sequential(
            nn.Conv2d(sarn_channels, relation_channels, kernel_size=1),
            nn.GroupNorm(8, relation_channels),
            nn.SiLU(),
        )
        relation_input_channels = 5 * relation_channels + 2
        self.relation_stem = nn.Sequential(
            nn.Conv2d(relation_input_channels, token_dim, kernel_size=1),
            nn.GroupNorm(8, token_dim),
            nn.SiLU(),
            nn.Conv2d(
                token_dim,
                token_dim,
                kernel_size=3,
                padding=1,
                groups=token_dim,
            ),
            nn.Conv2d(token_dim, token_dim, kernel_size=1),
            nn.GroupNorm(8, token_dim),
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
            and raw.shape[0] == aligned_sarn.shape[0]
            and raw.shape[-2:] == aligned_sarn.shape[-2:]
            and raw.shape[1] == self.raw_channels
            and aligned_sarn.shape[1] == self.sarn_channels,
            "SCORT Raw/aligned-SARN relation shapes differ",
        )
        _require(
            common_support.shape == (batch, 1, raw.shape[2], raw.shape[3])
            and available.shape == (batch,)
            and available.dtype == torch.bool,
            "SCORT relation support/availability shapes differ",
        )
        support = common_support.float().clamp(0.0, 1.0)
        support = support * available[:, None, None, None].to(support.dtype)
        raw_projected = self.raw_projection(raw.float())
        sarn_projected = self.sarn_projection(aligned_sarn.float())
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


class SCORTDualScaleRelationEncoder(nn.Module):
    def __init__(
        self,
        *,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        memory_grid_size: int = 4,
    ) -> None:
        super().__init__()
        _require(
            relation_channels >= 8 and relation_channels % 8 == 0,
            "SCORT relation channels must be a multiple of eight",
        )
        _require(
            token_dim >= 8 and token_dim % 8 == 0,
            "SCORT token width must be a multiple of eight",
        )
        _require(memory_grid_size >= 2, "SCORT memory grid is too small")
        self.token_dim = int(token_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.stride8 = _SCORTRelationScale(
            raw_channels=RAW_STRIDE8_CHANNELS,
            sarn_channels=SARN_STRIDE8_CHANNELS,
            relation_channels=relation_channels,
            token_dim=token_dim,
            memory_grid_size=memory_grid_size,
        )
        self.stride16 = _SCORTRelationScale(
            raw_channels=EFFICIENTNET_B0_MIDDLE_FEATURES,
            sarn_channels=SARN_STRIDE16_CHANNELS,
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


def multiscale_givens_edges(
    progress_bins: int,
    *,
    distances: tuple[int, ...] = TRANSPORT_DISTANCES,
) -> tuple[tuple[int, int, tuple[int, ...], tuple[int, ...]], ...]:
    """Return eight disjoint-edge stages ``(distance, parity, left, right)``."""

    _require(progress_bins >= 16, "SCORT transport needs at least 16 bins")
    _require(
        distances and all(distance >= 1 for distance in distances),
        "SCORT transport distances must be positive",
    )
    stages: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for distance in distances:
        for parity in TRANSPORT_PARITIES:
            left: list[int] = []
            right: list[int] = []
            for residue in range(distance):
                sequence = list(range(residue, progress_bins, distance))
                first = parity
                for index in range(first, len(sequence) - 1, 2):
                    left.append(sequence[index])
                    right.append(sequence[index + 1])
            _require(
                2 * len(left) == len(set(left + right)),
                "Givens stage edges are not disjoint",
            )
            stages.append((distance, parity, tuple(left), tuple(right)))
    return tuple(stages)


def apply_givens_rotation_stage(
    amplitudes: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    angles: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply one parallel set of disjoint Givens rotations."""

    _require(
        amplitudes.ndim == 2
        and amplitudes.is_floating_point()
        and left_indices.ndim == right_indices.ndim == 1
        and left_indices.shape == right_indices.shape
        and angles.shape == (amplitudes.shape[0], left_indices.numel()),
        "SCORT Givens stage shapes differ",
    )
    _require(
        left_indices.dtype == right_indices.dtype == torch.long
        and left_indices.device == right_indices.device == amplitudes.device
        and angles.device == amplitudes.device,
        "SCORT Givens indices/angles must share the amplitude device",
    )
    theta = -angles.float() if inverse else angles.float()
    value = amplitudes.float()
    left = value.index_select(1, left_indices)
    right = value.index_select(1, right_indices)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    rotated_left = cosine * left - sine * right
    rotated_right = sine * left + cosine * right
    result = value.scatter(
        1, left_indices[None].expand(value.shape[0], -1), rotated_left
    )
    result = result.scatter(
        1, right_indices[None].expand(value.shape[0], -1), rotated_right
    )
    return result


class SCORTOrthogonalTransport(nn.Module):
    """Predict and apply sample-specific multiscale Givens rotations."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        angle_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "SCORT transport needs at least 16 bins")
        _require(token_dim >= 8, "SCORT transport token width is too small")
        hidden = int(angle_hidden_dim or token_dim)
        _require(hidden >= 8, "SCORT angle hidden width is too small")
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        stages = multiscale_givens_edges(self.progress_bins)
        edge_left: list[int] = []
        edge_right: list[int] = []
        distance_ids: list[int] = []
        parity_ids: list[int] = []
        maximum_angles: list[float] = []
        stage_offsets = [0]
        for distance_index, (distance, parity, left, right) in enumerate(stages):
            roster_index = distance_index // len(TRANSPORT_PARITIES)
            edge_left.extend(left)
            edge_right.extend(right)
            distance_ids.extend([roster_index] * len(left))
            parity_ids.extend([parity] * len(left))
            maximum_angles.extend(
                [math.pi / (16.0 * math.sqrt(float(distance)))] * len(left)
            )
            stage_offsets.append(len(edge_left))
        self.stage_specs = tuple((stage[0], stage[1]) for stage in stages)
        self.stage_offsets = tuple(stage_offsets)
        self.register_buffer("edge_left", torch.tensor(edge_left, dtype=torch.long))
        self.register_buffer("edge_right", torch.tensor(edge_right, dtype=torch.long))
        self.register_buffer(
            "edge_distance_id", torch.tensor(distance_ids, dtype=torch.long)
        )
        self.register_buffer(
            "edge_parity_id", torch.tensor(parity_ids, dtype=torch.long)
        )
        self.register_buffer(
            "maximum_angle", torch.tensor(maximum_angles, dtype=torch.float32)
        )
        self.distance_embedding = nn.Embedding(len(TRANSPORT_DISTANCES), 8)
        self.parity_embedding = nn.Embedding(len(TRANSPORT_PARITIES), 4)
        angle_input_dim = 4 * token_dim + 12
        self.angle_trunk = nn.Sequential(
            nn.LayerNorm(angle_input_dim),
            nn.Linear(angle_input_dim, hidden),
            nn.SiLU(),
        )
        self.angle_output = nn.Linear(hidden, 1)
        # Exact active q0 at construction; the output head has a non-zero
        # first-step gradient and unlocks all upstream correction modules on
        # the following step.  There is no global structure-strength scalar.
        nn.init.zeros_(self.angle_output.weight)
        nn.init.zeros_(self.angle_output.bias)

    @property
    def edge_count(self) -> int:
        return int(self.edge_left.numel())

    def _angles(
        self,
        progress_tokens: torch.Tensor,
        correction_available: torch.Tensor,
    ) -> torch.Tensor:
        batch, bins, token_dim = progress_tokens.shape
        _require(
            bins == self.progress_bins
            and token_dim == self.token_dim
            and correction_available.shape == (batch,)
            and correction_available.dtype == torch.bool,
            "SCORT progress-token/availability shapes differ",
        )
        with torch.autocast(
            device_type=progress_tokens.device.type, enabled=False
        ):
            tokens = progress_tokens.float()
            left = tokens.index_select(1, self.edge_left)
            right = tokens.index_select(1, self.edge_right)
            distance = self.distance_embedding(self.edge_distance_id).float()
            parity = self.parity_embedding(self.edge_parity_id).float()
            metadata = torch.cat((distance, parity), dim=1)[None].expand(
                batch, -1, -1
            )
            features = torch.cat(
                (left, right, left - right, left * right, metadata), dim=2
            )
            logits = self.angle_output(self.angle_trunk(features)).squeeze(2)
            angles = torch.tanh(logits) * self.maximum_angle[None]
            angles = torch.where(
                correction_available[:, None], angles, torch.zeros_like(angles)
            )
        return angles

    def forward(
        self,
        raw_posterior: torch.Tensor,
        progress_tokens: torch.Tensor,
        *,
        correction_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            raw_posterior.shape
            == (progress_tokens.shape[0], self.progress_bins)
            and raw_posterior.is_floating_point(),
            "SCORT Raw posterior shape differs from transport tokens",
        )
        raw_input = raw_posterior.detach().float()
        _require(
            bool(torch.isfinite(raw_input).all())
            and bool((raw_input >= 0.0).all())
            and bool((raw_input.sum(dim=1) > 0.0).all()),
            "SCORT Raw posterior is not a finite probability distribution",
        )
        raw = raw_input / raw_input.sum(dim=1, keepdim=True)
        angles = self._angles(progress_tokens, correction_available)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            amplitudes = torch.sqrt(raw)
            initial_amplitudes = amplitudes
            layer_posteriors: list[torch.Tensor] = []
            layer_angle_residuals: list[torch.Tensor] = []
            for stage_index in range(len(self.stage_offsets) - 1):
                start = self.stage_offsets[stage_index]
                stop = self.stage_offsets[stage_index + 1]
                stage_angles = angles[:, start:stop]
                amplitudes = apply_givens_rotation_stage(
                    amplitudes,
                    self.edge_left[start:stop],
                    self.edge_right[start:stop],
                    stage_angles,
                )
                stage_posterior = amplitudes.square()
                stage_posterior = stage_posterior / stage_posterior.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0e-12)
                # sqrt -> square -> normalization is mathematically the
                # identity at zero cumulative angle, but not necessarily
                # bit-exact in FP32.  Anchor that exact forward value while
                # retaining the transported posterior's straight-through
                # derivative with respect to every angle.  This selection is
                # used only when all angles through the current stage are
                # exactly zero; any non-zero transport follows the ordinary
                # numerical path.
                cumulative_zero = (angles[:, :stop] == 0.0).all(dim=1)
                active_zero = correction_available & cumulative_zero
                straight_through_identity = (
                    stage_posterior - stage_posterior.detach() + raw_input
                )
                stage_posterior = torch.where(
                    active_zero[:, None],
                    straight_through_identity,
                    stage_posterior,
                )
                # Explicit selection is what gives missing/invalid rows exact
                # q0 values despite sqrt/square floating-point roundoff.
                stage_posterior = torch.where(
                    correction_available[:, None], stage_posterior, raw_input
                )
                layer_posteriors.append(stage_posterior)
                normalized_angle = stage_angles / self.maximum_angle[
                    start:stop
                ][None].clamp_min(1.0e-12)
                residual = (
                    normalized_angle.square().mean(dim=1)
                    if stop > start
                    else torch.zeros(
                        raw.shape[0], dtype=torch.float32, device=raw.device
                    )
                )
                residual = torch.where(
                    correction_available, residual, torch.zeros_like(residual)
                )
                layer_angle_residuals.append(residual)
            layer_stack = torch.stack(layer_posteriors, dim=1)
            residual_stack = torch.stack(layer_angle_residuals, dim=1)
            final = layer_stack[:, -1]
        return {
            "progress_posterior": final,
            "layer_posteriors": layer_stack,
            "transport_angles": angles,
            "layer_angle_residuals": residual_stack,
            "initial_amplitudes": initial_amplitudes,
            "final_amplitudes": amplitudes,
        }

    def inverse_amplitudes(
        self,
        final_amplitudes: torch.Tensor,
        transport_angles: torch.Tensor,
    ) -> torch.Tensor:
        _require(
            final_amplitudes.shape[1] == self.progress_bins
            and transport_angles.shape
            == (final_amplitudes.shape[0], self.edge_count),
            "SCORT inverse-transport shapes differ",
        )
        with torch.autocast(
            device_type=final_amplitudes.device.type, enabled=False
        ):
            value = final_amplitudes.float()
            for stage_index in reversed(range(len(self.stage_offsets) - 1)):
                start = self.stage_offsets[stage_index]
                stop = self.stage_offsets[stage_index + 1]
                value = apply_givens_rotation_stage(
                    value,
                    self.edge_left[start:stop],
                    self.edge_right[start:stop],
                    transport_angles[:, start:stop],
                    inverse=True,
                )
        return value


class SCORTProgressDecoder(nn.Module):
    """Condition 128 progress tokens on the dual-scale relation memory."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "SCORT decoder needs at least 16 bins")
        _require(
            token_dim >= 8
            and attention_heads >= 1
            and token_dim % attention_heads == 0,
            "SCORT token width/attention heads are incompatible",
        )
        _require(decoder_layers >= 1, "SCORT needs at least one decoder layer")
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.progress_position_embedding = nn.Parameter(
            torch.zeros(1, progress_bins, token_dim)
        )
        nn.init.trunc_normal_(self.progress_position_embedding, std=0.02)
        self.posterior_embedding = nn.Sequential(
            nn.Linear(2, token_dim),
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
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=decoder_layers
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        raw_posterior: torch.Tensor,
        relation_memory: torch.Tensor,
    ) -> torch.Tensor:
        _require(
            raw_posterior.ndim == 2
            and raw_posterior.shape[1] == self.progress_bins
            and relation_memory.ndim == 3
            and relation_memory.shape[0] == raw_posterior.shape[0]
            and relation_memory.shape[2] == self.token_dim,
            "SCORT posterior/relation memory shapes differ",
        )
        probability = raw_posterior.detach().float().clamp_min(1.0e-8)
        posterior_features = torch.stack(
            (probability, torch.log(probability)), dim=2
        )
        query = self.posterior_embedding(posterior_features)
        query = query + self.progress_position_embedding.to(query.dtype)
        decoded = self.decoder(tgt=query, memory=relation_memory)
        return self.output_norm(decoded)


class A11SCORTImageModel(nn.Module):
    """ImageNet Raw anchor plus independent SARN-conditioned SCORT core."""

    def __init__(
        self,
        *,
        imagenet_pretrained: bool = True,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = 4,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.memory_grid_size = int(memory_grid_size)
        self.raw_encoder = SCORTRawEfficientNetB0Encoder(
            imagenet_pretrained=imagenet_pretrained
        )
        self.raw_posterior_head = SCORTRawPosteriorHead(
            progress_bins=self.progress_bins
        )
        self.sarn_encoder = SCORTCompactSARNEncoder()
        self.stride8_aligner = _SCORTDifferentiableAligner(feature_stride=8)
        self.stride16_aligner = _SCORTDifferentiableAligner(feature_stride=16)
        self.relation_encoder = SCORTDualScaleRelationEncoder(
            relation_channels=self.relation_channels,
            token_dim=self.token_dim,
            memory_grid_size=self.memory_grid_size,
        )
        self.progress_decoder = SCORTProgressDecoder(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
            attention_heads=self.attention_heads,
            decoder_layers=self.decoder_layers,
        )
        self.orthogonal_transport = SCORTOrthogonalTransport(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
        )

    @staticmethod
    def _validate_inputs(
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> int:
        _require(
            original_view.shape == sarn_view.shape
            and original_view.ndim == 4
            and original_view.shape[1] == 3,
            "SCORT Raw/SARN image shapes differ",
        )
        batch, _channels, height, width = original_view.shape
        _require(height >= 32 and width >= 32, "SCORT images must be at least 32x32")
        _require(
            sarn_support_mask.shape == (batch, 1, height, width)
            and sarn_active.shape == (batch,)
            and sarn_active.dtype == torch.bool
            and raw_to_sarn_homography.shape == (batch, 3, 3),
            "SCORT support/activity/homography shapes differ",
        )
        tensors = (
            original_view,
            sarn_view,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        _require(
            all(value.device == original_view.device for value in tensors),
            "SCORT inputs must share one device",
        )
        _require(
            original_view.is_floating_point()
            and sarn_view.is_floating_point()
            and sarn_support_mask.is_floating_point()
            and raw_to_sarn_homography.is_floating_point(),
            "SCORT image/support/homography inputs must be floating",
        )
        _require(
            bool(torch.isfinite(original_view).all())
            and bool(torch.isfinite(sarn_view).all()),
            "SCORT image input is non-finite",
        )
        return batch

    def forward(
        self,
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        batch = self._validate_inputs(
            original_view,
            sarn_view,
            sarn_support_mask,
            sarn_active,
            raw_to_sarn_homography,
        )
        raw_features = self.raw_encoder(original_view)
        raw_logits = self.raw_posterior_head(raw_features["representation"])
        raw_posterior = torch.softmax(raw_logits.float(), dim=1)
        raw_mean, raw_variance = _posterior_moments(raw_posterior)

        sarn_features = self.sarn_encoder(sarn_view)
        input_hw = tuple(int(value) for value in original_view.shape[-2:])
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
            raw_stride8=raw_features["stride8"].detach(),
            aligned_sarn_stride8=alignment8["aligned_sarn"],
            support_stride8=alignment8["common_support"],
            raw_stride16=raw_features["stride16"].detach(),
            aligned_sarn_stride16=alignment16["aligned_sarn"],
            support_stride16=alignment16["common_support"],
            available=relation_available,
        )
        progress_tokens = self.progress_decoder(
            raw_posterior.detach(), relation["memory"]
        )
        transport = self.orthogonal_transport(
            raw_posterior.detach(),
            progress_tokens,
            correction_available=relation_available,
        )
        posterior = transport["progress_posterior"]
        mean, variance = _posterior_moments(posterior)
        layer_shape = transport["layer_posteriors"].shape
        layer_means, layer_variances = _posterior_moments(
            transport["layer_posteriors"].reshape(-1, self.progress_bins)
        )
        layer_means = layer_means.reshape(layer_shape[:2])
        layer_variances = layer_variances.reshape(layer_shape[:2])
        correction_active = relation_available[:, None].expand(
            -1, TRANSPORT_STAGE_COUNT
        )
        return {
            "architecture": A11_ARCHITECTURE,
            "progress_posterior": posterior,
            "mean": mean,
            "variance": variance,
            "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
            "raw_anchor_logits": raw_logits,
            "raw_anchor_posterior": raw_posterior,
            "raw_anchor_mean": raw_mean,
            "raw_anchor_variance": raw_variance,
            "raw_anchor_standard_deviation": torch.sqrt(
                raw_variance.clamp_min(0.0)
            ),
            # Compatibility aliases for matched-system reporting.
            "raw_posterior": raw_posterior,
            "raw_mean": raw_mean,
            "raw_variance": raw_variance,
            "layer_posteriors": transport["layer_posteriors"],
            "layer_means": layer_means,
            "layer_variances": layer_variances,
            "transport_angles": transport["transport_angles"],
            "layer_angle_residuals": transport["layer_angle_residuals"],
            "initial_amplitudes": transport["initial_amplitudes"],
            "final_amplitudes": transport["final_amplitudes"],
            "correction_active": correction_active,
            "relation_available": relation_available,
            "sarn_active": sarn_active,
            "raw_encoder_features": raw_features,
            "sarn_encoder_features": sarn_features,
            "projective_relation": relation,
            "progress_tokens": progress_tokens,
            "stride8_alignment": alignment8,
            "stride16_alignment": alignment16,
            "physical_outputs": {
                "progress_mean": mean,
                "progress_variance": variance,
            },
        }


def image_model_parameter_counts(model: A11SCORTImageModel) -> dict[str, int]:
    _require(isinstance(model, A11SCORTImageModel), "target is not A11 SCORT")

    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    components = {
        "raw_encoder": count(model.raw_encoder),
        "raw_posterior_head": count(model.raw_posterior_head),
        "sarn_encoder": count(model.sarn_encoder),
        "relation_encoder": count(model.relation_encoder),
        "progress_decoder": count(model.progress_decoder),
        "orthogonal_transport": count(model.orthogonal_transport),
    }
    total = count(model)
    return {
        **components,
        "raw_anchor": components["raw_encoder"]
        + components["raw_posterior_head"],
        "correction": total
        - components["raw_encoder"]
        - components["raw_posterior_head"],
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
        "transport_edges": model.orthogonal_transport.edge_count,
        "transport_stages": TRANSPORT_STAGE_COUNT,
    }


def raw_parent_parameter_counts(model: SCORTRawParent) -> dict[str, int]:
    _require(isinstance(model, SCORTRawParent), "target is not SCORT Raw parent")
    encoder = int(sum(parameter.numel() for parameter in model.encoder.parameters()))
    head = int(
        sum(parameter.numel() for parameter in model.posterior_head.parameters())
    )
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "encoder": encoder,
        "posterior_head": head,
        "total": total,
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }


__all__ = [
    "A11_ARCHITECTURE",
    "DEFAULT_PROGRESS_BINS",
    "DEFAULT_RELATION_CHANNELS",
    "DEFAULT_TOKEN_DIM",
    "TRANSPORT_DISTANCES",
    "TRANSPORT_STAGE_COUNT",
    "A11SCORTImageModel",
    "SCORTCompactSARNEncoder",
    "SCORTDualScaleRelationEncoder",
    "SCORTOrthogonalTransport",
    "SCORTProgressDecoder",
    "SCORTRawEfficientNetB0Encoder",
    "SCORTRawParent",
    "SCORTRawPosteriorHead",
    "apply_givens_rotation_stage",
    "deterministic_adaptive_average_pool2d",
    "image_model_parameter_counts",
    "multiscale_givens_edges",
    "raw_parent_parameter_counts",
]
