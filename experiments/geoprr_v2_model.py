"""Lightweight image-only meter reader with supervised spatial composition."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


BACKBONES = (
    "convnextv2_femto.fcmae_ft_in1k",
    "mobilenetv4_conv_medium.e500_r224_in1k",
)
VARIANTS = ("field", "multitask", "raw_distribution", "raw_scalar")


def two_hot_targets(values: torch.Tensor, bins: int) -> torch.Tensor:
    """Add a final bin dimension while preserving targets' continuous means."""
    if bins < 2:
        raise ValueError("at least two ordered bins are required")
    positions = values.float().clamp(0.0, 1.0) * (bins - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=bins - 1)
    fraction = positions - lower
    target = positions.new_zeros((*positions.shape, bins))
    target.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - fraction).unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), fraction.unsqueeze(-1))
    return target


def distribution_cdf_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Ordered-bin Wasserstein distance to continuous two-hot target labels."""
    target_probs = two_hot_targets(targets, probabilities.shape[-1])
    distances = (
        probabilities.float().cumsum(-1)[..., :-1]
        - target_probs.cumsum(-1)[..., :-1]
    ).abs().mean(-1)
    if reduction == "none":
        return distances
    if reduction == "sum":
        return distances.sum()
    if reduction == "mean":
        return distances.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def masked_field_readout(
    field_probs: torch.Tensor,
    point_probs: torch.Tensor,
    support_probs: torch.Tensor,
    support_threshold: float,
    appearance_probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compose U/Q/M, normalizing only rows with sufficient predicted support.

    Invalid field distributions and means are zero placeholders. Their validity
    mask must accompany them. Supplying appearance_probs defines the complete
    readout, which includes every sample through the appearance fallback.
    """
    if not 0.0 < support_threshold < 1.0:
        raise ValueError("support_threshold must be strictly between zero and one")
    if field_probs.ndim != 4:
        raise ValueError("field_probs must have shape B,K,H,W")
    expected = (field_probs.shape[0], 1, *field_probs.shape[-2:])
    if point_probs.shape != expected or support_probs.shape != expected:
        raise ValueError("point/support probabilities must share the field grid")
    weights = point_probs.float() * support_probs.float()
    support_mass = weights.sum((1, 2, 3))
    field_valid = support_mass >= support_threshold
    numerator = (field_probs.float() * weights).sum((2, 3))
    field_distribution = torch.zeros_like(numerator)
    field_distribution[field_valid] = (
        numerator[field_valid] / support_mass[field_valid, None]
    )
    coordinates = torch.linspace(
        0.0, 1.0, field_probs.shape[1], device=field_probs.device
    )
    field_reading = (field_distribution * coordinates).sum(1)
    distribution = field_distribution
    if appearance_probs is not None:
        if appearance_probs.shape != numerator.shape:
            raise ValueError("appearance probabilities must have shape B,K")
        distribution = torch.where(
            field_valid[:, None], field_distribution, appearance_probs.float()
        )
    return {
        "support_mass": support_mass,
        "field_valid": field_valid,
        "support_deficit": (support_threshold - support_mass).clamp_min(0.0),
        "field_distribution": field_distribution,
        "field_reading": field_reading,
        "distribution": distribution,
        "reading": (distribution * coordinates).sum(1),
    }


def apply_modal_field_readout(
    output: dict[str, torch.Tensor | None], radius: int
) -> dict[str, torch.Tensor | None]:
    """Read the local field mode while retaining unsupported rows' fallback."""
    field_distribution = output["field_distribution"]
    bins = field_distribution.shape[1]
    if not 0 < radius < bins:
        raise ValueError("radius must be positive and smaller than the bin count")
    valid = output["field_valid"]
    probabilities = field_distribution[valid]
    peaks = probabilities.argmax(1)
    indices = torch.arange(bins, device=probabilities.device)
    local = probabilities * ((indices[None, :] - peaks[:, None]).abs() <= radius)
    local = local / local.sum(1, keepdim=True)
    coordinates = torch.linspace(
        0.0, 1.0, bins, device=probabilities.device, dtype=probabilities.dtype
    )
    distribution = output["distribution"].clone()
    reading = output["reading"].clone()
    distribution[valid] = local
    reading[valid] = (local * coordinates).sum(1)
    return {**output, "distribution": distribution, "reading": reading}


class _Refine(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(8, channels), channels),
            nn.GELU(),
        )


class _DynamicRefine(nn.Module):
    """Aggregate four convolution kernels from the low-resolution features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernels = nn.Parameter(torch.empty(4, channels, channels, 3, 3))
        for kernel in self.kernels:
            nn.init.kaiming_uniform_(kernel, a=math.sqrt(5))
        self.router = nn.Linear(channels, 4)
        self.norm = nn.GroupNorm(math.gcd(8, channels), channels)

    def kernel_weights(self, features: torch.Tensor) -> torch.Tensor:
        return self.router(features.mean((2, 3))).softmax(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        weights = self.kernel_weights(features) @ self.kernels.flatten(1)
        kernels = weights.reshape(batch * channels, channels, 3, 3)
        decoded = F.conv2d(
            features.reshape(1, batch * channels, height, width),
            kernels,
            padding=1,
            groups=batch,
        ).reshape(batch, channels, height, width)
        return F.gelu(self.norm(decoded))


class _SpatialAppearanceHead(nn.Module):
    """Read multi-scale image features without using the geometry predictions."""

    def __init__(
        self, channels: tuple[int, ...], hidden: int, outputs: int
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel + 2, hidden, 1, bias=False),
                nn.GroupNorm(math.gcd(8, hidden), hidden),
                nn.GELU(),
            )
            for channel in channels
        )
        self.attention = nn.ModuleList(nn.Conv2d(hidden, 1, 1) for _ in channels)
        self.norm = nn.LayerNorm(len(channels) * hidden + channels[-1])
        self.output = nn.Linear(len(channels) * hidden + channels[-1], outputs)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, features: list[torch.Tensor], global_features: torch.Tensor
    ) -> torch.Tensor:
        pooled = []
        for feature, projection, attention in zip(
            features, self.projections, self.attention
        ):
            height, width = feature.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype),
                torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype),
                indexing="ij",
            )
            coordinates = torch.stack((xx, yy))[None].expand(feature.shape[0], -1, -1, -1)
            projected = projection(torch.cat((feature, coordinates), dim=1))
            spatial_weights = attention(projected).flatten(2).softmax(-1)
            pooled.append((projected.flatten(2) * spatial_weights).sum(-1))
        return self.output(self.norm(torch.cat((*pooled, global_features), dim=1)))


class GeoPRRV2(nn.Module):
    """A shared CNN and compact stride-four calibration/state decoder.

    Inference accepts normalized ROI pixels only. All geometry targets and
    cross-image coordinate maps belong to the training data and loss code.
    """

    def __init__(
        self,
        backbone: str = BACKBONES[0],
        *,
        pretrained: bool = True,
        bins: int = 256,
        decoder_dim: int = 64,
        support_threshold: float = 0.15,
        variant: str = "field",
        appearance_pooling: str = "global",
        detail_refinement: bool = False,
        decoder_kernel: str = "static",
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unsupported backbone: {backbone}")
        if variant not in VARIANTS:
            raise ValueError(f"unknown model variant: {variant}")
        if appearance_pooling not in ("global", "spatial"):
            raise ValueError(f"unknown appearance pooling: {appearance_pooling}")
        if decoder_kernel not in ("static", "dynamic"):
            raise ValueError(f"unknown decoder kernel: {decoder_kernel}")
        if bins < 2 or decoder_dim < 1:
            raise ValueError("bins must be at least two and decoder_dim positive")
        if not 0.0 < support_threshold < 1.0:
            raise ValueError("support_threshold must be strictly between zero and one")
        import timm

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True
        )
        reductions = self.backbone.feature_info.reduction()
        channels = self.backbone.feature_info.channels()
        self.feature_indices = tuple(reductions.index(s) for s in (4, 8, 16, 32))
        self.feature_channels = tuple(channels[i] for i in self.feature_indices)
        self.backbone_name = backbone
        self.bins = int(bins)
        self.variant = variant
        self.appearance_pooling = appearance_pooling
        self.detail_refinement = bool(detail_refinement)
        self.decoder_kernel = decoder_kernel
        self.support_threshold = float(support_threshold)
        self.output_stride = 4
        self._backbone_trainable = True
        self.register_buffer("bin_coordinates", torch.linspace(0, 1, bins))
        self.appearance_norm = nn.LayerNorm(self.feature_channels[-1])
        appearance_outputs = 1 if variant == "raw_scalar" else bins
        if appearance_pooling == "spatial":
            self.appearance_head = _SpatialAppearanceHead(
                self.feature_channels, decoder_dim, appearance_outputs
            )
        else:
            self.appearance_head = nn.Linear(
                self.feature_channels[-1], appearance_outputs
            )
        if variant in ("field", "multitask"):
            self.lateral = nn.ModuleList(
                nn.Conv2d(c, decoder_dim, 1) for c in self.feature_channels
            )
            self.refine = nn.ModuleList(
                _DynamicRefine(decoder_dim)
                if decoder_kernel == "dynamic" and level >= 2
                else _Refine(decoder_dim)
                for level in range(4)
            )
            self.field_head = nn.Conv2d(decoder_dim, bins, 1)
            if self.detail_refinement:
                self.coarse_point_head = nn.Conv2d(decoder_dim, 1, 1)
                self.point_head = nn.Sequential(
                    nn.Conv2d(2 * decoder_dim + 1, decoder_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(math.gcd(8, decoder_dim), decoder_dim),
                    nn.GELU(),
                    nn.Conv2d(decoder_dim, 1, 1),
                )
            else:
                self.point_head = nn.Conv2d(decoder_dim, 1, 1)
            self.support_head = nn.Conv2d(decoder_dim, 1, 1)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze parameters and normalization statistics for a warm-up stage."""
        self._backbone_trainable = bool(trainable)
        self.backbone.requires_grad_(trainable)
        self.backbone.train(self.training and trainable)

    def train(self, mode: bool = True) -> GeoPRRV2:
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape B,3,H,W")
        if images.shape[-2] % 32 or images.shape[-1] % 32:
            raise ValueError("ROI height and width must be multiples of 32")
        extracted = self.backbone(images)
        features = [extracted[i] for i in self.feature_indices]
        pooled = self.appearance_norm(features[-1].mean((2, 3)))
        if self.appearance_pooling == "spatial":
            appearance_logits = self.appearance_head(features, pooled).float()
        else:
            appearance_logits = self.appearance_head(pooled).float()
        outputs: dict[str, torch.Tensor | None] = {
            "appearance_logits": appearance_logits,
            "global_features": pooled,
            "coarse_point_logits": None,
            "field_logits": None,
            "point_logits": None,
            "support_logits": None,
            "field_probs": None,
            "point_probs": None,
            "support_probs": None,
            "field_distribution": None,
            "field_reading": None,
            "support_mass": images.new_zeros(images.shape[0], dtype=torch.float32),
            "field_valid": torch.zeros(
                images.shape[0], device=images.device, dtype=torch.bool
            ),
        }
        if self.variant == "raw_scalar":
            scalar_logits = appearance_logits[:, 0]
            scalar_reading = scalar_logits.sigmoid()
            outputs.update(
                scalar_logits=scalar_logits,
                scalar_reading=scalar_reading,
                appearance_reading=scalar_reading,
                appearance_probs=None,
                distribution=None,
                reading=scalar_reading,
            )
            return outputs
        appearance_probs = appearance_logits.softmax(1)
        appearance_reading = (appearance_probs * self.bin_coordinates).sum(1)
        outputs.update(
            appearance_probs=appearance_probs,
            appearance_reading=appearance_reading,
            distribution=appearance_probs,
            reading=appearance_reading,
        )
        if self.variant == "raw_distribution":
            return outputs
        projected = [lateral(feature) for lateral, feature in zip(self.lateral, features)]
        decoded = self.refine[-1](projected[-1])
        coarse_point_logits = None
        for level in range(2, -1, -1):
            lateral = projected[level]
            decoded = F.interpolate(
                decoded, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = self.refine[level](decoded + lateral)
            if self.detail_refinement and level == 2:
                coarse_point_logits = self.coarse_point_head(decoded)
        field_logits = self.field_head(decoded)
        if self.detail_refinement:
            coarse = F.interpolate(
                coarse_point_logits,
                size=decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            mid = F.interpolate(
                projected[1], size=decoded.shape[-2:], mode="bilinear", align_corners=False
            )
            point_logits = coarse + self.point_head(torch.cat((projected[0], mid, coarse), dim=1))
        else:
            point_logits = self.point_head(decoded)
        support_logits = self.support_head(decoded)
        field_probs = field_logits.float().softmax(1)
        point_probs = point_logits.float().flatten(1).softmax(1).view_as(point_logits)
        support_probs = support_logits.float().sigmoid()
        outputs.update(
            field_logits=field_logits,
            point_logits=point_logits,
            coarse_point_logits=coarse_point_logits,
            support_logits=support_logits,
            field_probs=field_probs,
            point_probs=point_probs,
            support_probs=support_probs,
        )
        outputs.update(
            masked_field_readout(
                field_probs,
                point_probs,
                support_probs,
                self.support_threshold,
                appearance_probs,
            )
        )
        if self.variant == "multitask":
            outputs.update(distribution=appearance_probs, reading=appearance_reading)
        return outputs
