"""Single-backbone ResNet-18 anchor for ReMST correction experiments.

Raw and SARN are two observations evaluated by one shared ResNet-18 parameter
set.  They are intentionally forwarded separately so the Raw operator order
and batch shape remain identical to the established Direct-ResNet18 reader.
The ReMST correction consumes stride-8/16 features from that same backbone;
there is no second image encoder or model-level output fusion.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
from torchvision.models import resnet18

from experiments.a15_2_mett import (
    DEFAULT_POSTERIOR_SCALE,
    MomentExactPosteriorHead,
    ReMSTCorrection,
)
from experiments.a15_fteb import DEFAULT_PROGRESS_BINS
from experiments.probabilistic_pivot_direction import IMAGENET_WEIGHTS
from experiments.resnet18_direct_progress import (
    DEFAULT_EPOCHS as DIRECT_RESNET18_EPOCHS,
    IMAGENET_INITIALIZATION,
    PROTOCOL as DIRECT_RESNET18_PROTOCOL,
    SCENE_SPLIT_PROTOCOL,
)


RESNET18_STRIDE8_CHANNELS: Final[int] = 128
RESNET18_STRIDE16_CHANNELS: Final[int] = 256
RESNET18_REPRESENTATION_FEATURES: Final[int] = 512
DIRECT_RESNET18_ARCHITECTURE: Final[str] = (
    "torchvision_resnet18_imagenet1k_v1_sigmoid_scalar"
)
REMST_RESNET18_ARCHITECTURE: Final[str] = (
    "ReMST-ResNet18-Single-Backbone-Dual-Observation-"
    "Relation-Encoded-Moment-Exact-Scalar-Transport"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ReMSTResNet18Encoder(nn.Module):
    """ResNet-18 split at strides 8 and 16 with one 512-D representation."""

    def __init__(self, *, imagenet_pretrained: bool = False) -> None:
        super().__init__()
        backbone = resnet18(
            weights=IMAGENET_WEIGHTS if imagenet_pretrained else None
        )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.imagenet_pretrained = bool(imagenet_pretrained)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _require(
            images.ndim == 4
            and images.shape[1] == 3
            and images.shape[2] >= 32
            and images.shape[3] >= 32
            and images.is_floating_point(),
            "ReMST-ResNet18 images must be floating Bx3xHxW with H,W >= 32",
        )
        _require(
            bool(torch.isfinite(images).all()),
            "ReMST-ResNet18 image is non-finite",
        )
        value = self.relu(self.bn1(self.conv1(images)))
        value = self.maxpool(value)
        value = self.layer1(value)
        stride8 = self.layer2(value)
        stride16 = self.layer3(stride8)
        final = self.layer4(stride16)
        representation = torch.flatten(self.avgpool(final), 1)
        _require(
            stride8.shape[1] == RESNET18_STRIDE8_CHANNELS
            and stride16.shape[1] == RESNET18_STRIDE16_CHANNELS
            and representation.shape
            == (images.shape[0], RESNET18_REPRESENTATION_FEATURES),
            "ReMST-ResNet18 feature channels drifted",
        )
        return {
            "stride8": stride8,
            "stride16": stride16,
            "final": final,
            "representation": representation,
        }


class MomentExactResNet18Anchor(nn.Module):
    """Direct-ResNet18 scalar reader with a moment-exact posterior interface."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        initial_scale: float = DEFAULT_POSTERIOR_SCALE,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.raw_encoder = ReMSTResNet18Encoder(imagenet_pretrained=False)
        self.raw_posterior_head = MomentExactPosteriorHead(
            feature_dim=RESNET18_REPRESENTATION_FEATURES,
            progress_bins=self.progress_bins,
            initial_scale=initial_scale,
        )

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        features = self.raw_encoder(image)
        lifted = self.raw_posterior_head.posterior_parameters(
            features["representation"]
        )
        return {
            "architecture": "Moment-Exact-Direct-ResNet18-Anchor",
            "progress_posterior": lifted["posterior"],
            "mean": lifted["posterior_mean"],
            "variance": lifted["posterior_variance"],
            "standard_deviation": torch.sqrt(
                lifted["posterior_variance"].clamp_min(0.0)
            ),
            "point_progress": lifted["point_progress"],
            "posterior_scale": lifted["posterior_scale"],
            "absolute_moment_error": lifted["absolute_moment_error"],
            "raw_encoder_features": features,
        }


def load_moment_exact_resnet18_anchor(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
    progress_bins: int = DEFAULT_PROGRESS_BINS,
    initial_scale: float = DEFAULT_POSTERIOR_SCALE,
) -> tuple[MomentExactResNet18Anchor, dict[str, Any]]:
    """Import one terminal scene-disjoint Direct-ResNet18 checkpoint."""

    source = Path(checkpoint_path).resolve()
    _require(
        source.is_file(),
        f"Direct-ResNet18 checkpoint does not exist: {source}",
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(checkpoint, Mapping),
        "Direct-ResNet18 checkpoint is malformed",
    )
    _require(
        checkpoint.get("protocol") == DIRECT_RESNET18_PROTOCOL
        and checkpoint.get("architecture") == DIRECT_RESNET18_ARCHITECTURE
        and checkpoint.get("pretrained_weights") == IMAGENET_INITIALIZATION
        and int(checkpoint.get("epochs", -1)) == DIRECT_RESNET18_EPOCHS
        and checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch",
        "ReMST requires a terminal matched Direct-ResNet18 checkpoint",
    )
    _require(
        checkpoint.get("scene_disjoint") is True
        and checkpoint.get("split_protocol") == SCENE_SPLIT_PROTOCOL,
        "ReMST-ResNet18 source must use the scene-disjoint Direct split",
    )
    state = checkpoint.get("model_state")
    _require(
        isinstance(state, Mapping),
        "Direct-ResNet18 model state is missing",
    )
    anchor = MomentExactResNet18Anchor(
        progress_bins=progress_bins,
        initial_scale=initial_scale,
    )
    feature_state = {
        str(name).removeprefix("backbone."): value
        for name, value in state.items()
        if str(name).startswith("backbone.")
        and not str(name).startswith("backbone.fc.")
    }
    feature_load = anchor.raw_encoder.load_state_dict(
        feature_state, strict=True
    )
    _require(
        not feature_load.missing_keys and not feature_load.unexpected_keys,
        "Direct-ResNet18 encoder state does not load strictly",
    )
    point_state = {
        str(name).removeprefix("backbone.fc."): value
        for name, value in state.items()
        if str(name).startswith("backbone.fc.")
    }
    point_load = anchor.raw_posterior_head.point_projection.load_state_dict(
        point_state, strict=True
    )
    _require(
        not point_load.missing_keys and not point_load.unexpected_keys,
        "Direct-ResNet18 scalar head does not load strictly",
    )
    target_device = torch.device(device)
    anchor = anchor.to(target_device).eval()
    metadata = {
        "source": str(source),
        "source_protocol": str(checkpoint["protocol"]),
        "source_architecture": str(checkpoint["architecture"]),
        "source_seed": int(checkpoint["seed"]),
        "source_epochs": int(checkpoint["epochs"]),
        "source_checkpoint_selection": str(
            checkpoint["checkpoint_selection"]
        ),
        "source_split_protocol": str(checkpoint["split_protocol"]),
        "source_scene_disjoint": True,
        "progress_bins": int(progress_bins),
        "initial_posterior_scale": float(initial_scale),
        "point_parameters_imported": True,
        "posterior_scale_parameters_fresh": True,
        "posterior_scale_mode": "fixed_during_correction_only_training",
        "single_backbone_parameter_set": True,
        "raw_and_sarn_observations": 2,
    }
    return anchor, metadata


def remst_resnet18_publication_identity() -> dict[str, str]:
    return {
        "short_name": "ReMST-ResNet18",
        "full_name": "ReMST with a shared Direct-ResNet18 anchor",
        "display_name": "ReMST-ResNet18 (single backbone)",
        "architecture": REMST_RESNET18_ARCHITECTURE,
        "backbone": "torchvision ResNet-18",
    }


class ReMSTResNet18Correction(ReMSTCorrection):
    """ReMST transport configured for shared ResNet-18 relation features."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            stride8_channels=RESNET18_STRIDE8_CHANNELS,
            stride16_channels=RESNET18_STRIDE16_CHANNELS,
            **kwargs,
        )

    def forward(self, *args: object, **kwargs: object) -> dict[str, Any]:
        output = super().forward(*args, **kwargs)
        output["architecture"] = REMST_RESNET18_ARCHITECTURE
        output["publication_model"] = remst_resnet18_publication_identity()
        output["single_backbone_parameter_set"] = True
        return output


__all__ = [
    "DIRECT_RESNET18_ARCHITECTURE",
    "MomentExactResNet18Anchor",
    "REMST_RESNET18_ARCHITECTURE",
    "RESNET18_REPRESENTATION_FEATURES",
    "RESNET18_STRIDE16_CHANNELS",
    "RESNET18_STRIDE8_CHANNELS",
    "ReMSTResNet18Correction",
    "ReMSTResNet18Encoder",
    "load_moment_exact_resnet18_anchor",
    "remst_resnet18_publication_identity",
]
