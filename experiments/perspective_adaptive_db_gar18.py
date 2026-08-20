"""PA-DB-GAR18: perspective-adaptive CBAM calibration for DB-GAR18.

The candidate is intentionally isolated from the frozen DB-GAR18 runner.  It
loads a terminal DB-GAR18 checkpoint and supports two isolated CBAM-internal
extensions under the same short paired-view calibration: a gradient-gated
axial residual (GGAR) in spatial attention, or an occupancy-normalized channel
residual (ON-CBAM).

GGAR retains the original 7x7 spatial logits and adds 1x7 and 7x1 residual
logit branches.  Their per-image gates depend on horizontal and vertical input
gradient statistics.  Both gate gains are initialized to exactly zero, so a
converted DB-GAR18 checkpoint has the same inference output before calibration.
ON-CBAM retains the original average/max descriptors and shared channel MLP.
It adds a detached, foreground-weighted average descriptor whose residual is
gated by the estimated unoccupied fraction.  Its channelwise gains also start
at exactly zero.

Training uses only the already-authorized 14,442 scene-disjoint SyncG fit rows
and 434 xiangmu2 real-development rows.  Each ordinary augmented view is paired
with an independently seeded 20--45 degree projective view.  Progress is
supervised on both views, transformed geometry is supervised only for SyncG,
and a progress-consistency term ties each projective prediction to its clean
counterpart.  The two-epoch checkpoint is terminal-only; no evaluation metric
is read during training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments import robustness_degradations
from experiments.domain_balanced_geoattn_resnet18 import (
    ARCHITECTURE as DB_GAR_ARCHITECTURE,
    DOMAIN_BATCH_SIZE,
    DOMAIN_PROGRESS_WEIGHT,
    EXPECTED_REAL_DEVELOPMENT_SAMPLES,
    EXPECTED_REAL_GROUPS,
    EXPECTED_SYNTHETIC_FIT_SAMPLES,
    EXPECTED_SYNTHETIC_HOLDOUT_IDS,
    PROTOCOL as DB_GAR_PROTOCOL,
    REAL_AUGMENT_ANCHORS,
    REAL_TARGET_BINS,
    RealProgressSample,
    ShufflePadSampler,
    TargetBinBalancedSampler,
    _sha256_file,
    domain_balanced_objective,
    load_real_progress_samples,
    strong_real_augmentation,
)
from experiments.geoattn_resnet18_progress import (
    CBAM,
    DIRECTION_LOSS_WEIGHT,
    PIVOT_LOSS_WEIGHT,
    REFERENCE_LOSS_WEIGHT,
    GeoAttnResNet18,
    _apply_homography,
    _geometry_targets,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    DirectSample,
    _canonical_json_bytes,
    _canonical_sha256,
    _configure_reproducibility,
    _require,
    load_training_samples,
    matched_cagh_augmentation,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


ATTENTION_MODES: Final[tuple[str, ...]] = ("legacy", "axial", "occupancy")
MODE_PROTOCOLS: Final[dict[str, str]] = {
    "legacy": "cbam_pic_v1",
    "axial": "axial_cbam_pic_v1",
    "occupancy": "on_cbam_pic_v1",
}
MODE_ARCHITECTURES: Final[dict[str, str]] = {
    "legacy": "CBAM-PIC",
    "axial": "Axial-CBAM-PIC",
    "occupancy": "ON-CBAM-PIC",
}
MODE_METHOD_PREFIXES: Final[dict[str, str]] = {
    "legacy": "cbam_pic",
    "axial": "axial_cbam_pic",
    "occupancy": "on_cbam_pic",
}
# Backward-friendly aliases name the default candidate, while every checkpoint
# also records its explicit attention_mode and mode-specific identity.
PROTOCOL: Final[str] = MODE_PROTOCOLS["axial"]
ARCHITECTURE: Final[str] = MODE_ARCHITECTURES["axial"]
METHOD_PREFIX: Final[str] = MODE_METHOD_PREFIXES["axial"]

CALIBRATION_EPOCHS: Final[int] = 2
CALIBRATION_LEARNING_RATE: Final[float] = 1e-5
OCCUPANCY_GATE_LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-4
PERSPECTIVE_DEGREES_MIN: Final[float] = 20.0
PERSPECTIVE_DEGREES_MAX: Final[float] = 45.0
PERSPECTIVE_PROGRESS_WEIGHT: Final[float] = 0.50
PERSPECTIVE_AUXILIARY_WEIGHT: Final[float] = 0.25
PROGRESS_CONSISTENCY_WEIGHT: Final[float] = 0.25
COMBINED_BLUR_PROBABILITY: Final[float] = 0.25
COMBINED_BLUR_SIGMA_FRACTION: Final[float] = 0.0030
PERSPECTIVE_SEED_OFFSET: Final[int] = 4_510_019


class OccupancyNormalizedChannelAttention(nn.Module):
    """Original CBAM channel logits plus a zero-gated occupancy residual."""

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        hidden = (
            max(1, int(channels) // int(reduction))
            if hidden_channels is None
            else int(hidden_channels)
        )
        _require(int(channels) >= 1 and hidden >= 1, "invalid ON-CBAM channels")
        self.shared = nn.Sequential(
            nn.Conv2d(int(channels), hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(channels), kernel_size=1, bias=False),
        )
        # A separate gain per output channel lets the existing shared MLP
        # decide which features benefit from foreground-normalized pooling.
        self.channelwise_gate_gain = nn.Parameter(
            torch.zeros(int(channels), dtype=torch.float32)
        )

    @staticmethod
    def foreground_statistics(
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return foreground-weighted descriptors and soft occupied fractions."""

        energy = torch.mean(torch.abs(value), dim=1, keepdim=True)
        peak = torch.amax(energy, dim=(2, 3), keepdim=True).clamp_min(1e-6)
        weights = torch.clamp(energy / peak, min=0.0, max=1.0).detach()
        occupancy = torch.mean(weights, dim=(2, 3), keepdim=True)
        denominator = torch.sum(weights, dim=(2, 3), keepdim=True).clamp_min(1e-6)
        weighted_average = torch.sum(
            value * weights, dim=(2, 3), keepdim=True
        ) / denominator
        return weighted_average, occupancy

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        global_average = F.adaptive_avg_pool2d(value, 1)
        average_logits = self.shared(global_average)
        maximum_logits = self.shared(F.adaptive_max_pool2d(value, 1))
        weighted_average, occupancy = self.foreground_statistics(value)
        descriptor_residual = self.shared(weighted_average) - average_logits
        # Occupancy remains an explicit linear modulator even after the learned
        # gain grows; placing it outside tanh avoids saturation bypassing the
        # per-image occupied-fraction signal.
        gate = (1.0 - occupancy) * torch.tanh(
            self.channelwise_gate_gain.view(1, -1, 1, 1)
        )
        logits = average_logits + maximum_logits + gate * descriptor_residual
        return value * torch.sigmoid(logits)


class PerspectiveAdaptiveSpatialAttention(nn.Module):
    """Original 7x7 logits plus zero-gated 1x7/7x1 residual logits."""

    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        _require(kernel_size == 7, "GGAR preserves the frozen 7x7 CBAM kernel")
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.horizontal = nn.Conv2d(
            2, 1, kernel_size=(1, 7), padding=(0, 3), bias=False
        )
        self.vertical = nn.Conv2d(
            2, 1, kernel_size=(7, 1), padding=(3, 0), bias=False
        )
        # tanh(0 * statistic) is exactly zero.  The nonzero residual kernels
        # make gate_gain trainable on the first optimization step.
        self.gate_gain = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def initialize_residuals_from_base(self) -> None:
        """Seed the two residual kernels from the loaded 7x7 CBAM kernel."""

        with torch.no_grad():
            self.horizontal.weight.copy_(self.conv.weight.mean(dim=2, keepdim=True))
            self.vertical.weight.copy_(self.conv.weight.mean(dim=3, keepdim=True))
            self.gate_gain.zero_()

    @staticmethod
    def _gradient_statistics(value: torch.Tensor) -> torch.Tensor:
        horizontal = torch.mean(torch.abs(value[..., 1:] - value[..., :-1]), dim=(1, 2, 3))
        vertical = torch.mean(torch.abs(value[..., 1:, :] - value[..., :-1, :]), dim=(1, 2, 3))
        denominator = (horizontal + vertical).clamp_min(1e-6)
        # Detaching prevents the feature extractor from changing the statistic
        # merely to open a gate; the learned gate still adapts per image.
        return torch.stack((horizontal / denominator, vertical / denominator), dim=1).detach()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = torch.mean(value, dim=1, keepdim=True)
        maximum = torch.max(value, dim=1, keepdim=True).values
        descriptor = torch.cat((average, maximum), dim=1)
        base_logits = self.conv(descriptor)
        statistics = self._gradient_statistics(value)
        gates = torch.tanh(statistics * self.gate_gain.view(1, 2))
        horizontal = self.horizontal(descriptor)
        vertical = self.vertical(descriptor)
        logits = (
            base_logits
            + gates[:, 0, None, None, None] * horizontal
            + gates[:, 1, None, None, None] * vertical
        )
        return value * torch.sigmoid(logits)


class PerspectiveAdaptiveGeoAttnResNet18(GeoAttnResNet18):
    """GeoAttnResNet18 with zero-effect GGAR and ON-CBAM candidates."""

    def __init__(self, *, imagenet_pretrained: bool = False) -> None:
        super().__init__(imagenet_pretrained=imagenet_pretrained)
        modules = [module for module in self.modules() if isinstance(module, CBAM)]
        _require(len(modules) == 8, "GeoAttnResNet18 no longer contains eight CBAM blocks")
        for module in modules:
            channels = int(module.channel.shared[0].in_channels)
            hidden = int(module.channel.shared[0].out_channels)
            module.channel = OccupancyNormalizedChannelAttention(
                channels, hidden_channels=hidden
            )
            module.spatial = PerspectiveAdaptiveSpatialAttention()

    def initialize_perspective_residuals(self) -> None:
        modules = [
            module
            for module in self.modules()
            if isinstance(module, PerspectiveAdaptiveSpatialAttention)
        ]
        _require(len(modules) == 8, "the candidate must contain eight GGAR modules")
        for module in modules:
            module.initialize_residuals_from_base()
        channel_modules = [
            module
            for module in self.modules()
            if isinstance(module, OccupancyNormalizedChannelAttention)
        ]
        _require(len(channel_modules) == 8, "the candidate must contain eight ON-CBAM modules")
        with torch.no_grad():
            for module in channel_modules:
                module.channelwise_gate_gain.zero_()


def load_db_gar_state_into_pa_model(
    model: PerspectiveAdaptiveGeoAttnResNet18,
    state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load DB-GAR while allowing only the zero-effect candidate keys."""

    incompatible = model.load_state_dict(state, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    missing = tuple(incompatible.missing_keys)
    _require(not unexpected, f"unexpected DB-GAR parent keys: {unexpected}")
    allowed_suffixes = (
        ".spatial.horizontal.weight",
        ".spatial.vertical.weight",
        ".spatial.gate_gain",
        ".channel.channelwise_gate_gain",
    )
    _require(
        len(missing) == 32 and all(key.endswith(allowed_suffixes) for key in missing),
        f"DB-GAR to PA-DB-GAR state mismatch: {missing}",
    )
    model.initialize_perspective_residuals()
    return missing


def configure_attention_mode(
    model: PerspectiveAdaptiveGeoAttnResNet18, attention_mode: str
) -> None:
    """Make each candidate arm differ only in its enabled CBAM residual."""

    _require(attention_mode in ATTENTION_MODES, "unknown attention mode")
    modules = [
        module
        for module in model.modules()
        if isinstance(module, PerspectiveAdaptiveSpatialAttention)
    ]
    _require(len(modules) == 8, "attention-mode configuration requires eight GGAR modules")
    channel_modules = [
        module
        for module in model.modules()
        if isinstance(module, OccupancyNormalizedChannelAttention)
    ]
    _require(
        len(channel_modules) == 8,
        "attention-mode configuration requires eight ON-CBAM modules",
    )
    with torch.no_grad():
        for module in channel_modules:
            module.channelwise_gate_gain.zero_()
        if attention_mode in ("legacy", "occupancy"):
            for module in modules:
                module.horizontal.weight.zero_()
                module.vertical.weight.zero_()
                module.gate_gain.zero_()


def _paired_projective_view(
    image: np.ndarray,
    points: np.ndarray | None,
    *,
    rng: np.random.Generator,
    epoch: int,
    total_epochs: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Apply the robustness protocol's projective family with a separate RNG."""

    _require(image is not None and image.size > 0, "paired perspective input is empty")
    _require(total_epochs >= 1 and 0 <= epoch < total_epochs, "invalid curriculum epoch")
    height, width = image.shape[:2]
    fraction = float(epoch + 1) / float(total_epochs)
    maximum = PERSPECTIVE_DEGREES_MIN + fraction * (
        PERSPECTIVE_DEGREES_MAX - PERSPECTIVE_DEGREES_MIN
    )
    degrees = float(rng.uniform(PERSPECTIVE_DEGREES_MIN, maximum))
    axis = "yaw" if int(rng.integers(0, 2)) == 0 else "pitch"
    sign = -1 if int(rng.integers(0, 2)) == 0 else 1
    destination = robustness_degradations._projected_corners(
        width, height, degrees, axis, sign
    )
    source = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=robustness_degradations._border_median(image),
    )
    blur_applied = bool(rng.random() < COMBINED_BLUR_PROBABILITY)
    if blur_applied:
        sigma = max(0.1, COMBINED_BLUR_SIGMA_FRACTION * float(min(height, width)))
        warped = cv2.GaussianBlur(
            warped,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    transformed: np.ndarray | None = None
    if points is not None:
        values = np.asarray(points, dtype=np.float32)
        _require(values.ndim == 2 and values.shape[1] == 2, "points must be Nx2")
        pixel_scale = np.asarray(
            [float(max(width - 1, 1)), float(max(height - 1, 1))], dtype=np.float32
        )
        pixel_points = values * pixel_scale
        transformed_pixels = cv2.perspectiveTransform(
            pixel_points.reshape(1, -1, 2), homography
        ).reshape(-1, 2)
        transformed = transformed_pixels / pixel_scale
        _require(bool(np.isfinite(transformed).all()), "projective labels are non-finite")
        _require(
            bool(((transformed >= -1e-5) & (transformed <= 1.0 + 1e-5)).all()),
            "projective labels escaped the fitted plane",
        )
        transformed = np.clip(transformed, 0.0, 1.0).astype(np.float32)
    metadata = {
        "degrees": degrees,
        "curriculum_max_degrees": maximum,
        "axis": axis,
        "sign": sign,
        "blur_applied": blur_applied,
        "homography": homography.tolist(),
    }
    return np.ascontiguousarray(warped), transformed, metadata


class PairedSyntheticDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        seed: int,
        total_epochs: int = CALIBRATION_EPOCHS,
        augmentation: PhotoAugmentation | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self.augmentation = augmentation or matched_cagh_augmentation()
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "paired synthetic dataset is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=IMAGE_SIZE)
        _require(bool(sample.protected_points_xy), f"{sample.sample_id}: geometry is missing")
        left, top, right, bottom = bounds
        scale = np.asarray([float(right - left), float(bottom - top)], dtype=np.float32)
        points = (
            np.asarray(sample.protected_points_xy, dtype=np.float32)
            - np.asarray([left, top], dtype=np.float32)
        ) / scale
        clean_rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 97
        )
        roi, forward, _geometry_code = _cagh_augment_geometry(
            roi, points, clean_rng, self.augmentation
        )
        points = _apply_homography(points, forward)
        roi, _photo_code = _cagh_augment_photo(
            roi, clean_rng, self.augmentation
        )
        clean_labels = _geometry_targets(points)
        perspective_rng = np.random.default_rng(
            self.seed
            + PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + int(index) * 193
        )
        perspective, perspective_points, _metadata = _paired_projective_view(
            roi,
            points,
            rng=perspective_rng,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
        )
        _require(perspective_points is not None, "synthetic projective labels disappeared")
        perspective_labels = _geometry_targets(perspective_points)
        return {
            "clean_image": normalized_rgb_tensor(roi),
            "perspective_image": normalized_rgb_tensor(perspective),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "clean_pivot": clean_labels["pivot"],
            "clean_direction_sin_cos": clean_labels["direction_sin_cos"],
            "clean_references": clean_labels["references"],
            "perspective_pivot": perspective_labels["pivot"],
            "perspective_direction_sin_cos": perspective_labels["direction_sin_cos"],
            "perspective_references": perspective_labels["references"],
        }


class PairedRealDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[RealProgressSample],
        *,
        seed: int,
        total_epochs: int = CALIBRATION_EPOCHS,
        augmentation: PhotoAugmentation | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self.augmentation = augmentation or strong_real_augmentation()
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "paired real dataset is empty")
        self._preloaded = tuple(self._load(index) for index in range(len(self.samples)))

    def _load(self, index: int) -> np.ndarray:
        _payload, image = load_canonical_roi(self.samples[index].roi)
        return np.ascontiguousarray(direct_resize_whole_roi(image, size=IMAGE_SIZE))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        if isinstance(index, tuple):
            source_index, draw_ordinal = int(index[0]), int(index[1])
        else:
            source_index, draw_ordinal = int(index), 0
        sample = self.samples[source_index]
        image = self._preloaded[source_index].copy()
        clean_rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + source_index * 97
            + draw_ordinal * 7_919
        )
        image, _forward, _geometry_code = _cagh_augment_geometry(
            image, REAL_AUGMENT_ANCHORS, clean_rng, self.augmentation
        )
        image, _photo_code = _cagh_augment_photo(image, clean_rng, self.augmentation)
        perspective_rng = np.random.default_rng(
            self.seed
            + PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + source_index * 193
            + draw_ordinal * 15_869
        )
        perspective, _points, _metadata = _paired_projective_view(
            image,
            None,
            rng=perspective_rng,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
        )
        return {
            "clean_image": normalized_rgb_tensor(image),
            "perspective_image": normalized_rgb_tensor(perspective),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
        }


def set_pa_calibration_stage(
    model: PerspectiveAdaptiveGeoAttnResNet18,
    *,
    attention_mode: str = "axial",
) -> tuple[str, ...]:
    """Apply the matched PIC stage and enable only the selected residual."""

    _require(attention_mode in ATTENTION_MODES, "unknown attention mode")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.progress_head, model.geometry_head, model.backbone.layer4):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, PerspectiveAdaptiveSpatialAttention):
            adaptive_parameters = (
                *module.horizontal.parameters(),
                *module.vertical.parameters(),
                module.gate_gain,
            )
            for parameter in adaptive_parameters:
                parameter.requires_grad_(attention_mode == "axial")
            if attention_mode != "axial":
                _require(
                    all(bool(torch.count_nonzero(parameter) == 0) for parameter in adaptive_parameters),
                    "disabled GGAR parameters must remain exactly zero",
                )
        elif isinstance(module, OccupancyNormalizedChannelAttention):
            module.channelwise_gate_gain.requires_grad_(attention_mode == "occupancy")
            if attention_mode != "occupancy":
                _require(
                    bool(torch.count_nonzero(module.channelwise_gate_gain) == 0),
                    "disabled ON-CBAM gate gains must remain exactly zero",
                )
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    _require(bool(names), "PIC calibration has no trainable parameters")
    return names


def _geometry_auxiliary(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pivot = F.smooth_l1_loss(
        outputs["pivot"], batch[f"{prefix}_pivot"], beta=0.05
    )
    direction = (
        1.0
        - torch.sum(
            outputs["direction_sin_cos"] * batch[f"{prefix}_direction_sin_cos"],
            dim=1,
        )
    ).mean()
    references = F.smooth_l1_loss(
        outputs["references"], batch[f"{prefix}_references"], beta=0.05
    )
    auxiliary = (
        PIVOT_LOSS_WEIGHT * pivot
        + DIRECTION_LOSS_WEIGHT * direction
        + REFERENCE_LOSS_WEIGHT * references
    )
    return auxiliary, {
        f"{prefix}_pivot": pivot,
        f"{prefix}_direction": direction,
        f"{prefix}_references": references,
    }


def perspective_adaptive_objective(
    synthetic_clean: Mapping[str, torch.Tensor],
    synthetic_perspective: Mapping[str, torch.Tensor],
    synthetic_batch: Mapping[str, torch.Tensor],
    real_clean: Mapping[str, torch.Tensor],
    real_perspective: Mapping[str, torch.Tensor],
    real_batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_synthetic_batch = {
        "progress": synthetic_batch["progress"],
        "pivot": synthetic_batch["clean_pivot"],
        "direction_sin_cos": synthetic_batch["clean_direction_sin_cos"],
        "references": synthetic_batch["clean_references"],
    }
    clean_loss, clean_components = domain_balanced_objective(
        synthetic_clean,
        clean_synthetic_batch,
        real_clean,
        real_batch,
    )
    synthetic_progress = F.smooth_l1_loss(
        synthetic_perspective["progress"], synthetic_batch["progress"], beta=0.05
    )
    real_progress = F.smooth_l1_loss(
        real_perspective["progress"], real_batch["progress"], beta=0.05
    )
    projective_progress = (
        DOMAIN_PROGRESS_WEIGHT * synthetic_progress
        + DOMAIN_PROGRESS_WEIGHT * real_progress
    )
    projective_auxiliary, projective_components = _geometry_auxiliary(
        synthetic_perspective, synthetic_batch, prefix="perspective"
    )
    clean_values = torch.cat(
        (synthetic_clean["progress"], real_clean["progress"]), dim=0
    ).detach()
    projective_values = torch.cat(
        (synthetic_perspective["progress"], real_perspective["progress"]), dim=0
    )
    consistency = F.smooth_l1_loss(projective_values, clean_values, beta=0.05)
    total = (
        clean_loss
        + PERSPECTIVE_PROGRESS_WEIGHT * projective_progress
        + PERSPECTIVE_AUXILIARY_WEIGHT * projective_auxiliary
        + PROGRESS_CONSISTENCY_WEIGHT * consistency
    )
    components = {
        "clean_loss": clean_loss,
        "clean_synthetic_progress": clean_components["synthetic_progress"],
        "clean_real_progress": clean_components["real_progress"],
        "clean_auxiliary": clean_components["auxiliary"],
        "projective_synthetic_progress": synthetic_progress,
        "projective_real_progress": real_progress,
        "projective_progress": projective_progress,
        "projective_auxiliary": projective_auxiliary,
        "consistency": consistency,
        **projective_components,
    }
    return total, components


def _slice_outputs(
    outputs: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in outputs.items()}


def _train_epoch(
    model: PerspectiveAdaptiveGeoAttnResNet18,
    synthetic_loader: DataLoader,
    real_loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    _require(len(synthetic_loader) == len(real_loader), "domain loader lengths differ")
    model.train()
    use_amp = device.type == "cuda"
    total_keys = (
        "loss",
        "clean_loss",
        "clean_synthetic_progress",
        "clean_real_progress",
        "clean_auxiliary",
        "projective_synthetic_progress",
        "projective_real_progress",
        "projective_progress",
        "projective_auxiliary",
        "consistency",
        "perspective_pivot",
        "perspective_direction",
        "perspective_references",
    )
    totals = {key: 0.0 for key in total_keys}
    steps = 0
    for synthetic_raw, real_raw in zip(synthetic_loader, real_loader, strict=True):
        synthetic = {
            key: value.to(device, non_blocking=use_amp)
            for key, value in synthetic_raw.items()
        }
        real = {
            key: value.to(device, non_blocking=use_amp) for key, value in real_raw.items()
        }
        count = int(synthetic["progress"].numel())
        _require(
            count == int(real["progress"].numel()) == DOMAIN_BATCH_SIZE,
            "paired domain batch is not balanced",
        )
        optimizer.zero_grad(set_to_none=True)
        images = torch.cat(
            (
                synthetic["clean_image"],
                synthetic["perspective_image"],
                real["clean_image"],
                real["perspective_image"],
            ),
            dim=0,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model.forward_training(images)
            synthetic_clean = _slice_outputs(outputs, 0, count)
            synthetic_perspective = _slice_outputs(outputs, count, 2 * count)
            real_clean = _slice_outputs(outputs, 2 * count, 3 * count)
            real_perspective = _slice_outputs(outputs, 3 * count, 4 * count)
            loss, components = perspective_adaptive_objective(
                synthetic_clean,
                synthetic_perspective,
                synthetic,
                real_clean,
                real_perspective,
                real,
            )
        _require(bool(torch.isfinite(loss)), "PA-DB-GAR18 loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 5.0
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        for key, value in components.items():
            totals[key] += float(value.detach().cpu())
        steps += 1
    _require(steps > 0, "PA-DB-GAR18 epoch is empty")
    return {**{key: value / steps for key, value in totals.items()}, "steps": float(steps)}


def _load_db_gar_parent(
    checkpoint_path: Path,
) -> tuple[PerspectiveAdaptiveGeoAttnResNet18, Mapping[str, Any], tuple[str, ...]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"DB-GAR18 parent checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "DB-GAR18 parent is not an object")
    _require(checkpoint.get("protocol") == DB_GAR_PROTOCOL, "DB-GAR18 parent protocol mismatch")
    _require(
        checkpoint.get("architecture") == DB_GAR_ARCHITECTURE,
        "DB-GAR18 parent architecture mismatch",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "DB-GAR18 parent model state is missing")
    model = PerspectiveAdaptiveGeoAttnResNet18(imagenet_pretrained=False)
    missing = load_db_gar_state_into_pa_model(model, state)
    return model, checkpoint, missing


def _parameter_inventory(model: PerspectiveAdaptiveGeoAttnResNet18) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    adaptive = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, PerspectiveAdaptiveSpatialAttention)
        for parameter in (
            *module.horizontal.parameters(),
            *module.vertical.parameters(),
            module.gate_gain,
        )
    )
    occupancy = sum(
        module.channelwise_gate_gain.numel()
        for module in model.modules()
        if isinstance(module, OccupancyNormalizedChannelAttention)
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "ggar": int(adaptive),
        "occupancy_gate": int(occupancy),
    }


def _base_cosine_multiplier(epoch: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * float(epoch) / CALIBRATION_EPOCHS))


def train(
    *,
    parent_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    real_manifest_path: Path,
    real_labels_path: Path,
    output_path: Path,
    seed: int,
    attention_mode: str = "axial",
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    """Run one arm of the fixed two-epoch paired-view calibration."""

    _require(workers >= 0, "workers must be non-negative")
    _require(attention_mode in ATTENTION_MODES, "unknown attention mode")
    synthetic_samples, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    real_samples = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "PA-DB-GAR18 requires the frozen 14,442/1,558 SyncG roster",
    )
    _require(
        len(real_samples) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_samples}) == EXPECTED_REAL_GROUPS,
        "PA-DB-GAR18 requires the frozen 434-row/11-group real-development roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent, missing = _load_db_gar_parent(parent_checkpoint_path)
    _require(int(parent["seed"]) == int(seed), "parent and calibration seeds differ")
    configure_attention_mode(model, attention_mode)
    model = model.to(device)
    trainable = set_pa_calibration_stage(model, attention_mode=attention_mode)
    synthetic_dataset = PairedSyntheticDataset(
        synthetic_samples, seed=seed, total_epochs=CALIBRATION_EPOCHS
    )
    real_dataset = PairedRealDataset(
        real_samples, seed=seed + 10_000, total_epochs=CALIBRATION_EPOCHS
    )
    padded_samples = int(
        math.ceil(len(synthetic_dataset) / DOMAIN_BATCH_SIZE) * DOMAIN_BATCH_SIZE
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if attention_mode == "occupancy":
        occupancy_gate_parameters = [
            module.channelwise_gate_gain
            for module in model.modules()
            if isinstance(module, OccupancyNormalizedChannelAttention)
            and module.channelwise_gate_gain.requires_grad
        ]
        _require(
            len(occupancy_gate_parameters) == 8,
            "occupancy arm requires eight trainable channelwise gates",
        )
        occupancy_ids = {id(parameter) for parameter in occupancy_gate_parameters}
        base_parameters = [
            parameter
            for parameter in trainable_parameters
            if id(parameter) not in occupancy_ids
        ]
        _require(bool(base_parameters), "occupancy arm has no base calibration parameters")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": base_parameters,
                    "lr": CALIBRATION_LEARNING_RATE,
                    "name": "base",
                },
                {
                    "params": occupancy_gate_parameters,
                    "lr": OCCUPANCY_GATE_LEARNING_RATE,
                    "name": "occupancy_gate",
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )
        # Only the base layer4/heads group follows the historical two-epoch
        # cosine schedule.  The newly introduced zero-gate group stays fixed.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[_base_cosine_multiplier, lambda _epoch: 1.0],
        )
        scheduler_name = "LambdaLR(base=cosine,occupancy_gate=fixed)"
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": trainable_parameters,
                    "lr": CALIBRATION_LEARNING_RATE,
                    "name": "base",
                }
            ],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CALIBRATION_EPOCHS
        )
        scheduler_name = "CosineAnnealingLR"
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(CALIBRATION_EPOCHS):
        synthetic_dataset.set_epoch(epoch)
        real_dataset.set_epoch(epoch)
        synthetic_sampler = ShufflePadSampler(
            len(synthetic_dataset),
            batch_size=DOMAIN_BATCH_SIZE,
            seed=seed + epoch,
        )
        real_sampler = TargetBinBalancedSampler(
            [sample.normalized_target for sample in real_samples],
            num_samples=padded_samples,
            bins=REAL_TARGET_BINS,
            seed=seed + 100_000 + epoch,
        )
        synthetic_loader = DataLoader(
            synthetic_dataset,
            batch_size=DOMAIN_BATCH_SIZE,
            sampler=synthetic_sampler,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            drop_last=False,
        )
        real_loader = DataLoader(
            real_dataset,
            batch_size=DOMAIN_BATCH_SIZE,
            sampler=real_sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        metrics = _train_epoch(
            model,
            synthetic_loader,
            real_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "learning_rates": {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            },
            "curriculum_max_degrees": PERSPECTIVE_DEGREES_MIN
            + float(epoch + 1) / float(CALIBRATION_EPOCHS)
            * (PERSPECTIVE_DEGREES_MAX - PERSPECTIVE_DEGREES_MIN),
            "train": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    parent_source = Path(parent_checkpoint_path).resolve()
    synthetic_manifest = Path(synthetic_manifest_path).resolve()
    synthetic_split = Path(synthetic_split_path).resolve()
    real_manifest = Path(real_manifest_path).resolve()
    real_labels = Path(real_labels_path).resolve()
    protocol = MODE_PROTOCOLS[attention_mode]
    architecture = MODE_ARCHITECTURES[attention_mode]
    method = f"{MODE_METHOD_PREFIXES[attention_mode]}_seed_{seed}"
    checkpoint = {
        "schema_version": 1,
        "protocol": protocol,
        "architecture": architecture,
        "architecture_detail": (
            "DB-GAR18 calibrated with paired full-canvas projective views; "
            + (
                "zero-initialized gradient-gated 1x7/7x1 axial residual logits "
                "inside every CBAM block"
                if attention_mode == "axial"
                else (
                    "zero-initialized Occupancy-Normalized CBAM residual in the "
                    "shared channel-attention logits; all GGAR parameters fixed "
                    "at exact zero"
                    if attention_mode == "occupancy"
                    else "legacy 7x7 CBAM spatial attention with all GGAR and "
                    "ON-CBAM parameters fixed at exact zero"
                )
            )
        ),
        "method": method,
        "attention_mode": attention_mode,
        "seed": int(seed),
        "image_size": IMAGE_SIZE,
        "parent_checkpoint": str(parent_source),
        "parent_checkpoint_sha256": _sha256_file(parent_source),
        "parent_protocol": str(parent["protocol"]),
        "parent_architecture": str(parent["architecture"]),
        "parent_seed": int(parent["seed"]),
        "parent_missing_keys_initialized": list(missing),
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": _sha256_file(synthetic_manifest),
        "synthetic_split": str(synthetic_split),
        "synthetic_split_sha256": _sha256_file(synthetic_split),
        "synthetic_fit_samples": len(synthetic_samples),
        "synthetic_holdout_samples": len(roster.validation_ids),
        "synthetic_fit_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "synthetic_holdout_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
        "synthetic_holdout_access_during_training": (
            "IDs/count only; no target, bbox, image path, or image"
        ),
        "real_manifest": str(real_manifest),
        "real_manifest_sha256": _sha256_file(real_manifest),
        "real_labels": str(real_labels),
        "real_labels_sha256": _sha256_file(real_labels),
        "real_development_samples": len(real_samples),
        "real_groups": len({sample.group_id for sample in real_samples}),
        "real_sample_ids_sha256": _canonical_sha256(
            sorted(sample.sample_id for sample in real_samples)
        ),
        "schedule": {
            "checkpoint_selection": "terminal_fixed_epoch",
            "epochs": CALIBRATION_EPOCHS,
            "learning_rate": CALIBRATION_LEARNING_RATE,
            "learning_rates": (
                {
                    "base": CALIBRATION_LEARNING_RATE,
                    "occupancy_gate": OCCUPANCY_GATE_LEARNING_RATE,
                }
                if attention_mode == "occupancy"
                else {"base": CALIBRATION_LEARNING_RATE}
            ),
            "weight_decay": WEIGHT_DECAY,
            "scheduler": scheduler_name,
            "gradient_clip_norm": 5.0,
            "domain_batch_size_each": DOMAIN_BATCH_SIZE,
            "paired_views_per_domain_sample": 2,
            "trainable_parameter_names": list(trainable),
        },
        "perspective_curriculum": {
            "family": robustness_degradations.ROBUSTNESS_PROTOCOL,
            "evaluation_robustness_seed_used": False,
            "training_seed_offset": PERSPECTIVE_SEED_OFFSET,
            "degrees_min": PERSPECTIVE_DEGREES_MIN,
            "degrees_max": PERSPECTIVE_DEGREES_MAX,
            "axes": ["yaw", "pitch"],
            "signs": [-1, 1],
            "border": "per-image border median constant",
            "combined_blur_probability": COMBINED_BLUR_PROBABILITY,
            "combined_blur_sigma_fraction": COMBINED_BLUR_SIGMA_FRACTION,
        },
        "loss_weights": {
            "clean_domain_objective": 1.0,
            "perspective_progress": PERSPECTIVE_PROGRESS_WEIGHT,
            "perspective_synthetic_geometry": PERSPECTIVE_AUXILIARY_WEIGHT,
            "progress_consistency": PROGRESS_CONSISTENCY_WEIGHT,
        },
        "parameter_inventory": _parameter_inventory(model),
        "history": history,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "protocol": protocol,
        "architecture": architecture,
        "method": method,
        "attention_mode": attention_mode,
        "checkpoint": str(output),
        "epochs": CALIBRATION_EPOCHS,
    }


def load_checkpoint_predictor(
    checkpoint_path: Path, *, device_name: str
) -> tuple[str, str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"PIC checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "PIC checkpoint is not an object")
    attention_mode = checkpoint.get("attention_mode")
    _require(attention_mode in ATTENTION_MODES, "PIC attention mode is missing or invalid")
    _require(
        checkpoint.get("protocol") == MODE_PROTOCOLS[attention_mode],
        "PIC protocol does not match attention mode",
    )
    _require(
        checkpoint.get("architecture") == MODE_ARCHITECTURES[attention_mode],
        "PIC architecture does not match attention mode",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "PIC model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = PerspectiveAdaptiveGeoAttnResNet18(imagenet_pretrained=False)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    missing = tuple(incompatible.missing_keys)
    _require(not unexpected, f"unexpected PIC checkpoint keys: {unexpected}")
    if attention_mode == "occupancy":
        _require(not missing, f"ON-CBAM checkpoint is incomplete: {missing}")
    else:
        # Checkpoints produced before the occupancy arm was introduced contain
        # the same legacy/axial graph except for these zero-effect gate tensors.
        _require(
            len(missing) in (0, 8)
            and all(key.endswith(".channel.channelwise_gate_gain") for key in missing),
            f"legacy PIC checkpoint is incomplete: {missing}",
        )
        if missing:
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, OccupancyNormalizedChannelAttention):
                        module.channelwise_gate_gain.zero_()
    model = model.to(device).eval()
    seed = int(checkpoint["seed"])

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        batch = torch.stack(
            [
                normalized_rgb_tensor(direct_resize_whole_roi(image, size=IMAGE_SIZE))
                for image in images_bgr
            ]
        ).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        result = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result),
            "PIC model returned invalid progress",
        )
        return result

    method = f"{MODE_METHOD_PREFIXES[attention_mode]}_seed_{seed}"
    _require(checkpoint.get("method") == method, "PIC method identity mismatch")
    return str(checkpoint["protocol"]), method, predict


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
) -> int:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    prediction_protocol, method, predictor = load_checkpoint_predictor(
        checkpoint_path, device_name=device_name
    )
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            images: list[np.ndarray] = []
            hashes: list[str] = []
            for condition in selected:
                image, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                images.append(image)
                hashes.append(canonical_roi_pixel_sha256(image))
            try:
                values: list[float | None] = [float(value) for value in predictor(images)]
                _require(len(values) == len(images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None
                        and math.isfinite(value)
                        and 0.0 <= value <= 1.0
                        for value in values
                    ),
                    "prediction outside [0,1]",
                )
                failures: list[str | None] = [None] * len(images)
            except Exception as exc:
                values = [None] * len(images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(images)
            for condition, condition_hash, progress, failure in zip(
                selected, hashes, values, failures, strict=True
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": prediction_protocol,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": condition_hash,
                }
                _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--parent-checkpoint", type=Path, required=True)
    training.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--synthetic-split", type=Path, required=True)
    training.add_argument("--real-manifest", type=Path, required=True)
    training.add_argument("--real-labels", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument(
        "--attention-mode",
        choices=ATTENTION_MODES,
        default="axial",
        help=(
            "legacy is the paired-view CBAM control; axial enables GGAR; "
            "occupancy enables the Occupancy-Normalized CBAM residual"
        ),
    )
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--workers", type=int, default=4)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--conditions", choices=("all", "clean"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = train(
            parent_checkpoint_path=args.parent_checkpoint,
            synthetic_manifest_path=args.synthetic_manifest,
            synthetic_split_path=args.synthetic_split,
            real_manifest_path=args.real_manifest,
            real_labels_path=args.real_labels,
            output_path=args.output,
            seed=args.seed,
            attention_mode=args.attention_mode,
            device_name=args.device,
            workers=args.workers,
        )
    else:
        selected = CONDITIONS if args.conditions == "all" else ("clean",)
        count = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            conditions=selected,
        )
        checkpoint_identity = torch.load(
            Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False
        )
        _require(isinstance(checkpoint_identity, Mapping), "PIC checkpoint is not an object")
        result = {
            "status": "complete",
            "protocol": str(checkpoint_identity["protocol"]),
            "method": str(checkpoint_identity["method"]),
            "attention_mode": str(checkpoint_identity["attention_mode"]),
            "output": str(Path(args.output).resolve()),
            "rows": count,
            "conditions": list(selected),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "ATTENTION_MODES",
    "METHOD_PREFIX",
    "MODE_ARCHITECTURES",
    "MODE_METHOD_PREFIXES",
    "MODE_PROTOCOLS",
    "PROTOCOL",
    "PerspectiveAdaptiveGeoAttnResNet18",
    "PerspectiveAdaptiveSpatialAttention",
    "configure_attention_mode",
    "load_db_gar_state_into_pa_model",
    "perspective_adaptive_objective",
    "run_prediction",
    "train",
]
