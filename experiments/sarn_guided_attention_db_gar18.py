"""Single-seed SARN-Guided Attention Module continuation for DB-GAR18.

SGAM is a clean-preserving integration of the frozen SARN-v2 runtime geometry
with the eight existing CBAM blocks.  SARN supplies an image-aligned valid-
support mask and a reliable gate.  Each block retains the complete legacy CBAM
path and adds two gated residuals:

* channel logits use support-masked average and maximum descriptors;
* spatial logits use a learned residual conditioned on the support mask.

Every old parameter and BatchNorm statistic remains frozen.  If SARN falls back
(``gate == 0``), SGAM directly returns the legacy CBAM result.  Consequently,
clean and blur inputs on which SARN is a no-op remain exactly the DB-GAR18
parent after continuation, rather than merely being regularized toward it.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments import perspective_adaptive_db_gar18 as matched_pic
from experiments import robustness_degradations
from experiments import support_aware_roi_normalization_v2 as sarn_v2
from experiments.domain_balanced_geoattn_resnet18 import (
    ARCHITECTURE as DB_GAR_ARCHITECTURE,
    DOMAIN_BATCH_SIZE,
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
    load_real_progress_samples,
    strong_real_augmentation,
)
from experiments.geoattn_resnet18_progress import CBAM, GeoAttnResNet18, _apply_homography, _geometry_targets
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
from experiments.v5_shared_roi_comparison_input import canonical_tight_roi_native, direct_resize_whole_roi


PROTOCOL: Final[str] = "sarn_guided_attention_db_gar18_v1"
ARCHITECTURE: Final[str] = "SARN-Guided-Attention-DB-GAR18"
METHOD_PREFIX: Final[str] = "sgam_db_gar18"
CALIBRATION_EPOCHS: Final[int] = 2
CHANNEL_GAIN_LEARNING_RATE: Final[float] = 1e-3
SPATIAL_RESIDUAL_LEARNING_RATE: Final[float] = 3e-4
WEIGHT_DECAY: Final[float] = 1e-4
MASK_MAX_THRESHOLD: Final[float] = 0.5
MIN_EFFECTIVE_SUPPORT_FRACTION: Final[float] = 0.05
FULL_SUPPORT_EPSILON: Final[float] = 1e-6
MASK_DENOMINATOR_EPSILON: Final[float] = 1e-6
SPATIAL_KERNEL_SIZE: Final[int] = 7


class SARNGuidedAttention(nn.Module):
    """Legacy CBAM plus reliable support-conditioned channel/spatial residuals."""

    def __init__(self, legacy: CBAM) -> None:
        super().__init__()
        _require(isinstance(legacy, CBAM), "SGAM requires one legacy CBAM")
        self.channel = legacy.channel
        self.spatial = legacy.spatial
        channels = int(self.channel.shared[0].in_channels)
        self.channel_residual_gain = nn.Parameter(torch.zeros(channels))
        self.spatial_residual = nn.Conv2d(
            3,
            1,
            kernel_size=SPATIAL_KERNEL_SIZE,
            padding=SPATIAL_KERNEL_SIZE // 2,
            bias=False,
        )
        nn.init.zeros_(self.spatial_residual.weight)
        self._support_mask: torch.Tensor | None = None
        self._support_gate: torch.Tensor | None = None

    def set_support_context(self, mask: torch.Tensor, gate: torch.Tensor) -> None:
        self._support_mask = mask
        self._support_gate = gate

    def clear_support_context(self) -> None:
        self._support_mask = None
        self._support_gate = None

    @staticmethod
    def _spatial_descriptors(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.mean(value, dim=1, keepdim=True),
            torch.max(value, dim=1, keepdim=True).values,
        )

    @staticmethod
    def masked_descriptors(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stable masked average/max; callers must provide nonempty masks."""

        _require(value.ndim == 4, "SGAM features must be BCHW")
        _require(
            mask.ndim == 4
            and mask.shape[0] == value.shape[0]
            and mask.shape[1] == 1
            and mask.shape[-2:] == value.shape[-2:],
            "SGAM feature mask shape mismatch",
        )
        denominator = torch.sum(mask, dim=(2, 3), keepdim=True).clamp_min(
            MASK_DENOMINATOR_EPSILON
        )
        average = torch.sum(value * mask, dim=(2, 3), keepdim=True) / denominator
        included = mask >= MASK_MAX_THRESHOLD
        maximum = value.masked_fill(~included, -torch.inf).amax(dim=(2, 3), keepdim=True)
        fallback = F.adaptive_max_pool2d(value, 1)
        maximum = torch.where(torch.isfinite(maximum), maximum, fallback)
        return average, maximum

    def _validated_context(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        mask = self._support_mask
        gate = self._support_gate
        if mask is None and gate is None:
            return None
        _require(mask is not None and gate is not None, "incomplete SGAM support context")
        _require(
            mask.ndim == 4
            and mask.shape[0] == value.shape[0]
            and mask.shape[1] == 1,
            "SGAM support mask must be Bx1xHxW",
        )
        _require(
            gate.ndim == 1 and gate.shape[0] == value.shape[0],
            "SGAM support gate must be B",
        )
        resized = F.interpolate(
            mask.detach().to(device=value.device, dtype=value.dtype),
            size=value.shape[-2:],
            mode="area",
        )
        finite_mask = torch.isfinite(resized).all(dim=(1, 2, 3))
        safe_mask = torch.where(torch.isfinite(resized), resized, torch.ones_like(resized))
        safe_mask = torch.clamp(safe_mask, 0.0, 1.0)
        support_fraction = torch.mean(safe_mask, dim=(1, 2, 3))
        raw_gate = gate.detach().to(device=value.device, dtype=value.dtype)
        finite_gate = torch.isfinite(raw_gate)
        safe_gate = torch.where(finite_gate, torch.clamp(raw_gate, 0.0, 1.0), torch.zeros_like(raw_gate))
        active = (
            finite_mask
            & finite_gate
            & (safe_gate > 0.0)
            & (support_fraction >= MIN_EFFECTIVE_SUPPORT_FRACTION)
            & (support_fraction < 1.0 - FULL_SUPPORT_EPSILON)
        )
        # Invalid/inactive samples use all-support values so the unused candidate
        # path remains finite even in a mixed batch.
        safe_mask = torch.where(active[:, None, None, None], safe_mask, torch.ones_like(safe_mask))
        return safe_mask, safe_gate, active

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        legacy_channel = self.channel(value)
        legacy = self.spatial(legacy_channel)
        context = self._validated_context(value)
        if context is None:
            return legacy
        support, gate, active = context
        if not bool(torch.any(active)):
            return legacy

        global_average_logits = self.channel.shared(F.adaptive_avg_pool2d(value, 1))
        global_maximum_logits = self.channel.shared(F.adaptive_max_pool2d(value, 1))
        legacy_channel_logits = global_average_logits + global_maximum_logits
        support_average, support_maximum = self.masked_descriptors(value, support)
        support_channel_logits = self.channel.shared(support_average) + self.channel.shared(
            support_maximum
        )
        gain = torch.tanh(self.channel_residual_gain).view(1, -1, 1, 1)
        q = gate[:, None, None, None]
        candidate_channel_logits = legacy_channel_logits + q * gain * (
            support_channel_logits - legacy_channel_logits
        )
        candidate_channel = value * torch.sigmoid(candidate_channel_logits)

        spatial_average, spatial_maximum = self._spatial_descriptors(candidate_channel)
        legacy_spatial_logits = self.spatial.conv(
            torch.cat((spatial_average, spatial_maximum), dim=1)
        )
        residual_spatial_logits = self.spatial_residual(
            torch.cat((spatial_average, spatial_maximum, support), dim=1)
        )
        candidate = candidate_channel * torch.sigmoid(
            legacy_spatial_logits + q * residual_spatial_logits
        )
        return torch.where(active[:, None, None, None], candidate, legacy)


class SARNGuidedGeoAttnResNet18(GeoAttnResNet18):
    """DB-GAR18 whose eight CBAMs consume one runtime SARN support context."""

    def __init__(self, *, imagenet_pretrained: bool = False) -> None:
        super().__init__(imagenet_pretrained=imagenet_pretrained)
        replaced = 0
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            for block in getattr(self.backbone, layer_name):
                _require(isinstance(block.attention, CBAM), "unexpected DB-GAR attention block")
                block.attention = SARNGuidedAttention(block.attention)
                replaced += 1
        _require(replaced == 8, "SGAM requires exactly eight attention blocks")

    def sgam_modules(self) -> tuple[SARNGuidedAttention, ...]:
        modules = tuple(
            module for module in self.modules() if isinstance(module, SARNGuidedAttention)
        )
        _require(len(modules) == 8, "SGAM module inventory drifted")
        return modules

    def _features_with_support(
        self,
        image: torch.Tensor,
        support_mask: torch.Tensor | None,
        support_gate: torch.Tensor | None,
    ) -> torch.Tensor:
        if support_mask is None and support_gate is None:
            return self.backbone(image)
        _require(
            support_mask is not None and support_gate is not None,
            "SGAM requires mask and gate together",
        )
        _require(
            support_mask.ndim == 4
            and support_mask.shape[0] == image.shape[0]
            and support_mask.shape[1] == 1
            and support_mask.shape[-2:] == image.shape[-2:],
            "SGAM input mask shape mismatch",
        )
        _require(
            support_gate.ndim == 1 and support_gate.shape[0] == image.shape[0],
            "SGAM input gate shape mismatch",
        )
        modules = self.sgam_modules()
        for module in modules:
            module.set_support_context(support_mask, support_gate)
        try:
            return self.backbone(image)
        finally:
            for module in modules:
                module.clear_support_context()

    def forward(
        self,
        image: torch.Tensor,
        support_mask: torch.Tensor | None = None,
        support_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._features_with_support(image, support_mask, support_gate)
        return torch.sigmoid(self.progress_head(features).squeeze(1))

    def forward_training(
        self,
        image: torch.Tensor,
        support_mask: torch.Tensor | None = None,
        support_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self._features_with_support(image, support_mask, support_gate)
        progress = torch.sigmoid(self.progress_head(features).squeeze(1))
        geometry = self.geometry_head(features)
        return {
            "progress": progress,
            "pivot": torch.sigmoid(geometry[:, 0:2]),
            "direction_sin_cos": F.normalize(geometry[:, 2:4], dim=1, eps=1e-6),
            "references": torch.sigmoid(geometry[:, 4:8]),
        }


def load_db_gar_state_into_sgam_model(
    model: SARNGuidedGeoAttnResNet18,
    state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    _require(not unexpected, f"unexpected DB-GAR parent keys: {unexpected}")
    allowed = (".channel_residual_gain", ".spatial_residual.weight")
    _require(
        len(missing) == 16 and all(key.endswith(allowed) for key in missing),
        f"DB-GAR to SGAM state mismatch: {missing}",
    )
    for module in model.sgam_modules():
        with torch.no_grad():
            module.channel_residual_gain.zero_()
            module.spatial_residual.weight.zero_()
    return missing


def set_sgam_calibration_stage(model: SARNGuidedGeoAttnResNet18) -> tuple[str, ...]:
    """Freeze every legacy value and enable only the two SGAM residual families."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.sgam_modules():
        module.channel_residual_gain.requires_grad_(True)
        module.spatial_residual.weight.requires_grad_(True)
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    _require(len(names) == 16, "SGAM trainable parameter inventory drifted")
    return names


def _transform_points_after_sarn_crop(
    points: np.ndarray | None,
    decision: sarn_v2.SARNv2Result,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    if points is None or not decision.applied:
        return points
    _require(decision.bbox_xyxy is not None, "applied SARN decision lacks crop bbox")
    left, top, right, bottom = decision.bbox_xyxy
    pixel_scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    pixels = np.asarray(points, dtype=np.float32) * pixel_scale
    pixels[:, 0] = (
        (pixels[:, 0] - float(left) + 0.5) * float(width) / float(right - left) - 0.5
    )
    pixels[:, 1] = (
        (pixels[:, 1] - float(top) + 0.5) * float(height) / float(bottom - top) - 0.5
    )
    transformed = np.clip(pixels / pixel_scale, 0.0, 1.0).astype(np.float32)
    _require(bool(np.isfinite(transformed).all()), "SARN-adjusted geometry is non-finite")
    return transformed


def sarn_normalize_projective_training_view(
    image: np.ndarray,
    points: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, torch.Tensor, torch.Tensor, sarn_v2.SARNv2Result]:
    """Apply the deployment SARN contract and align optional geometry labels."""

    decision = sarn_v2.normalize_support_aware_roi_v2(image)
    _require(decision.valid_support_mask is not None, "SARN runtime mask is missing")
    height, width = image.shape[:2]
    transformed = _transform_points_after_sarn_crop(
        points, decision, height=height, width=width
    )
    mask = torch.from_numpy(
        np.ascontiguousarray(decision.valid_support_mask[None, ...], dtype=np.float32)
    )
    gate = torch.tensor(decision.support_gate, dtype=torch.float32)
    return decision.image, transformed, mask, gate, decision


class SGAMPairedSyntheticDataset(Dataset[dict[str, torch.Tensor]]):
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
        _require(bool(self.samples), "SGAM synthetic dataset is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
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
        clean_rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index) * 97)
        roi, forward, _geometry_code = _cagh_augment_geometry(
            roi, points, clean_rng, self.augmentation
        )
        points = _apply_homography(points, forward)
        roi, _photo_code = _cagh_augment_photo(roi, clean_rng, self.augmentation)
        clean_labels = _geometry_targets(points)
        perspective_rng = np.random.default_rng(
            self.seed
            + matched_pic.PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + int(index) * 193
        )
        perspective, perspective_points, _metadata = matched_pic._paired_projective_view(
            roi,
            points,
            rng=perspective_rng,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
        )
        _require(perspective_points is not None, "synthetic projective labels disappeared")
        normalized, perspective_points, mask, gate, _decision = (
            sarn_normalize_projective_training_view(perspective, perspective_points)
        )
        _require(perspective_points is not None, "SARN-adjusted labels disappeared")
        perspective_labels = _geometry_targets(perspective_points)
        return {
            "clean_image": normalized_rgb_tensor(roi),
            "perspective_image": normalized_rgb_tensor(normalized),
            "perspective_support_mask": mask,
            "perspective_support_gate": gate,
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "clean_pivot": clean_labels["pivot"],
            "clean_direction_sin_cos": clean_labels["direction_sin_cos"],
            "clean_references": clean_labels["references"],
            "perspective_pivot": perspective_labels["pivot"],
            "perspective_direction_sin_cos": perspective_labels["direction_sin_cos"],
            "perspective_references": perspective_labels["references"],
        }


class SGAMPairedRealDataset(Dataset[dict[str, torch.Tensor]]):
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
        _require(bool(self.samples), "SGAM real dataset is empty")
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
            + matched_pic.PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + source_index * 193
            + draw_ordinal * 15_869
        )
        perspective, _points, _metadata = matched_pic._paired_projective_view(
            image,
            None,
            rng=perspective_rng,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
        )
        normalized, _points, mask, gate, _decision = sarn_normalize_projective_training_view(
            perspective, None
        )
        return {
            "clean_image": normalized_rgb_tensor(image),
            "perspective_image": normalized_rgb_tensor(normalized),
            "perspective_support_mask": mask,
            "perspective_support_gate": gate,
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
        }


def _slice_outputs(
    outputs: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in outputs.items()}


def _train_epoch(
    model: SARNGuidedGeoAttnResNet18,
    synthetic_loader: DataLoader,
    real_loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    _require(len(synthetic_loader) == len(real_loader), "domain loader lengths differ")
    model.train()
    # Frozen BatchNorm buffers are part of the parent identity and must not move.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
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
    gate_sum = 0.0
    gate_active = 0
    gate_count = 0
    steps = 0
    for synthetic_raw, real_raw in zip(synthetic_loader, real_loader, strict=True):
        synthetic = {key: value.to(device, non_blocking=use_amp) for key, value in synthetic_raw.items()}
        real = {key: value.to(device, non_blocking=use_amp) for key, value in real_raw.items()}
        count = int(synthetic["progress"].numel())
        _require(
            count == int(real["progress"].numel()) == DOMAIN_BATCH_SIZE,
            "paired domain batch is not balanced",
        )
        clean_mask = torch.ones(
            count,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=device,
            dtype=synthetic["clean_image"].dtype,
        )
        clean_gate = torch.zeros(count, device=device, dtype=torch.float32)
        images = torch.cat(
            (
                synthetic["clean_image"],
                synthetic["perspective_image"],
                real["clean_image"],
                real["perspective_image"],
            ),
            dim=0,
        )
        masks = torch.cat(
            (
                clean_mask,
                synthetic["perspective_support_mask"],
                clean_mask,
                real["perspective_support_mask"],
            ),
            dim=0,
        )
        gates = torch.cat(
            (
                clean_gate,
                synthetic["perspective_support_gate"],
                clean_gate,
                real["perspective_support_gate"],
            ),
            dim=0,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model.forward_training(images, masks, gates)
            synthetic_clean = _slice_outputs(outputs, 0, count)
            synthetic_perspective = _slice_outputs(outputs, count, 2 * count)
            real_clean = _slice_outputs(outputs, 2 * count, 3 * count)
            real_perspective = _slice_outputs(outputs, 3 * count, 4 * count)
            loss, components = matched_pic.perspective_adaptive_objective(
                synthetic_clean,
                synthetic_perspective,
                synthetic,
                real_clean,
                real_perspective,
                real,
            )
        _require(bool(torch.isfinite(loss)), "SGAM loss became non-finite")
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
        perspective_gates = torch.cat(
            (synthetic["perspective_support_gate"], real["perspective_support_gate"])
        )
        gate_sum += float(perspective_gates.sum().detach().cpu())
        gate_active += int(torch.count_nonzero(perspective_gates > 0.0).detach().cpu())
        gate_count += int(perspective_gates.numel())
        steps += 1
    _require(steps > 0 and gate_count > 0, "SGAM epoch is empty")
    return {
        **{key: value / steps for key, value in totals.items()},
        "steps": float(steps),
        "perspective_sarn_apply_fraction": float(gate_active / gate_count),
        "perspective_sarn_mean_gate": float(gate_sum / gate_count),
    }


def _load_db_gar_parent(
    checkpoint_path: Path,
) -> tuple[SARNGuidedGeoAttnResNet18, Mapping[str, Any], tuple[str, ...]]:
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
    _require(isinstance(state, Mapping), "DB-GAR18 parent state is missing")
    model = SARNGuidedGeoAttnResNet18(imagenet_pretrained=False)
    missing = load_db_gar_state_into_sgam_model(model, state)
    return model, checkpoint, missing


def _parameter_inventory(model: SARNGuidedGeoAttnResNet18) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    channel = sum(module.channel_residual_gain.numel() for module in model.sgam_modules())
    spatial = sum(module.spatial_residual.weight.numel() for module in model.sgam_modules())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "channel_residual_gain": int(channel),
        "spatial_support_residual": int(spatial),
        "candidate_total": int(channel + spatial),
    }


def train(
    *,
    parent_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    real_manifest_path: Path,
    real_labels_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    _require(workers >= 0, "workers must be non-negative")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite SGAM checkpoint: {output}")
    synthetic_samples, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    real_samples = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "SGAM requires the frozen 14,442/1,558 SyncG roster",
    )
    _require(
        len(real_samples) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_samples}) == EXPECTED_REAL_GROUPS,
        "SGAM requires the frozen 434-row/11-group real-development roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent, missing = _load_db_gar_parent(parent_checkpoint_path)
    _require(int(parent["seed"]) == int(seed), "parent and continuation seeds differ")
    model = model.to(device)
    trainable_names = set_sgam_calibration_stage(model)
    synthetic_dataset = SGAMPairedSyntheticDataset(
        synthetic_samples, seed=seed, total_epochs=CALIBRATION_EPOCHS
    )
    real_dataset = SGAMPairedRealDataset(
        real_samples, seed=seed + 10_000, total_epochs=CALIBRATION_EPOCHS
    )
    padded_samples = int(math.ceil(len(synthetic_dataset) / DOMAIN_BATCH_SIZE) * DOMAIN_BATCH_SIZE)
    channel_parameters = [module.channel_residual_gain for module in model.sgam_modules()]
    spatial_parameters = [module.spatial_residual.weight for module in model.sgam_modules()]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": channel_parameters,
                "lr": CHANNEL_GAIN_LEARNING_RATE,
                "weight_decay": 0.0,
                "name": "channel_gain",
            },
            {
                "params": spatial_parameters,
                "lr": SPATIAL_RESIDUAL_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "name": "spatial_residual",
            },
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(CALIBRATION_EPOCHS):
        synthetic_dataset.set_epoch(epoch)
        real_dataset.set_epoch(epoch)
        synthetic_sampler = ShufflePadSampler(
            len(synthetic_dataset), batch_size=DOMAIN_BATCH_SIZE, seed=seed + epoch
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
            persistent_workers=False,
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
            "learning_rates": {
                str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
            },
            "curriculum_max_degrees": matched_pic.PERSPECTIVE_DEGREES_MIN
            + float(epoch + 1)
            / float(CALIBRATION_EPOCHS)
            * (matched_pic.PERSPECTIVE_DEGREES_MAX - matched_pic.PERSPECTIVE_DEGREES_MIN),
            "train": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    parent_source = Path(parent_checkpoint_path).resolve()
    implementation_source = Path(__file__).resolve()
    sarn_source = Path(sarn_v2.__file__).resolve()
    method = f"{METHOD_PREFIX}_seed_{seed}"
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "method": method,
        "seed": int(seed),
        "screen_scope": "single_seed_go_no_go_only",
        "image_size": IMAGE_SIZE,
        "parent_checkpoint": str(parent_source),
        "parent_checkpoint_sha256": _sha256_file(parent_source),
        "parent_protocol": str(parent["protocol"]),
        "parent_architecture": str(parent["architecture"]),
        "parent_missing_keys_initialized": list(missing),
        "source_files": {
            "implementation": {
                "path": str(implementation_source),
                "sha256": _sha256_file(implementation_source),
            },
            "sarn_v2_runtime": {
                "path": str(sarn_source),
                "sha256": _sha256_file(sarn_source),
            },
        },
        "training_inputs": {
            "synthetic_manifest": str(Path(synthetic_manifest_path).resolve()),
            "synthetic_manifest_sha256": _sha256_file(Path(synthetic_manifest_path).resolve()),
            "synthetic_split": str(Path(synthetic_split_path).resolve()),
            "synthetic_split_sha256": _sha256_file(Path(synthetic_split_path).resolve()),
            "synthetic_fit_samples": len(synthetic_samples),
            "synthetic_holdout_samples": len(roster.validation_ids),
            "synthetic_fit_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
            "synthetic_holdout_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
            "real_manifest": str(Path(real_manifest_path).resolve()),
            "real_manifest_sha256": _sha256_file(Path(real_manifest_path).resolve()),
            "real_labels": str(Path(real_labels_path).resolve()),
            "real_labels_sha256": _sha256_file(Path(real_labels_path).resolve()),
            "real_development_samples": len(real_samples),
            "real_groups": len({sample.group_id for sample in real_samples}),
        },
        "candidate": {
            "blocks": 8,
            "runtime_inputs": ["SARN-normalized pixels", "valid-support mask", "support gate"],
            "channel": "masked avg+max shared-MLP logit residual",
            "spatial": "Conv2d([channel-avg,channel-max,support],3->1,k7) residual",
            "gate": "SARN confidence when applied, otherwise exactly zero",
            "fallback": "direct legacy CBAM path when gate is zero or support invalid/full",
            "mask_downsample": "area interpolation to each feature shape",
            "masked_max_threshold": MASK_MAX_THRESHOLD,
        },
        "schedule": {
            "checkpoint_selection": "terminal_fixed_epoch",
            "epochs": CALIBRATION_EPOCHS,
            "old_parameters": "all frozen",
            "batchnorm_running_statistics": "frozen eval mode",
            "learning_rates": {
                "channel_gain": CHANNEL_GAIN_LEARNING_RATE,
                "spatial_residual": SPATIAL_RESIDUAL_LEARNING_RATE,
            },
            "trainable_parameter_names": list(trainable_names),
            "domain_batch_size_each": DOMAIN_BATCH_SIZE,
        },
        "perspective_training": {
            "family": robustness_degradations.ROBUSTNESS_PROTOCOL,
            "degrees_min": matched_pic.PERSPECTIVE_DEGREES_MIN,
            "degrees_max": matched_pic.PERSPECTIVE_DEGREES_MAX,
            "actual_sarn_runtime_used": True,
            "oracle_support_used": False,
            "evaluation_robustness_seed_used": False,
        },
        "loss": "frozen matched perspective-adaptive DB-GAR18 task objective",
        "parameter_inventory": _parameter_inventory(model),
        "training_seconds": float(time.perf_counter() - started),
        "history": history,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "method": method,
        "checkpoint": str(output),
        "epochs": CALIBRATION_EPOCHS,
        "training_seconds": checkpoint["training_seconds"],
    }


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[str, SARNGuidedGeoAttnResNet18, Mapping[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"SGAM checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "SGAM checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "SGAM protocol mismatch")
    _require(checkpoint.get("architecture") == ARCHITECTURE, "SGAM architecture mismatch")
    _require(checkpoint.get("image_size") == IMAGE_SIZE, "SGAM image-size mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "SGAM model state is missing")
    seed = int(checkpoint["seed"])
    method = f"{METHOD_PREFIX}_seed_{seed}"
    _require(checkpoint.get("method") == method, "SGAM method identity mismatch")
    source_files = checkpoint.get("source_files")
    _require(isinstance(source_files, Mapping), "SGAM source-file binding is missing")
    expected_sources = {
        "implementation": Path(__file__).resolve(),
        "sarn_v2_runtime": Path(sarn_v2.__file__).resolve(),
    }
    for name, expected_path in expected_sources.items():
        binding = source_files.get(name)
        _require(isinstance(binding, Mapping), f"SGAM {name} source binding is missing")
        _require(
            Path(str(binding.get("path", ""))).resolve() == expected_path,
            f"SGAM {name} source path mismatch",
        )
        _require(
            binding.get("sha256") == _sha256_file(expected_path),
            f"SGAM {name} source hash drift",
        )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = SARNGuidedGeoAttnResNet18(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    return method, model.to(device).eval(), checkpoint


def _model_batch(
    model: SARNGuidedGeoAttnResNet18,
    decisions: Sequence[sarn_v2.SARNv2Result],
    *,
    device: torch.device,
) -> list[float]:
    _require(bool(decisions), "SGAM prediction batch is empty")
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    gates: list[float] = []
    for decision in decisions:
        _require(decision.valid_support_mask is not None, "SGAM decision mask is missing")
        resized_image = direct_resize_whole_roi(decision.image, size=IMAGE_SIZE)
        resized_mask = cv2.resize(
            decision.valid_support_mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32, copy=False)
        images.append(normalized_rgb_tensor(resized_image))
        masks.append(torch.from_numpy(np.ascontiguousarray(resized_mask[None, ...])))
        gates.append(float(decision.support_gate))
    image_batch = torch.stack(images).to(device)
    mask_batch = torch.stack(masks).to(device)
    gate_batch = torch.tensor(gates, dtype=torch.float32, device=device)
    with torch.inference_mode():
        values = model(image_batch, mask_batch, gate_batch).detach().cpu().tolist()
    result = [float(value) for value in values]
    _require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result),
        "SGAM returned invalid progress",
    )
    return result


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid SGAM evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    method, model, _checkpoint = load_checkpoint_model(
        checkpoint_path, device_name=device_name
    )
    device = torch.device(device_name)
    output = Path(output_path).resolve()
    summary = (
        Path(summary_path).resolve()
        if summary_path is not None
        else output.with_name(f"{output.stem}.summary.json")
    )
    _require(output != summary, "SGAM output and summary paths must differ")
    _require(not output.exists(), f"refusing to overwrite SGAM predictions: {output}")
    _require(not summary.exists(), f"refusing to overwrite SGAM summary: {summary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        condition: {"rows": 0, "applied": 0, "fallback": 0, "confidence_sum": 0.0}
        for condition in selected
    }
    row_count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            decisions: list[sarn_v2.SARNv2Result] = []
            hashes: list[str] = []
            for condition in selected:
                degraded, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                degraded = np.ascontiguousarray(degraded)
                hashes.append(canonical_roi_pixel_sha256(degraded))
                decisions.append(sarn_v2.normalize_support_aware_roi_v2(degraded))
            try:
                values: list[float | None] = _model_batch(model, decisions, device=device)
                failures: list[str | None] = [None] * len(values)
            except Exception as exc:
                values = [None] * len(decisions)
                failures = [f"model_exception:{type(exc).__name__}"] * len(decisions)
            for condition, condition_hash, decision, progress, failure in zip(
                selected, hashes, decisions, values, failures, strict=True
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
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
                _require(set(row) == set(OUTPUT_KEYS), "SGAM prediction schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                cell = counts[condition]
                cell["rows"] += 1
                cell["applied" if decision.applied else "fallback"] += 1
                cell["confidence_sum"] += float(decision.support_gate)
                row_count += 1
    condition_summary = {
        condition: {
            "rows": int(value["rows"]),
            "applied": int(value["applied"]),
            "fallback": int(value["fallback"]),
            "mean_support_gate": float(value["confidence_sum"] / max(value["rows"], 1)),
        }
        for condition, value in counts.items()
    }
    summary_value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "method": method,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _sha256_file(Path(checkpoint_path).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
        "conditions": list(selected),
        "rows": row_count,
        "condition_summary": condition_summary,
        "runtime_boundary": {
            "normalizer_inputs": ["uint8_bgr_pixels"],
            "condition_available_to_normalizer": False,
            "homography_available_to_normalizer": False,
            "label_available_to_normalizer": False,
            "model_inputs": ["normalized pixels", "valid-support mask", "support gate"],
        },
        "artifact": {"path": str(output), "sha256": _sha256_file(output)},
    }
    summary.write_bytes(_canonical_json_bytes(summary_value))
    return summary_value


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
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--workers", type=int, default=4)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--summary", type=Path)
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
            device_name=args.device,
            workers=args.workers,
        )
    else:
        selected = CONDITIONS if args.conditions == "all" else ("clean",)
        value = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            summary_path=args.summary,
            device_name=args.device,
            conditions=selected,
        )
        result = {
            "status": "complete",
            "protocol": PROTOCOL,
            "method": value["method"],
            "rows": value["rows"],
            "output": str(Path(args.output).resolve()),
            "output_sha256": _sha256_file(Path(args.output).resolve()),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "METHOD_PREFIX",
    "PROTOCOL",
    "SARNGuidedAttention",
    "SARNGuidedGeoAttnResNet18",
    "load_checkpoint_model",
    "load_db_gar_state_into_sgam_model",
    "run_prediction",
    "sarn_normalize_projective_training_view",
    "set_sgam_calibration_stage",
    "train",
]
