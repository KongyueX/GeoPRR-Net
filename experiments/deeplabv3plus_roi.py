"""DeepLabV3+-ResNet50 pointer segmentation on the canonical meter ROI."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights, resnet50

from experiments.roi_geometry_comparison import canonical_roi_bounds


IMAGENET_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
IMAGENET_MEAN = np.asarray(IMAGENET_WEIGHTS.transforms().mean, dtype=np.float32)
IMAGENET_STD = np.asarray(IMAGENET_WEIGHTS.transforms().std, dtype=np.float32)


def normalized_rgb_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image, got {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    values = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    values = (values - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(values)


def pointer_mask_path(sample: Any) -> Path:
    metadata = sample.metadata
    annotation_text = metadata.get("annotation_path") if isinstance(metadata, dict) else None
    if not annotation_text:
        raise ValueError(f"{sample.sample_id}: annotation_path is missing")
    annotation = Path(str(annotation_text)).resolve()
    root = annotation.parents[2]
    candidate = root / "masks" / str(sample.split) / f"{sample.sample_id}.png"
    if not candidate.is_file():
        raise FileNotFoundError(f"{sample.sample_id}: pointer mask is missing: {candidate}")
    return candidate


def load_pointer_mask(sample: Any) -> np.ndarray:
    mask = cv2.imread(str(pointer_mask_path(sample)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"{sample.sample_id}: pointer mask decode failed")
    # SyncG pointer pixels use the high-valued label (approximately 197);
    # interpolation around the contour can produce neighboring values.
    return (mask >= 64).astype(np.uint8)


def canonical_image_and_mask(sample: Any, *, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(
        str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise ValueError(f"{sample.sample_id}: source image decode failed")
    mask = load_pointer_mask(sample)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"{sample.sample_id}: image and pointer mask shapes differ")
    left, top, right, bottom = canonical_roi_bounds(image.shape, sample.dial_bbox)
    image_roi = image[top:bottom, left:right]
    mask_roi = mask[top:bottom, left:right]
    image_roi = cv2.resize(
        image_roi, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    mask_roi = cv2.resize(
        mask_roi, (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )
    return image_roi, mask_roi


def _geometric_augmentation(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    size = int(image.shape[0])
    angle = random.uniform(-20.0, 20.0)
    scale = random.uniform(0.92, 1.08)
    matrix = cv2.getRotationMatrix2D(
        ((size - 1) * 0.5, (size - 1) * 0.5), angle, scale
    )
    matrix[0, 2] += random.uniform(-0.04, 0.04) * size
    matrix[1, 2] += random.uniform(-0.04, 0.04) * size
    border = tuple(
        int(value)
        for value in np.median(
            np.concatenate((image[0], image[-1], image[:, 0], image[:, -1])),
            axis=0,
        )
    )
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    transformed_mask = cv2.warpAffine(
        mask,
        matrix,
        (size, size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return transformed_image, transformed_mask


def _photometric_augmentation(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    if random.random() < 0.8:
        values = values * random.uniform(0.70, 1.30) + random.uniform(-24.0, 24.0)
    values = np.clip(values, 0.0, 255.0)
    if random.random() < 0.35:
        sigma = random.uniform(0.2, 1.8)
        values = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if random.random() < 0.30:
        values += np.random.normal(0.0, random.uniform(1.0, 6.0), values.shape)
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


class SyncGPointerSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Any],
        *,
        image_size: int = 256,
        training: bool,
    ) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)
        self.training = bool(training)
        if not self.samples:
            raise ValueError("pointer-segmentation dataset is empty")
        if self.image_size < 64:
            raise ValueError("image_size must be at least 64")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, mask = canonical_image_and_mask(sample, image_size=self.image_size)
        if self.training:
            image, mask = _geometric_augmentation(image, mask)
            image = _photometric_augmentation(image)
        target = torch.from_numpy(mask.astype(np.float32)[None, ...])
        return normalized_rgb_tensor(image), target, sample.sample_id


def _norm(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            _norm(output_channels),
            nn.ReLU(inplace=True),
        )


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                _ConvNormAct(input_channels, output_channels, kernel_size=1),
                _ConvNormAct(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=12,
                    dilation=12,
                ),
                _ConvNormAct(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=24,
                    dilation=24,
                ),
                _ConvNormAct(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    padding=36,
                    dilation=36,
                ),
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvNormAct(input_channels, output_channels, kernel_size=1),
        )
        self.project = nn.Sequential(
            _ConvNormAct(output_channels * 5, output_channels, kernel_size=1),
            nn.Dropout(0.1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = features.shape[-2:]
        pooled = F.interpolate(
            self.pool(features), size=size, mode="bilinear", align_corners=False
        )
        values = [branch(features) for branch in self.branches]
        values.append(pooled)
        return self.project(torch.cat(values, dim=1))


class DeepLabV3PlusROI(nn.Module):
    """ResNet-50 encoder, ASPP, and low-level DeepLabV3+ decoder."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet50(
            weights=IMAGENET_WEIGHTS if imagenet_pretrained else None,
            replace_stride_with_dilation=(False, True, True),
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = AtrousSpatialPyramidPooling(2048, 256)
        self.low_projection = _ConvNormAct(256, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            _ConvNormAct(304, 256, kernel_size=3, padding=1),
            _ConvNormAct(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output_size = images.shape[-2:]
        features = self.stem(images)
        low = self.layer1(features)
        features = self.layer2(low)
        features = self.layer3(features)
        features = self.layer4(features)
        context = self.aspp(features)
        context = F.interpolate(
            context, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder(torch.cat((context, self.low_projection(low)), dim=1))
        return F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )


def binary_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("segmentation logits and target shapes differ")
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 40.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight.detach()
    )
    probability = torch.sigmoid(logits)
    intersection = torch.sum(probability * target, dim=(1, 2, 3))
    denominator = torch.sum(probability + target, dim=(1, 2, 3))
    dice_loss = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denominator + 1.0))
    return 0.5 * bce + 0.5 * dice_loss


@torch.no_grad()
def segmentation_batch_metrics(
    logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5
) -> tuple[float, float]:
    prediction = torch.sigmoid(logits) >= threshold
    expected = target >= 0.5
    intersection = (prediction & expected).sum(dim=(1, 2, 3)).float()
    union = (prediction | expected).sum(dim=(1, 2, 3)).float()
    prediction_count = prediction.sum(dim=(1, 2, 3)).float()
    target_count = expected.sum(dim=(1, 2, 3)).float()
    iou = ((intersection + 1.0) / (union + 1.0)).mean()
    dice = ((2.0 * intersection + 1.0) / (prediction_count + target_count + 1.0)).mean()
    return float(iou), float(dice)


__all__ = [
    "DeepLabV3PlusROI",
    "SyncGPointerSegmentationDataset",
    "binary_segmentation_loss",
    "canonical_image_and_mask",
    "load_pointer_mask",
    "normalized_rgb_tensor",
    "pointer_mask_path",
    "segmentation_batch_metrics",
]
