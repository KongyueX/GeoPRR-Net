"""Independent pivot-and-direction fallback model trained from SyncG keypoints.

The module intentionally depends only on PyTorch/torchvision and the local
dataset adapter.  It does not load or copy the external GPL VDN source.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

from experiments.vdn_baseline import affine_for_dial, transform_point


PIVOT_DIRECTION_PROTOCOL = "syncg_pivot_direction_fallback_v1"
IMAGENET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
IMAGENET_MEAN = np.asarray(IMAGENET_WEIGHTS.transforms().mean, dtype=np.float32)
IMAGENET_STD = np.asarray(IMAGENET_WEIGHTS.transforms().std, dtype=np.float32)


def normalized_rgb_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR image to torchvision's normalized RGB tensor."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected a three-channel image, got {image_bgr.shape}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(array)


def _jittered_affine(
    bbox: Sequence[float],
    *,
    output_size: int,
    expansion: float,
    scale: float,
    rotation_degrees: float,
    translation_x: float,
    translation_y: float,
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox[:4])
    side = max(x2 - x1, y2 - y1) * float(expansion) * float(scale)
    if side <= 0.0:
        raise ValueError(f"invalid dial bbox: {bbox}")
    center_x = (x1 + x2) * 0.5 + float(translation_x) * side
    center_y = (y1 + y2) * 0.5 + float(translation_y) * side
    matrix = cv2.getRotationMatrix2D(
        (center_x, center_y),
        float(rotation_degrees),
        float(output_size) / side,
    )
    matrix[0, 2] += output_size * 0.5 - center_x
    matrix[1, 2] += output_size * 0.5 - center_y
    return matrix.astype(np.float32)


def _gaussian_heatmap(
    point_xy: Sequence[float],
    *,
    image_size: int,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    stride = float(image_size) / float(heatmap_size)
    center_x = float(point_xy[0]) / stride
    center_y = float(point_xy[1]) / stride
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    heatmap = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2)
        / (2.0 * float(sigma) ** 2)
    ).astype(np.float32)
    return torch.from_numpy(heatmap[None, ...])


def _photometric_augmentation(image: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32)
    if random.random() < 0.80:
        contrast = random.uniform(0.60, 1.40)
        brightness = random.uniform(-30.0, 30.0)
        output = output * contrast + brightness
    output = np.clip(output, 0.0, 255.0)
    if random.random() < 0.55:
        gamma = math.exp(random.uniform(math.log(0.45), math.log(1.80)))
        output = 255.0 * np.power(output / 255.0, gamma)
    if random.random() < 0.35:
        sigma = random.uniform(0.25, 2.5)
        kernel = max(3, int(math.ceil(sigma * 4.0)) | 1)
        output = cv2.GaussianBlur(output, (kernel, kernel), sigmaX=sigma)
    if random.random() < 0.25:
        noise = np.random.normal(0.0, random.uniform(1.0, 9.0), output.shape)
        output = output + noise
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


class SyncGPivotDirectionDataset(Dataset):
    """Generate pivot heatmaps and unit pointer vectors from SyncG metadata."""

    def __init__(
        self,
        samples: Sequence[Any],
        *,
        image_size: int = 256,
        heatmap_size: int = 64,
        training: bool,
        expansion: float = 1.25,
        scale_factor: float = 0.10,
        rotation_factor: float = 90.0,
        translation_factor: float = 0.12,
        heatmap_sigma: float = 1.5,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.training = bool(training)
        self.expansion = float(expansion)
        self.scale_factor = float(scale_factor)
        self.rotation_factor = float(rotation_factor)
        self.translation_factor = float(translation_factor)
        self.heatmap_sigma = float(heatmap_sigma)
        if not self.samples:
            raise ValueError("pivot-direction dataset is empty")
        if self.image_size <= 0 or self.heatmap_size <= 0:
            raise ValueError("image and heatmap sizes must be positive")
        if self.image_size % self.heatmap_size != 0:
            raise ValueError("image_size must be divisible by heatmap_size")

    def __len__(self) -> int:
        return len(self.samples)

    def _transform(self, sample: Any) -> np.ndarray:
        if not self.training:
            return affine_for_dial(
                sample.dial_bbox,
                output_size=self.image_size,
                expansion=self.expansion,
            )
        scale = random.uniform(1.0 - self.scale_factor, 1.0 + self.scale_factor)
        rotation = random.uniform(-self.rotation_factor, self.rotation_factor)
        translation_x = random.uniform(-self.translation_factor, self.translation_factor)
        translation_y = random.uniform(-self.translation_factor, self.translation_factor)
        matrix = _jittered_affine(
            sample.dial_bbox,
            output_size=self.image_size,
            expansion=self.expansion,
            scale=scale,
            rotation_degrees=rotation,
            translation_x=translation_x,
            translation_y=translation_y,
        )
        pivot = transform_point(sample.pointer_tail, matrix)
        margin = self.image_size * 0.02
        if not (
            -margin <= float(pivot[0]) < self.image_size + margin
            and -margin <= float(pivot[1]) < self.image_size + margin
        ):
            return affine_for_dial(
                sample.dial_bbox,
                output_size=self.image_size,
                expansion=self.expansion,
            )
        return matrix

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(
            sample.image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            raise ValueError(f"failed to read {sample.image_path}")
        matrix = self._transform(sample)
        crop = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if self.training:
            crop = _photometric_augmentation(crop)
        pivot = transform_point(sample.pointer_tail, matrix)
        tip = transform_point(sample.pointer_tip, matrix)
        direction = np.asarray(tip - pivot, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError(f"{sample.sample_id}: pointer direction collapsed")
        direction /= norm
        heatmap = _gaussian_heatmap(
            pivot,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )
        stride = float(self.image_size) / float(self.heatmap_size)
        pivot_heatmap_xy = torch.tensor(
            [float(pivot[0]) / stride, float(pivot[1]) / stride],
            dtype=torch.float32,
        )
        return (
            normalized_rgb_tensor(crop),
            heatmap,
            torch.from_numpy(direction),
            pivot_heatmap_xy,
            sample.sample_id,
        )


class PivotDirectionNet(nn.Module):
    """ResNet-18 encoder with a pivot heatmap and global vector head."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=IMAGENET_WEIGHTS if imagenet_pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.pivot_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        self.direction_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(256, 2),
        )
        for module in self.pivot_head.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.direction_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        return self.pivot_head(features), self.direction_head(features)


def build_pivot_direction_model(*, imagenet_pretrained: bool) -> PivotDirectionNet:
    return PivotDirectionNet(imagenet_pretrained=imagenet_pretrained)


def decode_pivot_direction(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pivot_logits.ndim != 4 or pivot_logits.shape[1] != 1:
        raise ValueError(f"invalid pivot heatmap shape: {tuple(pivot_logits.shape)}")
    if direction_raw.ndim != 2 or direction_raw.shape[1] != 2:
        raise ValueError(f"invalid direction shape: {tuple(direction_raw.shape)}")
    batch, _, height, width = pivot_logits.shape
    probabilities = torch.sigmoid(pivot_logits[:, 0])
    confidence, index = probabilities.reshape(batch, -1).max(dim=1)
    pivot_y = torch.div(index, width, rounding_mode="floor").float()
    pivot_x = (index % width).float()
    pivot_xy = torch.stack((pivot_x, pivot_y), dim=1)
    norm = torch.linalg.vector_norm(direction_raw, dim=1, keepdim=True)
    valid = (
        torch.isfinite(direction_raw).all(dim=1)
        & torch.isfinite(confidence)
        & (norm[:, 0] > 1e-8)
    )
    direction = direction_raw / torch.clamp(norm, min=1e-8)
    return pivot_xy, direction, confidence, valid


def pivot_direction_loss(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    pivot_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probability = torch.sigmoid(pivot_logits.float())
    target_heatmap = target_heatmap.float()
    pixel_weight = 1.0 + 9.0 * target_heatmap
    pivot_loss = torch.mean(pixel_weight * (probability - target_heatmap) ** 2)
    predicted_direction = F.normalize(direction_raw.float(), dim=1, eps=1e-8)
    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-8)
    cosine_loss = torch.mean(1.0 - torch.sum(predicted_direction * target_direction, dim=1))
    vector_loss = F.smooth_l1_loss(predicted_direction, target_direction)
    direction_loss = cosine_loss + 0.25 * vector_loss
    total = direction_loss + float(pivot_weight) * pivot_loss
    return total, {
        "pivot_loss": pivot_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "cosine_loss": cosine_loss.detach(),
        "vector_loss": vector_loss.detach(),
    }


def tensor_from_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    image_size: int,
    expansion: float = 1.25,
) -> torch.Tensor:
    matrix = affine_for_dial(
        bbox,
        output_size=image_size,
        expansion=expansion,
    )
    crop = cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return normalized_rgb_tensor(crop)


def imagenet_checkpoint_path() -> Path:
    filename = Path(IMAGENET_WEIGHTS.url).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename
