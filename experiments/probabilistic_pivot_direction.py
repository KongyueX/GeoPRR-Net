"""Probabilistic, perspective-equivariant pivot and direction estimation.

This is the v2 direction expert used by the paper experiments.  The released
v1 fallback remains untouched so its signed checkpoints and comparisons stay
reproducible.  V2 adds three pieces that are useful under image degradation:

* a circular angle distribution and a learned angular variance;
* paired photometric/projective views with exactly transformed labels; and
* a differentiable homography-equivariance loss on pivot and direction.

The module only depends on PyTorch/torchvision and OpenCV.  It does not import
or copy the GPL VDN implementation.
"""
from __future__ import annotations

import math
import random
from typing import Any, NamedTuple, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.vdn_baseline import affine_for_dial, transform_point


PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL = (
    "syncg_probabilistic_perspective_equivariant_direction_v1"
)
IMAGENET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


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


def _transform_point_homography(
    point_xy: Sequence[float], homography: np.ndarray
) -> np.ndarray:
    point = np.asarray([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography)
    return transformed[0, 0].astype(np.float32)


def _projected_corners(
    size: int,
    *,
    degrees: float,
    axis: str,
    sign: int,
) -> np.ndarray:
    """Project a square plane after a virtual camera-relative 3-D rotation."""

    points = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    angle = math.radians(float(degrees) * int(sign))
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "yaw":
        rotation = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    elif axis == "pitch":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported perspective axis: {axis}")
    rotated = points @ rotation.T
    camera_distance = 2.75
    projected = camera_distance * rotated[:, :2] / (
        camera_distance + rotated[:, 2:3]
    )
    extent = np.ptp(projected, axis=0)
    usable = (size - 1) * 0.94
    scale = min(
        usable / max(float(extent[0]), 1e-12),
        usable / max(float(extent[1]), 1e-12),
    )
    projected = (projected - np.mean(projected, axis=0, keepdims=True)) * scale
    projected += (size - 1) * 0.5
    return projected.astype(np.float32)


def random_perspective_homography(
    image_size: int,
    *,
    probability: float,
    max_degrees: float,
) -> tuple[np.ndarray, float]:
    if random.random() >= float(probability) or float(max_degrees) <= 0.0:
        return np.eye(3, dtype=np.float32), 0.0
    # A small lower bound ensures that a sampled projective pair carries a
    # meaningful training signal rather than being numerically identical.
    degrees = random.uniform(min(8.0, float(max_degrees)), float(max_degrees))
    axis = "yaw" if random.random() < 0.5 else "pitch"
    sign = 1 if random.random() < 0.5 else -1
    source = np.asarray(
        [
            [0.0, 0.0],
            [image_size - 1.0, 0.0],
            [image_size - 1.0, image_size - 1.0],
            [0.0, image_size - 1.0],
        ],
        dtype=np.float32,
    )
    destination = _projected_corners(
        image_size,
        degrees=degrees,
        axis=axis,
        sign=sign,
    )
    return cv2.getPerspectiveTransform(source, destination).astype(np.float32), degrees


def _photometric_augmentation(
    image: np.ndarray,
    *,
    strong: bool,
    max_blur_sigma: float,
) -> np.ndarray:
    output = image.astype(np.float32)
    if random.random() < 0.85:
        contrast = random.uniform(0.50 if strong else 0.70, 1.50 if strong else 1.30)
        brightness = random.uniform(-40.0 if strong else -24.0, 40.0 if strong else 24.0)
        output = output * contrast + brightness
    output = np.clip(output, 0.0, 255.0)
    if random.random() < (0.65 if strong else 0.35):
        gamma = math.exp(
            random.uniform(
                math.log(0.40 if strong else 0.65),
                math.log(1.95 if strong else 1.45),
            )
        )
        output = 255.0 * np.power(output / 255.0, gamma)
    if random.random() < (0.75 if strong else 0.30):
        sigma = random.uniform(0.15, max(float(max_blur_sigma), 0.15))
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if strong and random.random() < 0.30:
        length = random.choice((3, 5, 7, 9, 11))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / float(length)
        rotation = cv2.getRotationMatrix2D(
            ((length - 1) * 0.5, (length - 1) * 0.5),
            random.uniform(0.0, 180.0),
            1.0,
        )
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        kernel /= max(float(kernel.sum()), 1e-8)
        output = cv2.filter2D(output, -1, kernel)
    if random.random() < (0.40 if strong else 0.20):
        noise_sigma = random.uniform(1.0, 12.0 if strong else 6.0)
        output += np.random.normal(0.0, noise_sigma, output.shape)
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


class SyncGProbabilisticDirectionDataset(Dataset):
    """Return clean targets and, for training, an exact projective pair."""

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
        perspective_probability: float = 0.80,
        max_perspective_degrees: float = 45.0,
        max_blur_sigma: float = 3.0,
        return_crop_affine: bool = False,
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
        self.perspective_probability = float(perspective_probability)
        self.max_perspective_degrees = float(max_perspective_degrees)
        self.max_blur_sigma = float(max_blur_sigma)
        self.return_crop_affine = bool(return_crop_affine)
        if not self.samples:
            raise ValueError("probabilistic direction dataset is empty")
        if self.image_size <= 0 or self.heatmap_size <= 0:
            raise ValueError("image and heatmap sizes must be positive")
        if self.image_size % self.heatmap_size != 0:
            raise ValueError("image_size must be divisible by heatmap_size")
        if not 0.0 <= self.perspective_probability <= 1.0:
            raise ValueError("perspective_probability must be in [0, 1]")

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_transform(self, sample: Any) -> np.ndarray:
        if not self.training:
            return affine_for_dial(
                sample.dial_bbox,
                output_size=self.image_size,
                expansion=self.expansion,
            )
        matrix = _jittered_affine(
            sample.dial_bbox,
            output_size=self.image_size,
            expansion=self.expansion,
            scale=random.uniform(1.0 - self.scale_factor, 1.0 + self.scale_factor),
            rotation_degrees=random.uniform(-self.rotation_factor, self.rotation_factor),
            translation_x=random.uniform(-self.translation_factor, self.translation_factor),
            translation_y=random.uniform(-self.translation_factor, self.translation_factor),
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

    def _targets(
        self, pivot: np.ndarray, tip: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = np.asarray(tip - pivot, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError("pointer direction collapsed")
        direction /= norm
        heatmap = _gaussian_heatmap(
            pivot,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )
        stride = float(self.image_size) / float(self.heatmap_size)
        pivot_heatmap = torch.tensor(
            [float(pivot[0]) / stride, float(pivot[1]) / stride],
            dtype=torch.float32,
        )
        return heatmap, torch.from_numpy(direction), pivot_heatmap

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(
            sample.image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            raise ValueError(f"failed to read {sample.image_path}")
        affine = self._crop_transform(sample)
        crop = cv2.warpAffine(
            image,
            affine,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        pivot = transform_point(sample.pointer_tail, affine).astype(np.float32)
        tip = transform_point(sample.pointer_tip, affine).astype(np.float32)
        heatmap, direction, pivot_heatmap = self._targets(pivot, tip)
        result: dict[str, Any] = {
            "image": normalized_rgb_tensor(
                _photometric_augmentation(
                    crop,
                    strong=False,
                    max_blur_sigma=self.max_blur_sigma,
                )
                if self.training
                else crop
            ),
            "heatmap": heatmap,
            "direction": direction,
            "pivot": pivot_heatmap,
            "sample_id": sample.sample_id,
        }
        if self.return_crop_affine:
            # Optional metadata for models whose auxiliary image-space vectors
            # must undergo the exact same sampled crop transform.  The default
            # remains False so existing PEPD loaders keep their prior contract.
            result["crop_affine"] = torch.from_numpy(affine.copy())
        if not self.training:
            return result

        homography, degrees = random_perspective_homography(
            self.image_size,
            probability=self.perspective_probability,
            max_degrees=self.max_perspective_degrees,
        )
        paired_crop = cv2.warpPerspective(
            crop,
            homography,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        paired_pivot = _transform_point_homography(pivot, homography)
        paired_tip = _transform_point_homography(tip, homography)
        paired_heatmap, paired_direction, paired_pivot_heatmap = self._targets(
            paired_pivot, paired_tip
        )
        result.update(
            {
                "paired_image": normalized_rgb_tensor(
                    _photometric_augmentation(
                        paired_crop,
                        strong=True,
                        max_blur_sigma=self.max_blur_sigma,
                    )
                ),
                "paired_heatmap": paired_heatmap,
                "paired_direction": paired_direction,
                "paired_pivot": paired_pivot_heatmap,
                "homography": torch.from_numpy(homography),
                "perspective_degrees": torch.tensor(degrees, dtype=torch.float32),
            }
        )
        return result


class ProbabilisticPivotDirectionNet(nn.Module):
    """ResNet-18 with pivot, circular-distribution and variance heads."""

    def __init__(
        self,
        *,
        angle_bins: int = 72,
        imagenet_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if int(angle_bins) < 8:
            raise ValueError("angle_bins must be at least 8")
        self.angle_bins = int(angle_bins)
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
        self.direction_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.vector_head = nn.Linear(256, 2)
        self.angle_head = nn.Linear(256, self.angle_bins)
        self.log_variance_head = nn.Linear(256, 1)
        for module in self.pivot_head.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.direction_features.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        for head in (self.vector_head, self.angle_head, self.log_variance_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        # One radian (57.3 degrees) at initialization avoids an overconfident
        # random direction and the corresponding early AMP overflows.  The
        # heteroscedastic NLL learns a smaller variance as the mean converges.
        nn.init.constant_(self.log_variance_head.bias, 0.0)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        pooled = self.direction_features(features)
        return (
            self.pivot_head(features),
            self.vector_head(pooled),
            self.angle_head(pooled),
            self.log_variance_head(pooled),
        )


def build_probabilistic_pivot_direction_model(
    *,
    angle_bins: int,
    imagenet_pretrained: bool,
) -> ProbabilisticPivotDirectionNet:
    return ProbabilisticPivotDirectionNet(
        angle_bins=angle_bins,
        imagenet_pretrained=imagenet_pretrained,
    )


def _angle_centers(
    angle_bins: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.arange(angle_bins, device=device, dtype=dtype) * (
        2.0 * math.pi / float(angle_bins)
    )


def circular_delta(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(first - second), torch.cos(first - second))


class ProbabilisticDirectionPrediction(NamedTuple):
    pivot_xy: torch.Tensor
    direction: torch.Tensor
    pivot_peak: torch.Tensor
    valid: torch.Tensor
    angle_std_degrees: torch.Tensor
    angle_entropy: torch.Tensor
    log_variance: torch.Tensor
    bin_resultant_length: torch.Tensor


def decode_probabilistic_pivot_direction(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
) -> ProbabilisticDirectionPrediction:
    if pivot_logits.ndim != 4 or pivot_logits.shape[1] != 1:
        raise ValueError(f"invalid pivot heatmap shape: {tuple(pivot_logits.shape)}")
    if direction_raw.ndim != 2 or direction_raw.shape[1] != 2:
        raise ValueError(f"invalid direction shape: {tuple(direction_raw.shape)}")
    if angle_logits.ndim != 2 or angle_logits.shape[0] != direction_raw.shape[0]:
        raise ValueError(f"invalid angle logits shape: {tuple(angle_logits.shape)}")
    if log_variance_raw.shape != (direction_raw.shape[0], 1):
        raise ValueError(f"invalid variance shape: {tuple(log_variance_raw.shape)}")
    batch, _, height, width = pivot_logits.shape
    probabilities = torch.sigmoid(pivot_logits[:, 0])
    pivot_peak, index = probabilities.reshape(batch, -1).max(dim=1)
    pivot_y = torch.div(index, width, rounding_mode="floor").float()
    pivot_x = (index % width).float()
    pivot_xy = torch.stack((pivot_x, pivot_y), dim=1)

    raw_norm = torch.linalg.vector_norm(direction_raw, dim=1, keepdim=True)
    raw_direction = direction_raw / torch.clamp(raw_norm, min=1e-8)
    bin_probability = torch.softmax(angle_logits.float(), dim=1)
    centers = _angle_centers(
        angle_logits.shape[1],
        device=angle_logits.device,
        dtype=bin_probability.dtype,
    )
    bin_vector = torch.stack(
        (
            torch.sum(bin_probability * torch.cos(centers), dim=1),
            torch.sum(bin_probability * torch.sin(centers), dim=1),
        ),
        dim=1,
    )
    bin_resultant = torch.linalg.vector_norm(bin_vector, dim=1)
    combined = raw_direction + bin_vector
    combined_norm = torch.linalg.vector_norm(combined, dim=1)
    direction = F.normalize(combined, dim=1, eps=1e-8)
    log_variance = torch.clamp(log_variance_raw[:, 0].float(), min=-9.0, max=2.0)
    angle_std_degrees = torch.exp(0.5 * log_variance) * (180.0 / math.pi)
    entropy = -torch.sum(
        bin_probability * torch.log(torch.clamp(bin_probability, min=1e-12)),
        dim=1,
    ) / math.log(float(angle_logits.shape[1]))
    valid = (
        torch.isfinite(direction).all(dim=1)
        & torch.isfinite(pivot_peak)
        & torch.isfinite(log_variance)
        & (raw_norm[:, 0] > 1e-8)
        & (combined_norm > 1e-8)
    )
    return ProbabilisticDirectionPrediction(
        pivot_xy=pivot_xy,
        direction=direction,
        pivot_peak=pivot_peak,
        valid=valid,
        angle_std_degrees=angle_std_degrees,
        angle_entropy=entropy,
        log_variance=log_variance,
        bin_resultant_length=bin_resultant,
    )


def soft_pivot_coordinates(pivot_logits: torch.Tensor) -> torch.Tensor:
    if pivot_logits.ndim != 4 or pivot_logits.shape[1] != 1:
        raise ValueError(f"invalid pivot heatmap shape: {tuple(pivot_logits.shape)}")
    batch, _, height, width = pivot_logits.shape
    probability = torch.softmax(pivot_logits[:, 0].float().reshape(batch, -1), dim=1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=pivot_logits.device, dtype=probability.dtype),
        torch.arange(width, device=pivot_logits.device, dtype=probability.dtype),
        indexing="ij",
    )
    return torch.stack(
        (
            torch.sum(probability * xx.reshape(1, -1), dim=1),
            torch.sum(probability * yy.reshape(1, -1), dim=1),
        ),
        dim=1,
    )


def transform_pivot_direction(
    pivot_input_xy: torch.Tensor,
    direction: torch.Tensor,
    homography: torch.Tensor,
    *,
    ray_length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiably transform a pivot and local pointer ray."""

    if pivot_input_xy.ndim != 2 or pivot_input_xy.shape[1] != 2:
        raise ValueError("pivot_input_xy must have shape [B, 2]")
    if direction.shape != pivot_input_xy.shape:
        raise ValueError("direction shape must match pivot shape")
    if homography.shape != (pivot_input_xy.shape[0], 3, 3):
        raise ValueError("homography must have shape [B, 3, 3]")
    one = torch.ones_like(pivot_input_xy[:, :1])
    pivot_h = torch.cat((pivot_input_xy, one), dim=1)
    tip_h = torch.cat((pivot_input_xy + direction * float(ray_length), one), dim=1)
    transformed_pivot_h = torch.bmm(homography.float(), pivot_h.unsqueeze(2))[:, :, 0]
    transformed_tip_h = torch.bmm(homography.float(), tip_h.unsqueeze(2))[:, :, 0]
    pivot_denominator = transformed_pivot_h[:, 2:3]
    tip_denominator = transformed_tip_h[:, 2:3]
    valid = (
        torch.isfinite(transformed_pivot_h).all(dim=1)
        & torch.isfinite(transformed_tip_h).all(dim=1)
        & (torch.abs(pivot_denominator[:, 0]) > 1e-6)
        & (torch.abs(tip_denominator[:, 0]) > 1e-6)
    )
    safe_pivot_denominator = torch.where(
        torch.abs(pivot_denominator) > 1e-6,
        pivot_denominator,
        torch.where(
            pivot_denominator < 0.0,
            torch.full_like(pivot_denominator, -1e-6),
            torch.full_like(pivot_denominator, 1e-6),
        ),
    )
    safe_tip_denominator = torch.where(
        torch.abs(tip_denominator) > 1e-6,
        tip_denominator,
        torch.where(
            tip_denominator < 0.0,
            torch.full_like(tip_denominator, -1e-6),
            torch.full_like(tip_denominator, 1e-6),
        ),
    )
    transformed_pivot = transformed_pivot_h[:, :2] / safe_pivot_denominator
    transformed_tip = transformed_tip_h[:, :2] / safe_tip_denominator
    transformed_direction = F.normalize(
        transformed_tip - transformed_pivot,
        dim=1,
        eps=1e-8,
    )
    valid &= torch.isfinite(transformed_direction).all(dim=1)
    return transformed_pivot, transformed_direction, valid


def _soft_circular_targets(
    target_direction: torch.Tensor,
    *,
    angle_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    target_angle = torch.atan2(target_direction[:, 1], target_direction[:, 0])
    centers = _angle_centers(
        angle_bins,
        device=target_direction.device,
        dtype=target_direction.dtype,
    )
    delta = circular_delta(centers[None, :], target_angle[:, None])
    sigma = max(float(sigma_bins), 1e-3) * (2.0 * math.pi / float(angle_bins))
    target = torch.exp(-0.5 * (delta / sigma) ** 2)
    return target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)


def probabilistic_direction_loss(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    pivot_weight: float,
    bin_weight: float,
    vector_weight: float,
    soft_target_sigma_bins: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probability = torch.sigmoid(pivot_logits.float())
    target_heatmap = target_heatmap.float()
    pixel_weight = 1.0 + 9.0 * target_heatmap
    pivot_loss = torch.mean(pixel_weight * (probability - target_heatmap) ** 2)

    prediction = decode_probabilistic_pivot_direction(
        pivot_logits.float(),
        direction_raw.float(),
        angle_logits.float(),
        log_variance_raw.float(),
    )
    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-8)
    predicted_angle = torch.atan2(prediction.direction[:, 1], prediction.direction[:, 0])
    target_angle = torch.atan2(target_direction[:, 1], target_direction[:, 0])
    delta = circular_delta(predicted_angle, target_angle)
    angular_nll = 0.5 * (
        delta.square() * torch.exp(-prediction.log_variance)
        + prediction.log_variance
    )
    angular_nll_loss = torch.mean(angular_nll)
    soft_target = _soft_circular_targets(
        target_direction,
        angle_bins=angle_logits.shape[1],
        sigma_bins=soft_target_sigma_bins,
    )
    bin_loss = torch.mean(
        torch.sum(-soft_target * F.log_softmax(angle_logits.float(), dim=1), dim=1)
    )
    cosine_loss = torch.mean(
        1.0 - torch.sum(prediction.direction * target_direction, dim=1)
    )
    direction_loss = (
        angular_nll_loss
        + float(bin_weight) * bin_loss
        + float(vector_weight) * cosine_loss
    )
    total = float(pivot_weight) * pivot_loss + direction_loss
    return total, {
        "pivot_loss": pivot_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "angular_nll_loss": angular_nll_loss.detach(),
        "bin_loss": bin_loss.detach(),
        "cosine_loss": cosine_loss.detach(),
        "mean_angle_std_degrees": prediction.angle_std_degrees.mean().detach(),
    }


def equivariance_loss(
    first: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    homography: torch.Tensor,
    *,
    image_size: int,
    heatmap_size: int,
    pivot_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    first_prediction = decode_probabilistic_pivot_direction(*first)
    second_prediction = decode_probabilistic_pivot_direction(*second)
    stride = float(image_size) / float(heatmap_size)
    first_pivot = soft_pivot_coordinates(first[0]) * stride
    second_pivot = soft_pivot_coordinates(second[0]) * stride
    expected_pivot, expected_direction, valid_h = transform_pivot_direction(
        first_pivot,
        first_prediction.direction,
        homography,
        ray_length=float(image_size) * 0.25,
    )
    valid = valid_h & first_prediction.valid & second_prediction.valid
    if bool(valid.any()):
        pivot_consistency = F.smooth_l1_loss(
            second_pivot[valid] / float(image_size),
            expected_pivot[valid] / float(image_size),
        )
        direction_consistency = torch.mean(
            1.0
            - torch.sum(
                second_prediction.direction[valid] * expected_direction[valid],
                dim=1,
            )
        )
    else:
        # Keep the result attached to the graph while contributing zero.
        pivot_consistency = first[0].sum() * 0.0
        direction_consistency = first[1].sum() * 0.0
    total = direction_consistency + float(pivot_weight) * pivot_consistency
    return total, {
        "equivariance_pivot_loss": pivot_consistency.detach(),
        "equivariance_direction_loss": direction_consistency.detach(),
        "equivariance_valid_fraction": valid.float().mean().detach(),
    }
