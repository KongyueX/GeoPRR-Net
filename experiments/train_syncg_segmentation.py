"""Fine-tune the pointer U2NetP using only the official SyncG train split.

The script deliberately creates a validation split from ``annotations/train``
with disjoint ``gauge_type::scene_name`` groups.  SyncG's official test split
is never opened here, so early stopping and mask-threshold selection cannot
leak test information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from experiments.datasets import (
    SYNCG_PINNED_COMMIT,
    SYNCG_SAMPLE_IDS_SHA256,
    SYNCG_TEST_ROWS,
    SYNCG_TRAIN_ROWS,
    syncg_sample_ids_sha256,
)
from utils.angleDetect.pointerSeg.detectSeg import load_u2net_state_dict
from utils.angleDetect.pointerSeg.u2netp import U2NETP


INPUT_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SyncGSegSample:
    sample_id: str
    group_id: str
    gauge_type: str
    scene_name: str
    image_path: str
    mask_path: str
    annotation_path: str
    dial_bbox: tuple[float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _find_syncg_root(root: Path, split: str = "train") -> Path:
    for candidate in (root, root / "syncG", root / "SyncG"):
        if (
            (candidate / "annotations" / split).is_dir()
            and (candidate / "images" / split).is_dir()
            and (candidate / "masks" / split).is_dir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot find SyncG {split} folders below {root}; expected "
        f"annotations/{split}, images/{split}, and masks/{split}"
    )


def _media_index(root: Path, suffixes: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        resolved = path.resolve()
        relative = path.relative_to(root).as_posix().lower()
        for key in (path.name.lower(), path.stem.lower(), relative):
            index.setdefault(key, resolved)
    return index


def _resolve_media(
    value: Any,
    fallback_stem: str,
    index: dict[str, Path],
    *,
    kind: str,
) -> Path:
    raw = Path(str(value or fallback_stem))
    keys = (
        str(value or "").replace("\\", "/").lower(),
        raw.name.lower(),
        raw.stem.lower(),
        fallback_stem.lower(),
    )
    for key in keys:
        if key and key in index:
            return index[key]
    raise FileNotFoundError(
        f"cannot resolve {kind} for {value!r} (fallback {fallback_stem!r})"
    )


def _parse_bbox(value: Any, annotation_path: Path) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(
            f"{annotation_path}: dial_bbox_annotations must contain four values"
        )
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{annotation_path}: invalid dial bounding box {value!r}") from exc
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{annotation_path}: non-finite dial bounding box {bbox!r}")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{annotation_path}: empty dial bounding box {bbox!r}")
    return bbox


def discover_syncg_seg_samples(
    root: Path,
    *,
    split: str,
    limit: int | None = None,
    strict_release: bool = True,
) -> tuple[Path, list[SyncGSegSample]]:
    """Resolve one official split's paired images, pointer masks, and groups."""
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported SyncG split: {split!r}")
    root = _find_syncg_root(root, split)
    annotation_dir = root / "annotations" / split
    image_index = _media_index(root / "images" / split, IMAGE_SUFFIXES)
    mask_index = _media_index(root / "masks" / split, MASK_SUFFIXES)
    annotation_paths = sorted(annotation_dir.glob("*.json"))

    expected_rows = (
        SYNCG_TRAIN_ROWS if split == "train" else SYNCG_TEST_ROWS
    )
    if strict_release and limit is None and len(annotation_paths) != expected_rows:
        raise ValueError(
            f"SyncG {split} has {len(annotation_paths)} annotations; "
            f"expected {expected_rows}. Use --allow-dataset-drift "
            "only for diagnostics."
        )
    release_ids_sha256 = syncg_sample_ids_sha256(
        path.stem for path in annotation_paths
    )
    expected_ids_sha256 = SYNCG_SAMPLE_IDS_SHA256[split]
    if (
        strict_release
        and limit is None
        and release_ids_sha256 != expected_ids_sha256
    ):
        raise ValueError(
            f"SyncG {split} sample identifiers do not match pinned release "
            f"{SYNCG_PINNED_COMMIT}: expected {expected_ids_sha256}, "
            f"got {release_ids_sha256}. Use --allow-dataset-drift only "
            "for diagnostics."
        )
    if limit is not None:
        annotation_paths = annotation_paths[: max(0, limit)]

    samples: list[SyncGSegSample] = []
    for annotation_path in annotation_paths:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path} is not a JSON object")
        file_name = annotation.get("file_name") or annotation_path.stem
        image_path = _resolve_media(
            file_name,
            annotation_path.stem,
            image_index,
            kind="image",
        )
        mask_path = _resolve_media(
            file_name,
            annotation_path.stem,
            mask_index,
            kind="pointer mask",
        )
        gauge_type = str(annotation.get("gauge_type") or "unknown_type")
        scene_name = str(annotation.get("scene_name") or "unknown_scene")
        group_id = f"{gauge_type}::{Path(scene_name).stem}"
        samples.append(
            SyncGSegSample(
                sample_id=annotation_path.stem,
                group_id=group_id,
                gauge_type=gauge_type,
                scene_name=scene_name,
                image_path=str(image_path),
                mask_path=str(mask_path),
                annotation_path=str(annotation_path.resolve()),
                dial_bbox=_parse_bbox(
                    annotation.get("dial_bbox_annotations"),
                    annotation_path,
                ),
            )
        )

    if not samples:
        raise ValueError(f"no SyncG {split} samples were discovered")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError(f"SyncG {split} contains duplicate sample identifiers")
    return root, samples


def discover_syncg_train_samples(
    root: Path,
    *,
    limit: int | None = None,
    strict_release: bool = True,
) -> tuple[Path, list[SyncGSegSample]]:
    """Resolve train pairs without exposing a test-split option to training."""
    return discover_syncg_seg_samples(
        root,
        split="train",
        limit=limit,
        strict_release=strict_release,
    )


def grouped_train_val_split(
    samples: Sequence[SyncGSegSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[SyncGSegSample], list[SyncGSegSample]]:
    """Create a deterministic split while keeping every group on one side."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    by_group: dict[str, list[SyncGSegSample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_id].append(sample)
    if len(by_group) < 2:
        raise ValueError("at least two gauge/scene groups are required")

    def group_order(group_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()

    ordered_groups = sorted(by_group, key=group_order)
    target = max(1, round(len(samples) * val_fraction))
    val_groups: set[str] = set()
    val_count = 0
    for group_id in ordered_groups:
        remaining_groups = len(ordered_groups) - len(val_groups)
        if val_count >= target and val_groups:
            break
        if remaining_groups <= 1:
            break
        val_groups.add(group_id)
        val_count += len(by_group[group_id])

    train = [sample for sample in samples if sample.group_id not in val_groups]
    val = [sample for sample in samples if sample.group_id in val_groups]
    if not train or not val:
        raise RuntimeError("grouped split unexpectedly produced an empty partition")
    train_groups = {sample.group_id for sample in train}
    if train_groups.intersection(val_groups):
        raise RuntimeError("group leakage detected in SyncG train/validation split")
    return train, val


def _crop_to_dial(
    image: Image.Image,
    mask: Image.Image,
    bbox: tuple[float, float, float, float],
    padding_ratio: float,
) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    if mask.size != image.size:
        raise ValueError(
            f"image/mask size mismatch: image={image.size}, mask={mask.size}"
        )
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, math.floor(x1 - pad_x))
    top = max(0, math.floor(y1 - pad_y))
    right = min(width, math.ceil(x2 + pad_x))
    bottom = min(height, math.ceil(y2 + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(
            f"dial crop is empty after clipping: {(left, top, right, bottom)}"
        )
    crop_box = (left, top, right, bottom)
    return image.crop(crop_box), mask.crop(crop_box)


def letterbox_pair(
    image: Image.Image,
    mask: Image.Image,
    size: int = INPUT_SIZE,
) -> tuple[Image.Image, Image.Image]:
    """Apply exactly the same geometry as production inference to image/mask."""
    width, height = image.size
    ratio = min(size / width, size / height)
    resized_width = max(1, int(width * ratio))
    resized_height = max(1, int(height * ratio))
    image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )
    mask = mask.resize(
        (resized_width, resized_height),
        Image.Resampling.NEAREST,
    )
    image_canvas = Image.new("RGB", (size, size), 0)
    mask_canvas = Image.new("L", (size, size), 0)
    offset = ((size - resized_width) // 2, (size - resized_height) // 2)
    image_canvas.paste(image, offset)
    mask_canvas.paste(mask, offset)
    return image_canvas, mask_canvas


class SyncGPointerSegDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SyncGSegSample],
        *,
        train: bool,
        crop_padding: float,
        crop_padding_jitter: float = 0.0,
        input_size: int = INPUT_SIZE,
    ):
        if crop_padding < 0.0:
            raise ValueError("crop_padding must be non-negative")
        if crop_padding_jitter < 0.0:
            raise ValueError("crop_padding_jitter must be non-negative")
        self.samples = list(samples)
        self.train = bool(train)
        self.crop_padding = float(crop_padding)
        self.crop_padding_jitter = float(crop_padding_jitter)
        self.input_size = int(input_size)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15,
            hue=0.03,
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB").copy()
        with Image.open(sample.mask_path) as handle:
            mask = handle.convert("L").copy()
        padding = self.crop_padding
        if self.train and self.crop_padding_jitter > 0.0:
            padding = max(
                0.0,
                padding
                + random.uniform(
                    -self.crop_padding_jitter,
                    self.crop_padding_jitter,
                ),
            )
        image, mask = _crop_to_dial(
            image,
            mask,
            sample.dial_bbox,
            padding,
        )

        if self.train:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            image = self.color_jitter(image)
            if random.random() < 0.15:
                image = ImageOps.grayscale(image).convert("RGB")
            if random.random() < 0.20:
                image = image.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2))
                )

        crop_width, crop_height = image.size
        ratio = min(
            self.input_size / crop_width,
            self.input_size / crop_height,
        )
        resized_width = max(1, int(crop_width * ratio))
        resized_height = max(1, int(crop_height * ratio))
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        image, mask = letterbox_pair(image, mask, self.input_size)
        image_tensor = self.normalize(self.to_tensor(image))
        mask_tensor = (self.to_tensor(mask) > 0.5).float()
        letterbox_content = torch.tensor(
            [left, top, resized_width, resized_height],
            dtype=torch.int64,
        )
        return image_tensor, mask_tensor, letterbox_content


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _single_output_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: float,
    dice_weight: float,
    from_logits: bool = False,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Compute the loss in FP32. Training consumes U2NetP's pre-sigmoid logits:
    # weighted BCE then has bounded derivatives and does not overflow
    # GradScaler for confidently misclassified pixels on a thin pointer.
    with torch.amp.autocast(device_type=output.device.type, enabled=False):
        target = target.float()
        if valid_mask is None:
            valid = torch.ones_like(target)
        else:
            valid = valid_mask.to(device=output.device, dtype=torch.float32)
            if valid.shape[-2:] != output.shape[-2:]:
                valid = torch.nn.functional.interpolate(
                    valid,
                    size=output.shape[-2:],
                    mode="nearest",
                )
            if valid.shape[0] != output.shape[0] or valid.shape[1] not in {
                1,
                output.shape[1],
            }:
                raise ValueError(
                    "valid_mask must have batch/channel dimensions compatible "
                    f"with output: mask={tuple(valid.shape)}, "
                    f"output={tuple(output.shape)}"
                )
            valid = valid.expand_as(output)
        valid_count = valid.sum().clamp_min(1.0)
        if from_logits:
            logits = output.float()
            per_pixel_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=torch.as_tensor(
                    positive_weight,
                    dtype=logits.dtype,
                    device=logits.device,
                ),
                reduction="none",
            )
            probability = torch.sigmoid(logits)
        else:
            probability = output.float().clamp(1e-6, 1.0 - 1e-6)
            per_pixel_bce = -(
                positive_weight * target * torch.log(probability)
                + (1.0 - target) * torch.log1p(-probability)
            )
        weighted_bce = (per_pixel_bce * valid).sum() / valid_count
        probability = probability * valid
        target = target * valid
        dimensions = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=dimensions)
        denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
        soft_dice_loss = 1.0 - (
            (2.0 * intersection + 1.0) / (denominator + 1.0)
        )
        return weighted_bce + dice_weight * soft_dice_loss.mean()


def deep_supervision_loss(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    *,
    positive_weight: float,
    dice_weight: float,
    side_weight: float,
    from_logits: bool = False,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not outputs:
        raise ValueError("U2NetP returned no outputs")
    main_loss = _single_output_loss(
        outputs[0],
        target,
        positive_weight=positive_weight,
        dice_weight=dice_weight,
        from_logits=from_logits,
        valid_mask=valid_mask,
    )
    if len(outputs) == 1 or side_weight <= 0.0:
        return main_loss
    side_losses = [
        _single_output_loss(
            output,
            target,
            positive_weight=positive_weight,
            dice_weight=dice_weight,
            from_logits=from_logits,
            valid_mask=valid_mask,
        )
        for output in outputs[1:]
    ]
    return main_loss + side_weight * torch.stack(side_losses).mean()


def build_letterbox_content_mask(
    letterbox_content: torch.Tensor,
    spatial_size: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return Bx1xHxW masks selecting only pixels from the source crop."""
    height, width = map(int, spatial_size)
    content = torch.as_tensor(letterbox_content, dtype=torch.int64).cpu()
    if content.ndim != 2 or content.shape[1] != 4:
        raise ValueError(
            "letterbox_content must be Bx4 [left, top, width, height]"
        )
    valid = torch.zeros(
        (content.shape[0], 1, height, width),
        dtype=torch.float32,
        device=device,
    )
    for index, geometry in enumerate(content.tolist()):
        left, top, resized_width, resized_height = map(int, geometry)
        right = left + resized_width
        bottom = top + resized_height
        if (
            left < 0
            or top < 0
            or resized_width <= 0
            or resized_height <= 0
            or right > width
            or bottom > height
        ):
            raise ValueError(
                f"invalid letterbox content {geometry} for {(height, width)}"
            )
        valid[index, :, top:bottom, left:right] = 1.0
    return valid


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    positive_weight: float,
    dice_weight: float,
    side_weight: float,
    gradient_clip: float,
    epoch: int,
) -> tuple[float, int, int]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False, dynamic_ncols=True)
    for images, targets, letterbox_content in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_mask = build_letterbox_content_mask(
            letterbox_content,
            targets.shape[-2:],
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            outputs = model(images, return_logits=True)
            loss = deep_supervision_loss(
                outputs,
                targets,
                positive_weight=positive_weight,
                dice_weight=dice_weight,
                side_weight=side_weight,
                from_logits=True,
                valid_mask=valid_mask,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scale_before_step = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        # A scale decrease means GradScaler detected non-finite gradients and
        # deliberately skipped optimizer.step().
        if scaler.is_enabled() and float(scaler.get_scale()) < scale_before_step:
            skipped_optimizer_steps += 1
        else:
            optimizer_steps += 1

        batch_size = images.shape[0]
        running_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{running_loss / sample_count:.4f}")
    if sample_count > 0 and optimizer_steps == 0:
        raise FloatingPointError(
            "GradScaler skipped every optimizer update in this epoch; "
            "check the loss and input tensors for non-finite values"
        )
    return (
        running_loss / max(1, sample_count),
        optimizer_steps,
        skipped_optimizer_steps,
    )


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    positive_weight: float,
    dice_weight: float,
    side_weight: float,
    thresholds: Sequence[float],
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    threshold_tensor = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1, 1)
    true_positive = torch.zeros(len(thresholds), dtype=torch.float64)
    false_positive = torch.zeros(len(thresholds), dtype=torch.float64)
    false_negative = torch.zeros(len(thresholds), dtype=torch.float64)
    running_loss = 0.0
    sample_count = 0

    progress = tqdm(loader, desc=f"valid {epoch}", leave=False, dynamic_ncols=True)
    for images, targets, letterbox_content in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        valid_mask = build_letterbox_content_mask(
            letterbox_content,
            targets.shape[-2:],
            device=device,
        )
        with _autocast(device, amp_enabled):
            outputs = model(images, return_logits=True)
            loss = deep_supervision_loss(
                outputs,
                targets,
                positive_weight=positive_weight,
                dice_weight=dice_weight,
                side_weight=side_weight,
                from_logits=True,
                valid_mask=valid_mask,
            )
        probability = torch.sigmoid(outputs[0].float())
        for sample_index, geometry in enumerate(letterbox_content.tolist()):
            left, top, resized_width, resized_height = map(int, geometry)
            sample_probability = probability[
                sample_index : sample_index + 1,
                :,
                top : top + resized_height,
                left : left + resized_width,
            ]
            sample_target = targets[
                sample_index : sample_index + 1,
                :,
                top : top + resized_height,
                left : left + resized_width,
            ].bool()
            predictions = sample_probability.unsqueeze(0) >= threshold_tensor
            target = sample_target.unsqueeze(0)
            dimensions = (1, 2, 3, 4)
            true_positive += (predictions & target).sum(dim=dimensions).cpu()
            false_positive += (predictions & ~target).sum(dim=dimensions).cpu()
            false_negative += (~predictions & target).sum(dim=dimensions).cpu()
        batch_size = images.shape[0]
        running_loss += float(loss) * batch_size
        sample_count += batch_size

    threshold_metrics = []
    for index, threshold in enumerate(thresholds):
        tp = float(true_positive[index])
        fp = float(false_positive[index])
        fn = float(false_negative[index])
        dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
        iou = (tp + 1.0) / (tp + fp + fn + 1.0)
        precision = (tp + 1.0) / (tp + fp + 1.0)
        recall = (tp + 1.0) / (tp + fn + 1.0)
        threshold_metrics.append(
            {
                "threshold": float(threshold),
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            }
        )
    best = max(
        threshold_metrics,
        key=lambda item: (item["dice"], -abs(item["threshold"] - 0.5)),
    )
    return {
        "loss": running_loss / max(1, sample_count),
        "recommended_threshold": best["threshold"],
        "dice": best["dice"],
        "iou": best["iou"],
        "precision": best["precision"],
        "recall": best["recall"],
        "threshold_metrics": threshold_metrics,
        "postprocess": "remove_letterbox_padding_then_raw_probability_threshold",
    }


def retain_largest_components(binary_masks: np.ndarray) -> np.ndarray:
    """Apply the same largest-component rule used by meterZeroShot."""
    masks = np.asarray(binary_masks, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, ...]
    output = np.zeros_like(masks, dtype=bool)
    for index, mask in enumerate(masks):
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
        if component_count <= 1:
            continue
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        output[index] = labels == largest_label
    return output


@torch.inference_mode()
def calibrate_deployment_threshold(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Select a frozen threshold after production-equivalent postprocessing."""
    model.eval()
    true_positive = np.zeros(len(thresholds), dtype=np.float64)
    false_positive = np.zeros(len(thresholds), dtype=np.float64)
    false_negative = np.zeros(len(thresholds), dtype=np.float64)
    sample_count = 0
    for images, targets, letterbox_content in tqdm(
        loader,
        desc="calibrate deployment threshold",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        with _autocast(device, amp_enabled):
            logits = model(images, return_logits=True)[0]
        probability_np = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
        target_np = targets.numpy()[:, 0].astype(bool)
        for sample_index, geometry in enumerate(letterbox_content.tolist()):
            left, top, resized_width, resized_height = map(int, geometry)
            sample_probability = probability_np[
                sample_index,
                top : top + resized_height,
                left : left + resized_width,
            ]
            sample_target = target_np[
                sample_index,
                top : top + resized_height,
                left : left + resized_width,
            ]
            for threshold_index, threshold in enumerate(thresholds):
                prediction = retain_largest_components(
                    sample_probability >= float(threshold)
                )[0]
                true_positive[threshold_index] += np.sum(
                    prediction & sample_target
                )
                false_positive[threshold_index] += np.sum(
                    prediction & ~sample_target
                )
                false_negative[threshold_index] += np.sum(
                    ~prediction & sample_target
                )
            sample_count += 1

    threshold_metrics = []
    for index, threshold in enumerate(thresholds):
        tp = float(true_positive[index])
        fp = float(false_positive[index])
        fn = float(false_negative[index])
        threshold_metrics.append(
            {
                "threshold": float(threshold),
                "dice": (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0),
                "iou": (tp + 1.0) / (tp + fp + fn + 1.0),
                "precision": (tp + 1.0) / (tp + fp + 1.0),
                "recall": (tp + 1.0) / (tp + fn + 1.0),
            }
        )
    best = max(
        threshold_metrics,
        key=lambda item: (item["dice"], -abs(item["threshold"] - 0.5)),
    )
    return {
        "samples": sample_count,
        "recommended_threshold": best["threshold"],
        "dice": best["dice"],
        "iou": best["iou"],
        "precision": best["precision"],
        "recall": best["recall"],
        "threshold_metrics": threshold_metrics,
        "postprocess": (
            "remove_letterbox_padding_then_largest_connected_component"
        ),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _save_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    threshold: float,
    epoch: int,
    validation: dict[str, Any],
    training_protocol: dict[str, Any],
) -> None:
    payload = {
        "format_version": "u2netp_pointer_seg_v1",
        "model_class": "U2NETP",
        "input_size": INPUT_SIZE,
        "state_dict": _cpu_state_dict(model),
        "probability_threshold": float(threshold),
        "epoch": int(epoch),
        "validation": validation,
        "training_protocol": training_protocol,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_split_file(
    path: Path,
    root: Path,
    train_samples: Sequence[SyncGSegSample],
    val_samples: Sequence[SyncGSegSample],
    *,
    seed: int,
    val_fraction: float,
) -> dict[str, Any]:
    train_groups = sorted({sample.group_id for sample in train_samples})
    val_groups = sorted({sample.group_id for sample in val_samples})
    split = {
        "protocol": "syncg_train_grouped_scene_gauge_holdout_v1",
        "syncg_root": str(root),
        "seed": seed,
        "requested_val_fraction": val_fraction,
        "train_rows": len(train_samples),
        "validation_rows": len(val_samples),
        "train_groups": train_groups,
        "validation_groups": val_groups,
        "train_sample_ids": [sample.sample_id for sample in train_samples],
        "validation_sample_ids": [sample.sample_id for sample in val_samples],
    }
    split["train_ids_sha256"] = _json_sha256(split["train_sample_ids"])
    split["validation_ids_sha256"] = _json_sha256(split["validation_sample_ids"])
    path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split


def _parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(not 0.0 < item < 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError(
            "thresholds must be a comma-separated list strictly between 0 and 1"
        )
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-weights",
        type=Path,
        default=Path("utils/angleDetect/pointerSeg/resultSeg/best.pt"),
    )
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--crop-padding-jitter", type=float, default=0.03)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--side-weight", type=float, default=0.4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds(
            "0.004,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
        ),
    )
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp-initial-scale",
        type=float,
        default=512.0,
        help=(
            "initial GradScaler scale; U2NetP overflows at PyTorch's 65536 "
            "default on the validated FP16 environment"
        ),
    )
    parser.add_argument("--limit", type=int, help="diagnostic subset; never use in paper")
    parser.add_argument("--allow-dataset-drift", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.positive_weight <= 0.0 or args.dice_weight < 0.0:
        raise ValueError("invalid loss hyperparameters")
    if args.amp_initial_scale <= 0.0:
        raise ValueError("amp-initial-scale must be positive")

    _set_determinism(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_enabled = bool(args.amp and device.type == "cuda")

    root, samples = discover_syncg_train_samples(
        args.root,
        limit=args.limit,
        strict_release=not args.allow_dataset_drift,
    )
    train_samples, val_samples = grouped_train_val_split(
        samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    if history_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{history_path} exists; pass --resume or --overwrite explicitly"
        )

    split = _write_split_file(
        args.output_dir / "split.json",
        root,
        train_samples,
        val_samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    initial_weights_sha256 = None
    if not args.from_scratch:
        if not args.initial_weights.is_file():
            raise FileNotFoundError(args.initial_weights)
        initial_weights_sha256 = _sha256(args.initial_weights)

    cuda_properties = (
        torch.cuda.get_device_properties(device)
        if device.type == "cuda"
        else None
    )
    run_config = {
        "protocol": "syncg_train_only_u2netp_finetune_v1",
        "created_unix": time.time(),
        "device": str(device),
        "amp": amp_enabled,
        "amp_initial_scale": args.amp_initial_scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "crop_padding": args.crop_padding,
        "crop_padding_jitter": args.crop_padding_jitter,
        "positive_weight": args.positive_weight,
        "dice_weight": args.dice_weight,
        "side_weight": args.side_weight,
        "gradient_clip": args.gradient_clip,
        "patience": args.patience,
        "seed": args.seed,
        "thresholds": args.thresholds,
        "limit": args.limit,
        "syncg_reference_commit": SYNCG_PINNED_COMMIT,
        "syncg_release_identity_verified": bool(
            args.limit is None
            and not args.allow_dataset_drift
            and len(samples) == SYNCG_TRAIN_ROWS
            and syncg_sample_ids_sha256(
                sample.sample_id for sample in samples
            )
            == SYNCG_SAMPLE_IDS_SHA256["train"]
        ),
        "syncg_sample_ids_sha256": syncg_sample_ids_sha256(
            sample.sample_id for sample in samples
        ),
        "initial_weights": None if args.from_scratch else str(args.initial_weights.resolve()),
        "initial_weights_sha256": initial_weights_sha256,
        "source_sha256": {
            "training": _sha256(Path(__file__).resolve()),
            "dataset_protocol": _sha256(
                Path(__file__).resolve().parent / "datasets.py"
            ),
            "u2netp": _sha256(
                Path(__file__).resolve().parents[1]
                / "utils"
                / "angleDetect"
                / "pointerSeg"
                / "u2netp.py"
            ),
            "checkpoint_loader": _sha256(
                Path(__file__).resolve().parents[1]
                / "utils"
                / "angleDetect"
                / "pointerSeg"
                / "detectSeg.py"
            ),
        },
        "split_protocol": split["protocol"],
        "train_ids_sha256": split["train_ids_sha256"],
        "validation_ids_sha256": split["validation_ids_sha256"],
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": (
            None if cuda_properties is None else cuda_properties.name
        ),
        "gpu_compute_capability": (
            None
            if cuda_properties is None
            else [
                int(cuda_properties.major),
                int(cuda_properties.minor),
            ]
        ),
    }
    run_config["signature"] = _json_sha256(
        {key: value for key, value in run_config.items() if key != "created_unix"}
    )
    config_path = args.output_dir / "run_config.json"
    if args.resume:
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if previous_config.get("signature") != run_config["signature"]:
            raise ValueError("resume configuration signature mismatch")
    else:
        config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    train_dataset = SyncGPointerSegDataset(
        train_samples,
        train=True,
        crop_padding=args.crop_padding,
        crop_padding_jitter=args.crop_padding_jitter,
    )
    val_dataset = SyncGPointerSegDataset(
        val_samples,
        train=False,
        crop_padding=args.crop_padding,
        crop_padding_jitter=0.0,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
        # Recreate workers each epoch so a saved DataLoader generator state can
        # reproduce shuffle order and augmentation RNG after --resume.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options,
    )

    model = U2NETP().to(device)
    initial_state_dict = None
    if not args.from_scratch:
        initial_state_dict, _ = load_u2net_state_dict(args.initial_weights, device)
        if not args.resume:
            model.load_state_dict(initial_state_dict)

    # A fair component comparison must not give only the fine-tuned weights a
    # train-validation-selected threshold. Package the untouched released
    # weights with a threshold calibrated on the exact same validation split.
    released_calibrated_path = args.output_dir / "released_calibrated.pt"
    released_calibration = None
    if initial_state_dict is not None and (
        not args.resume or not released_calibrated_path.is_file()
    ):
        model.load_state_dict(initial_state_dict)
        released_calibration = calibrate_deployment_threshold(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            thresholds=args.thresholds,
        )
        _save_model_checkpoint(
            released_calibrated_path,
            model,
            threshold=released_calibration["recommended_threshold"],
            epoch=0,
            validation={
                "role": "released_weights_validation_threshold_calibration",
                "deployment_threshold_calibration": released_calibration,
            },
            training_protocol=run_config,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=args.amp_initial_scale,
    )
    start_epoch = 1
    best_dice = -math.inf
    best_epoch = 0
    stale_epochs = 0
    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    if args.resume:
        try:
            checkpoint = torch.load(
                last_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("run_signature") != run_config["signature"]:
            raise ValueError("last.pt run signature does not match run_config.json")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        generator_state = checkpoint.get("data_loader_generator_state")
        if generator_state is not None:
            generator.set_state(generator_state)

    print(
        json.dumps(
            {
                "device": str(device),
                "amp": amp_enabled,
                "train_rows": len(train_samples),
                "validation_rows": len(val_samples),
                "train_groups": len({sample.group_id for sample in train_samples}),
                "validation_groups": len({sample.group_id for sample in val_samples}),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            ensure_ascii=False,
        )
    )

    history_mode = "a" if args.resume else "w"
    with history_path.open(history_mode, encoding="utf-8", newline="\n") as history:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, optimizer_steps, skipped_optimizer_steps = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device=device,
                amp_enabled=amp_enabled,
                positive_weight=args.positive_weight,
                dice_weight=args.dice_weight,
                side_weight=args.side_weight,
                gradient_clip=args.gradient_clip,
                epoch=epoch,
            )
            validation = validate(
                model,
                val_loader,
                device=device,
                amp_enabled=amp_enabled,
                positive_weight=args.positive_weight,
                dice_weight=args.dice_weight,
                side_weight=args.side_weight,
                thresholds=args.thresholds,
                epoch=epoch,
            )
            current_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "optimizer_steps": optimizer_steps,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "validation": validation,
                "learning_rate": current_lr,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            history.flush()

            improved = validation["dice"] > best_dice + 1e-8
            if improved:
                best_dice = float(validation["dice"])
                best_epoch = epoch
                stale_epochs = 0
                _save_model_checkpoint(
                    best_path,
                    model,
                    threshold=validation["recommended_threshold"],
                    epoch=epoch,
                    validation=validation,
                    training_protocol=run_config,
                )
            else:
                stale_epochs += 1

            last_payload = {
                "format_version": "u2netp_pointer_seg_training_state_v1",
                "state_dict": _cpu_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "run_signature": run_config["signature"],
                "data_loader_generator_state": generator.get_state(),
            }
            temporary = last_path.with_suffix(last_path.suffix + ".tmp")
            torch.save(last_payload, temporary)
            temporary.replace(last_path)
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"optimizer_steps={optimizer_steps} "
                f"skipped_steps={skipped_optimizer_steps} "
                f"val_loss={validation['loss']:.5f} "
                f"dice={validation['dice']:.5f} iou={validation['iou']:.5f} "
                f"threshold={validation['recommended_threshold']:.4f} "
                f"best_epoch={best_epoch}"
            )
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"early stopping after {stale_epochs} stale epochs")
                break

    best_state_dict, best_metadata = load_u2net_state_dict(best_path, device)
    model.load_state_dict(best_state_dict)
    deployment_calibration = calibrate_deployment_threshold(
        model,
        val_loader,
        device=device,
        amp_enabled=amp_enabled,
        thresholds=args.thresholds,
    )
    best_validation = dict(best_metadata.get("validation") or {})
    best_validation["deployment_threshold_calibration"] = deployment_calibration
    _save_model_checkpoint(
        best_path,
        model,
        threshold=deployment_calibration["recommended_threshold"],
        epoch=int(best_metadata.get("epoch", best_epoch)),
        validation=best_validation,
        training_protocol=best_metadata.get("training_protocol") or run_config,
    )

    summary = {
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _sha256(best_path),
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "deployment_probability_threshold": deployment_calibration[
            "recommended_threshold"
        ],
        "deployment_validation_dice": deployment_calibration["dice"],
        "deployment_postprocess": deployment_calibration["postprocess"],
        "completed_unix": time.time(),
        "run_signature": run_config["signature"],
    }
    if released_calibrated_path.is_file():
        _, released_metadata = load_u2net_state_dict(
            released_calibrated_path,
            torch.device("cpu"),
        )
        released_validation = released_metadata.get("validation") or {}
        released_calibration = released_validation.get(
            "deployment_threshold_calibration"
        ) or released_calibration
        summary.update(
            {
                "released_calibrated_checkpoint": str(
                    released_calibrated_path.resolve()
                ),
                "released_calibrated_checkpoint_sha256": _sha256(
                    released_calibrated_path
                ),
                "released_validation_probability_threshold": (
                    None
                    if released_calibration is None
                    else released_calibration["recommended_threshold"]
                ),
                "released_deployment_validation_dice": (
                    None
                    if released_calibration is None
                    else released_calibration["dice"]
                ),
            }
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
