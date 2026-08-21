"""Train the public-only CAGH ScaleMark head on canonical tight dial ROIs.

V4 inherited the historical ``1.25x`` square affine crop.  That crop may read
outside an already-tight ROI and fill the missing area with black pixels.  V5
deliberately uses a different, explicit contract:

* SyncG/train is the only readable dataset namespace;
* a public ground-truth dial bbox is clipped to the source image, sliced, and
  resized directly (no detector, expansion, letterbox, or constant border);
* a deployment ROI is represented by the whole input image (``bbox=None``);
* optional photo-style augmentation never uses constant/black padding and
  transforms ScaleMark labels with the exact same homography; and
* terminal public-validation predictions include endpoint, arc, and validity
  telemetry instead of only a scalar reading metric.

The old V4 code and artifacts remain untouched.  This entry trains only the
ScaleMark head on top of the frozen, pinned PEPD backbone; it does not claim to
retrain or replace the PEPD direction expert.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.cagh_scalemark_reference_head import scalemark_reference_loss
from experiments.cagh_scalemark_reference_head_v3 import dense_tick_heatmaps, dense_tick_loss
from experiments.fadr_multiseed_protocol import sha256_file
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import decode_probabilistic_pivot_direction
from experiments.screen_cagh_factorized_reference_oracle import derive_gt_scalemark_reference
from experiments.train_cagh_scalemark_reference_probe import load_pepd
from experiments.train_cagh_scalemark_reference_probe_v3 import build_head, weighted
from experiments.train_cagh_scalemark_reference_probe_v4 import set_stage
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sha256_source_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_public_v5_tight_roi_v1"
PROTOCOL_PATH = PROJECT_ROOT / "experiments/cagh_scalemark_reference_public_v5_protocol.json"
MANIFEST = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
MANIFEST_PROTOCOL = MANIFEST.with_name(MANIFEST.name + ".protocol.json")
BACKBONE_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/runs/pepd_convergence_phase2/seed_20260720/best.pt"
)
DEFAULT_OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_scalemark_reference_public_v5\runs")
DEFAULT_CACHE_ROOT = Path(r"C:\pointer_read\cagh_scalemark_reference_public_v5\cache")
EXPECTED_ALL = (16_000, 725)
EXPECTED_FIT = (14_400, 651)
EXPECTED_VALIDATION = (1_600, 74)
IMAGE_SIZE = 256
FORBIDDEN_PATH_TOKENS = frozenset(
    {
        "field",
        "confirmatory",
        "confirmation",
        "sealed",
        "test",
        "xiangmu1",
        "xiangmu2",
    }
)
CROP_CONTRACT = {
    "schema_version": 1,
    "name": "canonical_tight_roi_v1",
    "public_source_roi": "clip ground-truth xyxy bbox to image, direct slice, resize 256x256",
    "deployment_roi": "whole supplied ROI; bbox=None; detector forbidden",
    "bbox_expansion": 1.0,
    "preserve_aspect_ratio_with_letterbox": False,
    "constant_border": False,
    "geometric_augmentation_border": "reflect101",
    "label_transform": "exact normalized homography",
}


@dataclass(frozen=True)
class PhotoAugmentation:
    brightness_probability: float
    brightness_delta: float
    contrast_probability: float
    contrast_min: float
    contrast_max: float
    gamma_probability: float
    gamma_min: float
    gamma_max: float
    blur_probability: float
    blur_sigma_max: float
    noise_probability: float
    noise_sigma_max: float
    jpeg_probability: float
    jpeg_quality_min: int
    jpeg_quality_max: int
    perspective_probability: float
    perspective_fraction_max: float
    boundary_trim_probability: float
    boundary_trim_fraction_max: float

    @classmethod
    def disabled(cls) -> "PhotoAugmentation":
        return cls(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 100, 100, 0.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        probabilities = (
            self.brightness_probability,
            self.contrast_probability,
            self.gamma_probability,
            self.blur_probability,
            self.noise_probability,
            self.jpeg_probability,
            self.perspective_probability,
            self.boundary_trim_probability,
        )
        _require(all(0.0 <= value <= 1.0 for value in probabilities),
                 "augmentation probabilities must be in [0,1]")
        _require(0.0 <= self.brightness_delta <= 1.0, "invalid brightness delta")
        _require(0.0 < self.contrast_min <= self.contrast_max, "invalid contrast range")
        _require(0.0 < self.gamma_min <= self.gamma_max, "invalid gamma range")
        _require(self.blur_sigma_max >= 0.0 and self.noise_sigma_max >= 0.0,
                 "blur/noise magnitudes must be non-negative")
        _require(1 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100,
                 "invalid JPEG quality range")
        _require(0.0 <= self.perspective_fraction_max <= 0.10,
                 "perspective fraction must be in [0,.10]")
        _require(0.0 <= self.boundary_trim_fraction_max <= 0.15,
                 "boundary trim fraction must be in [0,.15]")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PublicRecord:
    sample: Any
    group_weight: float
    partition: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_restricted_path(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    for part in resolved.parts:
        normalized = part.casefold()
        if normalized in FORBIDDEN_PATH_TOKENS or any(
            normalized.startswith(f"{token}_") for token in FORBIDDEN_PATH_TOKENS
        ):
            raise ValueError(f"{label} enters forbidden namespace: {part!r}")
    return resolved


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = _guard_restricted_path(path, label=label)
    allowed = root.resolve()
    _require(resolved.is_relative_to(allowed), f"{label} is outside {allowed}")
    return resolved


def _load_protocol() -> Mapping[str, Any]:
    _require(PROTOCOL_PATH.is_file(), f"missing V5 protocol: {PROTOCOL_PATH}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(protocol.get("protocol") == PROTOCOL, "V5 protocol identity drift")
    _require(protocol.get("status") == "frozen_public_only", "V5 protocol is not frozen")
    for path, key in (
        (MANIFEST, "manifest_sha256"),
        (MANIFEST_PROTOCOL, "manifest_protocol_sha256"),
        (BACKBONE_CHECKPOINT, "backbone_checkpoint_sha256"),
    ):
        _require(path.is_file(), f"missing protocol input: {path}")
        _require(sha256_file(path) == protocol["inputs"][key], f"{key} drift")
    _require(
        canonical_sha256(CROP_CONTRACT) == protocol["crop_contract_sha256"],
        "canonical tight-ROI crop contract drift",
    )
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=20261206)
    parser.add_argument("--stage-a-epochs", type=int, default=6)
    parser.add_argument("--stage-b-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation-profile", choices=("photo", "none"), default="photo")
    parser.add_argument("--brightness-probability", type=float, default=.80)
    parser.add_argument("--brightness-delta", type=float, default=.16)
    parser.add_argument("--contrast-probability", type=float, default=.80)
    parser.add_argument("--contrast-min", type=float, default=.65)
    parser.add_argument("--contrast-max", type=float, default=1.40)
    parser.add_argument("--gamma-probability", type=float, default=.45)
    parser.add_argument("--gamma-min", type=float, default=.65)
    parser.add_argument("--gamma-max", type=float, default=1.55)
    parser.add_argument("--blur-probability", type=float, default=.35)
    parser.add_argument("--blur-sigma-max", type=float, default=1.6)
    parser.add_argument("--noise-probability", type=float, default=.30)
    parser.add_argument("--noise-sigma-max", type=float, default=10.0)
    parser.add_argument("--jpeg-probability", type=float, default=.35)
    parser.add_argument("--jpeg-quality-min", type=int, default=50)
    parser.add_argument("--jpeg-quality-max", type=int, default=94)
    parser.add_argument("--perspective-probability", type=float, default=.35)
    parser.add_argument("--perspective-fraction-max", type=float, default=.035)
    parser.add_argument("--boundary-trim-probability", type=float, default=.35)
    parser.add_argument("--boundary-trim-fraction-max", type=float, default=.06)
    parser.add_argument("--smoke-fit-samples", type=int, default=8)
    parser.add_argument("--smoke-validation-samples", type=int, default=8)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def augmentation_from_args(args: argparse.Namespace) -> PhotoAugmentation:
    if args.augmentation_profile == "none":
        return PhotoAugmentation.disabled()
    value = PhotoAugmentation(
        brightness_probability=args.brightness_probability,
        brightness_delta=args.brightness_delta,
        contrast_probability=args.contrast_probability,
        contrast_min=args.contrast_min,
        contrast_max=args.contrast_max,
        gamma_probability=args.gamma_probability,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        blur_probability=args.blur_probability,
        blur_sigma_max=args.blur_sigma_max,
        noise_probability=args.noise_probability,
        noise_sigma_max=args.noise_sigma_max,
        jpeg_probability=args.jpeg_probability,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        perspective_probability=args.perspective_probability,
        perspective_fraction_max=args.perspective_fraction_max,
        boundary_trim_probability=args.boundary_trim_probability,
        boundary_trim_fraction_max=args.boundary_trim_fraction_max,
    )
    value.validate()
    return value


def load_public_roster(protocol: Mapping[str, Any]) -> tuple[list[PublicRecord], list[PublicRecord]]:
    samples, manifest_protocol = load_syncg_manifest(MANIFEST, expected_split="train")
    _require(manifest_protocol.get("dataset") == "SyncG", "manifest protocol dataset drift")
    split_seed = int(protocol["split"]["seed"])
    split_fraction = float(protocol["split"]["validation_fraction"])
    fit_values, validation_values = grouped_train_val_split(
        samples, validation_fraction=split_fraction, seed=split_seed
    )
    fit_ids = {sample.sample_id for sample in fit_values}
    validation_ids = {sample.sample_id for sample in validation_values}
    fit_groups = {sample.group_id for sample in fit_values}
    validation_groups = {sample.group_id for sample in validation_values}
    _require(not fit_ids.intersection(validation_ids), "public fit/validation sample overlap")
    _require(not fit_groups.intersection(validation_groups), "public fit/validation group overlap")
    _require((len(samples), len({sample.group_id for sample in samples})) == EXPECTED_ALL,
             "SyncG inventory drift")
    _require((len(fit_ids), len(fit_groups)) == EXPECTED_FIT, "fit inventory drift")
    _require((len(validation_ids), len(validation_groups)) == EXPECTED_VALIDATION,
             "validation inventory drift")
    _require(canonical_sha256(sorted(fit_ids)) == protocol["split"]["fit_ids_sha256"],
             "fit roster hash drift")
    _require(canonical_sha256(sorted(validation_ids)) ==
             protocol["split"]["validation_ids_sha256"], "validation roster hash drift")
    _require(canonical_sha256(sorted(fit_groups)) == protocol["split"]["fit_groups_sha256"],
             "fit group hash drift")
    _require(canonical_sha256(sorted(validation_groups)) ==
             protocol["split"]["validation_groups_sha256"], "validation group hash drift")

    syncg_root = (PROJECT_ROOT / "datasets/SyncG/syncG").resolve()
    image_root = syncg_root / "images/train"
    annotation_root = syncg_root / "annotations/train"
    fit_samples: list[Any] = []
    validation_samples: list[Any] = []
    for sample in samples:
        _require(sample.dataset == "SyncG" and sample.split == "train",
                 f"non-public-train sample: {sample.sample_id}")
        _require_under(Path(sample.image_path), image_root, label=f"{sample.sample_id}.image")
        annotation = Path(str(sample.metadata.get("annotation_path") or ""))
        _require_under(annotation, annotation_root, label=f"{sample.sample_id}.annotation")
        if sample.sample_id in fit_ids:
            _require(sample.group_id in fit_groups, f"fit group drift: {sample.sample_id}")
            fit_samples.append(sample)
        elif sample.sample_id in validation_ids:
            _require(sample.group_id in validation_groups,
                     f"validation group drift: {sample.sample_id}")
            validation_samples.append(sample)
        else:
            raise ValueError(f"sample outside frozen public roster: {sample.sample_id}")

    def weighted_records(values: Sequence[Any], partition: str) -> list[PublicRecord]:
        counts = Counter(value.group_id for value in values)
        return [
            PublicRecord(
                sample=value,
                group_weight=len(values) / (len(counts) * counts[value.group_id]),
                partition=partition,
            )
            for value in values
        ]

    return weighted_records(fit_samples, "fit"), weighted_records(validation_samples, "validation")


def _scalemark_points(sample: Any) -> np.ndarray:
    entries = [
        item for item in sample.metadata.get("keypoints") or []
        if isinstance(item, Mapping) and str(item.get("type") or "").casefold() == "scalemark"
    ]
    _require(len(entries) == 1, f"{sample.sample_id}: expected one ScaleMark annotation")
    values = entries[0].get("all_kp")
    _require(isinstance(values, Sequence) and 7 <= len(values) <= 36,
             f"{sample.sample_id}: invalid ScaleMark inventory")
    points = np.asarray(values, dtype=np.float32)
    _require(points.shape == (len(values), 2) and np.isfinite(points).all(),
             f"{sample.sample_id}: invalid ScaleMark coordinates")
    return points


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(values, np.asarray(matrix, dtype=np.float32))
    return result.reshape(-1, 2).astype(np.float32)


def canonical_tight_roi(
    image: np.ndarray,
    bbox: Sequence[float] | None,
    *,
    output_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Slice one in-bounds ROI and resize it without any synthesized border.

    ``bbox=None`` is the deployment contract for an image that is already a
    canonical ROI.  A bbox is accepted only for public source images carrying
    ground-truth dial annotations; this function never invokes a detector.
    The returned matrices map original pixels to normalized crop coordinates
    and normalized crop coordinates back to isotropic dial coordinates.
    """

    _require(image.ndim == 3 and image.shape[2] == 3, "canonical ROI expects BGR image")
    height, width = image.shape[:2]
    if bbox is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        _require(len(bbox) >= 4, "dial bbox must contain xyxy")
        x1, y1, x2, y2 = map(float, bbox[:4])
        _require(np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1,
                 f"invalid canonical bbox: {bbox}")
        left = max(0, min(width - 1, int(math.floor(x1))))
        top = max(0, min(height - 1, int(math.floor(y1))))
        right = max(left + 1, min(width, int(math.ceil(x2))))
        bottom = max(top + 1, min(height, int(math.ceil(y2))))
    roi = image[top:bottom, left:right]
    _require(roi.size > 0 and roi.shape[0] >= 2 and roi.shape[1] >= 2,
             "canonical tight ROI collapsed")
    interpolation = cv2.INTER_AREA if max(roi.shape[:2]) > output_size else cv2.INTER_LINEAR
    resized = cv2.resize(roi, (output_size, output_size), interpolation=interpolation)
    roi_width, roi_height = float(right - left), float(bottom - top)
    original_to_normalized = np.asarray(
        [[1.0 / roi_width, 0.0, -left / roi_width],
         [0.0, 1.0 / roi_height, -top / roi_height],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    side = max(roi_width, roi_height)
    normalized_to_isotropic = np.asarray(
        [[roi_width / side, 0.0, 0.0],
         [0.0, roi_height / side, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return resized, original_to_normalized, normalized_to_isotropic, (left, top, right, bottom)


def _pixel_to_normalized_homography(matrix: np.ndarray, size: int) -> np.ndarray:
    scale = float(size - 1)
    to_pixels = np.diag([scale, scale, 1.0]).astype(np.float32)
    to_normalized = np.diag([1.0 / scale, 1.0 / scale, 1.0]).astype(np.float32)
    return (to_normalized @ np.asarray(matrix, dtype=np.float32) @ to_pixels).astype(np.float32)


def _augment_geometry(
    image: np.ndarray,
    protected_points: np.ndarray,
    rng: np.random.Generator,
    config: PhotoAugmentation,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply label-preserving trim/perspective with reflect borders only."""

    size = int(image.shape[0])
    _require(image.shape[:2] == (size, size), "augmentation expects a square canonical ROI")
    forward = np.eye(3, dtype=np.float32)
    code = 0
    points = np.asarray(protected_points, dtype=np.float32)

    if rng.random() < config.boundary_trim_probability and config.boundary_trim_fraction_max > 0:
        for _ in range(4):
            maximum = config.boundary_trim_fraction_max
            fractions = rng.uniform(0.0, maximum, size=4)
            x0 = int(round(float(fractions[0]) * (size - 1)))
            y0 = int(round(float(fractions[1]) * (size - 1)))
            x1 = int(round((1.0 - float(fractions[2])) * (size - 1)))
            y1 = int(round((1.0 - float(fractions[3])) * (size - 1)))
            if x1 - x0 < size // 2 or y1 - y0 < size // 2:
                continue
            pixel = np.asarray(
                [[(size - 1) / (x1 - x0), 0.0, -x0 * (size - 1) / (x1 - x0)],
                 [0.0, (size - 1) / (y1 - y0), -y0 * (size - 1) / (y1 - y0)],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            candidate = _pixel_to_normalized_homography(pixel, size)
            transformed = _apply_homography(points, candidate)
            if bool(((transformed >= .01) & (transformed <= .99)).all()):
                cropped = image[y0:y1 + 1, x0:x1 + 1]
                image = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
                forward = candidate @ forward
                points = transformed
                code |= 1
                break

    if rng.random() < config.perspective_probability and config.perspective_fraction_max > 0:
        source = np.asarray(
            [[0.0, 0.0], [size - 1.0, 0.0],
             [size - 1.0, size - 1.0], [0.0, size - 1.0]], dtype=np.float32
        )
        maximum = config.perspective_fraction_max * (size - 1)
        for _ in range(4):
            inward = rng.uniform(0.0, maximum, size=(4, 2)).astype(np.float32)
            destination = np.asarray(
                [[inward[0, 0], inward[0, 1]],
                 [size - 1.0 - inward[1, 0], inward[1, 1]],
                 [size - 1.0 - inward[2, 0], size - 1.0 - inward[2, 1]],
                 [inward[3, 0], size - 1.0 - inward[3, 1]]], dtype=np.float32
            )
            pixel = cv2.getPerspectiveTransform(source, destination)
            candidate = _pixel_to_normalized_homography(pixel, size)
            transformed = _apply_homography(points, candidate)
            if bool(((transformed >= .005) & (transformed <= .995)).all()):
                image = cv2.warpPerspective(
                    image, pixel, (size, size), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                forward = candidate @ forward
                points = transformed
                code |= 2
                break
    return image, forward, code


def _augment_photo(
    image: np.ndarray,
    rng: np.random.Generator,
    config: PhotoAugmentation,
) -> tuple[np.ndarray, int]:
    output = image.astype(np.float32)
    code = 0
    if rng.random() < config.contrast_probability:
        output *= float(rng.uniform(config.contrast_min, config.contrast_max)); code |= 4
    if rng.random() < config.brightness_probability:
        output += float(rng.uniform(-config.brightness_delta, config.brightness_delta)) * 255.0; code |= 8
    output = np.clip(output, 0.0, 255.0)
    if rng.random() < config.gamma_probability:
        gamma = float(np.exp(rng.uniform(math.log(config.gamma_min), math.log(config.gamma_max))))
        output = 255.0 * np.power(output / 255.0, gamma); code |= 16
    if rng.random() < config.blur_probability and config.blur_sigma_max > 0:
        sigma = float(rng.uniform(.15, max(.15, config.blur_sigma_max)))
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=sigma, sigmaY=sigma); code |= 32
    if rng.random() < config.noise_probability and config.noise_sigma_max > 0:
        sigma = float(rng.uniform(0.5, config.noise_sigma_max))
        output += rng.normal(0.0, sigma, output.shape).astype(np.float32); code |= 64
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    if rng.random() < config.jpeg_probability:
        quality = int(rng.integers(config.jpeg_quality_min, config.jpeg_quality_max + 1))
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
        _require(bool(ok), "JPEG augmentation encode failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        _require(decoded is not None and decoded.shape == output.shape,
                 "JPEG augmentation decode failed")
        output = decoded; code |= 128
    return output, code


class CanonicalTightROIDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PublicRecord],
        *,
        training: bool,
        seed: int,
        augmentation: PhotoAugmentation,
    ) -> None:
        self.records = tuple(records)
        self.training = bool(training)
        self.seed = int(seed)
        self.augmentation = augmentation if training else PhotoAugmentation.disabled()
        self.epoch = 0
        _require(bool(self.records), "canonical ROI dataset is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = record.sample
        image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        _require(image is not None, f"failed to read public image: {sample.sample_id}")
        crop, original_to_normalized, normalized_to_isotropic, bounds = canonical_tight_roi(
            image, sample.dial_bbox
        )
        marks = _apply_homography(_scalemark_points(sample), original_to_normalized)
        pointer = _apply_homography(
            np.asarray([sample.pointer_tail, sample.pointer_tip], dtype=np.float32),
            original_to_normalized,
        )
        protected = np.concatenate((marks, pointer), axis=0)
        forward = np.eye(3, dtype=np.float32)
        augmentation_code = 0
        if self.training:
            seed = self.seed + self.epoch * 1_000_003 + index * 97
            rng = np.random.default_rng(seed)
            crop, forward, augmentation_code = _augment_geometry(
                crop, protected, rng, self.augmentation
            )
            marks = _apply_homography(marks, forward)
            pointer = _apply_homography(pointer, forward)
            crop, photo_code = _augment_photo(crop, rng, self.augmentation)
            augmentation_code |= photo_code
        _require(np.isfinite(marks).all() and bool(((marks >= 0.0) & (marks <= 1.0)).all()),
                 f"{sample.sample_id}: transformed marks escaped canonical ROI")
        tick_xy = torch.from_numpy(marks)[None]
        tick_valid = torch.ones((1, len(marks)), dtype=torch.bool)
        heatmap = dense_tick_heatmaps(tick_xy, tick_valid)[0]
        final_to_isotropic = normalized_to_isotropic @ np.linalg.inv(forward)
        original_to_final = forward @ original_to_normalized
        _, _, start_degrees, range_degrees = derive_gt_scalemark_reference(
            sample.metadata, sample.sample_id
        )
        denominator = float(sample.scale_end) - float(sample.scale_start)
        _require(abs(denominator) > 1e-12, f"{sample.sample_id}: collapsed scale")
        target_progress = (float(sample.ground_truth) - float(sample.scale_start)) / denominator
        return {
            "image": normalized_rgb_tensor(crop),
            "pointer_tail": torch.from_numpy(pointer[0].astype(np.float32)),
            "pointer_tip": torch.from_numpy(pointer[1].astype(np.float32)),
            "endpoints": torch.from_numpy(np.stack((marks[0], marks[-1])).astype(np.float32)),
            "tick_heatmap": heatmap,
            "gt_start": torch.tensor(math.radians(start_degrees), dtype=torch.float32),
            "gt_range": torch.tensor(math.radians(range_degrees), dtype=torch.float32),
            "final_to_isotropic": torch.from_numpy(final_to_isotropic.astype(np.float32)),
            "crop_affine": torch.from_numpy(original_to_final[:2].astype(np.float32)),
            "group_weight": torch.tensor(record.group_weight, dtype=torch.float32),
            "target_progress": torch.tensor(target_progress, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "augmentation_code": torch.tensor(augmentation_code, dtype=torch.int32),
            "roi_bounds": torch.tensor(bounds, dtype=torch.int32),
        }


def _unproject(points: torch.Tensor, inverse_homography: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat((points.float(), torch.ones_like(points[..., :1])), dim=-1)
    transformed = torch.einsum("bij,bkj->bki", inverse_homography.float(), homogeneous)
    return transformed[..., :2] / transformed[..., 2:].clamp_min(1e-8)


def reference_geometry_projective(
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    pivot_xy: torch.Tensor,
    final_to_isotropic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.stack((start_xy, end_xy, pivot_xy), dim=1)
    original = _unproject(points, final_to_isotropic)
    start_v = original[:, 0] - original[:, 2]
    end_v = original[:, 1] - original[:, 2]
    radii = torch.stack(
        (torch.linalg.vector_norm(start_v, dim=1), torch.linalg.vector_norm(end_v, dim=1)),
        dim=1,
    )
    start = torch.remainder(torch.atan2(start_v[:, 0], -start_v[:, 1]) - math.pi,
                            2 * math.pi)
    end = torch.remainder(torch.atan2(end_v[:, 0], -end_v[:, 1]) - math.pi,
                          2 * math.pi)
    arc = torch.remainder(end - start, 2 * math.pi)
    valid = (
        (radii > .02).all(1)
        & (arc > math.radians(10.0))
        & (arc < math.radians(350.0))
        & torch.isfinite(original).all((1, 2))
        & torch.isfinite(arc)
    )
    return start, arc, radii, valid


def geometry_loss_v5(
    output: Any,
    endpoints: torch.Tensor,
    gt_start: torch.Tensor,
    gt_range: torch.Tensor,
    pivot: torch.Tensor,
    final_to_isotropic: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    endpoint_loss, endpoint_components = scalemark_reference_loss(
        output, endpoints, group_weight=weight
    )
    start, arc, radii, valid = reference_geometry_projective(
        output.start_xy, output.end_xy, pivot, final_to_isotropic
    )
    angle_row = 1.0 - torch.cos(start - gt_start) + F.smooth_l1_loss(
        arc / (2 * math.pi), gt_range / (2 * math.pi), reduction="none", beta=.02
    )
    arc_margin = (
        F.relu(math.radians(10) - arc) / math.radians(10)
    ).square() + (F.relu(arc - math.radians(350)) / math.radians(10)).square()
    radius_margin = (F.relu(.02 - radii) / .02).square().mean(1)
    geometry = .25 * weighted(angle_row, weight)
    margin = .5 * weighted(arc_margin + radius_margin, weight)
    loss = endpoint_loss + geometry + margin
    return loss, {
        **endpoint_components,
        "geometry_angle_arc_loss": geometry.detach(),
        "geometry_margin_loss": margin.detach(),
        "reference_valid_fraction": valid.float().mean().detach(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: CanonicalTightROIDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=pin_memory,
    )


def train_head_v5(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataset: CanonicalTightROIDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {"stage_a_tick": [], "stage_b_geometry": []}
    schedules = (
        ("tick", args.stage_a_epochs, "stage_a_tick", 0),
        ("geometry", args.stage_b_epochs, "stage_b_geometry", 1_000),
    )
    for stage, epochs, history_key, epoch_offset in schedules:
        trainable = set_stage(head, stage)
        _require(bool(trainable), f"V5 {stage} stage has no trainable head parameters")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in head.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for epoch in range(1, epochs + 1):
            dataset.set_epoch(epoch_offset + epoch)
            loader = _loader(
                dataset, batch_size=args.batch_size, workers=args.workers, shuffle=True,
                seed=args.seed + epoch_offset + epoch, pin_memory=device.type == "cuda",
            )
            head.train(); total = samples = 0; valid_total = 0.0; augmented = 0
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.no_grad():
                    features = backbone.forward_multiscale_features(images)
                    pooled = backbone.direction_features(features.c5)
                    decoded = decode_probabilistic_pivot_direction(
                        backbone.pivot_head(features.c5), backbone.vector_head(pooled),
                        backbone.angle_head(pooled), backbone.log_variance_head(pooled),
                    )
                output = head(features.c2.detach(), features.c5.detach())
                weights = batch["group_weight"].to(device, non_blocking=True)
                if stage == "tick":
                    loss, components = dense_tick_loss(
                        output, batch["tick_heatmap"].to(device, non_blocking=True),
                        group_weight=weights,
                    )
                    valid_fraction = 0.0
                else:
                    loss, components = geometry_loss_v5(
                        output,
                        batch["endpoints"].to(device, non_blocking=True),
                        batch["gt_start"].to(device, non_blocking=True),
                        batch["gt_range"].to(device, non_blocking=True),
                        decoded.pivot_xy / 63.0,
                        batch["final_to_isotropic"].to(device, non_blocking=True),
                        weights,
                    )
                    valid_fraction = float(components["reference_valid_fraction"])
                _require(bool(torch.isfinite(loss)), f"V5 {stage} loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in head.parameters() if parameter.requires_grad), 5.0
                )
                optimizer.step()
                count = int(images.shape[0])
                samples += count
                total += float(loss.detach()) * count
                valid_total += valid_fraction * count
                augmented += int((batch["augmentation_code"] != 0).sum())
            _require(samples == len(dataset), "V5 epoch sample inventory drift")
            row = {
                "epoch": epoch,
                "loss": total / samples,
                "samples": samples,
                "augmented_fraction": augmented / samples,
                "reference_valid_fraction": valid_total / samples if stage == "geometry" else None,
            }
            history[history_key].append(row)
            print(
                f"public-v5 seed={args.seed} stage={stage} epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f} augmented={row['augmented_fraction']:.3f}",
                flush=True,
            )
    return history


@torch.inference_mode()
def evaluate_with_telemetry(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataset: CanonicalTightROIDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _loader(
        dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False,
        seed=args.seed, pin_memory=device.type == "cuda",
    )
    backbone.eval(); head.eval(); rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = backbone.forward_multiscale_features(images)
        pooled = backbone.direction_features(features.c5)
        decoded = decode_probabilistic_pivot_direction(
            backbone.pivot_head(features.c5), backbone.vector_head(pooled),
            backbone.angle_head(pooled), backbone.log_variance_head(pooled),
        )
        output = head(features.c2, features.c5)
        pivot = decoded.pivot_xy / 63.0
        start, arc, radii, reference_valid = reference_geometry_projective(
            output.start_xy, output.end_xy, pivot,
            batch["final_to_isotropic"].to(device, non_blocking=True),
        )
        solver = differentiable_keypoint_reference_solver(
            pivot + .25 * decoded.direction,
            pivot,
            start,
            arc,
            batch["crop_affine"].to(device, non_blocking=True),
            reference_valid,
        )
        combined = (
            decoded.valid & reference_valid & solver.valid
            & torch.isfinite(solver.expected_progress)
        )
        target_endpoint = batch["endpoints"].to(device)
        predicted_endpoint = torch.stack((output.start_xy, output.end_xy), dim=1)
        endpoint_error = torch.linalg.vector_norm(predicted_endpoint - target_endpoint, dim=2).mean(1)
        target_start = batch["gt_start"].to(device)
        target_arc = batch["gt_range"].to(device)
        start_error = torch.abs(torch.atan2(torch.sin(start - target_start),
                                            torch.cos(start - target_start)))
        arc_error = torch.abs(arc - target_arc)
        for index, sample_id in enumerate(batch["sample_id"]):
            valid = bool(combined[index])
            prediction = float(solver.expected_progress[index]) if valid else None
            target = float(batch["target_progress"][index])
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": str(sample_id),
                    "group_id": str(batch["group_id"][index]),
                    "status": "ok" if valid else "failed",
                    "failure_code": None if valid else (
                        "invalid_pointer" if not bool(decoded.valid[index]) else
                        "invalid_reference_geometry" if not bool(reference_valid[index]) else
                        "invalid_pointer_solver"
                    ),
                    "target_progress": target,
                    "predicted_progress": prediction,
                    "absolute_progress_error": abs(prediction - target) if prediction is not None else 1.0,
                    "endpoint": {
                        "start_xy": output.start_xy[index].cpu().tolist(),
                        "end_xy": output.end_xy[index].cpu().tolist(),
                        "target_start_xy": target_endpoint[index, 0].cpu().tolist(),
                        "target_end_xy": target_endpoint[index, 1].cpu().tolist(),
                        "mean_l2_error": float(endpoint_error[index]),
                        "peak": output.endpoint_peak[index].cpu().tolist(),
                        "entropy": output.endpoint_entropy[index].cpu().tolist(),
                        "separation": float(output.endpoint_separation[index]),
                        "posterior_mass_error": output.posterior_mass_error[index].cpu().tolist(),
                    },
                    "arc": {
                        "start_degrees": math.degrees(float(start[index])),
                        "target_start_degrees": math.degrees(float(target_start[index])),
                        "start_absolute_error_degrees": math.degrees(float(start_error[index])),
                        "range_degrees": math.degrees(float(arc[index])),
                        "target_range_degrees": math.degrees(float(target_arc[index])),
                        "range_absolute_error_degrees": math.degrees(float(arc_error[index])),
                        "endpoint_radii": radii[index].cpu().tolist(),
                    },
                    "validity": {
                        "pointer": bool(decoded.valid[index]),
                        "reference": bool(reference_valid[index]),
                        "solver": bool(solver.valid[index]),
                        "combined": valid,
                    },
                    "roi_bounds": batch["roi_bounds"][index].tolist(),
                    "augmentation_code": int(batch["augmentation_code"][index]),
                }
            )
    _require(len(rows) == len(dataset), "V5 telemetry inventory drift")
    endpoint_errors = np.asarray([row["endpoint"]["mean_l2_error"] for row in rows])
    arc_errors = np.asarray([row["arc"]["range_absolute_error_degrees"] for row in rows])
    start_errors = np.asarray([row["arc"]["start_absolute_error_degrees"] for row in rows])
    full_errors = np.asarray([row["absolute_progress_error"] for row in rows])
    metrics = {
        "samples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "coverage": sum(row["validity"]["combined"] for row in rows) / len(rows),
        "pointer_valid_fraction": sum(row["validity"]["pointer"] for row in rows) / len(rows),
        "reference_valid_fraction": sum(row["validity"]["reference"] for row in rows) / len(rows),
        "solver_valid_fraction": sum(row["validity"]["solver"] for row in rows) / len(rows),
        "full_denominator_nmae": float(full_errors.mean()),
        "endpoint_mean_l2": float(endpoint_errors.mean()),
        "endpoint_p95_l2": float(np.quantile(endpoint_errors, .95)),
        "start_angle_mae_degrees": float(start_errors.mean()),
        "arc_mae_degrees": float(arc_errors.mean()),
        "arc_p95_error_degrees": float(np.quantile(arc_errors, .95)),
    }
    return metrics, rows


def validation_record(
    protocol: Mapping[str, Any],
    fit: Sequence[PublicRecord],
    validation: Sequence[PublicRecord],
    augmentation: PhotoAugmentation,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "mode": "smoke" if args.smoke else "formal" if args.run_formal else "validate_only",
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "fit_samples": len(fit),
            "fit_groups": len({record.sample.group_id for record in fit}),
            "validation_samples": len(validation),
            "validation_groups": len({record.sample.group_id for record in validation}),
            "field_samples_read": 0,
            "confirmatory_samples_read": 0,
            "xiangmu_samples_read": 0,
        },
        "preprocessing": {
            "crop_contract": CROP_CONTRACT,
            "crop_contract_sha256": canonical_sha256(CROP_CONTRACT),
            "canonical_tight_roi": True,
            "source_roi": "clipped public ground-truth dial bbox",
            "deployment_roi": "whole supplied ROI; no second detector",
            "bbox_expansion": 1.0,
            "letterbox": False,
            "constant_or_black_border": False,
            "geometric_augmentation_border": "cv2.BORDER_REFLECT_101",
        },
        "augmentation": augmentation.as_dict(),
        "training": {
            "seed": args.seed,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "backbone": "frozen pinned PEPD",
            "trainable": "DenseTickConditionedReferenceHead only",
        },
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "manifest": sha256_file(MANIFEST),
            "manifest_protocol": sha256_file(MANIFEST_PROTOCOL),
            "backbone_checkpoint": sha256_file(BACKBONE_CHECKPOINT),
            "trainer_source": sha256_source_file(Path(__file__).resolve()),
            "head_source": sha256_source_file(
                PROJECT_ROOT / "experiments/cagh_scalemark_reference_head_v3.py"
            ),
        },
        "protocol_snapshot": protocol,
    }


def validate_or_write_roster_cache(
    root: Path,
    fit: Sequence[PublicRecord],
    validation: Sequence[PublicRecord],
) -> Path:
    """Persist only public roster identity; stochastic image features are never cached."""

    resolved = _guard_restricted_path(root, label="V5 cache root")
    _require(resolved.resolve() != Path(resolved.anchor), "broad V5 cache root rejected")
    signature = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "crop_contract_sha256": canonical_sha256(CROP_CONTRACT),
        "manifest_sha256": sha256_file(MANIFEST),
        "fit_ids_sha256": canonical_sha256([record.sample.sample_id for record in fit]),
        "validation_ids_sha256": canonical_sha256(
            [record.sample.sample_id for record in validation]
        ),
        "fit_samples": len(fit),
        "validation_samples": len(validation),
        "feature_cache": False,
        "reason": "online stochastic photo augmentation requires live feature extraction",
    }
    path = resolved / f"roster_signature_{len(fit)}_{len(validation)}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _require(existing == signature, "stale or incompatible V5 public roster cache")
    else:
        atomic_json(path, signature)
    return path


def _save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    history: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "history": history,
        "validation": validation,
        "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def run(args: argparse.Namespace) -> Path:
    _require(args.stage_a_epochs > 0 and args.stage_b_epochs > 0, "epochs must be positive")
    _require(args.batch_size > 0 and args.workers >= 0, "invalid batch/workers")
    _require(args.learning_rate > 0 and args.weight_decay >= 0, "invalid optimizer settings")
    protocol = _load_protocol()
    augmentation = augmentation_from_args(args)
    fit, validation = load_public_roster(protocol)
    if args.smoke:
        _require(args.smoke_fit_samples >= 2 and args.smoke_validation_samples >= 2,
                 "smoke inventories must be at least two")
        fit = fit[: args.smoke_fit_samples]
        validation = validation[: args.smoke_validation_samples]
        args.stage_a_epochs = 1
        args.stage_b_epochs = 1
        args.workers = 0
    roster_cache_path = validate_or_write_roster_cache(
        Path(args.cache_root), fit, validation
    )
    output = Path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / ("smoke" if args.smoke else f"seed_{args.seed}")
    )
    output = output.resolve()
    _guard_restricted_path(output, label="V5 output")
    record = validation_record(protocol, fit, validation, augmentation, args)
    record["cache"] = {
        "root": str(Path(args.cache_root).resolve()),
        "roster_signature": str(roster_cache_path),
        "features_cached": False,
    }
    validation_path = output / "validation.json"
    if args.run_formal:
        defaults = protocol["training"]
        _require(args.seed in defaults["seeds"], "formal V5 seed is outside frozen list")
        _require((args.stage_a_epochs, args.stage_b_epochs, args.batch_size) ==
                 (defaults["stage_a_epochs"], defaults["stage_b_epochs"], defaults["batch_size"]),
                 "formal V5 schedule differs from frozen protocol")
        _require(
            math.isclose(args.learning_rate, float(defaults["learning_rate"]))
            and math.isclose(args.weight_decay, float(defaults["weight_decay"])),
            "formal V5 optimizer differs from frozen protocol",
        )
        _require(augmentation.as_dict() == protocol["augmentation"],
                 "formal V5 augmentation differs from frozen protocol")
        _require(not (output / "summary.json").exists(), "formal V5 run already exists")
    atomic_json(validation_path, record)
    if args.validate_only:
        print(validation_path, flush=True)
        return validation_path

    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
        _require("NVIDIA" in torch.cuda.get_device_name(device).upper(),
                 "V5 CUDA device is not an NVIDIA GPU")
    seed_everything(args.seed)
    backbone, backbone_identity = load_pepd(BACKBONE_CHECKPOINT, device)
    head = build_head().to(device)
    fit_dataset = CanonicalTightROIDataset(
        fit, training=True, seed=args.seed, augmentation=augmentation
    )
    validation_dataset = CanonicalTightROIDataset(
        validation, training=False, seed=args.seed, augmentation=PhotoAugmentation.disabled()
    )
    started = time.time()
    history = train_head_v5(backbone, head, fit_dataset, args=args, device=device)
    metrics, telemetry = evaluate_with_telemetry(
        backbone, head, validation_dataset, args=args, device=device
    )
    telemetry_path = output / "telemetry.jsonl"
    atomic_jsonl(telemetry_path, telemetry)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_sha = _save_checkpoint(checkpoint_path, head, history, record)
    summary = {
        **record,
        "status": "complete",
        "history": history,
        "metrics": metrics,
        "backbone": backbone_identity,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "telemetry": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    print(summary_path, flush=True)
    return summary_path


if __name__ == "__main__":
    run(parse_args())
