"""Canonical-ROI paired supervision for the inner-screen PEPD backbone.

The dataset reads only public source pixels, the public dial crop, and pointer
tip/tail geometry.  Numeric readings, ranges, ScaleMark annotations, errors,
and outer-holdout rows are neither required nor returned.  Both paired views
start from the exact same tight native ROI and use reflect-border projective
augmentation; no expanded square crop, letterbox, padding, or black border is
introduced.
"""
from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    IMAGE_SIZE,
    PhotoAugmentation,
    PublicRecord,
    _apply_homography,
    _augment_geometry,
    _augment_photo,
    canonical_tight_roi,
)


PROTOCOL = "cagh_v5_canonical_pepd_pair_v1"
HEATMAP_SIZE = 64
VIEW_SEED_STRIDE = 7_919
EPOCH_SEED_STRIDE = 1_000_003
INDEX_SEED_STRIDE = 97


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _heatmap(point_xy: np.ndarray, *, sigma: float = 1.5) -> torch.Tensor:
    point = np.asarray(point_xy, dtype=np.float32).reshape(2)
    _require(bool(np.isfinite(point).all()), "PEPD pivot target is non-finite")
    # Decoder coordinates are exactly 0..63, so normalized supervision uses
    # the same closed interval rather than the historical 0..64 approximation.
    center = point * float(HEATMAP_SIZE - 1)
    yy, xx = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE].astype(np.float32)
    value = np.exp(
        -((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
        / (2.0 * float(sigma) ** 2)
    ).astype(np.float32)
    return torch.from_numpy(value[None])


def _direction(pointer: np.ndarray) -> torch.Tensor:
    points = np.asarray(pointer, dtype=np.float32).reshape(2, 2)
    delta = points[1] - points[0]
    norm = float(np.linalg.norm(delta))
    _require(np.isfinite(delta).all() and norm > 1e-8, "pointer target collapsed")
    return torch.from_numpy((delta / norm).astype(np.float32))


def _normalized_homography(first: np.ndarray, second: np.ndarray) -> torch.Tensor:
    first_value = np.asarray(first, dtype=np.float64).reshape(3, 3)
    second_value = np.asarray(second, dtype=np.float64).reshape(3, 3)
    _require(
        np.isfinite(first_value).all()
        and np.isfinite(second_value).all()
        and abs(float(np.linalg.det(first_value))) > 1e-10
        and abs(float(np.linalg.det(second_value))) > 1e-10,
        "paired canonical transform is singular",
    )
    _require(
        float(np.linalg.cond(first_value)) < 1e6
        and float(np.linalg.cond(second_value)) < 1e6,
        "paired canonical transform is ill-conditioned",
    )
    normalized = second_value @ np.linalg.inv(first_value)
    _require(abs(float(normalized[2, 2])) > 1e-10, "paired homography scale collapsed")
    normalized /= normalized[2, 2]
    _require(
        bool(np.isfinite(normalized).all()),
        "paired normalized homography is non-finite",
    )
    return torch.from_numpy(normalized.astype(np.float32))


@dataclass(frozen=True)
class CanonicalPEPDRecord:
    """Narrow public projection; it cannot carry reading/range/ScaleMark data."""

    sample_id: str
    group_id: str
    image_path: str
    dial_bbox: tuple[float, float, float, float]
    pointer_tail: tuple[float, float]
    pointer_tip: tuple[float, float]
    group_weight: float


def _is_reparse(path: Path) -> bool:
    attributes = getattr(path.stat(), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _public_image(path: Any, root: Path) -> str:
    value = Path(str(path))
    _require(value.is_absolute(), "PEPD image path must be absolute")
    declared_root = Path(root)
    _require(declared_root.is_absolute(), "PEPD public root must be absolute")
    _require(
        not ({part.casefold() for part in value.parts} & {"field", "test", "sealed", "confirmatory"}),
        "PEPD image path entered restricted namespace",
    )
    # Check the lexical path before resolve(), otherwise a symlink component is
    # erased and can no longer be audited.
    lexical_value = Path(os.path.abspath(os.fspath(value)))
    lexical_root = Path(os.path.abspath(os.fspath(declared_root)))
    _require(lexical_value.is_relative_to(lexical_root), "PEPD image escaped public root")
    cursor = lexical_value
    while True:
        _require(cursor.exists(), "PEPD image lexical component is missing")
        _require(not cursor.is_symlink() and not _is_reparse(cursor), "PEPD image has reparse component")
        if cursor == lexical_root:
            break
        _require(cursor != cursor.parent, "PEPD lexical traversal escaped public root")
        cursor = cursor.parent
    for ancestor in lexical_root.parents:
        _require(
            not ancestor.is_symlink() and not _is_reparse(ancestor),
            "PEPD public-root ancestor has reparse component",
        )
    resolved = lexical_value.resolve(strict=True)
    public_root = lexical_root.resolve(strict=True)
    _require(resolved.is_relative_to(public_root), "PEPD image escaped public root")
    _require(resolved.is_file(), "PEPD public image is not a file")
    return os.fspath(resolved)


def _narrow_record(record: PublicRecord, public_image_root: Path) -> CanonicalPEPDRecord:
    _require(record.partition == "inner_train", "PEPD record is not inner_train")
    sample = record.sample
    _require(
        str(getattr(sample, "dataset", "")) == "SyncG"
        and str(getattr(sample, "split", "")) == "train",
        "PEPD record is outside public SyncG/train",
    )
    bbox = tuple(map(float, getattr(sample, "dial_bbox")))
    tail = tuple(map(float, getattr(sample, "pointer_tail")))
    tip = tuple(map(float, getattr(sample, "pointer_tip")))
    _require(len(bbox) == 4 and len(tail) == len(tip) == 2, "PEPD geometry shape drift")
    _require(
        np.isfinite(np.asarray((*bbox, *tail, *tip), dtype=np.float64)).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1],
        "PEPD geometry is invalid",
    )
    weight = float(record.group_weight)
    _require(math.isfinite(weight) and weight > 0.0, "PEPD group weight invalid")
    return CanonicalPEPDRecord(
        sample_id=str(getattr(sample, "sample_id")),
        group_id=str(getattr(sample, "group_id")),
        image_path=_public_image(getattr(sample, "image_path"), public_image_root),
        dial_bbox=bbox,
        pointer_tail=tail,
        pointer_tip=tip,
        group_weight=weight,
    )


class CanonicalPEPDPairDataset(Dataset):
    """Return two deterministic augmented views of one canonical tight ROI."""

    def __init__(
        self,
        records: Sequence[PublicRecord],
        *,
        seed: int,
        augmentation: PhotoAugmentation,
        public_image_root: Path,
    ) -> None:
        _require(Path(public_image_root).is_absolute(), "public image root must be absolute")
        self.records = tuple(
            _narrow_record(record, Path(public_image_root)) for record in records
        )
        self.seed = int(seed)
        self.augmentation = augmentation
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.records), "canonical PEPD pair dataset is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _view(
        self,
        crop: np.ndarray,
        pointer: np.ndarray,
        *,
        index: int,
        view: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        seed = (
            self.seed
            + self.epoch * EPOCH_SEED_STRIDE
            + int(index) * INDEX_SEED_STRIDE
            + int(view) * VIEW_SEED_STRIDE
        )
        rng = np.random.default_rng(seed)
        image, forward, code = _augment_geometry(
            crop.copy(), pointer.copy(), rng, self.augmentation
        )
        transformed = _apply_homography(pointer, forward)
        image, photo_code = _augment_photo(image, rng, self.augmentation)
        code |= int(photo_code)
        _require(
            bool(np.isfinite(transformed).all())
            and bool(((transformed >= 0.0) & (transformed <= 1.0)).all()),
            "augmented pointer escaped canonical ROI",
        )
        return image, transformed, forward, code

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        image = cv2.imread(
            record.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"failed to read public image: {record.sample_id}")
        crop, original_to_normalized, _, bounds = canonical_tight_roi(
            image, record.dial_bbox
        )
        pointer = _apply_homography(
            np.asarray([record.pointer_tail, record.pointer_tip], dtype=np.float32),
            original_to_normalized,
        )
        first_image, first_pointer, first_forward, first_code = self._view(
            crop, pointer, index=index, view=0
        )
        second_image, second_pointer, second_forward, second_code = self._view(
            crop, pointer, index=index, view=1
        )
        return {
            "image": normalized_rgb_tensor(first_image),
            "paired_image": normalized_rgb_tensor(second_image),
            "heatmap": _heatmap(first_pointer[0]),
            "paired_heatmap": _heatmap(second_pointer[0]),
            "direction": _direction(first_pointer),
            "paired_direction": _direction(second_pointer),
            "homography": _normalized_homography(first_forward, second_forward),
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "group_weight": torch.tensor(record.group_weight, dtype=torch.float32),
            "augmentation_code": torch.tensor(first_code, dtype=torch.int32),
            "paired_augmentation_code": torch.tensor(second_code, dtype=torch.int32),
            "roi_bounds": torch.tensor(bounds, dtype=torch.int32),
        }


def tensor_receipt(row: dict[str, Any]) -> str:
    """Receipt over every tensor consumed by the current paired PEPD loss.

    Sample/group identity, audit-only ROI bounds, and the currently unused group
    weight are bound separately by the frozen manifest and runner receipts.
    """

    digest = hashlib.sha256()
    for name in (
        "image",
        "paired_image",
        "heatmap",
        "paired_heatmap",
        "direction",
        "paired_direction",
        "homography",
        "augmentation_code",
        "paired_augmentation_code",
    ):
        value = torch.as_tensor(row[name]).detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "CanonicalPEPDPairDataset",
    "PROTOCOL",
    "tensor_receipt",
]
