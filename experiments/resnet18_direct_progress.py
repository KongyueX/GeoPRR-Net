"""Matched ResNet-18 direct-regression baseline for the CAGH paper.

The baseline consumes the same canonical tight dial ROI as the plain paper
runner and predicts only normalized progress in ``[0, 1]``.  It is deliberately
small and conventional: a torchvision ResNet-18 with one sigmoid regression
output.  It uses the same ImageNet-1K V1 initialization and frozen
photo/projective augmentation contract as CAGH-V5, so differences are not
caused by weaker initialization or training perturbations.

The module also materializes the fixed scene-stem-disjoint sensitivity split
and label-free ROI manifests.  Ground truth is read only by the training
dataset and scorer-side split preparation; prediction accepts the four-field
label-free manifest used by ``run_cagh_v5_plain_paper_batch.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import IMAGENET_WEIGHTS
from experiments.run_cagh_v5_solver_gated_screen import augmentation_config
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
    encode_lossless_png,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments import robustness_degradations


PROTOCOL: Final[str] = "syncg_resnet18_direct_progress_v2"
SCENE_SPLIT_PROTOCOL: Final[str] = "syncg_scene_stem_disjoint_clean_v1"
IMAGE_SIZE: Final[int] = 256
IMAGENET_INITIALIZATION: Final[str] = (
    "torchvision.ResNet18_Weights.IMAGENET1K_V1"
)
DEFAULT_EPOCHS: Final[int] = 30
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
DEFAULT_MANIFEST: Final[Path] = Path("artifacts/manifests/syncg_train.jsonl")
DEFAULT_COMPOSITIONAL_SPLIT: Final[Path] = Path(
    "artifacts/runs/syncg_segmentation/split.json"
)
DEFAULT_SCENE_PROTOCOL: Final[Path] = Path(
    "experiments/syncg_scene_disjoint_clean_protocol.json"
)


class DirectProgressError(ValueError):
    """The manifest, split, checkpoint, or direct-regression output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectProgressError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _scene_stem(metadata: Mapping[str, Any], sample_id: str) -> str:
    scene_name = metadata.get("scene_name")
    _require(isinstance(scene_name, str) and bool(scene_name), f"{sample_id}: no scene_name")
    stem = Path(scene_name).stem
    _require(bool(stem), f"{sample_id}: empty scene stem")
    return stem


@dataclass(frozen=True, slots=True)
class DirectSample:
    sample_id: str
    group_id: str
    scene_stem: str
    image_path: Path
    dial_bbox: tuple[float, float, float, float]
    normalized_target: float
    protected_points_xy: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ManifestIdentity:
    sample_id: str
    group_id: str
    scene_stem: str


@dataclass(frozen=True, slots=True)
class SplitRoster:
    protocol: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    scene_disjoint: bool


def _manifest_rows(manifest_path: Path):
    """Yield decoded rows without interpreting any supervision fields."""

    manifest = Path(manifest_path).resolve()
    _require(manifest.is_file(), f"manifest does not exist: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DirectProgressError(
                    f"manifest line {line_number} is invalid JSON"
                ) from exc
            _require(isinstance(row, Mapping), f"manifest line {line_number} is not an object")
            yield line_number, row


def _sample_identity(row: Mapping[str, Any]) -> ManifestIdentity:
    sample_id = row.get("sample_id")
    _require(isinstance(sample_id, str) and bool(sample_id), "empty sample_id")
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), f"{sample_id}: metadata is missing")
    group_id = row.get("group_id")
    _require(isinstance(group_id, str) and bool(group_id), f"{sample_id}: no group_id")
    return ManifestIdentity(
        sample_id=sample_id,
        group_id=group_id,
        scene_stem=_scene_stem(metadata, sample_id),
    )


def load_manifest_identities(manifest_path: Path) -> tuple[ManifestIdentity, ...]:
    """Load IDs and scene metadata only; target fields are never accessed."""

    identities: list[ManifestIdentity] = []
    seen: set[str] = set()
    for _line_number, row in _manifest_rows(manifest_path):
        identity = _sample_identity(row)
        _require(identity.sample_id not in seen, f"duplicate sample_id: {identity.sample_id}")
        seen.add(identity.sample_id)
        identities.append(identity)
    _require(bool(identities), "manifest is empty")
    return tuple(identities)


def _training_protected_points(
    metadata: Mapping[str, Any], *, sample_id: str
) -> tuple[tuple[float, float], ...]:
    """Return the same ScaleMark/pointer landmarks protected by CAGH augmentation."""

    keypoints = metadata.get("keypoints")
    _require(
        isinstance(keypoints, Sequence) and not isinstance(keypoints, (str, bytes)),
        f"{sample_id}: keypoints are missing",
    )
    pointers = [
        value
        for value in keypoints
        if isinstance(value, Mapping)
        and str(value.get("type") or "").casefold() == "pointer"
    ]
    scale_marks = [
        value
        for value in keypoints
        if isinstance(value, Mapping)
        and str(value.get("type") or "").casefold() == "scalemark"
    ]
    _require(len(pointers) == 1, f"{sample_id}: expected one Pointer annotation")
    _require(len(scale_marks) == 1, f"{sample_id}: expected one ScaleMark annotation")
    marks = scale_marks[0].get("all_kp")
    _require(
        isinstance(marks, Sequence)
        and not isinstance(marks, (str, bytes))
        and 7 <= len(marks) <= 36,
        f"{sample_id}: invalid ScaleMark inventory",
    )
    raw_points = [*marks, pointers[0].get("origin_kp"), pointers[0].get("outside_kp")]
    points: list[tuple[float, float]] = []
    for index, value in enumerate(raw_points):
        _require(
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) >= 2,
            f"{sample_id}: invalid protected point {index}",
        )
        point = (float(value[0]), float(value[1]))
        _require(
            math.isfinite(point[0]) and math.isfinite(point[1]),
            f"{sample_id}: non-finite protected point {index}",
        )
        points.append(point)
    return tuple(points)


def _direct_sample(row: Mapping[str, Any], *, manifest_root: Path) -> DirectSample:
    """Interpret target/image fields for one already-authorized fit row only."""

    identity = _sample_identity(row)
    sample_id = identity.sample_id
    metadata = row["metadata"]
    bbox = metadata.get("dial_bbox")
    _require(
        isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes))
        and len(bbox) >= 4,
        f"{sample_id}: dial_bbox is missing",
    )
    bbox_values = tuple(float(value) for value in bbox[:4])
    _require(
        all(math.isfinite(value) for value in bbox_values)
        and bbox_values[2] > bbox_values[0]
        and bbox_values[3] > bbox_values[1],
        f"{sample_id}: invalid dial_bbox",
    )
    try:
        ground_truth = float(row["ground_truth"])
        scale_start = float(row["scale_start"])
        scale_end = float(row["scale_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectProgressError(f"{sample_id}: invalid target fields") from exc
    _require(
        all(math.isfinite(value) for value in (ground_truth, scale_start, scale_end))
        and scale_end > scale_start,
        f"{sample_id}: invalid scale range",
    )
    target = (ground_truth - scale_start) / (scale_end - scale_start)
    _require(-1e-9 <= target <= 1.0 + 1e-9, f"{sample_id}: target outside [0,1]")
    image_text = row.get("image_path")
    _require(isinstance(image_text, str) and bool(image_text), f"{sample_id}: no image_path")
    image_path = Path(image_text)
    if not image_path.is_absolute():
        image_path = manifest_root / image_path
    return DirectSample(
        sample_id=sample_id,
        group_id=identity.group_id,
        scene_stem=identity.scene_stem,
        image_path=image_path.resolve(),
        dial_bbox=bbox_values,
        normalized_target=min(1.0, max(0.0, target)),
        protected_points_xy=_training_protected_points(
            metadata, sample_id=sample_id
        ),
    )


def load_syncg_samples(manifest_path: Path) -> tuple[DirectSample, ...]:
    """Load labeled rows; intended for tests/scoring helpers, not ``train``."""

    manifest = Path(manifest_path).resolve()
    samples: list[DirectSample] = []
    seen: set[str] = set()
    for _line_number, row in _manifest_rows(manifest):
        sample = _direct_sample(row, manifest_root=manifest.parent)
        sample_id = sample.sample_id
        _require(sample_id not in seen, f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        samples.append(sample)
    _require(bool(samples), "manifest is empty")
    return tuple(samples)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    protocol: str
    train: tuple[DirectSample, ...]
    validation: tuple[DirectSample, ...]
    scene_disjoint: bool


def load_split_roster(split_path: Path) -> SplitRoster:
    """Load explicit IDs before any labeled manifest rows are interpreted."""

    source = Path(split_path).resolve()
    _require(source.is_file(), f"split does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectProgressError(f"cannot read split: {source}") from exc
    _require(isinstance(value, Mapping), "split root is not an object")
    train_values = value.get("train_sample_ids")
    validation_values = value.get("validation_sample_ids")
    _require(isinstance(train_values, list) and bool(train_values),
             "split has no train_sample_ids")
    _require(isinstance(validation_values, list) and bool(validation_values),
             "split has no validation_sample_ids")
    train_ids = tuple(str(item) for item in train_values)
    validation_ids = tuple(str(item) for item in validation_values)
    _require(len(train_ids) == len(set(train_ids)), "duplicate train sample IDs")
    _require(len(validation_ids) == len(set(validation_ids)), "duplicate validation sample IDs")
    _require(not set(train_ids) & set(validation_ids), "train and validation IDs overlap")
    return SplitRoster(
        protocol=str(value.get("protocol") or "explicit_sample_id_split"),
        train_ids=train_ids,
        validation_ids=validation_ids,
        scene_disjoint=bool(value.get("scene_disjoint", False)),
    )


def load_training_samples(
    manifest_path: Path,
    split_path: Path,
) -> tuple[tuple[DirectSample, ...], SplitRoster]:
    """Load targets and image paths for fit IDs only.

    Holdout rows are used solely to confirm ID membership/count.  Their target,
    image path, bbox, and scene metadata are never accessed here.
    """

    manifest = Path(manifest_path).resolve()
    roster = load_split_roster(split_path)
    train_set = set(roster.train_ids)
    validation_set = set(roster.validation_ids)
    expected_ids = train_set | validation_set
    fit_by_id: dict[str, DirectSample] = {}
    seen: set[str] = set()
    for _line_number, row in _manifest_rows(manifest):
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and bool(sample_id), "empty sample_id")
        _require(sample_id not in seen, f"duplicate sample_id: {sample_id}")
        _require(sample_id in expected_ids, f"manifest ID absent from split: {sample_id}")
        seen.add(sample_id)
        if sample_id in train_set:
            fit_by_id[sample_id] = _direct_sample(row, manifest_root=manifest.parent)
        else:
            # Strict holdout boundary: do not inspect any other field in this row.
            _require(sample_id in validation_set, f"unexpected split membership: {sample_id}")
    _require(seen == expected_ids, "split and manifest sample rosters differ")
    _require(set(fit_by_id) == train_set, "fit sample loading is incomplete")
    return tuple(fit_by_id[sample_id] for sample_id in roster.train_ids), roster


def assign_split(
    samples: Sequence[DirectSample],
    split_value: Mapping[str, Any],
) -> SplitAssignment:
    """Bind an explicit ID split and verify its complete partition semantics."""

    all_by_id = {sample.sample_id: sample for sample in samples}
    _require(len(all_by_id) == len(samples), "sample IDs are not unique")
    train_values = split_value.get("train_sample_ids")
    validation_values = split_value.get("validation_sample_ids")
    _require(isinstance(validation_values, list) and bool(validation_values),
             "split has no validation_sample_ids")
    validation_ids = tuple(str(value) for value in validation_values)
    if train_values is None:
        validation_set = set(validation_ids)
        train_ids = tuple(
            sample.sample_id for sample in samples if sample.sample_id not in validation_set
        )
    else:
        _require(isinstance(train_values, list) and bool(train_values),
                 "split train_sample_ids is invalid")
        train_ids = tuple(str(value) for value in train_values)
    _require(len(train_ids) == len(set(train_ids)), "duplicate train sample IDs")
    _require(len(validation_ids) == len(set(validation_ids)), "duplicate validation sample IDs")
    train_set, validation_set = set(train_ids), set(validation_ids)
    _require(not train_set & validation_set, "train and validation sample IDs overlap")
    _require(train_set | validation_set == set(all_by_id), "split does not partition the manifest")
    train = tuple(all_by_id[sample_id] for sample_id in train_ids)
    validation = tuple(all_by_id[sample_id] for sample_id in validation_ids)
    scene_overlap = {sample.scene_stem for sample in train} & {
        sample.scene_stem for sample in validation
    }
    declared_scene_disjoint = bool(split_value.get("scene_disjoint", False))
    if declared_scene_disjoint:
        _require(not scene_overlap, "split declares scene-disjoint but scenes overlap")
    protocol = str(split_value.get("protocol") or "explicit_sample_id_split")
    return SplitAssignment(protocol, train, validation, declared_scene_disjoint)


def load_explicit_split(
    manifest_path: Path,
    split_path: Path,
) -> SplitAssignment:
    samples = load_syncg_samples(manifest_path)
    source = Path(split_path).resolve()
    _require(source.is_file(), f"split does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectProgressError(f"cannot read split: {source}") from exc
    _require(isinstance(value, Mapping), "split root is not an object")
    return assign_split(samples, value)


def build_scene_disjoint_split(
    *,
    manifest_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Expand the frozen scene roster into an explicit, scorer-ready ID split."""

    manifest = Path(manifest_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    _require(protocol_file.is_file(), f"scene protocol does not exist: {protocol_file}")
    value = json.loads(protocol_file.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "scene protocol root is not an object")
    _require(value.get("protocol") == SCENE_SPLIT_PROTOCOL, "unexpected scene protocol")
    expected_manifest_sha = value.get("source_manifest_sha256")
    _require(
        isinstance(expected_manifest_sha, str)
        and _sha256_file(manifest) == expected_manifest_sha,
        "scene protocol does not match the SyncG manifest",
    )
    scene_values = value.get("validation_scene_stems")
    _require(isinstance(scene_values, list) and bool(scene_values), "no validation scenes")
    validation_scenes = tuple(str(item) for item in scene_values)
    _require(len(validation_scenes) == len(set(validation_scenes)), "duplicate validation scenes")
    identities = load_manifest_identities(manifest)
    all_scenes = {sample.scene_stem for sample in identities}
    _require(set(validation_scenes) <= all_scenes, "validation scene absent from manifest")
    validation_set = set(validation_scenes)
    train_ids = sorted(
        sample.sample_id for sample in identities if sample.scene_stem not in validation_set
    )
    validation_ids = sorted(
        sample.sample_id for sample in identities if sample.scene_stem in validation_set
    )
    train_scenes = sorted(all_scenes - validation_set)
    expected = value.get("expected")
    _require(isinstance(expected, Mapping), "scene protocol has no expected counts")
    observed = {
        "all_samples": len(identities),
        "all_scenes": len(all_scenes),
        "fit_samples": len(train_ids),
        "fit_scenes": len(train_scenes),
        "validation_samples": len(validation_ids),
        "validation_scenes": len(validation_scenes),
        "fit_validation_scene_overlap": len(set(train_scenes) & validation_set),
        "fit_sample_ids_sha256": _canonical_sha256(train_ids),
        "validation_sample_ids_sha256": _canonical_sha256(validation_ids),
    }
    _require(observed == dict(expected), "scene-disjoint split identity drift")
    return {
        "schema_version": 1,
        "protocol": SCENE_SPLIT_PROTOCOL,
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256_file(manifest),
        "scene_key": value["scene_key"],
        "scene_disjoint": True,
        "train_scene_stems": train_scenes,
        "validation_scene_stems": list(validation_scenes),
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "identity": observed,
    }


def materialize_scene_disjoint_split(
    *, manifest_path: Path, protocol_path: Path, output_path: Path
) -> dict[str, Any]:
    value = build_scene_disjoint_split(
        manifest_path=manifest_path, protocol_path=protocol_path
    )
    _write_json(output_path, value)
    return value


@dataclass(frozen=True, slots=True)
class ValidationInput:
    sample_id: str
    image_path: Path
    dial_bbox: tuple[float, float, float, float]


def load_validation_inputs(
    manifest_path: Path,
    split_path: Path,
) -> tuple[tuple[ValidationInput, ...], SplitRoster]:
    """Load holdout pixels/bboxes after training, without accessing targets."""

    manifest = Path(manifest_path).resolve()
    roster = load_split_roster(split_path)
    expected_ids = set(roster.train_ids) | set(roster.validation_ids)
    validation_set = set(roster.validation_ids)
    values: dict[str, ValidationInput] = {}
    seen: set[str] = set()
    for _line_number, row in _manifest_rows(manifest):
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and bool(sample_id), "empty sample_id")
        _require(sample_id not in seen, f"duplicate sample_id: {sample_id}")
        _require(sample_id in expected_ids, f"manifest ID absent from split: {sample_id}")
        seen.add(sample_id)
        if sample_id not in validation_set:
            continue
        metadata = row.get("metadata")
        _require(isinstance(metadata, Mapping), f"{sample_id}: metadata is missing")
        bbox = metadata.get("dial_bbox")
        _require(
            isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes))
            and len(bbox) >= 4,
            f"{sample_id}: dial_bbox is missing",
        )
        bbox_values = tuple(float(item) for item in bbox[:4])
        _require(
            all(math.isfinite(item) for item in bbox_values)
            and bbox_values[2] > bbox_values[0]
            and bbox_values[3] > bbox_values[1],
            f"{sample_id}: invalid dial_bbox",
        )
        image_text = row.get("image_path")
        _require(isinstance(image_text, str) and bool(image_text), f"{sample_id}: no image_path")
        image_path = Path(image_text)
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        values[sample_id] = ValidationInput(
            sample_id=sample_id,
            image_path=image_path.resolve(),
            dial_bbox=bbox_values,
        )
    _require(seen == expected_ids, "split and manifest sample rosters differ")
    _require(set(values) == validation_set, "validation input loading is incomplete")
    return tuple(values[sample_id] for sample_id in roster.validation_ids), roster


def materialize_label_free_inputs(
    *, manifest_path: Path, split_path: Path, output_root: Path
) -> dict[str, Any]:
    """Create the same four-field tight-ROI input used by the plain runner."""

    validation_inputs, roster = load_validation_inputs(manifest_path, split_path)
    root = Path(output_root).resolve()
    _require(not root.exists(), f"output root already exists: {root}")
    roi_root = root / "rois"
    roi_root.mkdir(parents=True)
    manifest_out = root / "input_manifest.jsonl"
    with manifest_out.open("wb") as stream:
        for sample in validation_inputs:
            image = cv2.imread(
                str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            _require(image is not None, f"{sample.sample_id}: source image decode failed")
            roi, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
            payload = encode_lossless_png(roi)
            roi_path = roi_root / f"{sample.sample_id}.png"
            roi_path.write_bytes(payload)
            row = {
                "sample_id": sample.sample_id,
                "roi_path": roi_path.relative_to(root).as_posix(),
                "roi_png_sha256": hashlib.sha256(payload).hexdigest(),
                "roi_pixel_sha256": canonical_roi_pixel_sha256(roi),
            }
            stream.write(_canonical_json_bytes(row) + b"\n")
    summary = {
        "schema_version": 1,
        "protocol": f"{PROTOCOL}_label_free_inputs",
        "split_protocol": roster.protocol,
        "scene_disjoint": roster.scene_disjoint,
        "samples": len(validation_inputs),
        "input_manifest": str(manifest_out),
        "input_manifest_sha256": _sha256_file(manifest_out),
        "contains_targets": False,
    }
    _write_json(root / "preparation_summary.json", summary)
    return summary


def matched_cagh_augmentation() -> PhotoAugmentation:
    """Build the exact frozen augmentation configuration used by CAGH-V5."""

    value = PhotoAugmentation(**augmentation_config())
    value.validate()
    return value


class DirectProgressDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        training: bool,
        seed: int,
        image_size: int = IMAGE_SIZE,
        augmentation: PhotoAugmentation | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.training = bool(training)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.augmentation = (
            matched_cagh_augmentation()
            if augmentation is None and self.training
            else augmentation or PhotoAugmentation.disabled()
        )
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "direct-progress dataset is empty")
        _require(self.image_size >= 32, "image size is too small")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=self.image_size)
        if self.training:
            _require(
                bool(sample.protected_points_xy),
                f"{sample.sample_id}: training augmentation landmarks are missing",
            )
            left, top, right, bottom = bounds
            scale = np.asarray(
                [float(right - left), float(bottom - top)], dtype=np.float32
            )
            protected = (
                np.asarray(sample.protected_points_xy, dtype=np.float32)
                - np.asarray([left, top], dtype=np.float32)
            ) / scale
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
            roi, _forward, _geometry_code = _cagh_augment_geometry(
                roi, protected, rng, self.augmentation
            )
            roi, _photo_code = _cagh_augment_photo(
                roi, rng, self.augmentation
            )
        return (
            normalized_rgb_tensor(roi),
            torch.tensor(sample.normalized_target, dtype=torch.float32),
        )


class ResNet18DirectProgress(nn.Module):
    """ImageNet-initialized ResNet-18 followed by one sigmoid progress output."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        self.imagenet_pretrained = bool(imagenet_pretrained)
        self.backbone = resnet18(
            weights=IMAGENET_WEIGHTS if self.imagenet_pretrained else None
        )
        features = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Linear(features, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(image).squeeze(1))


def _configure_reproducibility(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _loader(
    dataset: DirectProgressDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    cuda: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
        persistent_workers=False,
        generator=generator,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0
    use_amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if use_amp else torch.bfloat16,
                enabled=use_amp,
            ):
                predictions = model(images)
                loss = F.smooth_l1_loss(predictions, targets, beta=0.05)
            if training:
                _require(scaler is not None, "training scaler is missing")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        count = int(targets.numel())
        total_loss += float(loss.detach().cpu()) * count
        total_absolute_error += float(
            torch.abs(predictions.detach() - targets).sum().cpu()
        )
        total_samples += count
    _require(total_samples > 0, "epoch produced no samples")
    return {
        "loss": total_loss / total_samples,
        "nmae": total_absolute_error / total_samples,
        "samples": float(total_samples),
    }


def train(
    *,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train a fixed terminal iterate without opening holdout images or labels."""

    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid training sizes")
    _require(learning_rate > 0.0 and weight_decay >= 0.0, "invalid optimizer values")
    fit_samples, roster = load_training_samples(manifest_path, split_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    fit_dataset = DirectProgressDataset(fit_samples, training=True, seed=seed)
    model = ResNet18DirectProgress().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        fit_dataset.set_epoch(epoch_index)
        train_loader = _loader(
            fit_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + epoch_index,
            cuda=device.type == "cuda",
        )
        train_metrics = _epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        scheduler.step()
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": "torchvision_resnet18_imagenet1k_v1_sigmoid_scalar",
        "pretrained_weights": IMAGENET_INITIALIZATION,
        "image_size": IMAGE_SIZE,
        "seed": int(seed),
        "split_protocol": roster.protocol,
        "scene_disjoint": roster.scene_disjoint,
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "train_sample_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "holdout_sample_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
        "holdout_access_during_training": "IDs/count only; no target, bbox, image path, or image",
        "epochs": int(epochs),
        "checkpoint_selection": "terminal_fixed_epoch",
        "loss": "smooth_l1_beta_0.05",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
        "augmentation": {
            "name": "matched_cagh_v5_photo_and_geometry",
            "configuration": augmentation_config(),
            "protected_landmarks": (
                "ordered ScaleMark points plus Pointer origin/outside; "
                "used only to keep label-preserving transforms in frame"
            ),
        },
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "history": history,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "method": f"resnet18_direct_seed_{seed}",
        "terminal_train_nmae": history[-1]["train"]["nmae"],
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "scene_disjoint": roster.scene_disjoint,
    }


def load_checkpoint_predictor(
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "checkpoint protocol mismatch")
    _require(
        checkpoint.get("architecture")
        == "torchvision_resnet18_imagenet1k_v1_sigmoid_scalar"
        and checkpoint.get("pretrained_weights") == IMAGENET_INITIALIZATION,
        "checkpoint is not the matched ImageNet-initialized ResNet-18",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "checkpoint model state is missing")
    seed = int(checkpoint["seed"])
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    # State loading must not fetch initialization weights a second time.
    model = ResNet18DirectProgress(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        tensors = [
            normalized_rgb_tensor(direct_resize_whole_roi(image, size=IMAGE_SIZE))
            for image in images_bgr
        ]
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        results = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in results),
            "model returned invalid progress",
        )
        return results

    return f"resnet18_direct_seed_{seed}", predict


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
    predictor_loader: Callable[..., tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]]
    = load_checkpoint_predictor,
) -> int:
    selected_conditions = tuple(conditions)
    _require(bool(selected_conditions), "no evaluation conditions selected")
    _require(
        len(selected_conditions) == len(set(selected_conditions))
        and set(selected_conditions) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    method, predictor = predictor_loader(checkpoint_path, device_name=device_name)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            conditioned: list[np.ndarray] = []
            condition_hashes: list[str] = []
            for condition in selected_conditions:
                image, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                conditioned.append(image)
                condition_hashes.append(canonical_roi_pixel_sha256(image))
            progress_values: list[float | None]
            failure_codes: list[str | None]
            try:
                predicted = predictor(conditioned)
                _require(len(predicted) == len(conditioned), "prediction batch length mismatch")
                progress_values = [float(value) for value in predicted]
                _require(
                    all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in progress_values
                    ),
                    "prediction outside [0,1]",
                )
                failure_codes = [None] * len(conditioned)
            except Exception as exc:  # Preserve every sample-condition row as a failure.
                progress_values = [None] * len(conditioned)
                failure_codes = [f"model_exception:{type(exc).__name__}"] * len(conditioned)
            for condition, condition_hash, progress, failure in zip(
                selected_conditions,
                condition_hashes,
                progress_values,
                failure_codes,
                strict=True,
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
                _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return count


def score_predictions(
    *,
    prediction_paths: Sequence[Path],
    manifest_path: Path,
    validation_ids_path: Path,
    output_path: Path,
    seeds: Sequence[int],
    conditions: Sequence[str],
    bootstrap_replicates: int = 2_000,
) -> dict[str, Any]:
    """Score direct predictions with the established failure-penalized scorer."""

    from experiments import score_cagh_v5_plain_paper_batch as scorer

    methods = tuple(f"resnet18_direct_seed_{int(seed)}" for seed in seeds)
    _require(len(methods) == len(set(methods)) and bool(methods), "seed roster is invalid")
    _require(len(methods) in (1, 3), "score expects one or three ResNet seeds")
    selected_conditions = tuple(conditions)
    validation_ids = scorer.load_validation_ids(validation_ids_path)
    targets = scorer.load_targets(manifest_path, validation_ids)
    expected_groups = len({target.group_id for target in targets})
    value = scorer.score(
        predictions_path=tuple(prediction_paths),
        manifest_path=manifest_path,
        validation_ids_path=validation_ids_path,
        methods=methods,
        conditions=selected_conditions,
        full_seed_methods=methods if len(methods) == 3 else (),
        expected_samples=len(validation_ids),
        expected_groups=expected_groups,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=ROBUSTNESS_SEED,
    )
    _write_json(output_path, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scene = subparsers.add_parser("materialize-scene-split")
    scene.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    scene.add_argument("--protocol", type=Path, default=DEFAULT_SCENE_PROTOCOL)
    scene.add_argument("--output", type=Path, required=True)

    inputs = subparsers.add_parser("prepare-inputs")
    inputs.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    inputs.add_argument("--split", type=Path, required=True)
    inputs.add_argument("--output-root", type=Path, required=True)

    training = subparsers.add_parser("train")
    training.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--split", type=Path, default=DEFAULT_COMPOSITIONAL_SPLIT)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    training.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)

    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--conditions", choices=("all", "clean"), default="all")

    scoring = subparsers.add_parser("score")
    scoring.add_argument("--predictions", type=Path, nargs="+", required=True)
    scoring.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    scoring.add_argument("--validation-ids", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--seeds", type=int, nargs="+", required=True)
    scoring.add_argument("--conditions", choices=("all", "clean"), default="all")
    scoring.add_argument("--bootstrap-replicates", type=int, default=2_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize-scene-split":
        value = materialize_scene_disjoint_split(
            manifest_path=args.manifest,
            protocol_path=args.protocol,
            output_path=args.output,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "fit_samples": len(value["train_sample_ids"]),
            "validation_samples": len(value["validation_sample_ids"]),
            "scene_overlap": value["identity"]["fit_validation_scene_overlap"],
        }
    elif args.command == "prepare-inputs":
        result = materialize_label_free_inputs(
            manifest_path=args.manifest,
            split_path=args.split,
            output_root=args.output_root,
        )
    elif args.command == "train":
        result = train(
            manifest_path=args.manifest,
            split_path=args.split,
            output_path=args.output,
            seed=args.seed,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.command == "predict":
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        count = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            conditions=conditions,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "rows": count,
            "conditions": list(conditions),
        }
    else:
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        value = score_predictions(
            prediction_paths=args.predictions,
            manifest_path=args.manifest,
            validation_ids_path=args.validation_ids,
            output_path=args.output,
            seeds=args.seeds,
            conditions=conditions,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "samples": value["identity"]["samples"],
            "methods": value["identity"]["methods"],
            "conditions": value["identity"]["conditions"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEEDS",
    "DirectProgressDataset",
    "DirectProgressError",
    "DirectSample",
    "ResNet18DirectProgress",
    "SplitAssignment",
    "assign_split",
    "build_scene_disjoint_split",
    "load_checkpoint_predictor",
    "load_explicit_split",
    "load_syncg_samples",
    "matched_cagh_augmentation",
    "main",
    "materialize_label_free_inputs",
    "materialize_scene_disjoint_split",
    "run_prediction",
    "score_predictions",
    "train",
]
