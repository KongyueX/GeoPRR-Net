"""Shared utilities for a pinned, external VDN retraining baseline.

The GPL-3.0 VDN source remains an ignored external checkout.  This module
does not copy that implementation; it verifies one upstream commit and loads
its model definition dynamically.  Dataset adaptation, training, and paper
metrics are implemented locally so the external source stays unmodified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from experiments.datasets import syncg_sample_ids_sha256


PROJECT_DIR = Path(__file__).resolve().parents[1]
VDN_REPOSITORY = "https://github.com/DrawZeroPoint/VectorDetectionNetwork.git"
VDN_PINNED_COMMIT = "68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb"
VDN_PROTOCOL = "vdn_architecture_syncg_retraining_v1"
VDN_RESNET18_PRETRAINED_URL = (
    "https://download.pytorch.org/models/resnet18-5c106cde.pth"
)
IMAGENET_NORMALIZATION = (
    np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
    np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
)


@dataclass(frozen=True)
class VDNSample:
    sample_id: str
    group_id: str
    dataset: str
    split: str
    image_path: str
    dial_bbox: tuple[float, float, float, float]
    pointer_tip: tuple[float, float]
    pointer_tail: tuple[float, float]
    ground_truth: float
    scale_start: float
    scale_end: float
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_vdn_source(path: Path) -> str:
    path = path.resolve()
    model_path = path / "libs" / "models" / "vdn_model.py"
    license_path = path / "LICENSE"
    if not model_path.is_file() or not license_path.is_file():
        raise FileNotFoundError(
            f"VDN source is incomplete at {path}; clone {VDN_REPOSITORY}"
        )
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if commit != VDN_PINNED_COMMIT:
        raise ValueError(
            f"VDN source commit is {commit}, expected {VDN_PINNED_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("formal VDN source checkout contains tracked modifications")
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    if "GNU GENERAL PUBLIC LICENSE" not in license_text or "Version 3" not in license_text:
        raise ValueError("VDN checkout does not contain the expected GPL-3.0 notice")
    return commit


def _load_vdn_model_module(vdn_source: Path) -> ModuleType:
    verify_vdn_source(vdn_source)
    model_path = vdn_source.resolve() / "libs" / "models" / "vdn_model.py"
    module_name = "external_vdn_model_68afe1e"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load VDN model definition from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def vdn_config(*, image_size: int = 384) -> SimpleNamespace:
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError("VDN image size must be a positive multiple of 32")
    heatmap_size = image_size // 4
    extra = SimpleNamespace(
        NUM_LAYERS=18,
        DECONV_WITH_BIAS=False,
        NUM_DECONV_LAYERS=3,
        NUM_DECONV_FILTERS=[256, 256, 256],
        NUM_DECONV_KERNELS=[4, 4, 4],
        FINAL_CONV_KERNEL=1,
        TARGET_TYPE="gaussian",
        HEATMAP_SIZE=np.asarray([heatmap_size, heatmap_size]),
        SIGMA=3,
    )
    model = SimpleNamespace(
        EXTRA=extra,
        STYLE="pytorch",
        NUM_JOINTS=1,
        IMAGE_SIZE=np.asarray([image_size, image_size]),
        INIT_WEIGHTS=False,
        PRETRAINED="",
    )
    return SimpleNamespace(MODEL=model)


def _initialize_vdn_heads(model: nn.Module) -> None:
    for module in model.deconv_layers.modules():
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight, std=0.001)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    for head in (model.final_layer_hm, model.final_layer_v):
        for module in head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def load_official_resnet18_initialization() -> tuple[dict[str, torch.Tensor], Path]:
    """Load the exact ResNet-18 file named by the pinned VDN train config."""
    state_dict = torch.hub.load_state_dict_from_url(
        VDN_RESNET18_PRETRAINED_URL,
        progress=True,
        check_hash=True,
        map_location="cpu",
    )
    checkpoint_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(VDN_RESNET18_PRETRAINED_URL).name
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"downloaded ResNet-18 checkpoint missing: {checkpoint_path}"
        )
    return state_dict, checkpoint_path.resolve()


def build_vdn_model(
    vdn_source: Path,
    *,
    image_size: int = 384,
    imagenet_pretrained: bool = False,
) -> nn.Module:
    module = _load_vdn_model_module(vdn_source)
    model = module.get_vdn_resnet(vdn_config(image_size=image_size), is_train=False)
    _initialize_vdn_heads(model)
    if imagenet_pretrained:
        state_dict, _ = load_official_resnet18_initialization()
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected != {"fc.weight", "fc.bias"}:
            raise RuntimeError(f"unexpected ImageNet keys for VDN: {sorted(unexpected)}")
        allowed_prefixes = ("deconv_layers.", "final_layer_hm.", "final_layer_v.")
        invalid_missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if invalid_missing:
            raise RuntimeError(
                f"ImageNet initialization missed backbone keys: {invalid_missing}"
            )
    return model


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _pointer_points(metadata: dict[str, Any], sample_id: str) -> tuple[
    tuple[float, float], tuple[float, float]
]:
    for item in metadata.get("keypoints") or []:
        if str(item.get("type") or "").strip().lower() != "pointer":
            continue
        tip = item.get("outside_kp")
        tail = item.get("origin_kp")
        if (
            isinstance(tip, Sequence)
            and len(tip) >= 2
            and isinstance(tail, Sequence)
            and len(tail) >= 2
        ):
            return (float(tip[0]), float(tip[1])), (
                float(tail[0]),
                float(tail[1]),
            )
    raise ValueError(f"{sample_id}: no pointer tip/tail keypoints in manifest")


def _dial_bbox(metadata: dict[str, Any], sample_id: str) -> tuple[float, float, float, float]:
    bbox = metadata.get("dial_bbox")
    if not isinstance(bbox, Sequence) or len(bbox) < 4:
        raise ValueError(f"{sample_id}: no four-value dial_bbox in manifest")
    x1, y1, x2, y2 = map(float, bbox[:4])
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{sample_id}: invalid dial_bbox {bbox}")
    return x1, y1, x2, y2


def load_syncg_manifest(
    manifest: Path,
    *,
    expected_split: str,
    limit: int | None = None,
) -> tuple[list[VDNSample], dict[str, Any]]:
    manifest = manifest.resolve()
    protocol_path = _protocol_path(manifest)
    if not manifest.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(f"manifest or protocol is missing: {manifest}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol") != "syncg_official_split_v1"
        or str(protocol.get("split")) != expected_split
        or protocol.get("release_identity_verified") is not True
    ):
        raise ValueError(f"{protocol_path} is not a verified SyncG {expected_split} split")

    raw_rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{manifest}:{line_number} is not a JSON object")
            raw_rows.append(row)
    expected_rows = int(protocol.get("expected_rows") or 0)
    if len(raw_rows) != expected_rows:
        raise ValueError(f"{manifest} has {len(raw_rows)} rows; expected {expected_rows}")
    sample_hash = syncg_sample_ids_sha256(str(row.get("sample_id")) for row in raw_rows)
    if sample_hash != protocol.get("expected_sample_ids_sha256"):
        raise ValueError(f"{manifest} sample identity hash does not match the pinned release")
    selected_rows = raw_rows if limit is None else raw_rows[: max(0, int(limit))]

    samples: list[VDNSample] = []
    for row in selected_rows:
        sample_id = str(row.get("sample_id") or "")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{sample_id}: metadata must be an object")
        tip, tail = _pointer_points(metadata, sample_id)
        samples.append(
            VDNSample(
                sample_id=sample_id,
                group_id=str(row.get("group_id") or row.get("meter_id") or sample_id),
                dataset=str(row.get("dataset") or "SyncG"),
                split=str(row.get("split") or expected_split),
                image_path=str(Path(str(row["image_path"])).resolve()),
                dial_bbox=_dial_bbox(metadata, sample_id),
                pointer_tip=tip,
                pointer_tail=tail,
                ground_truth=float(row["ground_truth"]),
                scale_start=float(row["scale_start"]),
                scale_end=float(row["scale_end"]),
                metadata=metadata,
            )
        )
    if not samples:
        raise ValueError(f"{manifest} selection is empty")
    return samples, protocol


def grouped_train_val_split(
    samples: Sequence[VDNSample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[VDNSample], list[VDNSample]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    by_group: dict[str, list[VDNSample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)
    if len(by_group) < 2:
        raise ValueError("at least two groups are required for validation")
    ordered = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    target = max(1, round(len(samples) * validation_fraction))
    validation_groups: set[str] = set()
    validation_count = 0
    for group in ordered:
        if validation_count >= target and validation_groups:
            break
        if len(by_group) - len(validation_groups) <= 1:
            break
        validation_groups.add(group)
        validation_count += len(by_group[group])
    train = [sample for sample in samples if sample.group_id not in validation_groups]
    validation = [sample for sample in samples if sample.group_id in validation_groups]
    if not train or not validation:
        raise RuntimeError("grouped VDN split produced an empty partition")
    return train, validation


def affine_for_dial(
    bbox: Sequence[float],
    *,
    output_size: int,
    expansion: float = 1.25,
    scale_augmentation: float = 1.0,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox[:4])
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
    side = max(x2 - x1, y2 - y1) * float(expansion) * float(scale_augmentation)
    if side <= 0.0:
        raise ValueError(f"invalid dial side for bbox {bbox}")
    matrix = cv2.getRotationMatrix2D(
        (float(center[0]), float(center[1])),
        float(rotation_degrees),
        float(output_size) / side,
    )
    matrix[0, 2] += output_size * 0.5 - center[0]
    matrix[1, 2] += output_size * 0.5 - center[1]
    return matrix.astype(np.float32)


def transform_point(point: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.asarray([float(point[0]), float(point[1]), 1.0])
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous).astype(np.float32)


def normalized_bgr_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"VDN expects a three-channel BGR image, got {image.shape}")
    array = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    mean, std = IMAGENET_NORMALIZATION
    array = (array - mean[:, None, None]) / std[:, None, None]
    return torch.from_numpy(array)


def vdn_tensor_from_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    image_size: int = 384,
    expansion: float = 1.25,
) -> torch.Tensor:
    """Create the official square VDN input from one detected xyxy dial box."""
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
    return normalized_bgr_tensor(crop)


def generate_vdn_targets(
    pointer_tip: Sequence[float],
    pointer_tail: Sequence[float],
    *,
    image_size: int,
    heatmap_size: int,
    sigma: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heatmap = np.zeros((1, heatmap_size, heatmap_size), dtype=np.float32)
    vector_map = np.zeros((2, heatmap_size, heatmap_size), dtype=np.float32)
    stride = float(image_size) / float(heatmap_size)
    head_x = int(float(pointer_tip[0]) / stride + 0.5)
    head_y = int(float(pointer_tip[1]) / stride + 0.5)
    tail_x = int(float(pointer_tail[0]) / stride + 0.5)
    tail_y = int(float(pointer_tail[1]) / stride + 0.5)
    radius = int(sigma) * 3
    upper_left = [head_x - radius, head_y - radius]
    bottom_right = [head_x + radius + 1, head_y + radius + 1]
    if (
        bottom_right[0] <= 0
        or bottom_right[1] <= 0
        or upper_left[0] >= heatmap_size
        or upper_left[1] >= heatmap_size
    ):
        raise ValueError("pointer tip fell outside the VDN training crop")
    size = 2 * radius + 1
    coordinates = np.arange(size, dtype=np.float32)
    gaussian = np.exp(
        -(
            (coordinates[None, :] - radius) ** 2
            + (coordinates[:, None] - radius) ** 2
        )
        / (2.0 * float(sigma) ** 2)
    )
    gaussian_x = max(0, -upper_left[0]), min(bottom_right[0], heatmap_size) - upper_left[0]
    gaussian_y = max(0, -upper_left[1]), min(bottom_right[1], heatmap_size) - upper_left[1]
    image_x = max(0, upper_left[0]), min(bottom_right[0], heatmap_size)
    image_y = max(0, upper_left[1]), min(bottom_right[1], heatmap_size)
    patch = gaussian[gaussian_y[0] : gaussian_y[1], gaussian_x[0] : gaussian_x[1]]
    heatmap[0, image_y[0] : image_y[1], image_x[0] : image_x[1]] = patch
    direction = np.asarray([head_x - tail_x, head_y - tail_y], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ValueError("pointer tip and tail collapse in the VDN heatmap")
    direction /= norm
    vector_map[:, image_y[0] : image_y[1], image_x[0] : image_x[1]] = direction[
        :, None, None
    ]
    return (
        torch.from_numpy(heatmap),
        torch.from_numpy(vector_map),
        torch.from_numpy(direction),
    )


class SyncGVDNDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VDNSample],
        *,
        image_size: int = 384,
        training: bool,
        scale_factor: float = 0.02,
        rotation_factor: float = 90.0,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.heatmap_size = self.image_size // 4
        self.training = bool(training)
        self.scale_factor = float(scale_factor)
        self.rotation_factor = float(rotation_factor)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"failed to read {sample.image_path}")
        scale_augmentation = 1.0
        rotation = 0.0
        if self.training:
            scale_augmentation = float(
                np.clip(
                    np.random.randn() * self.scale_factor + 1.0,
                    1.0 - self.scale_factor,
                    1.0 + self.scale_factor,
                )
            )
            if random.random() <= 0.5:
                rotation = float(
                    np.clip(
                        np.random.randn() * self.rotation_factor,
                        -2.0 * self.rotation_factor,
                        2.0 * self.rotation_factor,
                    )
                )
        matrix = affine_for_dial(
            sample.dial_bbox,
            output_size=self.image_size,
            scale_augmentation=scale_augmentation,
            rotation_degrees=rotation,
        )
        crop = cv2.warpAffine(
            image,
            matrix,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tip = transform_point(sample.pointer_tip, matrix)
        tail = transform_point(sample.pointer_tail, matrix)
        heatmap, vector_map, direction = generate_vdn_targets(
            tip,
            tail,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
        )
        return normalized_bgr_tensor(crop), heatmap, vector_map, direction, sample.sample_id


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict_directions(
    heatmaps: torch.Tensor,
    vector_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if heatmaps.ndim != 4 or vector_maps.ndim != 4:
        raise ValueError("VDN outputs must be BCHW tensors")
    batch, _, _, width = heatmaps.shape
    flattened = heatmaps[:, 0].reshape(batch, -1)
    confidence, index = torch.max(flattened, dim=1)
    y = torch.div(index, width, rounding_mode="floor")
    x = index % width
    batch_index = torch.arange(batch, device=heatmaps.device)
    directions = vector_maps[batch_index, :, y, x]
    norm = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    valid = torch.isfinite(directions).all(dim=1) & (norm[:, 0] > 1e-8)
    directions = directions / torch.clamp(norm, min=1e-8)
    return directions, confidence, valid


def angular_error_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = predicted / torch.clamp(
        torch.linalg.vector_norm(predicted, dim=1, keepdim=True), min=1e-8
    )
    target = target / torch.clamp(
        torch.linalg.vector_norm(target, dim=1, keepdim=True), min=1e-8
    )
    cosine = torch.sum(predicted * target, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def image_angle_from_direction(direction_xy: Sequence[float]) -> float:
    dx, dy = map(float, direction_xy[:2])
    if not np.isfinite([dx, dy]).all() or math.hypot(dx, dy) <= 1e-12:
        raise ValueError("VDN returned an invalid direction")
    return (math.degrees(math.atan2(dx, -dy)) - 180.0) % 360.0


def normalize_reference_points(
    center_start: Sequence[float] | None,
    center_end: Sequence[float] | None,
    *,
    image_width: int,
    distance_threshold: float | None = None,
    start_on_left: bool = True,
) -> tuple[Sequence[float] | None, Sequence[float] | None]:
    """Match the production start/end de-duplication and side convention."""
    if distance_threshold is None:
        distance_threshold = max(10.0, float(image_width) * 0.035)
    else:
        distance_threshold = max(0.0, float(distance_threshold))

    def on_start_side(point: Sequence[float] | None) -> bool:
        if point is None:
            return False
        if start_on_left:
            return float(point[0]) <= float(image_width) * 0.5
        return float(point[0]) >= float(image_width) * 0.5

    def classify_single(
        point: Sequence[float],
    ) -> tuple[Sequence[float] | None, Sequence[float] | None]:
        return (point, None) if on_start_side(point) else (None, point)

    if center_start is not None and center_end is not None:
        distance = float(
            np.linalg.norm(
                np.asarray(center_start, dtype=np.float32)
                - np.asarray(center_end, dtype=np.float32)
            )
        )
        if distance <= distance_threshold:
            return classify_single(center_start)
        if not on_start_side(center_start) and on_start_side(center_end):
            return center_end, center_start
    elif center_start is not None:
        return classify_single(center_start)
    elif center_end is not None:
        return classify_single(center_end)
    return center_start, center_end


def reference_angles(
    image_shape: Sequence[int],
    center_start: Sequence[float] | None,
    center_end: Sequence[float] | None,
    *,
    default_start_angle: float = 45.0,
    default_range_angle: float = 270.0,
) -> tuple[float, float, str]:
    """Resolve start/range angles with the same four production branches."""
    height, width = int(image_shape[0]), int(image_shape[1])
    center = (width // 2, height // 2)

    def point_angle(point: Sequence[float]) -> float:
        return image_angle_from_direction(
            (float(point[0]) - center[0], float(point[1]) - center[1])
        )

    if center_start is not None and center_end is None:
        return point_angle(center_start), float(default_range_angle), "start_only"
    if center_start is None and center_end is not None:
        end_angle = point_angle(center_end)
        start_angle = (end_angle - float(default_range_angle)) % 360.0
        return start_angle, float(default_range_angle), "end_only"
    if center_start is None and center_end is None:
        return (
            float(default_start_angle),
            float(default_range_angle),
            "default_start_end",
        )
    start_angle = point_angle(center_start)
    end_angle = point_angle(center_end)
    return start_angle, (end_angle - start_angle) % 360.0, "start_and_end"


def reading_from_pointer_angle(
    pointer_angle: float,
    *,
    start_angle: float,
    range_angle: float,
    scale_start: float,
    scale_end: float,
) -> tuple[float, float]:
    if not np.isfinite([pointer_angle, start_angle, range_angle]).all():
        raise ValueError("non-finite pointer/reference angle")
    if abs(float(range_angle)) <= 1e-8:
        raise ValueError("dial range angle is zero")
    relative = (float(pointer_angle) - float(start_angle)) % 360.0
    progress = relative / float(range_angle)
    if not 0.0 <= progress <= 1.0:
        distance_to_start = min(relative, 360.0 - relative)
        distance_to_end = abs(relative - float(range_angle))
        progress = 0.0 if distance_to_start <= distance_to_end else 1.0
    reading = float(scale_start) + progress * (float(scale_end) - float(scale_start))
    return reading, progress


def normalized_error(
    prediction: float | None,
    ground_truth: float,
    scale_start: float,
    scale_end: float,
    *,
    failure_penalty: float = 1.0,
) -> float:
    if prediction is None or not np.isfinite(float(prediction)):
        return float(failure_penalty)
    span = abs(float(scale_end) - float(scale_start))
    if span <= 0.0:
        raise ValueError("scale span must be positive")
    return abs(float(prediction) - float(ground_truth)) / span


def group_bootstrap_ci(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    value_array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    if iterations <= 0 or value_array.size == 0:
        return None
    unique_groups = np.unique(group_array)
    if unique_groups.size < 2:
        return None
    by_group = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        estimates.append(float(np.mean(value_array[indices])))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def summarize_scalar_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    errors = [
        normalized_error(
            row.get("prediction"),
            row["ground_truth"],
            row["scale_start"],
            row["scale_end"],
        )
        for row in rows
    ]
    success = [row.get("prediction") is not None for row in rows]
    groups = [str(row.get("group_id") or row.get("meter_id") or index) for index, row in enumerate(rows)]
    error_array = np.asarray(errors, dtype=np.float64)
    success_array = np.asarray(success, dtype=bool)
    return {
        "samples": len(rows),
        "successful": int(np.sum(success_array)),
        "coverage": float(np.mean(success_array)),
        "nmae": float(np.mean(error_array)),
        "nmae_failure_penalty": 1.0,
        "nmae_group_bootstrap_95ci": group_bootstrap_ci(
            errors,
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "acc_1pct": float(np.mean(success_array & (error_array <= 0.01))),
        "acc_2pct": float(np.mean(success_array & (error_array <= 0.02))),
    }


def sample_ids_hash(samples: Iterable[VDNSample]) -> str:
    return syncg_sample_ids_sha256(sample.sample_id for sample in samples)
