"""ROI data and audited dial-plane supervision for the lightweight reader.

Geometry is source-training supervision only.  ``require_geometry=False``
reads ordinary labeled ROI manifests without consulting structural labels.
Image/output pixel coordinates use the half-pixel, align_corners=False rule.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GeometryContractError(ValueError):
    """The source annotation cannot support the declared dial-plane labels."""


@dataclass(frozen=True)
class GeometryConfig:
    max_radial_error: float = 0.02
    max_anchor_progress_error: float = 0.01
    max_center_error: float = 0.05
    max_readout_error: float = 0.01
    ring_half_width: float = 0.20
    point_sigma: float = 1.0
    min_support_mass: float = 0.15


def read_rows(source: str | Path | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(source, (str, Path)):
        rows = [dict(row) for row in source]
    else:
        path = Path(source)
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            rows = value if isinstance(value, list) else value.get("samples", value.get("rows"))
        else:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain nonempty JSON object rows")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest sample_id values must be nonempty and unique")
    return rows


def normalized_target(row: Mapping[str, Any]) -> float:
    if all(key in row for key in ("ground_truth", "scale_start", "scale_end")):
        start, end, value = (float(row[key]) for key in ("scale_start", "scale_end", "ground_truth"))
        if not np.isfinite([start, end, value]).all() or end <= start:
            raise ValueError(f"{row['sample_id']}: invalid physical reading range")
        y = (value - start) / (end - start)
    elif "normalized_progress" in row:
        y = float(row["normalized_progress"])
    elif "y" in row:
        y = float(row["y"])
    else:
        raise ValueError(f"{row['sample_id']}: no reading target")
    if not math.isfinite(y) or not -1e-7 <= y <= 1 + 1e-7:
        raise ValueError(f"{row['sample_id']}: progress outside [0,1]: {y}")
    if "normalized_progress" in row and abs(float(row["normalized_progress"]) - y) > 1e-6:
        raise ValueError(f"{row['sample_id']}: physical and normalized targets disagree")
    return float(np.clip(y, 0, 1))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points.reshape(-1, 2), np.ones(points.size // 2)))
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    if not np.isfinite(transformed).all() or np.any(np.abs(transformed[:, 2]) < 1e-10):
        raise GeometryContractError("homography crosses its projective horizon")
    return (transformed[:, :2] / transformed[:, 2:]).reshape(points.shape)


def crop_and_resize(image: np.ndarray, bbox: Sequence[float] | None, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if bbox is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        if len(bbox) != 4 or not np.isfinite(bbox).all():
            raise ValueError("dial bbox must be finite xyxy")
        left, top = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
        right, bottom = min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3]))
    if right <= left or bottom <= top:
        raise ValueError("dial bbox does not intersect image")
    sx, sy = image_size / (right - left), image_size / (bottom - top)
    original_to_roi = np.array([[sx, 0, sx * (0.5 - left) - 0.5],
                                [0, sy, sy * (0.5 - top) - 0.5], [0, 0, 1]], dtype=np.float64)
    roi = cv2.resize(image[top:bottom, left:right], (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return roi, original_to_roi


def _metadata_geometry(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    metadata = row.get("metadata", {})
    marks_entries = [item for item in metadata.get("keypoints", [])
                     if isinstance(item, Mapping) and str(item.get("type", "")).lower() == "scalemark"]
    if len(marks_entries) != 1:
        raise GeometryContractError("exactly one ordered ScaleMark annotation is required")
    marks = np.asarray(marks_entries[0].get("all_kp"), dtype=np.float64)
    matrix = np.asarray(metadata.get("homography"), dtype=np.float64)
    if marks.ndim != 2 or marks.shape[1:] != (2,) or len(marks) < 4 or not np.isfinite(marks).all():
        raise GeometryContractError("at least four finite ordered scale points are required")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise GeometryContractError("missing or singular bbox-normalized dial homography")
    if int(metadata.get("long_num", -1)) != len(marks):
        raise GeometryContractError("long_num must count all major ticks, including both endpoints")
    return marks, matrix, metadata


def dial_calibration(row: Mapping[str, Any], original_to_roi: np.ndarray, config: GeometryConfig) -> dict[str, Any]:
    """Validate H against ordered ticks and construct a common metric dial frame.

    SyncG H maps bbox-normalized source coordinates to [0,1] dial-plane
    coordinates.  The center comes from the complete rectified tick circle;
    Pointer.origin_kp is above the dial and is not the plane center target.
    """
    sample_id = row["sample_id"]
    try:
        marks, h_ann, metadata = _metadata_geometry(row)
        bbox = np.asarray(metadata.get("homography_bbox", metadata.get("dial_bbox")), dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise GeometryContractError("missing valid homography bbox")
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        normalize_bbox = np.array([[1 / w, 0, -bbox[0] / w], [0, 1 / h, -bbox[1] / h], [0, 0, 1]])
        source_to_plane = h_ann @ normalize_bbox
        rectified = transform_points(marks, source_to_plane)
        lhs = np.column_stack((2 * rectified, np.ones(len(marks))))
        fit, _, rank, _ = np.linalg.lstsq(lhs, (rectified ** 2).sum(axis=1), rcond=None)
        if rank != 3:
            raise GeometryContractError("ordered scale points do not determine a circle")
        center = fit[:2]
        radius_squared = fit[2] + float(center @ center)
        if radius_squared <= 0:
            raise GeometryContractError("invalid reference ring radius")
        radius = math.sqrt(radius_squared)
        vectors = (rectified - center) / radius
        radial_error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))
        center_error = float(np.linalg.norm(center - 0.5) / radius)
        unwrapped = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        differences = np.diff(unwrapped)
        direction = 1.0 if np.median(differences) > 0 else -1.0
        angles = direction * (unwrapped - unwrapped[0])
        if np.any(np.diff(angles) <= 0) or not 0 < angles[-1] < 2 * np.pi - 1e-5:
            raise GeometryContractError("scale marks are not a single ordered arc")
        expected = np.linspace(0, 1, len(marks))
        anchor_error = float(np.max(np.abs(angles / angles[-1] - expected)))
        interval = float(metadata.get("long_interval_degree", float("nan")))
        arc_error = abs(float(angles[-1]) - math.radians(interval * (len(marks) - 1))) / float(angles[-1])
        y = normalized_target(row)
        pointer_angle = float(metadata.get("pointer_angle", float("nan")))
        physical_error = abs(pointer_angle / (interval * (len(marks) - 1)) - y)
        if not np.isfinite([interval, pointer_angle, arc_error, physical_error]).all() or interval <= 0:
            raise GeometryContractError("physical angle/interval labels are missing")
        if physical_error > 1e-6:
            raise GeometryContractError(f"physical y and pointer_rotate_degree disagree ({physical_error:.6g})")
        if radial_error > config.max_radial_error or center_error > config.max_center_error:
            raise GeometryContractError(f"H/bbox circle audit failed: radial={radial_error:.6g}, center={center_error:.6g}")
        if anchor_error > config.max_anchor_progress_error or arc_error > config.max_anchor_progress_error:
            raise GeometryContractError(f"ordered scale angle audit failed: anchor={anchor_error:.6g}, arc={arc_error:.6g}")
        # This layout-fixed frame puts the ordered first mark on +x and follows
        # the physical tick order.  It never uses the current pointer position.
        start = float(unwrapped[0])
        rotate = np.array([[math.cos(start), math.sin(start)],
                           [-direction * math.sin(start), direction * math.cos(start)]]) / radius
        plane_to_canonical = np.eye(3)
        plane_to_canonical[:2, :2] = rotate
        plane_to_canonical[:2, 2] = -rotate @ center
        roi_to_canonical = plane_to_canonical @ source_to_plane @ np.linalg.inv(original_to_roi)
        return {"roi_to_canonical": roi_to_canonical, "angles": angles,
                "progress_anchors": expected, "y": y,
                "audit": {"radial_error": radial_error, "center_error": center_error,
                          "anchor_progress_error": anchor_error, "arc_relative_error": arc_error,
                          "physical_progress_error": physical_error}}
    except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
        raise GeometryContractError(f"{sample_id}: {exc}") from exc


def geometry_targets(calibration: Mapping[str, Any], image_size: int, output_size: int,
                     config: GeometryConfig) -> dict[str, Any]:
    """Construct continuous U, a geometric ring M, and virtual dial point Q."""
    roi_to_canonical = np.asarray(calibration["roi_to_canonical"])
    centers = (np.arange(output_size) + 0.5) * image_size / output_size - 0.5
    xx, yy = np.meshgrid(centers, centers)
    canonical = transform_points(np.stack((xx, yy), axis=-1), roi_to_canonical)
    radii = np.linalg.norm(canonical, axis=-1)
    angles = np.mod(np.arctan2(canonical[..., 1], canonical[..., 0]), 2 * np.pi)
    anchors = np.asarray(calibration["angles"])
    progress = np.interp(angles, anchors, calibration["progress_anchors"])
    mask = ((radii >= 1 - config.ring_half_width) & (radii <= 1 + config.ring_half_width)
            & (angles <= anchors[-1])).astype(np.float32)
    if not mask.any():
        raise GeometryContractError("reference ring has no output-grid support")
    angle = float(np.interp(calibration["y"], calibration["progress_anchors"], anchors))
    virtual = np.array([[math.cos(angle), math.sin(angle)]])
    point_roi = transform_points(virtual, np.linalg.inv(roi_to_canonical))[0]
    point_grid = (point_roi + 0.5) * output_size / image_size - 0.5
    if np.any(point_grid < -0.5) or np.any(point_grid > output_size - 0.5):
        raise GeometryContractError("virtual dial-plane readout point is outside the ROI")
    # Define the kernel on the shared radius-one dial plane.  Its image-plane
    # density includes the projective Jacobian, so camera-pair Q targets obey
    # probability-mass transport rather than an isotropic image-kernel rule.
    sigma_canonical = 2 * config.point_sigma / output_size
    squared_distance = ((canonical - virtual[0]) ** 2).sum(axis=-1)
    denominator = roi_to_canonical[2, 0] * xx + roi_to_canonical[2, 1] * yy + roi_to_canonical[2, 2]
    jacobian = abs(np.linalg.det(roi_to_canonical)) / np.abs(denominator) ** 3
    logits = -squared_distance / (2 * sigma_canonical ** 2) + np.log(jacobian)
    point = np.exp(logits - logits.max())
    point /= point.sum()
    mass = float((point * mask).sum())
    if mass < config.min_support_mass:
        raise GeometryContractError(f"virtual point has insufficient geometric support ({mass:.6g})")
    decoded = float((point * mask * progress).sum() / mass)
    readout_error = abs(decoded - calibration["y"])
    if readout_error > config.max_readout_error:
        raise GeometryContractError(f"GT Q/M/U readout discretization exceeds tolerance ({readout_error:.6g})")
    audit = dict(calibration["audit"], gt_readout_error=readout_error, gt_support_mass=mass)
    return {"field_target": torch.from_numpy(progress.astype(np.float32)),
            "field_mask": torch.from_numpy(mask[None]),
            "support_target": torch.from_numpy(mask[None].copy()),
            "point_target": torch.from_numpy(point[None].astype(np.float32)),
            "roi_to_canonical": torch.from_numpy(roi_to_canonical.astype(np.float32)),
            "virtual_point_xy": torch.from_numpy(point_roi.astype(np.float32)),
            "geometry_audit": audit}


def _resolve_image(row: Mapping[str, Any], manifest_root: Path, dataset_root: Path | None) -> Path:
    value = row.get("roi_path", row.get("image_path"))
    if not value:
        raise ValueError(f"{row['sample_id']}: missing image_path or roi_path")
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [manifest_root / path]
    if dataset_root is not None:
        candidates.append(dataset_root / path)
        # Existing manifests may have been generated on a different host.
        parts = str(value).replace("\\", "/").split("/")
        for marker in ("images", "rois"):
            if marker in parts:
                relative = Path(*parts[parts.index(marker):])
                candidates.append(dataset_root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{row['sample_id']}: image not found: {value}")


def load_controlled_pairs(source: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Require complete per-image factor records, rather than seed equality."""
    records = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["sample_id"]): row for row in rows}
    pairs: dict[str, dict[str, Any]] = {}
    for pair in records:
        a, b = str(pair["sample_a"]), str(pair["sample_b"])
        if a not in by_id or b not in by_id or a == b or a in pairs:
            raise ValueError("controlled pairs require distinct in-partition IDs and one partner per sample_a")
        kind = pair.get("kind")
        changed = {"state": "r", "appearance": "a", "camera": "c"}.get(kind)
        if changed is None or not pair.get("layout_id"):
            raise ValueError("pair kind must be state, appearance, or camera with a layout_id")
        factors = []
        for side, sid in (("a", a), ("b", b)):
            metadata = by_id[sid].get("metadata", {})
            value = pair.get(f"factors_{side}")
            if not isinstance(value, dict) or set(value) != {"g", "r", "a", "c"}:
                raise ValueError("pair factor record must contain exactly g, r, a, c")
            if not isinstance(value["g"], dict) or not value["g"]:
                raise ValueError("pair g must record actual nonempty layout parameters")
            if metadata.get("factors") != value or metadata.get("layout_id") != pair["layout_id"]:
                raise ValueError("pair factors/layout must match the individual image generation records")
            if abs(float(value["r"]) - normalized_target(by_id[sid])) > 1e-6:
                raise ValueError("pair state r does not match its physical reading")
            factors.append(value)
        if factors[0][changed] == factors[1][changed]:
            raise ValueError("declared intervention did not change its recorded factor")
        if any(factors[0][key] != factors[1][key] for key in {"g", "r", "a", "c"} - {changed}):
            raise ValueError("non-intervened factors changed in controlled pair")
        pairs[a] = dict(pair, pair_same_state=kind != "state")
    return pairs


class GeoPRRV2Dataset(Dataset):
    def __init__(self, manifest_path: str | Path | Sequence[Mapping[str, Any]], *,
                 dataset_root: str | Path | None = None, image_size: int = 640,
                 output_size: int | None = None, augment: bool = False,
                 require_geometry: bool = True, pair_manifest: str | Path | None = None,
                 geometry_config: GeometryConfig | None = None) -> None:
        self.rows = read_rows(manifest_path)
        self.manifest_root = Path(manifest_path).resolve().parent if isinstance(manifest_path, (str, Path)) else Path.cwd()
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.image_size = image_size
        self.output_size = output_size or image_size // 4
        if image_size < 32 or image_size % 32 or self.output_size != image_size // 4:
            raise ValueError("image_size must be a multiple of 32 and output_size must be image_size/4")
        self.augment, self.require_geometry = augment, require_geometry
        self.geometry_config = geometry_config or GeometryConfig()
        self.pairs = load_controlled_pairs(pair_manifest, self.rows) if pair_manifest is not None else None
        self.by_id = {str(row["sample_id"]): index for index, row in enumerate(self.rows)}
        self._calibrations: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _sample(self, index: int, photometric: tuple[float, float, float] | None = None) -> dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve_image(row, self.manifest_root, self.dataset_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"{row['sample_id']}: failed to read {image_path}")
        metadata = row.get("metadata", {})
        if "image_width" in metadata and "image_height" in metadata:
            if image.shape[:2] != (int(metadata["image_height"]), int(metadata["image_width"])):
                raise GeometryContractError(f"{row['sample_id']}: decoded image dimensions disagree with source annotation coordinates")
        bbox = None if "roi_path" in row else row.get("metadata", {}).get("dial_bbox")
        roi, original_to_roi = crop_and_resize(image, bbox, self.image_size)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.augment:
            # Photometric augmentation changes no geometric or intervention records.
            contrast, brightness, saturation = photometric or self._photometric()
            rgb = (rgb - 0.5) * contrast + 0.5 + brightness
            gray = rgb.mean(axis=2, keepdims=True)
            rgb = np.clip(gray + saturation * (rgb - gray), 0, 1)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        tensor = (tensor - tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        y = torch.tensor(normalized_target(row), dtype=torch.float32)
        sample = {"image": tensor, "y": y, "target": y, "sample_id": str(row["sample_id"]),
                  "group_id": str(row.get("group_id", row["sample_id"])), "image_path": str(image_path)}
        if self.require_geometry:
            if index not in self._calibrations:
                self._calibrations[index] = dial_calibration(row, original_to_roi, self.geometry_config)
            sample.update(geometry_targets(self._calibrations[index], self.image_size, self.output_size, self.geometry_config))
        return sample

    @staticmethod
    def _photometric() -> tuple[float, float, float]:
        return (float(np.random.uniform(0.8, 1.2)), float(np.random.uniform(-0.1, 0.1)),
                float(np.random.uniform(0.8, 1.2)))

    def __getitem__(self, index: int) -> dict[str, Any]:
        photometric = self._photometric() if self.augment else None
        sample = self._sample(index, photometric)
        if self.pairs is None:
            return sample
        pair = self.pairs.get(sample["sample_id"])
        partner = self._sample(self.by_id[pair["sample_b"]], photometric) if pair is not None else sample
        keys = ["image", "y", "sample_id"]
        if self.require_geometry:
            keys += ["field_target", "field_mask", "point_target", "support_target", "roi_to_canonical"]
        for key in keys:
            sample[f"pair_{key}"] = partner[key]
        sample["pair_kind"] = pair["kind"] if pair is not None else "none"
        sample["pair_same_state"] = pair["pair_same_state"] if pair is not None else False
        valid = pair is not None
        if pair is not None and self.require_geometry:
            # B-grid pixels sample A through A_A^-1 A_B.  Eligibility uses GT
            # support/Q before prediction and cannot be hidden by a zero M head.
            a = sample["roi_to_canonical"].numpy()
            b = partner["roi_to_canonical"].numpy()
            scale = self.image_size / self.output_size
            output_to_roi = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
            b_to_a = np.linalg.inv(output_to_roi) @ np.linalg.inv(a) @ b @ output_to_roi
            gy, gx = np.indices((self.output_size, self.output_size), dtype=np.float64)
            locations = transform_points(np.stack((gx, gy), -1), b_to_a).astype(np.float32)
            support = cv2.remap(sample["support_target"][0].numpy(), locations[..., 0], locations[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            overlap = float((support * partner["support_target"][0].numpy() * partner["point_target"][0].numpy()).sum())
            valid = overlap >= self.geometry_config.min_support_mass
        sample["pair_valid"] = valid
        return sample
