"""Analytic source role/field supervision for the semantic RGB reader."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.geoprr_v2_data import (
    GeoPRRV2Dataset, GeometryConfig, GeometryContractError, dial_calibration, transform_points,
)


DATA_CONFIG = {"image_size": 256, "output_size": 64}
GEOMETRY_CONFIG = GeometryConfig()


def calibration_matrix(row: Mapping[str, Any], image_size: int = 256) -> np.ndarray:
    """Reproduce the RGB crop coordinates from the declared source dimensions."""
    metadata = row.get("metadata", {})
    dimensions = [metadata.get("image_width"), metadata.get("image_height")]
    if any(value is None or not np.isfinite(value) or int(value) != value or value <= 0 for value in dimensions):
        raise GeometryContractError(f"{row['sample_id']}: positive integer source image dimensions are required")
    width, height = map(int, dimensions)
    bbox: Sequence[float] | None = None if "roi_path" in row else metadata.get("dial_bbox")
    if bbox is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        if len(bbox) != 4 or not np.isfinite(bbox).all():
            raise GeometryContractError(f"{row['sample_id']}: dial bbox must be finite xyxy")
        left, top = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
        right, bottom = min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3]))
    if right <= left or bottom <= top:
        raise GeometryContractError(f"{row['sample_id']}: dial bbox does not intersect image")
    sx, sy = image_size / (right - left), image_size / (bottom - top)
    return np.array([[sx, 0, sx * (0.5 - left) - 0.5],
                     [0, sy, sy * (0.5 - top) - 0.5], [0, 0, 1]], dtype=np.float64)


def analytic_field_targets(calibration: Mapping[str, Any], image_size: int = 256,
                           output_size: int = 64,
                           config: GeometryConfig = GEOMETRY_CONFIG) -> dict[str, Any]:
    """Construct analytic U/M and retain the radius-one virtual role location.

    This contract has no Gaussian Q or Q/M/U readout target.  Its audit is
    the ordered physical dial calibration audit, without kernel support or
    readout-discretization claims.
    """
    if (image_size, output_size) != (256, 64):
        raise ValueError("analytic role/field source supervision requires image256/output64")
    if not np.isfinite(list(calibration["audit"].values())).all():
        raise GeometryContractError("physical dial calibration audit is not finite")
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
    return {"field_target": torch.from_numpy(progress.astype(np.float32)),
            "field_mask": torch.from_numpy(mask[None]),
            "roi_to_canonical": torch.from_numpy(roi_to_canonical.astype(np.float32)),
            "span_radians": torch.tensor(float(anchors[-1]), dtype=torch.float32),
            "virtual_point_xy": torch.from_numpy(point_roi.astype(np.float32)),
            "geometry_audit": dict(calibration["audit"])}


class RoleFieldRGBDataset(Dataset):
    """Keep the scalar RGB transform and add source-only analytic supervision."""

    def __init__(self, manifest, *, data_root, augment):
        self.base = GeoPRRV2Dataset(manifest, dataset_root=data_root, **DATA_CONFIG,
                                   require_geometry=False, augment=augment, pair_manifest=None)
        self.rows, self.augment = self.base.rows, augment
        self._calibrations: dict[int, dict[str, Any]] = {}

    def __len__(self):
        return len(self.rows)

    def calibration(self, index):
        if index not in self._calibrations:
            row = self.rows[index]
            self._calibrations[index] = dial_calibration(row, calibration_matrix(row), GEOMETRY_CONFIG)
        return self._calibrations[index]

    def __getitem__(self, index):
        sample = self.base[index]
        targets = analytic_field_targets(self.calibration(index))
        return {**{key: sample[key] for key in ("image", "y", "sample_id", "group_id")},
                **{key: targets[key] for key in ("roi_to_canonical", "field_target", "field_mask", "span_radians")},
                "input_failure": ""}
