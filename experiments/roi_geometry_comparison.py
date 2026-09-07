"""Shared geometry and metric helpers for matched ROI comparison models.

The comparison uses the same conditioned canonical dial pixels as GeoPRR.
Four keypoints are ordered as pivot, pointer tip, scale start, and scale end.
No dataset identity or checkpoint-freezing mechanism is introduced here; the
callers receive explicit paths and report the exact run configuration.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments.geometric_progress_decoder import decode_geometric_progress


KEYPOINT_NAMES: Final[tuple[str, ...]] = (
    "pivot",
    "pointer_tip",
    "reference_start",
    "reference_end",
)
FAILURE_ERROR: Final[float] = 1.0


def extract_syncg_keypoints(sample: Any) -> np.ndarray:
    """Read the ordered four-point geometry from one SyncG manifest sample."""

    metadata = sample.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{sample.sample_id}: metadata is not a mapping")
    pointer: Mapping[str, Any] | None = None
    scale: Mapping[str, Any] | None = None
    for item in metadata.get("keypoints") or ():
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "").casefold()
        if kind == "pointer":
            pointer = item
        elif kind == "scalemark":
            scale = item
    if pointer is None or scale is None:
        raise ValueError(f"{sample.sample_id}: pointer or scale keypoints are missing")
    marks = scale.get("all_kp")
    if not isinstance(marks, Sequence) or len(marks) < 2:
        raise ValueError(f"{sample.sample_id}: ordered scale endpoints are missing")
    values = (
        pointer.get("origin_kp"),
        pointer.get("outside_kp"),
        marks[0],
        marks[-1],
    )
    points = np.asarray(values, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{sample.sample_id}: invalid four-keypoint geometry")
    return points


def canonical_roi_bounds(
    image_shape: Sequence[int], bbox_xyxy: Sequence[float]
) -> tuple[int, int, int, int]:
    """Return the clipped floor/ceil crop used by the canonical ROI materializer."""

    if len(image_shape) < 2 or len(bbox_xyxy) < 4:
        raise ValueError("image shape or dial bbox is incomplete")
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    right = min(width, int(math.ceil(x2)))
    bottom = min(height, int(math.ceil(y2)))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid clipped dial bbox: {bbox_xyxy}")
    return left, top, right, bottom


def keypoints_in_canonical_roi(
    points_xy: np.ndarray,
    *,
    bbox_xyxy: Sequence[float],
    source_shape: Sequence[int],
) -> np.ndarray:
    """Translate source-image keypoints into native canonical-ROI coordinates."""

    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("four keypoints are required")
    left, top, _right, _bottom = canonical_roi_bounds(source_shape, bbox_xyxy)
    return points - np.asarray([left, top], dtype=np.float32)


def transform_points_homography(
    points_xy: np.ndarray, homography: Sequence[Sequence[float]]
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    matrix = np.asarray(homography, dtype=np.float64)
    if points.shape != (4, 2) or matrix.shape != (3, 3):
        raise ValueError("invalid keypoints or homography shape")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = homogeneous @ matrix.T
    denominators = projected[:, 2]
    if not np.isfinite(projected).all() or np.any(np.abs(denominators) <= 1.0e-10):
        raise ValueError("homography produced invalid keypoints")
    return (projected[:, :2] / denominators[:, None]).astype(np.float32)


def condition_keypoints(
    points_xy: np.ndarray, degradation_metadata: Mapping[str, Any]
) -> np.ndarray:
    perspective = degradation_metadata.get("perspective")
    if perspective is None:
        return np.asarray(points_xy, dtype=np.float32).copy()
    if not isinstance(perspective, Mapping):
        raise ValueError("perspective metadata is not a mapping")
    return transform_points_homography(points_xy, perspective.get("homography"))


def resize_keypoints(
    points_xy: np.ndarray,
    *,
    source_shape: Sequence[int],
    output_size: int,
) -> np.ndarray:
    height, width = int(source_shape[0]), int(source_shape[1])
    if height < 1 or width < 1 or output_size < 2:
        raise ValueError("invalid resize geometry")
    scale = np.asarray(
        [float(output_size) / float(width), float(output_size) / float(height)],
        dtype=np.float32,
    )
    return np.asarray(points_xy, dtype=np.float32) * scale


def decode_progress_from_keypoints(
    points_xy: np.ndarray, *, clockwise: bool = True
) -> tuple[float | None, str | None, dict[str, float]]:
    """Convert four points to normalized gauge progress with explicit failures."""

    points = np.asarray(points_xy, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None, "invalid_keypoints", {}
    pivot, tip, start, end = points
    direction_xy = tip - pivot
    pivot_tensor = torch.from_numpy(pivot[None, :])
    direction_sin_cos = torch.from_numpy(
        np.asarray([[direction_xy[0], -direction_xy[1]]], dtype=np.float32)
    )
    references = torch.from_numpy(np.stack((start, end), axis=0)[None, ...])
    try:
        decoded = decode_geometric_progress(
            pivot_tensor,
            direction_sin_cos,
            references,
            clockwise=clockwise,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return None, f"geometry_exception:{type(exc).__name__}", {}
    valid = bool(decoded["valid"][0].item())
    telemetry = {
        "arc_degrees": float(torch.rad2deg(decoded["arc_radians"])[0].item()),
        "offset_degrees": float(
            torch.rad2deg(decoded["offset_radians"])[0].item()
        ),
    }
    if not valid:
        return None, "invalid_geometry", telemetry
    value = float(decoded["progress"][0].item())
    if not math.isfinite(value):
        return None, "non_finite_progress", telemetry
    return float(np.clip(value, 0.0, 1.0)), None, telemetry


def direction_from_pointer_probability(
    probability: np.ndarray,
    pivot_xy: Sequence[float],
    *,
    threshold: float = 0.5,
    minimum_pixels: int = 8,
) -> tuple[np.ndarray | None, str | None, dict[str, float]]:
    """Recover the outward pointer ray from a probability map and offline pivot.

    The network predicts only the pointer mask.  The annotated pivot is used
    after inference to select and orient the component, matching the declared
    annotation-assisted component role of this comparison cell.
    """

    values = np.asarray(probability, dtype=np.float32)
    pivot = np.asarray(pivot_xy, dtype=np.float32)
    if values.ndim != 2 or pivot.shape != (2,) or not np.isfinite(pivot).all():
        return None, "invalid_mask_or_pivot", {}
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, 0.0)
    binary = (values >= float(threshold)).astype(np.uint8)
    components, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    candidates: list[tuple[float, int, int]] = []
    yy, xx = np.indices(values.shape, dtype=np.float32)
    for label in range(1, components):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(minimum_pixels):
            continue
        active = labels == label
        distance = np.hypot(xx[active] - pivot[0], yy[active] - pivot[1])
        candidates.append((float(distance.min()), -area, label))
    if not candidates:
        return None, "no_pointer_component", {"mask_pixels": float(binary.sum())}
    minimum_distance, negative_area, selected = min(candidates)
    height, width = values.shape
    maximum_gap = max(8.0, 0.12 * float(min(height, width)))
    if minimum_distance > maximum_gap:
        return None, "pointer_component_far_from_pivot", {
            "component_gap": minimum_distance,
            "component_pixels": float(-negative_area),
        }
    active = labels == selected
    coordinates = np.column_stack((xx[active], yy[active])).astype(np.float32)
    vectors = coordinates - pivot[None, :]
    distances = np.linalg.norm(vectors, axis=1)
    maximum_distance = float(distances.max(initial=0.0))
    if maximum_distance < 0.08 * float(min(height, width)):
        return None, "pointer_component_too_short", {
            "maximum_radius": maximum_distance,
            "component_pixels": float(len(coordinates)),
        }
    far_cutoff = float(np.quantile(distances, 0.65))
    far = distances >= far_cutoff
    weights = values[active][far] * np.square(
        np.maximum(distances[far] / max(maximum_distance, 1.0e-6), 0.05)
    )
    direction = np.sum(vectors[far] * weights[:, None], axis=0)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-6:
        return None, "degenerate_pointer_direction", {
            "maximum_radius": maximum_distance,
            "component_pixels": float(len(coordinates)),
        }
    direction = (direction / norm).astype(np.float32)
    return direction, None, {
        "component_gap": minimum_distance,
        "component_pixels": float(len(coordinates)),
        "maximum_radius": maximum_distance,
        "mean_probability": float(np.mean(values[active])),
    }


def normalized_target(sample: Any) -> float:
    span = float(sample.scale_end) - float(sample.scale_start)
    if not math.isfinite(span) or abs(span) <= 1.0e-12:
        raise ValueError(f"{sample.sample_id}: invalid scale span")
    value = (float(sample.ground_truth) - float(sample.scale_start)) / span
    if not math.isfinite(value):
        raise ValueError(f"{sample.sample_id}: non-finite normalized target")
    return value


def metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty prediction roster")
    errors = np.asarray([float(row["absolute_error"]) for row in rows], dtype=np.float64)
    passed = np.asarray([row.get("status") == "pass" for row in rows], dtype=bool)
    if not np.isfinite(errors).all():
        raise ValueError("prediction errors contain non-finite values")
    return {
        "rows": int(len(rows)),
        "nmae": float(np.mean(errors)),
        "nmae_percent_fs": float(100.0 * np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "acc_at_1_percent_fs": float(np.mean(errors <= 0.01)),
        "acc_at_2_percent_fs": float(np.mean(errors <= 0.02)),
        "acc_at_5_percent_fs": float(np.mean(errors <= 0.05)),
        "coverage": float(np.mean(passed)),
        "failures": int(np.sum(~passed)),
    }


def build_evaluation_summary(
    rows: Sequence[Mapping[str, Any]], *, method: str, seed: int, configuration: Mapping[str, Any]
) -> dict[str, Any]:
    conditions = tuple(dict.fromkeys(str(row["condition"]) for row in rows))
    return {
        "schema_version": 1,
        "method": method,
        "seed": int(seed),
        "failure_policy": "failed row receives normalized absolute error 1.0",
        "configuration": dict(configuration),
        "all_conditions": metric_summary(rows),
        "per_condition": {
            condition: metric_summary(
                [row for row in rows if str(row["condition"]) == condition]
            )
            for condition in conditions
        },
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "FAILURE_ERROR",
    "KEYPOINT_NAMES",
    "build_evaluation_summary",
    "canonical_roi_bounds",
    "condition_keypoints",
    "decode_progress_from_keypoints",
    "direction_from_pointer_probability",
    "extract_syncg_keypoints",
    "keypoints_in_canonical_roi",
    "metric_summary",
    "normalized_target",
    "resize_keypoints",
    "transform_points_homography",
    "write_json",
]
