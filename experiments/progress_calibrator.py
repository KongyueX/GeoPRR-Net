"""Train-only perspective-aware calibration from pointer angle to scale progress."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.quality_router import finite_float


PROGRESS_CALIBRATOR_PROTOCOL = "perspective_aware_progress_calibrator_v1"

BASE_FEATURE_NAMES = (
    "vector_progress",
    "pointer_angle_sin",
    "pointer_angle_cos",
    "start_angle_sin",
    "start_angle_cos",
    "range_angle_fraction",
    "reference_start_and_end",
    "reference_start_only",
    "reference_end_only",
    "meter_confidence",
)
NATIVE_UNCERTAINTY_FEATURE_NAMES = (
    "angle_std_fraction",
    "angle_bin_entropy",
    "angle_bin_resultant_length",
    "pivot_peak",
    "pivot_spatial_entropy",
    "pivot_top2_margin",
    "log1p_direction_raw_norm",
    "pivot_x_fraction",
    "pivot_y_fraction",
    "meter_bbox_aspect",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + NATIVE_UNCERTAINTY_FEATURE_NAMES


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _vector_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("vector")
    return _mapping(nested) if isinstance(nested, Mapping) else row


def _front_end_payload(
    row: Mapping[str, Any], reference_row: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    nested = row.get("front_end")
    if isinstance(nested, Mapping):
        return nested
    return reference_row if isinstance(reference_row, Mapping) else row


def _angle_features(value: Any) -> tuple[float, float]:
    angle = finite_float(value)
    if angle is None:
        return math.nan, math.nan
    radians = math.radians(angle)
    return math.sin(radians), math.cos(radians)


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return math.nan, math.nan
    if len(value) != 2:
        return math.nan, math.nan
    first = finite_float(value[0])
    second = finite_float(value[1])
    return (
        first if first is not None else math.nan,
        second if second is not None else math.nan,
    )


def _bbox_aspect(value: Any) -> float:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return math.nan
    if len(value) != 4:
        return math.nan
    coordinates = [finite_float(item) for item in value]
    if any(item is None for item in coordinates):
        return math.nan
    left, top, right, bottom = (float(item) for item in coordinates)
    height = bottom - top
    return (right - left) / height if abs(height) > 1e-8 else math.nan


def extract_progress_features(
    row: Mapping[str, Any],
    *,
    reference_row: Mapping[str, Any] | None = None,
    image_size: int = 256,
) -> dict[str, float]:
    """Extract runtime-only features without ground truth or sample identity."""
    vector = _vector_payload(row)
    front_end = _front_end_payload(row, reference_row)
    pointer_sin, pointer_cos = _angle_features(vector.get("pointer_angle"))
    start_sin, start_cos = _angle_features(front_end.get("start_angle"))
    branch = str(front_end.get("reference_branch") or "").strip().lower()
    pivot_x, pivot_y = _point(vector.get("pivot_input_xy"))
    direction_norm = finite_float(vector.get("direction_raw_norm"))
    angle_std = finite_float(vector.get("angle_std_degrees"))
    range_angle = finite_float(front_end.get("range_angle"))
    values = {
        "vector_progress": finite_float(vector.get("progress")),
        "pointer_angle_sin": pointer_sin,
        "pointer_angle_cos": pointer_cos,
        "start_angle_sin": start_sin,
        "start_angle_cos": start_cos,
        "range_angle_fraction": (
            range_angle / 360.0 if range_angle is not None else math.nan
        ),
        "reference_start_and_end": float(branch == "start_and_end"),
        "reference_start_only": float(branch == "start_only"),
        "reference_end_only": float(branch == "end_only"),
        "meter_confidence": finite_float(front_end.get("meter_confidence")),
        "angle_std_fraction": (
            angle_std / 180.0 if angle_std is not None else math.nan
        ),
        "angle_bin_entropy": finite_float(vector.get("angle_bin_entropy")),
        "angle_bin_resultant_length": finite_float(
            vector.get("angle_bin_resultant_length")
        ),
        "pivot_peak": finite_float(vector.get("pivot_peak")),
        "pivot_spatial_entropy": finite_float(vector.get("pivot_spatial_entropy")),
        "pivot_top2_margin": finite_float(vector.get("pivot_top2_margin")),
        "log1p_direction_raw_norm": (
            math.log1p(max(direction_norm, 0.0))
            if direction_norm is not None
            else math.nan
        ),
        "pivot_x_fraction": pivot_x / float(image_size),
        "pivot_y_fraction": pivot_y / float(image_size),
        "meter_bbox_aspect": _bbox_aspect(front_end.get("meter_bbox")),
    }
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError("progress-calibrator feature schema drift")
    return {
        name: float(value) if value is not None else math.nan
        for name, value in values.items()
    }


def feature_matrix(feature_rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row.get(name, math.nan)) for name in FEATURE_NAMES]
            for row in feature_rows
        ],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected progress-calibrator feature matrix shape")
    return matrix


def apply_progress_correction(
    progress: Any,
    residual: Any,
    *,
    correction_clip: float,
) -> float | None:
    progress_value = finite_float(progress)
    residual_value = finite_float(residual)
    if progress_value is None or residual_value is None:
        return None
    if not math.isfinite(correction_clip) or correction_clip <= 0.0:
        raise ValueError("correction_clip must be finite and positive")
    correction = float(np.clip(residual_value, -correction_clip, correction_clip))
    return float(np.clip(progress_value + correction, 0.0, 1.0))


def reading_from_progress(
    progress: Any, scale_start: Any, scale_end: Any
) -> float | None:
    progress_value = finite_float(progress)
    start = finite_float(scale_start)
    end = finite_float(scale_end)
    if progress_value is None or start is None or end is None:
        return None
    if abs(end - start) <= 1e-12:
        return None
    return float(start + progress_value * (end - start))


def ensemble_prediction(
    estimator: Any, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ExtraTrees mean residual and tree-disagreement standard deviation."""
    mean = np.asarray(estimator.predict(matrix), dtype=np.float64)
    imputer = estimator.named_steps.get("imputer")
    regressor = estimator.named_steps.get("regressor")
    trees = getattr(regressor, "estimators_", None)
    if imputer is None or not trees:
        raise ValueError("progress calibrator is not the expected fitted pipeline")
    transformed = imputer.transform(matrix)
    predictions = np.asarray(
        [tree.predict(transformed) for tree in trees], dtype=np.float64
    )
    return mean, np.std(predictions, axis=0)
