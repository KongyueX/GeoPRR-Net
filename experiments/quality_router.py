"""Runtime-only features and policies for mask/vector quality routing."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


QUALITY_ROUTER_PROTOCOL = "mask_vector_quality_router_v1"

RAW_QUALITY_FEATURES = (
    "p_geom",
    "p_geom_v2",
    "p_fusion",
    "v1_v2_progress_delta",
    "v1_confidence",
    "v2_confidence",
    "v1_axis_score",
    "v1_support_ratio",
    "v1_direction_consistency",
    "v1_tip_support_ratio",
    "v2_axis_score",
    "v2_support_ratio",
    "v2_vote_concentration",
    "v2_side_separation",
    "mask_component_area_ratio",
    "mask_candidate_ratio",
    "mask_center_distance_ratio",
    "mask_axis_threshold_ratio",
    "seg_probability_max",
    "seg_probability_p99",
    "seg_probability_mean",
    "seg_foreground_ratio",
    "ellipse_ratio",
    "ellipse_area_ratio",
)

FEATURE_NAMES = (
    "base_vector_progress_abs",
    "base_vector_progress_signed",
    "weighted_vector_progress_abs",
    "geometry_v1_vector_progress_abs",
    "geometry_v2_vector_progress_abs",
    "transformer_vector_progress_abs",
    "weighted_vector_angle_abs_fraction",
    "geometry_v1_vector_angle_abs_fraction",
    "geometry_v2_vector_angle_abs_fraction",
    "transformer_vector_angle_abs_fraction",
    "pivot_peak",
    "pivot_center_distance_fraction",
    "meter_confidence",
    "gate_probability",
    "residual_abs_normalized",
    "residual_std_normalized",
    "correction_applied",
    "range_angle_fraction",
    "branch_start_and_end",
    "branch_start_only",
    "branch_end_only",
    "raw_front_end_success",
) + RAW_QUALITY_FEATURES


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _value_or_nan(value: Any) -> float:
    result = finite_float(value)
    return result if result is not None else math.nan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row.get("sample_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate sample identifiers")
    return result


def normalized_error(row: Mapping[str, Any], prediction: Any, *, penalty: float = 1.0) -> float:
    value = finite_float(prediction)
    ground_truth = finite_float(row.get("ground_truth"))
    scale_start = finite_float(row.get("scale_start"))
    scale_end = finite_float(row.get("scale_end"))
    if value is None or ground_truth is None or scale_start is None or scale_end is None:
        return float(penalty)
    span = abs(scale_end - scale_start)
    if span <= 1e-12:
        return float(penalty)
    return abs(value - ground_truth) / span


def _method(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = (raw.get("methods") or {}).get(name) or {}
    return value if isinstance(value, Mapping) else {}


def _prediction_progress(
    prediction: Any,
    *,
    scale_start: float | None,
    scale_end: float | None,
) -> float | None:
    value = finite_float(prediction)
    if value is None or scale_start is None or scale_end is None:
        return None
    span = scale_end - scale_start
    if abs(span) <= 1e-12:
        return None
    return (value - scale_start) / span


def _progress_difference(first: float | None, second: float | None, *, signed: bool) -> float:
    if first is None or second is None:
        return math.nan
    value = first - second
    value = float(np.clip(value, -2.0, 2.0))
    return value if signed else abs(value)


def _circular_angle_difference(first: Any, second: Any) -> float:
    first_value = finite_float(first)
    second_value = finite_float(second)
    if first_value is None or second_value is None:
        return math.nan
    difference = abs((first_value - second_value + 180.0) % 360.0 - 180.0)
    return difference / 180.0


def _base_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("base")
    return nested if isinstance(nested, Mapping) else row


def _base_prediction(row: Mapping[str, Any]) -> float | None:
    nested = row.get("base")
    if isinstance(nested, Mapping):
        return finite_float(nested.get("prediction"))
    return finite_float((row.get("predictions") or {}).get("ours"))


def _vector_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get("vector")
    return nested if isinstance(nested, Mapping) else row


def _front_end_payload(
    row: Mapping[str, Any], reference: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    nested = row.get("front_end")
    if isinstance(nested, Mapping):
        return nested
    return reference or row


def extract_quality_features(
    *,
    raw_row: Mapping[str, Any],
    base_row: Mapping[str, Any],
    vector_row: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Build the frozen router feature vector without reading labels or IDs."""

    raw = raw_row.get("raw") if isinstance(raw_row.get("raw"), Mapping) else raw_row
    base = _base_payload(base_row)
    vector = _vector_payload(vector_row)
    front_end = _front_end_payload(vector_row, reference_row)
    scale_start = finite_float(base_row.get("scale_start"))
    scale_end = finite_float(base_row.get("scale_end"))
    if scale_start is None or scale_end is None:
        scale_start = finite_float(raw_row.get("scale_start"))
        scale_end = finite_float(raw_row.get("scale_end"))
    base_progress = _prediction_progress(
        _base_prediction(base_row), scale_start=scale_start, scale_end=scale_end
    )
    vector_progress = finite_float(vector.get("progress"))
    if vector_progress is None:
        vector_progress = _prediction_progress(
            vector.get("prediction"), scale_start=scale_start, scale_end=scale_end
        )
    methods = raw.get("methods") or {}
    method_progress: dict[str, float | None] = {}
    for name in ("weighted_fusion", "geometry_v1", "geometry_v2", "transformer"):
        method = methods.get(name) or {}
        progress = finite_float(method.get("progress"))
        if progress is None:
            progress = _prediction_progress(
                method.get("prediction"), scale_start=scale_start, scale_end=scale_end
            )
        method_progress[name] = progress

    pivot = vector.get("pivot_input_xy")
    pivot_distance = math.nan
    if isinstance(pivot, Sequence) and len(pivot) >= 2:
        x = finite_float(pivot[0])
        y = finite_float(pivot[1])
        if x is not None and y is not None:
            # All signed direction checkpoints use a 256 x 256 input.
            pivot_distance = math.hypot(x - 127.5, y - 127.5) / (math.sqrt(2) * 127.5)

    branch = str(raw.get("branch") or front_end.get("reference_branch") or "")
    range_angle = finite_float(front_end.get("range_angle"))
    if range_angle is None:
        range_angle = finite_float((raw.get("features") or {}).get("disAngle"))
    vector_angle = vector.get("pointer_angle")
    features: dict[str, float] = {
        "base_vector_progress_abs": _progress_difference(
            base_progress, vector_progress, signed=False
        ),
        "base_vector_progress_signed": _progress_difference(
            base_progress, vector_progress, signed=True
        ),
        "weighted_vector_progress_abs": _progress_difference(
            method_progress["weighted_fusion"], vector_progress, signed=False
        ),
        "geometry_v1_vector_progress_abs": _progress_difference(
            method_progress["geometry_v1"], vector_progress, signed=False
        ),
        "geometry_v2_vector_progress_abs": _progress_difference(
            method_progress["geometry_v2"], vector_progress, signed=False
        ),
        "transformer_vector_progress_abs": _progress_difference(
            method_progress["transformer"], vector_progress, signed=False
        ),
        "weighted_vector_angle_abs_fraction": _circular_angle_difference(
            _method(raw, "weighted_fusion").get("pointer_angle"), vector_angle
        ),
        "geometry_v1_vector_angle_abs_fraction": _circular_angle_difference(
            _method(raw, "geometry_v1").get("pointer_angle"), vector_angle
        ),
        "geometry_v2_vector_angle_abs_fraction": _circular_angle_difference(
            _method(raw, "geometry_v2").get("pointer_angle"), vector_angle
        ),
        "transformer_vector_angle_abs_fraction": _circular_angle_difference(
            _method(raw, "transformer").get("pointer_angle"), vector_angle
        ),
        "pivot_peak": _value_or_nan(vector.get("pivot_peak")),
        "pivot_center_distance_fraction": pivot_distance,
        "meter_confidence": _value_or_nan(front_end.get("meter_confidence")),
        "gate_probability": _value_or_nan(base.get("gate_probability")),
        "residual_abs_normalized": abs(_value_or_nan(base.get("residual_normalized"))),
        "residual_std_normalized": _value_or_nan(
            base.get("residual_std_normalized")
        ),
        "correction_applied": float(bool(base.get("correction_applied"))),
        "range_angle_fraction": abs(range_angle) / 360.0 if range_angle is not None else math.nan,
        "branch_start_and_end": float(branch == "start_and_end"),
        "branch_start_only": float(branch == "start_only"),
        "branch_end_only": float(branch == "end_only"),
        "raw_front_end_success": float(raw.get("status") is True),
    }
    raw_features = raw.get("features") or {}
    for name in RAW_QUALITY_FEATURES:
        value = finite_float(raw_features.get(name))
        features[name] = value if value is not None else math.nan
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("quality-router feature schema drift")
    return features


def feature_matrix(feature_rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, math.nan)) for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )


def route_prediction(
    *,
    base_prediction: Any,
    vector_prediction: Any,
    score: Any,
    threshold: float,
) -> tuple[float | None, str]:
    """Hard fallback is invariant; learned switching applies only to joint success."""

    base = finite_float(base_prediction)
    vector = finite_float(vector_prediction)
    quality = finite_float(score)
    if base is None:
        return (vector, "vector_hard_fallback") if vector is not None else (None, "failure")
    if vector is not None and quality is not None and quality > float(threshold):
        return vector, "vector_quality_switch"
    return base, "base"
