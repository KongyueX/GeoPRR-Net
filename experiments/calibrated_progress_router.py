"""Features and policy for routing between mask and calibrated-vector readings."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.quality_router import finite_float
from experiments.uncertainty_fusion import (
    FEATURE_NAMES as QUALITY_UNCERTAINTY_FEATURE_NAMES,
    extract_uncertainty_features,
)


CALIBRATED_PROGRESS_ROUTER_PROTOCOL = "mask_calibrated_vector_router_v1"
CALIBRATION_FEATURE_NAMES = (
    "calibrated_vector_progress",
    "predicted_progress_residual",
    "progress_ensemble_std",
    "predicted_progress_residual_abs",
    "raw_calibrated_progress_abs",
    "base_calibrated_progress_abs",
)
FEATURE_NAMES = QUALITY_UNCERTAINTY_FEATURE_NAMES + CALIBRATION_FEATURE_NAMES


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _base_prediction(row: Mapping[str, Any]) -> float | None:
    nested = row.get("base")
    if isinstance(nested, Mapping):
        return finite_float(nested.get("prediction"))
    return finite_float((row.get("predictions") or {}).get("ours"))


def _progress_from_reading(
    prediction: Any, scale_start: Any, scale_end: Any
) -> float | None:
    value = finite_float(prediction)
    start = finite_float(scale_start)
    end = finite_float(scale_end)
    if value is None or start is None or end is None or abs(end - start) <= 1e-12:
        return None
    return float((value - start) / (end - start))


def extract_calibrated_router_features(
    *,
    raw_row: Mapping[str, Any],
    base_row: Mapping[str, Any],
    vector_row: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None,
    calibration_row: Mapping[str, Any],
) -> dict[str, float]:
    features = dict(
        extract_uncertainty_features(
            raw_row=raw_row,
            base_row=base_row,
            vector_row=vector_row,
            reference_row=reference_row,
        )
    )
    calibrated_progress = finite_float(
        calibration_row.get("corrected_progress_oof")
        if "corrected_progress_oof" in calibration_row
        else calibration_row.get("progress")
    )
    predicted_residual = finite_float(
        calibration_row.get("predicted_residual_oof")
        if "predicted_residual_oof" in calibration_row
        else calibration_row.get("predicted_progress_residual")
    )
    ensemble_std = finite_float(
        calibration_row.get("ensemble_std_oof")
        if "ensemble_std_oof" in calibration_row
        else calibration_row.get("progress_ensemble_std")
    )
    raw_progress = finite_float(
        calibration_row.get("raw_progress")
        if "raw_progress" in calibration_row
        else _mapping(vector_row.get("vector")).get("progress")
    )
    if raw_progress is None:
        raw_progress = finite_float(vector_row.get("progress"))
    base_progress = _progress_from_reading(
        _base_prediction(base_row),
        base_row.get("scale_start"),
        base_row.get("scale_end"),
    )
    features.update(
        {
            "calibrated_vector_progress": calibrated_progress,
            "predicted_progress_residual": predicted_residual,
            "progress_ensemble_std": ensemble_std,
            "predicted_progress_residual_abs": (
                abs(predicted_residual) if predicted_residual is not None else math.nan
            ),
            "raw_calibrated_progress_abs": (
                abs(raw_progress - calibrated_progress)
                if raw_progress is not None and calibrated_progress is not None
                else math.nan
            ),
            "base_calibrated_progress_abs": (
                abs(base_progress - calibrated_progress)
                if base_progress is not None and calibrated_progress is not None
                else math.nan
            ),
        }
    )
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("calibrated-progress router feature schema drift")
    return {
        name: float(value) if value is not None else math.nan
        for name, value in features.items()
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
        raise ValueError("unexpected calibrated-progress router matrix shape")
    return matrix
