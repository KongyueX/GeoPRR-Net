"""Train-only features and runtime policy for uncertainty soft fusion."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments.quality_router import (
    FEATURE_NAMES as QUALITY_FEATURE_NAMES,
    extract_quality_features,
    finite_float,
)


UNCERTAINTY_FUSION_PROTOCOL = "mask_vector_uncertainty_soft_fusion_v1"
UNCERTAINTY_FUSION_OOF_PROTOCOL = "syncg_probabilistic_uncertainty_fusion_oof_v1"
NATIVE_UNCERTAINTY_FEATURE_NAMES = (
    "vector_angle_std_fraction",
    "vector_angle_variance_fraction2",
    "vector_angle_bin_entropy",
    "vector_angle_bin_resultant_length",
    "vector_pivot_spatial_entropy",
    "vector_pivot_top2_margin",
    "vector_direction_raw_norm",
)
FEATURE_NAMES = QUALITY_FEATURE_NAMES + NATIVE_UNCERTAINTY_FEATURE_NAMES


def _payload(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    nested = row.get(name)
    return nested if isinstance(nested, Mapping) else row


def _nan(value: Any) -> float:
    result = finite_float(value)
    return result if result is not None else math.nan


def extract_uncertainty_features(
    *,
    raw_row: Mapping[str, Any],
    base_row: Mapping[str, Any],
    vector_row: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Extract inference-only features; labels and sample IDs are never read."""

    features = extract_quality_features(
        raw_row=raw_row,
        base_row=base_row,
        vector_row=vector_row,
        reference_row=reference_row,
    )
    vector = _payload(vector_row, "vector")
    angle_std = finite_float(vector.get("angle_std_degrees"))
    angle_std_fraction = (
        float(np.clip(angle_std / 180.0, 0.0, 2.0))
        if angle_std is not None
        else math.nan
    )
    features.update(
        {
            "vector_angle_std_fraction": angle_std_fraction,
            "vector_angle_variance_fraction2": (
                angle_std_fraction**2
                if math.isfinite(angle_std_fraction)
                else math.nan
            ),
            "vector_angle_bin_entropy": _nan(vector.get("angle_bin_entropy")),
            "vector_angle_bin_resultant_length": _nan(
                vector.get("angle_bin_resultant_length")
            ),
            "vector_pivot_spatial_entropy": _nan(
                vector.get("pivot_spatial_entropy")
            ),
            "vector_pivot_top2_margin": _nan(vector.get("pivot_top2_margin")),
            "vector_direction_raw_norm": _nan(vector.get("direction_raw_norm")),
        }
    )
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("uncertainty-fusion feature schema drift")
    return features


def feature_matrix(feature_rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, math.nan)) for name in FEATURE_NAMES] for row in feature_rows],
        dtype=np.float64,
    )


def fit_robust_preprocessor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected uncertainty feature matrix shape")
    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, medians[None, :])
    lower = np.percentile(filled, 25.0, axis=0)
    upper = np.percentile(filled, 75.0, axis=0)
    scales = upper - lower
    standard = np.std(filled, axis=0)
    scales = np.where(scales > 1e-6, scales, standard)
    scales = np.where(scales > 1e-6, scales, 1.0)
    return medians.astype(np.float32), scales.astype(np.float32)


def transform_feature_matrix(
    matrix: np.ndarray,
    *,
    medians: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("unexpected uncertainty feature matrix shape")
    if medians.shape != (len(FEATURE_NAMES),) or scales.shape != medians.shape:
        raise ValueError("invalid uncertainty preprocessor shape")
    filled = np.where(np.isfinite(matrix), matrix, medians[None, :])
    transformed = (filled - medians[None, :]) / scales[None, :]
    return np.clip(transformed, -12.0, 12.0).astype(np.float32)


class DualVarianceMLP(nn.Module):
    """Predict log reading-error variances for the mask and vector experts."""

    def __init__(
        self,
        *,
        input_features: int = len(FEATURE_NAMES),
        hidden_features: int = 64,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.hidden_features = int(hidden_features)
        self.network = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, self.hidden_features // 2),
            nn.GELU(),
            nn.Linear(self.hidden_features // 2, 2),
        )
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # A normalized reading error near 10% is a neutral initialization.
        nn.init.constant_(self.network[-1].bias, math.log(0.10**2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.network(features.float()), min=-12.0, max=2.0)


def inverse_variance_weights(
    log_variances: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if log_variances.ndim != 2 or log_variances.shape[1] != 2:
        raise ValueError("log_variances must have shape [B, 2]")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.softmax(-log_variances.float() / float(temperature), dim=1)


def prediction_progress(
    prediction: Any,
    *,
    scale_start: Any,
    scale_end: Any,
) -> float | None:
    value = finite_float(prediction)
    start = finite_float(scale_start)
    end = finite_float(scale_end)
    if value is None or start is None or end is None or abs(end - start) <= 1e-12:
        return None
    return (value - start) / (end - start)


def soft_fusion_prediction(
    *,
    base_prediction: Any,
    vector_prediction: Any,
    scale_start: Any,
    scale_end: Any,
    mask_log_variance: Any,
    vector_log_variance: Any,
    temperature: float = 1.0,
) -> tuple[float | None, str, float | None, float | None]:
    """Fuse successful experts; preserve deterministic hard-failure behavior."""

    base = finite_float(base_prediction)
    vector = finite_float(vector_prediction)
    if base is None:
        return (
            (vector, "vector_hard_fallback", 0.0, None)
            if vector is not None
            else (None, "failure", None, None)
        )
    if vector is None:
        return base, "base_only", 1.0, None
    start = finite_float(scale_start)
    end = finite_float(scale_end)
    mask_variance = finite_float(mask_log_variance)
    vector_variance = finite_float(vector_log_variance)
    if (
        start is None
        or end is None
        or abs(end - start) <= 1e-12
        or mask_variance is None
        or vector_variance is None
    ):
        return base, "base_uncertainty_unavailable", 1.0, None
    logits = np.asarray(
        [-mask_variance / float(temperature), -vector_variance / float(temperature)],
        dtype=np.float64,
    )
    logits -= float(np.max(logits))
    precision = np.exp(logits)
    weights = precision / max(float(np.sum(precision)), 1e-12)
    mask_weight = float(weights[0])
    fused = mask_weight * base + (1.0 - mask_weight) * vector
    # This conditional variance is useful for diagnostics.  It intentionally
    # does not claim independence-calibrated coverage for the fused experts.
    effective_log_variance = -math.log(
        math.exp(-mask_variance) + math.exp(-vector_variance)
    )
    return float(fused), "uncertainty_soft_fusion", mask_weight, effective_log_variance
