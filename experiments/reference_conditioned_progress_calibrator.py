"""Reference-conditioned residual calibration with a train-selected safety deadband."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experiments.quality_router import finite_float


REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL = (
    "reference_conditioned_safe_progress_calibrator_v1"
)
REFERENCE_BRANCHES = (
    "default_start_end",
    "end_only",
    "start_and_end",
    "start_only",
)
DEFAULT_REFERENCE_BRANCH = "default_start_end"


def deterministic_ensemble_prediction(
    estimator: Any,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate trees in a fixed order to make OOF files byte-reproducible."""
    imputer = estimator.named_steps.get("imputer")
    regressor = estimator.named_steps.get("regressor")
    trees = getattr(regressor, "estimators_", None)
    if imputer is None or not trees:
        raise ValueError("calibrator is not the expected fitted tree pipeline")
    transformed = imputer.transform(np.asarray(matrix, dtype=np.float64))
    predictions = np.asarray(
        [tree.predict(transformed) for tree in trees],
        dtype=np.float64,
    )
    return np.mean(predictions, axis=0), np.std(predictions, axis=0)


def normalize_reference_branch(value: Any) -> str:
    """Map a runtime row or branch label to the finite training-time branch set."""
    if isinstance(value, Mapping):
        front_end = value.get("front_end")
        payload = front_end if isinstance(front_end, Mapping) else value
        value = payload.get("reference_branch")
    branch = str(value or "").strip().lower()
    return branch if branch in REFERENCE_BRANCHES else DEFAULT_REFERENCE_BRANCH


def apply_safe_residual(
    residual: Any,
    *,
    correction_clip: float,
    deadband: float,
) -> float | None:
    """Clip a predicted correction and abstain inside the fitted safety deadband."""
    value = finite_float(residual)
    if value is None:
        return None
    if not math.isfinite(correction_clip) or correction_clip <= 0.0:
        raise ValueError("correction_clip must be finite and positive")
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError("deadband must be finite and non-negative")
    if abs(value) < deadband:
        return 0.0
    return float(np.clip(value, -correction_clip, correction_clip))


def apply_safe_residual_array(
    residual: np.ndarray,
    *,
    correction_clip: float,
    deadband: float,
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("residual must be a one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("residual contains non-finite values")
    if not math.isfinite(correction_clip) or correction_clip <= 0.0:
        raise ValueError("correction_clip must be finite and positive")
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError("deadband must be finite and non-negative")
    applied = np.clip(values, -correction_clip, correction_clip)
    return np.where(np.abs(values) < deadband, 0.0, applied)


def _policy_for_branch(
    artifact: Mapping[str, Any],
    branch: str,
) -> tuple[float, float]:
    fallback = artifact.get("fallback_policy")
    fallback = fallback if isinstance(fallback, Mapping) else {}
    policies = artifact.get("branch_policies")
    policies = policies if isinstance(policies, Mapping) else {}
    selected = policies.get(branch)
    selected = selected if isinstance(selected, Mapping) else fallback
    clip = finite_float(selected.get("correction_clip"))
    deadband = finite_float(selected.get("deadband"))
    if clip is None or clip <= 0.0:
        raise ValueError(f"invalid correction clip for reference branch {branch!r}")
    if deadband is None or deadband < 0.0:
        raise ValueError(f"invalid deadband for reference branch {branch!r}")
    return clip, deadband


def predict_reference_conditioned_residual(
    artifact: Mapping[str, Any],
    matrix: np.ndarray,
    branches: Sequence[Any],
) -> dict[str, np.ndarray]:
    """Predict model and applied residuals using the matching reference branch."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    normalized = np.asarray(
        [normalize_reference_branch(branch) for branch in branches],
        dtype=object,
    )
    if len(normalized) != len(values):
        raise ValueError("branch count differs from feature row count")
    fallback_estimator = artifact.get("fallback_estimator")
    if fallback_estimator is None:
        raise ValueError("calibrator artifact has no fallback estimator")
    estimators = artifact.get("branch_estimators")
    estimators = estimators if isinstance(estimators, Mapping) else {}

    model_residual = np.full(len(values), math.nan, dtype=np.float64)
    applied_residual = np.full(len(values), math.nan, dtype=np.float64)
    ensemble_std = np.full(len(values), math.nan, dtype=np.float64)
    correction_clip = np.full(len(values), math.nan, dtype=np.float64)
    deadband = np.full(len(values), math.nan, dtype=np.float64)
    used_branch_model = np.zeros(len(values), dtype=bool)
    for branch in sorted(set(normalized.tolist())):
        selected = normalized == branch
        estimator = estimators.get(branch)
        branch_model = estimator is not None
        if estimator is None:
            estimator = fallback_estimator
        mean, std = deterministic_ensemble_prediction(
            estimator,
            values[selected],
        )
        clip, threshold = _policy_for_branch(artifact, branch)
        model_residual[selected] = mean
        applied_residual[selected] = apply_safe_residual_array(
            mean,
            correction_clip=clip,
            deadband=threshold,
        )
        ensemble_std[selected] = std
        correction_clip[selected] = clip
        deadband[selected] = threshold
        used_branch_model[selected] = branch_model
    if not (
        np.isfinite(model_residual).all()
        and np.isfinite(applied_residual).all()
        and np.isfinite(ensemble_std).all()
    ):
        raise RuntimeError("reference-conditioned prediction is incomplete")
    return {
        "model_residual": model_residual,
        "applied_residual": applied_residual,
        "ensemble_std": ensemble_std,
        "correction_clip": correction_clip,
        "deadband": deadband,
        "used_branch_model": used_branch_model,
        "reference_branch": normalized,
    }
