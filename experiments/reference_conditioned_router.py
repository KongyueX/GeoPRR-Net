"""Protocol and deterministic inference for reference-conditioned routing."""

from __future__ import annotations

from typing import Any

import numpy as np

REFERENCE_CONDITIONED_ROUTER_PROTOCOL = "mask_reference_conditioned_vector_router_v1"


def deterministic_router_prediction(
    estimator: Any,
    matrix: np.ndarray,
) -> np.ndarray:
    """Aggregate tree predictions in a fixed order across CPU configurations."""
    imputer = estimator.named_steps.get("imputer")
    regressor = estimator.named_steps.get("regressor")
    trees = getattr(regressor, "estimators_", None)
    if imputer is None or not trees:
        raise ValueError("router is not the expected fitted tree pipeline")
    transformed = imputer.transform(np.asarray(matrix, dtype=np.float64))
    predictions = np.asarray(
        [tree.predict(transformed) for tree in trees],
        dtype=np.float64,
    )
    return np.mean(predictions, axis=0)
