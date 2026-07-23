from __future__ import annotations

import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from experiments.evaluate_reference_conditioned_pipeline import _route
from experiments.reference_conditioned_progress_calibrator import (
    apply_safe_residual,
    apply_safe_residual_array,
    normalize_reference_branch,
    predict_reference_conditioned_residual,
)
from experiments.reference_conditioned_router import deterministic_router_prediction


def _constant_forest(value: float, seed: int) -> Pipeline:
    matrix = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    target = np.full(len(matrix), value, dtype=np.float64)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=8,
                    max_depth=2,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(matrix, target)
    return model


class ReferenceConditionedProgressCalibratorTest(unittest.TestCase):
    def test_reference_branch_normalization_is_finite(self) -> None:
        self.assertEqual(
            normalize_reference_branch(
                {"front_end": {"reference_branch": "START_AND_END"}}
            ),
            "start_and_end",
        )
        self.assertEqual(
            normalize_reference_branch({"reference_branch": "start_only"}),
            "start_only",
        )
        self.assertEqual(
            normalize_reference_branch("unseen_branch"),
            "default_start_end",
        )

    def test_safe_residual_deadband_and_clip(self) -> None:
        self.assertEqual(
            apply_safe_residual(0.02, correction_clip=0.4, deadband=0.05),
            0.0,
        )
        self.assertAlmostEqual(
            apply_safe_residual(0.7, correction_clip=0.4, deadband=0.05),
            0.4,
        )
        values = apply_safe_residual_array(
            np.asarray([-0.7, -0.01, 0.2]),
            correction_clip=0.4,
            deadband=0.05,
        )
        np.testing.assert_allclose(values, [-0.4, 0.0, 0.2])
        with self.assertRaises(ValueError):
            apply_safe_residual(0.1, correction_clip=0.4, deadband=-0.1)

    def test_prediction_uses_branch_model_and_fallback_policy(self) -> None:
        artifact = {
            "fallback_estimator": _constant_forest(-0.1, 1),
            "branch_estimators": {
                "start_and_end": _constant_forest(0.2, 2),
            },
            "fallback_policy": {
                "correction_clip": 0.4,
                "deadband": 0.15,
            },
            "branch_policies": {
                "start_and_end": {
                    "correction_clip": 0.4,
                    "deadband": 0.15,
                },
                "start_only": {
                    "correction_clip": 0.4,
                    "deadband": 0.15,
                },
            },
        }
        result = predict_reference_conditioned_residual(
            artifact,
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            ["start_and_end", "start_only"],
        )
        np.testing.assert_allclose(result["model_residual"], [0.2, -0.1])
        np.testing.assert_allclose(result["applied_residual"], [0.2, 0.0])
        np.testing.assert_array_equal(result["used_branch_model"], [True, False])

    def test_router_keeps_hard_fallback_independent_of_threshold(self) -> None:
        self.assertEqual(
            _route(None, 2.0, -100.0, 100.0),
            (2.0, "reference_conditioned_hard_fallback"),
        )
        self.assertEqual(
            _route(1.0, 2.0, 0.2, 0.1),
            (2.0, "reference_conditioned_quality_switch"),
        )
        self.assertEqual(_route(1.0, 2.0, 0.0, 0.1), (1.0, "base"))
        self.assertEqual(_route(None, None, 1.0, 0.0), (None, "failure"))

    def test_router_tree_aggregation_is_finite(self) -> None:
        model = _constant_forest(0.125, 3)
        prediction = deterministic_router_prediction(
            model,
            np.asarray([[0.0], [1.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(prediction, [0.125, 0.125])


if __name__ == "__main__":
    unittest.main()
