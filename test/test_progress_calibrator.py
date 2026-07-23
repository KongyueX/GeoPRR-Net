from __future__ import annotations

import math
import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from experiments.calibrated_progress_router import (
    FEATURE_NAMES as ROUTER_FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix as router_feature_matrix,
)
from experiments.progress_calibrator import (
    FEATURE_NAMES,
    apply_progress_correction,
    ensemble_prediction,
    extract_progress_features,
    feature_matrix,
    reading_from_progress,
)


class ProgressCalibratorTest(unittest.TestCase):
    def _oof_row(self) -> dict:
        return {
            "sample_id": "ignored",
            "ground_truth": 99.0,
            "scale_start": 0.0,
            "scale_end": 100.0,
            "front_end": {
                "start_angle": 45.0,
                "range_angle": 270.0,
                "reference_branch": "start_and_end",
                "meter_confidence": 0.8,
                "meter_bbox": [10.0, 20.0, 210.0, 220.0],
            },
            "vector": {
                "status": True,
                "progress": 0.25,
                "pointer_angle": 112.5,
                "angle_std_degrees": 2.0,
                "angle_bin_entropy": 0.2,
                "angle_bin_resultant_length": 0.9,
                "pivot_peak": 0.85,
                "pivot_spatial_entropy": 0.3,
                "pivot_top2_margin": 0.1,
                "direction_raw_norm": 4.0,
                "pivot_input_xy": [120.0, 130.0],
            },
        }

    def test_feature_schema_and_no_ground_truth_dependency(self) -> None:
        row = self._oof_row()
        first = extract_progress_features(row)
        changed = dict(row)
        changed["sample_id"] = "different"
        changed["ground_truth"] = -1000.0
        second = extract_progress_features(changed)
        self.assertEqual(tuple(first), FEATURE_NAMES)
        self.assertEqual(first, second)
        matrix = feature_matrix([first, second])
        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_flat_row_can_use_reference_fallback(self) -> None:
        vector = self._oof_row()["vector"]
        reference = self._oof_row()["front_end"]
        features = extract_progress_features(vector, reference_row=reference)
        self.assertAlmostEqual(features["range_angle_fraction"], 0.75)
        self.assertEqual(features["reference_start_and_end"], 1.0)

    def test_correction_and_reading_are_bounded(self) -> None:
        self.assertAlmostEqual(
            apply_progress_correction(0.25, 0.5, correction_clip=0.3), 0.55
        )
        self.assertAlmostEqual(
            apply_progress_correction(0.9, 0.5, correction_clip=0.3), 1.0
        )
        self.assertAlmostEqual(reading_from_progress(0.55, -10.0, 90.0), 45.0)
        self.assertIsNone(apply_progress_correction(None, 0.1, correction_clip=0.3))
        self.assertIsNone(reading_from_progress(None, 0.0, 1.0))
        with self.assertRaises(ValueError):
            apply_progress_correction(0.2, 0.1, correction_clip=0.0)

    def test_ensemble_prediction_returns_tree_std(self) -> None:
        x = np.asarray([[0.0], [1.0], [2.0], [math.nan]], dtype=np.float64)
        y = np.asarray([0.0, 0.5, 1.0, 0.25], dtype=np.float64)
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=8,
                        max_depth=3,
                        random_state=7,
                    ),
                ),
            ]
        )
        model.fit(x, y)
        mean, std = ensemble_prediction(model, x)
        self.assertEqual(mean.shape, (4,))
        self.assertEqual(std.shape, (4,))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue(np.all(std >= 0.0))

    def test_calibrated_router_feature_schema(self) -> None:
        row = self._oof_row()
        row["base"] = {"prediction": 20.0, "status": True}
        row["raw"] = {"status": True, "features": {}}
        calibration = {
            "corrected_progress_oof": 0.3,
            "predicted_residual_oof": 0.05,
            "ensemble_std_oof": 0.02,
            "raw_progress": 0.25,
        }
        features = extract_calibrated_router_features(
            raw_row=row,
            base_row=row,
            vector_row=row,
            reference_row=None,
            calibration_row=calibration,
        )
        self.assertEqual(tuple(features), ROUTER_FEATURE_NAMES)
        self.assertAlmostEqual(features["raw_calibrated_progress_abs"], 0.05)
        self.assertAlmostEqual(features["base_calibrated_progress_abs"], 0.1)
        self.assertEqual(
            router_feature_matrix([features]).shape,
            (1, len(ROUTER_FEATURE_NAMES)),
        )


if __name__ == "__main__":
    unittest.main()
