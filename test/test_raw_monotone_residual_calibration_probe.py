from __future__ import annotations

import unittest

import numpy as np

from experiments.raw_monotone_residual_calibration_probe import (
    MonotoneLinearSpline,
    apply_median_bias,
    fit_median_bias,
    fit_monotone_l1_spline,
)


class MonotoneResidualCalibrationTests(unittest.TestCase):
    def test_global_l1_spline_improves_known_monotone_curve(self) -> None:
        predictions = np.linspace(0.0, 1.0, 401)
        targets = predictions**2
        spline, metadata = fit_monotone_l1_spline(
            predictions, targets, knot_count=7
        )
        calibrated = spline.predict(predictions)
        identity_l1 = float(np.mean(np.abs(predictions - targets)))
        calibrated_l1 = float(np.mean(np.abs(calibrated - targets)))
        self.assertLess(calibrated_l1, identity_l1 * 0.1)
        self.assertTrue(np.all(np.diff(spline.knots_y) >= -1.0e-10))
        self.assertAlmostEqual(
            calibrated_l1, metadata["objective_exact_l1"], places=7
        )

    def test_identity_relation_is_recovered(self) -> None:
        predictions = np.linspace(0.02, 0.98, 257)
        spline, _ = fit_monotone_l1_spline(
            predictions, predictions, knot_count=7
        )
        calibrated = spline.predict(predictions)
        self.assertLess(float(np.max(np.abs(calibrated - predictions))), 1.0e-7)

    def test_prediction_is_bounded_and_monotone_outside_fit_range(self) -> None:
        spline = MonotoneLinearSpline(
            np.asarray([0.2, 0.5, 0.8]),
            np.asarray([0.1, 0.55, 0.9]),
        )
        values = spline.predict(np.asarray([-1.0, 0.3, 0.7, 2.0]))
        self.assertTrue(np.all(np.diff(values) >= 0.0))
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        self.assertAlmostEqual(float(values[0]), 0.1)
        self.assertAlmostEqual(float(values[-1]), 0.9)

    def test_median_bias_matches_l1_location(self) -> None:
        predictions = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        targets = predictions + np.asarray([0.03, 0.02, 0.02, 0.01, -0.10])
        bias = fit_median_bias(predictions, targets)
        self.assertAlmostEqual(bias, 0.02)
        corrected = apply_median_bias(predictions, bias)
        self.assertLess(
            float(np.mean(np.abs(corrected - targets))),
            float(np.mean(np.abs(predictions - targets))),
        )


if __name__ == "__main__":
    unittest.main()
