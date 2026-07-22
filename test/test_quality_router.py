from __future__ import annotations

import math
import unittest

from experiments.quality_router import (
    FEATURE_NAMES,
    extract_quality_features,
    feature_matrix,
    normalized_error,
    route_prediction,
)


class QualityRouterFeatureTests(unittest.TestCase):
    def _rows(self):
        raw = {
            "sample_id": "sample-a",
            "ground_truth": 55.0,
            "scale_start": 0.0,
            "scale_end": 100.0,
            "status": True,
            "branch": "start_and_end",
            "features": {
                "disAngle": 270.0,
                "p_geom": 0.3,
                "p_geom_v2": 0.4,
                "p_fusion": 0.35,
                "v1_confidence": 0.8,
                "v2_confidence": 0.7,
            },
            "methods": {
                "weighted_fusion": {
                    "prediction": 40.0,
                    "pointer_angle": 90.0,
                },
                "geometry_v1": {
                    "prediction": 42.0,
                    "pointer_angle": 92.0,
                },
                "geometry_v2": {
                    "prediction": 38.0,
                    "pointer_angle": 88.0,
                },
                "transformer": {
                    "prediction": 60.0,
                    "pointer_angle": 110.0,
                },
            },
        }
        base = {
            "sample_id": "sample-a",
            "ground_truth": 55.0,
            "scale_start": 0.0,
            "scale_end": 100.0,
            "predictions": {"ours": 45.0},
            "gate_probability": 0.75,
            "residual_normalized": 0.05,
            "residual_std_normalized": 0.02,
            "correction_applied": True,
        }
        vector = {
            "sample_id": "sample-a",
            "status": True,
            "prediction": 50.0,
            "progress": 0.5,
            "pointer_angle": 100.0,
            "pivot_peak": 0.9,
            "pivot_input_xy": [127.5, 127.5],
        }
        reference = {
            "meter_confidence": 0.95,
            "range_angle": 270.0,
            "reference_branch": "start_and_end",
        }
        return raw, base, vector, reference

    def test_feature_schema_and_key_disagreements(self):
        raw, base, vector, reference = self._rows()
        features = extract_quality_features(
            raw_row=raw,
            base_row=base,
            vector_row=vector,
            reference_row=reference,
        )
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertAlmostEqual(features["base_vector_progress_abs"], 0.05)
        self.assertAlmostEqual(features["weighted_vector_progress_abs"], 0.10)
        self.assertAlmostEqual(features["pivot_center_distance_fraction"], 0.0)
        self.assertAlmostEqual(features["meter_confidence"], 0.95)
        self.assertEqual(feature_matrix([features]).shape, (1, len(FEATURE_NAMES)))

    def test_features_do_not_depend_on_identity_or_ground_truth(self):
        raw, base, vector, reference = self._rows()
        first = extract_quality_features(
            raw_row=raw,
            base_row=base,
            vector_row=vector,
            reference_row=reference,
        )
        raw["sample_id"] = "changed"
        raw["ground_truth"] = -9999.0
        base["sample_id"] = "changed"
        base["ground_truth"] = 9999.0
        second = extract_quality_features(
            raw_row=raw,
            base_row=base,
            vector_row=vector,
            reference_row=reference,
        )
        for name in FEATURE_NAMES:
            if math.isnan(first[name]):
                self.assertTrue(math.isnan(second[name]))
            else:
                self.assertEqual(first[name], second[name])


class QualityRouterPolicyTests(unittest.TestCase):
    def test_hard_fallback_is_invariant(self):
        prediction, route = route_prediction(
            base_prediction=None,
            vector_prediction=12.0,
            score=-100.0,
            threshold=0.5,
        )
        self.assertEqual((prediction, route), (12.0, "vector_hard_fallback"))

    def test_joint_success_switches_only_above_threshold(self):
        self.assertEqual(
            route_prediction(
                base_prediction=1.0,
                vector_prediction=2.0,
                score=0.5,
                threshold=0.5,
            ),
            (1.0, "base"),
        )
        self.assertEqual(
            route_prediction(
                base_prediction=1.0,
                vector_prediction=2.0,
                score=0.5001,
                threshold=0.5,
            ),
            (2.0, "vector_quality_switch"),
        )

    def test_joint_failure_remains_failure(self):
        self.assertEqual(
            route_prediction(
                base_prediction=None,
                vector_prediction=None,
                score=1.0,
                threshold=0.0,
            ),
            (None, "failure"),
        )

    def test_normalized_error_uses_failure_penalty(self):
        row = {"ground_truth": 50.0, "scale_start": 0.0, "scale_end": 100.0}
        self.assertEqual(normalized_error(row, None), 1.0)
        self.assertAlmostEqual(normalized_error(row, 60.0), 0.1)


if __name__ == "__main__":
    unittest.main()
