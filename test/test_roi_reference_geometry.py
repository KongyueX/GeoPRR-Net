from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.roi_reference_geometry import extract_source_pose_references


def _result(points: np.ndarray, confidence: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        keypoints=SimpleNamespace(
            xy=torch.tensor(points, dtype=torch.float32),
            conf=torch.tensor(confidence, dtype=torch.float32),
        ),
        boxes=SimpleNamespace(conf=torch.full((len(points),), 0.9)),
    )


class SourcePoseReferenceTests(unittest.TestCase):
    def test_pointer_tip_changes_cannot_change_selection_output_or_success(self) -> None:
        points = np.asarray(
            [
                [[100, 110], [120, 130], [40, 50], [300, 310]],
                [[200, 210], [220, 230], [60, 70], [320, 330]],
            ],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [[0.9, 0.0, 0.9, 0.9], [0.8, 1.0, 0.8, 0.8]], dtype=np.float32
        )
        expected = points[0, [0, 2, 3]]
        baseline_telemetry = None
        for tip_points, tip_confidence in (
            ([[120, 130], [220, 230]], [0.0, 1.0]),
            ([[1e30, -1e30], [-1e30, 1e30]], [1.0, 0.0]),
            ([[np.nan, np.inf], [-np.inf, np.nan]], [np.nan, np.inf]),
        ):
            with self.subTest(tip_confidence=tip_confidence):
                changed_points = points.copy()
                changed_points[:, 1] = tip_points
                changed_confidence = confidence.copy()
                changed_confidence[:, 1] = tip_confidence
                actual, failure, telemetry = extract_source_pose_references(
                    _result(changed_points, changed_confidence),
                    source_shape=(384, 384),
                    image_size=384,
                )
                self.assertIsNone(failure)
                np.testing.assert_allclose(actual, expected)
                self.assertFalse(telemetry["pointer_tip_used"])
                if baseline_telemetry is None:
                    baseline_telemetry = telemetry
                else:
                    self.assertEqual(telemetry, baseline_telemetry)

    def test_reference_coordinates_map_back_to_non_square_roi(self) -> None:
        points = np.asarray(
            [[[192, 192], [300, 100], [96, 288], [288, 96]]], dtype=np.float32
        )
        actual, failure, _telemetry = extract_source_pose_references(
            _result(points, np.ones((1, 4), dtype=np.float32)),
            source_shape=(192, 768, 3),
            image_size=384,
        )
        self.assertIsNone(failure)
        np.testing.assert_allclose(actual, [[384, 96], [192, 144], [576, 48]])

    def test_confident_pointer_tip_cannot_rescue_low_reference_confidence(self) -> None:
        points = np.asarray(
            [[[192, 192], [300, 100], [96, 288], [288, 96]]], dtype=np.float32
        )
        confidence = np.asarray([[0.9, 1.0, 0.04, 0.9]], dtype=np.float32)
        actual, failure, telemetry = extract_source_pose_references(
            _result(points, confidence),
            source_shape=(384, 384),
            image_size=384,
            minimum_keypoint_confidence=0.05,
        )
        self.assertIsNone(actual)
        self.assertEqual(failure, "low_reference_confidence")
        self.assertAlmostEqual(telemetry["minimum_reference_confidence"], 0.04)


if __name__ == "__main__":
    unittest.main()
