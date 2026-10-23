"""Targeted synthetic checks for analytic source-only U/M supervision."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.geoprr_v2_data import (
    GeometryContractError, crop_and_resize, dial_calibration, geometry_targets,
)
from experiments.geoprr_v2_role_field_data import (
    GEOMETRY_CONFIG, analytic_field_targets, calibration_matrix,
)


def source_row(y):
    angles = np.linspace(0, 1.5 * math.pi, 7)
    marks = np.column_stack((128 + 88 * np.cos(angles), 128 + 88 * np.sin(angles))).tolist()
    return {"sample_id": "synthetic_source", "ground_truth": 10 + 20 * y,
            "scale_start": 10., "scale_end": 30., "normalized_progress": y,
            "metadata": {"image_width": 256, "image_height": 256, "dial_bbox": [0, 0, 256, 256],
                         "homography": np.eye(3).tolist(), "long_num": 7,
                         "long_interval_degree": 45., "pointer_angle": 270 * y,
                         "keypoints": [{"type": "ScaleMark", "all_kp": marks}]}}


class RoleFieldDataTests(unittest.TestCase):
    def test_crop_matrix_matches_rgb_fractional_clipped_and_roi_inputs(self):
        image = np.zeros((93, 117, 3), dtype=np.uint8)
        row = {"sample_id": "crop", "metadata": {"image_width": 117, "image_height": 93,
                                                    "dial_bbox": [-2.2, 7.8, 114.1, 100.6]}}
        _, expected = crop_and_resize(image, row["metadata"]["dial_bbox"], 256)
        np.testing.assert_array_equal(calibration_matrix(row), expected)
        row["roi_path"] = "existing_roi.png"
        _, expected = crop_and_resize(image, None, 256)
        np.testing.assert_array_equal(calibration_matrix(row), expected)
        del row["metadata"]["image_width"]
        with self.assertRaisesRegex(GeometryContractError, "dimensions"):
            calibration_matrix(row)

    def test_analytic_fields_equal_existing_synthetic_fields_without_gaussian(self):
        row = source_row(.4)
        calibration = dial_calibration(row, calibration_matrix(row), GEOMETRY_CONFIG)
        old = geometry_targets(calibration, 256, 64, GEOMETRY_CONFIG)
        with patch("experiments.geoprr_v2_data.geometry_targets", side_effect=AssertionError("Gaussian Q called")), \
                patch("experiments.geoprr_v2_role_field_data.np.exp", side_effect=AssertionError("Gaussian Q called")):
            new = analytic_field_targets(calibration)
        for key in ("field_target", "field_mask", "roi_to_canonical", "virtual_point_xy"):
            torch.testing.assert_close(new[key], old[key], atol=0, rtol=0)
        self.assertEqual(new["geometry_audit"], calibration["audit"])
        self.assertNotIn("point_target", new)
        self.assertNotIn("gt_readout_error", new["geometry_audit"])

    def test_physical_endpoints_have_distinct_inbounds_virtual_roles(self):
        points = []
        for y in (0., 1.):
            row = source_row(y)
            calibration = dial_calibration(row, calibration_matrix(row), GEOMETRY_CONFIG)
            targets = analytic_field_targets(calibration)
            self.assertEqual(calibration["y"], y)
            self.assertAlmostEqual(targets["span_radians"].item(), 1.5 * math.pi, places=6)
            self.assertEqual(calibration["audit"]["physical_progress_error"], 0.)
            point = targets["virtual_point_xy"]
            self.assertTrue(bool(((point >= -.5) & (point <= 255.5)).all()))
            points.append(point)
        self.assertFalse(torch.equal(*points))

    def test_empty_ring_and_outside_virtual_role_are_rejected(self):
        row = source_row(0.)
        calibration = dial_calibration(row, calibration_matrix(row), GEOMETRY_CONFIG)
        empty = dict(calibration, roi_to_canonical=np.array([[.001, 0., 0.], [0., .001, 0.], [0., 0., 1.]]))
        with self.assertRaisesRegex(GeometryContractError, "no output-grid support"):
            analytic_field_targets(empty)
        outside = dict(calibration, roi_to_canonical=np.array([[1/200, 0., 0.], [0., 1/200, -.6], [0., 0., 1.]]),
                       y=1.)
        with self.assertRaisesRegex(GeometryContractError, "outside the ROI"):
            analytic_field_targets(outside)


if __name__ == "__main__":
    unittest.main()
