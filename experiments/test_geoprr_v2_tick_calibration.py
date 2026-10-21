"""Check the tick-only calibration contract on deterministic synthetic geometry."""
import unittest

import numpy as np

from experiments.geoprr_v2_tick_calibration import fit_ordered_tick_calibration
from experiments.geoprr_v2_height_profile import fit_height_geometry
from experiments import test_geoprr_v2_height_profile as height_fixture


def project(matrix, points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return mapped[:, :2] / mapped[:, 2, None]


def observed_ticks(matrix, count, span):
    angles = -np.arange(count) * span / (count - 1)
    canonical = np.column_stack((np.cos(angles), np.sin(angles)))
    return project(matrix, canonical) - .5, canonical


class TickCalibrationTests(unittest.TestCase):
    def test_body_frame_without_homography_or_span_inputs(self):
        fixture = height_fixture.HeightGeometryTests()
        theta, first_tick, span = -.4, .7, 4.
        pixels, gt_matrix = fixture.observation([.7, -.4, 3.], theta)
        box = fixture.bbox
        crop = np.array([[box[2]-box[0], 0., box[0]], [0., box[3]-box[1], box[1]], [0., 0., 1.]])
        normalize = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
        plane_to_pixel = crop @ np.linalg.inv(gt_matrix) @ normalize
        angles = first_tick - np.arange(20) * span / 19
        tick_world = np.column_stack((.8*np.cos(angles), .8*np.sin(angles)))
        ticks = project(plane_to_pixel, tick_world) - .5
        calibration = fit_ordered_tick_calibration(ticks)
        matrix = normalize @ calibration["G"] @ crop
        fitted = fit_height_geometry(fixture.xyz, pixels, box, matrix)
        delta = (-fitted["theta_radians"]) % (2*np.pi)
        self.assertAlmostEqual(delta / calibration["span_radians"], (first_tick-theta)/span, places=7)
        self.assertLess(fitted["pixel_max"], 1e-6)

    def test_exact_ordered_arc_with_unknown_perspective_and_span(self):
        matrices = [np.array([[350., 0, 900], [0, 350., 500], [0, 0, 1.]]),
                    np.array([[350., 80., 900], [-70., 280., 500], [.18, -.12, 1.]])]
        for count in (7, 20, 36):
            for degrees in (60, 180, 350):
                for matrix in matrices:
                    with self.subTest(count=count, degrees=degrees, matrix=matrix.tolist()):
                        span = np.deg2rad(degrees)
                        pixels, canonical = observed_ticks(matrix, count, span)
                        result = fit_ordered_tick_calibration(pixels)
                        self.assertEqual(result["status"], "ok")
                        self.assertAlmostEqual(result["span_radians"], span, delta=1e-6)
                        self.assertLess(result["max_error_pixels"], 1e-6)
                        self.assertGreater(result["front_circle_margin"], 0)
                        np.testing.assert_allclose(project(result["G"], pixels + .5), canonical, atol=1e-6)

    def test_pixel_similarity_does_not_change_recovered_angles(self):
        matrix = np.array([[350., 80., 900], [-70., 280., 500], [.18, -.12, 1.]])
        pixels, _ = observed_ticks(matrix, 20, np.deg2rad(245))
        angle = .4
        similarity = np.array([[1.7*np.cos(angle), -1.7*np.sin(angle), 100.],
                               [1.7*np.sin(angle), 1.7*np.cos(angle), -30.], [0, 0, 1.]])
        transformed = project(similarity, pixels + .5) - .5
        first = fit_ordered_tick_calibration(pixels)
        second = fit_ordered_tick_calibration(transformed)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertAlmostEqual(first["span_radians"], second["span_radians"], delta=1e-6)
        probes = np.array([[.2, -.4], [.7, .2], [0., 0.]])
        np.testing.assert_allclose(project(first["G"], project(matrix, probes)), probes, atol=1e-6)
        np.testing.assert_allclose(project(second["G"], project(similarity @ matrix, probes)), probes, atol=1e-6)

    def test_four_ticks_cannot_calibrate_unknown_spacing(self):
        for count in (0, 1, 3, 4):
            result = fit_ordered_tick_calibration(np.zeros((count, 2)))
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["G"])

    def test_quantized_input_retains_full_population_and_is_deterministic(self):
        matrix = np.array([[350., 80., 900], [-70., 280., 500], [.18, -.12, 1.]])
        pixels, _ = observed_ticks(matrix, 20, np.deg2rad(245))
        pixels = np.round(pixels)
        first = fit_ordered_tick_calibration(pixels)
        second = fit_ordered_tick_calibration(pixels)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["tick_count"], 20)
        self.assertEqual(first["span_radians"], second["span_radians"])
        self.assertEqual(first["sse_pixels"], second["sse_pixels"])
        self.assertTrue(first["geometric_refinement"])
        self.assertEqual(first["pixel_center_offset"], .5)
        self.assertLess(first["rmse_pixels"], 1.)


if __name__ == "__main__":
    unittest.main()
