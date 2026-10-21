"""Synthetic checks for shared tick/material pixel reprojection geometry."""
import unittest

import numpy as np

from experiments.geoprr_v2_joint_geometry import _joint_prediction, _projection_failure_reason, fit_joint_geometry
from experiments.geoprr_v2_height_profile import fit_height_geometry
from experiments import test_geoprr_v2_height_profile as height_fixture
from experiments.test_geoprr_v2_tick_calibration import project


def observation(count, degrees, progress=.37):
    fixture = height_fixture.HeightGeometryTests()
    span, first = np.deg2rad(degrees), .7
    theta = first - progress*span
    pixels, mapping = fixture.observation([.7, -.4, 3.], theta)
    box = fixture.bbox
    crop = np.array([[box[2]-box[0], 0., box[0]], [0., box[3]-box[1], box[1]], [0., 0., 1.]])
    normalize = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
    plane_to_pixel = crop @ np.linalg.inv(mapping) @ normalize
    angles = first - np.arange(count)*span/(count-1)
    ticks = project(plane_to_pixel, .8*np.column_stack((np.cos(angles), np.sin(angles)))) - .5
    return fixture.xyz, pixels, ticks


class JointGeometryTests(unittest.TestCase):
    def test_unknown_camera_span_and_noncoplanar_directed_material(self):
        for count in (7, 20):
            for degrees in (70, 180, 330):
                with self.subTest(count=count, degrees=degrees):
                    result = fit_joint_geometry(*observation(count, degrees))
                    self.assertEqual(result["status"], "ok")
                    self.assertAlmostEqual(result["span_radians"], np.deg2rad(degrees), delta=1e-6)
                    self.assertAlmostEqual(result["raw_prediction"], .37, delta=1e-7)
                    self.assertLess(result["pixel_max"], 1e-6)
                    self.assertGreater(result["projection_depth_min"], 0.)
                    self.assertEqual(len(result["starts"]), 4)

    def test_direction_and_fixed_midpoint_unwrap(self):
        for progress in (.02, .98):
            with self.subTest(progress=progress):
                result = fit_joint_geometry(*observation(20, 330, progress))
                self.assertEqual(result["status"], "ok")
                self.assertAlmostEqual(result["raw_prediction"], progress, delta=1e-7)

    def test_joint_pixel_jacobian_matches_finite_difference(self):
        xyz = height_fixture.HeightGeometryTests.xyz
        parameters = np.array([1.2, .1, .3, -.2, .9, -.1, .1, -.07, 3.4,
                               .7, -.3, .1, -.08, .12, -.07, .3])
        _, jacobian, _ = _joint_prediction(parameters, xyz, 7)
        numerical = np.empty_like(jacobian)
        step = 1e-6
        for j in range(len(parameters)):
            high, low = parameters.copy(), parameters.copy()
            high[j] += step
            low[j] -= step
            numerical[:, j] = ((_joint_prediction(high, xyz, 7)[0]
                                - _joint_prediction(low, xyz, 7)[0]) / (2*step)).ravel()
        np.testing.assert_allclose(jacobian, numerical, atol=1e-8, rtol=1e-6)

    def test_positive_final_depth_requiring_clamp_is_rejected(self):
        xyz = np.array([[-1., 0., .1], [0., 0., .1], [1., 0., .1]])
        parameters = np.array([1., 0., 0., 0., 1., 0., 1/np.sqrt(3), 0., 1.,
                               1., 0., -.94+2e-13, 0., 0., 0., .3])
        prediction, _, depth = _joint_prediction(parameters, xyz, 5)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.all(depth > 0))
        self.assertLess(depth[5], 1e-12)
        self.assertEqual(_projection_failure_reason(depth), "final_projection_requires_numerical_clamp")

    def test_low_tick_cost_does_not_validate_wrong_material_identity(self):
        xyz, pixels, ticks = observation(7, 70)
        correct = fit_joint_geometry(xyz, pixels, ticks)
        wrong = pixels.copy()
        wrong[[0, 1]] = wrong[[1, 0]]
        contradicted = fit_joint_geometry(xyz, wrong, ticks)
        self.assertEqual(correct["status"], "ok")
        self.assertLess(correct["tick_pixel_rmse"], 1e-6)
        self.assertLess(correct["material_pixel_rmse"], 1e-6)
        if contradicted["status"] == "ok":
            self.assertGreater(contradicted["pixel_rmse"], 1.)
        self.assertAlmostEqual(correct["prediction"], .37, delta=1e-7)

    def test_circle_boost_low_tick_cost_is_contradicted_by_material(self):
        # This fixed alternative demonstrates rejection, not global uniqueness.
        span, boost = np.pi/3, .1
        half_wrong = 2*np.arctan(np.exp(boost)*np.tan(span/4))

        def rotation(angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

        ch, sh = np.cosh(boost), np.sinh(boost)
        circle_boost = np.array([[ch, 0., sh], [0., 1., 0.], [sh, 0., ch]])
        true_h = np.array([[350., 0., 530.], [0., 350., 390.], [0., 0., 1.]])
        wrong_h = true_h @ rotation(-span/2) @ circle_boost @ rotation(half_wrong)
        angles = -np.arange(7)*span/6
        observed_ticks = project(true_h, np.column_stack((np.cos(angles), np.sin(angles))))
        wrong_angles = -np.arange(7)*2*half_wrong/6
        alternative_ticks = project(wrong_h, np.column_stack((np.cos(wrong_angles), np.sin(wrong_angles))))
        tick_sse = float(np.sum((observed_ticks-alternative_ticks)**2))
        self.assertLess(np.sqrt(tick_sse/7), .25)
        self.assertGreater(abs(2*half_wrong-span), np.deg2rad(5.))
        xyz = np.array([[-.7, -.5, .02], [.7, -.5, .02], [.7, .5, .02], [-.7, .5, .02],
                        [-.55, -.3, .12], [.55, -.3, .12], [.55, .3, .12], [-.55, .3, .12],
                        [-.3, -.15, .07], [.3, -.15, .07], [.3, .15, .07], [-.3, .15, .07]])
        theta = -.25
        c, s = np.cos(theta), np.sin(theta)
        x, y, z = xyz.T
        material = np.column_stack((c*x-s*y+.04-.1*z, s*x+c*y-.08+.15*z, 1-.3*z))
        projected = material @ true_h.T
        pixels = projected[:, :2]/projected[:, 2, None]-.5
        bbox = np.array([0., 0., 1100., 800.])
        crop = np.diag([1100., 800., 1.])
        normalize = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
        true_body = fit_height_geometry(xyz, pixels, bbox, normalize @ np.linalg.inv(true_h) @ crop)
        wrong_body = fit_height_geometry(xyz, pixels, bbox, normalize @ np.linalg.inv(wrong_h) @ crop)
        self.assertLess(true_body["pixel_rmse"], 1e-7)
        self.assertGreater(wrong_body["pixel_rmse"], 1.)
        self.assertGreater(wrong_body["pixel_sse"], 10*tick_sse)
        self.assertGreater(tick_sse+wrong_body["pixel_sse"], true_body["pixel_sse"]+1.)


if __name__ == "__main__":
    unittest.main()
