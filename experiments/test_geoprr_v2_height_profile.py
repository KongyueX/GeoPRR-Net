"""Construct checks for full source-pixel geometry without camera labels."""
import unittest

import numpy as np

from experiments.geoprr_v2_height_profile import fit_height_geometry


class HeightGeometryTests(unittest.TestCase):
    xyz = np.array([[-.3, -.1, .02], [.4, -.1, .02], [.35, .15, .08], [-.2, .13, .08],
                    [0., .03, .12], [.1, -.02, .16], [-.18, .04, .16], [.22, .06, .04]])
    bbox = np.array([41., 29., 1001., 769.])

    def observation(self, camera, theta, translation=(.12, -.08), material=None):
        material = self.xyz if material is None else material
        camera = np.asarray(camera, dtype=np.float64)
        forward = np.array([.1, -.2, 0.]) - camera
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 1., 0.])
        right /= np.linalg.norm(right)
        rotation = np.stack((right, np.cross(forward, right), forward))
        intrinsic = np.array([[900., 0., 530.], [0., 850., 390.], [0., 0., 1.]])
        projection = intrinsic @ np.column_stack((rotation, -rotation @ camera))
        spin = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        world = material.copy()
        world[:, :2] = .8 * world[:, :2] @ spin.T + translation
        homogeneous = np.column_stack((world, np.ones(len(world)))) @ projection.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2, None] - .5
        crop_to_pixel = np.array([[960., 0., 41.], [0., 740., 29.], [0., 0., 1.]])
        world_to_canonical = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
        matrix = world_to_canonical @ np.linalg.inv(projection[:, [0, 1, 3]]) @ crop_to_pixel
        return pixels, matrix

    def test_pose_camera_translation_and_height_gauge(self):
        for camera in ([.7, -.4, 3.], [-.4, .5, 1.8]):
            for theta in (-2.4, .4, 2.1):
                for translation in ((.12, -.08), (-.3, .2)):
                    pixels, matrix = self.observation(camera, theta, translation)
                    for height_scale in (1., 2.):
                        material = self.xyz.copy()
                        material[:, 2] *= height_scale
                        fit = fit_height_geometry(material, pixels, self.bbox, matrix)
                        self.assertAlmostEqual(fit["theta_radians"], theta, places=8)
                        self.assertLess(fit["pixel_max"], 1e-7)
                        self.assertFalse(fit["readout_rotation_ambiguous"])
                        self.assertGreater(fit["camera_denominator_min"], 0.)

    def test_same_plane_retains_angle_with_nonunique_camera_nuisance(self):
        for height in (0., .08):
            material = self.xyz.copy()
            material[:, 2] = height
            pixels, matrix = self.observation([.7, -.4, 3.], 2.4, material=material)
            fit = fit_height_geometry(material, pixels, self.bbox, matrix)
            self.assertAlmostEqual(fit["theta_radians"], 2.4, places=8)
            self.assertLess(fit["pixel_max"], 1e-7)
            self.assertLess(fit["starts"][0]["normalized_pixel_jacobian_rank"], 7)

    def test_wrong_material_identity_keeps_nonzero_pixel_error(self):
        pixels, matrix = self.observation([.7, -.4, 3.], .4)
        fit = fit_height_geometry(self.xyz, pixels[[1, 0, 2, 3, 4, 5, 6, 7]], self.bbox, matrix)
        self.assertGreater(fit["pixel_rmse"], 1.)
        self.assertTrue(all(item["pixel_rmse"] > 1. for item in fit["starts"]))


if __name__ == "__main__":
    unittest.main()
