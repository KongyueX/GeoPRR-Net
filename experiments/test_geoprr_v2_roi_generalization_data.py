"""Focused CPU checks for source ROI expansion and calibration lifetime."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from experiments.geoprr_v2_data import (
    GeometryContractError, crop_and_resize, dial_calibration, transform_points,
)
from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset, reference_geometry_targets
from experiments.geoprr_v2_roi_generalization_data import ROIPaddingReferenceRelationDataset, _expanded_bbox


NEUTRAL_PHOTO = (1., 0., 1.)
PADDING_MODULE = "experiments.geoprr_v2_roi_generalization_data"


class SourceROIPaddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        yy, xx = np.indices((96, 100), dtype=np.uint8)
        self.bgr = np.stack((xx, yy, xx + yy), -1)
        self.image = self.root / "source.png"
        cv2.imwrite(str(self.image), self.bgr)
        self.bbox = [10, 8, 90, 88]
        self.center = np.array([50., 48.])
        self.radius = 32.
        self.marks = [list(self.center + self.radius * np.array([math.cos(a), math.sin(a)]))
                      for a in np.deg2rad([0., 60., 120., 180.])]
        self.rows = [{"sample_id": f"fixture_{index}", "group_id": "source_fixture",
                      "image_path": str(self.image), "normalized_progress": target,
                      "metadata": {"homography": np.eye(3).tolist(), "dial_bbox": list(self.bbox),
                                   "long_interval_degree": 60., "long_num": 4, "pointer_angle": 180 * target,
                                   "image_width": 100, "image_height": 96,
                                   "keypoints": [{"type": "ScaleMark", "all_kp": self.marks}]}}
                     for index, target in enumerate((.25, .75))]
        self.config = {"image_size": 128, "output_size": 32}

    def dataset(self, *, augment=True, probability=1., pair_manifest=None, rows=None):
        return ROIPaddingReferenceRelationDataset(self.rows if rows is None else rows, data_root=self.root,
            data_config=self.config, augment=augment, pair_manifest=pair_manifest,
            roi_policy={"probability": probability, "max_padding": .3})

    def assert_samples_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if torch.is_tensor(actual[key]):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            else:
                self.assertEqual(actual[key], expected[key])

    def test_independent_side_padding_uses_original_width_and_height(self):
        bbox = [20, 30, 120, 80]
        expanded = _expanded_bbox(bbox, [.1, .2, .3, .4])
        self.assertEqual(expanded, [10., 20., 150., 100.])
        self.assertLessEqual(expanded[0], bbox[0])
        self.assertLessEqual(expanded[1], bbox[1])
        self.assertGreaterEqual(expanded[2], bbox[2])
        self.assertGreaterEqual(expanded[3], bbox[3])

    def test_zero_probability_and_validation_are_original_identity(self):
        for augment, probability in ((True, 0.), (False, 1.)):
            with self.subTest(augment=augment):
                dataset = self.dataset(augment=augment, probability=probability)
                original = ReferenceRelationDataset(self.rows, data_root=self.root, data_config=self.config, augment=augment)
                before = deepcopy(dataset.rows)
                with patch(f"{PADDING_MODULE}.np.random.random", side_effect=AssertionError("identity drew padding")):
                    actual = dataset._sample(0, NEUTRAL_PHOTO if augment else None)
                expected = original._sample(0, NEUTRAL_PHOTO if augment else None)
                self.assert_samples_equal(actual, expected)
                self.assertEqual(dataset.rows, before)
                self.assertEqual(dataset.supervision_contract(), original.supervision_contract())

    def test_clipped_full_rgb_crop_updates_all_geometry_without_mutating_source(self):
        dataset = self.dataset()
        original_rows = deepcopy(dataset.rows)
        original_contract = dataset.supervision_contract()
        padding = np.array([.3, .2, .1, .3])
        expanded = _expanded_bbox(self.bbox, padding)
        _, original_to_old = crop_and_resize(self.bgr, self.bbox, 128)
        expected_roi, original_to_new = crop_and_resize(self.bgr, expanded, 128)
        old_calibration = dial_calibration(self.rows[0], original_to_old, dataset.base.geometry_config)
        expected_h = old_calibration["roi_to_canonical"] @ original_to_old @ np.linalg.inv(original_to_new)
        with patch(f"{PADDING_MODULE}.np.random.random", return_value=0.), \
             patch(f"{PADDING_MODULE}.np.random.uniform", return_value=padding) as draw:
            with dataset.base.source_roi(0):
                temporary = dataset.rows[0]
                self.assertEqual(temporary["metadata"]["homography_bbox"], self.bbox)
                self.assertEqual(temporary["metadata"]["dial_bbox"], expanded)
                sample = dataset.base._sample(0, NEUTRAL_PHOTO)
                span = dataset.base._calibrations[0]["angles"][-1]
        draw.assert_called_once_with(0, .3, 4)
        expected_rgb = cv2.cvtColor(expected_roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        expected_image = torch.from_numpy(expected_rgb.transpose(2, 0, 1).copy())
        expected_image = (expected_image - expected_image.new_tensor([.485, .456, .406])[:, None, None]) / expected_image.new_tensor([.229, .224, .225])[:, None, None]
        torch.testing.assert_close(sample["image"], expected_image, atol=1e-6, rtol=0)
        np.testing.assert_allclose(sample["roi_to_canonical"].numpy(), expected_h, atol=1e-7, rtol=0)
        mapped = transform_points(transform_points(np.array(self.marks), original_to_new), expected_h)
        expected_marks = np.column_stack((np.cos(np.deg2rad([0., 60., 120., 180.])),
                                          np.sin(np.deg2rad([0., 60., 120., 180.]))))
        np.testing.assert_allclose(mapped, expected_marks, atol=1e-12, rtol=0)
        centers = (np.arange(32) + .5) * 4 - .5
        xx, yy = np.meshgrid(centers, centers)
        canonical = transform_points(np.stack((xx, yy), -1), expected_h)
        angles = np.mod(np.arctan2(canonical[..., 1], canonical[..., 0]), 2 * math.pi)
        radii = np.linalg.norm(canonical, axis=-1)
        expected_mask = ((radii >= .8) & (radii <= 1.2) & (angles <= math.pi)).astype(np.float32)
        np.testing.assert_array_equal(sample["field_mask"].numpy()[0], expected_mask)
        np.testing.assert_array_equal(sample["support_target"].numpy()[0], expected_mask)
        np.testing.assert_allclose(sample["field_target"].numpy(), np.clip(angles / math.pi, 0, 1), atol=1e-7, rtol=0)
        source_point = self.center + self.radius * np.array([math.cos(math.pi * .25), math.sin(math.pi * .25)])
        expected_point = transform_points(source_point[None], original_to_new)[0]
        np.testing.assert_allclose(sample["virtual_point_xy"].numpy(), expected_point, atol=1e-5, rtol=0)
        self.assertAlmostEqual(float(sample["point_target"].sum()), 1., places=6)
        labels = reference_geometry_targets(sample["roi_to_canonical"][None], sample["y"][None],
                                            torch.tensor([span]), image_size=128)
        expected_center = (transform_points(self.center[None], original_to_new) + .5) / 128
        np.testing.assert_allclose(labels["center"].numpy(), expected_center, atol=1e-7, rtol=0)
        self.assertEqual(dataset.rows, original_rows)
        self.assertEqual(dataset.supervision_contract(), original_contract)
        self.assertEqual(dataset.base._calibrations, {})

    def test_repeated_source_sampling_refreshes_calibration_and_reads_span_before_restore(self):
        rows = deepcopy(self.rows)
        rows[0]["metadata"]["homography_bbox"] = list(self.bbox)
        dataset = self.dataset(rows=rows)
        original_row = dataset.rows[0]
        original = ReferenceRelationDataset._sample(dataset, 0, NEUTRAL_PHOTO)
        original_calibration = dataset.base._calibrations[0]
        before = deepcopy(dataset.rows)
        paddings = [np.array([.05, .1, .15, .2]), np.array([.3, .2, .1, .05])]
        with patch(f"{PADDING_MODULE}.np.random.random", return_value=0.), \
             patch(f"{PADDING_MODULE}.np.random.uniform", side_effect=paddings), \
             patch("experiments.geoprr_v2_data.dial_calibration", wraps=dial_calibration) as calibrate:
            first = dataset._sample(0, NEUTRAL_PHOTO)
            second = dataset._sample(0, NEUTRAL_PHOTO)
        self.assertEqual(calibrate.call_count, 2)
        self.assertFalse(torch.equal(first["roi_to_canonical"], second["roi_to_canonical"]))
        self.assertFalse(torch.equal(first["roi_to_canonical"], original["roi_to_canonical"]))
        self.assertEqual(float(first["span_radians"]), float(original["span_radians"]))
        self.assertEqual(float(second["span_radians"]), float(original["span_radians"]))
        self.assertIs(dataset.rows[0], original_row)
        self.assertIs(dataset.base._calibrations[0], original_calibration)
        self.assertEqual(dataset.rows, before)

    def test_pair_exposure_and_shared_photo_keep_independent_roi_draws(self):
        pairs = self.root / "pairs.jsonl"
        pairs.write_text(json.dumps({"sample_a": "fixture_0", "sample_b": "fixture_1"}) + "\n")
        dataset = self.dataset(pair_manifest=pairs)
        original = ReferenceRelationDataset(self.rows, data_root=self.root, data_config=self.config,
                                             augment=True, pair_manifest=pairs)
        contract = original.supervision_contract()
        self.assertEqual(dataset.items, original.items)
        self.assertEqual(dataset.supervision_contract(), contract)
        self.assertEqual((dataset.unique_image_count, dataset.item_count, len(dataset.edges)), (2, 2, 1))
        paddings = [np.array([.05, .1, .15, .2]), np.array([.3, .2, .1, .05])]
        photos = []
        original_sample = dataset.base._sample

        def sampled(index, photometric):
            photos.append(photometric)
            return original_sample(index, photometric)

        with patch.object(dataset.base, "_photometric", return_value=NEUTRAL_PHOTO), \
             patch.object(dataset.base, "_sample", side_effect=sampled), \
             patch(f"{PADDING_MODULE}.np.random.random", return_value=0.), \
             patch(f"{PADDING_MODULE}.np.random.uniform", side_effect=paddings) as draw:
            sample = dataset[0]
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(photos, [NEUTRAL_PHOTO, NEUTRAL_PHOTO])
        self.assertEqual(sample["sample_id"], "fixture_0")
        self.assertEqual(sample["pair_sample_id"], "fixture_1")
        self.assertTrue(sample["pair_valid"])
        self.assertEqual(float(sample["y"]), .25)
        self.assertEqual(float(sample["pair_y"]), .75)
        self.assertFalse(torch.equal(sample["roi_to_canonical"], sample["pair_roi_to_canonical"]))
        self.assertEqual(dataset.supervision_contract(), contract)
        self.assertEqual(dataset.base._calibrations, {})

    def test_training_geometry_failure_restores_source_and_validation_keeps_parent_policy(self):
        rows = deepcopy(self.rows)
        rows[0]["metadata"]["pointer_angle"] = 1.
        training = self.dataset(rows=rows)
        original_row = training.rows[0]
        with patch(f"{PADDING_MODULE}.np.random.random", return_value=0.), \
             patch(f"{PADDING_MODULE}.np.random.uniform", return_value=np.full(4, .1)):
            with self.assertRaises(GeometryContractError):
                training._sample(0, NEUTRAL_PHOTO)
        self.assertIs(training.rows[0], original_row)
        self.assertEqual(training.rows, rows)
        self.assertEqual(training.base._calibrations, {})
        validation = self.dataset(augment=False, rows=rows)
        original = ReferenceRelationDataset(rows, data_root=self.root, data_config=self.config, augment=False)
        self.assert_samples_equal(validation[0], original[0])
        self.assertTrue(validation[0]["input_failure"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
