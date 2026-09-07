from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from experiments.deeplabv3plus_roi import (
    DeepLabV3PlusROI,
    binary_segmentation_loss,
)
from experiments.evaluate_yolo11s_pose4kp import (
    _prediction_points,
    _square_training_input,
)
from experiments.roi_geometry_comparison import (
    build_evaluation_summary,
    decode_progress_from_keypoints,
    direction_from_pointer_probability,
    extract_syncg_keypoints,
    transform_points_homography,
)
from experiments.summarize_roi_geometry_full_experiment import (
    INDUSTRIAL_FIELD_SLUGS,
    SEEDS,
    SYNCG_SLUG,
    _geoprr_rows,
    _normalized_new_rows,
    _paired_group_bootstrap,
    _pooled_field_rows,
)
from experiments.roi_geometry_field import FIELD_DATASETS


class ROIGeometryComparisonTests(unittest.TestCase):
    def test_extracts_four_keypoints_in_declared_order(self) -> None:
        sample = SimpleNamespace(
            sample_id="example",
            metadata={
                "keypoints": [
                    {"type": "ScaleMark", "all_kp": [[1, 2], [3, 4], [5, 6]]},
                    {"type": "Pointer", "origin_kp": [7, 8], "outside_kp": [9, 10]},
                ]
            },
        )
        actual = extract_syncg_keypoints(sample)
        np.testing.assert_allclose(actual, [[7, 8], [9, 10], [1, 2], [5, 6]])

    def test_geometric_decoder_returns_mid_arc_progress(self) -> None:
        points = np.asarray(
            [[50.0, 50.0], [60.0, 40.0], [50.0, 30.0], [70.0, 50.0]],
            dtype=np.float32,
        )
        progress, failure, telemetry = decode_progress_from_keypoints(points)
        self.assertIsNone(failure)
        self.assertAlmostEqual(float(progress), 0.5, places=6)
        self.assertAlmostEqual(telemetry["arc_degrees"], 90.0, places=5)

    def test_segmentation_direction_selects_component_touching_pivot(self) -> None:
        probability = np.zeros((64, 64), dtype=np.float32)
        pivot = np.asarray([31.0, 33.0], dtype=np.float32)
        tip = np.asarray([49.0, 10.0], dtype=np.float32)
        cv2.line(
            probability,
            tuple(np.rint(pivot).astype(int)),
            tuple(np.rint(tip).astype(int)),
            1.0,
            thickness=3,
        )
        cv2.circle(probability, (5, 58), 4, 1.0, thickness=-1)
        direction, failure, telemetry = direction_from_pointer_probability(
            probability, pivot
        )
        self.assertIsNone(failure)
        expected = tip - pivot
        expected /= np.linalg.norm(expected)
        self.assertGreater(float(np.dot(direction, expected)), 0.98)
        self.assertGreater(telemetry["component_pixels"], 20)

    def test_homography_transforms_all_four_points(self) -> None:
        points = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        matrix = np.asarray([[2, 0, 3], [0, 4, 5], [0, 0, 1]], dtype=np.float32)
        actual = transform_points_homography(points, matrix)
        np.testing.assert_allclose(actual, [[3, 5], [5, 5], [3, 9], [5, 9]])

    def test_failure_rows_receive_error_one_in_summary(self) -> None:
        rows = [
            {"condition": "clean", "absolute_error": 0.02, "status": "pass"},
            {"condition": "clean", "absolute_error": 1.0, "status": "fail"},
        ]
        summary = build_evaluation_summary(
            rows, method="example", seed=1, configuration={}
        )
        self.assertAlmostEqual(summary["all_conditions"]["nmae"], 0.51)
        self.assertAlmostEqual(summary["all_conditions"]["coverage"], 0.5)

    def test_deeplab_forward_and_loss_are_finite(self) -> None:
        model = DeepLabV3PlusROI(imagenet_pretrained=False).eval()
        images = torch.randn(1, 3, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 20:45, 30:34] = 1.0
        with torch.no_grad():
            logits = model(images)
            loss = binary_segmentation_loss(logits, target)
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        self.assertTrue(math.isfinite(float(loss)))

    def test_pose_result_adapter_preserves_four_point_order(self) -> None:
        points = torch.tensor(
            [[[10.0, 10.0], [15.0, 5.0], [5.0, 10.0], [10.0, 5.0]]]
        )
        result = SimpleNamespace(
            keypoints=SimpleNamespace(xy=points, conf=torch.full((1, 4), 0.8)),
            boxes=SimpleNamespace(conf=torch.tensor([0.9])),
        )
        actual, failure, telemetry = _prediction_points(
            result, minimum_keypoint_confidence=0.05
        )
        self.assertIsNone(failure)
        np.testing.assert_allclose(actual, points[0].numpy())
        self.assertAlmostEqual(telemetry["box_confidence"], 0.9, places=6)

    def test_yolo_evaluation_matches_square_training_resolution(self) -> None:
        image = np.zeros((40, 90, 3), dtype=np.uint8)
        actual = _square_training_input(image, image_size=64)
        self.assertEqual(actual.shape, (64, 64, 3))
        self.assertTrue(actual.flags.c_contiguous)

    def test_prediction_normalizer_keeps_failures_in_denominator(self) -> None:
        records = (
            {
                "sample_id": "a",
                "scene_stem": "scene-1",
                "condition": "clean",
                "status": "pass",
                "absolute_error": 0.02,
            },
            {
                "sample_id": "b",
                "scene_stem": "scene-2",
                "condition": "clean",
                "status": "fail",
                "absolute_error": 0.0,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            actual = _normalized_new_rows(path)
        self.assertEqual(actual[0]["absolute_error"], 0.02)
        self.assertEqual(actual[1]["absolute_error"], 1.0)
        self.assertEqual(actual[1]["status"], "fail")

    def test_syncg_geoprr_loader_accepts_direct_mett_rows(self) -> None:
        payload = {
            "per_sample_condition": [
                {
                    "sample_id": "a",
                    "scene_stem": "scene-1",
                    "condition": "clean",
                    "mett": {"absolute_error": 0.03, "prediction": 0.4},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_20262020" / "full" / "syncg.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            actual = _geoprr_rows(
                Path(directory), seed=20262020, dataset_slug=SYNCG_SLUG
            )
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["group_id"], "scene-1")
        self.assertEqual(actual[0]["status"], "pass")
        self.assertAlmostEqual(actual[0]["absolute_error"], 0.03)

    def test_paired_group_bootstrap_uses_group_resampling(self) -> None:
        candidate = {
            ("a", "clean"): {"absolute_error": 0.1, "group_id": "g1"},
            ("b", "clean"): {"absolute_error": 0.2, "group_id": "g2"},
        }
        comparator = {
            ("a", "clean"): {"absolute_error": 0.2, "group_id": "g1"},
            ("b", "clean"): {"absolute_error": 0.3, "group_id": "g2"},
        }
        actual = _paired_group_bootstrap(
            candidate,
            comparator,
            conditions=("clean",),
            replicates=100,
            seed=7,
        )
        self.assertAlmostEqual(actual["observed"], -0.1)
        self.assertAlmostEqual(actual["ci95_lower"], -0.1)
        self.assertAlmostEqual(actual["ci95_upper"], -0.1)
        self.assertTrue(actual["geoprr_advantage_ci_excludes_zero"])

    def test_field_pool_prefixes_dataset_identity(self) -> None:
        registry = {
            dataset_slug: {
                "geoprr": {
                    seed: (
                        {
                            "sample_id": "shared-sample",
                            "group_id": "shared-group",
                            "condition": "clean",
                            "status": "pass",
                            "absolute_error": 0.1,
                        },
                    )
                    for seed in SEEDS
                }
            }
            for dataset_slug in FIELD_DATASETS
        }
        actual = _pooled_field_rows(registry, method="geoprr")
        for seed in SEEDS:
            self.assertEqual(len(actual[seed]), len(INDUSTRIAL_FIELD_SLUGS))
            self.assertEqual(
                len({row["sample_id"] for row in actual[seed]}),
                len(INDUSTRIAL_FIELD_SLUGS),
            )
            self.assertEqual(
                len({row["group_id"] for row in actual[seed]}),
                len(INDUSTRIAL_FIELD_SLUGS),
            )


if __name__ == "__main__":
    unittest.main()
