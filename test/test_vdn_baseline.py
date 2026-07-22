"""Deterministic tests for the external VDN baseline adapter."""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_vdn_baseline import (
    _dialbench_summary,
    _load_shared_predictions,
)
from experiments.robustness_degradations import ROBUSTNESS_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDNSample,
    affine_for_dial,
    angular_error_degrees,
    generate_vdn_targets,
    grouped_train_val_split,
    image_angle_from_direction,
    normalize_reference_points,
    predict_directions,
    reference_angles,
    reading_from_pointer_angle,
    sha256_file,
    summarize_scalar_predictions,
    transform_point,
)
from experiments.summarize_vdn_comparison import _paired_comparison
from experiments.verify_vdn_run import _state_health


def _sample(sample_id: str, group_id: str) -> VDNSample:
    return VDNSample(
        sample_id=sample_id,
        group_id=group_id,
        dataset="SyncG",
        split="train",
        image_path="unused.jpg",
        dial_bbox=(10.0, 20.0, 110.0, 100.0),
        pointer_tip=(70.0, 35.0),
        pointer_tail=(60.0, 60.0),
        ground_truth=0.5,
        scale_start=0.0,
        scale_end=1.0,
        metadata={},
    )


class VDNBaselineTest(unittest.TestCase):
    def test_affine_centers_square_crop_and_rotates_points(self):
        matrix = affine_for_dial(
            (10.0, 20.0, 110.0, 100.0),
            output_size=200,
            expansion=1.0,
            rotation_degrees=90.0,
        )
        center = transform_point((60.0, 60.0), matrix)
        right = transform_point((110.0, 60.0), matrix)
        np.testing.assert_allclose(center, (100.0, 100.0), atol=1e-5)
        np.testing.assert_allclose(right, (100.0, 0.0), atol=1e-5)

    def test_targets_match_tip_peak_and_tail_to_tip_direction(self):
        heatmap, vector_map, direction = generate_vdn_targets(
            (256.0, 192.0),
            (192.0, 192.0),
            image_size=384,
            heatmap_size=96,
        )
        self.assertEqual(tuple(heatmap.shape), (1, 96, 96))
        self.assertEqual(tuple(vector_map.shape), (2, 96, 96))
        self.assertAlmostEqual(float(heatmap[0, 48, 64]), 1.0)
        torch.testing.assert_close(direction, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(
            vector_map[:, 48, 64],
            torch.tensor([1.0, 0.0]),
        )

    def test_targets_match_pinned_official_code_when_available(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vendor"
            / "VectorDetectionNetwork"
        )
        dataset_source = source / "libs" / "dataset" / "JointsDataset.py"
        if not dataset_source.is_file():
            self.skipTest("ignored external VDN checkout is unavailable")
        sys.path.insert(0, str(source))
        try:
            spec = importlib.util.spec_from_file_location(
                "external_vdn_joints_dataset_test",
                dataset_source,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(source))
        official = object.__new__(module.JointsDataset)
        official.num_joints = 1
        official.heatmap_size = np.asarray([96, 96])
        official.image_size = np.asarray([384, 384])
        official.sigma = 3
        official.target_type = "gaussian"
        tip = (12.0, 20.0)
        tail = (180.0, 180.0)
        joints = np.asarray(
            [[[*tip, *tail, 1.0]]],
            dtype=np.float32,
        )
        official_heatmap, official_vectors = official.generate_target(joints)
        heatmap, vectors, _ = generate_vdn_targets(
            tip,
            tail,
            image_size=384,
            heatmap_size=96,
        )
        np.testing.assert_array_equal(official_heatmap, heatmap.numpy())
        np.testing.assert_array_equal(official_vectors.squeeze(0), vectors.numpy())
        official_affine = module.get_affine_transform(
            np.asarray([60.0, 60.0], dtype=np.float32),
            np.asarray([0.625, 0.625], dtype=np.float32),
            30.0,
            np.asarray([384, 384]),
        )
        local_affine = affine_for_dial(
            (10.0, 20.0, 110.0, 100.0),
            output_size=384,
            rotation_degrees=30.0,
        )
        np.testing.assert_allclose(local_affine, official_affine, atol=1e-5)

    def test_pinned_official_adam_does_not_apply_yaml_weight_decay(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vendor"
            / "VectorDetectionNetwork"
            / "libs"
            / "utils"
            / "utils.py"
        )
        if not source.is_file():
            self.skipTest("ignored external VDN checkout is unavailable")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        optimizer_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_optimizer"
        )
        adam_calls = [
            node
            for node in ast.walk(optimizer_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Adam"
        ]
        self.assertEqual(len(adam_calls), 1)
        self.assertNotIn(
            "weight_decay",
            {keyword.arg for keyword in adam_calls[0].keywords},
        )

    def test_prediction_samples_vector_at_heatmap_peak(self):
        heatmap = torch.zeros(2, 1, 4, 4)
        vectors = torch.zeros(2, 2, 4, 4)
        heatmap[0, 0, 1, 2] = 0.9
        vectors[0, :, 1, 2] = torch.tensor([3.0, 4.0])
        heatmap[1, 0, 3, 0] = 0.8
        directions, confidence, valid = predict_directions(heatmap, vectors)
        torch.testing.assert_close(directions[0], torch.tensor([0.6, 0.8]))
        torch.testing.assert_close(confidence, torch.tensor([0.9, 0.8]))
        self.assertEqual(valid.tolist(), [True, False])
        error = angular_error_degrees(
            directions[:1],
            torch.tensor([[0.6, 0.8]]),
        )
        self.assertAlmostEqual(float(error[0]), 0.0, places=4)

    def test_image_angle_and_reading_use_production_convention(self):
        self.assertAlmostEqual(image_angle_from_direction((0.0, 1.0)), 0.0)
        self.assertAlmostEqual(image_angle_from_direction((-1.0, 0.0)), 90.0)
        self.assertAlmostEqual(image_angle_from_direction((0.0, -1.0)), 180.0)
        self.assertAlmostEqual(image_angle_from_direction((1.0, 0.0)), 270.0)
        reading, progress = reading_from_pointer_angle(
            270.0,
            start_angle=180.0,
            range_angle=270.0,
            scale_start=0.0,
            scale_end=60.0,
        )
        self.assertAlmostEqual(progress, 1.0 / 3.0)
        self.assertAlmostEqual(reading, 20.0)

    def test_reference_points_follow_production_fallback_branches(self):
        start, end = normalize_reference_points(
            (20.0, 80.0),
            (22.0, 81.0),
            image_width=100,
        )
        self.assertEqual(start, (20.0, 80.0))
        self.assertIsNone(end)
        start_angle, range_angle, branch = reference_angles(
            (100, 100, 3),
            start,
            end,
        )
        self.assertEqual(branch, "start_only")
        self.assertAlmostEqual(range_angle, 270.0)
        self.assertTrue(0.0 <= start_angle < 360.0)

        start, end = normalize_reference_points(
            (80.0, 80.0),
            (20.0, 80.0),
            image_width=100,
        )
        self.assertEqual(start, (20.0, 80.0))
        self.assertEqual(end, (80.0, 80.0))
        _, range_angle, branch = reference_angles((100, 100, 3), start, end)
        self.assertEqual(branch, "start_and_end")
        self.assertAlmostEqual(range_angle, 270.0)

    def test_grouped_split_is_reproducible_without_leakage(self):
        samples = [
            _sample(f"sample-{group}-{index}", f"group-{group}")
            for group in range(10)
            for index in range(3)
        ]
        train_a, validation_a = grouped_train_val_split(
            samples,
            validation_fraction=0.2,
            seed=17,
        )
        train_b, validation_b = grouped_train_val_split(
            samples,
            validation_fraction=0.2,
            seed=17,
        )
        self.assertEqual(
            [sample.sample_id for sample in validation_a],
            [sample.sample_id for sample in validation_b],
        )
        self.assertEqual(len(train_a), 24)
        self.assertEqual(len(validation_a), 6)
        self.assertTrue(
            {sample.group_id for sample in train_a}.isdisjoint(
                {sample.group_id for sample in validation_a}
            )
        )

    def test_scalar_summary_penalizes_inference_failures(self):
        rows = [
            {
                "prediction": 0.51,
                "ground_truth": 0.50,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "a",
            },
            {
                "prediction": None,
                "ground_truth": 0.50,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "b",
            },
        ]
        summary = summarize_scalar_predictions(
            rows,
            bootstrap_iterations=20,
            seed=3,
        )
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["nmae"], 0.505)
        self.assertAlmostEqual(summary["acc_2pct"], 0.5)
        self.assertIsNotNone(summary["nmae_group_bootstrap_95ci"])

    def test_dialbench_accuracy_keeps_failures_in_full_denominator(self):
        rows = [
            {
                "prediction": 10.4,
                "ground_truth": 10.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
            {
                "prediction": None,
                "ground_truth": 10.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
            {
                "prediction": 0.0,
                "ground_truth": 0.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
        ]
        summary = _dialbench_summary(rows)
        self.assertEqual(summary["eligible_samples"], 3)
        self.assertEqual(summary["relative_samples"], 2)
        self.assertAlmostEqual(summary["ref_successful"], 0.002)
        self.assertAlmostEqual(
            summary["acc_epsilon_ref_le_1pct_e2e"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(summary["acc_theta_rel_lt_5pct_e2e"], 0.5)

    def test_checkpoint_health_rejects_collapsed_backbone(self):
        state = {
            "conv1.weight": torch.randn(8, 3, 3, 3) * 0.1,
            "layer1.0.conv1.weight": torch.randn(8, 8, 3, 3) * 0.05,
            "deconv_layers.0.weight": torch.randn(8, 8, 3, 3) * 0.001,
            "final_layer_hm.weight": torch.randn(1, 8, 1, 1) * 0.001,
            "final_layer_v.weight": torch.randn(2, 8, 3, 3) * 0.001,
        }
        health = _state_health(state)
        self.assertEqual(health["non_finite_tensors"], 0)
        state["conv1.weight"].zero_()
        with self.assertRaisesRegex(ValueError, "conv1 collapsed"):
            _state_health(state)

    def test_external_paired_comparison_keeps_failures_in_denominator(self):
        base = {
            "ground_truth": 0.5,
            "scale_start": 0.0,
            "scale_end": 1.0,
        }
        vdn = [
            {**base, "sample_id": "a", "group_id": "g1", "prediction": 0.6},
            {**base, "sample_id": "b", "group_id": "g2", "prediction": None},
        ]
        ours = [
            {
                **base,
                "sample_id": "a",
                "group_id": "g1",
                "predictions": {"ours": 0.55},
            },
            {
                **base,
                "sample_id": "b",
                "group_id": "g2",
                "predictions": {"ours": 0.7},
            },
        ]
        paired = _paired_comparison(vdn, ours, iterations=20, seed=4)
        self.assertAlmostEqual(paired["raw_vdn_nmae"], 0.55)
        self.assertAlmostEqual(paired["raw_ours_nmae"], 0.125)
        self.assertAlmostEqual(paired["delta_nmae_ours_minus_vdn"], -0.425)
        self.assertEqual(paired["common_successes"], 1)

    def test_shared_cache_requires_identical_degradation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            protocol = manifest.with_name(manifest.name + ".protocol.json")
            protocol.write_text('{"protocol":"test"}\n', encoding="utf-8")
            shared = root / "shared.jsonl"
            shared.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            meter_weights = root / "meter.pt"
            point_weights = root / "point.pt"
            meter_weights.write_bytes(b"meter")
            point_weights.write_bytes(b"point")
            signature = {
                "manifest_sha256": sha256_file(manifest),
                "manifest_protocol_sha256": sha256_file(protocol),
                "correction_mode": "off",
                "input_degradation": {
                    "condition": "clean",
                    "protocol": ROBUSTNESS_PROTOCOL,
                    "seed": 20260720,
                },
                "input_degradation_source_sha256": sha256_file(
                    PROJECT_DIR / "experiments" / "robustness_degradations.py"
                ),
                "weights_sha256": {
                    "meter_detector": sha256_file(meter_weights),
                    "keypoint_detector": sha256_file(point_weights),
                },
            }
            metadata = shared.with_name(shared.name + ".meta.json")
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            by_id, _ = _load_shared_predictions(
                shared,
                manifest=manifest,
                rows=[{"sample_id": "a"}],
                condition="clean",
                degradation_seed=20260720,
                meter_weights=meter_weights,
                point_weights=point_weights,
            )
            self.assertEqual(set(by_id), {"a"})

            signature["input_degradation"]["protocol"] = "different"
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "degradation protocol"):
                _load_shared_predictions(
                    shared,
                    manifest=manifest,
                    rows=[{"sample_id": "a"}],
                    condition="clean",
                    degradation_seed=20260720,
                    meter_weights=meter_weights,
                    point_weights=point_weights,
                )

            signature["input_degradation"] = {}
            signature.pop("input_degradation_source_sha256")
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            legacy, _ = _load_shared_predictions(
                shared,
                manifest=manifest,
                rows=[{"sample_id": "a"}],
                condition="clean",
                degradation_seed=20260720,
                meter_weights=meter_weights,
                point_weights=point_weights,
            )
            self.assertEqual(set(legacy), {"a"})
            with self.assertRaisesRegex(ValueError, "clean legacy"):
                _load_shared_predictions(
                    shared,
                    manifest=manifest,
                    rows=[{"sample_id": "a"}],
                    condition="blur_severe",
                    degradation_seed=20260720,
                    meter_weights=meter_weights,
                    point_weights=point_weights,
                )


if __name__ == "__main__":
    unittest.main()
