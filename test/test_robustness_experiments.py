"""Deterministic checks for the blur/perspective paper protocol."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.collect_predictions import _legacy_clean_resume_compatible
from experiments.make_robustness_report import build_report_data, plot_robustness
from experiments.robustness_degradations import (
    DEGRADATION_SPECS,
    apply_degradation,
    degradation_names,
)
from experiments.selective_experiment import _front_end_signature


class RobustnessProtocolTest(unittest.TestCase):
    def test_conditions_are_frozen_and_ordered(self):
        self.assertEqual(
            degradation_names(),
            (
                "clean",
                "blur_moderate",
                "blur_severe",
                "perspective_moderate",
                "perspective_severe",
                "combined_severe",
            ),
        )
        self.assertLess(
            DEGRADATION_SPECS["blur_moderate"].blur_sigma_fraction,
            DEGRADATION_SPECS["blur_severe"].blur_sigma_fraction,
        )
        self.assertLess(
            DEGRADATION_SPECS["perspective_moderate"].perspective_degrees,
            DEGRADATION_SPECS["perspective_severe"].perspective_degrees,
        )

    def test_blur_is_monotonic_on_a_high_frequency_image(self):
        yy, xx = np.indices((512, 768))
        checker = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
        image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
        variances = []
        for condition in ("clean", "blur_moderate", "blur_severe"):
            output, metadata = apply_degradation(
                image,
                condition,
                sample_id="checker",
                seed=17,
            )
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(metadata["condition"], condition)
            variances.append(float(cv2.Laplacian(output, cv2.CV_64F).var()))
        self.assertGreater(variances[0], variances[1])
        self.assertGreater(variances[1], variances[2])

    def test_perspective_is_deterministic_and_preserves_shape(self):
        image = np.zeros((240, 360, 3), dtype=np.uint8)
        image[40:200, 80:280] = (40, 180, 230)
        first, first_meta = apply_degradation(
            image,
            "perspective_severe",
            sample_id="sample-42",
            seed=20260720,
        )
        second, second_meta = apply_degradation(
            image,
            "perspective_severe",
            sample_id="sample-42",
            seed=20260720,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_meta, second_meta)
        self.assertEqual(first.shape, image.shape)
        self.assertFalse(np.array_equal(first, image))
        homography = np.asarray(first_meta["perspective"]["homography"])
        self.assertEqual(homography.shape, (3, 3))
        self.assertGreater(abs(float(np.linalg.det(homography))), 1e-8)

    def test_legacy_resume_only_accepts_the_clean_compatibility_case(self):
        previous = {
            "manifest_sha256": "manifest",
            "weights_sha256": {"model": "same"},
            "source_sha256": {"collector": "old", "model": "same"},
        }
        current = {
            "manifest_sha256": "manifest",
            "weights_sha256": {"model": "same"},
            "source_sha256": {"collector": "new", "model": "same"},
            "input_degradation": {"condition": "clean", "seed": 1},
            "input_degradation_source_sha256": "transform",
        }
        self.assertTrue(_legacy_clean_resume_compatible(previous, current))
        current["input_degradation"]["condition"] = "blur_moderate"
        self.assertFalse(_legacy_clean_resume_compatible(previous, current))
        current["input_degradation"]["condition"] = "clean"
        current["weights_sha256"]["model"] = "changed"
        self.assertFalse(_legacy_clean_resume_compatible(previous, current))

    def test_front_end_signature_ignores_test_degradation_not_model_changes(self):
        clean = {
            "manifest_sha256": "train",
            "device": "cuda",
            "weights_sha256": {"model": "same"},
            "source_sha256": {"collector": "old", "model": "same"},
        }
        degraded = {
            "manifest_sha256": "test",
            "device": "cpu",
            "weights_sha256": {"model": "same"},
            "source_sha256": {"collector": "new", "model": "same"},
            "input_degradation": {"condition": "perspective_severe", "seed": 7},
            "input_degradation_source_sha256": "transform",
        }
        self.assertEqual(_front_end_signature(clean), _front_end_signature(degraded))
        degraded["source_sha256"]["model"] = "changed"
        self.assertNotEqual(_front_end_signature(clean), _front_end_signature(degraded))

    def test_report_requires_all_conditions_and_writes_plot(self):
        def payload(condition: str, index: int) -> dict:
            metrics = {}
            for method_number, method in enumerate(
                (
                    "Original Transformer",
                    "Quality-weighted Fusion",
                    "Residual without Gate",
                    "Ours",
                )
            ):
                metrics[method] = {
                    "nmae": 0.1 + index * 0.01 + method_number * 0.001,
                    "acc_2pct": 0.5 - index * 0.02,
                    "coverage": 0.99 - index * 0.01,
                }
            return {
                "protocol": "frozen_model_evaluation",
                "samples_total": 20,
                "front_end_signature_verified": True,
                "calibrator": "same.joblib",
                "prediction_cache_signature": {
                    "input_degradation": {
                        "condition": condition,
                        "protocol": "controlled_blur_perspective_v1",
                        "seed": 7,
                    },
                    "input_degradation_source_sha256": "same-transform",
                },
                "metrics": metrics,
                "paired_comparisons": {
                    "Ours vs Original Transformer": {
                        "delta_nmae": -0.02,
                        "delta_nmae_group_bootstrap_95ci": [-0.03, -0.01],
                    }
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conditions = []
            for index, condition in enumerate(degradation_names()):
                condition_root = root / condition
                condition_root.mkdir()
                path = condition_root / "metrics.json"
                path.write_text(json.dumps(payload(condition, index)), encoding="utf-8")
                prediction_rows = []
                for sample_index in range(20):
                    prediction_rows.append(
                        {
                            "sample_id": f"sample-{sample_index}",
                            "group_id": f"group-{sample_index % 2}",
                            "ground_truth": 0.0,
                            "scale_start": 0.0,
                            "scale_end": 100.0,
                            "predictions": {
                                "transformer": 10.0 + index,
                                "weighted_fusion": 8.0 + index,
                                "residual_ungated": 7.0 + 1.5 * index,
                                "ours": 5.0 + 2.0 * index,
                            },
                        }
                    )
                (condition_root / "predictions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in prediction_rows),
                    encoding="utf-8",
                )
                conditions.append((condition, path))
            data = build_report_data(
                conditions,
                expected_samples=20,
                bootstrap_iterations=20,
            )
            self.assertEqual(len(data["controlled"]), 6)
            self.assertAlmostEqual(
                data["controlled"][1]["metrics"]["Ours"][
                    "nmae_increase_from_clean"
                ],
                0.01,
            )
            paired = data["controlled"][1]["paired_degradation_from_clean"]
            self.assertAlmostEqual(
                paired["method_nmae_change_from_clean"]["Original Transformer"],
                0.01,
            )
            self.assertAlmostEqual(
                paired["method_nmae_change_from_clean"]["Ours"],
                0.02,
            )
            self.assertAlmostEqual(
                paired["ours_vs_transformer_degradation_difference_nmae"],
                0.01,
            )
            output = root / "curves.png"
            plot_robustness(data, output)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
