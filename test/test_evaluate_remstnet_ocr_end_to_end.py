from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from experiments.evaluate_remstnet_ocr_end_to_end import (
    EXPECTED_SEEDS,
    ProductionPointGeometryProvider,
    _operating_point_metrics,
)
from experiments.vdn_baseline import image_angle_from_direction


class EndToEndOCRMetricTests(unittest.TestCase):
    def test_reviewed_aggregate_keeps_coverage_and_conditional_accuracy_separate(self) -> None:
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "remstnet_ocr_end_to_end_deployment.json"
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["cohort"]["unique_full_frames"], 153)
        self.assertEqual(payload["cohort"]["labeled_full_frames"], 33)
        self.assertEqual(payload["all_frame_coverage"]["decoder_default_physical_reading"], 27)
        self.assertEqual(payload["labeled_decoder_default"]["accepted"], 9)
        self.assertAlmostEqual(
            payload["labeled_decoder_default"]["conditional_nmae"]["mean"],
            0.03681359326604223,
        )
        self.assertAlmostEqual(
            payload["labeled_decoder_default"]["full_denominator_nmae_with_failure_error_one"]["mean"],
            0.7373127981634662,
        )
        self.assertTrue(
            payload["reproducibility"]["per_image_predictions_and_metrics_match_excluding_runtime"]
        )

    def test_failure_is_kept_on_full_denominator_but_not_conditional(self) -> None:
        progress = {str(seed): 0.5 for seed in EXPECTED_SEEDS}
        rows = [
            {
                "true_start": 0.0,
                "true_end": 10.0,
                "ground_truth": 5.0,
                "normalized_target": 0.5,
                "remst_progress_by_seed": progress,
                "physical_reading_by_seed": {str(seed): 5.0 for seed in EXPECTED_SEEDS},
                "accepted": True,
            },
            {
                "true_start": 0.0,
                "true_end": 10.0,
                "ground_truth": 2.5,
                "normalized_target": 0.25,
                "remst_progress_by_seed": progress,
                "physical_reading_by_seed": {},
                "accepted": False,
            },
        ]
        result = _operating_point_metrics(scored_rows=rows, status_key="accepted")
        aggregate = result["across_seed_mean_sd"]
        self.assertAlmostEqual(aggregate["coverage"]["mean"], 0.5)
        self.assertAlmostEqual(aggregate["nmae_full_denominator"]["mean"], 0.5)
        self.assertAlmostEqual(aggregate["nmae_conditional"]["mean"], 0.0)
        self.assertAlmostEqual(
            aggregate["oracle_range_nmae_on_accepted"]["mean"], 0.0
        )

    def test_reference_angle_inverse_matches_production_convention(self) -> None:
        for angle in (45.0, 90.0, 180.0, 270.0, 315.0):
            point = ProductionPointGeometryProvider._point_from_angle(angle)
            reconstructed = image_angle_from_direction(
                (point[0] - 0.5, point[1] - 0.5)
            )
            circular_error = abs((reconstructed - angle + 180.0) % 360.0 - 180.0)
            self.assertTrue(math.isfinite(reconstructed))
            self.assertLess(circular_error, 1e-9)


if __name__ == "__main__":
    unittest.main()
