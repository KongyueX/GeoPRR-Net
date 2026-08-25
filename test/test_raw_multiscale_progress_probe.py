from __future__ import annotations

import unittest

import torch

from experiments.raw_multiscale_progress_probe import (
    METHOD_BASE,
    METHOD_CONTEXT,
    METHOD_MULTISCALE,
    PredictionRecord,
    build_probe_from_foundation,
    refiner_parameter_counts,
    summarize_records,
)
from experiments.syncg_lightweight_regression_baselines import (
    LightweightProgressRegressor,
)


class RawMultiscaleProgressProbeTests(unittest.TestCase):
    def test_matched_heads_start_exactly_at_frozen_base(self) -> None:
        torch.manual_seed(7)
        foundation = LightweightProgressRegressor(
            "efficientnet_b0", imagenet_pretrained=False
        ).eval()
        images = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            expected = foundation(images)
        probe = build_probe_from_foundation(foundation).eval()
        with torch.inference_mode():
            outputs = probe(images)
        torch.testing.assert_close(outputs[METHOD_BASE], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            outputs[METHOD_CONTEXT], outputs[METHOD_BASE], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            outputs[METHOD_MULTISCALE], outputs[METHOD_BASE], rtol=0.0, atol=0.0
        )

    def test_only_parameter_matched_refiners_are_trainable(self) -> None:
        foundation = LightweightProgressRegressor(
            "efficientnet_b0", imagenet_pretrained=False
        )
        probe = build_probe_from_foundation(foundation)
        counts = refiner_parameter_counts(probe)
        difference = abs(counts[METHOD_CONTEXT] - counts[METHOD_MULTISCALE])
        maximum = max(counts[METHOD_CONTEXT], counts[METHOD_MULTISCALE])
        self.assertLessEqual(difference / maximum, 0.01)
        self.assertEqual(
            counts["total_trainable"],
            counts[METHOD_CONTEXT] + counts[METHOD_MULTISCALE],
        )
        self.assertTrue(all(not p.requires_grad for p in probe.encoder.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in probe.base_projection.parameters())
        )

    def test_scene_bootstrap_is_reproducible_and_paired(self) -> None:
        records = tuple(
            PredictionRecord(
                sample_id=f"sample_{index}",
                scene_stem=f"scene_{index // 2}",
                condition="clean",
                target=0.5,
                predictions={
                    METHOD_BASE: 0.60,
                    METHOD_CONTEXT: 0.58,
                    METHOD_MULTISCALE: 0.55,
                },
            )
            for index in range(8)
        )
        first = summarize_records(
            records, bootstrap_replicates=25, bootstrap_seed=11
        )
        second = summarize_records(
            records, bootstrap_replicates=25, bootstrap_seed=11
        )
        self.assertEqual(first, second)
        comparison = first["comparisons"]["raw_multiscale_minus_context_mlp"]
        self.assertAlmostEqual(comparison["mean_nmae_delta"], -0.03)
        self.assertEqual(
            comparison["scene_bootstrap_probability_delta_below_zero"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
