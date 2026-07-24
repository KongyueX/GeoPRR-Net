import unittest

from experiments.summarize_vdn_replicates import (
    _mean_std,
    paired_final_vs_vdn_replicates,
)


def _row(sample_id, prediction, *, group_id="gauge-a"):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "ground_truth": 50.0,
        "scale_start": 0.0,
        "scale_end": 100.0,
        "prediction": prediction,
    }


class VDNReplicateSummaryTests(unittest.TestCase):
    def test_mean_std_uses_sample_standard_deviation(self):
        result = _mean_std([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["std"], 1.0)

    def test_paired_comparison_averages_errors_not_predictions(self):
        final = [
            _row("a", 50.0, group_id="g1"),
            _row("b", 50.0, group_id="g2"),
        ]
        first = [
            _row("a", 40.0, group_id="g1"),
            _row("b", None, group_id="g2"),
        ]
        second = [
            _row("a", 60.0, group_id="g1"),
            _row("b", 70.0, group_id="g2"),
        ]
        result = paired_final_vs_vdn_replicates(
            [first, second],
            final,
            bootstrap_iterations=100,
            seed=7,
        )
        # Mean VDN errors: sample a=0.10 and sample b=(1.0+0.2)/2=0.60.
        self.assertAlmostEqual(
            result["vdn_nmae_mean_error_recomputed"],
            0.35,
        )
        self.assertAlmostEqual(
            result["delta_nmae_final_minus_vdn"],
            -0.35,
        )
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["groups"], 2)
        self.assertIsNotNone(
            result["delta_nmae_group_bootstrap_95ci"]
        )

    def test_paired_comparison_rejects_different_sample_sets(self):
        with self.assertRaisesRegex(ValueError, "different sample sets"):
            paired_final_vs_vdn_replicates(
                [[_row("a", 50.0)]],
                [_row("b", 50.0)],
                bootstrap_iterations=10,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
