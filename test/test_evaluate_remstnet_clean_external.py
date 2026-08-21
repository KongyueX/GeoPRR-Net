from __future__ import annotations

import unittest

from experiments.evaluate_remstnet_clean_external import (
    _bootstrap_mean_ci,
    _metrics,
)


class ReMSTNetCleanExternalTests(unittest.TestCase):
    def test_full_denominator_metrics_keep_failures(self) -> None:
        metrics = _metrics([0.01, 1.0, 0.03], [True, False, True])

        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["passed"], 2)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["nmae"], (0.01 + 1.0 + 0.03) / 3)
        self.assertAlmostEqual(metrics["acc_at_2_percent"], 1 / 3)
        self.assertAlmostEqual(metrics["acc_at_5_percent"], 2 / 3)

    def test_stratified_bootstrap_is_reproducible(self) -> None:
        first = _bootstrap_mean_ci(
            [0.01, 0.02, 0.20, 0.30],
            strata=["a", "a", "b", "b"],
            replicates=100,
            seed=17,
        )
        second = _bootstrap_mean_ci(
            [0.01, 0.02, 0.20, 0.30],
            strata=["a", "a", "b", "b"],
            replicates=100,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["strata"], ["a", "b"])
        self.assertAlmostEqual(first["estimate"], 0.1325)


if __name__ == "__main__":
    unittest.main()
