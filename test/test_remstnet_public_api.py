"""Tests for the stable paper-facing ReMSTNet-v3 import surface."""
from __future__ import annotations

import unittest

from remstnet import (
    ARCHITECTURE_ID,
    ReMSTNetV3,
    build_remstnet_v3,
    remstnet_parameter_counts,
)


class ReMSTNetPublicApiTests(unittest.TestCase):
    def test_builder_selects_the_final_architecture(self) -> None:
        model = build_remstnet_v3()

        self.assertIsInstance(model, ReMSTNetV3)
        self.assertEqual(model.architecture, ARCHITECTURE_ID)
        self.assertEqual(
            model.construction["architecture_variant"],
            "adaptive_budget_progress_mixing_v3",
        )
        self.assertTrue(model.use_progress_mixing)
        self.assertTrue(model.learnable_budget_gain)

        counts = remstnet_parameter_counts(model)
        self.assertEqual(counts["component_sum"], counts["total_unique"])
        self.assertGreater(counts["trainable"], 0)
        self.assertLess(counts["trainable"], counts["total_unique"])

    def test_builder_rejects_variant_overrides(self) -> None:
        with self.assertRaisesRegex(TypeError, "fixes these constructor options"):
            build_remstnet_v3(use_progress_mixing=False)


if __name__ == "__main__":
    unittest.main()
