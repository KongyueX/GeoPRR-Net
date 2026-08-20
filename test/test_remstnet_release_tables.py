"""Structural and internal-consistency checks for released paper tables."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "remstnet_v3_tables.json"


class ReMSTNetReleaseTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(TABLES.read_text(encoding="utf-8"))

    def test_public_result_schema_is_complete(self) -> None:
        self.assertEqual(self.report["schema_version"], 1)
        self.assertEqual(
            self.report["model"]["architecture_id"],
            "ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3",
        )
        self.assertEqual(len(self.report["dataset_inventory"]), 8)
        self.assertEqual(len(self.report["syncg_scene_holdout"]["rows"]), 4)
        self.assertEqual(
            len(self.report["real_source_controlled_transformations"]["rows"]),
            4,
        )

    def test_counts_and_values_are_finite_and_nonnegative(self) -> None:
        model = self.report["model"]
        self.assertEqual(
            model["total_unique_parameters"],
            model["trainable_parameters_in_remst_fit"]
            + model["frozen_foundation_parameters"],
        )
        for row in self.report["dataset_inventory"]:
            self.assertGreater(row["images"], 0)
            self.assertGreater(row["groups"], 0)
        for row in self.report["syncg_scene_holdout"]["rows"]:
            for condition in self.report["syncg_scene_holdout"]["conditions"]:
                value = row[condition]
                self.assertTrue(math.isfinite(value["mean"]))
                self.assertTrue(math.isfinite(value["sample_sd"]))
                self.assertGreaterEqual(value["mean"], 0.0)
                self.assertGreaterEqual(value["sample_sd"], 0.0)

    def test_reported_relative_reductions_are_derived_from_table_rows(self) -> None:
        rows = {
            row["method"]: row
            for row in self.report["syncg_scene_holdout"]["rows"]
        }
        baseline = rows["SARN-v2 + EfficientNet-B0"]
        final = rows["ReMSTNet-v3"]
        reductions = self.report["syncg_scene_holdout"][
            "relative_reduction_vs_sarn_efficientnet_b0"
        ]
        for condition, key in (
            ("all_conditions", "all_conditions_percent"),
            ("projective_pooled", "projective_pooled_percent"),
        ):
            derived = 100.0 * (
                1.0 - final[condition]["mean"] / baseline[condition]["mean"]
            )
            self.assertAlmostEqual(derived, reductions[key], places=2)

    def test_declared_fallback_and_repeat_preservation_are_exact(self) -> None:
        fallback = self.report["perspective_stress"]["rows"][-1]
        self.assertTrue(fallback["fallback_to_raw"])
        self.assertEqual(fallback["remstnet_nmae"], fallback["raw_nmae"])
        repeat = self.report["natural_repeat"]
        self.assertTrue(repeat["methods_identical"])
        for comparison in repeat["paired_differences"].values():
            self.assertEqual(comparison["estimate"], 0.0)
            self.assertEqual(comparison["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
