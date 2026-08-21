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
        self.assertEqual(self.report["schema_version"], 2)
        self.assertEqual(
            self.report["model"]["architecture_id"],
            "ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3",
        )
        self.assertEqual(len(self.report["dataset_inventory"]), 11)
        self.assertEqual(len(self.report["syncg_scene_holdout"]["rows"]), 4)
        self.assertEqual(
            len(self.report["vdn_double_holdout_intersection"]["aggregate_rows"]),
            3,
        )
        self.assertEqual(
            len(self.report["vdn_double_holdout_intersection"]["condition_rows"]),
            6,
        )
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

    def test_vdn_intersection_keeps_protocols_and_denominators_separate(self) -> None:
        comparison = self.report["vdn_double_holdout_intersection"]
        fixed = comparison["fixed_intersection"]
        self.assertEqual(
            fixed["rows"],
            fixed["samples"] * len(comparison["conditions"]),
        )
        self.assertEqual(fixed["scene_groups"], 6)
        self.assertTrue(comparison["same_condition_pixels_verified"])
        self.assertFalse(comparison["reported_selection_depends_on_vdn_outcome"])

        rows = {
            (row["method"], row["reference_mode"]): row
            for row in comparison["aggregate_rows"]
        }
        annotation_reference = rows[
            (
                "VDN architecture, SyncG-retrained terminal epoch 200",
                "annotation-derived pivot and ordered scale endpoints for offline direction-to-progress conversion",
            )
        ]
        denominators = comparison["comparison_denominators"]
        self.assertEqual(denominators["all_conditions"]["compared_rows"], 774)
        self.assertEqual(denominators["projective_pooled"]["compared_rows"], 387)
        self.assertEqual(denominators["all_conditions"]["coverage"], 1.0)
        for row in comparison["aggregate_rows"]:
            self.assertEqual(row["all_conditions"]["rows"], 774)
            self.assertEqual(row["projective_pooled"]["rows"], 387)
        self.assertTrue(annotation_reference["annotation_assisted"])
        self.assertFalse(annotation_reference["deployable_from_canonical_roi"])
        self.assertIn("nmae", annotation_reference["all_conditions"])

    def test_external_and_deployment_source_results_keep_task_boundaries(self) -> None:
        public = self.report["public_external_datasets"]
        rpm = public["rpm10k_scalar_reading"]
        self.assertEqual(rpm["images"], 1797)
        self.assertEqual(rpm["detector_passes"] + rpm["detector_failures"], 1797)
        self.assertAlmostEqual(
            rpm["detector_passes"] / rpm["images"],
            rpm["detector_coverage"],
        )
        self.assertFalse(rpm["relation_active"])
        self.assertNotIn("pointer10k_direction", public)
        self.assertEqual(rpm["environment_tag_counts"]["blur"], 474)
        self.assertEqual(rpm["environment_tag_counts"]["tilted"], 993)

        field = self.report["native_field_full_frame"]
        self.assertEqual(
            field["source_files_before_deduplication"],
            field["unique_full_frames"] + field["duplicate_files_removed"],
        )
        self.assertEqual(
            field["labeled_full_frames"] + field["unlabeled_full_frames"],
            field["unique_full_frames"],
        )
        self.assertEqual(
            field["detector_passes"] + field["detector_failures"],
            field["unique_full_frames"],
        )
        self.assertFalse(field["ocr_included"])
        self.assertIn("Full-Frame Diagnostic", field["dataset"])

        industrial = self.report["industrial_real_photo_baseline"]
        self.assertEqual(industrial["samples"], 1395)
        self.assertEqual(industrial["groups"], 52)
        identity = industrial["decoded_pixel_identity_audit"]
        self.assertEqual(identity["unified_not_in_existing_sources"], 0)
        self.assertEqual(identity["existing_sources_not_in_unified"], 0)
        industrial_rows = {row["scope"]: row for row in industrial["rows"]}
        for scope in (
            "perspective_moderate",
            "perspective_severe",
            "combined_severe",
            "all_conditions",
            "projective_pooled",
        ):
            row = industrial_rows[scope]
            self.assertLess(row["paired_delta"], 0.0)
            self.assertLess(row["ci95"][1], 0.0)

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

        vdn = self.report["vdn_double_holdout_intersection"]
        vdn_rows = {row["method"]: row for row in vdn["aggregate_rows"]}
        vdn_reference = vdn_rows[
            "VDN architecture, SyncG-retrained terminal epoch 200"
        ]
        remst = vdn_rows["ReMSTNet-v3"]
        vdn_reductions = vdn[
            "relative_reduction_vs_vdn_annotation_reference_percent"
        ]
        for condition, key in (
            ("all_conditions", "all_conditions"),
            ("projective_pooled", "projective_pooled"),
        ):
            derived = 100.0 * (
                1.0
                - remst[condition]["mean"]
                / vdn_reference[condition]["nmae"]
            )
            self.assertAlmostEqual(
                derived,
                vdn_reductions[key],
                places=2,
            )

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
