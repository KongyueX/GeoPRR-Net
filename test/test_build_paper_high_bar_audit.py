from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.build_paper_high_bar_audit import (
    PROTOCOL,
    _assert_safe_artifact_path,
    _comparator_audit,
    _domain_interaction,
    _extract_uncertainty,
    _summarize_route_changes,
    _write_idempotent,
    build_verification,
)


class PaperHighBarAuditTests(unittest.TestCase):
    def test_route_change_statistics_use_pepd_as_reference(self) -> None:
        rows = [
            {
                "group_id": "g1",
                "route": "base",
                "reference_prediction": 0.1,
                "candidate_prediction": 0.2,
                "reference_error": 0.10,
                "candidate_error": 0.05,
            },
            {
                "group_id": "g1",
                "route": "calibrated_quality_switch",
                "reference_prediction": 0.2,
                "candidate_prediction": 0.3,
                "reference_error": 0.10,
                "candidate_error": 0.20,
            },
            {
                "group_id": "g2",
                "route": "calibrated_quality_switch",
                "reference_prediction": 0.4,
                "candidate_prediction": 0.4,
                "reference_error": 0.10,
                "candidate_error": 0.10,
            },
            {
                "group_id": "g2",
                "route": "failure",
                "reference_prediction": None,
                "candidate_prediction": None,
                "reference_error": 1.0,
                "candidate_error": 1.0,
            },
        ]
        result = _summarize_route_changes(rows)
        self.assertEqual(result["changed_predictions"], 2)
        self.assertEqual(result["unchanged_predictions"], 2)
        self.assertEqual(result["helpful_changes"], 1)
        self.assertEqual(result["harmful_changes"], 1)
        self.assertEqual(result["groups_harmed"], 1)
        self.assertEqual(result["groups_helped"], 0)
        self.assertEqual(result["groups_tied"], 1)
        self.assertAlmostEqual(result["mean_nmae_delta_all_rows"], 0.0125)

    def test_field_comparator_audit_is_grouped_and_deterministic(self) -> None:
        records = [
            {
                "group_id": "g1",
                "pepd_error": 0.10,
                "vdn_error": 0.20,
                "base_mask_error": 0.30,
                "original_transformer_error": 0.40,
                "fadr_error": 0.15,
            },
            {
                "group_id": "g1",
                "pepd_error": 0.20,
                "vdn_error": 0.30,
                "base_mask_error": 0.40,
                "original_transformer_error": 0.50,
                "fadr_error": 0.25,
            },
            {
                "group_id": "g2",
                "pepd_error": 0.10,
                "vdn_error": 0.05,
                "base_mask_error": 0.20,
                "original_transformer_error": 0.30,
                "fadr_error": 0.10,
            },
        ]
        first = _comparator_audit(records, iterations=200, seed=17)
        second = _comparator_audit(records, iterations=200, seed=17)
        self.assertEqual(first, second)
        vdn = first["comparisons"]["vdn"]
        self.assertAlmostEqual(vdn["micro_nmae_effect"], 0.05)
        self.assertAlmostEqual(vdn["macro_group_nmae_effect"], 0.025)
        self.assertEqual(vdn["groups_favoring_pepd"], 1)
        self.assertEqual(vdn["groups_favoring_comparator"], 1)
        self.assertAlmostEqual(
            vdn["leave_one_group_out_micro_effect"]["minimum"], -0.05
        )
        self.assertAlmostEqual(
            vdn["leave_one_group_out_micro_effect"]["maximum"], 0.10
        )
        self.assertEqual(
            first["multiplicity"]["interval_method"],
            "Bonferroni-adjusted percentile cluster bootstrap",
        )

    def test_domain_interaction_resamples_domains_independently(self) -> None:
        source = [
            {"group_id": "s1", "pepd_error": 0.20, "fadr_error": 0.10},
            {"group_id": "s1", "pepd_error": 0.30, "fadr_error": 0.20},
            {"group_id": "s2", "pepd_error": 0.40, "fadr_error": 0.30},
        ]
        field = [
            {"group_id": "f1", "pepd_error": 0.10, "fadr_error": 0.30},
            {"group_id": "f2", "pepd_error": 0.20, "fadr_error": 0.40},
        ]
        result = _domain_interaction(source, field, iterations=100, seed=31)
        self.assertAlmostEqual(result["source"]["micro_fadr_minus_pepd"], -0.10)
        self.assertAlmostEqual(
            result["field_sensitivity"]["micro_fadr_minus_pepd"], 0.20
        )
        interaction = result["domain_by_route_interaction"]
        self.assertAlmostEqual(interaction["micro_effect"], 0.30)
        np.testing.assert_allclose(
            interaction["micro_independent_group_bootstrap_95ci"], [0.30, 0.30]
        )

    def test_uncertainty_extraction_preserves_claim_boundary_and_tail_misses(self) -> None:
        fixture = {
            "status": "complete",
            "protocol": "syncg_train_group_cross_conformal_angle_uncertainty_audit_v1",
            "scope": {
                "train_only": True,
                "test_sets_used": [],
                "field_sets_used": [],
            },
            "provenance": {
                "valid_uncertainty_rows": 10,
                "valid_uncertainty_groups": 3,
                "excluded_uncertainty_rows": 1,
            },
            "raw_sigma_diagnostics": {
                "mean_absolute_circular_angle_error_degrees": 0.8,
                "median_absolute_circular_angle_error_degrees": 0.4,
                "mean_angle_std_degrees": 1.7,
                "median_angle_std_degrees": 1.1,
                "pearson_error_sigma_correlation": 0.56,
                "named_gaussian_intervals": {
                    "one_sigma": {
                        "target_coverage": 0.6827,
                        "observed_coverage": 0.94,
                        "macro_physical_group_coverage": 0.93,
                        "coverage_gap": 0.2573,
                    },
                    "two_sigma": {
                        "target_coverage": 0.9545,
                        "observed_coverage": 0.99,
                        "macro_physical_group_coverage": 0.98,
                        "coverage_gap": 0.0355,
                    },
                },
            },
            "cross_group_conformal": {
                "method": "fixture_group_conformal",
                "folds": 5,
                "levels": {
                    "0.95": {
                        "target_coverage": 0.95,
                        "observed_coverage": 0.948,
                        "coverage_gap": -0.002,
                        "macro_physical_group_coverage": 0.947,
                        "macro_physical_group_coverage_gap": -0.003,
                        "mean_interval_width_degrees": 3.5,
                        "median_interval_width_degrees": 2.4,
                        "catastrophic_error_samples": 8,
                        "catastrophic_misses": 4,
                        "maximum_missed_error_degrees": 178.0,
                        "sample_weighted_multiplier_mean": 1.06,
                    }
                },
            },
        }
        result = _extract_uncertainty(fixture)
        self.assertEqual(result["samples"], 10)
        self.assertAlmostEqual(
            result["raw_sigma"]["pearson_error_sigma_correlation"], 0.56
        )
        level = result["cross_group_conformal"]["levels"][0]
        self.assertEqual(level["catastrophic_error_samples"], 8)
        self.assertEqual(level["catastrophic_misses"], 4)
        self.assertIn("no field-coverage claim", result["cross_group_conformal"]["claim_boundary"])

    def test_input_gate_rejects_images_public_test_and_sealed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "field_holdout" / "predictions.jsonl"
            safe.parent.mkdir(parents=True)
            safe.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                _assert_safe_artifact_path(safe, label="safe"), safe.resolve()
            )
            for relative in (
                "images/predictions.jsonl",
                "public/predictions.jsonl",
                "syncg_test/predictions.jsonl",
                "sealed/predictions.jsonl",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "forbidden path component"):
                    _assert_safe_artifact_path(path, label="restricted")

    def test_no_clobber_writer_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            payload = {"protocol": PROTOCOL, "status": "complete"}
            self.assertTrue(_write_idempotent(path, payload))
            self.assertFalse(_write_idempotent(path, payload))
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _write_idempotent(path, {"protocol": "changed"})
            verification = build_verification(payload)
            self.assertEqual(verification["summary_protocol"], PROTOCOL)


if __name__ == "__main__":
    unittest.main()
