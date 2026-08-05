import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_field_development import (
    build_summary,
    load_verified_development_inputs,
    paired_physical_group_bootstrap,
    render_chinese_markdown,
)


class SummarizeFieldDevelopmentTest(unittest.TestCase):
    def _jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _ids_hash(ids: list[str]) -> str:
        payload = json.dumps(
            sorted(ids),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _bundle(
        self,
        root: Path,
        *,
        sealed: bool = False,
        split: str = "field_development",
    ) -> dict[str, Path]:
        manifest = root / (
            "field_confirmatory.jsonl"
            if split == "field_confirmatory"
            else "field_development.jsonl"
        )
        identities = [
            ("s1", "g1", 0.0, ["natural_blur_q1"]),
            ("s2", "g1", 0.5, ["natural_blur_q1", "natural_glare_q4"]),
            ("s3", "g2", 0.2, ["natural_glare_q4"]),
            ("s4", "g2", 0.8, []),
        ]
        manifest_rows = [
            {
                "sample_id": sample_id,
                "group_id": group,
                "dataset": "SyntheticField",
                "split": split,
                "ground_truth": truth,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "metadata": {
                    "field_partition": split,
                    "quality": {"groups": quality},
                },
            }
            for sample_id, group, truth, quality in identities
        ]
        self._jsonl(manifest, manifest_rows)
        ids = [row["sample_id"] for row in manifest_rows]
        protocol_path = manifest.with_name(manifest.name + ".protocol.json")
        protocol = {
            "schema_version": 1,
            "protocol": "group_disjoint_field_development_confirmatory_split_v1",
            "split": split,
            "confirmatory_sealed": sealed,
            "assignment_uses_model_predictions": False,
            "assignment_uses_ground_truth_reading": False,
            "group_disjoint": True,
            "manifest_sha256": self._sha(manifest),
            "sample_ids_sha256": self._ids_hash(ids),
            "rows": 4,
            "groups": 2,
            "group_counts": {"g1": 2, "g2": 2},
        }
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        raw_rows = []
        base_rows = []
        vector_rows = []
        vdn_rows = []
        final_rows = []
        base_predictions = [0.0, 0.55, 0.4, 0.8]
        vector_predictions = [0.01, 0.60, 0.25, 0.70]
        vdn_predictions = [0.02, 0.50, 0.30, 0.90]
        calibrated_predictions = [0.0, 0.54, 0.21, 0.75]
        final_predictions = [0.0, None, 0.21, 0.75]
        routes = [
            "base",
            "failure",
            "reference_conditioned_quality_switch",
            "reference_conditioned_quality_switch",
        ]
        for index, row in enumerate(manifest_rows):
            identity = {
                key: row[key]
                for key in (
                    "sample_id",
                    "group_id",
                    "dataset",
                    "split",
                    "ground_truth",
                    "scale_start",
                    "scale_end",
                )
            }
            raw_rows.append(
                {
                    **identity,
                    "status": index != 1,
                    "error_code": "meter_not_found" if index == 1 else None,
                    "methods": {
                        "transformer": {
                            "status": index != 1,
                            "prediction": (
                                None if index == 1 else row["ground_truth"] + 0.03
                            ),
                        }
                    },
                }
            )
            base_rows.append(
                {
                    **identity,
                    "predictions": {
                        "ours": (
                            None if index == 1 else base_predictions[index]
                        )
                    },
                }
            )
            vector_rows.append(
                {
                    **identity,
                    "status": index != 1,
                    "prediction": (
                        None if index == 1 else vector_predictions[index]
                    ),
                    "error_code": "front_end_failed" if index == 1 else None,
                }
            )
            vdn_rows.append(
                {
                    **identity,
                    "status": True,
                    "prediction": vdn_predictions[index],
                    "error_code": None,
                }
            )
            final_rows.append(
                {
                    **identity,
                    "condition": split,
                    "status": final_predictions[index] is not None,
                    "prediction": final_predictions[index],
                    "route": routes[index],
                    "reference_conditioned_prediction": (
                        None if index == 1 else calibrated_predictions[index]
                    ),
                    "calibration_applied": index in (2, 3),
                }
            )
        files = {
            "raw": root / "raw.jsonl",
            "base": root / "base.jsonl",
            "vector": root / "vector.jsonl",
            "vdn": root / "vdn.jsonl",
            "final": root / "final.jsonl",
        }
        for label, rows in (
            ("raw", raw_rows),
            ("base", base_rows),
            ("vector", vector_rows),
            ("vdn", vdn_rows),
            ("final", final_rows),
        ):
            self._jsonl(files[label], rows)

        freeze_path = root / "freeze.json"
        confirmatory_claim = root / "sealed_field_confirmatory.jsonl"
        freeze = {
            "schema_version": 1,
            "protocol": "field_publication_evaluation_freeze_v1",
            "manifests": {
                "development": {
                    "path": str(manifest.resolve()),
                    "sha256": self._sha(manifest),
                    "protocol_sha256": self._sha(protocol_path),
                    "rows": 4,
                    "groups": 2,
                    "sample_ids_sha256": self._ids_hash(ids),
                    "group_ids": ["g1", "g2"],
                },
                "confirmatory": {
                    "path": str(confirmatory_claim.resolve()),
                    "sha256": "f" * 64,
                },
            },
            "evaluation": {
                "primary_method": "reference_conditioned_router",
                "failure_penalty_nmae": 1.0,
                "test_labels_used_for_selection": 0,
                "primary_metrics": [
                    "NMAE",
                    "Acc@1%",
                    "Acc@2%",
                    "Acc@5%",
                    "coverage",
                ],
                "secondary_metrics": [
                    "median normalized absolute error",
                    "P90 normalized absolute error",
                    "P95 normalized absolute error",
                    "catastrophic error rate >10%",
                    "catastrophic error rate >20%",
                    "macro NMAE by physical meter",
                    "quality-stratum metrics",
                ],
                "bootstrap": {
                    "iterations": 5000,
                    "seed": 123,
                    "unit": "physical meter group_id",
                    "paired": True,
                },
            },
        }
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

        verification_path = root / "verification.json"
        verification_files = {}
        key_mapping = {
            "raw": "raw",
            "base": "base",
            "vector": "probabilistic",
            "vdn": "vdn",
            "final": "final",
        }
        for local_name, verification_name in key_mapping.items():
            path = files[local_name].resolve()
            verification_files[verification_name] = {
                "path": str(path),
                "sha256": self._sha(path),
                "bytes": path.stat().st_size,
            }
        verification = {
            "schema_version": 1,
            "protocol": "field_development_reference_conditioned_verification_v1",
            "status": "verified",
            "scope": split,
            "confirmatory_evaluated": False,
            "row_level_recomputation": True,
            "base_recomputed_from_raw": True,
            "freeze": str(freeze_path.resolve()),
            "freeze_sha256": self._sha(freeze_path),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": self._sha(manifest),
            "sample_ids_sha256": self._ids_hash(ids),
            "samples": 4,
            "groups": 2,
            "files": verification_files,
        }
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        return {
            "manifest": manifest,
            **files,
            "freeze": freeze_path,
            "verification": verification_path,
        }

    def _load(self, paths: dict[str, Path]) -> dict:
        return load_verified_development_inputs(**paths)

    def test_full_denominator_failure_penalty_and_secondary_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = build_summary(self._load(self._bundle(Path(directory))))
        final = summary["methods"]["final"]
        # Errors: 0.0, failure=1.0, 0.01, 0.05.
        self.assertAlmostEqual(final["full_denominator_nmae"], 0.265)
        self.assertEqual(final["failures"], 1)
        self.assertAlmostEqual(final["coverage"], 0.75)
        self.assertEqual(
            final["full_denominator_catastrophic_gt_10pct_count"],
            1,
        )
        self.assertAlmostEqual(final["success_subset_median_nae"], 0.01)
        self.assertAlmostEqual(final["success_subset_p90_nae"], 0.042)
        self.assertAlmostEqual(final["macro_physical_group_nmae"], 0.265)

    def test_natural_quality_strata_are_manifest_driven_and_overlapping(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = build_summary(self._load(self._bundle(Path(directory))))
        strata = summary["natural_quality_strata"]
        self.assertEqual(set(strata), {"natural_blur_q1", "natural_glare_q4"})
        self.assertEqual(strata["natural_blur_q1"]["samples"], 2)
        self.assertEqual(strata["natural_glare_q4"]["samples"], 2)
        self.assertEqual(
            summary["manifest_identity"]["samples_without_natural_quality_group"],
            1,
        )
        # s2 belongs to both strata; memberships are intentionally overlapping.
        self.assertTrue(
            summary["manifest_identity"]["quality_groups_are_overlapping"]
        )

    def test_multi_group_paired_bootstrap_is_deterministic(self):
        first = paired_physical_group_bootstrap(
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.2, 0.3, 0.5],
            ["g1", "g1", "g2", "g2"],
            iterations=5000,
            seed=7,
        )
        second = paired_physical_group_bootstrap(
            [0.0, 0.1, 0.2, 0.3],
            [0.2, 0.2, 0.3, 0.5],
            ["g1", "g1", "g2", "g2"],
            iterations=5000,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["physical_groups"], 2)
        self.assertEqual(first["iterations"], 5000)
        self.assertLess(
            first["delta_full_denominator_nmae_final_minus_comparator"],
            0.0,
        )
        self.assertEqual(len(first["paired_physical_group_bootstrap_95ci"]), 2)

    def test_calibration_and_routing_migrations_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = build_summary(self._load(self._bundle(Path(directory))))
        migration = summary["migration_analysis"]
        self.assertEqual(
            migration["calibration_applied_only"]["calibration_applied_rows"],
            2,
        )
        self.assertEqual(
            migration["calibration_applied_only"]["positive_transfers"],
            2,
        )
        self.assertEqual(
            migration["routing_quality_switch_only"]["quality_switch_rows"],
            2,
        )
        self.assertEqual(
            migration["routing_quality_switch_only"]["positive_transfers"],
            1,
        )
        self.assertEqual(
            migration["routing_quality_switch_only"]["negative_transfers"],
            1,
        )

    def test_confirmatory_or_sealed_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._bundle(
                Path(directory),
                sealed=True,
                split="field_confirmatory",
            )
            with self.assertRaisesRegex(PermissionError, "REFUSED"):
                self._load(paths)

    def test_prediction_hash_must_be_bound_by_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._bundle(Path(directory))
            with paths["vector"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "not bound by verification"):
                self._load(paths)

    def test_summary_explicitly_forbids_automatic_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._load(self._bundle(Path(directory)))
            summary = build_summary(loaded)
            repeated = build_summary(loaded)
        self.assertFalse(summary["scope"]["automatic_threshold_tuning"])
        self.assertFalse(summary["scope"]["automatic_model_or_method_selection"])
        self.assertFalse(summary["scope"]["confirmatory_opened_or_read"])
        self.assertEqual(
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
            json.dumps(repeated, ensure_ascii=False, sort_keys=True),
        )
        markdown = render_chinese_markdown(summary)
        self.assertIn("现场开发集独立统计汇总", markdown)
        self.assertIn("confirmatory 未打开、未读取、未评估", markdown)


if __name__ == "__main__":
    unittest.main()
