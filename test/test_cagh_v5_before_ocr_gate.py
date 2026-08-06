from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import evaluate_cagh_v5_before_ocr_gate as gate


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V5BeforeOCRGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.v5_root = self.root / "v5"
        self.protocol = json.loads(gate.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

        cohort_hash = "a" * 64
        baseline = {
            "schema_version": 2,
            "protocol": gate.BASELINE_PROTOCOL_NAME,
            "status": "complete",
            "selection": {
                "held_out_seed": 20260720,
                "samples": 1625,
                "physical_groups": 73,
                "sample_ids_sha256": cohort_hash,
            },
            "scoring": {"failure_penalty": 1.0},
            "metrics": {
                "scalemark_reference_head_v4_three_seed_mean": {
                    "nmae": 0.018,
                    "coverage": 0.997,
                }
            },
        }
        baseline_path = self.root / "baseline.json"
        write_json(baseline_path, baseline)
        self.protocol["frozen_inputs"]["strict_common_holdout_v2"] = {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        }
        self.protocol["v5_artifacts"]["output_root"] = str(self.v5_root)
        self.protocol["folds"][0]["validation_sample_ids_sha256"] = cohort_hash

        for fold_spec in self.protocol["folds"]:
            seed = fold_spec["pepd_seed"]
            self._write_fold(
                seed,
                validation_hash=fold_spec["validation_sample_ids_sha256"],
                samples=fold_spec["strict_assigned_samples"],
                groups=fold_spec["strict_assigned_groups"],
            )
        self._write_aggregate()
        self.protocol_path = self.root / "gate_protocol.json"
        write_json(self.protocol_path, self.protocol)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_fold(
        self,
        seed: int,
        *,
        validation_hash: str,
        samples: int,
        groups: int,
        nmae: float = 0.012,
        coverage: float = 0.999,
    ) -> Path:
        fold_spec = next(row for row in self.protocol["folds"] if row["pepd_seed"] == seed)
        path = self.v5_root / "folds" / f"pepd_seed_{seed}" / "summary.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "protocol": gate.V5_PROTOCOL_NAME,
                "status": "complete",
                "jointly_unseen_contract": True,
                "pepd_seed": seed,
                "head_seed": fold_spec["head_seed"],
                "split_identity": {
                    "validation_sample_ids_sha256": validation_hash,
                    "group_overlap": 0,
                },
                "strict_assigned_metrics": {
                    "samples": samples,
                    "groups": groups,
                    "failed_rows_penalty": 1.0,
                    "full_denominator_nmae": nmae,
                    "coverage": coverage,
                    "group_macro_full_denominator_nmae": 0.013,
                    "p95_absolute_progress_error": 0.03,
                },
            },
        )
        return path

    def _write_aggregate(self) -> None:
        expected = self.protocol["v5_artifacts"]["expected_union"]
        audit = {
            **expected,
            "all_rows_jointly_unseen_by_pepd_and_head": True,
            "field_samples_read": 0,
            "public_test_samples_read": 0,
        }
        folds = []
        for spec in self.protocol["folds"]:
            path = self.v5_root / "folds" / f"pepd_seed_{spec['pepd_seed']}" / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            folds.append(
                {
                    "pepd_seed": spec["pepd_seed"],
                    "head_seed": spec["head_seed"],
                    "summary": str(path.resolve()),
                    "summary_sha256": sha256_file(path),
                    "strict_assigned_metrics": summary["strict_assigned_metrics"],
                }
            )
        strict = {
            "schema_version": 1,
            "protocol": gate.V5_PROTOCOL_NAME,
            "status": "complete",
            "metrics": {
                "samples": expected["eligible_union_samples"],
                "groups": expected["eligible_union_groups"],
                "failed_rows_penalty": 1.0,
            },
            "overlap_and_assignment_audit": audit,
            "folds": folds,
            "field_samples_read": 0,
            "public_test_samples_read": 0,
        }
        self._write_root_from_strict(strict)

    def _write_root_from_strict(self, strict: dict[str, object]) -> None:
        strict_path = self.v5_root / "strict_oof_summary.json"
        write_json(strict_path, strict)
        final = {
            "schema_version": 1,
            "protocol": gate.V5_PROTOCOL_NAME,
            "status": "complete",
            "scope": {
                "field_samples_read": 0,
                "public_test_samples_read": 0,
                "confirmatory_samples_read": 0,
                "sealed_samples_read": 0,
            },
            "input_sha256": {
                "protocol": self.protocol["frozen_inputs"]["v5_oof_protocol"]["sha256"],
                "runner_source": self.protocol["frozen_inputs"]["v5_oof_runner"]["sha256"],
            },
            "output_root": str(self.v5_root.resolve()),
            "overlap_and_assignment_audit": strict["overlap_and_assignment_audit"],
            "strict_oof": strict,
            "artifacts": {
                "strict_oof_summary": str(strict_path.resolve()),
                "strict_oof_summary_sha256": sha256_file(strict_path),
            },
        }
        write_json(self.v5_root / "summary.json", final)

    def test_passes_all_frozen_thresholds_and_binds_fold_hashes(self) -> None:
        result = gate.evaluate(protocol_path=self.protocol_path)

        self.assertEqual(result["decision"], "pass")
        self.assertAlmostEqual(result["same_cohort_comparison"]["relative_nmae_improvement"], 1.0 / 3.0)
        self.assertAlmostEqual(result["same_cohort_comparison"]["coverage_difference_signed"], 0.002)
        self.assertNotIn("coverage_difference_absolute", result["same_cohort_comparison"])
        self.assertTrue(all(check["passed"] for check in result["checks"]))
        self.assertEqual(result["data_access_audit"]["images_read"], 0)
        self.assertEqual(result["runtime_inputs"]["aggregate_chain"]["union_samples"], 4380)
        self.assertEqual(result["runtime_inputs"]["aggregate_chain"]["union_groups"], 197)
        for fold in result["runtime_inputs"]["fold_summaries"]:
            self.assertEqual(fold["summary_sha256"], sha256_file(Path(fold["summary"])))

    def test_completed_gate_records_fail_without_raising_for_metric_miss(self) -> None:
        spec = self.protocol["folds"][2]
        self._write_fold(
            spec["pepd_seed"],
            validation_hash=spec["validation_sample_ids_sha256"],
            samples=spec["strict_assigned_samples"],
            groups=spec["strict_assigned_groups"],
            nmae=0.0201,
        )
        self._write_aggregate()

        result = gate.evaluate(protocol_path=self.protocol_path)

        self.assertEqual(result["decision"], "fail")
        failed = {check["name"] for check in result["checks"] if not check["passed"]}
        self.assertEqual(failed, {"fold_20260722_nmae"})

    def test_rejects_same_cohort_hash_mismatch(self) -> None:
        spec = self.protocol["folds"][0]
        self._write_fold(
            spec["pepd_seed"],
            validation_hash="b" * 64,
            samples=spec["strict_assigned_samples"],
            groups=spec["strict_assigned_groups"],
        )

        with self.assertRaisesRegex(ValueError, "validation cohort hash drift"):
            gate.evaluate(protocol_path=self.protocol_path)

    def test_rejects_frozen_baseline_hash_drift(self) -> None:
        baseline_path = Path(self.protocol["frozen_inputs"]["strict_common_holdout_v2"]["path"])
        baseline_path.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "SHA256 drift"):
            gate.evaluate(protocol_path=self.protocol_path)

    def test_rejects_runtime_root_override_even_with_matching_summaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from the frozen protocol root"):
            gate.evaluate(protocol_path=self.protocol_path, v5_output_root=self.root / "other")

    def test_rejects_incomplete_aggregate_summary(self) -> None:
        aggregate_path = self.v5_root / "summary.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["status"] = "running"
        write_json(aggregate_path, aggregate)

        with self.assertRaisesRegex(ValueError, "aggregate summary is incomplete"):
            gate.evaluate(protocol_path=self.protocol_path)

    def test_rejects_aggregate_fold_sha_drift(self) -> None:
        strict_path = self.v5_root / "strict_oof_summary.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        strict["folds"][2]["summary_sha256"] = "0" * 64
        self._write_root_from_strict(strict)

        with self.assertRaisesRegex(ValueError, "aggregate fold summary SHA256 drift"):
            gate.evaluate(protocol_path=self.protocol_path)


if __name__ == "__main__":
    unittest.main()
