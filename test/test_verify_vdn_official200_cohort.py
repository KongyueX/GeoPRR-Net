from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.vdn_baseline import PROJECT_DIR, sha256_source_file
from experiments.vdn_official200_protocol import (
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_STOPPING_POLICY,
    OFFICIAL200_VERIFICATION_PROTOCOL,
    canonical_json_sha256,
)
from experiments.verify_vdn_official200 import VERIFICATION_SCHEMA_KEYS
from experiments.verify_vdn_official200_cohort import build_cohort_report


class VDNOfficial200CohortTests(unittest.TestCase):
    def _report(
        self,
        *,
        seed: int,
        run_dir: Path,
        preflight: dict,
        limitation: bool,
    ) -> dict:
        report = {key: None for key in VERIFICATION_SCHEMA_KEYS}
        for field, filename in (
            ("best_checkpoint_sha256", "best.pt"),
            ("last_checkpoint_sha256", "last.pt"),
            ("summary_sha256", "summary.json"),
        ):
            path = run_dir / filename
            path.write_bytes(f"{seed}:{filename}".encode())
            report[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        report.update(
            {
                "protocol": OFFICIAL200_VERIFICATION_PROTOCOL,
                "schema_version": 1,
                "verified": True,
                "training_artifacts_verified": True,
                "eligible_for_three_seed_cohort": True,
                "supporting_test_evaluation_authorized": False,
                "field_confirmatory_evaluation_authorized": False,
                "run_dir": str(run_dir),
                "seed": seed,
                "epochs": 200,
                "best_epoch": 180,
                "best_validation_angle_mae_degrees": 0.6 + seed % 3 / 10,
                "tail_diagnostic": {
                    "diagnostic_only": True,
                    "authorization_gate": False,
                    "plateau_observed": not limitation,
                    "manuscript_limitation_required": limitation,
                    "additional_training_authorized": False,
                    "phase4_authorized": False,
                },
                "stopping_policy": OFFICIAL200_STOPPING_POLICY,
                "official_stopping_boundary_reached": True,
                "additional_training_authorized": False,
                "phase4_authorized": False,
                "optimizer_attempts": 1000,
                "optimizer_steps": 1000,
                "skipped_optimizer_steps": 0,
                "skipped_optimizer_step_rate": 0.0,
                "full_run_max_skipped_optimizer_steps": 1,
                "preflight": preflight,
                "content_inventory": {"identity": "shared"},
                "determinism_policy": {"deterministic": True},
                "determinism_authorization": {
                    "protocol": (
                        "vdn_official200_full_epoch_determinism_probe_v1"
                    ),
                    "report_sha256": "d" * 64,
                },
                "runtime_environment": {
                    "gpu": "synthetic",
                    "pythonhashseed": str(seed),
                },
                "source_hash_protocol": "utf8_source_newlines_lf_v1",
                "training_source_sha256": {"trainer": "a" * 64},
                "verifier_source_sha256": sha256_source_file(
                    PROJECT_DIR
                    / "experiments"
                    / "verify_vdn_official200.py"
                ),
                "authoritative_checkpoint_health": {"valid": True},
                "best_model_health": {"valid": True},
                "test_data_opened_or_read": False,
                "public_data_opened_or_read": False,
                "field_data_opened_or_read": False,
                "sealed_data_opened_or_read": False,
                "confirmatory_data_opened_or_read": False,
            }
        )
        payload = dict(report)
        payload.pop("canonical_verification_payload_sha256")
        report["canonical_verification_payload_sha256"] = (
            canonical_json_sha256(payload)
        )
        return report

    def test_three_equal_200_epoch_runs_authorize_only_supporting_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary).resolve()
            preflight = {
                "protocol": "vdn_official200_training_preflight_v1",
                "report_path": str(run_root / "preflight.json"),
                "report_sha256": "a" * 64,
                "canonical_payload_sha256": "b" * 64,
            }
            entries = []
            for seed in OFFICIAL200_FORMAL_SEEDS:
                run_dir = run_root / f"seed_{seed}"
                run_dir.mkdir()
                report_path = run_dir / "verification_v1.json"
                report = self._report(
                    seed=seed,
                    run_dir=run_dir,
                    preflight=preflight,
                    limitation=seed == 20260721,
                )
                report_path.write_text("{}", encoding="utf-8")
                entries.append((seed, report_path, report))
            result = build_cohort_report(
                entries,
                run_root=run_root,
                preflight_report={
                    "formal_seeds": list(OFFICIAL200_FORMAL_SEEDS)
                },
                preflight_binding=preflight,
            )
            self.assertTrue(result["three_seed_equal_epoch_budget"])
            self.assertTrue(
                result["vdn_supporting_test_evaluation_authorized"]
            )
            self.assertFalse(
                result["field_confirmatory_evaluation_authorized"]
            )
            self.assertFalse(result["additional_training_authorized"])
            self.assertFalse(result["phase4_authorized"])
            self.assertEqual(
                result["tail_diagnostic"][
                    "manuscript_limitation_required_seeds"
                ],
                [20260721],
            )
            self.assertFalse(result["test_data_opened_or_read"])
            self.assertFalse(result["public_data_opened_or_read"])
            self.assertFalse(result["field_data_opened_or_read"])

    def test_incomplete_seed_list_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_cohort_report(
                [],
                run_root=Path("."),
                preflight_report={},
                preflight_binding={},
            )


if __name__ == "__main__":
    unittest.main()
