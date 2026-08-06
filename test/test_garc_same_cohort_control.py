from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from experiments import garc_same_cohort_control as control


def _plan(seed: int = 20260720) -> dict:
    return {
        "execution_mode": "formal_frozen",
        "parent_protocol": {"path": "C:/pointer_read/protocol.json"},
        "method": {"name": f"GARC-v5-tiny-top1-fold{seed}+auto-ref"},
        "progress_component": {
            "binding_file": {"path": f"C:/pointer_read/progress-{seed}.json"},
            "binding": {
                "artifact_sha256": {"checkpoint": f"progress-{seed}"},
            },
        },
        "progress_factory": {
            "path": "D:/project/PointerMeterReaderFastAPI/experiments/garc_pepd_progress_factory.py",
            "function": "build_progress_provider",
        },
        "joint_oof": {
            "status": "bound_for_overlap_audit",
            "summary": {"path": "C:/pointer_read/joint.json"},
        },
        "reference": {
            "mode": "method_internal_auto",
            "detector_sha256": "1" * 64,
        },
        "garc": {
            "recognizer_kind": "tiny",
            "consensus_mode": "top1",
            "geometry_mode": "v5",
            "geometry_provider": "enhanced_v5_oof_fold",
            "effective_decoder_protocol": "garc_top1_arithmetic_ransac_control_v1",
            "device": "cuda:0",
            "input_size": 768,
            "detector_threshold": 0.4,
            "posterior_top_k": 5,
            "consensus_config": {"x": 1},
            "geometry_fold": {
                "pepd_seed": seed,
                "joint_shard_all_component_unseen": True,
                "oof_summary": {"path": "C:/pointer_read/v5-oof.json"},
            },
            "artifacts": {
                "detector": {"path": "C:/pointer_read/detector.pt"},
                "recognizer": {"path": "C:/pointer_read/recognizer.pt"},
                "geometry": {"path": f"C:/pointer_read/head-{seed}.pt"},
                "geometry_backbone": {
                    "path": f"C:/pointer_read/backbone-{seed}.pt"
                },
            },
        },
        "audit": {
            "caller_numeric_range_permitted": False,
            "caller_geometry_permitted": False,
            "caller_reference_packet_permitted": False,
        },
    }


def _report(mapping: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "protocol": control.GARC_VALIDATION_PROTOCOL,
        "status": "independent_validation_complete",
        "mode": "formal",
        "claim_eligible": True,
        "claim_boundaries": {"full_1080_end_to_end_claim_allowed": False},
        "overlap_audit": {
            "jointly_unseen_samples": 412,
            "jointly_unseen_groups": 19,
            "joint_cohort_complete": True,
            "joint_mapping_sha256": (
                mapping or control.EXPECTED_JOINT_SUMMARY_SHA256
            ),
        },
        "evidence_eligibility": {
            "range_component_claim": False,
            "joint_oof_end_to_end_claim": True,
        },
        "metrics": {
            "joint_oof_range_frozen_acceptance": {
                "samples": 412,
                "groups": 19,
                "coverage": 0.82,
                "pair_rounded_exact_full_denominator": 0.73,
                "pair_rounded_exact_conditional": 0.89,
            },
            "joint_oof_end_to_end_frozen_acceptance": {
                "samples": 412,
                "groups": 19,
                "coverage": 0.78,
                "reading_nmae_full_denominator_failure_penalty_1": 0.24,
            },
        },
        "audit": {
            "validation_predictions_verified_before_public_values_opened": True,
            "calibration_frozen_before_validation_values_opened": True,
            "restricted_namespace_images_opened": 0,
            "numeric_range_values_supplied_to_model": 0,
            "manual_geometry_supplied_to_model": 0,
        },
    }


class GarcSameCohortControlTests(unittest.TestCase):
    def test_exact_control_plan_contract(self) -> None:
        control.assert_control_plan(_plan(), expected_seed=20260720)

    def test_control_plan_rejects_posterior_or_fusion_variants(self) -> None:
        for key, value in (
            ("consensus_mode", "topk"),
            ("geometry_mode", "v5_pepd_fusion"),
            ("recognizer_kind", "strong"),
        ):
            with self.subTest(key=key):
                value_plan = _plan()
                value_plan["garc"][key] = value
                with self.assertRaises(ValueError):
                    control.assert_control_plan(value_plan, expected_seed=20260720)

    def test_formal_validation_schema_is_accepted_by_promotion_parser(self) -> None:
        metrics = control.validate_control_report_contract(_report())
        self.assertEqual(
            metrics["joint_mapping_sha256"],
            control.EXPECTED_JOINT_SUMMARY_SHA256,
        )
        self.assertAlmostEqual(float(metrics["e2e_nmae"]), 0.24)

    def test_control_report_rejects_different_mapping(self) -> None:
        with self.assertRaises(ValueError):
            control.validate_control_report_contract(_report("a" * 64))

    def test_seed_materialization_spec_changes_only_fold_routes(self) -> None:
        primary = _plan(20260720)
        donor = _plan(20260721)
        donor["progress_component"]["binding_file"]["path"] = (
            "C:/pointer_read/progress-20260721.json"
        )
        donor["garc"]["artifacts"]["geometry"]["path"] = (
            "C:/pointer_read/head-20260721.pt"
        )
        donor["garc"]["artifacts"]["geometry_backbone"]["path"] = (
            "C:/pointer_read/backbone-20260721.pt"
        )
        spec = control._freeze_spec(
            seed20_control=primary,
            donor=donor,
            seed=20260721,
        )
        self.assertEqual(spec["recognizer_checkpoint"], primary["garc"]["artifacts"]["recognizer"]["path"])
        self.assertEqual(spec["detector_threshold"], 0.4)
        self.assertEqual(spec["progress_binding_path"], donor["progress_component"]["binding_file"]["path"])
        self.assertEqual(spec["geometry_checkpoint"], donor["garc"]["artifacts"]["geometry"]["path"])
        self.assertIn("20260721", spec["method_name"])

    def test_parent_chain_freezes_control_before_validation(self) -> None:
        audit = control._garc_source_order_audit()
        self.assertTrue(
            audit["control_plan_and_calibration_precede_independent_validation"]
        )
        self.assertLess(audit["control_plan_freeze"], audit["independent_validation"])

    def test_preregistration_and_zero_data_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg = root / "preregistration.json"
            output = root / "preflight.json"
            control.freeze_preregistration(
                output_path=prereg,
                garc_output_root=root / "future-garc-output",
                garc_process_id=13644,
                garc_process_start_utc="2026-08-06T15:19:32.565397Z",
                garc_process_command_sha256="b" * 64,
            )
            _, frozen = control.load_preregistration(prereg)
            self.assertTrue(
                frozen["absence_attestation"][
                    "all_expected_garc_result_and_control_paths_absent"
                ]
            )
            control.preflight(
                preregistration_path=prereg,
                output_path=output,
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["audit"]["images_opened"], 0)
            self.assertEqual(value["audit"]["annotations_opened"], 0)
            self.assertFalse(value["audit"]["inference_started"])
            self.assertFalse(value["audit"]["training_started"])

    def test_preregistration_rejects_already_revealed_garc_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            garc_root = root / "future-garc-output"
            garc_root.mkdir()
            (garc_root / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                control.freeze_preregistration(
                    output_path=root / "preregistration.json",
                    garc_output_root=garc_root,
                    garc_process_id=13644,
                    garc_process_start_utc="2026-08-06T15:19:32.565397Z",
                    garc_process_command_sha256="b" * 64,
                )

    def test_event_chain_is_wait_based_and_promotion_is_after_verified_seal(self) -> None:
        path = control.CONTROL_EVENT_CHAIN
        text = path.read_text(encoding="utf-8-sig")
        seal = text.find('"seal-formal-same-412-control"')
        verify = text.find('"verify-sealed-control-before-promotion"')
        promote = text.find('"common-split-promotion-after-control-seal"')
        self.assertGreaterEqual(seal, 0)
        self.assertLess(seal, verify)
        self.assertLess(verify, promote)
        self.assertIn("WaitForExit()", text)
        self.assertNotIn("Start-Process", text)
        self.assertNotIn("Start-Sleep", text)
        self.assertNotIn("send_feishu", text.casefold())

    def test_event_chain_powershell_ast_is_valid(self) -> None:
        command = (
            "$tokens=$null;$errors=$null;"
            "[void][Management.Automation.Language.Parser]::ParseFile("
            f"'{control.CONTROL_EVENT_CHAIN}',[ref]$tokens,[ref]$errors);"
            "if($errors.Count){$errors|%{$_.ToString()};exit 1}"
        )
        completed = subprocess.run(
            [
                r"C:\Program Files\PowerShell\7\pwsh.exe",
                "-NoLogo",
                "-NoProfile",
                "-Command",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
