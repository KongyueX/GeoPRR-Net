from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.fadr_feature_sets import FADR_ROUTER_FEATURE_SETS
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    FADR_INPUT_PREFLIGHT_PROTOCOL,
    FADR_JOINT_LINEAGE_PROTOCOL,
    FADR_JOINT_TRAINING_PROTOCOL,
    FADR_JOINT_VERIFICATION_PROTOCOL,
    FADR_MULTI_SEED_COHORT_PROTOCOL,
    FADR_PRIMARY_SEED,
    FADR_SEEDS,
    FADR_UDSF_HANDOFF_PROTOCOL,
)
from experiments.quality_router import normalized_error
from experiments.vdn_baseline import sha256_file
from experiments.verify_fadr_multiseed_cohort import (
    _distribution,
    _paired_group_bootstrap_from_seed_averaged_errors,
    _write_no_clobber,
    build_cohort,
)


class VerifyFadrMultiseedCohortTest(unittest.TestCase):
    def test_cohort_rejects_legacy_evidence_and_requires_udsf_context_refit(
        self,
    ) -> None:
        rows = [
            {
                "dataset": "SyncG",
                "split": "train",
                "sample_id": f"sample-{index}",
                "group_id": f"group-{index}",
                "held_out_seed": 20260720 + index % 3,
                "ground_truth": float(index),
                "scale_start": 0.0,
                "scale_end": 10.0,
                "base": {"status": True, "prediction": float(index) + 0.1},
            }
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oof = root / "oof.jsonl"
            oof.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            preflight = root / "preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "protocol": FADR_INPUT_PREFLIGHT_PROTOCOL,
                        "status": "verified",
                        "fadr_seeds": list(FADR_SEEDS),
                        "primary_fadr_seed": FADR_PRIMARY_SEED,
                        "oof_protocol": EXPECTED_OOF_PROTOCOL,
                        "direction_seeds": [20260720, 20260721, 20260722],
                        "input_authorization_sha256": "a" * 64,
                        "inputs": {
                            "oof_pairs": {
                                "path": str(oof.resolve()),
                                "sha256": sha256_file(oof),
                            }
                        },
                        "rows": {
                            "coverage": {
                                "complete_manifest_oof": False,
                            }
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            run_root = root / "run"
            joint_verifications: dict[int, dict] = {}
            component = {
                "evidence_role": "component evidence and final full-data fitting only",
                "combined_fadr_oof_authorized": False,
                "joint_outer_group_lineage_present": False,
                "calibrator": {"model_sha256": "b" * 64},
                "router": {"model_sha256": "c" * 64},
            }
            errors = np.asarray(
                [
                    normalized_error(
                        row,
                        float(row["base"]["prediction"]),
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            metrics = {
                "samples": len(rows),
                "successful": len(rows),
                "coverage": 1.0,
                "nmae": float(np.mean(errors)),
                "acc_1pct": float(np.mean(errors <= 0.01)),
                "acc_2pct": float(np.mean(errors <= 0.02)),
                "acc_5pct": float(np.mean(errors <= 0.05)),
            }
            routing = {
                "counts": {"base": len(rows)},
                "quality_switches": 0,
                "positive_transfers": 0,
                "negative_transfers": 0,
            }
            for seed in FADR_SEEDS:
                seed_root = run_root / f"seed_{seed}"
                joint_root = seed_root / "joint"
                joint_root.mkdir(parents=True)
                component_path = seed_root / "component_verification.json"
                component_path.write_text(
                    json.dumps(component, sort_keys=True),
                    encoding="utf-8",
                )
                diagnostics = []
                for row in rows:
                    base = float(row["base"]["prediction"])
                    diagnostics.append(
                        {
                            "sample_id": row["sample_id"],
                            "base_prediction": base,
                            "reference_conditioned_prediction": base,
                            "variants": {
                                variant: {
                                    "prediction": base,
                                    "normalized_error": normalized_error(
                                        row,
                                        base,
                                    ),
                                }
                                for variant in FADR_ROUTER_FEATURE_SETS
                            },
                        }
                    )
                diagnostics_path = joint_root / "joint_oof_predictions.jsonl"
                diagnostics_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in diagnostics
                    ),
                    encoding="utf-8",
                )
                lineage_path = joint_root / "joint_lineage.json"
                lineage_path.write_text(
                    json.dumps(
                        {"protocol": FADR_JOINT_LINEAGE_PROTOCOL},
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                summary_path = joint_root / "joint_summary.json"
                summary = {
                    "protocol": FADR_JOINT_TRAINING_PROTOCOL,
                    "parameters": {
                        "outer_folds": 3,
                        "bootstrap_iterations": 10,
                    },
                    "feature_sets": {
                        name: list(features)
                        for name, features in FADR_ROUTER_FEATURE_SETS.items()
                    },
                    "source_identity": {"joint_trainer": "d" * 64},
                    "variants": {
                        variant: {
                            "routing": routing,
                            "paired_comparisons": {},
                        }
                        for variant in FADR_ROUTER_FEATURE_SETS
                    },
                }
                summary_path.write_text(
                    json.dumps(summary, sort_keys=True),
                    encoding="utf-8",
                )
                verification_path = seed_root / "joint_verification.json"
                verification = {
                    "protocol": FADR_JOINT_VERIFICATION_PROTOCOL,
                    "status": "verified",
                    "joint_outer_group_nested": True,
                    "combined_fadr_oof_authorized": True,
                    "standalone_calibrator_oof_used": False,
                    "standalone_router_oof_used": False,
                    "input_oof_protocol": EXPECTED_OOF_PROTOCOL,
                    "input_sha256": sha256_file(oof),
                    "input_preflight_sha256": sha256_file(preflight),
                    "input_authorization_sha256": "a" * 64,
                    "samples": len(rows),
                    "groups": len(rows),
                    "metrics": {
                        variant: metrics
                        for variant in FADR_ROUTER_FEATURE_SETS
                    },
                    "source_identity": {"verifier": "e" * 64},
                }
                verification_path.write_text(
                    json.dumps(verification, sort_keys=True),
                    encoding="utf-8",
                )
                joint_verifications[seed] = verification

            def verify_joint(**kwargs):
                return joint_verifications[kwargs["expected_seed"]]

            with (
                patch(
                    "experiments.verify_fadr_multiseed_cohort.verify_training_pair",
                    return_value=component,
                ),
                patch(
                    "experiments.verify_fadr_multiseed_cohort."
                    "_validate_stored_seed_verification",
                    return_value=component,
                ),
                patch(
                    "experiments.verify_fadr_multiseed_cohort.verify_joint_run",
                    side_effect=verify_joint,
                ),
            ):
                cohort = build_cohort(
                    oof_pairs=oof,
                    input_preflight=preflight,
                    run_root=run_root,
                    bootstrap_iterations=20,
                    bootstrap_seed=7,
                )
        self.assertEqual(cohort["protocol"], FADR_MULTI_SEED_COHORT_PROTOCOL)
        self.assertTrue(cohort["joint_outer_group_nested"])
        self.assertFalse(
            cohort["legacy_sequential_combined_evidence_authorized"]
        )
        self.assertEqual(
            cohort["udsf_handoff"]["protocol"],
            FADR_UDSF_HANDOFF_PROTOCOL,
        )
        self.assertFalse(
            cohort["udsf_handoff"][
                "direct_repartition_of_global_joint_oof_allowed"
            ]
        )
        self.assertTrue(
            cohort["udsf_handoff"]["requires_context_specific_refit"]
        )

    def test_distribution_reports_sample_sd_and_range(self) -> None:
        result = _distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)
        self.assertEqual(result["minimum"], 1.0)
        self.assertEqual(result["maximum"], 3.0)
        self.assertEqual(result["range"], 2.0)

    def test_paired_bootstrap_averages_seeds_before_group_resampling(self) -> None:
        candidate = [
            np.asarray([0.0, 2.0, 1.0, 3.0]),
            np.asarray([2.0, 4.0, 3.0, 5.0]),
            np.asarray([4.0, 6.0, 5.0, 7.0]),
        ]
        baseline = [
            np.asarray([1.0, 1.0, 2.0, 2.0]),
            np.asarray([1.0, 1.0, 2.0, 2.0]),
            np.asarray([1.0, 1.0, 2.0, 2.0]),
        ]
        result = _paired_group_bootstrap_from_seed_averaged_errors(
            candidate,
            baseline,
            np.asarray(["physical-a", "physical-a", "physical-b", "physical-b"]),
            seed=7,
            iterations=100,
        )
        expected = float(
            np.mean(np.mean(np.stack(candidate), axis=0))
            - np.mean(np.mean(np.stack(baseline), axis=0))
        )
        self.assertEqual(result["delta_nmae"], expected)
        self.assertEqual(result["physical_groups"], 2)
        self.assertIn("seeds are not bootstrap units", result["replicate_handling"])

    def test_bootstrap_rejects_an_incomplete_seed_cohort(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            _paired_group_bootstrap_from_seed_averaged_errors(
                [np.asarray([0.0]), np.asarray([1.0])],
                [np.asarray([0.0]), np.asarray([1.0])],
                np.asarray(["group"]),
                seed=1,
                iterations=10,
            )

    def test_output_writer_is_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cohort.json"
            _write_no_clobber(path, b"first\n")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _write_no_clobber(path, b"second\n")
            self.assertEqual(path.read_bytes(), b"first\n")


if __name__ == "__main__":
    unittest.main()
