from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from experiments.fadr_feature_sets import FADR_ROUTER_FEATURE_SETS
from experiments.train_joint_nested_fadr import (
    DIAGNOSTICS_FILENAME,
    LINEAGE_FILENAME,
    SUMMARY_FILENAME,
    _serialize_json,
    _serialize_jsonl,
    _write_no_clobber,
    run_joint_fadr,
)
from experiments.vdn_baseline import sha256_file
from experiments.verify_joint_nested_fadr import verify_joint_run
from test.test_ablate_fadr_router_features import _synthetic_rows


def _arguments(root: Path, oof: Path) -> argparse.Namespace:
    preflight = root / "input_preflight.json"
    preflight.write_bytes(b"fixture")
    return argparse.Namespace(
        oof_pairs=oof,
        input_preflight=preflight,
        output_dir=root / "joint",
        outer_folds=3,
        calibrator_folds=3,
        router_inner_folds=3,
        calibrator_trees=3,
        calibrator_max_depth=3,
        calibrator_min_samples_leaf=1,
        calibrator_max_features=0.8,
        min_branch_samples=2,
        min_branch_groups=3,
        router_trees=3,
        router_max_depth=3,
        router_min_samples_leaf=1,
        router_max_features=0.7,
        bootstrap_iterations=20,
        seed=20260722,
    )


def _run(
    root: Path,
    rows: list[dict[str, object]],
) -> tuple[list[dict], dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    oof = root / "train_oof.jsonl"
    oof.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args = _arguments(root, oof)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        with (
            patch(
                "experiments.train_joint_nested_fadr._validate_preflight",
                return_value={"input_authorization_sha256": "a" * 64},
            ),
            patch(
                "experiments.train_joint_nested_fadr.MIN_CALIBRATOR_SAMPLES",
                1,
            ),
            patch(
                "experiments.train_joint_nested_fadr.MIN_ROUTER_JOINT_SAMPLES",
                1,
            ),
        ):
            return run_joint_fadr(args)


class TrainJointNestedFadrTest(unittest.TestCase):
    def test_materialized_joint_run_passes_independent_verifier(self) -> None:
        rows, _ = _synthetic_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            diagnostics, lineage, summary = _run(root, rows)
            joint_root = root / "joint"
            lineage_path = joint_root / LINEAGE_FILENAME
            diagnostics_path = joint_root / DIAGNOSTICS_FILENAME
            summary_path = joint_root / SUMMARY_FILENAME
            _write_no_clobber(lineage_path, _serialize_json(lineage))
            _write_no_clobber(
                diagnostics_path,
                _serialize_jsonl(diagnostics),
            )
            summary.update(
                {
                    "lineage": str(lineage_path.resolve()),
                    "lineage_sha256": sha256_file(lineage_path),
                    "diagnostics": str(diagnostics_path.resolve()),
                    "diagnostics_sha256": sha256_file(diagnostics_path),
                }
            )
            _write_no_clobber(summary_path, _serialize_json(summary))
            with patch(
                "experiments.verify_joint_nested_fadr._validate_preflight",
                return_value={"input_authorization_sha256": "a" * 64},
            ):
                verification = verify_joint_run(
                    oof_pairs=root / "train_oof.jsonl",
                    input_preflight=root / "input_preflight.json",
                    joint_root=joint_root,
                    expected_seed=20260722,
                )
            diagnostics[0]["variants"]["full"]["route"] = "tampered"
            diagnostics_path.write_bytes(_serialize_jsonl(diagnostics))
            summary["diagnostics_sha256"] = sha256_file(diagnostics_path)
            summary_path.write_bytes(_serialize_json(summary))
            with (
                patch(
                    "experiments.verify_joint_nested_fadr._validate_preflight",
                    return_value={"input_authorization_sha256": "a" * 64},
                ),
                self.assertRaisesRegex(ValueError, "route/error drifted"),
            ):
                verify_joint_run(
                    oof_pairs=root / "train_oof.jsonl",
                    input_preflight=root / "input_preflight.json",
                    joint_root=joint_root,
                    expected_seed=20260722,
                )
        self.assertEqual(verification["status"], "verified")
        self.assertTrue(verification["joint_outer_group_nested"])
        self.assertTrue(verification["combined_fadr_oof_authorized"])
        self.assertFalse(verification["standalone_calibrator_oof_used"])

    def test_outer_validation_labels_cannot_change_their_joint_predictions(
        self,
    ) -> None:
        rows, _ = _synthetic_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_diagnostics, first_lineage, first_summary = _run(
                root / "first",
                copy.deepcopy(rows),
            )
            held_group = next(
                row["group_id"]
                for row in first_diagnostics
                if row["joint_outer_fold"] == 1
            )
            rewritten = copy.deepcopy(rows)
            for row in rewritten:
                if row["group_id"] == held_group:
                    row["ground_truth"] = 100.0 - float(row["ground_truth"])
                    row["target_progress"] = -12345.0
                    row["direction_angle_error_degrees"] = 98765.0
            second_diagnostics, second_lineage, second_summary = _run(
                root / "second",
                rewritten,
            )

        first_by_id = {
            row["sample_id"]: row
            for row in first_diagnostics
            if row["group_id"] == held_group
        }
        second_by_id = {
            row["sample_id"]: row
            for row in second_diagnostics
            if row["group_id"] == held_group
        }
        self.assertEqual(set(first_by_id), set(second_by_id))
        for sample_id in first_by_id:
            first = first_by_id[sample_id]
            second = second_by_id[sample_id]
            self.assertEqual(
                first["reference_conditioned_prediction"],
                second["reference_conditioned_prediction"],
            )
            self.assertEqual(first["calibrator"], second["calibrator"])
            for variant in FADR_ROUTER_FEATURE_SETS:
                first_variant = dict(first["variants"][variant])
                second_variant = dict(second["variants"][variant])
                first_variant.pop("normalized_error")
                second_variant.pop("normalized_error")
                self.assertEqual(first_variant, second_variant)
        self.assertEqual(
            first_lineage["outer_folds"][0],
            second_lineage["outer_folds"][0],
        )
        self.assertTrue(first_summary["joint_outer_group_nested"])
        self.assertFalse(first_summary["standalone_calibrator_oof_used"])
        self.assertTrue(first_summary["combined_fadr_oof_authorized"])
        self.assertEqual(
            first_summary["feature_sets"],
            {
                name: list(features)
                for name, features in FADR_ROUTER_FEATURE_SETS.items()
            },
        )
        self.assertEqual(
            first_summary["parameters"],
            second_summary["parameters"],
        )

    def test_every_outer_fold_records_complete_exclusion_lineage(self) -> None:
        rows, _ = _synthetic_rows()
        with tempfile.TemporaryDirectory() as temporary:
            _, lineage, _ = _run(Path(temporary), rows)
        self.assertEqual(len(lineage["outer_folds"]), 3)
        for fold in lineage["outer_folds"]:
            self.assertEqual(fold["group_overlap"], 0)
            self.assertTrue(all(fold["outer_validation_exclusion"].values()))
            self.assertFalse(
                fold["bootstrap_prediction_tuning"]["performed"]
            )
            self.assertEqual(
                set(fold["router"]["variants"]),
                set(FADR_ROUTER_FEATURE_SETS),
            )
            self.assertEqual(
                len(fold["calibrator"]["cross_fitted_training_candidates"]),
                3,
            )
            self.assertEqual(len(fold["router"]["inner_folds"]), 3)
            for inner in fold["router"]["inner_folds"]:
                self.assertEqual(inner["group_overlap"], 0)
                self.assertEqual(inner["outer_validation_group_overlap"], 0)

    def test_writer_is_strictly_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "joint.json"
            _write_no_clobber(path, b"first\n")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _write_no_clobber(path, b"second\n")
        with self.assertRaises(ValueError):
            _serialize_json({"value": math.nan})


if __name__ == "__main__":
    unittest.main()
