from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.ablate_fadr_router_features import (
    _grouped_splits,
    _route,
    main as legacy_ablation_main,
    run_feature_ablation,
)
from experiments.calibrated_progress_router import (
    FEATURE_NAMES,
    extract_calibrated_router_features,
)
from experiments.fadr_feature_sets import FADR_ROUTER_FEATURE_SETS
from experiments.quality_router import RAW_QUALITY_FEATURES
from experiments.train_quality_router import _candidate_thresholds


def _synthetic_rows() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    calibrations: dict[str, dict[str, object]] = {}
    seeds = (20260720, 20260721, 20260722)
    for group_index in range(18):
        for within_group in range(2):
            index = group_index * 2 + within_group
            sample_id = f"sample-{index:03d}"
            truth_progress = 0.1 + 0.8 * index / 35.0
            base_bias = 0.035 if group_index % 2 == 0 else -0.015
            calibrated_bias = -0.008 if group_index % 2 == 0 else 0.025
            base_progress = float(np.clip(truth_progress + base_bias, 0.0, 1.0))
            vector_progress = float(
                np.clip(truth_progress + calibrated_bias * 1.4, 0.0, 1.0)
            )
            calibrated_progress = float(
                np.clip(truth_progress + calibrated_bias, 0.0, 1.0)
            )
            methods = {
                name: {
                    "progress": float(np.clip(vector_progress + offset, 0.0, 1.0)),
                    "prediction": float(
                        100.0 * np.clip(vector_progress + offset, 0.0, 1.0)
                    ),
                    "pointer_angle": float((index * 9 + angle_offset) % 360),
                }
                for name, offset, angle_offset in (
                    ("weighted_fusion", 0.010, 2),
                    ("geometry_v1", -0.012, 5),
                    ("geometry_v2", 0.018, 8),
                    ("transformer", -0.020, 11),
                )
            }
            raw_features = {
                name: 0.1 + ((index + feature_index) % 13) / 20.0
                for feature_index, name in enumerate(RAW_QUALITY_FEATURES)
            }
            row: dict[str, object] = {
                "dataset": "SyncG",
                "split": "train",
                "sample_id": sample_id,
                "group_id": f"group-{group_index:02d}",
                "held_out_seed": seeds[group_index % len(seeds)],
                "ground_truth": truth_progress * 100.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
                "raw": {
                    "status": True,
                    "branch": (
                        "start_and_end"
                        if group_index % 3 == 0
                        else "start_only"
                        if group_index % 3 == 1
                        else "end_only"
                    ),
                    "methods": methods,
                    "features": raw_features,
                },
                "front_end": {
                    "reference_branch": "start_and_end",
                    "range_angle": 270.0,
                    "meter_confidence": 0.85,
                },
                "base": {
                    "status": True,
                    "prediction": base_progress * 100.0,
                    "gate_probability": 0.2 + 0.6 * (group_index % 2),
                    "residual_normalized": base_bias,
                    "residual_std_normalized": 0.02,
                    "correction_applied": group_index % 2 == 0,
                },
                "vector": {
                    "status": True,
                    "prediction": vector_progress * 100.0,
                    "progress": vector_progress,
                    "pointer_angle": float((index * 9) % 360),
                    "pivot_peak": 0.7,
                    "pivot_input_xy": [120.0 + within_group, 130.0],
                },
            }
            rows.append(row)
            calibrations[sample_id] = {
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "corrected_prediction_oof": calibrated_progress * 100.0,
                "corrected_progress_oof": calibrated_progress,
                "predicted_residual_oof": calibrated_progress - vector_progress,
                "raw_progress": vector_progress,
                "ensemble_std_oof": 0.01 + 0.001 * (index % 5),
            }
    return rows, calibrations


class FadrFeatureAblationTest(unittest.TestCase):
    def test_legacy_sequential_ablation_cli_fails_closed(self) -> None:
        argv = [
            "ablate_fadr_router_features",
            "--oof-pairs",
            "oof.jsonl",
            "--input-preflight",
            "preflight.json",
            "--calibration-diagnostics",
            "calibration.jsonl",
            "--calibration-summary",
            "calibration_summary.json",
            "--calibrator",
            "calibrator.joblib",
            "--output-dir",
            "legacy_output",
            "--seed",
            "20260722",
        ]
        with patch("sys.argv", argv):
            with self.assertRaisesRegex(
                RuntimeError,
                "not valid combined FADR evidence",
            ):
                legacy_ablation_main()

    def test_all_router_features_are_invariant_to_scoring_label_rewrites(
        self,
    ) -> None:
        rows, calibration_by_id = _synthetic_rows()
        original = copy.deepcopy(rows[0])
        calibration = copy.deepcopy(calibration_by_id[original["sample_id"]])
        baseline = extract_calibrated_router_features(
            raw_row=original,
            base_row=original,
            vector_row=original,
            reference_row=None,
            calibration_row=calibration,
        )

        rewritten = copy.deepcopy(original)
        rewritten_calibration = copy.deepcopy(calibration)
        replacements = {
            "ground_truth": -9.87654321e8,
            "target_progress": 12345.6789,
            "direction_angle_error_degrees": -314159.0,
            "sample_id": "adversarial-label-only-identity",
        }
        for container in (
            rewritten,
            rewritten["raw"],
            rewritten["base"],
            rewritten["vector"],
            rewritten_calibration,
        ):
            container.update(replacements)
        changed = extract_calibrated_router_features(
            raw_row=rewritten,
            base_row=rewritten,
            vector_row=rewritten,
            reference_row=None,
            calibration_row=rewritten_calibration,
        )

        self.assertEqual(tuple(baseline), tuple(FEATURE_NAMES))
        self.assertEqual(tuple(changed), tuple(FEATURE_NAMES))
        baseline_vector = np.asarray(
            [baseline[name] for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        changed_vector = np.asarray(
            [changed[name] for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            baseline_vector,
            changed_vector,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        full_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        for variant, names in FADR_ROUTER_FEATURE_SETS.items():
            indices = [full_index[name] for name in names]
            with self.subTest(variant=variant):
                np.testing.assert_allclose(
                    baseline_vector[indices],
                    changed_vector[indices],
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )

    def test_hard_fallback_is_independent_of_router_score(self) -> None:
        self.assertEqual(
            _route(None, 4.0, -100.0, 100.0),
            (4.0, "calibrated_hard_fallback"),
        )
        self.assertEqual(_route(None, None, 1.0, 0.0), (None, "failure"))
        self.assertEqual(
            _route(2.0, 4.0, 0.3, 0.2),
            (4.0, "calibrated_quality_switch"),
        )
        self.assertEqual(_route(2.0, 4.0, 0.2, 0.2), (2.0, "base"))

    def test_grouped_splits_are_deterministic_and_leak_free(self) -> None:
        groups = np.asarray([f"group-{index // 2}" for index in range(24)])
        target = np.linspace(-1.0, 1.0, len(groups))
        first = _grouped_splits(
            groups,
            target,
            folds=3,
            seed=20260722,
            label="fixture",
        )
        second = _grouped_splits(
            groups,
            target,
            folds=3,
            seed=20260722,
            label="fixture",
        )
        for (train_a, validation_a), (train_b, validation_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(validation_a, validation_b)
            self.assertTrue(
                set(groups[train_a]).isdisjoint(set(groups[validation_a]))
            )

    def test_no_switch_threshold_is_finite_and_above_every_score(self) -> None:
        scores = np.asarray([-0.5, 0.0, 0.75], dtype=np.float64)
        thresholds = _candidate_thresholds(scores)
        self.assertTrue(np.isfinite(thresholds).all())
        self.assertGreater(float(np.max(thresholds)), float(np.max(scores)))

    def test_all_five_variants_share_one_nested_fold_assignment(self) -> None:
        rows, calibration_by_id = _synthetic_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oof = root / "train_oof.jsonl"
            oof.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            placeholders = {
                name: root / name
                for name in (
                    "input_preflight.json",
                    "calibration.jsonl",
                    "calibration_summary.json",
                    "calibrator.joblib",
                )
            }
            for path in placeholders.values():
                path.write_bytes(b"fixture")
            args = argparse.Namespace(
                oof_pairs=oof,
                input_preflight=placeholders["input_preflight.json"],
                calibration_diagnostics=placeholders["calibration.jsonl"],
                calibration_summary=placeholders["calibration_summary.json"],
                calibrator=placeholders["calibrator.joblib"],
                output_dir=root / "feature_ablation",
                folds=3,
                inner_folds=3,
                trees=4,
                max_depth=3,
                min_samples_leaf=1,
                max_features=0.70,
                bootstrap_iterations=20,
                seed=20260722,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Skipping features without any observed values",
                    category=UserWarning,
                )
                with (
                    patch(
                        "experiments.ablate_fadr_router_features._validate_preflight",
                        return_value={"input_authorization_sha256": "a" * 64},
                    ),
                    patch(
                        "experiments.ablate_fadr_router_features."
                        "_validate_calibration_inputs",
                        return_value=(
                            list(calibration_by_id.values()),
                            calibration_by_id,
                        ),
                    ),
                    patch(
                        "experiments.ablate_fadr_router_features.MIN_JOINT_SAMPLES",
                        1,
                    ),
                ):
                    diagnostics, summary = run_feature_ablation(args)

        self.assertEqual(
            tuple(summary["variants"]),
            tuple(FADR_ROUTER_FEATURE_SETS),
        )
        self.assertEqual(
            summary["feature_sets"],
            {
                name: list(features)
                for name, features in FADR_ROUTER_FEATURE_SETS.items()
            },
        )
        self.assertEqual(len(summary["fold_summaries"]), 3)
        for fold in summary["fold_summaries"]:
            self.assertEqual(fold["group_overlap"], 0)
            self.assertEqual(len(fold["inner_folds"]), 3)
            self.assertTrue(
                all(inner["group_overlap"] == 0 for inner in fold["inner_folds"])
            )
        for row in diagnostics:
            self.assertEqual(set(row["variants"]), set(FADR_ROUTER_FEATURE_SETS))
            if row["router_fold"] is not None:
                self.assertTrue(
                    all(
                        value["router_score_oof"] is not None
                        and value["nested_threshold"] is not None
                        for value in row["variants"].values()
                    )
                )


if __name__ == "__main__":
    unittest.main()
