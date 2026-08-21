from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import evaluate_a15_2_mett_real_domains as shared
from experiments.evaluate_remstnet_real_domains import PROTOCOL as EVALUATION_PROTOCOL
from remstnet.model import ADAPTIVE_REMST_NET_ARCHITECTURE
from experiments.summarize_remstnet_real_domains import (
    EXPECTED_SEEDS,
    INDUSTRIAL_DATASET_KEYS,
    summarize_evaluations,
)
from experiments.train_remstnet import ADAPTIVE_PROTOCOL


def _record(target: float, error: float) -> dict[str, object]:
    return {
        "status": "pass",
        "prediction": target + error,
        "absolute_error": error,
    }


def _payload(seed: int, fit_index: int) -> dict[str, object]:
    target = 0.4
    candidate_error = 0.04 + 0.01 * fit_index
    rows = []
    for condition in shared.CONDITIONS:
        for group in ("g1", "g2"):
            rows.append(
                {
                    "sample_id": f"{group}_sample",
                    "group_id": group,
                    "condition": condition,
                    "normalized_target": target,
                    "candidate": {"mett": _record(target, candidate_error)},
                    "efficientnet_b0": {
                        "sarn_v2": {
                            str(external_seed): _record(target, 0.1)
                            for external_seed in EXPECTED_SEEDS
                        }
                    },
                }
            )
    document = {
        "dataset": {
            "slug": "synthetic",
            "samples": 2,
            "groups": 2,
        },
        "baseline_inputs": {"synthetic": True},
        "endpoint_replay_audit": {
            "source_seed": seed,
            "rows": len(rows),
            "status_mismatch_rows": 0,
            "maximum_absolute_prediction_delta": 0.0,
            "tolerance": 5.0e-4,
            "within_tolerance": True,
        },
        "per_sample_condition": rows,
    }
    return {
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "scope": {
            "inference_precision": "float32",
            "training_or_adaptation_during_evaluation": False,
            "sample_router_or_prediction_fusion": False,
        },
        "source_seed": seed,
        "model": {
            "architecture": ADAPTIVE_REMST_NET_ARCHITECTURE,
            "protocol": ADAPTIVE_PROTOCOL,
            "epochs": 5,
            "construction": {
                "architecture_variant": "adaptive_budget_progress_mixing_v3",
                "use_progress_mixing": True,
                "learnable_budget_gain": True,
            },
            "parameter_counts": {"total_unique": 4_271_638},
            "seeds": {
                "initialization": 100 + fit_index,
                "sample_order": 200 + fit_index,
            },
            "source_foundation": {
                "source_seed": seed,
                "source_protocol": "syncg_lightweight_regression_baselines_v1",
                "source_architecture": "efficientnet_b0",
                "source_epochs": 30,
                "source_checkpoint_selection": "terminal_fixed_epoch",
            },
        },
        "datasets": {
            name: json.loads(json.dumps(document))
            for name in shared.REAL_DATASET_KEYS
        },
    }


class ReMSTNetRealDomainSummaryTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> list[Path]:
        paths = []
        for index, seed in enumerate(EXPECTED_SEEDS):
            path = root / f"seed_{seed}.json"
            path.write_text(
                json.dumps(_payload(seed, index)),
                encoding="utf-8",
            )
            paths.append(path)
        return paths

    def test_three_seed_summary_reports_conditions_and_external_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_evaluations(
                self._write_inputs(Path(directory)),
                bootstrap_replicates=11,
                bootstrap_seed=7,
            )
        self.assertEqual(result["seeds"], list(EXPECTED_SEEDS))
        all_conditions = result["datasets"][shared.REAL_DATASET_KEYS[0]][
            "subsets"
        ]["all_conditions"]
        nmae = all_conditions["three_seed"][
            "remstnet_metrics_mean_sample_sd"
        ]["nmae"]
        self.assertAlmostEqual(nmae["mean"], 0.05)
        self.assertAlmostEqual(nmae["sample_sd"], 0.01)
        paired = all_conditions["three_seed"][
            "mean_row_error_across_independent_seeds"
        ]["paired"]
        self.assertAlmostEqual(paired["delta_nmae"], -0.05)
        self.assertTrue(paired["superiority_ci95"])
        self.assertIn("clean", result["datasets"][shared.REAL_DATASET_KEYS[0]]["subsets"])
        industrial = result["industrial_real_photo_baseline"]
        self.assertEqual(
            industrial["dataset"]["source_datasets"],
            list(INDUSTRIAL_DATASET_KEYS),
        )
        self.assertEqual(industrial["dataset"]["samples"], 6)
        self.assertEqual(industrial["dataset"]["groups"], 6)
        pooled = industrial["subsets"]["projective_pooled"]
        self.assertEqual(pooled["rows_per_seed"], 18)
        self.assertEqual(pooled["groups"], 6)
        self.assertAlmostEqual(
            pooled["three_seed"]["remstnet_metrics_mean_sample_sd"]["nmae"][
                "mean"
            ],
            0.05,
        )

    def test_rejects_incomplete_endpoint_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_inputs(root)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            first = shared.REAL_DATASET_KEYS[0]
            payload["datasets"][first]["endpoint_replay_audit"]["rows"] -= 1
            paths[0].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "endpoint replay"):
                summarize_evaluations(paths, bootstrap_replicates=1)


if __name__ == "__main__":
    unittest.main()
