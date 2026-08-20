from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_a15_2_mett_syncg import CONDITIONS
from experiments.evaluate_remstnet_syncg import PROTOCOL as EVALUATION_PROTOCOL
from remstnet.model import ADAPTIVE_REMST_NET_ARCHITECTURE
from experiments.summarize_remstnet_multiseed import (
    EXPECTED_SOURCE_SEEDS,
    ReMSTNetMultiseedError,
    summarize_evaluations,
)
from experiments.train_remstnet import ADAPTIVE_PROTOCOL


class ReMSTNetMultiseedSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _external(target: float) -> dict[str, object]:
        result: dict[str, object] = {}
        for index, name in enumerate(
            ("Direct-ResNet18", "EfficientNet-B0", "MobileNetV3-Large")
        ):
            predictions = [target + 0.04 + index * 0.01] * 3
            result[name] = {
                "seeds": list(EXPECTED_SOURCE_SEEDS),
                "predictions": predictions,
                "absolute_errors": [abs(value - target) for value in predictions],
                "passed": [True, True, True],
            }
        return result

    def _payload(self, source_seed: int) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for scene_index, scene in enumerate(("scene_a", "scene_b")):
            target = 0.3 + 0.2 * scene_index
            for condition in CONDITIONS:
                prediction = target + 0.01 + 0.001 * (
                    EXPECTED_SOURCE_SEEDS.index(source_seed)
                )
                candidate = {
                    "prediction": prediction,
                    "absolute_error": abs(prediction - target),
                }
                rows.append(
                    {
                        "sample_id": f"sample_{scene_index}",
                        "scene_stem": scene,
                        "condition": condition,
                        "normalized_target": target,
                        "external": self._external(target),
                        "mett": candidate,
                        "remstnet": candidate,
                        "sarn_endpoint": {
                            "prediction": target + 0.05,
                            "absolute_error": 0.05,
                        },
                    }
                )
        return {
            "protocol": EVALUATION_PROTOCOL,
            "status": "pilot_complete",
            "scope": {
                "single_seed_architecture_pilot": True,
                "training_or_adaptation_during_evaluation": False,
                "prediction_dependent_routing": False,
                "inference_precision": "float32",
            },
            "evaluation_elapsed_seconds": 1.0,
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
                    "initialization": 100
                    + EXPECTED_SOURCE_SEEDS.index(source_seed),
                    "sample_order": 200
                    + EXPECTED_SOURCE_SEEDS.index(source_seed),
                },
                "source_foundation": {
                    "source_seed": source_seed,
                    "source_protocol": "syncg_lightweight_regression_baselines_v1",
                    "source_architecture": "efficientnet_b0",
                    "source_epochs": 30,
                    "source_checkpoint_selection": "terminal_fixed_epoch",
                },
            },
            "per_sample_condition": rows,
        }

    def _write(self, source_seed: int, payload: dict[str, object] | None = None) -> Path:
        path = self.root / f"seed_{source_seed}.json"
        value = payload if payload is not None else self._payload(source_seed)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_three_seed_external_comparison_is_paired(self) -> None:
        paths = [self._write(seed) for seed in EXPECTED_SOURCE_SEEDS]
        result = summarize_evaluations(paths, bootstrap_replicates=20)

        all_conditions = result["summary"]["all_conditions"]
        self.assertEqual(all_conditions["rows_per_seed"], 12)
        self.assertAlmostEqual(
            all_conditions["remstnet_v3"]["metric_across_seed_mean_sd"]["nmae"]["mean"],
            0.011,
        )
        paired = all_conditions["external_models"]["EfficientNet-B0"][
            "paired_rowwise_three_seed_mean"
        ]
        self.assertLess(paired["delta_nmae"], 0.0)
        self.assertTrue(paired["superiority_ci95"])
        self.assertFalse(result["scope"]["prediction_ensemble_used_for_claim"])

    def test_ablation_cannot_be_mislabeled_as_full_model(self) -> None:
        paths = [self._write(seed) for seed in EXPECTED_SOURCE_SEEDS]
        payload = self._payload(EXPECTED_SOURCE_SEEDS[0])
        payload["model"]["construction"]["use_progress_mixing"] = False
        paths[0] = self._write(EXPECTED_SOURCE_SEEDS[0], payload)

        with self.assertRaisesRegex(ReMSTNetMultiseedError, "mechanism identity"):
            summarize_evaluations(paths, bootstrap_replicates=2)

    def test_external_seed_roster_drift_is_rejected(self) -> None:
        paths = [self._write(seed) for seed in EXPECTED_SOURCE_SEEDS]
        payload = self._payload(EXPECTED_SOURCE_SEEDS[1])
        payload["per_sample_condition"][0]["external"]["EfficientNet-B0"][
            "seeds"
        ] = [20262021, 20262020, 20262022]
        paths[1] = self._write(EXPECTED_SOURCE_SEEDS[1], payload)

        with self.assertRaisesRegex(ReMSTNetMultiseedError, "seed roster"):
            summarize_evaluations(paths, bootstrap_replicates=2)

    def test_same_seed_endpoint_replay_drift_is_rejected(self) -> None:
        paths = [self._write(seed) for seed in EXPECTED_SOURCE_SEEDS]
        payload = self._payload(EXPECTED_SOURCE_SEEDS[2])
        payload["per_sample_condition"][0]["sarn_endpoint"]["prediction"] += 0.01
        paths[2] = self._write(EXPECTED_SOURCE_SEEDS[2], payload)

        with self.assertRaisesRegex(ReMSTNetMultiseedError, "endpoint replay"):
            summarize_evaluations(paths, bootstrap_replicates=2)


if __name__ == "__main__":
    unittest.main()
