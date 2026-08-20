from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.evaluate_a15_2_mett_syncg import CONDITIONS
from experiments.evaluate_remst_block_syncg import PROTOCOL as EVALUATION_PROTOCOL
from experiments.summarize_remstnet_ablations import (
    ARM_IDENTITIES,
    FACTORIAL_LABELS,
    ReMSTNetAblationError,
    summarize_ablations,
)


class ReMSTNetAblationSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _payload(self, label: str, error: float) -> dict[str, object]:
        architecture, protocol, variant, mixing, adaptive = ARM_IDENTITIES[label]
        rows: list[dict[str, object]] = []
        for scene_index, scene in enumerate(("scene_a", "scene_b")):
            target = 0.3 + 0.2 * scene_index
            for condition in CONDITIONS:
                candidate = {
                    "prediction": target + error,
                    "absolute_error": error,
                }
                endpoint = {"prediction": target + 0.04, "absolute_error": 0.04}
                rows.append(
                    {
                        "sample_id": f"sample_{scene_index}",
                        "scene_stem": scene,
                        "condition": condition,
                        "normalized_target": target,
                        "external": {"paired": True},
                        "raw_anchor": endpoint,
                        "sarn_endpoint": endpoint,
                        "mett": candidate,
                        "remstnet": candidate,
                    }
                )
        return {
            "protocol": EVALUATION_PROTOCOL,
            "status": "pilot_complete",
            "model": {
                "architecture": architecture,
                "protocol": protocol,
                "epochs": 5,
                "construction": {
                    "architecture_variant": variant,
                    "use_progress_mixing": mixing,
                    "learnable_budget_gain": adaptive,
                },
                "source_foundation": {"source_seed": 20262022},
                "seeds": {"initialization": 20262215, "sample_order": 20262219},
                "parameter_counts": {"trainable": 1},
            },
            "per_sample_condition": rows,
        }

    def _write(self, label: str, error: float) -> Path:
        path = self.root / f"{label}.json"
        path.write_text(json.dumps(self._payload(label, error)), encoding="utf-8")
        return path

    def _paths(self) -> dict[str, Path]:
        errors = {
            "coordinated_fixed_budget_no_mixing": 0.020,
            "progress_mixing_fixed_budget": 0.015,
            "adaptive_budget_no_mixing": 0.016,
            "full_v3": 0.010,
        }
        return {label: self._write(label, errors[label]) for label in FACTORIAL_LABELS}

    def test_factorial_effects_and_full_comparisons_are_reported(self) -> None:
        result = summarize_ablations(self._paths(), bootstrap_replicates=20)
        all_conditions = result["summary"]["all_conditions"]

        self.assertEqual(all_conditions["rows"], 12)
        self.assertAlmostEqual(all_conditions["metrics"]["full_v3"]["nmae"], 0.010)
        self.assertAlmostEqual(
            all_conditions["factorial_effects"]["progress_mixing_at_fixed_budget"]["delta_nmae"],
            -0.005,
        )
        self.assertLess(
            all_conditions["full_vs_ablation"]["adaptive_budget_no_mixing"]["delta_nmae"],
            0.0,
        )
        self.assertTrue(result["scope"]["external_model_comparison_is_reported_separately"])

    def test_mechanism_label_mismatch_is_rejected(self) -> None:
        paths = self._paths()
        payload = json.loads(paths["progress_mixing_fixed_budget"].read_text())
        payload["model"]["construction"]["use_progress_mixing"] = False
        paths["progress_mixing_fixed_budget"].write_text(json.dumps(payload))

        with self.assertRaisesRegex(ReMSTNetAblationError, "mechanism flags"):
            summarize_ablations(paths, bootstrap_replicates=2)

    def test_endpoint_drift_is_rejected(self) -> None:
        paths = self._paths()
        payload = json.loads(paths["adaptive_budget_no_mixing"].read_text())
        payload["per_sample_condition"][0]["sarn_endpoint"]["prediction"] += 0.01
        paths["adaptive_budget_no_mixing"].write_text(json.dumps(payload))

        with self.assertRaisesRegex(ReMSTNetAblationError, "endpoint replay"):
            summarize_ablations(paths, bootstrap_replicates=2)


if __name__ == "__main__":
    unittest.main()
