from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.run_geoprr_v2_compositional import validation_decision


def result(nmae: float, *, severe: int = 3, tail: float = 0.01) -> dict:
    return {"smoke_only": False, "modal_radius": 0, "summary": {
        "samples": 1506, "nmae": nmae, "errors_above_5_percent_fs": severe, "tail10_nmae": tail}}


class CompositionGateTests(unittest.TestCase):
    def test_both_equal_exposure_controls_and_original_are_required(self) -> None:
        candidate, consistency = result(.0018), result(.0020)
        decision = validation_decision(candidate, consistency, result(.0017), result(.0019))
        self.assertEqual(decision["decision"], "validation_rejected")
        self.assertFalse(decision["checks"]["native_strictly_better_than_supervised"])
        decision = validation_decision(candidate, consistency, result(.0019), result(.0017))
        self.assertEqual(decision["decision"], "validation_rejected")
        self.assertFalse(decision["checks"]["native_strictly_better_than_original"])
        decision = validation_decision(candidate, consistency, result(.0019), result(.0019))
        self.assertEqual(decision["decision"], "promising_source_validation")
        self.assertFalse(decision["automatic_seed_expansion"])

    def test_average_gain_cannot_hide_worse_severity_or_tail(self) -> None:
        for candidate in (result(.0018, severe=4), result(.0018, tail=.011)):
            self.assertEqual(validation_decision(candidate, result(.002), result(.002), result(.002))["decision"],
                             "validation_rejected")
        candidate = result(.0018)
        candidate["modal_radius"] = 8
        with self.assertRaises(ValueError):
            validation_decision(candidate, result(.002), result(.002), result(.002))

    def test_arm_configs_only_change_declared_cross_state_losses(self) -> None:
        folder = Path(__file__).resolve().parents[1] / "experiments/configs"
        arms = [json.loads((folder / f"geoprr_v2_compositional_{arm}.json").read_text())
                for arm in ("supervised", "consistency", "composition")]
        for config in arms:
            config.pop("name")
            weights = config.pop("loss_weights")
            self.assertFalse(any(value for key, value in weights.items() if key.startswith("pair_")))
            self.assertFalse(any(weights.get(key, 0) for key in ("q_physical_cdf", "u_local_cdf", "q_gt_support")))
        self.assertEqual(arms[0], arms[1])
        self.assertEqual(arms[1], arms[2])


if __name__ == "__main__":
    unittest.main()
