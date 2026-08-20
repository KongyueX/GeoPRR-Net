from __future__ import annotations

import unittest

from experiments.benchmark_remstnet_efficiency import (
    ARMS,
    Runtime,
    _latency_statistics,
    parameter_inventory,
)
from experiments.evaluate_remstnet_real_domains import (
    ReMSTNetRealDomainError,
    _validate_full_model,
)
from remstnet.model import (
    ADAPTIVE_REMST_NET_ARCHITECTURE,
    CoordinatedReMSTNet,
)
from experiments.train_remstnet import ADAPTIVE_PROTOCOL
import torch


def _full_metadata() -> dict[str, object]:
    return {
        "architecture": ADAPTIVE_REMST_NET_ARCHITECTURE,
        "protocol": ADAPTIVE_PROTOCOL,
        "epochs": 5,
        "construction": {
            "architecture_variant": "adaptive_budget_progress_mixing_v3",
            "use_progress_mixing": True,
            "learnable_budget_gain": True,
        },
        "source_foundation": {"source_seed": 20262022},
    }


class ReMSTNetExtendedEvaluatorTests(unittest.TestCase):
    def test_formal_evaluators_accept_only_full_five_epoch_v3(self) -> None:
        full = CoordinatedReMSTNet(
            use_progress_mixing=True,
            learnable_budget_gain=True,
        )
        self.assertEqual(_validate_full_model(full, _full_metadata()), 20262022)

        ablation = CoordinatedReMSTNet(
            use_progress_mixing=True,
            learnable_budget_gain=False,
        )
        with self.assertRaises(ReMSTNetRealDomainError):
            _validate_full_model(ablation, _full_metadata())

        wrong_epoch = _full_metadata()
        wrong_epoch["epochs"] = 1
        with self.assertRaises(ReMSTNetRealDomainError):
            _validate_full_model(full, wrong_epoch)

    def test_efficiency_parameter_inventory_counts_unique_modules(self) -> None:
        full = CoordinatedReMSTNet(
            use_progress_mixing=True,
            learnable_budget_gain=True,
        )
        full_runtime = Runtime(
            "full_remstnet",
            full,
            torch.device("cpu"),
            _full_metadata(),
        )
        full_inventory = parameter_inventory(full_runtime)
        self.assertEqual(full_inventory["total_unique"], 4_271_638)
        self.assertEqual(
            full_inventory["trainable_during_remstnet_fit"], 261_528
        )

        anchor_runtime = Runtime(
            "twin_endpoint",
            torch.nn.ModuleList(full._foundation_modules()),
            torch.device("cpu"),
            {},
        )
        anchor_inventory = parameter_inventory(anchor_runtime)
        self.assertEqual(anchor_inventory["total_unique"], 4_010_110)
        self.assertEqual(anchor_inventory["trainable_during_remstnet_fit"], 0)
        self.assertEqual(ARMS, ("raw_foundation", "twin_endpoint", "full_remstnet"))

    def test_latency_statistics_are_finite_and_use_requested_quantiles(self) -> None:
        result = _latency_statistics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["p50_ms"], 2.5)
        self.assertAlmostEqual(result["p95_ms"], 3.85)
        self.assertGreater(result["sample_sd_ms"], 0.0)
        with self.assertRaises(ValueError):
            _latency_statistics([0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
