from __future__ import annotations

import unittest

import torch

from experiments.analyze_unified_pointer_reader_routing import _summarize_rows
from experiments.train_unified_pointer_reader import ParameterEMA, _active_gain_loss
from experiments.unified_pointer_reader import (
    FIXED_ROUTING,
    FULL,
    NO_GEOMETRY_FUSION,
    NO_GEOMETRY_FIXED_ROUTING,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
    adaptive_routing_enabled,
    candidate_mask,
    fixed_geometry_base,
    geometry_fusion_enabled,
    mask_router_logits,
)


class UnifiedPointerReaderAblationTests(unittest.TestCase):
    def test_candidate_masks_remove_exactly_one_requested_branch(self) -> None:
        device = torch.device("cpu")
        expected = {
            FULL: [True, True, True],
            NO_GEOMETRY_FUSION: [True, True, True],
            NO_POLAR_EVIDENCE: [True, False, True],
            NO_RELATIONAL_TRANSPORT: [True, True, False],
            FIXED_ROUTING: [True, True, True],
            NO_GEOMETRY_FIXED_ROUTING: [True, True, True],
        }
        for variant, values in expected.items():
            with self.subTest(variant=variant):
                self.assertEqual(candidate_mask(variant, device=device).tolist(), values)

    def test_joint_factorial_cell_combines_both_inference_interventions(self) -> None:
        self.assertFalse(geometry_fusion_enabled(NO_GEOMETRY_FIXED_ROUTING))
        self.assertFalse(adaptive_routing_enabled(NO_GEOMETRY_FIXED_ROUTING))
        self.assertFalse(geometry_fusion_enabled(NO_GEOMETRY_FUSION))
        self.assertTrue(adaptive_routing_enabled(NO_GEOMETRY_FUSION))
        self.assertTrue(geometry_fusion_enabled(FIXED_ROUTING))
        self.assertFalse(adaptive_routing_enabled(FIXED_ROUTING))

    def test_fixed_geometry_fusion_respects_view_availability(self) -> None:
        raw = torch.tensor([0.2, 0.4, 0.6])
        normalized = torch.tensor([0.8, 0.2, 0.4])
        available = torch.tensor([True, False, True])
        result = fixed_geometry_base(raw, normalized, available)
        self.assertTrue(torch.allclose(result, torch.tensor([0.5, 0.4, 0.5])))

    def test_masked_router_has_zero_probability_for_removed_branch(self) -> None:
        logits = torch.tensor([[1.0, 50.0, 2.0], [-2.0, -1.0, 3.0]])
        active = torch.tensor([True, False, True])
        weights = mask_router_logits(logits, active)
        self.assertTrue(torch.equal(weights[:, 1], torch.zeros(2)))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2)))

    def test_active_gain_loss_ignores_removed_branch(self) -> None:
        predicted = torch.tensor([[0.1, 999.0]], requires_grad=True)
        errors = torch.tensor([[0.4, 0.2, 0.3]])
        active = torch.tensor([True, True, False])
        loss = _active_gain_loss(predicted, errors, active)
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertEqual(float(predicted.grad[0, 1]), 0.0)

    def test_parameter_ema_tracks_one_optimization_trajectory(self) -> None:
        module = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        ema = ParameterEMA(module, decay=0.5)
        with torch.no_grad():
            module.weight.fill_(3.0)
        ema.update(module)
        state = ema.state_dict(module)
        self.assertTrue(torch.equal(state["weight"], torch.full((1, 2), 2.0)))

    def test_routing_summary_compares_adaptive_fixed_and_oracle(self) -> None:
        rows = [
            {
                "normalized_target": 0.20,
                "prediction": 0.20,
                "candidate_predictions": [0.30, 0.20, 0.40],
                "routing_weights": [0.0, 1.0, 0.0],
                "predicted_candidate_gains": [0.10, -0.10],
                "polar_entropy": 0.20,
                "polar_concentration": 0.80,
                "relation_available": True,
            },
            {
                "normalized_target": 0.70,
                "prediction": 0.70,
                "candidate_predictions": [0.70, 0.60, 0.80],
                "routing_weights": [1.0, 0.0, 0.0],
                "predicted_candidate_gains": [-0.10, -0.10],
                "polar_entropy": 0.30,
                "polar_concentration": 0.70,
                "relation_available": False,
            },
        ]
        summary = _summarize_rows(rows)
        self.assertEqual(summary["nmae"], 0.0)
        self.assertGreater(summary["fixed_prior_nmae"], 0.0)
        self.assertEqual(summary["oracle_discrete_nmae"], 0.0)

if __name__ == "__main__":
    unittest.main()
