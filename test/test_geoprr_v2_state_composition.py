"""Fixed-camera state composition keeps GT-valid low-mass edges supervised."""
from __future__ import annotations

import unittest

import torch

from experiments.geoprr_v2_model import two_hot_targets
from experiments.geoprr_v2_objectives import state_composition_losses


def state_fixture(target: torch.Tensor, point: torch.Tensor, mask: torch.Tensor):
    batch_size = len(target)
    field = two_hot_targets(target, 5).movedim(-1, 1)
    output = {"field_probs": field, "support_probs": mask}
    partner = {"field_probs": field.clone(), "support_probs": mask.clone(), "point_probs": point}
    batch = {"field_target": target, "field_mask": mask, "pair_field_mask": mask.clone(),
             "pair_point_target": point, "pair_valid": torch.ones(batch_size, dtype=torch.bool),
             "pair_kind": ["state"] * batch_size, "roi_to_canonical": torch.eye(3).repeat(batch_size, 1, 1),
             "pair_roi_to_canonical": torch.eye(3).repeat(batch_size, 1, 1)}
    return output, partner, batch


class StateCompositionLossTests(unittest.TestCase):
    def test_all_gt_and_endpoint_mass_below_one_have_zero_loss(self):
        target = torch.tensor([[[0., .5, 1.]], [[0., .5, 1.]]])
        point = torch.tensor([[[[.2, .3, .5]]], [[[.5, .5, 0.]]]])
        mask = torch.tensor([[[[1., 1., 1.]]], [[[1., 0., 0.]]]])
        output, partner, batch = state_fixture(target, point, mask)
        losses, counts = state_composition_losses(output, partner, batch)
        for value in losses.values():
            self.assertLess(float(value), 1e-7)
        self.assertEqual(counts, {"eligible": 2, "field_valid": 2, "readout_supervised": 2})

    def test_low_and_zero_predicted_mass_remain_in_mean_and_have_gradients(self):
        target = torch.tensor([[[0., .5, 1.]], [[0., .5, 1.]]])
        point = torch.tensor([[[[1., 0., 0.]]], [[[1., 0., 0.]]]])
        output, partner, batch = state_fixture(target, point, torch.ones_like(point))
        predicted_mask = torch.tensor([[[[.01, .01, .01]]], [[[0., 0., 0.]]]], requires_grad=True)
        output["support_probs"] = predicted_mask
        losses, counts = state_composition_losses(output, partner, batch)
        self.assertEqual(counts, {"eligible": 2, "field_valid": 0, "readout_supervised": 2})
        self.assertAlmostEqual(float(losses["state_reading_cdf"].detach()), .995, places=6)
        self.assertAlmostEqual(float(losses["state_mass_l1"].detach()), .995, places=6)
        (losses["state_reading_cdf"] + losses["state_mass_l1"]).backward()
        self.assertTrue(bool(torch.isfinite(predicted_mask.grad).all()))
        self.assertLess(float(predicted_mask.grad[0, 0, 0, 0]), 0.)
        self.assertLess(float(predicted_mask.grad[1, 0, 0, 0]), 0.)

    def test_last_bin_still_recovers_zero_mass_through_mass_term(self):
        target = torch.ones(1, 1, 2)
        point = torch.tensor([[[[.5, .5]]]])
        mask = torch.tensor([[[[0., 1.]]]])
        output, partner, batch = state_fixture(target, point, mask)
        predicted_mask = torch.zeros_like(mask, requires_grad=True)
        output["support_probs"] = predicted_mask
        losses, counts = state_composition_losses(output, partner, batch)
        self.assertEqual(counts["readout_supervised"], 1)
        self.assertEqual(float(losses["state_reading_cdf"].detach()), 0.)
        self.assertEqual(float(losses["state_mass_l1"].detach()), 1.)
        losses["state_mass_l1"].backward()
        self.assertLess(float(predicted_mask.grad[0, 0, 0, 1]), 0.)

    def test_low_support_reaches_predicted_u_q_and_m_logits(self):
        target = torch.tensor([[[0., 1.]]])
        point = torch.tensor([[[[1., 0.]]]])
        output, partner, batch = state_fixture(target, point, torch.ones_like(point))
        u_logits = torch.tensor([[[[4., 0.]], [[0., 4.]], [[0., 0.]], [[0., 0.]], [[0., 0.]]]], requires_grad=True)
        q_logits = torch.zeros_like(point, requires_grad=True)
        m_logits = torch.full_like(point, -3., requires_grad=True)
        output["field_probs"] = u_logits.softmax(1)
        output["support_probs"] = m_logits.sigmoid()
        partner["point_probs"] = q_logits.flatten(1).softmax(1).reshape_as(point)
        losses, counts = state_composition_losses(output, partner, batch)
        self.assertEqual(counts["field_valid"], 0)
        self.assertEqual(counts["readout_supervised"], 1)
        (losses["state_reading_cdf"] + losses["state_mass_l1"]).backward()
        for logits in (u_logits, q_logits, m_logits):
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_non_state_and_different_camera_maps_are_rejected(self):
        target = torch.tensor([[[0., 1.]]])
        point = torch.tensor([[[[.5, .5]]]])
        output, partner, batch = state_fixture(target, point, torch.ones_like(point))
        batch["pair_kind"] = ["camera"]
        with self.assertRaisesRegex(ValueError, "state-intervention"):
            state_composition_losses(output, partner, batch)
        batch["pair_kind"] = ["state"]
        batch["pair_roi_to_canonical"][0, 0, 2] = .001
        with self.assertRaisesRegex(ValueError, "identical camera"):
            state_composition_losses(output, partner, batch)


if __name__ == "__main__":
    unittest.main()
