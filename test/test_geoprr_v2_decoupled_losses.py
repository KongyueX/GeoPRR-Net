"""Contracts for GT-conditioned U/Q readout supervision."""
from __future__ import annotations

import unittest

import torch

from experiments.geoprr_v2_model import distribution_cdf_loss, two_hot_targets
from experiments.geoprr_v2_objectives import decoupled_readout_losses


class DecoupledReadoutLossTests(unittest.TestCase):
    def test_sparse_projection_matches_dense_reference(self) -> None:
        generator = torch.Generator().manual_seed(7)
        field = torch.rand(2, 7, 2, 3, generator=generator).softmax(1)
        point = torch.rand(2, 1, 2, 3, generator=generator).flatten(1).softmax(1).reshape(2, 1, 2, 3)
        gt_point = torch.rand(2, 1, 2, 3, generator=generator).flatten(1).softmax(1).reshape(2, 1, 2, 3)
        mask = torch.tensor([[[[1., 1., 0.], [0., 1., 0.]]], [[[1., 0., 1.], [1., 0., 0.]]]])
        target_u = torch.tensor([[[0., 0.12, 0.5], [0.67, 0.99, 1.]], [[1., 0., 0.3], [0.8, 0.25, 0.6]]])
        batch = {"field_target": target_u, "field_mask": mask, "point_target": gt_point}
        result = decoupled_readout_losses({"field_probs": field, "point_probs": point}, batch)
        gt_field = two_hot_targets(target_u, 7).movedim(-1, 1)

        def compose(u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
            weights = q * mask
            return (u * weights).sum((2, 3)) / weights.sum((1, 2, 3))[:, None]

        target_cdf = compose(gt_field, gt_point).cumsum(1)[:, :-1]
        expected_q = (compose(gt_field, point).cumsum(1)[:, :-1] - target_cdf).abs().mean()
        expected_u = (compose(field, gt_point).cumsum(1)[:, :-1] - target_cdf).abs().mean()
        torch.testing.assert_close(result["q_physical_cdf"], expected_q)
        torch.testing.assert_close(result["u_local_cdf"], expected_u)

    def test_all_gt_is_zero_despite_scalar_label_discretization_mismatch(self) -> None:
        target_u = torch.tensor([[[0.2, 0.7]]])
        gt_point = torch.tensor([[[[0.4, 0.6]]]])
        gt_field = two_hot_targets(target_u, 7).movedim(-1, 1)
        batch = {"field_target": target_u, "point_target": gt_point,
                 "field_mask": torch.ones_like(gt_point), "y": torch.tensor([0.505])}
        losses = decoupled_readout_losses({"field_probs": gt_field, "point_probs": gt_point}, batch)
        self.assertLess(float(losses["q_physical_cdf"]), 1e-7)
        self.assertLess(float(losses["u_local_cdf"]), 1e-7)
        self.assertEqual(float(losses["q_gt_support"]), 0.)
        composition = (gt_field * gt_point).sum((2, 3))
        self.assertGreater(float(distribution_cdf_loss(composition, batch["y"])), 0.1)

    def test_each_auxiliary_reaches_only_its_predicted_head(self) -> None:
        u_logits = torch.zeros(1, 5, 1, 3, requires_grad=True)
        q_logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
        m_logits = torch.zeros_like(q_logits, requires_grad=True)
        output = {"field_probs": u_logits.softmax(1),
                  "point_probs": q_logits.flatten(1).softmax(1).reshape_as(q_logits),
                  "support_probs": m_logits.sigmoid()}
        batch = {"field_target": torch.tensor([[[0., 0.2, 1.]]]),
                 "point_target": torch.tensor([[[[0., 1., 0.]]]]),
                 "field_mask": torch.tensor([[[[1., 1., 0.]]]])}
        losses = decoupled_readout_losses(output, batch)
        for name, active_index in (("q_physical_cdf", 1), ("u_local_cdf", 0)):
            gradients = torch.autograd.grad(losses[name], (u_logits, q_logits, m_logits),
                                            allow_unused=True, retain_graph=True)
            for index, gradient in enumerate(gradients):
                if index == active_index:
                    self.assertGreater(float(gradient.abs().sum()), 0.)
                    self.assertTrue(bool(torch.isfinite(gradient).all()))
                else:
                    self.assertIsNone(gradient)

    def test_low_support_rows_keep_cdf_and_support_gradients(self) -> None:
        q_logits = torch.tensor([[[[0., 1., 6.]]]], requires_grad=True)
        q = q_logits.flatten(1).softmax(1).reshape_as(q_logits)
        field = two_hot_targets(torch.tensor([[[1., 1., 0.]]]), 5).movedim(-1, 1).requires_grad_()
        output = {"field_probs": field, "point_probs": q, "field_valid": torch.tensor([False])}
        batch = {"field_target": torch.tensor([[[0., 1., 0.]]]),
                 "point_target": torch.tensor([[[[1., 0., 0.]]]]),
                 "field_mask": torch.tensor([[[[1., 1., 0.]]]])}
        losses = decoupled_readout_losses(output, batch)
        self.assertLess(float((q * batch["field_mask"]).sum()), 0.15)
        self.assertAlmostEqual(float(losses["q_physical_cdf"]), float(torch.sigmoid(torch.tensor(1.))), places=6)
        self.assertAlmostEqual(float(losses["u_local_cdf"]), 1., places=6)
        self.assertGreater(float(losses["q_gt_support"]), 0.)
        for name in ("q_physical_cdf", "q_gt_support"):
            gradient = torch.autograd.grad(losses[name], q_logits, retain_graph=True)[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
