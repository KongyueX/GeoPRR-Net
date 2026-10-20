"""Check probability, angular topology and dependence in the physical readout."""
import math
import unittest

import torch

from experiments.geoprr_v2_relative_state import RelativeStateReadout, relative_state_nll, relative_targets


class RelativeStateTests(unittest.TestCase):
    def test_all_mass_matches_direct_atom_enumeration(self):
        torch.manual_seed(20262020)
        bins = 8
        joint = torch.rand(3, bins, bins)
        joint /= joint.sum((1, 2), keepdim=True)
        output = RelativeStateReadout(bins)(joint)
        expected = torch.zeros(3, bins, dtype=torch.float64)
        for batch in range(3):
            for d in range(bins):
                for l in range(bins):
                    mass = float(joint[batch, d, l])
                    if l in (0, bins - 1):
                        expected[batch] += mass / bins
                        continue
                    span = 2 * math.pi * l / (bins - 1)
                    delta = 2 * math.pi * d / bins
                    angle = (delta - span / 2 + math.pi) % (2 * math.pi) - math.pi + span / 2
                    position = min(1., max(0., angle / span)) * (bins - 1)
                    low = math.floor(position)
                    high = min(bins - 1, low + 1)
                    expected[batch, low] += mass * (1 - position + low)
                    expected[batch, high] += mass * (position - low)
        torch.testing.assert_close(output["distribution"].double(), expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(output["distribution"].sum(1), torch.ones(3), atol=2e-7, rtol=0)
        invalid = torch.zeros(1, bins, bins)
        invalid[0, 4, 0] = 1
        uncertain = RelativeStateReadout(bins)(invalid)
        torch.testing.assert_close(uncertain["distribution"], torch.full((1, bins), 1 / bins))
        torch.testing.assert_close(uncertain["reading"], torch.tensor([.5]))

    def test_same_marginals_do_not_preserve_ratio_reading(self):
        joint = torch.zeros(1, 8, 8)
        joint[0, 1, 2] = joint[0, 2, 4] = .5
        independent = joint.sum(2)[:, :, None] * joint.sum(1)[:, None, :]
        readout = RelativeStateReadout(8)
        torch.testing.assert_close(readout(joint)["reading"], torch.tensor([.4375]))
        torch.testing.assert_close(readout(independent)["reading"], torch.tensor([.4921875]))

    def test_roundoff_at_endpoint_does_not_create_out_of_range_prediction(self):
        joint = torch.zeros(1, 8, 8)
        joint[0, 4, 2] = 1 + 1e-6
        self.assertEqual(float(RelativeStateReadout(8)(joint)["reading"]), 1.)

    def test_delta_is_circular_but_span_is_ordered(self):
        a = relative_targets(torch.tensor([.01]), torch.tensor([2 * math.pi - .01]), 256)
        b = relative_targets(torch.tensor([2 * math.pi + .01]), torch.tensor([2 * math.pi - .01]), 256)
        for index in (0, 1, 3, 4):
            torch.testing.assert_close(a[index], b[index])
        self.assertEqual(int(a[3]), 254)
        self.assertEqual(int(a[4]), 255)
        self.assertGreater(float(a[5]), .5)

    def test_normalized_joint_has_finite_learning_gradients(self):
        torch.manual_seed(20262020)
        logits = torch.randn(2, 2, 2, 16, requires_grad=True)
        weights = torch.randn(2, 2, requires_grad=True)
        p = logits.softmax(-1)
        joint = torch.einsum("br,brd,brl->bdl", weights.softmax(-1), p[:, 0], p[:, 1])
        output = {"joint": joint, **RelativeStateReadout(16)(joint)}
        loss = relative_state_nll(output, torch.tensor([.3, 2.4]), torch.tensor([1.5, 4.2]))
        loss += (output["reading"] - torch.tensor([.2, .6])).abs().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(weights.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
