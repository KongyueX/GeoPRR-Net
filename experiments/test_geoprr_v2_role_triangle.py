"""Bounded CPU checks for exact association, deletion and joint-role supervision."""
import json
import math
import unittest

import torch

from experiments.geoprr_v2_role_triangle import pair_log_factor, triangle_contract


COUNTS = {"pair_factor_calls": 0, "triangle_contract_calls": 0, "gradient_queries": 0}


def factor(*args, **kwargs):
    COUNTS["pair_factor_calls"] += 1
    return pair_log_factor(*args, **kwargs)


def contract(*args, **kwargs):
    COUNTS["triangle_contract_calls"] += 1
    return triangle_contract(*args, **kwargs)


def enumerate_joint(log_a, log_p, log_b, log_factor):
    bins = log_a.shape[-1]
    weights = torch.stack([
        torch.stack([
            torch.stack([log_a[:, a]+log_p[:, a, d]+log_b[:, a, s]
                         +log_factor[:, (a+d) % bins, (a+s) % bins]
                         for s in range(bins)], -1)
            for d in range(bins)], -2)
        for a in range(bins)], -3)
    log_z = weights.flatten(1).logsumexp(-1)
    probability = (weights-log_z[:, None, None, None]).exp()
    return probability.sum(1), log_z, weights


class RoleTriangleTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"CPU_operator_tests": 3, **COUNTS, "CNN_image_forwards": 0,
                          "GPU_calls": 0, "optimizer_updates": 0}), flush=True)

    def setUp(self):
        torch.manual_seed(20262020)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)

    def test_enumeration_and_all_input_gradients(self):
        """Wrong indexing or checkpoint recomputation must change values or gradients."""
        batch, bins, dim = 2, 5, 16
        source = [torch.randn(batch, bins, dtype=torch.float64).log_softmax(-1),
                  torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1),
                  torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1),
                  torch.randn(batch, bins, dim, dtype=torch.float64),
                  torch.randn(batch, bins, dim, dtype=torch.float64),
                  torch.randn(dim, dim, dtype=torch.float64)/4]
        probe = torch.randn(batch, bins, bins, dtype=torch.float64)
        for association in ("full", "separable"):
            tiled_inputs = [value.clone().requires_grad_() for value in source]
            exact_inputs = [value.clone().requires_grad_() for value in source]
            tiled_factor = factor(*tiled_inputs[3:], association=association)
            exact_factor = factor(*exact_inputs[3:], association=association)
            joint, log_z = contract(*tiled_inputs[:3], tiled_factor, tile=2)
            expected, expected_log_z, _ = enumerate_joint(*exact_inputs[:3], exact_factor)
            self.assertClose(joint, expected)
            self.assertClose(log_z, expected_log_z)
            self.assertClose(joint.sum((1, 2)), torch.ones(batch, dtype=torch.float64))
            gradient = torch.autograd.grad((joint*probe).sum()+.3*log_z.sum(), tiled_inputs)
            expected_gradient = torch.autograd.grad((expected*probe).sum()+.3*expected_log_z.sum(), exact_inputs)
            COUNTS["gradient_queries"] += 2
            for actual, wanted in zip(gradient, expected_gradient):
                self.assertClose(actual, wanted)

    def test_zero_initialization_marginals_and_joint_cross_entropy(self):
        """Zero W must reproduce star; deletion must preserve the same factor marginals."""
        bins, dim = 5, 16
        log_a = torch.randn(1, bins, dtype=torch.float64).log_softmax(-1)
        log_p = torch.randn(1, bins, bins, dtype=torch.float64).log_softmax(-1)
        log_b = torch.randn(1, bins, bins, dtype=torch.float64).log_softmax(-1)
        h_p, h_b = (torch.randn(1, bins, dim, dtype=torch.float64) for _ in range(2))
        zero_w = torch.zeros(dim, dim, dtype=torch.float64)
        star = log_p.exp().transpose(1, 2) @ (log_a.exp()[:, :, None]*log_b.exp())
        zero_factor = factor(h_p, h_b, zero_w)
        joint, log_z = contract(log_a, log_p, log_b, zero_factor, tile=3)
        self.assertClose(joint, star)
        self.assertClose(log_z, torch.full((1,), -2*math.log(bins), dtype=torch.float64))
        self.assertClose(zero_factor, factor(h_p, h_b, zero_w, association="separable"))
        w = torch.randn(dim, dim, dtype=torch.float64)/4
        full = factor(h_p, h_b, w).exp()
        separable = factor(h_p, h_b, w, association="separable").exp()
        self.assertClose(full.sum(-1), separable.sum(-1))
        self.assertClose(full.sum(-2), separable.sum(-2))
        q_a, q_p, q_b = (torch.randn(1, bins, dtype=torch.float64).softmax(-1) for _ in range(3))
        absolute = (torch.arange(bins)[:, None]+torch.arange(bins)[None]) % bins
        q_relative = q_a[:, :, None, None]*q_p[:, absolute][:, :, :, None]*q_b[:, absolute][:, :, None, :]
        star_log = log_a[:, :, None, None]+log_p[:, :, :, None]+log_b[:, :, None, :]
        _, _, log_weight = enumerate_joint(log_a, log_p, log_b, zero_factor)
        global_ce = log_z-(q_relative*log_weight).sum((1, 2, 3))
        star_ce = -(q_relative*star_log).sum((1, 2, 3))
        self.assertClose(global_ce, star_ce)
        random_factor = factor(h_p, h_b, w)
        _, random_log_z = contract(log_a, log_p, log_b, random_factor, tile=2)
        _, enumerated_log_z, log_weight = enumerate_joint(log_a, log_p, log_b, random_factor)
        joint_ce = random_log_z-(q_relative*log_weight).sum((1, 2, 3))
        enumerated_ce = -(q_relative*(log_weight-enumerated_log_z[:, None, None, None])).sum((1, 2, 3))
        self.assertClose(joint_ce, enumerated_ce)

    def test_common_rotation_and_constant_factor_shift(self):
        """Absolute gauge or constant factor offsets must not change relative probabilities."""
        batch, bins = 2, 5
        log_a = torch.randn(batch, bins, dtype=torch.float64).log_softmax(-1)
        log_p = torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1)
        log_b = torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1)
        log_factor = torch.randn(batch, bins, bins, dtype=torch.float64).flatten(1).log_softmax(-1).reshape(batch, bins, bins)
        original, log_z = contract(log_a, log_p, log_b, log_factor, tile=2)
        rotated, rotated_log_z = contract(log_a.roll(2, -1), log_p.roll(2, 1), log_b.roll(2, 1),
                                         log_factor.roll((2, 2), (1, 2)), tile=3)
        shifted, shifted_log_z = contract(log_a, log_p, log_b, log_factor+19.25, tile=2)
        self.assertClose(rotated, original)
        self.assertClose(rotated_log_z, log_z)
        self.assertClose(shifted, original)
        self.assertClose(shifted_log_z, log_z+19.25)


if __name__ == "__main__":
    unittest.main()
