"""Three bounded CPU checks for the exact pair-preserving reference operator."""
import json
import unittest

import torch

from experiments.geoprr_v2_reference_pair_update import (
    reference_pair_ce_correction, reference_pair_contract,
)


COUNTS = {"operator_calls": 0, "CE_correction_calls": 0, "gradient_queries": 0}


def contract(*args, **kwargs):
    COUNTS["operator_calls"] += 1
    return reference_pair_contract(*args, **kwargs)


def correction(*args):
    COUNTS["CE_correction_calls"] += 1
    return reference_pair_ce_correction(*args)


def enumerate_absolute(log_a, log_p, log_b, log_f, value, mode):
    """Only tiny fixtures materialize the complete absolute (a,p,b) tensor."""
    bins = log_a.shape[-1]
    energy = torch.stack([
        torch.stack([
            torch.stack([log_a[:, a]+log_p[:, a, (p-a) % bins]
                         +log_b[:, a, (b-a) % bins]+log_f[:, p, b]
                         for b in range(bins)], -1)
            for p in range(bins)], -2)
        for a in range(bins)], -3)
    log_z_base = energy.flatten(1).logsumexp(-1)
    log_t = energy-log_z_base[:, None, None, None]
    log_m = log_t.logsumexp(1)
    log_c = (log_t+value[:, :, None, None]).logsumexp(1)
    if mode == "pair_preserving":
        log_q = log_m[:, None]+log_t+value[:, :, None, None]-log_c[:, None]
    else:
        changed = log_t+value[:, :, None, None]
        log_q = changed-changed.flatten(1).logsumexp(-1)[:, None, None, None]
    q = log_q.exp()
    joint = torch.stack([
        torch.stack([torch.stack([q[:, a, (a+d) % bins, (a+s) % bins]
                                 for a in range(bins)], -1).sum(-1)
                     for s in range(bins)], -1)
        for d in range(bins)], -2)
    return {"J": joint, "A": q.sum((2, 3)), "Q": q, "log_Q": log_q, "log_T": log_t,
            "log_M_abs": log_m, "log_c_abs": log_c, "log_Z_base": log_z_base,
            "log_Z_updated": (energy+value[:, :, None, None]).flatten(1).logsumexp(-1)}


def fixture(batch=2, bins=4):
    return [torch.randn(batch, bins, dtype=torch.float64).log_softmax(-1),
            torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1),
            torch.randn(batch, bins, bins, dtype=torch.float64).log_softmax(-1),
            torch.randn(batch, bins, bins, dtype=torch.float64).flatten(1).log_softmax(-1).reshape(batch, bins, bins),
            torch.randn(batch, bins, dtype=torch.float64)]


def target_roles(batch=2, bins=4):
    labels = torch.zeros(batch, 3, bins, dtype=torch.float64)
    for image in range(batch):
        for role in range(3):
            left = (image+role) % bins
            fraction = .2+.1*role
            labels[image, role, left] = 1-fraction
            labels[image, role, (left+1) % bins] = fraction
    return labels


def cross_entropy(log_probability, labels):
    target = labels[:, 0, :, None, None]*labels[:, 1, None, :, None]*labels[:, 2, None, None, :]
    return -(target*log_probability).sum((1, 2, 3))


class ReferencePairUpdateTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"CPU_operator_tests": 3, **COUNTS, "differentiable_operator_inputs": 5,
                          "CNN_image_forwards": 0, "GPU_calls": 0,
                          "optimizer_updates": 0}), flush=True)

    def setUp(self):
        torch.manual_seed(20262020)

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)

    def test_dense_probability_marginals_and_zero_constant_updates(self):
        """Indexing, normalization and preserved absolute pair mass agree with enumeration."""
        source = fixture()
        for mode in ("pair_preserving", "global"):
            for value in (torch.zeros_like(source[-1]), torch.full_like(source[-1], 3.25), source[-1]):
                inputs = source[:4]+[value]
                result = contract(*inputs, mode=mode, tile=3, return_reference=True)
                dense = enumerate_absolute(*inputs, mode)
                for key in ("J", "A", "log_Z_base", "log_Z_updated", "log_M_abs", "log_c_abs"):
                    self.close(result[key], dense[key])
                self.close(result["J"].sum((1, 2)), torch.ones(2, dtype=torch.float64))
                self.close(result["A"].sum(-1), torch.ones(2, dtype=torch.float64))
                if mode == "pair_preserving":
                    # Reconstruct the operator-implied absolute Q from its returned normalizers.
                    implied = (dense["log_T"]+value[:, :, None, None]
                               +result["log_M_abs"][:, None]-result["log_c_abs"][:, None]).exp()
                    self.close(implied, dense["Q"])
                    self.close(implied.sum(1), result["log_M_abs"].exp())
                if bool((value == value[:, :1]).all()):
                    original = enumerate_absolute(*(source[:4]+[torch.zeros_like(value)]), mode)
                    self.close(result["J"], original["J"])
                    self.close(result["A"], original["A"])
                    self.close(correction(result, target_roles(), value), torch.zeros(2, dtype=torch.float64))
                elif mode == "global":
                    self.assertGreater(float((dense["Q"].sum(1)-dense["log_M_abs"].exp()).abs().max()), 1e-6)

    def test_exact_role_cross_entropy_and_all_input_gradients(self):
        """No detached M/c term may alter CE or the base gradient at v=0."""
        source = fixture()
        labels = target_roles()
        joint_probe = torch.randn(2, 4, 4, dtype=torch.float64)
        reference_probe = torch.randn(2, 4, dtype=torch.float64)
        for mode in ("pair_preserving", "global"):
            for zero in (False, True):
                inputs = [value.clone().requires_grad_() for value in source]
                expected_inputs = [value.clone().requires_grad_() for value in source]
                if zero:
                    inputs[-1] = torch.zeros_like(source[-1]).requires_grad_()
                    expected_inputs[-1] = torch.zeros_like(source[-1]).requires_grad_()
                result = contract(*inputs, mode=mode, tile=2, return_reference=True)
                dense = enumerate_absolute(*expected_inputs, mode)
                actual_base = enumerate_absolute(*inputs, mode)["log_T"]
                change = correction(result, labels, inputs[-1])
                actual_ce = cross_entropy(actual_base, labels)+change
                expected_ce = cross_entropy(dense["log_Q"], labels)
                self.close(actual_ce, expected_ce)
                self.assertEqual(change.shape, (2,))
                if zero:
                    self.close(change, torch.zeros_like(change))
                    zero_gradient = torch.autograd.grad(change.sum(), inputs[:4], retain_graph=True)
                    COUNTS["gradient_queries"] += 1
                    for gradient in zero_gradient:
                        self.close(gradient, torch.zeros_like(gradient))
                loss = actual_ce.sum()+.37*(result["J"]*joint_probe).sum()+.11*(result["A"]*reference_probe).sum()
                expected_loss = expected_ce.sum()+.37*(dense["J"]*joint_probe).sum()+.11*(dense["A"]*reference_probe).sum()
                gradients = torch.autograd.grad(loss, inputs)
                expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)
                COUNTS["gradient_queries"] += 2
                self.assertEqual(len(gradients), 5)
                for actual, expected in zip(gradients, expected_gradients):
                    self.assertTrue(bool(torch.isfinite(actual).all()))
                    self.close(actual, expected)

    def test_roll_constant_shifts_tile_and_concentrated_stability(self):
        """Common angular gauge and numerical charts cannot change the probability result."""
        source = fixture(bins=5)
        source[0][0] = torch.tensor([0., -800., -1000., -900., -850.], dtype=torch.float64)
        source[0][1] = source[0][0].roll(2)
        source[1].fill_(-900.)
        source[2].fill_(-850.)
        for a in range(5):
            source[1][:, a, (2-a) % 5] = 0.
            source[2][:, a, (4-a) % 5] = 0.
        source[-1] = torch.tensor([[-500., 200., 0., -100., 600.],
                                   [300., -450., 750., 0., -200.]], dtype=torch.float64)
        shifts = (17., -8.5, 11.75, -4.)
        for mode in ("pair_preserving", "global"):
            base = contract(*source, mode=mode, tile=2, return_reference=True)
            dense = enumerate_absolute(*source, mode)
            self.close(base["J"], dense["J"])
            self.close(base["A"], dense["A"])
            rotated = [source[0].roll(2, 1), source[1].roll(2, 1), source[2].roll(2, 1),
                       source[3].roll((2, 2), (1, 2)), source[4].roll(2, 1)]
            rolled = contract(*rotated, mode=mode, tile=3, return_reference=True)
            self.close(rolled["J"], base["J"])
            self.close(rolled["A"], base["A"].roll(2, 1))
            self.close(rolled["log_M_abs"], base["log_M_abs"].roll((2, 2), (1, 2)))
            self.close(rolled["log_c_abs"], base["log_c_abs"].roll((2, 2), (1, 2)))
            shifted_inputs = [value+offset for value, offset in zip(source[:4], shifts)]+[source[-1]]
            shifted = contract(*shifted_inputs, mode=mode, tile=1, return_reference=True)
            wide = contract(*source, mode=mode, tile=9, return_reference=True)
            for other in (shifted, wide):
                for key in ("J", "A", "log_M_abs", "log_c_abs"):
                    self.assertTrue(bool(torch.isfinite(other[key]).all()))
                    self.close(other[key], base[key])
            self.close(shifted["log_Z_base"], base["log_Z_base"]+sum(shifts))
            self.close(shifted["log_Z_updated"], base["log_Z_updated"]+sum(shifts))


if __name__ == "__main__":
    unittest.main()
