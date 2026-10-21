"""Focused CPU checks for global reference normalization on generated tokens."""
import argparse
import json
import os
from pathlib import Path
import time
import unittest

import torch

from experiments.geoprr_v2_reference_relation import (
    CircularProgressPushforward, ReferenceRelationHead, circular_two_hot_targets,
)


REPORT = {"component_feature_forward_calls": 0, "component_feature_examples": 0,
          "component_feature_tokens": 0, "component_backward_calls": 0,
          "direct_global_probability_fixtures": 0, "row_offset_fixtures": 0}


def forward_tokens(head, features):
    output = head(features)
    REPORT["component_feature_forward_calls"] += 1
    REPORT["component_feature_examples"] += len(features)
    REPORT["component_feature_tokens"] += features.shape[0]*features.shape[1]
    return output


def raw_scores(head, features):
    a, p, b = [projection(features) for projection in head.projections]
    ua, up, ub = [layer(role).squeeze(-1) for layer, role in zip(head.unaries, (a, p, b))]
    return ua, (a @ p.transpose(1, 2))*head.scale+up[:, None], (a @ b.transpose(1, 2))*head.scale+ub[:, None]


def absolute_log_conditionals(output):
    bins = output["A"].shape[-1]
    relative = (torch.arange(bins)[None]-torch.arange(bins)[:, None]) % bins
    indices = relative[None].expand(len(output["A"]), -1, -1)
    return output["log_P_rel"].gather(-1, indices), output["log_B_rel"].gather(-1, indices)


class ReferenceFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)

    def test_global_triple_probability_matches_factored_head(self):
        bins = 7
        head = ReferenceRelationHead(feature_dim=4, bins=bins, reference_normalization="global").double()
        features = .7*torch.randn(2, bins, 4, dtype=torch.float64)
        output = forward_tokens(head, features)
        ua, ep, eb = raw_scores(head, features)
        torch.testing.assert_close(output["reference_unary"], ua, atol=0, rtol=0)
        torch.testing.assert_close(output["log_Z_P"], ep.logsumexp(-1), atol=0, rtol=0)
        torch.testing.assert_close(output["log_Z_B"], eb.logsumexp(-1), atol=0, rtol=0)
        torch.testing.assert_close(output["log_A"], (ua+ep.logsumexp(-1)+eb.logsumexp(-1)).log_softmax(-1),
                                   atol=1e-13, rtol=0)
        energy = ua[:, :, None, None]+ep[:, :, :, None]+eb[:, :, None, :]
        direct = energy.flatten(1).log_softmax(-1).reshape_as(energy).exp()
        lp, lb = absolute_log_conditionals(output)
        factored = output["A"][:, :, None, None]*lp.exp()[:, :, :, None]*lb.exp()[:, :, None, :]
        torch.testing.assert_close(factored, direct, atol=1e-14, rtol=1e-12)
        expected_joint = torch.zeros_like(output["J"])
        for a in range(bins):
            for d in range(bins):
                for s in range(bins):
                    expected_joint[:, d, s] += direct[:, a, (a+d) % bins, (a+s) % bins]
        torch.testing.assert_close(output["J"], expected_joint, atol=1e-14, rtol=1e-12)
        for key in ("J", "product"):
            torch.testing.assert_close(output[key].sum((1, 2)), torch.ones(2, dtype=torch.float64), atol=1e-13, rtol=0)
        REPORT["direct_global_probability_fixtures"] += 2

    def test_twohot_nll_cancels_row_partitions(self):
        bins = 7
        head = ReferenceRelationHead(feature_dim=4, bins=bins, reference_normalization="global").double()
        features = torch.randn(2, bins, 4, dtype=torch.float64)
        output = forward_tokens(head, features)
        ua, ep, eb = raw_scores(head, features)
        phases = torch.tensor([[.13, 1.17, 4.31], [5.93, .71, 3.42]], dtype=torch.float64)
        qa, qp, qb = circular_two_hot_targets(phases, bins).unbind(1)
        lp, lb = absolute_log_conditionals(output)
        role_nll = -(qa*output["log_A"]).sum(-1)
        role_nll -= (qa[:, :, None]*qp[:, None]*lp).sum((1, 2))
        role_nll -= (qa[:, :, None]*qb[:, None]*lb).sum((1, 2))
        energy = ua[:, :, None, None]+ep[:, :, :, None]+eb[:, :, None, :]
        label = qa[:, :, None, None]*qp[:, None, :, None]*qb[:, None, None, :]
        direct_nll = -(label*energy.flatten(1).log_softmax(-1).reshape_as(energy)).sum((1, 2, 3))
        cancelled = (ua+ep.logsumexp(-1)+eb.logsumexp(-1)).logsumexp(-1)-(qa*ua).sum(-1)
        cancelled -= (qa[:, :, None]*qp[:, None]*ep).sum((1, 2))
        cancelled -= (qa[:, :, None]*qb[:, None]*eb).sum((1, 2))
        torch.testing.assert_close(role_nll, direct_nll, atol=1e-12, rtol=0)
        torch.testing.assert_close(role_nll, cancelled, atol=1e-12, rtol=0)

    def test_matched_modes_keep_parameters_and_conditionals_identical(self):
        local = ReferenceRelationHead(feature_dim=5, bins=9, reference_normalization="local").double()
        global_head = ReferenceRelationHead(feature_dim=5, bins=9, reference_normalization="global").double()
        global_head.load_state_dict(local.state_dict(), strict=True)
        self.assertEqual(set(local.state_dict()), set(global_head.state_dict()))
        self.assertEqual(sum(p.numel() for p in local.parameters()), sum(p.numel() for p in global_head.parameters()))
        for key, value in local.state_dict().items():
            self.assertTrue(torch.equal(value, global_head.state_dict()[key]), key)
        features = torch.randn(2, 9, 5, dtype=torch.float64)
        local_output, global_output = forward_tokens(local, features), forward_tokens(global_head, features)
        for key in ("P_rel", "B_rel", "log_P_rel", "log_B_rel", "reference_unary", "log_Z_P", "log_Z_B"):
            self.assertTrue(torch.equal(local_output[key], global_output[key]), key)
        torch.testing.assert_close(local_output["log_A"], local_output["reference_unary"].log_softmax(-1), atol=0, rtol=0)
        self.assertGreater(float((local_output["A"]-global_output["A"]).abs().max().detach()), 1e-6)

    def test_circular_roll_preserves_joint_and_full_mass_readout(self):
        bins, shift = 11, 3
        head = ReferenceRelationHead(feature_dim=4, bins=bins, reference_normalization="global").double()
        features = torch.randn(2, bins, 4, dtype=torch.float64)
        original, rolled = forward_tokens(head, features), forward_tokens(head, features.roll(shift, 1))
        torch.testing.assert_close(rolled["A"], original["A"].roll(shift, 1), atol=1e-13, rtol=0)
        for key in ("P_rel", "B_rel"):
            torch.testing.assert_close(rolled[key], original[key].roll(shift, 1), atol=1e-13, rtol=0)
        torch.testing.assert_close(rolled["J"], original["J"], atol=1e-13, rtol=0)
        pushforward = CircularProgressPushforward(bins=bins, progress_bins=17)
        native, shifted = pushforward(original["J"]), pushforward(rolled["J"])
        torch.testing.assert_close(native["distribution"].sum(1), torch.ones(2), atol=3e-7, rtol=0)
        self.assertTrue(bool((native["invalid_mass"] > 0).all()))
        for key in ("distribution", "reading", "invalid_mass", "outside_arc_mass"):
            torch.testing.assert_close(shifted[key], native[key], atol=2e-7, rtol=0)

    def test_row_offsets_change_global_reference_without_changing_conditionals(self):
        ua = torch.zeros(1, 4, dtype=torch.float64)
        ep = torch.tensor([[[.2, -.1, .3, .4]]], dtype=torch.float64).expand(1, 4, 4)
        eb = torch.tensor([[[.1, .3, -.2, .5]]], dtype=torch.float64).expand(1, 4, 4)
        row_offsets = torch.tensor([[[0.], [2.3], [0.], [0.]]], dtype=torch.float64)
        shifted_ep = ep+row_offsets
        torch.testing.assert_close(ep.log_softmax(-1), shifted_ep.log_softmax(-1), atol=1e-15, rtol=0)
        before = (ua+ep.logsumexp(-1)+eb.logsumexp(-1)).softmax(-1)
        after = (ua+shifted_ep.logsumexp(-1)+eb.logsumexp(-1)).softmax(-1)
        torch.testing.assert_close(before, torch.full_like(before, .25), atol=1e-15, rtol=0)
        expected = torch.tensor(2.3, dtype=torch.float64).exp()/(3+torch.tensor(2.3, dtype=torch.float64).exp())
        torch.testing.assert_close(after[0, 1], expected, atol=1e-15, rtol=0)
        self.assertGreater(float(after[0, 1]-before[0, 1]), .5)
        REPORT["row_offset_fixtures"] += 1

    def test_global_role_nll_reaches_all_projection_and_unary_parameters(self):
        bins = 9
        head = ReferenceRelationHead(feature_dim=5, bins=bins, reference_normalization="global").double()
        features = torch.randn(2, bins, 5, dtype=torch.float64, requires_grad=True)
        output = forward_tokens(head, features)
        phases = torch.tensor([[.21, 1.07, 4.11], [5.73, .81, 3.12]], dtype=torch.float64)
        qa, qp, qb = circular_two_hot_targets(phases, bins).unbind(1)
        lp, lb = absolute_log_conditionals(output)
        nll = -(qa*output["log_A"]).sum(-1)
        nll -= (qa[:, :, None]*qp[:, None]*lp).sum((1, 2))
        nll -= (qa[:, :, None]*qb[:, None]*lb).sum((1, 2))
        loss = nll.mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        REPORT["component_backward_calls"] += 1
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        self.assertGreater(float(features.grad.abs().sum()), 0.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceFeedbackTests))
    REPORT.update(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), torch=torch.__version__, process_pid=os.getpid(),
                  wall_seconds=time.perf_counter()-started, cpu_threads=torch.get_num_threads(),
                  CNN_image_forwards=0, source_RGB_image_forwards=0, checkpoint_loads=0,
                  optimizer_updates=0, gpu_calls=0)
    args.output.write_text(json.dumps(REPORT, indent=2)+"\n")
    print(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
