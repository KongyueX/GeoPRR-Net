"""New CPU posterior/role-loss checks; no RGB or backbone execution."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.geoprr_v2_reference_relation import ReferenceRelationHead
from experiments.geoprr_v2_role_triangle import (
    RoleTriangleReader, pair_log_factor, role_triangle_losses, triangle_contract,
)


REPORT = {}


class TriangleReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)

    @classmethod
    def tearDownClass(cls):
        print(json.dumps({"CPU_integration_tests": 2, "CNN_image_forwards": 0, "GPU_calls": 0,
                          "optimizer_updates": 0, **REPORT}), flush=True)

    def test_streamed_reference_posterior_and_joint_role_ce(self):
        """A must be the true triangle marginal; CE must include the factor correction."""
        bins = 4
        head = ReferenceRelationHead(16, bins, reference_normalization="global").double()
        tokens = torch.randn(2, bins, 16, dtype=torch.float64)
        state = head(tokens)
        h_p, h_b = head.projections[1](tokens), head.projections[2](tokens)
        w = (torch.randn(16, 16, dtype=torch.float64)/4).requires_grad_()
        log_factor = pair_log_factor(h_p, h_b, w)
        joint, log_z, reference = triangle_contract(state["log_A"], state["log_P_rel"], state["log_B_rel"],
                                                   log_factor, tile=3, return_reference=True)
        index = (torch.arange(bins)[:, None]+torch.arange(bins)[None]) % bins
        full_log = (state["log_A"][:, :, None, None]+state["log_P_rel"][:, :, :, None]
                    +state["log_B_rel"][:, :, None, :]+log_factor[:, index[:, :, None], index[:, None, :]])
        expected_z = full_log.flatten(1).logsumexp(-1)
        full = (full_log-expected_z[:, None, None, None]).exp()
        torch.testing.assert_close(reference, full.sum((2, 3)), rtol=2e-11, atol=2e-12)
        torch.testing.assert_close(joint, full.sum(1), rtol=2e-11, atol=2e-12)
        labels = torch.randn(2, 3, bins, dtype=torch.float64).softmax(-1)
        q = labels[:, 0, :, None, None]*labels[:, 1, index][:, :, :, None]*labels[:, 2, index][:, :, None, :]
        star_log = state["log_A"][:, :, None, None]+state["log_P_rel"][:, :, :, None]+state["log_B_rel"][:, :, None, :]
        base_ce = -(q*star_log).sum((1, 2, 3))
        correction = log_z-torch.einsum("bp,bq,bpq->b", labels[:, 1], labels[:, 2], log_factor)
        expected_ce = -(q*(full_log-expected_z[:, None, None, None])).sum((1, 2, 3))
        torch.testing.assert_close(base_ce+correction, expected_ce, rtol=2e-11, atol=2e-12)
        actual_gradient = torch.autograd.grad((reference*labels[:, 0]).sum()+correction.sum(), w, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad((full.sum((2, 3))*labels[:, 0]).sum()
                                               +(expected_ce-base_ce).sum(), w)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-10, atol=2e-12)
        REPORT.update(feature_head_image_calls=2, triangle_contract_calls=1, gradient_queries=2)

    def test_reader_initialization_and_loss_uses_star_base(self):
        """Only Wpb is new and warmup-trainable; posterior A cannot replace star CE."""
        model = RoleTriangleReader(association="full", tile=16)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7461316)
        self.assertEqual(len(model.state_dict()), 497)
        source_state = {key: value.clone() for key, value in model.state_dict().items() if key != "state_head.Wpb"}
        source = {"epoch": 48, "protocol": "geoprr_v2_reference_feedback_source_only_v1",
                  "config": {"training": {"seed": 20262020}, "model": {"reference_normalization": "global"}},
                  "model": source_state}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"best.pt"
            torch.save(source, path)
            metadata = model.initialize_full(path)
        self.assertEqual(metadata["loaded_full_tensors"], 496)
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in source_state.items()))
        self.assertEqual(float(model.state_head.Wpb.detach().abs().sum()), 0.)
        model.set_shared_trainable(False)
        self.assertTrue(model.state_head.Wpb.requires_grad)
        self.assertTrue(any(p is model.state_head.Wpb for p in model.state_head.parameters()))
        self.assertFalse(any(p is model.state_head.Wpb for p in model.shared_parameters()))
        bins = 256
        output = {"star_log_A": torch.randn(1, bins).log_softmax(-1), "log_A": torch.zeros(1, bins),
                  "log_F_abs": torch.full((1, bins, bins), -2*math.log(bins)),
                  "log_Z_correction": torch.tensor([-2*math.log(bins)])}
        labels = torch.zeros(1, 3, bins)
        labels[:, :, 0] = 1
        base = {"role_nll_raw": torch.tensor(1.7), "role_nll_normalized": torch.tensor(0.),
                "field_l1": torch.tensor(.2), "reading_cdf": torch.tensor(.3), "total": torch.tensor(0.)}
        with patch("experiments.geoprr_v2_role_triangle.reference_relation_losses", return_value=base) as old_loss, \
                patch("experiments.geoprr_v2_role_triangle.reference_geometry_targets", return_value={"role_twohot": labels}):
            losses = role_triangle_losses(output, {"roi_to_canonical": torch.eye(3)[None], "y": torch.tensor([.4]),
                                                   "span_radians": torch.tensor([math.pi]), "image": torch.zeros(1, 3, 4, 4)})
        self.assertIs(old_loss.call_args.args[0]["log_A"], output["star_log_A"])
        torch.testing.assert_close(losses["role_nll_raw"], base["role_nll_raw"])
        torch.testing.assert_close(losses["total"], base["role_nll_raw"]/(3*math.log(bins))+.5)
        REPORT.update(parameters=7461316, state_tensors=497, loaded_old_tensors=496, zero_new_parameters=256)


if __name__ == "__main__":
    unittest.main()
