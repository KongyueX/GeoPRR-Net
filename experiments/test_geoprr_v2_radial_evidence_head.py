"""Focused CPU checks using generated features, without a backbone or checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import unittest

import torch

from experiments.geoprr_v2_radial_evidence_head import RadialEvidenceHead


PARITY = None
FORWARDS = {"calls": 0, "feature_items": 0}


def run_head(head, features, context):
    FORWARDS["calls"] += 1
    FORWARDS["feature_items"] += features.shape[0]
    return head(features, context)


class RadialEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)
        self.head = RadialEvidenceHead(192, PARITY).cpu()

    def test_pixel_centers_and_ordered_query_indices(self):
        with torch.no_grad():
            self.head.context[-1].weight.zero_()
            self.head.context[-1].bias.zero_()
            self.head.context[-1].bias[5] = math.log(3.)
            self.head.kernel.zero_()
            self.head.kernel[:, 0] = 1.
        width, height = 192, 192
        features = torch.zeros(1, 64, height, width)
        features[:, 0] = (torch.arange(width) + .5)[None] / width
        result = run_head(self.head, features, torch.zeros(1, 192))
        points, calibration = result["query_points"], result["calibration"]
        torch.testing.assert_close(result["logits"], points[..., 0].mean(dim=-1), atol=2e-7, rtol=2e-6)
        delta = points - calibration["center"][:, None, None]
        canonical = torch.einsum("bij,bklj->bkli", calibration["matrix"], delta)
        phase = torch.atan2(canonical[..., 1], canonical[..., 0])
        expected = calibration["span"][:, None, None] * torch.linspace(0., 1., 256)[None, :, None]
        circular_error = torch.atan2((phase-expected).sin(), (phase-expected).cos())
        self.assertLess(float(circular_error.abs().max()), 3e-6)
        self.assertTrue(bool((delta.norm(dim=-1).diff(dim=-1) > 0).all()))

    def test_distribution_shape_mass_and_finite_gradients(self):
        features = torch.randn(2, 64, 24, 32, requires_grad=True)
        context = torch.randn(2, 192, requires_grad=True)
        result = run_head(self.head, features, context)
        probability = result["distribution"]
        self.assertEqual(tuple(probability.shape), (2, 256))
        self.assertEqual(tuple(result["query_grid"].shape), (2, 256, 16, 2))
        torch.testing.assert_close(probability.sum(-1), torch.ones(2), atol=2e-7, rtol=0)
        self.assertTrue(bool((probability > 0).all()))
        self.assertTrue(bool(((result["reading"] >= 0) & (result["reading"] <= 1)).all()))
        self.assertTrue(bool(((result["query_points"] >= 0) & (result["query_points"] <= 1)).all()))
        loss = (probability * torch.linspace(0., 1., 256).square()).sum() + .1 * result["reading"].sum()
        loss.backward()
        for tensor in (features, context, *self.head.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0.)

    def test_zero_features_keep_all_bins_and_mean(self):
        result = run_head(self.head, torch.zeros(1, 64, 20, 20), torch.zeros(1, 192))
        torch.testing.assert_close(result["distribution"], torch.full((1, 256), 1/256), atol=0, rtol=0)
        torch.testing.assert_close(result["reading"], torch.tensor([.5]), atol=1e-7, rtol=0)
        self.assertEqual(float(result["distribution"][0, 0]), 1/256)
        self.assertEqual(float(result["distribution"][0, -1]), 1/256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    PARITY = args.parity
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RadialEvidenceTests))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "fixed_parity": PARITY,
              "untrained_prototype_forward_calls": FORWARDS["calls"],
              "untrained_prototype_feature_items_forwarded": FORWARDS["feature_items"],
              "backbone_image_forwards": 0, "checkpoint_loads": 0, "optimizer_updates": 0,
              "backward_calls": 1, "device": "cpu", "gpu_calls": 0,
              "parameters": sum(p.numel() for p in RadialEvidenceHead(192, PARITY).parameters()),
              "torch": torch.__version__, "wall_seconds": time.perf_counter()-started,
              "process_pid": os.getpid(), "scope": "generated-feature interface tests; no accuracy, novelty, or speed evidence"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
