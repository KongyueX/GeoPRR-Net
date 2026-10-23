"""Focused CPU checks for the direct scalar readout and shared native features."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import torch
from torch import nn

from experiments.geoprr_v2_direct_readout import (
    CONFIG, PROTOCOL, DirectScalarHead, FrozenFeatureCapture, fit_head, readout_rows, snapshot_state, verify_stationary,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return [self.block(image)]


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.feature_indices = (0,)

    def forward(self, image):
        features = self.backbone(image)[0]
        return {"reading": features.mean((1, 2, 3)).sigmoid(), "native_tensor": features[:, :1]}


class DirectReadoutTests(unittest.TestCase):
    def test_standardization_uses_source_features_only_and_checks_shape(self):
        source = torch.tensor([[1., 4., 2.], [3., 4., 6.]])
        validation = source + 100
        head = DirectScalarHead.from_source_features(source)
        self.assertTrue(torch.equal(head.mean, source.mean(0)))
        self.assertTrue(torch.equal(head.std, torch.tensor([1., 1e-6, 2.])))
        self.assertFalse(torch.equal(head.mean, validation.mean(0)))
        self.assertEqual(head(source).shape, (2,))
        self.assertTrue(torch.equal(head(source), torch.full((2,), .5)))
        with self.assertRaises(ValueError):
            DirectScalarHead.from_source_features(torch.ones(2, 3, 1))

    def test_head_gradients_bounded_point_estimate_and_fixed_half_failure_arithmetic(self):
        torch.manual_seed(20262020)
        native = TinyReader().eval().requires_grad_(False)
        capture = FrozenFeatureCapture(native)
        with torch.no_grad():
            _output, features = capture(torch.randn(2, 3, 4, 4))
        capture.close()
        head = DirectScalarHead.from_source_features(features)
        loss = (head(features) - torch.tensor([0., .2])).abs().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in native.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters()))
        prediction = head(features)
        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))
        geometry = [{"sample_id": "a", "group_id": "g", "target": .1, "prediction": .2, "valid": True,
                     "input_valid": True, "invalid_mass": .1, "outside_arc_mass": .1, "failure": ""},
                    {"sample_id": "b", "group_id": "g", "target": .8, "prediction": None, "valid": False,
                     "input_valid": True, "invalid_mass": None, "outside_arc_mass": None, "failure": "invalid"}]
        rows = readout_rows(geometry, torch.tensor([.6, .4]))
        self.assertAlmostEqual(rows["fixed_half"][0]["prediction"], .4, places=7)
        self.assertTrue(rows["direct"][1]["valid"])
        self.assertFalse(rows["fixed_half"][1]["valid"])
        self.assertEqual(rows["fixed_half"][1]["absolute_error"], 1.)

    def test_feature_hook_shares_one_cnn_pass_without_mutating_native_output_or_state(self):
        torch.manual_seed(20262020)
        native = TinyReader().eval().requires_grad_(False)
        image = torch.randn(2, 3, 4, 4)
        original = snapshot_state(native)
        with torch.inference_mode():
            expected = native(image)
            capture = FrozenFeatureCapture(native)
            actual, feature = capture(image)
            capture.close()
        self.assertEqual(capture.calls, 1)
        self.assertEqual(capture.images, 2)
        self.assertEqual(native.backbone.calls, 2)
        self.assertEqual(feature.shape, (2, 4))
        self.assertEqual(set(actual), set(expected))
        self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in actual))
        self.assertTrue(verify_stationary(native, original)["all_native_state_tensors_unchanged"])

    def test_terminal_head_resume_completes_without_additional_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, output = root / "cache", root / "head"
            cache.mkdir()
            output.mkdir()
            parent = {"fixture": "small-dimensional source cache; no native CNN"}
            for split, count in (("train", 18590), ("validation", 1506)):
                torch.save({"protocol": PROTOCOL, "config": CONFIG, "parent": parent,
                            "features": torch.zeros(count, 2),
                            "rows": [{"input_valid": True, "target": .25}] * count}, cache / f"{split}.pt")
            head = DirectScalarHead.from_source_features(torch.zeros(2, 2))
            optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=.00001)
            torch.save({"protocol": PROTOCOL, "config": CONFIG, "parent": parent, "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": 20, "best_epoch": 1, "best_nmae": .1, "history": [],
                        "torch_rng": torch.get_rng_state(),
                        "counts": {"head_optimizer_updates": 1460, "source_feature_presentations": 371800}},
                       output / "latest.pt")
            calls = []

            def unexpected_step(_optimizer, *args, **kwargs):
                calls.append(True)
                raise AssertionError("terminal resume must not update")

            with patch.object(torch.optim.AdamW, "step", new=unexpected_step):
                fit_head(cache, output)
            self.assertEqual(calls, [])
            completed = json.loads((output / "completed.json").read_text())
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["counts"]["head_optimizer_updates"], 1460)
            self.assertEqual(completed["counts"]["head_backward_calls"], 1460)
            self.assertEqual(completed["counts"]["source_feature_presentations"], 371800)
            self.assertEqual(completed["counts"]["cnn_forward_calls"], 0)


if __name__ == "__main__":
    unittest.main()
