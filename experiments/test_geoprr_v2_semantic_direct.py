"""Focused CPU checks for the full semantic scalar model and declared RGB inputs."""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.geoprr_v2_semantic_direct import ScalarRGBDataset, SemanticDirectReader, learning_rate, scalar_rows


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.body = nn.Conv2d(3, 4, 1)
        self.conv_head = nn.Conv2d(4, 5, 1)
        self.norm_head = nn.BatchNorm2d(5)

    def forward(self, image):
        return self.norm_head(self.conv_head(self.body(image))).mean((2, 3))


class SemanticDirectTests(unittest.TestCase):
    def test_complete_semantic_neck_retained_and_point_shape_gradients_finite(self):
        torch.manual_seed(20262020)
        encoder = TinyEncoder()
        original = {key: value.clone() for key, value in encoder.state_dict().items()}
        model = SemanticDirectReader(encoder)
        self.assertIs(model.encoder, encoder)
        self.assertEqual(model.point.in_features, 5)
        self.assertNotEqual(model.point.in_features, encoder.num_features)
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.encoder.state_dict().items()))
        self.assertTrue(any(key.startswith("encoder.conv_head.") for key in model.state_dict()))
        self.assertTrue(any(key.startswith("encoder.norm_head.") for key in model.state_dict()))
        prediction = model(torch.randn(3, 3, 4, 4))
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))
        torch.nn.functional.smooth_l1_loss(prediction, torch.tensor([.1, .3, .8]), beta=.05).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        self.assertTrue(any(bool(parameter.grad.abs().sum() > 0) for parameter in model.encoder.parameters()))
        self.assertTrue(bool(model.point.weight.grad.abs().sum() > 0))

    def test_source_wrapper_exposes_rgb_scalar_ids_without_geometry_targets(self):
        observed = {}

        class FakeBase:
            def __init__(self, manifest, **kwargs):
                observed.update(manifest=manifest, **kwargs)
                self.rows = [{"sample_id": "a", "group_id": "g"}]

            def __getitem__(self, index):
                return {"image": torch.zeros(3, 256, 256), "y": torch.tensor(.25), "sample_id": "a", "group_id": "g",
                        "roi_to_canonical": object(), "field_target": object(), "point_target": object()}

        with patch("experiments.geoprr_v2_semantic_direct.GeoPRRV2Dataset", FakeBase):
            dataset = ScalarRGBDataset("source/manifests/train.jsonl", data_root="source", augment=True)
            sample = dataset[0]
        self.assertEqual(set(sample), {"image", "y", "sample_id", "group_id", "input_failure"})
        self.assertEqual(observed["manifest"], "source/manifests/train.jsonl")
        self.assertFalse(observed["require_geometry"])
        self.assertIsNone(observed["pair_manifest"])
        self.assertEqual((observed["image_size"], observed["output_size"]), (256, 64))
        self.assertTrue(observed["augment"])

    def test_learning_rate_boundaries_match_two_explicit_cosine_phases(self):
        expected = {1: 3e-4, 30: 3e-4 * .5 * (1 + math.cos(29 * math.pi / 30)),
                    31: 3e-5, 60: 3e-5 * .5 * (1 + math.cos(29 * math.pi / 30))}
        for epoch, value in expected.items():
            self.assertAlmostEqual(learning_rate(epoch), value, places=15)
        with self.assertRaises(ValueError):
            learning_rate(61)

    def test_scalar_rows_retain_full_population_and_assign_failure_error_one(self):
        batch = {"sample_id": ["a", "b", "c", "d"], "group_id": ["g"] * 4,
                 "y": torch.tensor([.1, .2, .3, .4]), "input_failure": ["", "missing RGB", "", ""]}
        rows = scalar_rows(torch.tensor([.2, .1, float("nan"), 1.1]), batch)
        self.assertEqual([row["sample_id"] for row in rows], batch["sample_id"])
        self.assertTrue(rows[0]["valid"])
        self.assertAlmostEqual(rows[0]["absolute_error"], .1, places=7)
        self.assertEqual([row["absolute_error"] for row in rows[1:]], [1., 1., 1.])
        self.assertEqual([row["prediction"] for row in rows[1:]], [None, None, None])
        self.assertFalse(rows[1]["input_valid"])
        self.assertTrue(rows[2]["input_valid"])


if __name__ == "__main__":
    unittest.main()
