"""CPU tensor-contract tests for the fixed half task-anchor interpolation."""
import unittest

import torch
from torch import nn

from experiments.run_geoprr_v2_anchor_interpolation import half_interpolated_state


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.register_buffer("fixed_float", torch.tensor([.25, .5]))
        self.register_buffer("fixed_bool", torch.tensor([True, False]))
        self.register_buffer("fixed_int", torch.tensor([0, 1]))


class AnchorHalfInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyReader()
        self.anchor = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.endpoint = {key: value.clone() for key, value in self.anchor.items()}

    def test_parameters_and_bn_moments_average_but_counter_copies_anchor(self):
        for name, _ in self.model.named_parameters():
            self.anchor[name].fill_(2.)
            self.endpoint[name].fill_(6.)
        self.anchor["backbone.1.running_mean"].fill_(2.)
        self.endpoint["backbone.1.running_mean"].fill_(6.)
        self.anchor["backbone.1.running_var"].fill_(4.)
        self.endpoint["backbone.1.running_var"].fill_(8.)
        self.anchor["backbone.1.num_batches_tracked"].fill_(10)
        self.endpoint["backbone.1.num_batches_tracked"].fill_(20)
        result, counts = half_interpolated_state(self.model, self.anchor, self.endpoint)
        for name, _ in self.model.named_parameters():
            self.assertTrue(torch.equal(result[name], torch.full_like(result[name], 4.)))
        self.assertTrue(torch.equal(result["backbone.1.running_mean"], torch.tensor([4., 4.])))
        self.assertTrue(torch.equal(result["backbone.1.running_var"], torch.tensor([6., 6.])))
        self.assertEqual(int(result["backbone.1.num_batches_tracked"]), 10)
        self.assertEqual(counts["BN_counter_tensors_copied_from_anchor"], 1)
        self.assertEqual(counts["constructor_fixed_tensors_copied_from_anchor"], 3)

    def test_identical_components_preserve_every_state_exactly(self):
        result, _ = half_interpolated_state(self.model, self.anchor, self.endpoint)
        self.assertEqual(set(result), set(self.anchor))
        self.assertTrue(all(torch.equal(result[key], value) for key, value in self.anchor.items()))

    def test_fixed_buffers_dtype_and_invalid_variance_are_rejected(self):
        for key in ("fixed_float", "fixed_bool", "fixed_int"):
            with self.subTest(key=key):
                changed = {name: value.clone() for name, value in self.endpoint.items()}
                changed[key][0] = 0 if key == "fixed_bool" else 99
                with self.assertRaisesRegex(ValueError, "constructor-fixed"):
                    half_interpolated_state(self.model, self.anchor, changed)
        for key, value in (("backbone.1.running_var", torch.tensor([-1., 1.])),
                           ("backbone.0.weight", torch.ones(2, 2, dtype=torch.float64)),
                           ("backbone.1.running_mean", torch.tensor([float("nan"), 0.]))):
            with self.subTest(key=key):
                changed = dict(self.endpoint)
                changed[key] = value
                with self.assertRaises(ValueError):
                    half_interpolated_state(self.model, self.anchor, changed)


if __name__ == "__main__":
    unittest.main()
