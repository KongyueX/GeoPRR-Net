"""Focused CPU checks for replacement adapters inside a fixed semantic encoder."""
from copy import deepcopy
import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_deep_adapter import (
    DeepAdapterReader, frozen_encoder_evidence, frozen_reference, gradient_evidence,
    trainable_parameters,
)


class ToyEncoder(nn.Module):
    """Provide the two declared adapter shapes and a fixed downstream semantic neck."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.conv_stem = nn.Conv2d(3, 8, 4, stride=4)
        self.norm_stem = nn.BatchNorm2d(8)
        self.blocks = nn.ModuleList((nn.Identity(), nn.Conv2d(8, 32, 1, stride=2),
                                    nn.Sequential(nn.Conv2d(32, 160, 1, stride=2), nn.BatchNorm2d(160), nn.GELU()),
                                    nn.Conv2d(160, 64, 1, stride=2), nn.Conv2d(64, 960, 1)))
        self.conv_head = nn.Conv2d(960, 1280, 1)
        self.norm_head = nn.BatchNorm2d(1280)
        self.activation = nn.GELU()
        self.num_features = 960

    def forward(self, image):
        self.calls += 1
        feature = self.norm_stem(self.conv_stem(image))
        for block in self.blocks:
            feature = block(feature)
        return self.activation(self.norm_head(self.conv_head(feature))).mean((2, 3))


def activate_adapter_ups(model):
    with torch.no_grad():
        for adapter in model.adapters.values():
            adapter.up.weight.fill_(.01)
            adapter.up.bias.fill_(.05)


class DeepAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_zero_up_preserves_native_reading_and_nonzero_up_changes_downstream_output(self):
        model = DeepAdapterReader(ToyEncoder()).train()
        pristine = ToyEncoder().eval()
        pristine.load_state_dict(model.encoder.state_dict(), strict=True)
        image = torch.randn(1, 3, 256, 256)
        for adapter in model.adapters.values():
            self.assertTrue(bool((adapter.down.bias == 0).all()))
            self.assertTrue(bool((adapter.up.weight == 0).all()))
            self.assertTrue(bool((adapter.up.bias == 0).all()))
        with torch.inference_mode():
            expected = model.point(pristine(image)).squeeze(-1).float().sigmoid()
            identity = model(image)
        torch.testing.assert_close(identity, expected, rtol=0, atol=0)
        self.assertEqual(model.adapter_counts, {key: {"calls":1, "images":1} for key in ("2", "4")})
        activate_adapter_ups(model)
        with torch.inference_mode():
            changed = model(image)
        self.assertGreater(float((identity-changed).abs().max()), 1e-8)

    def test_first_backward_reaches_both_ups_and_point_through_frozen_downstream_chain(self):
        model = DeepAdapterReader(ToyEncoder()).train()
        reference = frozen_reference(model)
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        prediction = model(torch.randn(3, 3, 256, 256))
        F.smooth_l1_loss(prediction, torch.tensor([0., .2, 1.]), beta=.05).backward()
        finite, nonzero, down_zero = gradient_evidence(model)
        self.assertTrue(finite)
        self.assertTrue(all(nonzero.values()), nonzero)
        self.assertTrue(all(down_zero.values()), down_zero)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))
        self.assertTrue(all(torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()))
        self.assertTrue(frozen_encoder_evidence(model, reference)["all_original_state_tensors_unchanged"])
        self.assertEqual(model.last_adapter_shapes, {"2": [3, 160, 16, 16], "4": [3, 960, 8, 8]})
        self.assertEqual(model.last_semantic_shape, [3, 1280])

    def test_down_gradients_become_nonzero_when_independent_toy_ups_are_nonzero(self):
        model = DeepAdapterReader(ToyEncoder()).train()
        activate_adapter_ups(model)
        prediction = model(torch.randn(1, 3, 256, 256))
        F.smooth_l1_loss(prediction, torch.tensor([0.]), beta=.05).backward()
        for adapter in model.adapters.values():
            for parameter in adapter.down.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))

    def test_batch_and_partial_batch_readings_do_not_couple_and_bn_stays_fixed(self):
        model = DeepAdapterReader(ToyEncoder())
        activate_adapter_ups(model)
        reference = frozen_reference(model)
        image = torch.randn(3, 3, 256, 256)
        model.train()
        with torch.inference_mode():
            full = model(image)
            partial = model(image[1:])
            single = model(image[2:])
        torch.testing.assert_close(full[1:], partial, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(full[2:], single, rtol=1e-5, atol=1e-7)
        for mode in (False, True):
            model.train(mode)
            self.assertTrue(all(not module.training for module in model.encoder.modules()))
            self.assertEqual(model.point.training, mode)
            self.assertTrue(all(adapter.training == mode for adapter in model.adapters.values()))
            frozen_encoder_evidence(model, reference)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))

    def test_optimizer_has_only_ten_new_tensors_and_rgb_only_rejects_gt_before_cnn(self):
        model = DeepAdapterReader(ToyEncoder())
        parameters = trainable_parameters(model)
        expected = list(model.point.parameters()) + list(model.adapters.parameters())
        self.assertEqual(len(parameters), 10)
        self.assertEqual({id(parameter) for parameter in parameters}, {id(parameter) for parameter in expected})
        optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001)
        actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual({id(parameter) for parameter in actual}, {id(parameter) for parameter in expected})
        self.assertFalse({id(parameter) for parameter in actual} & {id(parameter) for parameter in model.encoder.parameters()})
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["image"])
        with self.assertRaises(TypeError):
            model(torch.randn(1, 3, 256, 256), ground_truth=torch.tensor([.5]))
        self.assertEqual(model.encoder.calls, 0)

    def test_full_state_strict_reload_keeps_original_keys_and_checks_fresh_reference(self):
        model = DeepAdapterReader(ToyEncoder()).eval()
        reference = frozen_reference(model)
        state = deepcopy(model.state_dict())
        self.assertEqual({key.removeprefix("encoder.") for key in state if key.startswith("encoder.")}, set(model.encoder.state_dict()))
        self.assertEqual(len(state)-len(reference), 10)
        restored = DeepAdapterReader(ToyEncoder()).eval()
        restored.load_state_dict(state, strict=True)
        frozen_encoder_evidence(restored, reference)
        image = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
        missing = dict(state); del missing["adapters.2.up.weight"]
        with self.assertRaises(RuntimeError): restored.load_state_dict(missing, strict=True)
        extra = dict(state); extra["readout.query_embeddings"] = torch.zeros(4, 128)
        with self.assertRaises(RuntimeError): restored.load_state_dict(extra, strict=True)
        changed = deepcopy(state)
        changed["encoder.norm_head.running_mean"][0] += 1
        restored.load_state_dict(changed, strict=True)
        with self.assertRaises(RuntimeError): frozen_encoder_evidence(restored, reference)


if __name__ == "__main__":
    unittest.main()
