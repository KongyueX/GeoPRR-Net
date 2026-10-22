"""Focused CPU checks on generated features; no data, weights, or GPU access."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.geoprr_v2_generalization_style import (
    _mix_style, freeze_backbone_bn_statistics, install_mixstyle_hooks,
)


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Identity(), nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.head = nn.BatchNorm2d(2)

    def forward(self, images):
        return self.head(self.backbone(images))


class GeneralizationStyleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262023)
        self.features = torch.tensor([
            [[[0., 1.], [2., 3.]], [[2., 4.], [6., 8.]]],
            [[[10., 13.], [16., 19.]], [[-8., -4.], [0., 4.]]],
        ])

    def test_detached_statistics_mixture_and_gradient(self):
        features = self.features.clone().requires_grad_()
        lam = torch.tensor([.25, .75]).reshape(2, 1, 1, 1)
        permutation = torch.tensor([1, 0])
        result = _mix_style(features, lam, permutation)
        mean = features.detach().mean((2, 3), keepdim=True)
        std = (features.detach().var((2, 3), keepdim=True) + 1e-6).sqrt()
        mixed_mean = lam * mean + (1 - lam) * mean[permutation]
        mixed_std = lam * std + (1 - lam) * std[permutation]
        expected = (features.detach() - mean) / std * mixed_std + mixed_mean
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(result.mean((2, 3), keepdim=True), mixed_mean)
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        torch.testing.assert_close(features.grad, (mixed_std / std).expand_as(features))

    def test_backbone_bn_statistics_only_and_affine_gradients(self):
        model = TinyReader().train()
        trainable = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        running = {name: value.clone() for name, value in model.backbone[2].state_dict().items()}
        freeze_backbone_bn_statistics(model)
        self.assertTrue(model.training)
        self.assertTrue(model.backbone.training)
        self.assertFalse(model.backbone[2].training)
        self.assertTrue(model.head.training)
        self.assertEqual(trainable, {name: parameter.requires_grad for name, parameter in model.named_parameters()})
        model(self.features).square().sum().backward()
        for name in ("weight", "bias"):
            gradient = getattr(model.backbone[2], name).grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            torch.testing.assert_close(model.backbone[2].state_dict()[name], running[name], atol=0, rtol=0)

    def test_hooks_mix_before_next_backbone_block_and_are_removable(self):
        model = TinyReader().train()
        freeze_backbone_bn_statistics(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        handles = install_mixstyle_hooks(model, ("0",), p=1)
        self.assertEqual(set(before), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        received = []
        observer = model.backbone[1].register_forward_pre_hook(lambda _module, inputs: received.append(inputs[0]))
        lam = torch.full((2, 1, 1, 1), .25)
        permutation = torch.tensor([1, 0])
        with patch("torch.distributions.Beta.sample", return_value=lam), patch("torch.randperm", return_value=permutation):
            model(self.features)
        torch.testing.assert_close(received[-1], _mix_style(self.features, lam, permutation), atol=0, rtol=0)
        handles[0].remove()
        model(self.features)
        self.assertIs(received[-1], self.features)
        observer.remove()

    def test_evaluation_zero_probability_and_frozen_backbone_are_identity(self):
        for mode in ("evaluation", "zero_probability", "frozen_backbone", "frozen_block"):
            with self.subTest(mode=mode):
                model = TinyReader().eval()
                baseline = model(self.features)
                handles = install_mixstyle_hooks(model, ("0",), p=0 if mode == "zero_probability" else 1)
                if mode != "evaluation":
                    model.train()
                    freeze_backbone_bn_statistics(model)
                    model.head.eval()
                    if mode == "frozen_backbone":
                        model.backbone.eval()
                    elif mode == "frozen_block":
                        model.backbone[0].eval()
                with patch("torch.distributions.Beta.sample", side_effect=AssertionError("identity path sampled style")):
                    torch.testing.assert_close(model(self.features), baseline, atol=0, rtol=0)
                for handle in handles:
                    handle.remove()

    def test_invalid_targets_fail_before_installing_any_hook(self):
        model = TinyReader()
        for names in ((), ("",), ("missing",), ("0", "0"), ("0", "missing")):
            with self.subTest(names=names), self.assertRaises(ValueError):
                install_mixstyle_hooks(model, names)
            self.assertTrue(all(not module._forward_hooks for module in model.backbone.modules()))


if __name__ == "__main__":
    unittest.main()
