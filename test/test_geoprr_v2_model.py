"""Numerical contracts for spatial readout; no gauge performance claims."""
from __future__ import annotations

import importlib.util
import unittest

import torch

from experiments.geoprr_v2_model import (
    BACKBONES,
    GeoPRRV2,
    distribution_cdf_loss,
    masked_field_readout,
    two_hot_targets,
)


class GeoPRRV2ReadoutTests(unittest.TestCase):
    def test_two_hot_preserves_endpoints_and_continuous_means(self) -> None:
        values = torch.tensor([0.0, 1.0, 0.123456, 0.5, 0.999])
        probabilities = two_hot_targets(values, 256)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(values))
        torch.testing.assert_close(
            (probabilities * torch.linspace(0, 1, 256)).sum(-1), values
        )
        self.assertEqual(int((probabilities[0] != 0).sum()), 1)
        self.assertEqual(int((probabilities[1] != 0).sum()), 1)

    def test_ordered_cdf_loss_measures_physical_bin_distance(self) -> None:
        probabilities = two_hot_targets(torch.tensor([0.0, 0.25]), 5)
        losses = distribution_cdf_loss(
            probabilities, torch.tensor([1.0, 0.75]), reduction="none"
        )
        torch.testing.assert_close(losses, torch.tensor([1.0, 0.5]))

    def test_composition_reads_the_selected_state_and_excludes_undefined_field(self) -> None:
        field = torch.tensor([[[[1.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]]]])
        support = torch.tensor([[[[1.0, 1.0, 0.0]]]])
        first = masked_field_readout(
            field, torch.tensor([[[[0.75, 0.25, 0.0]]]]), support, 0.15
        )
        second = masked_field_readout(
            field, torch.tensor([[[[0.0, 0.6, 0.4]]]]), support, 0.15
        )
        torch.testing.assert_close(first["reading"], torch.tensor([0.25]))
        torch.testing.assert_close(second["reading"], torch.tensor([1.0]))
        torch.testing.assert_close(second["support_mass"], torch.tensor([0.6]))
        torch.testing.assert_close(
            second["field_distribution"].sum(1), torch.ones(1)
        )

    def test_invalid_support_falls_back_without_dividing_by_zero(self) -> None:
        field = two_hot_targets(torch.tensor([[[0.2]], [[0.8]]]), 5)
        field = field.permute(0, 3, 1, 2).requires_grad_()
        point = torch.ones(2, 1, 1, 1)
        support = torch.tensor([0.0, 0.5]).view(2, 1, 1, 1).requires_grad_()
        appearance = two_hot_targets(torch.tensor([0.7, 0.3]), 5).requires_grad_()
        result = masked_field_readout(field, point, support, 0.15, appearance)
        self.assertEqual(result["field_valid"].tolist(), [False, True])
        torch.testing.assert_close(result["reading"], torch.tensor([0.7, 0.8]))
        self.assertEqual(float(result["field_distribution"][0].sum()), 0.0)
        self.assertTrue(bool(torch.isfinite(result["distribution"]).all()))
        result["reading"].sum().backward(retain_graph=True)
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertEqual(float(field.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(appearance.grad[0].abs().sum()), 0.0)
        support.grad.zero_()
        result["support_deficit"].sum().backward()
        self.assertLess(float(support.grad[0, 0, 0, 0]), 0.0)

    def test_all_invalid_support_has_finite_fallback_and_gradients(self) -> None:
        field = torch.full((2, 3, 2, 2), 1 / 3, requires_grad=True)
        point = torch.full((2, 1, 2, 2), 0.25)
        support = torch.zeros(2, 1, 2, 2, requires_grad=True)
        appearance_logits = torch.randn(2, 3, requires_grad=True)
        result = masked_field_readout(
            field, point, support, 0.15, appearance_logits.softmax(1)
        )
        result["reading"].sum().backward()
        self.assertFalse(bool(result["field_valid"].any()))
        self.assertTrue(bool(torch.isfinite(appearance_logits.grad).all()))
        self.assertGreater(float(appearance_logits.grad.abs().sum()), 0.0)


@unittest.skipUnless(importlib.util.find_spec("timm"), "timm is required")
class GeoPRRV2BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_both_backbones_use_image_only_stride_four_outputs(self) -> None:
        for backbone in BACKBONES:
            with self.subTest(backbone=backbone):
                model = GeoPRRV2(backbone, pretrained=False, bins=16).eval()
                with torch.no_grad():
                    result = model(torch.randn(2, 3, 64, 64))
                self.assertEqual(result["field_probs"].shape, (2, 16, 16, 16))
                torch.testing.assert_close(
                    result["field_probs"].sum(1), torch.ones(2, 16, 16)
                )
                torch.testing.assert_close(
                    result["point_probs"].sum((1, 2, 3)), torch.ones(2)
                )
                torch.testing.assert_close(
                    result["distribution"].sum(1), torch.ones(2)
                )

    def test_raw_and_multitask_controls_use_appearance_readout(self) -> None:
        images = torch.randn(2, 3, 64, 64)
        for variant in ("raw_scalar", "raw_distribution", "multitask"):
            with self.subTest(variant=variant):
                model = GeoPRRV2(pretrained=False, bins=16, variant=variant).eval()
                with torch.no_grad():
                    result = model(images)
                torch.testing.assert_close(
                    result["reading"], result["appearance_reading"]
                )
                if variant.startswith("raw_"):
                    self.assertFalse(hasattr(model, "field_head"))
                    self.assertIsNone(result["field_logits"])
                else:
                    self.assertIsNotNone(result["field_logits"])

    def test_frozen_backbone_keeps_its_normalization_statistics_frozen(self) -> None:
        model = GeoPRRV2(BACKBONES[1], pretrained=False, bins=16)
        model.set_backbone_trainable(False)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.field_head.training)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        model.set_backbone_trainable(True)
        self.assertTrue(model.backbone.training)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))


if __name__ == "__main__":
    unittest.main()
