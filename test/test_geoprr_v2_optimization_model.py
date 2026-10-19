from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_model import BACKBONES, GeoPRRV2, _DynamicRefine


CHANNELS = (8, 12, 16, 24)
REDUCTIONS = (4, 8, 16, 32)


class _FeatureInfo:
    def reduction(self) -> list[int]:
        return list(REDUCTIONS)

    def channels(self) -> list[int]:
        return list(CHANNELS)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_info = _FeatureInfo()
        self.projections = nn.ModuleList(nn.Conv2d(3, c, 1) for c in CHANNELS)

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        return [
            projection(F.avg_pool2d(images, stride))
            for projection, stride in zip(self.projections, REDUCTIONS)
        ]


def _model(**kwargs: object) -> GeoPRRV2:
    timm = types.SimpleNamespace(create_model=lambda *args, **kw: _Backbone())
    with patch.dict(sys.modules, {"timm": timm}):
        return GeoPRRV2(
            BACKBONES[1], pretrained=False, bins=8, decoder_dim=8, **kwargs
        )


def _legacy_state() -> dict[str, torch.Tensor]:
    """The original field checkpoint parameter names and tensor dimensions."""
    state = {
        "bin_coordinates": torch.linspace(0, 1, 8),
        "appearance_norm.weight": torch.ones(24),
        "appearance_norm.bias": torch.zeros(24),
        "appearance_head.weight": torch.randn(8, 24),
        "appearance_head.bias": torch.randn(8),
    }
    for level, channels in enumerate(CHANNELS):
        state.update({
            f"backbone.projections.{level}.weight": torch.randn(channels, 3, 1, 1),
            f"backbone.projections.{level}.bias": torch.randn(channels),
            f"lateral.{level}.weight": torch.randn(8, channels, 1, 1),
            f"lateral.{level}.bias": torch.randn(8),
            f"refine.{level}.0.weight": torch.randn(8, 8, 3, 3),
            f"refine.{level}.1.weight": torch.ones(8),
            f"refine.{level}.1.bias": torch.zeros(8),
        })
    for head, channels in (("field_head", 8), ("point_head", 1), ("support_head", 1)):
        state[f"{head}.weight"] = torch.randn(channels, 8, 1, 1)
        state[f"{head}.bias"] = torch.randn(channels)
    return state


class GeoPRRV2OptimizationModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)

    def test_default_accepts_original_field_checkpoint_strictly(self) -> None:
        model = _model().eval()
        model.load_state_dict(_legacy_state(), strict=True)
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertIsNone(outputs["coarse_point_logits"])
        self.assertEqual(outputs["global_features"].shape, (2, 24))
        self.assertTrue(torch.isfinite(outputs["reading"]).all())

    def test_spatial_raw_head_needs_no_geometry_modules_and_learns(self) -> None:
        model = _model(variant="raw_distribution", appearance_pooling="spatial")
        for name in ("lateral", "refine", "field_head", "point_head", "support_head"):
            self.assertFalse(hasattr(model, name))
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertIsNone(outputs["field_logits"])
        self.assertIsNone(outputs["coarse_point_logits"])
        self.assertLess(float(outputs["appearance_probs"].max()), 0.25)
        F.cross_entropy(outputs["appearance_logits"], torch.tensor([1, 6])).backward()
        for projection, attention in zip(
            model.appearance_head.projections, model.appearance_head.attention
        ):
            self.assertGreater(float(projection[0].weight.grad.abs().sum()), 0)
            self.assertGreater(float(attention.weight.grad.abs().sum()), 0)

    def test_spatial_expert_is_independent_of_geometry_predictions(self) -> None:
        model = _model(appearance_pooling="spatial").eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            before = model(images)
            for head in (model.field_head, model.point_head, model.support_head):
                head.weight.mul_(-7)
                head.bias.add_(5)
            after = model(images)
        torch.testing.assert_close(before["appearance_probs"], after["appearance_probs"])
        self.assertFalse(torch.allclose(before["field_probs"], after["field_probs"]))

    def test_coarse_and_fine_point_heads_receive_localization_gradients(self) -> None:
        model = _model(detail_refinement=True)
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(outputs["point_logits"].shape, (2, 1, 16, 16))
        self.assertEqual(outputs["coarse_point_logits"].shape, (2, 1, 4, 4))
        fine_loss = -outputs["point_logits"].flatten(1).log_softmax(1)[:, 39].mean()
        coarse_gradient = torch.autograd.grad(
            fine_loss, outputs["coarse_point_logits"], retain_graph=True
        )[0]
        self.assertGreater(float(coarse_gradient.abs().sum()), 0)
        coarse_loss = -outputs["coarse_point_logits"].flatten(1).log_softmax(1)[:, 6].mean()
        (fine_loss + coarse_loss).backward()
        for parameter in (
            model.coarse_point_head.weight,
            model.point_head[0].weight,
            model.point_head[-1].weight,
            model.lateral[0].weight,
        ):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_dynamic_kernels_depend_on_input_and_backpropagate(self) -> None:
        module = _DynamicRefine(8)
        features = torch.randn(2, 8, 6, 6)
        weights = module.kernel_weights(features)
        self.assertFalse(torch.allclose(weights, module.kernel_weights(features + 2)))
        output = module(features)
        individual = []
        for row, mixture in zip(features, weights):
            kernel = (mixture[:, None, None, None, None] * module.kernels).sum(0)
            individual.append(F.conv2d(row[None], kernel, padding=1))
        expected = F.gelu(module.norm(torch.cat(individual)))
        torch.testing.assert_close(output, expected)
        output[:, 0, 0, 0].sum().backward()
        self.assertGreater(float(module.router.weight.grad.abs().sum()), 0)
        self.assertGreater(float(module.kernels.grad.abs().sum()), 0)
        model = _model(decoder_kernel="dynamic")
        self.assertIsInstance(model.refine[2], _DynamicRefine)
        self.assertIsInstance(model.refine[3], _DynamicRefine)
        self.assertNotIsInstance(model.refine[0], _DynamicRefine)
        self.assertNotIsInstance(model.refine[1], _DynamicRefine)


if __name__ == "__main__":
    unittest.main()
