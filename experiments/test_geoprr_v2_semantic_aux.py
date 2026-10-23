"""Focused CPU checks for direct deployment and source-only structural loss."""
import math
import unittest

import torch
from torch import nn

from experiments.geoprr_v2_reference_relation import reference_relation_losses
from experiments.geoprr_v2_semantic_aux import SemanticAuxReader, auxiliary_losses
from experiments.geoprr_v2_semantic_direct import SemanticDirectReader


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 960
        self.blocks = nn.ModuleList((nn.Conv2d(3, 48, 1), nn.Conv2d(48, 80, 1, stride=2),
                                     nn.Conv2d(80, 160, 1, stride=2), nn.Identity(), nn.Conv2d(160, 960, 1, stride=2)))
        self.conv_head = nn.Conv2d(960, 5, 1)
        self.norm_head = nn.GroupNorm(1, 5)

    def forward(self, image):
        for block in self.blocks:
            image = block(image)
        return self.norm_head(self.conv_head(image)).mean((2, 3))


def batch_for(image):
    count, size = len(image), image.shape[-1]
    homography = torch.tensor([[1 / size, 0, -.5 + .5 / size], [0, 1 / size, -.5 + .5 / size], [0, 0, 1.]])
    return {"image": image, "y": torch.linspace(.2, .8, count),
            "roi_to_canonical": homography[None].repeat(count, 1, 1),
            "span_radians": torch.full((count,), 1.5 * math.pi),
            "field_target": torch.linspace(0, 1, size * size).reshape(1, size, size).repeat(count, 1, 1),
            "field_mask": torch.ones(count, 1, size, size)}


class SemanticAuxTests(unittest.TestCase):
    def test_direct_prediction_retains_base_and_does_not_execute_auxiliary_modules(self):
        torch.manual_seed(20262020)
        base = SemanticDirectReader(TinyEncoder()).eval()
        image = torch.randn(2, 3, 8, 8)
        expected = base(image).detach()
        original = {key: value.clone() for key, value in base.state_dict().items()}
        model = SemanticAuxReader(base).eval()
        calls = {"encoder": 0, "auxiliary": 0}
        encoder_hook = model.encoder.register_forward_hook(lambda *_: calls.__setitem__("encoder", calls["encoder"] + 1))
        auxiliary_hook = model.state_head.register_forward_hook(lambda *_: calls.__setitem__("auxiliary", calls["auxiliary"] + 1))
        try:
            prediction = model(image)
        finally:
            encoder_hook.remove()
            auxiliary_hook.remove()
        self.assertTrue(torch.equal(prediction, expected))
        self.assertEqual(calls, {"encoder": 1, "auxiliary": 0})
        self.assertIsNone(model._features)
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in original.items()))

    def test_training_has_one_encoder_pass_multiscale_gradients_and_strict_reload(self):
        torch.manual_seed(12)
        model = SemanticAuxReader(SemanticDirectReader(TinyEncoder())).train()
        image = torch.randn(2, 3, 8, 8)
        observed = []
        hooks = [module.register_forward_hook(lambda _m, inputs, _o: observed.append(tuple(inputs[0].shape)))
                 for module in model.lateral]
        calls = []
        hooks.append(model.encoder.register_forward_hook(lambda *_: calls.append(1)))
        try:
            prediction, auxiliary = model.forward_training(image)
            losses = auxiliary_losses(prediction, auxiliary, batch_for(image))
            losses["total"].backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(observed, [(2, 48, 8, 8), (2, 80, 4, 4), (2, 160, 2, 2), (2, 960, 1, 1)])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(model._features)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
        for module in (model.encoder, model.point, model.lateral, model.refine, model.state_head):
            self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters()))
            self.assertTrue(any(bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters()))
        restored = SemanticAuxReader(SemanticDirectReader(TinyEncoder()))
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval(), restored.eval()
        self.assertTrue(torch.equal(model(image), restored(image)))

    def test_conditional_role_and_field_terms_equal_original_formula_without_reading_cdf(self):
        torch.manual_seed(21)
        model = SemanticAuxReader(SemanticDirectReader(TinyEncoder())).eval()
        image = torch.randn(2, 3, 8, 8)
        prediction, auxiliary = model.forward_training(image)
        batch = batch_for(image)
        losses = auxiliary_losses(prediction, auxiliary, batch)
        self.assertNotIn("distribution", auxiliary)
        original = reference_relation_losses({**auxiliary, "distribution": torch.full((2, 256), 1 / 256)}, batch)
        for key in ("role_nll_raw", "role_nll_normalized", "field_l1"):
            self.assertTrue(torch.equal(losses[key], original[key]))
        expected_primary = torch.nn.functional.smooth_l1_loss(prediction, batch["y"], beta=.05)
        self.assertTrue(torch.equal(losses["primary"], expected_primary))
        self.assertTrue(torch.equal(losses["total"], expected_primary + .1 * (original["role_nll_normalized"] + original["field_l1"])))

    def test_geometry_changes_loss_but_is_never_an_argument_to_rgb_forward(self):
        model = SemanticAuxReader(SemanticDirectReader(TinyEncoder())).eval()
        image = torch.randn(2, 3, 8, 8)
        prediction, auxiliary = model.forward_training(image)
        batch = batch_for(image)
        original = auxiliary_losses(prediction, auxiliary, batch)
        changed = auxiliary_losses(prediction, auxiliary, {**batch, "span_radians": torch.ones(2), "field_target": 1 - batch["field_target"]})
        self.assertTrue(torch.equal(model(image), prediction))
        self.assertTrue(torch.equal(original["primary"], changed["primary"]))
        self.assertFalse(torch.equal(original["auxiliary"], changed["auxiliary"]))
        with self.assertRaises(TypeError):
            model(image, batch["roi_to_canonical"])


if __name__ == "__main__":
    unittest.main()
