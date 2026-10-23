"""Focused CPU checks for image-only semantic-conditioned relation deployment."""
import inspect
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments import geoprr_v2_semantic_aux as auxiliary
from experiments.geoprr_v2_semantic_relation import (
    CONFIG, SemanticRelationReader, auxiliary_losses, evaluate_population, execution_counter, model_metadata,
)


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


class SemanticRelationTests(unittest.TestCase):
    def test_one_encoder_and_deployed_relation_in_both_modes_with_semantic_context_gradient(self):
        torch.manual_seed(20262020)
        model = SemanticRelationReader(TinyEncoder()).train()
        image = torch.randn(2, 3, 8, 8)
        observed_context = []
        context_hook = model.state_head.context[0].register_forward_hook(
            lambda _m, inputs, _output: observed_context.append(inputs[0].detach().clone()))
        counts, hooks = execution_counter(model)
        try:
            prediction, details = model.forward_training(image)
            losses = auxiliary_losses(prediction, details, batch_for(image))
            losses["total"].backward()
            self.assertTrue(torch.equal(observed_context[0], details["global_features"].detach().float()))
            self.assertEqual(observed_context[0].shape, (2, 5))
            self.assertEqual(model.state_head.context[0].in_features, model.encoder.conv_head.out_channels)
            self.assertNotEqual(model.state_head.context[0].in_features, model.encoder.num_features)
            for module in (model.encoder, model.lateral, model.refine, model.state_head):
                self.assertTrue(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                    for parameter in module.parameters()))
                self.assertTrue(any(bool(parameter.grad.abs().sum() > 0) for parameter in module.parameters()))
            model.eval()
            with torch.no_grad():
                deployed = model(image)
            self.assertTrue(torch.equal(prediction, deployed))
        finally:
            context_hook.remove()
            for hook in hooks:
                hook.remove()
        self.assertEqual(counts["cnn_forward_calls"], 2)
        self.assertEqual(counts["cnn_image_forwards"], 4)
        self.assertEqual(counts["relation_forward_calls"], 2)
        self.assertEqual(counts["readout_forward_calls"], 2)
        self.assertIsNone(model._features)
        self.assertFalse(hasattr(model, "point"))
        self.assertTrue(torch.allclose(details["distribution"].sum(-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.equal(details["joint"], details["J"]))
        restored = SemanticRelationReader(TinyEncoder()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored(image), deployed))

    def test_geometry_labels_change_losses_but_never_enter_rgb_forward(self):
        model = SemanticRelationReader(TinyEncoder()).eval()
        image = torch.randn(2, 3, 8, 8)
        prediction, details = model.forward_training(image)
        batch = batch_for(image)
        original = auxiliary_losses(prediction, details, batch)
        changed = auxiliary_losses(prediction, details, {**batch, "span_radians": torch.ones(2),
                                                          "field_target": 1 - batch["field_target"]})
        self.assertTrue(torch.equal(model(image), prediction))
        self.assertTrue(torch.equal(original["primary"], changed["primary"]))
        self.assertFalse(torch.equal(original["auxiliary"], changed["auxiliary"]))
        for method in (model.forward, model.forward_training, model.forward_details):
            self.assertEqual(list(inspect.signature(method).parameters), ["image"])
        with self.assertRaises(TypeError):
            model(image, batch["roi_to_canonical"])

    def test_loss_is_exact_source_formula_and_has_no_distribution_cdf_term(self):
        self.assertIs(auxiliary_losses, auxiliary.auxiliary_losses)
        model = SemanticRelationReader(TinyEncoder()).eval()
        image = torch.randn(2, 3, 8, 8)
        prediction, details = model.forward_training(image)
        batch = batch_for(image)
        losses = auxiliary_losses(prediction, details, batch)
        primary = torch.nn.functional.smooth_l1_loss(prediction, batch["y"], beta=.05)
        expected = primary + .1 * (losses["role_nll_raw"] / (3 * math.log(256)) + losses["field_l1"])
        self.assertTrue(torch.equal(losses["total"], expected))
        changed_distribution = {**details, "distribution": torch.full_like(details["distribution"], float("nan"))}
        self.assertTrue(torch.equal(auxiliary_losses(prediction, changed_distribution, batch)["total"], expected))
        self.assertNotIn("reading_cdf", losses)
        self.assertFalse(CONFIG["training"]["reading_cdf"])

    def test_evaluation_counts_deployed_heads_once_and_preserves_failed_input_denominator(self):
        class Population(Dataset):
            rows = [{"sample_id": str(index), "group_id": "g"} for index in range(3)]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return {**self.rows[index], "image": torch.zeros(3, 8, 8), "y": torch.tensor(.3),
                        "input_failure": "missing RGB" if index == 1 else ""}

        def cpu_loader(dataset, *, batch_size, **_kwargs):
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = SemanticRelationReader(TinyEncoder()).eval()
        with patch("experiments.geoprr_v2_semantic_relation.make_loader", cpu_loader):
            rows, cost = evaluate_population(model, Population(), torch.device("cpu"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["absolute_error"], 1.)
        self.assertIsNone(rows[1]["prediction"])
        self.assertIsNone(rows[1]["invalid_mass"])
        self.assertTrue(math.isfinite(rows[0]["invalid_mass"]))
        self.assertTrue(math.isfinite(rows[0]["outside_arc_mass"]))
        for key in ("cnn_forward_calls", "state_head_forward_calls", "relation_forward_calls", "readout_forward_calls"):
            self.assertEqual(cost["counts"][key], 1)
        self.assertEqual(cost["counts"]["cnn_image_forwards"], 3)
        self.assertEqual(cost["counts"]["failed_input_placeholder_forwards"], 1)
        metadata = model_metadata(model)
        self.assertTrue(metadata["deployment_uses_geometry_readout"])
        self.assertFalse(metadata["point_head_present"])
        self.assertEqual(metadata["context_input_dim"], 5)
        self.assertEqual(metadata["parameters"], sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
