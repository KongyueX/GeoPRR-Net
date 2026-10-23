"""Focused CPU checks for the new semantic-to-role-content information path."""
import inspect
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments import geoprr_v2_semantic_relation as base
from experiments.geoprr_v2_reference_relation import CircularProgressPushforward, ReferenceRelationHead, ReferenceRelationStateHead
from experiments.geoprr_v2_semantic_role import (
    SemanticConditionedRelationHead, SemanticRoleStateHead, auxiliary_losses, condition_model,
    evaluate_population, model_metadata,
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


def tiny_state_head():
    head = ReferenceRelationStateHead(8, 4, 2, "local")
    head.context[0] = nn.Linear(5, 64)
    return head


def batch_for(image):
    count, size = len(image), image.shape[-1]
    h = torch.tensor([[1 / size, 0, -.5 + .5 / size], [0, 1 / size, -.5 + .5 / size], [0, 0, 1.]])
    return {"image": image, "y": torch.linspace(.2, .8, count), "roi_to_canonical": h[None].repeat(count, 1, 1),
            "span_radians": torch.full((count,), 1.5 * math.pi),
            "field_target": torch.linspace(0, 1, size * size).reshape(1, size, size).repeat(count, 1, 1),
            "field_mask": torch.ones(count, 1, size, size)}


class SemanticRoleTests(unittest.TestCase):
    def test_zero_query_exactly_preserves_role_outputs_and_reading_for_partial_batches(self):
        torch.manual_seed(21)
        original = tiny_state_head().eval()
        conditioned = SemanticRoleStateHead(original, 5).eval()
        readout = CircularProgressPushforward(8)
        for count in (1, 3):
            features, semantic = torch.randn(count, 4, 5, 5), torch.randn(count, 5)
            old, new = original(features, semantic), conditioned(features, semantic)
            for key in ("A", "P_rel", "B_rel", "log_A", "log_P_rel", "log_B_rel", "J", "product",
                        "reference_unary", "log_Z_P", "log_Z_B", "role_tokens", "query_points"):
                self.assertTrue(torch.equal(old[key], new[key]), key)
            for key in ("center", "spd", "raw"):
                self.assertTrue(torch.equal(old["calibration"][key], new["calibration"][key]))
            self.assertTrue(torch.equal(readout(old["J"])["reading"], readout(new["J"])["reading"]))
            self.assertEqual(int(torch.count_nonzero(new["semantic_unary_terms"])), 0)

    def test_conditioning_reuses_existing_modules_parameters_and_state_keys(self):
        model = base.SemanticRelationReader(TinyEncoder())
        state = {key: value.clone() for key, value in model.state_dict().items()}
        parameters = dict(model.named_parameters())
        original_head, encoder, lateral, readout = model.state_head, model.encoder, model.lateral, model.readout
        returned = condition_model(model)
        self.assertIs(returned, model)
        self.assertIs(model.encoder, encoder)
        self.assertIs(model.lateral, lateral)
        self.assertIs(model.readout, readout)
        self.assertIs(model.state_head.context, original_head.context)
        self.assertIs(model.state_head.radial_embedding, original_head.radial_embedding)
        self.assertIs(model.state_head.directions, original_head.directions)
        self.assertIs(model.state_head.radii, original_head.radii)
        self.assertIs(model.state_head.relation.projections, original_head.relation.projections)
        self.assertIs(model.state_head.relation.unaries, original_head.relation.unaries)
        self.assertIs(model.state_head.relation.relative_indices, original_head.relation.relative_indices)
        new_key = "state_head.relation.semantic_queries.weight"
        self.assertEqual(set(model.state_dict()), set(state) | {new_key})
        for key, value in state.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)
        for key, parameter in parameters.items():
            self.assertIs(dict(model.named_parameters())[key], parameter)
        self.assertEqual(int(torch.count_nonzero(model.state_dict()[new_key])), 0)
        self.assertIsNone(model.state_head.relation.semantic_queries.bias)

    def test_nonzero_semantic_queries_change_roles_with_fixed_tokens_and_calibration(self):
        torch.manual_seed(13)
        original = tiny_state_head().eval()
        with torch.no_grad():
            original.context[-1].weight.zero_()
            original.context[-1].bias.zero_()
        conditioned = SemanticRoleStateHead(original, 5).eval()
        with torch.no_grad():
            conditioned.relation.semantic_queries.weight.copy_(torch.arange(30).reshape(6, 5) / 100)
        features = torch.randn(3, 4, 5, 5)
        semantic = torch.tensor([[-1., 0., 1., 2., 3.], [3., 1., -1., 0., 2.], [0., 2., 1., -3., 4.]])
        first, second = conditioned(features, semantic), conditioned(features, semantic.roll(1, 0))
        self.assertTrue(torch.equal(first["role_tokens"], second["role_tokens"]))
        self.assertTrue(torch.equal(first["calibration"]["raw"], second["calibration"]["raw"]))
        self.assertGreater(float((first["role_unaries"] - second["role_unaries"]).abs().max().detach()), 1e-5)
        self.assertGreater(float((first["J"] - second["J"]).abs().max().detach()), 1e-6)
        old_first, old_second = original(features, semantic), original(features, semantic.roll(1, 0))
        self.assertTrue(torch.equal(old_first["J"], old_second["J"]))

    def test_query_formula_relative_reference_gather_and_shared_reference_joint(self):
        torch.manual_seed(17)
        original = ReferenceRelationHead(2, 8, "local")
        conditioned = SemanticConditionedRelationHead(original, 5)
        with torch.no_grad():
            conditioned.semantic_queries.weight.copy_(torch.randn(6, 5) * .3)
        tokens, semantic = torch.randn(3, 8, 2) * 2, torch.randn(3, 5)
        output = conditioned(tokens, semantic)
        roles = [projection(tokens) for projection in original.projections]
        queries = conditioned.semantic_queries(semantic).reshape(3, 3, 2)
        unaries = [layer(role).squeeze(-1) + (role * queries[:, index, None]).sum(-1) / math.sqrt(2)
                   for index, (layer, role) in enumerate(zip(original.unaries, roles, strict=True))]
        self.assertTrue(torch.allclose(output["role_unaries"], torch.stack(unaries, 1), atol=1e-6))
        for role_index, key in ((1, "log_P_rel"), (2, "log_B_rel")):
            energy = (roles[0] @ roles[role_index].transpose(1, 2)) / math.sqrt(2) + unaries[role_index][:, None]
            absolute_log = energy.log_softmax(-1)
            expected = torch.stack([torch.stack([absolute_log[:, a, (a + delta) % 8] for delta in range(8)], -1)
                                    for a in range(8)], 1)
            self.assertTrue(torch.allclose(output[key], expected, atol=1e-6))
        expected_joint = torch.stack([output["A"][:, a, None, None] * output["P_rel"][:, a, :, None]
                                      * output["B_rel"][:, a, None, :] for a in range(8)]).sum(0)
        self.assertTrue(torch.allclose(output["J"], expected_joint, atol=1e-6))
        self.assertGreater(float((output["J"] - output["product"]).abs().max().detach()), 1e-7)

    def test_query_gradient_image_only_api_and_strict_reload_with_source_loss(self):
        torch.manual_seed(23)
        model = condition_model(base.SemanticRelationReader(TinyEncoder())).eval()
        image = torch.randn(2, 3, 8, 8)
        prediction, details = model.forward_training(image)
        batch = batch_for(image)
        losses = auxiliary_losses(prediction, details, batch)
        losses["total"].backward()
        gradient = model.state_head.relation.semantic_queries.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertIs(auxiliary_losses, base.auxiliary_losses)
        changed = auxiliary_losses(prediction, details, {**batch, "field_target": 1 - batch["field_target"],
                                                          "span_radians": torch.ones(2)})
        self.assertTrue(torch.equal(losses["primary"], changed["primary"]))
        self.assertFalse(torch.equal(losses["field_l1"], changed["field_l1"]))
        self.assertTrue(torch.equal(model(image), prediction))
        for method in (model.forward, model.forward_training, model.forward_details):
            self.assertEqual(list(inspect.signature(method).parameters), ["image"])
        with self.assertRaises(TypeError):
            model(image, batch["roi_to_canonical"])
        with torch.no_grad():
            model.state_head.relation.semantic_queries.weight.add_(.01)
        restored = condition_model(base.SemanticRelationReader(TinyEncoder())).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored(image), model(image)))

    def test_evaluator_counts_query_once_and_retains_failed_original(self):
        class Population(Dataset):
            rows = [{"sample_id": str(index), "group_id": "g"} for index in range(3)]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return {**self.rows[index], "image": torch.zeros(3, 8, 8), "y": torch.tensor(.3),
                        "input_failure": "missing RGB" if index == 1 else ""}

        def cpu_loader(dataset, *, batch_size, **_kwargs):
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = condition_model(base.SemanticRelationReader(TinyEncoder())).eval()
        with patch("experiments.geoprr_v2_semantic_role.make_loader", cpu_loader):
            rows, cost = evaluate_population(model, Population(), torch.device("cpu"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["absolute_error"], 1.)
        self.assertIsNone(rows[1]["prediction"])
        self.assertIsNone(rows[1]["invalid_mass"])
        self.assertTrue(math.isfinite(rows[0]["invalid_mass"]))
        for key in ("cnn_forward_calls", "state_head_forward_calls", "relation_forward_calls", "readout_forward_calls", "semantic_query_forward_calls"):
            self.assertEqual(cost["counts"][key], 1)
        self.assertEqual(cost["counts"]["cnn_image_forwards"], 3)
        self.assertEqual(cost["counts"]["failed_input_placeholder_forwards"], 1)
        metadata = model_metadata(model)
        self.assertTrue(metadata["semantic_role_conditioning"])
        self.assertEqual(metadata["semantic_query_parameters"], 5 * 3 * 16)
        self.assertEqual(metadata["semantic_query_parameter_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
