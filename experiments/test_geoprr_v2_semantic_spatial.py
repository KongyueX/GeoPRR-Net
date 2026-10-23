"""Focused CPU checks for one-CNN spatial residual learning on a frozen parent."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_semantic_direct import SemanticDirectReader
from experiments.geoprr_v2_semantic_spatial import SemanticSpatialReader, frozen_base_evidence


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Conv2d(3, 80, 1), nn.Sequential(nn.BatchNorm2d(80), nn.SiLU()))
        self.conv_head = nn.Conv2d(80, 1280, 1)

    def forward(self, image):
        local = self.blocks(image)
        return self.conv_head(local.mean((2, 3), keepdim=True)).flatten(1)


def make_toy():
    torch.manual_seed(20262020)
    return SemanticSpatialReader(SemanticDirectReader(TinyEncoder()))


class SemanticSpatialTests(unittest.TestCase):
    def test_zero_initial_residual_is_exact_parent_including_extreme_logits(self):
        model = make_toy().eval()
        image = torch.randn(2, 3, 32, 32)
        point_outputs = []
        hook = model.base.point.register_forward_hook(lambda _module, _inputs, output: point_outputs.append(output.detach().clone()))
        try:
            with torch.no_grad():
                for value in (-1000., 0., 1000.):
                    model.base.point.bias.fill_(value)
                    prediction = model(image)
                    self.assertTrue(torch.equal(prediction, point_outputs[-1].squeeze(-1).float().sigmoid()))
        finally:
            hook.remove()

    def test_single_cnn_feature_shapes_and_normalized_attention(self):
        model = make_toy().eval()
        seen = {"encoder": [], "local": [], "semantic": [], "attention": []}
        hooks = [model.encoder.register_forward_hook(lambda _m, inputs, _out: seen["encoder"].append(len(inputs[0]))),
                 model.base.encoder.blocks[1].register_forward_hook(lambda _m, _in, out: seen["local"].append(tuple(out.shape))),
                 model.base.point.register_forward_pre_hook(lambda _m, inputs: seen["semantic"].append(tuple(inputs[0].shape))),
                 model.branch.attention.register_forward_hook(lambda _m, _in, out: seen["attention"].append(out.detach().clone()))]
        try:
            with torch.no_grad():
                prediction = model(torch.randn(2, 3, 32, 32))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(seen["encoder"], [2])
        self.assertEqual(seen["local"], [(2, 80, 32, 32)])
        self.assertEqual(seen["semantic"], [(2, 1280)])
        attention = seen["attention"][0]
        self.assertEqual(attention.shape, (2, 3, 1024))
        self.assertTrue(bool(torch.isfinite(attention).all()))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(2, 3), atol=1e-6))
        self.assertEqual(prediction.shape, (2,))

    def test_zero_last_first_gradient_and_nonzero_last_upstream_gradient(self):
        model = make_toy().train()
        image, target = torch.randn(2, 3, 32, 32), torch.tensor([.1, .3])
        F.smooth_l1_loss(model(image), target, beta=.05).backward()
        for name, parameter in model.branch.named_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            if name.startswith("residual.2."):
                self.assertTrue(bool(parameter.grad.abs().sum() > 0))
            else:
                self.assertEqual(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.branch.residual[-1].weight.fill_(.001)
        F.smooth_l1_loss(model(image), target, beta=.05).backward()
        for name, parameter in model.branch.named_parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertTrue(bool(parameter.grad.abs().sum() > 0), name)

    def test_parent_params_bn_buffers_and_only_new_optimizer_parameters(self):
        model = make_toy().train()
        original = {key: value.detach().clone() for key, value in model.base.state_dict().items()}
        optimizer = torch.optim.AdamW(model.branch.parameters(), lr=.001, weight_decay=.0001)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(optimized, {id(parameter) for parameter in model.branch.parameters()})
        self.assertFalse(optimized & {id(parameter) for parameter in model.base.parameters()})
        self.assertEqual(sum(parameter.numel() for parameter in model.branch.parameters()), 129057)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        self.assertFalse(model.base.training)
        self.assertTrue(model.branch.training)
        F.smooth_l1_loss(model(torch.randn(2, 3, 32, 32)), torch.tensor([.1, .3]), beta=.05).backward()
        optimizer.step()
        evidence = frozen_base_evidence(model, original)
        self.assertTrue(evidence["all_parent_state_tensors_unchanged"])
        self.assertEqual(evidence["BN_counter_increment"], 0)
        self.assertTrue(torch.equal(model.base.encoder.blocks[1][0].num_batches_tracked,
                                    original["encoder.blocks.1.0.num_batches_tracked"]))


if __name__ == "__main__":
    unittest.main()
