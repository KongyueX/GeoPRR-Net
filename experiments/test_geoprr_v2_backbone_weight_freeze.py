"""CPU checks for parameter freezing with source-updated BatchNorm statistics."""

import unittest

import torch
from torch import nn

from experiments.geoprr_v2_generalization_style import apply_generalization_training_policy


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(2, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.lateral = nn.Conv2d(4, 3, 1)
        self.refine = nn.Conv2d(3, 3, 1)
        self.state_head = nn.Conv2d(3, 1, 1)

    def set_shared_trainable(self, trainable):
        for module in (self.backbone, self.lateral, self.refine):
            module.requires_grad_(trainable)
            module.train(self.training and trainable)

    def forward(self, images):
        return self.state_head(self.refine(self.lateral(self.backbone(images))))


class BackboneWeightFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)
        self.model = TinyReader()
        self.policy = {"freeze_backbone_weights": True, "freeze_bn_statistics": False}

    def test_optimizer_keeps_backbone_parameters_and_updates_source_bn_and_remaining_weights(self):
        counts = apply_generalization_training_policy(self.model, self.policy)
        weights = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        running = {name: value.clone() for name, value in self.model.backbone[1].named_buffers()}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.03, weight_decay=.01)
        images = torch.randn(2, 2, 4, 4) + 2
        self.model(images).square().mean().backward()
        for name, parameter in self.model.named_parameters():
            if name.startswith("backbone."):
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        optimizer.step()
        for name, parameter in self.model.named_parameters():
            self.assertEqual(torch.equal(parameter, weights[name]), name.startswith("backbone."))
        bn = self.model.backbone[1]
        self.assertTrue(self.model.training and self.model.backbone.training and bn.training)
        self.assertEqual(int(bn.num_batches_tracked - running["num_batches_tracked"]), 1)
        self.assertFalse(torch.equal(bn.running_mean, running["running_mean"]))
        self.assertFalse(torch.equal(bn.running_var, running["running_var"]))
        self.assertEqual(counts["trainable_backbone_parameters"], 0)
        self.assertEqual(counts["frozen_backbone_parameters"], sum(p.numel() for p in self.model.backbone.parameters()))
        self.assertEqual(counts["trainable_decoder_parameters"],
                         sum(p.numel() for module in (self.model.lateral, self.model.refine) for p in module.parameters()))
        self.assertEqual(counts["trainable_head_parameters"], sum(p.numel() for p in self.model.state_head.parameters()))
        self.assertEqual(counts["trainable_parameters"],
                         counts["trainable_decoder_parameters"] + counts["trainable_head_parameters"])

    def test_policy_reapplies_after_shared_reset_evaluation_and_state_resume(self):
        apply_generalization_training_policy(self.model, self.policy)
        self.model.set_shared_trainable(True)
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.parameters()))
        self.model.eval()
        apply_generalization_training_policy(self.model, self.policy)
        resumed = TinyReader()
        resumed.load_state_dict(self.model.state_dict(), strict=True)
        apply_generalization_training_policy(resumed, self.policy)
        for model in (self.model, resumed):
            self.assertTrue(model.training and model.backbone[1].training)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            self.assertTrue(all(p.requires_grad for module in (model.lateral, model.refine, model.state_head)
                                for p in module.parameters()))

    def test_existing_normal_and_fixed_bn_policies_keep_all_parameters_trainable(self):
        for fixed_bn in (False, True):
            with self.subTest(freeze_bn_statistics=fixed_bn):
                model = TinyReader().eval()
                model.set_shared_trainable(False)
                counts = apply_generalization_training_policy(model, {"freeze_bn_statistics": fixed_bn})
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertTrue(model.training and model.backbone.training)
                self.assertEqual(model.backbone[1].training, not fixed_bn)
                self.assertEqual(counts["frozen_backbone_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
