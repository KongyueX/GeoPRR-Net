"""Focused CPU checks for stable ordered CDF supervision and full-mass expectation."""
from copy import deepcopy
import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_ordinal_distribution import (
    OrdinalDistributionReader, cdf_log_score, expected_reading, gradient_evidence,
    head_counter, log_cdf_terms, target_cdf, validate_nodes,
)


class ToyEncoder(nn.Module):
    """Expose one trainable semantic neck without constructing the real encoder."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.conv_stem = nn.Conv2d(3, 8, 1)
        self.activation = nn.GELU()
        self.conv_head = nn.Conv2d(8, 1280, 1)
        self.norm_head = nn.BatchNorm2d(1280)
        self.num_features = 8

    def forward(self, image):
        self.calls += 1
        feature = self.activation(self.conv_stem(image))
        return self.norm_head(self.conv_head(feature)).mean((2, 3))


class OrdinalDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)
        self.nodes = torch.linspace(0, 1, 256, dtype=torch.float32)

    def test_target_cdf_matches_two_hot_mass_at_endpoints_and_continuous_values(self):
        target = torch.tensor([0., 1., 17/255, .12345, .5, 254.9/255])
        coordinate = 255 * target
        lower = coordinate.floor().long()
        fraction = coordinate - lower
        mass = torch.zeros(len(target), 256)
        mass.scatter_add_(1, lower[:, None], (1-fraction)[:, None])
        mass.scatter_add_(1, (lower+1).clamp(max=255)[:, None], fraction[:, None])
        cdf = target_cdf(target)
        torch.testing.assert_close(cdf, mass.cumsum(-1)[:, :-1], rtol=0, atol=1e-6)
        self.assertTrue(bool((cdf[:, 1:] >= cdf[:, :-1]).all()))
        torch.testing.assert_close(mass.sum(-1), torch.ones(len(target)))
        torch.testing.assert_close(expected_reading(mass.log(), self.nodes), target, rtol=0, atol=1e-7)
        self.assertTrue(bool((cdf[0] == 1).all()))
        self.assertTrue(bool((cdf[1] == 0).all()))

    def test_log_cdf_and_loss_match_direct_moderate_probability_formula(self):
        log_probability = F.log_softmax(torch.randn(3, 256)*.2, dim=-1)
        probability = log_probability.exp()
        left, right = log_cdf_terms(log_probability)
        direct_left = probability.cumsum(-1)[:, :-1].log()
        direct_right = probability.flip(-1).cumsum(-1).flip(-1)[:, 1:].log()
        torch.testing.assert_close(left, direct_left, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(right, direct_right, rtol=1e-5, atol=1e-6)
        self.assertTrue(bool((left[:, 1:] >= left[:, :-1]).all()))
        self.assertTrue(bool((right[:, 1:] <= right[:, :-1]).all()))
        torch.testing.assert_close(probability.sum(-1), torch.ones(3), rtol=1e-6, atol=1e-7)
        target = torch.tensor([0., .37, 1.])
        truth = target_cdf(target)
        direct = -(truth*direct_left + (1-truth)*direct_right).mean()
        torch.testing.assert_close(cdf_log_score(log_probability, target), direct, rtol=1e-6, atol=1e-7)

    def test_extreme_finite_wrong_logits_keep_finite_loss_and_corrective_main_gradient(self):
        target = torch.tensor([0., 1., .37])
        wrong_nodes = [255, 0, 255]
        for magnitude in (80., 1000.):
            logits = torch.full((3, 256), -magnitude)
            logits[torch.arange(3), wrong_nodes] = magnitude
            logits.requires_grad_()
            log_probability = F.log_softmax(logits, dim=-1)
            loss = cdf_log_score(log_probability, target)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            for row, node in enumerate(wrong_nodes):
                self.assertGreater(float(logits.grad[row, node]), 0.)

    def test_full_mass_decode_is_continuous_and_bimodal_expectation_remains_between_peaks(self):
        fraction = torch.tensor([.1, .4, .9])
        mass = torch.zeros(3, 256)
        mass[:, 10], mass[:, 11] = 1-fraction, fraction
        readings = expected_reading(mass.log(), self.nodes)
        torch.testing.assert_close(readings, (10+fraction)/255, rtol=0, atol=1e-7)
        self.assertTrue(bool((readings[1:] > readings[:-1]).all()))
        self.assertTrue(bool(((readings*255)-(readings*255).round()).abs().gt(.01).all()))
        bimodal = torch.zeros(1, 256); bimodal[0, 0] = .5; bimodal[0, 255] = .5
        torch.testing.assert_close(expected_reading(bimodal.log(), self.nodes), torch.tensor([.5]))
        centered_logits = torch.full((1, 256), -20.); centered_logits[:, 127:129] = 0.
        endpoint_logits = torch.full((1, 256), -20.); endpoint_logits[:, [0, 255]] = 0.
        centered = F.log_softmax(centered_logits, dim=-1)
        endpoints = F.log_softmax(endpoint_logits, dim=-1)
        torch.testing.assert_close(expected_reading(centered, self.nodes), expected_reading(endpoints, self.nodes), rtol=1e-6, atol=1e-7)
        self.assertGreater(float(cdf_log_score(endpoints, torch.tensor([.5]))),
                           float(cdf_log_score(centered, torch.tensor([.5]))))

    def test_log_distribution_and_forward_share_one_head_and_partial_batch_readout(self):
        model = OrdinalDistributionReader(ToyEncoder()).eval()
        counts, hook = head_counter(model)
        image = torch.randn(3, 3, 8, 8)
        try:
            with torch.inference_mode():
                log_probability = model.log_distribution(image)
                training_readout = model.expected_reading(log_probability)
                deployed = model(image)
                partial = model(image[1:])
                single = model(image[2:])
        finally:
            hook.remove()
        torch.testing.assert_close(training_readout, deployed, rtol=0, atol=0)
        torch.testing.assert_close(deployed[1:], partial, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(deployed[2:], single, rtol=1e-5, atol=1e-7)
        self.assertEqual(model.encoder.calls, 4)
        self.assertEqual(counts, {"head_forward_calls":4, "head_image_forwards":9})
        self.assertEqual(tuple(log_probability.shape), (3, 256))
        self.assertEqual(log_probability.dtype, torch.float32)
        self.assertEqual(deployed.dtype, torch.float32)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(model).values()))

    def test_single_cdf_training_pass_reaches_encoder_neck_and_new_head(self):
        model = OrdinalDistributionReader(ToyEncoder()).train()
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        log_probability = model.log_distribution(torch.randn(3, 3, 8, 8))
        prediction = model.expected_reading(log_probability)
        loss = cdf_log_score(log_probability, torch.tensor([0., .37, 1.]))
        loss.backward()
        finite, nonzero = gradient_evidence(model)
        self.assertTrue(finite)
        self.assertTrue(all(nonzero.values()), nonzero)
        self.assertEqual(model.encoder.calls, 1)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))
        self.assertTrue(all(torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()))
        self.assertGreater(float(model.head.weight.abs().sum()), 0.)

    def test_both_rgb_only_entries_reject_gt_before_cnn_and_new_state_reload_is_strict(self):
        model = OrdinalDistributionReader(ToyEncoder()).eval()
        image = torch.randn(1, 3, 8, 8)
        for entry in (model.forward, model.log_distribution):
            self.assertEqual(list(inspect.signature(entry).parameters), ["image"])
            with self.assertRaises(TypeError): entry(image, target=torch.tensor([.5]))
        self.assertEqual(model.encoder.calls, 0)
        state = deepcopy(model.state_dict())
        self.assertEqual(len(state)-len(model.encoder.state_dict()), 3)
        self.assertIn("nodes", state)
        self.assertFalse(any("point" in key or "adapter" in key for key in state))
        restored = OrdinalDistributionReader(ToyEncoder()).eval()
        restored.load_state_dict(state, strict=True)
        validate_nodes(restored)
        with torch.inference_mode():
            torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
        missing = dict(state); del missing["nodes"]
        with self.assertRaises(RuntimeError): restored.load_state_dict(missing, strict=True)
        extra = dict(state); extra["point.weight"] = torch.zeros(1, 1280)
        with self.assertRaises(RuntimeError): restored.load_state_dict(extra, strict=True)
        changed = deepcopy(state); changed["nodes"][0] = .1
        restored.load_state_dict(changed, strict=True)
        with self.assertRaises(ValueError): validate_nodes(restored)


if __name__ == "__main__":
    unittest.main()
