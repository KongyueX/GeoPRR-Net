"""CPU toy checks for the fixed source-only feature moment mixing operator."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_source_stylemix as style


def controlled_moments(feature):
    value = feature.detach().float()
    mean = value.mean((2, 3), keepdim=True)
    sigma = (value.var((2, 3), unbiased=False, keepdim=True) + 1e-6).sqrt()
    return mean, sigma


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.num_features = 8
        self.conv_stem = nn.Conv2d(3, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.blocks = nn.Sequential(
            nn.Sequential(nn.Conv2d(4, 6, 1, bias=False), nn.BatchNorm2d(6), nn.Tanh()),
            nn.Sequential(nn.Conv2d(6, 8, 1, bias=False), nn.BatchNorm2d(8), nn.Tanh()))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(8, 12, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(12)

    def forward(self, image):
        self.calls += 1
        feature = self.blocks(self.bn1(self.conv_stem(image)))
        return self.norm_head(self.conv_head(self.global_pool(feature))).flatten(1)


class SourceStyleMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_exact_fixed_formula_uses_float32_moments_and_restores_input_dtype(self):
        operator = style.SourceStyleMix().train()
        feature = torch.arange(24, dtype=torch.float16).reshape(3, 2, 2, 2)
        coefficient = torch.tensor([0., .25, 1.]).reshape(3, 1, 1, 1)
        permutation = torch.tensor([2, 0, 1])
        observed = []
        handle = operator.moments.register_forward_hook(lambda _m, _i, output: observed.append(output))
        try:
            with patch("torch.distributions.Beta.sample", return_value=coefficient) as beta, \
                    patch("torch.randperm", return_value=permutation) as shuffle:
                actual = operator(feature)
        finally:
            handle.remove()
        mean, sigma = controlled_moments(feature)
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[permutation]
        mixed_sigma = coefficient * sigma + (1 - coefficient) * sigma[permutation]
        expected = (((feature.float() - mean) / sigma) * mixed_sigma + mixed_mean).to(feature.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, feature.dtype)
        self.assertEqual(len(observed), 1)
        self.assertTrue(all(value.dtype == torch.float32 for value in observed[0]))
        beta.assert_called_once_with((3, 1, 1, 1))
        shuffle.assert_called_once_with(3, device=feature.device)

    def test_gradients_flow_through_content_with_both_style_moments_detached(self):
        operator = style.SourceStyleMix().train()
        feature = torch.tensor([[[[0., 1.], [2., 3.]]], [[[1., 3.], [5., 7.]]]], requires_grad=True)
        coefficient = torch.zeros(2, 1, 1, 1)
        permutation = torch.tensor([1, 0])
        with patch("torch.distributions.Beta.sample", return_value=coefficient), \
                patch("torch.randperm", return_value=permutation):
            output = operator(feature)
        output.sum().backward()
        _, sigma = controlled_moments(feature)
        torch.testing.assert_close(feature.grad, (sigma[permutation] / sigma).expand_as(feature))
        feature32, mean, sigma = operator.moments(feature)
        self.assertTrue(feature32.requires_grad)
        self.assertFalse(mean.requires_grad)
        self.assertFalse(sigma.requires_grad)
        self.assertTrue(bool(torch.isfinite(feature.grad).all()))

    def test_eval_is_exact_identity_without_statistics_or_RNG_use(self):
        operator = style.SourceStyleMix().eval()
        feature = torch.randn(3, 2, 2, 2, requires_grad=True)
        before = torch.get_rng_state().clone()
        with patch.object(operator.moments, "forward", side_effect=AssertionError("eval computed moments")) as moments, \
                patch("torch.distributions.Beta", side_effect=AssertionError("eval sampled beta")) as beta, \
                patch("torch.randperm", side_effect=AssertionError("eval sampled permutation")) as shuffle:
            self.assertIs(operator(feature), feature)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        moments.assert_not_called()
        beta.assert_not_called()
        shuffle.assert_not_called()

    def test_partial_batch30_and_self_permutation_are_supported_without_parameters_or_state(self):
        operator = style.SourceStyleMix().train()
        feature = torch.randn(30, 2, 2, 2)
        with patch("torch.distributions.Beta.sample", return_value=torch.full((30, 1, 1, 1), .3)), \
                patch("torch.randperm", return_value=torch.arange(30)):
            output = operator(feature)
        torch.testing.assert_close(output, feature)
        self.assertEqual(tuple(output.shape), (30, 2, 2, 2))
        self.assertEqual(list(operator.parameters()), [])
        self.assertEqual(list(operator.buffers()), [])
        self.assertEqual(dict(operator.state_dict()), {})

    def test_source_reader_keeps_complete_state_BN_policy_one_CNN_point_and_actual_style_counts(self):
        encoder = ToyEncoder()
        original = deepcopy(encoder.state_dict())
        model = style.SourceStyleMixReader(encoder)
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in encoder.state_dict().items()))
        self.assertEqual(len(model.state_dict()), len(original) + 2)
        self.assertEqual((model.point.in_features, model.point.out_features), (12, 1))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["image"])
        counts, handles, intervals = style.style_counter(model, torch.device("cpu"))
        point_counts, point_handle = style.point_counter(model)
        counters = {key: value.clone() for key, value in model.state_dict().items() if key.endswith("num_batches_tracked")}
        try:
            model.train()
            prediction = model(torch.randn(4, 3, 8, 8))
            self.assertEqual((prediction.shape, prediction.dtype), (torch.Size([4]), torch.float32))
            F.smooth_l1_loss(prediction, torch.tensor([.02, .2, .3, .9]), beta=.05).backward()
            finite, groups = style.gradient_evidence(model)
            self.assertTrue(finite)
            self.assertTrue(all(groups.values()), groups)
            self.assertTrue(all(torch.equal(model.state_dict()[key], value + 1) for key, value in counters.items()))
            model.eval()
            with torch.inference_mode():
                model(torch.randn(2, 3, 8, 8))
            self.assertTrue(all(torch.equal(model.state_dict()[key], value + 1) for key, value in counters.items()))
        finally:
            point_handle.remove()
            for handle in handles:
                handle.remove()
        self.assertEqual(encoder.calls, 2)
        self.assertEqual(point_counts, {"point_forward_calls": 2, "point_image_forwards": 6})
        self.assertEqual(counts, {"calls": 2, "images": 6, "active_calls": 1, "active_images": 4, "inactive_calls": 1, "inactive_images": 2})
        self.assertEqual(style.style_cost(counts, intervals, torch.device("cpu"))["counts"], counts)
        with self.assertRaises(TypeError):
            model(torch.randn(1, 3, 8, 8), target=torch.tensor([.5]))
        self.assertEqual(encoder.calls, 2)

    def test_global_torch_RNG_snapshot_reproduces_beta_and_permutation(self):
        operator = style.SourceStyleMix().train()
        feature = torch.arange(32, dtype=torch.float32).reshape(4, 2, 2, 2)
        before = torch.get_rng_state().clone()
        first = operator(feature)
        after = torch.get_rng_state().clone()
        self.assertFalse(torch.equal(before, after))
        torch.set_rng_state(before)
        second = operator(feature)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_strict_reload_and_resume_reject_other_experiments_or_changed_recipe(self):
        model = style.SourceStyleMixReader(ToyEncoder())
        state = deepcopy(model.state_dict())
        restored = style.SourceStyleMixReader(ToyEncoder())
        restored.load_state_dict(state, strict=True)
        self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items()))
        missing = dict(state)
        del missing["point.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing, strict=True)
        contract = {"RGB_input_contract": deepcopy(style.RGB_INPUT_CONTRACT)}
        for protocol in ("geoprr_v2_high_resolution_direct_source_only_v1", "geoprr_v2_fixed_bn_semantic_source_only_v1", "geoprr_v2_spatial_semantic_source_only_v1"):
            with self.assertRaisesRegex(ValueError, "resume protocol"):
                style.validate_resume({"protocol": protocol, "config": style.CONFIG, "data_contract": contract}, contract)
        wrong = deepcopy(style.CONFIG)
        wrong["model"]["stylemix"]["source_training_probability"] = .5
        with self.assertRaisesRegex(ValueError, "resume protocol"):
            style.validate_resume({"protocol": style.PROTOCOL, "config": wrong, "data_contract": contract}, contract)


if __name__ == "__main__":
    unittest.main()
