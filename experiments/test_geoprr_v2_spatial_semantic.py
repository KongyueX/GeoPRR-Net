"""CPU toy checks for ordered2x2 semantic cells and the new scalar point."""
from copy import deepcopy
import inspect
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_spatial_semantic as spatial


class ToyPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Identity()

    def forward(self, feature):
        return self.flatten(self.pool(feature))


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.num_features = 4
        self.conv_stem = nn.Conv2d(3, 4, 1, bias=False)
        self.global_pool = ToyPool()
        self.conv_head = nn.Conv2d(4, 1280, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(1280)
        self.act2 = nn.Identity()
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Identity()

    def forward(self, image):
        self.calls += 1
        feature = self.conv_stem(image)
        feature = self.global_pool(feature)
        feature = self.conv_head(feature)
        feature = self.norm_head(feature)
        feature = self.act2(feature)
        return self.classifier(self.flatten(feature))


class SpatialSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_fixed2x2_retains_channel_major_cell_order_and_distinguishes_equal_GAP(self):
        encoder = ToyEncoder()
        with torch.no_grad():
            encoder.conv_stem.weight.zero_()
            encoder.conv_stem.weight[0, 0, 0, 0] = 1.
            encoder.conv_head.weight.zero_()
            encoder.conv_head.weight[:, 0, 0, 0] = 1.
            encoder.norm_head.weight.fill_((1 + encoder.norm_head.eps) ** .5)
        model = spatial.SpatialSemanticReader(encoder).eval()
        quadrants = torch.tensor([[-.25, .25], [.5, 0.]])
        first = quadrants.repeat_interleave(2, 0).repeat_interleave(2, 1)
        second = first.flip(-1)
        self.assertEqual(float(first.mean()), float(second.mean()))
        image = torch.zeros(2, 3, 4, 4)
        image[0, 0], image[1, 0] = first, second
        with torch.inference_mode():
            semantic = model.encoder(image)
        torch.testing.assert_close(semantic[0].reshape(1280, 2, 2), quadrants.expand(1280, 2, 2))
        torch.testing.assert_close(semantic[1].reshape(1280, 2, 2), quadrants.flip(-1).expand(1280, 2, 2))
        with torch.no_grad():
            model.point.weight.zero_()
            model.point.weight[0, 0] = 1.
            model.point.bias.zero_()
        with torch.inference_mode():
            prediction = model(image)
        self.assertNotEqual(float(prediction[0]), float(prediction[1]))
        torch.testing.assert_close(prediction, torch.tensor([-.25, .25]).sigmoid())

    def test_complete_neck_runs_on_four_cells_before5120_flatten_and_single_point(self):
        model = spatial.SpatialSemanticReader(ToyEncoder()).eval()
        observed = []
        neck_hook = model.encoder.norm_head.register_forward_hook(lambda _m, _i, output: observed.append(tuple(output.shape)))
        counts, point_hook = spatial.point_counter(model)
        try:
            with torch.inference_mode():
                prediction = model(torch.randn(3, 3, 8, 8))
        finally:
            neck_hook.remove()
            point_hook.remove()
        self.assertEqual(observed, [(3, 1280, 2, 2)])
        self.assertEqual(model.encoder.calls, 1)
        self.assertEqual(counts, {"point_forward_calls": 1, "point_image_forwards": 3})
        self.assertEqual((model.semantic_cell_dim, model.readout_dim), (1280, 5120))
        self.assertEqual((model.point.in_features, model.point.out_features), (5120, 1))
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["image"])
        with self.assertRaises(TypeError):
            model(torch.randn(1, 3, 8, 8), target=torch.tensor([.5]))
        self.assertEqual(model.encoder.calls, 1)

    def test_source_training_updates_BN_and_reaches_stem_neck_and_point_gradients(self):
        model = spatial.SpatialSemanticReader(ToyEncoder()).train()
        before = model.encoder.norm_head.num_batches_tracked.clone()
        prediction = model(torch.randn(4, 3, 8, 8))
        F.smooth_l1_loss(prediction, torch.tensor([.02, .2, .3, .9]), beta=.05).backward()
        finite, groups = spatial.gradient_evidence(model)
        self.assertTrue(finite)
        self.assertTrue(all(groups.values()), groups)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(int(model.encoder.norm_head.num_batches_tracked - before), 1)
        model.eval()
        with torch.inference_mode():
            model(torch.randn(2, 3, 8, 8))
        self.assertEqual(int(model.encoder.norm_head.num_batches_tracked - before), 1)

    def test_construction_retains_complete_encoder_state_without_forward_or_extra_pool_state(self):
        encoder = ToyEncoder()
        original = deepcopy(encoder.state_dict())
        model = spatial.SpatialSemanticReader(encoder)
        self.assertIs(model.encoder, encoder)
        self.assertEqual(encoder.calls, 0)
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in encoder.state_dict().items()))
        self.assertEqual(set(encoder.state_dict()), set(original))
        self.assertEqual(list(encoder.global_pool.state_dict()), [])
        self.assertEqual(sum(parameter.numel() for parameter in model.point.parameters()), 5121)
        self.assertEqual(len(model.state_dict()), len(original) + 2)
        self.assertTrue(bool((model.point.bias == 0).all()))

    def test_strict_state_reload_rejects_old1280_point_and_resume_rejects_other_protocols(self):
        model = spatial.SpatialSemanticReader(ToyEncoder())
        state = deepcopy(model.state_dict())
        restored = spatial.SpatialSemanticReader(ToyEncoder())
        restored.load_state_dict(state, strict=True)
        self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items()))
        old_head = dict(state)
        old_head["point.weight"] = torch.zeros(1, 1280)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(old_head, strict=True)
        missing = dict(state)
        del missing["point.bias"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing, strict=True)
        contract = {"RGB_input_contract": deepcopy(spatial.RGB_INPUT_CONTRACT)}
        for protocol in ("geoprr_v2_high_resolution_direct_source_only_v1", "geoprr_v2_fixed_bn_semantic_source_only_v1"):
            with self.assertRaisesRegex(ValueError, "resume protocol"):
                spatial.validate_resume({"protocol": protocol, "config": spatial.CONFIG, "data_contract": contract}, contract)
        wrong_config = deepcopy(spatial.CONFIG)
        wrong_config["model"]["readout_dim"] = 1280
        with self.assertRaisesRegex(ValueError, "resume protocol"):
            spatial.validate_resume({"protocol": spatial.PROTOCOL, "config": wrong_config, "data_contract": contract}, contract)


if __name__ == "__main__":
    unittest.main()
