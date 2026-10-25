"""CPU checks for the independent fixed-rank bilinear semantic readout."""
from copy import deepcopy
import inspect
import math
import unittest
from unittest import mock

import torch
from torch import nn

from experiments.geoprr_v2_bilinear_semantic import (
    BilinearSemanticReader, CONFIG, FactorizedBilinearPoint, PROTOCOL, READOUT_CONTRACT,
    create_model, gradient_evidence, model_metadata, point_counter, validate_external_checkpoint,
    validate_resume,
)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.conv_stem = nn.Conv2d(3, 4, 1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(4, 1280, 1, bias=False)
        self.norm_head = nn.BatchNorm2d(1280)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        features = self.global_pool(self.conv_stem(image))
        return self.norm_head(self.conv_head(features)).flatten(1)


class BilinearPointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)

    def test_exact_all64_rank_formula_with_biases(self):
        point = FactorizedBilinearPoint(1280)
        with torch.no_grad():
            point.projection.weight.zero_()
            point.projection.bias.zero_()
            for rank in range(64):
                point.projection.weight[rank, rank] = 1
                point.projection.weight[rank+64, rank+64] = 2
                point.projection.bias[rank] = .25
                point.projection.bias[rank+64] = -.5
            point.output_bias.fill_(.125)
        semantic = torch.zeros(3, 1280)
        semantic[:, :64] = torch.tensor([0., 1., 2.])[:, None]
        semantic[:, 64:128] = torch.tensor([1., 2., 3.])[:, None]
        expected = torch.tensor([[64*.25*1.5/8+.125],
                                 [64*1.25*3.5/8+.125],
                                 [64*2.25*5.5/8+.125]])
        torch.testing.assert_close(point(semantic), expected, rtol=0, atol=0)

    def test_channel_cross_term_has_nonzero_mixed_difference(self):
        point = FactorizedBilinearPoint(1280)
        with torch.no_grad():
            point.projection.weight.zero_()
            point.projection.bias.zero_()
            point.output_bias.zero_()
            point.projection.weight[0, 0] = 1
            point.projection.weight[64, 1] = 1
        semantic = torch.zeros(4, 1280)
        semantic[:, :2] = torch.tensor([[0., 0.], [2., 0.], [0., 3.], [2., 3.]])
        logits = point(semantic).flatten()
        torch.testing.assert_close(logits, torch.tensor([0., 0., 0., .75]), rtol=0, atol=0)
        self.assertEqual(float(logits[3]-logits[1]-logits[2]+logits[0]), .75)

    def test_projection_product_and_logit_remain_float32_under_autocast(self):
        point = FactorizedBilinearPoint(1280)
        with torch.no_grad():
            point.projection.weight.zero_()
            point.projection.bias.zero_()
            point.output_bias.zero_()
            point.projection.weight[0, 0] = 700.125
            point.projection.weight[64, 1] = 700.25
        semantic = torch.zeros(3, 1280, dtype=torch.float16)
        semantic[:, :2] = 10
        observed = []
        def observe(_module, inputs, output):
            observed.append((inputs[0].dtype, output.dtype, list(output.shape)))
        hooks = [point.projection.register_forward_hook(observe), point.product.register_forward_hook(observe)]
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                result = point(semantic)
        finally:
            for hook in hooks:
                hook.remove()
        expected = torch.full((3, 1), (700.125*10)*(700.25*10)/8)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertGreater(float(result.min()), torch.finfo(torch.float16).max)
        self.assertEqual(observed, [(torch.float32, torch.float32, [3,128]),
                                    (torch.float32, torch.float32, [3,64])])

    def test_fixed_asymmetric_initialization_and_three_tensor_state(self):
        point = FactorizedBilinearPoint(1280)
        self.assertEqual(sum(p.numel() for p in point.parameters()), 163969)
        self.assertEqual(set(point.state_dict()), {"projection.weight", "projection.bias", "output_bias"})
        self.assertEqual(len(list(point.parameters())), 3)
        self.assertEqual(len(point.product.state_dict()), 0)
        self.assertTrue(bool((point.projection.bias == 0).all()))
        self.assertTrue(bool((point.output_bias == 0).all()))
        self.assertAlmostEqual(float(point.projection.weight[:64].std()), 1/math.sqrt(1280), delta=.001)
        self.assertAlmostEqual(float(point.projection.weight[64:].std()), .001, delta=.00003)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)

    def test_reader_preserves_encoder_and_constructs_only_one_new_projection(self):
        encoder = ToyEncoder()
        state = {key:value.clone() for key, value in encoder.state_dict().items()}
        linear = nn.Linear
        with mock.patch("experiments.geoprr_v2_bilinear_semantic.nn.Linear", wraps=linear) as construction:
            reader = BilinearSemanticReader(encoder)
        construction.assert_called_once_with(1280, 128)
        self.assertIs(reader.encoder, encoder)
        self.assertEqual(encoder.calls, 0)
        self.assertEqual(list(inspect.signature(reader.forward).parameters), ["image"])
        self.assertEqual(set(reader.state_dict()), {"encoder."+key for key in state} |
                         {"point.projection.weight", "point.projection.bias", "point.output_bias"})
        for key, value in state.items():
            torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)

    def test_batch1_batch3_scalar_shapes_and_one_encoder_point_call(self):
        reader = BilinearSemanticReader(ToyEncoder()).eval()
        counts, hook = point_counter(reader)
        try:
            for batch in (1, 3):
                result = reader(torch.randn(batch, 3, 4, 4))
                self.assertEqual(tuple(result.shape), (batch,))
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertTrue(bool(((result >= 0) & (result <= 1)).all()))
        finally:
            hook.remove()
        self.assertEqual(reader.encoder.calls, 2)
        self.assertEqual(counts, {"point_forward_calls":2, "point_image_forwards":4})
        self.assertGreater(hook.seconds, 0)

    def test_both_factor_halves_and_encoder_stem_neck_have_finite_nonzero_gradients(self):
        reader = BilinearSemanticReader(ToyEncoder()).train()
        result = reader(torch.randn(3, 3, 4, 4))
        loss = nn.functional.smooth_l1_loss(result, torch.tensor([0., .4, 1.]), beta=.05)
        loss.backward()
        finite, nonzero = gradient_evidence(reader)
        self.assertTrue(finite)
        self.assertTrue(all(nonzero.values()), nonzero)
        self.assertEqual(set(nonzero), {"encoder", "encoder_stem", "semantic_conv_head", "semantic_norm_head",
                                        "point", "factor_A_weight", "factor_B_weight"})

    def test_strict_reload_rejects_a_linear_or_wrong_rank_head(self):
        reader = BilinearSemanticReader(ToyEncoder())
        state = {key:value.clone() for key, value in reader.state_dict().items()}
        reloaded = BilinearSemanticReader(ToyEncoder())
        reloaded.load_state_dict(state, strict=True)
        wrong_shape = dict(state)
        wrong_shape["point.projection.weight"] = torch.zeros(64, 1280)
        with self.assertRaises(RuntimeError):
            reloaded.load_state_dict(wrong_shape, strict=True)
        old_point = {key:value for key, value in state.items() if not key.startswith("point.")}
        old_point.update({"point.weight":torch.zeros(1,1280), "point.bias":torch.zeros(1)})
        with self.assertRaises(RuntimeError):
            reloaded.load_state_dict(old_point, strict=True)

    def test_new_protocol_and_readout_contract_are_required(self):
        contract = {"source": "toy_contract"}
        resume = {"protocol":PROTOCOL, "config":CONFIG, "data_contract":contract,
                  "readout_contract":READOUT_CONTRACT, "model":{}, "optimizer":{}, "scheduler":{},
                  "scaler":{}, "rng":{}, "history":[{"epoch":1, "validation":{}, "selection_score":.1}],
                  "totals":{}, "validation_rows":{"original":[None]*1506, "roll":[None]*512},
                  "initialization":{}, "checkpointed_process_seconds":0., "epoch":1, "best_score":.1, "best_epoch":1}
        with mock.patch("experiments.geoprr_v2_bilinear_semantic.selection_score", return_value=.1):
            validate_resume(resume, contract)
            resume["protocol"] = "geoprr_v2_high_resolution_direct_source_only_v1"
            with self.assertRaises(ValueError):
                validate_resume(resume, contract)
            resume["protocol"] = PROTOCOL
            resume["readout_contract"] = {**READOUT_CONTRACT, "rank":32}
            with self.assertRaises(ValueError):
                validate_resume(resume, contract)
        checkpoint = {"protocol":PROTOCOL, "config":CONFIG, "readout_contract":READOUT_CONTRACT,
                      "epoch":15, "best_epoch":15, "best_score":.1}
        completed = {"protocol":PROTOCOL, "config":CONFIG, "readout_contract":READOUT_CONTRACT,
                     "epochs":60, "best_epoch":15, "best_score":.1,
                     "status":"complete", "completion_reason":"source_score_patience"}
        validate_external_checkpoint(checkpoint, completed)
        completed["protocol"] = "geoprr_v2_high_resolution_direct_source_only_v1"
        with self.assertRaises(ValueError):
            validate_external_checkpoint(checkpoint, completed)
        completed["protocol"] = PROTOCOL
        completed["readout_contract"] = deepcopy(READOUT_CONTRACT)
        completed["readout_contract"]["precision"] = "float16"
        with self.assertRaises(ValueError):
            validate_external_checkpoint(checkpoint, completed)


class RealArchitectureTests(unittest.TestCase):
    def test_fresh_cpu_architecture_without_weights_or_forward(self):
        model = create_model(pretrained=False)
        metadata = model_metadata(model)
        self.assertEqual((metadata["semantic_dim"], metadata["encoder_reported_num_features"]), (1280, 960))
        self.assertEqual((metadata["encoder_parameters"], metadata["encoder_state_tensors"]), (8434512, 462))
        self.assertEqual((metadata["point_parameters"], metadata["point_parameter_tensors"], metadata["point_state_tensors"]), (163969, 3, 3))
        self.assertEqual((metadata["parameters"], metadata["trainable_parameters"], metadata["parameter_tensors"],
                          metadata["state_tensors"]), (8598481, 8598481, 234, 465))
        self.assertEqual(metadata["factor_projection_shape"], [1280,128])
        self.assertEqual(sum(key.endswith("num_batches_tracked") for key in model.state_dict()), 77)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
        state = {key:value.clone() for key, value in model.state_dict().items()}
        reloaded = create_model(pretrained=False)
        reloaded.load_state_dict(state, strict=True)
        for key, value in state.items():
            torch.testing.assert_close(reloaded.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
