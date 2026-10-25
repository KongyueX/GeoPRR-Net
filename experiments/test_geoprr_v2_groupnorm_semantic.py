"""CPU-only checks for the new GroupNorm8 conversion and checkpoint contract."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_groupnorm_semantic as gn
from experiments import run_geoprr_v2_groupnorm_semantic as runner


class BatchNormAct2d(nn.BatchNorm2d):
    def __init__(self, channels):
        super().__init__(channels, eps=1e-5)
        self.drop, self.act = nn.Identity(), nn.ReLU()


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = BatchNormAct2d(16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(16, 1280, 1)
        self.norm_head = BatchNormAct2d(1280)
        self.num_features = 16

    def forward(self, image):
        feature = self.bn1(self.conv_stem(image))
        return self.norm_head(self.conv_head(self.global_pool(feature))).flatten(1)


def toy_model():
    encoder = ToyEncoder()
    gn.replace_batch_norm(encoder)
    return gn.SemanticDirectReader(encoder)


def checkpoint_fixture():
    validation = {name:{"samples":count, "nmae":.2, "tail10_nmae":.4}
                  for name,count in (("original",1506),("roll",512))}
    score = gn.selection_score(validation)
    history = [{"epoch":1, "selection_score":score, "validation":validation,
                "optimizer_updates":546, "amp_skipped_updates":1}]
    totals = {"train_batches":547, "train_cnn_image_forwards":34974, "train_cnn_forward_calls":547,
              "backward_calls":547, "validation_images":2018, "validation_cnn_forward_calls":32,
              "train_point_forward_calls":547, "train_point_image_forwards":34974,
              "validation_point_forward_calls":32, "validation_point_image_forwards":2018,
              "optimizer_updates":546, "amp_skipped_updates":1}
    contract = {"RGB_input_contract":deepcopy(gn.RGB_INPUT_CONTRACT)}
    saved = {"protocol":gn.PROTOCOL, "config":deepcopy(gn.CONFIG),
             "normalization_contract":deepcopy(gn.NORMALIZATION_CONTRACT), "data_contract":contract,
             "model":{f"learned_{i}":torch.tensor(0.) for i in range(233)},
             "optimizer":{}, "scheduler":{}, "scaler":{}, "rng":{}, "history":history,
             "totals":totals, "validation_rows":{"original":[{}]*1506,"roll":[{}]*512},
             "initialization":{}, "checkpointed_process_seconds":1., "epoch":1, "best_score":score, "best_epoch":1}
    return saved,contract


class GroupNormSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_conversion_retains_all_learned_names_values_and_affine_objects(self):
        encoder = ToyEncoder()
        with torch.no_grad():
            encoder.bn1.weight.uniform_(.3, 1.7)
            encoder.bn1.bias.uniform_(-.2, .2)
        learned = {key:value.detach().clone() for key,value in encoder.named_parameters()}
        affine = encoder.bn1.weight, encoder.bn1.bias
        drop,act = encoder.bn1.drop,encoder.bn1.act
        before = sum(parameter.numel() for parameter in encoder.parameters())
        self.assertEqual(gn.replace_batch_norm(encoder),2)
        self.assertEqual(set(learned),set(dict(encoder.named_parameters())))
        self.assertEqual(before,sum(parameter.numel() for parameter in encoder.parameters()))
        for key,value in encoder.named_parameters():
            self.assertTrue(torch.equal(value,learned[key]))
        self.assertIs(encoder.bn1.weight,affine[0]); self.assertIs(encoder.bn1.bias,affine[1])
        self.assertIs(encoder.bn1.drop,drop); self.assertIs(encoder.bn1.act,act)
        self.assertEqual((encoder.bn1.num_groups,encoder.bn1.eps),(8,1e-5))
        self.assertEqual((list(encoder.buffers()),list(encoder.bn1.math.parameters()),encoder.bn1.math.state_dict()),([],[],{}))
        self.assertFalse(any(isinstance(module,nn.modules.batchnorm._BatchNorm) for module in encoder.modules()))
        self.assertEqual(set(encoder.state_dict()),set(learned))

    def test_conversion_preserves_mode_and_rejects_an_unfixed_group_or_epsilon(self):
        encoder = ToyEncoder().eval()
        gn.replace_batch_norm(encoder)
        self.assertFalse(encoder.bn1.training)
        self.assertFalse(encoder.norm_head.training)
        invalid = ToyEncoder(); invalid.bn1.eps = 1e-4
        with self.assertRaises(ValueError): gn.replace_batch_norm(invalid)
        invalid = ToyEncoder(); invalid.bn1 = BatchNormAct2d(10)
        with self.assertRaises(ValueError): gn.replace_batch_norm(invalid)

    def test_float32_reference_formula_casts_back_before_original_drop_then_act(self):
        events = []
        class Drop(nn.Module):
            def forward(self, feature):
                events.append(("drop",feature.dtype))
                return feature + .25
        class Act(nn.Module):
            def forward(self, feature):
                events.append(("act",feature.dtype))
                return feature.square()
        old = BatchNormAct2d(16)
        old.drop,old.act = Drop(),Act()
        with torch.no_grad():
            old.weight.uniform_(.3,1.7); old.bias.uniform_(-.2,.2)
        module = gn.GroupNormAct2d(old)
        feature = torch.randn(3,16,5,7).half()
        grouped = feature.float().reshape(3,8,-1)
        centered = grouped-grouped.mean(-1,keepdim=True)
        normalized = centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+1e-5)
        affine = normalized.reshape_as(feature)*module.weight[None,:,None,None]+module.bias[None,:,None,None]
        expected = (affine.to(feature.dtype)+.25).square()
        observed = []
        hook = module.math.register_forward_hook(lambda _module,inputs,output:observed.append(tuple(value.dtype for value in (*inputs,output))))
        try:
            with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
                actual = module(feature)
        finally:
            hook.remove()
        torch.testing.assert_close(actual,expected,rtol=2e-3,atol=2e-3)
        self.assertEqual(observed,[(torch.float32,)*4])
        self.assertEqual(events,[("drop",torch.float16),("act",torch.float16)])
        self.assertEqual(actual.dtype,feature.dtype)

    def test_per_image_group_statistics_are_mode_and_batch_independent(self):
        module = gn.GroupNormAct2d(BatchNormAct2d(16))
        images = torch.randn(3,16,7,5)
        outputs = []
        with torch.inference_mode():
            for training in (True,False):
                module.train(training)
                full = module(images)
                single = module(images[1:2])
                reordered = module(images[[2,1,0]])
                torch.testing.assert_close(full[1:2],single,rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(full,reordered[[2,1,0]],rtol=1e-6,atol=1e-7)
                outputs.append(full)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        self.assertEqual(list(module.named_buffers()),[])

    def test_rgb_only_scalar_model_supports_b1_b3_and_partial_batch_consistency(self):
        model = toy_model()
        images = torch.randn(3,3,32,32)
        for training in (True,False):
            model.train(training)
            with torch.inference_mode():
                full = model(images)
                single = model(images[1:2])
            self.assertEqual((tuple(full.shape),tuple(single.shape)),((3,),(1,)))
            self.assertEqual(full.dtype,torch.float32)
            self.assertTrue(bool(torch.isfinite(full).all()))
            self.assertTrue(bool(((full>0)&(full<1)).all()))
            torch.testing.assert_close(full[1:2],single,rtol=1e-6,atol=1e-7)
        self.assertEqual(list(inspect.signature(model.forward).parameters),["image"])
        with self.assertRaises(TypeError): model(images,target=torch.zeros(3))

    def test_stem_neck_norm_affines_and_point_receive_finite_gradients(self):
        model = toy_model().train()
        before = {key:value.clone() for key,value in model.named_parameters()}
        prediction = model(torch.randn(3,3,32,32))
        F.smooth_l1_loss(prediction,torch.tensor([.13,.21,.28]),beta=.05).backward()
        finite,nonzero = gn.gradient_evidence(model)
        self.assertTrue(finite)
        self.assertTrue(all(nonzero.values()),nonzero)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(torch.equal(value,before[key]) for key,value in model.named_parameters()))
        self.assertEqual(list(model.named_buffers()),[])

    def test_real_new_model_has233_learned_states_and_strict_reload_without_forward(self):
        model = gn.create_model(pretrained=False)
        metadata = gn.model_metadata(model)
        self.assertEqual((metadata["parameters"],metadata["encoder_parameters"]),(8435793,8434512))
        self.assertEqual((metadata["state_tensors"],metadata["parameter_tensors"],metadata["encoder_state_tensors"]),(233,233,231))
        self.assertEqual((metadata["GroupNorm_modules"],metadata["BatchNorm_modules"],metadata["buffer_tensors"]),(77,0,0))
        self.assertEqual((metadata["GroupNorm_groups"],metadata["GroupNorm_epsilons"]),([8],[1e-5]))
        self.assertTrue(metadata["all_parameters_trainable"])
        state = deepcopy(model.state_dict())
        restored = gn.create_model(pretrained=False)
        restored.load_state_dict(state,strict=True)
        self.assertTrue(all(torch.equal(value,restored.state_dict()[key]) for key,value in state.items()))
        old = dict(state); old["encoder.bn1.running_mean"] = torch.zeros(32)
        with self.assertRaises(RuntimeError): restored.load_state_dict(old,strict=True)
        missing = dict(state); del missing["point.weight"]
        with self.assertRaises(RuntimeError): restored.load_state_dict(missing,strict=True)

    def test_resume_requires_new_protocol_normalization_data_state_and_totals(self):
        saved,contract = checkpoint_fixture()
        gn.validate_resume(saved,contract)
        for mutation in (lambda s:s.update(protocol="old_BN_protocol"),
                         lambda s:s["normalization_contract"].update(groups=4),
                         lambda s:s["data_contract"].update(RGB_input_contract={"data":{"image_size":256}}),
                         lambda s:s["model"].update({"encoder.bn1.running_var":torch.zeros(32)}),
                         lambda s:s["totals"].update(optimizer_updates=547)):
            changed = deepcopy(saved); mutation(changed)
            with self.assertRaises(ValueError): gn.validate_resume(changed,contract)
        changed = deepcopy(saved); del changed["rng"]
        with self.assertRaises(ValueError): gn.validate_resume(changed,contract)

    def test_external_checkpoint_rejects_old_bn_states_and_incomplete_source(self):
        saved,_ = checkpoint_fixture()
        saved.update(epoch=60,best_epoch=60)
        completed = {"protocol":gn.PROTOCOL,"config":gn.CONFIG,"normalization_contract":gn.NORMALIZATION_CONTRACT,
                     "status":"complete","epochs":80,"best_epoch":60,"best_score":saved["best_score"],
                     "completion_reason":"source_score_patience"}
        gn.validate_external_checkpoint(saved,completed)
        for count in (464,465):
            changed = deepcopy(saved); changed["model"] = {f"old_{i}":torch.tensor(0.) for i in range(count)}
            with self.assertRaises(ValueError): gn.validate_external_checkpoint(changed,completed)
        changed = deepcopy(completed); changed["status"] = "running"
        with self.assertRaises(ValueError): gn.validate_external_checkpoint(saved,changed)

    def test_saved_preflight_requires_the_exact_gn_precision_and_contract(self):
        saved = {"status":"passed","protocol":gn.PROTOCOL,"config":gn.CONFIG,
                 "normalization_contract":deepcopy(gn.NORMALIZATION_CONTRACT),
                 "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":1,"combined_backward_calls":1,"optimizer_updates":0},
                 "validation_input_matches_declared512_pipeline":True,"all77_GroupNorm_float32_math_then_input_dtype":True,
                 "full_state_tensors":233,"parameter_tensors_unchanged_without_optimizer":233}
        runner.validate_preflight(saved)
        changed = deepcopy(saved); changed["normalization_contract"]["groups"] = 4
        with self.assertRaises(ValueError): runner.validate_preflight(changed)
        changed = deepcopy(saved); changed["all77_GroupNorm_float32_math_then_input_dtype"] = False
        with self.assertRaises(ValueError): runner.validate_preflight(changed)

    def test_partial_source_run_without_latest_stops_before_population_or_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output/"run.json").write_text("{}")
            with patch.object(gn,"source_populations",side_effect=AssertionError("population read after partial output")), \
                    patch.object(gn,"seed_everything"):
                with self.assertRaises(FileExistsError): gn.train(Path("/fixture"),output)

    def test_partial_target_csv_stops_before_loading_or_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output/"predictions.csv").write_text("sample_id\nkept\n")
            with patch.object(torch,"load",side_effect=AssertionError("checkpoint read after partial output")):
                with self.assertRaises(FileExistsError): gn.evaluate(Path("/fixture"),Path("/fixture/best.pt"),"industrial",output)


if __name__ == "__main__":
    unittest.main()
