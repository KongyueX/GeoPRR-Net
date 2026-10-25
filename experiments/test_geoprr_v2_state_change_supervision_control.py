"""Only new synthetic CPU checks for state targets, auxiliary binding and execution contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiments import geoprr_v2_state_change_supervision_control as control
from experiments.geoprr_v2_semantic_direct import SemanticDirectReader


def synthetic_energy():
    energy = np.zeros((4,512,512),dtype=np.float64)
    for state in range(4):
        energy[state,state*32:(state+1)*32,64:96] = state+1
    return energy


class TargetOperatorTests(unittest.TestCase):
    def test_sum_pooling_normalization_preserves_exact_block_mass(self):
        energy = synthetic_energy()
        energy[0,0:32,96:128] = 2
        q,report = control.q_from_state_energy(energy)
        self.assertEqual(q.shape,(4,16,16))
        self.assertEqual(q.dtype,np.float32)
        np.testing.assert_allclose(q.sum(axis=(1,2)),np.ones(4),rtol=0,atol=1e-6)
        self.assertAlmostEqual(float(q[0,0,2]),1/3,places=7)
        self.assertAlmostEqual(float(q[0,0,3]),2/3,places=7)
        self.assertEqual(np.count_nonzero(q[1]),1)
        self.assertEqual(float(q[1,1,2]),1.)
        self.assertGreaterEqual(report["minimum_next_slot_TV"],.05)

    def test_invalid_or_uncontrasted_state_target_stops_without_smoothing(self):
        cases = []
        bad = synthetic_energy(); bad[0] = 0
        cases.append((bad,"each_state_energy"))
        bad = synthetic_energy(); bad[0,0,0] = -1
        cases.append((bad,"D_finite_nonnegative"))
        bad = synthetic_energy(); bad[0,0,0] = np.nan
        cases.append((bad,"D_finite_nonnegative"))
        bad = synthetic_energy(); bad[1] = bad[0]
        cases.append((bad,"each_next_slot_total_variation"))
        cases.append((np.zeros((4,16,16)),"D_shape"))
        for energy,predicate in cases:
            with self.subTest(predicate=predicate),self.assertRaisesRegex(control.SourceObservationRejected,predicate):
                control.q_from_state_energy(energy)

    def test_cyclic_cache_binding_keeps_original_image_scalar_and_row_order(self):
        class Base:
            rows = [{"sample_id":str(i)} for i in range(8)]
            def __len__(self): return 8
            def __getitem__(self,index):
                return {"image":images[index],"y":index/10,"sample_id":str(index)}
        images = [torch.full((3,1,1),float(i)) for i in range(8)]
        cycle = control.next_slot_indices([[0,3,1,2],[4,7,5,6]],8)
        self.assertEqual(cycle,[3,2,0,1,7,6,4,5])
        targets = np.stack([np.full((16,16),i,dtype=np.float32) for i in range(8)])
        base = Base()
        correct = control.StateTargetDataset(base,targets,cycle,arm="correct")
        cyclic = control.StateTargetDataset(base,targets,cycle,arm="cyclic")
        for index in range(8):
            a,b = correct[index],cyclic[index]
            self.assertIs(a["image"],images[index])
            self.assertIs(b["image"],images[index])
            self.assertEqual(a["y"],b["y"])
            self.assertEqual(a["source_row_index"],index)
            self.assertEqual(b["assigned_target_row_index"],cycle[index])
            self.assertEqual(float(a["q"][0,0]),index)
            self.assertEqual(float(b["q"][0,0]),cycle[index])
        with self.assertRaisesRegex(control.SourceObservationRejected,"one_target_binding"):
            control.next_slot_indices([[0,1,2,3],[0,4,5,6]],8)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv2d(3,960,1,bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(960,1280,1,bias=False)
        self.norm_head = nn.Identity()

    def forward(self,image):
        feature = self.conv_stem(image)
        return self.norm_head(self.conv_head(self.global_pool(feature))).flatten(1)


class ToyScalarReader(SemanticDirectReader):
    def __init__(self):
        super().__init__(ToyEncoder())
        self.original_forward_calls = 0

    def forward(self,image):
        self.original_forward_calls += 1
        return super().forward(image)


class AuxiliaryTests(unittest.TestCase):
    def test_zero_auxiliary_has_no_bias_or_rng_draw_and_scalar_forward_is_retained(self):
        reader = ToyScalarReader()
        saved_rng = torch.get_rng_state().clone()
        model = control.StateChangeReader(reader)
        self.assertTrue(torch.equal(saved_rng,torch.get_rng_state()))
        self.assertEqual(tuple(model.aux_weight.shape),(1,960,1,1))
        self.assertEqual(model.aux_weight.numel(),960)
        self.assertEqual(set(model._parameters),{"aux_weight"})
        self.assertIs(model.point,reader.point)
        self.assertEqual(sum(1 for m in model.modules() if m is reader.point),1)
        self.assertTrue(bool((model.aux_weight == 0).all()))
        image = torch.rand(2,3,16,16)
        expected = reader(image)
        self.assertTrue(torch.equal(model(image),expected))
        model.train()
        cost = control.AuxiliaryCost()
        hooks_before = len(reader.encoder.global_pool._forward_pre_hooks)
        prediction,logits,witness = model.forward_with_auxiliary(image,cost)
        self.assertTrue(torch.equal(prediction,expected))
        self.assertEqual(witness["pool_shape"],[2,960,16,16])
        self.assertEqual(tuple(logits.shape),(2,1,16,16))
        self.assertEqual(len(reader.encoder.global_pool._forward_pre_hooks),hooks_before)
        self.assertEqual((cost.calls,cost.maps),(1,2))
        model.eval()
        calls = reader.original_forward_calls
        self.assertTrue(torch.equal(model(image),expected))
        self.assertEqual(reader.original_forward_calls,calls+1)
        self.assertEqual((cost.calls,cost.maps),(1,2))
        self.assertFalse(any(isinstance(value,torch.Tensor) for value in model.__dict__.values()))
        with self.assertRaisesRegex(ValueError,"training-only"):
            model.forward_with_auxiliary(image,cost)

    def test_local_hook_removed_if_original_forward_fails(self):
        reader = ToyScalarReader()
        model = control.StateChangeReader(reader).train()
        before = len(model.encoder.global_pool._forward_pre_hooks)
        with self.assertRaisesRegex(ValueError,"960x16x16"):
            model.forward_with_auxiliary(torch.zeros(1,3,8,8))
        self.assertEqual(len(model.encoder.global_pool._forward_pre_hooks),before)

    def test_outer_autocast_combined_gradient_and_float32_losses(self):
        model = control.StateChangeReader(ToyScalarReader()).train()
        q = torch.from_numpy(control.q_from_state_energy(synthetic_energy())[0][:2])
        image = torch.rand(2,3,16,16)
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
            prediction,logits,_ = model.forward_with_auxiliary(image)
            loss,parts = control.source_objective(prediction,torch.tensor([.1,.9]),logits,q)
        self.assertEqual(logits.dtype,torch.bfloat16)
        self.assertEqual(prediction.dtype,torch.float32)
        self.assertEqual(parts["log_softmax_dtype"],"torch.float32")
        self.assertEqual(parts["q_dtype"],"torch.float32")
        self.assertTrue(all(parts[key].dtype == torch.float32 for key in ("scalar_mean_smooth_l1","auxiliary_mean_KL","composite")))
        loss.backward()
        for parameter in (model.encoder.conv_stem.weight,model.encoder.conv_head.weight,model.point.weight,model.aux_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()),0)

    def test_uniform_initial_KL_tie_allows_different_state_bound_gradients(self):
        features = torch.zeros(2,960,16,16)
        features[0,0,0,0] = 2
        features[0,0,0,1] = -2
        features[1,0,0,0] = -2
        features[1,0,0,1] = 2
        q = torch.zeros(2,16,16)
        q[0,0,0] = 1; q[1,0,1] = 1
        gradients,losses = [],[]
        for assigned in (q,q[[1,0]]):
            weight = nn.Parameter(torch.zeros(1,960,1,1))
            logits = torch.nn.functional.conv2d(features,weight)
            loss,parts = control.source_objective(torch.tensor([.2,.8]),torch.tensor([.2,.8]),logits,assigned)
            loss.backward()
            gradients.append(weight.grad.clone())
            losses.append(float(parts["auxiliary_mean_KL"]))
        self.assertEqual(losses[0],losses[1])
        self.assertAlmostEqual(losses[0],np.log(256),places=6)
        self.assertTrue(torch.allclose(gradients[0],-gradients[1]))
        self.assertGreater(float(gradients[0].abs().sum()),0)
        with self.assertRaisesRegex(ValueError,"normalized16x16"):
            control.source_objective(torch.tensor([.2,.8]),torch.tensor([.2,.8]),torch.zeros(2,1,16,16),q*2)


def one_epoch_checkpoint():
    totals = control.initial_totals()
    totals.update(train_batches=86,train_cnn_image_forwards=5496,train_cnn_forward_calls=86,
        backward_calls=86,point_backward_calls=86,train_point_forward_calls=86,train_point_image_forwards=5496,
        optimizer_updates=85,amp_skipped_updates=1,auxiliary_forward_calls=86,auxiliary_map_forwards=5496,
        auxiliary_backward_calls=86,combined_backward_calls=86,training_seconds=3.)
    history = [{"epoch":1,"learning_rate":.0003,"train_counts":{"cnn_forward_calls":86,"cnn_image_forwards":5496},
        "train_point_counts":{"point_forward_calls":86,"point_image_forwards":5496},
        "CNN_backward_calls":86,"point_backward_calls":86,"auxiliary_backward_calls":86,"combined_backward_calls":86,
        "auxiliary_cost":{"auxiliary_calls":86,"auxiliary_maps":5496},"optimizer_updates":85,"amp_skipped_updates":1,
        "last_batch_size":56,"validation_calls":0,"cost":{"seconds":3.},
        "all_source_rows_once":True,"assigned_targets_match_arm":True,
        "source_training_scalar_mean_smooth_l1":.1,"source_training_auxiliary_mean_KL":1.,
        "source_training_composite":.1+.05/np.log(256)}]
    return {"protocol":control.PROTOCOL,"config":deepcopy(control.CONFIG),"source_contract":{"toy_source":True},
        "target_contract":{"toy_target":True},"arm":"correct","phase":"training","epoch":1,
        "model":{"aux_weight":torch.zeros(1,960,1,1),"reader.toy":torch.zeros(2)},
        "optimizer":{"param_groups":[{"lr":.0003,"params":[0,1]}],"state":{}},
        "scaler":{"scale":65536.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":0},
        "rng":{"python":None,"numpy":None,"torch":None,"cuda":None},"history":history,"totals":totals,
        "initialization":{"toy":True},"model_metadata":{"toy":True},"phases":{},"checkpointed_process_seconds":4.}


class ExecutionContractTests(unittest.TestCase):
    def test_new_cost_counts_and_conditional_negative_control_stages(self):
        totals = control.initial_totals()
        for epoch in range(1,16):
            for key,increment in {"train_batches":86,"train_cnn_image_forwards":5496,"train_cnn_forward_calls":86,
                    "backward_calls":86,"point_backward_calls":86,"train_point_forward_calls":86,
                    "train_point_image_forwards":5496,"optimizer_updates":85,"amp_skipped_updates":1,
                    "auxiliary_forward_calls":86,"auxiliary_map_forwards":5496,
                    "auxiliary_backward_calls":86,"combined_backward_calls":86}.items():
                totals[key] += increment
            control.validate_training_totals(totals,epoch)
        for cf,images,calls in ((False,82732,1295),(True,83024,1300)):
            observed = deepcopy(totals)
            n = 2 if cf else 1
            observed.update(validation_images=292*n,validation_cnn_forward_calls=5*n,
                validation_point_forward_calls=5*n,validation_point_image_forwards=292*n)
            control.validate_totals(observed,cf)
            counts = control.count_summary(observed)
            self.assertEqual((counts["cnn_image_forwards"],counts["cnn_forward_calls"]),(images,calls))
            self.assertEqual((counts["auxiliary_map_forwards"],counts["auxiliary_forward_calls"],counts["combined_backward_calls"]),(82440,1290,1290))
            self.assertEqual(counts["optimizer_updates"]+counts["amp_skipped_updates"],1290)
        for learning,joint in ((False,False),(False,True),(True,False),(True,True)):
            a = {"passed":learning}; b = {"passed":joint}
            self.assertEqual(control.conditional_stages("correct",a,b)["counterfactual"],learning and joint)
            self.assertTrue(control.conditional_stages("cyclic",a,b)["counterfactual"])
        self.assertEqual(control.PREFLIGHT_COUNTS["RGB_decode_calls"],12)
        self.assertEqual(control.PREFLIGHT_COUNTS["CNN_images"],8)
        self.assertFalse(control.compare_arms({"cyclic_eligible":False},None)["eligible"])
        with self.assertRaisesRegex(ValueError,"rejection"):
            control.compare_arms({"cyclic_eligible":False},{})

    def test_strict_full_state_save_reload_and_complete_epoch_resume(self):
        saved = one_epoch_checkpoint()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"new.pt"
            torch.save(saved,path)
            restored = torch.load(path,map_location="cpu",weights_only=False)
        control.validate_resume(restored,{"toy_source":True},{"toy_target":True},"correct",saved["model"],saved["optimizer"],saved["scaler"])
        mutations = {
            "old_protocol":lambda row:row.update(protocol="old464"),
            "config":lambda row:row["config"]["training"].update(epochs=16),
            "source":lambda row:row.update(source_contract={}),
            "target":lambda row:row.update(target_contract={}),
            "arm":lambda row:row.update(arm="cyclic"),
            "partial_epoch":lambda row:row.update(phase="active_epoch"),
            "history":lambda row:row.update(history=[]),
            "auxiliary_count":lambda row:row["totals"].update(auxiliary_forward_calls=85),
            "old464_model":lambda row:row["model"].pop("aux_weight"),
            "missing_scaler":lambda row:row.pop("scaler"),
            "optimizer":lambda row:row["optimizer"]["param_groups"][0].update(lr=.01),
            "default_scaler":lambda row:row["scaler"].update(growth_interval=100),
        }
        for label,mutate in mutations.items():
            changed = deepcopy(saved); mutate(changed)
            with self.subTest(label=label),self.assertRaises(ValueError):
                control.validate_resume(changed,{"toy_source":True},{"toy_target":True},"correct",saved["model"],saved["optimizer"],saved["scaler"])

    def test_partial_scalar_artifact_never_authorizes_another_forward(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            self.assertIsNone(control.load_evaluation(output,"source_id",{}, {},"correct",(0,1,2)))
            (output/"source_id_predictions.csv").write_text("sample_id\n0\n")
            with self.assertRaisesRegex(FileExistsError,"second forward"):
                control.load_evaluation(output,"source_id",{}, {},"correct",(0,1,2))

    def test_partial_preparation_cannot_be_overwritten_or_bypass_precheck(self):
        with tempfile.TemporaryDirectory() as name:
            home = Path(name)
            output = home/control.OUTPUT_RELATIVE/"preparation"
            output.mkdir(parents=True)
            (output/"partial.json").write_text("{}")
            with patch.object(control,"precheck",return_value={}),patch.object(control,"read_packet") as read_packet:
                with self.assertRaisesRegex(FileExistsError,"cannot be replaced"):
                    control.prepare(home,output)
                read_packet.assert_not_called()
            with patch.object(control,"precheck",side_effect=FileNotFoundError("required snapshot")):
                with self.assertRaisesRegex(FileNotFoundError,"required snapshot"):
                    control.prepare(home,output)
            self.assertEqual((output/"partial.json").read_text(),"{}")

    def test_new_protocol_target_contract_has_no_implicit_old_control_compatibility(self):
        self.assertNotEqual(control.PROTOCOL,control.shared.PROTOCOL)
        self.assertNotEqual(control.PROTOCOL,control.identity.PROTOCOL)
        self.assertEqual(control.CONFIG["model"]["parameters"],8436753)
        self.assertEqual(control.CONFIG["model"]["parameter_tensors"],234)
        self.assertEqual(control.CONFIG["model"]["state_tensors"],465)
        self.assertEqual(control.TARGET_FORMULA["payload_bytes"],5496*16*16*4)
        self.assertEqual(control.CONFIG["training"]["epochs"],15)
        self.assertIsNone(control.CONFIG["training"]["scheduler"])
        self.assertFalse(control.CONFIG["acceptance"]["cyclic_if_correct_fails"])
        self.assertEqual(json.loads(json.dumps(control.CONFIG)),control.CONFIG)


if __name__ == "__main__":
    unittest.main()
