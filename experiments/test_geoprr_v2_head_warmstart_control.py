"""CPU toy checks for the new LP/FT phase, RNG, cache and accounting boundaries."""
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_head_warmstart_control import (
    CONFIG, SEEN, count_summary, detached_semantic, enter_finetuning,
    evaluate_conditional_ood, initial_totals, lp_mode, new_optimizer,
    point_prediction, preflight_indices, primary_comparison, state_equal, validate_totals,
)
from experiments.train_geoprr_v2 import restore_rng, rng_state


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Linear(3,4)
        self.bn = nn.BatchNorm1d(4)
        self.conv_head = nn.Linear(4,4)
        self.norm_head = nn.BatchNorm1d(4)

    def forward(self,image):
        return self.norm_head(self.conv_head(self.bn(self.conv_stem(image)).tanh()))


class ToyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()
        self.point = nn.Linear(4,1)
        nn.init.normal_(self.point.weight,std=.05)
        nn.init.zeros_(self.point.bias)

    def forward(self,image):
        return point_prediction(self,self.encoder(image))


def toy_batch():
    return torch.tensor([[.1,.5,.3],[.8,.2,.4],[.6,.7,.9],[.2,.9,.1],[.9,.3,.7],[.4,.6,.2]]),torch.tensor([.1,.9,.3,.8,.2,.7])


def snapshot(module):
    return {key:value.detach().clone() for key,value in module.state_dict().items()}


def complete_totals(passed):
    values = initial_totals()
    values.update(cache_cnn_image_forwards=5496,cache_cnn_forward_calls=86,
        lp_point_forward_calls=100,lp_point_image_forwards=549600,lp_point_backward_calls=100,
        lp_optimizer_updates=99,lp_amp_skipped_updates=1,train_batches=430,
        train_cnn_image_forwards=27480,train_cnn_forward_calls=430,backward_calls=430,
        optimizer_updates=429,amp_skipped_updates=1,train_point_forward_calls=430,train_point_image_forwards=27480,
        validation_images=512 if passed else 292,validation_cnn_forward_calls=9 if passed else 5,
        validation_point_forward_calls=9 if passed else 5,validation_point_image_forwards=512 if passed else 292)
    return values


class PhaseBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ToyReader()
        self.image,self.target = toy_batch()

    def test_lp_freezes_every_encoder_state_and_backward(self):
        before = snapshot(self.model.encoder)
        lp_mode(self.model)
        features = detached_semantic(self.model,self.image)
        loss = F.smooth_l1_loss(point_prediction(self.model,features),self.target,beta=.05)
        (loss*128.).backward()
        for parameter in self.model.point.parameters():
            parameter.grad.div_(128.)
        self.assertTrue(state_equal(self.model.encoder,before))
        self.assertTrue(all(not module.training for module in self.model.encoder.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.encoder.parameters()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.model.point.parameters()))
        self.assertTrue(any(bool(p.grad.abs().sum()>0) for p in self.model.point.parameters()))

    def test_cache_detaches_an_existing_graph_from_encoder(self):
        self.model.eval()
        features = detached_semantic(self.model,self.image.requires_grad_())
        self.assertFalse(features.requires_grad)
        self.assertIsNone(features.grad_fn)
        point_prediction(self.model,features).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
        self.assertIsNone(self.image.grad)
        self.assertIsNotNone(self.model.point.weight.grad)

    def test_ft_restores_bn_and_all_gradients_after_clearing_lp(self):
        before_parameters = {key:value.detach().clone() for key,value in self.model.named_parameters()}
        lp_mode(self.model)
        F.smooth_l1_loss(point_prediction(self.model,detached_semantic(self.model,self.image)),self.target,beta=.05).backward()
        enter_finetuning(self.model,None,lambda _state:None)
        self.assertTrue(all(p.requires_grad and p.grad is None for p in self.model.parameters()))
        before_bn = {key:value.clone() for key,value in self.model.state_dict().items() if key.endswith("num_batches_tracked")}
        loss = F.smooth_l1_loss(self.model(self.image),self.target,beta=.05)
        (loss*128.).backward()
        for parameter in self.model.parameters():
            parameter.grad.div_(128.)
        self.assertTrue(all(torch.equal(self.model.state_dict()[key],value+1) for key,value in before_bn.items()))
        for module in (self.model.encoder.conv_stem,self.model.encoder.conv_head,self.model.encoder.norm_head,self.model.point):
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))
            self.assertTrue(any(bool(p.grad.abs().sum()>0) for p in module.parameters()))
        self.model.eval()
        before_eval = snapshot(self.model)
        with torch.no_grad():
            prediction = self.model(self.image)
        self.assertTrue(state_equal(self.model,before_eval))
        self.assertEqual(prediction.dtype,torch.float32)
        self.assertTrue(all(torch.equal(value,before_parameters[key]) for key,value in self.model.named_parameters()))

    def test_same_learned_head_fresh_optimizer_and_initialization_rng(self):
        random.seed(17)
        np.random.seed(17)
        initial_rng = rng_state()
        initial_head = snapshot(self.model.point)
        lp_mode(self.model)
        lp_optimizer = new_optimizer(self.model,point_only=True)
        features = detached_semantic(self.model,self.image)
        F.smooth_l1_loss(point_prediction(self.model,features),self.target,beta=.05).backward()
        lp_optimizer.step()
        self.assertFalse(state_equal(self.model.point,initial_head))
        learned_head = snapshot(self.model.point)
        self.assertEqual({id(p) for g in lp_optimizer.param_groups for p in g["params"]},{id(p) for p in self.model.point.parameters()})
        self.assertTrue(lp_optimizer.state)
        restore_rng(initial_rng)
        expected = (random.random(),np.random.random(),torch.rand(4))
        random.random(),np.random.random(),torch.rand(19)
        enter_finetuning(self.model,initial_rng,restore_rng)
        ft_optimizer = new_optimizer(self.model,point_only=False)
        actual = (random.random(),np.random.random(),torch.rand(4))
        self.assertEqual(actual[0],expected[0])
        self.assertEqual(actual[1],expected[1])
        self.assertTrue(torch.equal(actual[2],expected[2]))
        self.assertTrue(state_equal(self.model.point,learned_head))
        self.assertIsNot(ft_optimizer,lp_optimizer)
        self.assertFalse(ft_optimizer.state)
        self.assertEqual({id(p) for g in ft_optimizer.param_groups for p in g["params"]},{id(p) for p in self.model.parameters()})
        self.assertEqual((ft_optimizer.param_groups[0]["lr"],ft_optimizer.param_groups[0]["weight_decay"]),(.0003,.0001))


class NewAccountingTests(unittest.TestCase):
    def test_both_gate_paths_keep_cnn_and_point_budgets_distinct(self):
        for passed,expected in ((False,(33268,521,577372,535)),(True,(33488,525,577592,539))):
            values = complete_totals(passed)
            validate_totals(values,passed)
            counts = count_summary(values)
            self.assertEqual(tuple(counts[key] for key in ("cnn_image_forwards","cnn_forward_calls","point_image_forwards","point_forward_calls")),expected)
            self.assertEqual((counts["source_backward_calls"],counts["CNN_backward_calls"],counts["point_only_backward_calls"]),(530,430,100))

    def test_extra_or_missing_lp_work_and_ood_on_failed_gate_rejected(self):
        for key in ("cache_cnn_forward_calls","cache_point_forward_calls","lp_point_forward_calls","lp_point_image_forwards",
                    "lp_point_backward_calls","lp_CNN_backward_calls","lp_optimizer_updates"):
            values = complete_totals(False)
            values[key] += 1
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_totals(values,False)
        with self.assertRaises(ValueError):
            validate_totals(complete_totals(True),False)

    def test_primary_result_never_replaces_original_ood_gate(self):
        gate = {"passed":False,"model":{"nmae":0.,"tail10_nmae":0.}}
        self.assertFalse(primary_comparison(gate)["retain_this_warmstart_candidate"])
        gate["passed"] = True
        gate["model"].update(nmae=CONFIG["primary_comparison"]["baseline_mean"],tail10_nmae=CONFIG["primary_comparison"]["baseline_tail"])
        self.assertFalse(primary_comparison(gate)["retain_this_warmstart_candidate"])
        gate["model"]["tail10_nmae"] -= .001
        self.assertTrue(primary_comparison(gate)["retain_this_warmstart_candidate"])


class ConditionalSourcePathTests(unittest.TestCase):
    def test_failed_gate_never_imports_or_constructs_ood_dataset(self):
        module = ModuleType("experiments.geoprr_v2_high_resolution_direct")
        def forbidden(*_args,**_kwargs):
            self.fail("failed original gate constructed or evaluated a held-out source dataset")
        module.ScalarRGBDataset = forbidden
        with patch.dict(sys.modules,{module.__name__:module}):
            result = evaluate_conditional_ood({"passed":False},Path("/unused"),{},forbidden)
        self.assertIsNone(result)

    def test_passed_gate_opens_only_new_runs_source_ood_once(self):
        module = ModuleType("experiments.geoprr_v2_high_resolution_direct")
        constructed,evaluated = [],[]
        dataset = object()
        def constructor(*args,**kwargs):
            constructed.append((args,kwargs))
            return dataset
        def evaluate(name,value,images,calls):
            evaluated.append((name,value,images,calls))
            return [],{"source_only":True}
        module.ScalarRGBDataset = constructor
        prepared = {"manifests":{"ood":"manifests/source_ood.jsonl"}}
        with patch.dict(sys.modules,{module.__name__:module}):
            result = evaluate_conditional_ood({"passed":True},Path("/unused"),prepared,evaluate)
        self.assertEqual(len(constructed),1)
        self.assertEqual(evaluated,[("source_ood",dataset,220,4)])
        self.assertEqual(result,{"source_only":True})

    def test_new_preflight_lp_original0_ft_original_roll3_and_old_id_first(self):
        rows = []
        for alias in SEEN:
            parent = "parent_"+alias
            for view in ("original","roll"):
                for slot in range(4):
                    sample = f"{alias}_{view}_{slot}"
                    rows.append({"sample_id":sample,"roi_path":"/unused/"+sample,
                        "metadata":{"needle_control":{"pointer":alias,"view":view,"slot":slot,
                            "layout":parent if view == "original" else "roll_"+parent,"parent_layout":parent}}})
        id_rows = [{"sample_id":str(i),"roi_path":f"/unused/id{i}"} for i in range(4)]
        prepared = {"source_contract":{"populations":{"train":rows,"id":id_rows}},"pref_indices":{"id":[[0,1],[2,3]]}}
        before = deepcopy(prepared)
        indices = preflight_indices(prepared)
        self.assertEqual(indices["id"],[0,2])
        self.assertEqual([(rows[i]["metadata"]["needle_control"]["view"],rows[i]["metadata"]["needle_control"]["slot"]) for i in indices["lp"]],[("original",0)]*2)
        self.assertEqual([(rows[i]["metadata"]["needle_control"]["view"],rows[i]["metadata"]["needle_control"]["slot"]) for i in indices["ft"]],[("original",3),("roll",3)]*2)
        self.assertFalse(set(indices["lp"]) & set(indices["ft"]))
        self.assertEqual(prepared,before)


if __name__ == "__main__":
    unittest.main()
