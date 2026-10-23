"""CPU checks for roll-source exposure, selection, resumption and frozen scoring."""
from collections import Counter
from copy import deepcopy
import unittest

import torch
from torch.utils.data import Dataset

from experiments.geoprr_v2_roll_source import (
    CONFIG, PROTOCOL, completion_reason, selection_score, training_population,
    validate_external_checkpoint, validate_resume, validate_roll_split,
)
from experiments.geoprr_v2_semantic_direct import scalar_rows
from experiments.evaluate_geoprr_v2_role_external import summarize
from experiments.train_geoprr_v2 import make_loader, source_split_check


class IDDataset(Dataset):
    def __init__(self, prefix, count):
        self.prefix, self.count = prefix, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return f"{self.prefix}_{index}"


def validation(original, roll):
    return {"original":{"samples":1506,"nmae":original}, "roll":{"samples":512,"nmae":roll}}


def saved_state():
    scores = [validation(.4,.2), validation(.2,.2), validation(.2,.2)]
    return {"protocol":PROTOCOL, "config":deepcopy(CONFIG), "data_contract":{"source":"fixed"},
            "model":{}, "optimizer":{}, "scheduler":{}, "scaler":{}, "rng":{}, "totals":{},
            "initialization":{}, "checkpointed_process_seconds":1., "epoch":3, "best_score":.2, "best_epoch":2,
            "history":[{"epoch":i+1,"validation":v,"selection_score":selection_score(v)} for i,v in enumerate(scores)],
            "validation_rows":{"original":[{}]*1506, "roll":[{}]*512}}


class RollSourceTests(unittest.TestCase):
    def test_real_loader_old_once_roll_four_times_without_dropping_last30(self):
        population = training_population(IDDataset("old",18590), IDDataset("roll",4096))
        loader = make_loader(population,batch_size=64,workers=0,shuffle=True,seed=20262021,device=torch.device("cpu"))
        batches = list(loader)
        ids = [sid for batch in batches for sid in batch]
        counts = Counter(ids)
        self.assertEqual((len(ids),len(counts),len(batches),len(batches[-1])),(34974,22686,547,30))
        self.assertTrue(all(counts[f"old_{i}"] == 1 for i in range(18590)))
        self.assertTrue(all(counts[f"roll_{i}"] == 4 for i in range(4096)))
        repeat = make_loader(population,batch_size=64,workers=0,shuffle=True,seed=20262021,device=torch.device("cpu"))
        self.assertEqual(batches[0],next(iter(repeat)))
        changed = make_loader(population,batch_size=64,workers=0,shuffle=True,seed=20262022,device=torch.device("cpu"))
        self.assertNotEqual(batches[0],next(iter(changed)))

    def test_equal_population_source_score_and_minimum_stop_boundaries(self):
        self.assertAlmostEqual(selection_score(validation(.1,.7)),.4,places=15)
        with self.assertRaises(ValueError): selection_score({**validation(.1,.2),"industrial":{"samples":1395,"nmae":0.}})
        partial = validation(.1,.2); partial["roll"]["samples"] = 511
        with self.assertRaises(ValueError): selection_score(partial)
        self.assertIsNone(completion_reason(59,1))
        self.assertEqual(completion_reason(60,40),"source_score_patience")
        self.assertIsNone(completion_reason(60,41))
        self.assertEqual(completion_reason(61,41),"source_score_patience")
        self.assertEqual(completion_reason(160,160),"maximum_epochs")

    def test_plateau_scheduler_strict_ties_and_state_restore(self):
        optimizer = torch.optim.AdamW([torch.tensor(0.)],lr=.0003,weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**CONFIG["training"]["scheduler"])
        for _ in range(9): scheduler.step(.2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.0003)
        scheduler.step(.2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.00009)
        resumed_optimizer = torch.optim.AdamW([torch.tensor(0.)],lr=.0003,weight_decay=.0001)
        resumed_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resumed_optimizer,**CONFIG["training"]["scheduler"])
        resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        resumed_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
        for value in [.2]*9+[.1]+[.1]*18:
            scheduler.step(value); resumed_scheduler.step(value)
            self.assertEqual(optimizer.param_groups[0]["lr"],resumed_optimizer.param_groups[0]["lr"])
        self.assertGreaterEqual(optimizer.param_groups[0]["lr"],1e-6)

    def test_resume_keeps_first_tied_best_and_requires_full_source_state(self):
        state = saved_state()
        validate_resume(state,{"source":"fixed"})
        tied = deepcopy(state); tied["best_epoch"] = 3
        with self.assertRaises(ValueError): validate_resume(tied,{"source":"fixed"})
        for key in ("optimizer","scheduler","scaler","rng"):
            missing = deepcopy(state); del missing[key]
            with self.assertRaises(ValueError): validate_resume(missing,{"source":"fixed"})
        changed = deepcopy(state); changed["config"]["source"]["roll_repeat"] = 1
        with self.assertRaises(ValueError): validate_resume(changed,{"source":"fixed"})
        with self.assertRaises(ValueError): validate_resume(state,{"source":"changed"})
        missing_rows = deepcopy(state); missing_rows["validation_rows"]["roll"].pop()
        with self.assertRaises(ValueError): validate_resume(missing_rows,{"source":"fixed"})

    def test_shared_roll_background_is_explicit_and_original_scene_guard_still_rejects_overlap(self):
        def row(sid,group,split,scene,seed):
            return {"sample_id":sid,"group_id":group,"dataset":"syncg","split":split,
                    "metadata":{"scene_name":scene,"roll_source":{"layout_id":group,"render_seed":seed}}}
        train = [row("t","trainlayout","train","real_train_hdr",10)]
        roll_val = [row("v","vallayout","roll_validation","real_train_hdr",20)]
        validate_roll_split(train,roll_val)
        native_val = [row("oldv","oldvalgroup","train","heldout_hdr",30)]
        source_split_check(train,native_val)
        overlap = deepcopy(roll_val); overlap[0]["metadata"]["roll_source"]["render_seed"] = 10
        with self.assertRaises(ValueError): validate_roll_split(train,overlap)
        native_val[0]["metadata"]["scene_name"] = "real_train_hdr"
        with self.assertRaises(ValueError): source_split_check(train,native_val)

    def test_failed_external_scalars_keep_the_full_original_denominator(self):
        batch = {"sample_id":["a","b","c","d"],"group_id":["g"]*4,
                 "y":torch.tensor([.1,.2,.3,.4]),"input_failure":["","missing_RGB","",""]}
        rows = scalar_rows(torch.tensor([.2,.2,float("nan"),1.1]),batch)
        summary = summarize(rows)
        self.assertEqual([r["sample_id"] for r in rows],batch["sample_id"])
        self.assertEqual((summary["samples"],summary["failed"],summary["input_failures"]),(4,3,1))
        self.assertEqual([r["absolute_error"] for r in rows[1:]],[1.,1.,1.])
        self.assertAlmostEqual(summary["nmae"],.775,places=7)
        self.assertEqual(summary["coverage"],.25)

    def test_external_dispatch_rejects_unfinished_source_and_nonselected_checkpoint(self):
        checkpoint = {"protocol":PROTOCOL,"config":CONFIG,"epoch":20,"best_epoch":20,"best_score":.2}
        completed = {"protocol":PROTOCOL,"config":CONFIG,"status":"complete","epochs":60,
                     "best_epoch":20,"best_score":.2,"completion_reason":"source_score_patience"}
        validate_external_checkpoint(checkpoint,completed)
        latest = {**checkpoint,"epoch":60}
        with self.assertRaises(ValueError): validate_external_checkpoint(latest,completed)
        unfinished = {**completed,"best_epoch":50,"completion_reason":None}
        with self.assertRaises(ValueError): validate_external_checkpoint({**checkpoint,"epoch":50,"best_epoch":50},unfinished)
        with self.assertRaises(ValueError): validate_external_checkpoint({**checkpoint,"protocol":"old_v16"},completed)


if __name__ == "__main__":
    unittest.main()
