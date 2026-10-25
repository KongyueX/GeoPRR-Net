"""CPU toys for the new color operator, exposure, conditional gates and resume contract."""
from copy import deepcopy
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.geoprr_v2_cyclic_color_control import (
    CONFIG, COUNTERFACTUAL_ORDER, MEAN, NATIVE_ORDER, PROTOCOL, STD, TRAIN_ORDERS,
    ChannelOrderDataset, conditional_counterfactual, count_summary, counterfactual_gate,
    initial_totals, load_evaluation, native_primary, normalized_channel_permutation,
    training_order_index, validate_resume, validate_totals,
)


def normalized(raw):
    shape = (3,1,1) if raw.ndim == 3 else (1,3,1,1)
    return (raw-raw.new_tensor(MEAN).reshape(shape))/raw.new_tensor(STD).reshape(shape)


class ToyDataset:
    def __init__(self):
        self.rows = [{"sample_id":"toy0"},{"sample_id":"toy1"},{"sample_id":"toy2"}]
        self.calls = []
        self.image = normalized(torch.arange(12,dtype=torch.float32).reshape(3,2,2)/11)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        self.calls.append(index)
        return {"image":self.image,"y":.25,"sample_id":self.rows[index]["sample_id"],"group_id":"toy"}


class ColorOperatorTests(unittest.TestCase):
    def test_all_permutations_match_raw_RGB_then_original_normalization(self):
        raw = torch.arange(24,dtype=torch.float32).reshape(2,3,2,2)/13-.2
        image = normalized(raw)
        for order in (*TRAIN_ORDERS,COUNTERFACTUAL_ORDER,(0,2,1),(2,1,0)):
            with self.subTest(order=order):
                observed = normalized_channel_permutation(image,order)
                torch.testing.assert_close(observed,normalized(raw[:,list(order)]),rtol=2e-6,atol=2e-6)
                self.assertEqual(observed.dtype,torch.float32)
                self.assertEqual(observed.shape,image.shape)
                inverse = tuple(order.index(c) for c in range(3))
                torch.testing.assert_close(normalized_channel_permutation(observed,inverse),image,rtol=2e-6,atol=2e-6)
                torch.testing.assert_close(normalized_channel_permutation(image[0],order),observed[0],rtol=0,atol=0)

    def test_identity_returns_exact_same_tensor_without_changes(self):
        image = normalized(torch.arange(12,dtype=torch.float32).reshape(3,2,2))
        before = image.clone()
        self.assertIs(normalized_channel_permutation(image,NATIVE_ORDER),image)
        self.assertTrue(torch.equal(image,before))

    def test_operator_and_schedule_do_not_draw_any_RNG(self):
        image = normalized(torch.arange(12,dtype=torch.float32).reshape(3,2,2)/10)
        python_before,numpy_before,torch_before = random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        for epoch in range(1,16):
            normalized_channel_permutation(image,TRAIN_ORDERS[training_order_index(7,epoch)])
        normalized_channel_permutation(image,COUNTERFACTUAL_ORDER)
        self.assertEqual(random.getstate(),python_before)
        after = np.random.get_state()
        self.assertEqual(after[0],numpy_before[0])
        self.assertTrue(np.array_equal(after[1],numpy_before[1]))
        self.assertEqual(after[2:],numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_before))

    def test_training_wrapper_has_one_view_and_keeps_source_identity_and_label(self):
        base = ToyDataset()
        dataset = ChannelOrderDataset(base,epoch=2)
        sample = dataset[1]
        self.assertEqual(base.calls,[1])
        self.assertEqual(sample["channel_order_index"],2)
        self.assertEqual((sample["sample_id"],sample["group_id"],sample["y"]),("toy1","toy",.25))
        torch.testing.assert_close(sample["image"],normalized_channel_permutation(base.image,TRAIN_ORDERS[2]))
        native = ChannelOrderDataset(base,order=NATIVE_ORDER)
        self.assertIs(native[0]["image"],base.image)

    def test_exact_per_epoch_and_per_item_C3_exposures(self):
        per_item = [[0,0,0] for _ in range(5496)]
        for epoch in range(1,16):
            counts = [0,0,0]
            for index in range(5496):
                order = training_order_index(index,epoch)
                counts[order] += 1
                per_item[index][order] += 1
            self.assertEqual(counts,[1832,1832,1832])
        self.assertTrue(all(counts == [5,5,5] for counts in per_item))
        for epoch in (0,16):
            with self.assertRaises(ValueError):
                training_order_index(0,epoch)


def passing_native():
    limits = CONFIG["native_primary"]
    return {"passed":True,"model":{"nmae":limits["mean_max"],
        "tail10_nmae":math.nextafter(limits["tail_strict_max"],0.),"maximum_error":limits["maximum_max"],
        **{f"count_at_{threshold}_percent_fs":292-limit for threshold,limit in zip((1,2,5),limits["counts_over_1_2_5_max"],strict=True)}}}


def scalar_rows():
    return [{"sample_id":f"id{i:03}","group_id":f"group{i//4}","target":.5,"prediction":.5,
        "valid":True,"absolute_error":0.} for i in range(292)]


class NewGateTests(unittest.TestCase):
    def test_native_primary_requires_every_fixed_predicate_and_strict_tail(self):
        gate = passing_native()
        self.assertTrue(native_primary(gate)["passed"])
        for key in ("nmae","maximum_error"):
            changed = deepcopy(gate)
            changed["model"][key] = math.nextafter(changed["model"][key],math.inf)
            self.assertFalse(native_primary(changed)["passed"])
        changed = deepcopy(gate)
        changed["model"]["tail10_nmae"] = CONFIG["native_primary"]["tail_strict_max"]
        self.assertFalse(native_primary(changed)["passed"])
        for threshold in (1,2,5):
            changed = deepcopy(gate)
            changed["model"][f"count_at_{threshold}_percent_fs"] -= 1
            self.assertFalse(native_primary(changed)["passed"])

    def test_conditional_CF_requires_both_native_gates_and_reuses_same_ID_dataset(self):
        for learning,primary in ((False,False),(False,True),(True,False)):
            def forbidden(*_args):
                self.fail("ineligible native gates constructed or evaluated a counterfactual")
            self.assertIsNone(conditional_counterfactual({"passed":learning},{"passed":primary},object(),forbidden))
        base,calls = ToyDataset(),[]
        def evaluate(name,dataset,images,batches):
            calls.append((name,dataset,images,batches))
            return "saved scalar report"
        self.assertEqual(conditional_counterfactual({"passed":True},{"passed":True},base,evaluate),"saved scalar report")
        self.assertEqual(len(calls),1)
        name,dataset,images,batches = calls[0]
        self.assertEqual((name,images,batches),("source_counterfactual",292,5))
        self.assertIs(dataset.base,base)
        self.assertEqual(base.calls,[])
        torch.testing.assert_close(dataset[0]["image"],normalized_channel_permutation(base.image,COUNTERFACTUAL_ORDER))
        self.assertEqual(base.calls,[0])

    def test_CF_metric_gates_and_paired_rescues_regressions(self):
        native,changed = scalar_rows(),scalar_rows()
        native[0].update(prediction=.52,absolute_error=.02)
        changed[1].update(prediction=.52,absolute_error=.02)
        summary = {"nmae":CONFIG["counterfactual_gate"]["mean_max"],"tail10_nmae":CONFIG["counterfactual_gate"]["tail_max"]}
        report = counterfactual_gate(native,changed,summary)
        self.assertTrue(report["passed"])
        self.assertEqual((report["paired_error_rescues"],report["paired_error_regressions"],report["paired_error_unchanged"]),(1,1,290))
        self.assertEqual(report["paired_thresholds_percent_fs"]["1"],{"rescues":1,"regressions":1})
        for key in ("nmae","tail10_nmae"):
            failing = {**summary,key:math.nextafter(summary[key],math.inf)}
            self.assertFalse(counterfactual_gate(native,changed,failing)["passed"])

    def test_CF_drift_mean_and_ceil30_tail_are_separate_gates(self):
        native = scalar_rows()
        changed = deepcopy(native)
        for row in changed:
            row.update(prediction=.52,absolute_error=.02)
        report = counterfactual_gate(native,changed,{"nmae":.02,"tail10_nmae":.02})
        self.assertFalse(report["predicates"]["mean_drift"])
        self.assertTrue(report["predicates"]["tail_drift"])
        changed = deepcopy(native)
        for row in changed[:2]:
            row.update(prediction=None,valid=False,absolute_error=1.)
        report = counterfactual_gate(native,changed,{"nmae":2/292,"tail10_nmae":2/30})
        self.assertTrue(report["predicates"]["mean_drift"])
        self.assertFalse(report["predicates"]["tail_drift"])
        self.assertAlmostEqual(report["mean_absolute_scalar_drift"],2/292)
        self.assertAlmostEqual(report["tail30_absolute_scalar_drift"],2/30)
        self.assertEqual(report["failed_pairs"],2)

    def test_CF_alignment_rejects_label_identity_group_and_duplicate_changes(self):
        native = scalar_rows()
        for key,value in (("sample_id","other"),("group_id","other"),("target",.6)):
            changed = deepcopy(native)
            changed[0][key] = value
            with self.assertRaises(ValueError):
                counterfactual_gate(native,changed,{"nmae":0.,"tail10_nmae":0.})
        changed = deepcopy(native)
        changed[0],changed[1] = changed[1],changed[0]
        with self.assertRaises(ValueError):
            counterfactual_gate(native,changed,{"nmae":0.,"tail10_nmae":0.})
        duplicate = deepcopy(native)
        duplicate[1] = duplicate[0]
        with self.assertRaises(ValueError):
            counterfactual_gate(duplicate,duplicate,{"nmae":0.,"tail10_nmae":0.})

    def test_partial_CF_CSV_without_summary_never_authorizes_another_forward(self):
        fake = ModuleType("experiments.geoprr_v2_high_resolution_direct")
        fake.summarize = lambda _rows:self.fail("partial evaluation must stop before metric reconstruction")
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{fake.__name__:fake}):
            output = Path(folder)
            path = output/"source_counterfactual_predictions.csv"
            path.write_text("sample_id,prediction\nid0,0.5\n")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                load_evaluation(output,"source_counterfactual",{},COUNTERFACTUAL_ORDER)
            self.assertEqual(path.read_bytes(),before)


def complete_totals(eligible):
    result = initial_totals()
    result.update(train_batches=1290,train_cnn_image_forwards=82440,train_cnn_forward_calls=1290,
        backward_calls=1290,point_backward_calls=1290,optimizer_updates=1289,amp_skipped_updates=1,
        train_point_forward_calls=1290,train_point_image_forwards=82440,
        validation_images=584 if eligible else 292,validation_cnn_forward_calls=10 if eligible else 5,
        validation_point_forward_calls=10 if eligible else 5,validation_point_image_forwards=584 if eligible else 292)
    return result


def resume_toy():
    history = []
    for epoch in range(1,3):
        history.append({"epoch":epoch,"learning_rate":.0003,"train_counts":{"cnn_forward_calls":86,"cnn_image_forwards":5496},
            "train_point_counts":{"point_forward_calls":86,"point_image_forwards":5496},"CNN_backward_calls":86,
            "point_backward_calls":86,"optimizer_updates":85,"amp_skipped_updates":1,"last_batch_size":56,
            "order_exposures":[1832,1832,1832],"validation_calls":0,"cost":{"seconds":1.}})
    totals = initial_totals()
    totals.update(train_batches=172,train_cnn_image_forwards=10992,train_cnn_forward_calls=172,
        backward_calls=172,point_backward_calls=172,optimizer_updates=170,amp_skipped_updates=2,
        train_point_forward_calls=172,train_point_image_forwards=10992,training_seconds=2.)
    return {"protocol":PROTOCOL,"config":deepcopy(CONFIG),"source_contract":{"toy_source":"private metadata"},
        "model":{"toy_weight":torch.tensor([1.])},"optimizer":{"state":{},"param_groups":[{"lr":.0003,"weight_decay":.0001,"params":[0]}]},
        "scaler":{"scale":65536.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":1},
        "rng":{"python":"toy","numpy":"toy","torch":"toy","cuda":"toy"},"epoch":2,"history":history,
        "totals":totals,"initialization":{"cached":True},"model_metadata":{"toy":True},
        "phases":{"train_epoch_1":{"seconds":1.},"train_epoch_2":{"seconds":1.}},
        "checkpointed_process_seconds":3.,"phase":"training"}


class CountAndResumeTests(unittest.TestCase):
    def test_both_fixed_formal_counts_keep_AMP_skips_in_attempt_budget(self):
        for eligible,expected in ((False,(82732,1295)),(True,(83024,1300))):
            totals = complete_totals(eligible)
            validate_totals(totals,eligible)
            summary = count_summary(totals)
            self.assertEqual((summary["cnn_image_forwards"],summary["cnn_forward_calls"]),expected)
            self.assertEqual(summary["point_image_forwards"],expected[0])
            self.assertEqual(summary["point_forward_calls"],expected[1])
            self.assertEqual(summary["CNN_backward_calls"],1290)
            self.assertEqual(summary["optimizer_updates"]+summary["amp_skipped_updates"],1290)
            self.assertEqual(summary["source_OOD_RGB_images"],0)
            self.assertEqual(summary["external_CNN_calls"],0)
            for key in ("train_cnn_image_forwards","point_backward_calls","amp_skipped_updates","validation_images"):
                bad = {**totals,key:totals[key]+1}
                with self.assertRaises(ValueError):
                    validate_totals(bad,eligible)

    def test_new_resume_accepts_exact_complete_epoch_metadata(self):
        saved = resume_toy()
        validate_resume(saved,saved["source_contract"],saved["optimizer"],saved["scaler"])
        self.assertEqual(saved["epoch"]+1,3)

    def test_resume_rejects_old_protocol_config_data_and_partial_state(self):
        saved = resume_toy()
        changes = [("protocol","old_control"),("config",{}),("source_contract",{}),("phase","evaluated"),("epoch",3)]
        for key,value in changes:
            changed = deepcopy(saved)
            changed[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_resume(changed,saved["source_contract"],saved["optimizer"])
        for key in ("model","optimizer","scaler","rng","history","totals"):
            changed = deepcopy(saved)
            del changed[key]
            with self.subTest(missing=key),self.assertRaises(ValueError):
                validate_resume(changed,saved["source_contract"])
        for field in ("order_exposures","validation_calls","optimizer_updates"):
            changed = deepcopy(saved)
            changed["history"][0][field] = [1831,1832,1833] if field == "order_exposures" else 1
            with self.subTest(history=field),self.assertRaises(ValueError):
                validate_resume(changed,saved["source_contract"])
        for field in ("training_seconds","optimizer_updates","validation_images"):
            changed = deepcopy(saved)
            changed["totals"][field] += 1
            with self.subTest(totals=field),self.assertRaises(ValueError):
                validate_resume(changed,saved["source_contract"])
        changed = deepcopy(saved)
        changed["optimizer"]["param_groups"][0]["lr"] = .001
        with self.assertRaises(ValueError):
            validate_resume(changed,saved["source_contract"],saved["optimizer"])
        changed = deepcopy(saved)
        changed["scaler"]["growth_interval"] = 100
        with self.assertRaises(ValueError):
            validate_resume(changed,saved["source_contract"],saved["optimizer"],saved["scaler"])
        for key,field in (("scaler","scale"),("rng","cuda")):
            changed = deepcopy(saved)
            del changed[key][field]
            with self.assertRaises(ValueError):
                validate_resume(changed,saved["source_contract"])


if __name__ == "__main__":
    unittest.main()
