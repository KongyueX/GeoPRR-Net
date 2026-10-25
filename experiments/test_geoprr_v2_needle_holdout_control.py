"""New CPU checks for source needle selection, fixed exposure and the ID gate."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from experiments.geoprr_v2_needle_holdout_control import (
    CONFIG, HELDOUT, SEEN, build_roster, conditional_ood, learning_gate, read_tail_rows, validate_totals,
)


def toy_records():
    original, roll, validation = [], [], []
    for pointer_index, pointer in enumerate(("needle_z", "needle_a", "needle_m")):
        parent, rolled, val = (f"{name}_{pointer_index}" for name in ("parent","roll","validation"))
        for slot, state in enumerate((.1,.3,.6,.9)):
            sample = f"{parent}_sample{slot}"
            original.append({"sample_id":sample, "layout":parent, "state":state,
                             "pointer":pointer, "view":"original"})
            roll.append({"sample_id":f"{rolled}_sample{slot}", "layout":rolled,
                         "state":state, "slot":slot, "parent_layout":parent,
                         "parent_sample_id":sample, "parent_equal":True, "view":"roll"})
            validation.append({"sample_id":f"{val}_sample{slot}", "layout":val,
                               "state":state, "slot":slot, "pointer":pointer, "view":"roll"})
    return original, roll, validation


def scalar_rows(targets, errors):
    result = []
    for index, (target, error) in enumerate(zip(targets, errors, strict=True)):
        result.append({"sample_id":str(index), "group_id":str(index//4), "target":target,
                       "prediction":target-error if target>=.5 else target+error,
                       "valid":True, "input_valid":True, "absolute_error":error,
                       "invalid_mass":None, "outside_arc_mass":None, "failure":""})
    return result


def totals(passed):
    return {"train_batches":430, "train_cnn_image_forwards":27480, "train_cnn_forward_calls":430,
            "backward_calls":430, "optimizer_updates":429, "amp_skipped_updates":1,
            "train_point_forward_calls":430, "train_point_image_forwards":27480,
            "validation_images":512 if passed else 292,
            "validation_cnn_forward_calls":9 if passed else 5,
            "validation_point_forward_calls":9 if passed else 5,
            "validation_point_image_forwards":512 if passed else 292}


class NeedleSelectionTests(unittest.TestCase):
    def test_canonical_sort_and_heldout_exclusion(self):
        roster = build_roster(*toy_records())
        records = roster["records"]
        self.assertEqual((len(records["train"]),len(records["id"]),len(records["ood"])), (16,8,4))
        self.assertEqual({r["pointer"] for r in records["train"]}, set(SEEN))
        self.assertEqual({r["pointer"] for r in records["id"]}, set(SEEN))
        self.assertEqual({r["pointer"] for r in records["ood"]}, {HELDOUT})
        self.assertEqual({r["layout"] for r in records["ood"]}, {"validation_0"})

    def test_each_retained_image_once_and_parent_closure(self):
        roster = build_roster(*toy_records())
        train = roster["records"]["train"]
        self.assertEqual(len({r["sample_id"] for r in train}),16)
        parents = {r["sample_id"] for r in train if r["view"] == "original"}
        self.assertEqual(parents, {r["parent_sample_id"] for r in train if r["view"] == "roll"})
        self.assertEqual(len([r for r in train if r["view"] == "original"]),8)

    def test_pref_only_seen_first_parent_slot0_and_id(self):
        roster = build_roster(*toy_records())
        train = [roster["records"]["train"][i] for i in roster["pref_indices"]["train"]]
        self.assertEqual([r["slot"] for r in train],[0,0,0,0])
        self.assertEqual([r["view"] for r in train],["original","roll","original","roll"])
        self.assertEqual([r["pointer"] for r in train],[SEEN[0],SEEN[0],SEEN[1],SEEN[1]])
        for alias, indices in zip(SEEN,roster["pref_indices"]["id"],strict=True):
            self.assertEqual(len(indices),2)
            self.assertEqual({roster["records"]["id"][i]["pointer"] for i in indices},{alias})

    def test_input_records_remain_unchanged(self):
        records = toy_records()
        before = deepcopy(records)
        build_roster(*records)
        self.assertEqual(records,before)

    def test_duplicate_parent_sample_rejected(self):
        original,roll,validation = toy_records()
        roll[1]["parent_sample_id"] = roll[0]["parent_sample_id"]
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_incomplete_saved_slots_rejected(self):
        original,roll,validation = toy_records()
        roll[1]["slot"] = 0
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_changed_parent_state_rejected(self):
        original,roll,validation = toy_records()
        roll[0]["state"] = .2
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_saved_parent_factor_false_rejected(self):
        original,roll,validation = toy_records()
        roll[0]["parent_equal"] = False
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_validation_bank_difference_rejected(self):
        original,roll,validation = toy_records()
        for row in validation[:4]:
            row["pointer"] = "another_needle"
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_validation_identity_overlap_rejected(self):
        original,roll,validation = toy_records()
        validation[0]["sample_id"] = original[0]["sample_id"]
        with self.assertRaises(ValueError):
            build_roster(original,roll,validation)

    def test_tail_reader_does_not_parse_old_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"train.jsonl"
            path.write_text('unparseable old prefix\n{"sample_id":"a"}\n{"sample_id":"b"}\n')
            self.assertEqual(read_tail_rows(path,2),[{"sample_id":"a"},{"sample_id":"b"}])


class LearningGateTests(unittest.TestCase):
    def test_both_threshold_equalities_pass_on_existing_labels(self):
        rows = scalar_rows([0.,1.]*146,[.25]*292)
        gate,baseline = learning_gate(rows)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["tail_count"],30)
        self.assertEqual([r["target"] for r in baseline],[r["target"] for r in rows])
        self.assertEqual({r["prediction"] for r in baseline},{.5})
        self.assertEqual(gate["constant"]["nmae"],.5)

    def test_tail_failure_blocks_when_mean_passes(self):
        gate,_ = learning_gate(scalar_rows([0.,1.]*146,[.4]*30+[0.]*262))
        self.assertFalse(gate["passed"])
        self.assertLess(gate["model"]["nmae"],.5*gate["constant"]["nmae"])
        self.assertGreater(gate["model"]["tail10_nmae"],.5*gate["constant"]["tail10_nmae"])

    def test_mean_failure_blocks_when_tail_passes(self):
        gate,_ = learning_gate(scalar_rows([0.,1.]*15+[.5]*262,[.1]*292))
        self.assertFalse(gate["passed"])
        self.assertGreater(gate["model"]["nmae"],.5*gate["constant"]["nmae"])
        self.assertLess(gate["model"]["tail10_nmae"],.5*gate["constant"]["tail10_nmae"])

    def test_failure_stays_in_full292_denominator(self):
        rows = scalar_rows([0.,1.]*146,[0.]*292)
        rows[0].update(valid=False,input_valid=False,prediction=None,absolute_error=1.,failure="input")
        gate,baseline = learning_gate(rows)
        self.assertEqual(gate["model"]["samples"],292)
        self.assertAlmostEqual(gate["model"]["nmae"],1/292)
        self.assertEqual(len(baseline),292)
        self.assertTrue(baseline[0]["valid"])

    def test_incomplete_id_denominator_rejected(self):
        with self.assertRaises(ValueError):
            learning_gate(scalar_rows([0.]*291,[0.]*291))

    def test_failed_gate_never_opens_ood(self):
        def forbidden():
            self.fail("source-OOD callback was invoked after failed ID learning gate")
        self.assertIsNone(conditional_ood({"passed":False},forbidden))

    def test_passed_gate_opens_ood_once(self):
        calls = []
        def evaluate():
            calls.append(True)
            return {"images":220}
        self.assertEqual(conditional_ood({"passed":True},evaluate),{"images":220})
        self.assertEqual(len(calls),1)


class FixedBudgetTests(unittest.TestCase):
    def test_five_epochs_no_repeat_or_scheduler(self):
        training,source = CONFIG["training"],CONFIG["source"]
        self.assertEqual((training["epochs"],training["max_epochs"]),(5,5))
        self.assertIsNone(training["scheduler"])
        self.assertFalse(training["validation_before_epoch5"])
        self.assertFalse(training["resume"])
        self.assertEqual((source["train_unique_images"],source["train_repeat"]),(5496,1))
        self.assertEqual((source["train_calls_per_epoch"],source["last_train_batch"]),(86,56))

    def test_both_gate_budget_paths_accept_actual_amp_skip(self):
        for passed in (False,True):
            with self.subTest(passed=passed):
                validate_totals(totals(passed),passed)

    def test_extra_ood_on_failed_gate_rejected(self):
        with self.assertRaises(ValueError):
            validate_totals(totals(True),False)

    def test_missing_backward_or_optimizer_accounting_rejected(self):
        for key in ("backward_calls","optimizer_updates","train_cnn_image_forwards","train_point_forward_calls"):
            values = totals(False)
            values[key] -= 1
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_totals(values,False)


if __name__ == "__main__":
    unittest.main()
