"""New CPU toys for identity deletion context and the frozen matched decision."""
from copy import deepcopy
import unittest

import torch

from experiments import geoprr_v2_cyclic_color_control as shared
from experiments import geoprr_v2_matched_identity_control as identity


class IdentityDataset:
    rows = [{"sample_id":f"toy{i}"} for i in range(3)]

    def __init__(self):
        self.image = torch.tensor([[[.1]],[[.2]],[[.3]]],dtype=torch.float32)

    def __len__(self):
        return 3

    def __getitem__(self,index):
        return {"image":self.image,"y":.25,"sample_id":self.rows[index]["sample_id"]}


class IdentityDeletionTests(unittest.TestCase):
    def test_all_three_nominal_slots_return_original_RGB_and_record_actual_exposure(self):
        base = IdentityDataset()
        with identity.identity_context():
            dataset = shared.ChannelOrderDataset(base,epoch=1)
            samples = [dataset[i] for i in range(3)]
            self.assertEqual([sample["channel_order_index"] for sample in samples],[0,1,2])
            self.assertTrue(all(sample["image"] is base.image for sample in samples))
            self.assertTrue(all(sample["y"] == .25 for sample in samples))
            self.assertEqual(shared.TRAIN_ORDERS,((0,1,2),)*3)
            operator = shared.CONFIG["operator"]
            self.assertEqual(operator["nominal_slots_per_epoch"],[1832,1832,1832])
            self.assertEqual(operator["actual_RGB_GBR_BRG_per_epoch"],[5496,0,0])
            self.assertEqual(operator["each_item_actual_RGB_exposure_15_epochs"],15)
            self.assertNotIn("each_item_exposure_15_epochs",operator)

    def test_protocol_config_paths_and_entry_timer_are_isolated_and_restored(self):
        names = ("PROTOCOL","CONFIG","OUTPUT_RELATIVE","PREFLIGHT_RELATIVE","TRAIN_ORDERS","PROCESS_STARTED")
        before = {name:getattr(shared,name) for name in names}
        with self.assertRaisesRegex(RuntimeError,"context witness"):
            with identity.identity_context():
                self.assertEqual(shared.PROTOCOL,identity.PROTOCOL)
                self.assertIs(shared.CONFIG,identity.CONFIG)
                self.assertEqual(shared.OUTPUT_RELATIVE,identity.OUTPUT_RELATIVE)
                self.assertEqual(shared.PREFLIGHT_RELATIVE,identity.PREFLIGHT_RELATIVE)
                self.assertEqual(shared.PROCESS_STARTED,identity.PROCESS_STARTED)
                self.assertNotEqual(shared.PROTOCOL,before["PROTOCOL"])
                raise RuntimeError("context witness")
        self.assertTrue(all(getattr(shared,name) is value for name,value in before.items()))
        self.assertEqual(shared.CONFIG,identity.REFERENCE_CONFIG)

    def test_held_out_CF_route_keeps_original_RG_order_with_identity_context(self):
        base,calls = IdentityDataset(),[]
        def evaluate(name,dataset,images,batches):
            calls.append((name,dataset,images,batches))
            return "saved result"
        with identity.identity_context():
            result = shared.conditional_counterfactual({"passed":True},{"passed":True},base,evaluate)
        self.assertEqual(result,"saved result")
        self.assertEqual(len(calls),1)
        name,dataset,images,batches = calls[0]
        self.assertEqual((name,images,batches),("source_counterfactual",292,5))
        self.assertIs(dataset.base,base)
        self.assertEqual(dataset.order,(1,0,2))

    def test_old_C3_checkpoint_cannot_resume_as_new_identity_protocol(self):
        keys = {"protocol","config","source_contract","model","optimizer","scaler","rng","epoch",
            "history","totals","initialization","model_metadata","phases","checkpointed_process_seconds","phase"}
        old = dict.fromkeys(keys,None)
        old.update(protocol=identity.REFERENCE_PROTOCOL,config=deepcopy(identity.REFERENCE_CONFIG),source_contract={},phase="training",epoch=1)
        with identity.identity_context(),self.assertRaisesRegex(ValueError,"resume protocol"):
            shared.validate_resume(old,{})


def supported_identity():
    native = {**deepcopy(identity.REFERENCE_STATISTICS["source_id"]),"samples":292}
    cf = {**deepcopy(identity.REFERENCE_STATISTICS["source_counterfactual"]),"samples":292}
    for summary in (native,cf):
        summary["nmae"] += .01
        summary["tail10_nmae"] += .01
        summary["maximum_error"] += .02
        summary["count_at_5_percent_fs"] -= 1
    drift = {key:value+.01 for key,value in identity.REFERENCE_DRIFT.items()}
    drift["passed"] = False
    return {"source_id":{"summary":native},"source_counterfactual":{"summary":cf},
        "native_learning_gate":{"passed":True},"native_primary":{"passed":True},"counterfactual_eligible":True,
        "counterfactual_gate":drift,"totals":{"optimizer_updates":1288,"amp_skipped_updates":2}}


class MatchedDecisionTests(unittest.TestCase):
    def test_complete_contrast_supports_C3_without_requiring_identity_CF_acceptance(self):
        observed = supported_identity()
        report = identity.matched_decision(observed,True)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["verdict"],"MATCHED_SUPPORT_FOR_C3")
        self.assertTrue(all(report["predicates"].values()))
        self.assertEqual(report["C3"]["combined_errors_above_5_percent_fs"],59)
        expected = sum(identity.REFERENCE_STATISTICS[view][key] for view in ("source_id","source_counterfactual")
                       for key in ("nmae","tail10_nmae"))/4
        self.assertEqual(report["C3"]["R_view"],expected)
        self.assertEqual(report["identity_optimizer_updates"],1288)
        self.assertEqual(report["identity_amp_skipped_updates"],2)
        self.assertTrue(report["scheduled_budget_matched"])
        self.assertFalse(observed["counterfactual_gate"]["passed"])

    def test_each_frozen_risk_count_maximum_and_drift_predicate_can_reject_support(self):
        for predicate in ("R_view","combined_errors_above_5_percent_fs","maximum_across_both_views",
                          "mean_absolute_scalar_drift","tail30_absolute_scalar_drift"):
            observed = supported_identity()
            if predicate == "R_view":
                for view in ("source_id","source_counterfactual"):
                    for key in ("nmae","tail10_nmae"):
                        observed[view]["summary"][key] = identity.REFERENCE_STATISTICS[view][key]
            elif predicate == "combined_errors_above_5_percent_fs":
                observed["source_id"]["summary"]["count_at_5_percent_fs"] = 265
                observed["source_counterfactual"]["summary"]["count_at_5_percent_fs"] = 261
            elif predicate == "maximum_across_both_views":
                observed["source_id"]["summary"]["maximum_error"] = .54
                observed["source_counterfactual"]["summary"]["maximum_error"] = .53
            else:
                observed["counterfactual_gate"][predicate] = identity.REFERENCE_DRIFT[predicate]
            with self.subTest(predicate=predicate):
                report = identity.matched_decision(observed,True)
                self.assertTrue(report["eligible"])
                self.assertFalse(report["predicates"][predicate])
                self.assertEqual(report["verdict"],"NO_MATCHED_SUPPORT_FOR_C3")

    def test_count_and_maximum_allow_equality_while_risk_and_drift_remain_strict(self):
        observed = supported_identity()
        for view in ("source_id","source_counterfactual"):
            for key in ("maximum_error","count_at_5_percent_fs"):
                observed[view]["summary"][key] = identity.REFERENCE_STATISTICS[view][key]
        report = identity.matched_decision(observed,True)
        self.assertEqual(report["verdict"],"MATCHED_SUPPORT_FOR_C3")
        self.assertTrue(report["predicates"]["combined_errors_above_5_percent_fs"])
        self.assertTrue(report["predicates"]["maximum_across_both_views"])

    def test_missing_CF_or_failed_native_or_reference_qualification_is_incomplete(self):
        for branch in ("missing_CF","native_learning","native_primary","CF_ineligible","partial_CF","reference"):
            observed = supported_identity()
            qualified = True
            if branch == "missing_CF":
                observed["source_counterfactual"] = None
                observed["counterfactual_gate"] = None
            elif branch == "native_learning":
                observed["native_learning_gate"]["passed"] = False
            elif branch == "native_primary":
                observed["native_primary"]["passed"] = False
            elif branch == "CF_ineligible":
                observed["counterfactual_eligible"] = False
            elif branch == "partial_CF":
                observed["source_counterfactual"]["summary"]["samples"] = 291
            else:
                qualified = False
            with self.subTest(branch=branch):
                report = identity.matched_decision(observed,qualified)
                self.assertFalse(report["eligible"])
                self.assertEqual(report["verdict"],"MATCHED_CF_EVIDENCE_INCOMPLETE")
                self.assertIsNone(report["predicates"])
                self.assertIsNone(report["identity"])


if __name__ == "__main__":
    unittest.main()
