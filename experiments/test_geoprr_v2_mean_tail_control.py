"""New CPU toys for batch mean-tail loss and source-reference joint safeguards."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments import geoprr_v2_cyclic_color_control as shared
from experiments import geoprr_v2_mean_tail_control as control


def reference_rows(error=.02):
    return [{"sample_id":f"source{i:03}","group_id":f"group{i//4}","target":.5,"prediction":.5-error,
        "absolute_error":error,"valid":True,"input_valid":True} for i in range(292)]


def passing_summary(view):
    reference = control.REFERENCE_STATISTICS[view]
    return {"nmae":reference["nmae"],"tail10_nmae":math.nextafter(reference["tail10_nmae"],0.) if view == "native" else reference["tail10_nmae"],
        "maximum_error":reference["maximum_error"]}


class ObjectiveTests(unittest.TestCase):
    def test_fixed_ceil_k_and_complete_epoch_tail_slots(self):
        self.assertEqual([control.tail_count(b) for b in (1,4,56,64)],[1,1,6,7])
        self.assertEqual(85*control.tail_count(64)+control.tail_count(56),601)

    def test_exact_composite_aggregation_and_all_image_gradient_weights(self):
        for batch_size in (1,4,56,64):
            with self.subTest(batch_size=batch_size):
                losses = torch.arange(1,batch_size+1,dtype=torch.float64,requires_grad=True)
                k = control.tail_count(batch_size)
                expected = .5*losses.detach().mean()+.5*losses.detach()[-k:].mean()
                observed = control.aggregate_mean_tail(losses)
                torch.testing.assert_close(observed,expected,rtol=0,atol=0)
                observed.backward()
                weights = torch.full_like(losses,.5/batch_size)
                weights[-k:] += .5/k
                torch.testing.assert_close(losses.grad,weights,rtol=1e-14,atol=1e-14)
                self.assertTrue(bool((losses.grad>0).all()))
                self.assertTrue(bool(torch.isfinite(losses.grad).all()))
                self.assertAlmostEqual(float(losses.grad.sum()),1.)

    def test_one_primitive_per_image_loss_call_and_finite_prediction_autograd(self):
        primitive = F.smooth_l1_loss
        for batch_size in (1,4,56,64):
            prediction = torch.linspace(.1,.9,batch_size,dtype=torch.float32,requires_grad=True)
            target = torch.zeros(batch_size,dtype=torch.float32)
            calls = []
            def record(*args,**kwargs):
                calls.append(kwargs)
                return primitive(*args,**kwargs)
            loss = control.mean_tail_smooth_l1(prediction,target,record)
            expected = control.aggregate_mean_tail(primitive(prediction.detach(),target,beta=.05,reduction="none"))
            torch.testing.assert_close(loss,expected)
            loss.backward()
            self.assertEqual(calls,[{"beta":.05,"reduction":"none"}])
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
            self.assertTrue(bool((prediction.grad>0).all()))
        with self.assertRaises(ValueError):
            control.mean_tail_smooth_l1(prediction,target,primitive,beta=.1)

    def test_loss_and_protocol_context_restore_original_references_without_recursion(self):
        primitive = F.smooth_l1_loss
        names = ("PROTOCOL","CONFIG","OUTPUT_RELATIVE","PREFLIGHT_RELATIVE","TRAIN_ORDERS","PROCESS_STARTED",
                 "learning_gate","native_primary","counterfactual_gate")
        before = {name:getattr(shared,name) for name in names}
        references = {"native":reference_rows(),"counterfactual":reference_rows()}
        prediction,target = torch.tensor([.1,.2,.3,.4]),torch.zeros(4)
        expected = control.mean_tail_smooth_l1(prediction,target,primitive)
        with self.assertRaisesRegex(RuntimeError,"context boundary"):
            with control.mean_tail_context(references):
                self.assertIsNot(F.smooth_l1_loss,primitive)
                torch.testing.assert_close(F.smooth_l1_loss(prediction,target,beta=.05),expected)
                self.assertEqual(shared.PROTOCOL,control.PROTOCOL)
                self.assertIs(shared.CONFIG,control.CONFIG)
                self.assertEqual(shared.PROCESS_STARTED,control.PROCESS_STARTED)
                raise RuntimeError("context boundary")
        self.assertIs(F.smooth_l1_loss,primitive)
        self.assertTrue(all(getattr(shared,name) is value for name,value in before.items()))

    def test_new_protocol_and_explicit_objective_metadata_reject_old_resume(self):
        objective = control.CONFIG["objective"]
        self.assertEqual((objective["mean_weight"],objective["tail_weight"],objective["tail_fraction"]),(.5,.5,.1))
        self.assertEqual(objective["upweighted_tail_slots_per_complete_epoch"],601)
        self.assertIn("composite",objective["history_source_training_smooth_l1"])
        self.assertIn("not primitive",objective["history_source_training_smooth_l1"])
        self.assertIn("source_preflight_mean_tail_objective",objective)
        self.assertFalse(control.CONFIG["counterfactual_gate"]["thresholds_are_acceptance"])
        keys = {"protocol","config","source_contract","model","optimizer","scaler","rng","epoch",
            "history","totals","initialization","model_metadata","phases","checkpointed_process_seconds","phase"}
        saved = dict.fromkeys(keys,None)
        saved.update(protocol=control.identity.PROTOCOL,config=control.identity.CONFIG,source_contract={},phase="training",epoch=1)
        with control.mean_tail_context({}),self.assertRaisesRegex(ValueError,"resume protocol"):
            shared.validate_resume(saved,{})
        saved.update(protocol=control.PROTOCOL,config=deepcopy(control.CONFIG))
        saved["config"]["objective"]["tail_weight"] = .25
        with control.mean_tail_context({}),self.assertRaisesRegex(ValueError,"resume protocol"):
            shared.validate_resume(saved,{})


class SourceJointGuardTests(unittest.TestCase):
    def test_native_and_CF_joint_limits_include_strict_native_tail_only(self):
        reference,candidate = reference_rows(),reference_rows()
        for view in ("native","counterfactual"):
            summary = passing_summary(view)
            self.assertTrue(control.joint_view_gate(reference,candidate,summary,view=view)["passed"])
            for key in ("nmae","tail10_nmae","maximum_error"):
                bad = {**summary,key:math.nextafter(control.REFERENCE_STATISTICS[view][key],math.inf)}
                with self.subTest(view=view,key=key):
                    self.assertFalse(control.joint_view_gate(reference,candidate,bad,view=view)["passed"])
        native = passing_summary("native")
        native["tail10_nmae"] = control.REFERENCE_STATISTICS["native"]["tail10_nmae"]
        self.assertFalse(control.joint_view_gate(reference,candidate,native,view="native")["passed"])
        cf = passing_summary("counterfactual")
        self.assertTrue(control.joint_view_gate(reference,candidate,cf,view="counterfactual")["passed"])

    def test_accurate_reference_1pct_guard_differs_from_general_5pct_new_errors(self):
        reference,candidate = reference_rows(),reference_rows()
        reference[0]["absolute_error"] = .01
        candidate[0]["absolute_error"] = .05
        candidate[1]["absolute_error"] = .06
        report = control.joint_view_gate(reference,candidate,passing_summary("native"),view="native")
        self.assertTrue(report["passed"])
        paired = report["paired_reference"]
        self.assertEqual(paired["reference_accurate_at_1_percent_count"],1)
        self.assertEqual(paired["paired_thresholds_percent_fs"]["5"]["new_errors"],1)
        self.assertEqual(paired["new_over5_on_reference_accurate_count"],0)
        candidate[0]["absolute_error"] = math.nextafter(.05,math.inf)
        report = control.joint_view_gate(reference,candidate,passing_summary("native"),view="native")
        self.assertFalse(report["passed"])
        self.assertFalse(report["predicates"]["no_new_over5_on_reference_accurate"])
        self.assertEqual(report["paired_reference"]["new_over5_on_reference_accurate_ids"],["source000"])
        self.assertEqual(report["paired_reference"]["new_over5_on_reference_accurate_groups"],["group0"])

    def test_only_5pct_count_is_a_new_count_gate_and_paired_rescues_are_preserved(self):
        reference,candidate = reference_rows(.2),reference_rows(.04)
        report = control.joint_view_gate(reference,candidate,passing_summary("native"),view="native")
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts_over_1_2_5"],[292,292,0])
        self.assertEqual(report["paired_reference"]["paired_thresholds_percent_fs"]["5"]["rescues"],292)
        for view,limit in (("native",16),("counterfactual",17)):
            candidate = reference_rows(.02)
            for row in candidate[:limit]:
                row["absolute_error"] = .06
            self.assertTrue(control.joint_view_gate(reference,candidate,passing_summary(view),view=view)["passed"])
            candidate[limit]["absolute_error"] = .06
            self.assertFalse(control.joint_view_gate(reference,candidate,passing_summary(view),view=view)["passed"])

    def test_exact_row_label_alignment_is_required_without_normalizing_labels(self):
        reference = reference_rows()
        for key,value in (("sample_id","other"),("group_id","other"),("target",.6)):
            candidate = deepcopy(reference)
            candidate[0][key] = value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"not exactly aligned"):
                control.paired_reference_guard(reference,candidate)
        candidate = deepcopy(reference)
        candidate[1] = candidate[0]
        with self.assertRaises(ValueError):
            control.paired_reference_guard(reference,candidate)

    def test_new_native_joint_guard_controls_CF_eligibility(self):
        reference,candidate = reference_rows(),reference_rows()
        reference[0]["absolute_error"] = .005
        candidate[0]["absolute_error"] = .06
        learning = {"passed":True,"model":passing_summary("native")}
        def forbidden(*_args):
            self.fail("new native ordinary guard failure opened CF")
        with patch.object(shared,"learning_gate",return_value=(learning,[])):
            with control.mean_tail_context({"native":reference,"counterfactual":reference}):
                gate,_ = shared.learning_gate(candidate)
                primary = shared.native_primary(gate)
                self.assertFalse(primary["passed"])
                self.assertIsNone(shared.conditional_counterfactual(gate,primary,object(),forbidden))

    def test_CF_drift_failure_is_diagnostic_while_new_joint_guard_can_pass(self):
        reference = reference_rows()
        native,candidate = reference_rows(.01),reference_rows(.01)
        for row in candidate:
            row["prediction"] = .51
        with control.mean_tail_context({"native":reference,"counterfactual":reference}):
            report = shared.counterfactual_gate(native,candidate,{"nmae":.01,"tail10_nmae":.01,"maximum_error":.01})
        self.assertTrue(report["passed"])
        self.assertFalse(report["drift_thresholds_are_acceptance"])
        self.assertFalse(report["previous_color_control_diagnostic"]["passed"])
        self.assertGreater(report["mean_absolute_scalar_drift"],.01)
        candidate[0].update(prediction=None,absolute_error=1.,valid=False,input_valid=False)
        with control.mean_tail_context({"native":reference,"counterfactual":reference}):
            failed = shared.counterfactual_gate(native,candidate,{"nmae":.01,"tail10_nmae":.01,"maximum_error":1.})
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failed_pairs"],1)
        self.assertEqual(failed["maximum_absolute_scalar_drift"],1.)


if __name__ == "__main__":
    unittest.main()
