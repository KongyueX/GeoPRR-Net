"""New synthetic CPU checks for the fixed image measurement and thirty-epoch contract."""
from copy import deepcopy
from itertools import permutations
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_image_structure_control as control


RTOL=ATOL=2e-6
KX_REF=((-1,0,1),(-2,0,2),(-1,0,1))
KY_REF=((-1,-2,-1),(0,0,0),(1,2,1))


def normalize(raw):
    return (raw-raw.new_tensor(control.MEAN).view(1,3,1,1))/raw.new_tensor(control.STD).view(1,3,1,1)


def decoded(encoded):
    return encoded*encoded.new_tensor(control.STD).view(1,3,1,1)+encoded.new_tensor(control.MEAN).view(1,3,1,1)


def explicit_measurement(raw):
    result = torch.empty_like(raw)
    n,_,height,width = raw.shape
    def reflect(index,length):
        return -index if index<0 else 2*length-2-index if index>=length else index
    for batch in range(n):
        for y in range(height):
            for x in range(width):
                xx=yy=xy=0.
                for channel in range(3):
                    dx=dy=0.
                    for ky in range(3):
                        for kx in range(3):
                            value = float(raw[batch,channel,reflect(y+ky-1,height),reflect(x+kx-1,width)])
                            dx += KX_REF[ky][kx]*value
                            dy += KY_REF[ky][kx]*value
                    xx += dx*dx
                    yy += dy*dy
                    xy += dx*dy
                result[batch,:,y,x] = torch.tensor([xx/48,yy/48,.5+xy/96],dtype=torch.float32)
    return result


def fixtures():
    constant = torch.full((1,3,7,7),.37)
    xstep = torch.zeros(1,3,7,7)
    xstep[:,:,:,3:] = 1.
    ystep = torch.zeros(1,3,7,7)
    ystep[:,:,3:,:] = 1.
    diagonal = torch.zeros(1,3,7,7)
    for y in range(7):
        for x in range(7):
            diagonal[0,0,y,x] = float(x>y)
            diagonal[0,1,y,x] = float(x+y>=6)
    diagonal[0,2,1,2] = .8
    impulse = torch.zeros(1,3,7,7)
    impulse[0,:,0,1] = torch.tensor([.2,.6,1.])
    return {"constant":constant,"xstep":xstep,"ystep":ystep,"diagonal":diagonal,"boundary_impulse":impulse}


class MeasurementTests(unittest.TestCase):
    def test_explicit_sobel_reflection_coordinates_on_all_frozen_synthetic_patches(self):
        operator = control.FixedImageStructure()
        for name,raw in fixtures().items():
            with self.subTest(patch=name):
                expected = explicit_measurement(raw)
                observed = operator(normalize(raw))
                self.assertEqual(observed.dtype,torch.float32)
                self.assertEqual(observed.shape,raw.shape)
                torch.testing.assert_close(observed,normalize(expected),rtol=RTOL,atol=ATOL)
                field = decoded(observed)
                torch.testing.assert_close(field,expected,rtol=RTOL,atol=ATOL)
                self.assertTrue(bool(torch.isfinite(observed).all()))
                self.assertTrue(bool(((field>=-ATOL)&(field<=1+ATOL)).all()))
        report = operator.phase_report()
        self.assertEqual((report["fixed_filter_calls"],report["fixed_filter_images"]),(5,5))
        self.assertTrue(report["measurement_finite"] and report["measurement_bounded"])
        self.assertIsNone(report["operator_forward_gpu_seconds"])

    def test_constant_and_axis_steps_have_correct_channel_and_pixel_alignment(self):
        operator = control.FixedImageStructure()
        patches = fixtures()
        constant = decoded(operator(normalize(patches["constant"])))
        expected = torch.zeros_like(constant)
        expected[:,2] = .5
        torch.testing.assert_close(constant,expected,rtol=RTOL,atol=ATOL)
        xfield = decoded(operator(normalize(patches["xstep"])))
        yfield = decoded(operator(normalize(patches["ystep"])))
        torch.testing.assert_close(xfield[0,0,3,2:4],torch.ones(2),rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(xfield[0,1],torch.zeros(7,7),rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(yfield[0,1,2:4,3],torch.ones(2),rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(yfield[0,0],torch.zeros(7,7),rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(xfield[0,0,:,0],torch.zeros(7),rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(yfield[0,1,0,:],torch.zeros(7),rtol=RTOL,atol=ATOL)

    def test_all_six_raw_RGB_permutations_with_correct_renormalization(self):
        raw = (torch.arange(2*3*7*7,dtype=torch.float32).reshape(2,3,7,7)%19)/18
        operator = control.FixedImageStructure()
        expected = operator(normalize(raw))
        operator.reset_phase()
        for order in permutations(range(3)):
            observed = operator(normalize(raw[:,list(order)]))
            torch.testing.assert_close(observed,expected,rtol=RTOL,atol=ATOL)
        report = operator.phase_report()
        self.assertEqual((report["fixed_filter_calls"],report["fixed_filter_images"]),(6,12))
        self.assertTrue(report["shape_preserved"] and report["measurement_and_encoder_input_float32"])

    def test_fixed_F32_operator_disables_enclosing_autocast_and_uses_one_grouped_filter(self):
        raw = fixtures()["diagonal"].repeat(2,1,1,1)
        operator = control.FixedImageStructure()
        primitive = F.conv2d
        calls = []
        def record(image,weight,*args,**kwargs):
            calls.append((list(weight.shape),image.dtype,weight.dtype,kwargs["groups"]))
            return primitive(image,weight,*args,**kwargs)
        with patch.object(F,"conv2d",side_effect=record),torch.autocast(device_type="cpu",dtype=torch.bfloat16):
            result = operator(normalize(raw))
        self.assertEqual(calls,[([6,1,3,3],torch.float32,torch.float32,3)])
        self.assertEqual(result.dtype,torch.float32)
        torch.testing.assert_close(result,normalize(explicit_measurement(raw)),rtol=RTOL,atol=ATOL)
        self.assertEqual(operator.phase_report()["fixed_filter_images"],2)

    def test_zero_RNG_zero_parameters_one_persistent_F32_kernel_and_strict_reload(self):
        raw = fixtures()["boundary_impulse"]
        before_python,before_numpy,before_torch = random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        operator = control.FixedImageStructure()
        result = operator(normalize(raw))
        saved = deepcopy(operator.state_dict())
        clone = control.FixedImageStructure()
        clone.load_state_dict(saved,strict=True)
        torch.testing.assert_close(clone(normalize(raw)),result,rtol=0,atol=0)
        self.assertEqual(list(operator.parameters()),[])
        self.assertEqual(list(operator.named_buffers())[0][0],"kernel")
        self.assertEqual(set(saved),{"kernel"})
        self.assertEqual(tuple(saved["kernel"].shape),(6,1,3,3))
        self.assertEqual(saved["kernel"].dtype,torch.float32)
        self.assertTrue(torch.equal(saved["kernel"],torch.tensor((KX_REF,KY_REF)*3,dtype=torch.float32).reshape(6,1,3,3)))
        self.assertFalse(saved["kernel"].requires_grad)
        self.assertEqual(random.getstate(),before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0],before_numpy[0])
        self.assertTrue(np.array_equal(after_numpy[1],before_numpy[1]))
        self.assertEqual(after_numpy[2:],before_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),before_torch))

    def test_reader_wrap_preserves_existing_parameter_objects_and_has_no_unused_point(self):
        class ToyReader(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=nn.Identity()
                self.point=nn.Linear(3,1)
                self.semantic_dim=3
            def forward(self,image):
                semantic=self.encoder(image).mean((2,3))
                return self.point(semantic).squeeze(-1).float().sigmoid()
        base = ToyReader()
        rng = torch.get_rng_state().clone()
        wrapper = control.ImageStructureReader(base)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertIs(wrapper.encoder,base.encoder)
        self.assertIs(wrapper.point,base.point)
        self.assertEqual({id(p) for p in wrapper.parameters()},{id(p) for p in base.parameters()})
        self.assertEqual(len(wrapper.state_dict()),len(base.state_dict())+1)
        self.assertEqual(set(wrapper.state_dict()),{"reader.point.weight","reader.point.bias","measurement.kernel"})
        result = wrapper(normalize(fixtures()["diagonal"]))
        self.assertEqual(result.shape,(1,))
        self.assertEqual(wrapper.measurement.phase_report()["fixed_filter_calls"],1)


def operator_report(images=5496,calls=86):
    return {"fixed_filter_calls":calls,"fixed_filter_images":images,"operator_forward_gpu_seconds":.125,
        "shape_preserved":True,"input_float32":True,"measurement_and_encoder_input_float32":True,
        "measurement_finite":True,"measurement_bounded":True,"encoder_input_finite":True}


def complete_totals(eligible):
    result = control.initial_totals()
    result.update(train_batches=2580,train_cnn_image_forwards=164880,train_cnn_forward_calls=2580,
        backward_calls=2580,point_backward_calls=2580,optimizer_updates=2577,amp_skipped_updates=3,
        train_point_forward_calls=2580,train_point_image_forwards=164880,train_fixed_filter_calls=2580,train_fixed_filter_images=164880,
        validation_images=584 if eligible else 292,validation_cnn_forward_calls=10 if eligible else 5,
        validation_point_forward_calls=10 if eligible else 5,validation_point_image_forwards=584 if eligible else 292,
        validation_fixed_filter_calls=10 if eligible else 5,validation_fixed_filter_images=584 if eligible else 292)
    return result


def resume_toy():
    history=[]
    for epoch in (1,2):
        history.append({"epoch":epoch,"learning_rate":.0003,"train_counts":{"cnn_forward_calls":86,"cnn_image_forwards":5496},
            "train_point_counts":{"point_forward_calls":86,"point_image_forwards":5496},"CNN_backward_calls":86,
            "point_backward_calls":86,"optimizer_updates":85,"amp_skipped_updates":1,"last_batch_size":56,
            "validation_calls":0,"cost":{"seconds":1.},"operator":operator_report()})
    totals=control.initial_totals()
    totals.update(train_batches=172,train_cnn_image_forwards=10992,train_cnn_forward_calls=172,
        backward_calls=172,point_backward_calls=172,optimizer_updates=170,amp_skipped_updates=2,
        train_point_forward_calls=172,train_point_image_forwards=10992,training_seconds=2.,
        train_fixed_filter_calls=172,train_fixed_filter_images=10992,train_operator_forward_gpu_seconds=.25)
    return {"protocol":control.PROTOCOL,"config":deepcopy(control.CONFIG),"source_contract":{"toy":"provided source metadata"},
        "operator_contract":deepcopy(control.OPERATOR_CONTRACT),"model":{"measurement.kernel":control.FixedImageStructure().kernel.clone()},
        "optimizer":{"state":{},"param_groups":[{"lr":.0003,"weight_decay":.0001,"params":[0]}]},
        "scaler":{"scale":65536.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":1},
        "rng":{"python":"toy","numpy":"toy","torch":"toy","cuda":"toy"},"epoch":2,"history":history,
        "totals":totals,"initialization":{"cached":True},"model_metadata":{"toy":True},
        "phases":{"train_epoch_1":{"seconds":1.},"train_epoch_2":{"seconds":1.}},
        "checkpointed_process_seconds":3.,"phase":"training"}


class ThirtyEpochContractTests(unittest.TestCase):
    def test_new_encoder_point_filter_and_backward_budgets_include_AMP_skips(self):
        for eligible,images,calls in ((False,165172,2585),(True,165464,2590)):
            totals=complete_totals(eligible)
            control.validate_totals(totals,eligible)
            summary=control.count_summary(totals)
            for key in ("cnn_image_forwards","point_image_forwards","fixed_filter_images"):
                self.assertEqual(summary[key],images)
            for key in ("cnn_forward_calls","point_forward_calls","fixed_filter_calls"):
                self.assertEqual(summary[key],calls)
            self.assertEqual(summary["optimizer_updates"]+summary["amp_skipped_updates"],2580)
            self.assertEqual(summary["CNN_backward_calls"],2580)
            for key in ("train_fixed_filter_calls","validation_fixed_filter_images","amp_skipped_updates"):
                bad={**totals,key:totals[key]+1}
                with self.assertRaises(ValueError):
                    control.validate_totals(bad,eligible)

    def test_new_public_RGB_encoder_measurement_and30E_metadata_are_distinct(self):
        self.assertEqual(control.EPOCHS,30)
        self.assertEqual(control.CONFIG["training"]["epochs"],30)
        self.assertEqual(control.CONFIG["training"]["max_epochs"],30)
        self.assertEqual(control.CONFIG["model"]["state_tensors"],465)
        self.assertEqual(control.CONFIG["model"]["parameter_tensors"],233)
        self.assertEqual(control.CONFIG["input_operator"]["new_trainable_parameters"],0)
        self.assertEqual(control.CONFIG["input_operator"]["persistent_kernel_buffers"],1)
        self.assertFalse(control.CONFIG["input_operator"]["mean_std_registered_buffers"])
        self.assertIn("RGB",control.CONFIG["input_operator"]["public_input"])
        self.assertIn("second-moment",control.CONFIG["input_operator"]["encoder_input"])
        self.assertIn("twice identity15",control.CONFIG["budget_qualification"])
        self.assertNotIn("objective",control.CONFIG)
        self.assertEqual(control.PREFLIGHT_COUNTS["fixed_filter_calls"],3)
        self.assertEqual(control.PREFLIGHT_COUNTS["fixed_filter_images"],8)

    def test_operator_counts_timing_and_boolean_contract_failures_are_rejected(self):
        report=operator_report()
        control.validate_operator_report(report,5496,86)
        for key,value in (("fixed_filter_calls",87),("fixed_filter_images",5495),("operator_forward_gpu_seconds",None),
                          ("operator_forward_gpu_seconds",float("nan")),("measurement_bounded",False)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                control.validate_operator_report({**report,key:value},5496,86)

    def test_strict_new_resume_preserves_kernel_operator_and_completed30E_boundaries(self):
        saved=resume_toy()
        control.validate_resume(saved,saved["source_contract"],saved["optimizer"],saved["scaler"])
        changes=(("protocol","old15E"),("operator_contract",{}),("epoch",31),("config",control.guards.CONFIG))
        for key,value in changes:
            changed=deepcopy(saved)
            changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                control.validate_resume(changed,saved["source_contract"])
        changed=deepcopy(saved)
        changed["model"]["measurement.kernel"][0,0,0,0]=0
        with self.assertRaisesRegex(ValueError,"fixed kernel"):
            control.validate_resume(changed,saved["source_contract"])
        changed=deepcopy(saved)
        changed["history"][0]["operator"]["fixed_filter_calls"]+=1
        with self.assertRaises(ValueError):
            control.validate_resume(changed,saved["source_contract"])
        changed=deepcopy(saved)
        changed["totals"]["train_operator_forward_gpu_seconds"]+=1
        with self.assertRaises(ValueError):
            control.validate_resume(changed,saved["source_contract"])


if __name__ == "__main__":
    unittest.main()
