"""Focused CPU checks for the new512 RGB pipeline and retained source population contract."""
from copy import deepcopy
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset

from experiments import geoprr_v2_high_resolution_direct as high
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset
from experiments.geoprr_v2_semantic_direct import scalar_rows


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.conv_stem = nn.Conv2d(3, 8, 3, stride=32, padding=1)
        self.activation = nn.GELU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(8, 1280, 1)
        self.norm_head = nn.BatchNorm2d(1280)
        self.num_features = 8

    def forward(self, image):
        self.inputs.append(tuple(image.shape))
        feature = self.global_pool(self.activation(self.conv_stem(image)))
        return self.norm_head(self.conv_head(feature)).flatten(1)


class RosterDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.reads = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads += 1
        raise AssertionError("source reconstruction must not decode or fetch retained RGB")


def fixture_row():
    return {"sample_id":"fixture", "group_id":"group", "image_path":"synthetic.png", "normalized_progress":.4,
            "metadata":{"dial_bbox":[3.2,4.1,605.2,590.7], "keypoints":"unused geometry"}}


def normalize_rgb(rgb):
    tensor = torch.from_numpy(rgb.transpose(2,0,1).copy())
    return (tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]


class HighResolutionDirectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test512_model_has_one_semantic_point_pass_and_eval_partial_batch_consistency(self):
        model = high.SemanticDirectReader(ToyEncoder()).eval()
        counts, hook = high.point_counter(model)
        image = torch.randn(3,3,512,512)
        try:
            with torch.inference_mode():
                full = model(image)
                partial = model(image[1:])
                single = model(image[2:])
        finally:
            hook.remove()
        self.assertEqual(model.encoder.inputs, [(3,3,512,512),(2,3,512,512),(1,3,512,512)])
        self.assertEqual(counts, {"point_forward_calls":3,"point_image_forwards":6})
        torch.testing.assert_close(full[1:], partial, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(full[2:], single, rtol=1e-5, atol=1e-7)
        self.assertEqual(tuple(full.shape),(3,))
        self.assertTrue(bool(torch.isfinite(full).all()))
        self.assertEqual((model.point.in_features,model.point.out_features),(1280,1))
        self.assertEqual(list(inspect.signature(model.forward).parameters),["image"])
        before = len(model.encoder.inputs)
        with self.assertRaises(TypeError): model(image, target=torch.zeros(3))
        self.assertEqual(len(model.encoder.inputs),before)

    def test512_dataset_uses_declared_native_crop_linear_rgb_normalization_without_geometry(self):
        row = fixture_row()
        yy,xx = np.indices((593,607))
        image = np.stack(((xx*3+yy*7)%256,(xx*11+yy*5)%256,(xx+yy*13)%256),-1).astype(np.uint8)
        expected_roi = cv2.resize(image[4:591,3:606],(512,512),interpolation=cv2.INTER_LINEAR)
        expected = normalize_rgb(cv2.cvtColor(expected_roi,cv2.COLOR_BGR2RGB).astype(np.float32)/255)
        dataset = high.ScalarRGBDataset([row],data_root=Path("/fixture"),augment=False)
        with patch("experiments.geoprr_v2_data._resolve_image",return_value=Path("/fixture/synthetic.png")), \
                patch("experiments.geoprr_v2_data.cv2.imread",return_value=image), \
                patch("experiments.geoprr_v2_data.dial_calibration",side_effect=AssertionError("geometry accessed")):
            sample = dataset[0]
        self.assertEqual((dataset.base.image_size,dataset.base.output_size),(512,128))
        self.assertFalse(dataset.base.require_geometry)
        self.assertIsNone(dataset.base.pairs)
        self.assertEqual(set(sample),{"image","y","sample_id","group_id","input_failure"})
        self.assertEqual(tuple(sample["image"].shape),(3,512,512))
        torch.testing.assert_close(sample["image"],expected,rtol=0,atol=0)
        self.assertEqual(sample["input_failure"],"")
        self.assertEqual(dataset.rows,[row])

    def test512_training_preserves_existing_photometric_transform(self):
        row = fixture_row(); row["roi_path"] = row.pop("image_path")
        image = np.full((32,64,3),[51,102,153],dtype=np.uint8)
        dataset = high.ScalarRGBDataset([row],data_root=Path("/fixture"),augment=True)
        with patch("experiments.geoprr_v2_data._resolve_image",return_value=Path("/fixture/synthetic.png")), \
                patch("experiments.geoprr_v2_data.cv2.imread",return_value=image), \
                patch.object(dataset.base,"_photometric",return_value=(1.1,.03,.9)):
            sample = dataset[0]
        expected_rgb = np.broadcast_to(np.array([.618,.42,.222],dtype=np.float32),(512,512,3))
        torch.testing.assert_close(sample["image"],normalize_rgb(expected_rgb),rtol=0,atol=1e-6)
        self.assertTrue(dataset.augment)
        self.assertEqual(tuple(sample["image"].shape),(3,512,512))

    def test512_failed_source_and_external_inputs_keep512_placeholders_and_failure_mass(self):
        row = fixture_row()
        dataset = high.ScalarRGBDataset([row],data_root=Path("/fixture"),augment=False)
        external = ExternalPopulationDataset([row],high.CONFIG["data"])
        training = high.ScalarRGBDataset([row],data_root=Path("/fixture"),augment=True)
        with patch("experiments.geoprr_v2_data._resolve_image",side_effect=FileNotFoundError("fixture missing RGB")):
            failed = dataset[0]
            external_failed = external[0]
            with self.assertRaises(FileNotFoundError): training[0]
        for sample in (failed,external_failed):
            self.assertEqual(tuple(sample["image"].shape),(3,512,512))
            self.assertTrue(bool((sample["image"]==0).all()))
            self.assertTrue(sample["input_failure"])
        batch = {"sample_id":[failed["sample_id"]],"group_id":[failed["group_id"]],
                 "y":failed["y"][None],"input_failure":[failed["input_failure"]]}
        scored = scalar_rows(torch.tensor([.4]),batch)[0]
        self.assertFalse(scored["valid"])
        self.assertEqual(scored["absolute_error"],1.)
        self.assertEqual((external.base.image_size,external.base.output_size),(512,128))

    def test_source_rebuild_preserves_four_rosters_exposure_and_separate_rgb512_contract(self):
        home = Path("/fixture/home")
        sizes = {"original_train":18590,"roll_train":4096,"original_validation":1506,"roll_validation":512}
        retained = {name:RosterDataset([{"sample_id":f"{name}_{i}","group_id":f"g_{name}_{i}","normalized_progress":.4}
                                       for i in range(count)]) for name,count in sizes.items()}
        old_training = ConcatDataset([retained["original_train"]]+[retained["roll_train"]]*4)
        old_validation = {"original":retained["original_validation"],"roll":retained["roll_validation"]}
        old_contract = {"original_source":{"data":{"image_size":256,"output_size":64}},
                        "roll_prepared":{"status":"complete"},
                        "populations":{name:[dict(row,target=.4) for row in dataset.rows] for name,dataset in retained.items()}}
        paths = {home/high.ORIGINAL_SOURCE_RELATIVE/"manifests/train.jsonl":"original_train",
                 home/high.SOURCE_RELATIVE/"manifests/train_roll.jsonl":"roll_train",
                 home/"runs/geoprr_v2/manifests/validation.jsonl":"original_validation",
                 home/high.SOURCE_RELATIVE/"manifests/validation_roll.jsonl":"roll_validation"}
        def rebuild(manifest,*,data_root,augment):
            name = paths[manifest]
            self.assertEqual(data_root,home/"datasets")
            self.assertEqual(augment,name.endswith("train"))
            return RosterDataset(deepcopy(retained[name].rows))
        with patch.object(high,"retained_source_populations",return_value=(old_training,old_validation,old_contract)) as guard, \
                patch.object(high,"ScalarRGBDataset",side_effect=rebuild):
            training,validations,contract = high.source_populations(home)
        guard.assert_called_once_with(home)
        self.assertEqual(len(training),34974)
        self.assertEqual(len(training.datasets),5)
        self.assertTrue(all(dataset is training.datasets[1] for dataset in training.datasets[1:]))
        self.assertEqual(training.cumulative_sizes,[18590,22686,26782,30878,34974])
        self.assertEqual((len(validations["original"]),len(validations["roll"])),(1506,512))
        self.assertEqual(contract["populations"],old_contract["populations"])
        self.assertEqual(contract["population_provenance"],{key:value for key,value in old_contract.items() if key!="populations"})
        self.assertEqual(contract["RGB_input_contract"]["data"],{"image_size":512,"output_size":128})
        self.assertEqual(contract["population_provenance"]["original_source"]["data"],{"image_size":256,"output_size":64})
        self.assertTrue(all(dataset.reads==0 for dataset in retained.values()))
        changed = RosterDataset(deepcopy(retained["original_train"].rows)); changed.rows[0]["normalized_progress"] = .5
        def changed_rebuild(manifest,*,data_root,augment):
            return changed if paths[manifest]=="original_train" else rebuild(manifest,data_root=data_root,augment=augment)
        with patch.object(high,"retained_source_populations",return_value=(old_training,old_validation,old_contract)), \
                patch.object(high,"ScalarRGBDataset",side_effect=changed_rebuild):
            with self.assertRaises(ValueError): high.source_populations(home)

    def test_scalar_state_reload_is_strict_and_resume_rejects_old256_input_contract(self):
        model = high.SemanticDirectReader(ToyEncoder())
        state = deepcopy(model.state_dict())
        restored = high.SemanticDirectReader(ToyEncoder())
        restored.load_state_dict(state,strict=True)
        self.assertEqual(len(state)-len(model.encoder.state_dict()),2)
        self.assertTrue(all(torch.equal(value,restored.state_dict()[key]) for key,value in state.items()))
        missing = dict(state); del missing["point.weight"]
        with self.assertRaises(RuntimeError): restored.load_state_dict(missing,strict=True)
        extra = dict(state); extra["nodes"] = torch.linspace(0,1,256)
        with self.assertRaises(RuntimeError): restored.load_state_dict(extra,strict=True)
        config256 = deepcopy(high.CONFIG); config256["data"] = {"image_size":256,"output_size":64}
        contract = {"RGB_input_contract":deepcopy(high.RGB_INPUT_CONTRACT)}
        with self.assertRaises(ValueError):
            high.validate_resume({"protocol":high.PROTOCOL,"config":config256,"data_contract":contract},contract)
        old_contract = deepcopy(contract); old_contract["RGB_input_contract"]["data"] = {"image_size":256,"output_size":64}
        with self.assertRaises(ValueError):
            high.validate_resume({"protocol":high.PROTOCOL,"config":high.CONFIG,"data_contract":old_contract},contract)


if __name__ == "__main__":
    unittest.main()
