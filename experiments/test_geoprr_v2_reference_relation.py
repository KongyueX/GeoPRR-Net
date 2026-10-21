"""Targeted CPU data/model checks with synthetic states and one untrained ROI."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.geoprr_v2_reference_relation import (
    CircularProgressPushforward, ReferenceRelationDataset, ReferenceRelationReader,
    reference_geometry_targets, reference_relation_losses,
)


REPORT = {}


def numpy_transport(atoms, circle_bins=256):
    output = np.zeros(256)
    invalid, outside = 0., 0.
    for (d, s), mass in atoms.items():
        if s == 0:
            output += mass/256
            invalid += mass
            continue
        delta, span = 2*np.pi*d/circle_bins, 2*np.pi*s/circle_bins
        value = ((delta-span/2+np.pi) % (2*np.pi)-np.pi+span/2)/span
        outside += mass*int(value < 0 or value > 1)
        coordinate = np.clip(value, 0, 1)*255
        lo, weight = int(np.floor(coordinate)), coordinate-np.floor(coordinate)
        output[lo] += mass*(1-weight)
        output[min(lo+1, 255)] += mass*weight
    return output, invalid, outside


class ReferenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20262020)

    def test_torch_pushforward_matches_numpy_all_mass(self):
        atoms = {(5, 0): .25, (20, 40): .25, (240, 40): .25, (100, 150): .25}
        joint = torch.zeros(1, 256, 256, requires_grad=True)
        with torch.no_grad():
            for indices, mass in atoms.items():
                joint[0, indices[0], indices[1]] = mass
        result = CircularProgressPushforward()(joint)
        expected, invalid, outside = numpy_transport(atoms)
        np.testing.assert_allclose(result["distribution"].detach().numpy()[0], expected, atol=2e-7, rtol=0)
        self.assertAlmostEqual(float(result["invalid_mass"].item()), invalid)
        self.assertAlmostEqual(float(result["outside_arc_mass"].item()), outside)
        torch.testing.assert_close(result["distribution"].sum(1), torch.ones(1), atol=2e-7, rtol=0)
        result["reading"].sum().backward()
        self.assertTrue(bool(torch.isfinite(joint.grad).all()))

    def test_GT_polar_target_identity_and_half_pixel_frame(self):
        center = torch.tensor([[.43, .57]], dtype=torch.float64)
        chol = torch.tensor([[math.exp(.17), 0.], [-.13, math.exp(-.17)]], dtype=torch.float64)
        spd = chol @ chol.T
        phi = .73
        rotation = torch.tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]], dtype=torch.float64)
        a = 2.3*rotation @ spd
        h = torch.eye(3, dtype=torch.float64)[None]
        h[:, :2, :2] = a/64
        h[:, :2, 2] = -(a @ center[0])+.5*h[:, :2, :2].sum(-1)
        labels = reference_geometry_targets(h, torch.tensor([.31]), torch.tensor([1.6*math.pi]), image_size=64)
        torch.testing.assert_close(labels["center"], center, atol=1e-12, rtol=0)
        torch.testing.assert_close(labels["spd"], spd[None], atol=1e-12, rtol=0)
        self.assertAlmostEqual(labels["phi"].item(), phi, places=12)
        torch.testing.assert_close(labels["role_twohot"].sum(-1), torch.ones(1, 3, dtype=torch.float64), atol=1e-15, rtol=0)
        raw = labels["rectifier_raw"]
        reconstructed = torch.zeros(1, 2, 2, dtype=torch.float64)
        reconstructed[:, 0, 0], reconstructed[:, 1, 0], reconstructed[:, 1, 1] = raw[:, 2].exp(), raw[:, 3], (-raw[:, 2]).exp()
        torch.testing.assert_close(reconstructed @ reconstructed.transpose(1, 2), spd[None], atol=1e-12, rtol=0)

    def test_RGB_forward_loss_and_frozen_shared_modules(self):
        model = ReferenceRelationReader()
        model.set_shared_trainable(False)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.state_head.training)
        image = torch.randn(1, 3, 64, 64)
        output = model(image)
        self.assertEqual(tuple(output["role_tokens"].shape), (1, 256, 16))
        self.assertEqual(tuple(output["global_features"].shape), (1, 192))
        self.assertEqual(tuple(output["query_points"].shape), (1, 256, 16, 2))
        self.assertTrue(bool(((output["query_points"] >= 0) & (output["query_points"] <= 1)).all()))
        for key in ("J", "product", "joint"):
            torch.testing.assert_close(output[key].sum((1, 2)), torch.ones(1), atol=5e-7, rtol=0)
        y = torch.tensor([.31])
        span = torch.tensor([1.5*math.pi])
        h = torch.tensor([[[2/64, 0., -1+1/64], [0., 2/64, -1+1/64], [0., 0., 1.]]])
        axis = (torch.arange(16)+.5)/16
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        vectors = 2*torch.stack((xx-.5, yy-.5), -1)
        phase = torch.atan2(vectors[..., 1], vectors[..., 0]).remainder(2*math.pi)
        mask = ((vectors.norm(dim=-1) >= .8) & (vectors.norm(dim=-1) <= 1.2) & (phase <= span.item())).float()
        batch = {"image": image, "y": y, "span_radians": span, "roi_to_canonical": h,
                 "field_target": (phase/span.item()).clamp(0, 1)[None], "field_mask": mask[None, None]}
        losses = reference_relation_losses(output, batch)
        self.assertEqual(set(losses), {"role_nll_raw", "role_nll_normalized", "field_l1", "reading_cdf", "total"})
        self.assertTrue(all(bool(torch.isfinite(value)) for value in losses.values()))
        losses["total"].backward()
        for parameter in model.state_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.shared_parameters()))
        REPORT.update(parameters=sum(parameter.numel() for parameter in model.parameters()),
                      shared_parameters=sum(parameter.numel() for parameter in model.shared_parameters()),
                      head_parameters=sum(parameter.numel() for parameter in model.state_head.parameters()),
                      buffer_tensors=len(list(model.buffers())), buffer_elements=sum(buffer.numel() for buffer in model.buffers()),
                      buffer_bytes=sum(buffer.numel()*buffer.element_size() for buffer in model.buffers()),
                      shared_state_tensors=sum(key.startswith(("backbone.", "lateral.", "refine.")) for key in model.state_dict()),
                      untrained_RGB_image_forwards=1, source_RGB_image_forwards=0, model_backward_calls=1,
                      loss_values={key: float(value.detach()) for key, value in losses.items()})


class ReferenceDatasetTests(unittest.TestCase):
    def test_exposure_supervision_snapshot_and_failed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, target in enumerate((.25, .5, .75)):
                marks = [[64*(.5+.4*math.cos(angle)), 64*(.5+.4*math.sin(angle))]
                         for angle in (0., math.pi/3, 2*math.pi/3, math.pi)]
                rows.append({"sample_id": f"fixture_{index}", "group_id": "synthetic_geometry",
                             "image_path": str(root/f"missing_{index}.png"), "target": target, "normalized_progress": target,
                             "metadata": {"homography": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                          "homography_bbox": [0, 0, 64, 64], "dial_bbox": [0, 0, 64, 64],
                                          "long_interval_degree": 60., "long_num": 4, "pointer_angle": 180*target,
                                          "image_width": 64, "image_height": 64, "scene_name": "synthetic_scene",
                                          "keypoints": [{"type": "ScaleMark", "all_kp": marks},
                                                        {"type": "Pointer", "all_kp": [[1, 2], [3, 4]]}]}})
            manifest, pairs = root/"source.jsonl", root/"pairs.jsonl"
            manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
            pairs.write_text(json.dumps({"sample_a": "fixture_0", "sample_b": "fixture_1"})+"\n"
                             +json.dumps({"sample_a": "fixture_0", "sample_b": "fixture_2"})+"\n")
            train = ReferenceRelationDataset(manifest, data_root=root, data_config={"image_size": 64, "output_size": 16},
                                              augment=True, pair_manifest=pairs)
            self.assertEqual(len(train), 4)
            first = train.supervision_contract()
            self.assertEqual((first["unique_images"], first["donor_items"], first["recipient_items"]), (3, 4, 2))
            self.assertEqual(len(first["rows"]), 3)
            self.assertNotIn("Pointer", json.dumps(first))
            self.assertEqual(first["rows"][0]["geometry"]["image_width"], 64)
            self.assertEqual(first["rows"][0]["geometry"]["scene_name"], "synthetic_scene")
            original_path = train.rows[0]["image_path"]
            train.rows[0]["image_path"] = str(root/"changed_missing.png")
            self.assertNotEqual(train.supervision_contract(), first)
            train.rows[0]["image_path"] = original_path
            train.rows[0]["metadata"]["homography"][0][0] = 2.
            self.assertNotEqual(train.supervision_contract(), first)
            train.rows[0]["metadata"]["homography"][0][0] = 1.
            photos = []

            def generated(index, photometric=None):
                photos.append(photometric)
                row = train.rows[index]
                train.base._calibrations[index] = {"angles": [0., math.pi]}
                return {"image": torch.full((3, 64, 64), float(index)), "y": torch.tensor(row["target"]),
                        "sample_id": row["sample_id"], "group_id": row["group_id"],
                        "roi_to_canonical": torch.eye(3), "field_target": torch.zeros(16, 16),
                        "field_mask": torch.ones(1, 16, 16)}

            with patch.object(train.base, "_sample", side_effect=generated):
                batch = train[0]
            self.assertEqual(batch["pair_sample_id"], "fixture_1")
            self.assertTrue(batch["pair_valid"])
            self.assertEqual(photos[0], photos[1])
            self.assertEqual(batch["span_radians"].item(), batch["pair_span_radians"].item())
            validation = ReferenceRelationDataset(manifest, data_root=root,
                data_config={"image_size": 64, "output_size": 16}, augment=False)
            failed = validation[0]
            self.assertTrue(failed["input_failure"])
            self.assertEqual(failed["sample_id"], "fixture_0")
            self.assertEqual(tuple(failed["image"].shape), (3, 64, 64))
            self.assertEqual(float(failed["field_mask"].sum()), 0.)
            with self.assertRaises(OSError):
                train[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (ReferenceIntegrationTests, ReferenceDatasetTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(status="PASS" if result.wasSuccessful() else "FAIL", tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), torch=torch.__version__, process_pid=os.getpid(),
                  wall_seconds=time.perf_counter()-started, checkpoint_loads=0, optimizer_updates=0, gpu_calls=0)
    args.output.write_text(json.dumps(REPORT, indent=2)+"\n")
    print(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
