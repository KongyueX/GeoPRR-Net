"""Targeted CPU checks for physical camera roll and the source-only population contract."""
from __future__ import annotations

import json
import io
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.render_geoprr_v2_roll_group import local_roll_quaternion
from experiments import prepare_geoprr_v2_roll_source as prepare
from experiments.geoprr_v2_endpoint_source import STATES
from experiments.prepare_geoprr_v2_compositional_data import generation_plan as original_plan


def optical_axis(quaternion):
    w, x, y, z = quaternion
    return np.array((-2 * (x * z + w * y), -2 * (y * z - w * x), -(1 - 2 * (x * x + y * y))))


def populations():
    plan = original_plan()
    source = [{"dataset": "SyncG", "split": "train", "sample_id": f"original_{index}", "group_id": "original_train", "normalized_progress": .5,
               "metadata": {"scene_name": "train.exr"}} for index in range(14494)]
    for index, group in enumerate(plan):
        parent = f"geoprr_generalization_endpoint_v11_g{index:04d}" if index < 128 else group["layout_id"]
        for state_index, state in enumerate(STATES if index < 128 else group["states"]):
            source.append({"dataset": "SyncG", "split": "train", "sample_id": f"{parent}_s{state_index:02d}", "group_id": parent,
                           "normalized_progress": state, "metadata": {"scene_name": "train.exr",
                           "factors": {"reference": f"/source/{parent}/factors.json.gz"}}})
    validation = [{"dataset": "SyncG", "split": "train", "sample_id": f"validation_{index}", "group_id": "original_validation", "normalized_progress": .5,
                   "metadata": {"scene_name": "validation.exr"}} for index in range(1506)]
    return plan, source, validation


class CameraRollTests(unittest.TestCase):
    def test_local_roll_preserves_a_tilted_camera_optical_axis(self):
        base = np.array((.9, .2, -.3, .1))
        base /= np.linalg.norm(base)
        for degrees in (-180., -93., 0., 47., 179.):
            result = local_roll_quaternion(base, degrees)
            np.testing.assert_allclose(optical_axis(result), optical_axis(base), atol=1e-12, rtol=0)
            self.assertAlmostEqual(sum(value * value for value in result), 1., places=12)

    def test_composition_is_local_rather_than_global_z(self):
        base = np.array((.9, .2, -.3, .1))
        base /= np.linalg.norm(base)
        w, x, y, z = base
        cosine, sine = math.cos(math.pi / 4), math.sin(math.pi / 4)
        global_roll = (cosine * w - sine * z, cosine * x - sine * y,
                       cosine * y + sine * x, cosine * z + sine * w)
        self.assertGreater(float(np.linalg.norm(optical_axis(global_roll) - optical_axis(base))), .1)
        np.testing.assert_allclose(optical_axis(local_roll_quaternion(base, 90.)), optical_axis(base), atol=1e-12, rtol=0)

    def test_invalid_roll_stops(self):
        for degrees in (float("nan"), float("inf"), -181., 180.):
            with self.assertRaises(ValueError):
                local_roll_quaternion((1., 0., 0., 0.), degrees)

    def test_noncamera_factor_change_is_rejected(self):
        parent = {"g": {"meshes": {}, "objects": [{"name": "Camera", "rotation_euler": [0, 0, 0],
                    "rotation_quaternion": [1, 0, 0, 0], "rotation_axis_angle": [0, 0, 1, 0], "location": [0, 0, 6]}]},
                  "a": {"material": "black"}, "c": {"name": "Camera", "rotation_euler": [0, 0, 0],
                    "matrix_world": [1, 0, 0, 1], "location": [0, 0, 6], "fov": 1.}}
        rolled = json.loads(json.dumps(parent))
        rolled["g"]["objects"][0]["rotation_euler"] = [0, 0, 1]
        rolled["c"]["rotation_euler"] = [0, 0, 1]
        rolled["c"]["matrix_world"] = [0, 1, -1, 0]

        def read_snapshot(path, _mode):
            return io.StringIO(json.dumps({"fixed_factors": parent if Path(path).parent.name == "parent" else rolled}))

        with patch.object(prepare.gzip, "open", side_effect=read_snapshot):
            self.assertTrue(prepare.compare_parent_factors(Path("/parent"), Path("/rolled"))["parent_nonroll_factors_equal"])
            rolled["a"]["material"] = "white"
            with self.assertRaisesRegex(ValueError, "beyond camera rotation"):
                prepare.compare_parent_factors(Path("/parent"), Path("/rolled"))


class RollPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.source, cls.validation = populations()
        cls.plan = prepare.generation_plan(cls.original, cls.source, cls.validation)

    def test_current_endpoint_states_and_interior_states_are_retained(self):
        self.assertEqual(len(self.plan), 1152)
        self.assertEqual(sum(row["split"] == "train" for row in self.plan), 1024)
        self.assertEqual(self.plan[0]["states"], STATES)
        self.assertEqual(self.plan[127]["states"], STATES)
        self.assertEqual(self.plan[128]["states"], self.original[128]["states"])
        for row in self.plan[1024:]:
            self.assertTrue(all(index / 4 < state < (index + 1) / 4 for index, state in enumerate(row["states"])))
        self.assertEqual(prepare.generation_plan(self.original, self.source, self.validation), self.plan)
        self.assertEqual(len({row["layout_id"] for row in self.plan}), 1152)
        self.assertEqual([row["render_seed"] for row in self.plan[1024:]], list(range(20270000, 20270128)))
        self.assertTrue(all(-180 <= row["roll_degrees"] < 180 for row in self.plan))

    def test_recorded_original_validation_seed_collision_stops(self):
        changed = list(self.validation)
        changed[0] = {**changed[0], "metadata": {**changed[0]["metadata"], "seed": 20270000}}
        with self.assertRaisesRegex(ValueError, "recorded source seed"):
            prepare.generation_plan(self.original, self.source, changed)

    def test_changed_current_physical_state_stops(self):
        changed = list(self.source)
        changed[14494] = {**changed[14494], "normalized_progress": .4}
        with self.assertRaisesRegex(ValueError, "current source physical readings"):
            prepare.generation_plan(self.original, changed, self.validation)

    def run_assembly(self, old_validation):
        original_path, validation_path, root = Path("/old/train.jsonl"), Path("/old/validation.jsonl"), Path("/new/roll")
        groups = {group["layout_id"]: group for group in self.plan}

        def source_rows(path):
            if Path(path) == original_path:
                return self.source
            if Path(path) == validation_path:
                return old_validation
            group = groups[Path(path).parent.parent.name]
            return [{"dataset": "SyncG", "sample_id": f"{group['layout_id']}_s{index:02d}", "group_id": group["layout_id"],
                     "split": group["split"], "normalized_progress": state,
                     "metadata": {"scene_name": "train.exr", "roll_source": {
                         **{key: group[key] for key in ("split", "layout_id", "render_seed", "roll_degrees")},
                         "state_index": index}}} for index, state in enumerate(group["states"])]

        def report_text(path, *args, **kwargs):
            group = groups[path.parent.name]
            return json.dumps({"protocol": prepare.PROTOCOL, "status": "complete",
                               "render_seed": group["render_seed"], "roll_degrees": group["roll_degrees"],
                               "minimum_gt_support_mass": .2, "maximum_gt_composed_error_percent_fs": .5,
                               "optical_axis_max_abs_difference": 0.})

        with patch.object(prepare, "read_rows", side_effect=source_rows), patch.object(Path, "read_text", report_text), \
                patch.object(prepare, "write_rows") as writes, patch.object(prepare, "write_json"):
            report = prepare.assemble(root, self.plan, original_path, validation_path)
        self.assertEqual(writes.call_args_list[1].args[1][:18590], self.source)
        return report

    def test_layout_holdout_can_share_training_HDR_without_a_false_background_claim(self):
        report = self.run_assembly(self.validation)
        self.assertEqual(report["combined_training_images"], 22686)
        self.assertEqual(report["roll_validation_images"], 512)
        self.assertFalse(report["roll_validation_holdout"]["background_independent"])
        self.assertFalse(report["roll_validation_holdout"]["all_historical_raw_seed_provenance_available"])
        self.assertEqual(report["new_group_checks"]["passed_groups"], 1152)

    def test_original_validation_background_guard_remains_active(self):
        changed = list(self.validation)
        changed[0] = {**changed[0], "metadata": {"scene_name": "train.exr"}}
        with self.assertRaisesRegex(ValueError, "background"):
            self.run_assembly(changed)


if __name__ == "__main__":
    unittest.main()
