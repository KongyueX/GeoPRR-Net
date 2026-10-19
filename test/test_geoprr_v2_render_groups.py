from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments import render_geoprr_v2_state_group as renderer


class GeoPRRV2RenderGroupTests(unittest.TestCase):
    def run_group(self, directory: str, *, formal: bool = False, change_camera: bool = False):
        root = Path(directory)
        source, output = root / "source", root / "output"
        (source / "blender_project/dial_texture").mkdir(parents=True)
        (source / "font_files").mkdir()
        scene = types.SimpleNamespace(render=types.SimpleNamespace(
            image_settings=types.SimpleNamespace(quality=95)))
        handlers = types.SimpleNamespace(render_pre=[], render_post=[])
        pointer = types.SimpleNamespace(rotation_euler=[0.0, 0.0, 0.0])

        def render(**_):
            for callback in handlers.render_pre:
                callback(scene)
            for callback in handlers.render_post:
                callback(scene)

        bpy = types.SimpleNamespace(
            context=types.SimpleNamespace(scene=scene, view_layer=types.SimpleNamespace(update=lambda: None)),
            data=types.SimpleNamespace(objects={"pointer3": pointer}),
            ops=types.SimpleNamespace(wm=types.SimpleNamespace(open_mainfile=lambda **_: None),
                                      render=types.SimpleNamespace(render=render)),
            app=types.SimpleNamespace(version_string="fixture", handlers=handlers),
        )
        vendor = types.SimpleNamespace(
            __file__=str(source / "generate_syncg.py"), pointer_name="pointer3",
            world_to_camera=lambda *args: [0, 0], make_random_sense=lambda **_: "fixture.hdr",
            parameter_generate=lambda **_: ({"long_interval_degree": 20, "long_num": 10}, {}),
            sync_gauge=lambda **_: render(),
        )

        def output_json(**values):
            annotation = {"ground_truth": values["pointer_rotate_degree"]}
            (Path(values["save_folder"]) / f"{values['save_name']}.json").write_text(json.dumps(annotation))

        vendor.output_json = output_json
        baseline = {"g": {"ticks": [0, 20, 40]}, "a": {"material": "black"}, "c": {"lens": 50}}
        snapshots = 0

        def actual_factors(*_):
            nonlocal snapshots
            snapshots += 1
            actual = copy.deepcopy(baseline)
            if change_camera and snapshots > 2:
                actual["c"]["lens"] = 60
            return actual

        arguments = ["--source-root", str(source), "--output", str(output)]
        if formal:
            arguments += ["--role", "source_training", "--layout-id", "syncg_training_layout_0001",
                          "--states", "0.11", "0.31", "0.61", "0.91"]
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda _: None))
        with patch.dict(os.environ), patch.dict("sys.modules", {"bpy": bpy, "numpy": numpy}), \
                patch.object(renderer, "load_vendor", return_value=vendor), \
                patch.object(renderer, "configure_devices", return_value={"backend": "CPU", "devices": []}), \
                patch.object(renderer, "configure_render"), \
                patch.object(renderer, "scene_factors", side_effect=actual_factors):
            self.assertEqual(renderer.main(arguments), 0)
        return output, baseline, snapshots

    def test_default_diagnostic_keeps_inline_factors_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output, baseline, snapshots = self.run_group(directory)
            group = json.loads((output / "group.json").read_text())
            self.assertEqual(group["layout_id"], "syncg_state_pilot_20262020")
            self.assertEqual(group["fixed_factors"], baseline)
            self.assertTrue(group["diagnostic_only"])
            self.assertTrue(group["pilot_only"])
            self.assertEqual(group["role"], "geometry_diagnostic")
            self.assertFalse((output / "factors.json.gz").exists())
            self.assertEqual(snapshots, 11)
            for image in group["images"]:
                annotation = json.loads(Path(image["annotation_path"]).read_text())
                self.assertEqual(annotation["factors"], {**baseline, "r": image["normalized_state"]})
                self.assertTrue(annotation["diagnostic_only"])

    def test_formal_group_stores_one_snapshot_and_per_image_states(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(renderer.gzip, "open", wraps=gzip.open) as compressed:
                output, baseline, snapshots = self.run_group(directory, formal=True)
            self.assertEqual(compressed.call_count, 1)
            group = json.loads((output / "group.json").read_text())
            factor_path = (output / "factors.json.gz").resolve()
            with gzip.open(factor_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"layout_id": "syncg_training_layout_0001",
                                                     "fixed_factors": baseline})
            self.assertEqual(group["fixed_factors"], {"reference": str(factor_path)})
            self.assertFalse(group["diagnostic_only"])
            self.assertFalse(group["pilot_only"])
            self.assertEqual(group["role"], "source_training")
            self.assertEqual(group["states"], [0.11, 0.31, 0.61, 0.91])
            self.assertEqual(snapshots, 11)
            rows = [json.loads(line) for line in (output / "manifests/raw_images.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            for image, row in zip(group["images"], rows):
                annotation = json.loads(Path(image["annotation_path"]).read_text())
                self.assertEqual(annotation["factors"], {"reference": str(factor_path),
                                                         "r": image["normalized_state"]})
                self.assertEqual(annotation["role"], "source_training")
                self.assertFalse(annotation["diagnostic_only"])
                self.assertEqual(row["role"], "source_training")
                self.assertFalse(row["diagnostic_only"])

    def test_reference_does_not_hide_changed_actual_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "non-intervened scene factors changed"):
                self.run_group(directory, formal=True, change_camera=True)
            output = Path(directory) / "output"
            self.assertTrue((output / "factors.json.gz").is_file())
            self.assertFalse((output / "group.json").exists())


if __name__ == "__main__":
    unittest.main()
