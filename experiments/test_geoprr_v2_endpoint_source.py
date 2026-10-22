"""CPU checks for the frozen endpoint plan and exact source replacement."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.geoprr_v2_endpoint_source import (
    DATA_PROTOCOL, REPLACEMENT, endpoint_generation_plan, replace_source_population,
    verify_endpoint_training_contract,
)


STATES = [0., .005, .02, 1.]


def frozen_plan_fixture():
    return [{"index": index, "layout_id": f"geoprr_compositional_v1_g{index:04d}",
             "render_seed": 20262020 + index, "states": [.1, .35, .6, .85],
             "vertex_order": [0, 2, 1, 3]} for index in range(129)]


def training_edges(rows):
    return [{"sample_a": rows[a]["sample_id"], "sample_b": rows[b]["sample_id"],
             "kind": "state", "layout_id": rows[a]["group_id"],
             "factors_a": deepcopy(rows[a]["metadata"]["factors"]),
             "factors_b": deepcopy(rows[b]["metadata"]["factors"])}
            for a, b in ((0, 2), (0, 3), (2, 1), (2, 0), (1, 3), (1, 2), (3, 0), (3, 1))]


def source_rows(layout_id, states):
    return [{"sample_id": f"{layout_id}_s{index:02d}", "group_id": layout_id,
             "split": "train", "dataset": "SyncG", "image_path": f"/source/{layout_id}_{index}.jpg",
             "scale_start": 10., "scale_end": 20., "ground_truth": 10. + 10. * state,
             "metadata": {"layout_id": layout_id, "scene_name": "source_scene.hdr",
                          "annotation_path": f"/source/{layout_id}/annotations/{index}.json",
                          "homography": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          "dial_bbox": [0., 0., 80., 80.], "image_width": 100, "image_height": 100,
                          "long_interval_degree": 60., "long_num": 4, "pointer_angle": 180. * state,
                          "keypoints": [{"type": "ScaleMark", "all_kp": [[70., 40.], [55., 65.],
                                                                          [25., 65.], [10., 40.]]}],
                          "factors": {"reference": f"/source/{layout_id}/factors.json.gz", "r": state}}}
            for index, state in enumerate(states)]


class EndpointSourceTests(unittest.TestCase):
    def setUp(self):
        self.old_plan = frozen_plan_fixture()
        self.plan = endpoint_generation_plan(self.old_plan)
        self.anchor_rows = [{"sample_id": "ordinary_source", "group_id": "ordinary",
                             "normalized_progress": .4, "image_path": "/source/ordinary.jpg"}]
        self.anchor_edges, self.new_rows, self.new_edges = [], [], []
        for group in self.old_plan[:3]:
            rows = source_rows(group["layout_id"], group["states"])
            self.anchor_rows.extend(rows)
            self.anchor_edges.extend(training_edges(rows))
        for group in self.plan[:2]:
            rows = source_rows(group["layout_id"], group["states"])
            self.new_rows.extend(rows)
            self.new_edges.extend(training_edges(rows))

    def replace(self, *, plan=None, rows=None, edges=None):
        return replace_source_population(self.anchor_rows, self.anchor_edges,
            self.plan[:2] if plan is None else plan,
            self.new_rows if rows is None else rows,
            self.new_edges if edges is None else edges)

    def test_plan_preserves_source_layout_seeds_and_cycle_order(self):
        original = deepcopy(self.old_plan)
        self.assertEqual(len(self.plan), 128)
        for index, item in enumerate(self.plan):
            old = self.old_plan[index]
            self.assertEqual(item["index"], index)
            self.assertEqual(item["old_layout_id"], old["layout_id"])
            self.assertEqual(item["layout_id"], f"geoprr_generalization_endpoint_v11_g{index:04d}")
            self.assertEqual(item["render_seed"], old["render_seed"])
            self.assertEqual(item["states"], STATES)
            self.assertEqual(item["vertex_order"], old["vertex_order"])
        self.plan[0]["vertex_order"].reverse()
        self.assertEqual(self.old_plan, original)

    def test_plan_requires_the_full_frozen_128_group_prefix(self):
        with self.assertRaises(ValueError):
            endpoint_generation_plan(self.old_plan[:127])

    def test_population_and_cycle_edges_replace_at_original_positions(self):
        before = deepcopy((self.anchor_rows, self.anchor_edges, self.new_rows, self.new_edges))
        rows, edges, _ = self.replace()
        mapping = {f"{group['old_layout_id']}_s{state:02d}": f"{group['layout_id']}_s{state:02d}"
                   for group in self.plan[:2] for state in range(4)}
        replacements = {row["sample_id"]: row for row in self.new_rows}
        self.assertEqual(len(rows), len(self.anchor_rows))
        self.assertEqual(len(edges), len(self.anchor_edges))
        for actual, old in zip(rows, self.anchor_rows, strict=True):
            expected = replacements[mapping[old["sample_id"]]] if old["sample_id"] in mapping else old
            self.assertEqual(actual, expected)
        new_by_pair = {(edge["sample_a"], edge["sample_b"]): edge for edge in self.new_edges}
        for actual, old in zip(edges, self.anchor_edges, strict=True):
            key = (mapping.get(old["sample_a"], old["sample_a"]), mapping.get(old["sample_b"], old["sample_b"]))
            expected = new_by_pair[key] if old["sample_a"] in mapping else old
            self.assertEqual(actual, expected)
        self.assertEqual((self.anchor_rows, self.anchor_edges, self.new_rows, self.new_edges), before)

    def test_wrong_endpoint_state_is_rejected(self):
        rows = deepcopy(self.new_rows)
        rows[0]["ground_truth"] = 10.1
        with self.assertRaises(ValueError):
            self.replace(rows=rows)

    def test_fixed_layout_geometry_cannot_change_with_pointer_states(self):
        for key, value in (("dial_bbox", [0., 0., 90., 90.]), ("scene_name", "validation_scene.hdr"),
                           ("long_interval_degree", 50.)):
            with self.subTest(key=key):
                rows = deepcopy(self.new_rows)
                rows[0]["metadata"][key] = value
                with self.assertRaises(ValueError):
                    self.replace(rows=rows)

    def test_missing_or_duplicate_endpoint_id_is_rejected(self):
        for kind in ("missing", "duplicate", "foreign"):
            with self.subTest(kind=kind):
                rows = deepcopy(self.new_rows)
                if kind == "missing":
                    rows.pop()
                elif kind == "duplicate":
                    rows[-1] = deepcopy(rows[0])
                else:
                    rows[0]["sample_id"] = "foreign_source_s00"
                with self.assertRaises(ValueError):
                    self.replace(rows=rows)

    def test_cycle_topology_and_metadata_are_frozen(self):
        for kind in ("missing", "pair", "layout", "factors"):
            with self.subTest(kind=kind):
                edges = deepcopy(self.new_edges)
                if kind == "missing":
                    edges.pop()
                elif kind == "pair":
                    edges[0]["sample_b"] = edges[0]["sample_a"]
                elif kind == "layout":
                    edges[0]["layout_id"] = "other_layout"
                else:
                    edges[0]["factors_a"]["r"] = .8
                with self.assertRaises(ValueError):
                    self.replace(edges=edges)


def contract_rows(rows):
    result = []
    fixed = ("homography", "dial_bbox", "long_interval_degree", "long_num", "pointer_angle",
             "image_width", "image_height", "scene_name")
    for row in rows:
        metadata = row.get("metadata", {})
        geometry = {key: deepcopy(metadata[key]) for key in fixed if key in metadata}
        geometry["ScaleMark_keypoints"] = deepcopy(metadata.get("keypoints", []))
        y = ((row["ground_truth"] - row["scale_start"]) / (row["scale_end"] - row["scale_start"])
             if "ground_truth" in row else row["normalized_progress"])
        result.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "y": y,
                       "image_path": row["image_path"], "roi_path": None, "geometry": geometry})
    return result


class EndpointTrainingContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.old_root, self.new_root = root / "old", root / "new"
        self.old_root.mkdir()
        self.new_root.mkdir()
        old_plan = frozen_plan_fixture()
        plan = endpoint_generation_plan(old_plan)
        (self.old_root / "generation_plan.json").write_text(json.dumps(old_plan))
        (self.new_root / "generation_plan.json").write_text(json.dumps(plan))
        old_rows = [{"sample_id": "ordinary_source", "group_id": "ordinary",
                     "normalized_progress": .4, "image_path": "/source/ordinary.jpg"}]
        old_edges, replacement_rows, replacement_edges = [], [], []
        for item in old_plan:
            rows = source_rows(item["layout_id"], item["states"])
            old_rows.extend(rows)
            old_edges.extend(training_edges(rows))
        for item in plan:
            rows = source_rows(item["layout_id"], item["states"])
            replacement_rows.extend(rows)
            replacement_edges.extend(training_edges(rows))
        new_rows, new_edges, _ = replace_source_population(
            old_rows, old_edges, plan, replacement_rows, replacement_edges)

        def pairs(edges):
            return sorted([[edge["sample_a"], edge["sample_b"]] for edge in edges])

        def training_contract(rows, edges):
            return {"schema": "reference_relation_geometry_supervision_v1", "rows": contract_rows(rows),
                    "edges": pairs(edges), "image_size": 768, "output_size": 192,
                    "geometry_config": {"min_support_mass": .15, "max_readout_error": .01},
                    "unique_images": len(rows), "donor_items": len(rows) + 4 * 129,
                    "recipient_items": len(edges)}

        self.anchor = {"paths": {"train_manifest": str(self.old_root / "manifests/train.jsonl"),
                                 "pair_manifest": str(self.old_root / "manifests/train_edges.jsonl"),
                                 "val_manifest": "/source/validation.jsonl"},
                       "train": training_contract(old_rows, old_edges),
                       "validation": {"rows": [{"sample_id": "validation_0", "y": .7}]},
                       "exposure_edges": pairs(old_edges)}
        self.candidate = deepcopy(self.anchor)
        self.candidate["paths"].update(train_manifest=str(self.new_root / "manifests/train.jsonl"),
                                      pair_manifest=str(self.new_root / "manifests/train_edges.jsonl"))
        self.candidate.update(train=training_contract(new_rows, new_edges), exposure_edges=pairs(new_edges))
        self.prepared = {"protocol": DATA_PROTOCOL, "replacement": deepcopy(REPLACEMENT),
                         "original_source_root": str(self.old_root),
                         "replacement_contract": {"new_rows": contract_rows(replacement_rows),
                                                  "anchor_paths": deepcopy(self.anchor["paths"])}}
        self.write_prepared()

    def write_prepared(self):
        (self.new_root / "prepared.json").write_text(json.dumps(self.prepared))

    def test_exact_replacement_accepts_original_validation_and_geometry_settings(self):
        before = deepcopy((self.candidate, self.anchor))
        report = verify_endpoint_training_contract(self.candidate, self.anchor)
        self.assertEqual(report["replaced_unique_images"], 512)
        self.assertTrue(report["exact_source_replacement_verified"])
        self.assertEqual((self.candidate, self.anchor), before)

    def test_original_validation_rows_and_path_are_preserved(self):
        for kind in ("rows", "path"):
            with self.subTest(kind=kind):
                candidate = deepcopy(self.candidate)
                if kind == "rows":
                    candidate["validation"]["rows"][0]["y"] = .71
                else:
                    candidate["paths"]["val_manifest"] = "/source/different_validation.jsonl"
                with self.assertRaises(ValueError):
                    verify_endpoint_training_contract(candidate, self.anchor)

    def test_unchanged_source_row_cannot_be_modified(self):
        candidate = deepcopy(self.candidate)
        candidate["train"]["rows"][0]["image_path"] = "/source/replaced_ordinary.jpg"
        with self.assertRaises(ValueError):
            verify_endpoint_training_contract(candidate, self.anchor)

    def test_training_and_outer_exposure_edges_are_both_verified(self):
        for location in ("train", "outer"):
            with self.subTest(location=location):
                candidate = deepcopy(self.candidate)
                edges = candidate["train"]["edges"] if location == "train" else candidate["exposure_edges"]
                edges[0][1] = edges[0][0]
                with self.assertRaises(ValueError):
                    verify_endpoint_training_contract(candidate, self.anchor)

    def test_prepared_endpoint_targets_must_match_frozen_states(self):
        self.prepared["replacement_contract"]["new_rows"][0]["y"] = .2
        self.candidate["train"]["rows"][1]["y"] = .2
        self.write_prepared()
        with self.assertRaises(ValueError):
            verify_endpoint_training_contract(self.candidate, self.anchor)

    def test_geometry_threshold_and_exposure_budget_are_unchanged(self):
        for key, value in (("geometry_config", {"min_support_mass": .1, "max_readout_error": .02}),
                           ("recipient_items", self.candidate["train"]["recipient_items"] + 1)):
            with self.subTest(key=key):
                candidate = deepcopy(self.candidate)
                candidate["train"][key] = value
                with self.assertRaises(ValueError):
                    verify_endpoint_training_contract(candidate, self.anchor)


if __name__ == "__main__":
    unittest.main()
