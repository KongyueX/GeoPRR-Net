"""Prepare physical camera-roll source groups and a separate layout validation set."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_data import normalized_target, read_rows
from experiments.geoprr_v2_endpoint_source import STATES, _canonical_circle_faces, _normalize_asset_paths
from experiments.prepare_geoprr_v2_compositional_data import generation_plan as quartile_plan, prepare_group as prepare_state_group
from experiments.render_geoprr_v2_roll_group import PROTOCOL
from experiments.train_geoprr_v2 import source_split_check, write_json


ROLL_SEED = 20261024
VALIDATION_SEED_START = 20270000


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def recorded_seeds(rows):
    result, missing = set(), 0
    for row in rows:
        metadata = row.get("metadata", {})
        values = [row.get("seed"), metadata.get("seed"), metadata.get("render_seed"),
                  metadata.get("provenance", {}).get("seed"), metadata.get("roll_source", {}).get("render_seed")]
        found = {int(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value) and int(value) == value}
        result.update(found)
        missing += int(not found)
    return result, missing


def generation_plan(original_plan, current_train_rows, original_validation_rows):
    if len(original_plan) != 1024 or len(current_train_rows) != 18590 or len(original_validation_rows) != 1506:
        raise ValueError("roll source requires the existing1024 layouts and18590/1506 populations")
    current = {row["sample_id"]: row for row in current_train_rows}
    if len(current) != len(current_train_rows):
        raise ValueError("current source sample IDs are not unique")
    rng = np.random.default_rng(ROLL_SEED)
    angles = rng.uniform(-180., 180., 1152).tolist()
    result = []
    for index, old in enumerate(original_plan):
        if old["index"] != index or old["render_seed"] != 20262020 + index:
            raise ValueError("the existing source layout seed/order differs")
        parent = (f"geoprr_generalization_endpoint_v11_g{index:04d}" if index < 128 else old["layout_id"])
        states = list(STATES) if index < 128 else list(old["states"])
        ids = [f"{parent}_s{state:02d}" for state in range(4)]
        rows = [current[sample_id] for sample_id in ids]
        if any(abs(normalized_target(row) - state) > 1e-6 for row, state in zip(rows, states, strict=True)):
            raise ValueError("new roll states differ from the current source physical readings")
        parents = {str(Path(row["metadata"]["factors"]["reference"]).parent) for row in rows}
        if len(parents) != 1:
            raise ValueError("one current source group has inconsistent parent factor snapshots")
        result.append({"index": index, "split": "train", "layout_id": f"geoprr_roll_v22_train_g{index:04d}",
                       "render_seed": old["render_seed"], "states": states, "vertex_order": list(old["vertex_order"]),
                       "roll_degrees": angles[index], "parent_layout_id": parent, "parent_sample_ids": ids,
                       "parent_group_root": parents.pop(), "state_policy": "endpoint_v11" if index < 128 else "interior"})
    existing_seeds, _missing = recorded_seeds(current_train_rows + original_validation_rows)
    existing_seeds.update(row["render_seed"] for row in result)
    for index, generated in enumerate(quartile_plan(128, VALIDATION_SEED_START)):
        if generated["render_seed"] in existing_seeds:
            raise ValueError("new roll validation render seed intersects an existing recorded source seed")
        result.append({"index": 1024 + index, "split": "roll_validation",
                       "layout_id": f"geoprr_roll_v22_validation_g{index:04d}", "render_seed": generated["render_seed"],
                       "states": generated["states"], "vertex_order": generated["vertex_order"],
                       "roll_degrees": angles[1024 + index], "parent_layout_id": None,
                       "parent_sample_ids": [], "state_policy": "interior"})
    return result


def compare_parent_factors(parent_root, root):
    values, normalization = [], []
    for folder in (Path(parent_root), root):
        with gzip.open(folder / "factors.json.gz", "rt") as handle:
            snapshot = json.load(handle)["fixed_factors"]
        changes = []
        snapshot = _normalize_asset_paths(snapshot, folder / "asset_shim", changes)
        _canonical_circle_faces(snapshot)
        for obj in snapshot["g"]["objects"]:
            if obj["name"] == snapshot["c"]["name"]:
                for key in ("rotation_euler", "rotation_quaternion", "rotation_axis_angle"):
                    obj.pop(key)
        for key in ("rotation_euler", "matrix_world"):
            snapshot["c"].pop(key)
        values.append(snapshot)
        normalization.append(changes)
    if values[0] != values[1]:
        raise ValueError("rolled source changed parent factors beyond camera rotation and declared asset representations")
    return {"parent_nonroll_factors_equal": True,
            "declared_changes": ["g.objects.Camera orientation", "c.rotation_euler", "c.matrix_world"],
            "asset_path_resolutions": normalization,
            "Circle_faces": "existing complete oriented face/corner-UV enumeration normalization"}


def prepare_group(root, plan, allowed_scenes):
    marker = json.loads((root / "roll_group.json").read_text())
    expected = {key: plan[key] for key in ("split", "layout_id", "render_seed", "states", "state_policy", "roll_degrees")}
    if (marker["status"] != "complete" or marker["protocol"] != PROTOCOL
            or any(marker[key] != value for key, value in expected.items())
            or marker["camera_roll"]["optical_axis_max_abs_difference"] > 1e-6
            or marker["in_group_nonintervened_factors_equal"] is not True):
        raise ValueError("rendered physical roll group differs from the frozen source plan")
    report = prepare_state_group(root, plan, allowed_scenes)
    rows = read_rows(root / "manifests/source_images.jsonl")
    for index, row in enumerate(rows):
        row["split"] = plan["split"]
        row["metadata"]["roll_source"] = {
            "protocol": PROTOCOL, "split": plan["split"], "layout_id": plan["layout_id"],
            "render_seed": plan["render_seed"], "roll_degrees": plan["roll_degrees"], "state_index": index,
            "parent_layout_id": plan["parent_layout_id"],
            "parent_sample_id": plan["parent_sample_ids"][index] if plan["split"] == "train" else None,
            "camera_roll_record": str(root / "camera_roll.json")}
        bbox, metadata = row["metadata"]["dial_bbox"], row["metadata"]
        if not (0 <= bbox[0] < bbox[2] <= metadata["image_width"]
                and 0 <= bbox[1] < bbox[3] <= metadata["image_height"]):
            raise ValueError("new rolled casing bbox leaves its actual rendered image")
    write_rows(root / "manifests/source_images.jsonl", rows)
    comparison = compare_parent_factors(plan["parent_group_root"], root) if plan["split"] == "train" else None
    report.update(protocol=PROTOCOL, status="complete", split=plan["split"], roll_degrees=plan["roll_degrees"],
                  render_seed=plan["render_seed"], parent_comparison=comparison,
                  optical_axis_max_abs_difference=marker["camera_roll"]["optical_axis_max_abs_difference"],
                  full_projection_export_after_camera_roll=True, GT_or_population_clipped=False)
    write_json(root / "roll_prepared.json", report)
    return report


def assemble(root, plan, original_train, validation):
    original, old_validation = read_rows(original_train), read_rows(validation)
    if len(original) != 18590 or len(old_validation) != 1506:
        raise ValueError("original source training or validation population changed")
    train_rows, val_rows, reports = [], [], []
    for group in plan:
        directory = root / "groups" / group["layout_id"]
        report = json.loads((directory / "roll_prepared.json").read_text())
        if (report["status"] != "complete" or report["protocol"] != PROTOCOL
                or report["render_seed"] != group["render_seed"] or report["roll_degrees"] != group["roll_degrees"]):
            raise ValueError("roll source assembly found an incomplete physical group")
        rows = read_rows(directory / "manifests/source_images.jsonl")
        if len(rows) != 4 or any(row["group_id"] != group["layout_id"] or row["split"] != group["split"] for row in rows):
            raise ValueError("prepared roll group population differs")
        for index, row in enumerate(rows):
            declaration = row["metadata"]["roll_source"]
            if (row["sample_id"] != f"{group['layout_id']}_s{index:02d}"
                    or abs(normalized_target(row) - group["states"][index]) > 1e-6
                    or declaration["state_index"] != index
                    or any(declaration[key] != group[key] for key in ("split", "layout_id", "render_seed", "roll_degrees"))):
                raise ValueError("prepared roll row differs from its declared physical layout/state")
        reports.append(report)
        (train_rows if group["split"] == "train" else val_rows).extend(rows)
    combined = original + train_rows
    source_split_check(combined, old_validation)
    if (len(train_rows), len(val_rows), len(combined)) != (4096, 512, 22686):
        raise ValueError("physical roll source counts differ from4096/512/22686")
    if combined[:18590] != original:
        raise ValueError("roll source changed existing18590 rows")
    train_ids, val_ids = {row["sample_id"] for row in combined}, {row["sample_id"] for row in val_rows}
    existing_ids = train_ids | {row["sample_id"] for row in old_validation}
    train_groups, val_groups = {row["group_id"] for row in combined}, {row["group_id"] for row in val_rows}
    existing_groups = train_groups | {row["group_id"] for row in old_validation}
    train_plan = [row for row in plan if row["split"] == "train"]
    val_plan = [row for row in plan if row["split"] == "roll_validation"]
    train_seeds, val_seeds = {row["render_seed"] for row in train_plan}, {row["render_seed"] for row in val_plan}
    existing_seeds, missing = recorded_seeds(original + old_validation)
    if (len(train_ids) != len(combined) or len(val_ids) != len(val_rows) or existing_ids & val_ids
            or existing_groups & val_groups or (existing_seeds | train_seeds) & val_seeds):
        raise ValueError("roll validation overlaps existing sample IDs, layouts or recorded render seeds")
    paths = {"original_train_manifest": str(original_train), "train_roll_manifest": str(root / "manifests/train_roll.jsonl"),
             "combined_train_manifest": str(root / "manifests/train_combined.jsonl"),
             "roll_validation_manifest": str(root / "manifests/validation_roll.jsonl"),
             "original_validation_manifest": str(validation)}
    write_rows(Path(paths["train_roll_manifest"]), train_rows)
    write_rows(Path(paths["combined_train_manifest"]), combined)
    write_rows(Path(paths["roll_validation_manifest"]), val_rows)
    report = {"protocol": PROTOCOL, "status": "complete", "paths": paths, "original_training_images": len(original),
              "train_roll_images": len(train_rows), "combined_training_images": len(combined),
              "roll_validation_images": len(val_rows), "original_validation_images": len(old_validation),
              "train_layout_ids": [row["layout_id"] for row in train_plan],
              "roll_validation_layout_ids": [row["layout_id"] for row in val_plan],
              "train_render_seeds": sorted(train_seeds), "roll_validation_render_seeds": sorted(val_seeds),
              "original_source_rows_exactly_preserved": True, "source_original_validation_scene_disjoint": True,
              "roll_validation_holdout": {"sample_ID_disjoint": True, "layout_group_disjoint": True,
                  "generated_and_recorded_seed_disjoint": True, "background_independent": False,
                  "scope": "independent new layouts and render seeds; shared actual source-training HDR",
                  "existing_rows_without_recorded_seed": missing,
                  "all_historical_raw_seed_provenance_available": missing == 0},
              "roll_rng": {"seed": ROLL_SEED, "distribution_degrees": [-180., 180.], "one_shared_roll_per_group": True},
              "new_group_checks": {"passed_groups": len(reports),
                  "minimum_gt_support_mass": min(item["minimum_gt_support_mass"] for item in reports),
                  "maximum_gt_composed_error_percent_fs": max(item["maximum_gt_composed_error_percent_fs"] for item in reports),
                  "maximum_optical_axis_error": max(item["optical_axis_max_abs_difference"] for item in reports)},
              "failure_policy": "retain failed groups and stop; no state/roll resampling or dropped rows"}
    write_json(root / "prepared.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args(argv)
    original, validation = read_rows(args.original_train), read_rows(args.validation)
    home = args.original_train.resolve().parents[3]
    original_plan = json.loads((home / "rendering/compositional_source_v1/generation_plan.json").read_text())
    expected = generation_plan(original_plan, original, validation)
    stored = json.loads((args.root / "generation_plan.json").read_text())
    if stored != expected:
        raise ValueError("stored roll generation plan differs from current source and fixed source-only RNG")
    if args.group_index is None:
        report = assemble(args.root, stored, args.original_train, args.validation)
    else:
        group = stored[args.group_index]
        allowed = {Path(row["metadata"]["scene_name"]).stem for row in original}
        report = prepare_group(args.root / "groups" / group["layout_id"], group, allowed)
    print(json.dumps({key: report[key] for key in ("protocol", "status")}, allow_nan=False))


if __name__ == "__main__":
    main()
