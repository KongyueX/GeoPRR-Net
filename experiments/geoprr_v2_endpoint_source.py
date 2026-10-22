"""Physically rendered endpoint replacement for frozen source-training groups."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path


DATA_PROTOCOL = "geoprr_v2_endpoint_source_v11"
STATES = [0., .005, .02, 1.]
REPLACEMENT = {"groups": 128, "states": STATES, "protocol": DATA_PROTOCOL}
RENDER_SETTINGS = {"width": 1920, "height": 1080, "samples": 200, "threads": 4,
                   "device": "CUDA", "image_format": "JPEG", "image_quality": 95}
FIXED_GEOMETRY = ("homography", "homography_bbox", "dial_bbox", "long_interval_degree", "long_num",
                  "image_width", "image_height", "scene_name")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def endpoint_generation_plan(existing_plan):
    if len(existing_plan) < 128:
        raise ValueError("endpoint generation requires the first 128 frozen source groups")
    result = []
    for index, old in enumerate(existing_plan[:128]):
        if (old["index"] != index or old["render_seed"] != 20262020 + index
                or old["layout_id"] != f"geoprr_compositional_v1_g{index:04d}"
                or sorted(old["vertex_order"]) != list(range(4)) or len(old["states"]) != 4
                or any(not k / 4 < state < (k + 1) / 4 for k, state in enumerate(old["states"]))):
            raise ValueError("the original source generation plan differs")
        result.append({"index": index, "old_layout_id": old["layout_id"],
                       "layout_id": f"geoprr_generalization_endpoint_v11_g{index:04d}",
                       "render_seed": old["render_seed"], "states": list(STATES),
                       "old_states": list(old["states"]), "vertex_order": list(old["vertex_order"])})
    return result


def replacement_mapping(plan):
    return {f"{group['old_layout_id']}_s{index:02d}": f"{group['layout_id']}_s{index:02d}"
            for group in plan for index in range(4)}


def _target(row):
    if "ground_truth" in row:
        return (row["ground_truth"] - row["scale_start"]) / (row["scale_end"] - row["scale_start"])
    return row.get("normalized_progress", row.get("y"))


def _fixed_geometry(metadata):
    return {**{key: metadata[key] for key in FIXED_GEOMETRY if key in metadata},
            "ScaleMark_keypoints": [{"type": item.get("type"), "all_kp": item.get("all_kp")}
                for item in metadata.get("keypoints", []) if str(item.get("type", "")).lower() == "scalemark"]}


def _mapped_edges(edges, mapping):
    result = []
    for edge in edges:
        if isinstance(edge, dict):
            result.append((mapping.get(edge["sample_a"], edge["sample_a"]),
                           mapping.get(edge["sample_b"], edge["sample_b"])))
        else:
            result.append(tuple(mapping.get(sid, sid) for sid in edge))
    if len(result) != len(set(result)):
        raise ValueError("duplicate exposure edges")
    return sorted(result)


def replace_source_population(anchor_rows, anchor_edges, plan, new_rows, new_edges):
    mapping = replacement_mapping(plan)
    old = {row["sample_id"]: row for row in anchor_rows}
    new = {row["sample_id"]: row for row in new_rows}
    if len(old) != len(anchor_rows) or len(new) != len(new_rows) or set(new) != set(mapping.values()):
        raise ValueError("endpoint replacement IDs differ from the frozen group plan")
    if not set(mapping) <= set(old) or set(new) & set(old):
        raise ValueError("replacement must replace existing quartile IDs with distinct new IDs")
    for group in plan:
        if group["states"] != STATES:
            raise ValueError("endpoint states differ from the frozen replacement")
        for index, state in enumerate(STATES):
            a = old[f"{group['old_layout_id']}_s{index:02d}"]
            b = new[f"{group['layout_id']}_s{index:02d}"]
            if abs(_target(a) - group["old_states"][index]) > 1e-6 or abs(_target(b) - state) > 1e-6:
                raise ValueError("physical states differ from the frozen old/new plan")
            if (_fixed_geometry(a["metadata"]) != _fixed_geometry(b["metadata"])
                    or any(a.get(key) != b.get(key) for key in ("dataset", "split", "meter_id", "scale_start", "scale_end"))
                    or b["group_id"] != group["layout_id"]):
                raise ValueError("endpoint replacement changed fixed source geometry or physical scale")
            span = b["metadata"]["long_interval_degree"] * (b["metadata"]["long_num"] - 1)
            if not 0 < span < 360 or abs(b["metadata"]["pointer_angle"] / span - state) > 1e-6:
                raise ValueError("endpoint rotation or distinct start/end arc is invalid")
    selected_edges = [edge for edge in anchor_edges if edge["sample_a"] in mapping or edge["sample_b"] in mapping]
    if any(edge["sample_a"] not in mapping or edge["sample_b"] not in mapping for edge in selected_edges):
        raise ValueError("replacement group has an exposure edge crossing its fixed group population")
    if _mapped_edges(selected_edges, mapping) != _mapped_edges(new_edges, {}):
        raise ValueError("endpoint edges differ from the mapped original C4 exposure")
    for edge in new_edges:
        a, b = new[edge["sample_a"]], new[edge["sample_b"]]
        if (edge.get("kind") != "state" or edge.get("layout_id") != a["group_id"]
                or a["group_id"] != b["group_id"]
                or edge.get("factors_a") != a["metadata"].get("factors")
                or edge.get("factors_b") != b["metadata"].get("factors")):
            raise ValueError("endpoint exposure factors differ from their actual image records")
    by_edge = {(edge["sample_a"], edge["sample_b"]): edge for edge in new_edges}
    rows = [deepcopy(new[mapping[row["sample_id"]]] if row["sample_id"] in mapping else row) for row in anchor_rows]
    edges = [deepcopy(by_edge[(mapping[edge["sample_a"]], mapping[edge["sample_b"]])]
                      if edge["sample_a"] in mapping else edge) for edge in anchor_edges]
    if len(rows) != len(anchor_rows) or len(edges) != len(anchor_edges):
        raise ValueError("endpoint replacement changed source population or exposure counts")
    return rows, edges, {"groups": len(plan), "replaced_unique_images": len(mapping), "old_to_new": mapping,
                         "retained_unique_images": len(rows) - len(mapping), "recipient_images": len(edges)}


def _normalize_asset_paths(value, asset_shim, changes, location=""):
    if isinstance(value, dict):
        return {key: _normalize_asset_paths(item, asset_shim, changes, f"{location}/{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_asset_paths(item, asset_shim, changes, f"{location}/{index}") for index, item in enumerate(value)]
    if isinstance(value, str) and str(asset_shim) in value:
        resolved = str(Path(value).resolve())
        changes.append({"field": location, "recorded": value, "resolved": resolved})
        return resolved
    return value


def _circle_face_records(mesh):
    """Associate every ordered face corner with its actual mesh-loop UV."""
    count = sum(len(face[0]) for face in mesh["polygons"])
    if any(len(values) != count for values in mesh["uv_layers"].values()):
        raise ValueError("Circle UV layer length does not match its polygon corners")
    offset, records = 0, []
    for vertices, material, smooth in mesh["polygons"]:
        end = offset + len(vertices)
        uv = tuple((name, tuple(tuple(value) for value in values[offset:end]))
                   for name, values in sorted(mesh["uv_layers"].items()))
        records.append((tuple(vertices), material, smooth, uv))
        offset = end
    return sorted(records)


def _canonical_circle_faces(fixed):
    meshes = fixed["g"]["meshes"]
    if "Circle" not in meshes:
        return None
    mesh = meshes["Circle"]
    records = _circle_face_records(mesh)
    original = deepcopy(mesh["polygons"])
    mesh["polygons"] = [[list(vertices), material, smooth] for vertices, material, smooth, _uv in records]
    mesh["uv_layers"] = {name: [list(value) for _vertices, _material, _smooth, layers in records
                               for layer, values in layers if layer == name for value in values]
                         for name in mesh["uv_layers"]}
    return {"mesh": "Circle", "faces": len(records), "face_enumeration_changed": original != mesh["polygons"]}


def compare_fixed_factors(old_root, new_root):
    snapshots = []
    path_changes = []
    face_changes = []
    for root in (old_root, new_root):
        with gzip.open(root / "factors.json.gz", "rt") as handle:
            fixed = json.load(handle)["fixed_factors"]
        changes = []
        normalized = _normalize_asset_paths(fixed, root / "asset_shim", changes)
        face_changes.append(_canonical_circle_faces(normalized))
        snapshots.append(normalized)
        path_changes.append(changes)
    if snapshots[0] != snapshots[1]:
        raise ValueError("endpoint rendering changed fixed g/a/c beyond asset paths and Circle face enumeration")
    return {"fixed_g_a_c_equal_after_declared_representation_normalization": True,
            "normalization": "asset_shim paths resolve through actual symlinks; only g/meshes/Circle face enumeration is sorted as complete vertex-winding/material/smooth/per-layer ordered-corner-UV records",
            "asset_path_resolutions": {"old": path_changes[0], "new": path_changes[1]},
            "circle_face_enumeration": {"old": face_changes[0], "new": face_changes[1]},
            "face_corner_uv_associations_and_vertex_winding": "exact; no cyclic rotation, reversal or numeric tolerance",
            "numerical_or_render_setting_normalization": False}


def prepare_endpoint_group(root, old_source, group, allowed_scenes):
    import numpy as np
    import torch
    from experiments.geoprr_v2_data import GeoPRRV2Dataset
    from experiments.geoprr_v2_reference_relation import reference_geometry_targets
    from experiments.prepare_geoprr_v2_compositional_data import prepare_group

    result = prepare_group(root, group, allowed_scenes)
    rendered = read_json(root / "group.json")
    provenance = rendered["provenance"]
    if (provenance.get("state_policy") != "endpoint_v11" or provenance["execution"] != "CYCLES_CUDA"
            or provenance["samples"] != 200 or provenance["threads"] != 4
            or provenance["image_format"] != "JPEG" or provenance["image_quality"] != 95
            or provenance["longmark_projection_size"] != [1920, 1080]):
        raise ValueError("endpoint renderer differs from the fixed formal source settings")
    previous = old_source / "groups" / group["old_layout_id"]
    new_rows = read_jsonl(root / "manifests/source_images.jsonl")
    old_rows = read_jsonl(previous / "manifests/source_images.jsonl")
    old_edges = read_jsonl(previous / "manifests/train_edges.jsonl")
    new_edges = read_jsonl(root / "manifests/train_edges.jsonl")
    replace_source_population(old_rows, old_edges, [group], new_rows, new_edges)
    factors = compare_fixed_factors(previous, root)
    dataset = GeoPRRV2Dataset(root / "manifests/source_images.jsonl", augment=False, image_size=768, output_size=192)
    role_checks = []
    for index, row in enumerate(dataset.rows):
        sample = dataset[index]
        span = dataset._calibrations[index]["angles"][-1]
        labels = reference_geometry_targets(sample["roi_to_canonical"][None], sample["y"][None], torch.tensor([span]))
        if not 0 < span < 2 * math.pi or not all(torch.isfinite(value).all() for value in labels.values()):
            raise ValueError("endpoint virtual-role geometry is not finite on a distinct-endpoint arc")
        if not np.isfinite(sample["image"].numpy()).all():
            raise ValueError("endpoint RGB input is not finite")
        role_checks.append({"sample_id": row["sample_id"], "span_radians": float(span), "virtual_role_targets_finite": True})
    result.update(protocol=DATA_PROTOCOL, states=list(STATES), fixed_factors_comparison=factors,
                  role_target_checks=role_checks, render_settings=RENDER_SETTINGS)
    write_json(root / "endpoint_prepared.json", result)
    return result


def verify_endpoint_training_contract(data_contract, anchor_data_contract):
    root = Path(data_contract["paths"]["train_manifest"]).parent.parent
    prepared = read_json(root / "prepared.json")
    plan = read_json(root / "generation_plan.json")
    old_source = Path(prepared["original_source_root"])
    if (prepared["protocol"] != DATA_PROTOCOL or prepared["replacement"] != REPLACEMENT
            or plan != endpoint_generation_plan(read_json(old_source / "generation_plan.json"))):
        raise ValueError("endpoint prepared data or generation plan differs from the frozen source replacement")
    contract = prepared["replacement_contract"]
    expected_paths = {"train_manifest": str(root / "manifests/train.jsonl"),
                      "pair_manifest": str(root / "manifests/train_edges.jsonl"),
                      "val_manifest": anchor_data_contract["paths"]["val_manifest"]}
    if (any(data_contract["paths"][key] != value for key, value in expected_paths.items())
            or contract["anchor_paths"] != {key: anchor_data_contract["paths"][key] for key in expected_paths}
            or anchor_data_contract["paths"]["train_manifest"] != str(old_source / "manifests/train.jsonl")
            or anchor_data_contract["paths"]["pair_manifest"] != str(old_source / "manifests/train_edges.jsonl")):
        raise ValueError("endpoint source or validation manifest paths differ from the recorded replacement")
    if data_contract["validation"] != anchor_data_contract["validation"]:
        raise ValueError("endpoint training changed the original validation supervision")
    old, new = anchor_data_contract["train"], data_contract["train"]
    if {key: value for key, value in old.items() if key not in ("rows", "edges")} != {
            key: value for key, value in new.items() if key not in ("rows", "edges")}:
        raise ValueError("endpoint source settings or exposure counts changed")
    mapping = replacement_mapping(plan)
    replacement = {row["sample_id"]: row for row in contract["new_rows"]}
    if len(replacement) != 512 or set(replacement) != set(mapping.values()):
        raise ValueError("prepared endpoint contract does not contain the frozen 512 replacements")
    old_by_id = {row["sample_id"]: row for row in old["rows"]}
    for group in plan:
        for index, state in enumerate(STATES):
            a = old_by_id[f"{group['old_layout_id']}_s{index:02d}"]
            b = replacement[f"{group['layout_id']}_s{index:02d}"]
            if (abs(a["y"] - group["old_states"][index]) > 1e-6 or abs(b["y"] - state) > 1e-6
                    or b["group_id"] != group["layout_id"]
                    or {key: value for key, value in a["geometry"].items() if key != "pointer_angle"}
                    != {key: value for key, value in b["geometry"].items() if key != "pointer_angle"}):
                raise ValueError("prepared endpoint states or fixed supervision differ from the original groups")
            span = b["geometry"]["long_interval_degree"] * (b["geometry"]["long_num"] - 1)
            if not 0 < span < 360 or abs(b["geometry"]["pointer_angle"] / span - state) > 1e-6:
                raise ValueError("prepared endpoint physical rotation is inconsistent")
    expected_rows = [replacement[mapping[row["sample_id"]]] if row["sample_id"] in mapping else row for row in old["rows"]]
    expected_edges = _mapped_edges(old["edges"], mapping)
    if (new["rows"] != expected_rows or _mapped_edges(new["edges"], {}) != expected_edges
            or _mapped_edges(data_contract["exposure_edges"], {}) != expected_edges):
        raise ValueError("endpoint training changed an undeclared source row or mapped exposure edge")
    return {"protocol": DATA_PROTOCOL, "groups": 128, "replaced_unique_images": 512,
            "unchanged_validation": True, "exact_source_replacement_verified": True}


def assemble_endpoint_source(root, old_source, validation, data_root):
    from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset
    from experiments.train_geoprr_v2 import source_split_check

    plan = read_json(root / "generation_plan.json")
    if plan != endpoint_generation_plan(read_json(old_source / "generation_plan.json")):
        raise ValueError("endpoint plan differs from the frozen original first 128 groups")
    rows, edges, reports = [], [], []
    for group in plan:
        directory = root / "groups" / group["layout_id"]
        reports.append(read_json(directory / "endpoint_prepared.json"))
        rows.extend(read_jsonl(directory / "manifests/source_images.jsonl"))
        edges.extend(read_jsonl(directory / "manifests/train_edges.jsonl"))
    train, pairs, report = replace_source_population(read_jsonl(old_source / "manifests/train.jsonl"),
        read_jsonl(old_source / "manifests/train_edges.jsonl"), plan, rows, edges)
    val = read_jsonl(validation)
    source_split_check(train, val)
    write_rows(root / "manifests/train.jsonl", train)
    write_rows(root / "manifests/train_edges.jsonl", pairs)
    dataset = ReferenceRelationDataset(root / "manifests/train.jsonl", data_root=data_root,
        data_config={"image_size": 768, "output_size": 192}, augment=False,
        pair_manifest=root / "manifests/train_edges.jsonl")
    if (len(dataset.rows), len(dataset), len(dataset.edges), len(val)) != (18590, 22686, 8192, 1506):
        raise ValueError("endpoint replacement changed the fixed source populations or per-epoch exposure")
    replacement_rows = [row for row in dataset.supervision_contract()["rows"] if row["sample_id"] in set(report["old_to_new"].values())]
    result = {"protocol": DATA_PROTOCOL, "replacement": REPLACEMENT, "original_source_root": str(old_source),
              "population_replacement": report, "unique_images": 18590, "donor_images": 22686,
              "recipient_images": 8192, "image_forwards_per_epoch": 30878, "batches_per_epoch": 709,
              "validation_images": 1506, "render_settings": RENDER_SETTINGS,
              "maximum_gt_composed_error_percent_fs": max(item["maximum_gt_composed_error_percent_fs"] for item in reports),
              "minimum_gt_support_mass": min(item["minimum_gt_support_mass"] for item in reports),
              "replacement_contract": {"new_rows": replacement_rows, "anchor_paths": {
                  "train_manifest": str(old_source / "manifests/train.jsonl"),
                  "pair_manifest": str(old_source / "manifests/train_edges.jsonl"), "val_manifest": str(validation)}},
              "scope": "physical source data and exact replacement contract; no learned performance"}
    write_json(root / "prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args()
    root, old = args.root.resolve(), args.old_source.resolve()
    if args.group_index is None:
        result = assemble_endpoint_source(root, old, args.validation.resolve(), args.data_root.resolve())
    else:
        plan = read_json(root / "generation_plan.json")
        if plan != endpoint_generation_plan(read_json(old / "generation_plan.json")):
            raise ValueError("endpoint generation plan changed")
        group = plan[args.group_index]
        allowed = {Path(row["metadata"]["scene_name"]).stem for row in read_jsonl(args.original_train)}
        result = prepare_endpoint_group(root / "groups" / group["layout_id"], old, group, allowed)
    print(json.dumps({key: value for key, value in result.items() if key not in
        ("replacement_contract", "population_replacement", "geometry_audits", "gt_compositions", "fixed_factors_comparison")}), flush=True)


if __name__ == "__main__":
    main()
