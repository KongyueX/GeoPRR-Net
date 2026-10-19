"""Prepare the separately recorded source groups for state composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_data import GeoPRRV2Dataset, GeometryConfig, normalized_target, read_rows
from experiments.prepare_geoprr_v2_data import syncg_row
from experiments.train_geoprr_v2 import source_split_check, write_json


PROTOCOL = "geoprr_v2_compositional_source_v1"


def generation_plan(groups: int = 1024, seed: int = 20262020) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"index": index, "layout_id": f"geoprr_compositional_v1_g{index:04d}",
             "render_seed": seed + index, "states": ((np.arange(4) + rng.random(4)) / 4).tolist(),
             "vertex_order": rng.permutation(4).tolist()} for index in range(groups)]


def edges_for_group(rows: list[dict], order: list[int]) -> tuple[list[dict], list[dict]]:
    if sorted(order) != list(range(4)) or len(rows) != 4:
        raise ValueError("state groups require four images and a permutation of their vertices")
    training, heldout = [], []
    for position, donor in enumerate(order):
        for offset in (1, 2, 3):
            recipient = order[(position + offset) % 4]
            a, b = rows[donor], rows[recipient]
            edge = {"sample_a": a["sample_id"], "sample_b": b["sample_id"], "kind": "state",
                    "layout_id": a["metadata"]["layout_id"],
                    "factors_a": a["metadata"]["factors"], "factors_b": b["metadata"]["factors"]}
            (heldout if offset == 2 else training).append(edge)
    return training, heldout


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def prepare_group(root: Path, plan: dict, allowed_scenes: set[str]) -> dict:
    group = json.loads((root / "group.json").read_text())
    if group.get("diagnostic_only", True) or group.get("role") != "source_training":
        raise ValueError("formal source preparation requires the source_training rendering role")
    if (group["layout_id"] != plan["layout_id"] or group["states"] != plan["states"]
            or group["provenance"]["seed"] != plan["render_seed"] or len(group["images"]) != 4):
        raise ValueError("rendered group differs from its frozen generation plan")
    if len(group["identity_checks"]) != 11 or not all(row["nonintervened_equal"] for row in group["identity_checks"]):
        raise ValueError("the four-state render factor comparisons are incomplete")
    rows = []
    for image in group["images"]:
        row = syncg_row(Path(image["annotation_path"]), root / "images")
        row.update(split="train", group_id=group["layout_id"])
        if Path(row["metadata"]["scene_name"]).stem not in allowed_scenes:
            raise ValueError("new source image uses a background outside the original source-training partition")
        rows.append(row)
    if any(abs(normalized_target(row) - state) > 1e-6 for row, state in zip(rows, plan["states"], strict=True)):
        raise ValueError("generated physical readings differ from the frozen states")
    training, heldout = edges_for_group(rows, plan["vertex_order"])
    manifests = root / "manifests"
    write_rows(manifests / "source_images.jsonl", rows)
    write_rows(manifests / "train_edges.jsonl", training)
    write_rows(manifests / "heldout_edges.jsonl", heldout)
    dataset = GeoPRRV2Dataset(manifests / "source_images.jsonl", augment=False)
    samples = [dataset[index] for index in range(4)]
    for sample in samples[1:]:
        for key in ("field_target", "field_mask", "support_target", "roi_to_canonical"):
            if not np.array_equal(sample[key].numpy(), samples[0][key].numpy()):
                raise ValueError(f"fixed-camera state group changed {key}")
    config = GeometryConfig()
    compositions = []
    for donor, a in enumerate(samples):
        for recipient, b in enumerate(samples):
            weights = b["point_target"].numpy()[0] * a["field_mask"].numpy()[0]
            mass = float(weights.sum())
            prediction = float((a["field_target"].numpy() * weights).sum() / mass)
            error = abs(prediction - normalized_target(rows[recipient]))
            if mass < config.min_support_mass or error > config.max_readout_error:
                raise ValueError("GT cross-state composition violates the original source geometry contract")
            compositions.append({"donor": donor, "recipient": recipient, "gt_support_mass": mass,
                                 "gt_composed_error": error})
    paired = GeoPRRV2Dataset(manifests / "source_images.jsonl", augment=False,
                            pair_manifest=manifests / "train_edges.jsonl")
    # Factor validation and edge expansion use the exact training loader.
    if len(paired) != 8 or paired.controlled_edge_count != 8:
        raise ValueError("the four-state training cycle did not produce eight donor items")
    report = {"protocol": PROTOCOL, "layout_id": group["layout_id"], "images": 4,
              "training_edges": 8, "heldout_edges": 4, "fixed_field_and_camera_equal": True,
              "geometry_audits": [{"sample_id": row["sample_id"], **sample["geometry_audit"]}
                                  for row, sample in zip(rows, samples, strict=True)],
              "gt_compositions": compositions,
              "minimum_gt_support_mass": min(row["gt_support_mass"] for row in compositions),
              "maximum_gt_composed_error_percent_fs": 100 * max(row["gt_composed_error"] for row in compositions)}
    write_json(root / "prepared.json", report)
    return report


def assemble(root: Path, plan: list[dict], original_train: Path, validation: Path) -> dict:
    original = read_rows(original_train)
    val = read_rows(validation)
    if len(original) != 14494 or len(val) != 1506:
        raise ValueError("source composition requires the original 14494/1506 train/validation populations")
    images, training, heldout, reports = [], [], [], []
    for group in plan:
        directory = root / "groups" / group["layout_id"]
        reports.append(json.loads((directory / "prepared.json").read_text()))
        for output, name in ((images, "source_images"), (training, "train_edges"), (heldout, "heldout_edges")):
            output.extend(read_rows(directory / "manifests" / f"{name}.jsonl") if name == "source_images"
                          else [json.loads(line) for line in (directory / "manifests" / f"{name}.jsonl").read_text().splitlines()])
    combined = original + images
    source_split_check(combined, val)
    manifests = root / "manifests"
    write_rows(manifests / "train.jsonl", combined)
    write_rows(manifests / "source_groups.jsonl", images)
    write_rows(manifests / "train_edges.jsonl", training)
    write_rows(manifests / "heldout_edges.jsonl", heldout)
    dataset = GeoPRRV2Dataset(manifests / "train.jsonl", pair_manifest=manifests / "train_edges.jsonl")
    expected_items = len(original) + len(training)
    if len(dataset) != expected_items or dataset.unique_image_count != len(combined):
        raise ValueError("combined source edge expansion differs from the frozen exposure count")
    report = {"protocol": PROTOCOL, "groups": len(plan), "new_unique_images": len(images),
              "original_training_images": len(original), "unique_training_images": len(combined),
              "training_items_per_epoch": len(dataset), "recipient_forwards_per_epoch": len(training),
              "image_forwards_per_epoch": len(dataset) + len(training),
              "training_edges": len(training), "heldout_edges": len(heldout),
              "source_validation_images": len(val),
              "maximum_gt_composed_error_percent_fs": max(row["maximum_gt_composed_error_percent_fs"] for row in reports),
              "minimum_gt_support_mass": min(row["minimum_gt_support_mass"] for row in reports),
              "scope": "generated source data and GT geometry contract; no learned accuracy"}
    write_json(root / "prepared.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args()
    plan = json.loads((args.root / "generation_plan.json").read_text())
    if args.group_index is None:
        report = assemble(args.root, plan, args.original_train, args.validation)
    else:
        group = plan[args.group_index]
        allowed = {Path(row["metadata"]["scene_name"]).stem for row in read_rows(args.original_train)}
        report = prepare_group(args.root / "groups" / group["layout_id"], group, allowed)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
