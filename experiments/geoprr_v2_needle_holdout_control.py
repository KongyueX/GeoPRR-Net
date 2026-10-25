"""Fixed five-epoch source needle holdout control with a source-ID learning gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import time


PROTOCOL = "geoprr_v2_needle_holdout_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_needle_holdout_control_20261025"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_source_needle_holdout_control_preflight_20261025/result.json"
PREPARED_RELATIVE = "diagnostics/geoprr_source_needle_holdout_control_population_20261025"
ORIGINAL_SOURCE_RELATIVE = "rendering/generalization_endpoints_v11_20261023"
ROLL_SOURCE_RELATIVE = "rendering/generalization_roll_v22_20261024"
SEEN = ("canonical_pointer_01", "canonical_pointer_02")
HELDOUT = "canonical_pointer_03"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {"factory": "experiments.geoprr_v2_high_resolution_direct.create_model(pretrained=True)",
              "encoder": "mobilenetv4_conv_medium.e500_r224_in1k", "semantic_dim": 1280,
              "parameters": 8435793, "parameter_tensors": 233, "state_tensors": 464,
              "readout": "GAP1/full neck/Linear1280to1 std.001 bias0/float32 sigmoid",
              "all_encoder_and_point_trainable": True, "source_training_BN_updates": True},
    "data": {"image_size": 512, "output_size": 128, "require_geometry": False, "pair_manifest": None},
    "photometric": {"contrast": [.8, 1.2], "saturation": [.8, 1.2], "brightness": [-.1, .1]},
    "source": {"heldout_pointer": HELDOUT, "training_pointers": list(SEEN),
               "parent_layouts_train": 687, "original_train_images": 2748, "roll_train_images": 2748,
               "train_unique_images": 5496, "train_repeat": 1, "train_calls_per_epoch": 86,
               "last_train_batch": 56, "ID_layouts": 73, "ID_images": 292, "ID_calls": 5,
               "OOD_layouts": 55, "OOD_images": 220, "OOD_calls": 4},
    "training": {"seed": 20262020, "epochs": 5, "max_epochs": 5, "batch_size": 64,
                 "validation_batch_size": 64, "workers": 4, "optimizer": "AdamW",
                 "learning_rate": .0003, "weight_decay": .0001, "gradient_clip": 5.,
                 "loss": "SmoothL1", "smooth_l1_beta": .05, "amp": True,
                 "scheduler": None, "validation_before_epoch5": False, "resume": False},
    "learning_gate": {"constant_prediction": .5, "ratio": .5, "failure_error": 1.,
                      "denominator": 292, "tail_count": 30},
    "checkpoint": "last5 only; full state after every complete epoch; no best selection",
    "external_images_calls_backward_updates": 0,
}
EXPECTED_COUNTS = {
    "original_traceable": {"canonical_pointer_01": {"layouts": 338, "images": 1352},
                           "canonical_pointer_02": {"layouts": 349, "images": 1396},
                           "canonical_pointer_03": {"layouts": 337, "images": 1348}},
    "roll_train": {"canonical_pointer_01": {"layouts": 338, "images": 1352},
                   "canonical_pointer_02": {"layouts": 349, "images": 1396},
                   "canonical_pointer_03": {"layouts": 337, "images": 1348}},
    "roll_validation": {"canonical_pointer_01": {"layouts": 31, "images": 124},
                        "canonical_pointer_02": {"layouts": 42, "images": 168},
                        "canonical_pointer_03": {"layouts": 55, "images": 220}},
}
MANIFESTS = {"train": "manifests/train.jsonl", "id": "manifests/source_id.jsonl", "ood": "manifests/source_ood.jsonl"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tail_rows(path, count=4096):
    """Decode only the final traceable rows, leaving the old prefix unparsed."""
    with path.open("rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        blocks, newlines = [], 0
        while position and newlines <= count:
            size = min(position, 65536)
            position -= size
            handle.seek(position)
            block = handle.read(size)
            blocks.insert(0, block)
            newlines += block.count(b"\n")
    lines = b"".join(blocks).splitlines()
    if len(lines) < count:
        raise ValueError("the original traceable tail is incomplete")
    return [json.loads(line.decode("utf-8")) for line in lines[-count:]]


def _unique(records, field):
    result = {record[field]: record for record in records}
    if len(result) != len(records):
        raise ValueError(f"source metadata repeats {field}")
    return result


def _groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["layout"]].append(record)
    for group in groups.values():
        if (len(group) != 4 or {record["slot"] for record in group} != {0, 1, 2, 3}
                or len({record["state"] for record in group}) != 4
                or len({record["pointer"] for record in group}) != 1):
            raise ValueError("source layout lacks a complete four-state single-pointer closure")
    return groups


def build_roster(original, roll, validation):
    """Select complete parent closures using already saved metadata only."""
    roll = [dict(record) for record in roll]
    originals = _unique(original, "sample_id")
    _unique(roll, "sample_id")
    _unique(validation, "sample_id")
    if len(original) != len(roll) or set(originals) != {record["parent_sample_id"] for record in roll}:
        raise ValueError("roll rows do not uniquely cover the original parent samples")
    parent_slots, roll_parents = {}, {}
    for record in roll:
        parent = originals[record["parent_sample_id"]]
        if (record["parent_equal"] is not True or parent["layout"] != record["parent_layout"]
                or parent["state"] != record["state"]):
            raise ValueError("saved roll parent identity, factors or state differs")
        if record["parent_sample_id"] in parent_slots:
            raise ValueError("a roll sample repeats its original parent")
        parent_slots[record["parent_sample_id"]] = record["slot"]
        record["pointer"] = parent["pointer"]
        previous = roll_parents.setdefault(record["layout"], record["parent_layout"])
        if previous != record["parent_layout"]:
            raise ValueError("one rolled layout refers to multiple parent layouts")
    original = [{**record, "slot": parent_slots[record["sample_id"]]} for record in original]
    original_groups, roll_groups, validation_groups = map(_groups, (original, roll, validation))
    if {r["sample_id"] for r in original} & {r["sample_id"] for r in roll}:
        raise ValueError("original and rolled source sample identities intersect")
    if (len(set(roll_parents.values())) != len(roll_groups)
            or set(roll_parents.values()) != set(original_groups)):
        raise ValueError("rolled layouts are not in one-to-one parent closure")
    names = sorted({record["pointer"] for record in original})
    if len(names) != 3 or {record["pointer"] for record in validation} != set(names):
        raise ValueError("source training and validation must share exactly three saved pointer assets")
    aliases = {name: f"canonical_pointer_{index:02d}" for index, name in enumerate(names, 1)}
    populations = {"original_traceable": original, "roll_train": roll, "roll_validation": validation}
    counts = {}
    for name, records in populations.items():
        counts[name] = {alias: {"layouts": len({r["layout"] for r in records if aliases[r["pointer"]] == alias}),
                                "images": sum(aliases[r["pointer"]] == alias for r in records)} for alias in aliases.values()}
    train_ids = {r["sample_id"] for r in original + roll}
    if (train_ids & {r["sample_id"] for r in validation}
            or (set(original_groups) | set(roll_groups)) & set(validation_groups)):
        raise ValueError("source training and validation identities intersect")
    selected = {}
    for name, records in (("train", original + roll), ("id", validation), ("ood", validation)):
        selected[name] = [dict(record, pointer=aliases[record["pointer"]]) for record in records
                          if (aliases[record["pointer"]] == HELDOUT) == (name == "ood")]
        selected[name].sort(key=lambda r: (r.get("view", "roll"), r["layout"], r["slot"]))
    pref_train, pref_id = [], []
    for alias in SEEN:
        parents = sorted({r["layout"] for r in selected["train"] if r["view"] == "original" and r["pointer"] == alias})
        parent = parents[0]
        for view in ("original", "roll"):
            matches = [i for i, r in enumerate(selected["train"]) if r["view"] == view and r["slot"] == 0
                       and (r["layout"] if view == "original" else r["parent_layout"]) == parent]
            if len(matches) != 1:
                raise ValueError("preflight slot0 original/roll selection is ambiguous")
            pref_train.extend(matches)
        pref_id.append([i for i, r in enumerate(selected["id"]) if r["pointer"] == alias][:2])
    if any(len(indices) != 2 for indices in pref_id):
        raise ValueError("source-ID preflight needs two saved images per seen pointer")
    return {"records": selected, "counts": counts, "pref_indices": {"train": pref_train, "id": pref_id}}


def _pointer(reference):
    with gzip.open(reference, "rt") as handle:
        name = json.load(handle)["fixed_factors"]["g"]["selected_pointer"]
    if not isinstance(name, str) or not name:
        raise ValueError("selected_pointer must identify an existing source asset")
    return name


def _project_row(row, manifest_root, home, record):
    from experiments.geoprr_v2_data import _resolve_image
    result = {key: row[key] for key in ("sample_id", "group_id", "ground_truth", "scale_start", "scale_end",
                                       "normalized_progress", "y") if key in row}
    image_key = "roi_path" if "roi_path" in row else "image_path"
    result[image_key] = str(_resolve_image(row, manifest_root, home / "datasets").resolve())
    metadata = row.get("metadata", {})
    result["metadata"] = {key: metadata[key] for key in ("dial_bbox", "image_width", "image_height") if key in metadata}
    result["metadata"]["needle_control"] = {key: record[key] for key in
        ("pointer", "layout", "view", "slot", "state", "parent_layout", "parent_sample_id", "parent_equal") if key in record}
    return result


def _contract(populations, counts, pref_indices):
    return {"canonical_pointer_rule": "lexicographic actual selected_pointer; hold out canonical_pointer_03",
            "counts": counts, "complete_parent_closure": True, "parent_nonroll_factors_equal_saved": True,
            "state_equality_saved": True, "train_validation_layout_and_sample_disjoint": True,
            "old14494_and_original1506_read": False, "geometry_values_used": False,
            "train_repeat": 1, "RGB_input": "V26 native ROI/INTER_LINEAR512/ImageNet; geometryFalse/pairNone",
            "pref_indices": pref_indices, "populations": populations}


def prepare(home):
    destination = home / PREPARED_RELATIVE
    destination.mkdir(parents=True, exist_ok=False)
    original_path = home / ORIGINAL_SOURCE_RELATIVE / "manifests/train.jsonl"
    roll_root = home / ROLL_SOURCE_RELATIVE
    paths = {"original": original_path, "roll": roll_root / "manifests/train_roll.jsonl",
             "validation": roll_root / "manifests/validation_roll.jsonl"}
    original_rows = read_tail_rows(original_path)
    roll_rows, validation_rows = read_rows(paths["roll"]), read_rows(paths["validation"])
    if (len(original_rows), len(roll_rows), len(validation_rows)) != (4096, 4096, 512):
        raise ValueError("the frozen traceable source row counts differ")
    pointers, certifications = {}, {}
    original, roll, validation = [], [], []
    for row in original_rows:
        reference = row["metadata"]["factors"]["reference"]
        if reference not in pointers:
            pointers[reference] = _pointer(reference)
        original.append({"sample_id": row["sample_id"], "layout": row["group_id"],
                         "state": row["metadata"]["factors"]["r"], "pointer": pointers[reference], "view": "original"})
    if len(pointers) != 1024:
        raise ValueError("the original source must have1024 factor snapshots")
    for row in roll_rows:
        saved = row["metadata"]["roll_source"]
        layout = saved["layout_id"]
        if layout not in certifications:
            report = json.loads((roll_root / "groups" / layout / "roll_prepared.json").read_text())
            certifications[layout] = report["parent_comparison"]["parent_nonroll_factors_equal"]
        roll.append({"sample_id": row["sample_id"], "layout": layout, "slot": saved["state_index"],
                     "state": row["metadata"]["factors"]["r"], "parent_layout": saved["parent_layout_id"],
                     "parent_sample_id": saved["parent_sample_id"], "parent_equal": certifications[layout], "view": "roll"})
    if len(certifications) != 1024:
        raise ValueError("the roll source must have1024 saved parent certifications")
    validation_pointers = {}
    for row in validation_rows:
        saved = row["metadata"]["roll_source"]
        reference = row["metadata"]["factors"]["reference"]
        if reference not in validation_pointers:
            validation_pointers[reference] = _pointer(reference)
        validation.append({"sample_id": row["sample_id"], "layout": saved["layout_id"],
                           "slot": saved["state_index"], "state": row["metadata"]["factors"]["r"],
                           "pointer": validation_pointers[reference], "view": "roll"})
    if len(validation_pointers) != 128:
        raise ValueError("the source validation must have128 factor snapshots")
    roster = build_roster(original, roll, validation)
    if roster["counts"] != EXPECTED_COUNTS:
        raise ValueError("saved source pointer counts differ from the frozen feasibility evidence")
    by_id = {row["sample_id"]: (row, paths[view].parent) for view, rows in
             (("original", original_rows), ("roll", roll_rows), ("validation", validation_rows)) for row in rows}
    populations = {name: [_project_row(*by_id[record["sample_id"]], home, record) for record in records]
                   for name, records in roster["records"].items()}
    contract = _contract(populations, roster["counts"], roster["pref_indices"])
    for name, rows in populations.items():
        path = destination / MANIFESTS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    prepared = {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "manifests": MANIFESTS,
                "counts": roster["counts"], "source_contract": contract, "pref_indices": roster["pref_indices"],
                "preparation_counts": {"original_tail_rows":4096, "original_factor_snapshots":1024,
                    "roll_rows":4096, "roll_parent_certifications":1024, "validation_rows":512,
                    "validation_pointer_snapshots":128, "RGB_reads":0, "model_constructions":0,
                    "GPU_calls":0, "weight_reads":0}}
    write_json(destination / "prepared.json", prepared)
    validate_prepared(home)
    print(json.dumps({"protocol":PROTOCOL, "status":"complete", "counts":roster["counts"],
                      "preparation_counts":prepared["preparation_counts"]}), flush=True)


def validate_prepared(home):
    root = home / PREPARED_RELATIVE
    prepared = json.loads((root / "prepared.json").read_text())
    if (prepared["protocol"] != PROTOCOL or prepared["config"] != CONFIG or prepared["status"] != "complete"
            or prepared["counts"] != EXPECTED_COUNTS or prepared["manifests"] != MANIFESTS):
        raise ValueError("prepared needle control protocol, configuration or population counts differ")
    populations = {name: read_rows(root / path) for name, path in MANIFESTS.items()}
    if {name: len(rows) for name, rows in populations.items()} != {"train":5496, "id":292, "ood":220}:
        raise ValueError("prepared subset manifests have changed lengths")
    expected = _contract(populations, EXPECTED_COUNTS, prepared["pref_indices"])
    if expected != prepared["source_contract"]:
        raise ValueError("prepared subset rows or source contract differ; upstream metadata will not be rescanned")
    ids = [row["sample_id"] for rows in populations.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prepared source populations repeat sample identities")
    for name, rows in populations.items():
        aliases = {row["metadata"]["needle_control"]["pointer"] for row in rows}
        if aliases != ({HELDOUT} if name == "ood" else set(SEEN)):
            raise ValueError("heldout needle entered a prepared seen population")
    for population, expected_counts in (("train",{SEEN[0]:2704,SEEN[1]:2792}),
                                         ("id",{SEEN[0]:124,SEEN[1]:168}), ("ood",{HELDOUT:220})):
        actual = {alias:sum(row["metadata"]["needle_control"]["pointer"] == alias for row in populations[population])
                  for alias in expected_counts}
        if actual != expected_counts:
            raise ValueError("prepared pointer population counts differ from the frozen roster")
    train = [{"sample_id":row["sample_id"], **row["metadata"]["needle_control"]} for row in populations["train"]]
    original, roll = [r for r in train if r["view"] == "original"], [r for r in train if r["view"] == "roll"]
    original_groups, roll_groups = _groups(original), _groups(roll)
    if (len(original),len(roll),len(original_groups),len(roll_groups)) != (2748,2748,687,687):
        raise ValueError("prepared source lost its complete687 parent closures")
    parents = _unique(original, "sample_id")
    if set(parents) != {r["parent_sample_id"] for r in roll}:
        raise ValueError("prepared rolled samples do not cover each original parent once")
    if len({r["parent_layout"] for r in roll}) != 687:
        raise ValueError("prepared roll layouts do not have unique original parent layouts")
    for r in roll:
        parent = parents[r["parent_sample_id"]]
        if (r["parent_equal"] is not True or parent["layout"] != r["parent_layout"]
                or any(parent[key] != r[key] for key in ("slot","state","pointer"))):
            raise ValueError("prepared original/roll closure differs from saved metadata")
    validation_records = {name:[{"sample_id":row["sample_id"], **row["metadata"]["needle_control"]}
                               for row in populations[name]] for name in ("id","ood")}
    id_groups, ood_groups = _groups(validation_records["id"]), _groups(validation_records["ood"])
    if (len(id_groups),len(ood_groups)) != (73,55) or (set(original_groups)|set(roll_groups)) & (set(id_groups)|set(ood_groups)) or set(id_groups) & set(ood_groups):
        raise ValueError("prepared source validation layout closure or separation differs")
    pref_train, pref_id = [], []
    for alias in SEEN:
        parent = min(r["layout"] for r in original if r["pointer"] == alias)
        for view in ("original","roll"):
            pref_train.extend(i for i,r in enumerate(train) if r["view"] == view and r["slot"] == 0
                              and (r["layout"] if view == "original" else r["parent_layout"]) == parent)
        pref_id.append([i for i,r in enumerate(validation_records["id"]) if r["pointer"] == alias][:2])
    if prepared["pref_indices"] != {"train":pref_train,"id":pref_id}:
        raise ValueError("prepared source-ID preflight selection differs from the deterministic saved subset")
    return prepared


def source_populations(home, prepared):
    from experiments.geoprr_v2_high_resolution_direct import ScalarRGBDataset
    root = home / PREPARED_RELATIVE
    training = ScalarRGBDataset(root / prepared["manifests"]["train"], data_root=home / "datasets", augment=True)
    source_id = ScalarRGBDataset(root / prepared["manifests"]["id"], data_root=home / "datasets", augment=False)
    return training, source_id


def learning_gate(rows):
    from experiments.geoprr_v2_high_resolution_direct import summarize
    if len(rows) != 292 or len({row["sample_id"] for row in rows}) != 292:
        raise ValueError("the learning gate requires every one of the same292 source-ID rows")
    baseline = []
    for row in rows:
        target = row["target"]
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("the existing source scalar target is invalid")
        baseline.append({**row, "prediction": .5, "valid": True, "input_valid": True,
                         "absolute_error": abs(.5-target), "failure": ""})
    model_summary, constant_summary = summarize(rows), summarize(baseline)
    passed = (model_summary["nmae"] <= .5 * constant_summary["nmae"]
              and model_summary["tail10_nmae"] <= .5 * constant_summary["tail10_nmae"])
    return {"passed": bool(passed), "denominator":292, "tail_count":30, "constant_prediction":.5,
            "ratio":.5, "model":model_summary, "constant":constant_summary,
            "criteria": "model mean and tail10 each <= one half of constant0.5 on the same saved labels"}, baseline


def conditional_ood(gate, evaluate):
    return evaluate() if gate["passed"] else None


def validate_totals(totals, gate_passed):
    expected = {"train_batches":430, "train_cnn_image_forwards":27480, "train_cnn_forward_calls":430,
                "backward_calls":430, "train_point_forward_calls":430, "train_point_image_forwards":27480,
                "validation_images":512 if gate_passed else 292,
                "validation_cnn_forward_calls":9 if gate_passed else 5,
                "validation_point_forward_calls":9 if gate_passed else 5,
                "validation_point_image_forwards":512 if gate_passed else 292}
    if (any(totals[key] != value for key, value in expected.items())
            or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 430):
        raise ValueError("the fixed five-epoch control exposure differs from its frozen budget")


def preflight(home, output):
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_data import _resolve_image, crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import (
        create_model, encoder_counter, gradient_evidence, model_metadata, point_counter, runtime,
    )
    from experiments.train_geoprr_v2 import device_batch, seed_everything

    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("the needle control preflight already has complete or partial outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "started.json", {"protocol":PROTOCOL, "status":"started"})
    prepared = validate_prepared(home)
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    training, source_id = source_populations(home, prepared)
    indices = prepared["pref_indices"]
    train_batch = device_batch(default_collate([training[i] for i in indices["train"]]), device)
    validation_batches, identity_ids = [], []
    for subset in indices["id"]:
        samples = [source_id[i] for i in subset]
        for i, sample in zip(subset, samples, strict=True):
            row = source_id.rows[i]
            image = cv2.imread(str(_resolve_image(row, source_id.base.manifest_root, source_id.base.dataset_root)))
            roi, _ = crop_and_resize(image, None if "roi_path" in row else row.get("metadata", {}).get("dial_bbox"), 512)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            expected = (expected-expected.new_tensor([.485,.456,.406])[:,None,None])/expected.new_tensor([.229,.224,.225])[:,None,None]
            if sample["input_failure"] or sample["image"].shape != (3,512,512) or not torch.equal(sample["image"], expected):
                raise ValueError("source-ID preflight differs from the existing V26 input pipeline")
        validation_batches.append(device_batch(default_collate(samples), device))
        identity_ids.append([sample["sample_id"] for sample in samples])
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    original = {key:value.detach().clone() for key,value in model.state_dict().items()}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key:value.cpu() for key,value in original.items()}, strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key,value in original.items())
    del reloaded
    counters = {key:value.clone() for key,value in model.state_dict().items() if key.endswith("num_batches_tracked")}
    if (len(counters), metadata["parameters"], metadata["parameter_tensors"], metadata["state_tensors"]) != (77,8435793,233,464):
        raise ValueError("the unchanged complete V26 reader differs from the frozen architecture")
    cnn_counts, cnn_hook = encoder_counter(model)
    point_counts, point_hook = point_counter(model)
    backwards = []
    backward_hook = model.encoder.conv_stem.weight.register_hook(lambda gradient: backwards.append(True))
    observed_inputs, observed_pool, observed_point = [], [], []
    def input_shape(_module, inputs):
        observed_inputs.append({"shape":list(inputs[0].shape), "dtype":str(inputs[0].dtype)})
    def pool_shape(_module, inputs, result):
        observed_pool.append({"before":list(inputs[0].shape), "after":list(result.shape)})
    def point_shape(_module, inputs, result):
        observed_point.append({"semantic_shape":list(inputs[0].shape), "point_logit_shape":list(result.shape),
                               "semantic_dtype":str(inputs[0].dtype), "point_logit_dtype":str(result.dtype)})
    input_hook = model.encoder.register_forward_pre_hook(input_shape)
    pool_hook = model.encoder.global_pool.register_forward_hook(pool_shape)
    shape_hook = model.point.register_forward_hook(point_shape)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        if prediction.shape != (4,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()):
            raise ValueError("source preflight must return four finite float32 scalars")
        (loss * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero = gradient_evidence(model)
        if not all(torch.equal(model.state_dict()[key], value+1) for key,value in counters.items()):
            raise ValueError("all77 source BN counters must advance exactly once in preflight")
        model.eval()
        before_eval = {key:value.clone() for key,value in model.named_buffers()}
        with torch.inference_mode():
            for batch in validation_batches:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                if (prediction.shape != (2,) or prediction.dtype != torch.float32
                        or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all())):
                    raise ValueError("source-ID preflight scalars differ from the frozen scalar contract")
        buffers_unchanged = all(torch.equal(value, before_eval[key]) for key,value in model.named_buffers())
        parameters_unchanged = all(torch.equal(value, original[key]) for key,value in model.named_parameters())
    finally:
        for hook in (cnn_hook, point_hook, backward_hook, input_hook, pool_hook, shape_hook):
            hook.remove()
    expected_input = [{"shape":[n,3,512,512], "dtype":"torch.float32"} for n in (4,2,2)]
    expected_pool = [{"before":[n,960,16,16], "after":[n,960,1,1]} for n in (4,2,2)]
    expected_point = [{"semantic_shape":[n,1280], "point_logit_shape":[n,1],
                       "semantic_dtype":"torch.float16", "point_logit_dtype":"torch.float16"} for n in (4,2,2)]
    if (cnn_counts != {"cnn_forward_calls":3,"cnn_image_forwards":8}
            or point_counts != {"point_forward_calls":3,"point_image_forwards":8}
            or len(backwards) != 1 or observed_inputs != expected_input or observed_pool != expected_pool
            or observed_point != expected_point or not strict_reload or not finite or not all(nonzero.values())
            or not parameters_unchanged or not buffers_unchanged or not bool(torch.isfinite(loss))):
        raise ValueError("the source-ID-only zero-update preflight failed")
    write_json(output, {"protocol":PROTOCOL, "config":CONFIG, "source_contract":prepared["source_contract"],
        "status":"passed", "model_metadata":metadata, "initialization":model.initialization_evidence,
        "probe_sample_ids":train_batch["sample_id"], "source_ID_identity_sample_ids":identity_ids,
        "source_ID_pipeline_identity":True, "observed_encoder_inputs":observed_inputs,
        "observed_pool_shapes":observed_pool, "observed_point_shapes_and_dtypes":observed_point,
        "source_BN_counter_tensors_checked":77, "source_BN_counter_increment_training":1,
        "source_BN_counter_additional_increment_eval":0, "eval_model_buffers_unchanged":buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":233, "strict_model_reload":strict_reload,
        "full_state_tensors":464, "all_gradients_finite":finite, "gradient_groups_nonzero":nonzero,
        "source_smooth_l1":float(loss.detach()), "preflight_backward_loss_scale":128.,
        "point_counts":point_counts, "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":1,
            "combined_backward_calls":1,"optimizer_updates":0},
        "source_OOD_RGB_reads":0, "source_OOD_forward_calls":0, "external_images_calls_backward_updates":0,
        "formal_fresh_rebuild_required":True, "runtime":runtime(device,started)})


def _save_evaluation(output, name, rows, cost):
    from experiments.geoprr_v2_high_resolution_direct import save_csv, summarize
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    save_csv(output / f"{name}_predictions.csv", rows)
    group_rows = [{"group_id":group, **summarize(values)} for group,values in groups.items()]
    save_csv(output / f"{name}_groups.csv", group_rows)
    report = {"summary":summarize(rows), "by_group":{group:summarize(values) for group,values in groups.items()},
              "cost":cost, "failures":[row for row in rows if not row["valid"]]}
    write_json(output / f"{name}_summary.json", report)
    return report


def control(home, output):
    import torch
    from torch.nn import functional as F
    from experiments.geoprr_v2_high_resolution_direct import (
        ScalarRGBDataset, create_model, encoder_counter, evaluate_population, model_metadata,
        point_counter, runtime, synchronize,
    )
    from experiments.train_geoprr_v2 import device_batch, make_loader, rng_state, save_checkpoint, seed_everything

    if output.exists():
        raise FileExistsError("the fixed control cannot resume or overwrite complete or partial outputs")
    prepared = validate_prepared(home)
    witness = json.loads((home / PREFLIGHT_RELATIVE).read_text())
    if (witness["status"] != "passed" or witness["protocol"] != PROTOCOL or witness["config"] != CONFIG
            or witness["source_contract"] != prepared["source_contract"]):
        raise ValueError("the fresh control requires its completed source-ID-only preflight")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "started.json", {"protocol":PROTOCOL, "status":"started"})
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    training, source_id = source_populations(home, prepared)
    model = create_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    metadata, initialization = model_metadata(model), model.initialization_evidence
    totals = {key:0 for key in ("train_batches", "train_cnn_image_forwards", "train_cnn_forward_calls", "backward_calls",
        "optimizer_updates", "amp_skipped_updates", "validation_images", "validation_cnn_forward_calls",
        "train_point_forward_calls", "train_point_image_forwards", "validation_point_forward_calls", "validation_point_image_forwards")}
    totals.update(training_seconds=0., validation_seconds=0.)
    history = []
    write_json(output / "run.json", {"protocol":PROTOCOL, "config":CONFIG, "source_contract":prepared["source_contract"],
        "initialization":initialization, "model_metadata":metadata, "fresh_default_GradScaler":True,
        "preflight_state_reused":False, "source_unique_images":5496, "source_presentations_per_epoch":5496})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, 6):
        tick = time.perf_counter()
        model.train()
        loader = make_loader(training, batch_size=64, workers=4, shuffle=True, seed=20262020+epoch, device=device)
        counts, hook = encoder_counter(model)
        points, point_hook = point_counter(model)
        backwards = []
        backward_hook = model.encoder.conv_stem.weight.register_hook(lambda gradient: backwards.append(True))
        batches = updates = skips = 0
        last_batch = None
        loss_sum = error_sum = 0.
        try:
            for raw in loader:
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction, batch["y"], beta=.05)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite source scalar loss at epoch{epoch} batch{batches+1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                last_batch = len(batch["y"])
                batches += 1
                updates += int(not skipped)
                skips += int(skipped)
                loss_sum += float(loss.detach()) * last_batch
                error_sum += float((prediction.detach()-batch["y"]).abs().sum())
        finally:
            hook.remove()
            point_hook.remove()
            backward_hook.remove()
        synchronize(device)
        seconds = time.perf_counter()-tick
        if ((batches, last_batch, len(backwards)) != (86,56,86)
                or counts != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or points != {"point_forward_calls":86,"point_image_forwards":5496}
                or updates+skips != 86 or optimizer.param_groups[0]["lr"] != .0003):
            raise RuntimeError("a complete epoch differs from the fixed5496 image/86 batch budget")
        increments = {"train_batches":batches, "train_cnn_image_forwards":5496, "train_cnn_forward_calls":86,
            "backward_calls":len(backwards), "optimizer_updates":updates, "amp_skipped_updates":skips,
            "train_point_forward_calls":86, "train_point_image_forwards":5496, "training_seconds":seconds}
        for key,value in increments.items():
            totals[key] += value
        record = {"epoch":epoch, "learning_rate":.0003, "source_training_smooth_l1":loss_sum/5496,
            "source_training_nmae":error_sum/5496, "train_counts":counts, "train_point_counts":points,
            "backward_calls":len(backwards), "optimizer_updates":updates, "amp_skipped_updates":skips,
            "last_batch_size":last_batch, "training_seconds":seconds, "validation_calls":0}
        history.append(record)
        checkpoint = {"protocol":PROTOCOL, "config":CONFIG, "source_contract":prepared["source_contract"],
            "model":model.state_dict(), "optimizer":optimizer.state_dict(), "scaler":scaler.state_dict(),
            "rng":rng_state(), "epoch":epoch, "history":history, "totals":dict(totals),
            "initialization":initialization, "model_metadata":metadata,
            "checkpointed_process_seconds":time.perf_counter()-started}
        save_checkpoint(output / "last.pt", checkpoint)
        (output / "history.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history))
        print(json.dumps({"event":"epoch_complete", "epoch":epoch, "train_images":5496,
                          "CNN_calls":86, "backward_attempts":86, "validation_calls":0}), flush=True)

    def evaluate(name, dataset, images, calls):
        points, hook = point_counter(model)
        try:
            rows, cost = evaluate_population(model, dataset, device)
        finally:
            hook.remove()
        if (len(rows) != images or cost["counts"]["cnn_image_forwards"] != images
                or cost["counts"]["cnn_forward_calls"] != calls
                or points != {"point_forward_calls":calls,"point_image_forwards":images}):
            raise RuntimeError("source scalar evaluation differs from its frozen complete denominator")
        cost["point_counts"] = points
        for key,value in {"validation_images":images, "validation_cnn_forward_calls":calls,
            "validation_point_forward_calls":calls, "validation_point_image_forwards":images,
            "validation_seconds":cost["pipeline_seconds"]}.items():
            totals[key] += value
        return rows, _save_evaluation(output, name, rows, cost)

    id_rows, id_report = evaluate("source_id", source_id, 292, 5)
    gate, baseline = learning_gate(id_rows)
    _save_evaluation(output, "constant_source_id", baseline, {"CNN_calls":0,"CNN_images":0})
    write_json(output / "learning_gate.json", gate)
    def evaluate_ood():
        dataset = ScalarRGBDataset(home / PREPARED_RELATIVE / prepared["manifests"]["ood"],
                                   data_root=home / "datasets", augment=False)
        return evaluate("source_ood", dataset, 220, 4)[1]
    ood_report = conditional_ood(gate, evaluate_ood)
    validate_totals(totals, gate["passed"])
    checkpoint.update(totals=dict(totals), rng=rng_state(), source_id_rows=id_rows, learning_gate=gate,
                      source_ood=ood_report, checkpointed_process_seconds=time.perf_counter()-started)
    save_checkpoint(output / "last.pt", checkpoint)
    reason = "source_ood_observed" if gate["passed"] else "learning_gate_failed"
    write_json(output / "completed.json", {"protocol":PROTOCOL, "config":CONFIG,
        "source_contract":prepared["source_contract"], "status":"complete", "epochs":5,
        "completion_reason":reason, "learning_gate":gate, "source_id":id_report, "source_ood":ood_report,
        "totals":totals, "model_metadata":metadata, "initialization":initialization,
        "checkpoint":"last.pt", "source_OOD_RGB_images":220 if gate["passed"] else 0,
        "source_OOD_forward_calls":4 if gate["passed"] else 0,
        "external_counts":{"images":0,"CNN_calls":0,"point_calls":0,"backward_calls":0,"optimizer_updates":0},
        "runtime":runtime(device,started)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "control"))
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        if args.output is not None:
            parser.error("prepare has one frozen independent destination")
        prepare(args.home)
    elif args.stage == "preflight":
        preflight(args.home, args.output or args.home / PREFLIGHT_RELATIVE)
    else:
        control(args.home, args.output or args.home / OUTPUT_RELATIVE / "reader")


if __name__ == "__main__":
    main()
