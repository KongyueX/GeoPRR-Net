"""Evaluate frozen v7 full best64 on the declared Industrial or RF100 ROI population."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from experiments.geoprr_v2_data import GeoPRRV2Dataset, normalized_target, read_rows
from experiments.geoprr_v2_role_triangle import RoleTriangleReader
from experiments.train_geoprr_v2 import device_batch, make_loader
from experiments.train_geoprr_v2_reference_relation import validation_rows, validation_summary


TRAINING_PROTOCOL = "geoprr_v2_role_triangle_source_only_v1"
POPULATIONS = {"industrial": (1395, 52), "rf100": (151, 35)}
INDUSTRIAL_COMPONENTS = {"field_gauge_roi_test_a": 434, "field_gauge_roi_test_b": 814,
                         "field_gauge_external_roi": 147}


def load_population(dataset_root, dataset_name):
    inputs = read_rows(dataset_root / "input_manifest.jsonl")
    labels = {str(row["sample_id"]): row for row in read_rows(dataset_root / "labels.jsonl")}
    if {str(row["sample_id"]) for row in inputs} != set(labels):
        raise ValueError("input manifest and scoring labels have different sample IDs")
    components = {}
    if dataset_name == "industrial":
        index = read_rows(dataset_root / "industrial_roi_index.jsonl")
        if {str(row["sample_id"]) for row in index} != set(labels):
            raise ValueError("Industrial component index and population have different sample IDs")
        components = {str(row["sample_id"]): str(row["component"]) for row in index}
        if dict(Counter(components.values())) != INDUSTRIAL_COMPONENTS:
            raise ValueError("Industrial component counts differ from434/814/147")
    rows = []
    for item in inputs:
        sample_id = str(item["sample_id"])
        label = labels[sample_id]
        row = {"sample_id": sample_id, "group_id": str(label["group_id"]),
               "roi_path": str((dataset_root / item["roi_path"]).resolve()),
               "component": components[sample_id] if components else "rf100"}
        row.update({key: label[key] for key in
                    ("ground_truth", "scale_start", "scale_end", "normalized_progress") if key in label})
        normalized_target(row)
        rows.append(row)
    actual = len(rows), len({row["group_id"] for row in rows})
    if actual != POPULATIONS[dataset_name]:
        raise ValueError(f"{dataset_name} population differs: {actual} != {POPULATIONS[dataset_name]}")
    return rows


class ExternalPopulationDataset(Dataset):
    """Prepare RGB ROIs independently and retain each failed input in scoring."""

    def __init__(self, rows, data_config):
        self.base = GeoPRRV2Dataset(rows, **data_config, require_geometry=False, augment=False)
        self.rows = rows
        self.targets = [normalized_target(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        failure = ""
        try:
            sample = self.base[index]
        except (OSError, ValueError, cv2.error) as error:
            failure = f"{type(error).__name__}: {error}"
            sample = {"image": torch.zeros(3, self.base.image_size, self.base.image_size),
                      "y": torch.tensor(self.targets[index], dtype=torch.float32),
                      "sample_id": row["sample_id"], "group_id": row["group_id"]}
        return {**{key: sample[key] for key in ("image", "y", "sample_id", "group_id")},
                "input_failure": failure}


def load_reader(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["protocol"] != TRAINING_PROTOCOL or config["protocol"] != checkpoint["protocol"]:
        raise ValueError("checkpoint protocol differs from the selected v7 reader")
    if (checkpoint["epoch"] != 64 or checkpoint["best_epoch"] != 64
            or config["training"]["seed"] != 20262020 or config["model"]["association"] != "full"):
        raise ValueError("external evaluation requires the source-selected v7 full best64 seed20262020")
    if config["data"] != {"image_size": 768, "output_size": 192}:
        raise ValueError("selected checkpoint data configuration differs from768/192")
    model = RoleTriangleReader(**config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 7461316:
        raise ValueError("selected v7 reader parameter count differs")
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def summarize(rows):
    result = validation_summary(rows)
    maximum = max(row["absolute_error"] for row in rows)
    result["maximum_error_ids"] = [row["sample_id"] for row in rows if row["absolute_error"] == maximum]
    for key in ("median_error", "p90_error", "p95_error", "tail10_nmae", "maximum_error"):
        result[f"{key}_percent_fs"] = 100 * result[key]
    for threshold in (1, 2, 5):
        result[f"acc_at_{threshold}_percent"] = 100 * result[f"acc_at_{threshold}"]
        result[f"count_at_{threshold}_percent_fs"] = sum(
            row["valid"] and row["absolute_error"] <= threshold / 100 for row in rows)
    return result


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(*, dataset_root, dataset_name, checkpoint_path, output_dir, device_name, batch_size, workers):
    if batch_size < 1 or workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    population = load_population(dataset_root.resolve(), dataset_name)
    model, checkpoint = load_reader(checkpoint_path, device)
    config = checkpoint["config"]
    dataset = ExternalPopulationDataset(population, config["data"])
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False,
                         seed=config["training"]["seed"], device=device)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    counts = {"model_forward_calls": 0, "cnn_forward_calls": 0, "cnn_image_forwards": 0,
              "roi_image_forwards": 0, "failed_input_placeholder_forwards": 0}

    def count_backbone(module, inputs, output):
        counts["cnn_forward_calls"] += 1
        counts["cnn_image_forwards"] += len(inputs[0])

    hook = model.backbone.register_forward_hook(count_backbone)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rows, forward_seconds = [], 0.0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = device_batch(raw_batch, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(batch["image"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - forward_started
                counts["model_forward_calls"] += 1
                failures = sum(bool(value) for value in batch["input_failure"])
                counts["failed_input_placeholder_forwards"] += failures
                counts["roi_image_forwards"] += len(batch["sample_id"]) - failures
                rows.extend(validation_rows(output, batch))
    finally:
        hook.remove()
    elapsed = time.perf_counter() - started
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in population]:
        raise RuntimeError("evaluation lost or reordered declared samples")
    if counts["cnn_image_forwards"] != len(population) or counts["cnn_forward_calls"] != counts["model_forward_calls"]:
        raise RuntimeError("actual CNN exposure differs from one clean forward per declared input")
    by_group, by_component = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native", component=source["component"])
        by_group[row["group_id"]].append(row)
        by_component[row["component"]].append(row)
    summary = summarize(rows)
    result = {
        "protocol": "geoprr_v2_role_external_frozen_evaluation_v1", "status": "complete",
        "training_protocol": checkpoint["protocol"], "training_config": config,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_seed": config["training"]["seed"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_nmae": checkpoint["best_nmae"],
        "checkpoint_selection": "v7 full best64; source full1506 native NMAE, earliest tie; fixed before external inference",
        "dataset": dataset_name, "dataset_root": str(dataset_root.resolve()),
        "input_manifest": str((dataset_root / "input_manifest.jsonl").resolve()),
        "scoring_labels": str((dataset_root / "labels.jsonl").resolve()),
        "component_index": str((dataset_root / "industrial_roi_index.jsonl").resolve()) if dataset_name == "industrial" else None,
        "declared_samples": len(population), "declared_groups": POPULATIONS[dataset_name][1],
        "model_inputs": "original ROI RGB only; OpenCV linear resize768; ImageNet normalization",
        "readout": "native full-mass reading; no additional clipping, thresholds, routing or fitting",
        "target_convention": "normalized_target converted to float32 before original validation_rows scoring",
        "target_source": ("provided physical/normalized reading labels" if dataset_name == "industrial"
                          else "offline target derived from four annotated points; not an official scalar-reading benchmark"),
        "historical_external_results_previously_viewed": True,
        "training_or_adaptation_during_evaluation": False, "optimizer_updates": 0, "backward_calls": 0,
        "summary": summary, "by_group": {key: summarize(value) for key, value in by_group.items()},
        "by_component": {key: summarize(value) for key, value in by_component.items()},
        "forward_counts": counts,
        "timing": {"batch_size": batch_size, "workers": workers, "backbone_amp": amp_enabled,
                   "head_precision": "float32, inherited from the frozen reader",
                   "pipeline_seconds": elapsed, "pipeline_ms_per_declared_image": 1000 * elapsed / len(rows),
                   "forward_seconds": forward_seconds, "forward_ms_per_declared_image": 1000 * forward_seconds / len(rows),
                   "pipeline_boundary": "ROI read/prepare, transfer, model forward and prediction collection; batched throughput",
                   "warmup_excluded": False},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "groups.csv", [{"group_id": key, **value} for key, value in result["by_group"].items()])
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=tuple(POPULATIONS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = evaluate(dataset_root=args.dataset_root, dataset_name=args.dataset_name,
                      checkpoint_path=args.checkpoint, output_dir=args.output, device_name=args.device,
                      batch_size=args.batch_size, workers=args.workers)
    print(json.dumps({"output": str(args.output.resolve()), **result["summary"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
