"""Evaluate a frozen GeoPRR V2 checkpoint with image-only model inputs."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from experiments.train_geoprr_v2 import (
    DEFAULT_DATA_ROOT,
    PROTOCOL,
    device_batch,
    make_loader,
    prediction_rows,
    summarize_rows,
    write_json,
)


class FullPopulationDataset(Dataset):
    """Preserve declared rows when image loading or ROI preparation fails."""

    def __init__(self, dataset: Any) -> None:
        from experiments.geoprr_v2_data import normalized_target

        self.dataset = dataset
        self.targets = [normalized_target(row) for row in dataset.rows]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset.rows[index]
        valid, failure = True, ""
        try:
            sample = self.dataset[index]
        except (OSError, ValueError, KeyError) as exc:
            valid, failure = False, f"{type(exc).__name__}: {exc}"
            sample = {"image": torch.zeros(3, self.dataset.image_size, self.dataset.image_size),
                      "y": torch.tensor(self.targets[index], dtype=torch.float32),
                      "sample_id": str(row["sample_id"]), "group_id": str(row.get("group_id", row["sample_id"]))}
        sample["input_valid"] = valid
        sample["input_failure"] = failure
        sample["dataset"] = str(row.get("dataset", "unspecified"))
        sample["condition"] = str(row.get("condition", row.get("metadata", {}).get("condition", "native")))
        return {key: sample[key] for key in ("image", "y", "sample_id", "group_id", "input_valid", "input_failure", "dataset", "condition")}


def write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("sample_id", "group_id", "dataset", "condition", "target", "prediction", "absolute_error",
              "valid", "field_valid", "support_mass", "failure")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(*, checkpoint_path: Path, manifest_path: Path, dataset_root: Path, output_dir: Path,
             device_name: str, batch_size: int, workers: int, use_amp: bool, max_batches: int = 0) -> dict[str, Any]:
    from experiments.geoprr_v2_data import GeoPRRV2Dataset
    from experiments.geoprr_v2_model import GeoPRRV2

    if batch_size < 1 or workers < 0 or max_batches < 0:
        raise ValueError("batch size must be positive; workers and smoke batch limit must be nonnegative")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint protocol differs from GeoPRR V2 training")
    config = checkpoint["config"]
    model_options = dict(config["model"], pretrained=False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = GeoPRRV2(**model_options).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = FullPopulationDataset(GeoPRRV2Dataset(manifest_path, dataset_root=dataset_root,
                                                   **config["data"], require_geometry=False, augment=False))
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False, seed=config["training"]["seed"], device=device)
    rows, forward_seconds = [], 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = device_batch(raw_batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                output = model(batch["image"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - forward_start
            batch_rows = prediction_rows(output, batch)
            for index, row in enumerate(batch_rows):
                row.update(dataset=raw_batch["dataset"][index], condition=raw_batch["condition"][index])
                if not bool(raw_batch["input_valid"][index]):
                    row.update(prediction=None, valid=False, absolute_error=1.0, field_valid=False,
                               support_mass=None, failure=raw_batch["input_failure"][index])
            rows.extend(batch_rows)
    elapsed = time.perf_counter() - started
    if not max_batches and len(rows) != len(dataset):
        raise RuntimeError("evaluation lost declared samples")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
        by_condition[f"{row['dataset']}::{row['condition']}"].append(row)
        by_group[f"{row['dataset']}::{row['group_id']}"].append(row)
    summary = summarize_rows(rows)
    result = {
        "protocol": "geoprr_v2_frozen_evaluation_v1", "training_protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "manifest": str(manifest_path.resolve()), "declared_samples": len(dataset),
        "smoke_only": bool(max_batches or checkpoint.get("smoke_only", False)
                           or any(row.get("diagnostic_only", False) for row in dataset.dataset.rows)),
        "model": model_options, "training_config": config,
        "training_or_adaptation_during_evaluation": False,
        "model_inputs": "ImageNet-normalized ROI pixels only", "summary": summary,
        "by_dataset": {name: summarize_rows(population) for name, population in by_dataset.items()},
        "by_condition": {name: summarize_rows(population) for name, population in by_condition.items()},
        "by_group": {name: summarize_rows(population) for name, population in by_group.items()},
        "timing": {"batch_size": batch_size, "workers": workers, "amp": use_amp and device.type == "cuda",
                   "pipeline_seconds": elapsed, "pipeline_ms_per_image": 1000 * elapsed / len(rows),
                   "forward_seconds": forward_seconds, "forward_ms_per_image": 1000 * forward_seconds / len(rows),
                   "pipeline_boundary": "image read, ROI crop/resize, normalization, transfer, forward and prediction collection; batched throughput",
                   "warmup_excluded": False},
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    write_predictions(output_dir / "predictions.csv", rows)
    write_json(output_dir / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="Positive limits are smoke checks, not full evaluations.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(checkpoint_path=args.checkpoint, manifest_path=args.manifest, dataset_root=args.data_root,
                      output_dir=args.output, device_name=args.device, batch_size=args.batch_size, workers=args.workers,
                      use_amp=not args.no_amp, max_batches=args.max_batches)
    print(json.dumps({"output": str(args.output.resolve()), **result["summary"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
