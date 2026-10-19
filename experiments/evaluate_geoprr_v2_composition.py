"""Diagnose held-out state-composition edges between source-group images."""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import Dataset

from experiments.geoprr_v2_model import masked_field_readout
from experiments.train_geoprr_v2 import (
    DEFAULT_DATA_ROOT, PROTOCOL, device_batch, make_loader, summarize_rows, write_json,
)


SUPPORT_THRESHOLD = 0.15


class FullCompositionPopulation(Dataset):
    """Keep every declared edge when either image cannot be prepared."""

    def __init__(self, dataset: Any) -> None:
        from experiments.geoprr_v2_data import normalized_target

        self.dataset = dataset
        self.indices = [index for index, (_row, edge) in enumerate(dataset.items) if edge is not None]
        self.edges = [dataset.items[index][1] for index in self.indices]
        if not self.edges or any(edge["kind"] != "state" for edge in self.edges):
            raise ValueError("composition evaluation requires declared state-intervention edges")
        self.targets = [normalized_target(dataset.rows[dataset.by_id[edge["sample_b"]]]) for edge in self.edges]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        edge = self.edges[index]
        valid, failure = True, ""
        try:
            sample = self.dataset[self.indices[index]]
            image, partner = sample["image"], sample["pair_image"]
        except (OSError, ValueError, KeyError) as exc:
            valid, failure = False, f"{type(exc).__name__}: {exc}"
            image = torch.zeros(3, self.dataset.image_size, self.dataset.image_size)
            partner = torch.zeros_like(image)
        return {"image": image, "pair_image": partner, "target": torch.tensor(self.targets[index], dtype=torch.float32),
                "donor_id": str(edge["sample_a"]), "recipient_id": str(edge["sample_b"]),
                "layout_id": str(edge["layout_id"]), "input_valid": valid, "input_failure": failure}


def compose_predictions(donor: Mapping[str, torch.Tensor], recipient: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return masked_field_readout(donor["field_probs"], recipient["point_probs"],
                                donor["support_probs"] * recipient["support_probs"], SUPPORT_THRESHOLD)


def composition_rows(output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    predictions = output["reading"].detach().float().reshape(-1).cpu().tolist()
    masses = output["support_mass"].detach().float().reshape(-1).cpu().tolist()
    field_valid = output["field_valid"].detach().bool().reshape(-1).cpu().tolist()
    targets = batch["target"].detach().float().reshape(-1).cpu().tolist()
    rows = []
    for index, (prediction, mass, target) in enumerate(zip(predictions, masses, targets, strict=True)):
        failure = ""
        if not bool(batch["input_valid"][index]):
            failure = batch["input_failure"][index]
        elif not math.isfinite(mass) or not math.isfinite(prediction):
            failure = "nonfinite_composition"
        elif mass < SUPPORT_THRESHOLD or not field_valid[index]:
            failure = "insufficient_predicted_support"
        elif not 0 <= prediction <= 1:
            failure = "out_of_range_composition"
        valid = not failure
        donor, recipient, layout = batch["donor_id"][index], batch["recipient_id"][index], batch["layout_id"][index]
        rows.append({"sample_id": f"{donor}->{recipient}", "group_id": layout,
                     "donor_id": donor, "recipient_id": recipient, "layout_id": layout, "target": target,
                     "prediction": prediction if valid else None,
                     "support_mass": mass if bool(batch["input_valid"][index]) and math.isfinite(mass) else None,
                     "valid": valid, "field_valid": valid, "absolute_error": abs(prediction - target) if valid else 1.0,
                     "failure": failure})
    return rows


def audit_heldout_edges(checkpoint: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = checkpoint["manifests"]["pairs"]
    if source is None:
        raise ValueError("held-out composition requires the checkpoint's actual training pair manifest")
    path = Path(source)
    training_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    training_edges = {(str(row["sample_a"]), str(row["sample_b"])) for row in training_records}
    heldout_edges = {(str(row["sample_a"]), str(row["sample_b"])) for row in edges}
    overlap = training_edges & heldout_edges
    if overlap:
        raise ValueError(f"held-out composition contains {len(overlap)} training edges")
    image_ids = {str(edge[key]) for edge in edges for key in ("sample_a", "sample_b")}
    training_ids = set(checkpoint["run_metadata"]["sample_ids"]["train"])
    seen = image_ids & training_ids
    if seen != image_ids:
        raise ValueError("held-out composition requires images already present in checkpoint training")
    return {"training_pair_manifest": str(path.resolve()), "training_edges": len(training_edges),
            "heldout_edges": len(heldout_edges), "overlap_edges": 0, "heldout_unique_images": len(image_ids),
            "images_seen_in_training": len(seen), "unseen_images": 0,
            "scope": "unseen directed composition edges between seen source images; not unseen-image evaluation"}


def evaluate(*, checkpoint_path: Path, manifest_path: Path, pair_manifest_path: Path, dataset_root: Path,
             output_dir: Path, device_name: str = "cuda", batch_size: int = 16, workers: int = 4,
             use_amp: bool = True) -> dict[str, Any]:
    from experiments.geoprr_v2_data import GeoPRRV2Dataset
    from experiments.geoprr_v2_model import GeoPRRV2

    if batch_size < 1 or workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError("checkpoint protocol differs from GeoPRR V2 training")
    config = checkpoint["config"]
    if config["model"]["variant"] != "field":
        raise ValueError("composition evaluation requires the field reader")
    if config["training"].get("pair_objective") != "state_composition_v1":
        raise ValueError("this diagnostic requires a state_composition_v1 checkpoint")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_options = dict(config["model"], pretrained=False)
    source = GeoPRRV2Dataset(manifest_path, pair_manifest=pair_manifest_path, dataset_root=dataset_root,
                            **config["data"], require_geometry=False, augment=False)
    dataset = FullCompositionPopulation(source)
    edge_split = audit_heldout_edges(checkpoint, dataset.edges)
    model = GeoPRRV2(**model_options).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False,
                         seed=config["training"]["seed"], device=device)
    rows, forward_seconds, composition_seconds = [], 0.0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = device_batch(raw_batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                donor = model(batch["image"])
                recipient = model(batch["pair_image"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - step_started
            step_started = time.perf_counter()
            output = compose_predictions(donor, recipient)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            composition_seconds += time.perf_counter() - step_started
            rows.extend(composition_rows(output, raw_batch))
    elapsed = time.perf_counter() - started
    if len(rows) != len(dataset):
        raise RuntimeError("composition evaluation lost declared edges")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["layout_id"]].append(row)
    image_ids = {str(edge[key]) for edge in dataset.edges for key in ("sample_a", "sample_b")}
    reader_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "protocol": "geoprr_v2_heldout_state_composition_evaluation_v1", "diagnostic_only": True,
        "role": "heldout_source_image_composition", "training_protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "manifest": str(manifest_path.resolve()), "pair_manifest": str(pair_manifest_path.resolve()),
        "declared_edges": len(dataset), "declared_manifest_images": len(source.rows), "unique_edge_images": len(image_ids),
        "images_in_checkpoint_training": edge_split["images_seen_in_training"], "edge_split": edge_split,
        "model": model_options, "training_config": config,
        "smoke_only": bool(checkpoint.get("smoke_only", False) or any(row.get("diagnostic_only", False) for row in source.rows)),
        "training_or_adaptation_during_evaluation": False, "support_threshold": SUPPORT_THRESHOLD,
        "readout": "full-distribution native expectation; unsupported edges fail without fallback",
        "model_inputs": "two ImageNet-normalized ROI pixel tensors; no U/Q/M labels or GT support overlap in composition",
        "qualification": "state-pair factor provenance is checked by the loader; fixed-camera geometry qualification is established during data preparation",
        "scope": "declared held-out donor-recipient edges, not single-image accuracy or unseen-image transfer",
        "summary": summarize_rows(rows), "by_group": {group: summarize_rows(population) for group, population in by_group.items()},
        "reader_parameters": reader_parameters, "parameters": reader_parameters,
        "parameter_boundary": "one shared reader evaluated twice per edge; no second reader or router parameters",
        "timing": {"batch_size": batch_size, "workers": workers, "amp": use_amp and device.type == "cuda",
                   "pipeline_seconds": elapsed, "pipeline_ms_per_edge": 1000 * elapsed / len(rows),
                   "reader_forward_seconds": forward_seconds, "reader_forward_ms_per_edge": 1000 * forward_seconds / len(rows),
                   "composition_seconds": composition_seconds, "composition_ms_per_edge": 1000 * composition_seconds / len(rows),
                   "reader_image_forwards": 2 * len(rows), "warmup_excluded": False,
                   "pipeline_boundary": "two-image loading, annotation-defined ROI crop/resize, normalization, transfer, two shared-reader forwards, predicted-field composition and edge scoring; setup and export excluded"},
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(checkpoint_path=args.checkpoint, manifest_path=args.manifest, pair_manifest_path=args.pair_manifest,
                      dataset_root=args.data_root, output_dir=args.output, device_name=args.device,
                      batch_size=args.batch_size, workers=args.workers, use_amp=not args.no_amp)
    print(json.dumps({"output": str(args.output.resolve()), **result["summary"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
