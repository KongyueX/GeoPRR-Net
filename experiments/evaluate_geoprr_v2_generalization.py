"""Evaluate native frozen GeoPRR V2 readers on complete Industrial or RF100 ROIs."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path

import torch

from experiments.evaluate_geoprr_v2_role_external import (
    ExternalPopulationDataset, POPULATIONS, load_population, summarize, write_csv,
)
from experiments.geoprr_v2_model import GeoPRRV2
from experiments.geoprr_v2_reference_pair_reader import ReferencePairReader
from experiments.geoprr_v2_reference_relation import ReferenceRelationReader
from experiments.geoprr_v2_role_triangle import RoleTriangleReader
from experiments.train_geoprr_v2 import device_batch, make_loader, prediction_rows
from experiments.train_geoprr_v2_reference_relation import validation_rows


PROTOCOL = "geoprr_v2_native_roi_frozen_generalization_v1"
BASE_PROTOCOL = "geoprr_v2_source_only_v1"
REFERENCE_PROTOCOLS = (
    "geoprr_v2_reference_relation_source_only_v1",
    "geoprr_v2_reference_feedback_source_only_v1",
)
TRIANGLE_PROTOCOL = "geoprr_v2_role_triangle_source_only_v1"
PAIR_PROTOCOL = "geoprr_v2_reference_pair_update_source_only_v1"
GENERALIZATION_PROTOCOL = "geoprr_v2_generalization_source_only_v1"
ROI_GENERALIZATION_PROTOCOL = "geoprr_v2_roi_generalization_source_only_v1"
ENDPOINT_GENERALIZATION_PROTOCOL = "geoprr_v2_endpoint_generalization_source_only_v1"


def reader_from_checkpoint(checkpoint):
    """Use the saved architecture and native readout without loading initializers."""
    config, protocol = checkpoint["config"], checkpoint["protocol"]
    if config.get("protocol", protocol) != protocol:
        raise ValueError("checkpoint and saved config protocols differ")
    generalization = protocol in (GENERALIZATION_PROTOCOL, ROI_GENERALIZATION_PROTOCOL, ENDPOINT_GENERALIZATION_PROTOCOL)
    architecture_protocol = config.get("reader_protocol") if generalization else protocol
    if generalization and architecture_protocol not in (*REFERENCE_PROTOCOLS, TRIANGLE_PROTOCOL):
        raise ValueError("generalization checkpoint must declare its original angular reader protocol")
    if checkpoint.get("smoke_only", False):
        raise ValueError("smoke-only checkpoints are not full-population candidates")
    if config["data"] != {"image_size": 768, "output_size": 192}:
        raise ValueError("native ROI evaluation requires the original768/192 configuration")
    options = dict(config["model"])
    if architecture_protocol == BASE_PROTOCOL:
        if options.get("variant") != "field":
            raise ValueError("the base source protocol requires a native field checkpoint")
        options["pretrained"] = False
        model = GeoPRRV2(**options)
        readout = "native masked U/Q/M expectation with the checkpoint's appearance fallback"
    elif architecture_protocol in REFERENCE_PROTOCOLS:
        if architecture_protocol == REFERENCE_PROTOCOLS[1] and "reference_normalization" not in options:
            raise ValueError("feedback checkpoints must declare reference normalization")
        model = ReferenceRelationReader(**options)
        readout = "native circular full-mass progress expectation"
    elif architecture_protocol == TRIANGLE_PROTOCOL:
        model = RoleTriangleReader(**options)
        readout = "native three-role circular full-mass progress expectation"
    elif architecture_protocol == PAIR_PROTOCOL:
        model = ReferencePairReader(**options)
        readout = "native reference-update circular full-mass progress expectation"
    else:
        raise ValueError(f"unsupported training protocol: {protocol}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "architecture_protocol": architecture_protocol,
        "saved_model_options": config["model"], "constructor_options": options,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensors": len(model.state_dict()), "strict_state_load": True,
        "readout": readout,
    }
    if protocol != BASE_PROTOCOL:
        metadata["effective_reference_normalization"] = model.state_head.relation.reference_normalization
    return model, metadata


def native_prediction_rows(output, batch, training_protocol):
    if training_protocol != BASE_PROTOCOL:
        return validation_rows(output, batch)
    rows = prediction_rows(output, batch)
    for index, row in enumerate(rows):
        row.update(input_valid=not batch["input_failure"][index], invalid_mass=None, outside_arc_mass=None)
        if batch["input_failure"][index]:
            row.update(prediction=None, valid=False, absolute_error=1.0, field_valid=False,
                       support_mass=None, failure=batch["input_failure"][index])
    return rows


def evaluate(*, dataset_root, dataset_name, checkpoint_path, output_dir, device_name, batch_size, workers):
    evaluation_started = time.perf_counter()
    if batch_size < 1 or workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output_dir}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    population = load_population(dataset_root.resolve(), dataset_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, model_metadata = reader_from_checkpoint(checkpoint)
    model.to(device)
    config, training_protocol = checkpoint["config"], checkpoint["protocol"]
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
        torch.cuda.synchronize(device)
    rows, forward_seconds = [], 0.0
    pipeline_started = time.perf_counter()
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
                rows.extend(native_prediction_rows(output, batch, training_protocol))
    finally:
        hook.remove()
    pipeline_seconds = time.perf_counter() - pipeline_started
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in population]:
        raise RuntimeError("evaluation lost or reordered declared samples")
    if counts["cnn_image_forwards"] != len(population) or counts["cnn_forward_calls"] != counts["model_forward_calls"]:
        raise RuntimeError("actual CNN exposure differs from one forward per declared input")
    by_group, by_component = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native", component=source["component"])
        by_group[row["group_id"]].append(row)
        by_component[row["component"]].append(row)
    result = {
        "protocol": PROTOCOL, "status": "complete", "training_protocol": training_protocol,
        "training_config": config, "model_metadata": model_metadata,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_best_epoch": checkpoint.get("best_epoch"), "checkpoint_seed": config["training"]["seed"],
        "checkpoint_source_best_nmae": checkpoint["best_nmae"],
        "checkpoint_initialization": checkpoint.get("initialization"),
        "source_data_contract_retained_in_checkpoint": "data_contract" in checkpoint,
        "checkpoint_selection": "explicit frozen checkpoint path; no checkpoint selection within evaluation",
        "dataset": dataset_name, "dataset_root": str(dataset_root.resolve()),
        "input_manifest": str((dataset_root / "input_manifest.jsonl").resolve()),
        "scoring_labels": str((dataset_root / "labels.jsonl").resolve()),
        "component_index": str((dataset_root / "industrial_roi_index.jsonl").resolve()) if dataset_name == "industrial" else None,
        "declared_samples": len(population), "declared_groups": POPULATIONS[dataset_name][1],
        "model_inputs": "original ROI RGB only; OpenCV linear resize768; ImageNet normalization; no geometry input",
        "readout": model_metadata["readout"],
        "additional_postprocessing": "none",
        "target_convention": "normalized_target converted to float32 before original native validation scoring",
        "target_source": ("provided physical/normalized reading labels" if dataset_name == "industrial"
                          else "offline target derived from four annotated points; not an official scalar-reading benchmark"),
        "development_scope": "external results have been viewed; fixed native ROI regression evaluation, not an untouched confirmatory test",
        "historical_external_results_previously_viewed": True,
        "training_or_adaptation_during_evaluation": False, "optimizer_updates": 0, "backward_calls": 0,
        "failure_policy": "retain every declared input; input or invalid prediction failure receives normalized error1",
        "summary": summarize(rows), "by_group": {key: summarize(value) for key, value in by_group.items()},
        "by_component": {key: summarize(value) for key, value in by_component.items()},
        "forward_counts": counts,
        "timing": {"batch_size": batch_size, "workers": workers, "backbone_amp": amp_enabled,
                   "readout_precision": "float32, inherited from each native reader",
                   "pipeline_seconds": pipeline_seconds,
                   "pipeline_ms_per_declared_image": 1000 * pipeline_seconds / len(rows),
                   "forward_seconds": forward_seconds,
                   "forward_ms_per_declared_image": 1000 * forward_seconds / len(rows),
                   "pipeline_boundary": "ROI read/prepare, transfer, model forward and prediction collection; batched throughput",
                   "warmup_excluded": False},
        "parameters": model_metadata["parameters"], "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", rows)
    write_csv(output_dir / "groups.csv", [{"group_id": key, **value} for key, value in result["by_group"].items()])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["timing"].update(
        evaluation_seconds=time.perf_counter() - evaluation_started,
        full_process_seconds=time.perf_counter() - PROCESS_STARTED,
        full_process_boundary="module entry before torch import through saved predictions/groups; excludes final summary write and console output",
        process_cpu_user_seconds=usage.ru_utime, process_cpu_system_seconds=usage.ru_stime,
        worker_cpu_user_seconds=children.ru_utime, worker_cpu_system_seconds=children.ru_stime,
    )
    result["peak_process_rss_bytes"] = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=tuple(POPULATIONS), required=True)
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
