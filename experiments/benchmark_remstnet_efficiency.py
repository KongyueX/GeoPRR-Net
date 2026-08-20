"""Measure matched CUDA/BF16 efficiency for ReMSTNet-v3 and its endpoints."""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn

from experiments.a15_2_mett import load_moment_exact_anchor_from_direct_checkpoint
from experiments.benchmark_a15_2_fteb_efficiency import (
    DEVICE_NAME,
    FIXED_CONDITION,
    IMAGE_SIZE,
    PHYSICAL_BATCH_SIZE,
    PROFILE_BATCH_SIZES,
    TIMED_ITERATIONS,
    WARMUP_ITERATIONS,
    _validate_final_posterior_moments,
    load_fixed_physical_batch,
)
from experiments.benchmark_a15_2_mett_efficiency import _endpoint_from_anchor
from experiments.evaluate_remstnet_real_domains import _validate_full_model
from experiments.prepare_a13_correction_scene_split import DEFAULT_CORRECTION_TRAIN_MANIFEST
from experiments.remst_block_net import CoordinatedReMSTNet, remst_block_parameter_counts
from experiments.train_remst_block_pilot import load_remst_block_checkpoint


PROTOCOL: Final[str] = "remstnet_v3_matched_cuda_efficiency_v1"
ARMS: Final[tuple[str, ...]] = ("raw_foundation", "twin_endpoint", "full_remstnet")
DISPLAY_NAMES: Final[dict[str, str]] = {
    "raw_foundation": "External Raw foundation",
    "twin_endpoint": "Raw + SARN twin endpoint",
    "full_remstnet": "ReMSTNet-v3",
}
STAGE_NAMES: Final[tuple[str, ...]] = (
    "host_to_device",
    "model_forward",
    "posterior_validation",
)


class ReMSTNetEfficiencyError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetEfficiencyError(message)


def _latency_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(bool(array.size) and bool(np.isfinite(array).all()) and bool((array > 0.0).all()), "latencies invalid")
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "sample_sd_ms": float(array.std(ddof=1)) if len(array) >= 2 else 0.0,
    }


class Runtime:
    def __init__(self, arm: str, module: nn.Module, device: torch.device, metadata: Mapping[str, Any]) -> None:
        self.arm = arm
        self.module = module
        self.device = device
        self.metadata = dict(metadata)


def build_runtime(arm: str, checkpoint_path: Path, *, device_name: str = DEVICE_NAME) -> Runtime:
    _require(arm in ARMS, f"unknown efficiency arm: {arm}")
    device = torch.device(device_name)
    _require(device.type == "cuda" and torch.cuda.is_available(), "formal efficiency benchmark requires CUDA")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "formal efficiency benchmark requires BF16")
    checkpoint = Path(checkpoint_path).resolve()
    if arm == "full_remstnet":
        model, metadata = load_remst_block_checkpoint(checkpoint, device=device)
        _require(isinstance(model, CoordinatedReMSTNet), "checkpoint is not coordinated ReMSTNet")
        _validate_full_model(model, metadata)
        model.eval()
        return Runtime(arm, model, device, metadata)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "ReMSTNet checkpoint malformed")
    source = payload.get("source_foundation")
    _require(isinstance(source, Mapping), "source foundation metadata missing")
    source_path = Path(str(source.get("source", ""))).resolve()
    _require(source_path.is_file(), "source direct checkpoint missing")
    anchor, anchor_metadata = load_moment_exact_anchor_from_direct_checkpoint(source_path, device=device)
    anchor.eval()
    return Runtime(
        arm,
        anchor,
        device,
        {
            "source_foundation": dict(source),
            "anchor": anchor_metadata,
            "full_checkpoint": str(checkpoint),
        },
    )


def parameter_inventory(runtime: Runtime) -> dict[str, Any]:
    if runtime.arm == "full_remstnet":
        _require(isinstance(runtime.module, CoordinatedReMSTNet), "full runtime module differs")
        counts = remst_block_parameter_counts(runtime.module)
        return {
            "total_unique": counts["total_unique"],
            "trainable_during_remstnet_fit": counts["trainable"],
            "components": counts,
        }
    total = int(sum(parameter.numel() for parameter in runtime.module.parameters()))
    return {
        "total_unique": total,
        "trainable_during_remstnet_fit": 0,
        "components": {"external_scalar_foundation": total},
    }


def _required_names(arm: str) -> tuple[str, ...]:
    names = ("original_view",)
    if arm != "raw_foundation":
        names += ("sarn_view",)
    if arm == "full_remstnet":
        names += ("sarn_support_mask", "sarn_active", "raw_to_sarn_homography")
    return names


def _slice(batch: Mapping[str, torch.Tensor], *, batch_size: int, iteration: int) -> dict[str, torch.Tensor]:
    indices = slice(None) if batch_size == PHYSICAL_BATCH_SIZE else slice(iteration % PHYSICAL_BATCH_SIZE, iteration % PHYSICAL_BATCH_SIZE + 1)
    return {name: tensor[indices] for name, tensor in batch.items()}


def _forward(runtime: Runtime, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if runtime.arm == "full_remstnet":
        _require(isinstance(runtime.module, CoordinatedReMSTNet), "full module differs")
        output = runtime.module(
            batch["original_view"],
            batch["sarn_view"],
            batch["sarn_support_mask"],
            sarn_active=batch["sarn_active"].bool(),
            raw_to_sarn_homography=batch["raw_to_sarn_homography"],
        )
        return output["progress_posterior"], output["mean"], output["variance"]
    raw = _endpoint_from_anchor(runtime.module, batch["original_view"])
    endpoint = raw
    if runtime.arm == "twin_endpoint":
        endpoint = _endpoint_from_anchor(runtime.module, batch["sarn_view"])
    return endpoint["posterior"], endpoint["mean"], endpoint["variance"]


def _iteration(
    runtime: Runtime,
    cpu_batch: Mapping[str, torch.Tensor],
    *,
    timed: bool,
) -> tuple[float | None, dict[str, float]]:
    if timed:
        torch.cuda.synchronize(runtime.device)
        started = time.perf_counter_ns()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
    else:
        events = []
        started = 0
    device_batch = {
        name: cpu_batch[name].to(runtime.device, non_blocking=True)
        for name in _required_names(runtime.arm)
    }
    if timed:
        events[1].record()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        posterior, mean, variance = _forward(runtime, device_batch)
    if timed:
        events[2].record()
    _validate_final_posterior_moments(
        posterior,
        mean,
        variance,
        expected_rows=int(cpu_batch["original_view"].shape[0]),
    )
    if not timed:
        return None, {}
    events[3].record()
    events[3].synchronize()
    total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    stages = {
        name: max(float(events[index].elapsed_time(events[index + 1])), 1.0e-9)
        for index, name in enumerate(STAGE_NAMES)
    }
    _require(total_ms > 0.0, "wall latency non-positive")
    return total_ms, stages


def _measure_profile(runtime: Runtime, pinned: Mapping[str, torch.Tensor], *, batch_size: int) -> dict[str, Any]:
    for iteration in range(WARMUP_ITERATIONS):
        batch = _slice(pinned, batch_size=batch_size, iteration=iteration)
        _iteration(runtime, batch, timed=False)
    torch.cuda.synchronize(runtime.device)
    baseline_allocated = torch.cuda.memory_allocated(runtime.device)
    baseline_reserved = torch.cuda.memory_reserved(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    totals: list[float] = []
    stage_values = {name: [] for name in STAGE_NAMES}
    for iteration in range(TIMED_ITERATIONS):
        batch = _slice(pinned, batch_size=batch_size, iteration=iteration)
        total, stages = _iteration(runtime, batch, timed=True)
        _require(total is not None, "timed iteration missing")
        totals.append(total)
        for name in STAGE_NAMES:
            stage_values[name].append(stages[name])
    peak_allocated = torch.cuda.max_memory_allocated(runtime.device)
    peak_reserved = torch.cuda.max_memory_reserved(runtime.device)
    per_sample = [value / batch_size for value in totals]
    total_seconds = sum(totals) / 1000.0
    return {
        "batch_size": batch_size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "latency_per_batch_ms": _latency_statistics(totals),
        "latency_per_sample_ms": _latency_statistics(per_sample),
        "stage_latency_per_sample_ms": {
            name: _latency_statistics([value / batch_size for value in values])
            for name, values in stage_values.items()
        },
        "throughput_samples_per_second": float(batch_size * TIMED_ITERATIONS / total_seconds),
        "cuda_memory": {
            "baseline_allocated_mib": baseline_allocated / (1024.0**2),
            "baseline_reserved_mib": baseline_reserved / (1024.0**2),
            "peak_allocated_mib": peak_allocated / (1024.0**2),
            "peak_reserved_mib": peak_reserved / (1024.0**2),
            "incremental_peak_allocated_mib": max(0, peak_allocated - baseline_allocated) / (1024.0**2),
            "incremental_peak_reserved_mib": max(0, peak_reserved - baseline_reserved) / (1024.0**2),
        },
    }


def measure_runtime(runtime: Runtime, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    pinned = {name: tensor.pin_memory() for name, tensor in batch.items()}
    return {
        "profiles": {
            f"batch_{batch_size}": _measure_profile(runtime, pinned, batch_size=batch_size)
            for batch_size in PROFILE_BATCH_SIZES
        },
        "autocast": "bfloat16",
        "timing_scope": {
            "included": ["pinned H2D", "all executable model forwards", "posterior validation", "CUDA synchronization"],
            "excluded": ["checkpoint loading", "image decode", "SARN materialization", "accuracy scoring"],
            "sarn_note": "SARN H2D is included for twin/full; SARN materialization is excluded.",
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    b1 = report["measurement"]["profiles"]["batch_1"]
    b8 = report["measurement"]["profiles"][f"batch_{PHYSICAL_BATCH_SIZE}"]
    parameters = report["parameters"]
    lines = [
        "# ReMSTNet-v3 matched CUDA/BF16 efficiency",
        "",
        "| Arm | Parameters (M) | ReMSTNet-fit trainable (M) | B1 mean/P50/P95 (ms) | B8 per-sample mean/P50/P95 (ms) | B8 samples/s | B1 peak allocated (MiB) | B8 peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['display_name']} | {parameters['total_unique']/1e6:.3f} | "
            f"{parameters['trainable_during_remstnet_fit']/1e6:.3f} | "
            f"{b1['latency_per_sample_ms']['mean_ms']:.3f}/{b1['latency_per_sample_ms']['p50_ms']:.3f}/{b1['latency_per_sample_ms']['p95_ms']:.3f} | "
            f"{b8['latency_per_sample_ms']['mean_ms']:.3f}/{b8['latency_per_sample_ms']['p50_ms']:.3f}/{b8['latency_per_sample_ms']['p95_ms']:.3f} | "
            f"{b8['throughput_samples_per_second']:.2f} | {b1['cuda_memory']['peak_allocated_mib']:.1f} | {b8['cuda_memory']['peak_allocated_mib']:.1f} |"
        ),
        "",
        "SARN materialization is outside the timed region; its H2D transfer is inside the twin/full arms.",
        "FLOPs are not reported because the exact moment solver and grid sampling are not completely covered by dispatcher counters; measured latency and peak memory are primary.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    *,
    arm: str,
    checkpoint_path: Path,
    correction_train_manifest_path: Path,
    output_json: Path,
    output_markdown: Path | None = None,
    device_name: str = DEVICE_NAME,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    manifest = Path(correction_train_manifest_path).resolve()
    output = Path(output_json).resolve()
    markdown = output.with_suffix(".md") if output_markdown is None else Path(output_markdown).resolve()
    _require(arm in ARMS and checkpoint.is_file() and manifest.is_file(), "benchmark inputs invalid")
    _require(not output.exists() and not markdown.exists(), "benchmark output exists")
    batch, batch_identity = load_fixed_physical_batch(manifest)
    runtime = build_runtime(arm, checkpoint, device_name=device_name)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "display_name": DISPLAY_NAMES[arm],
        "checkpoint": str(checkpoint),
        "input": batch_identity,
        "execution": {
            "one_arm_per_process": True,
            "image_size": IMAGE_SIZE,
            "fixed_condition": FIXED_CONDITION,
            "profile_batch_sizes": list(PROFILE_BATCH_SIZES),
            "warmup_iterations_per_profile": WARMUP_ITERATIONS,
            "timed_iterations_per_profile": TIMED_ITERATIONS,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": device_name,
            "device_name": torch.cuda.get_device_name(runtime.device),
            "cuda_runtime": torch.version.cuda,
            "cudnn": int(torch.backends.cudnn.version() or 0),
            "autocast_dtype": "bfloat16",
        },
        "parameters": parameter_inventory(runtime),
        "loading_evidence": runtime.metadata,
        "measurement": measure_runtime(runtime, batch),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--correction-train-manifest", type=Path, default=DEFAULT_CORRECTION_TRAIN_MANIFEST)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default=DEVICE_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_benchmark(
        arm=args.arm,
        checkpoint_path=args.checkpoint,
        correction_train_manifest_path=args.correction_train_manifest,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
    )
    print(json.dumps({"status": report["status"], "arm": report["arm"], "output": str(Path(args.output_json).resolve()), "parameters": report["parameters"], "profiles": report["measurement"]["profiles"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARMS", "PROTOCOL", "parameter_inventory", "run_benchmark"]
