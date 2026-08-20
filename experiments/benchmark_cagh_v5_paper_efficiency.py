"""Batch-one efficiency benchmark for the four paper-facing CV methods.

The benchmark consumes the first ``N`` rows of the same label-free plain-ROI
manifest used by the accuracy runner.  PNG decoding is performed once before
timing.  Each timed call includes the method's native preprocessing, host to
device transfer, learned forward passes, analytic postprocessing, and device
synchronization.  It deliberately does not estimate FLOPs: the VDN and legacy
Transformer stacks contain data-dependent detection/postprocessing for which a
single static FLOP count would be misleading.

Run one method per process so peak CUDA memory is attributable to that method.
No ground-truth fields are accepted or read.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import platform
import statistics
import sys
import time
import types
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_cagh_v5_plain_paper_batch import (
    ModelPredictionFailure,
    load_canonical_roi,
    load_manifest,
)


PROTOCOL: Final[str] = "cagh_v5_paper_batch1_efficiency_v1"
METHODS: Final[tuple[str, ...]] = (
    "cagh_v5_final",
    "vdn_official200",
    "original_transformer_auto_reference",
    "resnet18_direct",
)
DEFAULT_MANIFEST: Final[Path] = Path(
    r"C:\pointer_read\cagh_v5_plain_paper_8x6_v1\input_manifest.jsonl"
)
DEFAULT_LIMIT: Final[int] = 100
DEFAULT_WARMUP: Final[int] = 20


class EfficiencyBenchmarkError(ValueError):
    """The label-free input or benchmark configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EfficiencyBenchmarkError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


@dataclass(frozen=True, slots=True)
class BenchmarkInput:
    sample_id: str
    image_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    method: str
    display_name: str
    predict: Callable[[np.ndarray], float]
    parameter_roots: tuple[Any, ...]
    accepted_failure_types: tuple[type[BaseException], ...] = ()


def load_benchmark_inputs(
    manifest_path: Path,
    *,
    limit: int,
) -> tuple[tuple[BenchmarkInput, ...], dict[str, Any]]:
    """Load only the first ``limit`` canonical ROIs from a label-free manifest."""

    _require(limit >= 1, "limit must be positive")
    manifest = Path(manifest_path).resolve()
    rows = load_manifest(manifest)
    _require(limit <= len(rows), f"limit {limit} exceeds manifest rows {len(rows)}")
    selected = rows[:limit]
    inputs: list[BenchmarkInput] = []
    for row in selected:
        _payload, image = load_canonical_roi(row)
        inputs.append(
            BenchmarkInput(
                sample_id=row.sample_id,
                image_bgr=np.ascontiguousarray(image),
            )
        )
    ids = [item.sample_id for item in inputs]
    return tuple(inputs), {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "manifest_rows": len(rows),
        "selection": "first_n_manifest_rows_in_file_order",
        "selected_rows": len(inputs),
        "selected_sample_ids_sha256": _canonical_sha256(ids),
        "first_sample_id": ids[0],
        "last_sample_id": ids[-1],
        "ground_truth_read": False,
    }


def build_target(
    method: str,
    *,
    device: str,
    resnet18_checkpoint: Path | None,
) -> BenchmarkTarget:
    """Build one method through the exact paper-inference loader."""

    _require(method in METHODS, f"unknown method: {method}")
    if method == "resnet18_direct":
        _require(
            resnet18_checkpoint is not None,
            "--resnet18-checkpoint is required for resnet18_direct",
        )
        from experiments.resnet18_direct_progress import load_checkpoint_predictor

        loaded_name, batch_predict = load_checkpoint_predictor(
            Path(resnet18_checkpoint), device_name=device
        )

        def predict(image_bgr: np.ndarray) -> float:
            values = batch_predict((image_bgr,))
            _require(len(values) == 1, "ResNet-18 returned the wrong batch size")
            return float(values[0])

        return BenchmarkTarget(
            method=method,
            display_name=f"ResNet-18 direct ({loaded_name})",
            predict=predict,
            parameter_roots=(batch_predict,),
        )

    from experiments.run_cagh_v5_plain_paper_batch import build_predictor

    runner_method, display_name = {
        "cagh_v5_final": (
            "full_seed_20262020",
            "CAGH-V5 final (seed 20262020)",
        ),
        "vdn_official200": (
            "vdn_official200_terminal_seed20",
            "VDN official-200 (terminal seed 20)",
        ),
        "original_transformer_auto_reference": (
            "original_transformer_legacy_auto_reference",
            "Original Transformer + automatic reference",
        ),
    }[method]
    predictor = build_predictor(runner_method, device=device)
    return BenchmarkTarget(
        method=method,
        display_name=display_name,
        predict=predictor,
        parameter_roots=(predictor,),
        accepted_failure_types=(ModelPredictionFailure,),
    )


def _object_children(value: Any) -> tuple[Any, ...]:
    """Return bounded object-graph children used to locate loaded modules."""

    if inspect.ismethod(value):
        return tuple(item for item in (value.__self__, value.__func__) if item is not None)
    if inspect.isfunction(value):
        closure = value.__closure__ or ()
        children: list[Any] = []
        for cell in closure:
            try:
                children.append(cell.cell_contents)
            except ValueError:
                continue
        return tuple(children)
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return tuple(value)
    if isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
            int,
            float,
            complex,
            bool,
            type(None),
            Path,
            np.ndarray,
            torch.Tensor,
            torch.device,
            types.ModuleType,
            type,
        ),
    ):
        return ()
    try:
        attributes = vars(value)
    except TypeError:
        return ()
    return tuple(attributes.values())


def parameter_inventory(roots: Sequence[Any]) -> dict[str, Any]:
    """Count unique parameters from every learned module reachable from roots."""

    queue: deque[Any] = deque(roots)
    visited: set[int] = set()
    modules: list[nn.Module] = []
    while queue:
        value = queue.popleft()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        _require(
            len(visited) <= 50_000,
            "parameter-root object graph exceeded the bounded traversal",
        )
        if isinstance(value, nn.Module):
            modules.append(value)
            # PyTorch recursively registers every learned child module.
            continue
        queue.extend(_object_children(value))

    unique_parameters: dict[int, nn.Parameter] = {}
    trainable_ids: set[int] = set()
    component_rows: list[dict[str, Any]] = []
    for module in modules:
        parameters = tuple(module.parameters(recurse=True))
        for parameter in parameters:
            unique_parameters.setdefault(id(parameter), parameter)
            if parameter.requires_grad:
                trainable_ids.add(id(parameter))
        component_rows.append(
            {
                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                "declared_parameters": sum(int(item.numel()) for item in parameters),
            }
        )
    _require(bool(unique_parameters), "no PyTorch model parameters found in predictor")
    return {
        "total_parameters": sum(
            int(parameter.numel()) for parameter in unique_parameters.values()
        ),
        "trainable_parameters": sum(
            int(unique_parameters[identity].numel()) for identity in trainable_ids
        ),
        "parameter_bytes": sum(
            int(parameter.numel() * parameter.element_size())
            for parameter in unique_parameters.values()
        ),
        "unique_parameter_tensors": len(unique_parameters),
        "learned_component_roots": component_rows,
        "counting_rule": (
            "unique torch.nn.Parameter objects reachable from the loaded inference "
            "predictor; shared tensors counted once"
        ),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "latency list is empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_statistics(latencies_ms: Sequence[float]) -> dict[str, float]:
    values = tuple(float(value) for value in latencies_ms)
    _require(
        bool(values) and all(math.isfinite(value) and value > 0.0 for value in values),
        "latencies must be finite and positive",
    )
    total = sum(values)
    return {
        "mean_ms": statistics.fmean(values),
        "sample_sd_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50_ms": _quantile(values, 0.50),
        "p90_ms": _quantile(values, 0.90),
        "p95_ms": _quantile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "throughput_images_per_second": 1000.0 * len(values) / total,
    }


def _finite_progress(value: Any) -> float:
    _require(not isinstance(value, bool), "predictor returned a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EfficiencyBenchmarkError("predictor returned a non-numeric value") from exc
    _require(
        math.isfinite(result) and 0.0 <= result <= 1.0,
        "predictor returned progress outside [0,1]",
    )
    return result


def _invoke(target: BenchmarkTarget, image_bgr: np.ndarray) -> tuple[bool, str | None]:
    try:
        _finite_progress(target.predict(np.ascontiguousarray(image_bgr).copy()))
    except target.accepted_failure_types as exc:
        return False, str(getattr(exc, "code", type(exc).__name__))[:128]
    return True, None


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_target(
    target: BenchmarkTarget,
    inputs: Sequence[BenchmarkInput],
    *,
    device_name: str,
    warmup: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Measure one already-loaded method; image decoding is outside this call."""

    _require(bool(inputs), "benchmark input roster is empty")
    _require(warmup >= 0, "warmup must be non-negative")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.cuda.set_device(device)

    warmup_failures: Counter[str] = Counter()
    for index in range(warmup):
        passed, code = _invoke(target, inputs[index % len(inputs)].image_bgr)
        if not passed:
            warmup_failures[str(code)] += 1
    _device_synchronize(device)

    gc.collect()
    memory: dict[str, Any]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _device_synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
    else:
        baseline_allocated = baseline_reserved = 0

    latencies_ms: list[float] = []
    failures: Counter[str] = Counter()
    passes = 0
    for item in inputs:
        _device_synchronize(device)
        started = clock_ns()
        passed, code = _invoke(target, item.image_bgr)
        _device_synchronize(device)
        elapsed_ms = (clock_ns() - started) / 1_000_000.0
        _require(elapsed_ms > 0.0, "clock produced non-positive latency")
        latencies_ms.append(elapsed_ms)
        if passed:
            passes += 1
        else:
            failures[str(code)] += 1

    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory = {
            "supported": True,
            "baseline_allocated_mib": baseline_allocated / (1024.0**2),
            "baseline_reserved_mib": baseline_reserved / (1024.0**2),
            "peak_allocated_mib": peak_allocated / (1024.0**2),
            "peak_reserved_mib": peak_reserved / (1024.0**2),
            "incremental_peak_allocated_mib": max(
                0, peak_allocated - baseline_allocated
            )
            / (1024.0**2),
            "measurement_api": "torch.cuda.reset_peak_memory_stats/max_memory_allocated",
        }
    else:
        memory = {
            "supported": False,
            "reason": "peak CUDA memory is unavailable on a non-CUDA device",
        }

    return {
        "batch_size": 1,
        "warmup_calls": int(warmup),
        "timed_calls": len(inputs),
        "successful_calls": passes,
        "model_reported_failure_calls": len(inputs) - passes,
        "failure_codes": dict(sorted(failures.items())),
        "warmup_failure_codes": dict(sorted(warmup_failures.items())),
        "latency": latency_statistics(latencies_ms),
        "raw_latency_ms": latencies_ms,
        "cuda_memory": memory,
        "timing_scope": {
            "included": [
                "model-native CPU preprocessing",
                "host-to-device transfer",
                "all learned forward passes",
                "analytic postprocessing and result validation",
                "CUDA synchronization",
            ],
            "excluded": [
                "model/checkpoint construction",
                "PNG file read and decode",
                "ground-truth loading and accuracy scoring",
            ],
        },
    }


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "cuda_runtime": torch.version.cuda if cuda else None,
        "cudnn": int(torch.backends.cudnn.version() or 0) if cuda else None,
    }


def render_efficiency_markdown(report: Mapping[str, Any]) -> str:
    measurement = report["measurement"]
    latency = measurement["latency"]
    memory = measurement["cuda_memory"]
    parameters = report["parameters"]
    peak = (
        f"{float(memory['peak_allocated_mib']):.1f}"
        if memory.get("supported")
        else "N/A"
    )
    lines = [
        "# Batch-1 efficiency",
        "",
        (
            "Timing excludes PNG decoding and includes model-native preprocessing, "
            "host-to-device transfer, inference, postprocessing, and CUDA synchronization."
        ),
        "",
        "| Method | Params (M) | Mean latency (ms) | P50 (ms) | P95 (ms) | Throughput (img/s) | Peak CUDA memory (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['display_name']} | "
            f"{int(parameters['total_parameters']) / 1_000_000.0:.3f} | "
            f"{float(latency['mean_ms']):.3f} | "
            f"{float(latency['p50_ms']):.3f} | "
            f"{float(latency['p95_ms']):.3f} | "
            f"{float(latency['throughput_images_per_second']):.3f} | {peak} |"
        ),
        "",
        (
            f"Roster: first {measurement['timed_calls']} rows; batch size 1; "
            f"warmup {measurement['warmup_calls']}; device "
            f"{report['environment']['device_name']}."
        ),
        "",
        "FLOPs: not reported. The multi-stage detection and postprocessing paths are data-dependent, so no static FLOP tool was applied.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    *,
    method: str,
    manifest_path: Path,
    output_json: Path,
    output_markdown: Path | None,
    device_name: str,
    limit: int,
    warmup: int,
    resnet18_checkpoint: Path | None = None,
    target_factory: Callable[..., BenchmarkTarget] = build_target,
) -> dict[str, Any]:
    inputs, input_identity = load_benchmark_inputs(manifest_path, limit=limit)
    target = target_factory(
        method,
        device=device_name,
        resnet18_checkpoint=resnet18_checkpoint,
    )
    _require(target.method == method, "target factory returned the wrong method")
    parameters = parameter_inventory(target.parameter_roots)
    device = torch.device(device_name)
    measurement = measure_target(
        target,
        inputs,
        device_name=device_name,
        warmup=warmup,
    )
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": target.method,
        "display_name": target.display_name,
        "input": input_identity,
        "environment": _runtime_environment(device),
        "parameters": parameters,
        "flops": {
            "value": None,
            "status": "not_reported",
            "reason": (
                "No static FLOP tool was applied because the compared pipelines "
                "contain data-dependent detection and analytic postprocessing."
            ),
        },
        "measurement": measurement,
    }
    output = Path(output_json).resolve()
    _require(
        output != Path(manifest_path).resolve(),
        "output JSON cannot overwrite the input manifest",
    )
    _write_json(output, report)
    markdown_path = (
        output.with_suffix(".md")
        if output_markdown is None
        else Path(output_markdown).resolve()
    )
    _require(markdown_path != output, "JSON and Markdown outputs must differ")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_efficiency_markdown(report), encoding="utf-8", newline="\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--resnet18-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        method=args.method,
        manifest_path=args.manifest,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
        limit=args.limit,
        warmup=args.warmup,
        resnet18_checkpoint=args.resnet18_checkpoint,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "method": report["method"],
                "output": str(Path(args.output_json).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkInput",
    "BenchmarkTarget",
    "EfficiencyBenchmarkError",
    "METHODS",
    "build_target",
    "latency_statistics",
    "load_benchmark_inputs",
    "measure_target",
    "parameter_inventory",
    "render_efficiency_markdown",
    "run_benchmark",
]
