"""Matched CUDA/BF16 efficiency benchmark for A15.2 METT.

Run one inference arm per process so that peak CUDA memory is attributable to
that arm.  The fixed first-eight correction-train samples are materialized at
256 x 256 under ``perspective_severe`` before timing.  Image decode,
projective corruption, and SARN-view/support/homography construction are
therefore excluded.  Required host-to-device transfers, one or two frozen
anchor forwards, the METT correction when selected, posterior validation, and
the final synchronization are included.

The twin-endpoint and full-METT arms deliberately call the *same* frozen
EfficientNet-B0 anchor twice (Raw then SARN).  Parameter totals deduplicate the
shared anchor and consequently count it once.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.a15_fteb import (  # noqa: E402
    _endpoint_from_anchor,
)
from experiments.benchmark_a15_2_fteb_efficiency import (  # noqa: E402
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
from experiments.benchmark_cagh_v5_paper_efficiency import (  # noqa: E402
    latency_statistics,
    parameter_inventory,
)
from experiments.prepare_a13_correction_scene_split import (  # noqa: E402
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
)
from experiments.train_a15_2_mett import (  # noqa: E402
    METT_VARIANTS,
    load_mett_models,
    publication_model_identity,
    validate_mett_variant_metadata,
)


PROTOCOL: Final[str] = "syncg_a15_2_mett_matched_cuda_efficiency_v1"
ARMS: Final[tuple[str, ...]] = (
    "raw_only",
    "twin_endpoint",
    "full_mett",
)
DISPLAY_NAMES: Final[dict[str, str]] = {
    "raw_only": "METT frozen Raw anchor",
    "twin_endpoint": "METT frozen Raw + SARN twin endpoint",
    "full_mett": "A15.2 METT full correction",
}
STAGE_NAMES: Final[tuple[str, ...]] = (
    "host_to_device",
    "raw_anchor_forward",
    "sarn_anchor_forward",
    "mett_correction",
    "posterior_moment_validation",
)
AUTOCAST_DTYPE_NAME: Final[str] = "bfloat16"


class METTEfficiencyError(ValueError):
    """A METT efficiency input or measurement is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise METTEfficiencyError(message)


@dataclass(slots=True)
class METTEfficiencyRuntime:
    """Loaded modules for one isolated METT benchmark arm."""

    arm: str
    anchor: nn.Module
    correction: nn.Module | None
    device: torch.device
    loading_evidence: dict[str, Any]
    experiment_variant: str = "full"

    @property
    def parameter_roots(self) -> tuple[nn.Module, ...]:
        # Repeating ``anchor`` documents the two-call twin path while the
        # shared inventory helper deduplicates the actual Parameter objects.
        roots: tuple[nn.Module, ...] = (self.anchor,)
        if self.arm != "raw_only":
            roots += (self.anchor,)
        if self.correction is not None:
            roots += (self.correction,)
        return roots


def build_runtime(
    arm: str,
    *,
    checkpoint_path: Path,
    device_name: str = DEVICE_NAME,
    expected_variant: str = "full",
) -> METTEfficiencyRuntime:
    """Strict-load one checkpoint and retain only modules used by ``arm``."""

    _require(arm in ARMS, f"unknown METT efficiency arm: {arm}")
    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    device = torch.device(device_name)
    _require(device.type == "cuda", "formal METT efficiency requires CUDA")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")

    anchor, loaded_correction, metadata = load_mett_models(
        Path(checkpoint_path).resolve(), device=device
    )
    validated_variant = validate_mett_variant_metadata(
        metadata, expected_variant=expected_variant
    )
    metadata = {
        **metadata,
        "validated_mett_variant": validated_variant,
    }
    if expected_variant == "full":
        metadata["validated_full_mett"] = validated_variant
    _require(
        all(not parameter.requires_grad for parameter in anchor.parameters()),
        "METT anchor is not frozen",
    )
    loaded_correction_parameters = sum(
        int(parameter.numel()) for parameter in loaded_correction.parameters()
    )
    correction: nn.Module | None
    if arm == "full_mett":
        correction = loaded_correction
    else:
        correction = None
        del loaded_correction
        gc.collect()
        torch.cuda.empty_cache()
    anchor.eval()
    if correction is not None:
        correction.eval()
    return METTEfficiencyRuntime(
        arm=arm,
        anchor=anchor,
        correction=correction,
        device=device,
        loading_evidence={
            **dict(metadata),
            "strict_checkpoint_loader": True,
            "loaded_correction_parameters": loaded_correction_parameters,
            "unused_correction_released_before_measurement": arm != "full_mett",
            "anchor_module_instances": 1,
            "anchor_forward_calls_per_inference": 1 if arm == "raw_only" else 2,
        },
        experiment_variant=expected_variant,
    )


def mett_parameter_inventory(runtime: METTEfficiencyRuntime) -> dict[str, Any]:
    """Count unique executable parameters and isolate correction trainables."""

    inventory = parameter_inventory(runtime.parameter_roots)
    anchor_parameters = tuple(runtime.anchor.parameters())
    correction_parameters = (
        tuple(runtime.correction.parameters())
        if runtime.correction is not None
        else ()
    )
    anchor_ids = {id(parameter) for parameter in anchor_parameters}
    correction_ids = {id(parameter) for parameter in correction_parameters}
    _require(
        anchor_ids.isdisjoint(correction_ids),
        "METT anchor and correction parameters overlap",
    )
    anchor_count = sum(int(parameter.numel()) for parameter in anchor_parameters)
    correction_count = sum(
        int(parameter.numel()) for parameter in correction_parameters
    )
    trainable_correction = sum(
        int(parameter.numel())
        for parameter in correction_parameters
        if parameter.requires_grad
    )
    _require(
        int(inventory["total_parameters"]) == anchor_count + correction_count,
        "METT unique parameter decomposition differs",
    )
    _require(
        int(inventory["trainable_parameters"]) == trainable_correction,
        "METT trainable parameters are not correction-only",
    )
    inventory.update(
        {
            "components": {
                "shared_frozen_anchor_counted_once": anchor_count,
                "mett_correction": correction_count,
            },
            "trainable_correction_parameters": trainable_correction,
            "anchor_forward_calls_per_inference": (
                1 if runtime.arm == "raw_only" else 2
            ),
            "shared_anchor_counting": (
                "one unique frozen anchor; twin/full execute it sequentially "
                "twice without duplicating parameters"
            ),
        }
    )
    return inventory


def _slice_cpu_batch(
    batch: Mapping[str, torch.Tensor], *, batch_size: int, iteration: int
) -> dict[str, torch.Tensor]:
    _require(
        batch_size in PROFILE_BATCH_SIZES,
        "METT efficiency profile batch size differs",
    )
    if batch_size == PHYSICAL_BATCH_SIZE:
        indices = slice(None)
    else:
        index = int(iteration) % PHYSICAL_BATCH_SIZE
        indices = slice(index, index + 1)
    return {name: tensor[indices] for name, tensor in batch.items()}


def _required_tensor_names(arm: str) -> tuple[str, ...]:
    _require(arm in ARMS, f"unknown METT efficiency arm: {arm}")
    names = ("original_view",)
    if arm != "raw_only":
        names += ("sarn_view",)
    if arm == "full_mett":
        names += (
            "sarn_support_mask",
            "sarn_active",
            "raw_to_sarn_homography",
        )
    return names


def _to_device(
    batch: Mapping[str, torch.Tensor],
    *,
    arm: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    names = _required_tensor_names(arm)
    _require(all(name in batch for name in names), "METT batch tensors are missing")
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in names
    }


def cuda_stage_durations(events: Sequence[Any]) -> dict[str, float]:
    """Convert six consecutive event boundaries into five stage durations."""

    _require(len(events) == len(STAGE_NAMES) + 1, "METT event roster differs")
    result = {
        name: float(events[index].elapsed_time(events[index + 1]))
        for index, name in enumerate(STAGE_NAMES)
    }
    event_total = float(events[0].elapsed_time(events[-1]))
    _require(
        math.isclose(
            sum(result.values()), event_total, rel_tol=1.0e-5, abs_tol=1.0e-3
        ),
        "METT consecutive CUDA stage sum differs from event total",
    )
    return result


def _run_iteration(
    runtime: METTEfficiencyRuntime,
    cpu_batch: Mapping[str, torch.Tensor],
    *,
    events: Sequence[torch.cuda.Event] | None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[float | None, dict[str, float]]:
    """Execute one inference arm, optionally timing one synchronized call."""

    rows = int(cpu_batch["original_view"].shape[0])
    _require(rows >= 1, "METT efficiency CPU batch is empty")
    if events is not None:
        _require(
            len(events) == len(STAGE_NAMES) + 1,
            "METT event roster differs",
        )
        torch.cuda.synchronize(runtime.device)
        started = clock_ns()
        events[0].record()
    else:
        started = 0

    device_batch = _to_device(
        cpu_batch, arm=runtime.arm, device=runtime.device
    )
    if events is not None:
        events[1].record()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        raw = _endpoint_from_anchor(
            runtime.anchor, device_batch["original_view"]
        )
        if events is not None:
            events[2].record()

        sarn: Mapping[str, Any] | None = None
        if runtime.arm != "raw_only":
            sarn = _endpoint_from_anchor(
                runtime.anchor, device_batch["sarn_view"]
            )
        if events is not None:
            events[3].record()

        if runtime.arm == "raw_only":
            posterior = raw["posterior"]
            mean = raw["mean"]
            variance = raw["variance"]
        elif runtime.arm == "twin_endpoint":
            _require(sarn is not None, "METT SARN endpoint is missing")
            posterior = sarn["posterior"]
            mean = sarn["mean"]
            variance = sarn["variance"]
        else:
            _require(
                sarn is not None and runtime.correction is not None,
                "METT correction runtime is incomplete",
            )
            output = runtime.correction(
                raw["posterior"],
                raw["features"],
                sarn["posterior"],
                sarn["features"],
                device_batch["sarn_support_mask"],
                sarn_active=device_batch["sarn_active"],
                raw_to_sarn_homography=device_batch[
                    "raw_to_sarn_homography"
                ],
            )
            posterior = output["progress_posterior"]
            mean = output["mean"]
            variance = output["variance"]
        if events is not None:
            events[4].record()

        _validate_final_posterior_moments(
            posterior, mean, variance, expected_rows=rows
        )

    if events is None:
        return None, {}
    events[5].record()
    events[5].synchronize()
    total_ms = (clock_ns() - started) / 1_000_000.0
    _require(total_ms > 0.0, "METT wall-clock latency is non-positive")
    return total_ms, cuda_stage_durations(events)


def _positive_latency_statistics(
    values: Sequence[float],
) -> dict[str, float]:
    # Back-to-back events around an intentional no-op can quantize to zero.
    return latency_statistics(tuple(max(float(value), 1.0e-9) for value in values))


def _measure_profile(
    runtime: METTEfficiencyRuntime,
    pinned_batch: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, Any]:
    from torch.utils.flop_counter import FlopCounterMode

    flop_batch = _slice_cpu_batch(
        pinned_batch, batch_size=batch_size, iteration=0
    )
    with FlopCounterMode(display=False) as flop_counter:
        _run_iteration(runtime, flop_batch, events=None)
    torch.cuda.synchronize(runtime.device)
    dispatcher_flops = int(flop_counter.get_total_flops())
    _require(dispatcher_flops > 0, "METT dispatcher FLOP count is empty")

    for iteration in range(WARMUP_ITERATIONS):
        selected = _slice_cpu_batch(
            pinned_batch, batch_size=batch_size, iteration=iteration
        )
        _run_iteration(runtime, selected, events=None)
    torch.cuda.synchronize(runtime.device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    baseline_allocated = int(torch.cuda.memory_allocated(runtime.device))
    baseline_reserved = int(torch.cuda.memory_reserved(runtime.device))

    events = tuple(
        torch.cuda.Event(enable_timing=True)
        for _ in range(len(STAGE_NAMES) + 1)
    )
    batch_latencies: list[float] = []
    stage_latencies: dict[str, list[float]] = {
        name: [] for name in STAGE_NAMES
    }
    for iteration in range(TIMED_ITERATIONS):
        selected = _slice_cpu_batch(
            pinned_batch, batch_size=batch_size, iteration=iteration
        )
        elapsed, stages = _run_iteration(runtime, selected, events=events)
        _require(elapsed is not None, "METT timed iteration has no latency")
        batch_latencies.append(float(elapsed))
        for name in STAGE_NAMES:
            stage_latencies[name].append(float(stages[name]))

    peak_allocated = int(torch.cuda.max_memory_allocated(runtime.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(runtime.device))
    per_sample = [value / float(batch_size) for value in batch_latencies]
    stage_per_sample = {
        name: [value / float(batch_size) for value in values]
        for name, values in stage_latencies.items()
    }
    return {
        "batch_size": batch_size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "timed_samples": TIMED_ITERATIONS * batch_size,
        "flops": {
            "dispatcher_supported_per_batch": dispatcher_flops,
            "dispatcher_supported_per_sample": (
                dispatcher_flops / float(batch_size)
            ),
            "counter": "torch.utils.flop_counter.FlopCounterMode",
            "convention": "one multiply-add counts as two FLOPs",
            "coverage": (
                "dispatcher estimate for registered ATen operators; "
                "unregistered custom/operator arithmetic is omitted"
            ),
        },
        "latency_batch_ms": latency_statistics(batch_latencies),
        "latency_per_sample_ms": latency_statistics(per_sample),
        "throughput_samples_per_second": (
            1000.0 * TIMED_ITERATIONS * batch_size / sum(batch_latencies)
        ),
        "stage_latency_per_sample_ms": {
            name: _positive_latency_statistics(values)
            for name, values in stage_per_sample.items()
        },
        "cuda_memory": {
            "baseline_allocated_mib": baseline_allocated / (1024.0**2),
            "baseline_reserved_mib": baseline_reserved / (1024.0**2),
            "peak_allocated_mib": peak_allocated / (1024.0**2),
            "peak_reserved_mib": peak_reserved / (1024.0**2),
            "incremental_peak_allocated_mib": max(
                0, peak_allocated - baseline_allocated
            )
            / (1024.0**2),
            "incremental_peak_reserved_mib": max(
                0, peak_reserved - baseline_reserved
            )
            / (1024.0**2),
            "peak_reset_independently_for_profile": True,
        },
    }


def measure_runtime(
    runtime: METTEfficiencyRuntime,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure fixed B1 latency and B8 throughput on CUDA BF16."""

    _require(runtime.device.type == "cuda", "formal METT efficiency requires CUDA")
    pinned = {name: tensor.pin_memory() for name, tensor in batch.items()}
    profiles = {
        f"batch_{batch_size}": _measure_profile(
            runtime, pinned, batch_size=batch_size
        )
        for batch_size in PROFILE_BATCH_SIZES
    }
    executed = ["host_to_device", "raw_anchor_forward"]
    if runtime.arm != "raw_only":
        executed.append("sarn_anchor_forward")
    if runtime.arm == "full_mett":
        executed.append("mett_correction")
    executed.append("posterior_moment_validation")
    return {
        "profiles": profiles,
        "autocast": AUTOCAST_DTYPE_NAME,
        "timing_scope": {
            "included": [
                "pinned host-to-device transfer for arm-required tensors",
                "one frozen Raw anchor forward",
                "one additional frozen SARN anchor forward for twin/full",
                "METT correction for full_mett",
                "posterior/moment validation and final CUDA synchronization",
            ],
            "excluded": [
                "checkpoint and module construction",
                "source image decode and canonical ROI extraction",
                "projective corruption",
                "SARN view, support mask, and homography construction",
                "ground-truth access and accuracy scoring",
            ],
            "sarn_preprocessing_accounting": (
                "SARN materialization is excluded; SARN H2D and the second "
                "anchor forward are included for twin/full"
            ),
        },
        "stage_semantics": {
            "uniform_stage_order": list(STAGE_NAMES),
            "executed_stage_order": executed,
            "intentional_no_op_stages": [
                name for name in STAGE_NAMES if name not in executed
            ],
            "anchor_forward_calls_per_inference": (
                1 if runtime.arm == "raw_only" else 2
            ),
            "cuda_stage_api": "torch.cuda.Event elapsed_time",
            "total_latency_api": (
                "perf_counter_ns around H2D/forward/validation/final sync"
            ),
        },
    }


def _environment(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    cuda = device.type == "cuda" and torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "synthetic",
        "cuda_runtime": torch.version.cuda if cuda else None,
        "cudnn": int(torch.backends.cudnn.version() or 0) if cuda else None,
        "autocast_dtype": AUTOCAST_DTYPE_NAME,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    profiles = report["measurement"]["profiles"]
    parameters = report["parameters"]
    publication_model = report.get("publication_model")
    paper_name = (
        str(publication_model.get("short_name"))
        if isinstance(publication_model, Mapping)
        else "A15.2 METT"
    )
    lines = [
        f"# {paper_name} matched CUDA/BF16 efficiency",
        "",
        (
            "Each file measures one isolated arm. The twin and full paths "
            "execute the shared anchor twice while counting its parameters once."
        ),
        "",
        "| Arm | Params (M) | Trainable correction (M) | Dispatcher FLOPs/sample (G) | B1 mean/P50/P95 (ms) | B8 per-sample mean/P50/P95 (ms) | B8 throughput (sample/s) | B1 peak alloc/reserved (MiB) | B8 peak alloc/reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    b1 = profiles["batch_1"]
    b8 = profiles[f"batch_{PHYSICAL_BATCH_SIZE}"]
    b1_latency = b1["latency_per_sample_ms"]
    b8_latency = b8["latency_per_sample_ms"]
    b1_memory = b1["cuda_memory"]
    b8_memory = b8["cuda_memory"]
    lines.append(
        f"| {report['display_name']} | "
        f"{int(parameters['total_parameters']) / 1_000_000.0:.3f} | "
        f"{int(parameters['trainable_correction_parameters']) / 1_000_000.0:.3f} | "
        f"{float(b1['flops']['dispatcher_supported_per_sample']) / 1_000_000_000.0:.3f} | "
        f"{float(b1_latency['mean_ms']):.3f}/"
        f"{float(b1_latency['p50_ms']):.3f}/"
        f"{float(b1_latency['p95_ms']):.3f} | "
        f"{float(b8_latency['mean_ms']):.3f}/"
        f"{float(b8_latency['p50_ms']):.3f}/"
        f"{float(b8_latency['p95_ms']):.3f} | "
        f"{float(b8['throughput_samples_per_second']):.3f} | "
        f"{float(b1_memory['peak_allocated_mib']):.1f}/"
        f"{float(b1_memory['peak_reserved_mib']):.1f} | "
        f"{float(b8_memory['peak_allocated_mib']):.1f}/"
        f"{float(b8_memory['peak_reserved_mib']):.1f} |"
    )
    lines.extend(
        [
            "",
            "## B1 stage latency (ms/sample)",
            "",
            "| Stage | Mean | P50 | P95 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in b1["stage_latency_per_sample_ms"].items():
        lines.append(
            f"| {name} | {float(values['mean_ms']):.3f} | "
            f"{float(values['p50_ms']):.3f} | "
            f"{float(values['p95_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "SARN decode/projective/materialization is outside the timed region; "
            "SARN H2D and the second anchor forward are inside it.",
            "FLOPs are dispatcher estimates; unsupported custom/operator "
            "arithmetic is explicitly outside the count.",
            "",
        ]
    )
    return "\n".join(lines)


BatchLoader = Callable[[Path], tuple[dict[str, torch.Tensor], dict[str, Any]]]
RuntimeFactory = Callable[..., METTEfficiencyRuntime]
MeasurementFunction = Callable[
    [METTEfficiencyRuntime, Mapping[str, torch.Tensor]], dict[str, Any]
]


def run_benchmark(
    *,
    arm: str,
    checkpoint_path: Path,
    correction_train_manifest_path: Path,
    output_json: Path,
    output_markdown: Path | None = None,
    batch_loader: BatchLoader = load_fixed_physical_batch,
    runtime_factory: RuntimeFactory = build_runtime,
    measurement_function: MeasurementFunction = measure_runtime,
    device_name: str = DEVICE_NAME,
    expected_variant: str = "full",
) -> dict[str, Any]:
    """Run one isolated arm and write JSON plus a concise Markdown table."""

    _require(arm in ARMS, f"unknown METT efficiency arm: {arm}")
    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    checkpoint = Path(checkpoint_path).resolve()
    manifest = Path(correction_train_manifest_path).resolve()
    output = Path(output_json).resolve()
    markdown = (
        output.with_suffix(".md")
        if output_markdown is None
        else Path(output_markdown).resolve()
    )
    _require(
        output not in {checkpoint, manifest}
        and markdown not in {checkpoint, manifest, output},
        "METT efficiency output cannot overwrite an input or sibling artifact",
    )
    _require(
        not output.exists() and not markdown.exists(),
        "METT efficiency output already exists",
    )

    batch, batch_identity = batch_loader(manifest)
    runtime = runtime_factory(
        arm,
        checkpoint_path=checkpoint,
        device_name=device_name,
        expected_variant=expected_variant,
    )
    _require(runtime.arm == arm, "runtime factory returned the wrong arm")
    parameters = mett_parameter_inventory(runtime)
    measurement = measurement_function(runtime, batch)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "display_name": (
            publication_model_identity(expected_variant)["display_name"]
            if arm == "full_mett" and expected_variant == "direct_scalar"
            else DISPLAY_NAMES[arm]
        ),
        "experiment_variant": expected_variant,
        "publication_model": publication_model_identity(expected_variant),
        "paper_comparison_arms": list(ARMS),
        "checkpoint": str(checkpoint),
        "input": batch_identity,
        "execution": {
            "one_arm_per_process": True,
            "image_size": IMAGE_SIZE,
            "fixed_condition": FIXED_CONDITION,
            "profile_batch_sizes": list(PROFILE_BATCH_SIZES),
            "warmup_iterations_per_profile": WARMUP_ITERATIONS,
            "timed_iterations_per_profile": TIMED_ITERATIONS,
            "batch_1_sampling": (
                "timed_iteration_index_modulo_8_over_fixed_roster"
            ),
            "batch_8_sampling": "all_eight_fixed_samples_each_iteration",
            "timing_used_for_model_selection": False,
        },
        "environment": _environment(device_name),
        "parameters": parameters,
        "loading_evidence": runtime.loading_evidence,
        "measurement": measurement,
        "data_access": {
            "correction_train_manifest": True,
            "checkpoint": True,
            "ground_truth_or_accuracy_scoring": False,
            "formal_holdout": False,
            "field_photos": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--experiment-variant", choices=tuple(METT_VARIANTS), default="full"
    )
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_benchmark(
        arm=args.arm,
        checkpoint_path=args.checkpoint,
        correction_train_manifest_path=args.correction_train_manifest,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        expected_variant=args.experiment_variant,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "output": str(Path(args.output_json).resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "METTEfficiencyRuntime",
    "PROTOCOL",
    "STAGE_NAMES",
    "build_argument_parser",
    "build_runtime",
    "cuda_stage_durations",
    "measure_runtime",
    "mett_parameter_inventory",
    "run_benchmark",
]
