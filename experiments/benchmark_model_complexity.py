"""Benchmark comparable active model stacks on one device.

The three paper methods share meter detection and reference-point detection but
activate different reading modules:

* Original Transformer: detector + U2NetP + meter-CLIP + reference detector.
* VDN: detector + VDN direction network + reference detector.
* Ours-final: detector + U2NetP + probabilistic direction expert + reference
  detector + the two frozen ExtraTrees post-processors.

The benchmark uses each module's deployed tensor resolution and invokes the
reference detector twice, matching the current ``correction_mode=off`` reading
path.  Neural FLOPs and latency exclude image decoding, resize/normalization,
YOLO NMS, and JSON serialization; those exclusions are explicit in the output.
FLOPs use PyTorch's aten flop counter, where a multiply-add counts as two FLOPs.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
ANGLE_DIR = PROJECT_DIR / "utils" / "angleDetect"
if str(ANGLE_DIR) not in sys.path:
    sys.path.insert(0, str(ANGLE_DIR))

from pointerSeg.detectSeg import load_u2net_state_dict
from pointerSeg.u2netp import U2NETP
from vitTranforms.meterCilp import meterClip

from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import (
    VDN_PINNED_COMMIT,
    build_vdn_model,
    predict_directions,
    sha256_file,
)

PROTOCOL = "paper_model_stack_efficiency_v1"
VDN_TRAINING_PROTOCOL = "vdn_architecture_syncg_retraining_v1"
DEFAULT_WEIGHTS = {
    "segmentation": ANGLE_DIR / "pointerSeg" / "resultSeg" / "best.pt",
    "meter_detector": (
        ANGLE_DIR / "yoloDetection" / "result" / "yolo_findMeter.pt"
    ),
    "meter_transformer": ANGLE_DIR / "vitTranforms" / "result" / "best.pt",
    "keypoint_detector": (
        ANGLE_DIR / "yoloDetection" / "result" / "yolo_pointbest.pt"
    ),
    "vdn": PROJECT_DIR
    / "artifacts"
    / "runs"
    / "vdn_syncg"
    / "seed_20260722"
    / "best.pt",
    "ours": PROJECT_DIR
    / "artifacts"
    / "runs"
    / "probabilistic_pivot_direction_syncg"
    / "seed_20260722"
    / "best.pt",
    "progress_calibrator": PROJECT_DIR
    / "artifacts"
    / "runs"
    / "progress_calibrator_syncg"
    / "model"
    / "progress_calibrator.joblib",
    "router": PROJECT_DIR
    / "artifacts"
    / "runs"
    / "calibrated_progress_router_syncg"
    / "model"
    / "calibrated_progress_router.joblib",
}
DEFAULT_VDN_SOURCE = (
    PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"
)


@dataclass
class Component:
    name: str
    model: nn.Module
    call: Callable[[], Any]
    input_description: str
    checkpoint: Path


@dataclass
class LoadedStack:
    label: str
    components: dict[str, Component]
    invocation_counts: dict[str, int]
    neural_forward: Callable[[], Any]
    neural_decode: Callable[[Any], None]
    cpu_postprocess: Callable[[], None] | None
    non_neural: dict[str, Any]


def _load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_yolo(path: Path, device: torch.device) -> nn.Module:
    model = YOLO(str(path)).model
    return model.to(device).eval()


def _load_u2net(path: Path, device: torch.device) -> nn.Module:
    model = U2NETP()
    state, _ = load_u2net_state_dict(path, "cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"U2Net checkpoint mismatch: {incompatible}")
    return model.to(device).eval()


def _load_meter_transformer(path: Path, device: torch.device) -> nn.Module:
    model = meterClip(
        image_size=256,
        patch_size=32,
        num_classes=512,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        channels=1,
        dropout=0.1,
        emb_dropout=0.1,
    )
    state = _load_checkpoint(path)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"meter transformer checkpoint mismatch: {incompatible}")
    return model.to(device).eval()


def _load_vdn(
    checkpoint_path: Path,
    source: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    signature = checkpoint.get("signature") or {}
    if signature.get("protocol") != VDN_TRAINING_PROTOCOL:
        raise ValueError("VDN checkpoint is not a formal SyncG retraining")
    if signature.get("vdn_source_commit") != VDN_PINNED_COMMIT:
        raise ValueError("VDN checkpoint source commit is not pinned")
    image_size = int(signature.get("image_size", 384))
    model = build_vdn_model(
        source,
        image_size=image_size,
        imagenet_pretrained=False,
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"VDN checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), signature


def _load_ours(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    signature = checkpoint.get("signature") or {}
    if signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("Ours checkpoint is not a formal probabilistic run")
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(signature.get("angle_bins", 72)),
        imagenet_pretrained=False,
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Ours checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), signature


def _unique_parameter_counts(models: Sequence[nn.Module]) -> dict[str, int]:
    seen: set[int] = set()
    total = 0
    trainable = 0
    for model in models:
        for parameter in model.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return {"total": int(total), "trainable": int(trainable)}


def _tree_nodes(value: Any) -> int:
    if hasattr(value, "tree_"):
        return int(value.tree_.node_count)
    if hasattr(value, "estimators_"):
        return sum(_tree_nodes(item) for item in value.estimators_)
    if hasattr(value, "named_steps"):
        return sum(_tree_nodes(item) for item in value.named_steps.values())
    return 0


def _set_tree_n_jobs(value: Any, n_jobs: int) -> None:
    if hasattr(value, "n_jobs"):
        value.n_jobs = int(n_jobs)
    if hasattr(value, "named_steps"):
        for item in value.named_steps.values():
            _set_tree_n_jobs(item, n_jobs)


def _load_tree_artifact(path: Path) -> tuple[dict[str, Any], Any, np.ndarray]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ValueError(f"{path} is not a signed estimator artifact")
    if payload.get("train_only_certified") is not True:
        raise ValueError(f"{path} is not certified train-only")
    estimator = payload["estimator"]
    feature_names = payload.get("feature_names") or []
    if len(feature_names) != int(estimator.n_features_in_):
        raise ValueError(f"{path} feature schema mismatch")
    features = np.zeros((1, len(feature_names)), dtype=np.float64)
    return payload, estimator, features


def _autocast_context(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager:
    if precision == "amp_fp16":
        return torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        )
    return contextlib.nullcontext()


def _common_components(
    args: argparse.Namespace,
    device: torch.device,
    inputs: dict[str, torch.Tensor],
) -> dict[str, Component]:
    meter_detector = _load_yolo(args.meter_detector_weights, device)
    keypoint_detector = _load_yolo(args.keypoint_detector_weights, device)
    return {
        "meter_detector": Component(
            name="meter_detector",
            model=meter_detector,
            call=lambda: meter_detector(inputs["detector"]),
            input_description="1x3x640x640",
            checkpoint=args.meter_detector_weights,
        ),
        "keypoint_detector": Component(
            name="keypoint_detector",
            model=keypoint_detector,
            call=lambda: keypoint_detector(inputs["detector"]),
            input_description="1x3x640x640",
            checkpoint=args.keypoint_detector_weights,
        ),
    }


def _load_stack(
    label: str,
    args: argparse.Namespace,
    device: torch.device,
) -> LoadedStack:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    inputs = {
        "detector": torch.rand(
            (1, 3, 640, 640),
            generator=generator,
            device=device,
        ),
        "fixed_256_rgb": torch.rand(
            (1, 3, 256, 256),
            generator=generator,
            device=device,
        ),
        "fixed_256_gray_a": torch.rand(
            (1, 1, 256, 256),
            generator=generator,
            device=device,
        ),
        "fixed_256_gray_b": torch.rand(
            (1, 1, 256, 256),
            generator=generator,
            device=device,
        ),
        "fixed_384_rgb": torch.rand(
            (1, 3, 384, 384),
            generator=generator,
            device=device,
        ),
    }
    components = _common_components(args, device, inputs)
    invocation_counts = {
        "meter_detector": 1,
        "keypoint_detector": 2,
    }
    cpu_postprocess: Callable[[], None] | None = None
    non_neural: dict[str, Any] = {}

    if label == "Original Transformer":
        segmenter = _load_u2net(args.segmentation_weights, device)
        transformer = _load_meter_transformer(
            args.meter_transformer_weights,
            device,
        )
        text = transformer.texteconder.tokenize(
            list(map(str, range(101)))
        ).to(device)
        components.update(
            {
                "segmentation": Component(
                    name="segmentation",
                    model=segmenter,
                    call=lambda: segmenter(inputs["fixed_256_rgb"]),
                    input_description="1x3x256x256",
                    checkpoint=args.segmentation_weights,
                ),
                "meter_transformer": Component(
                    name="meter_transformer",
                    model=transformer,
                    call=lambda: transformer(
                        inputs["fixed_256_gray_a"],
                        inputs["fixed_256_gray_b"],
                        text,
                    ),
                    input_description=(
                        "two 1x1x256x256 images + 101 fixed label tokens"
                    ),
                    checkpoint=args.meter_transformer_weights,
                ),
            }
        )
        invocation_counts.update(
            {"segmentation": 1, "meter_transformer": 1}
        )

        def branch_forward():
            segmenter(inputs["fixed_256_rgb"])
            return transformer(
                inputs["fixed_256_gray_a"],
                inputs["fixed_256_gray_b"],
                text,
            )

        def decode(output):
            output[0].argmax(dim=-1)

    elif label == "VDN":
        vdn, signature = _load_vdn(
            args.vdn_checkpoint,
            args.vdn_source,
            device,
        )
        image_size = int(signature.get("image_size", 384))
        if image_size != 384:
            raise ValueError(
                f"formal VDN benchmark expects 384 input, got {image_size}"
            )
        components["vdn"] = Component(
            name="vdn",
            model=vdn,
            call=lambda: vdn(inputs["fixed_384_rgb"]),
            input_description="1x3x384x384",
            checkpoint=args.vdn_checkpoint,
        )
        invocation_counts["vdn"] = 1

        def branch_forward():
            return vdn(inputs["fixed_384_rgb"])

        def decode(output):
            predict_directions(*output)

    elif label == "Ours-final":
        segmenter = _load_u2net(args.segmentation_weights, device)
        ours, signature = _load_ours(args.ours_checkpoint, device)
        image_size = int(signature.get("image_size", 256))
        if image_size != 256:
            raise ValueError(
                f"formal Ours benchmark expects 256 input, got {image_size}"
            )
        components.update(
            {
                "segmentation": Component(
                    name="segmentation",
                    model=segmenter,
                    call=lambda: segmenter(inputs["fixed_256_rgb"]),
                    input_description="1x3x256x256",
                    checkpoint=args.segmentation_weights,
                ),
                "probabilistic_direction": Component(
                    name="probabilistic_direction",
                    model=ours,
                    call=lambda: ours(inputs["fixed_256_rgb"]),
                    input_description="1x3x256x256",
                    checkpoint=args.ours_checkpoint,
                ),
            }
        )
        invocation_counts.update(
            {"segmentation": 1, "probabilistic_direction": 1}
        )
        progress_payload, progress, progress_features = _load_tree_artifact(
            args.progress_calibrator
        )
        router_payload, router, router_features = _load_tree_artifact(
            args.router
        )
        _set_tree_n_jobs(progress, args.tree_n_jobs)
        _set_tree_n_jobs(router, args.tree_n_jobs)

        def branch_forward():
            segmenter(inputs["fixed_256_rgb"])
            return ours(inputs["fixed_256_rgb"])

        def decode(output):
            decode_probabilistic_pivot_direction(*output)

        def postprocess():
            progress.predict(progress_features)
            router.predict(router_features)

        cpu_postprocess = postprocess
        non_neural = {
            "progress_calibrator": {
                "protocol": progress_payload.get("protocol"),
                "features": int(progress.shape[0])
                if hasattr(progress, "shape")
                else int(progress.n_features_in_),
                "tree_nodes": _tree_nodes(progress),
                "benchmark_n_jobs": args.tree_n_jobs,
                "artifact": str(args.progress_calibrator),
                "artifact_sha256": sha256_file(args.progress_calibrator),
                "artifact_bytes": args.progress_calibrator.stat().st_size,
            },
            "router": {
                "protocol": router_payload.get("protocol"),
                "features": int(router.n_features_in_),
                "tree_nodes": _tree_nodes(router),
                "benchmark_n_jobs": args.tree_n_jobs,
                "artifact": str(args.router),
                "artifact_sha256": sha256_file(args.router),
                "artifact_bytes": args.router.stat().st_size,
            },
        }
    else:
        raise ValueError(f"unknown model stack: {label}")

    meter_call = components["meter_detector"].call
    keypoint_call = components["keypoint_detector"].call

    def neural_forward():
        meter_call()
        output = branch_forward()
        keypoint_call()
        keypoint_call()
        return output

    return LoadedStack(
        label=label,
        components=components,
        invocation_counts=invocation_counts,
        neural_forward=neural_forward,
        neural_decode=decode,
        cpu_postprocess=cpu_postprocess,
        non_neural=non_neural,
    )


def _component_flops(
    component: Component,
    *,
    device: torch.device,
    precision: str,
) -> int:
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            with _autocast_context(device, precision):
                output = component.call()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        del output
    return int(counter.get_total_flops())


def _latency_metrics(milliseconds: Sequence[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "iterations": int(values.size),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "serial_fps_from_mean": float(1000.0 / np.mean(values)),
    }


def _benchmark_stack(
    label: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    stack = _load_stack(label, args, device)
    models = [component.model for component in stack.components.values()]
    parameter_counts = _unique_parameter_counts(models)

    component_metrics: dict[str, Any] = {}
    for name, component in stack.components.items():
        flops = _component_flops(
            component,
            device=device,
            precision=args.precision,
        )
        component_metrics[name] = {
            "parameters": _unique_parameter_counts([component.model])["total"],
            "flops_per_invocation": flops,
            "invocations_per_sample": stack.invocation_counts[name],
            "flops_per_sample": flops * stack.invocation_counts[name],
            "input": component.input_description,
            "checkpoint": str(component.checkpoint),
            "checkpoint_sha256": sha256_file(component.checkpoint),
            "checkpoint_bytes": component.checkpoint.stat().st_size,
        }
    total_flops = int(
        sum(value["flops_per_sample"] for value in component_metrics.values())
    )

    def neural_iteration() -> None:
        with torch.inference_mode(), _autocast_context(
            device,
            args.precision,
        ):
            output = stack.neural_forward()
            stack.neural_decode(output)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def cpu_iteration() -> None:
        if stack.cpu_postprocess is not None:
            stack.cpu_postprocess()

    def total_iteration() -> None:
        neural_iteration()
        cpu_iteration()

    for _ in range(args.warmup):
        total_iteration()
    neural_durations = []
    cpu_durations = []
    total_durations = []
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        neural_iteration()
        neural_durations.append((time.perf_counter() - started) * 1000.0)
    if stack.cpu_postprocess is not None:
        for _ in range(args.iterations):
            started = time.perf_counter()
            cpu_iteration()
            cpu_durations.append((time.perf_counter() - started) * 1000.0)
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        total_iteration()
        total_durations.append((time.perf_counter() - started) * 1000.0)

    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else None
    )
    checkpoint_paths = {
        component.checkpoint.resolve()
        for component in stack.components.values()
    }
    checkpoint_paths.update(
        path.resolve()
        for path in (
            (args.progress_calibrator, args.router)
            if label == "Ours-final"
            else ()
        )
    )
    result = {
        "label": label,
        "parameters": parameter_counts,
        "flops": total_flops,
        "flop_convention": "multiply-add counts as two FLOPs",
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "latency": {
            "neural_stack": _latency_metrics(neural_durations),
            "cpu_postprocess": (
                _latency_metrics(cpu_durations)
                if cpu_durations
                else None
            ),
            "total_active_stack": _latency_metrics(total_durations),
        },
        "components": component_metrics,
        "non_neural": stack.non_neural,
        "model_artifact_bytes": int(
            sum(path.stat().st_size for path in checkpoint_paths)
        ),
    }
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _fmt_count(value: int, divisor: float) -> str:
    return f"{float(value) / divisor:.2f}"


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    methods = payload.get("methods") or []
    expected_labels = (
        "Original Transformer",
        "VDN",
        "Ours-final",
    )
    if tuple(method.get("label") for method in methods) != expected_labels:
        raise ValueError("efficiency payload has an unexpected method order")
    checks = 0
    for method in methods:
        components = method.get("components") or {}
        expected_flops = sum(
            int(component["flops_per_sample"])
            for component in components.values()
        )
        if int(method["flops"]) != expected_flops:
            raise ValueError(f"{method['label']}: FLOP total mismatch")
        checks += 1
        expected_parameters = sum(
            int(component["parameters"])
            for component in components.values()
        )
        if int(method["parameters"]["total"]) != expected_parameters:
            raise ValueError(f"{method['label']}: parameter total mismatch")
        checks += 1
        peak = int(method["peak_gpu_allocated_bytes"])
        if not 0 < peak <= int(payload["gpu_total_memory_bytes"]):
            raise ValueError(f"{method['label']}: invalid peak GPU memory")
        checks += 1
        latency = method.get("latency") or {}
        for section in ("neural_stack", "total_active_stack"):
            value = latency.get(section) or {}
            if (
                int(value.get("iterations", -1))
                != int(payload["timed_iterations"])
                or float(value.get("mean_ms", 0.0)) <= 0.0
                or float(value.get("p95_ms", 0.0)) <= 0.0
            ):
                raise ValueError(
                    f"{method['label']}: invalid {section} latency"
                )
            checks += 1
    if methods[-1]["latency"].get("cpu_postprocess") is None:
        raise ValueError("Ours-final CPU postprocess latency is missing")
    checks += 1
    return {
        "verified": True,
        "checks": checks,
        "method_labels": list(expected_labels),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Unified model-stack efficiency and complexity",
        "",
        (
            f"All methods were measured on {payload['gpu']} with batch size 1, "
            f"{payload['precision']}, {payload['warmup_iterations']} warm-up "
            f"iterations and {payload['timed_iterations']} timed iterations. "
            "Active shared modules are counted in every method. FLOPs use "
            "deployed tensor resolutions and count a multiply-add as two "
            "operations."
        ),
        "",
        (
            "| Method | Neural params (M) | FLOPs (G) | Peak allocated VRAM (MiB) | "
            "Neural mean (ms) | CPU post (ms) | "
            "Total mean (ms) | Total median (ms) | Total P95 (ms) | Serial FPS |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in payload["methods"]:
        latency = method["latency"]
        neural = latency["neural_stack"]
        cpu = latency["cpu_postprocess"]
        total = latency["total_active_stack"]
        cpu_text = f"{cpu['mean_ms']:.2f}" if cpu is not None else "0.00"
        lines.append(
            f"| {method['label']} | "
            f"{_fmt_count(method['parameters']['total'], 1e6)} | "
            f"{_fmt_count(method['flops'], 1e9)} | "
            f"{_fmt_count(method['peak_gpu_allocated_bytes'], 1024**2)} | "
            f"{neural['mean_ms']:.2f} | "
            f"{cpu_text} | {total['mean_ms']:.2f} | "
            f"{total['median_ms']:.2f} | "
            f"{total['p95_ms']:.2f} | "
            f"{total['serial_fps_from_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "Included: raw neural forwards for meter detection, the "
                "method-specific reading modules, and two reference-detector "
                "forwards. Ours also includes batch-one prediction by its two "
                "frozen ExtraTrees post-processors in wall-clock latency."
            ),
            "",
            (
                "Excluded: image decode, resize/normalization, YOLO NMS, crop "
                "construction, result rendering, and JSON serialization. The "
                "table is therefore a controlled active-model-stack benchmark, "
                "not a camera-to-API latency claim."
            ),
            "",
            "## Component accounting",
            "",
            "| Method | Component | Calls/sample | Params (M) | FLOPs/call (G) | Input |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for method in payload["methods"]:
        for name, component in method["components"].items():
            lines.append(
                f"| {method['label']} | {name} | "
                f"{component['invocations_per_sample']} | "
                f"{_fmt_count(component['parameters'], 1e6)} | "
                f"{_fmt_count(component['flops_per_invocation'], 1e9)} | "
                f"{component['input']} |"
            )
    ours = payload["methods"][-1]
    lines.extend(
        [
            "",
            "## Ours non-neural post-processors",
            "",
            "| Module | Features | Trees/nodes | Benchmark threads |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in ours["non_neural"].items():
        lines.append(
            f"| {name} | {value['features']} | "
            f"{value['tree_nodes']} nodes | {value['benchmark_n_jobs']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp_fp16"),
        default="amp_fp16",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--tree-n-jobs",
        type=int,
        default=1,
        help=(
            "CPU threads for each frozen ExtraTrees estimator during the "
            "batch-one benchmark; changing n_jobs does not change predictions"
        ),
    )
    parser.add_argument(
        "--meter-detector-weights",
        type=Path,
        default=DEFAULT_WEIGHTS["meter_detector"],
    )
    parser.add_argument(
        "--keypoint-detector-weights",
        type=Path,
        default=DEFAULT_WEIGHTS["keypoint_detector"],
    )
    parser.add_argument(
        "--segmentation-weights",
        type=Path,
        default=DEFAULT_WEIGHTS["segmentation"],
    )
    parser.add_argument(
        "--meter-transformer-weights",
        type=Path,
        default=DEFAULT_WEIGHTS["meter_transformer"],
    )
    parser.add_argument(
        "--vdn-checkpoint",
        type=Path,
        default=DEFAULT_WEIGHTS["vdn"],
    )
    parser.add_argument(
        "--ours-checkpoint",
        type=Path,
        default=DEFAULT_WEIGHTS["ours"],
    )
    parser.add_argument(
        "--progress-calibrator",
        type=Path,
        default=DEFAULT_WEIGHTS["progress_calibrator"],
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=DEFAULT_WEIGHTS["router"],
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=DEFAULT_VDN_SOURCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/efficiency/model_stack_efficiency.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("benchmark requires warmup >= 1 and iterations >= 2")
    if args.tree_n_jobs == 0:
        raise ValueError("tree_n_jobs cannot be zero")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal VRAM benchmark requires an available CUDA device")
    for name in (
        "meter_detector_weights",
        "keypoint_detector_weights",
        "segmentation_weights",
        "meter_transformer_weights",
        "vdn_checkpoint",
        "ours_checkpoint",
        "progress_calibrator",
        "router",
    ):
        value = getattr(args, name).resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    args.vdn_source = args.vdn_source.resolve()
    if not args.vdn_source.is_dir():
        raise FileNotFoundError(args.vdn_source)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    methods = [
        _benchmark_stack(label, args, device)
        for label in ("Original Transformer", "VDN", "Ours-final")
    ]
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "batch_size": 1,
        "precision": args.precision,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "tree_n_jobs": args.tree_n_jobs,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_cudnn": torch.backends.cudnn.deterministic,
            "tf32_enabled": False,
        },
        "scope": {
            "included": (
                "active neural forwards at deployed resolutions; method decode; "
                "two reference detector forwards; Ours ExtraTrees prediction"
            ),
            "excluded": (
                "image decode, resize/normalization, YOLO NMS, crop construction, "
                "rendering, and JSON serialization"
            ),
            "reference_detector_calls_per_sample": 2,
            "flop_counter": "torch.utils.flop_counter.FlopCounterMode",
            "flop_convention": "multiply-add counts as two FLOPs",
        },
        "methods": methods,
        "source_sha256": {
            "benchmark": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["verification"] = validate_payload(payload)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
