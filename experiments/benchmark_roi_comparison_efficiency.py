"""Measure the three matched ROI comparators, one FP32 arm per process.

Native DeepLab and VDN arms end at a probability map and direction respectively;
their valid-output counts are not reading coverage. Automatic-geometry arms
execute the selected Industrial reference front-end (by default SyncG-trained
YOLO four-keypoint references), then the direction model only when geometry succeeds. Every ROI,
including early failures, contributes one latency measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark_cagh_v5_paper_efficiency import (
    BenchmarkTarget,
    _runtime_environment,
    load_benchmark_inputs,
    measure_target,
    parameter_inventory,
)
from experiments.roi_geometry_comparison import (
    decode_progress_from_keypoints,
    direction_from_pointer_probability,
    resize_keypoints,
    write_json,
)

ARMS = (
    "yolo11s_pose4kp",
    "deeplab_component",
    "vdn_component",
    "deeplab_auto_geometry",
    "vdn_auto_geometry",
)
DEFAULT_MANIFEST = Path("C:/pointer_read/unified_real_photo_progress_v1/input_manifest.jsonl")
DEFAULT_REFERENCE = PROJECT_ROOT / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt"


class PredictionFailure(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _progress(points: np.ndarray) -> float:
    value, failure, _telemetry = decode_progress_from_keypoints(points)
    if value is None or failure is not None:
        raise PredictionFailure(str(failure or "reading_decode_failed"))
    return float(value)


def default_checkpoint(arm: str, seed: int) -> Path:
    if arm.startswith("vdn"):
        return PROJECT_ROOT / f"artifacts/runs/geoprr_vdn_matched/seed_{seed}/last.pt"
    run_root = PROJECT_ROOT / f"artifacts/runs/roi_comparison_pilot/seed_{seed}"
    if arm == "yolo11s_pose4kp":
        return run_root / "yolo11s_pose4kp/ultralytics/weights/best.pt"
    return run_root / "deeplabv3plus_roi/best.pt"


def build_target(args: argparse.Namespace) -> tuple[BenchmarkTarget, dict[str, Any], list[tuple[str, Any, int]]]:
    from ultralytics import YOLO

    checkpoint = (args.checkpoint or default_checkpoint(args.arm, args.seed)).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "precision": "float32; no autocast; YOLO quantize=None",
        "condition": "clean",
    }

    if args.arm == "yolo11s_pose4kp":
        from experiments.evaluate_yolo11s_pose4kp import _prediction_points, _square_training_input

        model = YOLO(str(checkpoint))
        train_args = (model.ckpt or {}).get("train_args", {})
        image_size = int(train_args.get("imgsz", 384))
        checkpoint_seed = int(train_args.get("seed", args.seed))
        if checkpoint_seed != args.seed:
            raise ValueError("YOLO checkpoint seed differs from requested seed")
        model.to(device)
        model.model.float().eval()

        @torch.inference_mode()
        def predict(image_bgr: np.ndarray) -> float:
            square = _square_training_input(image_bgr, image_size=image_size)
            results = model.predict(
                source=[square], imgsz=image_size, batch=1, device=str(device),
                conf=0.05, iou=0.7, max_det=5, augment=False, save=False,
                verbose=False, quantize=None,
            )
            if len(results) != 1:
                raise RuntimeError("YOLO returned an unexpected batch size")
            points, failure, _telemetry = _prediction_points(
                results[0], minimum_keypoint_confidence=0.05
            )
            if points is None or failure is not None:
                raise PredictionFailure(str(failure or "pose_detection_failed"))
            return _progress(points)

        metadata.update({
            "image_size": image_size,
            "evaluation_scope": "provided ROI to normalized reading; four predicted keypoints",
            "success_definition": "valid decoded normalized reading; no accuracy labels read",
            "preprocessing": "OpenCV bilinear square resize, then Ultralytics native preprocessing",
        })
        return BenchmarkTarget(
            args.arm, "YOLO11s-Pose-4KP", predict, (model.model,), (PredictionFailure,),
        ), metadata, [("YOLO11s-Pose-4KP", model.model, image_size)]

    from experiments.evaluate_missing_roi_direction_baselines import (
        _extract_three_points, _load_deeplab, _load_vdn,
    )
    from experiments.deeplabv3plus_roi import normalized_rgb_tensor
    from experiments.vdn_baseline import normalized_bgr_tensor, predict_directions

    loader_args = argparse.Namespace(
        checkpoint=checkpoint, seed=args.seed, image_size=256,
        vdn_source=args.vdn_source, expected_epochs=args.expected_epochs,
    )
    is_deeplab = args.arm.startswith("deeplab")
    model, payload, _seed, image_size = (
        _load_deeplab(loader_args) if is_deeplab else _load_vdn(loader_args)
    )
    model.to(device).float().eval()
    component_name = "DeepLabV3+-ROI" if is_deeplab else "VDN"
    automatic = args.arm.endswith("auto_geometry")
    roots: list[Any] = [model]
    components = [(component_name, model, image_size)]
    detector = None
    reference_weights = None
    source_pose = args.reference_detector_kind == "source_pose4kp"
    reference_size = 384 if source_pose else 640
    if automatic:
        reference_weights = (
            args.reference_detector_weights
            or (default_checkpoint("yolo11s_pose4kp", args.seed) if source_pose else DEFAULT_REFERENCE)
        ).resolve()
        detector = YOLO(str(reference_weights))
        if source_pose:
            reference_train_args = (detector.ckpt or {}).get("train_args", {})
            reference_size = int(reference_train_args.get("imgsz", 384))
        detector.to(device)
        detector.model.float().eval()
        roots.append(detector.model)
        components.append(("source pose reference detector" if source_pose else "legacy three-point detector", detector.model, reference_size))

    from experiments.roi_reference_geometry import extract_source_pose_references, source_pose_input

    @torch.inference_mode()
    def predict(image_bgr: np.ndarray) -> float:
        points = None
        if detector is not None:
            if source_pose:
                detector_input = source_pose_input(image_bgr, image_size=reference_size)
            else:
                # Preserve the historical prepare_geometry input exactly.
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                detector_input = np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
            results = detector.predict(
                source=[detector_input], imgsz=reference_size, batch=1, device=str(device),
                conf=0.05 if source_pose else 0.25, iou=0.70,
                max_det=5 if source_pose else 20, verbose=False, quantize=None,
            )
            if len(results) != 1:
                raise RuntimeError("reference detector returned an unexpected batch size")
            if source_pose:
                points_three, failure, _telemetry = extract_source_pose_references(
                    results[0], source_shape=image_bgr.shape, image_size=reference_size,
                    minimum_keypoint_confidence=0.05,
                )
            else:
                points_three, failure, _telemetry = _extract_three_points(results[0])
            if points_three is None or failure is not None:
                raise PredictionFailure(f"automatic_geometry:{failure or 'failed'}")
            points = np.stack((points_three[0], points_three[0], points_three[1], points_three[2]))
            points = resize_keypoints(points, source_shape=image_bgr.shape, output_size=image_size)

        resized = cv2.resize(image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        tensor = normalized_rgb_tensor(resized) if is_deeplab else normalized_bgr_tensor(resized)
        inputs = tensor.unsqueeze(0).to(device, non_blocking=True)
        if is_deeplab:
            probability = torch.sigmoid(model(inputs)).float().cpu().numpy()[0, 0]
            if not np.isfinite(probability).all():
                raise PredictionFailure("non_finite_probability_map")
            if points is None:
                # Sentinel for the common timer's scalar validator; no reading
                # is produced by this component-only benchmark.
                return 0.0
            direction, failure, _telemetry = direction_from_pointer_probability(
                probability, points[0], threshold=0.5
            )
            if direction is None or failure is not None:
                raise PredictionFailure(str(failure or "invalid_pointer_mask"))
        else:
            heatmaps, vector_maps = model(inputs)
            directions, peaks, valid = predict_directions(heatmaps.float(), vector_maps.float())
            direction = directions.detach().cpu().numpy()[0].astype(np.float32)
            peak = float(peaks.detach().cpu().numpy()[0])
            is_valid = bool(valid.detach().cpu().numpy()[0])
            if not is_valid or not math.isfinite(peak):
                raise PredictionFailure("invalid_vdn_direction")
            if points is None:
                return 0.0
        points[1] = points[0] + direction * (0.30 * image_size)
        return _progress(points)

    metadata.update({
        "checkpoint_epoch": int(payload.get("epoch") or 0),
        "image_size": int(image_size),
        "preprocessing": "OpenCV bilinear square resize; ImageNet-normalized " + ("RGB" if is_deeplab else "BGR"),
        "evaluation_scope": (
            "provided ROI to normalized reading with Industrial automatic geometry"
            if automatic else ("ROI to pointer probability map" if is_deeplab else "ROI to normalized direction vector")
        ),
        "success_definition": (
            "valid decoded normalized reading; no accuracy labels read"
            if automatic else "valid native component output only; successful_calls is NOT reading coverage"
        ),
        "reference_detector": str(reference_weights) if automatic else None,
        "reference_detector_kind": args.reference_detector_kind if automatic else None,
        "reference_configuration": (
            {"imgsz": reference_size, "conf": 0.05 if source_pose else 0.25,
             "iou": 0.70, "max_det": 5 if source_pose else 20,
             "grayscale_three_channel": not source_pose,
             "pointer_tip_used": False,
             "keypoint_confidence": 0.05 if source_pose else None,
             "letterbox": "square training input" if source_pose else "Ultralytics batch-one native rectangular padding"}
            if automatic else None
        ),
        "geometry_failure_behavior": "record timed failure before direction-model inference" if automatic else None,
    })
    name = component_name + (
        (" + source pose references" if source_pose else " + legacy automatic geometry")
        if automatic else " (native component)"
    )
    return BenchmarkTarget(args.arm, name, predict, tuple(roots), (PredictionFailure,)), metadata, components


def neural_flops(components: Sequence[tuple[str, Any, int]], device: torch.device) -> dict[str, Any]:
    """Count fixed-shape neural forwards after all latency/memory measurements."""
    from torch.utils.flop_counter import FlopCounterMode

    rows = []
    for name, model, size in components:
        row: dict[str, Any] = {"component": name, "input_shape": [1, 3, size, size]}
        try:
            inputs = torch.zeros(1, 3, size, size, device=device, dtype=torch.float32)
            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                model(inputs)
            value = int(counter.get_total_flops())
            row.update({"status": "measured_supported_aten_ops", "flops": value, "gflops": value / 1e9})
        except (RuntimeError, NotImplementedError, TypeError, AttributeError) as exc:
            row.update({"status": "unavailable", "flops": None, "reason": str(exc)[:400]})
        rows.append(row)
    available = all(row["flops"] is not None for row in rows)
    total = sum(row["flops"] for row in rows) if available else None
    return {
        "status": "fixed_shape_neural_components" if available else "partially_unavailable",
        "value": total, "gflops": total / 1e9 if total is not None else None,
        "components": rows,
        "counting_rule": "PyTorch aten FlopCounterMode supported operators; multiply-add counts as two FLOPs",
        "scope": "one fixed-shape forward per listed neural module; excludes preprocessing, NMS, CPU geometry and unsupported operations",
        "pipeline_limitation": "not end-to-end FLOPs; each reference module is counted at its listed square resolution (legacy reference timed letterbox shapes may vary); automatic-geometry failures omit the direction forward during timing",
    }


def runtime_precision(target: BenchmarkTarget, image: np.ndarray, components: Sequence[tuple[str, Any, int]]) -> dict[str, Any]:
    """Observe a real extra inference after timing, without timing hook overhead."""
    observed: dict[str, list[str]] = {}
    handles = []
    for name, model, _size in components:
        observed[name] = []

        def record(_module: Any, inputs: tuple[Any, ...], *, component: str = name) -> None:
            observed[component].extend(
                str(value.dtype) for value in inputs
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            )

        handles.append(model.register_forward_pre_hook(record))
    failure = None
    try:
        target.predict(np.ascontiguousarray(image).copy())
    except PredictionFailure as exc:
        failure = exc.code
    finally:
        for handle in handles:
            handle.remove()
    dtypes = {dtype for values in observed.values() for dtype in values}
    if not dtypes or dtypes != {"torch.float32"}:
        raise RuntimeError(f"observed model inputs are not FP32: {observed}")
    return {
        "model_input_dtypes": {name: sorted(set(values)) for name, values in observed.items()},
        "failure_code": failure,
        "scope": "one additional real input inference after timing; empty component list means early geometry failure skipped that component",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    measurement = report["measurement"]
    latency = measurement["latency"]
    memory = measurement["cuda_memory"]
    peak = f"{memory['peak_allocated_mib']:.1f}" if memory.get("supported") else "N/A"
    gflops = report["flops"].get("gflops")
    flop_text = f"{gflops:.3f}" if gflops is not None else "N/A"
    return "\n".join([
        "# Matched ROI comparator efficiency", "",
        "| Arm | Params (M) | Fixed neural GFLOPs | Mean (ms) | P50 (ms) | P95 (ms) | img/s | Peak CUDA MiB | Valid outputs / calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {report['display_name']} | {report['parameters']['total_parameters'] / 1e6:.3f} | {flop_text} | {latency['mean_ms']:.3f} | {latency['p50_ms']:.3f} | {latency['p95_ms']:.3f} | {latency['throughput_images_per_second']:.3f} | {peak} | {measurement['successful_calls']} / {measurement['timed_calls']} |",
        "", report["configuration"]["evaluation_scope"] + ".",
        report["configuration"]["success_definition"] + ".", "",
        "FP32, batch 1, one arm per process. First manifest rows; image decode is outside timing. All model preprocessing, transfers, neural inference, decoding and synchronization are included. Failures remain timed.", "",
        report["flops"]["counting_rule"] + ". " + report["flops"]["pipeline_limitation"] + ".", "",
    ])


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    inputs, identity = load_benchmark_inputs(args.manifest, limit=args.limit)
    target, metadata, components = build_target(args)
    # Record checkpoint architecture before Ultralytics' native warmup fuses
    # convolution/batchnorm layers, matching existing parameter inventories.
    parameters = parameter_inventory(target.parameter_roots)
    measurement = measure_target(target, inputs, device_name=args.device, warmup=args.warmup)
    measurement["success_definition"] = metadata["success_definition"]
    metadata["observed_precision"] = runtime_precision(target, inputs[0].image_bgr, components)
    device = torch.device(args.device)
    report = {
        "schema_version": 1, "status": "complete",
        "protocol": "matched_roi_comparator_batch1_efficiency_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.arm, "display_name": target.display_name,
        "input": identity, "configuration": metadata,
        "environment": _runtime_environment(device), "parameters": parameters,
        "measurement": measurement, "flops": neural_flops(components, device),
    }
    report["parameters"]["scope"] = "loaded checkpoint neural modules before native inference fusion; automatic-geometry arms include reference detector"
    write_json(args.output_json, report)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--expected-epochs", type=int, default=200)
    parser.add_argument("--vdn-source", type=Path, default=PROJECT_ROOT / "artifacts/vendor/VectorDetectionNetwork")
    parser.add_argument("--reference-detector-kind", choices=("source_pose4kp", "legacy_three_point"), default="source_pose4kp")
    parser.add_argument("--reference-detector-weights", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1 or args.warmup < 0:
        raise ValueError("limit must be positive and warmup non-negative")
    report = run_benchmark(args)
    print(json.dumps({"method": report["method"], "status": report["status"], "latency": report["measurement"]["latency"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
