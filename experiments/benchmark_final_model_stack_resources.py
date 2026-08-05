"""Same-roster resource benchmark for the three frozen paper model stacks.

This is a controlled active-stack benchmark, not an accuracy evaluation.  Each
method receives the same 100-image SyncG/train roster and the same frozen dial
crop for method-specific inputs.  Image decode, crop construction, resize,
normalization, host-to-device transfer, YOLO post-processing, rendering, and
serialization are outside the timed region.  The timed region contains one raw
meter-detector forward, the method-specific branch, two raw reference-detector
forwards, and the method decoder at batch size one.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANGLE_DIR = PROJECT_DIR / "utils" / "angleDetect"
if str(ANGLE_DIR) not in sys.path:
    sys.path.insert(0, str(ANGLE_DIR))

from pointerSeg.detectSeg import load_u2net_state_dict  # noqa: E402
from pointerSeg.u2netp import U2NETP  # noqa: E402
from vitTranforms.meterCilp import meterClip  # noqa: E402

from experiments.pivot_direction_fallback import tensor_from_bbox  # noqa: E402
from experiments.probabilistic_pivot_direction import (  # noqa: E402
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import (  # noqa: E402
    VDN_PINNED_COMMIT,
    affine_for_dial,
    build_vdn_model,
    predict_directions,
    sha256_file,
    vdn_tensor_from_bbox,
)


PROTOCOL = "paper_final_model_stack_same_roster_resources_v2"
FREEZE_PROTOCOL = "paper_final_model_stack_same_roster_resources_freeze_v2"
ROSTER_PROTOCOL = "pepd_only_syncg_train_resource_roster_v1"
VDN_OFFICIAL_PROTOCOL = "vdn_syncg_official_200_epoch_from_scratch_v1"
METHODS = ("Original Transformer", "VDN official-200", "PEPD-final")
DEFAULT_ROSTER = (
    PROJECT_DIR
    / "artifacts"
    / "runs"
    / "pepd_e2e_resources_v2_final_frontend"
    / "roster.json"
)
DEFAULT_MANIFEST = PROJECT_DIR / "artifacts" / "manifests" / "syncg_train.jsonl"
DEFAULT_FREEZE = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "final_model_stack_same_roster_resources_v2_freeze.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "runs"
    / "efficiency"
    / "final_model_stack_same_roster_resources_v2"
)
DEFAULT_WEIGHTS = {
    "segmentation": PROJECT_DIR / "artifacts" / "runs" / "syncg_segmentation" / "best.pt",
    "meter_detector": ANGLE_DIR / "yoloDetection" / "result" / "yolo_findMeter.pt",
    "keypoint_detector": ANGLE_DIR / "yoloDetection" / "result" / "yolo_pointbest.pt",
    "transformer": ANGLE_DIR / "vitTranforms" / "result" / "best.pt",
    "vdn": (
        PROJECT_DIR
        / "artifacts"
        / "runs"
        / "vdn_syncg_official200"
        / "seed_20260720"
        / "best.pt"
    ),
    "pepd": (
        PROJECT_DIR
        / "artifacts"
        / "runs"
        / "pepd_convergence_phase2"
        / "seed_20260722"
        / "best.pt"
    ),
}
DEFAULT_VDN_SOURCE = PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"


@dataclass
class Component:
    model: nn.Module
    call: Callable[[], Any]
    checkpoint: Path
    invocations: int
    input_description: str


@dataclass
class LoadedStack:
    label: str
    components: dict[str, Component]
    forward: Callable[[], Any]
    decode: Callable[[Any], None]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_sha256(row)


def _load_roster_rows(
    roster_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roster = _read_json(roster_path)
    if (
        roster.get("protocol") != ROSTER_PROTOCOL
        or roster.get("status") != "frozen"
        or int(roster.get("samples", -1)) != 100
        or "SyncG/train only" not in str(roster.get("scope"))
    ):
        raise ValueError("benchmark requires the frozen 100-sample SyncG/train roster")
    if sha256_file(manifest_path) != roster.get("manifest_sha256"):
        raise ValueError("roster/manifest hash binding failed")
    manifest_by_id = {
        str(row.get("sample_id")): row for row in _read_jsonl(manifest_path)
    }
    entries = roster.get("entries") or []
    if len(entries) != 100:
        raise ValueError("roster entries are incomplete")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for entry in entries:
        sample_id = str(entry.get("sample_id") or "")
        if not sample_id or sample_id in seen or sample_id not in manifest_by_id:
            raise ValueError("roster has a blank, duplicate, or unknown sample_id")
        seen.add(sample_id)
        row = manifest_by_id[sample_id]
        if row.get("dataset") != "SyncG" or row.get("split") != "train":
            raise ValueError("restricted or non-train data entered the roster")
        if _row_sha256(row) != entry.get("manifest_row_sha256"):
            raise ValueError(f"{sample_id}: manifest row identity drifted")
        image_path = Path(str(row["image_path"])).resolve()
        if image_path != Path(str(entry["image_path"])).resolve() or not image_path.is_file():
            raise ValueError(f"{sample_id}: image path binding failed")
        bbox = ((row.get("metadata") or {}).get("dial_bbox"))
        if not isinstance(bbox, list) or len(bbox) < 4:
            raise ValueError(f"{sample_id}: frozen dial bbox is absent")
        image_hash = sha256_file(image_path)
        inventory.append(
            {
                "sample_id": sample_id,
                "image_sha256": image_hash,
                "image_bytes": image_path.stat().st_size,
            }
        )
        selected.append(
            {
                "sample_id": sample_id,
                "group_id": str(entry.get("group_id") or ""),
                "image_path": str(image_path),
                "dial_bbox": [float(value) for value in bbox[:4]],
            }
        )
    identity = {
        "roster_sha256": sha256_file(roster_path),
        "manifest_sha256": sha256_file(manifest_path),
        "image_content_inventory_sha256": _canonical_sha256(inventory),
        "samples": len(selected),
        "unique_groups": len({row["group_id"] for row in selected}),
    }
    return selected, identity


def _decode(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.size == 0:
        raise ValueError(f"cannot decode {path}")
    return image


def _normalized_rgb(image_bgr: np.ndarray, size: int) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _crop(image: np.ndarray, bbox: Sequence[float], size: int) -> np.ndarray:
    matrix = affine_for_dial(bbox, output_size=size, expansion=1.25)
    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _prepare_inputs(row: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    image = _decode(Path(str(row["image_path"])))
    bbox = row["dial_bbox"]
    crop_256 = _crop(image, bbox, 256)
    crop_640 = _crop(image, bbox, 640)
    gray = cv2.cvtColor(crop_256, cv2.COLOR_BGR2GRAY)
    gray_array = np.ascontiguousarray(gray[None, ...], dtype=np.float32) / 127.5 - 1.0
    cpu = {
        "meter_detector": _normalized_rgb(image, 640).unsqueeze(0),
        "reference_detector": _normalized_rgb(crop_640, 640).unsqueeze(0),
        "rgb_256": tensor_from_bbox(image, bbox, image_size=256, expansion=1.25).unsqueeze(0),
        "gray_256": torch.from_numpy(gray_array).unsqueeze(0),
        "vdn_384": vdn_tensor_from_bbox(image, bbox, image_size=384, expansion=1.25).unsqueeze(0),
    }
    return {name: tensor.to(device) for name, tensor in cpu.items()}


def _load_yolo(path: Path, device: torch.device) -> nn.Module:
    return YOLO(str(path)).model.to(device).eval()


def _load_u2net(path: Path, device: torch.device) -> nn.Module:
    model = U2NETP()
    state, _ = load_u2net_state_dict(path, "cpu")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"U2Net checkpoint mismatch: {incompatible}")
    return model.to(device).eval()


def _load_transformer(path: Path, device: torch.device) -> nn.Module:
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
    state = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Transformer checkpoint mismatch: {incompatible}")
    return model.to(device).eval()


def _load_vdn(path: Path, source: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    signature = checkpoint.get("signature") or {}
    if (
        signature.get("protocol") != VDN_OFFICIAL_PROTOCOL
        or signature.get("vdn_source_commit") != VDN_PINNED_COMMIT
        or int(signature.get("image_size", -1)) != 384
        or int(signature.get("epochs", -1)) != 200
    ):
        raise ValueError("VDN checkpoint is not the pinned official-200 run")
    model = build_vdn_model(source, image_size=384, imagenet_pretrained=False)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"VDN checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), signature


def _load_pepd(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    signature = checkpoint.get("signature") or {}
    if (
        signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
        or int(signature.get("seed", -1)) != 20260722
        or int(signature.get("image_size", -1)) != 256
    ):
        raise ValueError("PEPD checkpoint is not the frozen final seed-20260722 run")
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(signature["angle_bins"]),
        imagenet_pretrained=False,
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"PEPD checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), signature


def _autocast(device: torch.device) -> contextlib.AbstractContextManager:
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True)


def _build_stack(
    label: str,
    paths: Mapping[str, Path],
    source: Path,
    device: torch.device,
    inputs: dict[str, torch.Tensor],
) -> LoadedStack:
    meter = _load_yolo(paths["meter_detector"], device)
    reference = _load_yolo(paths["keypoint_detector"], device)
    components = {
        "meter_detector": Component(
            meter,
            lambda: meter(inputs["meter_detector"]),
            paths["meter_detector"],
            1,
            "1x3x640x640 full image",
        ),
        "reference_detector": Component(
            reference,
            lambda: reference(inputs["reference_detector"]),
            paths["keypoint_detector"],
            2,
            "1x3x640x640 frozen dial crop",
        ),
    }

    if label == "Original Transformer":
        segmenter = _load_u2net(paths["segmentation"], device)
        transformer = _load_transformer(paths["transformer"], device)
        tokens = transformer.texteconder.tokenize(list(map(str, range(101)))).to(device)
        components.update(
            {
                "pointer_segmentation": Component(
                    segmenter,
                    lambda: segmenter(inputs["rgb_256"]),
                    paths["segmentation"],
                    1,
                    "1x3x256x256 frozen dial crop",
                ),
                "original_transformer": Component(
                    transformer,
                    lambda: transformer(inputs["gray_256"], inputs["gray_256"], tokens),
                    paths["transformer"],
                    1,
                    "two 1x1x256x256 images + 101 fixed tokens",
                ),
            }
        )

        def branch() -> Any:
            segmentation = segmenter(inputs["rgb_256"])[0]
            pointer = segmentation.mul(2.0).sub(1.0)
            return transformer(pointer, inputs["gray_256"], tokens)

        def decode(output: Any) -> None:
            output[0].argmax(dim=-1)

    elif label == "VDN official-200":
        vdn, _ = _load_vdn(paths["vdn"], source, device)
        components["vdn_official_200"] = Component(
            vdn,
            lambda: vdn(inputs["vdn_384"]),
            paths["vdn"],
            1,
            "1x3x384x384 frozen dial crop",
        )

        def branch() -> Any:
            return vdn(inputs["vdn_384"])

        def decode(output: Any) -> None:
            predict_directions(*output)

    elif label == "PEPD-final":
        segmenter = _load_u2net(paths["segmentation"], device)
        pepd, _ = _load_pepd(paths["pepd"], device)
        components.update(
            {
                "pointer_segmentation": Component(
                    segmenter,
                    lambda: segmenter(inputs["rgb_256"]),
                    paths["segmentation"],
                    1,
                    "1x3x256x256 frozen dial crop",
                ),
                "pepd_direction": Component(
                    pepd,
                    lambda: pepd(inputs["rgb_256"]),
                    paths["pepd"],
                    1,
                    "1x3x256x256 frozen dial crop",
                ),
            }
        )

        def branch() -> Any:
            segmenter(inputs["rgb_256"])
            return pepd(inputs["rgb_256"])

        def decode(output: Any) -> None:
            decode_probabilistic_pivot_direction(*(value.float() for value in output))

    else:
        raise ValueError(f"unknown method {label}")

    def forward() -> Any:
        meter(inputs["meter_detector"])
        output = branch()
        reference(inputs["reference_detector"])
        reference(inputs["reference_detector"])
        return output

    return LoadedStack(label=label, components=components, forward=forward, decode=decode)


def _unique_parameters(models: Sequence[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for model in models:
        for parameter in model.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return int(total)


def _latency(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(statistics.fmean(values))
    return {
        "iterations": len(values),
        "mean_ms": mean,
        "std_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min_ms": float(array.min()),
        "median_ms": float(np.quantile(array, 0.50)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "max_ms": float(array.max()),
        "serial_throughput_images_per_second": 1000.0 / mean,
    }


def _component_metrics(stack: LoadedStack, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for name, component in stack.components.items():
            with FlopCounterMode(display=False) as counter:
                with _autocast(device):
                    output = component.call()
                torch.cuda.synchronize(device)
            del output
            parameters = _unique_parameters([component.model])
            per_call = int(counter.get_total_flops())
            result[name] = {
                "parameters": parameters,
                "flops_per_invocation": per_call,
                "invocations_per_sample": component.invocations,
                "flops_per_sample": per_call * component.invocations,
                "input": component.input_description,
                "checkpoint": str(component.checkpoint),
                "checkpoint_sha256": sha256_file(component.checkpoint),
                "checkpoint_bytes": component.checkpoint.stat().st_size,
            }
    return result


def _benchmark_method(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    paths: Mapping[str, Path],
    source: Path,
    device: torch.device,
    warmup: int,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    inputs = _prepare_inputs(rows[0], device)
    stack = _build_stack(label, paths, source, device, inputs)
    components = _component_metrics(stack, device)

    def run_one(row: Mapping[str, Any]) -> float:
        prepared = _prepare_inputs(row, device)
        inputs.clear()
        inputs.update(prepared)
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        with torch.inference_mode(), _autocast(device):
            output = stack.forward()
            stack.decode(output)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        del output
        return float(elapsed)

    for index in range(warmup):
        run_one(rows[index % len(rows)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = int(torch.cuda.memory_allocated(device))
    durations = [run_one(row) for row in rows]
    peak = int(torch.cuda.max_memory_allocated(device))
    models = [component.model for component in stack.components.values()]
    unique_parameters = _unique_parameters(models)
    result = {
        "label": label,
        "executed_samples": len(durations),
        "execution_success_rate": 1.0,
        "parameters": {"active_unique_total": unique_parameters},
        "flops_per_sample": int(
            sum(value["flops_per_sample"] for value in components.values())
        ),
        "components": components,
        "latency": _latency(durations),
        "memory": {
            "resident_allocated_bytes": resident,
            "peak_allocated_bytes": peak,
            "forward_incremental_peak_bytes": max(0, peak - resident),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    del stack, models, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _verify_freeze(
    freeze_path: Path,
    args: argparse.Namespace,
    roster_identity: Mapping[str, Any],
) -> dict[str, Any]:
    freeze = _read_json(freeze_path)
    if (
        freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("status") != "frozen_authorized_not_started"
        or freeze.get("benchmark_protocol") != PROTOCOL
    ):
        raise ValueError("resource-comparison freeze is absent or invalid")
    guard = freeze.get("restricted_data_guard") or {}
    if not guard or any(value is not False for value in guard.values()):
        raise ValueError("freeze does not prohibit all restricted data scopes")
    settings = freeze.get("settings") or {}
    expected_settings = {
        "batch_size": 1,
        "precision": "cuda_amp_fp16",
        "warmup_iterations": int(args.warmup),
        "timed_samples": 100,
        "device": str(args.device),
    }
    if settings != expected_settings:
        raise ValueError(f"runtime settings drifted from freeze: {settings}")
    bindings = freeze.get("bindings") or {}
    for name, binding in bindings.items():
        path = Path(str(binding.get("path"))).resolve()
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"freeze binding failed: {name}")
    if str(Path(str(freeze.get("output_directory"))).resolve()) != str(args.output_dir):
        raise ValueError("output directory drifted from freeze")
    roster_freeze = freeze.get("roster_identity") or {}
    for key, value in roster_identity.items():
        if roster_freeze.get(key) != value:
            raise ValueError(f"roster identity drifted: {key}")
    return freeze


def _validate_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "protocol": payload.get("protocol") == PROTOCOL,
        "complete": payload.get("status") == "complete",
        "train_only": payload.get("data_scope", {}).get("split") == "train",
        "roster_100": int(payload.get("data_scope", {}).get("samples", -1)) == 100,
        "method_order": tuple(item.get("label") for item in payload.get("methods", [])) == METHODS,
    }
    for method in payload.get("methods", []):
        label = str(method.get("label"))
        latency = method.get("latency") or {}
        checks[f"{label}_iterations"] = int(latency.get("iterations", -1)) == 100
        checks[f"{label}_positive_latency"] = all(
            math.isfinite(float(latency.get(key, math.nan))) and float(latency[key]) > 0
            for key in ("mean_ms", "median_ms", "p95_ms")
        )
        checks[f"{label}_quantiles"] = (
            float(latency.get("median_ms", math.inf))
            <= float(latency.get("p90_ms", -math.inf))
            <= float(latency.get("p95_ms", -math.inf))
            <= float(latency.get("max_ms", -math.inf))
        )
        checks[f"{label}_success"] = float(method.get("execution_success_rate", 0)) == 1.0
        component_parameters = sum(
            int(value["parameters"]) for value in (method.get("components") or {}).values()
        )
        checks[f"{label}_parameter_sum"] = component_parameters == int(
            method.get("parameters", {}).get("active_unique_total", -1)
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"resource summary verification failed: {failed}")
    return {"verified": True, "checks_passed": len(checks), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segmentation-weights", type=Path, default=DEFAULT_WEIGHTS["segmentation"])
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_WEIGHTS["meter_detector"])
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_WEIGHTS["keypoint_detector"])
    parser.add_argument("--transformer-weights", type=Path, default=DEFAULT_WEIGHTS["transformer"])
    parser.add_argument("--vdn-checkpoint", type=Path, default=DEFAULT_WEIGHTS["vdn"])
    parser.add_argument("--pepd-checkpoint", type=Path, default=DEFAULT_WEIGHTS["pepd"])
    parser.add_argument("--vdn-source", type=Path, default=DEFAULT_VDN_SOURCE)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 1:
        raise ValueError("warmup must be positive")
    for name in ("roster", "manifest", "freeze"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    paths = {
        "segmentation": args.segmentation_weights.resolve(),
        "meter_detector": args.meter_detector_weights.resolve(),
        "keypoint_detector": args.keypoint_detector_weights.resolve(),
        "transformer": args.transformer_weights.resolve(),
        "vdn": args.vdn_checkpoint.resolve(),
        "pepd": args.pepd_checkpoint.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.vdn_source = args.vdn_source.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.vdn_source.is_dir():
        raise FileNotFoundError(args.vdn_source)
    rows, roster_identity = _load_roster_rows(args.roster, args.manifest)
    freeze = _verify_freeze(args.freeze, args, roster_identity)

    summary_path = args.output_dir / "summary.json"
    verification_path = args.output_dir / "verification.json"
    if args.verify_only:
        summary = _read_json(summary_path)
        verification = _read_json(verification_path)
        fresh = _validate_summary(summary)
        if (
            verification.get("verified") is not True
            or verification.get("summary_sha256") != sha256_file(summary_path)
            or verification.get("freeze_sha256") != sha256_file(args.freeze)
            or verification.get("checks_passed") != fresh.get("checks_passed")
        ):
            raise ValueError("stored verification identity failed")
        print(json.dumps(verification, indent=2, sort_keys=True))
        return

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal resource comparison requires CUDA")
    torch.manual_seed(20260804)
    torch.cuda.manual_seed_all(20260804)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    methods = [
        _benchmark_method(label, rows, paths, args.vdn_source, device, args.warmup)
        for label in METHODS
    ]
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "protocol": PROTOCOL,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze": str(args.freeze),
        "freeze_sha256": sha256_file(args.freeze),
        "data_scope": {
            "dataset": "SyncG",
            "split": "train",
            "samples": 100,
            "unique_groups": roster_identity["unique_groups"],
            "roster": str(args.roster),
            **roster_identity,
            "restricted_data_used": False,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "measurement": {
            "batch_size": 1,
            "precision": "CUDA AMP fp16 with fp32 method decode",
            "warmup_iterations_per_method": args.warmup,
            "timed_samples_per_method": 100,
            "included": (
                "one raw meter-detector forward, method-specific neural branch, "
                "two raw reference-detector forwards, and method decode"
            ),
            "excluded": (
                "image decode, frozen crop construction, resize/normalization, "
                "host-to-device transfer, YOLO NMS/post-processing, geometric "
                "reading conversion, rendering, and serialization"
            ),
            "comparison_role": "controlled active-stack compute/resource table; not accuracy or API latency",
            "flop_convention": "multiply-add counts as two FLOPs",
        },
        "weights": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "methods": methods,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_authorized_command": freeze.get("authorized_command"),
    }
    validation = _validate_summary(payload)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(summary_path, payload)
    verification = {
        **validation,
        "protocol": f"{PROTOCOL}_verification_v1",
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "freeze": str(args.freeze),
        "freeze_sha256": sha256_file(args.freeze),
        "roster_sha256": roster_identity["roster_sha256"],
        "image_content_inventory_sha256": roster_identity["image_content_inventory_sha256"],
    }
    _write_json(verification_path, verification)
    print(json.dumps({"summary": str(summary_path), "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
