"""Reproducible batch-one resource benchmark for the frozen PEPD-only stack.

The input scope is deliberately restricted to a deterministic roster sampled
from the official SyncG *training* manifest.  The measured active stack is:

    decoded BGR image -> meter detector -> pointer segmenter -> reference-point
    detector -> frozen PEPD direction expert -> scalar reading

The original Transformer and the FADR calibrator/router are not loaded.  Model
construction/checkpoint I/O are reported as cold start and excluded from warm
latency.  The reference-point detector is reused for both semantic endpoints,
so its parameters are counted once even though production invokes it twice.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from loguru import logger


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANGLE_DIR = PROJECT_DIR / "utils" / "angleDetect"
if str(ANGLE_DIR) not in sys.path:
    sys.path.insert(0, str(ANGLE_DIR))

from pointerSeg.detectSeg import u2netpSeg  # noqa: E402
from yoloDetection.yoloDectect import targetDetectModel  # noqa: E402
from zeroShotMeter import meterZeroShot  # noqa: E402

from experiments.pivot_direction_fallback import tensor_from_bbox  # noqa: E402
from experiments.probabilistic_pivot_direction import (  # noqa: E402
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import sha256_file  # noqa: E402
from utils.angleDetect.reference_conditioned_runtime import (  # noqa: E402
    ReferenceConditionedFinalBackend,
    build_front_end_payload,
)


PROTOCOL = "pepd_only_syncg_train_e2e_resource_benchmark_v1"
ROSTER_PROTOCOL = "pepd_only_syncg_train_resource_roster_v1"
EXPECTED_MANIFEST_PROTOCOL = "syncg_official_split_v1"
DEFAULT_MANIFEST = PROJECT_DIR / "artifacts" / "manifests" / "syncg_train.jsonl"
DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "artifacts"
    / "runs"
    / "pepd_convergence_phase2"
    / "seed_20260722"
    / "best.pt"
)
DEFAULT_VERIFICATION = DEFAULT_CHECKPOINT.with_name("verification.json")
DEFAULT_SUMMARY = DEFAULT_CHECKPOINT.with_name("summary.json")
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "artifacts" / "runs" / "pepd_e2e_resources_v1"
)
DEFAULT_WEIGHTS = {
    "segmentation": ANGLE_DIR / "pointerSeg" / "resultSeg" / "best.pt",
    "meter_detector": (
        ANGLE_DIR / "yoloDetection" / "result" / "yolo_findMeter.pt"
    ),
    "keypoint_detector": (
        ANGLE_DIR / "yoloDetection" / "result" / "yolo_pointbest.pt"
    ),
}


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


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_suffix(manifest.suffix + ".protocol.json")


def _stable_score(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _row_identity(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_roster(
    manifest: Path,
    *,
    size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = _protocol_path(manifest)
    protocol = _read_json(protocol_path)
    if (
        protocol.get("protocol") != EXPECTED_MANIFEST_PROTOCOL
        or protocol.get("dataset") != "SyncG"
        or protocol.get("split") != "train"
        or protocol.get("strict_release") is not True
        or protocol.get("release_identity_verified") is not True
    ):
        raise ValueError("benchmark requires the verified official SyncG/train manifest")
    rows = _read_jsonl(manifest)
    if len(rows) != int(protocol.get("emitted_rows", -1)):
        raise ValueError("SyncG/train manifest row count disagrees with its protocol")
    if not 1 <= size <= len(rows):
        raise ValueError(f"roster size must be in [1, {len(rows)}]")
    identities: set[str] = set()
    for row in rows:
        if row.get("dataset") != "SyncG" or row.get("split") != "train":
            raise ValueError("non-SyncG/train row encountered in benchmark manifest")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in identities:
            raise ValueError("manifest has a blank or duplicate sample_id")
        identities.add(sample_id)
    selected = sorted(
        rows,
        key=lambda row: (
            _stable_score(str(row["sample_id"]), seed),
            str(row["sample_id"]),
        ),
    )[:size]
    entries = [
        {
            "sample_id": str(row["sample_id"]),
            "group_id": str(row.get("group_id") or ""),
            "image_path": str(Path(str(row["image_path"])).resolve()),
            "scale_start": float(row["scale_start"]),
            "scale_end": float(row["scale_end"]),
            "selection_score_sha256": _stable_score(str(row["sample_id"]), seed),
            "manifest_row_sha256": _row_identity(row),
        }
        for row in selected
    ]
    roster = {
        "protocol": ROSTER_PROTOCOL,
        "status": "frozen",
        "scope": "official verified SyncG/train only; no public/test/field/sealed/confirmatory input",
        "selection": "lowest SHA-256(seed:sample_id), then sample_id",
        "selection_seed": int(seed),
        "samples": len(entries),
        "unique_groups": len({entry["group_id"] for entry in entries}),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol": str(protocol_path.resolve()),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "entries": entries,
    }
    return selected, roster


def _decode_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("empty image file")
    image = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None or image.size == 0:
        raise ValueError("OpenCV image decode failed")
    return image


def _module_parameter_count(module: torch.nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in module.parameters())),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        ),
    }


def _unique_parameter_count(modules: Iterable[torch.nn.Module]) -> dict[str, int]:
    seen: set[int] = set()
    total = 0
    trainable = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel()
            if parameter.requires_grad:
                trainable += parameter.numel()
    return {"total": int(total), "trainable": int(trainable)}


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _latency_summary(milliseconds: Sequence[float]) -> dict[str, Any]:
    if not milliseconds:
        raise ValueError("cannot summarize an empty latency sample")
    mean = float(statistics.fmean(milliseconds))
    return {
        "iterations": len(milliseconds),
        "mean_ms": mean,
        "std_ms": (
            float(statistics.stdev(milliseconds))
            if len(milliseconds) > 1
            else 0.0
        ),
        "min_ms": float(min(milliseconds)),
        "median_ms": float(statistics.median(milliseconds)),
        "p90_ms": _percentile(milliseconds, 0.90),
        "p95_ms": _percentile(milliseconds, 0.95),
        "max_ms": float(max(milliseconds)),
        "serial_throughput_images_per_second": 1000.0 / mean,
    }


class PEPDOnlyRuntime:
    """Minimal production-method adapter using the audited direction decoder."""

    predict_direction = ReferenceConditionedFinalBackend.predict_direction

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        verification_path: Path,
        summary_path: Path,
        device: torch.device,
    ) -> None:
        checkpoint_hash = sha256_file(checkpoint_path)
        verification = _read_json(verification_path)
        summary = _read_json(summary_path)
        if verification.get("verified") is not True:
            raise ValueError("PEPD convergence checkpoint verification did not pass")
        if (
            verification.get("best_checkpoint_sha256") != checkpoint_hash
            or summary.get("best_checkpoint_sha256") != checkpoint_hash
            or sha256_file(summary_path) != verification.get("summary_sha256")
        ):
            raise ValueError("PEPD checkpoint/summary/verification identity mismatch")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        signature = checkpoint.get("signature") or {}
        if signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
            raise ValueError("checkpoint is not a formal PEPD model")
        if int(signature.get("seed", -1)) != 20260722:
            raise ValueError("benchmark is frozen to PEPD seed 20260722")
        self._lock = threading.RLock()
        self.device = device
        self.image_size = int(signature["image_size"])
        self.heatmap_size = int(signature["heatmap_size"])
        self.expansion = float(signature["expansion"])
        self.direction_model = build_probabilistic_pivot_direction_model(
            angle_bins=int(signature["angle_bins"]),
            imagenet_pretrained=False,
        )
        incompatible = self.direction_model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"PEPD checkpoint architecture mismatch: {incompatible}")
        self.direction_model.to(device).eval()
        self.amp_enabled = device.type == "cuda"
        self.audit = {
            "protocol": PROTOCOL,
            "method": "PEPD-only raw probabilistic vector",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_seed": int(signature["seed"]),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "verification": str(verification_path.resolve()),
            "verification_sha256": sha256_file(verification_path),
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "training_protocol": signature["protocol"],
            "image_size": self.image_size,
            "heatmap_size": self.heatmap_size,
            "expansion": self.expansion,
            "device": str(device),
            "amp_enabled": self.amp_enabled,
            "calibration_loaded": False,
            "router_loaded": False,
            "vdn_loaded": False,
            "transformer_loaded": False,
        }


def _build_active_meter(
    *,
    segmentation: Path,
    meter_detector: Path,
    keypoint_detector: Path,
    device: torch.device,
) -> meterZeroShot:
    """Construct the production inference class without its unused Transformer."""

    meter = meterZeroShot.__new__(meterZeroShot)
    meter.device = device
    meter.pointerSeg = u2netpSeg(str(segmentation), device)
    meter.meterDetect = targetDetectModel(str(meter_detector))
    meter.pointerDetect = targetDetectModel(str(keypoint_detector))
    meter.meterDetect.fire_detection_model.to(str(device))
    meter.pointerDetect.fire_detection_model.to(str(device))
    meter.vlmMeter = None
    meter.label_text_list = list(map(str, range(101)))
    meter.label_text = None
    meter.ornMeterNum = 155.25
    meter.oneTempAngle = 360.0 / 101.0
    meter.last_error_code = None
    meter.last_error_message = None
    meter._correction_debug = {}
    meter._last_reading_details = {}
    meter._last_training_artifacts = {}
    meter._residual_calibrator_cache = {}
    meter._residual_hybrid_cache = {}
    return meter


def _infer_one(
    meter: meterZeroShot,
    backend: PEPDOnlyRuntime,
    image: np.ndarray,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    _, result, _, crop, _ = meter.Inference(
        image,
        float(row["scale_start"]),
        float(row["scale_end"]),
        confidence=None,
        use_origin_when_no_meter=False,
        start_end_distance_threshold=None,
        start_end_position="start_left_end_right",
        snap_out_of_range_pointer=True,
        correction_mode="off",
        stretch_x_ratio=1.0,
        stretch_y_ratio=1.0,
        reading_offset=0.0,
        default_start_angle=45.0,
        default_range_angle=270.0,
        validate_mask_line=True,
        mask_center_threshold_ratio=0.10,
        reading_backend="probabilistic_vector",
        geometry_fallback_to_transformer=False,
        residual_calibrator_path="",
        reference_conditioned_backend=backend,
    )
    details = meter._last_reading_details or {}
    selected = details.get("probabilistic_vector") or {}
    success = bool(
        crop is not None
        and selected.get("status") is True
        and result is not None
        and math.isfinite(float(result))
    )
    error_code = None if success else (
        selected.get("error_code")
        or meter.last_error_code
        or "unknown_failure"
    )
    front_end = build_front_end_payload(meter._last_training_artifacts or {})
    return {
        "success": success,
        "error_code": error_code,
        "front_end": front_end,
    }


def _time_call(device: torch.device, call) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = call()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return result, float(elapsed_ms)


def _direction_only_benchmark(
    backend: PEPDOnlyRuntime,
    tensors: Sequence[torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if not tensors:
        raise ValueError("no valid PEPD crop tensors were collected")
    device = backend.device

    def forward(tensor: torch.Tensor) -> None:
        value = tensor.unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            device.type,
            enabled=backend.amp_enabled,
        ):
            outputs = backend.direction_model(value)
        decode_probabilistic_pivot_direction(
            *(output.float() for output in outputs)
        )

    for index in range(warmup):
        forward(tensors[index % len(tensors)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = int(torch.cuda.memory_allocated(device))
    durations: list[float] = []
    for index in range(iterations):
        _, elapsed = _time_call(
            device,
            lambda index=index: forward(tensors[index % len(tensors)]),
        )
        durations.append(elapsed)
    peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "scope": (
            "preprocessed 1x3x256x256 tensor transfer, PEPD forward and "
            "probabilistic decode; crop construction excluded"
        ),
        "batch_size": 1,
        "precision": "CUDA AMP fp16 with fp32 decode",
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "unique_train_crops": len(tensors),
        "latency": _latency_summary(durations),
        "memory": {
            "resident_allocated_bytes": resident,
            "peak_allocated_bytes": peak,
            "forward_incremental_peak_bytes": max(0, peak - resident),
        },
    }


def _verify_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["protocol"] = payload.get("protocol") == PROTOCOL
    checks["complete"] = payload.get("status") == "complete"
    checks["train_only"] = payload.get("data_scope", {}).get("split") == "train"
    checks["roster_complete"] = (
        int(payload["data_scope"]["roster_samples"])
        == int(payload["warm_benchmark"]["timed_samples"])
    )
    parameters = payload["parameters"]
    component_total = sum(
        int(value["total"])
        for value in parameters["components"].values()
    )
    checks["parameter_sum"] = (
        component_total == int(parameters["active_unique_total"]["total"])
    )
    checks["shared_plus_pepd"] = (
        int(parameters["shared_front_end_unique_total"]["total"])
        + int(parameters["components"]["pepd_direction"]["total"])
        == int(parameters["active_unique_total"]["total"])
    )
    for name in ("decoded_bgr_to_scalar", "filesystem_decode_to_scalar"):
        latency = payload["warm_benchmark"][name]["latency"]
        checks[f"{name}_positive"] = all(
            float(latency[key]) > 0.0
            for key in ("mean_ms", "median_ms", "p90_ms", "p95_ms")
        )
        checks[f"{name}_quantiles"] = (
            float(latency["median_ms"])
            <= float(latency["p90_ms"])
            <= float(latency["p95_ms"])
            <= float(latency["max_ms"])
        )
    checks["finite_failure_rate"] = math.isfinite(
        float(payload["warm_benchmark"]["failure_rate"])
    )
    checks["pepd_only"] = all(
        payload["method_identity"].get(key) is False
        for key in (
            "transformer_loaded",
            "fadr_calibration_loaded",
            "fadr_router_loaded",
            "vdn_loaded",
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"resource benchmark verification failed: {failed}")
    return {"verified": True, "checks": checks, "check_count": len(checks)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--segmentation-weights", type=Path, default=DEFAULT_WEIGHTS["segmentation"])
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_WEIGHTS["meter_detector"])
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_WEIGHTS["keypoint_detector"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--roster-size", type=int, default=100)
    parser.add_argument("--roster-seed", type=int, default=20260804)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--direction-warmup", type=int, default=30)
    parser.add_argument("--direction-iterations", type=int, default=200)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "summary.json"
    verification_output = output_dir / "verification.json"
    if args.verify_only:
        payload = _read_json(summary_path)
        verification = _verify_payload(payload)
        verification["summary_sha256"] = sha256_file(summary_path)
        _write_json(verification_output, verification)
        print(json.dumps(verification, indent=2, sort_keys=True))
        return
    if args.roster_size < 20 or args.warmup < 1:
        raise ValueError("formal benchmark requires roster-size >= 20 and warmup >= 1")
    if args.direction_iterations < 20 or args.direction_warmup < 1:
        raise ValueError("direction benchmark requires at least 20 timed iterations")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal warm GPU benchmark requires CUDA")
    torch.cuda.set_device(device)
    # Materialize the CUDA context before resetting allocator peak counters.
    torch.empty(1, device=device)

    paths = {
        "manifest": args.manifest.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "checkpoint_verification": args.verification.resolve(),
        "checkpoint_summary": args.training_summary.resolve(),
        "segmentation": args.segmentation_weights.resolve(),
        "meter_detector": args.meter_detector_weights.resolve(),
        "keypoint_detector": args.keypoint_detector_weights.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark inputs: {missing}")

    torch.manual_seed(args.roster_seed)
    torch.cuda.manual_seed_all(args.roster_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    logger.remove()

    rows, roster = _build_roster(
        paths["manifest"],
        size=args.roster_size,
        seed=args.roster_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = output_dir / "roster.json"
    _write_json(roster_path, roster)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cold_started = time.perf_counter_ns()
    meter = _build_active_meter(
        segmentation=paths["segmentation"],
        meter_detector=paths["meter_detector"],
        keypoint_detector=paths["keypoint_detector"],
        device=device,
    )
    backend = PEPDOnlyRuntime(
        checkpoint_path=paths["checkpoint"],
        verification_path=paths["checkpoint_verification"],
        summary_path=paths["checkpoint_summary"],
        device=device,
    )
    torch.cuda.synchronize(device)
    cold_start_ms = (time.perf_counter_ns() - cold_started) / 1_000_000.0
    cold_peak = int(torch.cuda.max_memory_allocated(device))

    components = {
        "meter_detector": meter.meterDetect.fire_detection_model.model,
        "pointer_segmentation": meter.pointerSeg.model,
        "reference_keypoint_detector": (
            meter.pointerDetect.fire_detection_model.model
        ),
        "pepd_direction": backend.direction_model,
    }
    component_parameters = {
        name: {
            **_module_parameter_count(module),
            "checkpoint": str(
                paths[
                    {
                        "meter_detector": "meter_detector",
                        "pointer_segmentation": "segmentation",
                        "reference_keypoint_detector": "keypoint_detector",
                        "pepd_direction": "checkpoint",
                    }[name]
                ]
            ),
            "checkpoint_sha256": sha256_file(
                paths[
                    {
                        "meter_detector": "meter_detector",
                        "pointer_segmentation": "segmentation",
                        "reference_keypoint_detector": "keypoint_detector",
                        "pepd_direction": "checkpoint",
                    }[name]
                ]
            ),
            "invocations_per_sample": (
                2 if name == "reference_keypoint_detector" else 1
            ),
        }
        for name, module in components.items()
    }
    shared_modules = [
        components["meter_detector"],
        components["pointer_segmentation"],
        components["reference_keypoint_detector"],
    ]

    decoded_images = [
        _decode_image(Path(str(row["image_path"])))
        for row in rows
    ]
    for index in range(args.warmup):
        row_index = index % len(rows)
        _infer_one(
            meter,
            backend,
            decoded_images[row_index].copy(),
            rows[row_index],
        )
    torch.cuda.synchronize(device)
    resident_allocated = int(torch.cuda.memory_allocated(device))
    resident_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    decoded_durations: list[float] = []
    sample_results: list[dict[str, Any]] = []
    crop_tensors: list[torch.Tensor] = []
    for row, image in zip(rows, decoded_images):
        result, elapsed = _time_call(
            device,
            lambda row=row, image=image: _infer_one(
                meter,
                backend,
                image.copy(),
                row,
            ),
        )
        decoded_durations.append(elapsed)
        sample_results.append(
            {
                "sample_id": str(row["sample_id"]),
                "success": bool(result["success"]),
                "error_code": result["error_code"],
                "decoded_bgr_to_scalar_ms": elapsed,
            }
        )
        front_end = result["front_end"]
        if front_end.get("status") is True:
            crop_tensors.append(
                tensor_from_bbox(
                    image,
                    front_end["meter_bbox"],
                    image_size=backend.image_size,
                    expansion=backend.expansion,
                )
            )
    pipeline_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    pipeline_peak_reserved = int(torch.cuda.max_memory_reserved(device))

    filesystem_durations: list[float] = []
    filesystem_successes: list[bool] = []
    for row in rows:
        path = Path(str(row["image_path"]))

        def decode_and_infer(row=row, path=path):
            image = _decode_image(path)
            return _infer_one(meter, backend, image, row)

        result, elapsed = _time_call(device, decode_and_infer)
        filesystem_durations.append(elapsed)
        filesystem_successes.append(bool(result["success"]))

    failures = [result for result in sample_results if not result["success"]]
    failure_counts = Counter(
        str(result["error_code"] or "unknown_failure") for result in failures
    )
    direction_metrics = _direction_only_benchmark(
        backend,
        crop_tensors,
        warmup=args.direction_warmup,
        iterations=args.direction_iterations,
    )
    # Ultralytics fuses convolution/batch-normalization layers on first predict.
    # Count the warmed, deployed modules after that in-place transformation so
    # component totals and the cross-component unique total share one state.
    component_parameters = {
        name: {
            **_module_parameter_count(module),
            "checkpoint": str(
                paths[
                    {
                        "meter_detector": "meter_detector",
                        "pointer_segmentation": "segmentation",
                        "reference_keypoint_detector": "keypoint_detector",
                        "pepd_direction": "checkpoint",
                    }[name]
                ]
            ),
            "checkpoint_sha256": sha256_file(
                paths[
                    {
                        "meter_detector": "meter_detector",
                        "pointer_segmentation": "segmentation",
                        "reference_keypoint_detector": "keypoint_detector",
                        "pepd_direction": "checkpoint",
                    }[name]
                ]
            ),
            "invocations_per_sample": (
                2 if name == "reference_keypoint_detector" else 1
            ),
        }
        for name, module in components.items()
    }
    properties = torch.cuda.get_device_properties(device)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "resource-only inference benchmark; no training, tuning, public, "
            "test, field, sealed, or confirmatory data"
        ),
        "method_identity": {
            "name": "PEPD-only",
            "checkpoint_seed": 20260722,
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "shared_front_end_weights_sha256": {
                name: sha256_file(paths[name])
                for name in (
                    "segmentation",
                    "meter_detector",
                    "keypoint_detector",
                )
            },
            "transformer_loaded": False,
            "fadr_calibration_loaded": False,
            "fadr_router_loaded": False,
            "vdn_loaded": False,
        },
        "data_scope": {
            "dataset": "SyncG",
            "split": "train",
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "roster": str(roster_path.resolve()),
            "roster_sha256": sha256_file(roster_path),
            "roster_samples": len(rows),
            "unique_groups": roster["unique_groups"],
            "selection_seed": args.roster_seed,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_enabled": False,
        },
        "parameters": {
            "counting_policy": (
                "unique torch Parameter object identities; the keypoint detector "
                "is invoked twice but counted once"
            ),
            "components": component_parameters,
            "shared_front_end_unique_total": _unique_parameter_count(shared_modules),
            "active_unique_total": _unique_parameter_count(components.values()),
            "inactive_transformer_parameters_counted": False,
            "non_neural_calibrator_router_parameters": 0,
        },
        "cold_start": {
            "scope": (
                "construct active shared front end and load/authenticate PEPD "
                "checkpoint through first CUDA synchronization; no inference"
            ),
            "milliseconds": float(cold_start_ms),
            "peak_cuda_allocated_bytes": cold_peak,
        },
        "warm_benchmark": {
            "batch_size": 1,
            "precision": "shared front end native precision; PEPD CUDA AMP fp16",
            "warmup_iterations": args.warmup,
            "timed_samples": len(rows),
            "repeats_per_roster_sample_per_scope": 1,
            "decoded_bgr_to_scalar": {
                "scope": (
                    "decoded BGR ndarray through meter detector, pointer "
                    "segmentation, two reference detector calls, PEPD and scalar decode"
                ),
                "latency": _latency_summary(decoded_durations),
            },
            "filesystem_decode_to_scalar": {
                "scope": (
                    "filesystem read and OpenCV decode plus decoded-BGR pipeline; "
                    "OS file cache is warm after roster preload"
                ),
                "latency": _latency_summary(filesystem_durations),
            },
            "successes": len(rows) - len(failures),
            "failures": len(failures),
            "failure_rate": len(failures) / len(rows),
            "failure_counts": dict(sorted(failure_counts.items())),
            "filesystem_scope_successes": int(sum(filesystem_successes)),
            "filesystem_scope_failures": int(
                len(filesystem_successes) - sum(filesystem_successes)
            ),
            "memory": {
                "resident_allocated_bytes": resident_allocated,
                "resident_reserved_bytes": resident_reserved,
                "peak_allocated_bytes": pipeline_peak_allocated,
                "peak_reserved_bytes": pipeline_peak_reserved,
                "forward_incremental_peak_bytes": max(
                    0,
                    pipeline_peak_allocated - resident_allocated,
                ),
            },
        },
        "pepd_direction_only": direction_metrics,
        "sample_measurements": sample_results,
        "source_sha256": {
            "benchmark": sha256_file(Path(__file__).resolve()),
            "zero_shot_meter": sha256_file(ANGLE_DIR / "zeroShotMeter.py"),
            "pepd_model": sha256_file(
                PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
            ),
            "production_runtime": sha256_file(
                ANGLE_DIR / "reference_conditioned_runtime.py"
            ),
        },
    }
    payload["verification"] = _verify_payload(payload)
    _write_json(summary_path, payload)
    standalone_verification = dict(payload["verification"])
    standalone_verification.update(
        protocol="pepd_only_syncg_train_e2e_resource_verification_v1",
        summary=str(summary_path.resolve()),
        summary_sha256=sha256_file(summary_path),
    )
    _write_json(verification_output, standalone_verification)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": str(summary_path.resolve()),
                "verification": str(verification_output.resolve()),
                "parameters": payload["parameters"],
                "cold_start": payload["cold_start"],
                "warm_benchmark": payload["warm_benchmark"],
                "pepd_direction_only": payload["pepd_direction_only"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
