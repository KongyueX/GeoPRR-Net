"""Evaluate a SyncG-retrained official VDN as an end-to-end reading baseline.

VDN replaces only pointer direction estimation.  Meter detection, start/end
references, scale conversion, failure handling, and controlled degradations
follow the same frozen protocol used by the project's paper-facing methods.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from experiments.robustness_degradations import (
    ROBUSTNESS_PROTOCOL,
    apply_degradation,
    degradation_names,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PROTOCOL,
    build_vdn_model,
    image_angle_from_direction,
    normalize_reference_points,
    predict_directions,
    reading_from_pointer_angle,
    reference_angles,
    sha256_file,
    summarize_scalar_predictions,
    vdn_tensor_from_bbox,
    verify_vdn_source,
)
from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel


EVALUATION_PROTOCOL = "vdn_syncg_external_baseline_e2e_v1"
DEFAULT_METER_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_findMeter.pt"
)
DEFAULT_POINT_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_pointbest.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=Path("artifacts/vendor/VectorDetectionNetwork"),
    )
    parser.add_argument("--shared-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--condition", choices=degradation_names(), default="clean")
    parser.add_argument("--degradation-seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_METER_WEIGHTS)
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_POINT_WEIGHTS)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _summary_path(output: Path) -> Path:
    return output.with_name(output.stem + ".summary.json")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_shared_predictions(
    path: Path | None,
    *,
    manifest: Path,
    rows: Sequence[dict[str, Any]],
    condition: str,
    degradation_seed: int,
    meter_weights: Path,
    point_weights: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    metadata_path = _metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"shared predictions or metadata missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("shared predictions use a different manifest")
    protocol_path = _protocol_path(manifest)
    expected_manifest_protocol = (
        sha256_file(protocol_path) if protocol_path.is_file() else None
    )
    if signature.get("manifest_protocol_sha256") != expected_manifest_protocol:
        raise ValueError("shared predictions use a different manifest protocol")
    if signature.get("correction_mode") != "off":
        raise ValueError("VDN comparison requires correction_mode=off shared references")
    degradation = signature.get("input_degradation") or {}
    expected_degradation_source = sha256_file(
        PROJECT_DIR / "experiments" / "robustness_degradations.py"
    )
    recorded_degradation_source = signature.get("input_degradation_source_sha256")
    if degradation:
        if degradation.get("protocol") != ROBUSTNESS_PROTOCOL:
            raise ValueError("shared predictions use a different degradation protocol")
        if recorded_degradation_source != expected_degradation_source:
            raise ValueError("shared predictions use different degradation source code")
    elif condition != "clean" or recorded_degradation_source is not None:
        raise ValueError("only clean legacy caches may omit the degradation signature")
    cached_condition = degradation.get("condition", "clean")
    if cached_condition != condition:
        raise ValueError(
            f"shared prediction condition is {cached_condition}, expected {condition}"
        )
    cached_seed = int(degradation.get("seed", degradation_seed))
    if condition != "clean" and cached_seed != degradation_seed:
        raise ValueError("shared predictions use a different degradation seed")
    weight_hashes = signature.get("weights_sha256") or {}
    expected_hashes = {
        "meter_detector": sha256_file(meter_weights),
        "keypoint_detector": sha256_file(point_weights),
    }
    for name, expected in expected_hashes.items():
        if weight_hashes.get(name) != expected:
            raise ValueError(f"shared predictions use different {name} weights")

    shared_rows = _read_jsonl(path)
    by_id = {str(row.get("sample_id")): row for row in shared_rows}
    if len(by_id) != len(shared_rows):
        raise ValueError("shared predictions contain duplicate sample identifiers")
    missing = [str(row.get("sample_id")) for row in rows if str(row.get("sample_id")) not in by_id]
    if missing:
        raise ValueError(f"shared predictions miss {len(missing)} selected samples")
    return by_id, metadata


def _shared_reference(
    row: dict[str, Any] | None,
) -> tuple[float, float, str] | None:
    if not row:
        return None
    features = row.get("features") or {}
    start_angle = _finite_float(features.get("startAngle"))
    range_angle = _finite_float(features.get("disAngle"))
    if start_angle is None or range_angle is None or abs(range_angle) <= 1e-8:
        return None
    return start_angle, range_angle, str(row.get("branch") or "shared_unknown")


def _fallback_reference(
    point_detector: targetDetectModel,
    crop: np.ndarray,
) -> tuple[float, float, str]:
    # Match zeroShotMeter exactly: its crop is converted BGR->RGB before PIL,
    # then the returned RGB array is passed to OpenCV using BGR2GRAY.
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    detector_input = np.stack((gray,) * 3, axis=-1)
    center_end, _ = point_detector.center_find(detector_input, classId=1)
    center_start, _ = point_detector.center_find(detector_input, classId=2)
    center_start, center_end = normalize_reference_points(
        center_start,
        center_end,
        image_width=crop.shape[1],
    )
    return reference_angles(crop.shape, center_start, center_end)


def _xyxy_from_detector_box(box: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(box, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"unexpected meter box shape: {array.shape}")
    x1, y1 = map(float, array[0])
    x2, y2 = map(float, array[2])
    if x2 <= x1 or y2 <= y1:
        raise ValueError("meter detector returned an empty box")
    return x1, y1, x2, y2


def _pointer_points(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    for item in metadata.get("keypoints") or []:
        if str(item.get("type") or "").strip().lower() != "pointer":
            continue
        tip = item.get("outside_kp")
        tail = item.get("origin_kp")
        if isinstance(tip, Sequence) and isinstance(tail, Sequence):
            return (
                np.asarray(tip[:2], dtype=np.float64),
                np.asarray(tail[:2], dtype=np.float64),
            )
    return None


def _target_direction(
    row: dict[str, Any],
    degradation: dict[str, Any],
) -> np.ndarray | None:
    points = _pointer_points(row.get("metadata") or {})
    if points is None:
        return None
    tip, tail = points
    perspective = degradation.get("perspective") or {}
    homography = perspective.get("homography")
    if homography is not None:
        stacked = np.asarray([[tail, tip]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(
            stacked,
            np.asarray(homography, dtype=np.float64),
        )[0]
        tail, tip = transformed[0], transformed[1]
    direction = np.asarray(tip, dtype=np.float64) - np.asarray(tail, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-8 else None


def _direction_error(predicted: np.ndarray, target: np.ndarray | None) -> float | None:
    if target is None:
        return None
    cosine = float(np.clip(np.dot(predicted, target), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _base_result(row: dict[str, Any], degradation: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("sample_id")),
        "group_id": str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")),
        "meter_id": row.get("meter_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "image_path": row.get("image_path"),
        "ground_truth": float(row["ground_truth"]),
        "scale_start": float(row["scale_start"]),
        "scale_end": float(row["scale_end"]),
        "metadata": row.get("metadata") or {},
        "degradation": degradation,
        "status": False,
        "prediction": None,
        "progress": None,
        "pointer_angle": None,
        "direction": None,
        "direction_angle_error_degrees": None,
        "heatmap_peak": None,
        "error_code": None,
        "error_message": None,
    }


def _failure_result(
    row: dict[str, Any],
    degradation: dict[str, Any],
    *,
    code: str,
    message: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    result = _base_result(row, degradation)
    result.update(
        {
            "error_code": code,
            "error_message": message,
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return result


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _component_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    available = [
        float(row["direction_angle_error_degrees"])
        for row in rows
        if row.get("direction_angle_error_degrees") is not None
    ]
    eligible = sum(_pointer_points(row.get("metadata") or {}) is not None for row in rows)
    if eligible == 0:
        return None
    errors = np.asarray(available, dtype=np.float64)
    return {
        "eligible_samples": eligible,
        "successful_directions": len(available),
        "coverage": len(available) / eligible,
        "angle_mae_degrees_success_only": float(np.mean(errors)) if len(errors) else None,
        "angle_median_degrees_success_only": float(np.median(errors)) if len(errors) else None,
        "angle_acc_1deg_all": float(np.sum(errors <= 1.0) / eligible),
        "angle_acc_3deg_all": float(np.sum(errors <= 3.0) / eligible),
        "angle_acc_5deg_all": float(np.sum(errors <= 5.0) / eligible),
    }


def _dialbench_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if abs(float(row["scale_end"]) - float(row["scale_start"])) > 1e-12
    ]
    successful = [row for row in eligible if row.get("prediction") is not None]
    reference_errors = [
        abs(float(row["prediction"]) - float(row["ground_truth"]))
        / abs(float(row["scale_end"]) - float(row["scale_start"]))
        for row in successful
    ]
    relative_eligible = [
        row for row in eligible if abs(float(row["ground_truth"])) > 1e-12
    ]
    relative_successful = [
        row for row in relative_eligible if row.get("prediction") is not None
    ]
    relative_errors = [
        abs(float(row["prediction"]) - float(row["ground_truth"]))
        / abs(float(row["ground_truth"]))
        for row in relative_successful
    ]
    return {
        "eligible_samples": len(eligible),
        "successful_samples": len(successful),
        "ref_successful": (
            float(np.mean(reference_errors)) if reference_errors else None
        ),
        "relative_samples": len(relative_eligible),
        "relative_successful": len(relative_successful),
        "rel_successful": (
            float(np.mean(relative_errors)) if relative_errors else None
        ),
        "acc_epsilon_ref_le_1pct_e2e": (
            sum(error <= 0.01 for error in reference_errors) / len(eligible)
            if eligible
            else None
        ),
        "acc_theta_rel_lt_5pct_e2e": (
            sum(error < 0.05 for error in relative_errors) / len(relative_eligible)
            if relative_eligible
            else None
        ),
    }


def _subgroup_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    memberships: dict[str, dict[str, list[dict[str, Any]]]] = {
        "meter_id": defaultdict(list),
        "environment_condition": defaultdict(list),
    }
    for row in rows:
        memberships["meter_id"][str(row.get("meter_id") or "unknown")].append(row)
        conditions = (row.get("metadata") or {}).get("environment_conditions") or []
        if isinstance(conditions, str):
            conditions = [conditions]
        for condition in conditions:
            memberships["environment_condition"][str(condition)].append(row)
    result: dict[str, Any] = {}
    for kind, values in memberships.items():
        result[kind] = {
            name: summarize_scalar_predictions(
                subset,
                bootstrap_iterations=0,
                seed=0,
            )
            for name, subset in sorted(values.items())
        }
    return result


def _run_signature(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    shared_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    protocol_path = _protocol_path(args.manifest)
    return {
        "protocol": EVALUATION_PROTOCOL,
        "training_protocol": VDN_PROTOCOL,
        "vdn_source_commit": verify_vdn_source(args.vdn_source),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_signature": checkpoint.get("signature"),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": (
            sha256_file(protocol_path) if protocol_path.is_file() else None
        ),
        "condition": args.condition,
        "degradation_protocol": ROBUSTNESS_PROTOCOL,
        "degradation_seed": int(args.degradation_seed),
        "meter_detector_weights_sha256": sha256_file(args.meter_detector_weights),
        "keypoint_detector_weights_sha256": sha256_file(args.keypoint_detector_weights),
        "shared_predictions_sha256": (
            sha256_file(args.shared_predictions) if args.shared_predictions else None
        ),
        "shared_predictions_signature": (
            shared_metadata.get("signature") if shared_metadata else None
        ),
        "image_size": int((checkpoint.get("signature") or {}).get("image_size", 384)),
        "batch_size": int(args.batch_size),
        "diagnostic_limit": args.limit,
        "source_sha256": {
            "adapter": sha256_file(PROJECT_DIR / "experiments" / "vdn_baseline.py"),
            "evaluation": sha256_file(Path(__file__).resolve()),
            "degradation": sha256_file(
                PROJECT_DIR / "experiments" / "robustness_degradations.py"
            ),
        },
        "crop_policy": "highest-confidence detected xyxy; square 1.25 expansion",
        "reference_policy": "shared frozen production reference; detector fallback",
        "failure_nmae_penalty": 1.0,
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.vdn_source = args.vdn_source.resolve()
    args.output = args.output.resolve()
    args.meter_detector_weights = args.meter_detector_weights.resolve()
    args.keypoint_detector_weights = args.keypoint_detector_weights.resolve()
    if args.shared_predictions is not None:
        args.shared_predictions = args.shared_predictions.resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0 or args.bootstrap_iterations < 0:
        raise ValueError("batch size must be positive and bootstrap iterations non-negative")
    required = [
        args.manifest,
        args.checkpoint,
        args.meter_detector_weights,
        args.keypoint_detector_weights,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required files are missing: {missing}")

    rows = _read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise ValueError("selected manifest is empty")
    shared, shared_metadata = _load_shared_predictions(
        args.shared_predictions,
        manifest=args.manifest,
        rows=rows,
        condition=args.condition,
        degradation_seed=args.degradation_seed,
        meter_weights=args.meter_detector_weights,
        point_weights=args.keypoint_detector_weights,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_signature = checkpoint.get("signature") or {}
    if training_signature.get("protocol") != VDN_PROTOCOL:
        raise ValueError("checkpoint is not a signed SyncG-retrained VDN artifact")
    if training_signature.get("vdn_source_commit") != verify_vdn_source(args.vdn_source):
        raise ValueError("checkpoint and external VDN source commits differ")
    image_size = int(training_signature.get("image_size", 384))
    signature = _run_signature(args, checkpoint, shared_metadata)
    metadata_path = _metadata_path(args.output)

    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        _summary_path(args.output).unlink(missing_ok=True)
    completed: set[str] = set()
    if args.resume:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("resume requires both output and metadata files")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("VDN evaluation resume signature mismatch")
        completed = {str(row.get("sample_id")) for row in _read_jsonl(args.output)}
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch()
        metadata = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "shared_predictions": (
                str(args.shared_predictions) if args.shared_predictions else None
            ),
            "signature": signature,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        }
        _atomic_json(metadata_path, metadata)

    pending = [row for row in rows if str(row.get("sample_id")) not in completed]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    model = build_vdn_model(
        args.vdn_source,
        image_size=image_size,
        imagenet_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    meter_detector = targetDetectModel(str(args.meter_detector_weights))
    point_detector: targetDetectModel | None = None
    batch: list[dict[str, Any]] = []

    @torch.inference_mode()
    def flush_batch() -> None:
        if not batch:
            return
        inputs = torch.stack([item["tensor"] for item in batch]).to(
            device,
            non_blocking=True,
        )
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            heatmaps, vector_maps = model(inputs)
        directions, peaks, valid = predict_directions(
            heatmaps.float(),
            vector_maps.float(),
        )
        directions_np = directions.detach().cpu().numpy()
        peaks_np = peaks.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        output_rows: list[dict[str, Any]] = []
        for index, item in enumerate(batch):
            row = item["row"]
            degradation = item["degradation"]
            runtime = time.perf_counter() - item["started"]
            if not bool(valid_np[index]):
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="invalid_direction",
                        message="VDN vector at the predicted tip has zero/invalid norm",
                        runtime_seconds=runtime,
                    )
                )
                continue
            direction = directions_np[index].astype(np.float64)
            try:
                pointer_angle = image_angle_from_direction(direction)
                reading, progress = reading_from_pointer_angle(
                    pointer_angle,
                    start_angle=item["start_angle"],
                    range_angle=item["range_angle"],
                    scale_start=float(row["scale_start"]),
                    scale_end=float(row["scale_end"]),
                )
            except ValueError as exc:
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="reading_conversion_failed",
                        message=str(exc),
                        runtime_seconds=runtime,
                    )
                )
                continue
            result = _base_result(row, degradation)
            result.update(
                {
                    "status": True,
                    "prediction": float(reading),
                    "progress": float(progress),
                    "pointer_angle": float(pointer_angle),
                    "direction": direction.tolist(),
                    "direction_angle_error_degrees": _direction_error(
                        direction,
                        item["target_direction"],
                    ),
                    "heatmap_peak": float(peaks_np[index]),
                    "meter_bbox": list(item["bbox"]),
                    "meter_confidence": item["meter_confidence"],
                    "start_angle": float(item["start_angle"]),
                    "range_angle": float(item["range_angle"]),
                    "reference_branch": item["reference_branch"],
                    "reference_source": item["reference_source"],
                    "runtime_seconds": float(runtime),
                }
            )
            output_rows.append(result)
        _append_rows(args.output, output_rows)
        batch.clear()

    for row in tqdm(pending, desc=f"VDN {args.condition}", dynamic_ncols=True):
        started = time.perf_counter()
        sample_id = str(row.get("sample_id"))
        degradation: dict[str, Any] = {
            "protocol": ROBUSTNESS_PROTOCOL,
            "condition": args.condition,
            "seed": int(args.degradation_seed),
        }
        image = cv2.imread(
            str(row.get("image_path")),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            _append_rows(
                args.output,
                [
                    _failure_result(
                        row,
                        degradation,
                        code="image_read_failed",
                        message=f"failed to read {row.get('image_path')}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
            continue
        try:
            image, degradation = apply_degradation(
                image,
                args.condition,
                sample_id=sample_id,
                seed=args.degradation_seed,
            )
            confidences, boxes, crops, _, best_index = meter_detector.image_crop(image)
            if best_index is None or not crops:
                _append_rows(
                    args.output,
                    [
                        _failure_result(
                            row,
                            degradation,
                            code="meter_not_found",
                            message="shared meter detector found no dial",
                            runtime_seconds=time.perf_counter() - started,
                        )
                    ],
                )
                continue
            crop = crops[best_index]
            bbox = _xyxy_from_detector_box(boxes[best_index])
            reference = _shared_reference(shared.get(sample_id))
            if reference is None:
                if point_detector is None:
                    point_detector = targetDetectModel(str(args.keypoint_detector_weights))
                reference = _fallback_reference(point_detector, crop)
                reference_source = "fallback_keypoint_detector"
            else:
                reference_source = "shared_frozen_production_cache"
            start_angle, range_angle, reference_branch = reference
            tensor = vdn_tensor_from_bbox(
                image,
                bbox,
                image_size=image_size,
            )
            batch.append(
                {
                    "row": row,
                    "degradation": degradation,
                    "tensor": tensor,
                    "bbox": bbox,
                    "meter_confidence": float(confidences[best_index]),
                    "start_angle": start_angle,
                    "range_angle": range_angle,
                    "reference_branch": reference_branch,
                    "reference_source": reference_source,
                    "target_direction": _target_direction(row, degradation),
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            _append_rows(
                args.output,
                [
                    _failure_result(
                        row,
                        degradation,
                        code="pipeline_exception",
                        message=f"{type(exc).__name__}: {exc}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
    flush_batch()

    output_rows = _read_jsonl(args.output)
    expected_ids = {str(row.get("sample_id")) for row in rows}
    output_ids = [str(row.get("sample_id")) for row in output_rows]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != expected_ids:
        raise RuntimeError("VDN output is incomplete or contains duplicate sample IDs")
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "condition": args.condition,
        "signature": signature,
        "metrics": summarize_scalar_predictions(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "dialbench_metrics": _dialbench_summary(output_rows),
        "direction_component": _component_summary(output_rows),
        "subgroups": _subgroup_summaries(output_rows),
        "failure_codes": {
            code: sum(row.get("error_code") == code for row in output_rows)
            for code in sorted(
                {str(row.get("error_code")) for row in output_rows if row.get("error_code")}
            )
        },
    }
    _atomic_json(_summary_path(args.output), summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, sort_keys=True))
    print(_summary_path(args.output))


if __name__ == "__main__":
    main()
