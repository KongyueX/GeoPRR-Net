"""Run the frozen front-end once and cache all geometry predictions as JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
ANGLE_DIR = PROJECT_DIR / "utils" / "angleDetect"
if str(ANGLE_DIR) not in sys.path:
    sys.path.insert(0, str(ANGLE_DIR))

from zeroShotMeter import meterZeroShot  # noqa: E402
from residual_calibrator import make_calibrator_row  # noqa: E402


DEFAULT_WEIGHTS = {
    "segmentation": ANGLE_DIR / "pointerSeg" / "resultSeg" / "best.pt",
    "meter_detector": ANGLE_DIR / "yoloDetection" / "result" / "yolo_findMeter.pt",
    "meter_transformer": ANGLE_DIR / "vitTranforms" / "result" / "best.pt",
    "keypoint_detector": ANGLE_DIR / "yoloDetection" / "result" / "yolo_pointbest.pt",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class MediaReader:
    def __init__(self) -> None:
        self._captures: dict[str, cv2.VideoCapture] = {}

    def read(self, row: dict[str, Any]) -> np.ndarray:
        image_path = row.get("image_path")
        if image_path:
            image = _imread_unicode(Path(image_path))
            if image is None:
                raise ValueError(f"cannot decode image: {image_path}")
            return image

        video_path = str(row.get("video_path") or "")
        frame_index = row.get("frame_index")
        timestamp_seconds = row.get("timestamp_seconds")
        if not video_path or (frame_index is None and timestamp_seconds is None):
            raise ValueError(
                "sample has neither image_path nor video_path with frame/timestamp"
            )
        capture = self._captures.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"cannot open video: {video_path}")
            self._captures[video_path] = capture
        if frame_index is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            location = f"frame {frame_index}"
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
            location = f"timestamp {timestamp_seconds}s"
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode {location} from {video_path}")
        return frame

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def _flatten_bbox(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("bbox", "box", "dial_bbox", "coordinates"):
            if key in value:
                return _flatten_bbox(value[key])
        aliases = (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("x", "y", "width", "height"),
        )
        for keys in aliases:
            if all(key in value for key in keys):
                return [float(value[key]) for key in keys]
    if isinstance(value, (list, tuple)):
        if len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4]):
            return [float(item) for item in value[:4]]
        for item in value:
            parsed = _flatten_bbox(item)
            if parsed is not None:
                return parsed
    return None


def _crop_from_manifest(
    image: np.ndarray,
    row: dict[str, Any],
    bbox_format: str,
) -> np.ndarray:
    metadata = row.get("metadata") or {}
    bbox = _flatten_bbox(metadata.get("dial_bbox"))
    if bbox is None:
        raise ValueError(f"{row.get('sample_id')}: manifest does not contain a usable dial_bbox")
    x1, y1, a, b = bbox
    if bbox_format == "xywh":
        x2, y2 = x1 + a, y1 + b
    else:
        x2, y2 = a, b
    height, width = image.shape[:2]
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), x1 + 1, width))
    y2 = int(np.clip(round(y2), y1 + 1, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"{row.get('sample_id')}: dial_bbox produced an empty crop")
    return crop


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _reading_payload(reading: dict[str, Any] | None) -> dict[str, Any]:
    reading = reading or {}
    return {
        "status": bool(reading.get("status")),
        "backend": reading.get("backend"),
        "prediction": _finite(reading.get("resultNum")),
        "progress": _finite(reading.get("progress_ratio")),
        "pointer_angle": _finite(reading.get("pointer_angle")),
        "confidence": _finite(reading.get("confidence")),
        "message": reading.get("message"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_completed_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed = set()
    for row in _read_jsonl(output):
        key = f"{row.get('dataset')}::{row.get('split')}::{row.get('sample_id')}"
        completed.add(key)
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _manifest_protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _prepare_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    weight_paths = {
        "segmentation": args.segmentation_weights.resolve(),
        "meter_detector": args.meter_detector_weights.resolve(),
        "meter_transformer": args.meter_transformer_weights.resolve(),
        "keypoint_detector": args.keypoint_detector_weights.resolve(),
    }
    source_paths = {
        "collector": Path(__file__).resolve(),
        "zero_shot_meter": ANGLE_DIR / "zeroShotMeter.py",
        "geometry_baseline": ANGLE_DIR / "geometry_baseline.py",
        "residual_features": ANGLE_DIR / "residual_calibrator.py",
        "pointer_seg_inference": ANGLE_DIR / "pointerSeg" / "detectSeg.py",
        "pointer_seg_model": ANGLE_DIR / "pointerSeg" / "u2netp.py",
        "letterbox": ANGLE_DIR / "dataloader.py",
        "meter_detector": ANGLE_DIR / "yoloDetection" / "yoloDectect.py",
        "detector_geometry": ANGLE_DIR / "yoloDetection" / "pointGet.py",
        "meter_transformer_adapter": ANGLE_DIR / "detect.py",
        "meter_transformer": ANGLE_DIR / "vitTranforms" / "meterCilp.py",
        "meter_transformer_encoder": ANGLE_DIR / "vitTranforms" / "encoder.py",
        "meter_transformer_decoder": ANGLE_DIR / "vitTranforms" / "decoder.py",
        "meter_transformer_image_encoder": (
            ANGLE_DIR / "vitTranforms" / "imgencoder.py"
        ),
        "meter_transformer_text_encoder": (
            ANGLE_DIR / "vitTranforms" / "textencoder.py"
        ),
        "meter_transformer_tokenizer": (
            ANGLE_DIR / "vitTranforms" / "simple_tokenizer.py"
        ),
    }
    protocol_path = _manifest_protocol_path(args.manifest)
    manifest_protocol = None
    if protocol_path.is_file():
        manifest_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_protocol, dict):
            raise ValueError(f"{protocol_path} must contain a JSON object")
    signature = {
        "manifest_sha256": _sha256(args.manifest),
        "manifest_protocol_sha256": (
            _sha256(protocol_path) if protocol_path.is_file() else None
        ),
        "weights_sha256": {
            name: _sha256(path) for name, path in weight_paths.items()
        },
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "device": str(args.device),
        "correction_mode": str(args.correction_mode),
        "confidence": args.confidence,
        "use_manifest_crop": bool(args.use_manifest_crop),
        "bbox_format": str(args.bbox_format),
        "validate_mask_line": not args.disable_mask_validation,
        "include_transformer": not args.skip_transformer,
        "reading_backend": (
            "geometry_fusion_weighted"
            if args.skip_transformer
            else "compare"
        ),
    }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_protocol_path": (
            str(protocol_path.resolve()) if protocol_path.is_file() else None
        ),
        "manifest_protocol": manifest_protocol,
        "weights": {name: str(path) for name, path in weight_paths.items()},
        "signature": signature,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--correction-mode", default="off")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-manifest-crop",
        action="store_true",
        help="diagnostic oracle-crop mode; do not mix with full-pipeline main results",
    )
    parser.add_argument("--bbox-format", choices=("xyxy", "xywh"), default="xyxy")
    parser.add_argument("--disable-mask-validation", action="store_true")
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="diagnostic speed option; formal paper caches include the original baseline",
    )
    parser.add_argument("--segmentation-weights", type=Path, default=DEFAULT_WEIGHTS["segmentation"])
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_WEIGHTS["meter_detector"])
    parser.add_argument("--meter-transformer-weights", type=Path, default=DEFAULT_WEIGHTS["meter_transformer"])
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_WEIGHTS["keypoint_detector"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")

    weights = (
        args.segmentation_weights,
        args.meter_detector_weights,
        args.meter_transformer_weights,
        args.keypoint_detector_weights,
    )
    missing = [str(path) for path in weights if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing model weights: {missing}")

    run_metadata = _prepare_run_metadata(args)
    metadata_path = _metadata_path(args.output)
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{metadata_path} is missing; cannot verify a safe resume"
            )
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_metadata.get("signature") != run_metadata.get("signature"):
            raise ValueError(
                "resume signature mismatch: manifest, weights, or inference options changed"
            )
    else:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rows = _read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    completed_ids = _load_completed_ids(args.output) if args.resume else set()
    mode = "a" if args.resume else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = meterZeroShot(
        str(args.segmentation_weights),
        str(args.meter_detector_weights),
        str(args.meter_transformer_weights),
        str(args.keypoint_detector_weights),
        device,
    )
    media_reader = MediaReader()
    processed = 0
    failures = 0
    started = time.perf_counter()
    try:
        with args.output.open(mode, encoding="utf-8", newline="\n") as output_handle:
            for index, sample in enumerate(rows, 1):
                sample_key = (
                    f"{sample.get('dataset')}::{sample.get('split')}::{sample.get('sample_id')}"
                )
                if sample_key in completed_ids:
                    continue
                result: dict[str, Any] = {
                    key: sample.get(key)
                    for key in (
                        "dataset",
                        "split",
                        "sample_id",
                        "group_id",
                        "meter_id",
                        "image_path",
                        "video_path",
                        "frame_index",
                        "timestamp_seconds",
                        "ground_truth",
                        "scale_start",
                        "scale_end",
                        "metadata",
                    )
                    if key in sample
                }
                sample_started = time.perf_counter()
                try:
                    image = media_reader.read(sample)
                    if args.use_manifest_crop:
                        image = _crop_from_manifest(image, sample, args.bbox_format)
                    model.Inference(
                        image,
                        float(sample["scale_start"]),
                        float(sample["scale_end"]),
                        confidence=args.confidence,
                        use_origin_when_no_meter=args.use_manifest_crop,
                        correction_mode=args.correction_mode,
                        reading_backend=(
                            "geometry_fusion_weighted"
                            if args.skip_transformer
                            else "compare"
                        ),
                        residual_calibrator_path="",
                        validate_mask_line=not args.disable_mask_validation,
                    )
                    details = model._last_reading_details or {}
                    transformer = details.get("transformer") or {}
                    v1 = details.get("geometry_direct") or {}
                    v2 = details.get("geometry_direct_v2") or {}
                    mean_fusion = details.get("geometry_fusion") or {}
                    weighted_fusion = details.get("geometry_fusion_weighted") or {}
                    artifacts = model._last_training_artifacts or {}
                    feature_row = make_calibrator_row(
                        v1,
                        v2,
                        weighted_fusion,
                        artifacts,
                        corrected_crop_bgr=artifacts.get("corrected_crop_bgr"),
                        scale_start=sample["scale_start"],
                        scale_end=sample["scale_end"],
                    )
                    result.update(
                        status=bool(weighted_fusion.get("status")),
                        error_code=model.last_error_code,
                        error_message=model.last_error_message,
                        branch=details.get("branch"),
                        methods={
                            "transformer": _reading_payload(transformer),
                            "geometry_v1": _reading_payload(v1),
                            "geometry_v2": _reading_payload(v2),
                            "mean_fusion": _reading_payload(mean_fusion),
                            "weighted_fusion": _reading_payload(weighted_fusion),
                        },
                        features=feature_row,
                        fusion_source_weights=weighted_fusion.get("fusion_source_weights") or {},
                        fusion_source_quality=weighted_fusion.get("fusion_source_quality") or {},
                    )
                    if not result["status"]:
                        if not result.get("error_code"):
                            result["error_code"] = "reading_backend_failed"
                            result["error_message"] = (
                                weighted_fusion.get("message")
                                or "weighted geometry reading failed"
                            )
                        failures += 1
                except Exception as exc:
                    failures += 1
                    result.update(
                        status=False,
                        error_code="collector_exception",
                        error_message=f"{type(exc).__name__}: {exc}",
                        methods={},
                        features={},
                    )
                result["runtime_seconds"] = time.perf_counter() - sample_started
                output_handle.write(
                    json.dumps(result, ensure_ascii=False, default=_json_default, sort_keys=True)
                    + "\n"
                )
                output_handle.flush()
                processed += 1
                if processed % 25 == 0 or index == len(rows):
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{index}/{len(rows)}] wrote={processed} failures={failures} "
                        f"elapsed={elapsed:.1f}s"
                    )
    finally:
        media_reader.close()

    elapsed = time.perf_counter() - started
    print(
        f"finished: wrote={processed}, failures={failures}, "
        f"seconds={elapsed:.1f}, output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
