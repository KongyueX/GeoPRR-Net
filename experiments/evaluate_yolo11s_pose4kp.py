"""Evaluate YOLO11s-Pose-4KP on GeoPRR's conditioned SyncG ROI holdout."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from experiments import robustness_degradations
from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments.roi_geometry_comparison import (
    FAILURE_ERROR,
    build_evaluation_summary,
    decode_progress_from_keypoints,
    normalized_target,
    write_json,
)
from experiments.vdn_baseline import load_syncg_manifest
from experiments.yolo11s_pose4kp import METHOD


EVALUATION_ROLE = "ROI-localized four-keypoint reading baseline"


def _square_training_input(image: np.ndarray, *, image_size: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image_size < 2:
        raise ValueError("YOLO evaluation image or image size is invalid")
    return np.ascontiguousarray(
        cv2.resize(
            image,
            (int(image_size), int(image_size)),
            interpolation=cv2.INTER_LINEAR,
        )
    )


def _prediction_points(
    result: Any, *, minimum_keypoint_confidence: float
) -> tuple[np.ndarray | None, str | None, dict[str, float]]:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None, "no_pose_detection", {}
    xy = keypoints.xy.detach().cpu().numpy()
    if xy.ndim != 3 or xy.shape[1:] != (4, 2):
        return None, "unexpected_keypoint_shape", {"detections": float(len(xy))}
    if keypoints.conf is None:
        confidence = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        confidence = keypoints.conf.detach().cpu().numpy().astype(np.float32)
    if boxes is None or boxes.conf is None:
        box_confidence = np.ones(len(xy), dtype=np.float32)
    else:
        box_confidence = boxes.conf.detach().cpu().numpy().astype(np.float32)
    scores = box_confidence * np.mean(confidence, axis=1)
    selected = int(np.argmax(scores))
    selected_confidence = confidence[selected]
    telemetry = {
        "detections": float(len(xy)),
        "box_confidence": float(box_confidence[selected]),
        "minimum_keypoint_confidence": float(np.min(selected_confidence)),
        "mean_keypoint_confidence": float(np.mean(selected_confidence)),
    }
    if not np.isfinite(xy[selected]).all() or not np.isfinite(selected_confidence).all():
        return None, "non_finite_pose", telemetry
    if float(np.min(selected_confidence)) < float(minimum_keypoint_confidence):
        return None, "low_keypoint_confidence", telemetry
    return xy[selected].astype(np.float32), None, telemetry


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    model = YOLO(str(checkpoint))
    checkpoint_payload = getattr(model, "ckpt", {}) or {}
    train_args = checkpoint_payload.get("train_args", {}) if isinstance(checkpoint_payload, Mapping) else {}
    seed = int(train_args.get("seed", args.seed)) if isinstance(train_args, Mapping) else int(args.seed)
    image_size = int(train_args.get("imgsz", args.image_size)) if isinstance(train_args, Mapping) else int(args.image_size)

    source_samples, _protocol = load_syncg_manifest(
        Path(args.syncg_manifest), expected_split="train"
    )
    by_id = {sample.sample_id: sample for sample in source_samples}
    roi_rows = primary_batch.load_manifest(Path(args.roi_manifest).resolve())
    if args.sample_limit is not None:
        roi_rows = roi_rows[: int(args.sample_limit)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError(f"evaluation output already exists in {output_dir}")

    records: list[dict[str, Any]] = []
    pending_images: list[np.ndarray] = []
    pending_context: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_images:
            return
        results = model.predict(
            source=list(pending_images),
            imgsz=image_size,
            batch=int(args.batch_size),
            device=str(args.device),
            conf=float(args.detection_confidence),
            iou=0.7,
            max_det=5,
            augment=False,
            save=False,
            verbose=False,
        )
        contexts = list(pending_context)
        pending_images.clear()
        pending_context.clear()
        if len(results) != len(contexts):
            raise RuntimeError("YOLO prediction count differs from input batch")
        for result, context in zip(results, contexts, strict=True):
            points, failure, telemetry = _prediction_points(
                result,
                minimum_keypoint_confidence=float(args.keypoint_confidence),
            )
            progress: float | None = None
            geometry_telemetry: dict[str, float] = {}
            if points is not None and failure is None:
                progress, failure, geometry_telemetry = decode_progress_from_keypoints(
                    points, clockwise=True
                )
            target = float(context["target"])
            passed = progress is not None and failure is None
            absolute_error = abs(float(progress) - target) if passed else FAILURE_ERROR
            records.append(
                {
                    "schema_version": 1,
                    "method": f"{METHOD}_seed_{seed}",
                    "seed": seed,
                    "sample_id": context["sample_id"],
                    "scene_stem": context["scene_stem"],
                    "condition": context["condition"],
                    "status": "pass" if passed else "fail",
                    "failure_code": None if passed else str(failure or "prediction_failed"),
                    "normalized_target": target,
                    "normalized_progress": float(progress) if passed else None,
                    "absolute_error": float(absolute_error),
                    "evaluation_role": EVALUATION_ROLE,
                    "deployable_within_provided_roi": True,
                    "model_input": (
                        "conditioned canonical ROI pixels resized to the square "
                        "training resolution"
                    ),
                    "offline_geometry": None,
                    "telemetry": {**telemetry, **geometry_telemetry},
                }
            )

    for roi_row in roi_rows:
        sample = by_id.get(roi_row.sample_id)
        if sample is None:
            raise ValueError(f"{roi_row.sample_id}: absent from SyncG source manifest")
        _payload, clean = primary_batch.load_canonical_roi(roi_row)
        target = normalized_target(sample)
        scene_stem = Path(str(sample.metadata.get("scene_name") or "unknown")).stem
        for condition in primary_batch.CONDITIONS:
            conditioned, _metadata = robustness_degradations.apply_degradation(
                clean,
                condition,
                sample_id=sample.sample_id,
                seed=primary_batch.ROBUSTNESS_SEED,
            )
            pending_images.append(
                _square_training_input(conditioned, image_size=image_size)
            )
            pending_context.append(
                {
                    "sample_id": sample.sample_id,
                    "scene_stem": scene_stem,
                    "condition": condition,
                    "target": target,
                }
            )
            if len(pending_images) >= int(args.batch_size):
                flush()
    flush()

    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = build_evaluation_summary(
        records,
        method=f"{METHOD}_seed_{seed}",
        seed=seed,
        configuration={
            "checkpoint": str(checkpoint),
            "image_size": image_size,
            "detection_confidence": float(args.detection_confidence),
            "keypoint_confidence": float(args.keypoint_confidence),
            "samples": len(roi_rows),
            "conditions": list(primary_batch.CONDITIONS),
            "evaluation_role": EVALUATION_ROLE,
            "same_conditioned_roi_protocol_as_geoprr": True,
            "preprocessing": (
                "apply the frozen condition in native ROI coordinates, then "
                "OpenCV bilinear resize to the square YOLO training resolution"
            ),
        },
    )
    summary["predictions"] = str(prediction_path)
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--syncg-manifest", type=Path, default=Path("artifacts/manifests/syncg_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--detection-confidence", type=float, default=0.05)
    parser.add_argument("--keypoint-confidence", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--sample-limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or (args.sample_limit is not None and args.sample_limit < 1):
        raise ValueError("batch-size must be positive")
    summary = evaluate(args)
    print(json.dumps(summary["all_conditions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVALUATION_ROLE", "build_parser", "evaluate", "main"]
