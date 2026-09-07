"""Evaluate YOLO11s-Pose-4KP on the labeled real-photo ROI cohorts."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from experiments import robustness_degradations
from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments.evaluate_yolo11s_pose4kp import (
    _prediction_points,
    _square_training_input,
)
from experiments.roi_geometry_comparison import (
    FAILURE_ERROR,
    build_evaluation_summary,
    decode_progress_from_keypoints,
    write_json,
)
from experiments.roi_geometry_field import FIELD_DATASETS, load_field_cohort
from experiments.yolo11s_pose4kp import METHOD


def evaluate_dataset(
    *,
    model: YOLO,
    checkpoint: Path,
    seed: int,
    image_size: int,
    dataset_slug: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset, manifest_rows, targets = load_field_cohort(dataset_slug)
    if args.sample_limit is not None:
        manifest_rows = manifest_rows[: int(args.sample_limit)]
    target_dir = Path(output_dir).resolve() / dataset.slug
    target_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = target_dir / "predictions.jsonl"
    summary_path = target_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError(f"field evaluation output already exists in {target_dir}")

    records: list[dict[str, Any]] = []
    pending_images: list[np.ndarray] = []
    pending_context: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_images:
            return
        results = model.predict(
            source=list(pending_images),
            imgsz=int(image_size),
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
            raise RuntimeError("YOLO field prediction count differs from input batch")
        for result, context in zip(results, contexts, strict=True):
            points, failure, telemetry = _prediction_points(
                result,
                minimum_keypoint_confidence=float(args.keypoint_confidence),
            )
            progress: float | None = None
            geometry_telemetry: dict[str, float] = {}
            if points is not None and failure is None:
                progress, failure, geometry_telemetry = decode_progress_from_keypoints(points)
            passed = progress is not None and failure is None
            target = float(context["target"])
            absolute_error = abs(float(progress) - target) if passed else FAILURE_ERROR
            records.append(
                {
                    "schema_version": 1,
                    "method": f"{METHOD}_seed_{seed}",
                    "seed": seed,
                    "dataset": dataset.paper_name,
                    "dataset_slug": dataset.slug,
                    "sample_id": context["sample_id"],
                    "group_id": context["group_id"],
                    "condition": context["condition"],
                    "status": "pass" if passed else "fail",
                    "failure_code": None if passed else str(failure or "prediction_failed"),
                    "normalized_target": target,
                    "normalized_progress": float(progress) if passed else None,
                    "absolute_error": float(absolute_error),
                    "evaluation_role": "ROI-localized four-keypoint field baseline",
                    "deployable_within_provided_roi": True,
                    "model_input": (
                        "conditioned real-photo ROI pixels resized to the square "
                        "training resolution"
                    ),
                    "offline_geometry": None,
                    "telemetry": {**telemetry, **geometry_telemetry},
                }
            )

    for source in manifest_rows:
        _payload, clean = primary_batch.load_canonical_roi(source)
        target = targets[source.sample_id]
        for condition in primary_batch.CONDITIONS:
            conditioned, _metadata = robustness_degradations.apply_degradation(
                clean,
                condition,
                sample_id=source.sample_id,
                seed=primary_batch.ROBUSTNESS_SEED,
            )
            pending_images.append(
                _square_training_input(conditioned, image_size=image_size)
            )
            pending_context.append(
                {
                    "sample_id": source.sample_id,
                    "group_id": target.group_id,
                    "condition": condition,
                    "target": target.normalized_target,
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
            "checkpoint": str(Path(checkpoint).resolve()),
            "image_size": int(image_size),
            "detection_confidence": float(args.detection_confidence),
            "keypoint_confidence": float(args.keypoint_confidence),
            "samples": len(manifest_rows),
            "conditions": list(primary_batch.CONDITIONS),
            "dataset": dataset.paper_name,
            "dataset_slug": dataset.slug,
            "groups": len({targets[row.sample_id].group_id for row in manifest_rows}),
            "group_unit": dataset.group_unit,
            "field_photos_role": "retrospective test only",
            "geometry_role": dataset.geometry_role,
            "preprocessing": (
                "apply the frozen condition in native ROI coordinates, then "
                "OpenCV bilinear resize to the square YOLO training resolution"
            ),
        },
    )
    summary["predictions"] = str(prediction_path)
    write_json(summary_path, summary)
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    model = YOLO(str(checkpoint))
    payload = getattr(model, "ckpt", {}) or {}
    train_args = payload.get("train_args", {}) if isinstance(payload, Mapping) else {}
    seed = int(train_args.get("seed", args.seed)) if isinstance(train_args, Mapping) else int(args.seed)
    image_size = int(train_args.get("imgsz", args.image_size)) if isinstance(train_args, Mapping) else int(args.image_size)
    summaries = {
        slug: evaluate_dataset(
            model=model,
            checkpoint=checkpoint,
            seed=seed,
            image_size=image_size,
            dataset_slug=slug,
            output_dir=args.output_dir,
            args=args,
        )
        for slug in args.datasets
    }
    index = {
        "schema_version": 1,
        "status": "complete",
        "method": f"{METHOD}_seed_{seed}",
        "seed": seed,
        "datasets": {
            slug: {
                "summary": str(Path(args.output_dir).resolve() / slug / "summary.json"),
                "all_conditions": value["all_conditions"],
            }
            for slug, value in summaries.items()
        },
    }
    write_json(Path(args.output_dir).resolve() / "index.json", index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(FIELD_DATASETS), default=tuple(FIELD_DATASETS))
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
        raise ValueError("batch-size and sample-limit must be positive")
    result = evaluate(args)
    print(json.dumps({"status": result["status"], "method": result["method"], "datasets": list(result["datasets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "evaluate", "evaluate_dataset", "main"]
