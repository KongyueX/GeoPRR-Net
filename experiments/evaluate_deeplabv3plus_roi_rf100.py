"""Evaluate annotation-assisted DeepLabV3+-ROI on real-photo RF100-VL."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from experiments import robustness_degradations
from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments.deeplabv3plus_roi import DeepLabV3PlusROI, normalized_rgb_tensor
from experiments.roi_geometry_comparison import (
    FAILURE_ERROR,
    build_evaluation_summary,
    condition_keypoints,
    decode_progress_from_keypoints,
    direction_from_pointer_probability,
    resize_keypoints,
    write_json,
)
from experiments.roi_geometry_field import load_field_cohort, load_rf100_keypoints
from experiments.train_deeplabv3plus_roi import METHOD


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("DeepLabV3+ checkpoint payload is invalid")
    seed = int(payload.get("seed") or args.seed)
    image_size = int(payload.get("image_size") or args.image_size)
    model = DeepLabV3PlusROI(imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)

    dataset, manifest_rows, targets = load_field_cohort("rf100")
    keypoints, geometry_audit = load_rf100_keypoints()
    if args.sample_limit is not None:
        manifest_rows = manifest_rows[: int(args.sample_limit)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError(f"RF100 DeepLab evaluation output already exists in {output_dir}")

    records: list[dict[str, Any]] = []
    pending_images: list[torch.Tensor] = []
    pending_context: list[dict[str, Any]] = []

    @torch.inference_mode()
    def flush() -> None:
        if not pending_images:
            return
        inputs = torch.stack(pending_images).to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            probabilities = torch.sigmoid(model(inputs)).float().cpu().numpy()[:, 0]
        contexts = list(pending_context)
        pending_images.clear()
        pending_context.clear()
        for probability, context in zip(probabilities, contexts, strict=True):
            points = context["points"]
            direction, failure, telemetry = direction_from_pointer_probability(
                probability,
                points[0],
                threshold=float(args.mask_threshold),
            )
            progress: float | None = None
            geometry_telemetry: dict[str, float] = {}
            if direction is not None and failure is None:
                predicted = points.copy()
                predicted[1] = predicted[0] + direction * (0.30 * image_size)
                progress, failure, geometry_telemetry = decode_progress_from_keypoints(predicted)
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
                    "evaluation_role": "annotation-assisted real-photo pointer-segmentation component",
                    "deployable": False,
                    "model_input": "conditioned real-photo ROI pixels only",
                    "offline_geometry": "RF100 annotated center, minimum, and maximum",
                    "telemetry": {**telemetry, **geometry_telemetry},
                }
            )

    for source in manifest_rows:
        _payload, clean = primary_batch.load_canonical_roi(source)
        target = targets[source.sample_id]
        base_points = keypoints[source.sample_id]
        for condition in primary_batch.CONDITIONS:
            conditioned, metadata = robustness_degradations.apply_degradation(
                clean,
                condition,
                sample_id=source.sample_id,
                seed=primary_batch.ROBUSTNESS_SEED,
            )
            points = condition_keypoints(base_points, metadata)
            points = resize_keypoints(points, source_shape=conditioned.shape, output_size=image_size)
            resized = cv2.resize(conditioned, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            pending_images.append(normalized_rgb_tensor(resized))
            pending_context.append(
                {
                    "sample_id": source.sample_id,
                    "group_id": target.group_id,
                    "condition": condition,
                    "target": target.normalized_target,
                    "points": points,
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
            "checkpoint_epoch": int(payload.get("epoch") or 0),
            "image_size": image_size,
            "mask_threshold": float(args.mask_threshold),
            "samples": len(manifest_rows),
            "conditions": list(primary_batch.CONDITIONS),
            "dataset": dataset.paper_name,
            "dataset_slug": dataset.slug,
            "groups": len({targets[row.sample_id].group_id for row in manifest_rows}),
            "group_unit": dataset.group_unit,
            "field_photos_role": "retrospective test only",
            "evaluation_role": "annotation-assisted pointer-segmentation component",
            "geometry_audit": geometry_audit,
        },
    )
    summary["predictions"] = str(prediction_path)
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or not 0.0 < args.mask_threshold < 1.0:
        raise ValueError("batch-size and mask-threshold are invalid")
    if args.sample_limit is not None and args.sample_limit < 1:
        raise ValueError("sample-limit must be positive")
    summary = evaluate(args)
    print(json.dumps(summary["all_conditions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "evaluate", "main"]
