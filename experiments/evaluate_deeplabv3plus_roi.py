"""Evaluate DeepLabV3+-ROI on GeoPRR's six-condition SyncG holdout pixels."""
from __future__ import annotations

import argparse
import json
import math
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
    extract_syncg_keypoints,
    normalized_target,
    resize_keypoints,
    write_json,
)
from experiments.train_deeplabv3plus_roi import METHOD
from experiments.vdn_baseline import load_syncg_manifest


EVALUATION_ROLE = "annotation-assisted pointer-segmentation component"


def _roi_keypoints(sample: Any) -> np.ndarray:
    points = extract_syncg_keypoints(sample)
    x1, y1, _x2, _y2 = map(float, sample.dial_bbox)
    origin = np.asarray(
        [max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))],
        dtype=np.float32,
    )
    return points - origin


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("DeepLabV3+ checkpoint payload is invalid")
    image_size = int(payload.get("image_size") or args.image_size)
    seed = int(payload.get("seed") or args.seed)
    model = DeepLabV3PlusROI(imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)

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
                progress, failure, geometry_telemetry = decode_progress_from_keypoints(
                    predicted, clockwise=True
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
                    "deployable": False,
                    "model_input": "conditioned canonical ROI pixels only",
                    "offline_geometry": "annotated pivot and ordered scale endpoints",
                    "telemetry": {**telemetry, **geometry_telemetry},
                }
            )

    for roi_row in roi_rows:
        sample = by_id.get(roi_row.sample_id)
        if sample is None:
            raise ValueError(f"{roi_row.sample_id}: absent from SyncG source manifest")
        _payload, clean = primary_batch.load_canonical_roi(roi_row)
        base_points = _roi_keypoints(sample)
        target = normalized_target(sample)
        scene_stem = Path(str(sample.metadata.get("scene_name") or "unknown")).stem
        for condition in primary_batch.CONDITIONS:
            conditioned, metadata = robustness_degradations.apply_degradation(
                clean,
                condition,
                sample_id=sample.sample_id,
                seed=primary_batch.ROBUSTNESS_SEED,
            )
            points = condition_keypoints(base_points, metadata)
            points = resize_keypoints(
                points, source_shape=conditioned.shape, output_size=image_size
            )
            resized = cv2.resize(
                conditioned,
                (image_size, image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            pending_images.append(normalized_rgb_tensor(resized))
            pending_context.append(
                {
                    "sample_id": sample.sample_id,
                    "scene_stem": scene_stem,
                    "condition": condition,
                    "target": target,
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
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch") or 0),
            "image_size": image_size,
            "mask_threshold": float(args.mask_threshold),
            "samples": len(roi_rows),
            "conditions": list(primary_batch.CONDITIONS),
            "evaluation_role": EVALUATION_ROLE,
            "same_conditioned_roi_protocol_as_geoprr": True,
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
    if (
        args.batch_size < 1
        or not 0.0 < args.mask_threshold < 1.0
        or (args.sample_limit is not None and args.sample_limit < 1)
    ):
        raise ValueError("invalid batch size or mask threshold")
    summary = evaluate(args)
    print(json.dumps(summary["all_conditions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVALUATION_ROLE", "build_parser", "evaluate", "main"]
