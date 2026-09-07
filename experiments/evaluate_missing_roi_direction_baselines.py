"""Run the missing field-ROI DeepLabV3+ and VDN comparison cells.

Industrial-1395 has normalized reading labels but no pixel geometry labels.  Its
DeepLabV3+ and VDN cells therefore share an automatic geometry front-end:
the historical production detector, or the source-trained YOLO pose model's
pivot/start/end outputs. The latter excludes its predicted pointer tip.
RF100-VL retains its annotated pivot
and ordered scale endpoints, matching the existing annotation-assisted
DeepLabV3+ comparison cell.  Pointer-tip annotations are never supplied to a
direction model at inference time.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

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
from experiments.roi_geometry_field import (
    FIELD_DATASETS,
    load_field_cohort,
    load_rf100_keypoints,
)
from experiments.train_deeplabv3plus_roi import METHOD as DEEPLAB_METHOD
from experiments.roi_reference_geometry import (
    SOURCE_POSE_GEOMETRY_ROLE,
    extract_source_pose_references,
    source_pose_input,
)
from experiments.vdn_baseline import (
    build_vdn_model,
    normalized_bgr_tensor,
    predict_directions,
    verify_vdn_source,
)


INDUSTRIAL_DATASETS = (
    "field_gauge_roi_test_a",
    "field_gauge_roi_test_b",
    "field_gauge_external_roi",
)
ALL_DATASETS = (*INDUSTRIAL_DATASETS, "rf100")
AUTO_GEOMETRY_ROLE = (
    "frozen automatic pivot/start/end detector on conditioned ROI pixels"
)
RF100_GEOMETRY_ROLE = "RF100 annotated pivot, minimum, and maximum"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).resolve().open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _geometry_summary_path(output: Path) -> Path:
    output = Path(output).resolve()
    return output.with_suffix(".summary.json")


def _extract_three_points(result: Any) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, "no_reference_detection", {"detections": 0}
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = boxes.conf.detach().cpu().numpy().astype(np.float64)
    coordinates = boxes.xyxy.detach().cpu().numpy().astype(np.float64)
    best: dict[int, tuple[float, np.ndarray]] = {}
    for class_id, confidence, xyxy in zip(
        classes, confidences, coordinates, strict=True
    ):
        class_id = int(class_id)
        if class_id not in (0, 1, 2):
            continue
        point = np.asarray(
            [0.5 * (xyxy[0] + xyxy[2]), 0.5 * (xyxy[1] + xyxy[3])],
            dtype=np.float32,
        )
        previous = best.get(class_id)
        if previous is None or float(confidence) > previous[0]:
            best[class_id] = (float(confidence), point)
    missing = [class_id for class_id in (0, 1, 2) if class_id not in best]
    telemetry: dict[str, Any] = {
        "detections": int(len(boxes)),
        "selected_confidence": {
            str(class_id): best[class_id][0] for class_id in sorted(best)
        },
    }
    if missing:
        telemetry["missing_class_ids"] = missing
        return None, "incomplete_reference_detection", telemetry
    # Detector classes: 0=pivot, 2=start, 1=end.
    points = np.stack((best[0][1], best[2][1], best[1][1])).astype(np.float32)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return None, "invalid_reference_detection", telemetry
    return points, None, telemetry


def prepare_geometry(args: argparse.Namespace) -> dict[str, Any]:
    from ultralytics import YOLO

    output = Path(args.output).resolve()
    summary_path = _geometry_summary_path(output)
    if output.exists() or summary_path.exists():
        raise FileExistsError(f"geometry output already exists: {output}")
    weights = Path(args.reference_detector_weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"reference detector weights are missing: {weights}")

    selected: list[tuple[Any, Sequence[Any]]] = []
    total = 0
    for slug in args.datasets:
        dataset, manifest_rows, _targets = load_field_cohort(slug)
        if args.sample_limit is not None:
            manifest_rows = manifest_rows[: int(args.sample_limit)]
        selected.append((dataset, manifest_rows))
        total += len(manifest_rows) * len(primary_batch.CONDITIONS)

    detector = YOLO(str(weights))
    detector_kind = getattr(args, "reference_detector_kind", "legacy_boxes")
    source_pose = detector_kind == "source_pose4kp"
    geometry_role = SOURCE_POSE_GEOMETRY_ROLE if source_pose else AUTO_GEOMETRY_ROLE
    train_args = (getattr(detector, "ckpt", {}) or {}).get("train_args", {})
    rows: list[dict[str, Any]] = []
    pending_images: list[np.ndarray] = []
    pending_contexts: list[dict[str, Any]] = []
    progress = tqdm(total=total, desc="automatic field geometry", dynamic_ncols=True)

    def flush() -> None:
        if not pending_images:
            return
        results = detector.predict(
            source=list(pending_images),
            imgsz=int(args.image_size),
            batch=len(pending_images),
            device=str(args.device),
            conf=float(args.confidence),
            iou=float(args.iou),
            max_det=int(args.max_det),
            verbose=False,
        )
        if len(results) != len(pending_contexts):
            raise RuntimeError("reference detector returned a different batch size")
        for result, context in zip(results, pending_contexts, strict=True):
            if source_pose:
                points, failure, telemetry = extract_source_pose_references(
                    result, source_shape=context["source_shape"],
                    image_size=int(args.image_size),
                    minimum_keypoint_confidence=float(args.keypoint_confidence),
                )
            else:
                points, failure, telemetry = _extract_three_points(result)
            passed = points is not None and failure is None
            rows.append(
                {
                    "schema_version": 1,
                    "dataset": context["dataset"],
                    "dataset_slug": context["dataset_slug"],
                    "sample_id": context["sample_id"],
                    "group_id": context["group_id"],
                    "condition": context["condition"],
                    "status": "pass" if passed else "fail",
                    "failure_code": None if passed else str(failure),
                    "points_pivot_start_end": points.tolist() if passed else None,
                    "source_shape": list(context["source_shape"]),
                    "geometry_role": geometry_role,
                    "telemetry": telemetry,
                }
            )
        progress.update(len(pending_contexts))
        pending_images.clear()
        pending_contexts.clear()

    try:
        for dataset, manifest_rows in selected:
            _dataset, _rows, targets = load_field_cohort(dataset.slug)
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
                    if source_pose:
                        detector_input = source_pose_input(
                            conditioned, image_size=int(args.image_size)
                        )
                    else:
                        gray = cv2.cvtColor(conditioned, cv2.COLOR_BGR2GRAY)
                        detector_input = np.repeat(gray[:, :, None], 3, axis=2)
                    pending_images.append(np.ascontiguousarray(detector_input))
                    pending_contexts.append(
                        {
                            "dataset": dataset.paper_name,
                            "dataset_slug": dataset.slug,
                            "sample_id": source.sample_id,
                            "group_id": target.group_id,
                            "condition": condition,
                            "source_shape": conditioned.shape[:2],
                        }
                    )
                    if len(pending_images) >= int(args.batch_size):
                        flush()
        flush()
    finally:
        progress.close()

    failures = Counter(
        str(row["failure_code"]) for row in rows if row["status"] != "pass"
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "rows": len(rows),
        "passes": sum(row["status"] == "pass" for row in rows),
        "failures": sum(row["status"] != "pass" for row in rows),
        "failure_codes": dict(sorted(failures.items())),
        "datasets": list(args.datasets),
        "conditions": list(primary_batch.CONDITIONS),
        "sample_limit_per_dataset": args.sample_limit,
        "geometry_role": geometry_role,
        "reference_detector_kind": detector_kind,
        "reference_detector_training": {
            key: train_args.get(key) for key in ("seed", "data", "epochs", "imgsz")
        },
        "reference_detector_weights": str(weights),
        "detector_configuration": {
            "image_size": int(args.image_size),
            "confidence": float(args.confidence),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "grayscale_three_channel": not source_pose,
            "keypoint_confidence": float(args.keypoint_confidence) if source_pose else None,
            "pointer_tip_used": False,
        },
        "predictions": str(output),
    }
    if len(rows) != total:
        raise RuntimeError(f"geometry output has {len(rows)} rows; expected {total}")
    _write_jsonl(output, rows)
    write_json(summary_path, summary)
    return summary


def _load_geometry_cache(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = (
            str(row.get("dataset_slug") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("condition") or ""),
        )
        if not all(key):
            raise ValueError("automatic geometry cache contains an incomplete key")
        if key in index:
            raise ValueError(f"automatic geometry cache contains duplicate key {key}")
        index[key] = row
    return index


def _load_deeplab(args: argparse.Namespace) -> tuple[Any, Mapping[str, Any], int, int]:
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("DeepLabV3+ checkpoint payload is invalid")
    checkpoint_seed = int(payload.get("seed", -1))
    if checkpoint_seed != int(args.seed):
        raise ValueError(
            f"DeepLabV3+ checkpoint seed {checkpoint_seed} differs from {args.seed}"
        )
    image_size = int(payload.get("image_size") or args.image_size)
    model = DeepLabV3PlusROI(imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload, checkpoint_seed, image_size


def _load_vdn(args: argparse.Namespace) -> tuple[Any, Mapping[str, Any], int, int]:
    checkpoint = Path(args.checkpoint).resolve()
    vdn_source = Path(args.vdn_source).resolve()
    verify_vdn_source(vdn_source)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("VDN checkpoint payload is invalid")
    signature = payload.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("VDN checkpoint signature is missing")
    checkpoint_seed = int(signature.get("seed", -1))
    if checkpoint_seed != int(args.seed):
        raise ValueError(f"VDN checkpoint seed {checkpoint_seed} differs from {args.seed}")
    if signature.get("matched_geoprr_split") is not True:
        raise ValueError("VDN checkpoint was not trained on the matched GeoPRR split")
    if int(signature.get("epochs", -1)) != int(args.expected_epochs):
        raise ValueError("VDN checkpoint training budget differs")
    if int(payload.get("epoch", -1)) != int(args.expected_epochs):
        raise ValueError("VDN checkpoint is not the requested terminal epoch")
    image_size = int(signature.get("image_size", 0))
    if image_size < 32 or image_size % 32:
        raise ValueError("VDN checkpoint image size is invalid")
    model = build_vdn_model(vdn_source, image_size=image_size, imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload, checkpoint_seed, image_size


def _failure_record(
    *,
    method: str,
    seed: int,
    dataset: Any,
    sample_id: str,
    group_id: str,
    condition: str,
    target: float,
    failure_code: str,
    geometry_role: str,
    deployable: bool,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": method,
        "seed": int(seed),
        "dataset": dataset.paper_name,
        "dataset_slug": dataset.slug,
        "sample_id": sample_id,
        "group_id": group_id,
        "condition": condition,
        "status": "fail",
        "failure_code": failure_code,
        "normalized_target": float(target),
        "normalized_progress": None,
        "absolute_error": float(FAILURE_ERROR),
        "evaluation_role": (
            "automatic-geometry real-photo direction component"
            if deployable
            else "annotation-assisted real-photo direction component"
        ),
        "deployable": bool(deployable),
        "model_input": "conditioned real-photo ROI pixels only",
        "offline_geometry": geometry_role,
        "telemetry": dict(telemetry or {}),
    }


def _method_name(method: str, seed: int, automatic_geometry: bool) -> str:
    geometry = "auto_geometry" if automatic_geometry else "annotation_geometry"
    if method == "deeplab":
        return f"{DEEPLAB_METHOD}_{geometry}_seed_{seed}"
    return f"VDN_matched_terminal_{geometry}_seed_{seed}"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    output_dir = Path(args.output_dir).resolve()
    index_path = output_dir / "evaluation_index.json"
    planned_paths = [
        output_dir / slug / name
        for slug in args.datasets
        for name in ("predictions.jsonl", "summary.json")
    ]
    collisions = [path for path in (*planned_paths, index_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"evaluation output already exists: {collisions[0]}")

    if args.method == "deeplab":
        model, payload, seed, image_size = _load_deeplab(args)
    else:
        model, payload, seed, image_size = _load_vdn(args)
    model.to(device).eval()
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)

    automatic_geometry_needed = any(slug in INDUSTRIAL_DATASETS for slug in args.datasets)
    geometry_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    geometry_cache_path: Path | None = None
    geometry_metadata: dict[str, Any] = {}
    if automatic_geometry_needed:
        if args.geometry_cache is None:
            raise ValueError("Industrial evaluation requires --geometry-cache")
        geometry_cache_path = Path(args.geometry_cache).resolve()
        geometry_cache = _load_geometry_cache(geometry_cache_path)
        metadata_path = _geometry_summary_path(geometry_cache_path)
        if metadata_path.is_file():
            geometry_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    rf100_keypoints: dict[str, np.ndarray] = {}
    rf100_geometry_audit: dict[str, Any] | None = None
    if "rf100" in args.datasets:
        rf100_keypoints, rf100_geometry_audit = load_rf100_keypoints()

    completed: list[dict[str, Any]] = []
    for slug in args.datasets:
        dataset, manifest_rows, targets = load_field_cohort(slug)
        if args.sample_limit is not None:
            manifest_rows = manifest_rows[: int(args.sample_limit)]
        automatic_geometry = slug in INDUSTRIAL_DATASETS
        geometry_role = (
            str(geometry_metadata.get("geometry_role", AUTO_GEOMETRY_ROLE))
            if automatic_geometry else RF100_GEOMETRY_ROLE
        )
        method_name = _method_name(args.method, seed, automatic_geometry)
        deployable = automatic_geometry
        records: list[dict[str, Any]] = []
        pending_images: list[torch.Tensor] = []
        pending_contexts: list[dict[str, Any]] = []
        progress = tqdm(
            total=len(manifest_rows) * len(primary_batch.CONDITIONS),
            desc=f"{args.method} {slug} seed {seed}",
            dynamic_ncols=True,
        )

        @torch.inference_mode()
        def flush() -> None:
            if not pending_images:
                return
            inputs = torch.stack(pending_images).to(device, non_blocking=True)
            contexts = list(pending_contexts)
            pending_images.clear()
            pending_contexts.clear()
            if args.method == "deeplab":
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    probabilities = torch.sigmoid(model(inputs)).float().cpu().numpy()[:, 0]
                predictions: list[tuple[np.ndarray | None, str | None, dict[str, Any]]] = []
                for probability, context in zip(probabilities, contexts, strict=True):
                    direction, failure, telemetry = direction_from_pointer_probability(
                        probability,
                        context["points"][0],
                        threshold=float(args.mask_threshold),
                    )
                    predictions.append((direction, failure, telemetry))
            else:
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    heatmaps, vector_maps = model(inputs)
                directions, peaks, valid = predict_directions(
                    heatmaps.float(), vector_maps.float()
                )
                directions_np = directions.detach().cpu().numpy()
                peaks_np = peaks.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy()
                predictions = []
                for direction, peak, is_valid in zip(
                    directions_np, peaks_np, valid_np, strict=True
                ):
                    failure = None if bool(is_valid) and math.isfinite(float(peak)) else "invalid_vdn_direction"
                    predictions.append(
                        (
                            direction.astype(np.float32) if failure is None else None,
                            failure,
                            {"heatmap_peak": float(peak)},
                        )
                    )

            for prediction, context in zip(predictions, contexts, strict=True):
                direction, failure, telemetry = prediction
                progress_value: float | None = None
                geometry_telemetry: dict[str, float] = {}
                if direction is not None and failure is None:
                    points = context["points"].copy()
                    points[1] = points[0] + direction * (0.30 * image_size)
                    progress_value, failure, geometry_telemetry = decode_progress_from_keypoints(points)
                passed = progress_value is not None and failure is None
                target_value = float(context["target"])
                if not passed:
                    records.append(
                        _failure_record(
                            method=method_name,
                            seed=seed,
                            dataset=dataset,
                            sample_id=context["sample_id"],
                            group_id=context["group_id"],
                            condition=context["condition"],
                            target=target_value,
                            failure_code=str(failure or "prediction_failed"),
                            geometry_role=geometry_role,
                            deployable=deployable,
                            telemetry={**telemetry, **geometry_telemetry},
                        )
                    )
                else:
                    records.append(
                        {
                            "schema_version": 1,
                            "method": method_name,
                            "seed": seed,
                            "dataset": dataset.paper_name,
                            "dataset_slug": dataset.slug,
                            "sample_id": context["sample_id"],
                            "group_id": context["group_id"],
                            "condition": context["condition"],
                            "status": "pass",
                            "failure_code": None,
                            "normalized_target": target_value,
                            "normalized_progress": float(progress_value),
                            "absolute_error": abs(float(progress_value) - target_value),
                            "evaluation_role": (
                                "automatic-geometry real-photo direction component"
                                if deployable
                                else "annotation-assisted real-photo direction component"
                            ),
                            "deployable": deployable,
                            "model_input": "conditioned real-photo ROI pixels only",
                            "offline_geometry": geometry_role,
                            "telemetry": {**telemetry, **geometry_telemetry},
                        }
                    )
            progress.update(len(contexts))

        try:
            for source in manifest_rows:
                _payload_bytes, clean = primary_batch.load_canonical_roi(source)
                target = targets[source.sample_id]
                for condition in primary_batch.CONDITIONS:
                    conditioned, metadata = robustness_degradations.apply_degradation(
                        clean,
                        condition,
                        sample_id=source.sample_id,
                        seed=primary_batch.ROBUSTNESS_SEED,
                    )
                    if automatic_geometry:
                        cache_key = (slug, source.sample_id, condition)
                        geometry_row = geometry_cache.get(cache_key)
                        if geometry_row is None:
                            raise ValueError(f"automatic geometry cache lacks {cache_key}")
                        cached_shape = tuple(int(value) for value in geometry_row.get("source_shape") or ())
                        if cached_shape != tuple(conditioned.shape[:2]):
                            raise ValueError(f"automatic geometry shape differs for {cache_key}")
                        if geometry_row.get("status") != "pass":
                            records.append(
                                _failure_record(
                                    method=method_name,
                                    seed=seed,
                                    dataset=dataset,
                                    sample_id=source.sample_id,
                                    group_id=target.group_id,
                                    condition=condition,
                                    target=target.normalized_target,
                                    failure_code=f"automatic_geometry:{geometry_row.get('failure_code') or 'failed'}",
                                    geometry_role=geometry_role,
                                    deployable=True,
                                    telemetry=geometry_row.get("telemetry") if isinstance(geometry_row.get("telemetry"), Mapping) else None,
                                )
                            )
                            progress.update(1)
                            continue
                        points_three = np.asarray(
                            geometry_row.get("points_pivot_start_end"), dtype=np.float32
                        )
                        if points_three.shape != (3, 2) or not np.isfinite(points_three).all():
                            raise ValueError(f"automatic geometry points are invalid for {cache_key}")
                        points = np.stack(
                            (points_three[0], points_three[0], points_three[1], points_three[2])
                        )
                    else:
                        annotated = condition_keypoints(rf100_keypoints[source.sample_id], metadata)
                        points = np.stack(
                            (annotated[0], annotated[0], annotated[2], annotated[3])
                        )
                    points = resize_keypoints(
                        points,
                        source_shape=conditioned.shape,
                        output_size=image_size,
                    )
                    resized = cv2.resize(
                        conditioned,
                        (image_size, image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    tensor = (
                        normalized_rgb_tensor(resized)
                        if args.method == "deeplab"
                        else normalized_bgr_tensor(resized)
                    )
                    pending_images.append(tensor)
                    pending_contexts.append(
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
        finally:
            progress.close()

        expected_rows = len(manifest_rows) * len(primary_batch.CONDITIONS)
        if len(records) != expected_rows:
            raise RuntimeError(
                f"{slug} produced {len(records)} rows; expected {expected_rows}"
            )
        prediction_path = output_dir / slug / "predictions.jsonl"
        summary_path = output_dir / slug / "summary.json"
        _write_jsonl(prediction_path, records)
        summary = build_evaluation_summary(
            records,
            method=method_name,
            seed=seed,
            configuration={
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(payload.get("epoch") or 0),
                "method_family": args.method,
                "image_size": image_size,
                "mask_threshold": float(args.mask_threshold) if args.method == "deeplab" else None,
                "samples": len(manifest_rows),
                "conditions": list(primary_batch.CONDITIONS),
                "dataset": dataset.paper_name,
                "dataset_slug": dataset.slug,
                "groups": len({targets[row.sample_id].group_id for row in manifest_rows}),
                "group_unit": dataset.group_unit,
                "field_photos_role": "retrospective test only",
                "evaluation_role": (
                    "automatic-geometry real-photo direction component"
                    if deployable
                    else "annotation-assisted real-photo direction component"
                ),
                "deployable": deployable,
                "geometry_role": geometry_role,
                "geometry_cache": str(geometry_cache_path) if automatic_geometry else None,
                "reference_detector_kind": geometry_metadata.get("reference_detector_kind") if automatic_geometry else None,
                "reference_detector_weights": geometry_metadata.get("reference_detector_weights") if automatic_geometry else None,
                "reference_detector_training": geometry_metadata.get("reference_detector_training") if automatic_geometry else None,
                "geometry_audit": rf100_geometry_audit if not automatic_geometry else None,
            },
        )
        summary["predictions"] = str(prediction_path)
        write_json(summary_path, summary)
        completed.append(
            {
                "dataset_slug": slug,
                "predictions": str(prediction_path),
                "summary": str(summary_path),
                "rows": len(records),
                "all_conditions": summary["all_conditions"],
            }
        )

    index = {
        "schema_version": 1,
        "status": "complete",
        "method": args.method,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "datasets": completed,
    }
    write_json(index_path, index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    geometry = subparsers.add_parser(
        "prepare-geometry", description="Cache automatic Industrial-1395 geometry"
    )
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument(
        "--datasets", nargs="+", choices=INDUSTRIAL_DATASETS, default=list(INDUSTRIAL_DATASETS)
    )
    geometry.add_argument("--reference-detector-weights", type=Path, required=True)
    geometry.add_argument("--reference-detector-kind", choices=("legacy_boxes", "source_pose4kp"), default="legacy_boxes")
    geometry.add_argument("--keypoint-confidence", type=float, default=0.05)
    geometry.add_argument("--device", default="0")
    geometry.add_argument("--batch-size", type=int, default=32)
    geometry.add_argument("--image-size", type=int, default=640)
    geometry.add_argument("--confidence", type=float, default=0.25)
    geometry.add_argument("--iou", type=float, default=0.70)
    geometry.add_argument("--max-det", type=int, default=20)
    geometry.add_argument("--sample-limit", type=int)

    evaluation = subparsers.add_parser(
        "evaluate", description="Evaluate one trained seed on field ROI datasets"
    )
    evaluation.add_argument("--method", choices=("deeplab", "vdn"), required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--vdn-source", type=Path)
    evaluation.add_argument("--seed", type=int, required=True)
    evaluation.add_argument("--expected-epochs", type=int, default=200)
    evaluation.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, required=True)
    evaluation.add_argument("--geometry-cache", type=Path)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--batch-size", type=int, default=16)
    evaluation.add_argument("--image-size", type=int, default=256)
    evaluation.add_argument("--mask-threshold", type=float, default=0.5)
    evaluation.add_argument("--sample-limit", type=int)
    evaluation.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.batch_size) < 1:
        raise ValueError("batch-size must be positive")
    if args.sample_limit is not None and int(args.sample_limit) < 1:
        raise ValueError("sample-limit must be positive")
    if args.command == "prepare-geometry":
        if not 0.0 < float(args.confidence) <= 1.0:
            raise ValueError("confidence must be in (0,1]")
        if not 0.0 < float(args.iou) <= 1.0:
            raise ValueError("iou must be in (0,1]")
        result = prepare_geometry(args)
    else:
        if not 0.0 < float(args.mask_threshold) < 1.0:
            raise ValueError("mask-threshold must be in (0,1)")
        if args.method == "vdn" and args.vdn_source is None:
            raise ValueError("VDN evaluation requires --vdn-source")
        result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "evaluate", "main", "prepare_geometry"]
