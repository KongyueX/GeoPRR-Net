"""Evaluate frozen ReMSTNet-v3 checkpoints on single-view diagnostic tracks.

Two label-compatible tracks are supported:

* ``rpm10k`` scores the predeclared 1,797-image RPM-10K single-pointer
  subset.  The existing meter detector's 22 failures remain in the full
  denominator with normalized error 1.0.
* ``field_full_frame`` replays the existing detector boxes on the 153 unique
  deployment-source full frames in ``unified_real_photo_progress_v1``, which
  was organized from ``data/`` and ``data-717/``.
  Thirty-three frames have scalar labels; the other 120 contribute only to
  system coverage.

Both tracks deliberately disable relation input by supplying identical views,
unit support, an inactive mask, and identity homography.  They therefore return
the raw EfficientNet-B0 foundation posterior exactly even when an input itself
is blurred or tilted.  These are raw-foundation/system diagnostics, not tests
of relation-module benefit and not OCR evaluations.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments.evaluate_remstnet_real_domains import _validate_full_model
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.train_remstnet import load_remstnet_checkpoint
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi
from remstnet.model import CoordinatedReMSTNet


PROTOCOL: Final[str] = "remstnet_v3_single_view_diagnostics_v2"
EXPECTED_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
IMAGE_SIZE: Final[int] = 256
FAILURE_ERROR: Final[float] = 1.0
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260821
DEFAULT_RPM_MANIFEST: Final[Path] = Path(
    "artifacts/manifests/rpm10k_single_pointer_test.jsonl"
)
DEFAULT_RPM_MATERIALIZED_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/rpm10k_natural_tilt_sarn_v2_v1/"
    "materialization/input_manifest.jsonl"
)
DEFAULT_RPM_DETECTOR_SIDECAR: Final[Path] = Path(
    "C:/pointer_read/rpm10k_natural_tilt_sarn_v2_v1/"
    "materialization/materialization_sidecar.jsonl"
)
DEFAULT_FIELD_ROOT: Final[Path] = Path(
    "C:/pointer_read/unified_real_photo_progress_v1"
)
DEFAULT_FIELD_LABELS: Final[Path] = DEFAULT_FIELD_ROOT / "full_frame_labels.jsonl"
DEFAULT_FIELD_LABELED_DETECTIONS: Final[Path] = Path(
    "C:/pointer_read/paper_syncg_only_retrain_v1/field_gauge_full_frame/"
    "field_gauge_full_frame_labeled_predictions.jsonl"
)
DEFAULT_FIELD_UNLABELED_DETECTIONS: Final[Path] = Path(
    "C:/pointer_read/paper_syncg_only_retrain_v1/field_gauge_full_frame/"
    "field_gauge_full_frame_unlabeled_predictions.jsonl"
)


class ReMSTNetCleanExternalError(RuntimeError):
    """The external evaluation assets or model outputs are inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetCleanExternalError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSONL input does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict),
                f"JSONL row is not an object: {source}:{line_number}",
            )
            rows.append(value)
    _require(bool(rows), f"JSONL input is empty: {source}")
    return rows


def _index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        _require(bool(sample_id), f"{label} row lacks sample_id")
        _require(sample_id not in result, f"{label} repeats sample_id {sample_id}")
        result[sample_id] = row
    return result


def _imread(path: Path) -> np.ndarray:
    source = Path(path).resolve()
    payload = np.fromfile(str(source), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    _require(image is not None and image.size > 0, f"cannot decode image: {source}")
    return image


def _crop_bbox(image: np.ndarray, bbox: Sequence[Any], *, sample_id: str) -> np.ndarray:
    _require(len(bbox) >= 4, f"{sample_id}: detector bbox is missing")
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    _require(x1 < x2 and y1 < y2, f"{sample_id}: detector bbox is invalid")
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def _metrics(errors: Sequence[float], passed: Sequence[bool]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    status = np.asarray(passed, dtype=np.bool_)
    _require(values.size > 0 and values.size == status.size, "metric vectors differ")
    _require(
        bool(np.isfinite(values).all()) and bool(np.all((0.0 <= values) & (values <= 1.0))),
        "normalized errors are invalid",
    )
    return {
        "samples": int(values.size),
        "passed": int(status.sum()),
        "coverage": float(status.mean()),
        "nmae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median_absolute_error": float(np.median(values)),
        "p90_absolute_error": float(np.quantile(values, 0.90)),
        "acc_at_2_percent": float(np.mean(status & (values <= 0.02))),
        "acc_at_5_percent": float(np.mean(status & (values <= 0.05))),
    }


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "seed metric vector is empty")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
    }


def _bootstrap_mean_ci(
    errors: Sequence[float],
    *,
    strata: Sequence[str] | None,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    _require(values.size > 0 and replicates >= 1, "bootstrap settings are invalid")
    if strata is None:
        grouped = [np.arange(values.size, dtype=np.int64)]
        unit = "image"
    else:
        _require(len(strata) == values.size, "bootstrap strata are misaligned")
        grouped = [
            np.flatnonzero(np.asarray(strata, dtype=object) == value)
            for value in sorted(set(strata))
        ]
        unit = "image within fixed label-derived stratum"
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in grouped]
        )
        draws[replicate] = float(values[sampled].mean())
    low, high = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "estimate": float(values.mean()),
        "ci95": [low, high],
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": unit,
        "strata": sorted(set(strata)) if strata is not None else None,
    }


def _prepare_rpm10k(
    manifest_path: Path,
    materialized_manifest_path: Path,
    detector_sidecar_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _read_jsonl(manifest_path)
    _require(len(labels) == 1797, "RPM-10K applicability-domain size differs")
    label_index = _index(labels, label="RPM labels")
    crops = _read_jsonl(materialized_manifest_path)
    crop_index = _index(crops, label="RPM materialized crops")
    detector = _read_jsonl(detector_sidecar_path)
    detector_index = _index(detector, label="RPM detector sidecar")
    _require(set(detector_index) == set(label_index), "RPM detector roster differs")
    materialized_root = Path(materialized_manifest_path).resolve().parent

    items: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    for sample_id, label in label_index.items():
        detector_row = detector_index[sample_id]
        detector_passed = str(detector_row.get("status")) == "pass"
        crop_path: Path | None = None
        if detector_passed:
            _require(sample_id in crop_index, f"{sample_id}: detected RPM crop is absent")
            crop_path = (materialized_root / str(crop_index[sample_id]["roi_path"])).resolve()
            _require(crop_path.is_file(), f"{sample_id}: RPM crop does not exist")
        else:
            failure_counts[str(detector_row.get("failure_code") or "detector_failure")] += 1
            _require(sample_id not in crop_index, f"{sample_id}: failed detector has a crop")
        scale_start = float(label["scale_start"])
        scale_end = float(label["scale_end"])
        ground_truth = float(label["ground_truth"])
        _require(scale_end > scale_start, f"{sample_id}: RPM scale is invalid")
        target = (ground_truth - scale_start) / (scale_end - scale_start)
        _require(0.0 <= target <= 1.0, f"{sample_id}: RPM target is invalid")
        metadata = label.get("metadata") or {}
        environment_conditions = list(metadata.get("environment_conditions") or [])
        environment_counts.update({str(value) for value in environment_conditions})
        items.append(
            {
                "sample_id": sample_id,
                "target": float(target),
                "detector_passed": detector_passed,
                "detector_confidence": detector_row.get("detector_confidence"),
                "detector_bbox_xyxy": detector_row.get("detector_bbox_xyxy"),
                "failure_code": detector_row.get("failure_code"),
                "image_path": str(crop_path) if crop_path is not None else None,
                "bbox_xyxy": None,
                "meter_type": str(metadata.get("meter_type") or label.get("group_id")),
                "environment_conditions": environment_conditions,
            }
        )
    return items, {
        "dataset": "RPM-10K single-pointer subset",
        "task": "full-denominator scalar reading from detector-produced meter ROI",
        "samples": len(items),
        "labeled_samples": len(items),
        "detector_passed": sum(bool(item["detector_passed"]) for item in items),
        "detector_coverage": float(
            np.mean([bool(item["detector_passed"]) for item in items])
        ),
        "detector_failure_counts": dict(sorted(failure_counts.items())),
        "environment_condition_counts": dict(sorted(environment_counts.items())),
        "selection": (
            "official test rows from six primary meter types with scalar reading "
            "inside the zero-based official range"
        ),
        "selection_uses_predictions": False,
        "training_or_adaptation_on_dataset": False,
        "manifest": str(Path(manifest_path).resolve()),
        "materialized_detector_manifest": str(Path(materialized_manifest_path).resolve()),
        "detector_sidecar": str(Path(detector_sidecar_path).resolve()),
    }


def _prepare_field_full_frame(
    field_root: Path,
    labels_path: Path,
    labeled_detections_path: Path,
    unlabeled_detections_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _index(_read_jsonl(labels_path), label="field full-frame labels")
    labeled = _read_jsonl(labeled_detections_path)
    unlabeled = _read_jsonl(unlabeled_detections_path)
    detections = labeled + unlabeled
    detection_index = _index(detections, label="field full-frame detector replay")
    _require(len(detection_index) == 153, "field full-frame unique-image count differs")
    _require(set(labels) == {str(row["sample_id"]) for row in labeled}, "field labeled roster differs")

    items: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for sample_id, detector_row in detection_index.items():
        detector_passed = str(detector_row.get("status")) == "pass"
        image_path = (Path(field_root).resolve() / "raw_photos" / f"{sample_id}.png").resolve()
        _require(image_path.is_file(), f"{sample_id}: field full frame is absent")
        label = labels.get(sample_id)
        target = float(label["normalized_progress"]) if label is not None else None
        if detector_passed:
            bbox = detector_row.get("best_bbox_xyxy")
            _require(isinstance(bbox, list) and len(bbox) >= 4, f"{sample_id}: bbox absent")
        else:
            bbox = None
            failure_counts[str(detector_row.get("failure_code") or "detector_failure")] += 1
        items.append(
            {
                "sample_id": sample_id,
                "target": target,
                "detector_passed": detector_passed,
                "detector_confidence": detector_row.get("best_confidence"),
                "detector_bbox_xyxy": bbox,
                "failure_code": detector_row.get("failure_code"),
                "image_path": str(image_path),
                "bbox_xyxy": bbox,
                "meter_type": None,
                "environment_conditions": [],
            }
        )
    labeled_count = sum(item["target"] is not None for item in items)
    _require(labeled_count == 33, "field labeled full-frame count differs")
    return items, {
        "dataset": "Industrial Full-Frame Diagnostic",
        "task": "industrial full frame -> cached YOLO meter detection -> scalar reading",
        "samples": len(items),
        "labeled_samples": labeled_count,
        "unlabeled_samples": len(items) - labeled_count,
        "detector_passed": sum(bool(item["detector_passed"]) for item in items),
        "detector_coverage": float(
            np.mean([bool(item["detector_passed"]) for item in items])
        ),
        "detector_failure_counts": dict(sorted(failure_counts.items())),
        "source": (
            "unified_real_photo_progress_v1/raw_photos, decoded-pixel-deduplicated "
            "from repository data/ and data-717/"
        ),
        "selection_uses_predictions": False,
        "training_or_adaptation_on_dataset": False,
        "detector_replay": (
            "existing best-confidence YOLO boxes are replayed; detector recall/IoU "
            "cannot be measured because bounding-box ground truth is unavailable"
        ),
        "field_root": str(Path(field_root).resolve()),
        "labels": str(Path(labels_path).resolve()),
        "labeled_detector_rows": str(Path(labeled_detections_path).resolve()),
        "unlabeled_detector_rows": str(Path(unlabeled_detections_path).resolve()),
    }


def _load_roi(item: Mapping[str, Any]) -> np.ndarray:
    image = _imread(Path(str(item["image_path"])))
    bbox = item.get("bbox_xyxy")
    return (
        _crop_bbox(image, bbox, sample_id=str(item["sample_id"]))
        if isinstance(bbox, Sequence)
        else image
    )


def _predict_checkpoint(
    checkpoint_path: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[int, dict[str, float], dict[str, str], dict[str, Any]]:
    model, metadata = load_remstnet_checkpoint(checkpoint_path, device=device)
    _require(isinstance(model, CoordinatedReMSTNet), "checkpoint is not coordinated ReMSTNet")
    source_seed = _validate_full_model(model, metadata)
    predictions: dict[str, float] = {}
    failures: dict[str, str] = {}
    detected = [item for item in items if bool(item["detector_passed"])]
    with torch.inference_mode():
        for offset in range(0, len(detected), batch_size):
            batch_items = detected[offset : offset + batch_size]
            try:
                images = torch.stack(
                    [
                        normalized_rgb_tensor(
                            direct_resize_whole_roi(_load_roi(item), size=IMAGE_SIZE)
                        )
                        for item in batch_items
                    ]
                ).to(device)
                count = images.shape[0]
                support = torch.zeros(
                    (count, 1, IMAGE_SIZE, IMAGE_SIZE),
                    dtype=images.dtype,
                    device=device,
                )
                active = torch.zeros(count, dtype=torch.bool, device=device)
                homography = torch.eye(3, dtype=images.dtype, device=device)[None].repeat(
                    count, 1, 1
                )
                output = model(
                    images,
                    images,
                    support,
                    sarn_active=active,
                    raw_to_sarn_homography=homography,
                )
                _require(
                    not bool(output["relation_available"].any().item()),
                    "relation-disabled single-view input unexpectedly activated the relation path",
                )
                values = output["mean"].detach().float().cpu().tolist()
                _require(len(values) == len(batch_items), "prediction batch length differs")
                for item, value in zip(batch_items, values, strict=True):
                    scalar = float(value)
                    _require(
                        math.isfinite(scalar) and 0.0 <= scalar <= 1.0,
                        f"{item['sample_id']}: prediction is invalid",
                    )
                    predictions[str(item["sample_id"])] = scalar
            except Exception as exc:
                code = f"model_exception:{type(exc).__name__}"
                for item in batch_items:
                    failures[str(item["sample_id"])] = code
    return source_seed, predictions, failures, metadata


def evaluate_clean_external(
    *,
    dataset: str,
    checkpoint_paths: Sequence[Path],
    output_path: Path,
    device_name: str = "cuda:0",
    batch_size: int = 64,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    rpm_manifest_path: Path = DEFAULT_RPM_MANIFEST,
    rpm_materialized_manifest_path: Path = DEFAULT_RPM_MATERIALIZED_MANIFEST,
    rpm_detector_sidecar_path: Path = DEFAULT_RPM_DETECTOR_SIDECAR,
    field_root: Path = DEFAULT_FIELD_ROOT,
    field_labels_path: Path = DEFAULT_FIELD_LABELS,
    field_labeled_detections_path: Path = DEFAULT_FIELD_LABELED_DETECTIONS,
    field_unlabeled_detections_path: Path = DEFAULT_FIELD_UNLABELED_DETECTIONS,
) -> dict[str, Any]:
    """Run one external track with the three frozen publication checkpoints."""

    _require(dataset in {"rpm10k", "field_full_frame"}, "external dataset is invalid")
    _require(len(checkpoint_paths) == 3, "exactly three checkpoints are required")
    _require(batch_size >= 1 and bootstrap_replicates >= 1, "runtime settings are invalid")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"output already exists: {output}")
    if dataset == "rpm10k":
        items, dataset_metadata = _prepare_rpm10k(
            rpm_manifest_path,
            rpm_materialized_manifest_path,
            rpm_detector_sidecar_path,
        )
    else:
        items, dataset_metadata = _prepare_field_full_frame(
            field_root,
            field_labels_path,
            field_labeled_detections_path,
            field_unlabeled_detections_path,
        )

    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    predictions_by_seed: dict[int, dict[str, float]] = {}
    model_failures_by_seed: dict[int, dict[str, str]] = {}
    model_metadata: dict[int, dict[str, Any]] = {}
    for checkpoint in checkpoint_paths:
        seed, predictions, failures, metadata = _predict_checkpoint(
            Path(checkpoint),
            items,
            device=device,
            batch_size=batch_size,
        )
        _require(seed not in predictions_by_seed, f"duplicate source seed {seed}")
        predictions_by_seed[seed] = predictions
        model_failures_by_seed[seed] = failures
        model_metadata[seed] = metadata
    _require(tuple(sorted(predictions_by_seed)) == EXPECTED_SEEDS, "checkpoint seed roster differs")

    rows: list[dict[str, Any]] = []
    for item in items:
        sample_id = str(item["sample_id"])
        target = item.get("target")
        row_predictions: dict[str, float | None] = {}
        row_errors: dict[str, float | None] = {}
        row_status: dict[str, bool] = {}
        row_failure_codes: dict[str, str | None] = {}
        for seed in EXPECTED_SEEDS:
            prediction = predictions_by_seed[seed].get(sample_id)
            passed = bool(item["detector_passed"]) and prediction is not None
            row_predictions[str(seed)] = prediction
            row_status[str(seed)] = passed
            row_failure_codes[str(seed)] = (
                None
                if passed
                else str(
                    item.get("failure_code")
                    or model_failures_by_seed[seed].get(sample_id)
                    or "reading_failure"
                )
            )
            row_errors[str(seed)] = (
                abs(float(prediction) - float(target))
                if target is not None and passed
                else (FAILURE_ERROR if target is not None else None)
            )
        rows.append(
            {
                "sample_id": sample_id,
                "normalized_target": target,
                "meter_type": item.get("meter_type"),
                "environment_conditions": item.get("environment_conditions"),
                "detector_passed": bool(item["detector_passed"]),
                "detector_confidence": item.get("detector_confidence"),
                "detector_bbox_xyxy": item.get("detector_bbox_xyxy"),
                "predictions": row_predictions,
                "absolute_errors": row_errors,
                "passed": row_status,
                "failure_codes": row_failure_codes,
            }
        )

    labeled_rows = [row for row in rows if row["normalized_target"] is not None]
    per_seed: dict[str, dict[str, float | int]] = {}
    for seed in EXPECTED_SEEDS:
        errors = [float(row["absolute_errors"][str(seed)]) for row in labeled_rows]
        passed = [bool(row["passed"][str(seed)]) for row in labeled_rows]
        per_seed[str(seed)] = _metrics(errors, passed)
    aggregate = {
        metric: _mean_sd([float(per_seed[str(seed)][metric]) for seed in EXPECTED_SEEDS])
        for metric in (
            "coverage",
            "nmae",
            "rmse",
            "median_absolute_error",
            "p90_absolute_error",
            "acc_at_2_percent",
            "acc_at_5_percent",
        )
    }
    rowwise_mean_errors = [
        statistics.fmean(
            float(row["absolute_errors"][str(seed)]) for seed in EXPECTED_SEEDS
        )
        for row in labeled_rows
    ]
    bootstrap_strata = (
        [str(row["meter_type"]) for row in labeled_rows]
        if dataset == "rpm10k"
        else None
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": dataset_metadata,
        "scope": {
            "single_view_relation_disabled_by_protocol": True,
            "input_may_contain_natural_degradations": True,
            "frozen_checkpoints": True,
            "training_or_adaptation_during_evaluation": False,
            "sarn_relation_active": False,
            "output_equals_raw_foundation_when_relation_is_unavailable": True,
            "ocr_included": False,
            "failure_error": FAILURE_ERROR,
        },
        "models": {str(seed): model_metadata[seed] for seed in EXPECTED_SEEDS},
        "metrics": {
            "per_seed": per_seed,
            "metric_across_seed_mean_sd": aggregate,
            "rowwise_three_seed_mean_error": {
                "nmae": float(statistics.fmean(rowwise_mean_errors)),
                "image_bootstrap": _bootstrap_mean_ci(
                    rowwise_mean_errors,
                    strata=bootstrap_strata,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
            },
        },
        "coverage": {
            "all_images": len(rows),
            "labeled_images": len(labeled_rows),
            "detector_passed_all_images": sum(row["detector_passed"] for row in rows),
            "detector_coverage_all_images": float(
                np.mean([row["detector_passed"] for row in rows])
            ),
            "reading_coverage_by_seed_all_images": {
                str(seed): float(np.mean([row["passed"][str(seed)] for row in rows]))
                for seed in EXPECTED_SEEDS
            },
        },
        "evaluation_elapsed_seconds": float(time.perf_counter() - started),
        "per_image": rows,
        "reporting_notes": [
            "Accuracy uses only the independently available scalar labels; unlabeled field frames contribute to coverage only.",
            "Detector and model failures receive normalized error 1.0 on labeled full denominators.",
            "The protocol disables relation input and exercises the frozen raw foundation through ReMSTNet's exact fallback; image blur or tilt tags do not change that boundary.",
            "Neither track includes scale OCR or automatic physical-range recovery.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("rpm10k", "field_full_frame"), required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--rpm-manifest", type=Path, default=DEFAULT_RPM_MANIFEST)
    parser.add_argument(
        "--rpm-materialized-manifest",
        type=Path,
        default=DEFAULT_RPM_MATERIALIZED_MANIFEST,
    )
    parser.add_argument(
        "--rpm-detector-sidecar", type=Path, default=DEFAULT_RPM_DETECTOR_SIDECAR
    )
    parser.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    parser.add_argument("--field-labels", type=Path, default=DEFAULT_FIELD_LABELS)
    parser.add_argument(
        "--field-labeled-detections",
        type=Path,
        default=DEFAULT_FIELD_LABELED_DETECTIONS,
    )
    parser.add_argument(
        "--field-unlabeled-detections",
        type=Path,
        default=DEFAULT_FIELD_UNLABELED_DETECTIONS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_clean_external(
        dataset=args.dataset,
        checkpoint_paths=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        rpm_manifest_path=args.rpm_manifest,
        rpm_materialized_manifest_path=args.rpm_materialized_manifest,
        rpm_detector_sidecar_path=args.rpm_detector_sidecar,
        field_root=args.field_root,
        field_labels_path=args.field_labels,
        field_labeled_detections_path=args.field_labeled_detections,
        field_unlabeled_detections_path=args.field_unlabeled_detections,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"]["dataset"],
                "samples": result["coverage"]["all_images"],
                "labeled": result["coverage"]["labeled_images"],
                "nmae": result["metrics"]["metric_across_seed_mean_sd"]["nmae"],
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "ReMSTNetCleanExternalError",
    "evaluate_clean_external",
]
