"""SARN: isolated pixel-only ROI normalization comparison for DB-18 models.

Support-Aware ROI Normalization (SARN) detects the constant exterior introduced
by a full-canvas perspective warp from image pixels alone.  It estimates a
shared flat corner color, keeps only matching pixels connected to the canvas
boundary, finds the largest non-padding component and its convex hull, then
conservatively crops and resizes that support back to the original dimensions.
If the evidence is not sufficiently flat, connected, quadrilateral, inset, and
oblique, the input is returned unchanged.

The same one-argument normalization function processes clean and every stress
condition.  It receives no condition name, homography, annotation, target, or
label.  The runner supports one frozen DB-GAR18 or matched DB-ResNet18
checkpoint and a strict four-field plain manifest.  Prediction JSONL retains
the established 12-field schema; SARN decisions and pre/post pixel hashes are
written to an aligned sidecar plus a compact run summary.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments import domain_balanced_geoattn_resnet18 as db_gar18
from experiments import domain_balanced_resnet18 as db_resnet18
from experiments import robustness_degradations
from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    PLAIN_MANIFEST_KEYS,
    ROBUSTNESS_SEED,
    ManifestRow,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)


PROTOCOL: Final[str] = "support_aware_roi_normalization_comparison_v1"
STAGE: Final[str] = "Support-Aware ROI Normalization"
ALGORITHM: Final[str] = "SARN"

CORNER_PATCH_FRACTION: Final[float] = 0.04
MAX_CORNER_FLATNESS: Final[float] = 6.0
MAX_CORNER_COLOR_SPREAD: Final[float] = 8.0
MIN_PADDING_FRACTION: Final[float] = 0.08
MAX_PADDING_FRACTION: Final[float] = 0.68
MIN_EDGE_PADDING_COVERAGE: Final[float] = 0.90
MIN_SUPPORT_AREA_FRACTION: Final[float] = 0.30
MAX_SUPPORT_AREA_FRACTION: Final[float] = 0.84
MIN_HULL_BBOX_FILL: Final[float] = 0.84
MIN_LONG_SPAN_FRACTION: Final[float] = 0.86
MIN_SHORT_SPAN_FRACTION: Final[float] = 0.42
MAX_SHORT_SPAN_FRACTION: Final[float] = 0.91
MIN_OBLIQUE_EDGE_DEGREES: Final[float] = 2.0
MARGIN_FRACTION: Final[float] = 0.02
MAX_NORMALIZED_CROP_AREA_FRACTION: Final[float] = 0.92
MIN_CONFIDENCE: Final[float] = 0.72

SIDECAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "stage",
        "sample_id",
        "method",
        "condition",
        "robustness_seed",
        "pre_normalization_pixel_sha256",
        "post_normalization_pixel_sha256",
        "normalization_applied",
        "normalization_bbox_xyxy",
        "detected_support_bbox_xyxy",
        "detected_support_area_fraction",
        "normalization_crop_area_fraction",
        "normalization_confidence",
        "normalization_fallback",
    }
)


class SARNError(RuntimeError):
    """Invalid SARN configuration or input artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SARNError(message)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SARNResult:
    image: np.ndarray
    applied: bool
    bbox_xyxy: tuple[int, int, int, int] | None
    detected_support_bbox_xyxy: tuple[int, int, int, int] | None
    support_area_fraction: float | None
    crop_area_fraction: float
    confidence: float
    fallback_reason: str | None


def _fallback(
    image: np.ndarray,
    reason: str,
    *,
    confidence: float = 0.0,
    support_bbox: tuple[int, int, int, int] | None = None,
    support_area_fraction: float | None = None,
) -> SARNResult:
    return SARNResult(
        image=np.ascontiguousarray(image),
        applied=False,
        bbox_xyxy=None,
        detected_support_bbox_xyxy=support_bbox,
        support_area_fraction=support_area_fraction,
        crop_area_fraction=1.0,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        fallback_reason=str(reason),
    )


def _corner_patches(image: np.ndarray, size: int) -> tuple[np.ndarray, ...]:
    return (
        image[:size, :size],
        image[:size, -size:],
        image[-size:, -size:],
        image[-size:, :size],
    )


def _boundary_connected_mask(candidate: np.ndarray) -> np.ndarray:
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros(candidate.shape, dtype=bool)
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    boundary_labels = boundary_labels[boundary_labels != 0]
    if boundary_labels.size == 0:
        return np.zeros(candidate.shape, dtype=bool)
    return np.isin(labels, boundary_labels)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def _edge_obliqueness(points: np.ndarray) -> tuple[float, int]:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores: list[float] = []
    for start, stop in zip(values, np.roll(values, -1, axis=0), strict=True):
        delta = stop - start
        angle = abs(math.degrees(math.atan2(float(delta[1]), float(delta[0])))) % 90.0
        scores.append(min(angle, 90.0 - angle))
    return max(scores, default=0.0), sum(
        score >= MIN_OBLIQUE_EDGE_DEGREES for score in scores
    )


def normalize_support_aware_roi(image_bgr: np.ndarray) -> SARNResult:
    """Normalize one ROI from pixels only; unreliable evidence is a no-op."""

    image = np.asarray(image_bgr)
    _require(
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and min(image.shape[:2]) >= 32,
        "SARN input must be uint8 BGR [H,W,3] with each side at least 32",
    )
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    short_side = min(height, width)
    patch_size = max(4, int(round(CORNER_PATCH_FRACTION * short_side)))
    patches = _corner_patches(image, patch_size)
    medians = np.stack(
        [np.median(patch.reshape(-1, 3), axis=0) for patch in patches]
    ).astype(np.float32)
    flatness = np.asarray(
        [
            np.percentile(
                np.max(
                    np.abs(patch.astype(np.float32) - median[None, None, :]),
                    axis=2,
                ),
                90.0,
            )
            for patch, median in zip(patches, medians, strict=True)
        ],
        dtype=np.float32,
    )
    max_flatness = float(np.max(flatness))
    color_spread = float(
        np.max(np.max(medians, axis=0) - np.min(medians, axis=0))
    )
    if max_flatness > MAX_CORNER_FLATNESS:
        return _fallback(image, "corner_not_flat")
    if color_spread > MAX_CORNER_COLOR_SPREAD:
        return _fallback(image, "corner_colors_inconsistent")

    prototype = np.median(medians, axis=0)
    tolerance = float(np.clip(5.0 + 2.0 * max_flatness, 5.0, 18.0))
    color_distance = np.max(
        np.abs(image.astype(np.float32) - prototype[None, None, :]), axis=2
    )
    padding = _boundary_connected_mask(color_distance <= tolerance)
    if not all(
        (
            bool(padding[0, 0]),
            bool(padding[0, -1]),
            bool(padding[-1, -1]),
            bool(padding[-1, 0]),
        )
    ):
        return _fallback(image, "constant_padding_not_corner_connected")
    edge_coverage = min(
        float(np.mean(padding[0])),
        float(np.mean(padding[-1])),
        float(np.mean(padding[:, 0])),
        float(np.mean(padding[:, -1])),
    )
    if edge_coverage < MIN_EDGE_PADDING_COVERAGE:
        return _fallback(image, "constant_padding_not_edge_consistent")
    padding_fraction = float(np.mean(padding))
    if not MIN_PADDING_FRACTION <= padding_fraction <= MAX_PADDING_FRACTION:
        return _fallback(image, "padding_area_outside_protocol_range")

    content = (~padding).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel, iterations=2)
    content = cv2.morphologyEx(content, cv2.MORPH_OPEN, kernel, iterations=1)
    component = _largest_component(content)
    if component is None:
        return _fallback(image, "valid_content_component_missing")
    contours, _hierarchy = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return _fallback(image, "valid_content_contour_missing")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    image_area = float(height * width)
    support_area_fraction = hull_area / image_area
    if not MIN_SUPPORT_AREA_FRACTION <= support_area_fraction <= MAX_SUPPORT_AREA_FRACTION:
        return _fallback(
            image,
            "support_area_outside_protocol_range",
            support_area_fraction=support_area_fraction,
        )

    x, y, box_width, box_height = cv2.boundingRect(hull)
    support_bbox = (int(x), int(y), int(x + box_width), int(y + box_height))
    bbox_area = float(box_width * box_height)
    hull_bbox_fill = hull_area / max(bbox_area, 1.0)
    if hull_bbox_fill < MIN_HULL_BBOX_FILL:
        return _fallback(
            image,
            "support_not_convex_quad_like",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )
    span_x = float(box_width) / float(width)
    span_y = float(box_height) / float(height)
    long_span = max(span_x, span_y)
    short_span = min(span_x, span_y)
    if not (
        long_span >= MIN_LONG_SPAN_FRACTION
        and MIN_SHORT_SPAN_FRACTION <= short_span <= MAX_SHORT_SPAN_FRACTION
    ):
        return _fallback(
            image,
            "support_span_not_full_canvas_projective",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )
    perimeter = float(cv2.arcLength(hull, True))
    polygon = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(polygon) != 4:
        return _fallback(
            image,
            "support_hull_not_quadrilateral",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )
    maximum_obliqueness, oblique_edges = _edge_obliqueness(polygon[:, 0, :])
    if oblique_edges < 2:
        return _fallback(
            image,
            "support_edges_not_projectively_oblique",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )

    flat_score = float(np.clip(1.0 - max_flatness / MAX_CORNER_FLATNESS, 0.0, 1.0))
    consistency_score = float(
        np.clip(1.0 - color_spread / MAX_CORNER_COLOR_SPREAD, 0.0, 1.0)
    )
    fill_score = float(
        np.clip(
            (hull_bbox_fill - MIN_HULL_BBOX_FILL)
            / max(1.0 - MIN_HULL_BBOX_FILL, 1e-6),
            0.0,
            1.0,
        )
    )
    oblique_score = float(np.clip(maximum_obliqueness / 8.0, 0.0, 1.0))
    confidence = (
        0.25 * flat_score
        + 0.25 * consistency_score
        + 0.20 * edge_coverage
        + 0.20 * fill_score
        + 0.10 * oblique_score
    )
    if confidence < MIN_CONFIDENCE:
        return _fallback(
            image,
            "support_confidence_below_threshold",
            confidence=confidence,
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )

    margin = max(2, int(round(MARGIN_FRACTION * short_side)))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(width, x + box_width + margin)
    bottom = min(height, y + box_height + margin)
    crop_area_fraction = float((right - left) * (bottom - top)) / image_area
    if crop_area_fraction > MAX_NORMALIZED_CROP_AREA_FRACTION:
        return _fallback(
            image,
            "conservative_crop_not_meaningful",
            confidence=confidence,
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
        )
    crop = image[top:bottom, left:right]
    normalized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return SARNResult(
        image=np.ascontiguousarray(normalized),
        applied=True,
        bbox_xyxy=(int(left), int(top), int(right), int(bottom)),
        detected_support_bbox_xyxy=support_bbox,
        support_area_fraction=support_area_fraction,
        crop_area_fraction=crop_area_fraction,
        confidence=float(confidence),
        fallback_reason=None,
    )


def load_strict_plain_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Accept exactly the four-field, label-free plain manifest schema."""

    manifest = Path(path).resolve()
    _require(manifest.is_file(), f"manifest does not exist: {manifest}")
    rows_seen = 0
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SARNError(f"manifest line {line_number} is invalid JSON") from exc
            _require(
                isinstance(value, Mapping),
                f"manifest line {line_number} is not an object",
            )
            _require(
                set(value) == set(PLAIN_MANIFEST_KEYS),
                f"manifest line {line_number} is not the strict label-free four-field schema",
            )
            rows_seen += 1
    _require(rows_seen > 0, "manifest is empty")
    return load_plain_manifest(manifest)


def load_sarn_predictor(
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[str, str, Callable[[Sequence[np.ndarray]], list[float]]]:
    """Load either frozen DB-18 family behind the same SARN frontend."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
    identity = (checkpoint.get("protocol"), checkpoint.get("architecture"))
    if identity == (db_gar18.PROTOCOL, db_gar18.ARCHITECTURE):
        base_method, predictor = db_gar18.load_checkpoint_predictor(
            source, device_name=device_name
        )
        _require(base_method.startswith("db_gar18_seed_"), "DB-GAR18 method drift")
        return "db_gar18", f"sarn_{base_method}", predictor
    if identity == (db_resnet18.PROTOCOL, db_resnet18.ARCHITECTURE):
        base_method, predictor = db_resnet18.load_checkpoint_predictor(
            source, device_name=device_name
        )
        _require(base_method.startswith("db_resnet18_seed_"), "DB-ResNet18 method drift")
        return "db_resnet18", f"sarn_{base_method}", predictor
    raise SARNError(f"unsupported checkpoint protocol/architecture: {identity}")


def _derived_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{suffix}")


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    sidecar_path: Path | None = None,
    summary_path: Path | None = None,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_strict_plain_manifest(manifest_path)
    family, method, predictor = load_sarn_predictor(
        checkpoint_path, device_name=device_name
    )
    output = Path(output_path).resolve()
    sidecar = (
        Path(sidecar_path).resolve()
        if sidecar_path is not None
        else _derived_path(output, "normalization.jsonl")
    )
    summary = (
        Path(summary_path).resolve()
        if summary_path is not None
        else _derived_path(output, "summary.json")
    )
    _require(len({output, sidecar, summary}) == 3, "output paths must be distinct")
    for target in (output, sidecar, summary):
        target.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        condition: {"rows": 0, "applied": 0, "fallback": 0, "confidence_sum": 0.0}
        for condition in selected
    }
    row_count = 0
    with output.open("w", encoding="utf-8", newline="\n") as prediction_stream, sidecar.open(
        "w", encoding="utf-8", newline="\n"
    ) as sidecar_stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            normalized_images: list[np.ndarray] = []
            pre_hashes: list[str] = []
            decisions: list[SARNResult] = []
            for condition in selected:
                degraded, _unused_metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                degraded = np.ascontiguousarray(degraded)
                decision = normalize_support_aware_roi(degraded)
                pre_hashes.append(canonical_roi_pixel_sha256(degraded))
                normalized_images.append(decision.image)
                decisions.append(decision)
            try:
                values: list[float | None] = [
                    float(value) for value in predictor(normalized_images)
                ]
                _require(len(values) == len(normalized_images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None
                        and math.isfinite(value)
                        and 0.0 <= value <= 1.0
                        for value in values
                    ),
                    "prediction outside [0,1]",
                )
                failures: list[str | None] = [None] * len(values)
            except Exception as exc:
                values = [None] * len(normalized_images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(values)

            for condition, pre_hash, decision, progress, failure in zip(
                selected, pre_hashes, decisions, values, failures, strict=True
            ):
                post_hash = canonical_roi_pixel_sha256(decision.image)
                passed = progress is not None and failure is None
                prediction_row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": pre_hash,
                }
                _require(set(prediction_row) == set(OUTPUT_KEYS), "prediction schema drift")
                sidecar_row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "stage": STAGE,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "pre_normalization_pixel_sha256": pre_hash,
                    "post_normalization_pixel_sha256": post_hash,
                    "normalization_applied": decision.applied,
                    "normalization_bbox_xyxy": list(decision.bbox_xyxy)
                    if decision.bbox_xyxy is not None
                    else None,
                    "detected_support_bbox_xyxy": list(
                        decision.detected_support_bbox_xyxy
                    )
                    if decision.detected_support_bbox_xyxy is not None
                    else None,
                    "detected_support_area_fraction": decision.support_area_fraction,
                    "normalization_crop_area_fraction": decision.crop_area_fraction,
                    "normalization_confidence": decision.confidence,
                    "normalization_fallback": decision.fallback_reason,
                }
                _require(set(sidecar_row) == set(SIDECAR_KEYS), "sidecar schema drift")
                prediction_stream.write(
                    _canonical_json_bytes(prediction_row).decode("utf-8") + "\n"
                )
                sidecar_stream.write(
                    _canonical_json_bytes(sidecar_row).decode("utf-8") + "\n"
                )
                condition_count = counts[condition]
                condition_count["rows"] += 1
                condition_count["applied" if decision.applied else "fallback"] += 1
                condition_count["confidence_sum"] += float(decision.confidence)
                row_count += 1

    condition_summary = {
        condition: {
            "rows": int(value["rows"]),
            "applied": int(value["applied"]),
            "fallback": int(value["fallback"]),
            "mean_confidence": float(value["confidence_sum"] / max(value["rows"], 1)),
        }
        for condition, value in counts.items()
    }
    summary_value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "algorithm": ALGORITHM,
        "stage": STAGE,
        "model_family": family,
        "method": method,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _sha256_file(Path(checkpoint_path).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
        "conditions": list(selected),
        "robustness_seed": ROBUSTNESS_SEED,
        "rows": row_count,
        "condition_summary": condition_summary,
        "runtime_boundary": {
            "normalizer_inputs": ["uint8_bgr_pixels"],
            "condition_available_to_normalizer": False,
            "homography_available_to_normalizer": False,
            "label_or_target_available_to_normalizer": False,
            "same_normalizer_for_clean_and_stress": True,
            "inference_inputs": ["pixels"],
        },
        "algorithm_parameters": {
            "corner_patch_fraction": CORNER_PATCH_FRACTION,
            "max_corner_flatness": MAX_CORNER_FLATNESS,
            "max_corner_color_spread": MAX_CORNER_COLOR_SPREAD,
            "padding_fraction_range": [MIN_PADDING_FRACTION, MAX_PADDING_FRACTION],
            "minimum_edge_padding_coverage": MIN_EDGE_PADDING_COVERAGE,
            "support_area_fraction_range": [
                MIN_SUPPORT_AREA_FRACTION,
                MAX_SUPPORT_AREA_FRACTION,
            ],
            "minimum_hull_bbox_fill": MIN_HULL_BBOX_FILL,
            "support_span_constraints": {
                "minimum_long": MIN_LONG_SPAN_FRACTION,
                "short_range": [MIN_SHORT_SPAN_FRACTION, MAX_SHORT_SPAN_FRACTION],
            },
            "minimum_oblique_edge_degrees": MIN_OBLIQUE_EDGE_DEGREES,
            "margin_fraction": MARGIN_FRACTION,
            "maximum_crop_area_fraction": MAX_NORMALIZED_CROP_AREA_FRACTION,
            "minimum_confidence": MIN_CONFIDENCE,
            "resize": "OpenCV INTER_LINEAR to original HxW",
            "fallback": "return original pixels unchanged",
        },
        "artifacts": {
            "predictions": {
                "path": str(output),
                "sha256": _sha256_file(output),
                "schema": "established 12-field prediction JSONL",
            },
            "normalization_sidecar": {
                "path": str(sidecar),
                "sha256": _sha256_file(sidecar),
                "schema": sorted(SIDECAR_KEYS),
            },
        },
    }
    with summary.open("wb") as stream:
        stream.write(_canonical_json_bytes(summary_value))
    return summary_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-sidecar", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditions", choices=("all", "clean"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = CONDITIONS if args.conditions == "all" else ("clean",)
    result = run_prediction(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_path=args.output,
        sidecar_path=args.normalization_sidecar,
        summary_path=args.summary,
        device_name=args.device,
        conditions=selected,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol": result["protocol"],
                "stage": result["stage"],
                "method": result["method"],
                "rows": result["rows"],
                "artifacts": result["artifacts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM",
    "PROTOCOL",
    "SARNError",
    "SARNResult",
    "STAGE",
    "load_sarn_predictor",
    "load_strict_plain_manifest",
    "normalize_support_aware_roi",
    "run_prediction",
]
