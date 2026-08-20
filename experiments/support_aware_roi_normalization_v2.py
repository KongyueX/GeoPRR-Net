"""SARN v2: conservative support-component ROI normalization.

Version 2 keeps the pixel-only and no-op-on-uncertainty contract of SARN v1.
It merges every non-padding component above a fixed image-relative noise
floor, takes their shared convex hull, and searches a frozen range of polygon
approximation scales for the best four-corner support.  Thus a single brittle
``approxPolyDP`` scale is no longer required to return exactly four vertices;
all other v1 evidence, fill, span, obliqueness, and confidence gates remain.

The module also provides two explicit commands:

``preflight``
    CPU-only, label-free fixed-sample inspection.  It reports clean/blur
    miscrops, perspective application coverage, bbox IoU against the frozen
    degradation geometry, and recoveries of the two targeted SARN-v1
    fallbacks.  It never loads a checkpoint or invokes a model.

``predict``
    Apply the same normalizer before one frozen DB-GAR18 or matched
    DB-ResNet18 checkpoint while retaining the established prediction schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from experiments import robustness_degradations
from experiments import support_aware_roi_normalization as sarn_v1
from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    ManifestRow,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
)


PROTOCOL: Final[str] = "support_aware_roi_normalization_comparison_v2"
PREFLIGHT_PROTOCOL: Final[str] = "support_aware_roi_normalization_preflight_v2"
STAGE: Final[str] = "Support-Aware ROI Normalization v2"
ALGORITHM: Final[str] = "SARN-v2"

# The padding evidence gates intentionally remain identical to v1.
CORNER_PATCH_FRACTION: Final[float] = 0.04
MAX_CORNER_FLATNESS: Final[float] = 6.0
MAX_CORNER_COLOR_SPREAD: Final[float] = 8.0
MIN_PADDING_FRACTION: Final[float] = 0.08
MAX_PADDING_FRACTION: Final[float] = 0.68
MIN_EDGE_PADDING_COVERAGE: Final[float] = 0.90
MIN_SUPPORT_AREA_FRACTION: Final[float] = 0.30
MAX_SUPPORT_AREA_FRACTION: Final[float] = 0.84
MIN_LONG_SPAN_FRACTION: Final[float] = 0.86
MIN_SHORT_SPAN_FRACTION: Final[float] = 0.42
MAX_SHORT_SPAN_FRACTION: Final[float] = 0.91
MARGIN_FRACTION: Final[float] = 0.005
MAX_NORMALIZED_CROP_AREA_FRACTION: Final[float] = 0.92

MIN_HULL_BBOX_FILL: Final[float] = 0.84
MIN_OBLIQUE_EDGE_DEGREES: Final[float] = 2.0
MIN_QUAD_HULL_AREA_RATIO: Final[float] = 0.84
MIN_COMPONENT_PIXELS: Final[int] = 9
MIN_COMPONENT_AREA_FRACTION: Final[float] = 1e-4
QUAD_EPSILON_FRACTIONS: Final[tuple[float, ...]] = tuple(
    0.020 + 0.0025 * index for index in range(17)
)
MIN_CONFIDENCE: Final[float] = 0.72

TARGETED_V1_FALLBACKS: Final[frozenset[str]] = frozenset(
    {"support_hull_not_quadrilateral", "support_not_convex_quad_like"}
)

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
        "significant_components_merged",
        "quad_epsilon_fraction",
        "quad_area_hull_ratio",
    }
)


class SARNv2Error(RuntimeError):
    """Invalid SARN-v2 configuration or input artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SARNv2Error(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SARNv2Result:
    image: np.ndarray
    applied: bool
    bbox_xyxy: tuple[int, int, int, int] | None
    detected_support_bbox_xyxy: tuple[int, int, int, int] | None
    support_area_fraction: float | None
    crop_area_fraction: float
    confidence: float
    fallback_reason: str | None
    significant_components_merged: int | None
    quad_epsilon_fraction: float | None
    quad_area_hull_ratio: float | None
    # Runtime-only geometry for support-conditioned consumers.  These fields do
    # not enter the frozen v2 sidecar schema or its historical artifact hashes.
    valid_support_mask: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    valid_support_quad_xy: tuple[tuple[float, float], ...] | None = None

    @property
    def support_gate(self) -> float:
        """Reliable residual gate; every fallback is exactly disabled."""

        return float(self.confidence) if self.applied else 0.0


def _fallback(
    image: np.ndarray,
    reason: str,
    *,
    confidence: float = 0.0,
    support_bbox: tuple[int, int, int, int] | None = None,
    support_area_fraction: float | None = None,
    significant_components_merged: int | None = None,
    quad_epsilon_fraction: float | None = None,
    quad_area_hull_ratio: float | None = None,
) -> SARNv2Result:
    return SARNv2Result(
        image=np.ascontiguousarray(image),
        applied=False,
        bbox_xyxy=None,
        detected_support_bbox_xyxy=support_bbox,
        support_area_fraction=support_area_fraction,
        crop_area_fraction=1.0,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        fallback_reason=str(reason),
        significant_components_merged=significant_components_merged,
        quad_epsilon_fraction=quad_epsilon_fraction,
        quad_area_hull_ratio=quad_area_hull_ratio,
        valid_support_mask=np.ones(image.shape[:2], dtype=np.float32),
        valid_support_quad_xy=None,
    )


def _order_quad_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    """Return a convex four-point polygon in stable TL,TR,BR,BL order."""

    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    _require(values.shape == (4, 2), "support quad must contain four points")
    center = np.mean(values, axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # In image coordinates the desired clockwise order has positive shoelace
    # area.  Reverse around TL if the detector returned the opposite winding.
    signed_twice_area = float(
        np.sum(
            ordered[:, 0] * np.roll(ordered[:, 1], -1)
            - ordered[:, 1] * np.roll(ordered[:, 0], -1)
        )
    )
    if signed_twice_area < 0.0:
        ordered = ordered[[0, 3, 2, 1]]
    return np.ascontiguousarray(ordered, dtype=np.float32)


def _aligned_support_geometry(
    polygon: np.ndarray,
    *,
    input_hw: tuple[int, int],
    crop_bbox_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Rasterize the trusted quad and apply the RGB crop/resize geometry."""

    height, width = (int(input_hw[0]), int(input_hw[1]))
    left, top, right, bottom = (int(value) for value in crop_bbox_xyxy)
    _require(
        height >= 1
        and width >= 1
        and 0 <= left < right <= width
        and 0 <= top < bottom <= height,
        "invalid support crop geometry",
    )
    ordered = _order_quad_tl_tr_br_bl(polygon)
    source_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, np.rint(ordered).astype(np.int32), 1)
    crop_mask = source_mask[top:bottom, left:right]
    aligned = cv2.resize(crop_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    aligned = np.ascontiguousarray(aligned.astype(np.float32, copy=False))
    scale_x = float(width) / float(right - left)
    scale_y = float(height) / float(bottom - top)
    transformed = ordered.copy()
    transformed[:, 0] = (transformed[:, 0] - float(left) + 0.5) * scale_x - 0.5
    transformed[:, 1] = (transformed[:, 1] - float(top) + 0.5) * scale_y - 0.5
    transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width - 1))
    transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height - 1))
    quad = tuple((float(point[0]), float(point[1])) for point in transformed)
    return aligned, quad


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


def _significant_component_union(mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None, 0
    height, width = mask.shape
    minimum_area = max(
        MIN_COMPONENT_PIXELS,
        int(math.ceil(MIN_COMPONENT_AREA_FRACTION * height * width)),
    )
    selected = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_area) + 1
    if selected.size == 0:
        return None, 0
    return np.isin(labels, selected).astype(np.uint8), int(selected.size)


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


def _best_adaptive_quad(
    hull: np.ndarray,
) -> tuple[np.ndarray | None, float | None, float | None]:
    perimeter = float(cv2.arcLength(hull, True))
    hull_area = float(cv2.contourArea(hull))
    best_polygon: np.ndarray | None = None
    best_epsilon: float | None = None
    best_ratio: float | None = None
    for epsilon_fraction in QUAD_EPSILON_FRACTIONS:
        polygon = cv2.approxPolyDP(hull, epsilon_fraction * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        ratio = float(cv2.contourArea(polygon)) / max(hull_area, 1.0)
        if best_ratio is None or ratio > best_ratio:
            best_polygon = polygon
            best_epsilon = float(epsilon_fraction)
            best_ratio = float(ratio)
    return best_polygon, best_epsilon, best_ratio


def normalize_support_aware_roi_v2(image_bgr: np.ndarray) -> SARNv2Result:
    """Normalize one ROI from BGR pixels only; uncertainty returns a bytewise no-op."""

    image = np.asarray(image_bgr)
    _require(
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and min(image.shape[:2]) >= 32,
        "SARN-v2 input must be uint8 BGR [H,W,3] with each side at least 32",
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
    color_spread = float(np.max(np.max(medians, axis=0) - np.min(medians, axis=0)))
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
    component, component_count = _significant_component_union(content)
    if component is None:
        return _fallback(image, "valid_content_component_missing")
    contours, _hierarchy = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return _fallback(image, "valid_content_contour_missing")
    hull = cv2.convexHull(np.concatenate(contours, axis=0))
    hull_area = float(cv2.contourArea(hull))
    image_area = float(height * width)
    support_area_fraction = hull_area / image_area
    if not MIN_SUPPORT_AREA_FRACTION <= support_area_fraction <= MAX_SUPPORT_AREA_FRACTION:
        return _fallback(
            image,
            "support_area_outside_protocol_range",
            support_area_fraction=support_area_fraction,
            significant_components_merged=component_count,
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
            significant_components_merged=component_count,
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
            significant_components_merged=component_count,
        )

    polygon, quad_epsilon, quad_area_hull_ratio = _best_adaptive_quad(hull)
    if polygon is None or quad_area_hull_ratio is None:
        return _fallback(
            image,
            "support_hull_not_quadrilateral",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
            significant_components_merged=component_count,
        )
    if quad_area_hull_ratio < MIN_QUAD_HULL_AREA_RATIO:
        return _fallback(
            image,
            "support_quad_area_below_hull_ratio",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
            significant_components_merged=component_count,
            quad_epsilon_fraction=quad_epsilon,
            quad_area_hull_ratio=quad_area_hull_ratio,
        )

    maximum_obliqueness, oblique_edges = _edge_obliqueness(polygon[:, 0, :])
    if oblique_edges < 2:
        return _fallback(
            image,
            "support_edges_not_projectively_oblique",
            support_bbox=support_bbox,
            support_area_fraction=support_area_fraction,
            significant_components_merged=component_count,
            quad_epsilon_fraction=quad_epsilon,
            quad_area_hull_ratio=quad_area_hull_ratio,
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
            significant_components_merged=component_count,
            quad_epsilon_fraction=quad_epsilon,
            quad_area_hull_ratio=quad_area_hull_ratio,
        )

    margin = max(1, int(round(MARGIN_FRACTION * short_side)))
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
            significant_components_merged=component_count,
            quad_epsilon_fraction=quad_epsilon,
            quad_area_hull_ratio=quad_area_hull_ratio,
        )
    crop = image[top:bottom, left:right]
    normalized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    support_mask, support_quad = _aligned_support_geometry(
        polygon,
        input_hw=(height, width),
        crop_bbox_xyxy=(left, top, right, bottom),
    )
    return SARNv2Result(
        image=np.ascontiguousarray(normalized),
        applied=True,
        bbox_xyxy=(int(left), int(top), int(right), int(bottom)),
        detected_support_bbox_xyxy=support_bbox,
        support_area_fraction=support_area_fraction,
        crop_area_fraction=crop_area_fraction,
        confidence=float(confidence),
        fallback_reason=None,
        significant_components_merged=component_count,
        quad_epsilon_fraction=quad_epsilon,
        quad_area_hull_ratio=quad_area_hull_ratio,
        valid_support_mask=support_mask,
        valid_support_quad_xy=support_quad,
    )


def load_strict_plain_manifest(path: Path) -> tuple[ManifestRow, ...]:
    """Reuse the audited four-field manifest parser without widening its schema."""

    try:
        return sarn_v1.load_strict_plain_manifest(path)
    except sarn_v1.SARNError as exc:
        raise SARNv2Error(str(exc)) from exc


def load_sarn_v2_predictor(
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[str, str, Callable[[Sequence[np.ndarray]], list[float]]]:
    family, v1_method, predictor = sarn_v1.load_sarn_predictor(
        checkpoint_path, device_name=device_name
    )
    _require(v1_method.startswith("sarn_"), "SARN-v1 predictor method drift")
    return family, f"sarn_v2_{v1_method.removeprefix('sarn_')}", predictor


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
    """Run one frozen DB-18 checkpoint behind the SARN-v2 frontend."""

    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_strict_plain_manifest(manifest_path)
    family, method, predictor = load_sarn_v2_predictor(
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
            decisions: list[SARNv2Result] = []
            for condition in selected:
                degraded, _unused_metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                degraded = np.ascontiguousarray(degraded)
                decision = normalize_support_aware_roi_v2(degraded)
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
                    "detected_support_bbox_xyxy": list(decision.detected_support_bbox_xyxy)
                    if decision.detected_support_bbox_xyxy is not None
                    else None,
                    "detected_support_area_fraction": decision.support_area_fraction,
                    "normalization_crop_area_fraction": decision.crop_area_fraction,
                    "normalization_confidence": decision.confidence,
                    "normalization_fallback": decision.fallback_reason,
                    "significant_components_merged": decision.significant_components_merged,
                    "quad_epsilon_fraction": decision.quad_epsilon_fraction,
                    "quad_area_hull_ratio": decision.quad_area_hull_ratio,
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
        "schema_version": 2,
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
            "padding_evidence_gates": "identical to SARN-v1",
            "support_estimator": "union significant non-padding components, shared convex hull",
            "minimum_component_pixels": MIN_COMPONENT_PIXELS,
            "minimum_component_area_fraction": MIN_COMPONENT_AREA_FRACTION,
            "quad_epsilon_fractions": list(QUAD_EPSILON_FRACTIONS),
            "minimum_quad_hull_area_ratio": MIN_QUAD_HULL_AREA_RATIO,
            "minimum_hull_bbox_fill": MIN_HULL_BBOX_FILL,
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
    summary.write_bytes(_canonical_json_bytes(summary_value))
    return summary_value


def _fixed_sha256_sample(rows: Sequence[ManifestRow], count: int) -> tuple[ManifestRow, ...]:
    _require(count > 0 and len(rows) >= count, "fixed sample count exceeds manifest")
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(row.sample_id.encode("utf-8")).hexdigest(),
            row.sample_id,
        ),
    )
    return tuple(ranked[:count])


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
    }


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return float(intersection / max(first_area + second_area - intersection, 1))


def _expected_bboxes(
    image: np.ndarray, metadata: Mapping[str, Any]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    perspective = metadata.get("perspective")
    _require(isinstance(perspective, Mapping), "perspective metadata missing")
    destination = np.asarray(perspective.get("destination_corners"), dtype=np.float32)
    _require(destination.shape == (4, 2), "destination corner metadata drift")
    x, y, width, height = cv2.boundingRect(destination)
    image_height, image_width = image.shape[:2]
    support = (
        max(0, int(x)),
        max(0, int(y)),
        min(image_width, int(x + width)),
        min(image_height, int(y + height)),
    )
    margin = max(1, int(round(MARGIN_FRACTION * min(image_height, image_width))))
    normalized = (
        max(0, support[0] - margin),
        max(0, support[1] - margin),
        min(image_width, support[2] + margin),
        min(image_height, support[3] + margin),
    )
    return support, normalized


def run_label_free_preflight(
    *,
    datasets: Sequence[tuple[str, Path]],
    output_path: Path,
    samples_per_dataset: int = 16,
) -> dict[str, Any]:
    """Run a fixed CPU-only pixel preflight; labels and checkpoints are absent."""

    _require(len(datasets) > 0, "at least one dataset is required")
    _require(len({name for name, _path in datasets}) == len(datasets), "duplicate dataset name")
    dataset_reports: dict[str, Any] = {}
    all_sample_ids: list[str] = []
    total_mis_crops = 0
    total_targeted = 0
    total_recovered = 0
    perspective_totals = {
        condition: {"samples": 0, "applied": 0, "normalization_ious": []}
        for condition in ("perspective_moderate", "perspective_severe", "combined_severe")
    }
    for dataset_name, manifest_path in datasets:
        rows = load_strict_plain_manifest(manifest_path)
        selected_rows = _fixed_sha256_sample(rows, samples_per_dataset)
        all_sample_ids.extend(f"{dataset_name}:{row.sample_id}" for row in selected_rows)
        conditions: dict[str, Any] = {}
        for condition in CONDITIONS:
            applied = 0
            v1_targeted = 0
            targeted_recovered = 0
            detected_ious: list[float] = []
            normalized_ious: list[float] = []
            fallback_counts: dict[str, int] = {}
            for row in selected_rows:
                _payload, clean = load_canonical_roi(row)
                degraded, metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=row.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                degraded = np.ascontiguousarray(degraded)
                v1_decision = sarn_v1.normalize_support_aware_roi(degraded)
                decision = normalize_support_aware_roi_v2(degraded)
                if decision.applied:
                    applied += 1
                else:
                    reason = decision.fallback_reason or "unspecified"
                    fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
                if v1_decision.fallback_reason in TARGETED_V1_FALLBACKS:
                    v1_targeted += 1
                    targeted_recovered += int(decision.applied)
                if condition.startswith("perspective") or condition == "combined_severe":
                    expected_support, expected_normalized = _expected_bboxes(degraded, metadata)
                    if decision.detected_support_bbox_xyxy is not None:
                        detected_ious.append(
                            _bbox_iou(decision.detected_support_bbox_xyxy, expected_support)
                        )
                    if decision.bbox_xyxy is not None:
                        normalized_ious.append(
                            _bbox_iou(decision.bbox_xyxy, expected_normalized)
                        )
            clean_like = condition in {"clean", "blur_moderate", "blur_severe"}
            miscrops = applied if clean_like else 0
            total_mis_crops += miscrops
            total_targeted += v1_targeted
            total_recovered += targeted_recovered
            count = len(selected_rows)
            if condition in perspective_totals:
                aggregate = perspective_totals[condition]
                aggregate["samples"] += count
                aggregate["applied"] += applied
                aggregate["normalization_ious"].extend(normalized_ious)
            conditions[condition] = {
                "samples": count,
                "applied": applied,
                "fallback": count - applied,
                "application_rate": float(applied / count),
                "clean_or_blur_miscrops": miscrops,
                "targeted_v1_fallbacks": v1_targeted,
                "targeted_v1_fallbacks_recovered": targeted_recovered,
                "detected_support_bbox_iou_applied_or_located": _distribution(
                    detected_ious
                ),
                "normalization_bbox_iou_applied": _distribution(normalized_ious),
                "normalization_bbox_iou_all_samples_fallback_zero": (
                    _distribution(normalized_ious + [0.0] * (count - len(normalized_ious)))
                    if not clean_like
                    else None
                ),
                "fallback_counts": dict(sorted(fallback_counts.items())),
            }
        dataset_reports[dataset_name] = {
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
            "manifest_rows": len(rows),
            "selection": "ascending sha256(utf8(sample_id)), then sample_id; take first N",
            "selected_sample_ids": [row.sample_id for row in selected_rows],
            "conditions": conditions,
        }

    perspective_gate = {}
    for condition, value in perspective_totals.items():
        sample_count = int(value["samples"])
        applied_count = int(value["applied"])
        ious = list(value["normalization_ious"])
        perspective_gate[condition] = {
            "samples": sample_count,
            "applied": applied_count,
            "application_rate": float(applied_count / max(sample_count, 1)),
            "normalization_bbox_iou_all_samples_fallback_zero": _distribution(
                ious + [0.0] * (sample_count - len(ious))
            ),
        }

    report = {
        "schema_version": 2,
        "protocol": PREFLIGHT_PROTOCOL,
        "algorithm": ALGORITHM,
        "normalizer": "normalize_support_aware_roi_v2(image_bgr)",
        "device": "cpu",
        "checkpoint_or_model_loaded": False,
        "labels_loaded": False,
        "model_results_loaded": False,
        "degradation_geometry_available_only_to_iou_scorer": True,
        "degradation_geometry_available_to_normalizer": False,
        "robustness_seed": ROBUSTNESS_SEED,
        "datasets": dataset_reports,
        "fixed_sample_count": len(all_sample_ids),
        "fixed_sample_ids_sha256": hashlib.sha256(
            "\n".join(all_sample_ids).encode("utf-8")
        ).hexdigest(),
        "clean_or_blur_miscrops": total_mis_crops,
        "targeted_v1_fallbacks": total_targeted,
        "targeted_v1_fallbacks_recovered": total_recovered,
        "perspective_gate_fixed_32": perspective_gate,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="run one DB-18 checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--normalization-sidecar", type=Path)
    predict.add_argument("--summary", type=Path)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--conditions", choices=("all", "clean"), default="all")

    preflight = commands.add_parser("preflight", help="CPU-only label-free pixel preflight")
    preflight.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "MANIFEST"),
        required=True,
    )
    preflight.add_argument("--samples-per-dataset", type=int, default=16)
    preflight.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = run_label_free_preflight(
            datasets=tuple((name, Path(path)) for name, path in args.dataset),
            samples_per_dataset=args.samples_per_dataset,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "protocol": result["protocol"],
                    "fixed_sample_count": result["fixed_sample_count"],
                    "clean_or_blur_miscrops": result["clean_or_blur_miscrops"],
                    "targeted_v1_fallbacks": result["targeted_v1_fallbacks"],
                    "targeted_v1_fallbacks_recovered": result[
                        "targeted_v1_fallbacks_recovered"
                    ],
                    "output": str(Path(args.output).resolve()),
                },
                sort_keys=True,
            )
        )
        return 0

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
    "PREFLIGHT_PROTOCOL",
    "PROTOCOL",
    "SARNv2Error",
    "SARNv2Result",
    "STAGE",
    "load_sarn_v2_predictor",
    "load_strict_plain_manifest",
    "normalize_support_aware_roi_v2",
    "run_label_free_preflight",
    "run_prediction",
]
