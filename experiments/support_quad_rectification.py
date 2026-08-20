"""One-pass quadrilateral rectification for trusted SARN-v2 decisions.

SARN-v2 exposes its valid-support quadrilateral in the coordinate system of
``decision.image``.  That image has already undergone a half-open bbox crop
followed by an OpenCV resize back to the original canvas size.  Rectifying
``decision.image`` would therefore interpolate RGB pixels twice.

This module reverses only that crop/resize *coordinate transform* using the
same pixel-centre convention as SARN-v2, then estimates a homography from the
recovered quadrilateral on the original degraded image directly to the full
output rectangle.  Consequently an applied result performs exactly one RGB
``warpPerspective`` operation.  It never consumes degradation metadata or an
oracle homography.

Every SARN fallback and every rejected quadrilateral is a bytewise image
no-op.  Rejection is represented in metadata rather than raised because this
helper sits behind an already conservative, pixel-only detector.  Invalid
image contracts still raise ``SupportQuadRectificationError``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import cv2
import numpy as np


PROTOCOL: Final[str] = "sarn_aligned_quad_direct_rectification_v1"
PIXEL_CENTER_FORMULA: Final[str] = (
    "source=(aligned+0.5)*crop_extent/output_extent+crop_origin-0.5"
)
MIN_QUAD_AREA_PIXELS: Final[float] = 16.0
MIN_SOURCE_QUAD_AREA_FRACTION: Final[float] = 0.01
MIN_EDGE_LENGTH_PIXELS: Final[float] = 2.0
MAX_HOMOGRAPHY_CONDITION_NUMBER: Final[float] = 1.0e10
COORDINATE_TOLERANCE_PIXELS: Final[float] = 1.0


class SupportQuadRectificationError(ValueError):
    """The caller supplied an invalid image rather than uncertain geometry."""


class _GeometryRejected(RuntimeError):
    """Internal signal for a safe, bytewise rectification fallback."""


@runtime_checkable
class SARNDecisionLike(Protocol):
    """Minimal read-only SARN-v2 decision surface used by this helper."""

    applied: bool
    bbox_xyxy: tuple[int, int, int, int] | None
    valid_support_quad_xy: tuple[tuple[float, float], ...] | None
    confidence: float
    fallback_reason: str | None


@dataclass(frozen=True)
class QuadRectificationMetadata:
    protocol: str
    applied: bool
    fallback_reason: str | None
    input_hw: tuple[int, int]
    output_hw: tuple[int, int]
    sarn_crop_bbox_xyxy: tuple[int, int, int, int] | None
    sarn_confidence: float
    aligned_quad_xy: tuple[tuple[float, float], ...] | None
    source_quad_xy: tuple[tuple[float, float], ...] | None
    destination_quad_xy: tuple[tuple[float, float], ...] | None
    homography: tuple[tuple[float, float, float], ...] | None
    source_quad_area_fraction: float | None
    homography_condition_number: float | None
    pixel_center_inverse: str
    rgb_resampling: str
    mask_resampling: str


@dataclass(frozen=True)
class QuadRectificationResult:
    image: np.ndarray
    valid_support_mask: np.ndarray
    metadata: QuadRectificationMetadata

    @property
    def applied(self) -> bool:
        return self.metadata.applied


def _quad_tuple(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    values = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return tuple((float(value[0]), float(value[1])) for value in values)


def _signed_twice_area(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return float(
        np.sum(
            values[:, 0] * np.roll(values[:, 1], -1)
            - values[:, 1] * np.roll(values[:, 0], -1)
        )
    )


def _order_quad_tl_tr_br_bl(points: object) -> np.ndarray:
    """Canonicalize any permutation of a convex quad to TL,TR,BR,BL."""

    try:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise _GeometryRejected("invalid_quad_shape") from exc
    if values.shape != (4, 2):
        raise _GeometryRejected("invalid_quad_shape")
    if not np.all(np.isfinite(values)):
        raise _GeometryRejected("non_finite_quad")

    pairwise = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    distances += np.eye(4, dtype=np.float64) * (MIN_EDGE_LENGTH_PIXELS + 1.0)
    if float(np.min(distances)) < MIN_EDGE_LENGTH_PIXELS:
        raise _GeometryRejected("duplicate_or_near_duplicate_corners")

    float32_values = values.astype(np.float32)
    if not np.all(np.isfinite(float32_values)):
        raise _GeometryRejected("quad_coordinates_out_of_range")
    try:
        hull = cv2.convexHull(float32_values.reshape(-1, 1, 2))
    except cv2.error as exc:
        raise _GeometryRejected("quad_hull_failed") from exc
    if hull is None or hull.reshape(-1, 2).shape != (4, 2):
        raise _GeometryRejected("quad_not_strictly_convex")
    hull_values = hull.reshape(4, 2).astype(np.float64)
    center = np.mean(hull_values, axis=0)
    angles = np.arctan2(
        hull_values[:, 1] - center[1], hull_values[:, 0] - center[0]
    )
    ordered = hull_values[np.argsort(angles)]

    # The smallest x+y corner is TL for the non-rotating yaw/pitch projective
    # family used by the robustness protocol.  y/x are deterministic tie
    # breakers for symmetric or integer-rounded quads.
    scores = ordered[:, 0] + ordered[:, 1]
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1], scores))[0])
    ordered = np.roll(ordered, -start, axis=0)
    if _signed_twice_area(ordered) < 0.0:
        ordered = ordered[[0, 3, 2, 1]]

    edge_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    if float(np.min(edge_lengths)) < MIN_EDGE_LENGTH_PIXELS:
        raise _GeometryRejected("quad_edge_too_short")
    area = 0.5 * abs(_signed_twice_area(ordered))
    if area < MIN_QUAD_AREA_PIXELS:
        raise _GeometryRejected("quad_area_too_small")
    return np.ascontiguousarray(ordered.astype(np.float32))


def recover_source_quad_from_sarn_alignment(
    aligned_quad_xy: object,
    *,
    crop_bbox_xyxy: tuple[int, int, int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Invert SARN-v2's crop/resize pixel-centre transform for four corners."""

    height, width = (int(output_hw[0]), int(output_hw[1]))
    if height < 1 or width < 1:
        raise _GeometryRejected("invalid_output_shape")
    try:
        left, top, right, bottom = (int(value) for value in crop_bbox_xyxy)
    except (TypeError, ValueError) as exc:
        raise _GeometryRejected("invalid_crop_bbox") from exc
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise _GeometryRejected("invalid_crop_bbox")

    aligned = _order_quad_tl_tr_br_bl(aligned_quad_xy).astype(np.float64)
    tolerance = COORDINATE_TOLERANCE_PIXELS
    if (
        float(np.min(aligned[:, 0])) < -tolerance
        or float(np.max(aligned[:, 0])) > float(width - 1) + tolerance
        or float(np.min(aligned[:, 1])) < -tolerance
        or float(np.max(aligned[:, 1])) > float(height - 1) + tolerance
    ):
        raise _GeometryRejected("aligned_quad_outside_canvas")

    crop_width = float(right - left)
    crop_height = float(bottom - top)
    source = aligned.copy()
    source[:, 0] = (
        (aligned[:, 0] + 0.5) * crop_width / float(width)
        + float(left)
        - 0.5
    )
    source[:, 1] = (
        (aligned[:, 1] + 0.5) * crop_height / float(height)
        + float(top)
        - 0.5
    )
    if (
        float(np.min(source[:, 0])) < -tolerance
        or float(np.max(source[:, 0])) > float(width - 1) + tolerance
        or float(np.min(source[:, 1])) < -tolerance
        or float(np.max(source[:, 1])) > float(height - 1) + tolerance
    ):
        raise _GeometryRejected("recovered_quad_outside_canvas")
    return _order_quad_tl_tr_br_bl(source)


def _fallback_result(
    image: np.ndarray,
    *,
    reason: str,
    confidence: float,
    bbox: tuple[int, int, int, int] | None,
    aligned_quad: tuple[tuple[float, float], ...] | None = None,
) -> QuadRectificationResult:
    height, width = image.shape[:2]
    metadata = QuadRectificationMetadata(
        protocol=PROTOCOL,
        applied=False,
        fallback_reason=str(reason),
        input_hw=(height, width),
        output_hw=(height, width),
        sarn_crop_bbox_xyxy=bbox,
        sarn_confidence=float(confidence),
        aligned_quad_xy=aligned_quad,
        source_quad_xy=None,
        destination_quad_xy=None,
        homography=None,
        source_quad_area_fraction=None,
        homography_condition_number=None,
        pixel_center_inverse=PIXEL_CENTER_FORMULA,
        rgb_resampling="none",
        mask_resampling="none",
    )
    return QuadRectificationResult(
        image=np.ascontiguousarray(image),
        valid_support_mask=np.ones((height, width), dtype=np.float32),
        metadata=metadata,
    )


def rectify_sarn_support_quad(
    degraded_image_bgr: np.ndarray,
    decision: SARNDecisionLike,
) -> QuadRectificationResult:
    """Rectify one trusted SARN quad from the original degraded image.

    The function intentionally does not accept a condition name, degradation
    homography, label, or target.  A non-applied SARN decision and any invalid
    applied geometry both return the input image byte-for-byte unchanged.
    """

    image = np.asarray(degraded_image_bgr)
    if not (
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and min(image.shape[:2]) >= 2
    ):
        raise SupportQuadRectificationError(
            "rectification input must be uint8 BGR [H,W,3] with each side at least 2"
        )
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    try:
        confidence = float(getattr(decision, "confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    bbox = getattr(decision, "bbox_xyxy", None)
    if not bool(getattr(decision, "applied", False)):
        sarn_reason = getattr(decision, "fallback_reason", None) or "unspecified"
        return _fallback_result(
            image,
            reason=f"sarn_not_applied:{sarn_reason}",
            confidence=confidence,
            bbox=bbox,
        )

    raw_aligned_quad = getattr(decision, "valid_support_quad_xy", None)
    aligned_tuple: tuple[tuple[float, float], ...] | None = None
    try:
        if bbox is None:
            raise _GeometryRejected("applied_decision_missing_crop_bbox")
        if raw_aligned_quad is None:
            raise _GeometryRejected("applied_decision_missing_support_quad")
        aligned = _order_quad_tl_tr_br_bl(raw_aligned_quad)
        aligned_tuple = _quad_tuple(aligned)
        source = recover_source_quad_from_sarn_alignment(
            aligned,
            crop_bbox_xyxy=bbox,
            output_hw=(height, width),
        )
        source_area = float(cv2.contourArea(source))
        source_area_fraction = source_area / float(height * width)
        if source_area_fraction < MIN_SOURCE_QUAD_AREA_FRACTION:
            raise _GeometryRejected("source_quad_area_fraction_too_small")
        destination = np.asarray(
            [
                [0.0, 0.0],
                [float(width - 1), 0.0],
                [float(width - 1), float(height - 1)],
                [0.0, float(height - 1)],
            ],
            dtype=np.float32,
        )
        try:
            homography = cv2.getPerspectiveTransform(source, destination)
        except cv2.error as exc:
            raise _GeometryRejected("homography_estimation_failed") from exc
        if not np.all(np.isfinite(homography)):
            raise _GeometryRejected("non_finite_homography")
        condition_number = float(np.linalg.cond(homography))
        if (
            not math.isfinite(condition_number)
            or condition_number > MAX_HOMOGRAPHY_CONDITION_NUMBER
        ):
            raise _GeometryRejected("ill_conditioned_homography")
        determinant = float(np.linalg.det(homography))
        if not math.isfinite(determinant) or abs(determinant) < 1.0e-12:
            raise _GeometryRejected("singular_homography")
    except _GeometryRejected as exc:
        return _fallback_result(
            image,
            reason=f"rectification_rejected:{exc}",
            confidence=confidence,
            bbox=bbox,
            aligned_quad=aligned_tuple,
        )

    rectified = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    source_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, np.rint(source).astype(np.int32), 1)
    rectified_mask = cv2.warpPerspective(
        source_mask,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_mask = np.ascontiguousarray(
        (rectified_mask > 0).astype(np.float32, copy=False)
    )
    metadata = QuadRectificationMetadata(
        protocol=PROTOCOL,
        applied=True,
        fallback_reason=None,
        input_hw=(height, width),
        output_hw=(height, width),
        sarn_crop_bbox_xyxy=tuple(int(value) for value in bbox),
        sarn_confidence=confidence,
        aligned_quad_xy=aligned_tuple,
        source_quad_xy=_quad_tuple(source),
        destination_quad_xy=_quad_tuple(destination),
        homography=tuple(
            tuple(float(value) for value in row) for row in homography
        ),
        source_quad_area_fraction=source_area_fraction,
        homography_condition_number=condition_number,
        pixel_center_inverse=PIXEL_CENTER_FORMULA,
        rgb_resampling="cv2.INTER_LINEAR; one RGB warp from original degraded image",
        mask_resampling="cv2.INTER_NEAREST",
    )
    return QuadRectificationResult(
        image=np.ascontiguousarray(rectified),
        valid_support_mask=rectified_mask,
        metadata=metadata,
    )


__all__ = [
    "PIXEL_CENTER_FORMULA",
    "PROTOCOL",
    "QuadRectificationMetadata",
    "QuadRectificationResult",
    "SARNDecisionLike",
    "SupportQuadRectificationError",
    "recover_source_quad_from_sarn_alignment",
    "rectify_sarn_support_quad",
]
