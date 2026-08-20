"""Build the two runtime views used by projective geometry-guided SIAM.

View A is exactly the conservative SARN-v2 output consumed by the frozen
DB-GAR18 parent.  View B is the one-pass quadrilateral rectification produced
from the original degraded pixels.  The returned homography maps coordinates
from View A to View B; it never uses robustness metadata or a ground-truth
degradation transform.

If SARN or quadrilateral rectification declines a sample, both views are View
A, the homography is identity, and the fusion gate is forced to zero.  This
keeps the deployed fallback identical to DB-GAR18 + SARN-v2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import cv2

from experiments import support_aware_roi_normalization_v2 as sarn_v2
from experiments.support_quad_rectification import (
    QuadRectificationResult,
    rectify_sarn_support_quad,
)


PROTOCOL: Final[str] = "projective_geometry_dual_view_v1"


@dataclass(frozen=True)
class ProjectiveGeometryViews:
    view_a_bgr: np.ndarray
    view_b_bgr: np.ndarray
    support_mask_a: np.ndarray
    homography_a_to_b: np.ndarray
    confidence: float
    active: bool
    fallback_reason: str | None
    sarn_decision: sarn_v2.SARNv2Result
    rectification: QuadRectificationResult


def sarn_source_to_aligned_homography(
    *,
    crop_bbox_xyxy: tuple[int, int, int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Return SARN's source-canvas to crop-resized pixel-centre transform."""

    height, width = int(output_hw[0]), int(output_hw[1])
    left, top, right, bottom = (int(value) for value in crop_bbox_xyxy)
    if height < 1 or width < 1:
        raise ValueError("output shape must be positive")
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("crop bbox must be a valid half-open box inside the image")
    scale_x = float(width) / float(right - left)
    scale_y = float(height) / float(bottom - top)
    transform = np.asarray(
        [
            [scale_x, 0.0, (-float(left) + 0.5) * scale_x - 0.5],
            [0.0, scale_y, (-float(top) + 0.5) * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform


def _identity_fallback(
    *,
    view_a: np.ndarray,
    decision: sarn_v2.SARNv2Result,
    rectification: QuadRectificationResult,
    reason: str,
) -> ProjectiveGeometryViews:
    height, width = view_a.shape[:2]
    return ProjectiveGeometryViews(
        view_a_bgr=view_a,
        view_b_bgr=view_a,
        support_mask_a=np.ones((height, width), dtype=np.float32),
        homography_a_to_b=np.eye(3, dtype=np.float64),
        confidence=0.0,
        active=False,
        fallback_reason=str(reason),
        sarn_decision=decision,
        rectification=rectification,
    )


def build_projective_geometry_views(
    degraded_image_bgr: np.ndarray,
) -> ProjectiveGeometryViews:
    """Build SARN-bbox and rectified views from one degraded uint8 BGR ROI."""

    degraded = np.asarray(degraded_image_bgr)
    if not (
        degraded.dtype == np.uint8
        and degraded.ndim == 3
        and degraded.shape[2] == 3
        and min(degraded.shape[:2]) >= 2
    ):
        raise ValueError("input must be uint8 BGR [H,W,3] with both sides >= 2")
    degraded = np.ascontiguousarray(degraded)
    decision = sarn_v2.normalize_support_aware_roi_v2(degraded)
    view_a = np.ascontiguousarray(decision.image)
    rectification = rectify_sarn_support_quad(degraded, decision)
    if not decision.applied:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason=f"sarn_not_applied:{decision.fallback_reason or 'unspecified'}",
        )
    if not rectification.applied:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason=rectification.metadata.fallback_reason or "rectification_not_applied",
        )
    if decision.bbox_xyxy is None or decision.valid_support_mask is None:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="applied_sarn_missing_runtime_geometry",
        )
    raw_homography = rectification.metadata.homography
    if raw_homography is None:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="applied_rectification_missing_homography",
        )

    height, width = view_a.shape[:2]
    source_to_a = sarn_source_to_aligned_homography(
        crop_bbox_xyxy=decision.bbox_xyxy,
        output_hw=(height, width),
    )
    source_to_b = np.asarray(raw_homography, dtype=np.float64).reshape(3, 3)
    try:
        a_to_b = source_to_b @ np.linalg.inv(source_to_a)
    except np.linalg.LinAlgError:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="singular_sarn_alignment_transform",
        )
    if not np.all(np.isfinite(a_to_b)):
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="non_finite_a_to_b_homography",
        )
    normalization = float(a_to_b[2, 2])
    if not math.isfinite(normalization) or abs(normalization) < 1.0e-12:
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="invalid_a_to_b_homography_scale",
        )
    a_to_b = np.ascontiguousarray(a_to_b / normalization)
    support_mask = np.asarray(decision.valid_support_mask, dtype=np.float32)
    if support_mask.shape != (height, width) or not np.all(np.isfinite(support_mask)):
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="invalid_sarn_support_mask",
        )
    support_mask = np.ascontiguousarray(np.clip(support_mask, 0.0, 1.0))
    confidence = float(np.clip(float(decision.confidence), 0.0, 1.0))
    if not math.isfinite(confidence):
        return _identity_fallback(
            view_a=view_a,
            decision=decision,
            rectification=rectification,
            reason="non_finite_sarn_confidence",
        )
    return ProjectiveGeometryViews(
        view_a_bgr=view_a,
        view_b_bgr=np.ascontiguousarray(rectification.image),
        support_mask_a=support_mask,
        homography_a_to_b=a_to_b,
        confidence=confidence,
        active=True,
        fallback_reason=None,
        sarn_decision=decision,
        rectification=rectification,
    )


def resize_projective_geometry_views(
    views: ProjectiveGeometryViews,
    *,
    output_hw: tuple[int, int],
) -> ProjectiveGeometryViews:
    """Resize both views and conjugate their pixel-center homography.

    This helper is needed when canonical ROIs are not already the network's
    input size.  OpenCV's resize mapping is represented explicitly, so the
    feature alignment never silently mixes native-image and network pixels.
    """

    output_height, output_width = int(output_hw[0]), int(output_hw[1])
    input_height, input_width = views.view_a_bgr.shape[:2]
    if min(output_height, output_width) < 2:
        raise ValueError("resized view dimensions must both be at least two")
    if views.view_b_bgr.shape[:2] != (input_height, input_width):
        raise ValueError("two-view native shapes differ")
    if (input_height, input_width) == (output_height, output_width):
        return views

    def resize_matrix(
        source_hw: tuple[int, int], destination_hw: tuple[int, int]
    ) -> np.ndarray:
        source_height, source_width = source_hw
        destination_height, destination_width = destination_hw
        scale_x = float(destination_width) / float(source_width)
        scale_y = float(destination_height) / float(source_height)
        return np.asarray(
            [
                [scale_x, 0.0, 0.5 * scale_x - 0.5],
                [0.0, scale_y, 0.5 * scale_y - 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    source_to_output = resize_matrix(
        (input_height, input_width), (output_height, output_width)
    )
    resized_homography = (
        source_to_output
        @ np.asarray(views.homography_a_to_b, dtype=np.float64)
        @ np.linalg.inv(source_to_output)
    )
    resized_homography /= resized_homography[2, 2]
    view_a = cv2.resize(
        views.view_a_bgr,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )
    view_b = cv2.resize(
        views.view_b_bgr,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )
    support_mask = cv2.resize(
        views.support_mask_a,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )
    return ProjectiveGeometryViews(
        view_a_bgr=np.ascontiguousarray(view_a),
        view_b_bgr=np.ascontiguousarray(view_b),
        support_mask_a=np.ascontiguousarray(
            np.clip(support_mask.astype(np.float32, copy=False), 0.0, 1.0)
        ),
        homography_a_to_b=np.ascontiguousarray(resized_homography),
        confidence=views.confidence,
        active=views.active,
        fallback_reason=views.fallback_reason,
        sarn_decision=views.sarn_decision,
        rectification=views.rectification,
    )


__all__ = [
    "PROTOCOL",
    "ProjectiveGeometryViews",
    "build_projective_geometry_views",
    "resize_projective_geometry_views",
    "sarn_source_to_aligned_homography",
]
