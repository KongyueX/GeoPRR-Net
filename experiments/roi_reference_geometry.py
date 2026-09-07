"""Reference geometry from the existing SyncG-trained four-keypoint reader.

Only pivot/start/end are used, including detection selection and confidence
checks. The pointer-tip output is not an input to DeepLab or VDN decoding.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


SOURCE_POSE_GEOMETRY_ROLE = (
    "SyncG-trained YOLO11s-Pose-4KP pivot/start/end on conditioned ROI pixels; "
    "predicted pointer tip excluded"
)


def source_pose_input(image_bgr: np.ndarray, *, image_size: int) -> np.ndarray:
    return np.ascontiguousarray(cv2.resize(
        image_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    ))


def extract_source_pose_references(
    result: Any, *, source_shape: tuple[int, ...], image_size: int,
    minimum_keypoint_confidence: float = 0.05,
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None, "no_pose_detection", {}
    xy = keypoints.xy.detach().cpu().numpy()
    if xy.ndim != 3 or xy.shape[1:] != (4, 2):
        return None, "unexpected_keypoint_shape", {}
    indices = [0, 2, 3]
    references = xy[:, indices, :]
    confidence = (
        np.ones(references.shape[:2], dtype=np.float32)
        if keypoints.conf is None
        else keypoints.conf.detach().cpu().numpy()[:, indices]
    )
    boxes = getattr(result, "boxes", None)
    box_confidence = (
        np.ones(len(xy), dtype=np.float32)
        if boxes is None or boxes.conf is None
        else boxes.conf.detach().cpu().numpy()
    )
    scores = box_confidence * confidence.mean(axis=1)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    selected = int(np.argmax(scores))
    points = references[selected].astype(np.float32)
    selected_confidence = confidence[selected]
    telemetry = {
        "detections": int(len(xy)),
        "box_confidence": float(box_confidence[selected]),
        "minimum_reference_confidence": float(selected_confidence.min()),
        "pointer_tip_used": False,
    }
    if not np.isfinite(points).all() or not np.isfinite(selected_confidence).all():
        return None, "non_finite_reference_pose", telemetry
    if float(selected_confidence.min()) < minimum_keypoint_confidence:
        return None, "low_reference_confidence", telemetry
    height, width = source_shape[:2]
    points *= np.asarray([width / image_size, height / image_size], dtype=np.float32)
    return points, None, telemetry
