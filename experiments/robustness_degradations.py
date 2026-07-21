"""Deterministic image degradations for the pointer-meter robustness protocol.

The transforms are deliberately applied only at evaluation time.  They keep
the image size and meter reading unchanged, and derive every stochastic choice
from ``sample_id`` plus a fixed seed so that all methods see exactly the same
input.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


ROBUSTNESS_PROTOCOL = "controlled_blur_perspective_v1"


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    blur_sigma_fraction: float = 0.0
    perspective_degrees: float = 0.0


DEGRADATION_SPECS: dict[str, DegradationSpec] = {
    "clean": DegradationSpec("clean"),
    "blur_moderate": DegradationSpec(
        "blur_moderate",
        blur_sigma_fraction=0.0015,
    ),
    "blur_severe": DegradationSpec(
        "blur_severe",
        blur_sigma_fraction=0.0030,
    ),
    "perspective_moderate": DegradationSpec(
        "perspective_moderate",
        perspective_degrees=25.0,
    ),
    "perspective_severe": DegradationSpec(
        "perspective_severe",
        perspective_degrees=45.0,
    ),
    "combined_severe": DegradationSpec(
        "combined_severe",
        blur_sigma_fraction=0.0030,
        perspective_degrees=45.0,
    ),
}


def degradation_names(*, include_clean: bool = True) -> tuple[str, ...]:
    names = tuple(DEGRADATION_SPECS)
    return names if include_clean else tuple(name for name in names if name != "clean")


def _sample_orientation(sample_id: str, seed: int) -> tuple[str, int]:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    axis = "yaw" if digest[0] % 2 == 0 else "pitch"
    sign = 1 if digest[1] % 2 == 0 else -1
    return axis, sign


def _projected_corners(
    width: int,
    height: int,
    degrees: float,
    axis: str,
    sign: int,
) -> np.ndarray:
    """Project an image plane after a virtual 3-D camera-relative rotation."""

    aspect = float(width) / float(max(height, 1))
    points = np.asarray(
        [
            [-aspect / 2.0, -0.5, 0.0],
            [aspect / 2.0, -0.5, 0.0],
            [aspect / 2.0, 0.5, 0.0],
            [-aspect / 2.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    angle = math.radians(float(degrees) * int(sign))
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "yaw":
        rotation = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float64,
        )
    elif axis == "pitch":
        rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported perspective axis: {axis}")

    rotated = points @ rotation.T
    camera_distance = max(2.75, aspect + 0.75)
    focal_length = camera_distance
    depth = camera_distance + rotated[:, 2]
    projected = np.column_stack(
        (
            focal_length * rotated[:, 0] / depth,
            focal_length * rotated[:, 1] / depth,
        )
    )

    # Fit the complete projected plane into the original canvas.  Uniform
    # scaling preserves the projective shape; the small margin avoids clipping.
    extent = np.ptp(projected, axis=0)
    margin_fraction = 0.03
    usable_width = max(1.0, (width - 1) * (1.0 - 2.0 * margin_fraction))
    usable_height = max(1.0, (height - 1) * (1.0 - 2.0 * margin_fraction))
    scale = min(
        usable_width / max(float(extent[0]), 1e-12),
        usable_height / max(float(extent[1]), 1e-12),
    )
    centered = (projected - np.mean(projected, axis=0, keepdims=True)) * scale
    centered[:, 0] += (width - 1) / 2.0
    centered[:, 1] += (height - 1) / 2.0
    return centered.astype(np.float32)


def _border_median(image: np.ndarray) -> tuple[int, ...]:
    border = np.concatenate(
        (image[0], image[-1], image[:, 0], image[:, -1]),
        axis=0,
    )
    median = np.median(border, axis=0)
    if image.ndim == 2:
        return (int(round(float(median))),)
    return tuple(int(round(float(value))) for value in np.ravel(median))


def apply_degradation(
    image: np.ndarray,
    condition: str,
    *,
    sample_id: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one frozen robustness condition and return audit metadata."""

    if condition not in DEGRADATION_SPECS:
        raise ValueError(
            f"unknown degradation {condition!r}; expected one of "
            f"{', '.join(DEGRADATION_SPECS)}"
        )
    if image is None or image.size == 0:
        raise ValueError("cannot degrade an empty image")

    spec = DEGRADATION_SPECS[condition]
    output = image.copy() if condition != "clean" else image
    height, width = image.shape[:2]
    metadata: dict[str, Any] = {
        "protocol": ROBUSTNESS_PROTOCOL,
        "condition": condition,
        "seed": int(seed),
        "parameters": asdict(spec),
    }

    if spec.perspective_degrees > 0.0:
        axis, sign = _sample_orientation(str(sample_id), int(seed))
        destination = _projected_corners(
            width,
            height,
            spec.perspective_degrees,
            axis,
            sign,
        )
        source = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        border_value = _border_median(image)
        output = cv2.warpPerspective(
            output,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        metadata["perspective"] = {
            "axis": axis,
            "sign": sign,
            "degrees": spec.perspective_degrees,
            "homography": homography.tolist(),
            "destination_corners": destination.tolist(),
        }

    if spec.blur_sigma_fraction > 0.0:
        sigma_pixels = max(
            0.1,
            float(spec.blur_sigma_fraction) * float(min(height, width)),
        )
        output = cv2.GaussianBlur(
            output,
            (0, 0),
            sigmaX=sigma_pixels,
            sigmaY=sigma_pixels,
            borderType=cv2.BORDER_REFLECT_101,
        )
        metadata["blur"] = {
            "kind": "gaussian",
            "sigma_fraction_of_short_side": spec.blur_sigma_fraction,
            "sigma_pixels": sigma_pixels,
        }

    return output, metadata

