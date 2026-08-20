"""Deterministic out-of-family stressors for the A15.2-METT evaluation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final

import cv2
import numpy as np

from experiments.robustness_degradations import (
    _border_median,
    _projected_corners,
    _sample_orientation,
)


PROTOCOL: Final[str] = "mett_continuous_perspective_and_corruption_sweep_v1"
STRESS_SEED: Final[int] = 20_260_818


@dataclass(frozen=True)
class StressSpec:
    name: str
    family: str
    severity: float
    perspective_degrees: float = 0.0
    noise_standard_deviation: float = 0.0
    brightness_factor: float = 1.0
    jpeg_quality: int = 100
    occlusion_area_fraction: float = 0.0


STRESS_SPECS: Final[dict[str, StressSpec]] = {
    spec.name: spec
    for spec in (
        StressSpec("clean", "clean", 0.0),
        StressSpec("perspective_15", "perspective", 0.25, perspective_degrees=15.0),
        StressSpec("perspective_30", "perspective", 0.50, perspective_degrees=30.0),
        StressSpec("perspective_45", "perspective", 0.75, perspective_degrees=45.0),
        StressSpec("perspective_60", "perspective", 1.00, perspective_degrees=60.0),
        StressSpec("noise_001", "sensor_noise", 0.20, noise_standard_deviation=0.01),
        StressSpec("noise_003", "sensor_noise", 0.60, noise_standard_deviation=0.03),
        StressSpec("noise_005", "sensor_noise", 1.00, noise_standard_deviation=0.05),
        StressSpec("brightness_060", "brightness", 1.00, brightness_factor=0.60),
        StressSpec("brightness_080", "brightness", 0.50, brightness_factor=0.80),
        StressSpec("brightness_120", "brightness", 0.50, brightness_factor=1.20),
        StressSpec("brightness_140", "brightness", 1.00, brightness_factor=1.40),
        StressSpec("jpeg_80", "jpeg", 0.25, jpeg_quality=80),
        StressSpec("jpeg_50", "jpeg", 0.625, jpeg_quality=50),
        StressSpec("jpeg_20", "jpeg", 1.00, jpeg_quality=20),
        StressSpec("occlusion_005", "occlusion", 0.25, occlusion_area_fraction=0.05),
        StressSpec("occlusion_010", "occlusion", 0.50, occlusion_area_fraction=0.10),
        StressSpec("occlusion_020", "occlusion", 1.00, occlusion_area_fraction=0.20),
    )
}
MAX_OCCLUSION_AREA_FRACTION: Final[float] = max(
    spec.occlusion_area_fraction for spec in STRESS_SPECS.values()
)


def _stable_sample_seed(sample_id: str, seed: int) -> int:
    # A small deterministic integer mixer is sufficient for paired corruption;
    # it is not used as an identity, contract, or integrity hash.
    value = int(seed) & 0xFFFFFFFF
    for index, character in enumerate(str(sample_id), 1):
        value = (value + index * ord(character) * 2_654_435_761) & 0xFFFFFFFF
    return value


def apply_stress(
    image: np.ndarray,
    spec: StressSpec,
    *,
    sample_id: str,
    seed: int = STRESS_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one paired stressor without changing image dimensions."""

    if image is None or image.size == 0:
        raise ValueError("cannot stress an empty image")
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("METT stress input must be an 8-bit image")
    height, width = image.shape[:2]
    output = image.copy()
    local_seed = _stable_sample_seed(sample_id, seed)
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL,
        "seed": int(seed),
        "sample_seed": int(local_seed),
        "spec": asdict(spec),
    }

    if spec.perspective_degrees > 0.0:
        axis, sign = _sample_orientation(sample_id, seed)
        source = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        destination = _projected_corners(
            width, height, spec.perspective_degrees, axis, sign
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        output = cv2.warpPerspective(
            output,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_border_median(image),
        )
        metadata["perspective"] = {"axis": axis, "sign": sign}

    if spec.noise_standard_deviation > 0.0:
        generator = np.random.default_rng(local_seed)
        noise = generator.normal(
            0.0,
            float(spec.noise_standard_deviation) * 255.0,
            size=output.shape,
        )
        output = np.clip(output.astype(np.float32) + noise, 0.0, 255.0).astype(
            np.uint8
        )

    if spec.brightness_factor != 1.0:
        if not math.isfinite(spec.brightness_factor) or spec.brightness_factor <= 0.0:
            raise ValueError("brightness factor must be positive and finite")
        output = np.clip(
            output.astype(np.float32) * float(spec.brightness_factor),
            0.0,
            255.0,
        ).astype(np.uint8)

    if spec.jpeg_quality < 100:
        if not 1 <= int(spec.jpeg_quality) <= 100:
            raise ValueError("JPEG quality is outside [1, 100]")
        encoded, payload = cv2.imencode(
            ".jpg", output, (cv2.IMWRITE_JPEG_QUALITY, int(spec.jpeg_quality))
        )
        if not encoded:
            raise ValueError("JPEG stress encoding failed")
        decoded = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("JPEG stress decoding failed")
        output = decoded

    if spec.occlusion_area_fraction > 0.0:
        fraction = float(spec.occlusion_area_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("occlusion area fraction is outside (0, 1)")
        side_fraction = math.sqrt(fraction)
        box_width = max(1, min(width, int(round(width * side_fraction))))
        box_height = max(1, min(height, int(round(height * side_fraction))))
        maximum_side_fraction = math.sqrt(MAX_OCCLUSION_AREA_FRACTION)
        maximum_box_width = max(
            1, min(width, int(round(width * maximum_side_fraction)))
        )
        maximum_box_height = max(
            1, min(height, int(round(height * maximum_side_fraction)))
        )
        minimum_center_x = maximum_box_width // 2
        maximum_center_x = width - (maximum_box_width - minimum_center_x)
        minimum_center_y = maximum_box_height // 2
        maximum_center_y = height - (maximum_box_height - minimum_center_y)
        center_x = minimum_center_x + local_seed % (
            maximum_center_x - minimum_center_x + 1
        )
        center_y = minimum_center_y + (local_seed // 65_537) % (
            maximum_center_y - minimum_center_y + 1
        )
        left = center_x - box_width // 2
        top = center_y - box_height // 2
        fill = _border_median(output)
        output[top : top + box_height, left : left + box_width] = fill
        metadata["occlusion_center_xy"] = [int(center_x), int(center_y)]
        metadata["occlusion_box"] = [
            int(left),
            int(top),
            int(left + box_width),
            int(top + box_height),
        ]

    return np.ascontiguousarray(output), metadata


__all__ = [
    "PROTOCOL",
    "STRESS_SEED",
    "STRESS_SPECS",
    "StressSpec",
    "apply_stress",
]
