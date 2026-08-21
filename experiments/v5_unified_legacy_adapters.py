"""Strict canonical-ROI legacy adapters for the unified V5 blind evaluator.

Strict component inference accepts one already-canonicalized *meter ROI* with
globally frozen detector thresholds.  It returns normalized progress only; physical
scale endpoints are joined later by the scorer or a separate production
wrapper and are not accepted at this inference boundary.
Accordingly these adapters are progress diagnostics, not complete automatic
reading methods for the primary paper table.
The meter detector is never invoked and no second crop or correction is
performed.  The canonical meter box is attested as ``[0, 0, width, height]``.

Within that unchanged ROI both adapters run the frozen pointer segmenter.  The
paper-evaluation Original Transformer adapter interprets its 101 classes as
absolute orientation bins, then runs the frozen automatic start/end ScaleMark
detector to convert that orientation to normalized dial progress.  Base
Mask-Geometry uses the same automatic reference requirement.  Both reject
single-endpoint and default-angle fallbacks.

No public ``predict`` method accepts ground truth, manual ScaleMark positions,
start/range angles, or a reference packet.  A full-scene meter-detector run or
a shared automatic-reference run must be a separately named secondary study
and cannot be emitted by these primary adapters.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image


PROTOCOL = "v5_unified_legacy_canonical_roi_progress_only_v5"
AUTOMATIC_REFERENCE_PROTOCOL = "v5_canonical_roi_automatic_reference_v1"
EVIDENCE_ROLE = "secondary_progress_component_diagnostic"
OUTPUT_MODE = "progress_only_component"

TASK_CONFIG_SCHEMA = {
    "schema_version": 1,
    "allowed": [
        "confidence",
        "start_end_distance_threshold",
        "start_end_position",
        "snap_out_of_range_pointer",
        "validate_mask_line",
        "mask_center_threshold_ratio",
    ],
    "forbidden": [
        "ground_truth",
        "target",
        "label",
        "actual",
        "scale_start",
        "scale_end",
        "scalemark",
        "start_angle",
        "range_angle",
        "reference_packet",
        "meter_bbox",
        "correction_mode",
    ],
}
OUTPUT_SCHEMA = {
    "schema_version": 5,
    "required": [
        "protocol",
        "comparison_tier",
        "evidence_role",
        "output_mode",
        "eligible_for_primary_metrics",
        "method",
        "status",
        "prediction_space",
        "prediction",
        "progress",
        "failure",
        "input_attestation",
        "automatic_reference",
        "component_sha256",
        "telemetry",
        "output_schema_sha256",
        "adapter_source_sha256",
        "record_sha256",
    ],
    "primary_input": "canonical meter ROI plus frozen non-scale task configuration only",
    "output": "normalized progress in [0,1]",
    "interpretation": (
        "progress component diagnostic only; no automatic numeric scale range is "
        "predicted, so this is not a complete automatic-reading primary result"
    ),
    "crop_rule": "meter detector not invoked; no second crop/correction",
    "reference_rule": (
        "paper-evaluation Original Transformer and Base Mask-Geometry require both "
        "ScaleMark endpoints detected automatically inside ROI; the separately named "
        "historical native-class adapter is retained for compatibility only"
    ),
    "prediction_space": "normalized_progress_0_1",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


TASK_CONFIG_SCHEMA_SHA256 = canonical_sha256(TASK_CONFIG_SCHEMA)
OUTPUT_SCHEMA_SHA256 = canonical_sha256(OUTPUT_SCHEMA)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_INPUT_KEYS = {
    "actual",
    "correctionmode",
    "groundtruth",
    "gt",
    "label",
    "labels",
    "manualreference",
    "meterbbox",
    "rangeangle",
    "reference",
    "referencepacket",
    "scaleend",
    "scalestart",
    "scalemark",
    "scalemarkpositions",
    "startangle",
    "target",
    "truth",
}
_CONFIG_KEYS = set(TASK_CONFIG_SCHEMA["allowed"])


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _reject_supervision(value: Any, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"forbidden supervision/crop/reference field at {path}.{key}")
            _reject_supervision(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_supervision(nested, path=f"{path}[{index}]")


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def sha256_file(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("canonical_meter_roi_bgr must be a NumPy array")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("canonical meter ROI must have shape [H,W,3] and dtype uint8")
    if image_bgr.shape[0] < 16 or image_bgr.shape[1] < 16:
        raise ValueError("canonical meter ROI is too small")
    return np.ascontiguousarray(image_bgr)


def image_sha256(image_bgr: np.ndarray) -> str:
    image = _validate_image(image_bgr)
    header = _canonical_json_bytes(
        {
            "protocol": "canonical_meter_roi_bgr_uint8_v1",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "channel_order": "BGR",
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenLegacyTaskConfig:
    """Only non-label task settings accepted by primary blind inference."""

    confidence: float | None = None
    start_end_distance_threshold: float | None = None
    start_end_position: str = "start_left_end_right"
    snap_out_of_range_pointer: bool = True
    validate_mask_line: bool = True
    mask_center_threshold_ratio: float = 0.10

    @classmethod
    def from_value(
        cls, value: "FrozenLegacyTaskConfig | Mapping[str, Any] | None"
    ) -> "FrozenLegacyTaskConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            _reject_supervision(value, path="task_config")
            unknown = set(value) - _CONFIG_KEYS
            if unknown:
                raise ValueError(f"unsupported task configuration fields: {sorted(unknown)}")
            result = cls(**dict(value))
        else:
            raise TypeError("task_config must be FrozenLegacyTaskConfig, mapping, or None")
        if result.confidence is not None and not math.isfinite(float(result.confidence)):
            raise ValueError("confidence must be finite or null")
        if result.start_end_distance_threshold is not None and not math.isfinite(
            float(result.start_end_distance_threshold)
        ):
            raise ValueError("start/end distance threshold must be finite or null")
        ratio = float(result.mask_center_threshold_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio < 0.5:
            raise ValueError("mask_center_threshold_ratio must be in (0, 0.5)")
        return result

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class RuntimeSnapshot:
    returned_end_num: float | None
    returned_prediction: float | None
    selected: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    error_code: str | None


class FrozenCanonicalROILegacyRuntime:
    """Run frozen legacy components directly on an unchanged canonical ROI."""

    REQUIRED_COMPONENTS = (
        "pointer_segmentation",
        "original_transformer",
        "reference_detector",
    )

    def __init__(
        self,
        runtime: Any,
        *,
        component_sha256: Mapping[str, str],
    ) -> None:
        required_attributes = (
            "pointerSeg",
            "vlmMeter",
            "pointerDetect",
            "label_text",
            "_find_largest_component",
            "_is_pointer_mask_line_valid",
            "_normalize_start_end_points",
            "calculate_angle",
            "_calculate_meter_result",
            "_build_geometry_direct_reading",
            "_build_geometry_direct_v2_reading",
            "_build_geometry_weighted_fusion_reading",
        )
        missing_attributes = [name for name in required_attributes if not hasattr(runtime, name)]
        if missing_attributes:
            raise TypeError(f"legacy runtime is missing attributes: {missing_attributes}")
        unknown = set(component_sha256) - set(self.REQUIRED_COMPONENTS)
        missing = set(self.REQUIRED_COMPONENTS) - set(component_sha256)
        if unknown or missing:
            raise ValueError(
                f"runtime component roster drift: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        self.runtime = runtime
        self.component_sha256 = {
            name: _require_sha256(component_sha256[name], name=f"{name} SHA-256")
            for name in self.REQUIRED_COMPONENTS
        }
        self._lock = threading.Lock()
        self._cached_roi_sha256: str | None = None
        self._cached_mask: np.ndarray | None = None

    @classmethod
    def from_checkpoints(
        cls,
        *,
        pointer_segmentation: Path,
        original_transformer: Path,
        reference_detector: Path,
        device: str | torch.device,
    ) -> "FrozenCanonicalROILegacyRuntime":
        from utils.angleDetect.detect import meterFormer
        from utils.angleDetect.pointerSeg.detectSeg import u2netpSeg
        from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel
        from utils.angleDetect.zeroShotMeter import meterZeroShot

        paths = {
            "pointer_segmentation": Path(pointer_segmentation).resolve(strict=True),
            "original_transformer": Path(original_transformer).resolve(strict=True),
            "reference_detector": Path(reference_detector).resolve(strict=True),
        }
        torch_device = torch.device(device)
        runtime = meterZeroShot.__new__(meterZeroShot)
        runtime.device = torch_device
        runtime.pointerSeg = u2netpSeg(str(paths["pointer_segmentation"]), torch_device)
        runtime.vlmMeter = meterFormer(str(paths["original_transformer"]), torch_device)
        runtime.pointerDetect = targetDetectModel(str(paths["reference_detector"]))
        runtime.label_text_list = list(map(str, range(101)))
        runtime.label_text = runtime.vlmMeter.model.texteconder.tokenize(
            runtime.label_text_list
        ).to(torch_device)
        runtime.ornMeterNum = 155.25
        runtime.oneTempAngle = 360.0 / 101.0
        return cls(
            runtime,
            component_sha256={name: sha256_file(path) for name, path in paths.items()},
        )

    def _pointer_mask(self, image: np.ndarray, roi_sha256: str) -> np.ndarray:
        if self._cached_roi_sha256 == roi_sha256 and self._cached_mask is not None:
            return self._cached_mask.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.asarray(self.runtime.pointerSeg.Inference(Image.fromarray(rgb)))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError("pointer segmenter changed the canonical ROI geometry")
        mask = np.ascontiguousarray(self.runtime._find_largest_component(mask.astype(np.uint8)))
        self._cached_roi_sha256 = roi_sha256
        self._cached_mask = mask.copy()
        return mask

    def infer(
        self,
        image_bgr: np.ndarray,
        config: FrozenLegacyTaskConfig,
        *,
        reading_backend: str,
    ) -> RuntimeSnapshot:
        with self._lock:
            image = _validate_image(image_bgr)
            roi_sha256 = image_sha256(image)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            mask = self._pointer_mask(image, roi_sha256)
            mask_line_valid = bool(
                self.runtime._is_pointer_mask_line_valid(
                    mask, center, float(config.mask_center_threshold_ratio)
                )
            )

            common_artifacts = {
                "meter_bbox": [0.0, 0.0, float(width), float(height)],
                "meter_detector_invoked": False,
                "mask_line_valid": mask_line_valid,
                "mask_validation_failed": bool(
                    config.validate_mask_line and not mask_line_valid
                ),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "mask_foreground_ratio": float(np.mean(mask > 0)),
            }
            if reading_backend == "transformer_automatic_reference":
                # Match zeroShotMeter's deployed path exactly: the canonical BGR
                # crop is first converted to RGB for PIL/segmentation, then that
                # RGB array is passed through OpenCV's BGR2GRAY conversion before
                # the automatic start/end detector.
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                detector_image = np.stack((gray,) * 3, axis=-1)
                raw_end, _ = self.runtime.pointerDetect.center_find(
                    detector_image, classId=1, confidence=config.confidence
                )
                raw_start, _ = self.runtime.pointerDetect.center_find(
                    detector_image, classId=2, confidence=config.confidence
                )
                raw_start = None if raw_start is None else tuple(map(float, raw_start))
                raw_end = None if raw_end is None else tuple(map(float, raw_end))
                if raw_start is not None and raw_end is not None:
                    normalized_start, normalized_end = self.runtime._normalize_start_end_points(
                        raw_start,
                        raw_end,
                        width,
                        config.start_end_distance_threshold,
                        config.start_end_position,
                    )
                elif raw_start is not None:
                    normalized_start, normalized_end = raw_start, None
                elif raw_end is not None:
                    normalized_start, normalized_end = None, raw_end
                else:
                    normalized_start = normalized_end = None
                if normalized_start is not None and normalized_end is not None:
                    branch = "start_and_end"
                elif normalized_start is not None:
                    branch = "start_only"
                elif normalized_end is not None:
                    branch = "end_only"
                else:
                    branch = "default_start_end"

                start_angle = (
                    None
                    if normalized_start is None
                    else float(self.runtime.calculate_angle(center, normalized_start))
                )
                end_angle = (
                    None
                    if normalized_end is None
                    else float(self.runtime.calculate_angle(center, normalized_end))
                )
                range_angle = (
                    None
                    if start_angle is None or end_angle is None
                    else float((end_angle - start_angle) % 360.0)
                )
                artifacts = {
                    **common_artifacts,
                    "branch": branch,
                    "reference_mode": "automatic_explicit",
                    "reference_detector_invoked": True,
                    "center_start": normalized_start,
                    "center_end": normalized_end,
                    "startAngle": start_angle,
                    "endAngle": end_angle,
                    "disAngle": range_angle,
                }
                if (
                    branch != "start_and_end"
                    or range_angle is None
                    or not 0.0 < range_angle < 360.0
                ):
                    return RuntimeSnapshot(
                        None, None, {}, artifacts, "automatic_reference_incomplete"
                    )
                if bool(config.validate_mask_line) and not mask_line_valid:
                    return RuntimeSnapshot(
                        None, None, {}, artifacts, "pointer_mask_line_invalid"
                    )

                pointer_image = Image.fromarray(mask).convert("L")
                meter_image = Image.fromarray(rgb).convert("L")
                with torch.inference_mode():
                    result = self.runtime.vlmMeter.Inference(
                        pointer_image, meter_image, self.runtime.label_text
                    )
                logits = result[0] if isinstance(result, (tuple, list)) else result
                logits = torch.as_tensor(logits).detach().float()
                if logits.numel() != 101 or not bool(torch.isfinite(logits).all()):
                    raise ValueError(
                        "Original Transformer returned invalid 101-class logits"
                    )
                class_index = int(
                    torch.argmax(logits.reshape(1, 101), dim=1).item()
                )
                start_angle = float(start_angle)
                range_angle = float(range_angle)
                meter_num = (float(self.runtime.ornMeterNum) - start_angle) % 360.0
                progress = _finite_optional(
                    self.runtime._calculate_meter_result(
                        float(class_index),
                        meter_num,
                        range_angle,
                        0.0,
                        1.0,
                        bool(config.snap_out_of_range_pointer),
                    )
                )
                if progress is None:
                    raise ValueError(
                        "Original Transformer orientation conversion returned a non-finite value"
                    )
                pointer_angle = (
                    float(self.runtime.oneTempAngle) * float(class_index) + meter_num
                ) % 360.0
                selected = {
                    "status": True,
                    "backend": "transformer_automatic_reference",
                    "resultNum": progress,
                    "progress_ratio": progress,
                    "endNum_float": float(class_index),
                    "pointer_angle": pointer_angle,
                    "reference_mode": "automatic_explicit",
                }
                return RuntimeSnapshot(
                    float(class_index), progress, selected, artifacts, None
                )
            if reading_backend == "transformer":
                artifacts = {
                    **common_artifacts,
                    "branch": "implicit_native",
                    "reference_mode": "implicit_native",
                    "reference_detector_invoked": False,
                    "center_start": None,
                    "center_end": None,
                    "startAngle": None,
                    "endAngle": None,
                    "disAngle": None,
                }
                if bool(config.validate_mask_line) and not mask_line_valid:
                    return RuntimeSnapshot(
                        None, None, {}, artifacts, "pointer_mask_line_invalid"
                    )
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pointer_image = Image.fromarray(mask).convert("L")
                meter_image = Image.fromarray(rgb).convert("L")
                with torch.inference_mode():
                    result = self.runtime.vlmMeter.Inference(
                        pointer_image, meter_image, self.runtime.label_text
                    )
                logits = result[0] if isinstance(result, (tuple, list)) else result
                logits = torch.as_tensor(logits).detach().float()
                if logits.numel() != 101 or not bool(torch.isfinite(logits).all()):
                    raise ValueError(
                        "Original Transformer returned invalid 101-class logits"
                    )
                class_index = int(
                    torch.argmax(logits.reshape(1, 101), dim=1).item()
                )
                progress = float(class_index / 100.0)
                prediction = progress
                selected = {
                    "status": True,
                    "backend": "transformer",
                    "resultNum": prediction,
                    "progress_ratio": progress,
                    "endNum_float": float(class_index),
                    "reference_mode": "implicit_native",
                }
                return RuntimeSnapshot(
                    float(class_index), prediction, selected, artifacts, None
                )
            if reading_backend != "geometry_fusion_weighted":
                raise ValueError(
                    f"unsupported canonical-ROI backend: {reading_backend}"
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector_image = np.stack((gray,) * 3, axis=-1)
            raw_end, _ = self.runtime.pointerDetect.center_find(
                detector_image, classId=1, confidence=config.confidence
            )
            raw_start, _ = self.runtime.pointerDetect.center_find(
                detector_image, classId=2, confidence=config.confidence
            )
            raw_start = None if raw_start is None else tuple(map(float, raw_start))
            raw_end = None if raw_end is None else tuple(map(float, raw_end))
            if raw_start is not None and raw_end is not None:
                normalized_start, normalized_end = self.runtime._normalize_start_end_points(
                    raw_start,
                    raw_end,
                    width,
                    config.start_end_distance_threshold,
                    config.start_end_position,
                )
                branch = "start_and_end"
            elif raw_start is not None:
                normalized_start, normalized_end = raw_start, None
                branch = "start_only"
            elif raw_end is not None:
                normalized_start, normalized_end = None, raw_end
                branch = "end_only"
            else:
                normalized_start = normalized_end = None
                branch = "default_start_end"

            start_angle = (
                None
                if normalized_start is None
                else float(self.runtime.calculate_angle(center, normalized_start))
            )
            end_angle = (
                None
                if normalized_end is None
                else float(self.runtime.calculate_angle(center, normalized_end))
            )
            range_angle = (
                None
                if start_angle is None or end_angle is None
                else float((end_angle - start_angle) % 360.0)
            )
            artifacts = {
                **common_artifacts,
                "branch": branch,
                "reference_mode": "automatic_explicit",
                "reference_detector_invoked": True,
                "center_start": normalized_start,
                "center_end": normalized_end,
                "startAngle": start_angle,
                "endAngle": end_angle,
                "disAngle": range_angle,
            }
            if branch != "start_and_end" or range_angle is None or not 0.0 < range_angle < 360.0:
                return RuntimeSnapshot(None, None, {}, artifacts, "automatic_reference_incomplete")
            if bool(config.validate_mask_line) and not mask_line_valid:
                return RuntimeSnapshot(None, None, {}, artifacts, "pointer_mask_line_invalid")

            start_angle = float(start_angle)
            range_angle = float(range_angle)
            first = self.runtime._build_geometry_direct_reading(
                mask,
                center,
                start_angle,
                range_angle,
                0.0,
                1.0,
                bool(config.snap_out_of_range_pointer),
                float(config.mask_center_threshold_ratio),
            )
            second = self.runtime._build_geometry_direct_v2_reading(
                mask,
                center,
                start_angle,
                range_angle,
                0.0,
                1.0,
                bool(config.snap_out_of_range_pointer),
                float(config.mask_center_threshold_ratio),
            )
            selected = dict(
                self.runtime._build_geometry_weighted_fusion_reading(first, second)
            )
            prediction = _finite_optional(selected.get("resultNum"))
            end_num = _finite_optional(selected.get("endNum_float"))
            return RuntimeSnapshot(end_num, prediction, selected, artifacts, None)


class FrozenBaseMaskGeometryRuntime(FrozenCanonicalROILegacyRuntime):
    """Minimal frozen runtime for Base Mask-Geometry only.

    Unlike :class:`FrozenCanonicalROILegacyRuntime`, this runtime's formal
    component roster contains only the pointer segmenter and automatic
    start/end reference detector.  It never constructs a Transformer model,
    never reads a Transformer checkpoint, and rejects the Transformer backend
    if it is requested accidentally.
    """

    REQUIRED_COMPONENTS = (
        "pointer_segmentation",
        "reference_detector",
    )
    RUNTIME_MODE = "base_mask_geometry_only_no_transformer"

    def __init__(
        self,
        runtime: Any,
        *,
        component_sha256: Mapping[str, str],
    ) -> None:
        required_attributes = (
            "pointerSeg",
            "pointerDetect",
            "_find_largest_component",
            "_is_pointer_mask_line_valid",
            "_normalize_start_end_points",
            "calculate_angle",
            "_calculate_meter_result",
            "_build_geometry_direct_reading",
            "_build_geometry_direct_v2_reading",
            "_build_geometry_weighted_fusion_reading",
        )
        missing_attributes = [
            name for name in required_attributes if not hasattr(runtime, name)
        ]
        if missing_attributes:
            raise TypeError(
                f"Base-only legacy runtime is missing attributes: {missing_attributes}"
            )
        unknown = set(component_sha256) - set(self.REQUIRED_COMPONENTS)
        missing = set(self.REQUIRED_COMPONENTS) - set(component_sha256)
        if unknown or missing:
            raise ValueError(
                "Base-only runtime component roster drift: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        self.runtime = runtime
        self.component_sha256 = {
            name: _require_sha256(
                component_sha256[name], name=f"{name} SHA-256"
            )
            for name in self.REQUIRED_COMPONENTS
        }
        self._lock = threading.Lock()
        self._cached_roi_sha256: str | None = None
        self._cached_mask: np.ndarray | None = None

    @classmethod
    def from_checkpoints(
        cls,
        *,
        pointer_segmentation: Path,
        reference_detector: Path,
        device: str | torch.device,
    ) -> "FrozenBaseMaskGeometryRuntime":
        """Load exactly the two components consumed by Base Mask-Geometry."""

        from utils.angleDetect.pointerSeg.detectSeg import u2netpSeg
        from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel
        from utils.angleDetect.zeroShotMeter import meterZeroShot

        paths = {
            "pointer_segmentation": Path(pointer_segmentation).resolve(strict=True),
            "reference_detector": Path(reference_detector).resolve(strict=True),
        }
        torch_device = torch.device(device)
        runtime = meterZeroShot.__new__(meterZeroShot)
        runtime.device = torch_device
        runtime.pointerSeg = u2netpSeg(str(paths["pointer_segmentation"]), torch_device)
        runtime.pointerDetect = targetDetectModel(str(paths["reference_detector"]))
        runtime.ornMeterNum = 155.25
        runtime.oneTempAngle = 360.0 / 101.0
        return cls(
            runtime,
            component_sha256={name: sha256_file(path) for name, path in paths.items()},
        )

    def infer(
        self,
        image_bgr: np.ndarray,
        config: FrozenLegacyTaskConfig,
        *,
        reading_backend: str,
    ) -> RuntimeSnapshot:
        if reading_backend != "geometry_fusion_weighted":
            raise ValueError(
                "Base-only runtime permits only geometry_fusion_weighted; "
                "Original Transformer is not loaded"
            )
        return super().infer(image_bgr, config, reading_backend=reading_backend)


def _adapter_source_sha256() -> str:
    return sha256_file(Path(__file__))


def _finite_optional(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    first, second = _finite_optional(value[0]), _finite_optional(value[1])
    return None if first is None or second is None else [first, second]


def _automatic_reference_attestation(
    snapshot: RuntimeSnapshot,
    *,
    input_image_sha256: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    artifacts = snapshot.artifacts
    branch = str(artifacts.get("branch") or "")
    start_point = _point(artifacts.get("center_start"))
    end_point = _point(artifacts.get("center_end"))
    start_angle = _finite_optional(artifacts.get("startAngle"))
    end_angle = _finite_optional(artifacts.get("endAngle"))
    range_angle = _finite_optional(artifacts.get("disAngle"))
    expected_bbox = [0.0, 0.0, float(width), float(height)]
    implicit_native = (
        branch == "implicit_native"
        and artifacts.get("reference_mode") == "implicit_native"
        and artifacts.get("reference_detector_invoked") is False
        and start_point is None
        and end_point is None
        and start_angle is None
        and end_angle is None
        and range_angle is None
    )
    automatic_explicit = (
        branch == "start_and_end"
        and artifacts.get("reference_mode") == "automatic_explicit"
        and artifacts.get("reference_detector_invoked") is True
        and start_point is not None
        and end_point is not None
        and start_angle is not None
        and end_angle is not None
        and range_angle is not None
        and 0.0 < range_angle < 360.0
        and artifacts.get("meter_bbox") == expected_bbox
        and artifacts.get("meter_detector_invoked") is False
    )
    valid = (
        (implicit_native or automatic_explicit)
        and artifacts.get("meter_bbox") == expected_bbox
        and artifacts.get("meter_detector_invoked") is False
    )
    reference_mode = "implicit_native" if implicit_native else "automatic_explicit"
    payload = {
        "protocol": AUTOMATIC_REFERENCE_PROTOCOL,
        "valid": bool(valid),
        "reference_mode": reference_mode,
        "input_image_sha256": input_image_sha256,
        "canonical_meter_roi": {
            "bbox_xyxy": expected_bbox,
            "meter_detector_invoked": False,
            "second_crop_applied": False,
            "correction_applied": False,
        },
        "reference_detection": {
            "source": (
                "native_transformer_101_class_progress"
                if implicit_native
                else "pointerDetect.center_find(classId=2,start;classId=1,end)"
            ),
            "scope": "unchanged canonical meter ROI",
            "reference_detector_invoked": bool(automatic_explicit),
            "branch": branch or None,
            "start_xy": start_point,
            "end_xy": end_point,
            "start_angle_degrees": start_angle,
            "end_angle_degrees": end_angle,
            "range_angle_degrees": range_angle,
        },
        "manual_reference_consumed": False,
        "default_angle_fallback_accepted": False,
    }
    payload["automatic_reference_sha256"] = canonical_sha256(payload)
    return payload


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    record["evidence_role"] = EVIDENCE_ROLE
    record["output_mode"] = OUTPUT_MODE
    record["eligible_for_primary_metrics"] = False
    record["output_schema_sha256"] = OUTPUT_SCHEMA_SHA256
    record["adapter_source_sha256"] = _adapter_source_sha256()
    record["record_sha256"] = canonical_sha256(record)
    return record


def _failure(
    *,
    method: str,
    stage: str,
    code: str,
    exception: Exception | None,
    image_hash: str | None,
    config_hash: str | None,
    components: Mapping[str, str],
    automatic_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _finalize(
        {
            "schema_version": 5,
            "protocol": PROTOCOL,
            "comparison_tier": "progress_component_diagnostic",
            "method": method,
            "status": "failed",
            "prediction_space": "normalized_progress_0_1",
            "prediction": None,
            "progress": None,
            "failure": {
                "code": str(code)[:96],
                "stage": str(stage)[:64],
                "exception_type": None if exception is None else type(exception).__name__,
            },
            "input_attestation": {
                "input_image_sha256": image_hash,
                "task_config_sha256": config_hash,
                "input_is_canonical_meter_roi": True,
                "meter_detector_invoked": False,
                "second_crop_applied": False,
                "ground_truth_consumed": False,
                "physical_scale_consumed": False,
                "manual_scalemark_consumed": False,
                "manual_reference_consumed": False,
                "shared_reference_packet_consumed": False,
            },
            "automatic_reference": automatic_reference,
            "component_sha256": dict(components),
            "telemetry": {},
        }
    )


class _CanonicalROIAdapter:
    METHOD = ""
    BACKEND = ""

    def __init__(
        self,
        runtime: FrozenCanonicalROILegacyRuntime,
        *,
        task_config: FrozenLegacyTaskConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        # Configuration is frozen once for the whole run.  It is deliberately
        # not a per-image ``predict`` input, preventing sample-specific tuning
        # after labels or expected readings are known.
        self.task_config = FrozenLegacyTaskConfig.from_value(task_config)
        self.task_config_sha256 = self.task_config.sha256
        project = Path(__file__).resolve().parents[1]
        self.pipeline_source_sha256 = sha256_file(
            project / "utils" / "angleDetect" / "zeroShotMeter.py"
        )

    @property
    def component_sha256(self) -> dict[str, str]:
        result = dict(self.runtime.component_sha256)
        result["production_pipeline_source"] = self.pipeline_source_sha256
        return result

    def _progress(self, selected: Mapping[str, Any]) -> float | None:
        value = _finite_optional(selected.get("progress_ratio"))
        if value is None:
            end_num = _finite_optional(selected.get("endNum_float"))
            value = None if end_num is None else end_num / 100.0
        return value

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> dict[str, Any]:
        image_hash: str | None = None
        config_hash: str | None = self.task_config_sha256
        try:
            if input_is_canonical_meter_roi is not True:
                raise ValueError("primary adapter requires input_is_canonical_meter_roi=True")
            image = _validate_image(canonical_meter_roi_bgr)
            image_hash = image_sha256(image)
            config = self.task_config
        except Exception as exc:
            return _failure(
                method=self.METHOD,
                stage="input_validation",
                code="invalid_canonical_roi_or_task_configuration",
                exception=exc,
                image_hash=image_hash,
                config_hash=config_hash,
                components=self.component_sha256,
            )
        try:
            snapshot = self.runtime.infer(image, config, reading_backend=self.BACKEND)
        except Exception as exc:
            return _failure(
                method=self.METHOD,
                stage="canonical_roi_pipeline",
                code="canonical_roi_pipeline_exception",
                exception=exc,
                image_hash=image_hash,
                config_hash=config_hash,
                components=self.component_sha256,
            )
        automatic_reference = _automatic_reference_attestation(
            snapshot,
            input_image_sha256=image_hash,
            width=image.shape[1],
            height=image.shape[0],
        )
        if not automatic_reference["valid"]:
            return _failure(
                method=self.METHOD,
                stage="automatic_reference_detection",
                code="complete_automatic_start_end_reference_unavailable",
                exception=None,
                image_hash=image_hash,
                config_hash=config_hash,
                components=self.component_sha256,
                automatic_reference=automatic_reference,
            )
        selected = snapshot.selected
        if not bool(selected.get("status")):
            return _failure(
                method=self.METHOD,
                stage="selected_backend",
                code=snapshot.error_code or "selected_backend_failed",
                exception=None,
                image_hash=image_hash,
                config_hash=config_hash,
                components=self.component_sha256,
                automatic_reference=automatic_reference,
            )
        prediction = _finite_optional(selected.get("resultNum"))
        if prediction is None:
            prediction = snapshot.returned_prediction
        progress = self._progress(selected)
        if prediction is None or progress is None or not 0.0 <= progress <= 1.0:
            return _failure(
                method=self.METHOD,
                stage="output_validation",
                code="non_finite_or_out_of_range_backend_output",
                exception=None,
                image_hash=image_hash,
                config_hash=config_hash,
                components=self.component_sha256,
                automatic_reference=automatic_reference,
            )
        return _finalize(
            {
                "schema_version": 5,
                "protocol": PROTOCOL,
                "comparison_tier": "progress_component_diagnostic",
                "method": self.METHOD,
                "status": "ok",
                "prediction_space": "normalized_progress_0_1",
                "prediction": float(prediction),
                "progress": float(progress),
                "failure": None,
                "input_attestation": {
                    "input_image_sha256": image_hash,
                    "task_config_sha256": config_hash,
                    "input_is_canonical_meter_roi": True,
                    "meter_bbox_xyxy": [0, 0, int(image.shape[1]), int(image.shape[0])],
                    "meter_detector_invoked": False,
                    "second_crop_applied": False,
                    "correction_applied": False,
                    "ground_truth_consumed": False,
                    "physical_scale_consumed": False,
                    "manual_scalemark_consumed": False,
                    "manual_reference_consumed": False,
                    "shared_reference_packet_consumed": False,
                    "reference_mode": automatic_reference["reference_mode"],
                },
                "automatic_reference": automatic_reference,
                "component_sha256": self.component_sha256,
                "telemetry": {
                    "selected_backend": str(selected.get("backend") or self.BACKEND),
                    "reference_mode": automatic_reference["reference_mode"],
                    "reference_detector_invoked": bool(
                        snapshot.artifacts.get("reference_detector_invoked")
                    ),
                    "mask_sha256": snapshot.artifacts.get("mask_sha256"),
                    "mask_foreground_ratio": _finite_optional(
                        snapshot.artifacts.get("mask_foreground_ratio")
                    ),
                    "mask_line_valid": bool(snapshot.artifacts.get("mask_line_valid")),
                    "selected_confidence": _finite_optional(selected.get("confidence")),
                    "source_delta_progress": _finite_optional(
                        selected.get("source_delta_progress")
                    ),
                },
            }
        )


class OriginalTransformerAdapter(_CanonicalROIAdapter):
    """Historical native-class diagnostic retained for compatibility only."""

    METHOD = "original_transformer"
    BACKEND = "transformer"

    @property
    def component_sha256(self) -> dict[str, str]:
        result = super().component_sha256
        # The shared runtime may also serve Base Mask-Geometry, but the native
        # Transformer path never invokes or consumes the reference detector.
        result.pop("reference_detector", None)
        return result

    def _progress(self, selected: Mapping[str, Any]) -> float | None:
        end_num = _finite_optional(selected.get("endNum_float"))
        return None if end_num is None else end_num / 100.0


class OriginalTransformerAutomaticReferenceAdapter(_CanonicalROIAdapter):
    """Original Transformer orientation converted by automatic dial references."""

    METHOD = "original_transformer_auto_reference"
    BACKEND = "transformer_automatic_reference"


class BaseMaskGeometryAdapter(_CanonicalROIAdapter):
    """Frozen Base Mask-Geometry on the unchanged canonical meter ROI."""

    METHOD = "base_mask_geometry"
    BACKEND = "geometry_fusion_weighted"

    @property
    def component_sha256(self) -> dict[str, str]:
        result = super().component_sha256
        result.pop("original_transformer", None)
        project = Path(__file__).resolve().parents[1]
        result["geometry_baseline_source"] = sha256_file(
            project / "utils" / "angleDetect" / "geometry_baseline.py"
        )
        return result


__all__ = [
    "AUTOMATIC_REFERENCE_PROTOCOL",
    "BaseMaskGeometryAdapter",
    "FrozenBaseMaskGeometryRuntime",
    "FrozenCanonicalROILegacyRuntime",
    "FrozenLegacyTaskConfig",
    "OriginalTransformerAdapter",
    "OriginalTransformerAutomaticReferenceAdapter",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_SHA256",
    "PROTOCOL",
    "TASK_CONFIG_SCHEMA",
    "TASK_CONFIG_SCHEMA_SHA256",
    "canonical_sha256",
    "image_sha256",
    "sha256_file",
]
