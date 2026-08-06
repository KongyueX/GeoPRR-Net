"""Image-only GARC geometry adapter for the existing Base Mask--Geometry.

The wrapped implementation is the already-audited production/research path in
``v5_unified_legacy_adapters``:

* U2Net pointer segmentation and largest-component filtering;
* the two line/tip estimators from ``geometry_baseline.py``;
* the frozen Base pivot policy (centre of the supplied canonical meter ROI);
* automatic YOLO ScaleMark start/end detection; and
* strict rejection unless *both* endpoints and the pointer-mask line are valid.

The ROI centre is an image-derived deterministic policy inherited from Base,
not a caller-supplied or annotated pivot.  This adapter never fabricates an
endpoint and never enables the legacy single-endpoint/default-angle fallbacks.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from experiments.automatic_numeric_range import (
    GeometryHint,
    GeometryProviderResult,
    image_sha256,
    reject_supervised_fields,
    validate_canonical_roi,
)
from experiments.v5_unified_legacy_adapters import (
    BaseMaskGeometryAdapter,
    FrozenBaseMaskGeometryRuntime,
    PROTOCOL as BASE_ADAPTER_PROTOCOL,
    canonical_sha256,
    image_sha256 as legacy_image_sha256,
    sha256_file,
)


PROTOCOL: Final[str] = "garc_existing_base_mask_geometry_provider_v1"
PIVOT_POLICY: Final[str] = (
    "frozen canonical-ROI centre used by existing Base Mask-Geometry"
)
REFERENCE_POLICY: Final[str] = (
    "automatic class-2 start and class-1 end detections; both required; "
    "single-endpoint and default-angle fallbacks forbidden"
)
_SOURCE = Path(__file__).resolve()
_BASE_SOURCE = _SOURCE.with_name("v5_unified_legacy_adapters.py")
_GEOMETRY_SOURCE = _SOURCE.parents[1] / "utils" / "angleDetect" / "geometry_baseline.py"


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    first, second = _finite(value[0]), _finite(value[1])
    if first is None or second is None:
        return None
    return float(first), float(second)


def _normalized_pixel_point(
    value: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    point = _point(value)
    if point is None or width <= 1 or height <= 1:
        return None
    normalized = (point[0] / float(width), point[1] / float(height))
    if not all(-0.05 <= coordinate <= 1.05 for coordinate in normalized):
        return None
    return normalized


def _foreground_quality(value: float | None) -> float:
    if value is None or not 0.0 < value < 0.35:
        return 0.0
    lower = min(1.0, value / 0.003)
    upper = min(1.0, (0.35 - value) / 0.10)
    return float(np.clip(min(lower, upper), 0.0, 1.0))


def _reference_geometry_quality(
    pivot: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> tuple[float, dict[str, float]]:
    pivot_array = np.asarray(pivot, dtype=np.float64)
    start_radius = float(np.linalg.norm(np.asarray(start, dtype=np.float64) - pivot_array))
    end_radius = float(np.linalg.norm(np.asarray(end, dtype=np.float64) - pivot_array))
    radius_balance = min(start_radius, end_radius) / max(start_radius, end_radius, 1e-8)
    mean_radius = 0.5 * (start_radius + end_radius)
    radius_extent = float(np.clip((mean_radius - 0.08) / 0.30, 0.0, 1.0))
    quality = float(np.clip(0.65 * radius_balance + 0.35 * radius_extent, 0.0, 1.0))
    return quality, {
        "start_radius_fraction": start_radius,
        "end_radius_fraction": end_radius,
        "endpoint_radius_balance": radius_balance,
        "endpoint_radius_extent": radius_extent,
    }


class BaseMaskGeometryUnavailable(RuntimeError):
    """Controlled failure of the existing fully automatic Base path."""

    def __init__(
        self,
        code: str,
        *,
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)[:128]
        self.telemetry = dict(telemetry or {})
        super().__init__(self.code)


class ExistingBaseMaskGeometryProvider:
    """Convert the real Base progress adapter into GARC geometry evidence."""

    def __init__(self, adapter: BaseMaskGeometryAdapter) -> None:
        if not isinstance(adapter, BaseMaskGeometryAdapter):
            raise TypeError("adapter must be BaseMaskGeometryAdapter")
        if getattr(adapter, "METHOD", None) != "base_mask_geometry":
            raise ValueError("adapter is not the frozen Base Mask-Geometry method")
        if getattr(adapter, "BACKEND", None) != "geometry_fusion_weighted":
            raise ValueError("Base Mask-Geometry backend binding drift")
        self.adapter = adapter
        self._lock = threading.Lock()
        components = {
            str(name): str(digest)
            for name, digest in adapter.component_sha256.items()
        }
        runtime_components = {
            str(name): str(digest)
            for name, digest in adapter.runtime.component_sha256.items()
        }
        self._identity = {
            "protocol": PROTOCOL,
            "provider": "existing_base_mask_geometry_to_garc",
            "base_adapter_protocol": BASE_ADAPTER_PROTOCOL,
            "base_method": adapter.METHOD,
            "base_backend": adapter.BACKEND,
            "base_task_config_sha256": adapter.task_config_sha256,
            "base_runtime_mode": getattr(
                adapter.runtime, "RUNTIME_MODE", "shared_legacy_runtime"
            ),
            "base_runtime_loaded_components": sorted(runtime_components),
            "original_transformer_present_in_runtime": (
                "original_transformer" in runtime_components
            ),
            "component_sha256": components,
            "source_sha256": {
                "provider": sha256_file(_SOURCE),
                "base_adapter": sha256_file(_BASE_SOURCE),
                "mask_line_geometry": sha256_file(_GEOMETRY_SOURCE),
            },
            "primary_input": "one unchanged canonical meter ROI BGR uint8",
            "pivot_policy": PIVOT_POLICY,
            "pivot_is_learned": False,
            "pivot_is_caller_supplied": False,
            "reference_policy": REFERENCE_POLICY,
            "requires_pointer_mask_line": True,
            "caller_supplied_points_allowed": False,
            "caller_supplied_reference_allowed": False,
            "caller_supplied_numeric_range_allowed": False,
            "default_endpoint_fallback_allowed": False,
        }

    @classmethod
    def from_base_only_checkpoints(
        cls,
        *,
        pointer_segmentation: Path,
        reference_detector: Path,
        device: Any = "cpu",
        task_config: Mapping[str, Any] | None = None,
    ) -> "ExistingBaseMaskGeometryProvider":
        """Build the formally bound Base provider without a Transformer.

        Only the U2Net pointer-segmentation and automatic ScaleMark reference
        detector checkpoints are accepted.  There is deliberately no
        Transformer checkpoint argument at this boundary.
        """

        runtime = FrozenBaseMaskGeometryRuntime.from_checkpoints(
            pointer_segmentation=Path(pointer_segmentation),
            reference_detector=Path(reference_detector),
            device=device,
        )
        return cls(BaseMaskGeometryAdapter(runtime, task_config=task_config))

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @staticmethod
    def _failure_code(record: Mapping[str, Any]) -> str:
        failure = record.get("failure")
        if isinstance(failure, Mapping):
            code = str(failure.get("code") or "")
            stage = str(failure.get("stage") or "")
            if stage == "automatic_reference_detection":
                return "automatic_start_end_reference_incomplete"
            if code:
                return f"base_adapter_{code}"[:128]
        return "base_mask_geometry_unavailable"

    def predict(self, image_bgr: np.ndarray) -> GeometryProviderResult:
        """Run Base from the same ROI; no geometry/reference kwargs exist."""

        image = validate_canonical_roi(image_bgr)
        automatic_digest = image_sha256(image)
        legacy_digest = legacy_image_sha256(image)
        adapter_image = image.copy()
        with self._lock:
            record = self.adapter.predict(
                adapter_image,
                input_is_canonical_meter_roi=True,
            )
        if legacy_image_sha256(adapter_image) != legacy_digest:
            raise BaseMaskGeometryUnavailable("base_adapter_mutated_input")
        if not isinstance(record, Mapping):
            raise BaseMaskGeometryUnavailable("base_adapter_output_not_mapping")
        record = dict(record)
        reject_supervised_fields(record, path="base_mask_geometry_output")
        declared_record_hash = str(record.get("record_sha256") or "")
        unhashed_record = dict(record)
        unhashed_record.pop("record_sha256", None)
        if canonical_sha256(unhashed_record) != declared_record_hash:
            raise BaseMaskGeometryUnavailable("base_adapter_record_hash_mismatch")

        input_attestation = record.get("input_attestation")
        if not isinstance(input_attestation, Mapping):
            raise BaseMaskGeometryUnavailable("base_input_attestation_missing")
        if input_attestation.get("input_image_sha256") != legacy_digest:
            raise BaseMaskGeometryUnavailable("base_input_image_binding_mismatch")
        if input_attestation.get("task_config_sha256") != self.adapter.task_config_sha256:
            raise BaseMaskGeometryUnavailable("base_task_config_binding_mismatch")
        negative_attestations = (
            "ground_truth_consumed",
            "physical_scale_consumed",
            "manual_scalemark_consumed",
            "manual_reference_consumed",
            "shared_reference_packet_consumed",
        )
        if any(input_attestation.get(name) is not False for name in negative_attestations):
            raise BaseMaskGeometryUnavailable("base_input_boundary_not_label_free")
        if (
            input_attestation.get("meter_detector_invoked") is not False
            or input_attestation.get("second_crop_applied") is not False
        ):
            raise BaseMaskGeometryUnavailable("base_adapter_changed_canonical_roi")

        telemetry = record.get("telemetry")
        telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
        reference = record.get("automatic_reference")
        reference = dict(reference) if isinstance(reference, Mapping) else {}
        declared_reference_hash = str(reference.get("automatic_reference_sha256") or "")
        unhashed_reference = dict(reference)
        unhashed_reference.pop("automatic_reference_sha256", None)
        if reference and canonical_sha256(unhashed_reference) != declared_reference_hash:
            raise BaseMaskGeometryUnavailable("base_reference_record_hash_mismatch")
        detection = reference.get("reference_detection")
        detection = dict(detection) if isinstance(detection, Mapping) else {}
        base_trace = {
            "protocol": PROTOCOL,
            "input_attestation": {
                "input_image_sha256": automatic_digest,
                "base_adapter_image_sha256": legacy_digest,
                "same_decoded_array": True,
                "one_whole_canonical_roi": True,
                "caller_points_consumed": False,
                "caller_reference_consumed": False,
                "physical_numeric_range_consumed": False,
            },
            "base_record_sha256": record.get("record_sha256"),
            "base_component_sha256": dict(record.get("component_sha256") or {}),
            "pivot_policy": PIVOT_POLICY,
            "reference_policy": REFERENCE_POLICY,
            "mask_line_valid": telemetry.get("mask_line_valid") is True,
            "reference_branch": detection.get("branch"),
            "reference_detector_invoked": detection.get("reference_detector_invoked"),
            "default_angle_fallback_accepted": reference.get(
                "default_angle_fallback_accepted"
            ),
        }
        if str(record.get("status") or "").casefold() != "ok":
            raise BaseMaskGeometryUnavailable(
                self._failure_code(record), telemetry=base_trace
            )
        if telemetry.get("mask_line_valid") is not True:
            raise BaseMaskGeometryUnavailable(
                "pointer_mask_line_invalid", telemetry=base_trace
            )
        if (
            reference.get("valid") is not True
            or detection.get("reference_detector_invoked") is not True
            or detection.get("branch") != "start_and_end"
            or reference.get("default_angle_fallback_accepted") is not False
        ):
            raise BaseMaskGeometryUnavailable(
                "automatic_start_end_reference_incomplete", telemetry=base_trace
            )

        height, width = image.shape[:2]
        start = _normalized_pixel_point(
            detection.get("start_xy"), width=width, height=height
        )
        end = _normalized_pixel_point(
            detection.get("end_xy"), width=width, height=height
        )
        if start is None or end is None:
            raise BaseMaskGeometryUnavailable(
                "automatic_endpoint_coordinates_invalid", telemetry=base_trace
            )
        # This is exactly the Base runtime's imgCenter=(width//2,height//2).
        pivot = (
            float(width // 2) / float(width),
            float(height // 2) / float(height),
        )
        geometry_quality, radius_trace = _reference_geometry_quality(
            pivot, start, end
        )
        foreground = _finite(telemetry.get("mask_foreground_ratio"))
        foreground_quality = _foreground_quality(foreground)
        selected_confidence = _finite(telemetry.get("selected_confidence"))
        mask_quality = (
            float(np.clip(selected_confidence, 0.0, 1.0))
            if selected_confidence is not None
            else 0.55
        )
        disagreement = _finite(telemetry.get("source_delta_progress"))
        agreement_quality = (
            0.65
            if disagreement is None
            else math.exp(-max(0.0, disagreement) / 0.08)
        )
        confidence = float(
            np.clip(
                0.42 * mask_quality
                + 0.18 * foreground_quality
                + 0.20 * agreement_quality
                + 0.20 * geometry_quality,
                0.0,
                0.82,  # endpoint detector exposes no class-specific score here
            )
        )
        pivot_sigma = 0.018 + 0.050 * (1.0 - confidence)
        endpoint_std = 5.0 + 18.0 * (1.0 - confidence)
        hint = GeometryHint(
            pivot_xy=pivot,
            start_xy=start,
            end_xy=end,
            confidence=confidence,
            source="existing_base_mask_geometry_strict_auto_reference",
        ).validate()
        output_trace = {
            **base_trace,
            "status": "accepted",
            "base_progress_used_as_geometry_input": False,
            "base_progress_available_for_consistency_only": record.get("progress"),
            "pivot_source": "canonical_roi_center_frozen_base_policy",
            "pivot_covariance_normalized": [
                [pivot_sigma**2, 0.0],
                [0.0, pivot_sigma**2],
            ],
            "endpoint_angle_std_degrees": [endpoint_std, endpoint_std],
            "confidence_kind": "uncalibrated_inference_reliability",
            "confidence_inputs": {
                "mask_quality": mask_quality,
                "mask_quality_native_available": selected_confidence is not None,
                "mask_foreground_fraction": foreground,
                "mask_foreground_quality": foreground_quality,
                "dual_estimator_agreement": agreement_quality,
                "reference_geometry_quality": geometry_quality,
                **radius_trace,
            },
            "automatic_endpoint_xy_normalized": {
                "start": list(start),
                "end": list(end),
            },
        }
        reject_supervised_fields(output_trace, path="base_mask_geometry_provider")
        return GeometryProviderResult(hint=hint, telemetry=output_trace)


__all__ = [
    "BaseMaskGeometryUnavailable",
    "ExistingBaseMaskGeometryProvider",
    "PIVOT_POLICY",
    "PROTOCOL",
    "REFERENCE_POLICY",
]
