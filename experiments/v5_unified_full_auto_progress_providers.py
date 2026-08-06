"""Image-only progress providers for the unified full-auto reading adapter.

The existing PEPD and VDN adapters intentionally expose a batched component
API that receives a reference packet.  That is useful for controlled studies,
but it is too permissive for full automatic reading.  The wrappers here move
the frozen reference detector *inside* the provider, so their public
``predict`` methods accept only one canonical ROI.

Original Transformer remains native-progress and never loads/invokes a
reference detector.  V5-CRRM already owns its automatic reference provider;
its wrapper converts the internally predicted pivot/start/end points to the
same image-angle convention used by PEPD/VDN and by V5 training:

``angle = degrees(atan2(dx, -dy)) - 180 (mod 360)``.

No wrapper accepts a caller reference, numeric range, geometry packet, crop,
label, or target.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import torch

from experiments.v5_unified_crrm_adapter import (
    AutomaticReferenceResult,
    UnifiedCRRMAdapter,
    canonical_sha256,
)
from experiments.v5_unified_direction_adapters import (
    LabelFreeInput,
    PEPDLabelFreeAdapter,
    REFERENCE_MODE_AUTO,
    VDNOfficial200LabelFreeAdapter,
    _automatic_reference_packet,
)
from experiments.v5_unified_full_auto_adapter import _progress_image_sha256
from experiments.v5_unified_legacy_adapters import (
    FrozenCanonicalROILegacyRuntime,
    FrozenLegacyTaskConfig,
    OriginalTransformerAdapter,
)
from experiments.vdn_baseline import (
    image_angle_from_direction,
    sha256_file,
    verify_vdn_source,
)


PROTOCOL: Final[str] = "v5_full_auto_progress_providers_v1"
REFERENCE_PROTOCOL: Final[str] = "internal_automatic_reference_provider_v1"
V5_REFERENCE_PROTOCOL: Final[str] = "v5_xy_to_standard_angle_reference_v1"
ANGLE_CONVENTION: Final[str] = (
    "degrees(atan2(point_x-pivot_x, -(point_y-pivot_y)))-180 modulo 360; "
    "directed range=(end-start) modulo 360"
)
_POINT_SPACES: Final[frozenset[str]] = frozenset({"normalized", "pixel"})
_WRAPPER_SOURCE: Final[Path] = Path(__file__).resolve()
_DIRECTION_ADAPTER_SOURCE: Final[Path] = (
    _WRAPPER_SOURCE.parent / "v5_unified_direction_adapters.py"
)
_LEGACY_ADAPTER_SOURCE: Final[Path] = (
    _WRAPPER_SOURCE.parent / "v5_unified_legacy_adapters.py"
)
_CRRM_ADAPTER_SOURCE: Final[Path] = (
    _WRAPPER_SOURCE.parent / "v5_unified_crrm_adapter.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    x_value, y_value = _finite(value[0]), _finite(value[1])
    return None if x_value is None or y_value is None else (x_value, y_value)


def _normalized_point(
    value: Any,
    *,
    coordinate_space: str,
    image_shape: Sequence[int],
) -> tuple[float, float] | None:
    point = _point(value)
    if point is None:
        return None
    _require(coordinate_space in _POINT_SPACES, "unknown reference point space")
    height, width = int(image_shape[0]), int(image_shape[1])
    _require(height > 1 and width > 1, "invalid canonical ROI shape")
    if coordinate_space == "pixel":
        point = (point[0] / float(width), point[1] / float(height))
    if not (-0.25 <= point[0] <= 1.25 and -0.25 <= point[1] <= 1.25):
        return None
    return point


def standard_reference_packet_from_points(
    *,
    pivot_xy: Sequence[float],
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    image_shape: Sequence[int],
    coordinate_space: str,
    branch_prefix: str = "v5_scalemark",
) -> dict[str, Any]:
    """Convert automatic XY points using the frozen production convention.

    All points must share one declared coordinate space.  The helper never
    substitutes the image center when the predicted pivot is absent because
    V5 was trained against its PEPD pivot, not an oracle/nominal dial center.
    """

    pivot = _normalized_point(
        pivot_xy, coordinate_space=coordinate_space, image_shape=image_shape
    )
    start = _normalized_point(
        start_xy, coordinate_space=coordinate_space, image_shape=image_shape
    )
    end = _normalized_point(
        end_xy, coordinate_space=coordinate_space, image_shape=image_shape
    )
    if pivot is None or start is None or end is None:
        return _failed_reference("invalid_or_missing_v5_reference_points", branch_prefix)
    start_vector = (start[0] - pivot[0], start[1] - pivot[1])
    end_vector = (end[0] - pivot[0], end[1] - pivot[1])
    if math.hypot(*start_vector) <= 1e-8 or math.hypot(*end_vector) <= 1e-8:
        return _failed_reference("collapsed_v5_reference_radius", branch_prefix)
    try:
        start_angle = image_angle_from_direction(start_vector)
        end_angle = image_angle_from_direction(end_vector)
    except ValueError:
        return _failed_reference("invalid_v5_reference_vector", branch_prefix)
    range_angle = (end_angle - start_angle) % 360.0
    if not 0.0 < range_angle < 360.0:
        return _failed_reference("collapsed_v5_reference_arc", branch_prefix)
    return {
        "status": True,
        "start_angle": float(start_angle),
        "range_angle": float(range_angle),
        "reference_branch": f"{branch_prefix}:start_and_end",
        "failure_code": None,
    }


def _failed_reference(code: str, branch_prefix: str) -> dict[str, Any]:
    return {
        "status": False,
        "start_angle": None,
        "range_angle": None,
        "reference_branch": f"{branch_prefix}:failed",
        "failure_code": str(code)[:128],
    }


@runtime_checkable
class FullAutoProgressProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]: ...


class FrozenProductionAutomaticReference:
    """Frozen point detector hidden behind an image-only API."""

    def __init__(
        self,
        point_detector: Any,
        *,
        detector_sha256: str,
        detector_loader_source_sha256: str,
    ) -> None:
        if not callable(getattr(point_detector, "center_find", None)):
            raise TypeError("point detector must expose center_find")
        self.point_detector = point_detector
        self.detector_sha256 = _sha256(detector_sha256, "reference detector")
        self.detector_loader_source_sha256 = _sha256(
            detector_loader_source_sha256, "reference detector loader"
        )
        self._lock = threading.Lock()
        self._identity = {
            "protocol": REFERENCE_PROTOCOL,
            "provider": "frozen_production_start_end_detector",
            "reference_detector_sha256": self.detector_sha256,
            "detector_loader_source_sha256": self.detector_loader_source_sha256,
            "primary_success_branch": "start_and_end",
            "single_endpoint_or_default_fallback_accepted": False,
            "external_reference_input": False,
            "angle_convention": ANGLE_CONVENTION,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @classmethod
    def from_weights(
        cls,
        weights_path: Path,
        *,
        expected_sha256: str,
    ) -> "FrozenProductionAutomaticReference":
        weights = Path(weights_path).resolve(strict=True)
        expected = _sha256(expected_sha256, "reference detector")
        actual = sha256_file(weights)
        _require(actual == expected, "reference detector checkpoint hash mismatch")
        from experiments.evaluate_vdn_baseline import _load_target_detector_class

        detector_class = _load_target_detector_class()
        detector = detector_class(str(weights))
        return cls(
            detector,
            detector_sha256=actual,
            detector_loader_source_sha256=sha256_file(
                Path(__import__(
                    "experiments.evaluate_vdn_baseline",
                    fromlist=["__file__"],
                ).__file__).resolve()
            ),
        )

    def predict(self, canonical_meter_roi_bgr: np.ndarray) -> dict[str, Any]:
        image = _canonical_image(canonical_meter_roi_bgr)
        with self._lock:
            return _automatic_reference_packet(self.point_detector, image)


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _canonical_image(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("canonical meter ROI must be a NumPy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("canonical meter ROI must be uint8 [H,W,3]")
    if min(value.shape[:2]) < 32:
        raise ValueError("canonical meter ROI is too small")
    return np.ascontiguousarray(value)


def _consume_component_record_sha256(
    record: dict[str, Any], *, label: str
) -> str | None:
    declared = record.pop("record_sha256", None)
    if declared is None:
        return None
    digest = _sha256(declared, f"{label} record")
    if canonical_sha256(record) != digest:
        raise ValueError(f"{label} component record hash drift")
    return digest


class _DirectionFullAutoProgressProvider:
    PROVIDER_NAME = "direction"

    def __init__(
        self,
        direction_adapter: Any,
        *,
        automatic_reference_provider: FrozenProductionAutomaticReference,
        amp_enabled: bool,
    ) -> None:
        if not callable(getattr(direction_adapter, "predict", None)):
            raise TypeError("direction adapter must expose predict")
        self.direction_adapter = direction_adapter
        self.reference_provider = automatic_reference_provider
        self.amp_enabled = bool(amp_enabled)
        checkpoint = _sha256(
            getattr(direction_adapter, "checkpoint_sha256", None),
            f"{self.PROVIDER_NAME} checkpoint",
        )
        verification = _sha256(
            getattr(direction_adapter, "verification_sha256", None),
            f"{self.PROVIDER_NAME} verification",
        )
        self._identity = {
            "protocol": PROTOCOL,
            "provider": self.PROVIDER_NAME,
            "checkpoint_sha256": checkpoint,
            "verification_sha256": verification,
            "verification_protocol": str(
                getattr(direction_adapter, "verification_protocol", "")
            ),
            "automatic_reference": dict(automatic_reference_provider.identity),
            "reference_detector_sha256": automatic_reference_provider.detector_sha256,
            "reference_is_internal": True,
            "external_reference_input": False,
            "input": "one unchanged canonical meter ROI",
            "amp_enabled": self.amp_enabled,
            "native_input_size": int(getattr(direction_adapter, "image_size", 0)),
            "direction_adapter_source_sha256": sha256_file(
                _DIRECTION_ADAPTER_SOURCE
            ),
            "wrapper_source_sha256": sha256_file(_WRAPPER_SOURCE),
        }
        vdn_source = getattr(direction_adapter, "vdn_source", None)
        if vdn_source is not None:
            self._identity["vdn_source_commit"] = verify_vdn_source(
                Path(vdn_source)
            )

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        if input_is_canonical_meter_roi is not True:
            raise ValueError("direction full-auto provider requires canonical ROI")
        image = _canonical_image(canonical_meter_roi_bgr)
        image_digest = _progress_image_sha256(image)
        reference = self.reference_provider.predict(image)
        item = LabelFreeInput(
            sample_id=f"runtime:{image_digest[:16]}",
            group_id=f"runtime:{image_digest[:16]}",
            image_path=Path("canonical-roi-in-memory"),
            image_sha256=image_digest,
            frame_sha256=image_digest,
            reference_input=None,
            reference_input_sha256=None,
        )
        records = self.direction_adapter.predict(
            [item],
            [image],
            [reference],
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=self.reference_provider.detector_sha256,
            amp_enabled=self.amp_enabled,
        )
        _require(
            isinstance(records, list) and len(records) == 1,
            "direction adapter returned wrong batch size",
        )
        record = dict(records[0])
        record["input_attestation"] = {
            "input_image_sha256": image_digest,
            "input_is_canonical_meter_roi": True,
            "meter_detector_invoked": False,
            "second_crop_applied": False,
            "ground_truth_consumed": False,
            "physical_scale_consumed": False,
            "manual_reference_consumed": False,
            "caller_reference_packet_consumed": False,
        }
        return record


class PEPDFullAutoProgressProvider(_DirectionFullAutoProgressProvider):
    PROVIDER_NAME = "pepd"

    @classmethod
    def from_frozen_files(
        cls,
        *,
        checkpoint_path: Path,
        expected_checkpoint_sha256: str,
        verification_path: Path,
        reference_detector_path: Path,
        expected_reference_detector_sha256: str,
        device: str | torch.device,
        amp_enabled: bool | None = None,
    ) -> "PEPDFullAutoProgressProvider":
        torch_device = torch.device(device)
        adapter = PEPDLabelFreeAdapter(
            checkpoint_path=Path(checkpoint_path),
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            verification_path=Path(verification_path),
            device=torch_device,
        )
        reference = FrozenProductionAutomaticReference.from_weights(
            reference_detector_path,
            expected_sha256=expected_reference_detector_sha256,
        )
        return cls(
            adapter,
            automatic_reference_provider=reference,
            amp_enabled=(torch_device.type == "cuda" if amp_enabled is None else amp_enabled),
        )


class VDNFullAutoProgressProvider(_DirectionFullAutoProgressProvider):
    PROVIDER_NAME = "vdn_official200"

    @classmethod
    def from_frozen_files(
        cls,
        *,
        checkpoint_path: Path,
        expected_checkpoint_sha256: str,
        verification_path: Path,
        vdn_source: Path,
        reference_detector_path: Path,
        expected_reference_detector_sha256: str,
        device: str | torch.device,
        amp_enabled: bool | None = None,
    ) -> "VDNFullAutoProgressProvider":
        torch_device = torch.device(device)
        adapter = VDNOfficial200LabelFreeAdapter(
            checkpoint_path=Path(checkpoint_path),
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            verification_path=Path(verification_path),
            vdn_source=Path(vdn_source),
            device=torch_device,
        )
        reference = FrozenProductionAutomaticReference.from_weights(
            reference_detector_path,
            expected_sha256=expected_reference_detector_sha256,
        )
        return cls(
            adapter,
            automatic_reference_provider=reference,
            amp_enabled=(torch_device.type == "cuda" if amp_enabled is None else amp_enabled),
        )


class TransformerFullAutoProgressProvider:
    """Original Transformer native progress; no automatic reference detector."""

    def __init__(self, adapter: OriginalTransformerAdapter) -> None:
        if not callable(getattr(adapter, "predict", None)):
            raise TypeError("Original Transformer adapter must expose predict")
        self.adapter = adapter
        components = {
            str(key): _sha256(value, f"transformer component {key}")
            for key, value in adapter.component_sha256.items()
        }
        _require(
            "reference_detector" not in components,
            "native Transformer identity unexpectedly binds a reference detector",
        )
        self._identity = {
            "protocol": PROTOCOL,
            "provider": "original_transformer_native_progress",
            "component_sha256": components,
            "task_config_sha256": _sha256(
                adapter.task_config_sha256, "Transformer task config"
            ),
            "reference_mode": "native_implicit_progress",
            "reference_detector_loaded": False,
            "reference_detector_invoked": False,
            "external_reference_input": False,
            "input": "one unchanged canonical meter ROI",
            "legacy_adapter_source_sha256": sha256_file(_LEGACY_ADAPTER_SOURCE),
            "wrapper_source_sha256": sha256_file(_WRAPPER_SOURCE),
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @classmethod
    def from_frozen_files(
        cls,
        *,
        pointer_segmentation_path: Path,
        original_transformer_path: Path,
        device: str | torch.device,
        task_config: FrozenLegacyTaskConfig | Mapping[str, Any] | None = None,
    ) -> "TransformerFullAutoProgressProvider":
        # Construct the same frozen Transformer/segmenter runtime without even
        # deserializing a reference detector.  The legacy runtime roster has a
        # reference slot because it also serves Base Mask-Geometry; a guarded
        # in-memory sentinel fills that unreachable slot for native progress.
        from utils.angleDetect.detect import meterFormer
        from utils.angleDetect.pointerSeg.detectSeg import u2netpSeg
        from utils.angleDetect.zeroShotMeter import meterZeroShot

        pointer_path = Path(pointer_segmentation_path).resolve(strict=True)
        transformer_path = Path(original_transformer_path).resolve(strict=True)
        torch_device = torch.device(device)
        native_runtime = meterZeroShot.__new__(meterZeroShot)
        native_runtime.device = torch_device
        native_runtime.pointerSeg = u2netpSeg(str(pointer_path), torch_device)
        native_runtime.vlmMeter = meterFormer(str(transformer_path), torch_device)
        native_runtime.pointerDetect = _ForbiddenReferenceDetector()
        native_runtime.label_text_list = list(map(str, range(101)))
        native_runtime.label_text = native_runtime.vlmMeter.model.texteconder.tokenize(
            native_runtime.label_text_list
        ).to(torch_device)
        native_runtime.ornMeterNum = 155.25
        native_runtime.oneTempAngle = 360.0 / 101.0
        runtime = FrozenCanonicalROILegacyRuntime(
            native_runtime,
            component_sha256={
                "pointer_segmentation": sha256_file(pointer_path),
                "original_transformer": sha256_file(transformer_path),
                "reference_detector": sha256_file(Path(__file__)),
            },
        )
        return cls(OriginalTransformerAdapter(runtime, task_config=task_config))

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        image = _canonical_image(canonical_meter_roi_bgr)
        record = dict(
            self.adapter.predict(
                image,
                input_is_canonical_meter_roi=input_is_canonical_meter_roi,
            )
        )
        component_record_sha256 = _consume_component_record_sha256(
            record, label="Original Transformer"
        )
        record["reference_detector_invoked"] = False
        record["component_record_sha256"] = component_record_sha256
        # Do not add auto_reference: the full-auto compositor treats this as
        # REFERENCE_MODE_NATIVE and rejects any explicit reference packet.
        record["wrapper_record_sha256"] = canonical_sha256(record)
        return record


class _ForbiddenReferenceDetector:
    def center_find(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("native Transformer must never invoke reference detection")


class _RecordingReferenceProvider:
    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.identity = provider.identity
        self.last_result: AutomaticReferenceResult | None = None

    def predict(self, canonical_meter_roi_bgr: np.ndarray) -> AutomaticReferenceResult:
        result = self.provider.predict(canonical_meter_roi_bgr)
        if not isinstance(result, AutomaticReferenceResult):
            raise TypeError("V5 automatic reference provider returned wrong type")
        self.last_result = result
        return result


def _mapping_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def _v5_pivot(
    reference: AutomaticReferenceResult,
    record: Mapping[str, Any],
    *,
    endpoint_space: str,
    image_shape: Sequence[int],
) -> tuple[tuple[float, float] | None, str]:
    """Find an internally predicted pivot without center substitution."""

    sources = (
        (reference.telemetry, ("pivot_xy_normalized",), "normalized"),
        (reference.context, ("pivot_xy_normalized",), "normalized"),
        (reference.telemetry, ("pivot_input_xy",), "pixel"),
        (reference.context, ("pivot_input_xy",), "pixel"),
        (reference.telemetry, ("pivot_xy",), endpoint_space),
        (reference.context, ("pivot_xy",), endpoint_space),
        (
            record,
            ("expert_results", "pepd", "telemetry", "pivot_xy_normalized"),
            "normalized",
        ),
        (
            record,
            ("expert_results", "pepd", "telemetry", "pivot_input_xy"),
            "pixel",
        ),
    )
    for container, path, space in sources:
        candidate = _mapping_path(container, path)
        normalized = _normalized_point(
            candidate, coordinate_space=space, image_shape=image_shape
        )
        if normalized is not None:
            return normalized, f"{'.'.join(path)}:{space}"
    return None, "unavailable"


class V5CRRMFullAutoProgressProvider:
    """V5 CRRM progress plus a standardized internal reference packet."""

    def __init__(
        self,
        adapter: UnifiedCRRMAdapter,
        *,
        automatic_reference_artifact_sha256: str,
        endpoint_coordinate_space: str = "normalized",
    ) -> None:
        if not isinstance(adapter, UnifiedCRRMAdapter):
            raise TypeError("adapter must be UnifiedCRRMAdapter")
        _require(
            endpoint_coordinate_space in _POINT_SPACES,
            "V5 endpoint coordinate space must be frozen",
        )
        original_reference = adapter.reference_provider
        self._recording_reference = _RecordingReferenceProvider(original_reference)
        # Rebuild rather than mutating the supplied adapter.  Provider and gate
        # identities are revalidated by UnifiedCRRMAdapter itself.
        self.adapter = UnifiedCRRMAdapter(
            automatic_reference_provider=self._recording_reference,
            pepd_provider=adapter.providers["pepd"],
            scalemark_provider=adapter.providers["scalemark"],
            mask_geometry_provider=adapter.providers["mask_geometry"],
            gate=adapter.gate,
            execution_mode=adapter.execution_mode,
        )
        self.reference_artifact_sha256 = _sha256(
            automatic_reference_artifact_sha256,
            "V5 automatic reference artifact",
        )
        self.endpoint_coordinate_space = endpoint_coordinate_space
        self._lock = threading.Lock()
        self._identity = {
            "protocol": PROTOCOL,
            "provider": "v5_crrm_progress",
            "component_bindings": self.adapter.component_bindings,
            "automatic_reference_artifact_sha256": self.reference_artifact_sha256,
            "reference_detector_sha256": self.reference_artifact_sha256,
            "endpoint_coordinate_space": endpoint_coordinate_space,
            "xy_to_angle_protocol": V5_REFERENCE_PROTOCOL,
            "angle_convention": ANGLE_CONVENTION,
            "missing_pivot_policy": "fail; image-center substitution forbidden",
            "reference_is_internal": True,
            "external_reference_input": False,
            "input": "one unchanged canonical meter ROI",
            "crrm_adapter_source_sha256": sha256_file(_CRRM_ADAPTER_SOURCE),
            "wrapper_source_sha256": sha256_file(_WRAPPER_SOURCE),
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        if input_is_canonical_meter_roi is not True:
            raise ValueError("V5 full-auto provider requires canonical ROI")
        image = _canonical_image(canonical_meter_roi_bgr)
        input_digest = _progress_image_sha256(image)
        with self._lock:
            self._recording_reference.last_result = None
            record = dict(
                self.adapter.predict(image, input_is_canonical_meter_roi=True)
            )
            reference = self._recording_reference.last_result
        component_record_sha256 = _consume_component_record_sha256(
            record, label="V5 CRRM"
        )
        if reference is None:
            packet = _failed_reference(
                "v5_automatic_reference_not_invoked", "v5_scalemark"
            )
            pivot_source = "unavailable"
        elif not reference.available:
            packet = _failed_reference(
                reference.failure_code or "v5_automatic_reference_unavailable",
                "v5_scalemark",
            )
            pivot_source = "unavailable"
        else:
            pivot, pivot_source = _v5_pivot(
                reference,
                record,
                endpoint_space=self.endpoint_coordinate_space,
                image_shape=image.shape,
            )
            if pivot is None:
                packet = _failed_reference(
                    "v5_predicted_pivot_unavailable", "v5_scalemark"
                )
            else:
                start = _normalized_point(
                    reference.start_xy,
                    coordinate_space=self.endpoint_coordinate_space,
                    image_shape=image.shape,
                )
                end = _normalized_point(
                    reference.end_xy,
                    coordinate_space=self.endpoint_coordinate_space,
                    image_shape=image.shape,
                )
                if start is None or end is None:
                    packet = _failed_reference(
                        "v5_endpoint_coordinate_invalid", "v5_scalemark"
                    )
                else:
                    packet = standard_reference_packet_from_points(
                        pivot_xy=pivot,
                        start_xy=start,
                        end_xy=end,
                        image_shape=image.shape,
                        coordinate_space="normalized",
                    )
        if packet["status"] is not True:
            record["status"] = "failed"
            record["progress"] = None
            record["failure"] = {
                "code": packet["failure_code"],
                "stage": "v5_standard_reference_attestation",
            }
        record["auto_reference"] = packet
        record["reference_detector_sha256"] = self.reference_artifact_sha256
        record["input_attestation"] = {
            **dict(record.get("input_attestation") or {}),
            "input_image_sha256": input_digest,
            "input_is_canonical_meter_roi": True,
            "meter_detector_invoked": False,
            "second_crop_applied": False,
            "ground_truth_consumed": False,
            "physical_scale_consumed": False,
            "manual_reference_consumed": False,
            "caller_reference_packet_consumed": False,
        }
        record["standard_reference_attestation"] = {
            "protocol": V5_REFERENCE_PROTOCOL,
            "status": packet["status"],
            "pivot_source": pivot_source,
            "endpoint_coordinate_space": self.endpoint_coordinate_space,
            "angle_convention": ANGLE_CONVENTION,
            "image_center_substitution_used": False,
        }
        record["component_record_sha256"] = component_record_sha256
        record["wrapper_record_sha256"] = canonical_sha256(record)
        return record


__all__ = [
    "ANGLE_CONVENTION",
    "FrozenProductionAutomaticReference",
    "FullAutoProgressProvider",
    "PEPDFullAutoProgressProvider",
    "PROTOCOL",
    "REFERENCE_PROTOCOL",
    "TransformerFullAutoProgressProvider",
    "V5CRRMFullAutoProgressProvider",
    "V5_REFERENCE_PROTOCOL",
    "VDNFullAutoProgressProvider",
    "standard_reference_packet_from_points",
]
