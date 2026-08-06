"""Frozen label-free composition of progress and automatic numeric range.

This module is the narrow bridge between the progress-only adapters and the
full-reading contract in :mod:`experiments.v5_unified_two_stage_retest`.
Its public inference boundary accepts one unchanged canonical meter ROI only.
It deliberately has no argument for a numeric range, geometry, reference
packet, crop, label, or target.

The two branches are independent and fail closed:

* a frozen progress provider predicts normalized pointer progress;
* :class:`AutomaticNumericRangePipeline` predicts numeric start/end values;
* the full record succeeds only when both branches succeed on byte-identical
  decoded pixels.

The CLI has three explicit phases.  ``freeze-bundle`` authenticates provider
identities and executable factory source before inference.  ``run`` validates
the complete label-free manifest before loading the factory or opening images.
``emit-evaluator-bundle`` creates the method bundle consumed by the V5
two-stage freezer.  No command accepts a label path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import cv2
import numpy as np

_PROJECT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_PROJECT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_PROJECT_BOOTSTRAP))

from experiments.automatic_numeric_range import (
    AutomaticNumericRangePipeline,
    NumericRangePrediction,
    PROTOCOL as AUTOMATIC_RANGE_PROTOCOL,
    image_sha256 as range_image_sha256,
    validate_canonical_roi,
)
from experiments.v5_unified_two_stage_retest import (
    AUTO_REFERENCE_INTERFACE,
    CANONICAL_TIGHT_ROI_CONTRACT,
    EVIDENCE_ROLE_PRIMARY,
    INFERENCE_PROTOCOL,
    METHOD_PREDICTION_KEYS,
    NATIVE_PROGRESS_INTERFACE,
    OUTPUT_MODE_FULL_READING,
    REFERENCE_MODE_AUTO,
    REFERENCE_MODE_NATIVE,
    assert_label_free,
    canonical_json_sha256,
    sha256_file,
)


PROTOCOL: Final[str] = "v5_unified_full_auto_adapter_v1"
BUNDLE_PROTOCOL: Final[str] = "v5_unified_full_auto_bundle_v1"
COMPONENT_PROTOCOL: Final[str] = "v5_frozen_provider_binding_v1"
RUN_PROTOCOL: Final[str] = "v5_unified_full_auto_label_free_run_v1"
FACTORY_FUNCTION_DEFAULT: Final[str] = "build_full_auto_providers"
EXECUTION_FORMAL: Final[str] = "formal_frozen"
EXECUTION_SYNTHETIC: Final[str] = "synthetic_smoke"
EXECUTION_MODES: Final[frozenset[str]] = frozenset(
    {EXECUTION_FORMAL, EXECUTION_SYNTHETIC}
)
ROI_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    CANONICAL_TIGHT_ROI_CONTRACT
)
AUTO_REFERENCE_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    AUTO_REFERENCE_INTERFACE
)
NATIVE_REFERENCE_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    NATIVE_PROGRESS_INTERFACE
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "name",
        "provider_protocol",
        "provider_identity",
        "provider_identity_sha256",
        "artifact_sha256",
        "source_sha256",
        "frozen",
        "verified_complete",
        "synthetic",
    }
)
_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "method_name",
        "execution_mode",
        "reference_mode",
        "reference_contract_sha256",
        "reference_detector_sha256",
        "roi_contract_sha256",
        "output_mode",
        "evidence_role",
        "progress_component",
        "automatic_numeric_range_component",
        "factory_source_sha256",
        "adapter_source_sha256",
        "inference_boundary",
    }
)
_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "group_id",
        "image_path",
        "image_sha256",
        "canonical_roi_sha256",
        "frame_sha256",
        "roi_contract_sha256",
    }
)
_NEGATIVE_ATTESTATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "groundtruthconsumed",
        "physicalscaleconsumed",
        "manualscalemarkconsumed",
        "manualreferenceconsumed",
        "sharedreferencepacketconsumed",
        "callerreferencepacketconsumed",
        "externalreferencepacketconsumed",
    }
)
_NO_SECOND_ROI_KEYS: Final[frozenset[str]] = frozenset(
    {
        "meterdetectorinvoked",
        "secondcropapplied",
        "secondcropinvoked",
        "correctionapplied",
    }
)
_TRANSIENT_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "createdutc",
        "elapsedseconds",
        "initializationseconds",
        "path",
        "runtimeroot",
    }
)


class ProtocolViolation(RuntimeError):
    """A hash, schema, or no-label invariant was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_key(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite provider identity/output is forbidden")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    raise TypeError(f"value is not JSON-safe: {type(value).__name__}")


def _stable_identity(value: Any) -> Any:
    """Remove paths/timing while retaining model hashes and parameters.

    RapidOCR exposes absolute runtime paths and initialization duration in its
    runtime identity.  Neither changes model semantics and neither is stable
    across machines.  Artifact hashes nested beside those paths are retained.
    """

    safe = _json_safe(value)
    assert_label_free(safe, location="provider_identity")

    def visit(nested: Any) -> Any:
        if isinstance(nested, Mapping):
            result: dict[str, Any] = {}
            for key, child in nested.items():
                normalized = _normalized_key(key)
                if (
                    normalized in _TRANSIENT_IDENTITY_KEYS
                    or normalized.endswith("path")
                ):
                    continue
                result[str(key)] = visit(child)
            return result
        if isinstance(nested, list):
            return [visit(child) for child in nested]
        return nested

    return visit(safe)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant is forbidden: {value}")
        ),
        object_pairs_hook=_strict_object,
    )
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant is forbidden: {item}")
                ),
                object_pairs_hook=_strict_object,
            )
            _require(isinstance(value, dict), f"non-object row: {path}:{line_number}")
            rows.append(value)
    _require(bool(rows), f"empty JSONL file: {path}")
    return rows


def _atomic_new(path: Path, payload: bytes) -> None:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _progress_image_sha256(image_bgr: np.ndarray) -> str:
    """Match the decoded-pixel identity used by legacy/CRRM adapters."""

    image = validate_canonical_roi(image_bgr)
    header = json.dumps(
        {
            "protocol": "canonical_meter_roi_bgr_uint8_v1",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "channel_order": "BGR",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _walk_items(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key), nested
            yield from _walk_items(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_items(nested)


def _verify_negative_attestations(value: Mapping[str, Any]) -> None:
    for key, nested in _walk_items(value):
        normalized = _normalized_key(key)
        if normalized in _NEGATIVE_ATTESTATION_KEYS and nested is not False:
            raise ProtocolViolation(f"provider reports forbidden consumption: {key}")
        if normalized in _NO_SECOND_ROI_KEYS and nested is True:
            raise ProtocolViolation(f"provider reports a second ROI operation: {key}")


def _find_values(value: Any, normalized_key: str) -> list[Any]:
    return [
        nested
        for key, nested in _walk_items(value)
        if _normalized_key(key) == normalized_key
    ]


@dataclass(frozen=True)
class FrozenComponentBinding:
    name: str
    provider_protocol: str
    provider_identity: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str]
    frozen: bool = True
    verified_complete: bool = True
    synthetic: bool = False

    def __post_init__(self) -> None:
        _require(bool(str(self.name).strip()), "component name is empty")
        _require(bool(str(self.provider_protocol).strip()), "provider protocol is empty")
        identity = _stable_identity(self.provider_identity)
        artifacts = {
            str(key): _require_sha256(value, name=f"{self.name}.artifact.{key}")
            for key, value in self.artifact_sha256.items()
        }
        sources = {
            str(key): _require_sha256(value, name=f"{self.name}.source.{key}")
            for key, value in self.source_sha256.items()
        }
        _require(bool(artifacts), f"{self.name}: at least one artifact hash is required")
        object.__setattr__(self, "provider_identity", identity)
        object.__setattr__(self, "artifact_sha256", artifacts)
        object.__setattr__(self, "source_sha256", sources)

    @property
    def provider_identity_sha256(self) -> str:
        return canonical_json_sha256(self.provider_identity)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": COMPONENT_PROTOCOL,
            "name": self.name,
            "provider_protocol": self.provider_protocol,
            "provider_identity": dict(self.provider_identity),
            "provider_identity_sha256": self.provider_identity_sha256,
            "artifact_sha256": dict(self.artifact_sha256),
            "source_sha256": dict(self.source_sha256),
            "frozen": bool(self.frozen),
            "verified_complete": bool(self.verified_complete),
            "synthetic": bool(self.synthetic),
        }

    @property
    def binding_sha256(self) -> str:
        return canonical_json_sha256(self.as_record())

    def write(self, path: Path) -> Path:
        target = Path(path).resolve()
        _atomic_new(target, _canonical_bytes(self.as_record()))
        return target

    @classmethod
    def from_provider(
        cls,
        *,
        name: str,
        provider: Any,
        artifact_sha256: Mapping[str, str],
        source_sha256: Mapping[str, str] | None = None,
        provider_protocol: str | None = None,
        synthetic: bool = False,
        verified_complete: bool = True,
    ) -> "FrozenComponentBinding":
        identity = getattr(provider, "identity", None)
        _require(isinstance(identity, Mapping), f"{name}: provider has no identity mapping")
        return cls(
            name=name,
            provider_protocol=str(
                provider_protocol or identity.get("protocol") or type(provider).__name__
            ),
            provider_identity=identity,
            artifact_sha256=artifact_sha256,
            source_sha256=source_sha256 or {},
            frozen=True,
            verified_complete=verified_complete,
            synthetic=synthetic,
        )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "FrozenComponentBinding":
        _require(set(value) == _COMPONENT_KEYS, "component binding schema drift")
        _require(value.get("schema_version") == 1, "component schema version drift")
        _require(value.get("protocol") == COMPONENT_PROTOCOL, "component protocol drift")
        result = cls(
            name=str(value.get("name") or ""),
            provider_protocol=str(value.get("provider_protocol") or ""),
            provider_identity=value.get("provider_identity") or {},
            artifact_sha256=value.get("artifact_sha256") or {},
            source_sha256=value.get("source_sha256") or {},
            frozen=value.get("frozen") is True,
            verified_complete=value.get("verified_complete") is True,
            synthetic=value.get("synthetic") is True,
        )
        _require(
            value.get("provider_identity_sha256") == result.provider_identity_sha256,
            f"{result.name}: provider identity hash drift",
        )
        return result

    def verify_provider(self, provider: Any) -> None:
        identity = getattr(provider, "identity", None)
        if not isinstance(identity, Mapping):
            raise ProtocolViolation(f"{self.name}: runtime provider lacks identity")
        actual = canonical_json_sha256(_stable_identity(identity))
        if actual != self.provider_identity_sha256:
            raise ProtocolViolation(f"{self.name}: runtime provider identity drift")


@dataclass(frozen=True)
class FrozenFullAutoBundle:
    descriptor: Mapping[str, Any]

    def __post_init__(self) -> None:
        value = _json_safe(self.descriptor)
        _require(set(value) == _BUNDLE_KEYS, "full-auto bundle schema drift")
        _require(value.get("schema_version") == 1, "bundle schema version drift")
        _require(value.get("protocol") == BUNDLE_PROTOCOL, "bundle protocol drift")
        _require(value.get("execution_mode") in EXECUTION_MODES, "execution mode drift")
        _require(value.get("output_mode") == OUTPUT_MODE_FULL_READING, "output mode drift")
        _require(value.get("evidence_role") == EVIDENCE_ROLE_PRIMARY, "evidence role drift")
        _require(value.get("roi_contract_sha256") == ROI_CONTRACT_SHA256, "ROI contract drift")
        mode = str(value.get("reference_mode") or "")
        _require(mode in (REFERENCE_MODE_AUTO, REFERENCE_MODE_NATIVE), "reference mode drift")
        expected_contract = (
            AUTO_REFERENCE_CONTRACT_SHA256
            if mode == REFERENCE_MODE_AUTO
            else NATIVE_REFERENCE_CONTRACT_SHA256
        )
        _require(
            value.get("reference_contract_sha256") == expected_contract,
            "reference contract hash drift",
        )
        method_name = str(value.get("method_name") or "")
        _require(bool(method_name), "method name is empty")
        if mode == REFERENCE_MODE_AUTO:
            _require(method_name.endswith("+auto-ref"), "auto method name must end +auto-ref")
            _require_sha256(
                value.get("reference_detector_sha256"),
                name="reference detector",
            )
        else:
            _require(
                not method_name.endswith("+auto-ref"),
                "native method name must not claim auto-reference",
            )
            _require(
                value.get("reference_detector_sha256") is None,
                "native progress must not bind a reference detector",
            )
        progress = FrozenComponentBinding.from_record(value["progress_component"])
        numeric_range = FrozenComponentBinding.from_record(
            value["automatic_numeric_range_component"]
        )
        _require(progress.name == "progress", "progress component name drift")
        _require(
            numeric_range.name == "automatic_numeric_range",
            "automatic numeric range component name drift",
        )
        for key in ("factory_source_sha256", "adapter_source_sha256"):
            _require_sha256(value.get(key), name=key)
        boundary = value.get("inference_boundary")
        _require(isinstance(boundary, Mapping), "inference boundary absent")
        _require(boundary.get("caller_range_allowed") is False, "caller range enabled")
        _require(boundary.get("caller_geometry_allowed") is False, "caller geometry enabled")
        _require(boundary.get("caller_reference_allowed") is False, "caller reference enabled")
        assert_label_free(value, location="full_auto_bundle")
        object.__setattr__(self, "descriptor", value)

    @property
    def method_name(self) -> str:
        return str(self.descriptor["method_name"])

    @property
    def execution_mode(self) -> str:
        return str(self.descriptor["execution_mode"])

    @property
    def reference_mode(self) -> str:
        return str(self.descriptor["reference_mode"])

    @property
    def reference_detector_sha256(self) -> str | None:
        value = self.descriptor.get("reference_detector_sha256")
        return None if value is None else str(value)

    @property
    def progress_binding(self) -> FrozenComponentBinding:
        return FrozenComponentBinding.from_record(self.descriptor["progress_component"])

    @property
    def range_binding(self) -> FrozenComponentBinding:
        return FrozenComponentBinding.from_record(
            self.descriptor["automatic_numeric_range_component"]
        )

    @property
    def range_binding_sha256(self) -> str:
        return self.range_binding.binding_sha256

    @property
    def payload(self) -> bytes:
        return _canonical_bytes(self.descriptor)

    @property
    def bundle_sha256(self) -> str:
        return _sha256_bytes(self.payload)

    @classmethod
    def create(
        cls,
        *,
        method_name: str,
        progress_binding: FrozenComponentBinding,
        range_binding: FrozenComponentBinding,
        factory_source_sha256: str,
        reference_mode: str,
        reference_detector_sha256: str | None = None,
        execution_mode: str = EXECUTION_FORMAL,
    ) -> "FrozenFullAutoBundle":
        _require(execution_mode in EXECUTION_MODES, "unsupported execution mode")
        if execution_mode == EXECUTION_FORMAL:
            for component in (progress_binding, range_binding):
                _require(component.frozen, f"{component.name}: component is not frozen")
                _require(
                    component.verified_complete,
                    f"{component.name}: component is not verified complete",
                )
                _require(not component.synthetic, f"{component.name}: synthetic component")
        source_hash = sha256_file(Path(__file__))
        descriptor = {
            "schema_version": 1,
            "protocol": BUNDLE_PROTOCOL,
            "method_name": str(method_name),
            "execution_mode": execution_mode,
            "reference_mode": reference_mode,
            "reference_contract_sha256": (
                AUTO_REFERENCE_CONTRACT_SHA256
                if reference_mode == REFERENCE_MODE_AUTO
                else NATIVE_REFERENCE_CONTRACT_SHA256
            ),
            "reference_detector_sha256": reference_detector_sha256,
            "roi_contract_sha256": ROI_CONTRACT_SHA256,
            "output_mode": OUTPUT_MODE_FULL_READING,
            "evidence_role": EVIDENCE_ROLE_PRIMARY,
            "progress_component": progress_binding.as_record(),
            "automatic_numeric_range_component": range_binding.as_record(),
            "factory_source_sha256": _require_sha256(
                factory_source_sha256, name="factory source"
            ),
            "adapter_source_sha256": source_hash,
            "inference_boundary": {
                "primary_input": "one unchanged canonical meter ROI BGR uint8",
                "caller_range_allowed": False,
                "caller_geometry_allowed": False,
                "caller_reference_allowed": False,
                "caller_crop_allowed": False,
                "full_success_requires": [
                    "valid progress in [0,1]",
                    "automatic finite nonzero numeric range",
                ],
            },
        }
        return cls(descriptor)

    def write(self, path: Path) -> Path:
        target = Path(path).resolve()
        _atomic_new(target, self.payload)
        _require(sha256_file(target) == self.bundle_sha256, "bundle write hash drift")
        return target

    @classmethod
    def load(cls, path: Path) -> "FrozenFullAutoBundle":
        resolved = Path(path).resolve(strict=True)
        result = cls(_strict_json(resolved))
        _require(sha256_file(resolved) == result.bundle_sha256, "bundle file hash drift")
        _require(
            result.descriptor["adapter_source_sha256"] == sha256_file(Path(__file__)),
            "full-auto adapter source changed after bundle freeze",
        )
        return result


@runtime_checkable
class ProgressProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ProgressResult:
    status: bool
    progress: float | None
    failure_code: str | None
    automatic_reference: Mapping[str, Any] | None
    telemetry: Mapping[str, Any]


def _normalize_reference_packet(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "status": False,
            "start_angle": None,
            "range_angle": None,
            "reference_branch": "full_auto_adapter:unavailable",
            "failure_code": "automatic_reference_attestation_unavailable",
        }
    status = value.get("status")
    if not isinstance(status, bool):
        # Convert the legacy canonical-ROI adapter attestation.
        detection = value.get("reference_detection")
        if isinstance(detection, Mapping) and isinstance(value.get("valid"), bool):
            status = bool(value.get("valid"))
            value = {
                "status": status,
                "start_angle": detection.get("start_angle_degrees"),
                "range_angle": detection.get("range_angle_degrees"),
                "reference_branch": str(detection.get("branch") or "unknown"),
                "failure_code": None if status else "automatic_reference_unavailable",
            }
        else:
            return {
                "status": False,
                "start_angle": None,
                "range_angle": None,
                "reference_branch": "full_auto_adapter:unavailable",
                "failure_code": "automatic_reference_attestation_unavailable",
            }
    start = _finite(value.get("start_angle"))
    arc = _finite(value.get("range_angle"))
    branch = str(value.get("reference_branch") or "unknown")
    if status:
        if start is None or arc is None or not 0.0 < arc < 360.0:
            return {
                "status": False,
                "start_angle": None,
                "range_angle": None,
                "reference_branch": "full_auto_adapter:invalid",
                "failure_code": "automatic_reference_attestation_invalid",
            }
        if branch.casefold().split(":")[-1] != "start_and_end":
            return {
                "status": False,
                "start_angle": None,
                "range_angle": None,
                "reference_branch": branch,
                "failure_code": "automatic_reference_incomplete",
            }
        return {
            "status": True,
            "start_angle": start,
            "range_angle": arc,
            "reference_branch": branch,
            "failure_code": None,
        }
    return {
        "status": False,
        "start_angle": None,
        "range_angle": None,
        "reference_branch": branch,
        "failure_code": str(value.get("failure_code") or "automatic_reference_unavailable"),
    }


def _normalize_progress_output(
    value: Any,
    *,
    reference_mode: str,
    expected_pixel_sha256: str,
    expected_reference_detector_sha256: str | None,
) -> ProgressResult:
    if not isinstance(value, Mapping):
        return ProgressResult(
            False, None, "progress_provider_returned_non_mapping", None, {}
        )
    safe = _json_safe(value)
    assert_label_free(safe, location="progress_provider_output")
    _verify_negative_attestations(safe)
    observed_hashes = {
        str(item).casefold()
        for item in _find_values(safe, "inputimagesha256")
        if item is not None
    }
    if observed_hashes and observed_hashes != {expected_pixel_sha256}:
        raise ProtocolViolation("progress provider decoded-pixel hash drift")
    status_value = safe.get("status")
    status = (
        status_value is True
        or (isinstance(status_value, str) and status_value.casefold() in {"ok", "success"})
    )
    progress_values = [
        _finite(safe.get("prediction_progress")),
        _finite(safe.get("progress")),
    ]
    present = [item for item in progress_values if item is not None]
    if len(present) == 2 and not math.isclose(present[0], present[1], abs_tol=1e-12):
        raise ProtocolViolation("progress provider exposes contradictory progress values")
    progress = present[0] if present else None
    if progress is None or not 0.0 <= progress <= 1.0:
        status = False
        progress = None
    failure = safe.get("failure_code")
    if failure is None and isinstance(safe.get("failure"), Mapping):
        failure = safe["failure"].get("code")
    automatic_reference: Mapping[str, Any] | None = None
    if reference_mode == REFERENCE_MODE_AUTO:
        detector_values = {
            str(item).casefold()
            for item in _find_values(safe, "referencedetectorsha256")
            if item is not None
        }
        expected_detector = str(expected_reference_detector_sha256 or "").casefold()
        if detector_values and detector_values != {expected_detector}:
            raise ProtocolViolation("progress reference detector hash drift")
        raw_reference = safe.get("auto_reference")
        if raw_reference is None:
            raw_reference = safe.get("automatic_reference")
        automatic_reference = _normalize_reference_packet(raw_reference)
        if not automatic_reference["status"]:
            status = False
            progress = None
            failure = automatic_reference["failure_code"]
    else:
        invoked_values = _find_values(safe, "referencedetectorinvoked")
        if any(value is not False for value in invoked_values):
            raise ProtocolViolation("native progress invoked a reference detector")
        if safe.get("auto_reference") is not None:
            raise ProtocolViolation("native progress emitted an auto-reference packet")
    if not status:
        progress = None
        failure = str(failure or "progress_unavailable")[:128]
    return ProgressResult(
        status=status,
        progress=progress,
        failure_code=None if status else failure,
        automatic_reference=automatic_reference,
        telemetry={
            "provider_protocol": safe.get("protocol"),
            "provider_status": status_value,
            "provider_record_sha256": canonical_json_sha256(safe),
            "observed_input_image_sha256": sorted(observed_hashes),
        },
    )


@dataclass(frozen=True)
class FullAutoPrediction:
    status: bool
    prediction_progress: float | None
    predicted_scale_start: float | None
    predicted_scale_end: float | None
    range_confidence: float | None
    failure_code: str | None
    automatic_reference: Mapping[str, Any] | None
    telemetry: Mapping[str, Any]

    def as_method_record(
        self,
        *,
        bundle: FrozenFullAutoBundle,
        canonical_roi_file_sha256: str,
    ) -> dict[str, Any]:
        file_hash = _require_sha256(
            canonical_roi_file_sha256, name="canonical ROI file"
        )
        record: dict[str, Any] = {
            "status": bool(self.status),
            "prediction_progress": self.prediction_progress,
            "failure_code": self.failure_code,
            "checkpoint_sha256": bundle.bundle_sha256,
            "roi_contract_sha256": ROI_CONTRACT_SHA256,
            "reference_mode": bundle.reference_mode,
            "reference_contract_sha256": (
                AUTO_REFERENCE_CONTRACT_SHA256
                if bundle.reference_mode == REFERENCE_MODE_AUTO
                else NATIVE_REFERENCE_CONTRACT_SHA256
            ),
            "evidence_role": EVIDENCE_ROLE_PRIMARY,
            "output_mode": OUTPUT_MODE_FULL_READING,
            "canonical_roi_sha256": file_hash,
            "predicted_scale_start": self.predicted_scale_start,
            "predicted_scale_end": self.predicted_scale_end,
            "range_confidence": self.range_confidence,
            "telemetry": _json_safe(self.telemetry),
        }
        if bundle.reference_mode == REFERENCE_MODE_AUTO:
            reference = _normalize_reference_packet(self.automatic_reference)
            record.update(
                {
                    "reference_detector_sha256": bundle.reference_detector_sha256,
                    "auto_reference": reference,
                    "auto_reference_sha256": canonical_json_sha256(reference),
                }
            )
        _require(
            set(record).issubset(METHOD_PREDICTION_KEYS),
            "full-auto method record has unsupported fields",
        )
        assert_label_free(record, location="full_auto_method_record")
        return record


class UnifiedFullAutoAdapter:
    """Compose one progress provider and one automatic range pipeline."""

    def __init__(
        self,
        *,
        progress_provider: ProgressProvider,
        automatic_numeric_range_pipeline: AutomaticNumericRangePipeline,
        bundle: FrozenFullAutoBundle,
    ) -> None:
        if not isinstance(progress_provider, ProgressProvider):
            raise TypeError("progress_provider does not implement ProgressProvider")
        if not isinstance(automatic_numeric_range_pipeline, AutomaticNumericRangePipeline):
            raise TypeError(
                "automatic_numeric_range_pipeline must be AutomaticNumericRangePipeline"
            )
        bundle.progress_binding.verify_provider(progress_provider)
        bundle.range_binding.verify_provider(automatic_numeric_range_pipeline)
        self.progress_provider = progress_provider
        self.range_pipeline = automatic_numeric_range_pipeline
        self.bundle = bundle

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "method_name": self.bundle.method_name,
            "bundle_sha256": self.bundle.bundle_sha256,
            "range_binding_sha256": self.bundle.range_binding_sha256,
            "primary_input": "one unchanged canonical meter ROI BGR uint8",
            "manual_range_input": False,
            "manual_geometry_input": False,
            "manual_reference_input": False,
        }

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> FullAutoPrediction:
        """Infer full reading factors from image only.

        Keyword expansion such as ``scale_start=``, ``geometry=``, or
        ``reference=`` is rejected by Python before either provider runs.
        """

        if input_is_canonical_meter_roi is not True:
            raise ValueError("full-auto adapter requires the canonical meter ROI")
        image = validate_canonical_roi(canonical_meter_roi_bgr)
        progress_digest = _progress_image_sha256(image)
        numeric_range_digest = range_image_sha256(image)
        progress_image = image.copy()
        range_image = image.copy()
        progress_output: Any
        try:
            progress_output = self.progress_provider.predict(
                progress_image,
                input_is_canonical_meter_roi=True,
            )
        except ProtocolViolation:
            raise
        except Exception as exc:  # A model failure is a row failure, not cohort abort.
            progress_output = {
                "status": "failed",
                "progress": None,
                "failure": {"code": f"progress_provider_exception:{type(exc).__name__}"},
                "input_attestation": {
                    "input_image_sha256": progress_digest,
                    "ground_truth_consumed": False,
                    "physical_scale_consumed": False,
                    "manual_reference_consumed": False,
                },
            }
        if _progress_image_sha256(progress_image) != progress_digest:
            raise ProtocolViolation("progress provider mutated its canonical ROI input")
        progress = _normalize_progress_output(
            progress_output,
            reference_mode=self.bundle.reference_mode,
            expected_pixel_sha256=progress_digest,
            expected_reference_detector_sha256=self.bundle.reference_detector_sha256,
        )
        try:
            numeric_range = self.range_pipeline.predict(range_image)
        except ProtocolViolation:
            raise
        except Exception as exc:
            numeric_range = NumericRangePrediction(
                protocol=AUTOMATIC_RANGE_PROTOCOL,
                status=False,
                prediction_space="real_numeric_scale_start_end",
                pred_start=None,
                pred_end=None,
                confidence=0.0,
                failure_reason=f"range_provider_exception:{type(exc).__name__}",
                telemetry={},
            )
        if range_image_sha256(range_image) != numeric_range_digest:
            raise ProtocolViolation("range provider mutated its canonical ROI input")
        if not isinstance(numeric_range, NumericRangePrediction):
            raise ProtocolViolation("range provider returned an invalid result type")
        range_record = _json_safe(numeric_range.as_dict())
        assert_label_free(range_record, location="automatic_numeric_range.output")
        range_telemetry = range_record["telemetry"]
        assert_label_free(range_telemetry, location="automatic_numeric_range.telemetry")
        adapter_attestation = range_telemetry.get("primary_adapter")
        if not isinstance(adapter_attestation, Mapping):
            raise ProtocolViolation("range provider lacks primary input attestation")
        if (
            adapter_attestation.get("input_image_sha256") != numeric_range_digest
            or adapter_attestation.get("geometry_and_ocr_same_image_sha256")
            != numeric_range_digest
        ):
            raise ProtocolViolation("range geometry/OCR image binding drift")
        if (
            adapter_attestation.get("accepts_manual_geometry") is not False
            or adapter_attestation.get("accepts_reference_packet") is not False
            or adapter_attestation.get("accepts_physical_scale_values") is not False
        ):
            raise ProtocolViolation("range provider exposes a supervised/manual input")
        start = _finite(numeric_range.pred_start)
        end = _finite(numeric_range.pred_end)
        confidence = _finite(numeric_range.confidence)
        range_ok = bool(
            numeric_range.status
            and start is not None
            and end is not None
            and start != end
            and confidence is not None
            and 0.0 <= confidence <= 1.0
        )
        full_status = bool(progress.status and range_ok)
        failures: list[str] = []
        if not progress.status:
            failures.append(f"progress:{progress.failure_code}")
        if not range_ok:
            failures.append(
                f"automatic_numeric_range:{numeric_range.failure_reason or 'invalid_output'}"
            )
        telemetry = {
            "protocol": PROTOCOL,
            "execution_mode": self.bundle.execution_mode,
            "claim_eligible": self.bundle.execution_mode == EXECUTION_FORMAL,
            "bundle_sha256": self.bundle.bundle_sha256,
            "range_binding_sha256": self.bundle.range_binding_sha256,
            "decoded_pixel_binding": {
                "progress_identity_sha256": progress_digest,
                "automatic_numeric_range_identity_sha256": numeric_range_digest,
                "same_decoded_array_content": True,
                "provider_inputs_isolated_against_mutation": True,
            },
            "progress_component": {
                "status": progress.status,
                "failure_code": progress.failure_code,
                **dict(progress.telemetry),
            },
            "automatic_numeric_range_component": {
                "status": range_ok,
                "failure_code": None if range_ok else numeric_range.failure_reason,
                "prediction_protocol": numeric_range.protocol,
                "prediction_space": numeric_range.prediction_space,
                "prediction_record_sha256": canonical_json_sha256(
                    range_record
                ),
            },
            "input_contract": {
                "one_whole_canonical_roi": True,
                "caller_range_consumed": False,
                "caller_geometry_consumed": False,
                "caller_reference_consumed": False,
                "label_consumed": False,
                "second_crop_applied": False,
            },
        }
        return FullAutoPrediction(
            status=full_status,
            prediction_progress=progress.progress if full_status else None,
            predicted_scale_start=start if full_status else None,
            predicted_scale_end=end if full_status else None,
            range_confidence=confidence if full_status else None,
            failure_code=None if full_status else ";".join(failures)[:256],
            automatic_reference=progress.automatic_reference,
            telemetry=telemetry,
        )


@dataclass(frozen=True)
class ManifestItem:
    sample_id: str
    group_id: str
    image_path: Path
    image_sha256: str
    frame_sha256: str


def validate_label_free_manifest(rows: Sequence[Mapping[str, Any]]) -> list[ManifestItem]:
    """Validate every row before provider loading or image I/O."""

    assert_label_free(rows, location="full_auto_manifest")
    _require(bool(rows), "full-auto manifest is empty")
    result: list[ManifestItem] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, 1):
        row = dict(raw)
        _require(set(row) == _MANIFEST_KEYS, f"row {index}: manifest schema drift")
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id) and bool(group_id), f"row {index}: missing identity")
        _require(sample_id not in seen, f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        image_hash = _require_sha256(row.get("image_sha256"), name=f"{sample_id}.image")
        canonical_hash = _require_sha256(
            row.get("canonical_roi_sha256"), name=f"{sample_id}.canonical ROI"
        )
        _require(canonical_hash == image_hash, f"{sample_id}: ROI is not the supplied image")
        frame_hash = _require_sha256(
            row.get("frame_sha256"), name=f"{sample_id}.frame"
        )
        _require(
            row.get("roi_contract_sha256") == ROI_CONTRACT_SHA256,
            f"{sample_id}: ROI contract drift",
        )
        path = Path(str(row.get("image_path") or ""))
        _require(str(path) not in ("", "."), f"{sample_id}: image path absent")
        result.append(
            ManifestItem(sample_id, group_id, path.resolve(), image_hash, frame_hash)
        )
    return result


def _decode_bound_image(item: ManifestItem) -> tuple[np.ndarray | None, str | None]:
    try:
        payload = item.image_path.read_bytes()
    except OSError:
        return None, "image_read_failed"
    if hashlib.sha256(payload).hexdigest() != item.image_sha256:
        raise ProtocolViolation(f"{item.sample_id}: image file hash drift")
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        return None, "image_decode_failed"
    return validate_canonical_roi(image), None


def _input_failure(
    *, bundle: FrozenFullAutoBundle, failure_code: str
) -> FullAutoPrediction:
    reference = None
    if bundle.reference_mode == REFERENCE_MODE_AUTO:
        reference = {
            "status": False,
            "start_angle": None,
            "range_angle": None,
            "reference_branch": "full_auto_adapter:not_invoked",
            "failure_code": "image_unavailable_for_automatic_reference",
        }
    return FullAutoPrediction(
        status=False,
        prediction_progress=None,
        predicted_scale_start=None,
        predicted_scale_end=None,
        range_confidence=None,
        failure_code=failure_code,
        automatic_reference=reference,
        telemetry={
            "protocol": PROTOCOL,
            "execution_mode": bundle.execution_mode,
            "claim_eligible": bundle.execution_mode == EXECUTION_FORMAL,
            "bundle_sha256": bundle.bundle_sha256,
            "input_failure": failure_code,
            "providers_invoked": False,
        },
    )


def run_label_free_manifest(
    *,
    input_path: Path,
    output_path: Path,
    adapter: UnifiedFullAutoAdapter,
) -> dict[str, Any]:
    input_file = Path(input_path).resolve(strict=True)
    rows = _strict_jsonl(input_file)
    # Complete validation precedes all image I/O.
    items = validate_label_free_manifest(rows)
    outputs: list[dict[str, Any]] = []
    success = 0
    for item in items:
        image, failure = _decode_bound_image(item)
        if image is None:
            prediction = _input_failure(
                bundle=adapter.bundle,
                failure_code=str(failure or "image_unavailable"),
            )
        else:
            prediction = adapter.predict(
                image, input_is_canonical_meter_roi=True
            )
        method_record = prediction.as_method_record(
            bundle=adapter.bundle,
            canonical_roi_file_sha256=item.image_sha256,
        )
        success += int(prediction.status)
        outputs.append(
            {
                "schema_version": 1,
                "protocol": INFERENCE_PROTOCOL,
                "sample_id": item.sample_id,
                "group_id": item.group_id,
                "image_sha256": item.image_sha256,
                "canonical_roi_sha256": item.image_sha256,
                "frame_sha256": item.frame_sha256,
                "reference_input": None,
                "reference_input_sha256": None,
                "methods": {adapter.bundle.method_name: method_record},
            }
        )
    assert_label_free(outputs, location="full_auto_raw_predictions")
    output = Path(output_path).resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            "refusing to overwrite full-auto output or its metadata sidecar"
        )
    _atomic_new(output, b"".join(_canonical_bytes(row) for row in outputs))
    metadata = {
        "schema_version": 1,
        "protocol": RUN_PROTOCOL,
        "status": "complete_label_free_predictions",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_eligible": adapter.bundle.execution_mode == EXECUTION_FORMAL,
        "label_files_opened": 0,
        "range_values_supplied_by_caller": 0,
        "geometry_packets_supplied_by_caller": 0,
        "reference_packets_supplied_by_caller": 0,
        "input": {
            "path": str(input_file),
            "sha256": sha256_file(input_file),
            "rows": len(items),
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": len(outputs),
            "successful": success,
            "failures": len(outputs) - success,
        },
        "method": {
            "name": adapter.bundle.method_name,
            "bundle_sha256": adapter.bundle.bundle_sha256,
            "range_binding_sha256": adapter.bundle.range_binding_sha256,
            "reference_mode": adapter.bundle.reference_mode,
            "output_mode": OUTPUT_MODE_FULL_READING,
        },
        "adapter_source_sha256": sha256_file(Path(__file__)),
    }
    _atomic_new(metadata_path, _canonical_bytes(metadata))
    return metadata


def build_evaluator_method_binding(
    *,
    bundle_path: Path,
    reference_detector_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create one entry for ``v5_unified_two_stage_retest.freeze``."""

    bundle_file = Path(bundle_path).resolve(strict=True)
    bundle = FrozenFullAutoBundle.load(bundle_file)
    _require(
        bundle.execution_mode == EXECUTION_FORMAL,
        "synthetic bundle cannot enter evaluator primary protocol",
    )
    detector_path: Path | None = None
    if bundle.reference_mode == REFERENCE_MODE_AUTO:
        _require(reference_detector_path is not None, "reference detector path required")
        detector_path = Path(reference_detector_path).resolve(strict=True)
        _require(
            sha256_file(detector_path) == bundle.reference_detector_sha256,
            "reference detector file differs from bundle",
        )
    else:
        _require(reference_detector_path is None, "native mode forbids detector path")
    binding = {
        "checkpoint_path": str(bundle_file),
        "checkpoint_sha256": bundle.bundle_sha256,
        "source_evaluation_checkpoint_sha256": bundle.bundle_sha256,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__)),
        "roi_contract_sha256": ROI_CONTRACT_SHA256,
        "reference_mode": bundle.reference_mode,
        "reference_contract_sha256": (
            AUTO_REFERENCE_CONTRACT_SHA256
            if bundle.reference_mode == REFERENCE_MODE_AUTO
            else NATIVE_REFERENCE_CONTRACT_SHA256
        ),
        "reference_detector_path": None if detector_path is None else str(detector_path),
        "reference_detector_sha256": bundle.reference_detector_sha256,
        "evidence_role": EVIDENCE_ROLE_PRIMARY,
        "output_mode": OUTPUT_MODE_FULL_READING,
    }
    return bundle.method_name, binding


def _load_factory(path: Path, function_name: str) -> tuple[Any, str]:
    source = Path(path).resolve(strict=True)
    digest = sha256_file(source)
    module_name = f"v5_full_auto_factory_{digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load provider factory: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"factory function is absent: {function_name}")
    return factory, digest


def _factory_providers(factory: Any, bundle: FrozenFullAutoBundle) -> tuple[Any, Any]:
    value = factory(dict(bundle.descriptor))
    if not isinstance(value, Mapping) or set(value) != {
        "progress_provider",
        "automatic_numeric_range_pipeline",
    }:
        raise TypeError("factory must return exactly two named providers")
    return value["progress_provider"], value["automatic_numeric_range_pipeline"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-bundle")
    freeze.add_argument("--method-name", required=True)
    freeze.add_argument(
        "--reference-mode",
        choices=(REFERENCE_MODE_AUTO, REFERENCE_MODE_NATIVE),
        required=True,
    )
    freeze.add_argument("--progress-binding", type=Path, required=True)
    freeze.add_argument("--range-binding", type=Path, required=True)
    freeze.add_argument("--factory-file", type=Path, required=True)
    freeze.add_argument("--reference-detector", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--synthetic-smoke", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--factory-file", type=Path, required=True)
    run.add_argument("--factory-function", default=FACTORY_FUNCTION_DEFAULT)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    emit = commands.add_parser("emit-evaluator-bundle")
    emit.add_argument("--bundle", type=Path, required=True)
    emit.add_argument("--reference-detector", type=Path)
    emit.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze-bundle":
        progress = FrozenComponentBinding.from_record(_strict_json(args.progress_binding))
        numeric_range = FrozenComponentBinding.from_record(_strict_json(args.range_binding))
        detector_hash = (
            None
            if args.reference_detector is None
            else sha256_file(Path(args.reference_detector).resolve(strict=True))
        )
        bundle = FrozenFullAutoBundle.create(
            method_name=args.method_name,
            progress_binding=progress,
            range_binding=numeric_range,
            factory_source_sha256=sha256_file(
                Path(args.factory_file).resolve(strict=True)
            ),
            reference_mode=args.reference_mode,
            reference_detector_sha256=detector_hash,
            execution_mode=(
                EXECUTION_SYNTHETIC if args.synthetic_smoke else EXECUTION_FORMAL
            ),
        )
        path = bundle.write(args.output)
        result = {"bundle": str(path), "sha256": bundle.bundle_sha256}
    elif args.command == "run":
        bundle = FrozenFullAutoBundle.load(args.bundle)
        factory, factory_hash = _load_factory(args.factory_file, args.factory_function)
        _require(
            factory_hash == bundle.descriptor["factory_source_sha256"],
            "provider factory source changed after bundle freeze",
        )
        progress_provider, range_pipeline = _factory_providers(factory, bundle)
        adapter = UnifiedFullAutoAdapter(
            progress_provider=progress_provider,
            automatic_numeric_range_pipeline=range_pipeline,
            bundle=bundle,
        )
        result = run_label_free_manifest(
            input_path=args.input,
            output_path=args.output,
            adapter=adapter,
        )
    elif args.command == "emit-evaluator-bundle":
        name, binding = build_evaluator_method_binding(
            bundle_path=args.bundle,
            reference_detector_path=args.reference_detector,
        )
        output = Path(args.output).resolve()
        _atomic_new(output, _canonical_bytes({"methods": {name: binding}}))
        result = {"method_bundle": str(output), "sha256": sha256_file(output)}
    else:  # pragma: no cover - argparse prevents this branch.
        raise RuntimeError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


__all__ = [
    "BUNDLE_PROTOCOL",
    "COMPONENT_PROTOCOL",
    "EXECUTION_FORMAL",
    "EXECUTION_SYNTHETIC",
    "FrozenComponentBinding",
    "FrozenFullAutoBundle",
    "FullAutoPrediction",
    "PROTOCOL",
    "ProgressProvider",
    "ProtocolViolation",
    "UnifiedFullAutoAdapter",
    "build_evaluator_method_binding",
    "run_label_free_manifest",
    "validate_label_free_manifest",
]


if __name__ == "__main__":
    main()
