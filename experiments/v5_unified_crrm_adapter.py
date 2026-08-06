"""Label-free canonical-ROI adapter for the frozen V5 CRRM model.

This progress-component inference boundary is intentionally narrow: one
canonical meter ROI enters and one normalized progress in ``[0, 1]`` leaves.  The
adapter never accepts ground truth, physical scale endpoints, manual
ScaleMarks, a meter box, or a precomputed reference packet.  Physical-unit
conversion and target joining belong to a separate scoring/production layer.
Because this adapter does not predict the numeric scale start/end values, its
output is diagnostic and cannot by itself support the full automatic-reading
primary metric.

Four frozen visual providers are dependency-injected:

* the production automatic start/end reference provider;
* PEPD;
* the V5 ScaleMark expert; and
* Base Mask-Geometry.

Their three normalized-progress predictions are ordered for the gate as
``(mask_geometry, pepd, scalemark)``.  Exactly 23 deployable, label-free
telemetry features are assembled, missing experts are explicitly masked, and
the frozen CRRM gate performs soft reliability fusion.  Every formal binding,
input, schema, intermediate result, and output is SHA-256 attested.

This module contains a conspicuously marked in-memory synthetic smoke path.
It is not eligible for metrics and must never be reported as a model result.
No dataset is read by the smoke path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from experiments.cagh_v5_crrm_gate import (
    EXPERT_NAMES,
    FEATURE_DIM,
    PROTOCOL as GATE_PROTOCOL,
    CRRMV5SoftReliabilityGate,
)


PROTOCOL: Final[str] = "v5_unified_crrm_canonical_roi_label_free_v2"
PREDICTION_SPACE: Final[str] = "normalized_progress_0_1"
EVIDENCE_ROLE: Final[str] = "secondary_progress_component_diagnostic"
OUTPUT_MODE: Final[str] = "progress_only_component"
EXECUTION_MODES: Final[frozenset[str]] = frozenset(
    {"formal_frozen", "synthetic_smoke"}
)

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "mask_geometry_confidence",
    "mask_line_residual",
    "mask_center_distance",
    "mask_axis_length",
    "mask_foreground_fraction",
    "pepd_pivot_peak",
    "pepd_angle_std_degrees",
    "pepd_angle_entropy",
    "pepd_bin_resultant_length",
    "pepd_pointer_length",
    "pepd_direction_quality",
    "scalemark_start_peak",
    "scalemark_end_peak",
    "scalemark_start_entropy",
    "scalemark_end_entropy",
    "scalemark_endpoint_separation",
    "scalemark_tick_scale_disagreement",
    "scalemark_endpoint_coordinate_disagreement",
    "scalemark_endpoint_js_divergence",
    "scalemark_radius_reliability",
    "scalemark_arc_length_reliability",
    "scalemark_gate_confidence",
    "scalemark_uncertainty_score",
)
if len(FEATURE_NAMES) != FEATURE_DIM:
    raise RuntimeError("CRRM feature roster drift")

FEATURE_OWNER: Final[Mapping[str, str]] = {
    **{name: "mask_geometry" for name in FEATURE_NAMES[:5]},
    **{name: "pepd" for name in FEATURE_NAMES[5:11]},
    **{name: "scalemark" for name in FEATURE_NAMES[11:]},
}
PROVIDER_NAMES: Final[tuple[str, ...]] = (
    "automatic_reference",
    "pepd",
    "scalemark",
    "mask_geometry",
)

OUTPUT_SCHEMA: Final[Mapping[str, Any]] = {
    "schema_version": 1,
    "protocol": PROTOCOL,
    "primary_input": "one canonical meter ROI (BGR uint8)",
    "output": PREDICTION_SPACE,
    "complete_automatic_reading": False,
    "forbidden_primary_inputs": [
        "ground_truth",
        "physical_scale_start",
        "physical_scale_end",
        "manual_scalemark",
        "manual_reference",
        "precomputed_reference_packet",
        "meter_bbox",
    ],
    "expert_order": list(EXPERT_NAMES),
    "feature_names": list(FEATURE_NAMES),
    "required": [
        "protocol",
        "status",
        "evidence_role",
        "output_mode",
        "eligible_for_primary_metrics",
        "prediction_space",
        "progress",
        "expert_progress",
        "expert_availability",
        "features",
        "feature_availability",
        "gate",
        "automatic_reference",
        "input_attestation",
        "component_bindings",
        "output_schema_sha256",
        "adapter_source_sha256",
        "record_sha256",
    ],
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "actual",
        "actualreading",
        "actualvalue",
        "expected",
        "groundtruth",
        "gt",
        "label",
        "labels",
        "manualreference",
        "manualscalemark",
        "meterbbox",
        "physicalscaleend",
        "physicalscalemax",
        "physicalscalemin",
        "physicalscalerange",
        "physicalscalestart",
        "rangeangle",
        "referencepacket",
        "scaleend",
        "scalemax",
        "scalemin",
        "scalerange",
        "scalemarkreference",
        "scalemarkpositions",
        "scalestart",
        "startangle",
        "target",
        "targetdirection",
        "targetprogress",
        "targetreading",
        "targetvalue",
        "trueprogress",
        "truereading",
        "truevalue",
        "truth",
    }
)


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


OUTPUT_SCHEMA_SHA256: Final[str] = canonical_sha256(OUTPUT_SCHEMA)
FEATURE_SCHEMA_SHA256: Final[str] = canonical_sha256(
    {"feature_names": FEATURE_NAMES, "feature_owner": FEATURE_OWNER}
)


def sha256_file(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    digest = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _reject_forbidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden supervised/manual field at {path}.{key}")
            _reject_forbidden_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_fields(nested, path=f"{path}[{index}]")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.ndim == 0:
            return _json_safe(detached.item())
        return _json_safe(detached.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(nested) for nested in value]
    raise TypeError(f"telemetry contains unsupported type: {type(value).__name__}")


def _finite_optional(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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
class ComponentIdentity:
    """Immutable, path-free identity for one injected frozen component."""

    name: str
    protocol: str
    artifact_sha256: Mapping[str, str]
    source_sha256: Mapping[str, str] = field(default_factory=dict)
    frozen: bool = True
    verified_complete: bool = True
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.protocol).strip():
            raise ValueError("component name and protocol must be non-empty")
        artifacts = {
            str(key): _require_sha256(value, name=f"{self.name}.{key}")
            for key, value in self.artifact_sha256.items()
        }
        sources = {
            str(key): _require_sha256(value, name=f"{self.name}.{key}")
            for key, value in self.source_sha256.items()
        }
        if not artifacts:
            raise ValueError(f"{self.name} requires at least one artifact hash")
        object.__setattr__(self, "artifact_sha256", artifacts)
        object.__setattr__(self, "source_sha256", sources)

    def as_record(self) -> dict[str, Any]:
        record = {
            "name": self.name,
            "protocol": self.protocol,
            "artifact_sha256": dict(self.artifact_sha256),
            "source_sha256": dict(self.source_sha256),
            "frozen": bool(self.frozen),
            "verified_complete": bool(self.verified_complete),
            "synthetic": bool(self.synthetic),
        }
        record["identity_sha256"] = canonical_sha256(record)
        return record


@dataclass(frozen=True)
class AutomaticReferenceResult:
    """Automatic production reference derived only from the current ROI."""

    available: bool
    start_xy: tuple[float, float] | None
    end_xy: tuple[float, float] | None
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None
    context: Any = field(default=None, repr=False, compare=False)

    def as_record(self, *, input_image_sha256: str) -> dict[str, Any]:
        telemetry = _json_safe(self.telemetry)
        _reject_forbidden_fields(telemetry, path="automatic_reference.telemetry")
        start = _point_or_none(self.start_xy)
        end = _point_or_none(self.end_xy)
        available = bool(self.available and start is not None and end is not None)
        record = {
            "available": available,
            "source": "automatic_production_reference",
            "scope": "unchanged canonical meter ROI",
            "start_xy": start,
            "end_xy": end,
            "failure_code": None if available else str(self.failure_code or "reference_unavailable")[:96],
            "telemetry": telemetry,
            "input_image_sha256": input_image_sha256,
            "manual_reference_consumed": False,
            "reference_packet_supplied_by_caller": False,
        }
        record["result_sha256"] = canonical_sha256(record)
        return record


@dataclass(frozen=True)
class ExpertResult:
    """One label-free normalized-progress expert output."""

    progress: float | None
    available: bool
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None
    context: Any = field(default=None, repr=False, compare=False)

    def normalized(self, *, name: str) -> "ExpertResult":
        progress = _finite_optional(self.progress)
        available = bool(self.available and progress is not None and 0.0 <= progress <= 1.0)
        telemetry = _json_safe(self.telemetry)
        _reject_forbidden_fields(telemetry, path=f"{name}.telemetry")
        return ExpertResult(
            progress=progress if available else None,
            available=available,
            telemetry=telemetry,
            failure_code=(
                None
                if available
                else str(self.failure_code or "non_finite_or_out_of_range_progress")[:96]
            ),
            context=self.context,
        )

    def as_record(self, *, name: str, input_image_sha256: str) -> dict[str, Any]:
        normalized = self.normalized(name=name)
        record = {
            "name": name,
            "prediction_space": PREDICTION_SPACE,
            "progress": normalized.progress,
            "available": normalized.available,
            "failure_code": normalized.failure_code,
            "telemetry": normalized.telemetry,
            "input_image_sha256": input_image_sha256,
            "ground_truth_consumed": False,
            "physical_scale_consumed": False,
            "manual_reference_consumed": False,
        }
        record["result_sha256"] = canonical_sha256(record)
        return record


def _point_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) < 2:
        return None
    x_value, y_value = _finite_optional(value[0]), _finite_optional(value[1])
    return None if x_value is None or y_value is None else [x_value, y_value]


@runtime_checkable
class AutomaticReferenceProvider(Protocol):
    identity: ComponentIdentity

    def predict(self, canonical_meter_roi_bgr: np.ndarray) -> AutomaticReferenceResult:
        ...


@runtime_checkable
class ProgressExpertProvider(Protocol):
    identity: ComponentIdentity

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        automatic_reference: AutomaticReferenceResult,
        dependencies: Mapping[str, ExpertResult],
    ) -> ExpertResult:
        ...


@dataclass(frozen=True)
class FrozenGateBinding:
    module: CRRMV5SoftReliabilityGate
    identity: ComponentIdentity

    def __post_init__(self) -> None:
        if self.identity.name != "crrm_gate":
            raise ValueError("gate binding identity must be named crrm_gate")
        if self.identity.protocol != GATE_PROTOCOL:
            raise ValueError("gate protocol drift")
        if self.module.training:
            raise ValueError("CRRM gate must be in eval mode")
        if any(parameter.requires_grad for parameter in self.module.parameters()):
            raise ValueError("CRRM gate parameters must be frozen")

    @property
    def device(self) -> torch.device:
        parameter = next(self.module.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.module.buffers(), None)
        return torch.device("cpu") if buffer is None else buffer.device


@dataclass(frozen=True)
class FrozenModuleBinding:
    """Authenticated frozen module for a concrete provider implementation."""

    module: torch.nn.Module
    identity: ComponentIdentity
    checkpoint_metadata: Mapping[str, Any]


class _IdentityBoundProvider:
    def __init__(self, provider: Any, identity: ComponentIdentity) -> None:
        if not callable(getattr(provider, "predict", None)):
            raise TypeError("provider must expose predict")
        self._provider = provider
        self.identity = identity

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._provider.predict(*args, **kwargs)


def bind_frozen_provider(
    provider: Any,
    *,
    name: str,
    protocol: str,
    artifact_paths: Mapping[str, Path],
    expected_artifact_sha256: Mapping[str, str],
    source_paths: Mapping[str, Path] | None = None,
    verified_complete: bool = True,
) -> Any:
    """Authenticate files and attach a path-free identity to a provider.

    Model construction remains provider-specific; this helper is the common
    frozen loading boundary for production reference and Base Mask-Geometry.
    """

    if set(artifact_paths) != set(expected_artifact_sha256):
        raise ValueError("artifact path/hash rosters differ")
    artifact_hashes: dict[str, str] = {}
    for key, path in artifact_paths.items():
        actual = sha256_file(Path(path))
        expected = _require_sha256(expected_artifact_sha256[key], name=f"expected {name}.{key}")
        if actual != expected:
            raise ValueError(f"{name}.{key} checkpoint SHA-256 mismatch")
        artifact_hashes[str(key)] = actual
    source_hashes = {
        str(key): sha256_file(Path(path)) for key, path in (source_paths or {}).items()
    }
    identity = ComponentIdentity(
        name=name,
        protocol=protocol,
        artifact_sha256=artifact_hashes,
        source_sha256=source_hashes,
        frozen=True,
        verified_complete=bool(verified_complete),
        synthetic=False,
    )
    return _IdentityBoundProvider(provider, identity)


def _authenticated_checkpoint(
    checkpoint: Path,
    expected_sha256: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    path = Path(checkpoint).resolve(strict=True)
    expected = _require_sha256(expected_sha256, name="expected checkpoint SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return path, actual, payload


def _freeze(module: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    module.to(torch.device(device))
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def load_frozen_crrm_gate(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
) -> FrozenGateBinding:
    """Load and strictly authenticate the terminal all-public CRRM gate."""

    _, actual, payload = _authenticated_checkpoint(checkpoint, expected_sha256)
    from experiments.train_cagh_v5_crrm_gate import (
        FEATURE_NAMES as TRAINING_FEATURE_NAMES,
        PROTOCOL as TRAINING_PROTOCOL,
    )

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported CRRM checkpoint schema")
    if payload.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("CRRM training protocol drift")
    if payload.get("gate_protocol") != GATE_PROTOCOL:
        raise ValueError("CRRM gate protocol drift")
    if payload.get("status") != "complete":
        raise ValueError("CRRM checkpoint is not complete")
    if tuple(payload.get("feature_names") or ()) != tuple(TRAINING_FEATURE_NAMES):
        raise ValueError("CRRM checkpoint feature roster drift")
    if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("adapter/CRRM feature roster drift")
    if tuple(payload.get("expert_names") or ()) != tuple(EXPERT_NAMES):
        raise ValueError("CRRM checkpoint expert roster drift")
    state = payload.get("gate_state")
    if not isinstance(state, Mapping):
        raise ValueError("CRRM checkpoint has no gate_state")
    gate = CRRMV5SoftReliabilityGate()
    gate.load_state_dict(dict(state), strict=True)
    _freeze(gate, device)
    source = Path(__file__).resolve().with_name("cagh_v5_crrm_gate.py")
    identity = ComponentIdentity(
        name="crrm_gate",
        protocol=GATE_PROTOCOL,
        artifact_sha256={"checkpoint": actual},
        source_sha256={"gate_source": sha256_file(source)},
    )
    return FrozenGateBinding(gate, identity)


def load_frozen_scalemark_v5_head(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
) -> FrozenModuleBinding:
    """Load the authenticated terminal V5 ScaleMark head for a provider."""

    _, actual, payload = _authenticated_checkpoint(checkpoint, expected_sha256)
    from experiments.cagh_scalemark_reference_head_v5 import (
        CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        build_head,
    )
    from experiments.train_cagh_scalemark_reference_probe_v5_enhanced import (
        PROTOCOL as TRAINING_PROTOCOL,
    )

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ScaleMark checkpoint schema")
    if payload.get("protocol") != TRAINING_PROTOCOL:
        raise ValueError("ScaleMark training protocol drift")
    if payload.get("head_protocol") != CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL:
        raise ValueError("ScaleMark head protocol drift")
    if payload.get("status") != "complete":
        raise ValueError("ScaleMark checkpoint is not complete")
    state = payload.get("head_state")
    if not isinstance(state, Mapping):
        raise ValueError("ScaleMark checkpoint has no head_state")
    head = build_head()
    head.load_state_dict(dict(state), strict=True)
    _freeze(head, device)
    source = Path(__file__).resolve().with_name("cagh_scalemark_reference_head_v5.py")
    identity = ComponentIdentity(
        name="scalemark",
        protocol=CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        artifact_sha256={"checkpoint": actual},
        source_sha256={"head_source": sha256_file(source)},
    )
    metadata = {
        "schema_version": payload.get("schema_version"),
        "protocol": payload.get("protocol"),
        "head_protocol": payload.get("head_protocol"),
        "status": payload.get("status"),
    }
    return FrozenModuleBinding(head, identity, metadata)


def load_frozen_pepd(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: str | torch.device,
) -> FrozenModuleBinding:
    """Load the authenticated frozen PEPD backbone for a concrete provider."""

    path = Path(checkpoint).resolve(strict=True)
    expected = _require_sha256(expected_sha256, name="expected PEPD SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("PEPD checkpoint SHA-256 mismatch")
    from experiments.train_cagh_scalemark_reference_probe import load_pepd

    model, loader_identity = load_pepd(path, torch.device(device))
    _freeze(model, device)
    source = Path(__file__).resolve().with_name("cagh_net.py")
    source_hashes = {"pepd_model_source": sha256_file(source)} if source.is_file() else {}
    identity = ComponentIdentity(
        name="pepd",
        protocol=str(loader_identity.get("protocol") or "frozen_pepd"),
        artifact_sha256={"checkpoint": actual},
        source_sha256=source_hashes,
    )
    metadata = {
        "checkpoint_sha256": actual,
        "loader_identity_sha256": canonical_sha256(_json_safe(loader_identity)),
    }
    return FrozenModuleBinding(model, identity, metadata)


class UnifiedCRRMAdapter:
    """Frozen, label-free S/P/M fusion on one unchanged canonical ROI."""

    def __init__(
        self,
        *,
        automatic_reference_provider: AutomaticReferenceProvider,
        pepd_provider: ProgressExpertProvider,
        scalemark_provider: ProgressExpertProvider,
        mask_geometry_provider: ProgressExpertProvider,
        gate: FrozenGateBinding,
        execution_mode: str = "formal_frozen",
    ) -> None:
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"unsupported execution mode: {execution_mode}")
        self.execution_mode = execution_mode
        self.reference_provider = automatic_reference_provider
        self.providers: dict[str, ProgressExpertProvider] = {
            "pepd": pepd_provider,
            "scalemark": scalemark_provider,
            "mask_geometry": mask_geometry_provider,
        }
        self.gate = gate
        identities = {
            "automatic_reference": automatic_reference_provider.identity,
            **{name: provider.identity for name, provider in self.providers.items()},
            "crrm_gate": gate.identity,
        }
        for expected_name, identity in identities.items():
            if identity.name != expected_name:
                raise ValueError(
                    f"provider identity mismatch: expected {expected_name}, got {identity.name}"
                )
            if execution_mode == "formal_frozen":
                if not identity.frozen or not identity.verified_complete or identity.synthetic:
                    raise ValueError(f"formal component is not complete/frozen: {expected_name}")
            elif not identity.synthetic:
                raise ValueError("synthetic smoke accepts synthetic identities only")
        self._identities = identities

    @property
    def component_bindings(self) -> dict[str, Any]:
        return {
            name: identity.as_record() for name, identity in self._identities.items()
        }

    def _reference(self, image: np.ndarray) -> AutomaticReferenceResult:
        try:
            result = self.reference_provider.predict(image)
            if not isinstance(result, AutomaticReferenceResult):
                raise TypeError("automatic reference provider returned wrong type")
            return result
        except Exception as exc:
            return AutomaticReferenceResult(
                available=False,
                start_xy=None,
                end_xy=None,
                telemetry={"exception_type": type(exc).__name__},
                failure_code="automatic_reference_provider_exception",
            )

    def _expert(
        self,
        name: str,
        image: np.ndarray,
        reference: AutomaticReferenceResult,
        dependencies: Mapping[str, ExpertResult],
    ) -> ExpertResult:
        try:
            result = self.providers[name].predict(
                image,
                automatic_reference=reference,
                dependencies=dependencies,
            )
            if not isinstance(result, ExpertResult):
                raise TypeError("expert provider returned wrong type")
            return result.normalized(name=name)
        except Exception as exc:
            return ExpertResult(
                progress=None,
                available=False,
                telemetry={"exception_type": type(exc).__name__},
                failure_code=f"{name}_provider_exception",
            )

    @staticmethod
    def _feature_vector(experts: Mapping[str, ExpertResult]) -> tuple[list[float], dict[str, Any], dict[str, bool]]:
        numeric: list[float] = []
        serialized: dict[str, Any] = {}
        available: dict[str, bool] = {}
        for feature_name in FEATURE_NAMES:
            owner = FEATURE_OWNER[feature_name]
            value = _finite_optional(experts[owner].telemetry.get(feature_name))
            numeric.append(float("nan") if value is None else value)
            serialized[feature_name] = value
            available[feature_name] = value is not None
        return numeric, serialized, available

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> dict[str, Any]:
        """Infer normalized progress without labels or physical scale metadata."""

        if input_is_canonical_meter_roi is not True:
            raise ValueError("V5 primary adapter requires input_is_canonical_meter_roi=True")
        image = _validate_image(canonical_meter_roi_bgr)
        input_hash = image_sha256(image)
        reference = self._reference(image)
        # PEPD is evaluated first because the ScaleMark provider may consume
        # its frozen feature context.  No serialized result contains that
        # in-memory context.
        pepd = self._expert("pepd", image, reference, {})
        scalemark = self._expert("scalemark", image, reference, {"pepd": pepd})
        mask = self._expert("mask_geometry", image, reference, {})
        experts = {"mask_geometry": mask, "pepd": pepd, "scalemark": scalemark}

        feature_values, feature_record, feature_availability = self._feature_vector(experts)
        progress_values = [
            float("nan") if experts[name].progress is None else float(experts[name].progress)
            for name in EXPERT_NAMES
        ]
        availability_values = [bool(experts[name].available) for name in EXPERT_NAMES]
        device = self.gate.device
        feature_tensor = torch.tensor([feature_values], dtype=torch.float32, device=device)
        progress_tensor = torch.tensor([progress_values], dtype=torch.float32, device=device)
        availability_tensor = torch.tensor([availability_values], dtype=torch.bool, device=device)
        pepd_anchor = torch.tensor(
            [float("nan") if pepd.progress is None else float(pepd.progress)],
            dtype=torch.float32,
            device=device,
        )
        pepd_anchor_available = torch.tensor([bool(pepd.available)], dtype=torch.bool, device=device)
        mask_fallback = torch.tensor(
            [float("nan") if mask.progress is None else float(mask.progress)],
            dtype=torch.float32,
            device=device,
        )
        mask_fallback_available = torch.tensor([bool(mask.available)], dtype=torch.bool, device=device)
        with torch.inference_mode():
            gate_output = self.gate.module(
                feature_tensor,
                progress_tensor,
                availability_tensor,
                production_anchor_progress=pepd_anchor,
                production_anchor_availability=pepd_anchor_available,
                mask_progress=mask_fallback,
                mask_availability=mask_fallback_available,
            )

        fused = _finite_optional(gate_output.fused_progress[0].detach().cpu().item())
        fused_available = bool(gate_output.fused_available[0].detach().cpu().item())
        if fused is None or not 0.0 <= fused <= 1.0:
            fused_available = False
            fused = None
        weights = gate_output.weights[0].detach().cpu().tolist()
        log_variances = gate_output.log_variances[0].detach().cpu().tolist()
        normalized_features = gate_output.normalized_features[0].detach().cpu().tolist()
        expert_records = {
            name: experts[name].as_record(name=name, input_image_sha256=input_hash)
            for name in EXPERT_NAMES
        }
        reference_record = reference.as_record(input_image_sha256=input_hash)
        record: dict[str, Any] = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "ok" if fused_available else "failed",
            "evidence_role": EVIDENCE_ROLE,
            "output_mode": OUTPUT_MODE,
            "failure": None if fused_available else {"code": "no_valid_fused_progress"},
            "prediction_space": PREDICTION_SPACE,
            "progress": fused,
            "expert_order": list(EXPERT_NAMES),
            "expert_progress": {name: expert_records[name]["progress"] for name in EXPERT_NAMES},
            "expert_availability": {
                name: bool(expert_records[name]["available"]) for name in EXPERT_NAMES
            },
            "expert_results": expert_records,
            "features": feature_record,
            "feature_availability": feature_availability,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "gate": {
                "protocol": GATE_PROTOCOL,
                "weights": {name: float(weights[index]) for index, name in enumerate(EXPERT_NAMES)},
                "log_variances": {
                    name: float(log_variances[index]) for index, name in enumerate(EXPERT_NAMES)
                },
                "normalized_features": {
                    name: float(normalized_features[index])
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "used_soft_fusion": bool(gate_output.used_soft_fusion[0].cpu().item()),
                "used_production_anchor": bool(
                    gate_output.used_production_anchor[0].cpu().item()
                ),
                "production_anchor_source": "pepd",
                "used_mask_fallback": bool(gate_output.used_mask_fallback[0].cpu().item()),
            },
            "automatic_reference": reference_record,
            "input_attestation": {
                "input_image_sha256": input_hash,
                "shape_hwc": list(image.shape),
                "dtype": str(image.dtype),
                "channel_order": "BGR",
                "input_is_canonical_meter_roi": True,
                "meter_bbox_xyxy": [0, 0, int(image.shape[1]), int(image.shape[0])],
                "meter_detector_invoked": False,
                "second_crop_applied": False,
                "correction_applied": False,
                "ground_truth_consumed": False,
                "physical_scale_consumed": False,
                "manual_scalemark_consumed": False,
                "manual_reference_consumed": False,
                "caller_reference_packet_consumed": False,
            },
            "component_bindings": self.component_bindings,
            "execution_mode": self.execution_mode,
            "synthetic_inputs": self.execution_mode == "synthetic_smoke",
            "eligible_for_metrics": self.execution_mode == "formal_frozen",
            "eligible_for_primary_metrics": False,
            "interpretation": (
                "progress component diagnostic only; automatic numeric scale "
                "start/end values are not predicted by this adapter"
            ),
            "output_schema_sha256": OUTPUT_SCHEMA_SHA256,
            "adapter_source_sha256": sha256_file(Path(__file__)),
        }
        record["record_sha256"] = canonical_sha256(record)
        return record


def _synthetic_identity(name: str, protocol: str) -> ComponentIdentity:
    return ComponentIdentity(
        name=name,
        protocol=protocol,
        artifact_sha256={"synthetic_stub": hashlib.sha256(f"{name}:stub".encode()).hexdigest()},
        source_sha256={},
        frozen=True,
        verified_complete=True,
        synthetic=True,
    )


class _SyntheticReference:
    identity = _synthetic_identity("automatic_reference", "synthetic_reference_smoke_v1")

    def predict(self, image: np.ndarray) -> AutomaticReferenceResult:
        height, width = image.shape[:2]
        return AutomaticReferenceResult(
            available=True,
            start_xy=(0.20 * width, 0.75 * height),
            end_xy=(0.80 * width, 0.75 * height),
            telemetry={"synthetic_stub": True},
        )


class _SyntheticExpert:
    def __init__(self, name: str, progress: float, values: Mapping[str, float]) -> None:
        self.identity = _synthetic_identity(name, f"synthetic_{name}_smoke_v1")
        self.progress = float(progress)
        self.values = dict(values)

    def predict(
        self,
        image: np.ndarray,
        *,
        automatic_reference: AutomaticReferenceResult,
        dependencies: Mapping[str, ExpertResult],
    ) -> ExpertResult:
        del image, automatic_reference, dependencies
        return ExpertResult(
            progress=self.progress,
            available=True,
            telemetry={**self.values, "synthetic_stub": True},
        )


def run_synthetic_smoke() -> dict[str, Any]:
    """Exercise tensor/schema wiring only; returns a non-metric synthetic record."""

    values = {name: 0.5 + index * 0.01 for index, name in enumerate(FEATURE_NAMES)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260806)
        gate_module = CRRMV5SoftReliabilityGate()
    _freeze(gate_module, "cpu")
    gate = FrozenGateBinding(
        gate_module,
        _synthetic_identity("crrm_gate", GATE_PROTOCOL),
    )
    adapter = UnifiedCRRMAdapter(
        automatic_reference_provider=_SyntheticReference(),
        pepd_provider=_SyntheticExpert(
            "pepd", 0.42, {name: values[name] for name in FEATURE_NAMES[5:11]}
        ),
        scalemark_provider=_SyntheticExpert(
            "scalemark", 0.45, {name: values[name] for name in FEATURE_NAMES[11:]}
        ),
        mask_geometry_provider=_SyntheticExpert(
            "mask_geometry", 0.40, {name: values[name] for name in FEATURE_NAMES[:5]}
        ),
        gate=gate,
        execution_mode="synthetic_smoke",
    )
    y_grid, x_grid = np.mgrid[0:64, 0:64]
    image = np.stack(
        (
            (x_grid * 3) % 256,
            (y_grid * 5) % 256,
            ((x_grid + y_grid) * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    record = adapter.predict(image, input_is_canonical_meter_roi=True)
    if record["eligible_for_metrics"] or not record["synthetic_inputs"]:
        raise AssertionError("synthetic smoke attestation failed")
    if record["status"] != "ok" or record["progress"] is None:
        raise AssertionError("synthetic smoke fusion failed")
    if set(record["features"]) != set(FEATURE_NAMES):
        raise AssertionError("synthetic smoke feature roster drift")
    return record


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="run only the in-memory non-metric synthetic wiring smoke",
    )
    args = parser.parse_args()
    if not args.synthetic_smoke:
        raise SystemExit(
            "Refusing implicit execution. Use --synthetic-smoke or construct a formal frozen adapter."
        )
    print(json.dumps(run_synthetic_smoke(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _main()


__all__ = [
    "AutomaticReferenceProvider",
    "AutomaticReferenceResult",
    "ComponentIdentity",
    "EXECUTION_MODES",
    "ExpertResult",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_SHA256",
    "FrozenGateBinding",
    "FrozenModuleBinding",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_SHA256",
    "PREDICTION_SPACE",
    "PROTOCOL",
    "ProgressExpertProvider",
    "UnifiedCRRMAdapter",
    "bind_frozen_provider",
    "canonical_sha256",
    "image_sha256",
    "load_frozen_crrm_gate",
    "load_frozen_pepd",
    "load_frozen_scalemark_v5_head",
    "run_synthetic_smoke",
    "sha256_file",
]
