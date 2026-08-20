"""Strict seed-20260720 VDN terminal provider for matched-ROI inference.

This module is intentionally a narrow adapter.  It accepts one already
materialized canonical meter ROI and owns the automatic start/end reference
detector internally.  It does not accept a source bounding box, sample ID,
labels, targets, physical scale values, or a caller-supplied reference packet.

The only eligible VDN artifact is the seed-20260720 epoch-200 ``last.pt``
checkpoint.  The adapter authenticates that container before deserialization
and loads only its ``current_model_state``.  The epoch-197 ``best.pt`` derived
artifact and its ``model_state`` envelope are explicitly ineligible.

This adapter measures candidate evidence but treats every in-process result as
``runtime_candidate_untrusted``.  It emits a candidate binding only as
recomputable telemetry for a later independent evaluation protocol to reload,
freeze, and pin.  Neither a caller-supplied digest nor a mutable runtime flag
can make this adapter emit a formal success.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import secrets
import stat
import struct
import sys
import tempfile
import types
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

# Capture this before importing any project module below.  Several shared V5
# modules legitimately import ``experiments.vdn_baseline`` as part of their own
# implementation.  A copy that was already present before this wrapper began
# importing is different: it could have supplied attacker-controlled helper
# callables while merely pointing ``__file__`` at the genuine source path.
_CRITICAL_RUNTIME_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "experiments.evaluate_vdn_baseline",
        "experiments.v5_shared_roi_comparison_input",
        "experiments.v5_unified_direction_adapters",
        "experiments.v5_unified_full_auto_adapter",
        "experiments.v5_unified_full_auto_progress_providers",
        "experiments.v5_unified_two_stage_retest",
        "experiments.vdn_baseline",
        "experiments.vdn_official200_protocol",
        "ultralytics",
        "utils.angleDetect.yoloDetection.pointGet",
        "utils.angleDetect.yoloDetection.yoloDectect",
    }
)
_PRELOADED_RUNTIME_MODULES_AT_IMPORT: Final[frozenset[str]] = frozenset(
    name for name in _CRITICAL_RUNTIME_MODULE_NAMES if name in sys.modules
)

from experiments.v5_shared_roi_comparison_input import canonical_roi_pixel_sha256
from experiments.v5_unified_direction_adapters import (
    VDNOfficial200LabelFreeAdapter,
    _method_record,
    direct_resize_whole_roi,
)
from experiments.v5_unified_full_auto_progress_providers import (
    FrozenProductionAutomaticReference,
    REFERENCE_PROTOCOL,
    VDNFullAutoProgressProvider,
)
from experiments.v5_unified_two_stage_retest import (
    assert_label_free,
    canonical_json_sha256,
)
PROTOCOL: Final[str] = "vdn_seed20_terminal_matched_candidate_v2"
SOURCE_BINDING_PROTOCOL: Final[str] = "vdn_seed20_terminal_candidate_binding_v2"
ADAPTER_RUNTIME_TELEMETRY_PROTOCOL: Final[str] = (
    "vdn_seed20_adapter_runtime_telemetry_v1"
)
CHECKPOINT_ROLE: Final[str] = "terminal_last"
MODEL_STATE_KEY: Final[str] = "current_model_state"
EXECUTION_RUNTIME_CANDIDATE_UNTRUSTED: Final[str] = "runtime_candidate_untrusted"
OFFICIAL200_PROTOCOL: Final[str] = "vdn_syncg_official_200_epoch_from_scratch_v1"
OFFICIAL200_CHECKPOINT_PROTOCOL: Final[str] = (
    "vdn_official200_atomic_authoritative_checkpoint_v1"
)
OFFICIAL200_VERIFICATION_PROTOCOL: Final[str] = (
    "formal_vdn_official200_verification_v1"
)
SEED: Final[int] = 20260720
TERMINAL_EPOCH: Final[int] = 200
BEST_EPOCH: Final[int] = 197
FIT_SAMPLES: Final[int] = 14375
HOLDOUT_SAMPLES: Final[int] = 1625
FIT_SAMPLE_IDS_SHA256: Final[str] = (
    "b38e477fdf8667bc012023731756aa4fa275fd6e931a87bc065fec9d8ef88979"
)
HOLDOUT_SAMPLE_IDS_SHA256: Final[str] = (
    "7550cf807f6669723c8a58cf80c7e1046af4aaea8bacfebc781dcb70d899fca2"
)
TERMINAL_CHECKPOINT_SHA256: Final[str] = (
    "cf346af87d979450c39057c2efa2e9377328610393430b23b5974a79fe863353"
)
REJECTED_BEST_CHECKPOINT_SHA256: Final[str] = (
    "678ef6b1c4bf42450ca927a9cfdf5adda7373af31f9379cfa0ec50e9b7810914"
)
VERIFICATION_SHA256: Final[str] = (
    "a1f651867c04116a30836f504996922d682655ffe31f38fa280985712bc117da"
)
VERIFICATION_PROTOCOL: Final[str] = OFFICIAL200_VERIFICATION_PROTOCOL
REFERENCE_DETECTOR_SHA256: Final[str] = (
    "2cb5c2523e364063ccdfd5c047390f17986ebcb622cef09d8604dfdaf038bdd6"
)
VDN_SOURCE_COMMIT: Final[str] = "68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb"
VDN_MODEL_SOURCE_SHA256: Final[str] = (
    "cce837d9719725c49744409152ab8a04e018179b60746430dd7f4c88d03b372b"
)
NATIVE_INPUT_SIZE: Final[int] = 384
HEATMAP_SIZE: Final[int] = 96

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_WRAPPER_SOURCE: Final[Path] = Path(__file__).absolute()
DEFAULT_CHECKPOINT: Final[Path] = (
    _PROJECT_ROOT
    / "artifacts/runs/vdn_syncg_official200/seed_20260720/last.pt"
)
DEFAULT_VERIFICATION: Final[Path] = DEFAULT_CHECKPOINT.with_name(
    "verification_v1.json"
)
DEFAULT_VDN_SOURCE: Final[Path] = (
    _PROJECT_ROOT / "artifacts/vendor/VectorDetectionNetwork"
)
DEFAULT_REFERENCE_DETECTOR: Final[Path] = (
    _PROJECT_ROOT / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt"
)
_DIRECTION_ADAPTER_SOURCE: Final[Path] = (
    _PROJECT_ROOT / "experiments/v5_unified_direction_adapters.py"
)
_FULL_AUTO_PROVIDER_SOURCE: Final[Path] = (
    _PROJECT_ROOT / "experiments/v5_unified_full_auto_progress_providers.py"
)
_VDN_RUNTIME_SOURCE: Final[Path] = _PROJECT_ROOT / "experiments/vdn_baseline.py"
_BEHAVIOR_SOURCE_PATHS: Final[dict[str, Path]] = {
    "shared_roi": _PROJECT_ROOT / "experiments/v5_shared_roi_comparison_input.py",
    "direction_adapter": _DIRECTION_ADAPTER_SOURCE,
    "full_auto_provider": _FULL_AUTO_PROVIDER_SOURCE,
    "full_auto_hash_helper": _PROJECT_ROOT / "experiments/v5_unified_full_auto_adapter.py",
    "label_free_contract": _PROJECT_ROOT / "experiments/v5_unified_two_stage_retest.py",
    "vdn_runtime": _VDN_RUNTIME_SOURCE,
    "vdn_official200_protocol": _PROJECT_ROOT / "experiments/vdn_official200_protocol.py",
    "reference_loader": _PROJECT_ROOT / "experiments/evaluate_vdn_baseline.py",
    "reference_detector_runtime": _PROJECT_ROOT / "utils/angleDetect/yoloDetection/yoloDectect.py",
    "reference_geometry_runtime": _PROJECT_ROOT / "utils/angleDetect/yoloDetection/pointGet.py",
}
_EXPECTED_BEHAVIOR_SOURCE_SHA256: Final[dict[str, str]] = {
    "shared_roi": "4efb025b984d7f592a3fa77f4812f96ed699000a7a1e02b2ee7ce88e76714210",
    "direction_adapter": "d03b7ca06e97a83c004dc72c8c98d830ffdcaa8c7fa1d8ec359e81407b11553f",
    "full_auto_provider": "018a73c6d7d7bfa26cf039c2135670b3ba9c61d261e26985de0e030fe609517e",
    "full_auto_hash_helper": "f50d1c1faad6841850d6c92e6ea4381d9cd93f1199818ad435d40797eb5d310d",
    "label_free_contract": "de099e52a5bc698e954406f03aa2b347ddc913cbbf5bd9b3458a37753083b4d7",
    "vdn_runtime": "fa7eb000ca10b049d2d2ad492b100fc83d2e2e1ac823d41d7a6c2315474e1344",
    "vdn_official200_protocol": "a6c11582b82403db97a41a7bfda843b991c2e5a381f79b6b88f38d3cac209303",
    "reference_loader": "d9bb604434fed9c8488dc5b6dbbb484a0aee689be30857c5edee66967b65bccf",
    "reference_detector_runtime": (
        "e3d1ec2a679bbcd0b96b4a984ef7daa2204e31622616aa0fb1924343f8550f69"
    ),
    "reference_geometry_runtime": (
        "f65f5c8aa5d0e42a501e13fc5b7b71859fcbbab162db8a38505739144f56f4f9"
    ),
}
_SOURCE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "binding_origin",
        "paper_eligibility_authority",
        "formal_success_eligible",
        "adapter_wrapper_sha256",
        "dependency_source_sha256",
        "vdn_source_commit",
        "vdn_model_source_sha256",
        "vdn_runtime_source_sha256",
        "checkpoint_container_sha256",
        "checkpoint_verification_sha256",
        "checkpoint_state_key",
        "checkpoint_state_sha256",
        "reference_detector_sha256",
        "reference_detector_state_sha256",
        "payload_sha256",
    }
)
_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {"field", "sealed", "confirmatory", "test", "tests"}
)
_LEGACY_VDN_MODULE_NAME: Final[str] = "external_vdn_model_68afe1e"
_AUTHENTICATED_VDN_MODULE_PREFIX: Final[str] = (
    "external_vdn_model_68afe1e_authenticated_"
)
_AUTHENTICATED_VDN_RUNTIME_MODULE_PREFIX: Final[str] = (
    "experiments._vdn_baseline_authenticated_"
)
_AUTHENTICATED_VDN_RUNTIME_CALLABLES: Final[dict[str, str]] = {
    "verify_vdn_source": "verify_vdn_source",
    "vdn_config": "vdn_config",
    "initialize_vdn_heads": "_initialize_vdn_heads",
    "normalized_bgr_tensor": "normalized_bgr_tensor",
    "predict_directions": "predict_directions",
    "image_angle_from_direction": "image_angle_from_direction",
    "reading_from_pointer_angle": "reading_from_pointer_angle",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _restricted_path_parts(path: Path) -> list[str]:
    normalized_parts = {part.casefold() for part in path.parts}
    return sorted(normalized_parts & _FORBIDDEN_PATH_PARTS)


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _reject_unsafe_path_components(path: Path, *, label: str) -> None:
    blocked = _restricted_path_parts(path)
    _require(not blocked, f"{label} path enters a restricted namespace: {blocked}")
    anchor = Path(path.anchor) if path.anchor else Path()
    current = anchor
    parts = path.parts[1:] if path.anchor else path.parts
    reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    for part in parts:
        current = current / part
        metadata = os.lstat(current)
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        _require(
            not stat.S_ISLNK(metadata.st_mode) and not (attributes & reparse_mask),
            f"{label} path contains a symlink or reparse point: {current}",
        )


def _guard_public_component_path(path: Path, *, label: str) -> Path:
    lexical = _absolute_lexical_path(path)
    _reject_unsafe_path_components(lexical, label=label)
    resolved = lexical.resolve(strict=True)
    _reject_unsafe_path_components(resolved, label=label)
    _require(resolved.is_file(), f"{label} is not a file")
    return resolved


def _guard_public_directory(path: Path, *, label: str) -> Path:
    lexical = _absolute_lexical_path(path)
    _reject_unsafe_path_components(lexical, label=label)
    resolved = lexical.resolve(strict=True)
    _reject_unsafe_path_components(resolved, label=label)
    _require(resolved.is_dir(), f"{label} is not a directory")
    return resolved


def _read_bytes_once(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read {label} exactly once: {path}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_digest(value: Any, *, label: str) -> str:
    raw = str(value or "").strip()
    digest = raw.casefold()
    _require(
        raw == digest
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} is not a lowercase SHA256 digest",
    )
    return digest


def _digest_part(hasher: Any, tag: str, payload: bytes = b"") -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(struct.pack(">Q", len(tag_bytes)))
    hasher.update(tag_bytes)
    hasher.update(struct.pack(">Q", len(payload)))
    hasher.update(payload)


def _tensor_mapping_sha256(value: Mapping[str, Any], *, label: str) -> str:
    """Hash a state mapping without pickle/storage-identity side effects."""

    _require(isinstance(value, Mapping) and bool(value), f"{label} is empty")
    hasher = hashlib.sha256()
    keys = sorted(value)
    _require(
        all(isinstance(key, str) for key in keys),
        f"{label} contains a non-string key",
    )
    _digest_part(hasher, "state_mapping_v1")
    _digest_part(hasher, "mapping_length", str(len(keys)).encode("ascii"))
    for key in keys:
        tensor = value[key]
        _require(torch.is_tensor(tensor), f"{label}.{key} is not a tensor")
        _require(
            tensor.layout == torch.strided,
            f"{label}.{key} has unsupported layout {tensor.layout}",
        )
        contiguous = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "layout": str(contiguous.layout),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _digest_part(hasher, "key", key.encode("utf-8"))
        _digest_part(hasher, "tensor_metadata", metadata)
        _digest_part(
            hasher,
            "tensor_bytes",
            contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    return hasher.hexdigest()


def model_state_sha256(state: Mapping[str, Any]) -> str:
    """Local stable state digest; never imported through a module cache."""

    return _tensor_mapping_sha256(state, label="VDN model state")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON data") from exc


def _strict_json_from_bytes(payload: bytes, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        with io.TextIOWrapper(
            io.BytesIO(payload), encoding="utf-8-sig", errors="strict"
        ) as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_constant,
            )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def _measure_behavior_sources() -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in _BEHAVIOR_SOURCE_PATHS.items():
        source = _guard_public_component_path(path, label=f"runtime source {name}")
        actual[name] = _sha256_bytes(
            _read_bytes_once(source, label=f"runtime source {name}")
        )
    _require(
        actual == _EXPECTED_BEHAVIOR_SOURCE_SHA256,
        "VDN matched runtime source drift",
    )
    return actual


@dataclass(frozen=True)
class _ReferenceRuntimeReceipt:
    detector_weights_sha256: str
    detector_state_sha256: str
    detector_type_module: str
    detector_type_name: str
    state_owner_type_module: str
    state_owner_type_name: str

    def as_identity(self) -> dict[str, str]:
        return asdict(self)


def _build_candidate_source_binding(
    *,
    terminal_binding: "Seed20TerminalBinding",
    reference_receipt: _ReferenceRuntimeReceipt,
) -> dict[str, Any]:
    """Emit evidence for a later, independently frozen evaluation protocol.

    This runtime-generated binding is deliberately *not* an eligibility
    authority.  Only a later protocol with an independently pinned digest may
    promote results derived from it.
    """

    wrapper = _guard_public_component_path(
        _WRAPPER_SOURCE, label="terminal adapter wrapper source"
    )
    dependencies = _measure_behavior_sources()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol": SOURCE_BINDING_PROTOCOL,
        "binding_origin": "runtime_candidate_untrusted",
        "paper_eligibility_authority": "external_frozen_evaluation_protocol_only",
        "formal_success_eligible": False,
        "adapter_wrapper_sha256": _sha256_bytes(
            _read_bytes_once(wrapper, label="terminal adapter wrapper source")
        ),
        "dependency_source_sha256": dependencies,
        "vdn_source_commit": VDN_SOURCE_COMMIT,
        "vdn_model_source_sha256": VDN_MODEL_SOURCE_SHA256,
        "vdn_runtime_source_sha256": dependencies["vdn_runtime"],
        "checkpoint_container_sha256": terminal_binding.checkpoint_sha256,
        "checkpoint_verification_sha256": terminal_binding.verification_sha256,
        "checkpoint_state_key": terminal_binding.model_state_key,
        "checkpoint_state_sha256": terminal_binding.model_state_sha256,
        "reference_detector_sha256": reference_receipt.detector_weights_sha256,
        "reference_detector_state_sha256": reference_receipt.detector_state_sha256,
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_candidate_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "candidate source binding is absent")
    binding = copy.deepcopy(dict(value))
    _require(
        set(binding) == _SOURCE_BINDING_KEYS,
        "candidate source binding schema drift",
    )
    declared = _sha256_digest(
        binding.pop("payload_sha256", ""),
        label="candidate binding payload SHA256",
    )
    _require(
        canonical_json_sha256(binding) == declared,
        "candidate binding payload SHA256 mismatch",
    )
    binding["payload_sha256"] = declared
    expected_scalars = {
        "schema_version": 2,
        "protocol": SOURCE_BINDING_PROTOCOL,
        "binding_origin": "runtime_candidate_untrusted",
        "paper_eligibility_authority": "external_frozen_evaluation_protocol_only",
        "formal_success_eligible": False,
        "vdn_source_commit": VDN_SOURCE_COMMIT,
        "vdn_model_source_sha256": VDN_MODEL_SOURCE_SHA256,
        "checkpoint_container_sha256": TERMINAL_CHECKPOINT_SHA256,
        "checkpoint_verification_sha256": VERIFICATION_SHA256,
        "checkpoint_state_key": MODEL_STATE_KEY,
        "reference_detector_sha256": REFERENCE_DETECTOR_SHA256,
    }
    for field, expected in expected_scalars.items():
        _require(binding.get(field) == expected, f"candidate binding drift: {field}")
    for field in ("checkpoint_state_sha256", "reference_detector_state_sha256"):
        binding[field] = _sha256_digest(
            binding.get(field), label=f"candidate binding {field}"
        )
    dependencies = binding.get("dependency_source_sha256")
    _require(
        dependencies == _EXPECTED_BEHAVIOR_SOURCE_SHA256,
        "candidate binding dependency inventory drift",
    )
    _require(
        binding.get("vdn_runtime_source_sha256") == dependencies["vdn_runtime"],
        "candidate binding VDN runtime source drift",
    )
    _require(
        _measure_behavior_sources() == dependencies,
        "candidate binding dependency bytes drift",
    )
    wrapper = _guard_public_component_path(
        _WRAPPER_SOURCE, label="terminal adapter wrapper source"
    )
    _require(
        _sha256_bytes(_read_bytes_once(wrapper, label="terminal adapter wrapper source"))
        == binding.get("adapter_wrapper_sha256"),
        "candidate binding wrapper bytes drift",
    )
    return binding


@dataclass(frozen=True)
class Seed20TerminalBinding:
    checkpoint_sha256: str
    verification_sha256: str
    verification_protocol: str
    model_state_sha256: str
    checkpoint_role: str = CHECKPOINT_ROLE
    model_state_key: str = MODEL_STATE_KEY
    seed: int = SEED
    terminal_epoch: int = TERMINAL_EPOCH
    fit_samples: int = FIT_SAMPLES
    holdout_samples: int = HOLDOUT_SAMPLES
    fit_sample_ids_sha256: str = FIT_SAMPLE_IDS_SHA256
    holdout_sample_ids_sha256: str = HOLDOUT_SAMPLE_IDS_SHA256

    def as_identity(self) -> dict[str, Any]:
        return asdict(self)


def _validate_terminal_binding(
    binding: Seed20TerminalBinding,
    *,
    expected_model_state_sha256: str | None = None,
) -> None:
    _require(isinstance(binding, Seed20TerminalBinding), "seed20 binding type drift")
    expected = {
        "checkpoint_sha256": TERMINAL_CHECKPOINT_SHA256,
        "verification_sha256": VERIFICATION_SHA256,
        "verification_protocol": VERIFICATION_PROTOCOL,
        "checkpoint_role": CHECKPOINT_ROLE,
        "model_state_key": MODEL_STATE_KEY,
        "seed": SEED,
        "terminal_epoch": TERMINAL_EPOCH,
        "fit_samples": FIT_SAMPLES,
        "holdout_samples": HOLDOUT_SAMPLES,
        "fit_sample_ids_sha256": FIT_SAMPLE_IDS_SHA256,
        "holdout_sample_ids_sha256": HOLDOUT_SAMPLE_IDS_SHA256,
    }
    identity = binding.as_identity()
    for field, value in expected.items():
        _require(identity.get(field) == value, f"seed20 terminal binding drift: {field}")
    state_sha256 = _sha256_digest(
        binding.model_state_sha256,
        label="seed20 terminal model state SHA256",
    )
    if expected_model_state_sha256 is not None:
        _require(
            state_sha256
            == _sha256_digest(
                expected_model_state_sha256,
                label="expected terminal model state SHA256",
            ),
            "seed20 terminal binding drift: model_state_sha256",
        )


def _validate_verification_metadata(
    verification: Mapping[str, Any],
    *,
    actual_checkpoint_sha256: str,
) -> None:
    _require(verification.get("schema_version") == 1, "verification schema drift")
    _require(
        verification.get("protocol") == VERIFICATION_PROTOCOL,
        "verification protocol drift",
    )
    _require(verification.get("verified") is True, "checkpoint is not verified")
    _require(
        verification.get("training_artifacts_verified") is True,
        "training artifacts are not verified",
    )
    _require(
        verification.get("eligible_for_three_seed_cohort") is True,
        "VDN run is not cohort-eligible",
    )
    _require(verification.get("seed") == SEED, "verification seed drift")
    _require(
        verification.get("epochs") == TERMINAL_EPOCH,
        "verification terminal epoch drift",
    )
    _require(
        verification.get("last_checkpoint_sha256")
        == actual_checkpoint_sha256,
        "verification.last_checkpoint_sha256 does not bind the terminal checkpoint",
    )
    _require(
        verification.get("best_checkpoint_sha256")
        == REJECTED_BEST_CHECKPOINT_SHA256,
        "historical best checkpoint identity drift",
    )
    _require(
        verification.get("best_checkpoint_sha256")
        != actual_checkpoint_sha256,
        "best checkpoint is ineligible for the terminal role",
    )
    _require(verification.get("best_epoch") == BEST_EPOCH, "best epoch audit drift")
    _require(
        verification.get("official_stopping_boundary_reached") is True,
        "official epoch-200 stopping boundary was not reached",
    )
    for key in (
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
    ):
        _require(verification.get(key) is False, f"verification scope drift: {key}")


def _validate_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> None:
    _require(checkpoint.get("schema_version") == 1, "checkpoint schema drift")
    _require(
        checkpoint.get("checkpoint_protocol") == OFFICIAL200_CHECKPOINT_PROTOCOL,
        "official200 checkpoint envelope drift",
    )
    _require(checkpoint.get("status") == "complete", "checkpoint is not complete")
    _require(
        checkpoint.get("epoch") == TERMINAL_EPOCH,
        "checkpoint is not terminal epoch 200",
    )
    _require(checkpoint.get("best_epoch") == BEST_EPOCH, "best epoch audit drift")
    _require(
        "model_state" not in checkpoint,
        "derived best checkpoint envelope is ineligible",
    )
    signature = checkpoint.get("signature")
    _require(isinstance(signature, Mapping), "checkpoint signature is absent")
    expected_signature = {
        "schema_version": 1,
        "protocol": OFFICIAL200_PROTOCOL,
        "seed": SEED,
        "epochs": TERMINAL_EPOCH,
        "train_samples": FIT_SAMPLES,
        "validation_samples": HOLDOUT_SAMPLES,
        "train_sample_ids_sha256": FIT_SAMPLE_IDS_SHA256,
        "validation_sample_ids_sha256": HOLDOUT_SAMPLE_IDS_SHA256,
        "image_size": NATIVE_INPUT_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "vdn_source_commit": VDN_SOURCE_COMMIT,
    }
    for key, expected in expected_signature.items():
        _require(signature.get(key) == expected, f"checkpoint signature drift: {key}")
    current_model_state = checkpoint.get(MODEL_STATE_KEY)
    _require(
        isinstance(current_model_state, Mapping) and bool(current_model_state),
        "terminal current_model_state is absent",
    )
    best_model_state = checkpoint.get("best_model_state")
    _require(
        isinstance(best_model_state, Mapping) and bool(best_model_state),
        "best-model audit state is absent",
    )


def _deserialize_verified_terminal_bytes(
    checkpoint_bytes: bytes,
    verification_bytes: bytes,
) -> tuple[Seed20TerminalBinding, Mapping[str, Any]]:
    actual_checkpoint_sha256 = _sha256_bytes(checkpoint_bytes)
    _require(
        actual_checkpoint_sha256 != REJECTED_BEST_CHECKPOINT_SHA256,
        "historical best checkpoint is explicitly forbidden",
    )
    _require(
        actual_checkpoint_sha256 == TERMINAL_CHECKPOINT_SHA256,
        "seed20 terminal checkpoint SHA256 mismatch",
    )
    actual_verification_sha256 = _sha256_bytes(verification_bytes)
    _require(
        actual_verification_sha256 == VERIFICATION_SHA256,
        "seed20 verification SHA256 mismatch",
    )
    verification = _strict_json_from_bytes(
        verification_bytes, label="VDN terminal verification"
    )
    _require(isinstance(verification, Mapping), "verification is not an object")
    _validate_verification_metadata(
        verification,
        actual_checkpoint_sha256=actual_checkpoint_sha256,
    )

    # Deserialization is allowed only after both exact local digests are bound.
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=False,
    )
    _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
    _validate_checkpoint_metadata(checkpoint)
    state_sha256 = model_state_sha256(checkpoint[MODEL_STATE_KEY])
    binding = Seed20TerminalBinding(
        checkpoint_sha256=actual_checkpoint_sha256,
        verification_sha256=actual_verification_sha256,
        verification_protocol=str(verification["protocol"]),
        model_state_sha256=state_sha256,
    )
    _validate_terminal_binding(
        binding,
        expected_model_state_sha256=state_sha256,
    )
    return binding, checkpoint


def _read_and_verify_seed20_terminal_artifacts(
    *,
    checkpoint_path: Path,
    verification_path: Path,
) -> tuple[
    Seed20TerminalBinding,
    Mapping[str, Any],
    bytes,
    bytes,
]:
    checkpoint_file = _guard_public_component_path(
        checkpoint_path, label="VDN terminal checkpoint"
    )
    verification_file = _guard_public_component_path(
        verification_path, label="VDN verification"
    )
    _require(
        checkpoint_file.name.casefold() != "best.pt",
        "historical best checkpoint is explicitly forbidden",
    )
    _require(
        checkpoint_file.name.casefold() == "last.pt",
        "eligible checkpoint must be named last.pt",
    )
    _require(
        verification_file.name.casefold() == "verification_v1.json",
        "eligible verification must be named verification_v1.json",
    )
    _require(
        verification_file.parent == checkpoint_file.parent,
        "verification must be adjacent to the terminal checkpoint",
    )
    checkpoint_bytes = _read_bytes_once(
        checkpoint_file, label="VDN terminal checkpoint"
    )
    verification_bytes = _read_bytes_once(
        verification_file, label="VDN terminal verification"
    )
    binding, checkpoint = _deserialize_verified_terminal_bytes(
        checkpoint_bytes,
        verification_bytes,
    )
    return binding, checkpoint, checkpoint_bytes, verification_bytes


def verify_seed20_terminal_artifacts(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    verification_path: Path = DEFAULT_VERIFICATION,
) -> tuple[Seed20TerminalBinding, Mapping[str, Any]]:
    """Authenticate the exact seed20 terminal artifact before deserialization."""

    binding, checkpoint, _, _ = _read_and_verify_seed20_terminal_artifacts(
        checkpoint_path=checkpoint_path,
        verification_path=verification_path,
    )
    return binding, checkpoint


@dataclass(frozen=True)
class _AuthenticatedTerminalArtifacts:
    checkpoint_bytes: bytes
    verification_bytes: bytes
    binding: Seed20TerminalBinding


def _validate_authenticated_terminal_artifacts(
    artifacts: _AuthenticatedTerminalArtifacts,
) -> tuple[Seed20TerminalBinding, Mapping[str, Any]]:
    _require(
        isinstance(artifacts, _AuthenticatedTerminalArtifacts),
        "authenticated terminal artifact envelope is absent",
    )
    _require(
        isinstance(artifacts.checkpoint_bytes, bytes)
        and isinstance(artifacts.verification_bytes, bytes),
        "authenticated terminal artifact envelope does not contain bytes",
    )
    binding, checkpoint = _deserialize_verified_terminal_bytes(
        artifacts.checkpoint_bytes,
        artifacts.verification_bytes,
    )
    _require(
        artifacts.binding == binding,
        "authenticated terminal artifact binding differs from raw bytes",
    )
    return binding, checkpoint


def _authenticate_terminal_artifacts(
    *,
    checkpoint_path: Path,
    verification_path: Path,
) -> _AuthenticatedTerminalArtifacts:
    binding, _, checkpoint_bytes, verification_bytes = (
        _read_and_verify_seed20_terminal_artifacts(
            checkpoint_path=checkpoint_path,
            verification_path=verification_path,
        )
    )
    return _AuthenticatedTerminalArtifacts(
        checkpoint_bytes=checkpoint_bytes,
        verification_bytes=verification_bytes,
        binding=binding,
    )


@dataclass(frozen=True)
class _AuthenticatedVDNRuntime:
    module_name: str
    source_sha256: str
    module: types.ModuleType
    verify_vdn_source: Any
    vdn_config: Any
    initialize_vdn_heads: Any
    normalized_bgr_tensor: Any
    predict_directions: Any
    image_angle_from_direction: Any
    reading_from_pointer_angle: Any


@dataclass(frozen=True)
class _AdapterRuntimeTelemetryReceipt:
    """Recomputable in-process telemetry; never an eligibility capability."""

    schema_version: int
    protocol: str
    checkpoint_container_sha256: str
    checkpoint_verification_sha256: str
    checkpoint_verification_protocol: str
    checkpoint_state_key: str
    checkpoint_state_sha256: str
    runtime_module_name: str
    runtime_source_sha256: str
    model_module_name: str
    model_source_sha256: str
    payload_sha256: str

    def as_identity(self) -> dict[str, Any]:
        return asdict(self)


def _validate_authenticated_vdn_runtime(
    runtime: Any,
) -> _AuthenticatedVDNRuntime:
    _require(
        type(runtime) is _AuthenticatedVDNRuntime,
        "authenticated VDN runtime receipt is absent",
    )
    expected_source = _EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"]
    _require(
        runtime.source_sha256 == expected_source,
        "authenticated VDN runtime source receipt drift",
    )
    _require(
        runtime.module_name.startswith(_AUTHENTICATED_VDN_RUNTIME_MODULE_PREFIX),
        "authenticated VDN runtime module name drift",
    )
    _require(
        isinstance(runtime.module, types.ModuleType)
        and sys.modules.get(runtime.module_name) is runtime.module,
        "authenticated VDN runtime module identity drift",
    )
    _require(
        runtime.module.__dict__.get("__authenticated_source_sha256__")
        == expected_source,
        "authenticated VDN runtime module source digest drift",
    )
    for receipt_name, source_name in _AUTHENTICATED_VDN_RUNTIME_CALLABLES.items():
        function = getattr(runtime, receipt_name, None)
        _require(
            callable(function),
            f"authenticated VDN runtime lacks {receipt_name}",
        )
        _require(
            getattr(function, "__module__", None) == runtime.module_name,
            f"authenticated VDN runtime callable provenance drift: {receipt_name}",
        )
        _require(
            getattr(runtime.module, source_name, None) is function,
            f"authenticated VDN runtime callable identity drift: {receipt_name}",
        )
    return runtime


def _recompute_adapter_runtime_telemetry(
    adapter: Any,
    *,
    terminal_binding: Seed20TerminalBinding,
) -> _AdapterRuntimeTelemetryReceipt:
    """Measure current adapter state without granting any in-process authority."""

    _validate_terminal_binding(terminal_binding)
    _require(
        getattr(adapter, "terminal_binding", None) == terminal_binding,
        "adapter runtime telemetry binding drift",
    )
    runtime = _validate_authenticated_vdn_runtime(
        getattr(adapter, "_authenticated_vdn_runtime", None)
    )
    model = getattr(adapter, "model", None)
    _require(
        isinstance(model, torch.nn.Module),
        "adapter runtime telemetry model is absent",
    )
    loaded_state_sha256 = model_state_sha256(model.state_dict())
    _require(
        loaded_state_sha256 == terminal_binding.model_state_sha256,
        "adapter runtime telemetry model state drift",
    )
    _require(
        getattr(adapter, "checkpoint_sha256", None) == terminal_binding.checkpoint_sha256,
        "adapter runtime telemetry checkpoint drift",
    )
    _require(
        getattr(adapter, "authenticated_loaded_model_state_sha256", None)
        == loaded_state_sha256,
        "adapter runtime telemetry loaded-state receipt drift",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ADAPTER_RUNTIME_TELEMETRY_PROTOCOL,
        "checkpoint_container_sha256": terminal_binding.checkpoint_sha256,
        "checkpoint_verification_sha256": terminal_binding.verification_sha256,
        "checkpoint_verification_protocol": terminal_binding.verification_protocol,
        "checkpoint_state_key": terminal_binding.model_state_key,
        "checkpoint_state_sha256": loaded_state_sha256,
        "runtime_module_name": runtime.module_name,
        "runtime_source_sha256": runtime.source_sha256,
        "model_module_name": type(model).__module__,
        "model_source_sha256": str(
            getattr(model, "authenticated_vdn_source_sha256", "")
        ),
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    return _AdapterRuntimeTelemetryReceipt(**payload)


def _validate_adapter_runtime_telemetry(
    adapter: Any,
    *,
    terminal_binding: Seed20TerminalBinding,
) -> _AdapterRuntimeTelemetryReceipt:
    receipt = getattr(adapter, "_runtime_telemetry_receipt", None)
    _require(
        type(receipt) is _AdapterRuntimeTelemetryReceipt,
        "adapter runtime telemetry receipt is absent",
    )
    recomputed = _recompute_adapter_runtime_telemetry(
        adapter,
        terminal_binding=terminal_binding,
    )
    _require(
        receipt == recomputed,
        "adapter runtime telemetry receipt drift",
    )
    return recomputed


def _load_authenticated_vdn_runtime_snapshot() -> _AuthenticatedVDNRuntime:
    """Compile VDN runtime helpers from authenticated bytes, never import cache."""

    _require(
        not _PRELOADED_RUNTIME_MODULES_AT_IMPORT,
        "critical VDN runtime module was preloaded before adapter import: "
        f"{sorted(_PRELOADED_RUNTIME_MODULES_AT_IMPORT)}",
    )
    source = _guard_public_component_path(
        _VDN_RUNTIME_SOURCE, label="authenticated VDN runtime source"
    )
    source_bytes = _read_bytes_once(source, label="authenticated VDN runtime source")
    _require(
        _sha256_bytes(source_bytes) == _EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"],
        "authenticated VDN runtime source SHA256 drift",
    )
    try:
        source_text = source_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("authenticated VDN runtime source is not UTF-8") from exc
    code = compile(
        source_text,
        f"<authenticated-vdn-runtime:{_EXPECTED_BEHAVIOR_SOURCE_SHA256['vdn_runtime']}>",
        "exec",
        dont_inherit=True,
    )
    module_name = _AUTHENTICATED_VDN_RUNTIME_MODULE_PREFIX + secrets.token_hex(16)
    _require(
        module_name not in sys.modules,
        "authenticated VDN runtime module cache collision",
    )
    module = types.ModuleType(module_name)
    # The authenticated runtime derives PROJECT_DIR from __file__.  Point it at
    # the already-guarded source path while executing the separately compiled
    # byte snapshot; no path-based importer is used.
    module.__file__ = str(source)
    module.__dict__["__authenticated_source_sha256__"] = (
        _EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"]
    )
    module.__package__ = "experiments"
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
        _require(
            sys.modules.get(module_name) is module,
            "authenticated VDN runtime module cache identity changed during load",
        )
        callables = {
            receipt_name: getattr(module, source_name, None)
            for receipt_name, source_name in (
                _AUTHENTICATED_VDN_RUNTIME_CALLABLES.items()
            )
        }
        for label, function in callables.items():
            _require(callable(function), f"authenticated VDN runtime lacks {label}")
            _require(
                getattr(function, "__module__", None) == module_name,
                f"authenticated VDN runtime callable provenance drift: {label}",
            )
        runtime = _AuthenticatedVDNRuntime(
            module_name=module_name,
            source_sha256=_EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"],
            module=module,
            verify_vdn_source=callables["verify_vdn_source"],
            vdn_config=callables["vdn_config"],
            initialize_vdn_heads=callables["initialize_vdn_heads"],
            normalized_bgr_tensor=callables["normalized_bgr_tensor"],
            predict_directions=callables["predict_directions"],
            image_angle_from_direction=callables["image_angle_from_direction"],
            reading_from_pointer_angle=callables["reading_from_pointer_angle"],
        )
        return _validate_authenticated_vdn_runtime(runtime)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise


def _load_vdn_model_from_authenticated_source_snapshot(
    vdn_source: Path,
    *,
    image_size: int,
    runtime: _AuthenticatedVDNRuntime,
) -> torch.nn.Module:
    """Compile the authenticated VDN model bytes under a fresh module name."""

    runtime = _validate_authenticated_vdn_runtime(runtime)
    _require(
        _LEGACY_VDN_MODULE_NAME not in sys.modules,
        "legacy VDN sys.modules cache is populated; refusing cache injection",
    )
    source = _guard_public_directory(vdn_source, label="VDN source checkout")
    _require(
        runtime.verify_vdn_source(source) == VDN_SOURCE_COMMIT,
        "VDN source commit differs from terminal checkpoint",
    )
    model_path = _guard_public_component_path(
        source / "libs/models/vdn_model.py",
        label="authenticated VDN model source",
    )
    source_bytes = _read_bytes_once(
        model_path,
        label="authenticated VDN model source",
    )
    _require(
        _sha256_bytes(source_bytes) == VDN_MODEL_SOURCE_SHA256,
        "authenticated VDN model source SHA256 drift",
    )
    try:
        source_text = source_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("authenticated VDN model source is not UTF-8") from exc
    code = compile(
        source_text,
        f"<authenticated-vdn:{VDN_MODEL_SOURCE_SHA256}>",
        "exec",
        dont_inherit=True,
    )
    module_name = _AUTHENTICATED_VDN_MODULE_PREFIX + secrets.token_hex(16)
    _require(
        module_name not in sys.modules,
        "fresh authenticated VDN module name collision",
    )
    module = types.ModuleType(module_name)
    module.__file__ = f"<authenticated-vdn:{VDN_MODEL_SOURCE_SHA256}>"
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
        _require(
            sys.modules.get(module_name) is module,
            "authenticated VDN module cache identity changed during load",
        )
        factory = getattr(module, "get_vdn_resnet", None)
        _require(callable(factory), "authenticated VDN source has no model factory")
        model = factory(runtime.vdn_config(image_size=image_size), is_train=False)
        runtime.initialize_vdn_heads(model)
        _require(
            type(model).__module__ == module_name
            and type(model).__name__ == "VDNModel",
            "authenticated VDN model concrete type drift",
        )
        setattr(model, "authenticated_vdn_module_name", module_name)
        setattr(
            model,
            "authenticated_vdn_source_sha256",
            VDN_MODEL_SOURCE_SHA256,
        )
        setattr(model, "authenticated_vdn_source_snapshot_verified", True)
        return model
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise


def _snapshot_file_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = os.lstat(path)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse_mask = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not (attributes & reparse_mask),
        "authenticated reference snapshot is not a regular private file",
    )
    _require(
        int(getattr(metadata, "st_nlink", 1)) == 1,
        "authenticated reference snapshot has unexpected hard links",
    )
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


@contextmanager
def _windows_no_write_delete_path_lock(path: Path):
    """Hold a Windows sharing lock that permits reads but no write/delete open."""

    if os.name != "nt":
        raise OSError(
            "reference path loading is fail-closed outside Windows because an "
            "equivalent no-write/no-delete path lock is not implemented"
        )
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    generic_read = 0x80000000
    file_share_read = 0x00000001
    open_existing = 3
    file_attribute_normal = 0x00000080
    handle = create_file(
        str(path),
        generic_read,
        file_share_read,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise OSError(error, f"could not lock authenticated reference snapshot: {path}")
    try:
        yield
    finally:
        if not close_handle(handle):
            error = ctypes.get_last_error()
            raise OSError(error, f"could not close reference snapshot lock: {path}")


def _reference_state_owner(reference: Any) -> torch.nn.Module:
    point_detector = getattr(reference, "point_detector", None)
    detector = getattr(point_detector, "fire_detection_model", None)
    owner = detector if isinstance(detector, torch.nn.Module) else getattr(detector, "model", None)
    _require(
        isinstance(owner, torch.nn.Module),
        "loaded reference detector exposes no torch state owner",
    )
    return owner


def _reference_runtime_receipt(
    reference: Any,
    *,
    detector_weights_sha256: str,
) -> _ReferenceRuntimeReceipt:
    expected = _sha256_digest(
        detector_weights_sha256, label="reference detector weights SHA256"
    )
    point_detector = getattr(reference, "point_detector", None)
    _require(point_detector is not None, "loaded reference point detector is absent")
    owner = _reference_state_owner(reference)
    state = owner.state_dict()
    receipt = _ReferenceRuntimeReceipt(
        detector_weights_sha256=expected,
        detector_state_sha256=_tensor_mapping_sha256(
            state, label="loaded reference detector state"
        ),
        detector_type_module=type(point_detector).__module__,
        detector_type_name=type(point_detector).__name__,
        state_owner_type_module=type(owner).__module__,
        state_owner_type_name=type(owner).__name__,
    )
    return receipt


def _validate_reference_runtime_receipt(
    reference: Any,
) -> _ReferenceRuntimeReceipt:
    receipt = getattr(reference, "authenticated_reference_runtime_receipt", None)
    _require(
        isinstance(receipt, _ReferenceRuntimeReceipt),
        "authenticated reference runtime receipt is absent",
    )
    recomputed = _reference_runtime_receipt(
        reference,
        detector_weights_sha256=receipt.detector_weights_sha256,
    )
    _require(
        recomputed == receipt,
        "loaded reference detector state receipt drift",
    )
    return receipt


def _load_reference_from_authenticated_snapshot(
    detector_bytes: bytes,
    *,
    expected_sha256: str,
) -> FrozenProductionAutomaticReference:
    """Load only a private immutable copy of already-authenticated weights."""

    expected = _sha256_digest(
        expected_sha256,
        label="reference detector snapshot SHA256",
    )
    _require(
        _sha256_bytes(detector_bytes) == expected,
        "reference detector bytes differ from authenticated digest",
    )
    if os.name != "nt":
        raise OSError(
            "authenticated reference path loading is supported only on Windows"
        )
    with tempfile.TemporaryDirectory(prefix="vdn_reference_authenticated_") as root:
        private_root = Path(root)
        os.chmod(private_root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        snapshot = private_root / f"reference-{secrets.token_hex(32)}.pt"
        with snapshot.open("xb") as stream:
            stream.write(detector_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(snapshot, stat.S_IRUSR)
        try:
            with _windows_no_write_delete_path_lock(snapshot):
                before = _snapshot_file_identity(snapshot)
                _require(
                    _sha256_bytes(
                        _read_bytes_once(snapshot, label="reference snapshot")
                    )
                    == expected,
                    "authenticated reference snapshot changed before load",
                )
                reference = FrozenProductionAutomaticReference.from_weights(
                    snapshot,
                    expected_sha256=expected,
                )
                after = _snapshot_file_identity(snapshot)
                _require(
                    after == before,
                    "authenticated reference snapshot identity changed during load",
                )
                _require(
                    _sha256_bytes(
                        _read_bytes_once(snapshot, label="reference snapshot")
                    )
                    == expected,
                    "authenticated reference snapshot changed during load",
                )
                receipt = _reference_runtime_receipt(
                    reference,
                    detector_weights_sha256=expected,
                )
        finally:
            if snapshot.exists():
                os.chmod(snapshot, stat.S_IRUSR | stat.S_IWUSR)
    _require(
        type(reference) is FrozenProductionAutomaticReference,
        "authenticated reference loader returned an unexpected provider type",
    )
    _require(
        reference.detector_sha256 == expected,
        "loaded reference provider detector digest drift",
    )
    setattr(reference, "authenticated_snapshot_sha256", expected)
    setattr(reference, "authenticated_snapshot_load_verified", True)
    setattr(reference, "authenticated_reference_runtime_receipt", receipt)
    _validate_reference_runtime_receipt(reference)
    return reference


def _terminal_status_from_authenticated_runtime(
    *,
    runtime: _AuthenticatedVDNRuntime,
    reference: Mapping[str, Any],
    direction_valid: bool,
    pointer_angle: float | None,
) -> tuple[bool, float | None, str | None]:
    if not direction_valid or pointer_angle is None:
        return False, None, "invalid_direction"
    if reference.get("status") is not True:
        return False, None, str(
            reference.get("failure_code") or "reference_unavailable"
        )
    try:
        start = float(reference.get("start_angle"))
        angle_range = float(reference.get("range_angle"))
    except (TypeError, ValueError):
        return False, None, "invalid_reference"
    if (
        not math.isfinite(start)
        or not math.isfinite(angle_range)
        or abs(angle_range) <= 1e-12
    ):
        return False, None, "invalid_reference"
    try:
        reading, progress = runtime.reading_from_pointer_angle(
            float(pointer_angle),
            start_angle=start,
            range_angle=angle_range,
            scale_start=0.0,
            scale_end=1.0,
        )
        reading = float(reading)
        progress = float(progress)
        _require(
            math.isfinite(reading)
            and math.isfinite(progress)
            and math.isclose(reading, progress, rel_tol=0.0, abs_tol=1e-12)
            and 0.0 <= progress <= 1.0,
            "authenticated VDN progress conversion drift",
        )
    except (TypeError, ValueError, ArithmeticError):
        return False, None, "progress_conversion_failed"
    return True, progress, None


class _Seed20TerminalVDNAdapter(VDNOfficial200LabelFreeAdapter):
    """Candidate VDN adapter built only from one authenticated byte envelope."""

    def __init__(
        self,
        *,
        authenticated_artifacts: _AuthenticatedTerminalArtifacts,
        vdn_source: Path,
        device: torch.device,
    ) -> None:
        binding, checkpoint = _validate_authenticated_terminal_artifacts(
            authenticated_artifacts
        )
        _validate_checkpoint_metadata(checkpoint)
        actual_model_state_sha256 = model_state_sha256(checkpoint[MODEL_STATE_KEY])
        _validate_terminal_binding(
            binding,
            expected_model_state_sha256=actual_model_state_sha256,
        )
        runtime = _load_authenticated_vdn_runtime_snapshot()
        source = _guard_public_directory(vdn_source, label="VDN source checkout")
        signature = checkpoint["signature"]
        self.checkpoint_sha256 = binding.checkpoint_sha256
        self.verification_sha256 = binding.verification_sha256
        self.verification_protocol = binding.verification_protocol
        self.model_state_sha256 = actual_model_state_sha256
        self.terminal_binding = binding
        self.authenticated_terminal_bytes_verified = True
        self.vdn_source = source
        self.image_size = int(signature["image_size"])
        self.heatmap_size = int(signature["heatmap_size"])
        self.model_state_key = MODEL_STATE_KEY
        self.authenticated_vdn_runtime_module_name = runtime.module_name
        self.authenticated_vdn_runtime_source_sha256 = runtime.source_sha256
        self._authenticated_vdn_runtime = runtime
        self.model = _load_vdn_model_from_authenticated_source_snapshot(
            source,
            image_size=self.image_size,
            runtime=runtime,
        )
        self.authenticated_vdn_module_name = getattr(
            self.model,
            "authenticated_vdn_module_name",
            None,
        )
        self.authenticated_vdn_source_sha256 = getattr(
            self.model,
            "authenticated_vdn_source_sha256",
            None,
        )
        self.authenticated_vdn_source_snapshot_verified = getattr(
            self.model,
            "authenticated_vdn_source_snapshot_verified",
            False,
        )
        self.model.load_state_dict(checkpoint[MODEL_STATE_KEY], strict=True)
        loaded_state_sha256 = model_state_sha256(self.model.state_dict())
        _require(
            loaded_state_sha256 == actual_model_state_sha256,
            "loaded VDN state differs from authenticated terminal state",
        )
        self.authenticated_loaded_model_state_sha256 = loaded_state_sha256
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self._runtime_telemetry_receipt = _recompute_adapter_runtime_telemetry(
            self,
            terminal_binding=binding,
        )
        _validate_adapter_runtime_telemetry(
            self,
            terminal_binding=binding,
        )
        self.runtime_telemetry_receipt_recomputed = True

    @torch.inference_mode()
    def predict(
        self,
        items: list[Any],
        images: list[np.ndarray],
        references: list[dict[str, Any]],
        *,
        reference_mode: str,
        reference_detector_sha256: str | None,
        amp_enabled: bool,
    ) -> list[dict[str, Any]]:
        """Run only helpers compiled from the authenticated VDN byte snapshot."""

        runtime = _validate_authenticated_vdn_runtime(
            self._authenticated_vdn_runtime
        )
        _require(
            len(items) == len(images) == len(references),
            "VDN batch/reference alignment drift",
        )
        tensors = [
            runtime.normalized_bgr_tensor(
                direct_resize_whole_roi(image, size=self.image_size)
            )
            for image in images
        ]
        inputs = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.type, enabled=amp_enabled):
            heatmaps, vector_maps = self.model(inputs)
        heatmaps = heatmaps.float()
        vector_maps = vector_maps.float()
        directions, peaks, valid = runtime.predict_directions(
            heatmaps, vector_maps
        )
        batch, _, _, width = heatmaps.shape
        flat = heatmaps[:, 0].reshape(batch, -1)
        indices = flat.argmax(dim=1)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        batch_indices = torch.arange(batch, device=heatmaps.device)
        raw_vectors = vector_maps[batch_indices, :, y, x]
        raw_norm = torch.linalg.vector_norm(raw_vectors, dim=1)
        directions_np = directions.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        stride = float(self.image_size) / float(heatmaps.shape[-1])
        records: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            direction = directions_np[index].astype(np.float64)
            pointer_angle: float | None = None
            if bool(valid_np[index]):
                try:
                    pointer_angle = runtime.image_angle_from_direction(direction)
                except ValueError:
                    pass
            status, progress, failure = _terminal_status_from_authenticated_runtime(
                runtime=runtime,
                reference=references[index],
                direction_valid=bool(valid_np[index]),
                pointer_angle=pointer_angle,
            )
            records.append(
                _method_record(
                    item=item,
                    checkpoint_sha256=self.checkpoint_sha256,
                    reference_mode=reference_mode,
                    reference_packet=references[index],
                    reference_packet_sha256=canonical_json_sha256(
                        references[index]
                    ),
                    reference_detector_sha256=reference_detector_sha256,
                    status=status,
                    prediction_progress=progress,
                    failure_code=failure,
                    telemetry={
                        "direction_xy": direction,
                        "pointer_angle": pointer_angle,
                        "heatmap_peak": float(peaks[index]),
                        "tip_heatmap_xy": [int(x[index]), int(y[index])],
                        "tip_input_xy": [
                            float(x[index]) * stride,
                            float(y[index]) * stride,
                        ],
                        "direction_raw_norm": float(raw_norm[index]),
                        "native_input_size": self.image_size,
                        "whole_roi_direct_resize": True,
                        "meter_bbox": [
                            0,
                            0,
                            int(images[index].shape[1]),
                            int(images[index].shape[0]),
                        ],
                        "meter_detector_not_invoked": True,
                        "second_crop_not_invoked": True,
                    },
                )
            )
        return records


def _validate_runtime_components_for_telemetry(
    direction_adapter: Any,
    reference_provider: Any,
    *,
    terminal_binding: Seed20TerminalBinding,
    candidate_binding: Mapping[str, Any],
) -> tuple[_AdapterRuntimeTelemetryReceipt, _ReferenceRuntimeReceipt]:
    _require(
        not _PRELOADED_RUNTIME_MODULES_AT_IMPORT,
        "candidate cannot use runtime modules preloaded before adapter import",
    )
    binding = _validate_candidate_source_binding(candidate_binding)
    _validate_terminal_binding(
        terminal_binding,
        expected_model_state_sha256=str(binding["checkpoint_state_sha256"]),
    )
    _require(
        type(direction_adapter) is _Seed20TerminalVDNAdapter,
        "candidate provider requires the concrete terminal VDN adapter",
    )
    adapter_telemetry = _validate_adapter_runtime_telemetry(
        direction_adapter,
        terminal_binding=terminal_binding,
    )
    _require(
        direction_adapter.terminal_binding == terminal_binding,
        "delegate terminal binding object drift",
    )
    expected_direction_identity = {
        "checkpoint_sha256": TERMINAL_CHECKPOINT_SHA256,
        "verification_sha256": VERIFICATION_SHA256,
        "verification_protocol": VERIFICATION_PROTOCOL,
        "model_state_key": MODEL_STATE_KEY,
        "model_state_sha256": terminal_binding.model_state_sha256,
        "image_size": NATIVE_INPUT_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "authenticated_terminal_bytes_verified": True,
        "authenticated_loaded_model_state_sha256": terminal_binding.model_state_sha256,
        "authenticated_vdn_runtime_source_sha256": (
            _EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"]
        ),
        "authenticated_vdn_source_sha256": VDN_MODEL_SOURCE_SHA256,
        "authenticated_vdn_source_snapshot_verified": True,
        "runtime_telemetry_receipt_recomputed": True,
    }
    for field, expected in expected_direction_identity.items():
        _require(
            getattr(direction_adapter, field, None) == expected,
            f"candidate direction component identity drift: {field}",
        )
    _require(
        type(direction_adapter.model).__module__.startswith(
            _AUTHENTICATED_VDN_MODULE_PREFIX
        )
        and type(direction_adapter.model).__name__ == "VDNModel",
        "candidate direction model concrete type drift",
    )
    runtime = _validate_authenticated_vdn_runtime(
        getattr(direction_adapter, "_authenticated_vdn_runtime", None)
    )
    source = _guard_public_directory(
        direction_adapter.vdn_source, label="candidate VDN source checkout"
    )
    _require(
        runtime.verify_vdn_source(source) == VDN_SOURCE_COMMIT,
        "candidate direction source commit drift",
    )
    _require(
        getattr(direction_adapter.model, "authenticated_vdn_module_name", None)
        == type(direction_adapter.model).__module__,
        "candidate VDN model module receipt drift",
    )
    _require(
        getattr(
            direction_adapter.model,
            "authenticated_vdn_source_sha256",
            None,
        )
        == VDN_MODEL_SOURCE_SHA256
        and getattr(
            direction_adapter.model,
            "authenticated_vdn_source_snapshot_verified",
            False,
        )
        is True,
        "candidate VDN model source snapshot receipt drift",
    )
    _require(
        model_state_sha256(direction_adapter.model.state_dict())
        == terminal_binding.model_state_sha256,
        "candidate loaded VDN state receipt drift",
    )
    _require(
        type(reference_provider) is FrozenProductionAutomaticReference,
        "candidate provider requires the concrete frozen reference provider",
    )
    _require(
        reference_provider.detector_sha256 == REFERENCE_DETECTOR_SHA256,
        "candidate reference detector attribute identity drift",
    )
    _require(
        getattr(reference_provider, "authenticated_snapshot_sha256", None)
        == REFERENCE_DETECTOR_SHA256
        and getattr(
            reference_provider,
            "authenticated_snapshot_load_verified",
            False,
        )
        is True,
        "candidate reference detector snapshot receipt drift",
    )
    reference_receipt = _validate_reference_runtime_receipt(reference_provider)
    _require(
        reference_receipt.detector_weights_sha256 == REFERENCE_DETECTOR_SHA256,
        "candidate reference weights receipt drift",
    )
    _require(
        reference_receipt.detector_state_sha256
        == binding["reference_detector_state_sha256"],
        "candidate reference state receipt drift",
    )
    reference_identity = reference_provider.identity
    _require(
        isinstance(reference_identity, Mapping),
        "candidate reference identity is absent",
    )
    expected_reference_identity = {
        "protocol": REFERENCE_PROTOCOL,
        "provider": "frozen_production_start_end_detector",
        "reference_detector_sha256": REFERENCE_DETECTOR_SHA256,
        "detector_loader_source_sha256": (
            _EXPECTED_BEHAVIOR_SOURCE_SHA256["reference_loader"]
        ),
        "primary_success_branch": "start_and_end",
        "external_reference_input": False,
    }
    for field, expected in expected_reference_identity.items():
        _require(
            reference_identity.get(field) == expected,
            f"candidate reference component identity drift: {field}",
        )
    point_detector_type = type(reference_provider.point_detector)
    _require(
        point_detector_type.__module__
        == "utils.angleDetect.yoloDetection.yoloDectect"
        and point_detector_type.__name__ == "targetDetectModel",
        "candidate reference detector concrete type drift",
    )
    return adapter_telemetry, reference_receipt


def _untrusted_runtime_source_identity() -> dict[str, Any]:
    wrapper = _guard_public_component_path(
        _WRAPPER_SOURCE, label="untrusted runtime adapter wrapper source"
    )
    return {
        "protocol": SOURCE_BINDING_PROTOCOL,
        "binding_origin": "runtime_candidate_untrusted_unrecomputed",
        "paper_eligibility_authority": "external_frozen_evaluation_protocol_only",
        "formal_success_eligible": False,
        "adapter_wrapper_sha256": _sha256_bytes(
            _read_bytes_once(wrapper, label="untrusted runtime adapter wrapper source")
        ),
        "dependency_source_sha256": _measure_behavior_sources(),
        "vdn_source_commit": VDN_SOURCE_COMMIT,
        "vdn_model_source_sha256": VDN_MODEL_SOURCE_SHA256,
        "vdn_runtime_source_sha256": _EXPECTED_BEHAVIOR_SOURCE_SHA256["vdn_runtime"],
    }


class VDNSeed20TerminalMatchedProvider(VDNFullAutoProgressProvider):
    """Image-only VDN runtime evidence with no in-process eligibility authority."""

    def __init__(
        self,
        direction_adapter: Any,
        *,
        automatic_reference_provider: Any,
        amp_enabled: bool,
        terminal_binding: Seed20TerminalBinding,
    ) -> None:
        _validate_terminal_binding(terminal_binding)
        source_binding = _untrusted_runtime_source_identity()
        super().__init__(
            direction_adapter,
            automatic_reference_provider=automatic_reference_provider,
            amp_enabled=amp_enabled,
        )
        _require(
            self._identity.get("checkpoint_sha256")
            == terminal_binding.checkpoint_sha256,
            "delegate checkpoint identity drift",
        )
        _require(
            getattr(direction_adapter, "model_state_key", None) == MODEL_STATE_KEY,
            "delegate model state role drift",
        )
        self.terminal_binding = terminal_binding
        self._candidate_binding_sha256: str | None = None
        identity = dict(self._identity)
        identity.update(
            {
                "protocol": PROTOCOL,
                "method": "vdn_official200",
                "execution_mode": "runtime_candidate_untrusted",
                "runtime_evidence_status": "not_recomputed_by_factory",
                "synthetic": True,
                "paper_eligible": False,
                "formal_success_eligible": False,
                "formal_source_binding_verified": False,
                "paper_eligibility_authority": (
                    "external_frozen_evaluation_protocol_only"
                ),
                "candidate_binding_sha256": None,
                "adapter_runtime_telemetry": None,
                "checkpoint_role": CHECKPOINT_ROLE,
                "model_state_key": MODEL_STATE_KEY,
                "model_state_sha256": terminal_binding.model_state_sha256,
                "terminal_binding": terminal_binding.as_identity(),
                "source_binding": source_binding,
                "adapter_wrapper_source_sha256": source_binding[
                    "adapter_wrapper_sha256"
                ],
                "runtime_source_sha256": source_binding[
                    "dependency_source_sha256"
                ],
                "matched_input_contract": {
                    "input": "one whole canonical meter ROI BGR uint8 array",
                    "direct_resize_to": [NATIVE_INPUT_SIZE, NATIVE_INPUT_SIZE],
                    "crop": False,
                    "letterbox": False,
                    "padding": False,
                    "second_crop": False,
                    "meter_bbox_consumed": False,
                    "caller_sample_or_group_id_consumed": False,
                    "id_lookup_performed": False,
                    "ground_truth_consumed": False,
                    "target_consumed": False,
                    "physical_scale_consumed": False,
                    "caller_reference_consumed": False,
                    "automatic_reference_internal": True,
                },
                "runtime_input_keys": [
                    "canonical_meter_roi_bgr",
                    "input_is_canonical_meter_roi",
                ],
            }
        )
        identity["identity_payload_sha256"] = canonical_json_sha256(identity)
        assert_label_free(identity, location="vdn_seed20_terminal.identity")
        self._identity = identity

    def _attach_recomputed_runtime_telemetry(
        self,
        *,
        source_binding: Mapping[str, Any],
        adapter_telemetry: _AdapterRuntimeTelemetryReceipt,
    ) -> None:
        """Attach non-authoritative evidence for an external frozen protocol."""

        binding = _validate_candidate_source_binding(source_binding)
        _require(
            type(adapter_telemetry) is _AdapterRuntimeTelemetryReceipt,
            "adapter runtime telemetry type drift",
        )
        candidate_binding_sha256 = str(binding["payload_sha256"])
        identity = dict(self._identity)
        identity.pop("identity_payload_sha256", None)
        identity.update(
            {
                "execution_mode": "runtime_candidate_untrusted",
                "runtime_evidence_status": "recomputed_in_process_untrusted",
                "synthetic": False,
                "paper_eligible": False,
                "formal_success_eligible": False,
                "formal_source_binding_verified": False,
                "paper_eligibility_authority": (
                    "external_frozen_evaluation_protocol_only"
                ),
                "candidate_binding_sha256": candidate_binding_sha256,
                "adapter_runtime_telemetry": adapter_telemetry.as_identity(),
                "source_binding": binding,
                "adapter_wrapper_source_sha256": binding[
                    "adapter_wrapper_sha256"
                ],
                "runtime_source_sha256": binding["dependency_source_sha256"],
            }
        )
        identity["identity_payload_sha256"] = canonical_json_sha256(identity)
        assert_label_free(identity, location="vdn_seed20_terminal.identity")
        self._candidate_binding_sha256 = candidate_binding_sha256
        self._identity = identity

    @classmethod
    def _from_test_components(
        cls,
        direction_adapter: Any,
        *,
        automatic_reference_provider: Any,
        amp_enabled: bool,
        terminal_binding: Seed20TerminalBinding,
    ) -> "VDNSeed20TerminalMatchedProvider":
        """Construct untrusted runtime evidence from unit-contract components."""

        return cls(
            direction_adapter,
            automatic_reference_provider=automatic_reference_provider,
            amp_enabled=amp_enabled,
            terminal_binding=terminal_binding,
        )

    @classmethod
    def from_authenticated_files(
        cls,
        *,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        verification_path: Path = DEFAULT_VERIFICATION,
        vdn_source: Path = DEFAULT_VDN_SOURCE,
        reference_detector_path: Path = DEFAULT_REFERENCE_DETECTOR,
        device: str | torch.device = "cpu",
        amp_enabled: bool | None = None,
    ) -> "VDNSeed20TerminalMatchedProvider":
        authenticated_artifacts = _authenticate_terminal_artifacts(
            checkpoint_path=checkpoint_path,
            verification_path=verification_path,
        )
        binding = authenticated_artifacts.binding
        source = _guard_public_directory(vdn_source, label="VDN source checkout")
        reference_file = _guard_public_component_path(
            reference_detector_path, label="automatic reference detector"
        )
        reference_bytes = _read_bytes_once(
            reference_file, label="automatic reference detector"
        )
        _require(
            _sha256_bytes(reference_bytes) == REFERENCE_DETECTOR_SHA256,
            "automatic reference detector SHA256 mismatch",
        )
        torch_device = torch.device(device)
        adapter = _Seed20TerminalVDNAdapter(
            authenticated_artifacts=authenticated_artifacts,
            vdn_source=source,
            device=torch_device,
        )
        reference = _load_reference_from_authenticated_snapshot(
            reference_bytes,
            expected_sha256=REFERENCE_DETECTOR_SHA256,
        )
        reference_receipt = _validate_reference_runtime_receipt(reference)
        source_binding = _build_candidate_source_binding(
            terminal_binding=binding,
            reference_receipt=reference_receipt,
        )
        adapter_telemetry, validated_reference_receipt = (
            _validate_runtime_components_for_telemetry(
                adapter,
                reference,
                terminal_binding=binding,
                candidate_binding=source_binding,
            )
        )
        _require(
            validated_reference_receipt == reference_receipt,
            "runtime reference telemetry changed during collection",
        )
        provider = cls(
            adapter,
            automatic_reference_provider=reference,
            amp_enabled=(
                torch_device.type == "cuda" if amp_enabled is None else amp_enabled
            ),
            terminal_binding=binding,
        )
        provider._attach_recomputed_runtime_telemetry(
            source_binding=source_binding,
            adapter_telemetry=adapter_telemetry,
        )
        return provider

    @property
    def identity(self) -> Mapping[str, Any]:
        identity = copy.deepcopy(self._identity)
        identity.pop("identity_payload_sha256", None)
        for key in (
            "eligible_for_paper",
            "publication_eligible",
            "formal_status",
        ):
            identity.pop(key, None)
        identity["execution_mode"] = "runtime_candidate_untrusted"
        identity["paper_eligible"] = False
        identity["formal_success_eligible"] = False
        identity["formal_source_binding_verified"] = False
        identity["paper_eligibility_authority"] = (
            "external_frozen_evaluation_protocol_only"
        )
        source_binding = identity.get("source_binding")
        if isinstance(source_binding, dict):
            source_binding.pop("eligible_for_paper", None)
            source_binding.pop("paper_eligible", None)
            source_binding.pop("publication_eligible", None)
            source_binding["formal_success_eligible"] = False
            source_binding["paper_eligibility_authority"] = (
                "external_frozen_evaluation_protocol_only"
            )
        identity["identity_payload_sha256"] = canonical_json_sha256(identity)
        return identity

    def _failure_record(
        self,
        *,
        image_sha256: str,
        failure_code: str,
        runtime_exception_type: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "method": "vdn_official200",
            "execution_mode": "runtime_candidate_untrusted",
            "paper_eligible": False,
            "paper_eligibility_authority": (
                "external_frozen_evaluation_protocol_only"
            ),
            "formal_success_eligible": False,
            "formal_status": "failure",
            "status": False,
            "prediction_progress": None,
            "failure_code": str(failure_code)[:128],
            "checkpoint_sha256": TERMINAL_CHECKPOINT_SHA256,
            "checkpoint_role": CHECKPOINT_ROLE,
            "model_state_key": MODEL_STATE_KEY,
            "model_state_sha256": self.terminal_binding.model_state_sha256,
            "candidate_binding_sha256": self._candidate_binding_sha256,
            "canonical_roi_sha256": image_sha256,
            "reference_detector_sha256": self.reference_provider.detector_sha256,
            "input_attestation": {
                "input_image_sha256": image_sha256,
                "input_is_canonical_meter_roi": True,
                "meter_detector_invoked": False,
                "second_crop_applied": False,
                "caller_sample_or_group_id_consumed": False,
                "id_lookup_performed": False,
                "ground_truth_consumed": False,
                "physical_scale_consumed": False,
                "manual_reference_consumed": False,
                "caller_reference_packet_consumed": False,
            },
            "telemetry": {
                "native_input_size": NATIVE_INPUT_SIZE,
                "heatmap_size": HEATMAP_SIZE,
                "whole_roi_contract_required": True,
                "letterbox_allowed": False,
                "padding_allowed": False,
                "runtime_exception_type": runtime_exception_type,
            },
            "component_identity": self.identity,
        }
        assert_label_free(record, location="vdn_seed20_terminal.failure")
        record["record_sha256"] = canonical_json_sha256(record)
        return record

    def _standardize_record(
        self,
        value: Any,
        *,
        image_sha256: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_output_not_mapping",
            )
        record = dict(value)
        for key in (
            "eligible_for_paper",
            "formal_status",
            "formal_success_eligible",
            "paper_eligibility_authority",
            "paper_eligible",
            "publication_eligible",
        ):
            record.pop(key, None)
        status = record.get("status")
        if status is not True and status is not False:
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_status_invalid",
            )
        prediction = record.get("prediction_progress")
        if status is True:
            try:
                progress = float(prediction)
            except (TypeError, ValueError):
                progress = math.nan
            if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
                return self._failure_record(
                    image_sha256=image_sha256,
                    failure_code="delegate_success_progress_invalid",
                )
            if record.get("failure_code") is not None:
                return self._failure_record(
                    image_sha256=image_sha256,
                    failure_code="delegate_success_has_failure_code",
                )
        elif prediction is not None or not record.get("failure_code"):
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_failure_record_invalid",
            )
        if record.get("checkpoint_sha256") != TERMINAL_CHECKPOINT_SHA256:
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_checkpoint_identity_drift",
            )
        if record.get("canonical_roi_sha256") != image_sha256:
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_roi_identity_drift",
            )
        if record.get("reference_detector_sha256") != REFERENCE_DETECTOR_SHA256:
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_reference_identity_drift",
            )
        telemetry = record.get("telemetry")
        if not isinstance(telemetry, Mapping) or (
            telemetry.get("whole_roi_direct_resize") is not True
            or telemetry.get("second_crop_not_invoked") is not True
            or telemetry.get("native_input_size") != NATIVE_INPUT_SIZE
        ):
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_preprocessing_attestation_invalid",
            )
        attestation = record.get("input_attestation")
        required_false = (
            "meter_detector_invoked",
            "second_crop_applied",
            "ground_truth_consumed",
            "physical_scale_consumed",
            "manual_reference_consumed",
            "caller_reference_packet_consumed",
        )
        if (
            not isinstance(attestation, Mapping)
            or attestation.get("input_image_sha256") != image_sha256
            or attestation.get("input_is_canonical_meter_roi") is not True
            or any(attestation.get(key) is not False for key in required_false)
        ):
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_input_attestation_invalid",
            )
        record["input_attestation"] = {
            **dict(attestation),
            "caller_sample_or_group_id_consumed": False,
            "id_lookup_performed": False,
        }
        record.update(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "method": "vdn_official200",
                "execution_mode": "runtime_candidate_untrusted",
                "paper_eligible": False,
                "paper_eligibility_authority": (
                    "external_frozen_evaluation_protocol_only"
                ),
                "formal_success_eligible": False,
                "formal_status": (
                    "runtime_candidate_untrusted"
                    if status is True
                    else "failure"
                ),
                "checkpoint_role": CHECKPOINT_ROLE,
                "model_state_key": MODEL_STATE_KEY,
                "model_state_sha256": self.terminal_binding.model_state_sha256,
                "candidate_binding_sha256": self._candidate_binding_sha256,
                "component_identity": self.identity,
            }
        )
        try:
            assert_label_free(record, location="vdn_seed20_terminal.output")
        except (TypeError, ValueError):
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="delegate_output_contract_invalid",
            )
        record["record_sha256"] = canonical_json_sha256(record)
        return record

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        if input_is_canonical_meter_roi is not True:
            raise ValueError("VDN terminal provider requires canonical ROI attestation")
        if (
            not isinstance(canonical_meter_roi_bgr, np.ndarray)
            or canonical_meter_roi_bgr.dtype != np.uint8
            or canonical_meter_roi_bgr.ndim != 3
            or canonical_meter_roi_bgr.shape[2] != 3
            or min(canonical_meter_roi_bgr.shape[:2]) < 2
        ):
            raise ValueError("canonical meter ROI must be uint8 BGR [H,W,3]")
        image = np.ascontiguousarray(canonical_meter_roi_bgr)
        image_sha256 = canonical_roi_pixel_sha256(image)
        runtime_image = image.copy()
        runtime_sha256 = canonical_roi_pixel_sha256(runtime_image)
        try:
            raw = super().predict(
                runtime_image,
                input_is_canonical_meter_roi=True,
            )
        except Exception as exc:
            mutated = canonical_roi_pixel_sha256(runtime_image) != runtime_sha256
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code=(
                    "runtime_mutated_canonical_roi_copy"
                    if mutated
                    else f"runtime_exception:{type(exc).__name__}"
                ),
                runtime_exception_type=type(exc).__name__,
            )
        if canonical_roi_pixel_sha256(runtime_image) != runtime_sha256:
            return self._failure_record(
                image_sha256=image_sha256,
                failure_code="runtime_mutated_canonical_roi_copy",
            )
        return self._standardize_record(raw, image_sha256=image_sha256)


def build_seed20_terminal_matched_provider(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    verification_path: Path = DEFAULT_VERIFICATION,
    vdn_source: Path = DEFAULT_VDN_SOURCE,
    reference_detector_path: Path = DEFAULT_REFERENCE_DETECTOR,
    device: str | torch.device = "cpu",
    amp_enabled: bool | None = None,
) -> VDNSeed20TerminalMatchedProvider:
    """Build recomputable runtime evidence; never grant paper eligibility."""

    return VDNSeed20TerminalMatchedProvider.from_authenticated_files(
        checkpoint_path=checkpoint_path,
        verification_path=verification_path,
        vdn_source=vdn_source,
        reference_detector_path=reference_detector_path,
        device=device,
        amp_enabled=amp_enabled,
    )


__all__ = [
    "CHECKPOINT_ROLE",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_REFERENCE_DETECTOR",
    "DEFAULT_VDN_SOURCE",
    "DEFAULT_VERIFICATION",
    "EXECUTION_RUNTIME_CANDIDATE_UNTRUSTED",
    "FIT_SAMPLE_IDS_SHA256",
    "HOLDOUT_SAMPLE_IDS_SHA256",
    "MODEL_STATE_KEY",
    "PROTOCOL",
    "REFERENCE_DETECTOR_SHA256",
    "SEED",
    "SOURCE_BINDING_PROTOCOL",
    "Seed20TerminalBinding",
    "TERMINAL_CHECKPOINT_SHA256",
    "TERMINAL_EPOCH",
    "VDN_MODEL_SOURCE_SHA256",
    "VDNSeed20TerminalMatchedProvider",
    "build_seed20_terminal_matched_provider",
    "verify_seed20_terminal_artifacts",
]
