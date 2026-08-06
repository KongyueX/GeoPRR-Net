"""Hash-bound two-stage evaluation for the frozen V5 model roster.

The entry point intentionally separates evaluation into three irreversible
steps:

``freeze``
    Bind a label-free image roster, per-method checkpoints, inference adapters,
    and the canonical-ROI/automatic-reference contracts before labels are
    accessed.

``infer``
    Validate and seal label-free predictions.  A complete primary method emits
    normalized progress plus an automatically inferred numeric scale start/end
    and range confidence.  Ground truth, caller-supplied scale values, targets,
    errors, and label-derived diagnostics are recursively forbidden.  Every
    method attests to the same canonical meter ROI.  A supplied common reference
    packet is permitted only in a separately frozen controlled diagnostic.

``score``
    Verify the inference seal and only then join the separately supplied labels.
    The full prediction is ``pred_start + progress * (pred_end - pred_start)``.
    Failures receive normalized error 1.0.  The report includes range-pair
    coverage/exactness, full end-to-end image-micro,
    physical-group macro, duplicate-frame-collapsed, non-zero-target, unique-GT,
    and zero-reading null-baseline results.

Progress-only outputs remain useful component diagnostics, but are explicitly
ineligible for the complete automatic-reading primary table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL: Final[str] = "v5_unified_two_stage_retest_protocol_v2"
INFERENCE_PROTOCOL: Final[str] = "v5_label_free_inference_packet_v2"
SEAL_PROTOCOL: Final[str] = "v5_label_free_inference_seal_v2"
SCORE_PROTOCOL: Final[str] = "v5_unified_two_stage_score_v2"
HASH_PROTOCOL: Final[str] = "sha256_raw_bytes_and_canonical_json_v1"
FAILURE_PENALTY: Final[float] = 1.0
RANGE_PAIR_ABS_TOLERANCE: Final[float] = 1e-6
RANGE_PAIR_REL_TOLERANCE: Final[float] = 1e-6
HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")

CANONICAL_TIGHT_ROI_CONTRACT: Final[dict[str, Any]] = {
    "name": "canonical_tight_roi_v1",
    "public_source_roi": (
        "clip ground-truth xyxy bbox to image, direct slice, resize 256x256"
    ),
    "deployment_roi": "whole supplied ROI; bbox=None; detector forbidden",
    "bbox_expansion": 1.0,
    "preserve_aspect_ratio_with_letterbox": False,
    "out_of_bounds_padding": "forbidden",
    "constant_black_border": "forbidden",
    "shared_content_identity": "SHA256 of the whole supplied ROI before resize",
    "model_native_resize_allowed": True,
    "v5_native_resize": "direct 256x256",
    "comparator_native_resize": "direct resize to the checkpoint-native input size",
}

COMMON_REFERENCE_INTERFACE: Final[dict[str, Any]] = {
    "name": "v5_common_reference_progress_interface_v1",
    "inputs": ["status", "start_angle", "range_angle", "reference_branch"],
    "model_output": "normalized prediction_progress only",
    "same_packet_for_every_method": True,
    "ground_truth_available": False,
    "scale_values_available_during_inference": False,
    "evidence_role": "secondary controlled direction analysis only",
    "allowed_in_primary_end_to_end_table": False,
}

NATIVE_PROGRESS_INTERFACE: Final[dict[str, Any]] = {
    "name": "native_progress_from_canonical_roi_v1",
    "input": "the same whole canonical ROI supplied to every primary method",
    "model_output": "normalized prediction_progress only",
    "reference_detector_invoked": False,
    "external_reference_packet_available": False,
    "ground_truth_available": False,
    "scale_values_available_during_inference": False,
    "evidence_role": "primary end-to-end external evaluation",
}

AUTO_REFERENCE_INTERFACE: Final[dict[str, Any]] = {
    "name": "production_auto_reference_from_canonical_roi_v1",
    "input": "the same whole canonical ROI consumed by the direction model",
    "detector": "frozen production reference detector bound by SHA256",
    "outputs": ["status", "start_angle", "range_angle", "reference_branch"],
    "primary_success_branch": "start_and_end only",
    "incomplete_detection_policy": (
        "start_only/end_only/default_start_end are explicit primary failures; "
        "legacy fallback is secondary sensitivity only"
    ),
    "ground_truth_scale_geometry_available": False,
    "manual_reference_available": False,
    "scale_values_available_during_inference": False,
    "evidence_role": "primary end-to-end external evaluation",
    "method_name_suffix": "+auto-ref",
}

REFERENCE_MODE_AUTO: Final[str] = "method_internal_auto"
REFERENCE_MODE_CONTROLLED: Final[str] = "shared_controlled_packet"
REFERENCE_MODE_NATIVE: Final[str] = "native_implicit_progress"

EVIDENCE_ROLE_PRIMARY: Final[str] = "primary_end_to_end_external_evaluation"
EVIDENCE_ROLE_DIAGNOSTIC: Final[str] = "secondary_controlled_direction_diagnostic"
EVIDENCE_ROLE_COMPONENT: Final[str] = "secondary_progress_component_diagnostic"

OUTPUT_MODE_FULL_READING: Final[str] = "automatic_progress_and_numeric_range"
OUTPUT_MODE_PROGRESS_ONLY: Final[str] = "progress_only_component"

UNLABELED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
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

RAW_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "group_id",
        "image_sha256",
        "canonical_roi_sha256",
        "frame_sha256",
        "reference_input",
        "reference_input_sha256",
        "methods",
    }
)

METHOD_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "prediction_progress",
        "failure_code",
        "checkpoint_sha256",
        "roi_contract_sha256",
        "reference_mode",
        "reference_contract_sha256",
        "evidence_role",
        "output_mode",
        "canonical_roi_sha256",
        "reference_detector_sha256",
        "auto_reference",
        "auto_reference_sha256",
        "reference_input_sha256",
        "predicted_scale_start",
        "predicted_scale_end",
        "range_confidence",
        "telemetry",
    }
)

FORBIDDEN_INFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "actual",
        "actual_reading",
        "actual_value",
        "direction_angle_error",
        "direction_angle_error_degrees",
        "error",
        "errors",
        "expected",
        "ground_truth",
        "groundtruth",
        "gt",
        "label",
        "labels",
        "nae",
        "nmae",
        "normalized_error",
        "pointer_tail",
        "pointer_tip",
        "manual_reference",
        "manual_range_angle",
        "manual_start_angle",
        "scale_geometry",
        "scale_max",
        "scale_min",
        "scale_range",
        "scale_start",
        "scale_end",
        "scalemark_reference",
        "target",
        "target_direction",
        "target_progress",
        "target_reading",
        "true_progress",
        "true_value",
        "truth",
    }
)

FORBIDDEN_INFERENCE_KEYS_NORMALIZED: Final[frozenset[str]] = frozenset(
    "".join(character for character in key.casefold() if character.isalnum())
    for key in FORBIDDEN_INFERENCE_KEYS
)

REFERENCE_KEYS: Final[frozenset[str]] = frozenset(
    {"status", "start_angle", "range_angle", "reference_branch", "failure_code"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )


def strict_jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            value = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def canonical_json_bytes(value: Any) -> bytes:
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


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.lower()).issubset(HEX_DIGITS)
    )


def _normalized_key(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def assert_label_free(value: Any, *, location: str = "root") -> None:
    """Recursively reject labels, targets, errors, and GT-derived diagnostics."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_INFERENCE_KEYS_NORMALIZED:
                raise ValueError(
                    f"label-derived key {key!r} is forbidden at {location}"
                )
            assert_label_free(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            assert_label_free(nested, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite inference value at {location}")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _atomic_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id") or "")
        _require(bool(sample_id), f"{label} row {index} has no sample_id")
        _require(sample_id not in result, f"{label} duplicates sample_id {sample_id}")
        result[sample_id] = row
    _require(bool(result), f"{label} is empty")
    return result


def _validate_unlabeled_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Keep the primary inference roster limited to ROI identity and routing.

    In particular, the manifest cannot carry a crop box, keypoints, a reference
    packet, ScaleMark geometry, physical ranges, or opaque metadata that could
    smuggle any of those values into a model runner.
    """

    for index, row in enumerate(rows, 1):
        unexpected = set(row) - UNLABELED_MANIFEST_KEYS
        _require(
            not unexpected,
            f"unlabeled manifest row {index} has non-canonical inference fields "
            f"{sorted(unexpected)}",
        )


def _manifest_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_unlabeled_manifest_rows(rows)
    index = _index_rows(rows, label="unlabeled manifest")
    groups: set[str] = set()
    sample_image_pairs: list[list[str]] = []
    sample_frame_pairs: list[list[str]] = []
    for sample_id, row in index.items():
        group_id = str(row.get("group_id") or "")
        image_sha256 = str(row.get("image_sha256") or "").lower()
        frame_sha256 = str(row.get("frame_sha256") or image_sha256).lower()
        _require(bool(group_id), f"{sample_id}: missing physical group_id")
        _require(_is_sha256(image_sha256), f"{sample_id}: invalid image_sha256")
        _require(_is_sha256(frame_sha256), f"{sample_id}: invalid frame_sha256")
        groups.add(group_id)
        sample_image_pairs.append([sample_id, image_sha256])
        sample_frame_pairs.append([sample_id, frame_sha256])
    return {
        "rows": len(index),
        "physical_groups": len(groups),
        "unique_frames": len({pair[1] for pair in sample_frame_pairs}),
        "sample_ids_sha256": canonical_json_sha256(sorted(index)),
        "sample_image_pairs_sha256": canonical_json_sha256(
            sorted(sample_image_pairs)
        ),
        "sample_frame_pairs_sha256": canonical_json_sha256(
            sorted(sample_frame_pairs)
        ),
    }


def _bind_file(
    path_value: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(str(path_value)).resolve(strict=True)
    expected = str(expected_sha256 or "").lower()
    _require(_is_sha256(expected), f"{label}: invalid expected SHA256")
    actual = sha256_file(path)
    _require(actual == expected, f"{label}: file hash differs from declaration")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def freeze_protocol(
    *,
    unlabeled_manifest_path: Path,
    method_bundle_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze inputs without opening an image or a label file."""

    manifest_path = unlabeled_manifest_path.resolve(strict=True)
    bundle_path = method_bundle_path.resolve(strict=True)
    rows = strict_jsonl_load(manifest_path)
    bundle = strict_json_load(bundle_path)
    _require(isinstance(bundle, Mapping), "method bundle must be an object")
    assert_label_free(rows, location="unlabeled_manifest")
    assert_label_free(bundle, location="method_bundle")
    identity = _manifest_identity(rows)

    methods_value = bundle.get("methods")
    _require(isinstance(methods_value, Mapping) and methods_value, "no methods")
    methods: dict[str, Any] = {}
    roi_hash = canonical_json_sha256(CANONICAL_TIGHT_ROI_CONTRACT)
    controlled_reference_hash = canonical_json_sha256(COMMON_REFERENCE_INTERFACE)
    auto_reference_hash = canonical_json_sha256(AUTO_REFERENCE_INTERFACE)
    native_progress_hash = canonical_json_sha256(NATIVE_PROGRESS_INTERFACE)
    reference_modes: set[str] = set()
    for name in sorted(str(key) for key in methods_value):
        method = methods_value.get(name)
        _require(isinstance(method, Mapping), f"{name}: method binding is invalid")
        checkpoint = _bind_file(
            method.get("checkpoint_path"),
            method.get("checkpoint_sha256"),
            label=f"{name} checkpoint",
        )
        source_hash = str(
            method.get("source_evaluation_checkpoint_sha256") or ""
        ).lower()
        _require(
            source_hash == checkpoint["sha256"],
            f"{name}: source and external evaluation checkpoints differ",
        )
        _require(
            method.get("roi_contract_sha256") == roi_hash,
            f"{name}: canonical ROI contract hash differs",
        )
        reference_mode = str(method.get("reference_mode") or "")
        _require(
            reference_mode
            in (
                REFERENCE_MODE_AUTO,
                REFERENCE_MODE_CONTROLLED,
                REFERENCE_MODE_NATIVE,
            ),
            f"{name}: reference_mode must be explicit",
        )
        reference_modes.add(reference_mode)
        expected_reference_hash = {
            REFERENCE_MODE_AUTO: auto_reference_hash,
            REFERENCE_MODE_CONTROLLED: controlled_reference_hash,
            REFERENCE_MODE_NATIVE: native_progress_hash,
        }[reference_mode]
        _require(
            method.get("reference_contract_sha256") == expected_reference_hash,
            f"{name}: reference contract hash differs from its declared mode",
        )
        output_mode = str(method.get("output_mode") or "")
        _require(
            output_mode in {OUTPUT_MODE_FULL_READING, OUTPUT_MODE_PROGRESS_ONLY},
            f"{name}: output_mode must be explicit",
        )
        if reference_mode == REFERENCE_MODE_CONTROLLED:
            _require(
                output_mode == OUTPUT_MODE_PROGRESS_ONLY,
                f"{name}: controlled-reference runs are diagnostic only",
            )
            expected_evidence_role = EVIDENCE_ROLE_DIAGNOSTIC
        elif output_mode == OUTPUT_MODE_FULL_READING:
            expected_evidence_role = EVIDENCE_ROLE_PRIMARY
        else:
            expected_evidence_role = EVIDENCE_ROLE_COMPONENT
        _require(
            method.get("evidence_role") == expected_evidence_role,
            f"{name}: evidence role differs from its reference mode",
        )
        reference_detector: dict[str, Any] | None = None
        if reference_mode == REFERENCE_MODE_AUTO:
            _require(
                name.endswith("+auto-ref"),
                f"{name}: primary automatic-reference method name must end '+auto-ref'",
            )
            reference_detector = _bind_file(
                method.get("reference_detector_path"),
                method.get("reference_detector_sha256"),
                label=f"{name} automatic reference detector",
            )
        else:
            _require(
                method.get("reference_detector_path") is None
                and method.get("reference_detector_sha256") is None,
                f"{name}: non-automatic method must not bind a reference detector",
            )
        adapter: dict[str, Any] | None = None
        if method.get("adapter_path") is not None:
            adapter = _bind_file(
                method.get("adapter_path"),
                method.get("adapter_sha256"),
                label=f"{name} inference adapter",
            )
        methods[name] = {
            "checkpoint": checkpoint,
            "source_evaluation_checkpoint_sha256": source_hash,
            "same_checkpoint_for_source_and_external": True,
            "adapter": adapter,
            "roi_contract_sha256": roi_hash,
            "reference_mode": reference_mode,
            "evidence_role": expected_evidence_role,
            "output_mode": output_mode,
            "reference_contract_sha256": expected_reference_hash,
            "reference_detector": reference_detector,
        }

    evidence_roles = {method["evidence_role"] for method in methods.values()}
    _require(
        len(evidence_roles) == 1,
        "full-reading primary methods and component/controlled diagnostics must "
        "be frozen in separate protocols",
    )
    evidence_role = next(iter(evidence_roles))
    has_controlled = REFERENCE_MODE_CONTROLLED in reference_modes
    _require(
        not has_controlled or reference_modes == {REFERENCE_MODE_CONTROLLED},
        "controlled-reference diagnostics must be frozen and run separately "
        "from all primary methods",
    )
    evaluation_tier = {
        EVIDENCE_ROLE_PRIMARY: "primary_full_automatic_reading",
        EVIDENCE_ROLE_COMPONENT: "progress_component_diagnostic",
        EVIDENCE_ROLE_DIAGNOSTIC: "controlled_reference_diagnostic",
    }[evidence_role]

    protocol = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "frozen_before_label_access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hash_protocol": HASH_PROTOCOL,
        "scope": {
            "label_files_opened": 0,
            "images_opened": 0,
            "predictions_available": 0,
            "method_or_threshold_selection_allowed_after_inference": False,
            "evaluation_tier": evaluation_tier,
            "eligible_for_primary_metrics": (
                evaluation_tier == "primary_full_automatic_reading"
            ),
        },
        "unlabeled_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            **identity,
        },
        "methods": methods,
        "roi_contract": CANONICAL_TIGHT_ROI_CONTRACT,
        "roi_contract_sha256": roi_hash,
        "reference_contracts": {
            REFERENCE_MODE_AUTO: AUTO_REFERENCE_INTERFACE,
            REFERENCE_MODE_CONTROLLED: COMMON_REFERENCE_INTERFACE,
            REFERENCE_MODE_NATIVE: NATIVE_PROGRESS_INTERFACE,
        },
        "primary_reference_contract_sha256": auto_reference_hash,
        "controlled_reference_contract_sha256": controlled_reference_hash,
        "native_progress_contract_sha256": native_progress_hash,
        "inference_contract": {
            "predicted_quantities": [
                "normalized prediction_progress",
                "predicted_scale_start (model output, never caller input)",
                "predicted_scale_end (model output, never caller input)",
                "range_confidence",
            ],
            "full_reading_equation": (
                "predicted_scale_start + prediction_progress * "
                "(predicted_scale_end-predicted_scale_start)"
            ),
            "progress_only_role": "component diagnostic; never a full-reading primary result",
            "labels_or_targets_present": False,
            "same_canonical_roi_for_all_methods": True,
            "primary_reference_policy": (
                "method-internal frozen automatic reference, or a frozen native "
                "progress head that invokes no reference detector"
            ),
            "manual_or_gt_reference_in_primary_table": False,
            "shared_reference_packet_role": "secondary controlled direction analysis only",
            "second_detector_or_recrop_allowed": False,
            "forbidden_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
        },
        "scoring_contract": {
            "primary_normalized_error": (
                "abs(predicted_physical-ground_truth)/abs(scale_end-scale_start)"
            ),
            "progress_component_error": "abs(prediction_progress-target_progress)",
            "target_progress": (
                "(ground_truth-scale_start)/(scale_end-scale_start)"
            ),
            "failure_penalty": FAILURE_PENALTY,
            "nonzero_subset": "abs(physical ground_truth) > 1e-12",
            "duplicate_frame_unit": "frame_sha256",
            "duplicate_frame_collapse": "mean row error per frame, then mean frames",
            "unique_gt_unit": "physical ground_truth rounded to 12 significant digits",
            "null_baseline": "constant physical reading 0",
            "range_pair_exact": (
                "ordered predicted start/end both match label endpoints under "
                f"abs_tol={RANGE_PAIR_ABS_TOLERANCE}, "
                f"rel_tol={RANGE_PAIR_REL_TOLERANCE}"
            ),
        },
        "method_bundle": {
            "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_new(output_path.resolve(), canonical_json_bytes(protocol))
    return protocol


def _validate_protocol_and_manifest(
    protocol_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    protocol = strict_json_load(protocol_path)
    _require(
        isinstance(protocol, dict)
        and protocol.get("protocol") == PROTOCOL
        and protocol.get("status") == "frozen_before_label_access",
        "invalid frozen V5 retest protocol",
    )
    manifest = manifest_path.resolve(strict=True)
    binding = protocol.get("unlabeled_manifest") or {}
    _require(sha256_file(manifest) == binding.get("sha256"), "manifest hash drift")
    rows = strict_jsonl_load(manifest)
    assert_label_free(rows, location="unlabeled_manifest")
    identity = _manifest_identity(rows)
    for key, value in identity.items():
        _require(binding.get(key) == value, f"manifest identity drift: {key}")
    return protocol, rows, _index_rows(rows, label="unlabeled manifest")


def _reference_packet(value: Any, *, sample_id: str) -> tuple[dict[str, Any], str]:
    _require(isinstance(value, Mapping), f"{sample_id}: missing reference_input")
    extra = set(value) - REFERENCE_KEYS
    _require(not extra, f"{sample_id}: unexpected reference keys {sorted(extra)}")
    status = value.get("status")
    _require(isinstance(status, bool), f"{sample_id}: reference status is not bool")
    start = _finite(value.get("start_angle"))
    arc = _finite(value.get("range_angle"))
    if status:
        _require(start is not None, f"{sample_id}: valid reference lacks start_angle")
        _require(
            arc is not None and 0.0 < arc < 360.0,
            f"{sample_id}: invalid range_angle",
        )
        failure_code = None
    else:
        _require(start is None and arc is None, f"{sample_id}: failed reference has angles")
        failure_code = str(value.get("failure_code") or "")
        _require(bool(failure_code), f"{sample_id}: failed reference lacks failure_code")
    packet = {
        "status": status,
        "start_angle": start,
        "range_angle": arc,
        "reference_branch": str(value.get("reference_branch") or "unknown"),
        "failure_code": failure_code,
    }
    return packet, canonical_json_sha256(packet)


def _primary_automatic_reference_packet(
    value: Any,
    *,
    sample_id: str,
) -> tuple[dict[str, Any], str]:
    packet, digest = _reference_packet(value, sample_id=sample_id)
    if packet["status"]:
        branch = str(packet["reference_branch"]).casefold().split(":")[-1]
        _require(
            branch == "start_and_end",
            f"{sample_id}: successful primary automatic reference did not detect "
            "both real endpoints",
        )
    return packet, digest


def seal_inference(
    *,
    protocol_path: Path,
    unlabeled_manifest_path: Path,
    raw_predictions_path: Path,
    sealed_predictions_path: Path,
    seal_path: Path,
) -> dict[str, Any]:
    """Validate and immutably seal one complete label-free inference pass."""

    protocol_file = protocol_path.resolve(strict=True)
    manifest_file = unlabeled_manifest_path.resolve(strict=True)
    raw_file = raw_predictions_path.resolve(strict=True)
    protocol, manifest_rows, manifest_by_id = _validate_protocol_and_manifest(
        protocol_file,
        manifest_file,
    )
    raw_rows = strict_jsonl_load(raw_file)
    assert_label_free(raw_rows, location="raw_predictions")
    raw_by_id = _index_rows(raw_rows, label="raw predictions")
    _require(set(raw_by_id) == set(manifest_by_id), "prediction ID roster drift")

    method_bindings = protocol.get("methods") or {}
    method_names = tuple(sorted(method_bindings))
    roi_contract_hash = str(protocol.get("roi_contract_sha256"))
    controlled_methods = {
        name
        for name in method_names
        if method_bindings[name].get("reference_mode")
        == REFERENCE_MODE_CONTROLLED
    }
    evaluation_tier = str((protocol.get("scope") or {}).get("evaluation_tier") or "")
    _require(
        evaluation_tier
        in {
            "primary_full_automatic_reading",
            "progress_component_diagnostic",
            "controlled_reference_diagnostic",
        },
        "frozen evaluation tier is invalid",
    )
    _require(
        bool(controlled_methods)
        == (evaluation_tier == "controlled_reference_diagnostic"),
        "reference modes disagree with the frozen evaluation tier",
    )
    sealed_rows: list[dict[str, Any]] = []
    success_counts = {name: 0 for name in method_names}
    for manifest in manifest_rows:
        sample_id = str(manifest["sample_id"])
        raw = raw_by_id[sample_id]
        unexpected_raw = set(raw) - RAW_PREDICTION_KEYS
        _require(
            not unexpected_raw,
            f"{sample_id}: unexpected raw inference fields {sorted(unexpected_raw)}",
        )
        image_hash = str(manifest["image_sha256"]).lower()
        frame_hash = str(manifest.get("frame_sha256") or image_hash).lower()
        _require(
            str(raw.get("image_sha256") or "").lower() == image_hash,
            f"{sample_id}: inference image hash drift",
        )
        # Deployment uses the entire supplied ROI; a second detector/crop would
        # necessarily produce a different content identity and is rejected.
        _require(
            str(raw.get("canonical_roi_sha256") or "").lower() == image_hash,
            f"{sample_id}: canonical ROI is not the supplied image",
        )
        controlled_reference: dict[str, Any] | None = None
        controlled_reference_hash: str | None = None
        if controlled_methods:
            controlled_reference, controlled_reference_hash = _reference_packet(
                raw.get("reference_input"), sample_id=sample_id
            )
            _require(
                str(raw.get("reference_input_sha256") or "").lower()
                == controlled_reference_hash,
                f"{sample_id}: top-level controlled reference hash drift",
            )
        else:
            _require(
                raw.get("reference_input") is None,
                f"{sample_id}: primary run contains an external reference packet",
            )
            _require(
                raw.get("reference_input_sha256") is None,
                f"{sample_id}: primary run contains an external reference hash",
            )
        raw_methods = raw.get("methods")
        _require(isinstance(raw_methods, Mapping), f"{sample_id}: missing methods")
        _require(
            set(raw_methods) == set(method_names),
            f"{sample_id}: method roster drift",
        )
        sealed_methods: dict[str, Any] = {}
        for method_name in method_names:
            record = raw_methods[method_name]
            _require(
                isinstance(record, Mapping),
                f"{sample_id}/{method_name}: invalid prediction record",
            )
            unexpected_record = set(record) - METHOD_PREDICTION_KEYS
            _require(
                not unexpected_record,
                f"{sample_id}/{method_name}: unexpected method output fields "
                f"{sorted(unexpected_record)}",
            )
            binding = method_bindings[method_name]
            expected_checkpoint = str(binding["checkpoint"]["sha256"])
            reference_mode = str(binding.get("reference_mode") or "")
            expected_reference_contract = str(
                binding.get("reference_contract_sha256") or ""
            )
            expected_evidence_role = str(binding.get("evidence_role") or "")
            expected_output_mode = str(binding.get("output_mode") or "")
            assertions = {
                "checkpoint_sha256": expected_checkpoint,
                "roi_contract_sha256": roi_contract_hash,
                "reference_contract_sha256": expected_reference_contract,
                "canonical_roi_sha256": image_hash,
            }
            reference_payload: dict[str, Any]
            if reference_mode == REFERENCE_MODE_AUTO:
                _require(
                    record.get("reference_mode") == REFERENCE_MODE_AUTO,
                    f"{sample_id}/{method_name}: primary method is not auto-reference",
                )
                auto_reference, auto_reference_hash = _primary_automatic_reference_packet(
                    record.get("auto_reference"),
                    sample_id=f"{sample_id}/{method_name}",
                )
                expected_detector = str(
                    ((binding.get("reference_detector") or {}).get("sha256")) or ""
                )
                _require(
                    str(record.get("reference_detector_sha256") or "").lower()
                    == expected_detector.lower(),
                    f"{sample_id}/{method_name}: automatic reference detector hash drift",
                )
                _require(
                    str(record.get("auto_reference_sha256") or "").lower()
                    == auto_reference_hash,
                    f"{sample_id}/{method_name}: automatic reference packet hash drift",
                )
                reference_payload = {
                    "reference_mode": REFERENCE_MODE_AUTO,
                    "reference_detector_sha256": expected_detector,
                    "auto_reference": auto_reference,
                    "auto_reference_sha256": auto_reference_hash,
                }
            elif reference_mode == REFERENCE_MODE_NATIVE:
                _require(
                    record.get("reference_mode") == REFERENCE_MODE_NATIVE,
                    f"{sample_id}/{method_name}: native progress mode drift",
                )
                forbidden_reference_fields = {
                    "reference_detector_sha256",
                    "auto_reference",
                    "auto_reference_sha256",
                    "reference_input_sha256",
                }.intersection(record)
                _require(
                    not forbidden_reference_fields,
                    f"{sample_id}/{method_name}: native progress method contains "
                    f"reference fields {sorted(forbidden_reference_fields)}",
                )
                reference_payload = {
                    "reference_mode": REFERENCE_MODE_NATIVE,
                    "reference_detector_invoked": False,
                    "external_reference_packet_consumed": False,
                }
            elif reference_mode == REFERENCE_MODE_CONTROLLED:
                _require(
                    record.get("reference_mode") == REFERENCE_MODE_CONTROLLED,
                    f"{sample_id}/{method_name}: controlled reference mode drift",
                )
                _require(
                    controlled_reference is not None
                    and controlled_reference_hash is not None,
                    f"{sample_id}/{method_name}: controlled reference absent",
                )
                _require(
                    str(record.get("reference_input_sha256") or "").lower()
                    == controlled_reference_hash,
                    f"{sample_id}/{method_name}: controlled reference hash drift",
                )
                reference_payload = {
                    "reference_mode": REFERENCE_MODE_CONTROLLED,
                    "reference_input_sha256": controlled_reference_hash,
                }
            else:
                raise ValueError(
                    f"{sample_id}/{method_name}: unsupported reference mode"
                )
            _require(
                record.get("evidence_role") == expected_evidence_role,
                f"{sample_id}/{method_name}: evidence role drift",
            )
            _require(
                record.get("output_mode") == expected_output_mode,
                f"{sample_id}/{method_name}: output mode drift",
            )
            for key, expected in assertions.items():
                actual = str(record.get(key) or "").lower()
                _require(
                    actual == expected.lower(),
                    f"{sample_id}/{method_name}: {key} drift",
                )
            status = record.get("status")
            _require(
                isinstance(status, bool),
                f"{sample_id}/{method_name}: status is not bool",
            )
            progress = _finite(record.get("prediction_progress"))
            predicted_scale_start = _finite(record.get("predicted_scale_start"))
            predicted_scale_end = _finite(record.get("predicted_scale_end"))
            range_confidence = _finite(record.get("range_confidence"))
            range_values = (
                predicted_scale_start,
                predicted_scale_end,
                range_confidence,
            )
            range_available = all(value is not None for value in range_values)
            _require(
                range_available or all(value is None for value in range_values),
                f"{sample_id}/{method_name}: partial numeric range output",
            )
            if range_available:
                _require(
                    predicted_scale_start != predicted_scale_end,
                    f"{sample_id}/{method_name}: predicted numeric range has zero span",
                )
                _require(
                    0.0 <= float(range_confidence) <= 1.0,
                    f"{sample_id}/{method_name}: invalid range confidence",
                )
            if expected_output_mode == OUTPUT_MODE_PROGRESS_ONLY:
                _require(
                    not range_available,
                    f"{sample_id}/{method_name}: progress-only diagnostic emitted a numeric range",
                )
            elif expected_output_mode == OUTPUT_MODE_FULL_READING:
                if status:
                    _require(
                        range_available,
                        f"{sample_id}/{method_name}: successful full reading lacks an automatic numeric range",
                    )
            else:
                raise ValueError(f"{sample_id}/{method_name}: unsupported output mode")
            if status:
                _require(
                    progress is not None and 0.0 <= progress <= 1.0,
                    f"{sample_id}/{method_name}: successful prediction is absent",
                )
                failure_code = None
                success_counts[method_name] += 1
            else:
                _require(
                    progress is None,
                    f"{sample_id}/{method_name}: failed row contains prediction",
                )
                failure_code = str(record.get("failure_code") or "")
                _require(
                    bool(failure_code),
                    f"{sample_id}/{method_name}: failure_code is absent",
                )
            telemetry = record.get("telemetry")
            if telemetry is not None:
                _require(
                    isinstance(telemetry, Mapping),
                    f"{sample_id}/{method_name}: telemetry is not an object",
                )
                assert_label_free(
                    telemetry,
                    location=f"raw_predictions.{sample_id}.{method_name}.telemetry",
                )
            sealed_methods[method_name] = {
                "status": status,
                "prediction_progress": progress,
                "failure_code": failure_code,
                **assertions,
                "evidence_role": expected_evidence_role,
                "output_mode": expected_output_mode,
                "predicted_scale_start": predicted_scale_start,
                "predicted_scale_end": predicted_scale_end,
                "range_confidence": range_confidence,
                **reference_payload,
                "telemetry": dict(telemetry) if telemetry is not None else None,
            }
        sealed_row = {
            "schema_version": 1,
            "protocol": INFERENCE_PROTOCOL,
            "sample_id": sample_id,
            "group_id": str(manifest["group_id"]),
            "image_sha256": image_hash,
            "frame_sha256": frame_hash,
            "methods": sealed_methods,
        }
        if controlled_reference is not None:
            sealed_row["reference_input"] = controlled_reference
            sealed_row["reference_input_sha256"] = controlled_reference_hash
        sealed_rows.append(sealed_row)

    sealed_payload = _jsonl_bytes(sealed_rows)
    sealed_output = sealed_predictions_path.resolve()
    _atomic_new(sealed_output, sealed_payload)
    seal = {
        "schema_version": 1,
        "protocol": SEAL_PROTOCOL,
        "status": "complete_label_free_predictions_sealed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_files_opened": 0,
        "ground_truth_fields_present": False,
        "evaluation_tier": evaluation_tier,
        "eligible_for_primary_metrics": (
            evaluation_tier == "primary_full_automatic_reading"
        ),
        "protocol_path": str(protocol_file),
        "protocol_sha256": sha256_file(protocol_file),
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file),
        "raw_predictions_path": str(raw_file),
        "raw_predictions_sha256": sha256_file(raw_file),
        "sealed_predictions_path": str(sealed_output),
        "sealed_predictions_sha256": hashlib.sha256(sealed_payload).hexdigest(),
        "rows": len(sealed_rows),
        "sample_ids_sha256": canonical_json_sha256(
            sorted(str(row["sample_id"]) for row in sealed_rows)
        ),
        "methods": {
            name: {
                "checkpoint_sha256": method_bindings[name]["checkpoint"]["sha256"],
                "reference_mode": method_bindings[name]["reference_mode"],
                "evidence_role": method_bindings[name]["evidence_role"],
                "output_mode": method_bindings[name]["output_mode"],
                "reference_contract_sha256": method_bindings[name][
                    "reference_contract_sha256"
                ],
                "reference_detector_sha256": (
                    (method_bindings[name].get("reference_detector") or {}).get(
                        "sha256"
                    )
                ),
                "successful": success_counts[name],
                "failures": len(sealed_rows) - success_counts[name],
            }
            for name in method_names
        },
        "roi_contract_sha256": roi_contract_hash,
        "primary_reference_contract_sha256": protocol[
            "primary_reference_contract_sha256"
        ],
        "controlled_reference_contract_sha256": protocol[
            "controlled_reference_contract_sha256"
        ],
        "native_progress_contract_sha256": protocol[
            "native_progress_contract_sha256"
        ],
    }
    _atomic_new(seal_path.resolve(), canonical_json_bytes(seal))
    return seal


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot compute percentile of empty values")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _micro_metrics(
    errors: Sequence[float],
    successes: Sequence[bool],
) -> dict[str, Any]:
    _require(len(errors) == len(successes) > 0, "empty/misaligned metric rows")
    return {
        "units": len(errors),
        "successful": sum(bool(value) for value in successes),
        "failures": sum(not bool(value) for value in successes),
        "coverage": statistics.fmean(1.0 if value else 0.0 for value in successes),
        "nmae": statistics.fmean(errors),
        "median_nae": _percentile(errors, 0.50),
        "p90_nae": _percentile(errors, 0.90),
        "p95_nae": _percentile(errors, 0.95),
        "acc_1pct": statistics.fmean(
            1.0 if ok and error <= 0.01 else 0.0
            for error, ok in zip(errors, successes)
        ),
        "acc_2pct": statistics.fmean(
            1.0 if ok and error <= 0.02 else 0.0
            for error, ok in zip(errors, successes)
        ),
        "acc_5pct": statistics.fmean(
            1.0 if ok and error <= 0.05 else 0.0
            for error, ok in zip(errors, successes)
        ),
        "failure_penalty": FAILURE_PENALTY,
    }


def _macro_metric(errors: Sequence[float], units: Sequence[str]) -> dict[str, Any]:
    _require(len(errors) == len(units) > 0, "empty/misaligned macro rows")
    by_unit: dict[str, list[float]] = defaultdict(list)
    for error, unit in zip(errors, units):
        by_unit[str(unit)].append(float(error))
    unit_errors = [statistics.fmean(values) for values in by_unit.values()]
    return {
        "units": len(unit_errors),
        "macro_nmae": statistics.fmean(unit_errors),
        "unit_min_nmae": min(unit_errors),
        "unit_max_nmae": max(unit_errors),
    }


def _gt_key(value: float) -> str:
    return format(float(value), ".12g")


def _method_score(
    scored_rows: Sequence[Mapping[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    errors = [float(row["methods"][method_name]["normalized_error"]) for row in scored_rows]
    successes = [bool(row["methods"][method_name]["status"]) for row in scored_rows]
    groups = [str(row["group_id"]) for row in scored_rows]
    frames = [str(row["frame_sha256"]) for row in scored_rows]
    gt_units = [_gt_key(float(row["ground_truth"])) for row in scored_rows]
    nonzero = [abs(float(row["ground_truth"])) > 1e-12 for row in scored_rows]

    frame_groups: dict[str, set[str]] = defaultdict(set)
    frame_gt: dict[str, set[tuple[float, float, float]]] = defaultdict(set)
    frame_success: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(scored_rows, successes):
        frame = str(row["frame_sha256"])
        frame_groups[frame].add(str(row["group_id"]))
        frame_gt[frame].add(
            (
                float(row["ground_truth"]),
                float(row["scale_start"]),
                float(row["scale_end"]),
            )
        )
        frame_success[frame].append(ok)
    for frame in frame_groups:
        _require(
            len(frame_groups[frame]) == 1,
            f"duplicate frame {frame} spans physical groups",
        )
        _require(
            len(frame_gt[frame]) == 1,
            f"duplicate frame {frame} has inconsistent labels/scales",
        )

    frame_macro = _macro_metric(errors, frames)
    frame_error_by_id: dict[str, list[float]] = defaultdict(list)
    for error, frame in zip(errors, frames):
        frame_error_by_id[frame].append(error)
    frame_errors = [statistics.fmean(frame_error_by_id[key]) for key in sorted(frame_error_by_id)]
    frame_successes = [all(frame_success[key]) for key in sorted(frame_success)]
    dedup_micro = _micro_metrics(frame_errors, frame_successes)
    dedup_micro["collapsed_duplicate_rows"] = len(errors) - len(frame_errors)

    nonzero_indices = [index for index, keep in enumerate(nonzero) if keep]
    nonzero_metrics = (
        _micro_metrics(
            [errors[index] for index in nonzero_indices],
            [successes[index] for index in nonzero_indices],
        )
        if nonzero_indices
        else {"units": 0, "status": "empty_subset"}
    )
    return {
        "image_micro": _micro_metrics(errors, successes),
        "physical_group": _macro_metric(errors, groups),
        "deduplicated_frame": {
            **dedup_micro,
            "macro_check": frame_macro["macro_nmae"],
        },
        "nonzero_physical_ground_truth": nonzero_metrics,
        "unique_ground_truth": {
            "unique_values": len(set(gt_units)),
            "macro_nmae": _macro_metric(errors, gt_units)["macro_nmae"],
            "rounding": "12 significant digits",
        },
    }


def _numeric_range_score(
    scored_rows: Sequence[Mapping[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    method_rows = [row["methods"][method_name] for row in scored_rows]
    available = [bool(row["range_available"]) for row in method_rows]
    covered = [row for row in method_rows if row["range_available"]]
    exact_all = [bool(row["range_pair_exact"]) for row in method_rows]
    return {
        "images": len(method_rows),
        "covered": sum(available),
        "coverage": statistics.fmean(1.0 if value else 0.0 for value in available),
        "ordered_pair_exact_rate_all_images": statistics.fmean(
            1.0 if value else 0.0 for value in exact_all
        ),
        "ordered_pair_exact_rate_covered": (
            statistics.fmean(1.0 if row["range_pair_exact"] else 0.0 for row in covered)
            if covered
            else None
        ),
        "start_mae_covered": (
            statistics.fmean(float(row["range_start_abs_error"]) for row in covered)
            if covered
            else None
        ),
        "end_mae_covered": (
            statistics.fmean(float(row["range_end_abs_error"]) for row in covered)
            if covered
            else None
        ),
        "mean_confidence_covered": (
            statistics.fmean(float(row["range_confidence"]) for row in covered)
            if covered
            else None
        ),
        "pair_order_matters": True,
        "absolute_tolerance": RANGE_PAIR_ABS_TOLERANCE,
        "relative_tolerance": RANGE_PAIR_REL_TOLERANCE,
    }


def score_sealed_inference(
    *,
    protocol_path: Path,
    sealed_predictions_path: Path,
    seal_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify the label-free seal, then join labels and compute frozen metrics."""

    protocol_file = protocol_path.resolve(strict=True)
    sealed_file = sealed_predictions_path.resolve(strict=True)
    seal_file = seal_path.resolve(strict=True)
    protocol = strict_json_load(protocol_file)
    seal = strict_json_load(seal_file)
    _require(protocol.get("protocol") == PROTOCOL, "scoring protocol is invalid")
    _require(seal.get("protocol") == SEAL_PROTOCOL, "inference seal is invalid")
    _require(
        seal.get("status") == "complete_label_free_predictions_sealed",
        "inference was not completely sealed",
    )
    _require(seal.get("label_files_opened") == 0, "seal reports label access")
    _require(
        seal.get("ground_truth_fields_present") is False,
        "seal reports label-bearing predictions",
    )
    _require(
        sha256_file(protocol_file) == seal.get("protocol_sha256"),
        "protocol changed after inference",
    )
    _require(
        sha256_file(sealed_file) == seal.get("sealed_predictions_sha256"),
        "sealed prediction hash mismatch",
    )

    predictions = strict_jsonl_load(sealed_file)
    assert_label_free(predictions, location="sealed_predictions")
    evaluation_tier = str((protocol.get("scope") or {}).get("evaluation_tier") or "")
    _require(
        evaluation_tier
        in {
            "primary_full_automatic_reading",
            "progress_component_diagnostic",
            "controlled_reference_diagnostic",
        },
        "frozen evaluation tier is invalid",
    )
    _require(
        seal.get("evaluation_tier") == evaluation_tier
        and seal.get("eligible_for_primary_metrics")
        is (evaluation_tier == "primary_full_automatic_reading"),
        "inference seal evaluation tier drift",
    )
    # The label path is not even resolved until the prediction bytes and their
    # label-free seal have been authenticated above.
    label_file = labels_path.resolve(strict=True)
    labels = strict_jsonl_load(label_file)
    prediction_by_id = _index_rows(predictions, label="sealed predictions")
    label_by_id = _index_rows(labels, label="labels")
    _require(set(prediction_by_id) == set(label_by_id), "label ID roster drift")
    _require(len(predictions) == int(seal.get("rows", -1)), "seal row count drift")
    _require(
        canonical_json_sha256(sorted(prediction_by_id))
        == seal.get("sample_ids_sha256"),
        "sealed sample-ID hash drift",
    )

    method_bindings = protocol.get("methods") or {}
    method_names = tuple(sorted(method_bindings))
    primary_method_names = tuple(
        name
        for name in method_names
        if method_bindings[name].get("evidence_role") == EVIDENCE_ROLE_PRIMARY
    )
    diagnostic_method_names = tuple(
        name
        for name in method_names
        if method_bindings[name].get("evidence_role") == EVIDENCE_ROLE_DIAGNOSTIC
    )
    component_method_names = tuple(
        name
        for name in method_names
        if method_bindings[name].get("evidence_role") == EVIDENCE_ROLE_COMPONENT
    )
    _require(
        set(primary_method_names).isdisjoint(diagnostic_method_names)
        and set(primary_method_names).isdisjoint(component_method_names)
        and set(diagnostic_method_names).isdisjoint(component_method_names)
        and len(primary_method_names)
        + len(diagnostic_method_names)
        + len(component_method_names)
        == len(method_names),
        "method evidence-role roster drift",
    )
    _require(
        bool(primary_method_names)
        == (evaluation_tier == "primary_full_automatic_reading")
        and bool(component_method_names)
        == (evaluation_tier == "progress_component_diagnostic")
        and bool(diagnostic_method_names)
        == (evaluation_tier == "controlled_reference_diagnostic"),
        "primary methods disagree with evaluation tier",
    )
    scored_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        label = label_by_id[sample_id]
        if label.get("group_id") is not None:
            _require(
                str(label.get("group_id")) == str(prediction["group_id"]),
                f"{sample_id}: label physical group drift",
            )
        if label.get("image_sha256") is not None:
            _require(
                str(label.get("image_sha256")).lower()
                == str(prediction["image_sha256"]).lower(),
                f"{sample_id}: label image hash drift",
            )
        ground_truth = _finite(label.get("ground_truth"))
        scale_start = _finite(label.get("scale_start"))
        scale_end = _finite(label.get("scale_end"))
        _require(ground_truth is not None, f"{sample_id}: invalid ground_truth")
        _require(scale_start is not None, f"{sample_id}: invalid scale_start")
        _require(scale_end is not None, f"{sample_id}: invalid scale_end")
        span = scale_end - scale_start
        _require(abs(span) > 1e-12, f"{sample_id}: zero scale span")
        target_progress = (ground_truth - scale_start) / span
        method_scores: dict[str, Any] = {}
        for method_name in method_names:
            record = prediction["methods"][method_name]
            success = bool(record["status"])
            progress = _finite(record.get("prediction_progress")) if success else None
            _require(
                (progress is not None) == success,
                f"{sample_id}/{method_name}: sealed status/prediction mismatch",
            )
            progress_error = (
                abs(float(progress) - target_progress)
                if progress is not None
                else FAILURE_PENALTY
            )
            predicted_scale_start = _finite(record.get("predicted_scale_start"))
            predicted_scale_end = _finite(record.get("predicted_scale_end"))
            range_confidence = _finite(record.get("range_confidence"))
            range_available = (
                predicted_scale_start is not None
                and predicted_scale_end is not None
                and range_confidence is not None
            )
            range_pair_exact = bool(
                range_available
                and math.isclose(
                    float(predicted_scale_start),
                    scale_start,
                    rel_tol=RANGE_PAIR_REL_TOLERANCE,
                    abs_tol=RANGE_PAIR_ABS_TOLERANCE,
                )
                and math.isclose(
                    float(predicted_scale_end),
                    scale_end,
                    rel_tol=RANGE_PAIR_REL_TOLERANCE,
                    abs_tol=RANGE_PAIR_ABS_TOLERANCE,
                )
            )
            predicted_physical = (
                float(predicted_scale_start)
                + float(progress)
                * (float(predicted_scale_end) - float(predicted_scale_start))
                if success and range_available and progress is not None
                else None
            )
            full_e2e_error = (
                abs(predicted_physical - ground_truth) / abs(span)
                if predicted_physical is not None
                else FAILURE_PENALTY
            )
            output_mode = str(record.get("output_mode") or "")
            if output_mode == OUTPUT_MODE_FULL_READING:
                normalized_error = full_e2e_error
                metric_space = "full_automatic_physical_reading"
            elif output_mode == OUTPUT_MODE_PROGRESS_ONLY:
                normalized_error = progress_error
                metric_space = "progress_component_diagnostic"
            else:
                raise ValueError(f"{sample_id}/{method_name}: sealed output mode drift")
            method_scores[method_name] = {
                "status": success,
                "prediction_progress": progress,
                "progress_component_normalized_error": float(progress_error),
                "predicted_scale_start": predicted_scale_start,
                "predicted_scale_end": predicted_scale_end,
                "range_confidence": range_confidence,
                "range_available": range_available,
                "range_pair_exact": range_pair_exact,
                "range_start_abs_error": (
                    abs(float(predicted_scale_start) - scale_start)
                    if range_available
                    else None
                ),
                "range_end_abs_error": (
                    abs(float(predicted_scale_end) - scale_end)
                    if range_available
                    else None
                ),
                "predicted_physical": predicted_physical,
                "full_e2e_normalized_error": float(full_e2e_error),
                "normalized_error": float(normalized_error),
                "metric_space": metric_space,
                "failure_code": record.get("failure_code"),
            }
        null_progress = (0.0 - scale_start) / span
        scored_rows.append(
            {
                "sample_id": sample_id,
                "group_id": str(prediction["group_id"]),
                "image_sha256": str(prediction["image_sha256"]),
                "frame_sha256": str(prediction["frame_sha256"]),
                "ground_truth": ground_truth,
                "scale_start": scale_start,
                "scale_end": scale_end,
                "target_progress": target_progress,
                "null_zero_reading_normalized_error": abs(
                    null_progress - target_progress
                ),
                "methods": method_scores,
            }
        )

    all_method_scores = {
        method_name: {
            "evidence_role": method_bindings[method_name]["evidence_role"],
            "output_mode": method_bindings[method_name]["output_mode"],
            "eligible_for_primary_metrics": (
                method_bindings[method_name]["evidence_role"]
                == EVIDENCE_ROLE_PRIMARY
            ),
            "metric_space": (
                "full_automatic_physical_reading"
                if method_bindings[method_name]["output_mode"]
                == OUTPUT_MODE_FULL_READING
                else "progress_component_diagnostic"
            ),
            "numeric_range": _numeric_range_score(scored_rows, method_name),
            **_method_score(scored_rows, method_name),
        }
        for method_name in method_names
    }
    methods = {name: all_method_scores[name] for name in primary_method_names}
    diagnostics = {
        name: all_method_scores[name]
        for name in (*component_method_names, *diagnostic_method_names)
    }
    null_rows = []
    for row in scored_rows:
        copied = dict(row)
        copied["methods"] = {
            "zero_reading_null": {
                "status": True,
                "normalized_error": row["null_zero_reading_normalized_error"],
            }
        }
        null_rows.append(copied)
    null_baseline = _method_score(null_rows, "zero_reading_null")

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "image_scores.jsonl"
    summary_path = output / "summary.json"
    _atomic_new(rows_path, _jsonl_bytes(scored_rows))
    summary = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "complete",
        "evaluation_tier": evaluation_tier,
        "eligible_for_primary_metrics": (
            evaluation_tier == "primary_full_automatic_reading"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "two_stage_integrity": {
            "inference_label_free": True,
            "predictions_sealed_before_label_join": True,
            "protocol_sha256": sha256_file(protocol_file),
            "seal_sha256": sha256_file(seal_file),
            "sealed_predictions_sha256": sha256_file(sealed_file),
            "labels_sha256": sha256_file(label_file),
        },
        "cohort": {
            "images": len(scored_rows),
            "physical_groups": len({row["group_id"] for row in scored_rows}),
            "unique_frames": len({row["frame_sha256"] for row in scored_rows}),
            "unique_ground_truth_values": len(
                {_gt_key(float(row["ground_truth"])) for row in scored_rows}
            ),
            "nonzero_ground_truth_images": sum(
                abs(float(row["ground_truth"])) > 1e-12 for row in scored_rows
            ),
        },
        "contracts": {
            "failure_penalty": FAILURE_PENALTY,
            "range_pair_abs_tolerance": RANGE_PAIR_ABS_TOLERANCE,
            "range_pair_rel_tolerance": RANGE_PAIR_REL_TOLERANCE,
            "roi_contract_sha256": protocol["roi_contract_sha256"],
            "primary_reference_contract_sha256": protocol[
                "primary_reference_contract_sha256"
            ],
            "controlled_reference_contract_sha256": protocol[
                "controlled_reference_contract_sha256"
            ],
            "method_reference_bindings": {
                name: {
                    "reference_mode": protocol["methods"][name][
                        "reference_mode"
                    ],
                    "evidence_role": protocol["methods"][name]["evidence_role"],
                    "output_mode": protocol["methods"][name]["output_mode"],
                    "reference_contract_sha256": protocol["methods"][name][
                        "reference_contract_sha256"
                    ],
                    "reference_detector_sha256": (
                        (
                            protocol["methods"][name].get(
                                "reference_detector"
                            )
                            or {}
                        ).get("sha256")
                    ),
                }
                for name in method_names
            },
            "method_checkpoints": {
                name: protocol["methods"][name]["checkpoint"]["sha256"]
                for name in method_names
            },
        },
        "methods": methods,
        "diagnostics": diagnostics,
        "interpretation_boundary": {
            "gt_or_manually_supplied_reference_is_diagnostic_only": True,
            "diagnostic_methods_in_primary_metrics": False,
            "progress_only_is_complete_automatic_reading": False,
            "numeric_scale_range_is_model_output_not_inference_input": True,
            "predicted_physical_equation": (
                "predicted_scale_start + prediction_progress * "
                "(predicted_scale_end-predicted_scale_start)"
            ),
            "primary_method_names": list(primary_method_names),
            "progress_component_method_names": list(component_method_names),
            "controlled_reference_method_names": list(diagnostic_method_names),
        },
        "null_baseline": {
            "definition": "constant physical reading 0",
            **null_baseline,
        },
        "image_scores": {
            "path": str(rows_path),
            "sha256": sha256_file(rows_path),
        },
    }
    _atomic_new(summary_path, canonical_json_bytes(summary))
    return summary


def audit_legacy_pepd_comparability(*, output_path: Path | None = None) -> dict[str, Any]:
    """Audit aggregate summaries and evaluator code; never open rows or images."""

    strict_path = PROJECT_ROOT / "artifacts/runs/strict_common_holdout_v2/summary.json"
    source_oof_path = (
        PROJECT_ROOT
        / "artifacts/runs/fadr_multiseed_v2_inputs_authoritative_v2/"
        "pepd_mixed_authoritative_oof.summary.json"
    )
    field_root = (
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026/confirmatory_one_shot_v1"
    )
    field_path = field_root / "summary.json"
    field_pepd_path = field_root / "pepd.summary.json"
    strict = strict_json_load(strict_path)
    source_oof = strict_json_load(source_oof_path)
    field = strict_json_load(field_path)
    field_pepd = strict_json_load(field_pepd_path)
    _require(
        strict.get("protocol") == "strict_current_experiment_common_holdout_v2",
        "unexpected strict source summary",
    )
    _require(
        field.get("protocol") == "field_confirmatory_one_shot_summary_v1",
        "unexpected historical field summary",
    )
    source_metric = (strict.get("metrics") or {}).get("pepd") or {}
    field_metric = (field.get("methods") or {}).get("pepd_vector") or {}
    direction_runs = ((source_oof.get("signature") or {}).get("direction_runs") or {})
    source_run = direction_runs.get("20260720") or {}
    field_signature = field_pepd.get("signature") or {}
    source_checkpoint = str(source_run.get("checkpoint_sha256") or "")
    field_checkpoint = str(field_signature.get("checkpoint_sha256") or "")
    _require(_is_sha256(source_checkpoint), "source checkpoint hash is missing")
    _require(_is_sha256(field_checkpoint), "field checkpoint hash is missing")
    source_samples = int(source_metric.get("samples"))
    source_groups = int(source_metric.get("groups"))
    field_identity = field.get("identity") or {}
    field_samples = int(field_identity.get("samples"))
    field_groups = int(field_identity.get("physical_groups"))
    source_nmae = float(source_metric.get("nmae"))
    field_nmae = float(field_metric.get("full_denominator_nmae"))
    source_coverage = float(source_metric.get("coverage"))
    field_coverage = float(field_metric.get("coverage"))
    source_failure = float(source_metric.get("nmae_failure_penalty"))
    field_failure = float((field.get("predeclared_statistics") or {}).get("failure_penalty_nmae"))

    report = {
        "schema_version": 1,
        "protocol": "legacy_pepd_source_field_comparability_audit_v1",
        "status": "complete_aggregate_only",
        "scope": {
            "images_opened": 0,
            "label_rows_opened": 0,
            "prediction_rows_opened": 0,
            "aggregate_summaries_opened": 4,
            "inference_performed": False,
        },
        "source_metric": {
            "cohort": "SyncG/train strict common holdout",
            "samples": source_samples,
            "physical_groups": source_groups,
            "nmae": source_nmae,
            "macro_group_nmae": float(source_metric.get("macro_group_nmae")),
            "coverage": source_coverage,
            "checkpoint_seed": 20260720,
            "checkpoint_sha256": source_checkpoint,
            "evidence_role": "post-hoc shared-component source sensitivity analysis",
            "front_end_assignment": (
                (source_oof.get("signature") or {}).get("assignment_policy")
            ),
        },
        "field_metric": {
            "cohort": "historical field confirmatory split",
            "samples": field_samples,
            "physical_groups": field_groups,
            "nmae": field_nmae,
            "macro_group_nmae": float(field_metric.get("macro_physical_group_nmae")),
            "coverage": field_coverage,
            "checkpoint_seed": 20260722,
            "checkpoint_sha256": field_checkpoint,
            "evidence_role": "historical external one-shot result",
            "front_end_policy": field_signature.get("front_end_policy"),
            "crop_policy": field_signature.get("crop_policy"),
            "reference_predictions_sha256": field_signature.get(
                "reference_predictions_sha256"
            ),
        },
        "matched_conditions": {
            "normalized_error_definition": True,
            "failure_penalty_1": source_failure == field_failure == 1.0,
            "coverage_both_near_complete": min(source_coverage, field_coverage) > 0.99,
        },
        "unmatched_conditions": {
            "same_checkpoint": source_checkpoint == field_checkpoint,
            "same_images": False,
            "same_physical_groups": False,
            "same_data_domain": False,
            "same_front_end_cache_or_reference_identity": False,
            "same_evidence_role": False,
        },
        "comparison": {
            "raw_source_to_field_nmae_ratio": source_nmae / field_nmae,
            "valid_matched_domain_transfer_estimate": False,
            "exact_reason": [
                "different PEPD checkpoint (seed 20260720 versus seed 20260722)",
                "different cohorts and physical groups (SyncG source versus field)",
                "different frozen front-end/reference artifacts",
                "source result is post-hoc shared-component validation; field result is an external historical run",
                "old evaluator writes label-bearing prediction rows and computes GT-derived direction diagnostics, so it lacks the new infer-before-score separation",
            ],
            "interpretation": (
                "0.1388247566 and 0.0257833017 are internally reproduced aggregate "
                "metrics, but their difference cannot establish source-to-field "
                "improvement, degradation, or leakage.  A same-checkpoint two-stage "
                "retest is required."
            ),
        },
        "sources": {
            "strict_summary": {"path": str(strict_path), "sha256": sha256_file(strict_path)},
            "source_oof_summary": {"path": str(source_oof_path), "sha256": sha256_file(source_oof_path)},
            "field_summary": {"path": str(field_path), "sha256": sha256_file(field_path)},
            "field_pepd_summary": {"path": str(field_pepd_path), "sha256": sha256_file(field_pepd_path)},
            "old_field_evaluator": {
                "path": str(PROJECT_ROOT / "experiments/evaluate_probabilistic_pivot_direction.py"),
                "sha256": sha256_file(PROJECT_ROOT / "experiments/evaluate_probabilistic_pivot_direction.py"),
            },
        },
    }
    if output_path is not None:
        _atomic_new(output_path.resolve(), canonical_json_bytes(report))
    return report


def factorial_error_attribution(
    *,
    predicted_pointer_runtime_reference: float,
    gt_pointer_runtime_reference: float,
    predicted_pointer_gt_reference: float,
    gt_pointer_gt_reference: float,
) -> dict[str, Any]:
    """Shapley attribution for the 2x2 pointer/reference oracle table.

    The two contributions sum exactly to the removable error above the
    annotation/solver floor.  This is preferable to assigning the interaction
    term arbitrarily to one branch.
    """

    e00 = float(predicted_pointer_runtime_reference)
    e10 = float(gt_pointer_runtime_reference)
    e01 = float(predicted_pointer_gt_reference)
    e11 = float(gt_pointer_gt_reference)
    _require(
        all(math.isfinite(value) for value in (e00, e10, e01, e11)),
        "factorial errors must be finite",
    )
    pointer = 0.5 * ((e00 - e10) + (e01 - e11))
    reference = 0.5 * ((e00 - e01) + (e10 - e11))
    removable = e00 - e11
    _require(
        math.isclose(pointer + reference, removable, rel_tol=0.0, abs_tol=1e-12),
        "factorial conservation failed",
    )
    positive_total = max(pointer, 0.0) + max(reference, 0.0)
    return {
        "pointer_direction_contribution_nmae": pointer,
        "production_reference_contribution_nmae": reference,
        "annotation_solver_floor_nmae": e11,
        "removable_nmae_above_floor": removable,
        "pointer_share_of_positive_attribution": (
            max(pointer, 0.0) / positive_total if positive_total > 0.0 else None
        ),
        "reference_share_of_positive_attribution": (
            max(reference, 0.0) / positive_total if positive_total > 0.0 else None
        ),
        "larger_branch": (
            "production_reference" if reference > pointer else "pointer_direction"
        ),
        "note": "factorial interaction is split equally between the two branches",
    }


def _gt_pointer_and_reference(
    metadata: Mapping[str, Any],
    *,
    sample_id: str,
) -> tuple[float, float, float]:
    """Return GT pointer/start/range angles from ordered SyncG annotations."""

    from experiments.vdn_baseline import image_angle_from_direction

    pointer: Mapping[str, Any] | None = None
    scale_mark: Mapping[str, Any] | None = None
    for item in metadata.get("keypoints") or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "").casefold()
        if kind == "pointer":
            pointer = item
        elif kind == "scalemark":
            scale_mark = item
    _require(pointer is not None, f"{sample_id}: GT Pointer annotation absent")
    _require(scale_mark is not None, f"{sample_id}: GT ScaleMark annotation absent")
    tip = pointer.get("outside_kp")
    tail = pointer.get("origin_kp")
    marks = scale_mark.get("all_kp")
    _require(
        isinstance(tip, Sequence)
        and not isinstance(tip, (str, bytes))
        and len(tip) >= 2,
        f"{sample_id}: malformed pointer tip",
    )
    _require(
        isinstance(tail, Sequence)
        and not isinstance(tail, (str, bytes))
        and len(tail) >= 2,
        f"{sample_id}: malformed pointer tail",
    )
    _require(
        isinstance(marks, Sequence)
        and not isinstance(marks, (str, bytes))
        and len(marks) >= 2,
        f"{sample_id}: malformed ordered ScaleMark points",
    )
    start = marks[0]
    end = marks[-1]
    _require(
        isinstance(start, Sequence)
        and isinstance(end, Sequence)
        and len(start) >= 2
        and len(end) >= 2,
        f"{sample_id}: malformed ScaleMark endpoints",
    )
    numeric = [
        float(tip[0]),
        float(tip[1]),
        float(tail[0]),
        float(tail[1]),
        float(start[0]),
        float(start[1]),
        float(end[0]),
        float(end[1]),
    ]
    _require(all(math.isfinite(value) for value in numeric), f"{sample_id}: nonfinite GT point")
    tip_x, tip_y, tail_x, tail_y, start_x, start_y, end_x, end_y = numeric
    pointer_angle = image_angle_from_direction((tip_x - tail_x, tip_y - tail_y))
    start_angle = image_angle_from_direction((start_x - tail_x, start_y - tail_y))
    end_angle = image_angle_from_direction((end_x - tail_x, end_y - tail_y))
    range_angle = (end_angle - start_angle) % 360.0
    _require(1e-8 < range_angle < 360.0, f"{sample_id}: invalid GT reference arc")
    return float(pointer_angle), float(start_angle), float(range_angle)


def _factor_metrics(
    errors: Sequence[float],
    successes: Sequence[bool],
    groups: Sequence[str],
) -> dict[str, Any]:
    result = _micro_metrics(errors, successes)
    result["covered_nmae"] = (
        statistics.fmean(
            error for error, success in zip(errors, successes) if success
        )
        if any(successes)
        else None
    )
    result["covered_p95_nae"] = (
        _percentile(
            [error for error, success in zip(errors, successes) if success],
            0.95,
        )
        if any(successes)
        else None
    )
    result["macro_group_nmae"] = _macro_metric(errors, groups)["macro_nmae"]
    return result


def audit_strict_common_pepd_factorization(
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Factor the public 1,625-row PEPD error into pointer/reference branches.

    Only existing SyncG/train JSONL artifacts are read.  No image, field row,
    sealed row, or confirmatory row is opened and no model is executed.
    """

    from experiments.vdn_baseline import reading_from_pointer_angle

    strict_summary_path = (
        PROJECT_ROOT / "artifacts/runs/strict_common_holdout_v2/summary.json"
    )
    oof_path = (
        PROJECT_ROOT
        / "artifacts/runs/fadr_multiseed_v2_inputs_authoritative_v2/"
        "pepd_mixed_authoritative_oof.jsonl"
    )
    oof_summary_path = oof_path.with_suffix(".summary.json")
    manifest_path = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
    strict_summary = strict_json_load(strict_summary_path)
    oof_summary = strict_json_load(oof_summary_path)
    signature = oof_summary.get("signature") or {}
    _require(
        sha256_file(manifest_path) == signature.get("manifest_sha256"),
        "public SyncG manifest hash differs from the PEPD OOF binding",
    )
    _require(
        sha256_file(oof_path) == oof_summary.get("output_sha256"),
        "public PEPD OOF hash drift",
    )
    all_oof = strict_jsonl_load(oof_path)
    rows = [row for row in all_oof if int(row.get("held_out_seed", -1)) == 20260720]
    _require(len(rows) == 1625, "strict-common PEPD row count drift")
    _require(
        len({str(row.get("group_id")) for row in rows}) == 73,
        "strict-common PEPD group count drift",
    )
    selected_ids = {str(row["sample_id"]) for row in rows}
    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
            sample_id = str(raw.get("sample_id") or "")
            if sample_id not in selected_ids:
                continue
            metadata = raw.get("metadata")
            _require(
                isinstance(metadata, Mapping),
                f"{manifest_path}:{line_number}: metadata absent",
            )
            metadata_by_id[sample_id] = metadata
    _require(set(metadata_by_id) == selected_ids, "public metadata roster drift")

    method_names = (
        "predicted_pointer_runtime_reference",
        "gt_pointer_runtime_reference",
        "predicted_pointer_gt_reference",
        "gt_pointer_gt_reference",
    )
    errors: dict[str, list[float]] = {name: [] for name in method_names}
    successes: dict[str, list[bool]] = {name: [] for name in method_names}
    groups: list[str] = []
    historical_errors: list[float] = []
    historical_success: list[bool] = []
    direction_errors: list[float] = []
    progress_replay_deltas: list[float] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        group_id = str(row["group_id"])
        groups.append(group_id)
        ground_truth = float(row["ground_truth"])
        scale_start = float(row["scale_start"])
        scale_end = float(row["scale_end"])
        span = scale_end - scale_start
        _require(abs(span) > 1e-12, f"{sample_id}: zero scale span")
        target_progress = (ground_truth - scale_start) / span
        vector = row.get("vector") or {}
        front_end = row.get("front_end") or {}
        vector_prediction = _finite(vector.get("prediction"))
        vector_progress = _finite(vector.get("progress"))
        predicted_pointer = _finite(vector.get("pointer_angle"))
        runtime_start = _finite(front_end.get("start_angle"))
        runtime_range = _finite(front_end.get("range_angle"))
        historical_ok = vector.get("status") is True and vector_prediction is not None
        historical_error = (
            abs(vector_prediction - ground_truth) / abs(span)
            if historical_ok
            else FAILURE_PENALTY
        )
        historical_errors.append(historical_error)
        historical_success.append(historical_ok)

        gt_pointer, gt_start, gt_range = _gt_pointer_and_reference(
            metadata_by_id[sample_id], sample_id=sample_id
        )
        if predicted_pointer is not None:
            delta = abs((predicted_pointer - gt_pointer + 180.0) % 360.0 - 180.0)
            direction_errors.append(delta)

        combinations = {
            "predicted_pointer_runtime_reference": (
                predicted_pointer,
                runtime_start,
                runtime_range,
            ),
            "gt_pointer_runtime_reference": (
                gt_pointer,
                runtime_start,
                runtime_range,
            ),
            "predicted_pointer_gt_reference": (
                predicted_pointer,
                gt_start,
                gt_range,
            ),
            "gt_pointer_gt_reference": (gt_pointer, gt_start, gt_range),
        }
        for name, (pointer, start, angle_range) in combinations.items():
            ok = (
                pointer is not None
                and start is not None
                and angle_range is not None
                and abs(angle_range) > 1e-12
            )
            progress: float | None = None
            if ok:
                try:
                    _, progress = reading_from_pointer_angle(
                        pointer,
                        start_angle=start,
                        range_angle=angle_range,
                        scale_start=scale_start,
                        scale_end=scale_end,
                    )
                except ValueError:
                    ok = False
            successes[name].append(ok)
            errors[name].append(
                abs(float(progress) - target_progress)
                if ok and progress is not None
                else FAILURE_PENALTY
            )
            if (
                name == "predicted_pointer_runtime_reference"
                and ok
                and progress is not None
                and vector_progress is not None
            ):
                progress_replay_deltas.append(abs(progress - vector_progress))

    metrics = {
        name: _factor_metrics(errors[name], successes[name], groups)
        for name in method_names
    }
    historical = _factor_metrics(historical_errors, historical_success, groups)
    strict_metric = (strict_summary.get("metrics") or {}).get("pepd") or {}
    _require(
        math.isclose(
            historical["nmae"],
            float(strict_metric.get("nmae")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "historical PEPD NMAE did not reproduce strict-common summary",
    )
    _require(
        max(progress_replay_deltas, default=0.0) <= 1e-12,
        "pointer/reference replay differs from stored PEPD progress",
    )
    attribution = factorial_error_attribution(
        predicted_pointer_runtime_reference=metrics[
            "predicted_pointer_runtime_reference"
        ]["nmae"],
        gt_pointer_runtime_reference=metrics["gt_pointer_runtime_reference"]["nmae"],
        predicted_pointer_gt_reference=metrics[
            "predicted_pointer_gt_reference"
        ]["nmae"],
        gt_pointer_gt_reference=metrics["gt_pointer_gt_reference"]["nmae"],
    )
    report = {
        "schema_version": 1,
        "protocol": "strict_common_pepd_pointer_reference_factorization_v1",
        "status": "complete_public_artifact_only",
        "evidence_role": "oracle_diagnostic_only",
        "eligible_for_primary_metrics": False,
        "complete_automatic_reading_evidence": False,
        "scope": {
            "dataset": "SyncG/train",
            "samples": len(rows),
            "physical_groups": len(set(groups)),
            "held_out_seed": 20260720,
            "images_opened": 0,
            "models_executed": 0,
            "field_rows_opened": 0,
            "field_labels_opened": 0,
        },
        "historical_pepd": historical,
        "factorial_cells": metrics,
        "direction_diagnostic": {
            "samples": len(direction_errors),
            "mean_absolute_angular_error_degrees": statistics.fmean(direction_errors),
            "p95_absolute_angular_error_degrees": _percentile(direction_errors, 0.95),
        },
        "factorial_attribution": attribution,
        "replay": {
            "stored_progress_max_absolute_delta": max(
                progress_replay_deltas, default=0.0
            ),
            "historical_summary_nmae_reproduced": True,
            "failure_penalty": FAILURE_PENALTY,
        },
        "interpretation_boundary": [
            "GT ScaleMark is an oracle diagnostic and is never a deployable input.",
            "GT pointer is an oracle diagnostic and is never a deployable input.",
            "The 2x2 Shapley attribution splits pointer/reference interaction equally; it is diagnostic, not a causal training effect.",
        ],
        "sources": {
            "strict_summary": {
                "path": str(strict_summary_path),
                "sha256": sha256_file(strict_summary_path),
            },
            "pepd_oof": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
            "pepd_oof_summary": {
                "path": str(oof_summary_path),
                "sha256": sha256_file(oof_summary_path),
            },
            "syncg_train_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
    }
    if output_path is not None:
        _atomic_new(output_path.resolve(), canonical_json_bytes(report))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--unlabeled-manifest", type=Path, required=True)
    freeze.add_argument("--method-bundle", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--protocol", type=Path, required=True)
    infer.add_argument("--unlabeled-manifest", type=Path, required=True)
    infer.add_argument("--raw-predictions", type=Path, required=True)
    infer.add_argument("--sealed-predictions", type=Path, required=True)
    infer.add_argument("--seal", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--sealed-predictions", type=Path, required=True)
    score.add_argument("--seal", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser("audit-legacy")
    audit.add_argument("--output", type=Path)
    factor = commands.add_parser("audit-source-factorization")
    factor.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "freeze":
        result = freeze_protocol(
            unlabeled_manifest_path=args.unlabeled_manifest,
            method_bundle_path=args.method_bundle,
            output_path=args.output,
        )
    elif args.command == "infer":
        result = seal_inference(
            protocol_path=args.protocol,
            unlabeled_manifest_path=args.unlabeled_manifest,
            raw_predictions_path=args.raw_predictions,
            sealed_predictions_path=args.sealed_predictions,
            seal_path=args.seal,
        )
    elif args.command == "score":
        result = score_sealed_inference(
            protocol_path=args.protocol,
            sealed_predictions_path=args.sealed_predictions,
            seal_path=args.seal,
            labels_path=args.labels,
            output_dir=args.output_dir,
        )
    elif args.command == "audit-legacy":
        result = audit_legacy_pepd_comparability(output_path=args.output)
    else:
        result = audit_strict_common_pepd_factorization(output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
