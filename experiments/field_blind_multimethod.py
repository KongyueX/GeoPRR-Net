"""One-shot, full-scene, image-only multi-method field-blind protocol.

This module deliberately has no default field path.  ``freeze`` authenticates
only owner/public metadata and frozen executable artifacts.  ``run-once``
creates its irreversible claim before opening the label-free full-scene
manifest, applies one shared frozen meter detector exactly once per image, and
feeds independent copies of that single ROI to every frozen full-reading
adapter.  All method predictions are sealed together before ``score-once`` may
open the separately frozen labels.

The five required method roles are intentionally not presented as equivalent
claims: GARC is the final end-to-end method; V5 is an end-to-end internal
ablation; PEPD and VDN are end-to-end progress-backbone controls using exactly
the same automatic numeric-range binding as GARC; Transformer is a sensitivity
analysis using that same range binding.  A progress-only adapter cannot enter
this protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.garc_field_blind_guard import (
    DATASET_IDENTITY_PROTOCOL,
    DEFAULT_PAPER_RESULTS_SUMMARY,
    DEFAULT_PAPER_TABLES_ROOT,
    MINIMUM_BLIND_IMAGES,
    PAPER_RESULTS_PROTOCOL,
    _load_paper_results,
    load_paper_authority,
)
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    EVIDENCE_ROLE_PRIMARY,
    OUTPUT_MODE_FULL_READING,
    FullAutoPrediction,
    FrozenFullAutoBundle,
    UnifiedFullAutoAdapter,
    _load_factory,
    _progress_image_sha256,
    sha256_file,
    validate_canonical_roi,
)
from experiments.v5_unified_two_stage_retest import (
    FAILURE_PENALTY,
    assert_label_free,
    canonical_json_bytes,
    canonical_json_sha256,
    strict_json_load,
    strict_jsonl_load,
)


PROTOCOL: Final[str] = "field_blind_full_scene_multimethod_v1"
FRONTEND_PROTOCOL: Final[str] = "field_blind_shared_meter_frontend_v1"
ROSTER_PROTOCOL: Final[str] = "field_blind_full_reading_method_roster_v1"
INFERENCE_CLAIM_PROTOCOL: Final[str] = "field_blind_multimethod_inference_claim_v1"
INFERENCE_COMPLETE_PROTOCOL: Final[str] = "field_blind_multimethod_inference_complete_v1"
PREDICTION_PROTOCOL: Final[str] = "field_blind_multimethod_prediction_v1"
PREDICTION_SEAL_PROTOCOL: Final[str] = "field_blind_multimethod_prediction_seal_v1"
SCORE_CLAIM_PROTOCOL: Final[str] = "field_blind_multimethod_score_claim_v1"
SCORE_PROTOCOL: Final[str] = "field_blind_multimethod_score_v1"

FULL_SCENE_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {"sample_id", "group_id", "image_path", "image_sha256", "frame_sha256"}
)
LABEL_KEYS: Final[frozenset[str]] = frozenset(
    {"sample_id", "group_id", "ground_truth", "scale_start", "scale_end"}
)
METHOD_ROLES: Final[tuple[str, ...]] = (
    "garc_final",
    "v5_complete",
    "pepd_shared_range",
    "vdn_shared_range",
    "transformer_shared_range",
)
CLAIM_TIERS: Final[dict[str, str]] = {
    "garc_final": "primary_end_to_end_final_model",
    "v5_complete": "secondary_end_to_end_internal_ablation",
    "pepd_shared_range": "secondary_end_to_end_progress_backbone_control",
    "vdn_shared_range": "secondary_end_to_end_progress_backbone_control",
    "transformer_shared_range": "secondary_end_to_end_sensitivity_only",
}
METHOD_NAME_TOKENS: Final[dict[str, str]] = {
    "garc_final": "garc",
    "v5_complete": "v5",
    "pepd_shared_range": "pepd",
    "vdn_shared_range": "vdn",
    "transformer_shared_range": "transformer",
}
FORBIDDEN_FRONTEND_KEYS: Final[frozenset[str]] = frozenset(
    {
        "bbox",
        "crop",
        "dialbbox",
        "manualbbox",
        "manualcrop",
        "groundtruthbbox",
        "gtbbox",
        "scalemark",
        "scalestart",
        "scaleend",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.casefold()).issubset(frozenset("0123456789abcdef"))
    )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    value = strict_json_load(Path(path).resolve(strict=True))
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _binding(value: Any, *, label: str, verify: bool = True) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} binding is absent")
    path = Path(str(value.get("path") or ""))
    _require(path.is_absolute(), f"{label}.path must be absolute")
    path = path.resolve(strict=verify)
    digest = str(value.get("sha256") or "").casefold()
    _require(_is_sha256(digest), f"{label}.sha256 is invalid")
    if verify:
        _require(sha256_file(path) == digest, f"{label} hash drift")
    return {"path": str(path), "sha256": digest}


def _atomic_new(path: Path, value: Mapping[str, Any] | bytes) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _assert_no_manual_frontend_inputs(value: Any, *, location: str) -> None:
    assert_label_free(value, location=location)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in FORBIDDEN_FRONTEND_KEYS:
                raise ValueError(f"manual/GT crop or range key {key!r} is forbidden at {location}")
            _assert_no_manual_frontend_inputs(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _assert_no_manual_frontend_inputs(nested, location=f"{location}[{index}]")


def _load_dataset_identity(path: Path) -> dict[str, Any]:
    """Read owner metadata only; never dereference its manifest/label paths."""

    identity_path = Path(path).resolve(strict=True)
    value = _strict_object(identity_path, label="dataset identity")
    _require(value.get("protocol") == DATASET_IDENTITY_PROTOCOL, "dataset identity protocol drift")
    _require(value.get("status") == "owner_frozen", "dataset is not owner-frozen")
    _require(value.get("definition_authority") == "dataset_owner", "dataset-owner authority absent")
    cohort = value.get("cohort_definition")
    _require(isinstance(cohort, Mapping), "cohort definition absent")
    declared = int(cohort.get("declared_images", -1))
    _require(declared >= MINIMUM_BLIND_IMAGES, "blind cohort has fewer than 1200 images")
    _require(cohort.get("deduplicated_before_freeze") is True, "cohort is not declared deduplicated")
    _require(cohort.get("declared_as_frozen_unseen_blind_test") is True, "cohort is not declared unseen blind")
    _require(cohort.get("source_unit") == "original_full_scene_photograph", "cohort source is not original full-scene photos")
    manifest = _binding(value.get("unlabeled_manifest"), label="full-scene manifest", verify=False)
    labels = _binding(value.get("labels"), label="labels", verify=False)
    manifest_value = value.get("unlabeled_manifest") or {}
    labels_value = value.get("labels") or {}
    for declaration, label in ((manifest_value, "manifest"), (labels_value, "labels")):
        _require(int(declaration.get("rows", -1)) == declared, f"{label} row declaration drift")
    _require(manifest_value.get("input_role") == "original_full_scene", "manifest is not full-scene input")
    _require(manifest_value.get("contains_labels") is False, "inference manifest contains labels")
    _require(manifest_value.get("contains_manual_or_gt_range") is False, "inference manifest contains range supervision")
    _require(manifest_value.get("contains_manual_or_gt_crop") is False, "inference manifest contains crop supervision")
    _require(Path(manifest["path"]) != Path(labels["path"]), "manifest and labels must be separate")
    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "one-shot authorization absent")
    for key in (
        "one_shot_image_inference",
        "one_shot_scoring_after_prediction_seal",
        "no_tuning_after_result",
    ):
        _require(authorization.get(key) is True, f"dataset authorization missing: {key}")
    return {
        "identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
        "cohort": dict(cohort),
        "manifest": {**manifest, "rows": declared},
        "labels": {**labels, "rows": declared},
        "authorization": dict(authorization),
    }


def load_frontend_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    plan_path = Path(path).resolve(strict=True)
    # A boolean ``public_data_only`` assertion is not sufficient evidence for
    # the irreversible blind run.  Authenticate the complete SyncG/train
    # corpus -> calibration selection -> independent validation -> best.pt
    # lineage (and its sibling seal) before accepting the generic frontend
    # schema below.  This also explicitly rejects the legacy yolo_findMeter
    # implementation.
    from experiments.syncg_meter_detector_frontend import verify_frontend_plan

    verified_path, value = verify_frontend_plan(plan_path)
    _require(verified_path == plan_path, "frontend verifier path drift")
    _require(value.get("protocol") == FRONTEND_PROTOCOL, "frontend protocol drift")
    _require(value.get("status") == "public_selected_frozen", "frontend is not frozen")
    _assert_no_manual_frontend_inputs(value, location="frontend_plan")
    detector = _binding(value.get("detector_checkpoint"), label="meter detector")
    source = _binding(value.get("detector_source"), label="meter detector source")
    _require(str(value.get("detector_class") or "") == "targetDetectModel", "unsupported detector class")
    threshold = _finite(value.get("confidence_threshold"))
    padding = _finite(value.get("padding_fraction"))
    _require(threshold is not None and 0.0 <= threshold <= 1.0, "invalid detector confidence")
    _require(padding is not None and 0.0 <= padding <= 0.25, "invalid ROI padding fraction")
    classes = value.get("accepted_class_ids")
    _require(isinstance(classes, list) and classes, "accepted detector classes absent")
    _require(all(isinstance(item, int) and item >= 0 for item in classes), "invalid class id")
    _require(len(set(classes)) == len(classes), "duplicate detector class id")
    contract = value.get("contract")
    _require(isinstance(contract, Mapping), "frontend contract absent")
    expected = {
        "input": "original_full_scene_bgr_uint8",
        "selection_rule": "highest_confidence_then_xyxy_lexicographic",
        "padding_rule": "fraction_of_detected_box_width_and_height_symmetric_clamped",
        "fallback_to_full_frame": False,
        "caller_bbox_allowed": False,
        "caller_crop_allowed": False,
        "ground_truth_geometry_allowed": False,
        "correction_or_warp_applied": False,
        "same_roi_for_all_methods": True,
        "detector_failure_penalty_nmae": FAILURE_PENALTY,
    }
    _require(dict(contract) == expected, "shared frontend contract drift")
    selection = value.get("selection_audit")
    _require(isinstance(selection, Mapping), "frontend selection audit absent")
    _require(selection.get("public_data_only") is True, "frontend was not selected on public data only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(selection.get(key) is False, f"frontend selection scope violation: {key}")
    return plan_path, {
        **value,
        "detector_checkpoint": detector,
        "detector_source": source,
        "confidence_threshold": threshold,
        "padding_fraction": padding,
        "accepted_class_ids": list(classes),
    }


def _runtime_artifact_hashes(value: Any, *, label: str) -> set[str]:
    _require(isinstance(value, list) and value, f"{label} runtime artifacts absent")
    result: set[str] = set()
    paths: set[str] = set()
    for index, raw in enumerate(value, 1):
        bound = _binding(raw, label=f"{label} runtime artifact {index}")
        _require(bound["path"] not in paths, f"{label} repeats a runtime artifact")
        paths.add(bound["path"])
        result.add(bound["sha256"])
    return result


def _assert_factory_declaration(path: Path, function_name: str, *, label: str) -> None:
    """Check a factory entrypoint statically without importing model code."""

    source = Path(path).resolve(strict=True).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    _require(function_name in names, f"{label} lacks top-level {function_name}")


def load_method_roster(path: Path) -> tuple[Path, dict[str, Any], dict[str, FrozenFullAutoBundle]]:
    roster_path = Path(path).resolve(strict=True)
    value = _strict_object(roster_path, label="method roster")
    _require(value.get("protocol") == ROSTER_PROTOCOL, "method roster protocol drift")
    _require(value.get("status") == "all_full_reading_methods_frozen", "method roster incomplete")
    _assert_no_manual_frontend_inputs(value, location="method_roster")
    audit = value.get("audit")
    _require(isinstance(audit, Mapping), "method roster audit absent")
    _require(audit.get("public_data_only") is True, "method roster was not selected on public data only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(audit.get(key) is False, f"method roster scope violation: {key}")
    methods = value.get("methods")
    _require(isinstance(methods, Mapping), "method roster methods absent")
    _require(set(methods) == set(METHOD_ROLES), "method roster must contain exactly five frozen roles")
    bundles: dict[str, FrozenFullAutoBundle] = {}
    normalized: dict[str, Any] = {}
    method_names: set[str] = set()
    for role in METHOD_ROLES:
        raw = methods[role]
        _require(isinstance(raw, Mapping), f"{role} method specification invalid")
        _require(raw.get("claim_tier") == CLAIM_TIERS[role], f"{role} claim tier drift")
        bundle_binding = _binding(raw.get("bundle"), label=f"{role} bundle")
        factory_binding = _binding(raw.get("factory"), label=f"{role} factory")
        function = str(raw.get("factory_function") or "")
        _require(
            function == "build_field_blind_full_auto_providers",
            f"{role} unsupported factory function",
        )
        _assert_factory_declaration(
            Path(factory_binding["path"]), function, label=f"{role} factory"
        )
        bundle = FrozenFullAutoBundle.load(Path(bundle_binding["path"]))
        _require(bundle.execution_mode == EXECUTION_FORMAL, f"{role} bundle is not formal")
        _require(METHOD_NAME_TOKENS[role] in bundle.method_name.casefold(), f"{role} method identity mismatch")
        _require(bundle.method_name not in method_names, "method display names are not unique")
        method_names.add(bundle.method_name)
        _require(bundle.descriptor["factory_source_sha256"] == factory_binding["sha256"], f"{role} factory differs from bundle")
        hashes = _runtime_artifact_hashes(raw.get("runtime_artifacts"), label=role)
        required_hashes = set(bundle.progress_binding.artifact_sha256.values())
        required_hashes.update(bundle.progress_binding.source_sha256.values())
        required_hashes.update(bundle.range_binding.artifact_sha256.values())
        required_hashes.update(bundle.range_binding.source_sha256.values())
        required_hashes.add(factory_binding["sha256"])
        required_hashes.add(str(bundle.descriptor["adapter_source_sha256"]))
        if bundle.reference_detector_sha256 is not None:
            required_hashes.add(bundle.reference_detector_sha256)
        missing = required_hashes - hashes
        _require(not missing, f"{role} runtime artifact inventory incomplete")
        for component in (bundle.progress_binding, bundle.range_binding):
            _require(component.frozen, f"{role}/{component.name} is not frozen")
            _require(component.verified_complete, f"{role}/{component.name} is incomplete")
            _require(not component.synthetic, f"{role}/{component.name} is synthetic")
        _require(raw.get("output_mode") == OUTPUT_MODE_FULL_READING, f"{role} is not a full-reading method")
        _require(raw.get("evidence_role") == EVIDENCE_ROLE_PRIMARY, f"{role} bundle evidence role drift")
        declared_range = str(raw.get("range_binding_sha256") or "").casefold()
        _require(declared_range == bundle.range_binding_sha256, f"{role} range binding drift")
        factory_config = raw.get("factory_config")
        _require(isinstance(factory_config, Mapping), f"{role} factory config absent")
        _assert_no_manual_frontend_inputs(
            factory_config, location=f"method_roster.methods.{role}.factory_config"
        )
        bundles[role] = bundle
        normalized[role] = {
            **dict(raw),
            "bundle": bundle_binding,
            "factory": factory_binding,
            "method_name": bundle.method_name,
            "bundle_sha256": bundle.bundle_sha256,
            "range_binding_sha256": bundle.range_binding_sha256,
            "factory_config": dict(factory_config),
        }
    garc_range = bundles["garc_final"].range_binding_sha256
    for role in ("pepd_shared_range", "vdn_shared_range", "transformer_shared_range"):
        _require(bundles[role].range_binding_sha256 == garc_range, f"{role} does not use GARC's frozen automatic range")
    _require(
        value.get("comparison_contract")
        == {
            "same_full_scene_frontend": True,
            "same_canonical_roi_per_sample": True,
            "all_predictions_sealed_together_before_labels": True,
            "progress_only_outputs_eligible": False,
            "garc_is_only_primary_final_model": True,
            "v5_is_internal_end_to_end_ablation": True,
            "pepd_vdn_share_garc_range_for_backbone_control": True,
            "transformer_is_sensitivity_only": True,
        },
        "comparison claim boundary drift",
    )
    return roster_path, {**value, "methods": normalized}, bundles


def freeze_protocol(
    *,
    dataset_identity_path: Path,
    paper_results_path: Path,
    paper_tables_root: Path = DEFAULT_PAPER_TABLES_ROOT,
    frontend_plan_path: Path,
    method_roster_path: Path,
    output_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Freeze all public/model decisions without opening bound field files."""

    output = Path(output_path).resolve()
    run = Path(run_root).resolve()
    if output.exists():
        raise FileExistsError(f"multimethod protocol already exists: {output}")
    if run.exists():
        raise FileExistsError(f"one-shot run root already exists: {run}")
    dataset = _load_dataset_identity(dataset_identity_path)
    paper_path = Path(paper_results_path).resolve(strict=True)
    paper_authority = load_paper_authority(
        paper_path, rendered_tables_root=paper_tables_root
    )
    paper = paper_authority["summary_value"]
    frontend_path, frontend = load_frontend_plan(frontend_plan_path)
    roster_path, roster, _ = load_method_roster(method_roster_path)
    source = Path(__file__).resolve()
    adapter_source = source.with_name("v5_unified_full_auto_adapter.py")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "frozen_authorized_not_started",
        "created_utc": _now(),
        "protocol_location": str(output),
        "dataset_identity": dataset["identity"],
        "cohort_definition": dataset["cohort"],
        "unlabeled_full_scene_manifest": dataset["manifest"],
        "labels": dataset["labels"],
        "authorization": dataset["authorization"],
        "paper_results": {"path": str(paper_path), "sha256": sha256_file(paper_path)},
        "paper_authority": {
            key: value
            for key, value in paper_authority.items()
            if key != "summary_value"
        },
        "paper_results_status": paper["status"],
        "frontend_plan": {"path": str(frontend_path), "sha256": sha256_file(frontend_path)},
        "frontend_identity": {
            "detector_checkpoint_sha256": frontend["detector_checkpoint"]["sha256"],
            "detector_source_sha256": frontend["detector_source"]["sha256"],
            "confidence_threshold": frontend["confidence_threshold"],
            "padding_fraction": frontend["padding_fraction"],
            "accepted_class_ids": frontend["accepted_class_ids"],
            "contract_sha256": canonical_json_sha256(frontend["contract"]),
        },
        "method_roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        "method_identity": {
            role: {
                "method_name": roster["methods"][role]["method_name"],
                "claim_tier": CLAIM_TIERS[role],
                "bundle_sha256": roster["methods"][role]["bundle_sha256"],
                "range_binding_sha256": roster["methods"][role]["range_binding_sha256"],
            }
            for role in METHOD_ROLES
        },
        "run": {
            "root": str(run),
            "output_root_must_not_preexist": True,
            "retry_after_claim": "forbidden_without_separate_adjudication_protocol",
        },
        "scoring_contract": {
            "full_denominator_failure_penalty_nmae": FAILURE_PENALTY,
            "primary_metric": "abs(predicted_reading-ground_truth)/abs(scale_end-scale_start)",
            "coverage": "successful full-reading predictions / all frozen images",
            "method_or_threshold_selection_after_result": False,
            "labels_available_during_inference": False,
        },
        "chronology": {
            "freeze_reads_owner_identity_and_public_artifacts_only": True,
            "field_manifest_opened_during_freeze": False,
            "field_labels_opened_during_freeze": False,
            "field_images_opened_during_freeze": False,
            "inference_claim_precedes_manifest_or_image_access": True,
            "all_method_predictions_sealed_together_before_labels": True,
            "score_claim_precedes_label_access": True,
        },
        "runtime_sources": {
            "runner": {"path": str(source), "sha256": sha256_file(source)},
            "full_auto_adapter": {
                "path": str(adapter_source),
                "sha256": sha256_file(adapter_source),
            },
        },
    }
    _atomic_new(output, payload)
    return payload


def _ledger_paths(protocol_path: Path) -> dict[str, Path]:
    protocol = Path(protocol_path).resolve(strict=False)
    prefix = protocol.with_suffix("")
    return {
        "inference_claim": prefix.with_name(prefix.name + ".inference-claim.json"),
        "inference_complete": prefix.with_name(prefix.name + ".inference-complete.json"),
        "score_claim": prefix.with_name(prefix.name + ".score-claim.json"),
        "score_complete": prefix.with_name(prefix.name + ".score-complete.json"),
    }


def load_protocol(path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, FrozenFullAutoBundle]]:
    protocol_path = Path(path).resolve(strict=True)
    value = _strict_object(protocol_path, label="multimethod field protocol")
    _require(value.get("protocol") == PROTOCOL, "multimethod protocol drift")
    _require(value.get("status") == "frozen_authorized_not_started", "protocol status drift")
    _require(Path(str(value.get("protocol_location") or "")).resolve(strict=False) == protocol_path, "protocol was moved or copied")
    for label, binding in (value.get("runtime_sources") or {}).items():
        _binding(binding, label=f"runtime source {label}")
    dataset_binding = _binding(value.get("dataset_identity"), label="dataset identity")
    dataset = _load_dataset_identity(Path(dataset_binding["path"]))
    _require(dataset["manifest"] == value.get("unlabeled_full_scene_manifest"), "manifest declaration drift")
    _require(dataset["labels"] == value.get("labels"), "label declaration drift")
    paper_binding = _binding(value.get("paper_results"), label="paper results")
    declared_authority = value.get("paper_authority")
    _require(isinstance(declared_authority, Mapping), "paper authority absent")
    tables_manifest = _binding(
        declared_authority.get("tables_manifest"), label="paper tables manifest"
    )
    paper_authority = load_paper_authority(
        Path(paper_binding["path"]),
        rendered_tables_root=Path(tables_manifest["path"]).parent,
    )
    expected_authority = {
        key: item
        for key, item in paper_authority.items()
        if key != "summary_value"
    }
    _require(dict(declared_authority) == expected_authority, "paper v2 authority drift")
    paper = paper_authority["summary_value"]
    _require(paper["status"] == value.get("paper_results_status"), "paper status drift")
    frontend_binding = _binding(value.get("frontend_plan"), label="frontend plan")
    _, frontend = load_frontend_plan(Path(frontend_binding["path"]))
    roster_binding = _binding(value.get("method_roster"), label="method roster")
    _, roster, bundles = load_method_roster(Path(roster_binding["path"]))
    expected_methods = {
        role: {
            "method_name": roster["methods"][role]["method_name"],
            "claim_tier": CLAIM_TIERS[role],
            "bundle_sha256": roster["methods"][role]["bundle_sha256"],
            "range_binding_sha256": roster["methods"][role]["range_binding_sha256"],
        }
        for role in METHOD_ROLES
    }
    _require(value.get("method_identity") == expected_methods, "frozen method identity drift")
    return protocol_path, value, frontend, roster, bundles


def static_preflight(protocol_path: Path) -> dict[str, Any]:
    """Authenticate frozen public/model state without field file access."""

    path, value, frontend, roster, _ = load_protocol(protocol_path)
    ledgers = _ledger_paths(path)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated_without_field_manifest_label_or_image_access",
        "protocol_sha256": sha256_file(path),
        "declared_images": int(value["cohort_definition"]["declared_images"]),
        "method_roles": list(METHOD_ROLES),
        "method_names": [roster["methods"][role]["method_name"] for role in METHOD_ROLES],
        "shared_frontend_detector_sha256": frontend["detector_checkpoint"]["sha256"],
        "same_roi_for_all_methods": True,
        "field_manifest_opened": False,
        "field_labels_opened": False,
        "field_images_opened": False,
        "inference_claim_exists": ledgers["inference_claim"].exists(),
        "inference_complete_exists": ledgers["inference_complete"].exists(),
        "score_complete_exists": ledgers["score_complete"].exists(),
    }


@dataclass(frozen=True)
class FullSceneItem:
    sample_id: str
    group_id: str
    image_path: Path
    image_sha256: str
    frame_sha256: str


def validate_full_scene_manifest(rows: Sequence[Mapping[str, Any]]) -> list[FullSceneItem]:
    assert_label_free(rows, location="full_scene_manifest")
    _require(bool(rows), "full-scene manifest is empty")
    result: list[FullSceneItem] = []
    samples: set[str] = set()
    image_hashes: set[str] = set()
    image_paths: set[Path] = set()
    for index, raw in enumerate(rows, 1):
        row = dict(raw)
        _require(set(row) == FULL_SCENE_MANIFEST_KEYS, f"row {index}: full-scene manifest schema drift")
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id) and bool(group_id), f"row {index}: missing identity")
        _require(sample_id not in samples, f"duplicate sample_id: {sample_id}")
        image_digest = str(row.get("image_sha256") or "").casefold()
        frame_digest = str(row.get("frame_sha256") or "").casefold()
        _require(_is_sha256(image_digest) and _is_sha256(frame_digest), f"{sample_id}: invalid image identity")
        _require(image_digest == frame_digest, f"{sample_id}: original full-scene image/frame hash mismatch")
        image_path = Path(str(row.get("image_path") or ""))
        _require(image_path.is_absolute(), f"{sample_id}: image path must be absolute")
        image_path = image_path.resolve(strict=False)
        _require(image_digest not in image_hashes, f"duplicate full-scene image bytes: {sample_id}")
        _require(image_path not in image_paths, f"duplicate full-scene image path: {sample_id}")
        samples.add(sample_id)
        image_hashes.add(image_digest)
        image_paths.add(image_path)
        result.append(FullSceneItem(sample_id, group_id, image_path, image_digest, frame_digest))
    return result


def _load_shared_detector(frontend: Mapping[str, Any]) -> Any:
    expected_source = (
        PROJECT_ROOT / "utils/angleDetect/yoloDetection/yoloDectect.py"
    ).resolve(strict=True)
    source = Path(frontend["detector_source"]["path"]).resolve(strict=True)
    _require(source == expected_source, "frontend detector source is not the production implementation")
    _require(sha256_file(source) == frontend["detector_source"]["sha256"], "frontend source drift")
    from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel

    return targetDetectModel(frontend["detector_checkpoint"]["path"])


def select_shared_roi(
    detector: Any,
    full_scene_bgr: np.ndarray,
    *,
    confidence_threshold: float,
    padding_fraction: float,
    accepted_class_ids: Sequence[int],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run one detector and deterministically select/pad one meter ROI."""

    image = validate_canonical_roi(full_scene_bgr)
    height, width = image.shape[:2]
    try:
        confidences, boxes, _, class_ids = detector.target_detection(
            image, confidence=float(confidence_threshold)
        )
    except Exception as error:
        return None, {
            "status": False,
            "failure_code": f"shared_meter_detector_exception:{type(error).__name__}",
            "candidate_count": 0,
            "fallback_to_full_frame": False,
            "detector_invocations": 1,
        }
    candidates: list[tuple[float, int, int, int, int, int]] = []
    accepted = set(int(value) for value in accepted_class_ids)
    _require(
        len(confidences) == len(boxes) == len(class_ids),
        "shared meter detector returned misaligned candidate arrays",
    )
    for confidence, box, class_id in zip(confidences, boxes, class_ids, strict=True):
        score = _finite(confidence)
        class_value = int(class_id)
        array = np.asarray(box, dtype=np.float64)
        if (
            score is None
            or score < confidence_threshold
            or class_value not in accepted
            or array.shape != (4, 2)
            or not np.isfinite(array).all()
        ):
            continue
        x1 = int(math.floor(float(array[:, 0].min())))
        y1 = int(math.floor(float(array[:, 1].min())))
        x2 = int(math.ceil(float(array[:, 0].max())))
        y2 = int(math.ceil(float(array[:, 1].max())))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        candidates.append((float(score), x1, y1, x2, y2, class_value))
    if not candidates:
        return None, {
            "status": False,
            "failure_code": "shared_meter_detector_no_valid_detection",
            "candidate_count": 0,
            "fallback_to_full_frame": False,
            "detector_invocations": 1,
        }
    # Descending score, then deterministic ascending coordinates/class.
    selected = sorted(candidates, key=lambda row: (-row[0], *row[1:]))[0]
    score, x1, y1, x2, y2, class_id = selected
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(math.ceil(box_width * float(padding_fraction)))
    pad_y = int(math.ceil(box_height * float(padding_fraction)))
    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    roi = image[py1:py2, px1:px2].copy()
    if min(roi.shape[:2]) < 32:
        return None, {
            "status": False,
            "failure_code": "shared_meter_detector_roi_too_small",
            "candidate_count": len(candidates),
            "fallback_to_full_frame": False,
            "detector_invocations": 1,
            "selected_confidence": score,
            "detected_xyxy": [x1, y1, x2, y2],
            "padded_xyxy": [px1, py1, px2, py2],
            "class_id": class_id,
        }
    return roi, {
        "status": True,
        "failure_code": None,
        "candidate_count": len(candidates),
        "fallback_to_full_frame": False,
        "detector_invocations": 1,
        "selected_confidence": score,
        "detected_xyxy": [x1, y1, x2, y2],
        "padded_xyxy": [px1, py1, px2, py2],
        "class_id": class_id,
        "padding_fraction": float(padding_fraction),
    }


def _load_adapters(
    roster: Mapping[str, Any],
    bundles: Mapping[str, FrozenFullAutoBundle],
) -> dict[str, UnifiedFullAutoAdapter]:
    """Authenticate and construct every method before opening one image."""

    result: dict[str, UnifiedFullAutoAdapter] = {}
    for role in METHOD_ROLES:
        spec = roster["methods"][role]
        factory, digest = _load_factory(
            Path(spec["factory"]["path"]), str(spec["factory_function"])
        )
        _require(digest == spec["factory"]["sha256"], f"{role} factory hash drift")
        value = factory(
            dict(bundles[role].descriptor), dict(spec["factory_config"])
        )
        _require(
            isinstance(value, Mapping)
            and set(value)
            == {"progress_provider", "automatic_numeric_range_pipeline"},
            f"{role} factory returned an invalid provider mapping",
        )
        progress = value["progress_provider"]
        numeric_range = value["automatic_numeric_range_pipeline"]
        result[role] = UnifiedFullAutoAdapter(
            progress_provider=progress,
            automatic_numeric_range_pipeline=numeric_range,
            bundle=bundles[role],
        )
    _require(set(result) == set(METHOD_ROLES), "not all formal adapters loaded")
    return result


def _failure_prediction(bundle: FrozenFullAutoBundle, code: str) -> FullAutoPrediction:
    automatic_reference: Mapping[str, Any] | None = None
    if bundle.reference_mode == "method_internal_auto":
        automatic_reference = {
            "status": False,
            "start_angle": None,
            "range_angle": None,
            "reference_branch": "shared_full_scene_frontend:not_invoked",
            "failure_code": "canonical_roi_unavailable",
        }
    return FullAutoPrediction(
        status=False,
        prediction_progress=None,
        predicted_scale_start=None,
        predicted_scale_end=None,
        range_confidence=None,
        failure_code=code,
        automatic_reference=automatic_reference,
        telemetry={
            "protocol": PROTOCOL,
            "shared_frontend_failure": code,
            "providers_invoked": False,
            "caller_range_consumed": False,
            "caller_crop_consumed": False,
            "ground_truth_consumed": False,
        },
    )


def _decode_full_scene(item: FullSceneItem) -> tuple[np.ndarray | None, str | None]:
    try:
        payload = item.image_path.read_bytes()
    except OSError:
        return None, "full_scene_read_failed"
    if hashlib.sha256(payload).hexdigest() != item.image_sha256:
        raise ValueError(f"{item.sample_id}: full-scene image hash drift")
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        return None, "full_scene_decode_failed"
    try:
        return validate_canonical_roi(image), None
    except (TypeError, ValueError):
        return None, "full_scene_invalid_pixels"


def _method_records(
    *,
    roi: np.ndarray | None,
    frontend_failure: str | None,
    adapters: Mapping[str, UnifiedFullAutoAdapter],
    bundles: Mapping[str, FrozenFullAutoBundle],
    canonical_roi_sha256: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    shared_before = None if roi is None else _progress_image_sha256(roi)
    for role in METHOD_ROLES:
        bundle = bundles[role]
        if roi is None:
            prediction = _failure_prediction(
                bundle, str(frontend_failure or "shared_frontend_failed")
            )
        else:
            method_input = roi.copy()
            prediction = adapters[role].predict(
                method_input, input_is_canonical_meter_roi=True
            )
            _require(
                _progress_image_sha256(method_input) == shared_before,
                f"{role} mutated its supplied shared ROI",
            )
        record = prediction.as_method_record(
            bundle=bundle,
            canonical_roi_file_sha256=canonical_roi_sha256,
        )
        _require(record["evidence_role"] == EVIDENCE_ROLE_PRIMARY, f"{role} emitted component output")
        _require(record["output_mode"] == OUTPUT_MODE_FULL_READING, f"{role} is not full-reading")
        records[bundle.method_name] = record
    _require(len(records) == len(METHOD_ROLES), "method name collision in predictions")
    if roi is not None:
        _require(_progress_image_sha256(roi) == shared_before, "shared ROI changed across methods")
    return records


def _claim_and_prepare(
    protocol_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, FrozenFullAutoBundle], list[FullSceneItem]]:
    path, protocol, frontend, roster, bundles = load_protocol(protocol_path)
    ledgers = _ledger_paths(path)
    run_root = Path(protocol["run"]["root"])
    if run_root.exists():
        raise FileExistsError(f"one-shot run root already exists: {run_root}")
    for ledger in ledgers.values():
        if ledger.exists():
            raise FileExistsError(f"one-shot ledger already exists: {ledger}")
    claim = {
        "schema_version": 1,
        "protocol": INFERENCE_CLAIM_PROTOCOL,
        "status": "claimed_before_field_manifest_label_or_image_access",
        "created_utc": _now(),
        "parent_protocol_sha256": sha256_file(path),
        "field_manifest_opened_before_claim": False,
        "field_labels_opened_before_claim": False,
        "field_images_opened_before_claim": False,
        "single_use": True,
    }
    _atomic_new(ledgers["inference_claim"], claim)
    manifest_binding = protocol["unlabeled_full_scene_manifest"]
    manifest = Path(manifest_binding["path"]).resolve(strict=True)
    _require(sha256_file(manifest) == manifest_binding["sha256"], "full-scene manifest hash drift")
    rows = strict_jsonl_load(manifest)
    items = validate_full_scene_manifest(rows)
    _require(len(items) == int(manifest_binding["rows"]), "full-scene manifest row count drift")
    return path, protocol, frontend, roster, bundles, items


def run_once(protocol_path: Path) -> dict[str, Any]:
    """Run the shared detector and all frozen methods, then seal together."""

    path, protocol, frontend, roster, bundles, items = _claim_and_prepare(protocol_path)
    ledgers = _ledger_paths(path)
    # All model code/artifacts must load successfully before the first field image.
    detector = _load_shared_detector(frontend)
    adapters = _load_adapters(roster, bundles)
    run_root = Path(protocol["run"]["root"])
    run_root.mkdir(parents=True, exist_ok=False)
    prediction_path = run_root / "predictions.sealed.jsonl"
    seal_path = run_root / "prediction_seal.json"
    rows: list[dict[str, Any]] = []
    for item in items:
        image, input_failure = _decode_full_scene(item)
        if image is None:
            roi = None
            frontend_record = {
                "status": False,
                "failure_code": input_failure,
                "detector_invocations": 0,
                "fallback_to_full_frame": False,
            }
        else:
            roi, frontend_record = select_shared_roi(
                detector,
                image,
                confidence_threshold=float(frontend["confidence_threshold"]),
                padding_fraction=float(frontend["padding_fraction"]),
                accepted_class_ids=frontend["accepted_class_ids"],
            )
        roi_hash = item.frame_sha256 if roi is None else _progress_image_sha256(roi)
        method_records = _method_records(
            roi=roi,
            frontend_failure=frontend_record.get("failure_code"),
            adapters=adapters,
            bundles=bundles,
            canonical_roi_sha256=roi_hash,
        )
        row = {
            "schema_version": 1,
            "protocol": PREDICTION_PROTOCOL,
            "sample_id": item.sample_id,
            "group_id": item.group_id,
            "frame_sha256": item.frame_sha256,
            "full_scene_image_sha256": item.image_sha256,
            "canonical_roi_sha256": roi_hash,
            "shared_frontend": {
                **frontend_record,
                "detector_checkpoint_sha256": frontend["detector_checkpoint"]["sha256"],
                "same_roi_for_all_methods": True,
                "manual_or_gt_crop_used": False,
            },
            "methods": method_records,
        }
        assert_label_free(row, location=f"multimethod_prediction.{item.sample_id}")
        rows.append(row)
    _atomic_new(prediction_path, _jsonl_bytes(rows))
    seal = {
        "schema_version": 1,
        "protocol": PREDICTION_SEAL_PROTOCOL,
        "status": "all_method_predictions_sealed_together_before_labels",
        "created_utc": _now(),
        "parent_protocol_sha256": sha256_file(path),
        "inference_claim_sha256": sha256_file(ledgers["inference_claim"]),
        "predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "rows": len(rows),
        },
        "method_identity": protocol["method_identity"],
        "field_labels_opened": False,
        "manual_or_gt_crop_used": False,
        "manual_or_gt_range_supplied_to_model": False,
        "same_shared_roi_for_all_methods": True,
    }
    _atomic_new(seal_path, seal)
    completion = {
        "schema_version": 1,
        "protocol": INFERENCE_COMPLETE_PROTOCOL,
        "status": "complete_all_predictions_sealed_before_labels",
        "created_utc": _now(),
        "parent_protocol_sha256": sha256_file(path),
        "prediction_seal": {"path": str(seal_path), "sha256": sha256_file(seal_path)},
        "predictions": seal["predictions"],
        "rows": len(rows),
        "field_labels_opened": False,
    }
    _atomic_new(ledgers["inference_complete"], completion)
    return completion


def _verify_sealed_predictions(
    protocol_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    path, protocol, _, roster, _ = load_protocol(protocol_path)
    ledgers = _ledger_paths(path)
    _require(ledgers["inference_claim"].is_file(), "inference claim absent")
    _require(ledgers["inference_complete"].is_file(), "inference completion absent")
    complete = _strict_object(ledgers["inference_complete"], label="inference completion")
    _require(complete.get("protocol") == INFERENCE_COMPLETE_PROTOCOL, "inference completion protocol drift")
    _require(complete.get("status") == "complete_all_predictions_sealed_before_labels", "inference incomplete")
    _require(complete.get("field_labels_opened") is False, "inference reports label access")
    seal_binding = _binding(complete.get("prediction_seal"), label="prediction seal")
    seal = _strict_object(Path(seal_binding["path"]), label="prediction seal")
    _require(seal.get("protocol") == PREDICTION_SEAL_PROTOCOL, "prediction seal protocol drift")
    _require(seal.get("status") == "all_method_predictions_sealed_together_before_labels", "predictions were not jointly sealed")
    _require(seal.get("parent_protocol_sha256") == sha256_file(path), "prediction seal parent drift")
    _require(seal.get("method_identity") == protocol["method_identity"], "sealed method identity drift")
    _require(seal.get("field_labels_opened") is False, "prediction seal reports label access")
    _require(seal.get("manual_or_gt_crop_used") is False, "prediction seal reports supervised crop")
    _require(seal.get("manual_or_gt_range_supplied_to_model") is False, "prediction seal reports supervised range")
    predictions_binding = _binding(seal.get("predictions"), label="sealed predictions")
    _require(complete.get("predictions") == seal.get("predictions"), "completion/prediction seal binding drift")
    predictions = strict_jsonl_load(Path(predictions_binding["path"]))
    _require(len(predictions) == int(predictions_binding.get("rows", -1)), "sealed prediction row count drift")
    method_names = {roster["methods"][role]["method_name"] for role in METHOD_ROLES}
    seen: set[str] = set()
    for index, row in enumerate(predictions, 1):
        assert_label_free(row, location=f"sealed_prediction[{index}]")
        _require(row.get("protocol") == PREDICTION_PROTOCOL, f"prediction {index} protocol drift")
        sample_id = str(row.get("sample_id") or "")
        _require(bool(sample_id) and sample_id not in seen, f"prediction {index} duplicate/missing sample")
        seen.add(sample_id)
        _require(_is_sha256(row.get("frame_sha256")), f"{sample_id}: frame hash invalid")
        _require(row.get("full_scene_image_sha256") == row.get("frame_sha256"), f"{sample_id}: full-scene identity drift")
        roi_hash = str(row.get("canonical_roi_sha256") or "")
        _require(_is_sha256(roi_hash), f"{sample_id}: ROI identity invalid")
        frontend = row.get("shared_frontend")
        _require(isinstance(frontend, Mapping), f"{sample_id}: shared frontend absent")
        _require(frontend.get("same_roi_for_all_methods") is True, f"{sample_id}: ROI is not shared")
        _require(frontend.get("manual_or_gt_crop_used") is False, f"{sample_id}: supervised crop reported")
        methods = row.get("methods")
        _require(isinstance(methods, Mapping) and set(methods) == method_names, f"{sample_id}: method roster drift")
        statuses: list[bool] = []
        for role in METHOD_ROLES:
            method_name = roster["methods"][role]["method_name"]
            record = methods[method_name]
            _require(isinstance(record, Mapping), f"{sample_id}/{method_name}: invalid record")
            _require(record.get("checkpoint_sha256") == roster["methods"][role]["bundle_sha256"], f"{sample_id}/{role}: bundle drift")
            _require(record.get("canonical_roi_sha256") == roi_hash, f"{sample_id}/{role}: ROI drift")
            _require(record.get("evidence_role") == EVIDENCE_ROLE_PRIMARY, f"{sample_id}/{role}: component output")
            _require(record.get("output_mode") == OUTPUT_MODE_FULL_READING, f"{sample_id}/{role}: not full-reading")
            status = record.get("status")
            _require(isinstance(status, bool), f"{sample_id}/{role}: invalid status")
            statuses.append(status)
            values = (
                _finite(record.get("prediction_progress")),
                _finite(record.get("predicted_scale_start")),
                _finite(record.get("predicted_scale_end")),
            )
            confidence = _finite(record.get("range_confidence"))
            if status:
                _require(all(value is not None for value in values), f"{sample_id}/{role}: successful output incomplete")
                _require(values[1] != values[2], f"{sample_id}/{role}: zero predicted span")
                _require(confidence is not None and 0.0 <= confidence <= 1.0, f"{sample_id}/{role}: invalid range confidence")
            else:
                _require(all(value is None for value in values), f"{sample_id}/{role}: failed output has prediction")
                _require(confidence is None, f"{sample_id}/{role}: failed output has range confidence")
        if frontend.get("status") is not True:
            _require(not any(statuses), f"{sample_id}: a method bypassed shared frontend failure")
    return path, protocol, roster, predictions


def _load_labels(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    rows = strict_jsonl_load(path)
    _require(len(rows) == expected_rows, "label row count drift")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, 1):
        row = dict(raw)
        _require(set(row) == LABEL_KEYS, f"label row {index} schema drift")
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id) and bool(group_id) and sample_id not in seen, f"label row {index} identity drift")
        truth = _finite(row.get("ground_truth"))
        start = _finite(row.get("scale_start"))
        end = _finite(row.get("scale_end"))
        _require(truth is not None and start is not None and end is not None and start != end, f"{sample_id}: invalid scoring labels")
        seen.add(sample_id)
        result.append({
            "sample_id": sample_id,
            "group_id": group_id,
            "ground_truth": truth,
            "scale_start": start,
            "scale_end": end,
        })
    return result


def _metric_summary(rows: Sequence[Mapping[str, Any]], method_name: str) -> dict[str, Any]:
    errors = [float(row["methods"][method_name]["normalized_error"]) for row in rows]
    successes = [bool(row["methods"][method_name]["status"]) for row in rows]
    by_group: dict[str, list[float]] = {}
    for row, error in zip(rows, errors, strict=True):
        by_group.setdefault(str(row["group_id"]), []).append(error)
    group_means = [sum(values) / len(values) for values in by_group.values()]
    ordered = sorted(errors)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "samples": len(rows),
        "physical_groups": len(by_group),
        "successful": sum(successes),
        "failures": len(rows) - sum(successes),
        "coverage": sum(successes) / len(rows),
        "full_denominator_nmae": sum(errors) / len(errors),
        "group_macro_nmae": sum(group_means) / len(group_means),
        "p95_nmae": ordered[p95_index],
        "failure_penalty_nmae": FAILURE_PENALTY,
    }


def score_once(protocol_path: Path) -> dict[str, Any]:
    """Verify the joint seal, claim scoring, then and only then open labels."""

    path, protocol, roster, predictions = _verify_sealed_predictions(protocol_path)
    ledgers = _ledger_paths(path)
    if ledgers["score_claim"].exists() or ledgers["score_complete"].exists():
        raise FileExistsError("one-shot scoring was already claimed")
    score_root = Path(protocol["run"]["root"]) / "score"
    if score_root.exists():
        raise FileExistsError(f"score output already exists: {score_root}")
    claim = {
        "schema_version": 1,
        "protocol": SCORE_CLAIM_PROTOCOL,
        "status": "claimed_after_joint_prediction_seal_before_label_access",
        "created_utc": _now(),
        "parent_protocol_sha256": sha256_file(path),
        "inference_complete_sha256": sha256_file(ledgers["inference_complete"]),
        "field_labels_opened_before_claim": False,
        "single_use": True,
    }
    _atomic_new(ledgers["score_claim"], claim)
    label_binding = protocol["labels"]
    labels_path = Path(label_binding["path"]).resolve(strict=True)
    _require(sha256_file(labels_path) == label_binding["sha256"], "label file hash drift")
    labels = _load_labels(labels_path, expected_rows=int(label_binding["rows"]))
    prediction_by_id = {str(row["sample_id"]): row for row in predictions}
    label_by_id = {str(row["sample_id"]): row for row in labels}
    _require(set(prediction_by_id) == set(label_by_id), "prediction/label sample roster drift")
    scored: list[dict[str, Any]] = []
    for sample_id in sorted(prediction_by_id):
        prediction = prediction_by_id[sample_id]
        label = label_by_id[sample_id]
        _require(prediction["group_id"] == label["group_id"], f"{sample_id}: physical group drift")
        span = abs(float(label["scale_end"]) - float(label["scale_start"]))
        method_scores: dict[str, Any] = {}
        for role in METHOD_ROLES:
            method_name = roster["methods"][role]["method_name"]
            record = prediction["methods"][method_name]
            if record["status"]:
                progress = float(record["prediction_progress"])
                predicted_start = float(record["predicted_scale_start"])
                predicted_end = float(record["predicted_scale_end"])
                predicted_reading = predicted_start + progress * (predicted_end - predicted_start)
                error = abs(predicted_reading - float(label["ground_truth"])) / span
            else:
                predicted_reading = None
                error = FAILURE_PENALTY
            method_scores[method_name] = {
                "role": role,
                "claim_tier": CLAIM_TIERS[role],
                "status": bool(record["status"]),
                "predicted_reading": predicted_reading,
                "normalized_error": float(error),
            }
        scored.append({
            "sample_id": sample_id,
            "group_id": label["group_id"],
            "ground_truth": label["ground_truth"],
            "scale_start": label["scale_start"],
            "scale_end": label["scale_end"],
            "methods": method_scores,
        })
    metrics = {
        role: {
            "method_name": roster["methods"][role]["method_name"],
            "claim_tier": CLAIM_TIERS[role],
            **_metric_summary(scored, roster["methods"][role]["method_name"]),
        }
        for role in METHOD_ROLES
    }
    score_root.mkdir(parents=True, exist_ok=False)
    scored_path = score_root / "scored_rows.jsonl"
    summary_path = score_root / "summary.json"
    seal_path = score_root / "seal.json"
    _atomic_new(scored_path, _jsonl_bytes(scored))
    summary = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "complete_one_shot_multimethod_blind_score",
        "created_utc": _now(),
        "parent_protocol_sha256": sha256_file(path),
        "score_claim_sha256": sha256_file(ledgers["score_claim"]),
        "prediction_seal_verified_before_label_access": True,
        "method_or_threshold_selection_after_result": False,
        "claim_boundaries": {
            role: CLAIM_TIERS[role] for role in METHOD_ROLES
        },
        "methods": metrics,
        "scored_rows": {
            "path": str(scored_path),
            "sha256": sha256_file(scored_path),
            "rows": len(scored),
        },
    }
    _atomic_new(summary_path, summary)
    seal = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "sealed",
        "summary_sha256": sha256_file(summary_path),
        "scored_rows_sha256": sha256_file(scored_path),
        "labels_sha256": sha256_file(labels_path),
    }
    _atomic_new(seal_path, seal)
    completion = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "complete_one_shot_multimethod_blind_score",
        "created_utc": _now(),
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "seal": {"path": str(seal_path), "sha256": sha256_file(seal_path)},
        "prediction_seal_verified_before_label_access": True,
        "no_post_result_tuning": True,
    }
    _atomic_new(ledgers["score_complete"], completion)
    return completion


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--dataset-identity", type=Path, required=True)
    freeze.add_argument(
        "--paper-results", type=Path, default=DEFAULT_PAPER_RESULTS_SUMMARY
    )
    freeze.add_argument(
        "--paper-tables-root", type=Path, default=DEFAULT_PAPER_TABLES_ROOT
    )
    freeze.add_argument("--frontend-plan", type=Path, required=True)
    freeze.add_argument("--method-roster", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--protocol", type=Path, required=True)
    run = commands.add_parser("run-once")
    run.add_argument("--protocol", type=Path, required=True)
    score = commands.add_parser("score-once")
    score.add_argument("--protocol", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze":
        result = freeze_protocol(
            dataset_identity_path=args.dataset_identity,
            paper_results_path=args.paper_results,
            paper_tables_root=args.paper_tables_root,
            frontend_plan_path=args.frontend_plan,
            method_roster_path=args.method_roster,
            output_path=args.output,
            run_root=args.run_root,
        )
    elif args.command == "preflight":
        result = static_preflight(args.protocol)
    elif args.command == "run-once":
        result = run_once(args.protocol)
    else:
        result = score_once(args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
