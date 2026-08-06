"""Fail-closed one-shot guard for the final GARC field blind test.

The guard has two deliberately different trust boundaries:

* ``freeze`` authenticates only the dataset-owner identity declaration and
  public-development artifacts.  It does not open the field manifest, labels,
  an image, or a field directory.
* ``run-once`` first creates an irreversible claim, then validates the exact
  label-free manifest, runs the frozen image-only GARC adapter, and seals every
  prediction before a score operation can be claimed.
* ``score-once`` authenticates the prediction seal and only then creates an
  irreversible score claim and opens the separately frozen label file.

The dataset-owner declaration is the authority for describing the deduplicated
1200+ image cohort as a frozen unseen blind test.  The runtime manifest is still
checked for exact hashes, an exact label-free schema, and duplicate image bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

# Support both ``python -m experiments.garc_field_blind_guard`` and the
# repository's usual direct-script invocation from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    FrozenFullAutoBundle,
    UnifiedFullAutoAdapter,
    _factory_providers,
    _load_factory,
    build_evaluator_method_binding,
    run_label_free_manifest,
    validate_label_free_manifest,
)
from experiments.v5_unified_two_stage_retest import (
    SEAL_PROTOCOL,
    assert_label_free,
    canonical_json_bytes,
    canonical_json_sha256,
    freeze_protocol as freeze_two_stage_protocol,
    score_sealed_inference,
    seal_inference,
    sha256_file,
    strict_json_load,
    strict_jsonl_load,
)


GUARD_PROTOCOL: Final[str] = "garc_field_blind_final_guard_v1"
DATASET_IDENTITY_PROTOCOL: Final[str] = "garc_field_blind_dataset_identity_v1"
FINALIZATION_PROTOCOL: Final[str] = "garc_final_public_selection_freeze_v1"
SELECTION_PROTOCOL: Final[str] = "garc_public_final_selection_summary_v1"
PAPER_RESULTS_PROTOCOL: Final[str] = "pointer_meter_paper_result_assembly_v1"
DEFAULT_PAPER_RESULTS_ROOT: Final[Path] = Path(
    r"C:\pointer_read\paper_final_results_v2"
)
DEFAULT_PAPER_RESULTS_SUMMARY: Final[Path] = DEFAULT_PAPER_RESULTS_ROOT / "summary.json"
DEFAULT_PAPER_TABLES_ROOT: Final[Path] = (
    PROJECT_ROOT / "paper/submission_mdpi/official/generated_tables"
)
INFERENCE_CLAIM_PROTOCOL: Final[str] = "garc_field_blind_inference_claim_v1"
INFERENCE_READY_PROTOCOL: Final[str] = "garc_field_blind_inference_ready_v1"
INFERENCE_COMPLETE_PROTOCOL: Final[str] = "garc_field_blind_inference_complete_v1"
SCORE_CLAIM_PROTOCOL: Final[str] = "garc_field_blind_score_claim_v1"
SCORE_COMPLETE_PROTOCOL: Final[str] = "garc_field_blind_score_complete_v1"
MINIMUM_BLIND_IMAGES: Final[int] = 1200

REQUIRED_INVENTORY_ROLES: Final[frozenset[str]] = frozenset(
    {
        "final_model_bundle",
        "provider_factory_source",
        "final_inference_config",
        "public_selection_summary",
        "paper_results_summary",
        "model_artifact",
        "component_source",
    }
)

FORBIDDEN_FIELD_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "scalemark",
        "scalemarks",
        "scalemarkpoints",
        "scalemarkgeometry",
        "manualscalestart",
        "manualscaleend",
        "manualrange",
        "groundtruthrange",
        "gtrange",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return set(value.casefold()).issubset(frozenset("0123456789abcdef"))


def _normalized_key(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _contains_exact_value(value: Any, expected: str) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_exact_value(nested, expected) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_exact_value(nested, expected) for nested in value)
    return value == expected


def _assert_no_manual_field_inputs(value: Any, *, location: str) -> None:
    """Reject GT/manual ScaleMark or physical-range values recursively."""

    assert_label_free(value, location=location)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in FORBIDDEN_FIELD_INPUT_KEYS:
                raise ValueError(
                    f"manual/GT field input key {key!r} is forbidden at {location}"
                )
            _assert_no_manual_field_inputs(
                nested,
                location=f"{location}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _assert_no_manual_field_inputs(
                nested,
                location=f"{location}[{index}]",
            )


def _atomic_new(path: Path, value: Mapping[str, Any] | bytes) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
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
    return target


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    value = strict_json_load(Path(path).resolve(strict=True))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _absolute_declared_path(value: Any, *, label: str) -> Path:
    path = Path(str(value or ""))
    _require(path.is_absolute(), f"{label} must be an absolute frozen path")
    return path.resolve(strict=False)


def _declared_file_binding(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} binding is absent")
    path = _absolute_declared_path(value.get("path"), label=f"{label}.path")
    digest = str(value.get("sha256") or "").casefold()
    _require(_is_sha256(digest), f"{label}.sha256 is invalid")
    rows = int(value.get("rows", -1))
    _require(rows >= MINIMUM_BLIND_IMAGES, f"{label}.rows is below 1200")
    return {"path": str(path), "sha256": digest, "rows": rows}


def _load_dataset_identity(path: Path) -> dict[str, Any]:
    """Read only the owner's identity declaration, never its bound data files."""

    identity_path = Path(path).resolve(strict=True)
    value = _strict_object(identity_path, label="dataset identity")
    _require(
        value.get("protocol") == DATASET_IDENTITY_PROTOCOL,
        "dataset identity protocol drift",
    )
    _require(value.get("status") == "owner_frozen", "dataset is not owner-frozen")
    _require(
        value.get("definition_authority") == "dataset_owner",
        "dataset-owner definition authority is absent",
    )
    cohort = value.get("cohort_definition")
    _require(isinstance(cohort, Mapping), "cohort definition is absent")
    declared_images = int(cohort.get("declared_images", -1))
    _require(declared_images >= MINIMUM_BLIND_IMAGES, "blind cohort has fewer than 1200 images")
    _require(cohort.get("deduplicated_before_freeze") is True, "cohort is not declared deduplicated")
    _require(
        cohort.get("declared_as_frozen_unseen_blind_test") is True,
        "cohort is not declared a frozen unseen blind test",
    )
    _require(
        cohort.get("paper_statement_authorized") is True,
        "paper cohort statement is not owner-authorized",
    )
    manifest = _declared_file_binding(
        value.get("unlabeled_manifest"),
        label="unlabeled manifest",
    )
    labels = _declared_file_binding(value.get("labels"), label="labels")
    _require(manifest["rows"] == declared_images, "manifest row declaration drift")
    _require(labels["rows"] == declared_images, "label row declaration drift")
    _require(
        Path(manifest["path"]) != Path(labels["path"]),
        "unlabeled manifest and labels must be separate files",
    )
    manifest_declaration = value["unlabeled_manifest"]
    _require(
        manifest_declaration.get("contains_labels") is False,
        "inference manifest is not declared label-free",
    )
    _require(
        manifest_declaration.get("contains_manual_or_gt_range") is False,
        "inference manifest may contain a manual/GT range",
    )
    _require(
        int(manifest_declaration.get("unique_image_sha256", -1))
        == declared_images,
        "owner declaration does not bind one unique image hash per row",
    )
    authorization = value.get("authorization")
    _require(isinstance(authorization, Mapping), "one-shot authorization is absent")
    _require(
        authorization.get("one_shot_image_inference") is True,
        "one-shot image inference is not authorized",
    )
    _require(
        authorization.get("one_shot_scoring_after_prediction_seal") is True,
        "post-seal one-shot scoring is not authorized",
    )
    _require(
        authorization.get("no_tuning_after_result") is True,
        "post-result tuning prohibition is absent",
    )
    return {
        "identity": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "cohort_definition": dict(cohort),
        "unlabeled_manifest": manifest,
        "labels": labels,
        "authorization": dict(authorization),
    }


def _inventory(finalization: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = finalization.get("frozen_files")
    _require(isinstance(rows, list) and rows, "final frozen-file inventory is absent")
    verified: list[dict[str, Any]] = []
    by_role: dict[str, list[dict[str, Any]]] = {}
    seen_paths: set[Path] = set()
    for index, raw in enumerate(rows, 1):
        _require(isinstance(raw, Mapping), f"frozen file {index} is invalid")
        _require(
            set(raw) == {"role", "path", "sha256"},
            f"frozen file {index} schema drift",
        )
        role = str(raw.get("role") or "")
        path = Path(str(raw.get("path") or "")).resolve(strict=True)
        digest = str(raw.get("sha256") or "").casefold()
        _require(role in REQUIRED_INVENTORY_ROLES or role in {"reference_detector"}, f"unsupported frozen-file role: {role}")
        _require(path not in seen_paths, f"duplicate frozen path: {path}")
        _require(_is_sha256(digest), f"invalid frozen hash: {path}")
        _require(sha256_file(path) == digest, f"frozen file changed: {path}")
        entry = {"role": role, "path": str(path), "sha256": digest}
        verified.append(entry)
        by_role.setdefault(role, []).append(entry)
        seen_paths.add(path)
    missing = REQUIRED_INVENTORY_ROLES - set(by_role)
    _require(not missing, f"final inventory lacks roles: {sorted(missing)}")
    for singleton in (
        "final_model_bundle",
        "provider_factory_source",
        "final_inference_config",
        "public_selection_summary",
        "paper_results_summary",
    ):
        _require(len(by_role[singleton]) == 1, f"{singleton} must be unique")
    return verified, by_role


def _load_public_selection(
    path: Path,
    *,
    method_name: str,
    bundle_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    value = _strict_object(path, label="public selection summary")
    _require(value.get("protocol") == SELECTION_PROTOCOL, "public selection protocol drift")
    _require(value.get("status") == "final_selected", "public selection is not final")
    _require(value.get("method_name") == method_name, "selected method name drift")
    _require(value.get("selected_bundle_sha256") == bundle_sha256, "selected bundle drift")
    _require(value.get("selected_config_sha256") == config_sha256, "selected config drift")
    _require(value.get("selection_complete") is True, "public selection is incomplete")
    _require(value.get("selection_gate_passed") is True, "public selection gate did not pass")
    _require(value.get("public_data_only") is True, "selection was not public-data-only")
    _require(value.get("field_manifest_opened") is False, "field manifest influenced selection")
    _require(value.get("field_images_opened") is False, "field images influenced selection")
    _require(value.get("field_labels_opened") is False, "field labels influenced selection")
    _require(value.get("no_post_selection_tuning") is True, "post-selection tuning is allowed")
    return value


def _load_paper_results(path: Path) -> dict[str, Any]:
    """Authenticate the completed public-only paper evidence bundle.

    The final field authorization is intentionally downstream of the paper
    result assembly.  This prevents an unfinished GARC/public comparison from
    being silently repaired after the blind-test result becomes visible.
    """

    value = _strict_object(path, label="paper results summary")
    _require(
        value.get("protocol") == PAPER_RESULTS_PROTOCOL,
        "paper results protocol drift",
    )
    _require(value.get("status") == "complete", "paper results are incomplete")
    audit = value.get("audit")
    _require(isinstance(audit, Mapping), "paper results audit is absent")
    for key in (
        "restricted_namespace_artifacts_opened",
        "field_samples_opened",
        "test_samples_opened",
        "sealed_samples_opened",
        "confirmatory_samples_opened",
    ):
        _require(int(audit.get(key, -1)) == 0, f"paper results audit is not public-only: {key}")
    _require(
        audit.get("manual_or_ground_truth_numeric_range_used_by_model") is False,
        "paper results used manual/GT numeric range during inference",
    )
    _require(
        audit.get("all_prediction_and_score_seals_verified_before_public_truth_opened")
        is True,
        "paper result prediction/score chronology is unverified",
    )
    _require(
        audit.get("bottom_up_metrics_match_all_sealed_summaries") is True,
        "paper result metrics are not fully reconciled",
    )
    _require(audit.get("inference_started") is False, "paper assembly reran inference")
    _require(audit.get("training_started") is False, "paper assembly reran training")
    return value


def load_paper_authority(
    summary_path: Path,
    *,
    rendered_tables_root: Path = DEFAULT_PAPER_TABLES_ROOT,
) -> dict[str, Any]:
    """Authenticate the v2 result root and separately rendered table seal."""

    summary_file = Path(summary_path).resolve(strict=True)
    summary = _load_paper_results(summary_file)
    result_seal_path = summary_file.with_name("seal.json").resolve(strict=True)
    result_seal = _strict_object(result_seal_path, label="paper result seal")
    _require(
        result_seal.get("protocol") == PAPER_RESULTS_PROTOCOL
        and result_seal.get("status") == "sealed",
        "paper result seal protocol/status drift",
    )
    artifacts = result_seal.get("artifacts") or {}
    _require(
        (artifacts.get("summary") or {}).get("sha256") == sha256_file(summary_file),
        "paper result seal does not bind summary",
    )
    tables_root = Path(rendered_tables_root).resolve(strict=True)
    manifest_path = (tables_root / "manifest.json").resolve(strict=True)
    table_seal_path = (tables_root / "seal.json").resolve(strict=True)
    manifest = _strict_object(manifest_path, label="rendered paper table manifest")
    table_seal = _strict_object(table_seal_path, label="rendered paper table seal")
    _require(
        manifest.get("protocol") == "paper_result_latex_tables_v1"
        and manifest.get("status") == "complete",
        "rendered table manifest protocol/status drift",
    )
    _require(
        table_seal.get("protocol") == "paper_result_latex_tables_v1"
        and table_seal.get("status") == "sealed",
        "rendered table seal protocol/status drift",
    )
    source = manifest.get("source") or {}
    _require(
        Path(str(source.get("path") or "")).resolve(strict=True)
        == summary_file.parent,
        "rendered tables bind the wrong paper result root",
    )
    _require(
        source.get("summary_sha256") == sha256_file(summary_file)
        and source.get("seal_sha256") == sha256_file(result_seal_path),
        "rendered tables source binding drift",
    )
    _require(
        ((table_seal.get("artifacts") or {}).get("manifest") or {}).get("sha256")
        == sha256_file(manifest_path),
        "rendered table seal does not bind manifest",
    )
    return {
        "summary": {"path": str(summary_file), "sha256": sha256_file(summary_file)},
        "result_seal": {
            "path": str(result_seal_path),
            "sha256": sha256_file(result_seal_path),
        },
        "tables_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "tables_seal": {
            "path": str(table_seal_path),
            "sha256": sha256_file(table_seal_path),
        },
        "summary_value": summary,
    }


def _load_finalization(path: Path) -> dict[str, Any]:
    finalization_path = Path(path).resolve(strict=True)
    value = _strict_object(finalization_path, label="GARC finalization")
    _require(value.get("protocol") == FINALIZATION_PROTOCOL, "finalization protocol drift")
    _require(value.get("status") == "final_frozen", "GARC finalization is not frozen")
    files, by_role = _inventory(value)
    bundle_entry = by_role["final_model_bundle"][0]
    bundle = FrozenFullAutoBundle.load(Path(bundle_entry["path"]))
    _require(bundle.execution_mode == EXECUTION_FORMAL, "final GARC bundle is not formal")
    _require("garc" in bundle.method_name.casefold(), "final method is not GARC")
    _require(value.get("method_name") == bundle.method_name, "final method name drift")

    factory_entry = by_role["provider_factory_source"][0]
    _require(
        factory_entry["sha256"] == bundle.descriptor["factory_source_sha256"],
        "provider factory differs from the final bundle",
    )
    adapter_hash = sha256_file(Path(__file__).with_name("v5_unified_full_auto_adapter.py"))
    _require(
        adapter_hash == bundle.descriptor["adapter_source_sha256"],
        "full-auto adapter differs from the final bundle",
    )

    config_entry = by_role["final_inference_config"][0]
    config = _strict_object(Path(config_entry["path"]), label="final inference config")
    _assert_no_manual_field_inputs(config, location="final_inference_config")
    _require(
        _contains_exact_value(bundle.descriptor, config_entry["sha256"]),
        "final inference config hash is not bound into the executable model bundle",
    )
    contract = value.get("inference_contract")
    _require(isinstance(contract, Mapping), "final inference contract is absent")
    for key in (
        "caller_range_allowed",
        "caller_scalemark_allowed",
        "caller_geometry_allowed",
        "caller_reference_allowed",
        "ground_truth_available",
    ):
        _require(contract.get(key) is False, f"forbidden inference contract enabled: {key}")
    _require(contract.get("input") == "image_only", "final inference is not image-only")
    _require(
        contract.get("outputs")
        == [
            "prediction_progress",
            "predicted_scale_start",
            "predicted_scale_end",
            "range_confidence",
        ],
        "final GARC output contract drift",
    )

    selection_entry = by_role["public_selection_summary"][0]
    selection = _load_public_selection(
        Path(selection_entry["path"]),
        method_name=bundle.method_name,
        bundle_sha256=bundle.bundle_sha256,
        config_sha256=config_entry["sha256"],
    )
    declared_selection = value.get("public_selection_summary")
    _require(isinstance(declared_selection, Mapping), "selection binding is absent")
    _require(
        str(Path(str(declared_selection.get("path") or "")).resolve(strict=True))
        == selection_entry["path"]
        and declared_selection.get("sha256") == selection_entry["sha256"],
        "selection binding differs from frozen inventory",
    )

    paper_entry = by_role["paper_results_summary"][0]
    paper = _load_paper_results(Path(paper_entry["path"]))
    declared_paper = value.get("paper_results_summary")
    _require(isinstance(declared_paper, Mapping), "paper results binding is absent")
    _require(
        str(Path(str(declared_paper.get("path") or "")).resolve(strict=True))
        == paper_entry["path"]
        and declared_paper.get("sha256") == paper_entry["sha256"],
        "paper results binding differs from frozen inventory",
    )

    inventory_hashes = {entry["sha256"] for entry in files}
    for component in (bundle.progress_binding, bundle.range_binding):
        _require(component.frozen, f"{component.name} is not frozen")
        _require(component.verified_complete, f"{component.name} is incomplete")
        _require(not component.synthetic, f"{component.name} is synthetic")
        _require(bool(component.source_sha256), f"{component.name} has no source hashes")
        missing_artifacts = set(component.artifact_sha256.values()) - inventory_hashes
        missing_sources = set(component.source_sha256.values()) - inventory_hashes
        _require(not missing_artifacts, f"{component.name} artifact hashes are not file-bound")
        _require(not missing_sources, f"{component.name} source hashes are not file-bound")
    detector_hash = bundle.reference_detector_sha256
    if detector_hash is not None:
        _require(detector_hash in inventory_hashes, "reference detector is not file-bound")

    return {
        "finalization": {
            "path": str(finalization_path),
            "sha256": sha256_file(finalization_path),
        },
        "method_name": bundle.method_name,
        "bundle": bundle_entry,
        "factory": factory_entry,
        "config": config_entry,
        "selection": selection_entry,
        "paper_results": paper_entry,
        "frozen_files": files,
        "selection_evidence": {
            "selection_complete": selection["selection_complete"],
            "selection_gate_passed": selection["selection_gate_passed"],
            "public_data_only": selection["public_data_only"],
            "field_manifest_opened": selection["field_manifest_opened"],
            "field_images_opened": selection["field_images_opened"],
            "field_labels_opened": selection["field_labels_opened"],
            "paper_results_complete": paper["status"] == "complete",
        },
    }


def freeze_finalization(
    *,
    bundle_path: Path,
    factory_path: Path,
    config_path: Path,
    public_selection_path: Path,
    paper_results_path: Path,
    model_artifacts: Sequence[Path],
    component_sources: Sequence[Path],
    output_path: Path,
    reference_detector_path: Path | None = None,
) -> dict[str, Any]:
    """Create the public-only final GARC record consumed by ``freeze_guard``."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"GARC finalization is immutable: {output}")
    bundle_file = Path(bundle_path).resolve(strict=True)
    factory_file = Path(factory_path).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    selection_file = Path(public_selection_path).resolve(strict=True)
    paper_file = Path(paper_results_path).resolve(strict=True)
    bundle = FrozenFullAutoBundle.load(bundle_file)
    _require(bundle.execution_mode == EXECUTION_FORMAL, "final GARC bundle is not formal")
    _require("garc" in bundle.method_name.casefold(), "final method is not GARC")
    _require(
        sha256_file(factory_file) == bundle.descriptor["factory_source_sha256"],
        "provider factory differs from final bundle",
    )
    config = _strict_object(config_file, label="final inference config")
    _assert_no_manual_field_inputs(config, location="final_inference_config")
    _load_paper_results(paper_file)
    _require(
        _contains_exact_value(bundle.descriptor, sha256_file(config_file)),
        "final inference config hash is not bound into the executable model bundle",
    )

    role_paths: list[tuple[str, Path]] = [
        ("final_model_bundle", bundle_file),
        ("provider_factory_source", factory_file),
        ("final_inference_config", config_file),
        ("public_selection_summary", selection_file),
        ("paper_results_summary", paper_file),
        *(('model_artifact', Path(path).resolve(strict=True)) for path in model_artifacts),
        *(('component_source', Path(path).resolve(strict=True)) for path in component_sources),
    ]
    if reference_detector_path is not None:
        role_paths.append(
            ("reference_detector", Path(reference_detector_path).resolve(strict=True))
        )
    _require(bool(model_artifacts), "at least one model artifact is required")
    _require(bool(component_sources), "at least one component source is required")
    paths = [path for _, path in role_paths]
    _require(len(set(paths)) == len(paths), "one frozen file was assigned multiple roles")
    payload = {
        "schema_version": 1,
        "protocol": FINALIZATION_PROTOCOL,
        "status": "final_frozen",
        "created_utc": _now(),
        "method_name": bundle.method_name,
        "public_selection_summary": {
            "path": str(selection_file),
            "sha256": sha256_file(selection_file),
        },
        "paper_results_summary": {
            "path": str(paper_file),
            "sha256": sha256_file(paper_file),
        },
        "inference_contract": {
            "input": "image_only",
            "caller_range_allowed": False,
            "caller_scalemark_allowed": False,
            "caller_geometry_allowed": False,
            "caller_reference_allowed": False,
            "ground_truth_available": False,
            "outputs": [
                "prediction_progress",
                "predicted_scale_start",
                "predicted_scale_end",
                "range_confidence",
            ],
        },
        "frozen_files": [
            {"role": role, "path": str(path), "sha256": sha256_file(path)}
            for role, path in role_paths
        ],
    }
    _atomic_new(output, payload)
    try:
        _load_finalization(output)
    except Exception:
        # The immutable file intentionally remains as evidence of a failed
        # finalization attempt.  A corrected attempt must use a new path.
        raise
    return payload


def freeze_guard(
    *,
    dataset_identity_path: Path,
    finalization_path: Path,
    output_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Freeze the final one-shot guard without reading any bound field data."""

    output = Path(output_path).resolve()
    run = Path(run_root).resolve()
    if output.exists():
        raise FileExistsError(f"guard freeze is immutable: {output}")
    if run.exists():
        raise FileExistsError(f"one-shot run root already exists: {run}")
    dataset = _load_dataset_identity(dataset_identity_path)
    finalization = _load_finalization(finalization_path)
    source = Path(__file__).resolve()
    full_auto_source = source.with_name("v5_unified_full_auto_adapter.py")
    two_stage_source = source.with_name("v5_unified_two_stage_retest.py")
    payload = {
        "schema_version": 1,
        "protocol": GUARD_PROTOCOL,
        "status": "frozen_authorized_not_started",
        "created_utc": _now(),
        "guard_location": str(output),
        "cohort_definition": dataset["cohort_definition"],
        "dataset_identity": dataset["identity"],
        "unlabeled_manifest": dataset["unlabeled_manifest"],
        "labels": dataset["labels"],
        "authorization": dataset["authorization"],
        "final_method": finalization,
        "run": {
            "root": str(run),
            "output_root_must_not_preexist": True,
            "one_shot_claim_is_outside_run_root": True,
            "retry_after_claim": "forbidden_without_separate_adjudication_protocol",
        },
        "chronology": {
            "freeze_reads_dataset_identity_only": True,
            "field_manifest_opened_during_freeze": False,
            "field_labels_opened_during_freeze": False,
            "field_images_opened_during_freeze": False,
            "claim_precedes_manifest_validation": True,
            "manifest_validation_precedes_image_io": True,
            "predictions_sealed_before_labels": True,
            "labels_opened_only_by_score_once": True,
        },
        "runtime_sources": {
            "guard": {"path": str(source), "sha256": sha256_file(source)},
            "full_auto_adapter": {
                "path": str(full_auto_source),
                "sha256": sha256_file(full_auto_source),
            },
            "two_stage_protocol": {
                "path": str(two_stage_source),
                "sha256": sha256_file(two_stage_source),
            },
        },
    }
    _atomic_new(output, payload)
    return payload


def _ledger_paths(guard_path: Path) -> dict[str, Path]:
    guard = Path(guard_path).resolve(strict=False)
    prefix = guard.with_suffix("")
    return {
        "inference_claim": prefix.with_name(prefix.name + ".inference-claim.json"),
        "inference_ready": prefix.with_name(prefix.name + ".inference-ready.json"),
        "inference_complete": prefix.with_name(prefix.name + ".inference-complete.json"),
        "score_claim": prefix.with_name(prefix.name + ".score-claim.json"),
        "score_complete": prefix.with_name(prefix.name + ".score-complete.json"),
    }


def _verify_guard(guard_path: Path) -> tuple[dict[str, Any], str]:
    path = Path(guard_path).resolve(strict=True)
    guard = _strict_object(path, label="field blind guard")
    _require(guard.get("protocol") == GUARD_PROTOCOL, "field guard protocol drift")
    _require(guard.get("status") == "frozen_authorized_not_started", "field guard status drift")
    _require(
        Path(str(guard.get("guard_location") or "")).resolve(strict=False) == path,
        "field guard was copied or moved away from its frozen ledger location",
    )
    guard_hash = sha256_file(path)
    for label, binding in (guard.get("runtime_sources") or {}).items():
        bound = Path(str(binding.get("path") or "")).resolve(strict=True)
        _require(sha256_file(bound) == binding.get("sha256"), f"runtime source changed: {label}")
    dataset_binding = guard.get("dataset_identity") or {}
    identity_path = Path(str(dataset_binding.get("path") or "")).resolve(strict=True)
    _require(sha256_file(identity_path) == dataset_binding.get("sha256"), "dataset identity changed")
    dataset = _load_dataset_identity(identity_path)
    for key in ("unlabeled_manifest", "labels"):
        _require(dataset[key] == guard.get(key), f"dataset {key} declaration drift")
    final_binding = (guard.get("final_method") or {}).get("finalization") or {}
    final_path = Path(str(final_binding.get("path") or "")).resolve(strict=True)
    _require(sha256_file(final_path) == final_binding.get("sha256"), "GARC finalization changed")
    finalization = _load_finalization(final_path)
    _require(finalization == guard.get("final_method"), "final GARC binding drift")
    return guard, guard_hash


def static_preflight(guard_path: Path) -> dict[str, Any]:
    """Verify public/freeze identities without opening field manifest or labels."""

    path = Path(guard_path).resolve(strict=True)
    guard, digest = _verify_guard(path)
    ledgers = _ledger_paths(path)
    return {
        "schema_version": 1,
        "protocol": GUARD_PROTOCOL,
        "status": "validated_without_field_data_access",
        "guard_sha256": digest,
        "method_name": guard["final_method"]["method_name"],
        "declared_images": guard["cohort_definition"]["declared_images"],
        "field_manifest_opened": False,
        "field_labels_opened": False,
        "field_images_opened": False,
        "inference_claim_exists": ledgers["inference_claim"].exists(),
        "inference_complete_exists": ledgers["inference_complete"].exists(),
        "score_complete_exists": ledgers["score_complete"].exists(),
    }


def _validate_runtime_manifest(
    guard: Mapping[str, Any],
) -> tuple[Path, list[Any], dict[str, Any]]:
    binding = guard["unlabeled_manifest"]
    manifest = Path(binding["path"]).resolve(strict=True)
    _require(sha256_file(manifest) == binding["sha256"], "field manifest hash drift")
    rows = strict_jsonl_load(manifest)
    _assert_no_manual_field_inputs(rows, location="field_unlabeled_manifest")
    items = validate_label_free_manifest(rows)
    _require(len(items) == int(binding["rows"]), "field manifest row count drift")
    image_hashes = [item.image_sha256 for item in items]
    image_paths = [item.image_path for item in items]
    _require(len(set(image_hashes)) == len(items), "field manifest contains duplicate image bytes")
    _require(len(set(image_paths)) == len(items), "field manifest repeats an image path")
    identity = {
        "rows": len(items),
        "unique_image_sha256": len(set(image_hashes)),
        "sample_ids_sha256": canonical_json_sha256(
            sorted(item.sample_id for item in items)
        ),
        "sample_image_pairs_sha256": canonical_json_sha256(
            sorted([item.sample_id, item.image_sha256] for item in items)
        ),
    }
    return manifest, items, identity


def _claim_and_validate_manifest(
    *,
    guard_path: Path,
) -> tuple[dict[str, Any], str, Path, dict[str, Any]]:
    guard_file = Path(guard_path).resolve(strict=True)
    guard, guard_hash = _verify_guard(guard_file)
    ledgers = _ledger_paths(guard_file)
    run_root = Path(guard["run"]["root"])
    if run_root.exists():
        raise FileExistsError(f"one-shot run root already exists: {run_root}")
    for path in ledgers.values():
        if path.exists():
            raise FileExistsError(f"one-shot ledger already exists: {path}")
    claim = {
        "schema_version": 1,
        "protocol": INFERENCE_CLAIM_PROTOCOL,
        "status": "claimed_before_field_manifest_or_image_access",
        "created_utc": _now(),
        "guard": {"path": str(guard_file), "sha256": guard_hash},
        "field_manifest_opened_before_claim": False,
        "field_labels_opened_before_claim": False,
        "field_images_opened_before_claim": False,
        "single_use": True,
    }
    _atomic_new(ledgers["inference_claim"], claim)
    manifest, items, identity = _validate_runtime_manifest(guard)
    ready = {
        "schema_version": 1,
        "protocol": INFERENCE_READY_PROTOCOL,
        "status": "ready_after_label_free_manifest_validation",
        "created_utc": _now(),
        "guard_sha256": guard_hash,
        "claim_sha256": sha256_file(ledgers["inference_claim"]),
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            **identity,
        },
        "field_labels_opened": False,
        "field_images_opened": False,
        "manual_or_gt_range_fields_present": False,
        "deduplicated_exact_image_hashes": True,
    }
    _atomic_new(ledgers["inference_ready"], ready)
    return guard, guard_hash, manifest, ready


def run_once(guard_path: Path) -> dict[str, Any]:
    """Claim, run the exact final GARC model, and seal predictions once."""

    guard_file = Path(guard_path).resolve(strict=True)
    guard, guard_hash, manifest, ready = _claim_and_validate_manifest(
        guard_path=guard_file
    )
    ledgers = _ledger_paths(guard_file)
    run_root = Path(guard["run"]["root"])
    run_root.mkdir(parents=True, exist_ok=False)
    method_bundle_path = run_root / "evaluator_method_bundle.json"
    protocol_path = run_root / "two_stage_protocol.json"
    raw_path = run_root / "raw_predictions.jsonl"
    sealed_path = run_root / "sealed_predictions.jsonl"
    seal_path = run_root / "prediction_seal.json"

    final = guard["final_method"]
    bundle_path = Path(final["bundle"]["path"])
    bundle = FrozenFullAutoBundle.load(bundle_path)
    detector_path: Path | None = None
    if bundle.reference_detector_sha256 is not None:
        matches = [
            Path(entry["path"])
            for entry in final["frozen_files"]
            if entry["sha256"] == bundle.reference_detector_sha256
        ]
        _require(len(matches) == 1, "reference detector binding is ambiguous")
        detector_path = matches[0]
    method_name, binding = build_evaluator_method_binding(
        bundle_path=bundle_path,
        reference_detector_path=detector_path,
    )
    _require(method_name == final["method_name"], "runtime method name drift")
    _atomic_new(method_bundle_path, {"methods": {method_name: binding}})
    freeze_two_stage_protocol(
        unlabeled_manifest_path=manifest,
        method_bundle_path=method_bundle_path,
        output_path=protocol_path,
    )

    factory, factory_hash = _load_factory(Path(final["factory"]["path"]), "build_full_auto_providers")
    _require(factory_hash == final["factory"]["sha256"], "provider factory hash drift")
    progress_provider, range_pipeline = _factory_providers(factory, bundle)
    adapter = UnifiedFullAutoAdapter(
        progress_provider=progress_provider,
        automatic_numeric_range_pipeline=range_pipeline,
        bundle=bundle,
    )
    run_metadata = run_label_free_manifest(
        input_path=manifest,
        output_path=raw_path,
        adapter=adapter,
    )
    _require(run_metadata.get("label_files_opened") == 0, "inference reports label access")
    _require(run_metadata.get("range_values_supplied_by_caller") == 0, "caller range reached inference")
    seal = seal_inference(
        protocol_path=protocol_path,
        unlabeled_manifest_path=manifest,
        raw_predictions_path=raw_path,
        sealed_predictions_path=sealed_path,
        seal_path=seal_path,
    )
    _require(seal.get("protocol") == SEAL_PROTOCOL, "prediction seal protocol drift")
    _require(seal.get("label_files_opened") == 0, "prediction seal reports label access")
    completion = {
        "schema_version": 1,
        "protocol": INFERENCE_COMPLETE_PROTOCOL,
        "status": "complete_predictions_sealed_before_labels",
        "created_utc": _now(),
        "guard_sha256": guard_hash,
        "claim_sha256": sha256_file(ledgers["inference_claim"]),
        "ready_sha256": sha256_file(ledgers["inference_ready"]),
        "rows": ready["manifest"]["rows"],
        "field_labels_opened": False,
        "manual_or_gt_range_supplied": False,
        "artifacts": {
            "evaluator_method_bundle": {
                "path": str(method_bundle_path),
                "sha256": sha256_file(method_bundle_path),
            },
            "two_stage_protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "raw_predictions": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "raw_predictions_metadata": {
                "path": str(raw_path.with_suffix(".metadata.json")),
                "sha256": sha256_file(raw_path.with_suffix(".metadata.json")),
            },
            "sealed_predictions": {"path": str(sealed_path), "sha256": sha256_file(sealed_path)},
            "prediction_seal": {"path": str(seal_path), "sha256": sha256_file(seal_path)},
        },
    }
    _atomic_new(ledgers["inference_complete"], completion)
    return completion


def _verify_inference_complete(
    guard_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Path]]:
    guard_file = Path(guard_path).resolve(strict=True)
    guard, guard_hash = _verify_guard(guard_file)
    ledgers = _ledger_paths(guard_file)
    for name in ("inference_claim", "inference_ready", "inference_complete"):
        _require(ledgers[name].is_file(), f"required ledger is absent: {name}")
    complete = _strict_object(ledgers["inference_complete"], label="inference completion")
    _require(complete.get("protocol") == INFERENCE_COMPLETE_PROTOCOL, "completion protocol drift")
    _require(complete.get("status") == "complete_predictions_sealed_before_labels", "inference is incomplete")
    _require(complete.get("guard_sha256") == guard_hash, "completion guard drift")
    _require(complete.get("field_labels_opened") is False, "completion reports label access")
    _require(complete.get("manual_or_gt_range_supplied") is False, "completion reports a caller range")
    artifact_paths: dict[str, Path] = {}
    for name, binding in (complete.get("artifacts") or {}).items():
        path = Path(str(binding.get("path") or "")).resolve(strict=True)
        _require(sha256_file(path) == binding.get("sha256"), f"completed artifact changed: {name}")
        artifact_paths[str(name)] = path
    _require(
        set(artifact_paths)
        == {
            "evaluator_method_bundle",
            "two_stage_protocol",
            "raw_predictions",
            "raw_predictions_metadata",
            "sealed_predictions",
            "prediction_seal",
        },
        "completion artifact roster drift",
    )
    seal = _strict_object(artifact_paths["prediction_seal"], label="prediction seal")
    _require(seal.get("protocol") == SEAL_PROTOCOL, "prediction seal protocol drift")
    _require(seal.get("status") == "complete_label_free_predictions_sealed", "predictions are not sealed")
    _require(seal.get("label_files_opened") == 0, "seal reports label access")
    _require(
        sha256_file(artifact_paths["sealed_predictions"])
        == seal.get("sealed_predictions_sha256"),
        "sealed prediction hash drift",
    )
    return guard, guard_hash, complete, artifact_paths


def score_once(guard_path: Path) -> dict[str, Any]:
    """Open labels exactly once, but only after the prediction seal verifies."""

    guard_file = Path(guard_path).resolve(strict=True)
    guard, guard_hash, complete, artifacts = _verify_inference_complete(guard_file)
    ledgers = _ledger_paths(guard_file)
    if ledgers["score_claim"].exists() or ledgers["score_complete"].exists():
        raise FileExistsError("one-shot score has already been claimed")
    score_root = Path(guard["run"]["root"]) / "score"
    if score_root.exists():
        raise FileExistsError(f"score output already exists: {score_root}")
    claim = {
        "schema_version": 1,
        "protocol": SCORE_CLAIM_PROTOCOL,
        "status": "claimed_after_prediction_seal_before_label_access",
        "created_utc": _now(),
        "guard_sha256": guard_hash,
        "inference_complete_sha256": sha256_file(ledgers["inference_complete"]),
        "prediction_seal_verified": True,
        "field_labels_opened_before_claim": False,
        "single_use": True,
    }
    _atomic_new(ledgers["score_claim"], claim)

    label_binding = guard["labels"]
    labels = Path(label_binding["path"]).resolve(strict=True)
    _require(sha256_file(labels) == label_binding["sha256"], "field label hash drift")
    result = score_sealed_inference(
        protocol_path=artifacts["two_stage_protocol"],
        sealed_predictions_path=artifacts["sealed_predictions"],
        seal_path=artifacts["prediction_seal"],
        labels_path=labels,
        output_dir=score_root,
    )
    summary_path = score_root / "summary.json"
    _require(summary_path.is_file(), "score summary was not created")
    completion = {
        "schema_version": 1,
        "protocol": SCORE_COMPLETE_PROTOCOL,
        "status": "complete_one_shot_blind_score",
        "created_utc": _now(),
        "guard_sha256": guard_hash,
        "score_claim_sha256": sha256_file(ledgers["score_claim"]),
        "inference_complete_sha256": sha256_file(ledgers["inference_complete"]),
        "labels": {"path": str(labels), "sha256": sha256_file(labels)},
        "prediction_seal_verified_before_label_access": True,
        "no_tuning_after_result": True,
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "cohort": result.get("cohort"),
    }
    _atomic_new(ledgers["score_complete"], completion)
    return completion


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("freeze-finalization")
    finalize.add_argument("--bundle", type=Path, required=True)
    finalize.add_argument("--factory", type=Path, required=True)
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--public-selection", type=Path, required=True)
    finalize.add_argument(
        "--paper-results", type=Path, default=DEFAULT_PAPER_RESULTS_SUMMARY
    )
    finalize.add_argument("--model-artifact", type=Path, action="append", required=True)
    finalize.add_argument("--component-source", type=Path, action="append", required=True)
    finalize.add_argument("--reference-detector", type=Path)
    finalize.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--dataset-identity", type=Path, required=True)
    freeze.add_argument("--finalization", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--guard", type=Path, required=True)
    run = commands.add_parser("run-once")
    run.add_argument("--guard", type=Path, required=True)
    score = commands.add_parser("score-once")
    score.add_argument("--guard", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze-finalization":
        result = freeze_finalization(
            bundle_path=args.bundle,
            factory_path=args.factory,
            config_path=args.config,
            public_selection_path=args.public_selection,
            paper_results_path=args.paper_results,
            model_artifacts=args.model_artifact,
            component_sources=args.component_source,
            reference_detector_path=args.reference_detector,
            output_path=args.output,
        )
    elif args.command == "freeze":
        result = freeze_guard(
            dataset_identity_path=args.dataset_identity,
            finalization_path=args.finalization,
            output_path=args.output,
            run_root=args.run_root,
        )
    elif args.command == "preflight":
        result = static_preflight(args.guard)
    elif args.command == "run-once":
        result = run_once(args.guard)
    else:
        result = score_once(args.guard)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
