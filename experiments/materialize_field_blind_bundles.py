"""Materialize the five frozen, image-only field-blind method bundles.

This module is intentionally metadata-only.  It accepts exact, hash-bound
public/model artifacts and never accepts a dataset path, image manifest, label
file, crop, ScaleMark, or numeric range.  The output can therefore be prepared
before the one-shot field protocol is authorized.

The formal workflow has three immutable stages:

``freeze-spec``
    Authenticate the completed public evidence and freeze every exact input.
``materialize``
    Rebind existing verified providers to the one audited runtime factory and
    write five :class:`FrozenFullAutoBundle` files plus the exact method roster
    consumed by :mod:`experiments.field_blind_multimethod`.
``verify``
    Recompute all hashes and independently validate the resulting roster.

No provider is invented here.  Progress and range identities are copied only
from already formal source bundles.  GARC's source bundle must additionally be
the runtime bundle sealed by the future GARC summary.  PEPD, VDN, and the
original Transformer keep their own frozen progress components but reuse the
exact GARC automatic-range binding.  V5 retains its complete, separately frozen
range stack because it is the internal end-to-end ablation rather than another
progress-backbone control.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.field_blind_multimethod import (
    CLAIM_TIERS,
    METHOD_NAME_TOKENS,
    METHOD_ROLES,
    ROSTER_PROTOCOL,
    load_frontend_plan,
    load_method_roster,
)
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    EVIDENCE_ROLE_PRIMARY,
    OUTPUT_MODE_FULL_READING,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    sha256_file,
)
from experiments.v5_unified_two_stage_retest import (
    assert_label_free,
    canonical_json_bytes,
    canonical_json_sha256,
    strict_json_load,
)


SPEC_PROTOCOL: Final[str] = "field_blind_bundle_materialization_spec_v1"
MATERIALIZATION_PROTOCOL: Final[str] = (
    "field_blind_full_reading_bundle_materialization_v1"
)
PREFLIGHT_PROTOCOL: Final[str] = "field_blind_bundle_materialization_preflight_v1"
RUNTIME_FACTORY_PROTOCOL: Final[str] = "field_blind_runtime_factory_v1"
FACTORY_FUNCTION: Final[str] = "build_field_blind_full_auto_providers"
GARC_SUMMARY_PROTOCOL: Final[str] = "garc_full_auto_public_event_chain_summary_v1"
PAPER_RESULTS_PROTOCOL: Final[str] = "pointer_meter_paper_result_assembly_v1"

ROLE_METHOD_NAMES: Final[dict[str, str]] = {
    "garc_final": "GARC-final+auto-ref",
    "v5_complete": "V5-complete+auto-ref",
    "pepd_shared_range": "PEPD-shared-range+auto-ref",
    "vdn_shared_range": "VDN-shared-range+auto-ref",
    "transformer_shared_range": "Original-Transformer-shared-range",
}
EXPECTED_FACTORY_MODES: Final[dict[str, str]] = {
    "garc_final": "garc_plan",
    "v5_complete": "v5_plan",
    "pepd_shared_range": "shared_garc_range",
    "vdn_shared_range": "shared_garc_range",
    "transformer_shared_range": "shared_garc_range",
}
EXPECTED_PROGRESS_KINDS: Final[dict[str, str]] = {
    "pepd_shared_range": "external_progress_factory",
    "vdn_shared_range": "external_progress_factory",
    "transformer_shared_range": "original_transformer",
}
FORBIDDEN_EXTERNAL_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "field",
        "test",
        "sealed",
        "confirmatory",
        "confirmation",
        "xiangmu1",
        "xiangmu2",
    }
)


class MaterializationError(ValueError):
    """Fail-closed metadata/materialization error."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.casefold()).issubset(frozenset("0123456789abcdef"))
    )


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in Path(path).parts:
        tokens.update(
            token
            for token in re.split(r"[\\/_ .-]+", str(part).casefold())
            if token
        )
    return tokens


def _guard_metadata_path(path: Path, *, label: str) -> Path:
    """Reject restricted external namespaces before any filesystem access.

    Repository source files and synthetic temporary fixtures are allowed.  The
    restriction applies to the external artifact root where the protected
    datasets live.  The function deliberately does not call ``resolve`` or
    ``exists`` until after the lexical check.
    """

    raw = Path(path)
    absolute = Path(os.path.abspath(str(raw)))
    pointer_root = Path(r"C:\pointer_read")
    try:
        in_pointer_root = os.path.commonpath(
            [str(absolute).casefold(), str(pointer_root).casefold()]
        ) == str(pointer_root).casefold()
    except ValueError:
        in_pointer_root = False
    if in_pointer_root:
        forbidden = sorted(_path_tokens(absolute) & FORBIDDEN_EXTERNAL_PATH_TOKENS)
        _require(
            not forbidden,
            f"{label} enters a restricted external namespace: {forbidden}",
        )
    return absolute


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    guarded = _guard_metadata_path(path, label=label)
    value = strict_json_load(guarded.resolve(strict=True))
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _binding(
    value: Any,
    *,
    label: str,
    must_exist: bool = True,
    path_type: str = "file",
) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{label} binding is absent")
    raw = Path(str(value.get("path") or ""))
    _require(raw.is_absolute(), f"{label}.path must be absolute")
    path = _guard_metadata_path(raw, label=label)
    digest = str(value.get("sha256") or "").casefold()
    _require(_is_sha256(digest), f"{label}.sha256 is invalid")
    if must_exist:
        path = path.resolve(strict=True)
        if path_type == "file":
            _require(path.is_file(), f"{label} is not a file: {path}")
            _require(sha256_file(path) == digest, f"{label} hash drift")
        elif path_type == "directory":
            _require(path.is_dir(), f"{label} is not a directory: {path}")
        else:
            raise MaterializationError(f"unsupported binding path type: {path_type}")
    return {"path": str(path), "sha256": digest}


def _bind_file(path: Path, *, label: str) -> dict[str, str]:
    guarded = _guard_metadata_path(path, label=label).resolve(strict=True)
    _require(guarded.is_file(), f"{label} is not a file: {guarded}")
    return {"path": str(guarded), "sha256": sha256_file(guarded)}


def _atomic_new(path: Path, value: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    payload = canonical_json_bytes(value)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _factory_has_entrypoint(path: Path) -> None:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    _require(FACTORY_FUNCTION in functions, "runtime factory entrypoint is absent")


def _resolve_relative_binding(
    value: Any, *, root: Path, label: str
) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{label} binding is absent")
    raw = Path(str(value.get("path") or ""))
    path = raw if raw.is_absolute() else root / raw
    guarded = _guard_metadata_path(path, label=label)
    return _binding(
        {"path": str(guarded), "sha256": value.get("sha256")},
        label=label,
    )


def _load_frontend_guarded(
    binding: Mapping[str, str],
) -> tuple[Path, dict[str, Any]]:
    """Guard the frontend's exact linked files before its normal verifier opens them."""

    raw = _strict_object(Path(binding["path"]), label="shared frontend plan")
    _binding(raw.get("detector_checkpoint"), label="shared frontend detector checkpoint")
    _binding(raw.get("detector_source"), label="shared frontend detector source")
    plan_path, plan = load_frontend_plan(Path(binding["path"]))
    _require(sha256_file(plan_path) == binding["sha256"], "frontend plan hash drift")
    return plan_path, plan


def _authenticate_garc_summary(path: Path) -> dict[str, Any]:
    summary_path = _guard_metadata_path(path, label="GARC summary").resolve(strict=True)
    value = _strict_object(summary_path, label="GARC summary")
    _require(value.get("protocol") == GARC_SUMMARY_PROTOCOL, "GARC summary protocol drift")
    _require(value.get("status") == "complete", "GARC summary is incomplete")
    selected = value.get("selected")
    _require(isinstance(selected, Mapping), "GARC selected plan is absent")
    plan = _binding(
        {"path": selected.get("plan"), "sha256": selected.get("plan_sha256")},
        label="selected GARC plan",
    )
    validation = _binding(value.get("validation"), label="GARC validation report")
    validation_value = _strict_object(
        Path(validation["path"]), label="GARC validation report"
    )
    _require(
        validation_value.get("protocol") == "garc_full_auto_public_validation_v1",
        "GARC validation protocol drift",
    )
    _require(
        validation_value.get("status") == "independent_validation_complete",
        "GARC validation is incomplete",
    )
    _require(
        validation_value.get("mode") == "formal"
        and validation_value.get("claim_eligible") is True,
        "GARC validation is not formal/eligible",
    )
    predictions = validation_value.get("validation_predictions")
    _require(isinstance(predictions, Mapping), "GARC validation prediction binding absent")
    prediction_root_raw = Path(str(predictions.get("path") or ""))
    _require(prediction_root_raw.is_absolute(), "GARC validation prediction root must be absolute")
    prediction_root = _guard_metadata_path(
        prediction_root_raw, label="GARC validation prediction root"
    ).resolve(strict=True)
    _require(prediction_root.is_dir(), "GARC validation prediction root is absent")
    prediction_summary_path = _guard_metadata_path(
        prediction_root / "summary.json", label="GARC validation prediction summary"
    ).resolve(strict=True)
    declared_summary_sha = str(predictions.get("summary_sha256") or "").casefold()
    _require(_is_sha256(declared_summary_sha), "GARC prediction summary hash absent")
    _require(
        sha256_file(prediction_summary_path) == declared_summary_sha,
        "GARC validation prediction summary drift",
    )
    prediction_summary = _strict_object(
        prediction_summary_path, label="GARC validation prediction summary"
    )
    runtime_bundle = _resolve_relative_binding(
        (prediction_summary.get("artifacts") or {}).get("bundle"),
        root=prediction_root,
        label="GARC validation runtime bundle",
    )
    audit = value.get("audit")
    if isinstance(audit, Mapping):
        for key in (
            "restricted_namespace_images_opened",
            "field_manifest_opened",
            "field_images_opened",
            "field_labels_opened",
        ):
            if key in audit:
                _require(int(audit[key]) == 0, f"GARC summary scope violation: {key}")
    return {
        "summary": _bind_file(summary_path, label="GARC summary"),
        "selected_plan": plan,
        "validation": validation,
        "runtime_bundle": runtime_bundle,
    }


def _authenticate_paper(summary_path: Path, seal_path: Path, garc: Mapping[str, Any]) -> dict[str, Any]:
    summary_file = _guard_metadata_path(
        summary_path, label="paper results summary"
    ).resolve(strict=True)
    seal_file = _guard_metadata_path(seal_path, label="paper results seal").resolve(
        strict=True
    )
    summary = _strict_object(summary_file, label="paper results summary")
    seal = _strict_object(seal_file, label="paper results seal")
    _require(summary.get("protocol") == PAPER_RESULTS_PROTOCOL, "paper summary protocol drift")
    _require(summary.get("status") == "complete", "paper results are incomplete")
    _require(seal.get("protocol") == PAPER_RESULTS_PROTOCOL, "paper seal protocol drift")
    _require(seal.get("status") == "sealed", "paper results are not sealed")
    _require(
        ((seal.get("artifacts") or {}).get("summary") or {}).get("sha256")
        == sha256_file(summary_file),
        "paper seal does not bind summary",
    )
    cohort = summary.get("cohort") or {}
    _require(
        int(cohort.get("samples", -1)) == 412 and int(cohort.get("groups", -1)) == 19,
        "paper evidence is not the frozen 412/19 cohort",
    )
    sources = summary.get("sources") or {}
    garc_source = _binding(
        sources.get("garc_summary"), label="paper-bound GARC summary"
    )
    _require(
        str(Path(garc_source["path"]).resolve())
        == str(Path(garc["summary"]["path"]).resolve()),
        "paper evidence binds a different GARC summary",
    )
    _require(
        garc_source["sha256"] == garc["summary"]["sha256"],
        "paper/GARC summary hash drift",
    )
    audit = summary.get("audit") or {}
    _require(audit.get("public_truth_used_for_scoring_only") is True, "paper truth scope drift")
    _require(
        audit.get("all_prediction_and_score_seals_verified_before_public_truth_opened")
        is True,
        "paper result chronology is unverified",
    )
    for key in (
        "restricted_namespace_artifacts_opened",
        "field_manifest_opened",
        "field_images_opened",
        "field_labels_opened",
    ):
        if key in audit:
            _require(int(audit[key]) == 0, f"paper result scope violation: {key}")
    return {
        "summary": _bind_file(summary_file, label="paper results summary"),
        "seal": _bind_file(seal_file, label="paper results seal"),
    }


def _load_component_file(value: Any, *, label: str) -> FrozenComponentBinding:
    bound = _binding(value, label=label)
    record = _strict_object(Path(bound["path"]), label=label)
    return FrozenComponentBinding.from_record(record)


def _walk_bindings(value: Any, *, location: str = "value") -> list[tuple[str, dict[str, str]]]:
    found: list[tuple[str, dict[str, str]]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            found.append((location, _binding(value, label=location)))
        for key, nested in value.items():
            found.extend(_walk_bindings(nested, location=f"{location}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(_walk_bindings(nested, location=f"{location}[{index}]"))
    return found


def _source_bundle(path: Path, *, role: str) -> FrozenFullAutoBundle:
    guarded = _guard_metadata_path(path, label=f"{role} source bundle").resolve(strict=True)
    bundle = FrozenFullAutoBundle.load(guarded)
    _require(bundle.execution_mode == EXECUTION_FORMAL, f"{role} source bundle is not formal")
    _require(
        METHOD_NAME_TOKENS[role] in bundle.method_name.casefold(),
        f"{role} source bundle identity mismatch",
    )
    for component in (bundle.progress_binding, bundle.range_binding):
        _require(component.frozen, f"{role}/{component.name} is not frozen")
        _require(component.verified_complete, f"{role}/{component.name} is incomplete")
        _require(not component.synthetic, f"{role}/{component.name} is synthetic")
    return bundle


def _validate_factory_config(
    role: str,
    config: Mapping[str, Any],
    *,
    garc_plan: Mapping[str, str],
    source_bundle: FrozenFullAutoBundle,
) -> dict[str, Any]:
    assert_label_free(config, location=f"bundle_materialization.{role}.factory_config")
    _require(config.get("protocol") == RUNTIME_FACTORY_PROTOCOL, f"{role} factory protocol drift")
    _require(config.get("mode") == EXPECTED_FACTORY_MODES[role], f"{role} factory mode drift")
    source_plan = _binding(config.get("source_plan"), label=f"{role} source plan")
    if role != "v5_complete":
        _require(source_plan == dict(garc_plan), f"{role} must use the selected GARC plan")
    if role in EXPECTED_PROGRESS_KINDS:
        _require(
            config.get("progress_kind") == EXPECTED_PROGRESS_KINDS[role],
            f"{role} progress kind drift",
        )
    if role in {"pepd_shared_range", "vdn_shared_range"}:
        component = _load_component_file(
            config.get("progress_binding"), label=f"{role} progress binding"
        )
        _require(
            component.binding_sha256 == source_bundle.progress_binding.binding_sha256,
            f"{role} config/source progress identity drift",
        )
        factory = config.get("progress_factory")
        _require(isinstance(factory, Mapping), f"{role} progress factory absent")
        _binding(factory, label=f"{role} progress factory")
        _require(bool(str(factory.get("function") or "")), f"{role} progress function absent")
    if role == "transformer_shared_range":
        for key in ("pointer_segmentation", "original_transformer"):
            _binding(config.get(key), label=f"Transformer {key}")
    return dict(config)


def _artifact_catalog(rows: Any) -> tuple[list[dict[str, str]], dict[str, list[Path]]]:
    _require(isinstance(rows, list) and rows, "runtime artifact catalog is absent")
    normalized: list[dict[str, str]] = []
    by_hash: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for index, raw in enumerate(rows, 1):
        bound = _binding(raw, label=f"runtime artifact {index}")
        path = Path(bound["path"])
        _require(path not in seen, f"runtime artifact path repeated: {path}")
        seen.add(path)
        normalized.append(bound)
        by_hash.setdefault(bound["sha256"], []).append(path)
    normalized.sort(key=lambda row: row["path"].casefold())
    return normalized, by_hash


def _required_runtime_hashes(bundle: FrozenFullAutoBundle) -> set[str]:
    hashes = set(bundle.progress_binding.artifact_sha256.values())
    hashes.update(bundle.progress_binding.source_sha256.values())
    hashes.update(bundle.range_binding.artifact_sha256.values())
    hashes.update(bundle.range_binding.source_sha256.values())
    hashes.add(str(bundle.descriptor["factory_source_sha256"]))
    hashes.add(str(bundle.descriptor["adapter_source_sha256"]))
    if bundle.reference_detector_sha256 is not None:
        hashes.add(bundle.reference_detector_sha256)
    return hashes


def _comparison_contract() -> dict[str, bool]:
    return {
        "same_full_scene_frontend": True,
        "same_canonical_roi_per_sample": True,
        "all_predictions_sealed_together_before_labels": True,
        "progress_only_outputs_eligible": False,
        "garc_is_only_primary_final_model": True,
        "v5_is_internal_end_to_end_ablation": True,
        "pepd_vdn_share_garc_range_for_backbone_control": True,
        "transformer_is_sensitivity_only": True,
    }


def validate_spec(path: Path) -> dict[str, Any]:
    spec_path = _guard_metadata_path(path, label="materialization spec").resolve(strict=True)
    spec = _strict_object(spec_path, label="materialization spec")
    _require(spec.get("schema_version") == 1, "materialization spec version drift")
    _require(spec.get("protocol") == SPEC_PROTOCOL, "materialization spec protocol drift")
    _require(spec.get("status") == "public_model_inputs_frozen", "materialization spec is not frozen")
    audit = spec.get("audit") or {}
    _require(audit.get("public_data_only") is True, "materialization spec is not public-only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(audit.get(key) is False, f"materialization spec scope violation: {key}")

    garc_binding = _binding(spec.get("garc_summary"), label="GARC summary")
    garc = _authenticate_garc_summary(Path(garc_binding["path"]))
    _require(garc_binding == garc["summary"], "GARC summary spec binding drift")
    paper = spec.get("paper_results") or {}
    paper_summary = _binding(paper.get("summary"), label="paper results summary")
    paper_seal = _binding(paper.get("seal"), label="paper results seal")
    authenticated_paper = _authenticate_paper(
        Path(paper_summary["path"]), Path(paper_seal["path"]), garc
    )
    _require(paper_summary == authenticated_paper["summary"], "paper summary binding drift")
    _require(paper_seal == authenticated_paper["seal"], "paper seal binding drift")

    frontend_binding = _binding(spec.get("shared_frontend_plan"), label="shared frontend plan")
    frontend_path, frontend = _load_frontend_guarded(frontend_binding)
    runtime_factory = _binding(spec.get("runtime_factory"), label="runtime factory")
    _factory_has_entrypoint(Path(runtime_factory["path"]))
    _require(
        Path(runtime_factory["path"]).resolve()
        == Path(__file__).with_name("field_blind_runtime_factory.py").resolve(),
        "formal materialization must reuse the audited runtime factory",
    )

    catalog, by_hash = _artifact_catalog(spec.get("runtime_artifact_catalog"))
    methods = spec.get("methods")
    _require(isinstance(methods, Mapping), "materialization methods are absent")
    _require(set(methods) == set(METHOD_ROLES), "materialization spec requires exactly five roles")
    sources: dict[str, FrozenFullAutoBundle] = {}
    configs: dict[str, dict[str, Any]] = {}
    source_bindings: dict[str, dict[str, str]] = {}
    for role in METHOD_ROLES:
        raw = methods[role]
        _require(isinstance(raw, Mapping), f"{role} specification invalid")
        source_binding = _binding(raw.get("source_bundle"), label=f"{role} source bundle")
        source = _source_bundle(Path(source_binding["path"]), role=role)
        source_bindings[role] = source_binding
        sources[role] = source
        config_binding = _binding(raw.get("factory_config"), label=f"{role} factory config")
        config = _strict_object(Path(config_binding["path"]), label=f"{role} factory config")
        configs[role] = _validate_factory_config(
            role, config, garc_plan=garc["selected_plan"], source_bundle=source
        )
        for location, linked in _walk_bindings(config, location=f"methods.{role}.factory_config"):
            _require(
                linked["sha256"] in by_hash,
                f"runtime artifact catalog lacks {location}: {linked['sha256']}",
            )
    _require(
        source_bindings["garc_final"] == garc["runtime_bundle"],
        "GARC source bundle is not the validation-sealed runtime bundle",
    )
    garc_range = sources["garc_final"].range_binding
    for role in METHOD_ROLES:
        source = sources[role]
        prospective = FrozenFullAutoBundle.create(
            method_name=ROLE_METHOD_NAMES[role],
            progress_binding=source.progress_binding,
            range_binding=source.range_binding if role == "v5_complete" else garc_range,
            factory_source_sha256=runtime_factory["sha256"],
            reference_mode=source.reference_mode,
            reference_detector_sha256=source.reference_detector_sha256,
            execution_mode=EXECUTION_FORMAL,
        )
        missing_hashes = sorted(
            _required_runtime_hashes(prospective) - set(by_hash)
        )
        _require(
            not missing_hashes,
            f"{role} runtime artifact catalog lacks hashes: {missing_hashes}",
        )
    return {
        "path": spec_path,
        "value": spec,
        "garc": garc,
        "paper": authenticated_paper,
        "frontend": {
            "plan": frontend_binding,
            "detector_checkpoint": dict(frontend["detector_checkpoint"]),
            "detector_source": dict(frontend["detector_source"]),
            "contract_sha256": canonical_json_sha256(frontend["contract"]),
        },
        "runtime_factory": runtime_factory,
        "catalog": catalog,
        "catalog_by_hash": by_hash,
        "sources": sources,
        "source_bindings": source_bindings,
        "configs": configs,
    }


def freeze_spec(
    *,
    garc_summary: Path,
    paper_summary: Path,
    paper_seal: Path,
    frontend_plan: Path,
    runtime_factory: Path,
    source_bundles: Mapping[str, Path],
    factory_configs: Mapping[str, Path],
    runtime_artifacts: Sequence[Path],
    output: Path,
) -> Path:
    _require(set(source_bundles) == set(METHOD_ROLES), "source bundle roster is incomplete")
    _require(set(factory_configs) == set(METHOD_ROLES), "factory config roster is incomplete")
    garc = _authenticate_garc_summary(garc_summary)
    paper = _authenticate_paper(paper_summary, paper_seal, garc)
    frontend = _bind_file(frontend_plan, label="shared frontend plan")
    _load_frontend_guarded(frontend)
    factory = _bind_file(runtime_factory, label="runtime factory")
    _factory_has_entrypoint(Path(factory["path"]))
    _require(
        Path(factory["path"]).resolve()
        == Path(__file__).with_name("field_blind_runtime_factory.py").resolve(),
        "formal materialization must reuse the audited runtime factory",
    )
    catalog = [_bind_file(path, label="runtime artifact") for path in runtime_artifacts]
    unique_catalog = {row["path"]: row for row in catalog}
    _require(len(unique_catalog) == len(catalog), "runtime artifact catalog repeats a path")
    methods: dict[str, Any] = {}
    for role in METHOD_ROLES:
        source = _bind_file(source_bundles[role], label=f"{role} source bundle")
        _source_bundle(Path(source["path"]), role=role)
        config = _bind_file(factory_configs[role], label=f"{role} factory config")
        methods[role] = {"source_bundle": source, "factory_config": config}
    value = {
        "schema_version": 1,
        "protocol": SPEC_PROTOCOL,
        "status": "public_model_inputs_frozen",
        "created_at": _now(),
        "garc_summary": garc["summary"],
        "paper_results": paper,
        "shared_frontend_plan": frontend,
        "runtime_factory": factory,
        "methods": methods,
        "runtime_artifact_catalog": sorted(
            catalog, key=lambda row: row["path"].casefold()
        ),
        "audit": {
            "public_data_only": True,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
            "providers_instantiated": False,
            "images_opened": 0,
        },
    }
    assert_label_free(value, location="bundle_materialization_spec")
    target = _atomic_new(output, value)
    try:
        validate_spec(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def materialize(*, spec_path: Path, output_root: Path) -> dict[str, Any]:
    checked = validate_spec(spec_path)
    output = _guard_metadata_path(output_root, label="materialization output")
    if output.exists():
        raise FileExistsError(f"immutable materialization root already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    runtime_factory = checked["runtime_factory"]
    sources: Mapping[str, FrozenFullAutoBundle] = checked["sources"]
    garc_range = sources["garc_final"].range_binding
    catalog_by_hash: Mapping[str, list[Path]] = checked["catalog_by_hash"]
    methods: dict[str, Any] = {}
    bundle_records: dict[str, Any] = {}
    for role in METHOD_ROLES:
        source = sources[role]
        range_binding = source.range_binding if role == "v5_complete" else garc_range
        bundle = FrozenFullAutoBundle.create(
            method_name=ROLE_METHOD_NAMES[role],
            progress_binding=source.progress_binding,
            range_binding=range_binding,
            factory_source_sha256=runtime_factory["sha256"],
            reference_mode=source.reference_mode,
            reference_detector_sha256=source.reference_detector_sha256,
            execution_mode=EXECUTION_FORMAL,
        )
        required_hashes = _required_runtime_hashes(bundle)
        missing = sorted(required_hashes - set(catalog_by_hash))
        _require(not missing, f"{role} runtime artifact hashes are missing: {missing}")
        runtime_rows = []
        for digest in sorted(required_hashes):
            path = sorted(catalog_by_hash[digest], key=lambda value: str(value).casefold())[0]
            runtime_rows.append({"path": str(path), "sha256": digest})
        for _, linked in _walk_bindings(
            checked["configs"][role], location=f"methods.{role}.factory_config"
        ):
            if linked not in runtime_rows:
                runtime_rows.append(linked)
        runtime_rows.sort(key=lambda row: row["path"].casefold())
        bundle_path = output / f"{role}.bundle.json"
        bundle.write(bundle_path)
        bundle_binding = _bind_file(bundle_path, label=f"{role} materialized bundle")
        methods[role] = {
            "claim_tier": CLAIM_TIERS[role],
            "bundle": bundle_binding,
            "factory": dict(runtime_factory),
            "factory_function": FACTORY_FUNCTION,
            "factory_config": dict(checked["configs"][role]),
            "runtime_artifacts": runtime_rows,
            "output_mode": OUTPUT_MODE_FULL_READING,
            "evidence_role": EVIDENCE_ROLE_PRIMARY,
            "range_binding_sha256": bundle.range_binding_sha256,
        }
        bundle_records[role] = {
            **bundle_binding,
            "bundle_sha256": bundle.bundle_sha256,
            "progress_binding_sha256": bundle.progress_binding.binding_sha256,
            "range_binding_sha256": bundle.range_binding_sha256,
            "reference_detector_sha256": bundle.reference_detector_sha256,
        }
    frontend = checked["frontend"]
    roster = {
        "schema_version": 1,
        "protocol": ROSTER_PROTOCOL,
        "status": "all_full_reading_methods_frozen",
        "created_at": _now(),
        "methods": methods,
        "shared_frontend": frontend,
        "range_contract": {
            "garc_range_binding_sha256": methods["garc_final"]["range_binding_sha256"],
            "shared_garc_range_roles": [
                "garc_final",
                "pepd_shared_range",
                "vdn_shared_range",
                "transformer_shared_range",
            ],
            "v5_complete_retains_own_frozen_range": True,
            "v5_range_binding_sha256": methods["v5_complete"]["range_binding_sha256"],
        },
        "comparison_contract": _comparison_contract(),
        "authority": {
            "materialization_spec": {
                "path": str(checked["path"]),
                "sha256": sha256_file(checked["path"]),
            },
            "garc_summary": checked["garc"]["summary"],
            "paper_results": checked["paper"],
        },
        "audit": {
            "public_data_only": True,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
            "providers_instantiated": False,
            "images_opened": 0,
            "source_bundles_reused_without_identity_synthesis": True,
        },
    }
    roster_path = _atomic_new(output / "method_roster.json", roster)
    load_method_roster(roster_path)
    manifest = {
        "schema_version": 1,
        "protocol": MATERIALIZATION_PROTOCOL,
        "status": "complete",
        "created_at": _now(),
        "spec": {"path": str(checked["path"]), "sha256": sha256_file(checked["path"])},
        "method_roster": _bind_file(roster_path, label="materialized method roster"),
        "bundles": bundle_records,
        "shared_frontend_binding_sha256": canonical_json_sha256(frontend),
        "garc_shared_range_binding_sha256": garc_range.binding_sha256,
        "audit": dict(roster["audit"]),
    }
    manifest_path = _atomic_new(output / "summary.json", manifest)
    seal_artifacts = {
        "summary": _bind_file(manifest_path, label="materialization summary"),
        "method_roster": _bind_file(roster_path, label="method roster"),
    }
    seal_artifacts.update(
        {f"bundle_{role}": dict(record) for role, record in bundle_records.items()}
    )
    seal = {
        "schema_version": 1,
        "protocol": MATERIALIZATION_PROTOCOL,
        "status": "sealed",
        "created_at": _now(),
        "artifacts": seal_artifacts,
        "bundle_set_sha256": canonical_json_sha256(
            {role: bundle_records[role]["sha256"] for role in METHOD_ROLES}
        ),
    }
    seal_path = _atomic_new(output / "seal.json", seal)
    verified = verify_materialization(output)
    return {
        "status": "complete",
        "summary": str(manifest_path),
        "summary_sha256": sha256_file(manifest_path),
        "seal": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "method_roster": str(roster_path),
        "method_roster_sha256": sha256_file(roster_path),
        "verified": verified["verified"],
    }


def verify_materialization(root: Path) -> dict[str, Any]:
    checked_root = _guard_metadata_path(root, label="materialization root").resolve(strict=True)
    _require(checked_root.is_dir(), "materialization root is not a directory")
    summary_path = checked_root / "summary.json"
    roster_path = checked_root / "method_roster.json"
    seal_path = checked_root / "seal.json"
    summary = _strict_object(summary_path, label="materialization summary")
    seal = _strict_object(seal_path, label="materialization seal")
    _require(summary.get("protocol") == MATERIALIZATION_PROTOCOL, "materialization protocol drift")
    _require(summary.get("status") == "complete", "materialization is incomplete")
    _require(seal.get("protocol") == MATERIALIZATION_PROTOCOL, "materialization seal drift")
    _require(seal.get("status") == "sealed", "materialization is not sealed")
    artifacts = seal.get("artifacts") or {}
    for name, raw in artifacts.items():
        bound = _binding(raw, label=f"sealed artifact {name}")
        expected_path = summary_path if name == "summary" else roster_path if name == "method_roster" else checked_root / f"{name.removeprefix('bundle_')}.bundle.json"
        _require(Path(bound["path"]).resolve() == expected_path.resolve(), f"sealed artifact path drift: {name}")
    _, roster, bundles = load_method_roster(roster_path)
    _require(set(bundles) == set(METHOD_ROLES), "verified roster is incomplete")
    frontend = roster.get("shared_frontend")
    _require(isinstance(frontend, Mapping), "shared frontend binding absent")
    plan = _binding(frontend.get("plan"), label="shared frontend plan")
    _, loaded_frontend = _load_frontend_guarded(plan)
    _require(
        dict(frontend.get("detector_checkpoint") or {})
        == dict(loaded_frontend["detector_checkpoint"]),
        "shared detector checkpoint binding drift",
    )
    _require(
        dict(frontend.get("detector_source") or {}) == dict(loaded_frontend["detector_source"]),
        "shared detector source binding drift",
    )
    garc_range = bundles["garc_final"].range_binding_sha256
    for role in ("pepd_shared_range", "vdn_shared_range", "transformer_shared_range"):
        _require(bundles[role].range_binding_sha256 == garc_range, f"{role} range drift")
    contract = roster.get("range_contract") or {}
    _require(contract.get("garc_range_binding_sha256") == garc_range, "range contract drift")
    _require(contract.get("v5_complete_retains_own_frozen_range") is True, "V5 claim scope drift")
    expected_bundle_set = canonical_json_sha256(
        {role: sha256_file(checked_root / f"{role}.bundle.json") for role in METHOD_ROLES}
    )
    _require(seal.get("bundle_set_sha256") == expected_bundle_set, "bundle-set seal drift")
    return {
        "schema_version": 1,
        "protocol": MATERIALIZATION_PROTOCOL,
        "status": "verified",
        "verified": True,
        "method_roles": list(METHOD_ROLES),
        "garc_shared_range_binding_sha256": garc_range,
        "shared_frontend_binding_sha256": canonical_json_sha256(frontend),
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }


def preflight_requirements(
    *,
    spec_path: Path,
    garc_summary: Path | None = None,
    paper_summary: Path | None = None,
    paper_seal: Path | None = None,
) -> dict[str, Any]:
    """Return an exact missing/invalid list without traversing directories."""

    declared = {
        "materialization_spec": spec_path,
        "garc_summary": garc_summary,
        "paper_results_summary": paper_summary,
        "paper_results_seal": paper_seal,
    }
    missing: list[dict[str, str]] = []
    for name, raw in declared.items():
        if raw is None:
            continue
        guarded = _guard_metadata_path(raw, label=name)
        if not guarded.is_file():
            missing.append({"requirement": name, "path": str(guarded)})
    invalid: list[dict[str, str]] = []
    unresolved_dependencies: list[str] = []
    if any(row["requirement"] == "materialization_spec" for row in missing):
        unresolved_dependencies = [
            "shared_frontend_plan",
            "runtime_factory",
            "runtime_artifact_catalog",
            *[
                f"methods.{role}.{field}"
                for role in METHOD_ROLES
                for field in ("source_bundle", "factory_config")
            ],
        ]
    ready = not missing
    if ready:
        try:
            checked = validate_spec(spec_path)
            expected = {
                "garc_summary": (garc_summary, checked["garc"]["summary"]),
                "paper_results_summary": (paper_summary, checked["paper"]["summary"]),
                "paper_results_seal": (paper_seal, checked["paper"]["seal"]),
            }
            for name, (raw, binding) in expected.items():
                if raw is not None and Path(raw).resolve() != Path(binding["path"]).resolve():
                    invalid.append(
                        {
                            "requirement": name,
                            "reason": "path_differs_from_frozen_spec",
                        }
                    )
        except Exception as exc:
            invalid.append(
                {
                    "requirement": "materialization_spec",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    ready = ready and not invalid
    return {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "missing": missing,
        "invalid": invalid,
        "unresolved_dependencies_blocked_by_missing_spec": unresolved_dependencies,
        "directory_enumerations": 0,
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }


def _parse_role_paths(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        role, separator, path = raw.partition("=")
        _require(separator == "=" and role in METHOD_ROLES and path, f"invalid {label}: {raw}")
        _require(role not in result, f"duplicate {label} role: {role}")
        result[role] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-spec")
    freeze.add_argument("--garc-summary", type=Path, required=True)
    freeze.add_argument("--paper-summary", type=Path, required=True)
    freeze.add_argument("--paper-seal", type=Path, required=True)
    freeze.add_argument("--frontend-plan", type=Path, required=True)
    freeze.add_argument(
        "--runtime-factory",
        type=Path,
        default=Path(__file__).with_name("field_blind_runtime_factory.py"),
    )
    freeze.add_argument("--source-bundle", action="append", default=[])
    freeze.add_argument("--factory-config", action="append", default=[])
    freeze.add_argument("--runtime-artifact", type=Path, action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--spec", type=Path, required=True)
    preflight.add_argument("--garc-summary", type=Path)
    preflight.add_argument("--paper-summary", type=Path)
    preflight.add_argument("--paper-seal", type=Path)

    create = commands.add_parser("materialize")
    create.add_argument("--spec", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-spec":
        path = freeze_spec(
            garc_summary=args.garc_summary,
            paper_summary=args.paper_summary,
            paper_seal=args.paper_seal,
            frontend_plan=args.frontend_plan,
            runtime_factory=args.runtime_factory,
            source_bundles=_parse_role_paths(args.source_bundle, label="source bundle"),
            factory_configs=_parse_role_paths(args.factory_config, label="factory config"),
            runtime_artifacts=args.runtime_artifact,
            output=args.output,
        )
        print(json.dumps({"status": "frozen", "path": str(path), "sha256": sha256_file(path)}))
        return 0
    if args.command == "preflight":
        result = preflight_requirements(
            spec_path=args.spec,
            garc_summary=args.garc_summary,
            paper_summary=args.paper_summary,
            paper_seal=args.paper_seal,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["ready"] else 2
    if args.command == "materialize":
        print(json.dumps(materialize(spec_path=args.spec, output_root=args.output_root), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_materialization(args.root), ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MATERIALIZATION_PROTOCOL",
    "PREFLIGHT_PROTOCOL",
    "ROLE_METHOD_NAMES",
    "SPEC_PROTOCOL",
    "freeze_spec",
    "materialize",
    "preflight_requirements",
    "validate_spec",
    "verify_materialization",
]
