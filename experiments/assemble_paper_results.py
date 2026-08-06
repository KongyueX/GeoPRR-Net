"""Assemble the immutable paper tables from authenticated public artifacts.

The formal table is restricted to the fixed 412-image/19-group jointly unseen
cohort.  The 1,080-row fixed-fold GARC result and Original Transformer are
always emitted as sensitivity evidence.  This module never runs inference or
training and never accepts field, test, sealed, or confirmatory namespaces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "pointer_meter_paper_result_assembly_v1"
EXPECTED_JOINT_SUMMARY_SHA256 = (
    "5fb85223204a52ce168b8aed10835054238aef0513bade2fec6047ece323dccb"
)
EXPECTED_COHORT_SHA256 = (
    "680cfb8f77fb20517903f62db52ab0931bf46f8b3b8db1e43d8040c0f7d03363"
)
EXPECTED_MAPPING_SHA256 = (
    "9d88314f339d3b173d9d04b3ea4aa3ecac15bb0affca8166bbf82bf0425a9de9"
)
RESTRICTED_NAMESPACES = frozenset(
    {"field", "test", "sealed", "confirmatory", "confirmation", "xiangmu1", "xiangmu2"}
)


class AssemblyError(RuntimeError):
    """Raised when a present evidence artifact is inconsistent or ineligible."""


class NotReady(AssemblyError):
    """Raised when one or more final summaries have not been produced yet."""

    def __init__(self, missing: Sequence[str]):
        self.missing = tuple(missing)
        super().__init__("final evidence is not ready: " + ", ".join(self.missing))


@dataclass(frozen=True)
class EvidencePaths:
    garc_summary: Path = Path(r"C:\pointer_read\garc_full_auto_formal_v1\summary.json")
    external_comparison: Path = Path(
        r"C:\pointer_read\garc_external_progress_412_formal_v1\comparison.json"
    )
    under_pressure_1080_score: Path = Path(
        r"C:\pointer_read\under_pressure_official_public_independent_validation_v1\score.json"
    )
    under_pressure_412_score: Path = Path(
        r"C:\pointer_read\under_pressure_official_public_independent_validation_v1\score_joint_oof_412.json"
    )
    v5_oof_summary: Path = Path(r"C:\pointer_read\cagh_v5_enhanced_oof\summary.json")

    def named(self) -> dict[str, Path]:
        return {
            "garc_summary": self.garc_summary,
            "external_comparison": self.external_comparison,
            "under_pressure_1080_score": self.under_pressure_1080_score,
            "under_pressure_412_score": self.under_pressure_412_score,
            "v5_oof_summary": self.v5_oof_summary,
        }


@dataclass(frozen=True)
class ExpectedEvidence:
    joint_samples: int = 412
    joint_groups: int = 19
    range_samples: int = 1_080
    range_groups: int = 50
    v5_samples: int = 4_380
    v5_groups: int = 197
    fixed_geometry_unseen_samples: int = 168
    fixed_geometry_unseen_groups: int = 8
    fixed_geometry_overlap_samples: int = 912
    fixed_geometry_overlap_groups: int = 42
    joint_summary_sha256: str = EXPECTED_JOINT_SUMMARY_SHA256
    cohort_sha256: str = EXPECTED_COHORT_SHA256
    mapping_sha256: str = EXPECTED_MAPPING_SHA256


@dataclass
class MethodVector:
    method: str
    errors: list[float]
    successes: list[bool]
    groups: list[str]


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssemblyError(message)


def _reject_constant(value: str) -> Any:
    raise AssemblyError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _path_is_restricted(path: Path) -> bool:
    tokens: set[str] = set()
    for part in path.parts:
        normalized = part.casefold()
        for separator in ("-", ".", " "):
            normalized = normalized.replace(separator, "_")
        tokens.update(token for token in normalized.split("_") if token)
    return bool(tokens & RESTRICTED_NAMESPACES)


def guard_path(path: Path | str, label: str, *, must_exist: bool = True) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve(strict=False)
    require(not _path_is_restricted(candidate), f"{label} enters a restricted namespace")
    if must_exist:
        require(candidate.is_file(), f"{label} is absent: {candidate}")
    return candidate


def sha256_file(path: Path | str) -> str:
    candidate = guard_path(path, "hashed artifact")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_json(path: Path | str, label: str = "JSON artifact") -> dict[str, Any]:
    candidate = guard_path(path, label)
    try:
        with candidate.open("r", encoding="utf-8-sig") as handle:
            value = json.load(
                handle,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def strict_jsonl(path: Path | str, label: str = "JSONL artifact") -> list[dict[str, Any]]:
    candidate = guard_path(path, label)
    rows: list[dict[str, Any]] = []
    try:
        with candidate.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                require(bool(line.strip()), f"{label} has blank row {line_number}")
                row = json.loads(
                    line,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_strict_object,
                )
                require(isinstance(row, dict), f"{label} row {line_number} is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read {label}: {exc}") from exc
    return rows


def resolve_binding(binding: Mapping[str, Any], label: str) -> Path:
    require(isinstance(binding, Mapping), f"{label} binding is absent")
    path = guard_path(str(binding.get("path") or ""), label)
    expected = str(binding.get("sha256") or "").casefold()
    require(len(expected) == 64, f"{label} SHA-256 binding is absent")
    require(sha256_file(path) == expected, f"{label} SHA-256 drift")
    return path


def _relative_artifact(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return guard_path(candidate, label)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _assert_zero_audit(audit: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    for key in keys:
        if key in audit:
            require(int(audit[key]) == 0, f"{label} reports forbidden access in {key}")


def readiness(paths: EvidencePaths) -> dict[str, Any]:
    """Check only final summaries; never follow artifact links or open truth."""

    missing: list[str] = []
    incomplete: list[str] = []
    expected_status = {
        "garc_summary": "complete",
        "external_comparison": "complete",
        "under_pressure_1080_score": "formal_public_independent_validation_complete",
        "under_pressure_412_score": "formal_joint_oof_score_complete",
        "v5_oof_summary": "complete",
    }
    for name, raw_path in paths.named().items():
        path = guard_path(raw_path, name, must_exist=False)
        if not path.is_file():
            missing.append(name)
            continue
        try:
            artifact = strict_json(path, name)
        except AssemblyError:
            incomplete.append(name)
            continue
        if artifact.get("status") != expected_status[name]:
            incomplete.append(name)
    ready = not missing and not incomplete
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "missing": missing,
        "incomplete": incomplete,
        "metrics_emitted": False,
        "linked_artifacts_opened": 0,
        "public_truth_opened": 0,
        "restricted_namespace_artifacts_opened": 0,
    }


def _load_handoff(
    garc: Mapping[str, Any], expected: ExpectedEvidence
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    handoff_binding = garc.get("external_comparison_handoff", {}).get("handoff", {})
    root = guard_path(str(handoff_binding.get("root") or ""), "GARC external handoff root", must_exist=False)
    require(root.is_dir(), "GARC external handoff root is absent")
    summary_path = guard_path(root / "summary.json", "GARC handoff summary")
    seal_path = guard_path(root / "seal.json", "GARC handoff seal")
    require(
        sha256_file(summary_path) == handoff_binding.get("summary_sha256"),
        "GARC handoff summary differs from final GARC binding",
    )
    require(
        sha256_file(seal_path) == handoff_binding.get("seal_sha256"),
        "GARC handoff seal differs from final GARC binding",
    )
    summary = strict_json(summary_path, "GARC handoff summary")
    seal = strict_json(seal_path, "GARC handoff seal")
    require(summary.get("protocol") == "garc_external_progress_412_handoff_v1", "GARC handoff protocol drift")
    require(summary.get("status") == "label_free_handoff_sealed", "GARC handoff is not sealed")
    rows_path = _relative_artifact(
        root, str(summary.get("artifacts", {}).get("rows", {}).get("path") or ""), "GARC handoff rows"
    )
    rows_sha = sha256_file(rows_path)
    require(rows_sha == summary["artifacts"]["rows"].get("sha256"), "GARC handoff row hash drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "GARC handoff summary seal drift")
    require(seal.get("rows_sha256") == rows_sha, "GARC handoff row seal drift")
    protocol_path = resolve_binding(summary.get("frozen_protocol", {}), "external comparison protocol")
    preflight_path = resolve_binding(summary.get("preflight", {}), "external comparison preflight")
    bundle = canonical_sha256(
        {
            "protocol_sha256": sha256_file(protocol_path),
            "preflight_sha256": sha256_file(preflight_path),
            "summary_sha256": sha256_file(summary_path),
            "rows_sha256": rows_sha,
        }
    )
    require(seal.get("bundle_sha256") == bundle, "GARC handoff bundle seal drift")
    protocol = strict_json(protocol_path, "external comparison protocol")
    require(protocol.get("protocol") == "garc_external_progress_412_comparison_v1", "external protocol drift")
    cohort = protocol.get("cohort", {})
    require(int(cohort.get("samples", -1)) == expected.joint_samples, "external cohort sample drift")
    require(int(cohort.get("physical_groups", -1)) == expected.joint_groups, "external cohort group drift")
    require(cohort.get("summary", {}).get("sha256") == expected.joint_summary_sha256, "joint summary identity drift")
    require(cohort.get("label_free_roster", {}).get("sha256") == expected.cohort_sha256, "joint cohort identity drift")
    require(cohort.get("progress_mapping", {}).get("sha256") == expected.mapping_sha256, "joint mapping identity drift")
    resolve_binding(cohort.get("summary", {}), "joint cohort summary")
    roster_path = resolve_binding(cohort.get("label_free_roster", {}), "joint cohort roster")
    resolve_binding(cohort.get("progress_mapping", {}), "joint progress mapping")
    roster = strict_jsonl(roster_path, "joint cohort roster")
    rows = strict_jsonl(rows_path, "GARC handoff rows")
    require(len(rows) == expected.joint_samples, "GARC handoff sample inventory drift")
    require(len({str(row.get("sample_id")) for row in rows}) == len(rows), "duplicate GARC handoff sample")
    require(len({str(row.get("group_id")) for row in rows}) == expected.joint_groups, "GARC handoff group inventory drift")
    require({str(row["sample_id"]) for row in rows} == {str(row["sample_id"]) for row in roster}, "GARC handoff/cohort roster drift")
    require(summary.get("garc", {}).get("all_412_progress_geometry_head_and_backbone_group_unseen") is True, "GARC handoff is not all-component group-unseen")
    require(summary.get("method_roles", {}).get("original_transformer") == "fixed_checkpoint_non_oof_sensitivity_only", "Transformer role drift in handoff")
    for row in rows:
        require(row.get("garc_all_components_group_unseen") is True, "GARC handoff contains an ineligible row")
    _assert_zero_audit(summary.get("audit", {}), ["restricted_namespace_images_opened"], "GARC handoff")
    return root, summary, rows, protocol, seal


def _validate_garc(
    path: Path, expected: ExpectedEvidence
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = strict_json(path, "GARC final summary")
    require(summary.get("protocol") == "garc_full_auto_public_event_chain_summary_v1", "GARC final protocol drift")
    require(summary.get("status") == "complete", "GARC final run is incomplete")
    resolve_binding(
        {"path": summary.get("selected", {}).get("plan"), "sha256": summary.get("selected", {}).get("plan_sha256")},
        "selected GARC plan",
    )
    resolve_binding(
        {"path": summary.get("selected", {}).get("calibration"), "sha256": summary.get("selected", {}).get("calibration_sha256")},
        "selected GARC calibration",
    )
    validation_path = resolve_binding(summary.get("validation", {}), "GARC validation report")
    validation = strict_json(validation_path, "GARC validation report")
    require(validation.get("protocol") == "garc_full_auto_public_validation_v1", "GARC validation protocol drift")
    require(validation.get("status") == "independent_validation_complete", "GARC validation incomplete")
    require(validation.get("mode") == "formal" and validation.get("claim_eligible") is True, "GARC validation is not formal")
    parent_plan_path = resolve_binding(validation.get("parent_plan", {}), "GARC validation parent plan")
    require(parent_plan_path == guard_path(str(summary.get("selected", {}).get("plan") or ""), "selected GARC plan"), "GARC selected/validated plan drift")
    resolve_binding(validation.get("parent_protocol", {}), "GARC validation parent protocol")
    resolve_binding(validation.get("frozen_calibration", {}), "GARC validation calibration")
    if isinstance(validation.get("code"), Mapping):
        resolve_binding(validation["code"], "GARC validation code")
    prediction_binding = validation.get("validation_predictions", {})
    prediction_root = guard_path(str(prediction_binding.get("path") or ""), "GARC validation prediction root", must_exist=False)
    require(prediction_root.is_dir(), "GARC validation prediction root is absent")
    prediction_summary_path = guard_path(prediction_root / "summary.json", "GARC validation prediction summary")
    require(prediction_binding.get("summary_sha256") == sha256_file(prediction_summary_path), "GARC validation prediction-summary hash drift")
    prediction_summary = strict_json(prediction_summary_path, "GARC validation prediction summary")
    prediction_seal_path = guard_path(prediction_root / "seal.json", "GARC validation prediction seal")
    prediction_seal = strict_json(prediction_seal_path, "GARC validation prediction seal")
    prediction_rows_path = _relative_artifact(
        prediction_root,
        str(prediction_summary.get("artifacts", {}).get("predictions", {}).get("path") or ""),
        "GARC validation prediction rows",
    )
    require(
        prediction_binding.get("predictions_sha256")
        == prediction_summary.get("artifacts", {}).get("predictions", {}).get("sha256")
        == sha256_file(prediction_rows_path),
        "GARC validation prediction-row hash drift",
    )
    require(prediction_seal.get("summary_sha256") == sha256_file(prediction_summary_path), "GARC validation prediction summary-seal drift")
    require(prediction_seal.get("predictions_sha256") == sha256_file(prediction_rows_path), "GARC validation prediction row-seal drift")
    require(prediction_seal.get("plan_sha256") == sha256_file(parent_plan_path), "GARC validation prediction plan-seal drift")
    bundle_binding = prediction_summary.get("artifacts", {}).get("bundle", {})
    if bundle_binding:
        bundle_path = _relative_artifact(
            prediction_root, str(bundle_binding.get("path") or ""), "GARC validation runtime bundle"
        )
        require(bundle_binding.get("sha256") == prediction_seal.get("bundle_sha256") == sha256_file(bundle_path), "GARC validation runtime-bundle hash drift")
    eligibility = validation.get("evidence_eligibility", {})
    require(eligibility.get("joint_oof_end_to_end_claim") is True, "GARC 412 claim is ineligible")
    require(eligibility.get("full_1080_end_to_end_claim") is False, "GARC falsely promotes fixed-fold 1080")
    require(eligibility.get("ocr_and_fixed_fold_geometry_sensitivity") is True, "GARC 1080 sensitivity is ineligible")
    overlap = validation.get("overlap_audit", {})
    require(int(overlap.get("jointly_unseen_samples", -1)) == expected.joint_samples, "GARC joint sample drift")
    require(int(overlap.get("jointly_unseen_groups", -1)) == expected.joint_groups, "GARC joint group drift")
    require(overlap.get("joint_cohort_complete") is True, "GARC joint cohort incomplete")
    require(overlap.get("joint_mapping_sha256") == expected.joint_summary_sha256, "GARC joint summary binding drift")
    fixed = overlap.get("primary_fixed_fold_sensitivity", {})
    fixed_expected = {
        "geometry_unseen_samples": expected.fixed_geometry_unseen_samples,
        "geometry_unseen_groups": expected.fixed_geometry_unseen_groups,
        "geometry_fit_overlap_samples": expected.fixed_geometry_overlap_samples,
        "geometry_fit_overlap_groups": expected.fixed_geometry_overlap_groups,
    }
    for key, value in fixed_expected.items():
        require(int(fixed.get(key, -1)) == value, f"GARC fixed-fold audit drift: {key}")
    require(fixed.get("full_1080_all_component_unseen") is False, "GARC fixed-fold 1080 falsely marked unseen")
    metrics = validation.get("metrics", {})
    joint = metrics.get("joint_oof_end_to_end_frozen_acceptance", {})
    joint_range = metrics.get("joint_oof_range_frozen_acceptance", {})
    require((int(joint.get("samples", -1)), int(joint.get("groups", -1))) == (expected.joint_samples, expected.joint_groups), "GARC joint metric inventory drift")
    require((int(joint_range.get("samples", -1)), int(joint_range.get("groups", -1))) == (expected.joint_samples, expected.joint_groups), "GARC joint range inventory drift")
    range_metric = metrics.get("range_component_frozen_acceptance", {})
    require((int(range_metric.get("samples", -1)), int(range_metric.get("groups", -1))) == (expected.range_samples, expected.range_groups), "GARC 1080 sensitivity inventory drift")
    final_validation = summary.get("validation", {})
    require(final_validation.get("paper_claim_allowed", {}).get("full_1080_all_component_unseen") is False, "GARC final summary promotes 1080")
    require(final_validation.get("paper_claim_allowed", {}).get("joint_412_all_component_oof_end_to_end") is True, "GARC final summary demotes 412")
    _assert_zero_audit(validation.get("audit", {}), ["restricted_namespace_images_opened"], "GARC validation")
    _assert_zero_audit(summary.get("audit", {}), ["restricted_namespace_images_opened"], "GARC final summary")
    handoff_root, handoff, rows, external_protocol, _ = _load_handoff(summary, expected)
    return summary, validation, handoff_root, handoff, rows, external_protocol


def _load_external_prediction_root(
    root_value: str,
    method: str,
    expected: ExpectedEvidence,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = guard_path(root_value, f"{method} prediction root", must_exist=False)
    require(root.is_dir(), f"{method} prediction root is absent")
    summary_path = guard_path(root / "summary.json", f"{method} prediction summary")
    seal_path = guard_path(root / "seal.json", f"{method} prediction seal")
    summary = strict_json(summary_path, f"{method} prediction summary")
    seal = strict_json(seal_path, f"{method} prediction seal")
    require(summary.get("protocol") == "garc_external_progress_412_predictions_v1", f"{method} prediction protocol drift")
    require(summary.get("status") == "predictions_sealed", f"{method} predictions are not sealed")
    require(summary.get("method") == method, f"{method} prediction method drift")
    protocol_path = resolve_binding(summary.get("frozen_protocol", {}), f"{method} frozen protocol")
    resolve_binding(summary.get("preflight", {}), f"{method} preflight")
    rows_path = _relative_artifact(root, str(summary.get("artifacts", {}).get("predictions", {}).get("path") or ""), f"{method} prediction rows")
    rows_sha = sha256_file(rows_path)
    require(rows_sha == summary["artifacts"]["predictions"].get("sha256"), f"{method} row hash drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), f"{method} summary seal drift")
    require(seal.get("predictions_sha256") == rows_sha, f"{method} row seal drift")
    handoff_bundle = str(summary.get("handoff", {}).get("bundle_sha256") or "")
    require(seal.get("handoff_bundle_sha256") == handoff_bundle, f"{method} handoff seal drift")
    expected_bundle = canonical_sha256(
        {
            "method": method,
            "protocol_sha256": sha256_file(protocol_path),
            "handoff_bundle_sha256": handoff_bundle,
            "summary_sha256": sha256_file(summary_path),
            "predictions_sha256": rows_sha,
        }
    )
    require(seal.get("bundle_sha256") == expected_bundle, f"{method} bundle seal drift")
    rows = strict_jsonl(rows_path, f"{method} prediction rows")
    require(len(rows) == expected.joint_samples, f"{method} sample inventory drift")
    require(len({str(row.get("sample_id")) for row in rows}) == len(rows), f"duplicate {method} sample")
    strict_expected = method == "vdn_official200"
    require(summary.get("strict_oof_eligible") is strict_expected, f"{method} summary eligibility drift")
    for row in rows:
        require(row.get("method") == method, f"mixed method in {method} rows")
        require(row.get("strict_oof_eligible") is strict_expected, f"{method} row eligibility drift")
    return summary, rows, seal


def _validate_external(
    path: Path,
    expected: ExpectedEvidence,
    handoff_root: Path,
    handoff: Mapping[str, Any],
    handoff_rows: Sequence[Mapping[str, Any]],
    external_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    score = strict_json(path, "VDN/Transformer comparison")
    require(score.get("protocol") == "garc_external_progress_412_score_v1", "external score protocol drift")
    require(score.get("status") == "complete", "external comparison incomplete")
    protocol_path = resolve_binding(score.get("frozen_protocol", {}), "external score protocol")
    require(strict_json(protocol_path) == external_protocol, "external score protocol differs from GARC handoff")
    resolve_binding(score.get("preflight", {}), "external score preflight")
    score_handoff = score.get("handoff", {})
    require(guard_path(str(score_handoff.get("path") or ""), "external score handoff", must_exist=False) == handoff_root, "external score uses a different GARC handoff")
    require(score_handoff.get("summary_sha256") == sha256_file(handoff_root / "summary.json"), "external score handoff summary drift")
    require(score_handoff.get("bundle_sha256") == strict_json(handoff_root / "seal.json").get("bundle_sha256"), "external score handoff bundle drift")
    eligibility = score.get("claim_eligibility", {})
    require(set(eligibility.get("strict_formal_table", [])) == {"garc", "vdn_official200"}, "strict external table role drift")
    require(eligibility.get("vdn_strict_progress_oof_claim") is True, "VDN strict OOF claim ineligible")
    require(eligibility.get("garc_all_component_joint_oof_claim") is True, "GARC joint claim ineligible in external score")
    require(eligibility.get("original_transformer_strict_oof_claim") is False, "Transformer falsely marked strict OOF")
    require(set(eligibility.get("fixed_checkpoint_sensitivity_table", [])) <= {"original_transformer"}, "unknown sensitivity method")
    sources = score.get("sources", {})
    require(set(sources) == {"vdn_official200", "original_transformer"}, "VDN/Transformer sources are incomplete")
    require(set(eligibility.get("fixed_checkpoint_sensitivity_table", [])) == {"original_transformer"}, "Transformer sensitivity role is absent")
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for method, binding in sources.items():
        require(method in {"vdn_official200", "original_transformer"}, f"unknown external method: {method}")
        root = guard_path(str(binding.get("path") or ""), f"{method} score source", must_exist=False)
        summary, rows, seal = _load_external_prediction_root(str(root), method, expected)
        require(binding.get("summary_sha256") == sha256_file(root / "summary.json"), f"{method} score summary binding drift")
        require(binding.get("bundle_sha256") == seal.get("bundle_sha256"), f"{method} score bundle binding drift")
        require({str(row["sample_id"]) for row in rows} == {str(row["sample_id"]) for row in handoff_rows}, f"{method}/GARC roster drift")
        rows_by_method[method] = rows
    metrics = score.get("metrics", {})
    for method in {"garc", *rows_by_method.keys()}:
        metric = metrics.get(method, {})
        require((int(metric.get("samples", -1)), int(metric.get("physical_groups", -1))) == (expected.joint_samples, expected.joint_groups), f"{method} metric inventory drift")
    require("original_transformer" not in metrics or "original_transformer" in eligibility.get("fixed_checkpoint_sensitivity_table", []), "Transformer metric escaped sensitivity table")
    _assert_zero_audit(
        score.get("audit", {}),
        ["restricted_namespace_images_opened", "field_samples_used", "test_samples_used", "sealed_samples_used", "confirmatory_samples_used"],
        "external score",
    )
    return score, rows_by_method


def _score_seal_path(score_path: Path) -> Path:
    return score_path.with_name(score_path.name + ".seal.json")


def _validate_under_pressure_1080(
    path: Path, expected: ExpectedEvidence
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, Any]]:
    score = strict_json(path, "Under Pressure 1080 score")
    require(score.get("protocol") == "under_pressure_official_public_score_v1", "Under Pressure 1080 protocol drift")
    require(score.get("status") == "formal_public_independent_validation_complete", "Under Pressure 1080 score incomplete")
    require(score.get("claim_eligible") is True, "Under Pressure 1080 score ineligible")
    metric = score.get("metrics", {})
    require((int(metric.get("samples", -1)), int(metric.get("groups", -1))) == (expected.range_samples, expected.range_groups), "Under Pressure 1080 metric inventory drift")
    score_seal = strict_json(_score_seal_path(path), "Under Pressure 1080 score seal")
    require(score_seal.get("protocol") == "under_pressure_official_public_score_v1", "Under Pressure 1080 score-seal protocol drift")
    require(score_seal.get("score_sha256") == sha256_file(path), "Under Pressure 1080 score seal drift")
    root = guard_path(str(score.get("prediction_bundle", {}).get("path") or ""), "Under Pressure prediction root", must_exist=False)
    require(root.is_dir(), "Under Pressure prediction root absent")
    prediction_summary_path = guard_path(root / "summary.json", "Under Pressure prediction summary")
    prediction_seal_path = guard_path(root / "seal.json", "Under Pressure prediction seal")
    prediction_summary = strict_json(prediction_summary_path, "Under Pressure prediction summary")
    prediction_seal = strict_json(prediction_seal_path, "Under Pressure prediction seal")
    require(prediction_summary.get("protocol") == "under_pressure_official_public_predictions_v1", "Under Pressure prediction protocol drift")
    require(prediction_summary.get("status") == "predictions_sealed", "Under Pressure predictions not sealed")
    require(prediction_summary.get("mode") == "formal" and prediction_summary.get("claim_eligible") is True, "Under Pressure predictions ineligible")
    require(prediction_seal.get("protocol") == "under_pressure_official_public_predictions_v1" and prediction_seal.get("status") == "sealed_before_labels", "Under Pressure prediction seal identity drift")
    require((int(prediction_summary.get("samples", -1)), int(prediction_summary.get("groups", -1))) == (expected.range_samples, expected.range_groups), "Under Pressure prediction inventory drift")
    predictions_binding = prediction_summary.get("artifacts", {}).get("predictions", {})
    predictions_path = _relative_artifact(root, str(predictions_binding.get("path") or ""), "Under Pressure prediction rows")
    predictions_sha = sha256_file(predictions_path)
    require(predictions_binding.get("sha256") == predictions_sha, "Under Pressure prediction row hash drift")
    require(prediction_seal.get("summary_sha256") == sha256_file(prediction_summary_path), "Under Pressure prediction summary seal drift")
    require(prediction_seal.get("predictions_sha256") == predictions_sha, "Under Pressure prediction row seal drift")
    require(score.get("prediction_bundle", {}).get("predictions_sha256") == predictions_sha, "Under Pressure score/prediction hash drift")
    require(score.get("prediction_bundle", {}).get("seal_sha256") == canonical_sha256(prediction_seal), "Under Pressure score/prediction seal digest drift")
    protocol_path = resolve_binding(prediction_summary.get("parent_protocol", {}), "Under Pressure formal protocol")
    require(prediction_seal.get("parent_protocol_sha256") == sha256_file(protocol_path), "Under Pressure parent protocol seal drift")
    formal_protocol = strict_json(protocol_path, "Under Pressure formal protocol")
    require(formal_protocol.get("protocol") == "under_pressure_official_public_independent_validation_v1", "Under Pressure frozen protocol drift")
    require(formal_protocol.get("status") == "frozen_before_formal_inference", "Under Pressure protocol was not frozen")
    require((int(formal_protocol.get("samples", -1)), int(formal_protocol.get("groups", -1))) == (expected.range_samples, expected.range_groups), "Under Pressure protocol inventory drift")
    resolve_binding(formal_protocol.get("parent_protocol", {}), "Under Pressure parent public protocol")
    partition_path = resolve_binding(formal_protocol.get("partition_manifest", {}), "Under Pressure label-free partition manifest")
    rows = strict_jsonl(predictions_path, "Under Pressure prediction rows")
    require(len(rows) == expected.range_samples, "Under Pressure prediction row count drift")
    require(len({str(row.get("sample_id")) for row in rows}) == len(rows), "duplicate Under Pressure prediction")
    require(len({str(row.get("group_id")) for row in rows}) == expected.range_groups, "Under Pressure prediction group count drift")
    partition_rows = strict_jsonl(partition_path, "Under Pressure label-free partition manifest")
    require(
        [(str(row.get("sample_id")), str(row.get("group_id"))) for row in rows]
        == [(str(row.get("sample_id")), str(row.get("group_id"))) for row in partition_rows],
        "Under Pressure prediction/partition roster drift",
    )
    _assert_zero_audit(prediction_summary.get("audit", {}), ["field_images_opened", "test_images_opened", "sealed_images_opened"], "Under Pressure predictions")
    _assert_zero_audit(score.get("audit", {}), ["field_test_sealed_images_opened"], "Under Pressure 1080 score")
    return score, root, rows, prediction_seal


def _validate_under_pressure_412(
    path: Path,
    expected: ExpectedEvidence,
    prediction_root: Path,
    prediction_seal: Mapping[str, Any],
    cohort_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    score = strict_json(path, "Under Pressure 412 score")
    require(score.get("protocol") == "under_pressure_official_joint_oof_score_v1", "Under Pressure 412 protocol drift")
    require(score.get("status") == "formal_joint_oof_score_complete", "Under Pressure 412 score incomplete")
    require(score.get("claim_eligible") is True, "Under Pressure 412 score ineligible")
    metric = score.get("metrics", {})
    require((int(metric.get("samples", -1)), int(metric.get("groups", -1))) == (expected.joint_samples, expected.joint_groups), "Under Pressure 412 metric inventory drift")
    score_seal = strict_json(_score_seal_path(path), "Under Pressure 412 score seal")
    require(score_seal.get("protocol") == "under_pressure_official_joint_oof_score_v1", "Under Pressure 412 score-seal protocol drift")
    require(score_seal.get("score_sha256") == sha256_file(path), "Under Pressure 412 score seal drift")
    parent = score.get("parent_prediction_bundle", {})
    require(guard_path(str(parent.get("path") or ""), "Under Pressure 412 parent", must_exist=False) == prediction_root, "Under Pressure 412 uses a different prediction bundle")
    require(parent.get("predictions_sha256") == prediction_seal.get("predictions_sha256"), "Under Pressure 412 prediction hash drift")
    require(parent.get("seal_sha256") == canonical_sha256(prediction_seal), "Under Pressure 412 parent seal digest drift")
    require(score_seal.get("prediction_seal_sha256") == canonical_sha256(prediction_seal), "Under Pressure 412 score-seal parent drift")
    require((int(parent.get("full_samples", -1)), int(parent.get("full_groups", -1))) == (expected.range_samples, expected.range_groups), "Under Pressure 412 parent inventory drift")
    selection = score.get("selection", {})
    joint_summary_path = guard_path(str(selection.get("joint_summary_path") or ""), "Under Pressure joint cohort summary")
    require(sha256_file(joint_summary_path) == expected.joint_summary_sha256, "Under Pressure joint cohort file drift")
    require(selection.get("joint_summary_sha256") == expected.joint_summary_sha256, "Under Pressure joint summary identity drift")
    require(selection.get("cohort_sha256") == expected.cohort_sha256, "Under Pressure cohort identity drift")
    require(selection.get("mapping_sha256") == expected.mapping_sha256, "Under Pressure mapping identity drift")
    require(score_seal.get("joint_summary_sha256") == expected.joint_summary_sha256, "Under Pressure joint score-seal cohort drift")
    cohort_ids = [str(row["sample_id"]) for row in cohort_rows]
    require(selection.get("sample_ids_sha256") == canonical_sha256(cohort_ids), "Under Pressure joint sample order drift")
    require(len(cohort_ids) == expected.joint_samples, "Under Pressure cohort row count drift")
    _assert_zero_audit(score.get("audit", {}), ["field_test_sealed_images_opened"], "Under Pressure 412 score")
    return score, cohort_ids


def _validate_v5(path: Path, expected: ExpectedEvidence) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = strict_json(path, "enhanced V5 OOF summary")
    require(summary.get("protocol") == "cagh_v5_enhanced_authoritative_pepd_oof_v1", "V5 OOF protocol drift")
    require(summary.get("status") == "complete", "V5 OOF is incomplete")
    strict = summary.get("strict_oof", {})
    require(strict.get("status") == "complete", "V5 strict OOF summary incomplete")
    audit = strict.get("overlap_and_assignment_audit", {})
    require(int(audit.get("eligible_union_samples", -1)) == expected.v5_samples, "V5 OOF sample inventory drift")
    require(int(audit.get("eligible_union_groups", -1)) == expected.v5_groups, "V5 OOF group inventory drift")
    require(audit.get("all_rows_jointly_unseen_by_pepd_and_head") is True, "V5 OOF unseen audit failed")
    _assert_zero_audit(audit, ["field_samples_read", "public_test_samples_read"], "V5 overlap audit")
    _assert_zero_audit(summary, ["field_samples_read", "public_test_samples_read"], "V5 final summary")
    metrics = strict.get("metrics", {})
    require((int(metrics.get("samples", -1)), int(metrics.get("groups", -1))) == (expected.v5_samples, expected.v5_groups), "V5 OOF metric inventory drift")
    artifacts = summary.get("artifacts", {})
    strict_path = guard_path(str(artifacts.get("strict_oof_summary") or ""), "V5 strict OOF artifact")
    require(artifacts.get("strict_oof_summary_sha256") == sha256_file(strict_path), "V5 strict summary hash drift")
    require(strict_json(strict_path, "V5 strict OOF artifact") == strict, "V5 nested strict summary drift")
    predictions_path = guard_path(str(artifacts.get("strict_oof_predictions") or ""), "V5 strict OOF predictions")
    require(artifacts.get("strict_oof_predictions_sha256") == sha256_file(predictions_path), "V5 OOF prediction hash drift")
    require(strict.get("artifacts", {}).get("strict_oof_predictions_sha256") == sha256_file(predictions_path), "V5 strict prediction binding drift")
    folds = strict.get("folds", [])
    require(isinstance(folds, list) and len(folds) == 3, "V5 must contain three authenticated OOF routes")
    for fold in folds:
        fold_path = guard_path(str(fold.get("summary") or ""), "V5 fold summary")
        require(fold.get("summary_sha256") == sha256_file(fold_path), "V5 fold summary hash drift")
        fold_summary = strict_json(fold_path, "V5 fold summary")
        require(fold_summary.get("status") == "complete", "V5 fold incomplete")
        require(int(fold_summary.get("pepd_seed", -1)) == int(fold.get("pepd_seed", -2)), "V5 fold seed drift")
        pepd_checkpoint = fold_summary.get("pepd_checkpoint")
        if isinstance(pepd_checkpoint, Mapping):
            resolve_binding(pepd_checkpoint, "V5 fold PEPD checkpoint")
            require(pepd_checkpoint.get("strict_pepd_load") is True, "V5 fold used a non-strict PEPD load")
        fold_artifacts = fold_summary.get("artifacts", {})
        for name in ("checkpoint", "full_holdout_telemetry", "strict_assigned_telemetry"):
            if name not in fold_artifacts:
                continue
            artifact_path = guard_path(str(fold_artifacts[name]), f"V5 fold {name}")
            require(fold_artifacts.get(f"{name}_sha256") == sha256_file(artifact_path), f"V5 fold {name} hash drift")
    return summary, strict


def _load_public_truth(
    external_protocol: Mapping[str, Any], cohort_rows: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[str, float, float, float]]:
    public_protocol_path = resolve_binding(external_protocol.get("public_protocol", {}), "public range protocol")
    public_protocol = strict_json(public_protocol_path, "public range protocol")
    source_bindings = public_protocol.get("source_bindings", {})
    resolve_binding(source_bindings.get("syncg_train_manifest_protocol", {}), "public SyncG/train manifest protocol")
    source_path = resolve_binding(source_bindings.get("syncg_train_manifest", {}), "public SyncG/train manifest")
    requested = {str(row["sample_id"]): str(row["group_id"]) for row in cohort_rows}
    truth: dict[str, tuple[str, float, float, float]] = {}
    for row in strict_jsonl(source_path, "public SyncG/train manifest"):
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in requested:
            continue
        group_id = str(row.get("group_id") or "")
        require(group_id == requested[sample_id], f"public truth group drift: {sample_id}")
        reading = _finite(row.get("ground_truth"))
        start = _finite(row.get("scale_start"))
        end = _finite(row.get("scale_end"))
        require(reading is not None and start is not None and end is not None and end > start, f"invalid public truth: {sample_id}")
        truth[sample_id] = (group_id, reading, start, end)
    require(set(truth) == set(requested), "public truth does not cover the authenticated cohort")
    return truth


def _vector_from_readings(
    method: str,
    handoff_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, tuple[str, float, float, float]],
    external_rows: Sequence[Mapping[str, Any]] | None = None,
) -> MethodVector:
    external = {str(row["sample_id"]): row for row in (external_rows or [])}
    if method != "garc":
        require(set(external) == {str(row["sample_id"]) for row in handoff_rows}, f"{method} roster drift")
    errors: list[float] = []
    successes: list[bool] = []
    groups: list[str] = []
    for handoff in handoff_rows:
        sample_id = str(handoff["sample_id"])
        group_id, target, truth_start, truth_end = truth[sample_id]
        require(group_id == str(handoff["group_id"]), f"{method} group drift: {sample_id}")
        prediction: float | None = None
        if bool(handoff.get("range_accepted")):
            if method == "garc":
                if bool(handoff.get("garc_full_status")):
                    prediction = _finite(handoff.get("garc_predicted_reading"))
            else:
                row = external[sample_id]
                progress = _finite(row.get("prediction_progress"))
                start = _finite(handoff.get("predicted_scale_start"))
                end = _finite(handoff.get("predicted_scale_end"))
                if bool(row.get("status")) and progress is not None and start is not None and end is not None and start != end:
                    prediction = start + progress * (end - start)
        ok = prediction is not None
        errors.append(abs(prediction - target) / (truth_end - truth_start) if ok else 1.0)
        successes.append(ok)
        groups.append(group_id)
    return MethodVector(method, errors, successes, groups)


def _under_pressure_vector(
    rows: Sequence[Mapping[str, Any]],
    cohort_ids: Sequence[str],
    truth: Mapping[str, tuple[str, float, float, float]],
) -> MethodVector:
    by_id = {str(row["sample_id"]): row for row in rows}
    require(set(cohort_ids) <= set(by_id), "Under Pressure predictions miss joint samples")
    errors: list[float] = []
    successes: list[bool] = []
    groups: list[str] = []
    for sample_id in cohort_ids:
        row = by_id[sample_id]
        group_id, target, start, end = truth[sample_id]
        require(str(row.get("group_id")) == group_id, f"Under Pressure group drift: {sample_id}")
        prediction = _finite(row.get("predicted_reading")) if bool(row.get("status")) else None
        ok = prediction is not None
        errors.append(abs(prediction - target) / (end - start) if ok else 1.0)
        successes.append(ok)
        groups.append(group_id)
    return MethodVector("under_pressure_official", errors, successes, groups)


def _quantile(values: Sequence[float], q: float) -> float:
    require(bool(values), "cannot take quantile of empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def vector_metrics(vector: MethodVector) -> dict[str, Any]:
    require(bool(vector.errors), f"{vector.method} vector is empty")
    require(len(vector.errors) == len(vector.successes) == len(vector.groups), f"{vector.method} vector length drift")
    covered = [error for error, success in zip(vector.errors, vector.successes, strict=True) if success]
    group_values: dict[str, list[float]] = {}
    for group, error in zip(vector.groups, vector.errors, strict=True):
        group_values.setdefault(group, []).append(error)
    return {
        "samples": len(vector.errors),
        "groups": len(group_values),
        "successful": sum(vector.successes),
        "coverage": statistics.mean(vector.successes),
        "full_denominator_nmae_failure_penalty_1": statistics.mean(vector.errors),
        "full_denominator_p95_normalized_error": _quantile(vector.errors, 0.95),
        "macro_group_nmae": statistics.mean(statistics.mean(values) for values in group_values.values()),
        "conditional_nmae": statistics.mean(covered) if covered else None,
    }


def _assert_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
    left = _finite(actual)
    right = _finite(expected)
    require(left is not None and right is not None, f"{label} is not finite")
    require(math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance), f"{label} differs from sealed score")


def _check_external_metric(recomputed: Mapping[str, Any], sealed: Mapping[str, Any], label: str) -> None:
    aliases = {
        "samples": "samples",
        "groups": "physical_groups",
        "successful": "successful",
        "coverage": "coverage",
        "full_denominator_nmae_failure_penalty_1": "full_denominator_nmae_failure_penalty_1",
        "full_denominator_p95_normalized_error": "full_denominator_p95_normalized_error",
        "macro_group_nmae": "macro_group_nmae",
    }
    for ours, theirs in aliases.items():
        _assert_close(recomputed[ours], sealed[theirs], f"{label}.{theirs}")


def _check_under_pressure_metric(recomputed: Mapping[str, Any], sealed: Mapping[str, Any]) -> None:
    aliases = {
        "samples": "samples",
        "groups": "groups",
        "successful": "successes",
        "coverage": "native_coverage",
        "full_denominator_nmae_failure_penalty_1": "full_denominator_nmae",
        "full_denominator_p95_normalized_error": "full_denominator_nmae_p95",
        "macro_group_nmae": "group_macro_full_denominator_nmae",
    }
    for ours, theirs in aliases.items():
        _assert_close(recomputed[ours], sealed[theirs], f"under_pressure.{theirs}")


def paired_group_bootstrap(
    reference: MethodVector,
    comparator: MethodVector,
    *,
    iterations: int,
    seed: int,
    evidence_role: str,
) -> dict[str, Any]:
    require(iterations >= 100, "bootstrap iterations must be at least 100")
    require(reference.groups == comparator.groups, f"{comparator.method} group order differs from GARC")
    require(len(reference.errors) == len(comparator.errors), f"{comparator.method} roster size differs from GARC")
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(reference.groups):
        by_group.setdefault(group, []).append(index)
    groups = sorted(by_group)
    rng = random.Random(seed)
    error_differences: list[float] = []
    coverage_differences: list[float] = []
    for _ in range(iterations):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        indices = [index for group in chosen for index in by_group[group]]
        error_differences.append(
            statistics.mean(comparator.errors[index] - reference.errors[index] for index in indices)
        )
        coverage_differences.append(
            statistics.mean(
                float(reference.successes[index]) - float(comparator.successes[index])
                for index in indices
            )
        )
    reference_mean = statistics.mean(reference.errors)
    comparator_mean = statistics.mean(comparator.errors)
    error_difference = comparator_mean - reference_mean
    coverage_difference = statistics.mean(reference.successes) - statistics.mean(comparator.successes)
    return {
        "evidence_role": evidence_role,
        "reference": reference.method,
        "comparator": comparator.method,
        "nmae_difference_definition": "comparator_minus_garc",
        "positive_nmae_difference_favors": "garc",
        "nmae_difference": error_difference,
        "nmae_difference_group_bootstrap_95ci": [
            _quantile(error_differences, 0.025),
            _quantile(error_differences, 0.975),
        ],
        "garc_relative_nmae_reduction_percent": (
            100.0 * error_difference / comparator_mean if comparator_mean != 0.0 else None
        ),
        "coverage_difference_definition": "garc_minus_comparator",
        "coverage_difference": coverage_difference,
        "coverage_difference_group_bootstrap_95ci": [
            _quantile(coverage_differences, 0.025),
            _quantile(coverage_differences, 0.975),
        ],
        "physical_groups": len(groups),
        "iterations": iterations,
        "seed": seed,
    }


def _main_row(method: str, metrics: Mapping[str, Any], note: str) -> dict[str, Any]:
    return {
        "table": "strict_joint_412_main",
        "method": method,
        "samples": metrics["samples"],
        "groups": metrics["groups"],
        "coverage": metrics["coverage"],
        "full_denominator_nmae_failure_penalty_1": metrics["full_denominator_nmae_failure_penalty_1"],
        "full_denominator_p95_normalized_error": metrics["full_denominator_p95_normalized_error"],
        "conditional_nmae": metrics["conditional_nmae"],
        "macro_group_nmae": metrics["macro_group_nmae"],
        "claim_eligible": True,
        "note": note,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def assemble(
    paths: EvidencePaths,
    output_root: Path,
    *,
    expected: ExpectedEvidence = ExpectedEvidence(),
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260806,
) -> Path:
    state = readiness(paths)
    if not state["ready"]:
        raise NotReady([*state["missing"], *state["incomplete"]])

    # Every label-free bundle and score seal is authenticated before truth opens.
    garc, garc_validation, handoff_root, handoff, handoff_rows, external_protocol = _validate_garc(
        guard_path(paths.garc_summary, "GARC final summary"), expected
    )
    external, external_rows = _validate_external(
        guard_path(paths.external_comparison, "external comparison"),
        expected,
        handoff_root,
        handoff,
        handoff_rows,
        external_protocol,
    )
    up1080, up_root, up_rows, up_prediction_seal = _validate_under_pressure_1080(
        guard_path(paths.under_pressure_1080_score, "Under Pressure 1080 score"), expected
    )
    cohort_path = resolve_binding(external_protocol.get("cohort", {}).get("label_free_roster", {}), "joint cohort roster")
    cohort_rows = strict_jsonl(cohort_path, "joint cohort roster")
    up412, cohort_ids = _validate_under_pressure_412(
        guard_path(paths.under_pressure_412_score, "Under Pressure 412 score"),
        expected,
        up_root,
        up_prediction_seal,
        cohort_rows,
    )
    _, v5_strict = _validate_v5(guard_path(paths.v5_oof_summary, "V5 OOF summary"), expected)

    # This is the first and only range-bearing read in the assembler.
    full_truth = _load_public_truth(external_protocol, up_rows)
    truth = {str(row["sample_id"]): full_truth[str(row["sample_id"])] for row in cohort_rows}
    garc_vector = _vector_from_readings("garc", handoff_rows, truth)
    vectors: dict[str, MethodVector] = {"garc": garc_vector}
    for method, rows in external_rows.items():
        vectors[method] = _vector_from_readings(method, handoff_rows, truth, rows)
    vectors["under_pressure_official"] = _under_pressure_vector(up_rows, cohort_ids, truth)
    recomputed = {method: vector_metrics(vector) for method, vector in vectors.items()}
    for method in {"garc", *external_rows.keys()}:
        _check_external_metric(recomputed[method], external["metrics"][method], method)
    for method, comparison in external.get("paired_group_bootstrap", {}).items():
        if method not in recomputed:
            continue
        _assert_close(
            comparison.get("point_estimate"),
            recomputed[method]["full_denominator_nmae_failure_penalty_1"]
            - recomputed["garc"]["full_denominator_nmae_failure_penalty_1"],
            f"external paired point estimate for {method}",
        )
    _check_under_pressure_metric(recomputed["under_pressure_official"], up412["metrics"])
    up1080_recomputed = vector_metrics(
        _under_pressure_vector(
            up_rows,
            [str(row["sample_id"]) for row in up_rows],
            full_truth,
        )
    )
    _check_under_pressure_metric(up1080_recomputed, up1080["metrics"])
    _assert_close(
        recomputed["garc"]["full_denominator_nmae_failure_penalty_1"],
        garc_validation["metrics"]["joint_oof_end_to_end_frozen_acceptance"]["reading_nmae_full_denominator_failure_penalty_1"],
        "GARC validation/external handoff NMAE",
    )

    main_table = [
        _main_row(
            "garc",
            recomputed["garc"],
            "all-component grouped OOF; automatic numeric range; fixed 412/19 cohort",
        ),
        _main_row(
            "vdn_official200",
            recomputed["vdn_official200"],
            "strict grouped-OOF progress backend with the identical frozen GARC automatic range",
        ),
        _main_row(
            "under_pressure_official",
            recomputed["under_pressure_official"],
            "unchanged image-only official pipeline on the identical frozen cohort and canonical ROI",
        ),
    ]

    range_sensitivity = garc_validation["metrics"]["range_component_frozen_acceptance"]
    sensitivity_table: list[dict[str, Any]] = [
        {
            "table": "sensitivity_only",
            "family": "fixed_fold_1080_range_component",
            "method": "garc_range_fixed_seed_20260720",
            "samples": expected.range_samples,
            "groups": expected.range_groups,
            "coverage": range_sensitivity.get("coverage"),
            "metric_name": "pair_rounded_exact_full_denominator",
            "metric_value": range_sensitivity.get("pair_rounded_exact_full_denominator"),
            "strict_main_table_eligible": False,
            "reason": "geometry backbone saw 912/1080 images from 42/50 groups",
        },
        {
            "table": "sensitivity_only",
            "family": "public_1080_external_full_auto",
            "method": "under_pressure_official",
            "samples": up1080["metrics"]["samples"],
            "groups": up1080["metrics"]["groups"],
            "coverage": up1080["metrics"]["native_coverage"],
            "metric_name": "full_denominator_nmae_failure_penalty_1",
            "metric_value": up1080["metrics"]["full_denominator_nmae"],
            "strict_main_table_eligible": False,
            "reason": "1080 result is not the all-component-unseen GARC comparison cohort",
        },
    ]
    if "original_transformer" in recomputed:
        transformer = recomputed["original_transformer"]
        sensitivity_table.append(
            {
                "table": "sensitivity_only",
                "family": "same_412_fixed_checkpoint_progress",
                "method": "original_transformer",
                "samples": transformer["samples"],
                "groups": transformer["groups"],
                "coverage": transformer["coverage"],
                "metric_name": "full_denominator_nmae_failure_penalty_1",
                "metric_value": transformer["full_denominator_nmae_failure_penalty_1"],
                "strict_main_table_eligible": False,
                "reason": "fixed checkpoint lacks an authenticated training roster and its segmenter overlaps the cohort",
            }
        )

    v5_metrics = v5_strict["metrics"]
    component_table = [
        {
            "table": "component_oof",
            "method": "enhanced_v5_geometry_head",
            "samples": v5_metrics["samples"],
            "groups": v5_metrics["groups"],
            "coverage": v5_metrics["coverage"],
            "full_denominator_nmae": v5_metrics["full_denominator_nmae"],
            "p95_absolute_progress_error": v5_metrics["p95_absolute_progress_error"],
            "claim_eligible": True,
            "note": "jointly unseen PEPD backbone and enhanced-V5 head on the authenticated OOF union; component evidence, not the 412 end-to-end table",
        }
    ]

    comparisons: list[dict[str, Any]] = []
    for index, method in enumerate(("vdn_official200", "under_pressure_official")):
        comparisons.append(
            paired_group_bootstrap(
                garc_vector,
                vectors[method],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + index,
                evidence_role="strict_joint_412_main",
            )
        )
    if "original_transformer" in vectors:
        comparisons.append(
            paired_group_bootstrap(
                garc_vector,
                vectors["original_transformer"],
                iterations=bootstrap_iterations,
                seed=bootstrap_seed + 2,
                evidence_role="same_412_sensitivity_only",
            )
        )

    source_paths = paths.named()
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "primary_evidence": "strict_joint_412_main",
        "cohort": {
            "samples": expected.joint_samples,
            "groups": expected.joint_groups,
            "joint_summary_sha256": expected.joint_summary_sha256,
            "cohort_sha256": expected.cohort_sha256,
            "mapping_sha256": expected.mapping_sha256,
        },
        "main_table": main_table,
        "sensitivity_table": sensitivity_table,
        "component_table": component_table,
        "paired_group_bootstrap": comparisons,
        "claim_boundaries": {
            "transformer": "sensitivity only",
            "garc_1080_fixed_fold": "sensitivity only",
            "main_table": "412 images / 19 physical groups / same canonical ROI; full-denominator failure penalty NMAE=1",
        },
        "code": {
            "assembler": {
                "path": str(Path(__file__).resolve(strict=True)),
                "sha256": sha256_file(Path(__file__).resolve(strict=True)),
            }
        },
        "sources": {
            name: {"path": str(guard_path(path, name)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "audit": {
            "all_prediction_and_score_seals_verified_before_public_truth_opened": True,
            "bottom_up_metrics_match_all_sealed_summaries": True,
            "manual_or_ground_truth_numeric_range_used_by_model": False,
            "public_truth_used_for_scoring_only": True,
            "restricted_namespace_artifacts_opened": 0,
            "field_samples_opened": 0,
            "test_samples_opened": 0,
            "sealed_samples_opened": 0,
            "confirmatory_samples_opened": 0,
            "inference_started": False,
            "training_started": False,
            "feishu_message_sent": False,
        },
    }

    output = guard_path(output_root, "paper result output", must_exist=False)
    require(not output.exists(), f"refusing to overwrite paper result output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    require(not staging.exists(), f"staging output already exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_json(staging / "summary.json", result)
        _write_csv(staging / "main_table.csv", main_table)
        _write_csv(staging / "sensitivity_table.csv", sensitivity_table)
        _write_csv(staging / "component_table.csv", component_table)
        comparison_rows = [
            {
                **{key: value for key, value in row.items() if not isinstance(value, list)},
                "nmae_difference_ci95_low": row["nmae_difference_group_bootstrap_95ci"][0],
                "nmae_difference_ci95_high": row["nmae_difference_group_bootstrap_95ci"][1],
                "coverage_difference_ci95_low": row["coverage_difference_group_bootstrap_95ci"][0],
                "coverage_difference_ci95_high": row["coverage_difference_group_bootstrap_95ci"][1],
            }
            for row in comparisons
        ]
        _write_csv(staging / "paired_group_bootstrap.csv", comparison_rows)
        artifacts = {
            name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in {
                "summary": staging / "summary.json",
                "main_table": staging / "main_table.csv",
                "sensitivity_table": staging / "sensitivity_table.csv",
                "component_table": staging / "component_table.csv",
                "paired_group_bootstrap": staging / "paired_group_bootstrap.csv",
            }.items()
        }
        _write_json(
            staging / "seal.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "sealed",
                "artifacts": artifacts,
                "bundle_sha256": canonical_sha256(artifacts),
            },
        )
        os.replace(staging, output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return output / "summary.json"


def _paths_from_args(args: argparse.Namespace) -> EvidencePaths:
    return EvidencePaths(
        garc_summary=args.garc_summary,
        external_comparison=args.external_comparison,
        under_pressure_1080_score=args.under_pressure_1080_score,
        under_pressure_412_score=args.under_pressure_412_score,
        v5_oof_summary=args.v5_oof_summary,
    )


def _add_evidence_args(parser: argparse.ArgumentParser) -> None:
    defaults = EvidencePaths()
    parser.add_argument("--garc-summary", type=Path, default=defaults.garc_summary)
    parser.add_argument("--external-comparison", type=Path, default=defaults.external_comparison)
    parser.add_argument("--under-pressure-1080-score", type=Path, default=defaults.under_pressure_1080_score)
    parser.add_argument("--under-pressure-412-score", type=Path, default=defaults.under_pressure_412_score)
    parser.add_argument("--v5-oof-summary", type=Path, default=defaults.v5_oof_summary)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_evidence_args(preflight_parser)
    assemble_parser = subparsers.add_parser("assemble")
    _add_evidence_args(assemble_parser)
    assemble_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"C:\pointer_read\paper_final_results_v1"),
    )
    assemble_parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    assemble_parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _paths_from_args(args)
    if args.command == "preflight":
        print(json.dumps(readiness(paths), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    try:
        output = assemble(
            paths,
            args.output_root,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    except NotReady as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "status": "not_ready",
                    "missing_or_incomplete": list(exc.missing),
                    "metrics_emitted": False,
                    "restricted_namespace_artifacts_opened": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps({"status": "complete", "output": str(output), "sha256": sha256_file(output)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
