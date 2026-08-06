"""Freeze, materialize, and seal the GARC SAME-412/19 pilot control.

The control is the calibration-preregistered ``v5-tiny-top1`` family: the
Tiny numeric recognizer, literal recognizer top-1 decoding, and enhanced-V5
fold geometry without PEPD/Base geometry fusion.  Seed 20260720's plan and
formal calibration are created by the parent GARC chain before independent
validation.  Seeds 20260721/20260722 are later materialized mechanically from
that frozen family plus the already-authenticated fold route artifacts.

The preflight commands in this module never open images or annotations, import
Torch, start inference/training, wait for a process, or send notifications.
Formal inference remains owned by the PowerShell event chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from experiments.automatic_numeric_range_public_protocol import (
    atomic_new_json,
    canonical_sha256,
    guard_public_path,
    require,
    sha256_file,
    strict_json,
)


SOURCE: Final[Path] = Path(__file__).resolve()
PROJECT_ROOT: Final[Path] = SOURCE.parents[1]
GARC_RUNNER: Final[Path] = SOURCE.with_name("garc_full_auto_public.py")
GARC_EVALUATOR: Final[Path] = SOURCE.with_name("evaluate_garc_full_auto_public.py")
GARC_EVENT_CHAIN: Final[Path] = SOURCE.with_name(
    "run_garc_full_auto_public_event_driven.ps1"
)
CONTROL_EVENT_CHAIN: Final[Path] = SOURCE.with_name(
    "run_garc_same_cohort_control_after_garc_event_driven.ps1"
)
PROGRESS_FACTORY: Final[Path] = SOURCE.with_name("garc_pepd_progress_factory.py")
PROMOTION_SOURCE: Final[Path] = SOURCE.with_name("garc_common_split_promotion.py")
COMMON_SPLIT_PROTOCOL: Final[Path] = SOURCE.with_name(
    "garc_common_split_progress_protocol.json"
)

PREREGISTRATION_PROTOCOL: Final[str] = (
    "garc_same_412_tiny_top1_control_preregistration_v1"
)
PREFLIGHT_PROTOCOL: Final[str] = "garc_same_412_control_preflight_v1"
EXECUTION_MANIFEST_PROTOCOL: Final[str] = (
    "garc_same_412_control_execution_manifest_v1"
)
MATERIALIZATION_PROTOCOL: Final[str] = "garc_same_412_control_materialization_v1"
SUMMARY_PROTOCOL: Final[str] = "garc_same_412_control_summary_v1"
SEAL_PROTOCOL: Final[str] = "garc_same_412_control_seal_v1"
GARC_FINAL_PROTOCOL: Final[str] = "garc_full_auto_public_event_chain_summary_v1"
GARC_RECOGNIZER_SELECTION_PROTOCOL: Final[str] = (
    "garc_numeric_recognizer_selection_v2"
)
GARC_VALIDATION_PROTOCOL: Final[str] = "garc_full_auto_public_validation_v1"

EXPECTED_SEEDS: Final[tuple[int, ...]] = (20260720, 20260721, 20260722)
EXPECTED_JOINT_COUNTS: Final[dict[int, int]] = {
    20260720: 168,
    20260721: 116,
    20260722: 128,
}
EXPECTED_JOINT_SUMMARY_SHA256: Final[str] = (
    "5fb85223204a52ce168b8aed10835054238aef0513bade2fec6047ece323dccb"
)
EXPECTED_COHORT_SHA256: Final[str] = (
    "680cfb8f77fb20517903f62db52ab0931bf46f8b3b8db1e43d8040c0f7d03363"
)
EXPECTED_MAPPING_SHA256: Final[str] = (
    "9d88314f339d3b173d9d04b3ea4aa3ecac15bb0affca8166bbf82bf0425a9de9"
)
EXPECTED_GARC_RELATIVE_PATHS: Final[dict[str, str]] = {
    "summary": "summary.json",
    "candidate_validation": "validation.json",
    "recognizer_selection": "recognizer_selection.json",
    "control_seed20_plan": "plans/v5-tiny-top1.seed_20260720.plan.json",
    "control_calibration": "calibration/tiny_top1.json",
    "external_handoff_summary": "external_progress_comparison/handoff/summary.json",
    "external_handoff_seal": "external_progress_comparison/handoff/seal.json",
}
_SOURCE_PATHS: Final[dict[str, Path]] = {
    "control_auditor": SOURCE,
    "control_event_chain": CONTROL_EVENT_CHAIN,
    "garc_runner": GARC_RUNNER,
    "garc_evaluator": GARC_EVALUATOR,
    "garc_event_chain": GARC_EVENT_CHAIN,
    "progress_factory": PROGRESS_FACTORY,
    "common_split_promotion": PROMOTION_SOURCE,
}


def _digest(value: Any, *, label: str) -> str:
    result = str(value or "").strip().casefold()
    require(
        len(result) == 64
        and set(result).issubset(frozenset("0123456789abcdef")),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return result


def _binding(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_binding(binding: Mapping[str, Any], *, label: str) -> Path:
    require(isinstance(binding, Mapping), f"{label} binding is absent")
    path = guard_public_path(Path(str(binding.get("path") or "")), label=label)
    require(
        sha256_file(path) == _digest(binding.get("sha256"), label=label),
        f"{label} hash drift",
    )
    return path


def _same_path(left: Path | str, right: Path | str) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _utc(value: str, *, label: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), f"{label} must end in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not ISO-8601") from error
    require(parsed.tzinfo is not None, f"{label} lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _source_bindings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in _SOURCE_PATHS.items():
        require(path.is_file(), f"required control source is absent: {path}")
        result[name] = _binding(path)
    return result


def _garc_source_order_audit() -> dict[str, int | bool]:
    """Prove the frozen parent chain orders control freeze before validation."""

    text = GARC_EVENT_CHAIN.read_text(encoding="utf-8-sig")
    markers = {
        "control_plan_freeze": '-Tag "v5-tiny-top1"',
        "control_calibration": '"seal-calibration-$($Candidate.Name)"',
        "recognizer_selection": '$RecognizerDecisionPath = Join-Path',
        "validation_transition": '"garc-full-auto-public-v1-calibration-to-validation-v1"',
        "independent_validation": '"validation-primary-1080"',
    }
    offsets = {name: text.find(marker) for name, marker in markers.items()}
    require(all(value >= 0 for value in offsets.values()), "GARC source markers drift")
    require(
        offsets["control_plan_freeze"]
        < offsets["control_calibration"]
        < offsets["recognizer_selection"]
        < offsets["validation_transition"]
        < offsets["independent_validation"],
        "GARC source no longer freezes the control before validation",
    )
    return {
        **offsets,
        "control_plan_and_calibration_precede_independent_validation": True,
    }


def _assert_safe_root(path: Path, *, label: str, must_exist: bool) -> Path:
    value = guard_public_path(
        path,
        label=label,
        must_exist=must_exist,
        expect_file=False,
    )
    require(value != Path(value.anchor), f"{label} cannot be a drive root")
    return value


def freeze_preregistration(
    *,
    output_path: Path,
    garc_output_root: Path,
    garc_process_id: int,
    garc_process_start_utc: str,
    garc_process_command_sha256: str,
) -> Path:
    """Freeze the control identity while GARC validation outputs are absent."""

    require(int(garc_process_id) > 0, "an authenticated running GARC PID is required")
    _utc(garc_process_start_utc, label="GARC process start")
    command_sha = _digest(
        garc_process_command_sha256, label="GARC process command line"
    )
    garc_root = _assert_safe_root(
        garc_output_root, label="future GARC output root", must_exist=False
    )
    absent: dict[str, bool] = {}
    for name, relative in EXPECTED_GARC_RELATIVE_PATHS.items():
        artifact = garc_root / Path(relative)
        absent[name] = not artifact.exists()
    require(
        all(absent.values()),
        "control preregistration must precede every GARC result/control artifact",
    )
    common_protocol = guard_public_path(
        COMMON_SPLIT_PROTOCOL, label="common-split protocol"
    )
    value = {
        "schema_version": 1,
        "protocol": PREREGISTRATION_PROTOCOL,
        "status": "frozen_before_garc_independent_validation_outputs_existed",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "garc_process": {
            "pid": int(garc_process_id),
            # Preserve Windows' seventh fractional-second digit.  Python's
            # datetime stores microseconds only, so re-serializing would lose
            # the final 100 ns of the authenticated process identity.
            "start_utc": garc_process_start_utc,
            "command_line_sha256": command_sha,
            "required_script_name": "run_garc_full_auto_public_event_driven.ps1",
        },
        "garc_output_root": str(garc_root),
        "garc_relative_artifacts": dict(EXPECTED_GARC_RELATIVE_PATHS),
        "control": {
            "name": "GARC-v5-tiny-top1 SAME-412/19 control",
            "recognizer_kind": "tiny",
            "consensus_mode": "top1",
            "geometry_mode": "v5",
            "geometry_provider": "enhanced_v5_oof_fold",
            "posterior_consensus_used": False,
            "pepd_geometry_fusion_used": False,
            "base_mask_geometry_fusion_used": False,
            "primary_plan_seed": 20260720,
            "fold_seeds": list(EXPECTED_SEEDS),
            "fold_sample_counts": {
                str(seed): count for seed, count in EXPECTED_JOINT_COUNTS.items()
            },
            "seed21_seed22_materialization": (
                "mechanical replacement of only the frozen progress binding, "
                "enhanced-V5 OOF head/backbone route, geometry fold seed, and "
                "method label; all OCR/decoder/threshold/reference/code settings "
                "remain identical to the preregistered seed20 control"
            ),
            "calibration_reuse": (
                "reuse exact formal tiny_top1 calibration selected on the common "
                "calibration partition before independent validation"
            ),
        },
        "cohort": {
            "samples": 412,
            "groups": 19,
            "joint_summary_sha256": EXPECTED_JOINT_SUMMARY_SHA256,
            "cohort_sha256": EXPECTED_COHORT_SHA256,
            "mapping_sha256": EXPECTED_MAPPING_SHA256,
        },
        "promotion": {
            "protocol": _binding(common_protocol),
            "must_run_after_control_summary_and_seal_verify": True,
            "training_started_by_chain": False,
        },
        "source_bindings": _source_bindings(),
        "parent_garc_source_order": _garc_source_order_audit(),
        "absence_attestation": {
            "paths_absent": absent,
            "all_expected_garc_result_and_control_paths_absent": True,
        },
        "audit": {
            "process_wait_started": False,
            "gpu_initialized": False,
            "training_started": False,
            "inference_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "restricted_namespace_images_opened": 0,
            "feishu_message_sent": False,
        },
    }
    output = guard_public_path(
        output_path, label="control preregistration", must_exist=False
    )
    atomic_new_json(output, value)
    return output


def load_preregistration(path: Path) -> tuple[Path, dict[str, Any]]:
    prereg_path = guard_public_path(path, label="control preregistration")
    value = strict_json(prereg_path)
    require(value.get("schema_version") == 1, "control preregistration schema drift")
    require(value.get("protocol") == PREREGISTRATION_PROTOCOL, "control preregistration identity drift")
    require(
        value.get("status")
        == "frozen_before_garc_independent_validation_outputs_existed",
        "control preregistration was not frozen before GARC validation",
    )
    _utc(str(value.get("created_utc") or ""), label="preregistration time")
    process = value.get("garc_process")
    require(isinstance(process, Mapping), "GARC process identity absent")
    require(int(process.get("pid", 0)) > 0, "GARC process PID drift")
    _utc(str(process.get("start_utc") or ""), label="GARC process start")
    _digest(process.get("command_line_sha256"), label="GARC command line")
    require(
        process.get("required_script_name")
        == "run_garc_full_auto_public_event_driven.ps1",
        "GARC process script drift",
    )
    require(
        value.get("garc_relative_artifacts") == EXPECTED_GARC_RELATIVE_PATHS,
        "GARC relative artifact contract drift",
    )
    _assert_safe_root(
        Path(str(value.get("garc_output_root") or "")),
        label="preregistered GARC output root",
        must_exist=False,
    )
    control = value.get("control")
    require(isinstance(control, Mapping), "control identity absent")
    require(
        (
            control.get("recognizer_kind"),
            control.get("consensus_mode"),
            control.get("geometry_mode"),
            control.get("geometry_provider"),
        )
        == ("tiny", "top1", "v5", "enhanced_v5_oof_fold"),
        "control algorithm identity drift",
    )
    require(
        control.get("posterior_consensus_used") is False
        and control.get("pepd_geometry_fusion_used") is False
        and control.get("base_mask_geometry_fusion_used") is False,
        "control unexpectedly enables posterior/fusion logic",
    )
    require(
        tuple(int(seed) for seed in control.get("fold_seeds", []))
        == EXPECTED_SEEDS,
        "control fold roster drift",
    )
    expected_counts = {str(k): v for k, v in EXPECTED_JOINT_COUNTS.items()}
    require(control.get("fold_sample_counts") == expected_counts, "control fold count drift")
    cohort = value.get("cohort")
    require(isinstance(cohort, Mapping), "control cohort identity absent")
    require((cohort.get("samples"), cohort.get("groups")) == (412, 19), "control cohort inventory drift")
    require(cohort.get("joint_summary_sha256") == EXPECTED_JOINT_SUMMARY_SHA256, "joint summary drift")
    require(cohort.get("cohort_sha256") == EXPECTED_COHORT_SHA256, "joint cohort drift")
    require(cohort.get("mapping_sha256") == EXPECTED_MAPPING_SHA256, "joint mapping drift")
    source_bindings = value.get("source_bindings")
    require(isinstance(source_bindings, Mapping), "control source bindings absent")
    require(set(source_bindings) == set(_SOURCE_PATHS), "control source roster drift")
    for name, expected_path in _SOURCE_PATHS.items():
        observed = _verify_binding(source_bindings[name], label=f"control source {name}")
        require(_same_path(observed, expected_path), f"control source path drift: {name}")
    protocol_path = _verify_binding(
        value.get("promotion", {}).get("protocol", {}),
        label="common-split promotion protocol",
    )
    require(_same_path(protocol_path, COMMON_SPLIT_PROTOCOL), "common-split protocol path drift")
    require(
        value.get("promotion", {}).get("must_run_after_control_summary_and_seal_verify")
        is True
        and value.get("promotion", {}).get("training_started_by_chain") is False,
        "control-to-promotion ordering contract drift",
    )
    order = value.get("parent_garc_source_order")
    require(
        isinstance(order, Mapping)
        and order.get("control_plan_and_calibration_precede_independent_validation")
        is True,
        "parent GARC source-order proof absent",
    )
    require(_garc_source_order_audit() == dict(order), "parent GARC source-order proof drift")
    absence = value.get("absence_attestation")
    absent_paths = absence.get("paths_absent") if isinstance(absence, Mapping) else None
    require(
        isinstance(absence, Mapping)
        and absence.get("all_expected_garc_result_and_control_paths_absent") is True
        and isinstance(absent_paths, Mapping)
        and set(absent_paths) == set(EXPECTED_GARC_RELATIVE_PATHS)
        and all(value is True for value in absent_paths.values()),
        "pre-result absence attestation drift",
    )
    audit = value.get("audit")
    require(isinstance(audit, Mapping), "preregistration audit absent")
    for key in (
        "process_wait_started",
        "gpu_initialized",
        "training_started",
        "inference_started",
        "feishu_message_sent",
    ):
        require(audit.get(key) is False, f"preregistration unexpectedly reports {key}")
    for key in ("images_opened", "annotations_opened", "restricted_namespace_images_opened"):
        require(audit.get(key) == 0, f"preregistration access audit drift: {key}")
    return prereg_path, value


def preflight(*, preregistration_path: Path, output_path: Path) -> Path:
    prereg_path, prereg = load_preregistration(preregistration_path)
    # Static protocol validation only; no public split or model artifact opens.
    from experiments import garc_common_split_promotion as promotion
    from experiments import garc_common_split_progress as common

    _, protocol = common.load_protocol(COMMON_SPLIT_PROTOCOL)
    contract = promotion.audit_static_contract(protocol)
    result = {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "validated_no_wait_no_data_no_gpu_no_inference_no_training_no_notification",
        "preregistration": _binding(prereg_path),
        "control": dict(prereg["control"]),
        "cohort": dict(prereg["cohort"]),
        "promotion_contract": contract,
        "deferred_until_parent_garc_completion": [
            "GARC final-summary authentication",
            "seed21/seed22 mechanical control-plan materialization",
            "public independent-validation inference",
            "formal SAME-412/19 scoring and seal",
            "common-split promotion decision",
        ],
        "audit": {
            "process_wait_started": False,
            "gpu_initialized": False,
            "training_started": False,
            "inference_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "restricted_namespace_images_opened": 0,
            "feishu_message_sent": False,
        },
    }
    output = guard_public_path(output_path, label="control preflight", must_exist=False)
    atomic_new_json(output, result)
    return output


def assert_control_plan(plan: Mapping[str, Any], *, expected_seed: int) -> None:
    """Validate the exact no-posterior/no-fusion control identity."""

    require(int(expected_seed) in EXPECTED_SEEDS, "unexpected control seed")
    require(
        plan.get("execution_mode") == "formal_frozen",
        "control plan is not formal",
    )
    garc = plan.get("garc")
    require(isinstance(garc, Mapping), "control GARC descriptor absent")
    require(
        (
            garc.get("recognizer_kind"),
            garc.get("consensus_mode"),
            garc.get("geometry_mode"),
            garc.get("geometry_provider"),
        )
        == ("tiny", "top1", "v5", "enhanced_v5_oof_fold"),
        "control plan is not frozen v5-tiny-top1",
    )
    require(
        garc.get("effective_decoder_protocol")
        == "garc_top1_arithmetic_ransac_control_v1",
        "control decoder protocol drift",
    )
    require(int(garc.get("posterior_top_k", -1)) == 5, "bound top-K width drift")
    fold = garc.get("geometry_fold")
    require(isinstance(fold, Mapping), "control geometry fold absent")
    require(int(fold.get("pepd_seed", -1)) == int(expected_seed), "control fold seed drift")
    require(fold.get("joint_shard_all_component_unseen") is True, "control fold unseen proof absent")
    require(
        set((garc.get("artifacts") or {}))
        == {"detector", "recognizer", "geometry", "geometry_backbone"},
        "control unexpectedly binds fusion-only artifacts",
    )
    require(
        plan.get("reference", {}).get("mode") == "method_internal_auto",
        "control reference mode drift",
    )
    require(plan.get("joint_oof", {}).get("status") == "bound_for_overlap_audit", "control joint cohort is unbound")
    require(
        plan.get("audit", {}).get("caller_numeric_range_permitted") is False
        and plan.get("audit", {}).get("caller_geometry_permitted") is False
        and plan.get("audit", {}).get("caller_reference_packet_permitted") is False,
        "control accepts supervised deployment inputs",
    )


def validate_control_report_contract(report: Mapping[str, Any]) -> dict[str, float | str]:
    """Return the exact promotion metrics or fail closed on schema/cohort drift."""

    from experiments import garc_common_split_progress as common

    metrics = common._pilot_metrics(report, label="SAME-412/19 tiny-top1 control")
    require(
        metrics["joint_mapping_sha256"] == EXPECTED_JOINT_SUMMARY_SHA256,
        "control report uses a different 412/19 mapping identity",
    )
    require(
        report.get("evidence_eligibility", {}).get("range_component_claim") is False,
        "control falsely promotes the fixed-fold 1080 range table",
    )
    require(
        report.get("claim_boundaries", {}).get("full_1080_end_to_end_claim_allowed")
        is False,
        "control falsely promotes 1080 rows to all-component OOF",
    )
    require(
        report.get("audit", {}).get("numeric_range_values_supplied_to_model") == 0
        and report.get("audit", {}).get("manual_geometry_supplied_to_model") == 0,
        "control report used supervised inference inputs",
    )
    return metrics


def _load_handoff_source_plans(
    *, garc_root: Path, final_summary: Mapping[str, Any]
) -> dict[int, tuple[Path, dict[str, Any]]]:
    from experiments.garc_full_auto_public import load_plan

    external = final_summary.get("external_comparison_handoff")
    require(isinstance(external, Mapping), "GARC external handoff binding absent")
    handoff = external.get("handoff")
    require(isinstance(handoff, Mapping), "GARC handoff summary binding absent")
    summary_path = garc_root / EXPECTED_GARC_RELATIVE_PATHS["external_handoff_summary"]
    seal_path = garc_root / EXPECTED_GARC_RELATIVE_PATHS["external_handoff_seal"]
    require(summary_path.is_file() and seal_path.is_file(), "GARC handoff bundle absent")
    require(handoff.get("root") and _same_path(handoff["root"], summary_path.parent), "GARC handoff root drift")
    require(handoff.get("summary_sha256") == sha256_file(summary_path), "GARC handoff summary hash drift")
    require(handoff.get("seal_sha256") == sha256_file(seal_path), "GARC handoff seal hash drift")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("status") == "label_free_handoff_sealed", "GARC handoff incomplete")
    require(seal.get("status") == "sealed", "GARC handoff is not sealed")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "GARC handoff seal drift")
    require(
        summary.get("garc", {}).get("all_412_progress_geometry_head_and_backbone_group_unseen")
        is True,
        "GARC handoff lacks all-component OOF proof",
    )
    require(summary.get("audit", {}).get("restricted_namespace_images_opened") == 0, "GARC handoff reports restricted access")
    routes: dict[int, tuple[Path, dict[str, Any]]] = {}
    for binding in summary.get("garc", {}).get("source_runs", []):
        require(isinstance(binding, Mapping), "GARC handoff source-run binding drift")
        plan_path = guard_public_path(Path(str(binding.get("plan") or "")), label="GARC route donor plan")
        require(binding.get("plan_sha256") == sha256_file(plan_path), "GARC route donor plan hash drift")
        checked, plan = load_plan(plan_path)
        seed = int(plan.get("garc", {}).get("geometry_fold", {}).get("pepd_seed", -1))
        require(seed in EXPECTED_SEEDS, "GARC route donor seed drift")
        require(seed not in routes, "duplicate GARC route donor seed")
        require(plan["garc"]["geometry_provider"] == "enhanced_v5_oof_fold", "route donor is not enhanced-V5 OOF")
        progress_sha = plan["progress_component"]["binding"]["artifact_sha256"]["checkpoint"]
        geometry_backbone_sha = plan["garc"]["artifacts"]["geometry_backbone"]["sha256"]
        require(progress_sha == geometry_backbone_sha, "route donor progress/geometry backbone mismatch")
        routes[seed] = (checked, plan)
    require(set(routes) == set(EXPECTED_SEEDS), "GARC handoff lacks three fold donors")
    return routes


def _freeze_spec(
    *, seed20_control: Mapping[str, Any], donor: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    require(seed in (20260721, 20260722), "only secondary control folds are materialized")
    assert_control_plan(seed20_control, expected_seed=20260720)
    require(int(donor["garc"]["geometry_fold"]["pepd_seed"]) == seed, "donor seed drift")
    return {
        "seed": seed,
        "method_name": f"GARC-v5-tiny-top1-control-fold{seed}+auto-ref",
        "protocol_path": seed20_control["parent_protocol"]["path"],
        "progress_binding_path": donor["progress_component"]["binding_file"]["path"],
        "progress_factory_path": seed20_control["progress_factory"]["path"],
        "progress_factory_function": seed20_control["progress_factory"]["function"],
        "detector_checkpoint": seed20_control["garc"]["artifacts"]["detector"]["path"],
        "recognizer_checkpoint": seed20_control["garc"]["artifacts"]["recognizer"]["path"],
        "geometry_checkpoint": donor["garc"]["artifacts"]["geometry"]["path"],
        "geometry_backbone_checkpoint": donor["garc"]["artifacts"]["geometry_backbone"]["path"],
        "geometry_oof_summary_path": seed20_control["garc"]["geometry_fold"]["oof_summary"]["path"],
        "joint_oof_summary_path": seed20_control["joint_oof"]["summary"]["path"],
        "reference_detector_sha256": seed20_control["reference"]["detector_sha256"],
        "execution_mode": seed20_control["execution_mode"],
        "device": seed20_control["garc"]["device"],
        "input_size": int(seed20_control["garc"]["input_size"]),
        "detector_threshold": float(seed20_control["garc"]["detector_threshold"]),
        "posterior_top_k": int(seed20_control["garc"]["posterior_top_k"]),
        "consensus_config": dict(seed20_control["garc"]["consensus_config"]),
        "donor_plan_sha256": canonical_sha256(donor),
    }


def prepare_execution(
    *,
    preregistration_path: Path,
    garc_summary_path: Path,
    output_root: Path,
) -> Path:
    """Authenticate completed GARC metadata and freeze a label-free run manifest."""

    from experiments import garc_common_split_promotion as promotion
    from experiments import evaluate_garc_full_auto_public as evaluator
    from experiments.garc_full_auto_public import load_plan

    prereg_path, prereg = load_preregistration(preregistration_path)
    summary_path, final_summary, candidate_path, _ = promotion.authenticate_garc_final(
        garc_summary_path
    )
    garc_root = _assert_safe_root(
        Path(prereg["garc_output_root"]), label="GARC output root", must_exist=True
    )
    require(_same_path(summary_path, garc_root / EXPECTED_GARC_RELATIVE_PATHS["summary"]), "GARC final summary path drift")
    selection_path = guard_public_path(
        garc_root / EXPECTED_GARC_RELATIVE_PATHS["recognizer_selection"],
        label="GARC recognizer selection",
    )
    selection = strict_json(selection_path)
    require(selection.get("protocol") == GARC_RECOGNIZER_SELECTION_PROTOCOL, "recognizer selection protocol drift")
    require(selection.get("status") == "frozen_calibration_only_selection", "recognizer selection is not frozen")
    require(selection.get("audit", {}).get("independent_validation_artifacts_opened") == 0, "recognizer selection touched validation")
    require(selection.get("audit", {}).get("fallback_is_tiny_top1") is True, "tiny-top1 control role drift")
    control_entry = selection.get("tiny_top1")
    require(isinstance(control_entry, Mapping), "tiny-top1 calibration candidate absent")
    expected_plan = garc_root / EXPECTED_GARC_RELATIVE_PATHS["control_seed20_plan"]
    expected_calibration = garc_root / EXPECTED_GARC_RELATIVE_PATHS["control_calibration"]
    require(_same_path(control_entry.get("plan", ""), expected_plan), "tiny-top1 plan path drift")
    require(_same_path(control_entry.get("calibration", ""), expected_calibration), "tiny-top1 calibration path drift")
    require(control_entry.get("plan_sha256") == sha256_file(expected_plan), "tiny-top1 plan hash drift")
    require(control_entry.get("calibration_sha256") == sha256_file(expected_calibration), "tiny-top1 calibration hash drift")
    plan_path, seed20_plan = load_plan(expected_plan)
    assert_control_plan(seed20_plan, expected_seed=20260720)
    candidate = evaluator._calibration_candidate(
        plan_path, expected_calibration, allow_smoke=False
    )
    require(
        (candidate["recognizer_kind"], candidate["consensus_mode"], candidate["geometry_mode"])
        == ("tiny", "top1", "v5"),
        "formal control calibration identity drift",
    )
    routes = _load_handoff_source_plans(
        garc_root=garc_root, final_summary=final_summary
    )
    root = _assert_safe_root(output_root, label="control output root", must_exist=False)
    require(not root.exists(), f"refusing to overwrite control output: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("plans", "predictions", "logs"):
        (root / directory).mkdir(parents=False, exist_ok=False)
    plans = {
        "20260720": _binding(plan_path),
        "20260721": {
            "path": str(root / "plans/control-v5-tiny-top1.seed_20260721.plan.json"),
            "sha256": None,
        },
        "20260722": {
            "path": str(root / "plans/control-v5-tiny-top1.seed_20260722.plan.json"),
            "sha256": None,
        },
    }
    manifest = {
        "schema_version": 1,
        "protocol": EXECUTION_MANIFEST_PROTOCOL,
        "status": "prepared_before_control_inference",
        "preregistration": _binding(prereg_path),
        "garc_summary": _binding(summary_path),
        "candidate_validation": _binding(candidate_path),
        "recognizer_selection": _binding(selection_path),
        "control_calibration": _binding(expected_calibration),
        "control_range_variant_sha256": candidate["range_variant_sha256"],
        "plans": plans,
        "freeze_specs": {
            str(seed): _freeze_spec(
                seed20_control=seed20_plan,
                donor=routes[seed][1],
                seed=seed,
            )
            for seed in (20260721, 20260722)
        },
        "route_donors": {
            str(seed): _binding(routes[seed][0]) for seed in EXPECTED_SEEDS
        },
        "prediction_roots": {
            "20260720": str(root / "predictions/primary-seed-20260720"),
            "20260721": str(root / "predictions/joint-shard-seed-20260721"),
            "20260722": str(root / "predictions/joint-shard-seed-20260722"),
        },
        "control_report": str(root / "validation.json"),
        "cohort": dict(prereg["cohort"]),
        "audit": {
            "control_chosen_from_candidate_validation_metrics": False,
            "configuration_fixed_by_preregistration": True,
            "only_fold_routes_materialized_after_parent_completion": True,
            "gpu_initialized": False,
            "training_started": False,
            "inference_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "restricted_namespace_images_opened": 0,
            "feishu_message_sent": False,
        },
    }
    manifest_path = root / "execution_manifest.json"
    atomic_new_json(manifest_path, manifest)
    atomic_new_json(
        root / "execution_manifest.seal.json",
        {
            "schema_version": 1,
            "protocol": EXECUTION_MANIFEST_PROTOCOL,
            "status": "sealed_before_control_inference",
            "manifest_sha256": sha256_file(manifest_path),
            "preregistration_sha256": sha256_file(prereg_path),
        },
    )
    return manifest_path


def _load_execution_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = guard_public_path(path, label="control execution manifest")
    manifest = strict_json(manifest_path)
    require(manifest.get("protocol") == EXECUTION_MANIFEST_PROTOCOL, "control manifest protocol drift")
    require(manifest.get("status") == "prepared_before_control_inference", "control manifest status drift")
    seal_path = guard_public_path(
        manifest_path.with_name("execution_manifest.seal.json"),
        label="control execution manifest seal",
    )
    seal = strict_json(seal_path)
    require(seal.get("protocol") == EXECUTION_MANIFEST_PROTOCOL, "control manifest seal protocol drift")
    require(seal.get("status") == "sealed_before_control_inference", "control manifest not sealed")
    require(seal.get("manifest_sha256") == sha256_file(manifest_path), "control manifest seal drift")
    prereg_path = _verify_binding(manifest.get("preregistration", {}), label="control preregistration")
    load_preregistration(prereg_path)
    return manifest_path, manifest


def materialize_plans(*, manifest_path: Path) -> Path:
    """Create only seed21/22 plans from the preregistered mechanical rule."""

    from experiments import evaluate_garc_full_auto_public as evaluator
    from experiments import garc_full_auto_public as garc

    checked_manifest, manifest = _load_execution_manifest(manifest_path)
    seed20_path, seed20 = garc.load_plan(
        Path(manifest["plans"]["20260720"]["path"])
    )
    require(
        manifest["plans"]["20260720"]["sha256"] == sha256_file(seed20_path),
        "seed20 control plan hash drift",
    )
    assert_control_plan(seed20, expected_seed=20260720)
    expected_variant = str(manifest["control_range_variant_sha256"])
    require(evaluator.range_variant_sha256(seed20) == expected_variant, "seed20 control variant drift")
    created: dict[str, dict[str, str]] = {}
    for seed in (20260721, 20260722):
        key = str(seed)
        spec = manifest["freeze_specs"][key]
        donor_path, donor = garc.load_plan(Path(manifest["route_donors"][key]["path"]))
        require(manifest["route_donors"][key]["sha256"] == sha256_file(donor_path), "route donor hash drift")
        require(canonical_sha256(donor) == spec["donor_plan_sha256"], "route donor content drift")
        rebuilt = _freeze_spec(seed20_control=seed20, donor=donor, seed=seed)
        require(rebuilt == spec, "mechanical control freeze specification drift")
        output = Path(manifest["plans"][key]["path"])
        garc.freeze_plan(
            protocol_path=Path(spec["protocol_path"]),
            output_path=output,
            progress_binding_path=Path(spec["progress_binding_path"]),
            progress_factory_path=Path(spec["progress_factory_path"]),
            progress_factory_function=str(spec["progress_factory_function"]),
            detector_checkpoint=Path(spec["detector_checkpoint"]),
            recognizer_checkpoint=Path(spec["recognizer_checkpoint"]),
            geometry_checkpoint=Path(spec["geometry_checkpoint"]),
            geometry_provider="enhanced_v5_oof_fold",
            geometry_backbone_checkpoint=Path(spec["geometry_backbone_checkpoint"]),
            geometry_oof_summary_path=Path(spec["geometry_oof_summary_path"]),
            geometry_oof_seed=seed,
            recognizer_kind="tiny",
            consensus_mode="top1",
            geometry_mode="v5",
            method_name=str(spec["method_name"]),
            reference_mode="method_internal_auto",
            reference_detector_sha256=str(spec["reference_detector_sha256"]),
            execution_mode=str(spec["execution_mode"]),
            device=str(spec["device"]),
            input_size=int(spec["input_size"]),
            detector_threshold=float(spec["detector_threshold"]),
            posterior_top_k=int(spec["posterior_top_k"]),
            consensus_config_path=None,
            joint_oof_summary_path=Path(spec["joint_oof_summary_path"]),
        )
        plan_path, plan = garc.load_plan(output)
        assert_control_plan(plan, expected_seed=seed)
        require(evaluator.range_variant_sha256(plan) == expected_variant, "materialized control range-family drift")
        created[key] = _binding(plan_path)
    output = checked_manifest.with_name("materialization.json")
    atomic_new_json(
        output,
        {
            "schema_version": 1,
            "protocol": MATERIALIZATION_PROTOCOL,
            "status": "secondary_fold_plans_frozen_before_control_inference",
            "execution_manifest": _binding(checked_manifest),
            "plans": created,
            "range_variant_sha256": expected_variant,
            "audit": {
                "configuration_selected_from_candidate_metrics": False,
                "only_fold_specific_routes_changed": True,
                "images_opened": 0,
                "annotations_opened": 0,
                "restricted_namespace_images_opened": 0,
                "inference_started": False,
            },
        },
    )
    return output


def _prediction_binding(root: Path) -> dict[str, Any]:
    resolved = _assert_safe_root(root, label="control prediction root", must_exist=True)
    summary = guard_public_path(resolved / "summary.json", label="control prediction summary")
    seal = guard_public_path(resolved / "seal.json", label="control prediction seal")
    return {
        "path": str(resolved),
        "summary_sha256": sha256_file(summary),
        "seal_sha256": sha256_file(seal),
    }


def seal_control(*, manifest_path: Path, control_report_path: Path) -> Path:
    """Authenticate the real formal report and seal it for promotion."""

    from experiments import evaluate_garc_full_auto_public as evaluator
    from experiments import garc_full_auto_public as garc

    checked_manifest, manifest = _load_execution_manifest(manifest_path)
    root = checked_manifest.parent
    materialization_path = guard_public_path(
        root / "materialization.json", label="control plan materialization"
    )
    materialization = strict_json(materialization_path)
    require(materialization.get("protocol") == MATERIALIZATION_PROTOCOL, "control materialization protocol drift")
    require(materialization.get("status") == "secondary_fold_plans_frozen_before_control_inference", "control plans were not frozen before inference")
    report_path = guard_public_path(control_report_path, label="formal control report")
    require(_same_path(report_path, manifest["control_report"]), "formal control report path drift")
    report = strict_json(report_path)
    metrics = validate_control_report_contract(report)
    calibration_path = _verify_binding(manifest["control_calibration"], label="control calibration")
    plans: dict[int, tuple[Path, dict[str, Any]]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        plan_path = Path(manifest["plans"][key]["path"])
        if seed == 20260720:
            require(manifest["plans"][key]["sha256"] == sha256_file(plan_path), "seed20 control plan drift")
        else:
            require(materialization["plans"][key]["sha256"] == sha256_file(plan_path), f"seed{seed} control plan drift")
        checked, plan = garc.load_plan(plan_path)
        assert_control_plan(plan, expected_seed=seed)
        require(evaluator.range_variant_sha256(plan) == manifest["control_range_variant_sha256"], "control family drift")
        prediction_root = Path(manifest["prediction_roots"][key])
        summary, rows, _, _ = garc.load_prediction_bundle(
            plan_path=checked,
            prediction_root=prediction_root,
            expected_partition="independent_validation",
            allow_smoke=False,
        )
        require(summary.get("mode") == "formal", f"seed{seed} prediction is not formal")
        if seed == 20260720:
            require((len(rows), len({row["group_id"] for row in rows})) == (1080, 50), "primary control inventory drift")
        else:
            eligible = [row for row in rows if row.get("joint_oof_eligible")]
            require(len(eligible) == EXPECTED_JOINT_COUNTS[seed], f"seed{seed} joint shard inventory drift")
        plans[seed] = (checked, plan)
        predictions[key] = _prediction_binding(prediction_root)
    primary_path = plans[20260720][0]
    require(report.get("parent_plan", {}).get("sha256") == sha256_file(primary_path), "control report parent-plan drift")
    require(_same_path(report.get("parent_plan", {}).get("path", ""), primary_path), "control report parent-plan path drift")
    require(report.get("frozen_calibration", {}).get("sha256") == sha256_file(calibration_path), "control report calibration drift")
    require(_same_path(report.get("frozen_calibration", {}).get("path", ""), calibration_path), "control report calibration path drift")
    require(_same_path(report.get("validation_predictions", {}).get("path", ""), manifest["prediction_roots"]["20260720"]), "control report primary prediction drift")
    require(report.get("code", {}).get("sha256") == sha256_file(GARC_EVALUATOR), "control evaluator source drift")
    summary_value = {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "status": "formal_same_412_control_sealed",
        "method": {
            "name": "GARC-v5-tiny-top1 SAME-412/19 control",
            "recognizer_kind": "tiny",
            "consensus_mode": "top1",
            "geometry_mode": "v5",
            "posterior_consensus_used": False,
            "geometry_fusion_used": False,
        },
        "preregistration": dict(manifest["preregistration"]),
        "execution_manifest": _binding(checked_manifest),
        "materialization": _binding(materialization_path),
        "garc_summary": dict(manifest["garc_summary"]),
        "candidate_validation": dict(manifest["candidate_validation"]),
        "control_report": _binding(report_path),
        "control_calibration": _binding(calibration_path),
        "plans": {str(seed): _binding(plans[seed][0]) for seed in EXPECTED_SEEDS},
        "predictions": predictions,
        "cohort": {
            "samples": 412,
            "groups": 19,
            "joint_mapping_sha256": metrics["joint_mapping_sha256"],
            "joint_summary_sha256": EXPECTED_JOINT_SUMMARY_SHA256,
            "cohort_sha256": EXPECTED_COHORT_SHA256,
            "mapping_sha256": EXPECTED_MAPPING_SHA256,
            "samples_by_seed": {
                str(seed): count for seed, count in EXPECTED_JOINT_COUNTS.items()
            },
        },
        "promotion_metrics": metrics,
        "chronology": {
            "method_preregistered_before_parent_validation_outputs": True,
            "seed20_plan_and_calibration_frozen_before_parent_validation": True,
            "seed21_seed22_plans_frozen_before_control_predictions": True,
            "all_control_predictions_authenticated_before_public_values_opened": True,
            "promotion_not_run_before_this_summary_and_seal": True,
        },
        "audit": {
            "candidate_metrics_used_to_choose_control": False,
            "numeric_range_values_supplied_to_model": 0,
            "manual_geometry_supplied_to_model": 0,
            "training_started": False,
            "restricted_namespace_images_opened": 0,
            "feishu_message_sent": False,
        },
    }
    summary_path = root / "summary.json"
    atomic_new_json(summary_path, summary_value)
    artifacts = {
        "summary.json": sha256_file(summary_path),
        "validation.json": sha256_file(report_path),
        "execution_manifest.json": sha256_file(checked_manifest),
        "materialization.json": sha256_file(materialization_path),
        "calibration": sha256_file(calibration_path),
        **{
            f"plan_seed_{seed}": sha256_file(plans[seed][0])
            for seed in EXPECTED_SEEDS
        },
        **{
            f"prediction_summary_seed_{seed}": predictions[str(seed)]["summary_sha256"]
            for seed in EXPECTED_SEEDS
        },
        **{
            f"prediction_seal_seed_{seed}": predictions[str(seed)]["seal_sha256"]
            for seed in EXPECTED_SEEDS
        },
    }
    atomic_new_json(
        root / "seal.json",
        {
            "schema_version": 1,
            "protocol": SEAL_PROTOCOL,
            "status": "sealed_before_common_split_promotion",
            "artifacts": artifacts,
            "bundle_sha256": canonical_sha256(artifacts),
        },
    )
    return summary_path


def verify_control_bundle(root: Path) -> Path:
    bundle_root = _assert_safe_root(root, label="control bundle", must_exist=True)
    summary_path = guard_public_path(bundle_root / "summary.json", label="control summary")
    seal_path = guard_public_path(bundle_root / "seal.json", label="control seal")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == SUMMARY_PROTOCOL, "control summary protocol drift")
    require(summary.get("status") == "formal_same_412_control_sealed", "control summary incomplete")
    require(seal.get("protocol") == SEAL_PROTOCOL, "control seal protocol drift")
    require(seal.get("status") == "sealed_before_common_split_promotion", "control was not sealed before promotion")
    artifacts = seal.get("artifacts")
    require(isinstance(artifacts, Mapping), "control seal artifacts absent")
    require(artifacts.get("summary.json") == sha256_file(summary_path), "control summary seal drift")
    require(seal.get("bundle_sha256") == canonical_sha256(dict(artifacts)), "control bundle digest drift")
    method = summary.get("method")
    require(
        isinstance(method, Mapping)
        and (
            method.get("recognizer_kind"),
            method.get("consensus_mode"),
            method.get("geometry_mode"),
        )
        == ("tiny", "top1", "v5")
        and method.get("posterior_consensus_used") is False
        and method.get("geometry_fusion_used") is False,
        "sealed control method identity drift",
    )
    prereg_path = _verify_binding(summary.get("preregistration", {}), label="sealed preregistration")
    load_preregistration(prereg_path)
    manifest_path = _verify_binding(
        summary.get("execution_manifest", {}), label="sealed execution manifest"
    )
    materialization_path = _verify_binding(
        summary.get("materialization", {}), label="sealed plan materialization"
    )
    calibration_path = _verify_binding(
        summary.get("control_calibration", {}), label="sealed control calibration"
    )
    require(
        artifacts.get("execution_manifest.json") == sha256_file(manifest_path),
        "control execution-manifest seal drift",
    )
    require(
        artifacts.get("materialization.json") == sha256_file(materialization_path),
        "control materialization seal drift",
    )
    require(
        artifacts.get("calibration") == sha256_file(calibration_path),
        "control calibration seal drift",
    )
    report_path = _verify_binding(summary.get("control_report", {}), label="sealed control report")
    require(artifacts.get("validation.json") == sha256_file(report_path), "control report seal drift")
    metrics = validate_control_report_contract(strict_json(report_path))
    require(metrics == summary.get("promotion_metrics"), "sealed control metric drift")
    require(summary.get("chronology", {}).get("promotion_not_run_before_this_summary_and_seal") is True, "control/promotion chronology proof absent")
    require(summary.get("audit", {}).get("restricted_namespace_images_opened") == 0, "control bundle reports restricted access")
    for seed in EXPECTED_SEEDS:
        plan_path = _verify_binding(summary["plans"][str(seed)], label=f"sealed control plan {seed}")
        require(artifacts.get(f"plan_seed_{seed}") == sha256_file(plan_path), f"seed{seed} plan seal drift")
        prediction_root = _assert_safe_root(
            Path(summary["predictions"][str(seed)]["path"]),
            label=f"sealed control prediction {seed}",
            must_exist=True,
        )
        require(
            summary["predictions"][str(seed)]["summary_sha256"]
            == sha256_file(prediction_root / "summary.json"),
            f"seed{seed} prediction summary binding drift",
        )
        require(
            summary["predictions"][str(seed)]["seal_sha256"]
            == sha256_file(prediction_root / "seal.json"),
            f"seed{seed} prediction seal binding drift",
        )
        require(
            artifacts.get(f"prediction_summary_seed_{seed}")
            == sha256_file(prediction_root / "summary.json"),
            f"seed{seed} prediction summary seal drift",
        )
        require(
            artifacts.get(f"prediction_seal_seed_{seed}")
            == sha256_file(prediction_root / "seal.json"),
            f"seed{seed} prediction seal drift",
        )
    return summary_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-preregistration")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--garc-output-root", type=Path, required=True)
    freeze.add_argument("--garc-process-id", type=int, required=True)
    freeze.add_argument("--garc-process-start-utc", required=True)
    freeze.add_argument("--garc-process-command-sha256", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--preregistration", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--preregistration", type=Path, required=True)
    prepare.add_argument("--garc-summary", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    materialize = commands.add_parser("materialize-plans")
    materialize.add_argument("--manifest", type=Path, required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--control-report", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze-preregistration":
        output = freeze_preregistration(
            output_path=args.output,
            garc_output_root=args.garc_output_root,
            garc_process_id=args.garc_process_id,
            garc_process_start_utc=args.garc_process_start_utc,
            garc_process_command_sha256=args.garc_process_command_sha256,
        )
    elif args.command == "preflight":
        output = preflight(
            preregistration_path=args.preregistration,
            output_path=args.output,
        )
    elif args.command == "prepare":
        output = prepare_execution(
            preregistration_path=args.preregistration,
            garc_summary_path=args.garc_summary,
            output_root=args.output_root,
        )
    elif args.command == "materialize-plans":
        output = materialize_plans(manifest_path=args.manifest)
    elif args.command == "seal":
        output = seal_control(
            manifest_path=args.manifest,
            control_report_path=args.control_report,
        )
    else:
        output = verify_control_bundle(args.root)
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXECUTION_MANIFEST_PROTOCOL",
    "EXPECTED_JOINT_SUMMARY_SHA256",
    "EXPECTED_SEEDS",
    "MATERIALIZATION_PROTOCOL",
    "PREFLIGHT_PROTOCOL",
    "PREREGISTRATION_PROTOCOL",
    "SEAL_PROTOCOL",
    "SUMMARY_PROTOCOL",
    "assert_control_plan",
    "freeze_preregistration",
    "load_preregistration",
    "materialize_plans",
    "preflight",
    "prepare_execution",
    "seal_control",
    "validate_control_report_contract",
    "verify_control_bundle",
]
