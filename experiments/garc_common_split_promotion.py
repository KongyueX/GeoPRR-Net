"""Authorize common-split training only after the frozen GARC-412 pilot passes.

This is a CPU-only evidence and authorization stage.  It never trains, runs
inference, sends notifications, or opens image pixels/annotations.  A passing
decision authorizes three fresh common-split PEPD backbones and a separately
matched V5/enhanced head for each backbone; historical seed-20260720 PEPD and
legacy V5 weights remain forbidden.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
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
from experiments import garc_common_split_progress as common


PROTOCOL: Final[str] = "garc_common_split_promotion_decision_v1"
AUTHORIZATION_PROTOCOL: Final[str] = "garc_common_split_training_authorization_v1"
PREFLIGHT_PROTOCOL: Final[str] = "garc_common_split_promotion_preflight_v1"
GARC_FINAL_PROTOCOL: Final[str] = "garc_full_auto_public_event_chain_summary_v1"
EXPECTED_JOINT_SUMMARY_SHA256: Final[str] = (
    "5fb85223204a52ce168b8aed10835054238aef0513bade2fec6047ece323dccb"
)
EXPECTED_JOINT_COHORT_SHA256: Final[str] = (
    "680cfb8f77fb20517903f62db52ab0931bf46f8b3b8db1e43d8040c0f7d03363"
)
EXPECTED_JOINT_MAPPING_SHA256: Final[str] = (
    "9d88314f339d3b173d9d04b3ea4aa3ecac15bb0affca8166bbf82bf0425a9de9"
)
EXPECTED_GATE: Final[dict[str, float]] = {
    "candidate_minimum_range_coverage": 0.80,
    "candidate_minimum_pair_rounded_exact_full_denominator": 0.70,
    "candidate_minimum_pair_rounded_exact_conditional": 0.85,
    "candidate_minimum_end_to_end_coverage": 0.75,
    "candidate_maximum_end_to_end_nmae_failure_penalty_1": 0.30,
    "minimum_absolute_end_to_end_nmae_improvement": 0.03,
    "minimum_relative_end_to_end_nmae_improvement": 0.10,
    "maximum_end_to_end_coverage_regression": 0.01,
}
EXPECTED_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "range_coverage",
        "pair_exact_full_denominator",
        "pair_exact_conditional",
        "end_to_end_coverage",
        "end_to_end_nmae",
        "absolute_end_to_end_improvement",
        "relative_end_to_end_improvement",
        "coverage_noninferiority",
    }
)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _verify_binding(binding: Mapping[str, Any], *, label: str) -> Path:
    require(isinstance(binding, Mapping), f"{label} binding is absent")
    path = guard_public_path(Path(str(binding.get("path") or "")), label=label)
    require(binding.get("sha256") == sha256_file(path), f"{label} hash drift")
    return path


def _assert_zero_restricted_audit(value: Mapping[str, Any], *, label: str) -> None:
    audit = value.get("audit")
    require(isinstance(audit, Mapping), f"{label} audit is absent")
    keys = (
        "restricted_namespace_images_opened",
        "field_images_read",
        "field_samples_used",
        "test_split_images_read",
        "test_samples_used",
        "sealed_images_read",
        "sealed_samples_used",
        "confirmatory_images_read",
        "confirmatory_samples_used",
    )
    for key in keys:
        if key in audit:
            require(audit.get(key) == 0, f"{label} reports forbidden access: {key}")


def audit_static_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the promotion contract without touching any dataset artifact."""

    gate = protocol.get("pilot_gate")
    require(isinstance(gate, Mapping), "pilot gate is absent")
    require(int(gate.get("cohort_samples", -1)) == 412, "pilot sample contract drift")
    require(int(gate.get("cohort_groups", -1)) == 19, "pilot group contract drift")
    require(gate.get("all_rows_jointly_unseen_required") is True, "jointly-unseen gate disabled")
    require(gate.get("same_cohort_control_required") is True, "same-cohort control gate disabled")
    for key, expected in EXPECTED_GATE.items():
        observed = gate.get(key)
        require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12),
            f"pilot threshold drift: {key}",
        )
    method = protocol.get("method")
    require(isinstance(method, Mapping), "method contract is absent")
    require(tuple(int(seed) for seed in method.get("seeds", [])) == common.EXPECTED_SEEDS, "three-seed roster drift")
    legacy = method.get("legacy_architecture_bindings")
    require(isinstance(legacy, Mapping), "legacy-weight policy is absent")
    pepd = legacy.get("pepd_seed20260720")
    v5 = legacy.get("v5_seed20261206")
    require(isinstance(pepd, Mapping) and isinstance(v5, Mapping), "legacy checkpoint bindings absent")
    require(pepd.get("formal_weight_reuse_allowed") is False, "legacy seed20 PEPD was authorized")
    require(v5.get("formal_weight_reuse_allowed") is False, "legacy V5 weights were authorized")
    initialization = method.get("initialization_policy")
    require(isinstance(initialization, Mapping), "initialization policy is absent")
    forbidden_text = " ".join(str(item) for item in initialization.get("forbidden", [])).casefold()
    require("legacy syncg checkpoint" in forbidden_text, "legacy SyncG initialization ban absent")
    optimization = protocol.get("optimization")
    require(isinstance(optimization, Mapping), "optimization contract is absent")
    require(optimization.get("gradient_partition") == "algorithm_fit", "gradient partition drift")
    require(optimization.get("selection_partition") == "calibration", "selection partition drift")
    require(optimization.get("report_partition") == "independent_validation", "report partition drift")
    require(int(optimization.get("independent_validation_reveal_count", -1)) == 1, "validation reveal contract drift")
    return {
        "thresholds": dict(EXPECTED_GATE),
        "seeds": list(common.EXPECTED_SEEDS),
        "legacy_pepd_checkpoint_sha256": str(pepd.get("checkpoint_sha256")),
        "legacy_v5_checkpoint_sha256": str(v5.get("checkpoint_sha256")),
        "legacy_pepd_weight_reuse_allowed": False,
        "legacy_v5_weight_reuse_allowed": False,
        "matched_v5_head_required_per_new_pepd_backbone": True,
    }


def preflight(*, protocol_path: Path, output_path: Path) -> Path:
    protocol_file, protocol = common.load_protocol(protocol_path)
    contract = audit_static_contract(protocol)
    result = {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "validated_no_wait_no_data_no_training_no_inference_no_notification",
        "frozen_protocol": _binding(protocol_file),
        "contract": contract,
        "deferred_until_formal_promotion": [
            "GARC process wait",
            "GARC final-summary authentication",
            "public split path/stat audit without pixel decoding",
            "pilot candidate/control gate evaluation",
        ],
        "audit": {
            "process_wait_started": False,
            "gpu_initialized": False,
            "training_started": False,
            "inference_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "field_test_sealed_confirmatory_images_opened": 0,
            "feishu_message_sent": False,
        },
    }
    output = guard_public_path(output_path, label="promotion preflight output", must_exist=False)
    atomic_new_json(output, result)
    return output


def authenticate_garc_final(
    garc_summary_path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    summary_file = guard_public_path(garc_summary_path, label="final GARC summary")
    summary = strict_json(summary_file)
    require(summary.get("schema_version") == 1, "final GARC schema drift")
    require(summary.get("protocol") == GARC_FINAL_PROTOCOL, "final GARC protocol drift")
    require(summary.get("status") == "complete", "final GARC run is incomplete")
    _assert_zero_restricted_audit(summary, label="final GARC summary")
    selected = summary.get("selected")
    require(isinstance(selected, Mapping), "selected GARC method binding absent")
    for name in ("plan", "calibration"):
        _verify_binding(
            {"path": selected.get(name), "sha256": selected.get(f"{name}_sha256")},
            label=f"selected GARC {name}",
        )
    validation = summary.get("validation")
    require(isinstance(validation, Mapping), "final GARC validation binding absent")
    validation_file = _verify_binding(validation, label="final GARC validation")
    report = strict_json(validation_file)
    common._pilot_metrics(report, label="final GARC candidate")
    _assert_zero_restricted_audit(report, label="final GARC validation")
    require(int(validation.get("joint_samples", -1)) == 412, "final summary joint sample drift")
    require(int(validation.get("joint_groups", -1)) == 19, "final summary joint group drift")
    claims = validation.get("paper_claim_allowed")
    require(isinstance(claims, Mapping), "final GARC paper-claim audit absent")
    require(claims.get("joint_412_all_component_oof_end_to_end") is True, "final GARC 412 claim ineligible")
    require(claims.get("full_1080_all_component_unseen") is False, "fixed-fold 1080 falsely promoted")
    nmae = report["metrics"]["joint_oof_end_to_end_frozen_acceptance"][
        "reading_nmae_full_denominator_failure_penalty_1"
    ]
    require(
        math.isclose(
            float(validation.get("joint_reading_nmae_full_denominator")),
            float(nmae),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "final GARC summary/validation NMAE drift",
    )
    return summary_file, summary, validation_file, report


def resolve_control_report(
    summary: Mapping[str, Any], explicit_path: Path | None
) -> Path:
    if explicit_path is not None:
        return guard_public_path(explicit_path, label="same-cohort pilot control")
    binding = summary.get("common_split_pilot", {}).get("control_report")
    require(isinstance(binding, Mapping), "same-cohort pilot control is absent")
    return _verify_binding(binding, label="same-cohort pilot control")


def authenticate_joint_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the three distinct frozen 412/19 identity artifacts.

    The summary, cohort roster, and progress mapping are deliberately kept as
    separate bindings.  Treating one digest as an alias for another would make
    the promotion gate either impossible to pass or vulnerable to binding the
    wrong artifact class.
    """

    selected = summary.get("selected")
    require(isinstance(selected, Mapping), "selected GARC method binding absent")
    plan_path = _verify_binding(
        {
            "path": selected.get("plan"),
            "sha256": selected.get("plan_sha256"),
        },
        label="selected GARC plan",
    )
    plan = strict_json(plan_path)
    joint = plan.get("joint_oof")
    require(isinstance(joint, Mapping), "selected GARC plan lacks joint OOF binding")
    joint_summary_path = _verify_binding(
        joint.get("summary"), label="joint OOF identity summary"
    )
    joint_summary = strict_json(joint_summary_path)
    require(
        joint_summary.get("protocol") == "garc_progress_joint_oof_cohort_v1",
        "joint OOF identity summary protocol drift",
    )
    require(
        joint_summary.get("status") == "frozen_label_free_joint_oof_cohort",
        "joint OOF identity summary is not frozen",
    )
    inventory = joint_summary.get("joint_oof")
    require(isinstance(inventory, Mapping), "joint OOF identity inventory absent")
    require(
        (int(inventory.get("samples", -1)), int(inventory.get("groups", -1)))
        == (412, 19),
        "joint OOF identity inventory drift",
    )
    artifacts = joint_summary.get("artifacts")
    require(isinstance(artifacts, Mapping), "joint OOF identity artifacts absent")
    cohort_path = _verify_binding(
        artifacts.get("cohort"), label="joint OOF cohort roster"
    )
    mapping_path = _verify_binding(
        artifacts.get("mapping"), label="joint OOF progress mapping"
    )
    return {
        "summary": _binding(joint_summary_path),
        "cohort": _binding(cohort_path),
        "mapping": _binding(mapping_path),
        "samples": 412,
        "groups": 19,
    }


def evaluate_promotion(
    protocol: Mapping[str, Any],
    *,
    garc_summary_path: Path,
    control_report_path: Path | None,
    expected_joint_summary_sha256: str = EXPECTED_JOINT_SUMMARY_SHA256,
    expected_joint_cohort_sha256: str = EXPECTED_JOINT_COHORT_SHA256,
    expected_joint_mapping_sha256: str = EXPECTED_JOINT_MAPPING_SHA256,
) -> dict[str, Any]:
    contract = audit_static_contract(protocol)
    summary_file, summary, candidate_file, _ = authenticate_garc_final(
        garc_summary_path
    )
    joint_identity = authenticate_joint_identity(summary)
    expected_identities = {
        "summary": expected_joint_summary_sha256,
        "cohort": expected_joint_cohort_sha256,
        "mapping": expected_joint_mapping_sha256,
    }
    for name, expected in expected_identities.items():
        require(
            joint_identity[name]["sha256"] == expected,
            f"pilot joint {name} is not the frozen GARC 412/19 {name} identity",
        )
    control_file = resolve_control_report(summary, control_report_path)
    gate = common.evaluate_pilot_gate(
        protocol,
        candidate_path=candidate_file,
        control_path=control_file,
    )
    require(
        gate.get("cohort", {}).get("joint_mapping_sha256")
        == joint_identity["mapping"]["sha256"],
        "pilot cohort mapping differs from the authenticated GARC mapping",
    )
    checks = gate.get("checks")
    require(isinstance(checks, Mapping) and set(checks) == EXPECTED_CHECKS, "pilot check roster drift")
    passed = gate.get("training_allowed") is True and all(
        value is True for value in checks.values()
    )
    return {
        "status": "authorized" if passed else "rejected",
        "training_allowed": passed,
        "garc_summary": _binding(summary_file),
        "candidate_report": _binding(candidate_file),
        "control_report": _binding(control_file),
        "joint_identity": joint_identity,
        "pilot_gate": gate,
        "contract": contract,
        "failed_checks": sorted(key for key, value in checks.items() if value is not True),
    }


def _authorization(
    *,
    protocol_file: Path,
    protocol: Mapping[str, Any],
    gate_report_path: Path,
    gate_report_sha256: str,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    require(decision.get("training_allowed") is True, "cannot authorize a rejected pilot")
    legacy = protocol["method"]["legacy_architecture_bindings"]
    forbidden = [
        {
            "component": "legacy_pepd_seed20260720",
            "path": legacy["pepd_seed20260720"]["checkpoint_path"],
            "sha256": legacy["pepd_seed20260720"]["checkpoint_sha256"],
        },
        {
            "component": "legacy_v5_seed20261206",
            "path": legacy["v5_seed20261206"]["checkpoint_path"],
            "sha256": legacy["v5_seed20261206"]["checkpoint_sha256"],
        },
    ]
    seed_runs = [
        {
            "seed": seed,
            "required_order": [
                "train_fresh_common_split_pepd_on_algorithm_fit",
                "train_matched_v5_enhanced_head_on_this_exact_pepd_feature_distribution",
                "fit_progress_fusion_on_algorithm_fit",
                "select_checkpoint_threshold_and_fusion_on_calibration_only",
                "seal_seed_bundle_before_any_independent_validation_access",
            ],
            "pepd_initialization": "allowed generic immutable provenance or deterministic fresh initialization; never legacy SyncG weights",
            "v5_enhanced_head_initialization": "deterministic_random_init",
            "v5_enhanced_head_random_seed": seed,
            "matched_backbone_checkpoint_required": True,
            "legacy_checkpoint_weights_loaded": 0,
        }
        for seed in common.EXPECTED_SEEDS
    ]
    return {
        "schema_version": 1,
        "protocol": AUTHORIZATION_PROTOCOL,
        "status": "authorized_for_common_split_training",
        "training_allowed": True,
        "frozen_protocol": _binding(protocol_file),
        "pilot_gate_report": {
            "path": str(gate_report_path),
            "sha256": gate_report_sha256,
        },
        "candidate_report": dict(decision["candidate_report"]),
        "control_report": dict(decision["control_report"]),
        "joint_identity": dict(decision["joint_identity"]),
        "seeds": list(common.EXPECTED_SEEDS),
        "seed_runs": seed_runs,
        "partition_contract": {
            "gradient_updates": "algorithm_fit only",
            "selection": "calibration only; zero gradient updates",
            "independent_validation": "zero access until all three seeds and ensemble are sealed",
            "development_excluded": "zero access",
        },
        "forbidden_weight_reuse": forbidden,
        "required_attestations": {
            "common_split_pepd_trained_per_seed": True,
            "matched_v5_enhanced_head_trained_per_exact_pepd_backbone": True,
            "legacy_pepd_checkpoint_weights_loaded": 0,
            "legacy_v5_checkpoint_weights_loaded": 0,
            "independent_validation_reveal_count_before_freeze": 0,
        },
        "audit": {
            "gpu_initialized_by_promotion": False,
            "training_started_by_promotion": False,
            "inference_started_by_promotion": False,
            "field_test_sealed_confirmatory_images_opened": 0,
            "feishu_message_sent": False,
        },
    }


def promote(
    *,
    protocol_path: Path,
    garc_summary_path: Path,
    control_report_path: Path | None,
    output_root: Path,
    expected_joint_summary_sha256: str = EXPECTED_JOINT_SUMMARY_SHA256,
    expected_joint_cohort_sha256: str = EXPECTED_JOINT_COHORT_SHA256,
    expected_joint_mapping_sha256: str = EXPECTED_JOINT_MAPPING_SHA256,
) -> Path:
    protocol_file, protocol = common.load_protocol(protocol_path)
    decision = evaluate_promotion(
        protocol,
        garc_summary_path=garc_summary_path,
        control_report_path=control_report_path,
        expected_joint_summary_sha256=expected_joint_summary_sha256,
        expected_joint_cohort_sha256=expected_joint_cohort_sha256,
        expected_joint_mapping_sha256=expected_joint_mapping_sha256,
    )
    # This authenticates only public SyncG/train rosters and path existence; it
    # does not decode an image or open an annotation/value-bearing manifest.
    split_audit = common.audit_public_split(protocol)
    output = guard_public_path(output_root, label="promotion output root", must_exist=False, expect_file=False)
    require(not output.exists(), f"refusing to overwrite promotion output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    require(not staging.exists(), f"promotion staging path exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        gate_report = {
            "schema_version": 1,
            "protocol": common.PREFLIGHT_PROTOCOL,
            "status": (
                "training_gate_passed"
                if decision["training_allowed"]
                else "training_not_authorized"
            ),
            "training_allowed": bool(decision["training_allowed"]),
            "frozen_protocol": _binding(protocol_file),
            "split_audit": split_audit,
            "pilot_gate": decision["pilot_gate"],
            "joint_identity": decision["joint_identity"],
            "audit": {
                "gpu_initialized": False,
                "training_started": False,
                "images_opened": 0,
                "annotations_opened": 0,
                "field_test_sealed_confirmatory_images_opened": 0,
            },
        }
        gate_path = staging / "pilot_gate.json"
        atomic_new_json(gate_path, gate_report)
        authorization_path: Path | None = None
        if decision["training_allowed"]:
            authorization_path = staging / "training_authorization.json"
            atomic_new_json(
                authorization_path,
                _authorization(
                    protocol_file=protocol_file,
                    protocol=protocol,
                    gate_report_path=output / "pilot_gate.json",
                    gate_report_sha256=sha256_file(gate_path),
                    decision=decision,
                ),
            )
        final_decision = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": decision["status"],
            "training_allowed": bool(decision["training_allowed"]),
            "frozen_protocol": _binding(protocol_file),
            "garc_summary": decision["garc_summary"],
            "candidate_report": decision["candidate_report"],
            "control_report": decision["control_report"],
            "joint_identity": decision["joint_identity"],
            "pilot_gate_report": {
                "path": str(output / "pilot_gate.json"),
                "sha256": sha256_file(gate_path),
            },
            "training_authorization": (
                None
                if authorization_path is None
                else {
                    "path": str(output / "training_authorization.json"),
                    "sha256": sha256_file(authorization_path),
                }
            ),
            "failed_checks": decision["failed_checks"],
            "training_process_started": False,
            "audit": {
                "public_split_authenticated_without_pixel_decode": True,
                "gpu_initialized": False,
                "training_started": False,
                "inference_started": False,
                "restricted_namespace_images_opened": 0,
                "feishu_message_sent": False,
            },
        }
        decision_path = staging / "decision.json"
        atomic_new_json(decision_path, final_decision)
        artifacts = {
            "decision.json": sha256_file(decision_path),
            "pilot_gate.json": sha256_file(gate_path),
        }
        if authorization_path is not None:
            artifacts["training_authorization.json"] = sha256_file(authorization_path)
        seal = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "sealed",
            "artifacts": artifacts,
            "bundle_sha256": canonical_sha256(artifacts),
        }
        atomic_new_json(staging / "seal.json", seal)
        os.replace(staging, output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise
    return output / "decision.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--protocol", type=Path, default=common.DEFAULT_PROTOCOL)
    pre.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("promote")
    run.add_argument("--protocol", type=Path, default=common.DEFAULT_PROTOCOL)
    run.add_argument("--garc-summary", type=Path, required=True)
    run.add_argument("--control-report", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "preflight":
        output = preflight(protocol_path=args.protocol, output_path=args.output)
    else:
        output = promote(
            protocol_path=args.protocol,
            garc_summary_path=args.garc_summary,
            control_report_path=args.control_report,
            output_root=args.output_root,
        )
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
    "AUTHORIZATION_PROTOCOL",
    "EXPECTED_GATE",
    "EXPECTED_JOINT_COHORT_SHA256",
    "EXPECTED_JOINT_MAPPING_SHA256",
    "EXPECTED_JOINT_SUMMARY_SHA256",
    "PREFLIGHT_PROTOCOL",
    "PROTOCOL",
    "audit_static_contract",
    "authenticate_garc_final",
    "authenticate_joint_identity",
    "evaluate_promotion",
    "preflight",
    "promote",
]
