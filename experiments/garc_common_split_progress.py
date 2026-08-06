"""CPU-only protocol auditor for the GARC common-split progress experiment.

This module does not train a model and never opens an image or annotation.  It
authenticates the frozen SyncG/train 551/100/50 group split, enforces the
412-image pilot gate, and validates later training/prediction artifacts before
the one-shot 1,080-image independent report is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    PUBLIC_IMAGE_ROOT,
    atomic_new_json,
    canonical_sha256,
    guard_public_path,
    load_frozen_protocol,
    load_partition_roster,
    require,
    resolve_public_image,
    sha256_file,
    strict_json,
    strict_jsonl,
    verify_bound_file,
)


PROTOCOL: Final[str] = "garc_common_split_progress_training_v1"
PREFLIGHT_PROTOCOL: Final[str] = "garc_common_split_progress_preflight_v1"
TRAINING_RUN_PROTOCOL: Final[str] = "garc_common_split_progress_run_v1"
PREDICTION_PROTOCOL: Final[str] = "garc_common_split_progress_predictions_v1"
PREDICTION_ROW_PROTOCOL: Final[str] = "garc_common_split_progress_prediction_row_v1"
VALIDATE_ONLY_PROTOCOL: Final[str] = "garc_common_split_progress_validate_only_v1"
GARC_VALIDATION_PROTOCOL: Final[str] = "garc_full_auto_public_validation_v1"
DEFAULT_PROTOCOL: Final[Path] = Path(__file__).with_name(
    "garc_common_split_progress_protocol.json"
)
EXPECTED_PARTITIONS: Final[dict[str, tuple[int, int]]] = {
    "algorithm_fit": (12_176, 551),
    "calibration": (2_224, 100),
    "independent_validation": (1_080, 50),
    "development_excluded": (520, 24),
}
EXPECTED_SEEDS: Final[tuple[int, ...]] = (20260816, 20260817, 20260818)
ALLOWED_PRETRAINING_PROVENANCE: Final[frozenset[str]] = frozenset(
    {"external_generic_immutable", "algorithm_fit_only", "deterministic_random_init"}
)
_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "partition",
        "sample_id",
        "group_id",
        "canonical_roi_sha256",
        "status",
        "prediction_progress",
        "progress_confidence",
        "failure_reason",
        "sample_seconds",
    }
)


def _digest(value: Any, *, label: str) -> str:
    result = str(value or "").strip().casefold()
    require(
        len(result) == 64
        and set(result).issubset(frozenset("0123456789abcdef")),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return result


def _finite_fraction(value: Any, *, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    result = float(value)
    require(0.0 <= result <= 1.0, f"{label} is outside [0,1]")
    return result


def _timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str) and bool(value.strip()), f"{label} is absent")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{label} is not ISO-8601") from error
    require(result.tzinfo is not None, f"{label} must include a timezone")
    return result


def _verify_artifact(binding: Mapping[str, Any], *, label: str) -> Path:
    require(isinstance(binding, Mapping), f"{label} binding is absent")
    path = guard_public_path(Path(str(binding.get("path") or "")), label=label)
    require(sha256_file(path) == _digest(binding.get("sha256"), label=label), f"{label} hash drift")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[Path, dict[str, Any]]:
    protocol_path = guard_public_path(path, label="common-split progress protocol")
    value = strict_json(protocol_path)
    require(value.get("schema_version") == 1, "common-split schema drift")
    require(value.get("protocol") == PROTOCOL, "common-split protocol drift")
    require(
        value.get("status") == "frozen_before_pilot_gate_and_training",
        "common-split protocol is not frozen",
    )
    require(value.get("scope", {}).get("dataset") == "SyncG", "dataset drift")
    require(value.get("scope", {}).get("split") == "train", "split drift")
    require(
        tuple(int(seed) for seed in value.get("method", {}).get("seeds", []))
        == EXPECTED_SEEDS,
        "training seed roster drift",
    )
    require(
        value.get("optimization", {}).get("independent_validation_reveal_count") == 1,
        "independent validation must be revealed exactly once",
    )
    return protocol_path, value


def audit_public_split(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate all four rosters without opening an image or annotation."""

    parent = protocol.get("parent_protocol")
    require(isinstance(parent, Mapping), "parent protocol binding is absent")
    parent_path = guard_public_path(
        Path(str(parent.get("path") or "")), label="parent public split protocol"
    )
    require(
        sha256_file(parent_path)
        == _digest(parent.get("sha256"), label="parent public split protocol"),
        "parent public split protocol hash drift",
    )
    _, parent_value = load_frozen_protocol(parent_path)
    require(parent_value.get("protocol") == parent.get("identity"), "parent identity drift")
    for key in ("syncg_train_manifest", "syncg_train_manifest_protocol"):
        verify_bound_file(parent_value, "source_bindings", key)
    source_protocol_path = verify_bound_file(
        parent_value, "source_bindings", "syncg_train_manifest_protocol"
    )
    source_protocol = strict_json(source_protocol_path)
    require(source_protocol.get("dataset") == "SyncG", "source dataset drift")
    require(source_protocol.get("split") == "train", "source manifest is not SyncG/train")
    require(source_protocol.get("strict_release") is True, "source release is not strict")

    declared_root = guard_public_path(
        Path(str(protocol.get("scope", {}).get("image_root") or "")),
        label="declared SyncG/train image root",
        expect_file=False,
    )
    require(
        declared_root == PUBLIC_IMAGE_ROOT.resolve(strict=True),
        "declared image root differs from SyncG/train",
    )

    audits: dict[str, dict[str, Any]] = {}
    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition, expected in EXPECTED_PARTITIONS.items():
        binding = protocol.get("partitions", {}).get(partition)
        require(isinstance(binding, Mapping), f"missing {partition} binding")
        _, manifest_path, rows, audit = load_partition_roster(parent_path, partition)
        require((audit["samples"], audit["groups"]) == expected, f"{partition} inventory drift")
        require(binding.get("samples") == expected[0], f"{partition}.samples drift")
        require(binding.get("groups") == expected[1], f"{partition}.groups drift")
        for key in ("sha256", "sample_ids_sha256", "group_ids_sha256"):
            observed = sha256_file(manifest_path) if key == "sha256" else audit[key]
            require(observed == binding.get(key), f"{partition}.{key} drift")
        # Resolve/stat every declared image under the one allowed root.  No
        # decoder is called, so this remains a label- and pixel-free audit.
        for row in rows:
            image_path = resolve_public_image(str(row["image_relpath"]))
            require(image_path.is_relative_to(declared_root), f"{partition} image escaped SyncG/train")
        rows_by_partition[partition] = rows
        audits[partition] = {
            "samples": audit["samples"],
            "groups": audit["groups"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "sample_ids_sha256": audit["sample_ids_sha256"],
            "group_ids_sha256": audit["group_ids_sha256"],
            "all_images_resolve_under_syncg_train": True,
        }

    overlaps: dict[str, dict[str, int]] = {}
    for left, right in itertools.combinations(EXPECTED_PARTITIONS, 2):
        left_samples = {str(row["sample_id"]) for row in rows_by_partition[left]}
        right_samples = {str(row["sample_id"]) for row in rows_by_partition[right]}
        left_groups = {str(row["group_id"]) for row in rows_by_partition[left]}
        right_groups = {str(row["group_id"]) for row in rows_by_partition[right]}
        sample_overlap = len(left_samples & right_samples)
        group_overlap = len(left_groups & right_groups)
        require(sample_overlap == 0, f"{left}/{right} sample leakage")
        require(group_overlap == 0, f"{left}/{right} group leakage")
        overlaps[f"{left}__{right}"] = {
            "sample_overlap": sample_overlap,
            "group_overlap": group_overlap,
        }

    legacy = protocol.get("method", {}).get("legacy_architecture_bindings", {})
    require(isinstance(legacy, Mapping), "legacy architecture audit bindings absent")
    reuse = legacy.get(
        "v5_seed20261206"
    )
    require(isinstance(reuse, Mapping), "legacy V5 audit binding is absent")
    require(reuse.get("formal_weight_reuse_allowed") is False, "legacy V5 weights were authorized")
    v5_protocol_path = guard_public_path(
        Path(str(reuse.get("protocol_path") or "")), label="frozen V5 protocol"
    )
    require(
        sha256_file(v5_protocol_path)
        == _digest(reuse.get("protocol_sha256"), label="frozen V5 protocol"),
        "frozen V5 protocol hash drift",
    )
    bound_v5_protocol = verify_bound_file(
        parent_value, "source_bindings", "v5_public_split_protocol"
    )
    require(bound_v5_protocol == v5_protocol_path, "parent/common-split V5 binding drift")
    v5 = strict_json(v5_protocol_path)
    require(
        v5.get("protocol") == "cagh_scalemark_reference_public_v5_tight_roi_v1"
        and v5.get("status") == "frozen_public_only",
        "frozen V5 protocol identity drift",
    )
    checkpoint_path = guard_public_path(
        Path(str(reuse.get("checkpoint_path") or "")), label="frozen V5 head checkpoint"
    )
    summary_path = guard_public_path(
        Path(str(reuse.get("summary_path") or "")), label="frozen V5 head summary"
    )
    require(
        sha256_file(checkpoint_path)
        == _digest(reuse.get("checkpoint_sha256"), label="frozen V5 head checkpoint"),
        "frozen V5 checkpoint hash drift",
    )
    require(
        sha256_file(summary_path)
        == _digest(reuse.get("summary_sha256"), label="frozen V5 head summary"),
        "frozen V5 summary hash drift",
    )
    v5_summary = strict_json(summary_path)
    require(v5_summary.get("status") == "complete", "frozen V5 run incomplete")
    require(v5_summary.get("mode") == "formal", "frozen V5 run is not formal")
    require(v5_summary.get("protocol") == v5.get("protocol"), "frozen V5 run protocol drift")
    require(
        v5_summary.get("artifacts", {}).get("checkpoint_sha256")
        == sha256_file(checkpoint_path),
        "frozen V5 summary/checkpoint drift",
    )

    all_rows = [
        row
        for partition in EXPECTED_PARTITIONS
        for row in rows_by_partition[partition]
    ]
    by_group: dict[str, list[str]] = {}
    for row in all_rows:
        by_group.setdefault(str(row["group_id"]), []).append(str(row["sample_id"]))
    require((len(all_rows), len(by_group)) == (16_000, 725), "V5 source inventory drift")
    split = v5.get("split")
    require(isinstance(split, Mapping), "frozen V5 split absent")
    seed = int(split.get("seed"))
    validation_fraction = float(split.get("validation_fraction"))
    ordered = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    target = max(1, round(len(all_rows) * validation_fraction))
    v5_validation_groups: set[str] = set()
    validation_count = 0
    for group in ordered:
        if validation_count >= target and v5_validation_groups:
            break
        if len(by_group) - len(v5_validation_groups) <= 1:
            break
        v5_validation_groups.add(group)
        validation_count += len(by_group[group])
    v5_fit_groups = set(by_group) - v5_validation_groups
    v5_validation_samples = {
        sample_id
        for group in v5_validation_groups
        for sample_id in by_group[group]
    }
    v5_fit_samples = {
        sample_id for group in v5_fit_groups for sample_id in by_group[group]
    }
    require(
        (len(v5_fit_samples), len(v5_fit_groups)) == (14_400, 651),
        "reconstructed V5 fit inventory drift",
    )
    require(
        (len(v5_validation_samples), len(v5_validation_groups)) == (1_600, 74),
        "reconstructed V5 validation inventory drift",
    )
    for values, key in (
        (v5_fit_samples, "fit_ids_sha256"),
        (v5_validation_samples, "validation_ids_sha256"),
        (v5_fit_groups, "fit_groups_sha256"),
        (v5_validation_groups, "validation_groups_sha256"),
    ):
        require(canonical_sha256(sorted(values)) == split.get(key), f"V5 {key} drift")
    independent_rows = rows_by_partition["independent_validation"]
    independent_samples = {str(row["sample_id"]) for row in independent_rows}
    independent_groups = {str(row["group_id"]) for row in independent_rows}
    fit_group_overlap = independent_groups & v5_fit_groups
    fit_sample_overlap = independent_samples & v5_fit_samples
    require(not fit_group_overlap, "independent validation overlaps frozen V5 fit groups")
    require(not fit_sample_overlap, "independent validation overlaps frozen V5 fit samples")
    require(
        independent_groups.issubset(v5_validation_groups),
        "independent validation is not contained in frozen V5 validation groups",
    )
    require(
        independent_samples.issubset(v5_validation_samples),
        "independent validation is not contained in frozen V5 validation samples",
    )
    v5_reuse_audit = {
        "status": "legacy_v5_split_authenticated_weights_formally_ineligible",
        "protocol": {"path": str(v5_protocol_path), "sha256": sha256_file(v5_protocol_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "v5_fit": {
            "samples": len(v5_fit_samples),
            "groups": len(v5_fit_groups),
            "sample_ids_sha256": canonical_sha256(sorted(v5_fit_samples)),
            "group_ids_sha256": canonical_sha256(sorted(v5_fit_groups)),
        },
        "v5_validation": {
            "samples": len(v5_validation_samples),
            "groups": len(v5_validation_groups),
            "sample_ids_sha256": canonical_sha256(sorted(v5_validation_samples)),
            "group_ids_sha256": canonical_sha256(sorted(v5_validation_groups)),
        },
        "independent_validation": {
            "samples": len(independent_samples),
            "groups": len(independent_groups),
            "all_samples_inside_v5_validation": True,
            "all_groups_inside_v5_validation": True,
            "v5_fit_sample_overlap": 0,
            "v5_fit_group_overlap": 0,
            "v5_fit_sample_overlap_sha256": canonical_sha256(sorted(fit_sample_overlap)),
            "v5_fit_group_overlap_sha256": canonical_sha256(sorted(fit_group_overlap)),
        },
        "historical_pepd_backbone_reused": False,
        "checkpoint_role": "architecture_and_pilot_reference_only",
        "formal_checkpoint_weight_reuse_allowed": False,
        "matched_common_split_v5_head_training_required": True,
    }

    pepd = legacy.get("pepd_seed20260720")
    require(isinstance(pepd, Mapping), "legacy PEPD audit binding absent")
    require(pepd.get("formal_weight_reuse_allowed") is False, "legacy PEPD weights were authorized")
    handoff_path = guard_public_path(
        Path(str(pepd.get("handoff_path") or "")), label="legacy PEPD handoff"
    )
    pepd_checkpoint_path = guard_public_path(
        Path(str(pepd.get("checkpoint_path") or "")), label="legacy PEPD checkpoint"
    )
    require(
        sha256_file(handoff_path)
        == _digest(pepd.get("handoff_sha256"), label="legacy PEPD handoff"),
        "legacy PEPD handoff hash drift",
    )
    require(
        sha256_file(pepd_checkpoint_path)
        == _digest(pepd.get("checkpoint_sha256"), label="legacy PEPD checkpoint"),
        "legacy PEPD checkpoint hash drift",
    )
    handoff = strict_json(handoff_path)
    row = handoff.get("authoritative_runs", {}).get("20260720")
    require(isinstance(row, Mapping), "legacy PEPD seed20260720 handoff absent")
    grouped = row.get("grouped_split")
    require(isinstance(grouped, Mapping), "legacy PEPD grouped split absent")
    require(grouped.get("split_seed") == 20260720, "legacy PEPD split seed drift")
    require(float(grouped.get("validation_fraction")) == 0.1, "legacy PEPD split fraction drift")
    pepd_ordered = sorted(
        by_group,
        key=lambda group: hashlib.sha256(f"20260720:{group}".encode()).digest(),
    )
    pepd_validation_groups: set[str] = set()
    pepd_validation_count = 0
    pepd_target = max(1, round(len(all_rows) * 0.1))
    for group in pepd_ordered:
        if pepd_validation_count >= pepd_target and pepd_validation_groups:
            break
        if len(by_group) - len(pepd_validation_groups) <= 1:
            break
        pepd_validation_groups.add(group)
        pepd_validation_count += len(by_group[group])
    pepd_fit_groups = set(by_group) - pepd_validation_groups
    pepd_validation_samples = {
        sample_id
        for group in pepd_validation_groups
        for sample_id in by_group[group]
    }
    pepd_fit_samples = {
        sample_id for group in pepd_fit_groups for sample_id in by_group[group]
    }
    require(
        (len(pepd_validation_samples), len(pepd_validation_groups)) == (1_625, 73),
        "legacy PEPD validation inventory drift",
    )
    require(
        (len(pepd_fit_samples), len(pepd_fit_groups)) == (14_375, 652),
        "legacy PEPD fit inventory drift",
    )
    require(
        canonical_sha256(sorted(pepd_validation_samples))
        == grouped.get("validation_sample_ids_sha256"),
        "legacy PEPD validation sample hash drift",
    )
    require(
        canonical_sha256(sorted(pepd_fit_samples))
        == grouped.get("train_sample_ids_sha256"),
        "legacy PEPD fit sample hash drift",
    )
    independent_pepd_unseen_samples = independent_samples & pepd_validation_samples
    independent_pepd_unseen_groups = independent_groups & pepd_validation_groups
    independent_pepd_seen_samples = independent_samples & pepd_fit_samples
    independent_pepd_seen_groups = independent_groups & pepd_fit_groups
    require(
        (
            len(independent_pepd_unseen_samples),
            len(independent_pepd_unseen_groups),
            len(independent_pepd_seen_samples),
            len(independent_pepd_seen_groups),
        )
        == (168, 8, 912, 42),
        "legacy PEPD independent exposure inventory drift",
    )
    for key, observed in (
        ("independent_validation_unseen_samples", len(independent_pepd_unseen_samples)),
        ("independent_validation_unseen_groups", len(independent_pepd_unseen_groups)),
        ("independent_validation_seen_samples", len(independent_pepd_seen_samples)),
        ("independent_validation_seen_groups", len(independent_pepd_seen_groups)),
    ):
        require(pepd.get(key) == observed, f"legacy PEPD {key} drift")
    legacy_pepd_audit = {
        "status": "legacy_pepd_exposure_authenticated_weights_formally_ineligible",
        "handoff": {"path": str(handoff_path), "sha256": sha256_file(handoff_path)},
        "checkpoint": {
            "path": str(pepd_checkpoint_path),
            "sha256": sha256_file(pepd_checkpoint_path),
        },
        "independent_validation": {
            "samples": len(independent_samples),
            "groups": len(independent_groups),
            "unseen_samples": len(independent_pepd_unseen_samples),
            "unseen_groups": len(independent_pepd_unseen_groups),
            "seen_samples": len(independent_pepd_seen_samples),
            "seen_groups": len(independent_pepd_seen_groups),
            "unseen_sample_ids_sha256": canonical_sha256(sorted(independent_pepd_unseen_samples)),
            "unseen_group_ids_sha256": canonical_sha256(sorted(independent_pepd_unseen_groups)),
            "seen_sample_ids_sha256": canonical_sha256(sorted(independent_pepd_seen_samples)),
            "seen_group_ids_sha256": canonical_sha256(sorted(independent_pepd_seen_groups)),
        },
        "formal_checkpoint_weight_reuse_allowed": False,
        "common_split_pepd_retraining_required": True,
        "matched_v5_head_retraining_required": True,
    }

    return {
        "status": "strict_public_split_authenticated",
        "parent_protocol": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
        },
        "source_manifest_protocol": {
            "path": str(source_protocol_path),
            "sha256": sha256_file(source_protocol_path),
            "dataset": "SyncG",
            "split": "train",
        },
        "partitions": audits,
        "pairwise_overlap": overlaps,
        "legacy_v5_audit": v5_reuse_audit,
        "legacy_pepd_audit": legacy_pepd_audit,
        "all_pairwise_sample_overlaps_zero": True,
        "all_pairwise_group_overlaps_zero": True,
        "images_opened": 0,
        "annotations_opened": 0,
        "restricted_namespace_images_opened": 0,
    }


def _pilot_metrics(report: Mapping[str, Any], *, label: str) -> dict[str, float | str]:
    require(report.get("protocol") == GARC_VALIDATION_PROTOCOL, f"{label} protocol drift")
    require(report.get("status") == "independent_validation_complete", f"{label} incomplete")
    require(report.get("mode") == "formal", f"{label} is not formal")
    require(report.get("claim_eligible") is True, f"{label} is not claim eligible")
    overlap = report.get("overlap_audit")
    require(isinstance(overlap, Mapping), f"{label} overlap audit absent")
    require(overlap.get("jointly_unseen_samples") == 412, f"{label} pilot sample drift")
    require(overlap.get("jointly_unseen_groups") == 19, f"{label} pilot group drift")
    require(overlap.get("joint_cohort_complete") is True, f"{label} pilot incomplete")
    eligibility = report.get("evidence_eligibility")
    require(isinstance(eligibility, Mapping), f"{label} eligibility absent")
    require(eligibility.get("joint_oof_end_to_end_claim") is True, f"{label} E2E ineligible")
    audit = report.get("audit")
    require(isinstance(audit, Mapping), f"{label} audit absent")
    require(
        audit.get("validation_predictions_verified_before_public_values_opened") is True,
        f"{label} prediction authentication proof absent",
    )
    require(
        audit.get("calibration_frozen_before_validation_values_opened") is True,
        f"{label} calibration chronology proof absent",
    )
    require(
        audit.get("restricted_namespace_images_opened") == 0,
        f"{label} reports restricted-namespace access",
    )
    metrics = report.get("metrics")
    require(isinstance(metrics, Mapping), f"{label} metrics absent")
    # The common-split trigger deliberately requires range metrics on the same
    # 412/19 cohort.  The legacy 1,080 range-only table is not a substitute.
    range_metrics = metrics.get("joint_oof_range_frozen_acceptance")
    require(isinstance(range_metrics, Mapping), f"{label} lacks 412/19 range metrics")
    require(range_metrics.get("samples") == 412, f"{label} range sample drift")
    require(range_metrics.get("groups") == 19, f"{label} range group drift")
    e2e = metrics.get("joint_oof_end_to_end_frozen_acceptance")
    require(isinstance(e2e, Mapping), f"{label} E2E metrics absent")
    require(e2e.get("samples") == 412, f"{label} E2E sample drift")
    require(e2e.get("groups") == 19, f"{label} E2E group drift")
    mapping_sha = _digest(overlap.get("joint_mapping_sha256"), label=f"{label} joint mapping")
    return {
        "joint_mapping_sha256": mapping_sha,
        "range_coverage": _finite_fraction(range_metrics.get("coverage"), label=f"{label} range coverage"),
        "pair_full": _finite_fraction(
            range_metrics.get("pair_rounded_exact_full_denominator"),
            label=f"{label} pair exact full denominator",
        ),
        "pair_conditional": _finite_fraction(
            range_metrics.get("pair_rounded_exact_conditional"),
            label=f"{label} pair exact conditional",
        ),
        "e2e_coverage": _finite_fraction(e2e.get("coverage"), label=f"{label} E2E coverage"),
        "e2e_nmae": _finite_fraction(
            e2e.get("reading_nmae_full_denominator_failure_penalty_1"),
            label=f"{label} E2E failure-penalty NMAE",
        ),
    }


def evaluate_pilot_gate(
    protocol: Mapping[str, Any],
    *,
    candidate_path: Path,
    control_path: Path,
) -> dict[str, Any]:
    candidate_file = guard_public_path(candidate_path, label="pilot candidate report")
    control_file = guard_public_path(control_path, label="pilot control report")
    candidate = _pilot_metrics(strict_json(candidate_file), label="pilot candidate")
    control = _pilot_metrics(strict_json(control_file), label="pilot control")
    require(
        candidate["joint_mapping_sha256"] == control["joint_mapping_sha256"],
        "pilot candidate/control cohorts differ",
    )
    gate = protocol["pilot_gate"]
    absolute_improvement = float(control["e2e_nmae"]) - float(candidate["e2e_nmae"])
    relative_improvement = absolute_improvement / max(float(control["e2e_nmae"]), 1e-12)
    coverage_regression = float(control["e2e_coverage"]) - float(candidate["e2e_coverage"])
    checks = {
        "range_coverage": float(candidate["range_coverage"])
        >= float(gate["candidate_minimum_range_coverage"]),
        "pair_exact_full_denominator": float(candidate["pair_full"])
        >= float(gate["candidate_minimum_pair_rounded_exact_full_denominator"]),
        "pair_exact_conditional": float(candidate["pair_conditional"])
        >= float(gate["candidate_minimum_pair_rounded_exact_conditional"]),
        "end_to_end_coverage": float(candidate["e2e_coverage"])
        >= float(gate["candidate_minimum_end_to_end_coverage"]),
        "end_to_end_nmae": float(candidate["e2e_nmae"])
        <= float(gate["candidate_maximum_end_to_end_nmae_failure_penalty_1"]),
        "absolute_end_to_end_improvement": absolute_improvement
        >= float(gate["minimum_absolute_end_to_end_nmae_improvement"]),
        "relative_end_to_end_improvement": relative_improvement
        >= float(gate["minimum_relative_end_to_end_nmae_improvement"]),
        "coverage_noninferiority": coverage_regression
        <= float(gate["maximum_end_to_end_coverage_regression"]),
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "training_allowed": passed,
        "candidate_report": {"path": str(candidate_file), "sha256": sha256_file(candidate_file)},
        "control_report": {"path": str(control_file), "sha256": sha256_file(control_file)},
        "cohort": {
            "samples": int(gate["cohort_samples"]),
            "groups": int(gate["cohort_groups"]),
            "joint_mapping_sha256": candidate["joint_mapping_sha256"],
        },
        "candidate": candidate,
        "control": control,
        "derived": {
            "absolute_end_to_end_nmae_improvement": absolute_improvement,
            "relative_end_to_end_nmae_improvement": relative_improvement,
            "end_to_end_coverage_regression": coverage_regression,
        },
        "checks": checks,
        "thresholds": {
            "candidate_minimum_range_coverage": float(
                gate["candidate_minimum_range_coverage"]
            ),
            "candidate_minimum_pair_rounded_exact_full_denominator": float(
                gate["candidate_minimum_pair_rounded_exact_full_denominator"]
            ),
            "candidate_minimum_pair_rounded_exact_conditional": float(
                gate["candidate_minimum_pair_rounded_exact_conditional"]
            ),
            "candidate_minimum_end_to_end_coverage": float(
                gate["candidate_minimum_end_to_end_coverage"]
            ),
            "candidate_maximum_end_to_end_nmae_failure_penalty_1": float(
                gate["candidate_maximum_end_to_end_nmae_failure_penalty_1"]
            ),
            "minimum_absolute_end_to_end_nmae_improvement": float(
                gate["minimum_absolute_end_to_end_nmae_improvement"]
            ),
            "minimum_relative_end_to_end_nmae_improvement": float(
                gate["minimum_relative_end_to_end_nmae_improvement"]
            ),
            "maximum_end_to_end_coverage_regression": float(
                gate["maximum_end_to_end_coverage_regression"]
            ),
        },
    }


def preflight(
    *,
    protocol_path: Path,
    output_path: Path,
    pilot_candidate: Path | None = None,
    pilot_control: Path | None = None,
) -> Path:
    protocol_file, protocol = load_protocol(protocol_path)
    split_audit = audit_public_split(protocol)
    require(
        (pilot_candidate is None) == (pilot_control is None),
        "pilot candidate and control must be supplied together",
    )
    if pilot_candidate is None:
        gate_result: dict[str, Any] = {
            "status": "waiting_for_formal_412_19_candidate_and_control",
            "training_allowed": False,
            "checks": {},
        }
    else:
        gate_result = evaluate_pilot_gate(
            protocol,
            candidate_path=pilot_candidate,
            control_path=pilot_control,  # type: ignore[arg-type]
        )
    result = {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": (
            "training_gate_passed"
            if gate_result["training_allowed"]
            else "training_not_authorized"
        ),
        "training_allowed": bool(gate_result["training_allowed"]),
        "frozen_protocol": {
            "path": str(protocol_file),
            "sha256": sha256_file(protocol_file),
        },
        "split_audit": split_audit,
        "pilot_gate": gate_result,
        "resource_estimate_rtx4060_8gb": protocol["resource_estimate_rtx4060_8gb"],
        "audit": {
            "gpu_initialized": False,
            "training_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "field_test_sealed_confirmatory_images_opened": 0,
        },
    }
    output = guard_public_path(output_path, label="common-split preflight output", must_exist=False)
    atomic_new_json(output, result)
    return output


def _validate_partition_binding(
    declared: Mapping[str, Any], protocol: Mapping[str, Any], partition: str
) -> None:
    expected = protocol["partitions"][partition]
    for key in ("samples", "groups", "sha256", "sample_ids_sha256", "group_ids_sha256"):
        require(declared.get(key) == expected.get(key), f"run {partition}.{key} drift")


def validate_training_run(
    protocol_file: Path,
    protocol: Mapping[str, Any],
    run_manifest_path: Path,
) -> tuple[Path, dict[str, Any]]:
    run_file = guard_public_path(run_manifest_path, label="common-split training run")
    run = strict_json(run_file)
    require(run.get("schema_version") == 1, "training run schema drift")
    require(run.get("protocol") == TRAINING_RUN_PROTOCOL, "training run protocol drift")
    require(run.get("status") == "training_and_calibration_frozen", "training run incomplete")
    parent = run.get("parent_protocol")
    require(isinstance(parent, Mapping), "training parent protocol absent")
    require(parent.get("sha256") == sha256_file(protocol_file), "training protocol hash drift")
    require(Path(str(parent.get("path") or "")).resolve(strict=True) == protocol_file, "training protocol path drift")

    gate_binding = run.get("pilot_gate_report")
    gate_path = _verify_artifact(gate_binding, label="pilot gate report")
    gate_report = strict_json(gate_path)
    require(gate_report.get("protocol") == PREFLIGHT_PROTOCOL, "pilot gate report drift")
    require(gate_report.get("status") == "training_gate_passed", "pilot gate did not pass")
    require(gate_report.get("training_allowed") is True, "training began without passing pilot gate")
    require(
        gate_report.get("frozen_protocol", {}).get("sha256") == sha256_file(protocol_file),
        "pilot gate/protocol drift",
    )
    pilot_gate = gate_report.get("pilot_gate")
    require(isinstance(pilot_gate, Mapping), "authenticated pilot gate details absent")
    require(pilot_gate.get("status") == "passed", "pilot gate details report failure")
    require(pilot_gate.get("training_allowed") is True, "pilot gate details deny training")
    checks = pilot_gate.get("checks")
    require(
        isinstance(checks, Mapping)
        and set(checks)
        == {
            "range_coverage",
            "pair_exact_full_denominator",
            "pair_exact_conditional",
            "end_to_end_coverage",
            "end_to_end_nmae",
            "absolute_end_to_end_improvement",
            "relative_end_to_end_improvement",
            "coverage_noninferiority",
        }
        and all(value is True for value in checks.values()),
        "pilot gate checks are incomplete or failed",
    )
    authorization_path = _verify_artifact(
        run.get("promotion_authorization"),
        label="common-split promotion authorization",
    )
    authorization = strict_json(authorization_path)
    require(
        authorization.get("protocol")
        == "garc_common_split_training_authorization_v1",
        "promotion authorization protocol drift",
    )
    require(
        authorization.get("status") == "authorized_for_common_split_training"
        and authorization.get("training_allowed") is True,
        "promotion authorization denies training",
    )
    require(
        authorization.get("frozen_protocol", {}).get("sha256")
        == sha256_file(protocol_file),
        "promotion authorization/protocol drift",
    )
    require(
        authorization.get("pilot_gate_report", {}).get("sha256")
        == sha256_file(gate_path),
        "promotion authorization/pilot gate drift",
    )
    require(
        tuple(int(seed) for seed in authorization.get("seeds", []))
        == EXPECTED_SEEDS,
        "promotion authorization seed roster drift",
    )
    forbidden = authorization.get("forbidden_weight_reuse")
    require(isinstance(forbidden, list), "promotion forbidden-weight roster absent")
    forbidden_hashes = {
        str(row.get("sha256"))
        for row in forbidden
        if isinstance(row, Mapping)
    }
    require(
        {
            str(
                protocol["method"]["legacy_architecture_bindings"][
                    "pepd_seed20260720"
                ]["checkpoint_sha256"]
            ),
            str(
                protocol["method"]["legacy_architecture_bindings"][
                    "v5_seed20261206"
                ]["checkpoint_sha256"]
            ),
        }.issubset(forbidden_hashes),
        "promotion authorization omits a legacy checkpoint ban",
    )

    partitions = run.get("partitions")
    require(isinstance(partitions, Mapping), "training partition bindings absent")
    for partition in EXPECTED_PARTITIONS:
        require(isinstance(partitions.get(partition), Mapping), f"run {partition} binding absent")
        _validate_partition_binding(partitions[partition], protocol, partition)

    seeds = run.get("seed_runs")
    require(isinstance(seeds, list), "seed run roster absent")
    require([int(row.get("seed")) for row in seeds if isinstance(row, Mapping)] == list(EXPECTED_SEEDS), "seed run roster drift")
    seed_finished: list[datetime] = []
    for row in seeds:
        require(isinstance(row, Mapping), "non-object seed run")
        seed = int(row["seed"])
        require(row.get("status") == "complete", f"seed {seed} incomplete")
        _verify_artifact(row.get("checkpoint"), label=f"seed {seed} checkpoint")
        _verify_artifact(row.get("training_summary"), label=f"seed {seed} summary")
        component_roles = row.get("component_roles")
        require(
            component_roles
            == {
                "pepd_backbone": "trained_on_algorithm_fit",
                "v5_enhanced_head": "trained_on_algorithm_fit_with_this_exact_pepd_backbone",
                "progress_fusion": "fit_on_algorithm_fit_selected_on_calibration",
            },
            f"seed {seed} common-split component bundle drift",
        )
        require(
            row.get("matched_feature_distribution_attested") is True,
            f"seed {seed} V5/PEPD feature compatibility is not attested",
        )
        inventory = row.get("inventory")
        require(isinstance(inventory, Mapping), f"seed {seed} inventory absent")
        fit = inventory.get("algorithm_fit")
        cal = inventory.get("calibration")
        require(isinstance(fit, Mapping) and isinstance(cal, Mapping), f"seed {seed} fit/cal inventory absent")
        _validate_partition_binding(fit, protocol, "algorithm_fit")
        _validate_partition_binding(cal, protocol, "calibration")
        independent = inventory.get("independent_validation")
        require(isinstance(independent, Mapping), f"seed {seed} validation audit absent")
        for key in ("images_read", "annotations_read", "gradient_updates", "selection_queries"):
            require(independent.get(key) == 0, f"seed {seed} validation {key} is nonzero")
        excluded = inventory.get("development_excluded")
        require(isinstance(excluded, Mapping), f"seed {seed} development audit absent")
        for key in ("images_read", "annotations_read", "gradient_updates", "selection_queries"):
            require(excluded.get(key) == 0, f"seed {seed} development {key} is nonzero")
        require(fit.get("gradient_updates", 0) > 0, f"seed {seed} has no fit updates")
        require(cal.get("gradient_updates") == 0, f"seed {seed} calibration updated weights")
        require(cal.get("selection_queries", 0) > 0, f"seed {seed} lacks calibration selection")
        provenance = row.get("initialization")
        require(isinstance(provenance, list) and provenance, f"seed {seed} initialization absent")
        for component in provenance:
            require(isinstance(component, Mapping), f"seed {seed} bad initialization record")
            require(
                component.get("provenance") in ALLOWED_PRETRAINING_PROVENANCE,
                f"seed {seed} has ineligible pretraining provenance",
            )
            if component.get("provenance") == "deterministic_random_init":
                require(component.get("artifact") is None, f"seed {seed} random init binds weights")
                require(component.get("random_seed") == seed, f"seed {seed} random init seed drift")
            else:
                initialization_path = _verify_artifact(
                    component.get("artifact"), label=f"seed {seed} initialization"
                )
                forbidden_hashes = {
                    str(
                        protocol["method"]["legacy_architecture_bindings"][
                            "pepd_seed20260720"
                        ]["checkpoint_sha256"]
                    ),
                    str(
                        protocol["method"]["legacy_architecture_bindings"][
                            "v5_seed20261206"
                        ]["checkpoint_sha256"]
                    ),
                }
                require(
                    sha256_file(initialization_path) not in forbidden_hashes,
                    f"seed {seed} initialized from a formally ineligible legacy checkpoint",
                )
        seed_finished.append(_timestamp(row.get("completed_at"), label=f"seed {seed} completed_at"))

    selection_path = _verify_artifact(run.get("selection_artifact"), label="calibration selection artifact")
    ensemble_path = _verify_artifact(run.get("ensemble_binding"), label="common-split ensemble binding")
    chronology = run.get("chronology")
    require(isinstance(chronology, Mapping), "training chronology absent")
    selection_frozen = _timestamp(chronology.get("selection_frozen_at"), label="selection_frozen_at")
    require(max(seed_finished) <= selection_frozen, "selection predates completed seed training")
    require(chronology.get("independent_validation_first_opened_at") is None, "validation was opened before validate-only")
    audit = run.get("audit")
    require(isinstance(audit, Mapping), "training audit absent")
    required_zero = (
        "independent_validation_images_read",
        "independent_validation_annotations_read",
        "independent_validation_selection_queries",
        "development_excluded_images_read",
        "field_images_read",
        "test_split_images_read",
        "sealed_images_read",
        "confirmatory_images_read",
    )
    for key in required_zero:
        require(audit.get(key) == 0, f"training audit violation: {key}")
    require(audit.get("gradient_partition") == "algorithm_fit", "gradient partition drift")
    require(audit.get("selection_partition") == "calibration", "selection partition drift")
    require(audit.get("independent_validation_reveal_count") == 0, "validation already revealed")
    require(
        audit.get("matched_v5_head_trained_with_same_common_split_pepd") is True,
        "matched common-split V5 head proof absent",
    )
    require(audit.get("legacy_pepd_checkpoint_weights_loaded") == 0, "legacy PEPD weights loaded")
    require(audit.get("legacy_v5_checkpoint_weights_loaded") == 0, "legacy V5 weights loaded")
    # The selection/ensemble files are authenticated above.  Their payload is
    # intentionally not interpreted here; the training manifest is the frozen
    # contract and later scoring cannot alter it.
    del selection_path, ensemble_path
    return run_file, run


def validate_prediction_bundle(
    protocol: Mapping[str, Any],
    *,
    training_run_file: Path,
    bundle_path: Path,
) -> dict[str, Any]:
    bundle_file = guard_public_path(bundle_path, label="common-split progress prediction bundle")
    bundle = strict_json(bundle_file)
    require(bundle.get("schema_version") == 1, "prediction bundle schema drift")
    require(bundle.get("protocol") == PREDICTION_PROTOCOL, "prediction bundle protocol drift")
    require(bundle.get("status") == "predictions_sealed_before_values_opened", "predictions are not sealed")
    require(bundle.get("partition") == "independent_validation", "prediction partition drift")
    require(bundle.get("validation_pass_index") == 1, "validation is not the single declared pass")
    parent = bundle.get("parent_training_run")
    require(isinstance(parent, Mapping), "prediction training binding absent")
    require(Path(str(parent.get("path") or "")).resolve(strict=True) == training_run_file, "prediction training path drift")
    require(parent.get("sha256") == sha256_file(training_run_file), "prediction training hash drift")
    prediction_path = _verify_artifact(bundle.get("predictions"), label="common-split progress predictions")
    seal_path = _verify_artifact(bundle.get("seal"), label="common-split progress prediction seal")
    seal = strict_json(seal_path)
    require(seal.get("protocol") == PREDICTION_PROTOCOL, "prediction seal protocol drift")
    require(seal.get("bundle_sha256") == sha256_file(bundle_file), "prediction bundle seal drift")
    require(seal.get("predictions_sha256") == sha256_file(prediction_path), "prediction table seal drift")
    rows = strict_jsonl(prediction_path)
    _, _, roster, roster_audit = load_partition_roster(
        Path(str(protocol["parent_protocol"]["path"])), "independent_validation"
    )
    roster_by_id = {str(row["sample_id"]): row for row in roster}
    observed: set[str] = set()
    groups: set[str] = set()
    for index, row in enumerate(rows):
        require(set(row) == _PREDICTION_KEYS, f"prediction[{index}] schema drift")
        require(row.get("schema_version") == 1, f"prediction[{index}] schema version drift")
        require(row.get("protocol") == PREDICTION_ROW_PROTOCOL, f"prediction[{index}] protocol drift")
        require(row.get("partition") == "independent_validation", f"prediction[{index}] partition drift")
        sample_id = str(row.get("sample_id") or "")
        require(sample_id in roster_by_id, f"prediction outside roster: {sample_id}")
        require(sample_id not in observed, f"duplicate prediction: {sample_id}")
        group_id = str(row.get("group_id") or "")
        require(group_id == str(roster_by_id[sample_id]["group_id"]), f"prediction group drift: {sample_id}")
        _digest(row.get("canonical_roi_sha256"), label=f"prediction {sample_id} ROI")
        require(isinstance(row.get("status"), bool), f"prediction {sample_id} status drift")
        confidence = _finite_fraction(row.get("progress_confidence"), label=f"prediction {sample_id} confidence")
        del confidence
        progress = row.get("prediction_progress")
        if row["status"]:
            _finite_fraction(progress, label=f"prediction {sample_id} progress")
            require(row.get("failure_reason") is None, f"successful prediction {sample_id} has failure")
        else:
            require(progress is None, f"failed prediction {sample_id} has progress")
            require(isinstance(row.get("failure_reason"), str) and row["failure_reason"], f"failed prediction {sample_id} lacks reason")
        seconds = row.get("sample_seconds")
        require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) >= 0.0,
            f"prediction {sample_id} timing drift",
        )
        observed.add(sample_id)
        groups.add(group_id)
    require(observed == roster_audit["sample_ids"], "independent-validation prediction roster incomplete")
    require(groups == roster_audit["group_ids"], "independent-validation prediction group roster incomplete")
    require(bundle.get("samples") == 1080 and bundle.get("groups") == 50, "prediction inventory drift")
    require(bundle.get("sample_ids_sha256") == canonical_sha256(sorted(observed)), "prediction sample hash drift")
    require(bundle.get("group_ids_sha256") == canonical_sha256(sorted(groups)), "prediction group hash drift")
    audit = bundle.get("audit")
    require(isinstance(audit, Mapping), "prediction audit absent")
    for key in (
        "numeric_range_values_supplied",
        "validation_annotations_opened",
        "field_images_read",
        "test_split_images_read",
        "sealed_images_read",
        "confirmatory_images_read",
    ):
        require(audit.get(key) == 0, f"prediction audit violation: {key}")
    require(audit.get("selection_changes_after_inference") == 0, "validation changed selection")
    return {
        "path": str(bundle_file),
        "sha256": sha256_file(bundle_file),
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "samples": len(observed),
        "groups": len(groups),
        "single_validation_pass_authenticated": True,
    }


def validate_only(
    *,
    protocol_path: Path,
    run_manifest_path: Path,
    output_path: Path,
    prediction_bundle_path: Path | None = None,
) -> Path:
    protocol_file, protocol = load_protocol(protocol_path)
    split_audit = audit_public_split(protocol)
    run_file, run = validate_training_run(protocol_file, protocol, run_manifest_path)
    prediction = (
        None
        if prediction_bundle_path is None
        else validate_prediction_bundle(
            protocol,
            training_run_file=run_file,
            bundle_path=prediction_bundle_path,
        )
    )
    result = {
        "schema_version": 1,
        "protocol": VALIDATE_ONLY_PROTOCOL,
        "status": (
            "ready_for_single_1080_progress_inference"
            if prediction is None
            else "ready_for_full_1080_end_to_end_join_and_one_shot_scoring"
        ),
        "frozen_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "training_run": {"path": str(run_file), "sha256": sha256_file(run_file)},
        "split_audit": split_audit,
        "prediction_bundle": prediction,
        "eligibility": {
            "common_split_progress_training_valid": True,
            "independent_validation_progress_predictions_valid": prediction is not None,
            "full_1080_end_to_end_claim_allowed_after_sealed_garc_join": prediction is not None,
            "validation_may_select_or_modify_model": False,
        },
        "audit": {
            "gpu_initialized": False,
            "images_opened_by_validate_only": 0,
            "annotations_opened_by_validate_only": 0,
            "validation_values_opened_by_validate_only": 0,
            "run_reports_zero_prior_validation_reveals": run["audit"]["independent_validation_reveal_count"] == 0,
            "restricted_namespace_images_opened": 0,
        },
    }
    output = guard_public_path(output_path, label="common-split validate-only output", must_exist=False)
    atomic_new_json(output, result)
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--pilot-candidate", type=Path)
    pre.add_argument("--pilot-control", type=Path)
    validate = commands.add_parser("validate-only")
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate.add_argument("--run-manifest", type=Path, required=True)
    validate.add_argument("--prediction-bundle", type=Path)
    validate.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect-protocol")
    inspect.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "preflight":
        output = preflight(
            protocol_path=args.protocol,
            output_path=args.output,
            pilot_candidate=args.pilot_candidate,
            pilot_control=args.pilot_control,
        )
        payload = {"preflight": str(output), "sha256": sha256_file(output)}
    elif args.command == "validate-only":
        output = validate_only(
            protocol_path=args.protocol,
            run_manifest_path=args.run_manifest,
            prediction_bundle_path=args.prediction_bundle,
            output_path=args.output,
        )
        payload = {"validation": str(output), "sha256": sha256_file(output)}
    else:
        protocol_file, protocol = load_protocol(args.protocol)
        payload = {
            "protocol": str(protocol_file),
            "sha256": sha256_file(protocol_file),
            "identity": protocol["protocol"],
            "pilot_gate": protocol["pilot_gate"],
            "resource_estimate_rtx4060_8gb": protocol["resource_estimate_rtx4060_8gb"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PROTOCOL",
    "EXPECTED_PARTITIONS",
    "EXPECTED_SEEDS",
    "PREDICTION_PROTOCOL",
    "PREDICTION_ROW_PROTOCOL",
    "PROTOCOL",
    "audit_public_split",
    "evaluate_pilot_gate",
    "load_protocol",
    "preflight",
    "validate_only",
    "validate_prediction_bundle",
    "validate_training_run",
]
