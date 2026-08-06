"""Audit, run, seal, and score external progress baselines on GARC-412.

The protocol is deliberately split into four phases:

``preflight``
    Reconstruct VDN grouped-validation membership, authenticate every
    checkpoint, and audit the historical caches without opening an image.
``build-handoff``
    Authenticate the three fold-routed GARC bundles and freeze a label-free
    412-row packet containing the exact canonical ROI identity and the same
    automatically predicted numeric range for every progress backend.
``infer``
    Run an unchanged VDN or Original Transformer backend on that exact ROI and
    seal progress-only predictions.  This command is executable but is never
    called by preflight or build-handoff.
``score``
    Authenticate all seals first, then and only then open public SyncG/train
    truth.  Failures receive normalized error 1.0.

No command accepts a field, test, sealed, or confirmatory path.  VDN is
eligible for the strict grouped-OOF table only after every row is routed to its
matching official-200 checkpoint.  The Original Transformer is hard-limited
to a sensitivity table because its training roster is unavailable and its
SyncG-finetuned segmentation front-end trained on 244/412 rows.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import experiments.train_cagh_scalemark_reference_probe_v5 as public_v5
from experiments.automatic_numeric_range import image_sha256
from experiments.automatic_numeric_range_public_protocol import (
    PUBLIC_IMAGE_ROOT,
    assert_range_label_free,
    atomic_new_json,
    atomic_new_jsonl,
    canonical_sha256,
    guard_public_path,
    require,
    resolve_public_image,
    sha256_file,
    strict_json,
    strict_jsonl,
)
from experiments.evaluate_garc_full_auto_public import (
    _load_joint_runs,
    load_calibration,
    load_public_truth_after_authentication,
    range_variant_sha256,
)
from experiments.garc_full_auto_public import (
    EXPECTED_JOINT_OOF,
    EXPECTED_JOINT_OOF_BY_SEED,
    load_joint_oof_cohort,
    load_plan,
    load_prediction_bundle,
)
from experiments.v5_unified_full_auto_progress_providers import (
    TransformerFullAutoProgressProvider,
    VDNFullAutoProgressProvider,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_VALIDATION_FRACTION,
)


PROTOCOL: Final[str] = "garc_external_progress_412_comparison_v1"
PREFLIGHT_PROTOCOL: Final[str] = "garc_external_progress_412_preflight_v1"
HANDOFF_PROTOCOL: Final[str] = "garc_external_progress_412_handoff_v1"
PREDICTION_PROTOCOL: Final[str] = "garc_external_progress_412_predictions_v1"
SCORE_PROTOCOL: Final[str] = "garc_external_progress_412_score_v1"
DEFAULT_PROTOCOL: Final[Path] = Path(__file__).with_name(
    "garc_external_progress_412_protocol.json"
)
DEFAULT_PREFLIGHT: Final[Path] = Path(
    r"C:\pointer_read\garc_external_progress_412_v1\preflight.json"
)
SOURCE: Final[Path] = Path(__file__).resolve()
VDN_METHOD: Final[str] = "vdn_official200"
TRANSFORMER_METHOD: Final[str] = "original_transformer"
METHODS: Final[tuple[str, str]] = (VDN_METHOD, TRANSFORMER_METHOD)
EXPECTED_SEEDS: Final[tuple[int, ...]] = tuple(sorted(EXPECTED_JOINT_OOF_BY_SEED))
EXPECTED_SEGMENTATION_OVERLAP: Final[dict[str, int]] = {
    "validation_samples": 168,
    "validation_groups": 8,
    "training_samples": 244,
    "training_groups": 11,
}
HANDOFF_ROWS_NAME: Final[str] = "handoff.label_free.jsonl"
PREDICTION_ROWS_NAME: Final[str] = "predictions.label_free.jsonl"

_HANDOFF_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "group_id",
        "image_relpath",
        "dial_bbox",
        "canonical_roi_sha256",
        "oof_seed",
        "vdn_checkpoint_sha256",
        "range_status",
        "predicted_scale_start",
        "predicted_scale_end",
        "range_confidence",
        "range_accepted",
        "garc_full_status",
        "garc_prediction_progress",
        "garc_predicted_reading",
        "garc_failure_reason",
        "garc_progress_checkpoint_sha256",
        "garc_geometry_head_checkpoint_sha256",
        "garc_geometry_backbone_checkpoint_sha256",
        "garc_all_components_group_unseen",
    }
)
_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "method",
        "sample_id",
        "group_id",
        "canonical_roi_sha256",
        "oof_seed",
        "status",
        "prediction_progress",
        "failure_reason",
        "checkpoint_sha256",
        "provider_identity_sha256",
        "provider_record_sha256",
        "strict_oof_eligible",
        "sample_seconds",
    }
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _path_from_protocol(value: Mapping[str, Any], *, label: str) -> Path:
    raw = Path(str(value.get("path") or ""))
    candidate = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    path = guard_public_path(candidate, label=label)
    expected = str(value.get("sha256") or "").casefold()
    require(len(expected) == 64, f"{label} has no SHA-256 binding")
    require(sha256_file(path) == expected, f"{label} hash drift")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[Path, dict[str, Any]]:
    protocol_path = guard_public_path(path, label="external comparison protocol")
    value = strict_json(protocol_path)
    require(value.get("protocol") == PROTOCOL, "external protocol identity drift")
    require(value.get("status") == "frozen_before_external_inference", "protocol is not frozen")
    require(value.get("cohort", {}).get("samples") == EXPECTED_JOINT_OOF[0], "cohort sample drift")
    require(value.get("cohort", {}).get("physical_groups") == EXPECTED_JOINT_OOF[1], "cohort group drift")
    require(
        {int(key): int(count) for key, count in value["cohort"]["samples_by_seed"].items()}
        == EXPECTED_JOINT_OOF_BY_SEED,
        "cohort seed inventory drift",
    )
    require(value.get("input_contract", {}).get("manual_or_ground_truth_numeric_range_forbidden") is True, "manual range is not forbidden")
    require(float(value.get("input_contract", {}).get("failure_penalty_nmae", -1.0)) == 1.0, "failure penalty drift")
    for method in METHODS:
        require(value.get(method, {}).get("architecture_or_weight_modification_allowed") is False, f"{method} mutation is not forbidden")
    return protocol_path, value


def _cohort(protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_path = _path_from_protocol(protocol["cohort"]["summary"], label="joint cohort summary")
    summary, roster, mapping = load_joint_oof_cohort(summary_path)
    require(sha256_file(Path(summary["artifacts"]["cohort"]["path"])) == protocol["cohort"]["label_free_roster"]["sha256"], "cohort roster binding drift")
    require(sha256_file(Path(summary["artifacts"]["mapping"]["path"])) == protocol["cohort"]["progress_mapping"]["sha256"], "cohort mapping binding drift")
    return roster, mapping, summary


def _checkpoint_signature(path: Path) -> Mapping[str, Any]:
    # Lazy import/load keeps ordinary unit tests and protocol inspection cheap.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), f"checkpoint is not a mapping: {path}")
    signature = checkpoint.get("signature")
    require(isinstance(signature, Mapping), f"checkpoint lacks signature: {path}")
    return signature


def _audit_vdn(
    protocol: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
    *,
    inspect_checkpoint_payloads: bool,
) -> dict[str, Any]:
    section = protocol[VDN_METHOD]
    preflight_path = _path_from_protocol(section["training_preflight"], label="VDN training preflight")
    cohort_path = _path_from_protocol(section["three_seed_cohort"], label="VDN three-seed cohort")
    historical_predictions_path = _path_from_protocol(section["historical_oof"]["predictions"], label="VDN historical OOF predictions")
    historical_summary_path = _path_from_protocol(section["historical_oof"]["summary"], label="VDN historical OOF summary")
    preflight = strict_json(preflight_path)
    three_seed = strict_json(cohort_path)
    historical_summary = strict_json(historical_summary_path)
    require(preflight.get("status") == "passed", "VDN preflight did not pass")
    require(three_seed.get("status") == "passed" and three_seed.get("verified") is True, "VDN cohort is unverified")
    require(historical_summary.get("status") == "complete", "VDN historical OOF is incomplete")
    require(historical_summary.get("predictions", {}).get("sha256") == sha256_file(historical_predictions_path), "VDN historical prediction seal drift")

    manifest_binding = preflight.get("manifest") or {}
    manifest_path = guard_public_path(Path(str(manifest_binding.get("path") or "")), label="VDN SyncG/train manifest")
    require(sha256_file(manifest_path) == manifest_binding.get("sha256"), "VDN manifest hash drift")
    samples, _ = load_syncg_manifest(manifest_path, expected_split="train")
    sample_by_id = {str(sample.sample_id): sample for sample in samples}
    require(len(sample_by_id) == len(samples) == 16_000, "VDN source manifest inventory drift")
    mapping_by_id = {str(row["sample_id"]): row for row in mapping}
    require(set(mapping_by_id) == {str(row["sample_id"]) for row in roster}, "VDN mapping/roster drift")

    preflight_runs = {int(row["seed"]): row for row in preflight.get("runs") or []}
    cohort_runs = {int(row["seed"]): row for row in three_seed.get("runs") or []}
    validation_by_seed: dict[int, set[str]] = {}
    validation_groups_by_seed: dict[int, set[str]] = {}
    train_groups_by_seed: dict[int, set[str]] = {}
    routes: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        train, validation = grouped_train_val_split(
            samples,
            validation_fraction=OFFICIAL200_VALIDATION_FRACTION,
            seed=seed,
        )
        validation_ids = {str(sample.sample_id) for sample in validation}
        validation_groups = {str(sample.group_id) for sample in validation}
        train_groups = {str(sample.group_id) for sample in train}
        require(not (validation_groups & train_groups), f"VDN seed {seed} group leakage")
        validation_by_seed[seed] = validation_ids
        validation_groups_by_seed[seed] = validation_groups
        train_groups_by_seed[seed] = train_groups
        route = section["routes"][str(seed)]
        checkpoint_path = _path_from_protocol(route["checkpoint"], label=f"VDN seed {seed} checkpoint")
        verification_path = _path_from_protocol(route["verification"], label=f"VDN seed {seed} verification")
        summary_path = _path_from_protocol(route["summary"], label=f"VDN seed {seed} summary")
        verification = strict_json(verification_path)
        run_summary = strict_json(summary_path)
        require(verification.get("verified") is True and int(verification.get("seed", -1)) == seed, f"VDN seed {seed} verification failed")
        require(verification.get("training_artifacts_verified") is True, f"VDN seed {seed} artifacts unverified")
        require(verification.get("best_checkpoint_sha256") == sha256_file(checkpoint_path), f"VDN seed {seed} verification/checkpoint drift")
        require(cohort_runs[seed].get("best_checkpoint_sha256") == sha256_file(checkpoint_path), f"VDN seed {seed} cohort/checkpoint drift")
        require(preflight_runs[seed].get("validation_sample_ids_sha256") == sample_ids_hash(validation), f"VDN seed {seed} recomputed validation split drift")
        require(route.get("validation_sample_ids_sha256") == sample_ids_hash(validation), f"VDN seed {seed} protocol split drift")
        summary_signature = run_summary.get("signature") or {}
        require(summary_signature.get("protocol") == OFFICIAL200_PROTOCOL, f"VDN seed {seed} summary protocol drift")
        require(int(summary_signature.get("seed", -1)) == seed, f"VDN seed {seed} summary seed drift")
        require(summary_signature.get("validation_sample_ids_sha256") == sample_ids_hash(validation), f"VDN seed {seed} summary split drift")
        if inspect_checkpoint_payloads:
            signature = _checkpoint_signature(checkpoint_path)
            require(signature.get("protocol") == OFFICIAL200_PROTOCOL, f"VDN seed {seed} checkpoint protocol drift")
            require(int(signature.get("seed", -1)) == seed, f"VDN seed {seed} checkpoint seed drift")
            require(signature.get("manifest_sha256") == sha256_file(manifest_path), f"VDN seed {seed} checkpoint manifest drift")
            require(signature.get("validation_sample_ids_sha256") == sample_ids_hash(validation), f"VDN seed {seed} checkpoint split drift")
        routes[str(seed)] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "verification_path": str(verification_path),
            "verification_sha256": sha256_file(verification_path),
            "summary_sha256": sha256_file(summary_path),
            "validation_samples": len(validation_ids),
            "validation_groups": len(validation_groups),
            "validation_sample_ids_sha256": sample_ids_hash(validation),
            "checkpoint_payload_signature_inspected": inspect_checkpoint_payloads,
        }

    per_seed: Counter[int] = Counter()
    groups_per_seed: dict[int, set[str]] = {seed: set() for seed in EXPECTED_SEEDS}
    for sample_id, route in mapping_by_id.items():
        seed = int(route["pepd_seed"])
        sample = sample_by_id.get(sample_id)
        require(sample is not None, f"VDN route names unknown sample: {sample_id}")
        require(str(sample.group_id) == str(route["group_id"]), f"VDN route group drift: {sample_id}")
        require(sample_id in validation_by_seed[seed], f"VDN selected checkpoint saw sample: {sample_id}")
        require(str(sample.group_id) in validation_groups_by_seed[seed], f"VDN selected checkpoint saw group: {sample_id}")
        require(str(sample.group_id) not in train_groups_by_seed[seed], f"VDN group leaks into training: {sample_id}")
        per_seed[seed] += 1
        groups_per_seed[seed].add(str(sample.group_id))
    require(dict(per_seed) == EXPECTED_JOINT_OOF_BY_SEED, "VDN routed sample inventory drift")

    historical_rows = strict_jsonl(historical_predictions_path)
    historical_by_id = {str(row["sample_id"]): row for row in historical_rows}
    require(len(historical_by_id) == len(historical_rows) == 4_380, "VDN historical OOF inventory drift")
    for sample_id, route in mapping_by_id.items():
        row = historical_by_id.get(sample_id)
        require(row is not None, f"VDN historical cache misses {sample_id}")
        require(str(row.get("group_id")) == str(route["group_id"]), f"VDN historical group drift: {sample_id}")
        require(int(row.get("held_out_seed", -1)) == int(route["pepd_seed"]), f"VDN historical seed drift: {sample_id}")

    reference_path = _path_from_protocol(section["automatic_reference_detector"], label="VDN automatic reference detector")
    source_path = guard_public_path(_PROJECT_ROOT / str(section["source"]["path"]), label="VDN source", expect_file=False)
    require(source_path.is_dir(), "VDN source is not a directory")
    require(verify_vdn_source(source_path) == section["source"]["commit"], "VDN source commit drift")
    return {
        "strict_412_progress_oof_eligible": True,
        "routed_samples": sum(per_seed.values()),
        "routed_groups": len({str(row["group_id"]) for row in mapping}),
        "samples_by_seed": {str(seed): per_seed[seed] for seed in EXPECTED_SEEDS},
        "groups_by_seed": {str(seed): len(groups_per_seed[seed]) for seed in EXPECTED_SEEDS},
        "routes": routes,
        "reference_detector": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
        "source": {"path": str(source_path), "commit": section["source"]["commit"]},
        "historical_cache": {
            "rows": len(historical_rows),
            "joint_rows_verified": len(mapping_by_id),
            "sha256": sha256_file(historical_predictions_path),
            "formal_same_input_reuse_allowed": False,
            "reason": section["historical_oof"]["reason"],
        },
        "architecture_or_weights_modified": False,
    }


def _audit_transformer(
    protocol: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    section = protocol[TRANSFORMER_METHOD]
    transformer_path = _path_from_protocol(section["transformer_checkpoint"], label="Original Transformer checkpoint")
    segmentation = section["pointer_segmentation"]
    segmentation_path = _path_from_protocol(segmentation, label="Transformer pointer segmentation")
    split_path = _path_from_protocol(segmentation["split"], label="Transformer segmentation split")
    segmentation_summary_path = _path_from_protocol(segmentation["summary"], label="Transformer segmentation summary")
    original_oof_path = _path_from_protocol(section["historical_oof"]["predictions"], label="Transformer historical OOF")
    original_summary_path = _path_from_protocol(section["historical_oof"]["summary"], label="Transformer historical OOF summary")
    original_metadata_path = _path_from_protocol(section["historical_oof"]["metadata"], label="Transformer historical OOF metadata")
    raw_metadata_path = _path_from_protocol(section["historical_oof"]["raw_prediction_metadata"], label="Transformer raw prediction metadata")
    split = strict_json(split_path)
    segmentation_summary = strict_json(segmentation_summary_path)
    original_summary = strict_json(original_summary_path)
    original_metadata = strict_json(original_metadata_path)
    raw_metadata = strict_json(raw_metadata_path)
    require(segmentation_summary.get("best_checkpoint_sha256") == sha256_file(segmentation_path), "Transformer segmentation summary drift")
    require(original_summary.get("output_sha256") == sha256_file(original_oof_path), "Transformer historical OOF summary drift")
    require(original_summary.get("status") == "complete", "Transformer historical OOF incomplete")
    require(
        (raw_metadata.get("signature") or {})
        .get("weights_sha256", {})
        .get("meter_transformer")
        == sha256_file(transformer_path),
        "Transformer historical cache weight drift",
    )
    require(
        (original_metadata.get("signature") or {}).get("raw_predictions_sha256")
        == original_summary.get("signature", {}).get("raw_predictions_sha256"),
        "Transformer OOF/raw cache lineage drift",
    )

    roster_by_id = {str(row["sample_id"]): row for row in roster}
    mapping_by_id = {str(row["sample_id"]): row for row in mapping}
    train_ids = {str(value) for value in split.get("train_sample_ids") or []}
    validation_ids = {str(value) for value in split.get("validation_sample_ids") or []}
    require(not (train_ids & validation_ids), "segmentation split sample leakage")
    joint_ids = set(roster_by_id)
    require(joint_ids <= train_ids | validation_ids, "segmentation split does not cover GARC-412")
    joint_train = joint_ids & train_ids
    joint_validation = joint_ids & validation_ids
    train_groups = {str(roster_by_id[sample_id]["group_id"]) for sample_id in joint_train}
    validation_groups = {str(roster_by_id[sample_id]["group_id"]) for sample_id in joint_validation}
    require(not (train_groups & validation_groups), "segmentation group split drift inside GARC-412")
    observed_overlap = {
        "validation_samples": len(joint_validation),
        "validation_groups": len(validation_groups),
        "training_samples": len(joint_train),
        "training_groups": len(train_groups),
    }
    require(observed_overlap == EXPECTED_SEGMENTATION_OVERLAP, "Transformer segmentation overlap audit drift")
    for sample_id, route in mapping_by_id.items():
        seed = int(route["pepd_seed"])
        if seed == 20260720:
            require(sample_id in joint_validation, f"seed-20 row not held out by segmentation: {sample_id}")
        else:
            require(sample_id in joint_train, f"seed-{seed} row unexpectedly held out by segmentation: {sample_id}")

    original_rows = strict_jsonl(original_oof_path)
    original_by_id = {str(row["sample_id"]): row for row in original_rows}
    require(len(original_by_id) == len(original_rows) == 4_380, "Transformer historical OOF inventory drift")
    successful = 0
    for sample_id, route in mapping_by_id.items():
        row = original_by_id.get(sample_id)
        require(row is not None, f"Transformer historical cache misses {sample_id}")
        require(str(row.get("group_id")) == str(route["group_id"]), f"Transformer historical group drift: {sample_id}")
        require(int(row.get("held_out_seed", -1)) == int(route["pepd_seed"]), f"Transformer carrier-row seed drift: {sample_id}")
        transformer = ((row.get("raw") or {}).get("methods") or {}).get("transformer") or {}
        successful += int(transformer.get("status") is True and _finite(transformer.get("prediction")) is not None)

    require(section.get("strict_412_table_eligible") is False, "Transformer protocol falsely permits strict table")
    return {
        "strict_412_progress_oof_eligible": False,
        "sensitivity_table_eligible_after_same_input_rerun": True,
        "transformer_checkpoint": {"path": str(transformer_path), "sha256": sha256_file(transformer_path)},
        "authenticated_training_roster_present": False,
        "pointer_segmentation": {
            "path": str(segmentation_path),
            "sha256": sha256_file(segmentation_path),
            "split_path": str(split_path),
            "split_sha256": sha256_file(split_path),
            "joint_overlap": observed_overlap,
        },
        "historical_cache": {
            "rows": len(original_rows),
            "joint_rows_verified": len(mapping_by_id),
            "joint_rows_with_finite_transformer_prediction": successful,
            "sha256": sha256_file(original_oof_path),
            "formal_same_input_reuse_allowed": False,
        },
        "strict_rejection_reasons": list(section["strict_rejection_reasons"]),
        "architecture_or_weights_modified": False,
    }


def build_preflight_report(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    inspect_checkpoint_payloads: bool = True,
) -> dict[str, Any]:
    protocol_file, protocol = load_protocol(protocol_path)
    roster, mapping, cohort_summary = _cohort(protocol)
    vdn = _audit_vdn(
        protocol,
        roster,
        mapping,
        inspect_checkpoint_payloads=inspect_checkpoint_payloads,
    )
    transformer = _audit_transformer(protocol, roster, mapping)
    result = {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "passed_with_transformer_sensitivity_only",
        "frozen_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "comparator_source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "cohort": {
            "summary_sha256": sha256_file(Path(protocol["cohort"]["summary"]["path"])),
            "roster_sha256": protocol["cohort"]["label_free_roster"]["sha256"],
            "mapping_sha256": protocol["cohort"]["progress_mapping"]["sha256"],
            "samples": len(roster),
            "physical_groups": len({str(row["group_id"]) for row in roster}),
            "samples_by_seed": cohort_summary["joint_oof"]["samples_by_pepd_seed"],
        },
        "methods": {VDN_METHOD: vdn, TRANSFORMER_METHOD: transformer},
        "formal_table": {
            "eligible_methods": ["garc", VDN_METHOD],
            "rejected_methods": [TRANSFORMER_METHOD],
            "transformer_may_be_reported_only_as": "fixed-checkpoint same-input sensitivity",
        },
        "audit": {
            "public_manifest_parsed_for_grouped_split_reconstruction": True,
            "public_images_opened": 0,
            "public_truth_scoring_performed": False,
            "external_inference_started": False,
            "gpu_operations_executed": False,
            "restricted_namespace_images_opened": 0,
            "field_samples_used": 0,
            "test_samples_used": 0,
            "sealed_samples_used": 0,
            "confirmatory_samples_used": 0,
        },
    }
    assert_range_label_free(result, location="external_preflight")
    return result


def write_preflight(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_path: Path = DEFAULT_PREFLIGHT,
    inspect_checkpoint_payloads: bool = True,
) -> Path:
    output = guard_public_path(output_path, label="external preflight output", must_exist=False)
    report = build_preflight_report(
        protocol_path=protocol_path,
        inspect_checkpoint_payloads=inspect_checkpoint_payloads,
    )
    atomic_new_json(output, report)
    return output


def _load_preflight(
    path: Path,
    *,
    protocol_file: Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    preflight_path = guard_public_path(path, label="external preflight")
    value = strict_json(preflight_path)
    require(value.get("protocol") == PREFLIGHT_PROTOCOL, "external preflight identity drift")
    require(value.get("status") == "passed_with_transformer_sensitivity_only", "external preflight did not pass")
    require(value.get("frozen_protocol", {}).get("sha256") == sha256_file(protocol_file), "external preflight protocol drift")
    require(
        value.get("comparator_source", {}).get("sha256") == sha256_file(SOURCE),
        "external comparator source changed after preflight",
    )
    require(value.get("cohort", {}).get("roster_sha256") == protocol["cohort"]["label_free_roster"]["sha256"], "external preflight cohort drift")
    require(value.get("methods", {}).get(VDN_METHOD, {}).get("strict_412_progress_oof_eligible") is True, "VDN is not preflight-eligible")
    require(value.get("methods", {}).get(TRANSFORMER_METHOD, {}).get("strict_412_progress_oof_eligible") is False, "Transformer preflight role drift")
    return preflight_path, value


def validate_handoff_row(
    row: Mapping[str, Any],
    *,
    roster_row: Mapping[str, Any],
    route: Mapping[str, Any],
    vdn_checkpoint_sha256: str,
) -> None:
    require(set(row) == _HANDOFF_KEYS, f"handoff row schema drift: {row.get('sample_id')}")
    assert_range_label_free(row, location=f"external_handoff.{row.get('sample_id')}")
    require(row.get("schema_version") == 1 and row.get("protocol") == HANDOFF_PROTOCOL, "handoff identity drift")
    require(row.get("sample_id") == roster_row.get("sample_id"), "handoff sample drift")
    require(row.get("group_id") == roster_row.get("group_id") == route.get("group_id"), "handoff group drift")
    require(row.get("image_relpath") == roster_row.get("image_relpath"), "handoff image drift")
    require(row.get("dial_bbox") == roster_row.get("dial_bbox"), "handoff ROI box drift")
    require(row.get("canonical_roi_sha256") == route.get("canonical_roi_sha256"), "handoff ROI hash drift")
    require(int(row.get("oof_seed", -1)) == int(route.get("pepd_seed", -2)), "handoff route seed drift")
    require(row.get("vdn_checkpoint_sha256") == vdn_checkpoint_sha256, "handoff VDN checkpoint drift")
    require(row.get("garc_progress_checkpoint_sha256") == route.get("pepd_checkpoint_sha256"), "handoff GARC progress checkpoint drift")
    require(row.get("garc_all_components_group_unseen") is True, "handoff includes a non-jointly-unseen GARC row")
    require(isinstance(row.get("range_status"), bool) and isinstance(row.get("range_accepted"), bool), "handoff range flags are not Boolean")
    confidence = _finite(row.get("range_confidence"))
    require(confidence is not None and 0.0 <= confidence <= 1.0, "handoff range confidence invalid")
    for key in (
        "predicted_scale_start",
        "predicted_scale_end",
        "garc_prediction_progress",
        "garc_predicted_reading",
    ):
        require(row.get(key) is None or _finite(row.get(key)) is not None, f"handoff has non-finite {key}")
    if row["range_status"]:
        require(row["predicted_scale_start"] is not None and row["predicted_scale_end"] is not None, "handoff successful range lacks endpoints")
        require(row["predicted_scale_start"] != row["predicted_scale_end"], "handoff range span is zero")
    if row["range_accepted"]:
        require(row["range_status"], "handoff accepts a failed range")
    if row["garc_full_status"]:
        require(row["range_status"] and row["garc_prediction_progress"] is not None and row["garc_predicted_reading"] is not None, "handoff GARC success lacks factors")


def build_handoff(
    *,
    protocol_path: Path,
    preflight_path: Path,
    primary_plan: Path,
    primary_prediction_root: Path,
    calibration_path: Path,
    extra_joint_runs: Sequence[tuple[Path, Path]],
    output_root: Path,
) -> Path:
    protocol_file, protocol = load_protocol(protocol_path)
    checked_preflight_path, preflight = _load_preflight(
        preflight_path,
        protocol_file=protocol_file,
        protocol=protocol,
    )
    roster, mapping, _ = _cohort(protocol)
    roster_by_id = {str(row["sample_id"]): row for row in roster}
    mapping_by_id = {str(row["sample_id"]): row for row in mapping}

    primary_summary, _, _, primary_bundle = load_prediction_bundle(
        plan_path=primary_plan,
        prediction_root=primary_prediction_root,
        expected_partition="independent_validation",
        allow_smoke=False,
    )
    plan_file, plan = load_plan(primary_plan)
    variant = range_variant_sha256(plan)
    calibration, threshold = load_calibration(
        calibration_path,
        range_binding_sha256=primary_bundle.range_binding_sha256,
        expected_range_variant_sha256=variant,
        allow_smoke=False,
    )
    require(primary_summary.get("mode") == "formal" and calibration.get("mode") == "formal", "external handoff requires formal GARC artifacts")
    joint_rows, joint_audit = _load_joint_runs(
        primary_plan=plan_file,
        primary_root=primary_prediction_root,
        extra_joint_runs=extra_joint_runs,
        expected_range_binding=primary_bundle.range_binding_sha256,
        expected_range_variant=variant,
        allow_smoke=False,
    )
    require(joint_audit.get("complete") is True, "GARC joint OOF bundle is incomplete")
    require(len(joint_rows) == EXPECTED_JOINT_OOF[0], "GARC joint OOF row drift")
    require({str(row["sample_id"]) for row in joint_rows} == set(roster_by_id), "GARC joint OOF roster drift")

    vdn_routes = preflight["methods"][VDN_METHOD]["routes"]
    handoff_rows: list[dict[str, Any]] = []
    for source in sorted(joint_rows, key=lambda row: str(row["sample_id"])):
        sample_id = str(source["sample_id"])
        roster_row = roster_by_id[sample_id]
        route = mapping_by_id[sample_id]
        seed = int(route["pepd_seed"])
        require(
            source.get("joint_oof_eligible") is True
            and source.get("progress_group_unseen") is True
            and source.get("geometry_group_unseen") is True,
            f"GARC row is not all-component unseen: {sample_id}",
        )
        require(source.get("canonical_roi_sha256") == route.get("canonical_roi_sha256"), f"GARC ROI mapping drift: {sample_id}")
        require(source.get("progress_checkpoint_sha256") == route.get("pepd_checkpoint_sha256"), f"GARC progress route drift: {sample_id}")
        require(int(source.get("geometry_oof_seed", -1)) == seed, f"GARC geometry seed drift: {sample_id}")
        range_accepted = bool(
            source["range_status"]
            and float(source["range_confidence"]) >= threshold
        )
        row = {
            "schema_version": 1,
            "protocol": HANDOFF_PROTOCOL,
            "sample_id": sample_id,
            "group_id": str(source["group_id"]),
            "image_relpath": str(roster_row["image_relpath"]),
            "dial_bbox": list(roster_row["dial_bbox"]),
            "canonical_roi_sha256": str(source["canonical_roi_sha256"]),
            "oof_seed": seed,
            "vdn_checkpoint_sha256": str(vdn_routes[str(seed)]["checkpoint_sha256"]),
            "range_status": bool(source["range_status"]),
            "predicted_scale_start": source["predicted_scale_start"],
            "predicted_scale_end": source["predicted_scale_end"],
            "range_confidence": float(source["range_confidence"]),
            "range_accepted": range_accepted,
            "garc_full_status": bool(source["full_status"]),
            "garc_prediction_progress": source["prediction_progress"],
            "garc_predicted_reading": source["predicted_reading"],
            "garc_failure_reason": source["failure_reason"],
            "garc_progress_checkpoint_sha256": source["progress_checkpoint_sha256"],
            "garc_geometry_head_checkpoint_sha256": source["geometry_head_checkpoint_sha256"],
            "garc_geometry_backbone_checkpoint_sha256": source["geometry_backbone_checkpoint_sha256"],
            "garc_all_components_group_unseen": True,
        }
        validate_handoff_row(
            row,
            roster_row=roster_row,
            route=route,
            vdn_checkpoint_sha256=str(vdn_routes[str(seed)]["checkpoint_sha256"]),
        )
        handoff_rows.append(row)

    output = guard_public_path(output_root, label="external handoff output", must_exist=False, expect_file=False)
    require(not output.exists(), f"refusing to overwrite external handoff: {output}")
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / HANDOFF_ROWS_NAME
    atomic_new_jsonl(rows_path, handoff_rows)
    source_runs = [
        {"plan": str(plan_file), "plan_sha256": sha256_file(plan_file), "prediction_root": str(Path(primary_prediction_root).resolve(strict=True))},
        *[
            {"plan": str(Path(extra_plan).resolve(strict=True)), "plan_sha256": sha256_file(Path(extra_plan)), "prediction_root": str(Path(extra_root).resolve(strict=True))}
            for extra_plan, extra_root in extra_joint_runs
        ],
    ]
    summary = {
        "schema_version": 1,
        "protocol": HANDOFF_PROTOCOL,
        "status": "label_free_handoff_sealed",
        "frozen_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "comparator_source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "preflight": {"path": str(checked_preflight_path), "sha256": sha256_file(checked_preflight_path)},
        "cohort": {
            "samples": len(handoff_rows),
            "physical_groups": len({row["group_id"] for row in handoff_rows}),
            "samples_by_seed": dict(Counter(str(row["oof_seed"]) for row in handoff_rows)),
            "roster_sha256": protocol["cohort"]["label_free_roster"]["sha256"],
            "mapping_sha256": protocol["cohort"]["progress_mapping"]["sha256"],
        },
        "garc": {
            "range_binding_sha256": primary_bundle.range_binding_sha256,
            "range_variant_sha256": variant,
            "minimum_range_confidence": threshold,
            "source_runs": source_runs,
            "all_412_progress_geometry_head_and_backbone_group_unseen": True,
        },
        "method_roles": {
            VDN_METHOD: "strict_grouped_oof_external_progress_comparator",
            TRANSFORMER_METHOD: "fixed_checkpoint_non_oof_sensitivity_only",
        },
        "artifacts": {
            "rows": {"path": HANDOFF_ROWS_NAME, "sha256": sha256_file(rows_path)},
        },
        "audit": {
            "predicted_ranges_reused_without_reinference": True,
            "same_canonical_roi_required": True,
            "public_truth_opened": False,
            "external_inference_started": False,
            "restricted_namespace_images_opened": 0,
        },
    }
    assert_range_label_free(summary, location="external_handoff_summary")
    summary_path = output / "summary.json"
    atomic_new_json(summary_path, summary)
    seal = {
        "schema_version": 1,
        "protocol": HANDOFF_PROTOCOL,
        "status": "sealed",
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
        "bundle_sha256": canonical_sha256(
            {
                "protocol_sha256": sha256_file(protocol_file),
                "preflight_sha256": sha256_file(checked_preflight_path),
                "summary_sha256": sha256_file(summary_path),
                "rows_sha256": sha256_file(rows_path),
            }
        ),
    }
    atomic_new_json(output / "seal.json", seal)
    return output


def load_handoff(
    root: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol_file, protocol = load_protocol(protocol_path)
    checked_preflight_path, preflight = _load_preflight(
        preflight_path,
        protocol_file=protocol_file,
        protocol=protocol,
    )
    bundle_root = guard_public_path(root, label="external handoff bundle", expect_file=False)
    require(bundle_root.is_dir(), "external handoff bundle is not a directory")
    summary_path = guard_public_path(bundle_root / "summary.json", label="external handoff summary")
    seal_path = guard_public_path(bundle_root / "seal.json", label="external handoff seal")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == HANDOFF_PROTOCOL and summary.get("status") == "label_free_handoff_sealed", "external handoff summary identity drift")
    require(seal.get("protocol") == HANDOFF_PROTOCOL and seal.get("status") == "sealed", "external handoff seal identity drift")
    require(summary.get("frozen_protocol", {}).get("sha256") == sha256_file(protocol_file), "external handoff protocol drift")
    require(summary.get("comparator_source", {}).get("sha256") == sha256_file(SOURCE), "external handoff comparator source drift")
    require(summary.get("preflight", {}).get("sha256") == sha256_file(checked_preflight_path), "external handoff preflight drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "external handoff summary seal drift")
    rows_path = guard_public_path(bundle_root / str(summary["artifacts"]["rows"]["path"]), label="external handoff rows")
    require(summary["artifacts"]["rows"]["sha256"] == seal.get("rows_sha256") == sha256_file(rows_path), "external handoff rows seal drift")
    expected_bundle = canonical_sha256(
        {
            "protocol_sha256": sha256_file(protocol_file),
            "preflight_sha256": sha256_file(checked_preflight_path),
            "summary_sha256": sha256_file(summary_path),
            "rows_sha256": sha256_file(rows_path),
        }
    )
    require(seal.get("bundle_sha256") == expected_bundle, "external handoff bundle seal drift")
    rows = strict_jsonl(rows_path)
    roster, mapping, _ = _cohort(protocol)
    roster_by_id = {str(row["sample_id"]): row for row in roster}
    mapping_by_id = {str(row["sample_id"]): row for row in mapping}
    require(len(rows) == EXPECTED_JOINT_OOF[0], "external handoff inventory drift")
    require(len({str(row["sample_id"]) for row in rows}) == len(rows), "external handoff duplicate samples")
    vdn_routes = preflight["methods"][VDN_METHOD]["routes"]
    for row in rows:
        sample_id = str(row["sample_id"])
        require(sample_id in roster_by_id and sample_id in mapping_by_id, f"handoff sample outside frozen cohort: {sample_id}")
        seed = int(mapping_by_id[sample_id]["pepd_seed"])
        validate_handoff_row(
            row,
            roster_row=roster_by_id[sample_id],
            route=mapping_by_id[sample_id],
            vdn_checkpoint_sha256=str(vdn_routes[str(seed)]["checkpoint_sha256"]),
        )
    return summary, rows, preflight


def validate_external_prediction_row(
    row: Mapping[str, Any],
    *,
    handoff_row: Mapping[str, Any],
    transformer_checkpoint_sha256: str,
) -> None:
    require(set(row) == _PREDICTION_KEYS, f"external prediction schema drift: {row.get('sample_id')}")
    assert_range_label_free(row, location=f"external_prediction.{row.get('sample_id')}")
    require(row.get("schema_version") == 1 and row.get("protocol") == PREDICTION_PROTOCOL, "external prediction identity drift")
    method = str(row.get("method") or "")
    require(method in METHODS, "unknown external prediction method")
    require(row.get("sample_id") == handoff_row.get("sample_id"), "external prediction sample drift")
    require(row.get("group_id") == handoff_row.get("group_id"), "external prediction group drift")
    require(row.get("canonical_roi_sha256") == handoff_row.get("canonical_roi_sha256"), "external prediction ROI drift")
    require(int(row.get("oof_seed", -1)) == int(handoff_row.get("oof_seed", -2)), "external prediction route drift")
    expected_checkpoint = (
        str(handoff_row["vdn_checkpoint_sha256"])
        if method == VDN_METHOD
        else transformer_checkpoint_sha256
    )
    require(row.get("checkpoint_sha256") == expected_checkpoint, "external prediction checkpoint drift")
    expected_eligibility = method == VDN_METHOD
    require(row.get("strict_oof_eligible") is expected_eligibility, "external prediction claim role drift")
    require(isinstance(row.get("status"), bool), "external prediction status is not Boolean")
    seconds = _finite(row.get("sample_seconds"))
    require(seconds is not None and seconds >= 0.0, "external prediction duration invalid")
    for key in ("provider_identity_sha256", "provider_record_sha256"):
        value = str(row.get(key) or "")
        require(len(value) == 64 and all(char in "0123456789abcdef" for char in value), f"bad {key}")
    progress = _finite(row.get("prediction_progress"))
    if row["status"]:
        require(progress is not None and 0.0 <= progress <= 1.0, "successful external progress is invalid")
        require(row.get("failure_reason") is None, "successful external prediction has failure reason")
    else:
        require(row.get("prediction_progress") is None, "failed external prediction has progress")
        require(bool(str(row.get("failure_reason") or "")), "failed external prediction lacks reason")


def _normalize_provider_record(
    record: Mapping[str, Any],
) -> tuple[bool, float | None, str | None]:
    status_value = record.get("status")
    status = status_value is True or str(status_value).casefold() in {"ok", "success", "successful"}
    progress = _finite(
        record.get("prediction_progress")
        if record.get("prediction_progress") is not None
        else record.get("progress")
    )
    if status and progress is not None and 0.0 <= progress <= 1.0:
        return True, progress, None
    failure = (
        record.get("failure_code")
        or record.get("failure_reason")
        or record.get("failure")
        or ("progress_out_of_range" if progress is not None else "progress_unavailable")
    )
    return False, None, str(failure)


def _build_vdn_provider(
    *,
    seed: int,
    preflight: Mapping[str, Any],
    device: str,
) -> VDNFullAutoProgressProvider:
    row = preflight["methods"][VDN_METHOD]["routes"][str(seed)]
    reference = preflight["methods"][VDN_METHOD]["reference_detector"]
    source = preflight["methods"][VDN_METHOD]["source"]
    return VDNFullAutoProgressProvider.from_frozen_files(
        checkpoint_path=Path(row["checkpoint_path"]),
        expected_checkpoint_sha256=str(row["checkpoint_sha256"]),
        verification_path=Path(row["verification_path"]),
        vdn_source=Path(source["path"]),
        reference_detector_path=Path(reference["path"]),
        expected_reference_detector_sha256=str(reference["sha256"]),
        device=device,
        amp_enabled=str(device).casefold().startswith("cuda"),
    )


def _build_transformer_provider(
    *,
    protocol: Mapping[str, Any],
    device: str,
) -> TransformerFullAutoProgressProvider:
    section = protocol[TRANSFORMER_METHOD]
    segmentation = _path_from_protocol(section["pointer_segmentation"], label="Transformer pointer segmentation")
    transformer = _path_from_protocol(section["transformer_checkpoint"], label="Original Transformer checkpoint")
    return TransformerFullAutoProgressProvider.from_frozen_files(
        pointer_segmentation_path=segmentation,
        original_transformer_path=transformer,
        device=device,
    )


def _failed_external_row(
    *,
    method: str,
    handoff_row: Mapping[str, Any],
    checkpoint_sha256: str,
    provider_identity_sha256: str,
    reason: str,
    sample_seconds: float,
) -> dict[str, Any]:
    record_identity = canonical_sha256(
        {"method": method, "sample_id": handoff_row["sample_id"], "failure": reason}
    )
    return {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "method": method,
        "sample_id": str(handoff_row["sample_id"]),
        "group_id": str(handoff_row["group_id"]),
        "canonical_roi_sha256": str(handoff_row["canonical_roi_sha256"]),
        "oof_seed": int(handoff_row["oof_seed"]),
        "status": False,
        "prediction_progress": None,
        "failure_reason": reason,
        "checkpoint_sha256": checkpoint_sha256,
        "provider_identity_sha256": provider_identity_sha256,
        "provider_record_sha256": record_identity,
        "strict_oof_eligible": method == VDN_METHOD,
        "sample_seconds": float(sample_seconds),
    }


def _seal_external_predictions(
    *,
    method: str,
    handoff_root: Path,
    protocol_path: Path,
    preflight_path: Path,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    provider_identities: Mapping[str, Any],
    device: str,
) -> Path:
    protocol_file, protocol = load_protocol(protocol_path)
    handoff_summary, handoff_rows, _ = load_handoff(
        handoff_root,
        protocol_path=protocol_file,
        preflight_path=preflight_path,
    )
    handoff_by_id = {str(row["sample_id"]): row for row in handoff_rows}
    transformer_sha = protocol[TRANSFORMER_METHOD]["transformer_checkpoint"]["sha256"]
    require(len(rows) == len(handoff_rows), "external prediction roster incomplete")
    require(len({str(row["sample_id"]) for row in rows}) == len(rows), "external prediction duplicate sample")
    for row in rows:
        sample_id = str(row["sample_id"])
        require(sample_id in handoff_by_id, f"external prediction outside handoff: {sample_id}")
        validate_external_prediction_row(
            row,
            handoff_row=handoff_by_id[sample_id],
            transformer_checkpoint_sha256=transformer_sha,
        )
        require(row["method"] == method, "mixed methods in external prediction bundle")
    output = guard_public_path(output_root, label="external prediction output", must_exist=False, expect_file=False)
    require(not output.exists(), f"refusing to overwrite external predictions: {output}")
    output.mkdir(parents=True, exist_ok=False)
    ordered = [dict(row) for row in sorted(rows, key=lambda row: str(row["sample_id"]))]
    rows_path = output / PREDICTION_ROWS_NAME
    atomic_new_jsonl(rows_path, ordered)
    successful = sum(int(row["status"]) for row in ordered)
    summary = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "status": "predictions_sealed",
        "method": method,
        "method_role": (
            "strict_grouped_oof_external_progress_comparator"
            if method == VDN_METHOD
            else "fixed_checkpoint_non_oof_sensitivity_only"
        ),
        "strict_oof_eligible": method == VDN_METHOD,
        "frozen_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "comparator_source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "preflight": {"path": str(Path(preflight_path).resolve(strict=True)), "sha256": sha256_file(Path(preflight_path))},
        "handoff": {
            "path": str(Path(handoff_root).resolve(strict=True)),
            "summary_sha256": sha256_file(Path(handoff_root) / "summary.json"),
            "bundle_sha256": strict_json(Path(handoff_root) / "seal.json")["bundle_sha256"],
            "range_binding_sha256": handoff_summary["garc"]["range_binding_sha256"],
        },
        "cohort": {
            "samples": len(ordered),
            "physical_groups": len({str(row["group_id"]) for row in ordered}),
            "successful_progress": successful,
            "progress_coverage": successful / len(ordered),
        },
        "provider_identities": dict(provider_identities),
        "runtime": {"device": str(device)},
        "artifacts": {"predictions": {"path": PREDICTION_ROWS_NAME, "sha256": sha256_file(rows_path)}},
        "audit": {
            "architecture_or_weights_modified": False,
            "same_canonical_roi_hash_verified_before_each_inference": True,
            "numeric_range_consumed_by_progress_backend": False,
            "manual_reference_consumed": False,
            "public_truth_opened": False,
            "restricted_namespace_images_opened": 0,
        },
    }
    assert_range_label_free(summary, location="external_prediction_summary")
    summary_path = output / "summary.json"
    atomic_new_json(summary_path, summary)
    seal = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "status": "sealed",
        "method": method,
        "summary_sha256": sha256_file(summary_path),
        "predictions_sha256": sha256_file(rows_path),
        "handoff_bundle_sha256": summary["handoff"]["bundle_sha256"],
        "bundle_sha256": canonical_sha256(
            {
                "method": method,
                "protocol_sha256": sha256_file(protocol_file),
                "handoff_bundle_sha256": summary["handoff"]["bundle_sha256"],
                "summary_sha256": sha256_file(summary_path),
                "predictions_sha256": sha256_file(rows_path),
            }
        ),
    }
    atomic_new_json(output / "seal.json", seal)
    return output


def infer_external(
    *,
    method: str,
    handoff_root: Path,
    protocol_path: Path,
    preflight_path: Path,
    output_root: Path,
    device: str,
) -> Path:
    require(method in METHODS, "unsupported external method")
    protocol_file, protocol = load_protocol(protocol_path)
    _, handoff_rows, preflight = load_handoff(
        handoff_root,
        protocol_path=protocol_file,
        preflight_path=preflight_path,
    )
    # cv2 import happens only in the explicit inference command.
    import cv2

    outputs: dict[str, dict[str, Any]] = {}
    provider_identities: dict[str, Any] = {}
    groups: Sequence[tuple[str, list[dict[str, Any]]]]
    if method == VDN_METHOD:
        groups = [
            (str(seed), [row for row in handoff_rows if int(row["oof_seed"]) == seed])
            for seed in EXPECTED_SEEDS
        ]
    else:
        groups = [("global", list(handoff_rows))]
    for route_name, route_rows in groups:
        provider = (
            _build_vdn_provider(seed=int(route_name), preflight=preflight, device=device)
            if method == VDN_METHOD
            else _build_transformer_provider(protocol=protocol, device=device)
        )
        identity = dict(provider.identity)
        identity_sha = canonical_sha256(identity)
        provider_identities[route_name] = {
            "identity": identity,
            "identity_sha256": identity_sha,
        }
        for handoff_row in route_rows:
            sample_id = str(handoff_row["sample_id"])
            checkpoint_sha = (
                str(handoff_row["vdn_checkpoint_sha256"])
                if method == VDN_METHOD
                else str(protocol[TRANSFORMER_METHOD]["transformer_checkpoint"]["sha256"])
            )
            started = time.perf_counter()
            image_path = resolve_public_image(str(handoff_row["image_relpath"]))
            require(image_path.is_relative_to(PUBLIC_IMAGE_ROOT), "external image escapes public train root")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None, f"cannot read public image: {image_path}")
            roi, _, _, _ = public_v5.canonical_tight_roi(
                image,
                handoff_row["dial_bbox"],
                output_size=int(protocol["input_contract"]["canonical_roi_size"]),
            )
            require(image_sha256(roi) == handoff_row["canonical_roi_sha256"], f"external canonical ROI hash drift: {sample_id}")
            try:
                record = provider.predict(roi, input_is_canonical_meter_roi=True)
                require(isinstance(record, Mapping), "external provider returned non-mapping")
                status, progress, failure = _normalize_provider_record(record)
                record_sha = canonical_sha256(record)
                row = {
                    "schema_version": 1,
                    "protocol": PREDICTION_PROTOCOL,
                    "method": method,
                    "sample_id": sample_id,
                    "group_id": str(handoff_row["group_id"]),
                    "canonical_roi_sha256": str(handoff_row["canonical_roi_sha256"]),
                    "oof_seed": int(handoff_row["oof_seed"]),
                    "status": status,
                    "prediction_progress": progress,
                    "failure_reason": failure,
                    "checkpoint_sha256": checkpoint_sha,
                    "provider_identity_sha256": identity_sha,
                    "provider_record_sha256": record_sha,
                    "strict_oof_eligible": method == VDN_METHOD,
                    "sample_seconds": float(time.perf_counter() - started),
                }
            except (RuntimeError, ValueError, TypeError) as error:
                row = _failed_external_row(
                    method=method,
                    handoff_row=handoff_row,
                    checkpoint_sha256=checkpoint_sha,
                    provider_identity_sha256=identity_sha,
                    reason=f"pipeline_exception:{type(error).__name__}",
                    sample_seconds=time.perf_counter() - started,
                )
            validate_external_prediction_row(
                row,
                handoff_row=handoff_row,
                transformer_checkpoint_sha256=str(protocol[TRANSFORMER_METHOD]["transformer_checkpoint"]["sha256"]),
            )
            require(sample_id not in outputs, f"duplicate external output: {sample_id}")
            outputs[sample_id] = row
        del provider
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    require(set(outputs) == {str(row["sample_id"]) for row in handoff_rows}, "external output roster incomplete")
    return _seal_external_predictions(
        method=method,
        handoff_root=handoff_root,
        protocol_path=protocol_file,
        preflight_path=preflight_path,
        output_root=output_root,
        rows=list(outputs.values()),
        provider_identities=provider_identities,
        device=device,
    )


def load_external_predictions(
    root: Path,
    *,
    handoff_root: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    preflight_path: Path = DEFAULT_PREFLIGHT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_file, protocol = load_protocol(protocol_path)
    _, handoff_rows, _ = load_handoff(
        handoff_root,
        protocol_path=protocol_file,
        preflight_path=preflight_path,
    )
    handoff_by_id = {str(row["sample_id"]): row for row in handoff_rows}
    bundle_root = guard_public_path(root, label="external prediction bundle", expect_file=False)
    require(bundle_root.is_dir(), "external prediction bundle is not a directory")
    summary_path = guard_public_path(bundle_root / "summary.json", label="external prediction summary")
    seal_path = guard_public_path(bundle_root / "seal.json", label="external prediction seal")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    method = str(summary.get("method") or "")
    require(method in METHODS, "external prediction method drift")
    require(summary.get("protocol") == PREDICTION_PROTOCOL and summary.get("status") == "predictions_sealed", "external prediction summary identity drift")
    require(seal.get("protocol") == PREDICTION_PROTOCOL and seal.get("status") == "sealed" and seal.get("method") == method, "external prediction seal identity drift")
    require(summary.get("frozen_protocol", {}).get("sha256") == sha256_file(protocol_file), "external prediction protocol drift")
    require(summary.get("comparator_source", {}).get("sha256") == sha256_file(SOURCE), "external prediction comparator source drift")
    handoff_seal = strict_json(Path(handoff_root) / "seal.json")
    require(summary.get("handoff", {}).get("bundle_sha256") == seal.get("handoff_bundle_sha256") == handoff_seal.get("bundle_sha256"), "external prediction handoff drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "external prediction summary seal drift")
    rows_path = guard_public_path(bundle_root / str(summary["artifacts"]["predictions"]["path"]), label="external prediction rows")
    require(summary["artifacts"]["predictions"]["sha256"] == seal.get("predictions_sha256") == sha256_file(rows_path), "external prediction rows seal drift")
    expected_bundle = canonical_sha256(
        {
            "method": method,
            "protocol_sha256": sha256_file(protocol_file),
            "handoff_bundle_sha256": handoff_seal["bundle_sha256"],
            "summary_sha256": sha256_file(summary_path),
            "predictions_sha256": sha256_file(rows_path),
        }
    )
    require(seal.get("bundle_sha256") == expected_bundle, "external prediction bundle seal drift")
    rows = strict_jsonl(rows_path)
    require(len(rows) == len(handoff_rows), "external prediction inventory drift")
    require(len({str(row["sample_id"]) for row in rows}) == len(rows), "external prediction duplicate samples")
    transformer_sha = protocol[TRANSFORMER_METHOD]["transformer_checkpoint"]["sha256"]
    for row in rows:
        sample_id = str(row["sample_id"])
        require(sample_id in handoff_by_id, f"external prediction outside handoff: {sample_id}")
        require(row["method"] == method, "external prediction mixed-method bundle")
        validate_external_prediction_row(
            row,
            handoff_row=handoff_by_id[sample_id],
            transformer_checkpoint_sha256=transformer_sha,
        )
    return summary, rows


def score_method(
    *,
    method: str,
    handoff_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    external_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    require(method == "garc" or method in METHODS, "unsupported score method")
    external_by_id = (
        {str(row["sample_id"]): row for row in external_rows}
        if external_rows is not None
        else {}
    )
    if method != "garc":
        require(set(external_by_id) == {str(row["sample_id"]) for row in handoff_rows}, f"{method} prediction roster drift")
    errors: list[float] = []
    covered_errors: list[float] = []
    absolute_errors: list[float] = []
    groups: list[str] = []
    successful: list[bool] = []
    failures: Counter[str] = Counter()
    for handoff in handoff_rows:
        sample_id = str(handoff["sample_id"])
        target = truth[sample_id]
        span = float(target.scale_end) - float(target.scale_start)
        require(span > 0.0, f"non-positive public scale span: {sample_id}")
        prediction: float | None = None
        reason: str | None = None
        if not handoff["range_accepted"]:
            reason = "shared_range_not_accepted"
        elif method == "garc":
            prediction = (
                _finite(handoff["garc_predicted_reading"])
                if handoff["garc_full_status"]
                else None
            )
            if prediction is None:
                reason = f"garc:{handoff['garc_failure_reason'] or 'progress_unavailable'}"
        else:
            external = external_by_id[sample_id]
            progress = _finite(external["prediction_progress"])
            if external["status"] and progress is not None:
                start = _finite(handoff["predicted_scale_start"])
                end = _finite(handoff["predicted_scale_end"])
                require(start is not None and end is not None and start != end, "accepted shared range lacks endpoints")
                prediction = start + progress * (end - start)
            else:
                reason = f"{method}:{external['failure_reason'] or 'progress_unavailable'}"
        ok = prediction is not None
        if ok:
            absolute = abs(float(prediction) - float(target.reading))
            error = absolute / span
            covered_errors.append(error)
            absolute_errors.append(absolute)
            failures["accepted"] += 1
        else:
            error = 1.0
            failures[reason or "unspecified"] += 1
        errors.append(error)
        successful.append(ok)
        groups.append(str(handoff["group_id"]))
    error_array = np.asarray(errors, dtype=np.float64)
    success_array = np.asarray(successful, dtype=bool)
    group_array = np.asarray(groups, dtype=object)
    unique_groups = np.unique(group_array)
    group_means = [float(np.mean(error_array[group_array == group])) for group in unique_groups]
    metrics = {
        "samples": len(error_array),
        "physical_groups": len(unique_groups),
        "successful": int(np.sum(success_array)),
        "coverage": float(np.mean(success_array)),
        "full_denominator_nmae_failure_penalty_1": float(np.mean(error_array)),
        "full_denominator_p95_normalized_error": float(np.quantile(error_array, 0.95)),
        "macro_group_nmae": float(np.mean(group_means)),
        "conditional_nmae": (
            float(np.mean(np.asarray(covered_errors, dtype=np.float64)))
            if covered_errors
            else None
        ),
        "conditional_reading_mae": (
            float(np.mean(np.asarray(absolute_errors, dtype=np.float64)))
            if absolute_errors
            else None
        ),
        "acc_1pct": float(np.mean(success_array & (error_array <= 0.01))),
        "acc_2pct": float(np.mean(success_array & (error_array <= 0.02))),
        "acc_5pct": float(np.mean(success_array & (error_array <= 0.05))),
        "failure_penalty_nmae": 1.0,
        "failure_breakdown": dict(sorted(failures.items())),
    }
    return metrics, error_array


def paired_group_bootstrap(
    *,
    comparator_errors: np.ndarray,
    garc_errors: np.ndarray,
    groups: Sequence[str],
    comparator_name: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    require(iterations > 0, "bootstrap iterations must be positive")
    group_array = np.asarray(groups, dtype=object)
    require(len(comparator_errors) == len(garc_errors) == len(group_array), "bootstrap vector drift")
    unique = np.unique(group_array)
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in chosen])
        differences[index] = float(
            np.mean(comparator_errors[sample_indices])
            - np.mean(garc_errors[sample_indices])
        )
    point = float(np.mean(comparator_errors) - np.mean(garc_errors))
    return {
        "comparator": comparator_name,
        "reference": "garc",
        "difference_definition": "comparator_nmae_minus_garc_nmae",
        "positive_favors": "garc",
        "point_estimate": point,
        "group_bootstrap_95ci": [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ],
        "physical_groups": len(unique),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def score_external(
    *,
    protocol_path: Path,
    preflight_path: Path,
    handoff_root: Path,
    external_roots: Mapping[str, Path],
    output_path: Path,
    bootstrap_iterations: int = 5_000,
    bootstrap_seed: int = 20260810,
) -> Path:
    protocol_file, protocol = load_protocol(protocol_path)
    handoff_summary, handoff_rows, preflight = load_handoff(
        handoff_root,
        protocol_path=protocol_file,
        preflight_path=preflight_path,
    )
    require(VDN_METHOD in external_roots, "strict score requires VDN predictions")
    require(set(external_roots) <= set(METHODS), "score received unknown method")
    authenticated: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    # All external bundles are authenticated before public numeric values open.
    for method, root in external_roots.items():
        summary, rows = load_external_predictions(
            root,
            handoff_root=handoff_root,
            protocol_path=protocol_file,
            preflight_path=preflight_path,
        )
        require(summary["method"] == method, f"external root method mismatch: {method}")
        authenticated[method] = (summary, rows)
    roster, _, _ = _cohort(protocol)
    truth = load_public_truth_after_authentication(
        protocol_path=_path_from_protocol(protocol["public_protocol"], label="automatic range public protocol"),
        roster=roster,
        include_text_boxes=False,
    )
    metrics: dict[str, Any] = {}
    error_vectors: dict[str, np.ndarray] = {}
    metrics["garc"], error_vectors["garc"] = score_method(
        method="garc",
        handoff_rows=handoff_rows,
        truth=truth,
    )
    for method, (_, rows) in authenticated.items():
        metrics[method], error_vectors[method] = score_method(
            method=method,
            handoff_rows=handoff_rows,
            truth=truth,
            external_rows=rows,
        )
    groups = [str(row["group_id"]) for row in handoff_rows]
    comparisons = {
        method: paired_group_bootstrap(
            comparator_errors=error_vectors[method],
            garc_errors=error_vectors["garc"],
            groups=groups,
            comparator_name=method,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + index,
        )
        for index, method in enumerate(authenticated)
    }
    result = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "complete",
        "scope": "same 412 public images / 19 physical entity groups / same canonical ROI / same frozen automatic numeric range",
        "frozen_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "comparator_source": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "preflight": {"path": str(Path(preflight_path).resolve(strict=True)), "sha256": sha256_file(Path(preflight_path))},
        "handoff": {
            "path": str(Path(handoff_root).resolve(strict=True)),
            "summary_sha256": sha256_file(Path(handoff_root) / "summary.json"),
            "bundle_sha256": strict_json(Path(handoff_root) / "seal.json")["bundle_sha256"],
            "range_binding_sha256": handoff_summary["garc"]["range_binding_sha256"],
        },
        "metrics": metrics,
        "paired_group_bootstrap": comparisons,
        "claim_eligibility": {
            "strict_formal_table": ["garc", VDN_METHOD],
            "fixed_checkpoint_sensitivity_table": (
                [TRANSFORMER_METHOD] if TRANSFORMER_METHOD in authenticated else []
            ),
            "original_transformer_strict_oof_claim": False,
            "vdn_strict_progress_oof_claim": preflight["methods"][VDN_METHOD]["strict_412_progress_oof_eligible"],
            "garc_all_component_joint_oof_claim": handoff_summary["garc"]["all_412_progress_geometry_head_and_backbone_group_unseen"],
        },
        "scoring": {
            "normalized_error": "abs(prediction-reading)/abs(scale_end-scale_start)",
            "failure_penalty": 1.0,
            "coverage": "finite successful end-to-end readings / 412",
            "shared_range": "the identical sealed GARC-predicted numeric endpoints and frozen acceptance threshold are used for every progress backend",
            "bootstrap_unit": "physical entity group",
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "sources": {
            method: {
                "path": str(Path(external_roots[method]).resolve(strict=True)),
                "summary_sha256": sha256_file(Path(external_roots[method]) / "summary.json"),
                "bundle_sha256": strict_json(Path(external_roots[method]) / "seal.json")["bundle_sha256"],
            }
            for method in authenticated
        },
        "audit": {
            "all_prediction_seals_authenticated_before_public_truth_opened": True,
            "manual_or_ground_truth_numeric_range_used_by_methods": False,
            "architecture_or_weights_modified": False,
            "restricted_namespace_images_opened": 0,
            "field_samples_used": 0,
            "test_samples_used": 0,
            "sealed_samples_used": 0,
            "confirmatory_samples_used": 0,
        },
    }
    output = guard_public_path(output_path, label="external score output", must_exist=False)
    atomic_new_json(output, result)
    return output


def _joint_run(value: str) -> tuple[Path, Path]:
    parts = str(value).split("::", 1)
    require(len(parts) == 2 and all(parts), "joint run must be PLAN::PREDICTION_ROOT")
    return Path(parts[0]), Path(parts[1])


def _external_root(value: str) -> tuple[str, Path]:
    parts = str(value).split("=", 1)
    require(len(parts) == 2 and parts[0] in METHODS and bool(parts[1]), "external root must be METHOD=ROOT")
    return parts[0], Path(parts[1])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--verify-only", action="store_true")

    handoff = commands.add_parser("build-handoff")
    handoff.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    handoff.add_argument("--primary-plan", type=Path, required=True)
    handoff.add_argument("--primary-prediction-root", type=Path, required=True)
    handoff.add_argument("--calibration", type=Path, required=True)
    handoff.add_argument("--joint-run", action="append", default=[])
    handoff.add_argument("--output-root", type=Path, required=True)

    infer = commands.add_parser("infer")
    infer.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    infer.add_argument("--handoff-root", type=Path, required=True)
    infer.add_argument("--method", choices=METHODS, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")

    score = commands.add_parser("score")
    score.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    score.add_argument("--handoff-root", type=Path, required=True)
    score.add_argument("--external", action="append", required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--bootstrap-iterations", type=int, default=5_000)
    score.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "preflight":
        report = build_preflight_report(protocol_path=args.protocol)
        if args.verify_only:
            output = guard_public_path(args.output, label="external preflight verification")
            require(strict_json(output) == report, "frozen external preflight differs")
            result = output
        else:
            result = write_preflight(protocol_path=args.protocol, output_path=args.output)
    elif args.command == "build-handoff":
        result = build_handoff(
            protocol_path=args.protocol,
            preflight_path=args.preflight,
            primary_plan=args.primary_plan,
            primary_prediction_root=args.primary_prediction_root,
            calibration_path=args.calibration,
            extra_joint_runs=[_joint_run(value) for value in args.joint_run],
            output_root=args.output_root,
        )
    elif args.command == "infer":
        result = infer_external(
            method=args.method,
            handoff_root=args.handoff_root,
            protocol_path=args.protocol,
            preflight_path=args.preflight,
            output_root=args.output_root,
            device=args.device,
        )
    else:
        roots = dict(_external_root(value) for value in args.external)
        require(len(roots) == len(args.external), "duplicate external method")
        result = score_external(
            protocol_path=args.protocol,
            preflight_path=args.preflight,
            handoff_root=args.handoff_root,
            external_roots=roots,
            output_path=args.output,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    print(json.dumps({"output": str(result), "sha256": sha256_file(result if result.is_file() else result / "summary.json")}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "HANDOFF_PROTOCOL",
    "PREDICTION_PROTOCOL",
    "PREFLIGHT_PROTOCOL",
    "PROTOCOL",
    "SCORE_PROTOCOL",
    "build_handoff",
    "build_preflight_report",
    "infer_external",
    "load_external_predictions",
    "load_handoff",
    "load_protocol",
    "paired_group_bootstrap",
    "score_external",
    "score_method",
    "validate_external_prediction_row",
    "validate_handoff_row",
    "write_preflight",
]
