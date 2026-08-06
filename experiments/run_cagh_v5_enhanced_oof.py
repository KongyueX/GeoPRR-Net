"""Run submission-grade enhanced-V5 OOF on authoritative PEPD holdouts.

The three authoritative PEPD seeds are grouped holdout replicates, not three
disjoint CV partitions.  This runner therefore trains one fresh ScaleMark V5
head on each PEPD training complement, evaluates that head only on the same
PEPD held-out groups, and deterministically assigns overlapping held-out groups
to the lowest eligible PEPD seed.  Rows outside the union of the three
authoritative validation splits receive no fallback prediction.

Only SyncG/train paths are admitted.  Validate-only loads manifests, signed
lineage files, and checkpoints but reads no image.  Preflight reads a bounded
number of public images and executes forward/backward plumbing without an
optimizer step.  Formal mode is the only mode that trains heads.
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
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.collect_uncertainty_fusion_oof as authoritative
import experiments.train_cagh_scalemark_reference_probe_v5 as base
import experiments.train_cagh_scalemark_reference_probe_v5_enhanced as enhanced
from experiments.cagh_scalemark_reference_head_v5 import (
    CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
    build_head,
    multiscale_dense_tick_loss,
)
from experiments.fadr_multiseed_protocol import sha256_file
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


PROTOCOL = "cagh_v5_enhanced_authoritative_pepd_oof_v1"
PROTOCOL_PATH = PROJECT_ROOT / "experiments/cagh_v5_enhanced_oof_protocol.json"
MANIFEST = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
MANIFEST_PROTOCOL = MANIFEST.with_name(MANIFEST.name + ".protocol.json")
HANDOFF = PROJECT_ROOT / "artifacts/runs/pepd_convergence_authoritative_v2/oof_handoff.json"
COHORT = PROJECT_ROOT / "artifacts/runs/pepd_convergence_authoritative_v2/cohort.json"
RUNNER_SOURCE = Path(__file__).resolve()
COLLECTOR_SOURCE = PROJECT_ROOT / "experiments/collect_uncertainty_fusion_oof.py"
HEAD_SOURCE = PROJECT_ROOT / "experiments/cagh_scalemark_reference_head_v5.py"
BASE_TRAINER_SOURCE = PROJECT_ROOT / "experiments/train_cagh_scalemark_reference_probe_v5.py"
ENHANCED_TRAINER_SOURCE = (
    PROJECT_ROOT / "experiments/train_cagh_scalemark_reference_probe_v5_enhanced.py"
)
DEFAULT_OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_v5_enhanced_oof")
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cache"
PEPD_SEEDS = (20260720, 20260721, 20260722)
HEAD_SEEDS = {
    20260720: 20261520,
    20260721: 20261521,
    20260722: 20261522,
}
EXPECTED_PUBLIC = (16_000, 725)
EXPECTED_UNION = (4_380, 197)
EXPECTED_ASSIGNED_SAMPLES = {20260720: 1_625, 20260721: 1_465, 20260722: 1_290}
EXPECTED_ASSIGNED_GROUPS = {20260720: 73, 20260721: 65, 20260722: 59}


@dataclass(frozen=True)
class FoldSpec:
    pepd_seed: int
    head_seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    train_samples: tuple[Any, ...]
    validation_samples: tuple[Any, ...]
    train_groups: frozenset[str]
    validation_groups: frozenset[str]
    assigned_sample_ids: frozenset[str]
    assigned_group_ids: frozenset[str]
    split_identity: Mapping[str, Any]


@dataclass(frozen=True)
class Discovery:
    samples: tuple[Any, ...]
    sample_by_id: Mapping[str, Any]
    folds: tuple[FoldSpec, ...]
    sample_assignment: Mapping[str, int]
    group_assignment: Mapping[str, int]
    audit: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value!r}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        _require(isinstance(row, dict), f"non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _line_set_sha256(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(map(str, values))):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assignment_sha256(assignments: Mapping[str, int]) -> str:
    return _line_set_sha256({f"{key}\t{int(value)}" for key, value in assignments.items()})


def _canonical_sha256(value: Any) -> str:
    return base.canonical_sha256(value)


def _public_c_root(path: Path, *, label: str) -> Path:
    resolved = base._guard_restricted_path(Path(path), label=label)
    _require(resolved != Path(resolved.anchor), f"{label} cannot be a drive root")
    _require(resolved.drive.casefold() == "c:", f"{label} must be on C: by protocol")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preflight-samples-per-partition", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only completed fold artifacts whose hashes and OOF roster revalidate.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def load_protocol() -> Mapping[str, Any]:
    _require(PROTOCOL_PATH.is_file(), f"missing OOF protocol: {PROTOCOL_PATH}")
    protocol = _strict_json(PROTOCOL_PATH)
    _require(protocol.get("schema_version") == 1, "OOF protocol schema drift")
    _require(protocol.get("protocol") == PROTOCOL, "OOF protocol identity drift")
    _require(protocol.get("status") == "frozen_public_only", "OOF protocol is not frozen")
    _require(
        protocol.get("public_test_field_evaluation_authorized") is False,
        "OOF protocol unexpectedly authorizes external evaluation",
    )
    artifacts = {
        "manifest_sha256": MANIFEST,
        "manifest_protocol_sha256": MANIFEST_PROTOCOL,
        "pepd_oof_handoff_sha256": HANDOFF,
        "pepd_cohort_sha256": COHORT,
    }
    sources = {
        "runner_source_sha256": RUNNER_SOURCE,
        "authoritative_collector_source_sha256": COLLECTOR_SOURCE,
        "enhanced_head_source_sha256": HEAD_SOURCE,
        "base_tight_roi_adapter_source_sha256": BASE_TRAINER_SOURCE,
        "enhanced_trainer_source_sha256": ENHANCED_TRAINER_SOURCE,
    }
    inputs = protocol.get("inputs")
    _require(isinstance(inputs, Mapping), "OOF protocol inputs are missing")
    for key, path in artifacts.items():
        _require(path.is_file(), f"missing OOF artifact input: {path}")
        _require(sha256_file(path) == inputs.get(key), f"{key} drift")
    for key, path in sources.items():
        _require(path.is_file(), f"missing OOF source input: {path}")
        _require(sha256_source_file(path) == inputs.get(key), f"{key} drift")
    _require(
        _canonical_sha256(base.CROP_CONTRACT) == protocol.get("crop_contract_sha256"),
        "canonical tight-ROI contract drift",
    )
    _require(
        protocol.get("head", {}).get("protocol")
        == CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        "enhanced V5 head protocol drift",
    )
    return protocol


def _validate_public_manifest(samples: Sequence[Any], manifest_protocol: Mapping[str, Any]) -> None:
    _require(manifest_protocol.get("dataset") == "SyncG", "manifest dataset drift")
    groups = {str(sample.group_id) for sample in samples}
    _require((len(samples), len(groups)) == EXPECTED_PUBLIC, "SyncG/train inventory drift")
    image_root = (PROJECT_ROOT / "datasets/SyncG/syncG/images/train").resolve()
    annotation_root = (PROJECT_ROOT / "datasets/SyncG/syncG/annotations/train").resolve()
    seen: set[str] = set()
    for sample in samples:
        _require(sample.sample_id not in seen, f"duplicate manifest sample: {sample.sample_id}")
        seen.add(sample.sample_id)
        _require(
            sample.dataset == "SyncG" and sample.split == "train",
            f"sample outside SyncG/train: {sample.sample_id}",
        )
        image_path = base._require_under(
            Path(sample.image_path), image_root, label=f"{sample.sample_id}.image"
        )
        annotation_value = sample.metadata.get("annotation_path")
        _require(isinstance(annotation_value, str) and annotation_value, "missing annotation path")
        annotation_path = base._require_under(
            Path(annotation_value), annotation_root, label=f"{sample.sample_id}.annotation"
        )
        _require(image_path.is_file(), f"missing public image: {sample.sample_id}")
        _require(annotation_path.is_file(), f"missing public annotation: {sample.sample_id}")
        base._scalemark_points(sample)


def _pairwise_counts(values: Mapping[int, set[str]]) -> dict[str, int]:
    return {
        f"{left}_{right}": len(values[left] & values[right])
        for index, left in enumerate(PEPD_SEEDS)
        for right in PEPD_SEEDS[index + 1 :]
    }


def _multiplicity(values: Mapping[int, set[str]]) -> dict[str, int]:
    union = set().union(*(values[seed] for seed in PEPD_SEEDS))
    counts = Counter(sum(value in values[seed] for seed in PEPD_SEEDS) for value in union)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def discover(protocol: Mapping[str, Any]) -> Discovery:
    samples, manifest_protocol = load_syncg_manifest(MANIFEST, expected_split="train")
    _validate_public_manifest(samples, manifest_protocol)
    sample_by_id = {str(sample.sample_id): sample for sample in samples}
    (
        runs,
        validation_ids,
        validation_groups,
        training_groups,
        handoff,
        cohort,
    ) = authoritative._load_authoritative_runs(HANDOFF, COHORT, MANIFEST, samples)
    _require(tuple(runs) == PEPD_SEEDS, "authoritative PEPD seed order drift")

    train_by_seed: dict[int, tuple[Any, ...]] = {}
    validation_by_seed: dict[int, tuple[Any, ...]] = {}
    fold_identities: dict[int, dict[str, Any]] = {}
    for seed in PEPD_SEEDS:
        signature = runs[seed]["checkpoint"].get("signature") or {}
        fraction = float(signature.get("validation_fraction", math.nan))
        _require(math.isfinite(fraction), f"seed {seed} validation fraction is invalid")
        train, validation = grouped_train_val_split(
            samples, validation_fraction=fraction, seed=seed
        )
        train_tuple, validation_tuple = tuple(train), tuple(validation)
        train_by_seed[seed] = train_tuple
        validation_by_seed[seed] = validation_tuple
        declared = handoff["authoritative_runs"][str(seed)]
        checkpoint_path = Path(runs[seed]["checkpoint_path"]).resolve(strict=True)
        identity = {
            "pepd_seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(declared["authoritative_best_epoch"]),
            "source_phase": str(declared["source_phase"]),
            "run_protocol": str(declared["authoritative_run_protocol"]),
            "validation_fraction": fraction,
            "train_samples": len(train_tuple),
            "train_groups": len(training_groups[seed]),
            "validation_samples": len(validation_tuple),
            "validation_groups": len(validation_groups[seed]),
            "train_sample_ids_sha256": sample_ids_hash(train_tuple),
            "validation_sample_ids_sha256": sample_ids_hash(validation_tuple),
            "train_group_ids_sha256": _line_set_sha256(training_groups[seed]),
            "validation_group_ids_sha256": _line_set_sha256(validation_groups[seed]),
            "group_overlap": len(training_groups[seed] & validation_groups[seed]),
        }
        _require(identity["group_overlap"] == 0, f"seed {seed} group leakage")
        fold_identities[seed] = identity

    group_members: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        group_members[str(sample.group_id)].add(str(sample.sample_id))
    union_ids = set().union(*(validation_ids[seed] for seed in PEPD_SEEDS))
    union_groups = set().union(*(validation_groups[seed] for seed in PEPD_SEEDS))
    sample_assignment: dict[str, int] = {}
    group_assignment: dict[str, int] = {}
    for group_id in sorted(union_groups):
        eligible = tuple(seed for seed in PEPD_SEEDS if group_id in validation_groups[seed])
        _require(bool(eligible), f"OOF group has no eligible PEPD seed: {group_id}")
        for seed in eligible:
            _require(
                group_members[group_id].issubset(validation_ids[seed]),
                f"seed {seed} does not hold out complete group {group_id}",
            )
            _require(group_id not in training_groups[seed], f"seed {seed} leaks group {group_id}")
        selected = min(eligible)
        group_assignment[group_id] = selected
        for sample_id in group_members[group_id]:
            _require(sample_id in union_ids, f"group member escaped OOF union: {sample_id}")
            sample_assignment[sample_id] = selected
    _require(set(sample_assignment) == union_ids, "strict OOF assignment does not cover union")
    _require(set(group_assignment) == union_groups, "strict OOF group assignment drift")
    _require((len(union_ids), len(union_groups)) == EXPECTED_UNION, "OOF union inventory drift")

    sample_sets = {seed: set(validation_ids[seed]) for seed in PEPD_SEEDS}
    group_sets = {seed: set(validation_groups[seed]) for seed in PEPD_SEEDS}
    assigned_samples = Counter(sample_assignment.values())
    assigned_groups = Counter(group_assignment.values())
    _require(dict(assigned_samples) == EXPECTED_ASSIGNED_SAMPLES, "assigned sample counts drift")
    _require(dict(assigned_groups) == EXPECTED_ASSIGNED_GROUPS, "assigned group counts drift")

    audit = {
        "schema_version": 1,
        "assignment_rule": "lowest eligible PEPD seed, applied to complete physical groups",
        "fallback_policy": "none",
        "manifest_samples": len(samples),
        "manifest_groups": len(groups := {str(sample.group_id) for sample in samples}),
        "eligible_union_samples": len(union_ids),
        "eligible_union_groups": len(union_groups),
        "eligible_manifest_fraction": len(union_ids) / len(samples),
        "unassigned_manifest_samples": len(samples) - len(union_ids),
        "pairwise_validation_sample_overlap": _pairwise_counts(sample_sets),
        "pairwise_validation_group_overlap": _pairwise_counts(group_sets),
        "sample_validation_multiplicity": _multiplicity(sample_sets),
        "group_validation_multiplicity": _multiplicity(group_sets),
        "assigned_samples_by_seed": {str(seed): assigned_samples[seed] for seed in PEPD_SEEDS},
        "assigned_groups_by_seed": {str(seed): assigned_groups[seed] for seed in PEPD_SEEDS},
        "union_sample_ids_sha256": _line_set_sha256(union_ids),
        "union_group_ids_sha256": _line_set_sha256(union_groups),
        "sample_assignment_sha256": _assignment_sha256(sample_assignment),
        "group_assignment_sha256": _assignment_sha256(group_assignment),
        "all_rows_jointly_unseen_by_pepd_and_head": True,
        "field_samples_read": 0,
        "public_test_samples_read": 0,
        "cohort_status": cohort.get("status"),
    }

    expected_assignment = protocol.get("assignment")
    _require(isinstance(expected_assignment, Mapping), "protocol assignment block missing")
    for key in (
        "eligible_union_samples",
        "eligible_union_groups",
        "unassigned_manifest_samples",
        "pairwise_validation_sample_overlap",
        "pairwise_validation_group_overlap",
        "sample_validation_multiplicity",
        "group_validation_multiplicity",
        "assigned_samples_by_seed",
        "assigned_groups_by_seed",
        "union_sample_ids_sha256",
        "union_group_ids_sha256",
        "sample_assignment_sha256",
        "group_assignment_sha256",
    ):
        _require(audit[key] == expected_assignment.get(key), f"assignment {key} drift")

    declared_folds = protocol.get("folds")
    _require(
        isinstance(declared_folds, list)
        and [row.get("pepd_seed") for row in declared_folds if isinstance(row, Mapping)]
        == list(PEPD_SEEDS),
        "protocol fold roster drift",
    )
    declared_by_seed = {int(row["pepd_seed"]): row for row in declared_folds}
    folds: list[FoldSpec] = []
    for seed in PEPD_SEEDS:
        identity = fold_identities[seed]
        declared = declared_by_seed[seed]
        for key, value in identity.items():
            _require(declared.get(key) == value, f"seed {seed} fold identity {key} drift")
        _require(declared.get("head_seed") == HEAD_SEEDS[seed], f"seed {seed} head seed drift")
        assigned_ids = frozenset(
            sample_id for sample_id, assigned in sample_assignment.items() if assigned == seed
        )
        assigned_group_ids = frozenset(
            group_id for group_id, assigned in group_assignment.items() if assigned == seed
        )
        checkpoint_path = Path(runs[seed]["checkpoint_path"]).resolve(strict=True)
        folds.append(
            FoldSpec(
                pepd_seed=seed,
                head_seed=HEAD_SEEDS[seed],
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=sha256_file(checkpoint_path),
                train_samples=train_by_seed[seed],
                validation_samples=validation_by_seed[seed],
                train_groups=frozenset(training_groups[seed]),
                validation_groups=frozenset(validation_groups[seed]),
                assigned_sample_ids=assigned_ids,
                assigned_group_ids=assigned_group_ids,
                split_identity=identity,
            )
        )

    del runs
    gc.collect()
    return Discovery(
        samples=tuple(samples),
        sample_by_id=sample_by_id,
        folds=tuple(folds),
        sample_assignment=sample_assignment,
        group_assignment=group_assignment,
        audit=audit,
    )


def _weighted_records(samples: Sequence[Any], partition: str) -> list[base.PublicRecord]:
    counts = Counter(str(sample.group_id) for sample in samples)
    _require(bool(counts), f"empty {partition} roster")
    return [
        base.PublicRecord(
            sample=sample,
            group_weight=len(samples) / (len(counts) * counts[str(sample.group_id)]),
            partition=partition,
        )
        for sample in samples
    ]


def _training_args(protocol: Mapping[str, Any], fold: FoldSpec, workers: int) -> argparse.Namespace:
    frozen = protocol["training"]
    augmentation = protocol["augmentation"]
    values = {
        "seed": fold.head_seed,
        "stage_a_epochs": int(frozen["stage_a_epochs"]),
        "stage_b_epochs": int(frozen["stage_b_epochs"]),
        "batch_size": int(frozen["batch_size"]),
        "workers": int(workers),
        "learning_rate": float(frozen["learning_rate"]),
        "weight_decay": float(frozen["weight_decay"]),
        "augmentation_profile": "photo",
    }
    values.update(augmentation)
    args = argparse.Namespace(**values)
    observed = base.augmentation_from_args(args)
    _require(observed.as_dict() == augmentation, "formal photo augmentation drift")
    return args


def _validate_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
        _require(
            "NVIDIA" in torch.cuda.get_device_name(device).upper(),
            "formal/preflight CUDA device is not NVIDIA",
        )
    return device


def _validation_record(
    protocol: Mapping[str, Any], discovery: Discovery, args: argparse.Namespace
) -> dict[str, Any]:
    mode = "formal" if args.run_formal else "preflight" if args.preflight else "validate_only"
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "mode": mode,
        "evidence_role": protocol["evidence_role"],
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "manifest_samples": len(discovery.samples),
            "manifest_groups": len({str(sample.group_id) for sample in discovery.samples}),
            "field_samples_read": 0,
            "public_test_samples_read": 0,
            "confirmatory_samples_read": 0,
            "sealed_samples_read": 0,
        },
        "folds": [
            {
                **dict(fold.split_identity),
                "head_seed": fold.head_seed,
                "assigned_samples": len(fold.assigned_sample_ids),
                "assigned_groups": len(fold.assigned_group_ids),
                "assigned_sample_ids_sha256": _line_set_sha256(fold.assigned_sample_ids),
                "assigned_group_ids_sha256": _line_set_sha256(fold.assigned_group_ids),
            }
            for fold in discovery.folds
        ],
        "overlap_and_assignment_audit": discovery.audit,
        "preprocessing": {
            "crop_contract": base.CROP_CONTRACT,
            "crop_contract_sha256": _canonical_sha256(base.CROP_CONTRACT),
            "constant_or_black_border": False,
            "detector_used": False,
        },
        "augmentation": protocol["augmentation"],
        "training": protocol["training"],
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "manifest": sha256_file(MANIFEST),
            "manifest_protocol": sha256_file(MANIFEST_PROTOCOL),
            "pepd_oof_handoff": sha256_file(HANDOFF),
            "pepd_cohort": sha256_file(COHORT),
            "runner_source": sha256_source_file(RUNNER_SOURCE),
            "authoritative_collector_source": sha256_source_file(COLLECTOR_SOURCE),
            "enhanced_head_source": sha256_source_file(HEAD_SOURCE),
            "base_tight_roi_adapter_source": sha256_source_file(BASE_TRAINER_SOURCE),
            "enhanced_trainer_source": sha256_source_file(ENHANCED_TRAINER_SOURCE),
        },
        "output_root": str(_public_c_root(args.output_root, label="OOF output root")),
        "cache_root": str(_public_c_root(args.cache_root, label="OOF cache root")),
    }


def _preflight_fold(
    protocol: Mapping[str, Any],
    fold: FoldSpec,
    *,
    count: int,
    workers: int,
    device: torch.device,
) -> dict[str, Any]:
    _require(count >= 1, "preflight sample count must be positive")
    fit = _weighted_records(fold.train_samples[:count], "preflight_fit")
    heldout = _weighted_records(fold.validation_samples[:count], "preflight_heldout")
    train_args = _training_args(protocol, fold, workers=0)
    train_args.batch_size = min(count, int(protocol["training"]["batch_size"]))
    train_dataset = base.CanonicalTightROIDataset(
        fit,
        training=True,
        seed=fold.head_seed,
        augmentation=base.augmentation_from_args(train_args),
    )
    train_dataset.set_epoch(1)
    heldout_dataset = base.CanonicalTightROIDataset(
        heldout,
        training=False,
        seed=fold.head_seed,
        augmentation=base.PhotoAugmentation.disabled(),
    )
    base.seed_everything(fold.head_seed)
    backbone, backbone_identity = base.load_pepd(fold.checkpoint_path, device)
    _require(backbone_identity["sha256"] == fold.checkpoint_sha256, "preflight checkpoint drift")
    head = build_head().to(device)
    loader = base._loader(
        train_dataset,
        batch_size=train_args.batch_size,
        workers=0,
        shuffle=False,
        seed=fold.head_seed,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader))
    images = batch["image"].to(device)
    with torch.no_grad():
        features = backbone.forward_multiscale_features(images)
        pooled = backbone.direction_features(features.c5)
        decoded = enhanced.decode_probabilistic_pivot_direction(
            backbone.pivot_head(features.c5),
            backbone.vector_head(pooled),
            backbone.angle_head(pooled),
            backbone.log_variance_head(pooled),
        )
    output = head(features.c2.detach(), features.c5.detach())
    weights = batch["group_weight"].to(device)
    tick_loss, _ = multiscale_dense_tick_loss(
        output, batch["tick_heatmap"].to(device), group_weight=weights
    )
    inverse = enhanced._projective_local_inverse(
        batch["final_to_isotropic"].to(device), decoded.pivot_xy / 63.0
    )
    geometry_loss = enhanced.enhanced_geometry_loss(
        output,
        batch["endpoints"].to(device),
        batch["gt_start"].to(device),
        batch["gt_range"].to(device),
        decoded.pivot_xy / 63.0,
        inverse,
        weights,
    )
    combined = tick_loss + geometry_loss
    _require(bool(torch.isfinite(combined)), f"seed {fold.pepd_seed} preflight loss non-finite")
    combined.backward()
    gradients = [
        parameter.grad
        for parameter in head.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    _require(bool(gradients), f"seed {fold.pepd_seed} preflight produced no head gradients")
    _require(
        all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
        f"seed {fold.pepd_seed} preflight gradient non-finite",
    )
    _require(
        all(parameter.grad is None for parameter in backbone.parameters()),
        f"seed {fold.pepd_seed} frozen PEPD received gradients",
    )
    metrics, rows = enhanced.evaluate_enhanced(
        backbone, head, heldout_dataset, args=train_args, device=device
    )
    _require(len(rows) == len(heldout), "preflight heldout inventory drift")
    result = {
        "pepd_seed": fold.pepd_seed,
        "head_seed": fold.head_seed,
        "checkpoint_sha256": fold.checkpoint_sha256,
        "public_fit_images_read": len(fit),
        "public_heldout_images_read": len(heldout),
        "field_images_read": 0,
        "optimizer_steps": 0,
        "tick_loss": float(tick_loss.detach()),
        "geometry_loss": float(geometry_loss.detach()),
        "head_gradients_finite": True,
        "backbone_gradients_absent": True,
        "augmentation_codes": [int(value) for value in batch["augmentation_code"]],
        "heldout_metrics": metrics,
    }
    del head, backbone, batch, features, output, gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def run_preflight(
    protocol: Mapping[str, Any],
    discovery: Discovery,
    validation: Mapping[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    device = _validate_device(args.device)
    started = time.time()
    folds = [
        _preflight_fold(
            protocol,
            fold,
            count=args.preflight_samples_per_partition,
            workers=args.workers,
            device=device,
        )
        for fold in discovery.folds
    ]
    record = {
        **dict(validation),
        "status": "preflight_complete",
        "fold_preflight": folds,
        "public_images_read": sum(
            row["public_fit_images_read"] + row["public_heldout_images_read"] for row in folds
        ),
        "field_images_read": 0,
        "optimizer_steps": 0,
        "elapsed_seconds": time.time() - started,
    }
    path = output_root / "preflight.json"
    base.atomic_json(path, record)
    return path


def _rewrite_fold_rows(
    rows: Sequence[Mapping[str, Any]], fold: FoldSpec, discovery: Discovery
) -> list[dict[str, Any]]:
    validation_ids = {str(sample.sample_id) for sample in fold.validation_samples}
    rewritten: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        sample_id = str(source.get("sample_id") or "")
        group_id = str(source.get("group_id") or "")
        _require(sample_id in validation_ids, f"seed {fold.pepd_seed} non-heldout row")
        _require(sample_id not in seen, f"seed {fold.pepd_seed} duplicate telemetry row")
        seen.add(sample_id)
        sample = discovery.sample_by_id[sample_id]
        _require(group_id == str(sample.group_id), f"{sample_id}: telemetry group drift")
        _require(group_id in fold.validation_groups, f"{sample_id}: group not held out")
        _require(group_id not in fold.train_groups, f"{sample_id}: group leaks into fit")
        selected = discovery.sample_assignment[sample_id] == fold.pepd_seed
        row = dict(source)
        row.update(
            {
                "schema_version": 2,
                "protocol": PROTOCOL,
                "underlying_evaluator_protocol": enhanced.PROTOCOL,
                "dataset": "SyncG",
                "split": "train",
                "pepd_held_out_seed": fold.pepd_seed,
                "head_training_seed": fold.head_seed,
                "pepd_checkpoint_sha256": fold.checkpoint_sha256,
                "strict_oof_selected": selected,
                "jointly_unseen": {
                    "pepd_group_absent_from_training": True,
                    "enhanced_head_group_absent_from_training": True,
                    "complete_physical_group_held_out": True,
                },
            }
        )
        rewritten.append(row)
    _require(seen == validation_ids, f"seed {fold.pepd_seed} telemetry inventory incomplete")
    return rewritten


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot score an empty OOF table")
    errors = np.asarray([float(row["absolute_progress_error"]) for row in rows], dtype=np.float64)
    _require(bool(np.isfinite(errors).all()), "OOF errors contain non-finite values")
    valid = np.asarray(
        [bool((row.get("validity") or {}).get("combined")) for row in rows], dtype=bool
    )
    groups: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors, strict=True):
        groups[str(row["group_id"])].append(float(error))
    group_means = np.asarray([np.mean(values) for values in groups.values()], dtype=np.float64)
    success_error = errors[valid]
    return {
        "samples": len(rows),
        "groups": len(groups),
        "coverage": float(valid.mean()),
        "full_denominator_nmae": float(errors.mean()),
        "success_only_nmae": float(success_error.mean()) if len(success_error) else None,
        "median_absolute_progress_error": float(np.median(errors)),
        "p95_absolute_progress_error": float(np.quantile(errors, 0.95)),
        "group_macro_full_denominator_nmae": float(group_means.mean()),
        "failed_rows_penalty": 1.0,
    }


def _save_fold_checkpoint(
    path: Path,
    fold: FoldSpec,
    head: torch.nn.Module,
    history: Mapping[str, Any],
    validation_record: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "pepd_seed": fold.pepd_seed,
        "head_seed": fold.head_seed,
        "pepd_checkpoint_sha256": fold.checkpoint_sha256,
        "split_identity_sha256": _canonical_sha256(dict(fold.split_identity)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "history": history,
        "metrics": metrics,
        "validation_record_sha256": _canonical_sha256(validation_record),
        "head_state": {
            name: tensor.detach().cpu() for name, tensor in head.state_dict().items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    fold: FoldSpec,
    discovery: Discovery,
    *,
    assigned_only: bool,
) -> None:
    expected = (
        set(fold.assigned_sample_ids)
        if assigned_only
        else {str(sample.sample_id) for sample in fold.validation_samples}
    )
    observed: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id in expected, f"unexpected fold row: {sample_id}")
        _require(sample_id not in observed, f"duplicate fold row: {sample_id}")
        observed.add(sample_id)
        _require(row.get("protocol") == PROTOCOL, f"{sample_id}: protocol drift")
        _require(row.get("pepd_held_out_seed") == fold.pepd_seed, f"{sample_id}: seed drift")
        _require(row.get("head_training_seed") == fold.head_seed, f"{sample_id}: head seed drift")
        _require(
            row.get("pepd_checkpoint_sha256") == fold.checkpoint_sha256,
            f"{sample_id}: checkpoint drift",
        )
        selected = discovery.sample_assignment[sample_id] == fold.pepd_seed
        _require(row.get("strict_oof_selected") is selected, f"{sample_id}: assignment drift")
        if assigned_only:
            _require(selected, f"{sample_id}: non-selected row in strict fold table")
        group_id = str(discovery.sample_by_id[sample_id].group_id)
        _require(group_id not in fold.train_groups, f"{sample_id}: resumed row leaks group")
        unseen = row.get("jointly_unseen") or {}
        _require(all(unseen.get(key) is True for key in unseen), f"{sample_id}: unseen audit drift")
    _require(observed == expected, "fold artifact roster is incomplete")


def _reuse_fold(
    fold_dir: Path, fold: FoldSpec, discovery: Discovery
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    summary_path = fold_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = _strict_json(summary_path)
    _require(summary.get("protocol") == PROTOCOL, "resume fold protocol drift")
    _require(summary.get("status") == "complete", "resume fold is incomplete")
    _require(summary.get("pepd_seed") == fold.pepd_seed, "resume PEPD seed drift")
    _require(summary.get("head_seed") == fold.head_seed, "resume head seed drift")
    _require(
        summary.get("split_identity_sha256") == _canonical_sha256(dict(fold.split_identity)),
        "resume split identity drift",
    )
    artifacts = summary.get("artifacts") or {}
    required = {
        "checkpoint": "checkpoint_sha256",
        "full_holdout_telemetry": "full_holdout_telemetry_sha256",
        "strict_assigned_telemetry": "strict_assigned_telemetry_sha256",
    }
    paths: dict[str, Path] = {}
    for path_key, hash_key in required.items():
        value = artifacts.get(path_key)
        _require(isinstance(value, str) and value, f"resume {path_key} missing")
        path = Path(value).resolve(strict=True)
        _require(path.is_relative_to(fold_dir.resolve()), f"resume {path_key} escaped fold dir")
        _require(sha256_file(path) == artifacts.get(hash_key), f"resume {path_key} hash drift")
        paths[path_key] = path
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "resume checkpoint is malformed")
    _require(checkpoint.get("protocol") == PROTOCOL, "resume checkpoint protocol drift")
    _require(checkpoint.get("pepd_seed") == fold.pepd_seed, "resume checkpoint seed drift")
    _require(isinstance(checkpoint.get("head_state"), Mapping), "resume head state missing")
    full_rows = _strict_jsonl(paths["full_holdout_telemetry"])
    strict_rows = _strict_jsonl(paths["strict_assigned_telemetry"])
    _validate_rows(full_rows, fold, discovery, assigned_only=False)
    _validate_rows(strict_rows, fold, discovery, assigned_only=True)
    return summary, strict_rows


def _run_fold(
    protocol: Mapping[str, Any],
    validation_record: Mapping[str, Any],
    fold: FoldSpec,
    discovery: Discovery,
    *,
    workers: int,
    device: torch.device,
    fold_dir: Path,
    cache_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_args = _training_args(protocol, fold, workers)
    augmentation = base.augmentation_from_args(train_args)
    fit_records = _weighted_records(fold.train_samples, "fit")
    validation_records = _weighted_records(fold.validation_samples, "heldout")
    cache_signature = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "pepd_seed": fold.pepd_seed,
        "head_seed": fold.head_seed,
        "pepd_checkpoint_sha256": fold.checkpoint_sha256,
        "crop_contract_sha256": _canonical_sha256(base.CROP_CONTRACT),
        "fit_sample_ids_sha256": sample_ids_hash(fold.train_samples),
        "heldout_sample_ids_sha256": sample_ids_hash(fold.validation_samples),
        "features_cached": False,
    }
    cache_path = cache_root / f"pepd_seed_{fold.pepd_seed}" / "roster_signature.json"
    if cache_path.is_file():
        _require(_strict_json(cache_path) == cache_signature, "fold roster cache drift")
    else:
        base.atomic_json(cache_path, cache_signature)

    base.seed_everything(fold.head_seed)
    backbone, backbone_identity = base.load_pepd(fold.checkpoint_path, device)
    _require(backbone_identity["sha256"] == fold.checkpoint_sha256, "formal checkpoint drift")
    head = build_head().to(device)
    fit_dataset = base.CanonicalTightROIDataset(
        fit_records, training=True, seed=fold.head_seed, augmentation=augmentation
    )
    heldout_dataset = base.CanonicalTightROIDataset(
        validation_records,
        training=False,
        seed=fold.head_seed,
        augmentation=base.PhotoAugmentation.disabled(),
    )
    started = time.time()
    history = enhanced.train_enhanced_head(
        backbone, head, fit_dataset, args=train_args, device=device
    )
    evaluator_metrics, source_rows = enhanced.evaluate_enhanced(
        backbone, head, heldout_dataset, args=train_args, device=device
    )
    full_rows = _rewrite_fold_rows(source_rows, fold, discovery)
    strict_rows = [row for row in full_rows if row["strict_oof_selected"]]
    _validate_rows(full_rows, fold, discovery, assigned_only=False)
    _validate_rows(strict_rows, fold, discovery, assigned_only=True)
    full_path = fold_dir / "full_holdout_telemetry.jsonl"
    strict_path = fold_dir / "strict_assigned_telemetry.jsonl"
    base.atomic_jsonl(full_path, full_rows)
    base.atomic_jsonl(strict_path, strict_rows)
    strict_metrics = _metrics(strict_rows)
    checkpoint_path = fold_dir / "checkpoint.pt"
    checkpoint_sha = _save_fold_checkpoint(
        checkpoint_path,
        fold,
        head,
        history,
        validation_record,
        strict_metrics,
    )
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "pepd_seed": fold.pepd_seed,
        "head_seed": fold.head_seed,
        "pepd_checkpoint": backbone_identity,
        "split_identity": dict(fold.split_identity),
        "split_identity_sha256": _canonical_sha256(dict(fold.split_identity)),
        "jointly_unseen_contract": True,
        "history": history,
        "full_holdout_metrics": evaluator_metrics,
        "strict_assigned_metrics": strict_metrics,
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "full_holdout_telemetry": str(full_path.resolve()),
            "full_holdout_telemetry_sha256": sha256_file(full_path),
            "strict_assigned_telemetry": str(strict_path.resolve()),
            "strict_assigned_telemetry_sha256": sha256_file(strict_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = fold_dir / "summary.json"
    base.atomic_json(summary_path, summary)
    del head, backbone, fit_dataset, heldout_dataset, source_rows, full_rows
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return summary, strict_rows


def run_formal(
    protocol: Mapping[str, Any],
    discovery: Discovery,
    validation_record: Mapping[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    cache_root: Path,
) -> Path:
    _require(args.workers >= 0, "workers must be non-negative")
    device = _validate_device(args.device)
    started = time.time()
    fold_summaries: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for fold in discovery.folds:
        fold_dir = output_root / "folds" / f"pepd_seed_{fold.pepd_seed}"
        reused = _reuse_fold(fold_dir, fold, discovery) if args.resume else None
        if not args.resume:
            _require(not (fold_dir / "summary.json").exists(), "formal fold exists; use --resume")
        if reused is not None:
            summary, rows = reused
            print(f"OOF fold pepd_seed={fold.pepd_seed} reused", flush=True)
        else:
            print(f"OOF fold pepd_seed={fold.pepd_seed} training", flush=True)
            summary, rows = _run_fold(
                protocol,
                validation_record,
                fold,
                discovery,
                workers=args.workers,
                device=device,
                fold_dir=fold_dir,
                cache_root=cache_root,
            )
        fold_summaries.append(summary)
        strict_rows.extend(rows)

    _require(len(strict_rows) == EXPECTED_UNION[0], "merged strict OOF sample count drift")
    observed_ids = [str(row["sample_id"]) for row in strict_rows]
    _require(len(set(observed_ids)) == len(observed_ids), "merged strict OOF duplicate rows")
    _require(set(observed_ids) == set(discovery.sample_assignment), "merged OOF roster drift")
    manifest_order = {str(sample.sample_id): index for index, sample in enumerate(discovery.samples)}
    strict_rows.sort(key=lambda row: manifest_order[str(row["sample_id"])])
    observed_groups = {str(row["group_id"]) for row in strict_rows}
    _require(observed_groups == set(discovery.group_assignment), "merged OOF group roster drift")
    for row in strict_rows:
        sample_id = str(row["sample_id"])
        _require(
            row["pepd_held_out_seed"] == discovery.sample_assignment[sample_id],
            f"{sample_id}: final assignment drift",
        )
    merged_path = output_root / "strict_oof_predictions.jsonl"
    base.atomic_jsonl(merged_path, strict_rows)
    metrics = _metrics(strict_rows)
    fold_artifacts = [
        {
            "pepd_seed": summary["pepd_seed"],
            "head_seed": summary["head_seed"],
            "summary": str(
                (output_root / "folds" / f"pepd_seed_{summary['pepd_seed']}" / "summary.json").resolve()
            ),
            "summary_sha256": sha256_file(
                output_root / "folds" / f"pepd_seed_{summary['pepd_seed']}" / "summary.json"
            ),
            "strict_assigned_metrics": summary["strict_assigned_metrics"],
        }
        for summary in fold_summaries
    ]
    strict_summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "evidence_role": protocol["evidence_role"],
        "metrics": metrics,
        "overlap_and_assignment_audit": discovery.audit,
        "folds": fold_artifacts,
        "artifacts": {
            "strict_oof_predictions": str(merged_path.resolve()),
            "strict_oof_predictions_sha256": sha256_file(merged_path),
        },
        "field_samples_read": 0,
        "public_test_samples_read": 0,
        "elapsed_seconds": time.time() - started,
    }
    strict_summary_path = output_root / "strict_oof_summary.json"
    base.atomic_json(strict_summary_path, strict_summary)
    final = {
        **dict(validation_record),
        "status": "complete",
        "strict_oof": strict_summary,
        "artifacts": {
            "strict_oof_summary": str(strict_summary_path.resolve()),
            "strict_oof_summary_sha256": sha256_file(strict_summary_path),
            "strict_oof_predictions": str(merged_path.resolve()),
            "strict_oof_predictions_sha256": sha256_file(merged_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_root / "summary.json"
    base.atomic_json(summary_path, final)
    return summary_path


def run(args: argparse.Namespace) -> Path:
    _require(args.workers >= 0, "workers must be non-negative")
    _require(not args.resume or args.run_formal, "--resume is valid only with --run-formal")
    output_root = _public_c_root(args.output_root, label="OOF output root")
    cache_root = _public_c_root(args.cache_root, label="OOF cache root")
    protocol = load_protocol()
    discovery = discover(protocol)
    validation = _validation_record(protocol, discovery, args)
    validation_path = output_root / "validation.json"
    base.atomic_json(validation_path, validation)
    if args.validate_only:
        print(validation_path, flush=True)
        return validation_path
    if args.preflight:
        path = run_preflight(protocol, discovery, validation, args, output_root)
        print(path, flush=True)
        return path
    path = run_formal(protocol, discovery, validation, args, output_root, cache_root)
    print(path, flush=True)
    return path


if __name__ == "__main__":
    run(parse_args())
