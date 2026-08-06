"""Collect leakage-free probabilistic direction predictions for fusion training.

This collector deliberately reuses the already signed mask-side OOF rows and
their frozen meter/reference geometry.  Each vector prediction is produced by
the probabilistic direction checkpoint whose grouped validation split contains
the whole meter group.  Therefore neither expert has fitted the row it labels,
and no SyncG test or RPM-10K sample is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pivot_direction_fallback import tensor_from_bbox
from experiments.fadr_multiseed_protocol import (
    PEPD_DIRECTION_SEEDS,
    assert_train_only_path,
)
from experiments.pepd_convergence_extension_v2_protocol import (
    EXTENSION_SEED,
    PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
    PEPD_AUTHORITATIVE_COLLECTOR_CONTRACT_PROTOCOL,
    PEPD_AUTHORITATIVE_HANDOFF_PROTOCOL,
    PEPD_AUTHORITATIVE_OOF_PROTOCOL,
    PEPD_EXTENSION_PROTOCOL,
    PEPD_EXTENSION_VERIFICATION_PROTOCOL,
)
from experiments.pepd_convergence_protocol import (
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
)
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    image_angle_from_direction,
    load_syncg_manifest,
    reading_from_pointer_angle,
    sample_ids_hash,
    sha256_file,
    sha256_source_file,
)
from experiments.strict_json import (
    STRICT_JSON_PROTOCOL,
    strict_json_load,
    strict_json_source_sha256,
    strict_jsonl_load,
)


DEFAULT_SEEDS = PEPD_DIRECTION_SEEDS
EXPECTED_SOURCE_OOF_PROTOCOL = "syncg_quality_router_cross_model_oof_v1"
PEPD_COHORT_PROTOCOL = PEPD_AUTHORITATIVE_COHORT_PROTOCOL
PEPD_OOF_HANDOFF_PROTOCOL = PEPD_AUTHORITATIVE_HANDOFF_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-oof",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pepd-cohort",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pepd-oof-handoff",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return strict_jsonl_load(path)


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"{path} already exists; refusing to overwrite") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _string_set_hash(values: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(map(str, values))):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _declared_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    assert_train_only_path(resolved, label=label)
    return resolved


def _require_hash(path: Path, expected: Any, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _load_authoritative_runs(
    pepd_oof_handoff: Path,
    pepd_cohort: Path,
    manifest: Path,
    samples: Sequence[Any],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, set[str]],
    dict[int, set[str]],
    dict[int, set[str]],
    dict[str, Any],
    dict[str, Any],
]:
    handoff = _read_json(pepd_oof_handoff)
    cohort = _read_json(pepd_cohort)
    if (
        handoff.get("schema_version") != 2
        or handoff.get("protocol") != PEPD_OOF_HANDOFF_PROTOCOL
        or handoff.get("status") != "authorized"
        or handoff.get("formal_seeds") != list(DEFAULT_SEEDS)
        or handoff.get("cohort_protocol") != PEPD_COHORT_PROTOCOL
        or handoff.get("cohort_sha256") != sha256_file(pepd_cohort)
        or handoff.get("manifest_sha256") != sha256_file(manifest)
        or handoff.get("manifest_protocol_sha256")
        != sha256_file(manifest.with_name(manifest.name + ".protocol.json"))
        or handoff.get("mixed_authority")
        != {
            "20260720": PEPD_CONTINUATION_PROTOCOL,
            "20260721": PEPD_EXTENSION_PROTOCOL,
            "20260722": PEPD_CONTINUATION_PROTOCOL,
        }
        or handoff.get("further_epoch_extension_authorized") is not False
        or handoff.get("public_test_field_evaluation_authorized") is not False
    ):
        raise ValueError("PEPD OOF handoff top-level audit failed")
    if _declared_path(handoff.get("cohort"), label="PEPD handoff cohort") != pepd_cohort:
        raise ValueError("PEPD OOF handoff cohort path mismatch")
    if _declared_path(handoff.get("manifest"), label="PEPD handoff manifest") != manifest:
        raise ValueError("PEPD OOF handoff manifest path mismatch")
    if (
        cohort.get("schema_version") != 2
        or cohort.get("protocol") != PEPD_COHORT_PROTOCOL
        or cohort.get("status") != "converged"
        or cohort.get("seeds") != list(DEFAULT_SEEDS)
        or cohort.get("all_runs_verified") is not True
        or cohort.get("all_runs_converged") is not True
        or cohort.get("mixed_authority")
        != {
            "20260720": "convergence_v1",
            "20260721": "bounded_extension_v2",
            "20260722": "convergence_v1",
        }
        or cohort.get("further_epoch_extension_authorized") is not False
        or cohort.get("public_test_field_evaluation_authorized") is not False
    ):
        raise ValueError("PEPD convergence cohort audit failed")
    cohort_runs = cohort.get("runs")
    if not isinstance(cohort_runs, list) or [
        row.get("seed") if isinstance(row, Mapping) else None for row in cohort_runs
    ] != list(DEFAULT_SEEDS):
        raise ValueError("PEPD convergence cohort run identities drifted")
    cohort_by_seed = {
        int(row["seed"]): row for row in cohort_runs if isinstance(row, Mapping)
    }

    contract = handoff.get("versioned_collector_contract")
    expected_contract_keys = {
        "protocol",
        "authorized",
        "collector",
        "collector_source_sha256",
        "required_cli",
        "legacy_checkpoint_fallback_allowed",
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != expected_contract_keys
        or contract.get("protocol")
        != PEPD_AUTHORITATIVE_COLLECTOR_CONTRACT_PROTOCOL
        or contract.get("authorized") is not True
        or contract.get("collector")
        != "experiments/collect_uncertainty_fusion_oof.py"
        or contract.get("collector_source_sha256")
        != sha256_source_file(Path(__file__).resolve())
        or contract.get("required_cli")
        != ["--pepd-oof-handoff", "--pepd-cohort"]
        or contract.get("legacy_checkpoint_fallback_allowed") is not False
    ):
        raise ValueError("PEPD handoff does not authorize this versioned collector")
    assignment_contract = handoff.get("oof_assignment_contract")
    if (
        not isinstance(assignment_contract, Mapping)
        or assignment_contract.get("source_field") != "held_out_seed"
        or assignment_contract.get("allowed_seeds") != list(DEFAULT_SEEDS)
        or assignment_contract.get("fallback_policy") != "none"
        or assignment_contract.get("legacy_checkpoint_fallback_allowed") is not False
        or assignment_contract.get("test_sets_used") != []
    ):
        raise ValueError("PEPD OOF assignment contract drifted")

    runs: dict[int, dict[str, Any]] = {}
    validation_ids: dict[int, set[str]] = {}
    validation_groups: dict[int, set[str]] = {}
    training_groups: dict[int, set[str]] = {}
    authoritative_runs = handoff.get("authoritative_runs")
    if not isinstance(authoritative_runs, Mapping) or set(authoritative_runs) != {
        str(seed) for seed in DEFAULT_SEEDS
    }:
        raise ValueError("PEPD handoff authoritative run set drifted")
    for seed in DEFAULT_SEEDS:
        declared = authoritative_runs[str(seed)]
        if not isinstance(declared, Mapping) or declared.get("seed") != seed:
            raise ValueError(f"PEPD handoff seed {seed} identity mismatch")
        checkpoint_path = _declared_path(
            declared.get("authoritative_best_checkpoint"),
            label=f"PEPD seed {seed} checkpoint",
        )
        summary_path = _declared_path(
            declared.get("summary"),
            label=f"PEPD seed {seed} summary",
        )
        verification_path = _declared_path(
            declared.get("verification"),
            label=f"PEPD seed {seed} verification",
        )
        _require_hash(
            checkpoint_path,
            declared.get("authoritative_best_checkpoint_sha256"),
            label=f"PEPD seed {seed} checkpoint",
        )
        _require_hash(
            summary_path,
            declared.get("summary_sha256"),
            label=f"PEPD seed {seed} summary",
        )
        _require_hash(
            verification_path,
            declared.get("verification_sha256"),
            label=f"PEPD seed {seed} verification",
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        signature = checkpoint.get("signature") or {}
        verification = _read_json(verification_path)
        summary = _read_json(summary_path)
        expected_run_protocol = (
            PEPD_EXTENSION_PROTOCOL
            if seed == EXTENSION_SEED
            else PEPD_CONTINUATION_PROTOCOL
        )
        expected_verification_protocol = (
            PEPD_EXTENSION_VERIFICATION_PROTOCOL
            if seed == EXTENSION_SEED
            else PEPD_RUN_VERIFICATION_PROTOCOL
        )
        expected_source_phase = (
            "bounded_extension_v2"
            if seed == EXTENSION_SEED
            else "convergence_v1"
        )
        changed_from_parent = declared.get(
            "checkpoint_changed_from_legacy_parent"
        )
        authoritative_best_epoch = declared.get("authoritative_best_epoch")
        if type(changed_from_parent) is not bool:
            raise ValueError(f"PEPD seed {seed} checkpoint-change flag is malformed")
        if (
            type(authoritative_best_epoch) is not int
            or authoritative_best_epoch < 1
        ):
            raise ValueError(f"PEPD seed {seed} authoritative epoch is malformed")
        if (
            not isinstance(signature, Mapping)
            or signature.get("protocol") != PEPD_TRAINING_PROTOCOL
            or declared.get("model_protocol") != PEPD_TRAINING_PROTOCOL
            or checkpoint.get("protocol") != PEPD_TRAINING_PROTOCOL
        ):
            raise ValueError(f"wrong checkpoint protocol: {checkpoint_path}")
        if type(signature.get("seed")) is not int or signature.get("seed") != seed:
            raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
        if signature.get("manifest_sha256") != sha256_file(manifest):
            raise ValueError(f"checkpoint manifest mismatch: {checkpoint_path}")
        if _mapping_sha256(signature) != declared.get("training_signature_sha256"):
            raise ValueError(f"PEPD seed {seed} training signature hash mismatch")
        base_v1_signature = declared.get("base_v1_continuation_signature")
        authoritative_signature = declared.get("authoritative_run_signature")
        lineage_signature = declared.get("checkpoint_lineage_signature")
        lineage_protocol = declared.get("checkpoint_lineage_protocol")
        if (
            declared.get("source_phase") != expected_source_phase
            or declared.get("authoritative_run_protocol")
            != expected_run_protocol
            or declared.get("verification_protocol")
            != expected_verification_protocol
            or not isinstance(base_v1_signature, Mapping)
            or base_v1_signature.get("protocol") != PEPD_CONTINUATION_PROTOCOL
            or _mapping_sha256(base_v1_signature)
            != declared.get("base_v1_continuation_signature_sha256")
            or not isinstance(authoritative_signature, Mapping)
            or authoritative_signature.get("protocol") != expected_run_protocol
            or _mapping_sha256(authoritative_signature)
            != declared.get("authoritative_run_signature_sha256")
            or not isinstance(lineage_signature, Mapping)
            or lineage_signature.get("protocol") != lineage_protocol
            or _mapping_sha256(lineage_signature)
            != declared.get("checkpoint_lineage_signature_sha256")
        ):
            raise ValueError(
                f"PEPD seed {seed} explicit mixed-authority lineage failed"
            )
        expected_summary_signature = (
            summary.get("extension_signature")
            if seed == EXTENSION_SEED
            else summary.get("continuation_signature")
        )
        expected_base_v1_signature = (
            summary.get("v1_continuation_signature")
            if seed == EXTENSION_SEED
            else summary.get("continuation_signature")
        )
        if (
            summary.get("protocol") != expected_run_protocol
            or summary.get("status") != "complete"
            or summary.get("seed") != seed
            or summary.get("best_checkpoint_sha256") != sha256_file(checkpoint_path)
            or summary.get("best_epoch") != authoritative_best_epoch
            or summary.get("checkpoint_changed_from_frozen_parent")
            is not changed_from_parent
            or expected_summary_signature != authoritative_signature
            or expected_base_v1_signature != base_v1_signature
        ):
            raise ValueError(
                f"PEPD seed {seed} mixed-authority summary audit failed"
            )
        if checkpoint.get("epoch") != authoritative_best_epoch:
            raise ValueError(f"PEPD seed {seed} checkpoint epoch binding failed")
        if checkpoint.get("continuation_signature") != base_v1_signature:
            raise ValueError(
                f"PEPD seed {seed} base-v1 checkpoint binding failed"
            )
        if lineage_protocol == PEPD_EXTENSION_PROTOCOL:
            if (
                seed != EXTENSION_SEED
                or authoritative_best_epoch <= 60
                or checkpoint.get("extension_signature") != lineage_signature
                or lineage_signature != authoritative_signature
            ):
                raise ValueError(
                    f"PEPD seed {seed} extension checkpoint lineage failed"
                )
        elif lineage_protocol == PEPD_CONTINUATION_PROTOCOL:
            if (
                (seed == EXTENSION_SEED and authoritative_best_epoch > 60)
                or checkpoint.get("extension_signature") is not None
                or lineage_signature != base_v1_signature
            ):
                raise ValueError(
                    f"PEPD seed {seed} v1 checkpoint lineage failed"
                )
        else:
            raise ValueError(
                f"PEPD seed {seed} checkpoint lineage protocol is unauthorized"
            )
        verifier_key = (
            "extension_verifier" if seed == EXTENSION_SEED else "verifier"
        )
        verifier_path = (
            PROJECT_ROOT
            / "experiments"
            / (
                "verify_pepd_convergence_extension_v2.py"
                if seed == EXTENSION_SEED
                else "verify_pepd_convergence_run.py"
            )
        )
        if (
            verification.get("protocol") != expected_verification_protocol
            or verification.get("verified") is not True
            or verification.get("converged") is not True
            or verification.get("seed") != seed
            or verification.get("best_epoch") != authoritative_best_epoch
            or verification.get("checkpoint_changed_from_frozen_parent")
            is not changed_from_parent
            or verification.get("summary_sha256") != sha256_file(summary_path)
            or verification.get("best_checkpoint_sha256")
            != sha256_file(checkpoint_path)
            or (verification.get("source_identity") or {}).get(verifier_key)
            != declared.get("verification_source_sha256")
            or declared.get("verification_source_sha256")
            != sha256_source_file(verifier_path)
            or verification.get("public_or_field_evaluation_authorized") is not False
        ):
            raise ValueError(f"PEPD seed {seed} verification audit failed")
        if seed == EXTENSION_SEED and (
            verification.get("checkpoint_lineage_protocol")
            != lineage_protocol
            or verification.get("checkpoint_lineage_signature")
            != lineage_signature
            or verification.get("extension_budget_exhausted") is not True
            or verification.get("further_extension_authorized") is not False
        ):
            raise ValueError("PEPD seed 20260721 bounded-extension gate drifted")

        validation_fraction = float(signature.get("validation_fraction", math.nan))
        if not math.isfinite(validation_fraction):
            raise ValueError(f"PEPD seed {seed} validation fraction is non-finite")
        train, validation = grouped_train_val_split(
            samples,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        train_ids_hash = sample_ids_hash(train)
        validation_ids_hash = sample_ids_hash(validation)
        train_group_set = {str(sample.group_id) for sample in train}
        validation_group_set = {str(sample.group_id) for sample in validation}
        split = declared.get("grouped_split")
        if (
            not isinstance(split, Mapping)
            or split.get("split_seed") != seed
            or not math.isclose(
                float(split.get("validation_fraction", math.nan)),
                validation_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or split.get("manifest_sha256") != sha256_file(manifest)
            or split.get("train_samples") != len(train)
            or split.get("validation_samples") != len(validation)
            or split.get("train_sample_ids_sha256") != train_ids_hash
            or split.get("validation_sample_ids_sha256") != validation_ids_hash
            or split.get("train_groups") != len(train_group_set)
            or split.get("validation_groups") != len(validation_group_set)
            or split.get("train_group_ids_sha256")
            != _string_set_hash(train_group_set)
            or split.get("validation_group_ids_sha256")
            != _string_set_hash(validation_group_set)
            or split.get("group_overlap") != 0
            or train_group_set & validation_group_set
            or signature.get("validation_sample_ids_sha256")
            != validation_ids_hash
        ):
            raise ValueError(f"cannot reconstruct authoritative split for seed {seed}")
        cohort_row = cohort_by_seed[seed]
        if (
            cohort_row.get("verified") is not True
            or cohort_row.get("converged") is not True
            or cohort_row.get("authoritative_run_protocol")
            != expected_run_protocol
            or cohort_row.get("verification_protocol")
            != expected_verification_protocol
            or cohort_row.get("source_phase") != expected_source_phase
            or cohort_row.get("best_checkpoint_sha256")
            != sha256_file(checkpoint_path)
            or cohort_row.get("summary_sha256") != sha256_file(summary_path)
            or cohort_row.get("verification_sha256")
            != sha256_file(verification_path)
        ):
            raise ValueError(f"PEPD cohort/handoff seed {seed} binding mismatch")
        runs[seed] = {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "summary_path": summary_path,
            "verification_path": verification_path,
        }
        validation_ids[seed] = {sample.sample_id for sample in validation}
        validation_groups[seed] = validation_group_set
        training_groups[seed] = train_group_set
    return (
        runs,
        validation_ids,
        validation_groups,
        training_groups,
        handoff,
        cohort,
    )


def _validate_source_assignments(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    sample_by_id: Mapping[str, Any],
    runs: Mapping[int, Mapping[str, Any]],
    validation_ids: Mapping[int, set[str]],
    validation_groups: Mapping[int, set[str]],
    training_groups: Mapping[int, set[str]],
    require_all_seeds: bool = True,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    group_seed: dict[str, set[int]] = {}
    for row_index, row in enumerate(source_rows):
        if row.get("dataset") != "SyncG" or row.get("split") != "train":
            raise ValueError(f"source OOF row {row_index} is outside SyncG/train")
        sample_id = row.get("sample_id")
        group_id = row.get("group_id")
        seed = row.get("held_out_seed")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"source OOF row {row_index} has an invalid sample ID")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"source OOF row {row_index} has an invalid group ID")
        if type(seed) is not int or seed not in DEFAULT_SEEDS or seed not in runs:
            raise ValueError(f"{sample_id}: held-out seed is not authorized")
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f"{sample_id}: sample is outside the SyncG train manifest")
        if str(sample.group_id) != group_id:
            raise ValueError(f"{sample_id}: source/manifest physical group mismatch")
        if sample_id not in validation_ids[seed]:
            raise ValueError(
                f"{sample_id}: held-out seed is not its authoritative validation split"
            )
        if group_id not in validation_groups[seed]:
            raise ValueError(
                f"{sample_id}: physical group is not in authoritative validation groups"
            )
        if group_id in training_groups[seed]:
            raise RuntimeError(f"{sample_id}: physical group leaks into seed training")
        assignments[sample_id] = seed
        group_seed.setdefault(group_id, set()).add(seed)
    if len(assignments) != len(source_rows):
        raise ValueError("source OOF contains duplicate sample IDs")
    leaking_group = next(
        (group for group, seeds in group_seed.items() if len(seeds) != 1),
        None,
    )
    if leaking_group is not None:
        raise ValueError(
            f"physical group is assigned to multiple held-out seeds: {leaking_group}"
        )
    if require_all_seeds:
        expected_ids = set().union(*(validation_ids[seed] for seed in DEFAULT_SEEDS))
        if set(assignments) != expected_ids:
            missing = len(expected_ids - set(assignments))
            extra = len(set(assignments) - expected_ids)
            raise ValueError(
                "source OOF is not the complete union of the three "
                f"authoritative validation splits: missing={missing}, extra={extra}"
            )
        wrong_lowest_seed = next(
            (
                sample_id
                for sample_id, seed in assignments.items()
                if seed
                != min(
                    candidate
                    for candidate in DEFAULT_SEEDS
                    if sample_id in validation_ids[candidate]
                )
            ),
            None,
        )
        if wrong_lowest_seed is not None:
            raise ValueError(
                f"{wrong_lowest_seed}: assignment is not the lowest "
                "authoritative held-out seed"
            )
        if set(assignments.values()) != set(DEFAULT_SEEDS):
            raise ValueError(
                "source OOF does not cover all authoritative direction seeds"
            )
    return assignments


def _validate_output_source_bindings(
    output_rows: Sequence[Mapping[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    mutable_output_fields = {"vector", "vector_protocol", "runtime_seconds"}
    for row in output_rows:
        sample_id = row.get("sample_id")
        source = source_by_id.get(str(sample_id))
        if source is None:
            raise ValueError(f"{sample_id}: output row is absent from source OOF")
        frozen_output = {
            key: value for key, value in row.items() if key not in mutable_output_fields
        }
        frozen_source = {
            key: value
            for key, value in source.items()
            if key not in mutable_output_fields
        }
        if frozen_output != frozen_source:
            raise ValueError(f"{sample_id}: frozen source OOF fields changed")


def main() -> None:
    args = parse_args()
    for name in (
        "manifest",
        "source_oof",
        "pepd_cohort",
        "pepd_oof_handoff",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    for label, path in {
        "SyncG manifest": args.manifest,
        "source OOF": args.source_oof,
        "PEPD cohort": args.pepd_cohort,
        "PEPD OOF handoff": args.pepd_oof_handoff,
        "probabilistic OOF output": args.output,
    }.items():
        assert_train_only_path(path, label=label)
    for path in (
        args.manifest,
        args.source_oof,
        args.pepd_cohort,
        args.pepd_oof_handoff,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_meta_path = _metadata_path(args.source_oof)
    source_summary_path = _summary_path(args.source_oof)
    source_meta = _read_json(source_meta_path)
    source_summary = _read_json(source_summary_path)
    source_signature = source_meta.get("signature") or {}
    if source_signature.get("protocol") != EXPECTED_SOURCE_OOF_PROTOCOL:
        raise ValueError("source OOF has the wrong protocol")
    if source_signature.get("test_sets_used") != []:
        raise ValueError("source OOF does not certify train-only collection")
    if source_signature.get("assignment_policy") != (
        "lowest seed whose grouped validation contains sample"
    ):
        raise ValueError("source OOF assignment policy drifted")
    if (
        source_summary.get("status") != "complete"
        or int(source_summary.get("group_leakage_count", -1)) != 0
        or int(source_summary.get("test_samples_used", -1)) != 0
        or source_summary.get("output_sha256") != sha256_file(args.source_oof)
    ):
        raise ValueError("source OOF summary is incomplete or inconsistent")

    samples, manifest_protocol = load_syncg_manifest(
        args.manifest,
        expected_split="train",
    )
    sample_by_id = {sample.sample_id: sample for sample in samples}
    source_rows = _read_jsonl(args.source_oof)
    source_by_id = {str(row.get("sample_id")): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source OOF contains duplicate sample IDs")
    (
        runs,
        validation_ids,
        validation_groups,
        training_groups,
        handoff,
        cohort,
    ) = _load_authoritative_runs(
        args.pepd_oof_handoff,
        args.pepd_cohort,
        args.manifest,
        samples,
    )
    assignments = _validate_source_assignments(
        source_rows,
        sample_by_id=sample_by_id,
        runs=runs,
        validation_ids=validation_ids,
        validation_groups=validation_groups,
        training_groups=training_groups,
    )
    manifest_groups = {str(sample.group_id) for sample in samples}
    assigned_groups = {str(row["group_id"]) for row in source_rows}
    coverage = {
        "design": (
            "union of the three authoritative grouped validation splits "
            "(nominally 10% each); not complete K-fold OOF"
        ),
        "manifest_samples": len(samples),
        "assigned_samples": len(assignments),
        "manifest_groups": len(manifest_groups),
        "assigned_groups": len(assigned_groups),
        "sample_fraction": float(len(assignments) / len(samples)),
        "group_fraction": float(len(assigned_groups) / len(manifest_groups)),
        "complete_manifest_oof": False,
    }

    signature = {
        "protocol": PEPD_AUTHORITATIVE_OOF_PROTOCOL,
        "split": "SyncG/train only",
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(
            args.manifest.with_name(args.manifest.name + ".protocol.json")
        ),
        "source_oof_sha256": sha256_file(args.source_oof),
        "source_oof_metadata_sha256": sha256_file(source_meta_path),
        "source_oof_summary_sha256": sha256_file(source_summary_path),
        "pepd_cohort": str(args.pepd_cohort),
        "pepd_cohort_sha256": sha256_file(args.pepd_cohort),
        "pepd_cohort_protocol": PEPD_COHORT_PROTOCOL,
        "pepd_oof_handoff": str(args.pepd_oof_handoff),
        "pepd_oof_handoff_sha256": sha256_file(args.pepd_oof_handoff),
        "pepd_oof_handoff_protocol": PEPD_OOF_HANDOFF_PROTOCOL,
        "direction_runs": {
            str(seed): {
                "checkpoint_sha256": sha256_file(info["checkpoint_path"]),
                "summary_sha256": sha256_file(info["summary_path"]),
                "verification_sha256": sha256_file(info["verification_path"]),
                "authoritative_run_protocol": handoff[
                    "authoritative_runs"
                ][str(seed)]["authoritative_run_protocol"],
                "verification_protocol": handoff["authoritative_runs"][
                    str(seed)
                ]["verification_protocol"],
                "checkpoint_lineage_protocol": handoff[
                    "authoritative_runs"
                ][str(seed)]["checkpoint_lineage_protocol"],
                "validation_samples": len(validation_ids[seed]),
                "validation_groups": len(validation_groups[seed]),
            }
            for seed, info in sorted(runs.items())
        },
        "assignment_policy": (
            "reuse only the signed mask-side row identity/front-end; rebuild "
            "every vector prediction from the handoff-authorized checkpoint "
            "whose grouped validation split contains the complete physical group"
        ),
        "legacy_checkpoint_fallback_allowed": False,
        "assigned_samples": len(assignments),
        "assigned_groups": len(assigned_groups),
        "oof_coverage": coverage,
        "strict_json_protocol": STRICT_JSON_PROTOCOL,
        "strict_json_source_sha256": strict_json_source_sha256(),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "test_sets_used": [],
    }
    metadata_path = _metadata_path(args.output)
    summary_path = _summary_path(args.output)
    if args.resume:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("resume requires output and metadata files")
        if summary_path.exists():
            raise FileExistsError("completed OOF output cannot be resumed")
        if (_read_json(metadata_path).get("signature") or {}) != signature:
            raise ValueError("OOF resume signature mismatch")
        existing_rows = (
            [] if args.output.stat().st_size == 0 else _read_jsonl(args.output)
        )
        completed = {str(row.get("sample_id")) for row in existing_rows}
        if len(completed) != len(existing_rows):
            raise ValueError("existing OOF output contains duplicate IDs")
        if not completed.issubset(assignments):
            raise ValueError("existing OOF output contains unauthorized sample IDs")
        _validate_output_source_bindings(
            existing_rows,
            source_by_id=source_by_id,
        )
        _validate_source_assignments(
            existing_rows,
            sample_by_id=sample_by_id,
            runs=runs,
            validation_ids=validation_ids,
            validation_groups=validation_groups,
            training_groups=training_groups,
            require_all_seeds=False,
        )
    else:
        existing = [
            path
            for path in (args.output, metadata_path, summary_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "OOF output namespace already exists; refusing to overwrite: "
                + ", ".join(map(str, existing))
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8"):
            pass
        completed: set[str] = set()
        _write_json_no_clobber(
            metadata_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "signature": signature,
                "manifest_protocol": manifest_protocol,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                },
                "handoff_scope": handoff.get("scope"),
                "cohort_scope": cohort.get("scope"),
            },
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp

    for seed, info in sorted(runs.items()):
        checkpoint = info["checkpoint"]
        training_signature = checkpoint.get("signature") or {}
        pending = [
            row
            for row in source_rows
            if assignments[str(row.get("sample_id"))] == seed
            and str(row.get("sample_id")) not in completed
        ]
        if not pending:
            continue
        model = build_probabilistic_pivot_direction_model(
            angle_bins=int(training_signature["angle_bins"]),
            imagenet_pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        image_size = int(training_signature["image_size"])
        heatmap_size = int(training_signature["heatmap_size"])
        expansion = float(training_signature["expansion"])
        batch: list[dict[str, Any]] = []

        @torch.inference_mode()
        def flush_batch() -> None:
            if not batch:
                return
            inputs = torch.stack([item["tensor"] for item in batch]).to(
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(inputs)
            prediction = decode_probabilistic_pivot_direction(
                *(value.float() for value in outputs)
            )
            pivot_prob = torch.sigmoid(outputs[0].float()[:, 0]).reshape(len(batch), -1)
            spatial = pivot_prob / torch.clamp(pivot_prob.sum(dim=1, keepdim=True), min=1e-8)
            pivot_entropy = -torch.sum(
                spatial * torch.log(torch.clamp(spatial, min=1e-12)), dim=1
            ) / math.log(float(pivot_prob.shape[1]))
            top2 = torch.topk(pivot_prob, k=2, dim=1).values
            raw_norm = torch.linalg.vector_norm(outputs[1].float(), dim=1)
            arrays = {
                "pivot": prediction.pivot_xy.cpu().numpy(),
                "direction": prediction.direction.cpu().numpy(),
                "peak": prediction.pivot_peak.cpu().numpy(),
                "valid": prediction.valid.cpu().numpy(),
                "angle_std": prediction.angle_std_degrees.cpu().numpy(),
                "angle_entropy": prediction.angle_entropy.cpu().numpy(),
                "log_variance": prediction.log_variance.cpu().numpy(),
                "bin_resultant": prediction.bin_resultant_length.cpu().numpy(),
                "pivot_entropy": pivot_entropy.cpu().numpy(),
                "pivot_margin": (top2[:, 0] - top2[:, 1]).cpu().numpy(),
                "raw_norm": raw_norm.cpu().numpy(),
            }
            stride = float(image_size) / float(heatmap_size)
            output_rows: list[dict[str, Any]] = []
            for index, item in enumerate(batch):
                source = item["source"]
                sample = item["sample"]
                payload: dict[str, Any] = {
                    "status": False,
                    "prediction": None,
                    "progress": None,
                    "pointer_angle": None,
                    "direction": None,
                    "pivot_heatmap_xy": arrays["pivot"][index].tolist(),
                    "pivot_input_xy": (arrays["pivot"][index] * stride).tolist(),
                    "pivot_peak": float(arrays["peak"][index]),
                    "pivot_spatial_entropy": float(arrays["pivot_entropy"][index]),
                    "pivot_top2_margin": float(arrays["pivot_margin"][index]),
                    "direction_raw_norm": float(arrays["raw_norm"][index]),
                    "angle_std_degrees": float(arrays["angle_std"][index]),
                    "angle_log_variance": float(arrays["log_variance"][index]),
                    "angle_bin_entropy": float(arrays["angle_entropy"][index]),
                    "angle_bin_resultant_length": float(arrays["bin_resultant"][index]),
                    "direction_angle_error_degrees": None,
                    "error_code": None,
                }
                if bool(arrays["valid"][index]):
                    direction = arrays["direction"][index].astype(np.float64)
                    try:
                        pointer_angle = image_angle_from_direction(direction)
                        reading, progress = reading_from_pointer_angle(
                            pointer_angle,
                            start_angle=item["start_angle"],
                            range_angle=item["range_angle"],
                            scale_start=sample.scale_start,
                            scale_end=sample.scale_end,
                        )
                        target = np.asarray(sample.pointer_tip) - np.asarray(sample.pointer_tail)
                        target /= max(float(np.linalg.norm(target)), 1e-12)
                        cosine = float(np.clip(np.dot(direction, target), -1.0, 1.0))
                        payload.update(
                            {
                                "status": True,
                                "prediction": float(reading),
                                "progress": float(progress),
                                "pointer_angle": float(pointer_angle),
                                "direction": direction.tolist(),
                                "direction_angle_error_degrees": float(
                                    np.degrees(np.arccos(cosine))
                                ),
                            }
                        )
                    except ValueError as exc:
                        payload["error_code"] = "reading_conversion_failed"
                        payload["error_message"] = str(exc)
                else:
                    payload["error_code"] = "invalid_direction"
                row = dict(source)
                row["vector"] = payload
                row["runtime_seconds"] = float(time.perf_counter() - item["started"])
                row["vector_protocol"] = PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
                output_rows.append(row)
            _append_rows(args.output, output_rows)
            completed.update(str(row["sample_id"]) for row in output_rows)
            batch.clear()

        for source in tqdm(pending, desc=f"probabilistic OOF {seed}", dynamic_ncols=True):
            started = time.perf_counter()
            sample_id = str(source.get("sample_id"))
            sample = sample_by_id[sample_id]
            front_end = source.get("front_end") or {}
            bbox = front_end.get("meter_bbox")
            start_angle = _finite(front_end.get("start_angle"))
            range_angle = _finite(front_end.get("range_angle"))
            if (
                not isinstance(bbox, Sequence)
                or len(bbox) < 4
                or start_angle is None
                or range_angle is None
                or abs(range_angle) <= 1e-8
            ):
                row = dict(source)
                row["vector"] = {
                    "status": False,
                    "prediction": None,
                    "error_code": "frozen_front_end_failed",
                }
                row["vector_protocol"] = PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
                row["runtime_seconds"] = float(time.perf_counter() - started)
                _append_rows(args.output, [row])
                completed.add(sample_id)
                continue
            image = cv2.imread(
                sample.image_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                raise ValueError(f"cannot read {sample.image_path}")
            batch.append(
                {
                    "source": source,
                    "sample": sample,
                    "tensor": tensor_from_bbox(
                        image,
                        [float(value) for value in bbox[:4]],
                        image_size=image_size,
                        expansion=expansion,
                    ),
                    "start_angle": start_angle,
                    "range_angle": range_angle,
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        flush_batch()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_rows = _read_jsonl(args.output)
    output_ids = [str(row.get("sample_id")) for row in output_rows]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(assignments):
        raise RuntimeError("probabilistic OOF output is incomplete or duplicated")
    output_assignments = _validate_source_assignments(
        output_rows,
        sample_by_id=sample_by_id,
        runs=runs,
        validation_ids=validation_ids,
        validation_groups=validation_groups,
        training_groups=training_groups,
    )
    if output_assignments != assignments:
        raise RuntimeError("probabilistic OOF output assignment identity drifted")
    _validate_output_source_bindings(
        output_rows,
        source_by_id=source_by_id,
    )
    membership_counts = Counter(
        sum(sample_id in ids for ids in validation_ids.values()) for sample_id in output_ids
    )
    base_success = sum((row.get("base") or {}).get("status") is True for row in output_rows)
    vector_success = sum((row.get("vector") or {}).get("status") is True for row in output_rows)
    joint_success = sum(
        (row.get("base") or {}).get("status") is True
        and (row.get("vector") or {}).get("status") is True
        for row in output_rows
    )
    summary = {
        "schema_version": 1,
        "protocol": PEPD_AUTHORITATIVE_OOF_PROTOCOL,
        "status": "complete",
        "samples": len(output_rows),
        "groups": len({str(row.get("group_id")) for row in output_rows}),
        "oof_coverage": coverage,
        "base_success": base_success,
        "vector_success": vector_success,
        "joint_success": joint_success,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "validation_membership_counts": {
            str(key): value for key, value in sorted(membership_counts.items())
        },
        "signature": signature,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    _write_json_no_clobber(summary_path, summary)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(summary_path)


if __name__ == "__main__":
    main()
