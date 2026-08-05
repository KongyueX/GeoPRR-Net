"""Build the authoritative PEPD checkpoint/split contract for OOF rebuilds.

The versioned ``collect_uncertainty_fusion_oof.py`` must consume this handoff
and the bound cohort explicitly.  Legacy checkpoint-root fallback is forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_SEEDS,
    PEPD_COHORT_PROTOCOL,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    formal_manifest_path,
    formal_output_dir,
    formal_parent_pin,
    sha256_file,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


HANDOFF_PROTOCOL = "pepd_epoch60_oof_handoff_v1"
VERSIONED_COLLECTOR_CONTRACT_PROTOCOL = (
    "pepd_epoch60_uncertainty_fusion_oof_collector_contract_v1"
)
COHORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_convergence_phase2"
    / "cohort.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_convergence_phase2"
    / "oof_handoff.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _string_set_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(map(str, values))):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _versioned_collector_contract() -> dict[str, Any]:
    collector_path = (
        PROJECT_ROOT
        / "experiments"
        / "collect_uncertainty_fusion_oof.py"
    )
    return {
        "protocol": VERSIONED_COLLECTOR_CONTRACT_PROTOCOL,
        "authorized": True,
        "collector": "experiments/collect_uncertainty_fusion_oof.py",
        "collector_source_sha256": sha256_source_file(collector_path),
        "required_cli": [
            "--pepd-oof-handoff",
            "--pepd-cohort",
        ],
        "legacy_checkpoint_fallback_allowed": False,
    }


def build_handoff(manifest: Path) -> dict[str, Any]:
    manifest = formal_manifest_path(manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    cohort = _load(COHORT_PATH)
    if cohort.get("protocol") != PEPD_COHORT_PROTOCOL:
        raise ValueError("PEPD convergence cohort protocol mismatch")
    if (
        cohort.get("all_runs_verified") is not True
        or cohort.get("all_runs_converged") is not True
    ):
        raise ValueError("PEPD convergence cohort is not verified/converged")
    if cohort.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("PEPD convergence cohort scope drifted")
    if tuple(cohort.get("seeds") or ()) != FORMAL_SEEDS:
        raise ValueError("PEPD convergence cohort seed set/order drifted")
    cohort_rows = cohort.get("runs")
    if not isinstance(cohort_rows, list):
        raise ValueError("PEPD convergence cohort has no run rows")
    cohort_by_seed = {
        int(row.get("seed", -1)): row
        for row in cohort_rows
        if isinstance(row, Mapping)
    }
    if set(cohort_by_seed) != set(FORMAL_SEEDS) or len(cohort_rows) != len(
        FORMAL_SEEDS
    ):
        raise ValueError("PEPD convergence cohort run membership drifted")
    expected_source_identity = {
        "strict_json": strict_json_source_sha256(),
        "protocol": sha256_source_file(
            PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
        ),
        "trainer": sha256_source_file(
            PROJECT_ROOT / "experiments" / "train_pepd_convergence_syncg.py"
        ),
        "verifier": sha256_source_file(
            PROJECT_ROOT / "experiments" / "verify_pepd_convergence_run.py"
        ),
        "summarizer": sha256_source_file(
            PROJECT_ROOT / "experiments" / "summarize_pepd_convergence.py"
        ),
    }
    if cohort.get("source_identity") != expected_source_identity:
        raise ValueError(
            "PEPD convergence cohort source identity is stale or incomplete"
        )
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    runs: dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        pin = formal_parent_pin(seed)
        run_dir = formal_output_dir(seed)
        summary_path = run_dir / "summary.json"
        verification_path = run_dir / "verification.json"
        checkpoint_path = run_dir / "best.pt"
        summary = _load(summary_path)
        verification = _load(verification_path)
        if verification.get("protocol") != PEPD_RUN_VERIFICATION_PROTOCOL:
            raise ValueError(f"seed {seed}: wrong continuation verifier")
        if (
            verification.get("verified") is not True
            or verification.get("converged") is not True
        ):
            raise ValueError(f"seed {seed}: run is not verified/converged")
        if int(verification.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed}: verification seed mismatch")
        if verification.get("summary_sha256") != sha256_file(summary_path):
            raise ValueError(f"seed {seed}: summary changed after verification")
        checkpoint_hash = sha256_file(checkpoint_path)
        if verification.get("best_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"seed {seed}: checkpoint changed after verification")
        verification_hash = sha256_file(verification_path)
        cohort_row = cohort_by_seed[seed]
        cohort_bindings = {
            "verified": True,
            "converged": True,
            "best_checkpoint_sha256": checkpoint_hash,
            "summary_sha256": sha256_file(summary_path),
            "verification_sha256": verification_hash,
        }
        for name, expected in cohort_bindings.items():
            if cohort_row.get(name) != expected:
                raise ValueError(
                    f"seed {seed}: cohort {name} binding mismatch"
                )
        if summary.get("protocol") != PEPD_CONTINUATION_PROTOCOL:
            raise ValueError(f"seed {seed}: continuation summary protocol mismatch")
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed}: continuation summary seed mismatch")
        if summary.get("best_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"seed {seed}: summary checkpoint hash mismatch")
        if int(summary.get("best_epoch", -1)) != int(
            verification.get("best_epoch", -2)
        ):
            raise ValueError(f"seed {seed}: best epoch binding mismatch")
        verification_sources = verification.get("source_identity")
        if not isinstance(verification_sources, Mapping):
            raise ValueError(f"seed {seed}: verification source identity missing")
        for name in ("strict_json", "protocol", "trainer", "verifier"):
            if verification_sources.get(name) != expected_source_identity[name]:
                raise ValueError(
                    f"seed {seed}: verification {name} source is stale"
                )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        training_signature = checkpoint.get("signature") or {}
        if training_signature.get("protocol") != PEPD_TRAINING_PROTOCOL:
            raise ValueError(f"seed {seed}: wrong model checkpoint protocol")
        if int(training_signature.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed}: checkpoint seed mismatch")
        if training_signature != summary.get("parent_training_signature"):
            raise ValueError(f"seed {seed}: training signature binding mismatch")
        if int(checkpoint.get("epoch", -1)) != int(summary["best_epoch"]):
            raise ValueError(f"seed {seed}: checkpoint epoch binding mismatch")
        train, validation = grouped_train_val_split(
            samples,
            validation_fraction=float(training_signature["validation_fraction"]),
            seed=seed,
        )
        train_ids_hash = sample_ids_hash(train)
        validation_ids_hash = sample_ids_hash(validation)
        if (
            train_ids_hash != pin.train_ids_sha256
            or validation_ids_hash != pin.validation_ids_sha256
        ):
            raise ValueError(f"seed {seed}: grouped split identity drifted")
        train_groups = {sample.group_id for sample in train}
        validation_groups = {sample.group_id for sample in validation}
        if train_groups & validation_groups:
            raise ValueError(f"seed {seed}: grouped split leaks groups")
        continuation_signature = summary.get("continuation_signature") or {}
        if continuation_signature.get("protocol") != PEPD_CONTINUATION_PROTOCOL:
            raise ValueError(f"seed {seed}: continuation signature mismatch")
        if bool(summary["checkpoint_changed_from_frozen_parent"]):
            if checkpoint.get("continuation_signature") != continuation_signature:
                raise ValueError(
                    f"seed {seed}: improved checkpoint continuation binding mismatch"
                )
        continuation_signature_sha256 = hashlib.sha256(
            json.dumps(
                continuation_signature,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        runs[str(seed)] = {
            "seed": seed,
            "authoritative_best_checkpoint": str(checkpoint_path),
            "authoritative_best_checkpoint_sha256": checkpoint_hash,
            "authoritative_best_epoch": int(summary["best_epoch"]),
            "checkpoint_changed_from_legacy_parent": bool(
                summary["checkpoint_changed_from_frozen_parent"]
            ),
            "model_protocol": PEPD_TRAINING_PROTOCOL,
            "training_signature_sha256": hashlib.sha256(
                json.dumps(
                    training_signature,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "continuation_protocol": PEPD_CONTINUATION_PROTOCOL,
            "continuation_signature": continuation_signature,
            "continuation_signature_sha256": (
                continuation_signature_sha256
            ),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "verification": str(verification_path),
            "verification_sha256": verification_hash,
            "verification_protocol": PEPD_RUN_VERIFICATION_PROTOCOL,
            "verification_source_sha256": verification["source_identity"][
                "verifier"
            ],
            "verified_source_identity": dict(verification_sources),
            "grouped_split": {
                "source_manifest": str(manifest),
                "manifest_sha256": FORMAL_MANIFEST_SHA256,
                "validation_fraction": float(
                    training_signature["validation_fraction"]
                ),
                "split_seed": seed,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "train_sample_ids_sha256": train_ids_hash,
                "validation_sample_ids_sha256": validation_ids_hash,
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "train_group_ids_sha256": _string_set_hash(train_groups),
                "validation_group_ids_sha256": _string_set_hash(
                    validation_groups
                ),
                "group_overlap": 0,
            },
        }
    production_changed = bool(
        runs["20260722"]["checkpoint_changed_from_legacy_parent"]
    )
    collector_contract = _versioned_collector_contract()
    collector_source_sha256 = collector_contract[
        "collector_source_sha256"
    ]
    return {
        "schema_version": 1,
        "protocol": HANDOFF_PROTOCOL,
        "status": "authorized",
        "scope": "SyncG official train grouped validation OOF rebuild only",
        "cohort": str(COHORT_PATH),
        "cohort_sha256": sha256_file(COHORT_PATH),
        "cohort_protocol": PEPD_COHORT_PROTOCOL,
        "cohort_source_identity": expected_source_identity,
        "manifest": str(manifest),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "formal_seeds": list(FORMAL_SEEDS),
        "authoritative_runs": runs,
        "oof_assignment_contract": {
            "source_field": "held_out_seed",
            "allowed_seeds": list(FORMAL_SEEDS),
            "membership_rule": (
                "sample_id must belong to the authoritative grouped-validation "
                "split recomputed from manifest, validation_fraction, and seed"
            ),
            "group_rule": (
                "the sample's complete group must be absent from that seed's "
                "training groups"
            ),
            "fallback_policy": "none",
            "legacy_checkpoint_fallback_allowed": False,
            "test_sets_used": [],
        },
        "versioned_collector_contract": collector_contract,
        "downstream_rebuild": {
            "seed_20260722_checkpoint_changed": production_changed,
            "fadr_rebuild_required": production_changed,
            "udsf_rebuild_required": production_changed,
            "oof_vector_predictions_rebuild_required": any(
                bool(run["checkpoint_changed_from_legacy_parent"])
                for run in runs.values()
            ),
        },
        "public_test_field_evaluation_authorized": False,
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "builder": sha256_source_file(Path(__file__).resolve()),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "grouped_split": sha256_source_file(
                PROJECT_ROOT / "experiments" / "vdn_baseline.py"
            ),
            "versioned_collector": collector_source_sha256,
        },
    }


def _write_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"{path} exists with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"formal OOF handoff output must be {DEFAULT_OUTPUT}")
    handoff = build_handoff(args.manifest)
    _write_or_validate(output, handoff)
    print(
        json.dumps(
            handoff,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
