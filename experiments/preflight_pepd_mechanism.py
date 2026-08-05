"""Preflight the frozen three-seed PEPD mechanism-ablation parents.

The primary mechanism cohort contains exactly:

* ``full``;
* ``paired_supervision_only`` (legacy alias: no-equivariance-loss);
* ``no_projective_pair``.

The two uncertainty-objective arms are implemented as a separate executable
same-start secondary cohort.  This script does not authorize or perform model
selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    GROUPED_VAL_CONDITIONS,
    LEGACY_MECHANISM_PARENT_PINS,
    LEGACY_PARENT_TRAINER_SOURCE_SHA256,
    MECHANISM_ARMS,
    PARENT_EPOCH,
    PEPD_MECHANISM_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PRIMARY_MECHANISM_ARMS,
    PROJECT_ROOT,
    expected_full_configuration,
    formal_manifest_path,
    formal_parent_pin,
    mechanism_parent_dir,
    primary_mechanism_arm,
    sha256_file,
)
from experiments.pepd_convergence_extension_v2_protocol import (
    PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
    authoritative_cohort_path,
)
from experiments.preflight_pepd_convergence import (
    _audit_parent_checkpoint,
    audit_parent,
)
from experiments.summarize_pepd_convergence_authoritative_v2 import (
    build_cohort as build_authoritative_v2_cohort,
)
from experiments.vdn_baseline import (
    SOURCE_TEXT_SHA256_PROTOCOL,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


AUTHORITATIVE_V2_COHORT_SHA256 = (
    "98fe962b3888b8f850143eb3430ec4726fa7acd222c9af9bd7ec5e459f0b4165"
)


def audit_authoritative_pepd_v2_cohort(
    path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed on the exact mixed-authoritative v2 PEPD authority.

    Rebuilding the document audits every source summary, verification, and
    checkpoint hash.  The separate manifest checks keep this mechanism-only
    gate bound to the frozen SyncG/train inventory without authorizing any
    public, test, field, sealed, or confirmatory input.
    """

    expected_path = authoritative_cohort_path()
    resolved = expected_path if path is None else Path(path).resolve()
    if resolved != expected_path:
        raise ValueError(
            f"formal authoritative PEPD v2 cohort must be {expected_path}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    cohort_sha256 = sha256_file(resolved)
    if cohort_sha256 != AUTHORITATIVE_V2_COHORT_SHA256:
        raise ValueError("authoritative PEPD v2 cohort SHA-256 drifted")

    manifest = formal_manifest_path(FORMAL_MANIFEST)
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train manifest protocol hash drifted")

    value = _load(resolved)
    recomputed = build_authoritative_v2_cohort()
    if value != recomputed:
        raise ValueError(
            "authoritative PEPD v2 cohort no longer matches its audited sources"
        )
    if (
        value.get("schema_version") != 2
        or value.get("protocol") != PEPD_AUTHORITATIVE_COHORT_PROTOCOL
        or value.get("status") != "converged"
        or value.get("all_runs_verified") is not True
        or value.get("all_runs_converged") is not True
        or value.get("seeds") != list(FORMAL_SEEDS)
    ):
        raise ValueError("authoritative PEPD v2 cohort is not verified/converged")
    rows = value.get("runs")
    if not isinstance(rows, list) or [
        row.get("seed") if isinstance(row, Mapping) else None for row in rows
    ] != list(FORMAL_SEEDS):
        raise ValueError("authoritative PEPD v2 run membership/order drifted")
    for row in rows:
        seed = int(row["seed"])
        if row.get("verified") is not True or row.get("converged") is not True:
            raise ValueError(f"authoritative PEPD v2 seed {seed} is not ready")
        for name in (
            "best_checkpoint_sha256",
            "summary_sha256",
            "verification_sha256",
        ):
            digest = row.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"seed {seed} has invalid {name}")
    if (
        value.get("grouped_validation_controlled_perspective_authorized")
        is not True
        or value.get("grouped_validation_controlled_robustness_authorized")
        is not True
        or value.get("public_test_field_evaluation_authorized") is not False
    ):
        raise ValueError("authoritative PEPD v2 scope/authorization drifted")
    return {
        "path": str(resolved),
        "sha256": cohort_sha256,
        "protocol": PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
        "schema_version": 2,
        "seeds": list(FORMAL_SEEDS),
        "verified": True,
        "converged": True,
        "manifest": str(manifest),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "public_test_field_evaluation_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--require-main-convergence",
        action="store_true",
        help=(
            "Fail closed unless the frozen three-seed main PEPD cohort is "
            "verified and converged."
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def expected_mechanism_configuration(name: str) -> dict[str, Any]:
    arm = primary_mechanism_arm(name)
    configuration = expected_full_configuration()
    configuration.update(
        {
            "perspective_probability": arm.perspective_probability,
            "paired_supervision_weight": arm.paired_supervision_weight,
            "equivariance_weight": arm.equivariance_weight,
        }
    )
    return configuration


def audit_mechanism_parent(
    name: str,
    seed: int,
    manifest: Path,
) -> dict[str, Any]:
    arm = primary_mechanism_arm(name)
    if name == "full":
        result = audit_parent(seed, manifest)
        result["arm"] = name
        return result
    pin = formal_parent_pin(seed)
    run_dir = mechanism_parent_dir(name, seed)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    for path in (summary_path, best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if seed == 20260722:
        legacy = LEGACY_MECHANISM_PARENT_PINS[name]
        exact = {
            summary_path: legacy["summary_sha256"],
            best_path: legacy["best_sha256"],
            last_path: legacy["last_sha256"],
        }
        for path, expected in exact.items():
            if sha256_file(path) != expected:
                raise ValueError(f"legacy mechanism parent hash drifted: {path}")
    summary = _load(summary_path)
    if summary.get("protocol") != PEPD_TRAINING_PROTOCOL:
        raise ValueError("mechanism parent training protocol mismatch")
    if summary.get("status") != "complete":
        raise ValueError("mechanism parent is not complete")
    signature = summary.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("mechanism parent has no signature")
    for key, expected in expected_mechanism_configuration(name).items():
        actual = signature.get(key)
        if isinstance(expected, float):
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"mechanism parent signature {key} mismatch")
        elif actual != expected:
            raise ValueError(f"mechanism parent signature {key} mismatch")
    exact_identity = {
        "seed": seed,
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
        "model_source_sha256": FORMAL_MODEL_SOURCE_SHA256,
    }
    for key, expected in exact_identity.items():
        if signature.get(key) != expected:
            raise ValueError(f"mechanism parent identity {key} mismatch")
    source_protocol = signature.get("source_hash_protocol")
    current_trainer_hash = sha256_source_file(
        PROJECT_ROOT
        / "experiments"
        / "train_probabilistic_pivot_direction_syncg.py"
    )
    if source_protocol is None:
        expected_trainer_hash = LEGACY_PARENT_TRAINER_SOURCE_SHA256
    elif source_protocol == SOURCE_TEXT_SHA256_PROTOCOL:
        expected_trainer_hash = current_trainer_hash
    else:
        raise ValueError("mechanism parent source-hash protocol is unsupported")
    if signature.get("trainer_source_sha256") != expected_trainer_hash:
        raise ValueError("mechanism parent trainer source mismatch")
    if summary.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise ValueError("mechanism parent best hash mismatch")
    if summary.get("last_checkpoint_sha256") != sha256_file(last_path):
        raise ValueError("mechanism parent last hash mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_epoch = int(summary.get("best_epoch", -1))
    best_health = _audit_parent_checkpoint(
        best,
        summary_signature=signature,
        expected_epoch=best_epoch,
        expect_optimizer_state=False,
    )
    last_health = _audit_parent_checkpoint(
        last,
        summary_signature=signature,
        expected_epoch=PARENT_EPOCH,
        expect_optimizer_state=True,
    )
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=seed,
    )
    if {sample.group_id for sample in train} & {
        sample.group_id for sample in validation
    }:
        raise ValueError("mechanism parent grouped split leaks groups")
    identity = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_sample_ids_sha256": sample_ids_hash(train),
        "validation_sample_ids_sha256": sample_ids_hash(validation),
    }
    if identity != {
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
    }:
        raise ValueError("mechanism parent grouped split identity drifted")
    return {
        "arm": name,
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": float(
            summary["best_validation_angle_mae_degrees"]
        ),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "group_overlap": 0,
        "last_optimizer_lr": float(
            last["optimizer_state"]["param_groups"][0]["lr"]
        ),
        "best_model_state_health": best_health,
        "last_model_state_health": last_health,
        "configuration": {
            "perspective_probability": arm.perspective_probability,
            "paired_supervision_weight": arm.paired_supervision_weight,
            "equivariance_weight": arm.equivariance_weight,
            "uncertainty_objective": arm.uncertainty_objective,
        },
    }


def build_plan(
    manifest: Path,
    *,
    require_complete: bool,
    require_main_convergence: bool = False,
) -> dict[str, Any]:
    manifest = formal_manifest_path(manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    runs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for name in PRIMARY_MECHANISM_ARMS:
        for seed in FORMAL_SEEDS:
            run_dir = mechanism_parent_dir(name, seed)
            if not (run_dir / "summary.json").is_file():
                entry = {
                    "arm": name,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "configuration": {
                        "perspective_probability": MECHANISM_ARMS[
                            name
                        ].perspective_probability,
                        "paired_supervision_weight": MECHANISM_ARMS[
                            name
                        ].paired_supervision_weight,
                        "equivariance_weight": MECHANISM_ARMS[
                            name
                        ].equivariance_weight,
                    },
                }
                missing.append(entry)
                continue
            runs.append(audit_mechanism_parent(name, seed, manifest))
    if require_complete and missing:
        raise FileNotFoundError(
            "missing frozen mechanism phase-1 parents: "
            + ", ".join(f"{row['arm']}:{row['seed']}" for row in missing)
        )
    main_gate = (
        audit_authoritative_pepd_v2_cohort()
        if require_main_convergence
        else {
            "checked": False,
            "required_authority": PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
            "required_before_gpu_mechanism_execution": True,
        }
    )
    return {
        "schema_version": 1,
        "protocol": PEPD_MECHANISM_PROTOCOL,
        "scope": "SyncG official train grouped validation only",
        "role": "mechanism ablation of retained PEPD; not algorithm selection",
        "primary_arms": list(PRIMARY_MECHANISM_ARMS),
        "alias_note": (
            "paired_supervision_only is the no-equivariance-loss/no-equiv arm; "
            "it is not an additional fourth geometry arm"
        ),
        "required_seeds": list(FORMAL_SEEDS),
        "complete_phase1_runs": runs,
        "missing_phase1_runs": missing,
        "phase1_complete": not missing,
        "main_convergence_gate": main_gate,
        "fixed_terminal_epoch": 60,
        "early_stopping": False,
        "controlled_validation_conditions": list(GROUPED_VAL_CONDITIONS),
        "uncertainty_arms": {
            name: {
                "status": arm.formal_priority,
                "objective": arm.uncertainty_objective,
                "interpretation": arm.interpretation,
            }
            for name, arm in MECHANISM_ARMS.items()
            if name in ("global_homoscedastic", "no_angular_nll")
        },
        "selection_policy": (
            "report all pre-declared contrasts; never select a replacement "
            "checkpoint from mechanism arms"
        ),
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "preflight": sha256_source_file(Path(__file__).resolve()),
        },
        "public_test_field_evaluation_authorized": False,
    }


def main() -> None:
    args = parse_args()
    report = build_plan(
        args.manifest,
        require_complete=args.require_complete,
        require_main_convergence=args.require_main_convergence,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
