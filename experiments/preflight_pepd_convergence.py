"""Read-only parent/cohort preflight for the frozen PEPD continuation.

The preflight opens only the pinned SyncG *train* manifest and the three
already-existing PEPD training runs.  It never decodes an image and never
touches SyncG test, RPM, Pointer, field, sealed, or confirmatory assets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    LEGACY_PARENT_TRAINER_SOURCE_SHA256,
    PARENT_EPOCH,
    PEPD_PREFLIGHT_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    formal_manifest_path,
    formal_parent_dir,
    formal_parent_pin,
    sha256_file,
    validate_parent_summary,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional no-clobber JSON report. Omit for a genuinely read-only "
            "preflight."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _state_health(state: Mapping[str, Any]) -> dict[str, int]:
    tensor_count = 0
    parameter_count = 0
    nonfinite_count = 0
    for value in state.values():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        parameter_count += int(value.numel())
        if value.is_floating_point() or value.is_complex():
            nonfinite_count += int((~torch.isfinite(value)).sum().item())
    if tensor_count <= 0 or parameter_count <= 0:
        raise ValueError("checkpoint model_state is empty")
    if nonfinite_count:
        raise ValueError("checkpoint model_state contains non-finite values")
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "nonfinite_count": nonfinite_count,
    }


def _audit_parent_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    summary_signature: Mapping[str, Any],
    expected_epoch: int,
    expect_optimizer_state: bool,
) -> dict[str, Any]:
    if checkpoint.get("protocol") != PEPD_TRAINING_PROTOCOL:
        raise ValueError("parent checkpoint has the wrong protocol")
    if checkpoint.get("signature") != summary_signature:
        raise ValueError("parent checkpoint signature does not match summary")
    if int(checkpoint.get("epoch", -1)) != int(expected_epoch):
        raise ValueError("parent checkpoint epoch mismatch")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != expected_epoch:
        raise ValueError("parent checkpoint history is incomplete")
    health = _state_health(checkpoint.get("model_state") or {})
    if expect_optimizer_state:
        optimizer = checkpoint.get("optimizer_state")
        scheduler = checkpoint.get("scheduler_state")
        scaler = checkpoint.get("scaler_state")
        if not isinstance(optimizer, Mapping) or not optimizer.get("state"):
            raise ValueError("parent last checkpoint has no AdamW state")
        groups = optimizer.get("param_groups")
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("parent AdamW param-group structure changed")
        if not math.isclose(
            float(groups[0].get("lr", math.nan)),
            CONTINUATION_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("parent optimizer is not at the frozen terminal LR")
        if not isinstance(scheduler, Mapping):
            raise ValueError("parent last checkpoint has no scheduler state")
        scheduler_exact = {
            "T_max": PARENT_EPOCH,
            "last_epoch": PARENT_EPOCH,
        }
        for name, expected in scheduler_exact.items():
            if int(scheduler.get(name, -1)) != expected:
                raise ValueError(f"parent scheduler {name} mismatch")
        if not math.isclose(
            float(scheduler.get("eta_min", math.nan)),
            CONTINUATION_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("parent scheduler eta_min mismatch")
        last_lr = scheduler.get("_last_lr")
        if not isinstance(last_lr, list) or len(last_lr) != 1 or not math.isclose(
            float(last_lr[0]),
            CONTINUATION_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("parent scheduler did not finish at eta_min")
        if not isinstance(scaler, Mapping) or float(scaler.get("scale", 0.0)) <= 0:
            raise ValueError("parent last checkpoint has no valid GradScaler state")
    return health


def audit_parent(seed: int, manifest: Path) -> dict[str, Any]:
    pin = formal_parent_pin(seed)
    run_dir = formal_parent_dir(seed)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    exact_hashes = {
        summary_path: pin.summary_sha256,
        best_path: pin.best_sha256,
        last_path: pin.last_sha256,
    }
    for path, expected in exact_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise ValueError(f"frozen parent artifact hash drifted: {path}")
    summary = _load_json(summary_path)
    validate_parent_summary(summary, seed=seed)
    if summary.get("best_checkpoint_sha256") != pin.best_sha256:
        raise ValueError("summary best checkpoint hash mismatch")
    if summary.get("last_checkpoint_sha256") != pin.last_sha256:
        raise ValueError("summary last checkpoint hash mismatch")
    signature = summary["signature"]
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_health = _audit_parent_checkpoint(
        best,
        summary_signature=signature,
        expected_epoch=pin.best_epoch,
        expect_optimizer_state=False,
    )
    last_health = _audit_parent_checkpoint(
        last,
        summary_signature=signature,
        expected_epoch=PARENT_EPOCH,
        expect_optimizer_state=True,
    )
    if int(best.get("best_epoch", -1)) != pin.best_epoch:
        raise ValueError("parent best checkpoint selection metadata mismatch")
    if int(last.get("best_epoch", -1)) != pin.best_epoch:
        raise ValueError("parent last checkpoint selection metadata mismatch")

    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=seed,
    )
    train_groups = {sample.group_id for sample in train}
    validation_groups = {sample.group_id for sample in validation}
    if train_groups & validation_groups:
        raise ValueError("grouped train/validation split leaks groups")
    identities = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_sample_ids_sha256": sample_ids_hash(train),
        "validation_sample_ids_sha256": sample_ids_hash(validation),
    }
    expected_identities = {
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
    }
    if identities != expected_identities:
        raise ValueError("recomputed grouped split does not match frozen parent")
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_sha256": pin.summary_sha256,
        "best_checkpoint_sha256": pin.best_sha256,
        "last_checkpoint_sha256": pin.last_sha256,
        "best_epoch": pin.best_epoch,
        "best_validation_angle_mae_degrees": pin.best_angle_mae_degrees,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "group_overlap": 0,
        "last_optimizer_states": len(last["optimizer_state"]["state"]),
        "last_optimizer_lr": float(
            last["optimizer_state"]["param_groups"][0]["lr"]
        ),
        "last_scheduler_epoch": int(last["scheduler_state"]["last_epoch"]),
        "last_scaler_scale": float(last["scaler_state"]["scale"]),
        "legacy_rng_state_present": all(
            name in last
            for name in (
                "python_rng_state",
                "numpy_rng_state",
                "torch_rng_state",
                "loader_generator_state",
            )
        ),
        "best_model_state_health": best_health,
        "last_model_state_health": last_health,
    }


def build_preflight_report(manifest: Path) -> dict[str, Any]:
    manifest = formal_manifest_path(manifest)
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train manifest protocol hash drifted")
    current_model_hash = sha256_source_file(
        PROJECT_ROOT / "experiments" / "probabilistic_pivot_direction.py"
    )
    if current_model_hash != FORMAL_MODEL_SOURCE_SHA256:
        raise ValueError("PEPD model source drifted from the parent checkpoints")
    parents = [audit_parent(seed, manifest) for seed in FORMAL_SEEDS]
    if any(parent["legacy_rng_state_present"] for parent in parents):
        raise ValueError(
            "unexpected legacy RNG state appeared; phase-boundary policy must "
            "be reviewed before training"
        )
    return {
        "schema_version": 1,
        "protocol": PEPD_PREFLIGHT_PROTOCOL,
        "authorized": True,
        "scope": "SyncG official train grouped validation only",
        "forbidden_scopes": [
            "SyncG test",
            "RPM",
            "Pointer",
            "field",
            "sealed",
            "confirmatory",
        ],
        "manifest": str(manifest),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "model_source_sha256": current_model_hash,
        "strict_json_source_sha256": strict_json_source_sha256(),
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "preflight": sha256_source_file(Path(__file__).resolve()),
        },
        "legacy_parent_trainer_source_sha256": (
            LEGACY_PARENT_TRAINER_SOURCE_SHA256
        ),
        "parents": parents,
        "continuation_boundary": {
            "parent_epoch": PARENT_EPOCH,
            "parent_optimizer_and_scaler_state_preserved": True,
            "parent_scheduler_reaches_eta_min": True,
            "legacy_rng_state_available": False,
            "resolution": (
                "deterministic pre-declared phase-boundary RNG reset; do not "
                "claim uninterrupted bitwise continuation"
            ),
        },
        "gpu_training_started": False,
    }


def _write_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    report = build_preflight_report(args.manifest)
    if args.output is not None:
        _write_no_clobber(args.output, report)
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
