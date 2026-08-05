"""Verify one completed PEPD epoch-60 convergence continuation."""
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
    PARENT_EPOCH,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    TERMINAL_EPOCH,
    build_continuation_signature,
    convergence_audit,
    formal_manifest_path,
    formal_output_dir,
    formal_parent_dir,
    formal_parent_pin,
    sha256_file,
    validate_combined_history,
)
from experiments.preflight_pepd_convergence import (
    _state_health,
    audit_parent,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _finite_state(name: str, state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} is missing")
    for key, value in state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name}.{key} contains non-finite values")


def _validate_phase2_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_epoch: int,
    parent_signature: Mapping[str, Any],
    continuation_signature: Mapping[str, Any],
) -> dict[str, int]:
    if checkpoint.get("protocol") != PEPD_TRAINING_PROTOCOL:
        raise ValueError("phase-2 checkpoint model protocol mismatch")
    if checkpoint.get("signature") != parent_signature:
        raise ValueError("phase-2 checkpoint parent signature mismatch")
    if checkpoint.get("continuation_signature") != continuation_signature:
        raise ValueError("phase-2 checkpoint continuation signature mismatch")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise ValueError("phase-2 checkpoint epoch mismatch")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != expected_epoch:
        raise ValueError("phase-2 checkpoint history mismatch")
    if "scheduler_state" in checkpoint:
        raise ValueError("phase-2 checkpoint unexpectedly contains a scheduler")
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, Mapping) or not optimizer.get("state"):
        raise ValueError("phase-2 checkpoint has no AdamW state")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("phase-2 optimizer param-group structure mismatch")
    if not math.isclose(
        float(groups[0].get("lr", math.nan)),
        CONTINUATION_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("phase-2 optimizer LR mismatch")
    scaler = checkpoint.get("scaler_state")
    if not isinstance(scaler, Mapping) or float(scaler.get("scale", 0.0)) <= 0:
        raise ValueError("phase-2 checkpoint has no valid GradScaler state")
    rng_names = (
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "loader_generator_state",
    )
    if any(name not in checkpoint for name in rng_names):
        raise ValueError("phase-2 checkpoint RNG state is incomplete")
    _finite_state("optimizer_state", optimizer.get("state") or {})
    return _state_health(checkpoint.get("model_state") or {})


def build_verification(seed: int, manifest: Path) -> dict[str, Any]:
    pin = formal_parent_pin(seed)
    manifest = formal_manifest_path(manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train manifest protocol hash drifted")
    audit_parent(seed, manifest)

    run_dir = formal_output_dir(seed)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    summary = _load_json(summary_path)
    if summary.get("protocol") != PEPD_CONTINUATION_PROTOCOL:
        raise ValueError("continuation summary protocol mismatch")
    if summary.get("status") != "complete":
        raise ValueError("continuation summary is not complete")
    if int(summary.get("seed", -1)) != seed:
        raise ValueError("continuation summary seed mismatch")
    if summary.get("scope") != "SyncG official train grouped validation only":
        raise ValueError("continuation summary scope mismatch")
    parent_signature = summary.get("parent_training_signature")
    if not isinstance(parent_signature, Mapping):
        raise ValueError("continuation summary has no parent signature")
    expected_continuation_signature = build_continuation_signature(
        seed=seed,
        parent_summary_sha256=pin.summary_sha256,
        parent_best_sha256=pin.best_sha256,
        parent_last_sha256=pin.last_sha256,
        continuation_source_sha256=sha256_source_file(
            PROJECT_ROOT / "experiments" / "train_pepd_convergence_syncg.py"
        ),
        imported_trainer_source_sha256=sha256_source_file(
            PROJECT_ROOT
            / "experiments"
            / "train_probabilistic_pivot_direction_syncg.py"
        ),
        model_source_sha256=FORMAL_MODEL_SOURCE_SHA256,
    )
    if summary.get("continuation_signature") != expected_continuation_signature:
        raise ValueError("continuation signature or source identity drifted")
    for path in (best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    best_hash = sha256_file(best_path)
    last_hash = sha256_file(last_path)
    if summary.get("best_checkpoint_sha256") != best_hash:
        raise ValueError("continuation best checkpoint hash mismatch")
    if summary.get("last_checkpoint_sha256") != last_hash:
        raise ValueError("continuation last checkpoint hash mismatch")

    history = summary.get("history")
    if not isinstance(history, list):
        raise ValueError("continuation summary has no history")
    validate_combined_history(
        history,
        train_samples=pin.train_samples,
        validation_samples=pin.validation_samples,
    )
    phase2 = summary.get("phase2_history")
    if phase2 != history[PARENT_EPOCH:]:
        raise ValueError("continuation phase-2 history slice mismatch")
    best_epoch = int(summary.get("best_epoch", -1))
    if not 1 <= best_epoch <= TERMINAL_EPOCH:
        raise ValueError("continuation best epoch is invalid")
    candidate_tuples = [
        (
            float(record["validation"]["angle_mae_degrees"]),
            float(record["validation"]["angular_calibration_nll"]),
            float(record["validation"]["pivot_mean_error_fraction"]),
        )
        for record in history
    ]
    expected_best_index = min(range(len(candidate_tuples)), key=candidate_tuples.__getitem__)
    expected_best_epoch = expected_best_index + 1
    if best_epoch != expected_best_epoch:
        raise ValueError("continuation best checkpoint selection is not lexicographic")
    expected_best = candidate_tuples[expected_best_index]
    summary_best = (
        float(summary["best_validation_angle_mae_degrees"]),
        float(summary["best_validation_angular_calibration_nll"]),
        float(summary["best_validation_pivot_error_fraction"]),
    )
    if summary_best != expected_best:
        raise ValueError("continuation best metrics do not match history")

    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last_health = _validate_phase2_checkpoint(
        last,
        expected_epoch=TERMINAL_EPOCH,
        parent_signature=parent_signature,
        continuation_signature=expected_continuation_signature,
    )
    if last.get("history") != history:
        raise ValueError("last checkpoint and summary histories differ")
    if int(last.get("best_epoch", -1)) != best_epoch:
        raise ValueError("last checkpoint best epoch mismatch")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best_hash == pin.best_sha256:
        if int(best.get("epoch", -1)) != pin.best_epoch:
            raise ValueError("unchanged parent best checkpoint epoch mismatch")
        if best.get("signature") != parent_signature:
            raise ValueError("unchanged parent best signature mismatch")
        best_health = _state_health(best.get("model_state") or {})
        selected_origin = "frozen_parent_best"
    else:
        best_health = _validate_phase2_checkpoint(
            best,
            expected_epoch=best_epoch,
            parent_signature=parent_signature,
            continuation_signature=expected_continuation_signature,
        )
        selected_origin = "continuation_epoch"
    changed = best_hash != pin.best_sha256
    if bool(summary.get("checkpoint_changed_from_frozen_parent")) != changed:
        raise ValueError("checkpoint-change flag mismatch")
    invalidation = summary.get("downstream_invalidation") or {}
    for name in (
        "fadr_rebuild_required",
        "udsf_rebuild_required",
        "all_dependent_public_outputs_rebuild_required",
    ):
        if bool(invalidation.get(name)) != changed:
            raise ValueError(f"downstream invalidation flag mismatch: {name}")

    expected_audit = convergence_audit(history, best_epoch=best_epoch)
    if summary.get("convergence_audit") != expected_audit:
        raise ValueError("stored convergence audit does not recompute exactly")
    if summary.get("public_or_field_evaluation_authorized") is not False:
        raise ValueError("single-seed run must not authorize public/field evaluation")

    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(parent_signature["validation_fraction"]),
        seed=seed,
    )
    identity = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_sample_ids_sha256": sample_ids_hash(train),
        "validation_sample_ids_sha256": sample_ids_hash(validation),
    }
    expected_identity = {
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
    }
    if identity != expected_identity:
        raise ValueError("recomputed grouped split identity drifted")

    return {
        "schema_version": 1,
        "protocol": PEPD_RUN_VERIFICATION_PROTOCOL,
        "verified": True,
        "converged": bool(expected_audit["converged"]),
        "seed": seed,
        "scope": "SyncG official train grouped validation only",
        "run_dir": str(run_dir),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint_sha256": last_hash,
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": expected_best[0],
        "selected_checkpoint_origin": selected_origin,
        "checkpoint_changed_from_frozen_parent": changed,
        "fadr_udsf_rebuild_required": changed,
        "convergence_audit": expected_audit,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "group_overlap": 0,
        "best_model_state_health": best_health,
        "last_model_state_health": last_health,
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "trainer": sha256_source_file(
                PROJECT_ROOT / "experiments" / "train_pepd_convergence_syncg.py"
            ),
            "verifier": sha256_source_file(Path(__file__).resolve()),
            "model": FORMAL_MODEL_SOURCE_SHA256,
            "imported_epoch_trainer": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "train_probabilistic_pivot_direction_syncg.py"
            ),
        },
        "public_or_field_evaluation_authorized": False,
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
            raise FileExistsError(
                f"{path} exists with different verification content"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    verification = build_verification(args.seed, args.manifest)
    output = formal_output_dir(args.seed) / "verification.json"
    _write_or_validate(output, verification)
    print(
        json.dumps(
            verification,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
