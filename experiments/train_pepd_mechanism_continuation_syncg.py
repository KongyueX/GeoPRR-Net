"""Continue the two primary PEPD geometry-ablation arms to epoch 60.

``full`` is supplied by :mod:`experiments.train_pepd_convergence_syncg`; this
script is restricted to ``paired_supervision_only`` and
``no_projective_pair``.  All arms use identical seed-specific grouped splits,
phase-boundary RNG seeds, batch order, epoch budget, and constant phase-2 LR.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    FORMAL_BATCH_SIZE,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    FORMAL_WORKERS,
    PARENT_EPOCH,
    PEPD_MECHANISM_RUN_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    TERMINAL_EPOCH,
    continuation_seed,
    convergence_audit,
    formal_manifest_path,
    formal_parent_pin,
    mechanism_output_dir,
    mechanism_parent_dir,
    primary_mechanism_arm,
    sha256_file,
    validate_combined_history,
)
from experiments.preflight_pepd_mechanism import (
    audit_authoritative_pepd_v2_cohort,
    audit_mechanism_parent,
)
from experiments.probabilistic_pivot_direction import (
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
)
from experiments.train_pepd_convergence_syncg import (
    _acquire_lock,
    _atomic_copy,
    _atomic_json,
    _atomic_torch,
    _candidate_tuple,
    _checkpoint_epoch,
    _configure_determinism,
    _phase2_checkpoint,
    _release_lock,
    _restore_rng_state,
    _training_namespace,
)
from experiments.train_probabilistic_pivot_direction_syncg import (
    _train_epoch,
    _validate,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    seed_worker,
    set_random_seed,
    sha256_source_file,
)


FORMAL_ARMS = ("paired_supervision_only", "no_projective_pair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=FORMAL_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def mechanism_continuation_signature(
    *,
    arm: str,
    seed: int,
    parent_summary_sha256: str,
    parent_best_sha256: str,
    parent_last_sha256: str,
    authoritative_pepd_v2_gate: Mapping[str, Any],
) -> dict[str, Any]:
    specification = primary_mechanism_arm(arm)
    return {
        "protocol": PEPD_MECHANISM_RUN_PROTOCOL,
        "scope": "SyncG official train grouped validation only",
        "role": "mechanism ablation of retained PEPD; not algorithm selection",
        "arm": arm,
        "seed": int(seed),
        "continuation_seed": continuation_seed(seed),
        "parent_epoch": PARENT_EPOCH,
        "terminal_epoch": TERMINAL_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "continuation_learning_rate": CONTINUATION_LEARNING_RATE,
        "learning_rate_policy": "constant_at_parent_cosine_eta_min",
        "early_stopping": False,
        "perspective_probability": specification.perspective_probability,
        "paired_supervision_weight": specification.paired_supervision_weight,
        "equivariance_weight": specification.equivariance_weight,
        "uncertainty_objective": specification.uncertainty_objective,
        "parent_summary_sha256": parent_summary_sha256,
        "parent_best_sha256": parent_best_sha256,
        "parent_last_sha256": parent_last_sha256,
        "authoritative_pepd_v2_gate": dict(authoritative_pepd_v2_gate),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "model_source_sha256": FORMAL_MODEL_SOURCE_SHA256,
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "authoritative_v2_gate": sha256_source_file(
                PROJECT_ROOT / "experiments" / "preflight_pepd_mechanism.py"
            ),
            "trainer": sha256_source_file(Path(__file__).resolve()),
            "imported_epoch_trainer": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "train_probabilistic_pivot_direction_syncg.py"
            ),
        },
        "rng_boundary_policy": (
            "same deterministic seed-specific phase-boundary reset as full"
        ),
    }


def _summary(
    *,
    arm: str,
    seed: int,
    status: str,
    parent_audit: Mapping[str, Any],
    parent_signature: Mapping[str, Any],
    continuation_signature: Mapping[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    best_angle: float,
    best_nll: float,
    best_pivot: float,
    best_path: Path,
    last_path: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    audit = (
        convergence_audit(history, best_epoch=best_epoch)
        if len(history) == TERMINAL_EPOCH
        else None
    )
    return {
        "schema_version": 1,
        "protocol": PEPD_MECHANISM_RUN_PROTOCOL,
        "status": status,
        "scope": "SyncG official train grouped validation only",
        "role": "mechanism ablation of retained PEPD; not algorithm selection",
        "arm": arm,
        "seed": seed,
        "parent": dict(parent_audit),
        "parent_training_signature": dict(parent_signature),
        "continuation_signature": dict(continuation_signature),
        "history": history,
        "phase2_history": history[PARENT_EPOCH:],
        "best_epoch": int(best_epoch),
        "best_validation_angle_mae_degrees": float(best_angle),
        "best_validation_angular_calibration_nll": float(best_nll),
        "best_validation_pivot_error_fraction": float(best_pivot),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "convergence_audit": audit,
        "eligible_for_model_selection": False,
        "downstream_fadr_udsf_rebuild": False,
        "public_test_field_evaluation_authorized": False,
        "environment": dict(environment),
    }


def run(args: argparse.Namespace) -> Path:
    arm = str(args.arm)
    seed = int(args.seed)
    specification = primary_mechanism_arm(arm)
    if arm not in FORMAL_ARMS:
        raise ValueError(f"mechanism continuation arm must be one of {FORMAL_ARMS}")
    manifest = formal_manifest_path(args.manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    if sha256_source_file(
        PROJECT_ROOT / "experiments" / "probabilistic_pivot_direction.py"
    ) != FORMAL_MODEL_SOURCE_SHA256:
        raise ValueError("PEPD model source drifted")
    main_gate = audit_authoritative_pepd_v2_cohort()
    parent_audit = audit_mechanism_parent(arm, seed, manifest)
    parent_dir = mechanism_parent_dir(arm, seed)
    parent_summary = strict_json_load(parent_dir / "summary.json")
    parent_signature = parent_summary["signature"]
    if not math.isclose(
        float(parent_signature["perspective_probability"]),
        specification.perspective_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("mechanism perspective configuration drifted")
    if not math.isclose(
        float(parent_signature["equivariance_weight"]),
        specification.equivariance_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("mechanism equivariance configuration drifted")
    parent_last = torch.load(
        parent_dir / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    signature = mechanism_continuation_signature(
        arm=arm,
        seed=seed,
        parent_summary_sha256=parent_audit["summary_sha256"],
        parent_best_sha256=parent_audit["best_checkpoint_sha256"],
        parent_last_sha256=parent_audit["last_checkpoint_sha256"],
        authoritative_pepd_v2_gate=main_gate,
    )

    output_dir = mechanism_output_dir(arm, seed)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    summary_path = output_dir / "summary.json"
    lock_path = output_dir / ".training.lock"
    if output_dir.exists() and not args.resume:
        entries = [path for path in output_dir.iterdir() if path.name != ".training.lock"]
        if entries:
            raise FileExistsError(
                f"{output_dir} is not empty; formal mechanism runs never overwrite"
            )
    if args.resume and not last_path.is_file():
        raise FileNotFoundError("resume requires mechanism phase-2 last.pt")
    descriptor = _acquire_lock(lock_path)
    try:
        device = torch.device(args.device)
        _configure_determinism(seed, device)
        samples, manifest_protocol = load_syncg_manifest(
            manifest,
            expected_split="train",
        )
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=float(parent_signature["validation_fraction"]),
            seed=seed,
        )
        pin = formal_parent_pin(seed)
        if (
            len(train_samples) != pin.train_samples
            or len(validation_samples) != pin.validation_samples
            or sample_ids_hash(train_samples) != pin.train_ids_sha256
            or sample_ids_hash(validation_samples) != pin.validation_ids_sha256
        ):
            raise ValueError("mechanism grouped split identity drifted")
        training_args = _training_namespace(parent_signature)
        common_dataset = {
            name: getattr(training_args, name)
            for name in (
                "image_size",
                "heatmap_size",
                "expansion",
                "scale_factor",
                "rotation_factor",
                "translation_factor",
                "heatmap_sigma",
                "perspective_probability",
                "max_perspective_degrees",
                "max_blur_sigma",
            )
        }
        train_dataset = SyncGProbabilisticDirectionDataset(
            train_samples,
            training=True,
            **common_dataset,
        )
        validation_dataset = SyncGProbabilisticDirectionDataset(
            validation_samples,
            training=False,
            **common_dataset,
        )
        generator = torch.Generator().manual_seed(continuation_seed(seed))
        train_loader = DataLoader(
            train_dataset,
            batch_size=FORMAL_BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=FORMAL_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=FORMAL_BATCH_SIZE,
            shuffle=False,
            num_workers=FORMAL_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=False,
        )
        model = build_probabilistic_pivot_direction_model(
            angle_bins=int(parent_signature["angle_bins"]),
            imagenet_pretrained=False,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(parent_signature["learning_rate"]),
            weight_decay=float(parent_signature["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=True,
            init_scale=float(parent_signature["grad_scaler_initial_scale"]),
        )
        if args.resume:
            checkpoint = torch.load(
                last_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("continuation_signature") != signature:
                raise ValueError("mechanism resume signature mismatch")
            epoch = int(checkpoint.get("epoch", -1))
            if not PARENT_EPOCH < epoch < TERMINAL_EPOCH:
                raise ValueError("mechanism resume epoch is invalid")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            history = list(checkpoint["history"])
            best_angle = float(checkpoint["best_angle"])
            best_nll = float(checkpoint["best_calibration_nll"])
            best_pivot = float(checkpoint["best_pivot_error"])
            best_epoch = int(checkpoint["best_epoch"])
            _restore_rng_state(checkpoint, generator)
            start_epoch = epoch + 1
            if best_epoch == epoch and _checkpoint_epoch(best_path) != epoch:
                _atomic_torch(best_path, checkpoint)
        else:
            model.load_state_dict(parent_last["model_state"])
            optimizer.load_state_dict(parent_last["optimizer_state"])
            scaler.load_state_dict(parent_last["scaler_state"])
            set_random_seed(continuation_seed(seed))
            generator.manual_seed(continuation_seed(seed))
            history = list(parent_last["history"])
            best_angle = float(parent_last["best_angle"])
            best_nll = float(parent_last["best_calibration_nll"])
            best_pivot = float(parent_last["best_pivot_error"])
            best_epoch = int(parent_last["best_epoch"])
            start_epoch = PARENT_EPOCH + 1
            _atomic_copy(parent_dir / "best.pt", best_path)
        if len(history) != start_epoch - 1:
            raise ValueError("mechanism history does not match start epoch")
        for group in optimizer.param_groups:
            if not math.isclose(
                float(group["lr"]),
                CONTINUATION_LEARNING_RATE,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("mechanism parent optimizer LR drifted")
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "pythonhashseed": os.environ["PYTHONHASHSEED"],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(start_epoch, TERMINAL_EPOCH + 1):
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device=device,
                amp_enabled=True,
                args=training_args,
                epoch=epoch,
            )
            validation_metrics = _validate(
                model,
                validation_loader,
                device=device,
                amp_enabled=True,
                args=training_args,
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": CONTINUATION_LEARNING_RATE,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            candidate = _candidate_tuple(validation_metrics)
            incumbent = (best_angle, best_nll, best_pivot)
            improved = candidate < incumbent
            if improved:
                best_angle, best_nll, best_pivot = candidate
                best_epoch = epoch
            checkpoint = _phase2_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                combined_history=history,
                parent_signature=parent_signature,
                continuation_signature=signature,
                best_angle=best_angle,
                best_calibration_nll=best_nll,
                best_pivot_error=best_pivot,
                best_epoch=best_epoch,
                generator=generator,
            )
            _atomic_torch(last_path, checkpoint)
            if improved:
                _atomic_torch(best_path, checkpoint)
            report = _summary(
                arm=arm,
                seed=seed,
                status="complete" if epoch == TERMINAL_EPOCH else "running",
                parent_audit=parent_audit,
                parent_signature=parent_signature,
                continuation_signature=signature,
                history=history,
                best_epoch=best_epoch,
                best_angle=best_angle,
                best_nll=best_nll,
                best_pivot=best_pivot,
                best_path=best_path,
                last_path=last_path,
                environment=environment,
            )
            report["manifest_protocol"] = manifest_protocol
            _atomic_json(summary_path, report)
            print(
                f"arm={arm} seed={seed} epoch={epoch}/{TERMINAL_EPOCH} "
                f"val_angle={candidate[0]:.6f}deg "
                f"best={best_angle:.6f}deg@{best_epoch}",
                flush=True,
            )
        validate_combined_history(
            history,
            train_samples=pin.train_samples,
            validation_samples=pin.validation_samples,
        )
        print(summary_path)
        return summary_path
    finally:
        _release_lock(lock_path, descriptor)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
