"""Continue the three frozen PEPD seeds to the fixed epoch-60 boundary.

Formal execution is CUDA-only, uses no early stopping, and is restricted to
the pinned SyncG train manifest with the original grouped validation split.
The epoch-30 AdamW and GradScaler states are restored exactly.  The completed
cosine scheduler is retired at its frozen eta_min (3e-6), which becomes the
constant phase-2 learning rate.

This script must not be used as an algorithm-search entry point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.strict_json import strict_json_load

from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    FORMAL_BATCH_SIZE,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    FORMAL_WORKERS,
    PARENT_EPOCH,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    TERMINAL_EPOCH,
    build_continuation_signature,
    continuation_seed,
    convergence_audit,
    formal_manifest_path,
    formal_output_dir,
    formal_parent_dir,
    formal_parent_pin,
    sha256_file,
    validate_combined_history,
)
from experiments.preflight_pepd_convergence import audit_parent
from experiments.probabilistic_pivot_direction import (
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
)
from experiments.train_probabilistic_pivot_direction_syncg import (
    _train_epoch,
    _validate,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    seed_worker,
    set_random_seed,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"formal continuation lock already exists: {path}; inspect the "
            "previous process before removing a stale lock"
        ) from exc
    os.write(
        descriptor,
        json.dumps(
            {"pid": os.getpid()},
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8"),
    )
    os.fsync(descriptor)
    return descriptor


def _release_lock(path: Path, descriptor: int) -> None:
    os.close(descriptor)
    path.unlink(missing_ok=True)


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "loader_generator_state": generator.get_state(),
    }


def _restore_rng_state(
    checkpoint: Mapping[str, Any],
    generator: torch.Generator,
) -> None:
    required = (
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "loader_generator_state",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "phase-2 resume checkpoint is missing RNG state: "
            + ", ".join(missing)
        )
    random.setstate(checkpoint["python_rng_state"])
    numpy_state = checkpoint["numpy_rng_state"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise ValueError("invalid NumPy RNG state in phase-2 checkpoint")
    np.random.set_state(numpy_state)
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint["cuda_rng_state_all"]
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("phase-2 checkpoint CUDA RNG topology mismatch")
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(checkpoint["loader_generator_state"])


def _training_namespace(parent_signature: Mapping[str, Any]) -> argparse.Namespace:
    names = (
        "image_size",
        "heatmap_size",
        "angle_bins",
        "pivot_loss_weight",
        "bin_loss_weight",
        "vector_loss_weight",
        "paired_supervision_weight",
        "equivariance_weight",
        "equivariance_pivot_weight",
        "soft_target_sigma_bins",
        "expansion",
        "scale_factor",
        "rotation_factor",
        "translation_factor",
        "heatmap_sigma",
        "perspective_probability",
        "max_perspective_degrees",
        "max_blur_sigma",
    )
    values = {name: parent_signature[name] for name in names}
    values.update(
        {
            "batch_size": FORMAL_BATCH_SIZE,
            "workers": FORMAL_WORKERS,
            "epochs": TERMINAL_EPOCH,
            "learning_rate": CONTINUATION_LEARNING_RATE,
        }
    )
    return argparse.Namespace(**values)


def _configure_determinism(seed: int, device: torch.device) -> None:
    expected_hash_seed = str(seed)
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise RuntimeError(
            f"formal run requires PYTHONHASHSEED={expected_hash_seed} to be set "
            "before Python starts"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal run requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )
    if device.type != "cuda":
        raise RuntimeError("formal PEPD continuation is CUDA-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_random_seed(continuation_seed(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _phase2_checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    combined_history: list[dict[str, Any]],
    parent_signature: Mapping[str, Any],
    continuation_signature: Mapping[str, Any],
    best_angle: float,
    best_calibration_nll: float,
    best_pivot_error: float,
    best_epoch: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        # Preserve the original model protocol/signature for downstream model
        # construction, while binding the new training phase separately.
        "protocol": PEPD_TRAINING_PROTOCOL,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": combined_history,
        "signature": dict(parent_signature),
        "continuation_signature": dict(continuation_signature),
        "best_angle": float(best_angle),
        "best_calibration_nll": float(best_calibration_nll),
        "best_pivot_error": float(best_pivot_error),
        "best_epoch": int(best_epoch),
    } | _rng_state(generator)


def _candidate_tuple(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["angle_mae_degrees"]),
        float(metrics["angular_calibration_nll"]),
        float(metrics["pivot_mean_error_fraction"]),
    )


def _checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        return -1
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1))


def _build_summary(
    *,
    seed: int,
    status: str,
    parent_summary: Mapping[str, Any],
    continuation_signature: Mapping[str, Any],
    combined_history: list[dict[str, Any]],
    best_epoch: int,
    best_angle: float,
    best_calibration_nll: float,
    best_pivot_error: float,
    best_path: Path,
    last_path: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    pin = formal_parent_pin(seed)
    best_hash = sha256_file(best_path)
    checkpoint_changed = best_hash != pin.best_sha256
    audit = (
        convergence_audit(combined_history, best_epoch=best_epoch)
        if len(combined_history) == TERMINAL_EPOCH
        else None
    )
    return {
        "schema_version": 1,
        "protocol": PEPD_CONTINUATION_PROTOCOL,
        "status": status,
        "scope": "SyncG official train grouped validation only",
        "seed": seed,
        "parent": {
            "run_dir": pin.run_dir,
            "summary_sha256": pin.summary_sha256,
            "best_checkpoint_sha256": pin.best_sha256,
            "last_checkpoint_sha256": pin.last_sha256,
            "best_epoch": pin.best_epoch,
            "best_validation_angle_mae_degrees": pin.best_angle_mae_degrees,
        },
        "parent_training_signature": parent_summary["signature"],
        "continuation_signature": dict(continuation_signature),
        "best_epoch": int(best_epoch),
        "best_validation_angle_mae_degrees": float(best_angle),
        "best_validation_angular_calibration_nll": float(best_calibration_nll),
        "best_validation_pivot_error_fraction": float(best_pivot_error),
        "history": combined_history,
        "phase2_history": combined_history[PARENT_EPOCH:],
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "checkpoint_changed_from_frozen_parent": checkpoint_changed,
        "downstream_invalidation": {
            "fadr_rebuild_required": checkpoint_changed,
            "udsf_rebuild_required": checkpoint_changed,
            "all_dependent_public_outputs_rebuild_required": checkpoint_changed,
            "reason": (
                "FADR/UDSF were fitted against the frozen PEPD checkpoint; "
                "a changed checkpoint invalidates their inputs"
            ),
        },
        "convergence_audit": audit,
        "public_or_field_evaluation_authorized": False,
        "environment": dict(environment),
    }


def run(args: argparse.Namespace) -> Path:
    seed = int(args.seed)
    pin = formal_parent_pin(seed)
    manifest = formal_manifest_path(args.manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train manifest protocol hash drifted")
    parent_audit = audit_parent(seed, manifest)
    if parent_audit["legacy_rng_state_present"]:
        raise ValueError("unexpected parent RNG state invalidates the boundary protocol")

    output_dir = formal_output_dir(seed)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    summary_path = output_dir / "summary.json"
    lock_path = output_dir / ".training.lock"
    if output_dir.exists() and not args.resume:
        entries = [path for path in output_dir.iterdir() if path.name != ".training.lock"]
        if entries:
            raise FileExistsError(
                f"{output_dir} is not empty; formal runs never overwrite"
            )
    if args.resume and not last_path.is_file():
        raise FileNotFoundError("resume requires the phase-2 last.pt checkpoint")
    descriptor = _acquire_lock(lock_path)
    try:
        device = torch.device(args.device)
        _configure_determinism(seed, device)

        parent_dir = formal_parent_dir(seed)
        parent_summary = strict_json_load(parent_dir / "summary.json")
        parent_last = torch.load(
            parent_dir / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        parent_signature = parent_summary["signature"]
        if sha256_source_file(
            PROJECT_ROOT / "experiments" / "probabilistic_pivot_direction.py"
        ) != FORMAL_MODEL_SOURCE_SHA256:
            raise ValueError("PEPD model source drifted")
        continuation_signature = build_continuation_signature(
            seed=seed,
            parent_summary_sha256=pin.summary_sha256,
            parent_best_sha256=pin.best_sha256,
            parent_last_sha256=pin.last_sha256,
            continuation_source_sha256=sha256_source_file(Path(__file__).resolve()),
            imported_trainer_source_sha256=sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "train_probabilistic_pivot_direction_syncg.py"
            ),
            model_source_sha256=FORMAL_MODEL_SOURCE_SHA256,
        )

        samples, manifest_protocol = load_syncg_manifest(
            manifest,
            expected_split="train",
        )
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=float(parent_signature["validation_fraction"]),
            seed=seed,
        )
        if len(train_samples) != pin.train_samples:
            raise ValueError("formal train sample count drifted")
        if len(validation_samples) != pin.validation_samples:
            raise ValueError("formal validation sample count drifted")
        training_args = _training_namespace(parent_signature)
        common_dataset = {
            "image_size": training_args.image_size,
            "heatmap_size": training_args.heatmap_size,
            "expansion": training_args.expansion,
            "scale_factor": training_args.scale_factor,
            "rotation_factor": training_args.rotation_factor,
            "translation_factor": training_args.translation_factor,
            "heatmap_sigma": training_args.heatmap_sigma,
            "perspective_probability": training_args.perspective_probability,
            "max_perspective_degrees": training_args.max_perspective_degrees,
            "max_blur_sigma": training_args.max_blur_sigma,
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
            if checkpoint.get("continuation_signature") != continuation_signature:
                raise ValueError("phase-2 resume signature mismatch")
            if checkpoint.get("signature") != parent_signature:
                raise ValueError("phase-2 parent training signature mismatch")
            epoch = int(checkpoint.get("epoch", -1))
            if not PARENT_EPOCH < epoch < TERMINAL_EPOCH:
                raise ValueError("phase-2 resume checkpoint epoch is invalid")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            combined_history = list(checkpoint["history"])
            best_angle = float(checkpoint["best_angle"])
            best_calibration_nll = float(checkpoint["best_calibration_nll"])
            best_pivot_error = float(checkpoint["best_pivot_error"])
            best_epoch = int(checkpoint["best_epoch"])
            _restore_rng_state(checkpoint, generator)
            start_epoch = epoch + 1
            # Recover the only meaningful atomicity gap: last.pt may have been
            # committed immediately before a newly improved best.pt.
            if best_epoch == epoch and _checkpoint_epoch(best_path) != epoch:
                _atomic_torch(best_path, checkpoint)
        else:
            model.load_state_dict(parent_last["model_state"])
            optimizer.load_state_dict(parent_last["optimizer_state"])
            scaler.load_state_dict(parent_last["scaler_state"])
            loaded_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isclose(
                loaded_lr,
                CONTINUATION_LEARNING_RATE,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("loaded parent AdamW state has the wrong LR")
            # Reset only the RNG streams at the explicitly declared phase
            # boundary.  Model, AdamW moments, and scaler remain untouched.
            set_random_seed(continuation_seed(seed))
            generator.manual_seed(continuation_seed(seed))
            combined_history = list(parent_last["history"])
            best_angle = float(parent_last["best_angle"])
            best_calibration_nll = float(parent_last["best_calibration_nll"])
            best_pivot_error = float(parent_last["best_pivot_error"])
            best_epoch = int(parent_last["best_epoch"])
            start_epoch = PARENT_EPOCH + 1
            _atomic_copy(parent_dir / "best.pt", best_path)

        if len(combined_history) != start_epoch - 1:
            raise ValueError("phase-2 history does not match resume epoch")
        for group in optimizer.param_groups:
            if not math.isclose(
                float(group["lr"]),
                CONTINUATION_LEARNING_RATE,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("phase-2 optimizer learning rate drifted")

        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
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
            combined_history.append(
                {
                    "epoch": epoch,
                    "learning_rate": CONTINUATION_LEARNING_RATE,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            candidate = _candidate_tuple(validation_metrics)
            incumbent = (
                best_angle,
                best_calibration_nll,
                best_pivot_error,
            )
            improved = candidate < incumbent
            if improved:
                (
                    best_angle,
                    best_calibration_nll,
                    best_pivot_error,
                ) = candidate
                best_epoch = epoch
            checkpoint = _phase2_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                combined_history=combined_history,
                parent_signature=parent_signature,
                continuation_signature=continuation_signature,
                best_angle=best_angle,
                best_calibration_nll=best_calibration_nll,
                best_pivot_error=best_pivot_error,
                best_epoch=best_epoch,
                generator=generator,
            )
            _atomic_torch(last_path, checkpoint)
            if improved:
                _atomic_torch(best_path, checkpoint)
            status = "complete" if epoch == TERMINAL_EPOCH else "running"
            summary = _build_summary(
                seed=seed,
                status=status,
                parent_summary=parent_summary,
                continuation_signature=continuation_signature,
                combined_history=combined_history,
                best_epoch=best_epoch,
                best_angle=best_angle,
                best_calibration_nll=best_calibration_nll,
                best_pivot_error=best_pivot_error,
                best_path=best_path,
                last_path=last_path,
                environment=environment,
            )
            summary["manifest_protocol"] = manifest_protocol
            _atomic_json(summary_path, summary)
            print(
                f"seed={seed} epoch={epoch}/{TERMINAL_EPOCH} "
                f"lr={CONTINUATION_LEARNING_RATE:.3e} "
                f"val_angle={candidate[0]:.6f}deg "
                f"best={best_angle:.6f}deg@{best_epoch}",
                flush=True,
            )

        validate_combined_history(
            combined_history,
            train_samples=pin.train_samples,
            validation_samples=pin.validation_samples,
        )
        final_summary = strict_json_load(summary_path)
        if final_summary.get("status") != "complete":
            raise RuntimeError("formal continuation did not reach epoch 60")
        print(summary_path)
        return summary_path
    finally:
        _release_lock(lock_path, descriptor)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
