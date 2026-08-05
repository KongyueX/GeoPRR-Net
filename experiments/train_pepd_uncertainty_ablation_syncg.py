"""Train the frozen secondary PEPD uncertainty-objective cohort from scratch.

All three arms start from the same pinned ImageNet ResNet-18 initialization and
the same zero-initialized auxiliary global log-variance scalar.  They use
identical data, grouped splits, RNG streams, architecture, geometry losses,
optimizer, and 60-epoch schedule.  Only the angular uncertainty objective
changes.  Continuing the retained PEPD checkpoint would be invalid because it
has already absorbed the learned heteroscedastic objective for 30 epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    FORMAL_BATCH_SIZE,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    FORMAL_WORKERS,
    GLOBAL_LOG_VARIANCE_INITIAL_VALUE,
    IMAGENET_RESNET18_INITIALIZATION_SHA256,
    PARENT_EPOCH,
    PEPD_UNCERTAINTY_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    TERMINAL_EPOCH,
    UNCERTAINTY_MECHANISM_ARMS,
    audit_main_convergence_cohort,
    convergence_audit,
    formal_manifest_path,
    formal_parent_pin,
    sha256_file,
    uncertainty_output_dir,
    validate_combined_history,
)
from experiments.pepd_uncertainty_objectives import (
    make_global_log_variance,
    probabilistic_direction_loss_with_uncertainty_mode,
    uncertainty_semantics,
)
from experiments.pepd_uncertainty_metrics import (
    CALIBRATION_THRESHOLDS_DEGREES,
    INVALID_DIRECTION_ERROR_DEGREES,
    PRIMARY_CALIBRATION_THRESHOLD_DEGREES,
    RISK_COVERAGE_LEVELS,
    UNCERTAINTY_DIAGNOSTIC_PROTOCOL,
)
from experiments.strict_json import strict_json_source_sha256
from experiments.probabilistic_pivot_direction import (
    IMAGENET_WEIGHTS,
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
    circular_delta,
    decode_probabilistic_pivot_direction,
    equivariance_loss,
)
from experiments.train_pepd_convergence_syncg import (
    _acquire_lock,
    _atomic_json,
    _atomic_torch,
    _checkpoint_epoch,
    _release_lock,
    _restore_rng_state,
    _rng_state,
)
from experiments.vdn_baseline import (
    angular_error_degrees,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    seed_worker,
    set_random_seed,
    sha256_source_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=UNCERTAINTY_MECHANISM_ARMS,
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _initialization_path() -> Path:
    return (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(IMAGENET_WEIGHTS.url).name
    )


def _signature(
    *,
    arm: str,
    seed: int,
    train_samples: list[Any],
    validation_samples: list[Any],
    manifest: Path,
    initial_model_state_sha256: str,
    main_convergence_cohort_sha256: str,
) -> dict[str, Any]:
    initialization = _initialization_path()
    if not initialization.is_file():
        raise FileNotFoundError(
            f"pinned ImageNet initialization is missing: {initialization}"
        )
    initialization_sha256 = sha256_file(initialization)
    if initialization_sha256 != IMAGENET_RESNET18_INITIALIZATION_SHA256:
        raise ValueError("ImageNet ResNet-18 initialization hash drifted")
    model_source_path = (
        PROJECT_ROOT / "experiments" / "probabilistic_pivot_direction.py"
    )
    model_source_sha256 = sha256_source_file(model_source_path)
    if model_source_sha256 != FORMAL_MODEL_SOURCE_SHA256:
        raise ValueError("PEPD model source drifted")
    return {
        "protocol": PEPD_UNCERTAINTY_TRAINING_PROTOCOL,
        "scope": "SyncG official train grouped validation only",
        "role": "secondary mechanism ablation; not algorithm selection",
        "arm": arm,
        "unique_difference": {
            "learned_heteroscedastic": (
                "original learned per-sample log-variance angular NLL"
            ),
            "global_homoscedastic": (
                "replace per-sample log-variance with one learned global "
                "log-variance scalar optimized on training batches only"
            ),
            "no_angular_nll": (
                "remove angular NLL; retain circular-bin and cosine losses"
            ),
        }[arm],
        "global_log_variance_initial_value": (
            GLOBAL_LOG_VARIANCE_INITIAL_VALUE
        ),
        "global_log_variance_learned": arm == "global_homoscedastic",
        "per_sample_variance_head_in_angular_objective": (
            arm == "learned_heteroscedastic"
        ),
        "seed": int(seed),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(
            manifest.with_name(manifest.name + ".protocol.json")
        ),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "model_source_sha256": model_source_sha256,
        "strict_json_source_sha256": strict_json_source_sha256(),
        "objective_source_sha256": sha256_source_file(
            PROJECT_ROOT / "experiments" / "pepd_uncertainty_objectives.py"
        ),
        "uncertainty_metrics_source_sha256": sha256_source_file(
            PROJECT_ROOT / "experiments" / "pepd_uncertainty_metrics.py"
        ),
        "trainer_source_sha256": sha256_source_file(Path(__file__).resolve()),
        "architecture": (
            "torchvision ResNet-18 + pivot heatmap + circular distribution + "
            "per-sample variance head; one auxiliary global log-variance "
            "scalar is present in every arm (architecture/state held constant)"
        ),
        "image_size": 256,
        "heatmap_size": 64,
        "angle_bins": 72,
        "epochs": TERMINAL_EPOCH,
        "phase1_epochs": PARENT_EPOCH,
        "phase2_epochs": TERMINAL_EPOCH - PARENT_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "learning_rate": 3.0e-4,
        "phase2_learning_rate": CONTINUATION_LEARNING_RATE,
        "weight_decay": 1.0e-4,
        "global_log_variance_weight_decay": 0.0,
        "pivot_loss_weight": 1.0,
        "bin_loss_weight": 0.20,
        "vector_loss_weight": 0.50,
        "paired_supervision_weight": 1.0,
        "equivariance_weight": 0.50,
        "equivariance_pivot_weight": 1.0,
        "soft_target_sigma_bins": 1.25,
        "validation_fraction": 0.10,
        "expansion": 1.25,
        "scale_factor": 0.10,
        "rotation_factor": 90.0,
        "translation_factor": 0.12,
        "heatmap_sigma": 1.5,
        "perspective_probability": 0.80,
        "max_perspective_degrees": 45.0,
        "max_blur_sigma": 3.0,
        "imagenet_pretrained": True,
        "imagenet_initialization": str(initialization),
        "imagenet_initialization_sha256": initialization_sha256,
        "initial_model_state_sha256": initial_model_state_sha256,
        "main_convergence_cohort_sha256": (
            main_convergence_cohort_sha256
        ),
        "optimizer": "AdamW",
        "schedule": (
            "CosineAnnealingLR(T_max=30,eta_min=3e-6) for epochs 1..30; "
            "constant 3e-6 for epochs 31..60"
        ),
        "mixed_precision": True,
        "grad_scaler_initial_scale": 512.0,
        "deterministic_algorithms": True,
        "rng_workers": 0,
        "early_stopping": False,
        "checkpoint_selection": (
            "lexicographic grouped-validation "
            "(angle_mae_degrees, pivot_mean_error_fraction); calibration is "
            "excluded because no_angular_nll has no trained variance semantics"
        ),
        "uncertainty_diagnostic_protocol": (
            UNCERTAINTY_DIAGNOSTIC_PROTOCOL
        ),
        "risk_coverage_levels": list(RISK_COVERAGE_LEVELS),
        "risk_coverage_unit": "physical_meter_group",
        "risk_coverage_tie_policy": "equal-score groups enter atomically",
        "calibration_thresholds_degrees": list(
            CALIBRATION_THRESHOLDS_DEGREES
        ),
        "primary_calibration_threshold_degrees": (
            PRIMARY_CALIBRATION_THRESHOLD_DEGREES
        ),
        "invalid_direction_error_degrees": (
            INVALID_DIRECTION_ERROR_DEGREES
        ),
    }


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(value.shape),
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    heatmap: torch.Tensor,
    direction: torch.Tensor,
    *,
    args: argparse.Namespace,
    global_log_variance: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | str | bool | None],
]:
    return probabilistic_direction_loss_with_uncertainty_mode(
        *outputs,
        heatmap,
        direction,
        pivot_weight=1.0,
        bin_weight=0.20,
        vector_weight=0.50,
        soft_target_sigma_bins=1.25,
        uncertainty_mode=args.arm,
        global_log_variance_raw=global_log_variance,
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_log_variance: torch.Tensor,
) -> dict[str, Any]:
    model.train()
    names = (
        "loss",
        "supervised_loss",
        "paired_supervised_loss",
        "pivot_loss",
        "direction_loss",
        "angular_nll_loss",
        "squared_angular_error_loss",
        "bin_loss",
        "cosine_loss",
        "equivariance_loss",
        "equivariance_pivot_loss",
        "equivariance_direction_loss",
        "mean_perspective_degrees",
    )
    totals = {name: 0.0 for name in names}
    samples = 0
    std_total = 0.0
    std_samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    progress = tqdm(
        loader,
        desc=f"PEPD uncertainty {args.arm} train {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        paired_images = batch["paired_image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        paired_heatmap = batch["paired_heatmap"].to(device, non_blocking=True)
        paired_direction = batch["paired_direction"].to(device, non_blocking=True)
        homography = batch["homography"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=True):
            joined = model(torch.cat((images, paired_images), dim=0))
        count = int(images.shape[0])
        first = tuple(value[:count] for value in joined)
        second = tuple(value[count:] for value in joined)
        first_loss, first_components = _loss(
            first,
            heatmap,
            direction,
            args=args,
            global_log_variance=global_log_variance,
        )
        second_loss, second_components = _loss(
            second,
            paired_heatmap,
            paired_direction,
            args=args,
            global_log_variance=global_log_variance,
        )
        consistency, consistency_components = equivariance_loss(
            first,
            second,
            homography,
            image_size=256,
            heatmap_size=64,
            pivot_weight=1.0,
        )
        loss = first_loss + second_loss + 0.50 * consistency
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1
        else:
            optimizer_steps += 1
        values = {
            "loss": loss.detach(),
            "supervised_loss": first_loss.detach(),
            "paired_supervised_loss": second_loss.detach(),
            "pivot_loss": 0.5
            * (
                first_components["pivot_loss"]
                + second_components["pivot_loss"]
            ),
            "direction_loss": 0.5
            * (
                first_components["direction_loss"]
                + second_components["direction_loss"]
            ),
            "angular_nll_loss": 0.5
            * (
                first_components["angular_nll_loss"]
                + second_components["angular_nll_loss"]
            ),
            "squared_angular_error_loss": 0.5
            * (
                first_components["squared_angular_error_loss"]
                + second_components["squared_angular_error_loss"]
            ),
            "bin_loss": 0.5
            * (first_components["bin_loss"] + second_components["bin_loss"]),
            "cosine_loss": 0.5
            * (
                first_components["cosine_loss"]
                + second_components["cosine_loss"]
            ),
            "equivariance_loss": consistency.detach(),
            "equivariance_pivot_loss": consistency_components[
                "equivariance_pivot_loss"
            ],
            "equivariance_direction_loss": consistency_components[
                "equivariance_direction_loss"
            ],
            "mean_perspective_degrees": batch["perspective_degrees"].mean(),
        }
        samples += count
        for name, value in values.items():
            totals[name] += float(value) * count
        first_std = first_components["mean_angle_std_degrees"]
        second_std = second_components["mean_angle_std_degrees"]
        if isinstance(first_std, torch.Tensor) and isinstance(
            second_std,
            torch.Tensor,
        ):
            std_total += float(0.5 * (first_std + second_std)) * count
            std_samples += count
        progress.set_postfix(loss=f"{totals['loss'] / samples:.4f}")
    result: dict[str, Any] = {
        name: total / max(samples, 1) for name, total in totals.items()
    } | {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
        "uncertainty_mode": args.arm,
        "angular_nll_in_training_objective": args.arm != "no_angular_nll",
        "per_sample_variance_head_in_training_objective": (
            args.arm == "learned_heteroscedastic"
        ),
        "global_log_variance_in_training_objective": (
            args.arm == "global_homoscedastic"
        ),
    }
    if args.arm == "no_angular_nll":
        result["mean_angle_std_degrees"] = None
        result["global_log_variance"] = None
    else:
        if std_samples != samples:
            raise RuntimeError("trained uncertainty arm has no finite std")
        result["mean_angle_std_degrees"] = std_total / max(std_samples, 1)
        result["global_log_variance"] = (
            float(global_log_variance.detach())
            if args.arm == "global_homoscedastic"
            else None
        )
    return result


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    global_log_variance: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    totals = {
        "loss": 0.0,
        "pivot_loss": 0.0,
        "direction_loss": 0.0,
        "angular_nll_loss": 0.0,
        "squared_angular_error_loss": 0.0,
        "bin_loss": 0.0,
        "cosine_loss": 0.0,
    }
    angle_errors: list[float] = []
    signed_errors: list[float] = []
    pivot_errors: list[float] = []
    angle_stds: list[float] = []
    valid_count = 0
    samples = 0
    for batch in tqdm(
        loader,
        desc=f"PEPD uncertainty {args.arm} validation",
        leave=False,
        dynamic_ncols=True,
    ):
        images = batch["image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        target_pivot = batch["pivot"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=True):
            outputs = model(images)
        loss, components = _loss(
            outputs,
            heatmap,
            direction,
            args=args,
            global_log_variance=global_log_variance,
        )
        prediction = decode_probabilistic_pivot_direction(
            *(value.float() for value in outputs)
        )
        semantics = uncertainty_semantics(
            outputs[3],
            mode=args.arm,
            global_log_variance_raw=global_log_variance,
        )
        angle = angular_error_degrees(prediction.direction, direction.float())
        predicted_angle = torch.atan2(
            prediction.direction[:, 1],
            prediction.direction[:, 0],
        )
        target_angle = torch.atan2(direction[:, 1], direction[:, 0])
        signed = circular_delta(predicted_angle, target_angle) * (
            180.0 / math.pi
        )
        pivot_distance = torch.linalg.vector_norm(
            prediction.pivot_xy - target_pivot,
            dim=1,
        )
        pivot_distance = pivot_distance * 4.0 / 256.0
        valid = prediction.valid
        angle_errors.extend(angle[valid].detach().cpu().tolist())
        signed_errors.extend(signed[valid].detach().cpu().tolist())
        pivot_errors.extend(pivot_distance.detach().cpu().tolist())
        if semantics.angle_std_degrees is not None:
            angle_stds.extend(
                semantics.angle_std_degrees[valid].detach().cpu().tolist()
            )
        valid_count += int(valid.sum().item())
        count = int(images.shape[0])
        samples += count
        totals["loss"] += float(loss) * count
        for name in totals:
            if name != "loss":
                totals[name] += float(components[name]) * count
    errors = np.asarray(angle_errors, dtype=np.float64)
    signed = np.asarray(signed_errors, dtype=np.float64)
    pivots = np.asarray(pivot_errors, dtype=np.float64)
    calibration: dict[str, Any]
    if angle_stds:
        stds = np.asarray(angle_stds, dtype=np.float64)
        variances = np.maximum(np.square(np.deg2rad(stds)), 1e-12)
        calibration_nll = float(
            np.mean(
                0.5
                * (
                    np.square(np.deg2rad(signed)) / variances
                    + np.log(variances)
                )
            )
        )
        calibration = {
            "mean_angle_std_degrees": float(np.mean(stds)),
            "angular_calibration_nll": calibration_nll,
            "angle_within_1sigma": float(np.mean(errors <= stds)),
            "angle_within_2sigma": float(np.mean(errors <= 2.0 * stds)),
        }
    else:
        calibration = {
            "mean_angle_std_degrees": None,
            "angular_calibration_nll": None,
            "angle_within_1sigma": None,
            "angle_within_2sigma": None,
        }
    result = {
        name: total / max(samples, 1) for name, total in totals.items()
    } | {
        "samples": samples,
        "valid_directions": valid_count,
        "direction_coverage": valid_count / max(samples, 1),
        "angle_mae_degrees": float(np.mean(errors)),
        "angle_median_degrees": float(np.median(errors)),
        "angle_acc_1deg": float(np.mean(errors <= 1.0)),
        "angle_acc_3deg": float(np.mean(errors <= 3.0)),
        "angle_acc_5deg": float(np.mean(errors <= 5.0)),
        "pivot_mean_error_fraction": float(np.mean(pivots)),
        "pivot_median_error_fraction": float(np.median(pivots)),
        "calibration_semantics": uncertainty_semantics(
            torch.zeros((1, 1), device=device),
            mode=args.arm,
            global_log_variance_raw=global_log_variance,
        ).calibration_semantics,
        "angular_nll_in_training_objective": args.arm != "no_angular_nll",
        "sample_ranking_available": (
            args.arm == "learned_heteroscedastic"
        ),
        "global_log_variance": (
            float(global_log_variance.detach())
            if args.arm == "global_homoscedastic"
            else None
        ),
    }
    return result | calibration


def _configure(seed: int, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal uncertainty ablation is CUDA-only")
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(f"formal run requires PYTHONHASHSEED={seed}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal run requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    set_random_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _checkpoint(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_log_variance: torch.Tensor,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None,
    history: list[dict[str, Any]],
    signature: Mapping[str, Any],
    best_angle: float,
    best_pivot: float,
    best_epoch: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    value = {
        "protocol": PEPD_UNCERTAINTY_TRAINING_PROTOCOL,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "global_log_variance_state": (
            global_log_variance.detach().cpu().clone()
        ),
        "history": history,
        "signature": dict(signature),
        "best_angle": best_angle,
        "best_pivot_error": best_pivot,
        "best_epoch": best_epoch,
    } | _rng_state(generator)
    if scheduler is not None and epoch < PARENT_EPOCH:
        value["scheduler_state"] = scheduler.state_dict()
    elif epoch >= PARENT_EPOCH:
        value["retired_scheduler"] = {
            "T_max": PARENT_EPOCH,
            "eta_min": CONTINUATION_LEARNING_RATE,
            "retired_after_epoch": PARENT_EPOCH,
        }
    return value


def _summary(
    *,
    status: str,
    signature: Mapping[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    best_angle: float,
    best_pivot: float,
    best_path: Path,
    last_path: Path,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PEPD_UNCERTAINTY_TRAINING_PROTOCOL,
        "status": status,
        "scope": "SyncG official train grouped validation only",
        "role": "secondary mechanism ablation; not algorithm selection",
        "signature": dict(signature),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": best_angle,
        "best_validation_pivot_error_fraction": best_pivot,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "convergence_audit": (
            convergence_audit(history, best_epoch=best_epoch)
            if len(history) == TERMINAL_EPOCH
            else None
        ),
        "eligible_for_model_selection": False,
        "public_test_field_evaluation_authorized": False,
        "environment": dict(environment),
    }


def run(args: argparse.Namespace) -> Path:
    arm = str(args.arm)
    seed = int(args.seed)
    manifest = formal_manifest_path(args.manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    main_gate = audit_main_convergence_cohort()
    output_dir = uncertainty_output_dir(arm, seed)
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
        raise FileNotFoundError("resume requires uncertainty last.pt")
    descriptor = _acquire_lock(lock_path)
    try:
        device = torch.device(args.device)
        _configure(seed, device)
        samples, manifest_protocol = load_syncg_manifest(
            manifest,
            expected_split="train",
        )
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=0.10,
            seed=seed,
        )
        pin = formal_parent_pin(seed)
        if (
            len(train_samples) != pin.train_samples
            or len(validation_samples) != pin.validation_samples
            or sample_ids_hash(train_samples) != pin.train_ids_sha256
            or sample_ids_hash(validation_samples) != pin.validation_ids_sha256
        ):
            raise ValueError("uncertainty cohort grouped split identity drifted")
        common_dataset = {
            "image_size": 256,
            "heatmap_size": 64,
            "expansion": 1.25,
            "scale_factor": 0.10,
            "rotation_factor": 90.0,
            "translation_factor": 0.12,
            "heatmap_sigma": 1.5,
            "perspective_probability": 0.80,
            "max_perspective_degrees": 45.0,
            "max_blur_sigma": 3.0,
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
        generator = torch.Generator().manual_seed(seed)
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
            angle_bins=72,
            imagenet_pretrained=True,
        ).to(device)
        global_log_variance = make_global_log_variance(device=device)
        initial_model_hash = model_state_sha256(model.state_dict())
        signature = _signature(
            arm=arm,
            seed=seed,
            train_samples=train_samples,
            validation_samples=validation_samples,
            manifest=manifest,
            initial_model_state_sha256=initial_model_hash,
            main_convergence_cohort_sha256=main_gate["sha256"],
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(model.parameters()),
                    "weight_decay": 1.0e-4,
                },
                {
                    "params": [global_log_variance],
                    "weight_decay": 0.0,
                },
            ],
            lr=3.0e-4,
        )
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=PARENT_EPOCH,
                eta_min=CONTINUATION_LEARNING_RATE,
            )
        )
        scaler = torch.amp.GradScaler(
            device.type,
            enabled=True,
            init_scale=512.0,
        )
        if args.resume:
            checkpoint = torch.load(
                last_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("signature") != signature:
                raise ValueError("uncertainty ablation resume signature mismatch")
            epoch = int(checkpoint.get("epoch", -1))
            if not 0 < epoch < TERMINAL_EPOCH:
                raise ValueError("uncertainty resume epoch is invalid")
            model.load_state_dict(checkpoint["model_state"])
            saved_global = checkpoint.get("global_log_variance_state")
            if not isinstance(saved_global, torch.Tensor) or saved_global.numel() != 1:
                raise ValueError(
                    "uncertainty resume has no global log-variance scalar"
                )
            global_log_variance.data.copy_(
                saved_global.to(device=device, dtype=torch.float32).reshape(())
            )
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            if epoch < PARENT_EPOCH:
                if "scheduler_state" not in checkpoint:
                    raise ValueError("phase-1 resume has no scheduler state")
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            else:
                scheduler = None
            history = list(checkpoint["history"])
            best_angle = float(checkpoint["best_angle"])
            best_pivot = float(checkpoint["best_pivot_error"])
            best_epoch = int(checkpoint["best_epoch"])
            _restore_rng_state(checkpoint, generator)
            start_epoch = epoch + 1
            if best_epoch == epoch and _checkpoint_epoch(best_path) != epoch:
                _atomic_torch(best_path, checkpoint)
        else:
            # Re-seed after deterministic head initialization so data streams
            # are identical across the three uncertainty arms.
            set_random_seed(seed)
            generator.manual_seed(seed)
            history = []
            best_angle = math.inf
            best_pivot = math.inf
            best_epoch = 0
            start_epoch = 1
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
            learning_rate = float(optimizer.param_groups[0]["lr"])
            if epoch > PARENT_EPOCH and not math.isclose(
                learning_rate,
                CONTINUATION_LEARNING_RATE,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("uncertainty phase-2 LR drifted")
            if any(
                not math.isclose(
                    float(group["lr"]),
                    learning_rate,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for group in optimizer.param_groups
            ):
                raise ValueError("uncertainty optimizer group LR drifted")
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device=device,
                args=args,
                epoch=epoch,
                global_log_variance=global_log_variance,
            )
            validation_metrics = _validate(
                model,
                validation_loader,
                device=device,
                args=args,
                global_log_variance=global_log_variance,
            )
            if scheduler is not None:
                scheduler.step()
                if epoch == PARENT_EPOCH:
                    scheduler = None
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
            )
            candidate = (
                float(validation_metrics["angle_mae_degrees"]),
                float(validation_metrics["pivot_mean_error_fraction"]),
            )
            improved = candidate < (best_angle, best_pivot)
            if improved:
                best_angle, best_pivot = candidate
                best_epoch = epoch
            checkpoint = _checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                global_log_variance=global_log_variance,
                scheduler=scheduler,
                history=history,
                signature=signature,
                best_angle=best_angle,
                best_pivot=best_pivot,
                best_epoch=best_epoch,
                generator=generator,
            )
            _atomic_torch(last_path, checkpoint)
            if improved:
                _atomic_torch(best_path, checkpoint)
            report = _summary(
                status="complete" if epoch == TERMINAL_EPOCH else "running",
                signature=signature,
                history=history,
                best_epoch=best_epoch,
                best_angle=best_angle,
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
            require_calibration_metric=arm != "no_angular_nll",
        )
        print(summary_path)
        return summary_path
    finally:
        _release_lock(lock_path, descriptor)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
