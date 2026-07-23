"""Train the probabilistic perspective-equivariant SyncG direction expert."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.probabilistic_pivot_direction import (
    IMAGENET_WEIGHTS,
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
    circular_delta,
    decode_probabilistic_pivot_direction,
    equivariance_loss,
    probabilistic_direction_loss,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    angular_error_degrees,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    seed_worker,
    set_random_seed,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260722"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--angle-bins", type=int, default=72)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pivot-loss-weight", type=float, default=1.0)
    parser.add_argument("--bin-loss-weight", type=float, default=0.20)
    parser.add_argument("--vector-loss-weight", type=float, default=0.50)
    parser.add_argument("--paired-supervision-weight", type=float, default=1.0)
    parser.add_argument("--equivariance-weight", type=float, default=0.50)
    parser.add_argument("--equivariance-pivot-weight", type=float, default=1.0)
    parser.add_argument("--soft-target-sigma-bins", type=float, default=1.25)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--scale-factor", type=float, default=0.10)
    parser.add_argument("--rotation-factor", type=float, default=90.0)
    parser.add_argument("--translation-factor", type=float, default=0.12)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
    parser.add_argument("--perspective-probability", type=float, default=0.80)
    parser.add_argument("--max-perspective-degrees", type=float, default=45.0)
    parser.add_argument("--max-blur-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    return parser.parse_args()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _initialization_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(IMAGENET_WEIGHTS.url).name


def _signature(args: argparse.Namespace, train_samples, validation_samples) -> dict[str, Any]:
    protocol_path = args.manifest.with_name(args.manifest.name + ".protocol.json")
    initialization = None if args.no_imagenet_pretrained else _initialization_path()
    if initialization is not None and not initialization.is_file():
        raise FileNotFoundError(f"torchvision initialization is missing: {initialization}")
    return {
        "protocol": PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "model_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
        ),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "architecture": (
            "torchvision ResNet-18 + pivot heatmap + circular distribution + "
            "heteroscedastic direction"
        ),
        "image_size": int(args.image_size),
        "heatmap_size": int(args.heatmap_size),
        "angle_bins": int(args.angle_bins),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "pivot_loss_weight": float(args.pivot_loss_weight),
        "bin_loss_weight": float(args.bin_loss_weight),
        "vector_loss_weight": float(args.vector_loss_weight),
        "paired_supervision_weight": float(args.paired_supervision_weight),
        "equivariance_weight": float(args.equivariance_weight),
        "equivariance_pivot_weight": float(args.equivariance_pivot_weight),
        "soft_target_sigma_bins": float(args.soft_target_sigma_bins),
        "validation_fraction": float(args.validation_fraction),
        "expansion": float(args.expansion),
        "scale_factor": float(args.scale_factor),
        "rotation_factor": float(args.rotation_factor),
        "translation_factor": float(args.translation_factor),
        "heatmap_sigma": float(args.heatmap_sigma),
        "perspective_probability": float(args.perspective_probability),
        "max_perspective_degrees": float(args.max_perspective_degrees),
        "max_blur_sigma": float(args.max_blur_sigma),
        "photometric_augmentation": (
            "paired contrast+brightness,gamma,gaussian/motion blur,gaussian noise"
        ),
        "projective_supervision": "exact transformed pivot and pointer ray",
        "imagenet_pretrained": not args.no_imagenet_pretrained,
        "imagenet_initialization": str(initialization) if initialization else None,
        "imagenet_initialization_sha256": (
            sha256_file(initialization) if initialization else None
        ),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": str(args.device).startswith("cuda") and not args.no_amp,
        "grad_scaler_initial_scale": 512.0,
        "seed": int(args.seed),
        "diagnostic_limit": args.limit,
    }


def _supervised_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    heatmap: torch.Tensor,
    direction: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return probabilistic_direction_loss(
        *outputs,
        heatmap,
        direction,
        pivot_weight=args.pivot_loss_weight,
        bin_weight=args.bin_loss_weight,
        vector_weight=args.vector_loss_weight,
        soft_target_sigma_bins=args.soft_target_sigma_bins,
    )


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    names = (
        "loss",
        "supervised_loss",
        "paired_supervised_loss",
        "pivot_loss",
        "direction_loss",
        "angular_nll_loss",
        "bin_loss",
        "cosine_loss",
        "equivariance_loss",
        "equivariance_pivot_loss",
        "equivariance_direction_loss",
        "mean_angle_std_degrees",
        "mean_perspective_degrees",
    )
    totals = {name: 0.0 for name in names}
    samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    progress = tqdm(loader, desc=f"probabilistic train {epoch}", leave=False, dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        paired_images = batch["paired_image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        paired_heatmap = batch["paired_heatmap"].to(device, non_blocking=True)
        paired_direction = batch["paired_direction"].to(device, non_blocking=True)
        homography = batch["homography"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            joined_outputs = model(torch.cat((images, paired_images), dim=0))
        count = int(images.shape[0])
        first = tuple(value[:count] for value in joined_outputs)
        second = tuple(value[count:] for value in joined_outputs)
        first_loss, first_components = _supervised_loss(
            first, heatmap, direction, args
        )
        second_loss, second_components = _supervised_loss(
            second, paired_heatmap, paired_direction, args
        )
        consistency, consistency_components = equivariance_loss(
            first,
            second,
            homography,
            image_size=args.image_size,
            heatmap_size=args.heatmap_size,
            pivot_weight=args.equivariance_pivot_weight,
        )
        loss = (
            first_loss
            + float(args.paired_supervision_weight) * second_loss
            + float(args.equivariance_weight) * consistency
        )
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1
        else:
            optimizer_steps += 1
        samples += count
        values = {
            "loss": loss.detach(),
            "supervised_loss": first_loss.detach(),
            "paired_supervised_loss": second_loss.detach(),
            "pivot_loss": 0.5
            * (first_components["pivot_loss"] + second_components["pivot_loss"]),
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
            "bin_loss": 0.5
            * (first_components["bin_loss"] + second_components["bin_loss"]),
            "cosine_loss": 0.5
            * (first_components["cosine_loss"] + second_components["cosine_loss"]),
            "equivariance_loss": consistency.detach(),
            "equivariance_pivot_loss": consistency_components[
                "equivariance_pivot_loss"
            ],
            "equivariance_direction_loss": consistency_components[
                "equivariance_direction_loss"
            ],
            "mean_angle_std_degrees": 0.5
            * (
                first_components["mean_angle_std_degrees"]
                + second_components["mean_angle_std_degrees"]
            ),
            "mean_perspective_degrees": batch["perspective_degrees"].mean(),
        }
        for name, value in values.items():
            totals[name] += float(value) * count
        progress.set_postfix(loss=f"{totals['loss'] / samples:.4f}")
    return {name: value / max(samples, 1) for name, value in totals.items()} | {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "pivot_loss": 0.0,
        "direction_loss": 0.0,
        "angular_nll_loss": 0.0,
        "bin_loss": 0.0,
        "cosine_loss": 0.0,
    }
    angle_errors: list[float] = []
    signed_angle_errors: list[float] = []
    pivot_errors: list[float] = []
    confidence: list[float] = []
    angle_stds: list[float] = []
    entropies: list[float] = []
    resultants: list[float] = []
    valid_count = 0
    samples = 0
    for batch in tqdm(loader, desc="probabilistic validation", leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        target_pivot = batch["pivot"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            outputs = model(images)
        loss, components = _supervised_loss(outputs, heatmap, direction, args)
        prediction = decode_probabilistic_pivot_direction(
            *(value.float() for value in outputs)
        )
        angle = angular_error_degrees(prediction.direction, direction.float())
        predicted_angle = torch.atan2(
            prediction.direction[:, 1], prediction.direction[:, 0]
        )
        target_angle = torch.atan2(direction[:, 1], direction[:, 0])
        signed = circular_delta(predicted_angle, target_angle) * (180.0 / math.pi)
        stride = float(args.image_size) / float(args.heatmap_size)
        pivot_distance = torch.linalg.vector_norm(
            prediction.pivot_xy - target_pivot, dim=1
        )
        pivot_distance = pivot_distance * stride / float(args.image_size)
        valid = prediction.valid
        angle_errors.extend(angle[valid].detach().cpu().tolist())
        signed_angle_errors.extend(signed[valid].detach().cpu().tolist())
        pivot_errors.extend(pivot_distance.detach().cpu().tolist())
        confidence.extend(prediction.pivot_peak.detach().cpu().tolist())
        angle_stds.extend(prediction.angle_std_degrees[valid].detach().cpu().tolist())
        entropies.extend(prediction.angle_entropy[valid].detach().cpu().tolist())
        resultants.extend(
            prediction.bin_resultant_length[valid].detach().cpu().tolist()
        )
        valid_count += int(valid.sum().item())
        count = int(images.shape[0])
        samples += count
        totals["loss"] += float(loss) * count
        for name in totals:
            if name != "loss":
                totals[name] += float(components[name]) * count
    errors = np.asarray(angle_errors, dtype=np.float64)
    signed = np.asarray(signed_angle_errors, dtype=np.float64)
    pivots = np.asarray(pivot_errors, dtype=np.float64)
    stds = np.asarray(angle_stds, dtype=np.float64)
    if errors.size and stds.size:
        calibration_nll = float(
            np.mean(
                0.5
                * (
                    np.square(np.deg2rad(signed))
                    / np.maximum(np.square(np.deg2rad(stds)), 1e-12)
                    + np.log(np.maximum(np.square(np.deg2rad(stds)), 1e-12))
                )
            )
        )
        one_sigma = float(np.mean(errors <= stds))
        two_sigma = float(np.mean(errors <= 2.0 * stds))
        uncertainty_correlation = (
            float(np.corrcoef(errors, stds)[0, 1])
            if errors.size > 1 and np.std(errors) > 0 and np.std(stds) > 0
            else math.nan
        )
    else:
        calibration_nll = math.inf
        one_sigma = 0.0
        two_sigma = 0.0
        uncertainty_correlation = math.nan
    return {name: value / max(samples, 1) for name, value in totals.items()} | {
        "samples": samples,
        "valid_directions": valid_count,
        "direction_coverage": valid_count / max(samples, 1),
        "angle_mae_degrees": float(np.mean(errors)) if errors.size else math.inf,
        "angle_median_degrees": float(np.median(errors)) if errors.size else math.inf,
        "angle_acc_1deg": float(np.mean(errors <= 1.0)) if errors.size else 0.0,
        "angle_acc_3deg": float(np.mean(errors <= 3.0)) if errors.size else 0.0,
        "angle_acc_5deg": float(np.mean(errors <= 5.0)) if errors.size else 0.0,
        "pivot_mean_error_fraction": float(np.mean(pivots)) if pivots.size else math.inf,
        "pivot_median_error_fraction": float(np.median(pivots)) if pivots.size else math.inf,
        "mean_pivot_peak": float(np.mean(confidence)) if confidence else math.nan,
        "mean_angle_std_degrees": float(np.mean(stds)) if stds.size else math.nan,
        "median_angle_std_degrees": float(np.median(stds)) if stds.size else math.nan,
        "mean_angle_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "mean_bin_resultant_length": float(np.mean(resultants)) if resultants else math.nan,
        "angular_calibration_nll": calibration_nll,
        "angle_within_1sigma": one_sigma,
        "angle_within_2sigma": two_sigma,
        "uncertainty_error_correlation": uncertainty_correlation,
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.image_size % 32 != 0 or args.heatmap_size != args.image_size // 4:
        raise ValueError("the ResNet-18 head requires heatmap_size == image_size / 4")
    if args.angle_bins < 8:
        raise ValueError("angle-bins must be at least 8")
    nonnegative = (
        args.pivot_loss_weight,
        args.bin_loss_weight,
        args.vector_loss_weight,
        args.paired_supervision_weight,
        args.equivariance_weight,
        args.equivariance_pivot_weight,
    )
    if any(float(value) < 0.0 for value in nonnegative):
        raise ValueError("loss weights must be non-negative")
    set_random_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    samples, manifest_protocol = load_syncg_manifest(
        args.manifest,
        expected_split="train",
        limit=args.limit,
    )
    train_samples, validation_samples = grouped_train_val_split(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    signature = _signature(args, train_samples, validation_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_dataset = {
        "image_size": args.image_size,
        "heatmap_size": args.heatmap_size,
        "expansion": args.expansion,
        "scale_factor": args.scale_factor,
        "rotation_factor": args.rotation_factor,
        "translation_factor": args.translation_factor,
        "heatmap_sigma": args.heatmap_sigma,
        "perspective_probability": args.perspective_probability,
        "max_perspective_degrees": args.max_perspective_degrees,
        "max_blur_sigma": args.max_blur_sigma,
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
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    summary_path = args.output_dir / "summary.json"
    model = build_probabilistic_pivot_direction_model(
        angle_bins=args.angle_bins,
        imagenet_pretrained=not args.no_imagenet_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.01,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=512.0,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_angle = math.inf
    best_calibration_nll = math.inf
    best_pivot_error = math.inf
    best_epoch = 0
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError("probabilistic direction resume signature mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history") or [])
        best_angle = float(checkpoint.get("best_angle", math.inf))
        best_calibration_nll = float(checkpoint.get("best_calibration_nll", math.inf))
        best_pivot_error = float(checkpoint.get("best_pivot_error", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            args=args,
            epoch=epoch,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            args=args,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        angle = float(validation_metrics["angle_mae_degrees"])
        calibration_nll = float(validation_metrics["angular_calibration_nll"])
        pivot_error = float(validation_metrics["pivot_mean_error_fraction"])
        candidate = (angle, calibration_nll, pivot_error)
        incumbent = (best_angle, best_calibration_nll, best_pivot_error)
        improved = candidate < incumbent
        if improved:
            best_angle, best_calibration_nll, best_pivot_error = candidate
            best_epoch = epoch
        checkpoint = {
            "protocol": PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "signature": signature,
            "best_angle": best_angle,
            "best_calibration_nll": best_calibration_nll,
            "best_pivot_error": best_pivot_error,
            "best_epoch": best_epoch,
        }
        _torch_save(last_path, checkpoint)
        if improved:
            _torch_save(best_path, checkpoint)
        summary = {
            "protocol": PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
            "status": "running" if epoch < args.epochs else "complete",
            "signature": signature,
            "manifest_protocol": manifest_protocol,
            "best_epoch": best_epoch,
            "best_validation_angle_mae_degrees": best_angle,
            "best_validation_angular_calibration_nll": best_calibration_nll,
            "best_validation_pivot_error_fraction": best_pivot_error,
            "history": history,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "elapsed_seconds": time.time() - started,
        }
        _json_write(summary_path, summary)
        print(
            f"epoch={epoch}/{args.epochs} lr={learning_rate:.3e} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_angle={angle:.4f}deg val_std="
            f"{validation_metrics['mean_angle_std_degrees']:.3f}deg "
            f"val_pivot={pivot_error:.5f} best={best_angle:.4f}deg@{best_epoch}",
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_summary.update(
        {
            "status": "complete",
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(last_path),
            "last_checkpoint_sha256": sha256_file(last_path),
            "optimizer_steps": sum(
                int(item["train"]["optimizer_steps"]) for item in history
            ),
            "skipped_optimizer_steps": sum(
                int(item["train"]["skipped_optimizer_steps"]) for item in history
            ),
        }
    )
    _json_write(summary_path, final_summary)
    print(summary_path)


if __name__ == "__main__":
    main()
