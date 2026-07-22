"""Train the independent SyncG pivot-and-direction fallback head."""
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

from experiments.pivot_direction_fallback import (
    PIVOT_DIRECTION_PROTOCOL,
    SyncGPivotDirectionDataset,
    build_pivot_direction_model,
    decode_pivot_direction,
    imagenet_checkpoint_path,
    pivot_direction_loss,
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
        default=Path("artifacts/runs/pivot_direction_syncg/seed_20260722"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pivot-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--expansion", type=float, default=1.25)
    parser.add_argument("--scale-factor", type=float, default=0.10)
    parser.add_argument("--rotation-factor", type=float, default=90.0)
    parser.add_argument("--translation-factor", type=float, default=0.12)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5)
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


def _signature(args: argparse.Namespace, train_samples, validation_samples) -> dict[str, Any]:
    protocol_path = args.manifest.with_name(args.manifest.name + ".protocol.json")
    initialization = None if args.no_imagenet_pretrained else imagenet_checkpoint_path()
    if initialization is not None and not initialization.is_file():
        raise FileNotFoundError(f"torchvision initialization is missing: {initialization}")
    return {
        "protocol": PIVOT_DIRECTION_PROTOCOL,
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "model_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "pivot_direction_fallback.py"
        ),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "architecture": "torchvision ResNet-18 + pivot heatmap + global unit vector",
        "image_size": int(args.image_size),
        "heatmap_size": int(args.heatmap_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "pivot_loss_weight": float(args.pivot_loss_weight),
        "validation_fraction": float(args.validation_fraction),
        "expansion": float(args.expansion),
        "scale_factor": float(args.scale_factor),
        "rotation_factor": float(args.rotation_factor),
        "translation_factor": float(args.translation_factor),
        "heatmap_sigma": float(args.heatmap_sigma),
        "photometric_augmentation": (
            "contrast+brightness,gamma,gaussian_blur,gaussian_noise"
        ),
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


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    pivot_weight: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "pivot_loss": 0.0, "direction_loss": 0.0}
    samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    progress = tqdm(loader, desc=f"fallback train {epoch}", leave=False, dynamic_ncols=True)
    for images, heatmap, direction, _, _ in progress:
        images = images.to(device, non_blocking=True)
        heatmap = heatmap.to(device, non_blocking=True)
        direction = direction.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            pivot_logits, direction_raw = model(images)
        loss, components = pivot_direction_loss(
            pivot_logits,
            direction_raw,
            heatmap,
            direction,
            pivot_weight=pivot_weight,
        )
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1
        else:
            optimizer_steps += 1
        count = int(images.shape[0])
        samples += count
        totals["loss"] += float(loss.detach()) * count
        totals["pivot_loss"] += float(components["pivot_loss"]) * count
        totals["direction_loss"] += float(components["direction_loss"]) * count
        progress.set_postfix(loss=f"{totals['loss'] / samples:.4f}")
    return {
        key: value / max(samples, 1) for key, value in totals.items()
    } | {
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
    pivot_weight: float,
    image_size: int,
    heatmap_size: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "pivot_loss": 0.0, "direction_loss": 0.0}
    angle_errors: list[float] = []
    pivot_errors: list[float] = []
    confidence: list[float] = []
    valid_count = 0
    samples = 0
    for images, heatmap, direction, target_pivot, _ in tqdm(
        loader,
        desc="fallback validation",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        heatmap = heatmap.to(device, non_blocking=True)
        direction = direction.to(device, non_blocking=True)
        target_pivot = target_pivot.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            pivot_logits, direction_raw = model(images)
        loss, components = pivot_direction_loss(
            pivot_logits,
            direction_raw,
            heatmap,
            direction,
            pivot_weight=pivot_weight,
        )
        pivot_xy, predicted_direction, peak, valid = decode_pivot_direction(
            pivot_logits.float(), direction_raw.float()
        )
        angle = angular_error_degrees(predicted_direction, direction.float())
        stride = float(image_size) / float(heatmap_size)
        pivot_distance = torch.linalg.vector_norm(pivot_xy - target_pivot, dim=1)
        pivot_distance = pivot_distance * stride / float(image_size)
        angle_errors.extend(angle[valid].detach().cpu().tolist())
        pivot_errors.extend(pivot_distance.detach().cpu().tolist())
        confidence.extend(peak.detach().cpu().tolist())
        valid_count += int(valid.sum().item())
        count = int(images.shape[0])
        samples += count
        totals["loss"] += float(loss) * count
        totals["pivot_loss"] += float(components["pivot_loss"]) * count
        totals["direction_loss"] += float(components["direction_loss"]) * count
    errors = np.asarray(angle_errors, dtype=np.float64)
    pivots = np.asarray(pivot_errors, dtype=np.float64)
    return {
        key: value / max(samples, 1) for key, value in totals.items()
    } | {
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
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.image_size % 32 != 0 or args.heatmap_size != args.image_size // 4:
        raise ValueError("the ResNet-18 head requires heatmap_size == image_size / 4")
    if args.pivot_loss_weight < 0.0:
        raise ValueError("pivot loss weight must be non-negative")
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

    train_dataset = SyncGPivotDirectionDataset(
        train_samples,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        training=True,
        expansion=args.expansion,
        scale_factor=args.scale_factor,
        rotation_factor=args.rotation_factor,
        translation_factor=args.translation_factor,
        heatmap_sigma=args.heatmap_sigma,
    )
    validation_dataset = SyncGPivotDirectionDataset(
        validation_samples,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        training=False,
        expansion=args.expansion,
        scale_factor=args.scale_factor,
        rotation_factor=args.rotation_factor,
        translation_factor=args.translation_factor,
        heatmap_sigma=args.heatmap_sigma,
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
    model = build_pivot_direction_model(
        imagenet_pretrained=not args.no_imagenet_pretrained
    )
    model.to(device)
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
    best_pivot_error = math.inf
    best_epoch = 0
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError("pivot-direction resume signature mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history") or [])
        best_angle = float(checkpoint.get("best_angle", math.inf))
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
            pivot_weight=args.pivot_loss_weight,
            epoch=epoch,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            pivot_weight=args.pivot_loss_weight,
            image_size=args.image_size,
            heatmap_size=args.heatmap_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        angle = float(validation_metrics["angle_mae_degrees"])
        pivot_error = float(validation_metrics["pivot_mean_error_fraction"])
        improved = angle < best_angle - 1e-10 or (
            abs(angle - best_angle) <= 1e-10 and pivot_error < best_pivot_error
        )
        if improved:
            best_angle = angle
            best_pivot_error = pivot_error
            best_epoch = epoch
        checkpoint = {
            "protocol": PIVOT_DIRECTION_PROTOCOL,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "signature": signature,
            "best_angle": best_angle,
            "best_pivot_error": best_pivot_error,
            "best_epoch": best_epoch,
        }
        _torch_save(last_path, checkpoint)
        if improved:
            _torch_save(best_path, checkpoint)
        summary = {
            "protocol": PIVOT_DIRECTION_PROTOCOL,
            "status": "running" if epoch < args.epochs else "complete",
            "signature": signature,
            "manifest_protocol": manifest_protocol,
            "best_epoch": best_epoch,
            "best_validation_angle_mae_degrees": best_angle,
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
            f"val_angle={angle:.4f}deg val_pivot={pivot_error:.5f} "
            f"best={best_angle:.4f}deg@{best_epoch}",
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
