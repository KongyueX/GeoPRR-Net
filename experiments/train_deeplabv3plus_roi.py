"""Train DeepLabV3+-ResNet50 on the matched SyncG canonical ROI split."""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.deeplabv3plus_roi import (
    DeepLabV3PlusROI,
    SyncGPointerSegmentationDataset,
    binary_segmentation_loss,
    segmentation_batch_metrics,
)
from experiments.roi_geometry_comparison import write_json
from experiments.train_vdn_syncg import _matched_geoprr_partition
from experiments.vdn_baseline import load_syncg_manifest, seed_worker, set_random_seed


METHOD = "DeepLabV3+-ResNet50-ROI"


def _subset(values: Sequence[Any], limit: int | None, *, seed: int) -> list[Any]:
    selected = list(values)
    if limit is None or limit >= len(selected):
        return selected
    if limit < 1:
        raise ValueError("subset limit must be positive")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(selected)), int(limit)))
    return [selected[index] for index in indices]


def _epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "samples": 0.0}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, target, _sample_ids in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                logits = model(images)
                loss = binary_segmentation_loss(logits.float(), target.float())
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            iou, dice = segmentation_batch_metrics(logits.float(), target.float())
            count = int(images.shape[0])
            totals["samples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["iou"] += iou * count
            totals["dice"] += dice * count
    count = max(totals.pop("samples"), 1.0)
    return {key: value / count for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)

    samples, _protocol = load_syncg_manifest(
        Path(args.manifest), expected_split="train"
    )
    train_samples, validation_samples = _matched_geoprr_partition(
        samples, Path(args.outer_split).resolve()
    )
    train_samples = _subset(train_samples, args.train_limit, seed=args.seed)
    validation_samples = _subset(
        validation_samples, args.validation_limit, seed=args.seed + 10_000
    )
    training_dataset = SyncGPointerSegmentationDataset(
        train_samples, image_size=args.image_size, training=True
    )
    validation_dataset = SyncGPointerSegmentationDataset(
        validation_samples, image_size=args.image_size, training=False
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    common = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.workers),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": bool(args.workers > 0),
    }
    training_loader = DataLoader(
        training_dataset, shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)

    model = DeepLabV3PlusROI(imagenet_pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(args.epochs), 1), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() or (output_dir / "best.pt").exists():
        raise FileExistsError(f"training checkpoint already exists in {output_dir}")

    history: list[dict[str, Any]] = []
    best_dice = -1.0
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        training_metrics = _epoch(
            model,
            training_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
        )
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": training_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        payload = {
            "method": METHOD,
            "seed": int(args.seed),
            "epoch": epoch,
            "image_size": int(args.image_size),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "configuration": vars(args),
        }
        torch.save(payload, output_dir / "last.pt")
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            best_epoch = epoch
            torch.save(payload, output_dir / "best.pt")
        print(
            json.dumps(
                {
                    "method": METHOD,
                    "epoch": epoch,
                    "train_loss": training_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_iou": validation_metrics["iou"],
                    "validation_dice": validation_metrics["dice"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "status": "complete",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "subset_run": bool(args.train_limit is not None or args.validation_limit is not None),
        "checkpoint_selection": "highest fixed-inner-validation Dice",
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "history": history,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/syncg_train.jsonl"))
    parser.add_argument("--outer-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.image_size < 64 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid epoch, image-size, batch-size, or workers")
    summary = train(args)
    print(json.dumps({key: summary[key] for key in ("status", "method", "seed", "best_epoch")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["METHOD", "build_parser", "main", "train"]
