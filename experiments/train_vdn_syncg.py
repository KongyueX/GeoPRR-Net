"""Retrain the pinned VDN architecture on the official SyncG train split."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PINNED_COMMIT,
    VDN_PROTOCOL,
    SyncGVDNDataset,
    angular_error_degrees,
    build_vdn_model,
    grouped_train_val_split,
    load_official_resnet18_initialization,
    load_syncg_manifest,
    predict_directions,
    sample_ids_hash,
    seed_worker,
    set_random_seed,
    sha256_file,
    verify_vdn_source,
)
from experiments.sgca_syncg_internal_pilot import (
    DEV_SCENES as GEOPRR_INNER_DEV_SCENES,
    FIT_SAMPLES as GEOPRR_FIT_SAMPLES,
    FIT_SCENES as GEOPRR_FIT_SCENES,
    INTERNAL_DEV_SCENE_STEMS,
    TRAIN_SCENES as GEOPRR_INNER_TRAIN_SCENES,
)


TRAIN_SAMPLE_ORDER_SHA256_PROTOCOL = (
    "sha256_length_prefixed_utf8_sample_ids_in_observed_batch_order_v1"
)


def sample_order_sha256(sample_ids) -> str:
    """Hash sample IDs in their observed order with unambiguous framing."""

    digest = hashlib.sha256()
    count = 0
    for sample_id in sample_ids:
        encoded = str(sample_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        count += 1
    if count <= 0:
        raise ValueError("cannot hash an empty VDN sample order")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=Path("artifacts/vendor/VectorDetectionNetwork"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/vdn_syncg/seed_20260720"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "Adam weight decay; the pinned official get_optimizer ignores the "
            "YAML WD field, so the faithful default is zero"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--scale-factor", type=float, default=0.02)
    parser.add_argument("--rotation-factor", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--outer-split",
        type=Path,
        help=(
            "optional scene-disjoint outer split; required with "
            "--matched-geoprr-split"
        ),
    )
    parser.add_argument(
        "--matched-geoprr-split",
        action="store_true",
        help=(
            "train only on the GeoPRR outer-fit roster and use its fixed "
            "117/14-scene inner train/validation partition"
        ),
    )
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


def _training_signature(
    args: argparse.Namespace,
    *,
    vdn_commit: str,
    train_samples,
    validation_samples,
    initialization_checkpoint: Path | None,
) -> dict[str, Any]:
    protocol_path = args.manifest.with_name(args.manifest.name + ".protocol.json")
    return {
        "protocol": VDN_PROTOCOL,
        "vdn_source_commit": vdn_commit,
        "vdn_model_source_sha256": sha256_file(
            args.vdn_source / "libs" / "models" / "vdn_model.py"
        ),
        "adapter_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "architecture": "official VDN ResNet-18 + 3x deconvolution heads",
        "image_size": int(args.image_size),
        "heatmap_size": int(args.image_size // 4),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "validation_fraction": float(args.validation_fraction),
        "scale_factor": float(args.scale_factor),
        "rotation_factor": float(args.rotation_factor),
        "imagenet_pretrained": not args.no_imagenet_pretrained,
        "imagenet_initialization_checkpoint": (
            str(initialization_checkpoint) if initialization_checkpoint else None
        ),
        "imagenet_initialization_sha256": (
            sha256_file(initialization_checkpoint) if initialization_checkpoint else None
        ),
        "optimizer": "Adam",
        "optimizer_fidelity_note": (
            "official get_optimizer does not pass the YAML WD value to Adam"
        ),
        "mixed_precision": str(args.device).startswith("cuda") and not args.no_amp,
        "grad_scaler_initial_scale": 512.0,
        "seed": int(args.seed),
        "matched_geoprr_split": bool(args.matched_geoprr_split),
        "outer_split": str(args.outer_split) if args.outer_split else None,
        "diagnostic_limit": args.limit,
    }


def _scene_stem(sample) -> str:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise ValueError(f"{sample.sample_id}: metadata is not a mapping")
    scene_name = str(metadata.get("scene_name") or "")
    if not scene_name:
        raise ValueError(f"{sample.sample_id}: scene_name is empty")
    return Path(scene_name).stem


def _matched_geoprr_partition(samples, outer_split: Path):
    payload = json.loads(outer_split.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "syncg_scene_stem_disjoint_clean_v1"
        or payload.get("scene_disjoint") is not True
    ):
        raise ValueError(f"not the GeoPRR scene-disjoint split: {outer_split}")
    fit_ids = {str(value) for value in payload.get("train_sample_ids") or []}
    holdout_ids = {
        str(value) for value in payload.get("validation_sample_ids") or []
    }
    all_ids = {sample.sample_id for sample in samples}
    if (
        len(fit_ids) != GEOPRR_FIT_SAMPLES
        or len(holdout_ids) != 1_558
        or fit_ids & holdout_ids
        or fit_ids | holdout_ids != all_ids
    ):
        raise ValueError("GeoPRR outer split does not partition SyncG train")

    fit_samples = [sample for sample in samples if sample.sample_id in fit_ids]
    fit_scenes = {_scene_stem(sample) for sample in fit_samples}
    dev_scenes = set(INTERNAL_DEV_SCENE_STEMS)
    if (
        len(fit_samples) != GEOPRR_FIT_SAMPLES
        or len(fit_scenes) != GEOPRR_FIT_SCENES
        or len(dev_scenes) != GEOPRR_INNER_DEV_SCENES
        or not dev_scenes < fit_scenes
    ):
        raise ValueError("GeoPRR fit or fixed inner-development roster differs")

    train_samples = [
        sample for sample in fit_samples if _scene_stem(sample) not in dev_scenes
    ]
    validation_samples = [
        sample for sample in fit_samples if _scene_stem(sample) in dev_scenes
    ]
    train_scenes = {_scene_stem(sample) for sample in train_samples}
    validation_scenes = {_scene_stem(sample) for sample in validation_samples}
    if (
        len(train_samples) != 12_866
        or len(validation_samples) != 1_576
        or len(train_scenes) != GEOPRR_INNER_TRAIN_SCENES
        or len(validation_scenes) != GEOPRR_INNER_DEV_SCENES
        or train_scenes & validation_scenes
    ):
        raise ValueError("GeoPRR fixed inner partition differs")
    return train_samples, validation_samples


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    vector_weight: float,
    epoch: int,
    record_scaler_trace: bool = False,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    heatmap_loss_total = 0.0
    vector_loss_total = 0.0
    samples = 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    skipped_batch_indices: list[int] = []
    observed_sample_ids: list[str] = []
    scaler_start_state = (
        copy.deepcopy(scaler.state_dict()) if record_scaler_trace else None
    )
    progress = tqdm(loader, desc=f"VDN train {epoch}", leave=False, dynamic_ncols=True)
    for batch_index, (
        images,
        target_heatmaps,
        target_vectors,
        _,
        sample_ids,
    ) in enumerate(progress):
        if len(sample_ids) != int(images.shape[0]):
            raise RuntimeError("VDN batch sample IDs do not match the image batch")
        observed_sample_ids.extend(str(sample_id) for sample_id in sample_ids)
        images = images.to(device, non_blocking=True)
        target_heatmaps = target_heatmaps.to(device, non_blocking=True)
        target_vectors = target_vectors.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            output_heatmaps, output_vectors = model(images)
        heatmap_loss = F.mse_loss(output_heatmaps.float(), target_heatmaps.float())
        vector_loss = F.mse_loss(output_vectors.float(), target_vectors.float())
        loss = heatmap_loss + float(vector_weight) * vector_loss
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_optimizer_steps += 1
            if record_scaler_trace:
                skipped_batch_indices.append(batch_index)
        else:
            optimizer_steps += 1
        count = int(images.shape[0])
        samples += count
        total_loss += float(loss.detach()) * count
        heatmap_loss_total += float(heatmap_loss.detach()) * count
        vector_loss_total += float(vector_loss.detach()) * count
        progress.set_postfix(loss=f"{total_loss / samples:.5f}")
    result = {
        "loss": total_loss / max(samples, 1),
        "heatmap_loss": heatmap_loss_total / max(samples, 1),
        "vector_loss": vector_loss_total / max(samples, 1),
        "samples": samples,
        "vector_weight": float(vector_weight),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "sample_order_sha256": sample_order_sha256(observed_sample_ids),
    }
    if record_scaler_trace:
        result.update(
            {
                "scaler_start_state": scaler_start_state,
                "scaler_skipped_batch_indices": skipped_batch_indices,
                "scaler_end_state": copy.deepcopy(scaler.state_dict()),
            }
        )
    return result


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    heatmap_loss_total = 0.0
    vector_loss_total = 0.0
    errors: list[float] = []
    confidence: list[float] = []
    valid_count = 0
    samples = 0
    for images, target_heatmaps, target_vectors, target_direction, _ in tqdm(
        loader,
        desc="VDN validation",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        target_heatmaps = target_heatmaps.to(device, non_blocking=True)
        target_vectors = target_vectors.to(device, non_blocking=True)
        target_direction = target_direction.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            output_heatmaps, output_vectors = model(images)
        heatmap_loss = F.mse_loss(output_heatmaps.float(), target_heatmaps.float())
        vector_loss = F.mse_loss(output_vectors.float(), target_vectors.float())
        loss = heatmap_loss + vector_loss
        predicted, peak, valid = predict_directions(
            output_heatmaps.float(), output_vectors.float()
        )
        angle = angular_error_degrees(predicted, target_direction.float())
        errors.extend(angle[valid].detach().cpu().tolist())
        confidence.extend(peak.detach().cpu().tolist())
        valid_count += int(valid.sum().item())
        count = int(images.shape[0])
        samples += count
        total_loss += float(loss) * count
        heatmap_loss_total += float(heatmap_loss) * count
        vector_loss_total += float(vector_loss) * count
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "loss": total_loss / max(samples, 1),
        "heatmap_loss": heatmap_loss_total / max(samples, 1),
        "vector_loss": vector_loss_total / max(samples, 1),
        "samples": samples,
        "valid_directions": valid_count,
        "direction_coverage": valid_count / max(samples, 1),
        "angle_mae_degrees": float(np.mean(error_array)) if errors else math.inf,
        "angle_median_degrees": float(np.median(error_array)) if errors else math.inf,
        "angle_acc_1deg": float(np.mean(error_array <= 1.0)) if errors else 0.0,
        "angle_acc_3deg": float(np.mean(error_array <= 3.0)) if errors else 0.0,
        "angle_acc_5deg": float(np.mean(error_array <= 5.0)) if errors else 0.0,
        "mean_heatmap_peak": float(np.mean(confidence)) if confidence else math.nan,
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.vdn_source = args.vdn_source.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.outer_split is not None:
        args.outer_split = args.outer_split.resolve()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.image_size % 32 != 0:
        raise ValueError("--image-size must be divisible by 32")
    if args.matched_geoprr_split and args.outer_split is None:
        raise ValueError("--matched-geoprr-split requires --outer-split")
    if args.matched_geoprr_split and args.limit is not None:
        raise ValueError("--limit cannot be used with --matched-geoprr-split")
    set_random_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    vdn_commit = verify_vdn_source(args.vdn_source)
    samples, manifest_protocol = load_syncg_manifest(
        args.manifest,
        expected_split="train",
        limit=args.limit,
    )
    if args.matched_geoprr_split:
        train_samples, validation_samples = _matched_geoprr_partition(
            samples,
            args.outer_split,
        )
    else:
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    initialization_checkpoint = None
    if not args.no_imagenet_pretrained:
        _, initialization_checkpoint = load_official_resnet18_initialization()
    signature = _training_signature(
        args,
        vdn_commit=vdn_commit,
        train_samples=train_samples,
        validation_samples=validation_samples,
        initialization_checkpoint=initialization_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SyncGVDNDataset(
        train_samples,
        image_size=args.image_size,
        training=True,
        scale_factor=args.scale_factor,
        rotation_factor=args.rotation_factor,
    )
    validation_dataset = SyncGVDNDataset(
        validation_samples,
        image_size=args.image_size,
        training=False,
        scale_factor=args.scale_factor,
        rotation_factor=args.rotation_factor,
    )
    generator = torch.Generator().manual_seed(args.seed)
    common_loader = {
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
        **common_loader,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        **common_loader,
    )

    last_path = args.output_dir / "last.pt"
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_angle = math.inf
    best_epoch = 0
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") != signature:
            raise ValueError("VDN resume signature mismatch")
        model = build_vdn_model(
            args.vdn_source,
            image_size=args.image_size,
            imagenet_pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_angle = float(checkpoint.get("best_angle", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history") or [])
    else:
        model = build_vdn_model(
            args.vdn_source,
            image_size=args.image_size,
            imagenet_pretrained=not args.no_imagenet_pretrained,
        )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    milestones = sorted(
        {
            max(1, int(round(args.epochs * 0.70))),
            max(1, int(round(args.epochs * 0.95))),
        }
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.1,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=512.0,
    )
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        vector_weight = (
            1.0
            if args.epochs == 1
            else float(epoch - 1) / float(args.epochs - 1)
        )
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            vector_weight=vector_weight,
            epoch=epoch,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        if train_metrics["optimizer_steps"] <= 0:
            raise RuntimeError("all VDN optimizer steps were skipped in this epoch")
        current_angle = float(validation_metrics["angle_mae_degrees"])
        improved = current_angle < best_angle
        if improved:
            best_angle = current_angle
            best_epoch = epoch
        epoch_result = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
            "best": improved,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(epoch_result)
        state = {
            "schema_version": 1,
            "signature": signature,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_angle": best_angle,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
        }
        _torch_save(last_path, state)
        if improved:
            _torch_save(
                best_path,
                {
                    "schema_version": 1,
                    "signature": signature,
                    "epoch": epoch,
                    "best_angle": best_angle,
                    "model_state": model.state_dict(),
                },
            )
        summary = {
            "protocol": VDN_PROTOCOL,
            "status": "running" if epoch < args.epochs else "complete",
            "signature": signature,
            "manifest_protocol": manifest_protocol,
            "best_epoch": best_epoch,
            "best_validation_angle_mae_degrees": best_angle,
            "history": history,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "amp": amp_enabled,
            },
        }
        _json_write(args.output_dir / "summary.json", summary)
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_angle_mae={current_angle:.4f}deg "
            f"best={best_angle:.4f}deg"
        )

    if not best_path.is_file():
        raise RuntimeError("VDN training completed without a best checkpoint")
    print(best_path)


if __name__ == "__main__":
    main()
