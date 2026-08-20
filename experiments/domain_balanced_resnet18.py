"""Matched DB-ResNet18 control for the DB-GAR18 adaptation experiment.

The data roster, augmentation, domain batches, target-bin sampler, progress
loss weights, optimizer schedule, and terminal-only policy are shared with
DB-GAR18.  The only method differences are the matched ResNet-18 parent and
the absence of CBAM and geometry auxiliary supervision.  Warmup trains only
the scalar progress head; stage two trains ResNet layer4 plus that head.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments import robustness_degradations
from experiments.domain_balanced_geoattn_resnet18 import (
    DOMAIN_BATCH_SIZE,
    DOMAIN_PROGRESS_WEIGHT,
    EXPECTED_REAL_DEVELOPMENT_SAMPLES,
    EXPECTED_REAL_GROUPS,
    EXPECTED_SYNTHETIC_FIT_SAMPLES,
    EXPECTED_SYNTHETIC_HOLDOUT_IDS,
    LAYER4_EPOCHS,
    LAYER4_LEARNING_RATE,
    REAL_TARGET_BINS,
    TOTAL_EPOCHS,
    WARMUP_EPOCHS,
    WARMUP_LEARNING_RATE,
    WEIGHT_DECAY,
    RealProgressDataset,
    ShufflePadSampler,
    TargetBinBalancedSampler,
    _sha256_file,
    load_real_progress_samples,
    strong_real_augmentation,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    DEFAULT_COMPOSITIONAL_SPLIT,
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    IMAGENET_INITIALIZATION,
    PROTOCOL as RESNET_PARENT_PROTOCOL,
    DirectProgressDataset,
    ResNet18DirectProgress,
    _canonical_json_bytes,
    _canonical_sha256,
    _configure_reproducibility,
    _require,
    load_training_samples,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "domain_balanced_resnet18_v1"
ARCHITECTURE: Final[str] = "DB-ResNet18"
METHOD_PREFIX: Final[str] = "db_resnet18"
PARENT_ARCHITECTURE: Final[str] = "torchvision_resnet18_imagenet1k_v1_sigmoid_scalar"


def set_fine_tune_stage(
    model: ResNet18DirectProgress, stage: str
) -> tuple[str, ...]:
    """Apply the matched two-stage head/layer4 schedule."""

    _require(stage in ("progress_head", "layer4"), "unknown DB-ResNet18 stage")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad_(True)
    if stage == "layer4":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad_(True)
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    _require(bool(names), "DB-ResNet18 stage has no trainable parameters")
    return names


def domain_balanced_progress_objective(
    synthetic_predictions: torch.Tensor,
    synthetic_targets: torch.Tensor,
    real_predictions: torch.Tensor,
    real_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The same equal-domain progress terms used by DB-GAR18."""

    synthetic_progress = F.smooth_l1_loss(
        synthetic_predictions, synthetic_targets, beta=0.05
    )
    real_progress = F.smooth_l1_loss(real_predictions, real_targets, beta=0.05)
    total = DOMAIN_PROGRESS_WEIGHT * synthetic_progress + DOMAIN_PROGRESS_WEIGHT * real_progress
    return total, {
        "synthetic_progress": synthetic_progress,
        "real_progress": real_progress,
    }


def _train_epoch(
    model: ResNet18DirectProgress,
    synthetic_loader: DataLoader,
    real_loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    _require(len(synthetic_loader) == len(real_loader), "domain loader lengths differ")
    model.train()
    use_amp = device.type == "cuda"
    totals = {
        "loss": 0.0,
        "synthetic_progress": 0.0,
        "real_progress": 0.0,
        "synthetic_nmae": 0.0,
        "real_nmae": 0.0,
    }
    steps = synthetic_samples = real_samples = 0
    for synthetic_raw, real_raw in zip(synthetic_loader, real_loader, strict=True):
        synthetic_images, synthetic_targets = synthetic_raw
        synthetic_images = synthetic_images.to(device, non_blocking=use_amp)
        synthetic_targets = synthetic_targets.to(device, non_blocking=use_amp)
        real_images = real_raw["image"].to(device, non_blocking=use_amp)
        real_targets = real_raw["progress"].to(device, non_blocking=use_amp)
        synthetic_count = int(synthetic_targets.numel())
        real_count = int(real_targets.numel())
        _require(
            synthetic_count == real_count == DOMAIN_BATCH_SIZE,
            "domain batch is not exactly balanced",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            predictions = model(torch.cat((synthetic_images, real_images), dim=0))
            synthetic_predictions = predictions[:synthetic_count]
            real_predictions = predictions[synthetic_count:]
            loss, components = domain_balanced_progress_objective(
                synthetic_predictions,
                synthetic_targets,
                real_predictions,
                real_targets,
            )
        _require(bool(torch.isfinite(loss)), "DB-ResNet18 loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 5.0
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        totals["synthetic_progress"] += float(components["synthetic_progress"].detach().cpu())
        totals["real_progress"] += float(components["real_progress"].detach().cpu())
        totals["synthetic_nmae"] += float(
            torch.abs(synthetic_predictions.detach() - synthetic_targets).sum().cpu()
        )
        totals["real_nmae"] += float(
            torch.abs(real_predictions.detach() - real_targets).sum().cpu()
        )
        steps += 1
        synthetic_samples += synthetic_count
        real_samples += real_count
    _require(steps > 0 and synthetic_samples == real_samples, "empty or imbalanced epoch")
    return {
        "loss": totals["loss"] / steps,
        "synthetic_progress": totals["synthetic_progress"] / steps,
        "real_progress": totals["real_progress"] / steps,
        "synthetic_nmae": totals["synthetic_nmae"] / synthetic_samples,
        "real_nmae": totals["real_nmae"] / real_samples,
        "steps": float(steps),
        "synthetic_samples": float(synthetic_samples),
        "real_samples": float(real_samples),
    }


def _load_parent_model(
    checkpoint_path: Path,
) -> tuple[ResNet18DirectProgress, Mapping[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"ResNet parent checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "ResNet parent checkpoint is not an object")
    _require(
        checkpoint.get("protocol") == RESNET_PARENT_PROTOCOL,
        "ResNet parent protocol mismatch",
    )
    _require(
        checkpoint.get("architecture") == PARENT_ARCHITECTURE
        and checkpoint.get("pretrained_weights") == IMAGENET_INITIALIZATION,
        "parent is not the matched ImageNet ResNet-18",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "ResNet parent state is missing")
    model = ResNet18DirectProgress(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def train(
    *,
    parent_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    real_manifest_path: Path,
    real_labels_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    """Run one fixed terminal DB-ResNet18 fine-tuning seed."""

    _require(workers >= 0, "workers must be non-negative")
    synthetic_samples, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    real_samples = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "DB-ResNet18 requires the frozen 14,442/1,558 scene-disjoint SyncG roster",
    )
    _require(
        len(real_samples) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_samples}) == EXPECTED_REAL_GROUPS,
        "DB-ResNet18 requires the frozen 434-row/11-group real-development roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent = _load_parent_model(parent_checkpoint_path)
    model = model.to(device)
    synthetic_dataset = DirectProgressDataset(
        synthetic_samples, training=True, seed=seed
    )
    real_dataset = RealProgressDataset(
        real_samples, training=True, seed=seed + 10_000, preload=True
    )
    padded_samples = int(
        math.ceil(len(synthetic_dataset) / DOMAIN_BATCH_SIZE) * DOMAIN_BATCH_SIZE
    )
    stages = (
        ("progress_head", WARMUP_EPOCHS, WARMUP_LEARNING_RATE),
        ("layer4", LAYER4_EPOCHS, LAYER4_LEARNING_RATE),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    global_epoch = 0
    stage_trainable: dict[str, list[str]] = {}
    for stage, stage_epochs, learning_rate in stages:
        trainable = set_fine_tune_stage(model, stage)
        stage_trainable[stage] = list(trainable)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )
        for stage_epoch in range(stage_epochs):
            synthetic_dataset.set_epoch(global_epoch)
            real_dataset.set_epoch(global_epoch)
            synthetic_sampler = ShufflePadSampler(
                len(synthetic_dataset),
                batch_size=DOMAIN_BATCH_SIZE,
                seed=seed + global_epoch,
            )
            real_sampler = TargetBinBalancedSampler(
                [sample.normalized_target for sample in real_samples],
                num_samples=padded_samples,
                bins=REAL_TARGET_BINS,
                seed=seed + 100_000 + global_epoch,
            )
            synthetic_loader = DataLoader(
                synthetic_dataset,
                batch_size=DOMAIN_BATCH_SIZE,
                sampler=synthetic_sampler,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                drop_last=False,
            )
            real_loader = DataLoader(
                real_dataset,
                batch_size=DOMAIN_BATCH_SIZE,
                sampler=real_sampler,
                num_workers=0,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            metrics = _train_epoch(
                model,
                synthetic_loader,
                real_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
            )
            row = {
                "epoch": global_epoch + 1,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            scheduler.step()
            global_epoch += 1
    _require(global_epoch == TOTAL_EPOCHS, "fixed epoch schedule drift")
    parent_source = Path(parent_checkpoint_path).resolve()
    synthetic_manifest = Path(synthetic_manifest_path).resolve()
    synthetic_split = Path(synthetic_split_path).resolve()
    real_manifest = Path(real_manifest_path).resolve()
    real_labels = Path(real_labels_path).resolve()
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "architecture_detail": (
            "matched direct ResNet-18 initialized from the scene-fit terminal; "
            "fixed domain-balanced progress-only adaptation"
        ),
        "pretrained_weights": IMAGENET_INITIALIZATION,
        "seed": int(seed),
        "image_size": IMAGE_SIZE,
        "parent_checkpoint": str(parent_source),
        "parent_checkpoint_sha256": _sha256_file(parent_source),
        "parent_seed": int(parent["seed"]),
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": _sha256_file(synthetic_manifest),
        "synthetic_split": str(synthetic_split),
        "synthetic_split_sha256": _sha256_file(synthetic_split),
        "synthetic_fit_samples": len(synthetic_samples),
        "synthetic_holdout_samples": len(roster.validation_ids),
        "synthetic_fit_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "synthetic_holdout_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
        "synthetic_holdout_access_during_training": (
            "IDs/count only; no target, bbox, image path, or image"
        ),
        "real_manifest": str(real_manifest),
        "real_manifest_sha256": _sha256_file(real_manifest),
        "real_labels": str(real_labels),
        "real_labels_sha256": _sha256_file(real_labels),
        "real_development_samples": len(real_samples),
        "real_groups": len({sample.group_id for sample in real_samples}),
        "real_sample_ids_sha256": _canonical_sha256(
            sorted(sample.sample_id for sample in real_samples)
        ),
        "schedule": {
            "checkpoint_selection": "terminal_fixed_epoch",
            "total_epochs": TOTAL_EPOCHS,
            "domain_batch_size_each": DOMAIN_BATCH_SIZE,
            "total_batch_size": 2 * DOMAIN_BATCH_SIZE,
            "stages": [
                {
                    "name": "progress_head",
                    "epochs": WARMUP_EPOCHS,
                    "learning_rate": WARMUP_LEARNING_RATE,
                },
                {
                    "name": "layer4",
                    "epochs": LAYER4_EPOCHS,
                    "learning_rate": LAYER4_LEARNING_RATE,
                },
            ],
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "per-stage CosineAnnealingLR",
            "gradient_clip_norm": 5.0,
            "trainable_parameter_names": stage_trainable,
        },
        "domain_balance": {
            "progress_weights": {
                "synthetic": DOMAIN_PROGRESS_WEIGHT,
                "real": DOMAIN_PROGRESS_WEIGHT,
            },
            "synthetic_epoch_sampling": (
                "all fit rows once plus deterministic padding to a full domain batch"
            ),
            "synthetic_samples_each_epoch": padded_samples,
            "real_epoch_sampling": (
                "replacement, round-robin across nonempty fixed-width target bins"
            ),
            "real_samples_each_epoch": padded_samples,
            "real_target_bins": REAL_TARGET_BINS,
            "geometry_auxiliary_supervision": False,
        },
        "real_augmentation": strong_real_augmentation().__dict__,
        "synthetic_augmentation": "matched_cagh_v5_photo_and_geometry",
        "history": history,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "method": f"{METHOD_PREFIX}_seed_{seed}",
        "epochs": TOTAL_EPOCHS,
        "synthetic_fit_samples": len(synthetic_samples),
        "real_development_samples": len(real_samples),
        "terminal_synthetic_nmae": history[-1]["train"]["synthetic_nmae"],
        "terminal_real_nmae": history[-1]["train"]["real_nmae"],
    }


def load_checkpoint_predictor(
    checkpoint_path: Path, *, device_name: str
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"DB-ResNet18 checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "DB-ResNet18 checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "DB-ResNet18 protocol mismatch")
    _require(checkpoint.get("architecture") == ARCHITECTURE, "DB-ResNet18 architecture mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "DB-ResNet18 state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = ResNet18DirectProgress(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    seed = int(checkpoint["seed"])

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        batch = torch.stack(
            [
                normalized_rgb_tensor(direct_resize_whole_roi(image, size=IMAGE_SIZE))
                for image in images_bgr
            ]
        ).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        result = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result),
            "DB-ResNet18 returned invalid progress",
        )
        return result

    return f"{METHOD_PREFIX}_seed_{seed}", predict


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
    predictor_loader: Callable[..., tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]] = load_checkpoint_predictor,
) -> int:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    method, predictor = predictor_loader(checkpoint_path, device_name=device_name)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            images: list[np.ndarray] = []
            hashes: list[str] = []
            for condition in selected:
                image, _metadata = robustness_degradations.apply_degradation(
                    clean, condition, sample_id=source.sample_id, seed=ROBUSTNESS_SEED
                )
                image = np.ascontiguousarray(image)
                images.append(image)
                hashes.append(canonical_roi_pixel_sha256(image))
            try:
                values: list[float | None] = [float(value) for value in predictor(images)]
                _require(len(values) == len(images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None and math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in values
                    ),
                    "prediction outside [0,1]",
                )
                failures: list[str | None] = [None] * len(images)
            except Exception as exc:
                values = [None] * len(images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(images)
            for condition, condition_hash, progress, failure in zip(
                selected, hashes, values, failures, strict=True
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": condition_hash,
                }
                _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--parent-checkpoint", type=Path, required=True)
    training.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--synthetic-split", type=Path, default=DEFAULT_COMPOSITIONAL_SPLIT)
    training.add_argument("--real-manifest", type=Path, required=True)
    training.add_argument("--real-labels", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--workers", type=int, default=4)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--conditions", choices=("all", "clean"), default="all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = train(
            parent_checkpoint_path=args.parent_checkpoint,
            synthetic_manifest_path=args.synthetic_manifest,
            synthetic_split_path=args.synthetic_split,
            real_manifest_path=args.real_manifest,
            real_labels_path=args.real_labels,
            output_path=args.output,
            seed=args.seed,
            device_name=args.device,
            workers=args.workers,
        )
    else:
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        count = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            conditions=conditions,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "rows": count,
            "conditions": list(conditions),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "PROTOCOL",
    "domain_balanced_progress_objective",
    "load_checkpoint_predictor",
    "main",
    "run_prediction",
    "set_fine_tune_stage",
    "train",
]
