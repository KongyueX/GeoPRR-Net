"""Strict 2x2 component ablation for DB-GAR18.

The two frozen factors are CBAM attention and train-only geometry auxiliary
supervision.  Existing DB-ResNet18 is the off/off control and existing
DB-GAR18 is the on/on arm.  This isolated runner adds only the missing arms:

``cbam_only``
    Eight-block CBAM ResNet-18 with a scalar progress head and no geometry
    head or geometry loss.

``geometry_aux_only``
    Plain ResNet-18 with the same scalar progress and eight-value geometry
    heads as DB-GAR18, but no CBAM modules.

Each new arm first trains its own scene-disjoint SyncG source parent with the
same fixed 30-epoch recipe as the corresponding existing source models.  It
then uses the exact DB-18 32+32 domain sampler and terminal 2+6 epoch schedule.
This prevents either ablation from inheriting a parent representation already
trained with the component that is declared absent.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

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
    SYNTHETIC_AUX_REPLAY_SCALE,
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
from experiments.geoattn_resnet18_progress import (
    CBAM,
    CBAMBasicBlock,
    DIRECTION_LOSS_WEIGHT,
    PIVOT_LOSS_WEIGHT,
    PROGRESS_LOSS_WEIGHT,
    REFERENCE_LOSS_WEIGHT,
    GeoAttnProgressDataset,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import IMAGENET_WEIGHTS
from experiments.resnet18_direct_progress import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPOSITIONAL_SPLIT,
    DEFAULT_EPOCHS,
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    IMAGENET_INITIALIZATION,
    DirectProgressDataset,
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
from experiments.run_cagh_v5_solver_gated_screen import augmentation_config
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


SOURCE_PROTOCOL: Final[str] = "db_gar18_component_ablation_source_v1"
PROTOCOL: Final[str] = "domain_balanced_db_gar18_component_ablation_v1"
ARM_CBAM_ONLY: Final[str] = "cbam_only"
ARM_GEOMETRY_AUX_ONLY: Final[str] = "geometry_aux_only"
ARMS: Final[tuple[str, ...]] = (ARM_CBAM_ONLY, ARM_GEOMETRY_AUX_ONLY)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm: str
    source_architecture: str
    adapted_architecture: str
    method_prefix: str
    use_cbam: bool
    use_geometry_aux: bool
    warmup_stage: str


ARM_SPECS: Final[dict[str, ArmSpec]] = {
    ARM_CBAM_ONLY: ArmSpec(
        arm=ARM_CBAM_ONLY,
        source_architecture="CBAM-Only-ResNet18",
        adapted_architecture="DB-CBAM-Only-ResNet18",
        method_prefix="db_cbam_only18",
        use_cbam=True,
        use_geometry_aux=False,
        warmup_stage="heads_cbam",
    ),
    ARM_GEOMETRY_AUX_ONLY: ArmSpec(
        arm=ARM_GEOMETRY_AUX_ONLY,
        source_architecture="GeometryAux-Only-ResNet18",
        adapted_architecture="DB-GeometryAux-Only-ResNet18",
        method_prefix="db_geometry_aux_only18",
        use_cbam=False,
        use_geometry_aux=True,
        warmup_stage="heads_geometry",
    ),
}


def arm_spec(arm: str) -> ArmSpec:
    _require(arm in ARM_SPECS, f"unknown component-ablation arm: {arm}")
    return ARM_SPECS[arm]


class FactorAblationResNet18(nn.Module):
    """One of the two missing cells in the frozen CBAM x geometry design."""

    def __init__(self, arm: str, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        self.spec = arm_spec(arm)
        self.imagenet_pretrained = bool(imagenet_pretrained)
        backbone = resnet18(
            weights=IMAGENET_WEIGHTS if self.imagenet_pretrained else None
        )
        if self.spec.use_cbam:
            for layer_name in ("layer1", "layer2", "layer3", "layer4"):
                layer = getattr(backbone, layer_name)
                setattr(
                    backbone,
                    layer_name,
                    nn.Sequential(*(CBAMBasicBlock(block) for block in layer)),
                )
        features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.progress_head = nn.Linear(features, 1)
        self.geometry_head: nn.Module | None
        if self.spec.use_geometry_aux:
            self.geometry_head = nn.Sequential(
                nn.Linear(features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
                nn.Linear(128, 8),
            )
        else:
            self.geometry_head = None

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.progress_head(self._features(image)).squeeze(1))

    def forward_training(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._features(image)
        output = {
            "progress": torch.sigmoid(self.progress_head(features).squeeze(1))
        }
        if self.geometry_head is not None:
            geometry = self.geometry_head(features)
            output.update(
                {
                    "pivot": torch.sigmoid(geometry[:, 0:2]),
                    "direction_sin_cos": F.normalize(
                        geometry[:, 2:4], dim=1, eps=1e-6
                    ),
                    "references": torch.sigmoid(geometry[:, 4:8]),
                }
            )
        return output


def parameter_inventory(model: FactorAblationResNet18) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    attention = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, CBAM)
        for parameter in module.parameters()
    )
    auxiliary = (
        sum(parameter.numel() for parameter in model.geometry_head.parameters())
        if model.geometry_head is not None
        else 0
    )
    return {
        "total": int(total),
        "cbam": int(attention),
        "training_only_auxiliary_head": int(auxiliary),
        "inference_parameters": int(total - auxiliary),
    }


def _geometry_terms(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pivot = F.smooth_l1_loss(outputs["pivot"], batch["pivot"], beta=0.05)
    direction = (
        1.0
        - torch.sum(
            outputs["direction_sin_cos"] * batch["direction_sin_cos"], dim=1
        )
    ).mean()
    references = F.smooth_l1_loss(
        outputs["references"], batch["references"], beta=0.05
    )
    auxiliary = (
        PIVOT_LOSS_WEIGHT * pivot
        + DIRECTION_LOSS_WEIGHT * direction
        + REFERENCE_LOSS_WEIGHT * references
    )
    return auxiliary, {
        "pivot": pivot,
        "direction": direction,
        "references": references,
        "auxiliary": auxiliary,
    }


def source_objective(
    arm: str,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact source loss for either missing factorial arm."""

    spec = arm_spec(arm)
    progress = F.smooth_l1_loss(
        outputs["progress"], batch["progress"], beta=0.05
    )
    parts: dict[str, torch.Tensor] = {"progress": progress}
    if not spec.use_geometry_aux:
        return PROGRESS_LOSS_WEIGHT * progress, parts
    auxiliary, geometry = _geometry_terms(outputs, batch)
    parts.update(geometry)
    return PROGRESS_LOSS_WEIGHT * progress + auxiliary, parts


def domain_balanced_objective(
    arm: str,
    synthetic_outputs: Mapping[str, torch.Tensor],
    synthetic_batch: Mapping[str, torch.Tensor],
    real_outputs: Mapping[str, torch.Tensor],
    real_batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Matched 0.5/0.5 domain progress plus optional synthetic geometry replay."""

    spec = arm_spec(arm)
    synthetic_progress = F.smooth_l1_loss(
        synthetic_outputs["progress"], synthetic_batch["progress"], beta=0.05
    )
    real_progress = F.smooth_l1_loss(
        real_outputs["progress"], real_batch["progress"], beta=0.05
    )
    total = (
        DOMAIN_PROGRESS_WEIGHT * synthetic_progress
        + DOMAIN_PROGRESS_WEIGHT * real_progress
    )
    parts: dict[str, torch.Tensor] = {
        "synthetic_progress": synthetic_progress,
        "real_progress": real_progress,
    }
    if spec.use_geometry_aux:
        auxiliary, geometry = _geometry_terms(synthetic_outputs, synthetic_batch)
        total = total + SYNTHETIC_AUX_REPLAY_SCALE * auxiliary
        parts.update(geometry)
    return total, parts


def set_fine_tune_stage(
    model: FactorAblationResNet18, stage: str
) -> tuple[str, ...]:
    """Apply the frozen two-stage trainability mask for one arm."""

    spec = model.spec
    _require(
        stage in (spec.warmup_stage, "layer4"),
        f"unknown {spec.arm} fine-tune stage: {stage}",
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.progress_head.parameters():
        parameter.requires_grad_(True)
    if model.geometry_head is not None:
        for parameter in model.geometry_head.parameters():
            parameter.requires_grad_(True)
    if spec.use_cbam:
        for module in model.modules():
            if isinstance(module, CBAM):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    if stage == "layer4":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad_(True)
    names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    _require(bool(names), f"{spec.arm} stage has no trainable parameters")
    return names


def _source_dataset(
    arm: str, samples: Sequence[Any], *, seed: int
) -> Dataset[Any]:
    spec = arm_spec(arm)
    if spec.use_geometry_aux:
        return GeoAttnProgressDataset(samples, training=True, seed=seed)
    return DirectProgressDataset(samples, training=True, seed=seed)


def _source_train_epoch(
    model: FactorAblationResNet18,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    use_amp = device.type == "cuda"
    totals: dict[str, float] = {"loss": 0.0, "nmae": 0.0}
    samples = 0
    for raw in loader:
        if model.spec.use_geometry_aux:
            batch = {
                key: value.to(device, non_blocking=use_amp)
                for key, value in raw.items()
            }
        else:
            images, progress = raw
            batch = {
                "image": images.to(device, non_blocking=use_amp),
                "progress": progress.to(device, non_blocking=use_amp),
            }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model.forward_training(batch["image"])
            loss, parts = source_objective(model.spec.arm, outputs, batch)
        _require(bool(torch.isfinite(loss)), "source ablation loss became non-finite")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = int(batch["progress"].numel())
        totals["loss"] += float(loss.detach().cpu()) * count
        for key, value in parts.items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value.detach().cpu()) * count
        totals["nmae"] += float(
            torch.abs(outputs["progress"].detach() - batch["progress"]).sum().cpu()
        )
        samples += count
    _require(samples > 0, "source epoch produced no samples")
    return {
        **{key: value / samples for key, value in totals.items()},
        "samples": float(samples),
    }


def train_source(
    *,
    arm: str,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    """Train one terminal 30-epoch source parent without opening holdout data."""

    spec = arm_spec(arm)
    _require(workers >= 0, "workers must be non-negative")
    fit_samples, roster = load_training_samples(manifest_path, split_path)
    _require(
        len(fit_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "factor source requires the frozen 14,442/1,558 scene-disjoint roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    dataset = _source_dataset(arm, fit_samples, seed=seed)
    model = FactorAblationResNet18(arm, imagenet_pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DEFAULT_EPOCHS
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch_index in range(DEFAULT_EPOCHS):
        dataset.set_epoch(epoch_index)  # type: ignore[attr-defined]
        loader = DataLoader(
            dataset,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(seed + epoch_index),
        )
        metrics = _source_train_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        row = {
            "epoch": epoch_index + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    manifest = Path(manifest_path).resolve()
    split = Path(split_path).resolve()
    checkpoint = {
        "schema_version": 1,
        "protocol": SOURCE_PROTOCOL,
        "architecture": spec.source_architecture,
        "arm": spec.arm,
        "factor_flags": {
            "cbam": spec.use_cbam,
            "geometry_auxiliary": spec.use_geometry_aux,
        },
        "pretrained_weights": IMAGENET_INITIALIZATION,
        "image_size": IMAGE_SIZE,
        "seed": int(seed),
        "synthetic_manifest": str(manifest),
        "synthetic_manifest_sha256": _sha256_file(manifest),
        "synthetic_split": str(split),
        "synthetic_split_sha256": _sha256_file(split),
        "split_protocol": roster.protocol,
        "scene_disjoint": roster.scene_disjoint,
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "train_sample_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "holdout_sample_ids_sha256": _canonical_sha256(
            sorted(roster.validation_ids)
        ),
        "holdout_access_during_training": (
            "IDs/count only; no target, bbox, image path, or image"
        ),
        "epochs": DEFAULT_EPOCHS,
        "checkpoint_selection": "terminal_fixed_epoch",
        "loss": {
            "progress": "smooth_l1_beta_0.05",
            "geometry_auxiliary": spec.use_geometry_aux,
            "weights": {
                "progress": PROGRESS_LOSS_WEIGHT,
                "pivot": PIVOT_LOSS_WEIGHT if spec.use_geometry_aux else 0.0,
                "direction": DIRECTION_LOSS_WEIGHT
                if spec.use_geometry_aux
                else 0.0,
                "references": REFERENCE_LOSS_WEIGHT
                if spec.use_geometry_aux
                else 0.0,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
        },
        "augmentation": {
            "name": "matched_cagh_v5_photo_and_geometry",
            "configuration": augmentation_config(),
            "co_transformed_auxiliary_labels": spec.use_geometry_aux,
        },
        "parameter_inventory": parameter_inventory(model),
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
        "stage": "source",
        "arm": arm,
        "checkpoint": str(output),
        "epochs": DEFAULT_EPOCHS,
        "terminal_train_nmae": history[-1]["train"]["nmae"],
    }


def _load_source_model(
    checkpoint_path: Path, arm: str
) -> tuple[FactorAblationResNet18, Mapping[str, Any]]:
    spec = arm_spec(arm)
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"component source checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "component source checkpoint is not an object")
    _require(checkpoint.get("protocol") == SOURCE_PROTOCOL, "source protocol mismatch")
    _require(checkpoint.get("architecture") == spec.source_architecture, "source architecture mismatch")
    _require(checkpoint.get("arm") == arm, "source arm mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "source model state is missing")
    model = FactorAblationResNet18(arm, imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def _slice_outputs(
    outputs: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in outputs.items()}


def _adapt_train_epoch(
    model: FactorAblationResNet18,
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
    totals: dict[str, float] = {
        "loss": 0.0,
        "synthetic_nmae": 0.0,
        "real_nmae": 0.0,
    }
    steps = synthetic_samples = real_samples = 0
    for synthetic_raw, real_raw in zip(
        synthetic_loader, real_loader, strict=True
    ):
        if model.spec.use_geometry_aux:
            synthetic = {
                key: value.to(device, non_blocking=use_amp)
                for key, value in synthetic_raw.items()
            }
        else:
            images, progress = synthetic_raw
            synthetic = {
                "image": images.to(device, non_blocking=use_amp),
                "progress": progress.to(device, non_blocking=use_amp),
            }
        real = {
            key: value.to(device, non_blocking=use_amp)
            for key, value in real_raw.items()
        }
        synthetic_count = int(synthetic["progress"].numel())
        real_count = int(real["progress"].numel())
        _require(
            synthetic_count == real_count == DOMAIN_BATCH_SIZE,
            "domain batch is not exactly 32+32",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model.forward_training(
                torch.cat((synthetic["image"], real["image"]), dim=0)
            )
            synthetic_outputs = _slice_outputs(outputs, 0, synthetic_count)
            real_outputs = _slice_outputs(
                outputs, synthetic_count, synthetic_count + real_count
            )
            loss, parts = domain_balanced_objective(
                model.spec.arm,
                synthetic_outputs,
                synthetic,
                real_outputs,
                real,
            )
        _require(bool(torch.isfinite(loss)), "adaptation loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        for key, value in parts.items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value.detach().cpu())
        totals["synthetic_nmae"] += float(
            torch.abs(
                synthetic_outputs["progress"].detach() - synthetic["progress"]
            ).sum().cpu()
        )
        totals["real_nmae"] += float(
            torch.abs(real_outputs["progress"].detach() - real["progress"])
            .sum()
            .cpu()
        )
        steps += 1
        synthetic_samples += synthetic_count
        real_samples += real_count
    _require(steps > 0 and synthetic_samples == real_samples, "empty or imbalanced epoch")
    result = {
        key: value / steps
        for key, value in totals.items()
        if not key.endswith("nmae")
    }
    result["synthetic_nmae"] = totals["synthetic_nmae"] / synthetic_samples
    result["real_nmae"] = totals["real_nmae"] / real_samples
    return {
        **result,
        "steps": float(steps),
        "synthetic_samples": float(synthetic_samples),
        "real_samples": float(real_samples),
    }


def adapt(
    *,
    arm: str,
    source_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    real_manifest_path: Path,
    real_labels_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    """Run the matched terminal 2+6 epoch domain-balanced adaptation."""

    spec = arm_spec(arm)
    _require(workers >= 0, "workers must be non-negative")
    synthetic_samples, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    real_samples = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "adaptation requires the frozen 14,442/1,558 SyncG roster",
    )
    _require(
        len(real_samples) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_samples})
        == EXPECTED_REAL_GROUPS,
        "adaptation requires the frozen 434-row/11-group real roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent = _load_source_model(source_checkpoint_path, arm)
    _require(int(parent["seed"]) == int(seed), "source/adaptation seed mismatch")
    model = model.to(device)
    synthetic_dataset = _source_dataset(arm, synthetic_samples, seed=seed)
    real_dataset = RealProgressDataset(
        real_samples, training=True, seed=seed + 10_000, preload=True
    )
    padded_samples = int(
        math.ceil(len(synthetic_dataset) / DOMAIN_BATCH_SIZE) * DOMAIN_BATCH_SIZE
    )
    stages = (
        (spec.warmup_stage, WARMUP_EPOCHS, WARMUP_LEARNING_RATE),
        ("layer4", LAYER4_EPOCHS, LAYER4_LEARNING_RATE),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    stage_trainable: dict[str, list[str]] = {}
    global_epoch = 0
    for stage, stage_epochs, learning_rate in stages:
        trainable = set_fine_tune_stage(model, stage)
        stage_trainable[stage] = list(trainable)
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )
        for stage_epoch in range(stage_epochs):
            synthetic_dataset.set_epoch(global_epoch)  # type: ignore[attr-defined]
            real_dataset.set_epoch(global_epoch)
            synthetic_loader = DataLoader(
                synthetic_dataset,
                batch_size=DOMAIN_BATCH_SIZE,
                sampler=ShufflePadSampler(
                    len(synthetic_dataset),
                    batch_size=DOMAIN_BATCH_SIZE,
                    seed=seed + global_epoch,
                ),
                num_workers=workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                drop_last=False,
            )
            real_loader = DataLoader(
                real_dataset,
                batch_size=DOMAIN_BATCH_SIZE,
                sampler=TargetBinBalancedSampler(
                    [sample.normalized_target for sample in real_samples],
                    num_samples=padded_samples,
                    bins=REAL_TARGET_BINS,
                    seed=seed + 100_000 + global_epoch,
                ),
                num_workers=0,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            metrics = _adapt_train_epoch(
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
    _require(global_epoch == TOTAL_EPOCHS, "fixed adaptation schedule drift")
    source = Path(source_checkpoint_path).resolve()
    synthetic_manifest = Path(synthetic_manifest_path).resolve()
    synthetic_split = Path(synthetic_split_path).resolve()
    real_manifest = Path(real_manifest_path).resolve()
    real_labels = Path(real_labels_path).resolve()
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": spec.adapted_architecture,
        "arm": spec.arm,
        "method": f"{spec.method_prefix}_seed_{seed}",
        "factor_flags": {
            "cbam": spec.use_cbam,
            "geometry_auxiliary": spec.use_geometry_aux,
        },
        "seed": int(seed),
        "image_size": IMAGE_SIZE,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256_file(source),
        "source_protocol": SOURCE_PROTOCOL,
        "source_seed": int(parent["seed"]),
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": _sha256_file(synthetic_manifest),
        "synthetic_split": str(synthetic_split),
        "synthetic_split_sha256": _sha256_file(synthetic_split),
        "synthetic_fit_samples": len(synthetic_samples),
        "synthetic_holdout_samples": len(roster.validation_ids),
        "synthetic_fit_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "synthetic_holdout_ids_sha256": _canonical_sha256(
            sorted(roster.validation_ids)
        ),
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
                    "name": spec.warmup_stage,
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
            "synthetic_auxiliary_replay_scale": (
                SYNTHETIC_AUX_REPLAY_SCALE if spec.use_geometry_aux else 0.0
            ),
            "real_geometry_supervision": False,
        },
        "real_augmentation": strong_real_augmentation().__dict__,
        "synthetic_augmentation": "matched_cagh_v5_photo_and_geometry",
        "parameter_inventory": parameter_inventory(model),
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
        "stage": "adaptation",
        "arm": arm,
        "checkpoint": str(output),
        "method": checkpoint["method"],
        "epochs": TOTAL_EPOCHS,
        "terminal_synthetic_nmae": history[-1]["train"]["synthetic_nmae"],
        "terminal_real_nmae": history[-1]["train"]["real_nmae"],
    }


def load_checkpoint_predictor(
    checkpoint_path: Path, *, device_name: str
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"component checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "component checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "component protocol mismatch")
    arm = str(checkpoint.get("arm"))
    spec = arm_spec(arm)
    _require(
        checkpoint.get("architecture") == spec.adapted_architecture,
        "component architecture mismatch",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "component model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = FactorAblationResNet18(arm, imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    method = str(checkpoint["method"])

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        batch = torch.stack(
            [
                normalized_rgb_tensor(
                    direct_resize_whole_roi(image, size=IMAGE_SIZE)
                )
                for image in images_bgr
            ]
        ).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        output = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in output),
            "component model returned invalid progress",
        )
        return output

    return method, predict


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
    predictor_loader: Callable[
        ..., tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]
    ] = load_checkpoint_predictor,
) -> int:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    method, predictor = predictor_loader(
        checkpoint_path, device_name=device_name
    )
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
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                images.append(image)
                hashes.append(canonical_roi_pixel_sha256(image))
            try:
                values: list[float | None] = [
                    float(value) for value in predictor(images)
                ]
                _require(len(values) == len(images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None
                        and math.isfinite(value)
                        and 0.0 <= value <= 1.0
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
                _require(set(row) == set(OUTPUT_KEYS), "prediction schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("train-source")
    source.add_argument("--arm", choices=ARMS, required=True)
    source.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    source.add_argument("--split", type=Path, default=DEFAULT_COMPOSITIONAL_SPLIT)
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--seed", type=int, required=True)
    source.add_argument("--device", default="cuda:0")
    source.add_argument("--workers", type=int, default=4)

    adaptation = commands.add_parser("adapt")
    adaptation.add_argument("--arm", choices=ARMS, required=True)
    adaptation.add_argument("--source-checkpoint", type=Path, required=True)
    adaptation.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_MANIFEST)
    adaptation.add_argument("--synthetic-split", type=Path, default=DEFAULT_COMPOSITIONAL_SPLIT)
    adaptation.add_argument("--real-manifest", type=Path, required=True)
    adaptation.add_argument("--real-labels", type=Path, required=True)
    adaptation.add_argument("--output", type=Path, required=True)
    adaptation.add_argument("--seed", type=int, required=True)
    adaptation.add_argument("--device", default="cuda:0")
    adaptation.add_argument("--workers", type=int, default=4)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train-source":
        result = train_source(
            arm=args.arm,
            manifest_path=args.manifest,
            split_path=args.split,
            output_path=args.output,
            seed=args.seed,
            device_name=args.device,
            workers=args.workers,
        )
    elif args.command == "adapt":
        result = adapt(
            arm=args.arm,
            source_checkpoint_path=args.source_checkpoint,
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
        count = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            conditions=args.conditions,
        )
        result = {
            "status": "complete",
            "stage": "prediction",
            "output": str(Path(args.output).resolve()),
            "rows": count,
            "conditions": list(args.conditions),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_CBAM_ONLY",
    "ARM_GEOMETRY_AUX_ONLY",
    "ARMS",
    "ARM_SPECS",
    "FactorAblationResNet18",
    "PROTOCOL",
    "SOURCE_PROTOCOL",
    "adapt",
    "arm_spec",
    "domain_balanced_objective",
    "load_checkpoint_predictor",
    "main",
    "parameter_inventory",
    "run_prediction",
    "set_fine_tune_stage",
    "source_objective",
    "train_source",
]
