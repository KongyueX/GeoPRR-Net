"""DB-GAR18: fixed domain-balanced adaptation of GeoAttn-ResNet18.

This runner adapts one already-trained GeoAttn-ResNet18 checkpoint with two
training domains only:

* the 14,442 scene-disjoint SyncG fit rows, retaining geometry supervision;
* the 434 xiangmu2 real-development canonical ROIs, using progress labels only.

Every optimizer step contains equally many synthetic and real images.  The two
progress losses receive equal weight; geometry auxiliary losses are evaluated
only on the synthetic half.  The real half is sampled with a deterministic
fixed-width target-bin sampler, while every synthetic fit row is seen once per
epoch (plus at most one batch of deterministic padding).  Training is a single
predeclared terminal schedule: two epochs for the heads and CBAM modules,
followed by six epochs that additionally unfreeze ResNet layer4.  No validation
metric is read and no checkpoint is selected by performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments import robustness_degradations
from experiments.geoattn_resnet18_progress import (
    ARCHITECTURE as GAR_ARCHITECTURE,
    DIRECTION_LOSS_WEIGHT,
    PIVOT_LOSS_WEIGHT,
    PROTOCOL as GAR_PROTOCOL,
    REFERENCE_LOSS_WEIGHT,
    CBAM,
    GeoAttnProgressDataset,
    GeoAttnResNet18,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    DEFAULT_COMPOSITIONAL_SPLIT,
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
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
    ManifestRow,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "domain_balanced_geoattn_resnet18_v1"
ARCHITECTURE: Final[str] = "DB-GAR18"
METHOD_PREFIX: Final[str] = "db_gar18"

# One fixed candidate, declared before evaluation.
DOMAIN_BATCH_SIZE: Final[int] = 32
WARMUP_EPOCHS: Final[int] = 2
LAYER4_EPOCHS: Final[int] = 6
TOTAL_EPOCHS: Final[int] = WARMUP_EPOCHS + LAYER4_EPOCHS
WARMUP_LEARNING_RATE: Final[float] = 1e-4
LAYER4_LEARNING_RATE: Final[float] = 3e-5
WEIGHT_DECAY: Final[float] = 1e-4
REAL_TARGET_BINS: Final[int] = 10
DOMAIN_PROGRESS_WEIGHT: Final[float] = 0.5
SYNTHETIC_AUX_REPLAY_SCALE: Final[float] = 0.5
EXPECTED_SYNTHETIC_FIT_SAMPLES: Final[int] = 14_442
EXPECTED_SYNTHETIC_HOLDOUT_IDS: Final[int] = 1_558
EXPECTED_REAL_DEVELOPMENT_SAMPLES: Final[int] = 434
EXPECTED_REAL_GROUPS: Final[int] = 11

REAL_LABEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "group_id",
        "ground_truth",
        "scale_start",
        "scale_end",
        "normalized_progress",
    }
)

# The real-development labels do not include landmarks.  These conservative
# interior anchors limit random crop/perspective transforms without pretending
# to be geometry supervision.
REAL_AUGMENT_ANCHORS: Final[np.ndarray] = np.asarray(
    [[0.06, 0.06], [0.94, 0.06], [0.94, 0.94], [0.06, 0.94], [0.50, 0.50]],
    dtype=np.float32,
)


def strong_real_augmentation() -> PhotoAugmentation:
    """Fixed real-domain augmentation; progress remains invariant to its warps."""

    value = PhotoAugmentation(
        brightness_probability=0.90,
        brightness_delta=0.20,
        contrast_probability=0.90,
        contrast_min=0.55,
        contrast_max=1.55,
        gamma_probability=0.60,
        gamma_min=0.55,
        gamma_max=1.75,
        blur_probability=0.50,
        blur_sigma_max=2.0,
        noise_probability=0.45,
        noise_sigma_max=12.0,
        jpeg_probability=0.50,
        jpeg_quality_min=40,
        jpeg_quality_max=94,
        perspective_probability=0.45,
        perspective_fraction_max=0.035,
        boundary_trim_probability=0.25,
        boundary_trim_fraction_max=0.035,
    )
    value.validate()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RealProgressSample:
    sample_id: str
    group_id: str
    roi: ManifestRow
    normalized_target: float


def load_real_progress_samples(
    input_manifest_path: Path, labels_path: Path
) -> tuple[RealProgressSample, ...]:
    """Join the label-free ROI roster to its separate development labels."""

    roi_rows = load_plain_manifest(input_manifest_path)
    labels_source = Path(labels_path).resolve()
    _require(labels_source.is_file(), f"real labels do not exist: {labels_source}")
    labels: dict[str, tuple[str, float]] = {}
    with labels_source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"real labels line {line_number} is invalid JSON") from exc
            _require(isinstance(row, Mapping), f"real labels line {line_number} is not an object")
            _require(set(row) == set(REAL_LABEL_KEYS), f"real labels line {line_number} schema drift")
            sample_id = row.get("sample_id")
            group_id = row.get("group_id")
            _require(isinstance(sample_id, str) and bool(sample_id), "real sample_id is empty")
            _require(sample_id not in labels, f"duplicate real label: {sample_id}")
            _require(isinstance(group_id, str) and bool(group_id), f"{sample_id}: group_id is empty")
            try:
                ground_truth = float(row["ground_truth"])
                scale_start = float(row["scale_start"])
                scale_end = float(row["scale_end"])
                declared = float(row["normalized_progress"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{sample_id}: invalid real target") from exc
            _require(
                all(math.isfinite(value) for value in (ground_truth, scale_start, scale_end, declared))
                and scale_end > scale_start,
                f"{sample_id}: invalid real scale range",
            )
            calculated = (ground_truth - scale_start) / (scale_end - scale_start)
            _require(abs(calculated - declared) <= 1e-9, f"{sample_id}: normalized target drift")
            _require(-1e-9 <= declared <= 1.0 + 1e-9, f"{sample_id}: target outside [0,1]")
            labels[sample_id] = (group_id, min(1.0, max(0.0, declared)))
    roi_ids = tuple(row.sample_id for row in roi_rows)
    _require(set(roi_ids) == set(labels), "real ROI and label rosters differ")
    return tuple(
        RealProgressSample(row.sample_id, labels[row.sample_id][0], row, labels[row.sample_id][1])
        for row in roi_rows
    )


class RealProgressDataset(Dataset[dict[str, torch.Tensor]]):
    """Hash-verified canonical real ROIs with progress-only supervision."""

    def __init__(
        self,
        samples: Sequence[RealProgressSample],
        *,
        training: bool,
        seed: int,
        image_size: int = IMAGE_SIZE,
        augmentation: PhotoAugmentation | None = None,
        preload: bool = True,
    ) -> None:
        self.samples = tuple(samples)
        self.training = bool(training)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.augmentation = (
            strong_real_augmentation()
            if augmentation is None and self.training
            else augmentation or PhotoAugmentation.disabled()
        )
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "real-progress dataset is empty")
        _require(self.image_size >= 32, "image size is too small")
        self._preloaded: tuple[np.ndarray, ...] | None = None
        if preload:
            self._preloaded = tuple(self._load(index) for index in range(len(self.samples)))

    def _load(self, index: int) -> np.ndarray:
        _payload, image = load_canonical_roi(self.samples[index].roi)
        return np.ascontiguousarray(direct_resize_whole_roi(image, size=self.image_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        # Replacement sampling carries a draw ordinal so repeated uses of one
        # real ROI receive distinct, yet fully reproducible, augmentations.
        if isinstance(index, tuple):
            source_index, draw_ordinal = (int(index[0]), int(index[1]))
        else:
            source_index, draw_ordinal = int(index), 0
        sample = self.samples[source_index]
        source = (
            self._preloaded[source_index]
            if self._preloaded is not None
            else self._load(source_index)
        )
        image = source.copy()
        if self.training:
            rng = np.random.default_rng(
                self.seed
                + self.epoch * 1_000_003
                + source_index * 97
                + draw_ordinal * 7_919
            )
            image, _forward, _geometry_code = _cagh_augment_geometry(
                image, REAL_AUGMENT_ANCHORS, rng, self.augmentation
            )
            image, _photo_code = _cagh_augment_photo(image, rng, self.augmentation)
        return {
            "image": normalized_rgb_tensor(image),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
        }


class ShufflePadSampler(Sampler[int]):
    """One full shuffled synthetic pass, padded to an exact batch multiple."""

    def __init__(self, size: int, *, batch_size: int, seed: int) -> None:
        _require(size >= 1 and batch_size >= 1, "invalid shuffle-pad sampler sizes")
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_samples = int(math.ceil(self.size / self.batch_size) * self.batch_size)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.size).tolist()
        padding = self.num_samples - self.size
        if padding:
            order.extend(order[:padding])
        return iter(int(index) for index in order)

    def __len__(self) -> int:
        return self.num_samples


class TargetBinBalancedSampler(Sampler[tuple[int, int]]):
    """Deterministic replacement sampler balanced across nonempty target bins."""

    def __init__(
        self, targets: Sequence[float], *, num_samples: int, bins: int, seed: int
    ) -> None:
        _require(bool(targets) and num_samples >= 1 and bins >= 2, "invalid target-bin sampler")
        self.targets = tuple(float(value) for value in targets)
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.targets),
            "target-bin sampler received an invalid target",
        )
        self.num_samples = int(num_samples)
        self.bins = int(bins)
        self.seed = int(seed)
        members: list[list[int]] = [[] for _ in range(self.bins)]
        for index, target in enumerate(self.targets):
            bin_index = min(self.bins - 1, int(target * self.bins))
            members[bin_index].append(index)
        self.members = tuple(tuple(row) for row in members if row)
        _require(bool(self.members), "target-bin sampler has no nonempty bin")

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng(self.seed)
        output: list[tuple[int, int]] = []
        while len(output) < self.num_samples:
            bin_order = rng.permutation(len(self.members))
            for raw_bin_index in bin_order:
                members = self.members[int(raw_bin_index)]
                source_index = int(members[int(rng.integers(0, len(members)))])
                output.append((source_index, len(output)))
                if len(output) == self.num_samples:
                    break
        return iter(output)

    def __len__(self) -> int:
        return self.num_samples


def set_fine_tune_stage(model: GeoAttnResNet18, stage: str) -> tuple[str, ...]:
    """Apply the only two authorized trainability stages."""

    _require(stage in ("heads_cbam", "layer4"), "unknown DB-GAR18 training stage")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.progress_head, model.geometry_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, CBAM):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    if stage == "layer4":
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad_(True)
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    _require(bool(names), "DB-GAR18 stage has no trainable parameters")
    return names


def domain_balanced_objective(
    synthetic_outputs: Mapping[str, torch.Tensor],
    synthetic_batch: Mapping[str, torch.Tensor],
    real_outputs: Mapping[str, torch.Tensor],
    real_batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-domain progress loss plus synthetic-only geometry replay."""

    synthetic_progress = F.smooth_l1_loss(
        synthetic_outputs["progress"], synthetic_batch["progress"], beta=0.05
    )
    real_progress = F.smooth_l1_loss(
        real_outputs["progress"], real_batch["progress"], beta=0.05
    )
    pivot = F.smooth_l1_loss(
        synthetic_outputs["pivot"], synthetic_batch["pivot"], beta=0.05
    )
    direction = (
        1.0
        - torch.sum(
            synthetic_outputs["direction_sin_cos"]
            * synthetic_batch["direction_sin_cos"],
            dim=1,
        )
    ).mean()
    references = F.smooth_l1_loss(
        synthetic_outputs["references"], synthetic_batch["references"], beta=0.05
    )
    auxiliary = (
        PIVOT_LOSS_WEIGHT * pivot
        + DIRECTION_LOSS_WEIGHT * direction
        + REFERENCE_LOSS_WEIGHT * references
    )
    total = (
        DOMAIN_PROGRESS_WEIGHT * synthetic_progress
        + DOMAIN_PROGRESS_WEIGHT * real_progress
        + SYNTHETIC_AUX_REPLAY_SCALE * auxiliary
    )
    return total, {
        "synthetic_progress": synthetic_progress,
        "real_progress": real_progress,
        "pivot": pivot,
        "direction": direction,
        "references": references,
        "auxiliary": auxiliary,
    }


def _slice_outputs(
    outputs: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in outputs.items()}


def _train_epoch(
    model: GeoAttnResNet18,
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
        key: 0.0
        for key in (
            "loss",
            "synthetic_progress",
            "real_progress",
            "pivot",
            "direction",
            "references",
            "auxiliary",
            "synthetic_nmae",
            "real_nmae",
        )
    }
    steps = synthetic_samples = real_samples = 0
    for synthetic_raw, real_raw in zip(synthetic_loader, real_loader, strict=True):
        synthetic = {
            key: value.to(device, non_blocking=use_amp) for key, value in synthetic_raw.items()
        }
        real = {key: value.to(device, non_blocking=use_amp) for key, value in real_raw.items()}
        synthetic_count = int(synthetic["progress"].numel())
        real_count = int(real["progress"].numel())
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
            outputs = model.forward_training(torch.cat((synthetic["image"], real["image"]), dim=0))
            synthetic_outputs = _slice_outputs(outputs, 0, synthetic_count)
            real_outputs = _slice_outputs(outputs, synthetic_count, synthetic_count + real_count)
            loss, components = domain_balanced_objective(
                synthetic_outputs, synthetic, real_outputs, real
            )
        _require(bool(torch.isfinite(loss)), "DB-GAR18 loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 5.0
        )
        scaler.step(optimizer)
        scaler.update()
        totals["loss"] += float(loss.detach().cpu())
        for key, value in components.items():
            totals[key] += float(value.detach().cpu())
        totals["synthetic_nmae"] += float(
            torch.abs(synthetic_outputs["progress"].detach() - synthetic["progress"]).sum().cpu()
        )
        totals["real_nmae"] += float(
            torch.abs(real_outputs["progress"].detach() - real["progress"]).sum().cpu()
        )
        steps += 1
        synthetic_samples += synthetic_count
        real_samples += real_count
    _require(steps > 0 and synthetic_samples == real_samples, "empty or imbalanced epoch")
    result = {key: value / steps for key, value in totals.items() if not key.endswith("nmae")}
    result["synthetic_nmae"] = totals["synthetic_nmae"] / synthetic_samples
    result["real_nmae"] = totals["real_nmae"] / real_samples
    return {
        **result,
        "steps": float(steps),
        "synthetic_samples": float(synthetic_samples),
        "real_samples": float(real_samples),
    }


def _load_parent_model(checkpoint_path: Path) -> tuple[GeoAttnResNet18, Mapping[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"GeoAttn parent checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "GeoAttn parent checkpoint is not an object")
    _require(checkpoint.get("protocol") == GAR_PROTOCOL, "GeoAttn parent protocol mismatch")
    _require(checkpoint.get("architecture") == GAR_ARCHITECTURE, "GeoAttn parent architecture mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "GeoAttn parent model state is missing")
    model = GeoAttnResNet18(imagenet_pretrained=False)
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
    """Run the one fixed, terminal-only DB-GAR18 adaptation schedule."""

    _require(workers >= 0, "workers must be non-negative")
    synthetic_samples, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    real_samples = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic_samples) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "DB-GAR18 requires the frozen 14,442/1,558 scene-disjoint SyncG roster",
    )
    _require(
        len(real_samples) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_samples}) == EXPECTED_REAL_GROUPS,
        "DB-GAR18 requires the frozen 434-row/11-group real-development roster",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent = _load_parent_model(parent_checkpoint_path)
    model = model.to(device)
    synthetic_dataset = GeoAttnProgressDataset(
        synthetic_samples, training=True, seed=seed
    )
    # Preloading verifies each ROI once and avoids decoding the 434 real images
    # repeatedly under replacement sampling.  The real loader stays in-process.
    real_dataset = RealProgressDataset(
        real_samples, training=True, seed=seed + 10_000, preload=True
    )
    padded_samples = int(
        math.ceil(len(synthetic_dataset) / DOMAIN_BATCH_SIZE) * DOMAIN_BATCH_SIZE
    )
    stages = (
        ("heads_cbam", WARMUP_EPOCHS, WARMUP_LEARNING_RATE),
        ("layer4", LAYER4_EPOCHS, LAYER4_LEARNING_RATE),
    )
    history: list[dict[str, Any]] = []
    global_epoch = 0
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
            "GeoAttn-ResNet18 initialized from a terminal checkpoint; fixed domain-balanced "
            "adaptation with synthetic-only geometry replay"
        ),
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
                    "name": "heads_cbam",
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
            "progress_weights": {"synthetic": DOMAIN_PROGRESS_WEIGHT, "real": DOMAIN_PROGRESS_WEIGHT},
            "synthetic_epoch_sampling": (
                "all fit rows once plus deterministic padding to a full domain batch"
            ),
            "synthetic_samples_each_epoch": padded_samples,
            "real_epoch_sampling": "replacement, round-robin across nonempty fixed-width target bins",
            "real_samples_each_epoch": padded_samples,
            "real_target_bins": REAL_TARGET_BINS,
            "synthetic_auxiliary_replay_scale": SYNTHETIC_AUX_REPLAY_SCALE,
            "real_geometry_supervision": False,
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
    _require(source.is_file(), f"DB-GAR18 checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "DB-GAR18 checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "DB-GAR18 protocol mismatch")
    _require(checkpoint.get("architecture") == ARCHITECTURE, "DB-GAR18 architecture mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "DB-GAR18 model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = GeoAttnResNet18(imagenet_pretrained=False)
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
            "DB-GAR18 returned invalid progress",
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
    "DOMAIN_BATCH_SIZE",
    "PROTOCOL",
    "RealProgressDataset",
    "RealProgressSample",
    "ShufflePadSampler",
    "TargetBinBalancedSampler",
    "domain_balanced_objective",
    "load_checkpoint_predictor",
    "load_real_progress_samples",
    "main",
    "run_prediction",
    "set_fine_tune_stage",
    "strong_real_augmentation",
    "train",
]
