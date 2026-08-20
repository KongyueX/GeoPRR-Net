"""GeoAttn-ResNet18: a CAGH-inspired, pixels-only gauge regressor.

The model keeps the matched ImageNet ResNet-18 direct-regression baseline and
adds CBAM attention to every BasicBlock.  During training only, three geometry
tasks regularize the shared visual representation: pivot localization, pointer
direction (clockwise sine/cosine), and start/end ScaleMark localization.
Inference consumes pixels only and returns normalized progress in ``[0, 1]``.

Scientific boundary
-------------------
``load_training_samples`` is reused from the matched ResNet baseline.  It
interprets supervision/image fields for fit IDs only and reads only
``sample_id`` for holdout rows.  Training uses a fixed terminal epoch; no
holdout pixels, labels, metrics, or checkpoint selection are permitted.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from experiments import robustness_degradations
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import IMAGENET_WEIGHTS
from experiments.resnet18_direct_progress import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPOSITIONAL_SPLIT,
    DEFAULT_EPOCHS,
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    IMAGENET_INITIALIZATION,
    DirectProgressError,
    DirectSample,
    _canonical_json_bytes,
    _canonical_sha256,
    _configure_reproducibility,
    _require,
    load_training_samples,
    matched_cagh_augmentation,
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
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "syncg_geoattn_resnet18_progress_v1"
ARCHITECTURE: Final[str] = "GeoAttn-ResNet18"
METHOD_PREFIX: Final[str] = "geoattn_resnet18"
CBAM_REDUCTION: Final[int] = 16
CBAM_SPATIAL_KERNEL: Final[int] = 7

# Fixed before terminal training.  Keeping explicit weights makes the candidate
# easier to audit than adaptive multi-task uncertainty weighting.
PROGRESS_LOSS_WEIGHT: Final[float] = 1.0
PIVOT_LOSS_WEIGHT: Final[float] = 0.25
DIRECTION_LOSS_WEIGHT: Final[float] = 0.25
REFERENCE_LOSS_WEIGHT: Final[float] = 0.125


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    _require(values.ndim == 2 and values.shape[1] == 2, "points must be Nx2")
    homogeneous = np.concatenate(
        (values, np.ones((len(values), 1), dtype=np.float32)), axis=1
    )
    warped = homogeneous @ np.asarray(matrix, dtype=np.float32).T
    denominator = warped[:, 2:3]
    _require(bool((np.abs(denominator) > 1e-8).all()), "homography sent a label to infinity")
    result = warped[:, :2] / denominator
    _require(bool(np.isfinite(result).all()), "homography produced a non-finite label")
    return result.astype(np.float32)


def _geometry_targets(points: np.ndarray) -> dict[str, torch.Tensor]:
    """Decode ordered ScaleMarks + Pointer origin/outside into auxiliary labels."""

    values = np.asarray(points, dtype=np.float32)
    _require(values.ndim == 2 and values.shape[1] == 2 and len(values) >= 9,
             "geometry supervision must contain >=7 marks plus pivot/tip")
    start = values[0]
    end = values[-3]
    pivot = values[-2]
    tip = values[-1]
    delta = tip - pivot
    norm = float(np.linalg.norm(delta))
    _require(norm > 1e-6 and math.isfinite(norm), "pointer direction collapsed")
    # Clock convention: angle zero points upward and increases clockwise.
    direction_sin_cos = np.asarray([delta[0] / norm, -delta[1] / norm], dtype=np.float32)
    _require(bool(((values >= -1e-5) & (values <= 1.0 + 1e-5)).all()),
             "geometry label escaped the canonical ROI")
    return {
        "pivot": torch.from_numpy(pivot.copy()),
        "direction_sin_cos": torch.from_numpy(direction_sin_cos),
        "references": torch.from_numpy(np.concatenate((start, end)).astype(np.float32)),
    }


class GeoAttnProgressDataset(Dataset[dict[str, torch.Tensor]]):
    """Matched CAGH augmentation with exactly co-transformed geometry labels."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        training: bool,
        seed: int,
        image_size: int = IMAGE_SIZE,
        augmentation: PhotoAugmentation | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.training = bool(training)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.augmentation = (
            matched_cagh_augmentation()
            if augmentation is None and self.training
            else augmentation or PhotoAugmentation.disabled()
        )
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "GeoAttn dataset is empty")
        _require(self.image_size >= 32, "image size is too small")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        from experiments.v5_shared_roi_comparison_input import canonical_tight_roi_native

        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=self.image_size)
        _require(bool(sample.protected_points_xy),
                 f"{sample.sample_id}: geometry landmarks are missing")
        left, top, right, bottom = bounds
        scale = np.asarray([float(right - left), float(bottom - top)], dtype=np.float32)
        points = (
            np.asarray(sample.protected_points_xy, dtype=np.float32)
            - np.asarray([left, top], dtype=np.float32)
        ) / scale
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
            roi, forward, _geometry_code = _cagh_augment_geometry(
                roi, points, rng, self.augmentation
            )
            points = _apply_homography(points, forward)
            roi, _photo_code = _cagh_augment_photo(roi, rng, self.augmentation)
        labels = _geometry_targets(points)
        return {
            "image": normalized_rgb_tensor(roi),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            **labels,
        }


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = self.shared(F.adaptive_avg_pool2d(value, 1))
        maximum = self.shared(F.adaptive_max_pool2d(value, 1))
        return value * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = CBAM_SPATIAL_KERNEL) -> None:
        super().__init__()
        _require(kernel_size in (3, 7), "CBAM spatial kernel must be 3 or 7")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        average = torch.mean(value, dim=1, keepdim=True)
        maximum = torch.max(value, dim=1, keepdim=True).values
        return value * torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(value))


class CBAMBasicBlock(nn.Module):
    """A torchvision BasicBlock with CBAM on its residual branch."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.relu = source.relu
        self.conv2 = source.conv2
        self.bn2 = source.bn2
        self.downsample = source.downsample
        self.stride = source.stride
        self.attention = CBAM(int(source.conv2.out_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        output = self.conv1(value)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.attention(output)
        if self.downsample is not None:
            identity = self.downsample(value)
        output = output + identity
        return self.relu(output)


class GeoAttnResNet18(nn.Module):
    """Eight-block CBAM ResNet-18 with train-only geometry auxiliary heads."""

    def __init__(self, *, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        self.imagenet_pretrained = bool(imagenet_pretrained)
        backbone = resnet18(weights=IMAGENET_WEIGHTS if imagenet_pretrained else None)
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(backbone, layer_name)
            setattr(backbone, layer_name, nn.Sequential(*(CBAMBasicBlock(block) for block in layer)))
        features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.progress_head = nn.Linear(features, 1)
        self.geometry_head = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(128, 8),
        )

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Pixels-only inference surface: normalized progress and nothing else."""
        return torch.sigmoid(self.progress_head(self._features(image)).squeeze(1))

    def forward_training(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._features(image)
        progress = torch.sigmoid(self.progress_head(features).squeeze(1))
        geometry = self.geometry_head(features)
        direction = F.normalize(geometry[:, 2:4], dim=1, eps=1e-6)
        return {
            "progress": progress,
            "pivot": torch.sigmoid(geometry[:, 0:2]),
            "direction_sin_cos": direction,
            "references": torch.sigmoid(geometry[:, 4:8]),
        }


def parameter_inventory(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    attention = sum(parameter.numel() for module in model.modules()
                    if isinstance(module, CBAM) for parameter in module.parameters())
    auxiliary = sum(parameter.numel() for parameter in model.geometry_head.parameters()) \
        if isinstance(model, GeoAttnResNet18) else 0
    return {
        "total": int(total),
        "trainable": int(trainable),
        "cbam": int(attention),
        "training_only_auxiliary_head": int(auxiliary),
        "inference_parameters": int(total - auxiliary),
    }


def geometry_auxiliary_loss(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    progress = F.smooth_l1_loss(outputs["progress"], batch["progress"], beta=0.05)
    pivot = F.smooth_l1_loss(outputs["pivot"], batch["pivot"], beta=0.05)
    direction = (1.0 - torch.sum(
        outputs["direction_sin_cos"] * batch["direction_sin_cos"], dim=1
    )).mean()
    references = F.smooth_l1_loss(outputs["references"], batch["references"], beta=0.05)
    total = (
        PROGRESS_LOSS_WEIGHT * progress
        + PIVOT_LOSS_WEIGHT * pivot
        + DIRECTION_LOSS_WEIGHT * direction
        + REFERENCE_LOSS_WEIGHT * references
    )
    return total, {
        "progress": progress,
        "pivot": pivot,
        "direction": direction,
        "references": references,
    }


def _loader(
    dataset: GeoAttnProgressDataset, *, batch_size: int, workers: int,
    seed: int, cuda: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_epoch(
    model: GeoAttnResNet18,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in ("loss", "progress", "pivot", "direction", "references", "nmae")}
    samples = 0
    use_amp = device.type == "cuda"
    for raw_batch in loader:
        batch = {key: value.to(device, non_blocking=use_amp) for key, value in raw_batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model.forward_training(batch["image"])
            loss, components = geometry_auxiliary_loss(outputs, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = int(batch["progress"].numel())
        totals["loss"] += float(loss.detach().cpu()) * count
        for key, value in components.items():
            totals[key] += float(value.detach().cpu()) * count
        totals["nmae"] += float(torch.abs(
            outputs["progress"].detach() - batch["progress"]
        ).sum().cpu())
        samples += count
    _require(samples > 0, "training epoch produced no samples")
    return {**{key: value / samples for key, value in totals.items()}, "samples": float(samples)}


def train(
    *,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train one fixed terminal iterate without opening holdout pixels/labels."""

    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid training sizes")
    fit_samples, roster = load_training_samples(manifest_path, split_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    dataset = GeoAttnProgressDataset(fit_samples, training=True, seed=seed)
    model = GeoAttnResNet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        dataset.set_epoch(epoch_index)
        loader = _loader(
            dataset, batch_size=batch_size, workers=workers,
            seed=seed + epoch_index, cuda=device.type == "cuda",
        )
        metrics = _train_epoch(
            model, loader, device=device, optimizer=optimizer, scaler=scaler
        )
        history.append({
            "epoch": epoch_index + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": metrics,
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        scheduler.step()
    inventory = parameter_inventory(model)
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "architecture_detail": (
            "ImageNet ResNet-18; CBAM in every BasicBlock residual branch; scalar "
            "progress head; training-only pivot/direction/start-end auxiliary head"
        ),
        "pretrained_weights": IMAGENET_INITIALIZATION,
        "image_size": IMAGE_SIZE,
        "seed": int(seed),
        "split_protocol": roster.protocol,
        "scene_disjoint": roster.scene_disjoint,
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "train_sample_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
        "holdout_sample_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
        "holdout_access_during_training": "IDs/count only; no target, bbox, image path, or image",
        "epochs": int(epochs),
        "checkpoint_selection": "terminal_fixed_epoch",
        "loss": {
            "progress": "smooth_l1_beta_0.05",
            "pivot": "smooth_l1_beta_0.05",
            "direction": "one_minus_cosine",
            "references": "smooth_l1_beta_0.05",
            "weights": {
                "progress": PROGRESS_LOSS_WEIGHT,
                "pivot": PIVOT_LOSS_WEIGHT,
                "direction": DIRECTION_LOSS_WEIGHT,
                "references": REFERENCE_LOSS_WEIGHT,
            },
        },
        "optimizer": {
            "name": "AdamW", "learning_rate": learning_rate,
            "weight_decay": weight_decay, "scheduler": "CosineAnnealingLR",
        },
        "augmentation": {
            "name": "matched_cagh_v5_photo_and_geometry",
            "configuration": augmentation_config(),
            "co_transformed_auxiliary_labels": True,
        },
        "parameter_inventory": inventory,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "history": history,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete", "checkpoint": str(output),
        "method": f"{METHOD_PREFIX}_seed_{seed}",
        "terminal_train_nmae": history[-1]["train"]["nmae"],
        "train_samples": len(fit_samples), "holdout_samples": len(roster.validation_ids),
        "scene_disjoint": roster.scene_disjoint, "parameters": inventory,
    }


def load_checkpoint_predictor(
    checkpoint_path: Path, *, device_name: str
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "checkpoint protocol mismatch")
    _require(checkpoint.get("architecture") == ARCHITECTURE, "checkpoint architecture mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "checkpoint model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = GeoAttnResNet18(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    seed = int(checkpoint["seed"])

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        batch = torch.stack([
            normalized_rgb_tensor(direct_resize_whole_roi(image, size=IMAGE_SIZE))
            for image in images_bgr
        ]).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        output = [float(value) for value in values]
        _require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in output),
                 "model returned invalid progress")
        return output

    return f"{METHOD_PREFIX}_seed_{seed}", predict


def run_prediction(
    *, checkpoint_path: Path, manifest_path: Path, output_path: Path,
    device_name: str = "cuda:0", conditions: Sequence[str] = CONDITIONS,
    predictor_loader: Callable[..., tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]]
    = load_checkpoint_predictor,
) -> int:
    selected = tuple(conditions)
    _require(bool(selected) and len(selected) == len(set(selected))
             and set(selected) <= set(CONDITIONS), "invalid evaluation conditions")
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
                _require(all(value is not None and math.isfinite(value)
                             and 0.0 <= value <= 1.0 for value in values),
                         "prediction outside [0,1]")
                failures: list[str | None] = [None] * len(images)
            except Exception as exc:
                values = [None] * len(images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(images)
            for condition, condition_hash, progress, failure in zip(
                selected, hashes, values, failures, strict=True
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1, "protocol": PROTOCOL,
                    "sample_id": source.sample_id, "method": method,
                    "condition": condition, "robustness_seed": ROBUSTNESS_SEED,
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
    training.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--split", type=Path, default=DEFAULT_COMPOSITIONAL_SPLIT)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    training.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
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
            manifest_path=args.manifest, split_path=args.split, output_path=args.output,
            seed=args.seed, device_name=args.device, epochs=args.epochs,
            batch_size=args.batch_size, workers=args.workers,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        )
    else:
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        count = run_prediction(
            checkpoint_path=args.checkpoint, manifest_path=args.manifest,
            output_path=args.output, device_name=args.device, conditions=conditions,
        )
        result = {"status": "complete", "output": str(Path(args.output).resolve()),
                  "rows": count, "conditions": list(conditions)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE", "GeoAttnProgressDataset", "GeoAttnResNet18", "PROTOCOL",
    "geometry_auxiliary_loss", "load_checkpoint_predictor", "main",
    "parameter_inventory", "run_prediction", "train",
]
