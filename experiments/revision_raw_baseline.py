"""Run C4 only when the R0 audit calls for a six-condition raw CNN control."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments import robustness_degradations
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    DirectSample, _configure_reproducibility, _epoch, _loader,
)
from experiments.revision_cache import load_revision_samples
from experiments.revision_diagnostics import SEEDS, write_json
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS, ROBUSTNESS_SEED
from experiments.syncg_lightweight_regression_baselines import LightweightProgressRegressor
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native, direct_resize_whole_roi,
)


PROTOCOL = "revision_raw_efficientnet_six_condition_v1"
METHOD = "raw_efficientnet_b0_six_condition"
EPOCHS = 60
BATCH_SIZE = 64


class RawConditionDataset(Dataset):
    """One deterministic degraded raw image per source image and epoch."""

    def __init__(self, samples: Sequence[DirectSample], *, seed: int,
                 condition: str | None = None):
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.condition = condition
        if not self.samples or len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("samples must be a nonempty unique roster")
        if condition is not None and condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)
        if self.condition is None:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
            self.condition_indices = rng.integers(len(CONDITIONS), size=len(self.samples))
        else:
            self.condition_indices = np.full(len(self.samples), CONDITIONS.index(self.condition))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"cannot decode source image: {sample.image_path}")
        image, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        image, _metadata = robustness_degradations.apply_degradation(
            image, CONDITIONS[int(self.condition_indices[index])],
            sample_id=sample.sample_id, seed=ROBUSTNESS_SEED)
        return (normalized_rgb_tensor(direct_resize_whole_roi(image, size=256)),
                torch.tensor(sample.normalized_target, dtype=torch.float32))


def learning_rate(epoch: int) -> float:
    if not 0 <= epoch < EPOCHS:
        raise ValueError("the fixed schedule contains epochs 0 through 59")
    initial = 3e-4 if epoch < 30 else 3e-5
    return initial * (1.0 + math.cos(math.pi * (epoch % 30) / 30)) / 2.0


def train_checkpoint(samples: Sequence[DirectSample], *, manifest: Path,
                     output: Path, seed: int, device: torch.device, workers: int) -> dict:
    if seed not in SEEDS or workers < 0:
        raise ValueError("use a declared source seed and nonnegative worker count")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    _configure_reproducibility(seed, device)
    dataset = RawConditionDataset(samples, seed=seed)
    model = LightweightProgressRegressor("efficientnet_b0", imagenet_pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(0), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        loader = _loader(dataset, batch_size=BATCH_SIZE, shuffle=True, workers=workers,
                         seed=seed + epoch, cuda=device.type == "cuda")
        values = _epoch(model, loader, device=device, optimizer=optimizer, scaler=scaler)
        if values["samples"] != len(samples) or not all(math.isfinite(value) for value in values.values()):
            raise ValueError("training epoch is incomplete or has nonfinite metrics")
        row = {"epoch": epoch + 1, "learning_rate": learning_rate(epoch), "train": values,
               "condition_counts": dict(zip(CONDITIONS, np.bincount(
                   dataset.condition_indices, minlength=len(CONDITIONS)).tolist()))}
        history.append(row)
        print(json.dumps(row), flush=True)
    metadata = {
        "protocol": PROTOCOL, "method": METHOD, "architecture": "efficientnet_b0",
        "seed": seed, "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "checkpoint_selection": "terminal_epoch_60", "availability_policy": "raw_only",
        "training_scope": {"cohort": "syncg_train", "manifest": str(manifest.resolve()),
                           "source_images": len(samples), "source_groups": len({s.group_id for s in samples}),
                           "target_domain_access": False, "source_test_access": False},
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "image_size": 256,
        "loss": "SmoothL1_beta_0.05", "optimizer": "AdamW", "weight_decay": 1e-4,
        "learning_rate": "30 epochs cosine from 3e-4, then 30 epochs cosine restarted at 3e-5; optimizer moments retained",
        "training_precision": "cuda_float16_autocast_with_grad_scaler" if device.type == "cuda" else "float32",
        "augmentation": {"protocol": robustness_degradations.ROBUSTNESS_PROTOCOL,
                         "conditions": list(CONDITIONS), "probability_per_condition": 1 / 6,
                         "sampling": "one uniform condition per source image per epoch; indices precomputed before worker creation",
                         "condition_rng": "numpy.default_rng(seed + epoch_zero_based * 1000003)",
                         "degradation_seed": ROBUSTNESS_SEED, "additional_random_augmentation": False},
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, output)
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL or payload.get("architecture") != "efficientnet_b0":
        raise ValueError("checkpoint is not a C4 raw EfficientNet-B0 fit")
    if payload.get("seed") not in SEEDS or payload.get("checkpoint_selection") != "terminal_epoch_60":
        raise ValueError("checkpoint seed or terminal selection differs from the C4 protocol")
    model = LightweightProgressRegressor("efficientnet_b0", imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


def evaluate(model, samples: Sequence[DirectSample], *, output: Path, seed: int,
             cohort: str, checkpoint: Path, manifest: Path, device: torch.device,
             batch_size: int, workers: int) -> dict:
    if cohort not in ("syncg", "rf100", "industrial") or seed not in SEEDS:
        raise ValueError("evaluation requires a declared external/test cohort and source seed")
    if batch_size < 1 or workers < 0:
        raise ValueError("invalid loader sizes")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    fields = ("method", "seed", "sample_id", "group_id", "condition", "target", "prediction", "error", "status")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows, failures = 0, 0
    with output.open("w", encoding="utf-8", newline="") as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition in CONDITIONS:
            dataset = RawConditionDataset(samples, seed=seed, condition=condition)
            loader = _loader(dataset, batch_size=batch_size, shuffle=False, workers=workers,
                             seed=seed, cuda=device.type == "cuda")
            offset = 0
            for images, targets in loader:
                predictions = model(images.to(device)).float().cpu()
                if predictions.shape != targets.shape:
                    raise ValueError("model output does not match the evaluation batch")
                for prediction, target in zip(predictions.tolist(), targets.tolist()):
                    sample = samples[offset]
                    valid = math.isfinite(prediction) and 0 <= prediction <= 1
                    writer.writerow({"method": METHOD, "seed": seed, "sample_id": sample.sample_id,
                                     "group_id": sample.group_id, "condition": condition, "target": target,
                                     "prediction": prediction if valid else "", "error": abs(prediction - target) if valid else 1.0,
                                     "status": "success" if valid else "failure"})
                    failures += not valid
                    rows += 1
                    offset += 1
            if offset != len(samples):
                raise ValueError("evaluation omitted source images")
    metadata = {"protocol": PROTOCOL, "method": METHOD, "dataset": cohort,
                "scope": "evaluation", "seed": seed, "availability_policy": "raw_only",
                "checkpoint": str(checkpoint.resolve()), "manifest": str(manifest.resolve()),
                "conditions": list(CONDITIONS), "images": len(samples),
                "groups": len({s.group_id for s in samples}), "rows": rows, "failures": failures,
                "precision": "float32", "degradation_seed": ROBUSTNESS_SEED,
                "condition_access": "deterministic pixel generation only; no condition label passed to model"}
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="explicit C4 run after the R0 augmentation audit")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--workers", type=int, default=4)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--cohort", choices=("syncg", "rf100", "industrial"), required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--batch-size", type=int, default=64)
    evaluation.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "train":
        samples = load_revision_samples(args.manifest, "syncg_train")
        train_checkpoint(samples, manifest=args.manifest, output=args.output, seed=args.seed,
                         device=device, workers=args.workers)
    else:
        samples = load_revision_samples(args.manifest, args.cohort)
        model, payload = load_model(args.checkpoint, device)
        evaluate(model, samples, output=args.output, seed=payload["seed"], cohort=args.cohort,
                 checkpoint=args.checkpoint, manifest=args.manifest, device=device,
                 batch_size=args.batch_size, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
