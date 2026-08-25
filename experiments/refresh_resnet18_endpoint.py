"""Refresh the Direct-ResNet18 endpoint and fit train-only Raw calibration."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.raw_monotone_residual_calibration_probe import (
    DEFAULT_KNOT_COUNT,
    fit_monotone_l1_spline,
)
from experiments.raw_multiscale_progress_probe import (
    ConditionedProgressDataset,
    load_inner_scene_population,
)
from experiments.remst_resnet18 import (
    MomentExactResNet18Anchor,
    load_moment_exact_resnet18_anchor,
)
from experiments.resnet18_direct_progress import (
    DEFAULT_EPOCHS as SOURCE_EPOCHS,
    IMAGENET_INITIALIZATION,
    PROTOCOL as SOURCE_PROTOCOL,
    SCENE_SPLIT_PROTOCOL,
    DirectProgressDataset,
    ResNet18DirectProgress,
    _configure_reproducibility,
    _loader,
    load_split_roster,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS


PROTOCOL: Final[str] = "remst_resnet18_endpoint_refresh_exact_l1_v1"
SHRUNK_ENDPOINT_PROTOCOL: Final[str] = (
    "remst_resnet18_anchored_midpoint_endpoint_v1"
)
SUPPORTED_ENDPOINT_PROTOCOLS: Final[frozenset[str]] = frozenset(
    (PROTOCOL, SHRUNK_ENDPOINT_PROTOCOL)
)
RAW_CONDITIONS: Final[tuple[str, ...]] = tuple(CONDITIONS[:3])
DEFAULT_SOURCE_CHECKPOINT: Final[Path] = Path(
    "C:/pointer_read/paper_syncg_only_retrain_v1/factorial/"
    "resnet18_direct/seed_20262020/terminal.pt"
)
DEFAULT_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/screen200_seed20262020/"
    "internal_diagnostics/syncg_formal_fit_14442.jsonl"
)
DEFAULT_SPLIT: Final[Path] = Path(
    "C:/pointer_read/syncg_scene_disjoint_clean_v1/split.json"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_final/seed_20262020/endpoint_refresh.pt"
)
DEFAULT_REFRESH_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 3.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _load_source_model(
    checkpoint_path: Path,
) -> tuple[ResNet18DirectProgress, dict[str, Any], Mapping[str, torch.Tensor]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"source checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "source checkpoint is malformed")
    _require(
        payload.get("protocol") == SOURCE_PROTOCOL
        and payload.get("architecture")
        == "torchvision_resnet18_imagenet1k_v1_sigmoid_scalar"
        and payload.get("pretrained_weights") == IMAGENET_INITIALIZATION
        and int(payload.get("epochs", -1)) == SOURCE_EPOCHS
        and payload.get("checkpoint_selection") == "terminal_fixed_epoch"
        and payload.get("scene_disjoint") is True
        and payload.get("split_protocol") == SCENE_SPLIT_PROTOCOL,
        "source checkpoint is not the terminal scene-disjoint Direct-ResNet18",
    )
    state = payload.get("model_state")
    _require(isinstance(state, Mapping), "source model state is missing")
    model = ResNet18DirectProgress(imagenet_pretrained=False)
    incompatibility = model.load_state_dict(state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "source Direct-ResNet18 does not load strictly",
    )
    metadata = {
        "source": str(source),
        "seed": int(payload["seed"]),
        "epochs": int(payload["epochs"]),
        "train_samples": int(payload["train_samples"]),
        "holdout_samples": int(payload["holdout_samples"]),
        "split_protocol": str(payload["split_protocol"]),
        "checkpoint_selection": str(payload["checkpoint_selection"]),
    }
    return model, metadata, state


def _train_endpoint_epoch(
    model: ResNet18DirectProgress,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.eval()
    model.backbone.fc.train(True)
    samples = 0
    absolute_error = 0.0
    optimizer_steps = 0
    use_amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            predictions = model(images)
            loss = F.l1_loss(predictions, targets)
        previous_scale = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
            optimizer_steps += 1
        count = int(targets.numel())
        samples += count
        absolute_error += float(
            torch.abs(predictions.detach() - targets).sum().cpu()
        )
    _require(samples > 0, "endpoint-refresh epoch is empty")
    return {
        "samples": float(samples),
        "optimizer_steps": float(optimizer_steps),
        "exact_l1": absolute_error / samples,
    }


def _collect_condition_predictions(
    model: ResNet18DirectProgress,
    samples: Sequence[Any],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ConditionedProgressDataset(samples, condition=condition)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )
    predictions: list[float] = []
    targets: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, expected, _sample_ids, _scene_stems in loader:
            values = model(images.to(device, non_blocking=device.type == "cuda"))
            predictions.extend(float(value) for value in values.float().cpu())
            targets.extend(float(value) for value in expected.float())
    _require(
        len(predictions) == len(targets) == len(samples),
        f"calibration replay row count differs: {condition}",
    )
    return (
        np.asarray(predictions, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def refresh_resnet18_endpoint(
    *,
    source_checkpoint_path: Path,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_REFRESH_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    knot_count: int = DEFAULT_KNOT_COUNT,
) -> dict[str, Any]:
    _require(
        epochs >= 1 and batch_size >= 1 and workers >= 0 and knot_count >= 3,
        "endpoint-refresh sizes are invalid",
    )
    _require(
        learning_rate > 0.0 and weight_decay >= 0.0,
        "endpoint-refresh optimizer settings are invalid",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"endpoint-refresh output exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model, source_metadata, source_state = _load_source_model(
        source_checkpoint_path
    )
    seed = int(source_metadata["seed"])
    _configure_reproducibility(seed, device)
    internal = load_inner_scene_population(manifest_path, split_path)
    fit_samples = tuple((*internal.train, *internal.dev))
    roster = load_split_roster(split_path)
    _require(
        roster.scene_disjoint
        and roster.protocol == SCENE_SPLIT_PROTOCOL
        and len(fit_samples) == int(source_metadata["train_samples"]),
        "endpoint-refresh training roster differs from the source checkpoint",
    )
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad_(True)
    source_encoder_state = {
        name: value.detach().cpu().clone()
        for name, value in source_state.items()
        if not str(name).startswith("backbone.fc.")
    }
    dataset = DirectProgressDataset(fit_samples, training=True, seed=seed)
    optimizer = torch.optim.AdamW(
        model.backbone.fc.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        dataset.set_epoch(SOURCE_EPOCHS + epoch_index)
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + 100_000 + epoch_index,
            cuda=device.type == "cuda",
        )
        epoch_started = time.perf_counter()
        metrics = _train_endpoint_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        row = {
            "epoch": epoch_index + 1,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    training_elapsed = time.perf_counter() - started
    terminal_state = _state_cpu(model)
    encoder_unchanged = all(
        name in terminal_state
        and torch.equal(expected, terminal_state[name])
        for name, expected in source_encoder_state.items()
    )
    _require(encoder_unchanged, "frozen ResNet18 encoder state changed")

    replay_started = time.perf_counter()
    pooled_predictions: list[np.ndarray] = []
    pooled_targets: list[np.ndarray] = []
    condition_fit: dict[str, Any] = {}
    for index, condition in enumerate(RAW_CONDITIONS):
        condition_started = time.perf_counter()
        predictions, targets = _collect_condition_predictions(
            model,
            fit_samples,
            condition=condition,
            device=device,
            batch_size=batch_size,
            workers=workers,
            seed=seed + 200_000 + index,
        )
        pooled_predictions.append(predictions)
        pooled_targets.append(targets)
        condition_fit[condition] = {
            "rows": len(predictions),
            "identity_nmae": float(np.mean(np.abs(predictions - targets))),
            "elapsed_seconds": time.perf_counter() - condition_started,
        }
        print(
            json.dumps(
                {"phase": "calibration_replay", "condition": condition,
                 **condition_fit[condition]},
                sort_keys=True,
            ),
            flush=True,
        )
    fit_predictions = np.concatenate(pooled_predictions)
    fit_targets = np.concatenate(pooled_targets)
    spline, solver = fit_monotone_l1_spline(
        fit_predictions, fit_targets, knot_count=knot_count
    )
    calibrated = spline.predict(fit_predictions)
    calibration = {
        "fit_population": "scene_disjoint_source_train_clean_blur_pooled",
        "fit_rows": int(len(fit_predictions)),
        "conditions": list(RAW_CONDITIONS),
        "knots_x": spline.knots_x.tolist(),
        "knots_y": spline.knots_y.tolist(),
        "identity_nmae": float(np.mean(np.abs(fit_predictions - fit_targets))),
        "calibrated_nmae": float(np.mean(np.abs(calibrated - fit_targets))),
        "solver": solver,
        "condition_replay": condition_fit,
        "replay_elapsed_seconds": time.perf_counter() - replay_started,
    }
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_direct_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source": source_metadata,
        "training_scope": {
            "source_train_only": True,
            "formal_holdout_access": False,
            "encoder_frozen": True,
            "only_original_linear_endpoint_trainable": True,
        },
        "training_data": {
            "manifest": str(Path(manifest_path).resolve()),
            "split": str(Path(split_path).resolve()),
            "samples": len(fit_samples),
            "split_protocol": roster.protocol,
        },
        "epochs": int(epochs),
        "loss": "exact_l1",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
        },
        "history": history,
        "training_elapsed_seconds": training_elapsed,
        "encoder_state_unchanged": encoder_unchanged,
        "model_state": terminal_state,
        "calibration": calibration,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "output": str(output),
        "seed": seed,
        "epochs": int(epochs),
        "training_elapsed_seconds": training_elapsed,
        "encoder_state_unchanged": encoder_unchanged,
        "terminal_train_exact_l1": history[-1]["train"]["exact_l1"],
        "calibration_identity_nmae": calibration["identity_nmae"],
        "calibration_fitted_nmae": calibration["calibrated_nmae"],
    }


def load_refreshed_moment_exact_resnet18_anchor(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MomentExactResNet18Anchor, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"endpoint-refresh checkpoint missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") in SUPPORTED_ENDPOINT_PROTOCOLS
        and int(payload.get("epochs", -1)) == DEFAULT_REFRESH_EPOCHS
        and payload.get("encoder_state_unchanged") is True,
        "endpoint-refresh checkpoint metadata differs",
    )
    state = payload.get("model_state")
    calibration = payload.get("calibration")
    _require(
        isinstance(state, Mapping) and isinstance(calibration, Mapping),
        "endpoint-refresh model/calibration state is missing",
    )
    anchor, source_metadata = load_moment_exact_resnet18_anchor(
        Path(str(payload["source_direct_checkpoint"])), device=device
    )
    point_state = {
        str(name).removeprefix("backbone.fc."): value
        for name, value in state.items()
        if str(name).startswith("backbone.fc.")
    }
    incompatibility = anchor.raw_posterior_head.point_projection.load_state_dict(
        point_state, strict=True
    )
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "refreshed endpoint does not load strictly",
    )
    anchor.raw_posterior_head.set_monotone_calibration(
        calibration["knots_x"], calibration["knots_y"]
    )
    anchor.eval()
    endpoint_protocol = str(payload["protocol"])
    metadata = {
        **source_metadata,
        "endpoint_refresh_checkpoint": str(source),
        "endpoint_refresh_protocol": endpoint_protocol,
        "endpoint_refresh_epochs": int(payload["epochs"]),
        "endpoint_refresh_loss": str(payload["loss"]),
        "monotone_calibration": {
            "fit_population": str(calibration["fit_population"]),
            "fit_rows": int(calibration["fit_rows"]),
            "conditions": list(calibration["conditions"]),
            "knots_x": list(calibration["knots_x"]),
            "knots_y": list(calibration["knots_y"]),
        },
    }
    if endpoint_protocol == SHRUNK_ENDPOINT_PROTOCOL:
        shrinkage = payload.get("endpoint_shrinkage")
        _require(isinstance(shrinkage, Mapping), "endpoint shrinkage is missing")
        metadata["endpoint_shrinkage"] = dict(shrinkage)
        metadata["parent_endpoint_refresh_checkpoint"] = str(
            payload["parent_endpoint_refresh_checkpoint"]
        )
    return anchor, metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_REFRESH_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--knot-count", type=int, default=DEFAULT_KNOT_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = refresh_resnet18_endpoint(
        source_checkpoint_path=args.source_checkpoint,
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        knot_count=args.knot_count,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "PROTOCOL",
    "SHRUNK_ENDPOINT_PROTOCOL",
    "SUPPORTED_ENDPOINT_PROTOCOLS",
    "load_refreshed_moment_exact_resnet18_anchor",
    "refresh_resnet18_endpoint",
]
