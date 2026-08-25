"""Train a zero-initialized Raw context residual on one frozen ResNet-18.

The image encoder and the existing scalar endpoint remain frozen.  A compact
MLP reads the same final 512-D representation and adds one residual in logit
space.  Fit representations for clean/moderate-blur/severe-blur are cached in
memory once, so the five fixed head epochs do not repeat encoder work.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.a15_2_mett import MomentExactPosteriorHead
from experiments.raw_layer4_endpoint_fullfit_probe import DEFAULT_FIT_MANIFEST
from experiments.remst_resnet18 import RESNET18_REPRESENTATION_FEATURES
from experiments.remst_resnet18_train_monotone_calibration_probe import (
    FitRawConditionDataset,
    RAW_CONDITIONS,
)
from experiments.resnet18_direct_progress import (
    _configure_reproducibility,
    load_syncg_samples,
)
from experiments.train_raw_layer4_fullfit_correction_adaptation import (
    load_raw_layer4_correction_adapted_remst,
)


PROTOCOL: Final[str] = "remst_resnet18_raw_context_residual_fullfit_v1"
DEFAULT_SOURCE_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_layer4_correction_adapt/"
    "seed_20262020/terminal.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_context_residual/"
    "seed_20262020/terminal.pt"
)
DEFAULT_CACHE: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_context_residual/"
    "seed_20262020/fit_feature_cache.pt"
)
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_EPOCHS: Final[int] = 5
DEFAULT_CACHE_BATCH_SIZE: Final[int] = 64
DEFAULT_TRAIN_BATCH_SIZE: Final[int] = 256
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_HIDDEN_FEATURES: Final[int] = 64
DEFAULT_LEARNING_RATE: Final[float] = 3.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_CONSISTENCY_WEIGHT: Final[float] = 0.05
DEFAULT_SOURCE_RETENTION_WEIGHT: Final[float] = 0.05
DEFAULT_RESIDUAL_L2_WEIGHT: Final[float] = 1.0e-4


class RawContextResidualError(ValueError):
    """The source checkpoint, fit cache, or residual training differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawContextResidualError(message)


class RawContextLogitResidual(nn.Module):
    """Final-representation logit residual with a zero endpoint projection."""

    def __init__(self, *, hidden_features: int = DEFAULT_HIDDEN_FEATURES) -> None:
        super().__init__()
        _require(hidden_features >= 1, "context residual width must be positive")
        self.hidden_features = int(hidden_features)
        self.normalization = nn.LayerNorm(RESNET18_REPRESENTATION_FEATURES)
        self.hidden_projection = nn.Linear(
            RESNET18_REPRESENTATION_FEATURES, self.hidden_features
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(self.hidden_features, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == RESNET18_REPRESENTATION_FEATURES,
            "context residual representation shape differs",
        )
        hidden = self.hidden_projection(self.normalization(representation.float()))
        return self.output_projection(self.activation(hidden)).squeeze(1)


class ResidualMomentExactPosteriorHead(MomentExactPosteriorHead):
    """Existing moment-exact head plus one shared-representation residual."""

    def __init__(
        self,
        source: MomentExactPosteriorHead,
        *,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
    ) -> None:
        super().__init__(
            feature_dim=int(source.feature_dim),
            progress_bins=int(source.progress_bins),
            initial_scale=float(source.log_scale_projection.bias.detach().exp().item()),
        )
        base_state = copy.deepcopy(source.state_dict())
        incompatibility = self.load_state_dict(base_state, strict=True)
        _require(
            not incompatibility.missing_keys and not incompatibility.unexpected_keys,
            "source posterior head does not copy strictly",
        )
        if source.calibration_knots_x.numel() > 0:
            self.set_monotone_calibration(
                source.calibration_knots_x.detach().cpu().tolist(),
                source.calibration_knots_y.detach().cpu().tolist(),
            )
        self.context_residual = RawContextLogitResidual(
            hidden_features=int(hidden_features)
        )

    def point_progress(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == self.feature_dim,
            "residual posterior representation shape differs",
        )
        base_logit = self.point_projection(representation.float()).squeeze(1)
        point = torch.sigmoid(base_logit + self.context_residual(representation))
        return self._calibrate(point)


def install_context_residual_head(
    anchor: nn.Module,
    *,
    hidden_features: int = DEFAULT_HIDDEN_FEATURES,
) -> ResidualMomentExactPosteriorHead:
    source = getattr(anchor, "raw_posterior_head", None)
    _require(
        isinstance(source, MomentExactPosteriorHead)
        and not isinstance(source, ResidualMomentExactPosteriorHead),
        "anchor source posterior head differs",
    )
    head = ResidualMomentExactPosteriorHead(
        source, hidden_features=int(hidden_features)
    ).to(next(anchor.parameters()).device)
    anchor.raw_posterior_head = head
    return head


def _apply_monotone_calibration(
    point: torch.Tensor,
    knots_x: torch.Tensor,
    knots_y: torch.Tensor,
) -> torch.Tensor:
    if knots_x.numel() == 0:
        return point
    _require(
        knots_x.ndim == knots_y.ndim == 1
        and knots_x.shape == knots_y.shape
        and knots_x.numel() >= 3,
        "context residual calibration knots differ",
    )
    value = point.float()
    x = knots_x.to(device=value.device, dtype=torch.float32)
    y = knots_y.to(device=value.device, dtype=torch.float32)
    clipped = torch.maximum(torch.minimum(value, x[-1]), x[0])
    upper = torch.searchsorted(x, clipped, right=True).clamp(1, x.numel() - 1)
    lower = upper - 1
    alpha = (clipped - x[lower]) / (x[upper] - x[lower])
    return (y[lower] + alpha * (y[upper] - y[lower])).clamp(0.0, 1.0)


def _collect_fit_cache(
    anchor: nn.Module,
    samples: Sequence[Any],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    head = anchor.raw_posterior_head
    _require(isinstance(head, MomentExactPosteriorHead), "fit cache endpoint differs")
    representations: list[torch.Tensor] = []
    base_logits: list[torch.Tensor] = []
    target_reference: torch.Tensor | None = None
    id_reference: tuple[str, ...] | None = None
    scene_reference: tuple[str, ...] | None = None
    started = time.perf_counter()
    anchor.eval()
    with torch.inference_mode():
        for condition_index, condition in enumerate(RAW_CONDITIONS):
            loader = DataLoader(
                FitRawConditionDataset(samples, condition=condition),
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=int(workers),
                pin_memory=device.type == "cuda",
                drop_last=False,
                persistent_workers=False,
                generator=torch.Generator().manual_seed(int(seed) + condition_index),
            )
            condition_representations: list[torch.Tensor] = []
            condition_logits: list[torch.Tensor] = []
            condition_targets: list[torch.Tensor] = []
            condition_ids: list[str] = []
            condition_scenes: list[str] = []
            for batch_index, batch in enumerate(loader):
                images = batch["image"].to(
                    device, non_blocking=device.type == "cuda"
                )
                features = anchor.raw_encoder(images)
                representation = features["representation"].float()
                logit = head.point_projection(representation).squeeze(1)
                condition_representations.append(representation.cpu())
                condition_logits.append(logit.float().cpu())
                condition_targets.append(batch["target"].float().cpu())
                condition_ids.extend(str(value) for value in batch["sample_id"])
                condition_scenes.extend(str(value) for value in batch["scene_stem"])
                if (batch_index + 1) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "context_cache",
                                "condition": condition,
                                "batches": batch_index + 1,
                                "rows": sum(
                                    int(value.shape[0])
                                    for value in condition_representations
                                ),
                                "elapsed_seconds": time.perf_counter() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            representation = torch.cat(condition_representations, dim=0)
            logits = torch.cat(condition_logits, dim=0)
            targets = torch.cat(condition_targets, dim=0)
            ids = tuple(condition_ids)
            scenes = tuple(condition_scenes)
            _require(
                representation.shape
                == (len(samples), RESNET18_REPRESENTATION_FEATURES)
                and logits.shape == targets.shape == (len(samples),),
                "fit cache tensor shape differs",
            )
            if target_reference is None:
                target_reference = targets
                id_reference = ids
                scene_reference = scenes
            else:
                _require(
                    torch.equal(target_reference, targets)
                    and id_reference == ids
                    and scene_reference == scenes,
                    "Raw condition fit cache ordering differs",
                )
            representations.append(representation)
            base_logits.append(logits)
    _require(
        target_reference is not None
        and id_reference is not None
        and scene_reference is not None,
        "fit cache is empty",
    )
    return {
        "representations": torch.stack(representations, dim=1),
        "base_logits": torch.stack(base_logits, dim=1),
        "targets": target_reference,
        "sample_ids": id_reference,
        "scene_stems": scene_reference,
        "calibration_knots_x": head.calibration_knots_x.detach().cpu().clone(),
        "calibration_knots_y": head.calibration_knots_y.detach().cpu().clone(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _train_epoch(
    residual: RawContextLogitResidual,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    consistency_weight: float,
    source_retention_weight: float,
    residual_l2_weight: float,
    calibration_knots_x: torch.Tensor,
    calibration_knots_y: torch.Tensor,
) -> dict[str, float]:
    residual.train(True)
    totals = defaultdict(float)
    rows = 0
    for representations, base_logits, targets in loader:
        representations = representations.to(
            device, non_blocking=device.type == "cuda"
        )
        base_logits = base_logits.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        batch_size, views, features = representations.shape
        _require(
            views == len(RAW_CONDITIONS)
            and features == RESNET18_REPRESENTATION_FEATURES,
            "context training cache shape differs",
        )
        deltas = residual(representations.flatten(0, 1)).reshape(batch_size, views)
        predictions = _apply_monotone_calibration(
            torch.sigmoid(base_logits + deltas),
            calibration_knots_x,
            calibration_knots_y,
        )
        exact_l1 = torch.stack(
            tuple(F.l1_loss(predictions[:, index], targets) for index in range(views))
        )
        supervised = exact_l1.mean()
        clean_reference = predictions[:, 0].detach()
        consistency = 0.5 * (
            F.l1_loss(predictions[:, 1], clean_reference)
            + F.l1_loss(predictions[:, 2], clean_reference)
        )
        source_clean = _apply_monotone_calibration(
            torch.sigmoid(base_logits[:, 0]),
            calibration_knots_x,
            calibration_knots_y,
        ).detach()
        source_retention = F.l1_loss(predictions[:, 0], source_clean)
        residual_l2 = deltas.square().mean()
        loss = (
            supervised
            + float(consistency_weight) * consistency
            + float(source_retention_weight) * source_retention
            + float(residual_l2_weight) * residual_l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), max_norm=5.0)
        optimizer.step()
        count = int(targets.numel())
        rows += count
        for name, value in (
            ("total", loss),
            ("supervised", supervised),
            ("consistency", consistency),
            ("source_retention", source_retention),
            ("residual_l2", residual_l2),
        ):
            totals[name] += float(value.detach().cpu()) * count
        for index, condition in enumerate(RAW_CONDITIONS):
            totals[f"exact_l1_{condition}"] += (
                float(exact_l1[index].detach().cpu()) * count
            )
    _require(rows >= 1, "context residual epoch is empty")
    return {name: value / rows for name, value in sorted(totals.items())}


def train_context_residual(
    *,
    source_checkpoint_path: Path,
    fit_manifest_path: Path,
    output_path: Path,
    cache_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    cache_batch_size: int = DEFAULT_CACHE_BATCH_SIZE,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    hidden_features: int = DEFAULT_HIDDEN_FEATURES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    source_retention_weight: float = DEFAULT_SOURCE_RETENTION_WEIGHT,
    residual_l2_weight: float = DEFAULT_RESIDUAL_L2_WEIGHT,
) -> dict[str, Any]:
    _require(
        epochs >= 1
        and cache_batch_size >= 1
        and train_batch_size >= 1
        and workers >= 0
        and hidden_features >= 1,
        "context residual training sizes are invalid",
    )
    _require(
        learning_rate > 0.0
        and weight_decay >= 0.0
        and consistency_weight >= 0.0
        and source_retention_weight >= 0.0
        and residual_l2_weight >= 0.0,
        "context residual optimizer/loss configuration is invalid",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"context residual output exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(int(seed), device)
    anchor, _correction, source_metadata = load_raw_layer4_correction_adapted_remst(
        source_checkpoint_path, device=device
    )
    source_anchor_state = {
        name: value.detach().cpu().clone() for name, value in anchor.state_dict().items()
    }
    samples = tuple(load_syncg_samples(Path(fit_manifest_path).resolve()))
    _require(
        len(samples) == 14_442
        and len({sample.scene_stem for sample in samples}) == 131,
        "context residual fit population identity differs",
    )
    cache_source = None if cache_path is None else Path(cache_path).resolve()
    if cache_source is not None and cache_source.exists():
        cache_payload = torch.load(cache_source, map_location="cpu", weights_only=False)
        _require(
            isinstance(cache_payload, Mapping)
            and cache_payload.get("protocol") == PROTOCOL
            and cache_payload.get("source_checkpoint")
            == str(Path(source_checkpoint_path).resolve())
            and int(cache_payload.get("seed", -1)) == int(seed)
            and isinstance(cache_payload.get("cache"), Mapping),
            "context residual cache metadata differs",
        )
        cache = dict(cache_payload["cache"])
        cache["elapsed_seconds"] = 0.0
        cache_mode = "reused"
    else:
        cache = _collect_fit_cache(
            anchor,
            samples,
            device=device,
            batch_size=int(cache_batch_size),
            workers=int(workers),
            seed=int(seed),
        )
        cache_mode = "created_in_memory" if cache_source is None else "created_and_saved"
        if cache_source is not None:
            cache_source.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "source_checkpoint": str(Path(source_checkpoint_path).resolve()),
                    "seed": int(seed),
                    "conditions": list(RAW_CONDITIONS),
                    "cache": cache,
                },
                cache_source,
            )
    _require(
        isinstance(cache.get("representations"), torch.Tensor)
        and cache["representations"].shape
        == (14_442, len(RAW_CONDITIONS), RESNET18_REPRESENTATION_FEATURES)
        and isinstance(cache.get("base_logits"), torch.Tensor)
        and cache["base_logits"].shape == (14_442, len(RAW_CONDITIONS))
        and isinstance(cache.get("targets"), torch.Tensor)
        and cache["targets"].shape == (14_442,),
        "context residual cache tensor identity differs",
    )
    head = install_context_residual_head(
        anchor, hidden_features=int(hidden_features)
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    for parameter in head.context_residual.parameters():
        parameter.requires_grad_(True)
    residual = head.context_residual
    residual.to(device)
    dataset = TensorDataset(
        cache["representations"], cache["base_logits"], cache["targets"]
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(int(seed) + 10_000),
    )
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        losses = _train_epoch(
            residual,
            loader,
            device=device,
            optimizer=optimizer,
            consistency_weight=float(consistency_weight),
            source_retention_weight=float(source_retention_weight),
            residual_l2_weight=float(residual_l2_weight),
            calibration_knots_x=cache["calibration_knots_x"],
            calibration_knots_y=cache["calibration_knots_y"],
        )
        row = {
            "epoch": epoch,
            "loss": losses,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps({"phase": "context_residual", **row}, sort_keys=True), flush=True)
    source_unchanged = all(
        torch.equal(source_anchor_state[name], value.detach().cpu())
        for name, value in anchor.state_dict().items()
        if not name.startswith("raw_posterior_head.context_residual.")
    )
    _require(source_unchanged, "frozen source anchor changed during head training")
    residual_changed = any(
        bool(torch.count_nonzero(value.detach().cpu()).item())
        for name, value in residual.state_dict().items()
        if name.startswith("output_projection.")
    )
    _require(residual_changed, "context residual terminal projection stayed zero")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "seed": int(seed),
        "source_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_metadata": source_metadata,
        "checkpoint_selection": "terminal_fixed_epoch",
        "architecture": {
            "image_encoder_modules": 1,
            "backbone": "shared_resnet18",
            "input": "existing_final_512d_representation",
            "residual_space": "scalar_logit",
            "hidden_features": int(hidden_features),
            "zero_initialized_output_projection": True,
            "additional_image_encoders": 0,
            "additional_inference_branches": 0,
            "trainable_parameters": sum(
                parameter.numel() for parameter in residual.parameters()
            ),
        },
        "training": {
            "population": "complete_original_source_fit_14442",
            "fit_scenes": 131,
            "conditions": list(RAW_CONDITIONS),
            "epochs": int(epochs),
            "cache_batch_size": int(cache_batch_size),
            "train_batch_size": int(train_batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "consistency_weight": float(consistency_weight),
            "source_retention_weight": float(source_retention_weight),
            "residual_l2_weight": float(residual_l2_weight),
            "development_access": False,
            "formal_holdout_access": False,
            "cache_elapsed_seconds": float(cache["elapsed_seconds"]),
            "cache_mode": cache_mode,
            "cache_path": None if cache_source is None else str(cache_source),
            "head_training_elapsed_seconds": time.perf_counter() - training_started,
            "history": history,
        },
        "single_backbone_evidence": {
            "source_anchor_unchanged": True,
            "source_correction_unchanged": True,
            "raw_and_sarn_share_parameter_objects": True,
            "image_encoder_modules": 1,
            "additional_image_encoders": 0,
        },
        "context_residual_state": {
            name: value.detach().cpu().clone()
            for name, value in residual.state_dict().items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "cache_elapsed_seconds": float(cache["elapsed_seconds"]),
        "cache_mode": cache_mode,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "terminal_loss": history[-1]["loss"],
        "trainable_parameters": payload["architecture"]["trainable_parameters"],
    }


def load_raw_context_residual_remst(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"context residual checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") == PROTOCOL
        and payload.get("status") == "complete",
        "context residual checkpoint metadata differs",
    )
    architecture = payload.get("architecture")
    state = payload.get("context_residual_state")
    _require(
        isinstance(architecture, Mapping)
        and isinstance(state, Mapping)
        and int(architecture.get("image_encoder_modules", -1)) == 1
        and int(architecture.get("additional_image_encoders", -1)) == 0,
        "context residual architecture/state differs",
    )
    anchor, correction, source_metadata = load_raw_layer4_correction_adapted_remst(
        Path(str(payload["source_checkpoint"])), device=device
    )
    head = install_context_residual_head(
        anchor, hidden_features=int(architecture["hidden_features"])
    )
    incompatibility = head.context_residual.load_state_dict(state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "context residual state does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    return anchor, correction, {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "seed": int(payload["seed"]),
        "checkpoint_selection": str(payload["checkpoint_selection"]),
        "architecture": dict(architecture),
        "training": dict(payload["training"]),
        "single_backbone_evidence": dict(payload["single_backbone_evidence"]),
        "source_checkpoint": source_metadata,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--cache-batch-size", type=int, default=DEFAULT_CACHE_BATCH_SIZE
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--hidden-features", type=int, default=DEFAULT_HIDDEN_FEATURES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT
    )
    parser.add_argument(
        "--source-retention-weight",
        type=float,
        default=DEFAULT_SOURCE_RETENTION_WEIGHT,
    )
    parser.add_argument(
        "--residual-l2-weight", type=float, default=DEFAULT_RESIDUAL_L2_WEIGHT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = train_context_residual(
            source_checkpoint_path=args.source_checkpoint,
            fit_manifest_path=args.fit_manifest,
            output_path=args.output,
            cache_path=args.cache,
            seed=args.seed,
            device_name=args.device,
            epochs=args.epochs,
            cache_batch_size=args.cache_batch_size,
            train_batch_size=args.train_batch_size,
            workers=args.workers,
            hidden_features=args.hidden_features,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            consistency_weight=args.consistency_weight,
            source_retention_weight=args.source_retention_weight,
            residual_l2_weight=args.residual_l2_weight,
        )
    except RawContextResidualError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "RawContextLogitResidual",
    "ResidualMomentExactPosteriorHead",
    "install_context_residual_head",
    "load_raw_context_residual_remst",
    "train_context_residual",
]
