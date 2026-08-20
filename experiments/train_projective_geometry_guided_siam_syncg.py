"""Train PG-SIAM with SyncG fit images only.

This is the paper-retraining entry point for the field-test-only protocol.  It
accepts a freshly trained scene-disjoint GeoAttn-ResNet18 source checkpoint and
never accepts a field manifest, field labels, or a field selection artifact.
The two PG-SIAM adapter vectors are trained for the existing fixed two-epoch
schedule; the terminal epoch is used without validation-based selection.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import DataLoader, Sampler

from experiments.evaluate_projective_geometry_guided_siam import run_prediction
from experiments.geoattn_resnet18_progress import (
    ARCHITECTURE as PARENT_ARCHITECTURE,
    PROTOCOL as PARENT_PROTOCOL,
)
from experiments.projective_geometry_guided_siam import (
    ARCHITECTURE as MODEL_ARCHITECTURE,
    PROTOCOL as MODEL_PROTOCOL,
    TRAINABLE_PARAMETERS,
    ProjectiveGeometryGuidedSIAM,
    load_db_gar_state_into_pg_siam_model,
    set_pg_siam_calibration_stage,
)
from experiments.resnet18_direct_progress import (
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    IMAGENET_INITIALIZATION,
    _canonical_json_bytes,
    _configure_reproducibility,
    load_training_samples,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.train_projective_geometry_guided_siam import (
    BATCH_SIZE,
    CONSISTENCY_WEIGHT,
    CVaR_TAIL_FRACTION,
    CVaR_WEIGHT,
    EPOCHS,
    GEOMETRY_WEIGHT,
    LEARNING_RATE,
    PGSIAMTrainingDataset,
    WEIGHT_DECAY,
    _train_epoch,
)


TRAINING_PROTOCOL: Final[str] = "projective_geometry_guided_siam_syncg_only_v1"
METHOD_PREFIX: Final[str] = "pgsiam_syncg"
EXPECTED_FIT_SAMPLES: Final[int] = 14_442
EXPECTED_HOLDOUT_SAMPLES: Final[int] = 1_558


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class SyntheticOnlySampler(Sampler[tuple[int, int]]):
    """Yield each synthetic fit row once in a deterministic shuffled order."""

    def __init__(self, sample_count: int, *, seed: int) -> None:
        _require(sample_count > 0, "synthetic sampler is empty")
        self.sample_count = int(sample_count)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator().manual_seed(self.seed)
        for draw, index in enumerate(
            torch.randperm(self.sample_count, generator=generator).tolist()
        ):
            yield int(index), int(draw)

    def __len__(self) -> int:
        return self.sample_count


def _load_syncg_parent(
    checkpoint_path: Path,
) -> tuple[ProjectiveGeometryGuidedSIAM, Mapping[str, Any], tuple[str, ...]]:
    """Load only a source checkpoint that has never used field photographs.

    A field-adapted checkpoint has the same tensor shapes, so ordinary state
    loading cannot distinguish it from the requested source model.  The
    existing checkpoint protocol and initialization metadata are therefore
    checked before its weights are accepted.
    """

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"SyncG parent checkpoint is missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "SyncG parent checkpoint is not an object")
    _require(checkpoint.get("protocol") == PARENT_PROTOCOL, "parent is not a SyncG source model")
    _require(checkpoint.get("architecture") == PARENT_ARCHITECTURE, "parent architecture mismatch")
    _require(
        checkpoint.get("pretrained_weights") == IMAGENET_INITIALIZATION,
        "parent was not initialized from the matched ImageNet weights",
    )
    _require(
        checkpoint.get("train_samples") == EXPECTED_FIT_SAMPLES
        and checkpoint.get("holdout_samples") == EXPECTED_HOLDOUT_SAMPLES,
        "parent does not use the 14,442/1,558 scene-disjoint SyncG split",
    )
    _require(
        checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch",
        "parent is not the fixed terminal epoch",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "SyncG parent model state is missing")
    model = ProjectiveGeometryGuidedSIAM(imagenet_pretrained=False)
    missing = load_db_gar_state_into_pg_siam_model(model, state)
    return model, checkpoint, missing


def train(
    *,
    parent_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
) -> dict[str, Any]:
    """Train one terminal PG-SIAM checkpoint without any field-data input."""

    _require(workers >= 0, "workers must be non-negative")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite checkpoint: {output}")
    synthetic, roster = load_training_samples(
        synthetic_manifest_path, synthetic_split_path
    )
    _require(
        len(synthetic) == EXPECTED_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_HOLDOUT_SAMPLES,
        "PG-SIAM requires the 14,442/1,558 scene-disjoint SyncG split",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    model, parent, missing = _load_syncg_parent(parent_checkpoint_path)
    _require(int(parent.get("seed", -1)) == int(seed), "parent seed mismatch")
    model = model.to(device)
    trainable_names = set_pg_siam_calibration_stage(model)
    dataset = PGSIAMTrainingDataset(synthetic, (), seed=seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=SyntheticOnlySampler(len(synthetic), seed=seed + epoch),
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        metrics = _train_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        _require(metrics["real_fraction"] == 0.0, "field sample entered training")
        row = {"epoch": epoch + 1, "train": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    method = f"{METHOD_PREFIX}_seed_{seed}"
    checkpoint = {
        "schema_version": 1,
        "protocol": MODEL_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "architecture": MODEL_ARCHITECTURE,
        "method": method,
        "seed": int(seed),
        "image_size": IMAGE_SIZE,
        "parent": {
            "checkpoint": str(Path(parent_checkpoint_path).resolve()),
            "protocol": parent.get("protocol"),
            "architecture": parent.get("architecture"),
            "seed": int(parent["seed"]),
            "pretrained_weights": parent.get("pretrained_weights"),
        },
        "parent_missing_keys_initialized": list(missing),
        "training_inputs": {
            "synthetic_manifest": str(Path(synthetic_manifest_path).resolve()),
            "synthetic_split": str(Path(synthetic_split_path).resolve()),
            "synthetic_fit_samples": len(synthetic),
            "synthetic_holdout_samples": len(roster.validation_ids),
            "field_training_samples": 0,
            "field_selection_samples": 0,
        },
        "schedule": {
            "epochs": EPOCHS,
            "checkpoint_selection": "terminal_fixed_epoch",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "trainable_parameter_names": list(trainable_names),
        },
        "loss": {
            "supervised_progress": True,
            "synthetic_geometry_auxiliary": True,
            "clean_parent_consistency_weight": CONSISTENCY_WEIGHT,
            "parent_relative_cvar_weight": CVaR_WEIGHT,
            "cvar_tail_fraction": CVaR_TAIL_FRACTION,
            "geometry_weight": GEOMETRY_WEIGHT,
        },
        "parameter_inventory": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "training_seconds": float(time.perf_counter() - started),
        "history": history,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    _require(
        checkpoint["parameter_inventory"]["trainable"] == TRAINABLE_PARAMETERS,
        "trainable parameter count mismatch",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "method": method,
        "training_seconds": checkpoint["training_seconds"],
        "field_training_samples": 0,
        "field_selection_samples": 0,
    }


def load_checkpoint_model(
    checkpoint_path: Path, *, device_name: str
) -> tuple[str, ProjectiveGeometryGuidedSIAM, Mapping[str, Any]]:
    """Load a synthetic-only terminal checkpoint for ordinary prediction."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"PG-SIAM checkpoint is missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "PG-SIAM checkpoint is not an object")
    _require(checkpoint.get("protocol") == MODEL_PROTOCOL, "model protocol mismatch")
    _require(
        checkpoint.get("training_protocol") == TRAINING_PROTOCOL,
        "checkpoint is not the SyncG-only retrain",
    )
    _require(checkpoint.get("architecture") == MODEL_ARCHITECTURE, "architecture mismatch")
    seed = int(checkpoint.get("seed", -1))
    method = f"{METHOD_PREFIX}_seed_{seed}"
    _require(checkpoint.get("method") == method, "method identity mismatch")
    inputs = checkpoint.get("training_inputs")
    _require(isinstance(inputs, Mapping), "training input summary is missing")
    _require(
        inputs.get("field_training_samples") == 0
        and inputs.get("field_selection_samples") == 0,
        "checkpoint used field data",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = ProjectiveGeometryGuidedSIAM(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return method, model.to(device).eval(), checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--parent-checkpoint", type=Path, required=True)
    training.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--synthetic-split", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--workers", type=int, default=4)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--diagnostics", type=Path, required=True)
    prediction.add_argument("--summary", type=Path, required=True)
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
            output_path=args.output,
            seed=args.seed,
            device_name=args.device,
            workers=args.workers,
        )
    else:
        selected = CONDITIONS if args.conditions == "all" else ("clean",)
        result = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            diagnostics_path=args.diagnostics,
            summary_path=args.summary,
            device_name=args.device,
            conditions=selected,
            checkpoint_loader=load_checkpoint_model,
        )
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD_PREFIX",
    "SyntheticOnlySampler",
    "TRAINING_PROTOCOL",
    "load_checkpoint_model",
    "main",
    "train",
]
