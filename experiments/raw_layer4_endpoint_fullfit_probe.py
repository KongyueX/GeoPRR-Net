"""Full-fit retry for conservative Raw layer4 + endpoint refinement.

This retry addresses two issues exposed by the first probe:

* the source Direct-ResNet18 had already seen the retrospective 14-scene
  inner split, so that split cannot serve as an independent fine-tuning dev;
* updating ``layer4`` BatchNorm running statistics on mixed clean/blur batches
  caused a large distribution shift.

The retry therefore uses the complete original 14,442-row fit population,
freezes every BatchNorm running statistic, applies a fixed two-epoch schedule,
and performs no epoch selection.  The already-used main-table development
cohort is evaluated only by a separate evaluator; the formal holdout remains
untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import DataLoader

from experiments.raw_layer4_endpoint_refinement_probe import (
    DEFAULT_FIT_MANIFEST,
    DEFAULT_SEED,
    DEFAULT_SOURCE_CHECKPOINT,
    PairedRawBlurDataset,
    _is_refined_state_name,
    _state_cpu,
    compare_raw_predictions,
    configure_refinement_scope,
    evaluate_raw_anchor,
    train_refinement_epoch,
)
from experiments.raw_multiscale_progress_probe import load_inner_scene_population
from experiments.remst_resnet18 import MomentExactResNet18Anchor
from experiments.resnet18_direct_progress import (
    _configure_reproducibility,
    load_split_roster,
    load_syncg_samples,
)
from experiments.train_remst_resnet18_probe import load_remst_resnet18_probe
from experiments.syncg_lightweight_regression_baselines import DEFAULT_SCENE_SPLIT


PROTOCOL: Final[str] = "remst_resnet18_raw_layer4_endpoint_fullfit_probe_v1"
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_layer4_fullfit/seed_20262020"
)
DEFAULT_EPOCHS: Final[int] = 2
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_LAYER4_LEARNING_RATE: Final[float] = 5.0e-6
DEFAULT_ENDPOINT_LEARNING_RATE: Final[float] = 5.0e-5
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_CONSISTENCY_WEIGHT: Final[float] = 0.05
DEFAULT_SOURCE_RETENTION_WEIGHT: Final[float] = 0.25
DEFAULT_L2SP_WEIGHT: Final[float] = 10.0


class RawLayer4FullFitError(ValueError):
    """The full-fit Raw refinement configuration or artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawLayer4FullFitError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _full_fit_samples(fit_manifest_path: Path, outer_split_path: Path):
    samples = tuple(load_syncg_samples(Path(fit_manifest_path).resolve()))
    roster = load_split_roster(Path(outer_split_path).resolve())
    _require(roster.scene_disjoint, "outer source split is not scene-disjoint")
    _require(
        len(samples) == 14_442
        and len({sample.scene_stem for sample in samples}) == 131,
        "full-fit population size differs",
    )
    _require(
        {sample.sample_id for sample in samples} == set(roster.train_ids),
        "full-fit manifest does not equal the source checkpoint fit roster",
    )
    return samples, roster


def build_full_fit_loader(
    dataset: PairedRawBlurDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    cuda: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(workers),
        pin_memory=bool(cuda),
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def load_raw_layer4_fullfit_remst(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[MomentExactResNet18Anchor, torch.nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"full-fit Raw checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping) and payload.get("protocol") == PROTOCOL,
        "full-fit Raw checkpoint metadata differs",
    )
    anchor_state = payload.get("anchor_state")
    _require(isinstance(anchor_state, Mapping), "full-fit anchor state is missing")
    target_device = torch.device(device)
    anchor, correction, source_metadata = load_remst_resnet18_probe(
        Path(str(payload["source_remst_checkpoint"])), device=target_device
    )
    incompatibility = anchor.load_state_dict(anchor_state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "full-fit Raw anchor does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    return anchor, correction, {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "source_remst_checkpoint": str(payload["source_remst_checkpoint"]),
        "epochs": int(payload["training"]["epochs"]),
        "checkpoint_selection": str(payload["training"]["checkpoint_selection"]),
        "training_population": str(payload["training"]["population"]),
        "batch_norm_running_statistics_frozen": bool(
            payload["training"]["batch_norm_running_statistics_frozen"]
        ),
        "single_backbone_evidence": dict(payload["single_backbone_evidence"]),
        "source_checkpoint": source_metadata,
    }


def run_full_fit_probe(
    *,
    source_checkpoint_path: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    layer4_learning_rate: float = DEFAULT_LAYER4_LEARNING_RATE,
    endpoint_learning_rate: float = DEFAULT_ENDPOINT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    source_retention_weight: float = DEFAULT_SOURCE_RETENTION_WEIGHT,
    l2sp_weight: float = DEFAULT_L2SP_WEIGHT,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    _require(
        epochs >= 1 and batch_size >= 1 and workers >= 0,
        "full-fit training sizes are invalid",
    )
    _require(
        layer4_learning_rate > 0.0
        and endpoint_learning_rate > 0.0
        and weight_decay >= 0.0
        and min(consistency_weight, source_retention_weight, l2sp_weight) >= 0.0,
        "full-fit optimizer/loss settings are invalid",
    )
    _require(
        max_train_samples is None or max_train_samples >= 1,
        "maximum training sample count is invalid",
    )
    root = Path(output_dir).resolve()
    checkpoint_path = root / "terminal.pt"
    results_path = root / "results.json"
    _require(
        not checkpoint_path.exists() and not results_path.exists(),
        "full-fit output artifact already exists",
    )
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(int(seed), device)

    all_samples, roster = _full_fit_samples(fit_manifest_path, outer_split_path)
    train_samples = all_samples
    partial_smoke = max_train_samples is not None
    if max_train_samples is not None:
        train_samples = train_samples[: int(max_train_samples)]
    _require(bool(train_samples), "full-fit training population is empty")

    source_anchor_cpu, source_correction, source_metadata = load_remst_resnet18_probe(
        source_checkpoint_path, device="cpu"
    )
    del source_correction
    student = copy.deepcopy(source_anchor_cpu).to(device)
    teacher = source_anchor_cpu.to(device).eval()
    trainable_names = configure_refinement_scope(student)
    source_state = _state_cpu(student)
    named_trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    source_trainable = {
        name: parameter.detach().float().clone()
        for name, parameter in named_trainable.items()
    }
    layer4_parameters = [
        parameter
        for name, parameter in named_trainable.items()
        if name.startswith("raw_encoder.layer4.")
    ]
    endpoint_parameters = [
        parameter
        for name, parameter in named_trainable.items()
        if name.startswith("raw_posterior_head.point_projection.")
    ]
    optimizer = torch.optim.AdamW(
        (
            {"params": layer4_parameters, "lr": float(layer4_learning_rate)},
            {"params": endpoint_parameters, "lr": float(endpoint_learning_rate)},
        ),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dataset = PairedRawBlurDataset(train_samples, seed=int(seed))

    # This is diagnostic replay only.  The source model has already seen these
    # rows, so the values are never used to select an epoch or hyperparameter.
    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)
    diagnostic_samples = tuple(internal.dev[: min(256, len(internal.dev))])
    source_diagnostic = evaluate_raw_anchor(
        teacher,
        diagnostic_samples,
        device=device,
        batch_size=batch_size,
        workers=workers,
        seed=int(seed) + 200_000,
    )

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        epoch_started = time.perf_counter()
        dataset.set_epoch(epoch_index)
        loader = build_full_fit_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            seed=int(seed) + 100_000 + epoch_index,
            cuda=device.type == "cuda",
        )
        train_metrics = train_refinement_epoch(
            student,
            teacher,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            source_parameters=source_trainable,
            consistency_weight=consistency_weight,
            source_retention_weight=source_retention_weight,
            l2sp_weight=l2sp_weight,
            freeze_layer4_batch_norm_stats=True,
        )
        candidate_diagnostic = evaluate_raw_anchor(
            student,
            diagnostic_samples,
            device=device,
            batch_size=batch_size,
            workers=workers,
            seed=int(seed) + 200_000,
        )
        diagnostic = compare_raw_predictions(
            source_diagnostic, candidate_diagnostic
        )
        row = {
            "epoch": epoch_index + 1,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "learning_rates": {
                "layer4": float(optimizer.param_groups[0]["lr"]),
                "point_projection": float(optimizer.param_groups[1]["lr"]),
            },
            "train": train_metrics,
            "seen_source_replay_diagnostic": diagnostic,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "phase": "raw_layer4_fullfit",
                    "epoch": row["epoch"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "train_total": train_metrics["loss"]["total"],
                    "seen_replay_raw_pooled_nmae": diagnostic["raw_pooled"][
                        "candidate_nmae"
                    ],
                    "seen_replay_relative_error_reduction": diagnostic[
                        "raw_pooled"
                    ]["relative_error_reduction"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()
    elapsed = time.perf_counter() - started
    terminal_state = _state_cpu(student)
    frozen_state_unchanged = all(
        name in terminal_state and torch.equal(expected, terminal_state[name])
        for name, expected in source_state.items()
        if not _is_refined_state_name(name)
    )
    _require(frozen_state_unchanged, "frozen stem-through-layer3 state changed")
    batch_norm_buffers_unchanged = all(
        name in terminal_state and torch.equal(expected, terminal_state[name])
        for name, expected in source_state.items()
        if name.startswith("raw_encoder.layer4.")
        and (name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"))
    )
    _require(
        batch_norm_buffers_unchanged,
        "layer4 BatchNorm running statistics changed despite the frozen protocol",
    )

    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_remst_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint": source_metadata,
        "training": {
            "population": "complete_original_source_fit_14442",
            "samples": len(train_samples),
            "available_samples": len(all_samples),
            "partial_smoke": partial_smoke,
            "epochs": int(epochs),
            "checkpoint_selection": "terminal_fixed_epoch",
            "batch_norm_running_statistics_frozen": True,
        },
        "single_backbone_evidence": {
            "image_encoder_modules": 1,
            "raw_and_sarn_share_parameter_objects": True,
            "source_correction_unchanged": True,
            "additional_inference_modules": 0,
            "stem_through_layer3_unchanged": frozen_state_unchanged,
            "layer4_batch_norm_buffers_unchanged": batch_norm_buffers_unchanged,
        },
        "anchor_state": terminal_state,
    }
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "single_seed_probe": True,
            "formal_holdout_access": False,
            "main_table_development_access": False,
            "automatic_epoch_selection": False,
            "retrospective_inner_replay_used_for_selection": False,
        },
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "split_protocol": roster.protocol,
            "available_samples": len(all_samples),
            "trained_samples": len(train_samples),
            "scenes": len({sample.scene_stem for sample in train_samples}),
        },
        "model": {
            "source_remst_checkpoint": str(Path(source_checkpoint_path).resolve()),
            "trainable_parameter_names": list(trainable_names),
            "trainable_parameters": sum(
                parameter.numel() for parameter in named_trainable.values()
            ),
            "additional_inference_parameters": 0,
            "source_correction_unchanged": True,
            "frozen_state_unchanged": frozen_state_unchanged,
            "layer4_batch_norm_buffers_unchanged": batch_norm_buffers_unchanged,
        },
        "training": {
            "seed": int(seed),
            "epochs": int(epochs),
            "checkpoint_selection": "terminal_fixed_epoch",
            "sample_balanced_full_coverage_shuffle": True,
            "batch_size": int(batch_size),
            "workers": int(workers),
            "optimizer": {
                "name": "AdamW",
                "layer4_learning_rate": float(layer4_learning_rate),
                "endpoint_learning_rate": float(endpoint_learning_rate),
                "weight_decay": float(weight_decay),
                "scheduler": "CosineAnnealingLR",
            },
            "loss": {
                "supervised": "mean_exact_l1_over_clean_and_two_random_blurs",
                "blur_consistency_weight": float(consistency_weight),
                "source_clean_retention_weight": float(source_retention_weight),
                "l2sp_source_weight_regularization": float(l2sp_weight),
            },
            "history": history,
            "elapsed_seconds": elapsed,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "results": str(results_path),
        },
    }
    _write_json(results_path, result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--layer4-learning-rate",
        type=float,
        default=DEFAULT_LAYER4_LEARNING_RATE,
    )
    parser.add_argument(
        "--endpoint-learning-rate",
        type=float,
        default=DEFAULT_ENDPOINT_LEARNING_RATE,
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT
    )
    parser.add_argument(
        "--source-retention-weight",
        type=float,
        default=DEFAULT_SOURCE_RETENTION_WEIGHT,
    )
    parser.add_argument("--l2sp-weight", type=float, default=DEFAULT_L2SP_WEIGHT)
    parser.add_argument("--max-train-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_full_fit_probe(
        source_checkpoint_path=args.source_checkpoint,
        fit_manifest_path=args.fit_manifest,
        outer_split_path=args.outer_split,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        layer4_learning_rate=args.layer4_learning_rate,
        endpoint_learning_rate=args.endpoint_learning_rate,
        weight_decay=args.weight_decay,
        consistency_weight=args.consistency_weight,
        source_retention_weight=args.source_retention_weight,
        l2sp_weight=args.l2sp_weight,
        max_train_samples=args.max_train_samples,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint": result["artifacts"]["checkpoint"],
                "training_seconds": result["training"]["elapsed_seconds"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "PROTOCOL",
    "build_full_fit_loader",
    "load_raw_layer4_fullfit_remst",
    "run_full_fit_probe",
]
