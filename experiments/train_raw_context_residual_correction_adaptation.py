"""Adapt ReMST correction to the frozen Raw context-residual endpoint."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.a15_fteb import fteb_parameter_counts
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    load_a13_correction_train_manifest,
)
from experiments.remst_resnet18 import (
    REMST_RESNET18_ARCHITECTURE,
    remst_resnet18_publication_identity,
)
from experiments.train_a15_2_fteb_correction_only import (
    SAMPLE_ORDER_SEED,
    WEIGHT_DECAY,
    build_a15_2_epoch_loader,
    build_a15_2_train_dataset,
    configure_a15_2_reproducibility,
    run_a15_2_correction_epoch,
)
from experiments.train_remst_resnet18_raw_context_residual import (
    load_raw_context_residual_remst,
)
from experiments.train_raw_layer4_fullfit_correction_adaptation import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WORKERS,
)


PROTOCOL: Final[str] = "remst_resnet18_raw_context_correction_adaptation_v1"
DEFAULT_CONTEXT_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_context_residual_aggressive/"
    "seed_20262020/terminal.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_context_correction_adapt/"
    "seed_20262020/terminal.pt"
)


class RawContextCorrectionAdaptationError(ValueError):
    """The context endpoint or correction adaptation differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawContextCorrectionAdaptationError(message)


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _states_equal(
    expected: Mapping[str, torch.Tensor], observed: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(expected) == tuple(observed) and all(
        torch.equal(expected[name], observed[name]) for name in expected
    )


def train_context_correction_adaptation(
    *,
    context_checkpoint_path: Path,
    correction_train_manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = DEFAULT_WORKERS,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    sample_order_seed: int = SAMPLE_ORDER_SEED + 50_000,
    max_steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    _require(workers >= 0 and epochs >= 1, "adaptation training sizes are invalid")
    _require(
        learning_rate > 0.0 and weight_decay >= 0.0,
        "adaptation optimizer settings are invalid",
    )
    _require(
        max_steps_per_epoch is None or max_steps_per_epoch >= 1,
        "adaptation maximum steps is invalid",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"adaptation output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a15_2_reproducibility(device, seed=int(sample_order_seed))
    samples = tuple(
        load_a13_correction_train_manifest(
            Path(correction_train_manifest_path).resolve()
        )
    )
    dataset = build_a15_2_train_dataset(samples)
    anchor, correction, context_metadata = load_raw_context_residual_remst(
        context_checkpoint_path, device=device
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    anchor_initial = _state_cpu(anchor)
    correction_initial = _state_cpu(correction)
    for parameter in correction.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    optimizer = torch.optim.AdamW(
        correction.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        epoch_started = time.perf_counter()
        loader = build_a15_2_epoch_loader(
            dataset,
            epoch=epoch_index,
            workers=workers,
            cuda=device.type == "cuda",
            sample_order_seed=int(sample_order_seed),
        )
        metrics = run_a15_2_correction_epoch(
            anchor,
            correction,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            anchor_state_reference=anchor_initial,
            max_steps=max_steps_per_epoch,
        )
        metrics.pop("_last_replay_batch", None)
        metrics.pop("step_trace", None)
        source_order = metrics.pop("source_index_order", None)
        _require(isinstance(source_order, list), "adaptation sample order is missing")
        if max_steps_per_epoch is None:
            _require(
                len(source_order) == len(samples)
                and len(set(source_order)) == len(samples),
                "adaptation full-epoch sample coverage differs",
            )
        row = {
            "epoch": epoch_index + 1,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "physical_samples": int(metrics["physical_samples"]),
            "metrics": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "phase": "context_correction_adaptation",
                    "epoch": row["epoch"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "physical_samples": row["physical_samples"],
                    "loss": metrics["loss_components_active_row_weighted"]["total"],
                    "anchor_unchanged": metrics[
                        "anchor_state_bit_exact_at_epoch_end"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()
    anchor_terminal = _state_cpu(anchor)
    correction_terminal = _state_cpu(correction)
    anchor_unchanged = _states_equal(anchor_initial, anchor_terminal)
    correction_changed = not _states_equal(correction_initial, correction_terminal)
    _require(anchor_unchanged, "Raw context anchor changed during adaptation")
    _require(correction_changed, "correction did not change during adaptation")
    elapsed = time.perf_counter() - started
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": REMST_RESNET18_ARCHITECTURE,
        "publication_model": remst_resnet18_publication_identity(),
        "source_context_checkpoint": str(Path(context_checkpoint_path).resolve()),
        "source_context_metadata": context_metadata,
        "training_scope": {
            "context_anchor_frozen": True,
            "source_correction_continued": True,
            "formal_holdout_access": False,
            "main_table_development_access": False,
            "automatic_epoch_selection": False,
        },
        "training_data": {
            "manifest": str(Path(correction_train_manifest_path).resolve()),
            "physical_samples": len(samples),
            "scenes": len({sample.scene_stem for sample in samples}),
        },
        "epochs": int(epochs),
        "checkpoint_selection": "terminal_fixed_epoch",
        "max_steps_per_epoch": max_steps_per_epoch,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
        },
        "sample_order_seed": int(sample_order_seed),
        "parameter_counts": {
            "anchor": sum(parameter.numel() for parameter in anchor.parameters()),
            **fteb_parameter_counts(correction),
        },
        "single_backbone_evidence": {
            "image_encoder_modules": 1,
            "raw_and_sarn_share_parameter_objects": True,
            "anchor_frozen": True,
            "anchor_state_unchanged": anchor_unchanged,
            "additional_image_encoders": 0,
        },
        "history": history,
        "training_elapsed_seconds": elapsed,
        "correction_initial_state": correction_initial,
        "correction_state": correction_terminal,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "epochs": int(epochs),
        "training_elapsed_seconds": elapsed,
        "anchor_state_unchanged": anchor_unchanged,
        "correction_changed": correction_changed,
        "terminal_loss": history[-1]["metrics"][
            "loss_components_active_row_weighted"
        ]["total"],
    }


def load_context_correction_adapted_remst(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"context adaptation checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") == PROTOCOL
        and payload.get("architecture") == REMST_RESNET18_ARCHITECTURE,
        "context adaptation checkpoint metadata differs",
    )
    correction_state = payload.get("correction_state")
    _require(
        isinstance(correction_state, Mapping), "adapted correction state is missing"
    )
    anchor, correction, context_metadata = load_raw_context_residual_remst(
        Path(str(payload["source_context_checkpoint"])), device=device
    )
    incompatibility = correction.load_state_dict(correction_state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "context-adapted correction does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    return anchor, correction, {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "source_context_checkpoint": str(payload["source_context_checkpoint"]),
        "epochs": int(payload["epochs"]),
        "checkpoint_selection": str(payload["checkpoint_selection"]),
        "training_scope": dict(payload["training_scope"]),
        "single_backbone_evidence": dict(payload["single_backbone_evidence"]),
        "source_context_metadata": context_metadata,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-checkpoint", type=Path, default=DEFAULT_CONTEXT_CHECKPOINT
    )
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--sample-order-seed", type=int, default=SAMPLE_ORDER_SEED + 50_000
    )
    parser.add_argument("--max-steps-per-epoch", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_context_correction_adaptation(
        context_checkpoint_path=args.context_checkpoint,
        correction_train_manifest_path=args.correction_train_manifest,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sample_order_seed=args.sample_order_seed,
        max_steps_per_epoch=args.max_steps_per_epoch,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "load_context_correction_adapted_remst",
    "train_context_correction_adaptation",
]
