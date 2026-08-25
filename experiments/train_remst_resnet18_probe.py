"""Train a one-seed ReMST-ResNet18 correction-only development probe."""
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
    RESNET18_STRIDE16_CHANNELS,
    RESNET18_STRIDE8_CHANNELS,
    MomentExactResNet18Anchor,
    ReMSTResNet18Correction,
    load_moment_exact_resnet18_anchor,
    remst_resnet18_publication_identity,
)
from experiments.train_a15_2_fteb_correction_only import (
    INITIALIZATION_SEED,
    LEARNING_RATE,
    SAMPLE_ORDER_SEED,
    TERMINAL_EPOCHS,
    WEIGHT_DECAY,
    build_a15_2_epoch_loader,
    build_a15_2_train_dataset,
    configure_a15_2_reproducibility,
    run_a15_2_correction_epoch,
)


PROTOCOL: Final[str] = "remst_resnet18_single_backbone_inner_probe_v1"
DEFAULT_DIRECT_CHECKPOINT: Final[Path] = Path(
    "C:/pointer_read/paper_syncg_only_retrain_v1/factorial/"
    "resnet18_direct/seed_20262020/terminal.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_single_backbone_probe/"
    "seed_20262020/terminal.pt"
)
CONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "relation_channels",
    "token_dim",
    "attention_heads",
    "decoder_layers",
    "memory_grid_size",
    "progress_bins",
    "use_relation_memory",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _construction() -> dict[str, Any]:
    return {
        "relation_channels": 48,
        "token_dim": 64,
        "attention_heads": 4,
        "decoder_layers": 2,
        "memory_grid_size": 4,
        "progress_bins": 128,
        "use_relation_memory": True,
        "stride8_channels": RESNET18_STRIDE8_CHANNELS,
        "stride16_channels": RESNET18_STRIDE16_CHANNELS,
    }


def train_remst_resnet18_probe(
    *,
    correction_train_manifest_path: Path,
    direct_checkpoint_path: Path,
    endpoint_refresh_checkpoint_path: Path | None = None,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    epochs: int = TERMINAL_EPOCHS,
    max_steps_per_epoch: int | None = None,
    initialization_seed: int = INITIALIZATION_SEED,
    sample_order_seed: int = SAMPLE_ORDER_SEED,
) -> dict[str, Any]:
    """Fit only the ReMST transport while the shared ResNet18 stays frozen."""

    _require(workers >= 0, "workers must be non-negative")
    _require(1 <= int(epochs) <= TERMINAL_EPOCHS, "epochs must be in [1, 5]")
    _require(
        max_steps_per_epoch is None or int(max_steps_per_epoch) >= 1,
        "max steps must be positive when supplied",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"probe output already exists: {output}")
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
    if endpoint_refresh_checkpoint_path is None:
        anchor, source_metadata = load_moment_exact_resnet18_anchor(
            direct_checkpoint_path, device=device
        )
    else:
        from experiments.refresh_resnet18_endpoint import (
            load_refreshed_moment_exact_resnet18_anchor,
        )

        anchor, source_metadata = load_refreshed_moment_exact_resnet18_anchor(
            endpoint_refresh_checkpoint_path, device=device
        )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    anchor_initial = _state_cpu(anchor)

    torch.manual_seed(int(initialization_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(initialization_seed))
    construction = _construction()
    correction = ReMSTResNet18Correction(
        **{name: construction[name] for name in CONSTRUCTION_FIELDS}
    ).to(device)
    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        loader = build_a15_2_epoch_loader(
            dataset,
            epoch=epoch,
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
        replay = metrics.pop("_last_replay_batch", None)
        metrics.pop("step_trace", None)
        source_order = metrics.pop("source_index_order", None)
        _require(isinstance(source_order, list), "source order is missing")
        if max_steps_per_epoch is None:
            _require(
                len(source_order) == len(samples)
                and len(set(source_order)) == len(samples),
                "full epoch sample coverage differs",
            )
        _require(replay is not None, "terminal replay batch is missing")
        row = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "physical_samples": int(metrics["physical_samples"]),
            "metrics": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "epoch": row["epoch"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "physical_samples": row["physical_samples"],
                    "loss": metrics["loss_components_active_row_weighted"][
                        "total"
                    ],
                    "active_fraction": metrics["active_fraction"],
                    "all_semantic_gradients_nonzero": all(
                        metrics["gradient_nonzero_seen"].values()
                    ),
                    "anchor_unchanged": metrics[
                        "anchor_state_bit_exact_at_epoch_end"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()

    anchor_terminal = _state_cpu(anchor)
    anchor_unchanged = _states_equal(anchor_initial, anchor_terminal)
    _require(anchor_unchanged, "frozen ResNet18 anchor changed during correction")
    total_elapsed = time.perf_counter() - started
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": REMST_RESNET18_ARCHITECTURE,
        "publication_model": remst_resnet18_publication_identity(),
        "probe_scope": {
            "correction_train_only": True,
            "inner_development_evaluation_only": True,
            "formal_holdout_access": False,
            "automatic_model_selection": False,
        },
        "source_direct_checkpoint": str(Path(direct_checkpoint_path).resolve()),
        "source_endpoint_refresh_checkpoint": (
            None
            if endpoint_refresh_checkpoint_path is None
            else str(Path(endpoint_refresh_checkpoint_path).resolve())
        ),
        "source_anchor": source_metadata,
        "construction": construction,
        "epochs": int(epochs),
        "max_steps_per_epoch": (
            None if max_steps_per_epoch is None else int(max_steps_per_epoch)
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
        },
        "seeds": {
            "initialization": int(initialization_seed),
            "sample_order": int(sample_order_seed),
        },
        "training_data": {
            "manifest": str(Path(correction_train_manifest_path).resolve()),
            "physical_samples": len(samples),
            "scenes": len({sample.scene_stem for sample in samples}),
        },
        "parameter_counts": {
            "anchor": sum(parameter.numel() for parameter in anchor.parameters()),
            **fteb_parameter_counts(correction),
        },
        "single_backbone_evidence": {
            "image_encoder_modules": 1,
            "raw_and_sarn_share_parameter_objects": True,
            "raw_and_sarn_forward_calls_separate": True,
            "anchor_frozen": True,
            "anchor_state_unchanged": anchor_unchanged,
        },
        "history": history,
        "training_elapsed_seconds": total_elapsed,
        "anchor_state": anchor_terminal,
        "correction_state": _state_cpu(correction),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "epochs": int(epochs),
        "max_steps_per_epoch": max_steps_per_epoch,
        "training_elapsed_seconds": total_elapsed,
        "anchor_state_unchanged": anchor_unchanged,
        "terminal_loss": history[-1]["metrics"][
            "loss_components_active_row_weighted"
        ]["total"],
    }


def load_remst_resnet18_probe(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[MomentExactResNet18Anchor, ReMSTResNet18Correction, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"probe checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("protocol") == PROTOCOL
        and checkpoint.get("architecture") == REMST_RESNET18_ARCHITECTURE,
        "probe checkpoint metadata differs",
    )
    construction = checkpoint.get("construction")
    anchor_state = checkpoint.get("anchor_state")
    correction_state = checkpoint.get("correction_state")
    _require(
        isinstance(construction, Mapping)
        and isinstance(anchor_state, Mapping)
        and isinstance(correction_state, Mapping),
        "probe checkpoint states are missing",
    )
    target_device = torch.device(device)
    endpoint_refresh = checkpoint.get("source_endpoint_refresh_checkpoint")
    if endpoint_refresh is None:
        anchor, _source_metadata = load_moment_exact_resnet18_anchor(
            Path(str(checkpoint["source_direct_checkpoint"])),
            device=target_device,
            progress_bins=int(construction["progress_bins"]),
        )
    else:
        from experiments.refresh_resnet18_endpoint import (
            load_refreshed_moment_exact_resnet18_anchor,
        )

        anchor, _source_metadata = load_refreshed_moment_exact_resnet18_anchor(
            Path(str(endpoint_refresh)), device=target_device
        )
    anchor_load = anchor.load_state_dict(anchor_state, strict=True)
    _require(
        not anchor_load.missing_keys and not anchor_load.unexpected_keys,
        "probe anchor does not load strictly",
    )
    correction = ReMSTResNet18Correction(
        **{name: construction[name] for name in CONSTRUCTION_FIELDS}
    ).to(target_device)
    correction_load = correction.load_state_dict(correction_state, strict=True)
    _require(
        not correction_load.missing_keys
        and not correction_load.unexpected_keys,
        "probe correction does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    return anchor, correction, {
        "checkpoint": str(source),
        "epochs": int(checkpoint["epochs"]),
        "max_steps_per_epoch": checkpoint.get("max_steps_per_epoch"),
        "source_direct_checkpoint": str(checkpoint["source_direct_checkpoint"]),
        "source_endpoint_refresh_checkpoint": checkpoint.get(
            "source_endpoint_refresh_checkpoint"
        ),
        "source_anchor": dict(checkpoint["source_anchor"]),
        "construction": dict(construction),
        "parameter_counts": dict(checkpoint["parameter_counts"]),
        "single_backbone_evidence": dict(
            checkpoint["single_backbone_evidence"]
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT
    )
    parser.add_argument("--endpoint-refresh-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=TERMINAL_EPOCHS)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument(
        "--initialization-seed", type=int, default=INITIALIZATION_SEED
    )
    parser.add_argument(
        "--sample-order-seed", type=int, default=SAMPLE_ORDER_SEED
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_remst_resnet18_probe(
        correction_train_manifest_path=args.correction_train_manifest,
        direct_checkpoint_path=args.direct_checkpoint,
        endpoint_refresh_checkpoint_path=args.endpoint_refresh_checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        epochs=args.epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        initialization_seed=args.initialization_seed,
        sample_order_seed=args.sample_order_seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DIRECT_CHECKPOINT",
    "DEFAULT_OUTPUT",
    "PROTOCOL",
    "load_remst_resnet18_probe",
    "train_remst_resnet18_probe",
]
