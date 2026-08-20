"""Train ReMSTNet-v3 and its publication ablations."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import DataLoader

from experiments.a15_1_fteb_targets import (
    a15_1_fteb_loss,
    build_a15_1_targets,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    load_a13_correction_train_manifest,
)
from remstnet.model import (
    ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
    ADAPTIVE_REMST_NET_ARCHITECTURE,
    COORDINATED_REMST_NET_ARCHITECTURE,
    PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
    REMST_BLOCK_NET_ARCHITECTURE,
    CoordinatedReMSTNet,
    ReMSTBlockNet,
    initialize_adaptive_budget_ablation_remst_net,
    initialize_adaptive_remst_net,
    initialize_coordinated_remst_net,
    initialize_progress_mixing_ablation_remst_net,
    initialize_remst_block_net,
    remstnet_parameter_counts,
)
from experiments.train_a15_2_fteb_correction_only import (
    INITIALIZATION_SEED,
    LEARNING_RATE,
    SAMPLE_ORDER_SEED,
    TERMINAL_EPOCHS,
    WEIGHT_DECAY,
    build_a15_2_train_dataset,
    collate_a15_2_physical_triplets,
    configure_a15_2_reproducibility,
    epoch_sample_order_seed,
)
from experiments.train_a15_2_mett import DEFAULT_DIRECT_CHECKPOINT


PROTOCOL: Final[str] = "remst_block_hierarchical_pilot_training_v1"
COORDINATED_PROTOCOL: Final[str] = (
    "remst_block_cross_scale_coordinated_pilot_training_v2"
)
ADAPTIVE_PROTOCOL: Final[str] = (
    "remst_block_adaptive_budget_progress_mixing_pilot_training_v3"
)
PROGRESS_MIXING_ABLATION_PROTOCOL: Final[str] = (
    "remstnet_progress_mixing_fixed_budget_ablation_training_v1"
)
ADAPTIVE_BUDGET_ABLATION_PROTOCOL: Final[str] = (
    "remstnet_adaptive_budget_no_progress_mixing_ablation_training_v1"
)
ARCHITECTURE_VARIANTS: Final[tuple[str, ...]] = (
    "hierarchical_v1",
    "cross_scale_coordinated_v2",
    "progress_mixing_fixed_budget_ablation",
    "adaptive_budget_no_progress_mixing_ablation",
    "adaptive_budget_progress_mixing_v3",
)
DEFAULT_PHYSICAL_BATCH_SIZE: Final[int] = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {name: _device(item, device) for name, item in value.items()}
    return value


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _foundation_state(model: ReMSTBlockNet) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for module_name in (
        "shared_scale8_encoder",
        "shared_scale16_encoder",
        "shared_context_encoder",
        "moment_readout",
    ):
        module = getattr(model, module_name)
        for name, value in module.state_dict().items():
            result[f"{module_name}.{name}"] = value.detach().cpu().clone()
    return result


def _states_equal(
    expected: Mapping[str, torch.Tensor], observed: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(expected) == tuple(observed) and all(
        torch.equal(expected[name], observed[name]) for name in expected
    )


def _loader(
    dataset: Any,
    *,
    epoch: int,
    workers: int,
    cuda: bool,
    physical_batch_size: int,
    sample_order_seed: int,
) -> DataLoader[dict[str, Any]]:
    dataset.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(
            epoch_sample_order_seed(epoch, base_seed=sample_order_seed)
        ),
        collate_fn=collate_a15_2_physical_triplets,
    )


def load_remstnet_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ReMSTBlockNet, dict[str, Any]]:
    """Load one self-contained pilot checkpoint with strict tensor replay."""

    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "ReMSTNet checkpoint is malformed")
    construction = payload.get("construction")
    state = payload.get("model_state")
    _require(
        isinstance(construction, Mapping) and isinstance(state, Mapping),
        "ReMSTNet checkpoint model payload is missing",
    )
    architecture = str(payload.get("architecture", ""))
    common = {
        "progress_bins": int(construction["progress_bins"]),
        "posterior_scale": float(construction["posterior_scale"]),
        "relation_channels": int(construction["relation_channels"]),
        "token_dim": int(construction["token_dim"]),
        "memory_grid_size": int(construction["memory_grid_size"]),
    }
    if architecture in {
        COORDINATED_REMST_NET_ARCHITECTURE,
        PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
        ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
        ADAPTIVE_REMST_NET_ARCHITECTURE,
    }:
        model: ReMSTBlockNet = CoordinatedReMSTNet(
            **common,
            attention_heads=int(construction["attention_heads"]),
            decoder_layers=int(construction["decoder_layers"]),
            use_progress_mixing=bool(
                construction.get("use_progress_mixing", False)
            ),
            learnable_budget_gain=bool(
                construction.get("learnable_budget_gain", False)
            ),
        )
    else:
        _require(
            architecture == REMST_BLOCK_NET_ARCHITECTURE,
            "ReMSTNet checkpoint architecture is unsupported",
        )
        model = ReMSTBlockNet(**common)
    incompatibility = model.load_state_dict(state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "ReMSTNet checkpoint does not load strictly",
    )
    model.to(torch.device(device)).eval()
    return model, {
        "checkpoint": str(source),
        "protocol": payload.get("protocol"),
        "architecture": payload.get("architecture"),
        "source_foundation": payload.get("source_foundation"),
        "construction": dict(construction),
        "parameter_counts": payload.get("parameter_counts"),
        "epochs": int(payload.get("epochs", 0)),
        "seeds": payload.get("seeds"),
        "training_elapsed_seconds": float(
            payload.get("training_elapsed_seconds", 0.0)
        ),
    }


def train_remstnet(
    *,
    correction_train_manifest_path: Path,
    direct_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    epochs: int = TERMINAL_EPOCHS,
    physical_batch_size: int = DEFAULT_PHYSICAL_BATCH_SIZE,
    initialization_seed: int = INITIALIZATION_SEED,
    sample_order_seed: int = SAMPLE_ORDER_SEED,
    max_steps_per_epoch: int | None = None,
    architecture_variant: str = "adaptive_budget_progress_mixing_v3",
) -> dict[str, Any]:
    """Fit ReMSTNet-v3 or one declared publication ablation."""

    _require(workers >= 0, "ReMSTNet workers must be non-negative")
    # The shared augmentation dataset and sample-order function intentionally
    # define exactly five publication epochs.  A larger value previously ran
    # five costly epochs and then failed before saving when set_epoch(5) met
    # that boundary; Git/version/types/keys and ordinary forward tests cannot
    # detect this runtime schedule mismatch.  Reject it at the formal training
    # entrypoint while preserving the established five-epoch protocol.
    _require(
        1 <= epochs <= TERMINAL_EPOCHS,
        "ReMSTNet epochs must stay within the five-epoch publication protocol",
    )
    _require(physical_batch_size >= 1, "ReMSTNet batch size must be positive")
    _require(
        max_steps_per_epoch is None or max_steps_per_epoch >= 1,
        "ReMSTNet max steps must be positive",
    )
    _require(
        architecture_variant in ARCHITECTURE_VARIANTS,
        "ReMSTNet architecture variant is unsupported",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"ReMSTNet output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "ReMSTNet device is unsupported")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a15_2_reproducibility(device, seed=int(sample_order_seed))
    torch.manual_seed(int(initialization_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(initialization_seed))

    samples = tuple(
        load_a13_correction_train_manifest(
            Path(correction_train_manifest_path).resolve()
        )
    )
    dataset = build_a15_2_train_dataset(samples)
    if architecture_variant == "adaptive_budget_progress_mixing_v3":
        model, source_metadata = initialize_adaptive_remst_net(
            direct_checkpoint_path,
            device=device,
        )
        training_protocol = ADAPTIVE_PROTOCOL
        architecture = ADAPTIVE_REMST_NET_ARCHITECTURE
    elif architecture_variant == "progress_mixing_fixed_budget_ablation":
        model, source_metadata = initialize_progress_mixing_ablation_remst_net(
            direct_checkpoint_path,
            device=device,
        )
        training_protocol = PROGRESS_MIXING_ABLATION_PROTOCOL
        architecture = PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE
    elif architecture_variant == "adaptive_budget_no_progress_mixing_ablation":
        model, source_metadata = initialize_adaptive_budget_ablation_remst_net(
            direct_checkpoint_path,
            device=device,
        )
        training_protocol = ADAPTIVE_BUDGET_ABLATION_PROTOCOL
        architecture = ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE
    elif architecture_variant == "cross_scale_coordinated_v2":
        model, source_metadata = initialize_coordinated_remst_net(
            direct_checkpoint_path,
            device=device,
        )
        training_protocol = COORDINATED_PROTOCOL
        architecture = COORDINATED_REMST_NET_ARCHITECTURE
    else:
        model, source_metadata = initialize_remst_block_net(
            direct_checkpoint_path,
            device=device,
        )
        training_protocol = PROTOCOL
        architecture = REMST_BLOCK_NET_ARCHITECTURE
    model.train()
    foundation_initial = _foundation_state(model)
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    _require(bool(trainable_parameters), "ReMSTNet has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        loader = _loader(
            dataset,
            epoch=epoch,
            workers=workers,
            cuda=device.type == "cuda",
            physical_batch_size=physical_batch_size,
            sample_order_seed=int(sample_order_seed),
        )
        weighted: dict[str, float] = {
            "loss": 0.0,
            "nmae": 0.0,
            "raw_nmae": 0.0,
            "sarn_nmae": 0.0,
            "relation_available": 0.0,
            "absolute_moment_error": 0.0,
            "absolute_total_shift": 0.0,
            "scale8_feature_residual_rms": 0.0,
            "scale16_feature_residual_rms": 0.0,
            "moment_budget_gain": 0.0,
        }
        rows_seen = 0
        physical_seen = 0
        steps = 0
        for raw_batch in loader:
            if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
                break
            batch = {
                name: _device(raw_batch[name], device)
                for name in (
                    "target",
                    "physical_group_ids",
                    "original_view",
                    "sarn_view",
                    "sarn_support_mask",
                    "sarn_active",
                    "raw_to_sarn_homography",
                )
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                prediction = model(
                    batch["original_view"],
                    batch["sarn_view"],
                    batch["sarn_support_mask"],
                    sarn_active=batch["sarn_active"].bool(),
                    raw_to_sarn_homography=batch["raw_to_sarn_homography"],
                )
                targets = build_a15_1_targets({"target": batch["target"]})
                loss, _components = a15_1_fteb_loss(
                    prediction,
                    targets,
                    batch["physical_group_ids"],
                )
            _require(bool(torch.isfinite(loss)), "ReMSTNet loss is non-finite")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            target = batch["target"].float()
            row_count = int(target.shape[0])
            available = prediction["relation_available"].float()
            metrics = {
                "loss": float(loss.detach()),
                "nmae": float(torch.abs(prediction["mean"].detach() - target).mean()),
                "raw_nmae": float(
                    torch.abs(prediction["raw_anchor_mean"].detach() - target).mean()
                ),
                "sarn_nmae": float(
                    torch.abs(prediction["sarn_endpoint_mean"].detach() - target).mean()
                ),
                "relation_available": float(available.mean()),
                "absolute_moment_error": float(
                    prediction["moment_absolute_error"].detach().mean()
                ),
                "absolute_total_shift": float(
                    prediction["total_moment_shift"].detach().abs().mean()
                ),
                "scale8_feature_residual_rms": float(
                    prediction["scale8_feature_residual_rms"].detach().mean()
                ),
                "scale16_feature_residual_rms": float(
                    prediction["scale16_feature_residual_rms"].detach().mean()
                ),
                "moment_budget_gain": float(
                    prediction.get(
                        "moment_budget_gain",
                        torch.ones((), device=target.device),
                    )
                    .detach()
                    .mean()
                ),
            }
            for name, value in metrics.items():
                weighted[name] += value * row_count
            rows_seen += row_count
            physical_seen += math.ceil(row_count / 3)
            steps += 1
            if steps % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": steps,
                            "steps_expected": math.ceil(
                                len(samples) / physical_batch_size
                            ),
                            "loss": metrics["loss"],
                            "nmae": metrics["nmae"],
                            "mean_abs_shift": metrics["absolute_total_shift"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        _require(steps >= 1 and rows_seen >= 1, "ReMSTNet epoch is empty")
        epoch_metrics = {
            name: value / float(rows_seen) for name, value in weighted.items()
        }
        row = {
            "epoch": epoch + 1,
            "steps": steps,
            "condition_rows": rows_seen,
            "physical_samples_seen": physical_seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
            "metrics": epoch_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()

    training_elapsed = float(time.perf_counter() - training_started)
    foundation_unchanged = _states_equal(
        foundation_initial, _foundation_state(model)
    )
    checkpoint = {
        "schema_version": 1,
        "protocol": training_protocol,
        "status": "pilot_complete",
        "architecture": architecture,
        "architecture_variant": architecture_variant,
        "construction": model.construction,
        "source_foundation": source_metadata,
        "epochs": int(epochs),
        "seeds": {
            "initialization": int(initialization_seed),
            "sample_order": int(sample_order_seed),
        },
        "training": {
            "correction_train_manifest": str(
                Path(correction_train_manifest_path).resolve()
            ),
            "physical_samples": len(samples),
            "physical_batch_size": int(physical_batch_size),
            "max_steps_per_epoch": max_steps_per_epoch,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "precision": (
                str(autocast_dtype).removeprefix("torch.")
                if autocast_enabled
                else "float32"
            ),
            "foundation_frozen": True,
            "foundation_state_unchanged": foundation_unchanged,
        },
        "parameter_counts": remstnet_parameter_counts(model),
        "history": history,
        "training_elapsed_seconds": training_elapsed,
        "model_state": _state_cpu(model),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "pilot_complete",
        "checkpoint": str(output),
        "epochs": int(epochs),
        "training_elapsed_seconds": training_elapsed,
        "foundation_state_unchanged": foundation_unchanged,
        "parameter_counts": checkpoint["parameter_counts"],
        "terminal_metrics": history[-1]["metrics"],
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=TERMINAL_EPOCHS)
    parser.add_argument(
        "--physical-batch-size", type=int, default=DEFAULT_PHYSICAL_BATCH_SIZE
    )
    parser.add_argument(
        "--initialization-seed", type=int, default=INITIALIZATION_SEED
    )
    parser.add_argument("--sample-order-seed", type=int, default=SAMPLE_ORDER_SEED)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument(
        "--architecture-variant",
        choices=ARCHITECTURE_VARIANTS,
        default="adaptive_budget_progress_mixing_v3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_remstnet(
        correction_train_manifest_path=args.correction_train_manifest,
        direct_checkpoint_path=args.direct_checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        epochs=args.epochs,
        physical_batch_size=args.physical_batch_size,
        initialization_seed=args.initialization_seed,
        sample_order_seed=args.sample_order_seed,
        max_steps_per_epoch=args.max_steps_per_epoch,
        architecture_variant=args.architecture_variant,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADAPTIVE_BUDGET_ABLATION_PROTOCOL",
    "ADAPTIVE_PROTOCOL",
    "ARCHITECTURE_VARIANTS",
    "COORDINATED_PROTOCOL",
    "PROGRESS_MIXING_ABLATION_PROTOCOL",
    "PROTOCOL",
    "load_remstnet_checkpoint",
    "train_remstnet",
]
