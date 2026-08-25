"""Train representation-conditioned arbitration over three frozen RCMT heads."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.r2mt_net import (
    ARCHITECTURE,
    DEFAULT_PRIOR_WEIGHTS,
    RISK_HEAD_NAMES,
    RiskArbitratedRelativeContextMomentTransport,
    parameter_counts,
    publication_model_identity,
)
from experiments.train_raw_context_residual_correction_adaptation import (
    load_context_correction_adapted_remst,
)
from experiments.train_remst_resnet18_relative_context_transport import (
    ARCHITECTURE as RCMT_ARCHITECTURE,
    PROTOCOL as RCMT_PROTOCOL,
)


PROTOCOL: Final[str] = "remst_resnet18_risk_conditioned_moment_arbitration_v1"
PUBLICATION_PROTOCOL: Final[str] = "r2mt_net_training_v1"
DEFAULT_EPOCHS: Final[int] = 40
DEFAULT_BATCH_SIZE: Final[int] = 256
DEFAULT_LEARNING_RATE: Final[float] = 1.0e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_CONDITION_WEIGHTS: Final[tuple[float, ...]] = (1.0, 1.0, 1.5)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_rcmt_payload(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") == RCMT_PROTOCOL
        and payload.get("architecture") == RCMT_ARCHITECTURE
        and isinstance(payload.get("construction"), Mapping)
        and isinstance(payload.get("context_head_state"), Mapping),
        f"risk arbitration expert differs: {source}",
    )
    return dict(payload)


def _build_model(
    expert_checkpoint_paths: Sequence[Path],
    *,
    device: torch.device,
    gate_latent_features: int,
    gate_hidden_features: int,
    prior_weights: Sequence[float],
    gate_representation_enabled: bool,
) -> tuple[torch.nn.Module, RiskArbitratedRelativeContextMomentTransport, dict[str, Any]]:
    _require(
        len(expert_checkpoint_paths) == len(RISK_HEAD_NAMES),
        "risk arbitration requires three experts",
    )
    payloads = tuple(_load_rcmt_payload(path) for path in expert_checkpoint_paths)
    source_checkpoint = str(payloads[0]["source_checkpoint"])
    construction = dict(payloads[0]["construction"])
    for payload in payloads[1:]:
        _require(
            str(payload["source_checkpoint"]) == source_checkpoint
            and dict(payload["construction"]) == construction,
            "risk arbitration experts do not share one source/construction",
        )
    anchor, correction, source_metadata = load_context_correction_adapted_remst(
        Path(source_checkpoint), device=device
    )
    model = RiskArbitratedRelativeContextMomentTransport(
        correction,
        progress_bins=int(construction["progress_bins"]),
        expert_latent_features=int(construction["latent_features"]),
        expert_hidden_features=int(construction["hidden_features"]),
        max_progress_shift=float(construction["max_progress_shift"]),
        gate_latent_features=int(gate_latent_features),
        gate_hidden_features=int(gate_hidden_features),
        prior_weights=prior_weights,
        gate_representation_enabled=bool(gate_representation_enabled),
    ).to(device)
    for head, payload in zip(model.risk_heads, payloads):
        incompatibility = head.load_state_dict(payload["context_head_state"], strict=True)
        _require(
            not incompatibility.missing_keys and not incompatibility.unexpected_keys,
            "risk arbitration expert state does not load strictly",
        )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base_correction.parameters():
        parameter.requires_grad_(False)
    for head in model.risk_heads:
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    anchor.eval()
    model.base_correction.eval()
    for head in model.risk_heads:
        head.eval()
    return anchor, model, {
        "source_checkpoint": source_checkpoint,
        "source_metadata": source_metadata,
        "expert_checkpoints": [
            str(Path(path).resolve()) for path in expert_checkpoint_paths
        ],
        "expert_training_losses": [payload.get("loss") for payload in payloads],
        "expert_construction": construction,
    }


def train_r2mt_net(
    *,
    expert_checkpoint_paths: Sequence[Path],
    cache_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gate_latent_features: int = 32,
    gate_hidden_features: int = 64,
    gate_representation_enabled: bool = True,
    prior_weights: Sequence[float] = DEFAULT_PRIOR_WEIGHTS,
    condition_weights: Sequence[float] = DEFAULT_CONDITION_WEIGHTS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    tail_weight: float = 0.1,
    consistency_weight: float = 0.02,
    route_imitation_weight: float = 0.0025,
    route_temperature: float = 0.0025,
    prior_kl_weight: float = 0.001,
) -> dict[str, Any]:
    _require(epochs >= 1 and batch_size >= 1, "risk arbitration sizes differ")
    _require(
        learning_rate > 0.0
        and weight_decay >= 0.0
        and tail_weight >= 0.0
        and consistency_weight >= 0.0
        and route_imitation_weight >= 0.0
        and route_temperature > 0.0
        and prior_kl_weight >= 0.0,
        "risk arbitration optimization differs",
    )
    condition_weights = tuple(float(value) for value in condition_weights)
    _require(
        len(condition_weights) == 3 and all(value > 0.0 for value in condition_weights),
        "risk arbitration condition weights differ",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"risk arbitration output exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    anchor, model, source_metadata = _build_model(
        expert_checkpoint_paths,
        device=device,
        gate_latent_features=int(gate_latent_features),
        gate_hidden_features=int(gate_hidden_features),
        prior_weights=prior_weights,
        gate_representation_enabled=bool(gate_representation_enabled),
    )
    cache_source = Path(cache_path).resolve()
    cache_payload = torch.load(cache_source, map_location="cpu", weights_only=False)
    _require(
        isinstance(cache_payload, Mapping)
        and cache_payload.get("source_checkpoint")
        == str(Path(source_metadata["source_checkpoint"]).resolve())
        and isinstance(cache_payload.get("cache"), Mapping),
        "risk arbitration cache differs",
    )
    cache = cache_payload["cache"]
    dataset = TensorDataset(
        cache["raw_representation"],
        cache["sarn_representation"],
        cache["geometry_features"],
        cache["raw_mean"],
        cache["sarn_mean"],
        cache["base_mean"],
        cache["target"],
        cache["active"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(int(seed) + 91_000),
    )
    optimizer = torch.optim.AdamW(
        model.risk_gate.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    weight_tensor = torch.as_tensor(
        condition_weights, dtype=torch.float32, device=device
    )[None]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.risk_gate.train(True)
        totals: defaultdict[str, float] = defaultdict(float)
        rows_seen = 0
        epoch_started = time.perf_counter()
        for batch in loader:
            (
                raw_representation,
                sarn_representation,
                geometry_features,
                raw_mean,
                sarn_mean,
                base_mean,
                target,
                active,
            ) = tuple(value.to(device, non_blocking=device.type == "cuda") for value in batch)
            physical = int(target.numel())
            flat_arguments = (
                raw_representation.flatten(0, 1),
                sarn_representation.flatten(0, 1),
                geometry_features.flatten(0, 1),
                raw_mean.flatten(),
                sarn_mean.flatten(),
                base_mean.flatten(),
            )
            with torch.no_grad():
                expert_shifts = torch.stack(
                    [head(*flat_arguments) for head in model.risk_heads], dim=1
                )
            optimizer.zero_grad(set_to_none=True)
            weights, _ = model.risk_gate(*flat_arguments, expert_shifts)
            active_flat = active.bool().flatten()
            applied_shift = torch.where(
                active_flat,
                (weights * expert_shifts).sum(dim=1),
                torch.zeros_like(base_mean.flatten()),
            )
            prediction = (base_mean.flatten() + applied_shift).clamp(0.0, 1.0)
            repeated_target = target[:, None].expand(-1, 3).flatten()
            errors = torch.abs(prediction - repeated_target)
            row_weights = weight_tensor.expand(physical, -1).flatten()
            weighted_exact = (
                (errors[active_flat] * row_weights[active_flat]).sum()
                / row_weights[active_flat].sum()
            )
            active_errors = errors[active_flat]
            tail_count = max(1, math.ceil(0.25 * int(active_errors.numel())))
            tail = torch.topk(active_errors, k=tail_count).values.mean()
            complete = active.bool().all(dim=1)
            prediction_triplet = prediction.reshape(physical, 3)
            if bool(complete.any()):
                selected = prediction_triplet[complete]
                consistency = (
                    torch.abs(selected[:, 0] - selected[:, 1]).mean()
                    + torch.abs(selected[:, 0] - selected[:, 2]).mean()
                    + torch.abs(selected[:, 1] - selected[:, 2]).mean()
                ) / 3.0
            else:
                consistency = prediction.sum() * 0.0
            expert_prediction = base_mean.flatten()[:, None] + expert_shifts
            expert_errors = torch.abs(
                expert_prediction - repeated_target[:, None]
            )
            route_target = torch.softmax(
                -expert_errors.detach() / float(route_temperature), dim=1
            )
            route_imitation = -(
                route_target[active_flat]
                * torch.log(weights[active_flat].clamp_min(1.0e-8))
            ).sum(dim=1).mean()
            prior = model.risk_gate.prior_weights[None]
            prior_kl = (
                weights[active_flat]
                * (
                    torch.log(weights[active_flat].clamp_min(1.0e-8))
                    - torch.log(prior)
                )
            ).sum(dim=1).mean()
            loss = (
                weighted_exact
                + float(tail_weight) * tail
                + float(consistency_weight) * consistency
                + float(route_imitation_weight) * route_imitation
                + float(prior_kl_weight) * prior_kl
            )
            _require(bool(torch.isfinite(loss)), "risk arbitration loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.risk_gate.parameters(), max_norm=5.0)
            optimizer.step()
            metrics = {
                "loss": loss,
                "weighted_exact_l1": weighted_exact,
                "tail_l1": tail,
                "consistency": consistency,
                "route_imitation": route_imitation,
                "prior_kl": prior_kl,
                "mean_risk_weight": weights[:, 0].mean(),
                "tail_risk_weight": weights[:, 1].mean(),
                "combined_risk_weight": weights[:, 2].mean(),
            }
            for name, value in metrics.items():
                totals[name] += float(value.detach().cpu()) * physical
            rows_seen += physical
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "metrics": {
                name: value / rows_seen for name, value in sorted(totals.items())
            },
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    model.risk_gate.eval()
    counts = parameter_counts(model)
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "source_checkpoint": str(Path(source_metadata["source_checkpoint"]).resolve()),
        "source_metadata": source_metadata["source_metadata"],
        "expert_checkpoints": source_metadata["expert_checkpoints"],
        "expert_training_losses": source_metadata["expert_training_losses"],
        "seed": int(seed),
        "epochs": int(epochs),
        "checkpoint_selection": "terminal_fixed_epoch",
        "training_scope": {
            "complete_original_fit": True,
            "source_anchor_frozen": True,
            "source_correction_frozen": True,
            "risk_experts_frozen": True,
            "formal_holdout_access": False,
            "main_table_development_access": False,
        },
        "training_data": {
            "cache": str(cache_source),
            "manifest": str(cache_payload.get("population_manifest")),
            "physical_samples": len(dataset),
            "conditions": [
                "perspective_moderate",
                "perspective_severe",
                "combined_severe",
            ],
        },
        "construction": {
            **source_metadata["expert_construction"],
            "gate_latent_features": int(gate_latent_features),
            "gate_hidden_features": int(gate_hidden_features),
            "gate_representation_enabled": bool(gate_representation_enabled),
            "prior_weights": [float(value) for value in prior_weights],
            "risk_head_names": list(RISK_HEAD_NAMES),
            "additional_image_encoders": 0,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
        },
        "loss": {
            "condition_weights": list(condition_weights),
            "tail_weight": float(tail_weight),
            "consistency_weight": float(consistency_weight),
            "route_imitation_weight": float(route_imitation_weight),
            "route_temperature": float(route_temperature),
            "prior_kl_weight": float(prior_kl_weight),
        },
        "parameter_counts": counts,
        "history": history,
        "training_elapsed_seconds": time.perf_counter() - started,
        "risk_head_states": {
            name: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            for name, head in zip(RISK_HEAD_NAMES, model.risk_heads)
        },
        "risk_gate_state": {
            key: value.detach().cpu().clone()
            for key, value in model.risk_gate.state_dict().items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    del anchor
    return {
        "status": "complete",
        "checkpoint": str(output),
        "training_elapsed_seconds": checkpoint["training_elapsed_seconds"],
        "parameter_counts": counts,
        "terminal_metrics": history[-1]["metrics"],
    }


def load_r2mt_net_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[torch.nn.Module, RiskArbitratedRelativeContextMomentTransport, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") == PROTOCOL
        and payload.get("architecture") == ARCHITECTURE
        and isinstance(payload.get("construction"), Mapping)
        and isinstance(payload.get("risk_head_states"), Mapping)
        and isinstance(payload.get("risk_gate_state"), Mapping),
        "risk arbitration checkpoint differs",
    )
    construction = payload["construction"]
    anchor, correction, source_metadata = load_context_correction_adapted_remst(
        Path(str(payload["source_checkpoint"])), device=device
    )
    model = RiskArbitratedRelativeContextMomentTransport(
        correction,
        progress_bins=int(construction["progress_bins"]),
        expert_latent_features=int(construction["latent_features"]),
        expert_hidden_features=int(construction["hidden_features"]),
        max_progress_shift=float(construction["max_progress_shift"]),
        gate_latent_features=int(construction["gate_latent_features"]),
        gate_hidden_features=int(construction["gate_hidden_features"]),
        prior_weights=construction["prior_weights"],
        gate_representation_enabled=bool(
            construction.get("gate_representation_enabled", True)
        ),
    ).to(device)
    for name, head in zip(RISK_HEAD_NAMES, model.risk_heads):
        head.load_state_dict(payload["risk_head_states"][name], strict=True)
    model.risk_gate.load_state_dict(payload["risk_gate_state"], strict=True)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    model.eval()
    metadata = {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "source_checkpoint": str(payload["source_checkpoint"]),
        "source_metadata": source_metadata,
        "seed": int(payload["seed"]),
        "epochs": int(payload["epochs"]),
        "checkpoint_selection": str(payload["checkpoint_selection"]),
        "training_scope": dict(payload["training_scope"]),
        "construction": dict(construction),
        "parameter_counts": dict(payload["parameter_counts"]),
        "publication_protocol": PUBLICATION_PROTOCOL,
        "publication_model": publication_model_identity(),
    }
    return anchor, model, metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-risk-checkpoint", type=Path, required=True)
    parser.add_argument("--tail-risk-checkpoint", type=Path, required=True)
    parser.add_argument("--combined-risk-checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gate-latent-features", type=int, default=32)
    parser.add_argument("--gate-hidden-features", type=int, default=64)
    parser.add_argument("--disable-gate-representation", action="store_true")
    parser.add_argument("--prior-weights", type=float, nargs=3, default=DEFAULT_PRIOR_WEIGHTS)
    parser.add_argument(
        "--condition-weights", type=float, nargs=3, default=DEFAULT_CONDITION_WEIGHTS
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--tail-weight", type=float, default=0.1)
    parser.add_argument("--consistency-weight", type=float, default=0.02)
    parser.add_argument("--route-imitation-weight", type=float, default=0.0025)
    parser.add_argument("--route-temperature", type=float, default=0.0025)
    parser.add_argument("--prior-kl-weight", type=float, default=0.001)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_r2mt_net(
        expert_checkpoint_paths=(
            args.mean_risk_checkpoint,
            args.tail_risk_checkpoint,
            args.combined_risk_checkpoint,
        ),
        cache_path=args.cache,
        output_path=args.output,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gate_latent_features=args.gate_latent_features,
        gate_hidden_features=args.gate_hidden_features,
        gate_representation_enabled=not args.disable_gate_representation,
        prior_weights=args.prior_weights,
        condition_weights=args.condition_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        tail_weight=args.tail_weight,
        consistency_weight=args.consistency_weight,
        route_imitation_weight=args.route_imitation_weight,
        route_temperature=args.route_temperature,
        prior_kl_weight=args.prior_kl_weight,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Compatibility aliases for internal ledgers created before the public rename.
load_risk_arbitration = load_r2mt_net_checkpoint
train_risk_arbitration = train_r2mt_net


__all__ = [
    "PROTOCOL",
    "PUBLICATION_PROTOCOL",
    "load_r2mt_net_checkpoint",
    "train_r2mt_net",
    "load_risk_arbitration",
    "train_risk_arbitration",
]
