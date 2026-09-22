"""Fit and evaluate the revision controls on frozen, explicitly scoped caches.

A cache payload contains ``cache`` (the twelve ROUTER_ARGUMENT_NAMES tensors,
``target`` and ``condition_index``), per-row ``sample_ids`` and ``group_ids``,
the six ``conditions``, an integer ``seed``, and ``scope`` equal to
``source_train`` or ``evaluation``. Optional ``cache.full_prediction`` stores
the complete reader's predictions for a matched, inference-only comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.pcrt_meter import ConditionalRegretRouter
from experiments.refine_pcrt_minimax_router import ROUTER_ARGUMENT_NAMES
from experiments.resnet18_direct_progress import _configure_reproducibility
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.train_unified_pointer_reader import (
    ParameterEMA,
    _train_router_stage_one,
    _train_router_stage_two,
)
from experiments.unified_pointer_reader import FULL


TRAINED_METHODS: Final[tuple[str, ...]] = ("linear_gate", "scalar_gate", "dual_view")
METHODS: Final[tuple[str, ...]] = TRAINED_METHODS + (
    "uniform", "fixed_prior", "base_only", "polar_only", "relational_only", "full",
)
PROTOCOL: Final[str] = "geoprr_revision_frozen_cache_controls_v1"
CSV_FIELDS: Final[tuple[str, ...]] = (
    "method", "sample_id", "group_id", "condition", "seed", "target",
    "prediction", "error", "status",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class LinearGainRouter(ConditionalRegretRouter):
    """Keep the shared 1280-to-32 projection and all 220 router inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.regret_network = nn.Sequential(nn.LayerNorm(220), nn.Linear(220, 2))
        nn.init.zeros_(self.regret_network[-1].weight)
        nn.init.zeros_(self.regret_network[-1].bias)


class ScalarGainRouter(nn.Module):
    """Use only the full router's eleven scalar inputs, without a projection."""

    def __init__(self) -> None:
        super().__init__()
        self.maximum_gain = 0.10
        self.gain_logit_scale = 100.0
        self.regret_network = nn.Sequential(
            nn.LayerNorm(11), nn.Linear(11, 96), nn.GELU(), nn.Linear(96, 2),
        )
        nn.init.zeros_(self.regret_network[-1].weight)
        nn.init.zeros_(self.regret_network[-1].bias)
        prior = torch.tensor([0.50, 0.25, 0.25])
        self.register_buffer("prior_weights", prior)
        self.register_buffer("prior_logits", prior.log())

    def forward(self, *arguments: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, polar, relation = arguments[3:6]
        scalar = torch.stack((
            base, polar, relation, polar - base, relation - base, polar - relation,
            arguments[6], arguments[7], arguments[8], arguments[10], arguments[11],
        ), dim=1).float()
        gains = self.maximum_gain * torch.tanh(self.regret_network(scalar))
        all_gains = torch.cat((torch.zeros_like(gains[:, :1]), gains), dim=1)
        logits = self.prior_logits[None] + self.gain_logit_scale * all_gains
        return torch.softmax(logits, dim=1), gains, logits


class DualViewHead(nn.Module):
    """Read progress from concatenated frozen raw and normalized B0 contexts."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2560), nn.Linear(2560, 320), nn.GELU(),
            nn.Linear(320, 1), nn.Sigmoid(),
        )

    def forward(self, raw: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((raw.float(), normalized.float()), dim=1)).squeeze(1)


def build_control(method: str) -> nn.Module:
    constructors = {
        "linear_gate": LinearGainRouter,
        "scalar_gate": ScalarGainRouter,
        "dual_view": DualViewHead,
    }
    _require(method in constructors, f"{method} is not a trainable control")
    return constructors[method]()


def candidate_lineage(payload: Mapping[str, Any]) -> dict[str, str]:
    fields = ("candidate_lineage", "candidate_variant", "availability_policy")
    result = {field: payload.get(field) for field in fields}
    _require(all(isinstance(value, str) and bool(value) for value in result.values()),
             "cache requires candidate_lineage, candidate_variant and availability_policy from its source run")
    return result


def load_cache(path: Path, *, scope: str, seed: int) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "cache payload must be a mapping")
    _require(payload.get("scope") == scope, f"cache scope must be {scope}")
    _require("seed" in payload and payload["seed"] == seed, "cache seed is missing or differs")
    _require(tuple(payload.get("conditions", ())) == tuple(CONDITIONS), "cache condition order differs")
    cache = payload.get("cache")
    _require(isinstance(cache, Mapping), "cache tensors are missing")
    target = cache.get("target")
    _require(isinstance(target, torch.Tensor) and target.ndim == 1, "target must be a vector")
    rows = len(target)
    _require(rows > 0, "cache is empty")
    dimensions = {"raw_representation": 1280, "sarn_representation": 1280,
                  "geometry_features": 13, "polar_posterior": 36}
    for name in (*ROUTER_ARGUMENT_NAMES, "target", "condition_index"):
        tensor = cache.get(name)
        shape = (rows, dimensions[name]) if name in dimensions else (rows,)
        _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == shape,
                 f"{name} must have shape {shape}")
        if scope == "source_train":
            _require(bool(torch.isfinite(tensor).all()), f"training cache {name} is non-finite")
    _require(bool(torch.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()),
             "targets must be finite normalized readings")
    conditions = cache["condition_index"]
    _require(bool(((conditions >= 0) & (conditions < 6) & (conditions == conditions.long())).all()),
             "condition indices must be integers from zero to five")
    sample_ids, group_ids = payload.get("sample_ids"), payload.get("group_ids")
    _require(isinstance(sample_ids, (list, tuple)) and len(sample_ids) == rows,
             "sample_ids must contain one identifier per row")
    _require(isinstance(group_ids, (list, tuple)) and len(group_ids) == rows,
             "group_ids must contain one identifier per row")
    sample_groups: dict[str, str] = {}
    sample_conditions: dict[str, set[int]] = {}
    for sample, group, condition in zip(sample_ids, group_ids, conditions.tolist(), strict=True):
        _require(isinstance(sample, str) and bool(sample) and isinstance(group, str) and bool(group),
                 "sample and group identifiers must be nonempty strings")
        _require(sample not in sample_groups or sample_groups[sample] == group,
                 f"group differs across conditions for {sample}")
        sample_groups[sample] = group
        seen = sample_conditions.setdefault(sample, set())
        _require(int(condition) not in seen, f"duplicate image-condition row for {sample}")
        seen.add(int(condition))
    if scope == "source_train":
        _require(all(value == set(range(6)) for value in sample_conditions.values()),
                 "training cache must contain all six conditions for each image")
    result = dict(payload)
    result["cache"] = {name: value.detach().cpu() for name, value in cache.items()}
    result["cache"]["condition_index"] = conditions.detach().cpu().long()
    if "full_prediction" in cache:
        _require(tuple(cache["full_prediction"].shape) == (rows,), "full_prediction must be a vector")
    return result


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _fit_dual_view(model: DualViewHead, cache: Mapping[str, torch.Tensor], *,
                   device: torch.device, seed: int) -> list[dict[str, Any]]:
    data = TensorDataset(cache["raw_representation"], cache["sarn_representation"], cache["target"])
    generator = torch.Generator().manual_seed(seed + 30_001)
    loader = DataLoader(data, batch_size=512, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    history = []
    model.train()
    for epoch in range(40):
        total = 0.0
        for raw, normalized, target in loader:
            raw, normalized, target = (value.to(device) for value in (raw, normalized, target))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.abs(model(raw, normalized) - target.float()).mean()
            _require(bool(torch.isfinite(loss)), "dual-view training loss is non-finite")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
        scheduler.step()
        history.append({"epoch": epoch + 1, "normalized_l1": total / len(data),
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
    return history


def train_control(cache_path: Path, output_dir: Path, *, method: str, seed: int,
                  smooth_max_temperature: float | None = None,
                  device: str = "cpu") -> dict[str, str]:
    _require(method in TRAINED_METHODS, "select a trainable control")
    if method != "dual_view":
        _require(smooth_max_temperature is not None and smooth_max_temperature > 0,
                 "gate training requires the recorded positive smooth_max_temperature")
    payload = load_cache(cache_path, scope="source_train", seed=seed)
    lineage = candidate_lineage(payload)
    cache = payload["cache"]
    target_device = torch.device(device)
    _configure_reproducibility(seed, target_device)
    model = build_control(method).to(target_device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config: dict[str, Any] = {
        "protocol": PROTOCOL, "method": method, "seed": seed,
        "candidate_lineage": lineage,
        "cache_path": str(Path(cache_path).resolve()), "scope": "source_train",
        "rows": len(cache["target"]), "images": len(set(payload["sample_ids"])),
        "groups": len(set(payload["group_ids"])), "conditions": list(CONDITIONS),
        "encoder_training": False, "candidate_training": False,
        "control_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": "AdamW", "weight_decay": 1.0e-4, "batch_size": 512,
        "schedule": "cosine", "evaluation_used_for_selection": False,
    }
    if method == "dual_view":
        config.update(architecture="LN(2560)-Linear(2560,320)-GELU-Linear(320,1)-Sigmoid",
                      epochs=40, learning_rate=3.0e-4, loss="normalized L1",
                      checkpoint_selection="fixed_epoch_terminal")
    else:
        config.update(architecture="shared projection + LN(220)-Linear(220,2)" if method == "linear_gate"
                      else "LN(11)-Linear(11,96)-GELU-Linear(96,2)",
                      stage_one_epochs=40, stage_two_epochs=120,
                      stage_one_learning_rate=1.0e-3, stage_two_learning_rate=3.0e-4,
                      smooth_max_temperature=smooth_max_temperature,
                      maximum_gain=0.10, gain_logit_scale=100.0, prior_weights=[.50, .25, .25],
                      gradient_clip_norm=5.0, ema_decay=.995,
                      loss_implementation="experiments.train_unified_pointer_reader._train_router_stage_one/_train_router_stage_two",
                      checkpoint_selection="fixed_epoch_terminal_and_fixed_trajectory_ema")
    config_path = output_dir / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if method == "dual_view":
        history = _fit_dual_view(model, cache, device=target_device, seed=seed)
        states = {"terminal": _state(model)}
    else:
        ema = ParameterEMA(model, decay=.995)
        history = _train_router_stage_one(model, cache, variant=FULL, device=target_device,
                                         seed=seed, epochs=40, batch_size=512, ema=ema)
        history += _train_router_stage_two(model, cache, variant=FULL, conditions=CONDITIONS,
                                          device=target_device, seed=seed, epochs=120,
                                          learning_rate=3.0e-4,
                                          smooth_max_temperature=float(smooth_max_temperature), ema=ema)
        states = {"terminal": _state(model), "ema": ema.state_dict(model)}
        config["ema_updates"] = ema.updates
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    paths = {"training_config": str(config_path)}
    for selection, state in states.items():
        path = output_dir / f"{selection}.pt"
        torch.save({"protocol": PROTOCOL, "method": method, "seed": seed,
                    "weight_variant": selection, "state_dict": state,
                    "training_config": config, "history": history}, path)
        paths[f"{selection}_checkpoint"] = str(path)
    return paths


def predict_batch(method: str, cache: Mapping[str, torch.Tensor],
                  model: nn.Module | None = None) -> torch.Tensor:
    _require(method in METHODS, f"unknown control: {method}")
    if method == "full":
        _require("full_prediction" in cache, "full evaluation requires cache.full_prediction")
        return cache["full_prediction"].float()
    if method in ("base_only", "polar_only", "relational_only"):
        key = {"base_only": "base_mean", "polar_only": "polar_mean", "relational_only": "r2mt_mean"}[method]
        return cache[key].float()
    if method == "dual_view":
        _require(model is not None, "dual-view checkpoint is required")
        return model(cache["raw_representation"], cache["sarn_representation"])
    candidates = torch.stack([cache[name] for name in ("base_mean", "polar_mean", "r2mt_mean")], dim=1).float()
    if method == "uniform":
        weights = torch.full_like(candidates, 1.0 / 3.0)
    elif method == "fixed_prior":
        weights = candidates.new_tensor([.50, .25, .25])[None]
    else:
        _require(model is not None, "gate checkpoint is required")
        weights, _gains, _logits = model(*(cache[name] for name in ROUTER_ARGUMENT_NAMES))
    epsilon = torch.finfo(torch.float32).eps
    prediction = (weights * candidates).sum(dim=1)
    return torch.where(torch.isfinite(prediction), prediction.clamp(epsilon, 1.0 - epsilon), prediction)


def evaluate_control(cache_path: Path, output_csv: Path, *, method: str, seed: int,
                     checkpoint: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    payload = load_cache(cache_path, scope="evaluation", seed=seed)
    _require(method in METHODS, f"unknown control: {method}")
    output_method = payload.get("model_metadata", {}).get("variant", method) if method == "full" else method
    target_device = torch.device(device)
    model = None
    if method in TRAINED_METHODS:
        _require(checkpoint is not None, "a trained control requires --checkpoint")
        fitted = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        _require(fitted.get("protocol") == PROTOCOL and fitted.get("method") == method,
                 "checkpoint method or protocol differs")
        _require(fitted.get("seed") == seed, "checkpoint seed differs")
        _require(fitted.get("training_config", {}).get("candidate_lineage") == candidate_lineage(payload),
                 "control checkpoint and evaluation cache candidate lineage or input policy differ")
        model = build_control(method).to(target_device)
        model.load_state_dict(fitted["state_dict"], strict=True)
        model.eval()
    else:
        _require(checkpoint is None, "this cache-only control does not use a checkpoint")
    cache = payload["cache"]
    rows = len(cache["target"])
    predictions = []
    with torch.inference_mode():
        for start in range(0, rows, 512):
            batch = {name: tensor[start:start + 512].to(target_device)
                     for name, tensor in cache.items()}
            predictions.append(predict_batch(method, batch, model).cpu())
    prediction = torch.cat(predictions)
    _require(tuple(prediction.shape) == (rows,), "prediction row count differs")
    valid = torch.isfinite(prediction) & (prediction >= 0) & (prediction <= 1)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in range(rows):
            success = bool(valid[i])
            writer.writerow({
                "method": output_method, "sample_id": payload["sample_ids"][i],
                "group_id": payload["group_ids"][i],
                "condition": payload["conditions"][int(cache["condition_index"][i])],
                "seed": seed, "target": float(cache["target"][i]),
                "prediction": float(prediction[i]) if success else "",
                "error": float(torch.abs(prediction[i] - cache["target"][i])) if success else 1.0,
                "status": "success" if success else "failure",
            })
    metadata = {
        "protocol": PROTOCOL, "method": output_method, "seed": seed,
        "scope": "evaluation", "dataset": payload.get("dataset"),
        "availability_policy": payload.get("availability_policy"),
        "cache_path": str(Path(cache_path).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        "rows": rows, "images": len(set(payload["sample_ids"])),
        "groups": len(set(payload["group_ids"])), "conditions": list(CONDITIONS),
        "failures": int((~valid).sum()), "failure_normalized_error": 1.0,
        "output_csv": str(output_csv),
    }
    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "metadata": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--method", choices=TRAINED_METHODS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--smooth-max-temperature", type=float)
    train.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--output-csv", type=Path, required=True)
    evaluate.add_argument("--method", choices=METHODS, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        result = train_control(args.cache, args.output, method=args.method, seed=args.seed,
                               smooth_max_temperature=args.smooth_max_temperature, device=args.device)
    else:
        result = evaluate_control(args.cache, args.output_csv, method=args.method, seed=args.seed,
                                  checkpoint=args.checkpoint, device=args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
