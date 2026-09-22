"""Conditional five-fold target-domain head fitting on declared frozen features.

The feature cache contains ``features`` (N by D), ``target`` (N), per-row
``sample_ids``, ``group_ids`` and ``condition`` lists, ``seed``, ``method`` and
an explicit ``feature_definition``. It must declare ``dataset="industrial"``
and ``scope="target_adaptation"`` and contain the complete 1,395-image,
52-group Industrial roster, with exactly the six formal conditions per image.
Each image retains the same target and acquisition group across conditions.
The author supplies the acquisition-group partition as JSON:
``{"group_to_fold": {"group-id": 0, ...}}``. Optimizer,
loss and activation settings are supplied explicitly, not inferred from a
checkpoint name or substituted for an undocumented historical experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS


EXPECTED_IMAGES = 1395
EXPECTED_GROUPS = 52


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class FeatureHead(nn.Module):
    def __init__(self, features: int, activation: str) -> None:
        super().__init__()
        _require(activation in ("gelu", "relu"), "activation must be gelu or relu")
        activate = nn.GELU if activation == "gelu" else nn.ReLU
        self.network = nn.Sequential(
            nn.Linear(features, 128), activate(), nn.Dropout(.1),
            nn.Linear(128, 64), activate(), nn.Dropout(.1),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def load_features(path: Path, *, method: str, seed: int) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "feature payload must be a mapping")
    _require(payload.get("dataset") == "industrial", "feature-cache dataset must be industrial")
    _require(payload.get("scope") == "target_adaptation", "feature-cache scope must be target_adaptation")
    _require(payload.get("method") == method, "feature-cache method differs")
    _require(payload.get("seed") == seed, "feature-cache seed is missing or differs")
    definition = payload.get("feature_definition")
    _require(isinstance(definition, str) and bool(definition.strip()),
             "supply the actual feature_definition in the cache")
    features, target = payload.get("features"), payload.get("target")
    _require(isinstance(features, torch.Tensor) and features.ndim == 2,
             "features must be an N by D tensor")
    rows, width = features.shape
    _require(rows > 0 and width > 0 and bool(torch.isfinite(features).all()),
             "features must be nonempty and finite")
    _require(isinstance(target, torch.Tensor) and tuple(target.shape) == (rows,),
             "target must be an N-vector")
    _require(bool(torch.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()),
             "targets must be finite normalized readings")
    for name in ("sample_ids", "group_ids", "condition"):
        values = payload.get(name)
        _require(isinstance(values, (tuple, list)) and len(values) == rows,
                 f"{name} must contain one value per row")
        _require(all(isinstance(v, str) and bool(v) for v in values),
                 f"{name} must contain nonempty strings")
    identities: set[tuple[str, str]] = set()
    groups: dict[str, str] = {}
    targets: dict[str, float] = {}
    conditions: dict[str, set[str]] = {}
    for sample, group, condition, value in zip(payload["sample_ids"], payload["group_ids"], payload["condition"], target.tolist(), strict=True):
        _require(condition in CONDITIONS, f"unknown evaluation condition: {condition}")
        _require((sample, condition) not in identities, "duplicate image-condition row")
        identities.add((sample, condition))
        _require(sample not in groups or groups[sample] == group,
                 "one image appears in different acquisition groups")
        _require(sample not in targets or targets[sample] == value,
                 "one image has different targets across conditions")
        groups[sample] = group
        targets[sample] = value
        conditions.setdefault(sample, set()).add(condition)
    _require(len(groups) == EXPECTED_IMAGES, f"Industrial cache must contain {EXPECTED_IMAGES} unique images")
    _require(len(set(groups.values())) == EXPECTED_GROUPS,
             f"Industrial cache must contain {EXPECTED_GROUPS} acquisition groups")
    _require(all(value == set(CONDITIONS) for value in conditions.values()),
             "each Industrial image must have all six formal conditions")
    result = dict(payload)
    result["features"] = features.detach().cpu().float()
    result["target"] = target.detach().cpu().float()
    return result


def load_folds(path: Path, group_ids: list[str]) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("group_to_fold")
    _require(isinstance(mapping, dict), "fold JSON requires group_to_fold")
    _require(set(mapping) == set(group_ids), "fold mapping must cover exactly the cache's acquisition groups")
    _require(all(type(v) is int and 0 <= v < 5 for v in mapping.values()),
             "fold indices must be integers from zero to four")
    _require(set(mapping.values()) == set(range(5)), "all five outer folds must contain groups")
    return mapping


def fold_partition(group_ids: list[str], group_to_fold: Mapping[str, int],
                   outer_fold: int) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    validation_fold = (outer_fold + 1) % 5
    assignment = torch.tensor([group_to_fold[group] for group in group_ids])
    masks = {
        "train": (assignment != outer_fold) & (assignment != validation_fold),
        "validation": assignment == validation_fold,
        "test": assignment == outer_fold,
    }
    groups = {name: sorted({group for group, include in zip(group_ids, mask.tolist(), strict=True) if include})
              for name, mask in masks.items()}
    return masks, groups


def training_normalization(features: torch.Tensor, train_mask: torch.Tensor,
                           enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return torch.zeros(features.shape[1]), torch.ones(features.shape[1])
    train = features[train_mask]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False)
    return mean, torch.where(scale > 0, scale, torch.ones_like(scale))


def selection_key(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    error = torch.abs(prediction - target)
    _require(bool(torch.isfinite(error).all()), "validation predictions are non-finite")
    return (float((error <= .05).float().mean()), float((error <= .02).float().mean()), -float(error.mean()))


def _fit_head(train_features: torch.Tensor, train_target: torch.Tensor,
              validation_features: torch.Tensor, validation_target: torch.Tensor,
              *, seed: int, activation: str, learning_rate: float, weight_decay: float,
              batch_size: int, loss: str, beta: float | None,
              device: torch.device) -> tuple[FeatureHead, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FeatureHead(train_features.shape[1], activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_features, train_target), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    validation_features = validation_features.to(device)
    validation_target = validation_target.to(device)
    best_key = (-math.inf, -math.inf, -math.inf)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        total = 0.0
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            objective = F.l1_loss(prediction, target) if loss == "l1" else F.smooth_l1_loss(prediction, target, beta=float(beta))
            _require(bool(torch.isfinite(objective)), "head training loss is non-finite")
            objective.backward()
            optimizer.step()
            total += float(objective.detach().cpu()) * len(target)
        model.eval()
        with torch.inference_mode():
            key = selection_key(model(validation_features), validation_target)
        history.append({"epoch": epoch, "training_loss": total / len(train_target),
                        "validation_acc5": key[0], "validation_acc2": key[1],
                        "validation_nmae": -key[2]})
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    _require(best_state is not None, "no validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {"head_seed": seed, "selected_epoch": best_epoch,
                   "selection_key": list(best_key), "history": history}


def train_oof(cache_path: Path, folds_path: Path, output_dir: Path, *, method: str,
              seed: int, learning_rate: float, weight_decay: float, batch_size: int,
              loss: str, activation: str, beta: float | None = None,
              standardize: bool = False, device: str = "cpu") -> dict[str, Any]:
    _require(math.isfinite(learning_rate) and learning_rate > 0, "provide the recorded positive learning rate")
    _require(math.isfinite(weight_decay) and weight_decay >= 0, "provide the recorded nonnegative weight decay")
    _require(batch_size > 0, "provide the recorded positive batch size")
    _require(loss in ("l1", "smooth_l1"), "loss must be l1 or smooth_l1")
    _require(activation in ("gelu", "relu"), "activation must be gelu or relu")
    if loss == "smooth_l1":
        _require(beta is not None and math.isfinite(beta) and beta > 0, "smooth_l1 requires the recorded positive beta")
    payload = load_features(cache_path, method=method, seed=seed)
    folds = load_folds(folds_path, payload["group_ids"])
    features, target = payload["features"], payload["target"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "protocol": "geoprr_revision_frozen_feature_group_oof_v1", "method": method, "seed": seed,
        "scope": "supervised_target_group_oof", "feature_definition": payload["feature_definition"],
        "feature_width": features.shape[1], "cache_path": str(Path(cache_path).resolve()),
        "source_checkpoint": payload.get("source_checkpoint"),
        "folds_path": str(Path(folds_path).resolve()), "outer_folds": 5,
        "validation_fold_rule": "(outer_fold + 1) % 5", "hidden_widths": [128, 64],
        "activation": activation, "dropout": .1, "output": "sigmoid",
        "maximum_epochs": 100, "patience": 20, "optimizer": "AdamW",
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "batch_size": batch_size, "loss": loss, "smooth_l1_beta": beta if loss == "smooth_l1" else None,
        "standardization": "training-fold mean and population SD" if standardize else "none",
        "selection": "inner-validation Acc5, Acc2, negative NMAE; earlier epoch wins exact ties",
        "head_seed_rule": "source seed + 1009 * outer_fold", "source_features_frozen": True,
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    predictions = torch.full_like(target, torch.nan)
    assigned = torch.zeros(len(target), dtype=torch.bool)
    outer_folds = torch.full((len(target),), -1, dtype=torch.long)
    records = []
    target_device = torch.device(device)
    for outer_fold in range(5):
        masks, groups = fold_partition(payload["group_ids"], folds, outer_fold)
        mean, scale = training_normalization(features, masks["train"], standardize)
        normalized = (features - mean) / scale
        model, training = _fit_head(
            normalized[masks["train"]], target[masks["train"]],
            normalized[masks["validation"]], target[masks["validation"]],
            seed=seed + 1009 * outer_fold, activation=activation,
            learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size,
            loss=loss, beta=beta, device=target_device,
        )
        test_indices = torch.nonzero(masks["test"], as_tuple=True)[0]
        _require(not bool(assigned[test_indices].any()), "OOF test rows were assigned twice")
        with torch.inference_mode():
            for indices in test_indices.split(batch_size):
                predictions[indices] = model(normalized[indices].to(target_device)).cpu()
        assigned[test_indices] = True
        outer_folds[test_indices] = outer_fold
        record = {"outer_fold": outer_fold, "validation_fold": (outer_fold + 1) % 5,
                  "groups": groups, "rows": {name: int(mask.sum()) for name, mask in masks.items()},
                  **training}
        records.append(record)
        torch.save({"training_config": config, **record, "normalization_mean": mean,
                    "normalization_scale": scale,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                   output_dir / f"fold_{outer_fold}.pt")
    _require(bool(assigned.all()) and bool((outer_folds >= 0).all()), "OOF roster is incomplete")
    (output_dir / "fold_assignments.json").write_text(
        json.dumps({"group_to_fold": folds, "folds": records}, indent=2) + "\n", encoding="utf-8")
    valid = torch.isfinite(predictions) & (predictions >= 0) & (predictions <= 1)
    csv_path = output_dir / "predictions.csv"
    fields = ["method", "sample_id", "group_id", "condition", "seed", "outer_fold", "target", "prediction", "error", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(target)):
            success = bool(valid[i])
            writer.writerow({"method": method, "sample_id": payload["sample_ids"][i],
                             "group_id": payload["group_ids"][i], "condition": payload["condition"][i],
                             "seed": seed, "outer_fold": int(outer_folds[i]), "target": float(target[i]),
                             "prediction": float(predictions[i]) if success else "",
                             "error": float(torch.abs(predictions[i] - target[i])) if success else 1.0,
                             "status": "success" if success else "failure"})
    metadata = {"protocol": config["protocol"], "scope": config["scope"], "method": method,
                "seed": seed, "dataset": payload.get("dataset"),
                "availability_policy": payload.get("availability_policy"),
                "feature_definition": payload["feature_definition"],
                "rows": len(target), "images": len(set(payload["sample_ids"])),
                "groups": len(folds), "failures": int((~valid).sum()),
                "outer_folds": 5, "output_csv": str(csv_path)}
    (output_dir / "predictions.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--loss", choices=("l1", "smooth_l1"), required=True)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--activation", choices=("gelu", "relu"), required=True)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(train_oof(args.cache, args.folds, args.output, method=args.method,
                              seed=args.seed, learning_rate=args.lr, weight_decay=args.weight_decay,
                              batch_size=args.batch_size, loss=args.loss, activation=args.activation,
                              beta=args.beta, standardize=args.standardize, device=args.device), indent=2))


if __name__ == "__main__":
    main()
