"""Fit a small risk router from source OOF caches, selecting on validation only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from experiments.geoprr_v2_router import FEATURE_PROTOCOL, ROUTER_PROTOCOL, ReliabilityRouter
from experiments.train_geoprr_v2 import prediction_rows, summarize_rows, write_json


def load_cache(path: Path, role: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key] for key in ("features", "expert_predictions", "field_valid", "target", "sample_ids", "group_ids", "scene_ids")}
    if (metadata.get("role") != role or metadata.get("dataset", "").lower() != "syncg"
            or metadata.get("split") != "train" or not metadata.get("heldout_verified")
            or metadata.get("feature_protocol") != FEATURE_PROTOCOL):
        raise ValueError("router fitting requires verified SyncG source-training OOF/validation caches; test caches are forbidden")
    if not metadata.get("fold_id"):
        raise ValueError("cache must identify its held-out fold")
    count = len(arrays["target"])
    if not count or any(len(value) != count for value in arrays.values()):
        raise ValueError("cache rows must be nonempty and aligned")
    if arrays["expert_predictions"].shape != (count, 2) or arrays["features"].ndim != 2:
        raise ValueError("cache must contain two experts and a feature matrix")
    if not all(np.isfinite(arrays[key]).all() for key in ("features", "expert_predictions", "target")):
        raise ValueError("cache contains nonfinite model outputs or targets")
    ids, scenes = set(arrays["sample_ids"].tolist()), set(arrays["scene_ids"].tolist())
    if ids != set(metadata.get("heldout_sample_ids", [])) or not metadata.get("reader_train_sample_ids"):
        raise ValueError("cache must preserve reader train and held-out sample identities")
    for prefix in ("reader_train", "reader_validation") if role == "oof_train" else ("reader_train",):
        if ids & set(metadata.get(f"{prefix}_sample_ids", [])) or scenes & set(metadata.get(f"{prefix}_scene_ids", [])):
            raise ValueError("cached held-out data overlaps reader fitting or selection data")
    return arrays, metadata


def load_fit_data(train_paths: list[Path], validation_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    caches = [load_cache(path, "oof_train") for path in train_paths]
    validation, validation_meta = load_cache(validation_path, "validation")
    provenance = [metadata for _, metadata in caches] + [validation_meta]
    if any(metadata["reader_config"] != validation_meta["reader_config"] for metadata in provenance):
        raise ValueError("OOF and validation readers must have the same model configuration")
    if any(metadata["reader_training"] != validation_meta["reader_training"] for metadata in provenance):
        raise ValueError("OOF and validation readers must have the same optimization and loss settings")
    if any(metadata["reader_seed"] != validation_meta["reader_seed"] for metadata in provenance):
        raise ValueError("OOF and validation readers must use the same experiment seed")
    train = {key: np.concatenate([arrays[key] for arrays, _ in caches]) for key in validation}
    for population in (train, validation):
        if len(set(population["sample_ids"].tolist())) != len(population["sample_ids"]):
            raise ValueError("duplicate sample IDs in router caches")
    for key in ("sample_ids", "group_ids", "scene_ids"):
        if set(train[key].tolist()) & set(validation[key].tolist()):
            raise ValueError(f"router train/validation overlap in {key}")
    return train, validation, provenance


def validation_metrics(router: ReliabilityRouter, data: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    with torch.no_grad():
        risks = router(torch.as_tensor(data["features"], device=device))
        selected = risks.argmin(1)
        valid = torch.as_tensor(data["field_valid"], device=device).bool()
        selected[~valid] = 1
        predictions = torch.as_tensor(data["expert_predictions"], device=device)
        reading = predictions[torch.arange(len(selected), device=device), selected]
        rows = prediction_rows({"reading": reading, "field_valid": valid},
                               {"y": torch.as_tensor(data["target"], device=device),
                                "sample_id": data["sample_ids"], "group_id": data["group_ids"]})
    return dict(summarize_rows(rows), selected_direct=int((selected == 1).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", action="append", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(20262020, 20262021, 20262022), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1:
        parser.error("epochs, patience, batch size and hidden dimension must be positive")
    train, validation, provenance = load_fit_data(args.train_cache, args.validation_cache)
    if any(metadata["reader_seed"] != args.seed for metadata in provenance):
        raise ValueError("router seed must match the experiment seed used by all cached readers")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    features = torch.as_tensor(train["features"], device=device, dtype=torch.float32)
    errors = 100 * np.abs(train["expert_predictions"] - train["target"][:, None])
    errors[~train["field_valid"].astype(bool), 0] = 100
    targets = torch.as_tensor(errors, device=device, dtype=torch.float32)
    router = ReliabilityRouter(features.shape[1], args.hidden_dim).to(device)
    router.feature_mean.copy_(features.mean(0))
    router.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_epoch, best_state, best_summary = float("inf"), 0, None, None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            router.train()
            indices, losses = torch.randperm(len(features), device=device), []
            for batch in indices.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(router(features[batch]), targets[batch], beta=1.0)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()) * len(batch))
            router.eval()
            summary = validation_metrics(router, validation, device)
            event = {"epoch": epoch, "risk_huber_loss": sum(losses) / len(features), "validation": summary}
            history.write(json.dumps(event, allow_nan=False) + "\n")
            history.flush()
            if summary["nmae"] < best:
                best, best_epoch, best_summary = summary["nmae"], epoch, summary
                best_state = copy.deepcopy(router.state_dict())
            if epoch - best_epoch >= args.patience:
                break
    checkpoint = {"protocol": ROUTER_PROTOCOL, "feature_protocol": FEATURE_PROTOCOL, "feature_dim": features.shape[1],
                  "hidden_dim": args.hidden_dim, "model": best_state, "seed": args.seed, "epoch": best_epoch,
                  "risk_unit": "percent_full_scale", "reader_config": provenance[-1]["reader_config"],
                  "source_caches": provenance, "selection": "source_validation_hard_selection_nmae"}
    torch.save(checkpoint, args.output / "router.pt")
    result = {"epoch": best_epoch, "seed": args.seed, "train_samples": len(features), "validation": best_summary}
    write_json(args.output / "best_validation.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
