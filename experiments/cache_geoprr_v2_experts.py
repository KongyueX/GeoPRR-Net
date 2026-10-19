"""Cache frozen, source-training held-out expert predictions for OOF routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.geoprr_v2_data import GeoPRRV2Dataset, read_rows
from experiments.geoprr_v2_router import FEATURE_PROTOCOL, build_router_features
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, PROTOCOL, device_batch, make_loader, source_split_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("oof_train", "validation"), required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    from experiments.geoprr_v2_model import GeoPRRV2

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("smoke_only", False):
        raise ValueError("cache requires a formal frozen source reader checkpoint")
    config = checkpoint["config"]
    rows = read_rows(args.manifest)
    reader_train = read_rows(checkpoint["manifests"]["train"])
    source_split_check(reader_train, rows)
    reader_validation = read_rows(checkpoint["manifests"]["validation"])
    identities = checkpoint.get("run_metadata", {}).get("sample_ids")
    if identities is None or identities != {"train": [row["sample_id"] for row in reader_train],
                                             "validation": [row["sample_id"] for row in reader_validation]}:
        raise ValueError("OOF caching requires the reader's saved source identities to match its manifests")
    if args.role == "oof_train":
        source_split_check(reader_validation, rows)
    device = torch.device(args.device)
    model = GeoPRRV2(**dict(config["model"], pretrained=False)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = GeoPRRV2Dataset(rows, dataset_root=args.data_root, **config["data"], require_geometry=False, augment=False)
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers, shuffle=False,
                         seed=config["training"]["seed"], device=device)
    collected: dict[str, list[np.ndarray]] = {key: [] for key in ("features", "expert_predictions", "field_valid", "target")}
    with torch.inference_mode():
        for raw in loader:
            batch = device_batch(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and not args.no_amp):
                output = model(batch["image"])
            values = {"features": build_router_features(output), "target": batch["y"],
                      "expert_predictions": torch.stack((output["field_reading"], output["appearance_reading"]), 1),
                      "field_valid": output["field_valid"]}
            for key, value in values.items():
                collected[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite expert outputs cannot be used to fit a router")
    metadata = {"feature_protocol": FEATURE_PROTOCOL, "role": args.role, "fold_id": args.fold_id,
                "dataset": "SyncG", "split": "train", "heldout_verified": True,
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint["epoch"],
                "manifest": str(args.manifest.resolve()), "reader_train_manifest": checkpoint["manifests"]["train"],
                "reader_config": {"model": config["model"], "data": config["data"]},
                "reader_training": {"training": config["training"], "loss_weights": config["loss_weights"]},
                "reader_seed": config["training"]["seed"],
                "reader_train_sample_ids": [str(row["sample_id"]) for row in reader_train],
                "reader_validation_sample_ids": [str(row["sample_id"]) for row in reader_validation],
                "reader_train_scene_ids": sorted({Path(str(row["metadata"]["scene_name"])).stem for row in reader_train}),
                "reader_validation_scene_ids": sorted({Path(str(row["metadata"]["scene_name"])).stem for row in reader_validation}),
                "heldout_sample_ids": [str(row["sample_id"]) for row in rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, sample_ids=np.asarray([str(row["sample_id"]) for row in rows]),
                        group_ids=np.asarray([str(row["group_id"]) for row in rows]),
                        scene_ids=np.asarray([Path(str(row["metadata"]["scene_name"])).stem for row in rows]),
                        metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps({"cache": str(args.output), "samples": len(rows), "role": args.role, "fold_id": args.fold_id}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
