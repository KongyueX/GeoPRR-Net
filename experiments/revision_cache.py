"""Export matched image-derived features and predictions for revision experiments."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import cv2
import torch

from experiments.resnet18_direct_progress import DirectSample, _loader, load_syncg_samples
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.train_pcrt_meter import PCRTConditionedMultiViewDataset
from experiments.unified_pointer_reader import load_unified_pointer_reader_checkpoint

COHORT_COUNTS = {"syncg_train": (16000, 145), "syncg": (4000, 145),
                 "rf100": (151, 35), "industrial": (1395, 52)}
AVAILABILITY_POLICIES = ("image_only", "clean_disabled")


def load_revision_samples(manifest: Path, cohort: str) -> tuple[DirectSample, ...]:
    """Read an official SyncG manifest or labeled, already localized field ROIs."""
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if cohort in ("syncg_train", "syncg"):
        split = "train" if cohort == "syncg_train" else "test"
        if any(row.get("dataset") != "SyncG" or row.get("split") != split for row in rows):
            raise ValueError(f"{cohort} requires the official SyncG {split} manifest")
        samples = tuple(replace(sample, group_id=sample.scene_stem)
                        for sample in load_syncg_samples(manifest))
    else:
        samples = []
        for row in rows:
            sample_id, group_id = str(row["sample_id"]), str(row["group_id"])
            target = float(row["normalized_target"])
            if not sample_id or not group_id or not math.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("field rows require IDs, acquisition/source groups and targets in [0,1]")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            bbox = row.get("dial_bbox")
            if bbox is None:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:
                    raise ValueError(f"cannot read ROI: {image_path}")
                bbox = (0, 0, image.shape[1], image.shape[0])
            bbox = tuple(map(float, bbox))
            if len(bbox) != 4 or not all(map(math.isfinite, bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"invalid dial_bbox: {sample_id}")
            samples.append(DirectSample(sample_id, group_id, group_id, image_path.resolve(), bbox, target))
        samples = tuple(samples)
    expected_images, expected_groups = COHORT_COUNTS[cohort]
    if len(samples) != expected_images or len({s.sample_id for s in samples}) != expected_images:
        raise ValueError(f"{cohort} must contain {expected_images} unique images")
    if len({s.group_id for s in samples}) != expected_groups:
        raise ValueError(f"{cohort} must contain {expected_groups} groups")
    missing = [str(s.image_path) for s in samples if not s.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images; first: {missing[0]}")
    return samples


def effective_availability(active: torch.Tensor, condition: str, policy: str) -> torch.Tensor:
    if policy not in AVAILABILITY_POLICIES:
        raise ValueError(f"unknown availability policy: {policy}")
    return active.bool() if policy == "image_only" else active.bool() & (condition != "clean")


def collect_revision_cache(model, samples: Sequence[DirectSample], *, cache_path: Path,
                           device: torch.device, seed: int, batch_size: int, workers: int,
                           scope: str, availability_policy: str):
    """Keep a complete condition-major roster and record the actual input policy."""
    if scope not in ("source_train", "evaluation"):
        raise ValueError("scope must be source_train or evaluation")
    if availability_policy not in AVAILABILITY_POLICIES:
        raise ValueError("availability_policy must be chosen from the actual run configuration")
    destination = Path(cache_path)
    if destination.exists():
        raise FileExistsError(destination)
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("samples must be a nonempty unique roster")
    model.eval()
    fields, sample_ids, group_ids = {}, [], []
    group_map = {s.sample_id: s.group_id for s in samples}
    started = time.perf_counter()
    with torch.inference_mode():
        for ci, condition in enumerate(CONDITIONS):
            dataset = PCRTConditionedMultiViewDataset(samples, condition=condition)
            loader = _loader(dataset, batch_size=batch_size, shuffle=False, workers=workers,
                             seed=seed+20000+ci, cuda=device.type == "cuda")
            for batch in loader:
                result, features = model.forward_experts(
                    batch["original_view"].to(device), batch["sarn_view"].to(device),
                    batch["sarn_support_mask"].to(device),
                    sarn_active=effective_availability(batch["sarn_active"].to(device), condition, availability_policy),
                    raw_to_sarn_homography=batch["raw_to_sarn_homography"].to(device))
                candidates = result["candidate_predictions"].float()
                weights = result["routing_weights"].float()
                target_mean = (candidates*weights).sum(1).clamp(torch.finfo(torch.float32).eps, 1-torch.finfo(torch.float32).eps)
                values = {
                    "raw_representation": features["raw_representation"].float(),
                    "sarn_representation": features["sarn_representation"].float(),
                    "geometry_features": features["geometry_features"].float(),
                    "base_mean": candidates[:, 0], "polar_mean": candidates[:, 1], "r2mt_mean": candidates[:, 2],
                    "raw_mean": result["raw_anchor_mean"].float(),
                    "sarn_mean": result["sarn_endpoint_mean"].float(),
                    "relation_available": result["relation_available"].float(),
                    "polar_posterior": result["polar_direction_posterior"].float(),
                    "polar_concentration": result["polar_concentration"].float(),
                    "polar_entropy": result["polar_entropy"].float(),
                    "target": batch["target"].float(),
                    "condition_index": torch.full((len(candidates),), ci, dtype=torch.long),
                    "full_prediction": result["mean"].float(), "routing_weights": weights,
                    "moment_target": target_mean, "moment_residual": result["mean"].float()-target_mean,
                }
                for key, value in values.items():
                    fields.setdefault(key, []).append(value.detach().cpu())
                ids = list(batch["sample_id"])
                sample_ids.extend(ids)
                group_ids.extend(group_map[x] for x in ids)
            print(json.dumps({"phase": "revision_cache", "condition": condition, "images": len(samples)}), flush=True)
    cache = {name: torch.cat(parts) for name, parts in fields.items()}
    expected = len(samples)*len(CONDITIONS)
    if len(sample_ids) != expected or any(v.shape[0] != expected for v in cache.values()):
        raise ValueError("cache does not retain the complete six-condition roster")
    elapsed = time.perf_counter()-started
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "scope": scope, "seed": seed, "samples": len(samples),
                "rows": expected, "dataset": "syncg_train" if scope == "source_train" else None,
                "conditions": list(CONDITIONS), "sample_ids": sample_ids,
                "group_ids": group_ids, "availability_policy": availability_policy,
                "elapsed_seconds": elapsed, "cache": cache}, destination)
    return cache, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cohort", required=True, choices=COHORT_COUNTS)
    parser.add_argument("--availability-policy", required=True, choices=AVAILABILITY_POLICIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    samples = load_revision_samples(args.manifest, args.cohort)
    model, metadata = load_unified_pointer_reader_checkpoint(args.checkpoint, device=args.device)
    if metadata["seed"] != args.seed or metadata["source_seed"] != args.seed:
        raise ValueError("checkpoint and foundation seed must match the requested seed")
    recorded_policy = metadata.get("training_scope", {}).get("availability_policy")
    if recorded_policy is not None and recorded_policy != args.availability_policy:
        raise ValueError("evaluation availability policy differs from the training checkpoint")
    collect_revision_cache(model, samples, cache_path=args.output, device=torch.device(args.device),
                           seed=args.seed, batch_size=args.batch_size, workers=args.workers,
                           scope="source_train" if args.cohort == "syncg_train" else "evaluation",
                           availability_policy=args.availability_policy)
    payload = torch.load(args.output, map_location="cpu", weights_only=False)
    payload.update(checkpoint=str(args.checkpoint.resolve()), manifest=str(args.manifest.resolve()), cohort=args.cohort, dataset=args.cohort,
                   candidate_lineage=metadata.get("training_scope", {}).get("candidate_lineage"),
                   candidate_variant=metadata["variant"], model_metadata=metadata)
    torch.save(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
