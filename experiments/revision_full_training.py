"""Official-source populations and independent same-seed polar initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OFFICIAL_TRAIN_IMAGES = 16_000
EXPECTED_SCENES = 145
WARM_PROTOCOL = "geoprr_independent_warm_polar_v1"


def read_source_rows(path: Path, *, expected_samples: int) -> tuple[dict[str, Any], ...]:
    """Read an explicitly labelled SyncG training manifest, never the test split."""
    rows = tuple(
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if len(rows) != expected_samples:
        raise ValueError(f"source manifest has {len(rows)} rows; expected {expected_samples}")
    identifiers: set[str] = set()
    for row in rows:
        if row.get("dataset") != "SyncG" or row.get("split") != "train":
            raise ValueError("source manifests must contain only SyncG split=train rows")
        identifier = row.get("sample_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("source sample IDs must be nonempty and unique")
        identifiers.add(identifier)
    return rows


def _samples(rows: Sequence[Mapping[str, Any]], manifest: Path):
    from dataclasses import replace
    from experiments.resnet18_direct_progress import _direct_sample

    samples = tuple(_direct_sample(row, manifest_root=Path(manifest).resolve().parent) for row in rows)
    return tuple(replace(sample, group_id=sample.scene_stem) for sample in samples)


def validate_official_scenes(scene_stems: Iterable[str]) -> None:
    count = len(set(scene_stems))
    if count != EXPECTED_SCENES:
        raise ValueError(f"official source pool has {count} scenes; expected {EXPECTED_SCENES}")


def load_official_train_population(manifest: Path):
    rows = read_source_rows(manifest, expected_samples=OFFICIAL_TRAIN_IMAGES)
    samples = _samples(rows, manifest)
    validate_official_scenes(sample.scene_stem for sample in samples)
    return samples


def select_warm_rows(
    official_rows: Sequence[Mapping[str, Any]], warm_rows: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    """Use the warm manifest only to select IDs from the official training rows."""
    selected = {row["sample_id"] for row in warm_rows}
    official_ids = {row["sample_id"] for row in official_rows}
    if not selected <= official_ids:
        raise ValueError("warm sample IDs must be a subset of the official training pool")
    return tuple(row for row in official_rows if row["sample_id"] in selected)


def official_training_scope(samples, manifest: Path, availability_policy: str) -> dict[str, Any]:
    if len(samples) != OFFICIAL_TRAIN_IMAGES:
        raise ValueError("official_full requires all 16,000 source-training images")
    if availability_policy not in {"image_only", "clean_disabled"}:
        raise ValueError("official_full requires an explicit availability policy")
    validate_official_scenes(sample.scene_stem for sample in samples)
    return {
        "population_mode": "official_full",
        "source_manifest": str(Path(manifest).resolve()),
        "source_split": "train",
        "source_unique_images": len(samples),
        "source_scenes": len({sample.scene_stem for sample in samples}),
        "polar_train_samples": len(samples),
        "router_train_samples": len(samples),
        "router_train_rows": len(samples) * 6,
        "polar_router_image_overlap": len(samples),
        "inner_scene_disjoint": False,
        "official_syncg_test_access": False,
        "industrial_image_access": False,
        "availability_policy": availability_policy,
    }


def validate_seed_chain(warm: Mapping[str, Any], bank: Mapping[str, Any], seed: int) -> None:
    if warm.get("protocol") != WARM_PROTOCOL or int(warm.get("seed", -1)) != seed:
        raise ValueError("official_full requires an independently fitted warm initializer for this seed")
    foundation = warm.get("foundation", {})
    bank_foundation = bank.get("source_metadata", {}).get("source_foundation", {})
    if int(foundation.get("seed", -1)) != seed or int(bank.get("seed", -1)) != seed:
        raise ValueError("warm initializer, foundation and candidate bank must use the same seed")
    if int(bank_foundation.get("source_seed", -1)) != seed:
        raise ValueError("candidate bank foundation seed is missing or differs")
    if int(foundation.get("train_samples", -1)) != OFFICIAL_TRAIN_IMAGES:
        raise ValueError("warm foundation must record 16,000 training images")


def _verify_bank_foundation(bank, foundation_state: Mapping[str, Any]) -> None:
    import torch

    stages = (
        *tuple(bank.base_model.shared_scale8_encoder.children()),
        *tuple(bank.base_model.shared_scale16_encoder.children()),
        *tuple(bank.base_model.shared_context_encoder.children()),
    )
    expected_names: set[str] = set()
    for index, stage in enumerate(stages):
        for name, value in stage.state_dict().items():
            key = f"backbone.features.{index}.{name}"
            expected_names.add(key)
            if key not in foundation_state or not torch.equal(value.detach().cpu(), foundation_state[key]):
                raise ValueError("warm initializer and candidate bank have different foundation tensors")
    feature_names = {name for name in foundation_state if name.startswith("backbone.features.")}
    if expected_names != feature_names:
        raise ValueError("warm foundation feature stages differ from the candidate bank")
    for name, value in bank.base_model.moment_readout.point_projection.state_dict().items():
        key = f"backbone.classifier.1.{name}"
        if key not in foundation_state or not torch.equal(value.detach().cpu(), foundation_state[key]):
            raise ValueError("warm initializer and candidate bank have different scalar foundations")


def load_independent_warm(path: Path, *, bank, bank_metadata: Mapping[str, Any], seed: int):
    import torch
    from experiments.raw_angular_moment_refiner_probe import AngularMomentLogitRefiner

    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validate_seed_chain(payload, bank_metadata, seed)
    foundation_path = Path(payload["foundation"]["path"])
    foundation = torch.load(foundation_path, map_location="cpu", weights_only=False)
    if int(foundation.get("seed", -1)) != seed:
        raise ValueError("warm foundation checkpoint seed differs")
    _verify_bank_foundation(bank, foundation["model_state"])
    polar = AngularMomentLogitRefiner()
    polar.load_state_dict(payload["angular_no_aux_state"], strict=True)
    return polar, {
        "path": str(source),
        "protocol": WARM_PROTOCOL,
        "seed": seed,
        "epochs": int(payload["epochs"]),
        "role": "independent_same_seed_foundation",
        "foundation": payload["foundation"],
        "training_scope": payload["training_scope"],
        "foundation_tensors_match_candidate_bank": True,
    }


def train_warm_polar(
    *, foundation_checkpoint: Path, official_train_manifest: Path, warm_manifest: Path,
    expected_warm_images: int, output: Path, seed: int, epochs: int, batch_size: int,
    workers: int, learning_rate: float, weight_decay: float, device_name: str,
) -> dict[str, Any]:
    import time
    from dataclasses import asdict
    import torch
    import torch.nn.functional as F
    from experiments.raw_angular_moment_refiner_probe import AngularMomentLogitRefiner, AngularProgressDataset
    from experiments.raw_multiscale_progress_probe import build_probe_from_foundation
    from experiments.resnet18_direct_progress import _configure_reproducibility, _loader, _scene_stem
    from experiments.syncg_lightweight_regression_baselines import LightweightProgressRegressor

    if epochs < 1 or batch_size < 1 or workers < 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid warm-polar training settings")
    if not 0 < expected_warm_images <= OFFICIAL_TRAIN_IMAGES:
        raise ValueError("warm population must be within the official training pool")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"warm checkpoint already exists: {destination}")
    official_rows = read_source_rows(official_train_manifest, expected_samples=OFFICIAL_TRAIN_IMAGES)
    validate_official_scenes(_scene_stem(row["metadata"], row["sample_id"]) for row in official_rows)
    warm_rows = read_source_rows(warm_manifest, expected_samples=expected_warm_images)
    selected_rows = select_warm_rows(official_rows, warm_rows)
    samples = _samples(selected_rows, official_train_manifest)
    source = Path(foundation_checkpoint).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "efficientnet_b0" or int(payload.get("seed", -1)) != seed:
        raise ValueError("warm foundation must be the corresponding seed's EfficientNet-B0")
    if int(payload.get("train_samples", -1)) != OFFICIAL_TRAIN_IMAGES or int(payload.get("epochs", 0)) < 1:
        raise ValueError("warm foundation must record a completed official-full source fit")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    _configure_reproducibility(seed, device)
    foundation = LightweightProgressRegressor("efficientnet_b0", imagenet_pretrained=False)
    foundation.load_state_dict(payload["model_state"], strict=True)
    split = build_probe_from_foundation(foundation)
    encoder, base_projection = split.encoder.to(device).eval(), split.base_projection.to(device).eval()
    del split, foundation
    polar = AngularMomentLogitRefiner().to(device)
    dataset = AngularProgressDataset(samples, training=True, seed=seed)
    optimizer = torch.optim.AdamW(polar.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        dataset.set_epoch(int(payload["epochs"]) + epoch)
        loader = _loader(dataset, batch_size=batch_size, shuffle=True, workers=workers,
                         seed=seed + 100_000 + epoch, cuda=device.type == "cuda")
        polar.train()
        error_sum, count = 0.0, 0
        for batch in loader:
            images, target = batch["image"].to(device), batch["progress"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    features = encoder(images)
                    logit = base_projection(features["representation"]).squeeze(1)
                result = polar(features["stride8"], features["stride16"], features["representation"])
                prediction = torch.sigmoid(logit + result["logit_delta"])
                loss = F.l1_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            error_sum += float(torch.abs(prediction.detach() - target).sum().cpu())
            count += int(target.numel())
        row = {"epoch": epoch + 1, "samples": count, "mean_absolute_error": error_sum / count,
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    scope = {
        "official_train_manifest": str(Path(official_train_manifest).resolve()),
        "warm_manifest": str(Path(warm_manifest).resolve()),
        "warm_manifest_role": "sample_id_selection_from_official_training_rows",
        "official_source_samples": len(official_rows), "warm_samples": len(samples),
        "warm_scenes": len({sample.scene_stem for sample in samples}),
        "warm_sample_ids": [sample.sample_id for sample in samples],
        "official_syncg_test_access": False, "industrial_image_access": False,
    }
    checkpoint = {
        "schema_version": 1, "protocol": WARM_PROTOCOL, "seed": seed, "epochs": epochs,
        "checkpoint_selection": "terminal_fixed_epoch", "progress_loss": "exact_l1_angular_no_aux",
        "foundation": {"path": str(source), "seed": int(payload["seed"]),
                       "train_samples": int(payload["train_samples"]), "epochs": int(payload["epochs"]),
                       "protocol": payload.get("protocol"), "checkpoint_selection": payload.get("checkpoint_selection")},
        "foundation_frozen": True, "training_scope": scope,
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": weight_decay,
                      "scheduler": "CosineAnnealingLR"},
        "augmentation": asdict(dataset.augmentation), "batch_size": batch_size,
        "training_elapsed_seconds": time.perf_counter() - started, "history": history,
        "angular_no_aux_state": {name: value.detach().cpu() for name, value in polar.state_dict().items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return {"checkpoint": str(destination), "seed": seed, "warm_samples": len(samples), "epochs": epochs}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    warm = subparsers.add_parser("warm-polar", help="fit one seed's independent angular-no-aux initializer")
    warm.add_argument("--foundation", type=Path, required=True)
    warm.add_argument("--official-train-manifest", type=Path, required=True)
    warm.add_argument("--warm-manifest", type=Path, required=True)
    warm.add_argument("--expected-warm-images", type=int, required=True)
    warm.add_argument("--output", type=Path, required=True)
    warm.add_argument("--seed", type=int, required=True)
    warm.add_argument("--epochs", type=int, required=True)
    warm.add_argument("--learning-rate", type=float, required=True)
    warm.add_argument("--weight-decay", type=float, required=True)
    warm.add_argument("--batch-size", type=int, default=64)
    warm.add_argument("--workers", type=int, default=4)
    warm.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_warm_polar(
        foundation_checkpoint=args.foundation, official_train_manifest=args.official_train_manifest,
        warm_manifest=args.warm_manifest, expected_warm_images=args.expected_warm_images,
        output=args.output, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        workers=args.workers, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        device_name=args.device,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
