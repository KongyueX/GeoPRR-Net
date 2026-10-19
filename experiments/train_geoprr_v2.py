"""Train lightweight GeoPRR V2 from disjoint source manifests."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.geoprr_v2_model import distribution_cdf_loss, two_hot_targets


PROTOCOL = "geoprr_v2_source_only_v1"
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/geoprrnet/datasets")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/geoprrnet/runs/geoprr_v2")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def two_hot(target: torch.Tensor, bins: int, *, dim: int = -1) -> torch.Tensor:
    """An ordered soft label whose mean equals the continuous target."""
    if bins < 2 or not torch.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("two-hot targets must be finite in [0,1] with at least two bins")
    return two_hot_targets(target, bins).movedim(-1, dim)


def distribution_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return distribution_cdf_loss(probability, target)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def structure_losses(output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    field = output["field_probs"].float()
    target = batch["field_target"].float()
    mask = batch["field_mask"][:, 0].float()
    label = two_hot(target, field.shape[1], dim=1)
    field_ce = masked_mean(-(label * output["field_logits"].float().log_softmax(1)).sum(1), mask)
    centers = torch.linspace(0, 1, field.shape[1], device=field.device).view(1, -1, 1, 1)
    field_l1 = masked_mean(((field * centers).sum(1) - target).abs(), mask)
    point_log_probability = output["point_logits"].float().flatten(1).log_softmax(1)
    point_ce = -(batch["point_target"].float().flatten(1) * point_log_probability).sum(1).mean()
    support_bce = F.binary_cross_entropy_with_logits(output["support_logits"].float(), batch["support_target"].float())
    return {"field_ce": field_ce, "field_l1": field_l1, "point_ce": point_ce, "support_bce": support_bce}


def warp_to_partner(source: torch.Tensor, source_to_canonical: torch.Tensor,
                    partner_to_canonical: torch.Tensor, *, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull source values onto the partner grid through their common dial plane."""
    batch, _channels, height, width = source.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device, dtype=torch.float32),
                            torch.arange(width, device=source.device, dtype=torch.float32), indexing="ij")
    x_roi = (xx + 0.5) * image_size / width - 0.5
    y_roi = (yy + 0.5) * image_size / height - 0.5
    pixels = torch.stack((x_roi, y_roi, torch.ones_like(xx)), dim=0).reshape(1, 3, -1)
    transform = torch.linalg.solve(source_to_canonical.float(), partner_to_canonical.float())
    homogeneous = transform @ pixels.expand(batch, -1, -1)
    z = homogeneous[:, 2]
    finite = z.abs() > 1e-8
    safe_z = torch.where(finite, z, torch.ones_like(z))
    x = homogeneous[:, 0] / safe_z
    y = homogeneous[:, 1] / safe_z
    gx = 2 * (x + 0.5) / image_size - 1
    gy = 2 * (y + 0.5) / image_size - 1
    valid = finite & (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    grid = torch.stack((gx, gy), dim=-1).reshape(batch, height, width, 2)
    grid = torch.where(valid.reshape(batch, height, width, 1), grid, torch.full_like(grid, 2))
    warped = F.grid_sample(source.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    jacobian = (torch.linalg.det(transform).abs().unsqueeze(1) / safe_z.abs().pow(3)).reshape(batch, 1, height, width)
    return warped, valid.reshape(batch, 1, height, width), jacobian


def controlled_pair_losses(output: Mapping[str, torch.Tensor], partner: Mapping[str, torch.Tensor],
                           batch: Mapping[str, Any], *, image_size: int, support_threshold: float) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Use only audited same-layout pairs; predicted support cannot discard a pair."""
    eligible = batch["pair_valid"].bool().reshape(-1)
    if not eligible.any():
        return {}, {"eligible": 0, "field_valid": 0}
    source_map, partner_map = batch["roi_to_canonical"], batch["pair_roi_to_canonical"]
    field_a = output["field_probs"].float()
    field_b = partner["field_probs"].float()
    mask_a, mask_b = output["support_probs"].float(), partner["support_probs"].float()
    point_b = partner["point_probs"].float()
    warped_mask, in_bounds, jacobian = warp_to_partner(mask_a, source_map, partner_map, image_size=image_size)
    warped_masked_field, _, _ = warp_to_partner(field_a * mask_a, source_map, partner_map, image_size=image_size)
    gt_mask_a, _, _ = warp_to_partner(batch["field_mask"], source_map, partner_map, image_size=image_size)
    overlap = gt_mask_a * batch["pair_field_mask"].float() * in_bounds
    weight_b = point_b * mask_b * overlap
    numerator = (warped_masked_field * weight_b).flatten(2).sum(2)
    mass = (warped_mask * weight_b).flatten(1).sum(1)
    valid = eligible & (mass >= support_threshold)
    losses = {"pair_support_mass": F.relu(support_threshold - mass[eligible]).mean()}
    if valid.any():
        probability = numerator[valid] / mass[valid, None]
        centers = torch.linspace(0, 1, field_a.shape[1], device=field_a.device)
        target = batch["pair_y"].float().reshape(-1)[valid]
        losses["pair_reading_l1"] = ((probability * centers).sum(1) - target).abs().mean()
        losses["pair_reading_cdf"] = distribution_loss(probability, target)
    centers = torch.linspace(0, 1, field_a.shape[1], device=field_a.device).view(1, -1, 1, 1)
    mean_a, _, _ = warp_to_partner((field_a * centers).sum(1, keepdim=True) * batch["field_mask"],
                                    source_map, partner_map, image_size=image_size)
    safe_overlap = (gt_mask_a > 0.999) * batch["pair_field_mask"].float() * in_bounds * eligible[:, None, None, None]
    losses["pair_field_l1"] = masked_mean((mean_a - (field_b * centers).sum(1, keepdim=True)).abs(), safe_overlap)
    losses["pair_support_l1"] = masked_mean((warped_mask - mask_b).abs(), in_bounds * eligible[:, None, None, None])
    same_state = eligible & batch["pair_same_state"].bool().reshape(-1)
    if same_state.any():
        point_a, _, _ = warp_to_partner(output["point_probs"], source_map, partner_map, image_size=image_size)
        point_a = point_a * jacobian * overlap
        q_b = point_b * overlap
        mass_a, mass_b = point_a.flatten(1).sum(1), q_b.flatten(1).sum(1)
        usable = same_state & (mass_a > 0) & (mass_b > 0)
        if usable.any():
            pa = point_a[usable] / mass_a[usable, None, None, None]
            pb = q_b[usable] / mass_b[usable, None, None, None]
            losses["pair_point_l1"] = (pa - pb).abs().flatten(1).sum(1).mean()
    return losses, {"eligible": int(eligible.sum()), "field_valid": int(valid.sum())}


def reading_losses(output: Mapping[str, torch.Tensor], target: torch.Tensor, *, structural_warmup: bool,
                   appearance_auxiliary: bool = False) -> dict[str, torch.Tensor]:
    result = {}
    if appearance_auxiliary:
        result["appearance_l1"] = (output["appearance_reading"].float().reshape(-1) - target).abs().mean()
        if output.get("appearance_probs") is not None:
            result["appearance_cdf"] = distribution_loss(output["appearance_probs"], target)
    if not structural_warmup:
        result["reading_l1"] = (output["reading"].float().reshape(-1) - target).abs().mean()
        if output.get("distribution") is not None:
            result["reading_cdf"] = distribution_loss(output["distribution"], target)
    return result


def select_batch(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    positions = indices.detach().cpu().tolist()
    return {key: value[indices] if isinstance(value, torch.Tensor) else [value[i] for i in positions]
            for key, value in batch.items() if isinstance(value, (torch.Tensor, list, tuple))}


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score all declared images, assigning normalized error one to failures."""
    if not rows:
        raise ValueError("cannot score an empty evaluation population")
    error = np.asarray([float(row["absolute_error"]) for row in rows], dtype=np.float64)
    valid = np.asarray([bool(row["valid"]) for row in rows])
    field_valid = np.asarray([bool(row.get("field_valid", False)) for row in rows])
    tail_count = max(1, math.ceil(0.1 * len(rows)))
    return {
        "samples": len(rows),
        "groups": len({str(row["group_id"]) for row in rows}),
        "nmae": float(error.mean()),
        "nmae_percent_fs": float(100 * error.mean()),
        "acc_at_1": float(np.mean(valid & (error <= 0.01))),
        "acc_at_2": float(np.mean(valid & (error <= 0.02))),
        "acc_at_5": float(np.mean(valid & (error <= 0.05))),
        "coverage": float(valid.mean()),
        "field_coverage": float(field_valid.mean()),
        "failed": int((~valid).sum()),
        "median_error": float(np.median(error)),
        "p90_error": float(np.quantile(error, 0.9)),
        "p95_error": float(np.quantile(error, 0.95)),
        "tail10_nmae": float(np.sort(error)[-tail_count:].mean()),
        "maximum_error": float(error.max()),
    }


def prediction_rows(output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    prediction = output["reading"].detach().float().reshape(-1).cpu()
    target = batch["y"].detach().float().reshape(-1).cpu()
    field_valid = output.get("field_valid", torch.zeros_like(prediction, dtype=torch.bool)).detach().cpu().reshape(-1)
    support = output.get("support_mass", torch.zeros_like(prediction)).detach().float().cpu().reshape(-1)
    rows = []
    for index, (pred, truth) in enumerate(zip(prediction.tolist(), target.tolist(), strict=True)):
        valid = math.isfinite(pred) and 0 <= pred <= 1
        rows.append({
            "sample_id": str(batch["sample_id"][index]),
            "group_id": str(batch["group_id"][index]),
            "target": truth,
            "prediction": pred if valid else None,
            "valid": valid,
            "absolute_error": abs(pred - truth) if valid else 1.0,
            "field_valid": bool(field_valid[index]) and valid,
            "support_mass": float(support[index]) if torch.isfinite(support[index]) else None,
            "failure": "" if valid else "nonfinite_or_out_of_range_prediction",
        })
    return rows


def source_split_check(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> None:
    for population in (train_rows, val_rows):
        for row in population:
            if str(row.get("dataset", "")).lower() != "syncg" or str(row.get("split", "")).lower() != "train":
                raise ValueError("training and model selection require SyncG source-training rows")
            if not row.get("metadata", {}).get("scene_name"):
                raise ValueError("source splitting requires metadata.scene_name")
    train_ids = {str(row["sample_id"]) for row in train_rows}
    val_ids = {str(row["sample_id"]) for row in val_rows}
    train_groups = {str(row["group_id"]) for row in train_rows}
    val_groups = {str(row["group_id"]) for row in val_rows}
    if len(train_ids) != len(train_rows) or len(val_ids) != len(val_rows):
        raise ValueError("duplicate sample IDs in a source manifest")
    if train_ids & val_ids or train_groups & val_groups:
        raise ValueError("source train and validation must have disjoint sample IDs and groups")
    train_scenes = {Path(str(row["metadata"]["scene_name"])).stem for row in train_rows}
    val_scenes = {Path(str(row["metadata"]["scene_name"])).stem for row in val_rows}
    if train_scenes & val_scenes:
        raise ValueError("source train and validation share scene backgrounds across gauge types")


def make_loader(dataset: Any, *, batch_size: int, workers: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)


def device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/geoprr_v2_femto.json")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=("raw_scalar", "raw_distribution", "multitask", "field"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from experiments.geoprr_v2_data import GeoPRRV2Dataset, GeometryConfig
    from experiments.geoprr_v2_model import GeoPRRV2

    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for field in ("epochs", "batch_size", "workers", "seed"):
        if getattr(args, field) is not None:
            config["training"][field] = getattr(args, field)
    if args.variant is not None:
        config["model"]["variant"] = args.variant
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    if args.no_amp:
        config["training"]["amp"] = False
    training = config["training"]
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["workers"] < 0:
        raise ValueError("epochs/batch-size must be positive and workers nonnegative")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("smoke batch limits must be nonnegative")
    seed = int(training["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a smoke check")
    output_dir = args.output or DEFAULT_OUTPUT_ROOT / config["name"] / config["model"]["variant"] / f"seed_{seed}"
    if (output_dir / "latest.pt").exists() and args.resume is None:
        raise FileExistsError(f"{output_dir}/latest.pt exists; use --resume or a new output directory")
    geometry = config["model"]["variant"] in {"field", "multitask"}
    dataset_options = dict(dataset_root=args.data_root, **config["data"], require_geometry=geometry,
                           geometry_config=GeometryConfig(min_support_mass=config["model"]["support_threshold"]))
    train_data = GeoPRRV2Dataset(args.train_manifest, augment=True, pair_manifest=args.pair_manifest, **dataset_options)
    val_data = GeoPRRV2Dataset(args.val_manifest, augment=False, require_geometry=False,
                              dataset_root=args.data_root, **config["data"])
    source_split_check(train_data.rows, val_data.rows)
    model_options = dict(config["model"])
    if args.resume is not None:
        model_options["pretrained"] = False
    model = GeoPRRV2(**model_options).to(device)
    encoder_ids = {id(parameter) for parameter in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": training["backbone_lr"]},
        {"params": [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids], "lr": training["head_lr"]},
    ], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=training["min_lr"])
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_nmae, global_step = 0, math.inf, 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != config:
            raise ValueError("resume protocol/config differs from checkpoint")
        expected = {"train": str(args.train_manifest.resolve()), "validation": str(args.val_manifest.resolve()),
                    "pairs": str(args.pair_manifest.resolve()) if args.pair_manifest is not None else None}
        if checkpoint["manifests"] != expected:
            raise ValueError("resume manifests differ from checkpoint")
        if checkpoint["batch_limits"] != {"train": args.max_train_batches, "validation": args.max_val_batches}:
            raise ValueError("resume smoke batch limits differ from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_nmae = float(checkpoint["best_nmae"])
        global_step = int(checkpoint["global_step"])
        restore_rng(checkpoint["rng"])
    diagnostic_data = any(row.get("diagnostic_only", False) for row in (*train_data.rows, *val_data.rows))
    smoke = bool(args.max_train_batches or args.max_val_batches or diagnostic_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    run_metadata = {
        "protocol": PROTOCOL, "config": config, "smoke_only": smoke,
        "diagnostic_data": diagnostic_data,
        "batch_limits": {"train": args.max_train_batches, "validation": args.max_val_batches},
        "model_selection": "source_validation_nmae_full_denominator",
        "train_samples": len(train_data), "validation_samples": len(val_data),
        "train_groups": len({str(row["group_id"]) for row in train_data.rows}),
        "validation_groups": len({str(row["group_id"]) for row in val_data.rows}),
        "manifests": {"train": str(args.train_manifest.resolve()), "validation": str(args.val_manifest.resolve()),
                      "pairs": str(args.pair_manifest.resolve()) if args.pair_manifest is not None else None},
        "data_root": str(args.data_root.resolve()), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "controlled_pairs": args.pair_manifest is not None,
        "geometry_target_config": vars(dataset_options["geometry_config"]) if geometry else None,
    }
    write_json(output_dir / "run.json", run_metadata)
    for epoch in range(start_epoch, training["epochs"]):
        started = time.perf_counter()
        warmup = epoch < training["warmup_epochs"]
        model.set_backbone_trainable(not warmup)
        model.train()
        train_loader = make_loader(train_data, batch_size=training["batch_size"], workers=training["workers"],
                                   shuffle=True, seed=seed + epoch, device=device)
        loss_sums: dict[str, float] = {}
        seen, steps, pair_images, eligible_pairs, valid_pairs = 0, 0, 0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index, raw_batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = device_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                details = model(batch["image"])
                field_variant = config["model"]["variant"] == "field"
                losses = reading_losses(details, batch["y"].float().reshape(-1), structural_warmup=warmup and field_variant,
                                        appearance_auxiliary=field_variant)
                if geometry:
                    losses.update(structure_losses(details, batch))
                    losses["support_mass"] = details["support_deficit"].mean()
                if "pair_kind" in batch:
                    indices = torch.tensor([i for i, kind in enumerate(batch["pair_kind"]) if kind != "none"], device=device)
                    if indices.numel():
                        pair_batch = select_batch(batch, indices)
                        partner_batch = {key.removeprefix("pair_"): value for key, value in pair_batch.items() if key.startswith("pair_")}
                        partner_details = model(partner_batch["image"])
                        partner_losses = reading_losses(partner_details, partner_batch["y"].float().reshape(-1),
                                                         structural_warmup=warmup and field_variant, appearance_auxiliary=field_variant)
                        if geometry:
                            partner_losses.update(structure_losses(partner_details, partner_batch))
                            partner_losses["support_mass"] = partner_details["support_deficit"].mean()
                        a_count, b_count = len(batch["y"]), int(indices.numel())
                        losses = {name: (value * a_count + partner_losses[name] * b_count) / (a_count + b_count)
                                  for name, value in losses.items()}
                        pair_images += b_count
                        if field_variant and not warmup:
                            source_details = {key: value[indices] if isinstance(value, torch.Tensor) else value for key, value in details.items()}
                            with torch.autocast(device_type=device.type, enabled=False):
                                pair_losses, pair_counts = controlled_pair_losses(source_details, partner_details, pair_batch,
                                    image_size=config["data"]["image_size"], support_threshold=config["model"]["support_threshold"])
                            losses.update(pair_losses)
                            eligible_pairs += pair_counts["eligible"]
                            valid_pairs += pair_counts["field_valid"]
                loss = sum(config["loss_weights"].get(name, 1.0) * value for name, value in losses.items())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch + 1}, batch {batch_index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            global_step += int(not amp_enabled or scaler.get_scale() >= old_scale)
            if batch_index == 0:
                print(json.dumps({"event": "training_batch", "epoch": epoch + 1, "batch": 1,
                                  "stage": "head_warmup" if warmup else "joint_finetune",
                                  "loss": float(loss.detach()), "device": str(device),
                                  "global_step": global_step, "smoke_only": smoke}), flush=True)
            count = len(batch["y"])
            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach()) * count
            seen += count
            steps += 1
        if seen == 0:
            raise ValueError("no training samples were consumed")
        model.eval()
        val_rows = []
        val_loader = make_loader(val_data, batch_size=training["batch_size"], workers=training["workers"],
                                 shuffle=False, seed=seed, device=device)
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(val_loader):
                if args.max_val_batches and batch_index >= args.max_val_batches:
                    break
                batch = device_batch(raw_batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    details = model(batch["image"])
                val_rows.extend(prediction_rows(details, batch))
        summary = summarize_rows(val_rows)
        improved = summary["nmae"] < best_nmae
        if improved:
            best_nmae = summary["nmae"]
        scheduler.step()
        record = {"epoch": epoch + 1, "stage": "head_warmup" if warmup else "joint_finetune",
                  "global_step": global_step, "train_samples": seen, "train_batches": steps,
                  "additional_pair_images": pair_images, "eligible_pairs": eligible_pairs,
                  "valid_cross_readouts": valid_pairs, "cross_readout_coverage": valid_pairs / eligible_pairs if eligible_pairs else None,
                  "losses": {name: value / seen for name, value in loss_sums.items()},
                  "validation": summary, "best_nmae": best_nmae, "elapsed_seconds": time.perf_counter() - started,
                  "learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "smoke_only": smoke}
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = {"protocol": PROTOCOL, "config": config, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": epoch + 1, "global_step": global_step, "best_nmae": best_nmae,
                      "rng": rng_state(), "manifests": run_metadata["manifests"], "smoke_only": smoke,
                      "batch_limits": run_metadata["batch_limits"],
                      "run_metadata": run_metadata}
        torch.save(checkpoint, output_dir / "latest.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
            write_json(output_dir / "best_validation.json", {"epoch": epoch + 1, "smoke_only": smoke, **summary})
        print(json.dumps(record, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
