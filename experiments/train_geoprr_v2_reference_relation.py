"""Train the reference-conditioned relation reader on frozen source populations."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationDataset, ReferenceRelationReader, reference_relation_losses,
)
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    DEFAULT_DATA_ROOT, device_batch, make_loader, restore_rng,
    rng_state, save_checkpoint, seed_everything, source_split_check, summarize_rows,
)


PROTOCOL = "geoprr_v2_reference_relation_source_only_v1"



def validation_rows(output, batch):
    reading = output["reading"].detach().float().cpu().reshape(-1)
    probability = output["distribution"].detach().float().cpu()
    invalid = output["invalid_mass"].detach().float().cpu().reshape(-1)
    outside = output["outside_arc_mass"].detach().float().cpu().reshape(-1)
    target = batch["y"].detach().float().cpu().reshape(-1)
    extra = {key: output[key].detach().float().cpu().reshape(-1).tolist() for key in
             ("reference_entropy", "reference_max", "joint_vs_product_state_l1", "counterfactual_reading")}
    rows = []
    for i, (pred, truth) in enumerate(zip(reading.tolist(), target.tolist(), strict=True)):
        valid_probability = bool(torch.isfinite(probability[i]).all() and (probability[i] >= 0).all()
                                 and abs(float(probability[i].sum()) - 1) <= 1e-4)
        valid_mass = (math.isfinite(float(invalid[i])) and -1e-4 <= float(invalid[i]) <= 1 + 1e-4
                      and math.isfinite(float(outside[i])) and -1e-4 <= float(outside[i]) <= 1 + 1e-4)
        valid = not batch["input_failure"][i] and valid_probability and valid_mass and math.isfinite(pred) and 0 <= pred <= 1
        rows.append({"sample_id": batch["sample_id"][i], "group_id": batch["group_id"][i],
                     "target": truth, "prediction": pred if valid else None, "valid": bool(valid),
                     "input_valid": not batch["input_failure"][i],
                     "absolute_error": abs(pred - truth) if valid else 1.0,
                     "invalid_mass": float(invalid[i]) if math.isfinite(float(invalid[i])) else None,
                     "outside_arc_mass": float(outside[i]) if math.isfinite(float(outside[i])) else None,
                     "failure": batch["input_failure"][i] or ("" if valid else "invalid_prediction_or_probability"),
                     **{key: values[i] for key, values in extra.items()}})
    return rows


def validation_summary(rows):
    summary = summarize_rows(rows)
    summary.pop("field_coverage")
    summary["input_failures"] = sum(not row["input_valid"] for row in rows)
    summary["errors_above_1_percent_fs"] = sum(row["absolute_error"] > 0.01 for row in rows)
    for key in ("invalid_mass", "outside_arc_mass"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def save_predictions(path, rows):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_history(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train-manifest", "val-manifest", "pair-manifest", "init-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    process_started = time.perf_counter()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    protocol = config.get("protocol", PROTOCOL)
    training = config["training"]
    seed = int(training["seed"])
    if seed != 20262020:
        raise ValueError("this protocol authorizes only seed20262020")
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if args.resume is None and any((args.output / name).exists() for name in ("latest.pt", "history.jsonl", "completed.json")):
        raise FileExistsError("training artifacts already exist; use --resume")
    train_data = ReferenceRelationDataset(args.train_manifest, data_root=args.data_root,
                                     data_config=config["data"], augment=True, pair_manifest=args.pair_manifest)
    val_data = ReferenceRelationDataset(args.val_manifest, data_root=args.data_root,
                                   data_config=config["data"], augment=False)
    if (len(train_data.rows), len(train_data), len(train_data.edges), len(val_data)) != (18590, 22686, 8192, 1506):
        raise ValueError("source populations or exposure differ from the frozen protocol")
    source_split_check(train_data.rows, val_data.rows)
    paths = {key: str(getattr(args, key).resolve()) for key in
             ("train_manifest", "val_manifest", "pair_manifest", "init_checkpoint")}
    data_contract = {"paths": paths, "train": train_data.supervision_contract(),
                     "validation": val_data.supervision_contract(), "exposure_edges": train_data.edges}
    model = ReferenceRelationReader(**config["model"])
    initialization = None
    if args.resume is None:
        initialization = (model.initialize_full(args.init_checkpoint) if config.get("initialization") == "full_best57"
                          else model.initialize_shared(args.init_checkpoint))
    model = model.to(device)
    head_parameters = list(model.state_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    shared_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": shared_parameters, "lr": training["backbone_lr"]},
                                  {"params": head_parameters, "lr": training["head_lr"]}],
                                 weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"], eta_min=training["min_lr"])
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_nmae, best_epoch = 0, math.inf, None
    totals = {"train_batches": 0, "optimizer_updates": 0, "amp_skipped_updates": 0,
              "donor_images": 0, "recipient_images": 0, "validation_images": 0,
              "training_wall_seconds": 0.0, "validation_wall_seconds": 0.0}
    history = []
    checkpointed_process_seconds = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["protocol"] != protocol or checkpoint["config"] != config or checkpoint["data_contract"] != data_contract:
            raise ValueError("resume protocol/config/source supervision or exposure differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_nmae, best_epoch = checkpoint["epoch"], checkpoint["best_nmae"], checkpoint["best_epoch"]
        totals = checkpoint["totals"]
        history = checkpoint["history"]
        checkpointed_process_seconds = checkpoint["checkpointed_process_wall_seconds"]
        initialization = checkpoint["initialization"]
        restore_rng(checkpoint["rng"])
    args.output.mkdir(parents=True, exist_ok=True)
    run = {"protocol": protocol, "config": config, "data_contract": data_contract,
           "train_unique_images": len(train_data.rows), "train_items": len(train_data),
           "expected_full_epoch": {"donor_images": 22686, "recipient_images": 8192,
                                   "total_image_forwards": 30878, "optimizer_batches": math.ceil(22686 / training["batch_size"])},
           "validation_images": len(val_data), "model_selection": "full1506_native_nmae_earliest_tie",
           "initialization": initialization, "parameters": sum(p.numel() for p in model.parameters()),
           "shared_parameters": sum(p.numel() for p in shared_parameters),
           "head_parameters": sum(p.numel() for p in head_parameters),
           "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "inference_input": "RGB only; H/span/y and virtual role phases are training/scoring labels",
           "recipient_batching": "donors and recipients concatenated in one forward; old reader used separate forwards",
           "comparison": "new arms match parameters, labels, initialization, exposure and updates; only shared-reference joint versus same-marginals product differs; old supervised differs in architecture, supervision and warm start history"}
    write_json(args.output / "config.json", config)
    write_json(args.output / "run.json", run)
    write_history(args.output / "history.jsonl", history)
    for epoch in range(start_epoch, training["epochs"]):
        epoch_started = time.perf_counter()
        warmup = epoch < training["warmup_epochs"]
        model.set_shared_trainable(not warmup)
        model.train()
        train_loader = make_loader(train_data, batch_size=training["batch_size"], workers=training["workers"],
                                   shuffle=True, seed=seed + epoch, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums = Counter()
        donor_seen = recipient_seen = batches = updates = skips = 0
        for batch_index, raw_batch in enumerate(train_loader):
            batch = device_batch(raw_batch, device)
            paired = batch["pair_valid"]
            images = torch.cat([batch["image"], batch["pair_image"][paired]])
            truth = torch.cat([batch["y"], batch["pair_y"][paired]])
            supervision = {"y": truth, "image": images}
            for key in ("roi_to_canonical", "field_target", "field_mask", "span_radians"):
                supervision[key] = torch.cat([batch[key], batch[f"pair_{key}"][paired]])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(images)
            terms = reference_relation_losses(output, supervision)
            loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch{epoch + 1}, batch{batch_index + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            if not amp_enabled and not torch.isfinite(gradient_norm):
                raise FloatingPointError("nonfinite unscaled training gradient")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = amp_enabled and scaler.get_scale() < old_scale
            count, donors = len(truth), len(batch["y"])
            for key, value in terms.items():
                sums[key] += float(value.detach()) * count
            for key in ("invalid_mass", "outside_arc_mass", "reference_entropy", "reference_max", "joint_vs_product_state_l1"):
                sums[key] += float(output[key].detach().float().mean()) * count
            donor_seen += donors
            recipient_seen += count - donors
            batches += 1
            updates += int(not skipped)
            skips += int(skipped)
            if batch_index == 0:
                print(json.dumps({"event": "training_batch", "epoch": epoch + 1, "batch": 1,
                                  "stage": "head_warmup" if warmup else "joint_finetune", "loss": float(loss.detach()),
                                  "donor_images": donors, "recipient_images": count - donors}), flush=True)
        if (donor_seen, recipient_seen, batches) != (22686, 8192, math.ceil(22686 / training["batch_size"])):
            raise RuntimeError("actual epoch exposure/batches differ from frozen protocol")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        model.eval()
        rows = []
        val_loader = make_loader(val_data, batch_size=training["validation_batch_size"], workers=training["workers"],
                                 shuffle=False, seed=seed, device=device)
        with torch.inference_mode():
            for raw_batch in val_loader:
                batch = device_batch(raw_batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(batch["image"])
                rows.extend(validation_rows(output, batch))
        if len(rows) != 1506 or {row["sample_id"] for row in rows} != {row["sample_id"] for row in val_data.rows}:
            raise RuntimeError("validation population is incomplete")
        validation_seconds = time.perf_counter() - validation_started
        summary = validation_summary(rows)
        improved = summary["nmae"] < best_nmae
        if improved:
            best_nmae, best_epoch = summary["nmae"], epoch + 1
        scheduler.step()
        totals["train_batches"] += batches
        totals["optimizer_updates"] += updates
        totals["amp_skipped_updates"] += skips
        totals["donor_images"] += donor_seen
        totals["recipient_images"] += recipient_seen
        totals["validation_images"] += len(rows)
        totals["training_wall_seconds"] += training_seconds
        totals["validation_wall_seconds"] += validation_seconds
        record = {"epoch": epoch + 1, "stage": "head_warmup" if warmup else "joint_finetune",
                  "train_batches": batches, "donor_images": donor_seen, "recipient_images": recipient_seen,
                  "total_image_forwards": donor_seen + recipient_seen, "optimizer_updates": updates,
                  "amp_skipped_updates": skips, "global_step": totals["optimizer_updates"],
                  "losses": {key: value / (donor_seen + recipient_seen) for key, value in sums.items()},
                  "validation": summary, "best_nmae": best_nmae, "best_epoch": best_epoch,
                  "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                  "elapsed_seconds": time.perf_counter() - epoch_started,
                  "learning_rates": [group["lr"] for group in optimizer.param_groups],
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        history.append(record)
        checkpoint = {"protocol": protocol, "config": config, "data_contract": data_contract,
                      "initialization": initialization, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                      "epoch": epoch + 1, "best_nmae": best_nmae, "best_epoch": best_epoch,
                      "global_step": totals["optimizer_updates"], "totals": totals, "history": history,
                      "checkpointed_process_wall_seconds": checkpointed_process_seconds + time.perf_counter() - process_started,
                      "run_metadata": run}
        if improved:
            save_checkpoint(args.output / "best.pt", checkpoint)
            write_json(args.output / "best_validation.json", {"epoch": epoch + 1, **summary})
            save_predictions(args.output / "best_predictions.csv.gz", rows)
        save_checkpoint(args.output / "latest.pt", checkpoint)
        write_history(args.output / "history.jsonl", history)
        print(json.dumps(record, allow_nan=False), flush=True)
    write_json(args.output / "completed.json", {"protocol": protocol, "epochs": training["epochs"],
               "best_epoch": best_epoch, "best_nmae": best_nmae, "totals": totals,
               "current_process_wall_seconds": time.perf_counter() - process_started,
               "checkpointed_process_wall_seconds_before_resume": checkpointed_process_seconds,
               "uncheckpointed_work_after_interruptions": "not included in checkpointed counters; use controller process logs",
               "seed": seed, "seed_expansion": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
