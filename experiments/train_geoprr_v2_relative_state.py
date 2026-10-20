"""Train a relative physical-state reader on the frozen source populations."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.geoprr_v2_data import GeoPRRV2Dataset, normalized_target, read_rows
from experiments.geoprr_v2_relative_state import RelativeStateReader, relative_state_nll
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    DEFAULT_DATA_ROOT, device_batch, distribution_loss, make_loader, restore_rng,
    rng_state, save_checkpoint, seed_everything, source_split_check, summarize_rows,
)


PROTOCOL = "geoprr_v2_relative_state_source_only_v1"


class RelativeStateDataset(Dataset):
    def __init__(self, manifest, labels, *, data_root, data_config, augment, pair_manifest=None):
        self.base = GeoPRRV2Dataset(manifest, dataset_root=data_root, augment=augment,
                                   require_geometry=False, pair_manifest=None, **data_config)
        self.rows = self.base.rows
        self.labels = {row["sample_id"]: row for row in read_rows(labels)}
        if set(self.labels) != {row["sample_id"] for row in self.rows}:
            raise ValueError("physical labels and source manifest IDs differ")
        for row in self.rows:
            label = self.labels[row["sample_id"]]
            if label["group_id"] != row["group_id"]:
                raise ValueError(f"{row['sample_id']}: label/source group differs")
            delta, span = float(label["delta_radians"]), float(label["span_radians"])
            if not (math.isfinite(delta) and math.isfinite(span) and 0 <= delta < 2 * math.pi and 0 < span < 2 * math.pi):
                raise ValueError(f"{row['sample_id']}: invalid relative-state label")
        by_id = {row["sample_id"]: index for index, row in enumerate(self.rows)}
        partners = defaultdict(list)
        edges = set()
        if pair_manifest is not None:
            for line in Path(pair_manifest).read_text().splitlines():
                if not line.strip():
                    continue
                pair = json.loads(line)
                a, b = pair["sample_a"], pair["sample_b"]
                if a not in by_id or b not in by_id or a == b or (a, b) in edges:
                    raise ValueError("invalid/duplicate in-partition exposure edge")
                if self.labels[a]["source_role"] != "supplement" or self.labels[b]["source_role"] != "supplement":
                    raise ValueError("exposure edges must connect supplementary training images")
                edges.add((a, b))
                partners[a].append(by_id[b])
        self.items = [(index, partner) for index, row in enumerate(self.rows)
                      for partner in partners.get(row["sample_id"], [None])]
        self.edges = sorted(edges)
        self.augment = augment
        roles = Counter(self.labels[row["sample_id"]]["source_role"] for row in self.rows)
        if augment:
            outgoing = Counter(a for a, _ in edges)
            incoming = Counter(b for _, b in edges)
            supplement = {key for key, label in self.labels.items() if label["source_role"] == "supplement"}
            if roles != {"original": 14494, "supplement": 4096} or len(edges) != 8192 or len(self.items) != 22686:
                raise ValueError("training population/exposure differs from the frozen source protocol")
            if any(outgoing[key] != 2 or incoming[key] != 2 for key in supplement):
                raise ValueError("each supplementary image requires two donor and two recipient exposures")
        elif roles != {"validation": 1506} or len(self.items) != 1506:
            raise ValueError("validation population differs from the frozen original1506")

    def __len__(self):
        return len(self.items)

    def _sample(self, index, photometric):
        row = self.rows[index]
        try:
            sample = self.base._sample(index, photometric)
            sample["input_failure"] = ""
        except (OSError, ValueError) as exc:
            if self.augment:
                raise
            sample = {"image": torch.zeros(3, self.base.image_size, self.base.image_size),
                      "y": torch.tensor(normalized_target(row), dtype=torch.float32),
                      "sample_id": row["sample_id"], "group_id": row["group_id"],
                      "input_failure": f"{type(exc).__name__}: {exc}"}
        label = self.labels[row["sample_id"]]
        sample["delta_radians"] = torch.tensor(label["delta_radians"], dtype=torch.float32)
        sample["span_radians"] = torch.tensor(label["span_radians"], dtype=torch.float32)
        return {key: sample[key] for key in ("image", "y", "sample_id", "group_id", "input_failure",
                                           "delta_radians", "span_radians")}

    def __getitem__(self, index):
        donor, partner = self.items[index]
        photometric = self.base._photometric() if self.augment else None
        sample = self._sample(donor, photometric)
        if not self.augment:
            return sample
        recipient = self._sample(partner, photometric) if partner is not None else sample
        for key in ("image", "y", "delta_radians", "span_radians"):
            sample[f"pair_{key}"] = recipient[key]
        sample["pair_valid"] = partner is not None
        return sample

    def supervision_contract(self):
        return [[row["sample_id"], row["group_id"], normalized_target(row),
                 self.labels[row["sample_id"]]["delta_radians"],
                 self.labels[row["sample_id"]]["span_radians"],
                 self.labels[row["sample_id"]]["source_role"]] for row in self.rows]


def validation_rows(output, batch):
    reading = output["reading"].detach().float().cpu().reshape(-1)
    probability = output["distribution"].detach().float().cpu()
    invalid = output["invalid_mass"].detach().float().cpu().reshape(-1)
    outside = output["outside_arc_mass"].detach().float().cpu().reshape(-1)
    target = batch["y"].detach().float().cpu().reshape(-1)
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
                     "failure": batch["input_failure"][i] or ("" if valid else "invalid_prediction_or_probability")})
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
    for name in ("config", "train-manifest", "val-manifest", "train-labels", "val-labels",
                 "pair-manifest", "init-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    process_started = time.perf_counter()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
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
    train_data = RelativeStateDataset(args.train_manifest, args.train_labels, data_root=args.data_root,
                                     data_config=config["data"], augment=True, pair_manifest=args.pair_manifest)
    val_data = RelativeStateDataset(args.val_manifest, args.val_labels, data_root=args.data_root,
                                   data_config=config["data"], augment=False)
    source_split_check(train_data.rows, val_data.rows)
    paths = {key: str(getattr(args, key).resolve()) for key in
             ("train_manifest", "val_manifest", "train_labels", "val_labels", "pair_manifest", "init_checkpoint")}
    data_contract = {"paths": paths, "train": train_data.supervision_contract(),
                     "validation": val_data.supervision_contract(), "exposure_edges": train_data.edges}
    model = RelativeStateReader(**config["model"])
    initialization = model.initialize_shared(args.init_checkpoint) if args.resume is None else None
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
        if checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != config or checkpoint["data_contract"] != data_contract:
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
    run = {"protocol": PROTOCOL, "config": config, "data_contract": data_contract,
           "train_unique_images": len(train_data.rows), "train_items": len(train_data),
           "expected_full_epoch": {"donor_images": 22686, "recipient_images": 8192,
                                   "total_image_forwards": 30878, "optimizer_batches": math.ceil(22686 / training["batch_size"])},
           "validation_images": len(val_data), "model_selection": "full1506_native_nmae_earliest_tie",
           "initialization": initialization, "parameters": sum(p.numel() for p in model.parameters()),
           "shared_parameters": sum(p.numel() for p in shared_parameters),
           "head_parameters": sum(p.numel() for p in head_parameters),
           "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "inference_input": "RGB only; state geometry and y are supervision/scoring labels",
           "recipient_batching": "donors and recipients concatenated in one forward; old reader used separate forwards",
           "comparison": "new arms have matched exposures and updates; old supervised has different head/objectives and warm start history"}
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
            delta = torch.cat([batch["delta_radians"], batch["pair_delta_radians"][paired]])
            span = torch.cat([batch["span_radians"], batch["pair_span_radians"][paired]])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(images)
                raw_nll = relative_state_nll(output, delta, span)
                state_nll = raw_nll / (2 * math.log(config["model"]["bins"]))
                reading_cdf = distribution_loss(output["distribution"].float(), truth.float())
                loss = state_nll + reading_cdf
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
            for key, value in (("total", loss), ("state_nll_raw", raw_nll), ("state_nll_normalized", state_nll),
                               ("reading_cdf", reading_cdf), ("invalid_mass", output["invalid_mass"].mean()),
                               ("outside_arc_mass", output["outside_arc_mass"].mean())):
                sums[key] += float(value.detach()) * count
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
        val_loader = make_loader(val_data, batch_size=training["batch_size"], workers=training["workers"],
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
        checkpoint = {"protocol": PROTOCOL, "config": config, "data_contract": data_contract,
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
    write_json(args.output / "completed.json", {"protocol": PROTOCOL, "epochs": training["epochs"],
               "best_epoch": best_epoch, "best_nmae": best_nmae, "totals": totals,
               "current_process_wall_seconds": time.perf_counter() - process_started,
               "checkpointed_process_wall_seconds_before_resume": checkpointed_process_seconds,
               "uncheckpointed_work_after_interruptions": "not included in checkpointed counters; use controller process logs",
               "seed": seed, "seed_expansion": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
