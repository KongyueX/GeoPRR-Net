"""Train a complete ImageNet MobileNetV4 semantic encoder for source-only scalar reading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize, write_csv
from experiments.geoprr_v2_data import GeoPRRV2Dataset, normalized_target
from experiments.geoprr_v2_endpoint_source import DATA_PROTOCOL, REPLACEMENT
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything, source_split_check,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_full_semantic_direct_source_only_v1"
ENCODER_NAME = "mobilenetv4_conv_medium.e500_r224_in1k"
SOURCE_RELATIVE = "rendering/generalization_endpoints_v11_20261023"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {"encoder": ENCODER_NAME, "num_classes": 0,
              "readout": "sigmoid(linear(full encoder semantic output)); semantic conv_head and norm_head retained",
              "point_initialization_std": .001, "point_initialization_bias": 0.},
    "data": {"image_size": 256, "output_size": 64},
    "source": {"unique_training_images": 18590, "validation_images": 1506,
               "source_replacement": REPLACEMENT, "require_geometry": False, "pair_manifest": None},
    "photometric": {"contrast": [.8, 1.2], "brightness": [-.1, .1], "saturation": [.8, 1.2]},
    "training": {"seed": 20262020, "epochs": 60, "batch_size": 64, "validation_batch_size": 64,
                 "workers": 4, "amp": True, "optimizer": "AdamW", "weight_decay": .0001,
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.,
                 "learning_rate": "epoch1..30 cosine3e-4; epoch31..60 cosine3e-5; phase position0..29/30"},
    "checkpoint_selection": "earliest minimum normalized scalar NMAE on complete source1506 within60epochs",
    "external_checkpoint": "source-selected best only after all60epochs complete",
    "inference_input": "ROI RGB, linear resize256, ImageNet normalization; no GT geometry or target scalar input",
    "external_training_or_adaptation": False,
}


class ScalarRGBDataset(Dataset):
    """Use the existing RGB transform without geometry targets or paired exposure."""

    def __init__(self, manifest, *, data_root, augment):
        self.base = GeoPRRV2Dataset(manifest, dataset_root=data_root, **CONFIG["data"],
                                   require_geometry=False, augment=augment, pair_manifest=None)
        self.rows, self.augment = self.base.rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        failure = ""
        try:
            sample = self.base[index]
        except (OSError, ValueError, cv2.error) as error:
            if self.augment:
                raise
            row = self.rows[index]
            failure = f"{type(error).__name__}: {error}"
            sample = {"image": torch.zeros(3, 256, 256), "y": torch.tensor(normalized_target(row), dtype=torch.float32),
                      "sample_id": str(row["sample_id"]), "group_id": str(row.get("group_id", row["sample_id"]))}
        return {**{key: sample[key] for key in ("image", "y", "sample_id", "group_id")}, "input_failure": failure}


class SemanticDirectReader(nn.Module):
    """Keep the complete classifier representation and replace only its scalar readout."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.semantic_dim = int(encoder.conv_head.out_channels)
        self.point = nn.Linear(self.semantic_dim, 1)
        nn.init.normal_(self.point.weight, std=.001)
        nn.init.zeros_(self.point.bias)

    def forward(self, image):
        semantic = self.encoder(image)
        if semantic.shape != (len(image), self.semantic_dim):
            raise ValueError("full encoder must return one semantic vector per RGB input")
        return self.point(semantic).squeeze(-1).float().sigmoid()


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    model = SemanticDirectReader(encoder)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"],
            metadata["parameters"], metadata["state_tensors"]) != (1280, 8434512, 462, 8435793, 464):
        raise ValueError("complete semantic encoder architecture differs from the frozen cached model")
    return model


def model_metadata(model):
    return {"class": f"{type(model).__module__}.{type(model).__name__}",
            "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
            "encoder_name": ENCODER_NAME, "semantic_dim": model.semantic_dim,
            "encoder_reported_num_features": model.encoder.num_features,
            "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
            "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
            "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
            "encoder_state_tensors": len(model.encoder.state_dict()),
            "point_parameters": sum(parameter.numel() for parameter in model.point.parameters()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_tensors": len(model.state_dict())}


def learning_rate(epoch):
    if not 1 <= epoch <= 60:
        raise ValueError("the new semantic source experiment has exactly60 epochs")
    base, position = (3e-4, epoch - 1) if epoch <= 30 else (3e-5, epoch - 31)
    return base * .5 * (1 + math.cos(math.pi * position / 30))


def scalar_rows(prediction, batch):
    prediction = prediction.detach().float().cpu().reshape(-1).tolist()
    target = batch["y"].detach().float().cpu().reshape(-1).tolist()
    if len(prediction) != len(target):
        raise ValueError("scalar output and target population lengths differ")
    result = []
    for index, (value, truth) in enumerate(zip(prediction, target, strict=True)):
        input_valid = not batch["input_failure"][index]
        valid = bool(input_valid and math.isfinite(value) and 0 <= value <= 1)
        result.append({"sample_id": batch["sample_id"][index], "group_id": batch["group_id"][index],
                       "target": truth, "prediction": value if valid else None, "valid": valid,
                       "input_valid": bool(input_valid), "absolute_error": abs(value - truth) if valid else 1.,
                       "invalid_mass": None, "outside_arc_mass": None,
                       "failure": batch["input_failure"][index] or ("" if valid else "invalid_scalar_prediction")})
    return result


def save_csv(path, rows):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            saved = list(csv.DictReader(handle))
        expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
        if saved != expected:
            raise ValueError("saved partial scalar rows differ; retain and inspect the failed evaluation")
    else:
        write_csv(path, rows)


def encoder_counter(model):
    counts = {"cnn_forward_calls": 0, "cnn_image_forwards": 0}

    def count(_module, inputs, _output):
        counts["cnn_forward_calls"] += 1
        counts["cnn_image_forwards"] += len(inputs[0])

    return counts, model.encoder.register_forward_hook(count)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_population(model, dataset, device):
    model.eval()
    counts, hook = encoder_counter(model)
    loader = make_loader(dataset, batch_size=64, workers=4, shuffle=False, seed=20262020, device=device)
    rows, forward_seconds = [], 0.
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for raw in loader:
                batch = device_batch(raw, device)
                synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    prediction = model(batch["image"])
                synchronize(device)
                forward_seconds += time.perf_counter() - tick
                rows.extend(scalar_rows(prediction, batch))
    finally:
        hook.remove()
    if ([row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in dataset.rows]
            or counts["cnn_image_forwards"] != len(dataset)
            or counts["cnn_forward_calls"] != math.ceil(len(dataset) / 64)):
        raise RuntimeError("scalar evaluation lost or repeated declared RGB inputs")
    counts.update(backward_calls=0, optimizer_updates=0,
                  failed_input_placeholder_forwards=sum(not row["input_valid"] for row in rows))
    return rows, {"counts": counts, "pipeline_seconds": time.perf_counter() - started,
                  "forward_seconds": forward_seconds, "batch_size": 64, "workers": 4}


def source_contract(home, training, validation):
    source = home / SOURCE_RELATIVE
    prepared = json.loads((source / "prepared.json").read_text())
    if (json.loads((source / "queue_state.json").read_text())["status"] != "complete"
            or prepared["protocol"] != DATA_PROTOCOL or prepared["replacement"] != REPLACEMENT):
        raise ValueError("the existing128-group512-image endpoint source is not complete")
    if len(training) != 18590 or len(validation) != 1506:
        raise ValueError("new scalar training must retain18590 unique source and1506 source validation images")
    source_split_check(training.rows, validation.rows)
    fields = ("sample_id", "group_id", "image_path", "roi_path")
    populations = {split: [{**{key: row.get(key) for key in fields}, "target": normalized_target(row),
                           "dial_bbox": row.get("metadata", {}).get("dial_bbox")}
                          for row in dataset.rows] for split, dataset in (("train", training), ("validation", validation))}
    return {"source_train_manifest": str((source / "manifests/train.jsonl").resolve()),
            "source_validation_manifest": str((home / "runs/geoprr_v2/manifests/validation.jsonl").resolve()),
            "source_replacement": REPLACEMENT, "data": CONFIG["data"], "require_geometry": False,
            "populations": populations, "pair_exposure": False}


def runtime(device, started):
    return {"process_seconds": time.perf_counter() - started,
            "full_process_seconds": time.perf_counter() - PROCESS_STARTED,
            "full_process_boundary": "module entry before torch import through saved source/external predictions; excludes final completion JSON",
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("source training is complete; the queue must reuse this stage")
    latest = output / "latest.pt"
    if not latest.exists() and any((output / name).exists() for name in ("history.jsonl", "best.pt")):
        raise FileExistsError("partial source outputs exist without latest.pt; inspect the retained failure")
    source = home / SOURCE_RELATIVE
    train_data = ScalarRGBDataset(source / "manifests/train.jsonl", data_root=home / "datasets", augment=True)
    validation = ScalarRGBDataset(home / "runs/geoprr_v2/manifests/validation.jsonl", data_root=home / "datasets", augment=False)
    contract = source_contract(home, train_data, validation)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    model = create_model(pretrained=saved is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {"train_batches": 0, "train_cnn_image_forwards": 0, "train_cnn_forward_calls": 0,
              "backward_calls": 0, "optimizer_updates": 0, "amp_skipped_updates": 0,
              "validation_images": 0, "validation_cnn_forward_calls": 0,
              "training_seconds": 0., "validation_seconds": 0.}
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
                      "retained_complete_encoder_state_tensors": 462,
                      "new_point_parameter_tensors": 2, "old_task_checkpoint_used": False}
    if saved is not None:
        if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
            raise ValueError("resumed source semantic experiment protocol, config or population differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch, history, totals = saved["epoch"], saved["best_nmae"], saved["best_epoch"], saved["history"], saved["totals"]
        checkpointed_seconds, latest_rows = saved["checkpointed_process_seconds"], saved["validation_rows"]
        initialization = saved["initialization"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
               "initialization": initialization, "model_metadata": metadata,
               "train_unique_images": 18590, "validation_images": 1506,
               "epoch_exposure": {"train_images": 18590, "train_batches": 291, "validation_images": 1506,
                                  "validation_batches": 24},
               "inference_input": CONFIG["inference_input"], "source_dataset_class": f"{__name__}.ScalarRGBDataset",
               "source_only_loss": "normalized source reading SmoothL1(beta=.05); no geometry or auxiliary loss",
               "comparison": "new full semantic ImageNet initialization and256 pipeline; independent60epoch source-only budget"})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, 61):
        tick = time.perf_counter()
        model.train()
        used_lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = used_lr
        loader = make_loader(train_data, batch_size=64, workers=4, shuffle=True, seed=20262020 + epoch, device=device)
        counts, hook = encoder_counter(model)
        batches, updates, skips, loss_sum, error_sum = 0, 0, 0, 0., 0.
        try:
            for index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction, batch["y"], beta=.05)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite scalar loss at epoch{epoch} batch{index + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                count = len(batch["y"])
                batches += 1
                updates += int(not skipped)
                skips += int(skipped)
                loss_sum += float(loss.detach()) * count
                error_sum += float((prediction.detach() - batch["y"]).abs().sum())
                if index == 0:
                    print(json.dumps({"event": "training_batch", "epoch": epoch, "batch": 1, "images": count,
                                      "loss": float(loss.detach()), "learning_rate": used_lr,
                                      "gradient_norm_finite": bool(torch.isfinite(gradient_norm))}), flush=True)
        finally:
            hook.remove()
        synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (291, 291, 18590):
            raise RuntimeError("source scalar training repeated or omitted unique RGB exposure")
        latest_rows, validation_cost = evaluate_population(model, validation, device)
        summary = summarize(latest_rows)
        improved = summary["nmae"] < best
        if improved:
            best, best_epoch = summary["nmae"], epoch
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
                           "train_cnn_forward_calls": counts["cnn_forward_calls"], "backward_calls": batches,
                           "optimizer_updates": updates, "amp_skipped_updates": skips,
                           "validation_images": len(latest_rows),
                           "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "source_training_smooth_l1": loss_sum / 18590,
                  "source_training_nmae": error_sum / 18590, "train_counts": counts,
                  "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
                  "validation": summary, "validation_cost": validation_cost,
                  "best_nmae": best, "best_epoch": best_epoch, "training_seconds": train_seconds,
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                      "epoch": epoch, "best_epoch": best_epoch, "best_nmae": best, "history": history,
                      "totals": totals, "validation_rows": latest_rows, "initialization": initialization,
                      "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            save_predictions(output / "best_predictions.csv.gz", latest_rows)
            write_json(output / "best_validation.json", summary)
        save_predictions(output / "latest_predictions.csv.gz", latest_rows)
        (output / "history.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
        save_checkpoint(latest, checkpoint)
        print(json.dumps({"event": "epoch_complete", "epoch": epoch, "nmae_percent_fs": summary["nmae_percent_fs"],
                          "best_epoch": best_epoch}), flush=True)
    if (len(history), totals["train_batches"], totals["train_cnn_image_forwards"], totals["validation_images"],
            totals["validation_cnn_forward_calls"], totals["backward_calls"]) != (60, 17460, 1115400, 90360, 1440, 17460):
        raise RuntimeError("completed scalar training does not contain the complete60epoch declared source exposure")
    if totals["optimizer_updates"] + totals["amp_skipped_updates"] != 17460:
        raise RuntimeError("effective update and AMP skip counts differ from actual attempts")
    if not (output / "latest_predictions.csv.gz").exists():
        save_predictions(output / "latest_predictions.csv.gz", latest_rows)
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": 60,
               "best_epoch": best_epoch, "best_nmae": best, "best_validation": history[best_epoch - 1]["validation"],
               "last_validation": history[-1]["validation"], "totals": totals, "model_metadata": metadata,
               "initialization": initialization, "source_sample_ids": {"train": [row["sample_id"] for row in train_data.rows],
                                                                        "validation": [row["sample_id"] for row in validation.rows]},
               "resume_from_epoch": start, "runtime": runtime(device, started),
               "checkpointed_process_seconds_before_resume": checkpointed_seconds,
               "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def evaluate(home, checkpoint_path, dataset_name, output):
    started, device = time.perf_counter(), torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = json.loads((checkpoint_path.parent / "completed.json").read_text())
    if (checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG
            or completed["status"] != "complete" or completed["epochs"] != 60
            or checkpoint["epoch"] != completed["best_epoch"] or checkpoint["epoch"] != checkpoint["best_epoch"]):
        raise ValueError("external evaluation requires the source-selected full semantic best after all60 epochs")
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    rows, cost = evaluate_population(model, dataset, device)
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256", component=source["component"])
        groups[row["group_id"]].append(row)
        components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": key, **summarize(value)} for key, value in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
               "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
               "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
               "checkpoint_source_best_nmae": checkpoint["best_nmae"], "model_metadata": model_metadata(model),
               "summary": summarize(rows), "by_group": {key: summarize(value) for key, value in groups.items()},
               "by_component": {key: summarize(value) for key, value in components.items()},
               "forward_counts": cost["counts"], "timing": cost, "runtime": runtime(device, started),
               "model_inputs": CONFIG["inference_input"], "training_or_adaptation_during_evaluation": False,
               "backward_calls": 0, "optimizer_updates": 0,
               "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
               "development_scope": "previously viewed external development; no target training, BN adaptation or checkpoint selection",
               "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
               "pipeline_change": "new256 linear resize for this complete semantic model; not old768 input identity"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-name", choices=("industrial", "rf100"))
    args = parser.parse_args(argv)
    if args.stage == "train":
        train(args.home, args.output)
    else:
        if args.checkpoint is None or args.dataset_name is None:
            parser.error("evaluation requires --checkpoint and --dataset-name")
        evaluate(args.home, args.checkpoint, args.dataset_name, args.output)


if __name__ == "__main__":
    main()
