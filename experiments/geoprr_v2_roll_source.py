"""Train the full semantic scalar reader with additional physically rendered source rolls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import inspect
import math
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import torch
import cv2
import numpy as np
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, default_collate

from experiments.geoprr_v2_data import _resolve_image, crop_and_resize, normalized_target, read_rows
from experiments.geoprr_v2_semantic_direct import (
    CONFIG as DIRECT_CONFIG, SOURCE_RELATIVE as ORIGINAL_SOURCE_RELATIVE,
    ScalarRGBDataset, create_model, encoder_counter, evaluate_population,
    model_metadata, save_csv, source_contract as original_source_contract, synchronize,
)
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything, source_split_check,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_roll_source_full_semantic_direct_source_only_v1"
SOURCE_RELATIVE = "rendering/generalization_roll_v22_20261024"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v22_roll_source_20261024/source_only_training"
CONFIG = {
    "protocol": PROTOCOL,
    "model": deepcopy(DIRECT_CONFIG["model"]),
    "data": deepcopy(DIRECT_CONFIG["data"]),
    "photometric": deepcopy(DIRECT_CONFIG["photometric"]),
    "source": {"old_unique": 18590, "roll_unique": 4096, "roll_repeat": 4, "total_unique": 22686,
               "train_presentations": 34974, "original_validation": 1506, "roll_validation": 512,
               "require_geometry": False, "pair_manifest": None,
               "roll_validation_split": "independent layout/seed/ID/group; same available HDR background"},
    "training": {"seed": 20262020, "min_epochs": 60, "max_epochs": 160, "early_stopping_patience": 20,
                 "batch_size": 64, "validation_batch_size": 64, "workers": 4, "amp": True,
                 "optimizer": "AdamW", "learning_rate": .0003, "weight_decay": .0001,
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.,
                 "scheduler": {"mode": "min", "factor": .3, "patience": 8, "threshold": 0.,
                               "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6, "eps": 1e-8}},
    "checkpoint_selection": "earliest strict minimum .5*(original1506 mean NMAE + roll512 mean NMAE)",
    "external_checkpoint": "source-selected best only after source stopping rule completes",
    "inference_input": DIRECT_CONFIG["inference_input"],
    "external_training_or_adaptation": False,
}


def training_population(original, roll):
    if (len(original), len(roll)) != (18590, 4096):
        raise ValueError("source training requires18590 retained and4096 new unique RGB images")
    return ConcatDataset([original] + [roll] * CONFIG["source"]["roll_repeat"])


def selection_score(validation):
    if set(validation) != {"original", "roll"}:
        raise ValueError("selection requires only the two complete source validations")
    for name, count in (("original", 1506), ("roll", 512)):
        value = validation[name]
        if value["samples"] != count or not math.isfinite(value["nmae"]) or not 0 <= value["nmae"] <= 1:
            raise ValueError("source selection population or NMAE is invalid")
    return .5 * (validation["original"]["nmae"] + validation["roll"]["nmae"])


def completion_reason(epoch, best_epoch):
    if epoch >= CONFIG["training"]["max_epochs"]:
        return "maximum_epochs"
    if (epoch >= CONFIG["training"]["min_epochs"]
            and best_epoch is not None and epoch - best_epoch >= CONFIG["training"]["early_stopping_patience"]):
        return "source_score_patience"
    return None


def source_populations(home):
    original = ScalarRGBDataset(home / ORIGINAL_SOURCE_RELATIVE / "manifests/train.jsonl",
                                data_root=home / "datasets", augment=True)
    validation = ScalarRGBDataset(home / "runs/geoprr_v2/manifests/validation.jsonl",
                                  data_root=home / "datasets", augment=False)
    original_contract = original_source_contract(home, original, validation)
    folder = home / SOURCE_RELATIVE
    prepared = json.loads((folder / "prepared.json").read_text())
    if (prepared["status"] != "complete" or prepared["protocol"] != "geoprr_v2_camera_roll_source_v22"
            or json.loads((folder / "queue_state.json").read_text())["status"] != "complete"):
        raise ValueError("physical roll source generation is incomplete")
    roll = ScalarRGBDataset(folder / "manifests/train_roll.jsonl", data_root=home / "datasets", augment=True)
    roll_validation = ScalarRGBDataset(folder / "manifests/validation_roll.jsonl", data_root=home / "datasets", augment=False)
    if (len(roll), len(roll_validation)) != (4096, 512):
        raise ValueError("physical roll source must contain4096 training and512 validation RGB images")
    merged_rows = original.rows + roll.rows
    if read_rows(folder / "manifests/train_combined.jsonl") != merged_rows:
        raise ValueError("combined source changed retained old rows, order or the new roll population")
    source_split_check(merged_rows, validation.rows)
    validate_roll_split(roll.rows, roll_validation.rows)
    for key in ("sample_id", "group_id"):
        if ({r[key] for r in roll_validation.rows} & {r[key] for r in merged_rows + validation.rows}):
            raise ValueError(f"roll validation overlaps another source {key} population")
    expected_counts = {"original_training_images":18590, "train_roll_images":4096,
                       "combined_training_images":22686, "roll_validation_images":512, "original_validation_images":1506}
    if {key:prepared[key] for key in expected_counts} != expected_counts:
        raise ValueError("prepared source counts differ from the declared population")
    for rows, prefix in ((roll.rows, "train"), (roll_validation.rows, "roll_validation")):
        for field, suffix in (("layout_id", "layout_ids"), ("render_seed", "render_seeds")):
            if {r["metadata"]["roll_source"][field] for r in rows} != set(prepared[f"{prefix}_{suffix}"]):
                raise ValueError("prepared physical roll layout or seed roster differs from manifest")
    if prepared["roll_validation_holdout"]["background_independent"] is not False:
        raise ValueError("roll validation must disclose its shared source background")
    datasets = {"original_train": original, "roll_train": roll,
                "original_validation": validation, "roll_validation": roll_validation}
    fields = ("sample_id", "group_id", "image_path", "roi_path", "dataset", "split")
    contract = {"original_source": original_contract, "roll_prepared": prepared,
                "roll_source_manifest": str(folder / "manifests/train_roll.jsonl"),
                "roll_validation_manifest": str(folder / "manifests/validation_roll.jsonl"),
                "roll_validation_background_independent": False,
                "populations": {name: [{**{k: row.get(k) for k in fields}, "target": normalized_target(row),
                    "metadata": {k: row.get("metadata", {}).get(k) for k in
                                 ("dial_bbox", "scene_name", "roll_source")}}
                    for row in dataset.rows] for name, dataset in datasets.items()}}
    return training_population(original, roll), {"original": validation, "roll": roll_validation}, contract


def validate_roll_split(train_rows, validation_rows):
    for rows, split in ((train_rows, "train"), (validation_rows, "roll_validation")):
        for row in rows:
            if str(row.get("dataset", "")).lower() != "syncg" or row.get("split") != split:
                raise ValueError("roll populations must contain their declared synthetic source split")
            if not row.get("metadata", {}).get("scene_name"):
                raise ValueError("roll metadata must retain the actual scene")
    for label, values in (
        ("sample_id", lambda row: str(row["sample_id"])),
        ("group_id", lambda row: str(row["group_id"])),
        ("layout_id", lambda row: str(row["metadata"]["roll_source"]["layout_id"])),
        ("render_seed", lambda row: int(row["metadata"]["roll_source"]["render_seed"])),
    ):
        left, right = ({values(row) for row in rows} for rows in (train_rows, validation_rows))
        if left & right:
            raise ValueError(f"roll training and validation share {label}")
    for rows in (train_rows, validation_rows):
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate physical roll sample IDs")


def validate_resume(saved, contract):
    if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
        raise ValueError("resume protocol, source population or configuration differs")
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "history", "totals", "validation_rows",
                "initialization", "checkpointed_process_seconds", "epoch", "best_score", "best_epoch"}
    if not required <= set(saved):
        raise ValueError("resume requires complete model, optimizer, scheduler, scaler, RNG and source state")
    history = saved["history"]
    if [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
        raise ValueError("resume history does not cover every completed epoch")
    scores = [selection_score(row["validation"]) for row in history]
    best = min(scores)
    if (saved["best_score"], saved["best_epoch"]) != (best, scores.index(best) + 1):
        raise ValueError("resume changed the earliest strict source minimum")
    if history[-1]["selection_score"] != scores[-1] or set(saved["validation_rows"]) != {"original", "roll"}:
        raise ValueError("resume lacks both source-validation populations")
    for name, n in (("original", 1506), ("roll", 512)):
        if len(saved["validation_rows"][name]) != n:
            raise ValueError("resume source scalar rows are incomplete")


def runtime(device, started):
    return {"process_seconds": time.perf_counter() - started,
            "full_process_seconds": time.perf_counter() - PROCESS_STARTED,
            "full_process_boundary": "module entry before torch import through saved predictions; excludes final completion JSON",
            "device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def save_source_rows(output, prefix, populations):
    for name, rows in populations.items():
        save_predictions(output / f"{prefix}_{name}_predictions.csv.gz", rows)


def preflight(home, output):
    if output.exists():
        raise FileExistsError("the new-source preflight already has its saved result")
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    training, validations, contract = source_populations(home)
    roll = training.datasets[1]
    train_batch = device_batch(default_collate([roll[i] for i in (0, 3, 4, 7)]), device)
    validation_batches = {}
    identity_samples = {}
    for name, dataset in validations.items():
        samples = [dataset[i] for i in (0, 1)]
        for i, sample in enumerate(samples):
            row = dataset.rows[i]
            image = cv2.imread(str(_resolve_image(row, dataset.base.manifest_root, dataset.base.dataset_root)))
            roi, _ = crop_and_resize(image, None if "roi_path" in row else row.get("metadata", {}).get("dial_bbox"), 256)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            expected = (expected-expected.new_tensor([.485,.456,.406])[:,None,None])/expected.new_tensor([.229,.224,.225])[:,None,None]
            if sample["input_failure"] or not torch.equal(sample["image"], expected):
                raise ValueError("source validation RGB differs from the declared256 linear ImageNet pipeline")
        validation_batches[name] = device_batch(default_collate(samples), device)
        identity_samples[name] = [s["sample_id"] for s in samples]
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key: value.cpu() for key, value in original.items()}, strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    counters = {key: value.clone() for key, value in model.state_dict().items() if key.endswith("num_batches_tracked")}
    if len(counters) != 77 or list(inspect.signature(model.forward).parameters) != ["image"]:
        raise ValueError("complete semantic model or RGB-only forward interface differs")
    counts, hook = encoder_counter(model)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        prediction = model(train_batch["image"])
        loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        loss.backward()
        all_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        nonzero = {"encoder": any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.encoder.parameters()),
                   "point": any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.point.parameters())}
        after_train = {key: model.state_dict()[key].clone() for key in counters}
        if not all(torch.equal(after_train[key], value+1) for key, value in counters.items()):
            raise ValueError("new source training did not update all encoder BN counters once")
        model.eval()
        before_eval = {key: value.clone() for key, value in model.named_buffers()}
        with torch.inference_mode():
            for batch in validation_batches.values():
                p = model(batch["image"])
                if p.shape != (2,) or not bool(torch.isfinite(p).all()) or not bool(((p>=0)&(p<=1)).all()):
                    raise ValueError("new source validation output is not one finite scalar per input")
        eval_buffers_unchanged = all(torch.equal(value, before_eval[key]) for key, value in model.named_buffers())
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        hook.remove()
    if (counts != {"cnn_forward_calls":3, "cnn_image_forwards":8} or not strict_reload or not all_finite
            or not all(nonzero.values()) or not unchanged or not eval_buffers_unchanged or not bool(torch.isfinite(loss))):
        raise ValueError("new source zero-update execution witness failed")
    write_json(output, {"protocol":PROTOCOL, "status":"passed", "config":CONFIG, "model_metadata":metadata,
        "source_generation_complete":True, "source_unique":22686, "retained_source_unique":18590,
        "new_roll_source_unique":4096, "source_presentations_per_epoch":len(training), "source_batches_per_epoch":547,
        "source_validation_populations":{"original":1506,"roll":512}, "source_validation_batches_per_epoch":32,
        "original_source_scene_split_guard_passed":True, "roll_layout_ID_group_generated_seed_disjoint":True,
        "roll_validation_background_independent":False, "source_probe_states":train_batch["y"].tolist(),
        "probe_sample_ids":train_batch["sample_id"], "validation_identity_sample_ids":identity_samples,
        "validation_input_matches_declared256_pipeline":True, "ImageNet_cached_only":True,
        "old_task_checkpoint_used":False, "model_forward_inputs":["image"], "target_datasets_used":False,
        "target_statistics_used":False, "source_selection_criterion":CONFIG["checkpoint_selection"],
        "source_BN_counter_tensors_checked":77, "source_BN_counter_increment_training":1,
        "source_BN_counter_additional_increment_eval":0, "eval_model_buffers_unchanged":eval_buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "all_gradients_finite":all_finite, "gradient_groups_nonzero":nonzero, "source_loss":float(loss.detach()),
        "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":1,"combined_backward_calls":1,"optimizer_updates":0},
        "runtime":runtime(device,started), "scope":"new real-roll source contract and zero-update execution witness"})


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(CONFIG["training"]["seed"])
    if (output / "completed.json").exists():
        raise FileExistsError("source training is complete; reuse the completed stage")
    latest = output / "latest.pt"
    if not latest.exists() and any((output / p).exists() for p in ("history.jsonl", "best.pt")):
        raise FileExistsError("partial source outputs exist without the complete latest checkpoint")
    training, validations, contract = source_populations(home)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    if saved is not None:
        validate_resume(saved, contract)
    model = create_model(pretrained=saved is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **CONFIG["training"]["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {k: 0 for k in ("train_batches", "train_cnn_image_forwards", "train_cnn_forward_calls", "backward_calls",
        "optimizer_updates", "amp_skipped_updates", "validation_images", "validation_cnn_forward_calls")}
    totals.update(training_seconds=0., validation_seconds=0.)
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": CONFIG["model"]["encoder"], "ImageNet_pretrained": True,
                      "existing_cached_weights_only": True, "retained_complete_encoder_state_tensors": 462,
                      "new_point_parameter_tensors": 2, "old_task_checkpoint_used": False}
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch = saved["epoch"], saved["best_score"], saved["best_epoch"]
        history, totals, latest_rows = saved["history"], saved["totals"], saved["validation_rows"]
        checkpointed_seconds, initialization = saved["checkpointed_process_seconds"], saved["initialization"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    exposure = {"train_unique_images": 22686, "old_train_images": 18590, "roll_unique_images": 4096,
                "roll_repetitions": 4, "train_images": len(training), "train_batches": 547,
                "original_validation_images": 1506, "roll_validation_images": 512,
                "validation_images": 2018, "validation_batches": 32}
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "SmoothL1(beta=.05) on scalar source labels"})
    torch.cuda.reset_peak_memory_stats(device)
    reason = completion_reason(start, best_epoch)
    for epoch in range(start + 1, CONFIG["training"]["max_epochs"] + 1):
        if reason is not None:
            break
        tick = time.perf_counter()
        model.train()
        used_lr = optimizer.param_groups[0]["lr"]
        loader = make_loader(training, batch_size=64, workers=4, shuffle=True,
                             seed=20262020 + epoch, device=device)
        counts, hook = encoder_counter(model)
        batches = updates = skips = 0
        loss_sum = error_sum = 0.
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
                n = len(batch["y"])
                batches += 1; updates += int(not skipped); skips += int(skipped)
                loss_sum += float(loss.detach()) * n
                error_sum += float((prediction.detach() - batch["y"]).abs().sum())
                if index == 0:
                    print(json.dumps({"event": "training_batch", "epoch": epoch, "batch": 1, "images": n,
                        "loss": float(loss.detach()), "learning_rate": used_lr,
                        "gradient_norm_finite": bool(torch.isfinite(gradient_norm))}), flush=True)
        finally:
            hook.remove()
        synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (547, 547, 34974):
            raise RuntimeError("source training differs from old-once plus roll-four-times exposure")
        latest_rows, validation, validation_cost = {}, {}, {}
        for name, dataset in validations.items():
            latest_rows[name], validation_cost[name] = evaluate_population(model, dataset, device)
            validation[name] = summarize(latest_rows[name])
        score = selection_score(validation)
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        scheduler.step(score)
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
            "train_cnn_forward_calls": counts["cnn_forward_calls"], "backward_calls": batches,
            "optimizer_updates": updates, "amp_skipped_updates": skips, "validation_images": 2018,
            "validation_cnn_forward_calls": sum(c["counts"]["cnn_forward_calls"] for c in validation_cost.values()),
            "training_seconds": train_seconds,
            "validation_seconds": sum(c["pipeline_seconds"] for c in validation_cost.values())}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "source_training_smooth_l1": loss_sum / 34974, "source_training_nmae": error_sum / 34974,
            "train_counts": counts, "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
            "validation": validation, "validation_cost": validation_cost, "selection_score": score,
            "best_score": best, "best_epoch": best_epoch, "epochs_since_strict_best": epoch-best_epoch,
            "training_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "epoch": epoch, "best_epoch": best_epoch, "best_score": best,
            "history": history, "totals": totals, "validation_rows": latest_rows, "initialization": initialization,
            "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            save_source_rows(output, "best", latest_rows)
            write_json(output / "best_validation.json", {"selection_score": score, "validation": validation})
        save_source_rows(output, "latest", latest_rows)
        write_json(output / "latest_validation.json", {"selection_score": score, "validation": validation})
        (output / "history.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
        save_checkpoint(latest, checkpoint)
        reason = completion_reason(epoch, best_epoch)
        print(json.dumps({"event": "epoch_complete", "epoch": epoch, "selection_score_percent_fs": score*100,
                          "best_epoch": best_epoch, "completion_reason": reason}), flush=True)
    epochs = len(history)
    if (reason is None or not 60 <= epochs <= 160 or totals["train_batches"] != 547*epochs
            or totals["train_cnn_image_forwards"] != 34974*epochs or totals["validation_images"] != 2018*epochs
            or totals["validation_cnn_forward_calls"] != 32*epochs
            or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 547*epochs):
        raise RuntimeError("source completion does not match the declared stopping rule and actual exposure")
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": epochs,
        "completion_reason": reason, "best_epoch": best_epoch, "best_score": best,
        "best_validation": history[best_epoch-1]["validation"], "last_validation": history[-1]["validation"],
        "totals": totals, "model_metadata": metadata, "initialization": initialization, "epoch_exposure": exposure,
        "source_sample_ids": {name: [r["sample_id"] for r in rows] for name, rows in contract["populations"].items()},
        "resume_from_epoch": start, "runtime": runtime(device, started),
        "checkpointed_process_seconds_before_resume": checkpointed_seconds,
        "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def validate_external_checkpoint(checkpoint, completed):
    reason = completion_reason(completed["epochs"], completed["best_epoch"])
    if (checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG
            or completed["protocol"] != PROTOCOL or completed["config"] != CONFIG
            or completed["status"] != "complete" or not 60 <= completed["epochs"] <= 160
            or reason is None or reason != completed["completion_reason"]
            or checkpoint["epoch"] != completed["best_epoch"] or checkpoint["epoch"] != checkpoint["best_epoch"]
            or checkpoint["best_score"] != completed["best_score"]):
        raise ValueError("external evaluation requires this protocol's source-selected best after complete source training")


def evaluate(home, checkpoint_path, dataset_name, output):
    started, device = time.perf_counter(), torch.device("cuda")
    if (output / "summary.json").exists():
        raise FileExistsError("this frozen external evaluation is complete; reuse it without another forward")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = json.loads((checkpoint_path.parent / "completed.json").read_text())
    validate_external_checkpoint(checkpoint, completed)
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
        row.update(dataset=dataset_name, condition="native256_physical_roll_source", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
        "summary": summarize(rows), "by_group": {g: summarize(rr) for g, rr in groups.items()},
        "by_component": {c: summarize(rr) for c, rr in components.items()}, "forward_counts": cost["counts"],
        "timing": cost, "runtime": runtime(device, started), "model_inputs": CONFIG["inference_input"],
        "training_or_adaptation_during_evaluation": False, "backward_calls": 0, "optimizer_updates": 0,
        "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
        "development_scope": "previously viewed external development; no target training, BN adaptation or selection",
        "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
        "source_completion_epochs": completed["epochs"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-name", choices=("industrial", "rf100"))
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        preflight(args.home, args.output)
    elif args.stage == "train":
        train(args.home, args.output)
    else:
        if args.checkpoint is None or args.dataset_name is None:
            parser.error("evaluation requires --checkpoint and --dataset-name")
        evaluate(args.home, args.checkpoint, args.dataset_name, args.output)


if __name__ == "__main__":
    main()
