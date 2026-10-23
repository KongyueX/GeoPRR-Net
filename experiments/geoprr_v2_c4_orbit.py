"""Train a fixed C4 orbit-pooled semantic reader using source RGB and scalar labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_semantic_direct as direct
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_c4_orbit_semantic_source_only_v1"
SOURCE_RELATIVE = direct.SOURCE_RELATIVE
ORBIT = {"quarter_turns": [0, 1, 2, 3], "input_order": "k-major concatenation of rotated original RGB tensors",
         "rotation": "torch.rot90 on spatial dimensions before the encoder; no interpolation or reflection",
         "pooling": "float32 arithmetic mean of four semantic vectors before the linear scalar logit",
         "label": "one unchanged physical scalar per original ROI", "geometry_input": False,
         "CNN_views_per_original": 4, "train_validation_external_policy_identical": True}
EPOCHS, BATCH_SIZE, BATCHES_PER_EPOCH, TOTAL_ATTEMPTS = 15, 16, 1162, 17430
CONFIG = deepcopy(direct.CONFIG)
CONFIG.update(protocol=PROTOCOL, orbit=ORBIT,
              checkpoint_selection="earliest minimum direct scalar NMAE on complete source1506 within15 orbit epochs",
              external_checkpoint="source-selected best only after all15 orbit epochs complete; no incumbent alias",
              inference_input="native ROI RGB, linear resize256, ImageNet normalization, fixed C4 tensor orbit and semantic mean; no GT geometry or target scalar input")
CONFIG["model"]["readout"] = "sigmoid(linear(mean_k(full encoder(rot90_k RGB)))); full semantic neck retained"
CONFIG["training"].update(epochs=EPOCHS, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE,
                          learning_rate="attempt i maps to virtual epoch1+floor(i*60/17430); use v16 two-phase cosine at that virtual epoch")
CONFIG["comparison"] = {"matched_training_CNN_views": 1115400, "v16_training_epochs": 60,
                        "orbit_training_epochs": 15, "source_photometric_draws_per_unique_image": 15,
                        "source_validation_checkpoint_choices": 15,
                        "BN_training_batch": "16 original ROIs with4 correlated orientations each; not64 independent ROIs",
                        "external_CNN_views_per_original": 4, "equal_CNN_view_count_is_not_equal_compute": True}


def orbit_views(image):
    if image.ndim != 4 or image.shape[-2] != image.shape[-1]:
        raise ValueError("the C4 reader requires a square BCHW RGB tensor")
    return torch.cat([torch.rot90(image, k, (-2, -1)) for k in ORBIT["quarter_turns"]], dim=0)


class C4OrbitReader(direct.SemanticDirectReader):
    """Pool an image orbit before scalar projection without adding parameters."""

    def __init__(self, encoder, point=None):
        if point is None:
            super().__init__(encoder)
        else:
            nn.Module.__init__(self)
            self.encoder, self.point = encoder, point
            self.semantic_dim = int(encoder.conv_head.out_channels)

    def forward(self, image):
        semantic = self.encoder(orbit_views(image))
        if semantic.shape != (4 * len(image), self.semantic_dim):
            raise ValueError("full encoder must return one semantic vector for each C4 RGB view")
        pooled = semantic.float().reshape(4, len(image), self.semantic_dim).mean(0)
        return self.point(pooled).squeeze(-1).float().sigmoid()


def create_model(*, pretrained):
    base = direct.create_model(pretrained=pretrained)
    return C4OrbitReader(base.encoder, base.point)


def model_metadata(model):
    return {**direct.model_metadata(model), "orbit": ORBIT, "new_parameters": 0,
            "inference_CNN_views_per_original": 4, "deployment_auxiliary_modules": 0}


def virtual_epoch(attempt_index):
    if not 0 <= attempt_index < TOTAL_ATTEMPTS:
        raise ValueError("the C4 experiment has exactly17430 backward attempts")
    return 1 + attempt_index * 60 // TOTAL_ATTEMPTS


def learning_rate(attempt_index):
    return direct.learning_rate(virtual_epoch(attempt_index))


def source_contract(home, training, validation):
    return {"rgb": direct.source_contract(home, training, validation), "orbit": ORBIT}


def runtime(device, started):
    result = direct.runtime(device, started)
    result["full_process_seconds"] = time.perf_counter() - PROCESS_STARTED
    return result


def evaluate_population(model, dataset, device):
    model.eval()
    counts, hook = direct.encoder_counter(model)
    loader = make_loader(dataset, batch_size=BATCH_SIZE, workers=4, shuffle=False, seed=20262020, device=device)
    rows, forward_seconds = [], 0.
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for raw in loader:
                batch = device_batch(raw, device)
                direct.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    prediction = model(batch["image"])
                direct.synchronize(device)
                forward_seconds += time.perf_counter() - tick
                rows.extend(direct.scalar_rows(prediction, batch))
    finally:
        hook.remove()
    if ([row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in dataset.rows]
            or counts["cnn_image_forwards"] != 4 * len(dataset)
            or counts["cnn_forward_calls"] != math.ceil(len(dataset) / BATCH_SIZE)):
        raise RuntimeError("orbit evaluation lost or repeated original inputs or their four CNN views")
    failed = sum(not row["input_valid"] for row in rows)
    counts.update(original_image_presentations=len(rows), backward_calls=0, optimizer_updates=0,
                  failed_original_inputs=failed, failed_placeholder_CNN_views=4 * failed)
    return rows, {"counts": counts, "pipeline_seconds": time.perf_counter() - started,
                  "forward_seconds": forward_seconds, "original_batch_size": BATCH_SIZE,
                  "maximum_encoder_batch_size": 4 * BATCH_SIZE, "workers": 4}


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("source orbit training is complete; reuse the saved stage")
    latest = output / "latest.pt"
    if not latest.exists() and any((output / name).exists() for name in ("history.jsonl", "best.pt")):
        raise FileExistsError("partial orbit outputs exist without latest.pt; inspect the retained failure")
    source = home / SOURCE_RELATIVE
    training = direct.ScalarRGBDataset(source / "manifests/train.jsonl", data_root=home / "datasets", augment=True)
    validation = direct.ScalarRGBDataset(home / "runs/geoprr_v2/manifests/validation.jsonl", data_root=home / "datasets", augment=False)
    contract = source_contract(home, training, validation)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    model = create_model(pretrained=saved is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate(0), weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {"train_batches": 0, "train_original_image_presentations": 0, "train_cnn_image_forwards": 0,
              "train_cnn_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0, "amp_skipped_updates": 0,
              "validation_original_image_presentations": 0, "validation_cnn_image_forwards": 0,
              "validation_cnn_forward_calls": 0, "training_seconds": 0., "validation_seconds": 0.,
              "virtual_epoch_attempts": [0] * 60}
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": direct.ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
                      "retained_complete_encoder_state_tensors": 462, "new_point_parameter_tensors": 2,
                      "base_creation_matches_v16_before_parameter_free_orbit_wrapper": True,
                      "old_task_checkpoint_used": False, "trained_v16_v17_v18_checkpoint_used": False}
    if saved is not None:
        if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
            raise ValueError("resumed orbit protocol, config or source contract differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch = saved["epoch"], saved["best_nmae"], saved["best_epoch"]
        history, totals = saved["history"], saved["totals"]
        checkpointed_seconds, latest_rows = saved["checkpointed_process_seconds"], saved["validation_rows"]
        initialization = saved["initialization"]
        if (saved["global_attempt_index"] != totals["backward_calls"]
                or totals["backward_calls"] != start * BATCHES_PER_EPOCH
                or sum(totals["virtual_epoch_attempts"]) != totals["backward_calls"]):
            raise ValueError("resumed virtual learning-rate position differs from completed backward attempts")
    output.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
               "initialization": initialization, "model_metadata": metadata, "train_unique_images": 18590,
               "validation_images": 1506, "epoch_exposure": {"train_original_images": 18590,
                   "train_CNN_views": 74360, "train_batches": 1162, "validation_original_images": 1506,
                   "validation_CNN_views": 6024, "validation_batches": 95},
               "source_dataset_class": f"{direct.__name__}.ScalarRGBDataset",
               "source_only_loss": "one SmoothL1(beta=.05) per original source scalar after semantic orbit pooling",
               "inference_input": CONFIG["inference_input"], "comparison": CONFIG["comparison"]})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, EPOCHS + 1):
        tick = time.perf_counter()
        model.train()
        loader = make_loader(training, batch_size=BATCH_SIZE, workers=4, shuffle=True,
                             seed=20262020 + epoch, device=device)
        counts, hook = direct.encoder_counter(model)
        batches, originals, updates, skips, loss_sum, error_sum = 0, 0, 0, 0, 0., 0.
        segments = []
        try:
            for index, raw in enumerate(loader):
                attempt = totals["backward_calls"] + index
                used_lr, used_virtual_epoch = learning_rate(attempt), virtual_epoch(attempt)
                for group in optimizer.param_groups:
                    group["lr"] = used_lr
                if not segments or segments[-1]["virtual_epoch"] != used_virtual_epoch:
                    segments.append({"virtual_epoch": used_virtual_epoch, "learning_rate": used_lr,
                                     "first_attempt_index": attempt, "last_attempt_index": attempt, "attempts": 0})
                segments[-1]["last_attempt_index"] = attempt
                segments[-1]["attempts"] += 1
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction, batch["y"], beta=.05)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite orbit scalar loss at epoch{epoch} batch{index + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                count = len(batch["y"])
                batches += 1
                originals += count
                updates += int(not skipped)
                skips += int(skipped)
                loss_sum += float(loss.detach()) * count
                error_sum += float((prediction.detach() - batch["y"]).abs().sum())
                if index == 0:
                    print(json.dumps({"event": "training_batch", "epoch": epoch, "batch": 1,
                                      "original_images": count, "CNN_views": 4 * count, "loss": float(loss.detach()),
                                      "global_attempt_index": attempt, "virtual_epoch": used_virtual_epoch,
                                      "learning_rate": used_lr, "gradient_norm_finite": bool(torch.isfinite(gradient_norm))}), flush=True)
        finally:
            hook.remove()
        direct.synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, originals, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (1162, 18590, 1162, 74360):
            raise RuntimeError("orbit source training repeated or omitted original RGB inputs or their CNN views")
        latest_rows, validation_cost = evaluate_population(model, validation, device)
        summary = summarize(latest_rows)
        improved = summary["nmae"] < best
        if improved:
            best, best_epoch = summary["nmae"], epoch
        for key, value in {"train_batches": batches, "train_original_image_presentations": originals,
                           "train_cnn_image_forwards": counts["cnn_image_forwards"], "train_cnn_forward_calls": counts["cnn_forward_calls"],
                           "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
                           "validation_original_image_presentations": len(latest_rows),
                           "validation_cnn_image_forwards": validation_cost["counts"]["cnn_image_forwards"],
                           "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        for segment in segments:
            totals["virtual_epoch_attempts"][segment["virtual_epoch"] - 1] += segment["attempts"]
        counts["original_image_presentations"] = originals
        record = {"epoch": epoch, "source_training_smooth_l1": loss_sum / originals,
                  "source_training_nmae": error_sum / originals, "train_counts": counts,
                  "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
                  "first_virtual_epoch": segments[0]["virtual_epoch"], "last_virtual_epoch": segments[-1]["virtual_epoch"],
                  "first_learning_rate": segments[0]["learning_rate"], "last_learning_rate": segments[-1]["learning_rate"],
                  "learning_rate_segments": segments, "validation": summary, "validation_cost": validation_cost,
                  "best_nmae": best, "best_epoch": best_epoch, "training_seconds": train_seconds,
                  "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                      "epoch": epoch, "best_epoch": best_epoch, "best_nmae": best, "history": history, "totals": totals,
                      "global_attempt_index": totals["backward_calls"], "validation_rows": latest_rows,
                      "initialization": initialization,
                      "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            save_predictions(output / "best_predictions.csv.gz", latest_rows)
            write_json(output / "best_validation.json", summary)
        save_predictions(output / "latest_predictions.csv.gz", latest_rows)
        (output / "history.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
        save_checkpoint(latest, checkpoint)
        print(json.dumps({"event": "epoch_complete", "epoch": epoch, "nmae_percent_fs": summary["nmae_percent_fs"],
                          "best_epoch": best_epoch, "global_attempt_index": totals["backward_calls"]}), flush=True)
    if (len(history), totals["train_original_image_presentations"], totals["train_cnn_image_forwards"],
            totals["train_cnn_forward_calls"], totals["backward_calls"], totals["validation_original_image_presentations"],
            totals["validation_cnn_image_forwards"], totals["validation_cnn_forward_calls"]) != (
            15, 278850, 1115400, 17430, 17430, 22590, 90360, 1425):
        raise RuntimeError("completed C4 training lacks the declared15 complete orbit epochs")
    expected_schedule = [sum(virtual_epoch(index) == epoch for index in range(TOTAL_ATTEMPTS)) for epoch in range(1, 61)]
    if (totals["optimizer_updates"] + totals["amp_skipped_updates"] != TOTAL_ATTEMPTS
            or totals["virtual_epoch_attempts"] != expected_schedule):
        raise RuntimeError("orbit effective updates, AMP skips or virtual learning-rate attempts differ")
    if not (output / "latest_predictions.csv.gz").exists():
        save_predictions(output / "latest_predictions.csv.gz", latest_rows)
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": EPOCHS,
               "best_epoch": best_epoch, "best_nmae": best, "best_validation": history[best_epoch - 1]["validation"],
               "last_validation": history[-1]["validation"], "totals": totals, "model_metadata": metadata,
               "initialization": initialization, "source_sample_ids": {"train": [row["sample_id"] for row in training.rows],
                   "validation": [row["sample_id"] for row in validation.rows]}, "resume_from_epoch": start,
               "global_attempt_index": totals["backward_calls"], "runtime": runtime(device, started),
               "checkpointed_process_seconds_before_resume": checkpointed_seconds,
               "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def evaluate(home, checkpoint_path, dataset_name, output):
    started, device = time.perf_counter(), torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = json.loads((checkpoint_path.parent / "completed.json").read_text())
    if (checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG or completed["status"] != "complete"
            or completed["epochs"] != EPOCHS or checkpoint["epoch"] != completed["best_epoch"]
            or checkpoint["epoch"] != checkpoint["best_epoch"]):
        raise ValueError("external orbit evaluation requires source-selected best after all15 complete source epochs")
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
        row.update(dataset=dataset_name, condition="native256_C4_semanticmean", component=source["component"])
        groups[row["group_id"]].append(row)
        components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    direct.save_csv(output / "predictions.csv", rows)
    direct.save_csv(output / "groups.csv", [{"group_id": key, **summarize(value)} for key, value in groups.items()])
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
               "development_scope": "previously viewed external development; fixed training-time orbit, no target training or policy selection",
               "failure_policy": "retain every original input; input or scalar failure receives normalized error1 once",
               "pipeline_change": "same native256 RGB preparation asv16, new four-view semantic-pooled model input; fourfold per-original CNN views"})


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
