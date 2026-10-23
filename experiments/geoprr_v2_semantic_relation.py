"""Source-only semantic-conditioned local relation reading with deployed circular transport."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_semantic_aux as auxiliary
from experiments import geoprr_v2_semantic_direct as direct
from experiments.geoprr_v2_model import _Refine
from experiments.geoprr_v2_reference_relation import CircularProgressPushforward, ReferenceRelationStateHead
from experiments.geoprr_v2_role_field_data import RoleFieldRGBDataset
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_full_semantic_relation_source_only_v1"
SOURCE_RELATIVE = direct.SOURCE_RELATIVE
CONFIG = deepcopy(auxiliary.CONFIG)
CONFIG.update(protocol=PROTOCOL)
CONFIG["model"] = {"encoder": direct.ENCODER_NAME, "num_classes": 0,
                   "semantic_context": "full encoder1280 semantic output into predicted center/SPD context",
                   "readout": "predicted local conditional A/P_rel/B_rel to full J to unchanged CircularProgressPushforward",
                   "bins": 256, "decoder_dim": 64, "token_dim": 16, "reference_normalization": "local",
                   "feature_modules": ["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
                   "feature_channels": [48, 80, 160, 960], "feature_reductions": [4, 8, 16, 32],
                   "point_head": False, "relation_deployed_during_training_and_inference": True}
CONFIG["source"]["require_geometry"] = "analytic U/M/H/span and virtual role labels for source loss only; no GT model input"
CONFIG["training"].pop("auxiliary_reading_cdf")
CONFIG["training"].update(role_normalization="local", reading_cdf=False)
CONFIG["checkpoint_selection"] = "earliest minimum deployed relation NMAE on complete source1506 within60epochs"
CONFIG["external_checkpoint"] = "source-selected best only after all60epochs complete"
CONFIG["inference_input"] = "ROI RGB, linear resize256, ImageNet normalization; full semantic context and predicted local relation circular reading; no GT geometry or target scalar input"


class SemanticRelationReader(nn.Module):
    """Predict dial calibration and relative role evidence from one complete semantic encoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.semantic_dim = int(encoder.conv_head.out_channels)
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 64, 1) for channels in (48, 80, 160, 960))
        self.refine = nn.ModuleList(_Refine(64) for _ in range(4))
        self.state_head = ReferenceRelationStateHead(256, 64, 16, "local")
        self.state_head.context[0] = nn.Linear(self.semantic_dim, 64)
        self.readout = CircularProgressPushforward(256)
        self._features = None
        modules = dict(self.encoder.named_modules())
        for name in CONFIG["model"]["feature_modules"]:
            modules[name].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if self._features is not None:
            self._features.append(output)

    def forward_details(self, image):
        self._features = []
        try:
            semantic = self.encoder(image)
            features = self._features
        finally:
            self._features = None
        if semantic.shape != (len(image), self.semantic_dim):
            raise ValueError("full encoder must return one complete semantic vector per RGB input")
        if [feature.shape[1] for feature in features] != [48, 80, 160, 960]:
            raise ValueError("semantic relation features differ from the four-scale encoder contract")
        projected = [layer(feature) for layer, feature in zip(self.lateral, features, strict=True)]
        decoded = self.refine[-1](projected[-1])
        for index in (2, 1, 0):
            decoded = self.refine[index](F.interpolate(decoded, size=projected[index].shape[-2:],
                                                       mode="bilinear", align_corners=False) + projected[index])
        with torch.autocast(device_type=image.device.type, enabled=False):
            relation = self.state_head(decoded.float(), semantic.float())
            reading = self.readout(relation["J"])
        return {**relation, "joint": relation["J"], **reading,
                "global_features": semantic, "decoded_features": decoded}

    def forward(self, image):
        return self.forward_details(image)["reading"]

    def forward_training(self, image):
        details = self.forward_details(image)
        return details["reading"], details


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(direct.ENCODER_NAME, pretrained=pretrained, num_classes=0)
    selected = [next(item for item in encoder.feature_info if item["module"] == name)
                for name in CONFIG["model"]["feature_modules"]]
    if ([item["num_chs"] for item in selected], [item["reduction"] for item in selected]) != ([48, 80, 160, 960], [4, 8, 16, 32]):
        raise ValueError("complete semantic encoder multi-scale module contract differs")
    model = SemanticRelationReader(encoder)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"]) != (1280, 8434512, 462):
        raise ValueError("cached complete semantic encoder architecture differs")
    return model


def model_metadata(model):
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"class": f"{type(model).__module__}.{type(model).__name__}",
            "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
            "encoder_name": direct.ENCODER_NAME, "semantic_dim": model.semantic_dim,
            "encoder_reported_num_features": model.encoder.num_features,
            "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
            "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
            "encoder_parameters": encoder_parameters, "encoder_state_tensors": len(model.encoder.state_dict()),
            "non_CNN_parameters": parameters - encoder_parameters,
            "parameters": parameters, "state_tensors": len(model.state_dict()),
            "context_input_dim": model.state_head.context[0].in_features,
            "point_head_present": False, "reference_normalization": model.state_head.relation.reference_normalization,
            "feature_modules": CONFIG["model"]["feature_modules"], "feature_channels": [48, 80, 160, 960],
            "feature_reductions": [4, 8, 16, 32], "deployment_uses_geometry_readout": True,
            "geometry_labels_in_model_inputs": False, "training_only_auxiliary_modules": False,
            "deployed_readout": "full conditional J to unchanged CircularProgressPushforward256"}


auxiliary_losses = auxiliary.auxiliary_losses


def source_contract(home, training, validation):
    contract = auxiliary.source_contract(home, training, validation)
    contract.pop("auxiliary_in_model_inputs")
    contract.pop("auxiliary_used_only_for_source_training_loss")
    contract.update(geometry_labels_in_model_inputs=False, geometry_labels_used_only_for_source_training_loss=True,
                    predicted_geometry_and_relation_deployed=True,
                    model_input_contract=CONFIG["inference_input"])
    return contract


def runtime(device, started):
    result = direct.runtime(device, started)
    result["full_process_seconds"] = time.perf_counter() - PROCESS_STARTED
    return result


def execution_counter(model):
    counts, encoder_hook = direct.encoder_counter(model)
    hooks = [encoder_hook]
    for key, module in (("state_head_forward_calls", model.state_head),
                        ("relation_forward_calls", model.state_head.relation), ("readout_forward_calls", model.readout)):
        counts[key] = 0

        def count(_module, _inputs, _output, name=key):
            counts[name] += 1

        hooks.append(module.register_forward_hook(count))
    return counts, hooks


def evaluate_population(model, dataset, device):
    model.eval()
    counts, hooks = execution_counter(model)
    loader = make_loader(dataset, batch_size=64, workers=4, shuffle=False, seed=20262020, device=device)
    rows, forward_seconds = [], 0.
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for raw in loader:
                batch = device_batch(raw, device)
                direct.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    details = model.forward_details(batch["image"])
                direct.synchronize(device)
                forward_seconds += time.perf_counter() - tick
                batch_rows = direct.scalar_rows(details["reading"], batch)
                for key in ("invalid_mass", "outside_arc_mass"):
                    values = details[key].detach().float().cpu().tolist()
                    for row, value in zip(batch_rows, values, strict=True):
                        row[key] = value if row["valid"] and math.isfinite(value) else None
                rows.extend(batch_rows)
    finally:
        for hook in hooks:
            hook.remove()
    expected_calls = math.ceil(len(dataset) / 64)
    if ([row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in dataset.rows]
            or counts["cnn_image_forwards"] != len(dataset)
            or any(counts[key] != expected_calls for key in ("cnn_forward_calls", "state_head_forward_calls",
                                                             "relation_forward_calls", "readout_forward_calls"))):
        raise RuntimeError("relation evaluation lost or repeated RGB, state-head or circular readout execution")
    counts.update(backward_calls=0, optimizer_updates=0,
                  failed_input_placeholder_forwards=sum(not row["input_valid"] for row in rows))
    return rows, {"counts": counts, "pipeline_seconds": time.perf_counter() - started,
                  "forward_seconds": forward_seconds, "batch_size": 64, "workers": 4}


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("source semantic relation training is complete; reuse the completed stage")
    latest = output / "latest.pt"
    if not latest.exists() and any((output / name).exists() for name in ("history.jsonl", "best.pt")):
        raise FileExistsError("partial source outputs exist without latest.pt; inspect the retained failure")
    train_data = RoleFieldRGBDataset(home / SOURCE_RELATIVE / "manifests/train.jsonl", data_root=home / "datasets", augment=True)
    validation = direct.ScalarRGBDataset(home / "runs/geoprr_v2/manifests/validation.jsonl", data_root=home / "datasets", augment=False)
    contract = source_contract(home, train_data, validation)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    model = create_model(pretrained=saved is None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {"train_batches": 0, "train_cnn_image_forwards": 0, "train_cnn_forward_calls": 0,
              "backward_calls": 0, "optimizer_updates": 0, "amp_skipped_updates": 0,
              "relation_forward_calls": 0, "relation_backward_attempts": 0, "readout_forward_calls": 0,
              "validation_images": 0, "validation_cnn_forward_calls": 0,
              "validation_relation_forward_calls": 0, "validation_readout_forward_calls": 0,
              "training_seconds": 0., "validation_seconds": 0.}
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": direct.ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
                      "retained_complete_encoder_state_tensors": 462,
                      "new_non_CNN_parameter_tensors": sum(1 for name, _ in model.named_parameters() if not name.startswith("encoder.")),
                      "sigmoid_point_head_allocated": False, "old_task_checkpoint_used": False,
                      "semantic_context_dimension": 1280, "deployed_readout": "local conditional J to CircularProgressPushforward"}
    if saved is not None:
        if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
            raise ValueError("resumed source semantic relation protocol, config or contracts differ")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch, history, totals = saved["epoch"], saved["best_nmae"], saved["best_epoch"], saved["history"], saved["totals"]
        checkpointed_seconds, latest_rows, initialization = saved["checkpointed_process_seconds"], saved["validation_rows"], saved["initialization"]
    output.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
               "initialization": initialization, "model_metadata": metadata, "train_unique_images": 18590, "validation_images": 1506,
               "epoch_exposure": {"train_images": 18590, "train_batches": 291, "validation_images": 1506, "validation_batches": 24},
               "inference_input": CONFIG["inference_input"], "source_dataset_class": "experiments.geoprr_v2_role_field_data.RoleFieldRGBDataset",
               "comparison": "fresh full ImageNet semantic context with deployed local relation geometry; same source RGB/60epoch CNN exposure asv16, different readout and losses"})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, 61):
        tick = time.perf_counter()
        model.train()
        used_lr = direct.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = used_lr
        loader = make_loader(train_data, batch_size=64, workers=4, shuffle=True, seed=20262020 + epoch, device=device)
        counts, hooks = execution_counter(model)
        batches, updates, skips, error_sum = 0, 0, 0, 0.
        loss_sums = defaultdict(float)
        try:
            for index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction, details = model.forward_training(batch["image"])
                losses = auxiliary_losses(prediction, details, batch)
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    raise FloatingPointError(f"nonfinite semantic relation loss at epoch{epoch} batch{index + 1}")
                scaler.scale(losses["total"]).backward()
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
                for key, value in losses.items():
                    loss_sums[key] += float(value.detach()) * count
                error_sum += float((prediction.detach() - batch["y"]).abs().sum())
                if index == 0:
                    print(json.dumps({"event": "training_batch", "epoch": epoch, "batch": 1, "images": count,
                                      "losses": {key: float(value.detach()) for key, value in losses.items()},
                                      "learning_rate": used_lr, "gradient_norm_finite": bool(torch.isfinite(gradient_norm))}), flush=True)
        finally:
            for hook in hooks:
                hook.remove()
        direct.synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"], counts["relation_forward_calls"],
                counts["readout_forward_calls"]) != (291, 291, 18590, 291, 291):
            raise RuntimeError("source semantic relation training repeated or omitted unique RGB exposure")
        latest_rows, validation_cost = evaluate_population(model, validation, device)
        summary = direct.summarize(latest_rows)
        improved = summary["nmae"] < best
        if improved:
            best, best_epoch = summary["nmae"], epoch
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
                           "train_cnn_forward_calls": counts["cnn_forward_calls"], "backward_calls": batches,
                           "optimizer_updates": updates, "amp_skipped_updates": skips,
                           "relation_forward_calls": counts["relation_forward_calls"], "relation_backward_attempts": batches,
                           "readout_forward_calls": counts["readout_forward_calls"],
                           "validation_images": len(latest_rows), "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "validation_relation_forward_calls": validation_cost["counts"]["relation_forward_calls"],
                           "validation_readout_forward_calls": validation_cost["counts"]["readout_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "source_training_losses": {key: value / 18590 for key, value in loss_sums.items()},
                  "source_training_nmae": error_sum / 18590, "train_counts": counts,
                  "backward_calls": batches, "relation_forward_calls": counts["relation_forward_calls"],
                  "relation_backward_attempts": batches, "readout_forward_calls": counts["readout_forward_calls"],
                  "optimizer_updates": updates, "amp_skipped_updates": skips, "validation": summary,
                  "validation_cost": validation_cost,
                  "validation_relation_forward_calls": validation_cost["counts"]["relation_forward_calls"],
                  "validation_readout_forward_calls": validation_cost["counts"]["readout_forward_calls"],
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
            totals["validation_cnn_forward_calls"], totals["backward_calls"], totals["relation_forward_calls"],
            totals["readout_forward_calls"], totals["validation_relation_forward_calls"],
            totals["validation_readout_forward_calls"]) != (60, 17460, 1115400, 90360, 1440, 17460, 17460, 17460, 1440, 1440):
        raise RuntimeError("completed semantic relation training differs from the declared60epoch exposure")
    if totals["optimizer_updates"] + totals["amp_skipped_updates"] != 17460:
        raise RuntimeError("effective updates and AMP skips differ from actual attempts")
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
        raise ValueError("external evaluation requires source-selected best after all60source epochs")
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = direct.load_population(home / "datasets" / folder, dataset_name)
    dataset = direct.ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    rows, cost = evaluate_population(model, dataset, device)
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256_semantic_local_relation", component=source["component"])
        groups[row["group_id"]].append(row)
        components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    direct.save_csv(output / "predictions.csv", rows)
    direct.save_csv(output / "groups.csv", [{"group_id": key, **direct.summarize(value)} for key, value in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
               "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
               "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
               "checkpoint_source_best_nmae": checkpoint["best_nmae"], "model_metadata": model_metadata(model),
               "summary": direct.summarize(rows), "by_group": {key: direct.summarize(value) for key, value in groups.items()},
               "by_component": {key: direct.summarize(value) for key, value in components.items()},
               "forward_counts": cost["counts"], "timing": cost, "runtime": runtime(device, started),
               "model_inputs": CONFIG["inference_input"], "training_or_adaptation_during_evaluation": False,
               "backward_calls": 0, "optimizer_updates": 0,
               "relation_forward_calls": cost["counts"]["relation_forward_calls"],
               "readout_forward_calls": cost["counts"]["readout_forward_calls"],
               "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
               "development_scope": "previously viewed external development; no target training, BN adaptation or checkpoint selection",
               "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
               "pipeline_change": "same native256 RGB preparation asv16; predicted semantic-conditioned geometry and local relation execute during deployment"})


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
