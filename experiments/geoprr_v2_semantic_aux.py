"""Source-only role/field supervision of a complete semantic scalar reader."""
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
from experiments.geoprr_v2_model import _Refine
from experiments.geoprr_v2_reference_relation import ReferenceRelationStateHead, reference_geometry_targets
from experiments.geoprr_v2_role_field_data import DATA_CONFIG, GEOMETRY_CONFIG, RoleFieldRGBDataset
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_semantic_role_field_aux_source_only_v1"
SOURCE_RELATIVE = direct.SOURCE_RELATIVE
CONFIG = deepcopy(direct.CONFIG)
CONFIG.update(protocol=PROTOCOL)
CONFIG["model"].update(auxiliary_decoder="four-scale64-channel FPN and local ReferenceRelationStateHead256/64/16; training only",
                       feature_modules=["blocks.0", "blocks.1", "blocks.2", "blocks.4"],
                       feature_channels=[48, 80, 160, 960], feature_reductions=[4, 8, 16, 32])
CONFIG["source"].update(require_geometry="analytic U/M/H and role source supervision only; no Gaussian Q or GT readout")
CONFIG["source"].update(analytic_label_contract="analytic_U_M_role_no_Gaussian_Q_v1",
                        analytic_data=deepcopy(DATA_CONFIG), analytic_geometry_config=deepcopy(vars(GEOMETRY_CONFIG)))
CONFIG["training"].update(loss="SmoothL1(beta=.05)+.1*(role_NLL/(3log256)+field_L1)", auxiliary_weight=.1,
                          role_normalization="local", auxiliary_reading_cdf=False)
CONFIG["inference_input"] = "ROI RGB, linear resize256, ImageNet normalization; direct semantic scalar only; no auxiliary decoder or GT geometry input"


class SemanticAuxReader(nn.Module):
    """Share one semantic encoder with a training-only spatial supervision branch."""

    def __init__(self, base):
        super().__init__()
        self.encoder, self.point, self.semantic_dim = base.encoder, base.point, base.semantic_dim
        self.lateral = nn.ModuleList(nn.Conv2d(channels, 64, 1) for channels in (48, 80, 160, 960))
        self.refine = nn.ModuleList(_Refine(64) for _ in range(4))
        self.state_head = ReferenceRelationStateHead(256, 64, 16, "local")
        self._features = None
        modules = dict(self.encoder.named_modules())
        for name in CONFIG["model"]["feature_modules"]:
            modules[name].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if self._features is not None:
            self._features.append(output)

    forward = direct.SemanticDirectReader.forward

    def forward_training(self, image):
        self._features = []
        try:
            prediction = self(image)
            features = self._features
        finally:
            self._features = None
        if [feature.shape[1] for feature in features] != [48, 80, 160, 960]:
            raise ValueError("source auxiliary features differ from the frozen four-scale encoder contract")
        projected = [layer(feature) for layer, feature in zip(self.lateral, features, strict=True)]
        context = torch.cat([feature.mean((2, 3)) for feature in projected[1:]], 1)
        decoded = self.refine[-1](projected[-1])
        for index in (2, 1, 0):
            decoded = self.refine[index](F.interpolate(decoded, size=projected[index].shape[-2:],
                                                       mode="bilinear", align_corners=False) + projected[index])
        with torch.autocast(device_type=image.device.type, enabled=False):
            auxiliary = self.state_head(decoded, context)
        return prediction, auxiliary


def create_model(*, pretrained):
    base = direct.create_model(pretrained=pretrained)
    features = base.encoder.feature_info
    selected = [next(item for item in features if item["module"] == name) for name in CONFIG["model"]["feature_modules"]]
    if ([item["num_chs"] for item in selected], [item["reduction"] for item in selected]) != ([48, 80, 160, 960], [4, 8, 16, 32]):
        raise ValueError("complete semantic encoder multi-scale module contract differs")
    model = SemanticAuxReader(base)
    metadata = model_metadata(model)
    if (metadata["parameters"], metadata["state_tensors"], metadata["auxiliary_parameters"]) != (8693701, 498, 257908):
        raise ValueError("source auxiliary model parameters or state tensors differ from the frozen architecture")
    return model


def model_metadata(model):
    metadata = direct.model_metadata(model)
    metadata.update(active_inference_parameters=8435793, active_inference_state_tensors=464,
                    auxiliary_parameters=sum(parameter.numel() for module in (model.lateral, model.refine, model.state_head)
                                             for parameter in module.parameters()),
                    auxiliary_only_during_training=True, direct_readout="complete semantic1280 to linear1 sigmoid",
                    auxiliary_feature_modules=CONFIG["model"]["feature_modules"], auxiliary_feature_channels=[48, 80, 160, 960],
                    auxiliary_feature_reductions=[4, 8, 16, 32], auxiliary_reference_normalization="local",
                    deployment_uses_geometry_readout=False)
    return metadata


def auxiliary_losses(prediction, output, batch):
    """Use the original conditional role and analytic field losses without CDF."""
    targets = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"],
                                          image_size=batch["image"].shape[-1])
    labels = targets["role_twohot"].float()
    role_nll = -(labels[:, 0] * output["log_A"]).sum(-1)
    bins = labels.shape[-1]
    phases = targets["phases"][:, 0] * bins / (2 * math.pi)
    low = phases.floor().long() % bins
    fraction = (phases - phases.floor()).float()
    batch_index, absolute = torch.arange(len(low), device=low.device), torch.arange(bins, device=low.device)
    for reference, weight in ((low, 1 - fraction), ((low + 1) % bins, fraction)):
        relative = (absolute[None] - reference[:, None]) % bins
        for key, role in (("log_P_rel", 1), ("log_B_rel", 2)):
            role_nll -= weight * (labels[:, role] * output[key][batch_index, reference].gather(1, relative)).sum(-1)
    raw = role_nll.mean()
    height, width = batch["field_target"].shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(height, device=low.device) + .5) / height,
                            (torch.arange(width, device=low.device) + .5) / width, indexing="ij")
    xy = torch.stack((xx, yy), -1)[None] - output["calibration"]["center"][:, None, None]
    vectors = torch.einsum("bij,bhwj->bhwi", output["calibration"]["spd"], xy)
    phase = (torch.atan2(vectors[..., 1], vectors[..., 0]) + targets["phi"].float()[:, None, None]).remainder(2 * math.pi)
    progress = (phase / targets["span"].float()[:, None, None]).clamp(0, 1)
    mask = batch["field_mask"][:, 0].float()
    field = ((progress - batch["field_target"].float()).abs() * mask).sum() / mask.sum().clamp_min(1)
    normalized = raw / (3 * math.log(bins))
    primary = F.smooth_l1_loss(prediction, batch["y"], beta=.05)
    auxiliary = normalized + field
    return {"primary": primary, "role_nll_raw": raw, "role_nll_normalized": normalized, "field_l1": field,
            "auxiliary": auxiliary, "total": primary + .1 * auxiliary}


def source_contract(home, training, validation):
    fields = ("homography", "homography_bbox", "dial_bbox", "long_interval_degree", "long_num", "pointer_angle",
              "image_width", "image_height", "scene_name")
    rows = []
    for row in training.rows:
        metadata = row.get("metadata", {})
        geometry = {key: metadata[key] for key in fields if key in metadata}
        geometry["ScaleMark_keypoints"] = [{"type": item.get("type"), "all_kp": item.get("all_kp")}
            for item in metadata.get("keypoints", []) if str(item.get("type", "")).lower() == "scalemark"]
        rows.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                     "target": direct.normalized_target(row), "geometry": geometry})
    return {"rgb": direct.source_contract(home, training, validation),
            "auxiliary": deepcopy({"schema": "source_analytic_role_field_supervision_256_64_v1", "rows": rows,
                "data": DATA_CONFIG, "geometry_config": vars(GEOMETRY_CONFIG),
                "labels": ["roi_to_canonical", "span_radians", "field_target_U", "field_mask_M", "analytic_virtual_A_P_B_roles"],
                "Gaussian_Q_or_GT_readout_constructed": False}),
            "auxiliary_in_model_inputs": False, "auxiliary_used_only_for_source_training_loss": True}


def runtime(device, started):
    result = direct.runtime(device, started)
    result["full_process_seconds"] = time.perf_counter() - PROCESS_STARTED
    return result


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("source auxiliary training is complete; reuse the completed stage")
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
              "auxiliary_forward_calls": 0, "auxiliary_backward_attempts": 0,
              "validation_images": 0, "validation_cnn_forward_calls": 0,
              "training_seconds": 0., "validation_seconds": 0.}
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": direct.ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
                      "retained_complete_encoder_state_tensors": 462, "new_point_parameter_tensors": 2,
                      "base_and_point_created_before_auxiliary_modules": True, "old_task_checkpoint_used": False}
    if saved is not None:
        if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
            raise ValueError("resumed source auxiliary protocol, config or contracts differ")
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
               "comparison": "fresh cached complete ImageNet semantic representation; source auxiliary U/M/H/role loss, direct-only evaluation"})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, 61):
        tick = time.perf_counter()
        model.train()
        used_lr = direct.learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = used_lr
        loader = make_loader(train_data, batch_size=64, workers=4, shuffle=True, seed=20262020 + epoch, device=device)
        counts, hook = direct.encoder_counter(model)
        batches, updates, skips, error_sum = 0, 0, 0, 0.
        loss_sums = defaultdict(float)
        try:
            for index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction, auxiliary = model.forward_training(batch["image"])
                losses = auxiliary_losses(prediction, auxiliary, batch)
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    raise FloatingPointError(f"nonfinite auxiliary scalar loss at epoch{epoch} batch{index + 1}")
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
            hook.remove()
        direct.synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (291, 291, 18590):
            raise RuntimeError("source auxiliary training repeated or omitted unique RGB exposure")
        latest_rows, validation_cost = direct.evaluate_population(model, validation, device)
        summary = direct.summarize(latest_rows)
        improved = summary["nmae"] < best
        if improved:
            best, best_epoch = summary["nmae"], epoch
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
                           "train_cnn_forward_calls": counts["cnn_forward_calls"], "backward_calls": batches,
                           "optimizer_updates": updates, "amp_skipped_updates": skips,
                           "auxiliary_forward_calls": batches, "auxiliary_backward_attempts": batches,
                           "validation_images": len(latest_rows), "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "source_training_losses": {key: value / 18590 for key, value in loss_sums.items()},
                  "source_training_nmae": error_sum / 18590, "train_counts": counts,
                  "backward_calls": batches, "auxiliary_forward_calls": batches, "auxiliary_backward_attempts": batches,
                  "optimizer_updates": updates, "amp_skipped_updates": skips, "validation": summary,
                  "validation_cost": validation_cost, "validation_auxiliary_forward_calls": 0,
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
            totals["validation_cnn_forward_calls"], totals["backward_calls"], totals["auxiliary_forward_calls"]) != (60, 17460, 1115400, 90360, 1440, 17460, 17460):
        raise RuntimeError("completed auxiliary training differs from the declared60epoch exposure")
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
    rows, cost = direct.evaluate_population(model, dataset, device)
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256_direct_only", component=source["component"])
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
               "backward_calls": 0, "optimizer_updates": 0, "auxiliary_forward_calls": 0,
               "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
               "development_scope": "previously viewed external development; no target training, BN adaptation or checkpoint selection",
               "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
               "pipeline_change": "same native256 direct evaluation asv16; geometry branch only supplies source training losses"})


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
