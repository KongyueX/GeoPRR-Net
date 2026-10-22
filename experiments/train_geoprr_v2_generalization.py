"""Fine-tune a frozen native reader with source-only generalization interventions."""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import torch
from torch import nn

from experiments.geoprr_v2_generalization_style import (
    apply_generalization_training_policy, install_mixstyle_hooks,
)
from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationDataset, ReferenceRelationReader, reference_relation_losses,
)
from experiments.geoprr_v2_role_triangle import RoleTriangleReader, role_triangle_losses
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint,
    seed_everything, source_split_check,
)
from experiments.train_geoprr_v2_reference_relation import (
    build_parser, save_predictions, validation_rows, validation_summary, write_history,
)


PROTOCOL = "geoprr_v2_generalization_source_only_v1"
ROI_PROTOCOL = "geoprr_v2_roi_generalization_source_only_v1"
ENDPOINT_PROTOCOL = "geoprr_v2_endpoint_generalization_source_only_v1"
WEIGHT_FROZEN_PROTOCOL = "geoprr_v2_endpoint_backbone_weight_freeze_source_only_v1"
REFERENCE_PROTOCOLS = (
    "geoprr_v2_reference_relation_source_only_v1",
    "geoprr_v2_reference_feedback_source_only_v1",
)
TRIANGLE_PROTOCOL = "geoprr_v2_role_triangle_source_only_v1"


def reader_and_loss(config):
    if config["protocol"] not in (PROTOCOL, ROI_PROTOCOL, ENDPOINT_PROTOCOL, WEIGHT_FROZEN_PROTOCOL):
        raise ValueError("requires the frozen generalization training protocol")
    required = {"seed": 20262020, "epochs": 4, "warmup_epochs": 0, "batch_size": 32,
                "workers": 8, "validation_batch_size": 16, "backbone_lr": 1e-5,
                "head_lr": 5e-5, "min_lr": 1e-6}
    if any(config["training"][key] != value for key, value in required.items()):
        raise ValueError("training differs from the frozen four-epoch generalization recipe")
    policy = config["generalization"]
    if (not isinstance(policy["freeze_bn_statistics"], bool)
            or not isinstance(policy["mixstyle_module_names"], list)
            or policy["mixstyle_p"] != .5 or policy["mixstyle_alpha"] != .1):
        raise ValueError("generalization policy differs from the frozen interventions")
    if config["protocol"] == ROI_PROTOCOL and policy.get("roi_padding") != {"probability": .8, "max_padding": .3}:
        raise ValueError("ROI padding differs from the frozen source-view intervention")
    if config["protocol"] in (ENDPOINT_PROTOCOL, WEIGHT_FROZEN_PROTOCOL):
        from experiments.geoprr_v2_endpoint_source import REPLACEMENT
        if policy.get("source_replacement") != REPLACEMENT or policy["mixstyle_module_names"]:
            raise ValueError("endpoint source replacement differs from the frozen data intervention")
    if config["protocol"] == WEIGHT_FROZEN_PROTOCOL:
        if (policy.get("freeze_backbone_weights") is not True
                or policy["freeze_bn_statistics"] is not False
                or config["reader_protocol"] not in REFERENCE_PROTOCOLS):
            raise ValueError("requires frozen backbone weights, updating source BN statistics and a native reference reader")
    elif policy.get("freeze_backbone_weights", False):
        raise ValueError("backbone weight freezing requires its independent training protocol")
    reader_protocol = config["reader_protocol"]
    if reader_protocol in REFERENCE_PROTOCOLS:
        if reader_protocol == REFERENCE_PROTOCOLS[1] and "reference_normalization" not in config["model"]:
            raise ValueError("feedback readers must declare their reference normalization")
        return ReferenceRelationReader(**config["model"]), reference_relation_losses
    if reader_protocol == TRIANGLE_PROTOCOL:
        return RoleTriangleReader(**config["model"]), role_triangle_losses
    raise ValueError(f"unsupported native reader protocol: {reader_protocol}")


def initialize_from_anchor(model, path, config, data_contract):
    source = torch.load(path, map_location="cpu", weights_only=False)
    if (source["protocol"] != config["reader_protocol"]
            or source["config"].get("protocol", source["protocol"]) != source["protocol"]
            or source["config"]["model"] != config["model"]
            or source["config"]["data"] != config["data"]
            or source["config"]["training"]["seed"] != 20262020):
        raise ValueError("anchor protocol, architecture, source data configuration or seed differs")
    source_replacement = None
    if config["protocol"] in (ENDPOINT_PROTOCOL, WEIGHT_FROZEN_PROTOCOL):
        from experiments.geoprr_v2_endpoint_source import verify_endpoint_training_contract
        source_replacement = verify_endpoint_training_contract(data_contract, source["data_contract"])
    else:
        for key in ("train", "validation", "exposure_edges"):
            if source["data_contract"][key] != data_contract[key]:
                raise ValueError("anchor source supervision or exposure differs")
        for key in ("train_manifest", "val_manifest", "pair_manifest"):
            if source["data_contract"]["paths"][key] != data_contract["paths"][key]:
                raise ValueError("anchor source manifest path differs")
    model.load_state_dict(source["model"], strict=True)
    loaded = model.state_dict()
    if any(loaded[key].dtype != value.dtype or not torch.equal(loaded[key], value)
           for key, value in source["model"].items()):
        raise ValueError("full anchor loading changed a trained tensor")
    initialization = {"source_checkpoint": str(path.resolve()), "source_protocol": source["protocol"],
            "source_epoch": int(source["epoch"]), "loaded_full_tensors": len(loaded),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "strict_state_load": True, "preserved_all_trained_weights": True,
            "optimizer": "fresh AdamW; anchor optimizer, scheduler and scaler are not continued"}
    if source_replacement is not None:
        initialization["source_replacement"] = source_replacement
    return initialization


def backbone_bn_buffers(model):
    return {f"{name}.{key}": value.detach().clone()
            for name, module in model.backbone.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for key, value in module.named_buffers(recurse=False)
            if key in ("running_mean", "running_var", "num_batches_tracked")}


def main(argv=None):
    process_started = time.perf_counter()
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    protocol = config["protocol"]
    training, policy = config["training"], config["generalization"]
    seed = int(training["seed"])
    seed_everything(seed)
    model, loss_function = reader_and_loss(config)
    reader_protocol = config["reader_protocol"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if args.resume is not None and args.resume.resolve() != (args.output / "latest.pt").resolve():
        raise ValueError("resume requires this output directory's latest.pt")
    if args.resume is None and any((args.output / name).exists() for name in ("latest.pt", "history.jsonl", "completed.json")):
        raise FileExistsError("training artifacts already exist; use --resume")
    dataset_options = dict(data_root=args.data_root, data_config=config["data"],
                           augment=True, pair_manifest=args.pair_manifest)
    if protocol == ROI_PROTOCOL:
        from experiments.geoprr_v2_roi_generalization_data import ROIPaddingReferenceRelationDataset
        train_data = ROIPaddingReferenceRelationDataset(args.train_manifest, **dataset_options,
                                                        roi_policy=policy["roi_padding"])
    else:
        train_data = ReferenceRelationDataset(args.train_manifest, **dataset_options)
    val_data = ReferenceRelationDataset(args.val_manifest, data_root=args.data_root,
                                       data_config=config["data"], augment=False)
    if (len(train_data.rows), len(train_data), len(train_data.edges), len(val_data)) != (18590, 22686, 8192, 1506):
        raise ValueError("source populations or exposure differ from the frozen protocol")
    source_split_check(train_data.rows, val_data.rows)
    paths = {key: str(getattr(args, key).resolve()) for key in
             ("train_manifest", "val_manifest", "pair_manifest", "init_checkpoint")}
    data_contract = {"paths": paths, "train": train_data.supervision_contract(),
                     "validation": val_data.supervision_contract(), "exposure_edges": train_data.edges}
    if protocol == ROI_PROTOCOL:
        data_contract["training_input_intervention"] = policy["roi_padding"]
    if protocol in (ENDPOINT_PROTOCOL, WEIGHT_FROZEN_PROTOCOL):
        data_contract["training_source_intervention"] = policy["source_replacement"]
    initialization = None
    if args.resume is None:
        initialization = initialize_from_anchor(model, args.init_checkpoint, config, data_contract)
    model = model.to(device)
    training_parameter_counts = apply_generalization_training_policy(model, policy)
    shared_parameters = [parameter for parameter in model.shared_parameters() if parameter.requires_grad]
    head_parameters = list(model.state_head.parameters())
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
    history, checkpointed_process_seconds = [], 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint["protocol"] != protocol or checkpoint["reader_protocol"] != reader_protocol
                or checkpoint["config"] != config or checkpoint["data_contract"] != data_contract):
            raise ValueError("resume protocol/config/source supervision or exposure differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_nmae, best_epoch = checkpoint["epoch"], checkpoint["best_nmae"], checkpoint["best_epoch"]
        totals, history = checkpoint["totals"], checkpoint["history"]
        checkpointed_process_seconds = checkpoint["checkpointed_process_wall_seconds"]
        initialization = checkpoint["initialization"]
        restore_rng(checkpoint["rng"])
        training_parameter_counts = apply_generalization_training_policy(model, policy)
    args.output.mkdir(parents=True, exist_ok=True)
    run = {"protocol": protocol, "reader_protocol": reader_protocol, "config": config,
           "data_contract": data_contract, "generalization": policy, "initialization": initialization,
           "train_unique_images": len(train_data.rows), "train_items": len(train_data),
           "expected_full_epoch": {"donor_images": 22686, "recipient_images": 8192,
                                   "total_image_forwards": 30878, "optimizer_batches": 709},
           "validation_images": len(val_data), "model_selection": "full1506_source_nmae_new_epochs_only_earliest_tie",
           "initial_checkpoint_evaluation": "performed separately; excluded from new-epoch best.pt selection",
           "parameters": sum(parameter.numel() for parameter in model.parameters()),
           "state_tensors": len(model.state_dict()),
           "shared_parameters": sum(parameter.numel() for parameter in shared_parameters),
           "head_parameters": sum(parameter.numel() for parameter in head_parameters),
           "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "inference_input": "RGB only; H/span/y and virtual role phases are training/scoring labels",
           "recipient_batching": "donors and recipients concatenated in one forward",
           "comparison": config.get("control"), "external_labels_used_for_training": False,
           "within_run_checkpoint_selection_uses_external_data": False,
           "anchor_selection_used_external_development": True}
    if protocol == WEIGHT_FROZEN_PROTOCOL:
        run.update(training_parameter_counts)
    write_json(args.output / "config.json", config)
    write_json(args.output / "run.json", run)
    write_history(args.output / "history.jsonl", history)
    handles = (install_mixstyle_hooks(model, policy["mixstyle_module_names"],
                                     p=policy["mixstyle_p"], alpha=policy["mixstyle_alpha"])
               if policy["mixstyle_module_names"] else ())
    try:
        for epoch in range(start_epoch, training["epochs"]):
            epoch_started = time.perf_counter()
            apply_generalization_training_policy(model, policy)
            weight_frozen = protocol == WEIGHT_FROZEN_PROTOCOL
            backbone_weights_before = ({name: parameter.detach().clone()
                                       for name, parameter in model.backbone.named_parameters()}
                                      if weight_frozen else {})
            bn_before = backbone_bn_buffers(model) if policy["freeze_bn_statistics"] or weight_frozen else {}
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
                terms = loss_function(output, supervision)
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
                                      "stage": "generalization_finetune", "loss": float(loss.detach()),
                                      "donor_images": donors, "recipient_images": count - donors}), flush=True)
            if (donor_seen, recipient_seen, batches) != (22686, 8192, 709):
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
            bn_after = backbone_bn_buffers(model) if policy["freeze_bn_statistics"] or weight_frozen else {}
            if policy["freeze_bn_statistics"] and (bn_before.keys() != bn_after.keys()
                    or any(not torch.equal(value, bn_after[key]) for key, value in bn_before.items())):
                raise RuntimeError("frozen backbone BatchNorm running statistics changed")
            if weight_frozen:
                if any(not torch.equal(backbone_weights_before[name], parameter.detach())
                       for name, parameter in model.backbone.named_parameters()):
                    raise RuntimeError("frozen backbone parameters changed")
                bn_counters = [key for key in bn_before if key.endswith(".num_batches_tracked")]
                bn_running_changed = sum(not torch.equal(value, bn_after[key])
                                         for key, value in bn_before.items() if key not in bn_counters)
                if (not bn_counters or not bn_running_changed
                        or any(not torch.equal(bn_after[key], bn_before[key] + batches) for key in bn_counters)):
                    raise RuntimeError("source BatchNorm running statistics did not update with the training batches")
            summary = validation_summary(rows)
            improved = summary["nmae"] < best_nmae
            if improved:
                best_nmae, best_epoch = summary["nmae"], epoch + 1
            scheduler.step()
            for key, value in {"train_batches": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
                               "donor_images": donor_seen, "recipient_images": recipient_seen, "validation_images": len(rows),
                               "training_wall_seconds": training_seconds, "validation_wall_seconds": validation_seconds}.items():
                totals[key] += value
            record = {"epoch": epoch + 1, "stage": "generalization_finetune", "train_batches": batches,
                      "donor_images": donor_seen, "recipient_images": recipient_seen,
                      "total_image_forwards": donor_seen + recipient_seen, "optimizer_updates": updates,
                      "amp_skipped_updates": skips, "global_step": totals["optimizer_updates"],
                      "losses": {key: value / (donor_seen + recipient_seen) for key, value in sums.items()},
                      "validation": summary, "best_nmae": best_nmae, "best_epoch": best_epoch,
                      "frozen_bn_buffers_checked": len(bn_before) if policy["freeze_bn_statistics"] else 0,
                      "frozen_bn_statistics_unchanged": True if policy["freeze_bn_statistics"] else None,
                      "training_seconds": training_seconds, "validation_seconds": validation_seconds,
                      "elapsed_seconds": time.perf_counter() - epoch_started,
                      "learning_rates": [group["lr"] for group in optimizer.param_groups],
                      "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
            if weight_frozen:
                record.update(frozen_backbone_parameter_tensors_checked=len(backbone_weights_before),
                              frozen_backbone_parameters_unchanged=True,
                              source_bn_buffers_checked=len(bn_before),
                              source_bn_running_buffers_changed=bn_running_changed,
                              source_bn_counters_checked=len(bn_counters),
                              source_bn_batches_tracked_increment=batches,
                              source_bn_counters_advanced=True)
            history.append(record)
            checkpoint = {"protocol": protocol, "reader_protocol": reader_protocol, "config": config,
                          "data_contract": data_contract, "initialization": initialization,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
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
            save_predictions(args.output / "latest_predictions.csv.gz", rows)
            write_history(args.output / "history.jsonl", history)
            print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    write_json(args.output / "completed.json", {"protocol": protocol, "reader_protocol": reader_protocol,
               "epochs": training["epochs"], "best_epoch": best_epoch, "best_nmae": best_nmae, "totals": totals,
               "current_process_wall_seconds": time.perf_counter() - process_started,
               "checkpointed_process_wall_seconds_before_resume": checkpointed_process_seconds,
               "uncheckpointed_work_after_interruptions": "not included in checkpointed counters; use controller process logs",
               "seed": seed, "seed_expansion": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
