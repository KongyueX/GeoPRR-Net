"""Source-only full-semantic conditioning of deployed local role content scores."""
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
from experiments import geoprr_v2_semantic_relation as base
from experiments.geoprr_v2_role_field_data import RoleFieldRGBDataset
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_semantic_role_conditioned_source_only_v1"
SOURCE_RELATIVE = base.SOURCE_RELATIVE
CONFIG = deepcopy(base.CONFIG)
CONFIG.update(protocol=PROTOCOL)
CONFIG["model"].update(
    semantic_role_conditioning="three semantic queries add dot(query, projected role token)/sqrt16 to corresponding role unary",
    semantic_queries={"input_dim": 1280, "roles": 3, "token_dim": 16, "bias": False, "initialization": "zero weight after complete fresh v20 model construction"},
    preserved_base_initial_state_tensors=502, new_parameter_tensors=1)
CONFIG["inference_input"] = "ROI RGB, linear resize256, ImageNet normalization; semantic-conditioned calibration and local role energies, full conditional J and circular reading; no GT geometry or target scalar input"
CONFIG["checkpoint_selection"] = "earliest minimum deployed semantic-role relation NMAE on complete source1506 within60epochs"


class SemanticConditionedRelationHead(nn.Module):
    """Keep local conditional matching while adding semantic role-content queries."""

    def __init__(self, relation, semantic_dim):
        super().__init__()
        self.reference_normalization = relation.reference_normalization
        self.projections, self.unaries = relation.projections, relation.unaries
        self.scale = relation.scale
        self.register_buffer("relative_indices", relation.relative_indices)
        self.token_dim = self.projections[0].out_features
        self.semantic_queries = nn.Linear(semantic_dim, 3 * self.token_dim, bias=False)
        nn.init.zeros_(self.semantic_queries.weight)

    def forward(self, features, semantic):
        roles = [projection(features) for projection in self.projections]
        queries = self.semantic_queries(semantic.float()).reshape(len(features), 3, self.token_dim)
        terms = [(role * queries[:, index, None]).sum(-1) * self.scale for index, role in enumerate(roles)]
        unary = [layer(role).squeeze(-1) + term for layer, role, term in zip(self.unaries, roles, terms, strict=True)]
        a, p, b = roles
        energy_p = (a @ p.transpose(1, 2)) * self.scale + unary[1][:, None]
        energy_b = (a @ b.transpose(1, 2)) * self.scale + unary[2][:, None]
        log_z_p, log_z_b = energy_p.logsumexp(-1), energy_b.logsumexp(-1)
        reference = unary[0] + log_z_p + log_z_b if self.reference_normalization == "global" else unary[0]
        log_a = reference.log_softmax(-1)
        indices = self.relative_indices[None].expand(len(features), -1, -1)
        log_p = energy_p.log_softmax(-1).gather(-1, indices)
        log_b = energy_b.log_softmax(-1).gather(-1, indices)
        start, pointer, end = log_a.exp(), log_p.exp(), log_b.exp()
        joint = pointer.transpose(1, 2) @ (start[:, :, None] * end)
        delta, span = (start[:, :, None] * pointer).sum(1), (start[:, :, None] * end).sum(1)
        return {"A": start, "P_rel": pointer, "B_rel": end, "log_A": log_a, "log_P_rel": log_p, "log_B_rel": log_b,
                "J": joint, "product": delta[:, :, None] * span[:, None], "delta_marginal": delta, "span_marginal": span,
                "reference_unary": unary[0], "log_Z_P": log_z_p, "log_Z_B": log_z_b,
                "semantic_role_queries": queries, "semantic_unary_terms": torch.stack(terms, 1),
                "role_unaries": torch.stack(unary, 1)}


class SemanticRoleStateHead(nn.Module):
    """Reuse calibration and ray sampling, then pass semantic context to role matching."""

    def __init__(self, state_head, semantic_dim):
        super().__init__()
        self.context = state_head.context
        self.radial_embedding = state_head.radial_embedding
        self.relation = SemanticConditionedRelationHead(state_head.relation, semantic_dim)
        self.register_buffer("directions", state_head.directions)
        self.register_buffer("radii", state_head.radii)

    def forward(self, features, context):
        raw = self.context(context.float())
        center, stretch, shear = raw[:, :2].sigmoid(), raw[:, 2], raw[:, 3]
        zero = torch.zeros_like(stretch)
        chol = torch.stack((stretch.exp(), zero, shear, (-stretch).exp()), -1).reshape(-1, 2, 2)
        spd = chol @ chol.transpose(1, 2)
        rays = torch.linalg.solve(spd, self.directions.T.expand(len(raw), -1, -1)).transpose(1, 2)
        rays = F.normalize(rays, dim=-1)
        divisor = torch.where(rays != 0, rays, torch.ones_like(rays))
        distances = torch.where(rays > 0, (1 - center[:, None]) / divisor, -center[:, None] / divisor)
        distances = torch.where(rays != 0, distances, torch.full_like(distances, float("inf")))
        points = center[:, None, None] + self.radii[None, None, :, None] * distances.amin(-1)[:, :, None, None] * rays[:, :, None]
        sampled = F.grid_sample(features.float(), 2 * points - 1, mode="bilinear", padding_mode="border", align_corners=False)
        tokens = torch.einsum("bckr,rcd->bkd", sampled, self.radial_embedding) / 16
        return {**self.relation(tokens, context.float()), "role_tokens": tokens, "query_points": points,
                "calibration": {"center": center, "spd": spd, "raw": raw}}


def condition_model(model):
    model.state_head = SemanticRoleStateHead(model.state_head, model.semantic_dim)
    return model


def create_model(*, pretrained):
    return condition_model(base.create_model(pretrained=pretrained))


def model_metadata(model):
    metadata = base.model_metadata(model)
    queries = model.state_head.relation.semantic_queries
    metadata.update(semantic_role_conditioning=True, semantic_query_input_dim=queries.in_features,
                    semantic_query_roles=3, semantic_query_token_dim=model.state_head.relation.token_dim,
                    semantic_query_parameters=sum(parameter.numel() for parameter in queries.parameters()),
                    semantic_query_parameter_tensors=sum(1 for _ in queries.parameters()),
                    semantic_query_initialization="zero after all502 fresh v20 states are constructed; trained values may differ",
                    preserved_base_initial_state_tensors=502,
                    semantic_role_head_class=f"{type(model.state_head).__module__}.{type(model.state_head).__name__}")
    return metadata


auxiliary_losses = base.auxiliary_losses


def source_contract(home, training, validation):
    contract = base.source_contract(home, training, validation)
    contract["model_input_contract"] = CONFIG["inference_input"]
    contract["semantic_role_energy_conditioning_deployed"] = True
    return contract


def runtime(device, started):
    result = direct.runtime(device, started)
    result["full_process_seconds"] = time.perf_counter() - PROCESS_STARTED
    return result


def execution_counter(model):
    counts, encoder_hook = direct.encoder_counter(model)
    hooks = [encoder_hook]
    for key, module in (("semantic_query_forward_calls", model.state_head.relation.semantic_queries),
                        ("state_head_forward_calls", model.state_head),
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
                                                             "relation_forward_calls", "readout_forward_calls", "semantic_query_forward_calls"))):
        raise RuntimeError("relation evaluation lost or repeated RGB, state-head or circular readout execution")
    counts.update(backward_calls=0, optimizer_updates=0,
                  failed_input_placeholder_forwards=sum(not row["input_valid"] for row in rows))
    return rows, {"counts": counts, "pipeline_seconds": time.perf_counter() - started,
                  "forward_seconds": forward_seconds, "batch_size": 64, "workers": 4}


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("source semantic-role relation training is complete; reuse the completed stage")
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
              "semantic_query_forward_calls": 0, "semantic_query_backward_attempts": 0,
              "validation_images": 0, "validation_cnn_forward_calls": 0,
              "validation_relation_forward_calls": 0, "validation_readout_forward_calls": 0,
              "validation_semantic_query_forward_calls": 0,
              "training_seconds": 0., "validation_seconds": 0.}
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = {"encoder": direct.ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
                      "retained_complete_encoder_state_tensors": 462,
                      "new_non_CNN_parameter_tensors": sum(1 for name, _ in model.named_parameters() if not name.startswith("encoder.")),
                      "sigmoid_point_head_allocated": False, "old_task_checkpoint_used": False,
                      "semantic_context_dimension": 1280, "deployed_readout": "semantic-conditioned local conditional J to CircularProgressPushforward",
                      "fresh_v20_base_constructed_before_new_query": True,
                      "preserved_base_initial_state_tensors": 502, "new_query_parameter_tensors": 1,
                      "new_query_initialization": "zero weight", "trained_task_checkpoint_used": False}
    if saved is not None:
        if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
            raise ValueError("resumed source semantic-role relation protocol, config or contracts differ")
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
               "comparison": "fresh full v20 ImageNet semantic-relation model plus zero-initialized semantic role queries; same source RGB and60epoch budget, no trained v20 weights"})
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
                counts["readout_forward_calls"], counts["semantic_query_forward_calls"]) != (291, 291, 18590, 291, 291, 291):
            raise RuntimeError("source semantic-role relation training repeated or omitted unique RGB exposure")
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
                           "semantic_query_forward_calls": counts["semantic_query_forward_calls"], "semantic_query_backward_attempts": batches,
                           "validation_images": len(latest_rows), "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "validation_relation_forward_calls": validation_cost["counts"]["relation_forward_calls"],
                           "validation_readout_forward_calls": validation_cost["counts"]["readout_forward_calls"],
                           "validation_semantic_query_forward_calls": validation_cost["counts"]["semantic_query_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "source_training_losses": {key: value / 18590 for key, value in loss_sums.items()},
                  "source_training_nmae": error_sum / 18590, "train_counts": counts,
                  "backward_calls": batches, "relation_forward_calls": counts["relation_forward_calls"],
                  "relation_backward_attempts": batches, "readout_forward_calls": counts["readout_forward_calls"],
                  "semantic_query_forward_calls": counts["semantic_query_forward_calls"], "semantic_query_backward_attempts": batches,
                  "optimizer_updates": updates, "amp_skipped_updates": skips, "validation": summary,
                  "validation_cost": validation_cost,
                  "validation_relation_forward_calls": validation_cost["counts"]["relation_forward_calls"],
                  "validation_readout_forward_calls": validation_cost["counts"]["readout_forward_calls"],
                  "validation_semantic_query_forward_calls": validation_cost["counts"]["semantic_query_forward_calls"],
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
            totals["validation_readout_forward_calls"], totals["semantic_query_forward_calls"],
            totals["semantic_query_backward_attempts"], totals["validation_semantic_query_forward_calls"]) != (60, 17460, 1115400, 90360, 1440, 17460, 17460, 17460, 1440, 1440, 17460, 17460, 1440):
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
        row.update(dataset=dataset_name, condition="native256_semantic_role_conditioned_relation", component=source["component"])
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
               "semantic_query_forward_calls": cost["counts"]["semantic_query_forward_calls"],
               "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
               "development_scope": "previously viewed external development; no target training, BN adaptation or checkpoint selection",
               "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
               "pipeline_change": "same native256 RGB preparation asv20; new semantic role-content queries execute with predicted local relation during deployment"})


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
