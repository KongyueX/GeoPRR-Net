"""Train a source-only spatial residual on one frozen full semantic RGB encoder."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import time

PROCESS_STARTED = time.perf_counter()

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_semantic_direct as direct
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_semantic_spatial_residual_source_only_v1"
PARENT_RELATIVE = "runs/geoprr_generalization/iteration_v16_full_semantic_direct_20261023/source_only_training"
CONFIG = {
    "protocol": PROTOCOL, "parent_protocol": direct.PROTOCOL, "parent_epoch": 59,
    "model": {"parent": "complete source-selected v16 semantic scalar reader, all parameters and BN buffers frozen",
              "local_feature": "encoder.blocks.1, reduction8,80channels,32x32 at256input",
              "spatial_tokens": 3, "token_dim": 32, "semantic_dim": 1280,
              "local_normalization": "per-example per-location layer_norm80",
              "semantic_normalization": "per-example layer_norm1280",
              "coordinates": "regular32x32 pixel-center xy in[-1,1]; no GT geometry",
              "residual": "three semantic-conditioned attention pools of projected local32+xy2; MLP102->32 GELU->1",
              "initialization": "final residual linear weight and bias zero; exact parent reading at epoch0"},
    "data": direct.CONFIG["data"], "source": direct.CONFIG["source"], "photometric": direct.CONFIG["photometric"],
    "training": {"seed": 20262020, "epochs": 8, "batch_size": 64, "validation_batch_size": 64,
                 "workers": 4, "amp": True, "optimizer": "AdamW", "weight_decay": .0001,
                 "lr": .001, "min_lr": .00001, "cosine_position": "(epoch-1)/8",
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.},
    "checkpoint_selection": "earliest minimum source1506 scalar NMAE over incumbent0 and all8 complete new epochs",
    "external_checkpoint": "source-selected best only after all8 epochs; incumbent0 reuses exact v16 external artifacts",
    "inference_input": direct.CONFIG["inference_input"], "external_training_or_adaptation": False,
}


class SpatialResidualReadout(nn.Module):
    """Pool generic spatial evidence using queries from the semantic vector."""

    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(80, 32, 1, bias=True)
        self.position = nn.Linear(2, 32, bias=False)
        self.queries = nn.Linear(1280, 96, bias=True)
        self.scores = nn.Parameter(torch.randn(3, 32) / math.sqrt(32))
        self.attention = nn.Softmax(dim=-1)
        self.residual = nn.Sequential(nn.Linear(102, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        locations = (torch.arange(32, dtype=torch.float32) + .5) / 16 - 1
        yy, xx = torch.meshgrid(locations, locations, indexing="ij")
        self.register_buffer("xy", torch.stack((xx, yy), -1).reshape(1024, 2), persistent=False)

    def forward(self, local, semantic):
        if local.shape[1:] != (80, 32, 32) or semantic.shape != (len(local), 1280):
            raise ValueError("spatial readout requires native blocks.1 B80x32x32 and B1280 semantic features")
        normalized = F.layer_norm(local.permute(0, 2, 3, 1).float(), (80,)).permute(0, 3, 1, 2)
        projected = self.local(normalized).flatten(2).transpose(1, 2)
        key = projected + self.position(self.xy)[None]
        queries = self.queries(F.layer_norm(semantic.float(), (1280,))).reshape(-1, 3, 32)
        energy = (self.scores[None, :, None] * torch.tanh(key[:, None] + queries[:, :, None])).sum(-1)
        attention = self.attention(energy.float())
        values = torch.cat((projected.float(), self.xy[None].expand(len(local), -1, -1)), -1)
        tokens = (attention[..., None] * values[:, None]).sum(2).flatten(1)
        return self.residual(tokens).squeeze(-1).float()


class SemanticSpatialReader(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.branch = SpatialResidualReadout()
        self._local = None
        self._hook = self.base.encoder.blocks[1].register_forward_hook(self._capture)

    @property
    def encoder(self):
        return self.base.encoder

    def _capture(self, _module, _inputs, output):
        if self._local is not None:
            raise RuntimeError("spatial feature stage was called more than once in the native encoder pass")
        self._local = output

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, image):
        self._local = None
        with torch.no_grad():
            semantic = self.base.encoder(image)
            parent_logit = self.base.point(semantic).squeeze(-1).float()
        local, self._local = self._local, None
        if local is None:
            raise RuntimeError("native encoder did not expose its declared blocks.1 spatial feature")
        return (parent_logit + self.branch(local, semantic).float()).sigmoid()


def load_parent(home):
    directory = home / PARENT_RELATIVE / "reader"
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    completed = json.loads((directory / "completed.json").read_text())
    summary = json.loads((directory / "best_validation.json").read_text())
    if (checkpoint["protocol"] != direct.PROTOCOL or checkpoint["config"] != direct.CONFIG
            or checkpoint["epoch"] != 59 or checkpoint["best_epoch"] != 59
            or completed["status"] != "complete" or completed["epochs"] != 60 or completed["best_epoch"] != 59
            or checkpoint["best_nmae"] != summary["nmae"]):
        raise ValueError("spatial residual requires the complete frozen source-selected v16 best59 parent")
    return checkpoint, summary, directory


def create_model(parent_state=None):
    base = direct.create_model(pretrained=False)
    if parent_state is not None:
        base.load_state_dict(parent_state, strict=True)
    model = SemanticSpatialReader(base)
    if (sum(parameter.numel() for parameter in model.branch.parameters()), len(model.base.state_dict()),
            sum(parameter.numel() for parameter in model.parameters()), len(model.state_dict())) != (129057, 464, 8564850, 474):
        raise ValueError("spatial branch or complete frozen semantic parent architecture differs")
    return model


def frozen_base_evidence(model, reference):
    state = model.base.state_dict()
    if set(state) != set(reference) or any(not torch.equal(value.detach().cpu(), reference[key]) for key, value in state.items()):
        raise RuntimeError("a frozen parent parameter or buffer changed")
    if model.base.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError("parent must remain eval with no trainable parameter or gradient")
    counters = [key for key in state if key.endswith("num_batches_tracked")]
    return {"parent_state_tensors_compared": len(state), "all_parent_state_tensors_unchanged": True,
            "BN_counters_compared": len(counters), "BN_counter_increment": 0,
            "cnn_backward_calls": 0, "parent_optimizer_updates": 0}


def model_metadata(model):
    return {"class": f"{type(model).__module__}.{type(model).__name__}", "parent": direct.model_metadata(model.base),
            "branch_parameters": sum(parameter.numel() for parameter in model.branch.parameters()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_tensors": len(model.state_dict()), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "CNN_passes_per_image": 1, "local_feature": "blocks.1:80x32x32", "spatial_tokens": "three generic learned tokens; no annotated role identity"}


def learning_rate(epoch):
    if not 1 <= epoch <= 8:
        raise ValueError("the independent spatial branch experiment has exactly8 epochs")
    return .00001 + (.001 - .00001) * .5 * (1 + math.cos(math.pi * (epoch - 1) / 8))


def incumbent_rows(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    return [{**{key: row[key] for key in ("sample_id", "group_id", "failure")},
             **{key: float(row[key]) for key in ("target", "absolute_error")},
             **{key: row[key] == "True" for key in ("valid", "input_valid")},
             **{key: float(row[key]) if row[key] else None for key in ("prediction", "invalid_mass", "outside_arc_mass")}}
            for row in saved]


def runtime(device, started):
    value = direct.runtime(device, started)
    value["full_process_seconds"] = time.perf_counter() - PROCESS_STARTED
    return value


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    if (output / "completed.json").exists():
        raise FileExistsError("spatial training is complete; reuse the declared stage")
    parent, incumbent, parent_directory = load_parent(home)
    reference = {key: value.detach().cpu().clone() for key, value in parent["model"].items()}
    source = home / direct.SOURCE_RELATIVE
    training = direct.ScalarRGBDataset(source / "manifests/train.jsonl", data_root=home / "datasets", augment=True)
    validation = direct.ScalarRGBDataset(home / "runs/geoprr_v2/manifests/validation.jsonl", data_root=home / "datasets", augment=False)
    contract = direct.source_contract(home, training, validation)
    if contract != parent["data_contract"]:
        raise ValueError("spatial residual must reuse exactly the parent source population and256 input contract")
    parent_rows = incumbent_rows(parent_directory / "best_predictions.csv.gz")
    if [row["sample_id"] for row in parent_rows] != [str(row["sample_id"]) for row in validation.rows]:
        raise ValueError("saved incumbent source rows differ from the declared validation population")
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if not latest.exists() and any((output / name).exists() for name in ("history.jsonl", "best.pt")):
        raise FileExistsError("partial spatial outputs have no latest checkpoint; inspect retained artifacts")
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    model = create_model(parent["model"]).to(device)
    optimizer = torch.optim.AdamW(model.branch.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {"train_batches": 0, "train_cnn_image_forwards": 0, "train_cnn_forward_calls": 0,
              "branch_backward_calls": 0, "cnn_backward_calls": 0, "optimizer_updates": 0, "amp_skipped_updates": 0,
              "validation_images": 0, "validation_cnn_forward_calls": 0, "training_seconds": 0., "validation_seconds": 0.}
    start, best_epoch, best, history, latest_rows, checkpointed_seconds = 0, 0, incumbent["nmae"], [], parent_rows, 0.
    parent_identity = {"checkpoint": str(parent_directory / "best.pt"), "protocol": direct.PROTOCOL,
                       "epoch": 59, "best_epoch": 59, "source_best_nmae": parent["best_nmae"]}
    if saved is not None:
        if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract
                or saved["parent"] != parent_identity):
            raise ValueError("resumed spatial experiment protocol, source or parent differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best_epoch, best, history, totals = saved["epoch"], saved["best_epoch"], saved["best_nmae"], saved["history"], saved["totals"]
        latest_rows, checkpointed_seconds = saved["validation_rows"], saved["checkpointed_process_seconds"]
    initial_evidence = frozen_base_evidence(model, reference)
    if initial_evidence["parent_state_tensors_compared"] != 464 or initial_evidence["BN_counters_compared"] != 77:
        raise ValueError("complete frozen parent state or BN inventory differs")

    def checkpoint_record(epoch):
        return {"protocol": PROTOCOL, "config": CONFIG, "model": model.state_dict(), "data_contract": contract,
                "parent": parent_identity, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                "epoch": epoch, "best_epoch": best_epoch, "best_nmae": best, "history": history, "totals": totals,
                "validation_rows": latest_rows, "incumbent_validation": incumbent,
                "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started}

    if saved is None:
        record = checkpoint_record(0)
        save_checkpoint(output / "best.pt", record)
        shutil.copy2(parent_directory / "best_predictions.csv.gz", output / "best_predictions.csv.gz")
        write_json(output / "best_validation.json", incumbent)
        save_checkpoint(latest, record)
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
               "parent": parent_identity, "model_metadata": model_metadata(model),
               "incumbent": {"epoch": 0, "validation": incumbent, "source_predictions_reused": True, "new_CNN_calls": 0},
               "initial_frozen_base_evidence": initial_evidence,
               "comparison": "new spatial residual only; exact same frozen complete parent, source population and256 RGB pipeline"})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start + 1, 9):
        tick = time.perf_counter()
        model.train()
        used_lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = used_lr
        loader = make_loader(training, batch_size=64, workers=4, shuffle=True, seed=20262020 + epoch, device=device)
        counts, hook = direct.encoder_counter(model)
        batches, updates, skips, loss_sum, error_sum = 0, 0, 0, 0., 0.
        try:
            for index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction, batch["y"], beta=.05)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite spatial scalar loss at epoch{epoch} batch{index + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.branch.parameters(), 5.)
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
        direct.synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (291, 291, 18590):
            raise RuntimeError("spatial branch source epoch exposure differs")
        latest_rows, validation_cost = direct.evaluate_population(model, validation, device)
        evidence = frozen_base_evidence(model, reference)
        summary = summarize(latest_rows)
        improved = summary["nmae"] < best
        if improved:
            best, best_epoch = summary["nmae"], epoch
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
                           "train_cnn_forward_calls": counts["cnn_forward_calls"], "branch_backward_calls": batches,
                           "optimizer_updates": updates, "amp_skipped_updates": skips, "validation_images": len(latest_rows),
                           "validation_cnn_forward_calls": validation_cost["counts"]["cnn_forward_calls"],
                           "training_seconds": train_seconds, "validation_seconds": validation_cost["pipeline_seconds"]}.items():
            totals[key] += value
        history.append({"epoch": epoch, "learning_rate": used_lr, "source_training_smooth_l1": loss_sum / 18590,
                        "source_training_nmae": error_sum / 18590, "train_counts": counts,
                        "branch_backward_calls": batches, "cnn_backward_calls": 0, "optimizer_updates": updates,
                        "amp_skipped_updates": skips, "validation": summary, "validation_cost": validation_cost,
                        "best_epoch": best_epoch, "best_nmae": best, "training_seconds": train_seconds,
                        "frozen_base_evidence": evidence, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        checkpoint = checkpoint_record(epoch)
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
            totals["validation_cnn_forward_calls"], totals["branch_backward_calls"], totals["cnn_backward_calls"]) != (8, 2328, 148720, 12048, 192, 2328, 0):
        raise RuntimeError("completed spatial residual does not contain all8 complete declared source epochs")
    if totals["optimizer_updates"] + totals["amp_skipped_updates"] != 2328:
        raise RuntimeError("spatial effective update and AMP skip counts differ from actual attempts")
    if not (output / "latest_predictions.csv.gz").exists():
        save_predictions(output / "latest_predictions.csv.gz", latest_rows)
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": 8,
               "parent": parent_identity, "best_epoch": best_epoch, "best_nmae": best,
               "best_validation": incumbent if best_epoch == 0 else history[best_epoch - 1]["validation"],
               "incumbent_validation": incumbent, "last_validation": history[-1]["validation"], "totals": totals,
               "model_metadata": model_metadata(model), "frozen_base_evidence": history[-1]["frozen_base_evidence"],
               "resume_from_epoch": start, "runtime": runtime(device, started),
               "checkpointed_process_seconds_before_resume": checkpointed_seconds,
               "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def evaluate(home, checkpoint_path, dataset_name, output):
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed = json.loads((checkpoint_path.parent / "completed.json").read_text())
    if (checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG or completed["status"] != "complete"
            or completed["epochs"] != 8 or checkpoint["epoch"] != completed["best_epoch"]
            or checkpoint["epoch"] != checkpoint["best_epoch"]):
        raise ValueError("external evaluation requires the best over incumbent0 and all8 completed source epochs")
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint["epoch"] == 0:
        parent_output = home / PARENT_RELATIVE / "evaluations" / dataset_name
        parent_result = json.loads((parent_output / "summary.json").read_text())
        if parent_result["status"] != "complete" or parent_result["checkpoint_epoch"] != 59:
            raise ValueError("incumbent alias requires complete parent external best59 results")
        for name in ("predictions.csv", "groups.csv"):
            source, destination = parent_output / name, output / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("partial incumbent alias differs from its saved parent artifact")
            if not destination.exists():
                shutil.copy2(source, destination)
        shutil.copy2(parent_output / "summary.json", output / "parent_summary.json")
        result = {**parent_result, "protocol": PROTOCOL, "config": CONFIG, "checkpoint": str(checkpoint_path),
                  "checkpoint_epoch": 0, "checkpoint_best_epoch": 0, "checkpoint_source_best_nmae": checkpoint["best_nmae"],
                  "parent": checkpoint["parent"], "model_metadata": completed["model_metadata"],
                  "reused_parent_model_metadata": parent_result["model_metadata"],
                  "reuses": str(parent_output), "alias_reason": "source-selected incumbent0 is exactly the frozen parent; no new target execution",
                  "parent_checkpoint_epoch": 59, "reused_parent_forward_counts": parent_result["forward_counts"],
                  "forward_counts": {"cnn_image_forwards": 0, "cnn_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0},
                  "timing": {"alias_copy_seconds": time.perf_counter() - started},
                  "runtime": {"device": "cpu", "new_CNN_execution": False, "full_process_seconds": time.perf_counter() - PROCESS_STARTED}}
        write_json(output / "summary.json", result)
        return
    device = torch.device("cuda")
    model = create_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    parent, _incumbent, _directory = load_parent(home)
    reference = {key: value.detach().cpu() for key, value in parent["model"].items()}
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    rows, cost = direct.evaluate_population(model, dataset, device)
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256", component=source["component"])
        groups[row["group_id"]].append(row)
        components[row["component"]].append(row)
    direct.save_csv(output / "predictions.csv", rows)
    direct.save_csv(output / "groups.csv", [{"group_id": key, **summarize(value)} for key, value in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
               "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
               "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
               "checkpoint_source_best_nmae": checkpoint["best_nmae"], "parent": checkpoint["parent"],
               "model_metadata": model_metadata(model), "frozen_base_evidence": frozen_base_evidence(model, reference),
               "summary": summarize(rows), "by_group": {key: summarize(value) for key, value in groups.items()},
               "by_component": {key: summarize(value) for key, value in components.items()},
               "forward_counts": cost["counts"], "timing": cost, "runtime": runtime(device, started),
               "model_inputs": CONFIG["inference_input"], "training_or_adaptation_during_evaluation": False,
               "backward_calls": 0, "optimizer_updates": 0,
               "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
               "development_scope": "previously viewed external development; no target fitting, BN adaptation or checkpoint selection",
               "failure_policy": "retain every declared input; input or scalar failure receives normalized error1"})


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
