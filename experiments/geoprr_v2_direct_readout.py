"""Probe a source-trained direct scalar head on one frozen native RGB feature pass."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import torch
from torch import nn

from experiments.evaluate_geoprr_v2_generalization import reader_from_checkpoint
from experiments.evaluate_geoprr_v2_role_external import (
    ExternalPopulationDataset, POPULATIONS, load_population, summarize, write_csv,
)
from experiments.geoprr_v2_endpoint_source import verify_endpoint_training_contract
from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import device_batch, make_loader, save_checkpoint, seed_everything, source_split_check
from experiments.train_geoprr_v2_reference_relation import save_predictions, validation_rows


PROTOCOL = "geoprr_v2_frozen_rgb_direct_readout_source_only_v1"
NATIVE_PROTOCOL = "geoprr_v2_reference_feedback_source_only_v1"
PARENT_RELATIVE = "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt"
SOURCE_RELATIVE = "rendering/generalization_endpoints_v11_20261023"
READOUTS = ("geometry", "direct", "fixed_half")
CONFIG = {
    "protocol": PROTOCOL, "parent_protocol": NATIVE_PROTOCOL, "parent_epoch": 48,
    "data": {"image_size": 768, "output_size": 192},
    "features": "last native CNN feature map, global average pooling; float32 cache",
    "normalization": {"fit_population": "source_train_unique_only", "std_unbiased": False, "epsilon": 1e-6},
    "head": "sigmoid(linear(standardized_global_feature)); zero weight and bias initialization",
    "training": {"seed": 20262020, "epochs": 20, "batch_size": 256, "lr": .001,
                 "weight_decay": .0001, "min_lr": .00001, "loss": "normalized L1", "device": "cpu"},
    "cache": {"source_train_batch_size": 32, "source_validation_batch_size": 16,
              "external_batch_size": 16, "workers": 4, "amp": True, "augment": False},
    "checkpoint_selection": "earliest minimum direct-head NMAE on complete source1506 within20epochs",
    "readouts": list(READOUTS), "fixed_half_alpha": .5,
    "external_training_or_adaptation": False,
}


class DirectScalarHead(nn.Module):
    """A bounded point estimate conditioned only on source feature statistics."""

    def __init__(self, mean, std):
        super().__init__()
        if mean.ndim != 1 or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("feature statistics must be matching finite vectors")
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("std", std.detach().float().clamp_min(CONFIG["normalization"]["epsilon"]).clone())
        self.linear = nn.Linear(len(mean), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @classmethod
    def from_source_features(cls, features):
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("source features must be a finite sample by channel matrix")
        return cls(features.float().mean(0), features.float().std(0, unbiased=False))

    def forward(self, features):
        return self.linear((features.float() - self.mean) / self.std).sigmoid().squeeze(-1)


class FrozenFeatureCapture:
    """Capture GAP features from the backbone already called by the native reader."""

    def __init__(self, native):
        self.native = native
        self.feature_index = native.feature_indices[-1]
        self.features = []
        self.calls = self.images = 0
        self.hook = native.backbone.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.calls += 1
        self.images += len(inputs[0])
        self.features.append(output[self.feature_index].float().mean((2, 3)))

    def __call__(self, images):
        if self.native.training or any(parameter.requires_grad for parameter in self.native.parameters()):
            raise ValueError("the entire native reader must remain frozen and in eval mode")
        self.features.clear()
        output = self.native(images)
        if len(self.features) != 1:
            raise RuntimeError("native forward must call its CNN exactly once")
        return output, self.features[0]

    def close(self):
        self.hook.remove()


class UniqueSourceDataset(ReferenceRelationDataset):
    """Read each source image once while retaining the original pair contract."""

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self._sample(index, None)


def snapshot_state(native):
    return {key: value.detach().cpu().clone() for key, value in native.state_dict().items()}


def verify_stationary(native, snapshot):
    state = native.state_dict()
    if set(state) != set(snapshot) or any(not torch.equal(value.detach().cpu(), snapshot[key]) for key, value in state.items()):
        raise RuntimeError("frozen native parameters or buffers changed")
    if native.training or any(parameter.requires_grad or parameter.grad is not None for parameter in native.parameters()):
        raise RuntimeError("frozen native reader mode or gradient state changed")
    return {"state_tensors_compared": len(state), "all_native_state_tensors_unchanged": True,
            "native_training": False, "native_backward_calls": 0, "native_optimizer_updates": 0}


def load_native(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_model = {"bins": 256, "decoder_dim": 64, "token_dim": 16, "coupling": "joint",
                      "reference_normalization": "global"}
    if (checkpoint["protocol"] != NATIVE_PROTOCOL or checkpoint["epoch"] != 48
            or checkpoint["best_epoch"] != 48 or checkpoint["config"]["training"]["seed"] != 20262020
            or checkpoint["config"]["model"] != expected_model or checkpoint["config"]["data"] != CONFIG["data"]
            or checkpoint["config"]["training"]["amp"] is not True):
        raise ValueError("direct probe requires the complete native feedback48 source-selected parent")
    native, metadata = reader_from_checkpoint(checkpoint)
    if metadata["parameters"] != 7461060 or metadata["state_tensors"] != 496:
        raise ValueError("native parent architecture differs")
    dimension = native.backbone.feature_info.channels()[native.feature_indices[-1]]
    native.to(device).eval().requires_grad_(False)
    return native, checkpoint, {**metadata, "feature_dim": dimension, "parent_checkpoint": str(path.resolve()),
                                "parent_epoch": 48, "parent_seed": 20262020}


def readout_rows(geometry, direct):
    """Keep every declared row; the fixed mean requires two valid estimates."""
    direct = torch.as_tensor(direct).detach().float().cpu().reshape(-1).tolist()
    if len(geometry) != len(direct):
        raise ValueError("direct predictions and native rows differ in length")
    result = {key: [] for key in READOUTS}
    for original, value in zip(geometry, direct, strict=True):
        direct_valid = original["input_valid"] and math.isfinite(value) and 0 <= value <= 1
        half_valid = bool(original["valid"] and direct_valid)
        estimates = {"geometry": (original["prediction"], original["valid"]), "direct": (value, direct_valid),
                     "fixed_half": ((original["prediction"] + value) / 2 if half_valid else None, half_valid)}
        for kind, (prediction, valid) in estimates.items():
            result[kind].append({
                "sample_id": original["sample_id"], "group_id": original["group_id"], "target": original["target"],
                "prediction": prediction if valid else None, "valid": bool(valid), "input_valid": original["input_valid"],
                "absolute_error": abs(prediction - original["target"]) if valid else 1.,
                "invalid_mass": original["invalid_mass"] if kind == "geometry" else None,
                "outside_arc_mass": original["outside_arc_mass"] if kind == "geometry" else None,
                "failure": original["failure"] if kind == "geometry" else (
                    "" if valid else original["failure"] or "invalid_direct_or_fixed_half_prediction"),
            })
    return result


def combined_rows(readouts, components=None):
    rows = []
    for index, base in enumerate(readouts["geometry"]):
        row = {key: base[key] for key in ("sample_id", "group_id", "target", "input_valid")}
        if components is not None:
            row["component"] = components[index]
        for kind in READOUTS:
            row.update({f"{kind}_{key}": readouts[kind][index][key]
                        for key in ("prediction", "valid", "absolute_error", "failure")})
        row.update(invalid_mass=base["invalid_mass"], outside_arc_mass=base["outside_arc_mass"])
        rows.append(row)
    return rows


def save_combined_csv(path, rows):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        expected = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
        if existing != expected:
            raise ValueError("saved partial prediction rows differ; preserve and inspect the failed stage")
        return
    write_csv(path, rows)


def readout_summaries(readouts, components=None):
    result = {}
    for kind, rows in readouts.items():
        groups, component_rows = defaultdict(list), defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["group_id"]].append(row)
            if components is not None:
                component_rows[components[index]].append(row)
        result[kind] = {"summary": summarize(rows), "by_group": {key: summarize(value) for key, value in groups.items()}}
        if components is not None:
            result[kind]["by_component"] = {key: summarize(value) for key, value in component_rows.items()}
    return result


def cache_population(native, dataset, *, batch_size, workers, device, destination):
    """Save completed batch features so a stopped cache stage can reuse them."""
    started = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=True)
    loader = make_loader(dataset, batch_size=batch_size, workers=workers, shuffle=False, seed=20262020, device=device)
    capture = FrozenFeatureCapture(native)
    chunks, forward_seconds = [], 0.
    try:
        with torch.inference_mode():
            for index, raw in enumerate(loader):
                path = destination / f"batch_{index:05d}.pt"
                if path.exists():
                    chunk = torch.load(path, map_location="cpu", weights_only=False)
                    if [row["sample_id"] for row in chunk["rows"]] != list(raw["sample_id"]):
                        raise ValueError("saved cache batch sample identities differ")
                else:
                    batch = device_batch(raw, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        output, features = capture(batch["image"])
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - tick
                    chunk = {"features": features.detach().float().cpu(), "rows": validation_rows(output, batch),
                             "forward_seconds": elapsed, "cnn_forward_calls": 1, "cnn_image_forwards": len(raw["sample_id"])}
                    save_checkpoint(path, chunk)
                chunks.append(chunk)
                forward_seconds += chunk["forward_seconds"]
    finally:
        capture.close()
    features = torch.cat([chunk["features"] for chunk in chunks])
    rows = [row for chunk in chunks for row in chunk["rows"]]
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in dataset.rows]:
        raise ValueError("cache lost, duplicated or reordered declared images")
    counts = {"cnn_image_forwards": sum(chunk["cnn_image_forwards"] for chunk in chunks),
              "cnn_forward_calls": sum(chunk["cnn_forward_calls"] for chunk in chunks),
              "cnn_backward_calls": 0, "cnn_optimizer_updates": 0,
              "failed_input_placeholder_forwards": sum(not row["input_valid"] for row in rows),
              "new_cnn_image_forwards_this_process": capture.images, "new_cnn_forward_calls_this_process": capture.calls}
    return {"features": features, "rows": rows, "counts": counts,
            "timing": {"pipeline_seconds_this_process": time.perf_counter() - started,
                       "saved_batch_forward_seconds": forward_seconds, "batch_size": batch_size, "workers": workers}}


def runtime_metadata(device, started):
    return {"process_seconds": time.perf_counter() - started, "device": str(device), "torch": torch.__version__,
            "full_process_seconds": time.perf_counter() - PROCESS_STARTED,
            "full_process_boundary": "module entry before torch import through saved predictions; excludes final JSON write",
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def cache_source(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    native, parent, metadata = load_native(home / PARENT_RELATIVE, device)
    frozen = snapshot_state(native)
    data_root, source = home / "datasets", home / SOURCE_RELATIVE
    training = UniqueSourceDataset(source / "manifests/train.jsonl", data_root=data_root, data_config=CONFIG["data"],
                                   augment=False, pair_manifest=source / "manifests/train_edges.jsonl")
    validation = UniqueSourceDataset(home / "runs/geoprr_v2/manifests/validation.jsonl", data_root=data_root,
                                     data_config=CONFIG["data"], augment=False)
    if (len(training.rows), len(training.items), len(training.edges), len(validation)) != (18590, 22686, 8192, 1506):
        raise ValueError("unique source or validation population differs")
    source_split_check(training.rows, validation.rows)
    contract = {"paths": {"train_manifest": str((source / "manifests/train.jsonl").resolve()),
                          "pair_manifest": str((source / "manifests/train_edges.jsonl").resolve()),
                          "val_manifest": str((home / "runs/geoprr_v2/manifests/validation.jsonl").resolve())},
                "train": training.supervision_contract(), "validation": validation.supervision_contract(),
                "exposure_edges": training.edges}
    replacement = verify_endpoint_training_contract(contract, parent["data_contract"])
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "data_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("saved partial source contract differs")
    write_json(contract_path, contract)
    result = {}
    for split, dataset, batch_size in (("train", training, 32), ("validation", validation, 16)):
        cached = cache_population(native, dataset, batch_size=batch_size, workers=4, device=device, destination=output / split)
        if cached["features"].shape != (len(dataset), metadata["feature_dim"]):
            raise ValueError("cached native feature dimensions differ")
        save_checkpoint(output / f"{split}.pt", {"protocol": PROTOCOL, "config": CONFIG, "parent": metadata, **cached})
        result[split] = {"samples": len(dataset), "geometry_summary": summarize(cached["rows"]),
                         "counts": cached["counts"], "timing": cached["timing"]}
    counts = {key: sum(result[split]["counts"][key] for split in result) for key in result["train"]["counts"]}
    if (counts["cnn_image_forwards"], counts["cnn_forward_calls"]) != (20096, 676):
        raise ValueError("source cache CNN exposure differs")
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "parent": metadata,
               "source_replacement": replacement, "counts": counts, "splits": result,
               "frozen_native_evidence": verify_stationary(native, frozen), "runtime": runtime_metadata(device, started)})


def head_from_checkpoint(checkpoint):
    if checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG:
        raise ValueError("head checkpoint protocol or frozen recipe differs")
    state = checkpoint["head"]
    head = DirectScalarHead(state["mean"], state["std"])
    head.load_state_dict(state, strict=True)
    return head.eval()


def fit_head(cache_dir, output):
    started = time.perf_counter()
    seed_everything(20262020)
    torch.set_num_threads(4)
    cache = {split: torch.load(cache_dir / f"{split}.pt", map_location="cpu", weights_only=False)
             for split in ("train", "validation")}
    for split, expected in (("train", 18590), ("validation", 1506)):
        item = cache[split]
        if item["protocol"] != PROTOCOL or item["config"] != CONFIG or len(item["rows"]) != expected:
            raise ValueError("source cache contract differs")
        if any(not row["input_valid"] for row in item["rows"]) or not torch.isfinite(item["features"]).all():
            raise ValueError("source head fitting requires every declared source RGB feature to be valid")
    if cache["train"]["parent"] != cache["validation"]["parent"]:
        raise ValueError("source train and validation cached native parents differ")
    features = cache["train"]["features"].float()
    target = torch.tensor([row["target"] for row in cache["train"]["rows"]], dtype=torch.float32)
    head = DirectScalarHead.from_source_features(features)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=.00001)
    output.mkdir(parents=True, exist_ok=True)
    start, best, best_epoch, history, updates, presentations = 0, math.inf, None, [], 0, 0
    latest = output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        head = head_from_checkpoint(saved)
        if saved["parent"] != cache["train"]["parent"]:
            raise ValueError("resumed head native parent differs")
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=.00001)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, best_epoch, history = saved["epoch"], saved["best_nmae"], saved["best_epoch"], saved["history"]
        updates, presentations = saved["counts"]["head_optimizer_updates"], saved["counts"]["source_feature_presentations"]
        torch.set_rng_state(saved["torch_rng"])
    training_seconds = 0.
    for epoch in range(start + 1, 21):
        tick = time.perf_counter()
        head.train()
        order = torch.randperm(len(features))
        loss_sum, epoch_updates = 0., 0
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (head(features[indices]) - target[indices]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite source head loss")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
                raise ValueError("nonfinite direct-head gradient")
            optimizer.step()
            updates += 1
            epoch_updates += 1
            presentations += len(indices)
            loss_sum += float(loss.detach()) * len(indices)
        elapsed = time.perf_counter() - tick
        training_seconds += elapsed
        head.eval()
        with torch.inference_mode():
            direct = head(cache["validation"]["features"])
        rows = readout_rows(cache["validation"]["rows"], direct)
        summaries = readout_summaries(rows)
        nmae = summaries["direct"]["summary"]["nmae"]
        improved = nmae < best
        if improved:
            best, best_epoch = nmae, epoch
        record = {"epoch": epoch, "source_training_l1": loss_sum / len(features), "validation": summaries,
                  "training_seconds": elapsed, "lr": optimizer.param_groups[0]["lr"],
                  "head_optimizer_updates": epoch_updates, "source_feature_presentations": len(features)}
        history.append(record)
        scheduler.step()
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "head": head.state_dict(),
                      "parent": cache["train"]["parent"], "epoch": epoch, "best_epoch": best_epoch,
                      "best_nmae": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "torch_rng": torch.get_rng_state(), "history": history,
                      "counts": {"head_optimizer_updates": updates, "head_backward_calls": updates,
                                 "source_feature_presentations": presentations, "cnn_image_forwards": 0,
                                 "cnn_forward_calls": 0, "cnn_backward_calls": 0, "cnn_optimizer_updates": 0}}
        save_checkpoint(latest, checkpoint)
        save_predictions(output / "latest_predictions.csv.gz", combined_rows(rows))
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
            save_predictions(output / "best_predictions.csv.gz", combined_rows(rows))
        (output / "history.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
        print(json.dumps({"epoch": epoch, "direct_source_nmae_percent_fs": 100 * nmae, "best_epoch": best_epoch}), flush=True)
    if (updates, presentations) != (1460, 371800):
        raise ValueError("head-only fitting exposure differs from20 source feature epochs")
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
               "epochs": 20, "best_epoch": best_epoch, "best_direct_nmae": best,
               "parameters": sum(parameter.numel() for parameter in head.parameters()),
               "feature_dim": features.shape[1], "parent": cache["train"]["parent"],
               "normalization_fit_samples": 18590,
               "counts": {"head_optimizer_updates": updates, "head_backward_calls": updates,
                          "source_feature_presentations": presentations, "cnn_image_forwards": 0,
                          "cnn_forward_calls": 0, "cnn_backward_calls": 0, "cnn_optimizer_updates": 0},
               "runtime": {"device": "cpu", "training_seconds_this_process": training_seconds,
                           "process_seconds": time.perf_counter() - started,
                           "full_process_seconds": time.perf_counter() - PROCESS_STARTED,
                           "full_process_boundary": "module entry before torch import through last saved head and source predictions; excludes final JSON write",
                           "torch_threads": torch.get_num_threads()}})


def evaluate(home, head_path, dataset_name, output):
    started, device = time.perf_counter(), torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    saved = torch.load(head_path, map_location="cpu", weights_only=False)
    if saved["epoch"] != saved["best_epoch"]:
        raise ValueError("external evaluation accepts only the source-selected best direct head")
    complete = json.loads((head_path.parent / "completed.json").read_text())
    if complete["status"] != "complete" or complete["epochs"] != 20 or complete["best_epoch"] != saved["epoch"]:
        raise ValueError("head fitting must finish before external evaluation")
    head = head_from_checkpoint(saved)
    native, _parent, metadata = load_native(home / PARENT_RELATIVE, device)
    if saved["parent"] != metadata:
        raise ValueError("external native parent differs from source feature cache")
    frozen = snapshot_state(native)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    cached = cache_population(native, dataset, batch_size=16, workers=4, device=device, destination=output / "batches")
    with torch.inference_mode():
        direct = head(cached["features"])
    readouts = readout_rows(cached["rows"], direct)
    components = [row["component"] for row in population]
    expected = (1395, 88) if dataset_name == "industrial" else (151, 10)
    if (cached["counts"]["cnn_image_forwards"], cached["counts"]["cnn_forward_calls"]) != expected:
        raise ValueError("external CNN exposure differs from one pass per declared RGB")
    save_combined_csv(output / "predictions.csv", combined_rows(readouts, components))
    write_json(output / "summary.json", {
        "protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(population), "declared_groups": POPULATIONS[dataset_name][1],
        "head_checkpoint": str(head_path.resolve()), "head_epoch": saved["epoch"], "source_best_direct_nmae": saved["best_nmae"],
        "parent": metadata, "readouts": readout_summaries(readouts, components), "counts": cached["counts"],
        "timing": cached["timing"], "runtime": runtime_metadata(device, started),
        "frozen_native_evidence": verify_stationary(native, frozen), "backward_calls": 0, "optimizer_updates": 0,
        "model_inputs": "original ROI RGB only; no target geometry, scalar or statistical adaptation",
        "development_scope": "previously viewed external development; no target fitting or checkpoint selection",
        "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
        "failure_policy": "all declared samples retained; each invalid readout receives normalized error1; fixed half requires both estimates valid",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cache_source", "fit_head", "evaluate"))
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--dataset-name", choices=tuple(POPULATIONS))
    args = parser.parse_args(argv)
    if args.stage == "cache_source":
        cache_source(args.home, args.output)
    elif args.stage == "fit_head":
        if args.cache is None:
            parser.error("fit_head requires --cache")
        fit_head(args.cache, args.output)
    else:
        if args.head_checkpoint is None or args.dataset_name is None:
            parser.error("evaluate requires --head-checkpoint and --dataset-name")
        evaluate(args.home, args.head_checkpoint, args.dataset_name, args.output)


if __name__ == "__main__":
    main()
