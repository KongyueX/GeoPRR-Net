"""Frozen source-only conditional view-gradient control with two fresh last15 arms."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import resource
import sys

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_needle_holdout_control import (
    CONFIG as PREPARATION_CONFIG, EXPECTED_COUNTS, MANIFESTS, PREPARED_RELATIVE,
    PROTOCOL as PREPARATION_PROTOCOL, SEEN, _save_evaluation, learning_gate, write_json,
)

PROTOCOL = "geoprr_v2_conditional_view_adversary_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_conditional_view_adversary_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_conditional_view_adversary_preflight_20261026/result.json"
REFERENCE_RELATIVE = "runs/geoprr_generalization/source_matched_identity_control_20261026/reader"
ARMS = {"detached_control": 0., "reversal": .05}
NATIVE_ORDER, CF_ORDER = (0, 1, 2), (1, 0, 2)
REFERENCE = {
    "source_id": {"nmae": .025431218243695233, "tail10_nmae": .11800955331418664,
        "maximum_error": .5379703640937805, "over5": 16, "accurate1": 102},
    "source_counterfactual": {"nmae": .026353762864682596, "tail10_nmae": .12945961876345488,
        "maximum_error": .5424728989601135, "over5": 17, "accurate1": 104},
}
CONFIG = {
    "protocol": PROTOCOL, "preparation_protocol": PREPARATION_PROTOCOL,
    "source": {"train_rows": 5496, "Z_counts": [2748, 2748], "parents": 687,
        "view_quartets": 1374, "pairs": 2748, "source_ID_rows": 292, "source_ID_groups": 73,
        "source03_calls": 0, "external_calls": 0, "metadata_pair_y_tolerance": 1e-6},
    "model": {"reader": "unchanged fresh cached MNV4/full GAP1280/Linear1/scalarF32",
        "critic": [1281, 64, 1], "critic_precision": "float32", "critic_bias": True,
        "critic_BN_dropout": False, "critic_parameters": 82113, "critic_parameter_tensors": 4,
        "expected_total_parameters": 8517906, "expected_parameter_tensors": 237,
        "expected_state_tensors": 468, "expected_source_BN_tensors": 77,
        "critic_initialization": "standard Linear within fork_rng(devices=[]); restore main CPU RNG"},
    "arms": ARMS, "data": {"image_size": 512, "output_size": 128, "geometry": False, "pair_manifest": None},
    "photometric": {"contrast": [.8, 1.2], "saturation": [.8, 1.2], "brightness": [-.1, .1]},
    "training": {"seed": 20262020, "epochs": 15, "batch_size": 64, "last_batch_size": 56,
        "workers": 4, "loader_seed": "20262020+epoch", "optimizer": "AdamW", "lr": .0003,
        "weight_decay": .0001, "joint_clip": 5., "fresh_default_GradScaler": True,
        "smooth_l1_beta": .05, "loss_values": "primitive mean SmoothL1 + mean BCEWithLogits",
        "encoder_gradient": "primitive gradient minus alpha times BCE feature gradient",
        "critic_gradient": "unscaled BCE descent", "scheduler": None, "intermediate_validation": False,
        "checkpoint": "last15 only; strict complete-epoch full new state resume"},
    "reference": REFERENCE, "inference_inputs": ["image"],
    "scope": "one-seed source procedure control; no independence, causality or real-domain claim",
}
PREFLIGHT_COUNTS = {"CNN_calls": 3, "CNN_images": 8, "point_calls": 3, "point_examples": 8,
    "critic_calls": 1, "critic_examples": 4, "combined_backward_calls": 1, "CNN_backward_calls": 1,
    "point_backward_calls": 1, "critic_backward_calls": 1, "optimizer_updates": 0,
    "unique_RGB_images": 8, "RGB_decode_operations": 12, "source_ID_CPU_redecodes": 4}


def load_json(path):
    return json.loads(path.read_text())


def peak_cpu_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class MetadataNotAdmitted(ValueError):
    """An existing training metadata prerequisite is unavailable or inconsistent."""


def validate_metadata_rows(rows, expected, *, target_reader=None):
    """Bind saved view tags to consumed FP32 targets without deriving a new target formula."""
    if target_reader is None:
        from experiments.geoprr_v2_data import normalized_target
        target_reader = normalized_target
    if rows != expected or len(rows) != 5496:
        raise MetadataNotAdmitted("train_packet_or_order")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != 5496:
        raise MetadataNotAdmitted("train_identity")
    bindings, groups, originals, rolls = [], defaultdict(list), {}, []
    for row in rows:
        try:
            saved = row["metadata"]["needle_control"]
            view = saved["view"]
            if view not in ("original", "roll") or saved["pointer"] not in SEEN:
                raise MetadataNotAdmitted("train_view_or_pointer")
            if type(saved["slot"]) is not int or saved["slot"] not in (0, 1, 2, 3):
                raise MetadataNotAdmitted("train_slot")
            y = float(torch.tensor(target_reader(row), dtype=torch.float32))
            if not math.isfinite(y) or not 0 <= y <= 1:
                raise MetadataNotAdmitted("consumed_target_invalid")
            binding = {"sample_id": str(row["sample_id"]), "group_id": str(row["group_id"]),
                "y": y, "Z": int(view == "roll"), **deepcopy(saved)}
            if binding["group_id"] != str(saved["layout"]):
                raise MetadataNotAdmitted("train_group_layout")
            groups[(view, saved["layout"])].append(binding)
            if view == "original":
                originals[binding["sample_id"]] = binding
            else:
                rolls.append(binding)
            bindings.append(binding)
        except MetadataNotAdmitted:
            raise
        except (KeyError, TypeError, ValueError, OverflowError):
            raise MetadataNotAdmitted("required_metadata_or_consumed_target") from None
    counts = [sum(row["Z"] == z for row in bindings) for z in (0, 1)]
    pointer_counts = {alias: sum(row["pointer"] == alias for row in bindings) for alias in SEEN}
    if counts != [2748, 2748] or pointer_counts != {SEEN[0]: 2704, SEEN[1]: 2792} or len(groups) != 1374:
        raise MetadataNotAdmitted("train_counts")
    for quartet in groups.values():
        if (len(quartet) != 4 or {r["slot"] for r in quartet} != {0, 1, 2, 3}
                or len({json.dumps(r["state"], sort_keys=True) for r in quartet}) != 4
                or len({r["pointer"] for r in quartet}) != 1):
            raise MetadataNotAdmitted("train_quartet")
    if (len(originals) != 2748 or len(rolls) != 2748
            or {r.get("parent_sample_id") for r in rolls} != set(originals)
            or len({r.get("parent_sample_id") for r in rolls}) != 2748):
        raise MetadataNotAdmitted("parent_identity")
    parent_layouts = {}
    differences = []
    for row in rolls:
        parent = originals[row["parent_sample_id"]]
        if (row.get("parent_equal") is not True or row.get("parent_layout") != parent["layout"]
                or any(row[key] != parent[key] for key in ("slot", "state", "pointer"))):
            raise MetadataNotAdmitted("parent_slot_state_pointer_binding")
        previous = parent_layouts.setdefault(row["layout"], row["parent_layout"])
        if previous != row["parent_layout"]:
            raise MetadataNotAdmitted("parent_layout_binding")
        differences.append(abs(row["y"] - parent["y"]))
    if (len(parent_layouts) != 687 or len(set(parent_layouts.values())) != 687
            or set(parent_layouts.values()) != {r["layout"] for r in originals.values()}):
        raise MetadataNotAdmitted("parent_layout_closure")
    if max(differences) > 1e-6:
        raise MetadataNotAdmitted("paired_consumed_target_not_preserved")
    return bindings, {"train_rows": 5496, "Z_counts": counts, "parents": 687, "view_quartets": 1374,
        "paired_examples": 2748, "max_paired_consumed_target_difference": max(differences),
        "paired_target_tolerance": 1e-6, "production_target_reader": "experiments.geoprr_v2_data.normalized_target then float32",
        "prepared_parse_scope": "private nested train/ID/OOD metadata parsed; only train drives this stage",
        "ID_OOD_manifests_opened": 0, "RGB_reads": 0, "model_constructions": 0, "GPU_calls": 0,
        "CNN_calls": 0, "point_calls": 0, "critic_calls": 0, "backward_calls": 0, "optimizer_updates": 0}


def metadata(home, output):
    destination = output / "source_metadata"
    if destination.exists():
        raise FileExistsError("source metadata has complete or partial output; reuse only the complete saved stage")
    admitted, reason, contract, aggregate = False, None, None, None
    try:
        prepared = load_json(home / PREPARED_RELATIVE / "prepared.json")
        if (prepared["protocol"] != PREPARATION_PROTOCOL or prepared["config"] != PREPARATION_CONFIG
                or prepared["status"] != "complete" or prepared["counts"] != EXPECTED_COUNTS
                or prepared["manifests"] != MANIFESTS):
            raise MetadataNotAdmitted("prepared_packet_contract")
        path = home / PREPARED_RELATIVE / "manifests/train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        bindings, aggregate = validate_metadata_rows(rows, prepared["source_contract"]["populations"]["train"])
        contract = {"protocol": PROTOCOL, "prepared": prepared, "train_rows": rows, "bindings": bindings}
        admitted = True
    except MetadataNotAdmitted as error:
        reason = str(error)
    except (OSError, KeyError, TypeError, ValueError):
        reason = "required_prepared_or_train_metadata_unavailable"
    elapsed = time.perf_counter() - PROCESS_STARTED
    if elapsed > 30:
        admitted, reason, contract = False, "metadata_module_entry_budget_exceeded", None
    destination.mkdir(parents=True, exist_ok=False)
    result = {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "admitted": admitted,
        "verdict": "SOURCE_VIEW_METADATA_ADMITTED" if admitted else "NOT_ADMITTED_SOURCE_VIEW_METADATA",
        "reason": reason, "aggregate": aggregate, "module_entry_seconds": elapsed,
        "limit_seconds": 30, "peak_cpu_memory_bytes": peak_cpu_bytes(), "all_model_stages": None,
        "counts": {"RGB_reads": 0, "CNN_calls": 0, "point_calls": 0, "critic_calls": 0,
            "backward_calls": 0, "optimizer_updates": 0, "GPU_calls": 0}}
    if admitted:
        write_json(destination / "private_contract.json", contract)
    write_json(destination / "completed.json", result)
    if not admitted:
        write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
            "verdict": result["verdict"], "completion_reason": reason, "source_metadata": result,
            "preflight": None, "detached_control": None, "reversal": None, "joint_acceptance": None,
            "critic_effect": None, "counts": result["counts"]})
    print(json.dumps({key: result[key] for key in ("protocol", "status", "admitted", "verdict", "reason", "counts")}), flush=True)


def admission(home, output):
    result = load_json(output / "source_metadata/completed.json")
    if (result["protocol"] != PROTOCOL or result["config"] != CONFIG or not result["admitted"]
            or result["status"] != "complete"):
        raise ValueError("model stage requires the complete admitted source metadata")
    contract = load_json(output / "source_metadata/private_contract.json")
    prepared = load_json(home / PREPARED_RELATIVE / "prepared.json")
    rows = [json.loads(line) for line in (home / PREPARED_RELATIVE / "manifests/train.jsonl").read_text().splitlines() if line.strip()]
    if contract["protocol"] != PROTOCOL or contract["prepared"] != prepared or contract["train_rows"] != rows:
        raise ValueError("saved admitted metadata or training packet changed")
    return contract


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, alpha):
        ctx.alpha = alpha
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def reverse_gradient(value, alpha):
    if alpha not in ARMS.values():
        raise ValueError("alpha differs from the two frozen arms")
    return _GradientReverse.apply(value, alpha)


def make_critic():
    with torch.random.fork_rng(devices=[]):
        return nn.Sequential(nn.Linear(1281, 64, bias=True), nn.ReLU(), nn.Linear(64, 1, bias=True)).float()


class ConditionalViewReader(nn.Module):
    """Keep the public scalar reader image-only; critic is called only by the training helper."""
    def __init__(self, reader):
        super().__init__()
        self.reader = reader
        self.critic = make_critic()

    @property
    def encoder(self):
        return self.reader.encoder

    @property
    def point(self):
        return self.reader.point

    def forward(self, image):
        return self.reader(image)


def training_forward(model, image, y, Z, alpha, *, cuda_events=None):
    """Capture only this original point input and reverse only its BCE feature gradient."""
    holder = []
    def capture(_module, inputs):
        holder.append(inputs[0])
    tick = time.perf_counter()
    handle = model.point.register_forward_pre_hook(capture)
    hook_host_seconds = time.perf_counter() - tick
    try:
        prediction = model(image)
    finally:
        tick = time.perf_counter()
        handle.remove()
        hook_host_seconds += time.perf_counter() - tick
    if len(holder) != 1 or holder[0].shape != (len(image), 1280):
        raise ValueError("the original point must consume one Bx1280 semantic input")
    h = holder.pop()
    events = None
    if cuda_events is not None:
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events[0].record()
    with torch.autocast(device_type=image.device.type, enabled=False):
        conditioning = y.float().detach().reshape(-1, 1)
        critic_input = torch.cat((reverse_gradient(h.float(), alpha), conditioning), dim=1)
        logits = model.critic(critic_input).squeeze(-1)
        primitive = F.smooth_l1_loss(prediction.float(), y.float(), beta=.05, reduction="mean")
        bce = F.binary_cross_entropy_with_logits(logits, Z.float(), reduction="mean")
        forward_sum = primitive + bce
    if events is not None:
        events[1].record()
        cuda_events.append(events)
    return {"prediction": prediction, "primitive": primitive, "critic_BCE": bce,
        "forward_sum": forward_sum, "logits": logits, "hook_host_seconds": hook_host_seconds,
        "semantic_dtype": str(h.dtype), "critic_input_dtype": str(critic_input.dtype),
        "critic_input_shape": list(critic_input.shape), "conditioning_requires_grad": conditioning.requires_grad}


class SourceViewDataset:
    def __init__(self, base, bindings):
        self.base, self.bindings, self.rows = base, bindings, base.rows
        if len(self.rows) != 5496 or [str(row["sample_id"]) for row in self.rows] != [r["sample_id"] for r in bindings]:
            raise ValueError("source view tags are not bound to this exact training order")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample, saved = self.base[index], self.bindings[index]
        if str(sample["sample_id"]) != saved["sample_id"] or float(sample["y"]) != saved["y"]:
            raise ValueError("production training sample no longer matches its admitted view-target binding")
        return {**sample, "Z": torch.tensor(saved["Z"], dtype=torch.float32)}


def datasets(home, contract):
    from experiments.geoprr_v2_high_resolution_direct import ScalarRGBDataset
    prepared = contract["prepared"]
    packet = home / PREPARED_RELATIVE
    train = ScalarRGBDataset(packet / MANIFESTS["train"], data_root=home / "datasets", augment=True)
    source_id = ScalarRGBDataset(packet / MANIFESTS["id"], data_root=home / "datasets", augment=False)
    if train.rows != contract["train_rows"] or source_id.rows != prepared["source_contract"]["populations"]["id"]:
        raise ValueError("source train/ID manifest differs from the saved prepared contract")
    return SourceViewDataset(train, contract["bindings"]), source_id


def create_model(*, pretrained):
    from experiments.geoprr_v2_high_resolution_direct import create_model as original_create
    return ConditionalViewReader(original_create(pretrained=pretrained))


def model_metadata(model):
    from experiments.geoprr_v2_high_resolution_direct import model_metadata as original_metadata
    return {"reader": original_metadata(model.reader),
        "parameters": sum(p.numel() for p in model.parameters()), "parameter_tensors": len(list(model.parameters())),
        "state_tensors": len(model.state_dict()), "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
        "critic_parameter_tensors": len(list(model.critic.parameters())), "critic_state_tensors": len(model.critic.state_dict()),
        "BN_counter_tensors": sum(key.endswith("num_batches_tracked") for key in model.state_dict()),
        "public_forward_inputs": ["image"], "alpha_state_tensors": 0}


def validate_model_metadata(saved):
    keys = ("parameters", "parameter_tensors", "state_tensors", "critic_parameters", "critic_parameter_tensors", "critic_state_tensors", "BN_counter_tensors")
    if tuple(saved[key] for key in keys) != (8517906, 237, 468, 82113, 4, 4, 77):
        raise ValueError("new reader plus critic architecture differs from its frozen expected counts")


def critic_counter(model):
    counts = {"critic_calls": 0, "critic_examples": 0}
    def count(_module, inputs, _output):
        counts["critic_calls"] += 1
        counts["critic_examples"] += len(inputs[0])
    return counts, model.critic.register_forward_hook(count)


def phase_cost(device, started):
    torch.cuda.synchronize(device)
    return {"seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device), "peak_cpu_memory_bytes": peak_cpu_bytes(),
        "individual_point_forward_seconds": None}


def runtime(device, phases, prior=0.):
    return {"module_entry_seconds": time.perf_counter() - PROCESS_STARTED,
        "checkpointed_module_process_seconds": prior + time.perf_counter() - PROCESS_STARTED,
        "prior_checkpointed_module_seconds": prior,
        "boundary": "this child module entry before torch import; excludes final JSON and downtime; not whole pipeline",
        "device": str(device), "GPU": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "phases": phases, "peak_cpu_memory_bytes": peak_cpu_bytes(),
        "individual_point_forward_seconds": None, "parent_supervisor_boundary": "runner screen_summary.json"}


def initial_totals():
    result = dict.fromkeys(("train_batches", "train_images", "train_CNN_calls", "train_point_calls", "train_critic_calls",
        "train_critic_examples", "combined_backward_attempts", "CNN_backward_calls", "point_backward_calls", "critic_backward_calls",
        "optimizer_updates", "amp_skipped_updates", "evaluation_images", "evaluation_CNN_calls", "evaluation_point_calls", "evaluation_critic_calls"), 0)
    result.update(training_seconds=0., critic_FP32_and_BCE_cuda_seconds=0., hook_registration_removal_host_seconds=0.)
    return result


def validate_training_totals(totals, epoch):
    expected = {"train_batches": 86 * epoch, "train_images": 5496 * epoch, "train_CNN_calls": 86 * epoch,
        "train_point_calls": 86 * epoch, "train_critic_calls": 86 * epoch, "train_critic_examples": 5496 * epoch,
        "combined_backward_attempts": 86 * epoch, "CNN_backward_calls": 86 * epoch,
        "point_backward_calls": 86 * epoch, "critic_backward_calls": 86 * epoch}
    if any(totals[key] != value for key, value in expected.items()) or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 86 * epoch:
        raise ValueError("training totals differ from the exact complete-epoch exposure")


def validate_resume(saved, contract, arm, optimizer_contract=None, scaler_contract=None):
    required = {"protocol", "config", "arm", "alpha", "source_contract", "source_metadata_contract", "model", "model_metadata",
        "initialization", "optimizer", "scaler", "rng", "history", "totals", "epoch", "phase", "phases", "checkpointed_module_seconds"}
    if not required <= set(saved):
        raise ValueError("resume requires complete new reader/critic optimizer/scaler/RNG/history/totals")
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["arm"] != arm or saved["alpha"] != ARMS[arm]
            or saved["source_contract"] != contract["prepared"]["source_contract"] or saved["source_metadata_contract"] != contract
            or saved["phase"] != "training" or type(saved["epoch"]) is not int or not 1 <= saved["epoch"] <= 15):
        raise ValueError("resume protocol/config/arm/alpha/source/complete-epoch boundary differs")
    epoch, history, totals = saved["epoch"], saved["history"], saved["totals"]
    if [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("resume history lacks every completed epoch")
    for row in history:
        if (row["learning_rate"] != .0003 or row["train_images"] != 5496 or row["train_batches"] != 86
                or row["last_batch_size"] != 56 or row["Z_counts"] != [2748, 2748]
                or row["critic_total"] != 5496 or not 0 <= row["critic_correct"] <= 5496
                or row["critic_online_accuracy"] != row["critic_correct"] / 5496
                or row["validation_calls"] != 0 or row["optimizer_updates"] + row["amp_skipped_updates"] != 86
                or any(row[key] != 86 for key in ("CNN_calls", "point_calls", "critic_calls", "combined_backward_attempts",
                    "CNN_backward_calls", "point_backward_calls", "critic_backward_calls"))
                or any(not math.isfinite(row[key]) for key in ("primitive_smooth_l1", "critic_BCE", "forward_numerical_sum"))):
            raise ValueError("resume history changed a frozen training or online critic boundary")
    validate_training_totals(totals, epoch)
    if (any(totals[key] for key in ("evaluation_images", "evaluation_CNN_calls", "evaluation_point_calls", "evaluation_critic_calls"))
            or totals["optimizer_updates"] != sum(r["optimizer_updates"] for r in history)
            or totals["amp_skipped_updates"] != sum(r["amp_skipped_updates"] for r in history)
            or totals["training_seconds"] != sum(r["cost"]["seconds"] for r in history)
            or totals["critic_FP32_and_BCE_cuda_seconds"] != sum(r["critic_FP32_and_BCE_cuda_seconds"] for r in history)
            or totals["hook_registration_removal_host_seconds"] != sum(r["hook_registration_removal_host_seconds"] for r in history)):
        raise ValueError("resume totals do not match saved complete history")
    critic_keys = {"critic.0.weight", "critic.0.bias", "critic.2.weight", "critic.2.bias"}
    if (len(saved["model"]) != 468 or not critic_keys <= set(saved["model"])
            or set(saved["rng"]) != {"python", "numpy", "torch", "cuda"}
            or not {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"} <= set(saved["scaler"])
            or not saved["optimizer"].get("param_groups")):
        raise ValueError("resume is missing the new 468-state model/critic or full stochastic optimizer state")
    validate_model_metadata(saved["model_metadata"])
    if optimizer_contract is not None and saved["optimizer"]["param_groups"] != optimizer_contract["param_groups"]:
        raise ValueError("resume fixed optimizer contract differs")
    if scaler_contract is not None and any(saved["scaler"][key] != scaler_contract[key] for key in ("growth_factor", "backoff_factor", "growth_interval")):
        raise ValueError("resume default scaler policy differs")


def guard_arm_output(output):
    if not output.exists():
        return False
    if ((output / "completed.json").exists() or not (output / "latest.pt").is_file()
            or (output / "active_epoch.json").exists() or any(output.glob("*_active.json"))):
        raise FileExistsError("complete or partial arm cannot be overwritten; resume needs intact complete-epoch latest")
    return True


def read_scalar_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("target", "absolute_error"):
            row[key] = float(row[key])
        row["prediction"] = None if row["prediction"] == "" else float(row["prediction"])
        for key in ("valid", "input_valid"):
            if row[key] not in ("True", "False"):
                raise ValueError("saved scalar validity differs")
            row[key] = row[key] == "True"
        for key in ("invalid_mass", "outside_arc_mass"):
            row[key] = None if row[key] == "" else float(row[key])
    return rows


def load_evaluation(output, name, contract, arm, order):
    from experiments.geoprr_v2_high_resolution_direct import summarize
    paths = [output / f"{name}_{suffix}" for suffix in ("predictions.csv", "groups.csv", "summary.json")]
    marker = output / f"{name}_active.json"
    if marker.exists():
        raise FileExistsError("partial scalar evaluation cannot be forwarded again")
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileExistsError("CSV without a complete summary cannot be forwarded again")
    report, rows = load_json(paths[2]), read_scalar_csv(paths[0])
    if (report.get("protocol") != PROTOCOL or report.get("config") != CONFIG or report.get("arm") != arm
            or report.get("alpha") != ARMS[arm] or report.get("source_metadata_contract") != contract
            or report.get("condition_order") != list(order) or report.get("epoch") != 15
            or len(rows) != 292 or len({row["group_id"] for row in rows}) != 73
            or [row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in contract["prepared"]["source_contract"]["populations"]["id"]]
            or report["summary"] != summarize(rows)):
        raise ValueError("saved scalar observation differs from this new exact last15 arm")
    return rows, report


def evaluate(model, source_id, device, output, name, contract, arm):
    from experiments.geoprr_v2_cyclic_color_control import ChannelOrderDataset
    from experiments.geoprr_v2_high_resolution_direct import evaluate_population, point_counter
    order = NATIVE_ORDER if name == "source_id" else CF_ORDER
    saved = load_evaluation(output, name, contract, arm, order)
    if saved is not None:
        return saved
    write_json(output / f"{name}_active.json", {"protocol": PROTOCOL, "arm": arm, "epoch": 15})
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    points, point_hook = point_counter(model)
    critics, critic_hook = critic_counter(model)
    try:
        rows, cost = evaluate_population(model, ChannelOrderDataset(source_id, order=order), device)
    finally:
        point_hook.remove()
        critic_hook.remove()
    if (len(rows) != 292 or cost["counts"]["cnn_forward_calls"] != 5 or cost["counts"]["cnn_image_forwards"] != 292
            or points != {"point_forward_calls": 5, "point_image_forwards": 292} or critics != {"critic_calls": 0, "critic_examples": 0}):
        raise RuntimeError("source evaluation differs from292/5 or executed the source-training critic")
    cost.update(point_counts=points, critic_counts=critics, phase_cost=phase_cost(device, tick))
    report = _save_evaluation(output, name, rows, cost)
    report.update(protocol=PROTOCOL, config=CONFIG, arm=arm, alpha=ARMS[arm], source_metadata_contract=contract,
        condition_order=list(order), epoch=15, critic_execution=False)
    write_json(output / f"{name}_summary.json", report)
    (output / f"{name}_active.json").unlink()
    return rows, report


def reference_rows(home, contract, name):
    saved = load_json(home / REFERENCE_RELATIVE / "completed.json")
    rows = read_scalar_csv(home / REFERENCE_RELATIVE / f"{name}_predictions.csv")
    summary = saved[name]["summary"]
    frozen = REFERENCE[name]
    if (saved["protocol"] != "geoprr_v2_matched_identity_source_control_v1" or saved["status"] != "complete"
            or saved["epochs"] != 15 or saved["source_contract"] != contract["prepared"]["source_contract"]
            or summary["samples"] != 292 or len(rows) != 292 or len({r["sample_id"] for r in rows}) != 292
            or any(summary[key] != frozen[key] for key in ("nmae", "tail10_nmae", "maximum_error"))
            or 292 - summary["count_at_5_percent_fs"] != frozen["over5"]
            or sum(r["valid"] and r["absolute_error"] <= .01 for r in rows) != frozen["accurate1"]):
        raise ValueError("frozen identity15 saved reference differs; no old model is evaluated")
    return rows


def paired_comparison(candidate, reference):
    if len(candidate) != 292 or len(reference) != 292:
        raise ValueError("joint comparison requires the complete292 denominator")
    thresholds = {str(t): {"rescues": 0, "new_errors": 0, "rescued_ids": [], "new_error_ids": []} for t in (1, 2, 5)}
    protected, newly_over5, better, worse, unchanged = [], [], 0, 0, 0
    for after, before in zip(candidate, reference, strict=True):
        if any(after[key] != before[key] for key in ("sample_id", "group_id", "target")):
            raise ValueError("paired rows differ in identity, group or consumed target")
        better += after["absolute_error"] < before["absolute_error"]
        worse += after["absolute_error"] > before["absolute_error"]
        unchanged += after["absolute_error"] == before["absolute_error"]
        if before["valid"] and before["absolute_error"] <= .01:
            protected.append(before["sample_id"])
            if after["absolute_error"] > .05:
                newly_over5.append(after["sample_id"])
        for threshold in (1, 2, 5):
            old = before["valid"] and before["absolute_error"] <= threshold / 100
            new = after["valid"] and after["absolute_error"] <= threshold / 100
            if new and not old:
                thresholds[str(threshold)]["rescues"] += 1
                thresholds[str(threshold)]["rescued_ids"].append(after["sample_id"])
            if old and not new:
                thresholds[str(threshold)]["new_errors"] += 1
                thresholds[str(threshold)]["new_error_ids"].append(after["sample_id"])
    private = {"protected_ids": protected, "new_over5_on_protected_ids": newly_over5, "thresholds": thresholds}
    public = {"paired_images": 292, "paired_groups": len({r["group_id"] for r in candidate}),
        "error_rescues": better, "error_regressions": worse, "error_unchanged": unchanged,
        "protected_accurate1_count": len(protected), "new_over5_on_protected": len(newly_over5),
        "thresholds_percent_fs": {t: {key: value[key] for key in ("rescues", "new_errors")} for t, value in thresholds.items()}}
    return public, private


def view_gate(summary, reference_summary, paired, *, strict_tail):
    predicates = {"mean": summary["nmae"] <= reference_summary["nmae"],
        "tail": summary["tail10_nmae"] < reference_summary["tail10_nmae"] if strict_tail else summary["tail10_nmae"] <= reference_summary["tail10_nmae"],
        "maximum": summary["maximum_error"] <= reference_summary["maximum_error"],
        "over5": 292 - summary["count_at_5_percent_fs"] <= reference_summary["over5"],
        "accurate1_protected": paired["new_over5_on_protected"] == 0}
    return {"passed": all(predicates.values()), "predicates": predicates, "strict_tail": strict_tail, "paired": paired}


def critic_effect(control, reversal):
    predicates = {"reversal_accuracy_interval": .4 <= reversal <= .6,
        "distance_reduction": abs(control - .5) - abs(reversal - .5) >= .10}
    return {"passed": all(predicates.values()), "predicates": predicates, "control_online_accuracy": control,
        "reversal_online_accuracy": reversal, "absolute_distance_reduction": abs(control - .5) - abs(reversal - .5),
        "scope": "last-epoch source online operational critic effect; not conditional independence or causality"}


def combined_counts(arms):
    result = {"CNN_images": 0, "CNN_calls": 0, "point_examples": 0, "point_calls": 0,
        "critic_examples": 0, "critic_calls": 0, "combined_backward_attempts": 0,
        "optimizer_updates": 0, "amp_skipped_updates": 0, "source03_calls": 0, "external_calls": 0}
    for report, cf in arms:
        totals = report["totals"]
        extra = 292 if cf is not None else 0
        result["CNN_images"] += totals["train_images"] + 292 + extra
        result["CNN_calls"] += totals["train_CNN_calls"] + 5 + (5 if cf is not None else 0)
        result["point_examples"] += totals["train_images"] + 292 + extra
        result["point_calls"] += totals["train_point_calls"] + 5 + (5 if cf is not None else 0)
        for target, source in (("critic_examples", "train_critic_examples"), ("critic_calls", "train_critic_calls"),
                ("combined_backward_attempts", "combined_backward_attempts"), ("optimizer_updates", "optimizer_updates"),
                ("amp_skipped_updates", "amp_skipped_updates")):
            result[target] += totals[source]
    return result


def preflight(home, output, root):
    import cv2
    import numpy as np
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_cyclic_color_control import MEAN, STD, preflight_indices
    from experiments.geoprr_v2_data import _resolve_image, crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter, point_counter, gradient_evidence
    from experiments.train_geoprr_v2 import device_batch, seed_everything

    contract = admission(home, root)
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("preflight has complete or partial outputs; no second preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "started.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "started"})
    device, started = torch.device("cuda"), time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training, source_id = datasets(home, contract)
    indices = preflight_indices(contract["prepared"])
    train_samples = [training[index] for index in indices["train"]]
    if ([sample["Z"].item() for sample in train_samples] != [0., 1., 0., 1.]
            or any(contract["bindings"][index]["slot"] != 0 for index in indices["train"])):
        raise ValueError("preflight requires the frozen canonical01/02 original/roll slot0 training quartet")
    train_batch = device_batch(default_collate(train_samples), device)
    validation_batches = []
    for subset in indices["id"]:
        samples = [source_id[index] for index in subset]
        for index, sample in zip(subset, samples, strict=True):
            row = source_id.rows[index]
            image = cv2.imread(str(_resolve_image(row, source_id.base.manifest_root, source_id.base.dataset_root)))
            roi, _ = crop_and_resize(image, None if "roi_path" in row else row.get("metadata", {}).get("dial_bbox"), 512)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            expected = (expected - expected.new_tensor(MEAN)[:, None, None]) / expected.new_tensor(STD)[:, None, None]
            if sample["input_failure"] or not torch.equal(sample["image"], expected):
                raise ValueError("source-ID preflight input differs from the unchanged native RGB pipeline")
        validation_batches.append(device_batch(default_collate(samples), device))
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    validate_model_metadata(metadata)
    initial = model.reader.initialization_evidence
    original = {key: value.detach().clone() for key, value in model.named_parameters()}
    counters = {key: value.clone() for key, value in model.state_dict().items() if key.endswith("num_batches_tracked")}
    reloaded = model.load_state_dict(deepcopy(model.state_dict()), strict=True)
    strict_reload = not reloaded.missing_keys and not reloaded.unexpected_keys
    cnn, cnn_hook = encoder_counter(model)
    points, point_hook = point_counter(model)
    critics, critic_hook = critic_counter(model)
    observed_inputs, observed_pool, observed_point, observed_critic = [], [], [], []
    backs = {"CNN": [], "point": [], "critic": []}
    def input_shape(_module, inputs):
        observed_inputs.append({"shape": list(inputs[0].shape), "dtype": str(inputs[0].dtype)})
    def pool_shape(_module, inputs, value):
        observed_pool.append({"before": list(inputs[0].shape), "after": list(value.shape)})
    def point_shape(_module, inputs, value):
        observed_point.append({"h": list(inputs[0].shape), "h_dtype": str(inputs[0].dtype),
            "point": list(value.shape), "point_dtype": str(value.dtype)})
    def critic_shape(_module, inputs, value):
        observed_critic.append({"input": list(inputs[0].shape), "input_dtype": str(inputs[0].dtype),
            "logits": list(value.shape), "logits_dtype": str(value.dtype)})
    handles = [cnn_hook, point_hook, critic_hook, model.encoder.register_forward_pre_hook(input_shape),
        model.encoder.global_pool.register_forward_hook(pool_shape), model.point.register_forward_hook(point_shape),
        model.critic.register_forward_hook(critic_shape),
        model.encoder.conv_stem.weight.register_hook(lambda g: backs["CNN"].append(True)),
        model.point.weight.register_hook(lambda g: backs["point"].append(True)),
        model.critic[0].weight.register_hook(lambda g: backs["critic"].append(True))]
    phases = {"initialization": phase_cost(device, started)}
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = training_forward(model, train_batch["image"], train_batch["y"], train_batch["Z"], .05)
        (result["forward_sum"] * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero = gradient_evidence(model.reader)
        critic_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.critic.parameters())
        critic_weight_nonzero = {str(index): bool(model.critic[index].weight.grad.abs().sum() > 0) for index in (0, 2)}
        if not all(torch.equal(model.state_dict()[key], value + 1) for key, value in counters.items()):
            raise ValueError("all77 original BN counters must increment once in training")
        model.eval()
        eval_buffers = {key: value.clone() for key, value in model.named_buffers()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in validation_batches:
                prediction = model(batch["image"])
                if (prediction.shape != (2,) or prediction.dtype != torch.float32
                        or not bool(torch.isfinite(prediction).all()) or not bool(((prediction >= 0) & (prediction <= 1)).all())):
                    raise ValueError("native scalar evaluation output differs")
        buffers_unchanged = all(torch.equal(value, eval_buffers[key]) for key, value in model.named_buffers())
        parameters_unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        for handle in handles:
            handle.remove()
    expected_inputs = [{"shape": [n, 3, 512, 512], "dtype": "torch.float32"} for n in (4, 2, 2)]
    expected_pool = [{"before": [n, 960, 16, 16], "after": [n, 960, 1, 1]} for n in (4, 2, 2)]
    expected_point = [{"h": [n, 1280], "h_dtype": "torch.float16", "point": [n, 1], "point_dtype": "torch.float16"} for n in (4, 2, 2)]
    expected_critic = [{"input": [4, 1281], "input_dtype": "torch.float32", "logits": [4, 1], "logits_dtype": "torch.float32"}]
    if (cnn != {"cnn_forward_calls": 3, "cnn_image_forwards": 8}
            or points != {"point_forward_calls": 3, "point_image_forwards": 8}
            or critics != {"critic_calls": 1, "critic_examples": 4} or any(len(value) != 1 for value in backs.values())
            or observed_inputs != expected_inputs or observed_pool != expected_pool or observed_point != expected_point
            or observed_critic != expected_critic or result["semantic_dtype"] != "torch.float16"
            or result["prediction"].shape != (4,) or result["prediction"].dtype != torch.float32
            or not bool(torch.isfinite(result["prediction"]).all())
            or not bool(((result["prediction"] >= 0) & (result["prediction"] <= 1)).all())
            or result["critic_BCE"].dtype != torch.float32
            or result["conditioning_requires_grad"] or not strict_reload or not finite or not critic_finite
            or not all(nonzero.values()) or not all(critic_weight_nonzero.values())
            or not parameters_unchanged or not buffers_unchanged or not bool(torch.isfinite(result["forward_sum"]))):
        raise ValueError("the one zero-update conditional source preflight failed")
    phases["preflight"] = phase_cost(device, tick)
    write_json(output, {"protocol": PROTOCOL, "config": CONFIG, "status": "passed", "alpha": .05,
        "source_metadata_contract": contract, "model_metadata": metadata, "initialization": initial,
        "counts": PREFLIGHT_COUNTS, "source_BN_counter_tensors_checked": 77, "training_BN_increment": 1,
        "eval_BN_increment": 0, "parameter_tensors_unchanged_without_optimizer": 237,
        "strict_model_reload": strict_reload, "full_state_tensors": 468, "all_reader_gradients_finite": finite,
        "reader_gradient_groups_nonzero": nonzero, "critic_gradients_finite": critic_finite,
        "critic_first_last_weight_gradients_nonzero": critic_weight_nonzero, "conditioning_detached": True,
        "observed_encoder_inputs": observed_inputs, "observed_pool": observed_pool,
        "observed_point": observed_point, "observed_critic": observed_critic,
        "source_ID_pipeline_identity": True, "fixed_backward_loss_scale": 128.,
        "primitive_smooth_l1": float(result["primitive"].detach()), "critic_BCE": float(result["critic_BCE"].detach()),
        "forward_numerical_sum": float(result["forward_sum"].detach()), "formal_fresh_rebuild_required": True,
        "formal_fresh_default_GradScaler_required": True, "runtime": runtime(device, phases)})


def witness(home, contract):
    saved = load_json(home / PREFLIGHT_RELATIVE)
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["status"] != "passed"
            or saved["source_metadata_contract"] != contract or saved["counts"] != PREFLIGHT_COUNTS):
        raise ValueError("formal arms require their exact one completed zero-update preflight")


def train_arm(home, root, arm):
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter, point_counter
    from experiments.train_geoprr_v2 import device_batch, make_loader, seed_everything, rng_state, restore_rng, save_checkpoint

    contract = admission(home, root)
    witness(home, contract)
    if arm == "reversal":
        control = load_json(root / "detached_control/completed.json")
        cf = load_json(root / "detached_control/counterfactual_completed.json")
        if (not control["prerequisite_passed"] or control["protocol"] != PROTOCOL or control["config"] != CONFIG
                or control["arm"] != "detached_control" or control["alpha"] != 0.
                or control["source_metadata_contract"] != contract or cf["status"] != "complete"
                or cf["protocol"] != PROTOCOL or cf["config"] != CONFIG or cf["arm"] != "detached_control"
                or cf["alpha"] != 0. or cf["source_metadata_contract"] != contract):
            raise ValueError("fresh reversal requires the learned control and its one completed CF observation")
    output = root / arm
    resuming = guard_arm_output(output)
    if not resuming:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "started.json", {"protocol": PROTOCOL, "config": CONFIG, "arm": arm, "alpha": ARMS[arm]})
    device, tick = torch.device("cuda"), time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training, source_id = datasets(home, contract)
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    validate_model_metadata(metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    base = {"protocol": PROTOCOL, "config": CONFIG, "arm": arm, "alpha": ARMS[arm],
        "source_contract": contract["prepared"]["source_contract"], "source_metadata_contract": contract,
        "model_metadata": metadata, "initialization": model.reader.initialization_evidence,
        "fresh_default_GradScaler": True, "preflight_state_reused": False}
    history, totals, start, prior = [], initial_totals(), 0, 0.
    phases = {"initialization": phase_cost(device, tick)}
    if resuming:
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, contract, arm, optimizer.state_dict(), scaler.state_dict())
        if saved["model_metadata"] != metadata or saved["initialization"] != base["initialization"]:
            raise ValueError("resume initialization evidence or model metadata differs")
        if (not (output / "history.jsonl").is_file()
                or (output / "history.jsonl").read_text() != "".join(json.dumps(row, allow_nan=False) + "\n" for row in saved["history"])):
            raise ValueError("resume history artifact differs from complete checkpoint history")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        history, totals, start, prior, phases = saved["history"], saved["totals"], saved["epoch"], saved["checkpointed_module_seconds"], saved["phases"]
        phases[f"resume_initialization_after_epoch_{start}"] = phase_cost(device, tick)
    else:
        write_json(output / "run.json", base)
    for epoch in range(start + 1, 16):
        write_json(output / "active_epoch.json", {"protocol": PROTOCOL, "arm": arm, "epoch": epoch, "last_complete_epoch": epoch - 1})
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        model.train()
        loader = make_loader(training, batch_size=64, workers=4, shuffle=True, seed=20262020 + epoch, device=device)
        cnn, cnn_hook = encoder_counter(model)
        points, point_hook = point_counter(model)
        critics, critic_hook = critic_counter(model)
        backs = {"CNN": [], "point": [], "critic": []}
        handles = [cnn_hook, point_hook, critic_hook,
            model.encoder.conv_stem.weight.register_hook(lambda g: backs["CNN"].append(True)),
            model.point.weight.register_hook(lambda g: backs["point"].append(True)),
            model.critic[0].weight.register_hook(lambda g: backs["critic"].append(True))]
        batches = updates = skips = correct = total = 0
        last_batch, Z_counts = None, [0, 0]
        primitive_sum = bce_sum = forward_sum = error_sum = hook_seconds = 0.
        events = []
        try:
            for raw in loader:
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = training_forward(model, batch["image"], batch["y"], batch["Z"], ARMS[arm], cuda_events=events)
                if not bool(torch.isfinite(result["forward_sum"])):
                    raise FloatingPointError("nonfinite source primitive or critic loss")
                # Online counts include this attempted presentation even when AMP later skips its update.
                correct += int(((result["logits"].detach() >= 0) == batch["Z"].bool()).sum())
                last_batch = len(batch["y"])
                total += last_batch
                Z_counts[0] += int((batch["Z"] == 0).sum())
                Z_counts[1] += int((batch["Z"] == 1).sum())
                scaler.scale(result["forward_sum"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                batches += 1
                updates += int(not skipped)
                skips += int(skipped)
                primitive_sum += float(result["primitive"].detach()) * last_batch
                bce_sum += float(result["critic_BCE"].detach()) * last_batch
                forward_sum += float(result["forward_sum"].detach()) * last_batch
                error_sum += float((result["prediction"].detach() - batch["y"]).abs().sum())
                hook_seconds += result["hook_host_seconds"]
        finally:
            for handle in handles:
                handle.remove()
        cost = phase_cost(device, tick)
        critic_seconds = sum(begin.elapsed_time(end) for begin, end in events) / 1000
        if ((batches, last_batch, total, Z_counts, updates + skips) != (86, 56, 5496, [2748, 2748], 86)
                or cnn != {"cnn_forward_calls": 86, "cnn_image_forwards": 5496}
                or points != {"point_forward_calls": 86, "point_image_forwards": 5496}
                or critics != {"critic_calls": 86, "critic_examples": 5496} or any(len(value) != 86 for value in backs.values())):
            raise RuntimeError("source arm epoch differs from5496/86 exact source exposure")
        record = {"epoch": epoch, "learning_rate": .0003, "train_images": 5496, "train_batches": 86,
            "last_batch_size": 56, "CNN_calls": 86, "point_calls": 86, "critic_calls": 86,
            "combined_backward_attempts": batches, "CNN_backward_calls": len(backs["CNN"]),
            "point_backward_calls": len(backs["point"]), "critic_backward_calls": len(backs["critic"]),
            "optimizer_updates": updates, "amp_skipped_updates": skips, "Z_counts": Z_counts,
            "critic_correct": correct, "critic_total": total, "critic_online_accuracy": correct / total,
            "primitive_smooth_l1": primitive_sum / total, "critic_BCE": bce_sum / total,
            "forward_numerical_sum": forward_sum / total, "source_training_nmae": error_sum / total,
            "forward_sum_scope": "numerical sum; reversed encoder is not minimizing this common objective",
            "critic_FP32_and_BCE_cuda_seconds": critic_seconds,
            "critic_timing_scope": "concat, critic, primitive, BCE and sum inside disabled-autocast region",
            "hook_registration_removal_host_seconds": hook_seconds, "validation_calls": 0, "cost": cost}
        increments = {"train_batches": batches, "train_images": total, "train_CNN_calls": cnn["cnn_forward_calls"],
            "train_point_calls": points["point_forward_calls"], "train_critic_calls": critics["critic_calls"],
            "train_critic_examples": critics["critic_examples"], "combined_backward_attempts": batches,
            "CNN_backward_calls": len(backs["CNN"]), "point_backward_calls": len(backs["point"]),
            "critic_backward_calls": len(backs["critic"]), "optimizer_updates": updates, "amp_skipped_updates": skips,
            "training_seconds": cost["seconds"], "critic_FP32_and_BCE_cuda_seconds": critic_seconds,
            "hook_registration_removal_host_seconds": hook_seconds}
        for key, value in increments.items():
            totals[key] += value
        history.append(record)
        phases[f"train_epoch_{epoch}"] = cost
        saved = base | {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "epoch": epoch, "history": history, "totals": dict(totals), "phase": "training",
            "phases": dict(phases), "checkpointed_module_seconds": prior + time.perf_counter() - PROCESS_STARTED}
        save_checkpoint(output / "latest.pt", saved)
        (output / "history.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
        (output / "active_epoch.json").unlink()
        print(json.dumps({"event": "epoch_complete", "arm": arm, "epoch": epoch, "train_images": total,
            "combined_backward_attempts": batches, "optimizer_updates": updates, "amp_skips": skips,
            "critic_correct": correct, "critic_total": total, "Z_counts": Z_counts}), flush=True)
    validate_training_totals(totals, 15)
    rows, report = evaluate(model, source_id, device, output, "source_id", contract, arm)
    gate, _baseline = learning_gate(rows)
    phases["source_id"] = report["cost"]["phase_cost"]
    totals.update(evaluation_images=292, evaluation_CNN_calls=5, evaluation_point_calls=5, evaluation_critic_calls=0)
    joint = None
    prerequisite = gate["passed"] and history[-1]["critic_correct"] / 5496 >= .65
    native_passed = False
    if arm == "reversal":
        baseline_rows = reference_rows(home, contract, "source_id")
        identity_pair, identity_private = paired_comparison(rows, baseline_rows)
        control_observation = load_evaluation(root / "detached_control", "source_id", contract, "detached_control", NATIVE_ORDER)
        if control_observation is None:
            raise ValueError("reversal native requires the exact learned detached-control observation")
        control_rows, control_report = control_observation
        control_pair, control_private = paired_comparison(rows, control_rows)
        control_limits = {**control_report["summary"], "over5": 292 - control_report["summary"]["count_at_5_percent_fs"]}
        joint = {"identity": view_gate(report["summary"], REFERENCE["source_id"], identity_pair, strict_tail=True),
            "detached_control": view_gate(report["summary"], control_limits, control_pair, strict_tail=True)}
        native_passed = gate["passed"] and all(value["passed"] for value in joint.values())
        write_json(output / "source_id_paired_private.json", {"identity": identity_private, "detached_control": control_private})
    completed = base | {"status": "complete", "epochs": 15, "native_learning_gate": gate,
        "prerequisite_passed": bool(prerequisite) if arm == "detached_control" else None,
        "critic_prerequisite_passed": history[-1]["critic_correct"] / 5496 >= .65 if arm == "detached_control" else None,
        "native_joint_passed": bool(native_passed) if arm == "reversal" else None, "native_joint": joint,
        "last_epoch_critic": {key: history[-1][key] for key in ("critic_correct", "critic_total", "critic_online_accuracy", "Z_counts", "critic_BCE")},
        "source_id": report, "source_counterfactual": None, "totals": totals,
        "checkpoint": "last.pt", "runtime": runtime(device, phases, prior)}
    # last.pt is the same completed last15 training state; evaluation performs no updates.
    saved.update(phase="evaluated", totals=totals, phases=phases)
    save_checkpoint(output / "last.pt", saved)
    write_json(output / "completed.json", completed)


def counterfactual(home, root, arm):
    from experiments.train_geoprr_v2 import seed_everything
    contract = admission(home, root)
    output = root / arm
    completed = load_json(output / "completed.json")
    eligible = completed["prerequisite_passed"] if arm == "detached_control" else completed["native_joint_passed"]
    if (completed["protocol"] != PROTOCOL or completed["config"] != CONFIG or completed["arm"] != arm
            or completed["alpha"] != ARMS[arm] or completed["source_metadata_contract"] != contract
            or not eligible or (output / "counterfactual_completed.json").exists()):
        raise ValueError("CF requires its exact eligible native last15, with no already completed CF")
    # A CSV or active marker without its complete matching summary never authorizes another forward.
    existing = load_evaluation(output, "source_counterfactual", contract, arm, CF_ORDER)
    device, tick = torch.device("cuda"), time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    phases = {}
    if existing is None:
        seed_everything(20262020)
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["arm"] != arm
                or saved["alpha"] != ARMS[arm] or saved["epoch"] != 15 or saved["phase"] != "evaluated"
                or saved["source_metadata_contract"] != contract or len(saved["model"]) != 468):
            raise ValueError("CF checkpoint differs from the exact completed last15 arm")
        model = create_model(pretrained=False).to(device)
        model.load_state_dict(saved["model"], strict=True)
        _training, source_id = datasets(home, contract)
        phases["initialization"] = phase_cost(device, tick)
        rows, report = evaluate(model, source_id, device, output, "source_counterfactual", contract, arm)
    else:
        rows, report = existing
    joint = None
    passed = None
    if arm == "reversal":
        identity_pair, identity_private = paired_comparison(rows, reference_rows(home, contract, "source_counterfactual"))
        control = load_evaluation(root / "detached_control", "source_counterfactual", contract, "detached_control", CF_ORDER)
        if control is None:
            raise ValueError("reversal CF requires the complete detached-control CF baseline")
        control_pair, control_private = paired_comparison(rows, control[0])
        limits = {**control[1]["summary"], "over5": 292 - control[1]["summary"]["count_at_5_percent_fs"]}
        joint = {"identity": view_gate(report["summary"], REFERENCE["source_counterfactual"], identity_pair, strict_tail=False),
            "detached_control": view_gate(report["summary"], limits, control_pair, strict_tail=True)}
        passed = all(value["passed"] for value in joint.values())
        write_json(output / "source_counterfactual_paired_private.json", {"identity": identity_private, "detached_control": control_private})
    phases["source_counterfactual"] = report["cost"]["phase_cost"]
    write_json(output / "counterfactual_completed.json", {"protocol": PROTOCOL, "config": CONFIG,
        "status": "complete", "arm": arm, "alpha": ARMS[arm], "source_metadata_contract": contract,
        "epoch": 15, "source_counterfactual": report, "CF_joint": joint, "CF_joint_passed": passed,
        "counts": {"CNN_images": 292, "CNN_calls": 5, "point_examples": 292, "point_calls": 5,
            "critic_calls": 0, "critic_examples": 0, "backward_calls": 0, "optimizer_updates": 0},
        "runtime": runtime(device, phases)})


def scalar_overview(report):
    summary = report["summary"]
    keys = ("samples", "nmae", "tail10_nmae", "median_error", "p90_error", "p95_error", "maximum_error",
        "count_at_1_percent_fs", "count_at_2_percent_fs", "count_at_5_percent_fs", "acc_at_1", "acc_at_2", "acc_at_5", "input_failures")
    return {key: summary[key] for key in keys}


def finalize(home, root):
    contract = admission(home, root)
    if (root / "completed.json").exists():
        raise FileExistsError("final source control summary is complete; reuse without another execution")
    control = load_json(root / "detached_control/completed.json")
    reports = {"detached_control": control, "reversal": None}
    cf_reports = {arm: None for arm in ARMS}
    for arm in ARMS:
        path = root / arm / "counterfactual_completed.json"
        if path.is_file():
            cf_reports[arm] = load_json(path)
    if control["prerequisite_passed"]:
        if cf_reports["detached_control"] is None:
            raise ValueError("learned-control prerequisite passed but its mandatory CF is absent")
        reports["reversal"] = load_json(root / "reversal/completed.json")
        if reports["reversal"]["native_joint_passed"] and cf_reports["reversal"] is None:
            raise ValueError("native joint passed but mandatory same-last15 reversal CF is absent")
        if not reports["reversal"]["native_joint_passed"] and cf_reports["reversal"] is not None:
            raise ValueError("native failure has an unauthorized reversal CF")
    elif cf_reports["detached_control"] is not None or (root / "reversal").exists():
        raise ValueError("failed learned-control prerequisite has unauthorized CF or reversal execution")
    for arm, report in reports.items():
        if report is None:
            continue
        if (report["protocol"] != PROTOCOL or report["config"] != CONFIG or report["arm"] != arm
                or report["alpha"] != ARMS[arm] or report["source_metadata_contract"] != contract
                or report["status"] != "complete" or report["epochs"] != 15):
            raise ValueError("final source stage protocol/config/source/arm differs")
        validate_training_totals(report["totals"], 15)
        if any(report["totals"][key] != value for key, value in {"evaluation_images": 292,
                "evaluation_CNN_calls": 5, "evaluation_point_calls": 5, "evaluation_critic_calls": 0}.items()):
            raise ValueError("native saved totals differ")
        cf = cf_reports[arm]
        if cf is not None and (cf["protocol"] != PROTOCOL or cf["config"] != CONFIG or cf["arm"] != arm
                or cf["alpha"] != ARMS[arm] or cf["source_metadata_contract"] != contract or cf["status"] != "complete"):
            raise ValueError("final CF stage differs from its exact same-last15 arm")
    reversal = reports["reversal"]
    effect, joint = None, None
    passed = False
    reason = "detached_control_learning_or_online_critic_prerequisite_failed"
    if reversal is not None:
        effect = critic_effect(control["last_epoch_critic"]["critic_online_accuracy"], reversal["last_epoch_critic"]["critic_online_accuracy"])
        joint = {"native": reversal["native_joint"], "native_learning_passed": reversal["native_learning_gate"]["passed"],
            "counterfactual": None if cf_reports["reversal"] is None else cf_reports["reversal"]["CF_joint"]}
        reading_passed = reversal["native_joint_passed"] and cf_reports["reversal"] is not None and cf_reports["reversal"]["CF_joint_passed"]
        passed = bool(reading_passed and effect["passed"])
        reason = "all_both_view_reading_and_critic_effect_passed" if passed else (
            "source_reading_passed_but_critic_effect_failed_mixed" if reading_passed else "source_reading_joint_failed")
    counts = combined_counts([(report, cf_reports[arm]) for arm, report in reports.items() if report is not None])
    expected = (82732, 1295, 82440, 1290) if reversal is None else (
        (166048, 2600, 164880, 2580) if cf_reports["reversal"] is not None else (165756, 2595, 164880, 2580))
    if tuple(counts[key] for key in ("CNN_images", "CNN_calls", "critic_examples", "critic_calls")) != expected:
        raise ValueError("conditional full execution differs from frozen aggregate budget")
    overviews = {}
    for arm, report in reports.items():
        overviews[arm] = None if report is None else {"alpha": ARMS[arm], "epochs": 15,
            "native": scalar_overview(report["source_id"]), "counterfactual": None if cf_reports[arm] is None else scalar_overview(cf_reports[arm]["source_counterfactual"]),
            "last_epoch_critic": report["last_epoch_critic"], "native_learning_passed": report["native_learning_gate"]["passed"],
            "optimizer_updates": report["totals"]["optimizer_updates"], "amp_skipped_updates": report["totals"]["amp_skipped_updates"]}
    write_json(root / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
        "verdict": "CONDITIONAL_VIEW_SOURCE_CONTROL_PASSED" if passed else "NOT_RETAINED", "completion_reason": reason,
        **overviews, "joint_acceptance": joint, "critic_effect": effect, "counts": counts,
        "source_metadata": load_json(root / "source_metadata/completed.json"), "preflight_counts_separate": PREFLIGHT_COUNTS,
        "runtime": {"finalize_child_module_entry_seconds": time.perf_counter() - PROCESS_STARTED,
            "boundary": "finalize child only; whole supervisor/screen/queue boundaries recorded separately",
            "all_executed_parent_processes_gone_verified": False, "GPU_empty_after_all_parents_verified": False},
        "scope": CONFIG["scope"], "automatic_promotion": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("metadata", "preflight", "detached_control", "control_cf", "reversal", "reversal_cf", "finalize"))
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.home / OUTPUT_RELATIVE
    from experiments.run_geoprr_v2_conditional_view_adversary_control import precheck
    precheck(args.home, root)
    if args.stage == "preflight":
        preflight(args.home, args.output or args.home / PREFLIGHT_RELATIVE, root)
        return
    output = args.output or root
    if args.stage == "metadata":
        metadata(args.home, output)
    elif args.stage in ARMS:
        train_arm(args.home, output, args.stage)
    elif args.stage in ("control_cf", "reversal_cf"):
        counterfactual(args.home, output, "detached_control" if args.stage == "control_cf" else "reversal")
    else:
        finalize(args.home, output)


if __name__ == "__main__":
    main()
