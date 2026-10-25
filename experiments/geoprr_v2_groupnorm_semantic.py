"""Train a source-only semantic scalar reader with per-image GroupNorm8."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import inspect
import json
import math
import os
from pathlib import Path
import time

PROCESS_STARTED = time.perf_counter()

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

from experiments.geoprr_v2_data import _resolve_image, crop_and_resize
from experiments.geoprr_v2_high_resolution_direct import (
    CONFIG as V26_CONFIG, RGB_INPUT_CONTRACT, SOURCE_RELATIVE, point_counter,
    save_source_rows, source_populations,
)
from experiments.geoprr_v2_roll_source import completion_reason
from experiments.geoprr_v2_semantic_tokens import selection_score
from experiments.geoprr_v2_semantic_direct import (
    ENCODER_NAME, SemanticDirectReader, encoder_counter, evaluate_population, save_csv, synchronize,
)
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
)


PROTOCOL = "geoprr_v2_groupnorm_semantic_source_only_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v31_groupnorm_semantic_20261025/source_only_training"
NORMALIZATION_CONTRACT = {
    "replacement": "all77 BatchNormAct2d replaced before optimizer construction by GroupNormAct2d",
    "groups": 8, "epsilon": 1e-5, "affine": True,
    "statistics": "instantaneous per-image grouped channel and spatial statistics in both train and eval",
    "batch_coupling": False, "running_statistics": False, "running_buffers": 0,
    "normalization_math": "float32 feature, affine and F.group_norm inside disabled autocast; cast result to input feature dtype before original drop then act",
    "outer_encoder_semantic_point_autocast": "float16", "scalar_output": "float32 sigmoid",
    "initialization": "all231 learned cached encoder tensors retained by direct equality;231 BN running buffers intentionally removed",
    "drop_activation_order": "original drop then original act", "pretrained_function_preserved": False,
    "target_dataset_statistic_fitting": False, "evaluation_parameter_updates": False,
}
CONFIG = deepcopy(V26_CONFIG)
CONFIG["protocol"] = PROTOCOL
CONFIG["model"].update(
    initialization="fresh cached ImageNet full encoder; replace all BN by GroupNorm8 retaining learned affine values; new scalar point",
    source_training_BN_updates=False, normalization=deepcopy(NORMALIZATION_CONTRACT))


class GroupNormMath(nn.Module):
    """Expose only the stateless float32 normalization operation."""

    def __init__(self, groups, eps):
        super().__init__()
        self.groups, self.eps = groups, eps

    def forward(self, feature, weight, bias):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return F.group_norm(feature, self.groups, weight, bias, self.eps)


class GroupNormAct2d(nn.GroupNorm):
    """Retain learned affine keys and the original post-normalization order."""

    def __init__(self, old):
        super().__init__(8, old.num_features, eps=old.eps, affine=True,
                         device=old.weight.device, dtype=old.weight.dtype)
        self.weight, self.bias = old.weight, old.bias
        self.math = GroupNormMath(self.num_groups, self.eps)
        self.drop, self.act = old.drop, old.act
        self.train(old.training)

    def forward(self, feature):
        normalized = self.math(feature.float(), self.weight.float(), self.bias.float()).to(feature.dtype)
        return self.act(self.drop(normalized))


def replace_batch_norm(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            if (type(child).__name__ != "BatchNormAct2d" or child.num_features % 8
                    or child.eps != 1e-5 or not child.affine):
                raise ValueError("cached encoder normalizers differ from the fixed GroupNorm8 conversion")
            setattr(module, name, GroupNormAct2d(child))
            count += 1
        else:
            count += replace_batch_norm(child)
    return count


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    reference = {key: value.detach().clone() for key, value in encoder.named_parameters()} if pretrained else None
    original_buffers = sum(1 for _ in encoder.buffers())
    converted = replace_batch_norm(encoder)
    model = SemanticDirectReader(encoder)
    metadata = model_metadata(model)
    if (converted, original_buffers, metadata["semantic_dim"], metadata["encoder_parameters"],
            metadata["encoder_state_tensors"], metadata["encoder_parameter_tensors"], metadata["parameters"],
            metadata["parameter_tensors"], metadata["state_tensors"], metadata["GroupNorm_modules"],
            metadata["BatchNorm_modules"], metadata["buffer_tensors"], metadata["semantic_conv_head_retained"],
            metadata["semantic_norm_head_retained"], metadata["all_parameters_trainable"]) != (
            77, 231, 1280, 8434512, 231, 231, 8435793, 233, 233, 77, 0, 0, True, True, True):
        raise ValueError("complete GroupNorm8 semantic encoder differs from the fixed architecture")
    if reference is not None:
        learned = dict(model.encoder.named_parameters())
        if set(learned) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in learned.items()):
            raise ValueError("GroupNorm8 conversion changed learned cached encoder tensors")
        model.initialization_evidence = {
            "encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
            "retained_learned_encoder_parameter_tensors": len(reference),
            "all_cached_learned_encoder_parameter_tensors_retained": True,
            "original_encoder_state_tensors": len(reference) + original_buffers,
            "intentionally_removed_BN_running_buffer_tensors": original_buffers,
            "converted_GroupNorm_modules": converted, "encoder_state_tensors_after_conversion": 231,
            "new_point_parameter_tensors": 2, "point_initialization_std": .001,
            "point_initialization_bias_zero": bool((model.point.bias == 0).all()),
            "old_task_checkpoint_used": False,
        }
    return model


def model_metadata(model):
    norms = [module for module in model.encoder.modules() if isinstance(module, GroupNormAct2d)]
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
        "encoder_name": ENCODER_NAME, "semantic_dim": model.semantic_dim,
        "encoder_reported_num_features": model.encoder.num_features,
        "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
        "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "encoder_parameter_tensors": sum(1 for _ in model.encoder.parameters()),
        "encoder_state_tensors": len(model.encoder.state_dict()),
        "point_parameters": sum(parameter.numel() for parameter in model.point.parameters()),
        "point_shape": [model.point.in_features, model.point.out_features],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "parameter_tensors": sum(1 for _ in model.parameters()), "state_tensors": len(model.state_dict()),
        "GroupNorm_modules": len(norms),
        "GroupNorm_groups": sorted({module.num_groups for module in norms}),
        "GroupNorm_epsilons": sorted({module.eps for module in norms}),
        "BatchNorm_modules": sum(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()),
        "buffer_tensors": sum(1 for _ in model.buffers()),
        "normalization_contract": deepcopy(NORMALIZATION_CONTRACT),
        "input_contract": CONFIG["data"], "CNN_passes_per_image": 1,
    }


def gradient_evidence(model):
    groups = {"encoder": list(model.encoder.parameters()), "encoder_stem": list(model.encoder.conv_stem.parameters()),
              "semantic_conv_head": list(model.encoder.conv_head.parameters()),
              "semantic_norm_head": list(model.encoder.norm_head.parameters()), "point": list(model.point.parameters()),
              "normalization_affine": [parameter for module in model.encoder.modules()
                  if isinstance(module, GroupNormAct2d) for parameter in (module.weight, module.bias)]}
    nonzero = {name: any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                         for parameter in parameters) for name, parameters in groups.items()}
    finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    return finite, nonzero


def validate_resume(saved, contract):
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract
            or saved.get("normalization_contract") != NORMALIZATION_CONTRACT):
        raise ValueError("resume protocol, source population or configuration differs")
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "history", "totals", "validation_rows",
                "initialization", "checkpointed_process_seconds", "epoch", "best_score", "best_epoch"}
    if not required <= set(saved):
        raise ValueError("resume requires complete model, optimizer, scheduler, scaler, RNG and source state")
    validate_checkpoint_state(saved)
    history = saved["history"]
    if not history or [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
        raise ValueError("resume history does not cover every completed epoch")
    scores = [selection_score(row["validation"]) for row in history]
    best = min(scores)
    if (saved["best_score"], saved["best_epoch"]) != (best, scores.index(best) + 1):
        raise ValueError("resume changed the earliest strict source minimum")
    if any(row["selection_score"] != score for row, score in zip(history, scores, strict=True)) or set(saved["validation_rows"]) != {"original", "roll"}:
        raise ValueError("resume lacks both source-validation populations")
    for name, n in (("original", 1506), ("roll", 512)):
        if len(saved["validation_rows"][name]) != n:
            raise ValueError("resume source scalar rows are incomplete")

    epochs, totals = saved["epoch"], saved["totals"]
    expected = {"train_batches":547*epochs, "train_cnn_image_forwards":34974*epochs,
                "train_cnn_forward_calls":547*epochs, "backward_calls":547*epochs,
                "validation_images":2018*epochs, "validation_cnn_forward_calls":32*epochs,
                "train_point_forward_calls":547*epochs, "train_point_image_forwards":34974*epochs,
                "validation_point_forward_calls":32*epochs, "validation_point_image_forwards":2018*epochs}
    if (any(totals[key] != value for key, value in expected.items())
            or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 547*epochs
            or sum(row["optimizer_updates"] for row in history) != totals["optimizer_updates"]
            or sum(row["amp_skipped_updates"] for row in history) != totals["amp_skipped_updates"]):
        raise ValueError("resume source totals or AMP update accounting differs from completed history")


def runtime(device, started):
    return {"process_seconds": time.perf_counter() - started,
            "full_process_seconds": time.perf_counter() - PROCESS_STARTED,
            "full_process_boundary": "module entry before torch import through saved predictions; excludes final completion JSON",
            "device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def validate_checkpoint_state(saved):
    state = saved["model"]
    if (len(state) != 233 or any(key.endswith(("running_mean", "running_var", "num_batches_tracked")) for key in state)):
        raise ValueError("checkpoint must contain233 GroupNorm learned tensors and no BN running buffers")


def preflight(home, output):
    if output.exists():
        raise FileExistsError("the GroupNorm8 semantic preflight already has its saved result")
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    training, validations, contract = source_populations(home)
    roll = training.datasets[1]
    train_batch = device_batch(default_collate([roll[i] for i in (0, 3, 4, 7)]), device)
    if len(set(train_batch["group_id"])) != 2 or train_batch["y"].tolist() != [0., 1., 0., 1.]:
        raise ValueError("source execution witness requires endpoint0/1 from two physical roll groups")
    validation_batches, identity_samples = {}, {}
    for name, dataset in validations.items():
        samples = [dataset[i] for i in (0, 1)]
        for i, sample in enumerate(samples):
            row = dataset.rows[i]
            image = cv2.imread(str(_resolve_image(row, dataset.base.manifest_root, dataset.base.dataset_root)))
            roi, _ = crop_and_resize(image, None if "roi_path" in row else row.get("metadata", {}).get("dial_bbox"), 512)
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            expected = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            expected = (expected-expected.new_tensor([.485,.456,.406])[:,None,None])/expected.new_tensor([.229,.224,.225])[:,None,None]
            if sample["input_failure"] or sample["image"].shape != (3,512,512) or not torch.equal(sample["image"], expected):
                raise ValueError("source validation RGB differs from the declared512 linear ImageNet pipeline")
        validation_batches[name] = device_batch(default_collate(samples), device)
        identity_samples[name] = [sample["sample_id"] for sample in samples]
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key: value.cpu() for key, value in original.items()}, strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    if list(model.named_buffers()) or list(inspect.signature(model.forward).parameters) != ["image"]:
        raise ValueError("GroupNorm8 scalar model must have no buffers and only RGB input")
    cnn_counts, cnn_hook = encoder_counter(model)
    point_counts, point_hook = point_counter(model)
    backward_counts = {"CNN_backward_calls": 0}
    def count_backward(_gradient):
        backward_counts["CNN_backward_calls"] += 1
    backward_hook = model.encoder.conv_stem.weight.register_hook(count_backward)
    observed_inputs, observed_pool, observed_point = [], [], []
    def observe_input(_module, inputs):
        observed_inputs.append({"shape":list(inputs[0].shape), "dtype":str(inputs[0].dtype)})
    def observe_pool(_module, inputs, output):
        observed_pool.append({"before":list(inputs[0].shape), "after":list(output.shape)})
    def observe_point(_module, inputs, output):
        observed_point.append({"semantic_shape":list(inputs[0].shape), "point_logit_shape":list(output.shape),
                               "semantic_dtype":str(inputs[0].dtype), "point_logit_dtype":str(output.dtype)})
    input_hook = model.encoder.register_forward_pre_hook(observe_input)
    pool_hook = model.encoder.global_pool.register_forward_hook(observe_pool)
    shape_hook = model.point.register_forward_hook(observe_point)
    norm_observations, norm_hooks = {}, []
    for name, module in model.encoder.named_modules():
        if not isinstance(module, GroupNormAct2d):
            continue
        record = {"math_calls":0, "math_dtypes":set(), "feature_dtypes":set()}
        norm_observations[name] = record
        def observe_math(_module, inputs, output, record=record):
            record["math_calls"] += 1
            record["math_dtypes"].add(tuple(str(value.dtype) for value in (*inputs, output)))
        def observe_feature(_module, inputs, output, record=record):
            record["feature_dtypes"].add((str(inputs[0].dtype), str(output.dtype)))
        norm_hooks.extend((module.math.register_forward_hook(observe_math), module.register_forward_hook(observe_feature)))
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        if prediction.shape != (4,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()):
            raise ValueError("source training must return four finite float32 scalar readings")
        (loss * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero = gradient_evidence(model)
        train_buffers_absent = not list(model.named_buffers())
        model.eval()
        with torch.inference_mode():
            for batch in validation_batches.values():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                if prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all()):
                    raise ValueError("source validation output is not one finite float32 scalar per input")
        eval_buffers_absent = not list(model.named_buffers())
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        for hook in (cnn_hook, point_hook, backward_hook, input_hook, pool_hook, shape_hook, *norm_hooks):
            hook.remove()
    expected_inputs = [{"shape":[batch,3,512,512], "dtype":"torch.float32"} for batch in (4,2,2)]
    expected_pool = [{"before":[batch,960,16,16], "after":[batch,960,1,1]} for batch in (4,2,2)]
    expected_point = [{"semantic_shape":[batch,1280], "point_logit_shape":[batch,1],
                       "semantic_dtype":"torch.float16", "point_logit_dtype":"torch.float16"} for batch in (4,2,2)]
    norm_precision = len(norm_observations) == 77 and all(
        record["math_calls"] == 3 and record["math_dtypes"] == {("torch.float32",)*4}
        and record["feature_dtypes"] == {("torch.float16", "torch.float16")} for record in norm_observations.values())
    if (cnn_counts != {"cnn_forward_calls":3, "cnn_image_forwards":8}
            or point_counts != {"point_forward_calls":3, "point_image_forwards":8}
            or backward_counts["CNN_backward_calls"] != 1 or observed_inputs != expected_inputs
            or observed_pool != expected_pool or observed_point != expected_point or not norm_precision
            or not strict_reload or not finite or not all(nonzero.values()) or not unchanged
            or not train_buffers_absent or not eval_buffers_absent or not bool(torch.isfinite(loss))):
        raise ValueError("the GroupNorm8 semantic scalar zero-update execution witness failed")
    norm_observations = {name:{key:sorted(value) if isinstance(value,set) else value for key,value in record.items()}
                         for name,record in norm_observations.items()}
    write_json(output, {"protocol":PROTOCOL, "status":"passed", "config":CONFIG, "model_metadata":metadata,
        "normalization_contract":NORMALIZATION_CONTRACT,
        "RGB_input_contract":contract["RGB_input_contract"], "initialization":model.initialization_evidence,
        "source_generation_complete":True, "source_unique":22686, "retained_source_unique":18590,
        "new_roll_source_unique":4096, "source_presentations_per_epoch":len(training), "source_batches_per_epoch":547,
        "source_validation_populations":{"original":1506,"roll":512}, "source_validation_batches_per_epoch":32,
        "roll_validation_background_independent":False, "source_probe_states":train_batch["y"].tolist(),
        "probe_sample_ids":train_batch["sample_id"], "validation_identity_sample_ids":identity_samples,
        "validation_input_matches_declared512_pipeline":True, "ImageNet_cached_only":True,
        "old_task_checkpoint_used":False, "model_forward_inputs":["image"], "target_datasets_used":False,
        "target_dataset_statistics_fitted":False, "per_image_GroupNorm_statistics_computed":True,
        "source_selection_criterion":CONFIG["checkpoint_selection"],
        "train_model_buffers_absent":train_buffers_absent, "eval_model_buffers_absent":eval_buffers_absent,
        "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "observed_encoder_inputs":observed_inputs, "observed_pool_shapes":observed_pool,
        "observed_point_shapes_and_dtypes":observed_point, "point_counts":point_counts,
        "all77_GroupNorm_float32_math_then_input_dtype":norm_precision,
        "observed_GroupNorm_dtypes":norm_observations,
        "all_gradients_finite":finite, "gradient_groups_nonzero":nonzero, "source_smooth_l1":float(loss.detach()),
        "autocast_dtype":"float16", "preflight_backward_loss_scale":128.,
        "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":backward_counts["CNN_backward_calls"],
                  "combined_backward_calls":1,"optimizer_updates":0},
        "runtime":runtime(device,started), "scope":"new per-image GroupNorm8 representation with retained512 source RGB pipeline"})


def train(home, output):
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(CONFIG["training"]["seed"])
    if (output / "completed.json").exists():
        raise FileExistsError("source training is complete; reuse the completed stage")
    latest = output / "latest.pt"
    if not latest.exists() and any((output / p).exists() for p in ("run.json", "history.jsonl", "best.pt")):
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
        "optimizer_updates", "amp_skipped_updates", "validation_images", "validation_cnn_forward_calls",
        "train_point_forward_calls", "train_point_image_forwards", "validation_point_forward_calls", "validation_point_image_forwards")}
    totals.update(training_seconds=0., validation_seconds=0.)
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = model.initialization_evidence if saved is None else saved["initialization"]
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
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "normalization_contract": NORMALIZATION_CONTRACT, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "scalar SmoothL1(beta=.05) on retained source labels with RGB512"})
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
        point_counts, point_hook = point_counter(model)
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
            hook.remove(); point_hook.remove()
        synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (547, 547, 34974):
            raise RuntimeError("source training differs from old-once plus roll-four-times exposure")
        if point_counts != {"point_forward_calls":547, "point_image_forwards":34974}:
            raise RuntimeError("scalar point training exposure differs from the single CNN source pass")
        latest_rows, validation, validation_cost = {}, {}, {}
        for name, dataset in validations.items():
            validation_point_counts, validation_point_hook = point_counter(model)
            try:
                latest_rows[name], validation_cost[name] = evaluate_population(model, dataset, device)
            finally:
                validation_point_hook.remove()
            cost = validation_cost[name]["counts"]
            if validation_point_counts != {"point_forward_calls":cost["cnn_forward_calls"], "point_image_forwards":cost["cnn_image_forwards"]}:
                raise RuntimeError("source validation scalar point exposure differs from its single CNN pass")
            validation_cost[name]["point_counts"] = validation_point_counts
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
            "train_point_forward_calls": point_counts["point_forward_calls"], "train_point_image_forwards":point_counts["point_image_forwards"],
            "validation_point_forward_calls":sum(cost["point_counts"]["point_forward_calls"] for cost in validation_cost.values()),
            "validation_point_image_forwards":sum(cost["point_counts"]["point_image_forwards"] for cost in validation_cost.values()),
            "training_seconds": train_seconds,
            "validation_seconds": sum(c["pipeline_seconds"] for c in validation_cost.values())}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "source_training_smooth_l1": loss_sum / 34974, "source_training_nmae": error_sum / 34974,
            "train_counts": counts, "train_point_counts":point_counts, "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
            "validation": validation, "validation_cost": validation_cost, "selection_score": score,
            "best_score": best, "best_epoch": best_epoch, "epochs_since_strict_best": epoch-best_epoch,
            "training_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "normalization_contract": NORMALIZATION_CONTRACT, "data_contract": contract, "model": model.state_dict(),
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
            or totals["train_point_forward_calls"] != 547*epochs or totals["train_point_image_forwards"] != 34974*epochs
            or totals["validation_point_forward_calls"] != 32*epochs or totals["validation_point_image_forwards"] != 2018*epochs
            or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 547*epochs):
        raise RuntimeError("source completion does not match the declared stopping rule and actual exposure")
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "normalization_contract": NORMALIZATION_CONTRACT, "status": "complete", "epochs": epochs,
        "completion_reason": reason, "best_epoch": best_epoch, "best_score": best,
        "best_validation": history[best_epoch-1]["validation"], "last_validation": history[-1]["validation"],
        "totals": totals, "model_metadata": metadata, "initialization": initialization, "epoch_exposure": exposure,
        "source_sample_ids": {name: [r["sample_id"] for r in rows] for name, rows in contract["populations"].items()},
        "resume_from_epoch": start, "runtime": runtime(device, started),
        "checkpointed_process_seconds_before_resume": checkpointed_seconds,
        "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def validate_external_checkpoint(checkpoint, completed):
    validate_checkpoint_state(checkpoint)
    reason = completion_reason(completed["epochs"], completed["best_epoch"])
    if (checkpoint["protocol"] != PROTOCOL or checkpoint["config"] != CONFIG
            or checkpoint.get("normalization_contract") != NORMALIZATION_CONTRACT
            or completed.get("normalization_contract") != NORMALIZATION_CONTRACT
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
    if any(output.glob("*.csv")) or any(output.glob("*.csv.gz")):
        raise FileExistsError("partial external CSV exists without summary; retain it without another forward")
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
    point_counts, point_hook = point_counter(model)
    try:
        rows, cost = evaluate_population(model, dataset, device)
    finally:
        point_hook.remove()
    if point_counts != {"point_forward_calls":cost["counts"]["cnn_forward_calls"], "point_image_forwards":len(dataset)}:
        raise RuntimeError("frozen scalar point exposure differs from its single CNN pass")
    cost["point_counts"] = point_counts
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native512_groupnorm8_semantic", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "normalization_contract": NORMALIZATION_CONTRACT, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
        "point_counts":point_counts, "summary": summarize(rows), "by_group": {g: summarize(rr) for g, rr in groups.items()},
        "by_component": {c: summarize(rr) for c, rr in components.items()}, "forward_counts": cost["counts"],
        "timing": cost, "runtime": runtime(device, started), "model_inputs": CONFIG["inference_input"],
        "training_or_adaptation_during_evaluation": False, "backward_calls": 0, "optimizer_updates": 0,
        "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
        "development_scope": "previously viewed external development; frozen per-image GroupNorm statistics; no target dataset fitting or selection",
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
