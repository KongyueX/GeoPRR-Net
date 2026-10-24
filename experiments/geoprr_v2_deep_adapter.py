"""Train deep residual adapters inside a fixed complete ImageNet RGB encoder."""
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
from experiments.geoprr_v2_roll_source import SOURCE_RELATIVE, completion_reason, source_populations
from experiments.geoprr_v2_semantic_tokens import selection_score
from experiments.geoprr_v2_semantic_direct import (
    ENCODER_NAME, encoder_counter, evaluate_population, save_csv, synchronize,
)
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_deep_adapter_source_only_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v24_deep_adapter_20261024/source_only_training"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {
        "encoder": ENCODER_NAME, "num_classes": 0, "semantic_dim": 1280,
        "initialization": "fresh complete cached ImageNet encoder, including conv_head and norm_head; new point then new adapters",
        "encoder_policy": "all original weights, BN affine and buffers fixed; encoder always eval; retain downstream input autograd",
        "adapters": {"2": {"channels": 160, "height": 16, "width": 16, "hidden": 40},
                     "4": {"channels": 960, "height": 8, "width": 8, "hidden": 240}},
        "adapter_transform": "out + Conv1x1(C/4,C,bias=True)(GELU(Conv1x1(C,C/4,bias=True)(out))); replacement hooks, no inplace",
        "adapter_initialization": "down weight default Conv2d; down bias zero; up weight and bias zero",
        "adapter_registration": "reader ModuleDict outside encoder; original encoder state keys retained",
        "point_initialization_std": .001, "point_initialization_bias": 0.,
        "scalar_readout": "new Linear1280to1 on complete semantic output, float sigmoid",
        "CNN_passes_per_image": 1,
    },
    "data": {"image_size": 256, "output_size": 64},
    "photometric": {"contrast": [.8, 1.2], "brightness": [-.1, .1], "saturation": [.8, 1.2]},
    "source": {"old_unique": 18590, "roll_unique": 4096, "roll_repeat": 4, "total_unique": 22686,
               "train_presentations": 34974, "original_validation": 1506, "roll_validation": 512,
               "require_geometry": False, "pair_manifest": None,
               "roll_validation_split": "independent layout/seed/ID/group; same available HDR background"},
    "training": {"seed": 20262020, "min_epochs": 60, "max_epochs": 160, "early_stopping_patience": 20,
                 "batch_size": 64, "validation_batch_size": 64, "workers": 4, "amp": True,
                 "optimizer": "AdamW", "optimizer_scope": "new point and two adapters only",
                 "learning_rate": .0003, "weight_decay": .0001,
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.,
                 "scheduler": {"mode": "min", "factor": .3, "patience": 8, "threshold": 0.,
                               "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6, "eps": 1e-8}},
    "checkpoint_selection": "earliest strict minimum R=(original1506 mean NMAE + roll512 mean NMAE + original1506 worst10percent NMAE + roll512 worst10percent NMAE)/4; each worst tail uses ceil(.1*N), including failed inputs/scalars at error1",
    "external_checkpoint": "source-selected best only after source stopping rule completes",
    "inference_input": "ROI RGB, linear resize256, ImageNet normalization; one RGB forward, no GT geometry or target scalar input",
    "external_training_or_adaptation": False,
}


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


class ResidualAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels // 4, 1, bias=True)
        self.activation = nn.GELU()
        self.up = nn.Conv2d(channels // 4, channels, 1, bias=True)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature):
        return self.up(self.activation(self.down(feature)))


class DeepAdapterReader(nn.Module):
    """Adapt two deep stage outputs while preserving every original encoder state."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.semantic_dim = int(encoder.conv_head.out_channels)
        self.point = nn.Linear(self.semantic_dim, 1)
        nn.init.normal_(self.point.weight, std=.001)
        nn.init.zeros_(self.point.bias)
        self.adapters = nn.ModuleDict({"2": ResidualAdapter(160), "4": ResidualAdapter(960)})
        self.reset_adapter_counts()
        self.last_adapter_shapes = {}
        self.last_semantic_shape = None
        self._hooks = [self.encoder.blocks[int(key)].register_forward_hook(self._adapt(key)) for key in self.adapters]

    def reset_adapter_counts(self):
        self.adapter_counts = {key: {"calls": 0, "images": 0} for key in ("2", "4")}

    def _adapt(self, key):
        def adapt(_module, _inputs, output):
            specification = CONFIG["model"]["adapters"][key]
            expected = (len(output), specification["channels"], specification["height"], specification["width"])
            if tuple(output.shape) != expected:
                raise ValueError("deep adapter stage shape differs from the declared256 RGB pipeline")
            self.adapter_counts[key]["calls"] += 1
            self.adapter_counts[key]["images"] += len(output)
            self.last_adapter_shapes[key] = list(output.shape)
            return output + self.adapters[key](output)
        return adapt

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image):
        before = {key: counts["calls"] for key, counts in self.adapter_counts.items()}
        semantic = self.encoder(image)
        if semantic.shape != (len(image), self.semantic_dim):
            raise ValueError("complete encoder must return one1280-channel semantic vector per RGB input")
        if any(counts["calls"] != before[key] + 1 for key, counts in self.adapter_counts.items()):
            raise RuntimeError("each deep adapter must execute exactly once in the native encoder pass")
        self.last_semantic_shape = list(semantic.shape)
        return self.point(semantic).squeeze(-1).float().sigmoid()


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    reference = {key: value.detach().clone() for key, value in encoder.state_dict().items()} if pretrained else None
    model = DeepAdapterReader(encoder)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"],
            metadata["semantic_conv_head_retained"], metadata["semantic_norm_head_retained"]) != (1280, 8434512, 462, True, True):
        raise ValueError("complete fixed encoder differs from the declared cached ImageNet architecture")
    if reference is not None:
        evidence = frozen_encoder_evidence(model, reference)
        if not all(bool((parameter == 0).all()) for adapter in model.adapters.values() for parameter in adapter.up.parameters()):
            raise ValueError("fresh deep-adapter up weights and biases must be exactly zero")
        model.initialization_evidence = {
            "encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
            "retained_complete_encoder_state_tensors": len(reference), "all_cached_encoder_state_tensors_retained": True,
            "new_trainable_parameter_tensors": len(trainable_parameters(model)),
            "new_point_initialization_std": .001,
            "adapter_up_weights_and_biases_zero": all(bool((parameter == 0).all()) for adapter in model.adapters.values() for parameter in adapter.up.parameters()),
            "adapter_down_biases_zero": all(bool((adapter.down.bias == 0).all()) for adapter in model.adapters.values()),
            "old_task_checkpoint_used": False, "frozen_encoder": evidence,
        }
    return model


def model_metadata(model):
    encoder_keys = set(model.encoder.state_dict())
    full_keys = set(model.state_dict())
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
        "encoder_name": ENCODER_NAME, "semantic_dim": model.semantic_dim,
        "encoder_reported_num_features": model.encoder.num_features,
        "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
        "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "encoder_state_tensors": len(encoder_keys),
        "original_encoder_state_keys_retained": {"encoder." + key for key in encoder_keys} == {key for key in full_keys if key.startswith("encoder.")},
        "adapter_parameters": {key: sum(parameter.numel() for parameter in module.parameters()) for key, module in model.adapters.items()},
        "point_parameters": sum(parameter.numel() for parameter in model.point.parameters()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters(model)),
        "trainable_parameter_tensors": len(trainable_parameters(model)),
        "parameter_tensors": sum(1 for _ in model.parameters()), "state_tensors": len(model.state_dict()),
        "adapter_stage_shapes_per_image": {"2": [160, 16, 16], "4": [960, 8, 8]},
        "adapter_hidden_channels": {key: adapter.down.out_channels for key, adapter in model.adapters.items()},
        "CNN_passes_per_image": 1, "original_encoder_always_eval": True,
        "original_encoder_parameter_gradients": False,
    }


def frozen_reference(model):
    return {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}


def frozen_encoder_evidence(model, reference):
    state = model.encoder.state_dict()
    if set(state) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in state.items()):
        raise RuntimeError("an original cached ImageNet encoder parameter or buffer changed")
    if (any(module.training for module in model.encoder.modules())
            or any(parameter.requires_grad or parameter.grad is not None for parameter in model.encoder.parameters())):
        raise RuntimeError("all original encoder parameters and BN state must remain fixed in eval mode")
    return {"original_state_tensors_compared": len(state), "all_original_state_tensors_unchanged": True,
            "original_encoder_parameters_frozen": True, "original_parameter_gradients": 0,
            "BN_counter_tensors_compared": sum(key.endswith("num_batches_tracked") for key in state),
            "BN_counter_increment": 0, "all_encoder_submodules_eval": True}


def gradient_evidence(model):
    parameters = trainable_parameters(model)
    finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
    nonzero = {"point": any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                             for parameter in model.point.parameters())}
    down_zero = {}
    for key, adapter in model.adapters.items():
        nonzero[f"adapter_{key}_up"] = any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                                           for parameter in adapter.up.parameters())
        down_zero[f"adapter_{key}_down"] = all(parameter.grad is not None and bool((parameter.grad == 0).all())
                                               for parameter in adapter.down.parameters())
    return finite, nonzero, down_zero


def validate_resume(saved, contract):
    if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
        raise ValueError("resume protocol, source population or configuration differs")
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "history", "totals", "validation_rows",
                "initialization", "checkpointed_process_seconds", "epoch", "best_score", "best_epoch", "frozen_encoder"}
    if not required <= set(saved):
        raise ValueError("resume requires complete model, optimizer, scheduler, scaler, RNG and source state")
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
        raise FileExistsError("the deep-adapter preflight already has its saved result")
    started, device = time.perf_counter(), torch.device("cuda")
    seed_everything(20262020)
    training, validations, contract = source_populations(home)
    roll = training.datasets[1]
    train_batch = device_batch(default_collate([roll[i] for i in (0, 3, 4, 7)]), device)
    if len(set(train_batch["group_id"])) != 2 or train_batch["y"].tolist() != [0., 1., 0., 1.]:
        raise ValueError("new-source execution witness requires endpoint0/1 from two physical roll groups")
    validation_batches, identity_samples = {}, {}
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
        identity_samples[name] = [sample["sample_id"] for sample in samples]
    model = create_model(pretrained=True).to(device)
    reference = frozen_reference(model)
    metadata = model_metadata(model)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key: value.cpu() for key, value in original.items()}, strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    if (sum(key.endswith("num_batches_tracked") for key in reference) != 77
            or list(inspect.signature(model.forward).parameters) != ["image"]):
        raise ValueError("complete fixed encoder or RGB-only forward interface differs")
    counts, hook = encoder_counter(model)
    backward_counts = {"first_adapter_up_backward_calls": 0}
    def count_backward(_gradient):
        backward_counts["first_adapter_up_backward_calls"] += 1
    backward_hook = model.adapters["2"].up.weight.register_hook(count_backward)
    trace = []
    source_backward_calls = 0
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        trace.append({"adapters": deepcopy(model.last_adapter_shapes), "semantic": model.last_semantic_shape})
        if prediction.shape != (4,):
            raise ValueError("deep-adapter training forward must return four scalar readings")
        (loss * 128.).backward()
        source_backward_calls += 1
        for parameter in trainable_parameters(model):
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero, down_zero = gradient_evidence(model)
        train_frozen = frozen_encoder_evidence(model, reference)
        model.eval()
        with torch.inference_mode():
            for batch in validation_batches.values():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                trace.append({"adapters": deepcopy(model.last_adapter_shapes), "semantic": model.last_semantic_shape})
                if prediction.shape != (2,) or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all()):
                    raise ValueError("new source validation output is not one finite scalar per input")
        eval_frozen = frozen_encoder_evidence(model, reference)
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        hook.remove(); backward_hook.remove()
    expected_trace = [{"adapters": {"2": [batch, 160, 16, 16], "4": [batch, 960, 8, 8]},
                       "semantic": [batch, 1280]} for batch in (4, 2, 2)]
    if (counts != {"cnn_forward_calls":3, "cnn_image_forwards":8} or source_backward_calls != 1
            or backward_counts["first_adapter_up_backward_calls"] != 1
            or model.adapter_counts != {key: {"calls":3, "images":8} for key in ("2", "4")}
            or trace != expected_trace or not strict_reload or not finite or not all(nonzero.values())
            or not all(down_zero.values()) or not unchanged or not bool(torch.isfinite(loss))):
        raise ValueError("deep-adapter zero-update execution witness failed")
    write_json(output, {"protocol":PROTOCOL, "status":"passed", "config":CONFIG, "model_metadata":metadata,
        "initialization":model.initialization_evidence, "source_generation_complete":True,
        "source_unique":22686, "retained_source_unique":18590, "new_roll_source_unique":4096,
        "source_presentations_per_epoch":len(training), "source_batches_per_epoch":547,
        "source_validation_populations":{"original":1506,"roll":512}, "source_validation_batches_per_epoch":32,
        "roll_validation_background_independent":False, "source_probe_states":train_batch["y"].tolist(),
        "probe_sample_ids":train_batch["sample_id"], "validation_identity_sample_ids":identity_samples,
        "validation_input_matches_declared256_pipeline":True, "ImageNet_cached_only":True,
        "old_task_checkpoint_used":False, "model_forward_inputs":["image"], "target_datasets_used":False,
        "target_statistics_used":False, "source_selection_criterion":CONFIG["checkpoint_selection"],
        "source_BN_counter_tensors_checked":77, "source_BN_counter_increment_training":0,
        "source_BN_counter_additional_increment_eval":0, "frozen_encoder_after_train":train_frozen,
        "frozen_encoder_after_eval":eval_frozen, "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "new_parameter_tensors_unchanged_without_optimizer":len(trainable_parameters(model)),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "observed_shapes":trace, "adapter_hook_counts":deepcopy(model.adapter_counts),
        "autocast_dtype":"float16", "preflight_backward_loss_scale":128.,
        "all_trainable_gradients_finite":finite, "gradient_groups_nonzero":nonzero,
        "initial_down_gradients_exist_and_zero":down_zero, "source_loss":float(loss.detach()),
        "first_adapter_up_backward_calls":backward_counts["first_adapter_up_backward_calls"],
        "counts":{"CNN_calls":3,"CNN_images":8,"source_backward_calls":source_backward_calls,
                  "combined_backward_calls":1,"optimizer_updates":0},
        "backward_scope":"new adapter and point parameters through fixed downstream operations; original encoder parameter gradients0",
        "runtime":runtime(device,started), "scope":"new fixed-encoder deep-adapter architecture and reused source RGB execution witness"})


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
    model = create_model(pretrained=True).to(device)
    reference = frozen_reference(model)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **CONFIG["training"]["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    totals = {k: 0 for k in ("train_batches", "train_cnn_image_forwards", "train_cnn_forward_calls", "backward_calls",
        "optimizer_updates", "amp_skipped_updates", "validation_images", "validation_cnn_forward_calls")}
    totals.update(training_seconds=0., validation_seconds=0.)
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = model.initialization_evidence
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        frozen_encoder_evidence(model, reference)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch = saved["epoch"], saved["best_score"], saved["best_epoch"]
        history, totals, latest_rows = saved["history"], saved["totals"], saved["validation_rows"]
        checkpointed_seconds, initialization = saved["checkpointed_process_seconds"], saved["initialization"]
    frozen = frozen_encoder_evidence(model, reference)
    output.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    exposure = {"train_unique_images": 22686, "old_train_images": 18590, "roll_unique_images": 4096,
                "roll_repetitions": 4, "train_images": len(training), "train_batches": 547,
                "original_validation_images": 1506, "roll_validation_images": 512,
                "validation_images": 2018, "validation_batches": 32}
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "SmoothL1(beta=.05) on scalar source labels",
        "frozen_encoder": frozen, "optimizer_parameter_tensors": len(parameters)})
    torch.cuda.reset_peak_memory_stats(device)
    reason = completion_reason(start, best_epoch)
    for epoch in range(start + 1, CONFIG["training"]["max_epochs"] + 1):
        if reason is not None:
            break
        tick = time.perf_counter()
        model.train()
        model.reset_adapter_counts()
        used_lr = optimizer.param_groups[0]["lr"]
        loader = make_loader(training, batch_size=64, workers=4, shuffle=True,
                             seed=20262020 + epoch, device=device)
        counts, hook = encoder_counter(model)
        batches = updates = skips = backwards = 0
        finite_gradient_batches = nonfinite_gradient_batches = 0
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
                backwards += 1
                scaler.unscale_(optimizer)
                if any(parameter.grad is None for parameter in parameters):
                    raise RuntimeError("a declared trainable adapter or point parameter lost its source gradient")
                finite_gradients = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
                finite_gradient_batches += int(finite_gradients)
                nonfinite_gradient_batches += int(not finite_gradients)
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.)
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
        train_adapter_counts = deepcopy(model.adapter_counts)
        if train_adapter_counts != {key: {"calls":547, "images":34974} for key in ("2", "4")} or backwards != 547:
            raise RuntimeError("deep adapter hooks or source backward attempts differ from the complete training exposure")
        model.reset_adapter_counts()
        latest_rows, validation, validation_cost = {}, {}, {}
        for name, dataset in validations.items():
            latest_rows[name], validation_cost[name] = evaluate_population(model, dataset, device)
            validation[name] = summarize(latest_rows[name])
        validation_adapter_counts = deepcopy(model.adapter_counts)
        if validation_adapter_counts != {key: {"calls":32, "images":2018} for key in ("2", "4")}:
            raise RuntimeError("deep adapter hooks differ from the complete source-validation exposure")
        frozen = frozen_encoder_evidence(model, reference)
        score = selection_score(validation)
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        scheduler.step(score)
        for key, value in {"train_batches": batches, "train_cnn_image_forwards": counts["cnn_image_forwards"],
            "train_cnn_forward_calls": counts["cnn_forward_calls"], "backward_calls": backwards,
            "optimizer_updates": updates, "amp_skipped_updates": skips, "validation_images": 2018,
            "validation_cnn_forward_calls": sum(c["counts"]["cnn_forward_calls"] for c in validation_cost.values()),
            "training_seconds": train_seconds,
            "validation_seconds": sum(c["pipeline_seconds"] for c in validation_cost.values())}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "source_training_smooth_l1": loss_sum / 34974, "source_training_nmae": error_sum / 34974,
            "train_counts": counts, "backward_calls": backwards, "optimizer_updates": updates, "amp_skipped_updates": skips,
            "train_adapter_counts": train_adapter_counts, "validation_adapter_counts": validation_adapter_counts,
            "trainable_gradient_finite_batches": finite_gradient_batches,
            "trainable_gradient_nonfinite_batches": nonfinite_gradient_batches, "frozen_encoder": frozen,
            "original_encoder_parameter_backward_calls": 0,
            "validation": validation, "validation_cost": validation_cost, "selection_score": score,
            "best_score": best, "best_epoch": best_epoch, "epochs_since_strict_best": epoch-best_epoch,
            "training_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "epoch": epoch, "best_epoch": best_epoch, "best_score": best,
            "history": history, "totals": totals, "validation_rows": latest_rows, "initialization": initialization,
            "frozen_encoder": frozen,
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
        "frozen_encoder": frozen_encoder_evidence(model, reference),
        "backward_scope": "source loss backward attempts through adapters and fixed downstream operations; original encoder parameters fixed",
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
    model = create_model(pretrained=True).to(device)
    reference = frozen_reference(model)
    model.load_state_dict(checkpoint["model"], strict=True)
    frozen_encoder_evidence(model, reference)
    model.eval().requires_grad_(False)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    rows, cost = evaluate_population(model, dataset, device)
    frozen = frozen_encoder_evidence(model, reference)
    if model.adapter_counts != {key: {"calls":cost["counts"]["cnn_forward_calls"], "images":len(dataset)} for key in ("2", "4")}:
        raise RuntimeError("frozen external adapter hooks differ from the declared RGB exposure")
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256_deep_adapter", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
        "frozen_encoder": frozen, "adapter_hook_counts": deepcopy(model.adapter_counts),
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
