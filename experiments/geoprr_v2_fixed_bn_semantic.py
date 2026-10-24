"""Train the complete semantic scalar reader with fixed cached ImageNet BN statistics."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from experiments.geoprr_v2_roll_source import SOURCE_RELATIVE
from experiments.geoprr_v2_semantic_tokens import selection_score
from experiments.geoprr_v2_semantic_direct import (
    ENCODER_NAME, SemanticDirectReader,
    encoder_counter, evaluate_population, save_csv, synchronize,
)
from experiments.geoprr_v2_high_resolution_direct import RGB_INPUT_CONTRACT, source_populations, point_counter
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_fixed_bn_semantic_source_only_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v27_fixed_bn_semantic_20261025/source_only_training"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {"encoder": ENCODER_NAME, "num_classes": 0, "semantic_dim": 1280,
              "initialization": "fresh complete cached ImageNet encoder, including conv_head and norm_head; new scalar point",
              "readout": "sigmoid(float(Linear1280to1(full semantic output)))",
              "point_initialization_std": .001, "point_initialization_bias": 0.,
              "CNN_passes_per_image": 1, "joint_encoder_training": True, "source_training_BN_updates": False,
              "BN_statistics": "original cached ImageNet running_mean/running_var/num_batches_tracked fixed in all77 BN modules",
              "BN_affine_trainable": True},
    "data": {"image_size": 512, "output_size": 128},
    "photometric": {"contrast": [.8, 1.2], "brightness": [-.1, .1], "saturation": [.8, 1.2]},
    "source": {"old_unique": 18590, "roll_unique": 4096, "roll_repeat": 4, "total_unique": 22686,
               "train_presentations": 34974, "original_validation": 1506, "roll_validation": 512,
               "require_geometry": False, "pair_manifest": None,
               "roll_validation_split": "independent layout/seed/ID/group; same available HDR background",
               "population_policy": "same four manifest rows and order as retained V22 source; authoritative RGB512 input contract"},
    "training": {"seed": 20262020, "min_epochs": 60, "max_epochs": 160, "early_stopping_patience": 20,
                 "batch_size": 64, "validation_batch_size": 64, "workers": 4, "amp": True,
                 "optimizer": "AdamW", "learning_rate": .0003, "weight_decay": .0001,
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.,
                 "scheduler": {"mode": "min", "factor": .3, "patience": 8, "threshold": 0.,
                               "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6, "eps": 1e-8}},
    "checkpoint_selection": "earliest strict minimum R=(original1506 mean NMAE + roll512 mean NMAE + original1506 worst10percent NMAE + roll512 worst10percent NMAE)/4; each worst tail uses ceil(.1*N), including failed inputs/scalars at error1",
    "external_checkpoint": "source-selected best only after source stopping rule completes",
    "inference_input": "provided native ROI RGB, linear resize512, ImageNet normalization; one RGB forward, no GT geometry or target scalar input",
    "external_training_or_adaptation": False,
}


def completion_reason(epoch, best_epoch):
    if epoch >= CONFIG["training"]["max_epochs"]:
        return "maximum_epochs"
    if (epoch >= CONFIG["training"]["min_epochs"] and best_epoch is not None
            and epoch - best_epoch >= CONFIG["training"]["early_stopping_patience"]):
        return "source_score_patience"
    return None


class FixedBNSemanticReader(SemanticDirectReader):
    """Train every parameter while retaining the cached ImageNet BN statistics."""

    def __init__(self, encoder, *, capture_reference):
        super().__init__(encoder)
        self._fixed_bn_reference = None
        if capture_reference:
            self._fixed_bn_reference = {key: value.detach().clone()
                                        for key, value in self.fixed_bn_buffers().items()}
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self._fixed_bn_reference is not None:
            self._fixed_bn_reference = {key: fn(value) for key, value in self._fixed_bn_reference.items()}
        return self

    def fixed_bn_buffers(self):
        buffers = {}
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for field in ("running_mean", "running_var", "num_batches_tracked"):
                    value = getattr(module, field)
                    if value is None:
                        raise ValueError("fixed ImageNet BN requires tracked running statistics")
                    buffers[f"encoder.{name}.{field}"] = value
        return buffers

    def assert_fixed_bn(self):
        buffers = self.fixed_bn_buffers()
        if self._fixed_bn_reference is None or set(buffers) != set(self._fixed_bn_reference):
            raise ValueError("original cached ImageNet BN reference is missing or incomplete")
        for key, value in buffers.items():
            reference = self._fixed_bn_reference[key]
            if (value.device != reference.device or value.dtype != reference.dtype
                    or value.shape != reference.shape or not torch.equal(value, reference)):
                raise ValueError(f"fixed cached ImageNet BN buffer changed: {key}")
        return {"buffer_tensors_checked": len(buffers), "all_equal_original_cached_ImageNet": True}

    def fixed_bn_checkpoint(self):
        self.assert_fixed_bn()
        return {key: value.detach().cpu().clone() for key, value in self._fixed_bn_reference.items()}

    def restore_fixed_bn_reference(self, reference):
        buffers = self.fixed_bn_buffers()
        if set(reference) != set(buffers):
            raise ValueError("checkpoint original ImageNet BN reference is incomplete")
        for key, value in buffers.items():
            if (reference[key].shape != value.shape or reference[key].dtype != value.dtype
                    or not torch.equal(reference[key].to(value.device), value)):
                raise ValueError(f"checkpoint BN state differs from its original ImageNet reference: {key}")
        self._fixed_bn_reference = {key: reference[key].detach().to(value.device).clone()
                                    for key, value in buffers.items()}
        self.assert_fixed_bn()


def validate_fixed_bn_checkpoint(saved):
    reference = saved.get("fixed_bn_reference")
    buffers = {key: value for key, value in saved["model"].items()
               if key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    if (reference is None or len(reference) != 231 or set(reference) != set(buffers)
            or saved["initialization"].get("fixed_BN_reference_origin") != "fresh_cached_ImageNet_encoder_before_source"
            or saved["initialization"].get("fixed_BN_reference_tensors") != 231):
        raise ValueError("checkpoint lacks the complete original cached ImageNet BN reference")
    if any(reference[key].shape != value.shape or reference[key].dtype != value.dtype
           or not torch.equal(reference[key], value) for key, value in buffers.items()):
        raise ValueError("checkpoint BN buffers differ from its original cached ImageNet reference")


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    reference = {key: value.detach().clone() for key, value in encoder.state_dict().items()} if pretrained else None
    model = FixedBNSemanticReader(encoder, capture_reference=pretrained)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"],
            metadata["semantic_conv_head_retained"], metadata["semantic_norm_head_retained"]) != (1280, 8434512, 462, True, True):
        raise ValueError("complete semantic encoder differs from the declared cached ImageNet architecture")
    if (metadata["parameters"], metadata["trainable_parameters"], metadata["parameter_tensors"],
            metadata["trainable_parameter_tensors"], metadata["state_tensors"], metadata["BN_modules"],
            metadata["fixed_BN_buffer_tensors"], metadata["trainable_BN_affine_tensors"]) != (8435793, 8435793, 233, 233, 464, 77, 231, 154):
        raise ValueError("fixed-BN semantic encoder parameters or buffers differ from the declared architecture")
    if reference is not None:
        state = model.encoder.state_dict()
        if set(state) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in state.items()):
            raise ValueError("constructing the new point changed cached ImageNet encoder state")
        model.initialization_evidence = {
            "encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
            "retained_complete_encoder_state_tensors": len(reference), "all_cached_encoder_state_tensors_retained": True,
            "new_point_parameter_tensors": sum(1 for _ in model.point.parameters()),
            "point_initialization_std": .001, "point_initialization_bias_zero": bool((model.point.bias == 0).all()),
            "old_task_checkpoint_used": False,
            "fixed_BN_reference_origin": "fresh_cached_ImageNet_encoder_before_source",
            "fixed_BN_reference_tensors": len(model.fixed_bn_buffers()),
        }
    return model


def model_metadata(model):
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
        "encoder_name": ENCODER_NAME, "semantic_dim": model.semantic_dim,
        "encoder_reported_num_features": model.encoder.num_features,
        "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
        "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "encoder_state_tensors": len(model.encoder.state_dict()),
        "point_parameters": sum(parameter.numel() for parameter in model.point.parameters()),
        "point_shape": [model.point.in_features, model.point.out_features],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_tensors": sum(1 for _ in model.parameters()), "state_tensors": len(model.state_dict()),
        "trainable_parameter_tensors": sum(1 for p in model.parameters() if p.requires_grad),
        "BN_modules": sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.encoder.modules()),
        "fixed_BN_buffer_tensors": len(model.fixed_bn_buffers()),
        "trainable_BN_affine_tensors": sum(1 for m in model.encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) for p in m.parameters() if p.requires_grad),
        "input_contract": CONFIG["data"], "CNN_passes_per_image": 1,
    }


def gradient_evidence(model):
    groups = {"encoder": model.encoder, "encoder_stem": model.encoder.conv_stem,
              "semantic_conv_head": model.encoder.conv_head, "semantic_norm_head": model.encoder.norm_head,
              "point": model.point}
    bn_affine = [p for m in model.encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) for p in m.parameters()]
    nonzero = {name: any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                         for parameter in module.parameters()) for name, module in groups.items()}
    nonzero["BN_affine_all_tensors"] = all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in bn_affine)
    finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    return finite, nonzero


def validate_resume(saved, contract):
    if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
        raise ValueError("resume protocol, source population or configuration differs")
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "history", "totals", "validation_rows",
                "initialization", "checkpointed_process_seconds", "epoch", "best_score", "best_epoch", "fixed_bn_reference"}
    if not required <= set(saved):
        raise ValueError("resume requires complete model, optimizer, scheduler, scaler, RNG and source state")
    validate_fixed_bn_checkpoint(saved)
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
        raise FileExistsError("the fixed-BN semantic preflight already has its saved result")
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
    reloaded.restore_fixed_bn_reference(model.fixed_bn_checkpoint())
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    counters = {key: value.clone() for key, value in model.fixed_bn_buffers().items() if key.endswith("num_batches_tracked")}
    fixed_before = model.assert_fixed_bn()
    if len(counters) != 77 or list(inspect.signature(model.forward).parameters) != ["image"]:
        raise ValueError("complete semantic scalar model or its RGB-only interface differs")
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
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        if prediction.shape != (4,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()):
            raise ValueError("source512 training forward must return four finite float32 scalar readings")
        (loss * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero = gradient_evidence(model)
        train_bn_check = model.assert_fixed_bn()
        if not all(torch.equal(model.fixed_bn_buffers()[key], value) for key, value in counters.items()):
            raise ValueError("source training changed fixed ImageNet BN counters")
        model.eval()
        before_eval = {key: value.clone() for key, value in model.named_buffers()}
        with torch.inference_mode():
            for batch in validation_batches.values():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                if prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all()):
                    raise ValueError("source512 validation output is not one finite float32 scalar per input")
        eval_bn_check = model.assert_fixed_bn()
        eval_buffers_unchanged = all(torch.equal(value, before_eval[key]) for key, value in model.named_buffers())
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        cnn_hook.remove(); point_hook.remove(); backward_hook.remove()
        input_hook.remove(); pool_hook.remove(); shape_hook.remove()
    expected_inputs = [{"shape":[batch,3,512,512], "dtype":"torch.float32"} for batch in (4,2,2)]
    expected_pool = [{"before":[batch,960,16,16], "after":[batch,960,1,1]} for batch in (4,2,2)]
    expected_point = [{"semantic_shape":[batch,1280], "point_logit_shape":[batch,1],
                       "semantic_dtype":"torch.float16", "point_logit_dtype":"torch.float16"} for batch in (4,2,2)]
    if (cnn_counts != {"cnn_forward_calls":3, "cnn_image_forwards":8}
            or point_counts != {"point_forward_calls":3, "point_image_forwards":8}
            or backward_counts["CNN_backward_calls"] != 1 or observed_inputs != expected_inputs
            or observed_pool != expected_pool or observed_point != expected_point
            or not strict_reload or not finite or not all(nonzero.values()) or not unchanged
            or not eval_buffers_unchanged or not bool(torch.isfinite(loss))):
        raise ValueError("the512 semantic scalar zero-update execution witness failed")
    write_json(output, {"protocol":PROTOCOL, "status":"passed", "config":CONFIG, "model_metadata":metadata,
        "RGB_input_contract":contract["RGB_input_contract"], "initialization":model.initialization_evidence,
        "source_generation_complete":True, "source_unique":22686, "retained_source_unique":18590,
        "new_roll_source_unique":4096, "source_presentations_per_epoch":len(training), "source_batches_per_epoch":547,
        "source_validation_populations":{"original":1506,"roll":512}, "source_validation_batches_per_epoch":32,
        "roll_validation_background_independent":False, "source_probe_states":train_batch["y"].tolist(),
        "probe_sample_ids":train_batch["sample_id"], "validation_identity_sample_ids":identity_samples,
        "validation_input_matches_declared512_pipeline":True, "ImageNet_cached_only":True,
        "old_task_checkpoint_used":False, "model_forward_inputs":["image"], "target_datasets_used":False,
        "target_statistics_used":False, "source_selection_criterion":CONFIG["checkpoint_selection"],
        "source_BN_counter_tensors_checked":77, "source_BN_counter_increment_training":0,
        "fixed_BN_buffers_before":fixed_before, "fixed_BN_buffers_after_training":train_bn_check,
        "fixed_BN_buffers_after_validation":eval_bn_check,
        "source_BN_counter_additional_increment_eval":0, "eval_model_buffers_unchanged":eval_buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "observed_encoder_inputs":observed_inputs, "observed_pool_shapes":observed_pool,
        "observed_point_shapes_and_dtypes":observed_point, "point_counts":point_counts,
        "all_gradients_finite":finite, "gradient_groups_nonzero":nonzero, "source_smooth_l1":float(loss.detach()),
        "autocast_dtype":"float16", "preflight_backward_loss_scale":128.,
        "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":backward_counts["CNN_backward_calls"],
                  "combined_backward_calls":1,"optimizer_updates":0},
        "runtime":runtime(device,started), "scope":"fixed cached ImageNet BN statistics with trainable complete semantic encoder and scalar point"})


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
        model.restore_fixed_bn_reference(saved["fixed_bn_reference"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, best, best_epoch = saved["epoch"], saved["best_score"], saved["best_epoch"]
        history, totals, latest_rows = saved["history"], saved["totals"], saved["validation_rows"]
        checkpointed_seconds, initialization = saved["checkpointed_process_seconds"], saved["initialization"]
    output.mkdir(parents=True, exist_ok=True)
    fixed_initial = model.assert_fixed_bn()
    metadata = model_metadata(model)
    exposure = {"train_unique_images": 22686, "old_train_images": 18590, "roll_unique_images": 4096,
                "roll_repetitions": 4, "train_images": len(training), "train_batches": 547,
                "original_validation_images": 1506, "roll_validation_images": 512,
                "validation_images": 2018, "validation_batches": 32}
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "scalar SmoothL1(beta=.05) on retained source labels with RGB512",
        "fixed_BN_buffers": fixed_initial})
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
        fixed_after_training = model.assert_fixed_bn()
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
            validation_cost[name]["fixed_BN_buffers"] = model.assert_fixed_bn()
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
            "fixed_BN_buffers_after_training": fixed_after_training,
            "fixed_BN_buffers_after_validation": model.assert_fixed_bn(),
            "training_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "epoch": epoch, "best_epoch": best_epoch, "best_score": best,
            "history": history, "totals": totals, "validation_rows": latest_rows, "initialization": initialization,
            "fixed_bn_reference": model.fixed_bn_checkpoint(),
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
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": epochs,
        "completion_reason": reason, "best_epoch": best_epoch, "best_score": best,
        "best_validation": history[best_epoch-1]["validation"], "last_validation": history[-1]["validation"],
        "totals": totals, "model_metadata": metadata, "initialization": initialization, "epoch_exposure": exposure,
        "source_sample_ids": {name: [r["sample_id"] for r in rows] for name, rows in contract["populations"].items()},
        "fixed_BN_buffers":model.assert_fixed_bn(), "resume_from_epoch": start, "runtime": runtime(device, started),
        "checkpointed_process_seconds_before_resume": checkpointed_seconds,
        "checkpointed_process_seconds": checkpointed_seconds + time.perf_counter() - started})


def validate_external_checkpoint(checkpoint, completed):
    validate_fixed_bn_checkpoint(checkpoint)
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
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.restore_fixed_bn_reference(checkpoint["fixed_bn_reference"])
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
        row.update(dataset=dataset_name, condition="native512_fixed_bn_semantic", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
        "point_counts":point_counts, "summary": summarize(rows), "by_group": {g: summarize(rr) for g, rr in groups.items()},
        "by_component": {c: summarize(rr) for c, rr in components.items()}, "forward_counts": cost["counts"],
        "timing": cost, "runtime": runtime(device, started), "model_inputs": CONFIG["inference_input"],
        "training_or_adaptation_during_evaluation": False, "backward_calls": 0, "optimizer_updates": 0,
        "target_source": "provided scalar labels" if dataset_name == "industrial" else "offline target derived from four annotated points",
        "development_scope": "previously viewed external development; no target training, BN adaptation or selection",
        "failure_policy": "retain every declared input; input or scalar failure receives normalized error1",
        "fixed_BN_buffers":model.assert_fixed_bn(), "source_completion_epochs": completed["epochs"]})


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
