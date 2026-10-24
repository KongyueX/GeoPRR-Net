"""Train a complete RGB encoder with one ordered reading distribution and CDF log-score."""
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


PROTOCOL = "geoprr_v2_ordinal_distribution_source_only_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v25_ordinal_distribution_20261024/source_only_training"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {
        "encoder": ENCODER_NAME, "num_classes": 0, "semantic_dim": 1280,
        "initialization": "fresh complete cached ImageNet encoder, including conv_head and norm_head; new ordered distribution head",
        "head": "Linear1280to256", "head_initialization_std": .001, "head_initialization_bias": 0.,
        "nodes": "fixed float32 linspace(0,1,256), saved buffer; no circular adjacency",
        "distribution": "float32 log_softmax of the single head output",
        "readout": "float32 sum(exp(logp)*all256nodes)/sum(exp(logp)); no clipping or peak selection",
        "encoder_and_head_precision": "AMP float16; log probability, CDF log-score and expectation float32 with autocast disabled",
        "CNN_passes_per_image": 1, "joint_encoder_training": True, "source_training_BN_updates": True,
    },
    "data": {"image_size": 256, "output_size": 64},
    "photometric": {"contrast": [.8, 1.2], "brightness": [-.1, .1], "saturation": [.8, 1.2]},
    "source": {"old_unique": 18590, "roll_unique": 4096, "roll_repeat": 4, "total_unique": 22686,
               "train_presentations": 34974, "original_validation": 1506, "roll_validation": 512,
               "require_geometry": False, "pair_manifest": None,
               "roll_validation_split": "independent layout/seed/ID/group; same available HDR background"},
    "training": {"seed": 20262020, "min_epochs": 60, "max_epochs": 160, "early_stopping_patience": 20,
                 "batch_size": 64, "validation_batch_size": 64, "workers": 4, "amp": True,
                 "optimizer": "AdamW", "learning_rate": .0003, "weight_decay": .0001,
                 "loss": "cdf_bernoulli_log_score",
                 "target_cdf": "G_j=clamp(j+1-255*y,0,1), j=0..254; source scalar loss side only",
                 "loss_formula": "-mean_B,j(G_j*logF_j+(1-G_j)*logS_j); left and right logcumsumexp of logp",
                 "gradient_clip": 5.,
                 "scheduler": {"mode": "min", "factor": .3, "patience": 8, "threshold": 0.,
                               "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6, "eps": 1e-8}},
    "checkpoint_selection": "earliest strict minimum R=(original1506 mean NMAE + roll512 mean NMAE + original1506 worst10percent NMAE + roll512 worst10percent NMAE)/4; each worst tail uses ceil(.1*N), including failed inputs/scalars at error1",
    "external_checkpoint": "source-selected best only after source stopping rule completes",
    "inference_input": "ROI RGB, linear resize256, ImageNet normalization; one RGB forward, no GT geometry or target scalar input",
    "external_training_or_adaptation": False,
}


def target_cdf(target):
    with torch.autocast(device_type=target.device.type, enabled=False):
        thresholds = torch.arange(1, 256, device=target.device, dtype=torch.float32)
        return (thresholds[None] - 255 * target.float().reshape(-1, 1)).clamp(0, 1)


def log_cdf_terms(log_probability):
    with torch.autocast(device_type=log_probability.device.type, enabled=False):
        log_probability = log_probability.float()
        left = torch.logcumsumexp(log_probability, dim=-1)[:, :-1]
        right = torch.logcumsumexp(log_probability.flip(-1), dim=-1).flip(-1)[:, 1:]
        return left, right


def cdf_log_score(log_probability, target):
    if log_probability.shape != (target.numel(), 256):
        raise ValueError("CDF log-score requires one256-node distribution per source scalar")
    with torch.autocast(device_type=log_probability.device.type, enabled=False):
        left, right = log_cdf_terms(log_probability)
        truth = target_cdf(target)
        return -(truth * left + (1 - truth) * right).mean()


def expected_reading(log_probability, nodes):
    with torch.autocast(device_type=log_probability.device.type, enabled=False):
        probability = log_probability.float().exp()
        return (probability * nodes.float()[None]).sum(-1) / probability.sum(-1)


class OrdinalDistributionReader(nn.Module):
    """Read one complete semantic representation as a fixed-node ordered distribution."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.semantic_dim = int(encoder.conv_head.out_channels)
        self.head = nn.Linear(self.semantic_dim, 256)
        nn.init.normal_(self.head.weight, std=.001)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("nodes", torch.linspace(0, 1, 256, dtype=torch.float32))

    def log_distribution(self, image):
        semantic = self.encoder(image)
        if semantic.shape != (len(image), self.semantic_dim):
            raise ValueError("complete encoder must return one1280-channel semantic vector per RGB input")
        logits = self.head(semantic)
        with torch.autocast(device_type=image.device.type, enabled=False):
            return F.log_softmax(logits.float(), dim=-1)

    def expected_reading(self, log_probability):
        return expected_reading(log_probability, self.nodes)

    def forward(self, image):
        return self.expected_reading(self.log_distribution(image))


def validate_nodes(model):
    expected = torch.linspace(0, 1, 256, dtype=torch.float32).to(device=model.nodes.device)
    if model.nodes.dtype != torch.float32 or not torch.equal(model.nodes, expected):
        raise ValueError("ordered readout must retain the fixed256-node float32 buffer")


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    reference = {key: value.detach().clone() for key, value in encoder.state_dict().items()} if pretrained else None
    model = OrdinalDistributionReader(encoder)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"],
            metadata["semantic_conv_head_retained"], metadata["semantic_norm_head_retained"]) != (1280, 8434512, 462, True, True):
        raise ValueError("complete semantic encoder differs from the declared cached ImageNet architecture")
    validate_nodes(model)
    if reference is not None:
        state = model.encoder.state_dict()
        if set(state) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in state.items()):
            raise ValueError("constructing the ordered head changed cached ImageNet encoder state")
        model.initialization_evidence = {
            "encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
            "retained_complete_encoder_state_tensors": len(reference), "all_cached_encoder_state_tensors_retained": True,
            "new_head_parameter_tensors": sum(1 for _ in model.head.parameters()),
            "head_initialization_std": .001, "head_initialization_bias_zero": bool((model.head.bias == 0).all()),
            "fixed_nodes_saved_as_buffer": True, "old_task_checkpoint_used": False,
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
        "head_parameters": sum(parameter.numel() for parameter in model.head.parameters()),
        "head_shape": [model.head.in_features, model.head.out_features],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_tensors": sum(1 for _ in model.parameters()), "state_tensors": len(model.state_dict()),
        "node_buffer_shape": list(model.nodes.shape), "node_buffer_dtype": str(model.nodes.dtype),
        "CNN_passes_per_image": 1,
    }


def head_counter(model):
    counts = {"head_forward_calls": 0, "head_image_forwards": 0}
    def count(_module, inputs, _output):
        counts["head_forward_calls"] += 1
        counts["head_image_forwards"] += len(inputs[0])
    return counts, model.head.register_forward_hook(count)


def gradient_evidence(model):
    groups = {"encoder": model.encoder, "encoder_stem": model.encoder.conv_stem,
              "semantic_conv_head": model.encoder.conv_head, "semantic_norm_head": model.encoder.norm_head,
              "ordered_head": model.head}
    nonzero = {name: any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                         for parameter in module.parameters()) for name, module in groups.items()}
    finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    return finite, nonzero


def validate_resume(saved, contract):
    if saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["data_contract"] != contract:
        raise ValueError("resume protocol, source population or configuration differs")
    required = {"model", "optimizer", "scheduler", "scaler", "rng", "history", "totals", "validation_rows",
                "initialization", "checkpointed_process_seconds", "epoch", "best_score", "best_epoch"}
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
        raise FileExistsError("the ordered-distribution preflight already has its saved result")
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
    metadata = model_metadata(model)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key: value.cpu() for key, value in original.items()}, strict=True)
    validate_nodes(reloaded)
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    counters = {key: value.clone() for key, value in model.state_dict().items() if key.endswith("num_batches_tracked")}
    if (len(counters) != 77 or list(inspect.signature(model.forward).parameters) != ["image"]
            or list(inspect.signature(model.log_distribution).parameters) != ["image"]):
        raise ValueError("complete semantic model or its two RGB-only interfaces differ")
    cnn_counts, cnn_hook = encoder_counter(model)
    head_counts, head_hook = head_counter(model)
    backward_counts = {"CNN_backward_calls": 0}
    def count_backward(_gradient):
        backward_counts["CNN_backward_calls"] += 1
    backward_hook = model.encoder.conv_stem.weight.register_hook(count_backward)
    observed_head = []
    def observe_head(_module, inputs, output):
        observed_head.append({"semantic_shape":list(inputs[0].shape), "head_logits_shape":list(output.shape),
                              "semantic_dtype":str(inputs[0].dtype), "head_logits_dtype":str(output.dtype)})
    shape_hook = model.head.register_forward_hook(observe_head)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            log_probability = model.log_distribution(train_batch["image"])
            prediction = model.expected_reading(log_probability)
            loss = cdf_log_score(log_probability, train_batch["y"])
        logp_finite = bool(torch.isfinite(log_probability).all())
        probability_sums = log_probability.exp().sum(-1)
        probability_normalized = bool(torch.allclose(probability_sums, torch.ones_like(probability_sums), rtol=1e-5, atol=1e-6))
        left, right = log_cdf_terms(log_probability)
        if (log_probability.shape != (4, 256) or prediction.shape != (4,)
                or any(tensor.dtype != torch.float32 for tensor in (log_probability, prediction, loss, left, right))
                or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all())):
            raise ValueError("source ordered distribution, CDF or expected reading differs from the float32 contract")
        (loss * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite, nonzero = gradient_evidence(model)
        after_train = {key: model.state_dict()[key].clone() for key in counters}
        if not all(torch.equal(after_train[key], value+1) for key, value in counters.items()):
            raise ValueError("source training did not update all encoder BN counters exactly once")
        model.eval()
        before_eval = {key: value.clone() for key, value in model.named_buffers()}
        with torch.inference_mode():
            for batch in validation_batches.values():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                if prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all()):
                    raise ValueError("source validation output is not one finite float32 expected reading per input")
        eval_buffers_unchanged = all(torch.equal(value, before_eval[key]) for key, value in model.named_buffers())
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
        validate_nodes(model)
    finally:
        cnn_hook.remove(); head_hook.remove(); backward_hook.remove(); shape_hook.remove()
    expected_head = [{"semantic_shape":[batch,1280], "head_logits_shape":[batch,256],
                      "semantic_dtype":"torch.float16", "head_logits_dtype":"torch.float16"} for batch in (4,2,2)]
    if (cnn_counts != {"cnn_forward_calls":3, "cnn_image_forwards":8}
            or head_counts != {"head_forward_calls":3, "head_image_forwards":8}
            or backward_counts["CNN_backward_calls"] != 1 or observed_head != expected_head
            or not strict_reload or not finite or not all(nonzero.values()) or not unchanged
            or not eval_buffers_unchanged or not logp_finite or not probability_normalized or not bool(torch.isfinite(loss))):
        raise ValueError("ordered-distribution zero-update execution witness failed")
    write_json(output, {"protocol":PROTOCOL, "status":"passed", "config":CONFIG, "model_metadata":metadata,
        "initialization":model.initialization_evidence, "source_generation_complete":True,
        "source_unique":22686, "retained_source_unique":18590, "new_roll_source_unique":4096,
        "source_presentations_per_epoch":len(training), "source_batches_per_epoch":547,
        "source_validation_populations":{"original":1506,"roll":512}, "source_validation_batches_per_epoch":32,
        "roll_validation_background_independent":False, "source_probe_states":train_batch["y"].tolist(),
        "probe_sample_ids":train_batch["sample_id"], "validation_identity_sample_ids":identity_samples,
        "validation_input_matches_declared256_pipeline":True, "ImageNet_cached_only":True,
        "old_task_checkpoint_used":False, "model_forward_inputs":["image"], "model_log_distribution_inputs":["image"],
        "target_datasets_used":False, "target_statistics_used":False,
        "source_selection_criterion":CONFIG["checkpoint_selection"], "source_BN_counter_tensors_checked":77,
        "source_BN_counter_increment_training":1, "source_BN_counter_additional_increment_eval":0,
        "eval_model_buffers_unchanged":eval_buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "observed_head_shapes_and_dtypes":observed_head, "head_counts":head_counts,
        "log_probability_dtype":str(log_probability.dtype), "CDF_dtype":str(left.dtype),
        "source_readout_dtype":"torch.float32", "source_loss_dtype":str(loss.dtype),
        "log_probability_finite":logp_finite, "probability_mass_sums":probability_sums.detach().cpu().tolist(),
        "probability_normalized":probability_normalized, "all_gradients_finite":finite,
        "gradient_groups_nonzero":nonzero, "source_cdf_log_score":float(loss.detach()),
        "autocast_dtype":"float16", "preflight_backward_loss_scale":128.,
        "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":backward_counts["CNN_backward_calls"],
                  "combined_backward_calls":1,"optimizer_updates":0},
        "runtime":runtime(device,started), "scope":"new ordered readout and CDF log-score with reused source RGB"})


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
        "train_head_forward_calls", "train_head_image_forwards", "validation_head_forward_calls", "validation_head_image_forwards")}
    totals.update(training_seconds=0., validation_seconds=0.)
    start, best, best_epoch, history, checkpointed_seconds, latest_rows = 0, math.inf, None, [], 0., None
    initialization = model.initialization_evidence if saved is None else saved["initialization"]
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        validate_nodes(model)
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
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "ordered CDF Bernoulli log-score on source scalar two-hot CDF"})
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
        head_counts, head_hook = head_counter(model)
        batches = updates = skips = 0
        loss_sum = error_sum = 0.
        try:
            for index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    log_probability = model.log_distribution(batch["image"])
                    prediction = model.expected_reading(log_probability)
                    loss = cdf_log_score(log_probability, batch["y"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite CDF log-score at epoch{epoch} batch{index + 1}")
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
            hook.remove(); head_hook.remove()
        synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (547, 547, 34974):
            raise RuntimeError("source training differs from old-once plus roll-four-times exposure")
        if head_counts != {"head_forward_calls":547, "head_image_forwards":34974}:
            raise RuntimeError("ordered head training exposure differs from the single CNN source pass")
        latest_rows, validation, validation_cost = {}, {}, {}
        for name, dataset in validations.items():
            validation_head_counts, validation_head_hook = head_counter(model)
            try:
                latest_rows[name], validation_cost[name] = evaluate_population(model, dataset, device)
            finally:
                validation_head_hook.remove()
            cost = validation_cost[name]["counts"]
            if validation_head_counts != {"head_forward_calls":cost["cnn_forward_calls"], "head_image_forwards":cost["cnn_image_forwards"]}:
                raise RuntimeError("source validation ordered head exposure differs from its single CNN pass")
            validation_cost[name]["head_counts"] = validation_head_counts
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
            "train_head_forward_calls": head_counts["head_forward_calls"], "train_head_image_forwards":head_counts["head_image_forwards"],
            "validation_head_forward_calls":sum(cost["head_counts"]["head_forward_calls"] for cost in validation_cost.values()),
            "validation_head_image_forwards":sum(cost["head_counts"]["head_image_forwards"] for cost in validation_cost.values()),
            "training_seconds": train_seconds,
            "validation_seconds": sum(c["pipeline_seconds"] for c in validation_cost.values())}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "source_training_cdf_log_score": loss_sum / 34974, "source_training_nmae": error_sum / 34974,
            "train_counts": counts, "train_head_counts":head_counts, "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
            "validation": validation, "validation_cost": validation_cost, "selection_score": score,
            "best_score": best, "best_epoch": best_epoch, "epochs_since_strict_best": epoch-best_epoch,
            "training_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract, "model": model.state_dict(),
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
            or totals["train_head_forward_calls"] != 547*epochs or totals["train_head_image_forwards"] != 34974*epochs
            or totals["validation_head_forward_calls"] != 32*epochs or totals["validation_head_image_forwards"] != 2018*epochs
            or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 547*epochs):
        raise RuntimeError("source completion does not match the declared stopping rule and actual exposure")
    write_json(output / "completed.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": epochs,
        "completion_reason": reason, "best_epoch": best_epoch, "best_score": best,
        "best_validation": history[best_epoch-1]["validation"], "last_validation": history[-1]["validation"],
        "totals": totals, "model_metadata": metadata, "initialization": initialization, "epoch_exposure": exposure,
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
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    validate_nodes(model)
    model.to(device).eval().requires_grad_(False)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    head_counts, head_hook = head_counter(model)
    try:
        rows, cost = evaluate_population(model, dataset, device)
    finally:
        head_hook.remove()
    if head_counts != {"head_forward_calls":cost["counts"]["cnn_forward_calls"], "head_image_forwards":len(dataset)}:
        raise RuntimeError("frozen ordered head exposure differs from its single CNN pass")
    cost["head_counts"] = head_counts
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256_ordinal_distribution", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
        "head_counts":head_counts, "summary": summarize(rows), "by_group": {g: summarize(rr) for g, rr in groups.items()},
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
