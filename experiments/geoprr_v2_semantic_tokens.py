"""Jointly train a full RGB encoder and generic multiscale semantic-token readout."""
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
from experiments.geoprr_v2_roll_source import (
    SOURCE_RELATIVE, completion_reason, source_populations,
)
from experiments.geoprr_v2_semantic_direct import (
    ENCODER_NAME, encoder_counter, evaluate_population, save_csv, synchronize,
)
from experiments.evaluate_geoprr_v2_role_external import ExternalPopulationDataset, load_population, summarize
from experiments.run_geoprr_v2_optimization import write_json
from experiments.train_geoprr_v2 import (
    device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
)
from experiments.train_geoprr_v2_reference_relation import save_predictions


PROTOCOL = "geoprr_v2_semantic_tokens_source_only_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/iteration_v23_semantic_tokens_20261024/source_only_training"
CONFIG = {
    "protocol": PROTOCOL,
    "model": {
        "encoder": ENCODER_NAME, "num_classes": 0, "semantic_dim": 1280,
        "initialization": "fresh complete cached ImageNet encoder, including conv_head and norm_head; new token readout",
        "feature_stages": ["blocks.0:48x64x64", "blocks.1:80x32x32", "blocks.2:160x16x16", "blocks.4:960x8x8"],
        "fine_stage": "pixel_unshuffle2:48x64x64 to192x32x32, preserving each ordered2x2 pattern",
        "projection_channels": [192, 80, 160, 960], "token_dim": 128,
        "memory_tokens": 2368, "memory_levels": 4,
        "memory_normalization": "independent1x1 projections and per-token LayerNorm128 at each level",
        "position_encoding": "fixed deterministic2D sine/cosine ROI cell centers normalized per axis to[0,2pi],128channels; four learned level embeddings",
        "queries": "four generic learned128 embeddings plus one shared Linear1280to128 semantic context; no annotated roles",
        "decoder": {"layers": 2, "d_model": 128, "nhead": 4, "dim_feedforward": 512,
                    "dropout": .1, "activation": "gelu", "batch_first": True, "norm_first": True,
                    "initialization": "two separately constructed TransformerDecoderLayer instances"},
        "scalar_readout": "LayerNorm128, flatten4x128, Linear512to128, GELU, Linear128to1, float sigmoid",
        "CNN_passes_per_image": 1, "joint_encoder_training": True,
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
                 "loss": "smooth_l1", "smooth_l1_beta": .05, "gradient_clip": 5.,
                 "scheduler": {"mode": "min", "factor": .3, "patience": 8, "threshold": 0.,
                               "threshold_mode": "abs", "cooldown": 0, "min_lr": 1e-6, "eps": 1e-8}},
    "checkpoint_selection": "earliest strict minimum R=(original1506 mean NMAE + roll512 mean NMAE + original1506 worst10percent NMAE + roll512 worst10percent NMAE)/4; each worst tail uses ceil(.1*N), including failed inputs/scalars at error1",
    "external_checkpoint": "source-selected best only after source stopping rule completes",
    "inference_input": "ROI RGB, linear resize256, ImageNet normalization; one RGB forward, no GT geometry or target scalar input",
    "external_training_or_adaptation": False,
}


def position_encoding(height, width):
    """Encode regular cell centers with64 channels per coordinate."""
    frequency = torch.exp(-math.log(10000.) * torch.arange(32, dtype=torch.float32) / 32)
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32) + .5,
                          torch.arange(width, dtype=torch.float32) + .5, indexing="ij")
    x = (x.reshape(-1, 1) / width * (2 * math.pi)) * frequency
    y = (y.reshape(-1, 1) / height * (2 * math.pi)) * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)


class SemanticTokenReadout(nn.Module):
    """Read multiscale RGB cell memories with four generic semantic-conditioned queries."""

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(channels, 128, 1) for channels in (192, 80, 160, 960))
        self.memory_norms = nn.ModuleList(nn.LayerNorm(128) for _ in range(4))
        self.level_embeddings = nn.Parameter(torch.randn(4, 128) * .02)
        self.query_embeddings = nn.Parameter(torch.randn(4, 128) * .02)
        self.semantic_context = nn.Linear(1280, 128)
        self.decoders = nn.ModuleList(nn.TransformerDecoderLayer(
            d_model=128, nhead=4, dim_feedforward=512, dropout=.1,
            activation="gelu", batch_first=True, norm_first=True) for _ in range(2))
        self.output_norm = nn.LayerNorm(128)
        self.output = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        for level, size in enumerate((32, 32, 16, 8)):
            self.register_buffer(f"position_{level}", position_encoding(size, size))

    def spatial_memory(self, features):
        if len(features) != 4:
            raise ValueError("semantic-token readout requires all four native feature levels")
        batch = len(features[0])
        expected = ((48, 64, 64), (80, 32, 32), (160, 16, 16), (960, 8, 8))
        if any(tuple(feature.shape) != (batch, *shape) for feature, shape in zip(features, expected, strict=True)):
            raise ValueError("native multiscale feature shapes differ from the declared256 RGB pipeline")
        memories = []
        for level, feature in enumerate(features):
            if level == 0:
                feature = F.pixel_unshuffle(feature, 2)
            token = self.projections[level](feature).flatten(2).transpose(1, 2)
            token = self.memory_norms[level](token)
            position = getattr(self, f"position_{level}").to(dtype=token.dtype)
            memories.append(token + position[None] + self.level_embeddings[level].to(dtype=token.dtype)[None, None])
        return torch.cat(memories, dim=1)

    def forward(self, features, semantic):
        memory = self.spatial_memory(features)
        if semantic.shape != (len(memory), 1280):
            raise ValueError("complete encoder must return one1280-channel semantic vector per RGB input")
        context = self.semantic_context(semantic)
        queries = context[:, None] + self.query_embeddings.to(dtype=context.dtype)[None]
        for decoder in self.decoders:
            queries = decoder(queries, memory)
        return self.output(self.output_norm(queries).flatten(1)).squeeze(-1).float().sigmoid()


class SemanticTokenReader(nn.Module):
    """Use one complete trainable encoder pass for semantics and four spatial levels."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.readout = SemanticTokenReadout()
        self._features = {}
        self._hooks = [self.encoder.blocks[index].register_forward_hook(self._capture(index)) for index in (0, 1, 2, 4)]
        self.last_feature_shapes = None

    def _capture(self, index):
        def capture(_module, _inputs, output):
            if index in self._features:
                raise RuntimeError("a declared native feature stage executed more than once in one encoder pass")
            self._features[index] = output
        return capture

    def forward(self, image):
        self._features.clear()
        try:
            semantic = self.encoder(image)
            if set(self._features) != {0, 1, 2, 4}:
                raise RuntimeError("complete encoder did not expose the four declared native feature stages")
            features = [self._features[index] for index in (0, 1, 2, 4)]
            self.last_feature_shapes = [list(feature.shape) for feature in features]
            return self.readout(features, semantic)
        finally:
            self._features.clear()


def create_model(*, pretrained):
    if pretrained and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("fresh ImageNet initialization requires HF_HUB_OFFLINE=1; new downloads are not authorized")
    import timm

    encoder = timm.create_model(ENCODER_NAME, pretrained=pretrained, num_classes=0)
    reference = {key: value.detach().clone() for key, value in encoder.state_dict().items()} if pretrained else None
    model = SemanticTokenReader(encoder)
    metadata = model_metadata(model)
    if (metadata["semantic_dim"], metadata["encoder_parameters"], metadata["encoder_state_tensors"],
            metadata["semantic_conv_head_retained"], metadata["semantic_norm_head_retained"]) != (1280, 8434512, 462, True, True):
        raise ValueError("complete semantic encoder differs from the declared cached ImageNet architecture")
    if reference is not None:
        state = model.encoder.state_dict()
        if set(state) != set(reference) or any(not torch.equal(value, reference[key]) for key, value in state.items()):
            raise ValueError("constructing the new readout changed cached ImageNet encoder state")
        model.initialization_evidence = {
            "encoder": ENCODER_NAME, "ImageNet_pretrained": True, "existing_cached_weights_only": True,
            "retained_complete_encoder_state_tensors": len(reference), "all_cached_encoder_state_tensors_retained": True,
            "new_readout_parameter_tensors": sum(1 for _ in model.readout.parameters()),
            "old_task_checkpoint_used": False,
        }
    return model


def model_metadata(model):
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "encoder_class": f"{type(model.encoder).__module__}.{type(model.encoder).__name__}",
        "encoder_name": ENCODER_NAME, "semantic_dim": int(model.encoder.conv_head.out_channels),
        "encoder_reported_num_features": model.encoder.num_features,
        "semantic_conv_head_retained": isinstance(model.encoder.conv_head, nn.Conv2d),
        "semantic_norm_head_retained": "norm_head" in dict(model.encoder.named_modules()),
        "encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "encoder_state_tensors": len(model.encoder.state_dict()),
        "readout_parameters": sum(parameter.numel() for parameter in model.readout.parameters()),
        "readout_parameter_tensors": sum(1 for _ in model.readout.parameters()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_tensors": sum(1 for _ in model.parameters()), "state_tensors": len(model.state_dict()),
        "feature_shapes_per_image": [[48, 64, 64], [80, 32, 32], [160, 16, 16], [960, 8, 8]],
        "memory_shape_per_image": [2368, 128], "query_shape_per_image": [4, 128],
        "fine_stage_rearrangement": "pixel_unshuffle2", "CNN_passes_per_image": 1,
        "generic_queries_without_annotated_roles": True,
    }


def gradient_evidence(model):
    groups = {"encoder": model.encoder, "semantic_conv_head": model.encoder.conv_head,
              "semantic_norm_head": model.encoder.norm_head, "semantic_context": model.readout.semantic_context,
              "output_norm": model.readout.output_norm, "scalar_readout": model.readout.output}
    groups.update({f"projection_level_{i}": layer for i, layer in enumerate(model.readout.projections)})
    groups.update({f"memory_norm_level_{i}": layer for i, layer in enumerate(model.readout.memory_norms)})
    groups.update({f"decoder_{i}": layer for i, layer in enumerate(model.readout.decoders)})
    nonzero = {name: any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in module.parameters())
               for name, module in groups.items()}
    for name, parameter in (("level_embeddings", model.readout.level_embeddings),
                            ("query_embeddings", model.readout.query_embeddings)):
        nonzero[name] = parameter.grad is not None and bool((parameter.grad.abs().sum(dim=1) > 0).all())
    finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    return finite, nonzero


def selection_score(validation):
    if set(validation) != {"original", "roll"}:
        raise ValueError("selection requires only the two complete source validations")
    for name, count in (("original", 1506), ("roll", 512)):
        population = validation[name]
        if population["samples"] != count:
            raise ValueError("source risk selection requires the complete declared source population")
        for key in ("nmae", "tail10_nmae"):
            value = population[key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("source mean or worst10percent NMAE is invalid")
    return (validation["original"]["nmae"] + validation["roll"]["nmae"]
            + validation["original"]["tail10_nmae"] + validation["roll"]["tail10_nmae"]) / 4


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
        raise FileExistsError("the semantic-token preflight already has its saved result")
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
    strict_reload = all(torch.equal(reloaded.state_dict()[key], value.cpu()) for key, value in original.items())
    del reloaded
    counters = {key: value.clone() for key, value in model.state_dict().items() if key.endswith("num_batches_tracked")}
    if len(counters) != 77 or list(inspect.signature(model.forward).parameters) != ["image"]:
        raise ValueError("complete semantic model or RGB-only forward interface differs")
    counts, hook = encoder_counter(model)
    backward_counts = {"CNN_backward_calls": 0}
    def count_backward(_gradient):
        backward_counts["CNN_backward_calls"] += 1
    backward_hook = model.encoder.conv_stem.weight.register_hook(count_backward)
    shapes = []
    memory_hook = model.readout.decoders[0].register_forward_pre_hook(
        lambda _module, inputs: shapes.append({"queries": list(inputs[0].shape), "memory": list(inputs[1].shape)}))
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction, train_batch["y"], beta=.05)
        train_shapes = model.last_feature_shapes
        if prediction.shape != (4,):
            raise ValueError("semantic-token training forward must return four scalar readings")
        (loss * 128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        all_finite, nonzero = gradient_evidence(model)
        after_train = {key: model.state_dict()[key].clone() for key in counters}
        if not all(torch.equal(after_train[key], value+1) for key, value in counters.items()):
            raise ValueError("new source training did not update all encoder BN counters once")
        model.eval()
        before_eval = {key: value.clone() for key, value in model.named_buffers()}
        with torch.inference_mode():
            for batch in validation_batches.values():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"])
                if prediction.shape != (2,) or not bool(torch.isfinite(prediction).all()) or not bool(((prediction>=0)&(prediction<=1)).all()):
                    raise ValueError("new source validation output is not one finite scalar per input")
        eval_buffers_unchanged = all(torch.equal(value, before_eval[key]) for key, value in model.named_buffers())
        unchanged = all(torch.equal(value, original[key]) for key, value in model.named_parameters())
    finally:
        hook.remove(); backward_hook.remove(); memory_hook.remove()
    expected_shapes = [{"queries": [batch, 4, 128], "memory": [batch, 2368, 128]} for batch in (4, 2, 2)]
    if (counts != {"cnn_forward_calls":3, "cnn_image_forwards":8} or backward_counts["CNN_backward_calls"] != 1
            or shapes != expected_shapes or not strict_reload or not all_finite or not all(nonzero.values())
            or not unchanged or not eval_buffers_unchanged or not bool(torch.isfinite(loss))):
        raise ValueError("semantic-token zero-update execution witness failed")
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
        "source_BN_counter_tensors_checked":77, "source_BN_counter_increment_training":1,
        "source_BN_counter_additional_increment_eval":0, "eval_model_buffers_unchanged":eval_buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":sum(1 for _ in model.parameters()),
        "strict_model_reload":strict_reload, "full_state_tensors":len(model.state_dict()),
        "train_native_feature_shapes":train_shapes, "decoder_input_shapes":shapes,
        "autocast_dtype":"float16", "preflight_backward_loss_scale":128.,
        "all_gradients_finite":all_finite, "gradient_groups_nonzero":nonzero, "source_loss":float(loss.detach()),
        "counts":{"CNN_calls":3,"CNN_images":8,"CNN_backward_calls":backward_counts["CNN_backward_calls"],
                  "combined_backward_calls":1,"optimizer_updates":0},
        "runtime":runtime(device,started), "scope":"new multiscale semantic-token architecture and reused source RGB execution witness"})


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
        "optimizer_updates", "amp_skipped_updates", "validation_images", "validation_cnn_forward_calls")}
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
    write_json(output / "run.json", {"protocol": PROTOCOL, "config": CONFIG, "data_contract": contract,
        "initialization": initialization, "model_metadata": metadata, "epoch_exposure": exposure,
        "inference_input": CONFIG["inference_input"], "source_only_loss": "SmoothL1(beta=.05) on scalar source labels"})
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
            hook.remove()
        synchronize(device)
        train_seconds = time.perf_counter() - tick
        if (batches, counts["cnn_forward_calls"], counts["cnn_image_forwards"]) != (547, 547, 34974):
            raise RuntimeError("source training differs from old-once plus roll-four-times exposure")
        latest_rows, validation, validation_cost = {}, {}, {}
        for name, dataset in validations.items():
            latest_rows[name], validation_cost[name] = evaluate_population(model, dataset, device)
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
            "training_seconds": train_seconds,
            "validation_seconds": sum(c["pipeline_seconds"] for c in validation_cost.values())}.items():
            totals[key] += value
        record = {"epoch": epoch, "learning_rate": used_lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
            "source_training_smooth_l1": loss_sum / 34974, "source_training_nmae": error_sum / 34974,
            "train_counts": counts, "backward_calls": batches, "optimizer_updates": updates, "amp_skipped_updates": skips,
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
    model.to(device).eval().requires_grad_(False)
    folder = "Industrial-1395" if dataset_name == "industrial" else "RF100-VL"
    population = load_population(home / "datasets" / folder, dataset_name)
    dataset = ExternalPopulationDataset(population, CONFIG["data"])
    torch.cuda.reset_peak_memory_stats(device)
    rows, cost = evaluate_population(model, dataset, device)
    groups, components = defaultdict(list), defaultdict(list)
    for row, source in zip(rows, population, strict=True):
        row.update(dataset=dataset_name, condition="native256_semantic_tokens", component=source["component"])
        groups[row["group_id"]].append(row); components[row["component"]].append(row)
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "predictions.csv", rows)
    save_csv(output / "groups.csv", [{"group_id": g, **summarize(rr)} for g, rr in groups.items()])
    write_json(output / "summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "dataset": dataset_name,
        "declared_samples": len(rows), "declared_groups": len(groups), "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_epoch": checkpoint["best_epoch"],
        "checkpoint_source_best_score": checkpoint["best_score"], "model_metadata": model_metadata(model),
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
