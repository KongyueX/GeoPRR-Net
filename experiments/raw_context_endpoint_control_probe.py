"""Endpoint-refresh controls for the Raw progress-refinement probe.

This continuation reuses the completed 30-epoch inner-scene EfficientNet-B0
foundation.  It isolates whether the earlier context-MLP improvement came from
nonlinear representation refinement or merely from five additional epochs of
exact-L1 endpoint optimization.

The three trainable arms share one frozen encoder and the same batches:

``linear_refresh``
    A copy of the original 1,280-to-1 endpoint, initialized exactly at base.
``activation_null``
    A parameter-matched context residual with GELU replaced by Identity.
``context_mlp``
    The same context residual with GELU enabled.

All arms start at the exact base prediction.  The formal SyncG holdout and all
field cohorts remain outside this runner.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.a11_scort import SCORTRawEfficientNetB0Encoder
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_LEARNING_RATE,
    DEFAULT_REFINER_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    PROTOCOL as SOURCE_PROBE_PROTOCOL,
    ConditionedProgressDataset,
    build_probe_from_foundation,
    load_inner_scene_population,
)
from experiments.resnet18_direct_progress import (
    DirectProgressDataset,
    DirectProgressError,
    DirectSample,
    _configure_reproducibility,
    _loader,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.support_geometry_multiview_efficientnet import (
    EFFICIENTNET_B0_FEATURES,
)
from experiments.syncg_lightweight_regression_baselines import (
    DEFAULT_SCENE_SPLIT,
    LightweightProgressRegressor,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    InternalSceneSplit,
)


PROTOCOL: Final[str] = "syncg_raw_context_endpoint_control_probe_v1"
DEFAULT_FOUNDATION_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_multiscale_progress_probe/seed_20262020/"
    "inner_foundation.pt"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/raw_context_endpoint_control_probe/seed_20262020"
)

METHOD_BASE: Final[str] = "base"
METHOD_LINEAR_REFRESH: Final[str] = "linear_refresh"
METHOD_ACTIVATION_NULL: Final[str] = "activation_null"
METHOD_CONTEXT_MLP: Final[str] = "context_mlp"
METHODS: Final[tuple[str, ...]] = (
    METHOD_BASE,
    METHOD_LINEAR_REFRESH,
    METHOD_ACTIVATION_NULL,
    METHOD_CONTEXT_MLP,
)
TRAINABLE_METHODS: Final[tuple[str, ...]] = METHODS[1:]
PAIRWISE_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (METHOD_LINEAR_REFRESH, METHOD_BASE),
    (METHOD_ACTIVATION_NULL, METHOD_BASE),
    (METHOD_CONTEXT_MLP, METHOD_BASE),
    (METHOD_ACTIVATION_NULL, METHOD_LINEAR_REFRESH),
    (METHOD_CONTEXT_MLP, METHOD_LINEAR_REFRESH),
    (METHOD_CONTEXT_MLP, METHOD_ACTIVATION_NULL),
)
SCOPE_CONDITIONS: Final[dict[str, tuple[str, ...]]] = {
    **{condition: (condition,) for condition in CONDITIONS},
    "clean_blur_pooled": tuple(CONDITIONS[:3]),
    "projective_pooled": tuple(CONDITIONS[3:]),
    "all_conditions": tuple(CONDITIONS),
}


class EndpointControlProbeError(ValueError):
    """The source foundation, control arm, or metric input is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointControlProbeError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class MatchedContextResidual(nn.Module):
    """Final-representation residual with an optional GELU treatment."""

    def __init__(self, *, nonlinear: bool, hidden_features: int = 81) -> None:
        super().__init__()
        _require(hidden_features >= 1, "context hidden width must be positive")
        self.nonlinear = bool(nonlinear)
        self.normalization = nn.LayerNorm(EFFICIENTNET_B0_FEATURES)
        self.hidden_projection = nn.Linear(
            EFFICIENTNET_B0_FEATURES, int(hidden_features)
        )
        self.activation = nn.GELU() if self.nonlinear else nn.Identity()
        self.output_projection = nn.Linear(int(hidden_features), 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == EFFICIENTNET_B0_FEATURES,
            "context representation shape differs",
        )
        hidden = self.hidden_projection(self.normalization(representation))
        return self.output_projection(self.activation(hidden)).squeeze(1)


class EndpointControlProbe(nn.Module):
    """Frozen B0 plus endpoint-refresh and matched context controls."""

    def __init__(
        self,
        encoder: SCORTRawEfficientNetB0Encoder,
        base_projection: nn.Linear,
    ) -> None:
        super().__init__()
        _require(
            base_projection.in_features == EFFICIENTNET_B0_FEATURES
            and base_projection.out_features == 1,
            "base endpoint shape differs",
        )
        self.encoder = encoder
        self.base_projection = base_projection
        self.linear_refresh = nn.Linear(EFFICIENTNET_B0_FEATURES, 1)
        self.linear_refresh.load_state_dict(
            self.base_projection.state_dict(), strict=True
        )
        self.context_mlp = MatchedContextResidual(nonlinear=True)
        self.activation_null = MatchedContextResidual(nonlinear=False)
        self.activation_null.load_state_dict(
            self.context_mlp.state_dict(), strict=True
        )
        for module in (self.encoder, self.base_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
        counts = control_parameter_counts(self)
        _require(
            counts[METHOD_CONTEXT_MLP] == counts[METHOD_ACTIVATION_NULL],
            "activation-null and context-MLP parameter counts differ",
        )
        _require(
            _module_states_equal(self.context_mlp, self.activation_null),
            "matched context arms do not share identical initial tensors",
        )

    def train(self, mode: bool = True) -> EndpointControlProbe:
        super().train(mode)
        self.encoder.eval()
        self.base_projection.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.encoder(images)
            representation = features["representation"]
            base_logit = self.base_projection(representation).squeeze(1)
        detached = representation.detach()
        return {
            METHOD_BASE: torch.sigmoid(base_logit),
            METHOD_LINEAR_REFRESH: torch.sigmoid(
                self.linear_refresh(detached).squeeze(1)
            ),
            METHOD_ACTIVATION_NULL: torch.sigmoid(
                base_logit + self.activation_null(detached)
            ),
            METHOD_CONTEXT_MLP: torch.sigmoid(
                base_logit + self.context_mlp(detached)
            ),
        }


def _module_states_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def control_parameter_counts(model: EndpointControlProbe) -> dict[str, int]:
    return {
        METHOD_LINEAR_REFRESH: sum(
            parameter.numel() for parameter in model.linear_refresh.parameters()
        ),
        METHOD_ACTIVATION_NULL: sum(
            parameter.numel() for parameter in model.activation_null.parameters()
        ),
        METHOD_CONTEXT_MLP: sum(
            parameter.numel() for parameter in model.context_mlp.parameters()
        ),
        "foundation_frozen": sum(
            parameter.numel()
            for module in (model.encoder, model.base_projection)
            for parameter in module.parameters()
        ),
        "total_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_inner_foundation(
    checkpoint_path: Path,
    *,
    internal: InternalSceneSplit,
    expected_seed: int,
) -> tuple[LightweightProgressRegressor, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"inner foundation does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "inner foundation is malformed")
    _require(
        checkpoint.get("protocol") == SOURCE_PROBE_PROTOCOL
        and checkpoint.get("phase") == "inner_foundation"
        and checkpoint.get("architecture") == "efficientnet_b0"
        and checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch",
        "inner foundation metadata differs",
    )
    _require(
        int(checkpoint.get("seed", -1)) == int(expected_seed),
        "inner foundation seed differs",
    )
    _require(
        int(checkpoint.get("train_samples", -1)) == len(internal.train)
        and tuple(checkpoint.get("train_scenes", ())) == tuple(internal.train_scenes),
        "inner foundation training partition differs",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "inner foundation state is missing")
    foundation = LightweightProgressRegressor(
        "efficientnet_b0", imagenet_pretrained=False
    )
    foundation.load_state_dict(state, strict=True)
    foundation.eval()
    metadata = {
        "path": str(source),
        "protocol": str(checkpoint["protocol"]),
        "seed": int(checkpoint["seed"]),
        "epochs": int(checkpoint["epochs"]),
        "checkpoint_selection": str(checkpoint["checkpoint_selection"]),
        "train_samples": int(checkpoint["train_samples"]),
    }
    return foundation, metadata


def build_control_from_foundation(
    foundation: LightweightProgressRegressor,
) -> EndpointControlProbe:
    split_probe = build_probe_from_foundation(foundation)
    model = EndpointControlProbe(
        split_probe.encoder,
        split_probe.base_projection,
    )
    del split_probe
    return model


def run_control_epoch(
    model: EndpointControlProbe,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    model.train(True)
    use_amp = device.type == "cuda"
    absolute_error_sums = {method: 0.0 for method in METHODS}
    samples = 0
    optimizer_steps = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(images)
            arm_losses = {
                method: F.l1_loss(outputs[method], targets)
                for method in TRAINABLE_METHODS
            }
            loss = sum(arm_losses.values())
        previous_scale = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
            optimizer_steps += 1
        count = int(targets.numel())
        for method in METHODS:
            absolute_error_sums[method] += float(
                torch.abs(outputs[method].detach() - targets).sum().cpu()
            )
        samples += count
    _require(samples > 0, "control epoch produced no samples")
    return {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "exact_l1": {
            method: absolute_error_sums[method] / samples for method in METHODS
        },
    }


@dataclass(frozen=True, slots=True)
class ControlPredictionRecord:
    sample_id: str
    scene_stem: str
    condition: str
    target: float
    predictions: dict[str, float]

    def as_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_stem": self.scene_stem,
            "condition": self.condition,
            "target": self.target,
            "predictions": dict(self.predictions),
            "absolute_errors": {
                method: abs(value - self.target)
                for method, value in self.predictions.items()
            },
        }


def evaluate_condition(
    model: EndpointControlProbe,
    samples: Sequence[DirectSample],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[ControlPredictionRecord, ...]:
    dataset = ConditionedProgressDataset(samples, condition=condition)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )
    model.eval()
    records: list[ControlPredictionRecord] = []
    with torch.inference_mode():
        for images, targets, sample_ids, scene_stems in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
            cpu_predictions = {
                method: outputs[method].float().cpu().tolist() for method in METHODS
            }
            for row_index, (sample_id, scene_stem, target) in enumerate(
                zip(sample_ids, scene_stems, targets.float().tolist(), strict=True)
            ):
                predictions = {
                    method: float(cpu_predictions[method][row_index])
                    for method in METHODS
                }
                _require(
                    all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in predictions.values()
                    ),
                    "control prediction is outside [0,1]",
                )
                records.append(
                    ControlPredictionRecord(
                        sample_id=str(sample_id),
                        scene_stem=str(scene_stem),
                        condition=condition,
                        target=float(target),
                        predictions=predictions,
                    )
                )
    _require(len(records) == len(samples), "condition evaluation row count differs")
    return tuple(records)


def summarize_records(
    records: Sequence[ControlPredictionRecord],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = tuple(records)
    _require(bool(rows), "metric record set is empty")
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    targets = np.asarray([row.target for row in rows], dtype=np.float64)
    errors = {
        method: np.abs(
            np.asarray([row.predictions[method] for row in rows], dtype=np.float64)
            - targets
        )
        for method in METHODS
    }
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_scene[row.scene_stem].append(index)
    scene_names = tuple(sorted(by_scene))
    _require(len(scene_names) >= 2, "scene bootstrap needs at least two scenes")
    scene_indices = tuple(
        np.asarray(by_scene[scene], dtype=np.int64) for scene in scene_names
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled_scenes = rng.integers(
        0,
        len(scene_names),
        size=(bootstrap_replicates, len(scene_names)),
    )
    comparisons: dict[str, Any] = {}
    for candidate, reference in PAIRWISE_COMPARISONS:
        delta = errors[candidate] - errors[reference]
        replicate_deltas = np.empty(bootstrap_replicates, dtype=np.float64)
        for replicate, selected in enumerate(sampled_scenes):
            selected_rows = np.concatenate(
                tuple(scene_indices[int(index)] for index in selected)
            )
            replicate_deltas[replicate] = float(np.mean(delta[selected_rows]))
        mean_delta = float(np.mean(delta))
        reference_nmae = float(np.mean(errors[reference]))
        comparisons[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate,
            "reference": reference,
            "mean_nmae_delta": mean_delta,
            "relative_error_reduction": (
                -mean_delta / reference_nmae if reference_nmae > 0.0 else None
            ),
            "scene_bootstrap_95_ci": [
                float(np.quantile(replicate_deltas, 0.025)),
                float(np.quantile(replicate_deltas, 0.975)),
            ],
            "scene_bootstrap_probability_delta_below_zero": float(
                np.mean(replicate_deltas < 0.0)
            ),
            "paired_sample_wins": int(np.sum(delta < 0.0)),
            "paired_sample_ties": int(np.sum(delta == 0.0)),
            "paired_sample_losses": int(np.sum(delta > 0.0)),
        }
    return {
        "rows": len(rows),
        "scenes": len(scene_names),
        "conditions": sorted({row.condition for row in rows}),
        "nmae": {method: float(np.mean(errors[method])) for method in METHODS},
        "comparisons": comparisons,
        "bootstrap": {
            "unit": "scene_stem",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
        },
    }


def _checkpoint_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def run_probe(
    *,
    foundation_checkpoint_path: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_REFINER_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid sizes")
    _require(
        learning_rate > 0.0 and weight_decay >= 0.0,
        "invalid optimizer settings",
    )
    root = Path(output_dir).resolve()
    paths = {
        "checkpoint": root / "endpoint_controls.pt",
        "predictions": root / "inner_dev_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "endpoint-control output artifact already exists",
    )
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)
    foundation, foundation_metadata = load_inner_foundation(
        foundation_checkpoint_path,
        internal=internal,
        expected_seed=seed,
    )
    foundation_epochs = int(foundation_metadata["epochs"])
    model = build_control_from_foundation(foundation).to(device)
    del foundation
    counts = control_parameter_counts(model)
    train_dataset = DirectProgressDataset(
        internal.train,
        training=True,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch_index in range(epochs):
        train_dataset.set_epoch(foundation_epochs + epoch_index)
        loader = _loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + 100_000 + epoch_index,
            cuda=device.type == "cuda",
        )
        epoch_started = time.perf_counter()
        metrics = run_control_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        row = {
            "phase": "endpoint_controls",
            "epoch": epoch_index + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "train": metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    training_elapsed = time.perf_counter() - training_started
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "seed": int(seed),
            "epochs": int(epochs),
            "checkpoint_selection": "terminal_fixed_epoch",
            "loss": "exact_l1_each_arm",
            "foundation": foundation_metadata,
            "foundation_frozen": True,
            "same_batches_for_all_arms": True,
            "all_arms_initialized_at_base": True,
            "matched_context_initial_tensors": True,
            "parameter_counts": counts,
            "training_elapsed_seconds": training_elapsed,
            "history": history,
            "linear_refresh_state": _checkpoint_state(model.linear_refresh),
            "activation_null_state": _checkpoint_state(model.activation_null),
            "context_mlp_state": _checkpoint_state(model.context_mlp),
        },
        paths["checkpoint"],
    )

    evaluation_started = time.perf_counter()
    all_records: list[ControlPredictionRecord] = []
    for condition_index, condition in enumerate(CONDITIONS):
        condition_started = time.perf_counter()
        records = evaluate_condition(
            model,
            internal.dev,
            condition=condition,
            device=device,
            batch_size=batch_size,
            workers=workers,
            seed=seed + 200_000 + condition_index,
        )
        all_records.extend(records)
        condition_nmae = {
            method: float(
                np.mean(
                    [abs(row.predictions[method] - row.target) for row in records]
                )
            )
            for method in METHODS
        }
        print(
            json.dumps(
                {
                    "phase": "inner_dev_evaluation",
                    "condition": condition,
                    "rows": len(records),
                    "nmae": condition_nmae,
                    "elapsed_seconds": time.perf_counter() - condition_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    with paths["predictions"].open("w", encoding="utf-8", newline="\n") as stream:
        for record in all_records:
            stream.write(
                json.dumps(
                    record.as_json(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
    scopes = {
        scope: summarize_records(
            tuple(
                row for row in all_records if row.condition in selected_conditions
            ),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=seed + 300_000 + scope_index * 1_009,
        )
        for scope_index, (scope, selected_conditions) in enumerate(
            SCOPE_CONDITIONS.items()
        )
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_question": (
            "Does nonlinear final-representation refinement outperform both "
            "simple exact-L1 endpoint refresh and a parameter-matched "
            "activation-null context control?"
        ),
        "interpretation": {
            "linear_refresh_minus_base": (
                "effect of five additional exact-L1 endpoint epochs"
            ),
            "context_mlp_minus_linear_refresh": (
                "additional effect of the wide normalized context parameterization"
            ),
            "context_mlp_minus_activation_null": (
                "isolated GELU treatment at matched parameter count and initialization"
            ),
            "one_seed_internal_probe_not_paper_table": True,
        },
        "seed": int(seed),
        "device": str(device),
        "foundation": foundation_metadata,
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "formal_holdout_content_access": False,
            "inner_train_samples": len(internal.train),
            "inner_train_scenes": len(internal.train_scenes),
            "inner_dev_samples": len(internal.dev),
            "inner_dev_scenes": len(internal.dev_scenes),
            "conditions": list(CONDITIONS),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "workers": int(workers),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "loss": "exact_l1_each_arm",
            "training_elapsed_seconds": training_elapsed,
            "evaluation_elapsed_seconds": time.perf_counter() - evaluation_started,
        },
        "parameter_counts": counts,
        "scopes": scopes,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["results"], result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation-checkpoint",
        type=Path,
        default=DEFAULT_FOUNDATION_CHECKPOINT,
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_REFINER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_probe(
            foundation_checkpoint_path=args.foundation_checkpoint,
            fit_manifest_path=args.fit_manifest,
            outer_split_path=args.outer_split,
            output_dir=args.output_dir,
            seed=args.seed,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (EndpointControlProbeError, DirectProgressError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    compact = {
        "status": "complete",
        "results": result["artifacts"]["results"],
        "parameter_counts": result["parameter_counts"],
        "clean_blur_pooled": result["scopes"]["clean_blur_pooled"],
        "projective_pooled": result["scopes"]["projective_pooled"],
        "all_conditions": result["scopes"]["all_conditions"],
    }
    print(json.dumps(compact, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EndpointControlProbe",
    "EndpointControlProbeError",
    "METHOD_ACTIVATION_NULL",
    "METHOD_BASE",
    "METHOD_CONTEXT_MLP",
    "METHOD_LINEAR_REFRESH",
    "MatchedContextResidual",
    "PROTOCOL",
    "build_control_from_foundation",
    "control_parameter_counts",
    "load_inner_foundation",
    "run_probe",
    "summarize_records",
]
