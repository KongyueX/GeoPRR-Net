"""Matched inner-scene probe for a Raw multi-scale progress refiner.

The probe asks a narrow question before ReMST is changed: after training an
EfficientNet-B0 scalar foundation only on the fixed 117-scene inner-train
partition, do its stride-8/stride-16 maps contain useful progress information
beyond the final 1,280-dimensional representation?

Three predictions are evaluated on the untouched 14-scene inner-dev split:

``base``
    The terminal scalar EfficientNet-B0 prediction.
``context_mlp``
    A logit-residual MLP that sees only the final global representation.
``raw_multiscale``
    A parameter-matched logit-residual head that retains 4x4 spatial cells from
    stride-8 and stride-16 maps and also sees the final representation.

Both refiners share the same frozen foundation, start exactly at ``base``, see
the same augmented batches, use exact L1, and are optimized together with
disjoint parameters.  The formal SyncG holdout and all field cohorts are
outside this runner.
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments import robustness_degradations
from experiments.a11_scort import (
    RAW_STRIDE8_CHANNELS,
    SCORTRawEfficientNetB0Encoder,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    IMAGE_SIZE,
    DirectProgressDataset,
    DirectProgressError,
    DirectSample,
    _configure_reproducibility,
    _epoch,
    _loader,
    load_split_roster,
    load_syncg_samples,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS, ROBUSTNESS_SEED
from experiments.sgca_syncg_internal_pilot import FIT_SAMPLES, FIT_SCENES
from experiments.support_geometry_multiview_efficientnet import (
    EFFICIENTNET_B0_FEATURES,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
)
from experiments.syncg_lightweight_regression_baselines import (
    BACKBONE_SPECS,
    DEFAULT_SCENE_SPLIT,
    LightweightProgressRegressor,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    InternalSceneSplit,
    make_internal_scene_split,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


PROTOCOL: Final[str] = "syncg_raw_multiscale_progress_refiner_probe_v1"
DEFAULT_FIT_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/screen200_seed20262020/"
    "internal_diagnostics/syncg_formal_fit_14442.jsonl"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/raw_multiscale_progress_probe/seed_20262020"
)
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_FOUNDATION_EPOCHS: Final[int] = 30
DEFAULT_REFINER_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 3.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 2_000

METHOD_BASE: Final[str] = "base"
METHOD_CONTEXT: Final[str] = "context_mlp"
METHOD_MULTISCALE: Final[str] = "raw_multiscale"
METHODS: Final[tuple[str, ...]] = (
    METHOD_BASE,
    METHOD_CONTEXT,
    METHOD_MULTISCALE,
)
PAIRWISE_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (METHOD_MULTISCALE, METHOD_BASE),
    (METHOD_MULTISCALE, METHOD_CONTEXT),
    (METHOD_CONTEXT, METHOD_BASE),
)
SCOPE_CONDITIONS: Final[dict[str, tuple[str, ...]]] = {
    **{condition: (condition,) for condition in CONDITIONS},
    "clean_blur_pooled": tuple(CONDITIONS[:3]),
    "projective_pooled": tuple(CONDITIONS[3:]),
    "all_conditions": tuple(CONDITIONS),
}


class RawMultiscaleProbeError(ValueError):
    """The probe input, model, optimization, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawMultiscaleProbeError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_inner_scene_population(
    fit_manifest_path: Path,
    outer_split_path: Path,
) -> InternalSceneSplit:
    """Load the Fit-only population and apply the pre-existing 117/14 split."""

    samples = load_syncg_samples(fit_manifest_path)
    roster = load_split_roster(outer_split_path)
    sample_ids = {sample.sample_id for sample in samples}
    _require(roster.scene_disjoint, "outer SyncG split is not scene-disjoint")
    _require(
        sample_ids == set(roster.train_ids),
        "Fit-only manifest does not match the outer split's train roster",
    )
    _require(len(samples) == FIT_SAMPLES, "Fit-only sample count is not 14,442")
    _require(
        len({sample.scene_stem for sample in samples}) == FIT_SCENES,
        "Fit-only scene count is not 131",
    )
    return make_internal_scene_split(samples)


class ContextOnlyLogitRefiner(nn.Module):
    """Capacity control using only the final pooled B0 representation."""

    def __init__(self, *, hidden_features: int = 81) -> None:
        super().__init__()
        _require(hidden_features >= 1, "context hidden width must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(EFFICIENTNET_B0_FEATURES),
            nn.Linear(EFFICIENTNET_B0_FEATURES, int(hidden_features)),
            nn.GELU(),
            nn.Linear(int(hidden_features), 1),
        )
        final = self.network[-1]
        _require(isinstance(final, nn.Linear), "context output projection differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == EFFICIENTNET_B0_FEATURES,
            "context representation shape differs",
        )
        return self.network(representation).squeeze(1)


class RawMultiScaleLogitRefiner(nn.Module):
    """Spatial Raw refiner retaining 4x4 cells at strides 8 and 16."""

    def __init__(
        self,
        *,
        scale_channels: int = 8,
        context_features: int = 64,
        fusion_features: int = 64,
        spatial_cells: int = 4,
    ) -> None:
        super().__init__()
        _require(
            min(scale_channels, context_features, fusion_features, spatial_cells) >= 1,
            "multi-scale widths must be positive",
        )
        self.spatial_cells = int(spatial_cells)
        self.stride8_projection = nn.Conv2d(
            RAW_STRIDE8_CHANNELS, int(scale_channels), kernel_size=1
        )
        self.stride16_projection = nn.Conv2d(
            EFFICIENTNET_B0_MIDDLE_FEATURES,
            int(scale_channels),
            kernel_size=1,
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(EFFICIENTNET_B0_FEATURES),
            nn.Linear(EFFICIENTNET_B0_FEATURES, int(context_features)),
            nn.GELU(),
        )
        scale_features = int(scale_channels) * self.spatial_cells**2
        fused_features = scale_features * 2 + int(context_features)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, int(fusion_features)),
            nn.GELU(),
            nn.Linear(int(fusion_features), 1),
        )
        final = self.fusion[-1]
        _require(isinstance(final, nn.Linear), "multi-scale output projection differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _spatial_vector(
        self,
        features: torch.Tensor,
        projection: nn.Conv2d,
    ) -> torch.Tensor:
        projected = F.gelu(projection(features))
        pooled = F.adaptive_avg_pool2d(
            projected, output_size=(self.spatial_cells, self.spatial_cells)
        )
        return pooled.flatten(1)

    def forward(
        self,
        stride8: torch.Tensor,
        stride16: torch.Tensor,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        _require(
            stride8.ndim == stride16.ndim == 4
            and stride8.shape[0] == stride16.shape[0] == representation.shape[0]
            and stride8.shape[1] == RAW_STRIDE8_CHANNELS
            and stride16.shape[1] == EFFICIENTNET_B0_MIDDLE_FEATURES,
            "multi-scale feature shapes differ",
        )
        fused = torch.cat(
            (
                self._spatial_vector(stride8, self.stride8_projection),
                self._spatial_vector(stride16, self.stride16_projection),
                self.context_projection(representation),
            ),
            dim=1,
        )
        return self.fusion(fused).squeeze(1)


class MatchedRawRefinerProbe(nn.Module):
    """Frozen B0 scalar foundation plus two matched residual refiners."""

    def __init__(
        self,
        encoder: SCORTRawEfficientNetB0Encoder,
        base_projection: nn.Linear,
    ) -> None:
        super().__init__()
        _require(
            base_projection.in_features == EFFICIENTNET_B0_FEATURES
            and base_projection.out_features == 1,
            "base scalar projection shape differs",
        )
        self.encoder = encoder
        self.base_projection = base_projection
        self.context_head = ContextOnlyLogitRefiner()
        self.multiscale_head = RawMultiScaleLogitRefiner()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base_projection.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.base_projection.eval()
        counts = refiner_parameter_counts(self)
        relative_difference = abs(
            counts[METHOD_CONTEXT] - counts[METHOD_MULTISCALE]
        ) / max(counts[METHOD_CONTEXT], counts[METHOD_MULTISCALE])
        _require(
            relative_difference <= 0.01,
            "context and multi-scale trainable capacities differ by more than 1%",
        )

    def train(self, mode: bool = True) -> MatchedRawRefinerProbe:
        super().train(mode)
        self.encoder.eval()
        self.base_projection.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.encoder(images)
            representation = features["representation"]
            base_logit = self.base_projection(representation).squeeze(1)
        context_delta = self.context_head(representation.detach())
        multiscale_delta = self.multiscale_head(
            features["stride8"].detach(),
            features["stride16"].detach(),
            representation.detach(),
        )
        return {
            METHOD_BASE: torch.sigmoid(base_logit),
            METHOD_CONTEXT: torch.sigmoid(base_logit + context_delta),
            METHOD_MULTISCALE: torch.sigmoid(base_logit + multiscale_delta),
            "context_logit_delta": context_delta,
            "multiscale_logit_delta": multiscale_delta,
        }


def refiner_parameter_counts(model: MatchedRawRefinerProbe) -> dict[str, int]:
    return {
        METHOD_CONTEXT: sum(
            parameter.numel() for parameter in model.context_head.parameters()
        ),
        METHOD_MULTISCALE: sum(
            parameter.numel() for parameter in model.multiscale_head.parameters()
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


def build_probe_from_foundation(
    foundation: LightweightProgressRegressor,
) -> MatchedRawRefinerProbe:
    """Copy a direct B0 into the split encoder without changing its endpoint."""

    _require(
        foundation.architecture == "efficientnet_b0",
        "probe foundation must be EfficientNet-B0",
    )
    encoder = SCORTRawEfficientNetB0Encoder(imagenet_pretrained=False)
    source_stages = tuple(foundation.backbone.features.children())
    target_stages = (
        *tuple(encoder.to_stride8.children()),
        *tuple(encoder.to_stride16.children()),
        *tuple(encoder.to_final.children()),
    )
    _require(
        len(source_stages) == len(target_stages) == 9,
        "EfficientNet-B0 stage layout differs",
    )
    for source, target in zip(source_stages, target_stages, strict=True):
        target.load_state_dict(source.state_dict(), strict=True)
    source_projection = foundation.backbone.classifier[-1]
    _require(
        isinstance(source_projection, nn.Linear)
        and source_projection.in_features == EFFICIENTNET_B0_FEATURES
        and source_projection.out_features == 1,
        "foundation endpoint projection differs",
    )
    base_projection = nn.Linear(EFFICIENTNET_B0_FEATURES, 1)
    base_projection.load_state_dict(source_projection.state_dict(), strict=True)
    return MatchedRawRefinerProbe(encoder, base_projection)


class ConditionedProgressDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]
):
    """One deterministic controlled condition over canonical tight Raw ROIs."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        condition: str,
        degradation_seed: int = ROBUSTNESS_SEED,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.samples = tuple(samples)
        self.condition = str(condition)
        self.degradation_seed = int(degradation_seed)
        self.image_size = int(image_size)
        _require(bool(self.samples), "conditioned dataset is empty")
        _require(self.condition in CONDITIONS, "unknown robustness condition")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        conditioned, _metadata = robustness_degradations.apply_degradation(
            roi,
            self.condition,
            sample_id=sample.sample_id,
            seed=self.degradation_seed,
        )
        resized = direct_resize_whole_roi(conditioned, size=self.image_size)
        return (
            normalized_rgb_tensor(resized),
            torch.tensor(sample.normalized_target, dtype=torch.float32),
            sample.sample_id,
            sample.scene_stem,
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
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


def run_refiner_epoch(
    model: MatchedRawRefinerProbe,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    """Train both disjoint refiners on the same frozen features and exact L1."""

    model.train(True)
    use_amp = device.type == "cuda"
    totals = {METHOD_CONTEXT: 0.0, METHOD_MULTISCALE: 0.0}
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
            context_loss = F.l1_loss(outputs[METHOD_CONTEXT], targets)
            multiscale_loss = F.l1_loss(outputs[METHOD_MULTISCALE], targets)
            loss = context_loss + multiscale_loss
        scaler.scale(loss).backward()
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
            optimizer_steps += 1
        count = int(targets.numel())
        totals[METHOD_CONTEXT] += float(context_loss.detach().cpu()) * count
        totals[METHOD_MULTISCALE] += float(multiscale_loss.detach().cpu()) * count
        samples += count
    _require(samples > 0, "refiner epoch produced no samples")
    return {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "exact_l1": {
            method: totals[method] / samples
            for method in (METHOD_CONTEXT, METHOD_MULTISCALE)
        },
    }


def evaluate_condition(
    model: MatchedRawRefinerProbe,
    samples: Sequence[DirectSample],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[PredictionRecord, ...]:
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
    records: list[PredictionRecord] = []
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
            target_values = targets.float().tolist()
            for row_index, (sample_id, scene_stem, target) in enumerate(
                zip(sample_ids, scene_stems, target_values, strict=True)
            ):
                predictions = {
                    method: float(cpu_predictions[method][row_index])
                    for method in METHODS
                }
                _require(
                    all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in predictions.values()),
                    "probe prediction is outside [0,1]",
                )
                records.append(
                    PredictionRecord(
                        sample_id=str(sample_id),
                        scene_stem=str(scene_stem),
                        condition=condition,
                        target=float(target),
                        predictions=predictions,
                    )
                )
    _require(len(records) == len(samples), "condition evaluation row count differs")
    return tuple(records)


def _quantile(values: np.ndarray, probability: float) -> float:
    _require(values.ndim == 1 and values.size >= 1, "quantile vector is empty")
    return float(np.quantile(values, probability))


def summarize_records(
    records: Sequence[PredictionRecord],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compute NMAE and scene-cluster paired bootstrap intervals."""

    rows = tuple(records)
    _require(bool(rows), "metric record set is empty")
    _require(bootstrap_replicates >= 1, "bootstrap replicate count must be positive")
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
    sampled_scene_indices = rng.integers(
        0,
        len(scene_names),
        size=(bootstrap_replicates, len(scene_names)),
    )
    comparisons: dict[str, Any] = {}
    for candidate, reference in PAIRWISE_COMPARISONS:
        delta = errors[candidate] - errors[reference]
        replicate_deltas = np.empty(bootstrap_replicates, dtype=np.float64)
        for replicate, selected_scenes in enumerate(sampled_scene_indices):
            selected_rows = np.concatenate(
                tuple(scene_indices[int(index)] for index in selected_scenes)
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
                _quantile(replicate_deltas, 0.025),
                _quantile(replicate_deltas, 0.975),
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


def summarize_scopes(
    records: Sequence[PredictionRecord],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = tuple(records)
    _require(
        set(row.condition for row in rows) == set(CONDITIONS),
        "evaluation condition roster differs",
    )
    return {
        scope: summarize_records(
            tuple(row for row in rows if row.condition in selected_conditions),
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + scope_index * 1_009,
        )
        for scope_index, (scope, selected_conditions) in enumerate(
            SCOPE_CONDITIONS.items()
        )
    }


def _checkpoint_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def run_probe(
    *,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    foundation_epochs: int = DEFAULT_FOUNDATION_EPOCHS,
    refiner_epochs: int = DEFAULT_REFINER_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    _require(
        foundation_epochs >= 1
        and refiner_epochs >= 1
        and batch_size >= 1
        and workers >= 0,
        "probe training sizes are invalid",
    )
    _require(
        learning_rate > 0.0 and weight_decay >= 0.0,
        "probe optimizer settings are invalid",
    )
    root = Path(output_dir).resolve()
    paths = {
        "foundation": root / "inner_foundation.pt",
        "refiners": root / "matched_refiners.pt",
        "predictions": root / "inner_dev_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "probe output artifact already exists",
    )
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)

    train_dataset = DirectProgressDataset(
        internal.train,
        training=True,
        seed=seed,
    )
    foundation = LightweightProgressRegressor(
        "efficientnet_b0", imagenet_pretrained=True
    ).to(device)
    foundation_optimizer = torch.optim.AdamW(
        foundation.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    foundation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        foundation_optimizer, T_max=foundation_epochs
    )
    foundation_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    foundation_history: list[dict[str, Any]] = []
    phase_started = time.perf_counter()
    for epoch_index in range(foundation_epochs):
        train_dataset.set_epoch(epoch_index)
        loader = _loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + epoch_index,
            cuda=device.type == "cuda",
        )
        epoch_started = time.perf_counter()
        metrics = _epoch(
            foundation,
            loader,
            device=device,
            optimizer=foundation_optimizer,
            scaler=foundation_scaler,
        )
        row = {
            "phase": "inner_foundation",
            "epoch": epoch_index + 1,
            "learning_rate": float(foundation_optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "train": metrics,
        }
        foundation_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        foundation_scheduler.step()
    foundation_elapsed = time.perf_counter() - phase_started
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "phase": "inner_foundation",
            "architecture": "efficientnet_b0",
            "pretrained_weights": BACKBONE_SPECS["efficientnet_b0"].weights_name,
            "seed": int(seed),
            "epochs": int(foundation_epochs),
            "checkpoint_selection": "terminal_fixed_epoch",
            "loss": "smooth_l1_beta_0.05",
            "train_samples": len(internal.train),
            "train_scenes": list(internal.train_scenes),
            "inner_dev_access_during_training": False,
            "training_elapsed_seconds": foundation_elapsed,
            "history": foundation_history,
            "model_state": _checkpoint_state(foundation),
        },
        paths["foundation"],
    )

    foundation = foundation.to("cpu").eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    probe = build_probe_from_foundation(foundation).to(device)
    del foundation
    counts = refiner_parameter_counts(probe)
    trainable = [parameter for parameter in probe.parameters() if parameter.requires_grad]
    refiner_optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    refiner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        refiner_optimizer, T_max=refiner_epochs
    )
    refiner_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    refiner_history: list[dict[str, Any]] = []
    phase_started = time.perf_counter()
    for epoch_index in range(refiner_epochs):
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
        metrics = run_refiner_epoch(
            probe,
            loader,
            device=device,
            optimizer=refiner_optimizer,
            scaler=refiner_scaler,
        )
        row = {
            "phase": "matched_refiners",
            "epoch": epoch_index + 1,
            "learning_rate": float(refiner_optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "train": metrics,
        }
        refiner_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        refiner_scheduler.step()
    refiner_elapsed = time.perf_counter() - phase_started
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "phase": "matched_refiners",
            "architecture": {
                METHOD_CONTEXT: "final-representation logit-residual MLP",
                METHOD_MULTISCALE: (
                    "Raw stride8/stride16 4x4 spatial cells plus final representation"
                ),
            },
            "seed": int(seed),
            "epochs": int(refiner_epochs),
            "checkpoint_selection": "terminal_fixed_epoch",
            "loss": "exact_l1_each_arm",
            "foundation_frozen": True,
            "same_batches_for_both_arms": True,
            "zero_initialized_at_base": True,
            "parameter_counts": counts,
            "training_elapsed_seconds": refiner_elapsed,
            "history": refiner_history,
            "context_head_state": _checkpoint_state(probe.context_head),
            "multiscale_head_state": _checkpoint_state(probe.multiscale_head),
        },
        paths["refiners"],
    )

    evaluation_started = time.perf_counter()
    all_records: list[PredictionRecord] = []
    for condition_index, condition in enumerate(CONDITIONS):
        condition_started = time.perf_counter()
        records = evaluate_condition(
            probe,
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
                    [
                        abs(row.predictions[method] - row.target)
                        for row in records
                    ]
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
    scopes = summarize_scopes(
        all_records,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 300_000,
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_question": (
            "Does Raw stride-8/stride-16 spatial information improve progress "
            "over both the frozen B0 endpoint and a parameter-matched final-feature MLP?"
        ),
        "interpretation": (
            "A raw_multiscale advantage over context_mlp isolates spatial multi-scale "
            "information from generic added capacity; this one-seed probe is not a "
            "paper-table estimate."
        ),
        "seed": int(seed),
        "device": str(device),
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "formal_holdout_content_access": False,
            "inner_train_samples": len(internal.train),
            "inner_train_scenes": len(internal.train_scenes),
            "inner_dev_samples": len(internal.dev),
            "inner_dev_scenes": len(internal.dev_scenes),
            "inner_dev_scene_stems": list(internal.dev_scenes),
            "conditions": list(CONDITIONS),
        },
        "training": {
            "foundation_epochs": int(foundation_epochs),
            "refiner_epochs": int(refiner_epochs),
            "batch_size": int(batch_size),
            "workers": int(workers),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "foundation_loss": "smooth_l1_beta_0.05",
            "refiner_loss": "exact_l1_each_arm",
            "foundation_elapsed_seconds": foundation_elapsed,
            "refiner_elapsed_seconds": refiner_elapsed,
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
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--foundation-epochs", type=int, default=DEFAULT_FOUNDATION_EPOCHS
    )
    parser.add_argument("--refiner-epochs", type=int, default=DEFAULT_REFINER_EPOCHS)
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
            fit_manifest_path=args.fit_manifest,
            outer_split_path=args.outer_split,
            output_dir=args.output_dir,
            seed=args.seed,
            device_name=args.device,
            foundation_epochs=args.foundation_epochs,
            refiner_epochs=args.refiner_epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (RawMultiscaleProbeError, DirectProgressError) as exc:
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
    "ContextOnlyLogitRefiner",
    "METHOD_BASE",
    "METHOD_CONTEXT",
    "METHOD_MULTISCALE",
    "MatchedRawRefinerProbe",
    "PredictionRecord",
    "PROTOCOL",
    "RawMultiScaleLogitRefiner",
    "RawMultiscaleProbeError",
    "build_probe_from_foundation",
    "load_inner_scene_population",
    "refiner_parameter_counts",
    "run_probe",
    "summarize_records",
]
