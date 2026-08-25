"""Inner-scene probe for a Raw angular-moment progress refiner.

This experiment reuses the completed 30-epoch EfficientNet-B0 foundation and
the completed five-epoch endpoint controls.  It asks whether an explicitly
polar, direction-aware readout extracts useful Raw-view progress information
beyond both a refreshed linear endpoint and a final-representation MLP.

Only the two angular arms are trained here.  The encoder, original endpoint,
linear refresh, and context MLP are loaded from prior terminal checkpoints and
remain frozen.  The formal SyncG holdout and all field cohorts stay untouched.
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
from experiments.geoattn_resnet18_progress import (
    _apply_homography,
    _geometry_targets,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.raw_context_endpoint_control_probe import (
    METHOD_CONTEXT_MLP,
    METHOD_LINEAR_REFRESH,
    PROTOCOL as ENDPOINT_CONTROL_PROTOCOL,
    MatchedContextResidual,
    load_inner_foundation,
)
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_LEARNING_RATE,
    DEFAULT_REFINER_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    build_probe_from_foundation,
    load_inner_scene_population,
)
from experiments.resnet18_direct_progress import (
    IMAGE_SIZE,
    DirectProgressError,
    DirectSample,
    _configure_reproducibility,
    _loader,
    matched_cagh_augmentation,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS, ROBUSTNESS_SEED
from experiments.support_geometry_multiview_efficientnet import (
    EFFICIENTNET_B0_FEATURES,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
)
from experiments.syncg_lightweight_regression_baselines import DEFAULT_SCENE_SPLIT
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    InternalSceneSplit,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


PROTOCOL: Final[str] = "syncg_raw_angular_moment_refiner_probe_v1"
DEFAULT_FOUNDATION_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_multiscale_progress_probe/seed_20262020/"
    "inner_foundation.pt"
)
DEFAULT_ENDPOINT_CONTROL_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_context_endpoint_control_probe/seed_20262020/"
    "endpoint_controls.pt"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/raw_angular_moment_refiner_probe/seed_20262020"
)

METHOD_BASE: Final[str] = "base"
METHOD_ANGULAR_NO_AUX: Final[str] = "angular_no_aux"
METHOD_ANGULAR_DIRECTION_AUX: Final[str] = "angular_direction_aux"
METHODS: Final[tuple[str, ...]] = (
    METHOD_BASE,
    METHOD_LINEAR_REFRESH,
    METHOD_CONTEXT_MLP,
    METHOD_ANGULAR_NO_AUX,
    METHOD_ANGULAR_DIRECTION_AUX,
)
ANGULAR_METHODS: Final[tuple[str, ...]] = (
    METHOD_ANGULAR_NO_AUX,
    METHOD_ANGULAR_DIRECTION_AUX,
)
PAIRWISE_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (METHOD_LINEAR_REFRESH, METHOD_BASE),
    (METHOD_CONTEXT_MLP, METHOD_LINEAR_REFRESH),
    (METHOD_ANGULAR_NO_AUX, METHOD_LINEAR_REFRESH),
    (METHOD_ANGULAR_DIRECTION_AUX, METHOD_LINEAR_REFRESH),
    (METHOD_ANGULAR_DIRECTION_AUX, METHOD_CONTEXT_MLP),
    (METHOD_ANGULAR_DIRECTION_AUX, METHOD_ANGULAR_NO_AUX),
)
SCOPE_CONDITIONS: Final[dict[str, tuple[str, ...]]] = {
    **{condition: (condition,) for condition in CONDITIONS},
    "clean_blur_pooled": tuple(CONDITIONS[:3]),
    "projective_pooled": tuple(CONDITIONS[3:]),
    "all_conditions": tuple(CONDITIONS),
}

POLAR_RADIAL_BINS: Final[int] = 8
POLAR_ANGULAR_BINS: Final[int] = 36
POLAR_MIN_RADIUS: Final[float] = 0.20
POLAR_MAX_RADIUS: Final[float] = 0.95
DIRECTION_AUXILIARY_WEIGHT: Final[float] = 0.01
DIRECTION_SMOOTH_L1_BETA: Final[float] = 0.10


class AngularMomentProbeError(ValueError):
    """The angular probe input, model, optimization, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AngularMomentProbeError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _polar_sampling_grid(
    *,
    radial_bins: int = POLAR_RADIAL_BINS,
    angular_bins: int = POLAR_ANGULAR_BINS,
    minimum_radius: float = POLAR_MIN_RADIUS,
    maximum_radius: float = POLAR_MAX_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a centered grid with zero angle up and clockwise-positive angle."""

    _require(radial_bins >= 2 and angular_bins >= 4, "polar grid is too small")
    _require(
        0.0 <= minimum_radius < maximum_radius <= 1.0,
        "polar radius interval is invalid",
    )
    radii = torch.linspace(minimum_radius, maximum_radius, radial_bins)
    angles = torch.arange(angular_bins, dtype=torch.float32)
    angles = angles * (2.0 * math.pi / float(angular_bins))
    radius_mesh = radii[:, None].expand(radial_bins, angular_bins)
    angle_mesh = angles[None, :].expand(radial_bins, angular_bins)
    # grid_sample coordinates are already centered in [-1, 1].
    x = radius_mesh * torch.sin(angle_mesh)
    y = -radius_mesh * torch.cos(angle_mesh)
    grid = torch.stack((x, y), dim=-1).unsqueeze(0)
    directions = torch.stack((torch.sin(angles), torch.cos(angles)), dim=1)
    return grid, radii, directions


class AngularMomentLogitRefiner(nn.Module):
    """Polar stride-8/16 moments plus a compact final-feature context path."""

    def __init__(
        self,
        *,
        scale_channels: int = 8,
        circular_channels: int = 32,
        angular_channels: int = 16,
        context_features: int = 42,
        fusion_features: int = 68,
        radial_bins: int = POLAR_RADIAL_BINS,
        angular_bins: int = POLAR_ANGULAR_BINS,
    ) -> None:
        super().__init__()
        _require(
            min(
                scale_channels,
                circular_channels,
                angular_channels,
                context_features,
                fusion_features,
            )
            >= 1,
            "angular refiner widths must be positive",
        )
        grid, radii, directions = _polar_sampling_grid(
            radial_bins=radial_bins,
            angular_bins=angular_bins,
        )
        self.radial_bins = int(radial_bins)
        self.angular_bins = int(angular_bins)
        self.register_buffer("polar_grid", grid, persistent=True)
        self.register_buffer("radii", radii, persistent=True)
        self.register_buffer("directions_sin_cos", directions, persistent=True)
        self.stride8_projection = nn.Conv2d(
            RAW_STRIDE8_CHANNELS, int(scale_channels), kernel_size=1
        )
        self.stride16_projection = nn.Conv2d(
            EFFICIENTNET_B0_MIDDLE_FEATURES,
            int(scale_channels),
            kernel_size=1,
        )
        moment_channels = int(scale_channels) * 2 * 2
        self.circular_encoder = nn.Sequential(
            nn.Conv1d(
                moment_channels,
                int(circular_channels),
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                int(circular_channels),
                int(angular_channels),
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.GELU(),
        )
        self.direction_projection = nn.Conv1d(
            int(angular_channels), 1, kernel_size=1
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(EFFICIENTNET_B0_FEATURES),
            nn.Linear(EFFICIENTNET_B0_FEATURES, int(context_features)),
            nn.GELU(),
        )
        fused_features = (
            int(angular_channels) * self.angular_bins
            + int(context_features)
            + 4
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, int(fusion_features)),
            nn.GELU(),
            nn.Linear(int(fusion_features), 1),
        )
        final = self.fusion[-1]
        _require(isinstance(final, nn.Linear), "angular output projection differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _radial_moments(
        self,
        features: torch.Tensor,
        projection: nn.Conv2d,
    ) -> torch.Tensor:
        projected = F.gelu(projection(features))
        grid = self.polar_grid.to(dtype=projected.dtype).expand(
            projected.shape[0], -1, -1, -1
        )
        sampled = F.grid_sample(
            projected,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        zeroth = sampled.mean(dim=2)
        radial_weights = self.radii.to(dtype=sampled.dtype)
        radial_weights = radial_weights / radial_weights.sum()
        first = (sampled * radial_weights[None, None, :, None]).sum(dim=2)
        return torch.cat((zeroth, first), dim=1)

    def forward(
        self,
        stride8: torch.Tensor,
        stride16: torch.Tensor,
        representation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            stride8.ndim == stride16.ndim == 4
            and representation.ndim == 2
            and stride8.shape[0] == stride16.shape[0] == representation.shape[0]
            and stride8.shape[1] == RAW_STRIDE8_CHANNELS
            and stride16.shape[1] == EFFICIENTNET_B0_MIDDLE_FEATURES
            and representation.shape[1] == EFFICIENTNET_B0_FEATURES,
            "angular feature shapes differ",
        )
        moments = torch.cat(
            (
                self._radial_moments(stride8, self.stride8_projection),
                self._radial_moments(stride16, self.stride16_projection),
            ),
            dim=1,
        )
        angular = self.circular_encoder(moments)
        direction_logits = self.direction_projection(angular).squeeze(1)
        posterior = torch.softmax(direction_logits, dim=1)
        direction_basis = self.directions_sin_cos.to(dtype=posterior.dtype)
        direction = posterior @ direction_basis
        concentration = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        entropy = -(posterior * torch.log(posterior.clamp_min(1.0e-7))).sum(
            dim=1, keepdim=True
        ) / math.log(float(self.angular_bins))
        fused = torch.cat(
            (
                angular.flatten(1),
                self.context_projection(representation),
                direction,
                concentration,
                entropy,
            ),
            dim=1,
        )
        return {
            "logit_delta": self.fusion(fused).squeeze(1),
            "direction_sin_cos": direction,
            "direction_concentration": concentration.squeeze(1),
            "direction_entropy": entropy.squeeze(1),
            "direction_posterior": posterior,
        }


class AngularMomentProbe(nn.Module):
    """Frozen prior controls plus matched angular heads with/without direction loss."""

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
        self.linear_refresh.load_state_dict(base_projection.state_dict(), strict=True)
        self.context_mlp = MatchedContextResidual(nonlinear=True)
        self.angular_no_aux = AngularMomentLogitRefiner()
        self.angular_direction_aux = AngularMomentLogitRefiner()
        self.angular_direction_aux.load_state_dict(
            self.angular_no_aux.state_dict(), strict=True
        )
        for module in (
            self.encoder,
            self.base_projection,
            self.linear_refresh,
            self.context_mlp,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
        _require(
            _module_states_equal(self.angular_no_aux, self.angular_direction_aux),
            "angular arms do not share identical initial tensors",
        )

    def train(self, mode: bool = True) -> AngularMomentProbe:
        super().train(mode)
        for module in (
            self.encoder,
            self.base_projection,
            self.linear_refresh,
            self.context_mlp,
        ):
            module.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.encoder(images)
            representation = features["representation"]
            base_logit = self.base_projection(representation).squeeze(1)
            base = torch.sigmoid(base_logit)
            linear = torch.sigmoid(self.linear_refresh(representation).squeeze(1))
            context = torch.sigmoid(base_logit + self.context_mlp(representation))
        detached_stride8 = features["stride8"].detach()
        detached_stride16 = features["stride16"].detach()
        detached_representation = representation.detach()
        no_aux = self.angular_no_aux(
            detached_stride8, detached_stride16, detached_representation
        )
        direction_aux = self.angular_direction_aux(
            detached_stride8, detached_stride16, detached_representation
        )
        return {
            METHOD_BASE: base,
            METHOD_LINEAR_REFRESH: linear,
            METHOD_CONTEXT_MLP: context,
            METHOD_ANGULAR_NO_AUX: torch.sigmoid(
                base_logit + no_aux["logit_delta"]
            ),
            METHOD_ANGULAR_DIRECTION_AUX: torch.sigmoid(
                base_logit + direction_aux["logit_delta"]
            ),
            f"{METHOD_ANGULAR_NO_AUX}_direction": no_aux["direction_sin_cos"],
            f"{METHOD_ANGULAR_DIRECTION_AUX}_direction": direction_aux[
                "direction_sin_cos"
            ],
            f"{METHOD_ANGULAR_NO_AUX}_concentration": no_aux[
                "direction_concentration"
            ],
            f"{METHOD_ANGULAR_DIRECTION_AUX}_concentration": direction_aux[
                "direction_concentration"
            ],
        }


def _module_states_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def angular_parameter_counts(model: AngularMomentProbe) -> dict[str, int]:
    values = {
        METHOD_LINEAR_REFRESH: sum(
            parameter.numel() for parameter in model.linear_refresh.parameters()
        ),
        METHOD_CONTEXT_MLP: sum(
            parameter.numel() for parameter in model.context_mlp.parameters()
        ),
        METHOD_ANGULAR_NO_AUX: sum(
            parameter.numel() for parameter in model.angular_no_aux.parameters()
        ),
        METHOD_ANGULAR_DIRECTION_AUX: sum(
            parameter.numel()
            for parameter in model.angular_direction_aux.parameters()
        ),
        "foundation_frozen": sum(
            parameter.numel()
            for module in (model.encoder, model.base_projection)
            for parameter in module.parameters()
        ),
        "prior_controls_frozen": sum(
            parameter.numel()
            for module in (model.linear_refresh, model.context_mlp)
            for parameter in module.parameters()
        ),
        "total_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    _require(
        values[METHOD_ANGULAR_NO_AUX]
        == values[METHOD_ANGULAR_DIRECTION_AUX],
        "angular arm parameter counts differ",
    )
    return values


def build_angular_probe_from_foundation(foundation: nn.Module) -> AngularMomentProbe:
    split_probe = build_probe_from_foundation(foundation)
    model = AngularMomentProbe(split_probe.encoder, split_probe.base_projection)
    del split_probe
    return model


def load_endpoint_controls(
    checkpoint_path: Path,
    *,
    model: AngularMomentProbe,
    expected_seed: int,
    expected_epochs: int,
    foundation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"endpoint controls do not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "endpoint controls are malformed")
    _require(
        checkpoint.get("protocol") == ENDPOINT_CONTROL_PROTOCOL
        and checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch"
        and checkpoint.get("loss") == "exact_l1_each_arm",
        "endpoint-control metadata differs",
    )
    _require(
        int(checkpoint.get("seed", -1)) == int(expected_seed)
        and int(checkpoint.get("epochs", -1)) == int(expected_epochs),
        "endpoint-control seed or epoch count differs",
    )
    source_foundation = checkpoint.get("foundation")
    _require(
        isinstance(source_foundation, Mapping)
        and int(source_foundation.get("seed", -1))
        == int(foundation_metadata["seed"])
        and int(source_foundation.get("epochs", -1))
        == int(foundation_metadata["epochs"])
        and int(source_foundation.get("train_samples", -1))
        == int(foundation_metadata["train_samples"]),
        "endpoint controls use a different foundation",
    )
    linear_state = checkpoint.get("linear_refresh_state")
    context_state = checkpoint.get("context_mlp_state")
    _require(
        isinstance(linear_state, Mapping) and isinstance(context_state, Mapping),
        "endpoint-control states are missing",
    )
    model.linear_refresh.load_state_dict(linear_state, strict=True)
    model.context_mlp.load_state_dict(context_state, strict=True)
    return {
        "path": str(source),
        "protocol": str(checkpoint["protocol"]),
        "seed": int(checkpoint["seed"]),
        "epochs": int(checkpoint["epochs"]),
        "checkpoint_selection": str(checkpoint["checkpoint_selection"]),
    }


def _normalized_geometry(
    sample: DirectSample,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    _require(
        bool(sample.protected_points_xy),
        f"{sample.sample_id}: geometry landmarks are missing",
    )
    left, top, right, bottom = bounds
    scale = np.asarray(
        [float(right - left), float(bottom - top)], dtype=np.float32
    )
    return (
        np.asarray(sample.protected_points_xy, dtype=np.float32)
        - np.asarray([left, top], dtype=np.float32)
    ) / scale


class AngularProgressDataset(Dataset[dict[str, torch.Tensor]]):
    """Direct-B0 images with exactly co-transformed pointer direction labels."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        training: bool,
        seed: int,
        image_size: int = IMAGE_SIZE,
        augmentation: PhotoAugmentation | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.training = bool(training)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.augmentation = (
            matched_cagh_augmentation()
            if augmentation is None and self.training
            else augmentation or PhotoAugmentation.disabled()
        )
        self.augmentation.validate()
        self.epoch = 0
        _require(bool(self.samples), "angular-progress dataset is empty")
        _require(self.image_size >= 32, "image size is too small")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=self.image_size)
        points = _normalized_geometry(sample, bounds)
        if self.training:
            rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + index * 97
            )
            roi, forward, _geometry_code = _cagh_augment_geometry(
                roi, points, rng, self.augmentation
            )
            points = _apply_homography(points, forward)
            roi, _photo_code = _cagh_augment_photo(
                roi, rng, self.augmentation
            )
        labels = _geometry_targets(points)
        return {
            "image": normalized_rgb_tensor(roi),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "direction_sin_cos": labels["direction_sin_cos"],
        }


class AngularConditionedProgressDataset(
    Dataset[dict[str, torch.Tensor | str]]
):
    """Controlled Raw conditions; direction labels are valid for non-projective rows."""

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
        _require(bool(self.samples), "conditioned angular dataset is empty")
        _require(self.condition in CONDITIONS, "unknown robustness condition")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        points = _normalized_geometry(sample, bounds)
        conditioned, _metadata = robustness_degradations.apply_degradation(
            roi,
            self.condition,
            sample_id=sample.sample_id,
            seed=self.degradation_seed,
        )
        resized = direct_resize_whole_roi(conditioned, size=self.image_size)
        labels = _geometry_targets(points)
        return {
            "image": normalized_rgb_tensor(resized),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "direction_sin_cos": labels["direction_sin_cos"],
            "direction_available": torch.tensor(
                self.condition in CONDITIONS[:3], dtype=torch.bool
            ),
            "sample_id": sample.sample_id,
            "scene_stem": sample.scene_stem,
        }


def run_angular_epoch(
    model: AngularMomentProbe,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    direction_auxiliary_weight: float = DIRECTION_AUXILIARY_WEIGHT,
) -> dict[str, Any]:
    _require(direction_auxiliary_weight >= 0.0, "direction loss weight is negative")
    model.train(True)
    use_amp = device.type == "cuda"
    absolute_error_sums = {method: 0.0 for method in METHODS}
    direction_loss_sum = 0.0
    samples = 0
    optimizer_steps = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=use_amp)
        targets = batch["progress"].to(device, non_blocking=use_amp)
        target_direction = batch["direction_sin_cos"].to(
            device, non_blocking=use_amp
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(images)
            no_aux_loss = F.l1_loss(outputs[METHOD_ANGULAR_NO_AUX], targets)
            aux_progress_loss = F.l1_loss(
                outputs[METHOD_ANGULAR_DIRECTION_AUX], targets
            )
            direction_loss = F.smooth_l1_loss(
                outputs[f"{METHOD_ANGULAR_DIRECTION_AUX}_direction"],
                target_direction,
                beta=DIRECTION_SMOOTH_L1_BETA,
            )
            loss = (
                no_aux_loss
                + aux_progress_loss
                + direction_auxiliary_weight * direction_loss
            )
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
        direction_loss_sum += float(direction_loss.detach().cpu()) * count
        samples += count
    _require(samples > 0, "angular epoch produced no samples")
    return {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "exact_l1": {
            method: absolute_error_sums[method] / samples for method in METHODS
        },
        "direction_smooth_l1": direction_loss_sum / samples,
        "direction_auxiliary_weight": float(direction_auxiliary_weight),
    }


@dataclass(frozen=True, slots=True)
class AngularPredictionRecord:
    sample_id: str
    scene_stem: str
    condition: str
    target: float
    predictions: dict[str, float]
    direction_available: bool
    target_direction: tuple[float, float] | None
    directions: dict[str, tuple[float, float]]
    concentrations: dict[str, float]

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
            "direction_available": self.direction_available,
            "target_direction_sin_cos": self.target_direction,
            "directions_sin_cos": dict(self.directions),
            "direction_concentrations": dict(self.concentrations),
        }


def evaluate_condition(
    model: AngularMomentProbe,
    samples: Sequence[DirectSample],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[AngularPredictionRecord, ...]:
    dataset = AngularConditionedProgressDataset(samples, condition=condition)
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
    records: list[AngularPredictionRecord] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
            cpu_predictions = {
                method: outputs[method].float().cpu().tolist() for method in METHODS
            }
            cpu_directions = {
                method: outputs[f"{method}_direction"].float().cpu().tolist()
                for method in ANGULAR_METHODS
            }
            cpu_concentrations = {
                method: outputs[f"{method}_concentration"].float().cpu().tolist()
                for method in ANGULAR_METHODS
            }
            targets = batch["progress"].float().tolist()
            target_directions = batch["direction_sin_cos"].float().tolist()
            availability = batch["direction_available"].bool().tolist()
            sample_ids = batch["sample_id"]
            scene_stems = batch["scene_stem"]
            for row_index, (sample_id, scene_stem, target) in enumerate(
                zip(sample_ids, scene_stems, targets, strict=True)
            ):
                predictions = {
                    method: float(cpu_predictions[method][row_index])
                    for method in METHODS
                }
                directions = {
                    method: tuple(
                        float(value)
                        for value in cpu_directions[method][row_index]
                    )
                    for method in ANGULAR_METHODS
                }
                concentrations = {
                    method: float(cpu_concentrations[method][row_index])
                    for method in ANGULAR_METHODS
                }
                _require(
                    all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in predictions.values()
                    )
                    and all(
                        math.isfinite(value)
                        for direction in directions.values()
                        for value in direction
                    ),
                    "angular evaluation produced a non-finite prediction",
                )
                available = bool(availability[row_index])
                target_direction = (
                    tuple(float(value) for value in target_directions[row_index])
                    if available
                    else None
                )
                records.append(
                    AngularPredictionRecord(
                        sample_id=str(sample_id),
                        scene_stem=str(scene_stem),
                        condition=condition,
                        target=float(target),
                        predictions=predictions,
                        direction_available=available,
                        target_direction=target_direction,
                        directions=directions,
                        concentrations=concentrations,
                    )
                )
    _require(len(records) == len(samples), "condition evaluation row count differs")
    return tuple(records)


def _angular_error_degrees(
    prediction: tuple[float, float],
    target: tuple[float, float],
) -> float:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    predicted_norm = float(np.linalg.norm(predicted))
    expected_norm = float(np.linalg.norm(expected))
    _require(
        predicted_norm > 1.0e-12 and expected_norm > 1.0e-12,
        "direction vector collapsed during metric computation",
    )
    cosine = float(np.dot(predicted, expected) / (predicted_norm * expected_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _bootstrap_paired_delta(
    delta: np.ndarray,
    *,
    scene_indices: tuple[np.ndarray, ...],
    sampled_scenes: np.ndarray,
) -> np.ndarray:
    replicates = np.empty(len(sampled_scenes), dtype=np.float64)
    for replicate, selected in enumerate(sampled_scenes):
        selected_rows = np.concatenate(
            tuple(scene_indices[int(index)] for index in selected)
        )
        replicates[replicate] = float(np.mean(delta[selected_rows]))
    return replicates


def summarize_records(
    records: Sequence[AngularPredictionRecord],
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
        replicate_deltas = _bootstrap_paired_delta(
            delta,
            scene_indices=scene_indices,
            sampled_scenes=sampled_scenes,
        )
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

    direction_rows = tuple(row for row in rows if row.direction_available)
    direction_summary: dict[str, Any] | None = None
    if direction_rows:
        direction_errors = {
            method: np.asarray(
                [
                    _angular_error_degrees(
                        row.directions[method],
                        row.target_direction,
                    )
                    for row in direction_rows
                    if row.target_direction is not None
                ],
                dtype=np.float64,
            )
            for method in ANGULAR_METHODS
        }
        direction_by_scene: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(direction_rows):
            direction_by_scene[row.scene_stem].append(index)
        direction_scene_names = tuple(sorted(direction_by_scene))
        direction_scene_indices = tuple(
            np.asarray(direction_by_scene[scene], dtype=np.int64)
            for scene in direction_scene_names
        )
        direction_rng = np.random.default_rng(int(bootstrap_seed) + 17)
        direction_sampled_scenes = direction_rng.integers(
            0,
            len(direction_scene_names),
            size=(bootstrap_replicates, len(direction_scene_names)),
        )
        direction_delta = (
            direction_errors[METHOD_ANGULAR_DIRECTION_AUX]
            - direction_errors[METHOD_ANGULAR_NO_AUX]
        )
        direction_replicates = _bootstrap_paired_delta(
            direction_delta,
            scene_indices=direction_scene_indices,
            sampled_scenes=direction_sampled_scenes,
        )
        direction_summary = {
            "rows": len(direction_rows),
            "scenes": len(direction_scene_names),
            "mean_absolute_angular_error_degrees": {
                method: float(np.mean(direction_errors[method]))
                for method in ANGULAR_METHODS
            },
            "median_absolute_angular_error_degrees": {
                method: float(np.median(direction_errors[method]))
                for method in ANGULAR_METHODS
            },
            "mean_concentration": {
                method: float(
                    np.mean([row.concentrations[method] for row in direction_rows])
                )
                for method in ANGULAR_METHODS
            },
            "direction_aux_minus_no_aux_mean_angular_error_degrees": float(
                np.mean(direction_delta)
            ),
            "direction_aux_minus_no_aux_scene_bootstrap_95_ci": [
                float(np.quantile(direction_replicates, 0.025)),
                float(np.quantile(direction_replicates, 0.975)),
            ],
            "direction_aux_minus_no_aux_bootstrap_probability_below_zero": float(
                np.mean(direction_replicates < 0.0)
            ),
        }
    return {
        "rows": len(rows),
        "scenes": len(scene_names),
        "conditions": sorted({row.condition for row in rows}),
        "nmae": {method: float(np.mean(errors[method])) for method in METHODS},
        "comparisons": comparisons,
        "direction": direction_summary,
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
    endpoint_control_checkpoint_path: Path,
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
    direction_auxiliary_weight: float = DIRECTION_AUXILIARY_WEIGHT,
) -> dict[str, Any]:
    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid sizes")
    _require(
        learning_rate > 0.0
        and weight_decay >= 0.0
        and direction_auxiliary_weight >= 0.0,
        "invalid optimizer or auxiliary-loss setting",
    )
    root = Path(output_dir).resolve()
    paths = {
        "checkpoint": root / "angular_refiners.pt",
        "predictions": root / "inner_dev_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "angular-probe output artifact already exists",
    )
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    internal: InternalSceneSplit = load_inner_scene_population(
        fit_manifest_path, outer_split_path
    )
    foundation, foundation_metadata = load_inner_foundation(
        foundation_checkpoint_path,
        internal=internal,
        expected_seed=seed,
    )
    foundation_epochs = int(foundation_metadata["epochs"])
    model = build_angular_probe_from_foundation(foundation)
    del foundation
    endpoint_control_metadata = load_endpoint_controls(
        endpoint_control_checkpoint_path,
        model=model,
        expected_seed=seed,
        expected_epochs=epochs,
        foundation_metadata=foundation_metadata,
    )
    model = model.to(device)
    counts = angular_parameter_counts(model)
    train_dataset = AngularProgressDataset(
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
        metrics = run_angular_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            direction_auxiliary_weight=direction_auxiliary_weight,
        )
        row = {
            "phase": "angular_refiners",
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
            "progress_loss": "exact_l1_each_angular_arm",
            "direction_loss": "smooth_l1_on_posterior_mean_sin_cos",
            "direction_smooth_l1_beta": DIRECTION_SMOOTH_L1_BETA,
            "direction_auxiliary_weight": float(direction_auxiliary_weight),
            "foundation": foundation_metadata,
            "endpoint_controls": endpoint_control_metadata,
            "foundation_and_controls_frozen": True,
            "same_batches_for_both_angular_arms": True,
            "angular_arms_identical_at_initialization": True,
            "zero_initialized_progress_residual": True,
            "parameter_counts": counts,
            "training_elapsed_seconds": training_elapsed,
            "history": history,
            "angular_no_aux_state": _checkpoint_state(model.angular_no_aux),
            "angular_direction_aux_state": _checkpoint_state(
                model.angular_direction_aux
            ),
        },
        paths["checkpoint"],
    )

    evaluation_started = time.perf_counter()
    all_records: list[AngularPredictionRecord] = []
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
                row
                for row in all_records
                if row.condition in selected_conditions
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
            "Does an explicitly polar Raw angular-moment head improve progress "
            "over the trained linear endpoint and context MLP, and does pointer-"
            "direction supervision add value at identical architecture/capacity?"
        ),
        "interpretation": {
            "angular_no_aux_minus_linear_refresh": (
                "effect of polar spatial moments beyond endpoint refresh"
            ),
            "angular_direction_aux_minus_angular_no_aux": (
                "effect of direction supervision at identical architecture"
            ),
            "intended_integration_scope": (
                "replace only relation-unavailable Raw fallback; keep relation-active "
                "projective transport unchanged"
            ),
            "formal_advancement_evidence_sought": (
                "negative angular-versus-linear NMAE deltas in clean, blur_moderate, "
                "and blur_severe; this runner does not automate promotion"
            ),
            "one_seed_internal_probe_not_paper_table": True,
        },
        "seed": int(seed),
        "device": str(device),
        "foundation": foundation_metadata,
        "endpoint_controls": endpoint_control_metadata,
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "formal_holdout_content_access": False,
            "inner_train_samples": len(internal.train),
            "inner_train_scenes": len(internal.train_scenes),
            "inner_dev_samples": len(internal.dev),
            "inner_dev_scenes": len(internal.dev_scenes),
            "conditions": list(CONDITIONS),
            "direction_metrics_conditions": list(CONDITIONS[:3]),
        },
        "architecture": {
            "polar_radial_bins": POLAR_RADIAL_BINS,
            "polar_angular_bins": POLAR_ANGULAR_BINS,
            "polar_radius_interval": [POLAR_MIN_RADIUS, POLAR_MAX_RADIUS],
            "angle_convention": "zero_up_clockwise_positive",
            "radial_statistics": ["zeroth_mean", "first_radius_weighted_mean"],
            "feature_sources": ["stride8", "stride16", "final_representation"],
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "workers": int(workers),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "progress_loss": "exact_l1_each_angular_arm",
            "direction_loss": "smooth_l1_on_posterior_mean_sin_cos",
            "direction_smooth_l1_beta": DIRECTION_SMOOTH_L1_BETA,
            "direction_auxiliary_weight": float(direction_auxiliary_weight),
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
    parser.add_argument(
        "--endpoint-control-checkpoint",
        type=Path,
        default=DEFAULT_ENDPOINT_CONTROL_CHECKPOINT,
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
    parser.add_argument(
        "--direction-auxiliary-weight",
        type=float,
        default=DIRECTION_AUXILIARY_WEIGHT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_probe(
            foundation_checkpoint_path=args.foundation_checkpoint,
            endpoint_control_checkpoint_path=args.endpoint_control_checkpoint,
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
            direction_auxiliary_weight=args.direction_auxiliary_weight,
        )
    except (AngularMomentProbeError, DirectProgressError) as exc:
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
    "ANGULAR_METHODS",
    "AngularConditionedProgressDataset",
    "AngularMomentLogitRefiner",
    "AngularMomentProbe",
    "AngularMomentProbeError",
    "AngularPredictionRecord",
    "AngularProgressDataset",
    "DIRECTION_AUXILIARY_WEIGHT",
    "METHOD_ANGULAR_DIRECTION_AUX",
    "METHOD_ANGULAR_NO_AUX",
    "METHOD_BASE",
    "PROTOCOL",
    "_polar_sampling_grid",
    "angular_parameter_counts",
    "build_angular_probe_from_foundation",
    "run_angular_epoch",
    "run_probe",
    "summarize_records",
]
