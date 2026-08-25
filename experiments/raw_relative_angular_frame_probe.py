"""Inner-scene probe for a scale-relative Raw angular frame refiner.

The preceding angular-moment probe recovered absolute pointer direction to
about two degrees but did not improve scalar progress.  This continuation tests
the task-aligned hypothesis: progress is the pointer phase relative to the
ordered scale-start/scale-end angular frame, not the absolute pointer angle.

Two identical relative-frame heads share one frozen EfficientNet-B0 foundation
and the same batches.  Both predict progress from stride-8/stride-16 polar
moments, three role-specific circular posteriors (pointer, scale start, scale
end), explicit circular pair features, and a differentiable clockwise phase
ratio.  Only ``relative_frame_aux`` receives the fixed direction auxiliary
loss.  Previously trained linear, context-MLP, and absolute-angular controls
remain frozen.  The formal SyncG holdout and field cohorts are not accessed.
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
from experiments.a11_scort import SCORTRawEfficientNetB0Encoder
from experiments.geoattn_resnet18_progress import _apply_homography
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.raw_angular_moment_refiner_probe import (
    DIRECTION_AUXILIARY_WEIGHT,
    DIRECTION_SMOOTH_L1_BETA,
    METHOD_ANGULAR_NO_AUX,
    POLAR_ANGULAR_BINS,
    POLAR_RADIAL_BINS,
    PROTOCOL as ABSOLUTE_ANGULAR_PROTOCOL,
    AngularMomentLogitRefiner,
    _normalized_geometry,
    load_endpoint_controls,
)
from experiments.raw_context_endpoint_control_probe import (
    METHOD_CONTEXT_MLP,
    METHOD_LINEAR_REFRESH,
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


PROTOCOL: Final[str] = "syncg_raw_relative_angular_frame_probe_v2"
DEFAULT_FOUNDATION_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_multiscale_progress_probe/seed_20262020/"
    "inner_foundation.pt"
)
DEFAULT_ENDPOINT_CONTROL_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_context_endpoint_control_probe/seed_20262020/"
    "endpoint_controls.pt"
)
DEFAULT_ABSOLUTE_ANGULAR_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/raw_angular_moment_refiner_probe/seed_20262020/"
    "angular_refiners.pt"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/raw_relative_angular_frame_probe/seed_20262020_fp32_phase"
)

METHOD_BASE: Final[str] = "base"
METHOD_PRIOR_ANGULAR: Final[str] = "prior_angular_no_aux"
METHOD_RELATIVE_FRAME_NO_AUX: Final[str] = "relative_frame_no_aux"
METHOD_RELATIVE_FRAME_AUX: Final[str] = "relative_frame_aux"
METHODS: Final[tuple[str, ...]] = (
    METHOD_BASE,
    METHOD_LINEAR_REFRESH,
    METHOD_CONTEXT_MLP,
    METHOD_PRIOR_ANGULAR,
    METHOD_RELATIVE_FRAME_NO_AUX,
    METHOD_RELATIVE_FRAME_AUX,
)
FRAME_METHODS: Final[tuple[str, ...]] = (
    METHOD_RELATIVE_FRAME_NO_AUX,
    METHOD_RELATIVE_FRAME_AUX,
)
FRAME_ROLES: Final[tuple[str, ...]] = (
    "pointer",
    "scale_start",
    "scale_end",
)
PAIRWISE_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (METHOD_LINEAR_REFRESH, METHOD_BASE),
    (METHOD_CONTEXT_MLP, METHOD_LINEAR_REFRESH),
    (METHOD_PRIOR_ANGULAR, METHOD_LINEAR_REFRESH),
    (METHOD_RELATIVE_FRAME_NO_AUX, METHOD_LINEAR_REFRESH),
    (METHOD_RELATIVE_FRAME_AUX, METHOD_LINEAR_REFRESH),
    (METHOD_RELATIVE_FRAME_NO_AUX, METHOD_PRIOR_ANGULAR),
    (METHOD_RELATIVE_FRAME_AUX, METHOD_PRIOR_ANGULAR),
    (METHOD_RELATIVE_FRAME_AUX, METHOD_RELATIVE_FRAME_NO_AUX),
)
SCOPE_CONDITIONS: Final[dict[str, tuple[str, ...]]] = {
    **{condition: (condition,) for condition in CONDITIONS},
    "clean_blur_pooled": tuple(CONDITIONS[:3]),
    "projective_pooled": tuple(CONDITIONS[3:]),
    "all_conditions": tuple(CONDITIONS),
}


class RelativeAngularFrameProbeError(ValueError):
    """The relative-frame probe input, model, optimization, or metric is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RelativeAngularFrameProbeError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _unit_clockwise_direction(delta: np.ndarray) -> np.ndarray:
    vector = np.asarray(delta, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    _require(norm > 1.0e-6 and math.isfinite(norm), "angular-frame ray collapsed")
    return np.asarray([vector[0] / norm, -vector[1] / norm], dtype=np.float32)


def _clockwise_angle(direction_sin_cos: np.ndarray) -> float:
    value = np.asarray(direction_sin_cos, dtype=np.float64)
    return math.atan2(float(value[0]), float(value[1])) % (2.0 * math.pi)


def angular_frame_targets(points: np.ndarray) -> dict[str, torch.Tensor]:
    """Decode pointer/start/end directions and their clockwise progress ratio."""

    values = np.asarray(points, dtype=np.float32)
    _require(
        values.ndim == 2 and values.shape[1] == 2 and len(values) >= 9,
        "angular-frame supervision must contain >=7 marks plus pivot/tip",
    )
    _require(bool(np.isfinite(values).all()), "angular-frame label is non-finite")
    start = values[0]
    end = values[-3]
    pivot = values[-2]
    tip = values[-1]
    frame = np.stack(
        (
            _unit_clockwise_direction(tip - pivot),
            _unit_clockwise_direction(start - pivot),
            _unit_clockwise_direction(end - pivot),
        ),
        axis=0,
    )
    pointer_angle, start_angle, end_angle = (
        _clockwise_angle(frame[index]) for index in range(3)
    )
    span = (end_angle - start_angle) % (2.0 * math.pi)
    phase = (pointer_angle - start_angle) % (2.0 * math.pi)
    _require(span > 1.0e-6, "scale angular span collapsed")
    return {
        "frame_sin_cos": torch.from_numpy(frame),
        "oracle_phase_progress": torch.tensor(
            phase / span, dtype=torch.float32
        ),
    }


class RelativeAngularFrameLogitRefiner(AngularMomentLogitRefiner):
    """Three semantic angular posteriors plus explicit scale-relative phase."""

    def __init__(self) -> None:
        super().__init__()
        angular_channels = 16
        context_features = 42
        fusion_features = 68
        self.direction_projection = nn.Conv1d(
            angular_channels, len(FRAME_ROLES), kernel_size=1
        )
        frame_statistics = 22
        fused_features = (
            angular_channels * POLAR_ANGULAR_BINS
            + context_features
            + frame_statistics
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Linear(fused_features, fusion_features),
            nn.GELU(),
            nn.Linear(fusion_features, 1),
        )
        final = self.fusion[-1]
        _require(isinstance(final, nn.Linear), "relative-frame output differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        dot = (left * right).sum(dim=1, keepdim=True)
        cross = (
            left[:, 0:1] * right[:, 1:2]
            - left[:, 1:2] * right[:, 0:1]
        )
        return torch.cat((dot, cross), dim=1)

    def _frame_statistics(
        self, angular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.direction_projection(angular)
        posterior = torch.softmax(logits, dim=2)
        direction_basis = self.directions_sin_cos.to(dtype=posterior.dtype)
        directions = torch.einsum("brt,td->brd", posterior, direction_basis)
        concentrations = torch.linalg.vector_norm(directions, dim=2)
        entropies = -(
            posterior * torch.log(posterior.clamp_min(1.0e-7))
        ).sum(dim=2) / math.log(float(self.angular_bins))

        pointer = directions[:, 0]
        start = directions[:, 1]
        end = directions[:, 2]
        pair_features = torch.cat(
            (
                self._pair_features(pointer, start),
                self._pair_features(pointer, end),
                self._pair_features(end, start),
            ),
            dim=1,
        )

        # A tiny upward bias makes atan2 well-defined when a new posterior is
        # nearly uniform.  Concentration/entropy remain available to the fusion
        # layer, so it can discount uncertain phase values.
        # Under AMP the posterior means are float16.  Atan2Backward evaluates
        # squared inputs; near-uniform circular posteriors can therefore make
        # its half-precision denominator underflow and return NaN even when the
        # upstream progress gradient is exactly zero.  Preserve the original
        # atan2/remainder geometry, but evaluate this numerical subgraph in
        # float32 and cast only its bounded fusion features back to the feature
        # dtype.  Direction supervision and every architectural connection are
        # otherwise unchanged.
        stable_directions = directions.float().clone()
        stable_directions[:, :, 1] = stable_directions[:, :, 1] + 1.0e-4
        angles = torch.atan2(
            stable_directions[:, :, 0], stable_directions[:, :, 1]
        )
        two_pi = 2.0 * math.pi
        span = torch.remainder(angles[:, 2] - angles[:, 1], two_pi)
        phase = torch.remainder(angles[:, 0] - angles[:, 1], two_pi)
        minimum_span = two_pi / float(self.angular_bins)
        phase_ratio = (phase / span.clamp_min(minimum_span)).clamp(0.0, 2.0)
        phase_features = torch.stack(
            (
                phase / two_pi,
                span / two_pi,
                phase_ratio,
                (span - phase) / two_pi,
            ),
            dim=1,
        ).to(dtype=directions.dtype)
        statistics = torch.cat(
            (
                directions.flatten(1),
                concentrations,
                entropies,
                pair_features,
                phase_features,
            ),
            dim=1,
        )
        _require(
            statistics.shape[1] == 22,
            "relative angular-frame statistic width differs",
        )
        return statistics, directions, concentrations, phase_ratio

    def forward(
        self,
        stride8: torch.Tensor,
        stride16: torch.Tensor,
        representation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        moments = torch.cat(
            (
                self._radial_moments(stride8, self.stride8_projection),
                self._radial_moments(stride16, self.stride16_projection),
            ),
            dim=1,
        )
        angular = self.circular_encoder(moments)
        statistics, directions, concentrations, phase_ratio = (
            self._frame_statistics(angular)
        )
        fused = torch.cat(
            (
                angular.flatten(1),
                self.context_projection(representation),
                statistics,
            ),
            dim=1,
        )
        return {
            "logit_delta": self.fusion(fused).squeeze(1),
            "frame_sin_cos": directions,
            "frame_concentrations": concentrations,
            "phase_progress": phase_ratio,
        }


class RelativeAngularFrameProbe(nn.Module):
    """Frozen prior endpoints/absolute-angular head plus two new frame arms."""

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
        from experiments.raw_context_endpoint_control_probe import MatchedContextResidual

        self.context_mlp = MatchedContextResidual(nonlinear=True)
        self.prior_angular_no_aux = AngularMomentLogitRefiner()
        self.relative_frame_no_aux = RelativeAngularFrameLogitRefiner()
        self.relative_frame_aux = RelativeAngularFrameLogitRefiner()
        self.relative_frame_aux.load_state_dict(
            self.relative_frame_no_aux.state_dict(), strict=True
        )
        for module in (
            self.encoder,
            self.base_projection,
            self.linear_refresh,
            self.context_mlp,
            self.prior_angular_no_aux,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()
        _require(
            _module_states_equal(
                self.relative_frame_no_aux, self.relative_frame_aux
            ),
            "relative-frame arms do not share identical initial tensors",
        )

    def train(self, mode: bool = True) -> RelativeAngularFrameProbe:
        super().train(mode)
        for module in (
            self.encoder,
            self.base_projection,
            self.linear_refresh,
            self.context_mlp,
            self.prior_angular_no_aux,
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
            prior = self.prior_angular_no_aux(
                features["stride8"], features["stride16"], representation
            )
            prior_progress = torch.sigmoid(base_logit + prior["logit_delta"])
        stride8 = features["stride8"].detach()
        stride16 = features["stride16"].detach()
        representation = representation.detach()
        no_aux = self.relative_frame_no_aux(stride8, stride16, representation)
        aux = self.relative_frame_aux(stride8, stride16, representation)
        return {
            METHOD_BASE: base,
            METHOD_LINEAR_REFRESH: linear,
            METHOD_CONTEXT_MLP: context,
            METHOD_PRIOR_ANGULAR: prior_progress,
            METHOD_RELATIVE_FRAME_NO_AUX: torch.sigmoid(
                base_logit + no_aux["logit_delta"]
            ),
            METHOD_RELATIVE_FRAME_AUX: torch.sigmoid(
                base_logit + aux["logit_delta"]
            ),
            f"{METHOD_RELATIVE_FRAME_NO_AUX}_frame": no_aux["frame_sin_cos"],
            f"{METHOD_RELATIVE_FRAME_AUX}_frame": aux["frame_sin_cos"],
            f"{METHOD_RELATIVE_FRAME_NO_AUX}_concentrations": no_aux[
                "frame_concentrations"
            ],
            f"{METHOD_RELATIVE_FRAME_AUX}_concentrations": aux[
                "frame_concentrations"
            ],
            f"{METHOD_RELATIVE_FRAME_NO_AUX}_phase": no_aux["phase_progress"],
            f"{METHOD_RELATIVE_FRAME_AUX}_phase": aux["phase_progress"],
        }


def _module_states_equal(left: nn.Module, right: nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return set(left_state) == set(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def relative_frame_parameter_counts(
    model: RelativeAngularFrameProbe,
) -> dict[str, int]:
    values = {
        METHOD_LINEAR_REFRESH: sum(
            parameter.numel() for parameter in model.linear_refresh.parameters()
        ),
        METHOD_CONTEXT_MLP: sum(
            parameter.numel() for parameter in model.context_mlp.parameters()
        ),
        METHOD_PRIOR_ANGULAR: sum(
            parameter.numel()
            for parameter in model.prior_angular_no_aux.parameters()
        ),
        METHOD_RELATIVE_FRAME_NO_AUX: sum(
            parameter.numel()
            for parameter in model.relative_frame_no_aux.parameters()
        ),
        METHOD_RELATIVE_FRAME_AUX: sum(
            parameter.numel()
            for parameter in model.relative_frame_aux.parameters()
        ),
        "foundation_frozen": sum(
            parameter.numel()
            for module in (model.encoder, model.base_projection)
            for parameter in module.parameters()
        ),
        "prior_controls_frozen": sum(
            parameter.numel()
            for module in (
                model.linear_refresh,
                model.context_mlp,
                model.prior_angular_no_aux,
            )
            for parameter in module.parameters()
        ),
        "total_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    _require(
        values[METHOD_RELATIVE_FRAME_NO_AUX]
        == values[METHOD_RELATIVE_FRAME_AUX],
        "relative-frame arm parameter counts differ",
    )
    relative_difference = abs(
        values[METHOD_RELATIVE_FRAME_NO_AUX] - values[METHOD_CONTEXT_MLP]
    ) / max(
        values[METHOD_RELATIVE_FRAME_NO_AUX], values[METHOD_CONTEXT_MLP]
    )
    _require(
        relative_difference <= 0.01,
        "relative-frame and context capacities differ by more than 1%",
    )
    return values


def build_relative_frame_probe_from_foundation(
    foundation: nn.Module,
) -> RelativeAngularFrameProbe:
    split_probe = build_probe_from_foundation(foundation)
    model = RelativeAngularFrameProbe(
        split_probe.encoder, split_probe.base_projection
    )
    del split_probe
    return model


def load_prior_angular_head(
    checkpoint_path: Path,
    *,
    model: RelativeAngularFrameProbe,
    expected_seed: int,
    expected_epochs: int,
    foundation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"absolute-angular checkpoint missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "absolute-angular checkpoint malformed")
    _require(
        checkpoint.get("protocol") == ABSOLUTE_ANGULAR_PROTOCOL
        and checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch",
        "absolute-angular checkpoint metadata differs",
    )
    _require(
        int(checkpoint.get("seed", -1)) == int(expected_seed)
        and int(checkpoint.get("epochs", -1)) == int(expected_epochs),
        "absolute-angular seed or epoch count differs",
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
        "absolute-angular head uses a different foundation",
    )
    state = checkpoint.get("angular_no_aux_state")
    _require(isinstance(state, Mapping), "prior absolute-angular state missing")
    model.prior_angular_no_aux.load_state_dict(state, strict=True)
    return {
        "path": str(source),
        "protocol": str(checkpoint["protocol"]),
        "seed": int(checkpoint["seed"]),
        "epochs": int(checkpoint["epochs"]),
        "checkpoint_selection": str(checkpoint["checkpoint_selection"]),
    }


class RelativeAngularFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Direct-B0 images with exactly co-transformed three-ray labels."""

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
        _require(bool(self.samples), "relative-frame dataset is empty")
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
        _require(image is not None, f"{sample.sample_id}: source decode failed")
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
        labels = angular_frame_targets(points)
        return {
            "image": normalized_rgb_tensor(roi),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            **labels,
        }


class RelativeFrameConditionedDataset(
    Dataset[dict[str, torch.Tensor | str]]
):
    """Six controlled conditions with frame labels valid for Raw conditions."""

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
        _require(bool(self.samples), "conditioned frame dataset is empty")
        _require(self.condition in CONDITIONS, "unknown robustness condition")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        points = _normalized_geometry(sample, bounds)
        conditioned, _metadata = robustness_degradations.apply_degradation(
            roi,
            self.condition,
            sample_id=sample.sample_id,
            seed=self.degradation_seed,
        )
        resized = direct_resize_whole_roi(conditioned, size=self.image_size)
        labels = angular_frame_targets(points)
        return {
            "image": normalized_rgb_tensor(resized),
            "progress": torch.tensor(sample.normalized_target, dtype=torch.float32),
            **labels,
            "frame_available": torch.tensor(
                self.condition in CONDITIONS[:3], dtype=torch.bool
            ),
            "sample_id": sample.sample_id,
            "scene_stem": sample.scene_stem,
        }


def run_relative_frame_epoch(
    model: RelativeAngularFrameProbe,
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
    phase_error_sums = {method: 0.0 for method in FRAME_METHODS}
    direction_loss_sum = 0.0
    samples = 0
    optimizer_steps = 0
    skipped_nonfinite_gradient_steps = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=use_amp)
        targets = batch["progress"].to(device, non_blocking=use_amp)
        target_frame = batch["frame_sin_cos"].to(
            device, non_blocking=use_amp
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(images)
            no_aux_loss = F.l1_loss(
                outputs[METHOD_RELATIVE_FRAME_NO_AUX], targets
            )
            aux_progress_loss = F.l1_loss(
                outputs[METHOD_RELATIVE_FRAME_AUX], targets
            )
            direction_loss = F.smooth_l1_loss(
                outputs[f"{METHOD_RELATIVE_FRAME_AUX}_frame"],
                target_frame,
                beta=DIRECTION_SMOOTH_L1_BETA,
            )
            loss = (
                no_aux_loss
                + aux_progress_loss
                + direction_auxiliary_weight * direction_loss
            )
        _require(bool(torch.isfinite(loss)), "relative-frame loss is non-finite")
        _require(
            all(
                bool(torch.isfinite(outputs[key]).all())
                for key in (
                    METHOD_RELATIVE_FRAME_NO_AUX,
                    METHOD_RELATIVE_FRAME_AUX,
                    f"{METHOD_RELATIVE_FRAME_NO_AUX}_frame",
                    f"{METHOD_RELATIVE_FRAME_AUX}_frame",
                    f"{METHOD_RELATIVE_FRAME_NO_AUX}_phase",
                    f"{METHOD_RELATIVE_FRAME_AUX}_phase",
                )
            ),
            "relative-frame forward output is non-finite",
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = tuple(
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        _require(bool(gradients), "relative-frame backward produced no gradients")
        gradients_finite = all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        scaler.step(optimizer)
        scaler.update()
        _require(float(scaler.get_scale()) > 0.0, "AMP scale collapsed to zero")
        _require(
            all(
                bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "relative-frame optimizer produced a non-finite parameter",
        )
        if gradients_finite:
            optimizer_steps += 1
        else:
            skipped_nonfinite_gradient_steps += 1
        count = int(targets.numel())
        for method in METHODS:
            absolute_error_sums[method] += float(
                torch.abs(outputs[method].detach() - targets).sum().cpu()
            )
        for method in FRAME_METHODS:
            phase_error_sums[method] += float(
                torch.abs(
                    outputs[f"{method}_phase"].detach() - targets
                ).sum().cpu()
            )
        direction_loss_sum += float(direction_loss.detach().cpu()) * count
        samples += count
    _require(samples > 0, "relative-frame epoch produced no samples")
    _require(optimizer_steps > 0, "relative-frame epoch completed no optimizer step")
    return {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "skipped_nonfinite_gradient_steps": skipped_nonfinite_gradient_steps,
        "exact_l1": {
            method: absolute_error_sums[method] / samples for method in METHODS
        },
        "phase_progress_l1": {
            method: phase_error_sums[method] / samples for method in FRAME_METHODS
        },
        "frame_direction_smooth_l1": direction_loss_sum / samples,
        "direction_auxiliary_weight": float(direction_auxiliary_weight),
    }


@dataclass(frozen=True, slots=True)
class RelativeFramePredictionRecord:
    sample_id: str
    scene_stem: str
    condition: str
    target: float
    predictions: dict[str, float]
    frame_available: bool
    target_frame: tuple[tuple[float, float], ...] | None
    oracle_phase_progress: float | None
    frames: dict[str, tuple[tuple[float, float], ...]]
    concentrations: dict[str, tuple[float, ...]]
    phase_predictions: dict[str, float]

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
            "frame_available": self.frame_available,
            "target_frame_sin_cos": self.target_frame,
            "oracle_phase_progress": self.oracle_phase_progress,
            "predicted_frames_sin_cos": dict(self.frames),
            "frame_concentrations": dict(self.concentrations),
            "phase_progress_predictions": dict(self.phase_predictions),
        }


def evaluate_condition(
    model: RelativeAngularFrameProbe,
    samples: Sequence[DirectSample],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[RelativeFramePredictionRecord, ...]:
    dataset = RelativeFrameConditionedDataset(samples, condition=condition)
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
    records: list[RelativeFramePredictionRecord] = []
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
            cpu_frames = {
                method: outputs[f"{method}_frame"].float().cpu().tolist()
                for method in FRAME_METHODS
            }
            cpu_concentrations = {
                method: outputs[f"{method}_concentrations"].float().cpu().tolist()
                for method in FRAME_METHODS
            }
            cpu_phases = {
                method: outputs[f"{method}_phase"].float().cpu().tolist()
                for method in FRAME_METHODS
            }
            targets = batch["progress"].float().tolist()
            target_frames = batch["frame_sin_cos"].float().tolist()
            oracle_phases = batch["oracle_phase_progress"].float().tolist()
            availability = batch["frame_available"].bool().tolist()
            sample_ids = batch["sample_id"]
            scene_stems = batch["scene_stem"]
            for row_index, (sample_id, scene_stem, target) in enumerate(
                zip(sample_ids, scene_stems, targets, strict=True)
            ):
                predictions = {
                    method: float(cpu_predictions[method][row_index])
                    for method in METHODS
                }
                frames = {
                    method: tuple(
                        tuple(float(value) for value in role_direction)
                        for role_direction in cpu_frames[method][row_index]
                    )
                    for method in FRAME_METHODS
                }
                concentrations = {
                    method: tuple(
                        float(value)
                        for value in cpu_concentrations[method][row_index]
                    )
                    for method in FRAME_METHODS
                }
                phase_predictions = {
                    method: float(cpu_phases[method][row_index])
                    for method in FRAME_METHODS
                }
                _require(
                    all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in predictions.values()
                    )
                    and all(
                        math.isfinite(value)
                        for frame in frames.values()
                        for direction in frame
                        for value in direction
                    )
                    and all(
                        math.isfinite(value) and 0.0 <= value <= 2.0
                        for value in phase_predictions.values()
                    ),
                    "relative-frame evaluation produced an invalid prediction",
                )
                available = bool(availability[row_index])
                target_frame = (
                    tuple(
                        tuple(float(value) for value in role_direction)
                        for role_direction in target_frames[row_index]
                    )
                    if available
                    else None
                )
                records.append(
                    RelativeFramePredictionRecord(
                        sample_id=str(sample_id),
                        scene_stem=str(scene_stem),
                        condition=condition,
                        target=float(target),
                        predictions=predictions,
                        frame_available=available,
                        target_frame=target_frame,
                        oracle_phase_progress=(
                            float(oracle_phases[row_index]) if available else None
                        ),
                        frames=frames,
                        concentrations=concentrations,
                        phase_predictions=phase_predictions,
                    )
                )
    _require(len(records) == len(samples), "condition evaluation row count differs")
    return tuple(records)


def _angular_error_degrees(
    prediction: tuple[float, float], target: tuple[float, float]
) -> float:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    predicted_norm = float(np.linalg.norm(predicted))
    expected_norm = float(np.linalg.norm(expected))
    _require(expected_norm > 1.0e-8, "target direction collapsed")
    if predicted_norm <= 1.0e-8:
        return 180.0
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
        rows = np.concatenate(
            tuple(scene_indices[int(index)] for index in selected)
        )
        replicates[replicate] = float(np.mean(delta[rows]))
    return replicates


def summarize_records(
    records: Sequence[RelativeFramePredictionRecord],
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

    frame_rows = tuple(row for row in rows if row.frame_available)
    geometry: dict[str, Any] | None = None
    if frame_rows:
        role_errors: dict[str, dict[str, np.ndarray]] = {
            method: {
                role: np.asarray(
                    [
                        _angular_error_degrees(
                            row.frames[method][role_index],
                            row.target_frame[role_index],
                        )
                        for row in frame_rows
                        if row.target_frame is not None
                    ],
                    dtype=np.float64,
                )
                for role_index, role in enumerate(FRAME_ROLES)
            }
            for method in FRAME_METHODS
        }
        overall_direction_errors = {
            method: np.mean(
                np.stack(tuple(role_errors[method].values()), axis=1), axis=1
            )
            for method in FRAME_METHODS
        }
        frame_by_scene: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(frame_rows):
            frame_by_scene[row.scene_stem].append(index)
        frame_scene_names = tuple(sorted(frame_by_scene))
        frame_scene_indices = tuple(
            np.asarray(frame_by_scene[scene], dtype=np.int64)
            for scene in frame_scene_names
        )
        frame_rng = np.random.default_rng(int(bootstrap_seed) + 17)
        frame_sampled_scenes = frame_rng.integers(
            0,
            len(frame_scene_names),
            size=(bootstrap_replicates, len(frame_scene_names)),
        )
        direction_delta = (
            overall_direction_errors[METHOD_RELATIVE_FRAME_AUX]
            - overall_direction_errors[METHOD_RELATIVE_FRAME_NO_AUX]
        )
        direction_replicates = _bootstrap_paired_delta(
            direction_delta,
            scene_indices=frame_scene_indices,
            sampled_scenes=frame_sampled_scenes,
        )
        phase_errors: dict[str, np.ndarray] = {
            "oracle_geometry": np.abs(
                np.asarray(
                    [row.oracle_phase_progress for row in frame_rows],
                    dtype=np.float64,
                )
                - np.asarray([row.target for row in frame_rows], dtype=np.float64)
            ),
            **{
                method: np.abs(
                    np.asarray(
                        [row.phase_predictions[method] for row in frame_rows],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        [row.target for row in frame_rows], dtype=np.float64
                    )
                )
                for method in FRAME_METHODS
            },
        }
        phase_delta = (
            phase_errors[METHOD_RELATIVE_FRAME_AUX]
            - phase_errors[METHOD_RELATIVE_FRAME_NO_AUX]
        )
        phase_replicates = _bootstrap_paired_delta(
            phase_delta,
            scene_indices=frame_scene_indices,
            sampled_scenes=frame_sampled_scenes,
        )
        geometry = {
            "rows": len(frame_rows),
            "scenes": len(frame_scene_names),
            "mean_absolute_angular_error_degrees": {
                method: {
                    role: float(np.mean(role_errors[method][role]))
                    for role in FRAME_ROLES
                }
                for method in FRAME_METHODS
            },
            "median_absolute_angular_error_degrees": {
                method: {
                    role: float(np.median(role_errors[method][role]))
                    for role in FRAME_ROLES
                }
                for method in FRAME_METHODS
            },
            "mean_concentration": {
                method: {
                    role: float(
                        np.mean(
                            [
                                row.concentrations[method][role_index]
                                for row in frame_rows
                            ]
                        )
                    )
                    for role_index, role in enumerate(FRAME_ROLES)
                }
                for method in FRAME_METHODS
            },
            "phase_progress_nmae": {
                method: float(np.mean(values))
                for method, values in phase_errors.items()
            },
            "direction_aux_minus_no_aux_mean_angular_error_degrees": float(
                np.mean(direction_delta)
            ),
            "direction_aux_minus_no_aux_scene_bootstrap_95_ci": [
                float(np.quantile(direction_replicates, 0.025)),
                float(np.quantile(direction_replicates, 0.975)),
            ],
            "phase_aux_minus_no_aux_mean_nmae_delta": float(np.mean(phase_delta)),
            "phase_aux_minus_no_aux_scene_bootstrap_95_ci": [
                float(np.quantile(phase_replicates, 0.025)),
                float(np.quantile(phase_replicates, 0.975)),
            ],
        }
    return {
        "rows": len(rows),
        "scenes": len(scene_names),
        "conditions": sorted({row.condition for row in rows}),
        "nmae": {method: float(np.mean(errors[method])) for method in METHODS},
        "comparisons": comparisons,
        "geometry": geometry,
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
    absolute_angular_checkpoint_path: Path,
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
        "checkpoint": root / "relative_angular_frames.pt",
        "predictions": root / "inner_dev_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "relative-frame output artifact already exists",
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
    model = build_relative_frame_probe_from_foundation(foundation)
    del foundation
    endpoint_control_metadata = load_endpoint_controls(
        endpoint_control_checkpoint_path,
        model=model,
        expected_seed=seed,
        expected_epochs=epochs,
        foundation_metadata=foundation_metadata,
    )
    prior_angular_metadata = load_prior_angular_head(
        absolute_angular_checkpoint_path,
        model=model,
        expected_seed=seed,
        expected_epochs=epochs,
        foundation_metadata=foundation_metadata,
    )
    model = model.to(device)
    counts = relative_frame_parameter_counts(model)
    train_dataset = RelativeAngularFrameDataset(
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
        metrics = run_relative_frame_epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            direction_auxiliary_weight=direction_auxiliary_weight,
        )
        row = {
            "phase": "relative_angular_frames",
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
            "progress_loss": "exact_l1_each_relative_frame_arm",
            "direction_loss": "smooth_l1_pointer_start_end_posterior_means",
            "direction_smooth_l1_beta": DIRECTION_SMOOTH_L1_BETA,
            "direction_auxiliary_weight": float(direction_auxiliary_weight),
            "foundation": foundation_metadata,
            "endpoint_controls": endpoint_control_metadata,
            "prior_absolute_angular": prior_angular_metadata,
            "all_prior_modules_frozen": True,
            "same_batches_for_both_relative_frame_arms": True,
            "relative_frame_arms_identical_at_initialization": True,
            "zero_initialized_progress_residual": True,
            "phase_computation_precision": "float32_inside_amp",
            "parameter_counts": counts,
            "training_elapsed_seconds": training_elapsed,
            "history": history,
            "relative_frame_no_aux_state": _checkpoint_state(
                model.relative_frame_no_aux
            ),
            "relative_frame_aux_state": _checkpoint_state(
                model.relative_frame_aux
            ),
        },
        paths["checkpoint"],
    )

    evaluation_started = time.perf_counter()
    all_records: list[RelativeFramePredictionRecord] = []
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
            "Does an explicit pointer/start/end clockwise angular frame improve "
            "Raw progress over endpoint refresh and the prior absolute-angular head, "
            "and does matched three-ray supervision add value?"
        ),
        "interpretation": {
            "relative_frame_no_aux_minus_prior_angular_no_aux": (
                "effect of three semantic posteriors and relative-phase features "
                "without geometry supervision"
            ),
            "relative_frame_aux_minus_relative_frame_no_aux": (
                "effect of matched pointer/start/end direction supervision"
            ),
            "oracle_phase_progress": (
                "label-only diagnostic; never supplied as a model input"
            ),
            "intended_integration_scope": (
                "replace only relation-unavailable Raw fallback; relation-active "
                "projective transport remains unchanged"
            ),
            "sequential_internal_probe_after_dev_inspection": True,
            "formal_holdout_selection_not_performed": True,
            "one_seed_internal_probe_not_paper_table": True,
            "predecessor_failed_run_preserved": (
                "artifacts/runs/raw_relative_angular_frame_probe/"
                "seed_20262020/relative_angular_frames.pt"
            ),
        },
        "seed": int(seed),
        "device": str(device),
        "foundation": foundation_metadata,
        "endpoint_controls": endpoint_control_metadata,
        "prior_absolute_angular": prior_angular_metadata,
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "formal_holdout_content_access": False,
            "inner_train_samples": len(internal.train),
            "inner_train_scenes": len(internal.train_scenes),
            "inner_dev_samples": len(internal.dev),
            "inner_dev_scenes": len(internal.dev_scenes),
            "conditions": list(CONDITIONS),
            "frame_metrics_conditions": list(CONDITIONS[:3]),
        },
        "architecture": {
            "backbone": "single_frozen_efficientnet_b0",
            "polar_radial_bins": POLAR_RADIAL_BINS,
            "polar_angular_bins": POLAR_ANGULAR_BINS,
            "frame_roles": list(FRAME_ROLES),
            "angle_convention": "zero_up_clockwise_positive",
            "phase_computation_precision": "float32_inside_amp",
            "relative_features": [
                "pair_dot_cross",
                "pointer_phase_turns",
                "scale_span_turns",
                "clockwise_phase_ratio",
                "inside_span_margin",
            ],
            "feature_sources": ["stride8", "stride16", "final_representation"],
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "workers": int(workers),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "progress_loss": "exact_l1_each_relative_frame_arm",
            "direction_loss": "smooth_l1_pointer_start_end_posterior_means",
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
        "--foundation-checkpoint", type=Path, default=DEFAULT_FOUNDATION_CHECKPOINT
    )
    parser.add_argument(
        "--endpoint-control-checkpoint",
        type=Path,
        default=DEFAULT_ENDPOINT_CONTROL_CHECKPOINT,
    )
    parser.add_argument(
        "--absolute-angular-checkpoint",
        type=Path,
        default=DEFAULT_ABSOLUTE_ANGULAR_CHECKPOINT,
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
            absolute_angular_checkpoint_path=args.absolute_angular_checkpoint,
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
    except (RelativeAngularFrameProbeError, DirectProgressError) as exc:
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
    "FRAME_METHODS",
    "FRAME_ROLES",
    "METHOD_BASE",
    "METHOD_PRIOR_ANGULAR",
    "METHOD_RELATIVE_FRAME_AUX",
    "METHOD_RELATIVE_FRAME_NO_AUX",
    "PROTOCOL",
    "RelativeAngularFrameDataset",
    "RelativeAngularFrameLogitRefiner",
    "RelativeAngularFrameProbe",
    "RelativeAngularFrameProbeError",
    "RelativeFramePredictionRecord",
    "angular_frame_targets",
    "build_relative_frame_probe_from_foundation",
    "relative_frame_parameter_counts",
    "run_probe",
    "summarize_records",
]
