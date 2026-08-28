"""Single-seed, SyncG-fit-only pilot for the SGCA multi-view EfficientNet.

The formal SyncG holdout is never opened by this runner.  ``load_training_samples``
returns only the formal fit samples (and holdout IDs/count for boundary auditing),
then this module makes a deterministic scene-disjoint train/dev partition inside
that fit roster.  The development partition is diagnostic only: checkpoints are
always the terminal epoch, never the best development epoch.

The original four pilot arms retain their small, interpretable staircase:

* A1: original projective ROI only;
* A2: original + SARN-v2, strictly uniform convex fusion;
* A3: A2 + heteroscedastic uncertainty-aware fusion and routing objectives;
* A4: A3 + support--geometry conditioned attention and optional rectification.

Two internal-screen extensions isolate geometry supervision from reliability:
``A3G`` is A3 plus a low-weight geometry auxiliary, while ``A5`` adds only a
geometry-consistency correction to the SARN fusion logit.  Neither extension
enables middle-stage geometry modulation.  Their geometry predictor is a
stop-gradient side head: auxiliary labels train that head but cannot steer the
shared encoder.

All enabled parameters, including the shared EfficientNet-B0 encoder, are trained
by the reading objective.  This file contains no external-photo input or
condition-specific parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.geoattn_resnet18_progress import (
    _apply_homography,
    _geometry_targets,
)
from experiments.perspective_adaptive_db_gar18 import (
    COMBINED_BLUR_SIGMA_FRACTION,
    PERSPECTIVE_SEED_OFFSET,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.projective_geometry_views import (
    build_projective_geometry_views,
    resize_projective_geometry_views,
)
from experiments.resnet18_direct_progress import (
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    DirectSample,
    matched_cagh_augmentation,
    load_training_samples,
)
from experiments.sarn_guided_attention_db_gar18 import (
    _transform_points_after_sarn_crop,
)
from experiments.sgca_syncg_internal_pilot import (
    FIT_SAMPLES,
    FIT_SCENES,
    INTERNAL_DEV_SCENE_STEMS,
    PROTOCOL as INTERNAL_SPLIT_PROTOCOL,
)
from experiments.support_geometry_multiview_efficientnet import (
    A5_ARCHITECTURE,
    ARCHITECTURE,
    VIEW_NAMES,
    ArchitectureAblation,
    EfficientNetB0GeometryConsistencyReliability,
    EfficientNetB0SupportGeometryMultiView,
)
from experiments.syncg_lightweight_regression_baselines import (
    PROTOCOL as LIGHTWEIGHT_BASELINE_PROTOCOL,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    PhotoAugmentation,
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments import robustness_degradations
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


PROTOCOL: Final[str] = "syncg_fit_internal_scene_pilot_sgca_multiview_v1"
DEFAULT_SYNCG_SCENE_SPLIT: Final[Path] = Path(
    "C:/pointer_read/syncg_scene_disjoint_clean_v1/split.json"
)
DEFAULT_SEED: Final[int] = 20262020
DEFAULT_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 3.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
TAIL_FRACTION: Final[float] = 0.10
ORACLE_TEMPERATURE: Final[float] = 0.05
TRAIN_CONDITION_MIX: Final[tuple[tuple[str, float], ...]] = (
    ("clean", 0.30),
    ("perspective_moderate", 0.25),
    ("perspective_severe", 0.25),
    ("combined_severe", 0.20),
)
DEV_CONDITIONS: Final[tuple[str, ...]] = tuple(
    condition for condition, _probability in TRAIN_CONDITION_MIX
)
CONDITION_SEED_OFFSET: Final[int] = 7_310_113
A9_MAX_ORDERED_TICKS: Final[int] = 36


class PilotTrainingError(ValueError):
    """The pilot configuration, sample, objective, or output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotTrainingError(message)


@dataclass(frozen=True, slots=True)
class InternalSceneSplit:
    train: tuple[DirectSample, ...]
    dev: tuple[DirectSample, ...]
    train_scenes: tuple[str, ...]
    dev_scenes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PilotArm:
    name: str
    description: str
    ablation: ArchitectureAblation
    supervise_all_available_views: bool
    use_geometry_auxiliary: bool
    geometry_auxiliary_scale: float
    use_oracle_routing: bool
    use_tail_no_harm: bool
    use_geometry_consistency_routing: bool
    detach_geometry_from_encoder: bool


@dataclass(frozen=True, slots=True)
class LossWeights:
    fused_mean: float = 1.0
    view_mean: float = 0.50
    fused_nll: float = 0.05
    view_nll: float = 0.025
    geometry_pivot: float = 0.25
    geometry_direction: float = 0.25
    geometry_references: float = 0.125
    oracle_routing: float = 0.10
    tail_no_harm: float = 0.20

    def __post_init__(self) -> None:
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in asdict(self).values()),
            "loss weights must be finite and non-negative",
        )


DEFAULT_LOSS_WEIGHTS: Final[LossWeights] = LossWeights()


def initialize_encoder_from_lightweight_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Warm-start the shared encoder from a trained EfficientNet-B0 baseline."""

    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location="cpu", weights_only=False
    )
    _require(isinstance(checkpoint, Mapping), "initial checkpoint is not an object")
    _require(
        checkpoint.get("protocol") == LIGHTWEIGHT_BASELINE_PROTOCOL,
        "initial checkpoint is not a lightweight baseline",
    )
    _require(
        checkpoint.get("architecture") == "efficientnet_b0",
        "initial checkpoint is not EfficientNet-B0",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "initial checkpoint state is missing")
    base_model = (
        model.base_model
        if isinstance(model, EfficientNetB0GeometryConsistencyReliability)
        else model
    )
    _require(
        isinstance(base_model, EfficientNetB0SupportGeometryMultiView),
        "initialization target is not the SGCA base model",
    )
    prefix = "backbone.features."
    source_features = {
        str(name)[len(prefix) :]: value
        for name, value in state.items()
        if str(name).startswith(prefix)
    }
    _require(bool(source_features), "initial checkpoint has no EfficientNet features")
    target_features = torch.nn.Sequential(
        *tuple(base_model.encoder.early.children()),
        *tuple(base_model.encoder.late.children()),
    )
    target_features.load_state_dict(source_features, strict=True)
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "protocol": str(checkpoint["protocol"]),
        "architecture": str(checkpoint["architecture"]),
        "seed": int(checkpoint["seed"]),
        "scope": "EfficientNet features only; SGCA, representation, heads, and fusion remain newly initialized",
    }


def resolve_pilot_arm(
    name: str,
    *,
    use_rectified_view: bool = False,
) -> PilotArm:
    """Resolve the causal base ladder and isolated A3G/A5 pilot extensions."""

    normalized = str(name).strip().upper()
    if normalized == "A1":
        return PilotArm(
            name="A1",
            description="original projective ROI only",
            ablation=ArchitectureAblation(
                use_support_mask=False,
                use_geometry_attention=False,
                use_multiview_fusion=False,
                use_learned_fusion=False,
                use_uncertainty_in_fusion=False,
                use_rectified_view=False,
            ),
            supervise_all_available_views=False,
            use_geometry_auxiliary=False,
            geometry_auxiliary_scale=0.0,
            use_oracle_routing=False,
            use_tail_no_harm=False,
            use_geometry_consistency_routing=False,
            detach_geometry_from_encoder=False,
        )
    if normalized == "A2":
        return PilotArm(
            name="A2",
            description="original plus SARN with strictly uniform convex fusion",
            ablation=ArchitectureAblation(
                use_support_mask=False,
                use_geometry_attention=False,
                use_multiview_fusion=True,
                use_learned_fusion=False,
                use_uncertainty_in_fusion=False,
                use_rectified_view=False,
            ),
            supervise_all_available_views=True,
            use_geometry_auxiliary=False,
            geometry_auxiliary_scale=0.0,
            use_oracle_routing=False,
            use_tail_no_harm=False,
            use_geometry_consistency_routing=False,
            detach_geometry_from_encoder=False,
        )
    if normalized == "A3":
        return PilotArm(
            name="A3",
            description="probabilistic reliability fusion of original and SARN",
            ablation=ArchitectureAblation(
                use_support_mask=False,
                use_geometry_attention=False,
                use_multiview_fusion=True,
                use_uncertainty_in_fusion=True,
                use_rectified_view=False,
            ),
            supervise_all_available_views=True,
            use_geometry_auxiliary=False,
            geometry_auxiliary_scale=0.0,
            use_oracle_routing=True,
            use_tail_no_harm=True,
            use_geometry_consistency_routing=False,
            detach_geometry_from_encoder=False,
        )
    if normalized in ("A3G", "A5"):
        use_consistency = normalized == "A5"
        return PilotArm(
            name=normalized,
            description=(
                "A3 uncertainty fusion plus low-weight geometry auxiliary"
                if not use_consistency
                else (
                    "A3G plus geometry-consistency reliability correction "
                    "without middle-stage geometry modulation"
                )
            ),
            ablation=ArchitectureAblation(
                use_support_mask=False,
                use_geometry_attention=False,
                use_multiview_fusion=True,
                use_learned_fusion=True,
                use_uncertainty_in_fusion=True,
                use_rectified_view=False,
            ),
            supervise_all_available_views=True,
            use_geometry_auxiliary=True,
            geometry_auxiliary_scale=0.20,
            use_oracle_routing=True,
            use_tail_no_harm=True,
            use_geometry_consistency_routing=use_consistency,
            detach_geometry_from_encoder=True,
        )
    if normalized == "A4":
        return PilotArm(
            name="A4",
            description=(
                "full support--geometry conditioned attention with probabilistic "
                "multi-view fusion"
            ),
            ablation=ArchitectureAblation(
                use_support_mask=True,
                use_geometry_attention=True,
                use_multiview_fusion=True,
                use_uncertainty_in_fusion=True,
                use_rectified_view=bool(use_rectified_view),
            ),
            supervise_all_available_views=True,
            use_geometry_auxiliary=True,
            geometry_auxiliary_scale=1.0,
            use_oracle_routing=True,
            use_tail_no_harm=True,
            use_geometry_consistency_routing=False,
            detach_geometry_from_encoder=False,
        )
    raise PilotTrainingError(f"unknown pilot arm: {name}")


def make_internal_scene_split(
    samples: Sequence[DirectSample],
    *,
    dev_scene_stems: Sequence[str] = INTERNAL_DEV_SCENE_STEMS,
) -> InternalSceneSplit:
    """Partition the formal fit roster with the fixed 14-scene pilot roster."""

    values = tuple(samples)
    _require(bool(values), "formal fit roster is empty")
    scenes = tuple(sorted({sample.scene_stem for sample in values}))
    requested_dev = tuple(str(scene) for scene in dev_scene_stems)
    _require(
        bool(requested_dev)
        and len(requested_dev) == len(set(requested_dev))
        and set(requested_dev) < set(scenes),
        "fixed internal dev scenes must be a proper subset of formal fit scenes",
    )
    dev_scenes = requested_dev
    dev_set = set(dev_scenes)
    train_scenes = tuple(scene for scene in scenes if scene not in dev_set)
    train = tuple(sample for sample in values if sample.scene_stem not in dev_set)
    dev = tuple(sample for sample in values if sample.scene_stem in dev_set)
    _require(bool(train) and bool(dev), "internal scene split produced an empty partition")
    _require(
        not ({sample.scene_stem for sample in train} & {sample.scene_stem for sample in dev}),
        "internal train and dev scenes overlap",
    )
    _require(
        {sample.sample_id for sample in train}.isdisjoint(
            {sample.sample_id for sample in dev}
        )
        and len(train) + len(dev) == len(values),
        "internal split does not partition the formal fit roster",
    )
    return InternalSceneSplit(
        train=train,
        dev=dev,
        train_scenes=train_scenes,
        dev_scenes=dev_scenes,
    )


def _points_after_pixel_homography(
    normalized_points: np.ndarray,
    homography: np.ndarray,
    *,
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Transform normalized labels with a pixel-centre homography."""

    points = np.asarray(normalized_points, dtype=np.float32)
    _require(points.ndim == 2 and points.shape[1] == 2, "points must be Nx2")
    input_height, input_width = (int(value) for value in input_hw)
    output_height, output_width = (int(value) for value in output_hw)
    input_scale = np.asarray(
        [max(input_width - 1, 1), max(input_height - 1, 1)], dtype=np.float32
    )
    output_scale = np.asarray(
        [max(output_width - 1, 1), max(output_height - 1, 1)], dtype=np.float32
    )
    pixel_points = points * input_scale
    transformed_pixels = _apply_homography(pixel_points, homography)
    transformed = transformed_pixels / output_scale
    _require(bool(np.isfinite(transformed).all()), "rectified geometry is non-finite")
    _require(
        bool(((transformed >= -1.0e-4) & (transformed <= 1.0 + 1.0e-4)).all()),
        "rectified geometry escaped its output canvas",
    )
    return np.clip(transformed, 0.0, 1.0).astype(np.float32)


def _normalized_raw_to_sarn_homography(
    decision: Any,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Return the exact normalized-coordinate affine used by SARN cropping.

    ``_transform_points_after_sarn_crop`` first maps normalized coordinates to
    source pixel centres, applies the half-pixel crop/resize equation, then
    normalizes by ``width-1``/``height-1``.  This matrix is that same equation
    collected into one normalized 3x3 homogeneous transform.
    """

    _require(height >= 2 and width >= 2, "SARN transform canvas is too small")
    identity = np.eye(3, dtype=np.float32)
    if not bool(decision.applied):
        return identity
    bbox = decision.bbox_xyxy
    _require(bbox is not None and len(bbox) == 4, "applied SARN decision lacks bbox")
    left, top, right, bottom = (int(value) for value in bbox)
    _require(
        0 <= left < right <= width and 0 <= top < bottom <= height,
        "SARN crop bbox is outside the raw view",
    )
    scale_x = float(width) / float(right - left)
    scale_y = float(height) / float(bottom - top)
    translate_x = (
        (-float(left) + 0.5) * scale_x - 0.5
    ) / float(width - 1)
    translate_y = (
        (-float(top) + 0.5) * scale_y - 0.5
    ) / float(height - 1)
    return np.asarray(
        (
            (scale_x, 0.0, translate_x),
            (0.0, scale_y, translate_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _geometry_tensor_triplet(points: np.ndarray) -> tuple[torch.Tensor, ...]:
    labels = _geometry_targets(points)
    return (
        labels["pivot"],
        labels["direction_sin_cos"],
        labels["references"],
    )


def _geometry_keypoints(points: np.ndarray) -> torch.Tensor:
    """Return pivot, pointer tip, start reference, and end reference."""

    values = np.asarray(points, dtype=np.float32)
    _require(
        values.ndim == 2 and values.shape[1] == 2 and len(values) >= 9,
        "geometry keypoints require >=7 marks plus pivot/tip",
    )
    return torch.from_numpy(
        np.ascontiguousarray(
            np.stack((values[-2], values[-1], values[0], values[-3]), axis=0),
            dtype=np.float32,
        )
    )


def _a9_ordered_scale_geometry(
    points: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose transformed Raw scale order without reconstructing it from four points."""

    values = np.asarray(points, dtype=np.float32)
    _require(
        values.ndim == 2
        and values.shape[1] == 2
        and 9 <= len(values) <= A9_MAX_ORDERED_TICKS + 2,
        "A9 ordered geometry requires 7--36 marks plus pivot/tip",
    )
    marks = values[:-2]
    padded = np.zeros((A9_MAX_ORDERED_TICKS, 2), dtype=np.float32)
    valid = np.zeros((A9_MAX_ORDERED_TICKS,), dtype=np.bool_)
    padded[: len(marks)] = marks
    valid[: len(marks)] = True
    _require(
        bool(np.isfinite(padded).all())
        and bool(((marks >= 0.0) & (marks <= 1.0)).all()),
        "A9 ordered Raw geometry is non-finite or outside the fitted plane",
    )
    return (
        torch.from_numpy(np.ascontiguousarray(padded)),
        torch.from_numpy(np.ascontiguousarray(valid)),
        torch.from_numpy(np.ascontiguousarray(values[-2])),
        torch.from_numpy(np.ascontiguousarray(values[-1])),
    )


def _training_condition_from_unit(value: float) -> str:
    _require(math.isfinite(value) and 0.0 <= value < 1.0, "invalid mixture draw")
    cumulative = 0.0
    for condition, probability in TRAIN_CONDITION_MIX:
        cumulative += probability
        if value < cumulative:
            return condition
    return TRAIN_CONDITION_MIX[-1][0]


def _apply_condition(
    image: np.ndarray,
    points: np.ndarray,
    *,
    condition: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one arm-independent member of the fixed training/dev condition set."""

    _require(condition in DEV_CONDITIONS, f"unknown training condition: {condition}")
    source_image = np.ascontiguousarray(image)
    source_points = np.asarray(points, dtype=np.float32)
    if condition == "clean":
        return source_image, source_points.copy(), {
            "condition": condition,
            "degrees": 0.0,
            "blur_applied": False,
        }

    degree_bounds = (
        (20.0, 30.0)
        if condition == "perspective_moderate"
        else (35.0, 45.0)
    )
    degrees = float(rng.uniform(*degree_bounds))
    axis = "yaw" if int(rng.integers(0, 2)) == 0 else "pitch"
    sign = -1 if int(rng.integers(0, 2)) == 0 else 1
    height, width = source_image.shape[:2]
    destination = robustness_degradations._projected_corners(
        width, height, degrees, axis, sign
    )
    source_corners = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_corners, destination)
    conditioned = cv2.warpPerspective(
        source_image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=robustness_degradations._border_median(source_image),
    )
    blur_applied = condition == "combined_severe"
    if blur_applied:
        sigma = max(
            0.1, COMBINED_BLUR_SIGMA_FRACTION * float(min(height, width))
        )
        conditioned = cv2.GaussianBlur(
            conditioned,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
    pixel_scale = np.asarray(
        [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
    )
    transformed = _apply_homography(source_points * pixel_scale, homography)
    transformed = transformed / pixel_scale
    _require(bool(np.isfinite(transformed).all()), "conditioned labels are non-finite")
    _require(
        bool(((transformed >= -1.0e-5) & (transformed <= 1.0 + 1.0e-5)).all()),
        "conditioned labels escaped the fitted plane",
    )
    return (
        np.ascontiguousarray(conditioned),
        np.clip(transformed, 0.0, 1.0).astype(np.float32),
        {
            "condition": condition,
            "degrees": degrees,
            "axis": axis,
            "sign": sign,
            "blur_applied": blur_applied,
        },
    )


class SyncGSupportGeometryMultiViewDataset(Dataset[dict[str, Any]]):
    """Generate aligned original, SARN, and rectified views from SyncG fit."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        training: bool,
        seed: int,
        total_epochs: int,
        image_size: int = IMAGE_SIZE,
        augmentation: PhotoAugmentation | None = None,
        condition: str | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.training = bool(training)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self.image_size = int(image_size)
        self.condition = None if condition is None else str(condition)
        self.augmentation = (
            matched_cagh_augmentation()
            if augmentation is None and self.training
            else augmentation or PhotoAugmentation.disabled()
        )
        self.augmentation.validate()
        self.epoch = 0 if self.training else self.total_epochs - 1
        _require(bool(self.samples), "multi-view dataset is empty")
        _require(self.total_epochs >= 1, "total epochs must be positive")
        _require(self.image_size >= 32, "image size is too small")
        _require(
            (self.training and (
                self.condition is None or self.condition in DEV_CONDITIONS
            ))
            or (not self.training and self.condition in DEV_CONDITIONS),
            (
                "training samples use the fixed mixture or one explicit fixed "
                "condition; dev samples require one fixed condition"
            ),
        )

    def set_epoch(self, epoch: int) -> None:
        _require(0 <= int(epoch) < self.total_epochs, "dataset epoch is out of range")
        self.epoch = int(epoch) if self.training else self.total_epochs - 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=self.image_size)
        _require(
            bool(sample.protected_points_xy),
            f"{sample.sample_id}: geometry landmarks are missing",
        )
        left, top, right, bottom = bounds
        scale = np.asarray([right - left, bottom - top], dtype=np.float32)
        _require(bool((scale > 0.0).all()), f"{sample.sample_id}: empty tight ROI")
        points = (
            np.asarray(sample.protected_points_xy, dtype=np.float32)
            - np.asarray([left, top], dtype=np.float32)
        ) / scale

        if self.training:
            augmentation_rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + int(index) * 97
            )
            roi, forward, _geometry_code = _cagh_augment_geometry(
                roi, points, augmentation_rng, self.augmentation
            )
            points = _apply_homography(points, forward)
            roi, _photo_code = _cagh_augment_photo(
                roi, augmentation_rng, self.augmentation
            )

        condition_rng = np.random.default_rng(
            self.seed
            + CONDITION_SEED_OFFSET
            + self.epoch * 3_000_049
            + int(index) * 389
        )
        condition = (
            str(self.condition)
            if self.condition is not None
            else _training_condition_from_unit(float(condition_rng.random()))
        )
        projective_rng = np.random.default_rng(
            self.seed
            + PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + int(index) * 193
            + DEV_CONDITIONS.index(condition) * 65_537
        )
        original_view, original_points, _metadata = _apply_condition(
            roi,
            points,
            condition=condition,
            rng=projective_rng,
        )

        views = build_projective_geometry_views(original_view)
        views = resize_projective_geometry_views(
            views, output_hw=(self.image_size, self.image_size)
        )
        decision = views.sarn_decision
        sarn_points = _transform_points_after_sarn_crop(
            original_points,
            decision,
            height=self.image_size,
            width=self.image_size,
        )
        _require(sarn_points is not None, "SARN geometry labels disappeared")

        raw_sarn_mask = decision.valid_support_mask
        sarn_active = bool(decision.applied)
        if raw_sarn_mask is None:
            sarn_active = False
            sarn_mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        else:
            sarn_mask = np.asarray(raw_sarn_mask, dtype=np.float32)
            if sarn_mask.shape != (self.image_size, self.image_size):
                sarn_mask = cv2.resize(
                    sarn_mask,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )
            sarn_mask = np.clip(sarn_mask, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(sarn_mask).all() or float(sarn_mask.sum()) <= 0.0:
                sarn_active = False
                sarn_mask = np.ones_like(sarn_mask)
        raw_to_sarn_homography = (
            _normalized_raw_to_sarn_homography(
                decision,
                height=self.image_size,
                width=self.image_size,
            )
            if sarn_active
            else np.eye(3, dtype=np.float32)
        )

        rectified_active = bool(views.active and sarn_active)
        rectified_points = np.asarray(sarn_points, dtype=np.float32)
        rectified_geometry_available = rectified_active
        rectified_mask = np.ones_like(sarn_mask)
        if rectified_active:
            try:
                rectified_points = _points_after_pixel_homography(
                    sarn_points,
                    views.homography_a_to_b,
                    input_hw=(self.image_size, self.image_size),
                    output_hw=(self.image_size, self.image_size),
                )
                rectified_mask = cv2.warpPerspective(
                    sarn_mask,
                    np.asarray(views.homography_a_to_b, dtype=np.float64),
                    (self.image_size, self.image_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                ).astype(np.float32)
                rectified_mask = np.clip(rectified_mask, 0.0, 1.0)
                if (
                    not np.isfinite(rectified_mask).all()
                    or float(rectified_mask.sum()) <= 0.0
                ):
                    rectified_active = False
                    rectified_geometry_available = False
                    rectified_mask = np.ones_like(sarn_mask)
            except PilotTrainingError:
                rectified_geometry_available = False
                rectified_points = np.asarray(sarn_points, dtype=np.float32)

        original_geometry = _geometry_tensor_triplet(original_points)
        sarn_geometry = _geometry_tensor_triplet(sarn_points)
        rectified_geometry = _geometry_tensor_triplet(rectified_points)
        a9_ticks, a9_tick_mask, a9_pivot, a9_pointer = (
            _a9_ordered_scale_geometry(original_points)
        )
        geometry_keypoints = torch.stack(
            (
                _geometry_keypoints(original_points),
                _geometry_keypoints(sarn_points),
                _geometry_keypoints(rectified_points),
            )
        )
        return {
            "sample_id": sample.sample_id,
            "scene_stem": sample.scene_stem,
            "condition_name": condition,
            "original_view": normalized_rgb_tensor(original_view),
            "sarn_view": normalized_rgb_tensor(views.view_a_bgr),
            "sarn_support_mask": torch.from_numpy(
                np.ascontiguousarray(sarn_mask[None], dtype=np.float32)
            ),
            "sarn_active": torch.tensor(sarn_active, dtype=torch.bool),
            "raw_to_sarn_homography": torch.from_numpy(
                np.ascontiguousarray(raw_to_sarn_homography)
            ),
            "rectified_view": normalized_rgb_tensor(views.view_b_bgr),
            "rectified_support_mask": torch.from_numpy(
                np.ascontiguousarray(rectified_mask[None], dtype=np.float32)
            ),
            "rectified_active": torch.tensor(rectified_active, dtype=torch.bool),
            "target": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "geometry_pivot": torch.stack(
                (original_geometry[0], sarn_geometry[0], rectified_geometry[0])
            ),
            "geometry_direction_sin_cos": torch.stack(
                (original_geometry[1], sarn_geometry[1], rectified_geometry[1])
            ),
            "geometry_references": torch.stack(
                (original_geometry[2], sarn_geometry[2], rectified_geometry[2])
            ),
            "geometry_keypoints": geometry_keypoints,
            "a9_ordered_ticks_xy": a9_ticks,
            "a9_ordered_tick_mask": a9_tick_mask,
            "a9_raw_pivot_xy": a9_pivot,
            "a9_raw_pointer_xy": a9_pointer,
            "geometry_available": torch.tensor(
                (True, sarn_active, rectified_geometry_available), dtype=torch.bool
            ),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _require(values.shape == mask.shape, "masked loss shapes differ")
    _require(mask.dtype == torch.bool, "masked loss selector must be boolean")
    count = mask.sum()
    _require(bool(count > 0), "masked loss has no valid value")
    return values.masked_select(mask).mean()


def _gaussian_nll(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    variance = variance.clamp_min(torch.finfo(variance.dtype).eps)
    return 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variance)
        + (mean - target).square() / variance
    )


def pilot_objective(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    arm: PilotArm,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    oracle_temperature: float = ORACLE_TEMPERATURE,
    tail_fraction: float = TAIL_FRACTION,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised distributions, geometry, routing, and no-harm losses."""

    _require(oracle_temperature > 0.0, "oracle temperature must be positive")
    _require(0.0 < tail_fraction <= 1.0, "tail fraction must be in (0,1]")
    target = batch["target"].float()
    mean = outputs["mean"].float()
    variance = outputs["variance"].float()
    primary_mean = outputs["primary_mean"].float()
    view_means = outputs["view_means"].float()
    view_variances = outputs["view_variances"].float()
    view_available = outputs["view_available"].bool()
    routing_weights = outputs["weights"].float()
    _require(mean.shape == variance.shape == target.shape, "fused output shape mismatch")
    _require(
        view_means.shape == view_variances.shape == routing_weights.shape,
        "per-view distribution shape mismatch",
    )
    _require(
        view_means.shape == view_available.shape
        and view_means.shape[0] == target.shape[0]
        and view_means.shape[1] == len(VIEW_NAMES),
        "per-view availability shape mismatch",
    )

    supervised_views = view_available.clone()
    if not arm.supervise_all_available_views:
        supervised_views[:, 1:] = False
    expanded_target = target[:, None].expand_as(view_means)
    fused_mean_loss = F.smooth_l1_loss(mean, target, beta=0.05)
    view_mean_loss = _masked_mean(
        F.smooth_l1_loss(
            view_means, expanded_target, beta=0.05, reduction="none"
        ),
        supervised_views,
    )
    fused_nll = _gaussian_nll(mean, variance, target).mean()
    view_nll = _masked_mean(
        _gaussian_nll(view_means, view_variances, expanded_target),
        supervised_views,
    )

    zero = mean.sum() * 0.0
    pivot_loss = zero
    direction_loss = zero
    references_loss = zero
    if arm.use_geometry_auxiliary:
        geometry = outputs["geometry"]
        geometry_available = batch["geometry_available"].bool() & supervised_views
        pivot_error = F.smooth_l1_loss(
            geometry["pivot"].float(),
            batch["geometry_pivot"].float(),
            beta=0.05,
            reduction="none",
        ).mean(dim=2)
        direction_error = 1.0 - torch.sum(
            geometry["direction_sin_cos"].float()
            * batch["geometry_direction_sin_cos"].float(),
            dim=2,
        )
        references_error = F.smooth_l1_loss(
            geometry["references"].float(),
            batch["geometry_references"].float(),
            beta=0.05,
            reduction="none",
        ).mean(dim=2)
        pivot_loss = _masked_mean(pivot_error, geometry_available)
        direction_loss = _masked_mean(direction_error, geometry_available)
        references_loss = _masked_mean(references_error, geometry_available)

    oracle_routing_loss = zero
    if arm.use_oracle_routing:
        absolute_view_error = torch.abs(view_means - expanded_target)
        oracle_logits = (-absolute_view_error / oracle_temperature).masked_fill(
            ~supervised_views, -torch.inf
        )
        oracle_weights = torch.softmax(oracle_logits, dim=1)
        oracle_routing_loss = torch.sum(
            oracle_weights
            * (
                torch.log(oracle_weights.clamp_min(1.0e-8))
                - torch.log(routing_weights.clamp_min(1.0e-8))
            ),
            dim=1,
        ).mean()

    excess_over_primary = torch.relu(
        torch.abs(mean - target) - torch.abs(primary_mean - target)
    )
    tail_count = max(1, int(math.ceil(tail_fraction * target.numel())))
    tail_no_harm_loss = (
        torch.topk(excess_over_primary, k=tail_count).values.mean()
        if arm.use_tail_no_harm
        else zero
    )

    total = (
        weights.fused_mean * fused_mean_loss
        + weights.view_mean * view_mean_loss
        + weights.fused_nll * fused_nll
        + weights.view_nll * view_nll
        + arm.geometry_auxiliary_scale * weights.geometry_pivot * pivot_loss
        + arm.geometry_auxiliary_scale * weights.geometry_direction * direction_loss
        + arm.geometry_auxiliary_scale * weights.geometry_references * references_loss
        + weights.oracle_routing * oracle_routing_loss
        + weights.tail_no_harm * tail_no_harm_loss
    )
    _require(bool(torch.isfinite(total)), "pilot objective became non-finite")
    return total, {
        "total": total,
        "fused_mean": fused_mean_loss,
        "view_mean": view_mean_loss,
        "fused_nll": fused_nll,
        "view_nll": view_nll,
        "geometry_pivot": pivot_loss,
        "geometry_direction": direction_loss,
        "geometry_references": references_loss,
        "oracle_routing": oracle_routing_loss,
        "tail_no_harm": tail_no_harm_loss,
        "excess_over_primary_mean": excess_over_primary.mean(),
    }


def _configure_reproducibility(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _loader(
    dataset: SyncGSupportGeometryMultiViewDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    cuda: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(cuda),
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _forward_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    arm: PilotArm,
) -> dict[str, Any]:
    if arm.use_geometry_consistency_routing:
        _require(
            isinstance(model, EfficientNetB0GeometryConsistencyReliability),
            "A5 requires the geometry-consistency reliability wrapper",
        )
        return model(
            batch["original_view"],
            batch["sarn_view"],
            batch["sarn_support_mask"],
            sarn_active=batch["sarn_active"],
            raw_to_sarn_homography=batch["raw_to_sarn_homography"],
        )
    _require(
        isinstance(model, EfficientNetB0SupportGeometryMultiView),
        "A1--A4/A3G require the base multi-view model",
    )
    use_rectified = arm.ablation.use_rectified_view
    if arm.detach_geometry_from_encoder:
        return model(
            batch["original_view"],
            batch["sarn_view"],
            batch["sarn_support_mask"],
            sarn_active=batch["sarn_active"],
            rectified_view=None,
            rectified_support_mask=None,
            rectified_active=None,
            ablation=arm.ablation,
            detach_geometry_from_encoder=True,
        )
    return model(
        batch["original_view"],
        batch["sarn_view"],
        batch["sarn_support_mask"],
        sarn_active=batch["sarn_active"],
        rectified_view=batch["rectified_view"] if use_rectified else None,
        rectified_support_mask=(
            batch["rectified_support_mask"] if use_rectified else None
        ),
        rectified_active=batch["rectified_active"] if use_rectified else None,
        ablation=arm.ablation,
    )


class ModelParameterEMA:
    """Low-pass one training trajectory without mixing independent modes."""

    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        _require(0.0 < decay < 1.0, "EMA decay must be between zero and one")
        named_parameters = tuple(model.named_parameters())
        _require(bool(named_parameters), "EMA model has no parameters")
        self.decay = float(decay)
        self.names = tuple(name for name, _parameter in named_parameters)
        self.parameters = tuple(
            parameter.detach().clone() for _name, parameter in named_parameters
        )
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = tuple(parameter.detach() for parameter in model.parameters())
        _require(len(current) == len(self.parameters), "EMA parameter roster differs")
        torch._foreach_mul_(self.parameters, self.decay)
        torch._foreach_add_(self.parameters, current, alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        _require(self.updates > 0, "EMA received no optimizer updates")
        output = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        _require(
            all(name in output for name in self.names),
            "EMA parameter names differ from model state",
        )
        for name, value in zip(self.names, self.parameters, strict=True):
            output[name] = value.detach().cpu()
        return output


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    arm: PilotArm,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    after_optimizer_step: Callable[[torch.nn.Module], None] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    component_sums: dict[str, float] = {}
    fused_absolute_error = 0.0
    primary_absolute_error = 0.0
    no_harm_violations = 0
    total_samples = 0
    total_steps = 0
    optimizer_steps = 0
    condition_counts = {condition: 0 for condition in DEV_CONDITIONS}
    consistency_valid = 0
    consistency_correction_absolute_sum = 0.0
    consistency_rows = 0
    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    for step, raw_batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = _device_batch(raw_batch, device)
        for condition in raw_batch["condition_name"]:
            _require(condition in condition_counts, "batch contains an unknown condition")
            condition_counts[condition] += 1
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if use_amp else torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = _forward_batch(model, batch, arm=arm)
                loss, components = pilot_objective(outputs, batch, arm=arm)
            if training:
                _require(scaler is not None, "training gradient scaler is missing")
                previous_scale = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
                    optimizer_steps += 1
                    if after_optimizer_step is not None:
                        after_optimizer_step(model)
        targets = batch["target"]
        count = int(targets.numel())
        for name, value in components.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(
                value.detach().cpu()
            ) * count
        fused_error = torch.abs(outputs["mean"].detach() - targets)
        primary_error = torch.abs(outputs["primary_mean"].detach() - targets)
        fused_absolute_error += float(fused_error.sum().cpu())
        primary_absolute_error += float(primary_error.sum().cpu())
        no_harm_violations += int((fused_error > primary_error + 1.0e-8).sum().cpu())
        if "geometry_consistency" in outputs:
            consistency = outputs["geometry_consistency"]
            consistency_valid += int(consistency["transform_valid"].sum().cpu())
            consistency_correction_absolute_sum += float(
                torch.abs(consistency["sarn_logit_correction"]).sum().detach().cpu()
            )
            consistency_rows += count
        total_samples += count
        total_steps += 1
    _require(total_samples > 0, "epoch produced no samples")
    result = {
        name: value / total_samples for name, value in component_sums.items()
    }
    result.update(
        {
            "fused_nmae": fused_absolute_error / total_samples,
            "primary_nmae": primary_absolute_error / total_samples,
            "no_harm_violation_rate": no_harm_violations / total_samples,
            "samples": float(total_samples),
            "steps": float(total_steps),
            "optimizer_steps": float(optimizer_steps),
            "condition_counts": condition_counts,
        }
    )
    if consistency_rows > 0:
        result["geometry_consistency_valid_rate"] = (
            consistency_valid / consistency_rows
        )
        result["geometry_consistency_abs_logit_correction"] = (
            consistency_correction_absolute_sum / consistency_rows
        )
    return result


def train_pilot(
    *,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    arm_name: str,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    imagenet_pretrained: bool = True,
    use_rectified_view: bool = False,
    initial_checkpoint_path: Path | None = None,
    ema_decay: float | None = None,
    max_train_steps: int | None = None,
    max_dev_steps: int | None = None,
) -> dict[str, Any]:
    """Train one terminal pilot checkpoint from only the formal SyncG fit roster."""

    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid training sizes")
    _require(learning_rate > 0.0 and weight_decay >= 0.0, "invalid optimizer values")
    _require(
        ema_decay is None or 0.0 < float(ema_decay) < 1.0,
        "EMA decay must be between zero and one",
    )
    _require(
        max_train_steps is None or max_train_steps >= 1,
        "max train steps must be positive",
    )
    _require(max_dev_steps is None or max_dev_steps >= 1, "max dev steps must be positive")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"pilot output already exists: {output}")
    fit_samples, formal_roster = load_training_samples(manifest_path, split_path)
    _require(formal_roster.scene_disjoint, "formal SyncG roster is not scene-disjoint")
    _require(len(fit_samples) == FIT_SAMPLES, "formal fit sample count is not 14442")
    _require(
        len({sample.scene_stem for sample in fit_samples}) == FIT_SCENES,
        "formal fit scene count is not 131",
    )
    internal = make_internal_scene_split(fit_samples)
    arm = resolve_pilot_arm(arm_name, use_rectified_view=use_rectified_view)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)

    train_dataset = SyncGSupportGeometryMultiViewDataset(
        internal.train,
        training=True,
        seed=seed,
        total_epochs=epochs,
    )
    dev_datasets = {
        condition: SyncGSupportGeometryMultiViewDataset(
            internal.dev,
            training=False,
            seed=seed + 31_337,
            total_epochs=epochs,
            condition=condition,
        )
        for condition in DEV_CONDITIONS
    }
    model: torch.nn.Module
    if arm.use_geometry_consistency_routing:
        model = EfficientNetB0GeometryConsistencyReliability(
            imagenet_pretrained=imagenet_pretrained,
        )
    else:
        model = EfficientNetB0SupportGeometryMultiView(
            imagenet_pretrained=imagenet_pretrained,
            default_ablation=arm.ablation,
        )
    initialization = (
        initialize_encoder_from_lightweight_checkpoint(model, initial_checkpoint_path)
        if initial_checkpoint_path is not None
        else {
            "path": None,
            "protocol": "torchvision ImageNet initialization",
            "architecture": "efficientnet_b0",
            "seed": None,
            "scope": "complete encoder only; task heads are newly initialized",
        }
    )
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    _require(
        total_parameters == trainable_parameters,
        "pilot must optimize the complete model end to end",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    use_fp16_scaler = (
        device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    ema = ModelParameterEMA(model, decay=float(ema_decay)) if ema_decay is not None else None
    history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        train_dataset.set_epoch(epoch_index)
        for dev_dataset in dev_datasets.values():
            dev_dataset.set_epoch(epoch_index)
        train_loader = _loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + epoch_index,
            cuda=device.type == "cuda",
        )
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            arm=arm,
            optimizer=optimizer,
            scaler=scaler,
            after_optimizer_step=ema.update if ema is not None else None,
            max_steps=max_train_steps,
        )
        dev_metrics: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for condition_index, condition in enumerate(DEV_CONDITIONS):
                dev_loader = _loader(
                    dev_datasets[condition],
                    batch_size=batch_size,
                    shuffle=False,
                    workers=workers,
                    seed=seed + 10_000 + epoch_index + condition_index * 1_009,
                    cuda=device.type == "cuda",
                )
                dev_metrics[condition] = run_epoch(
                    model,
                    dev_loader,
                    device=device,
                    arm=arm,
                    optimizer=None,
                    scaler=None,
                    max_steps=max_dev_steps,
                )
        history.append(
            {
                "epoch": epoch_index + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "dev": dev_metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()

    truncated = max_train_steps is not None or max_dev_steps is not None
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": (
            A5_ARCHITECTURE
            if arm.use_geometry_consistency_routing
            else ARCHITECTURE
        ),
        "arm": arm.name,
        "arm_description": arm.description,
        "architecture_ablation": asdict(arm.ablation),
        "seed": int(seed),
        "image_size": IMAGE_SIZE,
        "pretrained_weights": (
            "torchvision.EfficientNet_B0_Weights.IMAGENET1K_V1"
            if imagenet_pretrained
            else "none"
        ),
        "initialization": initialization,
        "data": {
            "source": "SyncG formal fit only",
            "formal_split_protocol": formal_roster.protocol,
            "formal_fit_samples": len(fit_samples),
            "formal_holdout_boundary": {
                "sample_id_count": len(formal_roster.validation_ids),
                "manifest_access": "sample_id_only",
            },
            "internal_split_protocol": INTERNAL_SPLIT_PROTOCOL,
            "internal_scene_disjoint": True,
            "internal_train_samples": len(internal.train),
            "internal_dev_samples": len(internal.dev),
            "internal_train_scenes": list(internal.train_scenes),
            "internal_dev_scenes": list(internal.dev_scenes),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
            "checkpoint_selection": "terminal_fixed_epoch_dev_is_diagnostic_only",
            "all_model_parameters_trainable": True,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "autocast_precision": (
                "bfloat16"
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else "float16"
                if device.type == "cuda"
                else "float32"
            ),
            "loss_weights": asdict(DEFAULT_LOSS_WEIGHTS),
            "oracle_temperature": ORACLE_TEMPERATURE,
            "tail_fraction": TAIL_FRACTION,
            "train_condition_mix": dict(TRAIN_CONDITION_MIX),
            "dev_conditions": list(DEV_CONDITIONS),
            "truncated_smoke": truncated,
            "max_train_steps": max_train_steps,
            "max_dev_steps": max_dev_steps,
            "ema_decay": float(ema_decay) if ema_decay is not None else None,
            "ema_scope": (
                "all trainable parameters; terminal non-parameter buffers"
                if ema is not None
                else None
            ),
            "ema_updates": int(ema.updates) if ema is not None else 0,
        },
        "history": history,
        "weight_variant": "ema" if ema is not None else "terminal",
        "model_state": (
            ema.state_dict(model)
            if ema is not None
            else {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
        ),
    }
    if arm.name in ("A3G", "A5"):
        checkpoint["internal_geometry_extension"] = {
            "geometry_auxiliary_weights": {
                "pivot": DEFAULT_LOSS_WEIGHTS.geometry_pivot
                * arm.geometry_auxiliary_scale,
                "direction": DEFAULT_LOSS_WEIGHTS.geometry_direction
                * arm.geometry_auxiliary_scale,
                "references": DEFAULT_LOSS_WEIGHTS.geometry_references
                * arm.geometry_auxiliary_scale,
            },
            "middle_stage_geometry_modulation": False,
            "geometry_encoder_stop_gradient": arm.detach_geometry_from_encoder,
            "geometry_consistency_routing": arm.use_geometry_consistency_routing,
            "raw_to_sarn_transform": (
                "normalized_3x3_pixel_center_affine; identity on SARN fallback"
            ),
            "screen_only": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "arm": arm.name,
        "terminal_train_nmae": history[-1]["train"]["fused_nmae"],
        "terminal_dev_nmae_by_condition": {
            condition: history[-1]["dev"][condition]["fused_nmae"]
            for condition in DEV_CONDITIONS
        },
        "train_samples_seen": history[-1]["train"]["samples"],
        "dev_samples_seen_per_condition": {
            condition: history[-1]["dev"][condition]["samples"]
            for condition in DEV_CONDITIONS
        },
        "truncated_smoke": truncated,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SYNCG_SCENE_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("A1", "A2", "A3", "A3G", "A4", "A5"), default="A4"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--imagenet-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rectified-view",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-dev-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_pilot(
        manifest_path=args.manifest,
        split_path=args.split,
        output_path=args.output,
        arm_name=args.arm,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        imagenet_pretrained=args.imagenet_pretrained,
        use_rectified_view=args.rectified_view,
        initial_checkpoint_path=args.initial_checkpoint,
        ema_decay=args.ema_decay,
        max_train_steps=args.max_train_steps,
        max_dev_steps=args.max_dev_steps,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOSS_WEIGHTS",
    "initialize_encoder_from_lightweight_checkpoint",
    "InternalSceneSplit",
    "LossWeights",
    "ModelParameterEMA",
    "PilotArm",
    "PilotTrainingError",
    "SyncGSupportGeometryMultiViewDataset",
    "build_argument_parser",
    "make_internal_scene_split",
    "pilot_objective",
    "resolve_pilot_arm",
    "run_epoch",
    "train_pilot",
]
