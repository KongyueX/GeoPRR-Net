"""Probe a Raw-only layer4 + scalar-endpoint refinement for ReMST-ResNet18.

The probe keeps the published single-backbone topology.  It starts from one
completed ReMST-ResNet18 checkpoint, freezes the stem through ``layer3`` and
the ReMST correction, and updates only ``layer4`` plus the existing 512-to-1
point projection.  Each fit sample contributes a clean view and two randomly
jittered Gaussian-blur views.  Selection is confined to the pre-existing
14-scene inner development partition; the formal holdout is never opened.

After Raw selection, the selected anchor is replayed with the unchanged ReMST
correction on the correction-development cohort.  This makes any projective
regression visible before the candidate is considered for the matched main
table.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from experiments.a10_pccot_protocol import (
    DEFAULT_SEED as CORRECTION_EVALUATION_SEED,
    condition_evaluation_specs,
)
from experiments.a13_correction_dev_protocol import (
    EVALUATION_CONDITIONS as CORRECTION_EVALUATION_CONDITIONS,
    EVALUATION_PIXEL_EPOCH,
    EVALUATION_PIXEL_TOTAL_EPOCHS,
    PROJECTIVE_CONDITIONS as CORRECTION_PROJECTIVE_CONDITIONS,
)
from experiments.evaluate_a15_2_fteb_correction_dev import (
    EVALUATION_BATCH_SIZE as CORRECTION_EVALUATION_BATCH_SIZE,
    build_a15_2_dev_evaluation_dataset,
)
from experiments.evaluate_remst_resnet18_probe import (
    METHOD_REMST,
    _condition_digest,
    _evaluate_remst_loader,
    _pool_condition_results,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_DEV_MANIFEST,
    load_a13_correction_dev_manifest,
)
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_FIT_MANIFEST,
    ConditionedProgressDataset,
    load_inner_scene_population,
)
from experiments.remst_resnet18 import MomentExactResNet18Anchor
from experiments.resnet18_direct_progress import (
    IMAGE_SIZE,
    DirectSample,
    _configure_reproducibility,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.train_remst_resnet18_probe import (
    load_remst_resnet18_probe,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader as correction_loader,
)
from experiments.syncg_lightweight_regression_baselines import (
    DEFAULT_SCENE_SPLIT,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor


PROTOCOL: Final[str] = "remst_resnet18_raw_layer4_endpoint_refinement_probe_v1"
DEFAULT_SOURCE_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_midpoint/seed_20262020/remst_terminal.pt"
)
DEFAULT_SOURCE_CORRECTION_DEV_RESULT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_midpoint/seed_20262020/"
    "correction_dev_results.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_layer4_probe/seed_20262020"
)
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_EPOCHS: Final[int] = 5
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_LAYER4_LEARNING_RATE: Final[float] = 2.0e-5
DEFAULT_ENDPOINT_LEARNING_RATE: Final[float] = 2.0e-4
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_CONSISTENCY_WEIGHT: Final[float] = 0.10
DEFAULT_SOURCE_RETENTION_WEIGHT: Final[float] = 0.05
DEFAULT_L2SP_WEIGHT: Final[float] = 1.0
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 2_000
RAW_CONDITIONS: Final[tuple[str, ...]] = tuple(CONDITIONS[:3])
TRAIN_VIEW_NAMES: Final[tuple[str, ...]] = (
    "clean",
    "random_blur_moderate",
    "random_blur_severe",
)
MODERATE_SIGMA_RANGE: Final[tuple[float, float]] = (0.20, 0.65)
SEVERE_SIGMA_RANGE: Final[tuple[float, float]] = (0.65, 1.25)


class RawLayer4ProbeError(ValueError):
    """The Raw layer4 probe configuration or artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawLayer4ProbeError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _is_refined_state_name(name: str) -> bool:
    return name.startswith("raw_encoder.layer4.") or name.startswith(
        "raw_posterior_head.point_projection."
    )


def configure_refinement_scope(anchor: MomentExactResNet18Anchor) -> tuple[str, ...]:
    """Freeze the anchor except layer4 and its pre-existing scalar endpoint."""

    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in anchor.raw_encoder.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in anchor.raw_posterior_head.point_projection.parameters():
        parameter.requires_grad_(True)
    names = tuple(name for name, value in anchor.named_parameters() if value.requires_grad)
    _require(bool(names), "Raw refinement has no trainable parameters")
    _require(
        all(_is_refined_state_name(name) for name in names),
        "Raw refinement opened parameters outside layer4 + point projection",
    )
    _require(
        any(name.startswith("raw_encoder.layer4.") for name in names)
        and any(
            name.startswith("raw_posterior_head.point_projection.") for name in names
        ),
        "Raw refinement scope is incomplete",
    )
    return names


def _set_refinement_train_mode(
    anchor: MomentExactResNet18Anchor,
    *,
    freeze_layer4_batch_norm_stats: bool = False,
) -> None:
    anchor.eval()
    anchor.raw_encoder.layer4.train(True)
    if freeze_layer4_batch_norm_stats:
        for module in anchor.raw_encoder.layer4.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    anchor.raw_posterior_head.point_projection.train(True)


def raw_point_progress(
    anchor: MomentExactResNet18Anchor, images: torch.Tensor
) -> torch.Tensor:
    features = anchor.raw_encoder(images)
    return anchor.raw_posterior_head.point_progress(features["representation"])


class PairedRawBlurDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]
):
    """Clean plus two jittered blur views from one canonical Raw ROI."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        seed: int,
        image_size: int = IMAGE_SIZE,
        moderate_sigma_range: tuple[float, float] = MODERATE_SIGMA_RANGE,
        severe_sigma_range: tuple[float, float] = SEVERE_SIGMA_RANGE,
    ) -> None:
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.moderate_sigma_range = tuple(float(value) for value in moderate_sigma_range)
        self.severe_sigma_range = tuple(float(value) for value in severe_sigma_range)
        self.epoch = 0
        _require(bool(self.samples), "paired Raw dataset is empty")
        _require(self.image_size >= 32, "paired Raw image size is too small")
        for name, bounds in (
            ("moderate", self.moderate_sigma_range),
            ("severe", self.severe_sigma_range),
        ):
            _require(
                len(bounds) == 2 and 0.0 < bounds[0] <= bounds[1],
                f"{name} blur sigma range is invalid",
            )
        _require(
            self.moderate_sigma_range[1] <= self.severe_sigma_range[1],
            "moderate blur range exceeds severe blur range",
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
        return cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        sample = self.samples[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source image decode failed")
        roi, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        clean = direct_resize_whole_roi(roi, size=self.image_size)
        generator = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + int(index) * 97
        )
        moderate_sigma = float(generator.uniform(*self.moderate_sigma_range))
        severe_sigma = float(generator.uniform(*self.severe_sigma_range))
        views = torch.stack(
            (
                normalized_rgb_tensor(clean),
                normalized_rgb_tensor(self._blur(clean, moderate_sigma)),
                normalized_rgb_tensor(self._blur(clean, severe_sigma)),
            )
        )
        return (
            views,
            torch.tensor(sample.normalized_target, dtype=torch.float32),
            sample.sample_id,
            sample.scene_stem,
        )


def scene_balancing_weights(samples: Sequence[DirectSample]) -> torch.Tensor:
    values = tuple(samples)
    _require(bool(values), "scene balancing population is empty")
    counts = Counter(sample.scene_stem for sample in values)
    _require(all(count >= 1 for count in counts.values()), "scene count is invalid")
    return torch.tensor(
        [1.0 / counts[sample.scene_stem] for sample in values],
        dtype=torch.double,
    )


def build_scene_balanced_loader(
    dataset: PairedRawBlurDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    cuda: bool,
) -> DataLoader:
    sampler = WeightedRandomSampler(
        scene_balancing_weights(dataset.samples),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(workers),
        pin_memory=bool(cuda),
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(int(seed) + 1),
    )


def _l2sp_loss(
    named_parameters: Mapping[str, nn.Parameter],
    source_parameters: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    _require(
        set(named_parameters) == set(source_parameters),
        "L2-SP parameter roster differs",
    )
    total: torch.Tensor | None = None
    elements = 0
    for name, parameter in named_parameters.items():
        difference = parameter.float() - source_parameters[name]
        term = difference.square().sum()
        total = term if total is None else total + term
        elements += int(parameter.numel())
    _require(total is not None and elements >= 1, "L2-SP parameter set is empty")
    return total / elements


def train_refinement_epoch(
    student: MomentExactResNet18Anchor,
    teacher: MomentExactResNet18Anchor,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    source_parameters: Mapping[str, torch.Tensor],
    consistency_weight: float,
    source_retention_weight: float,
    l2sp_weight: float,
    freeze_layer4_batch_norm_stats: bool = False,
) -> dict[str, Any]:
    _set_refinement_train_mode(
        student,
        freeze_layer4_batch_norm_stats=freeze_layer4_batch_norm_stats,
    )
    teacher.eval()
    use_amp = device.type == "cuda"
    trainable = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    totals = defaultdict(float)
    samples = 0
    optimizer_steps = 0
    for views, targets, _sample_ids, _scene_stems in loader:
        views = views.to(device, non_blocking=use_amp)
        targets = targets.to(device, non_blocking=use_amp)
        batch_size, view_count = int(views.shape[0]), int(views.shape[1])
        _require(view_count == len(TRAIN_VIEW_NAMES), "training view count differs")
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            source_clean = raw_point_progress(teacher, views[:, 0])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            predictions = raw_point_progress(
                student, views.flatten(0, 1)
            ).reshape(batch_size, view_count)
            exact_l1_by_view = torch.stack(
                tuple(
                    F.l1_loss(predictions[:, index], targets)
                    for index in range(view_count)
                )
            )
            supervised = exact_l1_by_view.mean()
            clean_reference = predictions[:, 0].detach()
            consistency = 0.5 * (
                F.l1_loss(predictions[:, 1], clean_reference)
                + F.l1_loss(predictions[:, 2], clean_reference)
            )
            source_retention = F.l1_loss(predictions[:, 0], source_clean)
            l2sp = _l2sp_loss(trainable, source_parameters)
            loss = (
                supervised
                + float(consistency_weight) * consistency
                + float(source_retention_weight) * source_retention
                + float(l2sp_weight) * l2sp
            )
        previous_scale = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
            optimizer_steps += 1
        count = int(targets.numel())
        samples += count
        totals["total"] += float(loss.detach().cpu()) * count
        totals["supervised"] += float(supervised.detach().cpu()) * count
        totals["consistency"] += float(consistency.detach().cpu()) * count
        totals["source_retention"] += float(source_retention.detach().cpu()) * count
        totals["l2sp"] += float(l2sp.detach().cpu()) * count
        for index, name in enumerate(TRAIN_VIEW_NAMES):
            totals[f"exact_l1_{name}"] += (
                float(exact_l1_by_view[index].detach().cpu()) * count
            )
    _require(samples >= 1, "Raw refinement epoch produced no samples")
    return {
        "samples": samples,
        "optimizer_steps": optimizer_steps,
        "loss": {name: value / samples for name, value in sorted(totals.items())},
    }


@dataclass(frozen=True, slots=True)
class RawPrediction:
    sample_id: str
    scene_stem: str
    condition: str
    target: float
    prediction: float


def evaluate_raw_anchor(
    anchor: MomentExactResNet18Anchor,
    samples: Sequence[DirectSample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[RawPrediction, ...]:
    anchor.eval()
    records: list[RawPrediction] = []
    for condition_index, condition in enumerate(RAW_CONDITIONS):
        dataset = ConditionedProgressDataset(samples, condition=condition)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(workers),
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(int(seed) + condition_index),
        )
        with torch.inference_mode():
            for images, targets, sample_ids, scene_stems in loader:
                predictions = raw_point_progress(
                    anchor, images.to(device, non_blocking=device.type == "cuda")
                ).float().cpu()
                for sample_id, scene_stem, target, prediction in zip(
                    sample_ids,
                    scene_stems,
                    targets.float().tolist(),
                    predictions.tolist(),
                    strict=True,
                ):
                    value = float(prediction)
                    _require(
                        math.isfinite(value) and 0.0 <= value <= 1.0,
                        "Raw evaluation prediction is invalid",
                    )
                    records.append(
                        RawPrediction(
                            sample_id=str(sample_id),
                            scene_stem=str(scene_stem),
                            condition=condition,
                            target=float(target),
                            prediction=value,
                        )
                    )
    _require(
        len(records) == len(samples) * len(RAW_CONDITIONS),
        "Raw evaluation row count differs",
    )
    return tuple(records)


def compare_raw_predictions(
    source: Sequence[RawPrediction],
    candidate: Sequence[RawPrediction],
) -> dict[str, Any]:
    source_rows = tuple(source)
    candidate_rows = tuple(candidate)
    _require(len(source_rows) == len(candidate_rows), "Raw comparison length differs")
    for expected, observed in zip(source_rows, candidate_rows, strict=True):
        _require(
            (
                expected.sample_id,
                expected.scene_stem,
                expected.condition,
                expected.target,
            )
            == (
                observed.sample_id,
                observed.scene_stem,
                observed.condition,
                observed.target,
            ),
            "Raw comparison roster/order differs",
        )

    def summarize(rows: Sequence[tuple[RawPrediction, RawPrediction]]) -> dict[str, float]:
        source_errors = np.asarray(
            [abs(left.prediction - left.target) for left, _right in rows],
            dtype=np.float64,
        )
        candidate_errors = np.asarray(
            [abs(right.prediction - right.target) for _left, right in rows],
            dtype=np.float64,
        )
        source_nmae = float(source_errors.mean())
        candidate_nmae = float(candidate_errors.mean())
        delta = candidate_nmae - source_nmae
        return {
            "source_nmae": source_nmae,
            "candidate_nmae": candidate_nmae,
            "candidate_minus_source": delta,
            "relative_error_reduction": -delta / source_nmae,
        }

    paired = tuple(zip(source_rows, candidate_rows, strict=True))
    conditions = {
        condition: summarize(
            tuple(pair for pair in paired if pair[0].condition == condition)
        )
        for condition in RAW_CONDITIONS
    }
    pooled = summarize(paired)
    return {"conditions": conditions, "raw_pooled": pooled}


def scene_cluster_bootstrap(
    source: Sequence[RawPrediction],
    candidate: Sequence[RawPrediction],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(replicates >= 1, "bootstrap replicate count must be positive")
    paired = tuple(zip(source, candidate, strict=True))

    def one_scope(
        selected: Sequence[tuple[RawPrediction, RawPrediction]], scope_seed: int
    ) -> dict[str, Any]:
        rows = tuple(selected)
        by_scene: dict[str, list[int]] = defaultdict(list)
        deltas = np.empty(len(rows), dtype=np.float64)
        for index, (left, right) in enumerate(rows):
            by_scene[left.scene_stem].append(index)
            deltas[index] = abs(right.prediction - right.target) - abs(
                left.prediction - left.target
            )
        scene_names = tuple(sorted(by_scene))
        _require(len(scene_names) >= 2, "scene bootstrap needs at least two scenes")
        scene_indices = tuple(
            np.asarray(by_scene[name], dtype=np.int64) for name in scene_names
        )
        generator = np.random.default_rng(int(scope_seed))
        replicate_values = np.empty(int(replicates), dtype=np.float64)
        for replicate in range(int(replicates)):
            selected_scenes = generator.integers(
                0, len(scene_names), size=len(scene_names)
            )
            selected_rows = np.concatenate(
                tuple(scene_indices[int(index)] for index in selected_scenes)
            )
            replicate_values[replicate] = float(deltas[selected_rows].mean())
        return {
            "rows": len(rows),
            "scenes": len(scene_names),
            "mean_nmae_delta": float(deltas.mean()),
            "scene_bootstrap_95_ci": [
                float(np.quantile(replicate_values, 0.025)),
                float(np.quantile(replicate_values, 0.975)),
            ],
            "probability_delta_below_zero": float(
                np.mean(replicate_values < 0.0)
            ),
        }

    return {
        **{
            condition: one_scope(
                tuple(pair for pair in paired if pair[0].condition == condition),
                int(seed) + index * 1_009,
            )
            for index, condition in enumerate(RAW_CONDITIONS)
        },
        "raw_pooled": one_scope(paired, int(seed) + len(RAW_CONDITIONS) * 1_009),
        "bootstrap": {
            "unit": "scene_stem",
            "replicates": int(replicates),
            "seed": int(seed),
        },
    }


def evaluate_projection_compatibility(
    anchor: MomentExactResNet18Anchor,
    correction: nn.Module,
    *,
    manifest_path: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    samples = tuple(load_a13_correction_dev_manifest(Path(manifest_path).resolve()))
    expected_ids = tuple(sample.sample_id for sample in samples)
    specs = condition_evaluation_specs(CORRECTION_EVALUATION_SEED)
    _require(
        tuple(spec.condition for spec in specs) == CORRECTION_EVALUATION_CONDITIONS,
        "correction evaluation condition roster differs",
    )
    anchor.eval()
    correction.eval()
    conditions: dict[str, dict[str, Any]] = {}
    for spec in specs:
        dataset = build_a15_2_dev_evaluation_dataset(
            samples,
            seed=spec.dataset_seed,
            total_epochs=spec.dataset_total_epochs,
            condition=spec.condition,
        )
        dataset.set_epoch(spec.transform_epoch)
        loader = correction_loader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            workers=int(workers),
            seed=spec.loader_seed,
            cuda=device.type == "cuda",
        )
        evaluated = _evaluate_remst_loader(
            anchor,
            correction,
            loader,
            device=device,
            expected_sample_ids=expected_ids,
            expected_condition=spec.condition,
        )
        conditions[spec.condition] = evaluated
        digest = _condition_digest(evaluated)
        print(
            json.dumps(
                {
                    "phase": "projection_compatibility",
                    "condition": spec.condition,
                    "rows": digest["rows"],
                    "candidate_nmae": digest["metrics"]["methods"][METHOD_REMST][
                        "nmae"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    pooled = _pool_condition_results(conditions)
    return {
        "data": {
            "manifest": str(Path(manifest_path).resolve()),
            "physical_samples": len(samples),
            "conditions": list(CORRECTION_EVALUATION_CONDITIONS),
            "projective_conditions": list(CORRECTION_PROJECTIVE_CONDITIONS),
            "pixel_total_epochs": EVALUATION_PIXEL_TOTAL_EPOCHS,
            "transform_epoch": EVALUATION_PIXEL_EPOCH,
        },
        "conditions": {
            name: _condition_digest(value) for name, value in conditions.items()
        },
        "pooled": pooled,
    }


def load_source_projection_reference(
    result_path: Path, *, expected_checkpoint: Path
) -> dict[str, Any]:
    source = Path(result_path).resolve()
    _require(source.is_file(), f"source correction result is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawLayer4ProbeError("source correction result cannot be read") from exc
    _require(isinstance(value, Mapping), "source correction result is malformed")
    checkpoint = value.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "source result checkpoint is missing")
    _require(
        Path(str(checkpoint.get("checkpoint"))).resolve()
        == Path(expected_checkpoint).resolve(),
        "source correction result belongs to another checkpoint",
    )
    return dict(value)


def _projection_comparison(
    source: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    condition_rows: dict[str, Any] = {}
    for condition in CORRECTION_EVALUATION_CONDITIONS:
        source_nmae = float(
            source["conditions"][condition]["metrics"]["methods"][METHOD_REMST][
                "nmae"
            ]
        )
        candidate_nmae = float(
            candidate["conditions"][condition]["metrics"]["methods"][METHOD_REMST][
                "nmae"
            ]
        )
        condition_rows[condition] = {
            "source_nmae": source_nmae,
            "candidate_nmae": candidate_nmae,
            "candidate_minus_source": candidate_nmae - source_nmae,
            "relative_change": (candidate_nmae - source_nmae) / source_nmae,
        }
    source_pooled = float(
        source["pooled"]["projective_conditions"]["methods"][METHOD_REMST]["nmae"]
    )
    candidate_pooled = float(
        candidate["pooled"]["projective_conditions"]["methods"][METHOD_REMST][
            "nmae"
        ]
    )
    return {
        "conditions": condition_rows,
        "projective_pooled": {
            "source_nmae": source_pooled,
            "candidate_nmae": candidate_pooled,
            "candidate_minus_source": candidate_pooled - source_pooled,
            "relative_change": (candidate_pooled - source_pooled) / source_pooled,
        },
    }


def load_raw_layer4_refined_remst(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[MomentExactResNet18Anchor, nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"Raw refinement checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping) and payload.get("protocol") == PROTOCOL,
        "Raw refinement checkpoint metadata differs",
    )
    anchor_state = payload.get("anchor_state")
    _require(isinstance(anchor_state, Mapping), "Raw refinement anchor state is missing")
    target_device = torch.device(device)
    anchor, correction, source_metadata = load_remst_resnet18_probe(
        Path(str(payload["source_remst_checkpoint"])), device=target_device
    )
    incompatibility = anchor.load_state_dict(anchor_state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "Raw refinement anchor does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    return anchor, correction, {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "source_remst_checkpoint": str(payload["source_remst_checkpoint"]),
        "selected_epoch": int(payload["selection"]["selected_epoch"]),
        "single_backbone_parameter_set": True,
        "source_correction_unchanged": True,
        "source_checkpoint": source_metadata,
    }


def run_probe(
    *,
    source_checkpoint_path: Path,
    source_correction_dev_result_path: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    correction_dev_manifest_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    layer4_learning_rate: float = DEFAULT_LAYER4_LEARNING_RATE,
    endpoint_learning_rate: float = DEFAULT_ENDPOINT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    source_retention_weight: float = DEFAULT_SOURCE_RETENTION_WEIGHT,
    l2sp_weight: float = DEFAULT_L2SP_WEIGHT,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    max_train_samples: int | None = None,
    max_dev_samples: int | None = None,
    skip_projection_evaluation: bool = False,
) -> dict[str, Any]:
    _require(
        epochs >= 1 and batch_size >= 1 and workers >= 0,
        "Raw refinement training sizes are invalid",
    )
    _require(
        layer4_learning_rate > 0.0
        and endpoint_learning_rate > 0.0
        and weight_decay >= 0.0
        and min(consistency_weight, source_retention_weight, l2sp_weight) >= 0.0,
        "Raw refinement optimizer/loss settings are invalid",
    )
    _require(
        bootstrap_replicates >= 1,
        "Raw refinement bootstrap replicate count is invalid",
    )
    _require(
        max_train_samples is None or max_train_samples >= 1,
        "maximum train sample count is invalid",
    )
    _require(
        max_dev_samples is None or max_dev_samples >= 1,
        "maximum dev sample count is invalid",
    )
    root = Path(output_dir).resolve()
    paths = {
        "checkpoint": root / "selected.pt",
        "predictions": root / "raw_inner_dev_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "Raw refinement output artifact already exists",
    )
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(int(seed), device)

    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)
    train_samples = tuple(internal.train)
    dev_samples = tuple(internal.dev)
    if max_train_samples is not None:
        train_samples = train_samples[: int(max_train_samples)]
    if max_dev_samples is not None:
        dev_samples = dev_samples[: int(max_dev_samples)]
    _require(bool(train_samples) and bool(dev_samples), "probe partition is empty")

    source_anchor_cpu, source_correction, source_metadata = load_remst_resnet18_probe(
        source_checkpoint_path, device="cpu"
    )
    student = copy.deepcopy(source_anchor_cpu).to(device)
    teacher = source_anchor_cpu.to(device).eval()
    trainable_names = configure_refinement_scope(student)
    source_full_state = _state_cpu(student)
    trainable_parameters = {
        name: parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    source_trainable_parameters = {
        name: parameter.detach().float().clone()
        for name, parameter in trainable_parameters.items()
    }
    layer4_parameters = [
        parameter
        for name, parameter in trainable_parameters.items()
        if name.startswith("raw_encoder.layer4.")
    ]
    endpoint_parameters = [
        parameter
        for name, parameter in trainable_parameters.items()
        if name.startswith("raw_posterior_head.point_projection.")
    ]
    _require(
        layer4_parameters and endpoint_parameters,
        "Raw refinement optimizer groups are incomplete",
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": layer4_parameters, "lr": float(layer4_learning_rate)},
            {"params": endpoint_parameters, "lr": float(endpoint_learning_rate)},
        ),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_dataset = PairedRawBlurDataset(train_samples, seed=int(seed))

    raw_evaluation_started = time.perf_counter()
    source_raw_records = evaluate_raw_anchor(
        teacher,
        dev_samples,
        device=device,
        batch_size=batch_size,
        workers=workers,
        seed=int(seed) + 200_000,
    )
    source_raw_metrics = compare_raw_predictions(
        source_raw_records, source_raw_records
    )
    print(
        json.dumps(
            {
                "phase": "raw_inner_dev_source",
                "rows": len(source_raw_records),
                "raw_pooled_nmae": source_raw_metrics["raw_pooled"]["source_nmae"],
                "conditions": {
                    name: value["source_nmae"]
                    for name, value in source_raw_metrics["conditions"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    history: list[dict[str, Any]] = []
    selected_epoch = 0
    selected_score = float(source_raw_metrics["raw_pooled"]["source_nmae"])
    selected_state = _state_cpu(student)
    selected_records = source_raw_records
    training_started = time.perf_counter()
    for epoch_index in range(int(epochs)):
        epoch_started = time.perf_counter()
        train_dataset.set_epoch(epoch_index)
        loader = build_scene_balanced_loader(
            train_dataset,
            batch_size=batch_size,
            workers=workers,
            seed=int(seed) + 100_000 + epoch_index,
            cuda=device.type == "cuda",
        )
        train_metrics = train_refinement_epoch(
            student,
            teacher,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            source_parameters=source_trainable_parameters,
            consistency_weight=consistency_weight,
            source_retention_weight=source_retention_weight,
            l2sp_weight=l2sp_weight,
        )
        candidate_records = evaluate_raw_anchor(
            student,
            dev_samples,
            device=device,
            batch_size=batch_size,
            workers=workers,
            seed=int(seed) + 200_000,
        )
        comparison = compare_raw_predictions(source_raw_records, candidate_records)
        score = float(comparison["raw_pooled"]["candidate_nmae"])
        if score < selected_score:
            selected_epoch = epoch_index + 1
            selected_score = score
            selected_state = _state_cpu(student)
            selected_records = candidate_records
        row = {
            "epoch": epoch_index + 1,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "learning_rates": {
                "layer4": float(optimizer.param_groups[0]["lr"]),
                "point_projection": float(optimizer.param_groups[1]["lr"]),
            },
            "train": train_metrics,
            "raw_inner_dev": comparison,
            "selected_after_epoch": selected_epoch,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "phase": "raw_refinement",
                    "epoch": row["epoch"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "train_total": train_metrics["loss"]["total"],
                    "raw_pooled_nmae": score,
                    "raw_pooled_relative_error_reduction": comparison["raw_pooled"][
                        "relative_error_reduction"
                    ],
                    "condition_deltas": {
                        name: value["candidate_minus_source"]
                        for name, value in comparison["conditions"].items()
                    },
                    "selected_epoch": selected_epoch,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()
    training_elapsed = time.perf_counter() - training_started
    selected_load = student.load_state_dict(selected_state, strict=True)
    _require(
        not selected_load.missing_keys and not selected_load.unexpected_keys,
        "selected Raw refinement state does not load strictly",
    )
    student.eval()
    terminal_state = _state_cpu(student)
    frozen_state_unchanged = all(
        name in terminal_state and torch.equal(expected, terminal_state[name])
        for name, expected in source_full_state.items()
        if not _is_refined_state_name(name)
    )
    _require(frozen_state_unchanged, "frozen stem-through-layer3 state changed")
    selected_raw_comparison = compare_raw_predictions(
        source_raw_records, selected_records
    )
    bootstrap = scene_cluster_bootstrap(
        source_raw_records,
        selected_records,
        replicates=bootstrap_replicates,
        seed=int(seed) + 300_000,
    )

    projection_candidate: dict[str, Any] | None = None
    projection_comparison: dict[str, Any] | None = None
    projection_elapsed = 0.0
    if not skip_projection_evaluation:
        projection_started = time.perf_counter()
        source_projection = load_source_projection_reference(
            source_correction_dev_result_path,
            expected_checkpoint=source_checkpoint_path,
        )
        source_correction = source_correction.to(device)
        projection_candidate = evaluate_projection_compatibility(
            student,
            source_correction,
            manifest_path=correction_dev_manifest_path,
            device=device,
            workers=workers,
            batch_size=CORRECTION_EVALUATION_BATCH_SIZE,
        )
        projection_comparison = _projection_comparison(
            source_projection, projection_candidate
        )
        projection_elapsed = time.perf_counter() - projection_started

    with paths["predictions"].open("w", encoding="utf-8", newline="\n") as stream:
        for source_row, candidate_row in zip(
            source_raw_records, selected_records, strict=True
        ):
            stream.write(
                json.dumps(
                    {
                        "sample_id": source_row.sample_id,
                        "scene_stem": source_row.scene_stem,
                        "condition": source_row.condition,
                        "target": source_row.target,
                        "source_prediction": source_row.prediction,
                        "candidate_prediction": candidate_row.prediction,
                        "source_absolute_error": abs(
                            source_row.prediction - source_row.target
                        ),
                        "candidate_absolute_error": abs(
                            candidate_row.prediction - candidate_row.target
                        ),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )

    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_remst_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_checkpoint": source_metadata,
        "selection": {
            "population": "pre_existing_14_scene_inner_development_partition",
            "metric": "clean_blur_pooled_nmae",
            "candidate_epochs": list(range(0, int(epochs) + 1)),
            "selected_epoch": int(selected_epoch),
            "automatic_model_selection": True,
        },
        "single_backbone": {
            "image_encoder_modules": 1,
            "trainable_scope": "shared_resnet18_layer4_plus_existing_512_to_1_endpoint",
            "additional_inference_modules": 0,
            "source_correction_unchanged": True,
            "stem_through_layer3_unchanged": frozen_state_unchanged,
        },
        "anchor_state": selected_state,
    }
    torch.save(checkpoint, paths["checkpoint"])

    condition_improvements = {
        name: value["candidate_minus_source"] < 0.0
        for name, value in selected_raw_comparison["conditions"].items()
    }
    raw_relative_improvement = float(
        selected_raw_comparison["raw_pooled"]["relative_error_reduction"]
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scientific_question": (
            "Can Raw-only layer4 + existing scalar-endpoint refinement produce a "
            "material clean/blur gain without adding a second backbone or erasing "
            "the existing ReMST projective advantage?"
        ),
        "scope": {
            "single_seed_probe": True,
            "formal_holdout_access": False,
            "main_table_cohort_access": False,
            "selection_population": "inner_scene_development_only",
            "automatic_epoch_selection": True,
        },
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "available_inner_train_samples": len(internal.train),
            "used_inner_train_samples": len(train_samples),
            "inner_train_scenes": len({sample.scene_stem for sample in train_samples}),
            "available_inner_dev_samples": len(internal.dev),
            "used_inner_dev_samples": len(dev_samples),
            "inner_dev_scenes": len({sample.scene_stem for sample in dev_samples}),
            "training_views": list(TRAIN_VIEW_NAMES),
            "moderate_sigma_pixels_range_at_256": list(MODERATE_SIGMA_RANGE),
            "severe_sigma_pixels_range_at_256": list(SEVERE_SIGMA_RANGE),
        },
        "model": {
            "source_remst_checkpoint": str(Path(source_checkpoint_path).resolve()),
            "trainable_parameter_names": list(trainable_names),
            "trainable_parameters": sum(
                parameter.numel() for parameter in trainable_parameters.values()
            ),
            "total_anchor_parameters": sum(
                parameter.numel() for parameter in student.parameters()
            ),
            "additional_inference_parameters": 0,
            "source_correction_unchanged": True,
            "frozen_state_unchanged": frozen_state_unchanged,
        },
        "training": {
            "seed": int(seed),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "workers": int(workers),
            "scene_balanced_sampling_with_replacement": True,
            "optimizer": {
                "name": "AdamW",
                "layer4_learning_rate": float(layer4_learning_rate),
                "endpoint_learning_rate": float(endpoint_learning_rate),
                "weight_decay": float(weight_decay),
                "scheduler": "CosineAnnealingLR",
            },
            "loss": {
                "supervised": "mean_exact_l1_over_clean_and_two_random_blurs",
                "blur_consistency_weight": float(consistency_weight),
                "source_clean_retention_weight": float(source_retention_weight),
                "l2sp_source_weight_regularization": float(l2sp_weight),
            },
            "history": history,
            "elapsed_seconds": training_elapsed,
        },
        "selection": {
            "selected_epoch": int(selected_epoch),
            "source_epoch_included": True,
            "metric": "clean_blur_pooled_nmae",
        },
        "raw_inner_dev": {
            "comparison": selected_raw_comparison,
            "scene_cluster_bootstrap": bootstrap,
            "evaluation_elapsed_seconds": time.perf_counter()
            - raw_evaluation_started,
        },
        "projection_compatibility": {
            "skipped": bool(skip_projection_evaluation),
            "candidate": projection_candidate,
            "comparison": projection_comparison,
            "elapsed_seconds": projection_elapsed,
        },
        "diagnostic": {
            "each_raw_condition_numerically_improved": all(
                condition_improvements.values()
            ),
            "raw_condition_improvements": condition_improvements,
            "raw_pooled_relative_error_reduction": raw_relative_improvement,
            "raw_pooled_gain_at_least_one_percent": raw_relative_improvement >= 0.01,
            "projective_pooled_relative_change": (
                None
                if projection_comparison is None
                else projection_comparison["projective_pooled"]["relative_change"]
            ),
        },
        "timing": {
            "training_seconds": training_elapsed,
            "projection_evaluation_seconds": projection_elapsed,
        },
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["results"], result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument(
        "--source-correction-dev-result",
        type=Path,
        default=DEFAULT_SOURCE_CORRECTION_DEV_RESULT,
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument(
        "--correction-dev-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_DEV_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--layer4-learning-rate",
        type=float,
        default=DEFAULT_LAYER4_LEARNING_RATE,
    )
    parser.add_argument(
        "--endpoint-learning-rate",
        type=float,
        default=DEFAULT_ENDPOINT_LEARNING_RATE,
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT
    )
    parser.add_argument(
        "--source-retention-weight",
        type=float,
        default=DEFAULT_SOURCE_RETENTION_WEIGHT,
    )
    parser.add_argument("--l2sp-weight", type=float, default=DEFAULT_L2SP_WEIGHT)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--skip-projection-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_probe(
        source_checkpoint_path=args.source_checkpoint,
        source_correction_dev_result_path=args.source_correction_dev_result,
        fit_manifest_path=args.fit_manifest,
        outer_split_path=args.outer_split,
        correction_dev_manifest_path=args.correction_dev_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        layer4_learning_rate=args.layer4_learning_rate,
        endpoint_learning_rate=args.endpoint_learning_rate,
        weight_decay=args.weight_decay,
        consistency_weight=args.consistency_weight,
        source_retention_weight=args.source_retention_weight,
        l2sp_weight=args.l2sp_weight,
        bootstrap_replicates=args.bootstrap_replicates,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        skip_projection_evaluation=args.skip_projection_evaluation,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_epoch": result["selection"]["selected_epoch"],
                "raw_pooled_relative_error_reduction": result["diagnostic"][
                    "raw_pooled_relative_error_reduction"
                ],
                "projective_pooled_relative_change": result["diagnostic"][
                    "projective_pooled_relative_change"
                ],
                "results": result["artifacts"]["results"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "PROTOCOL",
    "PairedRawBlurDataset",
    "build_scene_balanced_loader",
    "compare_raw_predictions",
    "configure_refinement_scope",
    "load_raw_layer4_refined_remst",
    "raw_point_progress",
    "run_probe",
    "scene_balancing_weights",
]
