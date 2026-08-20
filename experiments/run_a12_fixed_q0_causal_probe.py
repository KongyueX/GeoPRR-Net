"""Run the fixed-q0 A12 2x2 causal probe on A11 Core-train only.

One terminal A11 Raw anchor is loaded in eval mode and used exactly once to
materialize a canonical cache of 128 relation-active sample-condition pairs.
The cache contains q0, detached Raw stride features, and the exact SARN-side
pixels/geometry.  Four fresh correction arms consume the same cache and the
same 300x16 index schedule:

* SCORT with its original eight-layer risk objective;
* SCORT with final read loss only;
* LDRT with a final-output q0-relative risk analogue;
* LDRT with final read loss only.

Only the validated 7,939-row A11 train manifest is accepted.  There is no
audit/Fold-B/formal/field input, validation selection, automatic result gate,
or intermediate checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from experiments.a11_scort import (
    A11_ARCHITECTURE,
    A11SCORTImageModel,
    SCORTCompactSARNEncoder,
    SCORTDualScaleRelationEncoder,
    SCORTOrthogonalTransport,
    SCORTProgressDecoder,
    TRANSPORT_STAGE_COUNT,
    _SCORTDifferentiableAligner,
)
from experiments.a11_scort_targets import (
    EPSILON,
    SMOOTH_L1_BETA,
    UNIFORM_REGRESSION_BASELINE,
    a11_correction_loss,
    build_a11_targets,
)
from experiments.a12_fixed_q0_probe_protocol import (
    ARM_LDRT_FINAL_ONLY,
    ARM_LDRT_FULL,
    ARM_ORDER,
    ARM_SCORT_FINAL_ONLY,
    ARM_SCORT_FULL,
    BATCH_SCHEDULE_SEED,
    CVAR_ERROR_NORMALIZATION,
    CVAR_TAIL_FRACTION,
    CVAR_WEIGHT,
    DESCRIPTIVE_REFERENCE,
    DEVELOPMENT_SCOPE,
    FORBIDDEN_DATA_NAMESPACES,
    LDRT_INITIALIZATION_SEED,
    LOSS_COMPARABILITY_NOTE,
    PIXEL_CONDITION_SEED,
    PROBE_BATCH_SIZE,
    PROBE_SAMPLES,
    PROBE_STEPS,
    PROJECTIVE_CONDITION_CYCLE,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
    REGRET_SOFTPLUS_TEMPERATURE,
    REGRET_WEIGHT,
    SCORT_INITIALIZATION_SEED,
    arm_protocol_record,
    candidate_pairs,
    fixed_batch_schedule,
    paired_active_metrics,
    schedule_presentation_counts,
    select_first_relation_active_pairs,
)
from experiments.a12_ldrt import A12LDRTCorrection, ldrt_parameter_counts
from experiments.prepare_a11_core_scene_split import (
    DEFAULT_TRAIN_MANIFEST,
    PROTOCOL as TRAIN_MANIFEST_PROTOCOL,
    load_a11_train_manifest,
)
from experiments.train_a11_scort_syncg import (
    PROTOCOL as A11_TRAINING_PROTOCOL,
    optimizer_and_scaler_state_finite_evidence,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _configure_reproducibility,
)


PROTOCOL: Final[str] = "syncg_a12_fixed_q0_causal_probe_runner_v1"
DEFAULT_TERMINAL_A11_CHECKPOINT: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/"
    "a11_scort_terminal5_seed20262020/a11_terminal.pt"
)
LEARNING_RATE: Final[float] = 3.0e-4
WEIGHT_DECAY: Final[float] = 1.0e-4
OPTIMIZER: Final[str] = "AdamW"
STEP_NOISE_SEED_OFFSET: Final[int] = 71_000_003
CANDIDATE_SCAN_BATCH_SIZE: Final[int] = 16
INTEGRITY_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-6
TERMINAL_A11_ACCESS_SCHEMA: Final[dict[str, bool]] = {
    "train_manifest_access": True,
    "audit_manifest_access": False,
    "core_audit_predictions_generated": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}


class A12ProbeRunError(ValueError):
    """A fixed-q0 probe input, state, gradient, or output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12ProbeRunError(message)


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = posterior.float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    grid = torch.linspace(
        0.0,
        1.0,
        probability.shape[1],
        device=probability.device,
        dtype=torch.float32,
    )
    mean = (probability * grid[None]).sum(dim=1)
    variance = (probability * (grid[None] - mean[:, None]).square()).sum(dim=1)
    return mean, variance


def _device_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


def _state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def module_states_bit_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _module_state_finite(module: nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in module.state_dict().values()
    )


def validate_terminal_a11_access_flags(access: Mapping[str, Any]) -> dict[str, bool]:
    """Require the complete terminal A11 train-only access evidence schema."""

    observed = dict(access)
    _require(
        set(observed) == set(TERMINAL_A11_ACCESS_SCHEMA),
        "terminal A11 access evidence keys differ from the exact seven-key schema",
    )
    _require(
        all(observed[key] is expected for key, expected in TERMINAL_A11_ACCESS_SCHEMA.items()),
        "terminal A11 access evidence values differ from train-only expectations",
    )
    return dict(TERMINAL_A11_ACCESS_SCHEMA)


def _seed_step(step: int, device: torch.device) -> None:
    seed = BATCH_SCHEDULE_SEED + STEP_NOISE_SEED_OFFSET + int(step)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class FixedProbeCache:
    """One canonical CPU cache shared by all four correction arms."""

    sample_ids: tuple[str, ...]
    scene_stems: tuple[str, ...]
    condition_names: tuple[str, ...]
    source_indices: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    target: torch.Tensor
    original_view: torch.Tensor
    sarn_view: torch.Tensor
    sarn_support_mask: torch.Tensor
    sarn_active: torch.Tensor
    raw_to_sarn_homography: torch.Tensor
    raw_posterior: torch.Tensor
    raw_mean: torch.Tensor
    raw_variance: torch.Tensor
    raw_stride8: torch.Tensor
    raw_stride16: torch.Tensor

    def validate(self) -> None:
        rows = len(self.sample_ids)
        _require(rows == PROBE_SAMPLES, "A12 canonical cache must contain 128 rows")
        _require(
            len(self.scene_stems)
            == len(self.condition_names)
            == len(self.source_indices)
            == len(self.candidate_positions)
            == rows,
            "A12 canonical cache metadata lengths differ",
        )
        _require(len(set(self.sample_ids)) == rows, "A12 cache sample IDs repeat")
        _require(len(set(self.source_indices)) == rows, "A12 source indices repeat")
        _require(
            set(self.condition_names).issubset(PROJECTIVE_CONDITION_CYCLE)
            and "clean" not in self.condition_names,
            "A12 cache contains a non-projective or clean condition",
        )
        _require(
            self.target.shape == self.raw_mean.shape == self.raw_variance.shape == (rows,),
            "A12 target/q0 moment shapes differ",
        )
        _require(
            self.raw_posterior.ndim == 2
            and self.raw_posterior.shape[0] == rows
            and self.raw_posterior.shape[1] == 128,
            "A12 q0 posterior shape differs",
        )
        _require(
            self.original_view.shape == self.sarn_view.shape
            and self.original_view.shape[0] == rows
            and self.sarn_support_mask.shape
            == (rows, 1, self.sarn_view.shape[2], self.sarn_view.shape[3]),
            "A12 cached pixel/support shapes differ",
        )
        _require(
            self.sarn_active.shape == (rows,)
            and self.sarn_active.dtype == torch.bool
            and bool(self.sarn_active.all())
            and self.raw_to_sarn_homography.shape == (rows, 3, 3),
            "A12 selected rows are not all SARN-active",
        )
        _require(
            self.raw_stride8.ndim == self.raw_stride16.ndim == 4
            and self.raw_stride8.shape[0] == self.raw_stride16.shape[0] == rows,
            "A12 cached Raw feature shapes differ",
        )
        tensors = (
            self.target,
            self.original_view,
            self.sarn_view,
            self.sarn_support_mask,
            self.raw_to_sarn_homography,
            self.raw_posterior,
            self.raw_mean,
            self.raw_variance,
            self.raw_stride8,
            self.raw_stride16,
        )
        _require(
            all(value.device.type == "cpu" for value in tensors),
            "A12 canonical cache must reside on CPU",
        )
        _require(
            all(bool(torch.isfinite(value).all()) for value in tensors),
            "A12 canonical cache contains a non-finite tensor",
        )
        _require(
            bool((self.raw_posterior >= 0.0).all())
            and bool(
                torch.allclose(
                    self.raw_posterior.sum(dim=1),
                    torch.ones(rows),
                    rtol=0.0,
                    atol=INTEGRITY_ABSOLUTE_TOLERANCE,
                )
            ),
            "A12 canonical q0 is not a probability distribution",
        )
        observed_mean, observed_variance = _posterior_moments(self.raw_posterior)
        _require(
            torch.allclose(observed_mean, self.raw_mean, rtol=1.0e-6, atol=1.0e-7)
            and torch.allclose(
                observed_variance, self.raw_variance, rtol=1.0e-6, atol=1.0e-7
            ),
            "A12 cached q0 moments fail the cross-device semantic check",
        )

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
        force_sarn_off: bool = False,
    ) -> dict[str, Any]:
        index = torch.tensor(tuple(int(value) for value in indices), dtype=torch.long)
        _require(index.ndim == 1 and index.numel() > 0, "A12 cache batch is empty")
        _require(
            int(index.min()) >= 0 and int(index.max()) < PROBE_SAMPLES,
            "A12 cache batch index is outside [0,127]",
        )

        def take(value: torch.Tensor) -> torch.Tensor:
            return _device_tensor(value.index_select(0, index), device)

        active = take(self.sarn_active)
        if force_sarn_off:
            active = torch.zeros_like(active)
        return {
            "target": take(self.target),
            "original_view": take(self.original_view),
            "sarn_view": take(self.sarn_view),
            "sarn_support_mask": take(self.sarn_support_mask),
            "sarn_active": active,
            "raw_to_sarn_homography": take(self.raw_to_sarn_homography),
            "raw_posterior": take(self.raw_posterior),
            "raw_mean": take(self.raw_mean),
            "raw_variance": take(self.raw_variance),
            "raw_features": {
                "stride8": take(self.raw_stride8),
                "stride16": take(self.raw_stride16),
            },
        }


class A12SCORTCorrection(nn.Module):
    """Fresh A11 SCORT correction with the Raw anchor supplied externally."""

    def __init__(
        self,
        *,
        relation_channels: int = 48,
        token_dim: int = 64,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = 4,
        progress_bins: int = 128,
    ) -> None:
        super().__init__()
        self.relation_channels = int(relation_channels)
        self.token_dim = int(token_dim)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.memory_grid_size = int(memory_grid_size)
        self.progress_bins = int(progress_bins)
        self.sarn_encoder = SCORTCompactSARNEncoder()
        self.stride8_aligner = _SCORTDifferentiableAligner(feature_stride=8)
        self.stride16_aligner = _SCORTDifferentiableAligner(feature_stride=16)
        self.relation_encoder = SCORTDualScaleRelationEncoder(
            relation_channels=self.relation_channels,
            token_dim=self.token_dim,
            memory_grid_size=self.memory_grid_size,
        )
        self.progress_decoder = SCORTProgressDecoder(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
            attention_heads=self.attention_heads,
            decoder_layers=self.decoder_layers,
        )
        self.orthogonal_transport = SCORTOrthogonalTransport(
            progress_bins=self.progress_bins,
            token_dim=self.token_dim,
        )

    def forward(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        q0 = raw_posterior.detach().float()
        raw_stride8 = raw_encoder_features["stride8"].detach()
        raw_stride16 = raw_encoder_features["stride16"].detach()
        raw_mean, raw_variance = _posterior_moments(q0)
        sarn_features = self.sarn_encoder(sarn_view)
        input_hw = tuple(int(value) for value in sarn_view.shape[-2:])
        alignment8 = self.stride8_aligner(
            sarn_features=sarn_features["stride8"],
            sarn_support_mask=sarn_support_mask,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        alignment16 = self.stride16_aligner(
            sarn_features=sarn_features["stride16"],
            sarn_support_mask=sarn_support_mask,
            raw_to_sarn_homography=raw_to_sarn_homography,
            input_hw=input_hw,
        )
        relation_available = (
            sarn_active
            & alignment8["transform_valid"]
            & alignment16["transform_valid"]
            & alignment8["support_valid"]
            & alignment16["support_valid"]
        )
        relation = self.relation_encoder(
            raw_stride8=raw_stride8,
            aligned_sarn_stride8=alignment8["aligned_sarn"],
            support_stride8=alignment8["common_support"],
            raw_stride16=raw_stride16,
            aligned_sarn_stride16=alignment16["aligned_sarn"],
            support_stride16=alignment16["common_support"],
            available=relation_available,
        )
        progress_tokens = self.progress_decoder(q0, relation["memory"])
        transport = self.orthogonal_transport(
            q0,
            progress_tokens,
            correction_available=relation_available,
        )
        posterior = transport["progress_posterior"]
        mean, variance = _posterior_moments(posterior)
        layer_shape = transport["layer_posteriors"].shape
        layer_means, layer_variances = _posterior_moments(
            transport["layer_posteriors"].reshape(-1, self.progress_bins)
        )
        layer_means = layer_means.reshape(layer_shape[:2])
        layer_variances = layer_variances.reshape(layer_shape[:2])
        correction_active = relation_available[:, None].expand(
            -1, TRANSPORT_STAGE_COUNT
        )
        return {
            "architecture": A11_ARCHITECTURE,
            "progress_posterior": posterior,
            "mean": mean,
            "variance": variance,
            "raw_anchor_posterior": q0,
            "raw_anchor_mean": raw_mean,
            "raw_anchor_variance": raw_variance,
            "raw_posterior": q0,
            "raw_mean": raw_mean,
            "raw_variance": raw_variance,
            "layer_posteriors": transport["layer_posteriors"],
            "layer_means": layer_means,
            "layer_variances": layer_variances,
            "transport_angles": transport["transport_angles"],
            "layer_angle_residuals": transport["layer_angle_residuals"],
            "correction_active": correction_active,
            "relation_available": relation_available,
            "sarn_active": sarn_active,
            "projective_relation": relation,
            "progress_tokens": progress_tokens,
            "stride8_alignment": alignment8,
            "stride16_alignment": alignment16,
        }


def scort_parameter_counts(model: A12SCORTCorrection) -> dict[str, int]:
    def count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    components = {
        "sarn_encoder": count(model.sarn_encoder),
        "relation_encoder": count(model.relation_encoder),
        "progress_decoder": count(model.progress_decoder),
        "orthogonal_transport": count(model.orthogonal_transport),
    }
    total = count(model)
    return {
        **components,
        "raw_anchor": 0,
        "correction": total,
        "aligners": 0,
        "total": total,
        "component_sum": sum(components.values()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "progress_bins": model.progress_bins,
    }


def _correction_construction(checkpoint: Mapping[str, Any]) -> dict[str, int]:
    construction = checkpoint.get("construction")
    _require(isinstance(construction, Mapping), "A11 construction metadata is missing")
    keys = (
        "relation_channels",
        "token_dim",
        "attention_heads",
        "decoder_layers",
        "memory_grid_size",
        "progress_bins",
    )
    result = {key: int(construction[key]) for key in keys}
    _require(result["progress_bins"] == 128, "A12 requires the terminal 128-bin q0")
    return result


def load_frozen_terminal_a11(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[A11SCORTImageModel, dict[str, Any], dict[str, int]]:
    """Strict-load the train-only terminal A11 state and freeze every tensor."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"terminal A11 checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "terminal A11 checkpoint is not a mapping")
    _require(
        checkpoint.get("protocol") == A11_TRAINING_PROTOCOL
        and checkpoint.get("system") == "a11_scort_full",
        "checkpoint is not the terminal A11 SCORT training artifact",
    )
    training = checkpoint.get("training")
    access = checkpoint.get("access_flags")
    _require(isinstance(training, Mapping), "A11 training metadata is missing")
    _require(isinstance(access, Mapping), "A11 access evidence is missing")
    _require(
        int(training.get("epochs", -1)) == 5
        and training.get("validation_manifest") is None
        and training.get("terminal_checkpoint_selection")
        == "epoch_5_no_validation_selection",
        "A11 source is not the five-epoch no-selection terminal state",
    )
    validated_access = validate_terminal_a11_access_flags(access)
    construction = _correction_construction(checkpoint)
    model = A11SCORTImageModel(imagenet_pretrained=False, **construction)
    load = model.load_state_dict(checkpoint["model_state"], strict=True)
    _require(
        not load.missing_keys and not load.unexpected_keys,
        "terminal A11 strict load differs",
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    _require(
        not model.training
        and all(not parameter.requires_grad for parameter in model.parameters()),
        "terminal A11 Raw source is not frozen in eval mode",
    )
    evidence = {
        "source": str(source),
        "strict_load": True,
        "eval_mode": True,
        "all_parameters_requires_grad_false": True,
        "training_protocol": checkpoint["protocol"],
        "training_epochs": int(training["epochs"]),
        "validation_manifest": None,
        "access_flags": validated_access,
    }
    return model, evidence, construction


def _append_row(tensors: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    tensors[key].append(value.detach().cpu().clone())


def materialize_canonical_probe_cache(
    *,
    train_manifest_path: Path,
    frozen_terminal_model: A11SCORTImageModel,
    device: torch.device,
) -> tuple[FixedProbeCache, dict[str, Any]]:
    """Select and cache the first 128 active target-blind candidate pairs."""

    samples = load_a11_train_manifest(Path(train_manifest_path).resolve())
    datasets = {
        condition: SyncGSupportGeometryMultiViewDataset(
            samples,
            training=False,
            seed=PIXEL_CONDITION_SEED,
            total_epochs=1,
            condition=condition,
        )
        for condition in PROJECTIVE_CONDITION_CYCLE
    }
    pairs = candidate_pairs(len(samples))
    active_by_source_index = [False] * len(samples)
    metadata: dict[str, list[Any]] = {
        "sample_ids": [],
        "scene_stems": [],
        "condition_names": [],
        "source_indices": [],
        "candidate_positions": [],
    }
    tensors: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "target",
            "original_view",
            "sarn_view",
            "sarn_support_mask",
            "sarn_active",
            "raw_to_sarn_homography",
            "raw_posterior",
            "raw_mean",
            "raw_variance",
            "raw_stride8",
            "raw_stride16",
        )
    }
    candidate_rows_examined = 0
    frozen_terminal_model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pairs), CANDIDATE_SCAN_BATCH_SIZE):
            chunk_pairs = pairs[offset : offset + CANDIDATE_SCAN_BATCH_SIZE]
            items = [datasets[condition][index] for index, condition in chunk_pairs]
            raw_batch = default_collate(items)
            batch = {
                key: _device_tensor(raw_batch[key], device)
                for key in (
                    "original_view",
                    "sarn_view",
                    "sarn_support_mask",
                    "sarn_active",
                    "raw_to_sarn_homography",
                )
            }
            output = frozen_terminal_model(
                batch["original_view"],
                batch["sarn_view"],
                batch["sarn_support_mask"],
                sarn_active=batch["sarn_active"],
                raw_to_sarn_homography=batch["raw_to_sarn_homography"],
            )
            availability = output["relation_available"].detach().cpu().bool()
            for row, ((source_index, condition), is_active) in enumerate(
                zip(chunk_pairs, availability.tolist(), strict=True)
            ):
                position = offset + row
                candidate_rows_examined = position + 1
                active_by_source_index[source_index] = bool(is_active)
                if not is_active or len(metadata["sample_ids"]) >= PROBE_SAMPLES:
                    continue
                # Selection above consumed only the availability boolean.  The
                # target is read for training only after the pair is accepted.
                metadata["sample_ids"].append(str(raw_batch["sample_id"][row]))
                metadata["scene_stems"].append(str(raw_batch["scene_stem"][row]))
                metadata["condition_names"].append(condition)
                metadata["source_indices"].append(int(source_index))
                metadata["candidate_positions"].append(int(position))
                for key in (
                    "target",
                    "original_view",
                    "sarn_view",
                    "sarn_support_mask",
                    "sarn_active",
                    "raw_to_sarn_homography",
                ):
                    _append_row(tensors, key, raw_batch[key][row])
                for key in ("raw_anchor_posterior", "raw_anchor_mean", "raw_anchor_variance"):
                    destination = {
                        "raw_anchor_posterior": "raw_posterior",
                        "raw_anchor_mean": "raw_mean",
                        "raw_anchor_variance": "raw_variance",
                    }[key]
                    _append_row(tensors, destination, output[key][row])
                raw_features = output["raw_encoder_features"]
                _append_row(tensors, "raw_stride8", raw_features["stride8"][row])
                _append_row(tensors, "raw_stride16", raw_features["stride16"][row])
            if len(metadata["sample_ids"]) >= PROBE_SAMPLES:
                break
    selected_pairs = select_first_relation_active_pairs(
        pairs,
        active_by_source_index,
    )
    _require(
        selected_pairs
        == tuple(zip(metadata["source_indices"], metadata["condition_names"], strict=True)),
        "A12 cached roster/conditions differ from target-blind first-active pairs",
    )
    cache = FixedProbeCache(
        sample_ids=tuple(metadata["sample_ids"]),
        scene_stems=tuple(metadata["scene_stems"]),
        condition_names=tuple(metadata["condition_names"]),
        source_indices=tuple(metadata["source_indices"]),
        candidate_positions=tuple(metadata["candidate_positions"]),
        **{key: torch.stack(values) for key, values in tensors.items()},
    )
    cache.validate()
    evidence = {
        "train_manifest": str(Path(train_manifest_path).resolve()),
        "train_manifest_protocol": TRAIN_MANIFEST_PROTOCOL,
        "source_train_rows": len(samples),
        "candidate_rows_examined": candidate_rows_examined,
        "selected_rows": PROBE_SAMPLES,
        "selection_inputs": ["seeded_candidate_order", "relation_available"],
        "selection_excluded_inputs": ["target", "q0_error", "final_error"],
        "condition_assignment": "candidate_position_modulo_three_before_availability",
        "condition_counts": dict(sorted(Counter(cache.condition_names).items())),
        "clean_rows": 0,
        "sample_ids": list(cache.sample_ids),
        "scene_stems": list(cache.scene_stems),
        "condition_names": list(cache.condition_names),
        "source_indices": list(cache.source_indices),
        "candidate_positions": list(cache.candidate_positions),
        "canonical_q0_and_raw_features_computed_once": True,
    }
    return cache, evidence


def posterior_read_per_row(
    posterior: torch.Tensor,
    mean: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """The exact A11 final posterior-read objective, exposed for both arms."""

    _require(
        posterior.ndim == 2
        and mean.shape == target.shape == (posterior.shape[0],),
        "A12 posterior-read shapes differ",
    )
    probability = posterior.float().clamp_min(EPSILON)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(EPSILON)
    bins = posterior.shape[1]
    position = target.float().clamp(0.0, 1.0) * float(bins - 1)
    lower = torch.floor(position).long()
    upper = (lower + 1).clamp(max=bins - 1)
    upper_weight = position - lower.float()
    distribution_target = torch.zeros_like(probability)
    distribution_target.scatter_add_(1, lower[:, None], (1.0 - upper_weight)[:, None])
    distribution_target.scatter_add_(1, upper[:, None], upper_weight[:, None])
    posterior_ce = -(
        distribution_target * torch.log(probability)
    ).sum(dim=1) / math.log(float(bins))
    mean_loss = F.smooth_l1_loss(
        mean.float(), target.float(), beta=SMOOTH_L1_BETA, reduction="none"
    ) / UNIFORM_REGRESSION_BASELINE
    return 0.5 * posterior_ce + 0.5 * mean_loss


def final_only_loss(
    output: Mapping[str, Any], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    posterior = output["progress_posterior"].float()
    mean = output["mean"].float()
    per_row = posterior_read_per_row(posterior, mean, target.float())
    loss = per_row.mean()
    return loss, {
        "final_read": loss,
        "final_read_per_row": per_row,
        "final_softplus_regret": loss * 0.0,
        "final_relative_cvar25": loss * 0.0,
        "active_rows": int(output["relation_available"].sum().detach().cpu()),
        "cvar_tail_count": 0,
    }


def ldrt_full_risk_loss(
    output: Mapping[str, Any], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Final-only LDRT risk analogue; no layer or angle term is invented."""

    final_read, components = final_only_loss(output, target)
    mean = output["mean"].float()
    q0_mean = output["raw_anchor_mean"].detach().float()
    active = output["relation_available"].bool()
    relative = torch.abs(mean - target.float()) - torch.abs(q0_mean - target.float())
    active_relative = relative.masked_select(active)
    if active_relative.numel() == 0:
        softplus_regret = relative.sum() * 0.0
        cvar = relative.sum() * 0.0
        tail_count = 0
    else:
        softplus_regret = F.softplus(
            active_relative / REGRET_SOFTPLUS_TEMPERATURE
        ).mean()
        positive = F.relu(active_relative) / CVAR_ERROR_NORMALIZATION
        tail_count = max(1, int(math.ceil(CVAR_TAIL_FRACTION * positive.numel())))
        cvar = torch.topk(positive, k=tail_count, largest=True).values.mean()
    total = final_read + REGRET_WEIGHT * softplus_regret + CVAR_WEIGHT * cvar
    return total, {
        **components,
        "final_softplus_regret": softplus_regret,
        "final_relative_cvar25": cvar,
        "active_rows": int(active.sum().detach().cpu()),
        "cvar_tail_count": tail_count,
    }


def correction_objective(
    arm: str,
    output: Mapping[str, Any],
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if arm == ARM_SCORT_FULL:
        targets = build_a11_targets({"target": target})
        loss, components = a11_correction_loss(output, targets)
        return loss, components
    if arm in (ARM_SCORT_FINAL_ONLY, ARM_LDRT_FINAL_ONLY):
        return final_only_loss(output, target)
    if arm == ARM_LDRT_FULL:
        return ldrt_full_risk_loss(output, target)
    raise A12ProbeRunError(f"unknown A12 arm: {arm}")


def forward_correction(
    model: nn.Module,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    return model(
        batch["raw_posterior"],
        batch["raw_features"],
        batch["sarn_view"],
        batch["sarn_support_mask"],
        sarn_active=batch["sarn_active"],
        raw_to_sarn_homography=batch["raw_to_sarn_homography"],
    )


def semantic_parameter_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    names = (
        ("sarn_encoder", "relation_encoder", "progress_decoder", "orthogonal_transport")
        if isinstance(model, A12SCORTCorrection)
        else ("sarn_encoder", "relation_encoder", "residual_decoder")
    )
    groups = {
        name: tuple(getattr(model, name).parameters())
        for name in names
    }
    _require(all(groups.values()), "A12 semantic parameter group is empty")
    return groups


def _gradient_summary(parameters: Sequence[nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    absolute_sum = (
        sum(float(value.detach().abs().double().sum().cpu()) for value in gradients)
        if finite
        else 0.0
    )
    return {
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero": absolute_sum > 0.0,
        "absolute_sum": absolute_sum,
    }


def posterior_integrity_evidence(posterior: torch.Tensor) -> dict[str, Any]:
    probability = posterior.detach().float().cpu()
    _require(probability.ndim == 2 and probability.shape[1] == 128, "posterior shape differs")
    mass_error = torch.abs(probability.sum(dim=1) - 1.0)
    cdf = probability.cumsum(dim=1)
    cdf_steps = cdf[:, 1:] - cdf[:, :-1]
    negative_rows = (probability < -INTEGRITY_ABSOLUTE_TOLERANCE).any(dim=1)
    cdf_nonmonotone_rows = (cdf_steps < -INTEGRITY_ABSOLUTE_TOLERANCE).any(dim=1)
    cdf_terminal_error = torch.abs(cdf[:, -1] - 1.0)
    return {
        "rows": int(probability.shape[0]),
        "negative_probability_row_count": int(negative_rows.sum()),
        "mass_violation_row_count": int((mass_error > INTEGRITY_ABSOLUTE_TOLERANCE).sum()),
        "maximum_absolute_mass_error": float(mass_error.max()),
        "cdf_nonmonotone_row_count": int(cdf_nonmonotone_rows.sum()),
        "cdf_terminal_violation_row_count": int(
            (cdf_terminal_error > INTEGRITY_ABSOLUTE_TOLERANCE).sum()
        ),
        "maximum_absolute_cdf_terminal_error": float(cdf_terminal_error.max()),
        "absolute_tolerance": INTEGRITY_ABSOLUTE_TOLERANCE,
    }


def _evaluate_model(
    model: nn.Module,
    cache: FixedProbeCache,
    *,
    device: torch.device,
    force_sarn_off: bool = False,
) -> dict[str, torch.Tensor]:
    model.eval()
    values: dict[str, list[torch.Tensor]] = {
        "progress_posterior": [],
        "mean": [],
        "variance": [],
        "raw_anchor_posterior": [],
        "raw_anchor_mean": [],
        "raw_anchor_variance": [],
        "relation_available": [],
    }
    with torch.inference_mode():
        for offset in range(0, PROBE_SAMPLES, PROBE_BATCH_SIZE):
            indices = tuple(range(offset, min(offset + PROBE_BATCH_SIZE, PROBE_SAMPLES)))
            output = forward_correction(
                model,
                cache.batch(indices, device=device, force_sarn_off=force_sarn_off),
            )
            for name in values:
                values[name].append(output[name].detach().cpu())
    return {name: torch.cat(parts, dim=0) for name, parts in values.items()}


def _outputs_bit_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _factory_for_arm(
    arm: str, construction: Mapping[str, int]
) -> Callable[[], nn.Module]:
    if arm in (ARM_SCORT_FULL, ARM_SCORT_FINAL_ONLY):
        return lambda: A12SCORTCorrection(**construction)
    if arm in (ARM_LDRT_FULL, ARM_LDRT_FINAL_ONLY):
        return lambda: A12LDRTCorrection(**construction)
    raise A12ProbeRunError(f"unknown A12 arm: {arm}")


def train_probe_arm(
    *,
    arm: str,
    model: nn.Module,
    model_factory: Callable[[], nn.Module],
    cache: FixedProbeCache,
    schedule: Sequence[Sequence[int]],
    device: torch.device,
    record_step_parameter_updates: bool = False,
) -> dict[str, Any]:
    """Train one correction arm on a caller-supplied, already fixed schedule."""

    batches = tuple(tuple(int(index) for index in batch) for batch in schedule)
    _require(bool(batches), "A12 arm schedule is empty")
    model.to(device)
    initial_state = _state_cpu(model)
    initial_output = _evaluate_model(model, cache, device=device)
    _require(
        bool(initial_output["relation_available"].all()),
        f"{arm}: selected cache is not relation-active",
    )
    initial_identity = all(
        torch.equal(initial_output[name], initial_output[reference])
        for name, reference in (
            ("progress_posterior", "raw_anchor_posterior"),
            ("mean", "raw_anchor_mean"),
            ("variance", "raw_anchor_variance"),
        )
    )
    _require(initial_identity, f"{arm}: fresh active correction is not exact q0")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    semantic = semantic_parameter_groups(model)
    gradient_nonzero_seen = {name: False for name in semantic}
    gradient_finite_every_step = {name: True for name in semantic}
    optimizer_state_finite_every_step = True
    history: list[dict[str, Any]] = []
    schedule_trace: list[list[int]] = []
    for step, indices in enumerate(batches, start=1):
        _seed_step(step, device)
        model.train()
        batch = cache.batch(indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        parameters_before = (
            {
                name: tuple(parameter.detach().clone() for parameter in parameters)
                for name, parameters in semantic.items()
            }
            if record_step_parameter_updates
            else {}
        )
        output = forward_correction(model, batch)
        _require(
            bool(output["relation_available"].all()),
            f"{arm}: relation availability changed during the fixed probe",
        )
        _require(
            torch.equal(output["raw_anchor_posterior"], batch["raw_posterior"])
            and torch.equal(output["raw_anchor_mean"], batch["raw_mean"])
            and torch.equal(output["raw_anchor_variance"], batch["raw_variance"]),
            f"{arm}: correction forward changed canonical q0",
        )
        loss, components = correction_objective(arm, output, batch["target"])
        _require(bool(torch.isfinite(loss)), f"{arm}: loss is non-finite")
        loss.backward()
        step_gradient = {
            name: _gradient_summary(parameters) for name, parameters in semantic.items()
        }
        _require(
            all(summary["finite"] for summary in step_gradient.values()),
            f"{arm}: a semantic gradient is absent or non-finite",
        )
        for name, summary in step_gradient.items():
            gradient_nonzero_seen[name] = gradient_nonzero_seen[name] or bool(
                summary["nonzero"]
            )
            gradient_finite_every_step[name] = (
                gradient_finite_every_step[name] and bool(summary["finite"])
            )
        optimizer.step()
        parameter_updated = (
            {
                name: any(
                    not torch.equal(previous, current.detach())
                    for previous, current in zip(
                        parameters_before[name], parameters, strict=True
                    )
                )
                for name, parameters in semantic.items()
            }
            if record_step_parameter_updates
            else {}
        )
        state_evidence = optimizer_and_scaler_state_finite_evidence(optimizer, None)
        _require(bool(state_evidence["finite"]), f"{arm}: AdamW state is non-finite")
        optimizer_state_finite_every_step = (
            optimizer_state_finite_every_step and bool(state_evidence["finite"])
        )
        _require(_module_state_finite(model), f"{arm}: model state is non-finite")
        component_values = {
            name: float(value.detach().cpu())
            for name, value in components.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        history.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "components": component_values,
                "gradient": step_gradient,
                "parameter_updated": parameter_updated,
            }
        )
        schedule_trace.append(list(indices))
    _require(
        all(gradient_finite_every_step.values())
        and all(gradient_nonzero_seen.values()),
        f"{arm}: not every semantic group received a finite nonzero task gradient",
    )
    terminal_state = _state_cpu(model)
    updated_state_tensor_count = sum(
        not torch.equal(initial_state[name], terminal_state[name]) for name in initial_state
    )
    _require(updated_state_tensor_count > 0, f"{arm}: no correction state changed")
    terminal_output = _evaluate_model(model, cache, device=device)
    integrity = posterior_integrity_evidence(terminal_output["progress_posterior"])
    metrics = paired_active_metrics(
        final_mean=terminal_output["mean"].tolist(),
        q0_mean=terminal_output["raw_anchor_mean"].tolist(),
        target=cache.target.tolist(),
        active=terminal_output["relation_available"].tolist(),
    )
    fallback_output = _evaluate_model(model, cache, device=device, force_sarn_off=True)
    fallback_exact = (
        not bool(fallback_output["relation_available"].any())
        and torch.equal(
            fallback_output["progress_posterior"], fallback_output["raw_anchor_posterior"]
        )
        and torch.equal(fallback_output["mean"], fallback_output["raw_anchor_mean"])
        and torch.equal(
            fallback_output["variance"], fallback_output["raw_anchor_variance"]
        )
    )
    _require(fallback_exact, f"{arm}: SARN-off fallback is not exact q0")
    fresh = model_factory().to(device)
    load = fresh.load_state_dict(terminal_state, strict=True)
    fresh_output = _evaluate_model(fresh, cache, device=device)
    fresh_exact = (
        not load.missing_keys
        and not load.unexpected_keys
        and _outputs_bit_exact(terminal_output, fresh_output)
    )
    _require(fresh_exact, f"{arm}: fresh strict-load output differs")
    state_evidence = optimizer_and_scaler_state_finite_evidence(optimizer, None)
    return {
        "arm": arm,
        "steps": len(batches),
        "samples_per_step": len(batches[0]),
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clipping": None,
        "mixed_precision": False,
        "initial_active_exact_q0": initial_identity,
        "metrics": metrics,
        "posterior_integrity": integrity,
        "sarn_off_fallback_exact_q0": fallback_exact,
        "gradient_nonzero_seen": gradient_nonzero_seen,
        "gradient_finite_every_step": gradient_finite_every_step,
        "optimizer_state_finite_every_step": optimizer_state_finite_every_step,
        "terminal_optimizer_state": state_evidence,
        "updated_state_tensor_count": updated_state_tensor_count,
        "fresh_strict_load": True,
        "fresh_output_bit_exact": fresh_exact,
        "schedule_trace": schedule_trace,
        "history": history,
        "terminal_predictions": {
            "sample_ids": list(cache.sample_ids),
            "condition_names": list(cache.condition_names),
            "target": cache.target.tolist(),
            "q0_mean": terminal_output["raw_anchor_mean"].tolist(),
            "final_mean": terminal_output["mean"].tolist(),
            "relation_available": terminal_output["relation_available"].tolist(),
        },
        "model_state": terminal_state,
    }


def _arm_parameter_counts(model: nn.Module) -> dict[str, int]:
    if isinstance(model, A12SCORTCorrection):
        return scort_parameter_counts(model)
    _require(isinstance(model, A12LDRTCorrection), "unknown A12 correction model")
    return ldrt_parameter_counts(model)


def run_a12_fixed_q0_causal_probe(
    *,
    train_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A12 output already exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(BATCH_SCHEDULE_SEED, device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    terminal_model, terminal_evidence, construction = load_frozen_terminal_a11(
        terminal_a11_checkpoint_path, device=device
    )
    anchor_state_before = _state_cpu(terminal_model)
    cache, selection_evidence = materialize_canonical_probe_cache(
        train_manifest_path=train_manifest_path,
        frozen_terminal_model=terminal_model,
        device=device,
    )
    anchor_state_after_cache = _state_cpu(terminal_model)
    _require(
        module_states_bit_exact(anchor_state_before, anchor_state_after_cache),
        "terminal A11 anchor changed while creating canonical q0",
    )
    del terminal_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(SCORT_INITIALIZATION_SEED)
    scort_base = A12SCORTCorrection(**construction)
    scort_full = copy.deepcopy(scort_base)
    scort_final = copy.deepcopy(scort_base)
    scort_pair_same_initialization = module_states_bit_exact(
        _state_cpu(scort_full), _state_cpu(scort_final)
    )
    _require(scort_pair_same_initialization, "SCORT pair initialization differs")

    torch.manual_seed(LDRT_INITIALIZATION_SEED)
    ldrt_base = A12LDRTCorrection(**construction)
    ldrt_full = copy.deepcopy(ldrt_base)
    ldrt_final = copy.deepcopy(ldrt_base)
    ldrt_pair_same_initialization = module_states_bit_exact(
        _state_cpu(ldrt_full), _state_cpu(ldrt_final)
    )
    _require(ldrt_pair_same_initialization, "LDRT pair initialization differs")
    del scort_base, ldrt_base

    models: dict[str, nn.Module] = {
        ARM_SCORT_FULL: scort_full,
        ARM_SCORT_FINAL_ONLY: scort_final,
        ARM_LDRT_FULL: ldrt_full,
        ARM_LDRT_FINAL_ONLY: ldrt_final,
    }
    schedule = fixed_batch_schedule()
    schedule_counts = schedule_presentation_counts(schedule)
    arm_results: dict[str, Any] = {}
    parameter_counts: dict[str, Any] = {}
    for arm in ARM_ORDER:
        model = models[arm]
        parameter_counts[arm] = _arm_parameter_counts(model)
        arm_results[arm] = train_probe_arm(
            arm=arm,
            model=model,
            model_factory=_factory_for_arm(arm, construction),
            cache=cache,
            schedule=schedule,
            device=device,
        )
        del models[arm]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_trace = [list(batch) for batch in schedule]
    _require(
        all(result["schedule_trace"] == expected_trace for result in arm_results.values()),
        "four A12 arms did not consume the same sample/condition order",
    )
    access_flags = {
        "train_manifest_access": True,
        "terminal_a11_train_checkpoint_access": True,
        "core_audit_manifest_access": False,
        "core_audit_predictions_generated": False,
        "fold_a_content_access": False,
        "fold_b_content_access": False,
        "formal_holdout_content_access": False,
        "field_photo_content_access": False,
    }
    artifact = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "development_scope": DEVELOPMENT_SCOPE,
        "arm_protocol": arm_protocol_record(),
        "loss_comparability_note": LOSS_COMPARABILITY_NOTE,
        "cross_architecture_full_loss_interaction_is_a_pure_causal_effect": False,
        "primary_within_architecture_contrasts": [
            "scort_full_minus_scort_final_only",
            "ldrt_full_minus_ldrt_final_only",
        ],
        "descriptive_reference": dict(DESCRIPTIVE_REFERENCE),
        "automatic_execution_or_advancement_control": False,
        "terminal_a11": terminal_evidence,
        "canonical_cache": selection_evidence,
        "canonical_q0_state_unchanged_during_cache": True,
        "pair_initialization_evidence": {
            "scort_full_and_final_only_bit_exact": scort_pair_same_initialization,
            "ldrt_full_and_final_only_bit_exact": ldrt_pair_same_initialization,
        },
        "training": {
            "steps_per_arm": PROBE_STEPS,
            "batch_size": PROBE_BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "schedule": [list(batch) for batch in schedule],
            "condition_schedule": [
                [cache.condition_names[index] for index in batch] for batch in schedule
            ],
            "presentation_counts": schedule_counts,
            "shared_canonical_tensor_values": True,
            "shared_sample_condition_order": True,
            "clean_in_training": False,
            "sarn_off_in_training": False,
        },
        "construction": dict(construction),
        "parameter_counts": parameter_counts,
        "arms": arm_results,
        "access_flags": access_flags,
        "forbidden_data_namespaces": list(FORBIDDEN_DATA_NAMESPACES),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(artifact, handle)
    return {
        "status": "complete",
        "output": str(output),
        "protocol": PROTOCOL,
        "selected_rows": PROBE_SAMPLES,
        "steps_per_arm": PROBE_STEPS,
        "arm_metrics": {
            arm: arm_results[arm]["metrics"] for arm in ARM_ORDER
        },
        "access_flags": access_flags,
        "automatic_gate_used": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--terminal-a11-checkpoint",
        type=Path,
        default=DEFAULT_TERMINAL_A11_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_a12_fixed_q0_causal_probe(
        train_manifest_path=args.train_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        output_path=args.output,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A12ProbeRunError",
    "A12SCORTCorrection",
    "CANDIDATE_SCAN_BATCH_SIZE",
    "DEFAULT_TERMINAL_A11_CHECKPOINT",
    "FixedProbeCache",
    "INTEGRITY_ABSOLUTE_TOLERANCE",
    "LEARNING_RATE",
    "OPTIMIZER",
    "PROTOCOL",
    "WEIGHT_DECAY",
    "build_argument_parser",
    "correction_objective",
    "final_only_loss",
    "forward_correction",
    "ldrt_full_risk_loss",
    "load_frozen_terminal_a11",
    "materialize_canonical_probe_cache",
    "module_states_bit_exact",
    "posterior_integrity_evidence",
    "posterior_read_per_row",
    "run_a12_fixed_q0_causal_probe",
    "scort_parameter_counts",
    "semantic_parameter_groups",
    "train_probe_arm",
    "validate_terminal_a11_access_flags",
]
