"""Fit a final-representation relative moment transport for ReMST-ResNet18.

The Raw and SARN observations are still encoded by one shared, frozen
ResNet-18.  This stage reuses their already-computed 512-D final
representations and the established target-free geometry features to predict a
small, bounded first-moment residual on top of the existing ReMST correction.
When the projective relation is unavailable, the wrapped correction is
returned exactly.

The stage can use either the nested 6,616-sample correction roster or the
complete original 14,442-sample fit roster.  The expensive frozen features are
cached once; the compact transport head is then optimized with exact L1, a
tail term, and weak cross-condition consistency.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.a15_fteb import (
    BRIDGE_LAYERS,
    GEOMETRY_FEATURES,
    frozen_twin_endpoint_forward,
)
from experiments.a15_fteb_inner_scene_probe_protocol import PROJECTIVE_CONDITIONS
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    load_a13_correction_train_manifest,
)
from experiments.remst_resnet18 import (
    REMST_RESNET18_ARCHITECTURE,
    RESNET18_REPRESENTATION_FEATURES,
    remst_resnet18_publication_identity,
)
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.train_a15_2_fteb_correction_only import (
    A152ThreeConditionTrainDataset,
    CONDITIONS_PER_PHYSICAL,
    SAMPLE_ORDER_SEED,
    build_a15_2_epoch_loader,
    build_a15_2_train_dataset,
    configure_a15_2_reproducibility,
)
from experiments.resnet18_direct_progress import load_syncg_samples
from experiments.train_raw_context_residual_correction_adaptation import (
    load_context_correction_adapted_remst,
)


PROTOCOL: Final[str] = "remst_resnet18_relative_context_moment_transport_v1"
ARCHITECTURE: Final[str] = (
    "ReMST-ResNet18-Single-Backbone-Relative-Context-Moment-Transport"
)
DEFAULT_SOURCE_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_context_correction_adapt/"
    "seed_20262020/terminal.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_relative_context_transport/"
    "seed_20262020/terminal.pt"
)
DEFAULT_CACHE: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_relative_context_transport/"
    "seed_20262020/correction_fit_cache.pt"
)
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_EPOCHS: Final[int] = 20
DEFAULT_BATCH_SIZE: Final[int] = 256
DEFAULT_LATENT_FEATURES: Final[int] = 64
DEFAULT_HIDDEN_FEATURES: Final[int] = 128
DEFAULT_MAX_PROGRESS_SHIFT: Final[float] = 0.05
DEFAULT_LEARNING_RATE: Final[float] = 1.0e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1.0e-4
DEFAULT_TAIL_WEIGHT: Final[float] = 0.10
DEFAULT_CONSISTENCY_WEIGHT: Final[float] = 0.02
DEFAULT_RESIDUAL_L2_WEIGHT: Final[float] = 1.0e-4
DEFAULT_CONDITION_WEIGHTS: Final[tuple[float, float, float]] = (1.0, 1.0, 1.0)
MOMENT_SOLVER_STEPS: Final[int] = 20
MOMENT_SOLVER_LIMIT: Final[float] = 2048.0


class RelativeContextTransportError(ValueError):
    """The source model, cache, or relative transport configuration differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RelativeContextTransportError(message)


def _posterior_moments(
    posterior: torch.Tensor, grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = (posterior * grid).sum(dim=1)
    variance = (posterior * (grid - mean[:, None]).square()).sum(dim=1)
    return mean, variance


class RelativeContextMomentHead(nn.Module):
    """Shared-projection Raw/SARN context relation to one bounded moment shift."""

    def __init__(
        self,
        *,
        latent_features: int = DEFAULT_LATENT_FEATURES,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        max_progress_shift: float = DEFAULT_MAX_PROGRESS_SHIFT,
    ) -> None:
        super().__init__()
        _require(latent_features >= 8, "relative context latent width is too small")
        _require(hidden_features >= 8, "relative context hidden width is too small")
        _require(
            math.isfinite(float(max_progress_shift))
            and 0.0 < float(max_progress_shift) <= 0.25,
            "relative context maximum shift is invalid",
        )
        self.latent_features = int(latent_features)
        self.hidden_features = int(hidden_features)
        self.max_progress_shift = float(max_progress_shift)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(RESNET18_REPRESENTATION_FEATURES),
            nn.Linear(RESNET18_REPRESENTATION_FEATURES, self.latent_features),
            nn.GELU(),
        )
        self.scalar_features = 5
        relation_features = (
            5 * self.latent_features + GEOMETRY_FEATURES + self.scalar_features
        )
        self.relation_features = int(relation_features)
        self.residual_network = nn.Sequential(
            nn.LayerNorm(self.relation_features),
            nn.Linear(self.relation_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, 1),
        )
        output = self.residual_network[-1]
        _require(isinstance(output, nn.Linear), "relative context output differs")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        raw_representation: torch.Tensor,
        sarn_representation: torch.Tensor,
        geometry_features: torch.Tensor,
        raw_mean: torch.Tensor,
        sarn_mean: torch.Tensor,
        base_mean: torch.Tensor,
    ) -> torch.Tensor:
        batch = raw_representation.shape[0]
        _require(
            raw_representation.shape
            == sarn_representation.shape
            == (batch, RESNET18_REPRESENTATION_FEATURES),
            "relative context representation shape differs",
        )
        _require(
            geometry_features.shape == (batch, GEOMETRY_FEATURES)
            and raw_mean.shape == sarn_mean.shape == base_mean.shape == (batch,),
            "relative context geometry/scalar shape differs",
        )
        raw = self.shared_projection(raw_representation.float())
        sarn = self.shared_projection(sarn_representation.float())
        scalars = torch.stack(
            (
                raw_mean.float(),
                sarn_mean.float(),
                base_mean.float(),
                sarn_mean.float() - raw_mean.float(),
                base_mean.float() - sarn_mean.float(),
            ),
            dim=1,
        )
        relation = torch.cat(
            (
                raw,
                sarn,
                sarn - raw,
                torch.abs(sarn - raw),
                raw * sarn,
                geometry_features.float(),
                scalars,
            ),
            dim=1,
        )
        _require(
            relation.shape == (batch, self.relation_features),
            "relative context relation feature count differs",
        )
        logit = self.residual_network(relation).squeeze(1)
        return self.max_progress_shift * torch.tanh(logit)


class RelativeContextMomentTransport(nn.Module):
    """Wrap one ReMST correction with a moment-exact global relation residual."""

    def __init__(
        self,
        base_correction: nn.Module,
        *,
        progress_bins: int,
        latent_features: int = DEFAULT_LATENT_FEATURES,
        hidden_features: int = DEFAULT_HIDDEN_FEATURES,
        max_progress_shift: float = DEFAULT_MAX_PROGRESS_SHIFT,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "relative context progress grid is too small")
        self.base_correction = base_correction
        self.progress_bins = int(progress_bins)
        self.context_head = RelativeContextMomentHead(
            latent_features=int(latent_features),
            hidden_features=int(hidden_features),
            max_progress_shift=float(max_progress_shift),
        )
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, self.progress_bins, dtype=torch.float32),
        )

    def _transport(
        self,
        posterior: torch.Tensor,
        target_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self.progress_grid.float()[None]
        log_density = torch.log(posterior.float().clamp_min(1.0e-12))
        tilt = torch.zeros_like(target_mean.float())[:, None]
        for _ in range(MOMENT_SOLVER_STEPS):
            proposed = torch.softmax(log_density + tilt * grid, dim=1)
            mean, variance = _posterior_moments(proposed, grid)
            tilt = (
                tilt
                + (target_mean.float() - mean)[:, None]
                / variance[:, None].clamp_min(1.0e-10)
            ).clamp(-MOMENT_SOLVER_LIMIT, MOMENT_SOLVER_LIMIT)
        proposed = torch.softmax(log_density + tilt * grid, dim=1)
        _, variance = _posterior_moments(proposed, grid)
        return proposed, variance

    def forward(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_posterior: torch.Tensor,
        sarn_encoder_features: Mapping[str, torch.Tensor],
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        base = self.base_correction(
            raw_posterior,
            raw_encoder_features,
            sarn_posterior,
            sarn_encoder_features,
            sarn_support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        raw_representation = raw_encoder_features.get("representation")
        sarn_representation = sarn_encoder_features.get("representation")
        geometry = base.get("geometry_token")
        _require(
            isinstance(raw_representation, torch.Tensor)
            and isinstance(sarn_representation, torch.Tensor)
            and isinstance(geometry, Mapping)
            and isinstance(geometry.get("features"), torch.Tensor),
            "relative context inputs are unavailable",
        )
        available = base["relation_available"].bool()
        base_mean = base["mean"].float()
        shift = self.context_head(
            raw_representation,
            sarn_representation,
            geometry["features"],
            base["raw_anchor_mean"],
            base["sarn_endpoint_mean"],
            base_mean,
        )
        applied_shift = torch.where(available, shift, torch.zeros_like(shift))
        epsilon = torch.finfo(torch.float32).eps
        target_mean = (base_mean + applied_shift).clamp(epsilon, 1.0 - epsilon)
        proposed, proposed_variance = self._transport(
            base["progress_posterior"], target_mean
        )
        base_posterior = base["progress_posterior"].float()
        changed = available & (applied_shift != 0.0)
        zero_shift = applied_shift == 0.0
        posterior_candidate = torch.where(
            zero_shift[:, None],
            base_posterior + proposed - proposed.detach(),
            proposed,
        )
        mean_candidate = torch.where(
            zero_shift,
            base_mean + target_mean - target_mean.detach(),
            target_mean,
        )
        variance_candidate = torch.where(
            zero_shift,
            base["variance"].float()
            + proposed_variance
            - proposed_variance.detach(),
            proposed_variance,
        )
        posterior = torch.where(
            available[:, None], posterior_candidate, base_posterior
        )
        mean = torch.where(available, mean_candidate, base_mean)
        variance = torch.where(
            available, variance_candidate, base["variance"].float()
        )
        base_cdf = base_posterior.cumsum(dim=1)
        final_cdf = posterior.cumsum(dim=1)
        path_times = torch.arange(
            1,
            BRIDGE_LAYERS + 1,
            dtype=torch.float32,
            device=posterior.device,
        ) / float(BRIDGE_LAYERS)
        relative_context_layer_cdfs = base_cdf[:, None] + path_times[
            None, :, None
        ] * (final_cdf[:, None] - base_cdf[:, None])
        result = dict(base)
        result.update(
            {
                "architecture": ARCHITECTURE,
                "progress_posterior": posterior,
                "progress_cdf": final_cdf,
                "mean": mean,
                "variance": variance,
                "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
                "relative_context_shift": applied_shift,
                "relative_context_target_mean": target_mean,
                "relative_context_active": changed,
                "relative_context_base_mean": base_mean,
                "relative_context_base_posterior": base_posterior,
                "relative_context_layer_cdfs": relative_context_layer_cdfs,
                "publication_model": {
                    **remst_resnet18_publication_identity(),
                    "short_name": "ReMST-ResNet18-RCMT",
                    "display_name": "ReMST-ResNet18 Relative Context Moment Transport",
                },
            }
        )
        physical = dict(base.get("physical_outputs", {}))
        physical["progress_mean"] = mean
        physical["progress_variance"] = variance
        result["physical_outputs"] = physical
        return result


def _device_tensor(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    return value


def _collect_fit_cache(
    anchor: nn.Module,
    correction: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    workers: int,
    sample_order_seed: int,
) -> dict[str, Any]:
    loader = build_a15_2_epoch_loader(
        dataset,
        epoch=0,
        workers=int(workers),
        cuda=device.type == "cuda",
        sample_order_seed=int(sample_order_seed),
    )
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    source_indices: list[int] = []
    started = time.perf_counter()
    anchor.eval()
    correction.eval()
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode():
        for step, raw_batch in enumerate(loader, start=1):
            original = _device_tensor(raw_batch["original_view"], device)
            sarn = _device_tensor(raw_batch["sarn_view"], device)
            support = _device_tensor(raw_batch["sarn_support_mask"], device)
            active = _device_tensor(raw_batch["sarn_active"], device).bool()
            homography = _device_tensor(raw_batch["raw_to_sarn_homography"], device)
            target = _device_tensor(raw_batch["target"], device).float()
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                output = forward_a15_correction(
                    correction,
                    {
                        "raw_posterior": endpoints["raw_posterior"],
                        "sarn_posterior": endpoints["sarn_posterior"],
                        "raw_mean": endpoints["raw_mean"],
                        "sarn_mean": endpoints["sarn_mean"],
                        "raw_features": endpoints["raw_features"],
                        "sarn_features": endpoints["sarn_features"],
                        "sarn_support_mask": support,
                        "sarn_active": active,
                        "raw_to_sarn_homography": homography,
                    },
                    endpoint_null=False,
                )
            row_count = int(target.numel())
            _require(
                row_count % CONDITIONS_PER_PHYSICAL == 0,
                "relative context cache batch is not triplet aligned",
            )
            physical = row_count // CONDITIONS_PER_PHYSICAL
            condition_names = tuple(str(value) for value in raw_batch["condition_name"])
            _require(
                condition_names == tuple(PROJECTIVE_CONDITIONS) * physical,
                "relative context cache condition order differs",
            )
            tensors = {
                "raw_representation": endpoints["raw_features"]["representation"],
                "sarn_representation": endpoints["sarn_features"]["representation"],
                "geometry_features": output["geometry_token"]["features"],
                "raw_mean": output["raw_anchor_mean"],
                "sarn_mean": output["sarn_endpoint_mean"],
                "base_mean": output["mean"],
                "target": target,
                "active": output["relation_available"],
            }
            for name, tensor in tensors.items():
                values[name].append(tensor.detach().float().cpu())
            batch_sources = [int(value) for value in raw_batch["source_index"]]
            for offset in range(0, row_count, CONDITIONS_PER_PHYSICAL):
                triplet = batch_sources[offset : offset + CONDITIONS_PER_PHYSICAL]
                _require(len(set(triplet)) == 1, "relative context source triplet differs")
                source_indices.append(triplet[0])
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "relative_context_cache",
                            "steps": step,
                            "physical_samples": len(source_indices),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    cache = {name: torch.cat(parts, dim=0) for name, parts in values.items()}
    rows = int(cache["target"].numel())
    _require(
        rows == len(dataset) * CONDITIONS_PER_PHYSICAL
        and len(source_indices) == len(dataset)
        and len(set(source_indices)) == len(dataset),
        "relative context cache population differs",
    )
    physical = len(dataset)
    for name, tensor in tuple(cache.items()):
        cache[name] = tensor.reshape(physical, CONDITIONS_PER_PHYSICAL, *tensor.shape[1:])
    targets = cache["target"]
    _require(
        torch.equal(targets[:, 0], targets[:, 1])
        and torch.equal(targets[:, 0], targets[:, 2]),
        "relative context targets differ within a physical triplet",
    )
    cache["target"] = targets[:, 0]
    cache["source_indices"] = tuple(source_indices)
    cache["elapsed_seconds"] = time.perf_counter() - started
    return cache


def _train_epoch(
    head: RelativeContextMomentHead,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    condition_weights: Sequence[float],
    tail_weight: float,
    consistency_weight: float,
    residual_l2_weight: float,
) -> dict[str, float]:
    head.train(True)
    totals = defaultdict(float)
    physical_rows = 0
    for batch in loader:
        (
            raw_representation,
            sarn_representation,
            geometry_features,
            raw_mean,
            sarn_mean,
            base_mean,
            target,
            active,
        ) = tuple(value.to(device, non_blocking=device.type == "cuda") for value in batch)
        optimizer.zero_grad(set_to_none=True)
        physical, conditions, features = raw_representation.shape
        _require(
            conditions == CONDITIONS_PER_PHYSICAL
            and features == RESNET18_REPRESENTATION_FEATURES,
            "relative context training batch differs",
        )
        shift = head(
            raw_representation.flatten(0, 1),
            sarn_representation.flatten(0, 1),
            geometry_features.flatten(0, 1),
            raw_mean.flatten(),
            sarn_mean.flatten(),
            base_mean.flatten(),
        ).reshape(physical, conditions)
        active_mask = active.bool()
        applied_shift = torch.where(active_mask, shift, torch.zeros_like(shift))
        prediction = (base_mean + applied_shift).clamp(0.0, 1.0)
        errors = torch.abs(prediction - target[:, None])
        active_errors = errors[active_mask]
        _require(active_errors.numel() >= 1, "relative context batch has no active rows")
        supervised = active_errors.mean()
        weight_tensor = torch.as_tensor(
            condition_weights, dtype=errors.dtype, device=errors.device
        )[None].expand(physical, -1)
        active_weights = weight_tensor[active_mask]
        weighted_supervised = (
            (active_errors * active_weights).sum() / active_weights.sum()
        )
        tail_count = max(1, math.ceil(0.25 * int(active_errors.numel())))
        tail = torch.topk(active_errors, k=tail_count, largest=True).values.mean()
        complete = active_mask.all(dim=1)
        if bool(complete.any()):
            selected = prediction[complete]
            consistency = (
                torch.abs(selected[:, 0] - selected[:, 1]).mean()
                + torch.abs(selected[:, 0] - selected[:, 2]).mean()
                + torch.abs(selected[:, 1] - selected[:, 2]).mean()
            ) / 3.0
        else:
            consistency = prediction.sum() * 0.0
        residual_l2 = applied_shift[active_mask].square().mean()
        loss = (
            weighted_supervised
            + float(tail_weight) * tail
            + float(consistency_weight) * consistency
            + float(residual_l2_weight) * residual_l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        optimizer.step()
        physical_rows += physical
        for name, value in (
            ("total", loss),
            ("exact_l1", supervised),
            ("weighted_exact_l1", weighted_supervised),
            ("tail_l1", tail),
            ("consistency", consistency),
            ("residual_l2", residual_l2),
            ("mean_absolute_shift", applied_shift[active_mask].abs().mean()),
        ):
            totals[name] += float(value.detach().cpu()) * physical
    _require(physical_rows >= 1, "relative context epoch is empty")
    return {name: value / physical_rows for name, value in sorted(totals.items())}


def train_relative_context_transport(
    *,
    source_checkpoint_path: Path,
    correction_train_manifest_path: Path,
    fit_manifest_path: Path | None,
    output_path: Path,
    cache_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    workers: int = DEFAULT_WORKERS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    latent_features: int = DEFAULT_LATENT_FEATURES,
    hidden_features: int = DEFAULT_HIDDEN_FEATURES,
    max_progress_shift: float = DEFAULT_MAX_PROGRESS_SHIFT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    condition_weights: Sequence[float] = DEFAULT_CONDITION_WEIGHTS,
    tail_weight: float = DEFAULT_TAIL_WEIGHT,
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT,
    residual_l2_weight: float = DEFAULT_RESIDUAL_L2_WEIGHT,
    sample_order_seed: int = SAMPLE_ORDER_SEED + 70_000,
) -> dict[str, Any]:
    condition_weights = tuple(float(value) for value in condition_weights)
    _require(
        workers >= 0 and epochs >= 1 and batch_size >= 1,
        "relative context training sizes are invalid",
    )
    _require(
        learning_rate > 0.0
        and weight_decay >= 0.0
        and tail_weight >= 0.0
        and consistency_weight >= 0.0
        and residual_l2_weight >= 0.0,
        "relative context optimizer/loss configuration is invalid",
    )
    _require(
        len(condition_weights) == CONDITIONS_PER_PHYSICAL
        and all(math.isfinite(value) and value > 0.0 for value in condition_weights),
        "relative context condition weights are invalid",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"relative context output exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a15_2_reproducibility(device, seed=int(seed))
    if fit_manifest_path is None:
        population_manifest = Path(correction_train_manifest_path).resolve()
        samples = tuple(load_a13_correction_train_manifest(population_manifest))
        _require(
            len(samples) == 6_616
            and len({sample.scene_stem for sample in samples}) == 60,
            "relative context correction population differs",
        )
        dataset = build_a15_2_train_dataset(samples)
        population_role = "correction_subset"
    else:
        population_manifest = Path(fit_manifest_path).resolve()
        samples = tuple(load_syncg_samples(population_manifest))
        _require(
            len(samples) == 14_442
            and len({sample.scene_stem for sample in samples}) == 131,
            "relative context full-fit population differs",
        )
        dataset = A152ThreeConditionTrainDataset(samples)
        population_role = "complete_original_fit"
    anchor, correction, source_metadata = load_context_correction_adapted_remst(
        source_checkpoint_path, device=device
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    for parameter in correction.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    correction.eval()
    cache_source = None if cache_path is None else Path(cache_path).resolve()
    if cache_source is not None and cache_source.exists():
        payload = torch.load(cache_source, map_location="cpu", weights_only=False)
        _require(
            isinstance(payload, Mapping)
            and payload.get("protocol") == PROTOCOL
            and payload.get("source_checkpoint")
            == str(Path(source_checkpoint_path).resolve())
            and int(payload.get("seed", -1)) == int(seed)
            and payload.get("population_manifest") == str(population_manifest)
            and int(payload.get("physical_samples", -1)) == len(samples)
            and isinstance(payload.get("cache"), Mapping),
            "relative context cache metadata differs",
        )
        cache = dict(payload["cache"])
        cache_mode = "reused"
    else:
        cache = _collect_fit_cache(
            anchor,
            correction,
            dataset,
            device=device,
            workers=int(workers),
            sample_order_seed=int(sample_order_seed),
        )
        cache_mode = "created_in_memory" if cache_source is None else "created_and_saved"
        if cache_source is not None:
            cache_source.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "source_checkpoint": str(Path(source_checkpoint_path).resolve()),
                    "seed": int(seed),
                    "population_manifest": str(population_manifest),
                    "physical_samples": len(samples),
                    "cache": cache,
                },
                cache_source,
            )
    _require(
        cache["raw_representation"].shape
        == (len(samples), CONDITIONS_PER_PHYSICAL, RESNET18_REPRESENTATION_FEATURES)
        and cache["sarn_representation"].shape
        == cache["raw_representation"].shape
        and cache["geometry_features"].shape
        == (len(samples), CONDITIONS_PER_PHYSICAL, GEOMETRY_FEATURES)
        and cache["target"].shape == (len(samples),),
        "relative context cache tensor identity differs",
    )
    head = RelativeContextMomentHead(
        latent_features=int(latent_features),
        hidden_features=int(hidden_features),
        max_progress_shift=float(max_progress_shift),
    ).to(device)
    tensor_dataset = TensorDataset(
        cache["raw_representation"],
        cache["sarn_representation"],
        cache["geometry_features"],
        cache["raw_mean"],
        cache["sarn_mean"],
        cache["base_mean"],
        cache["target"],
        cache["active"],
    )
    loader = DataLoader(
        tensor_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=torch.Generator().manual_seed(int(seed) + 80_000),
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        epoch_started = time.perf_counter()
        metrics = _train_epoch(
            head,
            loader,
            device=device,
            optimizer=optimizer,
            condition_weights=condition_weights,
            tail_weight=float(tail_weight),
            consistency_weight=float(consistency_weight),
            residual_l2_weight=float(residual_l2_weight),
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "metrics": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "phase": "relative_context_training",
                    "epoch": epoch,
                    "learning_rate": row["learning_rate"],
                    **metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()
    elapsed = time.perf_counter() - started
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "publication_model": {
            **remst_resnet18_publication_identity(),
            "short_name": "ReMST-ResNet18-RCMT",
            "display_name": "ReMST-ResNet18 Relative Context Moment Transport",
        },
        "source_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_metadata": source_metadata,
        "seed": int(seed),
        "checkpoint_selection": "terminal_fixed_epoch",
        "training_scope": {
            "correction_train_only": population_role == "correction_subset",
            "complete_original_fit": population_role == "complete_original_fit",
            "source_anchor_frozen": True,
            "source_correction_frozen": True,
            "formal_holdout_access": False,
            "main_table_development_access": False,
        },
        "training_data": {
            "manifest": str(population_manifest),
            "role": population_role,
            "physical_samples": len(samples),
            "scenes": len({sample.scene_stem for sample in samples}),
            "conditions": list(PROJECTIVE_CONDITIONS),
        },
        "construction": {
            "progress_bins": int(getattr(correction, "progress_bins")),
            "latent_features": int(latent_features),
            "hidden_features": int(hidden_features),
            "max_progress_shift": float(max_progress_shift),
            "additional_image_encoders": 0,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "scheduler": "CosineAnnealingLR",
        },
        "loss": {
            "exact_l1": 1.0,
            "condition_weights": {
                name: weight
                for name, weight in zip(PROJECTIVE_CONDITIONS, condition_weights)
            },
            "tail_l1": float(tail_weight),
            "cross_condition_consistency": float(consistency_weight),
            "residual_l2": float(residual_l2_weight),
        },
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "sample_order_seed": int(sample_order_seed),
        "cache": {
            "mode": cache_mode,
            "path": None if cache_source is None else str(cache_source),
            "physical_samples": len(samples),
        },
        "parameter_counts": {
            "relative_context_head": sum(
                parameter.numel() for parameter in head.parameters()
            ),
            "additional_image_encoders": 0,
        },
        "history": history,
        "training_elapsed_seconds": elapsed,
        "context_head_state": {
            name: value.detach().cpu().clone()
            for name, value in head.state_dict().items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "cache_mode": cache_mode,
        "epochs": int(epochs),
        "parameters": checkpoint["parameter_counts"]["relative_context_head"],
        "training_elapsed_seconds": elapsed,
        "terminal_exact_l1": history[-1]["metrics"]["exact_l1"],
    }


def load_relative_context_transport(
    checkpoint_path: Path, *, device: torch.device | str
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"relative context checkpoint is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(payload, Mapping)
        and payload.get("protocol") == PROTOCOL
        and payload.get("architecture") == ARCHITECTURE
        and isinstance(payload.get("construction"), Mapping)
        and isinstance(payload.get("context_head_state"), Mapping),
        "relative context checkpoint metadata differs",
    )
    anchor, correction, source_metadata = load_context_correction_adapted_remst(
        Path(str(payload["source_checkpoint"])), device=device
    )
    construction = payload["construction"]
    wrapped = RelativeContextMomentTransport(
        correction,
        progress_bins=int(construction["progress_bins"]),
        latent_features=int(construction["latent_features"]),
        hidden_features=int(construction["hidden_features"]),
        max_progress_shift=float(construction["max_progress_shift"]),
    ).to(device)
    incompatibility = wrapped.context_head.load_state_dict(
        payload["context_head_state"], strict=True
    )
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "relative context head does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapped.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    wrapped.eval()
    metadata = {
        "checkpoint": str(source),
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "source_checkpoint": str(payload["source_checkpoint"]),
        "source_metadata": source_metadata,
        "seed": int(payload["seed"]),
        "epochs": int(payload["epochs"]),
        "checkpoint_selection": str(payload["checkpoint_selection"]),
        "training_scope": dict(payload["training_scope"]),
        "construction": dict(construction),
        "parameter_counts": dict(payload["parameter_counts"]),
    }
    if isinstance(payload.get("weight_space_average"), Mapping):
        metadata["weight_space_average"] = dict(payload["weight_space_average"])
    return anchor, wrapped, metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        help=(
            "Use the complete original fit population instead of the nested "
            "correction subset."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--latent-features", type=int, default=DEFAULT_LATENT_FEATURES)
    parser.add_argument("--hidden-features", type=int, default=DEFAULT_HIDDEN_FEATURES)
    parser.add_argument(
        "--max-progress-shift", type=float, default=DEFAULT_MAX_PROGRESS_SHIFT
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--condition-weights",
        type=float,
        nargs=CONDITIONS_PER_PHYSICAL,
        default=DEFAULT_CONDITION_WEIGHTS,
        metavar=("W25", "W45", "W45_BLUR"),
    )
    parser.add_argument("--tail-weight", type=float, default=DEFAULT_TAIL_WEIGHT)
    parser.add_argument(
        "--consistency-weight", type=float, default=DEFAULT_CONSISTENCY_WEIGHT
    )
    parser.add_argument(
        "--residual-l2-weight", type=float, default=DEFAULT_RESIDUAL_L2_WEIGHT
    )
    parser.add_argument(
        "--sample-order-seed", type=int, default=SAMPLE_ORDER_SEED + 70_000
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_relative_context_transport(
        source_checkpoint_path=args.source_checkpoint,
        correction_train_manifest_path=args.correction_train_manifest,
        fit_manifest_path=args.fit_manifest,
        output_path=args.output,
        cache_path=args.cache,
        seed=args.seed,
        device_name=args.device,
        workers=args.workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_features=args.latent_features,
        hidden_features=args.hidden_features,
        max_progress_shift=args.max_progress_shift,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        condition_weights=args.condition_weights,
        tail_weight=args.tail_weight,
        consistency_weight=args.consistency_weight,
        residual_l2_weight=args.residual_l2_weight,
        sample_order_seed=args.sample_order_seed,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "PROTOCOL",
    "RelativeContextMomentHead",
    "RelativeContextMomentTransport",
    "load_relative_context_transport",
    "train_relative_context_transport",
]
