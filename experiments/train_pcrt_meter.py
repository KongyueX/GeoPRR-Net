"""Train the single-seed PCRT-Meter polar expert and regret router.

Only the established SyncG formal-fit population is opened.  A fixed inner
scene split trains the polar expert on the inner-train scenes and the regret
router on the disjoint inner-development scenes.  Formal SyncG holdout and all
real photographs remain inaccessible until the terminal checkpoint is saved.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments import robustness_degradations
from experiments.pcrt_meter import (
    ARCHITECTURE,
    ConditionalRegretRouter,
    PCRTMeter,
    load_remst_r2mt_expert_bank,
    parameter_counts,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.raw_angular_moment_refiner_probe import (
    AngularMomentLogitRefiner,
    AngularProgressDataset,
)
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_FIT_MANIFEST,
    load_inner_scene_population,
)
from experiments.resnet18_direct_progress import (
    IMAGE_SIZE,
    DirectSample,
    _configure_reproducibility,
    _loader,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    ROBUSTNESS_SEED,
)
from experiments.support_aware_roi_normalization_v2 import (
    normalize_support_aware_roi_v2,
)
from experiments.syncg_lightweight_regression_baselines import DEFAULT_SCENE_SPLIT
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _normalized_raw_to_sarn_homography,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


PROTOCOL: Final[str] = "pcrt_meter_single_seed_training_v1"
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_SOURCE_R2MT: Final[Path] = Path(
    "artifacts/runs/remstnet_r2mt/seed_20262020/terminal.pt"
)
DEFAULT_WARM_POLAR: Final[Path] = Path(
    "artifacts/runs/raw_angular_moment_refiner_probe/seed_20262020/"
    "angular_refiners.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/pcrt_meter/seed_20262020/terminal.pt"
)
DEFAULT_ROUTER_CACHE: Final[Path] = Path(
    "artifacts/runs/pcrt_meter/seed_20262020/inner_dev_router_cache.pt"
)
DEFAULT_POLAR_EPOCHS: Final[int] = 5
DEFAULT_ROUTER_EPOCHS: Final[int] = 40
DEFAULT_POLAR_BATCH_SIZE: Final[int] = 64
DEFAULT_ROUTER_BATCH_SIZE: Final[int] = 512
DEFAULT_WORKERS: Final[int] = 4
CONDITION_WEIGHTS: Final[tuple[float, ...]] = (1.5, 1.5, 1.5, 1.0, 1.0, 1.5)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class PCRTConditionedMultiViewDataset(Dataset[dict[str, Any]]):
    """Six-condition SyncG rows with the exact evaluation-time SARN path."""

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        condition: str,
        degradation_seed: int = ROBUSTNESS_SEED,
    ) -> None:
        self.samples = tuple(samples)
        self.condition = str(condition)
        self.degradation_seed = int(degradation_seed)
        _require(bool(self.samples), "PCRT router dataset is empty")
        _require(self.condition in CONDITIONS, "PCRT condition is unknown")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source decode failed")
        clean, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        degraded, _metadata = robustness_degradations.apply_degradation(
            clean,
            self.condition,
            sample_id=sample.sample_id,
            seed=self.degradation_seed,
        )
        degraded = np.ascontiguousarray(degraded)
        decision = normalize_support_aware_roi_v2(degraded)
        height, width = degraded.shape[:2]
        active = bool(decision.applied)
        raw_mask = decision.valid_support_mask
        if raw_mask is None:
            active = False
            support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            support = cv2.resize(
                np.asarray(raw_mask, dtype=np.float32),
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            support = np.clip(support, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(support).all() or float(support.sum()) <= 0.0:
                active = False
                support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        homography = (
            _normalized_raw_to_sarn_homography(
                decision,
                height=height,
                width=width,
            )
            if active
            else np.eye(3, dtype=np.float32)
        )
        return {
            "sample_id": sample.sample_id,
            "scene_stem": sample.scene_stem,
            "condition_name": self.condition,
            "original_view": normalized_rgb_tensor(
                direct_resize_whole_roi(degraded, size=IMAGE_SIZE)
            ),
            "sarn_view": normalized_rgb_tensor(
                direct_resize_whole_roi(decision.image, size=IMAGE_SIZE)
            ),
            "sarn_support_mask": torch.from_numpy(
                np.ascontiguousarray(support[None], dtype=np.float32)
            ),
            "sarn_active": torch.tensor(active, dtype=torch.bool),
            "raw_to_sarn_homography": torch.from_numpy(
                np.ascontiguousarray(homography, dtype=np.float32)
            ),
            "target": torch.tensor(sample.normalized_target, dtype=torch.float32),
        }


def _raw_foundation_features(
    base_model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        stride8 = base_model.shared_scale8_encoder(images)
        stride16 = base_model.shared_scale16_encoder(stride8)
        context = base_model.shared_context_encoder(stride16)
        representation = context.mean(dim=(2, 3))
        point = base_model.moment_readout.point_projection
        logit = F.linear(
            representation.float(),
            point.weight.detach().float(),
            None if point.bias is None else point.bias.detach().float(),
        ).squeeze(1)
    return stride8.detach(), stride16.detach(), representation.detach(), logit.detach()


def _load_warm_polar(path: Path) -> tuple[AngularMomentLogitRefiner, dict[str, Any]]:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "warm polar checkpoint is malformed")
    state = payload.get("angular_no_aux_state")
    _require(isinstance(state, Mapping), "warm polar state is missing")
    model = AngularMomentLogitRefiner()
    model.load_state_dict(state, strict=True)
    return model, {
        "path": str(source),
        "protocol": str(payload.get("protocol")),
        "seed": int(payload.get("seed", -1)),
        "epochs": int(payload.get("epochs", -1)),
        "role": "initialization_only_different_inner_foundation",
    }


def _train_polar(
    model: PCRTMeter,
    samples: Sequence[DirectSample],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    dataset = AngularProgressDataset(samples, training=True, seed=seed)
    optimizer = torch.optim.AdamW(
        model.polar_expert.parameters(), lr=5.0e-4, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    model.expert_bank.eval()
    model.regret_router.eval()
    for epoch_index in range(epochs):
        dataset.set_epoch(30 + epoch_index)
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + 10_000 + epoch_index,
            cuda=device.type == "cuda",
        )
        model.polar_expert.train(True)
        absolute_sum = 0.0
        tail_sum = 0.0
        count = 0
        steps = 0
        started = time.perf_counter()
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            target = batch["progress"].to(
                device, non_blocking=device.type == "cuda"
            )
            stride8, stride16, representation, base_logit = _raw_foundation_features(
                model.expert_bank.base_model, images
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                details = model.polar_expert(stride8, stride16, representation)
                prediction = torch.sigmoid(base_logit + details["logit_delta"].float())
                errors = torch.abs(prediction - target)
                tail_count = max(1, int(math.ceil(0.10 * errors.numel())))
                tail = torch.topk(errors, k=tail_count, largest=True).values.mean()
                loss = errors.mean() + 0.10 * tail
            previous_scale = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
                steps += 1
            absolute_sum += float(errors.detach().sum().cpu())
            tail_sum += float(tail.detach().cpu()) * int(target.numel())
            count += int(target.numel())
        row = {
            "phase": "polar_expert",
            "epoch": epoch_index + 1,
            "samples": count,
            "optimizer_steps": steps,
            "nmae": absolute_sum / count,
            "tail10": tail_sum / count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    return history


def _collect_router_cache(
    model: PCRTMeter,
    samples: Sequence[DirectSample],
    *,
    cache_path: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, torch.Tensor], float]:
    destination = Path(cache_path).resolve()
    _require(not destination.exists(), f"router cache already exists: {destination}")
    model.eval()
    fields: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "raw_representation",
            "sarn_representation",
            "geometry_features",
            "base_mean",
            "polar_mean",
            "r2mt_mean",
            "raw_mean",
            "sarn_mean",
            "relation_available",
            "polar_posterior",
            "polar_concentration",
            "polar_entropy",
            "target",
            "condition_index",
        )
    }
    started = time.perf_counter()
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            dataset = PCRTConditionedMultiViewDataset(samples, condition=condition)
            loader = _loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=seed + 20_000 + condition_index,
                cuda=device.type == "cuda",
            )
            condition_rows = 0
            for batch in loader:
                raw = batch["original_view"].to(device)
                sarn = batch["sarn_view"].to(device)
                support = batch["sarn_support_mask"].to(device)
                active = batch["sarn_active"].to(device).bool()
                homography = batch["raw_to_sarn_homography"].to(device)
                effective_active = active & (condition != "clean")
                r2mt, features = model.expert_bank.forward_experts(
                    raw,
                    sarn,
                    support,
                    sarn_active=effective_active,
                    raw_to_sarn_homography=homography,
                )
                polar_mean, polar = model._polar_prediction(features)
                count = int(raw.shape[0])
                values = {
                    "raw_representation": features["raw_representation"].half(),
                    "sarn_representation": features["sarn_representation"].half(),
                    "geometry_features": features["geometry_features"].float(),
                    "base_mean": r2mt["r2mt_base_mean"].float(),
                    "polar_mean": polar_mean.float(),
                    "r2mt_mean": r2mt["mean"].float(),
                    "raw_mean": r2mt["raw_anchor_mean"].float(),
                    "sarn_mean": r2mt["sarn_endpoint_mean"].float(),
                    "relation_available": r2mt["relation_available"].float(),
                    "polar_posterior": polar["direction_posterior"].float(),
                    "polar_concentration": polar["direction_concentration"].float(),
                    "polar_entropy": polar["direction_entropy"].float(),
                    "target": batch["target"].float(),
                    "condition_index": torch.full(
                        (count,), condition_index, dtype=torch.long, device=device
                    ),
                }
                for name, value in values.items():
                    fields[name].append(value.detach().cpu())
                condition_rows += count
            print(
                json.dumps(
                    {
                        "phase": "router_cache",
                        "condition": condition,
                        "rows": condition_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    cache = {name: torch.cat(values, dim=0) for name, values in fields.items()}
    elapsed = time.perf_counter() - started
    expected = len(samples) * len(CONDITIONS)
    _require(
        all(int(value.shape[0]) == expected for value in cache.values()),
        "router cache row counts differ",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "seed": seed,
            "scope": "syncg_inner_dev_only",
            "samples": len(samples),
            "conditions": list(CONDITIONS),
            "rows": expected,
            "elapsed_seconds": elapsed,
            "cache": cache,
        },
        destination,
    )
    return cache, elapsed


def _router_arguments(batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(batch[:12])


def _train_router(
    model: PCRTMeter,
    cache: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    names = (
        "raw_representation",
        "sarn_representation",
        "geometry_features",
        "base_mean",
        "polar_mean",
        "r2mt_mean",
        "raw_mean",
        "sarn_mean",
        "relation_available",
        "polar_posterior",
        "polar_concentration",
        "polar_entropy",
        "target",
        "condition_index",
    )
    dataset = TensorDataset(*(cache[name] for name in names))
    optimizer = torch.optim.AdamW(
        model.regret_router.parameters(), lr=1.0e-3, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed + 30_001)
    history: list[dict[str, Any]] = []
    condition_weights = torch.tensor(CONDITION_WEIGHTS, device=device)
    for epoch_index in range(epochs):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model.regret_router.train(True)
        totals = {"loss": 0.0, "final_l1": 0.0, "gain": 0.0, "no_harm": 0.0}
        rows = 0
        started = time.perf_counter()
        for cpu_batch in loader:
            batch = tuple(value.to(device) for value in cpu_batch)
            target = batch[12].float()
            condition_index = batch[13].long()
            optimizer.zero_grad(set_to_none=True)
            weights, predicted_gains, _logits = model.regret_router(
                *_router_arguments(batch)
            )
            candidates = torch.stack((batch[3], batch[4], batch[5]), dim=1).float()
            prediction = (weights * candidates).sum(dim=1)
            errors = torch.abs(candidates - target[:, None])
            true_gains = errors[:, :1] - errors[:, 1:]
            row_weight = condition_weights[condition_index]
            final_errors = torch.abs(prediction - target)
            final_l1 = (row_weight * final_errors).sum() / row_weight.sum()
            gain_loss = F.smooth_l1_loss(
                predicted_gains,
                true_gains,
                beta=0.01,
            )
            raw_scope = condition_index <= 2
            if bool(raw_scope.any()):
                no_harm = torch.relu(
                    final_errors[raw_scope] - errors[raw_scope, 0]
                ).mean()
            else:
                no_harm = torch.zeros((), device=device)
            oracle = errors.argmin(dim=1)
            route_loss = F.nll_loss(torch.log(weights.clamp_min(1.0e-8)), oracle)
            loss = final_l1 + 0.50 * gain_loss + 0.25 * no_harm + 0.002 * route_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.regret_router.parameters(), 5.0)
            optimizer.step()
            count = int(target.numel())
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["final_l1"] += float(final_l1.detach().cpu()) * count
            totals["gain"] += float(gain_loss.detach().cpu()) * count
            totals["no_harm"] += float(no_harm.detach().cpu()) * count
            rows += count
        row = {
            "phase": "conditional_regret_router",
            "epoch": epoch_index + 1,
            "rows": rows,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": {name: value / rows for name, value in totals.items()},
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    return history


def train_pcrt_meter(
    *,
    source_r2mt_checkpoint: Path,
    warm_polar_checkpoint: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_path: Path,
    router_cache_path: Path,
    device_name: str,
    seed: int,
    polar_epochs: int,
    router_epochs: int,
    polar_batch_size: int,
    router_batch_size: int,
    workers: int,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    cache_path = Path(router_cache_path).resolve()
    _require(not output.exists(), f"PCRT checkpoint already exists: {output}")
    _require(not cache_path.exists(), f"PCRT router cache already exists: {cache_path}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)
    bank, bank_metadata = load_remst_r2mt_expert_bank(
        source_r2mt_checkpoint, device=device
    )
    polar, warm_metadata = _load_warm_polar(warm_polar_checkpoint)
    router = ConditionalRegretRouter()
    model = PCRTMeter(bank, polar, router).to(device)
    started = time.perf_counter()
    polar_history = _train_polar(
        model,
        internal.train,
        device=device,
        seed=seed,
        epochs=polar_epochs,
        batch_size=polar_batch_size,
        workers=workers,
    )
    cache, cache_elapsed = _collect_router_cache(
        model,
        internal.dev,
        cache_path=cache_path,
        device=device,
        seed=seed,
        batch_size=polar_batch_size,
        workers=workers,
    )
    router_history = _train_router(
        model,
        cache,
        device=device,
        seed=seed,
        epochs=router_epochs,
        batch_size=router_batch_size,
    )
    counts = parameter_counts(model)
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "checkpoint_selection": "terminal_fixed_epoch",
        "source_r2mt_checkpoint": str(Path(source_r2mt_checkpoint).resolve()),
        "source_r2mt_metadata": bank_metadata,
        "warm_polar": warm_metadata,
        "construction": {
            "router_latent_features": model.regret_router.latent_features,
            "router_hidden_features": model.regret_router.hidden_features,
            "maximum_gain": model.regret_router.maximum_gain,
            "gain_logit_scale": model.regret_router.gain_logit_scale,
            "prior_weights": model.regret_router.prior_weights.detach().cpu().tolist(),
            "expert_names": ["identity", "polar", "r2mt"],
            "polar_radial_bins": model.polar_expert.radial_bins,
            "polar_angular_bins": model.polar_expert.angular_bins,
            "additional_image_encoders": 0,
        },
        "training_scope": {
            "syncg_formal_fit_only": True,
            "polar_inner_train_samples": len(internal.train),
            "router_inner_dev_samples": len(internal.dev),
            "router_inner_dev_rows": len(internal.dev) * len(CONDITIONS),
            "inner_scene_disjoint": True,
            "formal_syncg_holdout_access": False,
            "real_photo_access": False,
            "real_photos_test_only": True,
        },
        "training": {
            "polar_epochs": polar_epochs,
            "router_epochs": router_epochs,
            "polar_history": polar_history,
            "router_history": router_history,
            "router_cache": str(cache_path),
            "router_cache_elapsed_seconds": cache_elapsed,
            "total_elapsed_seconds": time.perf_counter() - started,
        },
        "parameter_counts": counts,
        "polar_expert_state": {
            name: value.detach().cpu().clone()
            for name, value in model.polar_expert.state_dict().items()
        },
        "regret_router_state": {
            name: value.detach().cpu().clone()
            for name, value in model.regret_router.state_dict().items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "router_cache": str(cache_path),
        "elapsed_seconds": checkpoint["training"]["total_elapsed_seconds"],
        "polar_terminal_nmae": polar_history[-1]["nmae"],
        "router_terminal_metrics": router_history[-1]["metrics"],
        "parameter_counts": counts,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r2mt", type=Path, default=DEFAULT_SOURCE_R2MT)
    parser.add_argument("--warm-polar", type=Path, default=DEFAULT_WARM_POLAR)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--router-cache", type=Path, default=DEFAULT_ROUTER_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--polar-epochs", type=int, default=DEFAULT_POLAR_EPOCHS)
    parser.add_argument("--router-epochs", type=int, default=DEFAULT_ROUTER_EPOCHS)
    parser.add_argument("--polar-batch-size", type=int, default=DEFAULT_POLAR_BATCH_SIZE)
    parser.add_argument("--router-batch-size", type=int, default=DEFAULT_ROUTER_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_pcrt_meter(
        source_r2mt_checkpoint=args.source_r2mt,
        warm_polar_checkpoint=args.warm_polar,
        fit_manifest_path=args.fit_manifest,
        outer_split_path=args.outer_split,
        output_path=args.output,
        router_cache_path=args.router_cache,
        device_name=args.device,
        seed=args.seed,
        polar_epochs=args.polar_epochs,
        router_epochs=args.router_epochs,
        polar_batch_size=args.polar_batch_size,
        router_batch_size=args.router_batch_size,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PCRTConditionedMultiViewDataset",
    "PROTOCOL",
    "train_pcrt_meter",
]
