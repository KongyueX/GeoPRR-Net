"""Train the unified non-distilled reader and its compact ablation suite.

Only the established SyncG formal-fit population is opened.  The polar branch
is fitted on the inner-train scenes; candidate routing is fitted on the
disjoint inner-development scenes under all six robustness conditions.  The
formal SyncG holdout and every Industrial image remain inaccessible until all
terminal checkpoints have been written.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.pcrt_meter import ConditionalRegretRouter, load_remst_r2mt_expert_bank
from experiments.raw_angular_moment_refiner_probe import AngularProgressDataset
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_FIT_MANIFEST,
    load_inner_scene_population,
)
from experiments.refine_pcrt_minimax_router import ROUTER_ARGUMENT_NAMES
from experiments.resnet18_direct_progress import (
    DirectSample,
    _configure_reproducibility,
    _loader,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.syncg_lightweight_regression_baselines import DEFAULT_SCENE_SPLIT
from experiments.train_pcrt_meter import (
    CONDITION_WEIGHTS,
    _collect_router_cache,
    _load_warm_polar,
    _raw_foundation_features,
)
from experiments.unified_pointer_reader import (
    ARCHITECTURE,
    FIXED_ROUTING,
    FULL,
    NO_GEOMETRY_FUSION,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
    PROTOCOL,
    UnifiedPointerReader,
    candidate_mask,
    fixed_geometry_base,
    mask_router_logits,
    parameter_inventory,
)


DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_WARM_POLAR: Final[Path] = Path(
    "artifacts/runs/raw_angular_moment_refiner_probe/seed_20262020/"
    "angular_refiners.pt"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    "artifacts/runs/unified_pointer_reader/seed_20262020"
)
DYNAMIC_VARIANTS: Final[tuple[str, ...]] = (
    FULL,
    NO_GEOMETRY_FUSION,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
)
TRAIN_FIELDS: Final[tuple[str, ...]] = ROUTER_ARGUMENT_NAMES + (
    "target",
    "condition_index",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ParameterEMA:
    """Low-pass one optimization trajectory without ensembling seed modes."""

    def __init__(self, module: torch.nn.Module, *, decay: float) -> None:
        _require(0.0 < float(decay) < 1.0, "EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.names = tuple(name for name, _parameter in module.named_parameters())
        self.values = tuple(
            parameter.detach().clone() for parameter in module.parameters()
        )
        _require(bool(self.names), "EMA module has no parameters")
        self.updates = 0

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        current = tuple(parameter.detach() for parameter in module.parameters())
        _require(len(current) == len(self.values), "EMA parameter roster differs")
        torch._foreach_mul_(self.values, self.decay)
        torch._foreach_add_(self.values, current, alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        _require(self.updates > 0, "EMA received no optimizer update")
        state = {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
        }
        for name, value in zip(self.names, self.values, strict=True):
            _require(name in state, "EMA state name differs")
            state[name] = value.detach().cpu().clone()
        return state


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _construction(router: ConditionalRegretRouter, polar: torch.nn.Module) -> dict[str, Any]:
    return {
        "router_latent_features": int(router.latent_features),
        "router_hidden_features": int(router.hidden_features),
        "maximum_gain": float(router.maximum_gain),
        "gain_logit_scale": float(router.gain_logit_scale),
        "prior_weights": [
            float(value) for value in router.prior_weights.detach().cpu()
        ],
        "candidate_names": ["base", "polar_evidence", "relational_transport"],
        "polar_radial_bins": int(getattr(polar, "radial_bins")),
        "polar_angular_bins": int(getattr(polar, "angular_bins")),
        "additional_image_encoders": 0,
    }


def _build_router(
    construction: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> ConditionalRegretRouter:
    router = ConditionalRegretRouter(
        latent_features=int(construction["router_latent_features"]),
        hidden_features=int(construction["router_hidden_features"]),
        maximum_gain=float(construction["maximum_gain"]),
        gain_logit_scale=float(construction["gain_logit_scale"]),
        prior_weights=tuple(float(value) for value in construction["prior_weights"]),
    )
    router.load_state_dict(state, strict=True)
    return router.to(device)


def _train_polar_with_ema(
    model: UnifiedPointerReader,
    samples: Sequence[DirectSample],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    workers: int,
    ema_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    dataset = AngularProgressDataset(samples, training=True, seed=seed)
    optimizer = torch.optim.AdamW(
        model.polar_expert.parameters(), lr=5.0e-4, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ParameterEMA(model.polar_expert, decay=ema_decay)
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
        rows = 0
        optimizer_steps = 0
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
                prediction = torch.sigmoid(
                    base_logit + details["logit_delta"].float()
                )
                errors = torch.abs(prediction - target)
                tail_count = max(1, int(math.ceil(0.10 * errors.numel())))
                tail = torch.topk(errors, k=tail_count, largest=True).values.mean()
                loss = errors.mean() + 0.10 * tail
            previous_scale = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
                ema.update(model.polar_expert)
                optimizer_steps += 1
            count = int(target.numel())
            absolute_sum += float(errors.detach().sum().cpu())
            tail_sum += float(tail.detach().cpu()) * count
            rows += count
        row = {
            "phase": "polar_evidence",
            "epoch": epoch_index + 1,
            "samples": rows,
            "optimizer_steps": optimizer_steps,
            "nmae": absolute_sum / rows,
            "tail10": tail_sum / rows,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
    return history, _clone_state(model.polar_expert), ema.state_dict(model.polar_expert), ema.updates


def _variant_cache(
    cache: Mapping[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    _require(variant in DYNAMIC_VARIANTS, "unknown dynamic variant")
    result = {name: cache[name] for name in TRAIN_FIELDS}
    if variant == NO_GEOMETRY_FUSION:
        base = fixed_geometry_base(
            cache["raw_mean"],
            cache["sarn_mean"],
            cache["relation_available"],
        )
        result["r2mt_mean"] = (
            base + cache["r2mt_mean"].float() - cache["base_mean"].float()
        ).clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
        result["base_mean"] = base
    elif variant == NO_POLAR_EVIDENCE:
        base = cache["base_mean"].float()
        result["polar_mean"] = base
        result["polar_posterior"] = torch.full_like(
            cache["polar_posterior"],
            1.0 / float(cache["polar_posterior"].shape[1]),
        )
        result["polar_concentration"] = torch.zeros_like(
            cache["polar_concentration"]
        )
        result["polar_entropy"] = torch.ones_like(cache["polar_entropy"])
    elif variant == NO_RELATIONAL_TRANSPORT:
        result["r2mt_mean"] = cache["base_mean"].float()
    return result


def _router_forward(
    router: ConditionalRegretRouter,
    arguments: Sequence[torch.Tensor],
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _weights, gains, logits = router(*arguments)
    return mask_router_logits(logits, active), gains, logits


def _active_gain_loss(
    predicted_gains: torch.Tensor,
    errors: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    selected = active[1:]
    if not bool(selected.any()):
        return predicted_gains.sum() * 0.0
    true_gains = errors[:, :1] - errors[:, 1:]
    return F.smooth_l1_loss(
        predicted_gains[:, selected],
        true_gains[:, selected],
        beta=0.01,
    )


def _masked_oracle(errors: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    return errors.masked_fill(~active[None], torch.inf).argmin(dim=1)


def _condition_statistics(
    weights: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    condition_index: torch.Tensor,
    conditions: Sequence[str],
    active: torch.Tensor,
) -> dict[str, Any]:
    prediction = (weights * candidates).sum(dim=1)
    result: dict[str, Any] = {}
    for index, name in enumerate(conditions):
        selected = condition_index == index
        result[str(name)] = {
            "nmae": float(torch.abs(prediction[selected] - target[selected]).mean().cpu()),
            "candidate_nmae": [
                float(value)
                for value in torch.abs(
                    candidates[selected] - target[selected, None]
                ).mean(dim=0).cpu()
            ],
            "mean_routing_weights": [
                float(value) for value in weights[selected].mean(dim=0).cpu()
            ],
            "candidate_active": [bool(value) for value in active.cpu()],
        }
    return result


def _train_router_stage_one(
    router: ConditionalRegretRouter,
    cache: Mapping[str, torch.Tensor],
    *,
    variant: str,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    ema: ParameterEMA,
) -> list[dict[str, Any]]:
    dataset = TensorDataset(*(cache[name] for name in TRAIN_FIELDS))
    optimizer = torch.optim.AdamW(router.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(seed + 30_001)
    condition_weights = torch.tensor(CONDITION_WEIGHTS, device=device)
    active = candidate_mask(variant, device=device)
    history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        router.train(True)
        totals = {"loss": 0.0, "final_l1": 0.0, "gain": 0.0, "no_harm": 0.0, "route": 0.0}
        rows = 0
        for cpu_batch in loader:
            batch = tuple(value.to(device) for value in cpu_batch)
            target = batch[12].float()
            condition_index = batch[13].long()
            candidates = torch.stack((batch[3], batch[4], batch[5]), dim=1).float()
            optimizer.zero_grad(set_to_none=True)
            weights, gains, _logits = _router_forward(router, batch[:12], active)
            prediction = (weights * candidates).sum(dim=1)
            errors = torch.abs(candidates - target[:, None])
            row_weight = condition_weights[condition_index]
            final_errors = torch.abs(prediction - target)
            final_l1 = (row_weight * final_errors).sum() / row_weight.sum()
            gain_loss = _active_gain_loss(gains, errors, active)
            raw_scope = condition_index <= 2
            no_harm = torch.relu(
                final_errors[raw_scope] - errors[raw_scope, 0]
            ).mean()
            oracle = _masked_oracle(errors, active)
            route_loss = F.nll_loss(torch.log(weights.clamp_min(1.0e-8)), oracle)
            loss = final_l1 + 0.50 * gain_loss + 0.25 * no_harm + 0.002 * route_loss
            _require(bool(torch.isfinite(loss)), f"{variant}: stage-one loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
            optimizer.step()
            ema.update(router)
            count = int(target.numel())
            rows += count
            for name, value in (
                ("loss", loss),
                ("final_l1", final_l1),
                ("gain", gain_loss),
                ("no_harm", no_harm),
                ("route", route_loss),
            ):
                totals[name] += float(value.detach().cpu()) * count
        scheduler.step()
        row = {
            "phase": "conditional_router_stage_one",
            "variant": variant,
            "epoch": epoch_index + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": {name: value / rows for name, value in totals.items()},
        }
        history.append(row)
        if epoch_index == 0 or (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs:
            print(json.dumps(row, sort_keys=True), flush=True)
    return history


def _train_router_stage_two(
    router: ConditionalRegretRouter,
    cache: Mapping[str, torch.Tensor],
    *,
    variant: str,
    conditions: Sequence[str],
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    smooth_max_temperature: float,
    ema: ParameterEMA,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed + 40_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 40_001)
    arguments = tuple(cache[name].to(device) for name in ROUTER_ARGUMENT_NAMES)
    candidates = torch.stack(
        (cache["base_mean"], cache["polar_mean"], cache["r2mt_mean"]), dim=1
    ).float().to(device)
    target = cache["target"].float().to(device)
    condition_index = cache["condition_index"].long().to(device)
    relation_available = cache["relation_available"].bool().to(device)
    active = candidate_mask(variant, device=device)
    router.eval()
    with torch.inference_mode():
        initial_weights, _gains, _logits = _router_forward(router, arguments, active)
        initial_prediction = (initial_weights * candidates).sum(dim=1)
        reference_condition_l1 = torch.stack(
            tuple(
                torch.abs(initial_prediction[condition_index == index] - target[condition_index == index]).mean()
                for index in range(len(conditions))
            )
        )
        active_floor = torch.where(
            active[None],
            torch.abs(candidates - target[:, None]),
            torch.full_like(candidates, torch.inf),
        )
        expert_condition_floor = torch.stack(
            tuple(
                active_floor[condition_index == index].mean(dim=0).masked_fill(~active, torch.inf).min()
                for index in range(len(conditions))
            )
        )
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, Any]] = []
    for epoch_index in range(epochs):
        router.train(True)
        optimizer.zero_grad(set_to_none=True)
        weights, gains, _logits = _router_forward(router, arguments, active)
        prediction = (weights * candidates).sum(dim=1)
        row_errors = torch.abs(prediction - target)
        condition_l1 = torch.stack(
            tuple(row_errors[condition_index == index].mean() for index in range(len(conditions)))
        )
        condition_excess = condition_l1 - expert_condition_floor
        worst_excess = smooth_max_temperature * torch.logsumexp(
            condition_excess / smooth_max_temperature, dim=0
        )
        mean_condition_l1 = condition_l1.mean()
        condition_regression = torch.relu(
            condition_l1 - reference_condition_l1
        ).mean()
        no_relation = ~relation_available
        polar_consistency = (
            torch.abs(prediction[no_relation] - candidates[no_relation, 1]).mean()
            if bool(active[1]) and bool(no_relation.any())
            else prediction.sum() * 0.0
        )
        expert_errors = torch.abs(candidates - target[:, None])
        gain_loss = _active_gain_loss(gains, expert_errors, active)
        oracle = _masked_oracle(expert_errors, active)
        route_loss = F.nll_loss(torch.log(weights.clamp_min(1.0e-8)), oracle)
        loss = (
            mean_condition_l1
            + 2.0 * worst_excess
            + 2.0 * condition_regression
            + 0.20 * polar_consistency
            + 0.25 * gain_loss
            + 0.001 * route_loss
        )
        _require(bool(torch.isfinite(loss)), f"{variant}: stage-two loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
        optimizer.step()
        ema.update(router)
        scheduler.step()
        if epoch_index == 0 or (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs:
            row = {
                "phase": "conditional_router_stage_two",
                "variant": variant,
                "epoch": epoch_index + 1,
                "loss": float(loss.detach().cpu()),
                "mean_condition_l1": float(mean_condition_l1.detach().cpu()),
                "worst_excess": float(worst_excess.detach().cpu()),
                "condition_regression": float(condition_regression.detach().cpu()),
                "polar_consistency": float(polar_consistency.detach().cpu()),
                "gain_loss": float(gain_loss.detach().cpu()),
                "route_loss": float(route_loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return history


def _checkpoint_payload(
    *,
    seed: int,
    variant: str,
    weight_variant: str,
    source_r2mt_checkpoint: Path,
    source_r2mt_metadata: Mapping[str, Any],
    warm_metadata: Mapping[str, Any],
    construction: Mapping[str, Any],
    polar_state: Mapping[str, torch.Tensor],
    router_state: Mapping[str, torch.Tensor],
    training_scope: Mapping[str, Any],
    training: Mapping[str, Any],
    checkpoint_selection: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "seed": int(seed),
        "variant": str(variant),
        "weight_variant": str(weight_variant),
        "checkpoint_selection": str(checkpoint_selection),
        "source_r2mt_checkpoint": str(Path(source_r2mt_checkpoint).resolve()),
        "source_r2mt_metadata": dict(source_r2mt_metadata),
        "warm_initialization": dict(warm_metadata),
        "construction": dict(construction),
        "training_scope": dict(training_scope),
        "training": dict(training),
        "polar_expert_state": dict(polar_state),
        "regret_router_state": dict(router_state),
    }


def train_suite(
    *,
    source_r2mt_checkpoint: Path,
    warm_polar_checkpoint: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_root: Path,
    device_name: str,
    seed: int,
    polar_epochs: int,
    router_stage_one_epochs: int,
    router_stage_two_epochs: int,
    polar_batch_size: int,
    router_batch_size: int,
    workers: int,
    ema_decay: float,
    learning_rate: float,
    smooth_max_temperature: float,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _require(not output.exists(), f"unified-reader output already exists: {output}")
    _require(polar_epochs >= 1 and router_stage_one_epochs >= 1 and router_stage_two_epochs >= 1, "epochs must be positive")
    _require(polar_batch_size >= 1 and router_batch_size >= 1 and workers >= 0, "loader sizes are invalid")
    _require(0.0 < ema_decay < 1.0 and learning_rate > 0.0 and smooth_max_temperature > 0.0, "optimizer configuration is invalid")
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
    model = UnifiedPointerReader(bank, polar, router, variant=FULL).to(device)
    initial_router_state = _clone_state(model.regret_router)
    construction = _construction(model.regret_router, model.polar_expert)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)

    polar_history, _polar_terminal_state, polar_ema_state, polar_ema_updates = (
        _train_polar_with_ema(
            model,
            internal.train,
            device=device,
            seed=seed,
            epochs=polar_epochs,
            batch_size=polar_batch_size,
            workers=workers,
            ema_decay=ema_decay,
        )
    )
    model.polar_expert.load_state_dict(polar_ema_state, strict=True)
    cache_path = output / "inner_dev_cache_ema_polar.pt"
    cache, cache_elapsed = _collect_router_cache(
        model,
        internal.dev,
        cache_path=cache_path,
        device=device,
        seed=seed,
        batch_size=polar_batch_size,
        workers=workers,
    )
    cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cache_payload["protocol"] = PROTOCOL
    cache_payload["scope"] = "SyncG inner-development only; EMA polar evidence"
    cache_payload["polar_weight_variant"] = "ema"
    torch.save(cache_payload, cache_path)

    training_scope = {
        "syncg_formal_fit_only": True,
        "polar_inner_train_samples": len(internal.train),
        "router_inner_dev_samples": len(internal.dev),
        "router_inner_dev_rows": len(internal.dev) * len(CONDITIONS),
        "inner_scene_disjoint": True,
        "formal_syncg_holdout_access": False,
        "industrial_image_access": False,
        "industrial_test_only": True,
    }
    results: dict[str, Any] = {}
    for variant in DYNAMIC_VARIANTS:
        variant_started = time.perf_counter()
        transformed = _variant_cache(cache, variant)
        fitted_router = _build_router(
            construction, initial_router_state, device=device
        )
        router_ema = ParameterEMA(fitted_router, decay=ema_decay)
        stage_one = _train_router_stage_one(
            fitted_router,
            transformed,
            variant=variant,
            device=device,
            seed=seed,
            epochs=router_stage_one_epochs,
            batch_size=router_batch_size,
            ema=router_ema,
        )
        stage_two = _train_router_stage_two(
            fitted_router,
            transformed,
            variant=variant,
            conditions=tuple(str(value) for value in CONDITIONS),
            device=device,
            seed=seed,
            epochs=router_stage_two_epochs,
            learning_rate=learning_rate,
            smooth_max_temperature=smooth_max_temperature,
            ema=router_ema,
        )
        active = candidate_mask(variant, device=device)
        arguments = tuple(transformed[name].to(device) for name in ROUTER_ARGUMENT_NAMES)
        candidates = torch.stack(
            (
                transformed["base_mean"],
                transformed["polar_mean"],
                transformed["r2mt_mean"],
            ),
            dim=1,
        ).float().to(device)
        target = transformed["target"].float().to(device)
        condition_index = transformed["condition_index"].long().to(device)
        fitted_router.eval()
        with torch.inference_mode():
            terminal_weights, _gains, _logits = _router_forward(
                fitted_router, arguments, active
            )
        terminal_stats = _condition_statistics(
            terminal_weights,
            candidates,
            target,
            condition_index,
            CONDITIONS,
            active,
        )
        ema_router_state = router_ema.state_dict(fitted_router)
        ema_router = _build_router(construction, ema_router_state, device=device).eval()
        with torch.inference_mode():
            ema_weights, _gains, _logits = _router_forward(
                ema_router, arguments, active
            )
        ema_stats = _condition_statistics(
            ema_weights,
            candidates,
            target,
            condition_index,
            CONDITIONS,
            active,
        )
        common_training = {
            "polar_weight_variant": "ema",
            "polar_ema_decay": ema_decay,
            "polar_ema_updates": polar_ema_updates,
            "router_stage_one_epochs": router_stage_one_epochs,
            "router_stage_two_epochs": router_stage_two_epochs,
            "router_ema_decay": ema_decay,
            "router_ema_updates": router_ema.updates,
            "cache": str(cache_path),
            "cache_scope": "disjoint SyncG inner-development only",
            "candidate_active": [bool(value) for value in active.cpu()],
            "stage_one_history": stage_one,
            "stage_two_history": stage_two,
            "terminal_inner_dev": terminal_stats,
            "ema_inner_dev": ema_stats,
        }
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=False)
        terminal_path = variant_dir / "terminal_router.pt"
        ema_path = variant_dir / "ema.pt"
        torch.save(
            _checkpoint_payload(
                seed=seed,
                variant=variant,
                weight_variant="terminal_router_with_ema_polar",
                source_r2mt_checkpoint=source_r2mt_checkpoint,
                source_r2mt_metadata=bank_metadata,
                warm_metadata=warm_metadata,
                construction=construction,
                polar_state=polar_ema_state,
                router_state=_clone_state(fitted_router),
                training_scope=training_scope,
                training=common_training,
                checkpoint_selection="fixed_epoch_terminal_router",
            ),
            terminal_path,
        )
        torch.save(
            _checkpoint_payload(
                seed=seed,
                variant=variant,
                weight_variant="ema",
                source_r2mt_checkpoint=source_r2mt_checkpoint,
                source_r2mt_metadata=bank_metadata,
                warm_metadata=warm_metadata,
                construction=construction,
                polar_state=polar_ema_state,
                router_state=ema_router_state,
                training_scope=training_scope,
                training=common_training,
                checkpoint_selection="fixed_epoch_parameter_ema",
            ),
            ema_path,
        )
        results[variant] = {
            "terminal_checkpoint": str(terminal_path),
            "ema_checkpoint": str(ema_path),
            "terminal_inner_dev": terminal_stats,
            "ema_inner_dev": ema_stats,
            "elapsed_seconds": time.perf_counter() - variant_started,
        }
        del fitted_router, ema_router
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_dir = output / FIXED_ROUTING
    fixed_dir.mkdir(parents=True, exist_ok=False)
    fixed_path = fixed_dir / "ema.pt"
    fixed_model = UnifiedPointerReader(
        model.expert_bank,
        model.polar_expert,
        _build_router(construction, initial_router_state, device=device),
        variant=FIXED_ROUTING,
    )
    torch.save(
        _checkpoint_payload(
            seed=seed,
            variant=FIXED_ROUTING,
            weight_variant="ema_polar_fixed_router",
            source_r2mt_checkpoint=source_r2mt_checkpoint,
            source_r2mt_metadata=bank_metadata,
            warm_metadata=warm_metadata,
            construction=construction,
            polar_state=polar_ema_state,
            router_state=_clone_state(fixed_model.regret_router),
            training_scope=training_scope,
            training={
                "polar_weight_variant": "ema",
                "polar_ema_decay": ema_decay,
                "polar_ema_updates": polar_ema_updates,
                "router_training": False,
                "routing": "fixed normalized prior",
            },
            checkpoint_selection="fixed_prior_no_adaptive_routing",
        ),
        fixed_path,
    )
    results[FIXED_ROUTING] = {
        "ema_checkpoint": str(fixed_path),
        "learned_router": False,
        "parameter_inventory": parameter_inventory(fixed_model),
    }

    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "seed": seed,
        "source_r2mt_checkpoint": str(Path(source_r2mt_checkpoint).resolve()),
        "training_scope": training_scope,
        "parameter_averaging": {
            "method": "exponential moving average within each training trajectory",
            "decay": ema_decay,
            "polar_updates": polar_ema_updates,
            "independent_seed_mixing": False,
        },
        "polar": {
            "history": polar_history,
            "terminal_state_retained_for_stability_only": True,
            "ema_state_used_by_structural_ablations": True,
            "cache_elapsed_seconds": cache_elapsed,
        },
        "variants": results,
        "stability_comparison": {
            "scope": "conditional router only; shared EMA polar evidence",
            "without_router_parameter_averaging": results[FULL][
                "terminal_checkpoint"
            ],
            "with_router_parameter_averaging": results[FULL]["ema_checkpoint"],
            "same_optimization_trajectory": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r2mt", type=Path, required=True)
    parser.add_argument("--warm-polar", type=Path, default=DEFAULT_WARM_POLAR)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--polar-epochs", type=int, default=5)
    parser.add_argument("--router-stage-one-epochs", type=int, default=40)
    parser.add_argument("--router-stage-two-epochs", type=int, default=120)
    parser.add_argument("--polar-batch-size", type=int, default=64)
    parser.add_argument("--router-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--smooth-max-temperature", type=float, default=2.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_suite(
        source_r2mt_checkpoint=args.source_r2mt,
        warm_polar_checkpoint=args.warm_polar,
        fit_manifest_path=args.fit_manifest,
        outer_split_path=args.outer_split,
        output_root=args.output_root,
        device_name=args.device,
        seed=args.seed,
        polar_epochs=args.polar_epochs,
        router_stage_one_epochs=args.router_stage_one_epochs,
        router_stage_two_epochs=args.router_stage_two_epochs,
        polar_batch_size=args.polar_batch_size,
        router_batch_size=args.router_batch_size,
        workers=args.workers,
        ema_decay=args.ema_decay,
        learning_rate=args.learning_rate,
        smooth_max_temperature=args.smooth_max_temperature,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": result["seed"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DYNAMIC_VARIANTS",
    "ParameterEMA",
    "train_suite",
]
