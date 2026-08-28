"""Refine PCRT-Meter's router with a six-condition smooth-minimax objective.

The refinement consumes only the already materialized, disjoint SyncG
inner-development cache.  It never opens the formal SyncG evaluation manifest
or any real photograph.  The loss minimizes the worst condition-wise excess
over the best available expert while retaining the v1 router's projective
fusion and continuously favoring polar evidence when no geometric relation is
available.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from experiments.pcrt_meter import ConditionalRegretRouter


PROTOCOL: Final[str] = "pcrt_meter_inner_dev_smooth_minimax_router_v2"
DEFAULT_SOURCE: Final[Path] = Path(
    "artifacts/runs/pcrt_meter/seed_20262020/terminal.pt"
)
DEFAULT_CACHE: Final[Path] = Path(
    "artifacts/runs/pcrt_meter/seed_20262020/inner_dev_router_cache.pt"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/pcrt_meter/seed_20262020/minimax_terminal.pt"
)
ROUTER_ARGUMENT_NAMES: Final[tuple[str, ...]] = (
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
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), f"malformed payload: {path}")
    return dict(payload)


def _build_router(payload: Mapping[str, Any], device: torch.device) -> ConditionalRegretRouter:
    construction = payload.get("construction")
    state = payload.get("regret_router_state")
    _require(isinstance(construction, Mapping), "PCRT construction is missing")
    _require(isinstance(state, Mapping), "PCRT router state is missing")
    router = ConditionalRegretRouter(
        latent_features=int(construction["router_latent_features"]),
        hidden_features=int(construction["router_hidden_features"]),
        maximum_gain=float(construction["maximum_gain"]),
        gain_logit_scale=float(construction["gain_logit_scale"]),
        prior_weights=tuple(float(value) for value in construction["prior_weights"]),
    )
    router.load_state_dict(state, strict=True)
    return router.to(device)


def _condition_statistics(
    weights: torch.Tensor,
    prediction: torch.Tensor,
    candidates: torch.Tensor,
    target: torch.Tensor,
    condition_index: torch.Tensor,
    condition_names: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(condition_names):
        mask = condition_index == index
        expert_errors = torch.abs(candidates[mask] - target[mask, None]).mean(dim=0)
        condition_error = torch.abs(prediction[mask] - target[mask]).mean()
        result[str(name)] = {
            "nmae": float(condition_error.detach().cpu()),
            "expert_nmae": [float(value) for value in expert_errors.detach().cpu()],
            "mean_weights": [
                float(value) for value in weights[mask].mean(dim=0).detach().cpu()
            ],
        }
    return result


def refine_router(
    *,
    source_path: Path,
    cache_path: Path,
    output_path: Path,
    device_name: str,
    epochs: int,
    learning_rate: float,
    smooth_max_temperature: float,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"PCRT minimax output already exists: {output}")
    _require(epochs >= 1, "epochs must be positive")
    _require(learning_rate > 0.0, "learning rate must be positive")
    _require(smooth_max_temperature > 0.0, "smooth-max temperature must be positive")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    source = _load_mapping(source_path)
    cache_payload = _load_mapping(cache_path)
    cache = cache_payload.get("cache")
    conditions = cache_payload.get("conditions")
    _require(isinstance(cache, Mapping), "PCRT router cache tensor mapping is missing")
    _require(isinstance(conditions, Sequence) and len(conditions) == 6, "condition list differs")
    expected_rows = int(cache_payload.get("rows", -1))
    _require(expected_rows == 9456, "inner-development cache row count differs")

    seed = int(source["seed"])
    torch.manual_seed(seed + 40_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 40_001)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    router = _build_router(source, device)
    arguments = tuple(cache[name].to(device) for name in ROUTER_ARGUMENT_NAMES)
    candidates = torch.stack(
        (
            cache["base_mean"],
            cache["polar_mean"],
            cache["r2mt_mean"],
        ),
        dim=1,
    ).float().to(device)
    target = cache["target"].float().to(device)
    condition_index = cache["condition_index"].long().to(device)
    relation_available = cache["relation_available"].bool().to(device)
    raw_scope = ~relation_available

    router.eval()
    with torch.inference_mode():
        initial_weights, _initial_gains, _initial_logits = router(*arguments)
        initial_prediction = (initial_weights * candidates).sum(dim=1)
        reference_condition_l1 = torch.stack(
            tuple(
                torch.abs(initial_prediction[condition_index == index] - target[condition_index == index]).mean()
                for index in range(len(conditions))
            )
        )
        expert_condition_floor = torch.stack(
            tuple(
                torch.abs(
                    candidates[condition_index == index]
                    - target[condition_index == index, None]
                ).mean(dim=0).min()
                for index in range(len(conditions))
            )
        )
        initial_statistics = _condition_statistics(
            initial_weights,
            initial_prediction,
            candidates,
            target,
            condition_index,
            tuple(str(value) for value in conditions),
        )

    optimizer = torch.optim.AdamW(
        router.parameters(), lr=learning_rate, weight_decay=1.0e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch_index in range(epochs):
        router.train(True)
        optimizer.zero_grad(set_to_none=True)
        weights, predicted_gains, _logits = router(*arguments)
        prediction = (weights * candidates).sum(dim=1)
        row_errors = torch.abs(prediction - target)
        condition_l1 = torch.stack(
            tuple(
                row_errors[condition_index == index].mean()
                for index in range(len(conditions))
            )
        )
        condition_excess = condition_l1 - expert_condition_floor
        worst_excess = smooth_max_temperature * torch.logsumexp(
            condition_excess / smooth_max_temperature, dim=0
        )
        mean_condition_l1 = condition_l1.mean()
        condition_regression = torch.relu(
            condition_l1 - reference_condition_l1
        ).mean()
        raw_polar_consistency = torch.abs(
            prediction[raw_scope] - candidates[raw_scope, 1]
        ).mean()
        expert_errors = torch.abs(candidates - target[:, None])
        true_gains = expert_errors[:, :1] - expert_errors[:, 1:]
        gain_loss = F.smooth_l1_loss(
            predicted_gains, true_gains, beta=0.01
        )
        oracle = expert_errors.argmin(dim=1)
        route_loss = F.nll_loss(
            torch.log(weights.clamp_min(1.0e-8)), oracle
        )
        loss = (
            mean_condition_l1
            + 2.0 * worst_excess
            + 2.0 * condition_regression
            + 0.20 * raw_polar_consistency
            + 0.25 * gain_loss
            + 0.001 * route_loss
        )
        _require(bool(torch.isfinite(loss)), "minimax router loss is non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if epoch_index == 0 or (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs:
            row = {
                "epoch": epoch_index + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().cpu()),
                "mean_condition_l1": float(mean_condition_l1.detach().cpu()),
                "worst_excess": float(worst_excess.detach().cpu()),
                "condition_regression": float(condition_regression.detach().cpu()),
                "raw_polar_consistency": float(raw_polar_consistency.detach().cpu()),
                "gain_loss": float(gain_loss.detach().cpu()),
                "route_loss": float(route_loss.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                "condition_l1": [
                    float(value) for value in condition_l1.detach().cpu()
                ],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    router.eval()
    with torch.inference_mode():
        final_weights, _final_gains, _final_logits = router(*arguments)
        final_prediction = (final_weights * candidates).sum(dim=1)
        final_statistics = _condition_statistics(
            final_weights,
            final_prediction,
            candidates,
            target,
            condition_index,
            tuple(str(value) for value in conditions),
        )
    _require(
        all(
            math.isfinite(float(values["nmae"]))
            for values in final_statistics.values()
        ),
        "terminal condition metric is non-finite",
    )

    refined = dict(source)
    refined["schema_version"] = 1
    refined["protocol"] = PROTOCOL
    refined["checkpoint_selection"] = "terminal_fixed_epoch_inner_dev_minimax"
    refined["regret_router_state"] = {
        name: value.detach().cpu().clone()
        for name, value in router.state_dict().items()
    }
    refined["router_refinement"] = {
        "source_checkpoint": str(Path(source_path).resolve()),
        "cache": str(Path(cache_path).resolve()),
        "cache_scope": "disjoint SyncG inner-development only",
        "rows": expected_rows,
        "conditions": [str(value) for value in conditions],
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "smooth_max_temperature": float(smooth_max_temperature),
        "objective": {
            "mean_condition_l1": 1.0,
            "worst_expert_excess": 2.0,
            "v1_condition_regression": 2.0,
            "no_relation_polar_consistency": 0.20,
            "gain_regression": 0.25,
            "oracle_route": 0.001,
        },
        "initial": initial_statistics,
        "terminal": final_statistics,
        "history": history,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    refined["selection_context"] = {
        "formal_syncg_v1_pilot_observed_before_refinement": True,
        "real_photo_v1_metrics_observed_before_refinement": True,
        "formal_syncg_rows_loaded_by_refinement": False,
        "real_photo_rows_loaded_by_refinement": False,
        "real_photo_metrics_used_in_loss_or_checkpoint_selection": False,
        "terminal_epoch_predeclared": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refined, output)
    result = {
        "status": "complete",
        "checkpoint": str(output),
        "elapsed_seconds": refined["router_refinement"]["elapsed_seconds"],
        "initial": initial_statistics,
        "terminal": final_statistics,
    }
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--smooth-max-temperature", type=float, default=2.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = refine_router(
        source_path=args.source,
        cache_path=args.cache,
        output_path=args.output,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        smooth_max_temperature=args.smooth_max_temperature,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "refine_router"]
