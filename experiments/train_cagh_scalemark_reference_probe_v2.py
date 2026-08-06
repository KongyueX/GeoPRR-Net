"""Capacity/progress-supervised v2 ScaleMark reference-head probe."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.cagh_scalemark_reference_head import (
    ScaleMarkReferenceHead,
    scalemark_reference_loss,
)
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.screen_cagh_internal_oracle import RESTRICTED_NAMES
from experiments.train_cagh_scalemark_reference_probe import (
    BATCH_SIZE,
    BOOTSTRAP_REPETITIONS,
    DEFAULT_CACHE_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_COMPARISON,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    GO_THRESHOLDS,
    SEED,
    _cache_root,
    _load_shard,
    _metrics,
    build_or_validate_cache,
    load_pepd,
    load_targets,
    reference_from_endpoints,
)
from experiments.train_geopepd_progress_probe import progress_soft_targets
from experiments.train_geopepd_train_only_probe import (
    EXPECTED_FIT,
    _by_id,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
)
from experiments.train_uhpf_shared_mask_geometry_probe import load_scope_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_head_train_only_probe_v2"
EPOCHS = 18
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 1e-4
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/cagh_scalemark_reference_head_probe_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument("--comparison-predictions", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def build_head() -> ScaleMarkReferenceHead:
    return ScaleMarkReferenceHead(
        hidden_channels=64, coordconv=True, residual_blocks=2
    )


def _weighted(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.sum() * 0.0
    selected = weight[mask]
    return torch.sum(value[mask] * selected) / selected.sum().clamp_min(1e-8)


def train_fold_v2(
    holdout: int,
    cache_root: Path,
    index: Mapping[str, Any],
    records: Sequence[Any],
    targets: torch.Tensor,
    gt_start: torch.Tensor,
    gt_range: torch.Tensor,
    folds: torch.Tensor,
    group_weight: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[ScaleMarkReferenceHead, list[dict[str, float]]]:
    torch.manual_seed(SEED + 100 + holdout)
    head = build_head().to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    inverse = torch.from_numpy(np.stack([record.inverse_linear for record in records])).float()
    affine = torch.from_numpy(np.stack([record.affine for record in records])).float()
    target_progress = torch.tensor([record.target_progress for record in records])
    history: list[dict[str, float]] = []
    shard_rng = random.Random(SEED + holdout)
    for epoch in range(1, EPOCHS + 1):
        head.train()
        items = list(index["shards"])
        shard_rng.shuffle(items)
        totals = {"loss": 0.0, "samples": 0}
        for item in items:
            shard = _load_shard(cache_root, item)
            global_index = shard["indices"].long()
            selected = torch.where(folds[global_index] != holdout)[0]
            generator = torch.Generator().manual_seed(SEED + 10000 * epoch + holdout)
            selected = selected[torch.randperm(len(selected), generator=generator)]
            for offset in range(0, len(selected), BATCH_SIZE):
                local = selected[offset : offset + BATCH_SIZE]
                ids = global_index[local]
                output = head(
                    shard["c2"][local].to(device), shard["c5"][local].to(device)
                )
                endpoint_loss, _ = scalemark_reference_loss(
                    output,
                    targets[ids].to(device),
                    group_weight=group_weight[ids].to(device),
                )
                pivot = shard["pivot_xy"][local].to(device)
                direction = shard["direction"][local].to(device)
                start, angle_range, _ = reference_from_endpoints(
                    output.start_xy,
                    output.end_xy,
                    pivot,
                    inverse[ids].to(device),
                )
                solver = differentiable_keypoint_reference_solver(
                    pivot + 0.25 * direction,
                    pivot,
                    start,
                    angle_range,
                    affine[ids].to(device),
                    torch.ones(len(ids), dtype=torch.bool, device=device),
                )
                target_b = target_progress[ids].to(device).float()
                weight_b = group_weight[ids].to(device).float()
                valid = solver.valid & torch.isfinite(target_b)
                soft = progress_soft_targets(
                    target_b.clamp(0.0, 1.0), progress_bins=72, sigma_bins=1.25
                )
                progress_ce = _weighted(
                    -(soft * solver.progress_log_probability).sum(1), valid, weight_b
                )
                progress_expected = _weighted(
                    F.smooth_l1_loss(
                        solver.expected_progress,
                        target_b,
                        reduction="none",
                        beta=0.02,
                    ),
                    valid,
                    weight_b,
                )
                start_error = 1.0 - torch.cos(start - gt_start[ids].to(device))
                range_error = F.smooth_l1_loss(
                    angle_range / (2.0 * math.pi),
                    gt_range[ids].to(device) / (2.0 * math.pi),
                    reduction="none",
                    beta=0.02,
                )
                angle_loss = _weighted(start_error + range_error, torch.ones_like(valid), weight_b)
                loss = (
                    endpoint_loss
                    + progress_ce
                    + 4.0 * progress_expected
                    + 0.25 * angle_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("v2 ScaleMark loss is non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                totals["loss"] += float(loss.detach()) * len(local)
                totals["samples"] += len(local)
        row = {
            "epoch": float(epoch),
            "loss": totals["loss"] / totals["samples"],
            "samples": float(totals["samples"]),
        }
        history.append(row)
        print(
            f"ScaleMark-v2 fold={holdout} epoch={epoch}/{EPOCHS} loss={row['loss']:.6f}",
            flush=True,
        )
    return head, history


@torch.inference_mode()
def evaluate_v2(
    heads: Mapping[int, ScaleMarkReferenceHead],
    cache_root: Path,
    index: Mapping[str, Any],
    records: Sequence[Any],
    targets: torch.Tensor,
    gt_start: torch.Tensor,
    gt_range: torch.Tensor,
    folds: torch.Tensor,
    comparison_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparison_path = Path(comparison_path).resolve(strict=True)
    assert_train_only_path(comparison_path, label="scalemark_v2_comparison")
    comparison = _by_id(_read_jsonl(comparison_path, label="scalemark_v2_comparison"))
    if set(comparison) != {record.sample_id for record in records}:
        raise ValueError("v2 comparison inventory drifted")
    inverse = torch.from_numpy(np.stack([record.inverse_linear for record in records])).float()
    affine = torch.from_numpy(np.stack([record.affine for record in records])).float()
    target_progress = torch.tensor([record.target_progress for record in records])
    rows: list[dict[str, Any]] = []
    endpoint_errors: list[float] = []
    mass_errors: list[float] = []
    self_consistency: list[float] = []
    gt_oracle_errors: list[float] = []
    for item in index["shards"]:
        shard = _load_shard(cache_root, item)
        global_index = shard["indices"].long()
        for holdout in (0, 1):
            selected = torch.where(folds[global_index] == holdout)[0]
            head = heads[holdout].eval()
            for offset in range(0, len(selected), BATCH_SIZE):
                local = selected[offset : offset + BATCH_SIZE]
                ids = global_index[local]
                output = head(
                    shard["c2"][local].to(device), shard["c5"][local].to(device)
                )
                pivot = shard["pivot_xy"][local].to(device)
                direction = shard["direction"][local].to(device)
                start, angle_range, valid = reference_from_endpoints(
                    output.start_xy,
                    output.end_xy,
                    pivot,
                    inverse[ids].to(device),
                )
                solver = differentiable_keypoint_reference_solver(
                    pivot + 0.25 * direction,
                    pivot,
                    start,
                    angle_range,
                    affine[ids].to(device),
                    valid,
                )
                oracle = differentiable_keypoint_reference_solver(
                    pivot + 0.25 * direction,
                    pivot,
                    gt_start[ids].to(device),
                    gt_range[ids].to(device),
                    affine[ids].to(device),
                    torch.ones(len(ids), dtype=torch.bool, device=device),
                )
                oracle_repeat = differentiable_keypoint_reference_solver(
                    (pivot + 0.25 * direction).clone(),
                    pivot.clone(),
                    gt_start[ids].to(device).clone(),
                    gt_range[ids].to(device).clone(),
                    affine[ids].to(device).clone(),
                    torch.ones(len(ids), dtype=torch.bool, device=device),
                )
                self_consistency.extend(
                    torch.abs(
                        oracle.expected_progress - oracle_repeat.expected_progress
                    ).cpu().tolist()
                )
                gt_oracle_errors.extend(
                    torch.abs(oracle.expected_progress.cpu() - target_progress[ids]).tolist()
                )
                predicted_endpoint = torch.stack((output.start_xy, output.end_xy), dim=1)
                endpoint_error = torch.linalg.vector_norm(
                    predicted_endpoint - targets[ids].to(device), dim=2
                ) * 255.0
                endpoint_errors.extend(endpoint_error.flatten().cpu().tolist())
                mass_errors.extend(solver.posterior_mass_error.cpu().tolist())
                for position, sample_index in enumerate(ids.tolist()):
                    record = records[sample_index]
                    baseline = comparison[record.sample_id]["progress"]
                    candidate = (
                        float(solver.expected_progress[position])
                        if bool(solver.valid[position])
                        else None
                    )
                    progress = {
                        "scalemark_reference_head_v2": candidate,
                        "runtime_pepd": baseline["pepd"],
                        "v2_transport": baseline["v2_visual"],
                    }
                    errors = {
                        name: abs(float(value) - record.target_progress)
                        if value is not None
                        else 1.0
                        for name, value in progress.items()
                    }
                    rows.append(
                        {
                            "schema_version": 1,
                            "protocol": PROTOCOL,
                            "sample_id": record.sample_id,
                            "group_id": record.group_id,
                            "holdout_fold": f"support-{holdout:02d}",
                            "progress": progress,
                            "errors": errors,
                            "endpoint_error_pixels": endpoint_error[position].cpu().tolist(),
                            "reference_valid": bool(solver.valid[position]),
                        }
                    )
    rows.sort(key=lambda row: row["sample_id"])
    candidate_name = "scalemark_reference_head_v2"
    methods = {
        method: _metrics(rows, method)
        for method in (candidate_name, "runtime_pepd", "v2_transport")
    }
    bootstrap = grouped_bootstrap_delta(
        rows,
        candidate_name,
        "runtime_pepd",
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=SEED,
    )
    by_fold = {
        fold: {
            method: _metrics(
                [row for row in rows if row["holdout_fold"] == fold], method
            )
            for method in (candidate_name, "runtime_pepd")
        }
        for fold in ("support-00", "support-01")
    }
    endpoint = np.asarray(endpoint_errors)
    candidate = methods[candidate_name]
    consistency_max = max(self_consistency, default=0.0)
    rules = {
        "coverage_at_least_0_99": candidate["coverage"] >= GO_THRESHOLDS["coverage_min"],
        "full_nmae_at_most_0_10": candidate["full_denominator_nmae"] <= GO_THRESHOLDS["full_nmae_max"],
        "covered_p95_at_most_0_30": candidate["covered_p95_absolute_error"] <= GO_THRESHOLDS["covered_p95_max"],
        "delta_vs_runtime_pepd_at_most_minus_0_20": bootstrap["delta_nmae"] <= GO_THRESHOLDS["delta_vs_runtime_pepd_max"],
        "bootstrap_ci95_upper_below_zero": bootstrap["ci95_high"] < 0.0,
        "both_folds_improve": all(
            by_fold[fold][candidate_name]["full_denominator_nmae"]
            < by_fold[fold]["runtime_pepd"]["full_denominator_nmae"]
            for fold in by_fold
        ),
        "endpoint_mean_pixels_at_most_10": float(endpoint.mean()) <= GO_THRESHOLDS["endpoint_mean_pixels_max"],
        "endpoint_p95_pixels_at_most_24": float(np.quantile(endpoint, 0.95)) <= GO_THRESHOLDS["endpoint_p95_pixels_max"],
        "valid_ordered_arc_at_least_0_99": candidate["coverage"] >= GO_THRESHOLDS["valid_ordered_arc_min"],
        "posterior_mass_error_conserved": max(mass_errors, default=0.0) <= GO_THRESHOLDS["posterior_mass_error_max"],
        "deterministic_gt_oracle_self_consistent": consistency_max <= 1e-7,
    }
    return {
        "methods": methods,
        "grouped_bootstrap": {"head_v2_vs_runtime_pepd": bootstrap},
        "grouped_holdout": by_fold,
        "localization": {
            "endpoint_mean_pixels": float(endpoint.mean()),
            "endpoint_p95_pixels": float(np.quantile(endpoint, 0.95)),
        },
        "conservation": {
            "posterior_mass_error_max": max(mass_errors, default=0.0),
            "deterministic_gt_oracle_nmae": float(np.mean(gt_oracle_errors)),
            "deterministic_repeat_max_abs": consistency_max,
            "nonfinite": 0,
        },
        "decision": {"label": "GO" if all(rules.values()) else "NO_GO", "rules": rules},
    }, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="scalemark_v2_output")
    cache_root = _cache_root(Path(args.cache_root))
    records, masks, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({record.group_id for record in records})) != EXPECTED_FIT:
        raise ValueError("algorithm_fit inventory drifted")
    targets, gt_start, gt_range, folds, weights = load_targets(args, records)
    validation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "fit": {
            "samples": len(records),
            "groups": len({record.group_id for record in records}),
            "masks": len(masks),
            "support_00": int((folds == 0).sum()),
            "support_01": int((folds == 1).sum()),
        },
        "configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "checkpoint_selection": "none; fixed E18",
            "head_parameters": build_head().trainable_parameter_count,
            "coordconv": True,
            "residual_blocks": 2,
            "deployment_progress_supervision": True,
        },
        "cache": {"root": str(cache_root), "reused_v1_cache": True},
        "go_thresholds": GO_THRESHOLDS,
        "input_sha256": identities,
        "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES},
    }
    _write_json(output_dir / "validation.json", validation)
    if args.validate_only:
        return output_dir / "validation.json"
    if (output_dir / "summary.json").exists():
        raise RuntimeError("formal ScaleMark v2 probe already exists")
    device = torch.device(args.device)
    model, checkpoint_identity = load_pepd(Path(args.checkpoint), device)
    index = build_or_validate_cache(
        model,
        records,
        cache_root,
        checkpoint_sha256=checkpoint_identity["sha256"],
        device=device,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    heads, histories = {}, {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for holdout in (0, 1):
        head, history = train_fold_v2(
            holdout,
            cache_root,
            index,
            records,
            targets,
            gt_start,
            gt_range,
            folds,
            weights,
            device=device,
        )
        heads[holdout] = head
        histories[f"support-{holdout:02d}"] = history
        torch.save(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "holdout_fold": holdout,
                "epoch": EPOCHS,
                "head_state": head.state_dict(),
            },
            checkpoint_dir / f"holdout_{holdout}_e18.pt",
        )
    metrics, predictions = evaluate_v2(
        heads,
        cache_root,
        index,
        records,
        targets,
        gt_start,
        gt_range,
        folds,
        Path(args.comparison_predictions),
        device=device,
    )
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    summary = {
        **validation,
        "status": "complete",
        "history": histories,
        "metrics": metrics,
        "checkpoint": checkpoint_identity,
        "cache": {
            **validation["cache"],
            "shards": len(index["shards"]),
            "actual_bytes": sum(item["bytes"] for item in index["shards"]),
            "retained_after_run": True,
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "checkpoints": str(checkpoint_dir),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    return output_dir / "summary.json"


if __name__ == "__main__":
    print(run(parse_args()))
