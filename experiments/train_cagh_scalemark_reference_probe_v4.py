"""Conflict-decoupled staged dense-tick ScaleMark reference probe v4."""
from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.cagh_scalemark_reference_head import scalemark_reference_loss
from experiments.cagh_scalemark_reference_head_v3 import DenseTickConditionedReferenceHead, dense_tick_loss
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.screen_cagh_internal_oracle import RESTRICTED_NAMES
from experiments.train_cagh_scalemark_reference_probe import (
    BATCH_SIZE, BOOTSTRAP_REPETITIONS, DEFAULT_CHECKPOINT, DEFAULT_COMPARISON,
    DEFAULT_FIT_FEATURES, DEFAULT_FIT_MANIFEST, DEFAULT_RAW, GO_THRESHOLDS, SEED,
    _cache_root, _load_shard, _metrics, build_or_validate_cache, load_pepd, load_targets,
)
from experiments.train_cagh_scalemark_reference_probe_v2 import evaluate_v2
from experiments.train_cagh_scalemark_reference_probe_v3 import (
    DEFAULT_CACHE, build_head, load_tick_targets, reference_geometry, tick_conservation, weighted,
)
from experiments.train_geopepd_train_only_probe import (
    EXPECTED_FIT, _by_id, _read_jsonl, _write_json, _write_jsonl, grouped_bootstrap_delta,
)
from experiments.train_uhpf_shared_mask_geometry_probe import load_scope_records

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_head_train_only_probe_v4"
STAGE_A_EPOCHS, STAGE_B_EPOCHS, LEARNING_RATE, WEIGHT_DECAY = 6, 12, 4e-4, 1e-4
DEFAULT_V2_PREDICTIONS = PROJECT_ROOT / "artifacts/runs/cagh_scalemark_reference_head_probe_v2/predictions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/cagh_scalemark_reference_head_probe_v4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    p.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    p.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    p.add_argument("--comparison-predictions", type=Path, default=DEFAULT_COMPARISON)
    p.add_argument("--v2-predictions", type=Path, default=DEFAULT_V2_PREDICTIONS)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return p.parse_args()


def set_stage(head: DenseTickConditionedReferenceHead, stage: str) -> list[str]:
    for parameter in head.parameters(): parameter.requires_grad_(False)
    modules = (
        (head.c2_projection, head.c5_projection, head.fusion, head.shared_residual, head.tick_head)
        if stage == "tick" else (head.endpoint_conditioning, head.endpoint_head)
    )
    for module in modules:
        for parameter in module.parameters(): parameter.requires_grad_(True)
    return [name for name, parameter in head.named_parameters() if parameter.requires_grad]


def geometry_loss(out, endpoints, gt_start, gt_range, pivot, inverse, weight):
    endpoint, _ = scalemark_reference_loss(out, endpoints, group_weight=weight)
    start, arc, radii, _ = reference_geometry(out.start_xy, out.end_xy, pivot, inverse)
    angle = 1 - torch.cos(start - gt_start) + F.smooth_l1_loss(
        arc / (2 * math.pi), gt_range / (2 * math.pi), reduction="none", beta=.02)
    arc_margin = (F.relu(math.radians(10) - arc) / math.radians(10)).square() + (
        F.relu(arc - math.radians(350)) / math.radians(10)).square()
    radius_margin = (F.relu(.02 - radii) / .02).square().mean(1)
    return endpoint + .25 * weighted(angle, weight) + .5 * weighted(arc_margin + radius_margin, weight)


def train_fold(holdout: int, cache_root: Path, index: Mapping[str, Any], records: Sequence[Any], endpoints: torch.Tensor,
               ticks: torch.Tensor, gt_start: torch.Tensor, gt_range: torch.Tensor, folds: torch.Tensor,
               weights: torch.Tensor, *, device: torch.device):
    torch.manual_seed(SEED + 300 + holdout); head = build_head().to(device)
    inverse = torch.from_numpy(np.stack([r.inverse_linear for r in records])).float()
    history: dict[str, list[dict[str, float]]] = {"stage_a_tick": [], "stage_b_geometry": []}
    for stage, epochs in (("tick", STAGE_A_EPOCHS), ("geometry", STAGE_B_EPOCHS)):
        names = set_stage(head, stage)
        optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        rng = random.Random(SEED + 300 + holdout + (0 if stage == "tick" else 1000))
        for epoch in range(1, epochs + 1):
            head.train(); items = list(index["shards"]); rng.shuffle(items); total = samples = 0
            for item in items:
                shard = _load_shard(cache_root, item); global_index = shard["indices"].long()
                selected = torch.where(folds[global_index] != holdout)[0]
                gen = torch.Generator().manual_seed(SEED + 10000 * epoch + holdout + (0 if stage == "tick" else 100000))
                selected = selected[torch.randperm(len(selected), generator=gen)]
                for offset in range(0, len(selected), BATCH_SIZE):
                    local = selected[offset:offset+BATCH_SIZE]; ids = global_index[local]
                    out = head(shard["c2"][local].to(device), shard["c5"][local].to(device)); weight = weights[ids].to(device)
                    if stage == "tick":
                        loss, _ = dense_tick_loss(out, ticks[ids].to(device), group_weight=weight)
                    else:
                        loss = geometry_loss(out, endpoints[ids].to(device), gt_start[ids].to(device), gt_range[ids].to(device),
                            shard["pivot_xy"][local].to(device), inverse[ids].to(device), weight)
                    if not bool(torch.isfinite(loss)): raise FloatingPointError(f"v4 {stage} loss is non-finite")
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in head.parameters() if p.requires_grad), 5); optimizer.step()
                    total += float(loss.detach()) * len(local); samples += len(local)
            row = {"epoch": epoch, "loss": total / samples, "samples": samples}; history["stage_a_tick" if stage == "tick" else "stage_b_geometry"].append(row)
            print(f"ScaleMark-v4 fold={holdout} stage={stage} epoch={epoch}/{epochs} loss={row['loss']:.6f}", flush=True)
        if not names: raise RuntimeError(f"v4 {stage} has no trainable parameters")
    return head, history


def add_fixed_fallback(metrics: dict[str, Any], rows: list[dict[str, Any]], records: Sequence[Any], v2_path: Path):
    path = Path(v2_path).resolve(strict=True); assert_train_only_path(path, label="scalemark_v4_v2_predictions")
    v2 = _by_id(_read_jsonl(path, label="scalemark_v4_v2_predictions"))
    if set(v2) != {r.sample_id for r in records}: raise ValueError("v2 fallback inventory drifted")
    old, direct, fallback = "scalemark_reference_head_v2", "scalemark_reference_head_v4", "fixed_v2_runtime_v4_fallback"
    metrics["methods"][direct] = metrics["methods"].pop(old)
    metrics["grouped_bootstrap"]["direct_v4_vs_runtime_pepd"] = metrics["grouped_bootstrap"].pop("head_v2_vs_runtime_pepd")
    for fold in metrics["grouped_holdout"].values(): fold[direct] = fold.pop(old)
    target_by_id = {r.sample_id: r.target_progress for r in records}
    for row in rows:
        row["protocol"] = PROTOCOL; row["progress"][direct] = row["progress"].pop(old); row["errors"][direct] = row["errors"].pop(old)
        prior = v2[row["sample_id"]]["progress"][old]
        chosen = prior if prior is not None else row["progress"]["runtime_pepd"]
        if chosen is None: chosen = row["progress"][direct]
        row["progress"][fallback] = chosen
        row["errors"][fallback] = abs(float(chosen) - target_by_id[row["sample_id"]]) if chosen is not None else 1.0
    metrics["methods"][fallback] = _metrics(rows, fallback)
    metrics["grouped_bootstrap"]["fallback_vs_runtime_pepd"] = grouped_bootstrap_delta(rows, fallback, "runtime_pepd", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED)
    by_fold = {fold: {method: _metrics([r for r in rows if r["holdout_fold"] == fold], method) for method in (fallback, "runtime_pepd")}
               for fold in ("support-00", "support-01")}
    metrics["fallback_grouped_holdout"] = by_fold
    candidate = metrics["methods"][fallback]
    metrics["decision"] = {"label": "NO_GO", "rules": {
        "fallback_full_nmae_at_most_0_10": candidate["full_denominator_nmae"] <= .10,
        "fallback_coverage_at_least_0_99": candidate["coverage"] >= .99,
        "fallback_covered_p95_at_most_0_30": candidate["covered_p95_absolute_error"] <= .30,
        "fallback_both_folds_improve": all(by_fold[f][fallback]["full_denominator_nmae"] < by_fold[f]["runtime_pepd"]["full_denominator_nmae"] for f in by_fold),
        "endpoint_posterior_mass_conserved": metrics["conservation"]["posterior_mass_error_max"] <= GO_THRESHOLDS["posterior_mass_error_max"],
    }}


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).resolve(); assert_train_only_path(output, label="scalemark_v4_output")
    cache_root = _cache_root(Path(args.cache_root)); records, masks, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({r.group_id for r in records})) != EXPECTED_FIT: raise ValueError("fit inventory drifted")
    endpoints, gt_start, gt_range, folds, weights = load_targets(args, records); ticks, tick_xy, tick_valid = load_tick_targets(args, records)
    validation = {"schema_version": 1, "protocol": PROTOCOL, "status": "validated",
        "fit": {"samples": len(records), "groups": len({r.group_id for r in records}), "masks": len(masks), "dense_marks": int(tick_valid.sum())},
        "configuration": {"stage_a_epochs": STAGE_A_EPOCHS, "stage_b_epochs": STAGE_B_EPOCHS, "learning_rate": LEARNING_RATE,
            "checkpoint_selection": "none; fixed A6+B12", "progress_gradient_to_geometry": False, "fallback_order": ["v2", "runtime", "v4"],
            "head_parameters": build_head().trainable_parameter_count},
        "cache": {"root": str(Path(args.cache_root)), "reused_v1_cache": True}, "go_thresholds": GO_THRESHOLDS,
        "input_sha256": identities, "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES}}
    _write_json(output / "validation.json", validation)
    if args.validate_only: return output / "validation.json"
    if (output / "summary.json").exists(): raise RuntimeError("formal v4 already exists")
    device = torch.device(args.device); pepd, checkpoint = load_pepd(Path(args.checkpoint), device)
    index = build_or_validate_cache(pepd, records, cache_root, checkpoint_sha256=checkpoint["sha256"], device=device)
    started = time.time(); heads, history = {}, {}; checkpoint_dir = output / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for holdout in (0, 1):
        head, rows = train_fold(holdout, cache_root, index, records, endpoints, ticks, gt_start, gt_range, folds, weights, device=device)
        heads[holdout] = head; history[f"support-{holdout:02d}"] = rows
        torch.save({"schema_version": 1, "protocol": PROTOCOL, "holdout_fold": holdout, "stage_a_epoch": STAGE_A_EPOCHS,
            "stage_b_epoch": STAGE_B_EPOCHS, "head_state": head.state_dict()}, checkpoint_dir / f"holdout_{holdout}_a6_b12.pt")
    metrics, predictions = evaluate_v2(heads, cache_root, index, records, endpoints, gt_start, gt_range, folds, Path(args.comparison_predictions), device=device)
    add_fixed_fallback(metrics, predictions, records, Path(args.v2_predictions))
    tick = tick_conservation(heads, cache_root, index, ticks, tick_xy, tick_valid, folds, device=device)
    metrics["tick_localization"] = tick; metrics["decision"]["rules"]["tick_localization_conserved"] = tick["localized_and_conserved"]
    metrics["decision"]["label"] = "GO" if all(metrics["decision"]["rules"].values()) else "NO_GO"
    predictions_path = output / "predictions.jsonl"; _write_jsonl(predictions_path, predictions)
    summary = {**validation, "status": "complete", "history": history, "metrics": metrics, "checkpoint": checkpoint,
        "cache": {**validation["cache"], "shards": len(index["shards"]), "actual_bytes": sum(i["bytes"] for i in index["shards"]), "retained_after_run": True},
        "artifacts": {"predictions": str(predictions_path), "checkpoints": str(checkpoint_dir)}, "elapsed_seconds": time.time() - started}
    _write_json(output / "summary.json", summary); return output / "summary.json"


if __name__ == "__main__": print(run(parse_args()))
