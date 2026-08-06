"""Frozen two-fold dense-tick-conditioned ScaleMark reference probe v3."""
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

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.cagh_scalemark_reference_head import scalemark_reference_loss
from experiments.cagh_scalemark_reference_head_v3 import (
    DenseTickConditionedReferenceHead,
    dense_tick_heatmaps,
    dense_tick_loss,
)
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.screen_cagh_internal_oracle import RESTRICTED_NAMES
from experiments.train_cagh_scalemark_reference_probe import (
    BATCH_SIZE, DEFAULT_CHECKPOINT, DEFAULT_COMPARISON, DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST, DEFAULT_RAW, GO_THRESHOLDS, SEED, _cache_root,
    _load_shard, build_or_validate_cache, load_pepd, load_targets,
)
from experiments.train_cagh_scalemark_reference_probe_v2 import evaluate_v2
from experiments.train_geopepd_progress_probe import progress_soft_targets
from experiments.train_geopepd_train_only_probe import EXPECTED_FIT, _write_json, _write_jsonl
from experiments.train_uhpf_shared_mask_geometry_probe import _read_jsonl_subset, load_scope_records
from experiments.vdn_baseline import transform_point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_head_train_only_probe_v3"
EPOCHS, LEARNING_RATE, WEIGHT_DECAY = 24, 4e-4, 1e-4
DEFAULT_CACHE = Path(r"C:\pointer_read\cagh_scalemark_reference_head_probe_v1")
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/cagh_scalemark_reference_head_probe_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    p.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    p.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    p.add_argument("--comparison-predictions", type=Path, default=DEFAULT_COMPARISON)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return p.parse_args()


def build_head() -> DenseTickConditionedReferenceHead:
    return DenseTickConditionedReferenceHead(hidden_channels=64, residual_blocks=2)


def load_tick_targets(args: argparse.Namespace, records: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = {record.sample_id for record in records}
    raw = {row["sample_id"]: row for row in _read_jsonl_subset(Path(args.raw_clean), ids, label="scalemark_v3_fit_raw")}
    points: list[np.ndarray] = []
    for record in records:
        metadata = raw[record.sample_id]["metadata"]
        entry = next(item for item in metadata["keypoints"] if str(item.get("type", "")).casefold() == "scalemark")
        xy = np.stack([transform_point(mark, record.affine) / 255.0 for mark in entry["all_kp"]]).astype(np.float32)
        if not 7 <= len(xy) <= 36 or not np.isfinite(xy).all() or bool(((xy < 0) | (xy > 1)).any()):
            raise ValueError(f"{record.sample_id}: invalid dense ScaleMark labels")
        points.append(xy)
    maximum = max(map(len, points))
    padded = torch.zeros((len(points), maximum, 2), dtype=torch.float32)
    valid = torch.zeros((len(points), maximum), dtype=torch.bool)
    heatmaps = torch.empty((len(points), 64, 64), dtype=torch.float16)
    for offset in range(0, len(points), 64):
        for local, xy in enumerate(points[offset : offset + 64]):
            padded[offset + local, : len(xy)] = torch.from_numpy(xy)
            valid[offset + local, : len(xy)] = True
        stop = min(offset + 64, len(points))
        heatmaps[offset:stop] = dense_tick_heatmaps(padded[offset:stop], valid[offset:stop]).half()
    return heatmaps, padded, valid


def reference_geometry(start_xy: torch.Tensor, end_xy: torch.Tensor, pivot: torch.Tensor, inverse: torch.Tensor):
    start_v = torch.einsum("bij,bj->bi", inverse.float(), start_xy.float() - pivot.float())
    end_v = torch.einsum("bij,bj->bi", inverse.float(), end_xy.float() - pivot.float())
    radii = torch.stack((torch.linalg.vector_norm(start_v, dim=1), torch.linalg.vector_norm(end_v, dim=1)), dim=1)
    start = torch.remainder(torch.atan2(start_v[:, 0], -start_v[:, 1]) - math.pi, 2 * math.pi)
    end = torch.remainder(torch.atan2(end_v[:, 0], -end_v[:, 1]) - math.pi, 2 * math.pi)
    arc = torch.remainder(end - start, 2 * math.pi)
    valid = (radii > 0.02).all(1) & (arc > math.radians(10)) & (arc < math.radians(350)) & torch.isfinite(arc)
    return start, arc, radii, valid


def weighted(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / weight.sum().clamp_min(1e-8)


def weighted_cvar(value: torch.Tensor, weight: torch.Tensor, tail: float = 0.20) -> torch.Tensor:
    order = torch.argsort(value.detach(), descending=True)
    value, weight = value[order], weight[order]
    normalized = weight / weight.sum().clamp_min(1e-8)
    before = torch.cumsum(normalized, 0) - normalized
    used = torch.clamp(torch.tensor(tail, device=value.device) - before, min=0.0)
    used = torch.minimum(used, normalized)
    return torch.sum(value * used) / float(tail)


def train_fold(holdout: int, cache_root: Path, index: Mapping[str, Any], records: Sequence[Any], endpoints: torch.Tensor,
               ticks: torch.Tensor, gt_start: torch.Tensor, gt_range: torch.Tensor, folds: torch.Tensor,
               group_weight: torch.Tensor, *, device: torch.device):
    torch.manual_seed(SEED + 200 + holdout)
    head = build_head().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    inverse = torch.from_numpy(np.stack([r.inverse_linear for r in records])).float()
    affine = torch.from_numpy(np.stack([r.affine for r in records])).float()
    targets = torch.tensor([r.target_progress for r in records], dtype=torch.float32)
    history, rng = [], random.Random(SEED + 200 + holdout)
    for epoch in range(1, EPOCHS + 1):
        head.train(); items = list(index["shards"]); rng.shuffle(items); total = samples = 0
        for item in items:
            shard = _load_shard(cache_root, item); global_index = shard["indices"].long()
            selected = torch.where(folds[global_index] != holdout)[0]
            gen = torch.Generator().manual_seed(SEED + 10000 * epoch + holdout)
            selected = selected[torch.randperm(len(selected), generator=gen)]
            for offset in range(0, len(selected), BATCH_SIZE):
                local = selected[offset:offset+BATCH_SIZE]; ids = global_index[local]
                out = head(shard["c2"][local].to(device), shard["c5"][local].to(device))
                w = group_weight[ids].to(device)
                endpoint_loss, _ = scalemark_reference_loss(out, endpoints[ids].to(device), group_weight=w)
                tick_loss, _ = dense_tick_loss(out, ticks[ids].to(device), group_weight=w)
                pivot, direction = shard["pivot_xy"][local].to(device), shard["direction"][local].to(device)
                start, arc, radii, _ = reference_geometry(out.start_xy, out.end_xy, pivot, inverse[ids].to(device))
                # Keep gradients alive outside the deployment-valid 10--350 degree interval.
                train_arc = arc.clamp(1e-4, 2 * math.pi - 1e-4)
                solver = differentiable_keypoint_reference_solver(pivot + .25 * direction, pivot, start, train_arc,
                    affine[ids].to(device), torch.ones(len(ids), dtype=torch.bool, device=device))
                target = targets[ids].to(device)
                soft = progress_soft_targets(target.clamp(0, 1), progress_bins=72, sigma_bins=1.25)
                progress_row = -(soft * solver.progress_log_probability).sum(1) + 4 * F.smooth_l1_loss(
                    solver.expected_progress, target, reduction="none", beta=.02)
                progress_loss = weighted(progress_row, w) + weighted_cvar(progress_row, w)
                angle_row = 1 - torch.cos(start - gt_start[ids].to(device)) + F.smooth_l1_loss(
                    arc / (2 * math.pi), gt_range[ids].to(device) / (2 * math.pi), reduction="none", beta=.02)
                arc_margin = (F.relu(math.radians(10) - arc) / math.radians(10)).square() + (
                    F.relu(arc - math.radians(350)) / math.radians(10)).square()
                radius_margin = (F.relu(.02 - radii) / .02).square().mean(1)
                loss = endpoint_loss + tick_loss + progress_loss + .25 * weighted(angle_row, w) + .5 * weighted(arc_margin + radius_margin, w)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("v3 loss is non-finite")
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 5); optimizer.step()
                total += float(loss.detach()) * len(local); samples += len(local)
        row = {"epoch": epoch, "loss": total / samples, "samples": samples}; history.append(row)
        print(f"ScaleMark-v3 fold={holdout} epoch={epoch}/{EPOCHS} loss={row['loss']:.6f}", flush=True)
    return head, history


@torch.inference_mode()
def tick_conservation(heads, cache_root, index, ticks, tick_xy, tick_valid, folds, *, device):
    mass, point_values, background_values, dice_values = [], [], [], []
    for item in index["shards"]:
        shard = _load_shard(cache_root, item); global_index = shard["indices"].long()
        for holdout in (0, 1):
            selected = torch.where(folds[global_index] == holdout)[0]
            for offset in range(0, len(selected), BATCH_SIZE):
                local = selected[offset:offset+BATCH_SIZE]; ids = global_index[local]
                out = heads[holdout].eval()(shard["c2"][local].to(device), shard["c5"][local].to(device))
                target = ticks[ids].to(device).float(); probability = out.tick_probability[:, 0]
                grid = (tick_xy[ids].to(device) * 2 - 1)[:, :, None, :]
                sampled = F.grid_sample(probability[:, None], grid, align_corners=True)[:, 0, :, 0]
                valid = tick_valid[ids].to(device)
                point_values.extend(sampled[valid].cpu().tolist())
                bg = target < .05; background_values.extend(probability[bg].cpu().tolist())
                dice = (2 * (probability * target).sum((1,2)) + 1e-6) / (probability.sum((1,2)) + target.sum((1,2)) + 1e-6)
                dice_values.extend(dice.cpu().tolist()); mass.extend(out.tick_density_mass_error.flatten().cpu().tolist())
    point, background = float(np.mean(point_values)), float(np.mean(background_values))
    return {"dense_marks": int(tick_valid.sum()), "samples": len(ticks), "tick_density_mass_error_max": max(mass),
            "gt_point_likelihood_mean": point, "background_likelihood_mean": background, "soft_dice_mean": float(np.mean(dice_values)),
            "localized_and_conserved": bool(max(mass) <= 5e-6 and point > background)}


def rename_v2(metrics: dict[str, Any], rows: list[dict[str, Any]]):
    old, new = "scalemark_reference_head_v2", "scalemark_reference_head_v3"
    metrics["methods"][new] = metrics["methods"].pop(old)
    metrics["grouped_bootstrap"]["head_v3_vs_runtime_pepd"] = metrics["grouped_bootstrap"].pop("head_v2_vs_runtime_pepd")
    for fold in metrics["grouped_holdout"].values(): fold[new] = fold.pop(old)
    for row in rows:
        row["protocol"] = PROTOCOL; row["progress"][new] = row["progress"].pop(old); row["errors"][new] = row["errors"].pop(old)


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir).resolve(); assert_train_only_path(output, label="scalemark_v3_output")
    cache_root = _cache_root(Path(args.cache_root)); records, masks, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({r.group_id for r in records})) != EXPECTED_FIT: raise ValueError("fit inventory drifted")
    endpoints, gt_start, gt_range, folds, weights = load_targets(args, records)
    ticks, tick_xy, tick_valid = load_tick_targets(args, records)
    validation = {"schema_version": 1, "protocol": PROTOCOL, "status": "validated",
        "fit": {"samples": len(records), "groups": len({r.group_id for r in records}), "masks": len(masks), "dense_marks": int(tick_valid.sum())},
        "configuration": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "checkpoint_selection": "none; fixed E24",
            "head_parameters": build_head().trainable_parameter_count, "dense_tick_conditioning": True, "progress_cvar_tail": .20},
        "cache": {"root": str(Path(args.cache_root)), "reused_v1_cache": True}, "go_thresholds": GO_THRESHOLDS,
        "input_sha256": identities, "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES}}
    _write_json(output / "validation.json", validation)
    if args.validate_only: return output / "validation.json"
    if (output / "summary.json").exists(): raise RuntimeError("formal v3 already exists")
    device = torch.device(args.device); pepd, checkpoint = load_pepd(Path(args.checkpoint), device)
    index = build_or_validate_cache(pepd, records, cache_root, checkpoint_sha256=checkpoint["sha256"], device=device)
    started = time.time(); heads, history = {}, {}; checkpoint_dir = output / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for holdout in (0, 1):
        checkpoint_path = checkpoint_dir / f"holdout_{holdout}_e24.pt"
        if checkpoint_path.is_file():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("protocol") != PROTOCOL or payload.get("epoch") != EPOCHS or payload.get("holdout_fold") != holdout:
                raise ValueError("incompatible v3 fixed-epoch checkpoint")
            head = build_head().to(device); head.load_state_dict(payload["head_state"]); rows = []
        else:
            head, rows = train_fold(holdout, cache_root, index, records, endpoints, ticks, gt_start, gt_range, folds, weights, device=device)
            torch.save({"schema_version": 1, "protocol": PROTOCOL, "holdout_fold": holdout, "epoch": EPOCHS, "head_state": head.state_dict()}, checkpoint_path)
        heads[holdout] = head; history[f"support-{holdout:02d}"] = rows
    metrics, predictions = evaluate_v2(heads, cache_root, index, records, endpoints, gt_start, gt_range, folds, Path(args.comparison_predictions), device=device)
    rename_v2(metrics, predictions)
    tick_metrics = tick_conservation(heads, cache_root, index, ticks, tick_xy, tick_valid, folds, device=device)
    metrics["tick_localization"] = tick_metrics; metrics["decision"]["rules"]["tick_localization_conserved"] = tick_metrics["localized_and_conserved"]
    metrics["decision"]["label"] = "GO" if all(metrics["decision"]["rules"].values()) else "NO_GO"
    predictions_path = output / "predictions.jsonl"; _write_jsonl(predictions_path, predictions)
    summary = {**validation, "status": "complete", "history": history, "metrics": metrics, "checkpoint": checkpoint,
        "cache": {**validation["cache"], "shards": len(index["shards"]), "actual_bytes": sum(i["bytes"] for i in index["shards"]), "retained_after_run": True},
        "artifacts": {"predictions": str(predictions_path), "checkpoints": str(checkpoint_dir)}, "elapsed_seconds": time.time() - started}
    _write_json(output / "summary.json", summary); return output / "summary.json"


if __name__ == "__main__": print(run(parse_args()))
