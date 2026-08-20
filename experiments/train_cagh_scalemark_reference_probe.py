"""Two-fold train-only ScaleMark reference-head probe for frozen PEPD."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from experiments.cagh_net import CAGHNet, differentiable_keypoint_reference_solver
from experiments.cagh_scalemark_reference_head import (
    ScaleMarkReferenceHead,
    scalemark_reference_loss,
)
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.probabilistic_pivot_direction import decode_probabilistic_pivot_direction
from experiments.screen_cagh_factorized_reference_oracle import (
    derive_gt_scalemark_reference,
)
from experiments.screen_cagh_internal_oracle import RESTRICTED_NAMES
from experiments.train_geopepd_progress_v2_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    _progress_inputs_v2,
)
from experiments.train_geopepd_train_only_probe import (
    EXPECTED_FIT,
    _EncoderDataset,
    _by_id,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
    sha256_file,
)
from experiments.train_uhpf_shared_mask_geometry_probe import (
    DEFAULT_STAGE_A_CHECKPOINT,
    _read_jsonl_subset,
    load_scope_records,
    load_visual_model,
)
from experiments.vdn_baseline import transform_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_head_train_only_probe_v1"
SEED = 20260806
EPOCHS = 6
BATCH_SIZE = 64
CACHE_BATCH_SIZE = 64
SHARD_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPETITIONS = 5000
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/cagh_scalemark_reference_head_probe_v1"
DEFAULT_CACHE_ROOT = (
    Path(tempfile.gettempdir())
    / "PointerMeterReaderFastAPI"
    / "cagh_scalemark_reference_head_probe_v1"
)
DEFAULT_COMPARISON = (
    PROJECT_ROOT
    / "artifacts/runs/cagh_internal_oracle_train_only_screen_v1/predictions.jsonl"
)
GT_REFERENCE_ORACLE_NMAE = 0.014219591631253293
MIN_FREE_AFTER_WRITE_BYTES = 8 * 1024**3
GO_THRESHOLDS = {
    "coverage_min": 0.99,
    "full_nmae_max": 0.10,
    "covered_p95_max": 0.30,
    "delta_vs_runtime_pepd_max": -0.20,
    "endpoint_mean_pixels_max": 10.0,
    "endpoint_p95_pixels_max": 24.0,
    "valid_ordered_arc_min": 0.99,
    "posterior_mass_error_max": 5e-6,
    "gt_oracle_reproduction_tolerance": 1e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument("--comparison-predictions", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-cache", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    mode.add_argument("--reevaluate", action="store_true")
    return parser.parse_args()


def _cache_root(path: Path) -> Path:
    root = Path(path).resolve()
    allowed = DEFAULT_CACHE_ROOT.parent.resolve()
    if root != DEFAULT_CACHE_ROOT.resolve() and allowed not in root.parents:
        raise ValueError(f"cache root must remain below {allowed}: {root}")
    if str(root).startswith(str(Path.home())):
        raise ValueError("cache root may not use the user home directory")
    return root


def _free_space_guard(path: Path, planned_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free - int(planned_bytes) < MIN_FREE_AFTER_WRITE_BYTES:
        raise OSError(
            f"insufficient cache volume space: free={free}, planned={planned_bytes}, "
            f"required_reserve={MIN_FREE_AFTER_WRITE_BYTES}"
        )


def _sample_ids_sha256(records: Sequence[Any]) -> str:
    return hashlib.sha256(
        "\n".join(record.sample_id for record in records).encode("utf-8")
    ).hexdigest()


def load_targets(
    args: argparse.Namespace, records: Sequence[Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = {record.sample_id for record in records}
    raw = _by_id(
        _read_jsonl_subset(Path(args.raw_clean), ids, label="scalemark_probe_fit_raw")
    )
    features = _by_id(_read_jsonl(Path(args.fit_features), label="scalemark_probe_fit_features"))
    if set(features) != ids:
        raise ValueError("ScaleMark probe feature inventory differs from fit records")
    endpoint_xy, gt_start, gt_range, folds = [], [], [], []
    group_fold: dict[str, str] = {}
    for record in records:
        metadata = raw[record.sample_id].get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{record.sample_id}: metadata absent")
        _, _, start_degrees, range_degrees = derive_gt_scalemark_reference(
            metadata, record.sample_id
        )
        scale_entry = next(
            item
            for item in metadata.get("keypoints") or []
            if isinstance(item, Mapping)
            and str(item.get("type") or "").casefold() == "scalemark"
        )
        marks = scale_entry["all_kp"]
        start_xy = transform_point(marks[0], record.affine) / 255.0
        end_xy = transform_point(marks[-1], record.affine) / 255.0
        points = np.stack((start_xy, end_xy)).astype(np.float32)
        if not np.isfinite(points).all() or bool(((points < 0.0) | (points > 1.0)).any()):
            raise ValueError(f"{record.sample_id}: ScaleMark endpoints escaped crop")
        fold_name = str(features[record.sample_id].get("support_fold_id") or "")
        if fold_name not in {"support-00", "support-01"}:
            raise ValueError(f"{record.sample_id}: invalid support fold")
        prior = group_fold.setdefault(record.group_id, fold_name)
        if prior != fold_name:
            raise ValueError(f"{record.group_id}: group crosses support folds")
        endpoint_xy.append(points)
        gt_start.append(math.radians(start_degrees))
        gt_range.append(math.radians(range_degrees))
        folds.append(0 if fold_name == "support-00" else 1)
    group_sizes = Counter(record.group_id for record in records)
    weights = np.asarray(
        [len(records) / (len(group_sizes) * group_sizes[record.group_id]) for record in records],
        dtype=np.float32,
    )
    return (
        torch.from_numpy(np.stack(endpoint_xy)).float(),
        torch.tensor(gt_start, dtype=torch.float32),
        torch.tensor(gt_range, dtype=torch.float32),
        torch.tensor(folds, dtype=torch.long),
        torch.from_numpy(weights),
    )


def load_pepd(checkpoint: Path, device: torch.device) -> tuple[CAGHNet, dict[str, Any]]:
    checkpoint = Path(checkpoint).resolve(strict=True)
    assert_train_only_path(checkpoint, label="scalemark_probe_pepd_checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("PEPD checkpoint lacks model_state")
    model = CAGHNet(imagenet_pretrained=False)
    model.load_pepd_state_dict(payload["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model.to(device), {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "strict_pepd_load": True,
    }


@torch.inference_mode()
def build_or_validate_cache(
    model: CAGHNet,
    records: Sequence[Any],
    cache_root: Path,
    *,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    cache_root = _cache_root(cache_root)
    signature = {
        "protocol": PROTOCOL,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_ids_sha256": _sample_ids_sha256(records),
        "samples": len(records),
        "c2_shape": [64, 64, 64],
        "c5_shape": [512, 8, 8],
        "dtype": "float16",
        "shard_size": SHARD_SIZE,
    }
    index_path = cache_root / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("signature") != signature:
            raise ValueError("stale/incompatible ScaleMark feature cache")
        for item in index.get("shards") or []:
            path = cache_root / item["name"]
            if not path.is_file() or path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"missing/truncated ScaleMark cache shard: {path}")
        return index
    estimated = len(records) * (64 * 64 * 64 + 512 * 8 * 8) * 2
    _free_space_guard(cache_root, estimated)
    loader = DataLoader(
        _EncoderDataset(records),
        batch_size=CACHE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    pending: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("indices", "c2", "c5", "pivot_xy", "direction")
    }
    shards: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending["indices"]:
            return
        count = sum(int(value.shape[0]) for value in pending["indices"])
        planned = count * (64 * 64 * 64 + 512 * 8 * 8) * 2 + count * 32
        _free_space_guard(cache_root, planned)
        payload = {name: torch.cat(values, dim=0) for name, values in pending.items()}
        name = f"shard_{len(shards):03d}.pt"
        path = cache_root / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        shards.append({"name": name, "samples": count, "bytes": path.stat().st_size})
        for values in pending.values():
            values.clear()

    model.eval()
    for image, index in loader:
        image = image.to(device, non_blocking=True)
        features = model.forward_multiscale_features(image)
        pooled = model.direction_features(features.c5)
        pivot_logits = model.pivot_head(features.c5)
        decoded = decode_probabilistic_pivot_direction(
            pivot_logits,
            model.vector_head(pooled),
            model.angle_head(pooled),
            model.log_variance_head(pooled),
        )
        pending["indices"].append(index.cpu().long())
        pending["c2"].append(features.c2.cpu().half())
        pending["c5"].append(features.c5.cpu().half())
        pending["pivot_xy"].append((decoded.pivot_xy / 63.0).cpu().float())
        pending["direction"].append(decoded.direction.cpu().float())
        if sum(int(value.shape[0]) for value in pending["indices"]) >= SHARD_SIZE:
            flush()
    flush()
    index = {"schema_version": 1, "signature": signature, "shards": shards}
    _write_json(index_path, index)
    return index


def _load_shard(cache_root: Path, item: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(cache_root / str(item["name"]), map_location="cpu", weights_only=False)
    required = {"indices", "c2", "c5", "pivot_xy", "direction"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("ScaleMark cache shard schema drifted")
    return dict(payload)


def train_fold(
    holdout: int,
    cache_root: Path,
    index: Mapping[str, Any],
    targets: torch.Tensor,
    folds: torch.Tensor,
    group_weight: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[ScaleMarkReferenceHead, list[dict[str, float]]]:
    torch.manual_seed(SEED + holdout)
    head = ScaleMarkReferenceHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = []
    rng = random.Random(SEED + holdout)
    for epoch in range(1, EPOCHS + 1):
        head.train()
        shard_items = list(index["shards"])
        rng.shuffle(shard_items)
        total_loss, samples = 0.0, 0
        for item in shard_items:
            shard = _load_shard(cache_root, item)
            global_index = shard["indices"].long()
            selected = torch.where(folds[global_index] != holdout)[0]
            selected = selected[torch.randperm(len(selected), generator=torch.Generator().manual_seed(SEED + 1000 * epoch + holdout))]
            for offset in range(0, len(selected), BATCH_SIZE):
                local = selected[offset : offset + BATCH_SIZE]
                ids = global_index[local]
                output = head(shard["c2"][local].to(device), shard["c5"][local].to(device))
                loss, _ = scalemark_reference_loss(
                    output,
                    targets[ids].to(device),
                    group_weight=group_weight[ids].to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(local)
                samples += len(local)
        row = {"epoch": float(epoch), "loss": total_loss / samples, "samples": float(samples)}
        history.append(row)
        print(f"ScaleMark fold={holdout} epoch={epoch}/{EPOCHS} loss={row['loss']:.6f}", flush=True)
    return head, history


def reference_from_endpoints(
    start_xy: torch.Tensor,
    end_xy: torch.Tensor,
    pivot_xy: torch.Tensor,
    inverse_linear: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_vector = torch.einsum("bij,bj->bi", inverse_linear.float(), start_xy.float() - pivot_xy.float())
    end_vector = torch.einsum("bij,bj->bi", inverse_linear.float(), end_xy.float() - pivot_xy.float())
    start_norm = torch.linalg.vector_norm(start_vector, dim=1)
    end_norm = torch.linalg.vector_norm(end_vector, dim=1)
    start_angle = torch.remainder(torch.atan2(start_vector[:, 0], -start_vector[:, 1]) - math.pi, 2.0 * math.pi)
    end_angle = torch.remainder(torch.atan2(end_vector[:, 0], -end_vector[:, 1]) - math.pi, 2.0 * math.pi)
    angle_range = torch.remainder(end_angle - start_angle, 2.0 * math.pi)
    valid = (
        (start_norm > 0.02)
        & (end_norm > 0.02)
        & (angle_range > math.radians(10.0))
        & (angle_range < math.radians(350.0))
        & torch.isfinite(start_angle)
        & torch.isfinite(angle_range)
    )
    return start_angle, angle_range, valid


def _metrics(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    covered = [row["errors"][method] for row in rows if row["progress"][method] is not None]
    return {
        "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
        "coverage": len(covered) / len(rows),
        "covered_nmae": float(np.mean(covered)),
        "covered_p95_absolute_error": float(np.quantile(covered, 0.95)),
    }


@torch.inference_mode()
def evaluate_oof(
    heads: Mapping[int, ScaleMarkReferenceHead],
    visual: Any,
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
    assert_train_only_path(comparison_path, label="scalemark_probe_comparison")
    comparison = _by_id(_read_jsonl(comparison_path, label="scalemark_probe_comparison"))
    if set(comparison) != {record.sample_id for record in records}:
        raise ValueError("comparison inventory differs from ScaleMark fit records")
    inverse = torch.from_numpy(np.stack([record.inverse_linear for record in records])).float()
    affine = torch.from_numpy(np.stack([record.affine for record in records])).float()
    target_progress = torch.tensor([record.target_progress for record in records])
    progress_inputs = _progress_inputs_v2(records)
    rows: list[dict[str, Any]] = []
    endpoint_errors: list[float] = []
    mass_errors: list[float] = []
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
                output = head(shard["c2"][local].to(device), shard["c5"][local].to(device))
                pivot = shard["pivot_xy"][local].to(device)
                direction = shard["direction"][local].to(device)
                start, angle_range, valid = reference_from_endpoints(
                    output.start_xy, output.end_xy, pivot, inverse[ids].to(device)
                )
                input_batch = [value[ids].to(device) for value in progress_inputs]
                pooled = F.adaptive_avg_pool2d(
                    shard["c5"][local].to(device).float(), 1
                ).flatten(1)
                candidate_outputs = visual.forward_from_encoder_pooled(
                    pooled,
                    input_batch[0],
                    input_batch[1],
                    start,
                    angle_range,
                    input_batch[4],
                    input_batch[5],
                    valid,
                    transport_runtime_features=input_batch[9],
                )
                oracle_outputs = visual.forward_from_encoder_pooled(
                    pooled,
                    input_batch[0],
                    input_batch[1],
                    gt_start[ids].to(device),
                    gt_range[ids].to(device),
                    input_batch[4],
                    input_batch[5],
                    torch.ones(len(ids), dtype=torch.bool, device=device),
                    transport_runtime_features=input_batch[9],
                )
                predicted_endpoint = torch.stack((output.start_xy, output.end_xy), dim=1)
                endpoint_error = torch.linalg.vector_norm(
                    predicted_endpoint - targets[ids].to(device), dim=2
                ) * 255.0
                endpoint_errors.extend(endpoint_error.flatten().cpu().tolist())
                mass_errors.extend(
                    torch.abs(
                        torch.exp(candidate_outputs.raw_visual_progress_log_probability).sum(1)
                        - 1.0
                    ).cpu().tolist()
                )
                gt_oracle_errors.extend(
                    torch.abs(
                        oracle_outputs.raw_visual_expected_progress.cpu()
                        - target_progress[ids]
                    ).tolist()
                )
                for position, sample_index in enumerate(ids.tolist()):
                    record = records[sample_index]
                    baseline = comparison[record.sample_id]["progress"]
                    candidate = (
                        float(candidate_outputs.raw_visual_expected_progress[position])
                        if bool(candidate_outputs.valid[position])
                        else None
                    )
                    progress = {
                        "scalemark_reference_head": candidate,
                        "runtime_pepd": baseline["pepd"],
                        "v2_transport": baseline["v2_visual"],
                    }
                    errors = {
                        name: abs(float(value) - record.target_progress) if value is not None else 1.0
                        for name, value in progress.items()
                    }
                    rows.append({
                        "schema_version": 1,
                        "protocol": PROTOCOL,
                        "sample_id": record.sample_id,
                        "group_id": record.group_id,
                        "holdout_fold": f"support-{holdout:02d}",
                        "progress": progress,
                        "errors": errors,
                        "endpoint_error_pixels": endpoint_error[position].cpu().tolist(),
                        "reference_valid": bool(candidate_outputs.valid[position]),
                    })
    rows.sort(key=lambda row: row["sample_id"])
    methods = {method: _metrics(rows, method) for method in ("scalemark_reference_head", "runtime_pepd", "v2_transport")}
    bootstrap = grouped_bootstrap_delta(
        rows,
        "scalemark_reference_head",
        "runtime_pepd",
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=SEED,
    )
    by_fold = {
        fold: {
            method: _metrics([row for row in rows if row["holdout_fold"] == fold], method)
            for method in ("scalemark_reference_head", "runtime_pepd")
        }
        for fold in ("support-00", "support-01")
    }
    endpoint = np.asarray(endpoint_errors)
    gt_oracle_nmae = float(np.mean(gt_oracle_errors))
    candidate = methods["scalemark_reference_head"]
    rules = {
        "coverage_at_least_0_99": candidate["coverage"] >= GO_THRESHOLDS["coverage_min"],
        "full_nmae_at_most_0_10": candidate["full_denominator_nmae"] <= GO_THRESHOLDS["full_nmae_max"],
        "covered_p95_at_most_0_30": candidate["covered_p95_absolute_error"] <= GO_THRESHOLDS["covered_p95_max"],
        "delta_vs_runtime_pepd_at_most_minus_0_20": bootstrap["delta_nmae"] <= GO_THRESHOLDS["delta_vs_runtime_pepd_max"],
        "bootstrap_ci95_upper_below_zero": bootstrap["ci95_high"] < 0.0,
        "both_folds_improve": all(by_fold[fold]["scalemark_reference_head"]["full_denominator_nmae"] < by_fold[fold]["runtime_pepd"]["full_denominator_nmae"] for fold in by_fold),
        "endpoint_mean_pixels_at_most_10": float(endpoint.mean()) <= GO_THRESHOLDS["endpoint_mean_pixels_max"],
        "endpoint_p95_pixels_at_most_24": float(np.quantile(endpoint, 0.95)) <= GO_THRESHOLDS["endpoint_p95_pixels_max"],
        "valid_ordered_arc_at_least_0_99": candidate["coverage"] >= GO_THRESHOLDS["valid_ordered_arc_min"],
        "posterior_mass_error_conserved": max(mass_errors, default=0.0) <= GO_THRESHOLDS["posterior_mass_error_max"],
        "gt_reference_oracle_reproduced": abs(gt_oracle_nmae - GT_REFERENCE_ORACLE_NMAE) <= GO_THRESHOLDS["gt_oracle_reproduction_tolerance"],
    }
    return {
        "methods": methods,
        "grouped_bootstrap": {"head_vs_runtime_pepd": bootstrap},
        "grouped_holdout": by_fold,
        "localization": {
            "endpoint_mean_pixels": float(endpoint.mean()),
            "endpoint_p95_pixels": float(np.quantile(endpoint, 0.95)),
        },
        "conservation": {
            "posterior_mass_error_max": max(mass_errors, default=0.0),
            "gt_reference_oracle_nmae": gt_oracle_nmae,
            "nonfinite": 0,
        },
        "decision": {"label": "GO" if all(rules.values()) else "NO_GO", "rules": rules},
    }, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="scalemark_probe_output")
    cache_root = _cache_root(Path(args.cache_root))
    records, masks, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({record.group_id for record in records})) != EXPECTED_FIT:
        raise ValueError("algorithm_fit inventory drifted")
    targets, gt_start, gt_range, folds, weights = load_targets(args, records)
    estimated_cache = len(records) * (64 * 64 * 64 + 512 * 8 * 8) * 2
    validation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "fit": {"samples": len(records), "groups": len({record.group_id for record in records}), "masks": len(masks), "support_00": int((folds == 0).sum()), "support_01": int((folds == 1).sum())},
        "configuration": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "checkpoint_selection": "none; fixed E6", "head_parameters": ScaleMarkReferenceHead().trainable_parameter_count},
        "cache": {"root": str(cache_root), "estimated_bytes": estimated_cache, "shard_size": SHARD_SIZE, "workspace_disk_used_for_cache": False},
        "go_thresholds": GO_THRESHOLDS,
        "input_sha256": identities,
        "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES},
    }
    _write_json(output_dir / "validation.json", validation)
    if args.validate_only:
        return output_dir / "validation.json"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint_identity = load_pepd(Path(args.checkpoint), device)
    index = build_or_validate_cache(
        model,
        records,
        cache_root,
        checkpoint_sha256=checkpoint_identity["sha256"],
        device=device,
    )
    if args.prepare_cache:
        validation["status"] = "cache_prepared"
        validation["cache"]["shards"] = len(index["shards"])
        validation["cache"]["actual_bytes"] = sum(item["bytes"] for item in index["shards"])
        _write_json(output_dir / "validation.json", validation)
        return output_dir / "validation.json"
    if (output_dir / "summary.json").exists():
        raise RuntimeError("formal ScaleMark reference probe already exists")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    heads, histories = {}, {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for holdout in (0, 1):
        head, history = train_fold(
            holdout, cache_root, index, targets, folds, weights, device=device
        )
        heads[holdout] = head
        histories[f"support-{holdout:02d}"] = history
        torch.save(
            {"schema_version": 1, "protocol": PROTOCOL, "holdout_fold": holdout, "epoch": EPOCHS, "head_state": head.state_dict()},
            checkpoint_dir / f"holdout_{holdout}_e6.pt",
        )
    metrics, predictions = evaluate_oof(
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
    prediction_path = output_dir / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    summary = {
        **validation,
        "status": "complete",
        "history": histories,
        "metrics": metrics,
        "checkpoint": checkpoint_identity,
        "cache": {**validation["cache"], "shards": len(index["shards"]), "actual_bytes": sum(item["bytes"] for item in index["shards"]), "retained_after_run": True},
        "artifacts": {"predictions": str(prediction_path), "checkpoints": str(checkpoint_dir)},
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    return output_dir / "summary.json"


if __name__ == "__main__":
    print(run(parse_args()))
