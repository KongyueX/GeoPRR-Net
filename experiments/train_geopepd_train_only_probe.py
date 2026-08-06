"""Leakage-safe clean-condition Stage-A probe for GeoPEPD.

The only visual initialization accepted by this runner is the frozen
``darm-selection-pepd`` checkpoint.  Its encoder is evaluated once to cache
deterministic 512-D pooled features.  Stage A then trains only GeoPEPD's new
geometry/fusion heads on ``algorithm_fit`` and evaluates exactly once on
``algorithm_selection``.  Confirmation/public/test/field inputs are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.calibrated_progress_router import FEATURE_NAMES as ROUTER_FEATURE_NAMES
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd import (
    GEOMETRY_FEATURE_NAMES,
    GeoPEPDCircularFusionNet,
    decode_geopepd,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import (
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import (
    affine_for_dial,
    image_angle_from_direction,
    reading_from_pointer_angle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "geopepd_train_only_clean_stage_a_probe_v1"
SEED = 20260805
EXPECTED_FIT = (5253, 240)
EXPECTED_SELECTION = (1360, 60)

JOB_ROOT = PROJECT_ROOT / "artifacts/runs/darm_fadr_train_only_v1/selection_v1/jobs/darm-selection-pepd"
DEFAULT_CHECKPOINT = JOB_ROOT / "best.pt"
DEFAULT_FIT_MANIFEST = JOB_ROOT / "fit_manifest.jsonl"
DEFAULT_EVALUATION_MANIFEST = JOB_ROOT / "evaluation_manifest.jsonl"
DEFAULT_RAW = PROJECT_ROOT / "artifacts/runs/darm_fadr_train_only_v1/component_cache_v1/raw/clean/predictions.jsonl"
FEATURE_ROOT = PROJECT_ROOT / "artifacts/runs/darm_fadr_train_only_v1/component_models_v1/features"
DEFAULT_FIT_FEATURES = FEATURE_ROOT / "algorithm_fit/clean/rows.jsonl"
DEFAULT_SELECTION_FEATURES = FEATURE_ROOT / "algorithm_selection/clean/rows.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/geopepd_train_only_probe_v1"


@dataclass(frozen=True)
class ProbeRecord:
    sample_id: str
    group_id: str
    image_path: str
    dial_bbox: tuple[float, float, float, float]
    affine: np.ndarray
    inverse_linear: np.ndarray
    target_direction: np.ndarray
    target_progress: float
    scale_start: float
    scale_end: float
    reference_start: float | None
    reference_range: float | None
    geometry_features: np.ndarray
    geometry_direction: np.ndarray
    geometry_available: bool
    mgc_progress: float | None
    fadr_progress: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument("--selection-features", type=Path, default=DEFAULT_SELECTION_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--geometry-dropout", type=float, default=0.20)
    parser.add_argument("--stage-a-gate-bias", type=float, default=-2.2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    path = Path(path).resolve(strict=True)
    assert_train_only_path(path, label=label)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{number} is not an object")
            sample_id = str(value.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"{label}:{number} duplicate/empty sample_id")
            seen.add(sample_id)
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def map_runtime_geometry_features(row: Mapping[str, Any]) -> np.ndarray:
    """Map the frozen 59-value router vector to GeoPEPD's fixed 37 fields."""

    values = row.get("runtime_feature_values")
    if not isinstance(values, list) or len(values) != len(ROUTER_FEATURE_NAMES):
        raise ValueError("runtime_feature_values does not match router schema")
    runtime = {name: _finite(value) for name, value in zip(ROUTER_FEATURE_NAMES, values)}
    direct = {
        name: runtime.get(name)
        for name in GEOMETRY_FEATURE_NAMES
        if name in runtime
    }
    direct.update(
        {
            "base_gate_probability": runtime["gate_probability"],
            "base_residual_normalized": runtime["residual_abs_normalized"],
            "base_residual_std_normalized": runtime["residual_std_normalized"],
            "base_correction_applied": runtime["correction_applied"],
            "geometry_v1_available": float(runtime["geometry_v1_vector_progress_abs"] is not None),
            "geometry_v2_available": float(runtime["geometry_v2_vector_progress_abs"] is not None),
            "base_available": float(_finite(row.get("base_progress")) is not None),
            "branch_default_start_end": float(
                runtime["raw_front_end_success"] is not None
                and runtime["raw_front_end_success"] > 0.5
                and not any(
                    (runtime[name] or 0.0) > 0.5
                    for name in ("branch_start_and_end", "branch_start_only", "branch_end_only")
                )
            ),
        }
    )
    missing_schema = set(GEOMETRY_FEATURE_NAMES) - set(direct)
    if missing_schema:
        raise RuntimeError(f"unmapped GeoPEPD features: {sorted(missing_schema)}")
    array = np.asarray(
        [math.nan if direct[name] is None else float(direct[name]) for name in GEOMETRY_FEATURE_NAMES],
        dtype=np.float32,
    )
    if np.isinf(array).any():
        raise ValueError("infinite mapped geometry feature")
    return array


def _pointer_points(metadata: Mapping[str, Any], sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    for item in metadata.get("keypoints") or []:
        if isinstance(item, Mapping) and str(item.get("type") or "").casefold() == "pointer":
            tip, tail = item.get("outside_kp"), item.get("origin_kp")
            if isinstance(tip, Sequence) and isinstance(tail, Sequence) and len(tip) >= 2 and len(tail) >= 2:
                return np.asarray(tip[:2], np.float32), np.asarray(tail[:2], np.float32)
    raise ValueError(f"{sample_id}: pointer keypoints absent")


def _transform_direction(direction: np.ndarray, linear: np.ndarray) -> np.ndarray:
    result = np.asarray(linear, np.float64) @ np.asarray(direction, np.float64)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm <= 1e-10:
        raise ValueError("direction transform collapsed")
    return (result / norm).astype(np.float32)


def _records(
    manifest_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    raw_by_id: Mapping[str, Mapping[str, Any]],
    *,
    expected_scope: str,
    expected_inventory: tuple[int, int],
) -> list[ProbeRecord]:
    features = _by_id(feature_rows)
    manifest = _by_id(manifest_rows)
    if set(features) != set(manifest):
        raise ValueError(f"{expected_scope}: manifest/feature sample inventory differs")
    records: list[ProbeRecord] = []
    for sample_id in sorted(manifest):
        source, feature = manifest[sample_id], features[sample_id]
        raw = raw_by_id.get(sample_id)
        if raw is None:
            raise ValueError(f"{expected_scope}: raw row missing {sample_id}")
        group_id = str(source.get("group_id") or "")
        if (
            group_id != str(feature.get("group_id") or "")
            or group_id != str(raw.get("group_id") or "")
            or str(feature.get("scope")) != expected_scope
            or str(feature.get("condition")) != "clean"
            or feature.get("ground_truth") is not None
        ):
            raise ValueError(f"{sample_id}: scope/group/label contract drifted")
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{sample_id}: raw metadata absent")
        bbox = metadata.get("dial_bbox")
        if not isinstance(bbox, Sequence) or len(bbox) < 4:
            raise ValueError(f"{sample_id}: dial bbox absent")
        dial_bbox = tuple(map(float, bbox[:4]))
        affine = affine_for_dial(dial_bbox, output_size=256, expansion=1.25).astype(np.float32)
        inverse_linear = np.linalg.inv(affine[:, :2].astype(np.float64)).astype(np.float32)
        tip, tail = _pointer_points(metadata, sample_id)
        target = _transform_direction(tip - tail, affine[:, :2])
        ground_truth = _finite(raw.get("ground_truth"))
        scale_start = _finite(feature.get("scale_start"))
        scale_end = _finite(feature.get("scale_end"))
        if ground_truth is None or scale_start is None or scale_end is None or abs(scale_end - scale_start) <= 1e-12:
            raise ValueError(f"{sample_id}: invalid label/scale")
        target_progress = (ground_truth - scale_start) / (scale_end - scale_start)
        raw_features = raw.get("features") if isinstance(raw.get("features"), Mapping) else {}
        start = _finite(raw_features.get("startAngle"))
        angle_range = _finite(raw_features.get("disAngle"))
        calibrated = _finite(feature.get("calibrated_progress"))
        geometry_available = bool(calibrated is not None and start is not None and angle_range is not None and abs(angle_range) > 1e-8)
        geometry_direction = np.zeros(2, dtype=np.float32)
        if geometry_available:
            theta = math.radians(float(start + calibrated * angle_range))
            geometry_direction = _transform_direction(
                np.asarray([-math.sin(theta), math.cos(theta)], np.float32),
                affine[:, :2],
            )
        records.append(
            ProbeRecord(
                sample_id=sample_id,
                group_id=group_id,
                image_path=str(Path(str(source["image_path"])).resolve()),
                dial_bbox=dial_bbox,
                affine=affine,
                inverse_linear=inverse_linear,
                target_direction=target,
                target_progress=float(target_progress),
                scale_start=float(scale_start),
                scale_end=float(scale_end),
                reference_start=start,
                reference_range=angle_range,
                geometry_features=map_runtime_geometry_features(feature),
                geometry_direction=geometry_direction,
                geometry_available=geometry_available,
                mgc_progress=calibrated,
                fadr_progress=_finite(feature.get("fadr_progress")),
            )
        )
    inventory = (len(records), len({record.group_id for record in records}))
    if inventory != expected_inventory:
        raise ValueError(f"{expected_scope}: inventory {inventory} != {expected_inventory}")
    return records


def load_probe_records(args: argparse.Namespace) -> tuple[list[ProbeRecord], list[ProbeRecord], dict[str, str]]:
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(strict=True),
        "fit_manifest": Path(args.fit_manifest).resolve(strict=True),
        "evaluation_manifest": Path(args.evaluation_manifest).resolve(strict=True),
        "raw_clean": Path(args.raw_clean).resolve(strict=True),
        "fit_features": Path(args.fit_features).resolve(strict=True),
        "selection_features": Path(args.selection_features).resolve(strict=True),
    }
    for label, path in paths.items():
        assert_train_only_path(path, label=label)
    if paths["checkpoint"] != DEFAULT_CHECKPOINT.resolve():
        raise ValueError("GeoPEPD probe accepts only the frozen darm-selection-pepd best.pt")
    fit_manifest = _read_jsonl(paths["fit_manifest"], label="fit_manifest")
    evaluation_manifest = _read_jsonl(paths["evaluation_manifest"], label="evaluation_manifest")
    raw_rows = _read_jsonl(paths["raw_clean"], label="raw_clean")
    fit_features = _read_jsonl(paths["fit_features"], label="fit_features")
    selection_features = _read_jsonl(paths["selection_features"], label="selection_features")
    fit = _records(fit_manifest, fit_features, _by_id(raw_rows), expected_scope="algorithm_fit", expected_inventory=EXPECTED_FIT)
    selection = _records(evaluation_manifest, selection_features, _by_id(raw_rows), expected_scope="algorithm_selection", expected_inventory=EXPECTED_SELECTION)
    if {r.sample_id for r in fit} & {r.sample_id for r in selection} or {r.group_id for r in fit} & {r.group_id for r in selection}:
        raise ValueError("fit/selection leakage detected")
    identities = {label: sha256_file(path) for label, path in paths.items()}
    return fit, selection, identities


class _EncoderDataset(Dataset):
    def __init__(self, records: Sequence[ProbeRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = cv2.imread(record.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"failed to read {record.image_path}")
        crop = cv2.warpAffine(image, record.affine, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return normalized_rgb_tensor(crop), index


def _load_model(checkpoint: Path, device: torch.device, gate_bias: float) -> GeoPEPDCircularFusionNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("PEPD checkpoint lacks model_state")
    model = GeoPEPDCircularFusionNet(imagenet_pretrained=False, initial_geometry_logit=-6.0)
    model.load_pepd_state_dict(payload["model_state"])
    model.freeze_pepd()
    with torch.no_grad():
        model.geometry_reliability_head.bias.fill_(float(gate_bias))
    return model.to(device)


@torch.inference_mode()
def _encoder_cache(
    model: GeoPEPDCircularFusionNet,
    records: Sequence[ProbeRecord],
    path: Path,
    *,
    split: str,
    identities: Mapping[str, str],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    expected_ids = [r.sample_id for r in records]
    signature = {
        "protocol": PROTOCOL,
        "split": split,
        "checkpoint_sha256": identities["checkpoint"],
        "sample_ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
        "samples": len(records),
        "encoder_dim": 512,
    }
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(cached, Mapping) and cached.get("signature") == signature and cached.get("sample_ids") == expected_ids:
            pooled = cached.get("encoder_pooled")
            if isinstance(pooled, torch.Tensor) and pooled.shape == (len(records), 512) and torch.isfinite(pooled).all():
                return pooled.float()
        raise ValueError(f"stale/incompatible encoder cache: {path}")
    loader = DataLoader(_EncoderDataset(records), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    output = torch.empty((len(records), 512), dtype=torch.float32)
    model.eval()
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            features = model.encoder(images)
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        output[indices.long()] = pooled.float().cpu()
    if not torch.isfinite(output).all():
        raise RuntimeError("non-finite encoder cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"signature": signature, "sample_ids": expected_ids, "encoder_pooled": output}, temporary)
    os.replace(temporary, path)
    return output


def _soft_targets(direction: torch.Tensor, bins: int = 72, sigma_bins: float = 1.25) -> torch.Tensor:
    angle = torch.atan2(direction[:, 1], direction[:, 0])
    centers = torch.arange(bins, device=direction.device, dtype=direction.dtype) * (2.0 * math.pi / bins)
    delta = torch.atan2(torch.sin(centers[None] - angle[:, None]), torch.cos(centers[None] - angle[:, None]))
    sigma = sigma_bins * (2.0 * math.pi / bins)
    target = torch.exp(-0.5 * (delta / sigma).square())
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _direction_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits.float(), dim=1)
    centers = torch.arange(logits.shape[1], device=logits.device, dtype=probability.dtype) * (2.0 * math.pi / logits.shape[1])
    return F.normalize(torch.stack(((probability * torch.cos(centers)).sum(1), (probability * torch.sin(centers)).sum(1)), 1), dim=1, eps=1e-8)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def _tensor_inputs(records: Sequence[ProbeRecord]) -> tuple[torch.Tensor, ...]:
    geometry = torch.from_numpy(np.stack([r.geometry_features for r in records]))
    direction = torch.from_numpy(np.stack([r.geometry_direction for r in records]))
    available = torch.tensor([r.geometry_available for r in records], dtype=torch.bool)
    target = torch.from_numpy(np.stack([r.target_direction for r in records]))
    counts = Counter(r.group_id for r in records)
    weights = torch.tensor([1.0 / counts[r.group_id] for r in records], dtype=torch.float32)
    weights *= len(weights) / weights.sum()
    return geometry, direction, available, target, weights


def train_stage_a(
    model: GeoPEPDCircularFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProbeRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    geometry, direction, available, target, weights = _tensor_inputs(records)
    finite = torch.isfinite(geometry)
    safe = torch.where(finite, geometry, torch.zeros_like(geometry))
    count = finite.sum(0).clamp_min(1)
    mean = safe.sum(0) / count
    variance = torch.where(finite, (geometry - mean).square(), torch.zeros_like(geometry)).sum(0) / count
    std = torch.sqrt(variance.clamp_min(1e-6))
    model.set_geometry_normalization(mean, std)
    model.freeze_pepd()
    with torch.no_grad():
        model.geometry_reliability_head.bias.fill_(float(args.stage_a_gate_bias))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.learning_rate) * 0.05)
    dataset = TensorDataset(pooled, geometry, direction, available, target, weights)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, generator=generator, num_workers=0)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for module in (model.encoder, model.pivot_head, model.direction_features, model.vector_head, model.angle_head, model.log_variance_head):
            module.eval()
        totals = Counter()
        for batch in loader:
            pooled_b, geometry_b, direction_b, available_b, target_b, weight_b = [value.to(device) for value in batch]
            drop = available_b & (torch.rand(available_b.shape, device=device) < float(args.geometry_dropout))
            outputs = model.forward_from_encoder_pooled(pooled_b, geometry_b, direction_b, available_b, geometry_drop_mask=drop)
            soft = _soft_targets(target_b)
            fused_ce = -(soft * F.log_softmax(outputs.fused_angle_logits, dim=1)).sum(1)
            fused_cos = 1.0 - (_direction_from_logits(outputs.fused_angle_logits) * target_b).sum(1).clamp(-1, 1)
            geometry_ce = -(soft * F.log_softmax(outputs.geometry_angle_logits, dim=1)).sum(1)
            geometry_weight = weight_b * available_b.float()
            geometry_aux = _weighted_mean(geometry_ce, geometry_weight) if bool(available_b.any()) else fused_ce.new_zeros(())
            loss = _weighted_mean(fused_ce, weight_b) + 0.5 * _weighted_mean(fused_cos, weight_b) + 0.25 * geometry_aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            n = int(pooled_b.shape[0])
            totals["samples"] += n
            totals["loss"] += float(loss.detach()) * n
            totals["gate"] += float(outputs.geometry_weight.detach().mean()) * n
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "loss": totals["loss"] / totals["samples"],
            "mean_geometry_weight": totals["gate"] / totals["samples"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(f"GeoPEPD Stage-A epoch={epoch}/{args.epochs} loss={record['loss']:.6f} gate={record['mean_geometry_weight']:.4f}", flush=True)
    return history


def _progress_from_crop_direction(record: ProbeRecord, direction: np.ndarray) -> float | None:
    if record.reference_start is None or record.reference_range is None:
        return None
    image_direction = _transform_direction(direction, record.inverse_linear)
    pointer_angle = image_angle_from_direction(image_direction)
    _, progress = reading_from_pointer_angle(
        pointer_angle,
        start_angle=record.reference_start,
        range_angle=record.reference_range,
        scale_start=record.scale_start,
        scale_end=record.scale_end,
    )
    return float(progress)


def _angle_error(predicted: np.ndarray | None, target: np.ndarray) -> float | None:
    if predicted is None:
        return None
    cosine = float(np.clip(np.dot(predicted, target) / max(np.linalg.norm(predicted) * np.linalg.norm(target), 1e-12), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def grouped_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int = SEED,
) -> dict[str, float]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(row)
    groups = sorted(by_group)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        candidate_errors, baseline_errors = [], []
        for position in chosen:
            for row in by_group[groups[int(position)]]:
                candidate_errors.append(float(row["errors"][candidate]))
                baseline_errors.append(float(row["errors"][baseline]))
        samples[index] = np.mean(candidate_errors) - np.mean(baseline_errors)
    observed = float(np.mean([row["errors"][candidate] for row in rows]) - np.mean([row["errors"][baseline] for row in rows]))
    return {"delta_nmae": observed, "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)), "repetitions": int(repetitions)}


@torch.inference_mode()
def evaluate(
    model: GeoPEPDCircularFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProbeRecord],
    *,
    device: torch.device,
    batch_size: int,
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometry, geometry_direction, available, _, _ = _tensor_inputs(records)
    model.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        outputs = model.forward_from_encoder_pooled(
            pooled[start:stop].to(device),
            geometry[start:stop].to(device),
            geometry_direction[start:stop].to(device),
            available[start:stop].to(device),
        )
        geopepd = decode_geopepd(outputs).direction.cpu().numpy()
        pepd = decode_probabilistic_pivot_direction(*outputs.legacy_visual_tuple()).direction.cpu().numpy()
        gate = outputs.geometry_weight.cpu().numpy()
        for local, record in enumerate(records[start:stop]):
            geo_direction = geopepd[local].astype(np.float64)
            pepd_direction = pepd[local].astype(np.float64)
            mgc_direction = record.geometry_direction.astype(np.float64) if record.geometry_available else None
            fadr_direction = None
            if record.fadr_progress is not None and record.reference_start is not None and record.reference_range is not None:
                theta = math.radians(record.reference_start + record.fadr_progress * record.reference_range)
                fadr_direction = _transform_direction(np.asarray([-math.sin(theta), math.cos(theta)], np.float32), record.affine[:, :2]).astype(np.float64)
            progress = {
                "geopepd": _progress_from_crop_direction(record, geo_direction),
                "pepd": _progress_from_crop_direction(record, pepd_direction),
                "mgc": record.mgc_progress,
                "fadr": record.fadr_progress,
            }
            errors = {name: (abs(value - record.target_progress) if value is not None and math.isfinite(value) else 1.0) for name, value in progress.items()}
            angles = {
                "geopepd": _angle_error(geo_direction, record.target_direction),
                "pepd": _angle_error(pepd_direction, record.target_direction),
                "mgc": _angle_error(mgc_direction, record.target_direction),
                "fadr": _angle_error(fadr_direction, record.target_direction),
            }
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "ground_truth_progress": record.target_progress,
                    "progress": progress,
                    "errors": errors,
                    "angle_error_degrees": angles,
                    "geometry_weight": float(gate[local]),
                    "geometry_available": record.geometry_available,
                }
            )
    metrics: dict[str, Any] = {}
    for method in ("geopepd", "pepd", "mgc", "fadr"):
        angle_values = [float(row["angle_error_degrees"][method]) for row in rows if row["angle_error_degrees"][method] is not None]
        covered = sum(row["progress"][method] is not None for row in rows)
        metrics[method] = {
            "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
            "coverage": covered / len(rows),
            "covered_samples": covered,
            "angle_mae_degrees": float(np.mean(angle_values)) if angle_values else None,
            "angle_coverage": len(angle_values) / len(rows),
        }
    bootstrap = {
        f"geopepd_vs_{baseline}": grouped_bootstrap_delta(rows, "geopepd", baseline, repetitions=bootstrap_repetitions)
        for baseline in ("pepd", "mgc", "fadr")
    }
    return {"methods": metrics, "grouped_bootstrap": bootstrap}, rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    if int(args.epochs) <= 0 or not 0.0 <= float(args.geometry_dropout) < 1.0:
        raise ValueError("invalid epochs/geometry dropout")
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="output_dir")
    fit, selection, identities = load_probe_records(args)
    preparation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "prepared" if args.prepare_only else "running",
        "fit": {"samples": len(fit), "groups": len({r.group_id for r in fit})},
        "selection": {"samples": len(selection), "groups": len({r.group_id for r in selection})},
        "fit_selection_sample_overlap": 0,
        "fit_selection_group_overlap": 0,
        "input_sha256": identities,
        "restricted_data_use": {name: 0 for name in ("public_samples", "test_samples", "field_samples", "sealed_samples", "confirmatory_samples", "confirmation_a_samples", "confirmation_b_samples")},
    }
    _write_json(output_dir / "preparation.json", preparation)
    if args.prepare_only:
        return output_dir / "preparation.json"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = _load_model(Path(args.checkpoint).resolve(), device, float(args.stage_a_gate_bias))
    fit_pooled = _encoder_cache(model, fit, output_dir / "cache/algorithm_fit_clean_encoder.pt", split="algorithm_fit", identities=identities, device=device, batch_size=int(args.cache_batch_size), workers=int(args.workers))
    selection_pooled = _encoder_cache(model, selection, output_dir / "cache/algorithm_selection_clean_encoder.pt", split="algorithm_selection", identities=identities, device=device, batch_size=int(args.cache_batch_size), workers=int(args.workers))
    history = train_stage_a(model, fit_pooled, fit, args, device)
    metrics, predictions = evaluate(model, selection_pooled, selection, device=device, batch_size=int(args.batch_size), bootstrap_repetitions=int(args.bootstrap_repetitions))
    checkpoint_path = output_dir / "stage_a_last.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "seed": SEED,
            "stage": "A_frozen_pepd_new_heads_only",
            "model_state": model.state_dict(),
            "input_sha256": identities,
            "gate_initialization": {"constructor_bias": -6.0, "runner_override_bias": float(args.stage_a_gate_bias), "reason": "avoid sigmoid saturation while retaining PEPD-dominant initialization"},
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "seed": SEED,
        "stage": "A_frozen_pepd_new_heads_only",
        "configuration": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "geometry_dropout": float(args.geometry_dropout),
            "gate_constructor_bias": -6.0,
            "gate_runner_override_bias": float(args.stage_a_gate_bias),
            "loss": "group-balanced fused circular soft-CE + 0.5 cosine + 0.25 geometry auxiliary soft-CE",
        },
        "fit": preparation["fit"],
        "evaluation": preparation["selection"],
        "history": history,
        "metrics": metrics,
        "artifacts": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path), "rows": len(predictions)},
        },
        "input_sha256": identities,
        "elapsed_seconds": time.time() - started,
        "selection_labels_used_for_model_fit": 0,
        "restricted_data_use": preparation["restricted_data_use"],
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


if __name__ == "__main__":
    print(run(parse_args()))
