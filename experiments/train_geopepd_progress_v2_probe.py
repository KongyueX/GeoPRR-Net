"""Train-only GeoPEPD v2 progress probe with runtime-only visual transport.

This runner is intentionally independent from the trusted v1 artifact.  The
model sees a frozen 24-column runtime transport schema plus the existing raw
MGC quality tensor.  Calibrated/FADR/Transformer values remain comparison-only
and can never enter either model input.
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.calibrated_progress_router import (
    CALIBRATION_FEATURE_NAMES,
    FEATURE_NAMES as ROUTER_FEATURE_NAMES,
)
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd_progress import (
    TRANSPORT_RUNTIME_FEATURE_NAMES,
    GeoPEPDProgressFusionNet,
)
from experiments.probabilistic_pivot_direction import (
    decode_probabilistic_pivot_direction,
)
from experiments.train_geopepd_progress_probe import (
    ProgressProbeRecord,
    _contains_nonfinite,
    _freeze_visual_modules,
    load_progress_probe_records,
    progress_soft_targets,
)
from experiments.train_geopepd_train_only_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    DEFAULT_SELECTION_FEATURES,
    _encoder_cache,
    _finite,
    _progress_from_crop_direction,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "geopepd_progress_train_only_runtime_transport_probe_v2"
SEED = 20260805
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/geopepd_progress_train_only_probe_v2"
SHARED_CACHE_ROOT = PROJECT_ROOT / "artifacts/runs/geopepd_train_only_probe_v1/cache"
FROZEN_REFERENCE_CALIBRATED_NMAE = 0.359577

EXPECTED_TRANSPORT_SCHEMA = (
    "base_progress",
    "vector_progress",
    "base_vector_progress_signed",
    "base_vector_progress_abs",
    "vector_angle_std_fraction",
    "vector_angle_bin_entropy",
    "vector_angle_bin_resultant_length",
    "vector_pivot_spatial_entropy",
    "vector_pivot_top2_margin",
    "vector_direction_raw_norm_log1p",
    "range_angle_fraction",
    "reference_start_and_end",
    "reference_start_only",
    "reference_end_only",
    "reference_default_start_end",
    "meter_bbox_log_aspect",
    "pivot_center_distance_fraction",
    "gate_probability",
    "residual_abs_normalized",
    "residual_std_normalized",
    "v1_confidence",
    "v2_vote_concentration",
    "mask_component_area_ratio",
    "seg_probability_p99",
)
FORBIDDEN_MODEL_FEATURE_NAMES = frozenset(CALIBRATION_FEATURE_NAMES) | {
    "weighted_vector_progress_abs",
    "transformer_vector_progress_abs",
    "weighted_vector_angle_abs_fraction",
    "transformer_vector_angle_abs_fraction",
    "fadr_progress",
    "fadr_prediction",
    "fadr_router_score",
    "ground_truth",
    "target_progress",
    "sample_id",
    "group_id",
    "condition",
}

if tuple(TRANSPORT_RUNTIME_FEATURE_NAMES) != EXPECTED_TRANSPORT_SCHEMA:
    raise RuntimeError("GeoPEPD v2 transport schema drifted from the frozen runner contract")
if set(EXPECTED_TRANSPORT_SCHEMA) & FORBIDDEN_MODEL_FEATURE_NAMES:
    raise RuntimeError("forbidden field entered the frozen transport schema")


@dataclass(frozen=True)
class ProgressProbeRecordV2(ProgressProbeRecord):
    transport_runtime_features: np.ndarray | None = None


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
    parser.add_argument("--soft-target-sigma-bins", type=float, default=1.25)
    parser.add_argument("--expected-smooth-l1-weight", type=float, default=2.0)
    parser.add_argument("--geometry-aux-weight", type=float, default=0.10)
    parser.add_argument("--transport-residual-weight", type=float, default=2.0)
    parser.add_argument("--transport-expected-weight", type=float, default=2.0)
    parser.add_argument("--transport-nll-weight", type=float, default=0.02)
    parser.add_argument("--fusion-regret-weight", type=float, default=2.0)
    parser.add_argument("--transport-boundary-weight", type=float, default=0.10)
    parser.add_argument("--geometry-residual-l2-weight", type=float, default=0.02)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0 trains the full epoch; positive values are smoke-only truncation.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def transport_schema_sha256() -> str:
    payload = json.dumps(EXPECTED_TRANSPORT_SCHEMA, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_mapping(row: Mapping[str, Any]) -> dict[str, float | None]:
    values = row.get("runtime_feature_values")
    if not isinstance(values, list) or len(values) != len(ROUTER_FEATURE_NAMES):
        raise ValueError("runtime_feature_values does not match the frozen 59-column router schema")
    return {
        name: _finite(value)
        for name, value in zip(ROUTER_FEATURE_NAMES, values, strict=True)
    }


def map_transport_runtime_features(
    row: Mapping[str, Any], dial_bbox: Sequence[float]
) -> np.ndarray:
    """Build only the frozen deployable transport token.

    Every access is explicit.  In particular, the six calibration columns,
    all FADR outputs, Transformer-derived columns, labels, and identities are
    unreachable from this mapping.
    """

    runtime = _runtime_mapping(row)
    x1, y1, x2, y2 = map(float, dial_bbox[:4])
    width, height = x2 - x1, y2 - y1
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or width <= 0.0 or height <= 0.0:
        raise ValueError("invalid dial bbox for transport aspect")
    raw_norm = runtime["vector_direction_raw_norm"]
    if raw_norm is not None and raw_norm < 0.0:
        raise ValueError("vector_direction_raw_norm must be non-negative")
    branches = (
        runtime["branch_start_and_end"],
        runtime["branch_start_only"],
        runtime["branch_end_only"],
    )
    default_branch = float(
        (runtime["raw_front_end_success"] or 0.0) > 0.5
        and not any((value or 0.0) > 0.5 for value in branches)
    )
    mapped: dict[str, float | None] = {
        "base_progress": _finite(row.get("base_progress")),
        "vector_progress": _finite(row.get("raw_vector_progress")),
        "base_vector_progress_signed": runtime["base_vector_progress_signed"],
        "base_vector_progress_abs": runtime["base_vector_progress_abs"],
        "vector_angle_std_fraction": runtime["vector_angle_std_fraction"],
        "vector_angle_bin_entropy": runtime["vector_angle_bin_entropy"],
        "vector_angle_bin_resultant_length": runtime["vector_angle_bin_resultant_length"],
        "vector_pivot_spatial_entropy": runtime["vector_pivot_spatial_entropy"],
        "vector_pivot_top2_margin": runtime["vector_pivot_top2_margin"],
        "vector_direction_raw_norm_log1p": None if raw_norm is None else math.log1p(raw_norm),
        "range_angle_fraction": runtime["range_angle_fraction"],
        "reference_start_and_end": branches[0],
        "reference_start_only": branches[1],
        "reference_end_only": branches[2],
        "reference_default_start_end": default_branch,
        "meter_bbox_log_aspect": math.log(width / height),
        "pivot_center_distance_fraction": runtime["pivot_center_distance_fraction"],
        "gate_probability": runtime["gate_probability"],
        "residual_abs_normalized": runtime["residual_abs_normalized"],
        "residual_std_normalized": runtime["residual_std_normalized"],
        "v1_confidence": runtime["v1_confidence"],
        "v2_vote_concentration": runtime["v2_vote_concentration"],
        "mask_component_area_ratio": runtime["mask_component_area_ratio"],
        "seg_probability_p99": runtime["seg_probability_p99"],
    }
    if tuple(mapped) != EXPECTED_TRANSPORT_SCHEMA:
        raise RuntimeError("transport mapping order drifted")
    result = np.asarray(
        [math.nan if mapped[name] is None else float(mapped[name]) for name in EXPECTED_TRANSPORT_SCHEMA],
        dtype=np.float32,
    )
    if result.shape != (24,) or np.isinf(result).any():
        raise ValueError("invalid mapped transport vector")
    return result


def bind_transport_runtime_features(
    records: Sequence[ProgressProbeRecord],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope: str,
) -> list[ProgressProbeRecordV2]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in by_id:
            raise ValueError("transport feature rows contain duplicate/empty sample_id")
        if str(row.get("scope") or "") != expected_scope or str(row.get("condition") or "") != "clean":
            raise ValueError(f"{sample_id}: transport scope/condition drifted")
        if row.get("ground_truth") is not None:
            raise ValueError(f"{sample_id}: transport cache unexpectedly contains a label")
        by_id[sample_id] = row
    if set(by_id) != {record.sample_id for record in records}:
        raise ValueError("transport record/feature inventories differ")
    bound: list[ProgressProbeRecordV2] = []
    for record in records:
        values = dict(vars(record))
        values["transport_runtime_features"] = map_transport_runtime_features(
            by_id[record.sample_id], record.dial_bbox
        )
        bound.append(ProgressProbeRecordV2(**values))
    return bound


def transport_schema_diagnostics(
    records: Sequence[ProgressProbeRecordV2], *, reject_degenerate: bool = True
) -> dict[str, Any]:
    matrix = np.stack([record.transport_runtime_features for record in records]).astype(np.float64)
    if matrix.shape != (len(records), len(EXPECTED_TRANSPORT_SCHEMA)):
        raise ValueError("transport matrix shape drifted")
    columns: dict[str, Any] = {}
    for index, name in enumerate(EXPECTED_TRANSPORT_SCHEMA):
        finite = np.isfinite(matrix[:, index])
        values = matrix[finite, index]
        std = float(np.std(values)) if len(values) else None
        if reject_degenerate and (not len(values) or std is None or std <= 1e-12):
            raise ValueError(f"transport feature {name} is all-missing or zero-variance")
        columns[name] = {
            "finite_samples": int(finite.sum()),
            "finite_fraction": float(finite.mean()),
            "mean": float(np.mean(values)) if len(values) else None,
            "std": std,
            "minimum": float(np.min(values)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        }
    return {
        "feature_names": list(EXPECTED_TRANSPORT_SCHEMA),
        "schema_sha256": transport_schema_sha256(),
        "dimensions": len(EXPECTED_TRANSPORT_SCHEMA),
        "finite_indicators_appended_inside_model": True,
        "columns": columns,
    }


def load_progress_probe_records_v2(
    args: argparse.Namespace,
) -> tuple[list[ProgressProbeRecordV2], list[ProgressProbeRecordV2], dict[str, str]]:
    fit, selection, identities = load_progress_probe_records(args)
    fit_rows = _read_jsonl(Path(args.fit_features), label="v2_fit_transport_features")
    selection_rows = _read_jsonl(Path(args.selection_features), label="v2_selection_transport_features")
    return (
        bind_transport_runtime_features(fit, fit_rows, expected_scope="algorithm_fit"),
        bind_transport_runtime_features(selection, selection_rows, expected_scope="algorithm_selection"),
        identities,
    )


def _progress_inputs_v2(records: Sequence[ProgressProbeRecordV2]) -> tuple[torch.Tensor, ...]:
    from experiments.train_geopepd_progress_probe import _progress_inputs

    base = _progress_inputs(records)
    transport = torch.from_numpy(
        np.stack([record.transport_runtime_features for record in records])
    ).float()
    return (*base, transport)


def _load_model(checkpoint: Path, device: torch.device) -> GeoPEPDProgressFusionNet:
    checkpoint = Path(checkpoint).resolve(strict=True)
    assert_train_only_path(checkpoint, label="pepd_checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("PEPD checkpoint lacks model_state")
    model = GeoPEPDProgressFusionNet(imagenet_pretrained=False)
    model.load_pepd_state_dict(payload["model_state"])
    model.freeze_pepd()
    return model.to(device)


def _weighted_mean(row: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return row.new_zeros(())
    selected = weight.float()[mask]
    return (row.float()[mask] * selected).sum() / selected.sum().clamp_min(1e-8)


def group_balanced_progress_v2_loss(
    outputs: Any,
    target_progress: torch.Tensor,
    group_weight: torch.Tensor,
    *,
    geometry_available: torch.Tensor,
    sigma_bins: float,
    expected_weight: float,
    geometry_aux_weight: float,
    transport_residual_weight: float,
    transport_expected_weight: float,
    transport_nll_weight: float,
    fusion_regret_weight: float,
    transport_boundary_weight: float,
    geometry_residual_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            count = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(f"GeoPEPD v2 {name} has {count} non-finite values")

    required = (
        "progress_log_probability",
        "visual_progress_log_probability",
        "geometry_progress_log_probability",
        "expected_progress",
        "visual_expected_progress",
        "raw_visual_expected_progress",
        "visual_residual_mean",
        "visual_transport_scale",
        "geometry_progress_residual",
    )
    for name in required:
        require_finite(name, getattr(outputs, name).float())
    require_finite("target_progress", target_progress.float())
    require_finite("group_weight", group_weight.float())
    target = target_progress.float().clamp(0.0, 1.0)
    soft = progress_soft_targets(
        target,
        progress_bins=outputs.progress_log_probability.shape[1],
        sigma_bins=sigma_bins,
    )
    valid = outputs.valid.bool() & torch.isfinite(target_progress)
    if not bool(valid.any()):
        raise FloatingPointError("GeoPEPD v2 batch has no valid rows")
    fused_ce = _weighted_mean(
        -(soft * outputs.progress_log_probability.float()).sum(1), valid, group_weight
    )
    expected = _weighted_mean(
        F.smooth_l1_loss(outputs.expected_progress.float(), target, reduction="none", beta=0.02),
        valid,
        group_weight,
    )
    geometry_mask = (
        outputs.geometry_available.bool()
        & torch.as_tensor(geometry_available, device=valid.device).bool()
        & torch.isfinite(target_progress)
    )
    geometry_aux = _weighted_mean(
        -(soft * outputs.geometry_progress_log_probability.float()).sum(1),
        geometry_mask,
        group_weight,
    )
    transport_mask = outputs.transport_available.bool() & torch.isfinite(target_progress)
    if not bool(transport_mask.any()):
        raise FloatingPointError("GeoPEPD v2 batch has no runtime visual-transport rows")
    direct_target = torch.clamp(
        target - outputs.raw_visual_expected_progress.float().detach(), -0.40, 0.40
    )
    direct_error = outputs.visual_residual_mean.float() - direct_target
    transport_residual = _weighted_mean(
        F.smooth_l1_loss(
            outputs.visual_residual_mean.float(), direct_target, reduction="none", beta=0.02
        ),
        transport_mask,
        group_weight,
    )
    transport_expected = _weighted_mean(
        F.smooth_l1_loss(
            outputs.visual_expected_progress.float(), target, reduction="none", beta=0.02
        ),
        transport_mask,
        group_weight,
    )
    scale = outputs.visual_transport_scale.float().clamp(0.01, 0.50)
    transport_nll = _weighted_mean(
        0.5 * (direct_error / scale).square() + torch.log(scale),
        transport_mask,
        group_weight,
    )
    regret_row = F.relu(
        torch.abs(outputs.expected_progress.float() - target)
        - torch.abs(outputs.visual_expected_progress.float() - target)
    )
    fusion_regret = _weighted_mean(regret_row, valid, group_weight)
    transport_boundary = _weighted_mean(
        F.relu(outputs.visual_residual_mean.float().abs() - 0.38).square(),
        transport_mask,
        group_weight,
    )
    geometry_residual_l2 = _weighted_mean(
        outputs.geometry_progress_residual.float().square(), geometry_mask, group_weight
    )
    loss = (
        fused_ce
        + float(expected_weight) * expected
        + float(geometry_aux_weight) * geometry_aux
        + float(transport_residual_weight) * transport_residual
        + float(transport_expected_weight) * transport_expected
        + float(transport_nll_weight) * transport_nll
        + float(fusion_regret_weight) * fusion_regret
        + float(transport_boundary_weight) * transport_boundary
        + float(geometry_residual_l2_weight) * geometry_residual_l2
    )
    components = {
        "fused_soft_ce": fused_ce,
        "expected_smooth_l1": expected,
        "geometry_aux_soft_ce": geometry_aux,
        "transport_direct_residual": transport_residual,
        "transport_expected_smooth_l1": transport_expected,
        "transport_heteroscedastic_nll": transport_nll,
        "fusion_regret": fusion_regret,
        "transport_boundary_penalty": transport_boundary,
        "geometry_residual_l2": geometry_residual_l2,
        "valid_fraction": valid.float().mean(),
        "transport_fraction": transport_mask.float().mean(),
    }
    require_finite("total_loss", loss)
    for name, value in components.items():
        require_finite(name, value)
    return loss, {name: value.detach() for name, value in components.items()}


def _normalization(
    matrix: torch.Tensor, *, label: str, reject_degenerate: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(matrix)
    count = finite.sum(0)
    if reject_degenerate and bool((count == 0).any()):
        raise ValueError(f"{label} contains an all-missing column")
    safe = torch.where(finite, matrix, torch.zeros_like(matrix))
    safe_count = count.clamp_min(1)
    mean = safe.sum(0) / safe_count
    variance = torch.where(finite, (matrix - mean).square(), torch.zeros_like(matrix)).sum(0) / safe_count
    std = torch.sqrt(variance)
    if reject_degenerate and bool((std <= 1e-12).any()):
        raise ValueError(f"{label} contains a non-finite or zero-variance column")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError(f"{label} contains a non-finite normalization statistic")
    # The inherited 37-D MGC quality schema contains a historically all-missing
    # meter-confidence column.  Its finite indicator remains zero, so neutral
    # mean/std values preserve the old fail-safe behavior without introducing
    # a value.  The new 24-D transport schema stays strict and rejects this.
    if not reject_degenerate:
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        std = torch.where((count > 0) & (std > 1e-12), std, torch.ones_like(std))
    return mean, std.clamp_min(1e-6)


def train_stage_a_v2(
    model: GeoPEPDProgressFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProgressProbeRecordV2],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    tensors = _progress_inputs_v2(records)
    geometry, transport = tensors[0], tensors[-1]
    geometry_mean, geometry_std = _normalization(
        geometry, label="geometry features", reject_degenerate=False
    )
    transport_mean, transport_std = _normalization(transport, label="transport runtime features")
    model.set_geometry_normalization(geometry_mean, geometry_std)
    model.set_transport_runtime_normalization(transport_mean, transport_std)
    model.freeze_pepd()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("GeoPEPD v2 has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.learning_rate) * 0.05,
    )
    loader = DataLoader(
        TensorDataset(pooled, *tensors),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        _freeze_visual_modules(model)
        totals: Counter[str] = Counter()
        for batch_index, batch in enumerate(loader, start=1):
            (
                pooled_b,
                geometry_b,
                mgc_b,
                start_b,
                range_b,
                affine_b,
                geometry_available_b,
                reference_available_b,
                target_b,
                weight_b,
                transport_b,
            ) = [value.to(device) for value in batch]
            drop = geometry_available_b & (
                torch.rand(geometry_available_b.shape, device=device) < float(args.geometry_dropout)
            )
            outputs = model.forward_from_encoder_pooled(
                pooled_b,
                geometry_b,
                mgc_b,
                start_b,
                range_b,
                affine_b,
                geometry_available_b,
                reference_available_b,
                transport_runtime_features=transport_b,
                geometry_drop_mask=drop,
            )
            if bool((drop & ~outputs.transport_available.bool()).any()):
                raise RuntimeError("MGC dropout incorrectly disabled visual transport")
            loss, parts = group_balanced_progress_v2_loss(
                outputs,
                target_b,
                weight_b,
                geometry_available=geometry_available_b,
                sigma_bins=float(args.soft_target_sigma_bins),
                expected_weight=float(args.expected_smooth_l1_weight),
                geometry_aux_weight=float(args.geometry_aux_weight),
                transport_residual_weight=float(args.transport_residual_weight),
                transport_expected_weight=float(args.transport_expected_weight),
                transport_nll_weight=float(args.transport_nll_weight),
                fusion_regret_weight=float(args.fusion_regret_weight),
                transport_boundary_weight=float(args.transport_boundary_weight),
                geometry_residual_l2_weight=float(args.geometry_residual_l2_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite GeoPEPD v2 gradient: {name}")
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(f"non-finite GeoPEPD v2 parameter: {name}")
            size = int(pooled_b.shape[0])
            totals["samples"] += size
            totals["batches"] += 1
            totals["loss"] += float(loss.detach()) * size
            for name, value in parts.items():
                totals[name] += float(value) * size
            if int(args.max_train_batches) > 0 and batch_index >= int(args.max_train_batches):
                break
        scheduler.step()
        if totals["samples"] <= 0:
            raise RuntimeError("GeoPEPD v2 epoch processed no samples")
        row = {
            "epoch": float(epoch),
            "batches": float(totals["batches"]),
            "samples": float(totals["samples"]),
            "loss": totals["loss"] / totals["samples"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for name in parts:
            row[name] = totals[name] / totals["samples"]
        history.append(row)
        print(
            f"GeoPEPD-v2 epoch={epoch}/{args.epochs} batches={int(row['batches'])} "
            f"loss={row['loss']:.6f} transport={row['transport_fraction']:.3f}",
            flush=True,
        )
    return history


def go_decision_v2(
    metrics: Mapping[str, Mapping[str, float]],
    bootstrap: Mapping[str, Mapping[str, float]],
    residual_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    full = metrics["geopepd_progress_v2_full"]
    transport = metrics["geopepd_progress_v2_transport_only"]
    reference = metrics["reference_calibrated_pepd"]
    reference_comparison = bootstrap["geopepd_progress_v2_full_vs_reference_calibrated_pepd"]
    rules = {
        "lower_than_frozen_reference_calibrated_nmae_0.359577": float(full["full_denominator_nmae"]) < FROZEN_REFERENCE_CALIBRATED_NMAE,
        "lower_than_measured_reference_calibrated_pepd": float(full["full_denominator_nmae"]) < float(reference["full_denominator_nmae"]),
        "paired_group_bootstrap_reference_ci95_upper_below_zero": float(reference_comparison["ci95_high"]) < 0.0,
        "full_better_than_transport_only": float(full["full_denominator_nmae"]) < float(transport["full_denominator_nmae"]),
        "coverage_noninferiority_margin_0.005": float(full["coverage"]) >= float(reference["coverage"]) - 0.005,
        "visual_residual_boundary_fraction_below_0.10": float(residual_diagnostics["visual_residual_mean"]["fraction_abs_ge_0_38"]) < 0.10,
    }
    return {
        "label": "GO" if all(rules.values()) else "NO_GO",
        "rules": rules,
        "scope": "train-only algorithm_selection development gate; not confirmatory evidence",
    }


@torch.inference_mode()
def evaluate_v2(
    model: GeoPEPDProgressFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProgressProbeRecordV2],
    *,
    device: torch.device,
    batch_size: int,
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _progress_inputs_v2(records)
    rows: list[dict[str, Any]] = []
    model.eval()
    for offset in range(0, len(records), batch_size):
        stop = min(len(records), offset + batch_size)
        batch = [value[offset:stop].to(device) for value in tensors]
        (
            geometry_b,
            mgc_b,
            start_b,
            range_b,
            affine_b,
            geometry_available_b,
            reference_available_b,
            _,
            _,
            transport_b,
        ) = batch
        outputs = model.forward_from_encoder_pooled(
            pooled[offset:stop].to(device),
            geometry_b,
            mgc_b,
            start_b,
            range_b,
            affine_b,
            geometry_available_b,
            reference_available_b,
            transport_runtime_features=transport_b,
        )
        for name in (
            "progress_log_probability",
            "visual_progress_log_probability",
            "expected_progress",
            "visual_expected_progress",
            "raw_visual_expected_progress",
            "visual_residual_mean",
            "visual_progress_residual",
            "geometry_progress_residual",
            "visual_reliability",
            "geometry_reliability",
        ):
            if not bool(torch.isfinite(getattr(outputs, name)).all()):
                raise FloatingPointError(f"non-finite GeoPEPD v2 evaluation output: {name}")
        dummy_pivot = torch.zeros(
            (stop - offset, 1, 1, 1), device=device, dtype=outputs.direction_raw.dtype
        )
        pepd_direction = decode_probabilistic_pivot_direction(
            dummy_pivot,
            outputs.direction_raw,
            outputs.visual_angle_logits,
            outputs.visual_log_variance_raw,
        ).direction.cpu().numpy()
        full = outputs.expected_progress.cpu().numpy()
        transport_only = outputs.visual_expected_progress.cpu().numpy()
        valid = outputs.valid.cpu().numpy().astype(bool)
        transport_available = outputs.transport_available.cpu().numpy().astype(bool)
        visual_residual_mean = outputs.visual_residual_mean.cpu().numpy()
        visual_residual = outputs.visual_progress_residual.cpu().numpy()
        correction_probability = outputs.visual_correction_probability.cpu().numpy()
        geometry_reliability = outputs.geometry_reliability.cpu().numpy()
        for local, record in enumerate(records[offset:stop]):
            progress = {
                "geopepd_progress_v2_full": float(full[local]) if valid[local] else None,
                "geopepd_progress_v2_transport_only": float(transport_only[local]) if transport_available[local] else None,
                "pepd": _progress_from_crop_direction(record, pepd_direction[local]),
                "mgc": record.mgc_progress,
                "reference_calibrated_pepd": record.reference_calibrated_pepd_progress,
                "fadr": record.fadr_progress,
            }
            errors = {
                name: abs(float(value) - record.target_progress)
                if value is not None and math.isfinite(float(value))
                else 1.0
                for name, value in progress.items()
            }
            rows.append(
                {
                    "schema_version": 2,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "ground_truth_progress": record.target_progress,
                    "progress": progress,
                    "errors": errors,
                    "visual_residual_mean": float(visual_residual_mean[local]),
                    "visual_progress_residual": float(visual_residual[local]),
                    "visual_correction_probability": float(correction_probability[local]),
                    "geometry_reliability": float(geometry_reliability[local]),
                    "geometry_available": bool(record.geometry_available),
                    "transport_available": bool(transport_available[local]),
                }
            )
    method_metrics: dict[str, dict[str, float | int]] = {}
    methods = (
        "geopepd_progress_v2_full",
        "geopepd_progress_v2_transport_only",
        "pepd",
        "mgc",
        "reference_calibrated_pepd",
        "fadr",
    )
    for method in methods:
        covered = sum(row["progress"][method] is not None for row in rows)
        method_metrics[method] = {
            "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
            "coverage": covered / len(rows),
            "covered_samples": covered,
        }
    comparisons = (
        "geopepd_progress_v2_transport_only",
        "pepd",
        "mgc",
        "reference_calibrated_pepd",
        "fadr",
    )
    bootstrap = {
        f"geopepd_progress_v2_full_vs_{baseline}": grouped_bootstrap_delta(
            rows,
            "geopepd_progress_v2_full",
            baseline,
            repetitions=bootstrap_repetitions,
        )
        for baseline in comparisons
    }
    residuals = np.asarray(
        [row["visual_residual_mean"] for row in rows if row["transport_available"]],
        dtype=np.float64,
    )
    residual_diagnostics = {
        "visual_residual_mean": {
            "effective_samples": int(len(residuals)),
            "mean": float(np.mean(residuals)),
            "mean_absolute": float(np.mean(np.abs(residuals))),
            "p95_absolute": float(np.quantile(np.abs(residuals), 0.95)),
            "fraction_abs_ge_0_38": float(np.mean(np.abs(residuals) >= 0.38)),
        }
    }
    return {
        "methods": method_metrics,
        "grouped_bootstrap": bootstrap,
        "residual_diagnostics": residual_diagnostics,
        "decision": go_decision_v2(method_metrics, bootstrap, residual_diagnostics),
    }, rows


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    if int(args.epochs) <= 0 or int(args.batch_size) <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 <= float(args.geometry_dropout) < 1.0 or int(args.max_train_batches) < 0:
        raise ValueError("invalid geometry dropout/max train batches")
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="v2_output_dir")
    fit, selection, identities = load_progress_probe_records_v2(args)
    fit_transport = transport_schema_diagnostics(fit)
    selection_transport = transport_schema_diagnostics(selection)
    restricted = {
        name: 0
        for name in (
            "public_samples",
            "test_samples",
            "field_samples",
            "sealed_samples",
            "confirmatory_samples",
            "confirmation_a_samples",
            "confirmation_b_samples",
        )
    }
    preparation = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "prepared" if args.prepare_only else "running",
        "fit": {"samples": len(fit), "groups": len({record.group_id for record in fit})},
        "selection": {"samples": len(selection), "groups": len({record.group_id for record in selection})},
        "fit_selection_sample_overlap": len({r.sample_id for r in fit} & {r.sample_id for r in selection}),
        "fit_selection_group_overlap": len({r.group_id for r in fit} & {r.group_id for r in selection}),
        "transport_runtime_schema": {
            "fit": fit_transport,
            "selection": selection_transport,
            "normalization_source": "algorithm_fit only; selection never re-estimates normalization",
            "forbidden_model_features": sorted(FORBIDDEN_MODEL_FEATURE_NAMES),
            "calibrated_fadr_transformer_enter_model": False,
        },
        "geometry_center_binding": {
            "method_label": "true MGC",
            "source_field": "base_progress",
            "reference_calibrated_pepd_used_as_model_input": False,
        },
        "development_baseline_bindings": {
            "pepd": "frozen legacy visual direction decoded through the shared reference",
            "mgc": "raw base_progress",
            "reference_calibrated_pepd": "comparison only",
            "fadr": "comparison only",
        },
        "shared_encoder_cache_root": str(SHARED_CACHE_ROOT),
        "input_sha256": identities,
        "restricted_data_use": restricted,
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
    model = _load_model(Path(args.checkpoint), device)
    fit_pooled = _encoder_cache(
        model,
        fit,
        SHARED_CACHE_ROOT / "algorithm_fit_clean_encoder.pt",
        split="algorithm_fit",
        identities=identities,
        device=device,
        batch_size=int(args.cache_batch_size),
        workers=int(args.workers),
    )
    selection_pooled = _encoder_cache(
        model,
        selection,
        SHARED_CACHE_ROOT / "algorithm_selection_clean_encoder.pt",
        split="algorithm_selection",
        identities=identities,
        device=device,
        batch_size=int(args.cache_batch_size),
        workers=int(args.workers),
    )
    history = train_stage_a_v2(model, fit_pooled, fit, args, device)
    metrics, predictions = evaluate_v2(
        model,
        selection_pooled,
        selection,
        device=device,
        batch_size=int(args.batch_size),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
    )
    checkpoint_path = output_dir / "stage_a_last.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "seed": SEED,
            "stage": "A_frozen_pepd_runtime_visual_transport_and_bounded_mgc_fusion",
            "model_state": model.state_dict(),
            "transport_schema": list(EXPECTED_TRANSPORT_SCHEMA),
            "transport_schema_sha256": transport_schema_sha256(),
            "input_sha256": identities,
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    summary = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "complete",
        "seed": SEED,
        "stage": "A_frozen_pepd_runtime_visual_transport_and_bounded_mgc_fusion",
        "configuration": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "max_train_batches": int(args.max_train_batches),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "geometry_dropout": float(args.geometry_dropout),
            "loss": "group-balanced fused CE/expected + direct clipped visual-transport residual/expected/NLL + fusion regret + geometry auxiliary",
            "visual_residual_target_clip": [-0.40, 0.40],
            "fusion_regret_reference": "transport-only visual expected progress",
            "runner_reliability_bias_override": False,
        },
        "fit": preparation["fit"],
        "evaluation": preparation["selection"],
        "history": history,
        "metrics": metrics,
        "transport_runtime_schema": preparation["transport_runtime_schema"],
        "artifacts": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path), "rows": len(predictions)},
        },
        "input_sha256": identities,
        "elapsed_seconds": time.time() - started,
        "selection_labels_used_for_model_fit": 0,
        "restricted_data_use": restricted,
    }
    if _contains_nonfinite(summary):
        raise FloatingPointError("GeoPEPD v2 summary contains a non-finite value")
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


if __name__ == "__main__":
    parsed = parse_args()
    try:
        print(run(parsed))
    except Exception as error:
        failure_dir = Path(parsed.output_dir).resolve()
        assert_train_only_path(failure_dir, label="v2_failure_output_dir")
        _write_json(
            failure_dir / "failure.json",
            {
                "schema_version": 2,
                "protocol": PROTOCOL,
                "status": "failed",
                "summary_complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
