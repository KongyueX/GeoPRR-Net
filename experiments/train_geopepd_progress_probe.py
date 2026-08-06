"""Clean, train-only Stage-A probe for progress-posterior GeoPEPD.

This is deliberately separate from the first circular-direction probe.  It
reuses that probe's deterministic pooled-512 caches but trains and evaluates a
posterior over physical dial progress in [0, 1].  Only algorithm_fit labels are
used for fitting; algorithm_selection is opened once for development metrics.
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

from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd_progress import GeoPEPDProgressFusionNet
from experiments.probabilistic_pivot_direction import decode_probabilistic_pivot_direction
from experiments.train_geopepd_train_only_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    DEFAULT_SELECTION_FEATURES,
    GEOMETRY_FEATURE_NAMES,
    ProbeRecord,
    _encoder_cache,
    _finite,
    _progress_from_crop_direction,
    _read_jsonl,
    _tensor_inputs,
    _transform_direction,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
    load_probe_records,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "geopepd_progress_train_only_clean_stage_a_probe_v1"
SEED = 20260805
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/geopepd_progress_train_only_probe_v1"
SHARED_CACHE_ROOT = PROJECT_ROOT / "artifacts/runs/geopepd_train_only_probe_v1/cache"


@dataclass(frozen=True)
class ProgressProbeRecord(ProbeRecord):
    """Correctly separated component readings used by the progress probe."""

    reference_calibrated_pepd_progress: float | None = None


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (float, int)):
        return not math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, Sequence):
        return any(_contains_nonfinite(item) for item in value)
    return False


def invalidate_existing_nonfinite_run(output_dir: Path) -> Path | None:
    """Preserve and explicitly invalidate the completed NaN Stage-A artifact."""

    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.is_file():
        return None
    original_bytes = summary_path.read_bytes()
    value = json.loads(original_bytes.decode("utf-8"))
    if not isinstance(value, dict) or not _contains_nonfinite(value):
        return None
    backup_path = Path(output_dir) / "summary.invalidated-original.json"
    if not backup_path.exists():
        temporary = backup_path.with_suffix(backup_path.suffix + ".tmp")
        temporary.write_bytes(original_bytes)
        os.replace(temporary, backup_path)
    backup_sha = sha256_file(backup_path)
    invalidation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "invalidated_nonfinite_training",
        "trustworthy_result": False,
        "reason": (
            "epoch 1 and all later loss components were non-finite; masked "
            "reductions multiplied invalid missing-reference intermediates by zero"
        ),
        "preserved_original_summary": {
            "path": str(backup_path.resolve()),
            "sha256": backup_sha,
        },
        "preserved_checkpoint": str((Path(output_dir) / "stage_a_last.pt").resolve()),
        "preserved_predictions": str((Path(output_dir) / "predictions.jsonl").resolve()),
    }
    _write_json(Path(output_dir) / "invalidation.json", invalidation)
    value["status"] = "invalidated_nonfinite_training"
    value["trustworthy_result"] = False
    value["invalidation"] = invalidation
    _write_json(summary_path, value)
    return Path(output_dir) / "invalidation.json"


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
    parser.add_argument("--geometry-aux-weight", type=float, default=0.25)
    parser.add_argument("--residual-l2-weight", type=float, default=0.05)
    parser.add_argument("--residual-soft-clip", type=float, default=0.05)
    parser.add_argument("--residual-soft-clip-weight", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def _load_model(checkpoint: Path, device: torch.device) -> GeoPEPDProgressFusionNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("PEPD checkpoint lacks model_state")
    model = GeoPEPDProgressFusionNet(imagenet_pretrained=False)
    model.load_pepd_state_dict(payload["model_state"])
    model.freeze_pepd()
    # MGC is the strongest clean development component.  Start biased toward
    # geometry, but not at a saturated softmax, so reliability can move.
    with torch.no_grad():
        model.reliability_head.bias.copy_(torch.tensor([-1.0, 1.0]))
    return model.to(device)


def bind_mask_geometry_progress(
    records: Sequence[ProbeRecord],
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    expected_scope: str,
) -> list[ProgressProbeRecord]:
    """Bind MGC exclusively to ``base_progress`` from the component cache.

    ``calibrated_progress`` is the reference-conditioned PEPD component used
    inside FADR.  It is intentionally never read here: treating it as mask
    geometry would leak the visual expert into the geometry center and make
    the MGC label scientifically false.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for row in feature_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in by_id:
            raise ValueError("MGC binding feature rows contain duplicate/empty sample_id")
        if str(row.get("scope") or "") != expected_scope or str(row.get("condition") or "") != "clean":
            raise ValueError(f"{sample_id}: MGC binding scope/condition drifted")
        by_id[sample_id] = row
    if set(by_id) != {record.sample_id for record in records}:
        raise ValueError("MGC binding record/feature inventories differ")

    corrected: list[ProgressProbeRecord] = []
    for record in records:
        row = by_id[record.sample_id]
        base_progress = _finite(row.get("base_progress"))
        base_prediction = _finite(row.get("base_prediction"))
        if base_progress is not None and base_prediction is not None:
            reconstructed = record.scale_start + base_progress * (
                record.scale_end - record.scale_start
            )
            if not math.isclose(reconstructed, base_prediction, abs_tol=1e-6):
                raise ValueError(f"{record.sample_id}: base progress/prediction mismatch")
        available = bool(
            base_progress is not None
            and 0.0 <= base_progress <= 1.0
            and record.reference_start is not None
            and record.reference_range is not None
            and abs(record.reference_range) > 1e-8
        )
        direction = np.zeros(2, dtype=np.float32)
        if available:
            theta = math.radians(
                float(record.reference_start + base_progress * record.reference_range)
            )
            direction = _transform_direction(
                np.asarray([-math.sin(theta), math.cos(theta)], np.float32),
                record.affine[:, :2],
            )
        values = dict(vars(record))
        values.update(
            {
                "mgc_progress": base_progress,
                "geometry_available": available,
                "geometry_direction": direction,
                # Kept only as a separately named development baseline.  This
                # field is never forwarded to GeoPEPDProgressFusionNet.
                "reference_calibrated_pepd_progress": _finite(
                    row.get("calibrated_progress")
                ),
            }
        )
        corrected.append(ProgressProbeRecord(**values))
    return corrected


def load_progress_probe_records(
    args: argparse.Namespace,
) -> tuple[list[ProgressProbeRecord], list[ProgressProbeRecord], dict[str, str]]:
    fit, selection, identities = load_probe_records(args)
    fit_features = _read_jsonl(Path(args.fit_features), label="progress_fit_features")
    selection_features = _read_jsonl(
        Path(args.selection_features), label="progress_selection_features"
    )
    return (
        bind_mask_geometry_progress(
            fit, fit_features, expected_scope="algorithm_fit"
        ),
        bind_mask_geometry_progress(
            selection,
            selection_features,
            expected_scope="algorithm_selection",
        ),
        identities,
    )


def _progress_inputs(records: Sequence[ProbeRecord]) -> tuple[torch.Tensor, ...]:
    geometry, _, available, _, weights = _tensor_inputs(records)
    mgc = torch.tensor(
        [0.5 if record.mgc_progress is None else record.mgc_progress for record in records],
        dtype=torch.float32,
    )
    start = torch.tensor(
        [
            0.0
            if record.reference_start is None
            else math.radians(record.reference_start)
            for record in records
        ],
        dtype=torch.float32,
    )
    angle_range = torch.tensor(
        [
            0.0
            if record.reference_range is None
            else math.radians(record.reference_range)
            for record in records
        ],
        dtype=torch.float32,
    )
    affine = torch.from_numpy(np.stack([record.affine for record in records])).float()
    reference_available = torch.isfinite(start) & torch.isfinite(angle_range) & (angle_range.abs() > 1e-8)
    target = torch.tensor([record.target_progress for record in records], dtype=torch.float32)
    return geometry, mgc, start, angle_range, affine, available, reference_available, target, weights


def progress_soft_targets(
    target_progress: torch.Tensor,
    *,
    progress_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    if progress_bins < 8 or sigma_bins <= 0.0:
        raise ValueError("invalid progress soft-target configuration")
    grid = torch.linspace(0.0, 1.0, progress_bins, device=target_progress.device, dtype=target_progress.dtype)
    sigma = float(sigma_bins) / float(progress_bins - 1)
    target = torch.exp(-0.5 * ((grid[None, :] - target_progress[:, None]) / sigma).square())
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def group_balanced_progress_loss(
    outputs: Any,
    target_progress: torch.Tensor,
    group_weight: torch.Tensor,
    *,
    geometry_available: torch.Tensor,
    sigma_bins: float,
    expected_weight: float,
    geometry_aux_weight: float,
    residual_l2_weight: float,
    residual_soft_clip: float,
    residual_soft_clip_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            count = int((~torch.isfinite(value)).sum().item())
            raise FloatingPointError(
                f"GeoPEPD progress loss component {name} has {count} non-finite values"
            )

    for name in (
        "progress_log_probability",
        "geometry_progress_log_probability",
        "expected_progress",
        "visual_progress_residual",
        "geometry_progress_residual",
    ):
        require_finite(name, getattr(outputs, name).float())
    require_finite("target_progress", target_progress.float())
    require_finite("group_weight", group_weight.float())
    soft = progress_soft_targets(
        target_progress.clamp(0.0, 1.0),
        progress_bins=outputs.progress_log_probability.shape[1],
        sigma_bins=sigma_bins,
    )
    valid = outputs.valid.bool() & torch.isfinite(target_progress)
    if not bool(valid.any()):
        raise FloatingPointError("GeoPEPD progress batch has no valid reference rows")
    fused_ce_row = -(soft * outputs.progress_log_probability.float()).sum(1)
    smooth_row = F.smooth_l1_loss(
        outputs.expected_progress.float(),
        target_progress.float().clamp(0.0, 1.0),
        reduction="none",
        beta=0.02,
    )
    valid_weight = group_weight.float()[valid]
    fused_ce = (
        fused_ce_row[valid] * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1e-8)
    expected = (
        smooth_row[valid] * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1e-8)
    geometry_mask = (
        outputs.geometry_available.bool()
        & torch.as_tensor(geometry_available, device=valid.device).bool()
        & torch.isfinite(target_progress)
    )
    if bool(geometry_mask.any()):
        geometry_ce_row = -(soft * outputs.geometry_progress_log_probability.float()).sum(1)
        geometry_weight = group_weight.float()[geometry_mask]
        geometry_aux = (
            geometry_ce_row[geometry_mask] * geometry_weight
        ).sum() / geometry_weight.sum().clamp_min(1e-8)
    else:
        geometry_aux = fused_ce.new_zeros(())
    if bool(geometry_mask.any()):
        residual_weight = group_weight.float()[geometry_mask]
        visual_residual = outputs.visual_progress_residual.float()
        geometry_residual = outputs.geometry_progress_residual.float()
        visual_residual_l2 = (
            visual_residual[geometry_mask].square() * residual_weight
        ).sum() / residual_weight.sum().clamp_min(1e-8)
        geometry_residual_l2 = (
            geometry_residual[geometry_mask].square() * residual_weight
        ).sum() / residual_weight.sum().clamp_min(1e-8)
        visual_residual_clip = (
            F.relu(
                visual_residual[geometry_mask].abs() - float(residual_soft_clip)
            ).square()
            * residual_weight
        ).sum() / residual_weight.sum().clamp_min(1e-8)
        geometry_residual_clip = (
            F.relu(
                geometry_residual[geometry_mask].abs()
                - float(residual_soft_clip)
            ).square()
            * residual_weight
        ).sum() / residual_weight.sum().clamp_min(1e-8)
    else:
        visual_residual_l2 = fused_ce.new_zeros(())
        geometry_residual_l2 = fused_ce.new_zeros(())
        visual_residual_clip = fused_ce.new_zeros(())
        geometry_residual_clip = fused_ce.new_zeros(())
    loss = (
        fused_ce
        + float(expected_weight) * expected
        + float(geometry_aux_weight) * geometry_aux
        + float(residual_l2_weight)
        * (visual_residual_l2 + geometry_residual_l2)
        + float(residual_soft_clip_weight)
        * (visual_residual_clip + geometry_residual_clip)
    )
    components = {
        "fused_soft_ce": fused_ce,
        "expected_smooth_l1": expected,
        "geometry_aux_soft_ce": geometry_aux,
        "visual_residual_l2": visual_residual_l2,
        "geometry_residual_l2": geometry_residual_l2,
        "visual_residual_soft_clip": visual_residual_clip,
        "geometry_residual_soft_clip": geometry_residual_clip,
        "total_loss": loss,
    }
    for name, value in components.items():
        require_finite(name, value)
    return loss, {
        "fused_soft_ce": fused_ce.detach(),
        "expected_smooth_l1": expected.detach(),
        "geometry_aux_soft_ce": geometry_aux.detach(),
        "visual_residual_l2": visual_residual_l2.detach(),
        "geometry_residual_l2": geometry_residual_l2.detach(),
        "visual_residual_soft_clip": visual_residual_clip.detach(),
        "geometry_residual_soft_clip": geometry_residual_clip.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }


def _freeze_visual_modules(model: GeoPEPDProgressFusionNet) -> None:
    for name in (
        "encoder",
        "pivot_head",
        "direction_features",
        "vector_head",
        "angle_head",
        "log_variance_head",
    ):
        module = getattr(model, name, None)
        if module is not None:
            module.eval()


def train_stage_a(
    model: GeoPEPDProgressFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProbeRecord],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    tensors = _progress_inputs(records)
    geometry = tensors[0]
    finite = torch.isfinite(geometry)
    safe = torch.where(finite, geometry, torch.zeros_like(geometry))
    count = finite.sum(0).clamp_min(1)
    mean = safe.sum(0) / count
    variance = torch.where(finite, (geometry - mean).square(), torch.zeros_like(geometry)).sum(0) / count
    model.set_geometry_normalization(mean, torch.sqrt(variance.clamp_min(1e-6)))
    model.freeze_pepd()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("progress Stage A has no trainable parameters")
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
    dataset = TensorDataset(pooled, *tensors)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        _freeze_visual_modules(model)
        totals = Counter()
        for batch in loader:
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
            ) = [value.to(device) for value in batch]
            drop = geometry_available_b & (
                torch.rand(geometry_available_b.shape, device=device)
                < float(args.geometry_dropout)
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
                geometry_drop_mask=drop,
            )
            loss, parts = group_balanced_progress_loss(
                outputs,
                target_b,
                weight_b,
                geometry_available=geometry_available_b,
                sigma_bins=float(args.soft_target_sigma_bins),
                expected_weight=float(args.expected_smooth_l1_weight),
                geometry_aux_weight=float(args.geometry_aux_weight),
                residual_l2_weight=float(args.residual_l2_weight),
                residual_soft_clip=float(args.residual_soft_clip),
                residual_soft_clip_weight=float(args.residual_soft_clip_weight),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite GeoPEPD total loss at epoch={epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if (
                    parameter.requires_grad
                    and parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ):
                    raise FloatingPointError(
                        f"non-finite GeoPEPD gradient at epoch={epoch}: {name}"
                    )
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not bool(
                    torch.isfinite(parameter).all()
                ):
                    raise FloatingPointError(
                        f"non-finite GeoPEPD parameter after optimizer step: {name}"
                    )
            size = int(pooled_b.shape[0])
            totals["samples"] += size
            totals["loss"] += float(loss.detach()) * size
            for name, value in parts.items():
                totals[name] += float(value) * size
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "loss": totals["loss"] / totals["samples"],
            "fused_soft_ce": totals["fused_soft_ce"] / totals["samples"],
            "expected_smooth_l1": totals["expected_smooth_l1"] / totals["samples"],
            "geometry_aux_soft_ce": totals["geometry_aux_soft_ce"] / totals["samples"],
            "visual_residual_l2": totals["visual_residual_l2"] / totals["samples"],
            "geometry_residual_l2": totals["geometry_residual_l2"] / totals["samples"],
            "visual_residual_soft_clip": totals["visual_residual_soft_clip"] / totals["samples"],
            "geometry_residual_soft_clip": totals["geometry_residual_soft_clip"] / totals["samples"],
            "valid_fraction": totals["valid_fraction"] / totals["samples"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"GeoPEPD-progress Stage-A epoch={epoch}/{args.epochs} "
            f"loss={row['loss']:.6f} valid={row['valid_fraction']:.3f}",
            flush=True,
        )
    return history


def go_decision(metrics: Mapping[str, Mapping[str, float]], bootstrap: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    baselines = ("pepd", "mgc", "reference_calibrated_pepd", "fadr")
    best = min(baselines, key=lambda name: float(metrics[name]["full_denominator_nmae"]))
    comparison = bootstrap[f"geopepd_progress_vs_{best}"]
    candidate = metrics["geopepd_progress"]
    best_metrics = metrics[best]
    nmae_better = float(candidate["full_denominator_nmae"]) < float(best_metrics["full_denominator_nmae"])
    significant = float(comparison["ci95_high"]) < 0.0
    coverage_noninferior = float(candidate["coverage"]) >= float(best_metrics["coverage"]) - 0.005
    return {
        "label": "GO" if nmae_better and significant and coverage_noninferior else "NO_GO",
        "best_development_baseline": best,
        "rules": {
            "lower_nmae_than_best_baseline": nmae_better,
            "paired_group_bootstrap_ci95_upper_below_zero": significant,
            "coverage_noninferiority_margin_0.005": coverage_noninferior,
        },
        "scope": "adaptive train-only algorithm_selection development gate; not confirmatory evidence",
    }


def no_residual_expected_progress(
    outputs: Any, progress_grid: torch.Tensor
) -> torch.Tensor:
    """Ablate both visual transport and geometry-center residual in one pass."""

    grid = progress_grid.to(
        device=outputs.progress_log_probability.device, dtype=torch.float32
    )
    raw_geometry_logits = -0.5 * (
        (grid[None, :] - outputs.mgc_progress[:, None])
        / outputs.geometry_scale[:, None]
    ).square()
    raw_geometry_log = F.log_softmax(raw_geometry_logits, dim=1)
    log_probability = (
        outputs.visual_reliability[:, None]
        * outputs.raw_visual_progress_log_probability.float()
        + outputs.geometry_reliability[:, None] * raw_geometry_log
    )
    log_probability = log_probability - torch.logsumexp(
        log_probability, dim=1, keepdim=True
    )
    return torch.sum(torch.exp(log_probability) * grid[None, :], dim=1)


@torch.inference_mode()
def evaluate(
    model: GeoPEPDProgressFusionNet,
    pooled: torch.Tensor,
    records: Sequence[ProbeRecord],
    *,
    device: torch.device,
    batch_size: int,
    bootstrap_repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _progress_inputs(records)
    rows: list[dict[str, Any]] = []
    model.eval()
    for offset in range(0, len(records), batch_size):
        stop = min(len(records), offset + batch_size)
        batch = [value[offset:stop].to(device) for value in tensors]
        geometry_b, mgc_b, start_b, range_b, affine_b, geometry_available_b, reference_available_b, _, _ = batch
        outputs = model.forward_from_encoder_pooled(
            pooled[offset:stop].to(device),
            geometry_b,
            mgc_b,
            start_b,
            range_b,
            affine_b,
            geometry_available_b,
            reference_available_b,
        )
        for name in (
            "progress_log_probability",
            "visual_progress_log_probability",
            "raw_visual_progress_log_probability",
            "geometry_progress_log_probability",
            "expected_progress",
            "visual_expected_progress",
            "visual_progress_residual",
            "geometry_progress_residual",
            "visual_reliability",
            "geometry_reliability",
        ):
            value = getattr(outputs, name)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite GeoPEPD evaluation output: {name}"
                )
        dummy_pivot = torch.zeros(
            (stop - offset, 1, 1, 1),
            device=device,
            dtype=outputs.direction_raw.dtype,
        )
        pepd_direction = decode_probabilistic_pivot_direction(
            dummy_pivot,
            outputs.direction_raw,
            outputs.visual_angle_logits,
            outputs.visual_log_variance_raw,
        ).direction.cpu().numpy()
        candidate = outputs.expected_progress.cpu().numpy()
        valid = outputs.valid.cpu().numpy().astype(bool)
        grid = model.progress_grid.to(device=device, dtype=torch.float32)
        geometry_only = torch.sum(
            torch.exp(outputs.geometry_progress_log_probability.float())
            * grid[None, :],
            dim=1,
        ).cpu().numpy()
        visual_only = outputs.visual_expected_progress.cpu().numpy()
        no_residual = no_residual_expected_progress(outputs, grid).cpu().numpy()
        visual_reliability = outputs.visual_reliability.cpu().numpy()
        geometry_reliability = outputs.geometry_reliability.cpu().numpy()
        geometry_effective = outputs.geometry_available.cpu().numpy().astype(bool)
        visual_residual = outputs.visual_progress_residual.cpu().numpy()
        geometry_residual = outputs.geometry_progress_residual.cpu().numpy()
        for local, record in enumerate(records[offset:stop]):
            progress = {
                "geopepd_progress": float(candidate[local]) if valid[local] and math.isfinite(float(candidate[local])) else None,
                "geopepd_progress_no_residual": float(no_residual[local]) if valid[local] and math.isfinite(float(no_residual[local])) else None,
                "geopepd_progress_geometry_only": float(geometry_only[local]) if geometry_effective[local] and math.isfinite(float(geometry_only[local])) else None,
                "geopepd_progress_visual_only": float(visual_only[local]) if valid[local] and math.isfinite(float(visual_only[local])) else None,
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
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "ground_truth_progress": record.target_progress,
                    "progress": progress,
                    "errors": errors,
                    "visual_reliability": float(visual_reliability[local]),
                    "geometry_reliability": float(geometry_reliability[local]),
                    "visual_progress_residual": float(visual_residual[local]),
                    "geometry_progress_residual": float(geometry_residual[local]),
                    "geometry_available": record.geometry_available,
                    "reference_available": record.reference_start is not None and record.reference_range is not None,
                }
            )
    method_metrics: dict[str, dict[str, float | int]] = {}
    for method in (
        "geopepd_progress",
        "geopepd_progress_no_residual",
        "geopepd_progress_geometry_only",
        "geopepd_progress_visual_only",
        "pepd",
        "mgc",
        "reference_calibrated_pepd",
        "fadr",
    ):
        covered = sum(row["progress"][method] is not None for row in rows)
        method_metrics[method] = {
            "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
            "coverage": covered / len(rows),
            "covered_samples": covered,
        }
    bootstrap = {
        f"geopepd_progress_vs_{baseline}": grouped_bootstrap_delta(
            rows,
            "geopepd_progress",
            baseline,
            repetitions=bootstrap_repetitions,
        )
        for baseline in ("pepd", "mgc", "reference_calibrated_pepd", "fadr")
    }
    geometry_residual_values = np.asarray(
        [row["geometry_progress_residual"] for row in rows if row["geometry_available"]],
        dtype=np.float64,
    )
    visual_residual_values = np.asarray(
        [row["visual_progress_residual"] for row in rows if row["geometry_available"]],
        dtype=np.float64,
    )

    def residual_summary(values: np.ndarray) -> dict[str, float | int | None]:
        return {
            "effective_samples": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else None,
            "mean_absolute": float(np.mean(np.abs(values))) if len(values) else None,
            "p95_absolute": float(np.quantile(np.abs(values), 0.95)) if len(values) else None,
            "fraction_beyond_soft_clip": float(np.mean(np.abs(values) > 0.05)) if len(values) else None,
        }
    return {
        "methods": method_metrics,
        "grouped_bootstrap": bootstrap,
        "residual_diagnostics": {
            "visual_progress_residual": residual_summary(visual_residual_values),
            "geometry_progress_residual": residual_summary(
                geometry_residual_values
            ),
        },
        "decision": go_decision(method_metrics, bootstrap),
    }, rows


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    if int(args.epochs) <= 0 or not 0.0 <= float(args.geometry_dropout) < 1.0:
        raise ValueError("invalid epochs/geometry dropout")
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="output_dir")
    invalidate_existing_nonfinite_run(output_dir)
    fit, selection, identities = load_progress_probe_records(args)
    preparation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "prepared" if args.prepare_only else "running",
        "fit": {"samples": len(fit), "groups": len({record.group_id for record in fit})},
        "selection": {"samples": len(selection), "groups": len({record.group_id for record in selection})},
        "fit_selection_sample_overlap": 0,
        "fit_selection_group_overlap": 0,
        "shared_encoder_cache_root": str(SHARED_CACHE_ROOT),
        "geometry_center_binding": {
            "method_label": "MGC",
            "source_field": "component feature row base_progress",
            "source_prediction_field": "base_prediction",
            "calibrated_progress_used_as_geometry_center": False,
            "calibrated_progress_role": "separately labeled reference-calibrated PEPD development baseline and FADR internal component; never a model input",
            "reference_angle_units_forwarded_to_model": "radians",
            "fit_available_samples": sum(record.geometry_available for record in fit),
            "selection_available_samples": sum(
                record.geometry_available for record in selection
            ),
        },
        "development_baseline_bindings": {
            "pepd": "frozen checkpoint legacy vector-plus-angle-bin direction decoded through the shared reference",
            "mgc": "component feature row base_progress (raw mask--geometry calibrated baseline)",
            "reference_calibrated_pepd": "component feature row calibrated_progress; comparison only, never passed to GeoPEPD",
            "fadr": "component feature row fadr_progress",
        },
        "invalidated_prior_evidence": (
            "The first circular probe bound calibrated_progress as MGC; its MGC-labeled "
            "comparison is invalid and is not inherited by this progress probe."
        ),
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
    model = _load_model(Path(args.checkpoint).resolve(), device)
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
    history = train_stage_a(model, fit_pooled, fit, args, device)
    metrics, predictions = evaluate(
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
            "schema_version": 1,
            "protocol": PROTOCOL,
            "seed": SEED,
            "stage": "A_frozen_pepd_progress_posterior_heads_only",
            "model_state": model.state_dict(),
            "input_sha256": identities,
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
        "stage": "A_frozen_pepd_progress_posterior_heads_only",
        "configuration": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "geometry_dropout": float(args.geometry_dropout),
            "soft_target_sigma_bins": float(args.soft_target_sigma_bins),
            "expected_smooth_l1_weight": float(args.expected_smooth_l1_weight),
            "geometry_aux_weight": float(args.geometry_aux_weight),
            "residual_l2_weight": float(args.residual_l2_weight),
            "residual_soft_clip": float(args.residual_soft_clip),
            "residual_soft_clip_weight": float(args.residual_soft_clip_weight),
            "reliability_initialization_logits": {"visual": -1.0, "geometry": 1.0},
            "loss": "group-balanced progress soft-target CE + expected Smooth-L1 + geometry auxiliary CE + independent visual-transport and geometry-center residual L2/soft-clip",
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
        "geometry_center_binding": preparation["geometry_center_binding"],
        "development_baseline_bindings": preparation[
            "development_baseline_bindings"
        ],
        "invalidated_prior_evidence": preparation["invalidated_prior_evidence"],
        "elapsed_seconds": time.time() - started,
        "selection_labels_used_for_model_fit": 0,
        "restricted_data_use": preparation["restricted_data_use"],
    }
    if _contains_nonfinite(summary):
        raise FloatingPointError(
            "GeoPEPD summary contains a non-finite value and cannot be complete"
        )
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path


if __name__ == "__main__":
    parsed = parse_args()
    try:
        print(run(parsed))
    except Exception as error:
        failure_dir = Path(parsed.output_dir).resolve()
        assert_train_only_path(failure_dir, label="failure_output_dir")
        _write_json(
            failure_dir / "failure.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "failed",
                "summary_complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
