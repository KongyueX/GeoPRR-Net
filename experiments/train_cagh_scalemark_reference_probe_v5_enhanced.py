"""Train the enhanced reliability-aware ScaleMark V5 head on public tight ROIs.

This is an independent adapter around the already frozen public tight-ROI data
contract.  It intentionally does not modify or replace the running baseline V5
trainer/protocol.  The PEPD backbone is frozen; only the enhanced multi-scale,
dual-endpoint ScaleMark head is optimized.  Field, confirmatory, sealed, test,
xiangmu1, and xiangmu2 namespaces remain forbidden by the shared roster loader.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.cagh_scalemark_reference_head_v5 import (
    CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
    build_head,
    geometry_loss as enhanced_geometry_loss,
    multiscale_dense_tick_loss,
)
from experiments.fadr_multiseed_protocol import sha256_file
from experiments.probabilistic_pivot_direction import decode_probabilistic_pivot_direction
from experiments.vdn_baseline import sha256_source_file
import experiments.train_cagh_scalemark_reference_probe_v5 as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_scalemark_reference_public_v5_enhanced_tight_roi_v1"
PROTOCOL_PATH = (
    PROJECT_ROOT / "experiments/cagh_scalemark_reference_public_v5_enhanced_protocol.json"
)
OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_scalemark_reference_public_v5_enhanced\runs")
CACHE_ROOT = Path(r"C:\pointer_read\cagh_scalemark_reference_public_v5_enhanced\cache")
HEAD_SOURCE = PROJECT_ROOT / "experiments/cagh_scalemark_reference_head_v5.py"
BASE_TRAINER_SOURCE = PROJECT_ROOT / "experiments/train_cagh_scalemark_reference_probe_v5.py"
BASE_PROTOCOL_SOURCE = PROJECT_ROOT / "experiments/cagh_scalemark_reference_public_v5_protocol.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--seed", type=int, default=20261306)
    parser.add_argument("--stage-a-epochs", type=int, default=6)
    parser.add_argument("--stage-b-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augmentation-profile", choices=("photo", "none"), default="photo")
    parser.add_argument("--brightness-probability", type=float, default=.80)
    parser.add_argument("--brightness-delta", type=float, default=.16)
    parser.add_argument("--contrast-probability", type=float, default=.80)
    parser.add_argument("--contrast-min", type=float, default=.65)
    parser.add_argument("--contrast-max", type=float, default=1.40)
    parser.add_argument("--gamma-probability", type=float, default=.45)
    parser.add_argument("--gamma-min", type=float, default=.65)
    parser.add_argument("--gamma-max", type=float, default=1.55)
    parser.add_argument("--blur-probability", type=float, default=.35)
    parser.add_argument("--blur-sigma-max", type=float, default=1.6)
    parser.add_argument("--noise-probability", type=float, default=.30)
    parser.add_argument("--noise-sigma-max", type=float, default=10.0)
    parser.add_argument("--jpeg-probability", type=float, default=.35)
    parser.add_argument("--jpeg-quality-min", type=int, default=50)
    parser.add_argument("--jpeg-quality-max", type=int, default=94)
    parser.add_argument("--perspective-probability", type=float, default=.35)
    parser.add_argument("--perspective-fraction-max", type=float, default=.035)
    parser.add_argument("--boundary-trim-probability", type=float, default=.35)
    parser.add_argument("--boundary-trim-fraction-max", type=float, default=.06)
    parser.add_argument("--smoke-fit-samples", type=int, default=4)
    parser.add_argument("--smoke-validation-samples", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def load_protocol() -> Mapping[str, Any]:
    _require(PROTOCOL_PATH.is_file(), f"missing enhanced V5 protocol: {PROTOCOL_PATH}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(protocol.get("protocol") == PROTOCOL, "enhanced V5 protocol identity drift")
    _require(protocol.get("status") == "frozen_public_only", "enhanced V5 protocol is not frozen")
    identities = (
        (base.MANIFEST, "manifest_sha256"),
        (base.MANIFEST_PROTOCOL, "manifest_protocol_sha256"),
        (base.BACKBONE_CHECKPOINT, "backbone_checkpoint_sha256"),
        (HEAD_SOURCE, "enhanced_head_sha256"),
        (BASE_TRAINER_SOURCE, "base_tight_roi_adapter_sha256"),
        (BASE_PROTOCOL_SOURCE, "base_tight_roi_protocol_sha256"),
    )
    for path, key in identities:
        _require(path.is_file(), f"missing enhanced V5 input: {path}")
        _require(sha256_file(path) == protocol["inputs"][key], f"{key} drift")
    _require(
        base.canonical_sha256(base.CROP_CONTRACT) == protocol["crop_contract_sha256"],
        "enhanced V5 crop contract drift",
    )
    _require(
        protocol["head"]["protocol"] == CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        "enhanced head protocol drift",
    )
    return protocol


def _projective_local_inverse(
    homography: torch.Tensor,
    point_xy: torch.Tensor,
) -> torch.Tensor:
    """Jacobian of final-normalized -> isotropic coordinates at the pivot."""

    matrix = homography.float()
    point = point_xy.float()
    x, y = point[:, 0], point[:, 1]
    a, b, c = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    d, e, f = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    g, h, i = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
    denominator = (g * x + h * y + i).clamp_min(1e-6)
    first = a * x + b * y + c
    second = d * x + e * y + f
    square = denominator.square()
    jacobian = torch.stack(
        (
            (a * denominator - first * g) / square,
            (b * denominator - first * h) / square,
            (d * denominator - second * g) / square,
            (e * denominator - second * h) / square,
        ),
        dim=1,
    ).reshape(-1, 2, 2)
    _require(bool(torch.isfinite(jacobian).all()), "projective local inverse is non-finite")
    return jacobian


def _loader(
    dataset: base.CanonicalTightROIDataset,
    *,
    args: argparse.Namespace,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    return base._loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=shuffle,
        seed=seed,
        pin_memory=device.type == "cuda",
    )


def train_enhanced_head(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataset: base.CanonicalTightROIDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {"stage_a_tick": [], "stage_b_geometry": []}
    for stage, epochs, key, offset in (
        ("tick", args.stage_a_epochs, "stage_a_tick", 0),
        ("geometry", args.stage_b_epochs, "stage_b_geometry", 1_000),
    ):
        names = base.set_stage(head, stage)
        _require(bool(names), f"enhanced V5 {stage} stage has no trainable parameters")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in head.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for epoch in range(1, epochs + 1):
            dataset.set_epoch(offset + epoch)
            loader = _loader(
                dataset, args=args, shuffle=True, seed=args.seed + offset + epoch,
                device=device,
            )
            head.train(); total = samples = augmented = 0
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                with torch.no_grad():
                    features = backbone.forward_multiscale_features(images)
                    pooled = backbone.direction_features(features.c5)
                    decoded = decode_probabilistic_pivot_direction(
                        backbone.pivot_head(features.c5),
                        backbone.vector_head(pooled),
                        backbone.angle_head(pooled),
                        backbone.log_variance_head(pooled),
                    )
                output = head(features.c2.detach(), features.c5.detach())
                weight = batch["group_weight"].to(device, non_blocking=True)
                if stage == "tick":
                    loss, _ = multiscale_dense_tick_loss(
                        output,
                        batch["tick_heatmap"].to(device, non_blocking=True),
                        group_weight=weight,
                    )
                else:
                    pivot = decoded.pivot_xy / 63.0
                    inverse = _projective_local_inverse(
                        batch["final_to_isotropic"].to(device, non_blocking=True), pivot
                    )
                    loss = enhanced_geometry_loss(
                        output,
                        batch["endpoints"].to(device, non_blocking=True),
                        batch["gt_start"].to(device, non_blocking=True),
                        batch["gt_range"].to(device, non_blocking=True),
                        pivot,
                        inverse,
                        weight,
                    )
                _require(bool(torch.isfinite(loss)), f"enhanced V5 {stage} loss is non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in head.parameters() if parameter.requires_grad), 5.0
                )
                optimizer.step()
                count = int(images.shape[0])
                samples += count
                total += float(loss.detach()) * count
                augmented += int((batch["augmentation_code"] != 0).sum())
            _require(samples == len(dataset), "enhanced V5 epoch inventory drift")
            row = {
                "epoch": epoch,
                "loss": total / samples,
                "samples": samples,
                "augmented_fraction": augmented / samples,
            }
            history[key].append(row)
            print(
                f"enhanced-v5 seed={args.seed} stage={stage} epoch={epoch}/{epochs} "
                f"loss={row['loss']:.6f} augmented={row['augmented_fraction']:.3f}",
                flush=True,
            )
    return history


@torch.inference_mode()
def evaluate_enhanced(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataset: base.CanonicalTightROIDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _loader(dataset, args=args, shuffle=False, seed=args.seed, device=device)
    backbone.eval(); head.eval(); rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = backbone.forward_multiscale_features(images)
        pooled = backbone.direction_features(features.c5)
        decoded = decode_probabilistic_pivot_direction(
            backbone.pivot_head(features.c5), backbone.vector_head(pooled),
            backbone.angle_head(pooled), backbone.log_variance_head(pooled),
        )
        output = head(features.c2, features.c5)
        pivot = decoded.pivot_xy / 63.0
        start, arc, radii, reference_valid = base.reference_geometry_projective(
            output.start_xy,
            output.end_xy,
            pivot,
            batch["final_to_isotropic"].to(device, non_blocking=True),
        )
        solver = differentiable_keypoint_reference_solver(
            pivot + .25 * decoded.direction,
            pivot,
            start,
            arc,
            batch["crop_affine"].to(device, non_blocking=True),
            reference_valid,
        )
        combined = (
            decoded.valid & reference_valid & solver.valid & output.telemetry_valid
            & torch.isfinite(solver.expected_progress)
        )
        target_endpoint = batch["endpoints"].to(device)
        predicted_endpoint = torch.stack((output.start_xy, output.end_xy), dim=1)
        endpoint_error = torch.linalg.vector_norm(
            predicted_endpoint - target_endpoint, dim=2
        ).mean(1)
        target_start = batch["gt_start"].to(device)
        target_arc = batch["gt_range"].to(device)
        start_error = torch.abs(
            torch.atan2(torch.sin(start - target_start), torch.cos(start - target_start))
        )
        arc_error = torch.abs(arc - target_arc)
        for index, sample_id in enumerate(batch["sample_id"]):
            valid = bool(combined[index])
            prediction = float(solver.expected_progress[index]) if valid else None
            target = float(batch["target_progress"][index])
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "head_protocol": CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
                    "sample_id": str(sample_id),
                    "group_id": str(batch["group_id"][index]),
                    "status": "ok" if valid else "failed",
                    "failure_code": None if valid else (
                        "invalid_pointer" if not bool(decoded.valid[index]) else
                        "invalid_reference_geometry" if not bool(reference_valid[index]) else
                        "invalid_head_telemetry" if not bool(output.telemetry_valid[index]) else
                        "invalid_pointer_solver"
                    ),
                    "target_progress": target,
                    "predicted_progress": prediction,
                    "absolute_progress_error": abs(prediction - target) if prediction is not None else 1.0,
                    "endpoint": {
                        "start_xy": output.start_xy[index].cpu().tolist(),
                        "end_xy": output.end_xy[index].cpu().tolist(),
                        "target_start_xy": target_endpoint[index, 0].cpu().tolist(),
                        "target_end_xy": target_endpoint[index, 1].cpu().tolist(),
                        "mean_l2_error": float(endpoint_error[index]),
                        "peak": output.endpoint_peak[index].cpu().tolist(),
                        "entropy": output.endpoint_entropy[index].cpu().tolist(),
                        "separation": float(output.endpoint_separation[index]),
                        "coordinate_disagreement": float(
                            output.endpoint_coordinate_disagreement[index]
                        ),
                        "js_divergence": float(output.endpoint_js_divergence[index]),
                        "consistency_reliability": float(
                            output.endpoint_consistency_reliability[index]
                        ),
                    },
                    "arc": {
                        "start_degrees": math.degrees(float(start[index])),
                        "target_start_degrees": math.degrees(float(target_start[index])),
                        "start_absolute_error_degrees": math.degrees(float(start_error[index])),
                        "range_degrees": math.degrees(float(arc[index])),
                        "target_range_degrees": math.degrees(float(target_arc[index])),
                        "range_absolute_error_degrees": math.degrees(float(arc_error[index])),
                        "endpoint_radii": radii[index].cpu().tolist(),
                        "ordered_arc_fraction": float(output.ordered_arc_fraction[index]),
                        "ordered_arc_tick_mass": float(output.ordered_arc_tick_mass[index]),
                        "arc_length_reliability": float(output.arc_length_reliability[index]),
                    },
                    "reliability": {
                        "gate_confidence": float(output.gate_confidence[index]),
                        "uncertainty_score": float(output.uncertainty_score[index]),
                        "tick_scale_disagreement": float(output.tick_scale_disagreement[index]),
                        "tick_circle_center_xy": output.tick_circle_center_xy[index].cpu().tolist(),
                        "tick_radius_mean": float(output.tick_radius_mean[index]),
                        "tick_radius_cv": float(output.tick_radius_cv[index]),
                        "radius_reliability": float(output.radius_reliability[index]),
                        "endpoint_tick_support": float(output.endpoint_tick_support[index]),
                    },
                    "validity": {
                        "pointer": bool(decoded.valid[index]),
                        "reference": bool(reference_valid[index]),
                        "solver": bool(solver.valid[index]),
                        "telemetry": bool(output.telemetry_valid[index]),
                        "combined": valid,
                    },
                    "roi_bounds": batch["roi_bounds"][index].tolist(),
                    "augmentation_code": int(batch["augmentation_code"][index]),
                }
            )
    _require(len(rows) == len(dataset), "enhanced V5 telemetry inventory drift")
    endpoint = np.asarray([row["endpoint"]["mean_l2_error"] for row in rows])
    arc_error = np.asarray([row["arc"]["range_absolute_error_degrees"] for row in rows])
    start_error = np.asarray([row["arc"]["start_absolute_error_degrees"] for row in rows])
    full_error = np.asarray([row["absolute_progress_error"] for row in rows])
    gate = np.asarray([row["reliability"]["gate_confidence"] for row in rows])
    metrics = {
        "samples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "coverage": sum(row["validity"]["combined"] for row in rows) / len(rows),
        "pointer_valid_fraction": sum(row["validity"]["pointer"] for row in rows) / len(rows),
        "reference_valid_fraction": sum(row["validity"]["reference"] for row in rows) / len(rows),
        "solver_valid_fraction": sum(row["validity"]["solver"] for row in rows) / len(rows),
        "telemetry_valid_fraction": sum(row["validity"]["telemetry"] for row in rows) / len(rows),
        "full_denominator_nmae": float(full_error.mean()),
        "endpoint_mean_l2": float(endpoint.mean()),
        "endpoint_p95_l2": float(np.quantile(endpoint, .95)),
        "start_angle_mae_degrees": float(start_error.mean()),
        "arc_mae_degrees": float(arc_error.mean()),
        "arc_p95_error_degrees": float(np.quantile(arc_error, .95)),
        "gate_confidence_mean": float(gate.mean()),
        "gate_confidence_p05": float(np.quantile(gate, .05)),
    }
    return metrics, rows


def _roster_cache(
    root: Path,
    fit: Sequence[base.PublicRecord],
    validation: Sequence[base.PublicRecord],
) -> Path:
    resolved = base._guard_restricted_path(root, label="enhanced V5 cache root")
    _require(resolved != Path(resolved.anchor), "broad enhanced V5 cache root rejected")
    signature = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "crop_contract_sha256": base.canonical_sha256(base.CROP_CONTRACT),
        "fit_ids_sha256": base.canonical_sha256(
            [record.sample.sample_id for record in fit]
        ),
        "validation_ids_sha256": base.canonical_sha256(
            [record.sample.sample_id for record in validation]
        ),
        "fit_samples": len(fit),
        "validation_samples": len(validation),
        "feature_cache": False,
    }
    path = resolved / f"roster_signature_{len(fit)}_{len(validation)}.json"
    if path.is_file():
        _require(json.loads(path.read_text(encoding="utf-8")) == signature,
                 "enhanced V5 roster cache drift")
    else:
        base.atomic_json(path, signature)
    return path


def _save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    history: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "head_protocol": CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        "status": "complete",
        "history": history,
        "validation": validation,
        "head_state": {name: tensor.detach().cpu() for name, tensor in head.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def validation_record(
    protocol: Mapping[str, Any],
    fit: Sequence[base.PublicRecord],
    validation: Sequence[base.PublicRecord],
    augmentation: base.PhotoAugmentation,
    roster_cache: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "head_protocol": CAGH_SCALEMARK_REFERENCE_HEAD_V5_PROTOCOL,
        "status": "validated",
        "mode": "smoke" if args.smoke else "formal" if args.run_formal else "validate_only",
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "fit_samples": len(fit),
            "fit_groups": len({record.sample.group_id for record in fit}),
            "validation_samples": len(validation),
            "validation_groups": len({record.sample.group_id for record in validation}),
            "field_samples_read": 0,
            "confirmatory_samples_read": 0,
            "xiangmu_samples_read": 0,
        },
        "preprocessing": {
            "crop_contract": base.CROP_CONTRACT,
            "crop_contract_sha256": base.canonical_sha256(base.CROP_CONTRACT),
            "geometry_training_inverse": "local Jacobian of exact augmentation homography at predicted pivot",
            "telemetry_geometry_inverse": "exact projective unprojection",
        },
        "augmentation": augmentation.as_dict(),
        "training": {
            "seed": args.seed,
            "stage_a_epochs": args.stage_a_epochs,
            "stage_b_epochs": args.stage_b_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "backbone": "frozen pinned PEPD",
            "trainable": "enhanced multi-scale dual-endpoint ScaleMark head only",
        },
        "cache": {
            "root": str(Path(args.cache_root).resolve()),
            "roster_signature": str(roster_cache),
            "features_cached": False,
        },
        "input_sha256": {
            "protocol": sha256_file(PROTOCOL_PATH),
            "manifest": sha256_file(base.MANIFEST),
            "manifest_protocol": sha256_file(base.MANIFEST_PROTOCOL),
            "backbone_checkpoint": sha256_file(base.BACKBONE_CHECKPOINT),
            "enhanced_head_source": sha256_source_file(HEAD_SOURCE),
            "base_tight_roi_adapter": sha256_source_file(BASE_TRAINER_SOURCE),
            "trainer_source": sha256_source_file(Path(__file__).resolve()),
        },
        "known_limitation": protocol["known_limitation"],
    }


def run(args: argparse.Namespace) -> Path:
    _require(args.stage_a_epochs > 0 and args.stage_b_epochs > 0, "epochs must be positive")
    _require(args.batch_size > 0 and args.workers >= 0, "invalid batch/workers")
    protocol = load_protocol()
    augmentation = base.augmentation_from_args(args)
    fit, validation = base.load_public_roster(protocol)
    if args.smoke:
        _require(args.smoke_fit_samples >= 2 and args.smoke_validation_samples >= 2,
                 "smoke inventories must be at least two")
        fit = fit[: args.smoke_fit_samples]
        validation = validation[: args.smoke_validation_samples]
        args.stage_a_epochs = 1
        args.stage_b_epochs = 1
        args.workers = 0
    roster_cache = _roster_cache(Path(args.cache_root), fit, validation)
    output = Path(args.output_dir) if args.output_dir else (
        OUTPUT_ROOT / ("smoke" if args.smoke else f"seed_{args.seed}")
    )
    output = base._guard_restricted_path(output, label="enhanced V5 output")
    record = validation_record(
        protocol, fit, validation, augmentation, roster_cache, args
    )
    validation_path = output / "validation.json"
    if args.run_formal:
        frozen = protocol["training"]
        _require(args.seed in frozen["seeds"], "formal enhanced V5 seed outside frozen list")
        _require(
            (args.stage_a_epochs, args.stage_b_epochs, args.batch_size)
            == (frozen["stage_a_epochs"], frozen["stage_b_epochs"], frozen["batch_size"]),
            "formal enhanced V5 schedule drift",
        )
        _require(
            math.isclose(args.learning_rate, float(frozen["learning_rate"]))
            and math.isclose(args.weight_decay, float(frozen["weight_decay"])),
            "formal enhanced V5 optimizer drift",
        )
        _require(augmentation.as_dict() == protocol["augmentation"],
                 "formal enhanced V5 augmentation drift")
        _require(not (output / "summary.json").exists(),
                 "formal enhanced V5 run already exists")
    base.atomic_json(validation_path, record)
    if args.validate_only:
        print(validation_path, flush=True)
        return validation_path

    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
        _require("NVIDIA" in torch.cuda.get_device_name(device).upper(),
                 "enhanced V5 CUDA device is not NVIDIA")
    base.seed_everything(args.seed)
    backbone, backbone_identity = base.load_pepd(base.BACKBONE_CHECKPOINT, device)
    head = build_head().to(device)
    fit_dataset = base.CanonicalTightROIDataset(
        fit, training=True, seed=args.seed, augmentation=augmentation
    )
    validation_dataset = base.CanonicalTightROIDataset(
        validation,
        training=False,
        seed=args.seed,
        augmentation=base.PhotoAugmentation.disabled(),
    )
    started = time.time()
    history = train_enhanced_head(
        backbone, head, fit_dataset, args=args, device=device
    )
    metrics, telemetry = evaluate_enhanced(
        backbone, head, validation_dataset, args=args, device=device
    )
    telemetry_path = output / "telemetry.jsonl"
    base.atomic_jsonl(telemetry_path, telemetry)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_hash = _save_checkpoint(checkpoint_path, head, history, record)
    summary = {
        **record,
        "status": "complete",
        "history": history,
        "metrics": metrics,
        "backbone": backbone_identity,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "telemetry": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output / "summary.json"
    base.atomic_json(summary_path, summary)
    print(summary_path, flush=True)
    return summary_path


if __name__ == "__main__":
    run(parse_args())
