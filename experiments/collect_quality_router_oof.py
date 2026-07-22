"""Collect group-held-out mask/vector pairs for training a quality router.

The final SyncG and RPM-10K test splits are deliberately unsupported here.  A
sample is admitted only when its entire meter group was held out by at least
one verified pivot-direction training run.  The mask-side prediction is the
existing grouped OOF ``Ours`` prediction, so neither side of a routing label
has seen the routed sample's group during fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_vdn_baseline import (
    DEFAULT_METER_WEIGHTS,
    DEFAULT_POINT_WEIGHTS,
    _fallback_reference,
    _read_jsonl,
    _shared_reference,
    _target_direction,
    _xyxy_from_detector_box,
)
from experiments.pivot_direction_fallback import (
    PIVOT_DIRECTION_PROTOCOL,
    build_pivot_direction_model,
    decode_pivot_direction,
    tensor_from_bbox,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    grouped_train_val_split,
    image_angle_from_direction,
    load_syncg_manifest,
    reading_from_pointer_angle,
    sample_ids_hash,
    sha256_file,
)
from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel


QUALITY_ROUTER_OOF_PROTOCOL = "syncg_quality_router_cross_model_oof_v1"
DEFAULT_SEEDS = (20260720, 20260721, 20260722)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--raw-predictions",
        type=Path,
        default=Path("artifacts/predictions/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--base-oof-predictions",
        type=Path,
        default=Path("artifacts/runs/syncg_full/oof_predictions.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/runs/pivot_direction_syncg"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/oof_clean.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_METER_WEIGHTS)
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_POINT_WEIGHTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _summary_path(output: Path) -> Path:
    return output.with_name(output.stem + ".summary.json")


def _by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    result = {str(row.get("sample_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate sample identifiers")
    return result


def _method_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, method in (raw.get("methods") or {}).items():
        if not isinstance(method, dict):
            continue
        result[str(name)] = {
            "status": bool(method.get("status")),
            "prediction": _finite(method.get("prediction")),
            "progress": _finite(method.get("progress")),
            "pointer_angle": _finite(method.get("pointer_angle")),
            "confidence": _finite(method.get("confidence")),
        }
    return result


def _load_verified_runs(
    *,
    checkpoint_root: Path,
    seeds: Sequence[int],
    manifest: Path,
    samples,
    validation_fraction: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, set[str]]]:
    checkpoints: dict[int, dict[str, Any]] = {}
    validation_ids: dict[int, set[str]] = {}
    for seed in sorted(set(int(value) for value in seeds)):
        run_dir = checkpoint_root / f"seed_{seed}"
        checkpoint_path = run_dir / "best.pt"
        verification_path = run_dir / "verification.json"
        if not checkpoint_path.is_file() or not verification_path.is_file():
            raise FileNotFoundError(f"verified direction run is incomplete: {run_dir}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        signature = checkpoint.get("signature") or {}
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if signature.get("protocol") != PIVOT_DIRECTION_PROTOCOL:
            raise ValueError(f"{checkpoint_path} has the wrong training protocol")
        if int(signature.get("seed", -1)) != seed:
            raise ValueError(f"{checkpoint_path} seed does not match its run directory")
        if signature.get("manifest_sha256") != sha256_file(manifest):
            raise ValueError(f"{checkpoint_path} was trained with another manifest")
        if not math.isclose(
            float(signature.get("validation_fraction", math.nan)),
            float(validation_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{checkpoint_path} uses another validation fraction")
        if verification.get("verified") is not True:
            raise ValueError(f"{verification_path} did not pass formal verification")
        if verification.get("best_checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"{verification_path} belongs to another checkpoint")
        _, validation = grouped_train_val_split(
            samples,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        if sample_ids_hash(validation) != signature.get("validation_sample_ids_sha256"):
            raise ValueError(f"cannot reconstruct the validation split for seed {seed}")
        ids = {sample.sample_id for sample in validation}
        checkpoints[seed] = {
            "path": checkpoint_path,
            "verification": verification_path,
            "payload": checkpoint,
        }
        validation_ids[seed] = ids
    if not checkpoints:
        raise ValueError("at least one verified checkpoint is required")
    return checkpoints, validation_ids


def _assignment(validation_ids: dict[int, set[str]]) -> dict[str, int]:
    """Choose deterministically among checkpoints that held the group out."""

    assigned: dict[str, int] = {}
    for seed in sorted(validation_ids):
        for sample_id in sorted(validation_ids[seed]):
            assigned.setdefault(sample_id, seed)
    return assigned


def main() -> None:
    args = parse_args()
    for name in (
        "manifest",
        "raw_predictions",
        "base_oof_predictions",
        "checkpoint_root",
        "output",
        "meter_detector_weights",
        "keypoint_detector_weights",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    for path in (
        args.manifest,
        args.raw_predictions,
        args.base_oof_predictions,
        args.meter_detector_weights,
        args.keypoint_detector_weights,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    samples, protocol = load_syncg_manifest(args.manifest, expected_split="train")
    manifest_by_id = {sample.sample_id: sample for sample in samples}
    checkpoints, validation_ids = _load_verified_runs(
        checkpoint_root=args.checkpoint_root,
        seeds=args.seeds,
        manifest=args.manifest,
        samples=samples,
        validation_fraction=args.validation_fraction,
    )
    assigned_seed = _assignment(validation_ids)
    raw_by_id = _by_id(args.raw_predictions)
    base_by_id = _by_id(args.base_oof_predictions)
    missing = set(assigned_seed) - set(raw_by_id) | (set(assigned_seed) - set(base_by_id))
    if missing:
        raise ValueError(f"training caches miss {len(missing)} assigned samples")
    assigned_groups = {manifest_by_id[sample_id].group_id for sample_id in assigned_seed}
    train_groups_by_seed = {
        seed: {
            sample.group_id
            for sample in samples
            if sample.sample_id not in validation_ids[seed]
        }
        for seed in validation_ids
    }
    leakage = sorted(
        sample_id
        for sample_id, seed in assigned_seed.items()
        if manifest_by_id[sample_id].group_id in train_groups_by_seed[seed]
    )
    if leakage:
        raise RuntimeError(f"group leakage detected for {len(leakage)} assignments")

    signature = {
        "protocol": QUALITY_ROUTER_OOF_PROTOCOL,
        "split": "SyncG/train only",
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(
            args.manifest.with_name(args.manifest.name + ".protocol.json")
        ),
        "raw_predictions_sha256": sha256_file(args.raw_predictions),
        "base_oof_predictions_sha256": sha256_file(args.base_oof_predictions),
        "direction_runs": {
            str(seed): {
                "checkpoint_sha256": sha256_file(info["path"]),
                "verification_sha256": sha256_file(info["verification"]),
                "validation_samples": len(validation_ids[seed]),
            }
            for seed, info in sorted(checkpoints.items())
        },
        "assignment_policy": "lowest seed whose grouped validation contains sample",
        "assigned_samples": len(assigned_seed),
        "assigned_groups": len(assigned_groups),
        "validation_fraction": float(args.validation_fraction),
        "meter_detector_sha256": sha256_file(args.meter_detector_weights),
        "keypoint_detector_sha256": sha256_file(args.keypoint_detector_weights),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "test_sets_used": [],
    }
    metadata_path = _metadata_path(args.output)
    summary_path = _summary_path(args.output)
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if args.resume:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("resume requires output and metadata files")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("OOF collection resume signature mismatch")
        completed = {str(row.get("sample_id")) for row in _read_jsonl(args.output)}
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch()
        completed = set()
        _atomic_json(
            metadata_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "signature": signature,
                "manifest_protocol": protocol,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                },
            },
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    meter_detector = targetDetectModel(str(args.meter_detector_weights))
    point_detector: targetDetectModel | None = None

    for seed, info in sorted(checkpoints.items()):
        pending_ids = [
            sample.sample_id
            for sample in samples
            if assigned_seed.get(sample.sample_id) == seed and sample.sample_id not in completed
        ]
        if not pending_ids:
            continue
        checkpoint = info["payload"]
        training_signature = checkpoint.get("signature") or {}
        model = build_pivot_direction_model(imagenet_pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        image_size = int(training_signature["image_size"])
        heatmap_size = int(training_signature["heatmap_size"])
        expansion = float(training_signature["expansion"])
        batch: list[dict[str, Any]] = []

        @torch.inference_mode()
        def flush_batch() -> None:
            if not batch:
                return
            inputs = torch.stack([item["tensor"] for item in batch]).to(
                device, non_blocking=True
            )
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                pivot_logits, direction_raw = model(inputs)
            pivot_xy, directions, peaks, valid = decode_pivot_direction(
                pivot_logits.float(), direction_raw.float()
            )
            raw_norm = torch.linalg.vector_norm(direction_raw.float(), dim=1)
            probabilities = torch.sigmoid(pivot_logits.float()[:, 0]).reshape(len(batch), -1)
            spatial = probabilities / torch.clamp(probabilities.sum(dim=1, keepdim=True), min=1e-8)
            entropy = -torch.sum(spatial * torch.log(torch.clamp(spatial, min=1e-12)), dim=1)
            entropy = entropy / math.log(probabilities.shape[1])
            top2 = torch.topk(probabilities, k=2, dim=1).values
            arrays = {
                "pivot": pivot_xy.detach().cpu().numpy(),
                "direction": directions.detach().cpu().numpy(),
                "peak": peaks.detach().cpu().numpy(),
                "valid": valid.detach().cpu().numpy(),
                "raw_norm": raw_norm.detach().cpu().numpy(),
                "entropy": entropy.detach().cpu().numpy(),
                "top2_margin": (top2[:, 0] - top2[:, 1]).detach().cpu().numpy(),
            }
            output_rows: list[dict[str, Any]] = []
            stride = float(image_size) / float(heatmap_size)
            for index, item in enumerate(batch):
                sample = item["sample"]
                base = item["base"]
                raw = item["raw"]
                direction_payload: dict[str, Any] = {
                    "status": False,
                    "prediction": None,
                    "progress": None,
                    "pointer_angle": None,
                    "direction": None,
                    "pivot_heatmap_xy": arrays["pivot"][index].tolist(),
                    "pivot_input_xy": (arrays["pivot"][index] * stride).tolist(),
                    "pivot_peak": float(arrays["peak"][index]),
                    "pivot_spatial_entropy": float(arrays["entropy"][index]),
                    "pivot_top2_margin": float(arrays["top2_margin"][index]),
                    "direction_raw_norm": float(arrays["raw_norm"][index]),
                    "direction_angle_error_degrees": None,
                    "error_code": None,
                }
                if bool(arrays["valid"][index]):
                    direction = arrays["direction"][index].astype(np.float64)
                    try:
                        pointer_angle = image_angle_from_direction(direction)
                        reading, progress = reading_from_pointer_angle(
                            pointer_angle,
                            start_angle=item["start_angle"],
                            range_angle=item["range_angle"],
                            scale_start=sample.scale_start,
                            scale_end=sample.scale_end,
                        )
                        target = item["target_direction"]
                        angle_error = None
                        if target is not None:
                            cosine = float(np.clip(np.dot(direction, target), -1.0, 1.0))
                            angle_error = float(np.degrees(np.arccos(cosine)))
                        direction_payload.update(
                            {
                                "status": True,
                                "prediction": float(reading),
                                "progress": float(progress),
                                "pointer_angle": float(pointer_angle),
                                "direction": direction.tolist(),
                                "direction_angle_error_degrees": angle_error,
                            }
                        )
                    except ValueError as exc:
                        direction_payload["error_code"] = "reading_conversion_failed"
                        direction_payload["error_message"] = str(exc)
                else:
                    direction_payload["error_code"] = "invalid_direction"

                base_prediction = _finite((base.get("predictions") or {}).get("ours"))
                output_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "dataset": sample.dataset,
                        "split": sample.split,
                        "ground_truth": sample.ground_truth,
                        "scale_start": sample.scale_start,
                        "scale_end": sample.scale_end,
                        "held_out_seed": seed,
                        "raw": {
                            "status": raw.get("status") is True,
                            "branch": raw.get("branch"),
                            "error_code": raw.get("error_code"),
                            "features": raw.get("features") or {},
                            "methods": _method_snapshot(raw),
                        },
                        "base": {
                            "status": base_prediction is not None,
                            "prediction": base_prediction,
                            "gate_probability": _finite(base.get("gate_probability")),
                            "residual_normalized": _finite(base.get("residual_normalized")),
                            "residual_std_normalized": _finite(
                                base.get("residual_std_normalized")
                            ),
                            "correction_applied": bool(base.get("correction_applied")),
                        },
                        "vector": direction_payload,
                        "front_end": {
                            "meter_bbox": list(item["bbox"]),
                            "meter_confidence": item["meter_confidence"],
                            "start_angle": item["start_angle"],
                            "range_angle": item["range_angle"],
                            "reference_branch": item["reference_branch"],
                            "reference_source": item["reference_source"],
                        },
                        "runtime_seconds": float(time.perf_counter() - item["started"]),
                    }
                )
            _append_rows(args.output, output_rows)
            completed.update(row["sample_id"] for row in output_rows)
            batch.clear()

        for sample_id in tqdm(
            pending_ids,
            desc=f"quality OOF seed {seed}",
            dynamic_ncols=True,
        ):
            started = time.perf_counter()
            sample = manifest_by_id[sample_id]
            raw = raw_by_id[sample_id]
            base = base_by_id[sample_id]
            image = cv2.imread(
                sample.image_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                raise ValueError(f"cannot read {sample.image_path}")
            confidences, boxes, crops, _, best_index = meter_detector.image_crop(image)
            if best_index is None or not crops:
                # A detector failure is retained as a fully audited vector failure.
                _append_rows(
                    args.output,
                    [
                        {
                            "sample_id": sample.sample_id,
                            "group_id": sample.group_id,
                            "dataset": sample.dataset,
                            "split": sample.split,
                            "ground_truth": sample.ground_truth,
                            "scale_start": sample.scale_start,
                            "scale_end": sample.scale_end,
                            "held_out_seed": seed,
                            "raw": {
                                "status": raw.get("status") is True,
                                "branch": raw.get("branch"),
                                "error_code": raw.get("error_code"),
                                "features": raw.get("features") or {},
                                "methods": _method_snapshot(raw),
                            },
                            "base": {
                                "status": _finite((base.get("predictions") or {}).get("ours"))
                                is not None,
                                "prediction": _finite(
                                    (base.get("predictions") or {}).get("ours")
                                ),
                                "gate_probability": _finite(base.get("gate_probability")),
                                "residual_normalized": _finite(
                                    base.get("residual_normalized")
                                ),
                                "residual_std_normalized": _finite(
                                    base.get("residual_std_normalized")
                                ),
                                "correction_applied": bool(
                                    base.get("correction_applied")
                                ),
                            },
                            "vector": {
                                "status": False,
                                "prediction": None,
                                "error_code": "meter_not_found",
                            },
                            "front_end": {
                                "meter_bbox": None,
                                "meter_confidence": None,
                                "reference_source": None,
                            },
                            "runtime_seconds": float(time.perf_counter() - started),
                        }
                    ],
                )
                completed.add(sample.sample_id)
                continue
            crop = crops[best_index]
            bbox = _xyxy_from_detector_box(boxes[best_index])
            reference = _shared_reference(raw)
            if reference is None:
                if point_detector is None:
                    point_detector = targetDetectModel(str(args.keypoint_detector_weights))
                reference = _fallback_reference(point_detector, crop)
                reference_source = "fallback_keypoint_detector"
            else:
                reference_source = "shared_frozen_production_cache"
            start_angle, range_angle, reference_branch = reference
            manifest_row = {
                "metadata": sample.metadata,
            }
            batch.append(
                {
                    "sample": sample,
                    "raw": raw,
                    "base": base,
                    "tensor": tensor_from_bbox(
                        image,
                        bbox,
                        image_size=image_size,
                        expansion=expansion,
                    ),
                    "bbox": bbox,
                    "meter_confidence": float(confidences[best_index]),
                    "start_angle": float(start_angle),
                    "range_angle": float(range_angle),
                    "reference_branch": str(reference_branch),
                    "reference_source": reference_source,
                    "target_direction": _target_direction(manifest_row, {}),
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        flush_batch()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = _read_jsonl(args.output)
    output_ids = [str(row.get("sample_id")) for row in rows]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(assigned_seed):
        raise RuntimeError("OOF output is incomplete or contains duplicate sample IDs")
    base_success = sum((row.get("base") or {}).get("status") is True for row in rows)
    vector_success = sum((row.get("vector") or {}).get("status") is True for row in rows)
    joint_success = sum(
        (row.get("base") or {}).get("status") is True
        and (row.get("vector") or {}).get("status") is True
        for row in rows
    )
    membership_counts = {
        str(count): sum(
            sum(sample_id in ids for ids in validation_ids.values()) == count
            for sample_id in assigned_seed
        )
        for count in range(1, len(validation_ids) + 1)
    }
    summary = {
        "schema_version": 1,
        "protocol": QUALITY_ROUTER_OOF_PROTOCOL,
        "status": "complete",
        "signature": signature,
        "samples": len(rows),
        "groups": len(assigned_groups),
        "base_success": base_success,
        "vector_success": vector_success,
        "joint_success": joint_success,
        "validation_membership_counts": membership_counts,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
