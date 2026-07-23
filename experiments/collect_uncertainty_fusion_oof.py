"""Collect leakage-free probabilistic direction predictions for fusion training.

This collector deliberately reuses the already signed mask-side OOF rows and
their frozen meter/reference geometry.  Each vector prediction is produced by
the probabilistic direction checkpoint whose grouped validation split contains
the whole meter group.  Therefore neither expert has fitted the row it labels,
and no SyncG test or RPM-10K sample is admitted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
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

from experiments.pivot_direction_fallback import tensor_from_bbox
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import (
    grouped_train_val_split,
    image_angle_from_direction,
    load_syncg_manifest,
    reading_from_pointer_angle,
    sample_ids_hash,
    sha256_file,
)


DEFAULT_SEEDS = (20260720, 20260721, 20260722)
EXPECTED_SOURCE_OOF_PROTOCOL = "syncg_quality_router_cross_model_oof_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--source-oof",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/oof_clean.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/runs/probabilistic_pivot_direction_syncg"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/probabilistic_oof_clean.jsonl"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


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


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_verified_runs(
    checkpoint_root: Path,
    seeds: Sequence[int],
    manifest: Path,
    samples: Sequence[Any],
    validation_fraction: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, set[str]], dict[int, set[str]]]:
    runs: dict[int, dict[str, Any]] = {}
    validation_ids: dict[int, set[str]] = {}
    training_groups: dict[int, set[str]] = {}
    manifest_hash = sha256_file(manifest)
    verifier_source = (
        PROJECT_ROOT / "experiments" / "verify_probabilistic_pivot_direction_run.py"
    )
    for seed in sorted(set(map(int, seeds))):
        run_dir = checkpoint_root / f"seed_{seed}"
        checkpoint_path = run_dir / "best.pt"
        verification_path = run_dir / "verification.json"
        for path in (checkpoint_path, verification_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        signature = checkpoint.get("signature") or {}
        verification = _read_json(verification_path)
        if signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
            raise ValueError(f"wrong checkpoint protocol: {checkpoint_path}")
        if int(signature.get("seed", -1)) != seed:
            raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
        if signature.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"checkpoint manifest mismatch: {checkpoint_path}")
        if not math.isclose(
            float(signature.get("validation_fraction", math.nan)),
            float(validation_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"checkpoint validation fraction mismatch: {checkpoint_path}")
        if verification.get("verified") is not True:
            raise ValueError(f"run did not pass formal verification: {verification_path}")
        if verification.get("best_checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError(f"verification/checkpoint mismatch: {run_dir}")
        if verification.get("verifier_source_sha256") != sha256_file(verifier_source):
            raise ValueError("probabilistic training verifier changed after verification")
        train, validation = grouped_train_val_split(
            samples,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        if sample_ids_hash(validation) != signature.get("validation_sample_ids_sha256"):
            raise ValueError(f"cannot reconstruct validation split for seed {seed}")
        runs[seed] = {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "verification_path": verification_path,
        }
        validation_ids[seed] = {sample.sample_id for sample in validation}
        training_groups[seed] = {sample.group_id for sample in train}
    return runs, validation_ids, training_groups


def main() -> None:
    args = parse_args()
    for name in ("manifest", "source_oof", "checkpoint_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    for path in (args.manifest, args.source_oof):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_meta_path = _metadata_path(args.source_oof)
    source_summary_path = _summary_path(args.source_oof)
    source_meta = _read_json(source_meta_path)
    source_summary = _read_json(source_summary_path)
    source_signature = source_meta.get("signature") or {}
    if source_signature.get("protocol") != EXPECTED_SOURCE_OOF_PROTOCOL:
        raise ValueError("source OOF has the wrong protocol")
    if source_signature.get("test_sets_used") != []:
        raise ValueError("source OOF does not certify train-only collection")
    if (
        source_summary.get("status") != "complete"
        or int(source_summary.get("group_leakage_count", -1)) != 0
        or int(source_summary.get("test_samples_used", -1)) != 0
        or source_summary.get("output_sha256") != sha256_file(args.source_oof)
    ):
        raise ValueError("source OOF summary is incomplete or inconsistent")

    samples, manifest_protocol = load_syncg_manifest(
        args.manifest,
        expected_split="train",
    )
    sample_by_id = {sample.sample_id: sample for sample in samples}
    source_rows = _read_jsonl(args.source_oof)
    source_by_id = {str(row.get("sample_id")): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("source OOF contains duplicate sample IDs")
    if not set(source_by_id).issubset(sample_by_id):
        raise ValueError("source OOF contains samples outside the SyncG train manifest")

    runs, validation_ids, training_groups = _load_verified_runs(
        args.checkpoint_root,
        args.seeds,
        args.manifest,
        samples,
        args.validation_fraction,
    )
    assignments: dict[str, int] = {}
    leakage: list[str] = []
    for row in source_rows:
        sample_id = str(row.get("sample_id"))
        seed = int(row.get("held_out_seed", -1))
        if seed not in runs or sample_id not in validation_ids[seed]:
            raise ValueError(f"{sample_id}: held-out seed is not a v2 validation member")
        if sample_by_id[sample_id].group_id in training_groups[seed]:
            leakage.append(sample_id)
        assignments[sample_id] = seed
    if leakage:
        raise RuntimeError(f"group leakage detected for {len(leakage)} samples")

    signature = {
        "protocol": UNCERTAINTY_FUSION_OOF_PROTOCOL,
        "split": "SyncG/train only",
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(
            args.manifest.with_name(args.manifest.name + ".protocol.json")
        ),
        "source_oof_sha256": sha256_file(args.source_oof),
        "source_oof_metadata_sha256": sha256_file(source_meta_path),
        "source_oof_summary_sha256": sha256_file(source_summary_path),
        "direction_runs": {
            str(seed): {
                "checkpoint_sha256": sha256_file(info["checkpoint_path"]),
                "verification_sha256": sha256_file(info["verification_path"]),
                "validation_samples": len(validation_ids[seed]),
            }
            for seed, info in sorted(runs.items())
        },
        "assignment_policy": "reuse signed v1 OOF held-out seed after exact v2 membership check",
        "assigned_samples": len(assignments),
        "assigned_groups": len({row["group_id"] for row in source_rows}),
        "validation_fraction": float(args.validation_fraction),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "test_sets_used": [],
    }
    metadata_path = _metadata_path(args.output)
    summary_path = _summary_path(args.output)
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")
    if args.overwrite:
        for path in (args.output, metadata_path, summary_path):
            path.unlink(missing_ok=True)
    if args.resume:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("resume requires output and metadata files")
        if (_read_json(metadata_path).get("signature") or {}) != signature:
            raise ValueError("OOF resume signature mismatch")
        existing_rows = _read_jsonl(args.output)
        completed = {str(row.get("sample_id")) for row in existing_rows}
        if len(completed) != len(existing_rows):
            raise ValueError("existing OOF output contains duplicate IDs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch()
        completed: set[str] = set()
        _atomic_json(
            metadata_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "signature": signature,
                "manifest_protocol": manifest_protocol,
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

    for seed, info in sorted(runs.items()):
        checkpoint = info["checkpoint"]
        training_signature = checkpoint.get("signature") or {}
        pending = [
            row
            for row in source_rows
            if assignments[str(row.get("sample_id"))] == seed
            and str(row.get("sample_id")) not in completed
        ]
        if not pending:
            continue
        model = build_probabilistic_pivot_direction_model(
            angle_bins=int(training_signature["angle_bins"]),
            imagenet_pretrained=False,
        )
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
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = model(inputs)
            prediction = decode_probabilistic_pivot_direction(
                *(value.float() for value in outputs)
            )
            pivot_prob = torch.sigmoid(outputs[0].float()[:, 0]).reshape(len(batch), -1)
            spatial = pivot_prob / torch.clamp(pivot_prob.sum(dim=1, keepdim=True), min=1e-8)
            pivot_entropy = -torch.sum(
                spatial * torch.log(torch.clamp(spatial, min=1e-12)), dim=1
            ) / math.log(float(pivot_prob.shape[1]))
            top2 = torch.topk(pivot_prob, k=2, dim=1).values
            raw_norm = torch.linalg.vector_norm(outputs[1].float(), dim=1)
            arrays = {
                "pivot": prediction.pivot_xy.cpu().numpy(),
                "direction": prediction.direction.cpu().numpy(),
                "peak": prediction.pivot_peak.cpu().numpy(),
                "valid": prediction.valid.cpu().numpy(),
                "angle_std": prediction.angle_std_degrees.cpu().numpy(),
                "angle_entropy": prediction.angle_entropy.cpu().numpy(),
                "log_variance": prediction.log_variance.cpu().numpy(),
                "bin_resultant": prediction.bin_resultant_length.cpu().numpy(),
                "pivot_entropy": pivot_entropy.cpu().numpy(),
                "pivot_margin": (top2[:, 0] - top2[:, 1]).cpu().numpy(),
                "raw_norm": raw_norm.cpu().numpy(),
            }
            stride = float(image_size) / float(heatmap_size)
            output_rows: list[dict[str, Any]] = []
            for index, item in enumerate(batch):
                source = item["source"]
                sample = item["sample"]
                payload: dict[str, Any] = {
                    "status": False,
                    "prediction": None,
                    "progress": None,
                    "pointer_angle": None,
                    "direction": None,
                    "pivot_heatmap_xy": arrays["pivot"][index].tolist(),
                    "pivot_input_xy": (arrays["pivot"][index] * stride).tolist(),
                    "pivot_peak": float(arrays["peak"][index]),
                    "pivot_spatial_entropy": float(arrays["pivot_entropy"][index]),
                    "pivot_top2_margin": float(arrays["pivot_margin"][index]),
                    "direction_raw_norm": float(arrays["raw_norm"][index]),
                    "angle_std_degrees": float(arrays["angle_std"][index]),
                    "angle_log_variance": float(arrays["log_variance"][index]),
                    "angle_bin_entropy": float(arrays["angle_entropy"][index]),
                    "angle_bin_resultant_length": float(arrays["bin_resultant"][index]),
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
                        target = np.asarray(sample.pointer_tip) - np.asarray(sample.pointer_tail)
                        target /= max(float(np.linalg.norm(target)), 1e-12)
                        cosine = float(np.clip(np.dot(direction, target), -1.0, 1.0))
                        payload.update(
                            {
                                "status": True,
                                "prediction": float(reading),
                                "progress": float(progress),
                                "pointer_angle": float(pointer_angle),
                                "direction": direction.tolist(),
                                "direction_angle_error_degrees": float(
                                    np.degrees(np.arccos(cosine))
                                ),
                            }
                        )
                    except ValueError as exc:
                        payload["error_code"] = "reading_conversion_failed"
                        payload["error_message"] = str(exc)
                else:
                    payload["error_code"] = "invalid_direction"
                row = dict(source)
                row["vector"] = payload
                row["runtime_seconds"] = float(time.perf_counter() - item["started"])
                row["vector_protocol"] = PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
                output_rows.append(row)
            _append_rows(args.output, output_rows)
            completed.update(str(row["sample_id"]) for row in output_rows)
            batch.clear()

        for source in tqdm(pending, desc=f"probabilistic OOF {seed}", dynamic_ncols=True):
            started = time.perf_counter()
            sample_id = str(source.get("sample_id"))
            sample = sample_by_id[sample_id]
            front_end = source.get("front_end") or {}
            bbox = front_end.get("meter_bbox")
            start_angle = _finite(front_end.get("start_angle"))
            range_angle = _finite(front_end.get("range_angle"))
            if (
                not isinstance(bbox, Sequence)
                or len(bbox) < 4
                or start_angle is None
                or range_angle is None
                or abs(range_angle) <= 1e-8
            ):
                row = dict(source)
                row["vector"] = {
                    "status": False,
                    "prediction": None,
                    "error_code": "frozen_front_end_failed",
                }
                row["vector_protocol"] = PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
                row["runtime_seconds"] = float(time.perf_counter() - started)
                _append_rows(args.output, [row])
                completed.add(sample_id)
                continue
            image = cv2.imread(
                sample.image_path,
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                raise ValueError(f"cannot read {sample.image_path}")
            batch.append(
                {
                    "source": source,
                    "sample": sample,
                    "tensor": tensor_from_bbox(
                        image,
                        [float(value) for value in bbox[:4]],
                        image_size=image_size,
                        expansion=expansion,
                    ),
                    "start_angle": start_angle,
                    "range_angle": range_angle,
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        flush_batch()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_rows = _read_jsonl(args.output)
    output_ids = [str(row.get("sample_id")) for row in output_rows]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != set(assignments):
        raise RuntimeError("probabilistic OOF output is incomplete or duplicated")
    membership_counts = Counter(
        sum(sample_id in ids for ids in validation_ids.values()) for sample_id in output_ids
    )
    base_success = sum((row.get("base") or {}).get("status") is True for row in output_rows)
    vector_success = sum((row.get("vector") or {}).get("status") is True for row in output_rows)
    joint_success = sum(
        (row.get("base") or {}).get("status") is True
        and (row.get("vector") or {}).get("status") is True
        for row in output_rows
    )
    summary = {
        "schema_version": 1,
        "protocol": UNCERTAINTY_FUSION_OOF_PROTOCOL,
        "status": "complete",
        "samples": len(output_rows),
        "groups": len({str(row.get("group_id")) for row in output_rows}),
        "base_success": base_success,
        "vector_success": vector_success,
        "joint_success": joint_success,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "validation_membership_counts": {
            str(key): value for key, value in sorted(membership_counts.items())
        },
        "signature": signature,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
