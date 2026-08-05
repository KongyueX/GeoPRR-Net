"""Evaluate the independent pivot-direction fallback on frozen front-end data."""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from experiments.evaluate_vdn_baseline import (
    _append_rows,
    _base_result,
    _component_summary,
    _dialbench_summary,
    _failure_result,
    _metadata_path,
    _read_jsonl,
    _subgroup_summaries,
    _summary_path,
    _target_direction,
)
from experiments.pivot_direction_fallback import (
    PIVOT_DIRECTION_PROTOCOL,
    build_pivot_direction_model,
    decode_pivot_direction,
    tensor_from_bbox,
)
from experiments.robustness_degradations import (
    ROBUSTNESS_PROTOCOL,
    apply_degradation,
    degradation_names,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    image_angle_from_direction,
    reading_from_pointer_angle,
    sha256_file,
    summarize_scalar_predictions,
)


EVALUATION_PROTOCOL = "pivot_direction_fallback_e2e_v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a mutable legacy evaluation JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--condition", choices=degradation_names(), default="clean")
    parser.add_argument("--degradation-seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _load_reference_rows(
    path: Path,
    *,
    manifest: Path,
    rows: list[dict[str, Any]],
    condition: str,
    degradation_seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata_path = _metadata_path(path)
    summary_path = _summary_path(path)
    for required in (path, metadata_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if summary.get("status") != "complete":
        raise ValueError("reference front-end evaluation is incomplete")
    if signature.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("reference front-end uses a different manifest")
    protocol_path = _protocol_path(manifest)
    expected_protocol_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    if signature.get("manifest_protocol_sha256") != expected_protocol_hash:
        raise ValueError("reference front-end uses a different manifest protocol")
    if signature.get("condition") != condition:
        raise ValueError("reference front-end uses a different degradation condition")
    if int(signature.get("degradation_seed", -1)) != int(degradation_seed):
        raise ValueError("reference front-end uses a different degradation seed")
    if signature.get("degradation_protocol") != ROBUSTNESS_PROTOCOL:
        raise ValueError("reference front-end uses a different degradation protocol")
    expected_degradation_hash = sha256_file(
        PROJECT_DIR / "experiments" / "robustness_degradations.py"
    )
    source_hashes = signature.get("source_sha256") or {}
    if source_hashes.get("degradation") != expected_degradation_hash:
        raise ValueError("reference degradation source changed")
    reference_rows = _read_jsonl(path)
    by_id = {str(row.get("sample_id")): row for row in reference_rows}
    if len(by_id) != len(reference_rows):
        raise ValueError("reference front-end contains duplicate sample IDs")
    selected_ids = {str(row.get("sample_id")) for row in rows}
    missing_ids = selected_ids - set(by_id)
    if missing_ids:
        raise ValueError(
            f"reference front-end misses {len(missing_ids)} selected sample IDs"
        )
    invalid_failure_codes = sorted(
        {
            str(row.get("error_code"))
            for row in reference_rows
            if row.get("status") is not True
            and row.get("error_code") not in {"meter_not_found", "image_read_failed"}
        }
    )
    if invalid_failure_codes:
        raise ValueError(
            "reference rows include direction-dependent failures: "
            f"{invalid_failure_codes}"
        )
    for row in reference_rows:
        if row.get("status") is not True:
            continue
        required_fields = ("meter_bbox", "start_angle", "range_angle")
        if any(row.get(field) is None for field in required_fields):
            raise ValueError("successful reference row is missing front-end geometry")
    return by_id, metadata


def _evaluation_signature(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    verification: dict[str, Any],
    reference_metadata: dict[str, Any],
) -> dict[str, Any]:
    protocol_path = _protocol_path(args.manifest)
    return {
        "protocol": EVALUATION_PROTOCOL,
        "training_protocol": PIVOT_DIRECTION_PROTOCOL,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_signature": checkpoint.get("signature"),
        "verification_sha256": sha256_file(args.verification),
        "verification_summary_sha256": verification.get("summary_sha256"),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": (
            sha256_file(protocol_path) if protocol_path.is_file() else None
        ),
        "condition": args.condition,
        "degradation_protocol": ROBUSTNESS_PROTOCOL,
        "degradation_seed": int(args.degradation_seed),
        "reference_predictions_sha256": sha256_file(args.reference_predictions),
        "reference_predictions_signature": reference_metadata.get("signature"),
        "image_size": int((checkpoint.get("signature") or {}).get("image_size", 256)),
        "batch_size": int(args.batch_size),
        "diagnostic_limit": args.limit,
        "source_sha256": {
            "model": sha256_file(
                PROJECT_DIR / "experiments" / "pivot_direction_fallback.py"
            ),
            "evaluation": sha256_file(Path(__file__).resolve()),
            "degradation": sha256_file(
                PROJECT_DIR / "experiments" / "robustness_degradations.py"
            ),
        },
        "routing_role": "segmentation-independent direction fallback",
        "front_end_policy": "frozen meter box and start/end references",
        "crop_policy": "square 1.25 expansion around frozen detected xyxy box",
        "failure_nmae_penalty": 1.0,
    }


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.verification = (
        args.verification.resolve()
        if args.verification is not None
        else args.checkpoint.parent.resolve() / "verification.json"
    )
    args.reference_predictions = args.reference_predictions.resolve()
    args.output = args.output.resolve()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0 or args.bootstrap_iterations < 0:
        raise ValueError("batch size must be positive and bootstrap iterations non-negative")
    for path in (args.manifest, args.checkpoint, args.verification):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = _read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise ValueError("selected manifest is empty")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_signature = checkpoint.get("signature") or {}
    if training_signature.get("protocol") != PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("checkpoint is not a signed pivot-direction artifact")
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("verified") is not True:
        raise ValueError("formal training verification is missing or failed")
    checkpoint_hash = sha256_file(args.checkpoint)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("verification belongs to a different checkpoint")
    if verification.get("verifier_source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "verify_pivot_direction_run.py"
    ):
        raise ValueError("training verifier source changed")
    reference, reference_metadata = _load_reference_rows(
        args.reference_predictions,
        manifest=args.manifest,
        rows=rows,
        condition=args.condition,
        degradation_seed=args.degradation_seed,
    )
    signature = _evaluation_signature(
        args,
        checkpoint,
        verification,
        reference_metadata,
    )
    metadata_path = _metadata_path(args.output)
    summary_path = _summary_path(args.output)
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")
    if args.overwrite:
        args.output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    completed: set[str] = set()
    if args.resume:
        if not args.output.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("resume requires output and metadata files")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError("fallback evaluation resume signature mismatch")
        completed = {str(row.get("sample_id")) for row in _read_jsonl(args.output)}
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch()
        _atomic_json(
            metadata_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": str(args.manifest),
                "checkpoint": str(args.checkpoint),
                "verification": str(args.verification),
                "reference_predictions": str(args.reference_predictions),
                "signature": signature,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                },
            },
        )

    pending = [row for row in rows if str(row.get("sample_id")) not in completed]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    model = build_pivot_direction_model(imagenet_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    image_size = int(training_signature["image_size"])
    heatmap_size = int(training_signature["heatmap_size"])
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
            pivot_logits, direction_raw = model(inputs)
        pivot_xy, directions, peaks, valid = decode_pivot_direction(
            pivot_logits.float(), direction_raw.float()
        )
        pivots_np = pivot_xy.detach().cpu().numpy()
        directions_np = directions.detach().cpu().numpy()
        peaks_np = peaks.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        output_rows: list[dict[str, Any]] = []
        for index, item in enumerate(batch):
            row = item["row"]
            degradation = item["degradation"]
            runtime = time.perf_counter() - item["started"]
            if not bool(valid_np[index]):
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="invalid_direction",
                        message="pivot-direction head returned a zero/invalid vector",
                        runtime_seconds=runtime,
                    )
                )
                continue
            direction = directions_np[index].astype(np.float64)
            try:
                pointer_angle = image_angle_from_direction(direction)
                reading, progress = reading_from_pointer_angle(
                    pointer_angle,
                    start_angle=item["start_angle"],
                    range_angle=item["range_angle"],
                    scale_start=float(row["scale_start"]),
                    scale_end=float(row["scale_end"]),
                )
            except ValueError as exc:
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="reading_conversion_failed",
                        message=str(exc),
                        runtime_seconds=runtime,
                    )
                )
                continue
            target = item["target_direction"]
            direction_error = None
            if target is not None:
                cosine = float(np.clip(np.dot(direction, target), -1.0, 1.0))
                direction_error = float(np.degrees(np.arccos(cosine)))
            stride = float(image_size) / float(heatmap_size)
            result = _base_result(row, degradation)
            result.update(
                {
                    "status": True,
                    "prediction": float(reading),
                    "progress": float(progress),
                    "pointer_angle": float(pointer_angle),
                    "direction": direction.tolist(),
                    "direction_angle_error_degrees": direction_error,
                    "pivot_heatmap_xy": pivots_np[index].tolist(),
                    "pivot_input_xy": (pivots_np[index] * stride).tolist(),
                    "pivot_peak": float(peaks_np[index]),
                    "meter_bbox": item["meter_bbox"],
                    "start_angle": float(item["start_angle"]),
                    "range_angle": float(item["range_angle"]),
                    "reference_branch": item["reference_branch"],
                    "reference_source": "frozen_front_end_reference",
                    "runtime_seconds": float(runtime),
                }
            )
            output_rows.append(result)
        _append_rows(args.output, output_rows)
        batch.clear()

    for row in tqdm(pending, desc=f"fallback {args.condition}", dynamic_ncols=True):
        started = time.perf_counter()
        sample_id = str(row.get("sample_id"))
        front_end = reference[sample_id]
        degradation: dict[str, Any] = {
            "protocol": ROBUSTNESS_PROTOCOL,
            "condition": args.condition,
            "seed": int(args.degradation_seed),
        }
        if front_end.get("status") is not True:
            _append_rows(
                args.output,
                [
                    _failure_result(
                        row,
                        degradation,
                        code=str(front_end.get("error_code") or "front_end_failed"),
                        message="frozen meter/reference front end did not produce geometry",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
            continue
        image = cv2.imread(
            str(row.get("image_path")),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            _append_rows(
                args.output,
                [
                    _failure_result(
                        row,
                        degradation,
                        code="image_read_failed",
                        message=f"failed to read {row.get('image_path')}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
            continue
        try:
            image, degradation = apply_degradation(
                image,
                args.condition,
                sample_id=sample_id,
                seed=args.degradation_seed,
            )
            bbox = [float(value) for value in front_end["meter_bbox"]]
            batch.append(
                {
                    "row": row,
                    "degradation": degradation,
                    "tensor": tensor_from_bbox(
                        image,
                        bbox,
                        image_size=image_size,
                        expansion=float(training_signature["expansion"]),
                    ),
                    "meter_bbox": bbox,
                    "start_angle": float(front_end["start_angle"]),
                    "range_angle": float(front_end["range_angle"]),
                    "reference_branch": str(front_end.get("reference_branch") or "unknown"),
                    "target_direction": _target_direction(row, degradation),
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            _append_rows(
                args.output,
                [
                    _failure_result(
                        row,
                        degradation,
                        code="pipeline_exception",
                        message=f"{type(exc).__name__}: {exc}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
    flush_batch()

    output_rows = _read_jsonl(args.output)
    expected_ids = {str(row.get("sample_id")) for row in rows}
    output_ids = [str(row.get("sample_id")) for row in output_rows]
    if len(output_ids) != len(set(output_ids)) or set(output_ids) != expected_ids:
        raise RuntimeError("fallback output is incomplete or contains duplicate IDs")
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "condition": args.condition,
        "signature": signature,
        "metrics": summarize_scalar_predictions(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "direction_component": _component_summary(output_rows),
        "dialbench_metrics": _dialbench_summary(output_rows),
        "subgroups": _subgroup_summaries(output_rows),
        "failure_counts": dict(
            sorted(
                Counter(
                    str(row.get("error_code") or "success") for row in output_rows
                ).items()
            )
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "samples": len(output_rows),
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
