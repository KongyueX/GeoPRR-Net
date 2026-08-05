"""Evaluate the frozen probabilistic direction expert on one condition."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.evaluate_pivot_direction_fallback import (
    _atomic_json,
    _load_reference_rows,
    _protocol_path,
)
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
from experiments.pivot_direction_fallback import tensor_from_bbox
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.robustness_degradations import (
    ROBUSTNESS_PROTOCOL,
    apply_degradation,
    degradation_names,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    image_angle_from_direction,
    reading_from_pointer_angle,
    sha256_file,
    sha256_source_file,
    summarize_scalar_predictions,
)


EVALUATION_PROTOCOL = "probabilistic_pivot_direction_e2e_v1"
LEGACY_VERIFIER_SOURCE_SHA256 = {
    "formal_probabilistic_direction_run_verification_v1": (
        "b5fadf7008bb2a8a3b202886b53d2514c0dc41db0e455719f990a590bee785a6"
    ),
    "formal_probabilistic_direction_ablation_verification_v1": (
        "3e8cf11df760944c98ecaae4dcc4298f23c4b66f19a4ebe1b31346bc3f1fe542"
    ),
}


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
    parser.add_argument(
        "--direction-decoder",
        choices=("fused", "direct", "circular"),
        default="fused",
        help="Inference-only decoder ablation; training weights stay frozen.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _evaluation_signature(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    verification: dict[str, Any],
    reference_metadata: dict[str, Any],
) -> dict[str, Any]:
    protocol_path = _protocol_path(args.manifest)
    training_signature = checkpoint.get("signature") or {}
    return {
        "protocol": EVALUATION_PROTOCOL,
        "training_protocol": PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_signature": training_signature,
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
        "image_size": int(training_signature.get("image_size", 256)),
        "angle_bins": int(training_signature.get("angle_bins", 72)),
        "direction_decoder": args.direction_decoder,
        "batch_size": int(args.batch_size),
        "diagnostic_limit": args.limit,
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "source_sha256": {
            "model": sha256_source_file(
                PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
            ),
            "evaluation": sha256_source_file(Path(__file__).resolve()),
            "degradation": sha256_source_file(
                PROJECT_DIR / "experiments" / "robustness_degradations.py"
            ),
        },
        "routing_role": "probabilistic segmentation-independent direction expert",
        "front_end_policy": "frozen meter box and start/end references",
        "crop_policy": "square 1.25 expansion around frozen detected xyxy box",
        "failure_nmae_penalty": 1.0,
    }


def _select_direction_decoder(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    fused_prediction,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a frozen inference decoder for circular-head ablation."""

    if mode == "fused":
        return fused_prediction.direction, fused_prediction.valid
    direction_raw = outputs[1].float()
    if mode == "direct":
        norm = torch.linalg.vector_norm(direction_raw, dim=1)
        direction = F.normalize(direction_raw, dim=1, eps=1e-8)
        valid = torch.isfinite(direction).all(dim=1) & (norm > 1e-8)
        return direction, valid
    if mode == "circular":
        logits = outputs[2].float()
        probability = torch.softmax(logits, dim=1)
        centers = torch.arange(
            logits.shape[1], device=logits.device, dtype=probability.dtype
        ) * (2.0 * math.pi / float(logits.shape[1]))
        vector = torch.stack(
            (
                torch.sum(probability * torch.cos(centers), dim=1),
                torch.sum(probability * torch.sin(centers), dim=1),
            ),
            dim=1,
        )
        norm = torch.linalg.vector_norm(vector, dim=1)
        direction = F.normalize(vector, dim=1, eps=1e-8)
        valid = torch.isfinite(direction).all(dim=1) & (norm > 1e-8)
        return direction, valid
    raise ValueError(f"unsupported direction decoder: {mode}")


def _uncertainty_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    successful = [
        row
        for row in rows
        if row.get("status") is True
        and row.get("direction_angle_error_degrees") is not None
        and row.get("angle_std_degrees") is not None
    ]
    if not successful:
        return {
            "samples": 0,
            "mean_angle_std_degrees": math.nan,
            "angle_within_1sigma": 0.0,
            "angle_within_2sigma": 0.0,
            "uncertainty_error_correlation": math.nan,
            "angular_calibration_nll": math.nan,
        }
    errors = np.asarray(
        [float(row["direction_angle_error_degrees"]) for row in successful],
        dtype=np.float64,
    )
    stds = np.asarray(
        [float(row["angle_std_degrees"]) for row in successful],
        dtype=np.float64,
    )
    variances = np.maximum(np.square(np.deg2rad(stds)), 1e-12)
    radians = np.deg2rad(errors)
    correlation = (
        float(np.corrcoef(errors, stds)[0, 1])
        if len(errors) > 1 and np.std(errors) > 0.0 and np.std(stds) > 0.0
        else math.nan
    )
    return {
        "samples": len(successful),
        "mean_angle_std_degrees": float(np.mean(stds)),
        "median_angle_std_degrees": float(np.median(stds)),
        "angle_within_1sigma": float(np.mean(errors <= stds)),
        "angle_within_2sigma": float(np.mean(errors <= 2.0 * stds)),
        "uncertainty_error_correlation": correlation,
        "angular_calibration_nll": float(
            np.mean(0.5 * (np.square(radians) / variances + np.log(variances)))
        ),
    }


def _validate_training_verification_source(
    verification: dict[str, Any],
) -> Path:
    verification_protocol = str(verification.get("protocol") or "")
    verifier_sources = {
        "formal_probabilistic_direction_run_verification_v1": (
            PROJECT_DIR
            / "experiments"
            / "verify_probabilistic_pivot_direction_run.py"
        ),
        "formal_probabilistic_direction_run_verification_v2": (
            PROJECT_DIR
            / "experiments"
            / "verify_probabilistic_pivot_direction_run.py"
        ),
        "formal_probabilistic_direction_ablation_verification_v1": (
            PROJECT_DIR
            / "experiments"
            / "verify_probabilistic_direction_ablation.py"
        ),
        "formal_probabilistic_direction_ablation_verification_v2": (
            PROJECT_DIR
            / "experiments"
            / "verify_probabilistic_direction_ablation.py"
        ),
    }
    verifier_source = verifier_sources.get(verification_protocol)
    verification_source_hash_protocol = verification.get("source_hash_protocol")
    expected_verifier_hash = LEGACY_VERIFIER_SOURCE_SHA256.get(
        verification_protocol
    )
    if expected_verifier_hash is None and (
        verification_source_hash_protocol != SOURCE_TEXT_SHA256_PROTOCOL
    ):
        raise ValueError(
            "training verification uses an unsupported source hash protocol"
        )
    if (
        expected_verifier_hash is not None
        and verification_source_hash_protocol is not None
    ):
        raise ValueError("legacy training verification has unexpected hash metadata")
    if expected_verifier_hash is None and verifier_source is not None:
        expected_verifier_hash = sha256_source_file(verifier_source)
    if verifier_source is None or verification.get(
        "verifier_source_sha256"
    ) != expected_verifier_hash:
        raise ValueError("training verifier source changed or is unsupported")
    return verifier_source


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
    if training_signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("checkpoint is not a signed probabilistic direction artifact")
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("verified") is not True:
        raise ValueError("formal training verification is missing or failed")
    checkpoint_hash = sha256_file(args.checkpoint)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("verification belongs to a different checkpoint")
    _validate_training_verification_source(verification)
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
            raise ValueError("probabilistic evaluation resume signature mismatch")
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
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(training_signature["angle_bins"]),
        imagenet_pretrained=False,
    )
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
            outputs = model(inputs)
        prediction = decode_probabilistic_pivot_direction(
            *(value.float() for value in outputs)
        )
        selected_direction, selected_valid = _select_direction_decoder(
            outputs,
            prediction,
            args.direction_decoder,
        )
        pivot_probabilities = torch.sigmoid(outputs[0].float()[:, 0]).reshape(
            len(batch), -1
        )
        spatial = pivot_probabilities / torch.clamp(
            pivot_probabilities.sum(dim=1, keepdim=True), min=1e-8
        )
        pivot_entropy = -torch.sum(
            spatial * torch.log(torch.clamp(spatial, min=1e-12)), dim=1
        ) / math.log(float(pivot_probabilities.shape[1]))
        top2 = torch.topk(pivot_probabilities, k=2, dim=1).values
        raw_norm = torch.linalg.vector_norm(outputs[1].float(), dim=1)
        arrays = {
            "pivot": prediction.pivot_xy.detach().cpu().numpy(),
            "direction": selected_direction.detach().cpu().numpy(),
            "peak": prediction.pivot_peak.detach().cpu().numpy(),
            "valid": (prediction.valid & selected_valid).detach().cpu().numpy(),
            "angle_std": prediction.angle_std_degrees.detach().cpu().numpy(),
            "angle_entropy": prediction.angle_entropy.detach().cpu().numpy(),
            "log_variance": prediction.log_variance.detach().cpu().numpy(),
            "bin_resultant": prediction.bin_resultant_length.detach().cpu().numpy(),
            "pivot_entropy": pivot_entropy.detach().cpu().numpy(),
            "pivot_margin": (top2[:, 0] - top2[:, 1]).detach().cpu().numpy(),
            "raw_norm": raw_norm.detach().cpu().numpy(),
        }
        output_rows: list[dict[str, Any]] = []
        stride = float(image_size) / float(heatmap_size)
        for index, item in enumerate(batch):
            row = item["row"]
            degradation = item["degradation"]
            runtime = time.perf_counter() - item["started"]
            if not bool(arrays["valid"][index]):
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="invalid_direction",
                        message="probabilistic direction head returned an invalid vector",
                        runtime_seconds=runtime,
                    )
                )
                continue
            direction = arrays["direction"][index].astype(np.float64)
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
            result = _base_result(row, degradation)
            result.update(
                {
                    "status": True,
                    "prediction": float(reading),
                    "progress": float(progress),
                    "pointer_angle": float(pointer_angle),
                    "direction": direction.tolist(),
                    "direction_angle_error_degrees": direction_error,
                    "pivot_heatmap_xy": arrays["pivot"][index].tolist(),
                    "pivot_input_xy": (arrays["pivot"][index] * stride).tolist(),
                    "pivot_peak": float(arrays["peak"][index]),
                    "pivot_spatial_entropy": float(arrays["pivot_entropy"][index]),
                    "pivot_top2_margin": float(arrays["pivot_margin"][index]),
                    "direction_raw_norm": float(arrays["raw_norm"][index]),
                    "angle_std_degrees": float(arrays["angle_std"][index]),
                    "angle_log_variance": float(arrays["log_variance"][index]),
                    "angle_bin_entropy": float(arrays["angle_entropy"][index]),
                    "angle_bin_resultant_length": float(
                        arrays["bin_resultant"][index]
                    ),
                    "direction_decoder": args.direction_decoder,
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

    for row in tqdm(pending, desc=f"probabilistic {args.condition}", dynamic_ncols=True):
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
                    "reference_branch": str(
                        front_end.get("reference_branch") or "unknown"
                    ),
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
        raise RuntimeError("probabilistic output is incomplete or contains duplicate IDs")
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "condition": args.condition,
        "direction_decoder": args.direction_decoder,
        "signature": signature,
        "metrics": summarize_scalar_predictions(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "direction_component": _component_summary(output_rows),
        "uncertainty_component": _uncertainty_summary(output_rows),
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
