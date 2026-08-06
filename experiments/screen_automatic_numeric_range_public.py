"""Deterministic SyncG/train screen for automatic numeric range inference.

This is a development screen, not a confirmatory result.  It reads only the
frozen public SyncG/train validation partition.  Public dial boxes prepare the
canonical-ROI task input outside the model boundary; the primary estimator
then receives only the whole ROI plus geometry predicted by frozen PEPD and a
frozen ScaleMark head.  Ground-truth scale values are joined only after each
prediction has been finalized and are written to a separate score file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range import (
    AutomaticNumericRangePipeline,
    FrozenRapidOCRBackend,
    GeometryHint,
    GeometryProviderResult,
    NumericRangePrediction,
    PROTOCOL as RANGE_PROTOCOL,
    direct_endpoint_baseline,
    image_sha256,
    sha256_file,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import decode_probabilistic_pivot_direction
import experiments.train_cagh_scalemark_reference_probe_v5 as public_v5


PROTOCOL: Final[str] = "automatic_numeric_range_syncg_train_screen_v1"
DEFAULT_SCALEMARK_CHECKPOINT = Path(
    r"C:\pointer_read\cagh_scalemark_reference_public_v5\runs"
    r"\seed_20261206\checkpoint.pt"
)
DEFAULT_OUTPUT = Path(
    r"C:\pointer_read\automatic_numeric_range_public_screen_20260806_v1"
)
PUBLIC_IMAGE_ROOT = (PROJECT_ROOT / "datasets/SyncG/syncG/images/train").resolve()
PUBLIC_ANNOTATION_ROOT = (
    PROJECT_ROOT / "datasets/SyncG/syncG/annotations/train"
).resolve()
SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        + b"\n",
    )


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows),
    )


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes allowed root {root}: {resolved}") from error
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--subset-seed", type=int, default=20260806)
    parser.add_argument("--ocr-size", type=int, choices=(512, 768), default=768)
    parser.add_argument("--ocr-runtime", type=Path, default=Path(r"C:\pointer_read\rapidocr_runtime"))
    parser.add_argument("--ocr-cpu-threads", type=int, default=4)
    parser.add_argument("--torch-cpu-threads", type=int, default=2)
    parser.add_argument(
        "--scalemark-checkpoint", type=Path, default=DEFAULT_SCALEMARK_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


class FrozenPublicScaleMarkGeometryProvider:
    """CPU inference for frozen PEPD + frozen public V5 ScaleMark head."""

    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cpu")
        checkpoint_file = Path(checkpoint_path).resolve(strict=True)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        _require(isinstance(checkpoint, Mapping), "ScaleMark checkpoint is not a mapping")
        _require(checkpoint.get("protocol") == public_v5.PROTOCOL, "ScaleMark protocol drift")
        _require(checkpoint.get("status") == "complete", "ScaleMark checkpoint is incomplete")
        _require(isinstance(checkpoint.get("head_state"), Mapping), "head_state is absent")

        self.backbone, backbone_identity = public_v5.load_pepd(
            public_v5.BACKBONE_CHECKPOINT, self.device
        )
        self.head = public_v5.build_head().to(self.device)
        self.head.load_state_dict(checkpoint["head_state"], strict=True)
        self.backbone.eval()
        self.head.eval()
        self.identity = {
            "provider": "frozen_pepd_plus_public_v5_scalemark",
            "execution_device": "cpu",
            "pepd_checkpoint": {
                "path": str(public_v5.BACKBONE_CHECKPOINT.resolve(strict=True)),
                "sha256": sha256_file(public_v5.BACKBONE_CHECKPOINT),
                "loader_identity": backbone_identity,
            },
            "scalemark_checkpoint": {
                "path": str(checkpoint_file),
                "sha256": sha256_file(checkpoint_file),
                "protocol": str(checkpoint["protocol"]),
            },
        }

    def predict(self, canonical_roi_bgr: np.ndarray) -> GeometryProviderResult:
        interpolation = (
            cv2.INTER_AREA if max(canonical_roi_bgr.shape[:2]) > 256 else cv2.INTER_LINEAR
        )
        model_image = cv2.resize(
            canonical_roi_bgr, (256, 256), interpolation=interpolation
        )
        tensor = normalized_rgb_tensor(model_image)[None].to(self.device)
        with torch.inference_mode():
            features = self.backbone.forward_multiscale_features(tensor)
            pooled = self.backbone.direction_features(features.c5)
            pivot_logits = self.backbone.pivot_head(features.c5)
            direction = decode_probabilistic_pivot_direction(
                pivot_logits,
                self.backbone.vector_head(pooled),
                self.backbone.angle_head(pooled),
                self.backbone.log_variance_head(pooled),
            )
            endpoint = self.head(features.c2, features.c5)

        heat_height, heat_width = pivot_logits.shape[-2:]
        pivot = direction.pivot_xy[0].detach().cpu().numpy().astype(np.float64)
        pivot_normalized = (
            float(pivot[0] / max(heat_width - 1, 1)),
            float(pivot[1] / max(heat_height - 1, 1)),
        )
        start = tuple(float(value) for value in endpoint.start_xy[0].cpu().tolist())
        end = tuple(float(value) for value in endpoint.end_xy[0].cpu().tolist())
        entropy = [float(value) for value in endpoint.endpoint_entropy[0].cpu().tolist()]
        peaks = [float(value) for value in endpoint.endpoint_peak[0].cpu().tolist()]
        confidence = float(np.clip(1.0 - np.mean(entropy), 0.0, 1.0))
        hint = GeometryHint(
            pivot_xy=pivot_normalized,
            start_xy=start,
            end_xy=end,
            confidence=confidence,
            source="frozen_pepd_plus_public_v5_scalemark",
        ).validate()
        return GeometryProviderResult(
            hint=hint,
            telemetry={
                "pivot_peak": float(direction.pivot_peak[0].cpu()),
                "pivot_valid": bool(direction.valid[0].cpu()),
                "endpoint_peak": peaks,
                "endpoint_entropy": entropy,
                "endpoint_separation": float(endpoint.endpoint_separation[0].cpu()),
                "model_input_shape": list(model_image.shape),
            },
        )


def _load_roster() -> tuple[list[Any], Mapping[str, Any]]:
    with public_v5.PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    fit, validation = public_v5.load_public_roster(protocol)
    _require(len(fit) == 14_400 and len(validation) == 1_600, "public roster drift")
    return [record.sample for record in validation], protocol


def _select_subset(samples: Sequence[Any], *, count: int, seed: int) -> list[Any]:
    _require(1 <= count <= len(samples), "subset count is outside public validation")
    by_group: dict[str, list[Any]] = {}
    for sample in samples:
        by_group.setdefault(str(sample.group_id), []).append(sample)
    representatives: list[tuple[str, Any]] = []
    for group_id, group_samples in sorted(by_group.items()):
        ranked = sorted(
            group_samples,
            key=lambda row: hashlib.sha256(
                f"{seed}:{group_id}:{row.sample_id}".encode("utf-8")
            ).hexdigest(),
        )
        group_key = hashlib.sha256(f"{seed}:group:{group_id}".encode("utf-8")).hexdigest()
        representatives.append((group_key, ranked[0]))
    _require(count <= len(representatives), "count exceeds distinct validation groups")
    return [row for _, row in sorted(representatives, key=lambda item: item[0])[:count]]


def _exact_integer(value: Any, *, name: str) -> int:
    number = float(value)
    integer = int(round(number))
    _require(abs(number - integer) <= 1e-9, f"{name} is not an integer")
    return integer


def _metrics(rows: Sequence[Mapping[str, Any]], *, prefix: str) -> dict[str, Any]:
    total = len(rows)
    if prefix == "decoder":
        start_key, end_key = "pred_start", "pred_end"
        covered = [row for row in rows if row["status"]]
    elif prefix == "syncg_integer_diagnostic":
        start_key, end_key = "syncg_integer_start", "syncg_integer_end"
        covered = [
            row for row in rows
            if row[start_key] is not None and row[end_key] is not None
        ]
    else:
        start_key, end_key = "direct_start", "direct_end"
        covered = [
            row for row in rows
            if row[start_key] is not None and row[end_key] is not None
        ]
    start_correct = sum(
        row[start_key] is not None and row[start_key] == row["truth_start"] for row in rows
    )
    end_correct = sum(
        row[end_key] is not None and row[end_key] == row["truth_end"] for row in rows
    )
    pair_correct = sum(
        row[start_key] is not None
        and row[end_key] is not None
        and row[start_key] == row["truth_start"]
        and row[end_key] == row["truth_end"]
        for row in rows
    )
    start_correct_conditional = sum(
        row[start_key] == row["truth_start"] for row in covered
    )
    end_correct_conditional = sum(
        row[end_key] == row["truth_end"] for row in covered
    )
    pair_correct_conditional = sum(
        row[start_key] == row["truth_start"]
        and row[end_key] == row["truth_end"]
        for row in covered
    )
    start_errors = [abs(row[start_key] - row["truth_start"]) for row in covered]
    end_errors = [abs(row[end_key] - row["truth_end"]) for row in covered]
    return {
        "samples": total,
        "covered": len(covered),
        "coverage": len(covered) / total,
        "start_exact_full_denominator": start_correct / total,
        "end_exact_full_denominator": end_correct / total,
        "pair_exact_full_denominator": pair_correct / total,
        "start_exact_conditional": (
            start_correct_conditional / len(covered) if covered else None
        ),
        "end_exact_conditional": (
            end_correct_conditional / len(covered) if covered else None
        ),
        "pair_exact_conditional": (
            pair_correct_conditional / len(covered) if covered else None
        ),
        "start_mae_conditional": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_conditional": float(np.mean(end_errors)) if end_errors else None,
    }


def _continuous_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    covered = [
        row for row in rows
        if row["status"] and row["pred_start"] is not None and row["pred_end"] is not None
    ]
    start_errors = [abs(float(row["pred_start"]) - row["truth_start"]) for row in covered]
    end_errors = [abs(float(row["pred_end"]) - row["truth_end"]) for row in covered]
    start_rounded = sum(
        int(round(float(row["pred_start"]))) == row["truth_start"] for row in covered
    )
    end_rounded = sum(
        int(round(float(row["pred_end"]))) == row["truth_end"] for row in covered
    )
    pair_rounded = sum(
        int(round(float(row["pred_start"]))) == row["truth_start"]
        and int(round(float(row["pred_end"]))) == row["truth_end"]
        for row in covered
    )
    return {
        "samples": total,
        "covered": len(covered),
        "coverage": len(covered) / total,
        "start_mae_conditional": float(np.mean(start_errors)) if start_errors else None,
        "end_mae_conditional": float(np.mean(end_errors)) if end_errors else None,
        "start_rounded_exact_full_denominator": start_rounded / total,
        "end_rounded_exact_full_denominator": end_rounded / total,
        "pair_rounded_exact_full_denominator": pair_rounded / total,
        "note": "rounding is a SyncG integer-label evaluation only; deployment output remains real-valued",
    }


def _failure_prediction(reason: str, geometry: Mapping[str, Any]) -> NumericRangePrediction:
    return NumericRangePrediction(
        protocol=RANGE_PROTOCOL,
        status=False,
        prediction_space="real_numeric_scale_start_end",
        pred_start=None,
        pred_end=None,
        confidence=0.0,
        failure_reason=reason,
        telemetry={"geometry_failure": dict(geometry)},
    )


def main() -> None:
    args = parse_args()
    _require(1 <= args.ocr_cpu_threads <= 16, "invalid OCR CPU thread count")
    _require(1 <= args.torch_cpu_threads <= 8, "invalid torch CPU thread count")
    output = args.output_dir.resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"output must stay under {SAFE_OUTPUT_ROOT}") from error
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(args.torch_cpu_threads)
    torch.set_num_interop_threads(1)
    roster, split_protocol = _load_roster()
    subset = _select_subset(roster, count=args.count, seed=args.subset_seed)
    subset_identity = [
        {"sample_id": sample.sample_id, "group_id": sample.group_id} for sample in subset
    ]

    run_started = time.perf_counter()
    geometry_provider = FrozenPublicScaleMarkGeometryProvider(args.scalemark_checkpoint)
    ocr_backend = FrozenRapidOCRBackend(
        args.ocr_runtime, cpu_threads=args.ocr_cpu_threads
    )
    pipeline = AutomaticNumericRangePipeline(
        geometry_provider, ocr_backend, input_size=args.ocr_size
    )
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []

    for index, sample in enumerate(subset, 1):
        image_path = _require_under(
            Path(sample.image_path), PUBLIC_IMAGE_ROOT, label=f"{sample.sample_id}.image"
        )
        annotation_path = _require_under(
            Path(str(sample.metadata.get("annotation_path") or "")),
            PUBLIC_ANNOTATION_ROOT,
            label=f"{sample.sample_id}.annotation",
        )
        # The public bbox only defines the canonical-ROI benchmark input.  It
        # is not passed to the geometry provider, OCR backend or decoder.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        _require(image is not None, f"cannot read public image {image_path}")
        roi, _, _, bounds = public_v5.canonical_tight_roi(
            image, sample.dial_bbox, output_size=args.ocr_size
        )
        sample_started = time.perf_counter()
        try:
            prediction = pipeline.predict(roi)
            geometry_telemetry = prediction.telemetry["primary_adapter"][
                "geometry_telemetry"
            ]
        except (ValueError, RuntimeError) as error:
            geometry_telemetry = {"exception_type": type(error).__name__, "message": str(error)}
            prediction = _failure_prediction("automatic_geometry_failure", geometry_telemetry)
        sample_seconds = time.perf_counter() - sample_started
        direct_start, direct_end = direct_endpoint_baseline(prediction)
        fit = prediction.telemetry.get("fit") or {}
        integer_diagnostic = (
            fit.get("syncg_integer_progression_diagnostic")
            if isinstance(fit, Mapping)
            else None
        )
        syncg_integer_start = (
            integer_diagnostic.get("pred_start")
            if isinstance(integer_diagnostic, Mapping)
            else None
        )
        syncg_integer_end = (
            integer_diagnostic.get("pred_end")
            if isinstance(integer_diagnostic, Mapping)
            else None
        )
        prediction_row = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "sample_id": str(sample.sample_id),
            "group_id": str(sample.group_id),
            "canonical_roi_sha256": image_sha256(roi),
            "status": bool(prediction.status),
            "pred_start": prediction.pred_start,
            "pred_end": prediction.pred_end,
            "confidence": float(prediction.confidence),
            "failure_reason": prediction.failure_reason,
            "direct_endpoint_diagnostic": {
                "pred_start": direct_start,
                "pred_end": direct_end,
            },
            "syncg_integer_progression_diagnostic": integer_diagnostic,
            "automatic_geometry": dict(geometry_telemetry),
            "range_prediction": prediction.as_dict(),
            "sample_seconds": sample_seconds,
        }
        truth_start = _exact_integer(sample.scale_start, name="scale_start")
        truth_end = _exact_integer(sample.scale_end, name="scale_end")
        score_row = {
            "sample_id": str(sample.sample_id),
            "group_id": str(sample.group_id),
            "status": bool(prediction.status),
            "pred_start": prediction.pred_start,
            "pred_end": prediction.pred_end,
            "direct_start": direct_start,
            "direct_end": direct_end,
            "syncg_integer_start": syncg_integer_start,
            "syncg_integer_end": syncg_integer_end,
            "truth_start": truth_start,
            "truth_end": truth_end,
        }
        predictions.append(prediction_row)
        scores.append(score_row)
        print(
            f"[{index:02d}/{len(subset):02d}] {sample.sample_id} "
            f"status={prediction.status} pred=({prediction.pred_start},{prediction.pred_end}) "
            f"truth=({truth_start},{truth_end}) seconds={sample_seconds:.2f}",
            flush=True,
        )

    elapsed = time.perf_counter() - run_started
    predictions_path = output / "predictions.label_free.jsonl"
    scores_path = output / "scores.public_train_only.jsonl"
    _atomic_jsonl(predictions_path, predictions)
    _atomic_jsonl(scores_path, scores)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "screen_complete",
        "claim_boundary": "deterministic public SyncG/train development screen only",
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "partition": "frozen group-disjoint public validation",
            "samples": len(subset),
            "groups": len({sample.group_id for sample in subset}),
            "field_samples_read": 0,
            "test_samples_read": 0,
            "sealed_samples_read": 0,
        },
        "metrics": {
            "geometry_guided_decimal_capable_ap_ransac": _continuous_metrics(scores),
            "syncg_integer_progression_diagnostic": _metrics(
                scores, prefix="syncg_integer_diagnostic"
            ),
            "nearest_endpoint_ocr_diagnostic": _metrics(scores, prefix="direct"),
        },
        "timing": {
            "total_seconds_including_initialization": elapsed,
            "mean_sample_seconds": float(np.mean([row["sample_seconds"] for row in predictions])),
            "median_sample_seconds": float(
                np.median([row["sample_seconds"] for row in predictions])
            ),
        },
        "subset": {
            "seed": args.subset_seed,
            "selection": "one deterministic representative per group, then hash rank",
            "roster_sha256": canonical_sha256(subset_identity),
            "rows": subset_identity,
            "frozen_validation_ids_sha256": split_protocol["split"][
                "validation_ids_sha256"
            ],
        },
        "inference_contract": {
            "canonical_roi_preparation": "public GT dial bbox outside primary model boundary",
            "primary_input": "same whole canonical meter ROI",
            "ocr_size": args.ocr_size,
            "ocr_second_crop_or_detector": False,
            "geometry": "automatic frozen PEPD + frozen public V5 ScaleMark",
            "ground_truth_bbox_inside_primary_inference": False,
            "ground_truth_text_boxes_inside_primary_inference": False,
            "ground_truth_scale_values_inside_primary_inference": False,
            "minimum_ransac_inliers": 3,
            "deployment_output_supports_signed_decimals": True,
            "integer_progression_snapping_in_deployment_output": False,
            "ocr_inferred_decimal_ap_grid_in_deployment_output": True,
        },
        "bindings": {
            "geometry_provider": geometry_provider.identity,
            "ocr_backend": dict(ocr_backend.identity),
            "range_source": {
                "path": str(Path(__file__).with_name("automatic_numeric_range.py").resolve()),
                "sha256": sha256_file(Path(__file__).with_name("automatic_numeric_range.py")),
            },
            "screen_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
            },
            "crop_contract": public_v5.CROP_CONTRACT,
            "crop_contract_sha256": public_v5.canonical_sha256(public_v5.CROP_CONTRACT),
        },
        "artifacts": {
            "label_free_predictions": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
            },
            "scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
        },
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "seal.json",
        {
            "protocol": PROTOCOL,
            "summary_sha256": sha256_file(output / "summary.json"),
            "predictions_sha256": sha256_file(predictions_path),
            "scores_sha256": sha256_file(scores_path),
            "scope": summary["scope"],
        },
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
