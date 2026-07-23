"""Evaluate one frozen direction model on Pointer-10K's official test split.

This is deliberately a component benchmark.  It uses the official dial box
and pointer tip/tail labels, keeps only the predeclared single-pointer
applicability domain, and reports angular direction metrics.  It does not
derive scalar readings and it never trains or calibrates on Pointer-10K.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.datasets import (
    POINTER10K_SINGLE_POINTER_IDS_SHA256,
    sample_ids_sha256,
)
from experiments.extract_pointer10k_test import (
    POINTER10K_ARCHIVE_SHA256,
    POINTER10K_TEST_ANNOTATION_SHA256,
    POINTER10K_TEST_SINGLE_POINTER_IMAGES,
)
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.pivot_direction_fallback import tensor_from_bbox
from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PINNED_COMMIT,
    build_vdn_model,
    predict_directions,
    sha256_file,
    vdn_tensor_from_bbox,
)


EVALUATION_PROTOCOL = "pointer10k_zero_shot_single_pointer_direction_v1"
POINTER10K_MANIFEST_PROTOCOL = "pointer10k_official_test_single_pointer_v1"
VDN_TRAINING_PROTOCOL = "vdn_architecture_syncg_retraining_v1"
DEFAULT_VDN_SOURCE = PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"
ANGLE_THRESHOLDS = (2.0, 5.0, 10.0, 15.0, 20.0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _summary_path(output: Path) -> Path:
    return output.with_name(output.name + ".summary.json")


def _metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _pointer_points(
    row: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]]:
    metadata = row.get("metadata") or {}
    for item in metadata.get("keypoints") or []:
        if str(item.get("type") or "").strip().lower() != "pointer":
            continue
        tip = item.get("outside_kp")
        tail = item.get("origin_kp")
        if (
            isinstance(tip, Sequence)
            and len(tip) >= 2
            and isinstance(tail, Sequence)
            and len(tail) >= 2
        ):
            return (
                (float(tip[0]), float(tip[1])),
                (float(tail[0]), float(tail[1])),
            )
    raise ValueError(f"{row.get('sample_id')}: missing pointer tip/tail labels")


def _dial_bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = (row.get("metadata") or {}).get("dial_bbox")
    if not (
        isinstance(bbox, Sequence)
        and len(bbox) >= 4
        and all(isinstance(item, (int, float)) for item in bbox[:4])
    ):
        raise ValueError(f"{row.get('sample_id')}: missing xyxy dial_bbox")
    value = tuple(float(item) for item in bbox[:4])
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError(f"{row.get('sample_id')}: invalid xyxy dial_bbox")
    return value


def _target_direction(row: dict[str, Any]) -> np.ndarray:
    tip, tail = _pointer_points(row)
    direction = np.asarray(tip, dtype=np.float32) - np.asarray(
        tail,
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ValueError(f"{row.get('sample_id')}: pointer direction collapsed")
    return direction / norm


def _imread_unicode(path: Path) -> np.ndarray:
    payload = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


class Pointer10KDirectionDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        model_kind: str,
        image_size: int,
        expansion: float,
    ) -> None:
        self.rows = list(rows)
        self.model_kind = model_kind
        self.image_size = int(image_size)
        self.expansion = float(expansion)
        if not self.rows:
            raise ValueError("Pointer-10K evaluation set is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = _imread_unicode(Path(str(row["image_path"])))
        bbox = _dial_bbox(row)
        if self.model_kind == "probabilistic":
            tensor = tensor_from_bbox(
                image,
                bbox,
                image_size=self.image_size,
                expansion=self.expansion,
            )
        elif self.model_kind == "vdn":
            tensor = vdn_tensor_from_bbox(
                image,
                bbox,
                image_size=self.image_size,
                expansion=self.expansion,
            )
        else:
            raise ValueError(f"unsupported model kind: {self.model_kind}")
        return tensor, torch.from_numpy(_target_direction(row)), index


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if iterations <= 0 or values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[sample]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def summarize_direction_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 20260723,
) -> dict[str, Any]:
    """Summarize direction errors with failures kept in every denominator."""

    if not rows:
        raise ValueError("cannot summarize an empty Pointer-10K result")
    successful = [
        row
        for row in rows
        if row.get("status") is True
        and row.get("angle_error_degrees") is not None
        and math.isfinite(float(row["angle_error_degrees"]))
    ]
    successful_errors = np.asarray(
        [float(row["angle_error_degrees"]) for row in successful],
        dtype=np.float64,
    )
    failure_aware_errors = np.asarray(
        [
            float(row["angle_error_degrees"])
            if row.get("status") is True
            and row.get("angle_error_degrees") is not None
            and math.isfinite(float(row["angle_error_degrees"]))
            else 180.0
            for row in rows
        ],
        dtype=np.float64,
    )
    metrics: dict[str, Any] = {
        "samples": len(rows),
        "successful_directions": len(successful),
        "coverage": len(successful) / len(rows),
        "failure_penalty_degrees": 180.0,
        "mean_angle_error_degrees": float(np.mean(failure_aware_errors)),
        "mean_angle_error_bootstrap_95ci": _bootstrap_mean_ci(
            failure_aware_errors,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "mean_cosine_similarity": float(
            np.mean(np.cos(np.deg2rad(failure_aware_errors)))
        ),
    }
    if successful_errors.size:
        metrics.update(
            {
                "successful_mean_angle_error_degrees": float(
                    np.mean(successful_errors)
                ),
                "successful_median_angle_error_degrees": float(
                    np.median(successful_errors)
                ),
                "successful_p90_angle_error_degrees": float(
                    np.quantile(successful_errors, 0.90)
                ),
            }
        )
    else:
        metrics.update(
            {
                "successful_mean_angle_error_degrees": None,
                "successful_median_angle_error_degrees": None,
                "successful_p90_angle_error_degrees": None,
            }
        )
    for threshold in ANGLE_THRESHOLDS:
        key = f"acc_{int(threshold)}deg"
        metrics[key] = float(
            np.mean(
                [
                    row.get("status") is True
                    and row.get("angle_error_degrees") is not None
                    and float(row["angle_error_degrees"]) <= threshold
                    for row in rows
                ]
            )
        )
    return metrics


def summarize_quality_groups(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    group_names = sorted(
        {
            str(group)
            for row in rows
            for group in (row.get("quality_groups") or [])
        }
    )
    result = {
        "all": summarize_direction_rows(
            rows,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
    }
    for offset, group in enumerate(group_names, 1):
        selected = [row for row in rows if group in (row.get("quality_groups") or [])]
        if selected:
            result[group] = summarize_direction_rows(
                selected,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed + offset,
            )
    return result


def _load_manifest(
    manifest: Path,
    *,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_path = _protocol_path(manifest)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != POINTER10K_MANIFEST_PROTOCOL:
        raise ValueError(
            f"manifest protocol is {protocol.get('protocol')!r}, "
            f"expected {POINTER10K_MANIFEST_PROTOCOL!r}"
        )
    if protocol.get("release_identity_verified") is not True:
        raise ValueError("Pointer-10K release identity is not verified")
    if protocol.get("selection_uses_predictions") is not False:
        raise ValueError("Pointer-10K subset selection is not prediction-independent")
    if int(protocol.get("pointer10k_training_images_used", -1)) != 0:
        raise ValueError("Pointer-10K training images must not be used")
    if protocol.get("diagnostic_limit") is not None:
        raise ValueError("formal evaluation requires the complete Pointer-10K manifest")
    if protocol.get("source_archive_sha256") != POINTER10K_ARCHIVE_SHA256:
        raise ValueError("Pointer-10K source archive identity is not pinned")
    if (
        protocol.get("source_annotation_sha256")
        != POINTER10K_TEST_ANNOTATION_SHA256
    ):
        raise ValueError("Pointer-10K source annotation identity is not pinned")
    extraction_hash = protocol.get("extraction_audit_sha256")
    if not isinstance(extraction_hash, str) or len(extraction_hash) != 64:
        raise ValueError("Pointer-10K extraction audit is not bound to the manifest")

    rows = _read_jsonl(manifest)
    if len(rows) != int(protocol.get("emitted_rows", -1)):
        raise ValueError("Pointer-10K manifest row count disagrees with protocol")
    if not rows:
        raise ValueError("Pointer-10K manifest selection is empty")
    if len(rows) != POINTER10K_TEST_SINGLE_POINTER_IMAGES:
        raise ValueError(
            f"formal Pointer-10K manifest has {len(rows)} rows, "
            f"expected {POINTER10K_TEST_SINGLE_POINTER_IMAGES}"
        )
    sample_hash = sample_ids_sha256(str(int(row.get("sample_id"))) for row in rows)
    if sample_hash != POINTER10K_SINGLE_POINTER_IDS_SHA256:
        raise ValueError("Pointer-10K manifest sample identity is not pinned")
    if limit is not None:
        rows = rows[: max(0, limit)]
    if not rows:
        raise ValueError("Pointer-10K diagnostic selection is empty")
    for row in rows:
        if (
            row.get("dataset") != "Pointer-10K"
            or row.get("split") != "official_test_single_pointer"
        ):
            raise ValueError("manifest contains a non-Pointer-10K test row")
        if int((row.get("metadata") or {}).get("pointer_count", -1)) != 1:
            raise ValueError("manifest contains a non-single-pointer row")
    return rows, protocol


def _load_model(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    signature = checkpoint.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("checkpoint lacks a signed training signature")
    if args.model_kind == "probabilistic":
        if signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
            raise ValueError(
                "probabilistic checkpoint was not produced by the frozen "
                "SyncG training protocol"
            )
        image_size = int(signature.get("image_size", 256))
        angle_bins = int(signature.get("angle_bins", 72))
        model = build_probabilistic_pivot_direction_model(
            angle_bins=angle_bins,
            imagenet_pretrained=False,
        )
    else:
        if signature.get("protocol") != VDN_TRAINING_PROTOCOL:
            raise ValueError(
                "VDN checkpoint was not produced by the frozen SyncG retraining protocol"
            )
        if signature.get("vdn_source_commit") != VDN_PINNED_COMMIT:
            raise ValueError("VDN checkpoint refers to a different source commit")
        image_size = int(signature.get("image_size", 384))
        model = build_vdn_model(
            args.vdn_source,
            image_size=image_size,
            imagenet_pretrained=False,
        )
        angle_bins = None
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint/model mismatch: {incompatible}")
    return model, {
        "training_signature": signature,
        "image_size": image_size,
        "angle_bins": angle_bins,
        "expansion": float(signature.get("expansion", 1.25)),
    }


def _source_hashes() -> dict[str, str]:
    return {
        "evaluation": sha256_file(Path(__file__).resolve()),
        "dataset_adapter": sha256_file(PROJECT_DIR / "experiments" / "datasets.py"),
        "probabilistic_model": sha256_file(
            PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
        ),
        "vdn_adapter": sha256_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("probabilistic", "vdn"),
        required=True,
    )
    parser.add_argument("--model-label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vdn-source", type=Path, default=DEFAULT_VDN_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.vdn_source = args.vdn_source.resolve()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")
    for path in (args.manifest, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, _summary_path(args.output), _metadata_path(args.output)):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    rows, manifest_protocol = _load_manifest(args.manifest, limit=args.limit)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    model, model_info = _load_model(args, checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model.to(device).eval()
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(0)

    dataset = Pointer10KDirectionDataset(
        rows,
        model_kind=args.model_kind,
        image_size=int(model_info["image_size"]),
        expansion=float(model_info["expansion"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    results_by_index: dict[int, dict[str, Any]] = {}
    inference_seconds = 0.0
    with torch.inference_mode():
        for images, targets, indices in tqdm(
            loader,
            desc=f"Pointer-10K {args.model_kind}",
            unit="batch",
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(images)
            if args.model_kind == "probabilistic":
                prediction = decode_probabilistic_pivot_direction(*outputs)
                directions = prediction.direction
                valid = prediction.valid
                confidence = prediction.pivot_peak * prediction.bin_resultant_length
                uncertainty = {
                    "angle_std_degrees": prediction.angle_std_degrees,
                    "angle_entropy": prediction.angle_entropy,
                    "bin_resultant_length": prediction.bin_resultant_length,
                    "pivot_peak": prediction.pivot_peak,
                }
            else:
                directions, confidence, valid = predict_directions(*outputs)
                uncertainty = {}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started

            target_norm = targets / torch.clamp(
                torch.linalg.vector_norm(targets, dim=1, keepdim=True),
                min=1e-8,
            )
            cosine = torch.sum(directions.float() * target_norm.float(), dim=1)
            errors = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
            arrays = {
                "directions": directions.detach().cpu().float().numpy(),
                "valid": valid.detach().cpu().numpy(),
                "confidence": confidence.detach().cpu().float().numpy(),
                "errors": errors.detach().cpu().float().numpy(),
                **{
                    name: value.detach().cpu().float().numpy()
                    for name, value in uncertainty.items()
                },
            }
            for batch_index, raw_index in enumerate(indices.tolist()):
                source = rows[int(raw_index)]
                is_valid = bool(arrays["valid"][batch_index]) and math.isfinite(
                    float(arrays["errors"][batch_index])
                )
                result: dict[str, Any] = {
                    "dataset": source["dataset"],
                    "split": source["split"],
                    "sample_id": source["sample_id"],
                    "group_id": source["group_id"],
                    "model_kind": args.model_kind,
                    "model_label": args.model_label or args.model_kind,
                    "status": is_valid,
                    "direction": (
                        arrays["directions"][batch_index].tolist()
                        if is_valid
                        else None
                    ),
                    "target_direction": _target_direction(source).tolist(),
                    "angle_error_degrees": (
                        float(arrays["errors"][batch_index]) if is_valid else None
                    ),
                    "confidence": (
                        float(arrays["confidence"][batch_index])
                        if math.isfinite(float(arrays["confidence"][batch_index]))
                        else None
                    ),
                    "quality_groups": list(
                        (source.get("metadata") or {}).get("quality_groups") or []
                    ),
                    "quality": dict(
                        (source.get("metadata") or {}).get("quality") or {}
                    ),
                }
                for name in uncertainty:
                    value = float(arrays[name][batch_index])
                    result[name] = value if math.isfinite(value) else None
                results_by_index[int(raw_index)] = result

    if len(results_by_index) != len(rows):
        raise RuntimeError(
            f"evaluation produced {len(results_by_index)} rows for {len(rows)} samples"
        )
    output_rows = [results_by_index[index] for index in range(len(rows))]
    model_label = args.model_label or args.model_kind
    signature = {
        "protocol": EVALUATION_PROTOCOL,
        "model_kind": args.model_kind,
        "model_label": model_label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_signature": model_info["training_signature"],
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_protocol_sha256": sha256_file(_protocol_path(args.manifest)),
        "pointer10k_source_annotation_sha256": manifest_protocol.get(
            "source_annotation_sha256"
        ),
        "pointer10k_training_images_used": 0,
        "zero_shot_external_test": True,
        "selection_uses_predictions": False,
        "single_pointer_applicability_domain": True,
        "ground_truth_dial_box_used": True,
        "crop_expansion": float(model_info["expansion"]),
        "image_size": int(model_info["image_size"]),
        "mixed_precision": use_amp,
        "deterministic_cudnn": device.type == "cuda",
        "tf32_enabled": False,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "diagnostic_limit": args.limit,
        "source_sha256": _source_hashes(),
    }
    _write_jsonl(args.output, output_rows)
    signature["result_sha256"] = sha256_file(args.output)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    summary = {
        "protocol": EVALUATION_PROTOCOL,
        "model_kind": args.model_kind,
        "model_label": model_label,
        "formal_test_run": args.limit is None,
        "samples": len(output_rows),
        "inference_seconds": inference_seconds,
        "images_per_second": (
            len(output_rows) / inference_seconds if inference_seconds > 0.0 else None
        ),
        "metrics": summarize_direction_rows(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "quality_subgroups": summarize_quality_groups(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "signature": signature,
        "interpretation": (
            "zero-shot single-pointer direction result; not official full-test "
            "multi-pointer OKS/VDS and not a scalar-reading score"
        ),
    }
    _metadata_path(args.output).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _summary_path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
