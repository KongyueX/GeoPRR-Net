"""Evaluate official-200 VDN on the frozen SyncG/train grouped OOF cohort.

Every sample is routed to the checkpoint whose seed-specific grouped split
held out its complete physical meter group.  The sample set, detector box,
reference angles, and scale conversion are bound to the existing 4,380-row
Original-Transformer/FADR train-only OOF artifact.  No public, test, field,
sealed, confirmatory, or holdout path is accepted by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from experiments.vdn_baseline import (
    build_vdn_model,
    grouped_train_val_split,
    image_angle_from_direction,
    load_syncg_manifest,
    predict_directions,
    reading_from_pointer_angle,
    sample_ids_hash,
    sha256_file,
    vdn_tensor_from_bbox,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_VALIDATION_FRACTION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VDN_SEEDS = (20260720, 20260721, 20260722)
FADR_SEEDS = (20260722, 20260723, 20260724)
EXPECTED_SAMPLES = 4380
EXPECTED_GROUPS = 197

MANIFEST = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
ORIGINAL_OOF = PROJECT_ROOT / "artifacts/runs/quality_router_syncg/oof_clean.jsonl"
ORIGINAL_OOF_SUMMARY = ORIGINAL_OOF.with_suffix(".summary.json")
ORIGINAL_OOF_METADATA = ORIGINAL_OOF.with_name(ORIGINAL_OOF.name + ".meta.json")
FADR_ROOT = PROJECT_ROOT / "artifacts/runs/fadr_multiseed_v2_joint_authoritative_v2"
FADR_COHORT = FADR_ROOT / "cohort.json"
VDN_RUN_ROOT = PROJECT_ROOT / "artifacts/runs/vdn_syncg_official200"
VDN_COHORT = PROJECT_ROOT / "artifacts/protocols/vdn_official200_three_seed_cohort_v1.json"
VDN_PREFLIGHT = PROJECT_ROOT / "artifacts/protocols/vdn_official200_preflight_v1.json"
VDN_SOURCE = PROJECT_ROOT / "artifacts/vendor/VectorDetectionNetwork"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/runs/vdn_official200_train_oof_v1"

_FORBIDDEN_EXACT_PARTS = frozenset(
    {"public", "test", "tests", "field", "sealed", "confirmatory", "holdout"}
)
_FORBIDDEN_PREFIXES = tuple(
    f"{token}{suffix}"
    for token in ("public", "test", "field", "sealed", "confirmatory", "holdout")
    for suffix in ("_", "-")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_train_only_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path is outside the project root: {resolved}") from exc
    for raw_part in relative.parts:
        part = raw_part.casefold()
        if part in _FORBIDDEN_EXACT_PARTS or part.startswith(_FORBIDDEN_PREFIXES):
            raise ValueError(f"forbidden non-train path: {resolved}")
    return resolved


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalized_error(
    ground_truth: float,
    scale_start: float,
    scale_end: float,
    prediction: Any,
) -> float:
    prediction_value = _finite(prediction)
    if prediction_value is None:
        return 1.0
    span = abs(float(scale_end) - float(scale_start))
    if not span > 0.0:
        raise ValueError("scale span must be positive")
    return abs(prediction_value - float(ground_truth)) / span


def _metrics(errors: np.ndarray, successful: np.ndarray) -> dict[str, Any]:
    _require(errors.ndim == successful.ndim == 1, "metrics require vectors")
    _require(len(errors) == len(successful) > 0, "metric vectors are empty/mismatched")
    return {
        "samples": int(len(errors)),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "nmae_failure_penalty": 1.0,
        "acc_1pct": float(np.mean(successful & (errors <= 0.01))),
        "acc_2pct": float(np.mean(successful & (errors <= 0.02))),
        "acc_5pct": float(np.mean(successful & (errors <= 0.05))),
    }


def _paired_group_bootstrap(
    first_errors: np.ndarray,
    second_errors: np.ndarray,
    groups: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    _require(iterations > 0, "bootstrap iterations must be positive")
    _require(
        len(first_errors) == len(second_errors) == len(groups),
        "paired bootstrap vectors differ in length",
    )
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    first_values = np.empty(iterations, dtype=np.float64)
    second_values = np.empty(iterations, dtype=np.float64)
    deltas = np.empty(iterations, dtype=np.float64)
    reductions = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in selected])
        first = float(np.mean(first_errors[sample_indices]))
        second = float(np.mean(second_errors[sample_indices]))
        first_values[iteration] = first
        second_values[iteration] = second
        deltas[iteration] = first - second
        reductions[iteration] = 1.0 - first / second

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    first_point = float(np.mean(first_errors))
    second_point = float(np.mean(second_errors))
    return {
        "unit": "physical_group",
        "physical_groups": int(len(unique)),
        "iterations": int(iterations),
        "seed": int(seed),
        "confidence_level": 0.95,
        "first_method": first_name,
        "second_method": second_name,
        "first_nmae": first_point,
        "first_nmae_95ci": interval(first_values),
        "second_nmae": second_point,
        "second_nmae_95ci": interval(second_values),
        "delta_nmae_first_minus_second": first_point - second_point,
        "delta_nmae_95ci": interval(deltas),
        "relative_nmae_reduction_first_vs_second": 1.0 - first_point / second_point,
        "relative_nmae_reduction_95ci": interval(reductions),
        "first_better_rate": float(np.mean(first_errors < second_errors)),
        "ties_rate": float(np.mean(first_errors == second_errors)),
    }


def _direction_target(sample: Any) -> np.ndarray | None:
    direction = np.asarray(sample.pointer_tip, dtype=np.float64) - np.asarray(
        sample.pointer_tail, dtype=np.float64
    )
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-8 else None


def _direction_error(predicted: np.ndarray, target: np.ndarray | None) -> float | None:
    if target is None:
        return None
    cosine = float(np.clip(np.dot(predicted, target), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _validate_inputs() -> dict[str, Any]:
    read_paths = [
        MANIFEST,
        MANIFEST.with_name(MANIFEST.name + ".protocol.json"),
        ORIGINAL_OOF,
        ORIGINAL_OOF_SUMMARY,
        ORIGINAL_OOF_METADATA,
        FADR_COHORT,
        VDN_COHORT,
        VDN_PREFLIGHT,
        *[
            FADR_ROOT / f"seed_{seed}/joint/joint_oof_predictions.jsonl"
            for seed in FADR_SEEDS
        ],
        *[
            VDN_RUN_ROOT / f"seed_{seed}/{name}"
            for seed in VDN_SEEDS
            for name in ("best.pt", "summary.json", "verification_v1.json")
        ],
    ]
    for path in read_paths:
        _assert_train_only_path(path, must_exist=True)

    original_summary = _load_json(ORIGINAL_OOF_SUMMARY)
    original_metadata = _load_json(ORIGINAL_OOF_METADATA)
    fadr_cohort = _load_json(FADR_COHORT)
    vdn_cohort = _load_json(VDN_COHORT)
    vdn_preflight = _load_json(VDN_PREFLIGHT)

    _require(original_summary.get("status") == "complete", "Original OOF is incomplete")
    _require(
        original_summary.get("protocol") == "syncg_quality_router_cross_model_oof_v1",
        "unexpected Original OOF protocol",
    )
    _require(original_summary.get("samples") == EXPECTED_SAMPLES, "Original OOF row drift")
    _require(original_summary.get("groups") == EXPECTED_GROUPS, "Original OOF group drift")
    _require(original_summary.get("group_leakage_count") == 0, "Original OOF leaks groups")
    _require(original_summary.get("test_samples_used") == 0, "Original OOF used test data")
    _require(original_summary.get("output_sha256") == sha256_file(ORIGINAL_OOF), "Original OOF hash drift")
    original_signature = original_summary.get("signature") or {}
    _require(original_signature.get("split") == "SyncG/train only", "Original OOF is not train-only")
    _require(original_signature.get("test_sets_used") == [], "Original OOF names a test set")
    _require(
        (original_metadata.get("manifest_protocol") or {}).get("split") == "train",
        "Original OOF metadata is not train-only",
    )

    _require(fadr_cohort.get("status") == "verified", "FADR cohort is not verified")
    _require(fadr_cohort.get("samples") == EXPECTED_SAMPLES, "FADR cohort row drift")
    _require(fadr_cohort.get("groups") == EXPECTED_GROUPS, "FADR cohort group drift")
    for key in ("public_samples_used", "test_samples_used", "field_samples_used", "group_leakage_count"):
        _require(fadr_cohort.get(key) == 0, f"FADR cohort failed train-only check: {key}")

    _require(vdn_cohort.get("status") == "passed", "VDN official200 cohort did not pass")
    _require(vdn_cohort.get("verified") is True, "VDN official200 cohort is unverified")
    _require(vdn_cohort.get("three_seed_cohort_complete") is True, "VDN cohort is incomplete")
    _require(tuple(vdn_cohort.get("formal_seeds") or ()) == VDN_SEEDS, "VDN seed cohort drift")
    for key in (
        "public_data_opened_or_read",
        "test_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
    ):
        _require(vdn_cohort.get(key) is False, f"VDN cohort scope failed: {key}")
    _require(vdn_preflight.get("status") == "passed", "VDN official200 preflight failed")
    _require(
        (vdn_preflight.get("manifest") or {}).get("sha256") == sha256_file(MANIFEST),
        "VDN preflight manifest hash drift",
    )

    samples, manifest_protocol = load_syncg_manifest(MANIFEST, expected_split="train")
    sample_by_id = {sample.sample_id: sample for sample in samples}
    _require(len(sample_by_id) == len(samples) == 16000, "SyncG/train sample identity drift")
    original_rows = _load_jsonl(ORIGINAL_OOF)
    original_by_id = {str(row.get("sample_id")): row for row in original_rows}
    _require(len(original_by_id) == len(original_rows) == EXPECTED_SAMPLES, "OOF IDs are duplicated")

    validation_by_seed: dict[int, set[str]] = {}
    assignment: dict[str, int] = {}
    checkpoint_identity: dict[str, Any] = {}
    cohort_runs = {int(item["seed"]): item for item in vdn_cohort.get("runs") or []}
    preflight_runs = {int(item["seed"]): item for item in vdn_preflight.get("runs") or []}
    checkpoints: dict[int, dict[str, Any]] = {}
    for seed in VDN_SEEDS:
        _, validation = grouped_train_val_split(
            samples,
            validation_fraction=OFFICIAL200_VALIDATION_FRACTION,
            seed=seed,
        )
        validation_ids = {sample.sample_id for sample in validation}
        validation_by_seed[seed] = validation_ids
        for sample_id in sorted(validation_ids):
            assignment.setdefault(sample_id, seed)
        cohort_run = cohort_runs.get(seed) or {}
        preflight_run = preflight_runs.get(seed) or {}
        checkpoint_path = VDN_RUN_ROOT / f"seed_{seed}/best.pt"
        summary_path = VDN_RUN_ROOT / f"seed_{seed}/summary.json"
        verification_path = VDN_RUN_ROOT / f"seed_{seed}/verification_v1.json"
        _require(sha256_file(checkpoint_path) == cohort_run.get("best_checkpoint_sha256"), f"seed {seed}: checkpoint hash drift")
        _require(sha256_file(summary_path) == cohort_run.get("summary_sha256"), f"seed {seed}: summary hash drift")
        _require(sha256_file(verification_path) == cohort_run.get("verification_sha256"), f"seed {seed}: verification hash drift")
        verification = _load_json(verification_path)
        _require(verification.get("verified") is True, f"seed {seed}: failed verification")
        _require(verification.get("seed") == seed, f"seed {seed}: verification seed drift")
        _require(verification.get("training_artifacts_verified") is True, f"seed {seed}: training artifacts unverified")
        _require(
            sample_ids_hash(validation) == preflight_run.get("validation_sample_ids_sha256"),
            f"seed {seed}: recomputed validation split drift",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        signature = checkpoint.get("signature") or {}
        _require(signature.get("protocol") == OFFICIAL200_PROTOCOL, f"seed {seed}: checkpoint protocol drift")
        _require(signature.get("seed") == seed, f"seed {seed}: checkpoint seed drift")
        _require(signature.get("manifest_sha256") == sha256_file(MANIFEST), f"seed {seed}: manifest binding drift")
        _require(signature.get("validation_sample_ids_sha256") == sample_ids_hash(validation), f"seed {seed}: checkpoint split binding drift")
        checkpoints[seed] = checkpoint
        checkpoint_identity[str(seed)] = {
            "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "best_epoch": int(cohort_run["best_epoch"]),
            "best_validation_angle_mae_degrees": float(cohort_run["best_validation_angle_mae_degrees"]),
            "validation_samples": len(validation),
            "validation_groups": len({sample.group_id for sample in validation}),
            "validation_sample_ids_sha256": sample_ids_hash(validation),
        }

    ordered_ids = [str(row["sample_id"]) for row in original_rows]
    _require(set(ordered_ids) <= set(sample_by_id), "OOF contains unknown train samples")
    _require(set(ordered_ids) <= set(assignment), "OOF contains samples not held out by any VDN seed")
    groups: set[str] = set()
    for row in original_rows:
        sample_id = str(row["sample_id"])
        sample = sample_by_id[sample_id]
        held_out_seed = int(row.get("held_out_seed", -1))
        _require(held_out_seed == assignment[sample_id], f"{sample_id}: held_out_seed routing drift")
        _require(sample_id in validation_by_seed[held_out_seed], f"{sample_id}: selected VDN saw the sample")
        _require(row.get("dataset") == "SyncG" and row.get("split") == "train", f"{sample_id}: not SyncG/train")
        _require(str(row.get("group_id")) == sample.group_id, f"{sample_id}: physical group drift")
        for name in ("ground_truth", "scale_start", "scale_end"):
            _require(
                math.isclose(float(row[name]), float(getattr(sample, name)), rel_tol=0.0, abs_tol=1e-12),
                f"{sample_id}: {name} drift",
            )
        front_end = row.get("front_end") or {}
        bbox = front_end.get("meter_bbox")
        if bbox is None:
            _require(
                (row.get("raw") or {}).get("status") is False,
                f"{sample_id}: frozen bbox is missing without a front-end failure",
            )
            _require(
                (row.get("raw") or {}).get("error_code") == "meter_not_found",
                f"{sample_id}: unexpected frozen front-end failure",
            )
            groups.add(sample.group_id)
            continue
        _require(isinstance(bbox, list) and len(bbox) == 4, f"{sample_id}: invalid frozen bbox")
        _require(_finite(front_end.get("start_angle")) is not None, f"{sample_id}: missing start angle")
        _require(_finite(front_end.get("range_angle")) is not None, f"{sample_id}: missing range angle")
        _assert_train_only_path(Path(sample.image_path), must_exist=True)
        groups.add(sample.group_id)
    _require(len(groups) == EXPECTED_GROUPS, "OOF physical group count drift")

    verify_vdn_source(VDN_SOURCE)
    return {
        "samples": samples,
        "sample_by_id": sample_by_id,
        "original_rows": original_rows,
        "original_by_id": original_by_id,
        "checkpoints": checkpoints,
        "checkpoint_identity": checkpoint_identity,
        "manifest_protocol": manifest_protocol,
        "sample_ids_ordered_sha256": _sha256_strings(ordered_ids),
        "group_ids_sorted_sha256": _sha256_strings(sorted(groups)),
    }


def _evaluate(prepared: Mapping[str, Any], *, device: torch.device, batch_size: int, amp: bool) -> list[dict[str, Any]]:
    sample_by_id = prepared["sample_by_id"]
    original_rows = prepared["original_rows"]
    checkpoints = prepared["checkpoints"]
    outputs: dict[str, dict[str, Any]] = {}
    for seed in VDN_SEEDS:
        rows = [row for row in original_rows if int(row["held_out_seed"]) == seed]
        checkpoint = checkpoints[seed]
        image_size = int((checkpoint.get("signature") or {}).get("image_size", 384))
        model = build_vdn_model(VDN_SOURCE, image_size=image_size, imagenet_pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        for offset in tqdm(
            range(0, len(rows), batch_size),
            desc=f"VDN official200 OOF seed {seed}",
            dynamic_ncols=True,
        ):
            batch_rows = rows[offset : offset + batch_size]
            tensors: list[torch.Tensor] = []
            contexts: list[tuple[dict[str, Any], Any, float]] = []
            for row in batch_rows:
                started = time.perf_counter()
                sample = sample_by_id[str(row["sample_id"])]
                front_end = row.get("front_end") or {}
                if front_end.get("meter_bbox") is None:
                    outputs[sample.sample_id] = {
                        "sample_id": sample.sample_id,
                        "group_id": sample.group_id,
                        "dataset": "SyncG",
                        "split": "train",
                        "held_out_seed": seed,
                        "ground_truth": sample.ground_truth,
                        "scale_start": sample.scale_start,
                        "scale_end": sample.scale_end,
                        "status": False,
                        "prediction": None,
                        "progress": None,
                        "pointer_angle": None,
                        "direction": None,
                        "direction_angle_error_degrees": None,
                        "heatmap_peak": None,
                        "meter_bbox": None,
                        "meter_confidence": None,
                        "start_angle": None,
                        "range_angle": None,
                        "reference_branch": None,
                        "reference_source": None,
                        "error_code": "meter_not_found",
                        "runtime_seconds": float(time.perf_counter() - started),
                    }
                    continue
                image = cv2.imread(
                    sample.image_path,
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
                )
                if image is None:
                    raise ValueError(f"cannot read train image for {sample.sample_id}")
                bbox = [float(value) for value in row["front_end"]["meter_bbox"]]
                tensors.append(vdn_tensor_from_bbox(image, bbox, image_size=image_size))
                contexts.append((row, sample, started))
            if not tensors:
                continue
            inputs = torch.stack(tensors).to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast(device.type, enabled=amp):
                heatmaps, vector_maps = model(inputs)
            directions, peaks, valid = predict_directions(heatmaps.float(), vector_maps.float())
            directions_np = directions.detach().cpu().numpy()
            peaks_np = peaks.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()
            for index, (row, sample, started) in enumerate(contexts):
                front_end = row["front_end"]
                result: dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "dataset": "SyncG",
                    "split": "train",
                    "held_out_seed": seed,
                    "ground_truth": sample.ground_truth,
                    "scale_start": sample.scale_start,
                    "scale_end": sample.scale_end,
                    "status": False,
                    "prediction": None,
                    "progress": None,
                    "pointer_angle": None,
                    "direction": None,
                    "direction_angle_error_degrees": None,
                    "heatmap_peak": float(peaks_np[index]),
                    "meter_bbox": [float(value) for value in front_end["meter_bbox"]],
                    "meter_confidence": _finite(front_end.get("meter_confidence")),
                    "start_angle": float(front_end["start_angle"]),
                    "range_angle": float(front_end["range_angle"]),
                    "reference_branch": front_end.get("reference_branch"),
                    "reference_source": front_end.get("reference_source"),
                    "error_code": None,
                    "runtime_seconds": float(time.perf_counter() - started),
                }
                if not bool(valid_np[index]):
                    result["error_code"] = "invalid_direction"
                    outputs[sample.sample_id] = result
                    continue
                direction = directions_np[index].astype(np.float64)
                try:
                    pointer_angle = image_angle_from_direction(direction)
                    reading, progress = reading_from_pointer_angle(
                        pointer_angle,
                        start_angle=float(front_end["start_angle"]),
                        range_angle=float(front_end["range_angle"]),
                        scale_start=sample.scale_start,
                        scale_end=sample.scale_end,
                    )
                except ValueError as exc:
                    result["error_code"] = "reading_conversion_failed"
                    result["error_message"] = str(exc)
                    outputs[sample.sample_id] = result
                    continue
                result.update(
                    {
                        "status": True,
                        "prediction": float(reading),
                        "progress": float(progress),
                        "pointer_angle": float(pointer_angle),
                        "direction": direction.tolist(),
                        "direction_angle_error_degrees": _direction_error(
                            direction, _direction_target(sample)
                        ),
                    }
                )
                outputs[sample.sample_id] = result
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ordered = [outputs[str(row["sample_id"])] for row in original_rows]
    _require(len(ordered) == EXPECTED_SAMPLES, "VDN OOF output is incomplete")
    return ordered


def _build_artifacts(
    prepared: Mapping[str, Any],
    vdn_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_rows = prepared["original_rows"]
    original_by_id = prepared["original_by_id"]
    vdn_by_id = {str(row["sample_id"]): row for row in vdn_rows}
    sample_order = [str(row["sample_id"]) for row in original_rows]
    _require(set(vdn_by_id) == set(sample_order), "VDN prediction sample set drift")
    groups = np.asarray([str(original_by_id[sample_id]["group_id"]) for sample_id in sample_order], dtype=object)

    transformer_errors: list[float] = []
    transformer_success: list[bool] = []
    vdn_errors: list[float] = []
    vdn_success: list[bool] = []
    for sample_id in sample_order:
        source = original_by_id[sample_id]
        transformer = (((source.get("raw") or {}).get("methods") or {}).get("transformer") or {}).get("prediction")
        vdn_prediction = vdn_by_id[sample_id].get("prediction")
        transformer_errors.append(_normalized_error(source["ground_truth"], source["scale_start"], source["scale_end"], transformer))
        transformer_success.append(_finite(transformer) is not None)
        vdn_errors.append(_normalized_error(source["ground_truth"], source["scale_start"], source["scale_end"], vdn_prediction))
        vdn_success.append(_finite(vdn_prediction) is not None)
    transformer_error_array = np.asarray(transformer_errors, dtype=np.float64)
    transformer_success_array = np.asarray(transformer_success, dtype=bool)
    vdn_error_array = np.asarray(vdn_errors, dtype=np.float64)
    vdn_success_array = np.asarray(vdn_success, dtype=bool)

    fadr_error_arrays: list[np.ndarray] = []
    fadr_metrics: dict[str, Any] = {}
    fadr_sources: dict[str, Any] = {}
    cohort = _load_json(FADR_COHORT)
    cohort_by_seed = {int(item["seed"]): item for item in cohort.get("per_seed") or []}
    for seed in FADR_SEEDS:
        path = FADR_ROOT / f"seed_{seed}/joint/joint_oof_predictions.jsonl"
        rows = _load_jsonl(path)
        by_id = {str(row["sample_id"]): row for row in rows}
        _require(len(by_id) == len(rows) == EXPECTED_SAMPLES, f"FADR seed {seed}: row drift")
        _require(set(by_id) == set(sample_order), f"FADR seed {seed}: sample set drift")
        errors: list[float] = []
        successful: list[bool] = []
        for sample_id in sample_order:
            source = original_by_id[sample_id]
            prediction = ((by_id[sample_id].get("variants") or {}).get("full") or {}).get("prediction")
            errors.append(_normalized_error(source["ground_truth"], source["scale_start"], source["scale_end"], prediction))
            successful.append(_finite(prediction) is not None)
        error_array = np.asarray(errors, dtype=np.float64)
        metrics = _metrics(error_array, np.asarray(successful, dtype=bool))
        signed = ((cohort_by_seed.get(seed) or {}).get("metrics") or {}).get("full") or {}
        for name in ("coverage", "nmae", "acc_1pct", "acc_2pct", "acc_5pct"):
            _require(math.isclose(float(metrics[name]), float(signed[name]), rel_tol=0.0, abs_tol=1e-12), f"FADR seed {seed}: signed {name} drift")
        fadr_error_arrays.append(error_array)
        fadr_metrics[str(seed)] = metrics
        fadr_sources[str(seed)] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
        }
    fadr_seed_mean_errors = np.mean(np.stack(fadr_error_arrays, axis=0), axis=0)
    fadr_mean_metrics = {
        key: float(np.mean([fadr_metrics[str(seed)][key] for seed in FADR_SEEDS]))
        for key in ("samples", "successful", "coverage", "nmae", "nmae_failure_penalty", "acc_1pct", "acc_2pct", "acc_5pct")
    }
    fadr_mean_metrics["samples"] = EXPECTED_SAMPLES
    fadr_mean_metrics["successful"] = float(np.mean([fadr_metrics[str(seed)]["successful"] for seed in FADR_SEEDS]))

    per_fold: dict[str, Any] = {}
    for seed in VDN_SEEDS:
        indices = np.asarray([int(original_by_id[sample_id]["held_out_seed"]) == seed for sample_id in sample_order], dtype=bool)
        metrics = _metrics(vdn_error_array[indices], vdn_success_array[indices])
        angles = [
            float(vdn_by_id[sample_id]["direction_angle_error_degrees"])
            for sample_id, selected in zip(sample_order, indices, strict=True)
            if selected and _finite(vdn_by_id[sample_id].get("direction_angle_error_degrees")) is not None
        ]
        metrics["direction_angle_mae_degrees_success_only"] = float(np.mean(angles)) if angles else None
        metrics["physical_groups"] = int(len(np.unique(groups[indices])))
        per_fold[str(seed)] = metrics

    vdn_metrics = _metrics(vdn_error_array, vdn_success_array)
    direction_errors = [
        float(row["direction_angle_error_degrees"])
        for row in vdn_rows
        if _finite(row.get("direction_angle_error_degrees")) is not None
    ]
    vdn_metrics["direction_angle_mae_degrees_success_only"] = float(np.mean(direction_errors)) if direction_errors else None
    vdn_metrics["per_held_out_seed"] = per_fold

    comparisons = {
        "vdn_official200_minus_original_transformer": _paired_group_bootstrap(
            vdn_error_array,
            transformer_error_array,
            groups,
            first_name="VDN official200 grouped OOF",
            second_name="Original Transformer grouped OOF",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "fadr_v2_minus_vdn_official200": _paired_group_bootstrap(
            fadr_seed_mean_errors,
            vdn_error_array,
            groups,
            first_name="FADR-v2 three-seed mean",
            second_name="VDN official200 grouped OOF",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + 1,
        ),
        "fadr_v2_minus_original_transformer": _paired_group_bootstrap(
            fadr_seed_mean_errors,
            transformer_error_array,
            groups,
            first_name="FADR-v2 three-seed mean",
            second_name="Original Transformer grouped OOF",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + 2,
        ),
    }

    source_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "protocol": "vdn_official200_same_sample_train_grouped_oof_evaluation_v1",
        "status": "complete",
        "scope": "SyncG/train grouped OOF only",
        "claim_boundary": {
            "complete_manifest_oof": False,
            "description": "union of three seed-specific grouped validation splits; 4380 of 16000 SyncG/train rows and 197 of 725 physical groups",
            "public_samples_used": 0,
            "test_samples_used": 0,
            "field_samples_used": 0,
            "sealed_samples_used": 0,
            "confirmatory_samples_used": 0,
        },
        "cohort": {
            "samples": EXPECTED_SAMPLES,
            "physical_groups": EXPECTED_GROUPS,
            "vdn_seeds": list(VDN_SEEDS),
            "held_out_seed_routing": "lowest formal seed whose exact grouped validation split contains the sample, matching the frozen Original Transformer/FADR OOF assignment",
            "group_leakage_count": 0,
        },
        "metrics": vdn_metrics,
        "predictions": {
            "path": str(predictions_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(predictions_path),
        },
        "bindings": {
            "sample_ids_ordered_sha256": prepared["sample_ids_ordered_sha256"],
            "group_ids_sorted_sha256": prepared["group_ids_sorted_sha256"],
            "original_oof_sha256": sha256_file(ORIGINAL_OOF),
            "manifest_sha256": sha256_file(MANIFEST),
            "manifest_protocol_sha256": sha256_file(MANIFEST.with_name(MANIFEST.name + ".protocol.json")),
            "vdn_cohort_sha256": sha256_file(VDN_COHORT),
            "vdn_preflight_sha256": sha256_file(VDN_PREFLIGHT),
            "checkpoint_by_seed": prepared["checkpoint_identity"],
        },
        "generator": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(source_path),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    comparison = {
        "schema_version": 1,
        "protocol": "transformer_vdn_official200_fadr_v2_same_sample_train_oof_comparison_v1",
        "status": "frozen",
        "scope": "SyncG/train grouped OOF only",
        "cohort": summary["cohort"],
        "scoring": {
            "normalized_error": "abs(prediction-ground_truth)/abs(scale_end-scale_start)",
            "failure_penalty": 1.0,
            "denominator": "all 4380 bound rows",
            "acc_thresholds": [0.01, 0.02, 0.05],
            "bootstrap_unit": "physical_group",
        },
        "metrics": {
            "original_transformer": _metrics(transformer_error_array, transformer_success_array),
            "vdn_official200_grouped_oof": vdn_metrics,
            "fadr_v2_full": {
                "per_seed": fadr_metrics,
                "three_seed_mean": fadr_mean_metrics,
            },
        },
        "comparisons": comparisons,
        "claim_boundary": summary["claim_boundary"],
        "sources": {
            "original_transformer_oof": {
                "path": str(ORIGINAL_OOF.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(ORIGINAL_OOF),
            },
            "vdn_official200_predictions": summary["predictions"],
            "fadr_v2_joint_oof_by_seed": fadr_sources,
            "vdn_summary": {
                "path": "summary.json",
                "canonical_payload_sha256": _canonical_sha256(summary),
            },
        },
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "generator": summary["generator"],
    }
    return summary, comparison


def main() -> int:
    args = _parse_args()
    _require(args.batch_size > 0, "batch size must be positive")
    _require(args.bootstrap_iterations > 0, "bootstrap iterations must be positive")
    output_root = _assert_train_only_path(args.output_root, must_exist=False)
    predictions_path = output_root / "predictions.jsonl"
    summary_path = output_root / "summary.json"
    comparison_path = output_root / "comparison.json"
    for path in (predictions_path, summary_path, comparison_path):
        _assert_train_only_path(path, must_exist=False)
    prepared = _validate_inputs()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "samples": EXPECTED_SAMPLES,
                    "physical_groups": EXPECTED_GROUPS,
                    "restricted_data_opened_or_read": False,
                    "sample_ids_ordered_sha256": prepared["sample_ids_ordered_sha256"],
                },
                indent=2,
            )
        )
        return 0
    if any(path.exists() for path in (predictions_path, summary_path, comparison_path)):
        raise FileExistsError(f"frozen output already exists: {output_root}")
    device = torch.device(args.device)
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    amp = bool(device.type == "cuda" and not args.no_amp)
    vdn_rows = _evaluate(prepared, device=device, batch_size=args.batch_size, amp=amp)
    _atomic_jsonl(predictions_path, vdn_rows)
    summary, comparison = _build_artifacts(
        prepared,
        vdn_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        predictions_path=predictions_path,
    )
    _atomic_json(summary_path, summary)
    _atomic_json(comparison_path, comparison)
    print(summary_path)
    print(comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
