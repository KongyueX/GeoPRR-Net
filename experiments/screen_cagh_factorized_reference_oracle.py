"""Factorized train-only oracle for pointer and reference evidence.

The four cells independently replace the deployable PEPD direction and the
runtime reference arc with their annotation counterparts.  No network is
trained and only the frozen ``algorithm_fit`` inventory is opened.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd_progress import TRANSPORT_BRANCH_NAMES
from experiments.screen_cagh_internal_oracle import (
    BATCH_SIZE,
    BOOTSTRAP_REPETITIONS,
    PROGRESS_BINS,
    RESTRICTED_NAMES,
    _branch_indices,
    _method_metrics,
)
from experiments.train_geopepd_progress_v2_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    _progress_inputs_v2,
)
from experiments.train_geopepd_train_only_probe import (
    EXPECTED_FIT,
    _by_id,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
)
from experiments.train_uhpf_shared_mask_geometry_probe import (
    DEFAULT_STAGE_A_CHECKPOINT,
    _read_jsonl_subset,
    frozen_components,
    load_scope_records,
    load_visual_model,
    spatial_encoder_cache,
)
from experiments.vdn_baseline import image_angle_from_direction, transform_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_factorized_reference_oracle_train_only_screen_v1"
SEED = 20260806
METHODS = (
    "pepd_runtime_reference",
    "gt_pointer_runtime_reference",
    "pepd_gt_scalemark_reference",
    "gt_pointer_gt_scalemark_reference",
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/runs/cagh_factorized_reference_oracle_train_only_screen_v1"
)
DEFAULT_FIT_SPATIAL_CACHE = (
    PROJECT_ROOT
    / "artifacts/runs/uhpf_shared_mask_geometry_train_only_probe_v1/cache"
    / "algorithm_fit_encoder_spatial_fp16.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument(
        "--fit-spatial-cache", type=Path, default=DEFAULT_FIT_SPATIAL_CACHE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-screen", action="store_true")
    return parser.parse_args()


def derive_gt_scalemark_reference(
    metadata: Mapping[str, Any], sample_id: str
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return pointer points and the ordered annotation reference arc.

    ``ScaleMark.all_kp`` is ordered from the scale-start mark to the scale-end
    mark.  Angles use the production image convention: x right, y down, zero
    downward, and increasing counter-clockwise in the displayed image.
    """

    pointer: Mapping[str, Any] | None = None
    scale_mark: Mapping[str, Any] | None = None
    for item in metadata.get("keypoints") or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("type") or "").casefold()
        if kind == "pointer":
            pointer = item
        elif kind == "scalemark":
            scale_mark = item
    if pointer is None or scale_mark is None:
        raise ValueError(f"{sample_id}: Pointer/ScaleMark annotations absent")
    tip = np.asarray(pointer.get("outside_kp"), dtype=np.float32).reshape(-1)
    tail = np.asarray(pointer.get("origin_kp"), dtype=np.float32).reshape(-1)
    marks = scale_mark.get("all_kp")
    if tip.size < 2 or tail.size < 2 or not isinstance(marks, Sequence) or len(marks) < 2:
        raise ValueError(f"{sample_id}: malformed pointer/scale-mark annotations")
    start_mark = np.asarray(marks[0], dtype=np.float32).reshape(-1)
    end_mark = np.asarray(marks[-1], dtype=np.float32).reshape(-1)
    if start_mark.size < 2 or end_mark.size < 2:
        raise ValueError(f"{sample_id}: malformed ordered scale endpoints")
    numeric = np.concatenate((tip[:2], tail[:2], start_mark[:2], end_mark[:2]))
    if not np.isfinite(numeric).all():
        raise ValueError(f"{sample_id}: non-finite keypoint annotation")
    start_angle = image_angle_from_direction(start_mark[:2] - tail[:2])
    end_angle = image_angle_from_direction(end_mark[:2] - tail[:2])
    angle_range = (end_angle - start_angle) % 360.0
    if not 1e-8 < angle_range < 360.0:
        raise ValueError(f"{sample_id}: invalid ordered GT reference range")
    return tip[:2], tail[:2], float(start_angle), float(angle_range)


def _factorized_annotations(
    args: argparse.Namespace, records: Sequence[Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    ids = {record.sample_id for record in records}
    raw = _read_jsonl_subset(Path(args.raw_clean), ids, label="cagh_factorized_raw")
    raw_by_id = _by_id(raw)
    feature_rows = _read_jsonl(Path(args.fit_features), label="cagh_factorized_features")
    feature_by_id = _by_id(feature_rows)
    if set(feature_by_id) != ids:
        raise ValueError("factorized feature inventory differs from fit manifest")
    tips, tails, starts, ranges, folds = [], [], [], [], []
    group_fold: dict[str, str] = {}
    for record in records:
        metadata = raw_by_id[record.sample_id].get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{record.sample_id}: raw metadata absent")
        tip, tail, start, angle_range = derive_gt_scalemark_reference(
            metadata, record.sample_id
        )
        tip_crop = transform_point(tip, record.affine) / 255.0
        tail_crop = transform_point(tail, record.affine) / 255.0
        if (
            not np.isfinite(tip_crop).all()
            or not np.isfinite(tail_crop).all()
            or np.linalg.norm(tip_crop - tail_crop) <= 1e-6
        ):
            raise ValueError(f"{record.sample_id}: invalid transformed pointer ray")
        fold = str(feature_by_id[record.sample_id].get("support_fold_id") or "")
        if fold not in {"support-00", "support-01"}:
            raise ValueError(f"{record.sample_id}: invalid support fold {fold!r}")
        previous = group_fold.setdefault(record.group_id, fold)
        if previous != fold:
            raise ValueError(f"group {record.group_id} crosses support folds")
        tips.append(tip_crop)
        tails.append(tail_crop)
        starts.append(math.radians(start))
        ranges.append(math.radians(angle_range))
        folds.append(0 if fold == "support-00" else 1)
    counts = Counter(folds)
    if set(counts) != {0, 1}:
        raise ValueError("both grouped support folds are required")
    return (
        torch.from_numpy(np.stack(tips)).float(),
        torch.from_numpy(np.stack(tails)).float(),
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor(ranges, dtype=torch.float32),
        torch.tensor(folds, dtype=torch.long),
        {f"support-{index:02d}": int(counts[index]) for index in (0, 1)},
    )


def factorized_decision(
    methods: Mapping[str, Mapping[str, Any]],
    bootstraps: Mapping[str, Mapping[str, float]],
    conservation: Mapping[str, Any],
) -> dict[str, Any]:
    raw = methods["pepd_runtime_reference"]
    gt_reference = methods["pepd_gt_scalemark_reference"]
    comparison = bootstraps["pepd_gt_reference_vs_pepd_runtime_reference"]
    rules = {
        "pepd_gt_reference_better_than_raw_pepd": (
            gt_reference["full_denominator_nmae"] < raw["full_denominator_nmae"]
        ),
        "pepd_gt_reference_ci95_upper_below_zero": comparison["ci95_high"] < 0.0,
        "pepd_gt_reference_p95_improved": (
            gt_reference["covered_p95_absolute_error"]
            < raw["covered_p95_absolute_error"]
        ),
        "gt_both_all_samples_valid": (
            conservation["gt_both_valid_samples"]
            == conservation["expected_samples"]
        ),
        "gt_both_posterior_mass_conserved": (
            conservation["gt_both_posterior_mass_error_max"] <= 1e-6
        ),
        "gt_reference_ranges_valid": (
            conservation["gt_reference_range_min_degrees"] > 0.0
            and conservation["gt_reference_range_max_degrees"] < 360.0
        ),
        "all_numeric_diagnostics_finite": conservation["nonfinite"] == 0,
    }
    return {
        "label": "REFERENCE_HEAD_GO" if all(rules.values()) else "REFERENCE_HEAD_NO_GO",
        "rules": rules,
    }


def _subset_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {method: _method_metrics(rows, method) for method in METHODS}


@torch.inference_mode()
def evaluate_factorized_once(
    visual: Any,
    spatial: torch.Tensor,
    records: Sequence[Any],
    tips: torch.Tensor,
    tails: torch.Tensor,
    gt_start: torch.Tensor,
    gt_range: torch.Tensor,
    folds: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _progress_inputs_v2(records)
    branches = _branch_indices(records)
    rows: list[dict[str, Any]] = []
    gt_both_mass: list[float] = []
    runtime_pointer_mass: list[float] = []
    pepd_mass: list[float] = []
    for offset in range(0, len(records), BATCH_SIZE):
        stop = min(len(records), offset + BATCH_SIZE)
        inputs = [value[offset:stop].to(device) for value in tensors]
        spatial_batch = spatial[offset:stop].to(device)
        runtime_visual, _, _, _ = frozen_components(visual, spatial_batch, inputs)
        pooled = F.adaptive_avg_pool2d(spatial_batch.float(), 1).flatten(1)
        count = stop - offset
        gt_start_batch = gt_start[offset:stop].to(device)
        gt_range_batch = gt_range[offset:stop].to(device)
        gt_available = torch.ones(count, dtype=torch.bool, device=device)
        gt_visual = visual.forward_from_encoder_pooled(
            pooled,
            geometry_features=None,
            mgc_progress=None,
            reference_start_angle=gt_start_batch,
            reference_range_angle=gt_range_batch,
            crop_affine=inputs[4],
            geometry_available=None,
            reference_available=gt_available,
            transport_runtime_features=inputs[9],
        )
        runtime_pointer = differentiable_keypoint_reference_solver(
            tips[offset:stop].to(device),
            tails[offset:stop].to(device),
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[6],
            branches[offset:stop].to(device),
            progress_bins=PROGRESS_BINS,
        )
        gt_both = differentiable_keypoint_reference_solver(
            tips[offset:stop].to(device),
            tails[offset:stop].to(device),
            gt_start_batch,
            gt_range_batch,
            inputs[4],
            gt_available,
            branches[offset:stop].to(device),
            progress_bins=PROGRESS_BINS,
        )
        gt_both_mass.extend(gt_both.posterior_mass_error.cpu().tolist())
        runtime_pointer_mass.extend(
            runtime_pointer.posterior_mass_error.cpu().tolist()
        )
        pepd_mass.extend(
            torch.abs(
                torch.exp(gt_visual.raw_visual_progress_log_probability).sum(1) - 1.0
            ).cpu().tolist()
        )
        for local, record in enumerate(records[offset:stop]):
            runtime_pepd_valid = bool(runtime_visual.valid[local])
            runtime_pointer_valid = bool(runtime_pointer.valid[local])
            gt_pepd_valid = bool(gt_visual.valid[local])
            gt_both_valid = bool(gt_both.valid[local])
            progress = {
                "pepd_runtime_reference": (
                    float(runtime_visual.raw_visual_expected_progress[local])
                    if runtime_pepd_valid
                    else None
                ),
                "gt_pointer_runtime_reference": (
                    float(runtime_pointer.expected_progress[local])
                    if runtime_pointer_valid
                    else None
                ),
                "pepd_gt_scalemark_reference": (
                    float(gt_visual.raw_visual_expected_progress[local])
                    if gt_pepd_valid
                    else None
                ),
                "gt_pointer_gt_scalemark_reference": (
                    float(gt_both.expected_progress[local])
                    if gt_both_valid
                    else None
                ),
            }
            errors = {
                name: abs(value - record.target_progress) if value is not None else 1.0
                for name, value in progress.items()
            }
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "support_fold_id": f"support-{int(folds[offset + local]):02d}",
                    "runtime_reference_branch": TRANSPORT_BRANCH_NAMES[
                        int(branches[offset + local])
                    ],
                    "gt_reference": {
                        "start_angle_degrees": math.degrees(
                            float(gt_start[offset + local])
                        ),
                        "range_angle_degrees": math.degrees(
                            float(gt_range[offset + local])
                        ),
                    },
                    "valid": {
                        "pepd_runtime_reference": runtime_pepd_valid,
                        "gt_pointer_runtime_reference": runtime_pointer_valid,
                        "pepd_gt_scalemark_reference": gt_pepd_valid,
                        "gt_pointer_gt_scalemark_reference": gt_both_valid,
                    },
                    "progress": progress,
                    "errors": errors,
                }
            )
    methods = _subset_metrics(rows)
    comparisons = (
        (
            "pepd_gt_reference_vs_pepd_runtime_reference",
            "pepd_gt_scalemark_reference",
            "pepd_runtime_reference",
        ),
        (
            "gt_pointer_gt_reference_vs_gt_pointer_runtime_reference",
            "gt_pointer_gt_scalemark_reference",
            "gt_pointer_runtime_reference",
        ),
        (
            "gt_pointer_vs_pepd_at_runtime_reference",
            "gt_pointer_runtime_reference",
            "pepd_runtime_reference",
        ),
        (
            "gt_pointer_vs_pepd_at_gt_reference",
            "gt_pointer_gt_scalemark_reference",
            "pepd_gt_scalemark_reference",
        ),
    )
    bootstraps = {
        name: grouped_bootstrap_delta(
            rows,
            candidate,
            baseline,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=SEED,
        )
        for name, candidate, baseline in comparisons
    }
    all_numeric = np.asarray(
        gt_both_mass
        + runtime_pointer_mass
        + pepd_mass
        + gt_start.tolist()
        + gt_range.tolist(),
        dtype=np.float64,
    )
    conservation = {
        "expected_samples": len(rows),
        "gt_both_valid_samples": sum(
            bool(row["valid"]["gt_pointer_gt_scalemark_reference"])
            for row in rows
        ),
        "gt_both_posterior_mass_error_max": max(gt_both_mass, default=0.0),
        "runtime_pointer_posterior_mass_error_max": max(
            runtime_pointer_mass, default=0.0
        ),
        "pepd_gt_reference_posterior_mass_error_max": max(pepd_mass, default=0.0),
        "gt_reference_range_min_degrees": math.degrees(float(gt_range.min())),
        "gt_reference_range_max_degrees": math.degrees(float(gt_range.max())),
        "nonfinite": int((~np.isfinite(all_numeric)).sum()),
    }
    by_fold = {
        fold: _subset_metrics(
            [row for row in rows if row["support_fold_id"] == fold]
        )
        for fold in ("support-00", "support-01")
    }
    result = {
        "methods": methods,
        "grouped_support_holdout": by_fold,
        "grouped_bootstrap": bootstraps,
        "conservation": conservation,
    }
    result["decision"] = factorized_decision(methods, bootstraps, conservation)
    return result, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="cagh_factorized_output")
    records, mask_paths, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({record.group_id for record in records})) != EXPECTED_FIT:
        raise ValueError("algorithm_fit inventory drifted")
    tips, tails, gt_start, gt_range, folds, fold_counts = _factorized_annotations(
        args, records
    )
    validation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "mode": "algorithm_fit_only",
        "fit": {
            "samples": len(records),
            "groups": len({record.group_id for record in records}),
            "pointer_masks_inventoried": len(mask_paths),
            "pointer_and_ordered_scalemark_annotations": len(tips),
            "runtime_reference_available": int(_progress_inputs_v2(records)[6].sum()),
            "gt_reference_available": len(records),
            "support_folds": fold_counts,
        },
        "factorization": {
            "rows": ["runtime_pepd", "gt_pointer"],
            "columns": ["runtime_reference", "gt_scalemark_reference"],
            "gt_reference_derivation": {
                "center": "Pointer.origin_kp",
                "start_mark": "ordered ScaleMark.all_kp[0]",
                "end_mark": "ordered ScaleMark.all_kp[-1]",
                "image_coordinates": "x right, y down",
                "angle_degrees": "(degrees(atan2(dx, -dy)) - 180) mod 360",
                "range_degrees": "(end_angle - start_angle) mod 360",
                "progress_candidate_direction": "[-sin(start + range*p), cos(start + range*p)]",
                "crop_direction": "linear(original_to_crop_affine) @ original_direction",
            },
            "network_training": False,
            "progress_bins": PROGRESS_BINS,
        },
        "input_sha256": identities,
        "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES},
    }
    _write_json(output_dir / "validation.json", validation)
    if args.validate_only:
        return output_dir / "validation.json"
    if (output_dir / "summary.json").exists():
        raise RuntimeError("formal factorized CAGH oracle screen already exists")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    visual, stage_identity = load_visual_model(Path(args.stage_a_checkpoint), device)
    spatial = spatial_encoder_cache(
        visual,
        records,
        Path(args.fit_spatial_cache),
        scope="algorithm_fit",
        checkpoint_sha256=stage_identity["sha256"],
        device=device,
        batch_size=64,
        workers=0,
    )
    metrics, predictions = evaluate_factorized_once(
        visual,
        spatial,
        records,
        tips,
        tails,
        gt_start,
        gt_range,
        folds,
        device=device,
    )
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    summary = {
        **validation,
        "status": "complete",
        "metrics": metrics,
        "stage_a_checkpoint": stage_identity,
        "artifacts": {"predictions": str(predictions_path)},
    }
    _write_json(output_dir / "summary.json", summary)
    return output_dir / "summary.json"


if __name__ == "__main__":
    print(run(parse_args()))
