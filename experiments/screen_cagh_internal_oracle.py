"""Train-only CAGH keypoint/reference oracle screen.

This screen does not train a network and never opens algorithm_selection.  It
uses the frozen algorithm_fit inventory, grouped support folds, and the exact
keypoint/reference solver shared with CAGH to test whether the proposed
physical branch has enough oracle headroom to justify end-to-end training.
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

from experiments.cagh_net import differentiable_keypoint_reference_solver
from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd_progress import TRANSPORT_BRANCH_NAMES
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
from experiments.vdn_baseline import transform_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_internal_keypoint_oracle_train_only_screen_v1"
SEED = 20260806
PROGRESS_BINS = 72
BATCH_SIZE = 256
BOOTSTRAP_REPETITIONS = 5000
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/runs/cagh_internal_oracle_train_only_screen_v1"
)
DEFAULT_FIT_SPATIAL_CACHE = (
    PROJECT_ROOT
    / "artifacts/runs/uhpf_shared_mask_geometry_train_only_probe_v1/cache"
    / "algorithm_fit_encoder_spatial_fp16.pt"
)
RESTRICTED_NAMES = (
    "public_samples",
    "test_samples",
    "field_samples",
    "sealed_samples",
    "confirmatory_samples",
    "confirmation_a_samples",
    "confirmation_b_samples",
    "algorithm_selection_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument("--fit-spatial-cache", type=Path, default=DEFAULT_FIT_SPATIAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-screen", action="store_true")
    return parser.parse_args()


def _pointer_points(metadata: Mapping[str, Any], sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    for item in metadata.get("keypoints") or []:
        if not isinstance(item, Mapping) or str(item.get("type") or "").casefold() != "pointer":
            continue
        tip, tail = item.get("outside_kp"), item.get("origin_kp")
        if (
            isinstance(tip, Sequence)
            and not isinstance(tip, (str, bytes))
            and len(tip) >= 2
            and isinstance(tail, Sequence)
            and not isinstance(tail, (str, bytes))
            and len(tail) >= 2
        ):
            return np.asarray(tip[:2], np.float32), np.asarray(tail[:2], np.float32)
    raise ValueError(f"{sample_id}: pointer origin/outside keypoints absent")


def _fit_annotations(
    args: argparse.Namespace, records: Sequence[Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    ids = {record.sample_id for record in records}
    raw_rows = _read_jsonl_subset(Path(args.raw_clean), ids, label="cagh_fit_raw_subset")
    raw_by_id = _by_id(raw_rows)
    feature_rows = _read_jsonl(Path(args.fit_features), label="cagh_fit_features")
    feature_by_id = _by_id(feature_rows)
    if set(feature_by_id) != ids:
        raise ValueError("CAGH fit feature inventory differs from fit manifest")
    tips, tails, folds = [], [], []
    group_fold: dict[str, str] = {}
    for record in records:
        metadata = raw_by_id[record.sample_id].get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{record.sample_id}: raw metadata absent")
        tip, tail = _pointer_points(metadata, record.sample_id)
        tip_crop = transform_point(tip, record.affine) / 255.0
        tail_crop = transform_point(tail, record.affine) / 255.0
        if not np.isfinite(tip_crop).all() or not np.isfinite(tail_crop).all():
            raise ValueError(f"{record.sample_id}: non-finite transformed keypoint")
        if np.linalg.norm(tip_crop - tail_crop) <= 1e-6:
            raise ValueError(f"{record.sample_id}: collapsed transformed keypoint ray")
        fold_name = str(feature_by_id[record.sample_id].get("support_fold_id") or "")
        if fold_name not in {"support-00", "support-01"}:
            raise ValueError(f"{record.sample_id}: invalid support fold {fold_name!r}")
        prior = group_fold.setdefault(record.group_id, fold_name)
        if prior != fold_name:
            raise ValueError(f"group {record.group_id} crosses support folds")
        tips.append(tip_crop)
        tails.append(tail_crop)
        folds.append(0 if fold_name == "support-00" else 1)
    fold_counts = Counter(folds)
    if set(fold_counts) != {0, 1}:
        raise ValueError("both grouped support folds are required")
    return (
        torch.from_numpy(np.stack(tips)).float(),
        torch.from_numpy(np.stack(tails)).float(),
        torch.tensor(folds, dtype=torch.long),
        {f"support-{index:02d}": int(fold_counts[index]) for index in (0, 1)},
    )


def _branch_indices(records: Sequence[Any]) -> torch.Tensor:
    result = torch.zeros(len(records), dtype=torch.long)
    for index, record in enumerate(records):
        values = np.asarray(record.transport_runtime_features, dtype=np.float32)
        if values.shape != (24,):
            raise ValueError(f"{record.sample_id}: transport schema shape drifted")
        bits = np.nan_to_num(values[11:15], nan=0.0) > 0.5
        if int(bits.sum()) > 1:
            raise ValueError(f"{record.sample_id}: reference branch is not one-hot")
        if bits[0]:
            result[index] = 1
        elif bits[1]:
            result[index] = 2
        elif bits[2]:
            result[index] = 3
        else:
            result[index] = 0
    return result


def _method_metrics(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    if not rows:
        return {
            "full_denominator_nmae": None,
            "coverage": 0.0,
            "covered_samples": 0,
            "covered_nmae": None,
            "covered_p95_absolute_error": None,
        }
    covered_errors = [
        float(row["errors"][method])
        for row in rows
        if row["progress"][method] is not None
    ]
    return {
        "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
        "coverage": len(covered_errors) / len(rows),
        "covered_samples": len(covered_errors),
        "covered_nmae": float(np.mean(covered_errors)),
        "covered_p95_absolute_error": float(np.quantile(covered_errors, 0.95)),
    }


def _subset_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        method: _method_metrics(rows, method)
        for method in ("cagh_gt_keypoint_oracle", "pepd", "v2_visual")
    }


@torch.inference_mode()
def evaluate_fit_once(
    visual: Any,
    spatial: torch.Tensor,
    records: Sequence[Any],
    tips: torch.Tensor,
    tails: torch.Tensor,
    folds: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _progress_inputs_v2(records)
    branches = _branch_indices(records)
    rows: list[dict[str, Any]] = []
    mass_errors: list[float] = []
    for offset in range(0, len(records), BATCH_SIZE):
        stop = min(len(records), offset + BATCH_SIZE)
        inputs = [value[offset:stop].to(device) for value in tensors]
        visual_outputs, _, _, _ = frozen_components(
            visual, spatial[offset:stop].to(device), inputs
        )
        solver = differentiable_keypoint_reference_solver(
            tips[offset:stop].to(device),
            tails[offset:stop].to(device),
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[6],
            branches[offset:stop].to(device),
            progress_bins=PROGRESS_BINS,
        )
        mass_errors.extend(solver.posterior_mass_error.cpu().tolist())
        for local, record in enumerate(records[offset:stop]):
            valid = bool(solver.valid[local])
            baseline_valid = bool(visual_outputs.valid[local])
            progress = {
                "cagh_gt_keypoint_oracle": float(solver.expected_progress[local]) if valid else None,
                "pepd": float(visual_outputs.raw_visual_expected_progress[local]) if baseline_valid else None,
                "v2_visual": float(visual_outputs.visual_expected_progress[local]) if baseline_valid else None,
            }
            errors = {
                name: abs(value - record.target_progress) if value is not None else 1.0
                for name, value in progress.items()
            }
            branch_index = int(solver.reference_branch_index[local])
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "support_fold_id": f"support-{int(folds[offset + local]):02d}",
                    "reference_branch": TRANSPORT_BRANCH_NAMES[branch_index],
                    "progress": progress,
                    "errors": errors,
                    "solver": {
                        "valid": valid,
                        "quality": float(solver.quality[local]),
                        "pointer_length": float(solver.pointer_length[local]),
                        "concentration": float(solver.concentration[local]),
                        "posterior_mass_error": float(solver.posterior_mass_error[local]),
                    },
                }
            )
    methods = _subset_metrics(rows)
    by_fold = {
        fold: _subset_metrics([row for row in rows if row["support_fold_id"] == fold])
        for fold in ("support-00", "support-01")
    }
    by_branch = {}
    for branch in TRANSPORT_BRANCH_NAMES:
        subset = [row for row in rows if row["reference_branch"] == branch]
        by_branch[branch] = {"samples": len(subset), "methods": _subset_metrics(subset)}
    bootstraps = {
        "oracle_vs_pepd": grouped_bootstrap_delta(
            rows, "cagh_gt_keypoint_oracle", "pepd", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED
        ),
        "oracle_vs_v2_visual": grouped_bootstrap_delta(
            rows, "cagh_gt_keypoint_oracle", "v2_visual", repetitions=BOOTSTRAP_REPETITIONS, seed=SEED
        ),
    }
    quality = np.asarray([row["solver"]["quality"] for row in rows], dtype=np.float64)
    pointer_length = np.asarray([row["solver"]["pointer_length"] for row in rows], dtype=np.float64)
    branch_total = sum(item["samples"] for item in by_branch.values())
    numeric = np.concatenate(
        (np.asarray(mass_errors, dtype=np.float64), quality, pointer_length)
    )
    conservation = {
        "posterior_mass_error_max": max(mass_errors, default=0.0),
        "quality_min": float(quality.min()),
        "quality_max": float(quality.max()),
        "pointer_length_min": float(pointer_length.min()),
        "branch_samples_sum": branch_total,
        "expected_samples": len(rows),
        "nonfinite": int((~np.isfinite(numeric)).sum()),
    }
    rules = {
        "oracle_better_than_pepd": methods["cagh_gt_keypoint_oracle"]["full_denominator_nmae"] < methods["pepd"]["full_denominator_nmae"],
        "oracle_better_than_v2_visual": methods["cagh_gt_keypoint_oracle"]["full_denominator_nmae"] < methods["v2_visual"]["full_denominator_nmae"],
        "oracle_vs_pepd_ci95_upper_below_zero": bootstraps["oracle_vs_pepd"]["ci95_high"] < 0.0,
        "oracle_vs_v2_ci95_upper_below_zero": bootstraps["oracle_vs_v2_visual"]["ci95_high"] < 0.0,
        "coverage_conserved": methods["cagh_gt_keypoint_oracle"]["coverage"] == methods["v2_visual"]["coverage"],
        "p95_not_worse_than_best_baseline": methods["cagh_gt_keypoint_oracle"]["covered_p95_absolute_error"] <= min(methods["pepd"]["covered_p95_absolute_error"], methods["v2_visual"]["covered_p95_absolute_error"]),
        "posterior_mass_conserved": conservation["posterior_mass_error_max"] <= 1e-6,
        "quality_bounds_conserved": conservation["quality_min"] >= 0.0 and conservation["quality_max"] <= 1.0,
        "branch_accounting_conserved": branch_total == len(rows),
        "all_solver_diagnostics_finite": conservation["nonfinite"] == 0,
    }
    return {
        "methods": methods,
        "grouped_support_holdout": by_fold,
        "reference_branches": by_branch,
        "grouped_bootstrap": bootstraps,
        "conservation": conservation,
        "decision": {"label": "ORACLE_GO" if all(rules.values()) else "ORACLE_NO_GO", "rules": rules},
    }, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="cagh_oracle_output")
    records, mask_paths, identities = load_scope_records(args, scope="algorithm_fit")
    if (len(records), len({record.group_id for record in records})) != EXPECTED_FIT:
        raise ValueError("algorithm_fit inventory drifted")
    tips, tails, folds, fold_counts = _fit_annotations(args, records)
    branches = _branch_indices(records)
    synthetic = differentiable_keypoint_reference_solver(
        tips[:2], tails[:2], *_progress_inputs_v2(records[:2])[2:5], _progress_inputs_v2(records[:2])[6], branches[:2]
    )
    if not bool(torch.isfinite(synthetic.progress_log_probability).all()):
        raise FloatingPointError("CAGH solver preflight is non-finite")
    validation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "mode": "algorithm_fit_only",
        "fit": {
            "samples": len(records),
            "groups": len({record.group_id for record in records}),
            "pointer_masks_inventoried": len(mask_paths),
            "origin_outside_pairs": len(tips),
            "reference_available": int(_progress_inputs_v2(records)[6].sum()),
            "support_folds": fold_counts,
        },
        "solver": {
            "function": "experiments.cagh_net.differentiable_keypoint_reference_solver",
            "progress_bins": PROGRESS_BINS,
            "angular_sigma": 0.08,
            "network_training": False,
        },
        "input_sha256": identities,
        "restricted_data_use": {name: 0 for name in RESTRICTED_NAMES},
    }
    _write_json(output_dir / "validation.json", validation)
    if args.validate_only:
        return output_dir / "validation.json"
    if (output_dir / "summary.json").exists():
        raise RuntimeError("formal CAGH oracle screen already exists")
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
    metrics, predictions = evaluate_fit_once(
        visual, spatial, records, tips, tails, folds, device=device
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
