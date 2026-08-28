"""Measure routing behavior and candidate complementarity on the SyncG holdout."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import torch

from experiments.evaluate_a15_2_mett_syncg import DEFAULT_MANIFEST, DEFAULT_REFERENCE
from experiments.evaluate_a15_2_syncg_scene_holdout import CONDITIONS, _SharedPixelDataset
from experiments.evaluate_remstnet_syncg import _load_reference
from experiments.run_cagh_v5_plain_paper_batch import load_manifest
from experiments.train_support_geometry_multiview_efficientnet_pilot import _loader
from experiments.unified_pointer_reader import (
    CANDIDATE_NAMES,
    FULL,
    PUBLICATION_NAME,
    load_unified_pointer_reader_checkpoint,
)


PROTOCOL: Final[str] = "unified_pointer_reader_syncg_routing_analysis_v1"
FIXED_PRIOR: Final[tuple[float, float, float]] = (0.50, 0.25, 0.25)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    selected = np.isfinite(left) & np.isfinite(right)
    if int(selected.sum()) < 2:
        return None
    x = left[selected]
    y = right[selected]
    if float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _metric(values: np.ndarray) -> dict[str, float]:
    _require(values.ndim == 1 and bool(values.size), "metric values are empty")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "routing rows are empty")
    target = np.asarray([row["normalized_target"] for row in rows], dtype=np.float64)
    final = np.asarray([row["prediction"] for row in rows], dtype=np.float64)
    candidates = np.asarray([row["candidate_predictions"] for row in rows], dtype=np.float64)
    weights = np.asarray([row["routing_weights"] for row in rows], dtype=np.float64)
    predicted_gains = np.asarray([row["predicted_candidate_gains"] for row in rows], dtype=np.float64)
    entropy = np.asarray([row["polar_entropy"] for row in rows], dtype=np.float64)
    concentration = np.asarray([row["polar_concentration"] for row in rows], dtype=np.float64)
    relation = np.asarray([row["relation_available"] for row in rows], dtype=np.bool_)
    final_error = np.abs(final - target)
    candidate_errors = np.abs(candidates - target[:, None])
    fixed = candidates @ np.asarray(FIXED_PRIOR, dtype=np.float64)
    fixed_error = np.abs(fixed - target)
    oracle_error = candidate_errors.min(axis=1)
    selected_candidate = weights.argmax(axis=1)
    oracle_candidate = candidate_errors.argmin(axis=1)
    realized_gains = candidate_errors[:, :1] - candidate_errors[:, 1:]
    return {
        "rows": len(rows),
        "nmae": float(final_error.mean()),
        "acc_at_2_percent": float((final_error <= 0.02).mean()),
        "absolute_error": _metric(final_error),
        "candidate_nmae": {
            name: float(candidate_errors[:, index].mean())
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "fixed_prior_nmae": float(fixed_error.mean()),
        "oracle_discrete_nmae": float(oracle_error.mean()),
        "adaptive_minus_fixed_prior_nmae": float(final_error.mean() - fixed_error.mean()),
        "adaptive_minus_oracle_nmae": float(final_error.mean() - oracle_error.mean()),
        "adaptive_improvement_fraction_vs_fixed_prior": float((final_error < fixed_error).mean()),
        "mean_routing_weights": {
            name: float(weights[:, index].mean())
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "routing_argmax_fraction": {
            name: float((selected_candidate == index).mean())
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "oracle_candidate_fraction": {
            name: float((oracle_candidate == index).mean())
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "routing_argmax_matches_oracle_fraction": float(
            (selected_candidate == oracle_candidate).mean()
        ),
        "router_gain_calibration": {
            name: {
                "predicted_gain_mean": float(predicted_gains[:, index - 1].mean()),
                "realized_gain_mean": float(realized_gains[:, index - 1].mean()),
                "predicted_vs_realized_pearson": _pearson(
                    predicted_gains[:, index - 1], realized_gains[:, index - 1]
                ),
                "routing_weight_vs_realized_gain_pearson": _pearson(
                    weights[:, index], realized_gains[:, index - 1]
                ),
            }
            for index, name in enumerate(CANDIDATE_NAMES)
            if index > 0
        },
        "relation_available_fraction": float(relation.mean()),
        "polar_entropy_mean": float(entropy.mean()),
        "polar_concentration_mean": float(concentration.mean()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Unified pointer reader routing analysis",
        "",
        "| Condition | NMAE | Acc@2% | Base | Polar | Relational | w(base) | w(polar) | w(rel.) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ("all_conditions", *CONDITIONS):
        row = report["summary"][condition]
        candidates = row["candidate_nmae"]
        weights = row["mean_routing_weights"]
        lines.append(
            f"| {condition} | {row['nmae']:.6f} | {row['acc_at_2_percent']:.4f} | "
            f"{candidates['base']:.6f} | {candidates['polar_evidence']:.6f} | "
            f"{candidates['relational_transport']:.6f} | {weights['base']:.4f} | "
            f"{weights['polar_evidence']:.4f} | {weights['relational_transport']:.4f} |"
        )
    overall = report["summary"]["all_conditions"]
    lines.extend(
        [
            "",
            f"Adaptive − fixed-prior NMAE: {overall['adaptive_minus_fixed_prior_nmae']:.6f}",
            f"Adaptive − discrete-oracle NMAE: {overall['adaptive_minus_oracle_nmae']:.6f}",
            f"Router argmax/oracle agreement: {overall['routing_argmax_matches_oracle_fraction']:.4f}",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_routing(
    *,
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
    reference_path: Path,
    device_name: str,
    workers: int,
    batch_size: int,
    use_amp: bool,
    failure_cases: int,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"routing analysis output exists: {output}")
    _require(workers >= 0 and batch_size >= 1, "loader sizes are invalid")
    _require(failure_cases >= 1, "failure-case count must be positive")
    _reference, _reference_by_key, ordered_ids, targets = _load_reference(reference_path)
    manifest_rows = load_manifest(manifest_path)
    by_id = {row.sample_id: row for row in manifest_rows}
    _require(set(by_id) == set(ordered_ids), "SyncG holdout roster differs")
    ordered_manifest = tuple(by_id[sample_id] for sample_id in ordered_ids)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model, metadata = load_unified_pointer_reader_checkpoint(
        checkpoint_path, device=device
    )
    _require(metadata["variant"] == FULL, "routing analysis requires the full variant")
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            dataset = _SharedPixelDataset(ordered_manifest, targets, condition=condition)
            loader = _loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=20260818 + condition_index,
                cuda=device.type == "cuda",
            )
            for raw_batch in loader:
                raw = raw_batch["original_view"].to(device)
                normalized = raw_batch["sarn_view"].to(device)
                support = raw_batch["sarn_support_mask"].to(device)
                active = raw_batch["sarn_active"].to(device).bool()
                homography = raw_batch["raw_to_sarn_homography"].to(device)
                effective_active = active & (condition != "clean")
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    prediction = model(
                        raw,
                        normalized,
                        support,
                        sarn_active=effective_active,
                        raw_to_sarn_homography=homography,
                    )
                final = prediction["mean"].float().cpu()
                candidates = prediction["candidate_predictions"].float().cpu()
                weights = prediction["routing_weights"].float().cpu()
                gains = prediction["predicted_candidate_gains"].float().cpu()
                relation = prediction["relation_available"].bool().cpu()
                entropy = prediction["polar_entropy"].float().cpu()
                concentration = prediction["polar_concentration"].float().cpu()
                target = raw_batch["target"].float()
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                scenes = tuple(str(value) for value in raw_batch["scene_stem"])
                for index, sample_id in enumerate(ids):
                    scalar_target = float(target[index])
                    scalar_prediction = float(final[index])
                    row = {
                        "sample_id": sample_id,
                        "scene_stem": scenes[index],
                        "condition": str(condition),
                        "normalized_target": scalar_target,
                        "prediction": scalar_prediction,
                        "absolute_error": abs(scalar_prediction - scalar_target),
                        "candidate_predictions": [float(value) for value in candidates[index]],
                        "routing_weights": [float(value) for value in weights[index]],
                        "predicted_candidate_gains": [float(value) for value in gains[index]],
                        "relation_available": bool(relation[index]),
                        "polar_entropy": float(entropy[index]),
                        "polar_concentration": float(concentration[index]),
                    }
                    _require(
                        all(
                            math.isfinite(value)
                            for value in (
                                scalar_target,
                                scalar_prediction,
                                *row["candidate_predictions"],
                                *row["routing_weights"],
                                *row["predicted_candidate_gains"],
                            )
                        ),
                        "routing row contains a non-finite value",
                    )
                    rows.append(row)
    _require(
        len(rows) == len(ordered_ids) * len(CONDITIONS),
        "routing Cartesian row count differs",
    )
    summary = {"all_conditions": _summarize_rows(rows)}
    for condition in CONDITIONS:
        summary[str(condition)] = _summarize_rows(
            [row for row in rows if row["condition"] == condition]
        )
    hard = sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[
        :failure_cases
    ]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "publication_model": PUBLICATION_NAME,
        "model": metadata,
        "scope": {
            "formal_syncg_scene_holdout": True,
            "training_or_adaptation_during_analysis": False,
            "samples": len(ordered_ids),
            "conditions": list(CONDITIONS),
            "rows": len(rows),
            "fixed_prior": list(FIXED_PRIOR),
        },
        "candidate_names": list(CANDIDATE_NAMES),
        "summary": summary,
        "hardest_cases": hard,
        "per_sample_condition": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--failure-cases", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = analyze_routing(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        reference_path=args.reference,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
        failure_cases=args.failure_cases,
    )
    overall = report["summary"]["all_conditions"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(Path(args.output).resolve()),
                "nmae": overall["nmae"],
                "adaptive_minus_fixed_prior_nmae": overall[
                    "adaptive_minus_fixed_prior_nmae"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "analyze_routing", "render_markdown"]
