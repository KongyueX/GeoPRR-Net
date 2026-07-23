"""Evaluate a frozen train-only progress calibrator on one condition."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from experiments.evaluate_uncertainty_fusion import _validate_vector_evaluation
from experiments.progress_calibrator import (
    FEATURE_NAMES,
    PROGRESS_CALIBRATOR_PROTOCOL,
    apply_progress_correction,
    ensemble_prediction,
    extract_progress_features,
    feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float, normalized_error, rows_by_id
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


EVALUATION_PROTOCOL = "frozen_progress_calibrator_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--quality-predictions", type=Path)
    parser.add_argument("--uncertainty-router-predictions", type=Path)
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=Path(
            "artifacts/runs/progress_calibrator_syncg/model/progress_calibrator.joblib"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _base_prediction(row: Mapping[str, Any]) -> float | None:
    nested = row.get("base")
    if isinstance(nested, Mapping):
        return finite_float(nested.get("prediction"))
    return finite_float((row.get("predictions") or {}).get("ours"))


def _flat_prediction(row: Mapping[str, Any]) -> float | None:
    return finite_float(row.get("prediction")) if row.get("status", True) is not False else None


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float | None]
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, predictions)],
        dtype=np.float64,
    )
    successful = np.asarray([finite_float(value) is not None for value in predictions])
    return (
        {
            "samples": len(rows),
            "successful": int(np.sum(successful)),
            "failures": int(np.sum(~successful)),
            "coverage": float(np.mean(successful)),
            "nmae": float(np.mean(errors)),
            "acc_1pct": float(np.mean(errors <= 0.01)),
            "acc_2pct": float(np.mean(errors <= 0.02)),
            "acc_5pct": float(np.mean(errors <= 0.05)),
            "nrmse": float(np.sqrt(np.mean(errors**2))),
        },
        errors,
        successful,
    )


def _paired_group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    delta = float(np.mean(candidate) - np.mean(baseline))
    if iterations <= 0:
        return {
            "delta_nmae": delta,
            "group_bootstrap_95ci": None,
            "iterations": 0,
            "groups": int(len(np.unique(groups))),
        }
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected])
        deltas[iteration] = float(
            np.mean(candidate[selected_indices]) - np.mean(baseline[selected_indices])
        )
    return {
        "delta_nmae": delta,
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "iterations": int(iterations),
        "groups": int(len(unique)),
    }


def main() -> None:
    args = parse_args()
    path_names = (
        "vector_predictions",
        "reference_predictions",
        "calibrator",
        "output_dir",
    )
    for name in path_names:
        setattr(args, name, getattr(args, name).resolve())
    optional_names = (
        "base_predictions",
        "quality_predictions",
        "uncertainty_router_predictions",
    )
    for name in optional_names:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    required = [args.vector_predictions, args.reference_predictions, args.calibrator]
    required.extend(
        getattr(args, name) for name in optional_names if getattr(args, name) is not None
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")

    artifact = joblib.load(args.calibrator)
    if artifact.get("protocol") != PROGRESS_CALIBRATOR_PROTOCOL:
        raise ValueError("calibrator artifact has the wrong protocol")
    if artifact.get("train_only_certified") is not True:
        raise ValueError("calibrator artifact does not certify train-only fitting")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("calibrator feature schema changed")
    if artifact.get("test_sets_used") != []:
        raise ValueError("calibrator artifact lists test-set use")
    feature_source = PROJECT_DIR / "experiments" / "progress_calibrator.py"
    sources = artifact.get("source_sha256") or {}
    if sources.get("features_and_policy") != sha256_file(feature_source):
        raise ValueError("calibrator feature/policy source changed after fitting")
    vector_audit = _validate_vector_evaluation(args.vector_predictions, args.condition)

    mappings = {
        "vector": rows_by_id(args.vector_predictions),
        "reference": rows_by_id(args.reference_predictions),
    }
    optional_paths = {
        "base": args.base_predictions,
        "quality_router_v1": args.quality_predictions,
        "uncertainty_router": args.uncertainty_router_predictions,
    }
    for name, path in optional_paths.items():
        if path is not None:
            mappings[name] = rows_by_id(path)
    identifiers = list(mappings["vector"])
    expected = set(identifiers)
    for name, mapping in mappings.items():
        if set(mapping) != expected:
            raise ValueError(
                f"{name} IDs differ from vector: missing={len(expected-set(mapping))}, "
                f"extra={len(set(mapping)-expected)}"
            )
    rows = [mappings["vector"][sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")) for row in rows],
        dtype=object,
    )
    matrix = feature_matrix(
        [
            extract_progress_features(
                mappings["vector"][sample_id],
                reference_row=mappings["reference"][sample_id],
            )
            for sample_id in identifiers
        ]
    )
    residual, ensemble_std = ensemble_prediction(artifact["estimator"], matrix)
    correction_clip = float(artifact["correction_clip"])
    raw_values: list[float | None] = []
    corrected_progress: list[float | None] = []
    calibrated_values: list[float | None] = []
    routes: list[str] = []
    for row, predicted_residual in zip(rows, residual):
        raw = _flat_prediction(row)
        progress = (
            apply_progress_correction(
                row.get("progress"),
                predicted_residual,
                correction_clip=correction_clip,
            )
            if raw is not None
            else None
        )
        calibrated = reading_from_progress(
            progress, row.get("scale_start"), row.get("scale_end")
        )
        raw_values.append(raw)
        corrected_progress.append(progress)
        calibrated_values.append(calibrated)
        routes.append("calibrated" if calibrated is not None else "failure")

    predictions: dict[str, Sequence[float | None]] = {
        "raw_probabilistic_vector": raw_values,
        "calibrated_vector": calibrated_values,
        "vdn": [_flat_prediction(mappings["reference"][key]) for key in identifiers],
    }
    if "base" in mappings:
        predictions["base_mask"] = [_base_prediction(mappings["base"][key]) for key in identifiers]
    if "quality_router_v1" in mappings:
        predictions["quality_router_v1"] = [
            _flat_prediction(mappings["quality_router_v1"][key]) for key in identifiers
        ]
    if "uncertainty_router" in mappings:
        predictions["uncertainty_router"] = [
            _flat_prediction(mappings["uncertainty_router"][key]) for key in identifiers
        ]
    oracle_values: list[float | None] = []
    for row, raw, calibrated in zip(rows, raw_values, calibrated_values):
        if raw is None:
            oracle_values.append(calibrated)
        elif calibrated is None:
            oracle_values.append(raw)
        else:
            oracle_values.append(
                calibrated
                if normalized_error(row, calibrated) < normalized_error(row, raw)
                else raw
            )
    predictions["raw_calibrated_oracle"] = oracle_values

    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    successful: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name], successful[name] = _metrics(rows, values)
    corrected = successful["calibrated_vector"]
    positive = corrected & (errors["calibrated_vector"] < errors["raw_probabilistic_vector"])
    negative = corrected & (errors["calibrated_vector"] > errors["raw_probabilistic_vector"])

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    if any(path.exists() for path in (output_predictions, output_summary)) and not args.overwrite:
        raise FileExistsError("progress-calibrator evaluation exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(identifiers):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "group_id": rows[index].get("group_id"),
                        "dataset": rows[index].get("dataset"),
                        "split": rows[index].get("split"),
                        "ground_truth": rows[index].get("ground_truth"),
                        "scale_start": rows[index].get("scale_start"),
                        "scale_end": rows[index].get("scale_end"),
                        "condition": args.condition,
                        "status": calibrated_values[index] is not None,
                        "prediction": calibrated_values[index],
                        "progress": corrected_progress[index],
                        "route": routes[index],
                        "raw_prediction": raw_values[index],
                        "raw_progress": finite_float(rows[index].get("progress")),
                        "predicted_progress_residual": finite_float(residual[index]),
                        "progress_ensemble_std": finite_float(ensemble_std[index]),
                        "correction_clip": correction_clip,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    comparisons = {
        "calibrated_vs_raw_vector": _paired_group_bootstrap(
            errors["calibrated_vector"], errors["raw_probabilistic_vector"], groups,
            seed=args.seed, iterations=args.bootstrap_iterations,
        ),
        "calibrated_vs_vdn": _paired_group_bootstrap(
            errors["calibrated_vector"], errors["vdn"], groups,
            seed=args.seed + 1, iterations=args.bootstrap_iterations,
        ),
    }
    comparison_names = {
        "base_mask": "calibrated_vs_base_mask",
        "quality_router_v1": "calibrated_vs_quality_router_v1",
        "uncertainty_router": "calibrated_vs_uncertainty_router",
    }
    for offset, (name, comparison_name) in enumerate(comparison_names.items(), start=2):
        if name in errors:
            comparisons[comparison_name] = _paired_group_bootstrap(
                errors["calibrated_vector"], errors[name], groups,
                seed=args.seed + offset, iterations=args.bootstrap_iterations,
            )
    valid_uncertainty = corrected & np.isfinite(ensemble_std)
    uncertainty_correlation = (
        float(
            np.corrcoef(
                ensemble_std[valid_uncertainty],
                errors["calibrated_vector"][valid_uncertainty],
            )[0, 1]
        )
        if int(np.sum(valid_uncertainty)) > 1
        else math.nan
    )
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "condition": args.condition,
        "samples": len(rows),
        "groups": int(len(np.unique(groups))),
        "calibrator": str(args.calibrator),
        "calibrator_sha256": sha256_file(args.calibrator),
        "training_oof_pairs_sha256": artifact.get("training_oof_pairs_sha256"),
        "correction_clip": correction_clip,
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "transfers": {
            "positive": int(np.sum(positive)),
            "negative": int(np.sum(negative)),
            "ties": int(np.sum(corrected & (errors["calibrated_vector"] == errors["raw_probabilistic_vector"]))),
        },
        "uncertainty": {
            "mean_ensemble_std": float(np.mean(ensemble_std[valid_uncertainty])),
            "ensemble_std_error_correlation": uncertainty_correlation,
        },
        "inputs": {
            "vector": str(args.vector_predictions),
            "vector_sha256": sha256_file(args.vector_predictions),
            "vector_audit": vector_audit,
            "reference_vdn": str(args.reference_predictions),
            "reference_vdn_sha256": sha256_file(args.reference_predictions),
            **{
                name: str(path) if path is not None else None
                for name, path in optional_paths.items()
            },
            **{
                f"{name}_sha256": sha256_file(path) if path is not None else None
                for name, path in optional_paths.items()
            },
        },
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "feature_source_sha256": sha256_file(feature_source),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(output_summary)


if __name__ == "__main__":
    main()
