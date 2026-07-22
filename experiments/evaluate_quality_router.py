"""Apply the frozen train-only mask/vector quality router to one test condition."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.quality_router import (
    FEATURE_NAMES,
    QUALITY_ROUTER_PROTOCOL,
    extract_quality_features,
    feature_matrix,
    finite_float,
    normalized_error,
    route_prediction,
    rows_by_id,
    sha256_file,
)


EVALUATION_PROTOCOL = "frozen_quality_router_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        required=True,
        help="VDN comparison rows; only meter confidence is used by the router",
    )
    parser.add_argument(
        "--router",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/model/quality_router.joblib"),
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
    return finite_float((row.get("predictions") or {}).get("ours"))


def _vector_prediction(row: Mapping[str, Any]) -> float | None:
    return finite_float(row.get("prediction")) if row.get("status") is True else None


def _reference_prediction(row: Mapping[str, Any]) -> float | None:
    return finite_float(row.get("prediction")) if row.get("status") is True else None


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float | None]
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    errors = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, predictions)],
        dtype=np.float64,
    )
    successful = np.asarray([finite_float(value) is not None for value in predictions])
    metrics: dict[str, float | int] = {
        "samples": len(rows),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "nrmse": float(np.sqrt(np.mean(errors**2))),
    }
    return metrics, errors, successful


def _paired_group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
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
        "delta_nmae": float(np.mean(candidate) - np.mean(baseline)),
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "iterations": int(iterations),
        "groups": int(len(unique)),
    }


def _conditions(row: Mapping[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    values = metadata.get("environment_conditions") or []
    return {str(value).strip().lower() for value in values}


def _subgroup_summaries(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    selectors = {
        "blur": lambda values: "blur" in values,
        "tilted": lambda values: "tilted" in values,
        "blur_and_tilted": lambda values: "blur" in values and "tilted" in values,
        "low_light": lambda values: "low_light" in values,
        "occlusion": lambda values: "occlusion" in values,
    }
    row_conditions = [_conditions(row) for row in rows]
    result: dict[str, Any] = {}
    for name, selector in selectors.items():
        indices = [index for index, values in enumerate(row_conditions) if selector(values)]
        if not indices:
            continue
        subgroup_rows = [rows[index] for index in indices]
        result[name] = {
            "samples": len(indices),
            "metrics": {
                method: _metrics(
                    subgroup_rows, [values[index] for index in indices]
                )[0]
                for method, values in predictions.items()
            },
        }
    return result


def main() -> None:
    args = parse_args()
    for name in (
        "raw_predictions",
        "base_predictions",
        "vector_predictions",
        "reference_predictions",
        "router",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    for path in (
        args.raw_predictions,
        args.base_predictions,
        args.vector_predictions,
        args.reference_predictions,
        args.router,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact = joblib.load(args.router)
    if artifact.get("protocol") != QUALITY_ROUTER_PROTOCOL:
        raise ValueError("router artifact has the wrong protocol")
    if tuple(artifact.get("feature_names") or []) != FEATURE_NAMES:
        raise ValueError("router artifact feature schema changed")
    if artifact.get("test_sets_used") != []:
        raise ValueError("router artifact does not certify train-only fitting")
    source_hashes = artifact.get("source_sha256") or {}
    expected_feature_source = sha256_file(
        PROJECT_ROOT / "experiments" / "quality_router.py"
    )
    if source_hashes.get("features_and_policy") != expected_feature_source:
        raise ValueError("router runtime feature/policy source changed after fitting")
    threshold = float(artifact["threshold"])
    estimator = artifact["estimator"]

    raw = rows_by_id(args.raw_predictions)
    base = rows_by_id(args.base_predictions)
    vector = rows_by_id(args.vector_predictions)
    reference = rows_by_id(args.reference_predictions)
    identifiers = list(base)
    expected = set(identifiers)
    for name, mapping in (("raw", raw), ("vector", vector), ("reference", reference)):
        if set(mapping) != expected:
            raise ValueError(
                f"{name} IDs differ from base IDs: missing={len(expected-set(mapping))}, "
                f"extra={len(set(mapping)-expected)}"
            )
    rows = [base[sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")) for row in rows],
        dtype=object,
    )
    base_values = [_base_prediction(base[sample_id]) for sample_id in identifiers]
    vector_values = [_vector_prediction(vector[sample_id]) for sample_id in identifiers]
    reference_values = [
        _reference_prediction(reference[sample_id]) for sample_id in identifiers
    ]
    joint = np.asarray(
        [first is not None and second is not None for first, second in zip(base_values, vector_values)],
        dtype=bool,
    )
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    feature_rows = [
        extract_quality_features(
            raw_row=raw[sample_id],
            base_row=base[sample_id],
            vector_row=vector[sample_id],
            reference_row=reference[sample_id],
        )
        for sample_id in identifiers
    ]
    if np.any(joint):
        matrix = feature_matrix(feature_rows)
        scores[joint] = estimator.predict(matrix[joint])

    # Routing decisions are frozen before any ground-truth error is calculated.
    quality_values: list[float | None] = []
    routes: list[str] = []
    hard_values: list[float | None] = []
    oracle_values: list[float | None] = []
    for row, base_value, vector_value, score in zip(
        rows, base_values, vector_values, scores
    ):
        prediction, route = route_prediction(
            base_prediction=base_value,
            vector_prediction=vector_value,
            score=score,
            threshold=threshold,
        )
        quality_values.append(prediction)
        routes.append(route)
        hard_values.append(base_value if base_value is not None else vector_value)
        if base_value is None:
            oracle_values.append(vector_value)
        elif vector_value is None:
            oracle_values.append(base_value)
        else:
            base_error = normalized_error(row, base_value)
            vector_error = normalized_error(row, vector_value)
            oracle_values.append(vector_value if vector_error < base_error else base_value)

    predictions = {
        "base_mask": base_values,
        "vector": vector_values,
        "hard_fallback": hard_values,
        "quality_router": quality_values,
        "vdn": reference_values,
        "oracle": oracle_values,
    }
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    success: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name], success[name] = _metrics(rows, values)
    quality_switch = np.asarray([route == "vector_quality_switch" for route in routes])
    positive = quality_switch & (errors["vector"] < errors["base_mask"])
    negative = quality_switch & (errors["vector"] > errors["base_mask"])

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    for path in (output_predictions, output_summary):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(identifiers):
            payload = {
                "sample_id": sample_id,
                "group_id": rows[index].get("group_id"),
                "dataset": rows[index].get("dataset"),
                "split": rows[index].get("split"),
                "ground_truth": rows[index].get("ground_truth"),
                "scale_start": rows[index].get("scale_start"),
                "scale_end": rows[index].get("scale_end"),
                "condition": args.condition,
                "router_score": finite_float(scores[index]),
                "router_threshold": threshold,
                "route": routes[index],
                "prediction": quality_values[index],
                "base_prediction": base_values[index],
                "vector_prediction": vector_values[index],
                "vdn_prediction": reference_values[index],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "condition": args.condition,
        "samples": len(rows),
        "groups": int(len(np.unique(groups))),
        "router": str(args.router),
        "router_sha256": sha256_file(args.router),
        "router_training_oof_pairs_sha256": artifact.get("training_oof_pairs_sha256"),
        "router_threshold": threshold,
        "inputs": {
            "raw": str(args.raw_predictions),
            "raw_sha256": sha256_file(args.raw_predictions),
            "base": str(args.base_predictions),
            "base_sha256": sha256_file(args.base_predictions),
            "vector": str(args.vector_predictions),
            "vector_sha256": sha256_file(args.vector_predictions),
            "reference_vdn": str(args.reference_predictions),
            "reference_vdn_sha256": sha256_file(args.reference_predictions),
        },
        "metrics": metrics,
        "paired_comparisons": {
            "quality_vs_hard_fallback": _paired_group_bootstrap(
                errors["quality_router"],
                errors["hard_fallback"],
                groups,
                seed=args.seed,
                iterations=args.bootstrap_iterations,
            ),
            "quality_vs_vdn": _paired_group_bootstrap(
                errors["quality_router"],
                errors["vdn"],
                groups,
                seed=args.seed + 1,
                iterations=args.bootstrap_iterations,
            ),
        },
        "routing": {
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(quality_switch)),
            "positive_transfers": int(np.sum(positive)),
            "negative_transfers": int(np.sum(negative)),
            "ties": int(np.sum(quality_switch & (errors["vector"] == errors["base_mask"]))),
        },
        "subgroups": _subgroup_summaries(rows, predictions),
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "router_feature_source_sha256": expected_feature_source,
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
