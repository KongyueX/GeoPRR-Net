"""Evaluate the frozen train-only probabilistic uncertainty router."""
from __future__ import annotations

import argparse
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from experiments.evaluate_uncertainty_fusion import _validate_vector_evaluation
from experiments.quality_router import (
    finite_float,
    normalized_error,
    route_prediction,
    rows_by_id,
)
from experiments.train_uncertainty_router import UNCERTAINTY_ROUTER_PROTOCOL
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    extract_uncertainty_features,
    feature_matrix,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


EVALUATION_PROTOCOL = "frozen_probabilistic_uncertainty_router_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--quality-predictions", type=Path)
    parser.add_argument("--fusion-predictions", type=Path)
    parser.add_argument(
        "--router",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_router_syncg/model/uncertainty_router.joblib"
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


def _vector_prediction(row: Mapping[str, Any]) -> float | None:
    nested = row.get("vector")
    payload = nested if isinstance(nested, Mapping) else row
    return finite_float(payload.get("prediction")) if payload.get("status") is True else None


def _flat_prediction(row: Mapping[str, Any]) -> float | None:
    return finite_float(row.get("prediction")) if row.get("status", True) is not False else None


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[float | None]
) -> tuple[dict[str, float | int], np.ndarray]:
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


def _conditions(row: Mapping[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    return {
        str(value).strip().lower()
        for value in (metadata.get("environment_conditions") or [])
    }


def _subgroups(
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
    conditions = [_conditions(row) for row in rows]
    result: dict[str, Any] = {}
    for name, selector in selectors.items():
        indices = [index for index, values in enumerate(conditions) if selector(values)]
        if indices:
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
    path_names = (
        "raw_predictions",
        "base_predictions",
        "vector_predictions",
        "reference_predictions",
        "router",
        "output_dir",
    )
    for name in path_names:
        setattr(args, name, getattr(args, name).resolve())
    for name in ("quality_predictions", "fusion_predictions"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")
    required = [
        args.raw_predictions,
        args.base_predictions,
        args.vector_predictions,
        args.reference_predictions,
        args.router,
    ]
    required.extend(
        path
        for path in (args.quality_predictions, args.fusion_predictions)
        if path is not None
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    artifact = joblib.load(args.router)
    if artifact.get("protocol") != UNCERTAINTY_ROUTER_PROTOCOL:
        raise ValueError("router artifact has the wrong protocol")
    if artifact.get("train_only_certified") is not True:
        raise ValueError("router artifact does not certify train-only fitting")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("router feature schema changed")
    if artifact.get("test_sets_used") != []:
        raise ValueError("router artifact lists test-set use")
    sources = artifact.get("source_sha256") or {}
    feature_source = PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
    policy_source = PROJECT_DIR / "experiments" / "quality_router.py"
    if sources.get("features") != sha256_file(feature_source):
        raise ValueError("router feature source changed after fitting")
    if sources.get("policy") != sha256_file(policy_source):
        raise ValueError("router routing policy changed after fitting")
    vector_audit = _validate_vector_evaluation(args.vector_predictions, args.condition)

    mappings = {
        "raw": rows_by_id(args.raw_predictions),
        "base": rows_by_id(args.base_predictions),
        "vector": rows_by_id(args.vector_predictions),
        "reference": rows_by_id(args.reference_predictions),
    }
    if args.quality_predictions:
        mappings["quality"] = rows_by_id(args.quality_predictions)
    if args.fusion_predictions:
        mappings["fusion"] = rows_by_id(args.fusion_predictions)
    identifiers = list(mappings["base"])
    expected = set(identifiers)
    for name, mapping in mappings.items():
        if set(mapping) != expected:
            raise ValueError(
                f"{name} IDs differ from base: missing={len(expected-set(mapping))}, "
                f"extra={len(set(mapping)-expected)}"
            )

    rows = [mappings["base"][sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")) for row in rows],
        dtype=object,
    )
    base_values = [_base_prediction(mappings["base"][key]) for key in identifiers]
    vector_values = [_vector_prediction(mappings["vector"][key]) for key in identifiers]
    vdn_values = [_flat_prediction(mappings["reference"][key]) for key in identifiers]
    quality_values = (
        [_flat_prediction(mappings["quality"][key]) for key in identifiers]
        if "quality" in mappings
        else None
    )
    fusion_values = (
        [_flat_prediction(mappings["fusion"][key]) for key in identifiers]
        if "fusion" in mappings
        else None
    )
    features = [
        extract_uncertainty_features(
            raw_row=mappings["raw"][key],
            base_row=mappings["base"][key],
            vector_row=mappings["vector"][key],
            reference_row=mappings["reference"][key],
        )
        for key in identifiers
    ]
    matrix = feature_matrix(features)
    joint = np.asarray(
        [first is not None and second is not None for first, second in zip(base_values, vector_values)]
    )
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    if np.any(joint):
        scores[joint] = artifact["estimator"].predict(matrix[joint])

    router_values: list[float | None] = []
    hard_values: list[float | None] = []
    oracle_values: list[float | None] = []
    routes: list[str] = []
    threshold = float(artifact["threshold"])
    for row, base_value, vector_value, score in zip(
        rows, base_values, vector_values, scores
    ):
        prediction, route = route_prediction(
            base_prediction=base_value,
            vector_prediction=vector_value,
            score=score,
            threshold=threshold,
        )
        router_values.append(prediction)
        routes.append(
            "vector_uncertainty_switch"
            if route == "vector_quality_switch"
            else route
        )
        hard_values.append(base_value if base_value is not None else vector_value)
        if base_value is None:
            oracle_values.append(vector_value)
        elif vector_value is None:
            oracle_values.append(base_value)
        else:
            oracle_values.append(
                vector_value
                if normalized_error(row, vector_value) < normalized_error(row, base_value)
                else base_value
            )

    predictions: dict[str, Sequence[float | None]] = {
        "base_mask": base_values,
        "probabilistic_vector": vector_values,
        "hard_fallback": hard_values,
        "uncertainty_router": router_values,
        "vdn": vdn_values,
        "oracle": oracle_values,
    }
    if quality_values is not None:
        predictions["quality_router_v1"] = quality_values
    if fusion_values is not None:
        predictions["uncertainty_soft_fusion"] = fusion_values
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name] = _metrics(rows, values)
    switched = np.asarray([route == "vector_uncertainty_switch" for route in routes])
    positive = switched & (errors["probabilistic_vector"] < errors["base_mask"])
    negative = switched & (errors["probabilistic_vector"] > errors["base_mask"])

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    if any(path.exists() for path in (output_predictions, output_summary)) and not args.overwrite:
        raise FileExistsError("router evaluation exists; pass --overwrite")
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
                        "prediction": router_values[index],
                        "route": routes[index],
                        "router_score": finite_float(scores[index]),
                        "router_threshold": threshold,
                        "base_prediction": base_values[index],
                        "vector_prediction": vector_values[index],
                        "vdn_prediction": vdn_values[index],
                        "quality_router_v1_prediction": (
                            quality_values[index] if quality_values is not None else None
                        ),
                        "uncertainty_soft_fusion_prediction": (
                            fusion_values[index] if fusion_values is not None else None
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    comparisons = {
        "router_vs_base_mask": _paired_group_bootstrap(
            errors["uncertainty_router"], errors["base_mask"], groups,
            seed=args.seed, iterations=args.bootstrap_iterations,
        ),
        "router_vs_hard_fallback": _paired_group_bootstrap(
            errors["uncertainty_router"], errors["hard_fallback"], groups,
            seed=args.seed + 1, iterations=args.bootstrap_iterations,
        ),
        "router_vs_vdn": _paired_group_bootstrap(
            errors["uncertainty_router"], errors["vdn"], groups,
            seed=args.seed + 2, iterations=args.bootstrap_iterations,
        ),
    }
    if quality_values is not None:
        comparisons["router_vs_quality_router_v1"] = _paired_group_bootstrap(
            errors["uncertainty_router"], errors["quality_router_v1"], groups,
            seed=args.seed + 3, iterations=args.bootstrap_iterations,
        )
    if fusion_values is not None:
        comparisons["router_vs_uncertainty_soft_fusion"] = _paired_group_bootstrap(
            errors["uncertainty_router"], errors["uncertainty_soft_fusion"], groups,
            seed=args.seed + 4, iterations=args.bootstrap_iterations,
        )
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
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "routing": {
            "counts": dict(sorted(Counter(routes).items())),
            "uncertainty_switches": int(np.sum(switched)),
            "positive_transfers": int(np.sum(positive)),
            "negative_transfers": int(np.sum(negative)),
            "ties": int(np.sum(switched & (errors["probabilistic_vector"] == errors["base_mask"]))),
        },
        "subgroups": _subgroups(rows, predictions),
        "inputs": {
            "raw": str(args.raw_predictions),
            "raw_sha256": sha256_file(args.raw_predictions),
            "base": str(args.base_predictions),
            "base_sha256": sha256_file(args.base_predictions),
            "vector": str(args.vector_predictions),
            "vector_sha256": sha256_file(args.vector_predictions),
            "vector_audit": vector_audit,
            "reference_vdn": str(args.reference_predictions),
            "reference_vdn_sha256": sha256_file(args.reference_predictions),
            "quality_router_v1": str(args.quality_predictions) if args.quality_predictions else None,
            "quality_router_v1_sha256": sha256_file(args.quality_predictions) if args.quality_predictions else None,
            "uncertainty_soft_fusion": str(args.fusion_predictions) if args.fusion_predictions else None,
            "uncertainty_soft_fusion_sha256": sha256_file(args.fusion_predictions) if args.fusion_predictions else None,
        },
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "feature_source_sha256": sha256_file(feature_source),
        "policy_source_sha256": sha256_file(policy_source),
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
