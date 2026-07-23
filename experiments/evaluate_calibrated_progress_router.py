"""Evaluate the frozen mask/calibrated-vector selective router."""
from __future__ import annotations

import argparse
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from experiments.calibrated_progress_router import (
    CALIBRATED_PROGRESS_ROUTER_PROTOCOL,
    FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix,
)
from experiments.evaluate_progress_calibrator import (
    EVALUATION_PROTOCOL as CALIBRATOR_EVALUATION_PROTOCOL,
    _base_prediction,
    _flat_prediction,
    _metrics,
    _paired_group_bootstrap,
)
from experiments.quality_router import (
    finite_float,
    normalized_error,
    route_prediction,
    rows_by_id,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


EVALUATION_PROTOCOL = "frozen_calibrated_progress_router_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--calibrated-predictions", type=Path, required=True)
    parser.add_argument("--quality-predictions", type=Path)
    parser.add_argument("--uncertainty-router-predictions", type=Path)
    parser.add_argument(
        "--router",
        type=Path,
        default=Path(
            "artifacts/runs/calibrated_progress_router_syncg/model/"
            "calibrated_progress_router.joblib"
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


def _validate_calibration(path: Path, condition: str) -> dict[str, Any]:
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != CALIBRATOR_EVALUATION_PROTOCOL:
        raise ValueError("calibrated prediction summary has the wrong protocol")
    if summary.get("status") != "complete":
        raise ValueError("calibrated prediction summary is incomplete")
    if summary.get("predictions_sha256") != sha256_file(path):
        raise ValueError("calibrated prediction file hash mismatch")
    if str(summary.get("condition")) != condition:
        raise ValueError("calibrated prediction condition mismatch")
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "calibrator_sha256": summary.get("calibrator_sha256"),
    }


def main() -> None:
    args = parse_args()
    path_names = (
        "raw_predictions",
        "base_predictions",
        "vector_predictions",
        "reference_predictions",
        "calibrated_predictions",
        "router",
        "output_dir",
    )
    for name in path_names:
        setattr(args, name, getattr(args, name).resolve())
    for name in ("quality_predictions", "uncertainty_router_predictions"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    required = [
        args.raw_predictions,
        args.base_predictions,
        args.vector_predictions,
        args.reference_predictions,
        args.calibrated_predictions,
        args.router,
    ]
    required.extend(
        value
        for value in (args.quality_predictions, args.uncertainty_router_predictions)
        if value is not None
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")

    artifact = joblib.load(args.router)
    if artifact.get("protocol") != CALIBRATED_PROGRESS_ROUTER_PROTOCOL:
        raise ValueError("router artifact has the wrong protocol")
    if artifact.get("train_only_certified") is not True:
        raise ValueError("router artifact does not certify train-only fitting")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("router feature schema changed")
    if artifact.get("test_sets_used") != []:
        raise ValueError("router artifact lists test-set use")
    sources = artifact.get("source_sha256") or {}
    source_paths = {
        "features": PROJECT_DIR / "experiments" / "calibrated_progress_router.py",
        "quality_features": PROJECT_DIR / "experiments" / "uncertainty_fusion.py",
        "policy": PROJECT_DIR / "experiments" / "quality_router.py",
    }
    for name, path in source_paths.items():
        if sources.get(name) != sha256_file(path):
            raise ValueError(f"router {name} source changed after fitting")
    calibration_audit = _validate_calibration(
        args.calibrated_predictions, args.condition
    )

    mappings = {
        "raw": rows_by_id(args.raw_predictions),
        "base": rows_by_id(args.base_predictions),
        "vector": rows_by_id(args.vector_predictions),
        "reference": rows_by_id(args.reference_predictions),
        "calibrated": rows_by_id(args.calibrated_predictions),
    }
    if args.quality_predictions is not None:
        mappings["quality_router_v1"] = rows_by_id(args.quality_predictions)
    if args.uncertainty_router_predictions is not None:
        mappings["uncertainty_router_v2"] = rows_by_id(
            args.uncertainty_router_predictions
        )
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
    calibrated_values = [
        _flat_prediction(mappings["calibrated"][key]) for key in identifiers
    ]
    raw_vector_values = [
        _flat_prediction(mappings["vector"][key]) for key in identifiers
    ]
    vdn_values = [_flat_prediction(mappings["reference"][key]) for key in identifiers]
    feature_rows = [
        extract_calibrated_router_features(
            raw_row=mappings["raw"][key],
            base_row=mappings["base"][key],
            vector_row=mappings["vector"][key],
            reference_row=mappings["reference"][key],
            calibration_row=mappings["calibrated"][key],
        )
        for key in identifiers
    ]
    matrix = feature_matrix(feature_rows)
    joint = np.asarray(
        [first is not None and second is not None for first, second in zip(base_values, calibrated_values)]
    )
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    if np.any(joint):
        scores[joint] = artifact["estimator"].predict(matrix[joint])
    threshold = float(artifact["threshold"])
    routed_values: list[float | None] = []
    routes: list[str] = []
    hard_values: list[float | None] = []
    oracle_values: list[float | None] = []
    for row, base, calibrated, score in zip(
        rows, base_values, calibrated_values, scores
    ):
        prediction, route = route_prediction(
            base_prediction=base,
            vector_prediction=calibrated,
            score=score,
            threshold=threshold,
        )
        route = {
            "vector_quality_switch": "calibrated_quality_switch",
            "vector_hard_fallback": "calibrated_hard_fallback",
        }.get(route, route)
        routed_values.append(prediction)
        routes.append(route)
        hard_values.append(base if base is not None else calibrated)
        if base is None:
            oracle_values.append(calibrated)
        elif calibrated is None:
            oracle_values.append(base)
        else:
            oracle_values.append(
                calibrated
                if normalized_error(row, calibrated) < normalized_error(row, base)
                else base
            )

    predictions: dict[str, Sequence[float | None]] = {
        "base_mask": base_values,
        "raw_probabilistic_vector": raw_vector_values,
        "calibrated_vector": calibrated_values,
        "hard_fallback": hard_values,
        "calibrated_progress_router": routed_values,
        "vdn": vdn_values,
        "oracle": oracle_values,
    }
    if "quality_router_v1" in mappings:
        predictions["quality_router_v1"] = [
            _flat_prediction(mappings["quality_router_v1"][key])
            for key in identifiers
        ]
    if "uncertainty_router_v2" in mappings:
        predictions["uncertainty_router_v2"] = [
            _flat_prediction(mappings["uncertainty_router_v2"][key])
            for key in identifiers
        ]
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    successful: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name], successful[name] = _metrics(rows, values)
    switched = np.asarray([route == "calibrated_quality_switch" for route in routes])
    positive = switched & (errors["calibrated_vector"] < errors["base_mask"])
    negative = switched & (errors["calibrated_vector"] > errors["base_mask"])

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    if any(path.exists() for path in (output_predictions, output_summary)) and not args.overwrite:
        raise FileExistsError("calibrated-router evaluation exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as handle:
        for index, sample_id in enumerate(identifiers):
            calibration_row = mappings["calibrated"][sample_id]
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
                        "status": routed_values[index] is not None,
                        "prediction": routed_values[index],
                        "route": routes[index],
                        "router_score": finite_float(scores[index]),
                        "router_threshold": threshold,
                        "base_prediction": base_values[index],
                        "calibrated_prediction": calibrated_values[index],
                        "raw_vector_prediction": raw_vector_values[index],
                        "predicted_progress_residual": calibration_row.get(
                            "predicted_progress_residual"
                        ),
                        "progress_ensemble_std": calibration_row.get(
                            "progress_ensemble_std"
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    comparisons = {
        "router_vs_base_mask": _paired_group_bootstrap(
            errors["calibrated_progress_router"], errors["base_mask"], groups,
            seed=args.seed, iterations=args.bootstrap_iterations,
        ),
        "router_vs_calibrated_vector": _paired_group_bootstrap(
            errors["calibrated_progress_router"], errors["calibrated_vector"], groups,
            seed=args.seed + 1, iterations=args.bootstrap_iterations,
        ),
        "router_vs_vdn": _paired_group_bootstrap(
            errors["calibrated_progress_router"], errors["vdn"], groups,
            seed=args.seed + 2, iterations=args.bootstrap_iterations,
        ),
    }
    for offset, name in enumerate(("quality_router_v1", "uncertainty_router_v2"), start=3):
        if name in errors:
            comparisons[f"router_vs_{name}"] = _paired_group_bootstrap(
                errors["calibrated_progress_router"], errors[name], groups,
                seed=args.seed + offset, iterations=args.bootstrap_iterations,
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
        "router_threshold": threshold,
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "routing": {
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(np.sum(positive)),
            "negative_transfers": int(np.sum(negative)),
            "ties": int(np.sum(switched & (errors["calibrated_vector"] == errors["base_mask"]))),
        },
        "inputs": {
            "raw": str(args.raw_predictions),
            "raw_sha256": sha256_file(args.raw_predictions),
            "base": str(args.base_predictions),
            "base_sha256": sha256_file(args.base_predictions),
            "vector": str(args.vector_predictions),
            "vector_sha256": sha256_file(args.vector_predictions),
            "reference_vdn": str(args.reference_predictions),
            "reference_vdn_sha256": sha256_file(args.reference_predictions),
            "calibrated": str(args.calibrated_predictions),
            "calibrated_sha256": sha256_file(args.calibrated_predictions),
            "calibration_audit": calibration_audit,
            "quality_router_v1": str(args.quality_predictions) if args.quality_predictions else None,
            "uncertainty_router_v2": (
                str(args.uncertainty_router_predictions)
                if args.uncertainty_router_predictions else None
            ),
        },
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "feature_source_sha256": sha256_file(source_paths["features"]),
        "quality_feature_source_sha256": sha256_file(source_paths["quality_features"]),
        "policy_source_sha256": sha256_file(source_paths["policy"]),
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
