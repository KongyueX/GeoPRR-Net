"""Apply the frozen train-only uncertainty model to one evaluation condition."""
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

import numpy as np
import torch

from experiments.quality_router import finite_float, normalized_error, rows_by_id
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    UNCERTAINTY_FUSION_PROTOCOL,
    DualVarianceMLP,
    extract_uncertainty_features,
    feature_matrix,
    soft_fusion_prediction,
    transform_feature_matrix,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


EVALUATION_PROTOCOL = "frozen_uncertainty_soft_fusion_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vector-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument(
        "--quality-predictions",
        type=Path,
        help="Optional frozen v1 quality-router predictions for direct comparison.",
    )
    parser.add_argument(
        "--fusion-model",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/model/uncertainty_fusion.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cpu")
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


def _conditions(row: Mapping[str, Any]) -> set[str]:
    metadata = row.get("metadata") or {}
    return {
        str(value).strip().lower()
        for value in (metadata.get("environment_conditions") or [])
    }


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
                    subgroup_rows,
                    [values[index] for index in indices],
                )[0]
                for method, values in predictions.items()
            },
        }
    return result


def _expert_calibration(
    errors: np.ndarray,
    log_variances: np.ndarray,
    successful: np.ndarray,
) -> dict[str, float | int]:
    valid = successful & np.isfinite(log_variances)
    if not np.any(valid):
        return {"samples": 0}
    observed = errors[valid]
    logvar = np.clip(log_variances[valid], -12.0, 2.0)
    std = np.exp(0.5 * logvar)
    variance = np.exp(logvar)
    correlation = (
        float(np.corrcoef(std, observed)[0, 1])
        if len(observed) > 1 and np.std(std) > 1e-12 and np.std(observed) > 1e-12
        else 0.0
    )
    return {
        "samples": int(len(observed)),
        "mean_predicted_std_normalized": float(np.mean(std)),
        "mean_absolute_error_normalized": float(np.mean(observed)),
        "gaussian_nll": float(
            np.mean(0.5 * (np.square(observed) / variance + logvar))
        ),
        "coverage_1sigma": float(np.mean(observed <= std)),
        "coverage_2sigma": float(np.mean(observed <= 2.0 * std)),
        "uncertainty_error_correlation": correlation,
    }


def _validate_vector_evaluation(path: Path, condition: str) -> dict[str, Any]:
    metadata_path = path.with_name(path.name + ".meta.json")
    summary_path = path.with_name(path.stem + ".summary.json")
    for required in (metadata_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("protocol") != "probabilistic_pivot_direction_e2e_v1":
        raise ValueError("vector evaluation metadata has the wrong protocol")
    if summary.get("status") != "complete" or summary.get("output_sha256") != sha256_file(path):
        raise ValueError("vector evaluation summary is incomplete or stale")
    if str(summary.get("condition")) != condition and not (
        condition == "rpm10k" and str(summary.get("condition")) == "clean"
    ):
        raise ValueError("vector evaluation condition mismatch")
    return {
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_sha256": signature.get("checkpoint_sha256"),
    }


def main() -> None:
    args = parse_args()
    for name in (
        "raw_predictions",
        "base_predictions",
        "vector_predictions",
        "reference_predictions",
        "fusion_model",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.quality_predictions is not None:
        args.quality_predictions = args.quality_predictions.resolve()
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")
    required = [
        args.raw_predictions,
        args.base_predictions,
        args.vector_predictions,
        args.reference_predictions,
        args.fusion_model,
    ]
    if args.quality_predictions is not None:
        required.append(args.quality_predictions)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    artifact = torch.load(args.fusion_model, map_location="cpu", weights_only=False)
    if artifact.get("protocol") != UNCERTAINTY_FUSION_PROTOCOL:
        raise ValueError("fusion artifact has the wrong protocol")
    if artifact.get("train_only_certified") is not True:
        raise ValueError("fusion artifact does not certify train-only fitting")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("fusion feature schema changed")
    feature_source = PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
    if artifact.get("feature_source_sha256") != sha256_file(feature_source):
        raise ValueError("fusion feature/policy source changed after fitting")
    model = DualVarianceMLP(
        input_features=len(FEATURE_NAMES),
        hidden_features=int(artifact["hidden_features"]),
    )
    model.load_state_dict(artifact["model_state"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    medians = np.asarray(artifact["preprocessor_medians"], dtype=np.float32)
    scales = np.asarray(artifact["preprocessor_scales"], dtype=np.float32)
    temperature = float(artifact["temperature"])
    vector_audit = _validate_vector_evaluation(args.vector_predictions, args.condition)

    raw = rows_by_id(args.raw_predictions)
    base = rows_by_id(args.base_predictions)
    vector = rows_by_id(args.vector_predictions)
    reference = rows_by_id(args.reference_predictions)
    quality = rows_by_id(args.quality_predictions) if args.quality_predictions else None
    identifiers = list(base)
    expected = set(identifiers)
    for name, mapping in (("raw", raw), ("vector", vector), ("reference", reference)):
        if set(mapping) != expected:
            raise ValueError(
                f"{name} IDs differ from base: missing={len(expected-set(mapping))}, "
                f"extra={len(set(mapping)-expected)}"
            )
    if quality is not None and set(quality) != expected:
        raise ValueError("quality-router comparison IDs differ from base")

    rows = [base[sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")) for row in rows],
        dtype=object,
    )
    feature_rows = [
        extract_uncertainty_features(
            raw_row=raw[sample_id],
            base_row=base[sample_id],
            vector_row=vector[sample_id],
            reference_row=reference[sample_id],
        )
        for sample_id in identifiers
    ]
    matrix = transform_feature_matrix(
        feature_matrix(feature_rows),
        medians=medians,
        scales=scales,
    )
    log_variance_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(matrix), 2048):
            values = torch.from_numpy(matrix[start : start + 2048]).to(device)
            log_variance_parts.append(model(values).cpu().numpy())
    log_variance = np.concatenate(log_variance_parts, axis=0)

    base_values = [_base_prediction(base[sample_id]) for sample_id in identifiers]
    vector_values = [_vector_prediction(vector[sample_id]) for sample_id in identifiers]
    vdn_values = [_flat_prediction(reference[sample_id]) for sample_id in identifiers]
    quality_values = (
        [finite_float(quality[sample_id].get("prediction")) for sample_id in identifiers]
        if quality is not None
        else None
    )
    fusion_values: list[float | None] = []
    hard_values: list[float | None] = []
    oracle_values: list[float | None] = []
    routes: list[str] = []
    mask_weights: list[float | None] = []
    effective_log_variances: list[float | None] = []
    for index, (row, base_value, vector_value) in enumerate(
        zip(rows, base_values, vector_values)
    ):
        prediction, route, mask_weight, effective = soft_fusion_prediction(
            base_prediction=base_value,
            vector_prediction=vector_value,
            scale_start=row.get("scale_start"),
            scale_end=row.get("scale_end"),
            mask_log_variance=log_variance[index, 0],
            vector_log_variance=log_variance[index, 1],
            temperature=temperature,
        )
        fusion_values.append(prediction)
        routes.append(route)
        mask_weights.append(mask_weight)
        effective_log_variances.append(effective)
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
        "uncertainty_fusion": fusion_values,
        "vdn": vdn_values,
        "oracle": oracle_values,
    }
    if quality_values is not None:
        predictions["quality_router_v1"] = quality_values
    metrics: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    successful: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        metrics[name], errors[name], successful[name] = _metrics(rows, values)

    output_predictions = args.output_dir / "predictions.jsonl"
    output_summary = args.output_dir / "summary.json"
    if any(path.exists() for path in (output_predictions, output_summary)) and not args.overwrite:
        raise FileExistsError("fusion evaluation exists; pass --overwrite")
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
                        "prediction": fusion_values[index],
                        "route": routes[index],
                        "mask_weight": mask_weights[index],
                        "mask_log_variance": finite_float(log_variance[index, 0]),
                        "vector_log_variance": finite_float(log_variance[index, 1]),
                        "effective_log_variance": effective_log_variances[index],
                        "base_prediction": base_values[index],
                        "vector_prediction": vector_values[index],
                        "vdn_prediction": vdn_values[index],
                        "quality_router_v1_prediction": (
                            quality_values[index] if quality_values is not None else None
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    comparisons = {
        "fusion_vs_base_mask": _paired_group_bootstrap(
            errors["uncertainty_fusion"],
            errors["base_mask"],
            groups,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ),
        "fusion_vs_hard_fallback": _paired_group_bootstrap(
            errors["uncertainty_fusion"],
            errors["hard_fallback"],
            groups,
            seed=args.seed + 1,
            iterations=args.bootstrap_iterations,
        ),
        "fusion_vs_vdn": _paired_group_bootstrap(
            errors["uncertainty_fusion"],
            errors["vdn"],
            groups,
            seed=args.seed + 2,
            iterations=args.bootstrap_iterations,
        ),
    }
    if quality_values is not None:
        comparisons["fusion_vs_quality_router_v1"] = _paired_group_bootstrap(
            errors["uncertainty_fusion"],
            errors["quality_router_v1"],
            groups,
            seed=args.seed + 3,
            iterations=args.bootstrap_iterations,
        )
    summary = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "condition": args.condition,
        "samples": len(rows),
        "groups": int(len(np.unique(groups))),
        "fusion_model": str(args.fusion_model),
        "fusion_model_sha256": sha256_file(args.fusion_model),
        "fusion_training_oof_pairs_sha256": artifact.get("training_oof_pairs_sha256"),
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "routing": {
            "counts": dict(sorted(Counter(routes).items())),
            "mean_mask_weight": float(
                np.mean([value for value in mask_weights if value is not None])
            ),
            "median_mask_weight": float(
                np.median([value for value in mask_weights if value is not None])
            ),
        },
        "calibration": {
            "mask": _expert_calibration(
                errors["base_mask"], log_variance[:, 0], successful["base_mask"]
            ),
            "vector": _expert_calibration(
                errors["probabilistic_vector"],
                log_variance[:, 1],
                successful["probabilistic_vector"],
            ),
        },
        "subgroups": _subgroup_summaries(rows, predictions),
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
            "quality_router_v1": (
                str(args.quality_predictions) if args.quality_predictions else None
            ),
            "quality_router_v1_sha256": (
                sha256_file(args.quality_predictions) if args.quality_predictions else None
            ),
        },
        "predictions": str(output_predictions),
        "predictions_sha256": sha256_file(output_predictions),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "feature_source_sha256": sha256_file(feature_source),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    _atomic_json(output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(output_summary)


if __name__ == "__main__":
    main()
