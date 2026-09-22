"""Summarize matched revision predictions and inspect saved training metadata."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (20262020, 20262021, 20262022)
CONDITIONS = (
    "clean", "blur_moderate", "blur_severe", "perspective_moderate",
    "perspective_severe", "combined_severe",
)
FIELDS = {"method", "seed", "sample_id", "group_id", "condition",
          "target", "prediction", "error", "status"}


def load_predictions(paths: list[Path]) -> dict[str, Any]:
    cells: dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]] = {}
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not FIELDS.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: missing columns {sorted(FIELDS - set(reader.fieldnames or ())) }")
            for row in reader:
                method, seed = row["method"], int(row["seed"])
                key = (row["sample_id"], row["condition"])
                if not method or not key[0] or not row["group_id"]:
                    raise ValueError("method, sample_id and group_id must be nonempty")
                if seed not in SEEDS or key[1] not in CONDITIONS:
                    raise ValueError(f"unsupported seed or condition: {seed}, {key[1]}")
                if row["status"] not in {"success", "failure"}:
                    raise ValueError(f"unsupported status: {row['status']}")
                target, error = float(row["target"]), float(row["error"])
                if not np.isfinite([target, error]).all() or not (0 <= target <= 1 and 0 <= error <= 1):
                    raise ValueError("targets and errors must be finite normalized values")
                valid = row["status"] == "success"
                prediction = float(row["prediction"]) if valid else np.nan
                if valid and (not np.isfinite(prediction) or not 0 <= prediction <= 1
                              or abs(abs(prediction - target) - error) > 1e-6):
                    raise ValueError("successful prediction and supplied error disagree")
                if not valid and error != 1.0:
                    raise ValueError("failed predictions must retain error 1.0")
                cell = cells.setdefault((method, seed), {})
                if key in cell:
                    raise ValueError(f"duplicate prediction key: {method}, {seed}, {key}")
                cell[key] = {"group": row["group_id"], "target": target,
                             "prediction": prediction, "error": error, "valid": valid}
    if not cells:
        raise ValueError("prediction input is empty")
    methods = sorted({method for method, _seed in cells})
    if set(cells) != {(method, seed) for method in methods for seed in SEEDS}:
        raise ValueError("every method must contain all three source seeds")
    reference = cells[(methods[0], SEEDS[0])]
    samples = sorted({sample for sample, _condition in reference})
    keys = [(sample, condition) for sample in samples for condition in CONDITIONS]
    if set(reference) != set(keys):
        raise ValueError("each sample must have all six conditions")
    for sample in samples:
        entries = [reference[(sample, condition)] for condition in CONDITIONS]
        if len({r["group"] for r in entries}) != 1 or len({r["target"] for r in entries}) != 1:
            raise ValueError(f"sample group or target changes across conditions: {sample}")
    predictions, errors, validities = {}, {}, {}
    for method in methods:
        prediction_rows, error_rows, valid_rows = [], [], []
        for seed in SEEDS:
            cell = cells[(method, seed)]
            if set(cell) != set(keys):
                raise ValueError(f"evaluation roster mismatch: {method}, {seed}")
            for key in keys:
                if cell[key]["group"] != reference[key]["group"] or cell[key]["target"] != reference[key]["target"]:
                    raise ValueError(f"group or target mismatch: {method}, {seed}, {key}")
            prediction_rows.append([cell[key]["prediction"] for key in keys])
            error_rows.append([cell[key]["error"] for key in keys])
            valid_rows.append([cell[key]["valid"] for key in keys])
        predictions[method] = np.asarray(prediction_rows)
        errors[method] = np.asarray(error_rows)
        validities[method] = np.asarray(valid_rows, dtype=bool)
    return {"methods": methods, "keys": keys,
            "groups": np.asarray([reference[key]["group"] for key in keys]),
            "targets": np.asarray([reference[key]["target"] for key in keys]),
            "predictions": predictions, "errors": errors, "valid": validities}


def metrics(errors: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    return {"nmae_pct_fs": float(100 * errors.mean()),
            "rmse_pct_fs": float(100 * np.sqrt(np.mean(errors ** 2))),
            "acc_at_1_pct": float(100 * np.mean(valid & (errors <= 0.01))),
            "acc_at_2_pct": float(100 * np.mean(valid & (errors <= 0.02))),
            "acc_at_5_pct": float(100 * np.mean(valid & (errors <= 0.05))),
            "coverage_pct": float(100 * valid.mean())}


def summarize(data: dict[str, Any], *, reference: str = "full",
              replicates: int = 20_000, bootstrap_seed: int = 20261107) -> dict[str, Any]:
    if reference not in data["methods"]:
        raise ValueError(f"reference method is absent: {reference}")
    groups, group_index = np.unique(data["groups"], return_inverse=True)
    if len(groups) < 2 or replicates < 1:
        raise ValueError("group bootstrap requires at least two groups and one replicate")
    rng = np.random.default_rng(bootstrap_seed)
    choices = rng.integers(len(groups), size=(replicates, len(groups)))
    counts = np.zeros_like(choices, dtype=np.int32)
    np.add.at(counts, (np.arange(replicates)[:, None], choices), 1)
    per_seed, summary, paired = [], [], []
    scopes = {"all_conditions": np.ones(len(data["keys"]), dtype=bool)}
    scopes.update({condition: np.asarray([key[1] == condition for key in data["keys"]])
                   for condition in CONDITIONS})
    for scope, selected in scopes.items():
        selected_groups = group_index[selected]
        group_sizes = np.bincount(selected_groups, minlength=len(groups))
        bootstrap_sizes = counts @ group_sizes
        common = {"scope": scope, "images": int(selected.sum() // (6 if scope == "all_conditions" else 1)),
                  "rows": int(selected.sum()), "groups": len(groups), "source_seeds": 3}
        point_arrays: dict[tuple[str, str], np.ndarray] = {}
        for method in data["methods"]:
            errors, valid = data["errors"][method][:, selected], data["valid"][method][:, selected]
            fitted_metrics = []
            for index, seed in enumerate(SEEDS):
                values = metrics(errors[index], valid[index])
                fitted_metrics.append(values)
                per_seed.append({"method": method, "seed": seed, **common, **values})
            means = {key: float(np.mean([value[key] for value in fitted_metrics])) for key in fitted_metrics[0]}
            sds = {key + "_sample_sd": float(np.std([value[key] for value in fitted_metrics], ddof=1))
                   for key in fitted_metrics[0]}
            summary.append({"method": method, "aggregation": "seed_mean", **common,
                            "ensemble_members": 0, **means, **sds})
            point_arrays[(method, "seed_mean")] = np.column_stack([
                errors.mean(axis=0), (valid & (errors <= 0.05)).mean(axis=0)])
            ensemble_valid = valid.all(axis=0)
            ensemble_error = np.ones(selected.sum())
            predictions = data["predictions"][method][:, selected]
            ensemble_error[ensemble_valid] = np.abs(
                predictions[:, ensemble_valid].mean(axis=0) - data["targets"][selected][ensemble_valid])
            ensemble_metrics = metrics(ensemble_error, ensemble_valid)
            summary.append({"method": method, "aggregation": "equal_ensemble", **common,
                            "ensemble_members": 3, **ensemble_metrics,
                            **{key: None for key in sds}})
            point_arrays[(method, "equal_ensemble")] = np.column_stack([
                ensemble_error, ensemble_valid & (ensemble_error <= 0.05)])
        for aggregation in ("seed_mean", "equal_ensemble"):
            for method in data["methods"]:
                if method == reference:
                    continue
                delta = point_arrays[(method, aggregation)] - point_arrays[(reference, aggregation)]
                group_sums = np.zeros((len(groups), 2))
                np.add.at(group_sums, selected_groups, delta)
                draws = 100 * (counts @ group_sums) / bootstrap_sizes[:, None]
                low, high = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
                for column, metric in enumerate(("nmae_pct_fs", "acc_at_5_pct")):
                    paired.append({"candidate": method, "reference": reference,
                                   "aggregation": aggregation, "metric": metric, **common,
                                   "candidate_minus_reference": float(100 * delta[:, column].mean()),
                                   "ci95_low": float(low[column]), "ci95_high": float(high[column])})
    return {"protocol": {"source_seeds": list(SEEDS), "conditions": list(CONDITIONS),
                         "reference": reference, "bootstrap_replicates": replicates,
                         "bootstrap_seed": bootstrap_seed, "rng": "numpy.default_rng",
                         "resampling": "same group draw across methods, seeds, conditions and aggregations; retain all constituent rows",
                         "confidence_intervals": "pointwise percentile 95%, conditional on fitted models",
                         "paired_difference": "candidate minus reference; negative NMAE and positive accuracy favor candidate",
                         "failure_policy": "error 1.0 and incorrect at all accuracy thresholds; any failed ensemble member fails that row",
                         "ensemble": "equal average of same-row predictions, then score; never average member errors"},
            "per_seed": per_seed, "summary": summary, "paired": paired}


def routing_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    cache = payload.get("cache", payload)
    result: dict[str, Any] = {key: payload[key] for key in ("scope", "seed", "conditions") if key in payload}
    result["missing_fields"] = [key for key in ("routing_weights", "moment_residual") if key not in cache]
    for key in ("routing_weights", "moment_residual"):
        if key not in cache:
            continue
        value = cache[key]
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if not np.isfinite(array).all() or not array.size:
            raise ValueError(f"invalid {key}")
        if key == "routing_weights":
            if array.ndim != 2 or array.shape[1] != 3 or np.any(array < 0) or not np.allclose(array.sum(axis=1), 1, atol=1e-6, rtol=0):
                raise ValueError("routing weights must be an N by 3 probability simplex")
            array = array.astype(np.float64)
            entropy = -(array * np.log(np.maximum(array, np.finfo(float).tiny))).sum(axis=1) / np.log(3)
            result["routing"] = {"rows": len(array), "normalized_entropy_mean": float(entropy.mean()),
                                 "mean_expert_weights": array.mean(axis=0).tolist(),
                                 "fraction_max_weight_gt_095": float(np.mean(array.max(axis=1) > 0.95))}
        else:
            if array.ndim != 1:
                raise ValueError("moment residual must have shape N")
            result["moment_residual"] = {"rows": len(array), "signed_mean": float(array.mean()),
                                         "absolute_mean": float(np.abs(array).mean()),
                                         "absolute_max": float(np.abs(array).max())}
    if "routing" in result and "moment_residual" in result and result["routing"]["rows"] != result["moment_residual"]["rows"]:
        raise ValueError("routing and moment diagnostic rows disagree")
    if "condition_index" in cache and any(key in cache for key in ("routing_weights", "moment_residual")):
        index = cache["condition_index"]
        index = index.detach().cpu().numpy() if hasattr(index, "detach") else np.asarray(index)
        names = payload.get("conditions", CONDITIONS)
        expected_rows = result.get("routing", result.get("moment_residual"))["rows"]
        if index.shape != (expected_rows,) or not np.isin(index, range(len(names))).all():
            raise ValueError("condition indices disagree with diagnostic rows")
        result["per_condition"] = {}
        for ci, name in enumerate(names):
            selected = index == ci
            if not selected.any():
                continue
            subset = {}
            for key in ("routing_weights", "moment_residual"):
                if key in cache:
                    value = cache[key]
                    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
                    subset[key] = array[selected]
            result["per_condition"][name] = routing_diagnostics(subset)

    return result


def audit_saved_files(checkpoints: list[Path], splits: list[Path]) -> dict[str, Any]:
    def describe(value):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            if value and all(hasattr(item, "shape") for item in value.values()):
                return {"tensor_fields": {str(key): describe(item) for key, item in value.items()}}
            return {str(key): describe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    result = {"checkpoints": [], "split_files": []}
    if checkpoints:
        import torch
        for path in checkpoints:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"checkpoint must contain a mapping: {path}")
            fields = ("seed", "protocol", "construction", "training_scope", "warm_initialization", "checkpoint_selection")
            entry = {"path": str(path), "actual_fields": describe(payload),
                     "not_present_at_top_level": [key for key in fields if key not in payload]}
            if "cache" in payload or "routing_weights" in payload or "moment_residual" in payload:
                entry["diagnostics"] = routing_diagnostics(payload)
            result["checkpoints"].append(entry)
    for path in splits:
        result["split_files"].append({"path": str(path), "actual_fields": json.loads(path.read_text(encoding="utf-8-sig"))})
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--predictions", nargs="+", type=Path, required=True)
    summary.add_argument("--reference", default="full")
    summary.add_argument("--bootstrap-seed", type=int, default=20261107)
    summary.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    audit.add_argument("--splits", nargs="*", type=Path, default=[])
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        result = summarize(load_predictions(args.predictions), reference=args.reference,
                           bootstrap_seed=args.bootstrap_seed)
        write_json(args.output_dir / "summary.json", result)
        for name in ("per_seed", "summary", "paired"):
            records = result[name]
            with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
                fields = list(records[0]) if records else ["candidate", "reference", "scope", "metric"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
    else:
        if not args.checkpoints and not args.splits:
            parser.error("audit needs --checkpoints and/or --splits")
        write_json(args.output, audit_saved_files(args.checkpoints, args.splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
