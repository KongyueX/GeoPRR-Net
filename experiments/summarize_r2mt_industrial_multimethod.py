"""Summarize the matched four-model industrial real-photo evaluation.

The publication summary joins the completed R2MT-Net replay with the raw
Direct-ResNet18, EfficientNet-B0, and MobileNetV3-Large prediction ledgers on
the exact ``(sample_id, condition)`` key.  It reports all six conditions and
the two prespecified aggregates without dropping failed predictions.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np


SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
DATASETS: Final[tuple[str, ...]] = (
    "field_gauge_roi_test_a",
    "field_gauge_roi_test_b",
    "field_gauge_external_roi",
)
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE: Final[frozenset[str]] = frozenset(CONDITIONS[3:])
METHODS: Final[tuple[str, ...]] = (
    "R²MT-Net",
    "Direct-ResNet18",
    "EfficientNet-B0",
    "MobileNetV3-Large",
)
FAILURE_ERROR: Final[float] = 1.0


Key = tuple[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _prediction_error(
    *, prediction: Any, status: Any, target: float
) -> tuple[float, bool]:
    if status != "pass" or prediction is None:
        return FAILURE_ERROR, False
    value = float(prediction)
    if not np.isfinite(value):
        return FAILURE_ERROR, False
    return abs(value - target), True


def _load_r2mt_seed(
    path: Path,
) -> tuple[dict[Key, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = _read_json(path)
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete R²MT-Net replay: {path}")

    records: dict[Key, dict[str, Any]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}
    for dataset_name in DATASETS:
        dataset = payload["datasets"][dataset_name]
        dataset_meta[dataset_name] = {
            "samples": int(dataset["dataset"]["samples"]),
            "groups": int(dataset["dataset"]["groups"]),
            "group_unit": str(dataset["dataset"]["group_unit"]),
        }
        for row in dataset["per_sample_condition"]:
            key = (str(row["sample_id"]), str(row["condition"]))
            if key in records:
                raise ValueError(f"Duplicate industrial row: {key}")
            target = float(row["normalized_target"])
            candidate = row["candidate"]["mett"]
            error, passed = _prediction_error(
                prediction=candidate.get("prediction"),
                status=candidate.get("status"),
                target=target,
            )
            records[key] = {
                "dataset": dataset_name,
                "group_id": str(row["group_id"]),
                "target": target,
                "error": error,
                "passed": passed,
            }
    return records, dataset_meta


def _load_prediction_ledger(
    path: Path,
    *,
    reference: dict[Key, dict[str, Any]],
) -> tuple[dict[Key, dict[str, Any]], dict[Key, str]]:
    records: dict[Key, dict[str, Any]] = {}
    pixel_hashes: dict[Key, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), str(row["condition"]))
            if key in records:
                raise ValueError(f"Duplicate row in {path}:{line_number}: {key}")
            if key not in reference:
                raise ValueError(f"Unmatched row in {path}:{line_number}: {key}")
            target = float(reference[key]["target"])
            error, passed = _prediction_error(
                prediction=row.get("normalized_progress"),
                status=row.get("status"),
                target=target,
            )
            records[key] = {"error": error, "passed": passed}
            pixel_hashes[key] = str(row["condition_pixel_sha256"])
    return records, pixel_hashes


def _metric_conditions() -> dict[str, frozenset[str]]:
    return {
        **{condition: frozenset((condition,)) for condition in CONDITIONS},
        "all_conditions": frozenset(CONDITIONS),
        "projective_pooled": PROJECTIVE,
    }


def _mean_sample_sd(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "per_seed_nmae": [float(value) for value in array],
        "mean_nmae": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def _paired_group_effect(
    *,
    r2mt_errors: dict[Key, float],
    comparator_errors: dict[Key, float],
    group_by_key: dict[Key, str],
    selected_conditions: frozenset[str],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    row_differences: list[float] = []
    for key in sorted(r2mt_errors):
        if key[1] not in selected_conditions:
            continue
        difference = comparator_errors[key] - r2mt_errors[key]
        group = group_by_key[key]
        sums[group] = sums.get(group, 0.0) + difference
        counts[group] = counts.get(group, 0) + 1
        row_differences.append(difference)

    groups = sorted(sums)
    group_sums = np.asarray([sums[group] for group in groups], dtype=np.float64)
    group_counts = np.asarray([counts[group] for group in groups], dtype=np.float64)
    draws = rng.integers(
        0,
        len(groups),
        size=(int(bootstrap_replicates), len(groups)),
    )
    bootstrap = group_sums[draws].sum(axis=1) / group_counts[draws].sum(axis=1)
    estimate = float(group_sums.sum() / group_counts.sum())
    differences = np.asarray(row_differences, dtype=np.float64)
    return {
        "nmae_reduction": estimate,
        "nmae_reduction_percentage_points": 100.0 * estimate,
        "ci95_group_cluster": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "ci95_group_cluster_percentage_points": [
            100.0 * float(np.quantile(bootstrap, 0.025)),
            100.0 * float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
        "paired_row_improvement_fraction": float(np.mean(differences > 0.0)),
        "groups": len(groups),
        "paired_rows": int(differences.size),
    }


def summarize(
    *,
    project_root: Path,
    factorial_root: Path,
    lightweight_root: Path,
    output_path: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    r2mt_by_seed: dict[int, dict[Key, dict[str, Any]]] = {}
    methods_by_seed: dict[int, dict[str, dict[Key, dict[str, Any]]]] = {}
    dataset_meta: dict[str, dict[str, Any]] | None = None

    ledger_roots = {
        "Direct-ResNet18": (factorial_root, "resnet18_direct"),
        "EfficientNet-B0": (lightweight_root, "efficientnet_b0"),
        "MobileNetV3-Large": (lightweight_root, "mobilenet_v3_large"),
    }

    for seed in SEEDS:
        r2mt_path = (
            project_root
            / "artifacts"
            / "runs"
            / "r2mt_downstream_repro_20260825"
            / "real"
            / f"seed_{seed}.json"
        )
        r2mt_records, current_meta = _load_r2mt_seed(r2mt_path)
        if dataset_meta is None:
            dataset_meta = current_meta
        elif current_meta != dataset_meta:
            raise ValueError("Industrial dataset metadata changed across seeds")
        r2mt_by_seed[seed] = r2mt_records
        methods_by_seed[seed] = {"R²MT-Net": r2mt_records}

        reference_hashes: dict[Key, str] | None = None
        for method, (root, machine_key) in ledger_roots.items():
            merged: dict[Key, dict[str, Any]] = {}
            merged_hashes: dict[Key, str] = {}
            for dataset_name in DATASETS:
                path = (
                    root
                    / "evaluation"
                    / "raw"
                    / dataset_name
                    / machine_key
                    / f"seed_{seed}"
                    / "predictions.jsonl"
                )
                subset_reference = {
                    key: row
                    for key, row in r2mt_records.items()
                    if row["dataset"] == dataset_name
                }
                records, pixel_hashes = _load_prediction_ledger(
                    path,
                    reference=subset_reference,
                )
                if records.keys() != subset_reference.keys():
                    raise ValueError(f"Missing paired rows in {path}")
                merged.update(records)
                merged_hashes.update(pixel_hashes)
            if merged.keys() != r2mt_records.keys():
                raise ValueError(f"Industrial key mismatch for {method}, seed {seed}")
            if reference_hashes is None:
                reference_hashes = merged_hashes
            elif merged_hashes != reference_hashes:
                raise ValueError(
                    f"Condition pixels differ across baselines for seed {seed}"
                )
            methods_by_seed[seed][method] = merged

    assert dataset_meta is not None
    reference_keys = set(r2mt_by_seed[SEEDS[0]])
    for seed in SEEDS[1:]:
        if set(r2mt_by_seed[seed]) != reference_keys:
            raise ValueError("Industrial R²MT-Net keys changed across seeds")
        for key in reference_keys:
            reference = r2mt_by_seed[SEEDS[0]][key]
            current = r2mt_by_seed[seed][key]
            if (
                reference["dataset"] != current["dataset"]
                or reference["group_id"] != current["group_id"]
                or abs(reference["target"] - current["target"]) > 1.0e-12
            ):
                raise ValueError(f"Industrial target metadata changed for {key}")

    selections = _metric_conditions()
    metrics: dict[str, Any] = {}
    for metric, selected_conditions in selections.items():
        methods: dict[str, Any] = {}
        for method in METHODS:
            seed_nmae: list[float] = []
            seed_coverage: list[float] = []
            for seed in SEEDS:
                records = methods_by_seed[seed][method]
                selected = [
                    row
                    for key, row in records.items()
                    if key[1] in selected_conditions
                ]
                seed_nmae.append(float(np.mean([row["error"] for row in selected])))
                seed_coverage.append(float(np.mean([row["passed"] for row in selected])))
            methods[method] = {
                **_mean_sample_sd(seed_nmae),
                "per_seed_coverage": seed_coverage,
                "mean_coverage": float(np.mean(seed_coverage)),
            }
        rows_per_seed = sum(
            1 for key in reference_keys if key[1] in selected_conditions
        )
        r2mt_mean = float(methods["R²MT-Net"]["mean_nmae"])
        for method in METHODS[1:]:
            baseline_mean = float(methods[method]["mean_nmae"])
            methods[method]["r2mt_relative_reduction_percent"] = (
                100.0 * (baseline_mean - r2mt_mean) / baseline_mean
            )
        methods["R²MT-Net"]["r2mt_relative_reduction_percent"] = 0.0
        metrics[metric] = {
            "conditions": sorted(selected_conditions),
            "rows_per_seed": rows_per_seed,
            "methods": methods,
        }

    mean_errors: dict[str, dict[Key, float]] = {}
    for method in METHODS:
        mean_errors[method] = {
            key: float(
                np.mean(
                    [methods_by_seed[seed][method][key]["error"] for seed in SEEDS]
                )
            )
            for key in reference_keys
        }
    group_by_key = {
        key: str(r2mt_by_seed[SEEDS[0]][key]["group_id"])
        for key in reference_keys
    }
    rng = np.random.default_rng(20880825)
    paired_effects = {
        comparator: {
            metric: _paired_group_effect(
                r2mt_errors=mean_errors["R²MT-Net"],
                comparator_errors=mean_errors[comparator],
                group_by_key=group_by_key,
                selected_conditions=selected_conditions,
                bootstrap_replicates=int(bootstrap_replicates),
                rng=rng,
            )
            for metric, selected_conditions in selections.items()
        }
        for comparator in METHODS[1:]
    }

    result = {
        "schema_version": 1,
        "protocol": "r2mt_industrial_multimethod_summary_v1",
        "status": "complete",
        "publication_model": "R²MT-Net",
        "cohort": {
            "datasets": list(DATASETS),
            "samples": len({key[0] for key in reference_keys}),
            "condition_rows_per_seed": len(reference_keys),
            "physical_groups": len(set(group_by_key.values())),
            "dataset_metadata": dataset_meta,
            "same_sample_condition_keys": True,
            "same_condition_pixels_across_baselines": True,
        },
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "metric": {
            "name": "full-denominator normalized mean absolute error",
            "unit": "fraction of full scale",
            "failure_error": FAILURE_ERROR,
            "center": "mean across three independently fitted checkpoints",
            "spread": "sample standard deviation across checkpoints",
        },
        "bootstrap": {
            "unit": "complete physical instrument group",
            "replicates": int(bootstrap_replicates),
            "interval": "percentile 95% confidence interval",
            "estimand": (
                "comparator mean absolute error minus R²MT-Net mean absolute "
                "error after averaging row errors across the three fits"
            ),
        },
        "metrics": metrics,
        "paired_effects": paired_effects,
    }
    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    local_root = Path(r"C:\pointer_read\paper_syncg_only_retrain_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--factorial-root",
        type=Path,
        default=local_root / "factorial",
    )
    parser.add_argument(
        "--lightweight-root",
        type=Path,
        default=local_root / "lightweight_baselines",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "results" / "r2mt_industrial_multimethod.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = summarize(
        project_root=args.project_root.resolve(),
        factorial_root=args.factorial_root.resolve(),
        lightweight_root=args.lightweight_root.resolve(),
        output_path=args.output,
        bootstrap_replicates=int(args.bootstrap_replicates),
    )
    compact = {
        metric: {
            method: round(
                100.0 * float(payload["methods"][method]["mean_nmae"]), 4
            )
            for method in METHODS
        }
        for metric, payload in result["metrics"].items()
    }
    print(json.dumps(compact, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
