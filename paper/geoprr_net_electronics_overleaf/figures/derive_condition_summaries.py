"""Derive compact condition-wise CSVs for the GeoPRR-Net paper figures.

The source ledgers contain every sample, condition, and training seed.  This
script keeps the published six-condition rosters intact, averages paired row
errors across the three fitted seeds, and bootstraps the declared scene/source
groups for GeoPRR-Net-minus-Raw-EfficientNet-B0 effects.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from experiments.evaluate_paper_syncg_only_lightweight_baselines import (
    _paired_group_bootstrap_fast,
)


RUN_ROOT = REPO / "artifacts" / "runs" / "unified_pointer_reader"
REPORT_PATH = REPO / "artifacts" / "reports" / "unified_pointer_reader" / "paper_results.json"
SEEDS = (20262020, 20262021, 20262022)
CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
CONDITION_LABELS = {
    "clean": "Clean",
    "blur_moderate": "Blur moderate",
    "blur_severe": "Blur severe",
    "perspective_moderate": "Perspective moderate",
    "perspective_severe": "Perspective severe",
    "combined_severe": "Perspective severe + blur",
}
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260828


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def distribution(values: Iterable[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers), statistics.stdev(numbers)


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with (HERE / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(
    *,
    condition: str,
    proposed_seed_errors: list[np.ndarray],
    baseline_seed_errors: list[np.ndarray],
    groups: list[str],
    bootstrap_seed: int,
) -> dict[str, Any]:
    proposed_seed_nmae = [float(values.mean()) for values in proposed_seed_errors]
    baseline_seed_nmae = [float(values.mean()) for values in baseline_seed_errors]
    proposed_mean, proposed_sd = distribution(proposed_seed_nmae)
    baseline_mean, baseline_sd = distribution(baseline_seed_nmae)
    paired = _paired_group_bootstrap_fast(
        np.asarray(proposed_seed_errors, dtype=np.float64).mean(axis=0),
        np.asarray(baseline_seed_errors, dtype=np.float64).mean(axis=0),
        groups,
        replicates=BOOTSTRAP_REPLICATES,
        seed=bootstrap_seed,
    )
    interval = paired["paired_group_bootstrap_ci95"]
    return {
        "condition": condition,
        "label": CONDITION_LABELS[condition],
        "geo_nmae_mean": proposed_mean,
        "geo_nmae_sd": proposed_sd,
        "baseline_nmae_mean": baseline_mean,
        "baseline_nmae_sd": baseline_sd,
        "relative_reduction_percent": 100.0 * (baseline_mean - proposed_mean) / baseline_mean,
        "delta_nmae": paired["delta_nmae_candidate_minus_comparator"],
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "rows": len(groups),
        "groups": paired["groups"],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def derive_syncg() -> list[dict[str, Any]]:
    payloads = [load_json(RUN_ROOT / f"seed_{seed}" / "full" / "syncg.json") for seed in SEEDS]
    reference_rows = payloads[0]["per_sample_condition"]
    reference_keys = [(row["sample_id"], row["condition"]) for row in reference_rows]
    for payload in payloads[1:]:
        observed = [(row["sample_id"], row["condition"]) for row in payload["per_sample_condition"]]
        if observed != reference_keys:
            raise ValueError("SyncG row roster differs across GeoPRR-Net seeds")

    output: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITIONS):
        selected = [position for position, row in enumerate(reference_rows) if row["condition"] == condition]
        proposed = [
            np.asarray([payload["per_sample_condition"][position]["mett"]["absolute_error"] for position in selected], dtype=np.float64)
            for payload in payloads
        ]
        baseline_matrix = np.asarray(
            [reference_rows[position]["external"]["EfficientNet-B0"]["absolute_errors"] for position in selected],
            dtype=np.float64,
        )
        baseline = [baseline_matrix[:, seed_index] for seed_index in range(len(SEEDS))]
        groups = [str(reference_rows[position]["scene_stem"]) for position in selected]
        output.append(
            summarize_condition(
                condition=condition,
                proposed_seed_errors=proposed,
                baseline_seed_errors=baseline,
                groups=groups,
                bootstrap_seed=BOOTSTRAP_SEED + index,
            )
        )
    return output


def derive_syncg_all_models() -> list[dict[str, Any]]:
    """Summarize every paper model on the identical six-condition SyncG roster."""
    payloads = [load_json(RUN_ROOT / f"seed_{seed}" / "full" / "syncg.json") for seed in SEEDS]
    reference_rows = payloads[0]["per_sample_condition"]
    model_keys = (
        ("GeoPRR-Net", None),
        ("Raw ResNet-18", "Direct-ResNet18"),
        ("Raw EfficientNet-B0", "EfficientNet-B0"),
        ("Raw MobileNetV3-Large", "MobileNetV3-Large"),
    )
    output: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        selected = [position for position, row in enumerate(reference_rows) if row["condition"] == condition]
        groups = [str(reference_rows[position]["scene_stem"]) for position in selected]
        proposed = [
            np.asarray([payload["per_sample_condition"][position]["mett"]["absolute_error"] for position in selected], dtype=np.float64)
            for payload in payloads
        ]
        proposed_seed_nmae = [float(values.mean()) for values in proposed]
        proposed_mean, proposed_sd = distribution(proposed_seed_nmae)
        for model_index, (method, external_key) in enumerate(model_keys):
            if external_key is None:
                seed_errors = proposed
                mean, sd = proposed_mean, proposed_sd
                relative_reduction = 0.0
                delta, ci_low, ci_high = 0.0, 0.0, 0.0
            else:
                seed_errors = [
                    np.asarray(
                        [reference_rows[position]["external"][external_key]["absolute_errors"][seed_index] for position in selected],
                        dtype=np.float64,
                    )
                    for seed_index in range(len(SEEDS))
                ]
                seed_nmae = [float(values.mean()) for values in seed_errors]
                mean, sd = distribution(seed_nmae)
                paired = _paired_group_bootstrap_fast(
                    np.asarray(proposed, dtype=np.float64).mean(axis=0),
                    np.asarray(seed_errors, dtype=np.float64).mean(axis=0),
                    groups,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED + 200 + 10 * condition_index + model_index,
                )
                interval = paired["paired_group_bootstrap_ci95"]
                relative_reduction = 100.0 * (mean - proposed_mean) / mean
                delta = paired["delta_nmae_candidate_minus_comparator"]
                ci_low, ci_high = interval["low"], interval["high"]
            seed_acc2 = [float(np.mean(values <= 0.02)) for values in seed_errors]
            acc2_mean, acc2_sd = distribution(seed_acc2)
            output.append(
                {
                    "method": method,
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "nmae_mean": mean,
                    "nmae_sd": sd,
                    "acc2_mean": acc2_mean,
                    "acc2_sd": acc2_sd,
                    "geo_reduction_percent": relative_reduction,
                    "delta_geo_minus_method": delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "rows": len(groups),
                    "groups": len(set(groups)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return output


def industrial_rows(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for dataset_name in sorted(payload["datasets"]):
        rows.extend((dataset_name, row) for row in payload["datasets"][dataset_name]["per_sample_condition"])
    return rows


def derive_industrial() -> list[dict[str, Any]]:
    payloads = [load_json(RUN_ROOT / f"seed_{seed}" / "full" / "industrial.json") for seed in SEEDS]
    rosters = [industrial_rows(payload) for payload in payloads]
    reference = rosters[0]
    reference_keys = [(dataset, row["sample_id"], row["condition"]) for dataset, row in reference]
    for roster in rosters[1:]:
        observed = [(dataset, row["sample_id"], row["condition"]) for dataset, row in roster]
        if observed != reference_keys:
            raise ValueError("Industrial row roster differs across GeoPRR-Net seeds")

    output: list[dict[str, Any]] = []
    for index, condition in enumerate(CONDITIONS):
        selected = [position for position, (_dataset, row) in enumerate(reference) if row["condition"] == condition]
        proposed = [
            np.asarray([roster[position][1]["candidate"]["mett"]["absolute_error"] for position in selected], dtype=np.float64)
            for roster in rosters
        ]
        baseline = [
            np.asarray(
                [reference[position][1]["efficientnet_b0"]["raw"][str(seed)]["absolute_error"] for position in selected],
                dtype=np.float64,
            )
            for seed in SEEDS
        ]
        groups = [f"{reference[position][0]}:{reference[position][1]['group_id']}" for position in selected]
        output.append(
            summarize_condition(
                condition=condition,
                proposed_seed_errors=proposed,
                baseline_seed_errors=baseline,
                groups=groups,
                bootstrap_seed=BOOTSTRAP_SEED + 100 + index,
            )
        )
    return output


def derive_ablation(report: dict[str, Any]) -> list[dict[str, Any]]:
    variants = (
        ("no_geometry_fusion", "No geometry-aware fusion"),
        ("fixed_routing", "Fixed routing"),
        ("no_relational_transport", "No relational transport"),
        ("no_polar_evidence", "No polar evidence"),
    )
    output: list[dict[str, Any]] = []
    full = report["syncg_ablation"]["full"]["conditions"]
    for variant, variant_label in variants:
        conditions = report["syncg_ablation"][variant]["conditions"]
        for condition in CONDITIONS:
            penalties = np.asarray(conditions[condition]["per_seed"], dtype=np.float64) - np.asarray(
                full[condition]["per_seed"], dtype=np.float64
            )
            output.append(
                {
                    "variant": variant,
                    "variant_label": variant_label,
                    "condition": condition,
                    "label": CONDITION_LABELS[condition],
                    "penalty_mean": float(penalties.mean()),
                    "penalty_sd": float(penalties.std(ddof=1)),
                }
            )
    return output


def derive_routing() -> list[dict[str, Any]]:
    payload = load_json(RUN_ROOT / f"seed_{SEEDS[0]}" / "full" / "routing.json")
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        summary = payload["summary"][condition]
        weights = summary["mean_routing_weights"]
        oracle = summary["oracle_candidate_fraction"]
        output.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "base": weights["base"],
                "polar": weights["polar_evidence"],
                "relational": weights["relational_transport"],
                "oracle_base": oracle["base"],
                "oracle_polar": oracle["polar_evidence"],
                "oracle_relational": oracle["relational_transport"],
                "routing_gain": -float(summary["adaptive_minus_fixed_prior_nmae"]),
                "seed": SEEDS[0],
            }
        )
    return output


def main() -> None:
    report = load_json(REPORT_PATH)
    condition_fields = [
        "condition",
        "label",
        "geo_nmae_mean",
        "geo_nmae_sd",
        "baseline_nmae_mean",
        "baseline_nmae_sd",
        "relative_reduction_percent",
        "delta_nmae",
        "ci_low",
        "ci_high",
        "rows",
        "groups",
        "bootstrap_replicates",
    ]
    write_csv("syncg_condition_comparison.csv", condition_fields, derive_syncg())
    write_csv(
        "syncg_all_models_conditions.csv",
        [
            "method",
            "condition",
            "label",
            "nmae_mean",
            "nmae_sd",
            "acc2_mean",
            "acc2_sd",
            "geo_reduction_percent",
            "delta_geo_minus_method",
            "ci_low",
            "ci_high",
            "rows",
            "groups",
            "bootstrap_replicates",
        ],
        derive_syncg_all_models(),
    )
    write_csv("industrial_condition_comparison.csv", condition_fields, derive_industrial())
    write_csv(
        "ablation_condition_effects.csv",
        ["variant", "variant_label", "condition", "label", "penalty_mean", "penalty_sd"],
        derive_ablation(report),
    )
    write_csv(
        "routing_conditions.csv",
        [
            "condition",
            "label",
            "base",
            "polar",
            "relational",
            "oracle_base",
            "oracle_polar",
            "oracle_relational",
            "routing_gain",
            "seed",
        ],
        derive_routing(),
    )
    print("Derived all-model SyncG, Industrial-1395, ablation, and routing condition summaries.")


if __name__ == "__main__":
    main()
