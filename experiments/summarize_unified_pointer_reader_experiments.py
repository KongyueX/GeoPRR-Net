"""Assemble multiseed ablations, external comparisons, and stability tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.evaluate_paper_syncg_only_lightweight_baselines import (
    _paired_group_bootstrap_fast,
)
from experiments.evaluate_a15_2_syncg_scene_holdout import CONDITIONS
from experiments.unified_pointer_reader import (
    FIXED_ROUTING,
    FULL,
    NO_GEOMETRY_FUSION,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
)


PROTOCOL: Final[str] = "unified_pointer_reader_paper_experiment_summary_v3"
DEFAULT_SEEDS: Final[tuple[int, ...]] = (20_262_020, 20_262_021, 20_262_022)
VARIANTS: Final[tuple[str, ...]] = (
    FULL,
    NO_GEOMETRY_FUSION,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
    FIXED_ROUTING,
)
DISPLAY_NAMES: Final[dict[str, str]] = {
    FULL: "Full",
    NO_GEOMETRY_FUSION: "w/o geometry-aware fusion",
    NO_POLAR_EVIDENCE: "w/o polar evidence",
    NO_RELATIONAL_TRANSPORT: "w/o relational transport",
    FIXED_ROUTING: "w/o adaptive routing",
}
RAW_EXTERNAL_MODELS: Final[dict[str, tuple[str, str, str]]] = {
    # display name: (experiment directory, architecture directory, ledger method prefix)
    "ResNet-18": ("factorial", "resnet18_direct", "Direct-ResNet18"),
    "EfficientNet-B0": (
        "lightweight_baselines",
        "efficientnet_b0",
        "EfficientNet-B0",
    ),
    "MobileNetV3-Large": (
        "lightweight_baselines",
        "mobilenet_v3_large",
        "MobileNetV3-Large",
    ),
}
SYNCG_EXTERNAL_DATASET: Final[str] = "syncg_scene_holdout"
FAILURE_ERROR: Final[float] = 1.0
INDUSTRIAL_DATASET_NAMES: Final[dict[str, str]] = {
    "field_gauge_roi_test_a": "FieldGauge-ROI Test-A",
    "field_gauge_roi_test_b": "FieldGauge-ROI Test-B",
    "field_gauge_external_roi": "FieldGauge-External-ROI",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"missing experiment result: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("status") == "complete", f"incomplete result: {source}")
    return value


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    _require(bool(values), "empty seed metric")
    array = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(array)),
        "sample_sd": float(statistics.stdev(array)) if len(array) >= 2 else 0.0,
        "minimum": min(array),
        "maximum": max(array),
        "range": max(array) - min(array),
        "per_seed": array,
    }


def _metric(payload: Mapping[str, Any], scope: str) -> Mapping[str, Any]:
    return payload["summary"][scope]["candidate"]["mett"]


def _row_errors(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, str]], list[str], np.ndarray]:
    ordered: list[tuple[str, str]] | None = None
    groups: list[str] | None = None
    seed_errors: list[list[float]] = []
    for payload in payloads:
        rows = payload["per_sample_condition"]
        keys = [(str(row["sample_id"]), str(row["condition"])) for row in rows]
        scenes = [str(row["scene_stem"]) for row in rows]
        errors = [float(row["mett"]["absolute_error"]) for row in rows]
        if ordered is None:
            ordered = keys
            groups = scenes
        else:
            _require(keys == ordered and scenes == groups, "SyncG row roster differs across seeds")
        seed_errors.append(errors)
    _require(ordered is not None and groups is not None, "SyncG payloads are empty")
    return ordered, groups, np.asarray(seed_errors, dtype=np.float64).mean(axis=0)


def _variant_summary(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nmae = [float(_metric(payload, "all_conditions")["nmae"]) for payload in payloads]
    acc2 = [float(_metric(payload, "all_conditions")["acc_at_2_percent"]) for payload in payloads]
    per_condition = {
        str(condition): _distribution(
            [float(_metric(payload, str(condition))["nmae"]) for payload in payloads]
        )
        for condition in CONDITIONS
    }
    worst = max(per_condition, key=lambda name: per_condition[name]["mean"])
    active_parameters = {
        int(payload["model"]["parameter_inventory"]["active_executable"])
        for payload in payloads
    }
    _require(len(active_parameters) == 1, "active parameter count differs across seeds")
    return {
        "active_parameters": active_parameters.pop(),
        "nmae": _distribution(nmae),
        "acc_at_2_percent": _distribution(acc2),
        "conditions": per_condition,
        "worst_condition": worst,
        "worst_condition_nmae": per_condition[worst],
    }


def _raw_prediction_path(
    external_root: Path,
    *,
    display_name: str,
    dataset_key: str,
    seed: int,
) -> Path:
    experiment_directory, architecture_directory, _method_prefix = (
        RAW_EXTERNAL_MODELS[display_name]
    )
    return (
        Path(external_root).resolve()
        / experiment_directory
        / "evaluation"
        / "raw"
        / dataset_key
        / architecture_directory
        / f"seed_{seed}"
        / "predictions.jsonl"
    )


def _load_raw_prediction_ledger(
    path: Path,
    *,
    keys: Sequence[tuple[str, str]],
    targets: Mapping[tuple[str, str], float],
    expected_method: str,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], str], str]:
    source = Path(path).resolve()
    _require(source.is_file(), f"missing Raw external prediction file: {source}")
    errors: dict[tuple[str, str], float] = {}
    passed: dict[tuple[str, str], bool] = {}
    condition_hashes: dict[tuple[str, str], str] = {}
    protocol: str | None = None
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, Mapping), f"invalid external row: {source}:{line_number}")
            key = (str(row["sample_id"]), str(row["condition"]))
            _require(key in targets, f"unmatched Raw external row: {source}:{line_number}: {key}")
            _require(key not in errors, f"duplicate Raw external row: {source}:{line_number}: {key}")
            method = str(row.get("method"))
            _require(
                method == expected_method,
                f"Raw external method differs: expected {expected_method}, found {method}",
            )
            row_protocol = str(row.get("protocol"))
            if protocol is None:
                protocol = row_protocol
            else:
                _require(row_protocol == protocol, f"prediction protocol changes within {source}")
            condition_hash = str(row.get("condition_pixel_sha256", ""))
            _require(condition_hash, f"missing condition-pixel identity: {source}:{line_number}")
            condition_hashes[key] = condition_hash
            prediction = row.get("normalized_progress")
            is_pass = row.get("status") == "pass" and prediction is not None
            if is_pass:
                value = float(prediction)
                _require(math.isfinite(value), f"non-finite Raw external prediction: {key}")
                errors[key] = abs(value - float(targets[key]))
                passed[key] = True
            else:
                errors[key] = FAILURE_ERROR
                passed[key] = False
    _require(set(errors) == set(targets), f"Raw external row set differs: {source}")
    _require(protocol is not None, f"empty Raw external prediction file: {source}")
    return (
        np.asarray([errors[key] for key in keys], dtype=np.float64),
        np.asarray([passed[key] for key in keys], dtype=np.bool_),
        condition_hashes,
        protocol,
    )


def _load_raw_external_dataset(
    *,
    external_root: Path,
    dataset_key: str,
    keys: Sequence[tuple[str, str]],
    targets: Mapping[tuple[str, str], float],
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, np.ndarray]],
    dict[str, Any],
]:
    _require(len(keys) == len(targets), f"{dataset_key}: duplicate reference keys")
    errors_by_method: dict[str, dict[int, np.ndarray]] = {}
    passed_by_method: dict[str, dict[int, np.ndarray]] = {}
    provenance_models: dict[str, Any] = {}
    reference_hashes: dict[tuple[str, str], str] | None = None
    reference_hash_source: str | None = None
    for display_name, (
        experiment_directory,
        architecture_directory,
        method_prefix,
    ) in RAW_EXTERNAL_MODELS.items():
        errors_by_method[display_name] = {}
        passed_by_method[display_name] = {}
        per_seed: list[dict[str, Any]] = []
        for seed in DEFAULT_SEEDS:
            path = _raw_prediction_path(
                external_root,
                display_name=display_name,
                dataset_key=dataset_key,
                seed=seed,
            )
            expected_method = f"{method_prefix}_seed_{seed}"
            errors, passed, condition_hashes, protocol = _load_raw_prediction_ledger(
                path,
                keys=keys,
                targets=targets,
                expected_method=expected_method,
            )
            if reference_hashes is None:
                reference_hashes = condition_hashes
                reference_hash_source = str(path)
            else:
                _require(
                    condition_hashes == reference_hashes,
                    f"Raw degraded pixels differ: {reference_hash_source} versus {path}",
                )
            errors_by_method[display_name][seed] = errors
            passed_by_method[display_name][seed] = passed
            per_seed.append(
                {
                    "seed": seed,
                    "method": expected_method,
                    "protocol": protocol,
                    "prediction_file": str(path),
                    "rows": len(keys),
                    "passed": int(passed.sum()),
                    "failed": int((~passed).sum()),
                    "coverage": float(passed.mean()),
                }
            )
        provenance_models[display_name] = {
            "experiment_directory": experiment_directory,
            "architecture_directory": architecture_directory,
            "input_variant": "raw",
            "per_seed": per_seed,
        }
    _require(reference_hashes is not None, f"{dataset_key}: no Raw external ledgers")
    provenance = {
        "dataset_key": dataset_key,
        "input_variant": "raw",
        "rows_per_seed": len(keys),
        "exact_row_roster_match": True,
        "condition_pixel_identity_across_models_and_seeds": True,
        "models": provenance_models,
    }
    return errors_by_method, passed_by_method, provenance


def _raw_external_summary(
    errors_by_method: Mapping[str, Mapping[int, np.ndarray]],
    passed_by_method: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for display_name in RAW_EXTERNAL_MODELS:
        errors = errors_by_method[display_name]
        passed = passed_by_method[display_name]
        result[display_name] = {
            "input_variant": "raw",
            "nmae": _distribution(
                [float(errors[seed].mean()) for seed in DEFAULT_SEEDS]
            ),
            "acc_at_2_percent": _distribution(
                [float((errors[seed] <= 0.02).mean()) for seed in DEFAULT_SEEDS]
            ),
            "coverage": _distribution(
                [float(passed[seed].mean()) for seed in DEFAULT_SEEDS]
            ),
        }
    return result


def _industrial_proposed(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seed_nmae: list[float] = []
    seed_acc2: list[float] = []
    dataset_nmae: dict[str, list[float]] = {
        value: [] for value in INDUSTRIAL_DATASET_NAMES.values()
    }
    for payload in payloads:
        all_errors: list[float] = []
        for machine_key, display_name in INDUSTRIAL_DATASET_NAMES.items():
            rows = payload["datasets"][machine_key]["per_sample_condition"]
            errors = [float(row["candidate"]["mett"]["absolute_error"]) for row in rows]
            all_errors.extend(errors)
            dataset_nmae[display_name].append(float(np.mean(errors)))
        array = np.asarray(all_errors, dtype=np.float64)
        seed_nmae.append(float(array.mean()))
        seed_acc2.append(float((array <= 0.02).mean()))
    return {
        "nmae": _distribution(seed_nmae),
        "acc_at_2_percent": _distribution(seed_acc2),
        "datasets": {
            name: {"nmae": _distribution(values)}
            for name, values in dataset_nmae.items()
        },
    }


def _industrial_raw_external_summary(
    payloads: Sequence[Mapping[str, Any]],
    *,
    external_root: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    proposed_parts: list[np.ndarray] = []
    external_parts: dict[str, list[np.ndarray]] = {
        display_name: [] for display_name in RAW_EXTERNAL_MODELS
    }
    pooled_errors: dict[str, dict[int, list[np.ndarray]]] = {
        display_name: {seed: [] for seed in DEFAULT_SEEDS}
        for display_name in RAW_EXTERNAL_MODELS
    }
    pooled_passed: dict[str, dict[int, list[np.ndarray]]] = {
        display_name: {seed: [] for seed in DEFAULT_SEEDS}
        for display_name in RAW_EXTERNAL_MODELS
    }
    dataset_summaries: dict[str, dict[str, Any]] = {
        display_name: {} for display_name in RAW_EXTERNAL_MODELS
    }
    provenance_datasets: dict[str, Any] = {}
    grouped_rows: list[str] = []
    for internal_name, paper_name in INDUSTRIAL_DATASET_NAMES.items():
        reference_rows = payloads[0]["datasets"][internal_name]["per_sample_condition"]
        keys = [(str(row["sample_id"]), str(row["condition"])) for row in reference_rows]
        targets = {
            key: float(row["normalized_target"])
            for key, row in zip(keys, reference_rows, strict=True)
        }
        dataset_groups = [
            f"{internal_name}:{row['group_id']}" for row in reference_rows
        ]
        proposed_by_seed: list[np.ndarray] = []
        for payload in payloads:
            rows = payload["datasets"][internal_name]["per_sample_condition"]
            observed = [(str(row["sample_id"]), str(row["condition"])) for row in rows]
            _require(observed == keys, f"{internal_name}: proposed row roster differs across seeds")
            observed_targets = [float(row["normalized_target"]) for row in rows]
            _require(
                observed_targets == [targets[key] for key in keys],
                f"{internal_name}: proposed targets differ across seeds",
            )
            proposed_by_seed.append(
                np.asarray(
                    [float(row["candidate"]["mett"]["absolute_error"]) for row in rows],
                    dtype=np.float64,
                )
            )
        proposed_parts.append(np.mean(np.stack(proposed_by_seed, axis=0), axis=0))
        grouped_rows.extend(dataset_groups)

        errors_by_method, passed_by_method, provenance = _load_raw_external_dataset(
            external_root=external_root,
            dataset_key=internal_name,
            keys=keys,
            targets=targets,
        )
        provenance_datasets[paper_name] = provenance
        for display_name in RAW_EXTERNAL_MODELS:
            per_seed_errors: list[np.ndarray] = []
            per_seed_passed: list[np.ndarray] = []
            for seed in DEFAULT_SEEDS:
                errors = errors_by_method[display_name][seed]
                passed = passed_by_method[display_name][seed]
                per_seed_errors.append(errors)
                per_seed_passed.append(passed)
                pooled_errors[display_name][seed].append(errors)
                pooled_passed[display_name][seed].append(passed)
            external_parts[display_name].append(
                np.mean(np.stack(per_seed_errors, axis=0), axis=0)
            )
            dataset_summaries[display_name][paper_name] = {
                "input_variant": "raw",
                "nmae": _distribution(
                    [float(errors.mean()) for errors in per_seed_errors]
                ),
                "acc_at_2_percent": _distribution(
                    [float((errors <= 0.02).mean()) for errors in per_seed_errors]
                ),
                "coverage": _distribution(
                    [float(passed.mean()) for passed in per_seed_passed]
                ),
            }

    proposed = np.concatenate(proposed_parts)
    summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for index, (display_name, parts) in enumerate(external_parts.items(), start=1):
        per_seed_errors = {
            seed: np.concatenate(pooled_errors[display_name][seed])
            for seed in DEFAULT_SEEDS
        }
        per_seed_passed = {
            seed: np.concatenate(pooled_passed[display_name][seed])
            for seed in DEFAULT_SEEDS
        }
        summaries[display_name] = {
            "input_variant": "raw",
            "nmae": _distribution(
                [float(per_seed_errors[seed].mean()) for seed in DEFAULT_SEEDS]
            ),
            "acc_at_2_percent": _distribution(
                [
                    float((per_seed_errors[seed] <= 0.02).mean())
                    for seed in DEFAULT_SEEDS
                ]
            ),
            "coverage": _distribution(
                [float(per_seed_passed[seed].mean()) for seed in DEFAULT_SEEDS]
            ),
            "datasets": dataset_summaries[display_name],
        }
        comparison = _paired_group_bootstrap_fast(
            proposed,
            np.concatenate(parts),
            grouped_rows,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        comparison.update(
            {
                "rows": len(grouped_rows),
                "seed_handling": "average aligned per-row errors over three training seeds",
                "group_unit": "dataset-prefixed acquisition group",
            }
        )
        paired[f"Proposed_minus_{display_name}"] = comparison
    provenance = {
        "input_variant": "raw",
        "cohort": "Test-A + Test-B + External-ROI",
        "rows_per_seed": len(grouped_rows),
        "exact_row_roster_match": True,
        "datasets": provenance_datasets,
    }
    return summaries, paired, provenance


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"CSV rows are empty: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _efficiency_row(payload: Mapping[str, Any]) -> dict[str, float]:
    latency = payload["measurement"]["latency"]
    memory = payload["measurement"]["cuda_memory"]
    return {
        "parameters": int(payload["parameters"]["total_parameters"]),
        "p50_ms": float(latency["p50_ms"]),
        "p95_ms": float(latency["p95_ms"]),
        "throughput_images_per_second": float(
            latency["throughput_images_per_second"]
        ),
        "peak_allocated_mib": float(memory["peak_allocated_mib"]),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    full = report["syncg_ablation"][FULL]
    lines = [
        "# Unified pointer reader paper experiments",
        "",
        "## Structural ablation on SyncG",
        "",
        "| Variant | Active params (M) | NMAE mean±SD | Acc@2% mean±SD | Worst condition NMAE | ΔNMAE vs Full [95% scene bootstrap] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["syncg_ablation"][variant]
        nmae = row["nmae"]
        acc2 = row["acc_at_2_percent"]
        worst = row["worst_condition_nmae"]["mean"]
        if variant == FULL:
            contrast = "—"
        else:
            paired = report["paired_syncg"][f"full_minus_{variant}"]
            ci = paired["paired_group_bootstrap_ci95"]
            contrast = f"{paired['delta_nmae_candidate_minus_comparator']:.6f} [{ci['low']:.6f}, {ci['high']:.6f}]"
        lines.append(
            f"| {DISPLAY_NAMES[variant]} | {row['active_parameters']/1e6:.3f} | "
            f"{nmae['mean']:.6f}±{nmae['sample_sd']:.6f} | "
            f"{acc2['mean']:.4f}±{acc2['sample_sd']:.4f} | {worst:.6f} | {contrast} |"
        )
    polar = report["syncg_ablation"][NO_POLAR_EVIDENCE]
    lines.extend(
        [
            "",
            "The polar branch has a small, metric-dependent effect: removing it raises mean NMAE "
            f"by {polar['nmae']['mean'] - full['nmae']['mean']:.6f}, "
            "while Acc@2% changes slightly in the opposite direction.",
        ]
    )
    lines.extend(
        [
            "",
            "## External comparison on SyncG (Raw inputs)",
            "",
            "ResNet-18, EfficientNet-B0, and MobileNetV3-Large are scored from the "
            "Raw prediction ledgers. No SARN or proposed-model frontend is applied.",
            "",
            "| Method | NMAE mean±SD | Acc@2% mean±SD | ΔNMAE Proposed−method [95% scene bootstrap] |",
            "|---|---:|---:|---:|",
        ]
    )
    lines.append(
        f"| Proposed | {full['nmae']['mean']:.6f}±{full['nmae']['sample_sd']:.6f} | "
        f"{full['acc_at_2_percent']['mean']:.4f}±{full['acc_at_2_percent']['sample_sd']:.4f} | — |"
    )
    for method, row in report["syncg_external"].items():
        paired = report["paired_syncg"][f"full_minus_{method}"]
        ci = paired["paired_group_bootstrap_ci95"]
        lines.append(
            f"| {method} (Raw) | {row['nmae']['mean']:.6f}±{row['nmae']['sample_sd']:.6f} | "
            f"{row['acc_at_2_percent']['mean']:.4f}±{row['acc_at_2_percent']['sample_sd']:.4f} | "
            f"{paired['delta_nmae_candidate_minus_comparator']:.6f} [{ci['low']:.6f}, {ci['high']:.6f}] |"
        )
    stability = report["router_ema_stability"]
    stability_paired = stability["paired_ema_minus_terminal"]
    stability_ci = stability_paired["paired_group_bootstrap_ci95"]
    lines.extend(
        [
            "",
            "## Router parameter averaging stability",
            "",
            f"Terminal router: {stability['terminal_router']['nmae']['mean']:.6f}±{stability['terminal_router']['nmae']['sample_sd']:.6f}",
            f"EMA router: {stability['ema_router']['nmae']['mean']:.6f}±{stability['ema_router']['nmae']['sample_sd']:.6f}",
            f"EMA−terminal ΔNMAE: {stability_paired['delta_nmae_candidate_minus_comparator']:.8f} "
            f"[{stability_ci['low']:.8f}, {stability_ci['high']:.8f}]; the interval crosses zero.",
        ]
    )
    if report.get("industrial"):
        lines.extend(
            [
                "",
                "## Frozen Industrial test-only comparison with Raw external models (1,395 images × 6 conditions)",
                "",
                "| Method | Pooled NMAE mean±SD | Pooled Acc@2% mean±SD | ΔNMAE Proposed−method [95% group bootstrap] |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, row in report["industrial"].items():
            display_method = method if method == "Proposed" else f"{method} (Raw)"
            if method == "Proposed":
                contrast = "—"
            else:
                paired = report["paired_industrial"][f"Proposed_minus_{method}"]
                ci = paired["paired_group_bootstrap_ci95"]
                contrast = (
                    f"{paired['delta_nmae_candidate_minus_comparator']:.6f} "
                    f"[{ci['low']:.6f}, {ci['high']:.6f}]"
                )
            lines.append(
                f"| {display_method} | {row['nmae']['mean']:.6f}±{row['nmae']['sample_sd']:.6f} | "
                f"{row['acc_at_2_percent']['mean']:.4f}±{row['acc_at_2_percent']['sample_sd']:.4f} | "
                f"{contrast} |"
            )
        lines.extend(
            [
                "",
                "### Industrial per-domain NMAE",
                "",
                "| Method | FieldGauge-ROI Test-A | FieldGauge-ROI Test-B | FieldGauge-External-ROI |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, row in report["industrial"].items():
            display_method = method if method == "Proposed" else f"{method} (Raw)"
            cells = []
            for dataset_name in INDUSTRIAL_DATASET_NAMES.values():
                metric = row["datasets"][dataset_name]["nmae"]
                cells.append(
                    f"{metric['mean']:.6f}±{metric['sample_sd']:.6f}"
                )
            lines.append(f"| {display_method} | {' | '.join(cells)} |")
        reversals: list[str] = []
        proposed_datasets = report["industrial"]["Proposed"]["datasets"]
        for dataset_name in INDUSTRIAL_DATASET_NAMES.values():
            proposed_mean = proposed_datasets[dataset_name]["nmae"]["mean"]
            better_external = [
                method
                for method, row in report["industrial"].items()
                if method != "Proposed"
                and row["datasets"][dataset_name]["nmae"]["mean"] < proposed_mean
            ]
            if better_external:
                reversals.append(f"{dataset_name}: {', '.join(better_external)}")
        if reversals:
            lines.extend(
                [
                    "",
                    "Descriptive subgroup reversal relative to the pooled result: "
                    + "; ".join(reversals)
                    + ". The pooled result therefore does not imply per-domain dominance.",
                ]
            )
        efficientnet_pair = report["paired_industrial"]["Proposed_minus_EfficientNet-B0"]
        efficientnet_ci = efficientnet_pair["paired_group_bootstrap_ci95"]
        if efficientnet_ci["low"] <= 0.0 <= efficientnet_ci["high"]:
            lines.extend(
                [
                    "",
                    "Industrial NMAE is statistically tied with EfficientNet-B0 under the paired "
                    "52-group bootstrap; no Industrial NMAE superiority is claimed.",
                ]
            )
    mechanism = report.get("mechanism_analysis")
    if mechanism:
        all_conditions = mechanism["summary"]["all_conditions"]
        severe = mechanism["summary"]["combined_severe"]
        weights = all_conditions["mean_routing_weights"]
        severe_weights = severe["mean_routing_weights"]
        oracle = all_conditions["oracle_candidate_fraction"]
        hardest = mechanism["hardest_cases"]
        edge_cases = sum(
            float(row["normalized_target"]) <= 0.05
            or float(row["normalized_target"]) >= 0.95
            for row in hardest
        )
        lines.extend(
            [
                "",
                "## Routing mechanism analysis (prespecified first seed)",
                "",
                f"Adaptive routing improves NMAE over the fixed 0.50/0.25/0.25 prior by "
                f"{-all_conditions['adaptive_minus_fixed_prior_nmae']:.6f}. Mean Full weights "
                f"are {weights['base']:.3f}/{weights['polar_evidence']:.3f}/"
                f"{weights['relational_transport']:.3f} for base/polar/relation; under "
                f"combined-severe they shift to {severe_weights['base']:.3f}/"
                f"{severe_weights['polar_evidence']:.3f}/{severe_weights['relational_transport']:.3f}.",
                f"The oracle-best candidate fractions are {oracle['base']:.3f}/"
                f"{oracle['polar_evidence']:.3f}/{oracle['relational_transport']:.3f}, "
                "supporting candidate complementarity.",
                f"Among the {len(hardest)} largest-error rows, {edge_cases} have targets within "
                "5% of an endpoint; the raw rows remain in the routing artifact for audit.",
            ]
        )
    vdn = report.get("vdn_supplement")
    if vdn:
        vdn_all = vdn["summary"]["all_conditions"]
        proposed_vdn = vdn_all["unified_pointer_reader"]["metric_across_seed_mean_sd"]
        vdn_component = vdn_all["vdn_official200_annotation_reference"]["metrics"]
        vdn_pair = vdn_all["paired_scene_bootstrap"]["versus_vdn_annotation_reference"]
        vdn_ci = vdn_pair["scene_grouped_bootstrap_ci95"]
        automatic = vdn_all["vdn_official200_automatic_reference"]
        lines.extend(
            [
                "",
                "## Supplemental VDN comparison on the fixed double-holdout intersection",
                "",
                f"On {vdn['cohort']['intersection_samples']} samples from "
                f"{vdn['cohort']['intersection_scene_groups']} scenes "
                f"({vdn['cohort']['intersection_rows']} condition rows), Proposed reaches "
                f"{proposed_vdn['nmae']['mean']:.6f}±{proposed_vdn['nmae']['sample_sd']:.6f} "
                f"NMAE versus {vdn_component['nmae']:.6f} for the single-checkpoint VDN "
                "annotation-reference component.",
                f"Proposed−VDN ΔNMAE is {vdn_pair['delta_nmae']:.6f} "
                f"[{vdn_ci['low']:.6f}, {vdn_ci['high']:.6f}].",
                f"This is supplemental and not input-equivalent: the VDN component uses an "
                f"annotation-derived reference arc. Its deployable automatic-reference coverage "
                f"is only {automatic['coverage']:.1%}, so it is excluded from the main table.",
            ]
        )
    if report.get("efficiency_comparison"):
        lines.extend(
            [
                "",
                "## Matched batch-1 FP32 efficiency",
                "",
                "| Method | Params (M) | P50 (ms) | P95 (ms) | img/s | Peak CUDA MiB |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in report["efficiency_comparison"].items():
            lines.append(
                f"| {method} | {row['parameters']/1e6:.3f} | {row['p50_ms']:.3f} | "
                f"{row['p95_ms']:.3f} | {row['throughput_images_per_second']:.2f} | "
                f"{row['peak_allocated_mib']:.1f} |"
            )
        operation_estimate = report.get("operation_estimate")
        if operation_estimate:
            lines.extend(
                [
                    "",
                    f"Supported-operation estimate: {operation_estimate['supported_gmacs']:.3f} GMAC "
                    "at batch 1. This is a lower bound because tensor-only moment transport, "
                    "grid sampling, softmax, and unsupported operators are omitted.",
                ]
            )
    lines.extend(
        [
            "",
            "Negative ΔNMAE favors the proposed full model. Seeds are averaged before resampling the 14 SyncG scenes.",
            "All three external CNN baselines use Raw degraded ROI pixels and their model-native preprocessing only.",
            "Industrial data are test-only and are not used for checkpoint or variant selection.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(
    *,
    run_root: Path,
    output_dir: Path,
    seeds: Sequence[int],
    external_root: Path,
    vdn_report_path: Path | None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    output = Path(output_dir).resolve()
    _require(not output.exists(), f"paper summary output exists: {output}")
    _require(tuple(seeds) == DEFAULT_SEEDS, "formal summary requires the three fixed seeds")
    syncg: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        syncg[variant] = [
            _load(root / f"seed_{seed}" / variant / "syncg.json") for seed in seeds
        ]
    terminal = [
        _load(root / f"seed_{seed}" / FULL / "syncg_terminal_router.json")
        for seed in seeds
    ]
    ablation = {variant: _variant_summary(payloads) for variant, payloads in syncg.items()}
    full_keys, groups, full_errors = _row_errors(syncg[FULL])
    paired: dict[str, Any] = {}
    for index, variant in enumerate(VARIANTS[1:], start=1):
        keys, variant_groups, errors = _row_errors(syncg[variant])
        _require(keys == full_keys and variant_groups == groups, "ablation row roster differs")
        paired[f"full_minus_{variant}"] = _paired_group_bootstrap_fast(
            full_errors,
            errors,
            groups,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
    syncg_reference_rows = syncg[FULL][0]["per_sample_condition"]
    syncg_targets = {
        (str(row["sample_id"]), str(row["condition"])): float(
            row["normalized_target"]
        )
        for row in syncg_reference_rows
    }
    _require(
        list(syncg_targets) == full_keys,
        "SyncG target roster differs from proposed predictions",
    )
    syncg_external_errors, syncg_external_passed, syncg_external_provenance = (
        _load_raw_external_dataset(
            external_root=external_root,
            dataset_key=SYNCG_EXTERNAL_DATASET,
            keys=full_keys,
            targets=syncg_targets,
        )
    )
    external = _raw_external_summary(
        syncg_external_errors,
        syncg_external_passed,
    )
    for index, display_name in enumerate(RAW_EXTERNAL_MODELS, start=20):
        errors = np.mean(
            np.stack(
                [
                    syncg_external_errors[display_name][seed]
                    for seed in DEFAULT_SEEDS
                ],
                axis=0,
            ),
            axis=0,
        )
        paired[f"full_minus_{display_name}"] = _paired_group_bootstrap_fast(
            full_errors,
            errors,
            groups,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
    terminal_keys, terminal_groups, terminal_errors = _row_errors(terminal)
    _require(terminal_keys == full_keys and terminal_groups == groups, "stability row roster differs")
    stability = {
        "terminal_router": _variant_summary(terminal),
        "ema_router": ablation[FULL],
        "paired_ema_minus_terminal": _paired_group_bootstrap_fast(
            full_errors,
            terminal_errors,
            groups,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 50,
        ),
        "same_training_trajectory": True,
        "shared_polar_weight_variant": "ema",
    }
    industrial: dict[str, Any] = {}
    paired_industrial: dict[str, Any] = {}
    industrial_external_provenance: dict[str, Any] | None = None
    industrial_paths = [root / f"seed_{seed}" / FULL / "industrial.json" for seed in seeds]
    if all(path.is_file() for path in industrial_paths):
        industrial_payloads = [_load(path) for path in industrial_paths]
        industrial["Proposed"] = _industrial_proposed(industrial_payloads)
        external_industrial, paired_industrial, industrial_external_provenance = (
            _industrial_raw_external_summary(
                industrial_payloads,
                external_root=external_root,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 100,
            )
        )
        industrial.update(external_industrial)
    report: dict[str, Any] = {
        "schema_version": 3,
        "protocol": PROTOCOL,
        "status": "complete",
        "seeds": list(seeds),
        "external_baseline_protocol": {
            "input_variant": "raw",
            "preprocessing": "raw degraded ROI pixels with model-native preprocessing only",
            "sarn_or_proposed_frontend_applied": False,
            "training_seeds": list(seeds),
            "failure_error": FAILURE_ERROR,
            "syncg": syncg_external_provenance,
            "industrial": industrial_external_provenance,
        },
        "syncg_ablation": ablation,
        "syncg_external": external,
        "paired_syncg": paired,
        "router_ema_stability": stability,
        "industrial": industrial,
        "paired_industrial": paired_industrial,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "SyncG scene",
            "seed_handling": "average per-row errors over training seeds before scene resampling",
            "industrial_unit": "dataset-prefixed acquisition group",
        },
    }
    mechanism = root / f"seed_{seeds[0]}" / FULL / "routing.json"
    efficiency = root / f"seed_{seeds[0]}" / FULL / "efficiency.json"
    operation_estimate = root / f"seed_{seeds[0]}" / FULL / "efficiency_bf16.json"
    report["mechanism_analysis"] = _load(mechanism) if mechanism.is_file() else None
    report["vdn_supplement"] = (
        _load(vdn_report_path)
        if vdn_report_path is not None and Path(vdn_report_path).resolve().is_file()
        else None
    )
    report["efficiency"] = _load(efficiency) if efficiency.is_file() else None
    report["operation_estimate"] = (
        _load(operation_estimate)["operation_estimate"]
        if operation_estimate.is_file()
        else None
    )
    report["efficiency_comparison"] = {}
    if report["efficiency"] is not None:
        report["efficiency_comparison"]["Proposed"] = _efficiency_row(
            report["efficiency"]
        )
        efficiency_root = Path(external_root) / "efficiency"
        external_efficiency = {
            "ResNet-18": efficiency_root / "direct_resnet18.json",
            "EfficientNet-B0": efficiency_root / "efficientnet_b0.json",
            "MobileNetV3-Large": efficiency_root / "mobilenet_v3_large.json",
        }
        for name, path in external_efficiency.items():
            report["efficiency_comparison"][name] = _efficiency_row(_load(path))
    output.mkdir(parents=True, exist_ok=False)
    (output / "paper_results.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "paper_results.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    ablation_rows = []
    for variant in VARIANTS:
        row = ablation[variant]
        ablation_rows.append(
            {
                "variant": DISPLAY_NAMES[variant],
                "active_parameters": row["active_parameters"],
                "nmae_mean": row["nmae"]["mean"],
                "nmae_sd": row["nmae"]["sample_sd"],
                "acc2_mean": row["acc_at_2_percent"]["mean"],
                "acc2_sd": row["acc_at_2_percent"]["sample_sd"],
                "worst_condition": row["worst_condition"],
                "worst_condition_nmae": row["worst_condition_nmae"]["mean"],
            }
        )
    _write_csv(output / "syncg_ablation.csv", ablation_rows)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs/unified_pointer_reader"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/unified_pointer_reader"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("artifacts/runs/paper_syncg_only_retrain_v1"),
        help="Root containing factorial/ and lightweight_baselines/ Raw ledgers.",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_826)
    parser.add_argument(
        "--vdn-report",
        type=Path,
        default=Path("artifacts/reports/unified_pointer_reader_vdn_intersection.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = summarize(
        run_root=args.run_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        external_root=args.external_root,
        vdn_report_path=args.vdn_report,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(Path(args.output_dir).resolve()),
                "full_syncg_nmae": report["syncg_ablation"][FULL]["nmae"],
                "industrial_included": bool(report["industrial"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "render_markdown", "summarize"]
