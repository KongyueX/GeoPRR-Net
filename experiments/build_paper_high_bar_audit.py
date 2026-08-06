"""Build the frozen aggregate audit used by the high-bar paper review.

The script is CPU-only.  It reads an explicit allowlist of existing JSON/JSONL
aggregate and prediction caches, performs no inference or training, never opens
an image, and rejects public, test, sealed, dataset, and image paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.strict_json import (
    STRICT_JSON_PROTOCOL,
    strict_json_load,
    strict_json_source_sha256,
    strict_jsonl_load,
)


PROTOCOL = "paper_high_bar_aggregate_audit_v1"
VERIFICATION_PROTOCOL = "paper_high_bar_aggregate_audit_verification_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/runs/paper_high_bar_audit_v1"
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260805
FADR_SEEDS = (20260722, 20260723, 20260724)
COMPARATOR_ORDER = ("vdn", "base_mask", "original_transformer", "fadr")
PREDICTION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AuditInputs:
    uncertainty_audit: Path
    fadr_cohort: Path
    train_comparison: Path
    fadr_train_comparison: Path
    source_pepd: Path
    source_fadr: tuple[tuple[int, Path], ...]
    field_sensitivity: Path
    field_pepd: Path
    field_fadr: Path
    field_vdn: Path
    field_base: Path


DEFAULT_INPUTS = AuditInputs(
    uncertainty_audit=(
        PROJECT_ROOT / "artifacts/runs/angular_uncertainty_calibration/audit.json"
    ),
    fadr_cohort=(
        PROJECT_ROOT / "artifacts/runs/fadr_multiseed_v2_joint_authoritative_v2/cohort.json"
    ),
    train_comparison=(
        PROJECT_ROOT / "artifacts/runs/pepd_vdn_transformer_train_oof_v1/comparison.json"
    ),
    fadr_train_comparison=(
        PROJECT_ROOT
        / "artifacts/runs/original_transformer_vs_fadr_v2_train_oof_v1/comparison.json"
    ),
    source_pepd=(
        PROJECT_ROOT
        / "artifacts/runs/fadr_multiseed_v2_inputs_authoritative_v2"
        / "pepd_mixed_authoritative_oof.jsonl"
    ),
    source_fadr=tuple(
        (
            seed,
            PROJECT_ROOT
            / "artifacts/runs/fadr_multiseed_v2_joint_authoritative_v2"
            / f"seed_{seed}/joint/joint_oof_predictions.jsonl",
        )
        for seed in FADR_SEEDS
    ),
    field_sensitivity=(
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026"
        / "physical_entity_leakage_sensitivity_v1/summary.json"
    ),
    field_pepd=(
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026"
        / "confirmatory_one_shot_v1/pepd.jsonl"
    ),
    field_fadr=(
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026"
        / "confirmatory_one_shot_v1/final/predictions.jsonl"
    ),
    field_vdn=(
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026"
        / "official200_vdn_comparator_v1/predictions.jsonl"
    ),
    field_base=(
        PROJECT_ROOT
        / "artifacts/runs/field_holdout_xiangmu2_2026"
        / "confirmatory_one_shot_v1/base/predictions.jsonl"
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _script_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _assert_safe_artifact_path(path: Path, *, label: str) -> Path:
    """Reject any path that could name restricted data or an image."""

    resolved = Path(path).resolve()
    _require(
        resolved.suffix.lower() in {".json", ".jsonl"},
        f"{label} must be a JSON/JSONL artifact: {resolved}",
    )
    forbidden_exact = {"data", "datasets", "image", "images"}
    forbidden_tokens = {"public", "sealed", "test"}
    for part in resolved.parts:
        lowered = part.lower()
        tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
        _require(
            lowered not in forbidden_exact and not (tokens & forbidden_tokens),
            f"{label} resolves through a forbidden path component: {resolved}",
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _all_input_paths(inputs: AuditInputs) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "uncertainty_audit": inputs.uncertainty_audit,
        "fadr_cohort": inputs.fadr_cohort,
        "train_comparison": inputs.train_comparison,
        "fadr_train_comparison": inputs.fadr_train_comparison,
        "source_pepd": inputs.source_pepd,
        "field_sensitivity": inputs.field_sensitivity,
        "field_pepd": inputs.field_pepd,
        "field_fadr": inputs.field_fadr,
        "field_vdn": inputs.field_vdn,
        "field_base": inputs.field_base,
    }
    for seed, path in inputs.source_fadr:
        paths[f"source_fadr_seed_{seed}"] = path
    return paths


def _validate_input_paths(inputs: AuditInputs) -> dict[str, dict[str, Any]]:
    _require(
        tuple(seed for seed, _ in inputs.source_fadr) == FADR_SEEDS,
        f"FADR source seeds must be exactly {FADR_SEEDS}",
    )
    result: dict[str, dict[str, Any]] = {}
    for label, path in _all_input_paths(inputs).items():
        resolved = _assert_safe_artifact_path(path, label=label)
        result[label] = {
            "path": _relative(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": _sha256_file(resolved),
        }
    return result


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        _require(
            isinstance(sample_id, str) and bool(sample_id),
            f"{label} row {number} lacks sample_id",
        )
        _require(sample_id not in indexed, f"{label} duplicates sample_id {sample_id}")
        group_id = row.get("group_id")
        _require(
            isinstance(group_id, str) and bool(group_id),
            f"{label} row {number} lacks group_id",
        )
        indexed[sample_id] = row
    _require(bool(indexed), f"{label} is empty")
    return indexed


def _same_number(first: Any, second: Any) -> bool:
    first_value = _finite(first)
    second_value = _finite(second)
    return (
        first_value is not None
        and second_value is not None
        and math.isclose(first_value, second_value, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_row_binding(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str
) -> None:
    _require(
        reference.get("sample_id") == candidate.get("sample_id"),
        f"{label} sample_id binding mismatch",
    )
    _require(
        reference.get("group_id") == candidate.get("group_id"),
        f"{label} group_id binding mismatch for {reference.get('sample_id')}",
    )
    for field in ("ground_truth", "scale_start", "scale_end"):
        _require(
            _same_number(reference.get(field), candidate.get(field)),
            f"{label} {field} binding mismatch for {reference.get('sample_id')}",
        )


def _prediction_error(
    row: Mapping[str, Any], prediction: Any, *, successful: bool
) -> float:
    value = _finite(prediction) if successful else None
    if value is None:
        return 1.0
    ground_truth = _finite(row.get("ground_truth"))
    scale_start = _finite(row.get("scale_start"))
    scale_end = _finite(row.get("scale_end"))
    _require(
        ground_truth is not None and scale_start is not None and scale_end is not None,
        f"invalid target/scale binding for {row.get('sample_id')}",
    )
    span = abs(scale_end - scale_start)
    _require(span > 0.0, f"non-positive scale span for {row.get('sample_id')}")
    return abs(value - ground_truth) / span


def _predictions_differ(first: Any, second: Any) -> bool:
    first_value = _finite(first)
    second_value = _finite(second)
    if first_value is None or second_value is None:
        return (first_value is None) != (second_value is None)
    return abs(first_value - second_value) > PREDICTION_TOLERANCE


def _group_effects(
    deltas: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require(deltas.ndim == groups.ndim == 1, "group effects require vectors")
    _require(len(deltas) == len(groups) > 0, "group effects are empty/mismatched")
    unique = np.unique(groups.astype(str))
    sums = np.empty(len(unique), dtype=np.float64)
    counts = np.empty(len(unique), dtype=np.int64)
    means = np.empty(len(unique), dtype=np.float64)
    for index, group in enumerate(unique):
        mask = groups.astype(str) == group
        counts[index] = int(np.sum(mask))
        sums[index] = float(np.sum(deltas[mask]))
        means[index] = sums[index] / counts[index]
    return unique, sums, counts, means


def _cluster_bootstrap_samples(
    deltas: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require(iterations > 0, "bootstrap iterations must be positive")
    _, sums, counts, means = _group_effects(deltas, groups)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(sums), size=(iterations, len(sums)))
    micro = np.sum(sums[selected], axis=1) / np.sum(counts[selected], axis=1)
    macro = np.mean(means[selected], axis=1)
    return micro.astype(np.float64), macro.astype(np.float64)


def _interval(values: np.ndarray, *, level: float) -> list[float]:
    _require(0.0 < level < 1.0, "interval level must be in (0,1)")
    tail = (1.0 - level) / 2.0
    return [
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    ]


def _two_sided_exact_sign_p(wins: int, losses: int) -> float:
    non_ties = wins + losses
    if non_ties == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(non_ties, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**non_ties))


def _holm_adjust(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def _effect_summary(
    deltas: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
    family_size: int,
) -> dict[str, Any]:
    unique, _, _, group_means = _group_effects(deltas, groups)
    bootstrap_micro, bootstrap_macro = _cluster_bootstrap_samples(
        deltas, groups, iterations=iterations, seed=seed
    )
    wins = int(np.sum(group_means > PREDICTION_TOLERANCE))
    losses = int(np.sum(group_means < -PREDICTION_TOLERANCE))
    ties = int(len(group_means) - wins - losses)

    loo_values: list[tuple[str, float]] = []
    group_strings = groups.astype(str)
    for group in unique:
        retained = group_strings != group
        _require(bool(np.any(retained)), "leave-one-group-out requires >1 group")
        loo_values.append((str(group), float(np.mean(deltas[retained]))))
    loo_min = min(loo_values, key=lambda item: (item[1], item[0]))
    loo_max = max(loo_values, key=lambda item: (item[1], item[0]))
    simultaneous_level = 1.0 - 0.05 / family_size

    return {
        "samples": int(len(deltas)),
        "groups": int(len(unique)),
        "effect_definition": "comparator NMAE minus PEPD NMAE; positive favors PEPD",
        "micro_nmae_effect": float(np.mean(deltas)),
        "micro_group_bootstrap_95ci": _interval(bootstrap_micro, level=0.95),
        "micro_bonferroni_familywise_95ci": _interval(
            bootstrap_micro, level=simultaneous_level
        ),
        "macro_group_nmae_effect": float(np.mean(group_means)),
        "macro_group_bootstrap_95ci": _interval(bootstrap_macro, level=0.95),
        "macro_bonferroni_familywise_95ci": _interval(
            bootstrap_macro, level=simultaneous_level
        ),
        "bonferroni_marginal_level": float(simultaneous_level),
        "groups_favoring_pepd": wins,
        "groups_favoring_comparator": losses,
        "groups_tied": ties,
        "exact_two_sided_group_sign_p": _two_sided_exact_sign_p(wins, losses),
        "leave_one_group_out_micro_effect": {
            "minimum": float(loo_min[1]),
            "minimum_when_excluding_group": loo_min[0],
            "maximum": float(loo_max[1]),
            "maximum_when_excluding_group": loo_max[0],
        },
        "group_effects": [
            {"group_id": str(group), "effect": float(effect)}
            for group, effect in zip(unique, group_means, strict=True)
        ],
        "bootstrap": {
            "unit": "group_id",
            "iterations": int(iterations),
            "seed": int(seed),
            "interval": "percentile",
            "paired_within_resample": True,
        },
    }


def _comparator_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    _require(bool(records), "field comparator records are empty")
    groups = np.asarray([str(row["group_id"]) for row in records])
    pepd = np.asarray([float(row["pepd_error"]) for row in records])
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for offset, comparator in enumerate(COMPARATOR_ORDER):
        values = np.asarray(
            [float(row[f"{comparator}_error"]) for row in records],
            dtype=np.float64,
        )
        summary = _effect_summary(
            values - pepd,
            groups,
            iterations=iterations,
            seed=seed + offset,
            family_size=len(COMPARATOR_ORDER),
        )
        comparisons[comparator] = summary
        raw_p[comparator] = float(summary["exact_two_sided_group_sign_p"])
    adjusted = _holm_adjust(raw_p)
    for comparator, value in adjusted.items():
        comparisons[comparator]["holm_adjusted_group_sign_p"] = float(value)
    return {
        "analysis_status": "post_hoc_prediction_independent_leakage_sensitivity",
        "comparisons": comparisons,
        "multiplicity": {
            "family": list(COMPARATOR_ORDER),
            "family_size": len(COMPARATOR_ORDER),
            "interval_method": "Bonferroni-adjusted percentile cluster bootstrap",
            "sign_test_method": "Holm-adjusted exact two-sided group sign tests",
        },
    }


def _summarize_route_changes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(bool(records), "route-change records are empty")
    route_counts: Counter[str] = Counter()
    changed = helpful = harmful = tied = 0
    deltas: list[float] = []
    changed_deltas: list[float] = []
    groups: list[str] = []
    for row in records:
        route_counts[str(row["route"])] += 1
        delta = float(row["candidate_error"]) - float(row["reference_error"])
        deltas.append(delta)
        groups.append(str(row["group_id"]))
        if not _predictions_differ(
            row.get("candidate_prediction"), row.get("reference_prediction")
        ):
            continue
        changed += 1
        changed_deltas.append(delta)
        if delta < -PREDICTION_TOLERANCE:
            helpful += 1
        elif delta > PREDICTION_TOLERANCE:
            harmful += 1
        else:
            tied += 1
    _require(changed == helpful + harmful + tied, "route-change partition drift")
    _, _, _, group_means = _group_effects(
        np.asarray(deltas, dtype=np.float64), np.asarray(groups)
    )
    group_harmed = int(np.sum(group_means > PREDICTION_TOLERANCE))
    group_helped = int(np.sum(group_means < -PREDICTION_TOLERANCE))
    group_tied = int(len(group_means) - group_harmed - group_helped)
    return {
        "samples": int(len(records)),
        "groups": int(len(group_means)),
        "route_counts": dict(sorted(route_counts.items())),
        "changed_predictions": int(changed),
        "unchanged_predictions": int(len(records) - changed),
        "helpful_changes": int(helpful),
        "harmful_changes": int(harmful),
        "equal_error_changes": int(tied),
        "helpful_fraction_among_changes": (
            float(helpful / changed) if changed else None
        ),
        "harmful_fraction_among_changes": (
            float(harmful / changed) if changed else None
        ),
        "mean_nmae_delta_all_rows": float(np.mean(deltas)),
        "mean_nmae_delta_changed_rows": (
            float(np.mean(changed_deltas)) if changed_deltas else 0.0
        ),
        "groups_harmed": group_harmed,
        "groups_helped": group_helped,
        "groups_tied": group_tied,
        "delta_definition": "FADR NMAE minus PEPD-only NMAE; positive is harmful",
        "prediction_change_tolerance": PREDICTION_TOLERANCE,
    }


def _extract_uncertainty(audit: Mapping[str, Any]) -> dict[str, Any]:
    _require(audit.get("status") == "complete", "uncertainty audit is incomplete")
    _require(
        audit.get("protocol")
        == "syncg_train_group_cross_conformal_angle_uncertainty_audit_v1",
        "unexpected uncertainty protocol",
    )
    scope = audit.get("scope") or {}
    _require(scope.get("train_only") is True, "uncertainty audit is not train-only")
    _require(scope.get("test_sets_used") == [], "uncertainty audit used test data")
    _require(scope.get("field_sets_used") == [], "uncertainty audit used field data")
    provenance = audit.get("provenance") or {}
    raw = audit.get("raw_sigma_diagnostics") or {}
    named = raw.get("named_gaussian_intervals") or {}
    cross = audit.get("cross_group_conformal") or {}
    levels = cross.get("levels") or {}
    _require(bool(levels), "uncertainty audit has no conformal levels")

    def raw_interval(name: str) -> dict[str, Any]:
        value = named.get(name) or {}
        return {
            "target_coverage": float(value["target_coverage"]),
            "observed_coverage": float(value["observed_coverage"]),
            "macro_group_coverage": float(value["macro_physical_group_coverage"]),
            "coverage_gap": float(value["coverage_gap"]),
        }

    conformal: list[dict[str, Any]] = []
    for key in sorted(levels, key=float):
        value = levels[key]
        conformal.append(
            {
                "target_coverage": float(value["target_coverage"]),
                "observed_coverage": float(value["observed_coverage"]),
                "coverage_gap": float(value["coverage_gap"]),
                "macro_group_coverage": float(
                    value["macro_physical_group_coverage"]
                ),
                "macro_group_coverage_gap": float(
                    value["macro_physical_group_coverage_gap"]
                ),
                "mean_interval_width_degrees": float(
                    value["mean_interval_width_degrees"]
                ),
                "median_interval_width_degrees": float(
                    value["median_interval_width_degrees"]
                ),
                "catastrophic_error_samples": int(
                    value["catastrophic_error_samples"]
                ),
                "catastrophic_misses": int(value["catastrophic_misses"]),
                "maximum_missed_error_degrees": float(
                    value["maximum_missed_error_degrees"]
                ),
                "sample_weighted_scale_multiplier": float(
                    value["sample_weighted_multiplier_mean"]
                ),
            }
        )
    return {
        "scope": "SyncG/train grouped OOF only; no field/test/public/sealed data",
        "samples": int(provenance["valid_uncertainty_rows"]),
        "groups": int(provenance["valid_uncertainty_groups"]),
        "excluded_rows": int(provenance["excluded_uncertainty_rows"]),
        "raw_sigma": {
            "mean_absolute_error_degrees": float(
                raw["mean_absolute_circular_angle_error_degrees"]
            ),
            "median_absolute_error_degrees": float(
                raw["median_absolute_circular_angle_error_degrees"]
            ),
            "mean_sigma_degrees": float(raw["mean_angle_std_degrees"]),
            "median_sigma_degrees": float(raw["median_angle_std_degrees"]),
            "pearson_error_sigma_correlation": float(
                raw["pearson_error_sigma_correlation"]
            ),
            "one_sigma": raw_interval("one_sigma"),
            "two_sigma": raw_interval("two_sigma"),
            "interpretation": "raw sigma is a ranking/scale diagnostic, not a calibrated Gaussian interval",
        },
        "cross_group_conformal": {
            "method": str(cross["method"]),
            "folds": int(cross["folds"]),
            "levels": conformal,
            "claim_boundary": "cross-fitted source-domain coverage only; no field-coverage claim",
        },
    }


def _validate_hash(actual_path: Path, expected: Any, *, label: str) -> None:
    _require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label} lacks a SHA-256 binding",
    )
    _require(_sha256_file(actual_path) == expected, f"{label} hash drift")


def _load_source_records(
    inputs: AuditInputs,
    *,
    train_comparison: Mapping[str, Any],
    fadr_comparison: Mapping[str, Any],
    fadr_cohort: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(fadr_cohort.get("status") == "verified", "FADR cohort is unverified")
    _require(fadr_cohort.get("samples") == 4380, "FADR sample count drift")
    _require(fadr_cohort.get("groups") == 197, "FADR group count drift")
    _require(fadr_cohort.get("seeds") == list(FADR_SEEDS), "FADR seed drift")
    for key in ("field_samples_used", "public_samples_used", "test_samples_used"):
        _require(fadr_cohort.get(key) == 0, f"FADR cohort used restricted data: {key}")

    train_sources = train_comparison.get("sources") or {}
    _validate_hash(
        inputs.source_pepd,
        (train_sources.get("pepd_authoritative_oof") or {}).get("sha256"),
        label="source PEPD OOF",
    )
    fadr_sources = fadr_comparison.get("sources") or {}
    fadr_source_bindings = fadr_sources.get("fadr_v2_joint_oof_by_seed") or {}
    for seed, path in inputs.source_fadr:
        _validate_hash(
            path,
            (fadr_source_bindings.get(str(seed)) or {}).get("sha256"),
            label=f"source FADR seed {seed}",
        )

    pepd_rows = strict_jsonl_load(inputs.source_pepd)
    pepd_index = _index_rows(pepd_rows, label="source PEPD")
    _require(len(pepd_index) == 4380, "source PEPD row count drift")
    pepd_values: dict[str, dict[str, Any]] = {}
    for sample_id, row in pepd_index.items():
        vector = row.get("vector") or {}
        prediction = vector.get("prediction")
        successful = vector.get("status") is True and _finite(prediction) is not None
        pepd_values[sample_id] = {
            "group_id": str(row["group_id"]),
            "prediction": prediction,
            "error": _prediction_error(row, prediction, successful=successful),
            "binding": row,
        }

    fadr_by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    route_by_seed: dict[str, Any] = {}
    for seed, path in inputs.source_fadr:
        indexed = _index_rows(strict_jsonl_load(path), label=f"source FADR {seed}")
        _require(set(indexed) == set(pepd_index), f"source FADR {seed} sample drift")
        route_records: list[dict[str, Any]] = []
        for sample_id, row in indexed.items():
            pepd = pepd_values[sample_id]
            _require(
                row.get("group_id") == pepd["group_id"],
                f"source FADR {seed} group mismatch for {sample_id}",
            )
            variant = (row.get("variants") or {}).get("full") or {}
            candidate_prediction = variant.get("prediction")
            candidate_error = _finite(variant.get("normalized_error"))
            _require(
                candidate_error is not None,
                f"source FADR {seed} lacks error for {sample_id}",
            )
            recomputed = _prediction_error(
                pepd["binding"],
                candidate_prediction,
                successful=_finite(candidate_prediction) is not None,
            )
            _require(
                math.isclose(candidate_error, recomputed, rel_tol=0.0, abs_tol=1e-10),
                f"source FADR {seed} error drift for {sample_id}",
            )
            route_records.append(
                {
                    "group_id": pepd["group_id"],
                    "reference_prediction": pepd["prediction"],
                    "reference_error": pepd["error"],
                    "candidate_prediction": candidate_prediction,
                    "candidate_error": candidate_error,
                    "route": str(variant.get("route")),
                }
            )
        fadr_by_seed[seed] = indexed
        route_by_seed[str(seed)] = _summarize_route_changes(route_records)

    records: list[dict[str, Any]] = []
    for sample_id in sorted(pepd_index):
        pepd = pepd_values[sample_id]
        fadr_errors = [
            float(
                ((fadr_by_seed[seed][sample_id].get("variants") or {}).get("full") or {})[
                    "normalized_error"
                ]
            )
            for seed in FADR_SEEDS
        ]
        records.append(
            {
                "sample_id": sample_id,
                "group_id": pepd["group_id"],
                "pepd_error": float(pepd["error"]),
                "fadr_error": float(np.mean(fadr_errors)),
            }
        )

    train_metrics = train_comparison.get("metrics") or {}
    expected_pepd = float(
        (train_metrics.get("pepd_only_raw_probabilistic_vector") or {})["nmae"]
    )
    fadr_metrics = ((fadr_comparison.get("metrics") or {}).get("fadr_v2_full") or {})
    expected_fadr = float((fadr_metrics.get("three_seed_mean") or {})["nmae"])
    observed_pepd = float(np.mean([row["pepd_error"] for row in records]))
    observed_fadr = float(np.mean([row["fadr_error"] for row in records]))
    _require(
        math.isclose(observed_pepd, expected_pepd, rel_tol=0.0, abs_tol=1e-12),
        "source PEPD aggregate drift",
    )
    _require(
        math.isclose(observed_fadr, expected_fadr, rel_tol=0.0, abs_tol=1e-12),
        "source FADR aggregate drift",
    )
    return records, route_by_seed


def _load_field_records(
    inputs: AuditInputs,
    *,
    sensitivity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(sensitivity.get("status") == "complete", "field sensitivity incomplete")
    _require(
        sensitivity.get("analysis_status")
        == "post_hoc_prediction_independent_leakage_sensitivity",
        "unexpected field sensitivity status",
    )
    bindings = ((sensitivity.get("bindings") or {}).get("prediction_caches") or {})
    for key, path in (
        ("pepd_cache", inputs.field_pepd),
        ("fadr_cache", inputs.field_fadr),
        ("vdn_cache", inputs.field_vdn),
        ("base_cache", inputs.field_base),
    ):
        _validate_hash(path, (bindings.get(key) or {}).get("sha256"), label=key)

    pepd_index = _index_rows(strict_jsonl_load(inputs.field_pepd), label="field PEPD")
    fadr_index = _index_rows(strict_jsonl_load(inputs.field_fadr), label="field FADR")
    vdn_index = _index_rows(strict_jsonl_load(inputs.field_vdn), label="field VDN")
    base_index = _index_rows(strict_jsonl_load(inputs.field_base), label="field base")
    sample_set = set(pepd_index)
    _require(set(fadr_index) == sample_set, "field FADR sample binding drift")
    _require(set(vdn_index) == sample_set, "field VDN sample binding drift")
    _require(set(base_index) == sample_set, "field base sample binding drift")

    denominators = sensitivity.get("denominators") or {}
    excluded = set(denominators.get("excluded_confirmatory_group_ids") or [])
    _require(len(excluded) == 2, "field excluded-group count drift")
    records: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    for sample_id in sorted(sample_set):
        pepd_row = pepd_index[sample_id]
        if pepd_row.get("group_id") in excluded:
            continue
        fadr_row = fadr_index[sample_id]
        vdn_row = vdn_index[sample_id]
        base_row = base_index[sample_id]
        for label, row in (
            ("field FADR", fadr_row),
            ("field VDN", vdn_row),
            ("field base", base_row),
        ):
            _validate_row_binding(pepd_row, row, label=label)

        pepd_prediction = pepd_row.get("prediction")
        pepd_success = pepd_row.get("status") is True and _finite(pepd_prediction) is not None
        pepd_error = _prediction_error(
            pepd_row, pepd_prediction, successful=pepd_success
        )
        raw_vector_prediction = fadr_row.get("raw_vector_prediction")
        _require(
            not _predictions_differ(raw_vector_prediction, pepd_prediction),
            f"field FADR raw-vector binding drift for {sample_id}",
        )
        fadr_prediction = fadr_row.get("prediction")
        fadr_success = fadr_row.get("status") is True and _finite(fadr_prediction) is not None
        fadr_error = _prediction_error(
            fadr_row, fadr_prediction, successful=fadr_success
        )
        vdn_prediction = vdn_row.get("prediction")
        vdn_error = _prediction_error(
            vdn_row,
            vdn_prediction,
            successful=vdn_row.get("status") is True
            and _finite(vdn_prediction) is not None,
        )
        base_predictions = base_row.get("predictions") or {}
        base_prediction = base_predictions.get("ours")
        transformer_prediction = base_predictions.get("transformer")
        base_error = _prediction_error(
            base_row,
            base_prediction,
            successful=_finite(base_prediction) is not None,
        )
        transformer_error = _prediction_error(
            base_row,
            transformer_prediction,
            successful=_finite(transformer_prediction) is not None,
        )
        record = {
            "sample_id": sample_id,
            "group_id": str(pepd_row["group_id"]),
            "pepd_error": pepd_error,
            "fadr_error": fadr_error,
            "vdn_error": vdn_error,
            "base_mask_error": base_error,
            "original_transformer_error": transformer_error,
        }
        records.append(record)
        route_records.append(
            {
                "group_id": record["group_id"],
                "reference_prediction": pepd_prediction,
                "reference_error": pepd_error,
                "candidate_prediction": fadr_prediction,
                "candidate_error": fadr_error,
                "route": str(fadr_row.get("route")),
            }
        )

    expected_samples = int(denominators["retained_samples"])
    expected_groups = int(denominators["retained_groups"])
    _require(len(records) == expected_samples == 725, "retained field sample drift")
    _require(
        len({row["group_id"] for row in records}) == expected_groups == 18,
        "retained field group drift",
    )
    expected_metrics = sensitivity.get("sensitivity_metrics") or {}
    for output_name, artifact_name in (
        ("pepd", "pepd_only"),
        ("fadr", "pepd_fadr"),
        ("vdn", "vdn_official200"),
        ("base_mask", "base_mask"),
        ("original_transformer", "original_transformer"),
    ):
        observed = float(np.mean([row[f"{output_name}_error"] for row in records]))
        expected = float((expected_metrics.get(artifact_name) or {})["full_denominator_nmae"])
        _require(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12),
            f"field {output_name} aggregate drift",
        )
    return records, route_records


def _domain_interaction(
    source_records: Sequence[Mapping[str, Any]],
    field_records: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    source_delta = np.asarray(
        [float(row["fadr_error"]) - float(row["pepd_error"]) for row in source_records]
    )
    source_groups = np.asarray([str(row["group_id"]) for row in source_records])
    field_delta = np.asarray(
        [float(row["fadr_error"]) - float(row["pepd_error"]) for row in field_records]
    )
    field_groups = np.asarray([str(row["group_id"]) for row in field_records])
    source_micro, source_macro = _cluster_bootstrap_samples(
        source_delta, source_groups, iterations=iterations, seed=seed
    )
    field_micro, field_macro = _cluster_bootstrap_samples(
        field_delta, field_groups, iterations=iterations, seed=seed + 1
    )
    interaction_micro = field_micro - source_micro
    interaction_macro = field_macro - source_macro
    _, _, _, source_group_means = _group_effects(source_delta, source_groups)
    _, _, _, field_group_means = _group_effects(field_delta, field_groups)
    source_point_micro = float(np.mean(source_delta))
    field_point_micro = float(np.mean(field_delta))
    source_point_macro = float(np.mean(source_group_means))
    field_point_macro = float(np.mean(field_group_means))
    return {
        "effect_definition": "(field FADR-minus-PEPD) minus (source FADR-minus-PEPD); positive denotes reversal toward field harm",
        "source": {
            "samples": int(len(source_delta)),
            "groups": int(len(source_group_means)),
            "micro_fadr_minus_pepd": source_point_micro,
            "micro_group_bootstrap_95ci": _interval(source_micro, level=0.95),
            "macro_fadr_minus_pepd": source_point_macro,
            "macro_group_bootstrap_95ci": _interval(source_macro, level=0.95),
        },
        "field_sensitivity": {
            "samples": int(len(field_delta)),
            "groups": int(len(field_group_means)),
            "micro_fadr_minus_pepd": field_point_micro,
            "micro_group_bootstrap_95ci": _interval(field_micro, level=0.95),
            "macro_fadr_minus_pepd": field_point_macro,
            "macro_group_bootstrap_95ci": _interval(field_macro, level=0.95),
        },
        "domain_by_route_interaction": {
            "micro_effect": field_point_micro - source_point_micro,
            "micro_independent_group_bootstrap_95ci": _interval(
                interaction_micro, level=0.95
            ),
            "macro_effect": field_point_macro - source_point_macro,
            "macro_independent_group_bootstrap_95ci": _interval(
                interaction_macro, level=0.95
            ),
        },
        "bootstrap": {
            "iterations_per_domain": int(iterations),
            "source_seed": int(seed),
            "field_seed": int(seed + 1),
            "unit": "group_id independently within each domain",
            "interval": "percentile",
        },
        "claim_boundary": "post-hoc descriptive domain interaction; not a confirmatory causal estimand",
    }


def build_report(
    inputs: AuditInputs = DEFAULT_INPUTS,
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(bootstrap_iterations > 0, "bootstrap iterations must be positive")
    input_identity = _validate_input_paths(inputs)
    uncertainty_raw = strict_json_load(inputs.uncertainty_audit)
    fadr_cohort = strict_json_load(inputs.fadr_cohort)
    train_comparison = strict_json_load(inputs.train_comparison)
    fadr_train_comparison = strict_json_load(inputs.fadr_train_comparison)
    sensitivity = strict_json_load(inputs.field_sensitivity)

    source_records, source_route = _load_source_records(
        inputs,
        train_comparison=train_comparison,
        fadr_comparison=fadr_train_comparison,
        fadr_cohort=fadr_cohort,
    )
    field_records, field_route_records = _load_field_records(
        inputs, sensitivity=sensitivity
    )
    field_route = _summarize_route_changes(field_route_records)
    source_helpful = [
        float(source_route[str(seed)]["helpful_fraction_among_changes"])
        for seed in FADR_SEEDS
    ]
    source_harmful = [
        float(source_route[str(seed)]["harmful_fraction_among_changes"])
        for seed in FADR_SEEDS
    ]
    route_reversal = {
        "comparison": "FADR versus PEPD-only on identical rows",
        "source_by_fadr_seed": source_route,
        "source_seed_mean_helpful_fraction_among_changes": float(
            np.mean(source_helpful)
        ),
        "source_seed_mean_harmful_fraction_among_changes": float(
            np.mean(source_harmful)
        ),
        "field_sensitivity": field_route,
        "helpful_fraction_shift_field_minus_source_seed_mean": float(
            field_route["helpful_fraction_among_changes"] - np.mean(source_helpful)
        ),
        "claim_boundary": "field route statistics are post-hoc descriptive sensitivity evidence",
    }
    field_comparisons = _comparator_audit(
        field_records,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 100,
    )
    interaction = _domain_interaction(
        source_records,
        field_records,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 200,
    )
    uncertainty = _extract_uncertainty(uncertainty_raw)

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "analysis_role": "post-hoc aggregate strengthening audit of frozen predictions; no method selection",
        "scope": {
            "training_performed": False,
            "inference_performed": False,
            "threshold_or_model_selection_performed": False,
            "field_prediction_caches_read": True,
            "field_images_read": 0,
            "public_samples_read": 0,
            "test_samples_read": 0,
            "sealed_samples_read": 0,
            "input_policy": "explicit JSON/JSONL artifact allowlist; image/data/public/test/sealed paths rejected",
        },
        "cohorts": {
            "source": {"samples": 4380, "groups": 197, "fadr_seeds": list(FADR_SEEDS)},
            "field_sensitivity": {
                "samples": 725,
                "groups": 18,
                "excluded_groups": list(
                    (sensitivity.get("denominators") or {})[
                        "excluded_confirmatory_group_ids"
                    ]
                ),
                "analysis_status": sensitivity["analysis_status"],
            },
        },
        "uncertainty_calibration": uncertainty,
        "fadr_route_reversal": route_reversal,
        "field_group_robustness": field_comparisons,
        "source_to_field_route_interaction": interaction,
        "inputs": input_identity,
        "generator": {
            "path": _relative(Path(__file__)),
            "sha256": _script_sha256(),
            "strict_json_protocol": STRICT_JSON_PROTOCOL,
            "strict_json_sha256": strict_json_source_sha256(),
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "interpretation_boundary": [
            "The 725-image field cohort remains a post-hoc, prediction-independent leakage sensitivity set.",
            "The source-to-field interaction is descriptive and does not restore confirmatory status.",
            "Cross-conformal uncertainty coverage is restricted to grouped SyncG/train OOF.",
            "No result in this audit authorizes model, route, seed, threshold, or cohort selection.",
        ],
    }


def build_verification(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": VERIFICATION_PROTOCOL,
        "status": "verified",
        "summary_protocol": summary.get("protocol"),
        "summary_sha256": hashlib.sha256(_canonical_bytes(summary)).hexdigest(),
        "input_sha256": {
            name: value["sha256"] for name, value in (summary.get("inputs") or {}).items()
        },
        "generator": dict(summary.get("generator") or {}),
        "verified_boundaries": {
            "json_jsonl_only": True,
            "field_images_read": 0,
            "public_samples_read": 0,
            "test_samples_read": 0,
            "sealed_samples_read": 0,
            "training_or_inference_performed": False,
        },
    }


def _write_idempotent(path: Path, value: Mapping[str, Any]) -> bool:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    if destination.exists():
        if destination.read_bytes() == payload:
            return False
        raise FileExistsError(
            f"{destination} already exists with different bytes; refusing to overwrite"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() == payload:
                return False
            raise FileExistsError(
                f"{destination} appeared with different bytes; refusing to overwrite"
            )
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _verify_existing(path: Path, expected: Mapping[str, Any]) -> None:
    observed = strict_json_load(path)
    _require(observed == expected, f"{path} differs from deterministic regeneration")
    _require(
        path.read_bytes() == _canonical_bytes(expected),
        f"{path} is semantically equal but not canonical",
    )


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    summary_path = output_root / "summary.json"
    verification_path = output_root / "verification.json"
    summary = build_report(
        DEFAULT_INPUTS,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    verification = build_verification(summary)
    if args.verify_only:
        _verify_existing(summary_path, summary)
        _verify_existing(verification_path, verification)
    else:
        _write_idempotent(summary_path, summary)
        _write_idempotent(verification_path, verification)
    print(
        json.dumps(
            {
                "status": "verified" if args.verify_only else "complete",
                "summary": str(summary_path),
                "verification": str(verification_path),
                "field_samples": summary["cohorts"]["field_sensitivity"]["samples"],
                "source_samples": summary["cohorts"]["source"]["samples"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
