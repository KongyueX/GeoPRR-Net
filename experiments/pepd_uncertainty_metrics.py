"""Pre-frozen group-aware uncertainty diagnostics for the PEPD cohort.

The physical meter group is the analysis and bootstrap unit.  Invalid
directions remain in every denominator and receive the pre-declared worst-case
angular error of 180 degrees.  Risk--coverage ranking retains complete groups
in ascending mean predicted standard deviation; equal-score groups enter as
one atomic tie block, so a homoscedastic predictor cannot gain an artificial
ranking from sample order.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.pepd_convergence_protocol import (
    UNCERTAINTY_MECHANISM_ARMS,
    uncertainty_mechanism_arm,
)


UNCERTAINTY_DIAGNOSTIC_PROTOCOL = (
    "pepd_grouped_uncertainty_risk_coverage_calibration_v1"
)
INVALID_DIRECTION_ERROR_DEGREES = 180.0
RISK_COVERAGE_LEVELS = (1.0, 0.75, 0.50, 0.25, 0.10)
CALIBRATION_THRESHOLDS_DEGREES = (1.0, 3.0, 5.0)
PRIMARY_CALIBRATION_THRESHOLD_DEGREES = 3.0


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _group_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    uncertainty_available: bool,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("uncertainty diagnostics received no rows")
    grouped: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("group_id", ""))
        if not group_id:
            raise ValueError("uncertainty row has no group_id")
        valid = row.get("valid") is True
        if valid:
            error = _finite_float(
                row.get("angle_error_degrees"),
                name="angle_error_degrees",
            )
            if not 0.0 <= error <= INVALID_DIRECTION_ERROR_DEGREES + 1e-6:
                raise ValueError("valid angular error is outside [0, 180]")
            error = min(error, INVALID_DIRECTION_ERROR_DEGREES)
        else:
            error = INVALID_DIRECTION_ERROR_DEGREES
        sample: dict[str, float | bool] = {
            "valid": valid,
            "error": error,
        }
        if uncertainty_available:
            std = _finite_float(
                row.get("angle_std_degrees"),
                name="angle_std_degrees",
            )
            if std <= 0.0:
                raise ValueError("angle_std_degrees must be positive")
            error_rad = math.radians(error)
            std_rad = math.radians(std)
            variance = max(std_rad * std_rad, 1e-12)
            sample["std"] = std
            sample["nll"] = 0.5 * (
                error_rad * error_rad / variance + math.log(variance)
            )
            for threshold in CALIBRATION_THRESHOLDS_DEGREES:
                sample[f"predicted_{threshold:g}"] = math.erf(
                    math.radians(threshold)
                    / (math.sqrt(2.0) * std_rad)
                )
                sample[f"observed_{threshold:g}"] = error <= threshold
        grouped[group_id].append(sample)

    records: list[dict[str, Any]] = []
    for group_id, samples in sorted(grouped.items()):
        errors = np.asarray(
            [float(sample["error"]) for sample in samples],
            dtype=np.float64,
        )
        record: dict[str, Any] = {
            "group_id": group_id,
            "samples": len(samples),
            "invalid_directions": sum(
                sample["valid"] is not True for sample in samples
            ),
            "risk_degrees": float(np.mean(errors)),
        }
        if uncertainty_available:
            record["uncertainty_degrees"] = float(
                np.mean([float(sample["std"]) for sample in samples])
            )
            record["angular_nll"] = float(
                np.mean([float(sample["nll"]) for sample in samples])
            )
            record["thresholds"] = {
                f"{threshold:g}": {
                    "predicted_coverage": float(
                        np.mean(
                            [
                                float(sample[f"predicted_{threshold:g}"])
                                for sample in samples
                            ]
                        )
                    ),
                    "observed_coverage": float(
                        np.mean(
                            [
                                bool(sample[f"observed_{threshold:g}"])
                                for sample in samples
                            ]
                        )
                    ),
                }
                for threshold in CALIBRATION_THRESHOLDS_DEGREES
            }
        records.append(record)
    return records


def _risk_coverage(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        records,
        key=lambda row: (
            float(row["uncertainty_degrees"]),
            str(row.get("group_id", "")),
        ),
    )
    if not ordered:
        raise ValueError("risk--coverage requires at least one group")
    points: list[dict[str, float | int]] = []
    retained_risk = 0.0
    retained_groups = 0
    index = 0
    while index < len(ordered):
        score = float(ordered[index]["uncertainty_degrees"])
        end = index + 1
        while (
            end < len(ordered)
            and float(ordered[end]["uncertainty_degrees"]) == score
        ):
            end += 1
        block = ordered[index:end]
        retained_risk += sum(float(row["risk_degrees"]) for row in block)
        retained_groups += len(block)
        points.append(
            {
                "coverage": retained_groups / len(ordered),
                "risk_degrees": retained_risk / retained_groups,
                "uncertainty_threshold_degrees": score,
                "retained_groups": retained_groups,
            }
        )
        index = end
    previous_coverage = 0.0
    aurc = 0.0
    for point in points:
        coverage = float(point["coverage"])
        aurc += float(point["risk_degrees"]) * (
            coverage - previous_coverage
        )
        previous_coverage = coverage
    full_risk = float(points[-1]["risk_degrees"])
    distinct_scores = len(points)
    retained: dict[str, Any] = {}
    for nominal in RISK_COVERAGE_LEVELS:
        selected = next(
            point for point in points if float(point["coverage"]) >= nominal
        )
        retained[f"{nominal:g}"] = {
            "nominal_group_coverage": nominal,
            "actual_group_coverage": float(selected["coverage"]),
            "risk_degrees": float(selected["risk_degrees"]),
            "retained_groups": int(selected["retained_groups"]),
        }
    return {
        "ranking_unit": "physical_meter_group",
        "ranking_score": "group_mean_predicted_angle_std_degrees",
        "lower_score_retained_first": True,
        "tie_policy": "equal-score groups enter atomically",
        "integration": "right-continuous step integral over group coverage",
        "groups": len(ordered),
        "distinct_uncertainty_scores": distinct_scores,
        "group_ranking_available": distinct_scores > 1,
        "aurc_degrees": aurc,
        "full_coverage_macro_group_risk_degrees": full_risk,
        "aurc_skill_degrees": full_risk - aurc,
        "positive_aurc_skill_means_useful_ranking": True,
        "risk_at_predeclared_group_coverages": retained,
        "curve": points,
    }


def _calibration(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    macro_nll = float(
        np.mean([float(row["angular_nll"]) for row in records])
    )
    thresholds: dict[str, Any] = {}
    for threshold in CALIBRATION_THRESHOLDS_DEGREES:
        key = f"{threshold:g}"
        predicted = float(
            np.mean(
                [
                    float(row["thresholds"][key]["predicted_coverage"])
                    for row in records
                ]
            )
        )
        observed = float(
            np.mean(
                [
                    float(row["thresholds"][key]["observed_coverage"])
                    for row in records
                ]
            )
        )
        thresholds[key] = {
            "threshold_degrees": threshold,
            "predicted_coverage_macro_group": predicted,
            "observed_coverage_macro_group": observed,
            "signed_gap_predicted_minus_observed": predicted - observed,
            "absolute_gap": abs(predicted - observed),
        }
    return {
        "calibration_unit": "physical_meter_group_macro",
        "angular_nll_radians_macro_group": macro_nll,
        "threshold_coverage_reliability": thresholds,
        "primary_threshold_degrees": (
            PRIMARY_CALIBRATION_THRESHOLD_DEGREES
        ),
    }


def grouped_uncertainty_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    uncertainty_mechanism_arm(mode)
    available = mode != "no_angular_nll"
    records = _group_records(rows, uncertainty_available=available)
    total_samples = sum(int(row["samples"]) for row in records)
    invalid = sum(int(row["invalid_directions"]) for row in records)
    base = {
        "protocol": UNCERTAINTY_DIAGNOSTIC_PROTOCOL,
        "mode": mode,
        "groups": len(records),
        "samples": total_samples,
        "invalid_directions": invalid,
        "direction_coverage": (total_samples - invalid) / total_samples,
        "failure_denominator_policy": (
            "all samples retained; invalid direction receives 180-degree error"
        ),
        "uncertainty_available": available,
    }
    if not available:
        return base | {
            "risk_coverage": {
                "available": False,
                "reason": "no trained uncertainty score",
            },
            "calibration": {
                "available": False,
                "reason": "no trained uncertainty distribution",
            },
        }
    return base | {
        "risk_coverage": {
            "available": True,
        }
        | _risk_coverage(records),
        "calibration": {
            "available": True,
        }
        | _calibration(records),
    }


def grouped_uncertainty_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    uncertainty_mechanism_arm(mode)
    if int(iterations) <= 0:
        raise ValueError("bootstrap iterations must be positive")
    available = mode != "no_angular_nll"
    records = _group_records(rows, uncertainty_available=available)
    if not available:
        return {
            "available": False,
            "reason": "no trained uncertainty score/distribution",
            "iterations": int(iterations),
            "seed": int(seed),
            "resampling_unit": "physical_meter_group",
        }
    generator = np.random.default_rng(seed)
    aurc = np.empty(iterations, dtype=np.float64)
    skill = np.empty(iterations, dtype=np.float64)
    nll = np.empty(iterations, dtype=np.float64)
    gaps = {
        f"{threshold:g}": np.empty(iterations, dtype=np.float64)
        for threshold in CALIBRATION_THRESHOLDS_DEGREES
    }
    for index in range(iterations):
        selected = generator.integers(
            0,
            len(records),
            size=len(records),
        )
        draw = [records[int(position)] for position in selected]
        risk = _risk_coverage(draw)
        calibration = _calibration(draw)
        aurc[index] = float(risk["aurc_degrees"])
        skill[index] = float(risk["aurc_skill_degrees"])
        nll[index] = float(
            calibration["angular_nll_radians_macro_group"]
        )
        for key in gaps:
            gaps[key][index] = float(
                calibration["threshold_coverage_reliability"][key][
                    "signed_gap_predicted_minus_observed"
                ]
            )

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "available": True,
        "iterations": int(iterations),
        "seed": int(seed),
        "resampling_unit": "physical_meter_group",
        "aurc_degrees_95ci": interval(aurc),
        "aurc_skill_degrees_95ci": interval(skill),
        "angular_nll_radians_macro_group_95ci": interval(nll),
        "threshold_signed_gap_95ci": {
            key: interval(values) for key, values in gaps.items()
        },
    }


def uncertainty_metric_arm_names() -> tuple[str, ...]:
    return UNCERTAINTY_MECHANISM_ARMS
