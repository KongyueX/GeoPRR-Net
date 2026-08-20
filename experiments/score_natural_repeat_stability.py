"""Score clean-only natural repeated-capture stability on the physical XM2 cohort.

The cohort is formed without model outputs.  Unified-pool sample IDs are joined
back to the frozen XM2 manifest, then grouped by the exact pair
``(physical instrument group_id, ground_truth)``.  A unit is retained only when
it contains at least two distinct source images.  Multiple materialized crops
from one source image are averaged before within-unit spread is calculated.

The scorer accepts clean prediction JSONL files for the three frozen DB-GAR18
and matched DB-ResNet18 seeds.  It reports full-denominator NMAE, within-unit
population standard deviation and range, and physical-group cluster-bootstrap
confidence intervals for paired DB-GAR18 minus DB-ResNet18 effects.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final


PROTOCOL: Final[str] = "xm2_natural_repeat_stability_v1"
FAMILIES: Final[tuple[str, str]] = ("db_gar18", "db_resnet18")
DEFAULT_SEEDS: Final[tuple[int, int, int]] = (20262020, 20262021, 20262022)
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260811
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
PHYSICAL_GROUP_SOURCE: Final[str] = (
    "source_image directory interpreted as physical instrument identifier"
)


class RepeatStabilityError(RuntimeError):
    """An input violates the frozen repeated-capture scoring contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatStabilityError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSONL does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RepeatStabilityError(
                    f"invalid JSON at {source}:{line_number}"
                ) from exc
            _require(
                isinstance(value, Mapping),
                f"row is not an object at {source}:{line_number}",
            )
            rows.append(dict(value))
    _require(bool(rows), f"JSONL is empty: {source}")
    return rows


def _finite_number(value: Any, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _exact_number_key(value: Any, *, label: str) -> str:
    number = _finite_number(value, label=label)
    try:
        decimal = Decimal(str(number)).normalize()
    except InvalidOperation as exc:  # pragma: no cover - guarded above
        raise RepeatStabilityError(f"{label} is not a decimal") from exc
    if decimal == 0:
        return "0"
    return format(decimal, "f")


def _index_unique(
    rows: Iterable[Mapping[str, Any]],
    key: str,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        value = str(row.get(key) or "").strip()
        _require(bool(value), f"{label} row {index} has empty {key}")
        _require(value not in result, f"duplicate {label} {key}: {value}")
        result[value] = dict(row)
    return result


@dataclass(frozen=True)
class CohortRow:
    sample_id: str
    original_sample_id: str
    group_id: str
    ground_truth_key: str
    source_image: str
    target: float

    @property
    def unit_key(self) -> tuple[str, str]:
        return self.group_id, self.ground_truth_key


@dataclass(frozen=True)
class PredictionValue:
    passed: bool
    value: float | None


@dataclass(frozen=True)
class MethodIdentity:
    method: str
    family: str
    seed: int


def load_xm2_repeat_cohort(
    *,
    provenance_path: Path,
    labels_path: Path,
    xm2_manifest_path: Path,
) -> tuple[tuple[CohortRow, ...], dict[str, Any]]:
    """Join frozen manifests and retain exact-reading multi-image XM2 units."""

    provenance = _index_unique(
        _read_jsonl(provenance_path), "sample_id", label="provenance"
    )
    labels = _index_unique(_read_jsonl(labels_path), "sample_id", label="label")
    xm2 = _index_unique(
        _read_jsonl(xm2_manifest_path), "sample_id", label="XM2 manifest"
    )
    _require(set(provenance) == set(labels), "unified provenance/label roster drift")

    candidates: list[CohortRow] = []
    for sample_id in sorted(provenance):
        source = provenance[sample_id]
        original_id = str(source.get("original_sample_id") or "").strip()
        if original_id not in xm2:
            continue
        manifest = xm2[original_id]
        metadata = manifest.get("metadata")
        _require(isinstance(metadata, Mapping), f"XM2 metadata missing: {original_id}")
        _require(
            metadata.get("group_identity_source") == PHYSICAL_GROUP_SOURCE,
            f"XM2 physical-group evidence drift: {original_id}",
        )
        group_id = str(manifest.get("group_id") or "").strip()
        _require(
            group_id and group_id == str(source.get("original_group_id") or "").strip(),
            f"physical group drift: {sample_id}",
        )
        label = labels[sample_id]
        manifest_truth = _exact_number_key(
            manifest.get("ground_truth"), label=f"XM2 ground_truth[{original_id}]"
        )
        label_truth = _exact_number_key(
            label.get("ground_truth"), label=f"unified ground_truth[{sample_id}]"
        )
        _require(manifest_truth == label_truth, f"ground-truth drift: {sample_id}")
        target = _finite_number(
            label.get("normalized_progress"),
            label=f"normalized_progress[{sample_id}]",
        )
        _require(0.0 <= target <= 1.0, f"target outside [0,1]: {sample_id}")
        source_image = str(metadata.get("source_image") or "").strip()
        _require(bool(source_image), f"source image missing: {original_id}")
        candidates.append(
            CohortRow(
                sample_id=sample_id,
                original_sample_id=original_id,
                group_id=group_id,
                ground_truth_key=manifest_truth,
                source_image=source_image.replace("\\", "/"),
                target=target,
            )
        )

    _require(bool(candidates), "no physical XM2 rows joined into the unified pool")
    by_unit: dict[tuple[str, str], list[CohortRow]] = defaultdict(list)
    for row in candidates:
        by_unit[row.unit_key].append(row)
    retained_units = {
        key: rows
        for key, rows in by_unit.items()
        if len({row.source_image for row in rows}) >= 2
    }
    retained_ids = {
        row.sample_id for rows in retained_units.values() for row in rows
    }
    retained = tuple(row for row in candidates if row.sample_id in retained_ids)
    _require(bool(retained), "no exact-reading unit has two distinct source images")
    summary = {
        "joined_xm2_samples": len(candidates),
        "joined_physical_groups": len({row.group_id for row in candidates}),
        "joined_exact_reading_units": len(by_unit),
        "retained_samples": len(retained),
        "retained_physical_groups": len({row.group_id for row in retained}),
        "retained_exact_reading_units": len(retained_units),
        "retained_distinct_source_images": len(
            {row.source_image for row in retained}
        ),
        "unit_rule": "exact (physical group_id, ground_truth), >=2 distinct source_image",
        "same_source_image_rule": "average predictions before within-unit spread",
    }
    return retained, summary


_SEED_PATTERN = re.compile(r"seed[_-]?(\d{6,10})(?:\D|$)", re.IGNORECASE)


def _method_identity(method: str) -> MethodIdentity:
    lowered = method.lower()
    matches = [family for family in FAMILIES if family in lowered]
    _require(len(matches) == 1, f"cannot identify DB-18 family from method: {method}")
    seed_match = _SEED_PATTERN.search(lowered)
    _require(seed_match is not None, f"cannot identify training seed from method: {method}")
    return MethodIdentity(method=method, family=matches[0], seed=int(seed_match.group(1)))


def load_clean_predictions(
    paths: Sequence[Path],
    *,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[
    dict[tuple[str, int], dict[str, PredictionValue]],
    dict[tuple[str, int], str],
]:
    """Load exactly one clean method for every requested family/seed pair."""

    expected = {(family, int(seed)) for family in FAMILIES for seed in expected_seeds}
    values: dict[tuple[str, int], dict[str, PredictionValue]] = defaultdict(dict)
    names: dict[tuple[str, int], str] = {}
    for path in paths:
        for index, row in enumerate(_read_jsonl(path), 1):
            if row.get("condition") != "clean":
                continue
            method = str(row.get("method") or "").strip()
            _require(bool(method), f"prediction row {index} has empty method: {path}")
            identity = _method_identity(method)
            key = (identity.family, identity.seed)
            if key not in expected:
                continue
            previous_name = names.setdefault(key, method)
            _require(previous_name == method, f"multiple method names for {key}")
            sample_id = str(row.get("sample_id") or "").strip()
            _require(bool(sample_id), f"prediction row {index} has empty sample_id")
            _require(sample_id not in values[key], f"duplicate clean prediction: {key}/{sample_id}")
            status = str(row.get("status") or "").strip().lower()
            if status == "pass":
                prediction = _finite_number(
                    row.get("normalized_progress"),
                    label=f"prediction[{method}/{sample_id}]",
                )
                _require(
                    0.0 <= prediction <= 1.0,
                    f"prediction outside [0,1]: {method}/{sample_id}",
                )
                values[key][sample_id] = PredictionValue(True, prediction)
            else:
                values[key][sample_id] = PredictionValue(False, None)
    _require(set(values) == expected, f"prediction family/seed roster drift: {sorted(values)}")
    return dict(values), names


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile input is empty")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "aggregate input is empty")
    floats = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(floats)),
        "median": float(statistics.median(floats)),
        "p90": float(_percentile(floats, 0.90)),
    }


def _unit_predictions(
    rows: Sequence[CohortRow],
    predictions: Mapping[str, PredictionValue],
) -> list[float] | None:
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = predictions[row.sample_id]
        if not value.passed or value.value is None:
            return None
        by_source[row.source_image].append(value.value)
    return [statistics.fmean(values) for values in by_source.values()]


def score_one_method(
    cohort: Sequence[CohortRow],
    predictions: Mapping[str, PredictionValue],
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, float]]]:
    expected_ids = {row.sample_id for row in cohort}
    missing = expected_ids - set(predictions)
    _require(not missing, f"predictions miss {len(missing)} retained XM2 samples")
    errors: list[float] = []
    passed = 0
    for row in cohort:
        value = predictions[row.sample_id]
        if value.passed and value.value is not None:
            errors.append(abs(value.value - row.target))
            passed += 1
        else:
            errors.append(1.0)

    by_unit: dict[tuple[str, str], list[CohortRow]] = defaultdict(list)
    for row in cohort:
        by_unit[row.unit_key].append(row)
    details: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in sorted(by_unit.items()):
        values = _unit_predictions(rows, predictions)
        if values is None:
            continue
        details[key] = {
            "source_images": float(len(values)),
            "prediction_sd_population": float(statistics.pstdev(values)),
            "prediction_range": float(max(values) - min(values)),
        }
    sd_values = [value["prediction_sd_population"] for value in details.values()]
    ranges = [value["prediction_range"] for value in details.values()]
    summary = {
        "samples": len(cohort),
        "passed": passed,
        "coverage": float(passed / len(cohort)),
        "full_denominator_nmae": float(statistics.fmean(errors)),
        "failure_error": 1.0,
        "eligible_units": len(by_unit),
        "complete_stability_units": len(details),
        "within_unit_prediction_sd_population": _aggregate(sd_values),
        "within_unit_prediction_range": _aggregate(ranges),
    }
    return summary, details


def _seed_summary(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "seed summary is empty")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
    }


def _cluster_bootstrap(
    grouped_values: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(replicates >= 100, "bootstrap requires at least 100 replicates")
    groups = sorted(grouped_values)
    _require(len(groups) >= 2, "cluster bootstrap requires at least two groups")
    point_values = [value for group in groups for value in grouped_values[group]]
    point = float(statistics.fmean(point_values))
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(replicates):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        values = [value for group in sampled for value in grouped_values[group]]
        draws.append(float(statistics.fmean(values)))
    return {
        "point_estimate": point,
        "ci95": [float(_percentile(draws, 0.025)), float(_percentile(draws, 0.975))],
        "replicates": replicates,
        "seed": int(seed),
        "cluster_unit": "physical instrument group_id",
    }


def score_repeat_stability(
    *,
    provenance_path: Path,
    labels_path: Path,
    xm2_manifest_path: Path,
    prediction_paths: Sequence[Path],
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    cohort, cohort_summary = load_xm2_repeat_cohort(
        provenance_path=provenance_path,
        labels_path=labels_path,
        xm2_manifest_path=xm2_manifest_path,
    )
    predictions, method_names = load_clean_predictions(
        prediction_paths, expected_seeds=expected_seeds
    )
    method_scores: dict[tuple[str, int], dict[str, Any]] = {}
    method_units: dict[
        tuple[str, int], dict[tuple[str, str], dict[str, float]]
    ] = {}
    for key in sorted(predictions):
        summary, details = score_one_method(cohort, predictions[key])
        method_scores[key] = summary
        method_units[key] = details

    families: dict[str, Any] = {}
    for family in FAMILIES:
        per_seed: dict[str, Any] = {}
        for seed in expected_seeds:
            key = (family, int(seed))
            per_seed[str(seed)] = {
                "method": method_names[key],
                **method_scores[key],
            }
        families[family] = {
            "per_seed": per_seed,
            "three_seed": {
                "full_denominator_nmae": _seed_summary(
                    [method_scores[(family, int(seed))]["full_denominator_nmae"] for seed in expected_seeds]
                ),
                "mean_within_unit_prediction_sd_population": _seed_summary(
                    [method_scores[(family, int(seed))]["within_unit_prediction_sd_population"]["mean"] for seed in expected_seeds]
                ),
                "mean_within_unit_prediction_range": _seed_summary(
                    [method_scores[(family, int(seed))]["within_unit_prediction_range"]["mean"] for seed in expected_seeds]
                ),
            },
        }

    sample_error_difference: dict[str, list[float]] = defaultdict(list)
    for row in cohort:
        family_errors: dict[str, float] = {}
        for family in FAMILIES:
            seed_errors: list[float] = []
            for seed in expected_seeds:
                prediction = predictions[(family, int(seed))][row.sample_id]
                seed_errors.append(
                    abs(float(prediction.value) - row.target)
                    if prediction.passed and prediction.value is not None
                    else 1.0
                )
            family_errors[family] = statistics.fmean(seed_errors)
        sample_error_difference[row.group_id].append(
            family_errors["db_gar18"] - family_errors["db_resnet18"]
        )

    unit_sd_difference: dict[str, list[float]] = defaultdict(list)
    unit_range_difference: dict[str, list[float]] = defaultdict(list)
    all_unit_keys = sorted(
        set.intersection(
            *(set(method_units[(family, int(seed))]) for family in FAMILIES for seed in expected_seeds)
        )
    )
    for group_id, truth_key in all_unit_keys:
        family_sd: dict[str, float] = {}
        family_range: dict[str, float] = {}
        for family in FAMILIES:
            family_sd[family] = statistics.fmean(
                method_units[(family, int(seed))][(group_id, truth_key)][
                    "prediction_sd_population"
                ]
                for seed in expected_seeds
            )
            family_range[family] = statistics.fmean(
                method_units[(family, int(seed))][(group_id, truth_key)][
                    "prediction_range"
                ]
                for seed in expected_seeds
            )
        unit_sd_difference[group_id].append(
            family_sd["db_gar18"] - family_sd["db_resnet18"]
        )
        unit_range_difference[group_id].append(
            family_range["db_gar18"] - family_range["db_resnet18"]
        )

    comparison = {
        "effect_direction": "DB-GAR18 minus matched DB-ResNet18; negative favors DB-GAR18",
        "nmae_difference": _cluster_bootstrap(
            sample_error_difference,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "within_unit_sd_difference": _cluster_bootstrap(
            unit_sd_difference,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
        ),
        "within_unit_range_difference": _cluster_bootstrap(
            unit_range_difference,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2,
        ),
    }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "evaluation_name": "natural repeated-capture stability",
        "explicit_non_claims": [
            "not an angle or perspective-severity stratification",
            "not a claim that every repeated source image changes viewpoint",
        ],
        "condition": "clean",
        "cohort": cohort_summary,
        "families": families,
        "paired_group_bootstrap": comparison,
        "inputs": {
            "provenance": str(Path(provenance_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "xm2_manifest": str(Path(xm2_manifest_path).resolve()),
            "predictions": [str(Path(path).resolve()) for path in prediction_paths],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--xm2-manifest", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = score_repeat_stability(
        provenance_path=args.provenance,
        labels_path=args.labels,
        xm2_manifest_path=args.xm2_manifest,
        prediction_paths=args.prediction,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "retained_samples": result["cohort"]["retained_samples"],
                "retained_units": result["cohort"]["retained_exact_reading_units"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "RepeatStabilityError",
    "load_clean_predictions",
    "load_xm2_repeat_cohort",
    "score_one_method",
    "score_repeat_stability",
]
