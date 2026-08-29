"""Export unified, non-identifying Industrial-1395 result records.

Restricted images and per-sample predictions remain excluded. The three
private source partitions are combined into one 1,395-image cohort for every
reported statistic. Original acquisition-group names are replaced by one
global set of stable aliases so the 52-cluster bootstrap remains reproducible
without publishing source-partition labels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


SEEDS: Final[tuple[int, ...]] = (20_262_020, 20_262_021, 20_262_022)
EXPECTED_SAMPLES: Final[int] = 1_395
EXPECTED_GROUPS: Final[int] = 52
SOURCE_PARTITIONS: Final[dict[str, tuple[int, int]]] = {
    "field_gauge_external_roi": (147, 21),
    "field_gauge_roi_test_a": (434, 11),
    "field_gauge_roi_test_b": (814, 20),
}
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
SCOPE_CONDITIONS: Final[dict[str, tuple[str, ...]]] = {
    "all_conditions": CONDITIONS,
    **{condition: (condition,) for condition in CONDITIONS},
    "projective_pooled": (
        "perspective_moderate",
        "perspective_severe",
        "combined_severe",
    ),
}
SCOPES: Final[tuple[str, ...]] = tuple(SCOPE_CONDITIONS)
METRICS: Final[tuple[str, ...]] = (
    "coverage",
    "nmae",
    "rmse",
    "acc_at_1_percent",
    "acc_at_2_percent",
    "acc_at_5_percent",
    "p95_absolute_error",
    "p99_absolute_error",
    "maximum_absolute_error",
)
EXTERNAL_MODELS: Final[dict[str, tuple[str, str]]] = {
    "resnet18": ("factorial", "resnet18_direct"),
    "efficientnet_b0": ("lightweight_baselines", "efficientnet_b0"),
    "mobilenet_v3_large": ("lightweight_baselines", "mobilenet_v3_large"),
}
METHODS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("GeoPRR-Net", "candidate", "mett", "dual_view"),
    ("GeoPRR raw anchor", "candidate", "raw_anchor", "raw"),
    ("GeoPRR normalized endpoint", "candidate", "sarn_endpoint", "sarn_v2"),
    ("Raw ResNet-18", "external", "resnet18", "raw"),
    ("SARN-v2 + ResNet-18", "external", "resnet18", "sarn_v2"),
    ("Raw EfficientNet-B0", "external", "efficientnet_b0", "raw"),
    ("SARN-v2 + EfficientNet-B0", "external", "efficientnet_b0", "sarn_v2"),
    ("Raw MobileNetV3-Large", "external", "mobilenet_v3_large", "raw"),
    ("SARN-v2 + MobileNetV3-Large", "external", "mobilenet_v3_large", "sarn_v2"),
)
COMMON_FIELDS: Final[tuple[str, ...]] = (
    "method",
    "seed",
    "preprocessing",
    "cohort",
    "cohort_name",
    "cohort_images",
    "group_unit",
    "scope",
    "conditions",
)
GROUP_FIELDS: Final[tuple[str, ...]] = (
    *COMMON_FIELDS,
    "group_id",
    "rows",
    *METRICS,
)
COHORT_FIELDS: Final[tuple[str, ...]] = (
    *COMMON_FIELDS,
    "groups",
    "rows",
    *(f"pooled_{metric}" for metric in METRICS),
    *(f"group_macro_{metric}" for metric in METRICS),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, dict), "Industrial result is malformed")
    _require(payload.get("status") == "complete", "Industrial result is incomplete")
    _require(
        payload.get("protocol")
        == "unified_pointer_reader_industrial_1395_frozen_evaluation_v1",
        "Industrial protocol differs",
    )
    scope = payload.get("scope")
    _require(isinstance(scope, Mapping), "Industrial scope is missing")
    _require(int(scope.get("samples")) == EXPECTED_SAMPLES, "Industrial sample count differs")
    _require(scope.get("industrial_images_test_only") is True, "Industrial test-only scope differs")
    _require(
        scope.get("training_or_adaptation_during_evaluation") is False,
        "Industrial adaptation was reported",
    )
    return payload


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot calculate a quantile of an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metrics(errors: Sequence[float], passed: Sequence[bool]) -> dict[str, float]:
    _require(len(errors) == len(passed) and bool(errors), "Industrial metric rows are invalid")
    _require(all(math.isfinite(value) for value in errors), "Industrial error is non-finite")
    total = len(errors)
    return {
        "coverage": sum(passed) / total,
        "nmae": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "acc_at_1_percent": sum(value <= 0.01 for value in errors) / total,
        "acc_at_2_percent": sum(value <= 0.02 for value in errors) / total,
        "acc_at_5_percent": sum(value <= 0.05 for value in errors) / total,
        "p95_absolute_error": _quantile(errors, 0.95),
        "p99_absolute_error": _quantile(errors, 0.99),
        "maximum_absolute_error": max(errors),
    }


class _GzipCsvWriter:
    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.path.open("wb")
        compressed = gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
        )
        self._text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self._text, fieldnames=tuple(fields), lineterminator="\n"
        )
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(dict(row))
        self.rows += 1

    def close(self) -> None:
        self._text.flush()
        self._text.close()

    def __enter__(self) -> "_GzipCsvWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _source_rows(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    output: list[tuple[str, Mapping[str, Any]]] = []
    _require(set(payload["datasets"]) == set(SOURCE_PARTITIONS), "Industrial source roster differs")
    for source in sorted(SOURCE_PARTITIONS):
        expected_samples, expected_groups = SOURCE_PARTITIONS[source]
        dataset = payload["datasets"][source]
        metadata = dataset["dataset"]
        rows = dataset["per_sample_condition"]
        _require(int(metadata["samples"]) == expected_samples, "Industrial source sample count differs")
        _require(int(metadata["groups"]) == expected_groups, "Industrial source group count differs")
        _require(len(rows) == expected_samples * len(CONDITIONS), "Industrial source row count differs")
        output.extend((source, row) for row in rows)
    return output


def _row_key(source: str, row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return source, str(row["sample_id"]), str(row["condition"]), str(row["group_id"])


def _global_group_aliases(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[tuple[str, str], str]:
    original = sorted({(source, str(row["group_id"])) for source, row in rows})
    _require(len(original) == EXPECTED_GROUPS, "Industrial global group count differs")
    return {
        identity: f"group_{index:03d}"
        for index, identity in enumerate(original, start=1)
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    source_path = Path(path).resolve()
    _require(source_path.is_file(), f"Industrial external ledger is missing: {source_path}")
    with source_path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_external_predictions(
    external_root: Path,
) -> dict[tuple[str, str, int], dict[tuple[str, str, str], Mapping[str, Any]]]:
    root = Path(external_root).resolve()
    output: dict[
        tuple[str, str, int],
        dict[tuple[str, str, str], Mapping[str, Any]],
    ] = {}
    for model_key, (evaluation, model_directory) in EXTERNAL_MODELS.items():
        for preprocessing in ("raw", "sarn_v2"):
            for seed in SEEDS:
                predictions: dict[tuple[str, str, str], Mapping[str, Any]] = {}
                for source, (samples, _groups) in SOURCE_PARTITIONS.items():
                    path = (
                        root
                        / evaluation
                        / "evaluation"
                        / preprocessing
                        / source
                        / model_directory
                        / f"seed_{seed}"
                        / "predictions.jsonl"
                    )
                    rows = _load_jsonl(path)
                    _require(
                        len(rows) == samples * len(CONDITIONS),
                        "Industrial external ledger denominator differs",
                    )
                    for row in rows:
                        key = (source, str(row["sample_id"]), str(row["condition"]))
                        _require(key not in predictions, "duplicate Industrial external prediction")
                        _require(int(row["robustness_seed"]) == 20_260_720, "robustness seed differs")
                        predictions[key] = row
                _require(
                    len(predictions) == EXPECTED_SAMPLES * len(CONDITIONS),
                    "Industrial external unified denominator differs",
                )
                output[(model_key, preprocessing, seed)] = predictions
    return output


def _candidate_result(
    row: Mapping[str, Any],
    *,
    source_key: str,
) -> Mapping[str, Any]:
    result = row["candidate"][source_key]
    _require(isinstance(result, Mapping), "Industrial result row is missing")
    return result


def _collect_records(
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    aliases: Mapping[tuple[str, str], str],
    external_predictions: Mapping[
        tuple[str, str, int],
        Mapping[tuple[str, str, str], Mapping[str, Any]],
    ],
    *,
    scope: str,
    source_kind: str,
    source_key: str,
    preprocessing: str,
    seed: int,
) -> list[tuple[str, float, bool]]:
    selected_conditions = set(SCOPE_CONDITIONS[scope])
    records: list[tuple[str, float, bool]] = []
    for source, row in rows:
        if str(row["condition"]) not in selected_conditions:
            continue
        if source_kind == "candidate":
            result = _candidate_result(row, source_key=source_key)
            error = float(result["absolute_error"])
            passed = str(result["status"]) == "pass"
        else:
            key = (source, str(row["sample_id"]), str(row["condition"]))
            result = external_predictions[(source_key, preprocessing, seed)][key]
            passed = str(result["status"]) == "pass"
            error = (
                abs(float(result["normalized_progress"]) - float(row["normalized_target"]))
                if passed
                else 1.0
            )
        records.append(
            (
                aliases[(source, str(row["group_id"]))],
                error,
                passed,
            )
        )
    expected_rows = EXPECTED_SAMPLES * len(selected_conditions)
    _require(len(records) == expected_rows, "Industrial unified denominator differs")
    return records


def _summarize_records(
    records: Sequence[tuple[str, float, bool]],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
    pooled = _metrics([row[1] for row in records], [row[2] for row in records])
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for group_id, error, passed in records:
        grouped[group_id].append((error, passed))
    _require(len(grouped) == EXPECTED_GROUPS, "Industrial summary group count differs")
    per_group: dict[str, dict[str, Any]] = {}
    for group_id in sorted(grouped):
        values = grouped[group_id]
        per_group[group_id] = {
            "rows": len(values),
            **_metrics([value[0] for value in values], [value[1] for value in values]),
        }
    _require(
        sum(int(value["rows"]) for value in per_group.values()) == len(records),
        "Industrial groups do not recover the unified denominator",
    )
    macro = {
        metric: statistics.fmean(float(value[metric]) for value in per_group.values())
        for metric in METRICS
    }
    return pooled, macro, per_group


def _unify_public_summary(path: Path) -> None:
    summary_path = Path(path).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    industrial = payload["industrial_1395_aggregates_only"]
    for method in industrial["methods"].values():
        if isinstance(method, dict):
            method.pop("datasets", None)
    for comparison in industrial["paired"].values():
        comparison["group_unit"] = "acquisition cluster"
    payload["bootstrap"]["industrial_unit"] = "Industrial-1395 acquisition cluster"
    industrial["cohort"] = {
        "name": "Industrial-1395",
        "images": EXPECTED_SAMPLES,
        "conditions": len(CONDITIONS),
        "rows_per_seed": EXPECTED_SAMPLES * len(CONDITIONS),
        "acquisition_groups": EXPECTED_GROUPS,
        "reporting": "one unified cohort; no source-partition statistics",
    }
    industrial["note"] = (
        "Restricted images and per-sample ledgers are excluded; all published "
        "statistics pool the complete Industrial-1395 cohort."
    )
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_industrial_group_metrics(
    *,
    run_root: Path,
    external_root: Path,
    group_output: Path,
    cohort_output: Path,
    audit_output: Path,
    public_summary: Path | None = None,
) -> dict[str, Any]:
    payloads = {
        seed: _load(Path(run_root) / f"seed_{seed}/full/industrial.json")
        for seed in SEEDS
    }
    for seed, payload in payloads.items():
        _require(int(payload["model"]["seed"]) == seed, "Industrial model seed differs")

    source_rows = {seed: _source_rows(payload) for seed, payload in payloads.items()}
    reference_keys = [_row_key(source, row) for source, row in source_rows[SEEDS[0]]]
    for seed in SEEDS[1:]:
        observed = [_row_key(source, row) for source, row in source_rows[seed]]
        _require(observed == reference_keys, "Industrial row roster differs across seeds")
    aliases = _global_group_aliases(source_rows[SEEDS[0]])
    external_predictions = _load_external_predictions(external_root)
    external_roster = {
        (source, str(row["sample_id"]), str(row["condition"]))
        for source, row in source_rows[SEEDS[0]]
    }
    for predictions in external_predictions.values():
        _require(set(predictions) == external_roster, "Industrial external roster differs")

    # The independently loaded EfficientNet ledgers must exactly recover the
    # predictions already embedded in the unified-reader evaluation output.
    for source, row in source_rows[SEEDS[0]]:
        key = (source, str(row["sample_id"]), str(row["condition"]))
        target = float(row["normalized_target"])
        for preprocessing in ("raw", "sarn_v2"):
            for seed in SEEDS:
                embedded = row["efficientnet_b0"][preprocessing][str(seed)]
                external = external_predictions[("efficientnet_b0", preprocessing, seed)][key]
                external_passed = str(external["status"]) == "pass"
                external_error = (
                    abs(float(external["normalized_progress"]) - target)
                    if external_passed
                    else 1.0
                )
                _require(
                    external_passed == (str(embedded["status"]) == "pass")
                    and abs(float(external["normalized_progress"]) - float(embedded["prediction"])) <= 1e-12
                    and abs(external_error - float(embedded["absolute_error"])) <= 1e-12,
                    "Industrial external EfficientNet ledger differs from embedded results",
                )

    cohort_rows: list[dict[str, Any]] = []
    with _GzipCsvWriter(group_output, GROUP_FIELDS) as group_writer:
        for method, source_kind, source_key, preprocessing in METHODS:
            for seed in SEEDS:
                rows = source_rows[seed] if source_kind == "candidate" else source_rows[SEEDS[0]]
                for scope in SCOPES:
                    records = _collect_records(
                        rows,
                        aliases,
                        external_predictions,
                        scope=scope,
                        source_kind=source_kind,
                        source_key=source_key,
                        preprocessing=preprocessing,
                        seed=seed,
                    )
                    pooled, macro, per_group = _summarize_records(records)
                    common = {
                        "method": method,
                        "seed": seed,
                        "preprocessing": preprocessing,
                        "cohort": "industrial_1395",
                        "cohort_name": "Industrial-1395",
                        "cohort_images": EXPECTED_SAMPLES,
                        "group_unit": "acquisition cluster",
                        "scope": scope,
                        "conditions": ";".join(SCOPE_CONDITIONS[scope]),
                    }
                    for group_id, values in per_group.items():
                        group_writer.write(
                            {
                                **common,
                                "group_id": group_id,
                                "rows": int(values["rows"]),
                                **{metric: float(values[metric]) for metric in METRICS},
                            }
                        )
                    cohort_rows.append(
                        {
                            **common,
                            "groups": EXPECTED_GROUPS,
                            "rows": len(records),
                            **{f"pooled_{metric}": float(pooled[metric]) for metric in METRICS},
                            **{f"group_macro_{metric}": float(macro[metric]) for metric in METRICS},
                        }
                    )
        group_rows = group_writer.rows

    cohort_path = Path(cohort_output).resolve()
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    with cohort_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COHORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cohort_rows)

    expected_group_rows = len(METHODS) * len(SEEDS) * len(SCOPES) * EXPECTED_GROUPS
    expected_cohort_rows = len(METHODS) * len(SEEDS) * len(SCOPES)
    _require(group_rows == expected_group_rows, "Industrial public group row count differs")
    _require(len(cohort_rows) == expected_cohort_rows, "Industrial public cohort row count differs")
    if public_summary is not None:
        _unify_public_summary(public_summary)

    audit = {
        "schema_version": 2,
        "status": "pass",
        "cohort": "Industrial-1395",
        "samples": EXPECTED_SAMPLES,
        "conditions": len(CONDITIONS),
        "rows_per_seed": EXPECTED_SAMPLES * len(CONDITIONS),
        "groups": EXPECTED_GROUPS,
        "group_rows": group_rows,
        "cohort_summary_rows": len(cohort_rows),
        "methods": [method[0] for method in METHODS],
        "external_model_families": [
            "ResNet-18",
            "EfficientNet-B0",
            "MobileNetV3-Large",
        ],
        "seeds": list(SEEDS),
        "scopes": list(SCOPES),
        "reporting": "one unified cohort; no source-partition statistics",
        "source_partition_labels_released": False,
        "original_group_ids_released": False,
        "group_alias_rule": "one global group_001 ... group_052 roster",
        "restricted_images_or_per_sample_records_released": False,
        "all_group_rows_recover_unified_denominators": True,
        "all_group_macros_recomputed": True,
        "training_or_adaptation_during_evaluation": False,
    }
    audit_path = Path(audit_output).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/unified_pointer_reader"),
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        required=True,
        help="Root containing factorial/ and lightweight_baselines/ evaluation ledgers.",
    )
    parser.add_argument("--group-output", type=Path, required=True)
    parser.add_argument(
        "--cohort-output",
        "--dataset-output",
        dest="cohort_output",
        type=Path,
        required=True,
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    audit = export_industrial_group_metrics(
        run_root=args.run_root,
        external_root=args.external_root,
        group_output=args.group_output,
        cohort_output=args.cohort_output,
        audit_output=args.audit_output,
        public_summary=args.public_summary,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_industrial_group_metrics"]
