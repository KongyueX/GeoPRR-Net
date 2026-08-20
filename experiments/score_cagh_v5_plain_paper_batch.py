"""Score the plain eight-method paper batch against a fixed SyncG validation roster.

This is an ordinary, post-inference research scorer.  It contains no key,
signature, approval, one-shot, or lifecycle machinery.  The prediction input
is label-free; labels are joined only from the explicitly supplied original
SyncG manifest and fixed validation-ID file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


PROTOCOL: Final[str] = "cagh_v5_plain_paper_score_v1"
EXPECTED_SAMPLES: Final[int] = 1_625
EXPECTED_GROUPS: Final[int] = 73
METHODS: Final[tuple[str, ...]] = (
    "full_seed_20262020",
    "full_seed_20262021",
    "full_seed_20262022",
    "control_no_pepd_residual_seed_20262020",
    "control_no_mask_residual_seed_20262020",
    "control_solver_core_seed_20262020",
    "vdn_official200_terminal_seed20",
    "original_transformer_legacy_auto_reference",
)
FULL_SEED_METHODS: Final[tuple[str, ...]] = METHODS[:3]
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
METRIC_NAMES: Final[tuple[str, ...]] = (
    "nmae",
    "coverage",
    "p50",
    "p90",
    "p95",
    "p99",
    "acc_at_1pct",
    "acc_at_2pct",
    "acc_at_5pct",
)
_FORBIDDEN_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ground_truth",
        "target",
        "normalized_target",
        "scale_start",
        "scale_end",
        "label",
        "labels",
    }
)


class PlainPaperScoreError(ValueError):
    """A source artifact violates the scoring contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlainPaperScoreError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    _require(value == value.strip() and bool(value), f"{label} is blank or padded")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlainPaperScoreError(f"{label} must be numeric") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlainPaperScoreError(
                    f"{label} line {line_number} is invalid JSON"
                ) from exc
            _require(
                isinstance(value, Mapping),
                f"{label} line {line_number} is not an object",
            )
            rows.append(dict(value))
    _require(bool(rows), f"{label} is empty")
    return rows


def load_validation_ids(path: Path) -> tuple[str, ...]:
    """Load a JSON list/object, JSONL roster, or one-ID-per-line text file."""

    source = Path(path).resolve()
    _require(source.is_file(), f"validation IDs do not exist: {source}")
    text = source.read_text(encoding="utf-8-sig")
    _require(bool(text.strip()), "validation IDs are empty")
    values: Any = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        values = value
    elif isinstance(value, Mapping):
        for key in (
            "validation_sample_ids",
            "outer_sample_ids",
            "sample_ids",
            "validation_ids",
        ):
            if key in value:
                values = value[key]
                break
        _require(values is not None, "validation-ID JSON object has no supported ID list")
    elif value is not None:
        raise PlainPaperScoreError("validation-ID JSON must be a list or object")
    else:
        values = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise PlainPaperScoreError(
                        f"validation IDs line {line_number} is invalid JSON"
                    ) from exc
                _require(isinstance(row, Mapping), "validation-ID JSONL row is not an object")
                values.append(row.get("sample_id"))
            else:
                values.append(stripped)
    _require(isinstance(values, list), "validation IDs must be a list")
    ids = tuple(_text(value, label=f"validation_ids[{index}]") for index, value in enumerate(values))
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    _require(not duplicates, f"duplicate validation sample_id(s): {duplicates[:5]}")
    _require(bool(ids), "validation IDs are empty")
    return ids


@dataclass(frozen=True, slots=True)
class Target:
    sample_id: str
    group_id: str
    normalized_target: float


def load_targets(manifest_path: Path, validation_ids: Sequence[str]) -> tuple[Target, ...]:
    wanted = set(validation_ids)
    _require(len(wanted) == len(validation_ids), "validation IDs are not unique")
    targets: dict[str, Target] = {}
    seen_manifest_ids: set[str] = set()
    for index, row in enumerate(_read_jsonl(manifest_path, label="SyncG manifest")):
        sample_id = _text(row.get("sample_id"), label=f"manifest[{index}].sample_id")
        _require(sample_id not in seen_manifest_ids, f"duplicate manifest sample_id: {sample_id}")
        seen_manifest_ids.add(sample_id)
        if sample_id not in wanted:
            continue
        group_id = _text(row.get("group_id"), label=f"manifest[{index}].group_id")
        ground_truth = _finite_float(
            row.get("ground_truth"), label=f"manifest[{index}].ground_truth"
        )
        scale_start = _finite_float(
            row.get("scale_start"), label=f"manifest[{index}].scale_start"
        )
        scale_end = _finite_float(
            row.get("scale_end"), label=f"manifest[{index}].scale_end"
        )
        _require(scale_end > scale_start, f"{sample_id}: invalid scale range")
        normalized = (ground_truth - scale_start) / (scale_end - scale_start)
        _require(
            -1e-12 <= normalized <= 1.0 + 1e-12,
            f"{sample_id}: normalized target outside [0,1]",
        )
        targets[sample_id] = Target(
            sample_id=sample_id,
            group_id=group_id,
            normalized_target=min(1.0, max(0.0, normalized)),
        )
    missing = sorted(wanted - set(targets))
    _require(not missing, f"validation IDs absent from manifest: {missing[:5]}")
    return tuple(targets[sample_id] for sample_id in validation_ids)


@dataclass(frozen=True, slots=True)
class Prediction:
    sample_id: str
    method: str
    condition: str
    passed: bool
    normalized_progress: float | None


def _prediction_paths(value: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(value, Path):
        paths = (value.resolve(),)
    else:
        paths = tuple(Path(path).resolve() for path in value)
    _require(bool(paths), "at least one prediction JSONL is required")
    _require(len(paths) == len(set(paths)), "duplicate prediction input path")
    return paths


def load_predictions(
    path: Path | Sequence[Path],
    *,
    targets: Mapping[str, Target],
    methods: Sequence[str],
    conditions: Sequence[str],
) -> dict[tuple[str, str, str], Prediction]:
    method_set = set(methods)
    condition_set = set(conditions)
    expected_samples = set(targets)
    predictions: dict[tuple[str, str, str], Prediction] = {}
    rows = (
        row
        for source_path in _prediction_paths(path)
        for row in _read_jsonl(source_path, label=f"predictions ({source_path})")
    )
    for index, row in enumerate(rows):
        leaked = sorted(set(row) & _FORBIDDEN_PREDICTION_KEYS)
        _require(not leaked, f"prediction[{index}] contains label field(s): {leaked}")
        sample_id = _text(row.get("sample_id"), label=f"prediction[{index}].sample_id")
        method = _text(row.get("method"), label=f"prediction[{index}].method")
        condition = _text(row.get("condition"), label=f"prediction[{index}].condition")
        _require(sample_id in expected_samples, f"unexpected prediction sample_id: {sample_id}")
        _require(method in method_set, f"unexpected prediction method: {method}")
        _require(condition in condition_set, f"unexpected prediction condition: {condition}")
        if "group_id" in row:
            group_id = _text(row["group_id"], label=f"prediction[{index}].group_id")
            _require(group_id == targets[sample_id].group_id, f"{sample_id}: group_id mismatch")
        status = _text(row.get("status"), label=f"prediction[{index}].status")
        _require(status in {"pass", "fail"}, f"prediction[{index}]: invalid status")
        raw_progress = row.get("normalized_progress")
        if status == "pass":
            progress = _finite_float(
                raw_progress, label=f"prediction[{index}].normalized_progress"
            )
            _require(
                0.0 <= progress <= 1.0,
                f"prediction[{index}].normalized_progress outside [0,1]",
            )
        else:
            _require(raw_progress is None, f"prediction[{index}]: fail must have null progress")
            progress = None
        key = (sample_id, method, condition)
        _require(key not in predictions, f"duplicate prediction Cartesian key: {key}")
        predictions[key] = Prediction(sample_id, method, condition, status == "pass", progress)

    expected_count = len(expected_samples) * len(methods) * len(conditions)
    _require(
        len(predictions) == expected_count,
        f"prediction Cartesian count is {len(predictions)}, expected {expected_count}",
    )
    for sample_id in expected_samples:
        for method in methods:
            for condition in conditions:
                _require(
                    (sample_id, method, condition) in predictions,
                    f"missing prediction Cartesian key: {(sample_id, method, condition)}",
                )
    return predictions


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot calculate a quantile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _metrics(errors: Sequence[float], passed: Sequence[bool]) -> dict[str, float]:
    _require(len(errors) == len(passed) and bool(errors), "metric sample is invalid")
    total = len(errors)
    return {
        "nmae": statistics.fmean(errors),
        "coverage": sum(passed) / total,
        "p50": _quantile(errors, 0.50),
        "p90": _quantile(errors, 0.90),
        "p95": _quantile(errors, 0.95),
        "p99": _quantile(errors, 0.99),
        "acc_at_1pct": sum(error <= 0.01 for error in errors) / total,
        "acc_at_2pct": sum(error <= 0.02 for error in errors) / total,
        "acc_at_5pct": sum(error <= 0.05 for error in errors) / total,
    }


def _group_bootstrap_ci(
    errors: Sequence[float],
    passed: Sequence[bool],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    _require(replicates >= 1, "bootstrap replicates must be positive")
    _require(len(errors) == len(passed) == len(groups), "bootstrap arrays differ in length")
    group_rows: dict[str, list[int]] = {}
    for index, group_id in enumerate(groups):
        group_rows.setdefault(group_id, []).append(index)
    group_ids = sorted(group_rows)
    _require(bool(group_ids), "bootstrap has no groups")
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(replicates):
        sampled_indices: list[int] = []
        for _ in group_ids:
            sampled_indices.extend(group_rows[rng.choice(group_ids)])
        sampled = _metrics(
            [errors[index] for index in sampled_indices],
            [passed[index] for index in sampled_indices],
        )
        for name in METRIC_NAMES:
            draws[name].append(sampled[name])
    return {
        name: {
            "low": _quantile(values, 0.025),
            "high": _quantile(values, 0.975),
        }
        for name, values in draws.items()
    }


def score(
    *,
    predictions_path: Path | Sequence[Path],
    manifest_path: Path,
    validation_ids_path: Path,
    methods: Sequence[str] = METHODS,
    conditions: Sequence[str] = CONDITIONS,
    full_seed_methods: Sequence[str] = FULL_SEED_METHODS,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_groups: int = EXPECTED_GROUPS,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    methods = tuple(methods)
    conditions = tuple(conditions)
    full_seed_methods = tuple(full_seed_methods)
    _require(len(methods) == len(set(methods)) and bool(methods), "method roster is invalid")
    _require(
        len(conditions) == len(set(conditions)) and bool(conditions),
        "condition roster is invalid",
    )
    _require(
        len(full_seed_methods) in (0, 3)
        and len(full_seed_methods) == len(set(full_seed_methods))
        and set(full_seed_methods) <= set(methods),
        "full-seed roster must be empty or contain three unique methods",
    )
    validation_ids = load_validation_ids(validation_ids_path)
    _require(
        len(validation_ids) == expected_samples,
        f"validation sample count is {len(validation_ids)}, expected {expected_samples}",
    )
    targets_tuple = load_targets(manifest_path, validation_ids)
    targets = {target.sample_id: target for target in targets_tuple}
    groups = [target.group_id for target in targets_tuple]
    _require(
        len(set(groups)) == expected_groups,
        f"validation group count is {len(set(groups))}, expected {expected_groups}",
    )
    prediction_paths = _prediction_paths(predictions_path)
    predictions = load_predictions(
        prediction_paths,
        targets=targets,
        methods=methods,
        conditions=conditions,
    )

    results: list[dict[str, Any]] = []
    metrics_by_cell: dict[tuple[str, str], dict[str, float]] = {}
    for method_index, method in enumerate(methods):
        for condition_index, condition in enumerate(conditions):
            cell_errors: list[float] = []
            cell_passed: list[bool] = []
            for target in targets_tuple:
                prediction = predictions[(target.sample_id, method, condition)]
                passed = prediction.passed
                error = (
                    abs(float(prediction.normalized_progress) - target.normalized_target)
                    if passed
                    else 1.0
                )
                cell_errors.append(error)
                cell_passed.append(passed)
            metrics = _metrics(cell_errors, cell_passed)
            metrics_by_cell[(method, condition)] = metrics
            results.append(
                {
                    "method": method,
                    "condition": condition,
                    "samples": len(cell_errors),
                    "groups": len(set(groups)),
                    "passes": sum(cell_passed),
                    "failures": len(cell_passed) - sum(cell_passed),
                    "metrics": metrics,
                    "group_bootstrap_ci95": _group_bootstrap_ci(
                        cell_errors,
                        cell_passed,
                        groups,
                        replicates=bootstrap_replicates,
                        seed=(
                            bootstrap_seed
                            + method_index * len(conditions)
                            + condition_index
                        ),
                    ),
                }
            )

    multiseed: list[dict[str, Any]] = []
    if full_seed_methods:
        for condition in conditions:
            means: dict[str, float] = {}
            sample_sds: dict[str, float] = {}
            for name in METRIC_NAMES:
                values = [
                    metrics_by_cell[(method, condition)][name]
                    for method in full_seed_methods
                ]
                means[name] = statistics.fmean(values)
                sample_sds[name] = statistics.stdev(values)
            multiseed.append(
                {
                    "condition": condition,
                    "seed_methods": list(full_seed_methods),
                    "seed_count": 3,
                    "metric_mean": means,
                    "metric_sample_sd": sample_sds,
                }
            )

    expected_cartesian = expected_samples * len(methods) * len(conditions)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scoring_policy": {
            "normalized_target": "(ground_truth-scale_start)/(scale_end-scale_start)",
            "pass_error": "abs(normalized_progress-normalized_target)",
            "failure_error": 1.0,
            "accuracy_thresholds_inclusive": [0.01, 0.02, 0.05],
            "quantile_method": "linear_interpolation_on_penalized_sample_errors",
            "bootstrap_unit": "group_id",
            "bootstrap_ci": "percentile_95",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "input_bindings": {
            "prediction_files": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in prediction_paths
            ],
            "prediction_file_bindings_sha256": _canonical_sha256(
                [
                    {"path": str(path), "sha256": _sha256_file(path)}
                    for path in prediction_paths
                ]
            ),
            "syncg_manifest_path": str(Path(manifest_path).resolve()),
            "syncg_manifest_sha256": _sha256_file(manifest_path),
            "validation_ids_path": str(Path(validation_ids_path).resolve()),
            "validation_ids_sha256": _sha256_file(validation_ids_path),
        },
        "identity": {
            "samples": len(validation_ids),
            "groups": len(set(groups)),
            "methods": list(methods),
            "conditions": list(conditions),
            "expected_unique_cartesian_rows": expected_cartesian,
            "observed_unique_cartesian_rows": len(predictions),
            "validation_sample_ids_sha256": _canonical_sha256(sorted(validation_ids)),
            "validation_group_ids_sha256": _canonical_sha256(sorted(set(groups))),
            "validation_sample_group_pairs_sha256": _canonical_sha256(
                sorted([[target.sample_id, target.group_id] for target in targets_tuple])
            ),
        },
        "method_condition_results": results,
        "full_three_seed_summary": multiseed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).resolve()
    for source in (*args.predictions, args.manifest, args.validation_ids):
        _require(output != Path(source).resolve(), "output cannot overwrite an input")
    value = score(
        predictions_path=tuple(args.predictions),
        manifest_path=args.manifest,
        validation_ids_path=args.validation_ids,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    output.write_text(payload, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "EXPECTED_GROUPS",
    "EXPECTED_SAMPLES",
    "FULL_SEED_METHODS",
    "METHODS",
    "PlainPaperScoreError",
    "load_predictions",
    "load_targets",
    "load_validation_ids",
    "main",
    "score",
]
