"""Strict CPU scorer for the three-seed DB-GAR18 2x2 factor experiment.

The scorer performs no model loading or inference.  It joins label-free raw or
SARN prediction JSONL files to an explicitly supplied label file and roster,
penalizes prediction failures with normalized error 1.0, and reports the
CBAM x geometry-auxiliary factorial effects.

Command line entry point::

    python -m experiments.score_db_gar18_factorial_3seed \
      --spec factorial_score_spec.json --output factorial_score.json

The JSON spec has this shape (artifact values may instead be objects with
``path`` and an optional expected ``sha256``)::

    {
      "schema_version": 1,
      "protocol": "db_gar18_factorial_three_seed_score_spec_v1",
      "prediction_variant": "raw",
      "robustness_seed": 20260720,
      "seeds": [20262020, 20262021, 20262022],
      "bootstrap": {"seed": 20260811, "replicates": 20000},
      "checkpoints": {
        "00": {"20262020": "...pt", "20262021": "...pt", "20262022": "...pt"},
        "10": {"20262020": "...pt", "20262021": "...pt", "20262022": "...pt"},
        "01": {"20262020": "...pt", "20262021": "...pt", "20262022": "...pt"},
        "11": {"20262020": "...pt", "20262021": "...pt", "20262022": "...pt"}
      },
      "datasets": {
        "xm2": {
          "labels": "labels.jsonl",
          "roster": "sample_ids.json",
          "manifest": "input_manifest.jsonl",
          "expected_samples": 814,
          "expected_groups": 20,
          "conditions": ["clean", "blur_severe", "perspective_severe"],
          "key_conditions": ["clean", "perspective_severe"],
          "predictions": {
            "00": {
              "20262020": {
                "path": "predictions.jsonl",
                "method": "db_resnet18_seed_20262020"
              }
            }
          }
        }
      }
    }

Every cell in ``predictions`` must contain all three seed bindings; the
abbreviated example shows only one.  With ``prediction_variant`` set to
``sarn``, every prediction binding must additionally contain ``sidecar`` (and
may contain ``sidecar_sha256``).  For every sample/condition, all twelve SARN
runs must have identical pre-normalization hashes and identical
post-normalization hashes.  A file may also contain rows for other conditions
in the fixed six-condition robustness roster; those rows are counted and
ignored.  Conditions outside that fixed roster are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS as KNOWN_CONDITIONS,
    OUTPUT_KEYS,
    PLAIN_MANIFEST_KEYS,
)
from experiments.score_cagh_v5_plain_paper_batch import (
    PlainPaperScoreError,
    Target,
    _canonical_sha256,
    _finite_float,
    _metrics,
    _quantile,
    _read_jsonl,
    _require,
    _sha256_file,
    _text,
    load_targets,
    load_validation_ids,
)


PROTOCOL: Final[str] = "db_gar18_factorial_three_seed_score_v1"
SPEC_PROTOCOL: Final[str] = "db_gar18_factorial_three_seed_score_spec_v1"
CELLS: Final[tuple[str, ...]] = ("00", "10", "01", "11")
NON_BASELINE_CELLS: Final[tuple[str, ...]] = ("10", "01", "11")
CELL_FACTORS: Final[dict[str, dict[str, bool]]] = {
    "00": {"cbam": False, "geometry_auxiliary": False},
    "10": {"cbam": True, "geometry_auxiliary": False},
    "01": {"cbam": False, "geometry_auxiliary": True},
    "11": {"cbam": True, "geometry_auxiliary": True},
}
METRIC_NAMES: Final[tuple[str, ...]] = (
    "nmae",
    "coverage",
    "success_only_nmae",
    "p50",
    "p90",
    "p95",
    "p99",
    "acc_at_1pct",
    "acc_at_2pct",
    "acc_at_5pct",
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ground_truth",
        "target",
        "normalized_target",
        "scale_start",
        "scale_end",
        "group_id",
        "label",
        "labels",
        "angle",
        "min",
        "max",
        "pointer",
    }
)

FactorialScoreError = PlainPaperScoreError


@dataclass(frozen=True, slots=True)
class PredictionValue:
    passed: bool
    normalized_progress: float | None
    failure_code: str | None
    condition_pixel_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionRun:
    path: Path
    sha256: str
    method: str
    protocol: str
    rows: Mapping[tuple[str, str], PredictionValue]
    failure_codes: Mapping[str, int]
    ignored_extra_prediction_rows: Mapping[str, int]
    sidecar_path: Path | None
    sidecar_sha256: str | None
    sidecar_protocol: str | None
    sidecar_hashes: Mapping[tuple[str, str], tuple[str, str]] | None
    ignored_extra_sidecar_rows: Mapping[str, int]


def _integer(value: Any, *, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    return int(value)


def _positive_integer(value: Any, *, label: str) -> int:
    result = _integer(value, label=label)
    _require(result >= 1, f"{label} must be positive")
    return result


def _valid_sha256(value: Any, *, label: str) -> str:
    digest = _text(value, label=label)
    _require(bool(_SHA256_RE.fullmatch(digest)), f"{label} is not lowercase SHA256")
    return digest


def _resolve_path(raw: Any, *, label: str, base_dir: Path) -> Path:
    text = _text(raw, label=label)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise FactorialScoreError(f"{label} does not exist: {candidate}") from exc


def _artifact_binding(
    value: Any,
    *,
    label: str,
    base_dir: Path,
) -> tuple[Path, dict[str, str]]:
    if isinstance(value, str):
        raw_path = value
        expected_digest = None
    else:
        _require(isinstance(value, Mapping), f"{label} must be a path or object")
        _require(
            set(value) <= {"path", "sha256"} and "path" in value,
            f"{label} artifact binding schema drift",
        )
        raw_path = value.get("path")
        expected_digest = value.get("sha256")
    path = _resolve_path(raw_path, label=f"{label}.path", base_dir=base_dir)
    _require(path.is_file(), f"{label} is not a file: {path}")
    digest = _sha256_file(path)
    if expected_digest is not None:
        _require(
            _valid_sha256(expected_digest, label=f"{label}.sha256") == digest,
            f"{label} SHA256 drift",
        )
    return path, {"path": str(path), "sha256": digest}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _validate_seed_map(
    value: Any,
    *,
    seeds: Sequence[int],
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    expected = {str(seed) for seed in seeds}
    _require(set(mapping) == expected, f"{label} must contain exactly the three seeds")
    return mapping


def _load_plain_manifest(
    path: Path,
    *,
    sample_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path, label="label-free input manifest")
    observed_order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(
            set(row) == set(PLAIN_MANIFEST_KEYS),
            f"manifest[{index}] is not the strict four-field label-free schema",
        )
        sample_id = _text(row.get("sample_id"), label=f"manifest[{index}].sample_id")
        _require(sample_id not in by_id, f"duplicate manifest sample_id: {sample_id}")
        _text(row.get("roi_path"), label=f"manifest[{index}].roi_path")
        _valid_sha256(
            row.get("roi_png_sha256"), label=f"manifest[{index}].roi_png_sha256"
        )
        _valid_sha256(
            row.get("roi_pixel_sha256"), label=f"manifest[{index}].roi_pixel_sha256"
        )
        observed_order.append(sample_id)
        by_id[sample_id] = dict(row)
    _require(
        tuple(observed_order) == tuple(sample_ids),
        "label-free manifest order differs from the frozen roster",
    )
    return by_id, {
        "rows": len(rows),
        "sample_order_sha256": _canonical_sha256(observed_order),
        "sample_roi_hash_bindings_sha256": _canonical_sha256(
            [
                [
                    sample_id,
                    by_id[sample_id]["roi_png_sha256"],
                    by_id[sample_id]["roi_pixel_sha256"],
                ]
                for sample_id in observed_order
            ]
        ),
    }


def _load_targets_with_group_source(
    path: Path,
    sample_ids: Sequence[str],
    *,
    group_source: str,
) -> tuple[Target, ...]:
    """Load targets while keeping the declared independent cluster identity."""

    targets = load_targets(path, sample_ids)
    if group_source == "labels.group_id":
        return targets
    _require(
        group_source == "metadata.scene_name stem",
        f"unsupported target group source: {group_source}",
    )
    wanted = set(sample_ids)
    scenes: dict[str, str] = {}
    for index, row in enumerate(_read_jsonl(path, label="SyncG labels")):
        sample_id = _text(row.get("sample_id"), label=f"labels[{index}].sample_id")
        if sample_id not in wanted:
            continue
        _require(sample_id not in scenes, f"duplicate SyncG scene sample: {sample_id}")
        metadata = _mapping(row.get("metadata"), label=f"labels[{index}].metadata")
        scene_name = _text(
            metadata.get("scene_name"),
            label=f"labels[{index}].metadata.scene_name",
        )
        scene_stem = Path(scene_name).stem
        _require(bool(scene_stem), f"empty SyncG scene stem: {sample_id}")
        scenes[sample_id] = scene_stem
    _require(set(scenes) == wanted, "SyncG scene grouping does not cover the roster")
    return tuple(
        Target(
            sample_id=target.sample_id,
            group_id=scenes[target.sample_id],
            normalized_target=target.normalized_target,
        )
        for target in targets
    )


def _parse_prediction_binding(
    value: Any,
    *,
    label: str,
    base_dir: Path,
    require_sidecar: bool,
) -> tuple[Path, str, str | None, Path | None, str | None]:
    binding = _mapping(value, label=label)
    allowed = {
        "path",
        "method",
        "sha256",
        "protocol",
        "sidecar",
        "sidecar_sha256",
    }
    _require(set(binding) <= allowed, f"{label} prediction binding schema drift")
    _require("path" in binding and "method" in binding, f"{label} lacks path/method")
    path, artifact = _artifact_binding(
        {key: binding[key] for key in ("path", "sha256") if key in binding},
        label=f"{label}.prediction",
        base_dir=base_dir,
    )
    expected_method = _text(binding.get("method"), label=f"{label}.method")
    expected_protocol = (
        _text(binding.get("protocol"), label=f"{label}.protocol")
        if "protocol" in binding
        else None
    )
    sidecar_path: Path | None = None
    sidecar_digest: str | None = None
    if "sidecar" in binding:
        sidecar_path, sidecar_artifact = _artifact_binding(
            {
                "path": binding["sidecar"],
                **(
                    {"sha256": binding["sidecar_sha256"]}
                    if "sidecar_sha256" in binding
                    else {}
                ),
            },
            label=f"{label}.sidecar",
            base_dir=base_dir,
        )
        sidecar_digest = sidecar_artifact["sha256"]
    else:
        _require(
            "sidecar_sha256" not in binding,
            f"{label} has sidecar_sha256 without sidecar",
        )
    _require(
        (sidecar_path is not None) == require_sidecar,
        f"{label} sidecar presence differs from prediction_variant",
    )
    return path, expected_method, expected_protocol, sidecar_path, sidecar_digest


def _load_prediction_rows(
    path: Path,
    *,
    expected_method: str,
    expected_protocol: str | None,
    robustness_seed: int,
    targets: Mapping[str, Target],
    conditions: Sequence[str],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], PredictionValue],
    str,
    dict[str, int],
    dict[str, int],
]:
    rows = _read_jsonl(path, label=f"predictions ({path})")
    wanted_ids = set(targets)
    wanted_conditions = set(conditions)
    selected: dict[tuple[str, str], PredictionValue] = {}
    protocols: set[str] = set()
    failure_codes: Counter[str] = Counter()
    ignored_extra_rows: Counter[str] = Counter()
    for index, row in enumerate(rows):
        leaked = sorted(set(row) & _FORBIDDEN_PREDICTION_KEYS)
        _require(not leaked, f"prediction {path.name}[{index}] contains labels: {leaked}")
        _require(
            set(row) == set(OUTPUT_KEYS),
            f"prediction {path.name}[{index}] schema drift",
        )
        sample_id = _text(row.get("sample_id"), label=f"prediction[{index}].sample_id")
        method = _text(row.get("method"), label=f"prediction[{index}].method")
        condition = _text(row.get("condition"), label=f"prediction[{index}].condition")
        _require(
            condition in KNOWN_CONDITIONS,
            f"unexpected prediction condition: {condition}",
        )
        if condition not in wanted_conditions:
            ignored_extra_rows[condition] += 1
            continue
        _require(row.get("schema_version") == 1, "prediction schema version drift")
        protocol = _text(row.get("protocol"), label=f"prediction[{index}].protocol")
        protocols.add(protocol)
        _require(method == expected_method, f"{path.name}: method binding drift")
        _require(sample_id in wanted_ids, f"unexpected prediction sample_id: {sample_id}")
        _require(
            row.get("robustness_seed") == robustness_seed,
            f"{sample_id}/{condition}: robustness seed drift",
        )
        manifest_row = manifest_by_id[sample_id]
        for hash_name in ("roi_png_sha256", "roi_pixel_sha256"):
            digest = _valid_sha256(
                row.get(hash_name), label=f"{sample_id}/{condition}.{hash_name}"
            )
            _require(
                digest == manifest_row[hash_name],
                f"{sample_id}/{condition}: {hash_name} differs from manifest",
            )
        condition_hash = _valid_sha256(
            row.get("condition_pixel_sha256"),
            label=f"{sample_id}/{condition}.condition_pixel_sha256",
        )
        status = _text(row.get("status"), label=f"{sample_id}/{condition}.status")
        _require(status in {"pass", "fail"}, f"{sample_id}/{condition}: invalid status")
        if status == "pass":
            progress = _finite_float(
                row.get("normalized_progress"),
                label=f"{sample_id}/{condition}.normalized_progress",
            )
            _require(
                0.0 <= progress <= 1.0,
                f"{sample_id}/{condition}: normalized progress outside [0,1]",
            )
            _require(
                row.get("failure_code") is None,
                f"{sample_id}/{condition}: pass has failure_code",
            )
            failure_code = None
        else:
            _require(
                row.get("normalized_progress") is None,
                f"{sample_id}/{condition}: failure has progress",
            )
            failure_code = _text(
                row.get("failure_code"),
                label=f"{sample_id}/{condition}.failure_code",
            )
            failure_codes[f"{condition}::{failure_code}"] += 1
            progress = None
        key = (sample_id, condition)
        _require(key not in selected, f"duplicate prediction Cartesian key: {key}")
        selected[key] = PredictionValue(
            passed=status == "pass",
            normalized_progress=progress,
            failure_code=failure_code,
            condition_pixel_sha256=condition_hash,
        )
    _require(len(protocols) == 1, f"prediction file has multiple protocols: {path}")
    protocol = next(iter(protocols))
    if expected_protocol is not None:
        _require(protocol == expected_protocol, f"{path.name}: protocol binding drift")
    expected_count = len(targets) * len(conditions)
    _require(
        len(selected) == expected_count,
        f"{expected_method}: Cartesian count is {len(selected)}, expected {expected_count}",
    )
    for sample_id in targets:
        for condition in conditions:
            _require(
                (sample_id, condition) in selected,
                f"missing prediction Cartesian key: {(sample_id, condition)}",
            )
    return (
        selected,
        protocol,
        dict(sorted(failure_codes.items())),
        dict(sorted(ignored_extra_rows.items())),
    )


def _load_sarn_sidecar(
    path: Path,
    *,
    expected_method: str,
    prediction_protocol: str,
    robustness_seed: int,
    predictions: Mapping[tuple[str, str], PredictionValue],
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    str,
    dict[str, int],
]:
    rows = _read_jsonl(path, label=f"SARN sidecar ({path})")
    selected: dict[tuple[str, str], tuple[str, str]] = {}
    protocols: set[str] = set()
    ignored_extra_rows: Counter[str] = Counter()
    requested_conditions = {condition for _sample_id, condition in predictions}
    for index, row in enumerate(rows):
        sample_id = _text(row.get("sample_id"), label=f"sidecar[{index}].sample_id")
        method = _text(row.get("method"), label=f"sidecar[{index}].method")
        condition = _text(row.get("condition"), label=f"sidecar[{index}].condition")
        _require(
            condition in KNOWN_CONDITIONS,
            f"unexpected SARN sidecar condition: {condition}",
        )
        if condition not in requested_conditions:
            ignored_extra_rows[condition] += 1
            continue
        _require(row.get("schema_version") == 1, "SARN sidecar schema version drift")
        protocol = _text(row.get("protocol"), label=f"sidecar[{index}].protocol")
        protocols.add(protocol)
        _require(method == expected_method, f"{path.name}: sidecar method binding drift")
        _require(
            row.get("robustness_seed") == robustness_seed,
            f"{sample_id}/{condition}: sidecar robustness seed drift",
        )
        key = (sample_id, condition)
        _require(key in predictions, f"unexpected SARN sidecar Cartesian key: {key}")
        _require(key not in selected, f"duplicate SARN sidecar Cartesian key: {key}")
        pre_hash = _valid_sha256(
            row.get("pre_normalization_pixel_sha256"),
            label=f"{sample_id}/{condition}.pre_normalization_pixel_sha256",
        )
        post_hash = _valid_sha256(
            row.get("post_normalization_pixel_sha256"),
            label=f"{sample_id}/{condition}.post_normalization_pixel_sha256",
        )
        _require(
            pre_hash == predictions[key].condition_pixel_sha256,
            f"{sample_id}/{condition}: sidecar pre hash differs from prediction",
        )
        selected[key] = (pre_hash, post_hash)
    _require(len(protocols) == 1, f"SARN sidecar has multiple protocols: {path}")
    protocol = next(iter(protocols))
    _require(
        protocol == prediction_protocol,
        f"{path.name}: sidecar/prediction protocol drift",
    )
    _require(
        set(selected) == set(predictions),
        f"{path.name}: SARN sidecar Cartesian coverage drift",
    )
    return selected, protocol, dict(sorted(ignored_extra_rows.items()))


def _load_prediction_run(
    value: Any,
    *,
    label: str,
    base_dir: Path,
    require_sidecar: bool,
    robustness_seed: int,
    targets: Mapping[str, Target],
    conditions: Sequence[str],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> PredictionRun:
    (
        path,
        expected_method,
        expected_protocol,
        sidecar_path,
        sidecar_digest,
    ) = _parse_prediction_binding(
        value,
        label=label,
        base_dir=base_dir,
        require_sidecar=require_sidecar,
    )
    rows, protocol, failure_codes, ignored_prediction_rows = _load_prediction_rows(
        path,
        expected_method=expected_method,
        expected_protocol=expected_protocol,
        robustness_seed=robustness_seed,
        targets=targets,
        conditions=conditions,
        manifest_by_id=manifest_by_id,
    )
    sidecar_hashes = None
    sidecar_protocol = None
    ignored_sidecar_rows: dict[str, int] = {}
    if sidecar_path is not None:
        sidecar_hashes, sidecar_protocol, ignored_sidecar_rows = _load_sarn_sidecar(
            sidecar_path,
            expected_method=expected_method,
            prediction_protocol=protocol,
            robustness_seed=robustness_seed,
            predictions=rows,
        )
    return PredictionRun(
        path=path,
        sha256=_sha256_file(path),
        method=expected_method,
        protocol=protocol,
        rows=rows,
        failure_codes=failure_codes,
        ignored_extra_prediction_rows=ignored_prediction_rows,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_digest,
        sidecar_protocol=sidecar_protocol,
        sidecar_hashes=sidecar_hashes,
        ignored_extra_sidecar_rows=ignored_sidecar_rows,
    )


def _full_metrics(
    errors: Sequence[float],
    passed: Sequence[bool],
) -> dict[str, float | None]:
    result: dict[str, float | None] = dict(_metrics(errors, passed))
    successes = [error for error, success in zip(errors, passed, strict=True) if success]
    result["success_only_nmae"] = statistics.fmean(successes) if successes else None
    return result


def _mean_sample_sd(values: Sequence[float | None]) -> dict[str, Any]:
    _require(len(values) == 3, "three seed values are required")
    finite = [float(value) for value in values if value is not None]
    return {
        "values": [None if value is None else float(value) for value in values],
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "sample_sd": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def _direction_vs_00(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    _require(len(candidate) == len(baseline) == len(seeds) == 3, "direction vectors drift")
    per_seed: list[dict[str, Any]] = []
    for seed, candidate_value, baseline_value in zip(
        seeds, candidate, baseline, strict=True
    ):
        delta = float(candidate_value) - float(baseline_value)
        direction = "improved" if delta < 0.0 else ("tied" if delta == 0.0 else "worse")
        per_seed.append(
            {
                "seed": seed,
                "candidate_nmae": float(candidate_value),
                "baseline_00_nmae": float(baseline_value),
                "delta_nmae_cell_minus_00": delta,
                "direction": direction,
            }
        )
    improved = sum(row["direction"] == "improved" for row in per_seed)
    tied = sum(row["direction"] == "tied" for row in per_seed)
    return {
        "metric": "full_denominator_nmae",
        "lower_is_better": True,
        "delta_definition": "cell_nmae_minus_00_nmae",
        "per_seed": per_seed,
        "improved_seeds": improved,
        "tied_seeds": tied,
        "worse_seeds": 3 - improved - tied,
        "three_of_three_improved": improved == 3,
    }


def _paired_complete_group_bootstrap(
    candidate_by_seed: Sequence[Sequence[float]],
    baseline_by_seed: Sequence[Sequence[float]],
    groups: Sequence[str],
    sample_ids: Sequence[str],
    *,
    group_unit: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    _require(
        len(candidate_by_seed) == len(baseline_by_seed) == 3,
        "paired bootstrap requires exactly three seed arrays",
    )
    samples = len(groups)
    _require(samples == len(sample_ids) and samples >= 1, "paired bootstrap identity drift")
    for values in (*candidate_by_seed, *baseline_by_seed):
        _require(len(values) == samples, "paired bootstrap seed array length drift")
        _require(all(math.isfinite(float(value)) for value in values), "non-finite error")
    candidate = [
        statistics.fmean(float(values[index]) for values in candidate_by_seed)
        for index in range(samples)
    ]
    baseline = [
        statistics.fmean(float(values[index]) for values in baseline_by_seed)
        for index in range(samples)
    ]
    effects = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    group_rows: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_rows.setdefault(group, []).append(index)
    group_ids = sorted(group_rows)
    _require(len(group_ids) >= 2, "paired bootstrap requires at least two complete groups")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        indices: list[int] = []
        for _group in group_ids:
            indices.extend(group_rows[rng.choice(group_ids)])
        draws.append(statistics.fmean(effects[index] for index in indices))
    point = statistics.fmean(effects)
    low = _quantile(draws, 0.025)
    high = _quantile(draws, 0.975)
    return {
        "metric": "full_denominator_nmae",
        "lower_is_better": True,
        "delta_definition": "candidate_minus_00",
        "failure_error": 1.0,
        "seed_handling": (
            "average the three failure-penalized errors within each sample before "
            "resampling complete dataset group_id clusters; seeds are not bootstrap units"
        ),
        "candidate_seed_averaged_nmae": statistics.fmean(candidate),
        "baseline_00_seed_averaged_nmae": statistics.fmean(baseline),
        "delta_nmae": point,
        "paired_complete_group_bootstrap_ci95": {"low": low, "high": high},
        "samples": samples,
        "group_clusters": len(group_ids),
        "group_unit": group_unit,
        "replicates": replicates,
        "seed": seed,
        "three_seed_sample_effects_sha256": _canonical_sha256(
            [
                [sample_id, group, effect]
                for sample_id, group, effect in zip(
                    sample_ids, groups, effects, strict=True
                )
            ]
        ),
        "candidate_better": point < 0.0,
        "superiority_ci95": high < 0.0,
        "superiority_rule": "upper_ci95_below_zero",
    }


def _audit_run_pixels(
    runs: Mapping[tuple[str, int], PredictionRun],
    *,
    sample_ids: Sequence[str],
    conditions: Sequence[str],
    require_sidecar: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    condition_bindings: list[list[str]] = []
    sidecar_bindings: list[list[str]] = []
    for sample_id in sample_ids:
        for condition in conditions:
            condition_hashes = {
                run.rows[(sample_id, condition)].condition_pixel_sha256
                for run in runs.values()
            }
            _require(
                len(condition_hashes) == 1,
                f"{sample_id}/{condition}: prediction input pixels differ across arms/seeds",
            )
            condition_hash = next(iter(condition_hashes))
            condition_bindings.append([sample_id, condition, condition_hash])
            if require_sidecar:
                pairs = {
                    run.sidecar_hashes[(sample_id, condition)]
                    for run in runs.values()
                    if run.sidecar_hashes is not None
                }
                _require(
                    len(pairs) == 1,
                    f"{sample_id}/{condition}: SARN pre/post pixels differ across arms/seeds",
                )
                pre_hash, post_hash = next(iter(pairs))
                _require(
                    pre_hash == condition_hash,
                    f"{sample_id}/{condition}: SARN pre hash differs from prediction input",
                )
                sidecar_bindings.append([sample_id, condition, pre_hash, post_hash])
    prediction_audit = {
        "status": "verified",
        "same_condition_pixels_across_all_four_cells_and_three_seeds": True,
        "sample_condition_rows": len(condition_bindings),
        "condition_pixel_bindings_sha256": _canonical_sha256(condition_bindings),
    }
    if not require_sidecar:
        return prediction_audit, None
    return prediction_audit, {
        "status": "verified",
        "same_pre_normalization_pixels_across_all_four_cells_and_three_seeds": True,
        "same_post_normalization_pixels_across_all_four_cells_and_three_seeds": True,
        "pre_hash_matches_prediction_condition_hash": True,
        "sample_condition_rows": len(sidecar_bindings),
        "pre_post_pixel_bindings_sha256": _canonical_sha256(sidecar_bindings),
    }


def _score_dataset(
    name: str,
    value: Any,
    *,
    base_dir: Path,
    seeds: Sequence[int],
    prediction_variant: str,
    robustness_seed: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
    config = _mapping(value, label=f"datasets.{name}")
    allowed = {
        "labels",
        "roster",
        "manifest",
        "expected_samples",
        "expected_groups",
        "conditions",
        "key_conditions",
        "predictions",
        "group_unit",
        "group_source",
    }
    required = allowed - {"group_unit", "group_source"}
    _require(
        required <= set(config) <= allowed,
        f"datasets.{name} schema drift",
    )
    labels, labels_binding = _artifact_binding(
        config["labels"], label=f"datasets.{name}.labels", base_dir=base_dir
    )
    roster, roster_binding = _artifact_binding(
        config["roster"], label=f"datasets.{name}.roster", base_dir=base_dir
    )
    manifest, manifest_binding = _artifact_binding(
        config["manifest"], label=f"datasets.{name}.manifest", base_dir=base_dir
    )
    expected_samples = _positive_integer(
        config["expected_samples"], label=f"datasets.{name}.expected_samples"
    )
    expected_groups = _positive_integer(
        config["expected_groups"], label=f"datasets.{name}.expected_groups"
    )
    group_unit = _text(
        config.get("group_unit", "dataset group_id cluster"),
        label=f"datasets.{name}.group_unit",
    )
    group_source = _text(
        config.get("group_source", "labels.group_id"),
        label=f"datasets.{name}.group_source",
    )
    raw_conditions = config["conditions"]
    _require(isinstance(raw_conditions, list), f"datasets.{name}.conditions must be a list")
    conditions = tuple(
        _text(condition, label=f"datasets.{name}.conditions[{index}]")
        for index, condition in enumerate(raw_conditions)
    )
    _require(
        bool(conditions)
        and len(conditions) == len(set(conditions))
        and set(conditions) <= set(KNOWN_CONDITIONS),
        f"datasets.{name}.conditions roster is invalid",
    )
    raw_key_conditions = config["key_conditions"]
    _require(
        isinstance(raw_key_conditions, list),
        f"datasets.{name}.key_conditions must be a list",
    )
    key_conditions = tuple(
        _text(condition, label=f"datasets.{name}.key_conditions[{index}]")
        for index, condition in enumerate(raw_key_conditions)
    )
    _require(
        bool(key_conditions)
        and len(key_conditions) == len(set(key_conditions))
        and set(key_conditions) <= set(conditions),
        f"datasets.{name}.key_conditions roster is invalid",
    )

    sample_ids = load_validation_ids(roster)
    _require(
        len(sample_ids) == expected_samples,
        f"{name}: sample count is {len(sample_ids)}, expected {expected_samples}",
    )
    targets_tuple = _load_targets_with_group_source(
        labels,
        sample_ids,
        group_source=group_source,
    )
    targets = {target.sample_id: target for target in targets_tuple}
    groups = [target.group_id for target in targets_tuple]
    _require(
        len(set(groups)) == expected_groups,
        f"{name}: group count is {len(set(groups))}, expected {expected_groups}",
    )
    manifest_by_id, manifest_audit = _load_plain_manifest(
        manifest, sample_ids=sample_ids
    )

    prediction_matrix = _mapping(
        config["predictions"], label=f"datasets.{name}.predictions"
    )
    _require(
        set(prediction_matrix) == set(CELLS),
        f"datasets.{name}.predictions must contain exactly cells {CELLS}",
    )
    require_sidecar = prediction_variant == "sarn"
    runs: dict[tuple[str, int], PredictionRun] = {}
    methods: dict[tuple[str, int], str] = {}
    for cell in CELLS:
        seed_map = _validate_seed_map(
            prediction_matrix[cell],
            seeds=seeds,
            label=f"datasets.{name}.predictions.{cell}",
        )
        for seed in seeds:
            run = _load_prediction_run(
                seed_map[str(seed)],
                label=f"datasets.{name}.predictions.{cell}.{seed}",
                base_dir=base_dir,
                require_sidecar=require_sidecar,
                robustness_seed=robustness_seed,
                targets=targets,
                conditions=conditions,
                manifest_by_id=manifest_by_id,
            )
            runs[(cell, seed)] = run
            methods[(cell, seed)] = run.method
    _require(
        len({run.path for run in runs.values()}) == len(runs),
        f"{name}: prediction paths are not unique across the 12 runs",
    )
    _require(
        len(set(methods.values())) == len(methods),
        f"{name}: methods are not unique across the 12 runs",
    )
    if require_sidecar:
        _require(
            len({run.sidecar_path for run in runs.values()}) == len(runs),
            f"{name}: SARN sidecar paths are not unique across the 12 runs",
        )

    prediction_pixel_audit, sarn_pixel_audit = _audit_run_pixels(
        runs,
        sample_ids=sample_ids,
        conditions=conditions,
        require_sidecar=require_sidecar,
    )

    errors: dict[tuple[str, int, str], list[float]] = {}
    passed: dict[tuple[str, int, str], list[bool]] = {}
    metric_results: dict[tuple[str, int, str], dict[str, float | None]] = {}
    cells_output: dict[str, Any] = {}
    for cell in CELLS:
        condition_output: dict[str, Any] = {}
        for condition in conditions:
            seed_results: list[dict[str, Any]] = []
            for seed in seeds:
                run = runs[(cell, seed)]
                seed_errors: list[float] = []
                seed_passed: list[bool] = []
                cell_failure_codes: Counter[str] = Counter()
                for target in targets_tuple:
                    prediction = run.rows[(target.sample_id, condition)]
                    success = prediction.passed
                    error = (
                        abs(
                            float(prediction.normalized_progress)
                            - target.normalized_target
                        )
                        if success
                        else 1.0
                    )
                    seed_errors.append(error)
                    seed_passed.append(success)
                    if not success:
                        cell_failure_codes[str(prediction.failure_code)] += 1
                cell_metrics = _full_metrics(seed_errors, seed_passed)
                errors[(cell, seed, condition)] = seed_errors
                passed[(cell, seed, condition)] = seed_passed
                metric_results[(cell, seed, condition)] = cell_metrics
                seed_results.append(
                    {
                        "seed": seed,
                        "method": run.method,
                        "samples": len(sample_ids),
                        "groups": len(set(groups)),
                        "passes": sum(seed_passed),
                        "failures": len(seed_passed) - sum(seed_passed),
                        "failure_codes": dict(sorted(cell_failure_codes.items())),
                        "metrics": cell_metrics,
                    }
                )
            metric_summary = {
                metric: _mean_sample_sd(
                    [
                        metric_results[(cell, seed, condition)][metric]
                        for seed in seeds
                    ]
                )
                for metric in METRIC_NAMES
            }
            condition_output[condition] = {
                "seed_results": seed_results,
                "three_seed_metric_mean_and_sample_sd": metric_summary,
            }
        cells_output[cell] = {
            "factors": CELL_FACTORS[cell],
            "conditions": condition_output,
        }

    for condition in conditions:
        baseline = [
            float(metric_results[("00", seed, condition)]["nmae"])
            for seed in seeds
        ]
        cells_output["00"]["conditions"][condition]["direction_vs_00"] = {
            "role": "reference",
            "metric": "full_denominator_nmae",
            "lower_is_better": True,
        }
        for cell in NON_BASELINE_CELLS:
            candidate = [
                float(metric_results[(cell, seed, condition)]["nmae"])
                for seed in seeds
            ]
            cells_output[cell]["conditions"][condition]["direction_vs_00"] = (
                _direction_vs_00(candidate, baseline, seeds=seeds)
            )

    factorial_effects: dict[str, Any] = {}
    for condition in conditions:
        per_seed: list[dict[str, Any]] = []
        values_by_effect: dict[str, list[float]] = {
            "cbam_main_effect": [],
            "geometry_auxiliary_main_effect": [],
            "cbam_x_geometry_auxiliary_interaction": [],
        }
        for seed in seeds:
            y = {
                cell: float(metric_results[(cell, seed, condition)]["nmae"])
                for cell in CELLS
            }
            effects = {
                "cbam_main_effect": ((y["10"] + y["11"]) - (y["00"] + y["01"])) / 2.0,
                "geometry_auxiliary_main_effect": (
                    (y["01"] + y["11"]) - (y["00"] + y["10"])
                )
                / 2.0,
                "cbam_x_geometry_auxiliary_interaction": (
                    y["11"] - y["10"] - y["01"] + y["00"]
                ),
            }
            for effect_name, effect_value in effects.items():
                values_by_effect[effect_name].append(effect_value)
            per_seed.append({"seed": seed, "cell_nmae": y, **effects})
        factorial_effects[condition] = {
            "per_seed": per_seed,
            "across_seed_mean_and_sample_sd": {
                name: _mean_sample_sd(values)
                for name, values in values_by_effect.items()
            },
        }

    key_bootstrap: dict[str, Any] = {}
    for condition in key_conditions:
        comparisons: dict[str, Any] = {}
        baseline_by_seed = [errors[("00", seed, condition)] for seed in seeds]
        for cell in NON_BASELINE_CELLS:
            comparisons[f"{cell}_vs_00"] = _paired_complete_group_bootstrap(
                [errors[(cell, seed, condition)] for seed in seeds],
                baseline_by_seed,
                groups,
                sample_ids,
                group_unit=group_unit,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            )
            comparisons[f"{cell}_vs_00"]["candidate_cell"] = cell
            comparisons[f"{cell}_vs_00"]["baseline_cell"] = "00"
        key_bootstrap[condition] = comparisons

    prediction_bindings = {
        cell: {
            str(seed): {
                "path": str(runs[(cell, seed)].path),
                "sha256": runs[(cell, seed)].sha256,
                "method": runs[(cell, seed)].method,
                "protocol": runs[(cell, seed)].protocol,
                "ignored_extra_prediction_rows": {
                    "total": sum(
                        runs[(cell, seed)].ignored_extra_prediction_rows.values()
                    ),
                    "by_condition": dict(
                        runs[(cell, seed)].ignored_extra_prediction_rows
                    ),
                },
                "sidecar": (
                    {
                        "path": str(runs[(cell, seed)].sidecar_path),
                        "sha256": runs[(cell, seed)].sidecar_sha256,
                        "protocol": runs[(cell, seed)].sidecar_protocol,
                        "ignored_extra_rows": {
                            "total": sum(
                                runs[(cell, seed)].ignored_extra_sidecar_rows.values()
                            ),
                            "by_condition": dict(
                                runs[(cell, seed)].ignored_extra_sidecar_rows
                            ),
                        },
                    }
                    if runs[(cell, seed)].sidecar_path is not None
                    else None
                ),
            }
            for seed in seeds
        }
        for cell in CELLS
    }
    ignored_prediction_by_condition: Counter[str] = Counter()
    ignored_sidecar_by_condition: Counter[str] = Counter()
    ignored_by_run: dict[str, Any] = {}
    for cell in CELLS:
        ignored_by_run[cell] = {}
        for seed in seeds:
            run = runs[(cell, seed)]
            ignored_prediction_by_condition.update(run.ignored_extra_prediction_rows)
            ignored_sidecar_by_condition.update(run.ignored_extra_sidecar_rows)
            ignored_by_run[cell][str(seed)] = {
                "prediction": {
                    "total": sum(run.ignored_extra_prediction_rows.values()),
                    "by_condition": dict(run.ignored_extra_prediction_rows),
                },
                "sidecar": {
                    "total": sum(run.ignored_extra_sidecar_rows.values()),
                    "by_condition": dict(run.ignored_extra_sidecar_rows),
                },
            }
    expected_rows = len(sample_ids) * len(conditions) * len(CELLS) * len(seeds)
    return (
        {
            "status": "complete",
            "prediction_variant": prediction_variant,
            "input_bindings": {
                "labels": labels_binding,
                "roster": roster_binding,
                "label_free_manifest": manifest_binding,
                "predictions": prediction_bindings,
                "all_dataset_artifact_bindings_sha256": _canonical_sha256(
                    {
                        "labels": labels_binding,
                        "roster": roster_binding,
                        "manifest": manifest_binding,
                        "predictions": prediction_bindings,
                    }
                ),
            },
            "identity": {
                "samples": len(sample_ids),
                "groups": len(set(groups)),
                "group_unit": group_unit,
                "group_source": group_source,
                "conditions": list(conditions),
                "key_conditions": list(key_conditions),
                "cells": list(CELLS),
                "seeds": list(seeds),
                "sample_ids_sha256": _canonical_sha256(list(sample_ids)),
                "sample_group_pairs_sha256": _canonical_sha256(
                    [
                        [target.sample_id, target.group_id]
                        for target in targets_tuple
                    ]
                ),
            },
            "audit": {
                "coverage_and_cartesian": {
                    "status": "verified",
                    "expected_prediction_rows": expected_rows,
                    "observed_unique_prediction_rows": sum(
                        len(run.rows) for run in runs.values()
                    ),
                    "runs": len(runs),
                    "complete_four_cell_three_seed_cartesian": True,
                    "each_run_has_complete_sample_condition_cartesian": True,
                },
                "ignored_extra_rows": {
                    "policy": (
                        "known robustness conditions outside the requested condition "
                        "roster are audited then ignored; unknown conditions are rejected"
                    ),
                    "known_condition_roster": list(KNOWN_CONDITIONS),
                    "prediction_total": sum(ignored_prediction_by_condition.values()),
                    "prediction_by_condition": dict(
                        sorted(ignored_prediction_by_condition.items())
                    ),
                    "sidecar_total": sum(ignored_sidecar_by_condition.values()),
                    "sidecar_by_condition": dict(
                        sorted(ignored_sidecar_by_condition.items())
                    ),
                    "by_run": ignored_by_run,
                },
                "manifest": manifest_audit,
                "prediction_pixel_identity": prediction_pixel_audit,
                "sarn_sidecar_pixel_identity": sarn_pixel_audit,
            },
            "cells": cells_output,
            "factorial_effects": factorial_effects,
            "key_condition_paired_complete_group_bootstrap_vs_00": key_bootstrap,
        },
        methods,
    )


def score_factorial_spec(
    spec: Mapping[str, Any],
    *,
    base_dir: Path,
    spec_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "protocol",
        "prediction_variant",
        "robustness_seed",
        "seeds",
        "bootstrap",
        "checkpoints",
        "datasets",
    }
    _require(set(spec) == allowed, "factorial score spec schema drift")
    _require(spec.get("schema_version") == 1, "factorial score spec version drift")
    _require(spec.get("protocol") == SPEC_PROTOCOL, "factorial score spec protocol drift")
    prediction_variant = _text(
        spec.get("prediction_variant"), label="prediction_variant"
    )
    _require(
        prediction_variant in {"raw", "sarn"},
        "prediction_variant must be raw or sarn",
    )
    robustness_seed = _integer(spec.get("robustness_seed"), label="robustness_seed")
    raw_seeds = spec.get("seeds")
    _require(isinstance(raw_seeds, list), "seeds must be a list")
    seeds = tuple(
        _integer(seed, label=f"seeds[{index}]")
        for index, seed in enumerate(raw_seeds)
    )
    _require(len(seeds) == 3 and len(set(seeds)) == 3, "exactly three unique seeds are required")

    bootstrap = _mapping(spec.get("bootstrap"), label="bootstrap")
    _require(set(bootstrap) == {"seed", "replicates"}, "bootstrap schema drift")
    bootstrap_seed = _integer(bootstrap.get("seed"), label="bootstrap.seed")
    bootstrap_replicates = _positive_integer(
        bootstrap.get("replicates"), label="bootstrap.replicates"
    )

    checkpoints = _mapping(spec.get("checkpoints"), label="checkpoints")
    _require(set(checkpoints) == set(CELLS), "checkpoints must contain the strict 2x2 cells")
    checkpoint_bindings: dict[str, dict[str, dict[str, str]]] = {}
    checkpoint_paths: list[Path] = []
    for cell in CELLS:
        seed_map = _validate_seed_map(
            checkpoints[cell], seeds=seeds, label=f"checkpoints.{cell}"
        )
        checkpoint_bindings[cell] = {}
        for seed in seeds:
            path, binding = _artifact_binding(
                seed_map[str(seed)],
                label=f"checkpoints.{cell}.{seed}",
                base_dir=base_dir,
            )
            checkpoint_paths.append(path)
            checkpoint_bindings[cell][str(seed)] = binding
    _require(
        len(set(checkpoint_paths)) == len(checkpoint_paths),
        "checkpoint paths must be unique across all four cells and three seeds",
    )

    datasets_config = _mapping(spec.get("datasets"), label="datasets")
    _require(bool(datasets_config), "at least one dataset is required")
    datasets: dict[str, Any] = {}
    canonical_methods: dict[tuple[str, int], str] | None = None
    for raw_name, dataset_config in datasets_config.items():
        name = _text(raw_name, label="dataset name")
        dataset_result, methods = _score_dataset(
            name,
            dataset_config,
            base_dir=base_dir,
            seeds=seeds,
            prediction_variant=prediction_variant,
            robustness_seed=robustness_seed,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
        if canonical_methods is None:
            canonical_methods = methods
        else:
            _require(
                methods == canonical_methods,
                f"{name}: cell/seed method identities differ across datasets",
            )
        datasets[name] = dataset_result

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "spec_binding": dict(spec_binding) if spec_binding is not None else None,
        "factor_design": {
            "design": "strict_2x2_cbam_by_geometry_auxiliary",
            "cells": CELL_FACTORS,
            "cell_bit_order": "first_bit_cbam_second_bit_geometry_auxiliary",
            "reference_cell": "00",
            "seeds": list(seeds),
            "seed_count": 3,
            "response": "full_denominator_nmae",
            "lower_is_better": True,
            "factorial_effect_definitions": {
                "cbam_main_effect": "((NMAE_10+NMAE_11)-(NMAE_00+NMAE_01))/2",
                "geometry_auxiliary_main_effect": (
                    "((NMAE_01+NMAE_11)-(NMAE_00+NMAE_10))/2"
                ),
                "cbam_x_geometry_auxiliary_interaction": (
                    "NMAE_11-NMAE_10-NMAE_01+NMAE_00"
                ),
                "sign": "negative is beneficial because lower NMAE is better",
            },
        },
        "scoring_policy": {
            "prediction_variant": prediction_variant,
            "robustness_seed": robustness_seed,
            "normalized_target": "(ground_truth-scale_start)/(scale_end-scale_start)",
            "pass_error": "abs(normalized_progress-normalized_target)",
            "failure_error": 1.0,
            "seed_summary": "mean and sample standard deviation across exactly three seeds",
            "direction_vs_00": "strictly lower NMAE is improved; ties are not improvements",
            "extra_condition_rows": (
                "known all6 conditions outside each dataset's requested roster are "
                "audited then ignored; unknown conditions are rejected"
            ),
            "bootstrap": {
                "unit": "dataset-declared group_id cluster",
                "ci": "two-sided percentile 95%",
                "seed": bootstrap_seed,
                "replicates": bootstrap_replicates,
                "replicate_handling": (
                    "average three seed errors within sample before group resampling"
                ),
            },
        },
        "checkpoint_bindings": checkpoint_bindings,
        "checkpoint_bindings_sha256": _canonical_sha256(checkpoint_bindings),
        "cell_seed_method_identity": {
            cell: {
                str(seed): canonical_methods[(cell, seed)]
                for seed in seeds
            }
            for cell in CELLS
        },
        "datasets": datasets,
    }


def score_from_spec_path(path: Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactorialScoreError(f"cannot read factorial score spec: {source}") from exc
    _require(isinstance(value, Mapping), "factorial score spec root is not an object")
    return score_factorial_spec(
        value,
        base_dir=source.parent,
        spec_binding={"path": str(source), "sha256": _sha256_file(source)},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec_path = Path(args.spec).resolve(strict=True)
    output = Path(args.output).resolve()
    _require(output != spec_path, "output cannot overwrite the input spec")
    result = score_from_spec_path(spec_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        result,
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
                "datasets": sorted(result["datasets"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_FACTORS",
    "CELLS",
    "FactorialScoreError",
    "PROTOCOL",
    "SPEC_PROTOCOL",
    "main",
    "score_factorial_spec",
    "score_from_spec_path",
]
