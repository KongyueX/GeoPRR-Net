"""Summarize ReMSTNet and VDN on their independently held-out intersection.

The ReMSTNet scene holdout and the terminal epoch-200 VDN grouped holdout were
fixed by different protocols.  Their aggregate metrics are therefore not
directly mergeable.  This utility selects only samples present in both
holdouts and compares predictions on the same six condition pixels.

Two VDN views remain separate:

* automatic reference is the deployable ROI-to-progress pipeline and scores a
  missing reading as the maximum normalized error (1.0); and
* annotation reference reuses the same VDN direction prediction but converts
  it with the SyncG reference arc.  It provides the complete-intersection
  paper-facing component comparison, but remains non-deployable and is not an
  input-equivalent automatic-system result.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

PROTOCOL: Final[str] = "remstnet_v3_vdn_double_holdout_intersection_v2"
REMST_EVALUATION_PROTOCOL: Final[str] = "remst_block_syncg_pilot_evaluation_v1"
EXPECTED_SOURCE_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
VDN_AUTOMATIC_PROTOCOL: Final[str] = "cagh_v5_plain_paper_batch_v1"
VDN_AUTOMATIC_METHOD: Final[str] = "vdn_official200_terminal_seed20"
VDN_ORACLE_PROTOCOL: Final[str] = "vdn_syncg_oracle_reference_component_v1"
VDN_ORACLE_METHOD: Final[str] = (
    "vdn_official200_terminal_seed20_oracle_reference_component"
)
PROJECTIVE_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"perspective_moderate", "perspective_severe", "combined_severe"}
)
FAILURE_ERROR: Final[float] = 1.0
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260821


class ReMSTNetVDNIntersectionError(ValueError):
    """The two frozen evaluation ledgers cannot be compared as supplied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetVDNIntersectionError(message)


def _metrics(errors: Sequence[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    _require(bool(values.size), "metric vector is empty")
    _require(
        bool(np.isfinite(values).all())
        and bool(np.all((0.0 <= values) & (values <= FAILURE_ERROR))),
        "normalized errors are invalid",
    )
    return {
        "nmae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_absolute_error": float(np.quantile(values, 0.95)),
        "maximum_absolute_error": float(values.max()),
        "acc_at_1_percent": float(np.mean(values <= 0.01)),
        "acc_at_2_percent": float(np.mean(values <= 0.02)),
        "acc_at_5_percent": float(np.mean(values <= 0.05)),
    }


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "seed statistic is empty")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
    }


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSONL input does not exist: {source}")
    rows: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReMSTNetVDNIntersectionError(
                    f"invalid JSONL row: {source}:{line_number}"
                ) from exc
            _require(
                isinstance(value, Mapping),
                f"JSONL row is not an object: {source}:{line_number}",
            )
            rows.append(value)
    _require(bool(rows), f"JSONL input is empty: {source}")
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")), str(row.get("condition", ""))


def _candidate(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("remstnet", row.get("mett"))
    _require(isinstance(value, Mapping), "ReMSTNet prediction block is missing")
    return value


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        _require(key[0] and key[1] in CONDITIONS, f"{label} row key is invalid")
        _require(key not in indexed, f"{label} row key repeats: {key}")
        indexed[key] = row
    return indexed


def _load_remst_evaluation(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"ReMSTNet evaluation does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, Mapping), "ReMSTNet evaluation is malformed")
    _require(
        value.get("protocol") == REMST_EVALUATION_PROTOCOL
        and value.get("status") == "pilot_complete",
        "ReMSTNet evaluation protocol or status differs",
    )
    model = value.get("model")
    _require(isinstance(model, Mapping), "ReMSTNet model metadata is missing")
    foundation = model.get("source_foundation")
    _require(isinstance(foundation, Mapping), "source foundation metadata is missing")
    source_seed = int(foundation.get("source_seed", -1))
    _require(source_seed in EXPECTED_SOURCE_SEEDS, "ReMSTNet source seed differs")
    raw_rows = value.get("per_sample_condition")
    _require(isinstance(raw_rows, list) and bool(raw_rows), "ReMSTNet rows are missing")
    rows = _index_rows(raw_rows, label="ReMSTNet")
    condition_pixels = {
        key: str(row.get("condition_pixel_sha256", "")) for key, row in rows.items()
    }
    if not all(condition_pixels.values()):
        data = value.get("data")
        _require(
            isinstance(data, Mapping) and bool(data.get("reference")),
            "ReMSTNet condition pixels and reference ledger are both absent",
        )
        reference_path = Path(str(data["reference"])).resolve()
        _require(
            reference_path.is_file(),
            f"ReMSTNet reference ledger does not exist: {reference_path}",
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
        reference_rows = reference.get("per_sample_condition")
        _require(
            isinstance(reference_rows, list) and bool(reference_rows),
            "ReMSTNet reference rows are missing",
        )
        reference_index = _index_rows(reference_rows, label="ReMSTNet reference")
        _require(
            set(reference_index) == set(rows),
            "ReMSTNet reference roster differs",
        )
        condition_pixels = {
            key: str(reference_index[key].get("condition_pixel_sha256", ""))
            for key in rows
        }
        _require(
            all(condition_pixels.values()),
            "ReMSTNet reference condition pixel identity is absent",
        )
    for key, row in rows.items():
        target = float(row.get("normalized_target"))
        record = _candidate(row)
        prediction = float(record.get("prediction"))
        error = float(record.get("absolute_error"))
        _require(
            math.isfinite(target)
            and 0.0 <= target <= 1.0
            and math.isfinite(prediction)
            and 0.0 <= prediction <= 1.0
            and abs(abs(prediction - target) - error) <= 1.0e-8,
            f"ReMSTNet prediction/error is inconsistent: {key}",
        )
    return {
        "path": str(source),
        "source_seed": source_seed,
        "rows": rows,
        "condition_pixels": condition_pixels,
    }


def _load_vdn(
    path: Path, *, protocol: str, method: str, oracle: bool
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = _read_jsonl(path)
    indexed = _index_rows(rows, label="VDN oracle" if oracle else "VDN automatic")
    for key, row in indexed.items():
        _require(
            row.get("protocol") == protocol and row.get("method") == method,
            f"VDN protocol or method differs: {key}",
        )
        status = str(row.get("status"))
        _require(status in {"pass", "fail"}, f"VDN status is invalid: {key}")
        prediction = row.get("normalized_progress")
        if status == "pass":
            scalar = float(prediction)
            _require(
                math.isfinite(scalar) and 0.0 <= scalar <= 1.0,
                f"VDN progress is invalid: {key}",
            )
        else:
            _require(prediction is None, f"failed VDN row carries progress: {key}")
        if oracle:
            _require(status == "pass", f"VDN oracle row failed: {key}")
            _require(
                row.get("deployable") is False
                and row.get("primary_table_eligible") is False
                and row.get("oracle_reference_used_for_offline_progress_conversion")
                is True
                and row.get("used_as_runtime_input") is False,
                f"VDN oracle disclosure differs: {key}",
            )
    return indexed


def _scope_keys(
    keys: Sequence[tuple[str, str]], label: str
) -> tuple[tuple[str, str], ...]:
    if label == "all_conditions":
        return tuple(keys)
    if label == "projective_pooled":
        return tuple(key for key in keys if key[1] in PROJECTIVE_CONDITIONS)
    return tuple(key for key in keys if key[1] == label)


def _paired_scene_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    scenes: Sequence[str],
    *,
    replicates: int,
    seed: int,
    comparator: str,
) -> dict[str, Any]:
    _require(replicates >= 1, "bootstrap replicates must be positive")
    _require(
        len(candidate_errors) == len(comparator_errors) == len(scenes)
        and bool(scenes),
        "paired vectors are misaligned",
    )
    effects = np.asarray(candidate_errors, dtype=np.float64) - np.asarray(
        comparator_errors, dtype=np.float64
    )
    grouped: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        grouped.setdefault(str(scene), []).append(index)
    scene_ids = tuple(sorted(grouped))
    _require(len(scene_ids) >= 2, "scene bootstrap needs at least two scenes")
    sums = np.asarray([effects[grouped[scene]].sum() for scene in scene_ids])
    counts = np.asarray([len(grouped[scene]) for scene in scene_ids])
    rng = random.Random(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = [rng.randrange(len(scene_ids)) for _ in scene_ids]
        draws[replicate] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "comparator": comparator,
        "delta_definition": "ReMSTNet-v3 rowwise three-seed mean error minus comparator error",
        "delta_nmae": float(effects.mean()),
        "scene_grouped_bootstrap_ci95": {"low": low, "high": high},
        "superiority_ci95": high < 0.0,
        "scene_clusters": len(scene_ids),
        "rows": len(effects),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _paired_scene_bootstrap_if_identifiable(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    scenes: Sequence[str],
    *,
    replicates: int,
    seed: int,
    comparator: str,
) -> dict[str, Any] | None:
    """Return a grouped interval only when at least two scene groups remain."""

    if len(set(scenes)) < 2:
        return None
    return _paired_scene_bootstrap(
        candidate_errors,
        comparator_errors,
        scenes,
        replicates=replicates,
        seed=seed,
        comparator=comparator,
    )


def _external_b0_errors(
    row: Mapping[str, Any], target: float
) -> dict[int, float]:
    external = row.get("external")
    _require(isinstance(external, Mapping), "external model block is missing")
    block = external.get("EfficientNet-B0")
    _require(isinstance(block, Mapping), "EfficientNet-B0 block is missing")
    seeds = tuple(int(value) for value in block.get("seeds", ()))
    predictions = tuple(float(value) for value in block.get("predictions", ()))
    errors = tuple(float(value) for value in block.get("absolute_errors", ()))
    passed = tuple(bool(value) for value in block.get("passed", ()))
    _require(
        seeds == EXPECTED_SOURCE_SEEDS
        and len(predictions) == len(errors) == len(passed) == len(seeds)
        and all(passed),
        "EfficientNet-B0 seed vector differs",
    )
    _require(
        all(
            abs(abs(prediction - target) - error) <= 1.0e-8
            for prediction, error in zip(predictions, errors, strict=True)
        ),
        "EfficientNet-B0 prediction/error is inconsistent",
    )
    return dict(zip(seeds, errors, strict=True))


def summarize_vdn_intersection(
    evaluation_paths: Sequence[Path],
    vdn_automatic_path: Path,
    vdn_oracle_path: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build the aggregate comparison without mixing non-identical cohorts."""

    _require(len(evaluation_paths) == 3, "exactly three ReMSTNet evaluations required")
    evaluations = [_load_remst_evaluation(path) for path in evaluation_paths]
    evaluations.sort(key=lambda item: int(item["source_seed"]))
    _require(
        tuple(int(item["source_seed"]) for item in evaluations)
        == EXPECTED_SOURCE_SEEDS,
        "ReMSTNet source seed roster differs",
    )
    reference_rows = evaluations[0]["rows"]
    reference_pixels = evaluations[0]["condition_pixels"]
    reference_keys = tuple(reference_rows)
    for evaluation in evaluations[1:]:
        _require(
            tuple(evaluation["rows"]) == reference_keys,
            "ReMSTNet evaluation row order differs",
        )
        for key in reference_keys:
            left = reference_rows[key]
            right = evaluation["rows"][key]
            _require(
                str(left.get("scene_stem")) == str(right.get("scene_stem"))
                and float(left.get("normalized_target"))
                == float(right.get("normalized_target"))
                and reference_pixels[key] == evaluation["condition_pixels"][key],
                f"ReMSTNet evaluation alignment differs: {key}",
            )

    automatic = _load_vdn(
        vdn_automatic_path,
        protocol=VDN_AUTOMATIC_PROTOCOL,
        method=VDN_AUTOMATIC_METHOD,
        oracle=False,
    )
    oracle = _load_vdn(
        vdn_oracle_path,
        protocol=VDN_ORACLE_PROTOCOL,
        method=VDN_ORACLE_METHOD,
        oracle=True,
    )
    common = set(reference_rows) & set(automatic) & set(oracle)
    complete_samples = tuple(
        sorted(
            sample_id
            for sample_id in {key[0] for key in common}
            if {key[1] for key in common if key[0] == sample_id} == set(CONDITIONS)
        )
    )
    _require(bool(complete_samples), "the two holdouts have no complete intersection")
    complete = set(complete_samples)
    condition_rank = {condition: index for index, condition in enumerate(CONDITIONS)}
    selected_keys = tuple(
        sorted(
            (key for key in common if key[0] in complete),
            key=lambda key: (key[0], condition_rank[key[1]]),
        )
    )
    _require(
        len(selected_keys) == len(complete_samples) * len(CONDITIONS),
        "intersection is not a complete sample-by-condition product",
    )

    # Sample IDs and condition names alone cannot prove that independently
    # generated degradations used identical pixels.  Git state, versions,
    # primary keys, transactions, uniqueness/type constraints, and ordinary
    # unit tests can all remain valid while one ledger was rendered with a
    # different degradation implementation or seed.  The ledgers already
    # carry condition-pixel digests, so this publication comparison reuses
    # them and refuses mismatched pixels; no new digest or baseline is created.
    for key in selected_keys:
        expected = reference_pixels[key]
        _require(bool(expected), f"ReMSTNet condition pixel identity is absent: {key}")
        _require(
            expected == str(automatic[key].get("condition_pixel_sha256", ""))
            == str(oracle[key].get("condition_pixel_sha256", "")),
            f"condition pixel identity differs: {key}",
        )

    scenes_all = {str(reference_rows[key].get("scene_stem", "")) for key in selected_keys}
    _require("" not in scenes_all and len(scenes_all) >= 2, "intersection scenes are invalid")
    scopes = ("all_conditions", "projective_pooled", *CONDITIONS)
    summary: dict[str, Any] = {}
    for scope_index, label in enumerate(scopes):
        keys = _scope_keys(selected_keys, label)
        _require(bool(keys), f"{label} intersection is empty")
        scenes = [str(reference_rows[key]["scene_stem"]) for key in keys]
        remst_by_seed = {
            int(evaluation["source_seed"]): [
                float(_candidate(evaluation["rows"][key])["absolute_error"])
                for key in keys
            ]
            for evaluation in evaluations
        }
        remst_per_seed = {
            str(seed): _metrics(errors) for seed, errors in remst_by_seed.items()
        }
        remst_mean_errors = [
            statistics.fmean(remst_by_seed[seed][index] for seed in EXPECTED_SOURCE_SEEDS)
            for index in range(len(keys))
        ]

        b0_by_seed: dict[int, list[float]] = {seed: [] for seed in EXPECTED_SOURCE_SEEDS}
        automatic_errors: list[float] = []
        automatic_success_errors: list[float] = []
        automatic_success_indices: list[int] = []
        oracle_errors: list[float] = []
        automatic_failures = 0
        for row_index, key in enumerate(keys):
            target = float(reference_rows[key]["normalized_target"])
            b0 = _external_b0_errors(reference_rows[key], target)
            for seed in EXPECTED_SOURCE_SEEDS:
                b0_by_seed[seed].append(b0[seed])
            automatic_row = automatic[key]
            if automatic_row["status"] == "pass":
                error = abs(float(automatic_row["normalized_progress"]) - target)
                automatic_success_errors.append(error)
                automatic_success_indices.append(row_index)
                automatic_errors.append(error)
            else:
                automatic_failures += 1
                automatic_errors.append(FAILURE_ERROR)
            oracle_errors.append(abs(float(oracle[key]["normalized_progress"]) - target))

        b0_per_seed = {
            str(seed): _metrics(errors) for seed, errors in b0_by_seed.items()
        }
        b0_mean_errors = [
            statistics.fmean(b0_by_seed[seed][index] for seed in EXPECTED_SOURCE_SEEDS)
            for index in range(len(keys))
        ]
        successful_keys = tuple(keys[index] for index in automatic_success_indices)
        successful_scenes = [scenes[index] for index in automatic_success_indices]
        successful_remst_by_seed = {
            seed: [errors[index] for index in automatic_success_indices]
            for seed, errors in remst_by_seed.items()
        }
        successful_remst_per_seed = {
            str(seed): _metrics(errors)
            for seed, errors in successful_remst_by_seed.items()
        }
        successful_remst_mean_errors = [
            remst_mean_errors[index] for index in automatic_success_indices
        ]
        successful_b0_by_seed = {
            seed: [errors[index] for index in automatic_success_indices]
            for seed, errors in b0_by_seed.items()
        }
        successful_b0_per_seed = {
            str(seed): _metrics(errors)
            for seed, errors in successful_b0_by_seed.items()
        }
        successful_b0_mean_errors = [
            b0_mean_errors[index] for index in automatic_success_indices
        ]
        _require(
            len(successful_keys)
            == len(automatic_success_errors)
            == len(successful_scenes),
            f"{label} VDN-success vectors are misaligned",
        )
        summary[label] = {
            "conditions": (
                list(CONDITIONS)
                if label == "all_conditions"
                else (
                    sorted(PROJECTIVE_CONDITIONS)
                    if label == "projective_pooled"
                    else [label]
                )
            ),
            "rows": len(keys),
            "samples": len({key[0] for key in keys}),
            "scene_groups": len(set(scenes)),
            "remstnet_v3": {
                "per_seed": remst_per_seed,
                "metric_across_seed_mean_sd": {
                    metric: _mean_sd(
                        [remst_per_seed[str(seed)][metric] for seed in EXPECTED_SOURCE_SEEDS]
                    )
                    for metric in next(iter(remst_per_seed.values()))
                },
                "rowwise_three_seed_mean_error_metrics": _metrics(remst_mean_errors),
            },
            "sarn_v2_efficientnet_b0": {
                "per_seed": b0_per_seed,
                "metric_across_seed_mean_sd": {
                    metric: _mean_sd(
                        [b0_per_seed[str(seed)][metric] for seed in EXPECTED_SOURCE_SEEDS]
                    )
                    for metric in next(iter(b0_per_seed.values()))
                },
                "rowwise_three_seed_mean_error_metrics": _metrics(b0_mean_errors),
            },
            "vdn_official200_automatic_reference": {
                "single_checkpoint": True,
                "deployable_from_canonical_roi": True,
                "coverage": float((len(keys) - automatic_failures) / len(keys)),
                "passes": len(keys) - automatic_failures,
                "failures": automatic_failures,
                "failure_inclusive_metrics": _metrics(automatic_errors),
                "successful_only_metrics": (
                    _metrics(automatic_success_errors)
                    if automatic_success_errors
                    else None
                ),
            },
            "vdn_official200_annotation_reference": {
                "single_checkpoint": True,
                "deployable": False,
                "paper_facing_secondary_table_eligible": True,
                "interpretation": (
                    "full-coverage direction-component comparison with an "
                    "annotation-derived reference arc"
                ),
                "coverage": 1.0,
                "metrics": _metrics(oracle_errors),
            },
            "vdn_automatic_success_conditioned_comparison": {
                "selection": (
                    "rows on which the automatic VDN reference stage returned "
                    "a finite normalized progress"
                ),
                "selection_depends_on_vdn_outcome": True,
                "interpretation": (
                    "conditional accuracy comparison on one identical row set; "
                    "coverage against the fixed intersection must be reported beside it"
                ),
                "rows": len(successful_keys),
                "samples": len({key[0] for key in successful_keys}),
                "scene_groups": len(set(successful_scenes)),
                "coverage_against_fixed_intersection": float(
                    len(successful_keys) / len(keys)
                ),
                "remstnet_v3": {
                    "per_seed": successful_remst_per_seed,
                    "metric_across_seed_mean_sd": {
                        metric: _mean_sd(
                            [
                                successful_remst_per_seed[str(seed)][metric]
                                for seed in EXPECTED_SOURCE_SEEDS
                            ]
                        )
                        for metric in next(iter(successful_remst_per_seed.values()))
                    },
                    "rowwise_three_seed_mean_error_metrics": _metrics(
                        successful_remst_mean_errors
                    ),
                },
                "sarn_v2_efficientnet_b0": {
                    "per_seed": successful_b0_per_seed,
                    "metric_across_seed_mean_sd": {
                        metric: _mean_sd(
                            [
                                successful_b0_per_seed[str(seed)][metric]
                                for seed in EXPECTED_SOURCE_SEEDS
                            ]
                        )
                        for metric in next(iter(successful_b0_per_seed.values()))
                    },
                    "rowwise_three_seed_mean_error_metrics": _metrics(
                        successful_b0_mean_errors
                    ),
                },
                "vdn_official200_automatic_reference": {
                    "single_checkpoint": True,
                    "successful_only": True,
                    "metrics": _metrics(automatic_success_errors),
                },
                "paired_scene_bootstrap": {
                    "versus_sarn_v2_efficientnet_b0": (
                        _paired_scene_bootstrap_if_identifiable(
                            successful_remst_mean_errors,
                            successful_b0_mean_errors,
                            successful_scenes,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + scope_index * 10 + 3,
                            comparator=(
                                "SARN-v2 + EfficientNet-B0 rowwise three-seed "
                                "mean on VDN-success rows"
                            ),
                        )
                    ),
                    "versus_vdn_automatic_successful": (
                        _paired_scene_bootstrap_if_identifiable(
                            successful_remst_mean_errors,
                            automatic_success_errors,
                            successful_scenes,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + scope_index * 10 + 4,
                            comparator=(
                                "VDN official-200 automatic reference on its "
                                "successful rows"
                            ),
                        )
                    ),
                },
            },
            "paired_scene_bootstrap": {
                "versus_sarn_v2_efficientnet_b0": _paired_scene_bootstrap(
                    remst_mean_errors,
                    b0_mean_errors,
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 10,
                    comparator="SARN-v2 + EfficientNet-B0 rowwise three-seed mean",
                ),
                "versus_vdn_automatic_failure_inclusive": _paired_scene_bootstrap(
                    remst_mean_errors,
                    automatic_errors,
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 10 + 1,
                    comparator="VDN official-200 automatic reference, failures=1.0",
                ),
                "versus_vdn_annotation_reference": _paired_scene_bootstrap(
                    remst_mean_errors,
                    oracle_errors,
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 10 + 2,
                    comparator="VDN official-200 annotation-reference component",
                ),
            },
        }

    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "status": "complete",
        "cohort": {
            "selection": "set intersection of independently fixed ReMSTNet and VDN holdouts",
            "selection_uses_predictions_or_errors": False,
            "remstnet_holdout_samples": len({key[0] for key in reference_rows}),
            "vdn_holdout_samples": len({key[0] for key in automatic}),
            "intersection_samples": len(complete_samples),
            "intersection_rows": len(selected_keys),
            "intersection_scene_groups": len(scenes_all),
            "conditions": list(CONDITIONS),
            "same_condition_pixels_verified": True,
        },
        "training_scope": {
            "same_training_roster": False,
            "same_evaluation_roster": False,
            "test_samples_excluded_from_both_independent_fit_rosters": True,
            "interpretation": "cross-split external-architecture comparison, not a replacement for the 1,558-sample main table",
        },
        "scoring": {
            "normalized_error": "abs(predicted_progress-normalized_target)",
            "automatic_vdn_failure_error": FAILURE_ERROR,
            "accuracy_failures_count_as_incorrect": True,
            "bootstrap_unit": "ReMSTNet scene_stem",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
        },
        "summary": summary,
        "reporting_notes": [
            "The intersection is defined by the two pre-existing holdout memberships, not by model outcomes.",
            "The paper-facing VDN comparison uses the annotation-reference conversion on every row of the fixed intersection.",
            "The annotation-derived pivot and ordered scale endpoints are used only to convert VDN's direction prediction to normalized progress; ReMSTNet and B0 do not receive them.",
            "Automatic-reference and success-conditioned VDN fields are retained only as machine-readable stage diagnostics and are not used in the paper-facing comparison.",
            "The annotation-reference result is a non-deployable component comparison, not an input-equivalent automatic-system result.",
            "VDN uses one official-200 checkpoint, whereas ReMSTNet and SARN-v2 + EfficientNet-B0 report three fitted seeds.",
            "The VDN grouped holdout belongs to its independent fitting protocol and is not the ReMSTNet test roster.",
            "This smaller intersection does not replace the 1,558-sample ReMSTNet main comparison.",
        ],
    }


def write_summary(
    evaluation_paths: Sequence[Path],
    vdn_automatic_path: Path,
    vdn_oracle_path: Path,
    output_path: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"summary output already exists: {output}")
    result = summarize_vdn_intersection(
        evaluation_paths,
        vdn_automatic_path,
        vdn_oracle_path,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--vdn-automatic", type=Path, required=True)
    parser.add_argument("--vdn-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = write_summary(
        args.evaluation,
        args.vdn_automatic,
        args.vdn_oracle,
        args.output,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    all_conditions = result["summary"]["all_conditions"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "intersection_samples": result["cohort"]["intersection_samples"],
                "remstnet_nmae": all_conditions["remstnet_v3"][
                    "metric_across_seed_mean_sd"
                ]["nmae"],
                "vdn_oracle_nmae": all_conditions[
                    "vdn_official200_annotation_reference"
                ]["metrics"]["nmae"],
                "vdn_automatic_coverage": all_conditions[
                    "vdn_official200_automatic_reference"
                ]["coverage"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "EXPECTED_SOURCE_SEEDS",
    "FAILURE_ERROR",
    "PROTOCOL",
    "REMST_EVALUATION_PROTOCOL",
    "ReMSTNetVDNIntersectionError",
    "summarize_vdn_intersection",
    "write_summary",
]
