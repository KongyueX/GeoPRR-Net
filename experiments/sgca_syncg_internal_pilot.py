"""Pure-SyncG internal scene-disjoint pilot helpers for SGCA experiments.

This module deliberately has no checkpoint loader and no default filesystem
paths.  It consumes an already-filtered copy of the *formal fit* rows plus the
formal-fit sample/scene whitelists, then makes a fixed 117-scene train /
14-scene internal-development partition.  Formal holdout and field rows are
therefore outside the accepted input surface.

The scoring helper is a single-training-seed pilot for opaque arms A1--A4 on
clean plus three projective conditions.  It reports full-denominator NMAE,
Acc@5, coverage, and complete-scene bootstrap intervals.  It is an internal
screen only, not paper confirmation and not a license to inspect the formal
holdout or field cohorts.
"""
from __future__ import annotations

import math
import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


PROTOCOL: Final[str] = "sgca_syncg_internal_scene_pilot_v1"
FIT_SAMPLES: Final[int] = 14_442
FIT_SCENES: Final[int] = 131
TRAIN_SCENES: Final[int] = 117
DEV_SCENES: Final[int] = 14
ARMS: Final[tuple[str, ...]] = ("A1", "A2", "A3", "A4")
ARM_DEFINITIONS: Final[dict[str, str]] = {
    "A1": "original projective ROI only",
    "A2": "original plus SARN with deterministic convex fusion",
    "A3": "probabilistic reliability fusion of original and SARN",
    "A4": (
        "full support-geometry conditioned attention with probabilistic "
        "multi-view fusion"
    ),
}
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = CONDITIONS[1:]
SCOPES: Final[dict[str, tuple[str, ...]]] = {
    **{condition: (condition,) for condition in CONDITIONS},
    "projective_pooled": PROJECTIVE_CONDITIONS,
}
PAIRWISE_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("A2", "A1"),
    ("A3", "A1"),
    ("A4", "A1"),
    ("A3", "A2"),
    ("A4", "A2"),
    ("A4", "A3"),
)
FAILURE_ERROR: Final[float] = 1.0
ACC_AT_5_THRESHOLD: Final[float] = 0.05

# Fixed internal-development roster selected from the 131 formal-fit scenes.
# The explicit names plus ordinary membership/count checks make the split
# reviewable and independent of input row order.
INTERNAL_DEV_SCENE_STEMS: Final[tuple[str, ...]] = (
    "blocky_photo_studio_4k",
    "sunflowers_puresky_4k",
    "resting_place_2_4k",
    "rosendal_park_sunset_puresky_4k",
    "simons_town_road_4k",
    "rustig_koppie_puresky_4k",
    "tief_etz_4k",
    "sunset_jhbcentral_4k",
    "rogland_moonlit_night_4k",
    "je_gray_park_4k",
    "rostock_laage_airport_4k",
    "industrial_sunset_02_puresky_4k",
    "dreifaltigkeitsberg_4k",
    "scythian_tombs_2_4k",
)
_FORBIDDEN_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {"target", "ground_truth", "scale_start", "scale_end", "label"}
)


class SGCAPilotError(RuntimeError):
    """The internal pilot input or scoring configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SGCAPilotError(message)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SGCAPilotError(label) from exc
    _require(math.isfinite(result), label)
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    _require(bool(values) and 0.0 <= probability <= 1.0, "invalid quantile input")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scene_stem(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), "SyncG fit row lacks metadata")
    scene_name = str(metadata.get("scene_name", ""))
    _require(bool(scene_name), "SyncG fit row has an empty scene_name")
    stem = Path(scene_name).stem
    _require(bool(stem), "SyncG fit row has an invalid scene_name")
    return stem


@dataclass(frozen=True, slots=True)
class InternalSceneSplit:
    protocol: str
    train_scene_stems: tuple[str, ...]
    dev_scene_stems: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    dev_sample_ids: tuple[str, ...]
    scene_by_sample: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "source_scope": "formal SyncG fit only",
            "formal_holdout_accessed": False,
            "field_data_accessed": False,
            "train_scene_stems": list(self.train_scene_stems),
            "dev_scene_stems": list(self.dev_scene_stems),
            "train_sample_ids": list(self.train_sample_ids),
            "dev_sample_ids": list(self.dev_sample_ids),
            "identity": {
                "fit_samples": len(self.train_sample_ids) + len(self.dev_sample_ids),
                "fit_scenes": len(self.train_scene_stems) + len(self.dev_scene_stems),
                "train_samples": len(self.train_sample_ids),
                "dev_samples": len(self.dev_sample_ids),
                "train_scenes": len(self.train_scene_stems),
                "dev_scenes": len(self.dev_scene_stems),
                "scene_overlap": len(
                    set(self.train_scene_stems) & set(self.dev_scene_stems)
                ),
            },
        }


def split_formal_fit_scene_index(
    scene_by_sample: Mapping[str, str],
) -> InternalSceneSplit:
    """Split an already-authorized formal-fit sample/scene index.

    This is the minimal training-driver API: the caller supplies its 14,442
    pure-SyncG formal-fit samples as ``sample_id -> scene_stem`` and uses the
    returned ID sets to partition its in-memory sample objects.  No split file,
    checkpoint, formal holdout, or field path is accepted here.
    """

    normalized = {
        str(sample_id): str(scene_stem)
        for sample_id, scene_stem in scene_by_sample.items()
    }
    _require(
        len(normalized) == FIT_SAMPLES and all(normalized),
        "formal fit scene index must contain 14442 distinct sample IDs",
    )
    _require(
        all(normalized.values()),
        "formal fit scene index contains an empty scene stem",
    )
    scenes = set(normalized.values())
    _require(len(scenes) == FIT_SCENES, "formal fit scene index must contain 131 scenes")
    dev_scenes = set(INTERNAL_DEV_SCENE_STEMS)
    _require(
        len(INTERNAL_DEV_SCENE_STEMS) == DEV_SCENES and dev_scenes <= scenes,
        "fixed internal dev scenes are not a subset of the formal fit",
    )
    train_scenes = scenes - dev_scenes
    _require(
        len(train_scenes) == TRAIN_SCENES and len(dev_scenes) == DEV_SCENES,
        "internal scene split cardinality mismatch",
    )
    train_sample_ids = tuple(
        sorted(
            sample_id
            for sample_id, scene in normalized.items()
            if scene in train_scenes
        )
    )
    dev_sample_ids = tuple(
        sorted(
            sample_id
            for sample_id, scene in normalized.items()
            if scene in dev_scenes
        )
    )
    _require(
        set(train_sample_ids).isdisjoint(dev_sample_ids)
        and len(train_sample_ids) + len(dev_sample_ids) == FIT_SAMPLES,
        "internal sample split is not a complete disjoint partition",
    )
    return InternalSceneSplit(
        protocol=PROTOCOL,
        train_scene_stems=tuple(sorted(train_scenes)),
        dev_scene_stems=INTERNAL_DEV_SCENE_STEMS,
        train_sample_ids=train_sample_ids,
        dev_sample_ids=dev_sample_ids,
        scene_by_sample=normalized,
    )


def build_internal_scene_split(
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    formal_fit_sample_ids: Collection[str],
    formal_fit_scene_stems: Collection[str],
) -> InternalSceneSplit:
    """Partition only the whitelisted formal-fit rows into fixed train/dev scenes."""

    allowed_samples = {str(value) for value in formal_fit_sample_ids}
    allowed_scenes = {str(value) for value in formal_fit_scene_stems}
    _require(len(allowed_samples) == FIT_SAMPLES, "formal fit sample roster must have 14442 IDs")
    _require(len(allowed_scenes) == FIT_SCENES, "formal fit scene roster must have 131 stems")
    _require(len(fit_rows) == FIT_SAMPLES, "formal fit rows must contain 14442 samples")
    _require(
        len(INTERNAL_DEV_SCENE_STEMS) == DEV_SCENES
        and set(INTERNAL_DEV_SCENE_STEMS) <= allowed_scenes,
        "fixed internal dev scenes are not a subset of the formal fit",
    )
    scene_by_sample: dict[str, str] = {}
    for row in fit_rows:
        _require(str(row.get("dataset", "")) == "SyncG", "non-SyncG row in formal fit input")
        sample_id = str(row.get("sample_id", ""))
        _require(sample_id in allowed_samples, "row is outside the formal fit sample whitelist")
        _require(sample_id not in scene_by_sample, f"duplicate formal fit sample: {sample_id}")
        scene = _scene_stem(row)
        _require(scene in allowed_scenes, "row is outside the formal fit scene whitelist")
        scene_by_sample[sample_id] = scene

    _require(set(scene_by_sample) == allowed_samples, "formal fit sample roster is incomplete")
    _require(set(scene_by_sample.values()) == allowed_scenes, "formal fit scene roster is incomplete")
    return split_formal_fit_scene_index(scene_by_sample)


def _metric_point(errors: Sequence[float], passed: Sequence[bool]) -> dict[str, float]:
    _require(len(errors) == len(passed) and bool(errors), "metric arrays are misaligned")
    return {
        "nmae": sum(errors) / len(errors),
        "acc_at_5pct": sum(error <= ACC_AT_5_THRESHOLD for error in errors) / len(errors),
        "coverage": sum(bool(value) for value in passed) / len(passed),
    }


def _scene_bootstrap_metrics(
    errors: Sequence[float],
    passed: Sequence[bool],
    scenes: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(
        len(errors) == len(passed) == len(scenes) and bool(errors),
        "scene bootstrap arrays are misaligned",
    )
    _require(replicates >= 1, "scene bootstrap replicates must be positive")
    scene_rows: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        scene_rows.setdefault(scene, []).append(index)
    scene_ids = sorted(scene_rows)
    _require(len(scene_ids) >= 2, "scene bootstrap needs at least two scenes")
    summaries = {
        scene: (
            len(indices),
            sum(errors[index] for index in indices),
            sum(errors[index] <= ACC_AT_5_THRESHOLD for index in indices),
            sum(bool(passed[index]) for index in indices),
        )
        for scene, indices in scene_rows.items()
    }
    rng = random.Random(seed)
    draws = {"nmae": [], "acc_at_5pct": [], "coverage": []}
    for _ in range(replicates):
        sampled = [summaries[rng.choice(scene_ids)] for _scene in scene_ids]
        rows = sum(value[0] for value in sampled)
        draws["nmae"].append(sum(value[1] for value in sampled) / rows)
        draws["acc_at_5pct"].append(sum(value[2] for value in sampled) / rows)
        draws["coverage"].append(sum(value[3] for value in sampled) / rows)
    return {
        name: {"low": _quantile(values, 0.025), "high": _quantile(values, 0.975)}
        for name, values in draws.items()
    } | {"scenes": len(scene_ids), "replicates": replicates, "seed": seed}


def _paired_scene_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    scenes: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(
        len(candidate_errors) == len(comparator_errors) == len(scenes)
        and bool(scenes),
        "paired scene bootstrap arrays are misaligned",
    )
    _require(replicates >= 1, "paired scene bootstrap replicates must be positive")
    scene_rows: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        scene_rows.setdefault(scene, []).append(index)
    scene_ids = sorted(scene_rows)
    _require(len(scene_ids) >= 2, "paired scene bootstrap needs at least two scenes")
    summaries = {
        scene: (
            len(indices),
            sum(candidate_errors[index] - comparator_errors[index] for index in indices),
            sum(
                int(candidate_errors[index] <= ACC_AT_5_THRESHOLD)
                - int(comparator_errors[index] <= ACC_AT_5_THRESHOLD)
                for index in indices
            ),
        )
        for scene, indices in scene_rows.items()
    }
    rng = random.Random(seed)
    nmae_draws: list[float] = []
    acc_draws: list[float] = []
    for _ in range(replicates):
        sampled = [summaries[rng.choice(scene_ids)] for _scene in scene_ids]
        rows = sum(value[0] for value in sampled)
        nmae_draws.append(sum(value[1] for value in sampled) / rows)
        acc_draws.append(sum(value[2] for value in sampled) / rows)
    nmae_delta = sum(
        candidate - comparator
        for candidate, comparator in zip(candidate_errors, comparator_errors, strict=True)
    ) / len(scenes)
    acc_delta = sum(
        int(candidate <= ACC_AT_5_THRESHOLD) - int(comparator <= ACC_AT_5_THRESHOLD)
        for candidate, comparator in zip(candidate_errors, comparator_errors, strict=True)
    ) / len(scenes)
    return {
        "delta_definition": "candidate_minus_comparator",
        "delta_nmae": nmae_delta,
        "delta_nmae_scene_bootstrap_ci95": {
            "low": _quantile(nmae_draws, 0.025),
            "high": _quantile(nmae_draws, 0.975),
        },
        "delta_acc_at_5pct": acc_delta,
        "delta_acc_at_5pct_scene_bootstrap_ci95": {
            "low": _quantile(acc_draws, 0.025),
            "high": _quantile(acc_draws, 0.975),
        },
        "scenes": len(scene_ids),
        "replicates": replicates,
        "seed": seed,
    }


def score_single_seed_pilot(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    dev_sample_ids: Collection[str],
    scene_by_sample: Mapping[str, str],
    targets: Mapping[str, float],
    training_seed: int,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20260817,
) -> dict[str, Any]:
    """Score a complete A1--A4 single-seed internal-development prediction grid."""

    dev_ids = {str(value) for value in dev_sample_ids}
    _require(bool(dev_ids), "internal dev sample roster is empty")
    _require(set(scene_by_sample) >= dev_ids, "scene map does not cover internal dev")
    _require(set(targets) == dev_ids, "target roster must exactly match internal dev")
    scoring_scenes = {str(scene_by_sample[sample_id]) for sample_id in dev_ids}
    _require(
        scoring_scenes == set(INTERNAL_DEV_SCENE_STEMS),
        "internal dev scoring must contain exactly the fixed 14 scenes",
    )
    expected_keys = {
        (arm, sample_id, condition)
        for arm in ARMS
        for sample_id in dev_ids
        for condition in CONDITIONS
    }
    parsed: dict[tuple[str, str, str], tuple[float, bool]] = {}
    for row in prediction_rows:
        leaked = _FORBIDDEN_PREDICTION_KEYS & set(row)
        _require(not leaked, f"prediction row contains label fields: {sorted(leaked)}")
        _require(int(row.get("seed", -1)) == int(training_seed), "prediction seed mismatch")
        arm = str(row.get("arm", ""))
        sample_id = str(row.get("sample_id", ""))
        condition = str(row.get("condition", ""))
        status = str(row.get("status", ""))
        key = (arm, sample_id, condition)
        _require(key in expected_keys, "prediction row is outside the pilot Cartesian roster")
        _require(key not in parsed, f"duplicate pilot prediction: {key}")
        _require(status in {"pass", "fail"}, "prediction status must be pass or fail")
        passed = status == "pass"
        if passed:
            value = _finite_float(
                row.get("normalized_progress"), label="invalid normalized_progress"
            )
            _require(0.0 <= value <= 1.0, "normalized_progress outside [0,1]")
        else:
            _require(row.get("normalized_progress") is None, "failed row carries a prediction")
            value = FAILURE_ERROR
        parsed[key] = (value, passed)
    _require(set(parsed) == expected_keys, "pilot predictions are not a complete Cartesian grid")

    ordered_samples = sorted(dev_ids)
    vectors: dict[str, dict[str, dict[str, list[Any]]]] = {arm: {} for arm in ARMS}
    arm_results: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        arm_results[arm] = {"conditions": {}}
        for scope_index, (scope, conditions) in enumerate(SCOPES.items()):
            errors: list[float] = []
            passed_values: list[bool] = []
            scenes: list[str] = []
            for sample_id in ordered_samples:
                target = _finite_float(targets[sample_id], label="invalid pilot target")
                _require(0.0 <= target <= 1.0, "pilot target outside [0,1]")
                for condition in conditions:
                    prediction, passed = parsed[(arm, sample_id, condition)]
                    errors.append(abs(prediction - target) if passed else FAILURE_ERROR)
                    passed_values.append(passed)
                    scenes.append(str(scene_by_sample[sample_id]))
            vectors[arm][scope] = {
                "errors": errors,
                "passed": passed_values,
                "scenes": scenes,
            }
            arm_results[arm]["conditions"][scope] = {
                "rows": len(errors),
                "metrics": _metric_point(errors, passed_values),
                "scene_bootstrap_ci95": _scene_bootstrap_metrics(
                    errors,
                    passed_values,
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + arm_index * len(SCOPES) + scope_index,
                ),
            }

    paired: dict[str, Any] = {}
    for comparison_index, (candidate, comparator) in enumerate(PAIRWISE_COMPARISONS):
        label = f"{candidate}_minus_{comparator}"
        paired[label] = {
            "candidate": candidate,
            "comparator": comparator,
            "conditions": {},
        }
        for scope_index, scope in enumerate(SCOPES):
            candidate_vector = vectors[candidate][scope]
            comparator_vector = vectors[comparator][scope]
            _require(
                candidate_vector["scenes"] == comparator_vector["scenes"],
                "paired pilot scenes are misaligned",
            )
            paired[label]["conditions"][scope] = _paired_scene_bootstrap(
                candidate_vector["errors"],
                comparator_vector["errors"],
                candidate_vector["scenes"],
                replicates=bootstrap_replicates,
                seed=(
                    bootstrap_seed
                    + 100
                    + comparison_index * len(SCOPES)
                    + scope_index
                ),
            )

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "evaluation_role": "pure-SyncG internal scene-disjoint single-seed pilot",
        "paper_claim_eligible": False,
        "formal_holdout_accessed": False,
        "field_data_accessed": False,
        "training_seed": int(training_seed),
        "arms": list(ARMS),
        "arm_definitions": dict(ARM_DEFINITIONS),
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "samples": len(dev_ids),
        "scenes": DEV_SCENES,
        "failure_error": FAILURE_ERROR,
        "metrics": ["nmae", "acc_at_5pct", "coverage"],
        "arms_results": arm_results,
        "paired_comparisons": paired,
        "interpretation_boundary": (
            "Use only to choose whether SGCA warrants a later independently fixed "
            "multi-seed confirmation. Do not tune on or report the formal SyncG "
            "holdout or any field cohort from this pilot."
        ),
    }


__all__ = [
    "ARMS",
    "ARM_DEFINITIONS",
    "CONDITIONS",
    "DEV_SCENES",
    "FIT_SCENES",
    "FIT_SAMPLES",
    "INTERNAL_DEV_SCENE_STEMS",
    "InternalSceneSplit",
    "PAIRWISE_COMPARISONS",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "SGCAPilotError",
    "SCOPES",
    "TRAIN_SCENES",
    "build_internal_scene_split",
    "score_single_seed_pilot",
    "split_formal_fit_scene_index",
]
