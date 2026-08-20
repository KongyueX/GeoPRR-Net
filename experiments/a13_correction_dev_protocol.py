"""A13 correction-only scene split and descriptive development references.

The source population is exactly the physical 7,939-row, 72-scene A11
Core-train manifest.  Twelve scenes are chosen without model outputs by an
ordinary greedy balance over meter x normalized-target quartile and row
count.  A one-time seeded Fisher--Yates ordering is used only to break exact
score ties.  The seed is fixed before any A13 result and is never scanned.

The resulting 12 scenes are *correction development* data, not a system
holdout: the terminal A11 Raw anchor (q0) was already trained on all 72 source
scenes.  Therefore this split can only describe whether a newly trained,
frozen-q0 correction transfers to source-disjoint scenes.  It cannot estimate
end-to-end system generalization.

All numerical references below are descriptive.  This module deliberately
contains no automatic pass/fail, model-selection, or advancement gate.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from experiments.a11_scort_protocol import CORE_TRAIN_SCENES


PROTOCOL: Final[str] = "syncg_a13_correction_only_scene60_12_dev_v1"
SPLIT_SEED: Final[int] = 20_262_021
SOURCE_SAMPLES: Final[int] = 7_939
SOURCE_SCENES: Final[int] = 72
CORRECTION_TRAIN_SCENE_COUNT: Final[int] = 60
CORRECTION_DEV_SCENE_COUNT: Final[int] = 12
TARGET_QUARTILES: Final[int] = 4
ROW_BALANCE_WEIGHT: Final[float] = 0.25

EVALUATION_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = EVALUATION_CONDITIONS[1:]
EVALUATION_PIXEL_TOTAL_EPOCHS: Final[int] = 5
EVALUATION_PIXEL_EPOCH: Final[int] = 4

SPLIT_SELECTION_INPUTS: Final[tuple[str, ...]] = (
    "metadata.scene_name",
    "meter_id",
    "normalized_ground_truth_progress_quartile",
    "sample_count",
    "fixed_seed_20262021_for_exact_ties_only",
)
SPLIT_SELECTION_EXCLUDED_INPUTS: Final[tuple[str, ...]] = (
    "A11_or_A13_predictions",
    "A11_or_A13_losses",
    "A11_or_A13_checkpoint_parameters",
    "previous_Core_audit",
    "Fold-A",
    "Fold-B",
    "formal",
    "field",
)
SPLIT_SELECTION_METHOD: Final[str] = (
    "Sort the 72 A11-train scene names, apply one Fisher-Yates permutation "
    "with random.Random(20262021), and retain that rank only as the exact-tie "
    "breaker. For greedy step k=1..12, score every remaining scene after "
    "hypothetical addition. The primary score is the mean across all 20 "
    "meter_id x target-quartile cells of squared deviation from "
    "full_cell_count*k/72, each divided by the squared one-scene expected "
    "cell count (full_cell_count/72). Add 0.25 times the squared total-row "
    "deviation from 7939*k/72 divided by the squared one-scene expected row "
    "count (7939/72). Select minimum (score, fixed_tie_rank). The seed is not "
    "scanned and model results are never inputs."
)
DEVELOPMENT_INTERPRETATION: Final[str] = (
    "q0 was trained on every correction-dev scene; only the newly trained "
    "correction has a 60-train/12-dev scene boundary. Results therefore "
    "measure correction cross-scene generalization conditional on a seen-q0 "
    "anchor, not end-to-end system heldout generalization."
)

DESCRIPTIVE_REFERENCE: Final[dict[str, Any]] = {
    "projective_pooled_nmae_delta_final_minus_q0_maximum": -0.001,
    "projective_net_paired_win_minus_loss_minimum": 0.03,
    "projective_cvar25_delta_final_minus_q0_maximum": -0.001,
    "each_projective_condition_nmae_delta_final_minus_q0_maximum": 0.0002,
    "clean_final_posterior_and_moments_bit_exact_q0": True,
    "sarn_off_exact_q0": True,
    "posterior_mass_violation_total": 0,
    "cdf_monotonic_violation_total": 0,
    "automatic_execution_selection_or_advancement_control": False,
    "interpretation": DEVELOPMENT_INTERPRETATION,
}


class A13CorrectionProtocolError(ValueError):
    """The physical A11-train inventory or split evidence is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A13CorrectionProtocolError(message)


@dataclass(frozen=True, slots=True)
class PartitionDistribution:
    samples: int
    scenes: int
    target_mean: float
    target_population_std: float
    target_min: float
    target_max: float
    quartile_counts: Mapping[str, int]
    meter_counts: Mapping[str, int]
    meter_quartile_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scene_name(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), "A11-train row metadata is missing")
    scene = metadata.get("scene_name")
    _require(isinstance(scene, str) and bool(scene), "A11-train scene_name is missing")
    return scene


def _sample_id(row: Mapping[str, Any]) -> str:
    value = row.get("sample_id")
    _require(isinstance(value, str) and bool(value), "A11-train sample_id is missing")
    return value


def _meter_id(row: Mapping[str, Any]) -> str:
    value = row.get("meter_id")
    _require(isinstance(value, str) and bool(value), "A11-train meter_id is missing")
    return value


def _normalized_target(row: Mapping[str, Any]) -> float:
    try:
        value = float(row["ground_truth"])
        start = float(row["scale_start"])
        end = float(row["scale_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise A13CorrectionProtocolError("A11-train target fields are malformed") from exc
    _require(
        math.isfinite(value) and math.isfinite(start) and math.isfinite(end),
        "A11-train target fields are non-finite",
    )
    _require(end != start, "A11-train scale has zero range")
    target = (value - start) / (end - start)
    _require(
        math.isfinite(target) and 0.0 <= target <= 1.0,
        "A11-train normalized target is outside [0,1]",
    )
    return target


def _quartile(target: float) -> int:
    return min(TARGET_QUARTILES - 1, max(0, int(float(target) * TARGET_QUARTILES)))


def _cell(row: Mapping[str, Any]) -> str:
    return f"{_meter_id(row)}::q{_quartile(_normalized_target(row))}"


def _fisher_yates_tie_order(scene_names: Sequence[str]) -> tuple[str, ...]:
    values = sorted(str(scene) for scene in scene_names)
    generator = random.Random(SPLIT_SEED)
    for upper in range(len(values) - 1, 0, -1):
        index = generator.randrange(upper + 1)
        values[upper], values[index] = values[index], values[upper]
    return tuple(values)


def _partition_distribution(
    rows: Sequence[Mapping[str, Any]],
) -> PartitionDistribution:
    values = tuple(rows)
    _require(bool(values), "A13 split partition is empty")
    targets = tuple(_normalized_target(row) for row in values)
    mean = sum(targets) / len(targets)
    std = math.sqrt(sum((value - mean) ** 2 for value in targets) / len(targets))
    quartiles = Counter(f"q{_quartile(value)}" for value in targets)
    meters = Counter(_meter_id(row) for row in values)
    cells = Counter(_cell(row) for row in values)
    return PartitionDistribution(
        samples=len(values),
        scenes=len({_scene_name(row) for row in values}),
        target_mean=mean,
        target_population_std=std,
        target_min=min(targets),
        target_max=max(targets),
        quartile_counts=dict(sorted(quartiles.items())),
        meter_counts=dict(sorted(meters.items())),
        meter_quartile_counts=dict(sorted(cells.items())),
    )


def select_correction_dev_scenes(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_complete_source: bool = True,
) -> dict[str, Any]:
    """Run the predeclared ordinary greedy split using metadata only."""

    values = tuple(rows)
    _require(bool(values), "source A11-train rows are empty")
    ids = tuple(_sample_id(row) for row in values)
    _require(len(ids) == len(set(ids)), "source A11-train sample IDs are duplicated")
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in values:
        by_scene[_scene_name(row)].append(row)
    scene_names = tuple(sorted(by_scene))
    if require_complete_source:
        _require(len(values) == SOURCE_SAMPLES, "source A11-train row count differs")
        _require(len(scene_names) == SOURCE_SCENES, "source A11-train scene count differs")
        _require(
            set(scene_names) == set(CORE_TRAIN_SCENES),
            "source A11-train scene roster differs",
        )
    _require(
        len(scene_names) >= CORRECTION_DEV_SCENE_COUNT + 1,
        "source has too few scenes for the correction split",
    )

    full_cells = Counter(_cell(row) for row in values)
    cell_names = tuple(sorted(full_cells))
    expected_cell_per_scene = {
        name: float(full_cells[name]) / len(scene_names) for name in cell_names
    }
    _require(
        len(cell_names) >= 1
        and all(value > 0.0 for value in expected_cell_per_scene.values()),
        "source meter x quartile cells are empty",
    )
    scene_cells = {
        scene: Counter(_cell(row) for row in scene_rows)
        for scene, scene_rows in by_scene.items()
    }
    scene_rows = {scene: len(scene_values) for scene, scene_values in by_scene.items()}
    tie_order = _fisher_yates_tie_order(scene_names)
    tie_rank = {scene: rank for rank, scene in enumerate(tie_order)}
    selected: list[str] = []
    cumulative_cells: Counter[str] = Counter()
    cumulative_rows = 0
    trace: list[dict[str, Any]] = []
    expected_rows_per_scene = len(values) / len(scene_names)
    for step in range(1, CORRECTION_DEV_SCENE_COUNT + 1):
        candidates: list[tuple[float, int, str, int, Counter[str]]] = []
        for scene in scene_names:
            if scene in selected:
                continue
            proposed_cells = cumulative_cells + scene_cells[scene]
            cell_score = sum(
                (
                    (
                        proposed_cells[name]
                        - float(full_cells[name]) * step / len(scene_names)
                    )
                    / expected_cell_per_scene[name]
                )
                ** 2
                for name in cell_names
            ) / len(cell_names)
            proposed_rows = cumulative_rows + scene_rows[scene]
            row_score = (
                (
                    proposed_rows - len(values) * step / len(scene_names)
                )
                / expected_rows_per_scene
            ) ** 2
            score = float(cell_score + ROW_BALANCE_WEIGHT * row_score)
            candidates.append(
                (score, tie_rank[scene], scene, proposed_rows, proposed_cells)
            )
        score, rank, scene, proposed_rows, proposed_cells = min(
            candidates, key=lambda value: (value[0], value[1])
        )
        selected.append(scene)
        cumulative_rows = proposed_rows
        cumulative_cells = proposed_cells
        trace.append(
            {
                "step": step,
                "selected_scene": scene,
                "primary_score": score,
                "fixed_tie_rank": rank,
                "cumulative_rows": cumulative_rows,
                "cumulative_meter_quartile_counts": {
                    name: int(cumulative_cells[name]) for name in cell_names
                },
            }
        )

    dev_scene_set = set(selected)
    train_scenes = tuple(scene for scene in scene_names if scene not in dev_scene_set)
    dev_scenes = tuple(sorted(dev_scene_set))
    train_rows = tuple(row for row in values if _scene_name(row) not in dev_scene_set)
    dev_rows = tuple(row for row in values if _scene_name(row) in dev_scene_set)
    _require(
        len(train_rows) + len(dev_rows) == len(values),
        "A13 split did not assign every source row",
    )
    _require(
        not (set(train_scenes) & set(dev_scenes)),
        "A13 correction train/dev scene overlap is non-empty",
    )
    return {
        "protocol": PROTOCOL,
        "seed": SPLIT_SEED,
        "selection_method": SPLIT_SELECTION_METHOD,
        "selection_inputs": list(SPLIT_SELECTION_INPUTS),
        "selection_excluded_inputs": list(SPLIT_SELECTION_EXCLUDED_INPUTS),
        "tie_order": list(tie_order),
        "selection_trace": trace,
        "train_scenes": list(train_scenes),
        "dev_scenes": list(dev_scenes),
        "scene_overlap_count": 0,
        "source": _partition_distribution(values).as_dict(),
        "train": _partition_distribution(train_rows).as_dict(),
        "dev": _partition_distribution(dev_rows).as_dict(),
        "q0_has_seen_dev_scenes": True,
        "system_heldout_interpretation": False,
        "development_interpretation": DEVELOPMENT_INTERPRETATION,
    }


# These physical rosters/counts are filled from the single predeclared split
# execution over the 7,939-row A11-train source.  They are ordinary metadata
# records, not hashes, result-dependent baselines, contracts, or gates.
CORRECTION_DEV_SCENES: Final[tuple[str, ...]] = (
    "blue_photo_studio_4k.hdr",
    "goegap_road_4k.hdr",
    "harvest_4k.hdr",
    "lilienstein_4k.hdr",
    "modern_buildings_2_4k.hdr",
    "near_the_river_02_4k.hdr",
    "ouchy_pier_4k.hdr",
    "qwantani_patio_4k.hdr",
    "rogland_overcast_4k.hdr",
    "rosendal_plains_1_4k.hdr",
    "symmetrical_garden_02_4k.hdr",
    "zwartkops_curve_sunset_4k.hdr",
)
CORRECTION_TRAIN_SCENES: Final[tuple[str, ...]] = tuple(
    scene for scene in CORE_TRAIN_SCENES if scene not in set(CORRECTION_DEV_SCENES)
)
CORRECTION_DEV_SAMPLES: Final[int] = 1_323
CORRECTION_TRAIN_SAMPLES: Final[int] = 6_616
RECORDED_DISTRIBUTIONS: Final[dict[str, Mapping[str, Any]]] = {
    "source": {
        "samples": 7_939,
        "scenes": 72,
        "target_mean": 0.5038849419428205,
        "target_population_std": 0.28751220363281715,
        "target_min": 2.6649185874694226e-05,
        "target_max": 0.9999425817312968,
        "quartile_counts": {"q0": 1944, "q1": 2001, "q2": 1980, "q3": 2014},
        "meter_counts": {
            "bar": 1615,
            "oil": 1535,
            "pressure": 1629,
            "sf6gas": 1553,
            "temperature": 1607,
        },
        "meter_quartile_counts": {
            "bar::q0": 372, "bar::q1": 433, "bar::q2": 407, "bar::q3": 403,
            "oil::q0": 384, "oil::q1": 379, "oil::q2": 380, "oil::q3": 392,
            "pressure::q0": 414, "pressure::q1": 418,
            "pressure::q2": 395, "pressure::q3": 402,
            "sf6gas::q0": 370, "sf6gas::q1": 393,
            "sf6gas::q2": 386, "sf6gas::q3": 404,
            "temperature::q0": 404, "temperature::q1": 378,
            "temperature::q2": 412, "temperature::q3": 413,
        },
    },
    "train": {
        "samples": 6_616,
        "scenes": 60,
        "target_mean": 0.5036070086268154,
        "target_population_std": 0.2872204670070331,
        "target_min": 2.6649185874694226e-05,
        "target_max": 0.9999425817312968,
        "quartile_counts": {"q0": 1621, "q1": 1664, "q2": 1650, "q3": 1681},
        "meter_counts": {
            "bar": 1344,
            "oil": 1278,
            "pressure": 1354,
            "sf6gas": 1296,
            "temperature": 1344,
        },
        "meter_quartile_counts": {
            "bar::q0": 306, "bar::q1": 358, "bar::q2": 343, "bar::q3": 337,
            "oil::q0": 321, "oil::q1": 314, "oil::q2": 316, "oil::q3": 327,
            "pressure::q0": 346, "pressure::q1": 345,
            "pressure::q2": 328, "pressure::q3": 335,
            "sf6gas::q0": 308, "sf6gas::q1": 329,
            "sf6gas::q2": 323, "sf6gas::q3": 336,
            "temperature::q0": 340, "temperature::q1": 318,
            "temperature::q2": 340, "temperature::q3": 346,
        },
    },
    "dev": {
        "samples": 1_323,
        "scenes": 12,
        "target_mean": 0.5052748186009423,
        "target_population_std": 0.28896267712815765,
        "target_min": 0.00015683347909899677,
        "target_max": 0.9998130221480218,
        "quartile_counts": {"q0": 323, "q1": 337, "q2": 330, "q3": 333},
        "meter_counts": {
            "bar": 271,
            "oil": 257,
            "pressure": 275,
            "sf6gas": 257,
            "temperature": 263,
        },
        "meter_quartile_counts": {
            "bar::q0": 66, "bar::q1": 75, "bar::q2": 64, "bar::q3": 66,
            "oil::q0": 63, "oil::q1": 65, "oil::q2": 64, "oil::q3": 65,
            "pressure::q0": 68, "pressure::q1": 73,
            "pressure::q2": 67, "pressure::q3": 67,
            "sf6gas::q0": 62, "sf6gas::q1": 64,
            "sf6gas::q2": 63, "sf6gas::q3": 68,
            "temperature::q0": 64, "temperature::q1": 60,
            "temperature::q2": 72, "temperature::q3": 67,
        },
    },
}


def validate_recorded_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the fixed split and compare its ordinary roster/statistics."""

    result = select_correction_dev_scenes(rows, require_complete_source=True)
    _require(
        tuple(result["dev_scenes"]) == CORRECTION_DEV_SCENES,
        "recomputed correction-dev scene roster differs",
    )
    _require(
        set(result["train_scenes"]) == set(CORRECTION_TRAIN_SCENES),
        "recomputed correction-train scene roster differs",
    )
    for partition in ("source", "train", "dev"):
        observed = result[partition]
        recorded = RECORDED_DISTRIBUTIONS[partition]
        _require(
            set(observed) == set(recorded),
            f"recorded {partition} distribution fields differ",
        )
        for field, expected in recorded.items():
            value = observed[field]
            if isinstance(expected, float):
                _require(
                    math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-12),
                    f"recorded {partition} {field} differs",
                )
            else:
                _require(value == expected, f"recorded {partition} {field} differs")
    return result


_require(len(CORRECTION_DEV_SCENES) == CORRECTION_DEV_SCENE_COUNT, "dev roster width")
_require(
    len(CORRECTION_TRAIN_SCENES) == CORRECTION_TRAIN_SCENE_COUNT,
    "train roster width",
)
_require(
    not (set(CORRECTION_TRAIN_SCENES) & set(CORRECTION_DEV_SCENES)),
    "train/dev scene roster overlap",
)
_require(
    set(CORRECTION_TRAIN_SCENES) | set(CORRECTION_DEV_SCENES)
    == set(CORE_TRAIN_SCENES),
    "train/dev roster union differs from A11 train",
)
_require(
    CORRECTION_TRAIN_SAMPLES + CORRECTION_DEV_SAMPLES == SOURCE_SAMPLES,
    "train/dev sample counts do not sum to source",
)


__all__ = [
    "A13CorrectionProtocolError",
    "CORRECTION_DEV_SAMPLES",
    "CORRECTION_DEV_SCENES",
    "CORRECTION_DEV_SCENE_COUNT",
    "CORRECTION_TRAIN_SAMPLES",
    "CORRECTION_TRAIN_SCENES",
    "CORRECTION_TRAIN_SCENE_COUNT",
    "DESCRIPTIVE_REFERENCE",
    "DEVELOPMENT_INTERPRETATION",
    "EVALUATION_CONDITIONS",
    "EVALUATION_PIXEL_EPOCH",
    "EVALUATION_PIXEL_TOTAL_EPOCHS",
    "PROTOCOL",
    "PROJECTIVE_CONDITIONS",
    "RECORDED_DISTRIBUTIONS",
    "ROW_BALANCE_WEIGHT",
    "SOURCE_SAMPLES",
    "SOURCE_SCENES",
    "SPLIT_SEED",
    "SPLIT_SELECTION_EXCLUDED_INPUTS",
    "SPLIT_SELECTION_INPUTS",
    "SPLIT_SELECTION_METHOD",
    "select_correction_dev_scenes",
    "validate_recorded_split",
]
