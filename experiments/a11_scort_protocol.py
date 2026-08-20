"""Development-only protocol for A11 SCORT on the physical SyncG Core.

The split in this module was chosen before looking at any A11 output.  It is
an explicit scene split: every meter rendered in the same environment stays
on the same side.  The 17 audit scenes were selected by an ordinary,
deterministic greedy balance over meter type x normalized-target quartile and
sample count; lexicographic scene name was the tie-breaker.  No hash, model
prediction, checkpoint, Fold-B row, formal row, or field photograph is an
input to the split.

The screen values below are descriptive references.  This module deliberately
does not implement an automatic pass/fail or experiment-advancement gate.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final


PROTOCOL: Final[str] = "syncg_a11_scort_core_scene72_17_screen_v1"
CORE_SAMPLES: Final[int] = 9_825
CORE_SCENES: Final[int] = 89
CORE_TRAIN_SAMPLES: Final[int] = 7_939
CORE_TRAIN_SCENE_COUNT: Final[int] = 72
CORE_AUDIT_SAMPLES: Final[int] = 1_886
CORE_AUDIT_SCENE_COUNT: Final[int] = 17
PROGRESS_BINS: Final[int] = 128
CORRECTION_LAYERS: Final[int] = 8

EVALUATION_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = EVALUATION_CONDITIONS[1:]

SPLIT_SELECTION_INPUTS: Final[tuple[str, ...]] = (
    "metadata.scene_name",
    "meter_id",
    "normalized_ground_truth_progress_quartile",
    "sample_count",
)
SPLIT_SELECTION_METHOD: Final[str] = (
    "For greedy step k=1..17, form each candidate cumulative 20-cell vector "
    "over meter_id x floor(clamp(progress,0,1)*4). Compare it with "
    "full_Core_cell_count*k/89 using the mean squared deviation normalized "
    "by one-scene expected cell counts. Add 0.25 times the squared total-row "
    "deviation normalized by 9825/89. Select the minimum score and break an "
    "exact tie by lexicographic scene_name. A11 outputs are never inputs."
)
SPLIT_SELECTION_EXCLUDED_INPUTS: Final[tuple[str, ...]] = (
    "A11_outputs",
    "A11_checkpoints",
    "Fold-A",
    "Fold-B",
    "formal",
    "field",
)
DEVELOPMENT_DATA_SCOPE: Final[str] = (
    "physical_SyncG_Core_9825_rows_89_scenes_only;_FoldB_formal_and_field_"
    "are_not_development_inputs"
)

# Frozen explicit scene roster.  The extensions are part of SyncG's recorded
# scene_name and intentionally distinguish HDR from EXR environments.
CORE_AUDIT_SCENES: Final[tuple[str, ...]] = (
    "mud_road_puresky_4k.hdr",
    "buikslotermeerplein_4k.hdr",
    "zwartkops_curve_morning_4k.hdr",
    "modern_bathroom_4k.hdr",
    "hangar_interior_4k.hdr",
    "belfast_sunset_puresky_4k.hdr",
    "belfast_farmhouse_4k.hdr",
    "zwartkops_start_afternoon_4k.hdr",
    "golden_bay_4k.hdr",
    "zwartkops_straight_sunset_4k.hdr",
    "rogland_clear_night_4k.hdr",
    "qwantani_dawn_4k.hdr",
    "small_harbour_morning_4k.hdr",
    "steinbach_field_4k.hdr",
    "warm_restaurant_4k.hdr",
    "victoria_sunset_4k.hdr",
    "warm_restaurant_night_4k.hdr",
)

CORE_TRAIN_SCENES: Final[tuple[str, ...]] = (
    "DayEnvironmentHDRI012_4K-HDR.exr",
    "DayEnvironmentHDRI043_4K-HDR.exr",
    "alps_field_4k.hdr",
    "autumn_field_puresky_4k.hdr",
    "belfast_sunset_4k.hdr",
    "blue_photo_studio_4k.hdr",
    "boma_4k.hdr",
    "brown_photostudio_01_4k.hdr",
    "brown_photostudio_02_4k.hdr",
    "cannon_4k.hdr",
    "clarens_night_02_4k.hdr",
    "dikhololo_night_4k.hdr",
    "distribution_board_4k.hdr",
    "drackenstein_quarry_4k.hdr",
    "dry_orchard_meadow_4k.hdr",
    "evening_field_4k.hdr",
    "evening_road_01_puresky_4k.hdr",
    "factory_yard_4k.exr",
    "forgotten_miniland_4k.hdr",
    "goegap_road_4k.hdr",
    "graveyard_pathways_4k.hdr",
    "hanger_exterior_cloudy_4k.hdr",
    "harvest_4k.hdr",
    "hochsal_field_4k.hdr",
    "industrial_sunset_puresky_4k.hdr",
    "klippad_dawn_1_4k.hdr",
    "klippad_dawn_2_4k.hdr",
    "klippad_sunrise_1_4k.hdr",
    "kloofendal_43d_clear_4k.hdr",
    "kloofendal_43d_clear_puresky_4k.hdr",
    "kloppenheim_02_4k.hdr",
    "lake_pier_4k.hdr",
    "lakeside_night_4k.hdr",
    "lilienstein_4k.hdr",
    "limpopo_golf_course_4k.hdr",
    "lonely_road_afternoon_puresky_4k.hdr",
    "meadow_2_4k.hdr",
    "medieval_cafe_4k.hdr",
    "modern_buildings_2_4k.hdr",
    "near_the_river_02_4k.hdr",
    "ouchy_pier_4k.hdr",
    "overcast_soil_4k.hdr",
    "peppermint_powerplant_4k.hdr",
    "phalzer_forest_01_4k.hdr",
    "poly_haven_studio_4k.hdr",
    "pretoria_gardens_4k.hdr",
    "quarry_cloudy_4k.hdr",
    "qwantani_afternoon_4k.hdr",
    "qwantani_dusk_1_4k.hdr",
    "qwantani_late_afternoon_4k.hdr",
    "qwantani_moon_noon_4k.hdr",
    "qwantani_patio_4k.hdr",
    "qwantani_sunrise_4k.hdr",
    "qwantani_sunset_4k.hdr",
    "resting_place_4k.hdr",
    "rogland_overcast_4k.hdr",
    "rosendal_park_sunset_4k.hdr",
    "rosendal_plains_1_4k.hdr",
    "rosendal_plains_2_4k.hdr",
    "small_empty_room_3_4k.hdr",
    "snowy_forest_4k.hdr",
    "spree_bank_4k.hdr",
    "studio_garden_4k.hdr",
    "studio_small_09_4k.hdr",
    "syferfontein_0d_clear_4k.hdr",
    "symmetrical_garden_02_4k.hdr",
    "table_mountain_2_puresky_4k.hdr",
    "warm_bar_4k.hdr",
    "zwartkops_curve_afternoon_4k.hdr",
    "zwartkops_curve_sunset_4k.hdr",
    "zwartkops_start_sunset_4k.hdr",
    "zwartkops_straight_morning_4k.hdr",
)


class A11ProtocolError(ValueError):
    """A Core row or recorded A11 protocol value is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A11ProtocolError(message)


@dataclass(frozen=True, slots=True)
class TargetDistribution:
    samples: int
    scenes: int
    target_mean: float
    target_population_std: float
    target_min: float
    target_max: float
    quartile_counts: Mapping[str, int]
    meter_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


RECORDED_CORE_DISTRIBUTION: Final[dict[str, TargetDistribution]] = {
    "full": TargetDistribution(
        samples=9_825,
        scenes=89,
        target_mean=0.5027035768412109,
        target_population_std=0.2880364137796469,
        target_min=2.6649185874694226e-05,
        target_max=0.9999425817312968,
        quartile_counts={"q0": 2409, "q1": 2480, "q2": 2446, "q3": 2490},
        meter_counts={
            "bar": 1988,
            "oil": 1905,
            "pressure": 2021,
            "sf6gas": 1917,
            "temperature": 1994,
        },
    ),
    "train": TargetDistribution(
        samples=7_939,
        scenes=72,
        target_mean=0.5038849419428205,
        target_population_std=0.28751220363281715,
        target_min=2.6649185874694226e-05,
        target_max=0.9999425817312968,
        quartile_counts={"q0": 1944, "q1": 2001, "q2": 1980, "q3": 2014},
        meter_counts={
            "bar": 1615,
            "oil": 1535,
            "pressure": 1629,
            "sf6gas": 1553,
            "temperature": 1607,
        },
    ),
    "audit": TargetDistribution(
        samples=1_886,
        scenes=17,
        target_mean=0.49773069373321865,
        target_population_std=0.2901799341879617,
        target_min=0.00012030824383324479,
        target_max=0.9998439286810674,
        quartile_counts={"q0": 465, "q1": 479, "q2": 466, "q3": 476},
        meter_counts={
            "bar": 373,
            "oil": 370,
            "pressure": 392,
            "sf6gas": 364,
            "temperature": 387,
        },
    ),
}


A11_OUTPUT_CONTRACT: Final[dict[str, str]] = {
    "progress_posterior": "[B,128] final correction posterior",
    "mean": "[B] final posterior mean",
    "variance": "[B] final posterior variance",
    "standard_deviation": "[B] final posterior standard deviation",
    "raw_anchor_posterior": "[B,128] q0 posterior",
    "raw_anchor_mean": "[B] q0 mean",
    "raw_anchor_variance": "[B] q0 variance",
    "layer_posteriors": "[B,8,128] sequential correction posteriors",
    "layer_means": "[B,8] sequential correction means",
    "transport_angles": "[B,E] edge transport angles; E=497 for 128 bins",
    "layer_angle_residuals": "[B,8] mean normalized squared angle per layer",
    "correction_active": "[B,8] source-valid correction mask",
    "relation_available": "[B] projective relation availability",
    "sarn_active": "[B] requested SARN availability",
}

DETACHMENT_CONTRACT: Final[tuple[str, ...]] = (
    "the correction path consumes raw_anchor_posterior.detach()",
    "all Raw features entering relation/correction modules are detached",
    "correction inputs are detached from the Raw q0 learner",
    "all q0 references in regret and CVaR losses are detached",
    "the Raw anchor receives only its own q0 posterior read objective",
)
SARN_UNAVAILABLE_CONTRACT: Final[str] = (
    "when SARN/relation is unavailable, final and every layer posterior, mean, "
    "and variance equal detached q0 exactly and correction_active is false"
)

# These are reported as a predeclared Core-audit reference, never used by this
# module to control whether another command is allowed to execute.
CORE_SCREEN_REFERENCE: Final[dict[str, Any]] = {
    "audit_pooled_nmae_delta_full_minus_q0_maximum": -0.001,
    "audit_net_paired_win_minus_loss_minimum": 0.03,
    "audit_cvar25_delta_full_minus_q0_maximum": -0.001,
    "each_projective_condition_full_nmae_strictly_below_q0": True,
    "clean_nmae_delta_full_minus_q0_maximum": 0.0002,
    "sarn_off_exact_q0": True,
    "posterior_mass_violation_total": 0,
    "monotonic_violation_total": 0,
    "development_scope": DEVELOPMENT_DATA_SCOPE,
    "automatic_execution_or_advancement_control": False,
}

ABLATION_ROSTER: Final[tuple[str, ...]] = (
    "Full",
    "No-Regret",
    "Scalar",
    "Additive",
    "H=I",
    "SARN-off",
)
ABLATION_DEFINITIONS: Final[dict[str, str]] = {
    "Full": "eight-layer sequential conditional ordinal residual transport",
    "No-Regret": (
        "Full architecture with layer relative-regret and final relative-CVaR "
        "weights fixed to zero; q0/final read and angle terms unchanged"
    ),
    "Scalar": (
        "replace the structured edge transport field with one sample-wise "
        "scalar correction while retaining the same q0 and supervision"
    ),
    "Additive": (
        "predict all layer corrections from detached q0 and add them once, "
        "instead of sequential posterior composition"
    ),
    "H=I": "replace Raw-to-SARN homography with the identity matrix",
    "SARN-off": "set SARN unavailable; output must be exact detached q0",
}


def _scene_name(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), "Core row metadata is missing")
    scene = metadata.get("scene_name")
    _require(isinstance(scene, str) and bool(scene), "Core scene_name is missing")
    return scene


def _normalized_target(row: Mapping[str, Any]) -> float:
    try:
        start = float(row["scale_start"])
        end = float(row["scale_end"])
        value = float(row["ground_truth"])
    except (KeyError, TypeError, ValueError) as exc:
        raise A11ProtocolError("Core read target fields are malformed") from exc
    _require(
        math.isfinite(start) and math.isfinite(end) and math.isfinite(value),
        "Core read target is non-finite",
    )
    _require(end != start, "Core scale has zero range")
    progress = (value - start) / (end - start)
    _require(math.isfinite(progress), "normalized Core target is non-finite")
    return progress


def _partition_distribution(rows: Sequence[Mapping[str, Any]]) -> TargetDistribution:
    _require(bool(rows), "Core split partition is empty")
    targets = tuple(_normalized_target(row) for row in rows)
    mean = sum(targets) / len(targets)
    population_std = math.sqrt(
        sum((target - mean) ** 2 for target in targets) / len(targets)
    )
    quartiles: Counter[str] = Counter()
    meters: Counter[str] = Counter()
    for row, target in zip(rows, targets, strict=True):
        quartile = min(3, max(0, int(target * 4.0)))
        quartiles[f"q{quartile}"] += 1
        meter = row.get("meter_id")
        _require(isinstance(meter, str) and bool(meter), "Core meter_id is missing")
        meters[meter] += 1
    return TargetDistribution(
        samples=len(rows),
        scenes=len({_scene_name(row) for row in rows}),
        target_mean=mean,
        target_population_std=population_std,
        target_min=min(targets),
        target_max=max(targets),
        quartile_counts=dict(sorted(quartiles.items())),
        meter_counts=dict(sorted(meters.items())),
    )


def _compare_recorded(
    observed: TargetDistribution,
    recorded: TargetDistribution,
    *,
    partition: str,
) -> None:
    _require(
        observed.samples == recorded.samples and observed.scenes == recorded.scenes,
        f"{partition} Core sample/scene count differs from the recorded split",
    )
    _require(
        dict(observed.quartile_counts) == dict(recorded.quartile_counts),
        f"{partition} Core quartile distribution differs",
    )
    _require(
        dict(observed.meter_counts) == dict(recorded.meter_counts),
        f"{partition} Core meter distribution differs",
    )
    for field in (
        "target_mean",
        "target_population_std",
        "target_min",
        "target_max",
    ):
        _require(
            math.isclose(
                float(getattr(observed, field)),
                float(getattr(recorded, field)),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            f"{partition} Core {field} differs",
        )


def summarize_fixed_core_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_complete_core: bool = True,
) -> dict[str, Any]:
    """Assign rows by the explicit scene roster and report target balance.

    ``require_complete_core=False`` is useful for small unit fixtures.  The
    default additionally compares ordinary counts/statistics with the recorded
    9,825-row Core inventory; it is a data audit, not a model-result gate.
    """

    materialized = tuple(rows)
    _require(bool(materialized), "Core rows are empty")
    train_scene_set = set(CORE_TRAIN_SCENES)
    audit_scene_set = set(CORE_AUDIT_SCENES)
    overlap = train_scene_set & audit_scene_set
    _require(not overlap, f"fixed Core scene overlap is non-empty: {sorted(overlap)}")
    known = train_scene_set | audit_scene_set
    unknown = sorted({_scene_name(row) for row in materialized} - known)
    _require(not unknown, f"Core contains scenes outside the fixed roster: {unknown}")
    train = tuple(row for row in materialized if _scene_name(row) in train_scene_set)
    audit = tuple(row for row in materialized if _scene_name(row) in audit_scene_set)
    _require(len(train) + len(audit) == len(materialized), "Core rows were not assigned")
    full_distribution = _partition_distribution(materialized)
    train_distribution = _partition_distribution(train)
    audit_distribution = _partition_distribution(audit)
    if require_complete_core:
        observed_scenes = {_scene_name(row) for row in materialized}
        _require(observed_scenes == known, "complete Core scene roster differs")
        for name, observed in (
            ("full", full_distribution),
            ("train", train_distribution),
            ("audit", audit_distribution),
        ):
            _compare_recorded(
                observed,
                RECORDED_CORE_DISTRIBUTION[name],
                partition=name,
            )
    return {
        "protocol": PROTOCOL,
        "selection_inputs": SPLIT_SELECTION_INPUTS,
        "selection_method": SPLIT_SELECTION_METHOD,
        "selection_excluded_inputs": SPLIT_SELECTION_EXCLUDED_INPUTS,
        "train_scene_count": len({_scene_name(row) for row in train}),
        "audit_scene_count": len({_scene_name(row) for row in audit}),
        "scene_overlap_count": 0,
        "train_samples": len(train),
        "audit_samples": len(audit),
        "full": full_distribution.as_dict(),
        "train": train_distribution.as_dict(),
        "audit": audit_distribution.as_dict(),
    }


def validate_loss_coverage(
    *,
    read_valid: Sequence[bool],
    raw_anchor_read_applied: Sequence[bool],
    final_read_applied: Sequence[bool],
    correction_active: Sequence[Sequence[bool]],
    layer_regret_applied: Sequence[Sequence[bool]],
    cvar_eligible: Sequence[bool],
    angle_regularizer_applied: Sequence[Sequence[bool]],
) -> dict[str, int]:
    """Audit masks while keeping structure availability out of read coverage."""

    read = tuple(bool(value) for value in read_valid)
    raw = tuple(bool(value) for value in raw_anchor_read_applied)
    final = tuple(bool(value) for value in final_read_applied)
    active = tuple(tuple(bool(value) for value in row) for row in correction_active)
    regret = tuple(tuple(bool(value) for value in row) for row in layer_regret_applied)
    cvar = tuple(bool(value) for value in cvar_eligible)
    angle = tuple(
        tuple(bool(value) for value in row) for row in angle_regularizer_applied
    )
    _require(bool(read), "A11 loss coverage is empty")
    _require(len(raw) == len(final) == len(cvar) == len(read), "coverage lengths differ")
    _require(len(active) == len(regret) == len(angle) == len(read), "layer rows differ")
    _require(
        all(len(row) == CORRECTION_LAYERS for row in active + regret + angle),
        "A11 layer coverage width differs",
    )
    _require(raw == read, "q0 read loss does not cover every read-valid row")
    _require(final == read, "final read loss does not cover every read-valid row")
    expected_layers = tuple(
        tuple(valid and layer_active for layer_active in row)
        for valid, row in zip(read, active, strict=True)
    )
    _require(regret == expected_layers, "layer regret mask differs from read & active")
    _require(angle == expected_layers, "angle mask differs from read & active")
    expected_cvar = tuple(
        valid and any(row) for valid, row in zip(read, active, strict=True)
    )
    _require(cvar == expected_cvar, "CVaR eligibility differs from read & any-active")
    return {
        "rows": len(read),
        "read_valid_rows": sum(read),
        "raw_anchor_read_rows": sum(raw),
        "final_read_rows": sum(final),
        "layer_regret_cells": sum(sum(row) for row in regret),
        "cvar_eligible_rows": sum(cvar),
        "angle_regularizer_cells": sum(sum(row) for row in angle),
    }


_require(len(CORE_TRAIN_SCENES) == CORE_TRAIN_SCENE_COUNT, "train scene roster width")
_require(len(CORE_AUDIT_SCENES) == CORE_AUDIT_SCENE_COUNT, "audit scene roster width")
_require(
    not (set(CORE_TRAIN_SCENES) & set(CORE_AUDIT_SCENES)),
    "fixed train/audit scenes overlap",
)
_require(
    len(set(CORE_TRAIN_SCENES) | set(CORE_AUDIT_SCENES)) == CORE_SCENES,
    "fixed Core scene union is not 89",
)


__all__ = [
    "A11_OUTPUT_CONTRACT",
    "A11ProtocolError",
    "ABLATION_DEFINITIONS",
    "ABLATION_ROSTER",
    "CORE_AUDIT_SAMPLES",
    "CORE_AUDIT_SCENES",
    "CORE_AUDIT_SCENE_COUNT",
    "CORE_SAMPLES",
    "CORE_SCENES",
    "CORE_SCREEN_REFERENCE",
    "CORE_TRAIN_SAMPLES",
    "CORE_TRAIN_SCENES",
    "CORE_TRAIN_SCENE_COUNT",
    "CORRECTION_LAYERS",
    "DETACHMENT_CONTRACT",
    "DEVELOPMENT_DATA_SCOPE",
    "EVALUATION_CONDITIONS",
    "PROGRESS_BINS",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "RECORDED_CORE_DISTRIBUTION",
    "SARN_UNAVAILABLE_CONTRACT",
    "SPLIT_SELECTION_EXCLUDED_INPUTS",
    "SPLIT_SELECTION_INPUTS",
    "SPLIT_SELECTION_METHOD",
    "TargetDistribution",
    "summarize_fixed_core_split",
    "validate_loss_coverage",
]
