"""Fixed A8 scene folds drawn only from the former 117-scene internal train.

The caller must first obtain the existing 12,866-sample internal-train roster.
This module accepts no manifest, checkpoint, formal-holdout sample, or field
path.  Formal-holdout *scene identities* may be supplied solely to audit scene
disjointness.

Reusing the historical A3 checkpoint with these folds is only a residual-block
mechanism screen: that checkpoint was already trained on all 117 source scenes.
A scene-held-out system comparison must retrain a matched A3 separately for
each held-out fold and exclude that fold from every supervised stage.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Final

from experiments.resnet18_direct_progress import DirectSample
from experiments.sgca_syncg_internal_pilot import INTERNAL_DEV_SCENE_STEMS
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    DEV_CONDITIONS,
)


PROTOCOL: Final[str] = "a8_syncg_old_internal_train_two_scene_folds_v1"
SOURCE_SAMPLES: Final[int] = 12_866
SOURCE_SCENES: Final[int] = 117
FOLD_SCENES: Final[int] = 14

# Fixed before any A8 model result was inspected.  Selection used only scene
# identities, sample counts, and normalized-target decile balance.
A8_FOLD_A_SCENE_STEMS: Final[tuple[str, ...]] = (
    "autumn_field_4k",
    "clarens_midday_4k",
    "drachenfels_cellar_4k",
    "klippad_sunrise_2_4k",
    "kloofendal_48d_partly_cloudy_puresky_4k",
    "kloppenheim_02_puresky_4k",
    "lakeside_dawn_4k",
    "lakeside_sunrise_4k",
    "leibstadt_4k",
    "overcast_soil_puresky_4k",
    "qwantani_dusk_2_4k",
    "rural_asphalt_road_4k",
    "small_harbour_sunset_4k",
    "zwartkops_pit_4k",
)
A8_FOLD_B_SCENE_STEMS: Final[tuple[str, ...]] = (
    "billiard_hall_4k",
    "citrus_orchard_4k",
    "illovo_beach_balcony_4k",
    "kloofendal_overcast_puresky_4k",
    "kloppenheim_06_4k",
    "kloppenheim_06_puresky_4k",
    "little_paris_eiffel_tower_4k",
    "lonely_road_afternoon_4k",
    "rogland_sunset_4k",
    "small_empty_room_1_4k",
    "studio_small_08_4k",
    "wide_street_01_4k",
    "wildflower_field_4k",
    "zwartkops_straight_afternoon_4k",
)

# Keep evaluation pixels independent of whether training stops at 5 or 15
# epochs.  These values reproduce the established epoch-4 internal-dev pixels.
A8_DEV_DATASET_SEED_OFFSET: Final[int] = 31_337
A8_DEV_PIXEL_TOTAL_EPOCHS: Final[int] = 5
A8_DEV_PIXEL_EPOCH: Final[int] = 4
A8_DEV_LOADER_SEED_OFFSET: Final[int] = 10_000
A8_DEV_CONDITION_SEED_STRIDE: Final[int] = 1_009
TARGET_QUANTILE_PROBABILITIES: Final[tuple[float, ...]] = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
)


class A8FoldProtocolError(ValueError):
    """The supplied old-internal-train roster cannot realize the A8 folds."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A8FoldProtocolError(message)


@dataclass(frozen=True, slots=True)
class A8InternalFolds:
    """Three-way partition plus the two true cross-fold training rosters."""

    core_train: tuple[DirectSample, ...]
    fold_a: tuple[DirectSample, ...]
    fold_b: tuple[DirectSample, ...]
    core_scenes: tuple[str, ...]

    def validation(self, fold: str) -> tuple[DirectSample, ...]:
        name = str(fold).casefold()
        if name in {"a", "fold_a", "fold-a"}:
            return self.fold_a
        if name in {"b", "fold_b", "fold-b"}:
            return self.fold_b
        raise A8FoldProtocolError(f"unknown A8 validation fold: {fold}")

    def training_for_validation(self, fold: str) -> tuple[DirectSample, ...]:
        """Return all 117-scene source samples except the requested fold."""

        name = str(fold).casefold()
        if name in {"a", "fold_a", "fold-a"}:
            return self.core_train + self.fold_b
        if name in {"b", "fold_b", "fold-b"}:
            return self.core_train + self.fold_a
        raise A8FoldProtocolError(f"unknown A8 validation fold: {fold}")


@dataclass(frozen=True, slots=True)
class A8ConditionEvaluationSpec:
    condition: str
    dataset_seed: int
    dataset_total_epochs: int
    transform_epoch: int
    loader_seed: int


def build_a8_internal_folds(
    old_internal_train: Sequence[DirectSample],
    *,
    formal_holdout_scene_stems: Collection[str],
    old_internal_dev_scene_stems: Collection[str] = INTERNAL_DEV_SCENE_STEMS,
) -> A8InternalFolds:
    """Partition the old internal train without accepting protected data rows."""

    values = tuple(old_internal_train)
    _require(len(values) == SOURCE_SAMPLES, "A8 source must contain 12866 samples")
    sample_ids = tuple(sample.sample_id for sample in values)
    _require(
        len(set(sample_ids)) == len(sample_ids) and all(sample_ids),
        "A8 source sample IDs are empty or duplicated",
    )
    _require(
        all(
            math.isfinite(float(sample.normalized_target))
            and 0.0 <= float(sample.normalized_target) <= 1.0
            for sample in values
        ),
        "A8 source contains an invalid normalized target",
    )

    source_scenes = {sample.scene_stem for sample in values}
    fold_a_scenes = set(A8_FOLD_A_SCENE_STEMS)
    fold_b_scenes = set(A8_FOLD_B_SCENE_STEMS)
    old_dev_scenes = {str(scene) for scene in old_internal_dev_scene_stems}
    formal_scenes = {str(scene) for scene in formal_holdout_scene_stems}
    _require(len(source_scenes) == SOURCE_SCENES, "A8 source must contain 117 scenes")
    _require(
        len(fold_a_scenes) == len(fold_b_scenes) == FOLD_SCENES
        and fold_a_scenes.isdisjoint(fold_b_scenes),
        "A8 fixed folds are not two disjoint 14-scene sets",
    )
    _require(
        fold_a_scenes | fold_b_scenes <= source_scenes,
        "A8 fixed fold scene is absent from the old internal train",
    )
    _require(
        source_scenes.isdisjoint(old_dev_scenes),
        "A8 source overlaps the historical internal dev scenes",
    )
    _require(
        source_scenes.isdisjoint(formal_scenes),
        "A8 source overlaps formal holdout scene identities",
    )

    core_scene_set = source_scenes - fold_a_scenes - fold_b_scenes
    core = tuple(sample for sample in values if sample.scene_stem in core_scene_set)
    fold_a = tuple(sample for sample in values if sample.scene_stem in fold_a_scenes)
    fold_b = tuple(sample for sample in values if sample.scene_stem in fold_b_scenes)
    _require(bool(core) and bool(fold_a) and bool(fold_b), "A8 partition is empty")
    _require(
        len(core) + len(fold_a) + len(fold_b) == len(values),
        "A8 folds do not partition the old internal train",
    )
    return A8InternalFolds(
        core_train=core,
        fold_a=fold_a,
        fold_b=fold_b,
        core_scenes=tuple(sorted(core_scene_set)),
    )


def condition_evaluation_specs(seed: int) -> tuple[A8ConditionEvaluationSpec, ...]:
    """Return one fixed-pixel specification for every evaluation condition."""

    base_seed = int(seed)
    return tuple(
        A8ConditionEvaluationSpec(
            condition=condition,
            dataset_seed=base_seed + A8_DEV_DATASET_SEED_OFFSET,
            dataset_total_epochs=A8_DEV_PIXEL_TOTAL_EPOCHS,
            transform_epoch=A8_DEV_PIXEL_EPOCH,
            loader_seed=(
                base_seed
                + A8_DEV_LOADER_SEED_OFFSET
                + A8_DEV_PIXEL_EPOCH
                + index * A8_DEV_CONDITION_SEED_STRIDE
            ),
        )
        for index, condition in enumerate(DEV_CONDITIONS)
    )


def condition_target_rosters(
    samples: Sequence[DirectSample],
) -> Mapping[str, tuple[tuple[str, float], ...]]:
    """Replicate one sample/target roster across all fixed conditions."""

    rows = tuple(
        (sample.sample_id, float(sample.normalized_target)) for sample in samples
    )
    _require(bool(rows), "A8 condition roster is empty")
    return {condition: rows for condition in DEV_CONDITIONS}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def target_distribution_summary(samples: Sequence[DirectSample]) -> dict[str, Any]:
    """Summarize labels without reading images or creating model predictions."""

    values = tuple(float(sample.normalized_target) for sample in samples)
    _require(bool(values), "A8 target summary is empty")
    _require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values),
        "A8 target summary contains an invalid value",
    )
    bins = [0] * 10
    for value in values:
        bins[min(int(value * 10.0), 9)] += 1
    return {
        "samples": len(values),
        "scenes": len({sample.scene_stem for sample in samples}),
        "mean": fmean(values),
        "standard_deviation": pstdev(values),
        "quantiles": {
            str(probability): _quantile(values, probability)
            for probability in TARGET_QUANTILE_PROBABILITIES
        },
        "target_decile_probabilities": [count / len(values) for count in bins],
    }


__all__ = [
    "A8ConditionEvaluationSpec",
    "A8FoldProtocolError",
    "A8InternalFolds",
    "A8_DEV_PIXEL_EPOCH",
    "A8_DEV_PIXEL_TOTAL_EPOCHS",
    "A8_FOLD_A_SCENE_STEMS",
    "A8_FOLD_B_SCENE_STEMS",
    "PROTOCOL",
    "build_a8_internal_folds",
    "condition_evaluation_specs",
    "condition_target_rosters",
    "target_distribution_summary",
]
