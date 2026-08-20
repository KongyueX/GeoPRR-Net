"""Physical Core/Fold-B data boundary for A10 PCCOT.

The training loader accepts only the previously materialized 9,825-row Core
manifest.  The confirmation loader accepts only the future 1,508-row Fold-B
manifest with its exact 14-scene roster.  There is deliberately no Fold-A,
formal-holdout, or field-photo path in this module.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from experiments.a8_internal_crossfold_protocol import A8_FOLD_B_SCENE_STEMS
from experiments.a10_pccot_protocol import (
    CORE_SAMPLES,
    CORE_SCENES,
    FOLD_B_SAMPLES,
    FOLD_B_SCENES,
)
from experiments.resnet18_direct_progress import DirectSample, load_syncg_samples
from experiments.sgca_syncg_internal_pilot import INTERNAL_DEV_SCENE_STEMS
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
)


PROTOCOL: Final[str] = "a10_physical_core_train_fold_b_confirmation_v1"
DEFAULT_CORE_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a8_physical_manifests_v1/"
    "syncg_a8_core_9825.jsonl"
)
DEFAULT_FOLD_B_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a10_physical_manifests_v1/"
    "syncg_a10_fold_b_1508.jsonl"
)


class A10DataError(ValueError):
    """A supplied physical A10 roster is absent, malformed, or out of scope."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A10DataError(message)


def _validate_samples(
    samples: Sequence[DirectSample],
    *,
    expected_samples: int,
    expected_scenes: int,
    required_scenes: set[str] | None,
    excluded_scenes: set[str],
    label: str,
) -> tuple[DirectSample, ...]:
    values = tuple(samples)
    sample_ids = tuple(sample.sample_id for sample in values)
    scenes = {sample.scene_stem for sample in values}
    _require(len(values) == expected_samples, f"{label} sample count differs")
    _require(len(scenes) == expected_scenes, f"{label} scene count differs")
    _require(
        len(set(sample_ids)) == len(sample_ids) and all(sample_ids),
        f"{label} sample IDs are empty or duplicated",
    )
    _require(
        required_scenes is None or scenes == required_scenes,
        f"{label} scene roster differs",
    )
    _require(scenes.isdisjoint(excluded_scenes), f"{label} contains excluded scene")
    _require(
        all(
            math.isfinite(float(sample.normalized_target))
            and 0.0 <= float(sample.normalized_target) <= 1.0
            for sample in values
        ),
        f"{label} contains an invalid normalized target",
    )
    _require(
        all(
            9 <= len(sample.protected_points_xy) <= 38
            and all(
                len(point) == 2
                and math.isfinite(float(point[0]))
                and math.isfinite(float(point[1]))
                for point in sample.protected_points_xy
            )
            for sample in values
        ),
        f"{label} contains invalid ordered physical geometry",
    )
    return values


def validate_core_samples(samples: Sequence[DirectSample]) -> tuple[DirectSample, ...]:
    """Validate the fixed A10 training roster without opening another partition."""

    return _validate_samples(
        samples,
        expected_samples=CORE_SAMPLES,
        expected_scenes=CORE_SCENES,
        required_scenes=None,
        excluded_scenes=(
            set(A8_FOLD_B_SCENE_STEMS) | set(INTERNAL_DEV_SCENE_STEMS)
        ),
        label="A10 physical Core manifest",
    )


def validate_fold_b_samples(
    samples: Sequence[DirectSample],
) -> tuple[DirectSample, ...]:
    """Validate the sole future confirmation roster."""

    return _validate_samples(
        samples,
        expected_samples=FOLD_B_SAMPLES,
        expected_scenes=FOLD_B_SCENES,
        required_scenes=set(A8_FOLD_B_SCENE_STEMS),
        excluded_scenes=set(INTERNAL_DEV_SCENE_STEMS),
        label="A10 physical Fold-B manifest",
    )


def load_core_manifest(path: Path) -> tuple[DirectSample, ...]:
    return validate_core_samples(load_syncg_samples(Path(path).resolve()))


def load_fold_b_manifest(path: Path) -> tuple[DirectSample, ...]:
    return validate_fold_b_samples(load_syncg_samples(Path(path).resolve()))


def build_core_training_dataset(
    samples: Sequence[DirectSample],
    *,
    seed: int,
    total_epochs: int,
) -> SyncGSupportGeometryMultiViewDataset:
    """Build the shared-pixel dataset used by both independent systems."""

    values = validate_core_samples(samples)
    return SyncGSupportGeometryMultiViewDataset(
        values,
        training=True,
        seed=int(seed),
        total_epochs=int(total_epochs),
    )


def build_fold_b_evaluation_dataset(
    samples: Sequence[DirectSample],
    *,
    seed: int,
    total_epochs: int,
    condition: str,
) -> SyncGSupportGeometryMultiViewDataset:
    values = validate_fold_b_samples(samples)
    return SyncGSupportGeometryMultiViewDataset(
        values,
        training=False,
        seed=int(seed),
        total_epochs=int(total_epochs),
        condition=str(condition),
    )


def roster_metadata(samples: Sequence[DirectSample]) -> dict[str, Any]:
    values = tuple(samples)
    _require(bool(values), "A10 roster metadata is empty")
    return {
        "samples": len(values),
        "scenes": sorted({sample.scene_stem for sample in values}),
        "sample_ids": [sample.sample_id for sample in values],
    }


__all__ = [
    "A10DataError",
    "DEFAULT_CORE_MANIFEST",
    "DEFAULT_FOLD_B_MANIFEST",
    "PROTOCOL",
    "build_core_training_dataset",
    "build_fold_b_evaluation_dataset",
    "load_core_manifest",
    "load_fold_b_manifest",
    "roster_metadata",
    "validate_core_samples",
    "validate_fold_b_samples",
]
