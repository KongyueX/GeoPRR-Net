"""Frozen feature-family definitions for FADR router ablations.

The ablations in this module change only the evidence available to the routing
score.  They do not change the two candidate readings, the grouped folds, the
threshold-selection rule, or the hard-fallback policy.
"""

from __future__ import annotations

from collections.abc import Mapping

from experiments.calibrated_progress_router import (
    CALIBRATION_FEATURE_NAMES,
    FEATURE_NAMES,
)
from experiments.quality_router import RAW_QUALITY_FEATURES
from experiments.uncertainty_fusion import NATIVE_UNCERTAINTY_FEATURE_NAMES


FADR_ROUTER_FEATURE_ABLATION_PROTOCOL = (
    "syncg_strict_nested_fadr_router_feature_ablation_v1"
)

# Direct differences between the mask/vector family, the legacy geometric
# readers, and the reference-conditioned calibrated candidate.
CROSS_REPRESENTATION_DISAGREEMENT_FEATURES = (
    "base_vector_progress_abs",
    "base_vector_progress_signed",
    "weighted_vector_progress_abs",
    "geometry_v1_vector_progress_abs",
    "geometry_v2_vector_progress_abs",
    "transformer_vector_progress_abs",
    "weighted_vector_angle_abs_fraction",
    "geometry_v1_vector_angle_abs_fraction",
    "geometry_v2_vector_angle_abs_fraction",
    "transformer_vector_angle_abs_fraction",
    "raw_calibrated_progress_abs",
    "base_calibrated_progress_abs",
)

# These variables expose reference-conditioned evidence to the routing score.
# Removing them does not remove the calibrated candidate itself, so the
# resulting ablation must be described as a router-evidence ablation.
REFERENCE_CONDITIONED_ROUTING_FEATURES = (
    "range_angle_fraction",
    "branch_start_and_end",
    "branch_start_only",
    "branch_end_only",
) + CALIBRATION_FEATURE_NAMES


def _ordered_without(excluded: set[str]) -> tuple[str, ...]:
    return tuple(name for name in FEATURE_NAMES if name not in excluded)


def fadr_router_feature_sets() -> Mapping[str, tuple[str, ...]]:
    """Return the preregistered router feature sets in deterministic order."""

    feature_sets = {
        "full": tuple(FEATURE_NAMES),
        "disagreement_only": tuple(CROSS_REPRESENTATION_DISAGREEMENT_FEATURES),
        "without_mask_quality": _ordered_without(set(RAW_QUALITY_FEATURES)),
        "without_pepd_native_uncertainty": _ordered_without(
            set(NATIVE_UNCERTAINTY_FEATURE_NAMES)
        ),
        "without_reference_conditioned_router_evidence": _ordered_without(
            set(REFERENCE_CONDITIONED_ROUTING_FEATURES)
        ),
    }
    _validate_feature_sets(feature_sets)
    return feature_sets


def _validate_feature_sets(
    feature_sets: Mapping[str, tuple[str, ...]],
) -> None:
    full = tuple(FEATURE_NAMES)
    full_set = set(full)
    if len(full) != 59 or len(full_set) != len(full):
        raise RuntimeError("FADR router feature schema is not the frozen 59-vector")
    expected_variants = (
        "full",
        "disagreement_only",
        "without_mask_quality",
        "without_pepd_native_uncertainty",
        "without_reference_conditioned_router_evidence",
    )
    if tuple(feature_sets) != expected_variants:
        raise RuntimeError("FADR feature-ablation variant order drifted")
    for variant, names in feature_sets.items():
        if not names or len(names) != len(set(names)):
            raise RuntimeError(f"{variant} has empty or duplicate features")
        unknown = set(names) - full_set
        if unknown:
            raise RuntimeError(f"{variant} has unknown features: {sorted(unknown)}")
        expected_order = tuple(name for name in full if name in set(names))
        if tuple(names) != expected_order:
            raise RuntimeError(f"{variant} does not preserve the frozen schema order")


FADR_ROUTER_FEATURE_SETS = fadr_router_feature_sets()
