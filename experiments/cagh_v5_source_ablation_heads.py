"""Structural registry for the frozen unified CAGH-V5 source ablations.

The previous version of this module compared only standalone ScaleMark heads.
Those heads did not ablate the deployable image-to-progress model and therefore
cannot support the V5 mechanism claims.  The formal runner now constructs every
arm through :class:`CAGHV5UnifiedModel`; this small registry is retained as the
single audited description of which internal evidence paths each arm enables.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.cagh_v5_unified_model import (
    ARM_NAMES,
    FULL_UNIFIED,
    NO_MASK_GEOMETRY,
    NO_PEPD_DIRECTION,
    NO_REFERENCE_SOLVER,
    NO_ROBUSTNESS_AUGMENTATION,
    CAGHV5UnifiedModel,
)


@dataclass(frozen=True)
class UnifiedArmContract:
    name: str
    uses_mask_geometry: bool
    uses_pepd_direction: bool
    uses_reference_solver: bool
    uses_robustness_augmentation: bool
    replacement_control: str | None = None


ARM_CONTRACTS = {
    arm: UnifiedArmContract(
        name=arm,
        uses_mask_geometry=CAGHV5UnifiedModel.arm_uses_mask_geometry(arm),
        uses_pepd_direction=CAGHV5UnifiedModel.arm_uses_pepd(arm),
        uses_reference_solver=CAGHV5UnifiedModel.arm_uses_reference_solver(arm),
        uses_robustness_augmentation=arm != NO_ROBUSTNESS_AUGMENTATION,
        replacement_control=(
            "image_only_direct_progress_head"
            if arm == NO_REFERENCE_SOLVER
            else None
        ),
    )
    for arm in ARM_NAMES
}


def build_model(arm: str, *, progress_bins: int = 72, dropout: float = 0.10) -> CAGHV5UnifiedModel:
    """Build one unified model and apply the arm's internal trainability mask."""

    if arm not in ARM_CONTRACTS:
        raise ValueError(f"unknown unified CAGH-V5 arm: {arm!r}")
    model = CAGHV5UnifiedModel(progress_bins=progress_bins, dropout=dropout)
    model.configure_trainable_arm(arm)
    return model


__all__ = [
    "ARM_CONTRACTS",
    "ARM_NAMES",
    "FULL_UNIFIED",
    "NO_MASK_GEOMETRY",
    "NO_PEPD_DIRECTION",
    "NO_REFERENCE_SOLVER",
    "NO_ROBUSTNESS_AUGMENTATION",
    "UnifiedArmContract",
    "build_model",
]
