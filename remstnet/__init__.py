"""Stable public interface for the ReMSTNet-v3 paper model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from .model import (
    ADAPTIVE_REMST_NET_ARCHITECTURE,
    CoordinatedReMSTNet,
    initialize_adaptive_remst_net,
    remstnet_parameter_counts,
)


ARCHITECTURE_ID: Final[str] = ADAPTIVE_REMST_NET_ARCHITECTURE
ReMSTNetV3 = CoordinatedReMSTNet


def build_remstnet_v3(**kwargs: Any) -> ReMSTNetV3:
    """Construct the final v3 architecture without loading fitted weights.

    This constructor is useful for architecture inspection and smoke tests.
    Use :func:`load_foundation_initialized_remstnet_v3` to import the frozen
    foundation from a direct-reader checkpoint before fitting the ReMST blocks.
    """

    reserved = {"use_progress_mixing", "learnable_budget_gain"}.intersection(
        kwargs
    )
    if reserved:
        names = ", ".join(sorted(reserved))
        raise TypeError(f"ReMSTNet-v3 fixes these constructor options: {names}")
    return ReMSTNetV3(
        use_progress_mixing=True,
        learnable_budget_gain=True,
        **kwargs,
    )


def load_foundation_initialized_remstnet_v3(
    direct_checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[ReMSTNetV3, dict[str, Any]]:
    """Build v3 and import the frozen direct-reader foundation checkpoint."""

    return initialize_adaptive_remst_net(
        Path(direct_checkpoint_path),
        device=device,
    )


__all__ = [
    "ARCHITECTURE_ID",
    "ReMSTNetV3",
    "build_remstnet_v3",
    "load_foundation_initialized_remstnet_v3",
    "remstnet_parameter_counts",
]
