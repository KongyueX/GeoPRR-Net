"""Stable paper-facing API for R²MT-Net.

The checkpoint protocol predates the publication name and therefore retains
internal ``remst_*`` identifiers.  This module keeps those identifiers behind
the loader boundary while exposing one R²MT-Net identity to users.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn

from experiments.r2mt_net import (
    ARCHITECTURE as CHECKPOINT_ARCHITECTURE,
    DEFAULT_PRIOR_WEIGHTS,
    PUBLICATION_FULL_NAME,
    PUBLICATION_NAME,
    R2MTNet,
    R2MTRouter,
    publication_model_identity,
)
from experiments.train_r2mt_net import (
    PROTOCOL as CHECKPOINT_PROTOCOL,
    load_r2mt_net_checkpoint,
)


PUBLICATION_PROTOCOL: Final[str] = "r2mt_net_inference_v1"
DEFAULT_ADAPTIVE_STRENGTH: Final[float] = 0.5
DEFAULT_FUSION_MODE: Final[str] = "single_exact_transport"


def _unique_parameter_count(modules: Iterable[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += int(parameter.numel())
    return total


def r2mt_parameter_counts(anchor: nn.Module, model: nn.Module) -> dict[str, int]:
    """Count the shared anchor and correction parameters exactly once."""

    anchor_parameters = _unique_parameter_count((anchor,))
    correction_parameters = _unique_parameter_count((model,))
    return {
        "shared_anchor": anchor_parameters,
        "relation_and_risk_transport": correction_parameters,
        "total_unique": _unique_parameter_count((anchor, model)),
        "additional_image_encoders": 0,
    }


def load_r2mt_net(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    adaptive_strength: float = DEFAULT_ADAPTIVE_STRENGTH,
    fusion_mode: str = DEFAULT_FUSION_MODE,
) -> tuple[nn.Module, R2MTNet, dict[str, Any]]:
    """Load the publication model with its conservative inference settings."""

    if not 0.0 <= float(adaptive_strength) <= 1.0:
        raise ValueError("adaptive_strength must lie in [0, 1]")
    if fusion_mode not in {"single_exact_transport", "posterior_mixture"}:
        raise ValueError("unsupported R²MT-Net fusion mode")

    anchor, model, checkpoint_metadata = load_r2mt_net_checkpoint(
        Path(checkpoint_path), device=device
    )
    model.adaptive_strength = float(adaptive_strength)
    model.fusion_mode = str(fusion_mode)
    metadata = dict(checkpoint_metadata)
    metadata.update(
        {
            "protocol": PUBLICATION_PROTOCOL,
            "checkpoint_protocol": checkpoint_metadata["protocol"],
            "checkpoint_architecture": checkpoint_metadata["architecture"],
            "publication_model": publication_model_identity(),
            "inference_configuration": {
                "adaptive_strength": float(adaptive_strength),
                "fusion_mode": str(fusion_mode),
            },
            "parameter_counts_public": r2mt_parameter_counts(anchor, model),
        }
    )
    return anchor, model, metadata


__all__ = [
    "CHECKPOINT_ARCHITECTURE",
    "CHECKPOINT_PROTOCOL",
    "DEFAULT_ADAPTIVE_STRENGTH",
    "DEFAULT_FUSION_MODE",
    "DEFAULT_PRIOR_WEIGHTS",
    "PUBLICATION_FULL_NAME",
    "PUBLICATION_NAME",
    "PUBLICATION_PROTOCOL",
    "R2MTNet",
    "R2MTRouter",
    "load_r2mt_net",
    "publication_model_identity",
    "r2mt_parameter_counts",
]
