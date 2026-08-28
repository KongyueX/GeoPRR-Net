"""Paper-facing GeoPRR-Net model identity and checkpoint loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import torch

from experiments.unified_pointer_reader import (
    ARCHITECTURE,
    CANDIDATE_NAMES,
    PROTOCOL as CHECKPOINT_PROTOCOL,
    PUBLICATION_NAME,
    UnifiedPointerReader,
    load_unified_pointer_reader_checkpoint,
)


PUBLICATION_FULL_NAME: Final[str] = (
    "GeoPRR-Net: Geometry-Aware Polar-Relational Routing for Robust Analog "
    "Gauge Reading"
)
PUBLICATION_PROTOCOL: Final[str] = "geoprr_net_inference_v1"


def publication_model_identity() -> dict[str, Any]:
    """Return the stable paper identity without opening a checkpoint."""

    return {
        "machine_key": "geoprr_net",
        "display_name": PUBLICATION_NAME,
        "full_name": PUBLICATION_FULL_NAME,
        "inference_protocol": PUBLICATION_PROTOCOL,
        "checkpoint_protocol": CHECKPOINT_PROTOCOL,
        "checkpoint_architecture": ARCHITECTURE,
        "candidate_names": list(CANDIDATE_NAMES),
    }


def load_geoprr_net(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[UnifiedPointerReader, dict[str, Any]]:
    """Load one publication checkpoint and attach the public model identity."""

    model, metadata = load_unified_pointer_reader_checkpoint(
        Path(checkpoint_path), device=device
    )
    public_metadata = dict(metadata)
    public_metadata.update(
        {
            "protocol": PUBLICATION_PROTOCOL,
            "checkpoint_protocol": metadata["protocol"],
            "publication_model": publication_model_identity(),
        }
    )
    return model, public_metadata


__all__ = [
    "ARCHITECTURE",
    "CANDIDATE_NAMES",
    "CHECKPOINT_PROTOCOL",
    "PUBLICATION_FULL_NAME",
    "PUBLICATION_NAME",
    "PUBLICATION_PROTOCOL",
    "UnifiedPointerReader",
    "load_geoprr_net",
    "publication_model_identity",
]
