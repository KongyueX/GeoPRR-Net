"""A15.2 conservative fixed-scale FTEB correction.

A15.2 changes only the fixed magnitude of A15's learned centered natural-
parameter residual.  It retains the raw learned field for diagnostics and
applies one non-learned 0.25 multiplier uniformly to every path layer.
"""
from __future__ import annotations

from typing import Final

from experiments.a15_fteb import (
    A15FTEBCorrection,
    DEFAULT_MEMORY_GRID_SIZE,
    DEFAULT_PROGRESS_BINS,
)
from experiments.a11_scort import DEFAULT_RELATION_CHANNELS, DEFAULT_TOKEN_DIM


A15_2_ARCHITECTURE: Final[str] = (
    "FTEB-A15.2-Fixed-Quarter-Learned-Natural-Parameter-Residual"
)
A15_2_LEARNED_RESIDUAL_SCALE: Final[float] = 0.25


class A152FTEBCorrection(A15FTEBCorrection):
    """A15 FTEB with the learned residual fixed to one quarter strength."""

    def __init__(
        self,
        *,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = DEFAULT_MEMORY_GRID_SIZE,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        use_relation_memory: bool = True,
    ) -> None:
        super().__init__(
            relation_channels=relation_channels,
            token_dim=token_dim,
            attention_heads=attention_heads,
            decoder_layers=decoder_layers,
            memory_grid_size=memory_grid_size,
            progress_bins=progress_bins,
            learned_residual_scale=A15_2_LEARNED_RESIDUAL_SCALE,
            use_relation_memory=use_relation_memory,
        )

    def forward(self, *args: object, **kwargs: object) -> dict[str, object]:
        output = super().forward(*args, **kwargs)
        output["architecture"] = A15_2_ARCHITECTURE
        return output


__all__ = [
    "A15_2_ARCHITECTURE",
    "A15_2_LEARNED_RESIDUAL_SCALE",
    "A152FTEBCorrection",
]
