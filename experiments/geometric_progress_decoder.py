"""Analytic gauge progress from pivot, pointer direction, and endpoint geometry."""
from __future__ import annotations

import math
from typing import Final

import torch


MINIMUM_RADIUS: Final[float] = 1.0e-3
MINIMUM_ARC_RADIANS: Final[float] = math.radians(5.0)


class GeometricProgressError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometricProgressError(message)


def _clock_angle_from_xy(vector_xy: torch.Tensor) -> torch.Tensor:
    """Return clock angle: zero points up and positive rotation is clockwise."""

    return torch.atan2(vector_xy[..., 0], -vector_xy[..., 1])


def decode_geometric_progress(
    pivot: torch.Tensor,
    direction_sin_cos: torch.Tensor,
    references: torch.Tensor,
    *,
    clockwise: bool,
) -> dict[str, torch.Tensor]:
    """Project pointer angle onto the directed start-to-end scale arc.

    Points outside the arc are assigned to the nearest circular endpoint.  This
    avoids the common wrap-around bug where a pointer just before the start is
    incorrectly clamped to progress one.
    """

    _require(pivot.shape[-1:] == (2,), "pivot must end in two coordinates")
    _require(
        direction_sin_cos.shape == pivot.shape,
        "pointer direction shape differs from pivot",
    )
    if references.shape == pivot.shape[:-1] + (4,):
        reference_points = references.reshape(pivot.shape[:-1] + (2, 2))
    else:
        _require(
            references.shape == pivot.shape[:-1] + (2, 2),
            "references must end in four values or two points",
        )
        reference_points = references
    _require(
        pivot.is_floating_point()
        and direction_sin_cos.is_floating_point()
        and references.is_floating_point(),
        "geometry tensors must be floating point",
    )

    start_vector = reference_points[..., 0, :] - pivot
    end_vector = reference_points[..., 1, :] - pivot
    start_radius = torch.linalg.vector_norm(start_vector, dim=-1)
    end_radius = torch.linalg.vector_norm(end_vector, dim=-1)
    direction_norm = torch.linalg.vector_norm(direction_sin_cos, dim=-1)
    safe_direction = direction_sin_cos / direction_norm.clamp_min(1.0e-8)[..., None]

    start_angle = _clock_angle_from_xy(start_vector)
    end_angle = _clock_angle_from_xy(end_vector)
    pointer_angle = torch.atan2(safe_direction[..., 0], safe_direction[..., 1])
    sign = 1.0 if clockwise else -1.0
    full_circle = 2.0 * math.pi
    arc = torch.remainder(sign * (end_angle - start_angle), full_circle)
    offset = torch.remainder(sign * (pointer_angle - start_angle), full_circle)
    safe_arc = arc.clamp_min(MINIMUM_ARC_RADIANS)
    inside = offset <= arc
    inside_progress = torch.clamp(offset / safe_arc, 0.0, 1.0)

    distance_to_start = torch.minimum(offset, full_circle - offset)
    raw_distance_to_end = torch.abs(offset - arc)
    distance_to_end = torch.minimum(
        raw_distance_to_end, full_circle - raw_distance_to_end
    )
    endpoint_progress = (distance_to_end < distance_to_start).to(pivot.dtype)
    progress = torch.where(inside, inside_progress, endpoint_progress)

    finite = (
        torch.isfinite(pivot).all(dim=-1)
        & torch.isfinite(direction_sin_cos).all(dim=-1)
        & torch.isfinite(reference_points).all(dim=(-1, -2))
    )
    valid = (
        finite
        & (start_radius > MINIMUM_RADIUS)
        & (end_radius > MINIMUM_RADIUS)
        & (direction_norm > MINIMUM_RADIUS)
        & (arc > MINIMUM_ARC_RADIANS)
        & (arc < full_circle - MINIMUM_ARC_RADIANS)
    )
    progress = torch.where(valid, progress, torch.full_like(progress, 0.5))
    return {
        "progress": progress,
        "valid": valid,
        "inside_arc": inside & valid,
        "arc_radians": arc,
        "offset_radians": offset,
    }


__all__ = [
    "GeometricProgressError",
    "decode_geometric_progress",
]
