"""Projective transport and calibrated fusion for circular pointer posteriors.

This module is an experimental, v3 candidate algorithm.  It deliberately does
not change the frozen v1/v2 training or evaluation code.

The two direction heads are treated as correlated experts on a common circular
grid.  They are therefore fused with a generalized log-opinion pool, not an
independence-assuming product.  Under the v3 training contract,
``log_variance_raw`` has the explicit semantics

    log_variance_raw = log(sigma_direct ** 2),  sigma in radians,

so the direct expert precision is exactly ``tau = 1 / sigma_direct ** 2`` in
inverse radians squared.  After finite clipping and an explicit calibration
temperature, its discrete geodesic-normal log density is

    log p_direct(theta_k) =
        -0.5 * min(tau / T_direct, tau_cap)
        * d_circle(theta_k, mu_direct) ** 2
        + constant.

Both experts are evaluated on a shared 360-bin grid by default (one degree per
bin), so the legacy 72-bin expert does not quantize a sub-degree direct mean.
The fused posterior is a stable generalized product/log-opinion pool:

    p_fused(theta_k) proportional to
        p_bin(theta_k) ** w_bin * p_direct(theta_k) ** w_direct.

The default powers are ``w_bin = w_direct = 0.5`` and sum to one, reducing
double-counting when the two heads share an encoder and have correlated errors.
The classical untempered PoE ``(w_bin, w_direct) = (1, 1)`` is available only
as an explicit ablation.  Legacy checkpoints whose variance head was supervised
from a fused mean do *not* satisfy the v3 direct-variance contract and must not
be decoded as calibrated direct precision without retraining.

For a homography, posterior mass is pushed forward through the local
projective Jacobian at the predicted pivot.  Each mapped circular coordinate
is deposited into its two neighbouring bins with periodic linear splatting.
This preserves probability mass and avoids nearest-bin discontinuities.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


_TWO_PI = 2.0 * math.pi


class CircularPosteriorPrediction(NamedTuple):
    """Decoded PoE posterior and continuous circular summary.

    ``pivot_xy`` is expressed in the coordinate system of ``pivot_logits``
    (normally heatmap coordinates).  Callers must apply their known stride
    before using an input-image homography.
    """

    pivot_xy: torch.Tensor
    pivot_peak: torch.Tensor
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    mean_angle: torch.Tensor
    mean_direction: torch.Tensor
    resultant_length: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    angle_std_radians: torch.Tensor
    angle_std_degrees: torch.Tensor
    direct_precision: torch.Tensor
    effective_direct_precision: torch.Tensor
    bin_expert_power: torch.Tensor
    direct_expert_power: torch.Tensor
    valid: torch.Tensor


class CircularPosteriorTransport(NamedTuple):
    """Mass-conserving circular posterior push-forward."""

    probabilities: torch.Tensor
    transformed_pivot_xy: torch.Tensor
    mapped_bin_coordinate: torch.Tensor
    source_mass: torch.Tensor
    target_mass: torch.Tensor
    valid: torch.Tensor


class BidirectionalPosteriorConsistency(NamedTuple):
    """Forward/inverse projective posterior consistency diagnostics."""

    loss: torch.Tensor
    forward_js: torch.Tensor
    inverse_js: torch.Tensor
    valid: torch.Tensor
    valid_fraction: torch.Tensor


def _working_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _angle_centers(
    angle_bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(angle_bins) < 2:
        raise ValueError("angle_bins must be at least 2")
    return torch.arange(int(angle_bins), device=device, dtype=dtype) * (
        _TWO_PI / float(angle_bins)
    )


def circular_delta(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Signed shortest angular displacement ``first - second`` in radians."""

    return torch.atan2(torch.sin(first - second), torch.cos(first - second))


def periodic_linear_discrete_nll(
    log_probabilities: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    minimum_probability: float = 1e-12,
) -> torch.Tensor:
    """Score a circular target by linearly interpolating neighbouring masses.

    This is the single training/evaluation definition used by the v3
    protocol.  It computes ``-log((1-f) p_lower + f p_upper)`` on the periodic
    support in log space, rather than interpolating the two log probabilities.
    """

    if log_probabilities.ndim != 2 or log_probabilities.shape[1] < 2:
        raise ValueError("log_probabilities must have shape [B, K] with K >= 2")
    if target_direction.shape != (log_probabilities.shape[0], 2):
        raise ValueError("target_direction must have shape [B, 2]")
    floor = float(minimum_probability)
    if not math.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("minimum_probability must be finite and in (0, 1)")
    bins = int(log_probabilities.shape[1])
    target = target_direction.to(
        device=log_probabilities.device,
        dtype=log_probabilities.dtype,
    )
    angle = torch.remainder(torch.atan2(target[:, 1], target[:, 0]), _TWO_PI)
    coordinate = angle * (float(bins) / _TWO_PI)
    lower_unwrapped = torch.floor(coordinate)
    fraction = coordinate - lower_unwrapped
    lower = torch.remainder(lower_unwrapped.long(), bins)
    upper = torch.remainder(lower + 1, bins)
    rows = torch.arange(log_probabilities.shape[0], device=log_probabilities.device)
    lower_log_weight = torch.log1p(-fraction)
    upper_log_weight = torch.log(fraction)
    target_log_mass = torch.logsumexp(
        torch.stack(
            (
                log_probabilities[rows, lower] + lower_log_weight,
                log_probabilities[rows, upper] + upper_log_weight,
            ),
            dim=1,
        ),
        dim=1,
    )
    valid = (
        torch.isfinite(log_probabilities).all(dim=1)
        & torch.isfinite(target).all(dim=1)
        & (torch.linalg.vector_norm(target, dim=1) > 1e-12)
        & torch.isfinite(target_log_mass)
    )
    return torch.where(
        valid,
        -torch.clamp(target_log_mass, min=math.log(floor)),
        torch.full_like(target_log_mass, math.inf),
    )


def _soft_pivot(pivot_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = pivot_logits.shape
    flattened = pivot_logits[:, 0].reshape(batch, -1)
    spatial = torch.softmax(flattened, dim=1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=pivot_logits.device, dtype=spatial.dtype),
        torch.arange(width, device=pivot_logits.device, dtype=spatial.dtype),
        indexing="ij",
    )
    pivot = torch.stack(
        (
            torch.sum(spatial * xx.reshape(1, -1), dim=1),
            torch.sum(spatial * yy.reshape(1, -1), dim=1),
        ),
        dim=1,
    )
    peak = torch.sigmoid(flattened).max(dim=1).values
    return pivot, peak


def _validate_four_heads(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
) -> None:
    if pivot_logits.ndim != 4 or pivot_logits.shape[1] != 1:
        raise ValueError(f"invalid pivot heatmap shape: {tuple(pivot_logits.shape)}")
    batch = pivot_logits.shape[0]
    if direction_raw.shape != (batch, 2):
        raise ValueError(f"invalid direction shape: {tuple(direction_raw.shape)}")
    if angle_logits.ndim != 2 or angle_logits.shape[0] != batch:
        raise ValueError(f"invalid angle logits shape: {tuple(angle_logits.shape)}")
    if angle_logits.shape[1] < 8:
        raise ValueError("the circular expert must have at least 8 bins")
    if log_variance_raw.shape != (batch, 1):
        raise ValueError(f"invalid variance shape: {tuple(log_variance_raw.shape)}")
    devices = {
        pivot_logits.device,
        direction_raw.device,
        angle_logits.device,
        log_variance_raw.device,
    }
    if len(devices) != 1:
        raise ValueError("all four heads must be on the same device")


def _bounded_direct_precision(
    log_variance: torch.Tensor,
    *,
    min_direct_precision: float,
    max_direct_precision: float,
) -> torch.Tensor:
    """Convert log variance in radian squared to bounded inverse variance."""

    minimum = float(min_direct_precision)
    maximum = float(max_direct_precision)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("direct precision bounds must be finite")
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError(
            "direct precision bounds must satisfy 0 < minimum < maximum"
        )
    safe_log_variance = torch.nan_to_num(
        log_variance,
        nan=0.0,
        posinf=-math.log(minimum),
        neginf=-math.log(maximum),
    )
    log_precision = torch.clamp(
        -safe_log_variance,
        min=math.log(minimum),
        max=math.log(maximum),
    )
    return torch.exp(log_precision)


def _posterior_summary(
    probabilities: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    centers = _angle_centers(
        probabilities.shape[1],
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    cosine = torch.sum(probabilities * torch.cos(centers)[None, :], dim=1)
    sine = torch.sum(probabilities * torch.sin(centers)[None, :], dim=1)
    mean_vector = torch.stack((cosine, sine), dim=1)
    resultant = torch.linalg.vector_norm(mean_vector, dim=1)
    mean_direction = mean_vector / torch.clamp(resultant[:, None], min=1e-12)
    mean_angle = torch.atan2(mean_direction[:, 1], mean_direction[:, 0])
    entropy = -torch.sum(
        probabilities
        * torch.log(torch.clamp(probabilities, min=torch.finfo(probabilities.dtype).tiny)),
        dim=1,
    )
    normalized_entropy = entropy / math.log(float(probabilities.shape[1]))
    # This is the standard circular deviation sqrt(-2 log R).  A finite lower
    # clamp gives a finite diagnostic even for an exactly diffuse posterior.
    angle_std = torch.sqrt(
        torch.clamp(
            -2.0
            * torch.log(
                torch.clamp(
                    resultant,
                    min=max(torch.finfo(probabilities.dtype).eps, 1e-12),
                    max=1.0,
                )
            ),
            min=0.0,
        )
    )
    return (
        mean_angle,
        mean_direction,
        resultant,
        entropy,
        normalized_entropy,
        angle_std,
    )


def _periodic_linear_resample(
    values: torch.Tensor,
    *,
    target_bins: int,
) -> torch.Tensor:
    """Periodically resample circular values with first-order interpolation."""

    if values.ndim != 2:
        raise ValueError("circular values must have shape [B, K]")
    source_bins = values.shape[1]
    target = int(target_bins)
    if target < 8:
        raise ValueError("posterior_bins must be at least 8")
    if target == source_bins:
        return values
    coordinate = (
        torch.arange(target, device=values.device, dtype=values.dtype)
        * (float(source_bins) / float(target))
    )
    lower = torch.floor(coordinate).to(dtype=torch.long)
    fraction = coordinate - lower.to(dtype=values.dtype)
    upper = torch.remainder(lower + 1, source_bins)
    return (
        values[:, lower] * (1.0 - fraction)[None, :]
        + values[:, upper] * fraction[None, :]
    )


def fuse_circular_experts(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
    *,
    min_direct_precision: float = 0.05,
    direct_precision_cap: float = 400.0,
    direct_precision_temperature: float = 1.0,
    bin_expert_power: float = 0.5,
    direct_expert_power: float = 0.5,
    posterior_bins: int = 360,
) -> CircularPosteriorPrediction:
    """Fuse four v3 heads into a calibrated circular log-opinion pool.

    The 72-bin head remains an unrestricted categorical expert.  The direct
    vector head contributes a geodesic-normal circular expert whose precision
    is the temperature-scaled inverse angular variance, finitely bounded by
    ``direct_precision_temperature``.  Correlated evidence is combined as
    ``w_bin log(p_bin) + w_direct log(p_direct)``.

    Defaults use a convex log-opinion pool (powers sum to one).  Passing
    ``bin_expert_power=direct_expert_power=1`` recovers the classical PoE for
    ablation.  The categorical log density is periodically interpolated onto
    ``posterior_bins`` (360 by default) before pooling.  Invalid rows return a
    finite uniform posterior and ``valid=False``.
    """

    _validate_four_heads(
        pivot_logits,
        direction_raw,
        angle_logits,
        log_variance_raw,
    )
    dtype = _working_dtype(angle_logits)
    temperature = float(direct_precision_temperature)
    bin_power = float(bin_expert_power)
    direct_power = float(direct_expert_power)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("direct_precision_temperature must be positive and finite")
    if (
        not math.isfinite(bin_power)
        or not math.isfinite(direct_power)
        or bin_power < 0.0
        or direct_power < 0.0
        or bin_power + direct_power <= 0.0
    ):
        raise ValueError(
            "expert powers must be finite, non-negative, and not both zero"
        )
    pivot_work = pivot_logits.to(dtype=dtype)
    direction_work = direction_raw.to(dtype=dtype)
    angle_work = angle_logits.to(dtype=dtype)
    variance_work = log_variance_raw[:, 0].to(dtype=dtype)

    pivot_finite = torch.isfinite(pivot_work).reshape(
        pivot_work.shape[0], -1
    ).all(dim=1)
    direction_finite = torch.isfinite(direction_work).all(dim=1)
    angle_finite = torch.isfinite(angle_work).all(dim=1)
    variance_finite = torch.isfinite(variance_work)
    safe_pivot = torch.nan_to_num(pivot_work)
    safe_direction = torch.nan_to_num(direction_work)
    safe_angle_logits = torch.nan_to_num(angle_work)
    pivot_xy, pivot_peak = _soft_pivot(safe_pivot)

    direct_norm = torch.linalg.vector_norm(safe_direction, dim=1)
    direct_direction = safe_direction / torch.clamp(direct_norm[:, None], min=1e-12)
    direct_angle = torch.atan2(direct_direction[:, 1], direct_direction[:, 0])
    direct_precision = _bounded_direct_precision(
        variance_work,
        min_direct_precision=min_direct_precision,
        max_direct_precision=direct_precision_cap,
    )
    effective_direct_precision = torch.clamp(
        direct_precision / temperature,
        max=float(direct_precision_cap),
    )
    common_bins = int(posterior_bins)
    if common_bins < 8:
        raise ValueError("posterior_bins must be at least 8")
    centers = _angle_centers(
        common_bins,
        device=angle_work.device,
        dtype=dtype,
    )
    direct_delta = circular_delta(centers[None, :], direct_angle[:, None])
    direct_log_probability = F.log_softmax(
        -0.5 * effective_direct_precision[:, None] * direct_delta.square(),
        dim=1,
    )
    bin_log_probability = _periodic_linear_resample(
        F.log_softmax(safe_angle_logits, dim=1),
        target_bins=common_bins,
    )
    pooled_log_evidence = (
        bin_power * bin_log_probability
        + direct_power * direct_log_probability
    )
    fused_log_probability = F.log_softmax(
        pooled_log_evidence,
        dim=1,
    )
    fused_probability = torch.exp(fused_log_probability)

    direct_required = direct_power > 0.0
    bin_required = bin_power > 0.0
    preliminary_valid = (
        pivot_finite
        & ((not bin_required) | angle_finite)
        & (
            (not direct_required)
            | (direction_finite & variance_finite & (direct_norm > 1e-8))
        )
    )
    uniform = torch.full_like(
        fused_probability,
        1.0 / float(fused_probability.shape[1]),
    )
    fused_probability = torch.where(
        preliminary_valid[:, None],
        fused_probability,
        uniform,
    )
    fused_log_probability = torch.log(
        torch.clamp(fused_probability, min=torch.finfo(dtype).tiny)
    )
    (
        mean_angle,
        mean_direction,
        resultant,
        entropy,
        normalized_entropy,
        angle_std,
    ) = _posterior_summary(fused_probability)
    valid = (
        preliminary_valid
        & torch.isfinite(fused_probability).all(dim=1)
        & torch.isfinite(mean_direction).all(dim=1)
        & torch.isfinite(resultant)
        & (resultant > 1e-8)
    )
    return CircularPosteriorPrediction(
        pivot_xy=pivot_xy,
        pivot_peak=pivot_peak,
        probabilities=fused_probability,
        log_probabilities=fused_log_probability,
        mean_angle=mean_angle,
        mean_direction=mean_direction,
        resultant_length=resultant,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        angle_std_radians=angle_std,
        angle_std_degrees=angle_std * (180.0 / math.pi),
        direct_precision=direct_precision,
        effective_direct_precision=effective_direct_precision,
        bin_expert_power=torch.full_like(direct_precision, bin_power),
        direct_expert_power=torch.full_like(direct_precision, direct_power),
        valid=valid,
    )


def _validate_transport_inputs(
    probabilities: torch.Tensor,
    pivot_xy: torch.Tensor,
    homography: torch.Tensor,
) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [B, K] with K >= 2")
    batch = probabilities.shape[0]
    if pivot_xy.shape != (batch, 2):
        raise ValueError("pivot_xy must have shape [B, 2]")
    if homography.shape != (batch, 3, 3):
        raise ValueError("homography must have shape [B, 3, 3]")
    if not probabilities.is_floating_point():
        raise ValueError("probabilities must be floating point")
    if len({probabilities.device, pivot_xy.device, homography.device}) != 1:
        raise ValueError("transport inputs must be on the same device")


def _local_projective_jacobian(
    pivot_xy: torch.Tensor,
    homography: torch.Tensor,
    *,
    denominator_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projected pivot, local 2x2 Jacobian and row validity."""

    x = pivot_xy[:, 0]
    y = pivot_xy[:, 1]
    epsilon = float(denominator_epsilon)
    matrix_scale = torch.amax(torch.abs(homography), dim=(1, 2))
    finite_matrix = torch.isfinite(homography).all(dim=(1, 2))
    scale_valid = finite_matrix & (matrix_scale > torch.finfo(homography.dtype).tiny)
    safe_scale = torch.where(
        scale_valid,
        matrix_scale,
        torch.ones_like(matrix_scale),
    )
    # A homography is defined only up to a non-zero scalar.  Normalize before
    # every projective calculation so H, -H, 1e-12 H and 1e12 H have identical
    # geometry and validity decisions.
    h = homography / safe_scale[:, None, None]
    numerator_x = h[:, 0, 0] * x + h[:, 0, 1] * y + h[:, 0, 2]
    numerator_y = h[:, 1, 0] * x + h[:, 1, 1] * y + h[:, 1, 2]
    denominator = h[:, 2, 0] * x + h[:, 2, 1] * y + h[:, 2, 2]
    safe_denominator = torch.where(
        torch.abs(denominator) > epsilon,
        denominator,
        torch.where(
            denominator < 0.0,
            torch.full_like(denominator, -epsilon),
            torch.full_like(denominator, epsilon),
        ),
    )
    denominator_squared = safe_denominator.square()
    j00 = (
        h[:, 0, 0] * safe_denominator - numerator_x * h[:, 2, 0]
    ) / denominator_squared
    j01 = (
        h[:, 0, 1] * safe_denominator - numerator_x * h[:, 2, 1]
    ) / denominator_squared
    j10 = (
        h[:, 1, 0] * safe_denominator - numerator_y * h[:, 2, 0]
    ) / denominator_squared
    j11 = (
        h[:, 1, 1] * safe_denominator - numerator_y * h[:, 2, 1]
    ) / denominator_squared
    jacobian = torch.stack(
        (
            torch.stack((j00, j01), dim=1),
            torch.stack((j10, j11), dim=1),
        ),
        dim=1,
    )
    projected_pivot = torch.stack(
        (
            numerator_x / safe_denominator,
            numerator_y / safe_denominator,
        ),
        dim=1,
    )

    relative_determinant = torch.linalg.det(h)
    valid = (
        torch.isfinite(pivot_xy).all(dim=1)
        & scale_valid
        & torch.isfinite(projected_pivot).all(dim=1)
        & torch.isfinite(jacobian).all(dim=(1, 2))
        & torch.isfinite(relative_determinant)
        & (torch.abs(denominator) > epsilon)
        & (torch.abs(relative_determinant) > epsilon)
    )
    return projected_pivot, jacobian, valid


def projective_circular_pushforward(
    probabilities: torch.Tensor,
    pivot_xy: torch.Tensor,
    homography: torch.Tensor,
    *,
    denominator_epsilon: float = 1e-8,
) -> CircularPosteriorTransport:
    """Push a ``[B, K]`` circular mass function through a homography.

    Angular bin centres are transformed by the homography's local Jacobian at
    ``pivot_xy``.  Mapped mass is deposited with periodic linear splatting.
    Exact identity rows bypass arithmetic so their output is bitwise identical.
    For an invalid homography the input distribution is returned unchanged,
    mass remains finite, and ``valid`` is false.
    """

    _validate_transport_inputs(probabilities, pivot_xy, homography)
    dtype = _working_dtype(probabilities)
    source = probabilities.to(dtype=dtype)
    pivot = pivot_xy.to(dtype=dtype)
    matrix = homography.to(dtype=dtype)
    finite_probability = torch.isfinite(source).all(dim=1)
    nonnegative_probability = (source >= 0.0).all(dim=1)
    safe_source = torch.clamp(torch.nan_to_num(source), min=0.0)
    source_mass = safe_source.sum(dim=1)
    probability_valid = (
        finite_probability & nonnegative_probability & (source_mass > 0.0)
    )

    safe_pivot = torch.nan_to_num(pivot)
    safe_matrix = torch.nan_to_num(matrix)
    transformed_pivot, jacobian, geometry_valid = _local_projective_jacobian(
        safe_pivot,
        safe_matrix,
        denominator_epsilon=denominator_epsilon,
    )
    geometry_valid &= torch.isfinite(pivot).all(dim=1)
    geometry_valid &= torch.isfinite(matrix).all(dim=(1, 2))

    angle_bins = source.shape[1]
    centers = _angle_centers(
        angle_bins,
        device=source.device,
        dtype=dtype,
    )
    source_directions = torch.stack(
        (torch.cos(centers), torch.sin(centers)),
        dim=1,
    )
    mapped_vectors = torch.einsum("bij,kj->bki", jacobian, source_directions)
    mapped_norm = torch.linalg.vector_norm(mapped_vectors, dim=2)
    direction_valid = (
        torch.isfinite(mapped_vectors).all(dim=(1, 2))
        & (mapped_norm > float(denominator_epsilon)).all(dim=1)
    )
    safe_vectors = torch.nan_to_num(mapped_vectors)
    mapped_angle = torch.atan2(safe_vectors[:, :, 1], safe_vectors[:, :, 0])
    mapped_coordinate = torch.remainder(
        mapped_angle * (float(angle_bins) / _TWO_PI),
        float(angle_bins),
    )
    # At the negative-zero wrap boundary, finite-precision remainder can round
    # to exactly K.  Compute the interpolation fraction before periodic integer
    # wrapping so the scatter index is always in [0, K).
    unwrapped_lower = torch.floor(mapped_coordinate)
    fraction = mapped_coordinate - unwrapped_lower
    lower = torch.remainder(
        unwrapped_lower.to(dtype=torch.long),
        angle_bins,
    )
    mapped_coordinate = lower.to(dtype=dtype) + fraction
    upper = torch.remainder(lower + 1, angle_bins)

    # Dense one-hot deposition avoids CUDA scatter_add atomic-order
    # nondeterminism.  K=360 gives a modest matrix for the formal batch size.
    lower_deposition = F.one_hot(lower, num_classes=angle_bins).to(dtype=dtype)
    upper_deposition = F.one_hot(upper, num_classes=angle_bins).to(dtype=dtype)
    deposition = (
        lower_deposition * (1.0 - fraction)[:, :, None]
        + upper_deposition * fraction[:, :, None]
    )
    transported = torch.sum(safe_source[:, :, None] * deposition, dim=1)
    transported_mass = transported.sum(dim=1)
    mass_scale = source_mass / torch.clamp(
        transported_mass,
        min=torch.finfo(dtype).tiny,
    )
    transported = transported * mass_scale[:, None]

    eye = torch.eye(3, device=matrix.device, dtype=dtype)
    identity_scale = matrix[:, 2, 2]
    identity = (
        torch.eq(matrix, identity_scale[:, None, None] * eye[None, :, :]).all(
            dim=(1, 2)
        )
        & torch.isfinite(identity_scale)
        & (torch.abs(identity_scale) > torch.finfo(dtype).tiny)
    )
    transported = torch.where(identity[:, None], safe_source, transported)
    mapped_coordinate = torch.where(
        identity[:, None],
        torch.arange(angle_bins, device=source.device, dtype=dtype)[None, :],
        mapped_coordinate,
    )
    valid = probability_valid & geometry_valid & direction_valid
    transported = torch.where(valid[:, None], transported, safe_source)
    mapped_coordinate = torch.where(
        valid[:, None],
        mapped_coordinate,
        torch.arange(angle_bins, device=source.device, dtype=dtype)[None, :],
    )
    transformed_pivot = torch.where(
        valid[:, None],
        transformed_pivot,
        safe_pivot,
    )
    target_mass = transported.sum(dim=1)
    return CircularPosteriorTransport(
        probabilities=transported,
        transformed_pivot_xy=transformed_pivot,
        mapped_bin_coordinate=mapped_coordinate,
        source_mass=source_mass,
        target_mass=target_mass,
        valid=valid,
    )


def _normalise_mass(probabilities: torch.Tensor) -> torch.Tensor:
    safe = torch.clamp(torch.nan_to_num(probabilities), min=0.0)
    return safe / torch.clamp(
        safe.sum(dim=1, keepdim=True),
        min=torch.finfo(safe.dtype).tiny,
    )


def _jensen_shannon_rows(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first = _normalise_mass(first)
    second = _normalise_mass(second)
    midpoint = 0.5 * (first + second)
    tiny = torch.finfo(first.dtype).tiny
    first_kl = torch.sum(
        first
        * (
            torch.log(torch.clamp(first, min=tiny))
            - torch.log(torch.clamp(midpoint, min=tiny))
        ),
        dim=1,
    )
    second_kl = torch.sum(
        second
        * (
            torch.log(torch.clamp(second, min=tiny))
            - torch.log(torch.clamp(midpoint, min=tiny))
        ),
        dim=1,
    )
    return 0.5 * (first_kl + second_kl)


def bidirectional_projective_posterior_consistency(
    first_probabilities: torch.Tensor,
    second_probabilities: torch.Tensor,
    first_pivot_xy: torch.Tensor,
    second_pivot_xy: torch.Tensor,
    homography_first_to_second: torch.Tensor,
    *,
    denominator_epsilon: float = 1e-8,
    detach_pivot: bool = True,
) -> BidirectionalPosteriorConsistency:
    """Measure forward and inverse projective consistency with JS divergence.

    Predicted pivots are detached by default.  This prevents a network from
    reducing the posterior JS term by moving its pivot instead of improving
    direction equivariance.  Set ``detach_pivot=False`` only for an explicit
    joint-geometry ablation; the transport itself remains differentiable.
    """

    _validate_transport_inputs(
        first_probabilities,
        first_pivot_xy,
        homography_first_to_second,
    )
    _validate_transport_inputs(
        second_probabilities,
        second_pivot_xy,
        homography_first_to_second,
    )
    if first_probabilities.shape != second_probabilities.shape:
        raise ValueError("first and second posteriors must have identical shape")

    first_transport_pivot = (
        first_pivot_xy.detach() if detach_pivot else first_pivot_xy
    )
    second_transport_pivot = (
        second_pivot_xy.detach() if detach_pivot else second_pivot_xy
    )
    forward = projective_circular_pushforward(
        first_probabilities,
        first_transport_pivot,
        homography_first_to_second,
        denominator_epsilon=denominator_epsilon,
    )
    dtype = _working_dtype(first_probabilities)
    matrix = homography_first_to_second.to(dtype=dtype)
    finite_matrix = torch.isfinite(matrix).all(dim=(1, 2))
    safe_matrix = torch.where(
        finite_matrix[:, None, None],
        matrix,
        torch.eye(3, device=matrix.device, dtype=dtype)[None, :, :],
    )
    inverse_matrix, inverse_info = torch.linalg.inv_ex(
        safe_matrix,
        check_errors=False,
    )
    inverse_finite = torch.isfinite(inverse_matrix).all(dim=(1, 2))
    safe_inverse = torch.where(
        inverse_finite[:, None, None],
        inverse_matrix,
        torch.eye(3, device=matrix.device, dtype=dtype)[None, :, :],
    )
    inverse = projective_circular_pushforward(
        second_probabilities,
        second_transport_pivot,
        safe_inverse,
        denominator_epsilon=denominator_epsilon,
    )
    forward_js = _jensen_shannon_rows(
        forward.probabilities,
        second_probabilities.to(dtype=dtype),
    )
    inverse_js = _jensen_shannon_rows(
        inverse.probabilities,
        first_probabilities.to(dtype=dtype),
    )
    valid = (
        forward.valid
        & inverse.valid
        & finite_matrix
        & inverse_finite
        & (inverse_info == 0)
    )
    if bool(valid.any()):
        loss = torch.mean(0.5 * (forward_js[valid] + inverse_js[valid]))
    else:
        # Preserve a differentiable zero when no geometric pair is usable.
        loss = (
            first_probabilities.sum() * 0.0
            + second_probabilities.sum() * 0.0
        ).to(dtype=dtype)
    return BidirectionalPosteriorConsistency(
        loss=loss,
        forward_js=forward_js,
        inverse_js=inverse_js,
        valid=valid,
        valid_fraction=valid.to(dtype=dtype).mean(),
    )


def _soft_circular_targets(
    target_direction: torch.Tensor,
    *,
    angle_bins: int,
    sigma_bins: float,
) -> torch.Tensor:
    sigma = float(sigma_bins)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("soft_target_sigma_bins must be positive and finite")
    target_angle = torch.atan2(target_direction[:, 1], target_direction[:, 0])
    centers = _angle_centers(
        angle_bins,
        device=target_direction.device,
        dtype=target_direction.dtype,
    )
    sigma_radians = sigma * (_TWO_PI / float(angle_bins))
    delta = circular_delta(centers[None, :], target_angle[:, None])
    target = torch.exp(-0.5 * (delta / sigma_radians).square())
    return target / torch.clamp(
        target.sum(dim=1, keepdim=True),
        min=torch.finfo(target.dtype).tiny,
    )


def _geodesic_normal_log_normalizer(
    precision: torch.Tensor,
) -> torch.Tensor:
    """Log normalizer on the circle for exp(-0.5 * precision * delta^2).

    Integrating the shortest signed delta over ``[-pi, pi)`` gives

    ``Z = sqrt(2*pi) * sigma * erf(pi / (sqrt(2) * sigma))``.
    """

    sigma = torch.rsqrt(precision)
    erf_term = torch.erf(math.pi / (math.sqrt(2.0) * sigma))
    return (
        0.5 * math.log(_TWO_PI)
        + torch.log(sigma)
        + torch.log(
            torch.clamp(erf_term, min=torch.finfo(precision.dtype).tiny)
        )
    )


def _masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    graph_anchor: torch.Tensor,
) -> torch.Tensor:
    if bool(valid.any()):
        return torch.mean(values[valid])
    return graph_anchor.sum() * 0.0


def projective_circular_supervised_loss_v3(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    pivot_weight: float = 1.0,
    direct_nll_weight: float = 1.0,
    bin_ce_weight: float = 0.25,
    fused_posterior_weight: float = 0.50,
    fused_mean_weight: float = 0.25,
    overconfidence_weight: float = 0.10,
    precision_barrier_weight: float = 0.01,
    soft_target_sigma_bins: float = 1.25,
    min_direct_precision: float = 0.05,
    direct_precision_cap: float = 400.0,
    direct_precision_temperature: float = 1.0,
    bin_expert_power: float = 0.5,
    direct_expert_power: float = 0.5,
    posterior_bins: int = 360,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """V3 supervised objective for the projective circular PoE.

    It combines:

    * weighted pivot heatmap regression;
    * a proper heteroscedastic geodesic-normal circular NLL that explicitly
      trains
      ``log_variance_raw`` as the *direct* expert's log angular variance;
    * circular soft-target cross entropy for the bin expert;
    * KL and continuous-mean supervision for the fused posterior;
    * an explicit resultant-length and raw-precision barrier that discourages
      confidence sharper than the finite-width target distribution.
    """

    _validate_four_heads(
        pivot_logits,
        direction_raw,
        angle_logits,
        log_variance_raw,
    )
    batch = pivot_logits.shape[0]
    if target_heatmap.shape != pivot_logits.shape:
        raise ValueError("target_heatmap must match pivot_logits shape")
    if target_direction.shape != (batch, 2):
        raise ValueError("target_direction must have shape [B, 2]")
    weights = (
        pivot_weight,
        direct_nll_weight,
        bin_ce_weight,
        fused_posterior_weight,
        fused_mean_weight,
        overconfidence_weight,
        precision_barrier_weight,
    )
    if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights):
        raise ValueError("loss weights must be finite and non-negative")

    dtype = _working_dtype(angle_logits)
    pivot_work = pivot_logits.to(dtype=dtype)
    heatmap_work = target_heatmap.to(dtype=dtype)
    probability = torch.sigmoid(pivot_work)
    pixel_weight = 1.0 + 9.0 * heatmap_work
    pivot_rows = torch.mean(
        pixel_weight * (probability - heatmap_work).square(),
        dim=(1, 2, 3),
    )
    pivot_valid = (
        torch.isfinite(pivot_work).all(dim=(1, 2, 3))
        & torch.isfinite(heatmap_work).all(dim=(1, 2, 3))
    )
    pivot_loss = _masked_mean(
        pivot_rows,
        pivot_valid,
        graph_anchor=pivot_work,
    )

    direction_work = direction_raw.to(dtype=dtype)
    target_work = target_direction.to(dtype=dtype)
    target_norm = torch.linalg.vector_norm(target_work, dim=1)
    target_unit = target_work / torch.clamp(target_norm[:, None], min=1e-12)
    direct_norm = torch.linalg.vector_norm(direction_work, dim=1)
    direct_unit = direction_work / torch.clamp(direct_norm[:, None], min=1e-12)
    direct_angle = torch.atan2(direct_unit[:, 1], direct_unit[:, 0])
    target_angle = torch.atan2(target_unit[:, 1], target_unit[:, 0])
    precision = _bounded_direct_precision(
        log_variance_raw[:, 0].to(dtype=dtype),
        min_direct_precision=min_direct_precision,
        max_direct_precision=direct_precision_cap,
    )
    direct_delta = circular_delta(direct_angle, target_angle)
    direct_nll_rows = (
        0.5 * precision * direct_delta.square()
        + _geodesic_normal_log_normalizer(precision)
    )
    direction_target_valid = (
        torch.isfinite(target_work).all(dim=1) & (target_norm > 1e-8)
    )
    direct_valid = (
        direction_target_valid
        & torch.isfinite(direction_work).all(dim=1)
        & (direct_norm > 1e-8)
        & torch.isfinite(log_variance_raw[:, 0])
        & torch.isfinite(direct_nll_rows)
    )
    direct_nll = _masked_mean(
        direct_nll_rows,
        direct_valid,
        graph_anchor=direction_work,
    )

    soft_target = _soft_circular_targets(
        target_unit,
        angle_bins=angle_logits.shape[1],
        sigma_bins=soft_target_sigma_bins,
    )
    angle_work = angle_logits.to(dtype=dtype)
    bin_ce_rows = torch.sum(
        -soft_target * F.log_softmax(angle_work, dim=1),
        dim=1,
    )
    bin_valid = (
        direction_target_valid
        & torch.isfinite(angle_work).all(dim=1)
        & torch.isfinite(bin_ce_rows)
    )
    bin_ce = _masked_mean(
        bin_ce_rows,
        bin_valid,
        graph_anchor=angle_work,
    )

    fused = fuse_circular_experts(
        pivot_work,
        direction_work,
        angle_work,
        log_variance_raw.to(dtype=dtype),
        min_direct_precision=min_direct_precision,
        direct_precision_cap=direct_precision_cap,
        direct_precision_temperature=direct_precision_temperature,
        bin_expert_power=bin_expert_power,
        direct_expert_power=direct_expert_power,
        posterior_bins=posterior_bins,
    )
    tiny = torch.finfo(dtype).tiny
    fused_soft_target = _soft_circular_targets(
        target_unit,
        angle_bins=fused.probabilities.shape[1],
        sigma_bins=(
            float(soft_target_sigma_bins)
            * float(fused.probabilities.shape[1])
            / float(angle_logits.shape[1])
        ),
    )
    fused_kl_rows = torch.sum(
        fused_soft_target
        * (
            torch.log(torch.clamp(fused_soft_target, min=tiny))
            - fused.log_probabilities
        ),
        dim=1,
    )
    fused_delta = circular_delta(fused.mean_angle, target_angle)
    fused_mean_rows = 1.0 - torch.cos(fused_delta)
    target_resultant = torch.linalg.vector_norm(
        torch.stack(
            (
                torch.sum(
                    fused_soft_target
                    * torch.cos(
                        _angle_centers(
                            fused.probabilities.shape[1],
                            device=angle_work.device,
                            dtype=dtype,
                        )
                    )[None, :],
                    dim=1,
                ),
                torch.sum(
                    fused_soft_target
                    * torch.sin(
                        _angle_centers(
                            fused.probabilities.shape[1],
                            device=angle_work.device,
                            dtype=dtype,
                        )
                    )[None, :],
                    dim=1,
                ),
            ),
            dim=1,
        ),
        dim=1,
    )
    overconfidence_rows = torch.relu(
        fused.resultant_length - target_resultant
    ).square()
    fused_valid = direction_target_valid & fused.valid
    fused_kl = _masked_mean(
        fused_kl_rows,
        fused_valid,
        graph_anchor=angle_work,
    )
    fused_mean = _masked_mean(
        fused_mean_rows,
        fused_valid,
        graph_anchor=direction_work,
    )
    overconfidence = _masked_mean(
        overconfidence_rows,
        fused_valid,
        graph_anchor=angle_work,
    )

    raw_log_variance = log_variance_raw[:, 0].to(dtype=dtype)
    minimum_log_variance = -math.log(float(direct_precision_cap))
    maximum_log_variance = -math.log(float(min_direct_precision))
    precision_barrier_rows = (
        torch.relu(minimum_log_variance - raw_log_variance).square()
        + torch.relu(raw_log_variance - maximum_log_variance).square()
    )
    precision_barrier = _masked_mean(
        precision_barrier_rows,
        torch.isfinite(raw_log_variance),
        graph_anchor=raw_log_variance,
    )

    total = (
        float(pivot_weight) * pivot_loss
        + float(direct_nll_weight) * direct_nll
        + float(bin_ce_weight) * bin_ce
        + float(fused_posterior_weight) * fused_kl
        + float(fused_mean_weight) * fused_mean
        + float(overconfidence_weight) * overconfidence
        + float(precision_barrier_weight) * precision_barrier
    )
    supervised_valid = pivot_valid & direction_target_valid & fused.valid
    return total, {
        "pivot_loss": pivot_loss.detach(),
        "direct_circular_nll_loss": direct_nll.detach(),
        "bin_soft_target_ce_loss": bin_ce.detach(),
        "fused_posterior_kl_loss": fused_kl.detach(),
        "fused_mean_loss": fused_mean.detach(),
        "overconfidence_loss": overconfidence.detach(),
        "precision_barrier_loss": precision_barrier.detach(),
        "mean_direct_precision": precision.mean().detach(),
        "mean_fused_resultant": fused.resultant_length.mean().detach(),
        "supervised_valid_fraction": supervised_valid.to(dtype=dtype).mean().detach(),
    }
