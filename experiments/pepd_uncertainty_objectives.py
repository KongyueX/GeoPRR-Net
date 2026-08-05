"""Frozen uncertainty-objective variants for the secondary PEPD ablation.

The variants change only the angular uncertainty term:

* ``learned_heteroscedastic``: original per-sample learned log-variance NLL;
* ``global_homoscedastic``: one learned global log-variance scalar NLL;
* ``no_angular_nll``: remove the angular NLL term.

The global scalar is initialized a priori, optimized on training batches only,
and never selected or fitted on grouped validation.  The per-sample variance
head remains present in every arm so the predictive architecture is held
constant, but it is outside the objective for the latter two arms.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from experiments.pepd_convergence_protocol import (
    GLOBAL_LOG_VARIANCE_INITIAL_VALUE,
    LOG_VARIANCE_CLAMP_MAX,
    LOG_VARIANCE_CLAMP_MIN,
    UNCERTAINTY_MECHANISM_ARMS,
    uncertainty_mechanism_arm,
)
from experiments.probabilistic_pivot_direction import (
    _soft_circular_targets,
    circular_delta,
    decode_probabilistic_pivot_direction,
)


class UncertaintySemantics(NamedTuple):
    log_variance: torch.Tensor | None
    angle_std_degrees: torch.Tensor | None
    calibration_semantics: str
    sample_ranking_available: bool


def make_global_log_variance(
    *,
    device: torch.device,
) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.tensor(
            GLOBAL_LOG_VARIANCE_INITIAL_VALUE,
            device=device,
            dtype=torch.float32,
        )
    )


def uncertainty_semantics(
    log_variance_raw: torch.Tensor,
    *,
    mode: str,
    global_log_variance_raw: torch.Tensor | None = None,
) -> UncertaintySemantics:
    uncertainty_mechanism_arm(mode)
    batch = int(log_variance_raw.shape[0])
    if log_variance_raw.shape != (batch, 1):
        raise ValueError("log_variance_raw must have shape [batch, 1]")
    if mode == "learned_heteroscedastic":
        log_variance = torch.clamp(
            log_variance_raw[:, 0].float(),
            min=LOG_VARIANCE_CLAMP_MIN,
            max=LOG_VARIANCE_CLAMP_MAX,
        )
        semantics = "learned_per_sample_heteroscedastic_log_variance"
        ranking_available = True
    elif mode == "global_homoscedastic":
        if global_log_variance_raw is None:
            raise ValueError(
                "global_homoscedastic requires the learned global scalar"
            )
        if global_log_variance_raw.numel() != 1:
            raise ValueError("global log-variance must contain one scalar")
        scalar = torch.clamp(
            global_log_variance_raw.float().reshape(()),
            min=LOG_VARIANCE_CLAMP_MIN,
            max=LOG_VARIANCE_CLAMP_MAX,
        )
        log_variance = scalar.expand(batch)
        semantics = "learned_global_homoscedastic_log_variance"
        ranking_available = False
    else:
        return UncertaintySemantics(
            None,
            None,
            "unavailable_no_trained_uncertainty",
            False,
        )
    std = torch.exp(0.5 * log_variance) * (180.0 / math.pi)
    return UncertaintySemantics(
        log_variance,
        std,
        semantics,
        ranking_available,
    )


def probabilistic_direction_loss_with_uncertainty_mode(
    pivot_logits: torch.Tensor,
    direction_raw: torch.Tensor,
    angle_logits: torch.Tensor,
    log_variance_raw: torch.Tensor,
    target_heatmap: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    pivot_weight: float,
    bin_weight: float,
    vector_weight: float,
    soft_target_sigma_bins: float,
    uncertainty_mode: str,
    global_log_variance_raw: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | str | bool | None],
]:
    uncertainty_mechanism_arm(uncertainty_mode)
    probability = torch.sigmoid(pivot_logits.float())
    target_heatmap = target_heatmap.float()
    pixel_weight = 1.0 + 9.0 * target_heatmap
    pivot_loss = torch.mean(pixel_weight * (probability - target_heatmap) ** 2)

    prediction = decode_probabilistic_pivot_direction(
        pivot_logits.float(),
        direction_raw.float(),
        angle_logits.float(),
        log_variance_raw.float(),
    )
    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-8)
    predicted_angle = torch.atan2(
        prediction.direction[:, 1],
        prediction.direction[:, 0],
    )
    target_angle = torch.atan2(
        target_direction[:, 1],
        target_direction[:, 0],
    )
    delta = circular_delta(predicted_angle, target_angle)
    semantics = uncertainty_semantics(
        log_variance_raw,
        mode=uncertainty_mode,
        global_log_variance_raw=global_log_variance_raw,
    )
    squared_angular_error_loss = torch.mean(0.5 * delta.square())
    if semantics.log_variance is not None:
        angular_nll_loss = torch.mean(
            0.5
            * (
                delta.square() * torch.exp(-semantics.log_variance)
                + semantics.log_variance
            )
        )
        angular_objective = angular_nll_loss
    else:
        angular_nll_loss = delta.sum() * 0.0
        angular_objective = angular_nll_loss

    soft_target = _soft_circular_targets(
        target_direction,
        angle_bins=angle_logits.shape[1],
        sigma_bins=soft_target_sigma_bins,
    )
    bin_loss = torch.mean(
        torch.sum(
            -soft_target * F.log_softmax(angle_logits.float(), dim=1),
            dim=1,
        )
    )
    cosine_loss = torch.mean(
        1.0 - torch.sum(prediction.direction * target_direction, dim=1)
    )
    direction_loss = (
        angular_objective
        + float(bin_weight) * bin_loss
        + float(vector_weight) * cosine_loss
    )
    total = float(pivot_weight) * pivot_loss + direction_loss
    mean_std = (
        None
        if semantics.angle_std_degrees is None
        else semantics.angle_std_degrees.mean().detach()
    )
    global_value = (
        None
        if uncertainty_mode != "global_homoscedastic"
        else semantics.log_variance[0].detach()
    )
    return total, {
        "pivot_loss": pivot_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "angular_nll_loss": angular_nll_loss.detach(),
        "angular_nll_in_training_objective": (
            uncertainty_mode != "no_angular_nll"
        ),
        "squared_angular_error_loss": (
            squared_angular_error_loss.detach()
        ),
        "bin_loss": bin_loss.detach(),
        "cosine_loss": cosine_loss.detach(),
        "mean_angle_std_degrees": mean_std,
        "global_log_variance": global_value,
        "calibration_semantics": semantics.calibration_semantics,
        "sample_ranking_available": semantics.sample_ranking_available,
    }


def uncertainty_arm_names() -> tuple[str, ...]:
    return UNCERTAINTY_MECHANISM_ARMS
