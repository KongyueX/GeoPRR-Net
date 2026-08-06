"""CRRM-V5 deployable soft reliability fusion.

This module is deliberately data-agnostic.  It consumes only quantities that
are available at inference time:

* 23 deployable reliability features;
* three normalized-progress expert predictions; and
* explicit per-expert availability flags.

The feature normalizer is frozen from training-partition statistics.  The gate
predicts one log error variance per expert and converts those variances into a
masked inverse-variance softmax.  If no gated expert is available, the runtime
falls back first to the frozen production anchor and then to the mask-geometry
prediction.  No label, sample identity, or dataset-specific field is accepted
by :class:`CRRMV5SoftReliabilityGate.forward`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL: Final[str] = "cagh_crrm_v5_soft_reliability_gate_v1"
FEATURE_DIM: Final[int] = 23
HIDDEN_DIMS: Final[tuple[int, int]] = (32, 16)
EXPERT_NAMES: Final[tuple[str, str, str]] = (
    "mask_geometry",
    "pepd",
    "scalemark",
)
EXPERT_COUNT: Final[int] = len(EXPERT_NAMES)
DEFAULT_MASK_EXPERT_INDEX: Final[int] = 0


def fit_robust_median_iqr(
    features: torch.Tensor,
    *,
    minimum_iqr: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit finite-only per-coordinate median and IQR statistics.

    Columns containing no finite observation receive median ``0`` and scale
    ``1``.  Constant (or nearly constant) columns receive scale ``1``.  This
    keeps the transform total and deterministic without consulting evaluation
    data.
    """

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must have shape [N, {FEATURE_DIM}]")
    if features.shape[0] < 1:
        raise ValueError("at least one feature row is required")
    if not math.isfinite(float(minimum_iqr)) or float(minimum_iqr) <= 0.0:
        raise ValueError("minimum_iqr must be finite and positive")

    values = features.detach().to(dtype=torch.float32)
    medians = torch.zeros(FEATURE_DIM, dtype=values.dtype, device=values.device)
    iqrs = torch.ones_like(medians)
    for feature_index in range(FEATURE_DIM):
        column = values[:, feature_index]
        finite_values = column[torch.isfinite(column)]
        if finite_values.numel() == 0:
            continue
        median = torch.quantile(finite_values, 0.5)
        lower = torch.quantile(finite_values, 0.25)
        upper = torch.quantile(finite_values, 0.75)
        scale = upper - lower
        medians[feature_index] = median
        if bool(torch.isfinite(scale)) and float(scale) > float(minimum_iqr):
            iqrs[feature_index] = scale
    return medians, iqrs


class RobustMedianIQRNormalizer(nn.Module):
    """Frozen median-IQR transform with finite-value imputation."""

    def __init__(
        self,
        medians: torch.Tensor | None = None,
        iqrs: torch.Tensor | None = None,
        *,
        clip: float = 12.0,
    ) -> None:
        super().__init__()
        if (medians is None) != (iqrs is None):
            raise ValueError("medians and iqrs must be provided together")
        if not math.isfinite(float(clip)) or float(clip) <= 0.0:
            raise ValueError("clip must be finite and positive")
        if medians is None:
            median_tensor = torch.zeros(FEATURE_DIM, dtype=torch.float32)
            iqr_tensor = torch.ones(FEATURE_DIM, dtype=torch.float32)
        else:
            median_tensor = torch.as_tensor(medians, dtype=torch.float32).detach()
            iqr_tensor = torch.as_tensor(iqrs, dtype=torch.float32).detach()
            if median_tensor.shape != (FEATURE_DIM,):
                raise ValueError(f"medians must have shape [{FEATURE_DIM}]")
            if iqr_tensor.shape != (FEATURE_DIM,):
                raise ValueError(f"iqrs must have shape [{FEATURE_DIM}]")
            if not bool(torch.isfinite(median_tensor).all()):
                raise ValueError("medians must be finite")
            if not bool(torch.isfinite(iqr_tensor).all()):
                raise ValueError("iqrs must be finite")
            if not bool((iqr_tensor > 0.0).all()):
                raise ValueError("iqrs must be positive")
            median_tensor = median_tensor.clone()
            iqr_tensor = iqr_tensor.clone()
        self.register_buffer("medians", median_tensor)
        self.register_buffer("iqrs", iqr_tensor)
        self.clip = float(clip)

    @classmethod
    def from_training_features(
        cls,
        features: torch.Tensor,
        *,
        minimum_iqr: float = 1e-6,
        clip: float = 12.0,
    ) -> "RobustMedianIQRNormalizer":
        medians, iqrs = fit_robust_median_iqr(
            features,
            minimum_iqr=minimum_iqr,
        )
        return cls(medians, iqrs, clip=clip)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"features must have shape [B, {FEATURE_DIM}]")
        values = features.to(dtype=torch.float32)
        medians = self.medians.to(device=values.device)
        iqrs = self.iqrs.to(device=values.device)
        filled = torch.where(torch.isfinite(values), values, medians.unsqueeze(0))
        normalized = (filled - medians.unsqueeze(0)) / iqrs.unsqueeze(0)
        return torch.clamp(normalized, min=-self.clip, max=self.clip)


def masked_inverse_variance_softmax(
    log_variances: torch.Tensor,
    availability: torch.Tensor,
    *,
    temperature: float = 1.0,
    minimum_log_variance: float = -12.0,
    maximum_log_variance: float = 6.0,
) -> torch.Tensor:
    """Return stable inverse-variance weights with exact invalid-expert zeros."""

    if log_variances.ndim != 2 or log_variances.shape[1] != EXPERT_COUNT:
        raise ValueError(
            f"log_variances must have shape [B, {EXPERT_COUNT}]"
        )
    if availability.shape != log_variances.shape:
        raise ValueError("availability must match log_variances")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if float(minimum_log_variance) >= float(maximum_log_variance):
        raise ValueError("log-variance bounds are reversed")

    values = log_variances.to(dtype=torch.float32)
    valid = availability.to(dtype=torch.bool) & torch.isfinite(values)
    bounded = torch.nan_to_num(
        values,
        nan=float(maximum_log_variance),
        posinf=float(maximum_log_variance),
        neginf=float(minimum_log_variance),
    ).clamp(min=float(minimum_log_variance), max=float(maximum_log_variance))
    logits = -bounded / float(temperature)
    very_negative = torch.finfo(logits.dtype).min
    logits = torch.where(valid, logits, torch.full_like(logits, very_negative))

    has_valid = valid.any(dim=1, keepdim=True)
    # softmax([-inf, -inf, -inf]) is undefined.  A zero row is used solely for
    # that case and is removed by the validity mask immediately afterwards.
    stable_logits = torch.where(has_valid, logits, torch.zeros_like(logits))
    weights = torch.softmax(stable_logits, dim=1) * valid.to(logits.dtype)
    denominator = weights.sum(dim=1, keepdim=True)
    return torch.where(
        denominator > 0.0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny),
        torch.zeros_like(weights),
    )


@dataclass(frozen=True)
class CRRMV5Output:
    """Tensor-only gate output suitable for training and deployment logging."""

    normalized_features: torch.Tensor
    log_variances: torch.Tensor
    weights: torch.Tensor
    expert_progress: torch.Tensor
    expert_valid: torch.Tensor
    fused_progress: torch.Tensor
    fused_available: torch.Tensor
    anchor_progress: torch.Tensor
    anchor_available: torch.Tensor
    used_soft_fusion: torch.Tensor
    used_production_anchor: torch.Tensor
    used_mask_fallback: torch.Tensor


def _optional_batch_vector(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    result = value.to(device=device, dtype=torch.float32)
    if result.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [B]")
    return result


def _optional_availability(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    result = value.to(device=device, dtype=torch.bool)
    if result.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [B]")
    return result


class CRRMV5SoftReliabilityGate(nn.Module):
    """23->32->16->3 heteroscedastic gate for three frozen experts."""

    def __init__(
        self,
        *,
        normalizer: RobustMedianIQRNormalizer | None = None,
        temperature: float = 1.0,
        minimum_log_variance: float = -12.0,
        maximum_log_variance: float = 6.0,
        mask_expert_index: int = DEFAULT_MASK_EXPERT_INDEX,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if float(minimum_log_variance) >= float(maximum_log_variance):
            raise ValueError("log-variance bounds are reversed")
        if int(mask_expert_index) not in range(EXPERT_COUNT):
            raise ValueError("mask_expert_index is outside the expert roster")
        self.normalizer = normalizer or RobustMedianIQRNormalizer()
        self.temperature = float(temperature)
        self.minimum_log_variance = float(minimum_log_variance)
        self.maximum_log_variance = float(maximum_log_variance)
        self.mask_expert_index = int(mask_expert_index)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[1], EXPERT_COUNT),
        )
        for layer in self.network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Equal, moderately cautious initial reliabilities keep the initial
        # fusion unbiased across experts while still yielding useful NLL scale.
        nn.init.constant_(self.network[-1].bias, math.log(0.10**2))

    def forward(
        self,
        features: torch.Tensor,
        expert_progress: torch.Tensor,
        expert_availability: torch.Tensor,
        *,
        production_anchor_progress: torch.Tensor | None = None,
        production_anchor_availability: torch.Tensor | None = None,
        mask_progress: torch.Tensor | None = None,
        mask_availability: torch.Tensor | None = None,
    ) -> CRRMV5Output:
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"features must have shape [B, {FEATURE_DIM}]")
        batch_size = features.shape[0]
        device = features.device
        progress = expert_progress.to(device=device, dtype=torch.float32)
        available = expert_availability.to(device=device, dtype=torch.bool)
        expected_shape = (batch_size, EXPERT_COUNT)
        if progress.shape != expected_shape:
            raise ValueError(f"expert_progress must have shape {expected_shape}")
        if available.shape != expected_shape:
            raise ValueError(
                f"expert_availability must have shape {expected_shape}"
            )

        normalized = self.normalizer(features)
        raw_log_variances = self.network(normalized)
        log_variances = torch.nan_to_num(
            raw_log_variances,
            nan=self.maximum_log_variance,
            posinf=self.maximum_log_variance,
            neginf=self.minimum_log_variance,
        ).clamp(
            min=self.minimum_log_variance,
            max=self.maximum_log_variance,
        )
        expert_valid = available & torch.isfinite(progress)
        weights = masked_inverse_variance_softmax(
            log_variances,
            expert_valid,
            temperature=self.temperature,
            minimum_log_variance=self.minimum_log_variance,
            maximum_log_variance=self.maximum_log_variance,
        )
        safe_progress = torch.where(
            expert_valid,
            progress,
            torch.zeros_like(progress),
        )
        candidate = torch.sum(weights * safe_progress, dim=1)
        used_soft_fusion = expert_valid.any(dim=1)

        explicit_anchor = _optional_batch_vector(
            production_anchor_progress,
            batch_size=batch_size,
            device=device,
            name="production_anchor_progress",
        )
        explicit_anchor_available = _optional_availability(
            production_anchor_availability,
            batch_size=batch_size,
            device=device,
            name="production_anchor_availability",
        )
        default_mask = progress[:, self.mask_expert_index]
        default_mask_available = available[:, self.mask_expert_index]

        anchor = default_mask if explicit_anchor is None else explicit_anchor
        if explicit_anchor_available is None:
            anchor_flag = (
                default_mask_available
                if explicit_anchor is None
                else torch.ones(batch_size, dtype=torch.bool, device=device)
            )
        else:
            anchor_flag = explicit_anchor_available
        anchor_valid = anchor_flag & torch.isfinite(anchor)

        explicit_mask = _optional_batch_vector(
            mask_progress,
            batch_size=batch_size,
            device=device,
            name="mask_progress",
        )
        explicit_mask_available = _optional_availability(
            mask_availability,
            batch_size=batch_size,
            device=device,
            name="mask_availability",
        )
        mask = default_mask if explicit_mask is None else explicit_mask
        if explicit_mask_available is None:
            mask_flag = (
                default_mask_available
                if explicit_mask is None
                else torch.ones(batch_size, dtype=torch.bool, device=device)
            )
        else:
            mask_flag = explicit_mask_available
        mask_valid = mask_flag & torch.isfinite(mask)

        used_production_anchor = (~used_soft_fusion) & anchor_valid
        used_mask_fallback = (
            (~used_soft_fusion) & (~anchor_valid) & mask_valid
        )
        fused_available = (
            used_soft_fusion | used_production_anchor | used_mask_fallback
        )
        fused = candidate
        fused = torch.where(used_production_anchor, anchor, fused)
        fused = torch.where(used_mask_fallback, mask, fused)
        fused = torch.where(
            fused_available,
            fused,
            torch.full_like(fused, float("nan")),
        )

        # Anchor-regret training uses the same deployable fallback hierarchy.
        regret_anchor = torch.where(anchor_valid, anchor, mask)
        regret_anchor_available = anchor_valid | ((~anchor_valid) & mask_valid)
        regret_anchor = torch.where(
            regret_anchor_available,
            regret_anchor,
            torch.full_like(regret_anchor, float("nan")),
        )
        return CRRMV5Output(
            normalized_features=normalized,
            log_variances=log_variances,
            weights=weights,
            expert_progress=progress,
            expert_valid=expert_valid,
            fused_progress=fused,
            fused_available=fused_available,
            anchor_progress=regret_anchor,
            anchor_available=regret_anchor_available,
            used_soft_fusion=used_soft_fusion,
            used_production_anchor=used_production_anchor,
            used_mask_fallback=used_mask_fallback,
        )


@dataclass(frozen=True)
class CRRMV5Loss:
    total: torch.Tensor
    expert_nll: torch.Tensor
    fused_huber: torch.Tensor
    anchor_regret: torch.Tensor
    valid_expert_terms: torch.Tensor
    valid_fused_rows: torch.Tensor
    valid_anchor_rows: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("masked mean inputs must have matching shapes")
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    denominator = mask.to(values.dtype).sum().clamp_min(1.0)
    return safe_values.sum() / denominator


def crrm_v5_loss(
    output: CRRMV5Output,
    target_progress: torch.Tensor,
    *,
    expert_nll_weight: float = 1.0,
    fused_huber_weight: float = 2.0,
    anchor_regret_weight: float = 0.5,
    huber_beta: float = 0.02,
    anchor_regret_margin: float = 0.0,
    maximum_absolute_residual: float = 1e3,
) -> CRRMV5Loss:
    """Joint heteroscedastic, fused-regression, and anchor-regret objective."""

    scalar_parameters = {
        "expert_nll_weight": expert_nll_weight,
        "fused_huber_weight": fused_huber_weight,
        "anchor_regret_weight": anchor_regret_weight,
    }
    for name, value in scalar_parameters.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(float(huber_beta)) or float(huber_beta) <= 0.0:
        raise ValueError("huber_beta must be finite and positive")
    if (
        not math.isfinite(float(anchor_regret_margin))
        or float(anchor_regret_margin) < 0.0
    ):
        raise ValueError("anchor_regret_margin must be finite and nonnegative")
    if (
        not math.isfinite(float(maximum_absolute_residual))
        or float(maximum_absolute_residual) <= 0.0
    ):
        raise ValueError("maximum_absolute_residual must be finite and positive")

    target = target_progress.to(
        device=output.fused_progress.device,
        dtype=torch.float32,
    )
    batch_size = output.fused_progress.shape[0]
    if target.shape != (batch_size,):
        raise ValueError("target_progress must have shape [B]")
    target_valid = torch.isfinite(target)
    safe_target = torch.where(target_valid, target, torch.zeros_like(target))

    expert_target = safe_target.unsqueeze(1)
    safe_expert = torch.where(
        output.expert_valid,
        output.expert_progress,
        torch.zeros_like(output.expert_progress),
    )
    residual = (safe_expert - expert_target).clamp(
        min=-float(maximum_absolute_residual),
        max=float(maximum_absolute_residual),
    )
    bounded_log_variance = output.log_variances.clamp(min=-12.0, max=6.0)
    expert_terms = 0.5 * (
        torch.exp(-bounded_log_variance) * residual.square()
        + bounded_log_variance
    )
    expert_mask = output.expert_valid & target_valid.unsqueeze(1)
    expert_nll = _masked_mean(expert_terms, expert_mask)

    fused_mask = output.fused_available & target_valid
    safe_fused = torch.where(
        output.fused_available,
        output.fused_progress,
        torch.zeros_like(output.fused_progress),
    )
    fused_terms = F.smooth_l1_loss(
        safe_fused,
        safe_target,
        reduction="none",
        beta=float(huber_beta),
    )
    fused_huber = _masked_mean(fused_terms, fused_mask)

    anchor_mask = fused_mask & output.anchor_available
    safe_anchor = torch.where(
        output.anchor_available,
        output.anchor_progress,
        torch.zeros_like(output.anchor_progress),
    )
    fused_absolute_error = torch.abs(safe_fused - safe_target)
    anchor_absolute_error = torch.abs(safe_anchor - safe_target)
    regret_terms = torch.relu(
        fused_absolute_error
        - anchor_absolute_error
        - float(anchor_regret_margin)
    )
    anchor_regret = _masked_mean(regret_terms, anchor_mask)

    total = (
        float(expert_nll_weight) * expert_nll
        + float(fused_huber_weight) * fused_huber
        + float(anchor_regret_weight) * anchor_regret
    )
    return CRRMV5Loss(
        total=total,
        expert_nll=expert_nll,
        fused_huber=fused_huber,
        anchor_regret=anchor_regret,
        valid_expert_terms=expert_mask.sum(),
        valid_fused_rows=fused_mask.sum(),
        valid_anchor_rows=anchor_mask.sum(),
    )


__all__ = [
    "CRRMV5Loss",
    "CRRMV5Output",
    "CRRMV5SoftReliabilityGate",
    "DEFAULT_MASK_EXPERT_INDEX",
    "EXPERT_COUNT",
    "EXPERT_NAMES",
    "FEATURE_DIM",
    "HIDDEN_DIMS",
    "PROTOCOL",
    "RobustMedianIQRNormalizer",
    "crrm_v5_loss",
    "fit_robust_median_iqr",
    "masked_inverse_variance_softmax",
]
