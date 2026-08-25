"""Representation-conditioned arbitration over complementary RCMT risk heads."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import torch
from torch import nn

from experiments.remst_resnet18 import (
    RESNET18_REPRESENTATION_FEATURES,
)
from experiments.train_remst_resnet18_relative_context_transport import (
    BRIDGE_LAYERS,
    GEOMETRY_FEATURES,
    RelativeContextMomentHead,
    RelativeContextMomentTransport,
)


ARCHITECTURE: Final[str] = (
    "ReMST-ResNet18-Representation-Conditioned-Multi-Risk-Moment-Arbitration"
)
PUBLICATION_NAME: Final[str] = "R²MT-Net"
PUBLICATION_FULL_NAME: Final[str] = (
    "Representation-Conditioned Multi-Risk Moment Transport Network"
)
RISK_HEAD_NAMES: Final[tuple[str, ...]] = (
    "mean_risk",
    "tail_risk",
    "combined_risk",
)
DEFAULT_PRIOR_WEIGHTS: Final[tuple[float, ...]] = (0.475, 0.280, 0.245)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def publication_model_identity() -> dict[str, str]:
    """Return the paper-facing identity without altering checkpoint keys."""

    return {
        "machine_key": "r2mt_net",
        "short_name": PUBLICATION_NAME,
        "display_name": PUBLICATION_NAME,
        "full_name": PUBLICATION_FULL_NAME,
        "architecture": "R2MT-Net-v1",
        "checkpoint_architecture": ARCHITECTURE,
        "backbone": "torchvision ResNet-18 (one shared parameter set)",
    }


class RepresentationConditionedRiskGate(nn.Module):
    """Predict simplex deviations from a validated global risk prior."""

    def __init__(
        self,
        *,
        latent_features: int = 32,
        hidden_features: int = 64,
        prior_weights: Sequence[float] = DEFAULT_PRIOR_WEIGHTS,
        representation_enabled: bool = True,
    ) -> None:
        super().__init__()
        prior = tuple(float(value) for value in prior_weights)
        _require(
            latent_features >= 8
            and hidden_features >= 8
            and len(prior) == len(RISK_HEAD_NAMES)
            and all(math.isfinite(value) and value > 0.0 for value in prior),
            "risk arbitration gate configuration differs",
        )
        normalized = torch.as_tensor(prior, dtype=torch.float32)
        normalized = normalized / normalized.sum()
        self.latent_features = int(latent_features)
        self.hidden_features = int(hidden_features)
        self.representation_enabled = bool(representation_enabled)
        self.scalar_features = 10
        self.relation_features = (
            5 * self.latent_features + GEOMETRY_FEATURES + self.scalar_features
        )
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(RESNET18_REPRESENTATION_FEATURES),
            nn.Linear(RESNET18_REPRESENTATION_FEATURES, self.latent_features),
            nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(self.relation_features),
            nn.Linear(self.relation_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, len(RISK_HEAD_NAMES)),
        )
        output = self.router[-1]
        _require(isinstance(output, nn.Linear), "risk arbitration output differs")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.register_buffer("prior_weights", normalized)
        self.register_buffer("prior_logits", normalized.log())

    def forward(
        self,
        raw_representation: torch.Tensor,
        sarn_representation: torch.Tensor,
        geometry_features: torch.Tensor,
        raw_mean: torch.Tensor,
        sarn_mean: torch.Tensor,
        base_mean: torch.Tensor,
        expert_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(raw_representation.shape[0])
        _require(
            raw_representation.shape
            == sarn_representation.shape
            == (batch, RESNET18_REPRESENTATION_FEATURES)
            and geometry_features.shape == (batch, GEOMETRY_FEATURES)
            and raw_mean.shape == sarn_mean.shape == base_mean.shape == (batch,)
            and expert_shifts.shape == (batch, len(RISK_HEAD_NAMES)),
            "risk arbitration inputs differ",
        )
        raw = self.shared_projection(raw_representation.float())
        sarn = self.shared_projection(sarn_representation.float())
        if not self.representation_enabled:
            raw = torch.zeros_like(raw)
            sarn = torch.zeros_like(sarn)
        scalar = torch.cat(
            (
                torch.stack((raw_mean, sarn_mean, base_mean), dim=1).float(),
                expert_shifts.float(),
                torch.stack(
                    (
                        expert_shifts[:, 1] - expert_shifts[:, 0],
                        expert_shifts[:, 2] - expert_shifts[:, 0],
                        expert_shifts[:, 2] - expert_shifts[:, 1],
                        torch.abs(sarn_mean.float() - raw_mean.float()),
                    ),
                    dim=1,
                ),
            ),
            dim=1,
        )
        relation = torch.cat(
            (
                raw,
                sarn,
                sarn - raw,
                torch.abs(sarn - raw),
                raw * sarn,
                geometry_features.float(),
                scalar,
            ),
            dim=1,
        )
        _require(
            relation.shape == (batch, self.relation_features),
            "risk arbitration relation features differ",
        )
        logit_delta = self.router(relation)
        weights = torch.softmax(self.prior_logits[None] + logit_delta, dim=1)
        return weights, logit_delta


class RiskArbitratedRelativeContextMomentTransport(RelativeContextMomentTransport):
    """Fuse three risk-specialized shifts, then transport one full posterior."""

    def __init__(
        self,
        base_correction: nn.Module,
        *,
        progress_bins: int,
        expert_latent_features: int,
        expert_hidden_features: int,
        max_progress_shift: float,
        gate_latent_features: int = 32,
        gate_hidden_features: int = 64,
        prior_weights: Sequence[float] = DEFAULT_PRIOR_WEIGHTS,
        gate_representation_enabled: bool = True,
        adaptive_strength: float = 1.0,
        fusion_mode: str = "single_exact_transport",
    ) -> None:
        super().__init__(
            base_correction,
            progress_bins=int(progress_bins),
            latent_features=int(expert_latent_features),
            hidden_features=int(expert_hidden_features),
            max_progress_shift=float(max_progress_shift),
        )
        mean_risk_head = self.context_head
        del self.context_head
        self.mean_risk_head = mean_risk_head
        self.tail_risk_head = RelativeContextMomentHead(
            latent_features=int(expert_latent_features),
            hidden_features=int(expert_hidden_features),
            max_progress_shift=float(max_progress_shift),
        )
        self.combined_risk_head = RelativeContextMomentHead(
            latent_features=int(expert_latent_features),
            hidden_features=int(expert_hidden_features),
            max_progress_shift=float(max_progress_shift),
        )
        self.risk_gate = RepresentationConditionedRiskGate(
            latent_features=int(gate_latent_features),
            hidden_features=int(gate_hidden_features),
            prior_weights=prior_weights,
            representation_enabled=bool(gate_representation_enabled),
        )
        _require(
            0.0 <= float(adaptive_strength) <= 1.0
            and fusion_mode in {"single_exact_transport", "posterior_mixture"},
            "risk arbitration fusion configuration differs",
        )
        self.adaptive_strength = float(adaptive_strength)
        self.fusion_mode = str(fusion_mode)

    @property
    def risk_heads(self) -> tuple[RelativeContextMomentHead, ...]:
        return (
            self.mean_risk_head,
            self.tail_risk_head,
            self.combined_risk_head,
        )

    def forward(
        self,
        raw_posterior: torch.Tensor,
        raw_encoder_features: Mapping[str, torch.Tensor],
        sarn_posterior: torch.Tensor,
        sarn_encoder_features: Mapping[str, torch.Tensor],
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        base = self.base_correction(
            raw_posterior,
            raw_encoder_features,
            sarn_posterior,
            sarn_encoder_features,
            sarn_support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        raw_representation = raw_encoder_features.get("representation")
        sarn_representation = sarn_encoder_features.get("representation")
        geometry = base.get("geometry_token")
        _require(
            isinstance(raw_representation, torch.Tensor)
            and isinstance(sarn_representation, torch.Tensor)
            and isinstance(geometry, Mapping)
            and isinstance(geometry.get("features"), torch.Tensor),
            "risk arbitration context inputs are unavailable",
        )
        available = base["relation_available"].bool()
        base_mean = base["mean"].float()
        head_arguments = (
            raw_representation,
            sarn_representation,
            geometry["features"],
            base["raw_anchor_mean"],
            base["sarn_endpoint_mean"],
            base_mean,
        )
        expert_shifts = torch.stack(
            [head(*head_arguments) for head in self.risk_heads], dim=1
        )
        adaptive_weights, logit_delta = self.risk_gate(
            *head_arguments[:3],
            base["raw_anchor_mean"],
            base["sarn_endpoint_mean"],
            base_mean,
            expert_shifts,
        )
        prior_weights = self.risk_gate.prior_weights[None].to(
            adaptive_weights
        )
        weights = prior_weights + self.adaptive_strength * (
            adaptive_weights - prior_weights
        )
        shift = (weights * expert_shifts).sum(dim=1)
        applied_shift = torch.where(available, shift, torch.zeros_like(shift))
        epsilon = torch.finfo(torch.float32).eps
        target_mean = (base_mean + applied_shift).clamp(epsilon, 1.0 - epsilon)
        base_posterior = base["progress_posterior"].float()
        changed = available & (applied_shift != 0.0)
        if self.fusion_mode == "single_exact_transport":
            proposed, proposed_variance = self._transport(
                base_posterior, target_mean
            )
            zero_shift = applied_shift == 0.0
            posterior_candidate = torch.where(
                zero_shift[:, None],
                base_posterior + proposed - proposed.detach(),
                proposed,
            )
            mean_candidate = torch.where(
                zero_shift,
                base_mean + target_mean - target_mean.detach(),
                target_mean,
            )
            variance_candidate = torch.where(
                zero_shift,
                base["variance"].float()
                + proposed_variance
                - proposed_variance.detach(),
                proposed_variance,
            )
        else:
            expert_target_means = (
                base_mean[:, None] + expert_shifts
            ).clamp(epsilon, 1.0 - epsilon)
            expert_posteriors = torch.stack(
                [
                    self._transport(base_posterior, expert_target_means[:, index])[0]
                    for index in range(len(RISK_HEAD_NAMES))
                ],
                dim=1,
            )
            mixed = (weights[:, :, None] * expert_posteriors).sum(dim=1)
            all_zero = (expert_shifts == 0.0).all(dim=1)
            posterior_candidate = torch.where(
                all_zero[:, None],
                base_posterior + mixed - mixed.detach(),
                mixed,
            )
            grid = self.progress_grid.float()[None]
            mixed_mean = (posterior_candidate * grid).sum(dim=1)
            mixed_variance = (
                posterior_candidate
                * (grid - mixed_mean[:, None]).square()
            ).sum(dim=1)
            mean_candidate = mixed_mean
            variance_candidate = mixed_variance
        posterior = torch.where(
            available[:, None], posterior_candidate, base_posterior
        )
        mean = torch.where(available, mean_candidate, base_mean)
        variance = torch.where(
            available, variance_candidate, base["variance"].float()
        )
        base_cdf = base_posterior.cumsum(dim=1)
        final_cdf = posterior.cumsum(dim=1)
        path_times = torch.arange(
            1,
            BRIDGE_LAYERS + 1,
            dtype=torch.float32,
            device=posterior.device,
        ) / float(BRIDGE_LAYERS)
        layer_cdfs = base_cdf[:, None] + path_times[None, :, None] * (
            final_cdf[:, None] - base_cdf[:, None]
        )
        result = dict(base)
        result.update(
            {
                "architecture": ARCHITECTURE,
                "progress_posterior": posterior,
                "progress_cdf": final_cdf,
                "mean": mean,
                "variance": variance,
                "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
                "relative_context_shift": applied_shift,
                "relative_context_target_mean": target_mean,
                "relative_context_active": changed,
                "relative_context_base_mean": base_mean,
                "relative_context_base_posterior": base_posterior,
                "relative_context_layer_cdfs": layer_cdfs,
                "risk_arbitration_weights": weights,
                "risk_arbitration_adaptive_weights": adaptive_weights,
                "risk_arbitration_logit_delta": logit_delta,
                "risk_head_shifts": expert_shifts,
                "risk_arbitration_adaptive_strength": self.adaptive_strength,
                "risk_arbitration_fusion_mode": self.fusion_mode,
                "publication_model": publication_model_identity(),
            }
        )
        physical = dict(base.get("physical_outputs", {}))
        physical["progress_mean"] = mean
        physical["progress_variance"] = variance
        result["physical_outputs"] = physical
        return result


def parameter_counts(
    model: RiskArbitratedRelativeContextMomentTransport,
) -> dict[str, int]:
    return {
        "mean_risk_head": sum(p.numel() for p in model.mean_risk_head.parameters()),
        "tail_risk_head": sum(p.numel() for p in model.tail_risk_head.parameters()),
        "combined_risk_head": sum(
            p.numel() for p in model.combined_risk_head.parameters()
        ),
        "risk_gate": sum(p.numel() for p in model.risk_gate.parameters()),
        "additional_image_encoders": 0,
    }


# Clean paper-facing aliases. The longer class names remain importable because
# released checkpoints and internal experiment ledgers use them.
R2MTNet = RiskArbitratedRelativeContextMomentTransport
R2MTRouter = RepresentationConditionedRiskGate
r2mt_parameter_counts = parameter_counts


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_PRIOR_WEIGHTS",
    "PUBLICATION_FULL_NAME",
    "PUBLICATION_NAME",
    "R2MTNet",
    "R2MTRouter",
    "RISK_HEAD_NAMES",
    "RepresentationConditionedRiskGate",
    "RiskArbitratedRelativeContextMomentTransport",
    "parameter_counts",
    "publication_model_identity",
    "r2mt_parameter_counts",
]
