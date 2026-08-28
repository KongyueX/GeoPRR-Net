"""PCRT-Meter: polar evidence with conditional-regret transport.

The model keeps the frozen ReMSTNet-v3 EfficientNet-B0 foundation, restores
the completed ReMSTNet-R2MT expert bank, adds one light polar-ray head over
the *same* stride-8/stride-16 features, and predicts the conditional benefit
of the polar and R2MT actions relative to the ReMST identity action.

The conditional-regret router addresses one measured failure mode: the R2MT
transport improves the synthetic projective conditions but can harm individual
real photographs.  Repository/version metadata and ordinary unit tests cannot
identify which inference action lowers an unseen sample's reading loss, so the
choice is learned from a disjoint SyncG inner-development partition.  It is a
model component, not a release or data-access gate.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.a15_fteb import GEOMETRY_FEATURES
from experiments.raw_angular_moment_refiner_probe import AngularMomentLogitRefiner
from experiments.train_remstnet import load_remstnet_checkpoint
from remstnet.model import _moment_exact_transport, _posterior_moments


ARCHITECTURE: Final[str] = (
    "PCRT-Meter-Polar-Conditional-Regret-Moment-Transport-v1"
)
PUBLICATION_NAME: Final[str] = "PCRT-Meter"
REPRESENTATION_FEATURES: Final[int] = 1280
RISK_HEAD_NAMES: Final[tuple[str, ...]] = (
    "mean_risk",
    "tail_risk",
    "combined_risk",
)
EXPERT_NAMES: Final[tuple[str, ...]] = ("identity", "polar", "r2mt")
DEFAULT_REGRET_PRIOR: Final[tuple[float, ...]] = (0.50, 0.25, 0.25)
FUNCTIONAL_VARIANTS: Final[tuple[str, ...]] = (
    "full",
    "backbone_only",
    "no_polar_evidence",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class RelativeContextMomentHead1280(nn.Module):
    """Exact 1280-D reconstruction of one completed R2MT risk head."""

    def __init__(
        self,
        *,
        latent_features: int,
        hidden_features: int,
        max_progress_shift: float,
    ) -> None:
        super().__init__()
        self.latent_features = int(latent_features)
        self.hidden_features = int(hidden_features)
        self.max_progress_shift = float(max_progress_shift)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(REPRESENTATION_FEATURES),
            nn.Linear(REPRESENTATION_FEATURES, self.latent_features),
            nn.GELU(),
        )
        self.scalar_features = 5
        self.relation_features = (
            5 * self.latent_features + GEOMETRY_FEATURES + self.scalar_features
        )
        self.residual_network = nn.Sequential(
            nn.LayerNorm(self.relation_features),
            nn.Linear(self.relation_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, 1),
        )

    def forward(
        self,
        raw_representation: torch.Tensor,
        sarn_representation: torch.Tensor,
        geometry_features: torch.Tensor,
        raw_mean: torch.Tensor,
        sarn_mean: torch.Tensor,
        base_mean: torch.Tensor,
    ) -> torch.Tensor:
        batch = int(raw_representation.shape[0])
        _require(
            raw_representation.shape
            == sarn_representation.shape
            == (batch, REPRESENTATION_FEATURES),
            "R2MT representation shape differs",
        )
        raw = self.shared_projection(raw_representation.float())
        sarn = self.shared_projection(sarn_representation.float())
        scalars = torch.stack(
            (
                raw_mean.float(),
                sarn_mean.float(),
                base_mean.float(),
                sarn_mean.float() - raw_mean.float(),
                base_mean.float() - sarn_mean.float(),
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
                scalars,
            ),
            dim=1,
        )
        logit = self.residual_network(relation).squeeze(1)
        return self.max_progress_shift * torch.tanh(logit)


class RepresentationRiskRouter1280(nn.Module):
    """Exact 1280-D reconstruction of the completed R2MT risk router."""

    def __init__(
        self,
        *,
        latent_features: int,
        hidden_features: int,
        prior_weights: Sequence[float],
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(tuple(prior_weights), dtype=torch.float32)
        prior = prior / prior.sum()
        self.latent_features = int(latent_features)
        self.hidden_features = int(hidden_features)
        self.scalar_features = 10
        self.relation_features = (
            5 * self.latent_features + GEOMETRY_FEATURES + self.scalar_features
        )
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(REPRESENTATION_FEATURES),
            nn.Linear(REPRESENTATION_FEATURES, self.latent_features),
            nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(self.relation_features),
            nn.Linear(self.relation_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, len(RISK_HEAD_NAMES)),
        )
        self.register_buffer("prior_weights", prior)
        self.register_buffer("prior_logits", prior.log())

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
        raw = self.shared_projection(raw_representation.float())
        sarn = self.shared_projection(sarn_representation.float())
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
        logit_delta = self.router(relation)
        weights = torch.softmax(self.prior_logits[None] + logit_delta, dim=1)
        return weights, logit_delta


class ReMSTR2MTExpertBank(nn.Module):
    """Frozen ReMST identity and completed R2MT transport experts."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        progress_bins: int,
        latent_features: int,
        hidden_features: int,
        max_progress_shift: float,
        gate_latent_features: int,
        gate_hidden_features: int,
        prior_weights: Sequence[float],
        adaptive_strength: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.risk_heads = nn.ModuleDict(
            {
                name: RelativeContextMomentHead1280(
                    latent_features=latent_features,
                    hidden_features=hidden_features,
                    max_progress_shift=max_progress_shift,
                )
                for name in RISK_HEAD_NAMES
            }
        )
        self.risk_router = RepresentationRiskRouter1280(
            latent_features=gate_latent_features,
            hidden_features=gate_hidden_features,
            prior_weights=prior_weights,
        )
        self.adaptive_strength = float(adaptive_strength)
        self.progress_bins = int(progress_bins)
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, self.progress_bins, dtype=torch.float32),
        )

    def _base_forward_with_features(
        self,
        raw_view: torch.Tensor,
        sarn_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        captured: dict[str, list[Any]] = {
            "scale8": [],
            "scale16": [],
            "context": [],
            "geometry": [],
        }

        def capture(name: str):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                captured[name].append(output)

            return hook

        handles = (
            self.base_model.shared_scale8_encoder.register_forward_hook(
                capture("scale8")
            ),
            self.base_model.shared_scale16_encoder.register_forward_hook(
                capture("scale16")
            ),
            self.base_model.shared_context_encoder.register_forward_hook(
                capture("context")
            ),
            self.base_model.moment_coordinator.geometry_encoder.register_forward_hook(
                capture("geometry")
            ),
        )
        try:
            base = self.base_model(
                raw_view,
                sarn_view,
                support_mask,
                sarn_active=sarn_active,
                raw_to_sarn_homography=raw_to_sarn_homography,
            )
        finally:
            for handle in handles:
                handle.remove()
        _require(
            len(captured["scale8"]) >= 2
            and len(captured["scale16"]) >= 2
            and len(captured["context"]) >= 2
            and len(captured["geometry"]) == 1,
            "ReMST feature capture differs",
        )
        geometry = captured["geometry"][0]
        _require(
            isinstance(geometry, Mapping)
            and isinstance(geometry.get("features"), torch.Tensor),
            "ReMST geometry features are unavailable",
        )
        raw_context = captured["context"][0]
        sarn_context = captured["context"][1]
        features = {
            "raw_stride8": captured["scale8"][0],
            "raw_stride16": captured["scale16"][0],
            "raw_representation": raw_context.mean(dim=(2, 3)),
            "sarn_representation": sarn_context.mean(dim=(2, 3)),
            "geometry_features": geometry["features"],
        }
        return base, features

    def forward_experts(
        self,
        raw_view: torch.Tensor,
        sarn_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        base, features = self._base_forward_with_features(
            raw_view,
            sarn_view,
            support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        arguments = (
            features["raw_representation"],
            features["sarn_representation"],
            features["geometry_features"],
            base["raw_anchor_mean"],
            base["sarn_endpoint_mean"],
            base["mean"],
        )
        shifts = torch.stack(
            tuple(self.risk_heads[name](*arguments) for name in RISK_HEAD_NAMES),
            dim=1,
        )
        adaptive, logit_delta = self.risk_router(*arguments, shifts)
        prior = self.risk_router.prior_weights[None].to(adaptive)
        weights = prior + self.adaptive_strength * (adaptive - prior)
        proposed_shift = (weights * shifts).sum(dim=1)
        available = base["relation_available"].bool()
        applied_shift = torch.where(
            available, proposed_shift, torch.zeros_like(proposed_shift)
        )
        base_mean = base["mean"].float()
        target_mean = (base_mean + applied_shift).clamp(
            torch.finfo(torch.float32).eps,
            1.0 - torch.finfo(torch.float32).eps,
        )
        transported = _moment_exact_transport(
            base["progress_posterior"], target_mean, self.progress_grid
        )
        posterior = torch.where(
            available[:, None], transported["posterior"], base["progress_posterior"]
        )
        mean, variance = _posterior_moments(posterior)
        result = dict(base)
        result.update(
            {
                "architecture": "ReMSTNet-R2MT-reconstructed-v1",
                "progress_posterior": posterior,
                "progress_cdf": posterior.cumsum(dim=1),
                "mean": mean,
                "variance": variance,
                "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
                "relative_context_shift": applied_shift,
                "relative_context_target_mean": target_mean,
                "risk_head_shifts": shifts,
                "risk_arbitration_weights": weights,
                "risk_arbitration_adaptive_weights": adaptive,
                "risk_arbitration_logit_delta": logit_delta,
                "r2mt_base_mean": base_mean,
                "r2mt_base_posterior": base["progress_posterior"],
            }
        )
        return result, features

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward_experts(*args, **kwargs)[0]


class ConditionalRegretRouter(nn.Module):
    """Predict per-sample action benefit, then form a conservative simplex."""

    def __init__(
        self,
        *,
        latent_features: int = 32,
        hidden_features: int = 96,
        maximum_gain: float = 0.10,
        gain_logit_scale: float = 100.0,
        prior_weights: Sequence[float] = DEFAULT_REGRET_PRIOR,
    ) -> None:
        super().__init__()
        prior = torch.as_tensor(tuple(prior_weights), dtype=torch.float32)
        _require(prior.numel() == len(EXPERT_NAMES), "PCRT prior size differs")
        prior = prior / prior.sum()
        self.latent_features = int(latent_features)
        self.hidden_features = int(hidden_features)
        self.maximum_gain = float(maximum_gain)
        self.gain_logit_scale = float(gain_logit_scale)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(REPRESENTATION_FEATURES),
            nn.Linear(REPRESENTATION_FEATURES, self.latent_features),
            nn.GELU(),
        )
        self.scalar_features = 11
        self.polar_features = 36
        self.relation_features = (
            5 * self.latent_features
            + GEOMETRY_FEATURES
            + self.polar_features
            + self.scalar_features
        )
        self.regret_network = nn.Sequential(
            nn.LayerNorm(self.relation_features),
            nn.Linear(self.relation_features, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, 2),
        )
        final = self.regret_network[-1]
        _require(isinstance(final, nn.Linear), "PCRT regret output differs")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("prior_weights", prior)
        self.register_buffer("prior_logits", prior.log())

    def forward(
        self,
        raw_representation: torch.Tensor,
        sarn_representation: torch.Tensor,
        geometry_features: torch.Tensor,
        base_mean: torch.Tensor,
        polar_mean: torch.Tensor,
        r2mt_mean: torch.Tensor,
        raw_mean: torch.Tensor,
        sarn_mean: torch.Tensor,
        relation_available: torch.Tensor,
        polar_posterior: torch.Tensor,
        polar_concentration: torch.Tensor,
        polar_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.shared_projection(raw_representation.float())
        sarn = self.shared_projection(sarn_representation.float())
        scalar = torch.stack(
            (
                base_mean.float(),
                polar_mean.float(),
                r2mt_mean.float(),
                polar_mean.float() - base_mean.float(),
                r2mt_mean.float() - base_mean.float(),
                polar_mean.float() - r2mt_mean.float(),
                raw_mean.float(),
                sarn_mean.float(),
                relation_available.float(),
                polar_concentration.float(),
                polar_entropy.float(),
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
                polar_posterior.float(),
                scalar,
            ),
            dim=1,
        )
        gains = self.maximum_gain * torch.tanh(self.regret_network(relation))
        all_gains = torch.cat((torch.zeros_like(gains[:, :1]), gains), dim=1)
        logits = self.prior_logits[None] + self.gain_logit_scale * all_gains
        weights = torch.softmax(logits, dim=1)
        return weights, gains, logits


class PCRTMeter(nn.Module):
    """Three-action, one-backbone PCRT-Meter inference graph."""

    def __init__(
        self,
        expert_bank: ReMSTR2MTExpertBank,
        polar_expert: AngularMomentLogitRefiner,
        regret_router: ConditionalRegretRouter,
        *,
        functional_variant: str = "full",
    ) -> None:
        super().__init__()
        _require(
            functional_variant in FUNCTIONAL_VARIANTS,
            f"unknown PCRT functional variant: {functional_variant}",
        )
        self.expert_bank = expert_bank
        self.polar_expert = polar_expert
        self.regret_router = regret_router
        self.functional_variant = str(functional_variant)

    @staticmethod
    def _neutral_polar_evidence(
        base_mean: torch.Tensor,
        *,
        angular_bins: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Replace the polar branch by label-free neutral evidence.

        The router shape and parameter count stay unchanged.  The polar action
        becomes an exact copy of the identity action, while the directional
        posterior carries no preferred angle.
        """
        batch = int(base_mean.shape[0])
        posterior = torch.full(
            (batch, int(angular_bins)),
            1.0 / float(angular_bins),
            dtype=base_mean.dtype,
            device=base_mean.device,
        )
        return base_mean, {
            "direction_posterior": posterior,
            "direction_sin_cos": torch.zeros(
                (batch, 2), dtype=base_mean.dtype, device=base_mean.device
            ),
            "direction_concentration": torch.zeros_like(base_mean),
            "direction_entropy": torch.ones_like(base_mean),
        }

    def _polar_prediction(
        self,
        features: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        details = self.polar_expert(
            features["raw_stride8"],
            features["raw_stride16"],
            features["raw_representation"],
        )
        point = self.expert_bank.base_model.moment_readout.point_projection
        raw_logit = F.linear(
            features["raw_representation"].float(),
            point.weight.detach().float(),
            None if point.bias is None else point.bias.detach().float(),
        ).squeeze(1)
        prediction = torch.sigmoid(raw_logit + details["logit_delta"].float())
        return prediction, details

    def forward_experts(
        self,
        raw_view: torch.Tensor,
        sarn_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        r2mt, features = self.expert_bank.forward_experts(
            raw_view,
            sarn_view,
            support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )
        base_mean = r2mt["r2mt_base_mean"].float()
        r2mt_mean = r2mt["mean"].float()
        if self.functional_variant in ("backbone_only", "no_polar_evidence"):
            polar_mean, polar = self._neutral_polar_evidence(
                base_mean,
                angular_bins=self.polar_expert.angular_bins,
            )
        else:
            polar_mean, polar = self._polar_prediction(features)
        candidates = torch.stack((base_mean, polar_mean, r2mt_mean), dim=1)
        if self.functional_variant == "backbone_only":
            weights = torch.zeros_like(candidates)
            weights[:, 0] = 1.0
            gains = torch.zeros_like(candidates[:, 1:])
            logits = torch.full_like(candidates, float("-inf"))
            logits[:, 0] = 0.0
        else:
            weights, gains, logits = self.regret_router(
                features["raw_representation"],
                features["sarn_representation"],
                features["geometry_features"],
                base_mean,
                polar_mean,
                r2mt_mean,
                r2mt["raw_anchor_mean"],
                r2mt["sarn_endpoint_mean"],
                r2mt["relation_available"],
                polar["direction_posterior"],
                polar["direction_concentration"],
                polar["direction_entropy"],
            )
        target_mean = (weights * candidates).sum(dim=1).clamp(
            torch.finfo(torch.float32).eps,
            1.0 - torch.finfo(torch.float32).eps,
        )
        transported = _moment_exact_transport(
            r2mt["r2mt_base_posterior"],
            target_mean,
            self.expert_bank.progress_grid,
        )
        posterior = transported["posterior"]
        mean, variance = _posterior_moments(posterior)
        result = dict(r2mt)
        result.update(
            {
                "architecture": ARCHITECTURE,
                "progress_posterior": posterior,
                "progress_cdf": posterior.cumsum(dim=1),
                "mean": mean,
                "variance": variance,
                "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
                "pcrt_expert_predictions": candidates,
                "pcrt_regret_weights": weights,
                "pcrt_predicted_gains": gains,
                "pcrt_regret_logits": logits,
                "pcrt_polar_direction_posterior": polar["direction_posterior"],
                "pcrt_polar_direction_sin_cos": polar["direction_sin_cos"],
                "pcrt_polar_concentration": polar["direction_concentration"],
                "pcrt_polar_entropy": polar["direction_entropy"],
            }
        )
        physical = dict(result.get("physical_outputs", {}))
        physical["progress_mean"] = mean
        physical["progress_variance"] = variance
        result["physical_outputs"] = physical
        return result, features

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward_experts(*args, **kwargs)[0]


def load_remst_r2mt_expert_bank(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[ReMSTR2MTExpertBank, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "ReMST-R2MT checkpoint is malformed")
    construction = payload.get("construction")
    _require(isinstance(construction, Mapping), "ReMST-R2MT construction missing")
    base, base_metadata = load_remstnet_checkpoint(
        Path(str(payload["source_checkpoint"])), device=device
    )
    bank = ReMSTR2MTExpertBank(
        base,
        progress_bins=int(construction["progress_bins"]),
        latent_features=int(construction["expert_latent_features"]),
        hidden_features=int(construction["expert_hidden_features"]),
        max_progress_shift=float(construction["max_progress_shift"]),
        gate_latent_features=int(construction["gate_latent_features"]),
        gate_hidden_features=int(construction["gate_hidden_features"]),
        prior_weights=construction["prior_weights"],
        adaptive_strength=float(construction["adaptive_strength"]),
    )
    head_states = payload.get("risk_head_states")
    _require(isinstance(head_states, Mapping), "ReMST-R2MT head states missing")
    for name in RISK_HEAD_NAMES:
        bank.risk_heads[name].load_state_dict(head_states[name], strict=True)
    bank.risk_router.load_state_dict(payload["risk_gate_state"], strict=True)
    bank.to(torch.device(device)).eval()
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    return bank, {
        "checkpoint": str(source),
        "protocol": str(payload.get("protocol")),
        "seed": int(payload.get("seed", -1)),
        "source_checkpoint": str(payload["source_checkpoint"]),
        "source_metadata": base_metadata,
        "construction": dict(construction),
    }


def load_pcrt_meter_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[PCRTMeter, dict[str, Any]]:
    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "PCRT checkpoint is malformed")
    _require(payload.get("architecture") == ARCHITECTURE, "PCRT architecture differs")
    construction = payload.get("construction")
    _require(isinstance(construction, Mapping), "PCRT construction missing")
    bank, bank_metadata = load_remst_r2mt_expert_bank(
        Path(str(payload["source_r2mt_checkpoint"])), device=device
    )
    polar = AngularMomentLogitRefiner()
    polar.load_state_dict(payload["polar_expert_state"], strict=True)
    router = ConditionalRegretRouter(
        latent_features=int(construction["router_latent_features"]),
        hidden_features=int(construction["router_hidden_features"]),
        maximum_gain=float(construction["maximum_gain"]),
        gain_logit_scale=float(construction["gain_logit_scale"]),
        prior_weights=construction["prior_weights"],
    )
    router.load_state_dict(payload["regret_router_state"], strict=True)
    functional_variant = str(payload.get("functional_variant") or "full")
    model = PCRTMeter(
        bank,
        polar,
        router,
        functional_variant=functional_variant,
    ).to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_seed = int(bank_metadata["source_metadata"]["source_foundation"]["source_seed"])
    metadata = {
        "checkpoint": str(source),
        "architecture": ARCHITECTURE,
        "protocol": str(payload.get("protocol")),
        "seed": int(payload.get("seed", -1)),
        "source_seed": source_seed,
        "source_r2mt": bank_metadata,
        "construction": dict(construction),
        "training_scope": dict(payload.get("training_scope", {})),
        "checkpoint_selection": str(payload.get("checkpoint_selection")),
        "parameter_counts": dict(payload.get("parameter_counts", {})),
        "functional_variant": functional_variant,
        "controlled_ablation": dict(payload.get("controlled_ablation", {})),
    }
    return model, metadata


def parameter_counts(model: PCRTMeter) -> dict[str, int]:
    return {
        "shared_remst_r2mt": sum(
            parameter.numel() for parameter in model.expert_bank.parameters()
        ),
        "polar_expert": sum(
            parameter.numel() for parameter in model.polar_expert.parameters()
        ),
        "regret_router": sum(
            parameter.numel() for parameter in model.regret_router.parameters()
        ),
        "additional_image_encoders": 0,
    }


__all__ = [
    "ARCHITECTURE",
    "EXPERT_NAMES",
    "FUNCTIONAL_VARIANTS",
    "PCRTMeter",
    "ConditionalRegretRouter",
    "ReMSTR2MTExpertBank",
    "load_pcrt_meter_checkpoint",
    "load_remst_r2mt_expert_bank",
    "parameter_counts",
]
