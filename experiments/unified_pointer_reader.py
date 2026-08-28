"""Publication-facing unified pointer reader and compact structural ablations.

The model has one shared dual-observation image encoder and three reading
candidates: the geometry-aware base estimate, polar evidence, and relational
moment transport.  A conditional-regret router combines the candidates before
an exact first-moment readout.  The implementation reuses the already tested
low-level research modules, but exposes one model identity to the paper.

The four ablations are executable interventions rather than display aliases:

``no_geometry_fusion``
    Replace the geometry-aware base fusion by an availability-aware fixed mean
    of the raw and normalized-view endpoints.  The relational residual is
    translated onto that base so the separate transport module remains present.
``no_polar_evidence``
    Skip the polar head and renormalize routing over base and transport.
``no_relational_transport``
    Skip the relational risk heads and renormalize routing over base and polar.
``fixed_routing``
    Keep all candidates but use the fixed prior simplex instead of the learned
    conditional router.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.pcrt_meter import (
    ConditionalRegretRouter,
    ReMSTR2MTExpertBank,
    _moment_exact_transport,
    _posterior_moments,
    load_remst_r2mt_expert_bank,
)
from experiments.raw_angular_moment_refiner_probe import AngularMomentLogitRefiner


PROTOCOL: Final[str] = "unified_pointer_reader_training_v1"
ARCHITECTURE: Final[str] = (
    "Shared-Dual-View-Geometry-Polar-Relational-Conditional-Moment-Reader-v1"
)
PUBLICATION_NAME: Final[str] = "GeoPRR-Net"
CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "base",
    "polar_evidence",
    "relational_transport",
)
FULL: Final[str] = "full"
NO_GEOMETRY_FUSION: Final[str] = "no_geometry_fusion"
NO_POLAR_EVIDENCE: Final[str] = "no_polar_evidence"
NO_RELATIONAL_TRANSPORT: Final[str] = "no_relational_transport"
FIXED_ROUTING: Final[str] = "fixed_routing"
VARIANTS: Final[tuple[str, ...]] = (
    FULL,
    NO_GEOMETRY_FUSION,
    NO_POLAR_EVIDENCE,
    NO_RELATIONAL_TRANSPORT,
    FIXED_ROUTING,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def candidate_mask(variant: str, *, device: torch.device) -> torch.Tensor:
    """Return the active three-candidate mask for one structural variant."""

    _require(variant in VARIANTS, f"unknown unified-reader variant: {variant}")
    active = torch.ones(len(CANDIDATE_NAMES), dtype=torch.bool, device=device)
    if variant == NO_POLAR_EVIDENCE:
        active[1] = False
    elif variant == NO_RELATIONAL_TRANSPORT:
        active[2] = False
    return active


def fixed_geometry_base(
    raw_mean: torch.Tensor,
    normalized_mean: torch.Tensor,
    relation_available: torch.Tensor,
) -> torch.Tensor:
    """Availability-aware, label-free fixed fusion used by the geometry ablation."""

    _require(
        raw_mean.shape == normalized_mean.shape == relation_available.shape,
        "fixed-fusion inputs differ",
    )
    average = 0.5 * (raw_mean.float() + normalized_mean.float())
    return torch.where(relation_available.bool(), average, raw_mean.float())


def mask_router_logits(
    logits: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Renormalize the learned router on exactly the candidates still present."""

    _require(logits.ndim == 2, "router logits must be BC")
    _require(
        active.shape == (logits.shape[1],) and active.dtype == torch.bool,
        "candidate mask differs",
    )
    _require(bool(active[0]) and bool(active.any()), "base candidate must remain active")
    masked = logits.masked_fill(~active[None], -torch.inf)
    return torch.softmax(masked, dim=1)


class UnifiedPointerReader(nn.Module):
    """One non-distilled model with three candidates and moment-consistent output."""

    def __init__(
        self,
        expert_bank: ReMSTR2MTExpertBank,
        polar_expert: AngularMomentLogitRefiner,
        regret_router: ConditionalRegretRouter,
        *,
        variant: str = FULL,
    ) -> None:
        super().__init__()
        _require(variant in VARIANTS, f"unknown unified-reader variant: {variant}")
        self.expert_bank = expert_bank
        self.polar_expert = polar_expert
        self.regret_router = regret_router
        self.variant = str(variant)

    @staticmethod
    def _neutral_polar(
        base_mean: torch.Tensor,
        *,
        angular_bins: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

    def _base_and_features(
        self,
        raw_view: torch.Tensor,
        normalized_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        normalized_active: torch.Tensor,
        raw_to_normalized_homography: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], bool]:
        if self.variant == NO_RELATIONAL_TRANSPORT:
            base, features = self.expert_bank._base_forward_with_features(
                raw_view,
                normalized_view,
                support_mask,
                sarn_active=normalized_active,
                raw_to_sarn_homography=raw_to_normalized_homography,
            )
            return dict(base), features, False
        transported, features = self.expert_bank.forward_experts(
            raw_view,
            normalized_view,
            support_mask,
            sarn_active=normalized_active,
            raw_to_sarn_homography=raw_to_normalized_homography,
        )
        return dict(transported), features, True

    def forward_experts(
        self,
        raw_view: torch.Tensor,
        normalized_view: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        result, features, transport_executed = self._base_and_features(
            raw_view,
            normalized_view,
            support_mask,
            normalized_active=sarn_active,
            raw_to_normalized_homography=raw_to_sarn_homography,
        )
        raw_mean = result["raw_anchor_mean"].float()
        normalized_mean = result["sarn_endpoint_mean"].float()
        relation_available = result["relation_available"].bool()

        if transport_executed:
            original_base_mean = result["r2mt_base_mean"].float()
            original_base_posterior = result["r2mt_base_posterior"]
            relational_mean = result["mean"].float()
        else:
            original_base_mean = result["mean"].float()
            original_base_posterior = result["progress_posterior"]
            relational_mean = original_base_mean

        base_mean = original_base_mean
        base_posterior = original_base_posterior
        if self.variant == NO_GEOMETRY_FUSION:
            base_mean = fixed_geometry_base(
                raw_mean,
                normalized_mean,
                relation_available,
            ).clamp(
                torch.finfo(torch.float32).eps,
                1.0 - torch.finfo(torch.float32).eps,
            )
            base_posterior = _moment_exact_transport(
                original_base_posterior,
                base_mean,
                self.expert_bank.progress_grid,
            )["posterior"]
            relational_mean = (
                base_mean + (relational_mean - original_base_mean)
            ).clamp(
                torch.finfo(torch.float32).eps,
                1.0 - torch.finfo(torch.float32).eps,
            )

        if self.variant == NO_POLAR_EVIDENCE:
            polar_mean, polar = self._neutral_polar(
                base_mean,
                angular_bins=self.polar_expert.angular_bins,
            )
        else:
            polar_mean, polar = self._polar_prediction(features)

        if self.variant == NO_RELATIONAL_TRANSPORT:
            relational_mean = base_mean

        candidates = torch.stack((base_mean, polar_mean, relational_mean), dim=1)
        active = candidate_mask(self.variant, device=candidates.device)
        if self.variant == FIXED_ROUTING:
            prior = self.regret_router.prior_weights.to(candidates)
            prior = prior * active.to(prior.dtype)
            prior = prior / prior.sum()
            weights = prior[None].expand_as(candidates)
            gains = torch.zeros_like(candidates[:, 1:])
            logits = torch.log(prior.clamp_min(torch.finfo(prior.dtype).tiny))[None]
            logits = logits.expand_as(candidates)
        else:
            _unmasked_weights, gains, logits = self.regret_router(
                features["raw_representation"],
                features["sarn_representation"],
                features["geometry_features"],
                base_mean,
                polar_mean,
                relational_mean,
                raw_mean,
                normalized_mean,
                relation_available,
                polar["direction_posterior"],
                polar["direction_concentration"],
                polar["direction_entropy"],
            )
            weights = mask_router_logits(logits, active)

        target_mean = (weights * candidates).sum(dim=1).clamp(
            torch.finfo(torch.float32).eps,
            1.0 - torch.finfo(torch.float32).eps,
        )
        posterior = _moment_exact_transport(
            base_posterior,
            target_mean,
            self.expert_bank.progress_grid,
        )["posterior"]
        mean, variance = _posterior_moments(posterior)
        result.update(
            {
                "architecture": ARCHITECTURE,
                "architecture_variant": self.variant,
                "progress_posterior": posterior,
                "progress_cdf": posterior.cumsum(dim=1),
                "mean": mean,
                "variance": variance,
                "standard_deviation": torch.sqrt(variance.clamp_min(0.0)),
                "candidate_names": CANDIDATE_NAMES,
                "candidate_active": active[None].expand_as(candidates),
                "candidate_predictions": candidates,
                "routing_weights": weights,
                "predicted_candidate_gains": gains,
                "routing_logits": logits,
                "polar_direction_posterior": polar["direction_posterior"],
                "polar_direction_sin_cos": polar["direction_sin_cos"],
                "polar_concentration": polar["direction_concentration"],
                "polar_entropy": polar["direction_entropy"],
                "relational_transport_executed": torch.full_like(
                    relation_available, transport_executed
                ),
            }
        )
        physical = dict(result.get("physical_outputs", {}))
        physical["progress_mean"] = mean
        physical["progress_variance"] = variance
        result["physical_outputs"] = physical
        return result, features

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.forward_experts(*args, **kwargs)[0]


def parameter_inventory(model: UnifiedPointerReader) -> dict[str, Any]:
    """Return stored and actually active parameter counts for a variant."""

    components = {
        "shared_dual_view_geometry_core": sum(
            parameter.numel() for parameter in model.expert_bank.base_model.parameters()
        ),
        "relational_transport": sum(
            parameter.numel()
            for module in (model.expert_bank.risk_heads, model.expert_bank.risk_router)
            for parameter in module.parameters()
        ),
        "polar_evidence": sum(
            parameter.numel() for parameter in model.polar_expert.parameters()
        ),
        "conditional_router": sum(
            parameter.numel() for parameter in model.regret_router.parameters()
        ),
    }
    active = dict(components)
    if model.variant == NO_RELATIONAL_TRANSPORT:
        active["relational_transport"] = 0
    if model.variant == NO_POLAR_EVIDENCE:
        active["polar_evidence"] = 0
    if model.variant == FIXED_ROUTING:
        active["conditional_router"] = 0
    return {
        "stored_unique": int(sum(components.values())),
        "active_executable": int(sum(active.values())),
        "components_stored": components,
        "components_active": active,
        "additional_image_encoders": 0,
    }


def load_unified_pointer_reader_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[UnifiedPointerReader, dict[str, Any]]:
    """Load one publication-facing checkpoint without any teacher dependency."""

    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "unified-reader checkpoint is malformed")
    _require(payload.get("protocol") == PROTOCOL, "unified-reader protocol differs")
    _require(payload.get("architecture") == ARCHITECTURE, "architecture differs")
    construction = payload.get("construction")
    _require(isinstance(construction, Mapping), "construction metadata is missing")
    variant = str(payload.get("variant") or FULL)
    _require(variant in VARIANTS, "checkpoint variant differs")
    bank, bank_metadata = load_remst_r2mt_expert_bank(
        Path(str(payload["source_r2mt_checkpoint"])),
        device=device,
    )
    polar = AngularMomentLogitRefiner()
    polar.load_state_dict(payload["polar_expert_state"], strict=True)
    router = ConditionalRegretRouter(
        latent_features=int(construction["router_latent_features"]),
        hidden_features=int(construction["router_hidden_features"]),
        maximum_gain=float(construction["maximum_gain"]),
        gain_logit_scale=float(construction["gain_logit_scale"]),
        prior_weights=tuple(float(value) for value in construction["prior_weights"]),
    )
    router.load_state_dict(payload["regret_router_state"], strict=True)
    model = UnifiedPointerReader(bank, polar, router, variant=variant).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    source_seed = int(
        bank_metadata["source_metadata"]["source_foundation"]["source_seed"]
    )
    return model, {
        "checkpoint": str(source),
        "architecture": ARCHITECTURE,
        "publication_name": PUBLICATION_NAME,
        "protocol": PROTOCOL,
        "seed": int(payload["seed"]),
        "source_seed": source_seed,
        "variant": variant,
        "weight_variant": str(payload.get("weight_variant") or "terminal"),
        "checkpoint_selection": str(payload.get("checkpoint_selection")),
        "construction": dict(construction),
        "training_scope": dict(payload.get("training_scope", {})),
        "parameter_inventory": parameter_inventory(model),
    }


__all__ = [
    "ARCHITECTURE",
    "CANDIDATE_NAMES",
    "FIXED_ROUTING",
    "FULL",
    "NO_GEOMETRY_FUSION",
    "NO_POLAR_EVIDENCE",
    "NO_RELATIONAL_TRANSPORT",
    "PROTOCOL",
    "PUBLICATION_NAME",
    "UnifiedPointerReader",
    "VARIANTS",
    "candidate_mask",
    "fixed_geometry_base",
    "load_unified_pointer_reader_checkpoint",
    "mask_router_logits",
    "parameter_inventory",
]
