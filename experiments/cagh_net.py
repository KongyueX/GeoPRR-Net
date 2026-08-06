"""Coupled Angle--Geometry Hybrid Network (CAGH-Net).

CAGH-Net retains the signed PEPD ResNet and prediction heads, exposes its
multi-scale features, and adds an internally coupled high-resolution CNN and
global Transformer stream.  Two bidirectional feature-coupling stages exchange
local and global representations before dense vector/Hough and keypoint
evidence are constructed.  All evidence is processed by one 1-D refiner and a
single final softmax; no scalar or posterior averaging is used.

The final evidence residual is zero initialized.  Loading a PEPD state therefore
makes the initial CAGH progress distribution exactly equal to PEPD's visual
progress distribution while leaving every new branch available for supervised
auxiliary losses and subsequent end-to-end optimization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from experiments.geopepd import von_mises_concentration_from_resultant
from experiments.geopepd_progress import (
    _affine_linear,
    _batch_mask,
    _batch_vector,
    _legacy_visual_summary,
)
from experiments.probabilistic_pivot_direction import ProbabilisticPivotDirectionNet


class CAGHMultiscaleFeatures(NamedTuple):
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    c5: torch.Tensor


class CAGHKeypointSolverOutputs(NamedTuple):
    progress_log_probability: torch.Tensor
    expected_progress: torch.Tensor
    angle_evidence: torch.Tensor
    keypoint_direction: torch.Tensor
    pointer_length: torch.Tensor
    quality: torch.Tensor
    concentration: torch.Tensor
    reference_branch_index: torch.Tensor
    valid: torch.Tensor
    posterior_mass_error: torch.Tensor


class CAGHOutputs(NamedTuple):
    progress_log_probability: torch.Tensor
    expected_progress: torch.Tensor
    base_pepd_progress_log_probability: torch.Tensor
    base_pepd_expected_progress: torch.Tensor
    global_angle_evidence: torch.Tensor
    coupled_global_angle_evidence: torch.Tensor
    local_hough_angle_evidence: torch.Tensor
    keypoint_angle_evidence: torch.Tensor
    interaction_angle_evidence: torch.Tensor
    unified_angle_logits: torch.Tensor
    mask_support_logits: torch.Tensor
    mask_support_probability: torch.Tensor
    tip_heatmap_logits: torch.Tensor
    tip_heatmap_probability: torch.Tensor
    tail_heatmap_logits: torch.Tensor
    tail_heatmap_probability: torch.Tensor
    dense_vector: torch.Tensor
    dense_confidence_logits: torch.Tensor
    dense_confidence_probability: torch.Tensor
    dense_uncertainty: torch.Tensor
    tip_xy: torch.Tensor
    tail_xy: torch.Tensor
    keypoint_direction: torch.Tensor
    coupled_global_direction: torch.Tensor
    pivot_logits: torch.Tensor
    direction_raw: torch.Tensor
    visual_angle_logits: torch.Tensor
    visual_log_variance_raw: torch.Tensor
    valid: torch.Tensor
    reference_available: torch.Tensor


def differentiable_keypoint_reference_solver(
    tip_xy: torch.Tensor,
    tail_xy: torch.Tensor,
    reference_start_angle: torch.Tensor | None,
    reference_range_angle: torch.Tensor | None,
    crop_affine: torch.Tensor | None = None,
    reference_available: torch.Tensor | None = None,
    reference_branch_index: torch.Tensor | None = None,
    *,
    progress_bins: int = 72,
    angular_sigma: float = 0.08,
) -> CAGHKeypointSolverOutputs:
    """Map normalized crop-space tip/tail evidence to reference progress.

    This is the exact low-cost oracle-screen geometry used by CAGH.  It is a
    pure differentiable tensor operation: no model, fitted parameter, cache,
    threshold search, or dataset access occurs here.  ``crop_affine`` is the
    original-to-crop affine; only its linear 2×2 part acts on directions.
    """

    tip = torch.as_tensor(tip_xy, dtype=torch.float32)
    tail = torch.as_tensor(tail_xy, device=tip.device, dtype=torch.float32)
    if tip.ndim != 2 or tip.shape[1] != 2 or tail.shape != tip.shape:
        raise ValueError("tip_xy and tail_xy must have shape [B,2]")
    if int(progress_bins) < 16 or not 0.0 < float(angular_sigma) <= math.pi:
        raise ValueError("invalid solver progress_bins/angular_sigma")
    batch_size = tip.shape[0]
    device = tip.device
    start = _batch_vector(
        reference_start_angle,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
        fill=0.0,
        name="reference_start_angle",
    )
    angle_range = _batch_vector(
        reference_range_angle,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
        fill=0.0,
        name="reference_range_angle",
    )
    affine = _affine_linear(
        crop_affine,
        batch_size=batch_size,
        device=device,
        dtype=torch.float32,
    )
    supplied = _batch_mask(
        reference_available,
        batch_size=batch_size,
        device=device,
        default=(
            reference_start_angle is not None
            and reference_range_angle is not None
        ),
        name="reference_available",
    )
    delta = tip - tail
    pointer_length = torch.linalg.vector_norm(torch.nan_to_num(delta), dim=1)
    direction = F.normalize(torch.nan_to_num(delta), dim=1, eps=1e-8)
    valid = (
        supplied
        & torch.isfinite(tip).all(dim=1)
        & torch.isfinite(tail).all(dim=1)
        & (pointer_length > 1e-7)
        & torch.isfinite(start)
        & torch.isfinite(angle_range)
        & (angle_range.abs() > 1e-7)
        & torch.isfinite(affine).all(dim=(1, 2))
        & (torch.linalg.det(torch.nan_to_num(affine)).abs() > 1e-8)
    )
    grid = torch.linspace(0.0, 1.0, int(progress_bins), device=device)
    theta = torch.nan_to_num(start)[:, None] + torch.nan_to_num(angle_range)[
        :, None
    ] * grid[None, :]
    original = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=2)
    crop_direction = F.normalize(
        torch.einsum("bij,bkj->bki", torch.nan_to_num(affine), original),
        dim=2,
        eps=1e-8,
    )
    # Length is a deployable geometric quality signal, not a target-dependent
    # gate.  Saturation at one quarter of the crop keeps short annotations from
    # producing overconfident oracle evidence.
    quality = torch.clamp(pointer_length / 0.25, 0.0, 1.0)
    concentration = quality / float(angular_sigma) ** 2 + 1.0
    evidence = concentration[:, None] * torch.sum(
        crop_direction * direction[:, None, :], dim=2
    )
    log_probability = F.log_softmax(evidence, dim=1)
    uniform = torch.full_like(log_probability, -math.log(float(progress_bins)))
    log_probability = torch.where(valid[:, None], log_probability, uniform)
    probability = torch.exp(log_probability)
    expected = torch.sum(probability * grid[None, :], dim=1)
    mass_error = torch.abs(probability.sum(dim=1) - 1.0)
    if reference_branch_index is None:
        branch = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        branch = torch.as_tensor(reference_branch_index, device=device).reshape(-1).long()
        if branch.shape != (batch_size,) or bool(((branch < 0) | (branch > 3)).any()):
            raise ValueError("reference_branch_index must have shape [B] in [0,3]")
    return CAGHKeypointSolverOutputs(
        progress_log_probability=log_probability,
        expected_progress=expected,
        angle_evidence=evidence,
        keypoint_direction=direction,
        pointer_length=pointer_length,
        quality=quality,
        concentration=concentration,
        reference_branch_index=branch,
        valid=valid,
        posterior_mass_error=mass_error,
    )


class BidirectionalFeatureCouplingUnit(nn.Module):
    """One local↔global coupling stage at 64×64 and 8×8 token resolution."""

    def __init__(
        self,
        *,
        local_channels: int = 48,
        token_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.local_to_global = nn.Conv2d(local_channels, token_dim, 1, bias=False)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=2 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_to_local = nn.Conv2d(token_dim, local_channels, 1, bias=False)
        self.local_refine = nn.Sequential(
            nn.Conv2d(local_channels, local_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, local_channels),
            nn.GELU(),
        )
        self.local_gate = nn.Parameter(torch.tensor(0.0))
        self.global_gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, local: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local.ndim != 4 or tokens.ndim != 3:
            raise ValueError("FCU expects local [B,C,H,W] and tokens [B,N,D]")
        pooled = F.adaptive_avg_pool2d(local, (8, 8))
        local_tokens = self.local_to_global(pooled).flatten(2).transpose(1, 2)
        if local_tokens.shape != tokens.shape:
            raise ValueError("FCU local/global token shapes differ")
        tokens = self.transformer(
            tokens + torch.tanh(self.global_gate) * local_tokens
        )
        global_map = tokens.transpose(1, 2).reshape(
            local.shape[0], tokens.shape[2], 8, 8
        )
        injected = F.interpolate(
            self.global_to_local(global_map),
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        local = local + torch.tanh(self.local_gate) * injected
        local = local + self.local_refine(local)
        return local, tokens


class CAGHNet(ProbabilisticPivotDirectionNet):
    """End-to-end coupled local/global angle-evidence model."""

    _PEPD_PREFIXES = (
        "encoder.",
        "pivot_head.",
        "direction_features.",
        "vector_head.",
        "angle_head.",
        "log_variance_head.",
    )
    _NEW_MODULE_NAMES = (
        "local_c2_projection",
        "local_c3_projection",
        "local_c4_projection",
        "local_c5_projection",
        "pivot_feature_projection",
        "global_projection",
        "coupling_stages",
        "mask_support_head",
        "tip_heatmap_head",
        "tail_residual_head",
        "dense_vote_head",
        "coupled_global_direction_head",
        "coupled_global_uncertainty_head",
        "evidence_refiner",
    )

    def __init__(
        self,
        *,
        progress_bins: int = 72,
        angle_bins: int = 72,
        local_channels: int = 48,
        token_dim: int = 128,
        imagenet_pretrained: bool = False,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(
            angle_bins=angle_bins, imagenet_pretrained=imagenet_pretrained
        )
        if int(progress_bins) < 16:
            raise ValueError("progress_bins must be at least 16")
        if int(local_channels) <= 0 or int(token_dim) < 32:
            raise ValueError("invalid CAGH feature dimensions")
        self.progress_bins = int(progress_bins)
        self.local_channels = int(local_channels)
        self.token_dim = int(token_dim)
        self.register_buffer(
            "progress_grid", torch.linspace(0.0, 1.0, self.progress_bins)
        )

        def projection(input_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(input_channels, local_channels, 1, bias=False),
                nn.GroupNorm(8, local_channels),
                nn.GELU(),
            )

        self.local_c2_projection = projection(64)
        self.local_c3_projection = projection(128)
        self.local_c4_projection = projection(256)
        self.local_c5_projection = projection(512)
        self.pivot_feature_projection = projection(64)
        self.global_projection = nn.Conv2d(512, token_dim, 1, bias=False)
        self.coupling_stages = nn.ModuleList(
            [
                BidirectionalFeatureCouplingUnit(
                    local_channels=local_channels,
                    token_dim=token_dim,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )

        self.mask_support_head = nn.Conv2d(local_channels, 1, 1)
        self.tip_heatmap_head = nn.Conv2d(local_channels, 1, 1)
        self.tail_residual_head = nn.Conv2d(local_channels, 1, 1)
        # vector x/y, confidence logit, and angular uncertainty raw value.
        self.dense_vote_head = nn.Conv2d(local_channels, 4, 1)
        self.coupled_global_direction_head = nn.Linear(token_dim, 2)
        self.coupled_global_uncertainty_head = nn.Linear(token_dim, 1)
        # base PEPD, coupled-global, dense-Hough, keypoint, product, difference.
        self.evidence_refiner = nn.Sequential(
            nn.Conv1d(6, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )
        self._configured_stage: str | None = None
        self._initialize_cagh()

    def _initialize_cagh(self) -> None:
        for name in self._NEW_MODULE_NAMES:
            module = getattr(self, name)
            for child in module.modules():
                if isinstance(child, (nn.Conv2d, nn.Conv1d)):
                    nn.init.kaiming_normal_(
                        child.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                elif isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    nn.init.zeros_(child.bias)
        # Tail starts at the signed PEPD pivot posterior.  Most importantly,
        # the unified geometry residual starts exactly zero, making the first
        # final softmax identical to the signed PEPD visual softmax.
        nn.init.zeros_(self.tail_residual_head.weight)
        nn.init.zeros_(self.tail_residual_head.bias)
        final = self.evidence_refiner[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def load_pepd_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        own = self.state_dict()
        required = {key for key in own if key.startswith(self._PEPD_PREFIXES)}
        supplied_legacy = {
            key for key in state_dict if key.startswith(self._PEPD_PREFIXES)
        }
        missing = sorted(required - supplied_legacy)
        mismatched = sorted(
            key
            for key in required & supplied_legacy
            if tuple(own[key].shape) != tuple(state_dict[key].shape)
        )
        if missing or mismatched:
            raise ValueError(
                "incompatible PEPD state: "
                f"missing={missing[:5]}, shape_mismatch={mismatched[:5]}"
            )
        filtered = {key: state_dict[key] for key in required}
        self.load_state_dict(filtered, strict=False)

    def forward_multiscale_features(
        self, image: torch.Tensor
    ) -> CAGHMultiscaleFeatures:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B,3,H,W]")
        if image.shape[-2:] != (256, 256):
            raise ValueError("CAGH currently requires normalized 256x256 crops")
        result = image
        for module in self.encoder[:4]:
            result = module(result)
        c2 = self.encoder[4](result)
        c3 = self.encoder[5](c2)
        c4 = self.encoder[6](c3)
        c5 = self.encoder[7](c4)
        return CAGHMultiscaleFeatures(c2=c2, c3=c3, c4=c4, c5=c5)

    def _pivot_features(self, c5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = c5
        for module in tuple(self.pivot_head.children())[:-1]:
            result = module(result)
        return result, self.pivot_head[-1](result)

    @staticmethod
    def _spatial_probability_and_xy(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = logits.shape
        if channels != 1:
            raise ValueError("spatial logits must have one channel")
        probability = torch.softmax(logits.flatten(1).float(), dim=1).reshape_as(logits)
        coordinate_x = torch.linspace(
            0.0, 1.0, width, device=logits.device, dtype=probability.dtype
        )
        coordinate_y = torch.linspace(
            0.0, 1.0, height, device=logits.device, dtype=probability.dtype
        )
        yy, xx = torch.meshgrid(coordinate_y, coordinate_x, indexing="ij")
        x = torch.sum(probability[:, 0] * xx[None], dim=(1, 2))
        y = torch.sum(probability[:, 0] * yy[None], dim=(1, 2))
        return probability, torch.stack((x, y), dim=1)

    @staticmethod
    def _standardize_evidence(evidence: torch.Tensor) -> torch.Tensor:
        centered = evidence - evidence.mean(dim=1, keepdim=True)
        scale = torch.sqrt(centered.square().mean(dim=1, keepdim=True) + 1e-6)
        return centered / scale

    def configure_trainable_stage(self, stage: str) -> tuple[str, ...]:
        """Apply the frozen 3+3+2 epoch trainability schedule."""

        if stage not in {"head", "c4_c5", "c3_c5"}:
            raise ValueError("stage must be head, c4_c5, or c3_c5")
        self._configured_stage = stage
        nn.Module.train(self, False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name in self._NEW_MODULE_NAMES:
            module = getattr(self, name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if stage in {"c4_c5", "c3_c5"}:
            for module in (
                self.encoder[6],
                self.encoder[7],
                self.direction_features,
                self.vector_head,
                self.angle_head,
                self.log_variance_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if stage == "c3_c5":
            for module in (self.encoder[5], self.pivot_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self._set_configured_stage_mode()
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    def _configured_train_modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [
            getattr(self, name) for name in self._NEW_MODULE_NAMES
        ]
        if self._configured_stage in {"c4_c5", "c3_c5"}:
            modules.extend(
                (
                    self.encoder[6],
                    self.encoder[7],
                    self.direction_features,
                    self.vector_head,
                    self.angle_head,
                    self.log_variance_head,
                )
            )
        if self._configured_stage == "c3_c5":
            modules.extend((self.encoder[5], self.pivot_head))
        return tuple(modules)

    def _set_configured_stage_mode(self) -> None:
        """Keep frozen BatchNorm/Dropout layers in evaluation mode."""

        nn.Module.train(self, False)
        for module in self._configured_train_modules():
            module.train(True)

    def train(self, mode: bool = True) -> "CAGHNet":
        """Respect the configured freeze stage across ordinary train calls."""

        if bool(mode) and self._configured_stage is not None:
            self._set_configured_stage_mode()
            return self
        nn.Module.train(self, bool(mode))
        return self

    def forward(
        self,
        image: torch.Tensor,
        reference_start_angle: torch.Tensor | None,
        reference_range_angle: torch.Tensor | None,
        crop_affine: torch.Tensor | None = None,
        reference_available: torch.Tensor | None = None,
    ) -> CAGHOutputs:
        features = self.forward_multiscale_features(image)
        batch_size = image.shape[0]
        device = image.device
        pivot_features, pivot_logits = self._pivot_features(features.c5)

        def upsample(value: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                value, size=(64, 64), mode="bilinear", align_corners=False
            )

        local = (
            self.local_c2_projection(features.c2)
            + upsample(self.local_c3_projection(features.c3))
            + upsample(self.local_c4_projection(features.c4))
            + upsample(self.local_c5_projection(features.c5))
            + self.pivot_feature_projection(pivot_features)
        )
        tokens = self.global_projection(features.c5).flatten(2).transpose(1, 2)
        for coupling in self.coupling_stages:
            local, tokens = coupling(local, tokens)

        pooled_visual = self.direction_features(features.c5)
        direction_raw = self.vector_head(pooled_visual)
        visual_angle_logits = self.angle_head(pooled_visual)
        visual_log_variance = self.log_variance_head(pooled_visual)
        visual_direction, visual_resultant = _legacy_visual_summary(
            direction_raw, visual_angle_logits
        )
        visual_concentration = von_mises_concentration_from_resultant(
            visual_resultant.clamp(1e-4, 0.995)
        )

        mask_support_logits = self.mask_support_head(local)
        tip_logits = self.tip_heatmap_head(local)
        tail_logits = pivot_logits + self.tail_residual_head(local)
        tip_probability, tip_xy = self._spatial_probability_and_xy(tip_logits)
        tail_probability, tail_xy = self._spatial_probability_and_xy(tail_logits)
        keypoint_delta = tip_xy - tail_xy
        keypoint_direction = F.normalize(keypoint_delta, dim=1, eps=1e-8)

        vote = self.dense_vote_head(local)
        dense_vector = F.normalize(vote[:, :2].float(), dim=1, eps=1e-8)
        dense_confidence_logits = vote[:, 2:3]
        dense_uncertainty = torch.clamp(
            F.softplus(vote[:, 3:4].float()) + 0.03, min=0.03, max=1.0
        )
        mask_support_probability = torch.sigmoid(mask_support_logits.float())
        dense_confidence_probability = torch.sigmoid(
            dense_confidence_logits.float()
        )

        start = _batch_vector(
            reference_start_angle,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            fill=0.0,
            name="reference_start_angle",
        )
        angle_range = _batch_vector(
            reference_range_angle,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
            fill=0.0,
            name="reference_range_angle",
        )
        affine = _affine_linear(
            crop_affine,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        supplied_reference = _batch_mask(
            reference_available,
            batch_size=batch_size,
            device=device,
            default=(
                reference_start_angle is not None
                and reference_range_angle is not None
            ),
            name="reference_available",
        )
        effective_reference = (
            supplied_reference
            & torch.isfinite(start)
            & torch.isfinite(angle_range)
            & (angle_range.abs() > 1e-7)
            & torch.isfinite(affine).all(dim=(1, 2))
            & (torch.linalg.det(torch.nan_to_num(affine)).abs() > 1e-8)
        )
        safe_start = torch.nan_to_num(start)
        safe_range = torch.nan_to_num(angle_range)
        safe_affine = torch.nan_to_num(affine)
        grid = self.progress_grid.to(device=device, dtype=torch.float32)
        theta = safe_start[:, None] + safe_range[:, None] * grid[None, :]
        original_direction = torch.stack(
            (-torch.sin(theta), torch.cos(theta)), dim=2
        )
        crop_direction = F.normalize(
            torch.einsum("bij,bkj->bki", safe_affine, original_direction),
            dim=2,
            eps=1e-8,
        )

        global_evidence = visual_concentration[:, None] * torch.sum(
            crop_direction * visual_direction[:, None, :], dim=2
        )
        global_token = tokens.mean(dim=1)
        coupled_global_direction = F.normalize(
            self.coupled_global_direction_head(global_token).float(), dim=1, eps=1e-8
        )
        coupled_scale = torch.clamp(
            F.softplus(self.coupled_global_uncertainty_head(global_token)[:, 0])
            + 0.05,
            min=0.05,
            max=2.0,
        )
        coupled_global_evidence = torch.sum(
            crop_direction * coupled_global_direction[:, None, :], dim=2
        ) / coupled_scale[:, None].square()

        support_log_weight = F.log_softmax(
            (mask_support_logits.float() + dense_confidence_logits.float()).flatten(1),
            dim=1,
        )
        vector_flat = dense_vector.flatten(2).transpose(1, 2)
        uncertainty_flat = dense_uncertainty.flatten(1)
        vote_similarity = torch.einsum("bnd,bkd->bnk", vector_flat, crop_direction)
        local_hough_evidence = torch.logsumexp(
            support_log_weight[:, :, None]
            + vote_similarity / uncertainty_flat[:, :, None].square(),
            dim=1,
        )
        keypoint_solver = differentiable_keypoint_reference_solver(
            tip_xy,
            tail_xy,
            start,
            angle_range,
            affine,
            effective_reference,
            progress_bins=self.progress_bins,
            angular_sigma=0.08,
        )
        keypoint_evidence = keypoint_solver.angle_evidence

        standardized_global = self._standardize_evidence(global_evidence)
        standardized_coupled = self._standardize_evidence(coupled_global_evidence)
        standardized_local = self._standardize_evidence(local_hough_evidence)
        standardized_keypoint = self._standardize_evidence(keypoint_evidence)
        refiner_input = torch.stack(
            (
                standardized_global,
                standardized_coupled,
                standardized_local,
                standardized_keypoint,
                standardized_coupled * standardized_local,
                standardized_local - standardized_keypoint,
            ),
            dim=1,
        )
        interaction_evidence = self.evidence_refiner(refiner_input)[:, 0, :]
        unified_logits = global_evidence + interaction_evidence
        final_log_probability = F.log_softmax(unified_logits, dim=1)
        base_log_probability = F.log_softmax(global_evidence, dim=1)
        uniform = torch.full_like(
            final_log_probability, -math.log(float(self.progress_bins))
        )
        final_log_probability = torch.where(
            effective_reference[:, None], final_log_probability, uniform
        )
        base_log_probability = torch.where(
            effective_reference[:, None], base_log_probability, uniform
        )
        expected = torch.sum(
            torch.exp(final_log_probability) * grid[None, :], dim=1
        )
        base_expected = torch.sum(
            torch.exp(base_log_probability) * grid[None, :], dim=1
        )
        visual_valid = (
            torch.isfinite(visual_direction).all(dim=1)
            & (torch.linalg.vector_norm(torch.nan_to_num(visual_direction), dim=1) > 1e-8)
        )
        return CAGHOutputs(
            progress_log_probability=final_log_probability,
            expected_progress=expected,
            base_pepd_progress_log_probability=base_log_probability,
            base_pepd_expected_progress=base_expected,
            global_angle_evidence=global_evidence,
            coupled_global_angle_evidence=coupled_global_evidence,
            local_hough_angle_evidence=local_hough_evidence,
            keypoint_angle_evidence=keypoint_evidence,
            interaction_angle_evidence=interaction_evidence,
            unified_angle_logits=unified_logits,
            mask_support_logits=mask_support_logits,
            mask_support_probability=mask_support_probability,
            tip_heatmap_logits=tip_logits,
            tip_heatmap_probability=tip_probability,
            tail_heatmap_logits=tail_logits,
            tail_heatmap_probability=tail_probability,
            dense_vector=dense_vector,
            dense_confidence_logits=dense_confidence_logits,
            dense_confidence_probability=dense_confidence_probability,
            dense_uncertainty=dense_uncertainty,
            tip_xy=tip_xy,
            tail_xy=tail_xy,
            keypoint_direction=keypoint_direction,
            coupled_global_direction=coupled_global_direction,
            pivot_logits=pivot_logits,
            direction_raw=direction_raw,
            visual_angle_logits=visual_angle_logits,
            visual_log_variance_raw=visual_log_variance,
            valid=effective_reference & visual_valid,
            reference_available=effective_reference,
        )


def _soft_progress_target(target: torch.Tensor, bins: int) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, bins, device=target.device, dtype=target.dtype)
    sigma = 1.25 / float(bins - 1)
    probability = torch.exp(-0.5 * ((grid[None, :] - target[:, None]) / sigma).square())
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)


def cagh_multitask_loss(
    outputs: CAGHOutputs,
    target_progress: torch.Tensor,
    target_tip_xy: torch.Tensor,
    target_tail_xy: torch.Tensor,
    target_mask_probability: torch.Tensor | None = None,
    group_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Minimal fixed multitask loss using only progress and real tip/tail labels."""

    target = torch.as_tensor(
        target_progress,
        device=outputs.expected_progress.device,
        dtype=torch.float32,
    ).reshape(-1)
    tip = torch.as_tensor(target_tip_xy, device=target.device, dtype=torch.float32)
    tail = torch.as_tensor(target_tail_xy, device=target.device, dtype=torch.float32)
    batch = outputs.expected_progress.shape[0]
    if target.shape != (batch,) or tip.shape != (batch, 2) or tail.shape != (batch, 2):
        raise ValueError("targets must be progress [B] and tip/tail [B,2]")
    if not bool(torch.isfinite(tip).all() and torch.isfinite(tail).all()):
        raise ValueError("CAGH tip/tail targets must be finite")
    weight = (
        torch.ones(batch, device=target.device, dtype=torch.float32)
        if group_weight is None
        else torch.as_tensor(group_weight, device=target.device, dtype=torch.float32).reshape(-1)
    )
    if weight.shape != (batch,) or not bool(torch.isfinite(weight).all()) or bool(
        (weight <= 0.0).any()
    ):
        raise ValueError("group_weight must be finite, positive, and shape [B]")

    def weighted(row: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        selected = torch.ones(batch, dtype=torch.bool, device=target.device) if mask is None else mask
        if not bool(selected.any()):
            return row.sum() * 0.0
        return (row[selected] * weight[selected]).sum() / weight[selected].sum().clamp_min(1e-8)

    valid = (
        outputs.valid
        & torch.isfinite(target)
        & (target >= 0.0)
        & (target <= 1.0)
    )
    safe_target = torch.nan_to_num(target, nan=0.5).clamp(0.0, 1.0)
    soft = _soft_progress_target(
        safe_target, outputs.progress_log_probability.shape[1]
    )
    final_ce = weighted(
        -(soft * outputs.progress_log_probability).sum(1), valid
    )
    final_expected = weighted(
        F.smooth_l1_loss(
            outputs.expected_progress, safe_target, beta=0.02, reduction="none"
        ),
        valid,
    )

    coordinate = torch.linspace(0.0, 1.0, 64, device=target.device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")

    def heatmap_ce(logits: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        distance = (
            (xx[None] - point[:, 0, None, None]).square()
            + (yy[None] - point[:, 1, None, None]).square()
        )
        target_probability = torch.exp(-0.5 * distance / (0.025**2))
        target_probability = target_probability / target_probability.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-8)
        row = -(
            target_probability.flatten(1)
            * F.log_softmax(logits.flatten(1), dim=1)
        ).sum(1)
        return weighted(row)

    tip_ce = heatmap_ce(outputs.tip_heatmap_logits, tip)
    tail_ce = heatmap_ce(outputs.tail_heatmap_logits, tail)
    coordinate_loss = weighted(
        F.smooth_l1_loss(outputs.tip_xy, tip, beta=0.02, reduction="none").mean(1)
        + F.smooth_l1_loss(
            outputs.tail_xy, tail, beta=0.02, reduction="none"
        ).mean(1)
    )
    target_direction = F.normalize(tip - tail, dim=1, eps=1e-8)
    keypoint_direction_loss = weighted(
        1.0 - torch.sum(outputs.keypoint_direction * target_direction, dim=1)
    )
    dense_cosine = torch.sum(
        outputs.dense_vector * target_direction[:, :, None, None], dim=1
    )
    dense_weight = outputs.mask_support_probability[:, 0].detach()
    dense_vector_row = (
        ((1.0 - dense_cosine) * dense_weight).sum((1, 2))
        / dense_weight.sum((1, 2)).clamp_min(1e-8)
    )
    dense_vector_loss = weighted(dense_vector_row)
    if target_mask_probability is None:
        mask_support_bce = outputs.mask_support_logits.sum() * 0.0
    else:
        target_mask = torch.as_tensor(
            target_mask_probability,
            device=target.device,
            dtype=torch.float32,
        )
        if target_mask.shape == (batch, 64, 64):
            target_mask = target_mask[:, None, :, :]
        if target_mask.shape != outputs.mask_support_logits.shape:
            raise ValueError("target_mask_probability must match [B,1,64,64]")
        if not bool(torch.isfinite(target_mask).all()):
            raise ValueError("target_mask_probability must be finite")
        mask_support_bce = weighted(
            F.binary_cross_entropy_with_logits(
                outputs.mask_support_logits.float(),
                target_mask.clamp(0.0, 1.0),
                reduction="none",
            ).mean((1, 2, 3))
        )
    auxiliary_evidence = outputs.progress_log_probability.sum() * 0.0
    for evidence in (
        outputs.coupled_global_angle_evidence,
        outputs.local_hough_angle_evidence,
        outputs.keypoint_angle_evidence,
    ):
        auxiliary_evidence = auxiliary_evidence + weighted(
            -(soft * F.log_softmax(evidence, dim=1)).sum(1), valid
        )
    loss = (
        final_ce
        + 2.0 * final_expected
        + 0.25 * (tip_ce + tail_ce)
        + 0.5 * coordinate_loss
        + 0.25 * keypoint_direction_loss
        + 0.10 * dense_vector_loss
        + 0.25 * mask_support_bce
        + 0.05 * auxiliary_evidence
    )
    parts = {
        "final_soft_ce": final_ce,
        "final_expected_smooth_l1": final_expected,
        "tip_heatmap_ce": tip_ce,
        "tail_heatmap_ce": tail_ce,
        "keypoint_coordinate_loss": coordinate_loss,
        "keypoint_direction_loss": keypoint_direction_loss,
        "dense_vector_loss": dense_vector_loss,
        "mask_support_bce": mask_support_bce,
        "auxiliary_angle_evidence": auxiliary_evidence,
        "valid_progress_fraction": valid.float().mean(),
    }
    if not bool(torch.isfinite(loss)) or any(
        not bool(torch.isfinite(value)) for value in parts.values()
    ):
        raise FloatingPointError("CAGH multitask loss is non-finite")
    return loss, {name: value.detach() for name, value in parts.items()}
