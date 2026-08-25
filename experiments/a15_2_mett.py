"""A15.2 moment-exact endpoint-tangent transport (METT).

METT addresses two limitations of the original A15.2 formulation without
introducing a sample router or an expert mixture:

* a scalar EfficientNet-B0 prediction is lifted to a full progress posterior
  by an exponential-family projection whose discrete mean is the scalar
  prediction; and
* the correction path reaches the SARN endpoint before applying a learned
  endpoint-tangent extrapolation.  Its free shape field is orthogonal to the
  endpoint's mass and first-moment directions.

The result is one continuous posterior-valued model.  Raw and SARN are two
observations of the same frozen anchor, not independently selected experts.
Missing geometry still returns the Raw posterior exactly.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from experiments.a11_scort import (
    DEFAULT_RELATION_CHANNELS,
    DEFAULT_TOKEN_DIM,
    EFFICIENTNET_B0_FEATURES,
    EFFICIENTNET_B0_MIDDLE_FEATURES,
    RAW_STRIDE8_CHANNELS,
    SCORTRawEfficientNetB0Encoder,
)
from experiments.a15_2_fteb import A15_2_LEARNED_RESIDUAL_SCALE
from experiments.a15_fteb import (
    BRIDGE_LAYERS,
    DEFAULT_MEMORY_GRID_SIZE,
    DEFAULT_PROGRESS_BINS,
    LOG_RATIO_LIMIT,
    MAX_ENDPOINT_RESIDUAL_GAIN,
    MAX_FREE_FIELD,
    PROBABILITY_EPSILON,
    A15FTEBCorrection,
    _posterior_moments,
    _validate_posterior,
)
from experiments.syncg_lightweight_regression_baselines import (
    BACKBONE_SPECS,
    DEFAULT_EPOCHS as DIRECT_BASELINE_EPOCHS,
    PROTOCOL as DIRECT_BASELINE_PROTOCOL,
    LightweightProgressRegressor,
)


A15_2_METT_ARCHITECTURE: Final[str] = (
    "A15.2-METT-Moment-Exact-Endpoint-Tangent-Natural-Parameter-Transport"
)
# Publication-facing identity of the empirically selected direct-scalar
# construction.  The A15.2/METT identifiers above remain unchanged so that
# already-produced checkpoints and machine-readable experiment keys continue
# to replay without migration.
REMST_SHORT_NAME: Final[str] = "ReMST"
REMST_FULL_NAME: Final[str] = (
    "Relation-Encoded Moment-Exact Scalar Transport"
)
REMST_ARCHITECTURE: Final[str] = (
    "ReMST-EfficientNetB0-Dual-View-Relation-Encoded-"
    "Moment-Exact-Scalar-Transport"
)
REMST_LEGACY_VARIANT: Final[str] = "direct_scalar"
DEFAULT_POSTERIOR_SCALE: Final[float] = 0.025
MIN_POSTERIOR_SCALE: Final[float] = 0.004
MAX_POSTERIOR_SCALE: Final[float] = 0.15
MOMENT_SOLVER_STEPS: Final[int] = 20
MOMENT_SOLVER_LIMIT: Final[float] = 2048.0
MAX_DIRECT_PROGRESS_RESIDUAL: Final[float] = 0.1


def remst_publication_identity() -> dict[str, str]:
    """Return the paper-facing name without changing legacy artifact IDs."""

    return {
        "short_name": REMST_SHORT_NAME,
        "full_name": REMST_FULL_NAME,
        "display_name": f"{REMST_SHORT_NAME} (ours)",
        "architecture": REMST_ARCHITECTURE,
        "legacy_family": "A15.2-METT",
        "legacy_experiment_variant": REMST_LEGACY_VARIANT,
        "legacy_checkpoint_architecture": A15_2_METT_ARCHITECTURE,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class MomentExactPosteriorHead(nn.Module):
    """Lift one scalar prediction into a mean-constrained posterior.

    The base density is a Gaussian on the fixed progress grid.  A scalar
    natural parameter is solved by Newton updates so that the posterior
    expectation equals the sigmoid point prediction.  The implementation
    retains a scale projection for controlled sensitivity experiments, but the
    correction-only METT protocol freezes the anchor and therefore uses the
    fixed initial scale rather than claiming learned uncertainty calibration.
    """

    def __init__(
        self,
        *,
        feature_dim: int = EFFICIENTNET_B0_FEATURES,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        initial_scale: float = DEFAULT_POSTERIOR_SCALE,
        calibration_knots_x: Sequence[float] | None = None,
        calibration_knots_y: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        _require(feature_dim >= 1, "METT anchor feature width must be positive")
        _require(progress_bins >= 16, "METT needs at least 16 progress bins")
        _require(
            MIN_POSTERIOR_SCALE <= float(initial_scale) <= MAX_POSTERIOR_SCALE,
            "METT initial posterior scale is out of range",
        )
        self.feature_dim = int(feature_dim)
        self.progress_bins = int(progress_bins)
        self.point_projection = nn.Linear(self.feature_dim, 1)
        self.log_scale_projection = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.log_scale_projection.weight)
        nn.init.constant_(
            self.log_scale_projection.bias, math.log(float(initial_scale))
        )
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, self.progress_bins, dtype=torch.float32),
        )
        self.register_buffer(
            "calibration_knots_x", torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "calibration_knots_y", torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        if calibration_knots_x is not None or calibration_knots_y is not None:
            _require(
                calibration_knots_x is not None
                and calibration_knots_y is not None,
                "METT calibration needs both knot axes",
            )
            self.set_monotone_calibration(
                calibration_knots_x, calibration_knots_y
            )

    def set_monotone_calibration(
        self,
        knots_x: Sequence[float],
        knots_y: Sequence[float],
    ) -> None:
        x = torch.as_tensor(tuple(knots_x), dtype=torch.float32)
        y = torch.as_tensor(tuple(knots_y), dtype=torch.float32)
        _require(
            x.ndim == y.ndim == 1
            and x.shape == y.shape
            and x.numel() >= 3
            and bool(torch.isfinite(x).all())
            and bool(torch.isfinite(y).all())
            and bool((x[1:] > x[:-1]).all())
            and bool((y[1:] >= y[:-1]).all())
            and float(x[0]) >= 0.0
            and float(x[-1]) <= 1.0
            and float(y[0]) >= 0.0
            and float(y[-1]) <= 1.0,
            "METT monotone calibration knots are malformed",
        )
        device = self.progress_grid.device
        self.calibration_knots_x = x.to(device)
        self.calibration_knots_y = y.to(device)

    def clear_monotone_calibration(self) -> None:
        device = self.progress_grid.device
        self.calibration_knots_x = torch.empty(
            0, dtype=torch.float32, device=device
        )
        self.calibration_knots_y = torch.empty(
            0, dtype=torch.float32, device=device
        )

    def _calibrate(self, point: torch.Tensor) -> torch.Tensor:
        if self.calibration_knots_x.numel() == 0:
            return point
        with torch.autocast(device_type=point.device.type, enabled=False):
            value = point.float()
            x = self.calibration_knots_x.float()
            y = self.calibration_knots_y.float()
            # Keep the calibration bounds on-device.  Converting either CUDA
            # scalar to ``float`` here forces a host synchronization on every
            # Raw/SARN endpoint forward.
            clipped = torch.maximum(torch.minimum(value, x[-1]), x[0])
            upper = torch.searchsorted(x, clipped, right=True).clamp(
                1, x.numel() - 1
            )
            lower = upper - 1
            alpha = (clipped - x[lower]) / (x[upper] - x[lower])
            calibrated = y[lower] + alpha * (y[upper] - y[lower])
        return calibrated.clamp(0.0, 1.0)

    def point_progress(self, representation: torch.Tensor) -> torch.Tensor:
        _require(
            representation.ndim == 2
            and representation.shape[1] == self.feature_dim,
            "METT anchor representation has the wrong shape",
        )
        point = torch.sigmoid(
            self.point_projection(representation.float()).squeeze(1)
        )
        return self._calibrate(point)

    def posterior_parameters(
        self, representation: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        point = self.point_progress(representation)
        with torch.autocast(device_type=representation.device.type, enabled=False):
            features = representation.float()
            target_mean = point.float().clamp(
                torch.finfo(torch.float32).eps,
                1.0 - torch.finfo(torch.float32).eps,
            )
            log_scale = self.log_scale_projection(features).squeeze(1).clamp(
                math.log(MIN_POSTERIOR_SCALE),
                math.log(MAX_POSTERIOR_SCALE),
            )
            scale = log_scale.exp()
            grid = self.progress_grid.float()[None]
            centered = grid - target_mean[:, None]
            base_logits = -0.5 * (centered / scale[:, None]).square()
            natural_tilt = torch.zeros_like(target_mean[:, None])
            for _ in range(MOMENT_SOLVER_STEPS):
                logits = base_logits + natural_tilt * grid
                posterior = torch.softmax(logits, dim=1)
                observed_mean = (posterior * grid).sum(dim=1, keepdim=True)
                variance = (
                    posterior * (grid - observed_mean).square()
                ).sum(dim=1, keepdim=True).clamp_min(1.0e-10)
                natural_tilt = (
                    natural_tilt
                    + (target_mean[:, None] - observed_mean) / variance
                ).clamp(-MOMENT_SOLVER_LIMIT, MOMENT_SOLVER_LIMIT)
            logits = base_logits + natural_tilt * grid
            log_posterior = torch.log_softmax(logits, dim=1)
            posterior = log_posterior.exp()
            posterior_mean, posterior_variance = _posterior_moments(posterior)
        return {
            "logits": log_posterior,
            "log_posterior": log_posterior,
            "posterior": posterior,
            "point_progress": point.float(),
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "posterior_scale": scale,
            "natural_tilt": natural_tilt.squeeze(1),
            "absolute_moment_error": torch.abs(
                posterior_mean - target_mean
            ),
        }

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        # FTEB's frozen endpoint adapter expects logits.  Normalized log
        # probabilities are valid logits and avoid a second numerical scale.
        return self.posterior_parameters(representation)["logits"]


class MomentExactEfficientNetB0Anchor(nn.Module):
    """EfficientNet-B0 point anchor with a moment-exact posterior interface."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        initial_scale: float = DEFAULT_POSTERIOR_SCALE,
    ) -> None:
        super().__init__()
        self.progress_bins = int(progress_bins)
        self.raw_encoder = SCORTRawEfficientNetB0Encoder(
            imagenet_pretrained=False
        )
        self.raw_posterior_head = MomentExactPosteriorHead(
            progress_bins=self.progress_bins,
            initial_scale=initial_scale,
        )

    def forward(self, image: torch.Tensor) -> dict[str, Any]:
        features = self.raw_encoder(image)
        lifted = self.raw_posterior_head.posterior_parameters(
            features["representation"]
        )
        return {
            "architecture": "METT-Moment-Exact-EfficientNetB0-Anchor",
            "progress_posterior": lifted["posterior"],
            "mean": lifted["posterior_mean"],
            "variance": lifted["posterior_variance"],
            "standard_deviation": torch.sqrt(
                lifted["posterior_variance"].clamp_min(0.0)
            ),
            "point_progress": lifted["point_progress"],
            "posterior_scale": lifted["posterior_scale"],
            "absolute_moment_error": lifted["absolute_moment_error"],
            "raw_encoder_features": features,
        }


def load_moment_exact_anchor_from_direct_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
    progress_bins: int = DEFAULT_PROGRESS_BINS,
    initial_scale: float = DEFAULT_POSTERIOR_SCALE,
) -> tuple[MomentExactEfficientNetB0Anchor, dict[str, Any]]:
    """Import a trained scalar EfficientNet-B0 as the METT point anchor."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"METT direct checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "METT direct checkpoint is malformed")
    _require(
        checkpoint.get("protocol") == DIRECT_BASELINE_PROTOCOL
        and checkpoint.get("architecture") == "efficientnet_b0"
        and int(checkpoint.get("epochs", -1)) == DIRECT_BASELINE_EPOCHS
        and checkpoint.get("checkpoint_selection") == "terminal_fixed_epoch",
        "METT requires a formal direct EfficientNet-B0 checkpoint",
    )
    _require(
        checkpoint.get("pretrained_weights")
        == BACKBONE_SPECS["efficientnet_b0"].weights_name,
        "METT source EfficientNet-B0 initialization metadata differs",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "METT source model state is missing")
    direct = LightweightProgressRegressor(
        "efficientnet_b0", imagenet_pretrained=False
    )
    incompatibility = direct.load_state_dict(state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "METT source EfficientNet-B0 state does not load strictly",
    )
    anchor = MomentExactEfficientNetB0Anchor(
        progress_bins=progress_bins,
        initial_scale=initial_scale,
    )
    source_stages = tuple(direct.backbone.features.children())
    target_stages = (
        *tuple(anchor.raw_encoder.to_stride8.children()),
        *tuple(anchor.raw_encoder.to_stride16.children()),
        *tuple(anchor.raw_encoder.to_final.children()),
    )
    _require(
        len(source_stages) == len(target_stages) == 9,
        "METT EfficientNet-B0 feature stage layout differs",
    )
    for source_stage, target_stage in zip(
        source_stages, target_stages, strict=True
    ):
        stage_load = target_stage.load_state_dict(
            source_stage.state_dict(), strict=True
        )
        _require(
            not stage_load.missing_keys and not stage_load.unexpected_keys,
            "METT EfficientNet-B0 feature stage does not load strictly",
        )
    source_projection = direct.backbone.classifier[-1]
    _require(
        isinstance(source_projection, nn.Linear)
        and source_projection.in_features == EFFICIENTNET_B0_FEATURES
        and source_projection.out_features == 1,
        "METT source EfficientNet-B0 point head differs",
    )
    point_load = anchor.raw_posterior_head.point_projection.load_state_dict(
        source_projection.state_dict(), strict=True
    )
    _require(
        not point_load.missing_keys and not point_load.unexpected_keys,
        "METT point head does not load strictly",
    )
    del direct
    target_device = torch.device(device)
    anchor = anchor.to(target_device).eval()
    metadata = {
        "source": str(source),
        "source_protocol": str(checkpoint["protocol"]),
        "source_architecture": str(checkpoint["architecture"]),
        "source_seed": int(checkpoint["seed"]),
        "source_epochs": int(checkpoint["epochs"]),
        "source_checkpoint_selection": str(checkpoint["checkpoint_selection"]),
        "progress_bins": int(progress_bins),
        "initial_posterior_scale": float(initial_scale),
        "point_parameters_imported": True,
        "posterior_scale_parameters_fresh": True,
        "posterior_scale_mode": (
            "fixed_initial_scale_during_correction_only_training"
        ),
    }
    return anchor, metadata


class EndpointTangentNaturalParameterBridge(nn.Module):
    """Reach qS, then learn a tangent extrapolation and shape deformation."""

    def __init__(
        self,
        *,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        learned_residual_scale: float = A15_2_LEARNED_RESIDUAL_SCALE,
        use_tangent: bool = True,
        use_shape: bool = True,
        use_direct_scalar_residual: bool = False,
    ) -> None:
        super().__init__()
        _require(progress_bins >= 16, "METT bridge needs at least 16 bins")
        _require(token_dim >= 8, "METT bridge token width is too small")
        _require(
            math.isfinite(float(learned_residual_scale))
            and float(learned_residual_scale) >= 0.0,
            "METT residual scale must be finite and non-negative",
        )
        self.progress_bins = int(progress_bins)
        self.token_dim = int(token_dim)
        self.learned_residual_scale = float(learned_residual_scale)
        self.use_tangent = bool(use_tangent)
        self.use_shape = bool(use_shape)
        self.use_direct_scalar_residual = bool(use_direct_scalar_residual)
        _require(
            self.use_direct_scalar_residual
            or self.use_tangent
            or self.use_shape,
            "METT bridge has no learned mechanism",
        )
        self.tangent_output = nn.Linear(token_dim, 1)
        self.shape_output = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.tangent_output.weight)
        nn.init.zeros_(self.tangent_output.bias)
        nn.init.zeros_(self.shape_output.weight)
        nn.init.zeros_(self.shape_output.bias)
        self.register_buffer(
            "layer_times",
            torch.arange(1, BRIDGE_LAYERS + 1, dtype=torch.float32)
            / float(BRIDGE_LAYERS),
        )
        self.register_buffer(
            "progress_grid",
            torch.linspace(0.0, 1.0, self.progress_bins, dtype=torch.float32),
        )

    def forward(
        self,
        q0: torch.Tensor,
        q_sarn: torch.Tensor,
        progress_tokens: torch.Tensor,
        *,
        correction_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _require(
            q0.shape == q_sarn.shape
            == (progress_tokens.shape[0], self.progress_bins)
            and progress_tokens.shape
            == (q0.shape[0], self.progress_bins, self.token_dim),
            "METT bridge endpoint/token shapes differ",
        )
        _require(
            correction_available.shape == (q0.shape[0],)
            and correction_available.dtype == torch.bool
            and correction_available.device == q0.device,
            "METT bridge availability is malformed",
        )
        _validate_posterior(q0, label="METT q0")
        _validate_posterior(q_sarn, label="METT q_sarn")
        with torch.autocast(device_type=q0.device.type, enabled=False):
            raw = q0.detach().float()
            sarn = q_sarn.detach().float()
            raw_log_density = torch.log(raw.clamp_min(PROBABILITY_EPSILON))
            sarn_log_density = torch.log(sarn.clamp_min(PROBABILITY_EPSILON))
            endpoint_log_ratio = (sarn_log_density - raw_log_density).clamp(
                -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT
            )
            endpoint_center = (sarn * endpoint_log_ratio).sum(
                dim=1, keepdim=True
            )
            endpoint_direction = endpoint_log_ratio - endpoint_center

            tokens = progress_tokens.float()
            learned_tangent_logits = self.tangent_output(tokens).squeeze(2)
            learned_shape_logits = self.shape_output(tokens).squeeze(2)
            tangent_logits = learned_tangent_logits
            if not self.use_tangent:
                tangent_logits = torch.zeros_like(tangent_logits)
            tangent_summary = (sarn * tangent_logits).sum(dim=1)
            tangent_amplitude = MAX_ENDPOINT_RESIDUAL_GAIN * torch.tanh(
                tangent_summary
            )

            raw_shape_logits = learned_shape_logits
            if not self.use_shape:
                raw_shape_logits = torch.zeros_like(raw_shape_logits)
            bounded_shape = MAX_FREE_FIELD * torch.tanh(raw_shape_logits)
            grid = self.progress_grid.float()[None]
            sarn_mean, sarn_variance = _posterior_moments(sarn)
            centered_grid = grid - sarn_mean[:, None]
            shape_center = (sarn * bounded_shape).sum(dim=1, keepdim=True)
            centered_shape = bounded_shape - shape_center
            shape_linear_coefficient = (
                (sarn * centered_shape * centered_grid).sum(
                    dim=1, keepdim=True
                )
                / sarn_variance[:, None].clamp_min(1.0e-10)
            )
            orthogonal_shape = (
                centered_shape
                - shape_linear_coefficient * centered_grid
            )
            uncentered_delta = (
                tangent_amplitude[:, None] * endpoint_log_ratio
                + bounded_shape
            )
            centered_delta = (
                tangent_amplitude[:, None] * endpoint_direction
                + orthogonal_shape
            )
            effective_centered_delta = (
                self.learned_residual_scale * centered_delta
            )
            direct_scalar_shift = torch.zeros_like(sarn_mean)
            direct_scalar_target_mean = sarn_mean
            if self.use_direct_scalar_residual:
                # Parameter-matched scalar control: both 64->1 heads contribute
                # to one pooled reading residual (130 parameters in total), but
                # cannot express a free posterior shape or endpoint tangent.
                direct_summary = 0.5 * (
                    (sarn * learned_tangent_logits).sum(dim=1)
                    + (sarn * learned_shape_logits).sum(dim=1)
                )
                direct_scalar_shift = (
                    self.learned_residual_scale
                    * MAX_DIRECT_PROGRESS_RESIDUAL
                    * torch.tanh(direct_summary)
                )
                direct_scalar_target_mean = (
                    sarn_mean + direct_scalar_shift
                ).clamp(
                    torch.finfo(torch.float32).eps,
                    1.0 - torch.finfo(torch.float32).eps,
                )
                direct_tilt = torch.zeros_like(sarn_mean[:, None])
                for _ in range(MOMENT_SOLVER_STEPS):
                    direct_posterior = torch.softmax(
                        sarn_log_density + direct_tilt * grid, dim=1
                    )
                    direct_mean, direct_variance = _posterior_moments(
                        direct_posterior
                    )
                    direct_tilt = (
                        direct_tilt
                        + (
                            direct_scalar_target_mean[:, None]
                            - direct_mean[:, None]
                        )
                        / direct_variance[:, None].clamp_min(1.0e-10)
                    ).clamp(-MOMENT_SOLVER_LIMIT, MOMENT_SOLVER_LIMIT)
                effective_centered_delta = direct_tilt * centered_grid
                if self.learned_residual_scale > 0.0:
                    centered_delta = (
                        effective_centered_delta
                        / self.learned_residual_scale
                    )
                else:
                    centered_delta = torch.zeros_like(
                        effective_centered_delta
                    )
                uncentered_delta = centered_delta
                tangent_amplitude = torch.zeros_like(tangent_amplitude)
                orthogonal_shape = torch.zeros_like(orthogonal_shape)
            delta_zero = (effective_centered_delta == 0.0).all(dim=1)
            if self.use_direct_scalar_residual:
                # Preserve the exact SARN endpoint at zero initialization even
                # if Newton arithmetic leaves a sub-ULP natural tilt.
                delta_zero = direct_scalar_shift == 0.0

            proposed_tangent_base = sarn
            tangent_base = torch.where(
                correction_available[:, None], proposed_tangent_base, raw
            )
            raw_cdf = raw.cumsum(dim=1)
            raw_mean, raw_variance = _posterior_moments(raw)
            tangent_base_cdf = tangent_base.cumsum(dim=1)
            tangent_base_mean, tangent_base_variance = _posterior_moments(
                tangent_base
            )

            layer_posteriors: list[torch.Tensor] = []
            layer_cdfs: list[torch.Tensor] = []
            layer_means: list[torch.Tensor] = []
            layer_variances: list[torch.Tensor] = []
            layer_fields: list[torch.Tensor] = []
            for layer_index, layer_time in enumerate(self.layer_times):
                # The base path is Raw -> SARN.  The learned deformation enters
                # quadratically, so it is tangent-free at the Raw boundary and
                # reaches its full extrapolation only after the SARN endpoint.
                proposed_layer_field = (
                    layer_time * endpoint_direction
                    + layer_time.square() * effective_centered_delta
                )
                if layer_index == BRIDGE_LAYERS - 1:
                    proposed = torch.softmax(
                        sarn_log_density + effective_centered_delta, dim=1
                    )
                    exact_sarn_ste = (
                        proposed - proposed.detach() + proposed_tangent_base
                    )
                    proposed = torch.where(
                        delta_zero[:, None], exact_sarn_ste, proposed
                    )
                else:
                    proposed = torch.softmax(
                        raw_log_density + proposed_layer_field, dim=1
                    )
                posterior = torch.where(
                    correction_available[:, None], proposed, raw
                )
                layer_field = torch.where(
                    correction_available[:, None],
                    proposed_layer_field,
                    torch.zeros_like(proposed_layer_field),
                )
                proposed_mean, proposed_variance = _posterior_moments(posterior)
                mean_identity = proposed_mean - proposed_mean.detach() + raw_mean
                variance_identity = (
                    proposed_variance
                    - proposed_variance.detach()
                    + raw_variance
                )
                mean = torch.where(
                    correction_available, proposed_mean, mean_identity
                )
                variance = torch.where(
                    correction_available, proposed_variance, variance_identity
                )
                layer_posteriors.append(posterior)
                layer_cdfs.append(posterior.cumsum(dim=1))
                layer_means.append(mean)
                layer_variances.append(variance)
                layer_fields.append(layer_field)

            posterior_stack = torch.stack(layer_posteriors, dim=1)
            cdf_stack = torch.stack(layer_cdfs, dim=1)
            mean_stack = torch.stack(layer_means, dim=1)
            variance_stack = torch.stack(layer_variances, dim=1)
            field_stack = torch.stack(layer_fields, dim=1)
            final = posterior_stack[:, -1]
            final_cdf = cdf_stack[:, -1]
            final_mean = mean_stack[:, -1]
            final_variance = variance_stack[:, -1]
            applied_delta = torch.where(
                correction_available[:, None],
                effective_centered_delta,
                torch.zeros_like(effective_centered_delta),
            )
            field_logits = torch.stack(
                (learned_tangent_logits, learned_shape_logits), dim=2
            )
        return {
            "progress_posterior": final,
            "progress_cdf": final_cdf,
            "mean": final_mean,
            "variance": final_variance,
            "layer_posteriors": posterior_stack,
            "layer_cdfs": cdf_stack,
            "layer_means": mean_stack,
            "layer_variances": variance_stack,
            "layer_fields": field_stack,
            "field_logits": field_logits,
            "endpoint_gain": tangent_amplitude[:, None].expand_as(raw),
            "free_field": orthogonal_shape,
            "endpoint_log_ratio": endpoint_log_ratio,
            "uncentered_delta": uncentered_delta,
            "delta_center": (sarn * uncentered_delta).sum(dim=1),
            "centered_delta": centered_delta,
            "effective_centered_delta": effective_centered_delta,
            "learned_residual_scale": self.learned_residual_scale,
            "use_tangent": self.use_tangent,
            "use_shape": self.use_shape,
            "use_direct_scalar_residual": self.use_direct_scalar_residual,
            "direct_scalar_shift": direct_scalar_shift,
            "direct_scalar_target_mean": direct_scalar_target_mean,
            "delta_zero": delta_zero,
            "base_direction": endpoint_direction,
            "base_logits": sarn_log_density,
            # Compatibility aliases let the established A15.1 loss treat the
            # reached SARN endpoint as its one-sided no-harm reference.
            "proposed_geometric_base": proposed_tangent_base,
            "geometric_base": tangent_base,
            "geometric_base_cdf": tangent_base_cdf,
            "geometric_base_mean": tangent_base_mean,
            "geometric_base_variance": tangent_base_variance,
            "raw_log_density": raw_log_density,
            "raw_cdf": raw_cdf,
            "raw_mean": raw_mean,
            "raw_variance": raw_variance,
            "path_field_energy": (
                posterior_stack * field_stack.square()
            ).sum(dim=2),
            "learned_delta_energy": (sarn * applied_delta.square()).sum(dim=1),
            "tangent_amplitude": tangent_amplitude,
            "shape_center": shape_center.squeeze(1),
            "shape_linear_coefficient": shape_linear_coefficient.squeeze(1),
        }


class A152METTCorrection(A15FTEBCorrection):
    """A15.2 correction with endpoint-tangent natural-parameter transport."""

    def __init__(
        self,
        *,
        stride8_channels: int = RAW_STRIDE8_CHANNELS,
        stride16_channels: int = EFFICIENTNET_B0_MIDDLE_FEATURES,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = DEFAULT_MEMORY_GRID_SIZE,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        use_relation_memory: bool = True,
        use_tangent: bool = True,
        use_shape: bool = True,
        use_direct_scalar_residual: bool = False,
    ) -> None:
        super().__init__(
            stride8_channels=stride8_channels,
            stride16_channels=stride16_channels,
            relation_channels=relation_channels,
            token_dim=token_dim,
            attention_heads=attention_heads,
            decoder_layers=decoder_layers,
            memory_grid_size=memory_grid_size,
            progress_bins=progress_bins,
            learned_residual_scale=A15_2_LEARNED_RESIDUAL_SCALE,
            use_relation_memory=use_relation_memory,
        )
        self.natural_parameter_bridge = EndpointTangentNaturalParameterBridge(
            progress_bins=progress_bins,
            token_dim=token_dim,
            learned_residual_scale=A15_2_LEARNED_RESIDUAL_SCALE,
            use_tangent=use_tangent,
            use_shape=use_shape,
            use_direct_scalar_residual=use_direct_scalar_residual,
        )

    def forward(self, *args: object, **kwargs: object) -> dict[str, Any]:
        output = super().forward(*args, **kwargs)
        if self.natural_parameter_bridge.use_direct_scalar_residual:
            output["architecture"] = REMST_ARCHITECTURE
            output["publication_model"] = remst_publication_identity()
            output["legacy_checkpoint_architecture"] = (
                A15_2_METT_ARCHITECTURE
            )
        else:
            output["architecture"] = A15_2_METT_ARCHITECTURE
        output["proposed_tangent_base"] = output["proposed_geometric_base"]
        output["tangent_base"] = output["geometric_base"]
        output["tangent_base_cdf"] = output["geometric_base_cdf"]
        output["tangent_base_mean"] = output["geometric_base_mean"]
        output["tangent_base_variance"] = output["geometric_base_variance"]
        return output


class ReMSTCorrection(A152METTCorrection):
    """Publication-facing canonical ReMST correction architecture.

    The subclass fixes the selected mechanism to scalar first-moment
    transport.  Its state-dict layout is identical to the legacy
    ``A152METTCorrection(..., use_direct_scalar_residual=True)`` construction.
    """

    def __init__(
        self,
        *,
        stride8_channels: int = RAW_STRIDE8_CHANNELS,
        stride16_channels: int = EFFICIENTNET_B0_MIDDLE_FEATURES,
        relation_channels: int = DEFAULT_RELATION_CHANNELS,
        token_dim: int = DEFAULT_TOKEN_DIM,
        attention_heads: int = 4,
        decoder_layers: int = 2,
        memory_grid_size: int = DEFAULT_MEMORY_GRID_SIZE,
        progress_bins: int = DEFAULT_PROGRESS_BINS,
        use_relation_memory: bool = True,
    ) -> None:
        super().__init__(
            stride8_channels=stride8_channels,
            stride16_channels=stride16_channels,
            relation_channels=relation_channels,
            token_dim=token_dim,
            attention_heads=attention_heads,
            decoder_layers=decoder_layers,
            memory_grid_size=memory_grid_size,
            progress_bins=progress_bins,
            use_relation_memory=use_relation_memory,
            use_tangent=False,
            use_shape=False,
            use_direct_scalar_residual=True,
        )


__all__ = [
    "A15_2_METT_ARCHITECTURE",
    "A152METTCorrection",
    "DEFAULT_POSTERIOR_SCALE",
    "EndpointTangentNaturalParameterBridge",
    "MAX_DIRECT_PROGRESS_RESIDUAL",
    "MAX_POSTERIOR_SCALE",
    "MIN_POSTERIOR_SCALE",
    "MomentExactEfficientNetB0Anchor",
    "MomentExactPosteriorHead",
    "REMST_ARCHITECTURE",
    "REMST_FULL_NAME",
    "REMST_LEGACY_VARIANT",
    "REMST_SHORT_NAME",
    "ReMSTCorrection",
    "load_moment_exact_anchor_from_direct_checkpoint",
    "remst_publication_identity",
]
