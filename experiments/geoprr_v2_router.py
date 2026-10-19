"""Image-only reliability routing between a shared reader's two experts."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROUTER_PROTOCOL = "geoprr_v2_oof_risk_router_v1"
FEATURE_PROTOCOL = "prediction_statistics_and_image_context_v1"


def build_router_features(output: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Use predictions and image context only; never consume labels or geometry."""
    field, direct = output["field_distribution"].float(), output["appearance_probs"].float()
    coordinates = torch.linspace(0, 1, field.shape[1], device=field.device)
    statistics = []
    for probabilities in (field, direct):
        mean = (probabilities * coordinates).sum(1)
        variance = (probabilities * (coordinates - mean[:, None]).square()).sum(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1) / math.log(field.shape[1])
        peaks = probabilities.topk(2, dim=1).values
        statistics.extend((mean, variance, entropy, peaks[:, 0], peaks[:, 0] - peaks[:, 1]))
    statistics.extend(((output["field_reading"] - output["appearance_reading"]).abs(),
                       (field.cumsum(1) - direct.cumsum(1)).abs().mean(1),
                       output["support_mass"].float(), output["field_valid"].float()))
    point = output["point_probs"].float().flatten(1)
    statistics.extend((point.max(1).values, -(point * point.clamp_min(1e-8).log()).sum(1) / math.log(point.shape[1])))
    return torch.cat((torch.stack(statistics, 1), output["global_features"].float()), 1).detach()


class ReliabilityRouter(nn.Module):
    """Predict each expert's absolute error in percentage of full scale."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(hidden_dim, 2), nn.Softplus())
        self.reader_config: dict[str, Any] = {}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features.float() - self.feature_mean) / self.feature_std)


def apply_expert_mode(output: Mapping[str, torch.Tensor], expert_mode: str = "field", *,
                      router: ReliabilityRouter | None = None, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Select an entire expert distribution; oracle mode is diagnostic only."""
    if expert_mode not in ("field", "direct", "mean", "oracle", "router"):
        raise ValueError(f"unknown expert mode: {expert_mode}")
    valid = output["field_valid"].bool()
    predictions = torch.stack((output["field_reading"], output["appearance_reading"]), 1)
    risks = None
    if router is not None:
        risks = router(build_router_features(output))
        selected = risks.argmin(1)
    elif expert_mode == "router":
        raise ValueError("router mode requires a trained router")
    elif expert_mode == "oracle":
        if target is None:
            raise ValueError("diagnostic oracle mode requires target labels")
        selected = (predictions - target.reshape(-1, 1)).abs().argmin(1)
    else:
        selected = torch.full_like(valid, int(expert_mode == "direct"), dtype=torch.long)
    selected = torch.where(valid, selected, torch.ones_like(selected))
    distributions = torch.stack((output["field_distribution"], output["appearance_probs"]), 1)
    rows = torch.arange(len(selected), device=selected.device)
    distribution, reading = distributions[rows, selected], predictions[rows, selected]
    if expert_mode == "mean" and router is None:
        distribution = torch.where(valid[:, None], distributions.mean(1), output["appearance_probs"])
        reading = torch.where(valid, predictions.mean(1), output["appearance_reading"])
        selected = torch.where(valid, torch.full_like(selected, -1), selected)
    result = dict(output, distribution=distribution, reading=reading, expert_index=selected)
    if risks is not None:
        result["predicted_risk_percent_fs"] = risks
    return result


def load_router(path: str | Path, device: str | torch.device = "cpu") -> ReliabilityRouter:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != ROUTER_PROTOCOL or checkpoint.get("feature_protocol") != FEATURE_PROTOCOL:
        raise ValueError("unsupported router checkpoint protocol")
    router = ReliabilityRouter(checkpoint["feature_dim"], checkpoint["hidden_dim"]).to(device)
    router.load_state_dict(checkpoint["model"], strict=True)
    router.reader_config = checkpoint["reader_config"]
    router.reader_seed = checkpoint["seed"]
    return router.eval()
