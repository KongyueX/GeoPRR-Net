"""RGB-only joint relative-angle and scale-span reader."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_model import GeoPRRV2


def relative_targets(delta: torch.Tensor, span: torch.Tensor, bins: int):
    d = delta.float().remainder(2 * math.pi) * bins / (2 * math.pi)
    dl = d.floor().long()
    du = (dl + 1) % bins
    dw = d - dl
    l = span.float().clamp(0, 2 * math.pi) * (bins - 1) / (2 * math.pi)
    ll = l.floor().long()
    lu = (ll + 1).clamp(max=bins - 1)
    lw = l - ll
    return dl, du, dw, ll, lu, lw


def relative_state_nll(output, delta: torch.Tensor, span: torch.Tensor):
    joint = output["joint"]
    dl, du, dw, ll, lu, lw = relative_targets(delta, span, joint.shape[-1])
    batch = torch.arange(len(delta), device=joint.device)
    nll = joint.new_zeros(len(delta))
    for di, wi in ((dl, 1 - dw), (du, dw)):
        for li, wj in ((ll, 1 - lw), (lu, lw)):
            nll -= wi * wj * joint[batch, di, li].clamp_min(1e-12).log()
    return nll.mean()


class RelativeStateReadout(nn.Module):
    """Push all joint state mass through one frozen ratio readout."""

    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        if bins < 4:
            raise ValueError("relative-state grid needs at least four bins")
        delta = torch.arange(bins, dtype=torch.float64) * (2 * math.pi / bins)
        span = torch.linspace(0, 2 * math.pi, bins, dtype=torch.float64)
        invalid = torch.zeros(bins, bins, dtype=torch.bool)
        invalid[:, 0] = invalid[:, -1] = True
        safe_span = span.clone()
        safe_span[0] = 1.
        unwrapped = torch.remainder(delta[:, None] - span[None, :] / 2 + math.pi, 2 * math.pi) - math.pi + span[None, :] / 2
        raw = unwrapped / safe_span[None, :]
        outside = ((raw < 0) | (raw > 1)) & ~invalid
        progress = raw.clamp(0, 1) * (bins - 1)
        lower = progress.floor().long()
        upper = (lower + 1).clamp(max=bins - 1)
        fraction = progress - lower
        self.bins = bins
        self.register_buffer("invalid_atoms", invalid.flatten())
        self.register_buffer("outside_atoms", outside.flatten())
        self.register_buffer("lower_indices", lower.flatten())
        self.register_buffer("upper_indices", upper.flatten())
        self.register_buffer("upper_weights", fraction.flatten().float())
        self.register_buffer("bin_coordinates", torch.linspace(0, 1, bins))

    def forward(self, joint: torch.Tensor):
        mass = joint.float().flatten(1)
        invalid = mass[:, self.invalid_atoms].sum(1)
        outside = mass[:, self.outside_atoms].sum(1)
        valid_mass = mass.masked_fill(self.invalid_atoms[None, :], 0)
        distribution = mass.new_zeros((len(mass), self.bins))
        distribution.scatter_add_(1, self.lower_indices[None, :].expand(len(mass), -1), valid_mass * (1 - self.upper_weights))
        distribution.scatter_add_(1, self.upper_indices[None, :].expand(len(mass), -1), valid_mass * self.upper_weights)
        distribution = distribution + invalid[:, None] / self.bins
        return {"distribution": distribution,
                "reading": (distribution * self.bin_coordinates).sum(1).clamp(0, 1),
                "invalid_mass": invalid, "outside_arc_mass": outside}


class RelativeStateHead(nn.Module):
    def __init__(self, channels: int, global_channels: int, bins: int, components: int):
        super().__init__()
        self.components = components
        self.attention = nn.Conv2d(channels + 2, 2 * components, 1)
        self.norm = nn.LayerNorm(channels + 2 + global_channels)
        self.roles = nn.ModuleList(nn.Linear(channels + 2 + global_channels, bins) for _ in range(2))
        self.mixture = nn.Linear(global_channels, components)
        for layer in (*self.roles, self.mixture):
            nn.init.normal_(layer.weight, std=.001)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, global_feature):
        height, width = feature.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype),
                                torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype), indexing="ij")
        xy = torch.stack((xx, yy))[None].expand(len(feature), -1, -1, -1)
        spatial = torch.cat((feature, xy), dim=1)
        weights = self.attention(spatial).float().flatten(2).softmax(-1)
        pooled = torch.bmm(weights, spatial.float().flatten(2).transpose(1, 2))
        pooled = pooled.reshape(len(feature), 2, self.components, -1)
        context = global_feature.float()[:, None, None, :].expand(-1, 2, self.components, -1)
        inputs = self.norm(torch.cat((pooled, context), dim=-1))
        probabilities = [role(inputs[:, index]).float().softmax(-1) for index, role in enumerate(self.roles)]
        return self.mixture(global_feature).float().softmax(-1), *probabilities


class RelativeStateReader(nn.Module):
    def __init__(self, bins=256, decoder_dim=64, components=2, coupling="joint"):
        super().__init__()
        if coupling not in ("joint", "independent"):
            raise ValueError("coupling must be joint or independent")
        base = GeoPRRV2(backbone="mobilenetv4_conv_medium.e500_r224_in1k", pretrained=False,
                        bins=bins, decoder_dim=decoder_dim, variant="field")
        self.backbone = base.backbone
        self.lateral = base.lateral
        self.refine = base.refine
        self.appearance_norm = base.appearance_norm
        self.feature_indices = base.feature_indices
        self.state_head = RelativeStateHead(decoder_dim, base.feature_channels[-1], bins, components)
        self.readout = RelativeStateReadout(bins)
        self.bins, self.coupling = bins, coupling
        self._shared_trainable = True

    def shared_parameters(self):
        for module in (self.backbone, self.lateral, self.refine, self.appearance_norm):
            yield from module.parameters()

    def initialize_shared(self, checkpoint: str | Path):
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = source["model"]
        prefixes = ("backbone.", "lateral.", "refine.", "appearance_norm.")
        selected = {k: v for k, v in saved.items() if k.startswith(prefixes)}
        expected = {k for k in self.state_dict() if k.startswith(prefixes)}
        if set(selected) != expected:
            raise ValueError("initial checkpoint shared RGB modules differ")
        result = self.load_state_dict(selected, strict=False)
        if result.unexpected_keys or any(k.startswith(prefixes) for k in result.missing_keys):
            raise ValueError("shared checkpoint loading failed")
        return {"checkpoint": str(checkpoint), "source_epoch": int(source["epoch"]),
                "source_protocol": source["protocol"], "loaded_shared_tensors": len(selected),
                "new_head": "random initialization under the declared seed"}

    def set_shared_trainable(self, trainable: bool):
        self._shared_trainable = bool(trainable)
        for module in (self.backbone, self.lateral, self.refine, self.appearance_norm):
            module.requires_grad_(trainable)
            module.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self._shared_trainable:
            for module in (self.backbone, self.lateral, self.refine, self.appearance_norm):
                module.eval()
        return self

    def forward(self, images):
        extracted = self.backbone(images)
        features = [extracted[i] for i in self.feature_indices]
        global_feature = self.appearance_norm(features[-1].mean((2, 3)))
        projected = [layer(feature) for layer, feature in zip(self.lateral, features)]
        decoded = self.refine[-1](projected[-1])
        for index in (2, 1, 0):
            decoded = self.refine[index](F.interpolate(decoded, size=projected[index].shape[-2:],
                                                       mode="bilinear", align_corners=False) + projected[index])
        mixture, delta, span = self.state_head(decoded, global_feature)
        with torch.autocast(device_type=images.device.type, enabled=False):
            if self.coupling == "joint":
                joint = torch.einsum("br,brd,brl->bdl", mixture.float(), delta.float(), span.float())
            else:
                marginal_delta = torch.einsum("br,brd->bd", mixture.float(), delta.float())
                marginal_span = torch.einsum("br,brl->bl", mixture.float(), span.float())
                joint = marginal_delta[:, :, None] * marginal_span[:, None, :]
            readout = self.readout(joint)
        return {"joint": joint, "mixture_probs": mixture, "delta_probs": delta, "span_probs": span,
                **readout}
