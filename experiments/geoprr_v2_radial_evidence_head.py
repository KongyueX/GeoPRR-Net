"""Empirical ordered feature queries in a learned dial coordinate."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RadialEvidenceHead(nn.Module):
    """Accept F4 and pooled context only; rays are not physical needle paths."""

    def __init__(self, global_dim, parity):
        super().__init__()
        if parity not in (-1, 1):
            raise ValueError("parity must be the single frozen coordinate-frame sign")
        self.parity = int(parity)
        self.context = nn.Sequential(nn.Linear(global_dim, 64), nn.SiLU(), nn.Linear(64, 6))
        nn.init.normal_(self.context[-1].weight, std=.001)
        nn.init.zeros_(self.context[-1].bias)
        with torch.no_grad():
            self.context[-1].bias[5] = math.log(3.)
        self.kernel = nn.Parameter(torch.randn(16, 64) / 8)
        self.register_buffer("bins", torch.linspace(0., 1., 256))
        self.register_buffer("radii", torch.arange(1., 17.) / 17)
        self.register_buffer("frame", torch.diag(torch.tensor([1., float(parity)])))

    def coordinates(self, global_features):
        raw = self.context(global_features)
        center = raw[:, :2].sigmoid()
        phi, stretch, shear = raw[:, 2], raw[:, 3], raw[:, 4]
        cosine, sine = phi.cos(), phi.sin()
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
        zero = torch.zeros_like(stretch)
        triangular = torch.stack((stretch.exp(), shear, zero, (-stretch).exp()), dim=-1).reshape(-1, 2, 2)
        matrix = rotation @ triangular @ self.frame
        span = 2 * math.pi * raw[:, 5].sigmoid()
        angle = span[:, None] * self.bins
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        rays = torch.linalg.solve(matrix, direction.transpose(1, 2)).transpose(1, 2)
        rays = F.normalize(rays, dim=-1)
        nonzero = rays != 0
        divisor = torch.where(nonzero, rays, torch.ones_like(rays))
        distance = torch.where(rays > 0, (1-center[:, None]) / divisor, -center[:, None] / divisor)
        distance = torch.where(nonzero, distance, torch.full_like(distance, float("inf")))
        boundary = distance.amin(dim=-1)
        points = center[:, None, None] + self.radii[None, None, :, None] * boundary[:, :, None, None] * rays[:, :, None]
        return points, {"center": center, "matrix": matrix, "span": span,
                        "boundary_distance": boundary, "parity": self.parity}

    def forward(self, f4, global_features):
        points, calibration = self.coordinates(global_features)
        grid = 2 * points - 1
        sampled = F.grid_sample(f4, grid, mode="bilinear", padding_mode="border", align_corners=False)
        logits = torch.einsum("bdkj,jd->bk", sampled, self.kernel) / 16
        distribution = logits.float().softmax(dim=-1)
        reading = (distribution * self.bins).sum(dim=-1)
        return {"distribution": distribution, "reading": reading, "logits": logits,
                "query_grid": grid, "query_points": points, "calibration": calibration}
