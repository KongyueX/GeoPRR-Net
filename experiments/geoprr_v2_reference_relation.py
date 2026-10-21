"""RGB-only reference-conditioned circular evidence and full-mass reading."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from experiments.geoprr_v2_data import GeoPRRV2Dataset, normalized_target
from experiments.geoprr_v2_model import GeoPRRV2, distribution_cdf_loss


def circular_two_hot_targets(phases, bins=256):
    positions = phases.remainder(2*math.pi)*bins/(2*math.pi)
    lower = positions.floor().long() % bins
    fraction = positions-positions.floor()
    labels = positions.new_zeros((*positions.shape, bins))
    labels.scatter_add_(-1, lower[..., None], (1-fraction)[..., None])
    labels.scatter_add_(-1, ((lower+1) % bins)[..., None], fraction[..., None])
    return labels


def reference_geometry_targets(homography, target, span, image_size=768, bins=256):
    """Training-only polar labels; pointer phase denotes the virtual dial point."""
    h = homography.double()
    a = image_size*h[:, :2, :2]
    b = h[:, :2, 2]-.5*(h[:, :2, 0]+h[:, :2, 1])
    center = torch.linalg.solve(a, -b[..., None]).squeeze(-1)
    denominator = (h[:, 2, :2]*(image_size*center-.5)).sum(-1)+h[:, 2, 2]
    a = a*torch.where(denominator > 0, 1., -1.)[:, None, None]
    determinant = torch.linalg.det(a)
    if bool((determinant <= 0).any()) or bool(((center <= 0) | (center >= 1)).any()):
        raise ValueError("source target requires positive angular parity and an interior center")
    phi = torch.atan2(a[:, 1, 0]-a[:, 0, 1], a[:, 0, 0]+a[:, 1, 1])
    cosine, sine = phi.cos(), phi.sin()
    rotation = torch.stack((cosine, -sine, sine, cosine), -1).reshape(-1, 2, 2)
    spd = rotation.transpose(1, 2) @ (a/determinant.sqrt()[:, None, None])
    spd = (spd+spd.transpose(1, 2))/2
    spd = spd/torch.linalg.det(spd).sqrt()[:, None, None]
    chol = torch.linalg.cholesky(spd)
    raw = torch.cat((center.log()-torch.log1p(-center), chol[:, 0, 0].log()[:, None], chol[:, 1, 0, None]), 1)
    span, target = span.double().reshape(-1), target.double().reshape(-1)
    phases = torch.stack((-phi, -phi+span*target, -phi+span), 1)
    return {"center": center, "spd": spd, "rectifier_raw": raw, "phi": phi,
            "phases": phases, "role_twohot": circular_two_hot_targets(phases, bins), "span": span}


class ReferenceRelationDataset(Dataset):
    """Frozen donor/recipient exposure with existing geometry labels, no pair loss."""

    def __init__(self, manifest, *, data_root, data_config, augment, pair_manifest=None):
        self.base = GeoPRRV2Dataset(manifest, dataset_root=data_root, augment=augment,
                                   require_geometry=True, pair_manifest=None, **data_config)
        self.rows, self.augment = self.base.rows, augment
        by_id = {row["sample_id"]: index for index, row in enumerate(self.rows)}
        partners, edges = {}, set()
        pairs = [json.loads(line) for line in Path(pair_manifest).read_text().splitlines() if line.strip()] if pair_manifest else []
        for pair in pairs:
            a, b = pair["sample_a"], pair["sample_b"]
            if a not in by_id or b not in by_id or a == b or (a, b) in edges:
                raise ValueError("invalid or duplicate source exposure edge")
            edges.add((a, b))
            partners.setdefault(a, []).append(by_id[b])
        self.items = [(index, partner) for index, row in enumerate(self.rows)
                      for partner in partners.get(row["sample_id"], [None])]
        self.edges = sorted(edges)
        self.unique_image_count, self.item_count = len(self.rows), len(self.items)

    def supervision_contract(self):
        """Freeze the actual inputs used by calibration/targets for resume checks."""
        fields = ("homography", "homography_bbox", "dial_bbox", "long_interval_degree", "long_num", "pointer_angle",
                  "image_width", "image_height", "scene_name")
        rows = []
        for row in self.rows:
            metadata = row.get("metadata", {})
            geometry = {key: metadata[key] for key in fields if key in metadata}
            geometry["ScaleMark_keypoints"] = [{"type": item.get("type"), "all_kp": item.get("all_kp")}
                for item in metadata.get("keypoints", []) if str(item.get("type", "")).lower() == "scalemark"]
            rows.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                         "y": normalized_target(row), "image_path": row.get("image_path"),
                         "roi_path": row.get("roi_path"), "geometry": geometry})
        return deepcopy({"schema": "reference_relation_geometry_supervision_v1", "rows": rows, "edges": self.edges,
                         "image_size": self.base.image_size, "output_size": self.base.output_size,
                         "geometry_config": vars(self.base.geometry_config), "unique_images": len(self.rows),
                         "donor_items": len(self.items), "recipient_items": len(self.edges)})

    def __len__(self):
        return len(self.items)

    def _sample(self, index, photometric):
        row = self.rows[index]
        try:
            sample = self.base._sample(index, photometric)
            sample["span_radians"] = torch.tensor(self.base._calibrations[index]["angles"][-1], dtype=torch.float32)
            sample["input_failure"] = ""
        except (OSError, ValueError) as error:
            if self.augment:
                raise
            sample = {"image": torch.zeros(3, self.base.image_size, self.base.image_size),
                      "y": torch.tensor(normalized_target(row)), "sample_id": row["sample_id"], "group_id": row["group_id"],
                      "roi_to_canonical": torch.eye(3), "span_radians": torch.tensor(1.),
                      "field_target": torch.zeros(self.base.output_size, self.base.output_size),
                      "field_mask": torch.zeros(1, self.base.output_size, self.base.output_size),
                      "input_failure": f"{type(error).__name__}: {error}"}
        return {key: sample[key] for key in ("image", "y", "sample_id", "group_id", "input_failure",
                                             "roi_to_canonical", "field_target", "field_mask", "span_radians")}

    def __getitem__(self, index):
        donor, recipient = self.items[index]
        photo = self.base._photometric() if self.augment else None
        sample = self._sample(donor, photo)
        if self.augment:
            partner = self._sample(recipient, photo) if recipient is not None else sample
            sample.update({f"pair_{key}": value for key, value in partner.items()})
            sample["pair_valid"] = recipient is not None
        return sample


class CircularProgressPushforward(nn.Module):
    """Keep every circular state, including uniform transport of zero-span mass."""

    def __init__(self, bins=256, progress_bins=256):
        super().__init__()
        delta = torch.arange(bins, dtype=torch.float64)*2*math.pi/bins
        span = delta[None]
        invalid = (torch.arange(bins)[None].expand(bins, -1) == 0)
        raw = ((delta[:, None]-span/2+math.pi).remainder(2*math.pi)-math.pi+span/2)/span.masked_fill(invalid, 1.)
        progress = raw.clamp(0, 1)*(progress_bins-1)
        lower = progress.floor().long()
        self.register_buffer("invalid_atoms", invalid.flatten())
        self.register_buffer("outside_atoms", (((raw < 0) | (raw > 1)) & ~invalid).flatten())
        self.register_buffer("lower", lower.flatten())
        self.register_buffer("upper", (lower+1).clamp(max=progress_bins-1).flatten())
        self.register_buffer("fraction", (progress-lower).flatten().float())
        self.register_buffer("coordinates", torch.linspace(0, 1, progress_bins))

    def forward(self, joint):
        mass = joint.float().flatten(1)
        invalid_mass = mass[:, self.invalid_atoms].sum(1)
        valid = mass.masked_fill(self.invalid_atoms[None], 0)
        distribution = mass.new_zeros((len(mass), len(self.coordinates)))
        distribution.scatter_add_(1, self.lower[None].expand_as(mass), valid*(1-self.fraction))
        distribution.scatter_add_(1, self.upper[None].expand_as(mass), valid*self.fraction)
        distribution += invalid_mass[:, None]/len(self.coordinates)
        return {"distribution": distribution, "reading": (distribution*self.coordinates).sum(1).clamp(0, 1),
                "invalid_mass": invalid_mass, "outside_arc_mass": mass[:, self.outside_atoms].sum(1)}


class ReferenceRelationHead(nn.Module):
    """Content matching conditions pointer/end rows on each start-bin hypothesis."""

    def __init__(self, feature_dim=16, bins=256, reference_normalization="local"):
        super().__init__()
        if reference_normalization not in ("local", "global"):
            raise ValueError("reference normalization must be local or global")
        self.reference_normalization = reference_normalization
        self.projections = nn.ModuleList(nn.Linear(feature_dim, feature_dim, bias=False) for _ in range(3))
        self.unaries = nn.ModuleList(nn.Linear(feature_dim, 1, bias=False) for _ in range(3))
        self.scale = feature_dim**-.5
        self.register_buffer("relative_indices", (torch.arange(bins)[:, None]+torch.arange(bins)[None]) % bins)

    def forward(self, features):
        roles = [projection(features) for projection in self.projections]
        a, p, b = roles
        unary = [layer(role).squeeze(-1) for layer, role in zip(self.unaries, roles)]
        energy_p = (a @ p.transpose(1, 2))*self.scale+unary[1][:, None]
        energy_b = (a @ b.transpose(1, 2))*self.scale+unary[2][:, None]
        log_z_p, log_z_b = energy_p.logsumexp(-1), energy_b.logsumexp(-1)
        reference = unary[0]+log_z_p+log_z_b if self.reference_normalization == "global" else unary[0]
        log_a = reference.log_softmax(-1)
        indices = self.relative_indices[None].expand(len(features), -1, -1)
        log_p = energy_p.log_softmax(-1).gather(-1, indices)
        log_b = energy_b.log_softmax(-1).gather(-1, indices)
        start, pointer, end = log_a.exp(), log_p.exp(), log_b.exp()
        joint = pointer.transpose(1, 2) @ (start[:, :, None]*end)
        delta, span = (start[:, :, None]*pointer).sum(1), (start[:, :, None]*end).sum(1)
        return {"A": start, "P_rel": pointer, "B_rel": end, "log_A": log_a, "log_P_rel": log_p, "log_B_rel": log_b,
                "J": joint, "product": delta[:, :, None]*span[:, None], "delta_marginal": delta, "span_marginal": span,
                "reference_unary": unary[0], "log_Z_P": log_z_p, "log_Z_B": log_z_b}


class ReferenceRelationStateHead(nn.Module):
    """Angular line-pencil rectifier and empirical boundary-to-center feature rays."""

    def __init__(self, bins=256, channels=64, token_dim=16, reference_normalization="local"):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(3*channels, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.normal_(self.context[-1].weight, std=.001)
        nn.init.zeros_(self.context[-1].bias)
        self.radial_embedding = nn.Parameter(torch.randn(16, channels, token_dim)/math.sqrt(channels))
        self.relation = ReferenceRelationHead(token_dim, bins, reference_normalization)
        angle = torch.arange(bins)*2*math.pi/bins
        self.register_buffer("directions", torch.stack((angle.cos(), angle.sin()), -1))
        self.register_buffer("radii", torch.arange(1., 17.)/17)

    def forward(self, features, context):
        raw = self.context(context.float())
        center, stretch, shear = raw[:, :2].sigmoid(), raw[:, 2], raw[:, 3]
        zero = torch.zeros_like(stretch)
        chol = torch.stack((stretch.exp(), zero, shear, (-stretch).exp()), -1).reshape(-1, 2, 2)
        spd = chol @ chol.transpose(1, 2)
        rays = torch.linalg.solve(spd, self.directions.T.expand(len(raw), -1, -1)).transpose(1, 2)
        rays = F.normalize(rays, dim=-1)
        divisor = torch.where(rays != 0, rays, torch.ones_like(rays))
        distances = torch.where(rays > 0, (1-center[:, None])/divisor, -center[:, None]/divisor)
        distances = torch.where(rays != 0, distances, torch.full_like(distances, float("inf")))
        points = center[:, None, None]+self.radii[None, None, :, None]*distances.amin(-1)[:, :, None, None]*rays[:, :, None]
        sampled = F.grid_sample(features.float(), 2*points-1, mode="bilinear", padding_mode="border", align_corners=False)
        tokens = torch.einsum("bckr,rcd->bkd", sampled, self.radial_embedding)/16
        return {**self.relation(tokens), "role_tokens": tokens, "query_points": points,
                "calibration": {"center": center, "spd": spd, "raw": raw}}


class ReferenceRelationReader(nn.Module):
    def __init__(self, bins=256, decoder_dim=64, token_dim=16, coupling="joint", reference_normalization="local"):
        super().__init__()
        if (bins, decoder_dim, token_dim) != (256, 64, 16) or coupling not in ("joint", "independent"):
            raise ValueError("requires the frozen256/64/16 architecture and joint or independent coupling")
        base = GeoPRRV2(backbone="mobilenetv4_conv_medium.e500_r224_in1k", pretrained=False, bins=bins,
                        decoder_dim=decoder_dim, variant="field")
        self.backbone, self.lateral, self.refine = base.backbone, base.lateral, base.refine
        self.feature_indices = base.feature_indices
        self.state_head = ReferenceRelationStateHead(bins, decoder_dim, token_dim, reference_normalization)
        self.readout = CircularProgressPushforward(bins)
        self.bins, self.coupling, self._shared_trainable = bins, coupling, True

    def shared_parameters(self):
        for module in (self.backbone, self.lateral, self.refine):
            yield from module.parameters()

    def set_shared_trainable(self, trainable):
        self._shared_trainable = bool(trainable)
        for module in (self.backbone, self.lateral, self.refine):
            module.requires_grad_(trainable)
            module.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self._shared_trainable:
            for module in (self.backbone, self.lateral, self.refine):
                module.eval()
        return self

    def initialize_shared(self, checkpoint):
        source = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        prefixes = ("backbone.", "lateral.", "refine.")
        selected = {key: value for key, value in source["model"].items() if key.startswith(prefixes)}
        if set(selected) != {key for key in self.state_dict() if key.startswith(prefixes)}:
            raise ValueError("ordinary checkpoint shared RGB modules differ")
        self.load_state_dict(selected, strict=False)
        return {"source_epoch": int(source["epoch"]), "source_protocol": source["protocol"],
                "loaded_shared_tensors": len(selected), "shared_prefixes": prefixes,
                "new_head": "reference rectifier/radial embedding/content matching; no old U/Q/M/direct heads"}

    def initialize_full(self, checkpoint):
        source = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if source["epoch"] != 57 or source["config"]["training"]["seed"] != 20262020:
            raise ValueError("feedback initialization requires the frozen best57 seed20262020")
        self.load_state_dict(source["model"], strict=True)
        return {"source_epoch": 57, "source_protocol": source["protocol"],
                "loaded_full_tensors": len(source["model"]), "preserved_all_trained_weights": True}

    def forward(self, images):
        extracted = self.backbone(images)
        features = [extracted[index] for index in self.feature_indices]
        projected = [layer(feature) for layer, feature in zip(self.lateral, features)]
        context = torch.cat([feature.mean((2, 3)) for feature in projected[1:]], 1)
        decoded = self.refine[-1](projected[-1])
        for index in (2, 1, 0):
            decoded = self.refine[index](F.interpolate(decoded, size=projected[index].shape[-2:], mode="bilinear",
                                                       align_corners=False)+projected[index])
        with torch.autocast(device_type=images.device.type, enabled=False):
            output = self.state_head(decoded, context)
            joint = output["J"] if self.coupling == "joint" else output["product"]
            other = output["product"] if self.coupling == "joint" else output["J"]
            with torch.no_grad():
                counterfactual = self.readout(other.detach())["reading"]
            return {**output, "joint": joint, **self.readout(joint), "global_features": context,
                    "decoded_features": decoded, "reference_entropy": -(output["A"]*output["log_A"]).sum(-1),
                    "reference_max": output["A"].amax(-1),
                    "joint_vs_product_state_l1": (output["J"]-output["product"]).abs().sum((1, 2)),
                    "counterfactual_reading": counterfactual}


def reference_relation_losses(output, batch):
    targets = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"],
                                          image_size=batch["image"].shape[-1])
    labels = targets["role_twohot"].float()
    role_nll = -(labels[:, 0]*output["log_A"]).sum(-1)
    bins = labels.shape[-1]
    phases = targets["phases"][:, 0]*bins/(2*math.pi)
    low = phases.floor().long() % bins
    fraction = (phases-phases.floor()).float()
    batch_index, absolute = torch.arange(len(low), device=low.device), torch.arange(bins, device=low.device)
    for reference, weight in ((low, 1-fraction), ((low+1) % bins, fraction)):
        relative = (absolute[None]-reference[:, None]) % bins
        for key, role in (("log_P_rel", 1), ("log_B_rel", 2)):
            role_nll -= weight*(labels[:, role]*output[key][batch_index, reference].gather(1, relative)).sum(-1)
    raw = role_nll.mean()
    height, width = batch["field_target"].shape[-2:]
    yy, xx = torch.meshgrid((torch.arange(height, device=low.device)+.5)/height,
                            (torch.arange(width, device=low.device)+.5)/width, indexing="ij")
    xy = torch.stack((xx, yy), -1)[None]-output["calibration"]["center"][:, None, None]
    vectors = torch.einsum("bij,bhwj->bhwi", output["calibration"]["spd"], xy)
    phase = (torch.atan2(vectors[..., 1], vectors[..., 0])+targets["phi"].float()[:, None, None]).remainder(2*math.pi)
    progress = (phase/targets["span"].float()[:, None, None]).clamp(0, 1)
    mask = batch["field_mask"][:, 0].float()
    field_l1 = ((progress-batch["field_target"].float()).abs()*mask).sum()/mask.sum().clamp_min(1)
    cdf = distribution_cdf_loss(output["distribution"], batch["y"])
    normalized = raw/(3*math.log(bins))
    return {"role_nll_raw": raw, "role_nll_normalized": normalized, "field_l1": field_l1,
            "reading_cdf": cdf, "total": normalized+field_l1+cdf}
