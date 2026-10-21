"""Exact three-role association and its matched marginal-product deletion."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationReader, reference_geometry_targets, reference_relation_losses,
)


def pair_log_factor(hP, hB, W, association="full"):
    """Normalize one bilinear P--B factor, optionally retaining only its marginals."""
    energy = (hP @ W @ hB.transpose(1, 2))/math.sqrt(hP.shape[-1])
    log_factor = energy.flatten(1).log_softmax(-1).reshape_as(energy)
    if association == "full":
        return log_factor
    if association == "separable":
        return log_factor.logsumexp(-1)[:, :, None]+log_factor.logsumexp(-2)[:, None, :]
    raise ValueError("association must be full or separable")


def _block_log_weight(log_a, log_p, log_b, log_factor, absolute_indices):
    factor = log_factor[:, absolute_indices[:, :, None], absolute_indices[:, None, :]]
    return log_a[:, :, None, None]+log_p[:, :, :, None]+log_b[:, :, None, :]+factor


def _block_sum(log_a, log_p, log_b, log_factor, absolute_indices, shift):
    weight = _block_log_weight(log_a, log_p, log_b, log_factor, absolute_indices)
    mass = (weight-shift[:, None, None, None]).exp()
    return mass.sum(1), mass.sum((2, 3))


def triangle_contract(logA, logP_rel, logB_rel, logF_abs, tile=16, return_reference=False):
    """Return normalized relative J and the log normalizer relative to the star.

    The supplied logA/logP_rel/logB_rel define a normalized star distribution.
    Each A block is recomputed during backward; no full K-cubed tensor is saved.
    Tile changes temporary storage only and is independent of model association.
    """
    bins = logA.shape[-1]
    relative = torch.arange(bins, device=logA.device)
    with torch.no_grad():
        shift = logA.new_full((len(logA),), -torch.inf)
        for start in range(0, bins, tile):
            stop = min(start+tile, bins)
            indices = (relative[start:stop, None]+relative[None]) % bins
            weight = _block_log_weight(logA[:, start:stop], logP_rel[:, start:stop],
                                       logB_rel[:, start:stop], logF_abs, indices)
            shift = torch.maximum(shift, weight.amax((1, 2, 3)))
        del weight
    joint_raw = logA.new_zeros((len(logA), bins, bins))
    reference_raw = []
    for start in range(0, bins, tile):
        stop = min(start+tile, bins)
        indices = (relative[start:stop, None]+relative[None]) % bins
        inputs = (logA[:, start:stop], logP_rel[:, start:stop], logB_rel[:, start:stop],
                  logF_abs, indices, shift)
        block, reference = (checkpoint(_block_sum, *inputs, use_reentrant=False)
                            if torch.is_grad_enabled() else _block_sum(*inputs))
        joint_raw = joint_raw+block
        if return_reference:
            reference_raw.append(reference)
    total = joint_raw.sum((1, 2))
    result = joint_raw/total[:, None, None], shift+total.log()
    if return_reference:
        return result[0], result[1], torch.cat(reference_raw, 1)/total[:, None]
    return result


class RoleTriangleReader(ReferenceRelationReader):
    """Add one zero-initialized P--B edge to the retained global-star reader."""

    def __init__(self, bins=256, decoder_dim=64, token_dim=16, association="full", tile=16):
        super().__init__(bins, decoder_dim, token_dim, coupling="joint", reference_normalization="global")
        if association not in ("full", "separable"):
            raise ValueError("association must be full or separable")
        self.association, self.tile = association, tile
        self.state_head.register_parameter("Wpb", nn.Parameter(torch.zeros(token_dim, token_dim)))

    def initialize_full(self, checkpoint):
        source = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if source["epoch"] != 48 or source["config"]["training"]["seed"] != 20262020:
            raise ValueError("triangle initialization requires frozen feedback best48 seed20262020")
        if source["config"]["model"]["reference_normalization"] != "global":
            raise ValueError("triangle initialization requires global reference feedback")
        expected = set(self.state_dict())-{"state_head.Wpb"}
        if set(source["model"]) != expected:
            raise ValueError("best48 trained tensors differ beyond the new P--B matrix")
        missing, unexpected = self.load_state_dict(source["model"], strict=False)
        if missing != ["state_head.Wpb"] or unexpected:
            raise ValueError("only the new P--B matrix may be absent from best48")
        with torch.no_grad():
            self.state_head.Wpb.zero_()
        loaded = self.state_dict()
        if any(not torch.equal(loaded[key].cpu(), value.cpu()) for key, value in source["model"].items()):
            raise ValueError("best48 tensor loading changed a retained trained value")
        return {"source_epoch": 48, "source_protocol": source["protocol"],
                "loaded_full_tensors": len(source["model"]), "preserved_all_trained_weights": True,
                "new_zero_parameter": "state_head.Wpb", "new_parameters": self.state_head.Wpb.numel()}

    def forward(self, images):
        star = super().forward(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            relation = self.state_head.relation
            tokens = star["role_tokens"].float()
            h_p, h_b = relation.projections[1](tokens), relation.projections[2](tokens)
            log_factor = pair_log_factor(h_p, h_b, self.state_head.Wpb, self.association)
            joint, log_z, reference = triangle_contract(star["log_A"], star["log_P_rel"], star["log_B_rel"],
                                                       log_factor, tile=self.tile, return_reference=True)
            delta, span = joint.sum(-1), joint.sum(-2)
            product = delta[:, :, None]*span[:, None, :]
            log_reference = reference.clamp_min(torch.finfo(reference.dtype).tiny).log()
            with torch.no_grad():
                counterfactual = self.readout(product.detach())["reading"]
            return {**star, "star_log_A": star["log_A"], "star_A": star["A"],
                    "A": reference, "log_A": log_reference, "J": joint, "joint": joint, "product": product,
                    "delta_marginal": delta, "span_marginal": span,
                    "log_F_abs": log_factor, "log_Z_correction": log_z, **self.readout(joint),
                    "reference_entropy": -(reference*log_reference).sum(-1),
                    "reference_max": reference.amax(-1),
                    "joint_vs_product_state_l1": (joint-product).abs().sum((1, 2)),
                    "counterfactual_reading": counterfactual}


def role_triangle_losses(output, batch):
    """Use the existing three objectives with the exact triangle joint-role CE."""
    losses = reference_relation_losses({**output, "log_A": output["star_log_A"]}, batch)
    labels = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"],
                                         image_size=batch["image"].shape[-1])["role_twohot"].float()
    expected_log_factor = torch.einsum("bp,bq,bpq->b", labels[:, 1], labels[:, 2], output["log_F_abs"])
    raw = losses["role_nll_raw"]+(output["log_Z_correction"]-expected_log_factor).mean()
    normalized = raw/(3*math.log(labels.shape[-1]))
    return {**losses, "role_nll_raw": raw, "role_nll_normalized": normalized,
            "total": normalized+losses["field_l1"]+losses["reading_cdf"]}
