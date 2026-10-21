"""RGB reference evidence with pair-preserving and global posterior updates."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from experiments.geoprr_v2_reference_relation import reference_geometry_targets
from experiments.geoprr_v2_role_triangle import RoleTriangleReader, role_triangle_losses
from experiments.geoprr_v2_reference_pair_update import reference_pair_contract, reference_pair_ce_correction


class ReferenceEvidenceVerifier(nn.Module):
    """Read ordered angular neighborhoods from the existing circular tokens."""

    def __init__(self, token_dim=16):
        super().__init__()
        self.register_buffer("offsets", torch.tensor([-8, -4, -2, -1, 0, 1, 2, 4, 8]), persistent=False)
        self.first = nn.Linear(9 * token_dim, 32)
        self.activation = nn.SiLU()
        self.last = nn.Linear(32, 1, bias=False)
        nn.init.zeros_(self.last.weight)

    def forward(self, tokens):
        bins = tokens.shape[1]
        indices = (torch.arange(bins, device=tokens.device)[:, None] + self.offsets[None]) % bins
        neighborhoods = tokens[:, indices].flatten(2)
        return self.last(self.activation(self.first(neighborhoods))).squeeze(-1)


class ReferencePairReader(RoleTriangleReader):
    """Update reference explanations within the existing absolute role pairs."""

    def __init__(self, bins=256, decoder_dim=64, token_dim=16, reference_update="pair_preserving", tile=16):
        if reference_update not in ("pair_preserving", "global"):
            raise ValueError("reference_update must be pair_preserving or global")
        super().__init__(bins, decoder_dim, token_dim, association="full", tile=tile)
        self.reference_update = reference_update
        self.state_head.add_module("reference_verifier", ReferenceEvidenceVerifier(token_dim))

    def initialize_full(self, checkpoint):
        source = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if (source["epoch"] != 64 or source["config"]["training"]["seed"] != 20262020
                or source["config"]["model"]["association"] != "full"):
            raise ValueError("reference update requires the retained full best64 seed20262020")
        prefix = "state_head.reference_verifier."
        expected = {key for key in self.state_dict() if not key.startswith(prefix)}
        if set(source["model"]) != expected:
            raise ValueError("best64 state differs beyond the new reference verifier")
        missing, unexpected = self.load_state_dict(source["model"], strict=False)
        if set(missing) != {prefix + name for name in ("first.weight", "first.bias", "last.weight")} or unexpected:
            raise ValueError("only the three reference verifier tensors may be absent")
        with torch.no_grad():
            self.state_head.reference_verifier.last.weight.zero_()
        loaded = self.state_dict()
        if any(not torch.equal(loaded[key].cpu(), value.cpu()) for key, value in source["model"].items()):
            raise ValueError("best64 initialization changed a retained tensor")
        return {"source_epoch": 64, "loaded_full_tensors": len(source["model"]),
                "preserved_all_trained_weights": True, "new_parameters": 4672,
                "new_zero_parameter": prefix + "last.weight"}

    def update_reference(self, base):
        """Reuse the base RGB features; retain exact normalization gradients."""
        with torch.autocast(device_type=base["role_tokens"].device.type, enabled=False):
            evidence = self.state_head.reference_verifier(base["role_tokens"].float())
            update = reference_pair_contract(base["star_log_A"], base["log_P_rel"], base["log_B_rel"],
                                             base["log_F_abs"], evidence, mode=self.reference_update,
                                             tile=self.tile, return_reference=True)
            joint, reference = update["J"], update["A"]
            delta, span = joint.sum(-1), joint.sum(-2)
            product = delta[:, :, None] * span[:, None, :]
            log_reference = reference.clamp_min(torch.finfo(reference.dtype).tiny).log()
            with torch.no_grad():
                counterfactual = self.readout(product.detach())["reading"]
            return {**base, "A": reference, "log_A": log_reference, "J": joint, "joint": joint,
                    "product": product, "delta_marginal": delta, "span_marginal": span,
                    "reference_evidence": evidence, "reference_update_result": update,
                    **self.readout(joint), "counterfactual_reading": counterfactual,
                    "reference_entropy": -(reference * log_reference).sum(-1),
                    "reference_max": reference.amax(-1),
                    "joint_vs_product_state_l1": (joint - product).abs().sum((1, 2))}

    def forward(self, images):
        return self.update_reference(super().forward(images))


def reference_pair_losses(output, batch):
    """Use the original three objectives under the updated joint distribution."""
    losses = role_triangle_losses(output, batch)
    labels = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"],
                                         image_size=batch["image"].shape[-1])["role_twohot"].float()
    correction = reference_pair_ce_correction(output["reference_update_result"], labels,
                                             output["reference_evidence"]).mean()
    raw = losses["role_nll_raw"] + correction
    normalized = raw / (3 * math.log(labels.shape[-1]))
    return {**losses, "role_nll_raw": raw, "role_nll_normalized": normalized,
            "total": normalized + losses["field_l1"] + losses["reading_cdf"],
            "reference_ce_correction": correction}
