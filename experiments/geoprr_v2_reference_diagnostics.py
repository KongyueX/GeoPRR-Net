"""Frozen-state role interventions and reference contribution diagnostics."""
from __future__ import annotations

import math

import torch


def _relative_role_labels(role_labels):
    bins = role_labels.shape[-1]
    indices = (torch.arange(bins, device=role_labels.device)[:, None]
               +torch.arange(bins, device=role_labels.device)[None]) % bins
    indices = indices[None].expand(len(role_labels), -1, -1)
    return tuple(role_labels[:, role, None].expand(-1, bins, -1).gather(-1, indices)
                 for role in (1, 2))


def _transport_matrices(readout, bins):
    coordinates = readout.coordinates
    mean = ((1-readout.fraction)*coordinates[readout.lower]
            +readout.fraction*coordinates[readout.upper])
    invalid = readout.invalid_atoms.reshape(bins, bins)
    mean = torch.where(readout.invalid_atoms, coordinates.mean(), mean).reshape(bins, bins)
    angles = torch.arange(bins, dtype=torch.float64, device=coordinates.device)*2*math.pi/bins
    span = angles[None]
    raw = ((angles[:, None]-span/2+math.pi).remainder(2*math.pi)-math.pi+span/2)
    raw = raw/span.masked_fill(invalid, 1.)
    low, high = (raw < 0) & ~invalid, (raw > 1) & ~invalid
    return mean, invalid.float(), low.float(), high.float()


def _entropy(probabilities):
    return -(probabilities*probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(-1)


@torch.no_grad()
def diagnose_state(state, readout, role_labels):
    """Decompose a joint reading without a per-reference joint-state cube."""
    start, pointer, end = (state[key].float() for key in ("A", "P_rel", "B_rel"))
    delta = (start[:, :, None]*pointer).sum(1)
    span = (start[:, :, None]*end).sum(1)
    joint = pointer.transpose(1, 2) @ (start[:, :, None]*end)
    product = delta[:, :, None]*span[:, None]
    joint_output, product_output = readout(joint), readout(product)
    mean, invalid, low, high = _transport_matrices(readout, start.shape[-1])
    pointer_mean = pointer @ mean
    conditional = (pointer_mean*end).sum(-1)
    product_reference = (pointer_mean*span[:, None]).sum(-1)
    contribution = start*(conditional-product_reference)
    teacher_pointer, teacher_end = _relative_role_labels(role_labels.float())
    return {
        "A": start,
        "delta_marginal": delta,
        "span_marginal": span,
        "progress_joint": joint_output["distribution"],
        "progress_product": product_output["distribution"],
        "conditional_reading": conditional,
        "conditional_P_entropy": _entropy(pointer),
        "conditional_B_entropy": _entropy(end),
        "conditional_P_correct_mass": (pointer*(teacher_pointer > 0)).sum(-1),
        "conditional_B_correct_mass": (end*(teacher_end > 0)).sum(-1),
        "conditional_invalid_mass": ((pointer @ invalid)*end).sum(-1),
        "conditional_outside_low_mass": ((pointer @ low)*end).sum(-1),
        "conditional_outside_high_mass": ((pointer @ high)*end).sum(-1),
        "signed_reading_contribution": contribution,
        "conditional_reading_residual": (start*conditional).sum(-1)-joint_output["reading"],
        "signed_contribution_residual": contribution.sum(-1)
            -(joint_output["reading"]-product_output["reading"]),
    }


@torch.no_grad()
def role_interventions(state, readout, role_labels):
    """Replace teacher roles in an already teacher-rectified diagnostic state."""
    start, pointer, end = (state[key].float() for key in ("A", "P_rel", "B_rel"))
    labels = role_labels.float()
    teacher_pointer, teacher_end = _relative_role_labels(labels)
    outputs = {}
    for name in ("A", "P", "B", "PB", "AP", "AB", "APB"):
        a = labels[:, 0] if "A" in name else start
        p = teacher_pointer if "P" in name else pointer
        b = teacher_end if "B" in name else end
        outputs[name] = readout(p.transpose(1, 2) @ (a[:, :, None]*b))
    return outputs
