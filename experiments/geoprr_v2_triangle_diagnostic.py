"""Stable frozen-triangle reference and role-evidence summaries."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def summarize_triangle(logA, logP_rel, logB_rel, logF_abs, reading_atoms,
                       role_twohot, tile=16, *, atom_masks):
    """Summarize full triangle conditionals without dividing by global A.

    ``reading_atoms[d,s]`` is the existing progress transport's reading
    expectation, including 0.5 for zero-span states. Correct-role mass is the
    probability of the nonzero two-hot support, while GT-A uses its weights.
    ``logZ_a`` includes logA and the per-reference conditional log normalizer.
    """
    batch, bins = logA.shape
    relative = torch.arange(bins, device=logA.device)
    support_p = role_twohot[:, 1] > 0
    support_b = role_twohot[:, 2] > 0
    log_normalizers, conditional_readings = [], []
    correct_p, correct_b, correct_joint, conditional_mass_errors = [], [], [], []
    conditional_atom_masses = {key: [] for key in ("invalid", "outside")}
    for start in range(0, bins, tile):
        stop = min(start + tile, bins)
        absolute = (relative[start:stop, None] + relative[None]) % bins
        factor = logF_abs[:, absolute[:, :, None], absolute[:, None, :]]
        log_weight = logP_rel[:, start:stop, :, None] + logB_rel[:, start:stop, None, :] + factor
        flat = log_weight.flatten(2)
        log_normalizers.append(logA[:, start:stop] + flat.logsumexp(-1))
        conditional = flat.softmax(-1).reshape_as(log_weight)
        conditional_readings.append((conditional * reading_atoms[None, None]).sum((2, 3)))
        p_support = support_p[:, absolute].to(conditional.dtype)
        b_support = support_b[:, absolute].to(conditional.dtype)
        correct_p.append((conditional.sum(-1) * p_support).sum(-1))
        correct_b.append((conditional.sum(-2) * b_support).sum(-1))
        correct_joint.append(((conditional * p_support[:, :, :, None]).sum(-2) * b_support).sum(-1))
        for key in conditional_atom_masses:
            conditional_atom_masses[key].append((conditional * atom_masks[key][None, None]).sum((2, 3)))
        conditional_mass_errors.append((conditional.sum((2, 3)) - 1).abs())
    log_z_a = torch.cat(log_normalizers, 1)
    reference = log_z_a.softmax(-1)
    conditional_reading = torch.cat(conditional_readings, 1)
    mass_error = torch.maximum((reference.sum(-1) - 1).abs(),
                               torch.cat(conditional_mass_errors, 1).amax(-1))
    invalid = torch.cat(conditional_atom_masses["invalid"], 1)
    outside = torch.cat(conditional_atom_masses["outside"], 1)
    return {"A": reference, "conditional_reading": conditional_reading,
            "reading": (reference * conditional_reading).sum(-1),
            "GTA_reading": (role_twohot[:, 0] * conditional_reading).sum(-1),
            "conditional_P_correct_mass": torch.cat(correct_p, 1),
            "conditional_B_correct_mass": torch.cat(correct_b, 1),
            "joint_correct_mass": torch.cat(correct_joint, 1), "logZ_a": log_z_a,
            "conditional_invalid_mass": invalid, "conditional_outside_arc_mass": outside,
            "invalid_mass": (reference * invalid).sum(-1), "outside_arc_mass": (reference * outside).sum(-1),
            "GTA_invalid_mass": (role_twohot[:, 0] * invalid).sum(-1),
            "GTA_outside_arc_mass": (role_twohot[:, 0] * outside).sum(-1),
            "mass_error": mass_error,
            "counts": {"triangle_summary_calls": 1, "reference_blocks": math.ceil(bins / tile),
                       "summary_image_equivalents": batch}}
