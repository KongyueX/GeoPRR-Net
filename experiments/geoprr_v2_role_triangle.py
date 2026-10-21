"""Exact three-role association and its matched marginal-product deletion."""
from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint


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
    return (weight-shift[:, None, None, None]).exp().sum(1)


def triangle_contract(logA, logP_rel, logB_rel, logF_abs, tile=16):
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
    for start in range(0, bins, tile):
        stop = min(start+tile, bins)
        indices = (relative[start:stop, None]+relative[None]) % bins
        inputs = (logA[:, start:stop], logP_rel[:, start:stop], logB_rel[:, start:stop],
                  logF_abs, indices, shift)
        block = (checkpoint(_block_sum, *inputs, use_reentrant=False)
                 if torch.is_grad_enabled() else _block_sum(*inputs))
        joint_raw = joint_raw+block
    total = joint_raw.sum((1, 2))
    return joint_raw/total[:, None, None], shift+total.log()
