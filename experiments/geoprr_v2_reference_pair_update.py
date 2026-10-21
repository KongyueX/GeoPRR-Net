"""Exact reference updates with a preserved absolute P--B marginal."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def _absolute_block_marginals(log_a, log_p, log_b, log_factor, value, reference_bins):
    bins = log_p.shape[-1]
    absolute = torch.arange(bins, device=log_p.device)
    relative = (absolute[None] - reference_bins[:, None]) % bins
    indices = relative[None].expand(len(log_a), -1, -1)
    p_absolute = log_p.gather(-1, indices)
    b_absolute = log_b.gather(-1, indices)
    energy = (log_a[:, :, None, None] + p_absolute[:, :, :, None]
              + b_absolute[:, :, None, :] + log_factor[:, None])
    return energy.logsumexp(1), (energy + value[:, :, None, None]).logsumexp(1)


def _relative_block(log_a, log_p, log_b, log_factor, correction, value,
                    reference_bins, log_normalizer):
    relative = torch.arange(log_p.shape[-1], device=log_p.device)
    absolute = (reference_bins[:, None] + relative[None]) % log_p.shape[-1]
    factor = log_factor[:, absolute[:, :, None], absolute[:, None, :]]
    adjustment = correction[:, absolute[:, :, None], absolute[:, None, :]]
    log_weight = (log_a[:, :, None, None] + log_p[:, :, :, None] + log_b[:, :, None, :]
                  + factor + value[:, :, None, None] + adjustment - log_normalizer[:, None, None, None])
    probability = log_weight.exp()
    return probability.sum(1), probability.sum((2, 3))


def reference_pair_contract(logA_star, logP_rel, logB_rel, logF_abs, v, *,
                            mode="pair_preserving", tile=16, return_reference=False):
    """Transport either exact reference update to relative delta/span states.

    M and c are computed from the normalized base T, so c need not sum to1.
    All reference blocks and both absolute pair marginals remain differentiable.
    """
    if mode not in ("pair_preserving", "global"):
        raise ValueError("mode must be pair_preserving or global")
    bins = logA_star.shape[-1]
    log_m_u = log_c_u = None
    for start in range(0, bins, tile):
        stop = min(start + tile, bins)
        reference_bins = torch.arange(start, stop, device=logA_star.device)
        inputs = (logA_star[:, start:stop], logP_rel[:, start:stop], logB_rel[:, start:stop],
                  logF_abs, v[:, start:stop], reference_bins)
        block_m, block_c = (checkpoint(_absolute_block_marginals, *inputs, use_reentrant=False)
                            if torch.is_grad_enabled() else _absolute_block_marginals(*inputs))
        log_m_u = block_m if log_m_u is None else torch.logaddexp(log_m_u, block_m)
        log_c_u = block_c if log_c_u is None else torch.logaddexp(log_c_u, block_c)
    log_z_base = log_m_u.flatten(1).logsumexp(-1)
    log_z_updated = log_c_u.flatten(1).logsumexp(-1)
    correction = log_m_u - log_c_u if mode == "pair_preserving" else torch.zeros_like(log_m_u)
    normalizer = log_z_base if mode == "pair_preserving" else log_z_updated
    joint = torch.zeros_like(logF_abs)
    reference = []
    for start in range(0, bins, tile):
        stop = min(start + tile, bins)
        reference_bins = torch.arange(start, stop, device=logA_star.device)
        inputs = (logA_star[:, start:stop], logP_rel[:, start:stop], logB_rel[:, start:stop],
                  logF_abs, correction, v[:, start:stop], reference_bins, normalizer)
        block, reference_mass = (checkpoint(_relative_block, *inputs, use_reentrant=False)
                                 if torch.is_grad_enabled() else _relative_block(*inputs))
        joint = joint + block
        if return_reference:
            reference.append(reference_mass)
    result = {"J": joint, "log_Z_base": log_z_base, "log_Z_updated": log_z_updated,
              "log_M_abs": log_m_u - log_z_base[:, None, None],
              "log_c_abs": log_c_u - log_z_base[:, None, None], "mode": mode}
    if return_reference:
        result["A"] = torch.cat(reference, 1)
    return result


def reference_pair_ce_correction(result, role_twohot, v):
    """Return the per-image correction to the original joint-role CE."""
    expected_value = (role_twohot[:, 0] * v).sum(-1)
    if result["mode"] == "global":
        return result["log_Z_updated"] - result["log_Z_base"] - expected_value
    pair_label = role_twohot[:, 1, :, None] * role_twohot[:, 2, None, :]
    return -expected_value + (pair_label * (result["log_c_abs"] - result["log_M_abs"])).sum((1, 2))
