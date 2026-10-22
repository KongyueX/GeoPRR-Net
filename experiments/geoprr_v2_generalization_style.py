"""Training-only backbone normalization and early-block style interventions."""

import torch
from torch import nn


def freeze_backbone_bn_statistics(model):
    """Call after model.train(); preserve every parameter's requires_grad state."""
    for module in model.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def apply_generalization_training_policy(model, policy):
    """Restore training mode, then apply the declared parameter and BN policy."""
    model.set_shared_trainable(True)
    model.train()
    if policy.get("freeze_backbone_weights", False):
        model.backbone.requires_grad_(False)
        model.state_head.requires_grad_(True)
    if policy["freeze_bn_statistics"]:
        freeze_backbone_bn_statistics(model)
    backbone = list(model.backbone.parameters())
    decoder = [parameter for module in (model.lateral, model.refine)
               for parameter in module.parameters()]
    head = list(model.state_head.parameters())
    return {
        "frozen_backbone_parameters": sum(parameter.numel() for parameter in backbone if not parameter.requires_grad),
        "trainable_backbone_parameters": sum(parameter.numel() for parameter in backbone if parameter.requires_grad),
        "trainable_decoder_parameters": sum(parameter.numel() for parameter in decoder if parameter.requires_grad),
        "trainable_head_parameters": sum(parameter.numel() for parameter in head if parameter.requires_grad),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _mix_style(features, lam, permutation):
    """Mix detached spatial channel statistics with explicit pairing and weights."""
    mean = features.mean(dim=(2, 3), keepdim=True)
    std = (features.var(dim=(2, 3), keepdim=True) + 1e-6).sqrt()
    mean, std = mean.detach(), std.detach()
    normalized = (features - mean) / std
    mixed_mean = lam * mean + (1 - lam) * mean[permutation]
    mixed_std = lam * std + (1 - lam) * std[permutation]
    return normalized * mixed_std + mixed_mean


def install_mixstyle_hooks(model, module_names, p=.5, alpha=.1):
    """Return removable hooks for configured actual blocks inside model.backbone.

    Names are relative to the backbone, before later backbone blocks execute.
    The caller removes each returned handle when the training intervention ends.
    """
    if not 0 <= p <= 1 or alpha <= 0:
        raise ValueError("MixStyle requires 0 <= p <= 1 and alpha > 0")
    if not module_names or len(set(module_names)) != len(module_names):
        raise ValueError("MixStyle requires unique nonempty backbone block names")
    backbone = model.backbone
    modules = dict(backbone.named_modules())
    selected = []
    for name in module_names:
        if not name or name not in modules:
            raise ValueError(f"unknown backbone block: {name!r}")
        module = modules[name]
        if type(module).forward is nn.Module.forward:
            raise ValueError(f"backbone module has no forward: {name!r}")
        selected.append(module)

    def mix_output(module, _inputs, output):
        if not (model.training and backbone.training and module.training) or p == 0:
            return output
        if torch.rand(()) >= p:
            return output
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("MixStyle backbone blocks must output B,C,H,W tensors")
        batch_size = output.shape[0]
        lam = torch.distributions.Beta(alpha, alpha).sample((batch_size, 1, 1, 1))
        lam = lam.to(device=output.device, dtype=output.dtype)
        permutation = torch.randperm(batch_size, device=output.device)
        return _mix_style(output, lam, permutation)

    return tuple(module.register_forward_hook(mix_output) for module in selected)
