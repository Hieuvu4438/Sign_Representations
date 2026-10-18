"""AdamW with no weight decay on biases, LayerNorm, embeddings and learned query tokens."""
from __future__ import annotations

import torch
from torch import nn


def _no_decay(name: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or name.endswith(".bias") or "emb" in name or "mask_token" in name or ".norm" in name


def param_groups(named_modules: dict[str, nn.Module], weight_decay: float, lr_scale: dict | None = None):
    groups = []
    seen = set()
    for tag, module in named_modules.items():
        if module is None:
            continue
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (no_decay if _no_decay(name, p) else decay).append(p)
        scale = (lr_scale or {}).get(tag, 1.0)
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay, "lr_scale": scale, "name": f"{tag}/decay"})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0, "lr_scale": scale, "name": f"{tag}/no_decay"})
    return groups


def build_adamw(groups, lr: float, betas, eps: float):
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps)


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
