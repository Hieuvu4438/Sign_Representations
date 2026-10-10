"""Paper optimizer recipe with explicit schedule and layer-decay conventions."""

import math
import torch


def layerwise_parameter_groups(model, learning_rate=1e-4, weight_decay=.5, layer_decay=.85):
    backbone = model.backbone if hasattr(model, "backbone") else model.encoder.backbone
    depth = len(backbone.blocks)
    last_layer = depth + 1
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        encoder_name = name.removeprefix("encoder.")
        if encoder_name.startswith("backbone.blocks."):
            layer = int(encoder_name.split(".")[2]) + 1
        elif encoder_name.startswith("backbone.patch_embed.") or encoder_name.startswith("backbone.pos_embed"):
            layer = 0
        else:
            layer = last_layer
        no_decay = parameter.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name
        decay = 0. if no_decay else weight_decay
        key = (layer, decay)
        if key not in groups:
            scale = layer_decay ** (last_layer - layer)
            groups[key] = {"params": [], "parameter_names": [], "lr_scale": scale,
                           "lr": learning_rate * scale, "weight_decay": decay}
        groups[key]["params"].append(parameter)
        groups[key]["parameter_names"].append(name)
    return list(groups.values())


class WarmupCosine:
    """One-based optimizer-update number: warmup end reaches peak, final reaches zero."""
    def __init__(self, optimizer, max_lr=1e-4, warmup_steps=50000, total_steps=500000):
        if not 0 < warmup_steps < total_steps:
            raise ValueError("Schedule requires 0 < warmup < total steps")
        self.optimizer = optimizer
        self.max_lr, self.warmup_steps, self.total_steps = max_lr, warmup_steps, total_steps
        self.completed_steps = 0

    def rate(self, update):
        if not 1 <= update <= self.total_steps:
            raise ValueError("Optimizer update outside schedule")
        if update <= self.warmup_steps:
            return self.max_lr * update / self.warmup_steps
        progress = (update - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.max_lr * .5 * (1 + math.cos(math.pi * progress))

    def before_update(self, update):
        if update != self.completed_steps + 1:
            raise ValueError("Schedule update order diverged")
        rate = self.rate(update)
        for group in self.optimizer.param_groups:
            group["lr"] = rate * group.get("lr_scale", 1.)
        return rate

    def after_update(self):
        self.completed_steps += 1

    def state_dict(self):
        return {"max_lr": self.max_lr, "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps, "completed_steps": self.completed_steps}

    def load_state_dict(self, state):
        for key in ["max_lr", "warmup_steps", "total_steps"]:
            if state[key] != getattr(self, key):
                raise ValueError(f"Scheduler changed on resume: {key}")
        self.completed_steps = state["completed_steps"]


def pretraining_optimizers(model, discriminator, max_lr=1e-4, warmup_steps=50000, total_steps=500000):
    encoder_parameters = {id(p) for p in model.parameters()}
    discriminator_parameters = {id(p) for p in discriminator.parameters()}
    if encoder_parameters & discriminator_parameters:
        raise ValueError("Encoder/discriminator parameters overlap")
    encoder_optimizer = torch.optim.AdamW(layerwise_parameter_groups(model, max_lr),
                                          lr=max_lr, betas=(.9, .95))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=max_lr,
                                               betas=(.5, .9), weight_decay=1e-3)
    return (encoder_optimizer, discriminator_optimizer,
            WarmupCosine(encoder_optimizer, max_lr, warmup_steps, total_steps),
            WarmupCosine(discriminator_optimizer, max_lr, warmup_steps, total_steps))
