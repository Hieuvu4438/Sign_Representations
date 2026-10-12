import math
import torch


def cosine_momentum(successful_step, total_steps, start=.996, end=1.0):
    if total_steps <= 0 or not 0 <= successful_step <= total_steps or not 0 <= start <= end <= 1:
        raise ValueError("Invalid EMA schedule")
    return end - (end - start) * (1 + math.cos(math.pi * successful_step / total_steps)) / 2


@torch.no_grad()
def update_teacher(online, teacher, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError("Invalid EMA momentum")
    source, target = online.state_dict(), teacher.state_dict()
    if source.keys() != target.keys():
        raise ValueError("Teacher and online state schemas differ")
    for name in source:
        if source[name].shape != target[name].shape or source[name].dtype != target[name].dtype:
            raise ValueError(f"Teacher state mismatch: {name}")
        if target[name].is_floating_point():
            target[name].mul_(momentum).add_(source[name], alpha=1 - momentum)
        else:
            target[name].copy_(source[name])
    teacher.requires_grad_(False).eval()
