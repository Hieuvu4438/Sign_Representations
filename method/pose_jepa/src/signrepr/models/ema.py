import math
import torch


def momentum(step, total_steps, initial=0.999):
    return 1 - (1-initial) * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)) / 2


@torch.no_grad()
def update_encoder_ema(student, teacher, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError("EMA momentum must be in [0,1]")
    src, dst = dict(student.named_parameters()), dict(teacher.named_parameters())
    if src.keys() != dst.keys():
        raise ValueError("EMA parameter topology mismatch")
    for k, value in src.items():
        if value.shape != dst[k].shape:
            raise ValueError(f"EMA shape mismatch: {k}")
        dst[k].mul_(momentum).add_(value.detach().to(dst[k].dtype), alpha=1-momentum)
    sb, tb = dict(student.named_buffers()), dict(teacher.named_buffers())
    if sb.keys() != tb.keys():
        raise ValueError("EMA buffer topology mismatch")
    for k, value in sb.items():
        tb[k].copy_(value)
    teacher.eval()
