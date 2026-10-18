"""EMA teacher update: once per successful optimizer step, never per microbatch."""
from __future__ import annotations

import torch


@torch.no_grad()
def ema_update(pairs, momentum: float) -> None:
    for teacher, online in pairs:
        t_params = dict(teacher.named_parameters())
        o_params = dict(online.named_parameters())
        if t_params.keys() != o_params.keys():
            raise RuntimeError("teacher/online parameter names differ")
        for name, tp in t_params.items():
            tp.mul_(momentum).add_(o_params[name].detach().to(tp.dtype), alpha=1.0 - momentum)
        o_bufs = dict(online.named_buffers())
        for name, tb in teacher.named_buffers():
            if name in o_bufs:
                tb.copy_(o_bufs[name])
