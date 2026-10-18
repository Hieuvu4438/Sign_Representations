"""Thin torch.distributed helpers. Single-process runs work without initialising a process group."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def init_from_env(backend: str = "nccl") -> tuple[int, int, int]:
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; otherwise stays single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        import datetime

        # long timeout: rank 0 alone extracts ~50k validation embeddings while the others wait at a barrier
        dist.init_process_group(backend, timeout=datetime.timedelta(hours=2))
        return rank(), world_size(), local
    return 0, 1, 0


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class _GatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        g = torch.stack(grads)
        dist.all_reduce(g)
        return g[rank()]


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Autograd-compatible all_gather (global-consistency statistics over physical batch x world size)."""
    if not is_dist():
        return x
    return torch.cat(_GatherWithGrad.apply(x), 0)


def barrier():
    if is_dist():
        dist.barrier()
