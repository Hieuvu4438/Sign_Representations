"""Collectives with identical call order on empty and populated ranks."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def sum_detached(value):
    result = value.detach().clone()
    if world_size() > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_true(condition, device):
    flag = torch.tensor(int(bool(condition)), dtype=torch.int32, device=device)
    if world_size() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def initialize(device):
    """Initialize torchrun environments; no implicit CUDA substitution."""
    device = torch.device(device)
    launched = "RANK" in os.environ
    if launched and not dist.is_initialized():
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=120))
    return device
