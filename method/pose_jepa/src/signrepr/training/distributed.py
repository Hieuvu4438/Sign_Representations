import os
import torch
import torch.distributed as dist


def initialize_distributed(device=None):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if device is None:
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
    if distributed and not dist.is_initialized():
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def global_sum(tensor):
    result = tensor.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def all_finite(parameters, device):
    value = torch.tensor(int(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)), device=device)
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return bool(value)


def loss_for_backward(local_numerator, global_update_denominator, world):
    if global_update_denominator.item() <= 0:
        return local_numerator*0
    return world*local_numerator/global_update_denominator
