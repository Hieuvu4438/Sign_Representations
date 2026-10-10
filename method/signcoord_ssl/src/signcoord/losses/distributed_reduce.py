import torch
import torch.distributed as dist


def weighted_mean(numerator, denominator, eps=1e-6):
    """Gradient scaling accounts for DDP's subsequent rank-average."""
    den = denominator.detach().clone()
    num_log = numerator.detach().clone()
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(den)
        dist.all_reduce(num_log)
    loss = numerator * world / den.clamp_min(eps)
    return loss, {"numerator": num_log.item(), "denominator": den.item(),
                  "mean": (num_log / den.clamp_min(eps)).item(), "active": den.item() > 0}
