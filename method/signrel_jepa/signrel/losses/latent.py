import torch
import torch.nn.functional as F


def latent_loss(pred, target, valid, kind="l1", normalize="layernorm", distributed=False):
    if target.requires_grad:
        raise AssertionError("Target must be stop-gradient")
    global_count = None
    if distributed and torch.distributed.is_initialized():
        global_count = valid.sum().detach().to(torch.float32)
        torch.distributed.all_reduce(global_count)
        if not bool(global_count > 0):
            raise AssertionError("Zero valid prediction queries across all ranks")
    elif not bool(valid.any()):
        raise AssertionError("Zero valid prediction queries")
    if normalize == "layernorm":
        pred = F.layer_norm(pred,(pred.shape[-1],))
        target = F.layer_norm(target,(target.shape[-1],))
    elif normalize != "none":
        raise ValueError(normalize)
    # Index before arithmetic: invalid/NaN targets cannot contaminate a zero-weight loss.
    p,t = pred[valid],target[valid]
    if kind == "l1":
        per_query = (p-t).abs().mean(-1)
    elif kind == "smooth_l1":
        per_query = F.smooth_l1_loss(p,t,reduction="none").mean(-1)
    elif kind == "cosine":
        per_query = 1-F.cosine_similarity(p,t,dim=-1)
    else:
        raise ValueError(kind)
    if global_count is not None:
        # DDP averages parameter gradients; compensate to recover the global query mean.
        return per_query.sum()*torch.distributed.get_world_size()/global_count
    return per_query.mean()
