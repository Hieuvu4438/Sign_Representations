"""Local-anchor SupCon against global autograd-gathered keys."""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from ..training.distributed import rank, world_size, global_sum


def gather_metadata(labels, valid):
    if not dist.is_initialized():
        return labels, valid
    ls, vs = [torch.empty_like(labels) for _ in range(world_size())], [torch.empty_like(valid) for _ in range(world_size())]
    dist.all_gather(ls, labels)
    dist.all_gather(vs, valid)
    return torch.cat(ls), torch.cat(vs)


def anchor_support(labels, valid):
    keys, key_valid = gather_metadata(labels, valid)
    b = len(labels)
    own = torch.arange(b, device=labels.device)+rank()*b
    same = (labels[:, None] == keys[None]) & key_valid[None] & valid[:, None]
    same[torch.arange(b, device=labels.device), own] = False
    return same.any(-1), same, key_valid, own


def supcon_terms(embeddings, labels, valid, temperature=.07):
    if temperature <= 0 or embeddings.shape[0] != len(labels):
        raise ValueError('Invalid SupCon temperature/batch')
    anchors, positives, key_valid, own = anchor_support(labels, valid)
    keys = torch.cat(all_gather(embeddings), 0) if dist.is_initialized() else embeddings
    logits = embeddings.float() @ keys.float().T / temperature
    allowed = key_valid[None].expand_as(logits).clone()
    allowed[torch.arange(len(labels), device=labels.device), own] = False
    if not bool(anchors.any()):
        # Keep autograd gather on every rank even when no local anchor is valid.
        return (logits.sum()*0), anchors.sum().float().detach()
    scores = logits[anchors]
    denominator = torch.logsumexp(scores.masked_fill(~allowed[anchors], -torch.inf), dim=-1)
    positive = positives[anchors]
    loss = -((scores-denominator[:, None])*positive).sum(-1)/positive.sum(-1)
    return loss.sum(), anchors.sum().float().detach()
