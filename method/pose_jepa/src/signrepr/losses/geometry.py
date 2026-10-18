import torch
import torch.nn.functional as F


def physical_huber_terms(pred, target, weights, valid, beta):
    pred = torch.where(valid[..., None], pred, 0).float()
    target = torch.where(valid[..., None], target, 0).float()
    w = torch.where(valid, weights, 0).float()
    error = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    return (error*w).sum(), w.sum().detach()
