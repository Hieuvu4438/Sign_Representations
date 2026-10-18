import torch
import torch.nn.functional as F


def latent_ce_terms(pred, target, mask, quality, center, student_temperature=0.1, teacher_temperature=0.06):
    pred = torch.where(mask[..., None], pred, 0).float()
    target = torch.where(mask[..., None], target.detach(), 0).float()
    weights = torch.where(mask, quality, 0).float()
    prob = F.softmax((target-center.float())/teacher_temperature, -1)
    per_token = -(prob*F.log_softmax(pred/student_temperature, -1)).sum(-1)
    return (per_token*weights).sum(), weights.sum().detach(), (target*weights[..., None]).sum((0, 1)).detach()
