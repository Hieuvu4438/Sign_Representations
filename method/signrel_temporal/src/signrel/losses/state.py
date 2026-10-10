"""FP32 clean-weighted losses with stream-balanced hand reduction."""
import torch
import torch.nn.functional as F


def validate_inputs(pred,target,valid,quality,mask):
    if pred.shape != target.shape or valid.shape != pred.shape[:-1] or quality.shape != valid.shape or mask.shape != valid.shape:
        raise ValueError('State tensor shape mismatch')
    if valid.dtype != torch.bool or mask.dtype != torch.bool:
        raise ValueError('Validity and SSL mask must be bool')
    if not all(torch.isfinite(x).all() for x in (pred,target,quality)):
        raise ValueError('Non-finite loss input')
    if ((quality<0)|(quality>1)).any():
        raise ValueError('Quality outside [0,1]')


def weighted_mean(value,weight,zero_reference):
    if value.shape != weight.shape:
        raise ValueError('Weighted mean shape mismatch')
    denominator = weight.detach().float().sum()
    numerator = (value.float()*weight.detach().float()).sum()
    zero = zero_reference.float().sum()*0
    return torch.where(denominator>0,numerator/denominator.clamp_min(1e-12),zero),denominator


def hand_state_loss(pred,target,valid,quality,ssl_mask):
    validate_inputs(pred,target,valid,quality,ssl_mask)
    error = (pred.float()-target.detach().float()).square().sum(-1)
    weight = valid.float()*ssl_mask.float()*quality.detach().float()
    terms,denominators = zip(*(weighted_mean(error[:,:,a],weight[:,:,a],pred) for a in range(pred.shape[2])))
    denominators,terms = torch.stack(denominators),torch.stack(terms)
    active = (denominators>0).float()
    return (terms*active).sum()/active.sum().clamp_min(1),denominators,terms


def body_state_loss(pred,target,body_valid,coord_valid,coord_quality,ssl_mask,beta=1):
    if pred.shape != target.shape or coord_valid.shape != pred.shape or coord_quality.shape != pred.shape:
        raise ValueError('BODY coordinate shape mismatch')
    if body_valid.shape != pred.shape[:2] or ssl_mask.shape != body_valid.shape:
        raise ValueError('BODY token shape mismatch')
    if any(x.dtype != torch.bool for x in (body_valid,coord_valid,ssl_mask)):
        raise ValueError('BODY validity and mask must be bool')
    if not all(torch.isfinite(x).all() for x in (pred,target,coord_quality)) or ((coord_quality<0)|(coord_quality>1)).any():
        raise ValueError('Non-finite BODY input or invalid quality')
    error = F.smooth_l1_loss(pred.float(),target.detach().float(),reduction='none',beta=beta)
    weight = ssl_mask[...,None].float()*body_valid[...,None].float()*coord_valid.float()*coord_quality.detach().float()
    return weighted_mean(error,weight,pred)
