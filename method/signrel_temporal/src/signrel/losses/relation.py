import torch
import torch.nn.functional as F
from .pairs import validate_pairs
from .state import validate_inputs,weighted_mean


def relation_values(pred,target,pairs):
    validate_pairs(pairs,pred.shape[:3])
    i,t,a,u,b = pairs.unbind(-1)
    p = F.normalize(pred.float(),dim=-1,eps=1e-6)
    q = F.normalize(target.detach().float(),dim=-1,eps=1e-6)
    return (p[i,t,a]*p[i,u,b]).sum(-1),(q[i,t,a]*q[i,u,b]).sum(-1)


def shuffle_targets(values,pairs,generator):
    """Cross-window cyclic donors within lag; singleton groups are excluded."""
    result,keep = values.detach().clone(),torch.zeros(len(values),dtype=torch.bool,device=values.device)
    lags = (pairs[:,3]-pairs[:,1]).cpu()
    samples = pairs[:,0].cpu()
    for lag in sorted(set(lags.tolist())):
        windows = sorted(set(samples[lags==lag].tolist()))
        if len(windows)<2: continue
        order = torch.randperm(len(windows),generator=generator).tolist()
        windows = [windows[j] for j in order]
        for j,window in enumerate(windows):
            recipient = torch.nonzero((lags==lag)&(samples==window)).flatten()
            donor = torch.nonzero((lags==lag)&(samples==windows[(j+1)%len(windows)])).flatten()
            selected = donor[torch.randint(len(donor),(len(recipient),),generator=generator)]
            result[recipient.to(values.device)] = values[selected.to(values.device)]
            keep[recipient.to(values.device)] = True
    return result,keep


def relation_loss(pred,target,valid,quality,ssl_mask,pairs,beta=.1,mode='cross_signed',generator=None):
    validate_inputs(pred,target,valid,quality,ssl_mask)
    validate_pairs(pairs,pred.shape[:3])
    if not len(pairs):
        return pred.float().sum()*0,pred.new_zeros((),dtype=torch.float32),{'count':0,'target_mean':None,'target_std':None,'prediction_mean':None}
    predicted,clean = relation_values(pred,target,pairs)
    i,t,a,u,b = pairs.unbind(-1)
    eligible = valid[i,t,a]&valid[i,u,b]&(ssl_mask[i,t,a]|ssl_mask[i,u,b])
    if mode == 'shuffled':
        if generator is None: raise ValueError('Shuffled control requires its own generator')
        clean,keep = shuffle_targets(clean,pairs,generator)
        eligible &= keep
    weight = eligible.float()*torch.minimum(quality[i,t,a],quality[i,u,b]).detach().float()
    error = F.smooth_l1_loss(predicted,clean,reduction='none',beta=beta)
    loss,denominator = weighted_mean(error,weight,pred)
    active = weight>0
    values = clean[active].detach()
    stats = {'count':int(active.sum()),'target_mean':float(values.mean()) if len(values) else None,
             'target_std':float(values.std(unbiased=False)) if len(values) else None,
             'prediction_mean':float(predicted[active].detach().mean()) if len(values) else None,
             'lag_counts':{str(lag):int(((u-t==lag)&active).sum()) for lag in sorted(set((u-t).tolist()))}}
    stats['by_lag']={str(lag):{'target_mean':float(clean[(u-t==lag)&active].detach().mean()),
                                       'target_std':float(clean[(u-t==lag)&active].detach().std(unbiased=False)),
                                       'physical_lag_ms_nominal':int(lag)*40}
                         for lag in sorted(set((u-t).tolist())) if ((u-t==lag)&active).any()}
    return loss,denominator,stats
