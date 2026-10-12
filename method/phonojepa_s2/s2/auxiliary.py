"""Explicit S1 targets. Geometry is computed in the student's known view frame.

No target enters the encoder. Only raw-hidden endpoint-supported dimensions are
supervised. Always-visible shoulders are excluded from explicit body targets.
Normalization is fitted on training clips only, with a separate scale per group.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from s2 import features as Fe
from s2.joints import NOSE_B, LHIP, RHIP, RWRI, LWRI, TORSO, D_LOC, N_LOC

BODY_IDS = [NOSE_B, LHIP, RHIP]
# Five vectors: D/body, N/body, D/head, N/head, D/N; vector + norm.
REL_SUPPORT = ((RWRI,LHIP,RHIP), (LWRI,LHIP,RHIP), (RWRI,NOSE_B),
               (LWRI,NOSE_B), (RWRI,LWRI))


def geometry(xy, obs, valid_f, anchor=None, aff=None):
    B,T = valid_f.shape
    c,s,a = Fe.anchors(xy,obs) if anchor is None else anchor
    p = Fe.to_body_frame(xy,c,s,a,aff)
    valid = obs & valid_f[...,None]
    ids = BODY_IDS
    ok = valid[:,:,ids]
    level = torch.where(ok[...,None],p[:,:,ids],0).reshape(B,T,6)
    # Paired differences only: no implicit interpolation through a missing frame.
    lb = level.reshape(B,T//2,2,6); ob = ok.repeat_interleave(2,-1).reshape(B,T//2,2,6)
    count = ob.sum(2)
    mean = (lb*ob).sum(2)/count.clamp(min=1)
    delta = lb[:,:,1]-lb[:,:,0]; dok = ob[:,:,0]&ob[:,:,1]
    body = torch.cat([mean,delta],-1)
    bw = torch.cat([count>0,dok],-1).float()
    hip = (p[:,:,LHIP]+p[:,:,RHIP])/2
    vectors = [p[:,:,RWRI]-hip,p[:,:,LWRI]-hip,p[:,:,RWRI]-p[:,:,NOSE_B],
               p[:,:,LWRI]-p[:,:,NOSE_B],p[:,:,RWRI]-p[:,:,LWRI]]
    rel = torch.stack([torch.cat([v,v.norm(dim=-1,keepdim=True)],-1) for v in vectors],-2).flatten(-2)
    rw = torch.stack([valid[:,:,list(js)].all(-1) for js in REL_SUPPORT],-1).repeat_interleave(3,-1)
    rr = rel.reshape(B,T//2,2,15); ww = rw.reshape(B,T//2,2,15)
    cnt = ww.sum(2)
    rel = (torch.where(ww,rr,0)).sum(2)/cnt.clamp(min=1)
    return {'body': (body,bw), 'relations':(rel,(cnt>0).float())}


def loss_weights(v, targets):
    """Weights include per-dimension validity, confidence, and hidden endpoints.

    Relations assigned to D_LOC/N_LOC/TORSO queries only when an endpoint of
    that relation was hidden by the relevant query's channel mask. This prevents
    learning a visible-coordinate copying task under an unrelated masked query.
    """
    B,T = v['valid_f'].shape
    out = {}
    for group,(y,valid) in targets.items():
        out[group] = torch.zeros(B,T//2,8,y.shape[-1],device=y.device)
    hidden_joint = v['t_obs'] & ~v['s_obs']
    h = hidden_joint.reshape(B,T//2,2,-1).any(2)
    conf = (v['t_conf']*v['t_obs']).reshape(B,T//2,2,-1).mean(2)
    body_valid = targets['body'][1]
    # Explicit body head predicts only nose/hips, never anchor shoulders.
    bh = h[:,:,BODY_IDS].repeat_interleave(2,-1)
    bc = conf[:,:,BODY_IDS].repeat_interleave(2,-1)
    weight = torch.cat([bh,bh],-1)*torch.cat([bc,bc],-1)*body_valid
    out['body'][:,:,TORSO] = weight*v['target'][:,TORSO,:,None]
    rel_valid = targets['relations'][1]
    for ch, joint in [(D_LOC,RWRI),(N_LOC,LWRI),(TORSO,None)]:
        for k,ids in enumerate(REL_SUPPORT):
            endpoints = [j for j in ids if (j==joint if joint is not None else j in BODY_IDS)]
            if not endpoints: continue
            unseen = h[:,:,endpoints].any(-1)
            quality = conf[:,:,list(ids)].amin(-1)
            out['relations'][:,:,ch,3*k:3*k+3] = (unseen*quality*v['target'][:,ch])[...,None]*rel_valid[:,:,3*k:3*k+3]
    return out


class Auxiliary(nn.Module):
    def __init__(self, dp):
        super().__init__()
        self.heads=nn.ModuleDict({'body':nn.Linear(dp,12),'relations':nn.Linear(dp,15)})
        for name,d in [('body',12),('relations',15)]:
            self.register_buffer(name+'_mean',torch.zeros(d))
            self.register_buffer(name+'_std',torch.ones(d))

    def loss(self, q, targets, weights):
        q=q.reshape(q.shape[0],32,8,-1)
        out={}
        for name,(y,_) in targets.items():
            y=(y-getattr(self,name+'_mean'))/getattr(self,name+'_std')
            y=y.clamp(-8,8)
            err=F.smooth_l1_loss(self.heads[name](q).float(),y[:,:,None].expand(-1,-1,8,-1),reduction='none',beta=.5)
            w=weights[name]
            # Mean dimensions within query, then mean valid queries.
            per=(err*w).sum(-1)/w.sum(-1).clamp(min=1e-8)
            present=w.sum(-1)>0
            out[name]=(per*present).sum()/present.sum().clamp(min=1)
        return out

    @torch.no_grad()
    def fit(self, batches):
        for name in self.heads:
            vals=torch.cat([torch.where(b[name][1]>0,b[name][0],torch.nan).reshape(-1,b[name][0].shape[-1]) for b in batches])
            mu=torch.nan_to_num(torch.nanmean(vals,0))
            sd=torch.nan_to_num(torch.nanmean((vals-mu)**2,0),nan=1.).sqrt().clamp(min=.05)
            getattr(self,name+'_mean').copy_(mu)
            getattr(self,name+'_std').copy_(sd)
