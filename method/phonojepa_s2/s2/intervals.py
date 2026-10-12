"""S2 deterministic interval targets; no target values enter the primary encoder.

Log-signatures use the free Lie basis (level 1, then [e_i,e_j], i<j).
For piecewise-linear paths the degree-two coefficient is signed Levy area.
All supports below are explicit. Missing samples invalidate the entire interval;
they are never bridged, including for differences at larger lags.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from s2 import features as Fe
from s2.joints import (RWRI,LWRI,RELB,LELB,RSHO,LSHO,RH0,LH0,NOSE_B,
                       OWNED,DUPLICATES,NC,J)


@dataclass(frozen=True)
class PathSpec:
    name: str
    terms: tuple
    group: int  # 0 absolute hand, 1 arm, 2 shape, 3 relative placement


PATHS = (
    PathSpec('D_WRIST',((RWRI,1.),),0), PathSpec('N_WRIST',((LWRI,1.),),0),
    PathSpec('D_INDEX',((RH0+8,1.),),0), PathSpec('N_INDEX',((LH0+8,1.),),0),
    PathSpec('D_FOREARM',((RWRI,1.),(RELB,-1.)),1),
    PathSpec('N_FOREARM',((LWRI,1.),(LELB,-1.)),1),
    PathSpec('D_UPPERARM',((RELB,1.),(RSHO,-1.)),1),
    PathSpec('N_UPPERARM',((LELB,1.),(LSHO,-1.)),1),
    PathSpec('D_SHAPE',((RH0+8,1.),(RH0,-1.)),2),
    PathSpec('N_SHAPE',((LH0+8,1.),(LH0,-1.)),2),
    PathSpec('D_HEAD',((RWRI,1.),(NOSE_B,-1.)),3),
    PathSpec('N_HEAD',((LWRI,1.),(NOSE_B,-1.)),3),
    PathSpec('HAND_REL',((RWRI,1.),(LWRI,-1.)),3),
    PathSpec('D_CHAIN_BEND',((RELB,1.),(RSHO,-.5),(RWRI,-.5)),1),
    PathSpec('N_CHAIN_BEND',((LELB,1.),(LSHO,-.5),(LWRI,-.5)),1),
)
SCALES = (8,16,32)
P = len(PATHS)
MOTION_D, SIGNATURE_D, LEVEL_D, INPUT_D = 12,10,4,27


def layout(device=None):
    """Scale-major, then path-major, then start; same order in all consumers."""
    rows=[(p,s,n,k) for k,n in enumerate(SCALES) for p in range(P)
          for s in range(0,64-n+1,n//2)]
    return torch.tensor(rows,device=device,dtype=torch.long).unbind(-1)


def lead_lag(x):
    """Standard axis-aligned lift: (x0,x0),(x1,x0),(x1,x1),... ."""
    out=torch.stack([torch.cat([x[...,1:,:],x[...,:-1,:]],-1),
                     torch.cat([x[...,1:,:],x[...,1:,:]],-1)],-2).flatten(-3,-2)
    return torch.cat([torch.cat([x[...,:1,:],x[...,:1,:]],-1),out],-2)


def logsig2(x):
    """Exact degree-two log-signature of piecewise-linear (...,samples,d)."""
    dx=x[...,1:,:]-x[...,:-1,:]
    prev=dx.cumsum(-2)-dx
    ij=torch.triu_indices(x.shape[-1],x.shape[-1],offset=1,device=x.device)
    area=.5*(prev[...,ij[0]]*dx[...,ij[1]]-prev[...,ij[1]]*dx[...,ij[0]]).sum(-2)
    return torch.cat([dx.sum(-2),area],-1)


def motion(x,times,lags=(1,4,8),normalization='physical'):
    """12 fixed slots: (mean signed dx/dt, mean norm dx/dt)x3, length,
    mean step speed, mean absolute radial change/dt. Frame control uses dt=1.
    Radial change on SHAPE paths measures index-to-wrist extension change.
    """
    if normalization not in ('physical','frame'):raise ValueError(normalization)
    out=[];ok=[]
    for lag in (1,4,8):
        use=lag in lags and lag<x.shape[-2]
        if use:
            dt=times[...,lag:]-times[...,:-lag]
            delta=x[...,lag:,:]-x[...,:-lag,:]
            vel=delta/(dt[...,None] if normalization=='physical' else 1.)
            out.append(torch.cat([vel.mean(-2),vel.norm(dim=-1).mean(-1,keepdim=True)],-1))
        else:out.append(x.new_zeros(*x.shape[:-2],3))
        ok.extend([use]*3)
    dx=x[...,1:,:]-x[...,:-1,:];dt=times[...,1:]-times[...,:-1]
    speed=dx.norm(dim=-1)/(dt if normalization=='physical' else 1.)
    radial=(x[...,1:,:].norm(dim=-1)-x[...,:-1,:].norm(dim=-1)).abs()
    radial=radial/(dt if normalization=='physical' else 1.)
    out.append(torch.stack([dx.norm(dim=-1).sum(-1),speed.mean(-1),radial.mean(-1)],-1))
    return torch.cat(out,-1),torch.tensor(ok+[True]*3,device=x.device)


def descriptors(xy,obs,conf,valid,times,anchor=None,aff=None,forbidden=None,
                signature='plain',lags=(1,4,8),normalization='physical'):
    """Return clean or scrubbed path descriptors with dimension-level validity.

    `forbidden` is the OR of exact raw interpolation supports, separate from
    observation dropout. Shoulders are declared visible coordinate side-info.
    Computation is fp32 even inside autocast. Values of invalid intervals are 0.
    """
    if signature not in ('plain','leadlag'):raise ValueError(signature)
    if xy.shape[1]!=64 or times.shape!=valid.shape:raise ValueError('Expected 64 samples and matching times')
    if not torch.isfinite(times).all() or not ((times[:,1:]-times[:,:-1])>0).all():
        raise ValueError('Source timestamps must be finite and strictly increasing')
    with torch.autocast(device_type=xy.device.type,enabled=False):
        c,s,a=Fe.anchors(xy,obs) if anchor is None else anchor
        body=Fe.to_body_frame(xy.float(),c,s,a,aff)
        coeff=body.new_zeros(P,J)
        for i,path in enumerate(PATHS):
            for j,w in path.terms:coeff[i,j]=w
        body=torch.where((obs&torch.isfinite(xy).all(-1))[...,None],body,0.)
        points=torch.einsum('btjd,pj->bptd',body,coeff)
        support=coeff!=0
        observed=obs&valid[...,None]&torch.isfinite(xy).all(-1)
        path_ok=(observed[:,None]|~support[None,:,None]).all(-1)
        path_conf=torch.where(support[None,:,None],conf[:,None].float(),1.).amin(-1)
        if forbidden is not None:
            path_ok &= ~(forbidden[:,None]&support[None,:,None]).any(-1)
        result={k:[] for k in ('motion','signature','level','duration','valid','quality','motion_valid','signature_valid','mae','mae_valid')}
        for n in SCALES:
            x=points.unfold(2,n,n//2).transpose(-1,-2) # B,P,W,L,2
            ts=times.unfold(1,n,n//2)[:,None].expand(-1,P,-1,-1)
            good=path_ok.unfold(2,n,n//2).all(-1)
            quality=path_conf.unfold(2,n,n//2).amin(-1)
            # Never feed zero-filled discontinuities through a path estimator.
            x=torch.where(good[...,None,None],x,0.)
            mot,mv=motion(x,ts,lags,normalization)
            sig=logsig2(lead_lag(x) if signature=='leadlag' else x)
            sv=torch.arange(SIGNATURE_D,device=xy.device)<sig.shape[-1]
            sig=torch.nn.functional.pad(sig,(0,SIGNATURE_D-sig.shape[-1]))
            level=torch.cat([x[...,0,:],x[...,-1,:]],-1)
            mae=torch.nn.functional.pad((x[...,1:,:]-x[...,:-1,:]).flatten(-2),(0,62-2*(n-1)))
            maev=good[...,None]&(torch.arange(62,device=xy.device)<2*(n-1))
            values={'motion':mot,'signature':sig,'level':level,
                    'duration':(ts[...,-1]-ts[...,0])[...,None],
                    'valid':good,'quality':quality*good,
                    'motion_valid':good[...,None]&mv,'signature_valid':good[...,None]&sv,
                    'mae':mae,'mae_valid':maev}
            for key,val in values.items():
                result[key].append(val.flatten(1,2))
        result={k:torch.cat(v,1) for k,v in result.items()}
        # Finite and benign even for invalid intervals, including timing field.
        for key in ('motion','signature','level'):
            result[key]=torch.where(result['valid'][...,None],result[key],0.)
        result['input']=torch.cat([result['motion'],result['signature'],result['level'],
                                   result['duration']],-1)
        return result


def selectors(v):
    """[B,375,256] masked base queries with a hidden endpoint in this interval.

    Duplicate-aware ownership is checked at the same bin as the raw support;
    unrelated masked tokens cannot supervise a fully visible path.
    """
    hidden=(v['forbidden']&v['valid_f'][...,None]).clone()
    for a,b in DUPLICATES:
        both=hidden[:,:,a]|hidden[:,:,b]
        hidden[:,:,a]=both;hidden[:,:,b]=both
    owners=torch.zeros(P,NC,J,dtype=torch.bool,device=hidden.device)
    for p,path in enumerate(PATHS):
        ids={j for j,_ in path.terms}
        for a,b in DUPLICATES:
            if a in ids or b in ids:ids.update((a,b))
        for ch,js in OWNED.items():
            owners[p,ch,list(ids&set(js))]=True
    h=(hidden[:,None,:,None,:]&owners[None,:,None]).any(-1)
    h=h.reshape(len(hidden),P,32,2,NC).any(3)
    cells=h&v['target'].transpose(1,2)[:,None]
    p,start,n,_=layout(hidden.device)
    bins=torch.arange(32,device=hidden.device)
    within=(bins[None]*2>=start[:,None])&(bins[None]*2<start[:,None]+n[:,None])
    return (cells[:,p]&within[None,:,:,None]).flatten(-2).float()
