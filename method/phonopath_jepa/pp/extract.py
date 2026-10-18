"""Frozen teacher clip embeddings, ordered/activity readouts, real-fps robustness."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pp.base import window as W
from pp.base import features as S
from pp.base.joints import RWRI
from pp.config import Config
from pp.data import PoseSource,at_positions,resample
from pp.features import assemble
from pp.model import Encoder
from pp.signature import RobustStats
from pp.fit_stats import fingerprint


def weighted_pool(h,w):
    return (h*w[...,None]).sum(1)/w.sum(1).clamp_min(1e-6)[:,None]


def readouts(h,pack,cfg,view):
    B,N,d=h.shape;out={};offset=0
    valid=view['valid_f'];c,s,a=S.anchors(view['t_xy'],view['t_obs'])
    p=S.to_body_frame(view['t_xy'],c,s,a)
    height=-p[:,:,RWRI,1]
    observed=view['t_obs'][:,:,RWRI]&valid
    # Rest level across each inference window; only inference uses clip statistics.
    heights=height.masked_fill(~observed,float('nan'))
    rest=view.get('clip_rest',torch.nan_to_num(torch.nanquantile(heights,.1,dim=1),nan=0.)+.25)
    activity=torch.sigmoid((height-rest[:,None])/.1)*observed
    energy=torch.zeros(B,32,device=h.device)
    if cfg.paths:
        en=pack['p']['energy'][:,1,:15]
        centres=torch.tensor([start+(size-1)/2 for start,size,scale in __import__('pp.ptokens',fromlist=['WINDOWS']).WINDOWS[:15]],device=h.device)
        bins=torch.arange(32,device=h.device)*2+.5
        idx=(bins[:,None]-centres[None]).abs().argmin(-1)
        energy=en[:,idx]
    mw=activity.reshape(B,32,2).mean(-1)*(1+energy)
    # Split at the temporal midpoint of the active interval, preserving order.
    active=(activity>.5)&valid
    times=torch.arange(64,device=h.device)[None].expand(B,-1)
    lo=times.masked_fill(~active,64).min(-1).values;hi=times.masked_fill(~active,-1).max(-1).values
    middle=view.get('active_midpoint',torch.where(hi>=lo,(lo+hi).float()/2,torch.full_like(lo,31.5,dtype=torch.float32)))
    def sgroup(timeweight):
        hs=h[:,:224].reshape(B,32,7,d);use=pack['s_present'].transpose(1,2)&valid.reshape(B,32,2).any(-1)[:,:,None]
        return torch.cat([weighted_pool(hs[:,:,list(g)].flatten(1,2),
                                       (use[:,:,list(g)]*timeweight[:,:,None]).flatten(1,2))
                          for g in ((0,1),(2,3),(4,5,6))],-1)
    def pgroup(timeweight):
        hp=h[:,offset:offset+110].reshape(B,5,22,d);use=pack['p']['present']
        return torch.cat([weighted_pool(hp[:,list(g)].flatten(1,2),
                                       (use[:,list(g)]*timeweight[:,None]).flatten(1,2)) for g in ((0,1),(2,3),(4,))],-1)
    if cfg.state:
        out['grp']=sgroup(torch.ones_like(mw));offset=224
    if cfg.paths:
        if not cfg.state:out['grp']=pgroup(torch.ones(B,22,device=h.device))
        out['grp+P']=torch.cat(([out['grp']] if cfg.state else [])+[pgroup(torch.ones(B,22,device=h.device))],-1)
    else:out['grp+P']=out['grp']
    from pp.ptokens import WINDOWS
    pt=torch.tensor([(start+(size-1)/2) for start,size,scale in WINDOWS],device=h.device)
    nearest=(pt[:,None]-(torch.arange(32,device=h.device)*2+.5)[None]).abs().argmin(-1)
    parts=[sgroup(mw)] if cfg.state else []
    if cfg.paths:parts.append(pgroup(mw[:,nearest]))
    out['mw']=torch.cat(parts,-1)
    halves=[]
    for first in (True,False):
        sb=(torch.arange(32,device=h.device)[None]*2+.5<=middle[:,None])
        pb=pt[None]<=middle[:,None]
        if not first:sb=~sb;pb=~pb
        parts=[sgroup(sb.float())] if cfg.state else []
        if cfg.paths:parts.append(pgroup(pb.float()))
        halves.append(torch.cat(parts,-1))
    out['ord2']=torch.cat(halves,-1)
    return out


class Extractor:
    def __init__(self,checkpoint,device='cpu'):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.cfg=Config(**ck['config']);self.device=device
        self.encoder=Encoder(self.cfg).to(device).eval().requires_grad_(False)
        self.encoder.load_state_dict({k[len('teacher.'):]:v for k,v in ck['model'].items() if k.startswith('teacher.')})
        self.stats=RobustStats(ck['stats'],self.cfg.asinh) if self.cfg.paths else None
    @torch.inference_mode()
    def windows(self,xy,conf,fps,stride=2,batch=32,target_fps=25.,drop=0.,jitter=0.,reverse=False,seed=0):
        rng=np.random.default_rng(seed)
        if reverse:xy=xy[::-1].copy();conf=conf[::-1].copy()
        xy,conf,_=W.canonicalise(xy,conf)
        if jitter:xy=W.keypoint_noise(xy,np.zeros_like(conf),rng,jitter)
        if drop:
            keep=rng.random(len(xy))>=drop
            # Real frame loss: mark missing frames, keep original timestamps.
            xy=xy.copy();conf=conf.copy();xy[~keep]=np.nan;conf[~keep]=0
        xx=torch.as_tensor(np.nan_to_num(xy)[None],device=self.device)
        oo=torch.as_tensor(np.isfinite(xy).all(-1)[None],device=self.device)
        c,s,a=S.anchors(xx,oo);body=S.to_body_frame(xx,c,s,a)
        height=-body[0,:,RWRI,1]
        finite=oo[0,:,RWRI]
        rest=float(torch.quantile(height[finite],.1)+.25) if finite.any() else 0.
        active=torch.nonzero(finite&(height>rest)).flatten()
        active_mid=float((active[0]+active[-1])/2)/fps*target_fps if len(active) else (len(xy)-1)/fps*target_fps/2
        # target_fps<25 is genuine sparse sampling, never restored to 64/25Hz.
        nsamp=max(1,int(np.floor((len(xy)-1)/fps*target_fps))+1)
        starts=list(range(0,max(1,nsamp-64+1),stride))
        collected={}
        for st in range(0,len(starts),batch):
            xs=[];cs=[];vs=[]
            for start in starts[st:st+batch]:
                ids=at_positions(len(xy),(start+np.arange(64))/target_fps*fps)
                x,c=resample(xy,conf,*ids);xs.append(x);cs.append(c);vs.append(ids[-1])
            x=np.stack(xs);c=np.stack(cs);valid=np.stack(vs);obs=np.isfinite(x).all(-1)
            tx=torch.as_tensor(np.nan_to_num(x),device=self.device);ob=torch.as_tensor(obs,device=self.device)
            tc=torch.as_tensor(c,device=self.device);vf=torch.as_tensor(valid,device=self.device)
            view=dict(t_xy=tx,t_obs=ob,t_conf=tc,valid_f=vf,output_hidden=torch.zeros_like(ob),
                      target=torch.zeros(len(x),7,32,device=self.device,dtype=torch.bool),
                      t_aff=torch.eye(2,device=self.device).expand(len(x),-1,-1))
            view['clip_rest']=torch.full((len(x),),rest,device=self.device)
            view['active_midpoint']=torch.tensor([active_mid-start for start in starts[st:st+batch]],device=self.device)
            pack=assemble(view,self.cfg,self.stats,teacher=True)
            with torch.autocast(device_type=torch.device(self.device).type,dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
                h,_,_=self.encoder(pack)
            for k,v in readouts(h.float(),pack,self.cfg,view).items():collected.setdefault(k,[]).append(v.cpu().numpy())
        return {k:np.concatenate(v) for k,v in collected.items()}
    def clip(self,*args,**kw):return {k:v.mean(0,dtype=np.float64).astype(np.float32) for k,v in self.windows(*args,**kw).items()}


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--split',choices=['train','val','test'],required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--fps',type=float,default=25);ap.add_argument('--drop',type=float,default=0)
    ap.add_argument('--jitter',type=float,default=0);ap.add_argument('--reverse',action='store_true');ap.add_argument('--stride',type=int,default=2)
    ap.add_argument('--dtype',choices=['float16','float32'],default='float16')
    a=ap.parse_args(argv);torch.set_num_threads(4)
    source=PoseSource(a.source,a.split);ext=Extractor(a.ckpt,a.device);acc={};rows=[]
    if source.rows is None:raise ValueError('Extraction needs metadata JSONL with labels/signer for evaluator')
    for i in range(min(a.limit or len(source),len(source))):
        x,c,fps=source.clip(i)
        values=ext.clip(x,c,fps,stride=a.stride,target_fps=a.fps,drop=a.drop,jitter=a.jitter,reverse=a.reverse,seed=i)
        for k,v in values.items():acc.setdefault(k,[]).append(v)
        rows.append(source.rows[i])
        if i%250==0:print(i,len(source),flush=True)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError(f'Output already exists: {path}')
    meta=dict(split=a.split,checkpoint_sha256=fingerprint(a.ckpt),source_sha256=fingerprint(a.source),stride=a.stride,fps=a.fps,
              drop=a.drop,jitter=a.jitter,reverse=a.reverse,limited=bool(a.limit),raw_window_mean=True,dtype=a.dtype)
    temporary=path.with_suffix('.partial.npz')
    np.savez(temporary,**{k:np.stack(v).astype(a.dtype) for k,v in acc.items()},ids=np.array([r['sample_id'] for r in rows]),
             labels=np.array([r['class_id'] for r in rows]),gloss=np.array([r['gloss'] for r in rows]),
             signer=np.array([r['signer_id'] for r in rows]),metadata=np.array(json.dumps(meta)),rows=np.array(json.dumps(rows)))
    temporary.replace(path)
if __name__=='__main__':main()
