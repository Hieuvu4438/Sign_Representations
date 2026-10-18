"""Read-only pose sources and shared-warp, source-scrubbed near/far pairs."""
from pathlib import Path
import json
import numpy as np
import torch
from pp.base import window as W
from pp.base.joints import COCO_INDEX,OWNED,NC,F_HD,F_UP,F_MO,RH0,LH0,RWRI,LWRI,LSHO,RSHO
from pp.base.masks import sample_view
from pp.closure import source_hidden,resampled_hidden,state_visible


def observation(xy,conf,wh):
    """No score threshold. Pixels preserve aspect ratio; NaN denotes missing."""
    xy=xy.copy(); conf=np.clip(conf.copy(),0,1)
    wh=np.asarray(wh)
    if wh.ndim==1: wh=wh[None,:]
    obs=np.isfinite(xy).all(-1)&((xy>=0)&(xy<=wh[:,None,:])).all(-1)
    sh=obs[:,LSHO]&obs[:,RSHO]
    if sh.any():
        width=np.median(np.linalg.norm(xy[sh,LSHO]-xy[sh,RSHO],axis=-1))
        for base,wrist in ((RH0,RWRI),(LH0,LWRI)):
            far=obs[:,base]&obs[:,wrist]&(np.linalg.norm(xy[:,base]-xy[:,wrist],axis=-1)>.30*width)
            obs[far,base:base+21]=False
    xy[~obs]=np.nan; conf[~obs]=0
    return xy.astype(np.float32),conf.astype(np.float32)


class PoseSource:
    """ASM metadata JSONL or PLctx NPZ. Labels are read only by downstream code.

    NPZ reuse preserves the older PLctx score gates; fresh ASM input is recommended
    for proposal16's observation policy. No data/cache is written here.
    """
    def __init__(self,path,split='train'):
        self.path=Path(path); self.split=split; self.rows=None
        if self.path.suffix=='.npz':
            z=np.load(self.path,allow_pickle=False)
            self.xy=z['xy']; self.conf=z['conf']; self.offsets=z['offsets']; self.fps=z['fps']
            self.wh=z['wh']; self.ids=z['ids']
        else:
            self.rows=[json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
            if any(r.get('official_split',r.get('split',split))!=split for r in self.rows): raise ValueError('Mixed / incorrect split')
            self.ids=np.array([r['sample_id'] for r in self.rows])
        if len(set(self.ids))!=len(self.ids): raise ValueError('Duplicate sample IDs')
    def __len__(self): return len(self.ids)
    def clip(self,i):
        if self.rows is None:
            a,b=self.offsets[i:i+2]
            xy,c=observation(self.xy[a:b].astype(np.float32),self.conf[a:b].astype(np.float32),self.wh[i])
            return xy,c,float(self.fps[i])
        row=self.rows[i]; pose=self.path.parent/row['pose_relpath']
        if pose.suffix=='.pkl':
            import pickle
            with pose.open('rb') as f:raw=pickle.load(f)
            n=len(raw['keypoints']);x=np.full((n,103,2),np.nan,np.float32);c=np.zeros((n,103),np.float32)
            for t,(kp,sc) in enumerate(zip(raw['keypoints'],raw['scores'])):
                kp=np.asarray(kp);sc=np.asarray(sc)
                if kp.ndim==3:
                    if not len(kp):continue
                    kp=kp[0]
                if sc.ndim==2:
                    if not len(sc):continue
                    sc=sc[0]
                if kp.shape!=(133,2) or sc.shape!=(133,):raise ValueError(f'Invalid COCO133 frame in {pose}')
                if not np.any(sc>0):continue  # detector-failure flag, not a confidence threshold
                x[t]=kp[COCO_INDEX]*np.array([row['width'],row['height']])
                c[t]=np.clip(sc[COCO_INDEX]/10.,0,1)
            x,c=observation(x,c,[row['width'],row['height']]);fps=float(row['fps'])
        else:
            with np.load(pose,allow_pickle=False) as z:
                x=z['xy'].astype(np.float32)[:,COCO_INDEX]
                if 'frame_valid' in z:x[~z['frame_valid']]=np.nan
                # ASM caches carry all finite detector points, not score-gated observations.
                scores=z['score'] if 'score' in z else z['scores'] if 'scores' in z else z['confidence'] if 'confidence' in z else None
                if scores is None: raise ValueError(f'{pose}: missing scores/confidence')
                c=np.clip(scores[:,COCO_INDEX].astype(np.float32)/10.,0,1)
                x,c=observation(x,c,z['image_wh'])
                ts=z['timestamps_sec']
                fps=1/float(np.median(np.diff(ts))) if len(ts)>1 else float(row['fps'])
        if len(x)==0 or not np.isfinite(fps) or fps<=0: raise ValueError(f'Invalid pose {pose}')
        return x,c,fps


def positions(n,fps,rng,cfg,augment=True):
    speed=float(rng.uniform(*cfg.speed)) if augment else 1.
    duration=(n-1)/fps
    span=63*speed/25
    start=float(rng.uniform(0,max(0,duration-span))) if augment else 0.
    u=np.linspace(0,1,64)
    if augment and cfg.warp:
        knots=np.array([0.,1/3,2/3,1.])
        values=knots.copy(); values[1:3]+=rng.uniform(-cfg.warp,cfg.warp,2)/3
        from scipy.interpolate import PchipInterpolator
        u=PchipInterpolator(knots,values)(u)
    times=(start+u*span)*fps
    return at_positions(n,times)


def at_positions(n,times):
    i0=np.floor(times+1e-8).astype(np.int64); w=(times-i0).astype(np.float32)
    valid=(times>=0)&(times<=n-1+1e-6)
    i0=i0.clip(0,n-1); i1=(i0+1).clip(0,n-1)
    w=np.where((w<1e-6)|(i0==i1)|~valid,0,w).astype(np.float32)
    return i0,i1,w,valid


def resample(x,c,i0,i1,w,valid):
    # Unlike the inherited 0*NaN expression, an unused endpoint cannot erase a sample.
    need=w>0; ok=np.isfinite(x[i0]).all(-1)&(~need[:,None]|np.isfinite(x[i1]).all(-1))&valid[:,None]
    b=np.where(need[:,None,None],x[i1],x[i0])
    out=(1-w[:,None,None])*x[i0]+w[:,None,None]*b
    cc=(1-w[:,None])*c[i0]+w[:,None]*np.where(need[:,None],c[i1],c[i0])
    return np.where(ok[...,None],out,np.nan).astype(np.float32),np.where(ok,cc,0).astype(np.float32)


def sample_mask(nvalid,rng,cfg,kind,xy):
    mask=sample_view(nvalid,32,rng,kind,ctx_drop=cfg.ctx_drop)
    if cfg.mask=='mixed_subtree' and kind=='far' and rng.random()<.3:
        art=((0,1),(2,3),(4,5,6))[int(rng.integers(3))]
        mask.hidden[:]=False; mask.hidden[list(art),:nvalid]=True
    elif cfg.mask in ('motion','motion_subtree'):
        # Declared content-dependent arms only. Select seeded articulator, extend subtree.
        energy=[]
        for ch in range(4):
            p=xy[:,OWNED[ch]]
            delta=np.diff(p,axis=0)
            energy.append(float(np.nansum(delta*delta))+1e-6)
        prob=np.array(energy)/sum(energy)
        ch=int(rng.choice(4,p=prob))
        sampled_hidden=mask.hidden.copy()
        mask.hidden[:]=False
        if cfg.mask=='motion_subtree':
            # A3: full-time kinematic subtree, ~25% of owned joints.
            # 21-joint hand + 2-joint arm =23/98. Face subtree 52/98 is excluded.
            mask.hidden[[0,1] if ch<2 else [2,3],:nvalid]=True
        else:
            # Preserve the PLCM cell budget and block structure while relocating
            # channels/time according to motion. This isolates content seeding.
            order=np.argsort(-sampled_hidden[:4].sum(-1))
            destinations=rng.choice(4,4,replace=False,p=prob)
            moved=sampled_hidden.copy();drop=mask.drop.copy()
            for old,new in zip(order,destinations):
                moved[new]=sampled_hidden[old];drop[new]=mask.drop[old]
            points=xy[:,OWNED[ch]];de=np.diff(points,axis=0)
            e=np.r_[0.,np.nansum(de*de,axis=(1,2))].reshape(32,2).sum(-1)[:nvalid]+1e-6
            desired=int(rng.choice(nvalid,p=e/e.sum())) if nvalid else 0
            occupied=np.flatnonzero(moved[:,:nvalid].any(0));anchor=int(occupied[0]) if len(occupied) else 0
            mask.hidden[:,:nvalid]=np.roll(moved[:,:nvalid],desired-anchor,axis=1)
            mask.drop[:,:nvalid]=np.roll(drop[:,:nvalid],desired-anchor,axis=1)
    return mask


def build_pair(xy,conf,fps,rng,cfg,kind,override_hidden=None):
    i0,i1,w,valid=positions(len(xy),fps,rng,cfg)
    support=W.support(i0,i1,w,valid)
    tx,tc=resample(xy,conf,i0,i1,w,valid)
    m=sample_mask(int(valid.reshape(32,2).any(1).sum()),rng,cfg,kind,tx)
    if override_hidden is not None: m.hidden[:]=override_hidden
    h=source_hidden(m.hidden,support,len(xy))
    oh=resampled_hidden(h,support)
    p_hidden=m.hidden.copy()
    if cfg.p_budget:
        # Explicit experimental A1-pb: P sees only short S-mask runs.
        # This relaxes S->P closure and is NOT leakage-free relative to S targets.
        p_hidden[:]=False
        for ch in range(NC):
            padded=np.r_[False,m.hidden[ch],False].astype(np.int8)
            bounds=np.flatnonzero(np.diff(padded))
            for start,end in zip(bounds[::2],bounds[1::2]):
                if end-start<=4:p_hidden[ch,start:end]=True
    ph=resampled_hidden(source_hidden(p_hidden,support,len(xy)),support)
    sx=xy.copy(); sc=conf.copy(); sx[h]=np.nan; sc[h]=0
    sx,sc=resample(sx,sc,i0,i1,w,valid)
    # Metadata/closure are independent of hidden values. Same warp for teacher/student.
    t_aff=W.camera_affine(rng,5.,.05,0.)
    s_aff=W.camera_affine(rng,10.,.10,.10)
    aug_state=rng.bit_generator.state
    if cfg.morph: sx=W.morphology(sx,rng)
    if cfg.noise: sx=W.keypoint_noise(sx,sc,rng,cfg.noise)
    rng.bit_generator.state=aug_state
    px=tx.copy()
    if cfg.morph: px=W.morphology(px,rng)
    if cfg.noise: px=W.keypoint_noise(px,tc,rng,cfg.noise)
    aligned=np.arange(32,dtype=np.float32)
    if cfg.aligned_warp:
        times=i0+w*(i1-i0)
        regular=np.linspace(times[0],times[-1],64)
        ti=at_positions(len(xy),regular)
        tx,tc=resample(xy,conf,*ti)
        frame=(times-regular[0])/max(1e-8,regular[-1]-regular[0])*63
        aligned=((frame.reshape(32,2).mean(-1)-.5)/2).clip(0,31).astype(np.float32)
    # P student augmentation needs full coordinates with the SAME RNG realization.
    # Reset augmentation RNG, then scrub by closure after features are formed.
    # Store P input separately to avoid cross-channel per-clip statistics in S.
    return dict(t_xy=np.nan_to_num(tx),t_obs=np.isfinite(tx).all(-1),t_conf=tc,
                s_xy=np.nan_to_num(sx),s_obs=np.isfinite(sx).all(-1),s_conf=sc,
                valid_f=valid,valid_b=m.valid,hidden=m.hidden,target=m.target,
                visible=state_visible(m.hidden,oh,m.valid,m.drop),output_hidden=oh,p_output_hidden=ph,alignment=aligned,
                p_xy=np.nan_to_num(px),p_obs=np.isfinite(px).all(-1),p_conf=tc,
                t_aff=t_aff,s_aff=s_aff,positions=np.stack([i0,i1,w],-1),source_hidden=h)


class Windows(torch.utils.data.Dataset):
    def __init__(self,source,cfg,seed=None):
        self.source,self.cfg=source,cfg; self.seed=cfg.seed if seed is None else seed; self.epoch=0
    def __len__(self): return len(self.source)
    def __getitem__(self,i):
        rng=np.random.default_rng([self.seed,self.epoch,i])
        x,c,fps=self.source.clip(i)
        x,c,flipped=W.canonicalise(x,c)
        return {kind:build_pair(x,c,fps,rng,self.cfg,kind) for kind in ('near','far')}


def collate(items):
    keys=('t_xy','t_obs','t_conf','s_xy','s_obs','s_conf','p_xy','p_obs','p_conf','valid_f','valid_b','hidden','target','visible','output_hidden','p_output_hidden','alignment','t_aff','s_aff','positions')
    return {v:{k:torch.from_numpy(np.stack([it[v][k] for it in items])) for k in keys} for v in ('near','far')}
