"""Pose-only window/group extraction; frozen PL gate0 and fusion share readout."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import torch
from fp.paths import ASLC_ROOT,ASLC_SPLITS,OBS_META
from fp.data.cache import load_clip
from fp.data import window as W
from fp.utils import atomic_json,sha256


def split_rows(split):
    with open(ASLC_SPLITS/(split+'.csv')) as f:
        return [{'sample_id':Path(r['Video file']).stem,'video':str(ASLC_ROOT/'videos'/r['Video file']),
                 'gloss':r['Gloss'],'signer':r['Participant ID'],'split':split} for r in csv.DictReader(f)]


class Prepare(torch.utils.data.Dataset):
    def __init__(self,rows,cache=None,stride=32):
        self.rows=rows;self.cache=cache;self.stride=stride
        self.lookup={str(s):i for i,s in enumerate(cache.ids)} if cache is not None else {}
        self.meta=json.loads(OBS_META.read_text())
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];sid=row['sample_id']
        if sid in self.lookup:xy,conf,fps=self.cache.clip(self.lookup[sid])
        else:
            if sid in self.meta:width,height,fps,_=self.meta[sid]
            else:
                import cv2
                cv2.setNumThreads(1);cap=cv2.VideoCapture(row['video'])
                width,height,fps=[cap.get(k) for k in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS)]
                cap.release()
                if min(width,height,fps)<=0:raise ValueError('Invalid video metadata: '+sid)
            xy,conf=load_clip(sid,width,height)
        if not len(xy):raise ValueError('Empty pose: '+sid)
        xy,conf,_=W.canonicalise(xy,conf)
        eff=min(fps,25.);nsamp=max(1,int(np.floor(len(xy)/fps*eff)))
        starts=list(range(0,max(1,nsamp-64+1),self.stride))
        result={k:[] for k in ('t_xy','t_obs','t_conf','valid_f','valid_b')}
        for start in starts:
            i0,i1,w,valid=W.window_positions(len(xy),fps,64,start/eff,1.)
            x,c,_=W.resample(xy,conf,i0,i1,w,valid)
            result['t_xy'].append(np.nan_to_num(x));result['t_obs'].append(np.isfinite(x).all(-1));result['t_conf'].append(c)
            result['valid_f'].append(valid);result['valid_b'].append(valid.reshape(32,2).any(-1))
        return {k:np.stack(v) for k,v in result.items()}|{'sample_id':sid}


def l2(x):return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-12)


@torch.no_grad()
def encode_prepared(enc,p,device,chunk=32):
    fused=[];base=[];graphs=[]
    for start in range(0,len(p['valid_f']),chunk):
        v={k:torch.as_tensor(x[start:start+chunk],device=device) for k,x in p.items() if k!='sample_id'}
        v['s_xy']=v['t_xy'];v['s_obs']=v['t_obs'];v['s_conf']=v['t_conf']
        n=len(v['t_xy']);v['s_jh']=torch.zeros_like(v['t_obs']);v['forbidden']=torch.zeros_like(v['t_obs'])
        v['t_aff']=v['s_aff']=torch.eye(2,device=device)[None].expand(n,-1,-1)
        v['visible']=v['valid_b'][:,None].expand(-1,8,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):o=enc(v,'t')
        h=o['fused'].float().reshape(n,32,7,enc.d);b=o['base'].float().reshape_as(h)
        use=(o['present'][:,:7]&v['visible'][:,:7]).transpose(1,2)
        def pool(x):
            out=[]
            for ids in ((0,1),(2,3),(4,5,6)):
                w=use[:,:,list(ids)];out.append((x[:,:,list(ids)]*w[...,None]).sum((1,2))/w.sum((1,2)).clamp(min=1)[:,None])
            return torch.cat(out,-1).cpu().numpy()
        fused.append(pool(h));base.append(pool(b))
        gt=o['graph_tokens'].float();gv=o['graph_present'].reshape(n,32,4)
        gp=[]
        for ids in ((1,),(2,),(0,3)):
            w=gv[:,:,list(ids)]
            gp.append((gt[:,:,list(ids)]*w[...,None]).sum((1,2))/w.sum((1,2)).clamp(min=1)[:,None])
        graphs.append(torch.cat(gp,-1).cpu().numpy())
    # Raw window mean then video L2 (official dictionary convention).
    return {'fusion':l2(np.concatenate(fused).mean(0,keepdims=True))[0].astype(np.float32),
            'pl_gate0':l2(np.concatenate(base).mean(0,keepdims=True))[0].astype(np.float32),
            'graph_only':l2(np.concatenate(graphs).mean(0,keepdims=True))[0].astype(np.float32)}


def worker_init(_):torch.set_num_threads(1)


def extract(enc,rows,cache,out,identity,device,workers=4,stride=32,stop=lambda:False,notify=lambda *a:None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    existing={};pending=[]
    for i,row in enumerate(rows):
        sid=row['sample_id'];path=out/(sid+'.npz')
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as z:
                    if str(z['identity'])==identity:
                        existing[i]={k:z[k].copy() for k in ('fusion','pl_gate0','graph_only')};continue
            except (ValueError,OSError,EOFError):pass
        pending.append((i,row))
    dataset=Prepare([r for _,r in pending],cache,stride)
    loader=torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=workers,
        multiprocessing_context='spawn' if workers else None,worker_init_fn=worker_init,generator=torch.Generator().manual_seed(918))
    enc.eval()
    for j,p in enumerate(loader):
        if stop():raise InterruptedError('Extraction interrupted; per-clip cache retained')
        i,row=pending[j];value=encode_prepared(enc,p,device)
        path=out/(row['sample_id']+'.npz');tmp=path.with_suffix('.tmp.npz')
        np.savez(tmp,**value,identity=np.array(identity));tmp.replace(path)
        existing[i]=value
        if j%100==0 or j+1==len(pending):notify('extract',len(existing),len(rows))
    if stop():raise InterruptedError('Extraction interrupted')
    return {k:np.stack([existing[i][k] for i in range(len(rows))]) for k in ('fusion','pl_gate0','graph_only')}
