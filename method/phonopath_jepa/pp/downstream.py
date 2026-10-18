"""Supervised G0 veto, frozen linear probe, fine-tune and k-shot ISLR.

Supervised artifacts are separate and cannot initialize pp.train (SSL).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pp.config import Config
from pp.data import PoseSource,Windows,collate
from pp.features import assemble
from pp.model import Encoder
from pp.extract import readouts
from pp.signature import RobustStats
from pp.train import atomic_json,to_device,lr_at
from pp.fit_stats import fingerprint


def select_labels(rows,fraction=1.,k=0,n_glosses=0,seed=0):
    """Fixed random gloss selection (seed 0 across seeds); examples vary by seed."""
    y=np.array([r['gloss'] for r in rows]);classes=np.unique(y)
    if n_glosses:
        if n_glosses>len(classes):raise ValueError('Not enough glosses for requested subset')
        classes=np.sort(np.random.default_rng(0).choice(classes,n_glosses,replace=False))
    rng=np.random.default_rng(seed);idx=[]
    for c in classes:
        candidates=np.flatnonzero(y==c)
        if k and len(candidates)<k:raise ValueError(f'Insufficient examples for {k}-shot: {c}')
        count=k if k else max(1,int(round(len(candidates)*fraction)))
        idx.extend(rng.choice(candidates,count,replace=False).tolist())
    return np.array(idx),classes


def scores(logits,labels):
    pred=logits.argmax(-1);top=np.argsort(logits,axis=1)[:,-min(5,logits.shape[1]):]
    return dict(top1=float(100*np.mean(pred==labels)),top5=float(100*np.mean((top==labels[:,None]).any(-1))),
                macro=float(100*np.mean([np.mean(pred[labels==c]==c) for c in np.unique(labels)])),n=len(labels))


def run(cfg,train,val,out,steps,checkpoint=None,linear=False,fraction=1.,k=0,n_glosses=0):
    if train.rows is None or val.rows is None:raise ValueError('Supervised mode requires metadata JSONL')
    if train.split!='train' or val.split!='val':raise ValueError('G0 / model selection may use train and val only')
    if set(train.ids)&set(val.ids):raise ValueError('Train/val ID overlap')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed)
    encoder=Encoder(cfg).to(cfg.device)
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck.get('supervised'):raise ValueError('Expected SSL checkpoint for downstream transfer')
        encoder.load_state_dict({k[len('teacher.'):]:v for k,v in ck['model'].items() if k.startswith('teacher.')})
        stats=RobustStats(ck['stats'],cfg.asinh) if cfg.paths else None
    else:stats=RobustStats.load(cfg.stats,cfg.asinh) if cfg.paths else None
    idx,classes=select_labels(train.rows,fraction,k,n_glosses,cfg.seed)
    labels={str(c):i for i,c in enumerate(classes)}
    val_idx=[i for i,r in enumerate(val.rows) if r['gloss'] in labels]
    if not val_idx:raise ValueError('No validation examples for selected classes')
    vds=Windows(val,cfg);tds=Windows(train,cfg)
    dim=cfg.d*3*(int(cfg.state)+int(cfg.paths))
    head=nn.Linear(dim,len(classes)).to(cfg.device)
    if linear:encoder.requires_grad_(False);encoder.eval()
    params=list(head.parameters())+([] if linear else list(encoder.parameters()))
    opt=torch.optim.AdamW(params,lr=cfg.lr,weight_decay=cfg.wd,betas=(.9,.95))
    selection=dict(indices=idx.tolist(),ids=train.ids[idx].tolist(),glosses=classes.tolist(),fraction=fraction,k=k,
                   seed=cfg.seed,train_sha256=fingerprint(train.path),val_sha256=fingerprint(val.path))
    atomic_json(out/'selection.json',selection)
    def encode(items,training):
        v=to_device(collate(items)['near'],cfg.device)
        if not training:
            v['t_aff']=torch.eye(2,device=cfg.device).expand(len(items),-1,-1)
        pack=assemble(v,cfg,stats,teacher=True)
        with torch.set_grad_enabled(training and not linear):
            h,_,_=encoder(pack)
            z=readouts(h.float(),pack,cfg,v)['grp+P']
        return z
    @torch.no_grad()
    def validate():
        encoder.eval();all_logits=[];ys=[]
        for s in range(0,len(val_idx),cfg.batch):
            ii=val_idx[s:s+cfg.batch]
            items=[]
            # Fixed positions and no augmentation during validation.
            for i in ii:
                from pp.data import at_positions,resample
                from pp.base import window as W
                x,c,fps=val.clip(i);x,c,_=W.canonicalise(x,c)
                pos=at_positions(len(x),np.arange(64)/25*fps)
                xx,cc=resample(x,c,*pos);vf=pos[-1];ob=np.isfinite(xx).all(-1)
                v=dict(t_xy=np.nan_to_num(xx),t_obs=ob,t_conf=cc,valid_f=vf,valid_b=vf.reshape(32,2).any(-1),
                       target=np.zeros((7,32),bool),output_hidden=np.zeros_like(ob),t_aff=np.eye(2,dtype=np.float32))
                # collate expects training fields, but validation assembles teacher directly below.
                items.append(v)
            v={key:torch.from_numpy(np.stack([it[key] for it in items])).to(cfg.device) for key in items[0]}
            pack=assemble(v,cfg,stats,teacher=True);h,_,_=encoder(pack)
            z=readouts(h.float(),pack,cfg,v)['grp+P'];all_logits.append(head(z).cpu().numpy())
            ys.extend(labels[val.rows[i]['gloss']] for i in ii)
        logits=np.concatenate(all_logits);truth=np.array(ys)
        predictions=dict(pred=logits.argmax(-1),truth=truth,ids=val.ids[val_idx],
                         signer=np.array([val.rows[i]['signer_id'] for i in val_idx]),
                         gloss=np.array([val.rows[i]['gloss'] for i in val_idx]))
        return scores(logits,truth),predictions
    rng=np.random.default_rng(cfg.seed);best=-1.;epoch=0;step=0
    with (out/'train_log.jsonl').open('w') as log:
        while step<steps:
            tds.epoch=epoch;order=rng.permutation(idx)
            for start in range(0,len(order),cfg.batch):
                ii=order[start:start+cfg.batch]
                encoder.train(not linear);head.train();opt.zero_grad(set_to_none=True)
                z=encode([tds[int(i)] for i in ii],True)
                y=torch.tensor([labels[train.rows[int(i)]['gloss']] for i in ii],device=cfg.device)
                loss=F.cross_entropy(head(z),y);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.)
                for g in opt.param_groups:g['lr']=lr_at(step+1,steps,cfg.lr)
                opt.step();step+=1
                log.write(json.dumps(dict(step=step,loss=float(loss.detach())))+'\n')
                if step%max(1,steps//10)==0 or step==steps:
                    metric,predictions=validate();log.write(json.dumps(dict(step=step,val=metric))+'\n');log.flush()
                    if metric['top1']>best:
                        best=metric['top1'];torch.save(dict(supervised=True,config=cfg.to_dict(),encoder=encoder.state_dict(),
                            head=head.state_dict(),classes=classes.tolist(),stats=None if stats is None else stats.state,
                            metrics=metric,step=step,linear=linear),out/'best.pt')
                        np.savez(out/'best_val_predictions.npz',**predictions)
                if step>=steps:break
            epoch+=1
    result=dict(best_val_top1=best,steps=step,config=cfg.to_dict(),linear=linear,supervised=True,selection=selection,
                checkpoint=None if checkpoint is None else fingerprint(checkpoint))
    atomic_json(out/'results.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--ckpt');ap.add_argument('--train',required=True)
    ap.add_argument('--val',required=True);ap.add_argument('--out',required=True);ap.add_argument('--linear',action='store_true')
    ap.add_argument('--steps',type=int,default=30000);ap.add_argument('--fraction',type=float,default=1.)
    ap.add_argument('--k',type=int,default=0);ap.add_argument('--glosses',type=int,default=0);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(4)
    if a.ckpt:cfg=Config(**torch.load(a.ckpt,map_location='cpu',weights_only=False)['config'])
    elif a.config:cfg=Config.load(a.config)
    else:raise ValueError('Need --config (G0/scratch) or --ckpt (transfer)')
    if not 0<a.fraction<=1 or a.k<0 or a.steps<1:raise ValueError('Invalid supervised budget')
    cfg.device=a.device;cfg.seed=a.seed
    print(json.dumps(run(cfg,PoseSource(a.train,'train'),PoseSource(a.val,'val'),a.out,a.steps,a.ckpt,a.linear,a.fraction,a.k,a.glosses)))
if __name__=='__main__':main()
