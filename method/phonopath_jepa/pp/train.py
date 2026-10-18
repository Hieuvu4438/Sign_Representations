"""Resumable sequential SSL runner. Gloss/signer are never consumed."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
from pp.base.losses import CenteredCE
from pp.config import Config
from pp.data import PoseSource,Windows,collate
from pp.features import assemble
from pp.model import PhonoPathJEPA,momentum
from pp.signature import RobustStats
from pp.fit_stats import fingerprint
from pp.closure import mask_report
from pp.recovery import atomic_json,save_checkpoint,recover_checkpoint,quarantine_corrupt_latest


def lr_at(step,total,peak):
    warm=max(1,int(total*.1))
    if step<=warm:return peak*step/warm
    p=(step-warm)/max(1,total-warm)
    return peak*(.01+.99*(1+math.cos(math.pi*p))/2)


def to_device(v,device):return {k:x.to(device,non_blocking=True) if torch.is_tensor(x) else x for k,x in v.items()}


class Trainer:
    def __init__(self,cfg,stats):
        self.cfg=cfg;self.stats=stats;self.model=PhonoPathJEPA(cfg).to(cfg.device)
        self.params=[p for p in self.model.parameters() if p.requires_grad]
        dec=[];nodec=[]
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                (nodec if p.ndim<=1 or name.split('.')[-1] in ('bias','ch','pchan','absent','cls','mask') else dec).append(p)
        self.opt=torch.optim.AdamW([dict(params=dec,weight_decay=cfg.wd),dict(params=nodec,weight_decay=0.)],lr=cfg.lr,betas=(.9,.95))
        self.centres={s:CenteredCE(cfg.d,device=cfg.device) for s in ('S','P')};self.buffers=[]

    def view_loss(self,view,step,total):
        cfg=self.cfg;v=to_device(view,cfg.device)
        with torch.no_grad():
            tp=assemble(v,cfg,self.stats,True);h,cls,layers=self.model.teacher(tp,return_layers=4)
            target=F.layer_norm(torch.stack([F.layer_norm(l.float(),(cfg.d,)) for l in layers]).mean(0),(cfg.d,))
            if cfg.aligned_warp:
                # B3: latent interpolation at the student's actual source-time bins.
                u=v['alignment'];lo=u.floor().long();hi=(lo+1).clamp_max(31);alpha=(u-lo)[...,None,None]
                grid=target.view(len(target),32,7,cfg.d);batch=torch.arange(len(target),device=target.device)[:,None]
                target=((1-alpha)*grid[batch,lo]+alpha*grid[batch,hi]).flatten(1,2)
        sp=assemble(v,cfg,self.stats)
        h,cls,_=self.model.student(sp)
        pred=self.model.predictor(h,cls,sp['visible'],sp['target'])
        # Teacher reliability and presence define loss; student observations never remove targets.
        target_mask=[]
        if cfg.state: target_mask.append(v['target'].transpose(1,2).flatten(1)&tp['s_present'].transpose(1,2).flatten(1))
        if cfg.paths:
            # Closure is computed from student's support; teacher availability gates it.
            target_mask.append(sp['p']['target'].flatten(1)&tp['p']['present'].flatten(1))
            if cfg.global_mode!='none':target_mask.append(torch.zeros(len(h),5,device=h.device,dtype=torch.bool))
        w=torch.cat(target_mask,1)*tp['weights']
        loss=pred.sum()*0;terms={};offset=0
        for stream,n,lam in [('S',224 if cfg.state else 0,1.),('P',110 if cfg.paths else 0,cfg.lambda_p)]:
            if not n:continue
            ps,ts,ws=pred[:,offset:offset+n],target[:,offset:offset+n],w[:,offset:offset+n]
            tau=.04+.02*min(1.,step/max(1,total*.1))
            if cfg.loss=='cosine':
                per=1-F.cosine_similarity(ps.float(),ts.detach()-self.centres[stream].c,dim=-1)
                self.buffers.append((stream,ts.detach(),ws.detach()))
            else:
                _,per=self.centres[stream](ps,ts,ws,tau,update=False)
                self.buffers.append((stream,ts.detach(),ws.detach()))
            if stream=='S':
                groups=torch.tensor([0,0,0,0,1,1,1]*32,device=ws.device)
                value=per.sum()*0;den=0.
                for group,gamma in ((0,1.),(1,.5)):
                    wg=ws*(groups==group)
                    if wg.sum()>0:value=value+gamma*(per*wg).sum()/wg.sum();den+=gamma
                value=value/max(den,1e-6)
            else:value=(per*ws).sum()/ws.sum().clamp_min(1e-6)
            loss=loss+lam*value;terms[stream]=float(value.detach());terms[stream+'_targets']=int((ws>0).sum())
            offset+=n
        if cfg.paths:terms['closure']=mask_report(sp['p'])
        return loss,terms

    def step(self,batch,step,total):
        self.opt.zero_grad(set_to_none=True);self.buffers=[]
        terms={};loss=0
        with torch.autocast(device_type=torch.device(self.cfg.device).type,dtype=torch.bfloat16,enabled=self.cfg.device.startswith('cuda')):
            for view in ('near','far'):
                l,t=self.view_loss(batch[view],step,total);loss=loss+l;terms[view]=t
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; checkpoint preserved')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(self.params,1.)
        if not torch.isfinite(gn):raise FloatingPointError('Non-finite gradients; checkpoint preserved')
        self.opt.step();self.model.ema_update(momentum(step+1,total))
        with torch.no_grad():
            for stream,cce in self.centres.items():
                values=[(t,w) for s,t,w in self.buffers if s==stream]
                den=sum(int((w>0).sum()) for t,w in values)
                if den:
                    num=sum((t*(w>0)[...,None]).sum((0,1)) for t,w in values)
                    cce.c.mul_(.9).add_(num/den,alpha=.1)
        return float(loss.detach()),float(gn),terms

    def save(self,path,step,epoch,cursor,source_hash,total,elapsed_seconds=0.):
        state=dict(config=self.cfg.to_dict(),model=self.model.state_dict(),opt=self.opt.state_dict(),step=step,epoch=epoch,cursor=cursor,
                   centres={k:v.c.cpu() for k,v in self.centres.items()},stats=None if self.stats is None else self.stats.state,
                   stats_provenance=None if self.stats is None else self.stats.provenance,source_hash=source_hash,total=total,
                   rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   np_rng=np.random.get_state(),python_rng=random.getstate(),elapsed_seconds=elapsed_seconds)
        save_checkpoint(path,state)

    def load(self,path,source_hash,total):
        ck=torch.load(path,map_location=self.cfg.device,weights_only=False)
        if ck['config']!=self.cfg.to_dict() or ck['source_hash']!=source_hash or ck['total']!=total:
            raise ValueError('Resume configuration, source, or schedule differs')
        self.model.load_state_dict(ck['model']);self.opt.load_state_dict(ck['opt'])
        for k,v in self.centres.items():v.c=ck['centres'][k].to(self.cfg.device)
        if self.stats is not None and any(not torch.equal(value,self.stats.state[k][field].to(value))
                                         for k,v in ck['stats'].items() for field,value in v.items()):
            raise ValueError('Normalization changed since checkpoint')
        torch.set_rng_state(ck['rng'].cpu());np.random.set_state(ck['np_rng']);random.setstate(ck['python_rng'])
        if ck['cuda_rng']:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        return ck['step'],ck['epoch'],ck['cursor']


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--max-steps',type=int,help='Smoke budget; not a scientific run')
    ap.add_argument('--checkpoint-steps',type=int,default=200)
    ap.add_argument('--checkpoint-seconds',type=float,default=60.)
    a=ap.parse_args(argv);cfg=Config.load(a.config).validate();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() and not a.resume:raise ValueError('Run exists; use a new directory or --resume')
    torch.set_num_threads(4);torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    src=PoseSource(cfg.cache);ds=Windows(src,cfg)
    if len(ds)<cfg.batch:raise ValueError('Training source smaller than batch; no optimizer steps possible')
    source_hash=fingerprint(src.path);per_epoch=len(ds)//cfg.batch;total=a.max_steps or per_epoch*cfg.epochs
    if a.checkpoint_steps<1 or a.checkpoint_seconds<=0:raise ValueError('Invalid checkpoint interval')
    stats=None;recovered=None;recovery_errors=[];loaded=None
    if a.resume:
        recovered,loaded,recovery_errors=recover_checkpoint(out,cfg.to_dict(),source_hash,total)
        quarantine_corrupt_latest(out,recovery_errors)
    if cfg.paths:
        if a.resume:
            stats=RobustStats(loaded['stats'],cfg.asinh);stats.provenance=loaded['stats_provenance']
        else:stats=RobustStats.load(cfg.stats,cfg.asinh)
    if stats and stats.provenance['source_sha256']!=fingerprint(src.path):raise ValueError('Stats/source mismatch')
    tr=Trainer(cfg,stats);step=epoch=cursor=0;elapsed_before=0.
    if a.resume:
        step,epoch,cursor=tr.load(recovered,source_hash,total);elapsed_before=loaded.get('elapsed_seconds',0.)
    del loaded
    # Signals request a save at the next successful optimizer boundary.
    requested={'stop':False,'checkpoint':False}
    def stop_handler(sig,frame):requested['stop']=True
    def checkpoint_handler(sig,frame):requested['checkpoint']=True
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    signal.signal(signal.SIGUSR1,checkpoint_handler);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    manifest=dict(config=cfg.to_dict(),source_hash=source_hash,n_clips=len(src),steps_planned=total,smoke=bool(a.max_steps),status='running',pid=os.getpid())
    atomic_json(out/'manifest.json',manifest);start=time.monotonic();last_save=start;last_status=0.;recent=[]
    checkpoint_step=step
    def commit(path=None):
        nonlocal last_save,checkpoint_step
        tr.save(path or out/'latest.pt',step,epoch,cursor,source_hash,total,elapsed_before+time.monotonic()-start)
        last_save=time.monotonic();checkpoint_step=step;requested['checkpoint']=False
    def progress(state='running',record=None):
        rate=len(recent)/sum(recent) if recent and sum(recent)>0 else None
        value=dict(status=state,pid=os.getpid(),updated_unix=time.time(),step=step,total_steps=total,epoch=epoch+cursor/per_epoch,
                   percent=100*step/total,checkpoint_step=checkpoint_step,steps_per_second=rate,
                   eta_seconds=(total-step)/rate if rate else None,elapsed_seconds=elapsed_before+time.monotonic()-start,
                   resumed_from=str(recovered) if recovered else None,recovery_errors=recovery_errors)
        if record:value.update(loss=record['loss'],grad_norm=record['grad_norm'])
        atomic_json(out/'progress.json',value)
    if not a.resume:
        commit(out/'initial.pt');commit()
    progress()
    try:
        with (out/'train_log.jsonl').open('a') as log:
            if a.resume:log.write(json.dumps(dict(event='resume',step=step,checkpoint=str(recovered),recovery_errors=recovery_errors))+'\n');log.flush()
            while step<total and not requested['stop']:
                ds.epoch=epoch
                order=np.random.default_rng([cfg.seed,epoch,90210]).permutation(len(ds))[:per_epoch*cfg.batch]
                indices=order[cursor*cfg.batch:].tolist()
                dl=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=cfg.batch,num_workers=cfg.workers,
                                              collate_fn=collate,pin_memory=cfg.device.startswith('cuda'))
                for b in dl:
                    if requested['stop']:break
                    tick=time.monotonic()
                    for g in tr.opt.param_groups:g['lr']=lr_at(step+1,total,cfg.lr)
                    loss,gn,terms=tr.step(b,step,total);step+=1;cursor+=1
                    recent.append(time.monotonic()-tick)
                    recent=recent[-100:]
                    record=dict(step=step,epoch=epoch+cursor/per_epoch,loss=loss,grad_norm=gn,seconds=time.monotonic()-start,terms=terms)
                    log.write(json.dumps(record)+'\n');log.flush()
                    if step==1 or step%50==0:print(json.dumps({k:v for k,v in record.items() if k!='terms'}),flush=True)
                    if step%a.checkpoint_steps==0 or step==total or time.monotonic()-last_save>=a.checkpoint_seconds or requested['checkpoint']:commit()
                    if step==1 or time.monotonic()-last_status>=5:
                        progress(record=record);last_status=time.monotonic()
                    if step>=total:break
                if cursor>=per_epoch:
                    epoch+=1;cursor=0
                    if epoch in cfg.save_epochs:commit();commit(out/f'epoch_{epoch:03d}.pt')
            if step==total:
                if cursor>=per_epoch:
                    epoch+=1;cursor=0
                    if epoch in cfg.save_epochs:commit(out/f'epoch_{epoch:03d}.pt')
                commit();commit(out/'final.pt');state='complete'
            else:commit();state='interrupted'
        manifest.update(status=state,steps_done=step,seconds=elapsed_before+time.monotonic()-start)
        progress(state)
    except BaseException as exc:
        try:commit()
        except Exception as save_error:manifest['checkpoint_error']=repr(save_error)
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),steps_done=step)
        progress(manifest['status'])
        raise
    finally:atomic_json(out/'manifest.json',manifest)
if __name__=='__main__':main()
