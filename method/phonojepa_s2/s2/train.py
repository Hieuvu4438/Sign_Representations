"""S2 scratch pretraining: masked contextual + multiscale interval targets."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from s2 import features as Fe
from s2.auxiliary import Auxiliary,geometry,loss_weights
from s2.data.cache import ClipCache,read_corpus
from s2.data.dataset import Cfg,PhonoWindows,collate
from s2.joints import NC,GROUP,TORSO
from s2.losses import CenteredCE
from s2.model import PhonoJEPA,momentum
from s2.intervals import descriptors,selectors
from s2.interval_heads import IntervalHeads
from s2.paths import ROOT,REPO,RUNTIME,ASLC002_CORPUS

DEFAULT_CACHE=REPO/'method/phonojepa/runtime/cache/aslc002_train.npz'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--input',choices=['real','capacity','shuffle'],default='real')
    for name,default in [('steps',4000),('batch',32),('seed',0),('workers',6),('d',256),('depth',6),('heads',4),('dp',128),('pdepth',4),('log-every',50),('save-every',500),('stats-clips',640)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('lr',3e-4),('wd',.05),('lambda-body',0.),('lambda-relations',0.),
                         ('lambda-ctx',1.),('lambda-motion',.1),('lambda-signature',0.)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--signature',choices=['plain','leadlag'],default='plain')
    p.add_argument('--motion-lags',choices=['1','1,4,8'],default='1,4,8')
    p.add_argument('--motion-normalization',choices=['physical','frame'],default='physical')
    p.add_argument('--interval-target',choices=['motion','latent','mae'],default='motion')
    p.add_argument('--descriptor-input',choices=['none','motion','signature','both'],default='none')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Mark non-protocol dimensions/budget as smoke; still use all 3000 clips')
    pre,_=p.parse_known_args(argv)
    if pre.config:
        conf=json.loads(pre.config.read_text())
        unknown=set(conf)-{x.dest for x in p._actions}
        if unknown: p.error(f'Unknown config keys: {unknown}')
        p.set_defaults(**conf)
    a=p.parse_args(argv)
    for action in p._actions:
        if action.choices is not None and getattr(a,action.dest) not in action.choices:
            p.error(f'Invalid {action.dest}: {getattr(a,action.dest)}')
    if min(a.steps,a.batch,a.stats_clips,a.save_every,a.log_every)<=0:p.error('steps/batch/stats/save/log must be positive')
    if min(a.d,a.depth,a.heads,a.dp,a.pdepth)<=0:p.error('model dimensions/depth/heads must be positive')
    if a.batch<2 or a.workers<0:p.error('batch >= 2, workers >= 0 required')
    weights=[a.lambda_body,a.lambda_relations,a.lambda_ctx,a.lambda_motion,a.lambda_signature]
    if not all(math.isfinite(x) and x>=0 for x in weights) or not any(weights):p.error('loss weights must be finite, nonnegative, and at least one positive')
    if not all(math.isfinite(x) and x>0 for x in [a.lr,a.wd]):p.error('lr/wd must be finite positive')
    if a.interval_target=='latent' and a.lambda_motion>0 and a.lambda_ctx==0:
        p.error('Latent interval control requires contextual anti-collapse loss')
    if a.input!='real' and (a.lambda_motion or a.lambda_signature or a.descriptor_input!='none'):
        p.error('S1 body controls cannot also use S2 interval evidence; disable interval weights/inputs')
    if a.input!='real' and (a.lambda_body or a.lambda_relations):p.error('controls require both auxiliary weights = 0')
    if not a.smoke and (a.batch,a.d,a.depth,a.heads,a.dp,a.pdepth,a.stats_clips)!=(32,256,6,4,128,4,640):
        p.error('Protocol training requires batch32, encoder256x6/4heads, predictor128x4, stats640; use --smoke for diagnostics')
    if a.d%a.heads or a.dp%a.heads or (a.d//a.heads)%2 or (a.dp//a.heads)%2:p.error('RoPE requires even head dimensions')
    return a


def verify_corpus(cache,path):
    expected=[r['sample_id'] for r in read_corpus()]
    if len(cache)!=3000 or cache.ids.tolist()!=expected:raise ValueError('Cache IDs/order differ from the agreed 3000-video corpus')
    return {'n_clips':len(cache),'corpus_sha256':sha256(ASLC002_CORPUS),'cache_sha256':sha256(path)}


class Windows(PhonoWindows):
    """Index = (optimizer step, draw, clip). Reproducible across workers/resume."""
    def __getitem__(self,index):
        step,draw,i=index
        return self.item(i,np.random.default_rng([self.seed,step,draw]))


class Batches:
    def __init__(self,n,batch,seed,start,steps):self.n,self.batch,self.seed,self.start,self.steps=n,batch,seed,start,steps
    def __iter__(self):
        if self.batch>self.n: raise ValueError('Batch exceeds training corpus')
        per_epoch=self.n//self.batch
        previous=None
        for step in range(self.start,self.steps):
            epoch=step//per_epoch
            if epoch!=previous:
                order=np.random.default_rng([self.seed,epoch,98765]).permutation(self.n)
                previous=epoch
            offset=(step%per_epoch)*self.batch
            ids=order[offset:offset+self.batch]
            yield [(step,j,int(i)) for j,i in enumerate(ids)]
    def __len__(self):return self.steps-self.start


def input_control(ins,pres,mode,permutation=None):
    ins=list(ins);pres=pres.clone()
    if mode=='capacity':
        ins[TORSO]=torch.zeros_like(ins[TORSO]);pres[:,TORSO]=True
    elif mode=='shuffle':
        if permutation is None:raise ValueError('Explicit donor permutation required')
        ins[TORSO]=ins[TORSO][permutation];pres[:,TORSO]=pres[:,TORSO][permutation]
    return ins,pres


def lr_at(step,total,peak):
    warm=max(1,int(total*.1))
    if step<=warm:return peak*step/warm
    p=(step-warm)/max(1,total-warm)
    return peak*(.01+.99*.5*(1+math.cos(math.pi*p)))


def grouped_mean(err,w):
    groups=torch.as_tensor(np.tile(GROUP,32),device=w.device)
    values=[];gammas=[]
    for g,gamma in enumerate([1.,.5,1.]):
        weight=w*(groups==g)
        if weight.sum()>0:
            values.append((err*weight).sum()/weight.sum()*gamma);gammas.append(gamma)
    return sum(values)/sum(gammas) if values else err.sum()*0


class Trainer:
    def __init__(self,a):
        self.a=a;self.dev=a.device
        self.model=PhonoJEPA(a.d,a.depth,a.heads,a.dp,a.pdepth,descriptor_input=a.descriptor_input).to(a.device)
        # Legacy per-token intrinsic heads are unused: interval heads replace them.
        for p in self.model.heads.int.parameters():p.requires_grad_(False)
        self.aux=Auxiliary(a.dp).to(a.device)
        self.intervals=IntervalHeads(a.dp,a.d).to(a.device)
        if a.lambda_body==0 and a.lambda_relations==0:
            for p in self.aux.parameters():p.requires_grad_(False)
        self.params=[p for m in (self.model,self.aux,self.intervals) for p in m.parameters() if p.requires_grad]
        dec=[];nodec=[]
        for m in (self.model,self.aux,self.intervals):
            for n,p in m.named_parameters():
                if p.requires_grad:
                    (nodec if p.ndim<=1 or n.split('.')[-1] in ('bias','ch','absent','cls','mask') else dec).append(p)
        self.opt=torch.optim.AdamW([{'params':dec,'weight_decay':a.wd},{'params':nodec,'weight_decay':0.}],lr=a.lr,betas=(.9,.95))
        self.cce=CenteredCE(a.d,device=a.device)
        self.centers=[]

    @torch.no_grad()
    def fit_stats(self,ds):
        indices=np.random.default_rng(123).choice(len(ds),min(self.a.stats_clips,len(ds)),replace=False)
        batches=[];interval_batches=[]
        for k in range(0,len(indices),32):
            items=[ds.item(int(i),np.random.default_rng([123,int(i)])) for i in indices[k:k+32]]
            v={k:x.to(self.dev) for k,x in collate(items)['near'].items() if torch.is_tensor(x)}
            batches.append(geometry(v['t_xy'],v['t_obs'],v['valid_f']))
            anchor=Fe.anchors(v['s_xy'],v['s_obs'])
            desc=self.describe(v,'t',anchor,aff=v['s_aff'])
            interval_batches.append({k:x.cpu() for k,x in desc.items()})
        self.aux.fit(batches)
        self.intervals.fit(interval_batches)
        for enc in (self.model.student,self.model.teacher):
            if enc.descriptor is not None:enc.descriptor.set_stats(self.intervals)

    def describe(self,v,side,anchor,aff=None):
        a=self.a
        return descriptors(v[side+'_xy'],v[side+'_obs'],v[side+'_conf'],v['valid_f'],v['times'],
                           anchor,aff if aff is not None else v[side+'_aff'],
                           v['forbidden'] if side=='s' else None,a.signature,
                           tuple(int(x) for x in a.motion_lags.split(',')),a.motion_normalization)

    def view_loss(self,v,step):
        a=self.a;m=self.model
        v={k:x.to(self.dev) for k,x in v.items() if torch.is_tensor(x)}
        cells=v['valid_b'][:,None].expand(-1,NC,-1)
        # Always-visible shoulder side information only. Both views share this frame.
        anchor=Fe.anchors(v['s_xy'],v['s_obs'])
        perm=torch.arange(v['t_xy'].shape[0],device=self.dev).roll(1)
        with torch.no_grad():
            ft=Fe.frame_features(v['t_xy'],v['t_obs'],v['t_conf'],torch.zeros_like(v['s_jh']),v['t_aff'],anchor)
            it,pt=Fe.tokens(ft,v['valid_f']);it,pt=input_control(it,pt,a.input,perm)
            td=self.describe(v,'t',anchor) if a.descriptor_input!='none' else None
            ht,_,layers=m.teacher(it,pt,cells,return_layers=4,descriptors=td)
            target=F.layer_norm(torch.stack([F.layer_norm(l.float(),(a.d,)) for l in layers]).mean(0),(a.d,))
            reliability=Fe.reliability(v['t_conf'],v['t_obs'],v['valid_f'])
            if a.input=='capacity':reliability[:,TORSO]=1.
            elif a.input=='shuffle':reliability[:,TORSO]=reliability[:,TORSO][perm]
            w=(v['target']&cells&pt).float()*reliability
            w=w.transpose(1,2).flatten(1)
            # Auxiliary targets use student affine, not independently augmented teacher affine.
            ys=geometry(v['t_xy'],v['t_obs'],v['valid_f'],anchor,v['s_aff'])
            weights=loss_weights(v,ys)
            yd=self.describe(v,'t',anchor,aff=v['s_aff'])
            select=selectors(v)
            sd=self.describe(v,'s',anchor) if a.descriptor_input!='none' else None
        fs=Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'],v['s_aff'],anchor)
        ins,ps=Fe.tokens(fs,v['valid_f']);ins,ps=input_control(ins,ps,a.input,perm)
        hs,cs,_=m.student(ins,ps,v['visible'],descriptors=sd)
        q=m.predictor(hs,cs,v['visible'],v['target'])
        tau=.04+.02*min(1.,step/max(1,a.steps*.1))
        _,ce=self.cce(m.heads.ctx(q),target,w,tau,update=False)
        ctx=grouped_mean(ce,w)
        aux=self.aux.loss(q,ys,weights)
        pred=self.intervals(q,select,yd['duration'])
        interval=self.intervals.loss(pred,yd,select,target)
        loss=a.lambda_ctx*ctx+a.lambda_body*aux['body']+a.lambda_relations*aux['relations']
        loss=loss+a.lambda_motion*interval[a.interval_target]+a.lambda_signature*(interval['signature']+.25*interval['level'])
        self.centers.append((target.detach(),w.detach()))
        with torch.no_grad():
            entropy=CenteredCE.entropy(self.cce.target(target,tau))
            target_entropy=float((entropy*w).sum()/w.sum().clamp(min=1))
            active=target[w>0]
            target_std=float(active.std(0,unbiased=False).mean()) if len(active) else 0.0
        terms={'ctx':float(ctx.detach()),'teacher_entropy':target_entropy,'target_std':target_std,**{k:float(x.detach()) for k,x in aux.items()},'target_weight':float(w.sum()),
               **{'interval_'+k:float(x.detach()) if torch.is_tensor(x) else x for k,x in interval.items()},
               'body_queries':int((weights['body'].sum(-1)>0).sum()),'relation_queries':int((weights['relations'].sum(-1)>0).sum()),
               'visible_frac':float(v['visible'].float().mean()),'hidden_frac':float(v['hidden'].float().mean()),
               'valid_visible_frac':float(v['visible'].sum()/cells.sum().clamp(min=1)),
               'valid_hidden_frac':float(v['hidden'].sum()/cells.sum().clamp(min=1)),
               'context_drop_frac':float(v['drop'].sum()/cells.sum().clamp(min=1))}
        return loss,terms

    def step(self,b,step):
        self.centers=[]
        terms={};loss=0
        with torch.autocast(device_type='cuda' if self.dev.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=self.dev.startswith('cuda')):
            for view in ['near','far']:
                l,t=self.view_loss(b[view],step);loss=loss+l
                terms.update({view+'_'+k:v for k,v in t.items()})
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss; stopped without advancing step')
        loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(self.params,1.)
        if not torch.isfinite(gn):raise FloatingPointError('nonfinite gradient; stopped without advancing step')
        for g in self.opt.param_groups:g['lr']=lr_at(step+1,self.a.steps,self.a.lr)
        self.opt.step()
        with torch.no_grad():
            num=sum((t*(w>0)[...,None]).sum((0,1)) for t,w in self.centers)
            den=sum((w>0).sum() for _,w in self.centers)
            if den>0:self.cce.c.mul_(.9).add_(num/den,alpha=.1)
        self.model.ema_update(momentum(step+1,self.a.steps))
        return {'loss':float(loss.detach()),'grad_norm':float(gn),'lr':self.opt.param_groups[0]['lr'],**terms}

    def save(self,path,step,provenance):
        state={'student':self.model.student.state_dict(),'teacher':self.model.teacher.state_dict(),
            'model':self.model.state_dict(),'auxiliary':self.aux.state_dict(),'intervals':self.intervals.state_dict(),'opt':self.opt.state_dict(),'center':self.cce.c,
            'step':step,'config':serialize(vars(self.a)),'provenance':provenance,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if self.dev.startswith('cuda') else [],
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

    def load(self,path,provenance):
        ck=torch.load(path,map_location=self.dev,weights_only=False)
        ignored={'out','resume','workers','config','log_every','save_every','device'}
        current=serialize(vars(self.a))
        for k,v in ck['config'].items():
            if k not in ignored and current.get(k)!=v:raise ValueError(f'Resume config mismatch: {k}')
        if ck['provenance']!=provenance:raise ValueError('Resume corpus/code provenance mismatch')
        self.model.load_state_dict(ck['model']);self.aux.load_state_dict(ck['auxiliary']);self.opt.load_state_dict(ck['opt'])
        self.intervals.load_state_dict(ck['intervals'])
        self.cce.c=ck['center'].to(self.dev);torch.set_rng_state(ck['torch_rng'].cpu())
        if ck['cuda_rng'] and self.dev.startswith('cuda'):torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
        np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        return ck['step']


def serialize(d):return json.loads(json.dumps(d,default=str))


def provenance(cache,path):
    p=verify_corpus(cache,path)
    p['source_sha256']={str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'s2').rglob('*.py'))}
    p['torch']=torch.__version__;p['numpy']=np.__version__
    return p


def main(argv=None):
    a=args(argv)
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    cache=ClipCache(a.cache);proof=provenance(cache,a.cache)
    ds=Windows(cache,Cfg(),a.seed);tr=Trainer(a)
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:raise FileExistsError('Output is not empty; choose a new run or --resume')
    if a.resume and not (a.out/'ckpt.pt').exists():raise FileNotFoundError('No resume checkpoint')
    a.out.mkdir(parents=True,exist_ok=True)
    start=tr.load(a.out/'ckpt.pt',proof) if a.resume else 0
    if not a.resume:tr.fit_stats(ds)
    manifest={'config':serialize(vars(a)),'provenance':proof,'steps_done':start,'steps_planned':a.steps,
              'protocol_run':not a.smoke,'pretrained':False,'anchor_policy':'always_visible_shoulders; excluded_from_aux_targets',
              'data_rng':'seed/optimizer_step/draw; worker-independent','status':'running'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    loader=torch.utils.data.DataLoader(ds,batch_sampler=Batches(len(ds),a.batch,a.seed,start,a.steps),num_workers=a.workers,
            collate_fn=collate,pin_memory=a.device.startswith('cuda'),
            multiprocessing_context='spawn' if a.workers else None)
    t0=time.monotonic()
    with open(a.out/'train_log.jsonl','a') as log:
        for step,b in enumerate(loader,start):
            result=tr.step(b,step);done=step+1
            if done==1 or done%a.log_every==0 or done==a.steps:
                row={'step':done,'seconds':time.monotonic()-t0,**result}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if done%a.save_every==0 or done==a.steps:tr.save(a.out/'ckpt.pt',done,proof)
    tr.save(a.out/'final.pt',a.steps,proof)
    manifest.update(steps_done=a.steps,status='complete',seconds=time.monotonic()-t0,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else 0)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',a.steps,flush=True)

if __name__=='__main__':main()
