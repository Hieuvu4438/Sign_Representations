"""Train only graph/fusion/predictor; PL reference and EMA teacher stay frozen."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from fp.model import FusionJEPA,momentum
from fp.auxiliary import Auxiliary,geometry,loss_weights
from fp.losses import CenteredCE
from fp import features as Fe
from fp.data.dataset import collate
from fp.utils import rng_state,restore_rng


class Trainer:
    def __init__(self,c,initialize=True):
        self.c=c;self.device=c['device']
        self.model=FusionJEPA(c['d'],c['pl_depth'],c['graph_depth'],c['dp'],c['pdepth'],c['gate_max'])
        if initialize:self.model.initialize(c['pl_checkpoint'])
        self.model.to(self.device);self.aux=Auxiliary(c['dp']).to(self.device)
        self.params=[p for m in (self.model,self.aux) for p in m.parameters() if p.requires_grad]
        decay=[p for p in self.params if p.ndim>1];other=[p for p in self.params if p.ndim<=1]
        self.opt=torch.optim.AdamW([{'params':decay,'weight_decay':c['wd']},{'params':other,'weight_decay':0.}],lr=c['lr'],betas=(.9,.95))
        self.center=CenteredCE(c['d'],device=self.device)

    @torch.no_grad()
    def fit_stats(self,ds):
        ids=np.random.default_rng(123).choice(len(ds),min(len(ds),self.c['stats_clips']),replace=False)
        batches=[]
        for k in range(0,len(ids),32):
            b=collate([ds.item(int(i),np.random.default_rng([123,int(i)])) for i in ids[k:k+32]])['near']
            v={k:x.to(self.device) for k,x in b.items() if torch.is_tensor(x)}
            batches.append(geometry(v['t_xy'],v['t_obs'],v['valid_f'],Fe.anchors(v['s_xy'],v['s_obs']),v['s_aff']))
        self.aux.fit(batches)

    def view_loss(self,v,step):
        c=self.c;m=self.model
        v={k:x.to(self.device) for k,x in v.items() if torch.is_tensor(x)}
        with torch.no_grad():
            t=m.teacher(v,'t')
            target=F.layer_norm(t['h'].float(),(c['d'],))
            reliability=Fe.reliability(v['t_conf'],v['t_obs'],v['valid_f'])
            weight=(v['target']&t['present']).float()*reliability
            weight=weight.transpose(1,2).flatten(1)
            anchor=Fe.anchors(v['s_xy'],v['s_obs'])
            ys=geometry(v['t_xy'],v['t_obs'],v['valid_f'],anchor,v['s_aff']);weights=loss_weights(v,ys)
        s=m.student(v,'s')
        q=m.predictor(s['h'],s['cls'],s['visible'],v['target'])
        tau=.04+.02*min(1.,step/max(1,c['warmup_steps']))
        _,ce=self.center(m.heads.ctx(q),target,weight,tau,update=False)
        # Equal weight of hands, face, torso, independent of token counts.
        groups=torch.tensor([0,0,0,0,1,1,1,2],device=self.device).repeat(32)
        ctx=[]
        for g in range(3):
            w=weight*(groups==g)
            if w.sum()>0:ctx.append((ce*w).sum()/w.sum())
        ctx=torch.stack(ctx).mean() if ctx else ce.sum()*0
        aux=self.aux.loss(q,ys,weights)
        use=(s['visible'][:,:7]&s['present'][:,:7]).transpose(1,2).flatten(1)
        err=(s['fused'].float()-s['base'].float()).square().mean(-1)
        retain=(err*use).sum()/use.sum().clamp(min=1)
        loss=ctx+c['lambda_body']*aux['body']+c['lambda_relations']*aux['relations']+c['lambda_retention']*retain
        self.centers.append((target,weight))
        return loss,{'ctx':float(ctx.detach()),'body':float(aux['body'].detach()),'relations':float(aux['relations'].detach()),
             'retention':float(retain.detach()),'target_weight':float(weight.sum()),
             'body_queries':int((weights['body'].sum(-1)>0).sum()),'relation_queries':int((weights['relations'].sum(-1)>0).sum()),
             'graph_visible':int(s['graph_present'].sum()),'gate_max_actual':float(s['gates'].max().detach()),
             'target_std':float(target[weight>0].std(0,unbiased=False).mean()) if (weight>0).any() else 0.}

    def update(self,b,step):
        c=self.c;self.centers=[]
        ramp=min(1.,max(0.,(step-c['warmup_steps'])/c['gate_ramp_steps']))
        self.model.student.gate_ramp.fill_(ramp);self.model.teacher.gate_ramp.fill_(ramp)
        terms={};loss=0.
        with torch.autocast(device_type='cuda' if self.device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            for view in ('near','far'):
                value,metrics=self.view_loss(b[view],step);loss=loss+value
                terms.update({view+'_'+k:v for k,v in metrics.items()})
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        warm=min(10000,max(1,c['steps']//10))
        lr=c['lr']*(step+1)/warm if step<warm else c['lr']*(.01+.99*.5*(1+math.cos(math.pi*(step+1-warm)/max(1,c['steps']-warm))))
        for group in self.opt.param_groups:group['lr']=lr
        self.opt.step()
        with torch.no_grad():
            count=sum((w>0).sum() for t,w in self.centers)
            if count>0:
                mean=sum((t*(w>0)[...,None]).sum((0,1)) for t,w in self.centers)/count
                self.center.c.mul_(.9).add_(mean,alpha=.1)
        self.model.ema_update(momentum(step+1,c['steps']))
        return {'loss':float(loss.detach()),'grad_norm':float(norm),'lr':lr,**terms}

    def state(self,step,proof,progress):
        return {'model':self.model.state_dict(),'aux':self.aux.state_dict(),'optimizer':self.opt.state_dict(),
                'center':self.center.c,'rng':rng_state(),'step':step,'config':self.c,'proof':proof,'progress':progress}

    def restore(self,ck,proof):
        if ck['proof']!=proof or ck['config']!=self.c:raise ValueError('Resume scientific config/source/data mismatch')
        self.model.load_state_dict(ck['model']);self.aux.load_state_dict(ck['aux']);self.opt.load_state_dict(ck['optimizer'])
        self.center.c=ck['center'].to(self.device);restore_rng(ck['rng'])
        return ck['step'],ck['progress']
