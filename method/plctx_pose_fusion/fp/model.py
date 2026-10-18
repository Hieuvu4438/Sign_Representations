"""Frozen PLctx + trainable body/relations graph, bounded token residual fusion."""
from __future__ import annotations
import copy,math
import torch
from torch import nn
import torch.nn.functional as F
from fp.pl.model import Encoder as PLEncoder
from fp.pl import features as PLFe
from fp.model_base import Predictor,Heads,momentum
from fp.graph import BodyGraph
from fp import features as Fe


class FusionEncoder(nn.Module):
    def __init__(self,d=256,pl_depth=6,graph_depth=8,gate_max=.3):
        super().__init__();self.d=d;self.gate_max=gate_max
        self.pl=PLEncoder(d,pl_depth,4)
        for p in self.pl.parameters():p.requires_grad_(False)
        self.graph=BodyGraph(d,graph_depth,4 if d<256 else 8)
        self.query=nn.Linear(d,d);self.key=nn.Linear(d,d);self.value=nn.Linear(d,d)
        self.out=nn.Linear(d,d);self.rnorm=nn.LayerNorm(d)
        self.gate=nn.Parameter(torch.full((3,),-4.))
        self.register_buffer('gate_ramp',torch.tensor(0.))

    def forward(self,v,side='s',gate_override=None):
        xy,obs,conf=v[side+'_xy'],v[side+'_obs'],v[side+'_conf']
        cells=v['visible'] if side=='s' else v['valid_b'][:,None].expand(-1,8,-1)
        anchor=Fe.anchors(v['s_xy'],v['s_obs'])
        aff=v[side+'_aff'];valid=v['valid_f']
        forbidden=v['forbidden'] if side=='s' else torch.zeros_like(obs)
        with torch.no_grad():
            # Reviewed safe anchor fallback, same body frame for both branches.
            frames=PLFe.frame_features(xy,obs,conf,v['s_jh'] if side=='s' else torch.zeros_like(obs),aff,anchor)
            ins,pres=PLFe.tokens(frames,valid)
            self.pl.eval();base,cls,_=self.pl(ins,pres,cells[:,:7])
        context,key,observed=self.graph(xy,obs,conf,valid,forbidden,cells,anchor,aff)
        B=xy.shape[0]
        weights=(self.query(base)@self.key(context).transpose(1,2))/math.sqrt(self.d)
        weights=weights.masked_fill(~key[:,None],-torch.inf).softmax(-1)
        residual=self.rnorm(self.out(weights@self.value(context)))
        gates=self.gate_max*self.gate.sigmoid()*self.gate_ramp
        channel_group=torch.tensor([0,0,1,1,2,2,2],device=xy.device).repeat(32)
        strength=gates[channel_group][None,:,None]
        if gate_override is not None:strength=float(gate_override)
        fused=base+strength*residual
        body=context[:,1:].reshape(B,32,4,self.d)
        ob=observed.reshape(B,32,4)
        body=(body*ob[...,None]).sum(2)/ob.sum(2,keepdim=True).clamp(min=1)
        h=torch.cat([fused.reshape(B,32,7,self.d),body[:,:,None]],2).flatten(1,2)
        # Predictor's CLS uses frozen PL information plus bounded graph summary.
        graphmean=(context[:,1:]*observed[...,None]).sum(1)/observed.sum(1,keepdim=True).clamp(min=1)
        fusedcls=cls+gates.mean()*graphmean
        visible=cells.clone();present=torch.cat([pres,ob.any(-1)[:,None]],1)
        visible[:,7] &= ob.any(-1)
        return {'h':h,'cls':fusedcls,'base':base,'fused':fused,'present':present,
                'visible':visible,'graph_present':observed,'graph_tokens':context[:,1:].reshape(B,32,4,self.d),'gates':gates}


class FusionJEPA(nn.Module):
    def __init__(self,d=256,pl_depth=6,graph_depth=8,dp=128,pdepth=4,gate_max=.3):
        super().__init__();self.student=FusionEncoder(d,pl_depth,graph_depth,gate_max)
        self.teacher=copy.deepcopy(self.student)
        self.predictor=Predictor(d,dp,pdepth,4);self.heads=Heads(d,dp)
        for p in self.teacher.parameters():p.requires_grad_(False)
        for p in self.heads.int.parameters():p.requires_grad_(False)

    def initialize(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.student.pl.load_state_dict(ck['teacher'],strict=True)
        # Predictor/ctx heads reuse the original training state where shapes match.
        if 'predictor' in ck:
            state=self.predictor.state_dict()
            for k,v in ck['predictor'].items():
                if k=='ch':state[k][:7]=v
                elif k in state and state[k].shape==v.shape:state[k]=v
            self.predictor.load_state_dict(state)
        if 'heads' in ck:
            self.heads.load_state_dict({k:v for k,v in ck['heads'].items() if k.startswith('ctx.')},strict=False)
        self.teacher.load_state_dict(self.student.state_dict())

    @torch.no_grad()
    def ema_update(self,m):
        for ps,pt in zip(self.student.parameters(),self.teacher.parameters()):
            if ps.requires_grad:pt.mul_(m).add_(ps,alpha=1-m)
        for bs,bt in zip(self.student.buffers(),self.teacher.buffers()):bt.copy_(bs)
