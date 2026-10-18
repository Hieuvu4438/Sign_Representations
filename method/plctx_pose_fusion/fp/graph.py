"""Frame-local body GCN and conservative raw-support-aware graph tokens."""
from __future__ import annotations
import torch
from torch import nn
from fp import features as Fe
from fp.joints import LSHO,RSHO,NOSE_B,LHIP,RHIP,RWRI,LWRI,RELB,LELB
from fp.pl.model import Block

PARTS=((NOSE_B,LSHO,RSHO,LHIP,RHIP),(RSHO,RELB,RWRI),(LSHO,LELB,LWRI))
EDGES=(((0,1),(0,2),(1,2),(1,3),(2,4),(3,4)),((0,1),(1,2)),((0,1),(1,2)))
RELATIONS=((RWRI,LHIP,RHIP),(LWRI,LHIP,RHIP),(RWRI,NOSE_B),(LWRI,NOSE_B),(RWRI,LWRI))


class SpatialBlock(nn.Module):
    def __init__(self,cin,cout,adj):
        super().__init__();self.register_buffer('adj',adj)
        self.self_linear=nn.Linear(cin,cout);self.neighbor=nn.Linear(cin,cout)
        self.residual=nn.Linear(cin,cout) if cin!=cout else nn.Identity()
        self.norm=nn.LayerNorm(cout)

    def forward(self,x,valid):
        x=torch.where(valid[...,None],x,0.)
        message=torch.einsum('vu,btuc->btvc',self.adj,self.neighbor(x)*valid[...,None])
        return torch.where(valid[...,None],torch.nn.functional.gelu(self.norm(
            self.self_linear(x)+message+self.residual(x))),0.)


class SpatialPart(nn.Module):
    def __init__(self,joints,edges,d):
        super().__init__();a=torch.zeros(joints,joints)
        for u,v in edges:a[u,v]=a[v,u]=1.
        a=a/a.sum(-1,keepdim=True).clamp(min=1)
        self.inp=nn.Linear(3,64)
        self.blocks=nn.ModuleList([SpatialBlock(i,o,a) for i,o in [(64,64),(64,128),(128,256)]])
        self.out=nn.Sequential(nn.Linear(joints*256,d),nn.LayerNorm(d),nn.GELU())

    def forward(self,x,valid):
        x=self.inp(torch.where(valid[...,None],x,0.))*valid[...,None]
        for block in self.blocks:x=block(x,valid)
        return self.out(x.flatten(-2))


class BodyGraph(nn.Module):
    def __init__(self,d=256,depth=8,heads=8):
        super().__init__();self.d=d
        self.parts=nn.ModuleList([SpatialPart(len(js),edges,d) for js,edges in zip(PARTS,EDGES)])
        self.relation=nn.Sequential(nn.Linear(30,d),nn.LayerNorm(d),nn.GELU())
        self.pair=nn.Linear(2*d+2,d);self.part_type=nn.Parameter(torch.randn(4,d)*.02)
        self.null=nn.Parameter(torch.randn(1,1,d)*.02)
        self.blocks=nn.ModuleList([Block(d,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(d)

    def forward(self,xy,obs,conf,valid,forbidden,cells,anchor,aff):
        p=Fe.to_body_frame(xy,*anchor,aff)
        observed=obs&valid[...,None]
        p=torch.where(observed[...,None],p,0.)
        B,T=valid.shape;tokens=[];oks=[]
        for i,(ids,part,ch) in enumerate(zip(PARTS,self.parts,(7,1,3))):
            o=observed[:,:,list(ids)]
            h=part(torch.cat([p[:,:,list(ids)],conf[:,:,list(ids),None]],-1),o)
            # Pair-local GCN; whole token forbidden if any supporting source hidden.
            allowed=~forbidden[:,:,list(ids)].any(-1)
            ok=(o.any(-1)&valid).reshape(B,32,2).any(-1)
            ok &= allowed.reshape(B,32,2).all(-1)&cells[:,ch]
            pair=torch.cat([(h*o.any(-1)[...,None]).reshape(B,32,2*self.d),
                            o.any(-1).reshape(B,32,2).float()],-1)
            tokens.append(self.pair(pair)+self.part_type[i]);oks.append(ok)
        rel=[];ro=[]
        for ids in RELATIONS:
            a=p[:,:,ids[0]];b=p[:,:,list(ids[1:])].mean(-2);delta=a-b
            good=observed[:,:,list(ids)].all(-1)
            rel.append(torch.where(good[...,None],torch.cat([delta,delta.norm(dim=-1,keepdim=True)],-1),0.))
            ro.append(good)
        r=torch.cat(rel,-1);o=torch.stack(ro,-1)
        rh=self.relation(r.reshape(B,32,30))+self.part_type[3]
        ids=sorted({j for js in RELATIONS for j in js})
        ok=o.reshape(B,32,2,5).any((-1,-2))
        ok &= ~forbidden[:,:,ids].reshape(B,32,2,-1).any((-1,-2))
        ok &= cells[:,1]&cells[:,3]&cells[:,7]
        tokens.append(rh);oks.append(ok)
        x=torch.stack(tokens,2).flatten(1,2)
        present=torch.stack(oks,2).flatten(1,2)
        x=torch.where(present[...,None],x,0.)
        x=torch.cat([self.null.expand(B,-1,-1),x],1)
        key=torch.cat([torch.ones(B,1,dtype=torch.bool,device=x.device),present],1)
        pos=torch.cat([x.new_zeros(1),torch.arange(32,device=x.device).repeat_interleave(4).float()])
        rot=torch.ones(129,dtype=torch.bool,device=x.device);rot[0]=False
        for block in self.blocks:x=block(x,pos,rot,key)
        return self.norm(x),key,present
