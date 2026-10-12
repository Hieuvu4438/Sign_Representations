"""Equal-capacity masked interval heads and train-only robust target statistics."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from s2.intervals import P,PATHS,layout


def balanced(err,w):
    """Equal dimensions within interval, intervals within path, paths within group.
    Confidence weights intervals; groups do not grow with their number of paths.
    """
    p,_,_,_=layout(w.device)
    per=(err*w).sum(-1)/w.sum(-1).clamp(min=1e-8)
    quality=w.amax(-1)
    groups=[]
    for group in range(4):
        paths=[]
        for i,spec in enumerate(PATHS):
            if spec.group!=group:continue
            q=quality[:,p==i];e=per[:,p==i]
            if q.sum()>0:paths.append((e*q).sum()/q.sum())
        if paths:groups.append(torch.stack(paths).mean())
    return torch.stack(groups).mean() if groups else err.sum()*0


class IntervalHeads(nn.Module):
    def __init__(self,dp,d):
        super().__init__()
        self.path=nn.Embedding(P,dp);self.scale=nn.Embedding(3,dp)
        self.meta=nn.Linear(3,dp)
        self.trunk=nn.Sequential(nn.LayerNorm(dp),nn.Linear(dp,dp),nn.GELU())
        self.heads=nn.ModuleDict({k:nn.Linear(dp,n) for k,n in
            [('motion',12),('signature',10),('level',4),('mae',62),('latent',d)]})
        for k,n in [('motion',12),('signature',10),('level',4),('mae',62)]:
            self.register_buffer(k+'_center',torch.zeros(P,n))
            self.register_buffer(k+'_scale',torch.ones(P,n))
            self.register_buffer(k+'_count',torch.zeros(P,n,dtype=torch.long))

    def forward(self,q,select,duration):
        count=select.sum(-1,keepdim=True)
        pooled=torch.bmm(select.to(q.dtype),q)/count.clamp(min=1).to(q.dtype)
        p,start,n,scale=layout(q.device)
        meta=torch.stack([start[None].expand(len(q),-1)/64,
                          n[None].expand(len(q),-1)/64,duration[...,0].log()],-1)
        h=self.trunk(pooled+self.path(p)+self.scale(scale)+self.meta(meta.to(q.dtype)))
        return {k:head(h) for k,head in self.heads.items()}

    def normalized(self,key,y):
        p,_,_,_=layout(y.device)
        return ((y-getattr(self,key+'_center')[p])/getattr(self,key+'_scale')[p]).clamp(-8,8)

    def loss(self,pred,target,select,teacher):
        active=(select.sum(-1)>0)&target['valid']
        quality=target['quality']*active
        out={}
        for key in ('motion','signature','level','mae'):
            dimvalid=target.get(key+'_valid',target['valid'][...,None])
            weight=quality[...,None]*dimvalid
            y=self.normalized(key,target[key])
            # MAMP-style dense motion reconstruction MSE; others robust Huber.
            err=(pred[key].float()-y).square() if key=='mae' else F.smooth_l1_loss(pred[key].float(),y,reduction='none',beta=.5)
            out[key]=balanced(err,weight)
        latent=torch.bmm(select,teacher.float())/select.sum(-1,keepdim=True).clamp(min=1)
        latent=F.layer_norm(latent,(latent.shape[-1],))
        out['latent']=balanced((F.layer_norm(pred['latent'].float(),(latent.shape[-1],))-latent).square(),
                               quality[...,None].expand_as(latent))
        out['queries']=int(active.sum())
        return out

    @torch.no_grad()
    def fit(self,batches):
        """Median/IQR per path/dimension; empty dimensions stay center0 scale1.
        Fit only clean windows sampled from the verified training corpus.
        """
        p,_,_,_=layout()
        for key in ('motion','signature','level','mae'):
            vals=torch.cat([torch.where(b.get(key+'_valid',b['valid'][...,None]),
                                        b[key],torch.nan).cpu() for b in batches])
            for i in range(P):
                x=vals[:,p==i].flatten(0,1)
                count=torch.isfinite(x).sum(0)
                quant=torch.nanquantile(x,torch.tensor([.25,.5,.75]),dim=0)
                center=torch.nan_to_num(quant[1])
                floor=.01 if key=='signature' else .05
                scale=torch.nan_to_num((quant[2]-quant[0])/1.349,nan=1.).clamp(min=floor)
                getattr(self,key+'_center')[i].copy_(center)
                getattr(self,key+'_scale')[i].copy_(scale)
                getattr(self,key+'_count')[i].copy_(count)


class DescriptorTokenizer(nn.Module):
    """Optional input factor; fixed slots and padded feature width in all arms."""
    def __init__(self,d,mode):
        super().__init__()
        self.mode=mode
        self.net=nn.Sequential(nn.Linear(27,256),nn.GELU(),nn.Linear(256,d),nn.LayerNorm(d))
        self.path=nn.Embedding(P,d);self.scale=nn.Embedding(3,d)
        self.meta=nn.Linear(3,d)
        self.register_buffer('center',torch.zeros(P,27))
        self.register_buffer('spread',torch.ones(P,27))

    @torch.no_grad()
    def set_stats(self,aux):
        self.center[:,:26].copy_(torch.cat([aux.motion_center,aux.signature_center,aux.level_center],-1))
        self.spread[:,:26].copy_(torch.cat([aux.motion_scale,aux.signature_scale,aux.level_scale],-1))

    def forward(self,desc):
        p,start,n,k=layout(self.center.device)
        x=((desc['input']-self.center[p])/self.spread[p]).clamp(-8,8)
        # Padded/inactive dimensions must stay zero after centering too.
        mask=torch.cat([desc['motion_valid'],desc['signature_valid'],
                        desc['valid'][...,None].expand(-1,-1,5)],-1)
        if self.mode=='motion':mask[...,12:22]=False
        if self.mode=='signature':mask[...,:12]=False
        x=torch.where(mask,x,0.)
        meta=torch.stack([start[None].expand(len(x),-1)/64,n[None].expand(len(x),-1)/64,
                          desc['duration'][...,0].log()],-1)
        h=self.net(x)+self.path(p)+self.scale(k)+self.meta(meta)
        valid=desc['valid']
        return torch.where(valid[...,None],h,0.),valid,(start+(n-1)/2)/2
