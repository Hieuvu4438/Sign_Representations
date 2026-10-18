"""Shared S/P/G transformer, continuous RoPE, structural bias, metadata-only predictor."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from pp.base.model import Tokenizer,Block,rope,momentum,Predictor as ReferencePredictor
from pp.paths import token_dims
from pp.ptokens import WINDOWS


def metadata(cfg,device):
    ch=[]; scale=[]; time=[]; stream=[]; width=[]
    if cfg.state:
        ch+=list(range(7))*32; scale+=[0]*224
        time+=[float(i) for i in range(32) for _ in range(7)]; stream+=[0]*224; width+=[2.]*224
    if cfg.paths:
        for c in range(5):
            for start,size,s in WINDOWS:
                ch.append(c+7); scale.append(s); time.append((start+(size-1)/2)/2); stream.append(1); width.append(float(size))
        if cfg.global_mode!='none':
            ch+=list(range(7,12)); scale+=[0]*5; time+=[15.75]*5; stream+=[2]*5; width+=[64.]*5
    return {k:torch.tensor(v,device=device,dtype=torch.float32 if k in ('time','width') else torch.long)
            for k,v in dict(ch=ch,scale=scale,time=time,stream=stream,width=width).items()}


class AttentionBlock(Block):
    def forward(self,x,pos,rot,key_ok,bias=None):
        if bias is None: return super().forward(x,pos,rot,key_ok)
        B,N,d=x.shape
        q,k,v=self.qkv(self.n1(x)).view(B,N,3,self.h,d//self.h).permute(2,0,3,1,4)
        r=rot[None,None,:,None]
        q,k=torch.where(r,rope(q,pos),q),torch.where(r,rope(k,pos),k)
        mask=bias[None,None].expand(B,1,N,N).to(q.dtype).masked_fill(~key_ok[:,None,None,:],float('-inf'))
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        x=x+self.proj(a.transpose(1,2).reshape(B,N,d))
        return x+self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.d=cfg.d
        self.tok=Tokenizer(cfg.d) if cfg.state else None
        if cfg.state_relations:
            self.tok.mlps[1]=nn.Sequential(nn.Linear(24,256),nn.GELU(),nn.Linear(256,cfg.d),nn.LayerNorm(cfg.d))
        dims=token_dims(cfg.time_channel,cfg.path_means)
        self.path_mlps=nn.ModuleDict({str(c):nn.Sequential(nn.Linear(dims[c],256),nn.GELU(),nn.Linear(256,cfg.d),nn.LayerNorm(cfg.d))
                                    for c in (0,1,4)}) if cfg.paths else None
        self.pchan=nn.Parameter(torch.randn(5,cfg.d)*.02) if cfg.paths else None
        self.scale=nn.Embedding(3,cfg.d); self.stream=nn.Embedding(3,cfg.d)
        nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.stream.weight)
        self.absent=nn.Parameter(torch.randn(cfg.d)*.02)
        self.cls=nn.Parameter(torch.randn(1,1,cfg.d)*.02)
        self.blocks=nn.ModuleList([AttentionBlock(cfg.d,cfg.heads) for _ in range(cfg.depth)])
        self.norm=nn.LayerNorm(cfg.d)
        self.bias_channel=nn.Parameter(torch.zeros(12,12)); self.bias_scale=nn.Parameter(torch.zeros(3,3))
        self.bias_time=nn.Parameter(torch.zeros(8))

    def structural(self,meta):
        if not self.cfg.structural_bias or self.cfg.arm=='A0': return None
        c,l,t=meta['ch'],meta['scale'],meta['time']
        dist=(t[:,None]-t[None,:]).abs()
        bucket=(dist+1).log2().long().clamp_max(7)
        b=self.bias_channel[c[:,None],c[None,:]]+self.bias_scale[l[:,None],l[None,:]]+self.bias_time[bucket]
        return F.pad(b,(1,0,1,0))

    def forward(self,pack,return_layers=0):
        meta=metadata(self.cfg,self.cls.device); parts=[]
        if self.cfg.state: parts.append(self.tok(pack['s_inputs'],pack['s_present']))
        if self.cfg.paths:
            p=pack['p']
            for c in range(5):
                typ=str(c if c<2 or c==4 else c-2)
                z=self.path_mlps[typ](p['inputs'][c])
                z=torch.where(p['present'][:,c,:,None],z,self.absent.to(z.dtype))
                parts.append(z+self.pchan[c])
            if self.cfg.global_mode!='none':
                for c in range(5):
                    typ=str(c if c<2 or c==4 else c-2)
                    parts.append((self.path_mlps[typ](pack['global'][c])+self.pchan[c])[:,None])
        x=torch.cat(parts,1)
        if self.cfg.arm!='A0':x=x+self.scale(meta['scale'])+self.stream(meta['stream'])
        vis=pack['visible']; x=torch.where(vis[...,None],x,torch.zeros_like(x))
        B=x.shape[0]; x=torch.cat([self.cls.expand(B,-1,-1).to(x.dtype),x],1)
        key=torch.cat([torch.ones(B,1,dtype=torch.bool,device=x.device),vis],1)
        pos=F.pad(meta['time'],(1,0)); rot=torch.ones_like(pos,dtype=torch.bool);rot[0]=False
        bias=self.structural(meta); layers=[]
        for i,block in enumerate(self.blocks):
            x=block(x,pos,rot,key,bias)
            if return_layers and i>=len(self.blocks)-return_layers: layers.append(x[:,1:])
        x=self.norm(x)
        return x[:,1:],x[:,0],layers


class Predictor(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        if cfg.arm=='A0':
            self.reference=ReferencePredictor(cfg.d,cfg.dp,cfg.pdepth,cfg.heads)
            self.head=nn.Linear(cfg.dp,cfg.d)
            return
        self.inp=nn.Linear(cfg.d,cfg.dp); self.ch=nn.Embedding(12,cfg.dp)
        self.scale=nn.Embedding(3,cfg.dp); self.stream=nn.Embedding(3,cfg.dp)
        self.width=nn.Linear(1,cfg.dp,bias=False); self.mask=nn.Parameter(torch.randn(cfg.dp)*.02)
        self.blocks=nn.ModuleList([Block(cfg.dp,cfg.heads) for _ in range(cfg.pdepth)])
        self.norm=nn.LayerNorm(cfg.dp); self.head=nn.Linear(cfg.dp,cfg.d)
    def forward(self,h,cls,visible,target):
        if self.cfg.arm=='A0':
            vis=visible.view(len(h),32,7).transpose(1,2)
            tgt=target.view(len(h),32,7).transpose(1,2)
            return self.head(self.reference(h,cls,vis,tgt))
        meta=metadata(self.cfg,h.device); B,N,_=h.shape
        q=self.mask+self.ch(meta['ch'])+self.scale(meta['scale'])+self.stream(meta['stream'])+self.width(meta['width'][:,None]/64)
        ctx=torch.where(visible[...,None],self.inp(h),torch.zeros(B,N,self.cfg.dp,device=h.device,dtype=h.dtype))
        x=torch.cat([self.inp(cls)[:,None],ctx,q[None].expand(B,-1,-1)],1)
        key=torch.cat([torch.ones(B,1,dtype=torch.bool,device=h.device),visible,target],1)
        pos=torch.cat([h.new_zeros(1),meta['time'],meta['time']]); rot=torch.ones_like(pos,dtype=torch.bool);rot[0]=False
        for block in self.blocks: x=block(x,pos,rot,key)
        return self.head(self.norm(x[:,N+1:]))


class PhonoPathJEPA(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.student=Encoder(cfg); self.teacher=copy.deepcopy(self.student)
        self.teacher.requires_grad_(False);self.predictor=Predictor(cfg)
    @torch.no_grad()
    def ema_update(self,m):
        for s,t in zip(self.student.parameters(),self.teacher.parameters()):t.mul_(m).add_(s,alpha=1-m)
        for s,t in zip(self.student.buffers(),self.teacher.buffers()):t.copy_(s)
