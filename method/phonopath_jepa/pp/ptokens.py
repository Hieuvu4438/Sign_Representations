"""Batched P/G tokenizer; bounded interpolation and SG5 carry source support."""
from __future__ import annotations
import torch
from pp.base import features as S
from pp.base.joints import RH0,LH0,RWRI,LWRI
from pp.paths import families,signature_dims
from pp.signature import logsig2,chen,unwrap

WINDOWS=[(i,8,1) for i in range(0,57,4)]+[(i,16,2) for i in range(0,49,8)]
SEGMENTS=[(i,min(i+4,63)) for i in range(0,63,4)]


def fill_short_gaps(x,obs,unsafe,max_gap=2):
    x=x.clone();obs=obs.clone();unsafe=unsafe.clone();original=obs.clone()
    for gap in range(1,max_gap+1):
        starts=torch.arange(1,x.shape[1]-gap,device=x.device);end=starts+gap
        ok=original[:,starts-1]&original[:,end]
        for k in range(gap):ok=ok&~original[:,starts+k]
        dep=unsafe[:,starts-1]|unsafe[:,end]
        for k in range(gap):
            v=(1-(k+1)/(gap+1))*x[:,starts-1]+((k+1)/(gap+1))*x[:,end]
            x[:,starts+k]=torch.where(ok[...,None],v,x[:,starts+k])
            obs[:,starts+k] |= ok
            unsafe[:,starts+k] |= ok&dep
    return x,obs,unsafe


def smooth_hands(x,obs,unsafe):
    x=x.clone();obs=obs.clone();unsafe=unsafe.clone()
    joints=[RWRI,LWRI]+list(range(LH0,LH0+21))+list(range(RH0,RH0+21))
    idx=torch.arange(x.shape[1],device=x.device)
    ids=[(idx+k).clamp(0,x.shape[1]-1) for k in (-2,-1,0,1,2)]
    vals=[x[:,i][:,:,joints] for i in ids]
    v=sum(a*w for a,w in zip(vals,(-3/35,12/35,17/35,12/35,-3/35)))
    obs[:,:,joints]=torch.stack([obs[:,i][:,:,joints] for i in ids]).all(0)
    unsafe[:,:,joints]=torch.stack([unsafe[:,i][:,:,joints] for i in ids]).any(0)
    x[:,:,joints]=v
    return x,obs,unsafe


def window_path(path,indices,time,global_time=False):
    x=path.value[:,indices].clone()
    if path.angle:x[...,0]=unwrap(x[...,0])
    if time:
        t=(indices.float()/63 if global_time else torch.linspace(0,1,indices.shape[-1],device=x.device)).expand(*x.shape[:-1])
        x=torch.cat([x,t[...,None]],-1)
    return x


def build(xy,obs,conf,valid,hidden,cfg,stats=None,aff=None,raw=False):
    with torch.autocast(device_type=xy.device.type,enabled=False):
        xy=xy.float();conf=conf.float();obs=obs&valid[...,None]
        c,s,a=S.anchors(xy,obs);body=S.to_body_frame(xy,c,s,a,aff)
        height=-S.to_body_frame(xy,c,s,a)[:,:,RWRI,1]
        rest=height<=torch.quantile(height,.1,dim=1)[:,None]+.25
        p,ob,unsafe=fill_short_gaps(body,obs,hidden)
        if cfg.smooth:p,ob,unsafe=smooth_hands(p,ob,unsafe)
        fam=families(p,ob,conf,unsafe,obs);B=xy.shape[0];device=xy.device
        records={};inputs=[];presents=[];visibles=[];targets=[];drops=[];rels=[];energies=[];stills=[];gtokens=[];gstokens=[]
        for ch,paths in enumerate(fam):
            channel={k:[] for k in ('input','present','visible','target','drop','rel','energy','still')}
            for size,stride,scale in ((8,4,1),(16,8,2)):
                starts=torch.arange(0,65-size,stride,device=device)
                ids=starts[:,None]+torch.arange(size,device=device)[None]
                sigs=[];flags=[];es=[];qs=[];bases=[];means=[];bads=[]
                for path in paths:
                    x=window_path(path,ids,cfg.time_channel)
                    sigs.append(logsig2(x,cfg.moment))
                    flags.append((path.observed[:,ids]&valid[:,ids]).all(-1))
                    es.append((x[:,:,1:]-x[:,:,:-1]).norm(dim=-1).sum(-1))
                    qs.append(path.confidence[:,ids].mean(-1));bads.append(path.unsafe[:,ids])
                    if path.basepoint:bases.append(path.value[:,starts])
                    if cfg.path_means:means.append(x[...,:path.value.shape[-1]].mean(-2))
                present=torch.stack(flags,-1)
                if ch==4 and not cfg.relations:present=present&False
                flags=list(present.unbind(-1));sig=torch.cat(sigs,-1);energy=torch.stack(es,-1)
                key=f'{ch}/{scale}'
                coeff_ok=torch.cat([ok[...,None].expand_as(v) for v,ok in zip(sigs,flags)],-1)
                if raw:
                    r=rest[:,ids].all(-1)[...,None].expand_as(present)
                    records.setdefault(key,[]).append(tuple(v.flatten(0,1) for v in (sig,energy,present,r,coeff_ok)))
                    z=sig
                else:
                    if stats is None:raise ValueError('Run pp.fit_stats for frozen train-only statistics')
                    z=stats.transform(key,sig)
                eps=energy.new_zeros(len(paths)) if stats is None else stats.state[key]['eps'].to(energy)
                still=energy<=eps
                chunks=z.split([x.shape[-1] for x in sigs],-1)
                z=torch.cat([torch.where((ok&~st)[...,None],v,torch.zeros_like(v)) for v,ok,st in zip(chunks,flags,still.unbind(-1))],-1)
                base=[];i=0
                for path,ok in zip(paths,flags):
                    if path.basepoint:base.append(bases[i]*ok[...,None]);i+=1
                base=torch.cat(base,-1) if base else z[...,:0]
                mean=torch.cat([v*ok[...,None] for v,ok in zip(means,flags)],-1) if means else z[...,:0]
                q=(torch.stack(qs,-1)*present).sum(-1)/present.sum(-1).clamp_min(1)
                coverage=torch.stack([path.coverage[:,ids].mean(-1) for path in paths],-1).mean(-1)
                if ch==4 and not cfg.relations:coverage=coverage*0
                token=torch.cat([z,base,present.float(),q[...,None],coverage[...,None],mean],-1)
                unsafe_time=torch.stack(bads).any(0)
                touched=unsafe_time.any(-1)
                fraction=(unsafe_time&valid[:,ids]).sum(-1)/valid[:,ids].sum(-1).clamp_min(1)
                # §9.1(b) clarifies §7.2: retain a token with ANY available path;
                # missing paths have individual flags and normalized zero values.
                in_clip=valid[:,ids].all(-1);has=present.any(-1)&in_clip
                values=dict(input=token,present=has,visible=~touched&in_clip,target=touched&(fraction>=.5)&has,
                            drop=touched&(fraction<.5),rel=q*coverage,energy=(energy*present*~still).sum(-1),still=still)
                for k,v in values.items():channel[k].append(v)
            values={k:torch.cat(v,1) for k,v in channel.items()}
            for dst,key in ((inputs,'input'),(presents,'present'),(visibles,'visible'),(targets,'target'),(drops,'drop'),
                            (rels,'rel'),(energies,'energy'),(stills,'still')):dst.append(values[key])
            # Shared endpoints; final [60,63] is padded by repeating endpoint 63 (zero increment).
            starts=torch.arange(0,63,4,device=device)
            ids=(starts[:,None]+torch.arange(5,device=device)[None]).clamp_max(63)
            sig_t=[];sig_s=[];base_t=[];base_s=[];flags_t=[];flags_s=[];energy_t=[];energy_s=[];q_t=[];q_s=[];cov_t=[];cov_s=[]
            for path in paths:
                x=window_path(path,ids,cfg.time_channel,global_time=True)
                logs=logsig2(x,cfg.moment)
                vt=(path.observed[:,ids]&valid[:,ids]).all(-1)
                vs=vt&~path.unsafe[:,ids].any(-1)
                if ch==4 and not cfg.relations:vt=vt&False;vs=vs&False
                if cfg.global_mode=='full':vs=vt
                d=x.shape[-1]
                sig_t.append(chen(logs,vt,d,cfg.moment));sig_s.append(chen(logs,vs,d,cfg.moment))
                flags_t.append(vt.any(-1));flags_s.append(vs.any(-1))
                e=(x[:,:,1:]-x[:,:,:-1]).norm(dim=-1).sum(-1)
                energy_t.append((e*vt).sum(-1));energy_s.append((e*vs).sum(-1))
                for masks,qdest,cdest in ((vt,q_t,cov_t),(vs,q_s,cov_s)):
                    counts=torch.zeros(B,64,device=device)
                    counts.scatter_add_(1,ids.flatten()[None].expand(B,-1),masks[:,:,None].expand(-1,-1,5).flatten(1).float())
                    selected=counts>0
                    den=selected.sum(-1).clamp_min(1)
                    qdest.append(torch.where(selected,path.confidence,0.).sum(-1)/den)
                    cdest.append(torch.where(selected,path.coverage,0.).sum(-1)/den)
                if path.basepoint:
                    for masks,dest in ((vt,base_t),(vs,base_s)):
                        first=masks.long().argmax(-1)
                        val=path.value[torch.arange(B,device=device),starts[first]]
                        dest.append(torch.where(masks.any(-1)[:,None],val,torch.zeros_like(val)))
            def global_input(sigs,bases,flags,es,qs,covs,record=False):
                sig=torch.cat(sigs,-1);present=torch.stack(flags,-1);energy=torch.stack(es,-1)
                key=f'{ch}/0';coeff_ok=torch.cat([ok[:,None].expand_as(v) for v,ok in zip(sigs,flags)],-1)
                if raw:
                    if record:records.setdefault(key,[]).append((sig,energy,present,rest.all(-1)[:,None].expand_as(present),coeff_ok))
                    z=sig
                else:z=stats.transform(key,sig)
                eps=energy.new_zeros(len(paths)) if stats is None else stats.state[key]['eps'].to(energy)
                chunks=z.split([x.shape[-1] for x in sigs],-1)
                z=torch.cat([torch.where((ok&(en>ep))[:,None],v,torch.zeros_like(v)) for v,ok,en,ep in zip(chunks,flags,es,eps)],-1)
                base=torch.cat(bases,-1) if bases else z[:,:0]
                extra=z.new_zeros(B,sum(path.value.shape[-1] for path in paths)) if cfg.path_means else z[:,:0]
                q=(torch.stack(qs,-1)*present).sum(-1)/present.sum(-1).clamp_min(1)
                cov=torch.stack(covs,-1).mean(-1)
                return torch.cat([z,base,present.float(),q[:,None],cov[:,None],extra],-1)
            gtokens.append(global_input(sig_t,base_t,flags_t,energy_t,q_t,cov_t,True))
            gstokens.append(global_input(sig_s,base_s,flags_s,energy_s,q_s,cov_s))
        return dict(inputs=inputs,present=torch.stack(presents,1),visible=torch.stack(visibles,1),target=torch.stack(targets,1),
                    drop=torch.stack(drops,1),reliability=torch.stack(rels,1),energy=torch.stack(energies,1),still=stills,
                    global_t=gtokens,global_s=gstokens,height=height,records=records)
