"""Assemble S/P/G inputs; no hidden value reaches encoder or predictor query."""
import torch
from pp.base import features as S
from pp.ptokens import build


def assemble(view,cfg,stats,teacher=False,raw=False):
    prefix='t' if teacher else 's'
    valid=view['valid_f']; B=valid.shape[0]; device=valid.device
    chunks=[]; targets=[]; weights=[]
    result={}
    if cfg.state:
        obs=view[f'{prefix}_obs']; xy=view[f'{prefix}_xy']; conf=view[f'{prefix}_conf']
        feats=S.frame_features(xy,obs,conf,torch.zeros_like(obs),view[f'{prefix}_aff'])
        inputs,present=S.tokens(feats,valid)
        if cfg.state_relations:
            # B1 appends body-frame wrist-nose / hand-hand vectors per sample.
            from pp.base.joints import NOSE_TIP,RWRI,LWRI
            c,s,a=S.anchors(xy,obs);p=S.to_body_frame(xy,c,s,a)
            for ch,wr,other in ((1,RWRI,LWRI),(3,LWRI,RWRI)):
                rel=torch.cat([p[:,:,wr]-p[:,:,NOSE_TIP],p[:,:,wr]-p[:,:,other]],-1)
                safe=(obs[:,:,wr]&obs[:,:,NOSE_TIP]&obs[:,:,other]&valid)
                rr=(rel*safe[...,None]).reshape(B,32,2,4)
                inputs[ch]=torch.cat([inputs[ch],rr.flatten(-2)],-1)
        result.update(s_inputs=inputs,s_present=present)
        vis=(valid.reshape(B,32,2).any(-1)[:,None].expand(-1,7,-1) if teacher else view['visible'])
        if cfg.state_relations and not teacher:
            # Relations depend on both LOC channels and F_HD.
            bad=view['output_hidden'][:,:,[NOSE_TIP,RWRI,LWRI]].any(-1).reshape(B,32,2).any(-1)
            vis=vis.clone();vis[:,[1,3]] &= ~bad[:,None]
        chunks.append(vis.transpose(1,2).flatten(1))
        # Hidden student cells are naturally ABSENT after scrubbing, but still
        # need metadata query keys. Teacher presence gates loss, not queries.
        targets.append(view['target'].transpose(1,2).flatten(1))
        weights.append((S.reliability(view['t_conf'],view['t_obs'],valid)*present).transpose(1,2).flatten(1))
    if cfg.paths:
        # Student P gets its own augmented coordinates, with metadata closure from raw source H.
        pp='t' if teacher else 'p'
        pack=build(view[f'{pp}_xy'],view[f'{pp}_obs'],view[f'{pp}_conf'],valid,
                   torch.zeros_like(view['output_hidden']) if teacher else view.get('p_output_hidden',view['output_hidden']),cfg,stats,
                   view[f'{prefix}_aff'],raw)
        result['p']=pack
        chunks.append(pack['visible'].flatten(1))
        targets.append(pack['target'].flatten(1));weights.append(pack['reliability'].flatten(1))
        result['global']=pack['global_t' if teacher else 'global_s']
        if cfg.global_mode!='none':
            # A3 ports Path-JEPA's whole-time joint masking: a full-global path
            # touching the masked subtree must leave context with its local tokens.
            # A5 deliberately keeps those globals visible as a leakage control.
            gvis=pack['visible'].any(-1) if cfg.arm=='A3' and not teacher else torch.ones(B,5,device=device,dtype=torch.bool)
            chunks.append(gvis)
            targets.append(torch.zeros(B,5,device=device,dtype=torch.bool));weights.append(torch.zeros(B,5,device=device))
    result['visible']=torch.cat(chunks,1);result['target']=torch.cat(targets,1);result['weights']=torch.cat(weights,1)
    return result
