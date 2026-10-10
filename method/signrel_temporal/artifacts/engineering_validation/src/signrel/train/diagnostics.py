import numpy as np
import torch


def embedding_health(features):
    x = np.asarray(features,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        return {'status':'insufficient_samples'}
    norm = np.linalg.norm(x,axis=1,keepdims=True)
    unit = x/np.maximum(norm,1e-12)
    mean_cos = (np.square(unit.sum(0)).sum()-np.square(unit).sum())/(len(x)*(len(x)-1))
    centered = x-x.mean(0)
    singular = np.linalg.svd(centered,compute_uv=False)
    energy = singular**2
    p = energy[energy>1e-12]/max(energy.sum(),1e-12)
    return {'status':'evaluated','samples':len(x),'dimensions':x.shape[1],
            'mean_per_dimension_std':float(x.std(0).mean()),'mean_pairwise_cosine':float(mean_cos),
            'effective_rank':float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0,
            'maximum_sample_rank':min(len(x)-1,x.shape[1]),'mean_norm':float(norm.mean())}


def gradient_norms(model):
    out = {}
    for group in ('hand_projection','body_projection','transformer','heads'):
        parameters = getattr(model,group).parameters()
        out[group] = float(torch.stack([p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]).sum().sqrt())
    return out


def heldout_diagnostics(checkpoint,dev_manifest,output,device='cpu'):
    """Fixed unlabeled dev draws and train-only constant/lag shortcut fits."""
    from pathlib import Path
    from ..utils import read_jsonl,atomic_json,sha256,stable_seed
    from ..data.cache import read_cache
    from ..data.windows import WindowDataset,collate,to_device
    from ..eval.embed import restore_encoder,embed_arrays
    from ..losses.state import hand_state_loss,body_state_loss
    from ..losses.pairs import build_pairs
    from ..losses.relation import relation_loss,relation_values
    import torch.nn.functional as F
    model,cfg,state=restore_encoder(checkpoint,device)
    train_rows=read_jsonl(cfg['paths']['cache_manifest']);dev_rows=read_jsonl(dev_manifest)
    if {r['video_id'] for r in train_rows}&{r['video_id'] for r in dev_rows}: raise ValueError('Diagnostics dev/train overlap')
    sums=torch.zeros(2,384);counts=torch.zeros(2);body_sum=torch.zeros(14);body_count=torch.zeros(14)
    lag_values={lag:[] for lag in cfg['loss']['relation']['lags']}
    for row in train_rows:
        a,_=read_cache(row['cache_path']);x=torch.tensor(a['appearance'].astype(np.float32));v=torch.tensor(a['app_valid'])
        for slot in range(2):
            sums[slot]+=x[:,slot][v[:,slot]].sum(0);counts[slot]+=v[:,slot].sum()
        body_sum+=torch.tensor(a['body']).sum(0);body_count+=torch.tensor(a['body_coord_valid']).sum(0)
        x=F.normalize(x,dim=-1)
        for lag in lag_values:
            t=torch.arange(max(0,-lag),min(len(x),len(x)-lag));u=t+lag
            keep=v[t,0]&v[u,1]
            if lag: keep&=torch.tensor(a['source_frame_index'][t.numpy()]!=a['source_frame_index'][u.numpy()])
            lag_values[lag].extend((x[t[keep],0]*x[u[keep],1]).sum(-1).tolist())
    constant=sums/counts.clamp_min(1)[:,None];constant_body=body_sum/body_count.clamp_min(1)
    lag_means={lag:float(np.mean(values)) if values else 0 for lag,values in lag_values.items()}
    data=WindowDataset(dev_manifest,cfg,2026,{'synthetic':False})
    totals={kind:[] for kind in ('learned','train_mean','nearest_visible','lag_only_relation')}
    torch.set_num_threads(4)
    with torch.inference_mode():
        for draw in range(16):
            batch=to_device(collate([data.sample(2*draw),data.sample(2*draw+1)]),device)
            c,s=batch['clean'],batch['student'];p=model(s);r=cfg['loss']['relation']
            pairs=build_pairs(c['app_valid'],c['app_quality'],s['app_ssl_mask'],batch['mapping']['source_frame_index'],s['time_padding'],r['lags'],'cross_signed',32,torch.Generator().manual_seed(draw),.5)
            nearest=c['appearance'].clone();nearest_body=c['body'].clone()
            for i in range(len(nearest)):
                for slot in range(2):
                    visible=torch.nonzero(c['app_valid'][i,:,slot]&~s['app_ssl_mask'][i,:,slot]).flatten()
                    for t in torch.nonzero(s['app_ssl_mask'][i,:,slot]).flatten():
                        nearest[i,t,slot]=c['appearance'][i,visible[(visible-t).abs().argmin()],slot] if len(visible) else constant[slot].to(device)
                visible=torch.nonzero(c['body_valid'][i]&~s['body_ssl_mask'][i]).flatten()
                for t in torch.nonzero(s['body_ssl_mask'][i]).flatten():
                    nearest_body[i,t]=c['body'][i,visible[(visible-t).abs().argmin()]] if len(visible) else constant_body.to(device)
            predictions={'learned':(p['app_prediction'],p['body_prediction']),
                         'train_mean':(constant.to(device).expand_as(nearest),constant_body.to(device).expand_as(nearest_body)),
                         'nearest_visible':(nearest,nearest_body)}
            for kind,(app,body) in predictions.items():
                hand=hand_state_loss(app,c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'])[0]
                body_loss=body_state_loss(body,c['body'],c['body_valid'],c['body_coord_valid'],c['body_coord_quality'],s['body_ssl_mask'])[0]
                rel=relation_loss(app,c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'],pairs,r['smooth_l1_beta'])[0]
                totals[kind].append([float(hand),float(body_loss),float(rel)])
            if len(pairs):
                _,target=relation_values(c['appearance'],c['appearance'],pairs)
                predicted=torch.tensor([lag_means[int(v)] for v in (pairs[:,3]-pairs[:,1]).tolist()],device=device)
                weight=torch.minimum(c['app_quality'][pairs[:,0],pairs[:,1],pairs[:,2]],c['app_quality'][pairs[:,0],pairs[:,3],pairs[:,4]])
                error=F.smooth_l1_loss(predicted,target,reduction='none',beta=r['smooth_l1_beta'])
                totals['lag_only_relation'].append(float((error*weight).sum()/weight.sum().clamp_min(1e-12)))
        learned,teacher=[],[]
        raw_context=[];raw_teacher=[];prediction_norm=[]
        for row in dev_rows:
            a,_=read_cache(row['cache_path'])
            for collection,encoder in ((learned,model),(teacher,None)):
                result=embed_arrays(a,encoder,device=device,decoder_diagnostics=encoder is not None)
                if result['status']=='ok':
                    collection.append(result['embedding'])
                    (raw_context if encoder is not None else raw_teacher).append(result['raw_stream_mean'])
                    if encoder is not None: prediction_norm.append(result['decoder_prediction_norm'])
        # Pooled diagnostics have only three held-out clips: rank is capped at two.
    result={'status':'measured','checkpoint_sha256':sha256(checkpoint),'dev_manifest_sha256':sha256(dev_manifest),
            'fixed_draw_seed':2026,'microbatches':16,'train_only_shortcut_fit':True,
            'errors':{k:np.mean(v,axis=0).tolist() if v else None for k,v in totals.items()},
            'error_order':['hand_squared_L2_stream_mean','body_smooth_L1','relation_smooth_L1'],
            'learned_pooled':embedding_health(np.stack(learned)) if len(learned)>1 else {'status':'insufficient_samples'},
            'teacher_pooled':embedding_health(np.stack(teacher)) if len(teacher)>1 else {'status':'insufficient_samples'},
            'learned_raw':embedding_health(np.stack(raw_context)) if len(raw_context)>1 else {'status':'insufficient_samples'},
            'teacher_raw':embedding_health(np.stack(raw_teacher)) if len(raw_teacher)>1 else {'status':'insufficient_samples'},
            'decoder_prediction_norm_mean':float(np.mean(prediction_norm)) if prediction_norm else None,
            'scope':'Three fixed held-out SSL clips; descriptive diagnostics, sample rank cap matters. Nearest-visible is piecewise constant, not linear interpolation.'}
    atomic_json(output,result);return result
