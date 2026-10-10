"""Aggregate overlapping contextual windows before fixed physical-time pooling."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..data.windows import gather_window
from ..data.corruption import student_inputs
from ..models.encoder import initialized_model
from ..utils import sha256


def window_starts(length,window=64,stride=32):
    if length<1 or not 1<=stride<=window: raise ValueError('Invalid window/stride')
    if length<=window: return [0]
    starts = list(range(0,length-window+1,stride))
    if starts[-1]!=length-window: starts.append(length-window)
    return starts


def temporal_pyramid(streams,valid,time_ms):
    if len(streams)!=len(valid): raise ValueError('Stream validity mismatch')
    time_ms = torch.as_tensor(time_ms,dtype=torch.float64)
    midpoint = (time_ms.min()+time_ms.max())/2
    bins = (torch.ones(len(time_ms),dtype=torch.bool),time_ms<midpoint,time_ms>=midpoint)
    blocks = []
    for values,observed in zip(streams,valid):
        values,observed = torch.as_tensor(values,dtype=torch.float32),torch.as_tensor(observed,dtype=torch.bool)
        if len(values)!=len(time_ms) or observed.shape != time_ms.shape: raise ValueError('Pooling shape mismatch')
        for temporal in bins:
            selected = observed&temporal
            vector = values[selected].mean(0) if selected.any() else torch.zeros(values.shape[-1])
            blocks.append(F.normalize(vector,dim=0,eps=1e-12))
    return F.normalize(torch.cat(blocks),dim=0,eps=1e-12)


@torch.inference_mode()
def embed_arrays(arrays,model=None,window=64,stride=32,quality_threshold=.5,device='cpu',decoder_diagnostics=False):
    length = len(arrays['appearance'])
    app_valid = arrays['app_valid']&(arrays['app_quality']>=quality_threshold)
    body_valid = arrays['body_valid']&(arrays['body_quality']>=quality_threshold)
    if not app_valid.any():
        return {'status':'insufficient_observation','coverage':{'hand_observations':0},'embedding':None}
    natural = np.concatenate([app_valid,body_valid[:,None]],axis=1)
    if model is None:
        streams = [arrays['appearance'][:,a].astype(np.float32) for a in range(2)]+[arrays['body']]
        vector = temporal_pyramid(streams,[natural[:,s] for s in range(3)],arrays['grid_time_ms'])
        counts = natural.astype(np.int64)
        raw_mean=np.concatenate([values[observed].mean(0) if observed.any() else np.zeros(values.shape[1]) for values,observed in zip(streams,[natural[:,s] for s in range(3)])])
        decoder_norm=None
    else:
        model.eval()
        total = torch.zeros(length,3,model.width)
        decoder_norms=[]
        counts = torch.zeros(length,3,dtype=torch.int64)
        for start in window_starts(length,window,stride):
            g = np.arange(start,min(length,start+window))
            clean,padding,time,_ = gather_window(arrays,g,window,quality_threshold)
            if not (clean['app_valid'].any() or clean['body_valid'].any()): continue
            student = student_inputs(clean,torch.zeros(window,2,dtype=torch.bool),torch.zeros(window,dtype=torch.bool),padding,time)
            student = {k:v[None].to(device) for k,v in student.items()}
            if decoder_diagnostics:
                if not hasattr(model,'heads'): raise ValueError('Decoder diagnostics require a full training checkpoint')
                outputs=model(student)
                context=outputs['context'][0,:len(g)].float().cpu()
            else:
                context=model.encode(student)[0,:len(g)].float().cpu()
            observed=torch.tensor(app_valid[g],device=device)
            if decoder_diagnostics and observed.any(): decoder_norms.append(float(outputs['app_prediction'][0,:len(g)][observed].float().norm(dim=-1).mean()))
            eligible = torch.cat((clean['app_valid'],clean['body_valid'][:,None]),dim=1)[:len(g)]
            total[g] += context*eligible[...,None]
            counts[g] += eligible.long()
        merged = total/counts.clamp_min(1)[...,None]
        vector = temporal_pyramid([merged[:,s] for s in range(3)],[counts[:,s]>0 for s in range(3)],arrays['grid_time_ms'])
        raw_mean=torch.cat([merged[:,s][counts[:,s]>0].mean(0) if (counts[:,s]>0).any() else torch.zeros(model.width) for s in range(3)]).numpy()
        decoder_norm=float(np.mean(decoder_norms)) if decoder_norms else None
    if not torch.isfinite(vector).all() or vector.norm()==0:
        return {'status':'insufficient_observation','coverage':{'hand_observations':int(app_valid.sum())},'embedding':None}
    return {'status':'ok','embedding':vector.numpy(),'raw_stream_mean':raw_mean,'decoder_prediction_norm':decoder_norm,'coverage':{'hand_observations':int(app_valid.sum()),
            'body_observations':int(body_valid.sum()),'canonical_frames':length,'tail_covered':bool(np.asarray(counts)[-1].any()),
            'unique_frame_ratio':float(len(set(arrays['source_frame_index'][arrays['frame_valid']].tolist()))/max(1,arrays['frame_valid'].sum()))}}


def restore_encoder(path,device='cpu'):
    state = torch.load(path,map_location='cpu',weights_only=False)
    if 'encoder_state' in state:
        cfg=state['config_resolved']
        model=initialized_model(cfg['model'],0).to(device)
        missing,unexpected=model.load_state_dict(state['encoder_state'],strict=False)
        if unexpected or any(not k.startswith('heads.') for k in missing): raise ValueError('Invalid inference artifact state')
        del model.heads
        model.eval().requires_grad_(False)
        return model,cfg,state
    cfg = state['config_resolved']
    model = initialized_model(cfg['model'],0).to(device)
    model.load_state_dict(state['model_state'],strict=True)
    model.eval().requires_grad_(False)
    return model,cfg,state


def embed_cached(recording_cache,encoder_checkpoint):
    from ..data.cache import read_cache
    from ..data.provenance import inference_expectations
    model,cfg,state = restore_encoder(encoder_checkpoint)
    arrays,metadata = read_cache(recording_cache,inference_expectations(state,cfg))
    result = embed_arrays(arrays,model,cfg['eval']['window_frames'],cfg['eval']['stride_frames'],cfg['data']['quality_threshold'])
    result.update(video_id=metadata['video_id'],encoder_sha256=sha256(encoder_checkpoint),preprocess_fingerprint=metadata['preprocess_fingerprint'])
    return result
