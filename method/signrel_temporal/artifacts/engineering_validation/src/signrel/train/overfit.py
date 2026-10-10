"""Bounded fitting diagnostic on two fixed real masked windows, not evaluation."""
import time
import torch
from ..models.encoder import initialized_model
from ..data.windows import collate,to_device
from ..losses.state import hand_state_loss,body_state_loss
from ..losses.pairs import build_pairs
from ..losses.relation import relation_loss
from ..utils import atomic_json
from ..budget import charge
from .loop import validate_training_inputs,choose_device,optimizer_for


def overfit(cfg,output,updates=100):
    data,_=validate_training_inputs(cfg);device=choose_device(cfg)
    torch.set_num_threads(4);torch.manual_seed(17)
    model=initialized_model(cfg['model'],17).to(device)
    # Dropout disabled only for this fitting check, outside all experimental runs.
    model.eval();optimizer=optimizer_for(model,cfg['train'])
    batch=to_device(collate([data.sample(0),data.sample(1)]),device)
    c,s=batch['clean'],batch['student'];r=cfg['loss']['relation']
    pairs=build_pairs(c['app_valid'],c['app_quality'],s['app_ssl_mask'],batch['mapping']['source_frame_index'],s['time_padding'],r['lags'],'cross_signed',32,torch.Generator().manual_seed(17),.5)
    history=[];start=time.monotonic()
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True);p=model(s)
        h=hand_state_loss(p['app_prediction'],c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'])[0]
        b=body_state_loss(p['body_prediction'],c['body'],c['body_valid'],c['body_coord_valid'],c['body_coord_quality'],s['body_ssl_mask'])[0]
        rel=relation_loss(p['app_prediction'],c['appearance'],c['app_valid'],c['app_quality'],s['app_ssl_mask'],pairs)[0]
        loss=h+b+.1*rel
        if not torch.isfinite(loss): raise ValueError('Overfit loss nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        history.append({'update':step+1,'hand':float(h.detach()),'body':float(b.detach()),'relation':float(rel.detach()),'total':float(loss.detach())})
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    result={'status':'passed' if history[-1]['total']<history[0]['total'] else 'failed_fitting','fixed_windows':batch['metadata'],
            'updates':updates,'dropout_disabled_for_debug_only':True,'elapsed_seconds':elapsed,'history':history,
            'research_evidence':False,'saved_encoder':False}
    atomic_json(output,result);charge('overfit:'+output,'debug',gpu_seconds=elapsed if device.type=='cuda' else 0,wall_training_seconds=elapsed)
    return result
