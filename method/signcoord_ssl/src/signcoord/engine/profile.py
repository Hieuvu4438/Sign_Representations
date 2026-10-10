import random
import time

import numpy as np
import torch

from signcoord.data.collate import collate_ssl
from signcoord.provenance import execution_stamp,object_hash,sha256_file
from signcoord.targets.statistics import read_statistics
from .train import batch_to_device,compute_loss,make_dataset,make_model,optimizer_groups,paired_indices


def profile(c,warmup_steps=50,measure_steps=500):
    if c['arm'] in ('A0','A1_ref'): raise ValueError('reference is frozen; training profile is not applicable')
    if warmup_steps<1 or measure_steps<1: raise ValueError('positive profile warmup/measurement required')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    device=torch.device(c['train']['device'])
    ds=make_dataset(c); indices,_=paired_indices(len(ds),c['train']['microbatch_per_gpu'],0,c['seed'])
    batch=batch_to_device(collate_ssl([ds[i] for i in indices]),device)
    stats=read_statistics(c['data']['target_stats_path'],c['data']['ssl_train_manifest'],ds)
    phase=(read_statistics(c['data']['phase_calibration_path'],c['data']['ssl_train_manifest'],ds)
           if c['loss']['relation']['kind'] in ('phase','phase_basic','phase_shuffle','magnitude') else None)
    model=make_model(c).to(device).train()
    optimizer=torch.optim.AdamW(optimizer_groups(model,c),betas=tuple(c['train']['betas']))
    precision=c['train']['precision']
    dtype=torch.bfloat16 if precision=='bf16' or (precision=='bf16_if_supported_else_fp16' and torch.cuda.is_bf16_supported()) else torch.float16
    amp=device.type=='cuda' and precision!='fp32'
    scaler=torch.amp.GradScaler(device.type,enabled=amp and dtype==torch.float16)
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=dtype,enabled=amp):
            loss,log=compute_loss(model,batch,c,stats,phase,c['loss']['relation']['ramp_end_update'])
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),c['train']['grad_clip_norm'],error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        return log
    for _ in range(warmup_steps): step()
    if device.type=='cuda': torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    for _ in range(measure_steps): log=step()
    if device.type=='cuda': torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    return {**execution_stamp(),'status':'passed','arm':c['arm'],'config_hash':object_hash(c),
            'manifest_hash':sha256_file(c['data']['ssl_train_manifest']),
            'warmup_steps':warmup_steps,'measured_microsteps':measure_steps,'elapsed_seconds':elapsed,
            'clips_per_second':measure_steps*len(indices)/elapsed,'microbatch':len(indices),
            'stats_batch':len(indices),'device':str(device),'precision':precision,
            'parameters':sum(p.numel() for p in model.parameters()),
            'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
            'fixed_sample_ids':log['sample_ids'],'relation_weight_during_profile':log['relation_weight'],
            'profile_policy':'full_relation_weight; fixed_predecoded_batch; includes_optimizer; excludes_io_and_init',
            'stable_budget_estimate':measure_steps>=500,'training_gpu_hours':elapsed/3600 if device.type=='cuda' else 0.}
