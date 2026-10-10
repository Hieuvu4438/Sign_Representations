"""Bounded single-process training with deterministic plans and exact resume."""
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from ..config import ROOT
from ..utils import sha256,fingerprint,code_hash,atomic_json,read_jsonl,stable_seed
from ..data.manifest import read_ssl,reject_semantics
from ..data.windows import WindowDataset,collate,to_device
from ..models.encoder import initialized_model
from ..losses.state import hand_state_loss,body_state_loss
from ..losses.pairs import build_pairs
from ..losses.relation import relation_loss
from .checkpoint import rng_states,restore_rng,atomic_checkpoint,load_checkpoint,tensor_hash
from .diagnostics import gradient_norms
from ..budget import remaining,charge


def choose_device(cfg,override=None):
    name = override or cfg['train']['device']
    if name=='auto': name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    return torch.device(name)


def choose_precision(cfg,device):
    name = cfg['train']['precision']
    if name=='bf16_if_supported_else_fp32':
        name = 'bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if device.type=='cpu' and name!='fp32':
        raise ValueError('Core CPU smoke uses FP32; mixed precision must be GPU verified')
    return name


def lr_at(update,train):
    warmup,max_updates = train['warmup_updates'],train['max_updates']
    if warmup>0 and update<warmup:
        return train['learning_rate']*(update+1)/warmup
    progress = min(1,max(0,(update-warmup)/max(1,max_updates-warmup)))
    return train['min_learning_rate']+.5*(train['learning_rate']-train['min_learning_rate'])*(1+math.cos(math.pi*progress))


def optimizer_for(model,train):
    decay,no_decay = [],[]
    for name,param in model.named_parameters():
        if param.ndim<=1 or name in ('part','mask_token','missing_token'):
            no_decay.append(param)
        else: decay.append(param)
    return torch.optim.AdamW([{'params':decay,'weight_decay':train['weight_decay']},
                             {'params':no_decay,'weight_decay':0}],lr=train['learning_rate'],betas=train['betas'])


def validate_training_inputs(cfg):
    rows = read_ssl(cfg['paths']['ssl_manifest'])
    cache_rows = read_jsonl(cfg['paths']['cache_manifest'])
    for row in cache_rows: reject_semantics(row,'cache_manifest')
    ssl_ids = {r['video_id'] for r in rows}
    if any(r['video_id'] not in ssl_ids for r in cache_rows):
        raise ValueError('Training cache includes a video outside SSL manifest')
    expected = None
    if not cfg['experiment']['synthetic']:
        lock = json.loads(Path(cfg['paths']['sources_lock']).read_text())
        if lock['status']!='VERIFIED_RUNTIME' or not lock['teacher']['runtime_verified']:
            raise ValueError('Real training requires runtime-verified source lock')
        if lock['teacher']['feature_dim']!=cfg['teacher']['feature_dim']:
            raise ValueError('Teacher dimension differs from lock')
        protocol=json.loads(Path(cfg['paths']['protocol']).read_text())
        expected = {'teacher_sha256':lock['teacher']['checkpoint_sha256'],'synthetic':False,
                    'split_fingerprint':protocol['split_fingerprint'],'source_commit':lock['teacher']['commit_sha'],
                    'perception_hashes':{k:v['sha256'] for k,v in lock['perception'].items()}}
        for asset in lock['perception'].values():
            if not asset['runtime_verified']: raise ValueError('Perception dependency not runtime verified')
        audit_path = Path('artifacts/split_audit.json')
        if not audit_path.exists() or json.loads(audit_path.read_text()).get('status')!='passed':
            raise ValueError('Real training requires clean split audit')
        audit=json.loads(audit_path.read_text())
        if audit.get('ssl_train_sha256')!=sha256(cfg['paths']['ssl_manifest']): raise ValueError('SSL train manifest differs from split audit')
        if len({r['video_id'] for r in cache_rows})!=len(cache_rows): raise ValueError('Duplicate cache IDs')
    else:
        if any(not r.get('synthetic') for r in cache_rows): raise ValueError('Synthetic experiment mixing real caches')
        expected = {'synthetic':True}
    dataset = WindowDataset(cfg['paths']['cache_manifest'],cfg,cfg['experiment']['seed'],expected)
    return dataset,expected


def train(cfg,resume=None,stop_after=None,device_override=None,dry_run=False):
    if device_override:
        import copy
        cfg=copy.deepcopy(cfg); cfg["train"]["device"]=device_override
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(4)
    dataset,expected = validate_training_inputs(cfg)
    device = choose_device(cfg,device_override)
    precision = choose_precision(cfg,device)
    experiment,tc = cfg['experiment'],cfg['train']
    seed = experiment['seed']
    random.seed(stable_seed(seed,'python'))
    np.random.seed(stable_seed(seed,'numpy')%(2**32))
    torch.manual_seed(stable_seed(seed,'dropout'))
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)
    model = initialized_model(cfg['model'],stable_seed(seed,'initialization')).to(device)
    optimizer = optimizer_for(model,tc)
    scaler = torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    out = Path(experiment['output_dir'])
    source_lock = Path(cfg['paths']['sources_lock'])
    protocol = Path(cfg['paths']['protocol'])
    contract = {'schema_version':1,'config_sha256':fingerprint(cfg),'source_sha256':code_hash(ROOT),
                'sources_lock_sha256':sha256(source_lock) if source_lock.exists() and not experiment['synthetic'] else 'synthetic',
                'cache_manifest_sha256':sha256(cfg['paths']['cache_manifest']),
                'ssl_manifest_sha256':sha256(cfg['paths']['ssl_manifest']),
                'split_fingerprint':json.loads(protocol.read_text()).get('split_fingerprint',sha256(protocol)),
                'eval_protocol_sha256':sha256(protocol),
                'cache_files_fingerprint':fingerprint({r['video_id']:sha256(r['cache_path']) for r in dataset.rows})}
    step,consumed,elapsed_before = 0,0,0.
    if resume:
        state = load_checkpoint(resume,contract)
        model.load_state_dict(state['model_state'],strict=True)
        optimizer.load_state_dict(state['optimizer_state'])
        scaler.load_state_dict(state['scaler_state_or_null'])
        step,consumed,elapsed_before = state['successful_update'],state['consumed_microbatch_index'],state['elapsed_seconds']
        restore_rng(state['rng_states_python_numpy_torch_cpu_cuda'])
        if step>=tc['max_updates']:
            return {'status':'already_complete','successful_updates':step,'run_dir':str(out)}
    elif (out/'run_manifest.json').exists():
        prior = json.loads((out/'run_manifest.json').read_text())
        if prior['config_sha256']!=contract['config_sha256']:
            raise ValueError('Refusing overwrite of a run directory with a different config')
        raise ValueError('Existing run requires explicit --resume or a new output directory')
    if dry_run:
        batch = to_device(collate([dataset.sample(i) for i in range(tc['microbatch_size'])]),device)
        with torch.no_grad(): outputs = model(batch['student'])
        return {'status':'validated','synthetic':experiment['synthetic'],'context_shape':list(outputs['context'].shape),'checkpoint_created':False}
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    if not resume:
        init_root = Path('runs/initializations')
        init_root.mkdir(parents=True,exist_ok=True)
        init_path = init_root/f'{seed}_{fingerprint(cfg["model"])}.pt'
        if init_path.exists():
            init = torch.load(init_path,map_location='cpu',weights_only=True)
            if tensor_hash(init)!=tensor_hash(model.state_dict()): raise ValueError('Paired initialization differs')
        else: atomic_checkpoint(init_path,model.state_dict())
        with tarfile.open(out/'source.tar.gz','w:gz') as archive:
            for folder in ('src','configs','tests'):
                for path in sorted((ROOT/folder).rglob('*')):
                    if path.suffix in ('.py','.yaml'): archive.add(path,arcname=str(path.relative_to(ROOT)))
    init_sha = sha256(Path('runs/initializations')/f'{seed}_{fingerprint(cfg["model"])}.pt')
    git_commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    git_patch = subprocess.run(['git','diff','--',str(ROOT)],cwd=ROOT,capture_output=True,text=True).stdout
    (out/'git_diff.patch').write_text(git_patch)
    manifest = {**contract,'run_id':experiment['name']+f'_seed{seed}','variant':experiment['variant'],'seed':seed,
                'synthetic':experiment['synthetic'],'status':'running','precision':precision,'device':str(device),
                'init_checkpoint_sha256':init_sha,'effective_batch':tc['microbatch_size']*tc['gradient_accumulation'],
                'parameter_count':sum(p.numel() for p in model.parameters()),'git_commit':git_commit,'dirty_patch_sha256':sha256(out/'git_diff.patch')}
    atomic_json(out/'run_manifest.json',manifest)
    for name in ("requirements-core.lock.txt","requirements-perception.lock.txt"):
        if (ROOT/name).exists(): shutil.copy2(ROOT/name,out/name)
    for name,path in (("sources.lock.json",source_lock),("protocol.json",protocol)):
        if path.exists(): shutil.copy2(path,out/name)
    global_gpu_remaining=remaining("gpu_seconds")
    global_wall_remaining=remaining("wall_training_seconds")
    start = time.monotonic()
    stop_reason = 'complete'
    plan_history = []

    def save(status):
        elapsed = elapsed_before+time.monotonic()-start
        payload = {**contract,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                   'scheduler_state':{'successful_update':step,'schedule':'warmup_cosine_by_successful_update'},
                   'scaler_state_or_null':scaler.state_dict(),'successful_update':step,'consumed_microbatch_index':consumed,
                   'sampler_state':{'consumed_microbatch_index':consumed},'rng_states_python_numpy_torch_cpu_cuda':rng_states(),
                   'config_resolved':cfg,'metric_history_path':str(out/'metrics.jsonl'),'elapsed_seconds':elapsed,
                   'budget_consumed':{'wall_seconds':elapsed,'gpu_seconds':elapsed if device.type=='cuda' else 0},
                   'git_commit_and_dirty_patch_hash':[git_commit,manifest['dirty_patch_sha256']],'stop_reason':status,
                   'init_checkpoint_sha256':init_sha}
        atomic_checkpoint(out/'checkpoints/last.pt',payload)
        if step and step%tc['checkpoint_every']==0:
            path = out/f'checkpoints/update_{step:06d}.pt'
            if not path.exists():
                try: path.hardlink_to(out/'checkpoints/last.pt')
                except OSError: atomic_checkpoint(path,payload)
        manifest.update(status=status,successful_updates=step,consumed_microbatch_index=consumed,
                        elapsed_seconds=elapsed,checkpoint_sha256=sha256(out/'checkpoints/last.pt'),
                        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
        atomic_json(out/'run_manifest.json',manifest)
        charge('train:'+str(out.resolve()),'training',gpu_seconds=elapsed if device.type=='cuda' else 0,
               wall_training_seconds=elapsed)

    def budget_exhausted():
        elapsed = elapsed_before+time.monotonic()-start
        invocation_elapsed=time.monotonic()-start
        return (elapsed>=cfg['budget']['max_wall_minutes']*60 or invocation_elapsed>=global_wall_remaining or
                (device.type=='cuda' and (elapsed>=cfg['budget']['max_gpu_hours']*3600 or invocation_elapsed>=global_gpu_remaining)))

    model.train()
    try:
        while step<tc['max_updates'] and (stop_after is None or step<stop_after):
            if budget_exhausted(): stop_reason='budget_stopped'; break
            optimizer.zero_grad(set_to_none=True)
            components = []
            local_plans = []
            relation_cfg = cfg['loss']['relation']
            relation_weight = relation_cfg['max_weight']*min(1,step/max(1,relation_cfg['ramp_updates'])) if relation_cfg['enabled'] else 0
            active_targets = 0
            partial_budget_stop = False
            learning_rate = lr_at(step,tc)
            for group in optimizer.param_groups: group['lr'] = learning_rate
            for micro in range(tc['gradient_accumulation']):
                if budget_exhausted(): partial_budget_stop=True; break
                index = consumed+micro
                samples = [dataset.sample(index*tc['microbatch_size']+j) for j in range(tc['microbatch_size'])]
                batch = to_device(collate(samples),device)
                clean,student = batch['clean'],batch['student']
                pair_rng = torch.Generator().manual_seed(stable_seed(seed,'pair',index))
                pairs = build_pairs(clean['app_valid'],clean['app_quality'],student['app_ssl_mask'],
                                     batch['mapping']['source_frame_index'],student['time_padding'],relation_cfg['lags'],
                                     relation_cfg['mode'],relation_cfg['max_pairs_per_sample'],pair_rng,cfg['data']['quality_threshold'])
                dtype = torch.bfloat16 if precision=='bf16' else torch.float16
                with torch.autocast(device.type,dtype=dtype,enabled=precision!='fp32'):
                    outputs = model(student)
                hand,hd,hterms = hand_state_loss(outputs['app_prediction'],clean['appearance'],clean['app_valid'],clean['app_quality'],student['app_ssl_mask'])
                body,bd = body_state_loss(outputs['body_prediction'],clean['body'],clean['body_valid'],clean['body_coord_valid'],clean['body_coord_quality'],student['body_ssl_mask'],cfg['loss']['body_smooth_l1_beta'])
                relation,rd,rs = relation_loss(outputs['app_prediction'],clean['appearance'],clean['app_valid'],clean['app_quality'],student['app_ssl_mask'],pairs,
                                               relation_cfg['smooth_l1_beta'],relation_cfg['mode'],pair_rng)
                if len(pairs):
                    actual=student['relative_time_ms']
                    physical=actual[pairs[:,0],pairs[:,3]]-actual[pairs[:,0],pairs[:,1]]
                    rs['physical_lag_distribution_ms']={'mean':float(physical.mean()),'std':float(physical.std(unbiased=False)),
                                                        'min':float(physical.min()),'max':float(physical.max())}
                total = hand+cfg['loss']['body_weight']*body+relation_weight*relation
                if not torch.isfinite(total): raise FloatingPointError('Non-finite training loss')
                active_targets += int((hd>0).sum())+int(bd>0)
                scaler.scale(total/tc['gradient_accumulation']).backward()
                components.append({'loss':float(total.detach()),'hand':float(hand.detach()),'hand_LH':float(hterms[0].detach()),
                                   'hand_RH':float(hterms[1].detach()),'body':float(body.detach()),'relation':float(relation.detach()),
                                   'hand_denominators':hd.tolist(),'body_denominator':float(bd),'relation_denominator':float(rd),
                                   'relation_stats':rs,'prediction_norm':float(outputs['app_prediction'].detach().float().norm(dim=-1).mean()),
                                   'context_std':float(outputs['context'].detach().float().std(dim=(0,1,2)).mean()),
                                   'hand_mask_count':int(student['app_ssl_mask'].sum()),'body_mask_count':int(student['body_ssl_mask'].sum()),
                                   'natural_hand_count':int(clean['app_valid'].sum()),'natural_body_count':int(clean['body_valid'].sum())})
                local_plans.append({'microbatch_index':index,'samples':batch['metadata'],'app_mask_sha256':fingerprint(student['app_ssl_mask'].cpu().tolist()),
                                    'body_mask_sha256':fingerprint(student['body_ssl_mask'].cpu().tolist())})
            if partial_budget_stop:
                optimizer.zero_grad(set_to_none=True)
                stop_reason='budget_stopped'; break
            consumed += tc['gradient_accumulation']
            if active_targets==0:
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite training gradients')
            per_module = gradient_norms(model)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip_norm']))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            with (out/'window_plan.jsonl').open('a') as file:
                for plan in local_plans: file.write(json.dumps(plan)+'\n')
            event = {'successful_update':step,'consumed_microbatch_index':consumed,'elapsed_seconds':elapsed_before+time.monotonic()-start,
                     'learning_rate':learning_rate,'grad_norm':grad_norm,'gradient_norms':per_module,
                     'relation_weight':relation_weight,'scaled_body_weight':cfg['loss']['body_weight'],'microbatches':components,
                     'examples':tc['microbatch_size']*tc['gradient_accumulation'],'synthetic':experiment['synthetic']}
            with (out/'metrics.jsonl').open('a') as file: file.write(json.dumps(event,allow_nan=False)+'\n')
            if step%tc['log_every']==0:
                print(json.dumps({'update':step,'hand':np.mean([r['hand'] for r in components]),'relation':np.mean([r['relation'] for r in components]),'elapsed_seconds':event['elapsed_seconds']}),flush=True)
            if step%tc['checkpoint_every']==0: save('running')
        if stop_reason=='complete' and step<tc['max_updates']: stop_reason='stopped_at_update_boundary'
        save(stop_reason)
    except Exception as exc:
        atomic_json(out/'failure.json',{'error':str(exc),'step':step,'samples':local_plans if 'local_plans' in locals() else []})
        # Never persist partially updated gradients/parameters as a successful boundary.
        manifest.update(status='failed',error=str(exc))
        atomic_json(out/'run_manifest.json',manifest)
        raise
    return {'status':stop_reason,'synthetic':experiment['synthetic'],'successful_updates':step,'run_dir':str(out),
            'checkpoint_sha256':manifest['checkpoint_sha256'],'elapsed_seconds':manifest['elapsed_seconds']}
