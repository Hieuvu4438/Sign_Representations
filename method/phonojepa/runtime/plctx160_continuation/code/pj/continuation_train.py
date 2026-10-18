"""Independent LR branches from native PLctx160, full state and exact post-fork resume."""
from copy import deepcopy
import json
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace
import numpy as np
import torch
from pj.train import Trainer
from pj.data.cache import ClipCache
from pj.data.dataset import Cfg
from pj.continuation_eval import evaluate_checkpoint
from pj.continuation_support import (atomic_json,atomic_torch,sha256,continuation_lr,SeededWindows,
    SampleStream,ReadOnlyCache,training_state,save_last,load_latest,rng_state,restore_rng)


def run(job, root, arm, device='cuda'):
    root=Path(root);folder=root/'arms'/arm['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    torch.set_num_threads(4)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(job['cuda_memory_fraction'])
    baseline=json.loads((root/'baseline/validation/metrics.json').read_text())
    base=torch.load(job['base_checkpoint'],map_location='cpu',weights_only=False)
    legacy_config=deepcopy(base['config']);del base
    legacy_config.update(device=device,workers=job['train_workers'])
    a=SimpleNamespace(**legacy_config)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    trainer=Trainer(a,device);base_step=trainer.load(job['base_checkpoint'])
    if base_step!=200640:raise ValueError('Continuation requires the original200640 checkpoint')
    cfg=Cfg(family=a.family,use_r1=a.use_r1,curriculum=False,morph=a.morph,
            baseline_rate=a.baseline_rate or None,handedness=a.handedness)
    cache=ReadOnlyCache(job['memmap_cache']);dataset=SeededWindows(cache,cfg,seed=0)
    stream=SampleStream(dataset,a.batch,job['sampler_seed'],job['train_workers'])
    config={'arm':arm,'base_checkpoint_sha256':job['base_checkpoint_sha256'],
            'training_cache_sha256':job['training_cache_sha256'],'job_lock_sha256':sha256(root/'job_lock.json'),
            'sampler_seed':job['sampler_seed'],'validation_interval':job['validation_interval'],
            'patience':job['patience'],'ema_momentum':0.9999,'teacher_temperature':0.06,
            'legacy_model_config':legacy_config,'drop_last':True}
    metadata,recovery=load_latest(trainer,stream,config,folder)
    if metadata is None:
        metadata={'branch_step':0,'global_step':base_step,'best_validation_Rec1':baseline['metrics']['Rec@1'],
                  'best_global_step':base_step,'last_validation_branch_step':0,'stale_validations':0,
                  'active_training_seconds':0.,'initialization':'full native checkpoint, retained optimizer/teacher/centers; new deterministic post-fork sampler'}
        atomic_torch(folder/'best.pt',training_state(trainer,stream,metadata,config));save_last(trainer,stream,metadata,config,folder)
    elif recovery:atomic_json(folder/'resume.json',recovery)
    stop={'value':False,'signal':None}
    def halt(number,frame):stop.update(value=True,signal=number)
    handlers={s:signal.signal(s,halt) for s in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]}
    stopped=lambda:stop['value'] or (root/'pause.request').exists()
    def progress(values=None):
        record={'phase':'train','status':'running','arm':arm['name'],'pid':__import__('os').getpid(),
                **metadata,'branch_max_steps':arm['additional_steps'],'updated_unix':time.time(),**(values or {})}
        atomic_json(root/'progress.json',record)
        (root/'progress.txt').write_text(f"PLctx160 continuation | {record['status']} | {arm['name']}\n"
            f"Global step: {metadata['global_step']}\nBranch: {metadata['branch_step']}/{arm['additional_steps']}\n"
            f"Activity: {record.get('activity','training')}\nBest valRec@1: {metadata['best_validation_Rec1']:.6f}\n"
            f"Updated: {time.ctime()}\n")
    def validation():
        rng=rng_state();n=metadata['branch_step']
        try:
            result=evaluate_checkpoint(folder/'last.pt',job,folder/'validation'/f'{n:08d}',encoder=trainer.model.teacher,
                stopped=stopped,progress=lambda value:progress({'activity':'validation',**value}))
        finally:restore_rng(rng)
        score=result['metrics']['Rec@1'];improved=score>metadata['best_validation_Rec1']
        metadata['last_validation_branch_step']=n
        if improved:
            metadata.update(best_validation_Rec1=score,best_global_step=metadata['global_step'],stale_validations=0)
            atomic_torch(folder/'best.pt',training_state(trainer,stream,metadata,config))
        else:metadata['stale_validations']+=1
        save_last(trainer,stream,metadata,config,folder)
        progress({'activity':'training','last_validation_metrics':result['metrics']})
    try:
        progress()
        n=metadata['branch_step'];interval=job['validation_interval']
        if n>metadata['last_validation_branch_step'] and (n%interval==0 or n==arm['additional_steps']) and not stopped():validation()
        while metadata['branch_step']<arm['additional_steps'] and metadata['stale_validations']<job['patience']:
            if stopped():break
            started=time.monotonic();batch=stream.next()
            lr=continuation_lr(metadata['branch_step'],arm['additional_steps'],arm['peak_lr'])
            for group in trainer.opt.param_groups:group['lr']=lr
            trainer.model.train();trainer.model.teacher.eval()
            trainer.opt.zero_grad(set_to_none=True)
            # Legacy schedules stay at their mature160epoch endpoint. Only LR
            # gets a new explicit continuation schedule; model/loss recipe fixed.
            loss,terms=trainer.step(batch,base_step)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite continuation loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainer.params,1.,error_if_nonfinite=True)
            trainer.opt.step();trainer.bal.tick();trainer.commit_centres();trainer.model.ema_update(.9999)
            metadata['branch_step']+=1;metadata['global_step']+=1
            duration=time.monotonic()-started;metadata['active_training_seconds']+=duration
            log={'step':metadata['global_step'],'branch_step':metadata['branch_step'],'loss':float(loss.detach()),
                 **terms,'lr':lr,'grad_norm_before_clip':float(norm),'seconds':duration,'unix':time.time(),
                 'peak_allocated_vram':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None}
            with (folder/'train_log.jsonl').open('a') as file:file.write(json.dumps(log)+'\n')
            if metadata['branch_step']%100==0 or metadata['branch_step']==1:save_last(trainer,stream,metadata,config,folder)
            progress({'last_update':log})
            if stopped():break
            if metadata['branch_step']%interval==0 or metadata['branch_step']==arm['additional_steps']:
                save_last(trainer,stream,metadata,config,folder);validation()
        save_last(trainer,stream,metadata,config,folder)
        if stopped():progress({'status':'paused','stop_signal':stop['signal']});return {'status':'paused',**metadata}
        result={'status':'early_stopped' if metadata['stale_validations']>=job['patience'] else 'complete',
                **metadata,'best_checkpoint_sha256':sha256(folder/'best.pt'),'finished_unix':time.time()}
        atomic_json(folder/'complete.json',result);return result
    except InterruptedError:
        save_last(trainer,stream,metadata,config,folder);progress({'status':'paused'});return {'status':'paused',**metadata}
    finally:
        stream.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
