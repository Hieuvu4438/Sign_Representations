"""Resumable full training with pending-validation transactions and stage saves."""
from __future__ import annotations
import argparse,copy,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from fp.utils import atomic_json,atomic_torch,load_config,rng_state,restore_rng,sha256
from fp.preflight import verify
from fp.data.dataset import Cfg,collate
from fp.data.stream import Windows,Batches
from fp.trainer import Trainer
from fp import checkpoint
from fp.inference import split_rows,extract
from fp.evaluation import dictionary


def validation_due(step,c,done):
    return step not in done and (step==0 or step==c['warmup_steps'] or step%c['validate_every']==0 or step==c['steps'])


def run(config,root):
    c=load_config(config);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if c['device'].startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.23)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    import random
    random.seed(c['seed'])
    cache,proof=verify(c);tr=Trainer(c)
    previous,recovery=checkpoint.load(root)
    done=0;progress={'validated':[],'best':None,'pending_validation':None,'started_unix':time.time()}
    if previous is not None:done,progress=tr.restore(previous,proof)
    else:
        if (root/'started.json').exists():raise RuntimeError('Run already started but no recovery checkpoint exists')
        ds=Windows(cache,Cfg(),c['seed']);tr.fit_stats(ds)
        checkpoint.save(root,tr.state(0,proof,progress))
        atomic_json(root/'started.json',{'config':c,'proof':proof,'initial_checkpoint_written':True})
    if recovery['read_errors']:
        atomic_json(root/'recovery.json',recovery)
        # Preserve the good previous generation when latest was unreadable.
        if recovery['generation']=='previous.pt' and (root/'last.pt').exists():
            (root/'last.pt').replace(root/f'corrupt_last_{int(time.time())}.pt')
    stop={'value':False}
    def halt(*_):stop['value']=True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,halt)
    def stopped():return stop['value'] or (root.parent/'pause.request').exists()
    def status(phase,**kw):
        atomic_json(root.parent/'progress.json',{'status':'running','phase':phase,'pid':os.getpid(),
                    'step':done,'max_steps':c['steps'],'updated_unix':time.time(),'best_validation':progress['best'],**kw})
    def save():checkpoint.save(root,tr.state(done,proof,progress))
    def validate():
        nonlocal progress
        progress['pending_validation']=done;save()
        rng=rng_state()
        try:
            status('validation',activity='starting',stride=32)
            train_rows=split_rows('train');val_rows=split_rows('val')
            if c['validation_limit']:
                # Smoke uses matching gloss small gallery/query and is marked nonprotocol.
                val_rows=val_rows[:c['validation_limit']]
                keep={r['gloss'] for r in val_rows};train_rows=[r for r in train_rows if r['gloss'] in keep][:max(32,c['validation_limit']*10)]
            import hashlib
            h=hashlib.sha256(json.dumps({'step':done,'proof':proof,'stride':32},sort_keys=True).encode())
            for tensor in tr.model.teacher.state_dict().values():h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            identity=h.hexdigest()
            folder=root.parent/'validation_features'/str(done)
            kwargs=dict(cache=cache,identity=identity,device=c['device'],workers=c['extract_workers'],stride=32,stop=stopped,
                        notify=lambda stage,n,total:status('validation',activity=stage,completed=n,total=total))
            gallery=extract(tr.model.teacher,train_rows,out=folder/'train',**kwargs)
            query=extract(tr.model.teacher,val_rows,out=folder/'val',**kwargs)
            results={}
            for mode in ('fusion','pl_gate0'):
                m,ranks=dictionary(gallery[mode],query[mode],train_rows,val_rows,c['device'],stopped)
                results[mode]=m
            record={'step':done,'results':results,'checkpoint_sha256':identity,'protocol_complete':not c['smoke'],
                    'readout':'raw window mean -> videoL2; stride32 validation; no PCA','gates':tr.model.teacher.gate.detach().cpu().tolist()}
            atomic_json(root/'validation'/f'{done:08}.json',record)
            score=results['fusion']['Rec@1']
            progress['validated'].append(done);progress['pending_validation']=None
            restore_rng(rng)
            if progress['best'] is None or score>progress['best']['score']:
                progress['best']={'step':done,'score':score}
                atomic_torch(root/'best_validation.pt',tr.state(done,proof,progress))
            save()
            if done in (0,c['warmup_steps'],c['steps']):atomic_torch(root/f'stage_{done:08}.pt',tr.state(done,proof,progress))
            # Only committed validation caches are disposable; no raw data touched.
            import shutil
            shutil.rmtree(folder)
            status('train',last_validation=record)
        finally:restore_rng(rng)
    try:
        if progress['pending_validation'] is not None or validation_due(done,c,progress['validated']):validate()
        ds=Windows(cache,Cfg(),c['seed'])
        loader=torch.utils.data.DataLoader(ds,batch_sampler=Batches(len(ds),c['batch'],c['seed'],done,c['steps']),
              num_workers=c['workers'],collate_fn=collate,pin_memory=c['device'].startswith('cuda'),
              multiprocessing_context='spawn' if c['workers'] else None,generator=torch.Generator().manual_seed(917))
        with (root/'train.jsonl').open('a') as log:
            for batch in loader:
                if stopped():break
                metrics=tr.update(batch,done);done+=1
                if done==1 or done%50==0 or done==c['steps']:
                    log.write(json.dumps({'step':done,'unix':time.time(),**metrics})+'\n');log.flush()
                    status('train',metrics=metrics)
                if done%c['save_every']==0:save()
                if validation_due(done,c,progress['validated']):validate()
        save()
        if stopped():status('paused',reason='graceful interruption');return 75
        if done!=c['steps']:raise RuntimeError('Data stream ended before budget')
        if validation_due(done,c,progress['validated']):validate()
        atomic_torch(root/'final.pt',tr.state(done,proof,progress))
        atomic_json(root/'complete.json',{'status':'complete','steps':done,'best':progress['best'],
                    'validated':progress['validated'],'proof':proof,'finished_unix':time.time()})
        status('training_complete');return 0
    except InterruptedError:
        save();status('paused',reason='validation/extraction interrupted');return 75
    except torch.cuda.OutOfMemoryError:
        # Commit only the last fully completed optimizer update, from disk; never save partially updated state.
        atomic_json(root.parent/'progress.json',{'status':'waiting_for_gpu','phase':'oom_requeue','step':done,'pid':os.getpid(),'updated_unix':time.time()})
        return 76


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    a=p.parse_args()
    try:code=run(a.config,a.root)
    except torch.cuda.OutOfMemoryError:
        print('CUDA_OOM_DURING_INITIALIZATION: requeue from last committed checkpoint',flush=True);code=76
    raise SystemExit(code)
