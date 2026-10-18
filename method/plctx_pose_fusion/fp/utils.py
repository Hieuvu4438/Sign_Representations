from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with open(tmp,'w') as f:
        json.dump(value,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def load_config(path):
    c=json.loads(Path(path).read_text())
    required={'steps','batch','seed','workers','d','pl_depth','graph_depth','dp','pdepth',
              'lr','wd','lambda_body','lambda_relations','lambda_retention','gate_max',
              'warmup_steps','gate_ramp_steps','validate_every','save_every','stats_clips',
              'cache','pl_checkpoint','device','smoke','extract_workers','validation_limit'}
    if set(c)!=required:raise ValueError(f'Config keys: missing={required-set(c)}, unknown={set(c)-required}')
    for key in ['steps','batch','d','pl_depth','graph_depth','dp','pdepth','warmup_steps','gate_ramp_steps','validate_every','save_every','stats_clips']:
        if not isinstance(c[key],int) or c[key]<=0:raise ValueError(key)
    if c['batch']<2 or c['workers']<0 or c['extract_workers']<0:raise ValueError('batch/workers')
    for key in ['lr','wd','lambda_body','lambda_relations','lambda_retention','gate_max']:
        if not np.isfinite(c[key]) or c[key]<0:raise ValueError(key)
    if not 0<c['gate_max']<=.5:raise ValueError('gate_max must preserve residual bound <=.5')
    if not c['smoke'] and (c['batch'],c['d'],c['pl_depth'],c['graph_depth'],c['dp'],c['pdepth'],c['stats_clips'],c['validation_limit'])!=(32,256,6,8,128,4,640,0):
        raise ValueError('Full protocol configuration mismatch')
    return c
