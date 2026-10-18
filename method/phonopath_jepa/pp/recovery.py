"""Durable checkpoint commits and bounded recovery without losing the last good state."""
from pathlib import Path
import json
import os
import torch


def sync_directory(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_directory(path.parent)


def save_checkpoint(path,state):
    """A crash during save leaves latest/previous untouched; previous is a hard link."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    if path.name=='latest.pt' and path.exists():
        previous=path.with_name('previous.pt');link=path.with_name('previous.pt.partial')
        link.unlink(missing_ok=True);os.link(path,link);os.replace(link,previous)
    os.replace(tmp,path);sync_directory(path.parent)
    atomic_json(path.with_name(path.name+'.json'),dict(step=state['step'],epoch=state['epoch'],cursor=state['cursor'],
                total=state['total'],bytes=path.stat().st_size))


def recover_checkpoint(out,config,source_hash,total):
    """Only corruption permits fallback; configuration mismatches never restart from scratch."""
    errors=[]
    for name in ('latest.pt','previous.pt','initial.pt'):
        path=Path(out)/name
        if not path.exists():continue
        try:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            required=('config','model','opt','step','epoch','cursor','centres','stats','source_hash','total','rng')
            if any(k not in ck for k in required):raise ValueError('Incomplete checkpoint structure')
        except Exception as exc:
            errors.append(dict(path=str(path),error=repr(exc)));continue
        if ck['config']!=config or ck['source_hash']!=source_hash or ck['total']!=total:
            raise ValueError(f'Resume identity/schedule mismatch: {path}')
        return path,ck,errors
    raise RuntimeError(f'No valid checkpoint; refusing to reset training: {errors}')


def quarantine_corrupt_latest(out,errors):
    """Do not rotate a corrupt latest over the good fallback at the next save."""
    import time
    for error in errors:
        path=Path(error['path'])
        if path.exists():os.replace(path,path.with_name(path.name+f'.corrupt.{time.time_ns()}'))
