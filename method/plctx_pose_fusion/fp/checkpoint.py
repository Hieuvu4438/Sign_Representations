"""Atomic generations; fallback for corrupt/unreadable bytes, never wrong identity."""
from pathlib import Path
import os,shutil
import torch
from fp.utils import atomic_torch


def save(root,state):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    latest=root/'last.pt';previous=root/'previous.pt'
    if latest.exists():
        temp=root/'previous.tmp'
        # Atomic old-generation hard link saves space and is safe across replacement.
        if temp.exists():temp.unlink()
        os.link(latest,temp);temp.replace(previous)
    atomic_torch(latest,state)


def load(root):
    root=Path(root);errors=[]
    for name in ('last.pt','previous.pt'):
        path=root/name
        if not path.exists():continue
        try:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if not all(k in ck for k in ('step','config','proof','model','progress')):raise ValueError('Checkpoint schema incomplete')
        except Exception as exc:errors.append(f'{name}: {type(exc).__name__}: {exc}');continue
        return ck,{'generation':name,'read_errors':errors}
    if errors:raise RuntimeError('No readable checkpoint; refusing fresh restart: '+'; '.join(errors))
    return None,{'generation':None,'read_errors':[]}
