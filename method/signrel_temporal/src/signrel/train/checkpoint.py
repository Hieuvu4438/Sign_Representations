import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch


def tensor_hash(state):
    h = hashlib.sha256()
    for name,value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_states():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch_cpu'].cpu())
    if states['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in states['torch_cuda']])


def atomic_checkpoint(path,payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temp)
    os.replace(temp,path)


def load_checkpoint(path,contract):
    # Only owned project checkpoint paths enter this full RNG/optimizer loader.
    payload = torch.load(path,map_location='cpu',weights_only=False)
    for key,value in contract.items():
        if payload.get(key)!=value:
            raise ValueError(f'Resume contract mismatch: {key}')
    return payload
