from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def rng_state(device=None):
    device=torch.device(device) if device is not None else None
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":None,
            "cuda_device_rng":torch.cuda.get_rng_state(device) if device is not None and device.type=='cuda' else None,
            "cuda_device_index":device.index if device is not None and device.type=='cuda' else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get('cuda_device_rng') is not None:
        torch.cuda.set_rng_state(state['cuda_device_rng'],state['cuda_device_index'])


def atomic_checkpoint(path, content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".")
    try:
        with os.fdopen(fd,"wb") as f:
            torch.save(content,f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_checkpoint(path):
    # Only for locally generated trusted training checkpoints containing RNG.
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("schema_version") != "signcoord_checkpoint_v1" or state.get("microstep_boundary") != 0:
        raise ValueError("unsupported checkpoint or partial accumulation boundary")
    return state
