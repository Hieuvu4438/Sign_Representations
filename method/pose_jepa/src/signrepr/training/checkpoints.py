"""Atomic checkpoints with exact RNG and per-rank data/mask state."""
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.distributed as dist
from .distributed import rank, world_size


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training(trainer, path, stream_state=None, metadata=None):
    local = {'rng': rng_state(), 'mask_rng': trainer.mask_rng.get_state(), 'stream': stream_state}
    states = [None]*world_size()
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank() == 0:
        state = {'schema_version': '1.0', 'config': trainer.config,
                 'system': trainer.system.state_dict(), 'teacher': trainer.teacher.state_dict(),
                 'criterion': trainer.criterion.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                 'scaler': trainer.scaler.state_dict(), 'step': trainer.step,
                 'scheduler': {'successful_updates': trainer.step}, 'rank_states': states,
                 'world_size': world_size(), 'metadata': metadata or {}}
        if hasattr(trainer.system, 'metric_head'):
            state['metric_head'] = trainer.system.metric_head.state_dict()
            state['vocabulary'] = trainer.vocabulary
            state['phase'] = trainer.phase
        atomic_save(state, path)
    if dist.is_initialized():
        dist.barrier()


def load_training(trainer, path):
    # Training .pt files contain optimizer and Python RNG state; trusted artifacts only.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config'] != trainer.config:
        raise ValueError('Resume requires identical scientific and run configuration')
    if state['world_size'] != world_size():
        raise ValueError('Exact resume requires the same distributed topology')
    if hasattr(trainer, 'vocabulary') and state.get('vocabulary') != trainer.vocabulary:
        raise ValueError('Resume train-vocabulary mapping mismatch')
    trainer.system.load_state_dict(state['system'], strict=True)
    trainer.teacher.load_state_dict(state['teacher'], strict=True)
    trainer.criterion.load_state_dict(state['criterion'], strict=True)
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.scaler.load_state_dict(state['scaler'])
    trainer.step = state['step']
    local = state['rank_states'][rank()]
    trainer.mask_rng.set_state(local['mask_rng'])
    restore_rng(local['rng'])
    trainer.teacher.eval()
    return local['stream'], state['metadata']
