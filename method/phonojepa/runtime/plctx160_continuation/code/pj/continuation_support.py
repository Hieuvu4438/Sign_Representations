"""Durable state and deterministic post-fork samples for PLctx continuation."""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import numpy as np
import torch
from pj.data.dataset import PhonoWindows, collate


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda:file.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.flush(); os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent); os.close(fd)
    try:
        torch.save(state, tmp)
        with open(tmp, 'rb') as file: os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all(state['cuda'])


def continuation_lr(step, total, peak, start=3e-6, final=3e-6, warmup=500):
    if not 0 <= step < total or not 0 < warmup < total: raise ValueError('Invalid continuation LR step/budget')
    if step < warmup: return start+(peak-start)*(step+1)/warmup
    fraction = (step+1-warmup)/(total-warmup)
    return final+(peak-final)*(1+math.cos(math.pi*fraction))/2


class SeededWindows(PhonoWindows):
    def __getitem__(self, plan):
        index, seed = plan
        return self.item(index, np.random.default_rng(seed))


class ReadOnlyCache:
    """Byte-identical extracted NPZ arrays; workers mmap pages instead of copying GB."""
    def __init__(self, directory):
        self.directory=str(directory)
        self.size=len(np.load(Path(directory)/'offsets.npy',mmap_mode='r'))-1
        self.arrays=None
    def __len__(self):return self.size
    def __getstate__(self):return {'directory':self.directory,'size':self.size,'arrays':None}
    def clip(self,index):
        if self.arrays is None:
            self.arrays={key:np.load(Path(self.directory)/(key+'.npy'),mmap_mode='r') for key in ['xy','conf','offsets','fps']}
        z=self.arrays;a,b=z['offsets'][index:index+2]
        return z['xy'][a:b],z['conf'][a:b].astype(np.float32),float(z['fps'][index])


def worker_init(_):
    torch.set_num_threads(1)


class PlannedBatches:
    def __init__(self, stream): self.stream = stream
    def __iter__(self):
        while not self.stream.stopped:
            plan = self.stream.plan(); self.stream.pending.append(self.stream.current())
            yield plan


class SampleStream:
    """Drop shuffled epoch remainder; commit consumed cursor, not worker prefetch."""
    def __init__(self, dataset, batch=32, seed=160200640, workers=4):
        if len(dataset) < batch: raise ValueError('Dataset smaller than training batch')
        self.dataset, self.batch = dataset, batch
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor = 160, 0
        self.order = self.new_order(); self.pending = deque(); self.consumed = None
        self.stopped = False; self.iterator = None
        self.loader = torch.utils.data.DataLoader(dataset, batch_sampler=PlannedBatches(self), collate_fn=collate,
            num_workers=workers, worker_init_fn=worker_init, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0, multiprocessing_context='spawn' if workers else None,
            generator=torch.Generator().manual_seed(seed+1))

    def new_order(self):
        n = len(self.dataset)//self.batch*self.batch
        return torch.randperm(len(self.dataset), generator=self.generator)[:n].tolist()

    def plan(self):
        if self.cursor == len(self.order):
            self.order = self.new_order(); self.cursor = 0; self.epoch += 1
        indices = self.order[self.cursor:self.cursor+self.batch]; self.cursor += self.batch
        seeds = torch.randint(0, 2**31, (self.batch,), generator=self.generator).tolist()
        return list(zip(indices, seeds))

    def current(self):
        return {'epoch':self.epoch, 'cursor':self.cursor, 'order':self.order, 'rng':self.generator.get_state(),
                'batch':self.batch, 'size':len(self.dataset)}

    def next(self):
        if self.iterator is None: self.iterator = iter(self.loader)
        batch = next(self.iterator); self.consumed = self.pending.popleft(); return batch

    def state_dict(self): return self.consumed or self.current()

    def load_state_dict(self, state):
        if state['batch'] != self.batch or state['size'] != len(self.dataset): raise ValueError('Sampler topology mismatch')
        self.close(); self.epoch, self.cursor, self.order = state['epoch'], state['cursor'], state['order']
        self.generator.set_state(state['rng']); self.consumed = state; self.pending.clear()

    def close(self):
        if self.iterator is not None:
            self.stopped = True
            for _ in self.iterator: pass
            if self.loader.num_workers: self.iterator._shutdown_workers()
            self.loader._iterator = None; self.iterator = None; self.pending.clear(); self.stopped = False


def training_state(trainer, stream, metadata, run_config):
    m = trainer.model
    return {'student':m.student.state_dict(), 'teacher':m.teacher.state_dict(), 'predictor':m.predictor.state_dict(),
            'heads':m.heads.state_dict(), 'opt':trainer.opt.state_dict(), 'step':metadata['global_step'],
            'config':vars(trainer.a), 'center':trainer.cce.c, 'center_cls':trainer.cce_cls.c,
            'bal':trainer.bal.state_dict(), 'rng':torch.get_rng_state(),
            'continuation':{'schema':1, 'run_config':run_config, 'metadata':metadata,
                            'stream':stream.state_dict(), 'rng':rng_state()}}


def save_last(trainer, stream, metadata, config, root):
    root = Path(root); last = root/'last.pt'; previous = root/'previous.pt'; receipt = root/'checkpoint.json'
    if last.exists() and receipt.exists() and json.loads(receipt.read_text()).get('sha256') == sha256(last):
        tmp = root/'.previous.link'; tmp.unlink(missing_ok=True); os.link(last,tmp); os.replace(tmp,previous)
    atomic_torch(last, training_state(trainer,stream,metadata,config))
    atomic_json(receipt, {'step':metadata['global_step'], 'branch_step':metadata['branch_step'], 'sha256':sha256(last)})


def load_latest(trainer, stream, config, root):
    errors = []
    for name in ['last.pt','previous.pt']:
        path = Path(root)/name
        if not path.exists(): continue
        try: state = torch.load(path, map_location='cpu', weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            errors.append(f'{name}: {exc}'); continue
        record = state['continuation']
        if record['run_config'] != config: raise ValueError('Continuation config/provenance mismatch')
        metadata = record['metadata']; sample_state = record['stream']; rng = record['rng']; del state
        trainer.load(path); stream.load_state_dict(sample_state); restore_rng(rng)
        return metadata, {'checkpoint':str(path),'step':metadata['global_step'],'branch_step':metadata['branch_step'],'fallback_errors':errors}
    if errors: raise RuntimeError('No readable continuation checkpoint: '+'; '.join(errors))
    return None, None
