"""Step-aware deterministic batch sampler.

Microbatch m (global, counted from the start of the run) of rank r covers positions of epoch e = m // per_epoch
of a seeded permutation; every item carries its own seed = f(base_seed, m, position, index), so sampling,
augmentation and masks are reproducible after resume (state = number of microbatches consumed).
"""
from __future__ import annotations

import numpy as np

from asm_jepa.utils import sample_seed


class StepBatchSampler:
    def __init__(self, n_items: int, batch_size: int, base_seed: int, start_micro: int = 0, rank: int = 0,
                 world_size: int = 1, infinite: bool = True, shuffle: bool = True):
        if n_items < batch_size * world_size:
            raise ValueError(f"dataset of {n_items} items is smaller than one global batch "
                             f"({batch_size} x {world_size})")
        self.n, self.bs, self.seed = n_items, batch_size, base_seed
        self.start_micro, self.rank, self.ws = start_micro, rank, world_size
        self.infinite, self.shuffle = infinite, shuffle
        self.per_epoch = n_items // (batch_size * world_size)  # drop_last
        self._perm_cache = {}

    def _perm(self, epoch):
        if epoch not in self._perm_cache:
            self._perm_cache = {epoch: (np.random.default_rng([self.seed, epoch, 7]).permutation(self.n)
                                        if self.shuffle else np.arange(self.n))}
        return self._perm_cache[epoch]

    def batch(self, micro: int):
        epoch, k = divmod(micro, self.per_epoch)
        perm = self._perm(epoch)
        start = (k * self.ws + self.rank) * self.bs
        idx = perm[start:start + self.bs]
        return [(int(i), sample_seed(self.seed, micro, self.rank * self.bs + pos, int(i))) for pos, i in enumerate(idx)]

    def __iter__(self):
        m = self.start_micro
        stop = None if self.infinite else self.per_epoch * (1 + self.start_micro // self.per_epoch)
        while stop is None or m < stop:
            yield self.batch(m)
            m += 1

    def __len__(self):
        return self.per_epoch
