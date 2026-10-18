"""Deterministic sharded sample cursor, independent of global model/mask RNG."""
import torch
from collections import deque
from torch.utils.data import DataLoader
from ..data.collate import collate_pose


class PlannedBatchSampler:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        while not self.stream.stopped:
            plan = self.stream.plan()
            self.stream.pending.append(self.stream.current_state())
            yield plan


class SampleStream:
    def __init__(self, dataset, batch_size, seed, rank=0, world_size=1,
                 num_workers=0, pin_memory=False, persistent_workers=True):
        if len(dataset) < world_size:
            raise ValueError('Dataset must contain at least one sample per rank')
        self.dataset, self.batch_size = dataset, batch_size
        self.rank, self.world_size = rank, world_size
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch, self.cursor = 0, 0
        self.order = self.new_order()
        self.num_workers = num_workers
        self.pending = deque()
        self.consumed_state = None
        self.iterator = None
        self.stopped = False
        self.loader = DataLoader(dataset, batch_sampler=PlannedBatchSampler(self), collate_fn=collate_pose,
                                 num_workers=num_workers, pin_memory=pin_memory,
                                 persistent_workers=persistent_workers and num_workers > 0,
                                 multiprocessing_context='spawn' if num_workers > 0 else None,
                                 generator=torch.Generator().manual_seed(seed+500001))

    def new_order(self):
        order = torch.randperm(len(self.dataset), generator=self.generator)
        # Equal rank epoch lengths prevent ranks drifting into different permutations.
        # Drop a shuffled remainder rather than duplicate clips to pad a rank.
        usable = len(order)//self.world_size*self.world_size
        return order[:usable][self.rank::self.world_size].tolist()

    def plan(self):
        plan = []
        for _ in range(self.batch_size):
            if self.cursor == len(self.order):
                self.order = self.new_order()
                self.cursor = 0
                self.epoch += 1
            index = self.order[self.cursor]
            self.cursor += 1
            seed = int(torch.randint(0, 2**31, (), generator=self.generator))
            plan.append((index, seed))
        return plan

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        result = next(self.iterator)
        # Prefetch may have advanced the planning cursor. Checkpoint only what
        # the trainer actually consumed, never the queued worker cursor.
        self.consumed_state = self.pending.popleft()
        return result

    def current_state(self):
        return {'epoch': self.epoch, 'cursor': self.cursor, 'order': self.order, 'rng': self.generator.get_state(),
                'batch_size': self.batch_size, 'rank': self.rank, 'world_size': self.world_size, 'size': len(self.dataset)}

    def state_dict(self):
        return self.consumed_state or self.current_state()

    def load_state_dict(self, state):
        for k in ('batch_size', 'rank', 'world_size'):
            if state[k] != getattr(self, k):
                raise ValueError(f'Data stream topology mismatch: {k}')
        if state['size'] != len(self.dataset):
            raise ValueError('Data stream dataset size mismatch')
        self.epoch, self.cursor, self.order = state['epoch'], state['cursor'], state['order']
        self.generator.set_state(state['rng'])
        self.consumed_state = state
        self.pending.clear()
        self.close()

    def close(self):
        if self.iterator is not None:
            self.stopped = True
            # Drain already queued tensor transfers before workers exit. This
            # avoids a native IPC-thread abort during a prefetched shutdown.
            for _ in self.iterator:
                pass
            if self.num_workers > 0:
                self.iterator._shutdown_workers()
            # Drop the loader's persistent iterator so workers shut down before
            # restoring or abandoning the prefetched planning state.
            self.loader._iterator = None
            self.iterator = None
            self.pending.clear()
            self.stopped = False
