from collections import defaultdict
import torch


def gloss_key(row):
    if row.get('gloss_id') is None or not row.get('gloss_namespace') or not row.get('language'):
        raise ValueError('Metric labels require language, verified namespace and gloss ID')
    return (row['language'], row['gloss_namespace'], str(row['gloss_id']))


class PKSampler:
    """Same deterministic global P*K batch on every rank, then shard distinct clips."""
    def __init__(self, rows, p=40, k=2, seed=42, rank=0, world_size=1):
        self.rows = rows
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            if row['split'] not in ('train', 'pretrain'):
                raise ValueError('Metric sampler accepts train splits only')
            groups[gloss_key(row)].append(i)
        self.groups = {label: indices for label, indices in groups.items() if len({rows[i]['sample_id'] for i in indices}) >= k}
        self.excluded_classes = [label for label in groups if label not in self.groups]
        if not self.groups or k < 2:
            raise ValueError('SupCon requires at least two distinct clips per eligible class')
        self.labels = sorted(self.groups)
        self.p, self.k = min(p, len(self.labels)), k
        self.rank, self.world_size = rank, world_size
        if self.p*self.k % world_size:
            raise ValueError('Global contrastive batch must divide evenly across ranks')
        self.generator = torch.Generator().manual_seed(seed)

    def next(self):
        labels = torch.randperm(len(self.labels), generator=self.generator)[:self.p].tolist()
        batch = []
        for label in labels:
            pool = self.groups[self.labels[label]]
            ordered = [pool[i] for i in torch.randperm(len(pool), generator=self.generator).tolist()]
            chosen, signers, identities = [], set(), set()
            # Prefer verified distinct signers; unknown signers confer no such claim.
            for index in ordered:
                row = self.rows[index]
                signer = row.get('signer_id')
                if signer and signer not in signers and row['sample_id'] not in identities:
                    chosen.append(index); signers.add(signer); identities.add(row['sample_id'])
                    if len(chosen) == self.k: break
            for index in ordered:
                if len(chosen) == self.k: break
                if self.rows[index]['sample_id'] not in identities:
                    chosen.append(index); identities.add(self.rows[index]['sample_id'])
            batch.extend(chosen)
        return batch[self.rank::self.world_size]

    def state_dict(self):
        return {'rng': self.generator.get_state(), 'labels': self.labels, 'p': self.p, 'k': self.k,
                'rank': self.rank, 'world_size': self.world_size}

    def load_state_dict(self, state):
        for k in ('labels', 'p', 'k', 'rank', 'world_size'):
            if state[k] != getattr(self, k):
                raise ValueError(f'Metric sampler mismatch: {k}')
        self.generator.set_state(state['rng'])
