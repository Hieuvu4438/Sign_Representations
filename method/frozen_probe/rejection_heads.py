"""Declared frozen-feature adaptations of BSD FR-Disc and a linear control.

Author architecture: HSP-IIT/fsosar c8115c09, utils.py
BinaryClassificationModelSAFSAR. Original BSD notice/license retained in references.
Training/sampling here are explicitly adapted, not the full video method.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HEAD_NAMES = ('residual_linear', 'FRDisc_frozen_MLP')


class ResidualLinear(nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.fc = nn.Linear(dimension, 1)

    def forward(self, residual):
        return self.fc(residual).squeeze(-1)


class FRDiscMLP(nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 2 * dimension)
        self.fc2 = nn.Linear(2 * dimension, dimension)
        self.fc3 = nn.Linear(dimension, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, residual):
        x = F.relu(self.fc1(residual))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x).squeeze(-1)


def make_heads(seed, device='cpu'):
    torch.manual_seed(seed)
    return [ResidualLinear().to(device), FRDiscMLP().to(device)]


def selected_bce_indices(similarity, targets):
    if similarity.ndim != 2 or targets.shape != (len(similarity),):
        raise ValueError('Wrong episode score/target shape')
    if not torch.isfinite(similarity).all() or torch.any((targets < -1) | (targets >= similarity.shape[1])):
        raise ValueError('Invalid episode scores/targets')
    correct = similarity.argmax(dim=1) == targets
    known = torch.nonzero(correct, as_tuple=False).flatten()
    if len(known) == 0:
        return None
    unknown = torch.nonzero(targets == -1, as_tuple=False).flatten()[:len(known)]
    indices = torch.cat((unknown, known))
    return indices, (targets[indices] != -1).to(similarity.dtype)


def balanced_logit_loss(logits, similarity, targets):
    selected = selected_bce_indices(similarity, targets)
    if selected is None:
        return None
    indices, labels = selected
    return F.binary_cross_entropy_with_logits(logits[indices], labels.to(logits.dtype))


def training_plan(rows, steps=1000, size=50, queries=16, seed=1701):
    ids = [r['sample_id'] for r in rows]
    if ids != sorted(set(ids)) or len(rows) > np.iinfo(np.int16).max:
        raise ValueError('Unsorted/duplicate/oversized source roster')
    classes = sorted({r['class_id'] for r in rows})
    people = sorted({r['signer'] for r in rows})
    if type(steps) is not int or steps <= 0 or not 1 < size < len(classes) or queries <= 0:
        raise ValueError('Invalid fixed episode dimensions')
    signer_rows = {p: [i for i, r in enumerate(rows) if r['signer'] == p] for p in people}
    support_pools = {}
    for person, indices in signer_rows.items():
        if len({rows[i]['class_id'] for i in indices}) < 2:
            raise ValueError('Signer has fewer than two query classes: ' + person)
        source_groups = {rows[i]['source_group'] for i in indices}
        support_pools[person] = {}
        for cls in classes:
            pool = [i for i, r in enumerate(rows) if r['class_id'] == cls
                    and r['signer'] != person and r['source_group'] not in source_groups]
            if not pool:
                raise ValueError('Empty signer/class support pool: ' + str((person, cls)))
            support_pools[person][cls] = pool
    rng = np.random.default_rng(seed)
    plan = np.empty((steps, size + 2 * queries), dtype=np.int16)
    for step in range(steps):
        person = people[int(rng.integers(len(people)))]
        qclasses = sorted({rows[i]['class_id'] for i in signer_rows[person]})
        known_anchor, unknown_anchor = rng.choice(qclasses, size=2, replace=False).tolist()
        remaining = [c for c in classes if c not in (known_anchor, unknown_anchor)]
        gallery = sorted([known_anchor] + rng.choice(remaining, size=size - 1, replace=False).tolist())
        support = [int(rng.choice(support_pools[person][cls])) for cls in gallery]
        known = [i for i in signer_rows[person] if rows[i]['class_id'] in gallery]
        unknown = [i for i in signer_rows[person] if rows[i]['class_id'] not in gallery]
        assert known and unknown
        plan[step] = support + rng.choice(known, size=queries, replace=True).tolist() + rng.choice(unknown, size=queries, replace=True).tolist()
    return plan


def validate_training_plan(rows, plan, size=50, queries=16):
    if plan.dtype != np.int16 or plan.ndim != 2 or plan.shape[1] != size + 2 * queries:
        raise ValueError('Wrong plan schema')
    if np.any(plan < 0) or np.any(plan >= len(rows)):
        raise ValueError('Invalid training index')
    for episode in plan:
        support = [rows[int(i)] for i in episode[:size]]
        known = [rows[int(i)] for i in episode[size:size + queries]]
        unknown = [rows[int(i)] for i in episode[size + queries:]]
        gallery = [r['class_id'] for r in support]
        if gallery != sorted(set(gallery)) or len(gallery) != size:
            raise ValueError('Duplicate/unordered support class')
        if any(r['class_id'] not in gallery for r in known) or any(r['class_id'] in gallery for r in unknown):
            raise ValueError('Query membership differs from declared role')
        query_signers = {r['signer'] for r in known + unknown}
        if len(query_signers) != 1 or query_signers & {r['signer'] for r in support}:
            raise ValueError('Query/support signer contract differs')
        if {r['source_group'] for r in support} & {r['source_group'] for r in known + unknown}:
            raise ValueError('Source-byte overlap in episode')
    return True


def episode_inputs(vectors, labels, episode, size=50):
    support, query = episode[:size], episode[size:]
    gallery = labels[support]
    assert np.array_equal(gallery, np.unique(gallery))
    similarity = np.clip(vectors[query] @ vectors[support].T, -1, 1)
    prediction = similarity.argmax(axis=1)
    residual = vectors[query].astype(np.float32) - vectors[support[prediction]].astype(np.float32)
    target = np.asarray([int(np.flatnonzero(gallery == cls)[0]) if cls in gallery else -1
                         for cls in labels[query]], dtype=np.int64)
    return residual, similarity, target


def flat_parameters(heads):
    schema, pieces, offset = [], [], 0
    for name, head in zip(HEAD_NAMES, heads):
        for field, value in head.state_dict().items():
            a = value.detach().cpu().numpy().astype(np.float32, copy=True)
            schema.append(dict(head=name, field=field, shape=list(a.shape), start=offset, stop=offset + a.size))
            pieces.append(a.reshape(-1)); offset += a.size
    return np.concatenate(pieces), schema
