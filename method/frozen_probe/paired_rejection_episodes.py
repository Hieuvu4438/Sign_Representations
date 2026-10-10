"""Prospective same-query paired dictionaries, retaining single-class signers."""
import numpy as np


def paired_plan(rows, steps=1000, pairs=16, size=50, seed=1701):
    ids = [r['sample_id'] for r in rows]
    classes = sorted({r['class_id'] for r in rows})
    people = sorted({r['signer'] for r in rows})
    if ids != sorted(set(ids)) or len(rows) > 32767 or not 1 < size < len(classes) or steps <= 0 or pairs <= 0:
        raise ValueError('Invalid fixed paired roster/dimensions')
    signer_rows = {p: [i for i, r in enumerate(rows) if r['signer'] == p] for p in people}
    candidate = {}
    for person in people:
        candidate[person] = {}
        for cls in classes:
            pool = [i for i, r in enumerate(rows) if r['class_id'] == cls and r['signer'] != person]
            if not pool: raise ValueError('Empty source signer/class support pool: ' + str((person, cls)))
            candidate[person][cls] = pool
    rng = np.random.default_rng(seed); plan = np.empty((steps, pairs, size + 2), dtype=np.int16)
    for step in range(steps):
        for pair in range(pairs):
            person = people[int(rng.integers(len(people)))]; query = int(rng.choice(signer_rows[person]))
            true = rows[query]['class_id']; group = rows[query]['source_group']
            others = rng.choice([c for c in classes if c != true], size=size, replace=False).tolist()
            supports = []
            for cls in [true] + others:
                pool = [i for i in candidate[person][cls] if rows[i]['source_group'] != group]
                if not pool: raise ValueError('Empty source-disjoint paired support pool')
                supports.append(int(rng.choice(pool)))
            plan[step, pair] = [query] + supports
    return plan


def validate_paired_plan(rows, plan, size=50):
    if plan.dtype != np.int16 or plan.ndim != 3 or plan.shape[2] != size + 2 or np.any(plan < 0) or np.any(plan >= len(rows)):
        raise ValueError('Invalid paired index schema')
    for pair in plan.reshape(-1, size + 2):
        query = rows[int(pair[0])]; supports = [rows[int(i)] for i in pair[1:]]
        if supports[0]['class_id'] != query['class_id'] or len({r['class_id'] for r in supports}) != size + 1:
            raise ValueError('Duplicate class or wrong positive anchor')
        if any(r['signer'] == query['signer'] or r['source_group'] == query['source_group'] for r in supports):
            raise ValueError('Paired signer/source overlap')
        # Shared supports are literally the same IDs in both dictionaries.
        positive = [supports[0]] + supports[2:]; negative = supports[1:]
        if len(positive) != size or len(negative) != size or any(r['class_id'] == query['class_id'] for r in negative):
            raise ValueError('Invalid paired memberships')
    return True


def paired_inputs(vectors, labels, episode):
    if episode.ndim != 2 or episode.shape[1] < 4:
        raise ValueError('Invalid paired episode schema')
    query = episode[:, 0]
    positive = np.concatenate((episode[:, 1:2], episode[:, 3:]), axis=1)
    negative = episode[:, 2:]
    positive = np.take_along_axis(positive, np.argsort(labels[positive], axis=1, kind='stable'), axis=1)
    negative = np.take_along_axis(negative, np.argsort(labels[negative], axis=1, kind='stable'), axis=1)
    q = vectors[query]
    cp = np.clip(np.einsum('id,ijd->ij', q, vectors[positive]), -1, 1)
    cn = np.clip(np.einsum('id,ijd->ij', q, vectors[negative]), -1, 1)
    pp, pn = cp.argmax(axis=1), cn.argmax(axis=1)
    rp = q.astype(np.float32) - vectors[positive[np.arange(len(query)), pp]].astype(np.float32)
    rn = q.astype(np.float32) - vectors[negative[np.arange(len(query)), pn]].astype(np.float32)
    target = (labels[positive] == labels[query, None]).argmax(axis=1)
    assert np.all(np.any(labels[positive] == labels[query, None], axis=1))
    correct = pp == target
    order = np.concatenate((np.flatnonzero(correct), np.flatnonzero(~correct)))
    return np.concatenate((rp, rn[order])), np.concatenate((cp, cn[order])), np.concatenate((target, np.full(len(query), -1, dtype=np.int64)))
