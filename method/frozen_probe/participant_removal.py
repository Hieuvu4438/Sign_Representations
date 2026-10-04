"""Known subspace-removal controls for H005 task-harm diagnostic; no novelty."""
import numpy as np
from probes import normalize_rows

MODES = ('raw', 'center', 'random', 'residual_pca', 'participant')


def leading_basis(matrix, rank):
    x = np.asarray(matrix, dtype=np.float64)
    if (x.ndim != 2 or not np.isfinite(x).all() or type(rank) is not int
            or not 0 < rank < min(x.shape)):
        raise ValueError('Invalid basis input/rank')
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    if singular[rank-1] <= 1e-12:
        raise ValueError('Insufficient effective rank; no silent rank replacement')
    basis = vt[:rank].T
    if not np.allclose(basis.T@basis, np.eye(rank), atol=1e-10, rtol=0):
        raise ValueError('Nonorthogonal basis')
    return basis, singular


def fit_removal(features, class_labels, people, rank, seed):
    x = np.asarray(features, dtype=np.float64)
    if (x.ndim != 2 or len(x) != len(class_labels) or len(x) != len(people)
            or not len(x) or not np.isfinite(x).all()):
        raise ValueError('Invalid fit coverage/features')
    residual = x.copy()
    for label in sorted(set(class_labels)):
        indices = [i for i,c in enumerate(class_labels) if c == label]
        residual[indices] -= x[indices].mean(axis=0)
    people_order = sorted(set(people))
    means = np.stack([residual[[i for i,p in enumerate(people) if p == person]].mean(axis=0)
                      for person in people_order])
    means -= means.mean(axis=0)
    participant, participant_s = leading_basis(means, rank)
    pca, pca_s = leading_basis(residual, rank)
    random, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1],rank)), mode='reduced')
    return dict(mean=x.mean(axis=0), participant=participant, residual_pca=pca, random=random,
                participant_singular=participant_s, residual_singular=pca_s,
                rank=rank, fit_count=len(x), people=people_order)


def transform(features, fitted, mode):
    if mode not in MODES:
        raise ValueError('Unknown transform')
    x = np.asarray(features, dtype=np.float64)
    if mode == 'raw':
        return normalize_rows(x)
    if x.ndim != 2 or x.shape[1] != len(fitted['mean']):
        raise ValueError('Transform dimension differs')
    value = x-fitted['mean']
    if mode != 'center':
        basis = fitted[mode]
        value -= (value@basis)@basis.T
    # Invalid/zero vectors are a technical failure; no sample exclusion.
    return normalize_rows(value)
