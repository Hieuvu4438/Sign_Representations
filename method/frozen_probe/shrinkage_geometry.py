"""Base-only standard Ledoit-Wolf geometry; not a new projection method."""
import numpy as np
from sklearn.covariance import LedoitWolf
from geometry_transfer import class_residuals
from probes import normalize_rows

FAMILIES = ('residual', 'total', 'shuffled')
NEW_MODES = tuple(f + '_lw_' + metric for f in FAMILIES for metric in ('cos', 'maha'))


def whitening(covariance):
    cov = np.asarray(covariance, dtype=np.float64)
    if (cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, atol=1e-12, rtol=0)):
        raise ValueError('Invalid covariance')
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues[0] <= 1e-12:
        raise ValueError('Nonpositive/degenerate regularized covariance; no floor fallback')
    transform = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return transform, eigenvalues


def fit_shrinkage(features, labels, seed):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(labels) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid base training features')
    shuffled = np.random.default_rng(seed).permutation(labels).tolist()
    mean = x.mean(axis=0)
    residuals = {'residual': class_residuals(x, labels), 'total': x - mean,
                 'shuffled': class_residuals(x, shuffled)}
    fitted = {'mean': mean}
    metadata = {'shuffled_labels': shuffled, 'families': {}}
    for family, r in residuals.items():
        model = LedoitWolf(assume_centered=True, store_precision=False).fit(r)
        transform, eigenvalues = whitening(model.covariance_)
        fitted[family + '_covariance'] = model.covariance_
        fitted[family + '_whitening'] = transform
        fitted[family + '_eigenvalues'] = eigenvalues
        metadata['families'][family] = {'shrinkage': float(model.shrinkage_),
            'sample_count': len(r), 'covariance_denominator': len(r),
            'mean_eigenvalue': float(np.trace(model.covariance_) / x.shape[1]),
            'minimum_eigenvalue': float(eigenvalues[0])}
    return fitted, metadata


def transformed(features, fitted, mode):
    if mode not in NEW_MODES:
        raise ValueError('Unknown shrinkage mode')
    family, _, metric = mode.split('_')
    x = (np.asarray(features, dtype=np.float64) - fitted['mean']) @ fitted[family + '_whitening']
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed features')
    return normalize_rows(x) if metric == 'cos' else x


def lookup_scores(queries, gallery, mode):
    q = np.asarray(queries, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError('Invalid lookup features')
    if mode not in NEW_MODES:
        raise ValueError('Unknown lookup mode')
    if mode.endswith('_cos'):
        return q @ g.T
    # Common covariance Mahalanobis distance after whitening, without L2 scaling.
    return -(np.sum(q * q, axis=1)[:, None] + np.sum(g * g, axis=1)[None, :] - 2 * q @ g.T)
