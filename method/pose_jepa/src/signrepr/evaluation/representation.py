import numpy as np


def representation_diagnostics(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('Expected clip feature matrix')
    valid = np.isfinite(x).all(-1) & (np.linalg.norm(x, axis=-1) > 1e-12)
    x = x[valid]
    if len(x) < 2:
        return {'valid_clips': len(x), 'feature_std_mean': None, 'effective_rank': None, 'mean_interclip_cosine': None}
    centered = x-x.mean(0)
    # SVD avoids allocating a huge feature-by-feature covariance for small cohorts.
    eigen = np.linalg.svd(centered, compute_uv=False)**2/(len(x)-1)
    if eigen.sum() <= 1e-20:
        effective_rank = 0.
    else:
        prob = eigen[eigen > 0]/eigen.sum()
        effective_rank = float(np.exp(-(prob*np.log(prob)).sum()))
    normalized = x/np.linalg.norm(x, axis=-1, keepdims=True)
    # Sum over all off-diagonal pairs without allocating the pairwise matrix.
    cosine = (np.square(normalized.sum(0)).sum()-len(x))/(len(x)*(len(x)-1))
    return {'valid_clips': len(x), 'feature_std_mean': float(x.std(0, ddof=1).mean()),
            'effective_rank': effective_rank, 'maximum_empirical_rank': min(len(x)-1, x.shape[1]),
            'mean_interclip_cosine': float(cosine)}
