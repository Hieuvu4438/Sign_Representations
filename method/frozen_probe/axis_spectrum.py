"""Fixed numerical factorial controls; not a new whitening method."""
import numpy as np

MODES = ('RR', 'TR', 'RT', 'TT')  # first letter: directions; second: spectrum


def factor(covariance, gap_floor=1e-12, atol=1e-10):
    c = np.asarray(covariance, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all() or not np.allclose(c, c.T, atol=1e-12, rtol=0):
        raise ValueError('Invalid covariance')
    eigenvalues, basis = np.linalg.eigh(c)
    if eigenvalues[0] <= 1e-12:
        raise ValueError('Nonpositive eigenvalue; no floor or fallback')
    gap = float(np.min(np.diff(eigenvalues)) / eigenvalues[-1]) if len(eigenvalues) > 1 else 1.
    if gap <= gap_floor:
        raise ValueError('Nonidentifiable/near-degenerate eigenvector directions')
    orthogonal = float(np.max(np.abs(basis.T @ basis - np.eye(len(c)))))
    reconstruction = float(np.max(np.abs((basis * eigenvalues) @ basis.T - c)) / np.max(np.abs(c)))
    if max(orthogonal, reconstruction) > atol:
        raise ValueError('Eigensystem equation failed')
    return eigenvalues, basis, dict(minimum_relative_gap=gap, orthogonality_error=orthogonal,
                                   relative_reconstruction_error=reconstruction)


def transforms(bases, spectra):
    result = {}
    for mode in MODES:
        v, eigenvalues = bases[mode[0]], spectra[mode[1]]
        if v.shape != (len(eigenvalues), len(eigenvalues)) or not np.isfinite(v).all() or np.any(eigenvalues <= 1e-12):
            raise ValueError('Invalid matched-spectrum inputs')
        result[mode] = (v / np.sqrt(eigenvalues)) @ v.T
    return result


def unit_rows(values):
    x = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm <= 1e-12):
        raise ValueError('Invalid transformed vector; no zero fallback')
    return x / norm
