"""Fixed existing rejection scores and empirical source-known order statistic."""
from numbers import Integral, Real
import numpy as np

SCORE_NAMES = ('maxcosine', 'margin', 'maxsoftmax', 'NAC16')


def fixed_scores(logits, class_ids, nac_k=16, scale=1.):
    raw = np.asarray(logits)
    classes = np.asarray(class_ids)
    if raw.ndim != 2 or min(raw.shape) < 1 or raw.shape[1] < 2 or not np.issubdtype(raw.dtype, np.number) or not np.isrealobj(raw):
        raise ValueError('Require nonempty finite real cosine-logit matrix, at least two classes')
    if classes.ndim != 1 or len(classes) != raw.shape[1] or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < 0 for x in class_ids):
        raise ValueError('Invalid global gallery class IDs')
    if list(class_ids) != sorted(set(class_ids)):
        raise ValueError('Classes must be unique and ascending for deterministic ties')
    if isinstance(nac_k, (bool, np.bool_)) or not isinstance(nac_k, Integral) or nac_k < 2:
        raise ValueError('NAC neighborhood size must be integer >=2')
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid positive finite scale')
    x = raw.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite cosine logits')
    order = np.argsort(-x, axis=1, kind='stable')
    ranked = np.take_along_axis(x, order, axis=1)
    shifted = np.exp((ranked - ranked[:, :1]) * scale)
    k = min(int(nac_k), raw.shape[1])
    scores = np.stack((ranked[:, 0], ranked[:, 0] - ranked[:, 1],
                       1 / shifted.sum(axis=1), 1 / shifted[:, :k].sum(axis=1)), axis=1)
    ranking = classes[order].astype(np.int64)
    return scores, ranking[:, 0], ranking


def fit_source_known_threshold(scores, known_mask, numerator=19, denominator=20):
    raw, mask = np.asarray(scores), np.asarray(known_mask)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.number) or not np.isrealobj(raw):
        raise ValueError('Invalid calibration scores')
    if mask.shape != raw.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Known calibration mask must be complete Boolean with nonzero known count')
    if type(numerator) is not int or type(denominator) is not int or not 0 < numerator <= denominator:
        raise ValueError('Invalid positive target acceptance fraction')
    x = raw.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite calibration score')
    selected = np.sort(x[mask])[::-1]
    minimum_accept_count = (numerator * len(selected) + denominator - 1) // denominator
    threshold = float(selected[minimum_accept_count - 1])
    accepted = int((selected >= threshold).sum())
    return dict(threshold=threshold, known_count=len(selected), minimum_accept_count=minimum_accept_count,
                accepted_known_count=accepted, empirical_known_acceptance=accepted / len(selected),
                target_fraction=[numerator, denominator], tie_overshoot=accepted - minimum_accept_count,
                rule='Highest observed source-known threshold accepting at least ceil(num*Nknown/den); inclusive >=; no unknown or target fitting')
