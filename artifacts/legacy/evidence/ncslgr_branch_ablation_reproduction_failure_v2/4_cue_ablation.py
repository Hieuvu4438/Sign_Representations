"""Fixed cue-zeroing diagnostics; interventions do not alter observation masks."""
import numpy as np

STREAMS = ('face', 'left_hand', 'right_hand', 'body_posture')
CONDITIONS = {
    'all': STREAMS,
    'without_face': STREAMS[1:],
    'without_hands': ('face', 'body_posture'),
    'without_body': STREAMS[:3],
    'face_only': ('face',),
    'hands_only': ('left_hand', 'right_hand'),
    'body_only': ('body_posture',),
    'zero_streams': (),
}


def intervene(streams, condition):
    if condition not in CONDITIONS or set(streams) != set(STREAMS):
        raise ValueError('Unknown condition or incompatible cue streams')
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in streams.items()}
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in arrays.values()) or len({len(v) for v in arrays.values()}) != 1:
        raise ValueError('Invalid cue values or source alignment')
    return {k: v.copy() if k in CONDITIONS[condition] else np.zeros_like(v) for k, v in arrays.items()}


def pool_conditions(embeddings, observed, centers, start_sec, end_sec):
    """Identical observed source tokens across all interventions; no gold core."""
    embeddings, observed, centers = np.asarray(embeddings), np.asarray(observed), np.asarray(centers)
    if embeddings.ndim != 3 or embeddings.shape[0] != len(CONDITIONS) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid condition feature dimensions/values')
    n = embeddings.shape[1]
    if observed.dtype != bool or observed.shape != (n, 4) or centers.shape != (n,) or not np.isfinite(centers).all() or not (np.diff(centers)>0).all():
        raise ValueError('Invalid original observation mask or clock')
    if not np.isfinite([start_sec, end_sec]).all() or not start_sec < end_sec:
        raise ValueError('Invalid known utterance support')
    keep = observed.all(1) & (centers >= start_sec) & (centers < end_sec)
    if not keep.any():
        raise ValueError('Locked common population lost valid utterance-supported tokens')
    pooled = embeddings[:, keep].mean(1)
    norms = np.linalg.norm(pooled, axis=1)
    if (norms <= 1e-10).any():
        raise ValueError('Zero pooled frozen embedding')
    return (pooled / norms[:, None]).astype(np.float32), keep
