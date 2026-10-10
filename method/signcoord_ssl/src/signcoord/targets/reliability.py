"""Finite missing values and bounded offline interpolation."""
from __future__ import annotations

import torch


def fill_short_gaps(xy, confidence, observed, times, track_ids=None, max_gap=2, min_conf=0.30,
                    reliability_factor=.5,gap_allowed=None):
    """Single source timeline [F,J,2]. Never interpolate through a track change.

    Returns finite coordinates, confidence, observed-or-interpolated mask and an
    interpolated mask. Low confidence observations remain separate from absence.
    """
    if xy.ndim != 3 or xy.shape[-1] != 2 or confidence.shape != xy.shape[:-1]:
        raise ValueError("expected xy[F,J,2], confidence/observed[F,J]")
    if times.shape != (len(xy),) or not torch.all(times[1:] > times[:-1]):
        raise ValueError("source timestamps must increase")
    if not 0<=reliability_factor<=1 or max_gap<0:raise ValueError('invalid source gap policy')
    valid = observed.bool() & torch.isfinite(xy).all(-1) & torch.isfinite(confidence)
    conf = torch.nan_to_num(confidence).clamp(0, 1) * valid
    out = torch.where(valid[..., None], xy, 0).clone()
    usable = valid.clone()
    interpolated = torch.zeros_like(valid)
    for j in range(xy.shape[1]):
        anchors = (valid[:, j] & (conf[:, j] >= min_conf)).nonzero().flatten().tolist()
        for a, b in zip(anchors, anchors[1:]):
            if not 1 <= b - a - 1 <= max_gap or valid[a+1:b, j].any():
                continue
            if gap_allowed is not None and not gap_allowed[a+1:b,j].all():continue
            if track_ids is not None and not torch.all(track_ids[a:b+1] == track_ids[a]):
                continue
            fraction = ((times[a+1:b] - times[a]) / (times[b] - times[a])).to(xy.dtype)
            out[a+1:b, j] = xy[a, j] + fraction[:, None] * (xy[b, j] - xy[a, j])
            conf[a+1:b, j] = reliability_factor * torch.minimum(conf[a, j], conf[b, j])
            usable[a+1:b, j] = True
            interpolated[a+1:b, j] = True
    return out, conf, usable, interpolated
