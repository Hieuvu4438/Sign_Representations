"""Differentiable step-2 log signatures in the Lyndon (i<j) basis."""
from __future__ import annotations
import torch


def wedge(a, b):
    i, j = torch.triu_indices(a.shape[-1], a.shape[-1], 1, device=a.device)
    return a[..., i] * b[..., j] - a[..., j] * b[..., i]


def logsig2(x, moment=False):
    """[..., samples, d] -> [displacement, signed area / dimension-matched QV]."""
    dx = x[..., 1:, :] - x[..., :-1, :]
    a = dx.sum(-2)
    if moment:
        q = torch.einsum('...ti,...tj->...ij', dx, dx)
        i, j = torch.triu_indices(x.shape[-1], x.shape[-1], device=x.device)
        area = q[..., i, j][..., :x.shape[-1] * (x.shape[-1] - 1) // 2]
    else:
        prefix = x[..., :-1, :] - x[..., :1, :]
        area = 0.5 * wedge(prefix, dx).sum(-2)
    return torch.cat([a, area], -1)


def chen(logs, visible, d, moment=False):
    """Compose increments of visible segments without connecting endpoints across gaps.

    logs [..., K, d+d(d-1)/2]; visible [...,K]. Segments must share
    endpoints when adjacent. No bridging increment is ever inserted.
    """
    z = torch.where(visible[..., None], logs, torch.zeros_like(logs))
    a, area = z[..., :d], z[..., d:]
    prev = a.cumsum(-2) - a
    second=area.sum(-2) if moment else area.sum(-2)+0.5*wedge(prev,a).sum(-2)
    return torch.cat([a.sum(-2),second],-1)


def lead_lag(x):
    """Standard discrete lead-lag; output has 2N-1 points and 2d channels."""
    lead = x.repeat_interleave(2, -2)[..., 1:, :]
    lag = x.repeat_interleave(2, -2)[..., :-1, :]
    return torch.cat([lead, lag], -1)


def unwrap(angle):
    diff = angle[..., 1:] - angle[..., :-1]
    wrapped = (diff + torch.pi).remainder(2 * torch.pi) - torch.pi
    return torch.cat([angle[..., :1], angle[..., :1] + wrapped.cumsum(-1)], -1)


class RobustStats:
    """Frozen train-only median/IQR and per-path still thresholds, keyed by channel/scale."""
    def __init__(self, state=None, asinh=True):
        self.state = state or {}
        self.asinh = asinh

    def transform(self, key, x):
        if key not in self.state:
            raise ValueError(f'Missing train-only statistics: {key}; run pp.fit_stats first')
        s = self.state[key]
        z = (x - s['median'].to(x)) / s['iqr'].to(x)
        return z.asinh() if self.asinh else z

    @classmethod
    def fit(cls, batches, provenance):
        """batches: key -> list of (signature, energy, present, rest) tensors.

        Only present paths contribute. Rest is determined by public-body-frame
        dominant wrist height, without labels. Sampling is bounded by fit_stats.
        """
        state = {}
        for key, values in batches.items():
            x = torch.cat([v[0] for v in values]).float()
            e = torch.cat([v[1] for v in values]).float()
            ok = torch.cat([v[2] for v in values]).bool()
            rest = torch.cat([v[3] for v in values]).bool()
            coeff_ok = torch.cat([v[4] for v in values]).bool()
            masked = x.masked_fill(~coeff_ok, float('nan'))
            med = torch.nan_to_num(torch.nanquantile(masked, .5, dim=0))
            q = torch.nanquantile(masked, torch.tensor([.25,.75]), dim=0)
            iqr = torch.nan_to_num(q[1]-q[0], nan=1.).clamp_min(1e-4)
            eps = torch.nanquantile(e.masked_fill(~(ok & rest), float('nan')), .2, dim=0)
            # No resting observations: zero threshold, never invent a still distribution.
            state[key] = dict(median=med, iqr=iqr, eps=torch.nan_to_num(eps))
        obj = cls(state); obj.provenance = provenance
        return obj

    def save(self, path):
        from pathlib import Path
        import os
        from pp.recovery import sync_directory
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.partial')
        with tmp.open('wb') as f:
            torch.save(dict(state=self.state,provenance=self.provenance),f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);sync_directory(path.parent)

    @classmethod
    def load(cls, path, asinh=True):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        obj = cls(payload['state'], asinh); obj.provenance = payload['provenance']
        if obj.provenance.get('split') != 'train':
            raise ValueError('Statistics must be fit on train only')
        return obj
