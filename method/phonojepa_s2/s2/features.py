"""Channel features, intrinsic targets and teacher reliability (torch, GPU, batched) — proposal12 v2.1 §6.4 / §6.7.

Input per view: xy [B, T, J, 2] source pixels (0 where unobserved), obs [B, T, J] bool, conf [B, T, J], jh [B, T, J]
(joint-hidden flags; student R1), valid_f [B, T] (frame inside the clip).
Body frame: rotate by the clip-median shoulder angle around the clip-median shoulder centre, unit = median shoulder
width (anchors only -> leak-free). Per-clip statistics (bone p90, neutral face, median log-scales) are computed over
the frames that are observed IN THIS VIEW (scrubbed / hidden frames of the student are unobserved by construction).

Per frame (level features, without conf/flags) and token dims (2 frames per bin, with conf/flags):
  CONF 43 = u(2) sign(1) logscale(1) foreshortening(1) palm polygon(8) finger bones(30)   token 2 x (43+21+21) = 170
  LOC   6 = wrist(2) elbow(2) within-bin wrist velocity(2)                                token 2 x (6+2)       = 16
  UP   36 = 18 brow/eyelid points in the eye frame minus neutral                          token 2 x (36+18)     = 108
  MO   42 = 21 mouth+chin points in the eye frame minus neutral                           token 2 x (42+21)     = 126
  HD    7 = eye-centre (2) + log IOD (1) minus median, roll (2), nose tip in eye frame minus median (2)  token 2 x (7+3) = 20
Intrinsic targets per bin: [3-bin smoothed level, delta of the smoothed level] (HD: delta only) -> 86 / 12 / 72 / 84 / 7, z-scored.
"""
from __future__ import annotations

import numpy as np
import torch

from s2.joints import (BROWS, CHIN, EYE_CORNERS, EYELIDS, FINGER_BONES, LELB, LH0, LSHO, LWRI, MOUTH, NC, NOSE,
                       NOSE_TIP, NOSE_B, LHIP, RHIP, TORSO, OWNED, PALM_POLY, RELB, RH0, RSHO, RWRI)

LEVEL_DIMS = (43, 6, 43, 6, 36, 42, 7, 20)
EXTRA_DIMS = (42, 2, 42, 2, 18, 21, 3, 5)
TOKEN_DIMS = tuple(2 * (a + b) for a, b in zip(LEVEL_DIMS, EXTRA_DIMS))        # 170 16 170 16 108 126 20
TARGET_DIMS = (86, 12, 86, 12, 72, 84, 7, 40)
TYPE_TOKEN_DIMS = (170, 16, 108, 126, 20, 50)                                        # CONF, LOC, UP, MO, HD
TYPE_TARGET_DIMS = (86, 12, 72, 84, 7, 40)
UP_PTS = BROWS + EYELIDS
MO_PTS = MOUTH + [CHIN]


def _nanmedian(x, dim):
    m = torch.nanmedian(x, dim=dim).values
    return torch.nan_to_num(m, nan=0.0)


def _nanq(x, q, dim, fill):
    v = torch.nanquantile(x, q, dim=dim)
    return torch.nan_to_num(v, nan=fill)


def anchors(xy, obs):
    """clip-median shoulder centre c [B,2], width s [B], angle a [B] (fallbacks: origin / 100 pixels / 0)."""
    L, R = xy[:, :, LSHO], xy[:, :, RSHO]
    ok = obs[:, :, LSHO] & obs[:, :, RSHO]
    nan = torch.full_like(L, float("nan"))
    mid = torch.where(ok[..., None], (L + R) / 2, nan)
    v = torch.where(ok[..., None], L - R, nan)
    c = _nanmedian(mid, 1)
    w = torch.nan_to_num(torch.nanmedian(v.norm(dim=-1), dim=1).values, nan=0.0)
    vx, vy = _nanmedian(v[..., 0], 1), _nanmedian(v[..., 1], 1)
    a = torch.atan2(vy, vx)
    bad = ~ok.any(1)
    if bad.any():                                                                  # no shoulders at all
        c = torch.where(bad[:, None], torch.zeros_like(c), c)
        w = torch.where(bad, torch.ones_like(w) * 100.0, w)
        a = torch.where(bad, torch.zeros_like(a), a)
    return c, w.clamp(min=1e-3), a


def to_body_frame(xy, c, s, a, aff=None):
    """explicit fp32 multiply-adds (no matmul -> unaffected by autocast). aff [B,2,2] optional augmentation applied in
    the body frame (rotation / scale / shear around the shoulder centre; content-independent)."""
    xy = xy.float(); c = c.float(); s = s.float(); a = a.float()
    ca, sa = torch.cos(-a)[:, None, None], torch.sin(-a)[:, None, None]
    dx, dy = xy[..., 0] - c[:, None, None, 0], xy[..., 1] - c[:, None, None, 1]
    x, y = (ca * dx - sa * dy) / s[:, None, None], (sa * dx + ca * dy) / s[:, None, None]
    if aff is not None:
        A = aff.float()
        x, y = A[:, None, None, 0, 0] * x + A[:, None, None, 0, 1] * y, A[:, None, None, 1, 0] * x + A[:, None, None, 1, 1] * y
    return torch.stack([x, y], -1)


def _hand(p, o, conf, jh, base):
    """p [B,T,J,2] body frame; returns level [B,T,43], extras [B,T,42], frame-valid [B,T]."""
    h, ho = p[:, :, base:base + 21], o[:, :, base:base + 21]
    fv = ho[..., 0] & ho[..., 9]
    v = h[..., 9, :] - h[..., 0, :]
    L = v.norm(dim=-1)
    Le = L.clamp(min=0.1)
    u = v / L.clamp(min=1e-6)[..., None]
    P = (h - h[..., :1, :]) / Le[..., None, None]
    a5, a17 = h[..., 5, :] - h[..., 0, :], h[..., 17, :] - h[..., 0, :]
    sign = torch.sign(a5[..., 0] * a17[..., 1] - a5[..., 1] * a17[..., 0]) * (ho[..., 5] & ho[..., 17] & fv)
    logs = torch.log(Le)
    logs_n = torch.where(fv, logs, torch.full_like(logs, float("nan")))
    logs = logs - _nanmedian(logs_n, 1)[:, None]
    fore = 1.0 - L / Le
    poly_ok = ho[..., PALM_POLY] & fv[..., None]
    poly = torch.where(poly_ok[..., None], P[..., PALM_POLY, :], torch.zeros_like(P[..., PALM_POLY, :])).flatten(-2)
    pa, ch = [a for a, _ in FINGER_BONES], [b for _, b in FINGER_BONES]
    bones = P[..., ch, :] - P[..., pa, :]                                           # [B,T,15,2]
    bok = ho[..., ch] & ho[..., pa] & fv[..., None]
    bl = torch.where(bok, bones.norm(dim=-1), torch.full_like(bones[..., 0], float("nan")))
    p90 = _nanq(bl, 0.9, 1, 1.0).clamp(min=1e-3)                                     # [B,15]
    bones = torch.where(bok[..., None], bones / p90[:, None, :, None], torch.zeros_like(bones)).flatten(-2)
    z = torch.zeros_like(L)
    lev = torch.cat([u * fv[..., None], sign[..., None], torch.where(fv, logs, z)[..., None],
                     torch.where(fv, fore, z)[..., None], poly, bones], -1)
    ext = torch.cat([conf[:, :, base:base + 21], jh[:, :, base:base + 21].float()], -1)
    return torch.nan_to_num(lev).clamp(-6, 6), ext, fv


def _loc(p, o, conf, wr, el):
    w, e = p[:, :, wr], p[:, :, el]
    fv = o[:, :, wr] & o[:, :, el]
    B, T = fv.shape
    wb = w.view(B, T // 2, 2, 2); okb = o[:, :, wr].view(B, T // 2, 2)
    d = (wb[:, :, 1] - wb[:, :, 0]) * (okb[:, :, 0] & okb[:, :, 1])[..., None]     # within-bin velocity (no cross-bin stencil)
    vel = d[:, :, None].expand(B, T // 2, 2, 2).reshape(B, T, 2)
    lev = torch.cat([w, e, vel], -1) * fv[..., None]
    ext = torch.stack([conf[:, :, wr], conf[:, :, el]], -1)
    return torch.nan_to_num(lev).clamp(-6, 6), ext, fv


def _face(p, o, conf, c_unused=None):
    E = EYE_CORNERS
    fv = o[:, :, E].all(-1)
    eR, eL = p[:, :, E[:2]].mean(2), p[:, :, E[2:]].mean(2)
    org = (eR + eL) / 2
    ax = eL - eR
    iod = ax.norm(dim=-1).clamp(min=1e-4)
    ex = ax / iod[..., None]
    ey = torch.stack([-ex[..., 1], ex[..., 0]], -1)
    def eye_frame(pts):
        q = p[:, :, pts] - org[:, :, None]
        return torch.stack([(q * ex[:, :, None]).sum(-1), (q * ey[:, :, None]).sum(-1)], -1) / iod[..., None, None]
    nan = float("nan")
    def neutral(F, ok):
        Fn = torch.where(ok[..., None], F, torch.full_like(F, nan))
        return F - _nanmedian(Fn, 1)[:, None]
    up = eye_frame(UP_PTS); up_ok = o[:, :, UP_PTS] & fv[..., None]
    up = torch.where(up_ok[..., None], neutral(up, up_ok), torch.zeros_like(up)).flatten(-2)
    mo = eye_frame(MO_PTS); mo_ok = o[:, :, MO_PTS] & fv[..., None]
    mo = torch.where(mo_ok[..., None], neutral(mo, mo_ok), torch.zeros_like(mo)).flatten(-2)
    lsc = torch.log(iod)[..., None]
    nose = eye_frame([NOSE_TIP])[:, :, 0]
    rel = torch.cat([org, lsc, nose], -1)
    nose_ok = o[:, :, NOSE_TIP] & fv
    rel_ok = torch.cat([fv[..., None].expand(-1, -1, 3), nose_ok[..., None].expand(-1, -1, 2)], -1)
    reln = torch.where(rel_ok, rel, torch.full_like(rel, nan))
    rel = rel - _nanmedian(reln, 1)[:, None]
    rel = torch.where(fv[..., None], rel, torch.zeros_like(rel))
    rel[..., 3:5] = rel[..., 3:5] * nose_ok[..., None]
    hd = torch.cat([rel[..., 0:3], ex * fv[..., None], rel[..., 3:5]], -1)
    ext_up, ext_mo = conf[:, :, UP_PTS], conf[:, :, MO_PTS]
    ext_hd = torch.stack([conf[:, :, E].mean(-1), conf[:, :, NOSE_TIP], conf[:, :, NOSE].mean(-1)], -1)
    cl = lambda x: torch.nan_to_num(x).clamp(-6, 6)
    return ((cl(up), ext_up, up_ok.any(-1)), (cl(mo), ext_mo, mo_ok.any(-1)), (cl(hd), ext_hd, fv))


@torch.autocast("cuda", enabled=False)
@torch.autocast("cpu", enabled=False)
def frame_features(xy, obs, conf, jh, aff=None, anchor=None):
    """-> list over NC channels of (level [B,T,Dl], extras [B,T,De], frame-valid [B,T]). Always fp32 (autocast off)."""
    xy, conf = xy.float(), conf.float()
    xy = torch.where(obs[..., None], xy, torch.zeros_like(xy))
    c, s, a = anchors(xy.masked_fill(~obs[..., None], float("nan")), obs) if anchor is None else anchor
    p = to_body_frame(xy, c, s, a, aff)
    out = [None] * NC
    out[0] = _hand(p, obs, conf, jh, RH0)
    out[1] = _loc(p, obs, conf, RWRI, RELB)
    out[2] = _hand(p, obs, conf, jh, LH0)
    out[3] = _loc(p, obs, conf, LWRI, LELB)
    out[4], out[5], out[6] = _face(p, obs, conf)
    out[TORSO] = torso_features(p, obs, conf)
    return out


def tokens(feats, valid_f):
    """-> inputs list[NC] [B,Tb,token_dim], present [B,NC,Tb] (False = ABSENT: no valid frame in the bin)."""
    B, T = valid_f.shape
    ins, pres = [], []
    for lev, ext, fv in feats:
        fv = fv & valid_f
        x = torch.cat([lev, ext], -1) * fv[..., None]
        ins.append(x.reshape(B, T // 2, -1))
        pres.append(fv.view(B, T // 2, 2).any(-1))
    return ins, torch.stack(pres, 1)


def reliability(conf, obs, valid_f):
    """teacher-side reliability per (channel, bin): mean conf of the channel's owned joints over the bin's frames."""
    B, T, _ = conf.shape
    out = []
    for ch in range(NC):
        idx = OWNED[ch]
        cc = (conf[:, :, idx] * obs[:, :, idx]).mean(-1) * valid_f
        out.append(cc.view(B, T // 2, 2).mean(-1))
    return torch.stack(out, 1)


def intrinsic_targets(feats, valid_f):
    """per channel: y [B,Tb,Dt] and per-dim weight [B,Tb,Dt] (teacher view)."""
    B, T = valid_f.shape
    Tb = T // 2
    ys, ws = [], []
    for ch, (lev, _, fv) in enumerate(feats):
        fv = (fv & valid_f).float()
        lb = lev.view(B, Tb, 2, -1); fb = fv.view(B, Tb, 2, 1)
        cnt = fb.sum(2)
        mean = (lb * fb).sum(2) / cnt.clamp(min=1)                                   # bin means
        ok = (cnt[..., 0] > 0).float()                                               # [B,Tb]
        # 3-bin smoothing over valid bins (teacher side; leak-free)
        num = mean * ok[..., None]
        sm_num = num.clone(); sm_den = ok.clone()
        sm_num[:, 1:] += num[:, :-1]; sm_den[:, 1:] += ok[:, :-1]
        sm_num[:, :-1] += num[:, 1:]; sm_den[:, :-1] += ok[:, 1:]
        smooth = sm_num / sm_den.clamp(min=1)[..., None]
        delta = torch.zeros_like(mean); dok = torch.zeros_like(ok)
        delta[:, 1:] = smooth[:, 1:] - smooth[:, :-1]; dok[:, 1:] = ok[:, 1:] * ok[:, :-1]   # spec §6.7: delta of smoothed
        if ch == 6:                                                                  # HEAD: delta only
            y = delta; w = dok[..., None].expand_as(delta)
        else:
            y = torch.cat([smooth, delta], -1)
            w = torch.cat([ok[..., None].expand_as(smooth), dok[..., None].expand_as(delta)], -1)
        ys.append(y); ws.append(w)
    return ys, ws


class TargetStats:
    """per-channel mean/std of the intrinsic targets (fit on unmasked training windows)."""

    def __init__(self, path=None):
        self.m = self.s = None
        from pathlib import Path
        if path is not None and not Path(path).exists():
            import warnings
            warnings.warn(f"target statistics {path} missing: intrinsic targets are NOT z-scored")
        if path is not None and Path(path).exists():
            z = np.load(path)
            self.m = [torch.as_tensor(z[f"m{c}"]) for c in range(NC)]
            self.s = [torch.as_tensor(z[f"s{c}"]) for c in range(NC)]

    def __call__(self, ys):
        if self.m is None:
            return ys
        return [((y - m.to(y)) / s.to(y)).clamp(-8, 8) for y, m, s in zip(ys, self.m, self.s)]   # tiny face stds: bound outliers

    @staticmethod
    def fit(batches, path):
        acc = [[] for _ in range(NC)]
        for ys, ws in batches:
            for c in range(NC):
                y, w = ys[c].flatten(0, 1), ws[c].flatten(0, 1)
                acc[c].append(torch.where(w > 0, y, torch.full_like(y, float("nan"))).cpu())
        out = {}
        for c in range(NC):
            a = torch.cat(acc[c])
            m = torch.nan_to_num(torch.nanmean(a, 0))
            s = torch.sqrt(torch.nan_to_num(torch.nanmean((a - m) ** 2, 0), nan=1.0)).clamp(min=1e-3)
            out[f"m{c}"], out[f"s{c}"] = m.numpy(), s.numpy()
        np.savez(path, **out)


def torso_features(p, obs, conf):
    """Five point levels + within-bin displacements; no cross-bin student stencil.

    Shoulders are always-visible side information, NOT hidden torso targets.
    Nose/hips are scrubbed by the torso mask. Per-point validity zeros missing data.
    """
    ids = [NOSE_B, LSHO, RSHO, LHIP, RHIP]
    B, T = obs.shape[:2]
    ok = obs[:, :, ids]
    level = torch.where(ok[..., None], p[:, :, ids], 0.0)
    lp = level.reshape(B, T // 2, 2, 5, 2)
    op = ok.reshape(B, T // 2, 2, 5)
    delta = (lp[:, :, 1] - lp[:, :, 0]) * (op[:, :, 0] & op[:, :, 1])[..., None]
    delta = delta[:, :, None].expand(-1, -1, 2, -1, -1).reshape(B, T, 5, 2)
    lev = torch.cat([level.flatten(-2), delta.flatten(-2)], -1)
    # A token with only anchor shoulders present carries no new body evidence.
    fv = ok[:, :, [0, 3, 4]].any(-1)
    return lev.clamp(-6, 6), conf[:, :, ids] * ok, fv
