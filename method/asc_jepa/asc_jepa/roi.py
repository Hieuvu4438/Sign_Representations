"""ROI construction, ROI→tile weights, local states, observability gate and change pairs
(spec §6.3, §9). Adaptation (declared): ROIs/geometry come from the local COCO-WholeBody 2-D
keypoints (unknown estimator, unnormalised scores; validity = single person + calibrated score
threshold) instead of MediaPipe Tasks. Pose never enters the predictor; it is used only after
prediction, to pool states for the losses."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KP_DIR = Path("/home/dongvk/datasets/ASL_Citizen/keypoints")
PARTS = ("LH", "RH", "FACE", "BODY")
IDX = {"LH": np.arange(91, 112), "RH": np.arange(112, 133), "FACE": np.arange(23, 91),
       "BODY": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12])}
# 20%-quantile score thresholds calibrated on 2 held-out ASL Citizen train signers (CIP cache, EX03)
THR = {"LH": 1.692, "RH": 1.692, "FACE": 8.033, "BODY": 5.876}


def load_kp(sample_id):
    d = pickle.load(open(KP_DIR / f"{sample_id}.pkl", "rb"))
    T = len(d["keypoints"])
    K = np.full((T, 133, 2), np.nan, np.float32)
    Sc = np.zeros((T, 133), np.float32)
    for t, (k, s) in enumerate(zip(d["keypoints"], d["scores"])):
        k, s = np.asarray(k), np.asarray(s)
        if k.ndim == 3 and k.shape[0] == 1:
            K[t], Sc[t] = k[0], s[0]
    return K, Sc


def frame_boxes(K, Sc):
    """K [F,133,2] normalised xy, Sc [F,133] -> boxes [F,4,4] (x0,y0,x1,y1 in [0,1]) and valid [F,4]."""
    Fn = len(K)
    boxes = np.zeros((Fn, 4, 4), np.float32)
    valid = np.zeros((Fn, 4), bool)
    for p, name in enumerate(PARTS):
        pts, sc = K[:, IDX[name]], Sc[:, IDX[name]]
        ok = np.isfinite(pts).all(-1) & (sc >= THR[name] * 0.5)
        for f in range(Fn):
            if sc[f].mean() < THR[name] or ok[f].sum() < 3:
                continue
            q = pts[f][ok[f]]
            x0, y0 = q.min(0); x1, y1 = q.max(0)
            ext = max(x1 - x0, y1 - y0)
            mg = max(0.15 * ext, 0.02)
            boxes[f, p] = np.clip([x0 - mg, y0 - mg, x1 + mg, y1 + mg], 0, 1)
            valid[f, p] = (boxes[f, p, 2] > boxes[f, p, 0]) and (boxes[f, p, 3] > boxes[f, p, 1])
    return boxes, valid


def geometry_targets(K, Sc):
    """2-D geometry targets per frame (spec §6.4 subset): per hand 21 joints rel. wrist / palm span
    (42) + wrist rel. shoulder midpoint / shoulder width (2) + unit wrist→MCP9 (2) = 46; x2 hands = 92.
    Returns targets [F,92], mask [F,92]."""
    Fn = len(K)
    tg = np.zeros((Fn, 92), np.float32)
    mk = np.zeros((Fn, 92), bool)
    sh = (K[:, 5] + K[:, 6]) / 2
    sw = np.linalg.norm(K[:, 5] - K[:, 6], axis=1)
    for p, name in enumerate(("LH", "RH")):
        h = K[:, IDX[name]]
        sc = Sc[:, IDX[name]].mean(1)
        palm = np.linalg.norm(h[:, 5] - h[:, 17], axis=1)
        ok = np.isfinite(h).all((1, 2)) & (sc >= THR[name]) & (palm > 1e-3)
        o = p * 46
        rel = (h - h[:, :1]) / np.where(ok, palm, 1)[:, None, None]
        rel[..., 1] *= -1  # y-up convention
        tg[ok, o:o + 42] = rel[ok].reshape(-1, 42); mk[ok, o:o + 42] = True
        ok2 = ok & np.isfinite(sw) & (sw > 1e-3)
        loc = (h[:, 0] - sh) / np.where(ok2, sw, 1)[:, None]; loc[:, 1] *= -1
        tg[ok2, o + 42:o + 44] = loc[ok2]; mk[ok2, o + 42:o + 44] = True
        v = h[:, 9] - h[:, 0]; n = np.linalg.norm(v, axis=1)
        ok3 = ok & (n > 1e-4)
        u = v / np.where(ok3, n, 1)[:, None]; u[:, 1] *= -1
        tg[ok3, o + 44:o + 46] = u[ok3]; mk[ok3, o + 44:o + 46] = True
    return tg, mk


def roi_tile_weights(boxes, valid, T, H, W, frames_per_token=2):
    """boxes [B,F,P,4], valid [B,F,P] (torch) -> pi [B,P,T,H,W] = mean over the token's frames of
    intersection-area / tile-area (0 for invalid frames); support [B,P,T] = #valid frames."""
    B, Fn, P, _ = boxes.shape
    x0 = torch.arange(W, device=boxes.device).float() / W; x1 = x0 + 1.0 / W
    y0 = torch.arange(H, device=boxes.device).float() / H; y1 = y0 + 1.0 / H
    bx0, by0, bx1, by1 = [boxes[..., i] for i in range(4)]  # B,F,P
    ix = (torch.minimum(bx1[..., None], x1) - torch.maximum(bx0[..., None], x0)).clamp(min=0) * W  # B,F,P,W
    iy = (torch.minimum(by1[..., None], y1) - torch.maximum(by0[..., None], y0)).clamp(min=0) * H  # B,F,P,H
    pi_f = iy[..., :, None] * ix[..., None, :] * valid[..., None, None].float()  # B,F,P,H,W
    Fu = T * frames_per_token
    pi_f = pi_f[:, :Fu].reshape(B, T, frames_per_token, P, H, W)
    vf = valid[:, :Fu].reshape(B, T, frames_per_token, P).float()
    pi = pi_f.sum(2) / vf.sum(2).clamp(min=1)[..., None, None]
    return pi.permute(0, 2, 1, 3, 4).contiguous(), vf.sum(2).permute(0, 2, 1)  # B,P,T,H,W ; B,P,T


def pool_states(grid, pi, eps=1e-6):
    """grid [B,T,H,W,C], pi [B,P,T,H,W] -> LN_D(weighted mean) [B,P,T,C]."""
    num = torch.einsum("bpthw,bthwc->bptc", pi, grid.float())
    den = pi.sum((3, 4))[..., None]
    return F.layer_norm(num / (den + eps), (grid.shape[-1],))


def observability(pi, visible, radius_tokens=2, eps=1e-6):
    """A[b,p,t] = sum_{tau in window} sum_i pi*v / sum pi (spec §9.3); visible [B,T,H,W] bool."""
    num = torch.einsum("bpthw,bthw->bpt", pi, visible.float())
    den = pi.sum((3, 4))
    k = 2 * radius_tokens + 1
    ker = torch.ones(1, 1, k, device=pi.device)
    B, P, T = num.shape
    nw = F.conv1d(num.reshape(B * P, 1, T), ker, padding=radius_tokens).reshape(B, P, T)
    dw = F.conv1d(den.reshape(B * P, 1, T), ker, padding=radius_tokens).reshape(B, P, T)
    return nw / (dw + eps)


def change_pairs(token_ms, horizons=(80.0, 160.0), tol_frac=0.25, tol_min=10.0, T=8):
    """Token lags that realise each horizon at this clip's token spacing (spec §9.4); returns list of
    (lag_tokens, horizon) with duplicates removed (closest horizon kept)."""
    out = {}
    for h in horizons:
        lag = max(1, int(round(h / token_ms)))
        if lag < T and abs(lag * token_ms - h) <= max(tol_min, tol_frac * h):
            if lag not in out or abs(lag * token_ms - h) < abs(lag * token_ms - out[lag]):
                out[lag] = h
    return sorted(out.items())
