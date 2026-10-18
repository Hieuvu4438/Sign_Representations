"""Dependency-safe geometry features for proposal B (proposal11 §6.6, §8.2, §8.4, §11.5).

Rule: a student feature u is valid only if every raw observation it depends on, D(u), is visible
(observed AND not synthetically hidden). Otherwise the value is 0 and its validity flag is false.
All functions are numpy and operate on one window: P [T, J, 2] source pixels, vis/obs [T, J] bool,
quality [T, J], reason [T, J], timestamps [T] seconds.

Main node vector (19 slots per frame, §8.2):
  [0:2] body-relative xy (p - c) / s          D = {p} + body anchors (c, s estimated from VISIBLE shoulders)
  [2:4] hand-root-relative xy (p - p_root)/s  D = {p, root}            (hand nodes only)
  [4:6] parent-bone xy (p - p_parent)/s       D = {p, parent}
  [6:8] velocity xy (p_t - p_t-1)/(dt s)      D = {p_t, p_t-1}
  [8:12] validity of the four groups; [12] observed&visible; [13] q; [14] quality-known; [15:19] reason one-hot
"""
from __future__ import annotations

import numpy as np

from oj.data.landmarks import (BODY, BROWS, FACE, FACE_EYE_ANCHORS, HAND_CHAINS, J, LH, LSH, LWR, MOUTH, NOSE, PARENT,
                               REASON_MEASURED, REASON_MISSING, REASON_PADDING, REASON_SYNTHETIC, RH, RSH, RWR)

S_MIN = 1.0
NODE_SLOTS = 19
LOCAL_SLOTS = 8  # root-rel xy, parent-bone xy, 2 validity flags, observed, q
HAND_MCP = (5, 9, 13, 17)
FINGER_JOINT_TRIPLES = [(c[i - 1], c[i], c[i + 1]) for c in HAND_CHAINS for i in range(1, 4)]  # 15 bends per hand


def body_anchors(P, vis, img_wh):
    """robust clip estimate of shoulder midpoint c and distance s from VISIBLE shoulders only."""
    ok = vis[:, LSH] & vis[:, RSH]
    if ok.sum() >= 1:
        mid = (P[ok, LSH] + P[ok, RSH]) / 2
        dist = np.linalg.norm(P[ok, LSH] - P[ok, RSH], axis=-1)
        return np.median(mid, 0).astype(np.float32), float(max(np.median(dist), S_MIN)), True
    w, h = img_wh  # fixed image-scale fallback (flagged, not 'body-relative')
    return np.array([w / 2, h / 2], np.float32), float(max(w, h) / 4), False


def node_features(P, vis, obs, quality, reason, ts, anchors, quality_known=True):
    T = len(P)
    c, s, _ = anchors
    out = np.zeros((T, J, NODE_SLOTS), np.float32)
    Pz = np.where(vis[..., None], P, 0.0)
    # [0:2] body-relative
    out[..., 0:2] = np.where(vis[..., None], (Pz - c) / s, 0.0); out[..., 8] = vis
    # [2:4] hand-root-relative
    for part in (LH, RH):
        root = part[0]
        ok = vis[:, part] & vis[:, root:root + 1]
        out[:, part, 2:4] = np.where(ok[..., None], (Pz[:, part] - Pz[:, root:root + 1]) / s, 0.0)
        out[:, part, 9] = ok
    # [4:6] parent bone
    has = PARENT >= 0
    par = np.where(has, PARENT, 0)
    ok = vis & vis[:, par] & has[None]
    out[..., 4:6] = np.where(ok[..., None], (Pz - Pz[:, par]) / s, 0.0); out[..., 10] = ok
    # [6:8] velocity with real dt
    dt = np.diff(ts, prepend=ts[0] - 1.0 / 25)
    okv = np.zeros((T, J), bool); okv[1:] = vis[1:] & vis[:-1]
    v = np.zeros((T, J, 2), np.float32); v[1:] = (Pz[1:] - Pz[:-1]) / (np.maximum(dt[1:, None, None], 1e-3) * s)
    out[..., 6:8] = np.where(okv[..., None], v, 0.0); out[..., 11] = okv
    out[..., 12] = vis
    out[..., 13] = np.where(vis, quality, 0.0)
    out[..., 14] = float(quality_known)
    out[..., 15:19] = np.eye(4, dtype=np.float32)[reason]
    return out


def local_features(P, vis, quality):
    """per-frame local articulation inputs for hands (21 nodes) and face (18 nodes); scale/centre from the SAME
    2-frame bin only (hand: visible palm bones; face: visible eye corners). No velocity, no clip statistics."""
    T = len(P)
    out = {"lh": np.zeros((T, 21, LOCAL_SLOTS), np.float32), "rh": np.zeros((T, 21, LOCAL_SLOTS), np.float32),
           "face": np.zeros((T, 18, LOCAL_SLOTS), np.float32)}
    for b in range(T // 2):
        fr = slice(2 * b, 2 * b + 2)
        for name, part in (("lh", LH), ("rh", RH)):
            root = part[0]
            bones = [np.linalg.norm(P[t, part[k]] - P[t, root]) for t in range(2 * b, 2 * b + 2) for k in HAND_MCP
                     if vis[t, part[k]] and vis[t, root]]
            if not bones:
                continue
            sc = max(float(np.mean(bones)), S_MIN)
            for t in range(fr.start, fr.stop):
                okr = vis[t, part] & vis[t, root]
                out[name][t, :, 0:2] = np.where(okr[:, None], (P[t, part] - P[t, root]) / sc, 0.0)
                par = PARENT[part]; has = par >= 0
                okb = vis[t, part] & np.where(has, vis[t, np.where(has, par, 0)], False)
                out[name][t, :, 2:4] = np.where(okb[:, None], (P[t, part] - P[t, np.where(has, par, 0)]) / sc, 0.0)
                out[name][t, :, 4] = okr; out[name][t, :, 5] = okb; out[name][t, :, 6] = vis[t, part]
                out[name][t, :, 7] = np.where(vis[t, part], quality[t, part], 0.0)
        a, bb = FACE_EYE_ANCHORS
        okf = [t for t in range(2 * b, 2 * b + 2) if vis[t, a] and vis[t, bb]]
        if okf:
            cen = np.mean([(P[t, a] + P[t, bb]) / 2 for t in okf], 0)
            sc = max(float(np.mean([np.linalg.norm(P[t, a] - P[t, bb]) for t in okf])), S_MIN)
            for t in range(fr.start, fr.stop):
                okn = vis[t, FACE]
                out["face"][t, :, 0:2] = np.where(okn[:, None], (P[t, FACE] - cen) / sc, 0.0)
                par = PARENT[FACE]; has = par >= 0
                okb = okn & np.where(has, vis[t, np.where(has, par, 0)], False)
                out["face"][t, :, 2:4] = np.where(okb[:, None], (P[t, FACE] - P[t, np.where(has, par, 0)]) / sc, 0.0)
                out["face"][t, :, 4] = okn; out["face"][t, :, 5] = okb; out["face"][t, :, 6] = okn
                out["face"][t, :, 7] = np.where(okn, quality[t, FACE], 0.0)
    return out


def _bend(a, b, c):
    u, v = a - b, c - b
    nu = np.linalg.norm(u, axis=-1) + 1e-6; nv = np.linalg.norm(v, axis=-1) + 1e-6
    return np.stack([(u * v).sum(-1) / (nu * nv), (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) / (nu * nv)], -1)


ANCHOR_DIMS = {"shape": 2 * (40 + 30), "orient": 2 * 4, "loc": 2 * 4, "motion": 2 * 2, "rel": 2, "face": 2 + 10}


def anchor_targets(P, sup, anchors):
    """observable geometry anchors per 2-frame bin (§8.4) from measurements P with support `sup` [T,J]
    (student: visible; teacher: observed). Returns {family: (value [T', D], mask [T', D])}.
      shape  : root-relative finger vectors (20 joints x 2, hand-local palm scale) + 15 bends (cos, sin), per hand;
               support checked per joint (joint, root, >= 1 palm bone) / per bend triple
      orient : unit wrist->MCP9 and MCP5->MCP17 directions in image axes, per hand
      loc    : wrist - shoulder midpoint and wrist - nose (/ s), per hand
      motion : wrist displacement between the two frames of the bin (/ s), per hand
      rel    : right wrist - left wrist (/ s)
      face   : mouth width & opening, brow-to-eye vertical offsets (/ eye distance)
    An element is valid in a bin only when its full raw support is in `sup` in BOTH frames of the bin."""
    T = len(P); Tb = T // 2; c, s, _ = anchors
    out = {k: (np.zeros((Tb, d), np.float32), np.zeros((Tb, d), bool)) for k, d in ANCHOR_DIMS.items()}
    for b in range(Tb):
        t0, t1 = 2 * b, 2 * b + 1
        both = sup[t0] & sup[t1]
        Pm = (P[t0] + P[t1]) / 2
        for hi, part in enumerate((LH, RH)):
            root = part[0]
            pal = [np.linalg.norm(Pm[part[k]] - Pm[root]) for k in HAND_MCP if both[part[k]] and both[root]]
            if pal:  # per-joint support: joint + root (+ >= 1 visible palm bone for the scale)
                sc = max(float(np.mean(pal)), S_MIN)
                v, mk = out["shape"]; o = hi * 70
                okj = both[part[1:]] & both[root]
                v[b, o:o + 40] = np.where(okj[:, None], (Pm[part[1:]] - Pm[root]) / sc, 0.0).reshape(-1)
                mk[b, o:o + 40] = np.repeat(okj, 2)
                for q, (x, m_, z) in enumerate(FINGER_JOINT_TRIPLES):
                    if both[part[x]] and both[part[m_]] and both[part[z]]:
                        v[b, o + 40 + 2 * q:o + 42 + 2 * q] = _bend(Pm[part[x]], Pm[part[m_]], Pm[part[z]]); mk[b, o + 40 + 2 * q:o + 42 + 2 * q] = True
            if both[[root, part[9], part[5], part[17]]].all():
                d1 = Pm[part[9]] - Pm[root]; d2 = Pm[part[17]] - Pm[part[5]]
                v, mk = out["orient"]; o = hi * 4
                v[b, o:o + 2] = d1 / (np.linalg.norm(d1) + 1e-6); v[b, o + 2:o + 4] = d2 / (np.linalg.norm(d2) + 1e-6); mk[b, o:o + 4] = True
            wr = LWR if hi == 0 else RWR
            if both[[wr, LSH, RSH, NOSE]].all():
                v, mk = out["loc"]; o = hi * 4
                v[b, o:o + 2] = (Pm[wr] - (Pm[LSH] + Pm[RSH]) / 2) / s; v[b, o + 2:o + 4] = (Pm[wr] - Pm[NOSE]) / s; mk[b, o:o + 4] = True
            if both[wr]:
                v, mk = out["motion"]; v[b, hi * 2:hi * 2 + 2] = (P[t1, wr] - P[t0, wr]) / s; mk[b, hi * 2:hi * 2 + 2] = True
        if both[[LWR, RWR]].all():
            v, mk = out["rel"]; v[b] = (Pm[RWR] - Pm[LWR]) / s; mk[b] = True
        ea, eb = FACE_EYE_ANCHORS
        if both[ea] and both[eb]:  # per-element support: eye anchors + the element's own landmarks
            ed = max(np.linalg.norm(Pm[ea] - Pm[eb]), S_MIN); eye_y = (Pm[ea, 1] + Pm[eb, 1]) / 2
            v, mk = out["face"]
            if both[MOUTH[0]] and both[MOUTH[1]]:
                v[b, 0] = np.linalg.norm(Pm[MOUTH[0]] - Pm[MOUTH[1]]) / ed; mk[b, 0] = True
            if both[MOUTH[2]] and both[MOUTH[3]]:
                v[b, 1] = np.linalg.norm(Pm[MOUTH[2]] - Pm[MOUTH[3]]) / ed; mk[b, 1] = True
            for k, j in enumerate(BROWS):
                if both[j]:
                    v[b, 2 + k] = (eye_y - Pm[j, 1]) / ed; mk[b, 2 + k] = True
    return out
