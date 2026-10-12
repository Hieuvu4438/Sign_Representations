"""Pilot diagnostics v2 for proposal12 (after critical review). Training-free, CPU only.

Fixes vs v1: no constant / near-constant columns (v1 had the middle-MCP column = 1/palm-length proxy),
features match the revised tokenisation exactly (hand = CONF + LOC; face = UPPER, MOUTH, HEAD),
correct iBUG face mirroring, palm-length floor instead of dropping foreshortened frames,
gloss-balanced signer probe, one change at a time between descriptor variants.

Channels (per frame, dominant hand canonicalised by mirroring):
  CONF  hand configuration: palm axis u (2), palm-facing sign (1), log palm scale (1), foreshortening flag (1),
        palm polygon CMC1/MCP5/MCP13/MCP17 in palm frame (8), 15 finger bones in palm frame (30)            = 43
  LOC   body wrist (2), body elbow (2) relative to shoulder centre / shoulder width, wrist velocity (2)        = 6
  F_UP  brows + eyelids (18 pts) in eye frame (36);  F_MO mouth (20 pts) in eye frame (40)
  F_HD  eye-frame origin rel. body (2), IOD/sigma (1), roll (2), nose tip + chin in eye frame (4)               = 9
Usage: python diag_v2.py OUT_DIR [--quick]
"""
import json, pickle, sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, r2_score

D = sys.argv[1]
QUICK = '--quick' in sys.argv
NT = 16
# local joint layout of load_kp.py: body 9, face 68, LH 21, RH 21
NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP = range(9)
F0, LH0, RH0 = 9, 77, 98
IBUG_MIRROR = list(range(16, -1, -1)) + list(range(26, 16, -1)) + [27, 28, 29, 30] + [35, 34, 33, 32, 31] + \
    [45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40] + [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55] + \
    [64, 63, 62, 61, 60, 67, 66, 65]
assert sorted(IBUG_MIRROR) == list(range(68))
EYEC = [36, 39, 42, 45]
UPPER = list(range(17, 27)) + [37, 38, 40, 41, 43, 44, 46, 47]
MOUTH = list(range(48, 68))
PALM = [1, 5, 13, 17]
FINGER_BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
HTHR, FTHR = 2.0, 2.0


def mirror_clip(kp, sc, cx):
    kp = kp.copy(); kp[..., 0] = 2 * cx - kp[..., 0]
    perm = list(range(kp.shape[1]))
    for a, b in [(LSHO, RSHO), (LELB, RELB), (LWRI, RWRI), (LHIP, RHIP)]:
        perm[a], perm[b] = b, a
    perm[F0:F0 + 68] = [F0 + i for i in IBUG_MIRROR]
    perm[LH0:LH0 + 21], perm[RH0:RH0 + 21] = list(range(RH0, RH0 + 21)), list(range(LH0, LH0 + 21))
    return kp[:, perm], sc[:, perm]


def hand_conf(h, sigma, bone_norm, rel_scale, valid):
    v = h[:, 9] - h[:, 0]; L = np.linalg.norm(v, axis=-1)
    Leff = np.maximum(L, 0.1 * sigma)
    u = v / (L[:, None] + 1e-9)
    R = np.stack([np.stack([u[:, 1], -u[:, 0]], -1), np.stack([u[:, 0], u[:, 1]], -1)], -2)   # maps u -> (0, 1)
    P = np.einsum('tij,tkj->tki', R, h - h[:, :1]) / Leff[:, None, None]
    a5, a17 = h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]
    sign = np.sign(a5[:, 0] * a17[:, 1] - a5[:, 1] * a17[:, 0])
    logs = np.log(Leff / sigma)
    if rel_scale and valid.sum() > 0:
        logs = logs - np.median(logs[valid])
    fore = 1.0 - L / Leff
    bones = np.stack([P[:, b] - P[:, a] for a, b in FINGER_BONES], 1)                          # T,15,2
    if bone_norm and valid.sum() > 0:
        bl = np.linalg.norm(bones, axis=-1)
        bones = bones / (np.percentile(bl[valid], 90, axis=0) + 1e-6)[None, :, None]
    conf = np.concatenate([u, sign[:, None], logs[:, None], fore[:, None], P[:, PALM].reshape(len(h), -1),
                           bones.reshape(len(h), -1)], -1)
    return np.clip(conf, -6, 6)


def face_feats(kp, c, sigma, valid, neutral, rel_scale):
    f = kp[:, F0:F0 + 68]
    eyeR, eyeL = f[:, [36, 39]].mean(1), f[:, [42, 45]].mean(1)
    o = (eyeR + eyeL) / 2; ax = eyeL - eyeR; iod = np.linalg.norm(ax, axis=-1) + 1e-9; ex = ax / iod[:, None]
    R = np.stack([ex, np.stack([-ex[:, 1], ex[:, 0]], -1)], -2)                                   # rows: x-axis, y-axis
    Fe = np.einsum('tij,tkj->tki', R, f - o[:, None]) / iod[:, None, None]
    up, mo = Fe[:, UPPER].reshape(len(f), -1), Fe[:, MOUTH].reshape(len(f), -1)
    if neutral and valid.sum() > 0:
        up = up - np.median(up[valid], 0); mo = mo - np.median(mo[valid], 0)
    sc_ = np.log(iod / sigma)
    if rel_scale and valid.sum() > 0:
        sc_ = sc_ - np.median(sc_[valid])
    hd = np.concatenate([(o - c) / sigma[:, None], sc_[:, None], ex, Fe[:, 30], Fe[:, 8]], -1)
    return np.clip(up, -6, 6), np.clip(mo, -6, 6), np.clip(hd, -6, 6)


def clip_channels(x, bone_norm=True, rel_scale=True, neutral=True):
    kp = x['kp'].astype(np.float32); sc = x['sc'].astype(np.float32)
    cx = np.median((kp[:, LSHO, 0] + kp[:, RSHO, 0]) / 2)
    bw = (sc[:, [LWRI, RWRI]] > 1.0)
    ml = np.abs(np.diff(kp[:, LWRI], axis=0)).sum(-1)[bw[1:, 0] & bw[:-1, 0]].sum()
    mr = np.abs(np.diff(kp[:, RWRI], axis=0)).sum(-1)[bw[1:, 1] & bw[:-1, 1]].sum()
    if ml > mr:
        kp, sc = mirror_clip(kp, sc, cx)
    c = (kp[:, LSHO] + kp[:, RSHO]) / 2
    sigma = np.linalg.norm(kp[:, LSHO] - kp[:, RSHO], axis=-1)
    sigma = np.maximum(sigma, 0.5 * np.median(sigma)) + 1e-9
    out = {'signer': x['signer'], 'gloss': x['gloss'], 'code': x['code']}
    ch, val = {}, {}
    for side, H0, W, E in [('D', RH0, RWRI, RELB), ('N', LH0, LWRI, LELB)]:
        hv = sc[:, H0:H0 + 21].mean(1) > HTHR
        ch[side + '_CONF'] = hand_conf(kp[:, H0:H0 + 21], sigma, bone_norm, rel_scale, hv)
        w, e = (kp[:, W] - c) / sigma[:, None], (kp[:, E] - c) / sigma[:, None]
        vel = np.vstack([np.zeros((1, 2)), np.diff(w, axis=0)])
        ch[side + '_LOC'] = np.clip(np.concatenate([w, e, vel], -1), -6, 6)
        val[side + '_CONF'] = hv; val[side + '_LOC'] = sc[:, [W, E]].min(1) > 1.0
    fv = sc[:, F0:F0 + 68].mean(1) > FTHR
    ch['F_UP'], ch['F_MO'], ch['F_HD'] = face_feats(kp, c, sigma, fv, neutral, rel_scale)
    for k in ['F_UP', 'F_MO', 'F_HD']:
        val[k] = fv
    if val['D_CONF'].sum() < 4:
        return None
    out['ch'], out['val'] = ch, val
    return out


def resample(a, idx):
    t = np.linspace(0, len(idx) - 1, NT)
    xs = a[idx]
    return np.stack([np.interp(t, np.arange(len(idx)), xs[:, j]) for j in range(xs.shape[1])], 1)


def descriptor(o, keys):
    idx = np.where(o['val']['D_CONF'])[0]
    parts = []
    for k in keys:
        a = np.where(o['val'][k][:, None], o['ch'][k], 0.0)
        parts.append(resample(a, idx))
    return np.concatenate(parts, 1).reshape(-1).astype(np.float32)


def build(path, maxn=None, **kw):
    data = pickle.load(open(path, 'rb'))
    if maxn:
        data = data[:maxn]
    return [o for o in (clip_channels(x, **kw) for x in data) if o is not None]


def signer_probe(Fv, X, balanced3=False):
    y = np.array([o['signer'] for o in Fv])
    if balanced3:
        su, sn = np.unique(y, return_counts=True)
        big = list(su[np.argsort(-sn)[:3]])                                   # three largest signers
        gl = [set(o['gloss'] for o in Fv if o['signer'] == s) for s in big]
        common = set.intersection(*gl)
        keep = np.array([(o['signer'] in big) and (o['gloss'] in common) for o in Fv])
    else:
        keep = np.isin(y, [s for s, n in zip(*np.unique(y, return_counts=True)) if n > 50])
    Xs = StandardScaler().fit_transform(np.nan_to_num(X[keep])); ys = y[keep]
    pred = cross_val_predict(LogisticRegression(max_iter=300, C=0.1), Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return round(float(balanced_accuracy_score(ys, pred)), 4), int(keep.sum())


def off_retrieval(Ftr, Fva, Xtr, Xva):
    ga = np.array([o['gloss'] for o in Ftr]); gb = np.array([o['gloss'] for o in Fva])
    classes = np.unique(ga); cid = {c: i for i, c in enumerate(classes)}
    ya = np.array([cid[g] for g in ga]); yb = np.array([cid.get(g, -1) for g in gb]); keep = yb >= 0
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd; B = (Xva - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-6; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-6
    order = np.argsort(ya, kind='stable'); bounds = np.r_[0, np.flatnonzero(np.diff(ya[order])) + 1]
    A = A[order]
    hit = 0
    for s in range(0, len(B), 1024):
        C = np.maximum.reduceat(B[s:s + 1024] @ A.T, bounds, axis=1)
        yy = np.clip(yb[s:s + 1024], 0, None)
        rank = (C > C[np.arange(len(C)), yy][:, None]).sum(1)
        hit += ((rank == 0) & keep[s:s + 1024]).sum()
    return round(float(hit / keep.sum()), 4)


def r2(Ytr, Yte, Ptr_fn):
    pass


def main():
    res = {}
    nmax = 3000 if QUICK else None
    Ftr_full = build(f'{D}/train_full.pkl', maxn=6000 if QUICK else None)
    Fva = build(f'{D}/val.pkl', maxn=nmax)
    HAND = ['D_CONF', 'D_LOC', 'N_CONF', 'N_LOC']
    FACE = ['F_UP', 'F_MO', 'F_HD']
    # ---------- D1/D5: one change at a time
    variants = {
        'v2_fact_plain':        dict(bone_norm=False, rel_scale=False, neutral=False),
        'v2_fact_bn':           dict(bone_norm=True,  rel_scale=False, neutral=False),
        'v2_fact_bn_rel':       dict(bone_norm=True,  rel_scale=True,  neutral=False),
    }
    D1 = {}
    for name, kw in variants.items():
        tr = Ftr_full if kw == dict(bone_norm=True, rel_scale=True, neutral=True) else build(f'{D}/train_full.pkl', maxn=6000 if QUICK else None, **kw)
        va = build(f'{D}/val.pkl', maxn=nmax, **kw)
        for keys, tag in [(HAND, 'hands'), (HAND + FACE, 'hands+face')]:
            Xtr = np.stack([descriptor(o, keys) for o in tr]); Xva = np.stack([descriptor(o, keys) for o in va])
            D1[f'{name}|{tag}'] = dict(signer5=signer_probe(va, Xva), signer3_balanced=signer_probe(va, Xva, True),
                                     offval_R1=off_retrieval(tr, va, Xtr, Xva))
            print('D1', name, tag, D1[f'{name}|{tag}'], flush=True)
    # final variant (neutral face) = default proposal tokenisation
    for keys, tag in [(HAND, 'hands'), (HAND + FACE, 'hands+face')]:
        Xtr = np.stack([descriptor(o, keys) for o in Ftr_full]); Xva = np.stack([descriptor(o, keys) for o in Fva])
        D1[f'v2_full_default|{tag}'] = dict(signer5=signer_probe(Fva, Xva), signer3_balanced=signer_probe(Fva, Xva, True),
                                          offval_R1=off_retrieval(Ftr_full, Fva, Xtr, Xva))
        print('D1 default', tag, D1[f'v2_full_default|{tag}'], flush=True)
    res['D1_D5'] = D1
    # ---------- D4: per channel (default tokenisation) + raw-face variants
    D4 = {}
    Fva_raw = build(f'{D}/val.pkl', maxn=nmax, bone_norm=True, rel_scale=False, neutral=False)
    Ftr_raw = build(f'{D}/train_full.pkl', maxn=6000 if QUICK else None, bone_norm=True, rel_scale=False, neutral=False)
    for k in HAND + FACE:
        Xtr = np.stack([descriptor(o, [k]) for o in Ftr_full]); Xva = np.stack([descriptor(o, [k]) for o in Fva])
        D4[k] = dict(signer5=signer_probe(Fva, Xva), signer3_balanced=signer_probe(Fva, Xva, True), offval_R1=off_retrieval(Ftr_full, Fva, Xtr, Xva))
        print('D4', k, D4[k], flush=True)
    for k in FACE:
        Xtr = np.stack([descriptor(o, [k]) for o in Ftr_raw]); Xva = np.stack([descriptor(o, [k]) for o in Fva_raw])
        D4[k + '_raw'] = dict(signer5=signer_probe(Fva_raw, Xva), signer3_balanced=signer_probe(Fva_raw, Xva, True), offval_R1=off_retrieval(Ftr_raw, Fva_raw, Xtr, Xva))
        print('D4', k + '_raw', D4[k + '_raw'], flush=True)
    res['D4'] = D4
    # ---------- D2: predictability ladder of the dominant finger bones (30 dims, cols 13:43 of CONF)
    D2 = {}
    FB = slice(13, 43)
    def frames_of(F, key='D_CONF'):
        out = []
        for o in F:
            v = o['val'][key]; out.append((o['ch'][key][:, FB], v))
        return out
    for gap in [1, 2, 4, 8]:
        P, Y = [], []
        for S, v in frames_of(Fva):
            t = np.where(v)[0]; ok = set(t)
            for tt in t:
                if tt - gap in ok and tt + gap in ok:
                    P.append((S[tt - gap] + S[tt + gap]) / 2); Y.append(S[tt])
        P, Y = np.array(P), np.array(Y); sc_ = StandardScaler().fit(Y)
        D2[f'interp_gap{gap}'] = round(float(r2_score(np.clip(sc_.transform(Y), -5, 5), np.clip(sc_.transform(P), -5, 5))), 4)
    # intra-hand: PIP/DIP/TIP bones of middle, ring, pinky (bones 6..14) from thumb+index bones (0..5) + palm polygon
    def framemat(F):
        X = [o['ch']['D_CONF'][o['val']['D_CONF']] for o in F]
        return np.concatenate(X)
    rng = np.random.default_rng(0)
    Xtr_f = framemat(Ftr_full); Xva_f = framemat(Fva)
    Xtr_f = Xtr_f[rng.choice(len(Xtr_f), min(200000, len(Xtr_f)), replace=False)]
    Xva_f = Xva_f[rng.choice(len(Xva_f), min(80000, len(Xva_f)), replace=False)]
    vis = list(range(5, 13)) + list(range(13, 25)); hid = list(range(25, 43))     # palm polygon + bones 0..5 -> bones 6..14
    for nm, mdl in [('ridge', Ridge(1.0)), ('knn20', KNeighborsRegressor(20))]:
        sx, sy = StandardScaler().fit(Xtr_f[:, vis]), StandardScaler().fit(Xtr_f[:, hid])
        mdl.fit(sx.transform(Xtr_f[:, vis]), sy.transform(Xtr_f[:, hid]))
        D2[f'intrahand_{nm}'] = round(float(r2_score(np.clip(sy.transform(Xva_f[:, hid]), -5, 5), np.clip(mdl.predict(sx.transform(Xva_f[:, vis])), -5, 5))), 4)
    # clip level: D-CONF descriptor (middle 8 of 16 steps, mean) from other channels
    def clipdesc(F, keys):
        return np.stack([descriptor(o, keys) for o in F])
    def tgt(F):
        Z = clipdesc(F, ['D_CONF']).reshape(len(F), NT, -1)
        return Z[:, 4:12].mean(1)
    Ytr, Yva = tgt(Ftr_full), tgt(Fva); sy = StandardScaler().fit(Ytr); Ytr_s, Yva_s = sy.transform(Ytr), np.clip(sy.transform(Yva), -5, 5)
    for nm, keys in [('D_LOC', ['D_LOC']), ('D_LOC+N', ['D_LOC', 'N_CONF', 'N_LOC']), ('D_LOC+N+face', ['D_LOC', 'N_CONF', 'N_LOC'] + FACE),
                     ('N_CONF', ['N_CONF']), ('face', FACE)]:
        Xtr, Xva = clipdesc(Ftr_full, keys), clipdesc(Fva, keys); sx = StandardScaler().fit(Xtr)
        knn = KNeighborsRegressor(20, weights='distance').fit(sx.transform(Xtr), Ytr_s)
        D2[f'clip_CONF<-{nm}'] = round(float(r2_score(Yva_s, np.clip(knn.predict(sx.transform(Xva)), -5, 5))), 4)
        print('D2', nm, D2[f'clip_CONF<-{nm}'], flush=True)
    byg = defaultdict(list)
    for o, y in zip(Ftr_full, Ytr_s):
        byg[o['gloss']].append(y)
    gm = {g: np.mean(v, 0) for g, v in byg.items()}
    keep = np.array([o['gloss'] in gm for o in Fva])
    D2['clip_CONF<-lexical_oracle'] = round(float(r2_score(Yva_s[keep], np.stack([gm[o['gloss']] for o, k in zip(Fva, keep) if k]))), 4)
    bys = defaultdict(list)
    for i, o in enumerate(Fva):
        bys[o['signer']].append(i)
    pred = np.zeros_like(Yva_s)
    for s, ii in bys.items():
        tot = Yva_s[ii].sum(0)
        for i in ii:
            pred[i] = (tot - Yva_s[i]) / max(len(ii) - 1, 1)
    D2['clip_CONF<-signer_offset_loo'] = round(float(r2_score(Yva_s, pred)), 4)
    res['D2'] = D2
    print('D2', D2, flush=True)
    # ---------- D6: instantaneous leak audit between final channels (frame level, robust)
    keys = HAND + FACE
    def fr(F, n, seed):
        rows = {k: [] for k in keys}
        for o in F:
            ok = np.ones(len(o['ch']['D_CONF']), bool)
            for k in keys:
                ok &= o['val'][k]
            for k in keys:
                rows[k].append(o['ch'][k][ok])
        rows = {k: np.concatenate(v) for k, v in rows.items()}
        m = len(rows[keys[0]]); idx = np.random.default_rng(seed).choice(m, min(n, m), replace=False)
        return {k: v[idx] for k, v in rows.items()}
    tr6, va6 = fr(Ftr_full, 150000, 0), fr(Fva, 40000, 1)
    for k in keys:
        lo, hi = np.percentile(tr6[k], 1, 0), np.percentile(tr6[k], 99, 0)
        tr6[k], va6[k] = np.clip(tr6[k], lo, hi), np.clip(va6[k], lo, hi)
    D6 = {}
    for t in keys:
        sy = StandardScaler().fit(tr6[t]); Yt, Yv = sy.transform(tr6[t]), np.clip(sy.transform(va6[t]), -5, 5)
        for s in keys:
            if s == t:
                continue
            sx = StandardScaler().fit(tr6[s]); Xt, Xv = sx.transform(tr6[s]), sx.transform(va6[s])
            rl = r2_score(Yv, np.clip(Ridge(1.0).fit(Xt, Yt).predict(Xv), -5, 5))
            rk = r2_score(Yv[:20000], np.clip(KNeighborsRegressor(20).fit(Xt[:60000], Yt[:60000]).predict(Xv[:20000]), -5, 5))
            D6[f'{s}->{t}'] = [round(float(rl), 3), round(float(rk), 3)]
    res['D6'] = D6
    print('D6', json.dumps(D6), flush=True)
    json.dump(res, open(f'{D}/diag_v2_results{"_quick" if QUICK else ""}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
