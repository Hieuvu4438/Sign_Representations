"""Pilot diagnostics for proposal12 (CPU only, no training of any SSL model).

D1  signer leakage vs gloss information of simple skeleton feature sets (morphology / raw / factorised / bone-normalised)
D2  predictability ladder of the dominant handshape: temporal interpolation, intra-hand, cross-articulator, signer prior, lexical oracle
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
NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI = 0, 1, 2, 3, 4, 5, 6
FACE0, LH0, RH0 = 9, 77, 98
MOUTH = [FACE0 + i for i in range(48, 68)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HTHR = 2.0
NT = 16


def interp_valid(x, valid, n=NT):
    """Resample the valid frames of x (T,...) to n frames (linear)."""
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return None
    xs = x[idx].reshape(len(idx), -1)
    t = np.linspace(0, len(idx) - 1, n)
    out = np.stack([np.interp(t, np.arange(len(idx)), xs[:, j]) for j in range(xs.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def hand_local(h):
    """h: (T,21,2) -> local shape (T,20,2) in palm frame, orientation feats (T,4), palm length (T,)."""
    v = h[:, 9] - h[:, 0]
    L = np.linalg.norm(v, axis=-1) + 1e-6
    c, s = v[:, 1] / L, v[:, 0] / L                 # rotate v onto +y
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)   # (T,2,2)
    loc = np.clip(np.einsum('tij,tkj->tki', R, h - h[:, :1]) / L[:, None, None], -4, 4)
    width = np.linalg.norm(h[:, 5] - h[:, 17], axis=-1) / L
    ori = np.stack([v[:, 0] / L, v[:, 1] / L, width, np.log(L)], -1)
    return loc[:, 1:], ori, L


def clip_feats(x):
    kp = x['kp'].astype(np.float32)
    sc = x['sc'].astype(np.float32)
    T = len(kp)
    center = np.median((kp[:, LSHO] + kp[:, RSHO]) / 2, 0)
    sw = np.median(np.linalg.norm(kp[:, LSHO] - kp[:, RSHO], axis=-1)) + 1e-6
    kp = (kp - center) / sw
    lv = sc[:, LH0:LH0 + 21].mean(1) > HTHR
    rv = sc[:, RH0:RH0 + 21].mean(1) > HTHR
    # dominant hand = larger wrist motion among valid frames
    def mot(w, v):
        return np.nanstd(kp[v, w], 0).sum() if v.sum() > 2 else -1
    left_dom = mot(LWRI, lv) > mot(RWRI, rv)
    if left_dom:   # mirror so that the dominant hand is always "right"
        kp = kp.copy(); kp[..., 0] *= -1
        perm = list(range(len(kp[0])))
        perm[LSHO], perm[RSHO], perm[LELB], perm[RELB], perm[LWRI], perm[RWRI] = RSHO, LSHO, RELB, LELB, RWRI, LWRI
        perm[LH0:LH0 + 21], perm[RH0:RH0 + 21] = list(range(RH0, RH0 + 21)), list(range(LH0, LH0 + 21))
        kp = kp[:, perm]; lv, rv = rv, lv
    dom, nd = kp[:, RH0:RH0 + 21], kp[:, LH0:LH0 + 21]
    def palm_ok(h, v):
        L = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)
        if v.sum() < 2:
            return v
        return v & (L > 0.5 * np.median(L[v]))
    dv, nv = palm_ok(dom, rv), palm_ok(nd, lv)
    if dv.sum() < 4:
        return None
    dshape, dori, dL = hand_local(dom)
    nshape, nori, nL = hand_local(nd)
    nshape = np.where(nv[:, None, None], nshape, 0.0); nori = np.where(nv[:, None], nori, 0.0)
    dloc = dom[:, 0]; nloc = np.where(nv[:, None], nd[:, 0], 0.0)
    mouth = kp[:, MOUTH] - kp[:, NOSE:NOSE + 1]
    # static morphology: clip-median bone lengths (shoulder-width units)
    def blen(a, b):
        return np.median(np.linalg.norm(kp[:, a] - kp[:, b], axis=-1))
    morph = [blen(RSHO, RELB), blen(RELB, RWRI), blen(LSHO, LELB), blen(LELB, LWRI), sw,
             blen(NOSE, LSHO), blen(NOSE, RSHO), blen(FACE0 + 36, FACE0 + 45), blen(FACE0 + 0, FACE0 + 16)]
    hb = [np.median(np.linalg.norm(dom[dv, a] - dom[dv, b], axis=-1)) for a, b in HAND_BONES]
    morph += hb + [b / (hb[8] + 1e-6) for b in hb]
    # per-clip per-bone length normalisation of the dominant hand: bone vectors / their clip 90th pct length
    bvec = np.stack([dom[:, b] - dom[:, a] for a, b in HAND_BONES], 1)            # T,20,2
    bl = np.linalg.norm(bvec, axis=-1)
    p90 = np.percentile(bl[dv], 90, axis=0) + 1e-6
    bnorm = bvec / p90[None, :, None]
    r = lambda a, v=dv: interp_valid(a, v)
    out = dict(signer=x['signer'], gloss=x['gloss'], code=x['code'], morph=np.array(morph, np.float32))
    seq_dshape, seq_dori, seq_dloc = r(dshape), r(dori), r(dloc)
    seq_nshape, seq_nori, seq_nloc = r(nshape), r(nori), r(nloc)
    seq_mouth = r(mouth)
    seq_raw = r(np.concatenate([dom, nd, kp[:, [RSHO, RELB, RWRI, LSHO, LELB, LWRI]]], 1))
    seq_bnorm = r(bnorm)
    out.update(raw=seq_raw.reshape(-1),
               fact=np.concatenate([seq_dloc.reshape(NT, -1), seq_dori, seq_dshape.reshape(NT, -1),
                                    seq_nloc.reshape(NT, -1), seq_nori, seq_nshape.reshape(NT, -1)], 1).reshape(-1),
               fact_bn=np.concatenate([seq_dloc.reshape(NT, -1), seq_dori[:, :3], seq_bnorm.reshape(NT, -1),
                                       seq_nloc.reshape(NT, -1), seq_nori[:, :3], seq_nshape.reshape(NT, -1)], 1).reshape(-1),
               dloc=seq_dloc.reshape(-1), dori=seq_dori.reshape(-1), nd_all=np.concatenate(
                   [seq_nloc.reshape(NT, -1), seq_nori, seq_nshape.reshape(NT, -1)], 1).reshape(-1),
               mouth=seq_mouth.reshape(-1),
               dshape_clip=seq_dshape[4:12].mean(0).reshape(-1))
    # frame-level material for interpolation / intra-hand tests
    idx = np.where(dv)[0]
    out['frames_shape'] = dshape[idx].reshape(len(idx), -1)
    out['frames_t'] = idx
    out['frames_raw'] = (dom[idx] - dom[idx, :1]).reshape(len(idx), -1)   # wrist-relative, shoulder-width units
    return out


def build(path):
    data = pickle.load(open(path, 'rb'))
    feats = [clip_feats(x) for x in data]
    return [f for f in feats if f is not None]


def stack(F, k):
    return np.stack([f[k] for f in F]).astype(np.float32)


def signer_probe(F, k):
    y = np.array([f['signer'] for f in F]); keep = np.isin(y, [s for s, c in zip(*np.unique(y, return_counts=True)) if c > 50])
    X = StandardScaler().fit_transform(np.nan_to_num(stack(F, k)[keep])); y = y[keep]
    pred = cross_val_predict(LogisticRegression(max_iter=300, C=0.1), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return balanced_accuracy_score(y, pred)


def gloss_1nn(Ftr, Fva, k):
    sc = StandardScaler().fit(np.nan_to_num(stack(Ftr, k)))
    A = sc.transform(np.nan_to_num(stack(Ftr, k))); B = sc.transform(np.nan_to_num(stack(Fva, k)))
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-6; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-6
    ga = np.array([f['gloss'] for f in Ftr]); gb = np.array([f['gloss'] for f in Fva])
    keep = np.isin(gb, ga)
    nn = (B[keep] @ A.T).argmax(1)
    return float((ga[nn] == gb[keep]).mean()), int(keep.sum())


def main():
    Ftr = build(f'{D}/train14k.pkl'); Fva = build(f'{D}/val.pkl')
    res = {'n_train': len(Ftr), 'n_val': len(Fva)}
    # ---------------- D1
    d1 = {}
    for k in ['morph', 'raw', 'fact', 'fact_bn']:
        d1[k] = dict(signer_bal_acc=round(signer_probe(Fva, k), 4), gloss_1nn_cross_signer=gloss_1nn(Ftr, Fva, k))
        print('D1', k, d1[k], flush=True)
    res['D1'] = d1
    # ---------------- D2 clip-level: predict dominant handshape descriptor
    ytr = stack(Ftr, 'dshape_clip'); yva = stack(Fva, 'dshape_clip')
    ys = StandardScaler().fit(ytr); ytr_s, yva_s = ys.transform(ytr), ys.transform(yva)
    def r2(p):
        return float(r2_score(yva_s, p, multioutput='uniform_average'))
    d2 = {}
    ctxs = {'dloc': ['dloc'], 'dloc+dori': ['dloc', 'dori'], 'dloc+dori+nondom': ['dloc', 'dori', 'nd_all'],
            'dloc+dori+nondom+mouth': ['dloc', 'dori', 'nd_all', 'mouth']}
    for name, keys in ctxs.items():
        Xtr = np.concatenate([stack(Ftr, k) for k in keys], 1); Xva = np.concatenate([stack(Fva, k) for k in keys], 1)
        sx = StandardScaler().fit(np.nan_to_num(Xtr))
        knn = KNeighborsRegressor(20, weights='distance').fit(sx.transform(np.nan_to_num(Xtr)), ytr_s)
        d2['knn20:' + name] = round(r2(knn.predict(sx.transform(np.nan_to_num(Xva)))), 4)
        print('D2', name, d2['knn20:' + name], flush=True)
    # lexical oracle: mean handshape of same-gloss train clips (different signers)
    by_g = defaultdict(list)
    for f, y in zip(Ftr, ytr_s):
        by_g[f['gloss']].append(y)
    gm = {g: np.mean(v, 0) for g, v in by_g.items()}
    keep = np.array([f['gloss'] in gm for f in Fva])
    pred = np.stack([gm[f['gloss']] for f in np.array(Fva, dtype=object)[keep]])
    d2['oracle_same_gloss_other_signers'] = round(float(r2_score(yva_s[keep], pred)), 4)
    # signer prior: leave-one-out mean of the same signer's other val clips
    by_s = defaultdict(list)
    for i, f in enumerate(Fva):
        by_s[f['signer']].append(i)
    pred = np.zeros_like(yva_s)
    for s, ii in by_s.items():
        tot = yva_s[ii].sum(0)
        for i in ii:
            pred[i] = (tot - yva_s[i]) / max(len(ii) - 1, 1)
    d2['signer_prior_loo'] = round(r2(pred), 4)
    # ---------------- D2 frame-level: temporal interpolation and intra-hand
    for key, gap in [(k, g) for k in ['frames_shape', 'frames_raw'] for g in [1, 2, 4, 8]]:
        P, Y = [], []
        for f in Fva:
            S, t = f[key], f['frames_t']
            pos = {tt: i for i, tt in enumerate(t)}
            for i, tt in enumerate(t):
                if tt - gap in pos and tt + gap in pos:
                    P.append((S[pos[tt - gap]] + S[pos[tt + gap]]) / 2); Y.append(S[i])
        P, Y = np.array(P), np.array(Y)
        fs = StandardScaler().fit(Y)
        d2[f'{key}_interp_gap{gap}'] = round(float(r2_score(np.clip(fs.transform(Y), -5, 5), np.clip(fs.transform(P), -5, 5))), 4)
        d2[f'{key}_interp_gap{gap}_n'] = len(Y)
    # intra-hand: middle/ring/pinky (local joints 9..20 -> rows 8..19) from thumb+index (rows 0..7), ridge on train frames
    def fr(Fs, n):
        X = np.concatenate([f['frames_shape'] for f in Fs]); rng = np.random.default_rng(0)
        return X[rng.choice(len(X), min(n, len(X)), replace=False)]
    Xtr_f, Xva_f = fr(Ftr, 200000), fr(Fva, 100000)
    vis, hid = slice(0, 16), slice(16, 40)
    for name, mdl in [('ridge', Ridge(1.0)), ('knn20', KNeighborsRegressor(20))]:
        sxa = StandardScaler().fit(Xtr_f[:, vis]); sya = StandardScaler().fit(Xtr_f[:, hid])
        mdl.fit(sxa.transform(Xtr_f[:, vis]), sya.transform(Xtr_f[:, hid]))
        d2[f'frame_intrahand_{name}'] = round(float(r2_score(sya.transform(Xva_f[:, hid]), mdl.predict(sxa.transform(Xva_f[:, vis])))), 4)
    res['D2'] = d2
    print(json.dumps(res, indent=1))
    json.dump(res, open(f'{D}/diag_results_v2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
