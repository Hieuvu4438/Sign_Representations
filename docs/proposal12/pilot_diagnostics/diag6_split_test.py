"""D6: instantaneous cross-channel leak audit. For every ordered pair (source channel -> target channel) at the SAME frame,
R^2 of predicting the target channel features from the source channel features (ridge and kNN-20; fit on train frames,
test on val frames, cross-signer). High R^2 = a channel mask is trivially solvable from another visible channel."""
import sys, json, pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
D = sys.argv[1]
NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI = 0, 1, 2, 3, 4, 5, 6
FACE0, LH0, RH0 = 9, 77, 98
MOUTH = [FACE0 + i for i in range(48, 68)]
EYEC = [FACE0 + i for i in (36, 39, 42, 45)]
HTHR = 2.0

FINGER_BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
PALM = [1, 5, 9, 13, 17]
SPLIT = True

def hand_feats(h, sw):
    v = h[:, 9] - h[:, 0]; L = np.linalg.norm(v, axis=-1) + 1e-6
    c, s = v[:, 1] / L, v[:, 0] / L
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    shape = np.clip(np.einsum('tij,tkj->tki', R, h - h[:, :1]) / L[:, None, None], -4, 4)[:, 1:].reshape(len(h), -1)
    a5, a17 = h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]; cr = a5[:, 0] * a17[:, 1] - a5[:, 1] * a17[:, 0]
    ori = np.stack([v[:, 0] / L, v[:, 1] / L, np.sign(cr), np.linalg.norm(h[:, 5] - h[:, 17], axis=-1) / L, np.log(L / sw)], -1)
    if SPLIT:   # proposal12 v2: SHAPE owns only finger bones, ORI owns the palm polygon
        loc = np.clip(np.einsum('tij,tkj->tki', R, h - h[:, :1]) / L[:, None, None], -4, 4)
        bones = np.stack([loc[:, b] - loc[:, a] for a, b in FINGER_BONES], 1)
        shape = bones.reshape(len(h), -1)
        ori = np.concatenate([np.stack([v[:, 0] / L, v[:, 1] / L, np.sign(cr), np.log(L / sw)], -1),
                              loc[:, PALM].reshape(len(h), -1)], -1)
    return shape, ori, L

def frames(x):
    kp = x['kp'].astype(np.float32); sc = x['sc'].astype(np.float32)
    c = (kp[:, LSHO] + kp[:, RSHO]) / 2; sw = np.linalg.norm(kp[:, LSHO] - kp[:, RSHO], axis=-1) + 1e-6
    lv = sc[:, LH0:LH0 + 21].mean(1) > HTHR; rv = sc[:, RH0:RH0 + 21].mean(1) > HTHR
    ml = np.std(kp[lv, LWRI], 0).sum() if lv.sum() > 2 else -1
    mr = np.std(kp[rv, RWRI], 0).sum() if rv.sum() > 2 else -1
    if ml > mr:   # mirror: dominant = left -> treat as right
        kp = kp.copy(); kp[..., 0] = 2 * c[:, None, 0] - kp[..., 0]
        D_, N_, dw, nw, de, ne, dv, nv = kp[:, LH0:LH0 + 21], kp[:, RH0:RH0 + 21], LWRI, RWRI, LELB, RELB, lv, rv
    else:
        D_, N_, dw, nw, de, ne, dv, nv = kp[:, RH0:RH0 + 21], kp[:, LH0:LH0 + 21], RWRI, LWRI, RELB, LELB, rv, lv
    swm = np.median(sw); body_ok = (sw > 0.5 * swm) & (sw < 2 * swm) & (sc[:, [LWRI, RWRI, LELB, RELB]].min(1) > 1.0)
    iod_raw = np.linalg.norm(kp[:, EYEC[0]] - kp[:, EYEC[3]], axis=-1); face_ok = (iod_raw > 0.15 * sw) & (sc[:, MOUTH].mean(1) > 2.0)
    ok = body_ok & face_ok & dv & nv & (np.linalg.norm(D_[:, 9] - D_[:, 0], axis=-1) > 0.1 * sw) & (np.linalg.norm(N_[:, 9] - N_[:, 0], axis=-1) > 0.1 * sw)
    if ok.sum() < 2:
        return None
    sw_ = sw[:, None]
    dsh, dor, _ = hand_feats(D_, sw); nsh, nor, _ = hand_feats(N_, sw)
    dlo = np.concatenate([(kp[:, dw] - c) / sw_, (kp[:, de] - c) / sw_], -1)
    nlo = np.concatenate([(kp[:, nw] - c) / sw_, (kp[:, ne] - c) / sw_], -1)
    eyem = kp[:, EYEC].mean(1); iod = np.linalg.norm(kp[:, EYEC[0]] - kp[:, EYEC[3]], axis=-1)[:, None, None] + 1e-6
    mouth = ((kp[:, MOUTH] - eyem[:, None]) / iod)
    mouth = (mouth - np.median(mouth[ok], 0)).reshape(len(kp), -1)        # neutral-face deviation (clip median)
    head = np.concatenate([(eyem - c) / sw_, iod[:, :, 0] / sw_], -1)
    F = dict(D_SH=dsh, D_OR=dor, D_LO=dlo, N_SH=nsh, N_OR=nor, N_LO=nlo, F_MO=mouth, F_HD=head)
    return {k: v[ok] for k, v in F.items()}, x['signer']

def collect(path, maxf, seed):
    data = pickle.load(open(path, 'rb')); out = {}
    for x in data:
        r = frames(x)
        if r is None: continue
        for k, v in r[0].items(): out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    n = len(next(iter(out.values()))); idx = np.random.default_rng(seed).choice(n, min(maxf, n), replace=False)
    return {k: np.nan_to_num(v[idx]) for k, v in out.items()}

tr = collect(f'{D}/train14k.pkl', 150000, 0); va = collect(f'{D}/val.pkl', 60000, 1)
for k in tr:
    lo, hi = np.percentile(tr[k], 1, axis=0), np.percentile(tr[k], 99, axis=0)
    tr[k] = np.clip(tr[k], lo, hi); va[k] = np.clip(va[k], lo, hi)
keys = list(tr); res = {}
for t in keys:
    sy = StandardScaler().fit(tr[t]); Ytr, Yva = sy.transform(tr[t]), np.clip(sy.transform(va[t]), -5, 5)
    for s in keys:
        if s == t: continue
        sx = StandardScaler().fit(tr[s]); Xtr, Xva = sx.transform(tr[s]), sx.transform(va[s])
        r_lin = r2_score(Yva, np.clip(Ridge(1.0).fit(Xtr, Ytr).predict(Xva), -5, 5))
        knn = KNeighborsRegressor(20).fit(Xtr[:60000], Ytr[:60000])
        r_knn = r2_score(Yva[:20000], np.clip(knn.predict(Xva[:20000]), -5, 5))
        res[f'{s}->{t}'] = [round(float(r_lin), 3), round(float(r_knn), 3)]
        print(f'{s:5s} -> {t:5s}  ridge {r_lin:6.3f}  knn {r_knn:6.3f}', flush=True)
json.dump(res, open(f'{D}/diag6_results.json' if SPLIT else f'{D}/diag6_results_nosplit.json', 'w'), indent=1)
