"""D3: geometric leakage between ORIENTATION and SHAPE channels for two hand factorisations.

v1 = similarity canonicalisation (rotate wrist->middle-MCP to +y, scale by palm length); ori = axis, palm width ratio, log L
v2 = affine palm canonicalisation (least-squares 2x2 affine from a canonical palm template of wrist+5 MCP/CMC points);
     ori = affine params (rotation, log sx, log sy, shear) ; shape = A^-1 (h - wrist) for the 15 finger joints only
Leakage = R^2 of predicting SHAPE from ORI of the same frame (ridge + kNN), cross-signer split.
Also reports clip-level 1-NN gloss accuracy (train14k gallery -> val queries) of [loc, ori, shape] descriptors.
"""
import pickle, sys, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

D = sys.argv[1]
LSHO, RSHO, LWRI, RWRI = 1, 2, 5, 6
LH0, RH0 = 77, 98
PALM = [0, 1, 5, 9, 13, 17]
FING = [i for i in range(21) if i not in PALM]
HTHR = 2.0


def dominant(x):
    kp = x['kp'].astype(np.float32); sc = x['sc'].astype(np.float32)
    c = np.median((kp[:, LSHO] + kp[:, RSHO]) / 2, 0); sw = np.median(np.linalg.norm(kp[:, LSHO] - kp[:, RSHO], axis=-1)) + 1e-6
    kp = (kp - c) / sw
    lv = sc[:, LH0:LH0 + 21].mean(1) > HTHR; rv = sc[:, RH0:RH0 + 21].mean(1) > HTHR
    ml = np.std(kp[lv, LWRI], 0).sum() if lv.sum() > 2 else -1
    mr = np.std(kp[rv, RWRI], 0).sum() if rv.sum() > 2 else -1
    if ml > mr:
        h, v = kp[:, LH0:LH0 + 21].copy(), lv; h[..., 0] *= -1
    else:
        h, v = kp[:, RH0:RH0 + 21], rv
    L = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)
    if v.sum() < 4:
        return None
    v = v & (L > 0.5 * np.median(L[v]))
    return h[v] if v.sum() >= 4 else None


def v1(h):
    vv = h[:, 9] - h[:, 0]; L = np.linalg.norm(vv, axis=-1) + 1e-6
    c, s = vv[:, 1] / L, vv[:, 0] / L
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    loc = np.clip(np.einsum('tij,tkj->tki', R, h - h[:, :1]) / L[:, None, None], -4, 4)
    width = np.linalg.norm(h[:, 5] - h[:, 17], axis=-1) / L
    ori = np.stack([vv[:, 0] / L, vv[:, 1] / L, width, np.log(L)], -1)
    return ori, loc[:, 1:].reshape(len(h), -1)


def make_template(H):
    P = []
    for h in H:
        _, loc = v1(h)
        P.append(np.concatenate([np.zeros((len(h), 1, 2)), loc.reshape(len(h), 20, 2)], 1)[:, PALM])
    return np.median(np.concatenate(P), 0)   # (6,2) canonical palm, wrist at origin


def v2(h, tmpl):
    X = h[:, PALM] - h[:, :1]                               # T,6,2 observed palm (wrist origin)
    # least squares A: X ~ tmpl @ A^T  ->  A^T = pinv(tmpl) @ X
    At = np.einsum('ij,tjk->tik', np.linalg.pinv(tmpl), X)   # T,2,2
    A = np.transpose(At, (0, 2, 1))
    # polar-ish decomposition: rotation angle + log singular values + shear proxy
    U, S, Vt = np.linalg.svd(A)
    Rm = U @ Vt
    ang = np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0])
    det_sign = np.sign(np.linalg.det(A))
    ori = np.stack([np.cos(ang), np.sin(ang), np.log(S[:, 0] + 1e-6), np.log(S[:, 1] + 1e-6), det_sign], -1)
    Ainv = np.linalg.pinv(A)
    F = np.einsum('tij,tkj->tki', Ainv, h[:, FING] - h[:, :1])
    return ori, np.clip(F, -4, 4).reshape(len(h), -1)


def frames(H, fn):
    O, S, G = [], [], []
    for i, h in enumerate(H):
        o, s = fn(h); O.append(o); S.append(s); G.append(np.full(len(h), i))
    return np.concatenate(O), np.concatenate(S), np.concatenate(G)


def r2(Xtr, Ytr, Xte, Yte, mdl):
    sx, sy = StandardScaler().fit(Xtr), StandardScaler().fit(Ytr)
    mdl.fit(sx.transform(Xtr), sy.transform(Ytr))
    return float(r2_score(np.clip(sy.transform(Yte), -5, 5), np.clip(mdl.predict(sx.transform(Xte)), -5, 5)))


def clipdesc(H, fn):
    out = []
    for h in H:
        o, s = fn(h)
        idx = np.linspace(0, len(h) - 1, 8).round().astype(int)
        out.append(np.concatenate([o[idx].reshape(-1), s[idx].reshape(-1)]))
    return np.stack(out)


def main():
    tr = pickle.load(open(f'{D}/train14k.pkl', 'rb')); va = pickle.load(open(f'{D}/val.pkl', 'rb'))
    Htr, gtr = [], []
    for x in tr:
        h = dominant(x)
        if h is not None:
            Htr.append(h); gtr.append(x['gloss'])
    Hva, gva = [], []
    for x in va:
        h = dominant(x)
        if h is not None:
            Hva.append(h); gva.append(x['gloss'])
    tmpl = make_template(Htr[:3000])
    res = {'template': tmpl.round(3).tolist()}
    rng = np.random.default_rng(0)
    for name, fn in [('v1_similarity', v1), ('v2_affine', lambda h: v2(h, tmpl))]:
        Otr, Str, _ = frames(Htr, fn); Ova, Sva, _ = frames(Hva, fn)
        a = rng.choice(len(Otr), min(200000, len(Otr)), replace=False); b = rng.choice(len(Ova), min(100000, len(Ova)), replace=False)
        res[name] = dict(
            leak_shape_from_ori_ridge=round(r2(Otr[a], Str[a], Ova[b], Sva[b], Ridge(1.0)), 4),
            leak_shape_from_ori_knn20=round(r2(Otr[a], Str[a], Ova[b], Sva[b], KNeighborsRegressor(20)), 4))
        Ctr, Cva = clipdesc(Htr, fn), clipdesc(Hva, fn)
        sc = StandardScaler().fit(Ctr); A = sc.transform(Ctr); B = sc.transform(Cva)
        A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
        ga, gb = np.array(gtr), np.array(gva); keep = np.isin(gb, ga)
        res[name]['handonly_gloss_1nn'] = round(float((ga[(B[keep] @ A.T).argmax(1)] == gb[keep]).mean()), 4)
        print(name, res[name], flush=True)
    json.dump(res, open(f'{D}/diag3_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
