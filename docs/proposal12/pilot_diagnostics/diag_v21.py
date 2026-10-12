"""Pilot diagnostics v2.1 (after second review). Training-free, CPU only.

Changes vs v2:
  * pixel-aspect correction (x *= W/H per video; ASL Citizen is mostly 640x480, some 960x540)
  * body frame = rotate the whole skeleton by the clip-median shoulder-line angle (anchor-only, leak-free)
  * chin (iBUG 8) moved from F-HEAD to F-MOUTH (jaw opening belongs to the mouth channel)
  * clip-median shoulder anchors (D7b)
  * D6'': per-column (max / mean) R^2 of every target channel from every source channel, pooled AND within-clip
          (per-clip demeaned), for the default body-frame tokenisation
  * D7': palm frame vs body frame after aspect correction (OFF-val R@1, signer probes, finger-bone interpolation R^2)
Usage: python diag_v21.py OUT_DIR   (OUT_DIR holds val.pkl, train_full.pkl with 'stem', aspect.json)
"""
import json, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
sys.argv = [sys.argv[0], sys.argv[1]]
import diag_v2 as V

D = sys.argv[1]
ASPECT = json.load(open(f'{D}/aspect.json'))
MOUTH21 = list(range(48, 68)) + [8]


def hand_conf(h, sigma, frame, valid):
    v = h[:, 9] - h[:, 0]; L = np.linalg.norm(v, axis=-1); Leff = np.maximum(L, 0.1 * sigma)
    u = v / (L[:, None] + 1e-9)
    if frame == 'palm':
        R = np.stack([np.stack([u[:, 1], -u[:, 0]], -1), np.stack([u[:, 0], u[:, 1]], -1)], -2)
        P = np.einsum('tij,tkj->tki', R, h - h[:, :1]) / Leff[:, None, None]
    else:
        P = (h - h[:, :1]) / Leff[:, None, None]
    a5, a17 = h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]
    sign = np.sign(a5[:, 0] * a17[:, 1] - a5[:, 1] * a17[:, 0])
    logs = np.log(Leff / sigma)
    if valid.sum() > 0:
        logs = logs - np.median(logs[valid])
    fore = 1.0 - L / Leff
    bones = np.stack([P[:, b] - P[:, a] for a, b in V.FINGER_BONES], 1)
    if valid.sum() > 0:
        bl = np.linalg.norm(bones, axis=-1); bones = bones / (np.percentile(bl[valid], 90, axis=0) + 1e-6)[None, :, None]
    return np.clip(np.concatenate([u, sign[:, None], logs[:, None], fore[:, None], P[:, V.PALM].reshape(len(h), -1),
                                   bones.reshape(len(h), -1)], -1), -6, 6)


def face_feats(kp, c, sigma, valid):
    f = kp[:, V.F0:V.F0 + 68]
    eyeR, eyeL = f[:, [36, 39]].mean(1), f[:, [42, 45]].mean(1)
    o = (eyeR + eyeL) / 2; ax = eyeL - eyeR; iod = np.linalg.norm(ax, axis=-1) + 1e-9; ex = ax / iod[:, None]
    R = np.stack([ex, np.stack([-ex[:, 1], ex[:, 0]], -1)], -2)
    Fe = np.einsum('tij,tkj->tki', R, f - o[:, None]) / iod[:, None, None]
    up, mo = Fe[:, V.UPPER].reshape(len(f), -1), Fe[:, MOUTH21].reshape(len(f), -1)
    org = (o - c) / sigma[:, None]; lsc = np.log(iod / sigma)[:, None]; nose = Fe[:, 30]
    if valid.sum() > 0:
        up = up - np.median(up[valid], 0); mo = mo - np.median(mo[valid], 0)
        org = org - np.median(org[valid], 0); lsc = lsc - np.median(lsc[valid], 0); nose = nose - np.median(nose[valid], 0)
    hd = np.concatenate([org, lsc, ex, nose], -1)                                   # 7 dims, all relative to clip median
    return np.clip(up, -6, 6), np.clip(mo, -6, 6), np.clip(hd, -6, 6)


def clip_channels(x, frame='body', aspect=True):
    kp = x['kp'].astype(np.float32).copy(); sc = x['sc'].astype(np.float32)
    if aspect:
        kp[..., 0] *= ASPECT.get(x['stem'], 4 / 3)
    cx = np.median((kp[:, V.LSHO, 0] + kp[:, V.RSHO, 0]) / 2)
    bw = sc[:, [V.LWRI, V.RWRI]] > 1.0
    ml = np.abs(np.diff(kp[:, V.LWRI], axis=0)).sum(-1)[bw[1:, 0] & bw[:-1, 0]].sum()
    mr = np.abs(np.diff(kp[:, V.RWRI], axis=0)).sum(-1)[bw[1:, 1] & bw[:-1, 1]].sum()
    if ml > mr:
        kp, sc = V.mirror_clip(kp, sc, cx)
    # clip-median anchors; body frame = rotate by median shoulder-line angle (anchor-only)
    c = np.median((kp[:, V.LSHO] + kp[:, V.RSHO]) / 2, 0)
    sv = kp[:, V.LSHO] - kp[:, V.RSHO]; sigma0 = np.median(np.linalg.norm(sv, axis=-1)) + 1e-9
    ang = np.arctan2(np.median(sv[:, 1]), np.median(sv[:, 0]))
    if frame == 'body':
        ca, sa = np.cos(-ang), np.sin(-ang); Rb = np.array([[ca, -sa], [sa, ca]])
        kp = (kp - c) @ Rb.T + c
    T = len(kp); cc = np.repeat(c[None], T, 0); sigma = np.full(T, sigma0)
    out = {'signer': x['signer'], 'gloss': x['gloss'], 'stem': x['stem']}
    ch, val = {}, {}
    for side, H0, W, E in [('D', V.RH0, V.RWRI, V.RELB), ('N', V.LH0, V.LWRI, V.LELB)]:
        hv = sc[:, H0:H0 + 21].mean(1) > V.HTHR
        ch[side + '_CONF'] = hand_conf(kp[:, H0:H0 + 21], sigma, 'palm' if frame == 'palm' else 'body', hv)
        w, e = (kp[:, W] - cc) / sigma0, (kp[:, E] - cc) / sigma0
        vel = np.vstack([np.zeros((1, 2)), np.diff(w, axis=0)])
        ch[side + '_LOC'] = np.clip(np.concatenate([w, e, vel], -1), -6, 6)
        val[side + '_CONF'] = hv; val[side + '_LOC'] = sc[:, [W, E]].min(1) > 1.0
    fv = sc[:, V.F0:V.F0 + 68].mean(1) > V.FTHR
    ch['F_UP'], ch['F_MO'], ch['F_HD'] = face_feats(kp, cc, sigma, fv)
    for k in ['F_UP', 'F_MO', 'F_HD']:
        val[k] = fv
    if val['D_CONF'].sum() < 4:
        return None
    out['ch'], out['val'] = ch, val
    return out


def build(path, **kw):
    import pickle
    data = pickle.load(open(path, 'rb'))
    return [o for o in (clip_channels(x, **kw) for x in data) if o is not None]


def interp_r2(F, gap):
    P, Y = [], []
    for o in F:
        S = o['ch']['D_CONF'][:, 13:43]; t = np.where(o['val']['D_CONF'])[0]; ok = set(t)
        for tt in t:
            if tt - gap in ok and tt + gap in ok:
                P.append((S[tt - gap] + S[tt + gap]) / 2); Y.append(S[tt])
    P, Y = np.array(P), np.array(Y); sc = StandardScaler().fit(Y)
    return round(float(r2_score(np.clip(sc.transform(Y), -5, 5), np.clip(sc.transform(P), -5, 5))), 4)


def main():
    HAND = ['D_CONF', 'D_LOC', 'N_CONF', 'N_LOC']; FACE = ['F_UP', 'F_MO', 'F_HD']
    res = {'D7p': {}, 'D1p': {}, 'D6pp': {}}
    store = {}
    for frame in ['palm', 'body']:
        tr = build(f'{D}/train_full.pkl', frame=frame); va = build(f'{D}/val.pkl', frame=frame)
        store[frame] = (tr, va)
        for keys, tag in [(HAND, 'hands'), (['D_CONF'], 'D_CONF'), (HAND + FACE, 'hands+face'), (HAND + ['F_UP', 'F_MO'], 'hands+UP+MO')]:
            if frame == 'palm' and tag.startswith('hands+'):
                continue
            Xtr = np.stack([V.descriptor(o, keys) for o in tr]); Xva = np.stack([V.descriptor(o, keys) for o in va])
            r = dict(signer5=V.signer_probe(va, Xva), signer3_balanced=V.signer_probe(va, Xva, True), offval_R1=V.off_retrieval(tr, va, Xtr, Xva))
            (res['D7p'] if not tag.startswith('hands+') else res['D1p'])[f'{frame}|{tag}'] = r
            print(frame, tag, r, flush=True)
        res['D7p'][f'{frame}|interp'] = [interp_r2(va, g) for g in (1, 2, 4)]
        print(frame, 'interp g1/2/4', res['D7p'][f'{frame}|interp'], flush=True)
    # D6'': per-column leak audit on body frame, pooled and within-clip
    tr, va = store['body']
    keys = HAND + FACE
    def frames(F, n, seed, demean):
        rows = {k: [] for k in keys}
        for o in F:
            ok = np.ones(len(o['ch']['D_CONF']), bool)
            for k in keys:
                ok &= o['val'][k]
            if ok.sum() < 3:
                continue
            for k in keys:
                a = o['ch'][k][ok]
                rows[k].append(a - a.mean(0) if demean else a)
        rows = {k: np.concatenate(v) for k, v in rows.items()}
        m = len(rows[keys[0]]); idx = np.random.default_rng(seed).choice(m, min(n, m), replace=False)
        return {k: v[idx] for k, v in rows.items()}
    for demean in [False, True]:
        tr6, va6 = frames(tr, 150000, 0, demean), frames(va, 40000, 1, demean)
        for k in keys:
            lo, hi = np.percentile(tr6[k], 1, 0), np.percentile(tr6[k], 99, 0)
            tr6[k], va6[k] = np.clip(tr6[k], lo, hi), np.clip(va6[k], lo, hi)
        for t in keys:
            sy = StandardScaler().fit(tr6[t]); Yt, Yv = sy.transform(tr6[t]), np.clip(sy.transform(va6[t]), -5, 5)
            ok_cols = sy.scale_ > 1e-6
            for s in keys:
                if s == t:
                    continue
                sx = StandardScaler().fit(tr6[s]); Xt, Xv = sx.transform(tr6[s]), sx.transform(va6[s])
                P = np.clip(Ridge(1.0).fit(Xt, Yt).predict(Xv), -5, 5)
                per = r2_score(Yv[:, ok_cols], P[:, ok_cols], multioutput='raw_values')
                res['D6pp'][f'{"within" if demean else "pooled"}|{s}->{t}'] = dict(mean=round(float(per.mean()), 3), max=round(float(per.max()), 3),
                                                                                  argmax=int(np.flatnonzero(ok_cols)[per.argmax()]))
        print('D6pp done demean', demean, flush=True)
    json.dump(res, open(f'{D}/diag_v21_results.json', 'w'), indent=1)
    print(json.dumps(res['D6pp']), flush=True)


if __name__ == '__main__':
    main()
