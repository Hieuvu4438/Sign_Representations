"""D7: hand configuration in palm frame (rotation-canonicalised) vs image/body frame; and face channel subsets.
Same pipeline as diag_v2.py; only hand_conf's frame changes. OFF-val R@1 + signer probes + finger-bone interpolation R^2."""
import sys, json
import numpy as np
import diag_v2 as V
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
D = sys.argv[1]
orig = V.hand_conf
def hand_conf_img(h, sigma, bone_norm, rel_scale, valid):
    v = h[:, 9] - h[:, 0]; L = np.linalg.norm(v, axis=-1); Leff = np.maximum(L, 0.1 * sigma)
    u = v / (L[:, None] + 1e-9)
    P = (h - h[:, :1]) / Leff[:, None, None]                     # NO rotation: body/image-aligned frame
    a5, a17 = h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]
    sign = np.sign(a5[:, 0] * a17[:, 1] - a5[:, 1] * a17[:, 0])
    logs = np.log(Leff / sigma)
    if rel_scale and valid.sum() > 0: logs = logs - np.median(logs[valid])
    fore = 1.0 - L / Leff
    bones = np.stack([P[:, b] - P[:, a] for a, b in V.FINGER_BONES], 1)
    if bone_norm and valid.sum() > 0:
        bl = np.linalg.norm(bones, axis=-1); bones = bones / (np.percentile(bl[valid], 90, axis=0) + 1e-6)[None, :, None]
    conf = np.concatenate([u, sign[:, None], logs[:, None], fore[:, None], P[:, V.PALM].reshape(len(h), -1), bones.reshape(len(h), -1)], -1)
    return np.clip(conf, -6, 6)
if __name__ == '__main__':
    HAND = ['D_CONF', 'D_LOC', 'N_CONF', 'N_LOC']
    res = {}
    for frame, fn in [('palm', orig), ('image', hand_conf_img)]:
        V.hand_conf = fn
        for bn in [False, True]:
            tr = V.build(f'{D}/train_full.pkl', bone_norm=bn, rel_scale=True, neutral=True)
            va = V.build(f'{D}/val.pkl', bone_norm=bn, rel_scale=True, neutral=True)
            for keys, tag in [(HAND, 'hands'), (HAND + ['F_UP', 'F_MO'], 'hands+UP+MO'), (['D_CONF'], 'D_CONF')]:
                Xtr = np.stack([V.descriptor(o, keys) for o in tr]); Xva = np.stack([V.descriptor(o, keys) for o in va])
                r = dict(signer5=V.signer_probe(va, Xva), signer3_balanced=V.signer_probe(va, Xva, True), offval_R1=V.off_retrieval(tr, va, Xtr, Xva))
                res[f'{frame}|bn={bn}|{tag}'] = r; print(frame, 'bn', bn, tag, r, flush=True)
            # frame-level finger-bone interpolation R^2 (gap 1, 4)
            for gap in [1, 4]:
                P_, Y_ = [], []
                for o in va:
                    S = o['ch']['D_CONF'][:, 13:43]; t = np.where(o['val']['D_CONF'])[0]; ok = set(t)
                    for tt in t:
                        if tt - gap in ok and tt + gap in ok:
                            P_.append((S[tt - gap] + S[tt + gap]) / 2); Y_.append(S[tt])
                P_, Y_ = np.array(P_), np.array(Y_); sc = StandardScaler().fit(Y_)
                res[f'{frame}|bn={bn}|interp_gap{gap}'] = round(float(r2_score(np.clip(sc.transform(Y_), -5, 5), np.clip(sc.transform(P_), -5, 5))), 4)
                print(frame, bn, 'gap', gap, res[f'{frame}|bn={bn}|interp_gap{gap}'], flush=True)
    json.dump(res, open(f'{D}/diag_v2b_results.json', 'w'), indent=1)
