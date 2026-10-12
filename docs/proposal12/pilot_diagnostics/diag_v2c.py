"""D7b: anchor normalisation per frame (c_t, sigma_t) vs clip-level median anchors (both leak-free: anchors are never masked).
Hand CONF in image frame + per-bone p90 (D7 best). Hands-only descriptor; OFF-val R@1 + signer probes."""
import sys, json
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
import diag_v2 as V
import diag_v2b as B   # noqa: F401  (defines hand_conf_img; its main loop guarded? no -> avoid running)
D = sys.argv[1]
V.hand_conf = B.hand_conf_img
orig_clip = V.clip_channels
def clip_channels_clipanchor(x, **kw):
    kp = x['kp'].astype(np.float32)
    c = np.median((kp[:, V.LSHO] + kp[:, V.RSHO]) / 2, 0)
    s = np.median(np.linalg.norm(kp[:, V.LSHO] - kp[:, V.RSHO], axis=-1))
    # replace shoulders by their clip medians (rigid), so c_t, sigma_t become constant per clip
    x2 = dict(x); kp2 = kp.copy()
    kp2[:, V.LSHO] = c + np.array([s / 2, 0.0]); kp2[:, V.RSHO] = c - np.array([s / 2, 0.0])
    x2['kp'] = kp2
    return orig_clip(x2, **kw)
HAND = ['D_CONF', 'D_LOC', 'N_CONF', 'N_LOC']
res = {}
for mode, fn in [('frame', orig_clip), ('clip', clip_channels_clipanchor)]:
    V.clip_channels = fn
    tr = V.build(f'{D}/train_full.pkl', bone_norm=True, rel_scale=True, neutral=True)
    va = V.build(f'{D}/val.pkl', bone_norm=True, rel_scale=True, neutral=True)
    for keys, tag in [(HAND, 'hands'), (['D_LOC', 'N_LOC'], 'LOC')]:
        Xtr = np.stack([V.descriptor(o, keys) for o in tr]); Xva = np.stack([V.descriptor(o, keys) for o in va])
        r = dict(signer5=V.signer_probe(va, Xva), signer3_balanced=V.signer_probe(va, Xva, True), offval_R1=V.off_retrieval(tr, va, Xtr, Xva))
        res[f'{mode}|{tag}'] = r; print(mode, tag, r, flush=True)
json.dump(res, open(f'{D}/diag_v2c_results.json', 'w'), indent=1)
