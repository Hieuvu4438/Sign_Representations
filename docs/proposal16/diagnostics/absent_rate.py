"""Phase 0(b): share of P-token windows (8 / 16 samples at ~25 fps) whose support is observed, after the in-image +
hand-wrist filters of prep_pose.py. A window counts as present if every support joint is observed in all but at most
2 of its frames (the proposal's <= 2-sample interpolation rule).
Reported for all windows and for 'active' windows (dominant body wrist above the clip rest height + 0.25 shoulder widths
for at least half of the window), since rest frames often have the hand outside the image."""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
x, ob, lens, meta = z["x"], z["ob"], z["lens"], json.loads(str(z["meta"]))
off = np.concatenate([[0], np.cumsum(lens)])
SUP = {"D-CONF (21 D-hand joints)": list(range(7, 28)), "N-CONF (21 N-hand joints)": list(range(28, 49)),
       "D-LOC (D body wrist/elbow/shoulder + nose)": [0, 1, 3, 5], "N-LOC": [0, 2, 4, 6]}
res = {}
for W in (8, 16):
    cnt = {k: [0, 0, 0, 0] for k in SUP}
    for i in range(len(lens)):
        o = ob[off[i]:off[i + 1]]
        h = np.nan_to_num(x[off[i]:off[i + 1], 5, 1].astype(float))
        step = max(1, int(round(meta[i]["fps"] / 25.0)))
        o, h = o[::step], h[::step]
        T = len(o)
        if T < W:
            continue
        act = h > np.percentile(h, 10) + 0.25
        oks = {k: o[:, j].all(1) for k, j in SUP.items()}
        for s in range(0, T - W + 1, W // 2):
            a = act[s:s + W].mean() >= 0.5
            for k, ok in oks.items():
                p = (~ok[s:s + W]).sum() <= 2
                cnt[k][0] += int(p); cnt[k][1] += 1
                if a:
                    cnt[k][2] += int(p); cnt[k][3] += 1
    res[W] = {k: {"all": round(100 * c[0] / c[1], 1), "active": round(100 * c[2] / max(c[3], 1), 1),
                  "active_share": round(100 * c[3] / c[1], 1)} for k, c in cnt.items()}
print(json.dumps(res, indent=1))
