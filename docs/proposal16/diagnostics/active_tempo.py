"""proposal16 diagnostic: signer vs gloss share of *active signing* duration and of movement speed.
Active frame = dominant wrist above its clip-rest height (lowest 10% height + 0.25 shoulder widths) OR moving fast."""
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
x, lens, meta = z["x"], z["lens"], json.loads(str(z["meta"]))
off = np.concatenate([[0], np.cumsum(lens)])
act, spd, apl, fr = [], [], [], []
for i in range(len(lens)):
    c = np.nan_to_num(x[off[i]:off[i+1]].astype(np.float64))
    fps = meta[i]["fps"]
    w = c[:, 5]  # dominant body wrist
    h = w[:, 1]
    rest = np.percentile(h, 10)
    v = np.r_[0, np.linalg.norm(np.diff(w, axis=0), axis=1)] * fps
    a = (h > rest + 0.25) | (v > 1.0)
    act.append(a.sum() / fps); fr.append(a.mean())
    spd.append(np.median(v[a]) if a.any() else 0.0)
    # path length of dominant wrist during active frames (shoulder widths)
    apl.append(float(np.linalg.norm(np.diff(w[a], axis=0), axis=1).sum()) if a.sum() > 1 else 0.0)
g = np.unique([m["gloss"] for m in meta], return_inverse=True)[1]
s = np.unique([m["signer_id"] for m in meta], return_inverse=True)[1]
def decomp(y, name):
    y = np.asarray(y, float); mu = y.mean()
    def means(idx, v): return (np.bincount(idx, v) / np.bincount(idx))[idx]
    def r2(p): return 1 - ((y - p) ** 2).sum() / ((y - mu) ** 2).sum()
    a = np.zeros_like(y); b = np.zeros_like(y)
    for _ in range(50):
        a = means(g, y - mu - b); b = means(s, y - mu - a)
    print(f"{name:28s} R2 gloss={r2(means(g,y)):.3f} signer={r2(means(s,y)):.3f} both={r2(mu+a+b):.3f}  "
          f"p50={np.median(y):.3f}")
decomp(np.log(np.maximum(act, 0.1)), "log active duration")
decomp(np.log(np.array([m['duration_sec'] for m in meta])), "log clip duration")
decomp(fr, "active fraction")
decomp(np.log(np.maximum(spd, 1e-2)), "log median wrist speed")
decomp(np.log(np.maximum(apl, 1e-2)), "log active wrist path len")
print("n", len(meta), "active sec p5/50/95", np.percentile(act, [5, 50, 95]).round(2), "active frac p50", np.median(fr).round(2))
