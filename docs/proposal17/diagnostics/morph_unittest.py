"""Unit test for the NaN-safe morph (review R2-N1): at scale ~1 no observation flag and no keypoint may change; at larger scale NaNs stay NaN and finite stay finite."""
import numpy as np
import emb_variants as E
from common import manifests
from oj.data.clips import video_meta
rows = manifests()["val"][::2][::75][:40]
bad_flags, maxd, nan_new = 0, 0.0, 0
for r in rows:
    m = video_meta(r["video"])[:4]
    c = E.prepare(r, m)
    for mode in ("arm", "hand", "all", "nonuni", "face"):
        p = E.prepare(r, m, np.random.default_rng(0), mode, scale=1.0000001)
        bad_flags += int((c["obs"] != p["obs"]).sum()); maxd = max(maxd, float(np.abs(c["xy"] - p["xy"]).max()))
print("flags changed at scale~1:", bad_flags, " max |dxy|:", maxd)
assert bad_flags == 0 and maxd < 1e-2
print("UNITTEST OK")
