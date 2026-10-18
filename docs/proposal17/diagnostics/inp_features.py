"""Input-feature control (proposal17 diagnostic G15): per-clip mean of the PLctx INPUT tokens (before any encoder), per channel,
for the same train/val every-2nd subsets and windows as emb_variants.py.  Dims per channel: CONF 170, LOC 16, UP 108, MO 126, HD 20.
Gives the 'input-feature ceiling' (whiten / LDA R@1) so that the value added by the trained encoder is measurable."""
import sys, time, numpy as np, torch
sys.path.insert(0, "/home/haipd/Sign_Representations/method/phonojepa"); sys.path.insert(0, "/home/haipd/Sign_Representations/method/obs_jepa")
from pj import features as Fe
from pj.joints import NC
from oj.data.clips import video_meta
import emb_variants as E
from common import SCRATCH, manifests
torch.set_num_threads(4)
def run(split):
    rows = manifests()[split][::2]; metas = [video_meta(r["video"])[:4] for r in rows]
    dl = torch.utils.data.DataLoader(E.DS(rows, metas), batch_size=None, num_workers=3, prefetch_factor=4,
        worker_init_fn=lambda _: (torch.set_num_threads(1), __import__("cv2").setNumThreads(1)))
    out = np.zeros((len(rows), 626), np.float32); t0 = time.time()
    for i, p in enumerate(dl):
        p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in p.items()}
        xy, obs, conf, valid = [torch.as_tensor(p[k]) for k in ("xy", "obs", "conf", "valid")]
        f = Fe.frame_features(xy, obs, conf, torch.zeros_like(obs)); ins, pres = Fe.tokens(f, valid)
        B = xy.shape[0]; vb = valid.view(B, -1, 2).any(-1)
        feats = []
        for c in range(NC):
            w = (pres[:, c, :] & vb).float()[..., None]               # [B, TB, 1]
            feats.append(((ins[c] * w).sum(1) / w.sum(1).clamp(min=1)))
        out[i] = torch.cat(feats, -1).mean(0).numpy()
        if i % 1000 == 0: print(split, i, len(rows), f"{time.time()-t0:.0f}s", flush=True)
    np.savez(f"{SCRATCH}/emb/{split}_inputs.npz", inp=out.astype(np.float16), ids=np.array([r["sample_id"] for r in rows]))
for s in ("val", "train"): run(s)
print("INPDONE")
