"""EXPLORATORY: leave-one-signer-out retrieval inside ASL Citizen val (gallery = other 5 val
signers). Used only to orient diagnostics while the train gallery is being extracted."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores
d = Path(sys.argv[1])
clips = [json.loads(l) for l in open(d / "clips.jsonl")]
off = np.load(d / "offsets.npy"); act = np.load(d / "windows_active.npy")
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
res = {}
for key in ("features", "latent"):
    win = np.load(d / f"windows_{key}.npy")
    for pool in ("activity", "mean"):
        p, fb = pool_clips(win, off, activity_weights(act) if pool == "activity" else None)
        p = l2n(p); ranks = np.zeros(len(y), int)
        for s in np.unique(sg):
            q, g = sg == s, sg != s
            ranks[q] = ranks_from_scores(class_max_scores(p[q], p[g], y[g], len(gl_)), y[q])
        res[f"{key}|{pool}"] = metrics_from_ranks(ranks, y)
        print(key, pool, json.dumps(res[f"{key}|{pool}"]), flush=True)
