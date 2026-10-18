"""proposal16 diagnostic: how much of clip duration (tempo) is explained by signer vs gloss? (metadata only)"""
import json
import numpy as np
ROOT = "/home/haipd/Sign_Representations/method/asm_jepa/data/processed/aslc"
m = [json.loads(l) for l in open(f"{ROOT}/metadata_train.jsonl")]
y = np.log(np.array([r["duration_sec"] for r in m]))
g = np.unique([r["gloss"] for r in m], return_inverse=True)[1]
s = np.unique([r["signer_id"] for r in m], return_inverse=True)[1]
def r2(pred): return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def means(idx, v): return (np.bincount(idx, v) / np.bincount(idx))[idx]
mu = y.mean()
print("n", len(y), "glosses", g.max() + 1, "signers", s.max() + 1)
print("R2 gloss only  %.3f" % r2(means(g, y)))
print("R2 signer only %.3f" % r2(means(s, y)))
# additive two-way fixed effects by backfitting
a = np.zeros_like(y); b = np.zeros_like(y)
for _ in range(50):
    a = means(g, y - mu - b); b = means(s, y - mu - a)
print("R2 gloss+signer %.3f" % r2(mu + a + b))
sig_eff = np.exp(np.unique(np.c_[s, b], axis=0)[:, 1])
print("signer tempo factor (exp of signer effect) min/median/max %.2f %.2f %.2f" % (sig_eff.min(), np.median(sig_eff), sig_eff.max()))
print("duration sec p5/p50/p95", np.percentile(np.exp(y), [5, 50, 95]).round(2))
