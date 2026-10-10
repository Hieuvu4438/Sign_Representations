"""Synthetic predictive-dependence fixture (proposal5 §16.3): part 1's future copies part 0's
recent past, other parts are independent noise. Expected: G1 << G0 ≈ Gsh on edge 0->1,
and no gain on an edge with no dependence (2->3). CPU, ~1-2 min."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p4 import cip_forecast as CF  # noqa: E402
from cip_gate import evaluate, eval_set, train_one  # noqa: E402

rng = np.random.default_rng(0)
nv, T, D = 300, 40, 102
S = np.zeros((nv * T, 4, D), np.float32)
V = np.zeros((nv * T, 4, D), bool)
V[:, :, :8] = True
off = np.arange(nv + 1) * T
for v in range(nv):
    walk = np.cumsum(rng.normal(0, 0.3, (T, 8)), 0)
    S[v * T:(v + 1) * T, 0, :8] = walk
    S[v * T:(v + 1) * T, 1, :8] = np.r_[np.zeros((4, 8)), walk[:-4]] + rng.normal(0, 0.05, (T, 8))  # lag-4 copy
    S[v * T:(v + 1) * T, 2:, :8] = rng.normal(0, 1, (T, 2, 8))
tr, va, te = np.arange(0, 200), np.arange(200, 230), np.arange(230, 300)
models = {m: train_one(m, S, V, off, tr, va, 0, 600, "cpu")[0] for m in ("g0", "g1", "gsh")}
R = evaluate(models, S, V, off, eval_set(S, V, off, te, rng, 3), "cpu")
for name, (a, b) in (("0->1 (dependent)", (0, 1)), ("2->3 (independent)", (2, 3))):
    ei = CF.EDGES.index((a, b))
    x = R[(R[:, 1] == ei) & (R[:, 2] == 1)]  # horizon 4
    print(name, "g0 %.3f g1 %.3f gsh %.3f persist %.3f" % tuple(x[:, 3:7].mean(0)))
x = R[(R[:, 1] == CF.EDGES.index((0, 1))) & (R[:, 2] == 1)][:, 3:6].mean(0)
y = R[(R[:, 1] == CF.EDGES.index((2, 3))) & (R[:, 2] == 1)][:, 3:6].mean(0)
assert x[1] < 0.7 * x[0] and x[1] < 0.7 * x[2], x
assert abs(y[1] - y[0]) < 0.15 * y[0], y
print("CIP FIXTURE OK")
