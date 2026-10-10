"""Debug why the H004 Aggregator does not fit (supervised CE on ASL Citizen train windows)."""
import sys, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from h004_train import Aggregator, load_seqs
from sup_aggregator import CosineHead
dev = "cuda"
tc, TX, TA, TM = load_seqs("aslc_train_s2")
gl_ = sorted({c["gloss"] for c in tc}); lab = {g: i for i, g in enumerate(gl_)}
y = torch.from_numpy(np.array([lab[c["gloss"]] for c in tc])).to(dev)
X, A, M = torch.from_numpy(TX).to(dev), torch.from_numpy(TA).to(dev), torch.from_numpy(TM).to(dev)
def run(name, model, amp, lr=3e-4, steps=800):
    torch.manual_seed(0); head = CosineHead(768 if not hasattr(model, "outdim") else model.outdim, len(gl_)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=0.0); rng = np.random.default_rng(0)
    for s in range(1, steps + 1):
        idx = torch.from_numpy(rng.choice(len(y), 256, replace=False)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            z = model(X[idx], A[idx], M[idx])
        loss = F.cross_entropy(head(z.float()), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    gn = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"{name:28s} loss@{steps} {loss.item():.3f} grad-norm {gn.item():.3e}", flush=True)
class MeanMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f = nn.Sequential(nn.Linear(768, 768))
    def forward(self, x, a, m):
        w = (m.float())[..., None]; return F.normalize(self.f((x.float() * w).sum(1) / w.sum(1)), dim=-1)
run("mean+linear (sanity)", MeanMLP().to(dev), False)
run("Aggregator fp32", Aggregator(out=768).to(dev), False)
run("Aggregator bf16", Aggregator(out=768).to(dev), True)
run("Aggregator fp32 lr1e-3", Aggregator(out=768).to(dev), False, lr=1e-3)
