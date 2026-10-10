"""R002/E6 (SUPERVISED reference, val only): attentive-pooling transformer over the SAME frozen
SignRep window embeddings, trained with ASL Citizen TRAIN gloss labels (cross-entropy). Its pooled
embedding is evaluated with the official retrieval protocol (train gallery -> val queries), and its
classifier top-1 on val is reported. This is labelled supervision; it is a reference point for the
training-free readouts (R0 59.16 / R1 64.91 / R2 68.40 R@1 on val), not a label-free method."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h004_train import Aggregator, embed, load_seqs  # noqa: E402
from p4.retrieval import class_max_scores, metrics_from_ranks, ranks_from_scores  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


class CosineHead(nn.Module):
    """cosine classifier: s * <z, W_c/|W_c|> (z is already L2-normalised); fixes the
    vanishing-logit failure of a plain Linear head on unit-norm inputs (first run, INVALID)."""

    def __init__(self, d, c, s=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(c, d) * 0.01); self.s = s

    def forward(self, z):
        return self.s * F.normalize(z, dim=-1) @ F.normalize(self.W, dim=-1).T


def linear_probe_reference(dev):
    """sanity reference: cosine classifier on the frozen active-span weighted mean (no aggregator)."""
    from p4.retrieval import l2n
    tc, TX, TA, TM = load_seqs("aslc_train_s2"); vc, VX, VA, VM = load_seqs("aslc_val_s2")
    wm = lambda X, A, M: l2n((X.astype(np.float32) * ((A * M)[..., None] + 1e-6 * M[..., None])).sum(1))
    gl_ = sorted({c["gloss"] for c in tc}); lab = {g: i for i, g in enumerate(gl_)}
    ty = torch.from_numpy(np.array([lab[c["gloss"]] for c in tc])).to(dev); vy = np.array([lab[c["gloss"]] for c in vc])
    Xt, Xv = torch.from_numpy(wm(TX, TA, TM)).to(dev), torch.from_numpy(wm(VX, VA, VM)).to(dev)
    head = CosineHead(768, len(gl_)).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    rng = np.random.default_rng(0)
    for step in range(3000):
        idx = torch.from_numpy(rng.choice(len(ty), 512, replace=False)).to(dev)
        loss = F.cross_entropy(head(Xt[idx]), ty[idx], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rc = ranks_from_scores(head(Xv).cpu().numpy(), vy)
    return metrics_from_ranks(rc, vy), float(loss.item())


def main(steps=3000, batch=256, seed=0):
    torch.manual_seed(seed); dev = "cuda"
    tc, TX, TA, TM = load_seqs("aslc_train_s2"); vc, VX, VA, VM = load_seqs("aslc_val_s2")
    gl_ = sorted({c["gloss"] for c in tc}); lab = {g: i for i, g in enumerate(gl_)}
    ty = np.array([lab[c["gloss"]] for c in tc]); vy = np.array([lab[c["gloss"]] for c in vc])
    model = Aggregator(out=768).to(dev)
    head = CosineHead(768, len(gl_)).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=0.05)
    Xg, Ag, Mg = torch.from_numpy(TX).to(dev), torch.from_numpy(TA).to(dev), torch.from_numpy(TM).to(dev)
    yg = torch.from_numpy(ty).to(dev); rng = np.random.default_rng(seed); t0 = time.time()
    for step in range(1, steps + 1):
        lr = 3e-4 * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.from_numpy(rng.choice(len(ty), batch, replace=False)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(Xg[idx], Ag[idx], Mg[idx])
            loss = F.cross_entropy(head(z.float()), yg[idx], label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(step, round(loss.item(), 3), round(time.time() - t0), flush=True)
    model.eval(); head.eval()
    G, Q = embed(model, TX, TA, TM, dev), embed(model, VX, VA, VM, dev)
    r = ranks_from_scores(class_max_scores(Q, G, ty, len(gl_)), vy)
    with torch.no_grad():
        logits = head(torch.from_numpy(Q).to(dev)).cpu().numpy()
    rc = ranks_from_scores(logits, vy)
    lp, lp_loss = linear_probe_reference(dev)
    res = {"supervision": "ASL Citizen train gloss labels (cross-entropy)", "linear_probe_on_wmean_val": lp, "final_train_loss": float(loss.item()),
           "retrieval_val": metrics_from_ranks(r, vy),
           "classifier_val": metrics_from_ranks(rc, vy), "steps": steps, "batch": batch, "seed": seed}
    (RT / "eval").mkdir(exist_ok=True)
    np.save(RT / "eval" / "sup_aggregator_ranks_val.npy", r)
    json.dump(res, open(RT / "eval" / "sup_aggregator_val.json", "w"), indent=1)
    print(json.dumps({k: (v["R@1"] if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
