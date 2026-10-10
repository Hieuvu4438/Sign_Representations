"""H004 / EX04-v1: label-free set-to-vector sign embedding on frozen SignRep windows.

--arm H4 : listwise distillation of chamfer similarities (teacher) into cos(g(X), g(Y))
--arm CM : same, teacher = activity-weighted-mean cosine (control)
--arm CI : instance discrimination with two temporal crops (control)
Train clips only, no labels/signers. Evaluates on the official gallery protocol at the end.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, ranks_from_scores  # noqa

RT = Path(__file__).resolve().parents[1] / "runtime"
L = 48


def load_seqs(name, thr=0.5):
    d = RT / "feats" / name
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    off = np.load(d / "offsets.npy")
    W = np.load(d / "windows_features.npy")
    G = activity_weights(np.load(d / "windows_active.npy"))
    n = len(off) - 1
    X = np.zeros((n, L, W.shape[1]), np.float16)
    A = np.zeros((n, L), np.float32)
    M = np.zeros((n, L), bool)
    for i in range(n):
        w, g = W[off[i]:off[i + 1]], G[off[i]:off[i + 1]]
        idx = np.where(g >= thr)[0]
        s, e = (idx[0], idx[-1] + 1) if len(idx) else (0, len(w))
        w, g = w[s:e], g[s:e]
        if len(w) > L:
            sel = np.round(np.linspace(0, len(w) - 1, L)).astype(int)
            w, g = w[sel], g[sel]
        k = len(w)
        X[i, :k] = l2n(w.astype(np.float32))
        A[i, :k] = g
        M[i, :k] = True
    return clips, X, A, M


class Aggregator(nn.Module):
    def __init__(self, din=768, d=512, out=1024, nq=4):
        super().__init__()
        self.in_norm = nn.LayerNorm(din)  # bug fix (2026-10-06): unit-norm windows made content ~30x smaller than pos/activity
        self.inp = nn.Linear(din, d)
        self.act = nn.Linear(1, d)
        self.pos = nn.Linear(16, d)
        el = nn.TransformerEncoderLayer(d, 8, 2048, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, 2)
        self.q = nn.Parameter(torch.randn(nq, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(nq * d, out)

    def forward(self, x, a, m):
        n = m.sum(1, keepdim=True).clamp(min=1).float()
        t = (torch.cumsum(m.float(), 1) - 1) / (n - 1).clamp(min=1)  # 0..1 position in span
        f = torch.arange(8, device=x.device).float()
        pe = torch.cat([torch.sin(t[..., None] * math.pi * 2 ** f / 2), torch.cos(t[..., None] * math.pi * 2 ** f / 2)], -1)
        h = self.inp(self.in_norm(x.float())) + self.act(a[..., None]) + self.pos(pe)
        if self.training and torch.is_grad_enabled():  # activation checkpointing: same math, less memory
            h = torch.utils.checkpoint.checkpoint(lambda hh: self.enc(hh, src_key_padding_mask=~m), h, use_reentrant=False)
        else:
            h = self.enc(h, src_key_padding_mask=~m)
        q = self.q[None].expand(len(x), -1, -1)
        z, _ = self.attn(q, self.norm(h), self.norm(h), key_padding_mask=~m)
        return F.normalize(self.head(z.flatten(1)), dim=-1)


class SoftSegPool(nn.Module):
    """EX04-v2: soft temporal segment pooling, initialised at the 2-bin readout."""

    def __init__(self, din=768, K=2):
        super().__init__()
        self.mu = nn.Parameter(torch.linspace(0.5 / K, 1 - 0.5 / K, K))
        self.logbeta = nn.Parameter(torch.tensor(math.log(40.0)))
        self.mlp = nn.Sequential(nn.Linear(din + 1, 128), nn.GELU(), nn.Linear(128, K))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, a, m):
        x = x.float()
        n = m.sum(1, keepdim=True).clamp(min=1).float()
        t = (torch.cumsum(m.float(), 1) - 0.5) / n  # bin-centre position in (0,1)
        logits = -self.logbeta.exp() * (t[..., None] - self.mu).abs() + self.mlp(torch.cat([x, t[..., None]], -1))
        w = F.softmax(logits, -1) * (a * m)[..., None]
        s = F.normalize(torch.einsum("blk,bld->bkd", w, x), dim=-1)
        return F.normalize(s.flatten(1), dim=-1)


def chamfer(xa, ma, xb, mb, mode="sym"):
    """xa [B,L,D], xb [B,K,L,D] -> [B,K] chamfer similarity (sym, or 'q2g' averaged over both
    query->gallery directions, i.e. 0.5*(a->b one-directional + b->a one-directional) = sym; 'q2g'
    here uses only the anchor->candidate direction, matching the val readout analysis)."""
    S = torch.einsum("bld,bkmd->bklm", xa.float(), xb.float())
    S1 = S.masked_fill(~mb[:, :, None, :], -9).amax(3)
    S2 = S.masked_fill(~ma[:, None, :, None], -9).amax(2)
    a = (S1 * ma[:, None]).sum(2) / ma.sum(1, keepdim=True).clamp(min=1)
    b = (S2 * mb).sum(2) / mb.sum(2).clamp(min=1)
    return a if mode == "q2g" else 0.5 * (a + b)


def tdiff(X, M, lag):
    """temporal differences of L2-normalised windows (static appearance cancels to first order).
    X [..., L, D], M [..., L] (valid prefix). Clips with <= lag windows fall back to last-minus-first."""
    D = X[..., lag:, :].float() - X[..., :-lag, :].float()
    MD = M[..., lag:] & M[..., :-lag]
    n = M.sum(-1)
    short = MD.sum(-1) == 0
    if short.any():
        idx_last = (n - 1).clamp(min=0)
        last = torch.gather(X.float(), -2, idx_last[..., None, None].expand(*idx_last.shape, 1, X.shape[-1])).squeeze(-2)
        fb = last - X[..., 0, :].float() + 1e-6
        D[..., 0, :] = torch.where(short[..., None], fb, D[..., 0, :])
        MD[..., 0] = MD[..., 0] | short
    D = F.normalize(D, dim=-1)
    pad = torch.zeros(*D.shape[:-2], lag, D.shape[-1], device=D.device)
    return torch.cat([D, pad], -2), torch.cat([MD, torch.zeros(*MD.shape[:-1], lag, dtype=torch.bool, device=MD.device)], -1)


def wmean_vec(X, A, M):
    w = (A * M)[..., None] + 1e-6 * M[..., None]
    return F.normalize((X.float() * w).sum(1), dim=-1)


def crop(x, a, m, rng_frac=0.6):
    """random contiguous crop of >= 60% of each span (labels-free augmentation)."""
    B = len(x)
    n = m.sum(1)
    out_m = torch.zeros_like(m)
    for i in range(B):
        k = int(n[i])
        c = max(1, int(math.ceil(k * (rng_frac + (1 - rng_frac) * torch.rand(1).item()))))
        s = int(torch.randint(0, k - c + 1, (1,)).item())
        out_m[i, s:s + c] = True
    # shift kept windows to the front
    idx = torch.argsort((~out_m).float() + torch.arange(L, device=x.device)[None] * 1e-3, dim=1)
    g = lambda t: torch.gather(t, 1, idx if t.dim() == 2 else idx[..., None].expand_as(t))
    return g(x), g(a), g(out_m)


@torch.no_grad()
def embed(model, X, A, M, dev, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(A[i:i + bs]).to(dev),
                         torch.from_numpy(M[i:i + bs]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["H4", "CM", "CI"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student", default="transformer", choices=["transformer", "softseg"])
    ap.add_argument("--teacher", default="sym", choices=["sym", "q2g", "q2g+diff"])
    ap.add_argument("--diff-lag", type=int, default=4)
    ap.add_argument("--train-feats", default="aslc_train_s2", help="unlabelled training windows (= retrieval gallery)")
    ap.add_argument("--query-feats", default="aslc_val_s2")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--xcluster", default="", help="H012b: label-free pseudo-signer cluster ids of the train clips (.npy)")
    ap.add_argument("--xmode", default="exclude", choices=["exclude", "penalty"])
    ap.add_argument("--xpen", type=float, default=0.1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    t0 = time.time()
    tc, TX, TA, TM = load_seqs(a.train_feats)
    n = len(tc)
    perm = np.random.default_rng(a.seed).permutation(n)
    hold, trn = perm[: n // 20], perm[n // 20:]
    Xg = torch.from_numpy(TX).to(dev); Ag = torch.from_numpy(TA).to(dev); Mg = torch.from_numpy(TM).to(dev)
    # neighbour candidates by wmean (labels never used); computed within the train partition
    Wm = wmean_vec(Xg, Ag, Mg)
    nbr = torch.zeros((n, a.k), dtype=torch.long, device=dev)
    pool = torch.from_numpy(trn).to(dev)
    xcl = torch.from_numpy(np.load(a.xcluster)).to(dev) if a.xcluster else None
    if xcl is not None:
        assert len(xcl) == n
    for i in range(0, n, 2048):
        s = Wm[i:i + 2048] @ Wm[pool].T
        s[(pool[None] == torch.arange(i, min(i + 2048, n), device=dev)[:, None])] = -9
        if xcl is not None and a.xmode == "exclude":  # H012b: candidates only from other pseudo-signer clusters
            s[xcl[i:i + 2048][:, None] == xcl[pool][None]] = -9
        nbr[i:i + 2048] = pool[s.topk(a.k, dim=1).indices]
    print(f"loaded {n} clips, neighbours ready ({time.time()-t0:.0f}s)", flush=True)
    model = (Aggregator() if a.student == "transformer" else SoftSegPool()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    rng = np.random.default_rng(a.seed + 1)
    log = open(out / "train_log.jsonl", "w")

    def distill_loss(idx):
        idx_t = torch.from_numpy(idx).to(dev)
        cand = nbr[idx_t]  # B,K
        # in-batch extra candidates: other anchors
        inb = idx_t[torch.randint(0, len(idx_t), (len(idx_t), 8), device=dev)]
        cand = torch.cat([cand, inb], 1)
        with torch.no_grad():
            if a.arm == "H4":
                if a.teacher == "q2g+diff":  # H011: window chamfer + temporal-difference chamfer (exploratory)
                    T = chamfer(Xg[idx_t], Mg[idx_t], Xg[cand], Mg[cand], "q2g") + \
                        chamfer(*tdiff(Xg[idx_t], Mg[idx_t], a.diff_lag), *tdiff(Xg[cand], Mg[cand], a.diff_lag), "q2g")
                else:
                    T = chamfer(Xg[idx_t], Mg[idx_t], Xg[cand], Mg[cand], a.teacher)
            else:
                T = (Wm[idx_t][:, None] * Wm[cand]).sum(-1)
            if xcl is not None and a.xmode == "penalty":  # H012b: debias the teacher against same pseudo-signer
                T = T - a.xpen * (xcl[idx_t][:, None] == xcl[cand]).float()
        uniq, inv = torch.unique(torch.cat([idx_t, cand.flatten()]), return_inverse=True)
        Z = model(Xg[uniq], Ag[uniq], Mg[uniq])
        za, zc = Z[inv[:len(idx_t)]], Z[inv[len(idx_t):]].view(len(idx_t), -1, Z.shape[1])
        Ssc = (za[:, None].float() * zc.float()).sum(-1)
        p = F.softmax(T / a.tau, -1)
        return F.kl_div(F.log_softmax(Ssc / a.tau, -1), p, reduction="batchmean")

    def inst_loss(idx):
        idx_t = torch.from_numpy(idx).to(dev)
        x1, a1, m1 = crop(Xg[idx_t], Ag[idx_t], Mg[idx_t])
        x2, a2, m2 = crop(Xg[idx_t], Ag[idx_t], Mg[idx_t])
        z1, z2 = model(x1, a1, m1), model(x2, a2, m2)
        logits = z1.float() @ z2.float().T / 0.1
        y = torch.arange(len(idx_t), device=dev)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

    lossf = inst_loss if a.arm == "CI" else distill_loss
    if a.student == "softseg":  # init check: equals the hard 2-bin readout up to soft edges
        model.eval()
        with torch.no_grad():
            vc0, VX0, VA0, VM0 = load_seqs("aslc_val_s2")
            G0, Q0 = embed(model, TX, TA, TM, dev), embed(model, VX0, VA0, VM0, dev)
            gl0 = sorted({c["gloss"] for c in tc}); lab0 = {g: i for i, g in enumerate(gl0)}
            r0 = ranks_from_scores(class_max_scores(Q0, G0, np.array([lab0[c["gloss"]] for c in tc]), len(gl0)),
                                   np.array([lab0[c["gloss"]] for c in vc0]))
            print("INIT R@1", metrics_from_ranks(r0)["R@1"], flush=True)
    model.train()
    for step in range(1, a.steps + 1):
        lr = 3e-4 * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(rng.choice(trn, a.batch, replace=False)).float()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hl = lossf(rng.choice(hold, a.batch, replace=False)).item()
            model.train()
            st = {"step": step, "loss": loss.item(), "holdout_loss": hl, "sec": time.time() - t0}
            log.write(json.dumps(st) + "\n"); log.flush()
            if step % 500 == 0:
                print(json.dumps(st), flush=True)
    torch.save(model.state_dict(), out / "last.pt")
    # official gallery evaluation (labels used only here)
    model.eval()
    vc, VX, VA, VM = load_seqs(a.query_feats)
    if any(c["split"] == "test" for c in vc) and not a.allow_test:
        raise SystemExit("test queries need --allow-test under a LOCKED protocol")
    G, Q = embed(model, TX, TA, TM, dev), embed(model, VX, VA, VM, dev)
    gl_ = sorted({c["gloss"] for c in tc}); lab = {g: i for i, g in enumerate(gl_)}
    gy = np.array([lab[c["gloss"]] for c in tc]); qy = np.array([lab[c["gloss"]] for c in vc])
    r = ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy)
    np.save(out / "ranks_val.npy", r)
    m = metrics_from_ranks(r, qy)
    json.dump({"metrics": m, "args": vars(a), "seconds": time.time() - t0}, open(out / "eval.json", "w"), indent=1)
    print("EVAL", a.arm, json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
