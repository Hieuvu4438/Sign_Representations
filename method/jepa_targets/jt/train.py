"""Continued pretraining of V-JEPA 2.1 ViT-B on unlabelled ASL Citizen train windows (EX15).

Every arm: same init (official encoder + predictor body), same data stream, masks, optimiser, steps,
batch and EMA. Arms differ only in the targets:
  --latent      JEPA: L1 to layer-normed EMA-encoder tokens at masked tokens (+ 2.1 context loss, 0.5)
  --pixel       MAE: MSE to per-patch normalised pixels at masked tokens
  --hog         MSE to HOG of masked tokens
  --priors G..  frame-level priors (POSE, REL, NM, MOT) regressed from the online encoder's context
                tokens pooled per temporal slice (SignRep-style auxiliary regression)
The EMA encoder (momentum 0.99925, official) is kept in every arm and is the evaluated model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jt.data import ClipWindows, PriorStats, fit_prior_stats  # noqa: E402
from jt.priors import DIMS  # noqa: E402
from jt.targets import hog_targets, pixel_targets  # noqa: E402
from jt.vjepa import Masks, apply_masks, build_encoder, build_predictor, normalize_video  # noqa: E402

RT = ROOT / "runtime"


class PriorHead(nn.Module):
    """context tokens -> mean per temporal slice (8 slices of 2 frames) -> MLP -> concatenated priors."""

    def __init__(self, dout, d=768, slices=8):
        super().__init__()
        self.slices = slices
        self.t = nn.Embedding(slices, d)
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, dout))

    def forward(self, z, idx, per_slice=196):
        ts = (idx // per_slice).clamp(max=self.slices - 1)  # [B,K]
        oh = F.one_hot(ts, self.slices).float()  # [B,K,S]
        pooled = torch.einsum("bks,bkd->bsd", oh, z.float()) / oh.sum(1).clamp(min=1)[..., None]
        return self.net(pooled + self.t.weight[None])  # [B,S,dout]


def slice_targets(batch, groups, dev):
    """per-frame priors [B,16,D] -> per-slice mean over valid frames [B,8,D] and mask."""
    vs, ms = [], []
    for g in groups:
        v = batch[g].to(dev).float(); m = batch[g + "_mask"].to(dev).float()
        B, Fr, D = v.shape
        v, m = v.view(B, Fr // 2, 2, D), m.view(B, Fr // 2, 2, D)
        cnt = m.sum(2)
        vs.append((v * m).sum(2) / cnt.clamp(min=1)); ms.append(cnt > 0)
    return torch.cat(vs, -1), torch.cat(ms, -1)


@torch.no_grad()
def collapse_stats(enc, video):
    z = enc(video).mean(1).float()
    zn = F.normalize(z, dim=-1)
    cos = (zn @ zn.T)[~torch.eye(len(z), dtype=bool, device=z.device)].mean().item()
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return {"mean_cos": cos, "eff_rank": float(torch.exp(-(p * (p + 1e-12).log()).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", action="store_true")
    ap.add_argument("--pixel", action="store_true")
    ap.add_argument("--hog", action="store_true")
    ap.add_argument("--priors", default="", help="comma list of POSE,REL,NM,MOT")
    ap.add_argument("--prior-w", type=float, default=1.0)
    ap.add_argument("--vicreg", type=float, default=0.0, help="SignRep-style variance/covariance reg. weight on pooled context features (S arm)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--ema", type=float, default=0.99925)
    ap.add_argument("--ctx-w", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    groups = [g for g in a.priors.split(",") if g]
    assert a.latent or a.pixel or a.hog or groups, "no target"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    stats_path = RT / "prior_stats_aslc_train.npz"
    if groups and not stats_path.exists():
        fit_prior_stats(ClipWindows("train").rows, stats_path)
    ds = ClipWindows("train", groups, PriorStats(stats_path) if groups else None, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True, prefetch_factor=4, pin_memory=True,
                                     worker_init_fn=lambda _: (torch.set_num_threads(1), __import__('cv2').setNumThreads(1)))
    dims = {"latent": 768 if a.latent else 0, "pixel": 1536 if a.pixel else 0, "hog": 72 if a.hog else 0}
    dense = sum(dims.values())
    enc = build_encoder().to(dev)
    ema = copy.deepcopy(enc).eval()
    for p in ema.parameters():
        p.requires_grad = False
    pred = build_predictor(dense, context_head=a.latent).to(dev) if dense else None
    pdim = sum(DIMS[g] for g in groups)
    phead = PriorHead(pdim).to(dev) if groups else None
    params = [p for m in (enc, pred, phead) if m is not None for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.999))
    masks = Masks(seed=a.seed)
    fixed = None
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time(); step = 0
    while step < a.steps:
        for batch in dl:
            step += 1
            lr = a.lr * min(1.0, step / 300) * (0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps)) * 0.99 + 0.01)
            for g in opt.param_groups:
                g["lr"] = lr
            v = normalize_video(batch["video"].to(dev, non_blocking=True))
            if fixed is None:
                fixed = v[:16].clone()
            B = v.shape[0]
            me, mp = masks(B, dev)
            terms = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tg = []
                if a.latent:
                    with torch.no_grad():
                        tg.append(F.layer_norm(ema(v).float(), (768,)))
                if a.pixel:
                    tg.append(pixel_targets(v.float()))
                if a.hog:
                    tg.append(hog_targets(v.float()))
                h = torch.cat(tg, -1) if tg else None
                loss = 0.0
                if groups:
                    ptgt, pmask = slice_targets(batch, groups, dev)
                for i, (e, p) in enumerate(zip(me, mp)):
                    z = enc(v, masks=[e], training=True)
                    if dense:
                        zp, zc = pred(z, [e], [p], mask_index=i)
                        ht = apply_masks(h, [p])
                        o = 0
                        if a.latent:
                            l = (zp[..., :768].float() - ht[..., :768]).abs().mean()
                            lc = (zc[..., :768].float() - apply_masks(h[..., :768], [e])).abs().mean()
                            terms["latent"] = terms.get("latent", 0) + l / len(me)
                            terms["ctx"] = terms.get("ctx", 0) + lc / len(me)
                            loss = loss + (l + a.ctx_w * lc) / len(me); o = 768
                        for k in ("pixel", "hog"):
                            if dims[k]:
                                l = F.mse_loss(zp[..., o:o + dims[k]].float(), ht[..., o:o + dims[k]])
                                terms[k] = terms.get(k, 0) + l / len(me)
                                loss = loss + l / len(me); o += dims[k]
                    if a.vicreg > 0:  # SignRep 5.2 feature regularisation (variance + covariance) on pooled tokens
                        zz = z.float().mean(1)
                        zz = zz - zz.mean(0)
                        std = (zz.var(0) + 1e-4).sqrt()
                        cov = (zz.T @ zz) / (len(zz) - 1)
                        off = cov - torch.diag(torch.diag(cov))
                        lv = F.relu(1 - std).mean() + (off ** 2).sum() / zz.shape[1]
                        terms["vicreg"] = terms.get("vicreg", 0) + lv / len(me)
                        loss = loss + a.vicreg * lv / len(me)
                    if groups:
                        pp = phead(z, e)
                        l = (F.smooth_l1_loss(pp.float(), ptgt, reduction="none") * pmask).sum() / pmask.sum().clamp(min=1)
                        terms["priors"] = terms.get("priors", 0) + l / len(me)
                        loss = loss + a.prior_w * l / len(me)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), enc.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
            if step % 50 == 0 or step == 1:
                st = {"step": step, "loss": float(loss), **{k: float(v_) for k, v_ in terms.items()}, "lr": lr, "sec": time.time() - t0}
                if step % 500 == 0 or step == 1:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        st.update(collapse_stats(ema, fixed))
                    print(json.dumps(st), flush=True)
                if not math.isfinite(float(loss)):
                    raise SystemExit(f"non-finite loss at step {step}")
                log.write(json.dumps(st) + "\n"); log.flush()
            if step >= a.steps:
                break
    torch.save({"ema_encoder": ema.state_dict(), "encoder": enc.state_dict(), "config": vars(a), "step": step}, out / "last.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
