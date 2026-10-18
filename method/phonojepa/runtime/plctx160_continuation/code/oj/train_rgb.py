"""Train proposal A (RGB-JEPA) and C (A + frozen-B visible geometry distillation) on unlabelled clips.

Recipe (proposal11 §7.5, §11.1): V-JEPA 2.1 ViT-B init; predictor 4 x 384 (new); AdamW betas (0.9, 0.95), wd 0.05
(no decay on norms/biases), encoder lr 1e-4, predictor/new heads 3e-4, 10% warm-up, cosine to 1%, clip 1.0, bf16;
EMA m 0.996 -> 0.9999 (cosine) updated once after each successful optimizer step (after micro-batch accumulation).
Losses: L_A = L_M (region-balanced or uniform masked L1 over target levels) + lambda_C L_C (AD: 0.05);
C adds lambda_G,obs (0.1) x L_G,obs on complete-visible-support (bin, part) cells.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.landmarks import FACE, LH, RH  # noqa: E402
from oj.data.pose_dataset import PoseStats, read_corpus  # noqa: E402
from oj.data.rgb_dataset import RGBWindows, collate  # noqa: E402
from oj.losses import context_loss, per_token_l1, region_masked_loss  # noqa: E402
from oj.masks.rgb_masks import G_FACE, G_LH, G_RH, GROUPS  # noqa: E402
from oj.models.graph import GraphEncoder  # noqa: E402
from oj.models.rgb import RGBJepa, gather, momentum, normalize_video  # noqa: E402
from oj.paths import ASLC002_CORPUS, RUNTIME  # noqa: E402
from oj.train_graph import corpus_meta, lr_at  # noqa: E402


def groups_for(module, lr, wd):
    dec, no = [], []
    for n, p in module.named_parameters():
        if p.requires_grad:
            (no if p.ndim <= 1 or n.endswith(".bias") else dec).append(p)
    return [{"params": dec, "weight_decay": wd, "peak": lr}, {"params": no, "weight_decay": 0.0, "peak": lr}]


class GeoHeads(nn.Module):
    """C: per-part heads D_r on ROI-pooled VISIBLE student tokens -> frozen-B local articulation embedding (256)."""

    def __init__(self, d_in=768, d_out=256):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out)) for _ in range(3)])

    def forward(self, pooled):  # [B,Tb,3,d_in]
        return torch.stack([h(pooled[:, :, k]) for k, h in enumerate(self.heads)], 2)


def roi_pool(z, ctx, owner, Tb, gg):
    """mean of student context features per (bin, part) over context tokens owned by lh/rh/face -> [B,Tb,3,d], count."""
    B, K, d = z.shape
    own = torch.gather(owner, 1, ctx)  # [B,K]
    b_of = ctx // gg
    out = z.new_zeros(B, Tb, 3, d, dtype=torch.float32); cnt = z.new_zeros(B, Tb, 3, dtype=torch.float32)
    for k, gid in enumerate((G_LH, G_RH, G_FACE)):
        sel = (own == gid).float()
        idx = b_of
        out[:, :, k].scatter_add_(1, idx[..., None].expand(-1, -1, d), z.float() * sel[..., None])
        cnt[:, :, k].scatter_add_(1, idx, sel)
    return out / cnt.clamp(min=1)[..., None], cnt


@torch.no_grad()
def b_local_targets(benc, b_loc, b_vis):
    lf = benc.local(b_loc, b_vis)
    outs = []
    for name, nodes in (("lh", LH), ("rh", RH), ("face", FACE)):
        v = b_vis[:, :, nodes].float()
        pooled = (lf[name].float() * v[..., None]).sum(2) / v.sum(2).clamp(min=1)[..., None]
        outs.append(F.layer_norm(pooled, (pooled.shape[-1],)))
    return torch.stack(outs, 2)  # [B,Tb,3,256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="AR", choices=["AR", "AD"])
    ap.add_argument("--policy", default="partial", choices=["generic", "generic_cov", "partial", "whole", "jitter"])
    ap.add_argument("--weighting", default="balanced", choices=["balanced", "uniform"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64, help="global clips per optimizer step")
    ap.add_argument("--micro", type=int, default=16, help="micro-batch size (accumulation)")
    ap.add_argument("--lambda-ctx", type=float, default=None, help="default 0 (AR) / 0.05 (AD)")
    ap.add_argument("--b-teacher", default="", help="C: frozen B final.pt (teacher encoder)")
    ap.add_argument("--lambda-geo", type=float, default=0.1)
    ap.add_argument("--c-reserve", action="store_true", help="C-reserved masks (use for C and ALL its controls)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    lam_ctx = a.lambda_ctx if a.lambda_ctx is not None else (0.05 if a.recipe == "AD" else 0.0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=a.resume)
    json.dump({**vars(a), "lambda_ctx": lam_ctx}, open(out / "config.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
    dev = "cuda"
    rows = read_corpus(ASLC002_CORPUS); meta = corpus_meta(rows, RUNTIME / "meta_aslc002.json")
    use_c = bool(a.b_teacher)
    pstats = PoseStats(RUNTIME / "b_stats_T64.npz") if (use_c or a.c_reserve) else None
    ds = RGBWindows(rows, meta, size=a.size, policy=a.policy, seed=a.seed, c_mode=(use_c or a.c_reserve), pose_stats=pstats)

    def winit(_):
        torch.set_num_threads(1); cv2.setNumThreads(1)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.micro, shuffle=True, drop_last=True, num_workers=a.workers,
                                     collate_fn=collate, persistent_workers=True, prefetch_factor=4, pin_memory=True,
                                     worker_init_fn=winit)
    model = RGBJepa(size=a.size, recipe=a.recipe).to(dev)
    geo = GeoHeads().to(dev) if use_c else None
    benc = None
    if use_c:
        benc = GraphEncoder().to(dev).eval()
        benc.load_state_dict(torch.load(a.b_teacher, map_location="cpu", weights_only=False)["teacher"])
        for p in benc.parameters():
            p.requires_grad = False
    pg = groups_for(model.student, 1e-4, 0.05) + groups_for(model.predictor, 3e-4, 0.05) + (groups_for(geo, 3e-4, 0.05) if geo else [])
    opt = torch.optim.AdamW(pg, lr=1e-4, betas=(0.9, 0.95))
    params = [p for g in pg for p in g["params"]]
    accum = max(1, a.batch // a.micro)
    step = 0
    if a.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if geo is not None:
            geo.load_state_dict(ck["geo"])
    Tb, g = 8, a.size // 16; gg = g * g
    log = open(out / "train_log.jsonl", "a"); t0 = time.time()
    it = iter(dl)
    while step < a.steps:
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step + 1, a.steps, gr["peak"])
        opt.zero_grad(set_to_none=True)
        agg = {}; moved = 0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            moved += b["ctx_moved_to_guard"]
            x = normalize_video(b["video"].to(dev, non_blocking=True))
            ctx, tgt = b["ctx"].to(dev), b["tgt"].to(dev); grp, w = b["tgt_group"].to(dev), b["tgt_w"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.targets(x)
                p_t, p_c, z = model(x, ctx, tgt)
                err = per_token_l1(p_t, gather(h, tgt), model.n_out)
                lm, cov = region_masked_loss(err, grp, w, balanced=(a.weighting == "balanced"))
                loss = lm; terms = {"L_M": lm}
                if lam_ctx > 0:
                    lc = context_loss(per_token_l1(p_c, gather(h, ctx), model.n_out), torch.ones_like(ctx, dtype=torch.float32))
                    loss = loss + lam_ctx * lc; terms["L_C"] = lc
                if use_c:
                    zf = z[..., -768:]
                    pooled, cnt = roi_pool(zf, ctx, b["owner"].to(dev), Tb, gg)
                    tg = b_local_targets(benc, {k: v.to(dev) for k, v in b["b_loc"].items()}, b["b_vis"].to(dev))
                    sup = b["c_support"].to(dev).float() * (cnt > 0).float()
                    pr = geo(pooled)
                    e = (pr.float() - tg).abs().mean(-1)  # [B,Tb,3]
                    parts = [(e[..., k] * sup[..., k]).sum() / sup[..., k].sum() for k in range(3) if sup[..., k].sum() > 0]
                    if parts:
                        lg = torch.stack(parts).mean(); loss = loss + a.lambda_geo * lg; terms["L_G_obs"] = lg
                    terms["c_support_frac"] = sup.mean()
            (loss / accum).backward()
            for k, v in terms.items():
                agg[k] = agg.get(k, 0.0) + float(v) / accum
            for gid, n in cov.items():
                agg["cov_" + GROUPS[gid]] = agg.get("cov_" + GROUPS[gid], 0.0) + n
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); print("skip non-finite step", step, flush=True); continue
        opt.step(); step += 1
        model.ema_update(momentum(step, a.steps))
        if step % 25 == 0 or step == 1:
            st = {"step": step, **agg, "ctx_moved_to_guard": moved, "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn), "sec": time.time() - t0}
            log.write(json.dumps(st) + "\n"); log.flush()
            if step % 250 == 0 or step == 1:
                print(json.dumps(st), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": vars(a)}
            if geo is not None:
                ck["geo"] = geo.state_dict()
            torch.save(ck, out / "ckpt.tmp"); (out / "ckpt.tmp").replace(out / "ckpt.pt")
    torch.save({"student": model.student.state_dict(), "teacher": model.teacher.state_dict(), "config": vars(a), "step": step,
                "recipe": a.recipe, "size": a.size}, out / "final.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
