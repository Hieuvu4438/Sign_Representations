"""ASC-JEPA training (SignRep warm-start branch, spec §2.4 / §11). One script, runs B0..B4 and
controls by config. Saves the online backbone in SignRep checkpoint layout ({"student": {backbone.*,
fc.*}}) so the p4 extraction/evaluation pipeline evaluates it with identical readouts."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from asc_jepa import roi as R  # noqa: E402
from asc_jepa.backbone import build_online_and_teacher  # noqa: E402
from asc_jepa.losses import ChangeRMS, change_loss, generic_jepa, geometry_loss, state_loss, var_cov  # noqa: E402
from asc_jepa.predictor import Predictor, grid_coords  # noqa: E402

sys.path.insert(0, str(HERE.parents[0] / "p4_signssl"))
from p4 import signrep as S  # noqa: E402
from p4.data import aslc_rows, decode_resized  # noqa: E402

RUNS = {  # spec §14.2 controlled chain (+ C4a random-gate control)
    "B0": dict(layers=["final"], kappa=0.0, state=False, change=False, routing="off"),
    "B1": dict(layers=["mid", "final"], kappa=0.05, state=False, change=False, routing="off"),
    "B2": dict(layers=["mid", "final"], kappa=0.05, state=True, change=False, routing="off"),
    "B3": dict(layers=["mid", "final"], kappa=0.05, state=True, change=True, routing="off"),
    "B4": dict(layers=["mid", "final"], kappa=0.05, state=True, change=True, routing="coverage"),
    "C4a": dict(layers=["mid", "final"], kappa=0.05, state=True, change=True, routing="random_matched"),
}


class ClipDataset(torch.utils.data.Dataset):
    """Unlabelled ASL Citizen train clips (labels never read) -> one 16-frame window + pose sidecars."""

    def __init__(self, rows, seed=0):
        self.rows, self.seed = rows, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, i, random.random())))
        r = self.rows[i]
        arr, fps = decode_resized(r["path"])
        try:
            K, Sc = R.load_kp(r["sample_id"])
        except Exception:
            K = np.full((len(arr), 133, 2), np.nan, np.float32); Sc = np.zeros((len(arr), 133), np.float32)
        n = min(len(arr), len(K))
        arr, K, Sc = arr[:n], K[:n], Sc[:n]
        if n < 16:
            pad = 16 - n
            arr = np.concatenate([arr, np.repeat(arr[-1:], pad, 0)])
            K = np.concatenate([K, np.full((pad, 133, 2), np.nan, np.float32)]); Sc = np.concatenate([Sc, np.zeros((pad, 133), np.float32)])
            n = 16
        t = rng.randint(0, n - 16)
        K, Sc = K[t:t + 16], Sc[t:t + 16]
        boxes, valid = R.frame_boxes(K, Sc)
        geo, gmask = R.geometry_targets(K, Sc)
        return (torch.from_numpy(arr[t:t + 16]), torch.from_numpy(boxes), torch.from_numpy(valid),
                torch.from_numpy(geo), torch.from_numpy(gmask), float(fps or 30.0))


class GeoHead(nn.Module):
    def __init__(self, d=768, frames=16, out=92):
        super().__init__()
        self.t = nn.Embedding(frames, 64)
        self.net = nn.Sequential(nn.Linear(d + 64, 512), nn.GELU(), nn.Linear(512, out))

    def forward(self, pooled):
        B = pooled.shape[0]
        te = self.t.weight[None].expand(B, -1, -1)
        return self.net(torch.cat([pooled[:, None].expand(-1, te.shape[1], -1), te], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-heads", type=float, default=3e-4)
    ap.add_argument("--hidden", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps-run", type=int, default=0)
    a = ap.parse_args()
    cfg = RUNS[a.run]
    W = dict(alpha=0.5, beta=0.1, gamma=0.1, eta=0.1, gate_thr=0.25)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = "cuda"
    online, teacher, fc, _act, info = build_online_and_teacher(dev)
    online.train()
    dims = {"mid": 384, "final": 768}
    pred = Predictor({k: dims[k] for k in ("mid", "final")}).to(dev)
    geo = GeoHead().to(dev)
    proj = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 128)).to(dev)
    bb_params = list(online.parameters())
    head_params = list(pred.parameters()) + list(geo.parameters()) + list(proj.parameters())
    nd = lambda ps: [p for p in ps if p.ndim > 1]
    dcy = lambda ps: [p for p in ps if p.ndim <= 1]
    opt = torch.optim.AdamW([
        {"params": nd(bb_params), "lr": a.lr_backbone, "base": a.lr_backbone, "weight_decay": 0.05},
        {"params": dcy(bb_params), "lr": a.lr_backbone, "base": a.lr_backbone, "weight_decay": 0.0},
        {"params": nd(head_params), "lr": a.lr_heads, "base": a.lr_heads, "weight_decay": 0.05},
        {"params": dcy(head_params), "lr": a.lr_heads, "base": a.lr_heads, "weight_decay": 0.0}])
    rows = [{k: r[k] for k in ("sample_id", "path")} for r in aslc_rows("train")]
    dl = torch.utils.data.DataLoader(ClipDataset(rows, a.seed), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     drop_last=True, generator=torch.Generator().manual_seed(a.seed), persistent_workers=True, prefetch_factor=4)
    mask_gen = torch.Generator().manual_seed(a.seed + 1)
    rand_gate_gen = torch.Generator(device=dev).manual_seed(a.seed + 2)
    T, H, Wd = 8, 14, 14
    rms = ChangeRMS(4, [1, 2])
    json.dump(dict(vars(a), cfg=cfg, weights=W, init=info, branch="SignRep warm-start (spec §2.4)",
                   adaptations=["COCO-WholeBody 2-D keypoints for ROI/geometry instead of MediaPipe",
                                "no photometric augmentation", "backbone LR 2e-5 for warm-start (spec pilot 1e-4 is for MAE init)"]),
              open(out / "config.json", "w"), indent=1, default=str)
    log = open(out / "train_log.jsonl", "w")
    warm = max(1, int(0.1 * a.steps))
    step, t0 = 0, time.time()
    while step < a.steps:
        for u8, boxes, valid, gt, gm, fps in dl:
            if step >= a.steps or (a.max_steps_run and step >= a.max_steps_run):
                break
            sc = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = max(g["base"] * sc, 1e-7)
            lw = min(1.0, (step + 1) / warm)  # local-term warmup
            B = u8.shape[0]
            x = S.normalize_uint8(u8.reshape(-1, *u8.shape[2:]).cuda(non_blocking=True), dev).view(B, 16, 3, 224, 224).permute(0, 2, 1, 3, 4).contiguous()
            boxes, valid = boxes.to(dev), valid.to(dev)
            frame_ms = 1000.0 / float(np.median(fps.numpy()))
            coords = {k: grid_coords(T, H, Wd, frame_ms, 2, dev) for k in ("mid", "final")}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tgt, _ = teacher.forward_full(x)
            tgt = {k: F.layer_norm(v.grid.float(), (v.grid.shape[-1],)) for k, v in tgt.items()}
            mask = online.random_mask(B, a.hidden, generator=mask_gen, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx, fin_tokens = online.forward_masked(x, mask)
                pr = pred(ctx, coords)
            stats = {}
            L = 0.0
            lw_gen = {k: 1.0 / len(cfg["layers"]) for k in cfg["layers"]}
            for k in cfg["layers"]:
                lj, st = generic_jepa(pr[k], tgt[k].reshape(B, -1, dims[k]), ctx[k].visible.reshape(B, -1), cfg["kappa"])
                L = L + lw_gen[k] * lj; stats[f"J_{k}"] = st["mask"]
            if cfg["state"]:
                pi, support = R.roi_tile_weights(boxes, valid, T, H, Wd)
                occ = pi.sum((3, 4))
                svalid = ((support > 0) & (occ >= 1.0)).float()
                Ls, Ld, act_frac = 0.0, 0.0, 0.0
                for k in ("mid", "final"):
                    ps = R.pool_states(pr[k].float().view(B, T, H, Wd, -1), pi)
                    ts = R.pool_states(tgt[k], pi).detach()
                    if cfg["routing"] == "coverage":
                        gate = (R.observability(pi, ctx[k].visible) >= W["gate_thr"]).float()
                    elif cfg["routing"] == "random_matched":  # C4a: random retention with the same expected rate
                        rate = ((R.observability(pi, ctx[k].visible) >= W["gate_thr"]).float() * svalid).sum() / svalid.sum().clamp(min=1)
                        gate = (torch.rand(svalid.shape, device=dev, generator=rand_gate_gen) < rate).float()
                    else:
                        gate = torch.ones_like(svalid)
                    l_s, act_frac = state_loss(ps, ts, svalid * gate)
                    Ls = Ls + 0.5 * l_s
                    if cfg["change"]:
                        lags = R.change_pairs(2 * frame_ms, T=T)
                        l_d, _ = change_loss(ps, ts, svalid, gate, lags, rms)
                        Ld = Ld + 0.5 * l_d
                L = L + lw * W["alpha"] * Ls + (lw * W["beta"] * Ld if cfg["change"] else 0.0)
                stats.update(S=float(Ls), D=float(Ld) if cfg["change"] else 0.0, local_active=act_frac,
                             gate_rate=float((gate * svalid).sum() / svalid.sum().clamp(min=1)))
            pooled = fin_tokens.float().mean(1)  # visible online final tokens (masked path returns tokens)
            Lg = geometry_loss(geo(pooled), gt.to(dev), gm.to(dev).float())
            Lr = var_cov(proj(pooled))
            L = L + lw * W["gamma"] * Lg + W["eta"] * Lr
            stats.update(G=float(Lg), R=float(Lr))
            if not torch.isfinite(L):
                raise RuntimeError(f"non-finite loss at step {step}: {stats}")
            opt.zero_grad(set_to_none=True)
            L.backward()
            gn = torch.nn.utils.clip_grad_norm_(bb_params + head_params, 1.0)
            opt.step()
            # after a successful optimizer step: EMA teacher, RMS stats
            m = 0.9999 - (0.9999 - 0.996) * (1 + math.cos(math.pi * step / a.steps)) / 2
            with torch.no_grad():
                for pt, po in zip(teacher.parameters(), online.parameters()):
                    pt.mul_(m).add_(po.detach(), alpha=1 - m)
            rms.step()
            step += 1
            if step % 25 == 0 or step == 1:
                stats.update(step=step, loss=float(L), grad_norm=float(gn), ema_m=m, sec=time.time() - t0)
                log.write(json.dumps(stats) + "\n"); log.flush()
                if step % 250 == 0:
                    print(json.dumps(stats), flush=True)
        if a.max_steps_run and step >= a.max_steps_run:
            break
    sd = {f"backbone.{k[len('bb.'):]}": v for k, v in online.state_dict().items()}
    sd.update({f"fc.{k}": v for k, v in fc.state_dict().items()})
    torch.save({"student": sd, "step": step}, out / "last.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
