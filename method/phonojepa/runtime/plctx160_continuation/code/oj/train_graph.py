"""Train proposal B (graph-JEPA) on unlabelled clips. objective: latent (B12) | anchors (B13, core) | cce (B11) | raw (B10).
Recipe (proposal11 §11.1): AdamW lr 3e-4 from scratch, betas (0.9, 0.95), wd 0.05 (no decay on norms/biases/embeddings),
10% warm-up, cosine to 1%, clip 1.0, bf16, EMA m: 0.996 -> 0.9999 (cosine), EMA after each successful step.
Checkpoints every --save-every steps (full state) so a crash can resume with --resume.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.clips import video_meta  # noqa: E402
from oj.data.landmarks import FACE, LH, RH  # noqa: E402
from oj.data.pose_dataset import PoseStats, PoseWindows, collate, fit_stats, read_corpus  # noqa: E402
from oj.features.geometry import ANCHOR_DIMS  # noqa: E402
from oj.losses import CenteredCE, masked_smooth_l1  # noqa: E402
from oj.models.graph import GraphJepa  # noqa: E402
from oj.models.rgb import momentum  # noqa: E402
from oj.paths import ASLC002_CORPUS, RUNTIME  # noqa: E402


def corpus_meta(rows, path):
    if Path(path).exists():
        return json.load(open(path))
    meta = {r["sample_id"]: video_meta(r["video"]) for r in rows}
    json.dump(meta, open(path, "w")); return meta


def param_groups(model, lr, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no if p.ndim <= 1 or n.endswith(".bias") or "embed" in n.lower() or "node.weight" in n or "part.weight" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no, "weight_decay": 0.0, "lr": lr}]


def lr_at(step, total, peak, warm=0.10, final=0.01):
    w = max(1, int(warm * total))
    if step <= w:
        return peak * step / w
    p = (step - w) / max(1, total - w)
    return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p)))


def to(b, dev):
    return {k: ({kk: vv.to(dev, non_blocking=True) for kk, vv in v.items()} if isinstance(v, dict) else
                (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)) for k, v in b.items()}


@torch.no_grad()
def collapse_stats(model, b):
    e = model.embed(b["te_node"], b["te_loc"], b["obs"], b["t_bin"]).float()
    zn = F.normalize(e, dim=-1); C = zn @ zn.T
    s = torch.linalg.svdvals(e - e.mean(0)); p = s / s.sum()
    return {"mean_cos": float(C[~torch.eye(len(e), dtype=bool, device=e.device)].mean()), "eff_rank": float(torch.exp(-(p * (p + 1e-12).log()).sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="anchors", choices=["latent", "anchors", "cce", "raw"])
    ap.add_argument("--mask-policy", default="core", choices=["core", "whole", "generic"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lambda-obs", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=a.resume)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
    dev = "cuda"
    rows = read_corpus(ASLC002_CORPUS)
    meta = corpus_meta(rows, RUNTIME / "meta_aslc002.json")
    stats_path = RUNTIME / f"b_stats_T{a.T}.npz"
    if not stats_path.exists():
        fit_stats(PoseWindows(rows, meta, T=a.T), stats_path)
    ds = PoseWindows(rows, meta, T=a.T, policy=a.mask_policy, stats=PoseStats(stats_path), seed=a.seed)

    def winit(_):
        torch.set_num_threads(1); cv2.setNumThreads(1)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                                     collate_fn=collate, persistent_workers=True, prefetch_factor=4, pin_memory=True,
                                     worker_init_fn=winit)
    model = GraphJepa(a.objective).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.wd), lr=a.lr, betas=(0.9, 0.95))
    cce = CenteredCE(256, device=dev) if a.objective == "cce" else None
    step = 0
    if a.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        if cce is not None and "center" in ck:
            cce.c = ck["center"].to(dev)
        torch.set_rng_state(ck["rng"])
    log = open(out / "train_log.jsonl", "a"); t0 = time.time(); fixed = None
    while step < a.steps:
        for b in dl:
            b = to(b, dev)
            if fixed is None:
                fixed = {k: (v[:32] if torch.is_tensor(v) else ({kk: vv[:32] for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in b.items()}
            lr = lr_at(step + 1, a.steps, a.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            hid = b["hid"]
            terms = {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    ht, _, _, _ = model.teacher(b["te_node"], b["te_loc"], b["obs"], b["t_bin"])
                    tgt = F.layer_norm(ht.float(), (ht.shape[-1],))
                hs, summ, pv, lf = model.student(b["st_node"], b["st_loc"], b["vis"], b["t_bin"])
                pred = model.predictor(hs, b["vis"], hid, b["t_bin"])
                w = hid.float()
                if a.objective in ("latent", "anchors"):
                    l = ((pred.float() - tgt).abs().mean(-1) * w).sum() / w.sum().clamp(min=1)
                    terms["latent"] = l; loss = l
                elif a.objective == "cce":
                    l = cce(pred, ht.float(), w); terms["cce"] = l; loss = l
                else:  # raw masked reconstruction of teacher-side geometry
                    m = b["raw_valid"].float() * w[..., None]
                    l = (F.smooth_l1_loss(pred.float(), b["raw"], reduction="none") * m).sum() / m.sum().clamp(min=1)
                    terms["raw"] = l; loss = l
                if a.objective == "anchors":
                    ph = model.anchors(lf, b["vis"], summ)
                    fam = []
                    for k in ANCHOR_DIMS:
                        mk = b["ancm_" + k].float()
                        if mk.sum() > 0:
                            fam.append(masked_smooth_l1(ph[k], b["anc_" + k], mk)); terms["anc_" + k] = fam[-1]
                    if fam:
                        lo = torch.stack(fam).mean(); terms["anchors"] = lo; loss = loss + a.lambda_obs * lo
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True); print("skip non-finite step", step, flush=True); continue
            opt.step(); step += 1
            model.ema_update(momentum(step, a.steps))
            if step % 50 == 0 or step == 1:
                st = {"step": step, "loss": float(loss), **{k: float(v) for k, v in terms.items()}, "lr": lr,
                      "hidden_frac": float(hid.sum() / b["obs"].sum()), "sec": time.time() - t0}
                if step % 500 == 0 or step == 1:
                    st.update(collapse_stats(model, fixed)); print(json.dumps(st), flush=True)
                log.write(json.dumps(st) + "\n"); log.flush()
            if step % a.save_every == 0 or step == a.steps:
                ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "rng": torch.get_rng_state(), "config": vars(a)}
                if cce is not None:
                    ck["center"] = cce.c.cpu()
                torch.save(ck, out / "ckpt.tmp"); (out / "ckpt.tmp").replace(out / "ckpt.pt")
            if step >= a.steps:
                break
    torch.save({"teacher": model.teacher.state_dict(), "student": model.student.state_dict(), "config": vars(a), "step": step}, out / "final.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
