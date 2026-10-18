"""EX16: SignRep-reproduction vs JEPA with the SAME SignRep 3-D sign priors (user request 2026-10-08).

Shared by every arm: the ASLC002 corpus (3,000 unlabelled ASL Citizen train videos with 3-D teacher
pose, `method/jepa_baseline/runtime/ASLC002/corpus`), two 16-frame views per video (A2), the
reproduction's prior targets/masks (`signrep_baseline.data.clip_targets`), its sign decoder
(ProjectionModel + activity head on z_avg = fc(mean tokens)), its prior loss (Eq. 2, Table 9 weights and
scales), its augmentation recipe (resize 256 -> Planckian jitter -> random resized crop 224 -> blur ->
grayscale; same probabilities, executed on GPU), the optimiser recipe (AdamW betas .9/.95, wd .5,
layer decay .85, lr 1e-4, 10% warm-up, cosine), batch 32 views (16 videos x 2), 4,000 steps, bf16.

Arms
  sr3d  SignRep reproduction (signrep_baseline.Pretrainer): priors + variance/covariance + adversarial
        style loss; Hiera-B Kinetics-MAE init; 80% random mask units.
  jh3d  JEPA on the same Hiera-B Kinetics-MAE init and the same 80% random mask units: priors + latent
        prediction (L1 to layer-normed EMA-teacher final tokens at masked units; predictor 6x384).
        No variance/covariance/adversarial terms (those are SignRep-paradigm components).
  jv3d  JEPA on V-JEPA 2.1 ViT-B (official encoder + predictor body, multi-block masks): priors (same
        decoder on z_avg of context tokens) + latent prediction (+ 2.1 context loss).
  srvj  SignRep recipe UNCHANGED (Pretrainer: priors + var/cov + adversarial style, same decoder/heads/optimiser)
        on the V-JEPA 2.1 ViT-B backbone; masking keeps SignRep's geometry: random 80% of 8x7x7 mask units, one unit
        = 2x2 ViT tokens = 32x32 px x 2 frames (Hiera mask unit). Control that separates init from objective.
"""
from __future__ import annotations

import argparse
import copy
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
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BASE = REPO / "method" / "jepa_baseline"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BASE / "src"))
from signrep_baseline.augmentation import AugmentationConfig  # noqa: E402
from signrep_baseline.data import PairedVideoDataset  # noqa: E402
from signrep_baseline.losses import reconstruction_loss  # noqa: E402
from signrep_baseline.model import SignRep  # noqa: E402
from signrep_baseline.optim import WarmupCosine, layerwise_parameter_groups  # noqa: E402
from signrep_baseline.schema import PRIOR_SHAPES  # noqa: E402
from signrep_baseline.training import Pretrainer  # noqa: E402
from signrep_baseline.upstream_planckian.planckianTransforms import PlanckianJitter  # noqa: E402
from signrep_baseline.upstream_signrep.models.head_models.dict_head_v6_FINAL_thoughtful import ProjectionModel  # noqa: E402
from signrep_baseline.upstream_signrep.models.hiera.hiera_clstoken import Hiera  # noqa: E402

CORPUS = BASE / "runtime" / "ASLC002" / "corpus" / "pretrain.jsonl"
INIT = BASE / "runtime" / "ASLC002" / "initial.model.pt"
RT = ROOT / "runtime"
PL = torch.tensor(PlanckianJitter("blackbody").pl, dtype=torch.float32)
MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def resize256(frames):
    """worker-side: signer-cropped uint8 T x H x W x 3 -> T x 256 x 256 x 3 (antialiased, = first recipe step)."""
    return np.stack([cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA if min(f.shape[:2]) > 256 else cv2.INTER_LINEAR) for f in frames])


class Raw:
    def __call__(self, frames):
        return torch.from_numpy(resize256(np.asarray(frames)))


def gpu_augment(views, cfg=AugmentationConfig()):
    """views: [N,16,256,256,3] uint8 (cuda) -> [N,3,16,224,224] normalised; per-view random params as in
    signrep_baseline.augmentation.VideoTransform (Planckian -> RRC 224 -> blur 23x23 -> grayscale)."""
    out = []
    pl = PL.to(views.device)
    for v in views:
        x = v.permute(0, 3, 1, 2).float() / 255.0  # T,3,256,256
        if torch.rand(()) < cfg.planckian_probability:
            i = torch.randint(len(pl), (1,)).item()
            x = x * torch.stack([pl[i, 0] / pl[i, 1], torch.ones((), device=x.device), pl[i, 2] / pl[i, 1]]).view(1, 3, 1, 1)
            x = torch.floor(x.clamp(max=1.0) * 255) / 255  # uint8 round-trip of the PIL implementation
        top, left, h, w = RandomResizedCrop.get_params(x[0], cfg.crop_scale, cfg.crop_ratio)
        x = TF.resized_crop(x, top, left, h, w, [224, 224], antialias=True)
        if torch.rand(()) < cfg.blur_probability:
            s = torch.empty(()).uniform_(*cfg.blur_sigma).item()
            x = TF.gaussian_blur(x, [23, 23], [s, s])
        if torch.rand(()) < cfg.grayscale_probability:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)
        x = (x.clamp(0, 1) - MEAN.to(x.device)) / STD.to(x.device)
        out.append(x.permute(1, 0, 2, 3))
    return torch.stack(out)


class PairBatches(torch.utils.data.Sampler):
    """batches of `videos` distinct video indices (two views each come from the dataset item)."""

    def __init__(self, n, videos, steps, seed):
        self.n, self.videos, self.steps, self.seed = n, videos, steps, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            yield torch.randperm(self.n, generator=g)[: self.videos].tolist()

    def __len__(self):
        return self.steps


def _worker_init(_):
    torch.set_num_threads(1); cv2.setNumThreads(1)  # 6 single-threaded workers instead of oversubscribed ones


def collate(items):
    return {"views": torch.cat([it["images"] for it in items]),  # [2B,16,256,256,3] uint8
            "targets": {k: torch.cat([it["targets"][k] for it in items]) for k in items[0]["targets"]},
            "masks": {k: torch.cat([it["masks"][k] for it in items]) for k in items[0]["masks"]},
            "video_ids": [v for it in items for v in it["video_ids"]]}


def sincos3d(T, H, W, d, device):
    t, y, x = torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij")
    feats = []
    for c, n in ((t, d // 3), (y, d // 3), (x, d - 2 * (d // 3))):
        nf = n // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(nf) / nf)
        a = c.flatten()[:, None].float() * fr[None]
        feats += [a.sin(), a.cos()]
        if n % 2:
            feats.append(torch.zeros(len(a), 1))
    return torch.cat(feats, 1).to(device)  # [T*H*W, d]


class TokenPredictor(nn.Module):
    """JEPA predictor for Hiera final tokens (8x14x14 grid, unit = 1x2x2 tokens)."""

    def __init__(self, d_in=768, d=384, depth=6, heads=6):
        super().__init__()
        self.inp = nn.Linear(d_in, d); self.out = nn.Linear(d, d_in)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos3d(8, 14, 14, d, "cpu"), persistent=False)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(el, depth); self.norm = nn.LayerNorm(d)

    def forward(self, vis_tok, vis_idx, tgt_idx):
        B = vis_tok.shape[0]
        x = self.inp(vis_tok) + self.pos[vis_idx]
        q = self.mask_token.expand(B, tgt_idx.shape[1], -1) + self.pos[tgt_idx]
        h = self.blocks(torch.cat([x, q], 1))
        return self.out(self.norm(h[:, x.shape[1]:]))


def unit_token_index(units):
    """unit ids [B,U] on the (8,7,7) grid -> canonical flat token ids [B,U*4] on (8,14,14), block order (dy,dx)."""
    ut, uy, ux = units // 49, (units % 49) // 7, units % 7
    dy = torch.tensor([0, 0, 1, 1], device=units.device); dx = torch.tensor([0, 1, 0, 1], device=units.device)
    flat = (ut[..., None] * 14 + uy[..., None] * 2 + dy) * 14 + ux[..., None] * 2 + dx
    return flat.reshape(units.shape[0], -1)


class VJSignRep(nn.Module):
    """SignRep model interface (encode/forward outputs used by signrep_baseline.Pretrainer) on V-JEPA 2.1 ViT-B."""

    def __init__(self, mask_ratio=0.8):
        super().__init__()
        from jt.vjepa import build_encoder
        self.mask_ratio = mask_ratio
        self.backbone = build_encoder()  # official ema_encoder weights, 224 px, 16 frames
        self.cls_head = ProjectionModel(latents_params={"hidden_dim": 512, "out_dim": 384},
                                        projection_params={k: {"hidden_dim": None, "output_dim": math.prod(s)} for k, s in PRIOR_SHAPES.items()},
                                        total_frames=16, in_dim=768)
        self.fc = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False))
        self.active_head = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 2))
        units = torch.arange(392)
        ut, uy, ux = units // 49, (units % 49) // 7, units % 7
        dy = torch.tensor([0, 0, 1, 1]); dx = torch.tensor([0, 1, 0, 1])
        self.register_buffer("unit_tokens", ((ut[:, None] * 14 + uy[:, None] * 2 + dy) * 14 + ux[:, None] * 2 + dx), persistent=False)

    def forward(self, images):
        B = images.shape[0]
        if self.training and self.mask_ratio > 0:
            keep = int(392 * (1 - self.mask_ratio))
            units = torch.rand(B, 392, device=images.device).argsort(1)[:, :keep].sort(1).values
            idx = self.unit_tokens[units].reshape(B, -1).sort(1).values
            tokens = self.backbone(images, masks=[idx], training=True)
        else:
            tokens = self.backbone(images)
        feats = self.fc(tokens.float().mean(1))
        return {"features": feats, "tokens": tokens, "cls_head": self.cls_head(feats), "cls_active": self.active_head(feats)}


def prior_predictions(head, active, z):
    out = head(z)
    return {**{k: out[k] for k in PRIOR_SHAPES}, "activity": active(z)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["sr3d", "jh3d", "jv3d", "srvj"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--videos", type=int, default=16)
    ap.add_argument("--jepa-w", type=float, default=1.0)
    ap.add_argument("--prior-w", type=float, default=1.0,
                    help="JH3D prior loss weight; 0 disables prior supervision (latent-only ablation)")
    ap.add_argument("--ema", type=float, default=0.99925)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    if not math.isfinite(a.prior_w) or a.prior_w < 0:
        ap.error("--prior-w must be finite and nonnegative")
    if a.arm != "jh3d" and a.prior_w != 1.0:
        ap.error("--prior-w is only supported for --arm jh3d")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
    dev = "cuda"
    ds = PairedVideoDataset(CORPUS, Raw(), cache_capacity=64)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=PairBatches(len(ds), a.videos, a.steps, a.seed), num_workers=a.workers,
                                     collate_fn=collate, prefetch_factor=4, persistent_workers=True, pin_memory=True,
                                     worker_init_fn=_worker_init)
    warm = max(1, a.steps // 10)
    init = torch.load(INIT, map_location="cpu", weights_only=True)
    assert init["initialization"] == "kinetics_mae"
    log = open(out / "train_log.jsonl", "w"); t0 = time.time()

    if a.arm in ("sr3d", "srvj"):
        if a.arm == "sr3d":
            model = SignRep(); model.load_state_dict(init["model"], strict=True)
        else:
            torch.manual_seed(a.seed); model = VJSignRep()
        tr = Pretrainer(model, device=dev, total_steps=a.steps, warmup_steps=warm, max_lr=1e-4, amp=True, amp_dtype="bfloat16")
        for step, b in enumerate(dl, 1):
            batch = {"images": gpu_augment(b["views"].to(dev, non_blocking=True)), "targets": b["targets"], "masks": b["masks"], "video_ids": b["video_ids"]}
            m = tr.update(batch)
            if step % 50 == 0 or step == 1:
                st = {k: m[k] for k in ("step", "loss", "reconstruction", "variance", "covariance", "adversarial", "discriminator", "lr")}
                st["sec"] = time.time() - t0; log.write(json.dumps(st) + "\n"); log.flush()
                if step % 500 == 0 or step == 1:
                    print(json.dumps(st), flush=True)
        if a.arm == "sr3d":
            torch.save({"kind": "hiera", "model": tr.model.state_dict(), "step": step, "config": vars(a)}, out / "last.pt")
        else:
            torch.save({"kind": "vjepa", "encoder": tr.model.backbone.state_dict(), "fc": tr.model.fc.state_dict(),
                        "model": tr.model.state_dict(), "step": step, "config": vars(a)}, out / "last.pt")
        print("DONE", step, time.time() - t0, flush=True)
        return

    if a.arm == "jh3d":
        model = SignRep(); model.load_state_dict(init["model"], strict=True); model.to(dev)
        teacher = copy.deepcopy(model.backbone).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        pred = TokenPredictor().to(dev)
        groups = layerwise_parameter_groups(model, 1e-4) + [{"params": list(pred.parameters()), "lr_scale": 1.0, "lr": 1e-4, "weight_decay": 0.05}]
        opt = torch.optim.AdamW(groups, lr=1e-4, betas=(.9, .95))
        sched = WarmupCosine(opt, 1e-4, warm, a.steps)
        params = [p for g in groups for p in g["params"]]
        for step, b in enumerate(dl, 1):
            model.train(); sched.before_update(step)
            x = gpu_augment(b["views"].to(dev, non_blocking=True))
            tg = {k: v.to(dev) for k, v in b["targets"].items()}; mk = {k: v.to(dev) for k, v in b["masks"].items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.encode(x)  # 80% random mask units (train mode)
                keep = o["mask"]
                vis_units = torch.stack([r.nonzero().squeeze(1) for r in keep]); hid_units = torch.stack([(~r).nonzero().squeeze(1) for r in keep])
                vis_tok = o["intermediates_feature_map"][-1].reshape(len(x), -1, 768)
                with torch.no_grad():
                    full = torch.ones_like(keep)
                    _, inter, _ = Hiera.forward(teacher, x, mask=full, return_intermediates=True)
                    tgt = F.layer_norm(inter[-1].reshape(len(x), -1, 768).float(), (768,))  # [B,1568] in unit-block order
                hid_tok_idx = unit_token_index(hid_units)
                tgt_hid = torch.gather(tgt, 1, (hid_units[..., None] * 4 + torch.arange(4, device=dev)).reshape(len(x), -1)[..., None].expand(-1, -1, 768))
                p = pred(vis_tok, unit_token_index(vis_units), hid_tok_idx)
                l_lat = (p.float() - tgt_hid).abs().mean()
                if a.prior_w > 0:
                    pr = prior_predictions(model.cls_head, model.active_head, o["features"])
                    l_rec, _ = reconstruction_loss(pr, tg, mk)
                    loss = a.prior_w * l_rec + a.jepa_w * l_lat
                else:
                    # Do not connect prior heads/fc to autograd: this arm learns only through JEPA.
                    l_rec = l_lat.new_zeros(())
                    loss = a.jepa_w * l_lat
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.after_update()
            with torch.no_grad():
                for pe, pm in zip(teacher.parameters(), model.backbone.parameters()):
                    pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
            if step % 50 == 0 or step == 1:
                st = {"step": step, "loss": float(loss), "reconstruction": float(l_rec), "latent": float(l_lat), "lr": sched.rate(step), "sec": time.time() - t0}
                log.write(json.dumps(st) + "\n"); log.flush()
                if step % 500 == 0 or step == 1:
                    print(json.dumps(st), flush=True)
        ema_model = copy.deepcopy(model); ema_model.backbone.load_state_dict(teacher.state_dict())
        torch.save({"kind": "hiera", "model": model.state_dict(), "ema_model": ema_model.state_dict(), "step": step, "config": vars(a)}, out / "last.pt")
        print("DONE", step, time.time() - t0, flush=True)
        return

    # jv3d: V-JEPA 2.1 ViT-B
    from jt.vjepa import Masks, apply_masks, build_encoder, build_predictor
    enc = build_encoder().to(dev); teacher = copy.deepcopy(enc).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    predv = build_predictor(768).to(dev)
    fc = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768, bias=False)).to(dev)
    head = ProjectionModel(latents_params={"hidden_dim": 512, "out_dim": 384},
                           projection_params={k: {"hidden_dim": None, "output_dim": math.prod(s)} for k, s in PRIOR_SHAPES.items()},
                           total_frames=16, in_dim=768).to(dev)
    active = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
    wrap = nn.Module(); wrap.backbone = enc; wrap.fc, wrap.cls_head, wrap.active_head, wrap.pred = fc, head, active, predv
    groups = layerwise_parameter_groups(wrap, 1e-4)
    opt = torch.optim.AdamW(groups, lr=1e-4, betas=(.9, .95)); sched = WarmupCosine(opt, 1e-4, warm, a.steps)
    params = [p for g in groups for p in g["params"]]
    masks = Masks(seed=0)
    for step, b in enumerate(dl, 1):
        sched.before_update(step)
        x = gpu_augment(b["views"].to(dev, non_blocking=True))
        tg = {k: v.to(dev) for k, v in b["targets"].items()}; mk = {k: v.to(dev) for k, v in b["masks"].items()}
        me, mp = masks(len(x), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h = F.layer_norm(teacher(x).float(), (768,))
            l_lat, l_rec, zs = 0.0, 0.0, []
            for i, (e, p_) in enumerate(zip(me, mp)):
                z = enc(x, masks=[e], training=True)
                zp, zc = predv(z, [e], [p_], mask_index=i)
                l_lat = l_lat + ((zp.float() - apply_masks(h, [p_])).abs().mean() + 0.5 * (zc.float() - apply_masks(h, [e])).abs().mean()) / len(me)
                pr = prior_predictions(head, active, fc(z.float().mean(1)))
                lr_, _ = reconstruction_loss(pr, tg, mk); l_rec = l_rec + lr_ / len(me)
            loss = l_rec + a.jepa_w * l_lat
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.after_update()
        with torch.no_grad():
            for pe, pm in zip(teacher.parameters(), enc.parameters()):
                pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        if step % 50 == 0 or step == 1:
            st = {"step": step, "loss": float(loss), "reconstruction": float(l_rec), "latent": float(l_lat), "lr": sched.rate(step), "sec": time.time() - t0}
            log.write(json.dumps(st) + "\n"); log.flush()
            if step % 500 == 0 or step == 1:
                print(json.dumps(st), flush=True)
    torch.save({"kind": "vjepa", "encoder": enc.state_dict(), "ema_encoder": teacher.state_dict(), "fc": fc.state_dict(), "step": step, "config": vars(a)}, out / "last.pt")
    print("DONE", step, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
