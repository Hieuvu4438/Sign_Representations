"""Train PhonoJEPA and its matched baselines (proposal12 v2.1 §6, §9) on the proposal11 comparison protocol:
the 3,000-video ASLC002 corpus, 4,000 optimizer steps x 32 clips, T = 64 frames at min(fps, 25), 1 seed.

Arms are configuration switches of the same code (§9 factorial):
  --family PL|R4|R0 (|PLfull for gate G2)   --target ctx|int|both   --loss cce|proto|l1sig
  --glob (gate G1 / B-glob with --target none)   --use-r1 --curriculum (G2)   --morph (G3)   --hands-only
  --shuffle-int (H2 control)   --baseline-rate 0.6 --int-part level (B-MAE) / 0.9 --int-part delta (B-MAMP)
Optimizer recipe = obs_jepa B (proposal11 §11.1): AdamW lr 3e-4, betas (0.9, 0.95), wd 0.05 (no decay on norms /
biases / embeddings), 10% warm-up, cosine to 1%, grad-clip 1.0, bf16, EMA 0.996 -> 0.9999 (cosine).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pj import features as Fe  # noqa: E402
from pj.data.cache import ClipCache  # noqa: E402
from pj.data.dataset import Cfg, PhonoWindows, collate  # noqa: E402
from pj.joints import FACE_CH, GROUP, NC  # noqa: E402
from pj.losses import SIGReg, CenteredCE, LossBalancer, intrinsic_loss, koleo, l1_layernorm  # noqa: E402
from pj.model import PhonoJEPA, momentum  # noqa: E402
from pj.paths import RUNTIME  # noqa: E402

STATS = RUNTIME / "target_stats.npz"
GAMMA = (1.0, 0.5)                                    # hands, face


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="PL", choices=["PL", "PLfull", "R4", "R0"])
    ap.add_argument("--target", default="both", choices=["ctx", "int", "both", "none"])
    ap.add_argument("--loss", default="cce", choices=["cce", "proto", "l1sig"])
    ap.add_argument("--no-ema", action="store_true", help="l1sig only: teacher = current student (stop-grad), no EMA")
    ap.add_argument("--glob", action="store_true")
    ap.add_argument("--use-r1", action="store_true")
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--morph", action="store_true")
    ap.add_argument("--hands-only", action="store_true")
    ap.add_argument("--handedness", default="canonical", choices=["canonical", "random"], help="A8(g): random L/R flip")
    ap.add_argument("--shuffle-int", action="store_true")
    ap.add_argument("--baseline-rate", type=float, default=0.0)
    ap.add_argument("--int-part", default="both", choices=["both", "level", "delta"])
    ap.add_argument("--alpha-int", type=float, default=1.0)
    ap.add_argument("--ctx-scale", type=float, default=1.0, help="ctx-only weight multiplier (control)")
    ap.add_argument("--lam-glob", type=float, default=1.0)
    ap.add_argument("--lam-koleo", type=float, default=0.05)
    ap.add_argument("--lam-sig", type=float, default=0.02)
    ap.add_argument("--n-proto", type=int, default=4096)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dp", type=int, default=128)
    ap.add_argument("--pdepth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=str(RUNTIME / "cache" / "aslc002_train.npz"))
    return ap.parse_args(argv)


def lr_at(step, total, peak, warm=0.10, final=0.01):
    w = max(1, int(warm * total))
    if step <= w:
        return peak * step / w
    p = (step - w) / max(1, total - w)
    return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * p)))


def tau_t_at(step, total, t0=0.04, t1=0.06, warm=0.10):
    return t1 if step >= warm * total else t0 + (t1 - t0) * step / max(1, warm * total)


def param_groups(mod, lr, wd):
    dec, no = [], []
    for n, p in mod.named_parameters():
        if not p.requires_grad:
            continue
        last = n.split(".")[-1]
        (no if p.ndim <= 1 or last in ("bias", "ch", "absent", "cls", "mask") else dec).append(p)
    return [{"params": dec, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def file_sig(path):
    st = Path(path).stat()
    return hashlib.sha1(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:12]


def view_tensors(v, dev, hands_only):
    v = {k: (x.to(dev, non_blocking=True) if torch.is_tensor(x) else x) for k, x in v.items()}
    valid_cells = v["valid_b"][:, None, :].expand(-1, NC, -1).clone()
    if hands_only:
        for k in ("visible", "target", "hidden"):
            v[k] = v[k].clone(); v[k][:, list(FACE_CH)] = False
        valid_cells[:, list(FACE_CH)] = False
    v["valid_cells"] = valid_cells
    return v


def group_weighted(per_tok, w):
    """per_tok, w [B,N] (bin-major tokens) -> sum_g gamma_g * mean_g / sum gamma over present groups."""
    grp = torch.as_tensor(np.tile(GROUP, 32), device=w.device)
    tot, gs = 0.0, 0.0
    for g, gam in enumerate(GAMMA):
        wg = w * (grp == g)
        if wg.sum() > 0:
            tot = tot + gam * (per_tok * wg).sum() / wg.sum(); gs += gam
    return tot / gs if gs else (per_tok * 0).sum()


def target_weights(target, valid_cells, rel, present_teacher):
    """every target cell is queried; its loss weight = teacher reliability, 0 when the teacher token is ABSENT."""
    return (target & valid_cells & present_teacher).float() * rel


def fit_target_stats(ds, dev, n=640, bs=64):
    """intrinsic-target mean/std from unmasked teacher windows (no labels)."""
    batches = []
    rng = np.random.default_rng(123)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    for s in range(0, len(idx), bs):
        b = collate([ds.item(int(i), np.random.default_rng(int(i))) for i in idx[s:s + bs]])["near"]
        t = {k: b[k].to(dev) for k in ("t_xy", "t_obs", "t_conf", "s_jh", "valid_f")}
        f = Fe.frame_features(t["t_xy"], t["t_obs"], t["t_conf"], torch.zeros_like(t["s_jh"]))
        batches.append(Fe.intrinsic_targets(f, t["valid_f"]))
    STATS.parent.mkdir(parents=True, exist_ok=True)
    Fe.TargetStats.fit(batches, STATS)


class Trainer:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.model = PhonoJEPA(a.d, a.depth, a.heads, a.dp, a.pdepth, n_proto=a.n_proto if a.loss == "proto" else 0,
                               glob=a.glob, sigreg=a.loss == "l1sig").to(dev)
        self.model.copy_student_to_teacher()
        trainable = [self.model.student, self.model.predictor, self.model.heads]
        for extra in (self.model.g_cls, self.model.proto, self.model.sig_tok):
            if extra is not None:
                trainable.append(extra)
        self.params = [p for m in trainable for p in m.parameters() if p.requires_grad]
        groups = []
        for m in trainable:
            groups += param_groups(m, a.lr, a.wd)
        self.opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
        self.cce = CenteredCE(a.n_proto if a.loss == "proto" else a.d, device=dev)
        self.cce_cls = CenteredCE(a.d, device=dev)
        self.sig = SIGReg().to(dev) if a.loss == "l1sig" else None
        self.bal = LossBalancer(freeze_step=int(0.1 * a.steps))
        self.tstats = Fe.TargetStats(STATS)

    # ----------------------------------------------------------------- one view
    def view_loss(self, v, step, terms):
        a, m = self.a, self.model
        tau_t = tau_t_at(step, a.steps)
        with torch.no_grad():
            ft = Fe.frame_features(v["t_xy"], v["t_obs"], v["t_conf"], torch.zeros_like(v["s_jh"]), v["t_aff"])
            ins_t, pres_t = Fe.tokens(ft, v["valid_f"])
            teacher = m.student if a.no_ema else m.teacher
            h_t, cls_t, layers = teacher(ins_t, pres_t, v["valid_cells"], return_layers=4)
            tgt = F.layer_norm(torch.stack([F.layer_norm(l.float(), (l.shape[-1],)) for l in layers]).mean(0), (h_t.shape[-1],))
            # intrinsic targets from the UN-augmented teacher view (review N5: the student cannot know t_aff)
            ft0 = Fe.frame_features(v["t_xy"], v["t_obs"], v["t_conf"], torch.zeros_like(v["s_jh"])) if a.target in ("int", "both") else ft
            ys, ws = Fe.intrinsic_targets(ft0, v["valid_f"])
            ys = self.tstats(ys)
            if a.int_part != "both":
                ws = [self._part_mask(c, w) for c, w in enumerate(ws)]
            if a.shuffle_int:
                perm = torch.randperm(ys[0].shape[0], device=ys[0].device)
                ys, ws = [y[perm] for y in ys], [w[perm] for w in ws]
            rel = Fe.reliability(v["t_conf"], v["t_obs"], v["valid_f"])         # [B,NC,Tb]
        fs = Fe.frame_features(v["s_xy"], v["s_obs"], v["s_conf"], v["s_jh"], v["s_aff"])
        ins_s, pres_s = Fe.tokens(fs, v["valid_f"])
        h_s, cls_s, _ = m.student(ins_s, pres_s, v["visible"])
        out = {"cls_s": cls_s, "cls_t": cls_t}
        if a.target == "none":
            return 0.0, out
        q = m.predictor(h_s, cls_s, v["visible"], v["target"])
        tw = target_weights(v["target"], v["valid_cells"], rel, pres_t)                  # [B,NC,Tb], 0 for teacher-ABSENT
        w = tw.transpose(1, 2).flatten(1)                                               # [B,N] bin-major
        loss = 0.0
        if a.target in ("ctx", "both"):
            pred = m.heads.ctx(q)
            if a.loss == "cce":
                _, ce = self.cce(pred, tgt, w, tau_t, update=False)
                self._center_buf.append((tgt.detach(), w.detach()))
                lc = group_weighted(ce, w)
                with torch.no_grad():
                    terms["teacher_entropy"] = float((CenteredCE.entropy(self.cce.target(tgt, tau_t)) * w).sum() / w.sum().clamp(min=1))
            elif a.loss == "proto":
                with torch.no_grad():
                    tl = m.t_proto(F.normalize(tgt, dim=-1))
                _, ce = self.cce(m.proto(F.normalize(pred.float(), dim=-1)), tl, w, tau_t, update=False)
                self._center_buf.append((tl.detach(), w.detach()))
                lc = group_weighted(ce, w)
            else:
                e = (pred.float() - tgt.detach()).abs().mean(-1)
                lc = group_weighted(e, w)
            terms["ctx"] = float(lc.detach())
            loss = loss + a.ctx_scale * (self.bal("ctx", lc) if a.target == "both" else lc)
        if a.target in ("int", "both"):
            preds = m.heads.intrinsic(q)
            li = 0.0; gs = 0.0
            for g, gam in enumerate(GAMMA):
                chs = [c for c in range(NC) if GROUP[c] == g]
                lg = intrinsic_loss([preds[c] for c in chs], [ys[c] for c in chs], [ws[c] for c in chs], [tw[:, c] for c in chs])
                if any((tw[:, c] * (ws[c].sum(-1) > 0)).sum() > 0 for c in chs):
                    li = li + gam * lg; gs += gam
            li = li / max(gs, 1e-6)
            if not torch.is_tensor(li):
                li = q.sum() * 0.0
            terms["int"] = float(li.detach())
            loss = loss + a.alpha_int * (self.bal("int", li) if a.target == "both" else li)
        if self.sig is not None:
            vis = v["visible"].transpose(1, 2).flatten(1)
            proj = m.sig_tok(h_s[vis])
            ls = self.sig(proj) + self.sig(m.sig_tok(cls_s))
            terms["sigreg"] = float(ls.detach()); loss = loss + a.lam_sig * ls
        return loss, out

    def _part_mask(self, c, w):
        if c == 6:                                       # HEAD target is delta-only already
            return w if self.a.int_part == "delta" else torch.zeros_like(w)
        half = w.shape[-1] // 2
        w = w.clone()
        if self.a.int_part == "level":
            w[..., half:] = 0
        else:
            w[..., :half] = 0
        return w

    # ----------------------------------------------------------------- one step
    def step(self, b, step):
        a, m = self.a, self.model
        terms = {}
        self._center_buf = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
            vn = view_tensors(b["near"], self.dev, a.hands_only)
            vf = view_tensors(b["far"], self.dev, a.hands_only)
            wfar = 0.3 + 0.7 * (step / a.steps) ** 2 if a.curriculum else 1.0
            ln, on = self.view_loss(vn, step, terms)
            lf, of = self.view_loss(vf, step, terms)
            loss = ln + wfar * lf
            if a.glob:
                tau_t = tau_t_at(step, a.steps)
                with torch.no_grad():
                    tc = [m.t_cls(on["cls_t"].float()), m.t_cls(of["cls_t"].float())]
                lg = 0.0
                for o in (on, of):
                    ps = m.g_cls(o["cls_s"].float())
                    for t in tc:
                        l, _ = self.cce_cls(ps[:, None], t[:, None], torch.ones(ps.shape[0], 1, device=ps.device), tau_t, update=False)
                        lg = lg + l / 4
                with torch.no_grad():
                    bc = torch.cat(tc).mean(0); self.cce_cls.c.mul_(0.9).add_(bc, alpha=0.1)
                lk = koleo(on["cls_s"]) + koleo(of["cls_s"])
                terms["glob"], terms["koleo"] = float(lg.detach()), float(lk.detach())
                loss = loss + a.lam_glob * lg + a.lam_koleo * lk
        return loss, terms

    def commit_centres(self):
        """one centre update per step from near + far targets; called only after a successful (finite) step."""
        if self._center_buf:
            with torch.no_grad():
                num = sum(((t.float() * (w > 0).float()[..., None]).reshape(-1, t.shape[-1]).sum(0)) for t, w in self._center_buf)
                den = sum(float((w > 0).sum()) for _, w in self._center_buf)
                if den > 0:
                    self.cce.c.mul_(self.cce.beta).add_(num / den, alpha=1 - self.cce.beta)
        self._center_buf = []

    def save(self, path, step, extra=None):
        m = self.model
        ck = {"student": m.student.state_dict(), "teacher": m.teacher.state_dict(), "predictor": m.predictor.state_dict(),
              "heads": m.heads.state_dict(), "opt": self.opt.state_dict(), "step": step, "config": vars(self.a),
              "center": self.cce.c.cpu(), "center_cls": self.cce_cls.c.cpu(), "bal": self.bal.state_dict(),
              "rng": torch.get_rng_state()}
        for name in ("g_cls", "t_cls", "proto", "t_proto", "sig_tok"):
            mod = getattr(m, name)
            if mod is not None:
                ck[name] = mod.state_dict()
        if extra:
            ck.update(extra)
        torch.save(ck, str(path) + ".tmp"); Path(str(path) + ".tmp").replace(path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = self.model
        m.student.load_state_dict(ck["student"]); m.teacher.load_state_dict(ck["teacher"])
        m.predictor.load_state_dict(ck["predictor"]); m.heads.load_state_dict(ck["heads"])
        for name in ("g_cls", "t_cls", "proto", "t_proto", "sig_tok"):
            if name in ck and getattr(m, name) is not None:
                getattr(m, name).load_state_dict(ck[name])
        self.opt.load_state_dict(ck["opt"]); self.cce.c = ck["center"].to(self.dev); self.cce_cls.c = ck["center_cls"].to(self.dev)
        self.bal.load_state_dict(ck["bal"]); torch.set_rng_state(ck["rng"])
        return ck["step"]


@torch.no_grad()
def diagnostics(model, b, dev, hands_only):
    v = view_tensors(b["near"], dev, hands_only)
    f = Fe.frame_features(v["t_xy"], v["t_obs"], v["t_conf"], torch.zeros_like(v["s_jh"]))
    ins, pres = Fe.tokens(f, v["valid_f"])
    h, cls, _ = model.teacher(ins, pres, v["valid_cells"])
    cls = cls.float(); zn = F.normalize(cls, dim=-1); C = zn @ zn.T
    off = C[~torch.eye(len(C), dtype=torch.bool, device=C.device)].mean()
    tok = h[v["valid_cells"].transpose(1, 2).flatten(1)].float()
    s = torch.linalg.svdvals(tok - tok.mean(0)); p = s / s.sum()
    return {"cls_mean_cos": float(off), "tok_rankme": float(torch.exp(-(p * (p + 1e-12).log()).sum())), "tok_std": float(tok.std(0).mean())}


def main(argv=None):
    a = get_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pt").exists() and not a.resume:
        raise SystemExit(f"{out} already finished")
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
    dev = a.device
    cache = ClipCache(a.cache)
    cfg = Cfg(family=a.family, use_r1=a.use_r1, curriculum=a.curriculum, morph=a.morph,
              baseline_rate=a.baseline_rate or None, handedness=a.handedness)
    ds = PhonoWindows(cache, cfg, seed=a.seed)
    if not STATS.exists():
        fit_target_stats(PhonoWindows(cache, Cfg(), seed=0), dev)
    tr = Trainer(a, dev)
    step = tr.load(out / "ckpt.pt") if a.resume and (out / "ckpt.pt").exists() else 0
    manifest = {"config": vars(a), "cache_sig": file_sig(a.cache), "stats_sig": file_sig(STATS), "n_clips": len(cache),
                "steps_planned": a.steps, "clips_per_step": a.batch, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "a"); t0 = time.time(); fixed = None; alarms = 0

    def winit(_):
        torch.set_num_threads(1)
    while step < a.steps:
        ds.epoch = step // max(1, len(ds) // a.batch)
        ds.cfg.epoch_frac = step / a.steps
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                                         collate_fn=collate, pin_memory=dev.startswith("cuda"), worker_init_fn=winit,
                                         prefetch_factor=4 if a.workers else None, persistent_workers=False)
        for b in dl:
            if fixed is None:
                fixed = b
            for g in tr.opt.param_groups:
                g["lr"] = lr_at(step + 1, a.steps, a.lr)
            loss, terms = tr.step(b, step)
            tr.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(tr.params, 1.0)
            if not torch.isfinite(gn):
                tr.opt.zero_grad(set_to_none=True); print("skip non-finite step", step, flush=True); continue
            tr.opt.step(); step += 1; tr.bal.tick(); tr.commit_centres()
            if not a.no_ema:
                tr.model.ema_update(momentum(step, a.steps))
            else:
                tr.model.copy_student_to_teacher()
            if step % a.log_every == 0 or step == 1:
                st = {"step": step, "loss": float(loss.detach()), **terms, "lr": tr.opt.param_groups[0]["lr"], "gnorm": float(gn),
                      "sec": round(time.time() - t0, 1),
                      "near_hidden": float(b["near"]["hidden"].float().mean()), "far_hidden": float(b["far"]["hidden"].float().mean())}
                if step % (5 * a.log_every) == 0 or step == 1:
                    st.update(diagnostics(tr.model, fixed, dev, a.hands_only))
                    if st["cls_mean_cos"] > 0.99 and step > 0.1 * a.steps:         # random init already ~0.92-0.97
                        alarms += 1; st["collapse_alarm"] = alarms
                    print(json.dumps(st), flush=True)
                log.write(json.dumps(st) + "\n"); log.flush()
            if step % a.save_every == 0 or step == a.steps:
                tr.save(out / "ckpt.pt", step)
            if step >= a.steps:
                break
    tr.save(out / "final.pt", step)
    manifest.update({"steps_done": step, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "collapse_alarms": alarms})
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print("DONE", step, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
