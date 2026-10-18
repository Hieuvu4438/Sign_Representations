"""Training windows: two (teacher, student) pairs per clip — near view (pair 1) and far view (pair 2).

Per pair (proposal12 v2.1 §6.3):
  window placement (start, speed) from the RNG and the clip length only
  -> source support of every output frame (linear interpolation)
  -> content-independent mask on the 8 x 32 grid (s2.masks)
  -> hidden source frames per channel; scrub the student source (owned joints -> NaN) for fully hidden cells and R1 joints
  -> propagation: a non-hidden cell whose frames read a source frame hidden for any channel it DEPENDS on is context-dropped
  -> teacher = full source resampled + mild camera aug; student = scrubbed source resampled + student aug
Gloss / signer fields are never read.
"""
from __future__ import annotations

import numpy as np
import torch

from fp.data import window as W
from fp.joints import D_CONF, DEPENDS, DUPLICATES, J, LH0, N_CONF, NC, OWNED, RH0
from fp.masks import random_view, sample_view

T, BIN = 64, 2
TB = T // BIN


class Cfg(dict):
    """dataset options (all defaults = MVM, proposal12 §6.11)."""
    defaults = dict(T=64, family="PL", use_r1=False, curriculum=False, speed=(0.8, 1.25), teacher_rot=5.0,
                    teacher_scale=0.05, student_rot=10.0, student_scale=0.10, student_shear=0.10, noise=0.02,
                    hand_drop=0.05, morph=False, ctx_drop=0.10, baseline_rate=None, epoch_frac=1.0,
                    handedness="canonical")       # canonical (content-based, declared 1-bit leak) | random (A8g, leak-free)

    def __init__(self, **kw):
        super().__init__({**self.defaults, **kw})
        self.__dict__ = self


def owned_mask(cells_hidden, frames_of_bin):
    """[NC, Tb] hidden cells -> list of (channel, output frames)."""
    out = []
    for c in range(NC):
        b = np.flatnonzero(cells_hidden[c])
        if len(b):
            out.append((c, np.concatenate([frames_of_bin[x] for x in b])))
    return out


def build_pair(xy, conf, fps, rng, cfg: Cfg, kind: str, rng_aug=None):
    """returns dict of numpy arrays for one (teacher, student) pair."""
    rng_aug = rng_aug or rng
    n = len(xy)
    lo, hi = cfg.speed if cfg.speed else (1.0, 1.0)
    s = float(rng.uniform(lo, hi))
    st = W.random_start(n, fps, cfg.T, s, rng)
    i0, i1, w, valid = W.window_positions(n, fps, cfg.T, st, s)
    sup = W.support(i0, i1, w, valid)
    n_valid_bins = int(valid.reshape(TB, BIN).any(1).sum())
    if cfg.baseline_rate:
        m = random_view(n_valid_bins, TB, rng, cfg.baseline_rate)
    else:
        m = sample_view(n_valid_bins, TB, rng, kind, cfg.family, cfg.epoch_frac, cfg.curriculum, cfg.use_r1, cfg.ctx_drop)
    frames_of_bin = [np.arange(b * BIN, (b + 1) * BIN) for b in range(TB)]

    # hidden source frames per channel (full cells) and joint-level R1 hidden source frames
    src_hidden = np.zeros((NC, n), bool)
    for c, fr in owned_mask(m.hidden, frames_of_bin):
        for k in fr:
            src_hidden[c, sup[k]] = True
    scrub_full = np.zeros((n, J), bool)
    for c in range(NC):
        if src_hidden[c].any():
            scrub_full[np.ix_(src_hidden[c], OWNED[c])] = True
    scrub_r1 = np.zeros((n, J), bool)
    for hand, base in ((0, RH0), (1, LH0)):
        for b in range(TB):
            js = np.flatnonzero(m.r1[b, hand])
            if len(js):
                for k in frames_of_bin[b]:
                    if sup[k]:
                        scrub_r1[np.ix_(sup[k], base + js)] = True
    # Close duplicates before any interpolation or view statistics.
    scrub = scrub_full | scrub_r1
    for a, b in DUPLICATES:
        both = scrub[:, a] | scrub[:, b]
        scrub[:, a] = both; scrub[:, b] = both
    # A visible token cannot read a forbidden raw source via another representation.
    drop = m.drop.copy()
    for c in range(NC):
        dep = np.zeros(n, bool)
        for d in DEPENDS[c]:
            dep |= scrub[:, OWNED[d]].any(-1)
        for b in range(TB):
            if not m.hidden[c, b] and m.valid[b]:
                if any(dep[sup[k]].any() for k in frames_of_bin[b] if sup[k]):
                    drop[c, b] = True

    # teacher view (full) and student view (scrubbed)
    t_xy, t_c, _ = W.resample(xy, conf, i0, i1, w, valid)
    sxy, sconf = xy.copy(), conf.copy()
    sxy[scrub] = np.nan; sconf[scrub] = 0.0
    s_xy, s_c, s_jh = W.resample(sxy, sconf, i0, i1, w, valid, flag=scrub_r1)
    # Exact forbidden raw-support closure, separate from stochastic dropout.
    _, _, forbidden = W.resample(sxy, sconf, i0, i1, w, valid, flag=scrub)
    times = (np.arange(cfg.T, dtype=np.float64) * s / min(fps, W.TARGET_FPS)).astype(np.float32)
    # augmentation (per view, on output frames)
    t_aff = W.camera_affine(rng_aug, cfg.teacher_rot, cfg.teacher_scale, 0.0)          # applied in the body frame
    if cfg.morph:
        s_xy = W.morphology(s_xy, rng_aug)
    s_aff = W.camera_affine(rng_aug, cfg.student_rot, cfg.student_scale, cfg.student_shear)
    if cfg.noise:
        s_xy = W.keypoint_noise(s_xy, s_c, rng_aug, cfg.noise)
    if cfg.hand_drop:
        s_xy, s_c = W.hand_dropout(s_xy, s_c, rng_aug, cfg.hand_drop)
    s_c = np.where(np.isfinite(s_xy).all(-1), s_c, 0.0).astype(np.float32)
    t_c = np.where(np.isfinite(t_xy).all(-1), t_c, 0.0).astype(np.float32)
    target = m.target
    visible = m.valid[None] & ~m.hidden & ~drop
    return {
        "t_xy": np.nan_to_num(t_xy, nan=0.0), "t_obs": np.isfinite(t_xy).all(-1), "t_conf": t_c,
        "s_xy": np.nan_to_num(s_xy, nan=0.0), "s_obs": np.isfinite(s_xy).all(-1), "s_conf": s_c,
        "times": times, "forbidden": forbidden, "s_jh": s_jh,                                             # joint-hidden flags (R1 and its propagation)
        "valid_f": valid, "valid_b": m.valid, "hidden": m.hidden, "target": target, "visible": visible,
        "drop": drop, "raw_scrub": scrub, "stats": m.stats, "speed": s, "t_aff": t_aff, "s_aff": s_aff,
    }


class PhonoWindows(torch.utils.data.Dataset):
    def __init__(self, cache, cfg: Cfg, seed=0):
        self.cache, self.cfg, self.seed = cache, cfg, seed
        self.epoch = 0

    def __len__(self):
        return len(self.cache)

    def item(self, i, rng):
        xy, conf, fps = self.cache.clip(i)
        if self.cfg.handedness == "random":                  # shared by teacher and student, independent of content
            flipped = bool(rng.random() < 0.5)
            if flipped:
                xy, conf = W.mirror(xy, conf)
        else:
            xy, conf, flipped = W.canonicalise(xy, conf)
        near = build_pair(xy, conf, fps, rng, self.cfg, "near")
        far = build_pair(xy, conf, fps, rng, self.cfg, "far")
        return {"near": near, "far": far, "flipped": flipped}

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i, int(np.random.randint(1 << 30))])
        return self.item(i, rng)


_ARR = ("t_xy", "t_obs", "t_conf", "s_xy", "s_obs", "s_conf", "s_jh", "valid_f", "valid_b", "hidden", "target",
        "visible", "drop", "t_aff", "s_aff", "times", "forbidden")


def collate(items):
    out = {}
    for v in ("near", "far"):
        out[v] = {k: torch.from_numpy(np.stack([it[v][k] for it in items])) for k in _ARR}
        out[v]["stats"] = [it[v]["stats"] for it in items]
    out["flipped"] = torch.tensor([it["flipped"] for it in items])
    return out
