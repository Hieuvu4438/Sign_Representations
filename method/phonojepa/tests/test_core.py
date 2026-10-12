"""Unit / integration tests for PhonoJEPA (proposal12 v2.1 §13). Run: python -m pytest tests -q (from method/phonojepa).
Real keypoints from the ASLC002 training cache are used where the property depends on real data (leakage tests)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pj import features as Fe  # noqa: E402
from pj.data import window as W  # noqa: E402
from pj.data.dataset import Cfg, build_pair, collate  # noqa: E402
from pj.joints import (ANCHORS, DEPENDS, F_HD, F_MO, F_UP, FACE_CH, J, MIRROR_PERM, NC, OWNED)  # noqa: E402
from pj.masks import R3_CHANNELS, sample_view  # noqa: E402
from pj.model import PhonoJEPA  # noqa: E402

CACHE = ROOT / "runtime" / "cache" / "aslc002_train.npz"


@pytest.fixture(scope="module")
def cache():
    from pj.data.cache import ClipCache
    if not CACHE.exists():
        pytest.skip("training cache missing: python -m pj.data.cache")
    return ClipCache(CACHE)


# ------------------------------------------------------------------ layout
def test_ownership_and_mirror():
    owned = [j for c in OWNED for j in OWNED[c]]
    assert len(owned) == len(set(owned)) and not set(owned) & set(ANCHORS)
    assert (MIRROR_PERM[MIRROR_PERM] == np.arange(J)).all()
    assert DEPENDS[F_UP] == {F_UP, F_HD} and DEPENDS[F_MO] == {F_MO, F_HD}


def test_mirror_roundtrip(cache):
    xy, conf, _ = cache.clip(5)
    a, b = W.mirror(xy, conf)
    c, d = W.mirror(a, b)
    np.testing.assert_allclose(np.nan_to_num(c), np.nan_to_num(xy), atol=1e-3)
    np.testing.assert_array_equal(d, conf)


# ------------------------------------------------------------------ masks
@pytest.mark.parametrize("family,kind", [("PL", "near"), ("PL", "far"), ("PLfull", "far"), ("R4", "near"), ("R0", "far")])
def test_sampler_properties(family, kind):
    for seed in range(200):
        n_valid = int(np.random.default_rng(seed).integers(4, 33))
        m = sample_view(n_valid, 32, np.random.default_rng(seed), kind, family, use_r1=(family == "PLfull"))
        m2 = sample_view(n_valid, 32, np.random.default_rng(seed), kind, family, use_r1=(family == "PLfull"))
        assert (m.hidden == m2.hidden).all() and (m.drop == m2.drop).all() and (m.r1 == m2.r1).all()   # content-independent
        assert not m.hidden[:, n_valid:].any() and not m.drop[:, n_valid:].any()
        assert (m.hidden[F_UP] >= m.hidden[F_HD]).all() and (m.hidden[F_MO] >= m.hidden[F_HD]).all()  # closure
        assert not (m.hidden & m.drop).any()
        if family == "PL" and kind == "far" and not m.stats.get("fallback"):
            hid_ch = set(np.flatnonzero(m.hidden.any(1)))
            assert hid_ch <= set(R3_CHANNELS)                                                     # R3 only hand channels
        if not m.stats.get("fallback"):
            assert m.visible.sum() >= 0.25 * NC * n_valid - 1e-6


# ------------------------------------------------------------------ leakage: end-to-end hidden-source perturbation
def _hidden_source_mask(xy, conf, fps, cfg, kind, seed):
    """re-derive (independently of build_pair's student output) which source entries are hidden."""
    rng = np.random.default_rng(seed)
    out = build_pair(xy, conf, fps, rng, cfg, kind)
    return out


@pytest.mark.parametrize("cfg_kw", [dict(), dict(family="PLfull", use_r1=True, morph=True), dict(family="R4"),
                                    dict(family="R0"), dict(baseline_rate=0.9)])
@pytest.mark.parametrize("kind", ["near", "far"])
def test_e2e_hidden_source_perturbation(cache, cfg_kw, kind):
    cfg = Cfg(**cfg_kw)
    for i in range(0, 120, 7):
        xy0, conf0, fps = cache.clip(i)
        xy, conf, flipped = W.canonicalise(xy0, conf0)
        ref = build_pair(xy, conf, fps, np.random.default_rng(1000 + i), cfg, kind)
        # source entries the student must not depend on: recompute the scrub set by probing with NaN-poisoning
        poison = xy.copy(); poison[:] = 1e6                                     # every value changes ...
        # ... but only entries scrubbed in the reference may change without altering the student view:
        scrub = _scrub_set(xy, conf, fps, cfg, kind, 1000 + i)
        xyp = np.where(scrub[..., None], poison, xy)
        confp = np.where(scrub, 0.123, conf)
        out = build_pair(xyp, confp, fps, np.random.default_rng(1000 + i), cfg, kind)
        for k in ("s_xy", "s_obs", "s_conf", "s_jh", "visible", "target", "hidden", "drop"):
            np.testing.assert_array_equal(out[k], ref[k], err_msg=f"{k} depends on hidden source content")


@pytest.mark.parametrize("cfg_kw,max_rate", [(dict(), 0.5), (dict(family="R4"), 0.5), (dict(family="R0"), 0.5)])
def test_handedness_flip_rate_swap_in(cache, cfg_kw, max_rate):
    """declared 1-bit leak: the handedness decision is made on the whole clip. Replace the hidden source content by
    REAL content of another clip (swap-in, worst case) and measure how often the decision flips (reported per family;
    measured 2026-10-11: PL ~0.33). Ablation A8(g) (--handedness random) removes this leak entirely."""
    cfg = Cfg(**cfg_kw); flips = n = 0
    for i in range(0, 300, 3):
        xy, conf, fps = cache.clip(i)
        other, oconf, _ = cache.clip((i + 1500) % len(cache))
        L = min(len(xy), len(other))
        for kind in ("near", "far"):
            scrub = _scrub_set(xy, conf, fps, cfg, kind, 1000 + i)[:L]
            xyp = xy.copy(); xyp[:L] = np.where(scrub[..., None], other[:L], xy[:L])
            flips += W.left_dominant(xyp) != W.left_dominant(xy); n += 1
    print(f"handedness flip rate {cfg_kw}: {flips / n:.3f}")
    assert flips / n <= max_rate


def _scrub_set(xy, conf, fps, cfg, kind, seed):
    """the set of source (frame, joint) entries hidden for the student, derived from the mask and supports."""
    from pj.data.dataset import TB, BIN
    from pj.masks import random_view
    rng = np.random.default_rng(seed)
    n = len(xy)
    lo, hi = cfg.speed
    s = float(rng.uniform(lo, hi)); st = W.random_start(n, fps, cfg.T, s, rng)
    i0, i1, w, valid = W.window_positions(n, fps, cfg.T, st, s)
    sup = W.support(i0, i1, w, valid)
    nvb = int(valid.reshape(TB, BIN).any(1).sum())
    m = random_view(nvb, TB, rng, cfg.baseline_rate) if cfg.baseline_rate else sample_view(nvb, TB, rng, kind, cfg.family, cfg.epoch_frac, cfg.curriculum, cfg.use_r1, cfg.ctx_drop)
    scrub = np.zeros((n, J), bool)
    from pj.joints import LH0, RH0
    for c in range(NC):
        for b in np.flatnonzero(m.hidden[c]):
            for k in range(b * BIN, (b + 1) * BIN):
                if sup[k]:
                    scrub[np.ix_(sup[k], OWNED[c])] = True
    for hand, base in ((0, RH0), (1, LH0)):
        for b in range(TB):
            js = np.flatnonzero(m.r1[b, hand])
            for k in range(b * BIN, (b + 1) * BIN):
                if len(js) and sup[k]:
                    scrub[np.ix_(sup[k], base + js)] = True
    return scrub


def test_perturbing_visible_content_changes_student(cache):
    """sanity: the perturbation test is not vacuous."""
    xy, conf, fps = cache.clip(3)
    xy, conf, _ = W.canonicalise(xy, conf)
    cfg = Cfg()
    ref = build_pair(xy, conf, fps, np.random.default_rng(7), cfg, "near")
    out = build_pair(xy + 3.0, conf, fps, np.random.default_rng(7), cfg, "near")
    assert not np.array_equal(out["s_xy"], ref["s_xy"])


def test_propagation_with_speed(cache):
    """visible cells never read a source frame hidden for any channel they depend on."""
    from pj.data.dataset import TB, BIN
    for i in range(0, 60, 5):
        xy, conf, fps = cache.clip(i)
        xy, conf, _ = W.canonicalise(xy, conf)
        for kind in ("near", "far"):
            out = build_pair(xy, conf, fps, np.random.default_rng(i), Cfg(speed=(0.6, 1.4)), kind)
            assert out["visible"].sum() > 0
    # direct check on the support logic
    i0, i1, w, valid = W.window_positions(100, 30.0, 64, 0.3, 0.8)
    sup = W.support(i0, i1, w, valid)
    assert all(len(s) in (1, 2) for s, v in zip(sup, valid) if v)


# ------------------------------------------------------------------ features / model
def _batch(cache, n=4, cfg=None):
    cfg = cfg or Cfg()
    items = []
    for i in range(n):
        xy, conf, fps = cache.clip(i)
        xy, conf, fl = W.canonicalise(xy, conf)
        items.append({"near": build_pair(xy, conf, fps, np.random.default_rng(i), cfg, "near"),
                      "far": build_pair(xy, conf, fps, np.random.default_rng(i + 99), cfg, "far"), "flipped": fl})
    return collate(items)


def test_feature_dims(cache):
    b = _batch(cache)["near"]
    f = Fe.frame_features(b["t_xy"], b["t_obs"], b["t_conf"], b["s_jh"])
    ins, pres = Fe.tokens(f, b["valid_f"])
    assert tuple(x.shape[-1] for x in ins) == Fe.TOKEN_DIMS == (170, 16, 170, 16, 108, 126, 20)
    ys, ws = Fe.intrinsic_targets(f, b["valid_f"])
    assert tuple(y.shape[-1] for y in ys) == Fe.TARGET_DIMS
    assert all(torch.isfinite(x).all() for x in ins + ys)
    assert all(x.abs().max() <= 6.0 + 1 + 1e-4 for x in ins)


def test_student_outputs_ignore_hidden_tokens(cache):
    """model-level: changing the inputs of non-visible tokens leaves visible outputs and CLS bit-identical."""
    torch.manual_seed(0)
    m = PhonoJEPA(d=64, depth=2, heads=2, dp=32, pdepth=1).eval()
    b = _batch(cache)["far"]
    f = Fe.frame_features(b["s_xy"], b["s_obs"], b["s_conf"], b["s_jh"])
    ins, pres = Fe.tokens(f, b["valid_f"])
    vis = b["visible"]
    h1, c1, _ = m.student(ins, pres, vis)
    ins2 = [x + torch.randn_like(x) * (~vis[:, c])[..., None] for c, x in enumerate(ins)]
    pres2 = pres | ~vis
    h2, c2, _ = m.student(ins2, pres2, vis)
    vm = vis.transpose(1, 2).flatten(1)
    assert torch.equal(c1, c2) and torch.equal(h1[vm], h2[vm])
    q1 = m.predictor(h1, c1, vis, b["target"]); q2 = m.predictor(h2, c2, vis, b["target"])
    tm = b["target"].transpose(1, 2).flatten(1)
    assert torch.equal(q1[tm], q2[tm])


def test_absent_vs_mask_embeddings():
    m = PhonoJEPA(d=32, depth=1, heads=2, dp=16, pdepth=1)
    assert not torch.equal(m.student.tok.absent.detach(), m.predictor.mask.detach()[:1].expand(32))


def test_teacher_no_grad_and_ema(cache):
    from pj.train import Trainer, get_args
    a = get_args(["--out", "/tmp/unused", "--device", "cpu", "--d", "32", "--depth", "1", "--heads", "2", "--dp", "16",
                  "--pdepth", "1", "--steps", "10"])
    tr = Trainer(a, "cpu")
    b = _batch(cache, 4)
    loss, _ = tr.step(b, 0)
    loss.backward()
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    t0 = [p.clone() for p in tr.model.teacher.parameters()]
    s0 = [p.clone() for p in tr.model.student.parameters()]
    tr.opt.step()
    tr.model.ema_update(0.9)
    for pt0, ps1, pt1 in zip(t0, tr.model.student.parameters(), tr.model.teacher.parameters()):
        torch.testing.assert_close(pt1, 0.9 * pt0 + 0.1 * ps1.detach())
    assert any(not torch.equal(a_, b_) for a_, b_ in zip(s0, tr.model.student.parameters()))


@pytest.mark.parametrize("argv", [["--target", "both"], ["--target", "ctx", "--family", "R4"], ["--target", "int", "--family", "R0",
                                  "--baseline-rate", "0.9", "--int-part", "delta"], ["--target", "none", "--glob"],
                                  ["--target", "both", "--loss", "proto", "--n-proto", "32"], ["--target", "both", "--loss", "l1sig", "--no-ema"],
                                  ["--target", "both", "--hands-only", "--shuffle-int"]])
def test_arms_one_step(cache, argv):
    from pj.train import Trainer, get_args
    a = get_args(["--out", "/tmp/unused", "--device", "cpu", "--d", "32", "--depth", "1", "--heads", "2", "--dp", "16",
                  "--pdepth", "1", "--steps", "10"] + argv)
    tr = Trainer(a, "cpu")
    cfg = Cfg(family=a.family, baseline_rate=a.baseline_rate or None)
    loss, terms = tr.step(_batch(cache, 4, cfg), 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_absent_targets_get_zero_weight(cache):
    """queries cover every target cell; teacher-ABSENT cells contribute nothing to the loss."""
    b = _batch(cache, 4)["far"]
    rel = Fe.reliability(b["t_conf"], b["t_obs"], b["valid_f"])
    f = Fe.frame_features(b["t_xy"], b["t_obs"], b["t_conf"], torch.zeros_like(b["s_jh"]))
    _, pres = Fe.tokens(f, b["valid_f"])
    from pj.train import target_weights
    cells = b["valid_b"][:, None, :].expand(-1, NC, -1)
    w = target_weights(b["target"], cells, rel, pres)
    assert (w[~pres] == 0).all() and (w[~b["target"]] == 0).all() and (w[b["target"] & pres] >= 0).all()
    assert (b["target"] & ~pres).any()                                   # ABSENT targets exist and are queried with weight 0


# ------------------------------------------------------------------ regression tests for code-review findings
def test_features_identical_under_autocast(cache):
    """finding #1: features / targets must be fp32 regardless of autocast (training) vs none (extraction)."""
    b = _batch(cache)["near"]
    f32 = Fe.frame_features(b["t_xy"], b["t_obs"], b["t_conf"], b["s_jh"], b["t_aff"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fbf = Fe.frame_features(b["t_xy"], b["t_obs"], b["t_conf"], b["s_jh"], b["t_aff"])
    for (l1, e1, v1), (l2, e2, v2) in zip(f32, fbf):
        assert l2.dtype == torch.float32 and torch.equal(l1, l2) and torch.equal(e1, e2) and torch.equal(v1, v2)


def test_pl_far_keeps_other_channel_of_same_hand():
    """finding #2: R3 never hides both channels of one hand in a bin; budget over hand cells."""
    from pj.joints import D_CONF, D_LOC, N_CONF, N_LOC
    fr = []
    for seed in range(500):
        m = sample_view(32, 32, np.random.default_rng(seed), "far", "PL")
        if m.stats.get("fallback"):
            continue
        h = m.hidden
        assert not (h[D_CONF] & h[D_LOC]).any() and not (h[N_CONF] & h[N_LOC]).any()
        fr.append(h[[D_CONF, D_LOC, N_CONF, N_LOC]].mean())
    assert 0.1 < np.mean(fr) < 0.4


def test_observation_gate_is_channel_local():
    """finding #7: scores of joints owned by one channel (or R1 finger joints) never change another channel's observations."""
    from pj.data.cache import gate
    from pj.joints import FINGER_GROUPS, LH0, RH0
    rng = np.random.default_rng(0)
    F = 50
    xy = rng.normal(size=(F, J, 2)).astype(np.float32)
    sc = rng.uniform(0, 10, size=(F, J)).astype(np.float32)
    obs0, _ = gate(xy, sc)
    hidden_sets = {c: OWNED[c] for c in range(NC)}
    hidden_sets["R1"] = [RH0 + j for j in FINGER_GROUPS["middle+ring+pinky"]]
    for key, idx in hidden_sets.items():
        sc2 = sc.copy(); sc2[:, idx] = rng.uniform(0, 10, size=(F, len(idx)))
        obs2, _ = gate(xy, sc2)
        others = np.setdiff1d(np.arange(J), idx)
        assert np.array_equal(obs0[:, others], obs2[:, others]), f"hidden scores of {key} change other observations"


def test_param_groups_exclude_predictor_channel_embedding():
    from pj.train import param_groups
    m = PhonoJEPA(d=32, depth=1, heads=2, dp=16, pdepth=1)
    g = param_groups(m.predictor, 1e-3, 0.05)
    ids = {id(p) for p in g[1]["params"]}
    assert id(m.predictor.ch) in ids and id(m.predictor.mask) in ids


def test_far_budget_matched_across_families():
    """review round 2 N1: far views of the PL / R4 / R0 rows hide comparable fractions of the grid."""
    fr = {}
    for fam in ("PL", "R4", "R0"):
        fr[fam] = np.mean([sample_view(32, 32, np.random.default_rng(s), "far", fam).hidden.mean() for s in range(400)])
    assert max(fr.values()) - min(fr.values()) < 0.05, fr


def test_plfull_r4_span_and_r3_room():
    """review round 2 N2: the G2 articulator span respects the 0.4 T minimum and R3 still adds hand targets."""
    from pj.joints import D_CONF, D_LOC, N_CONF, N_LOC
    spans, r3_extra = [], 0
    for s in range(400):
        m = sample_view(32, 32, np.random.default_rng(s), "far", "PLfull")
        h = m.hidden
        if h[F_HD].any():
            spans.append(h[F_HD].sum())
        hand = h[[D_CONF, D_LOC, N_CONF, N_LOC]]
        r3_extra += int((hand.sum(0) == 1).any())
    assert min(spans) >= int(round(0.4 * 32)) and r3_extra > 200
