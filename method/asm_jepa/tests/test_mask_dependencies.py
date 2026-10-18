"""Normalisation leakage, dependency closure and the counterfactual (donor) leakage test (§9.3, §16)."""
import copy

import numpy as np
import pytest
import torch

from asm_jepa.data.dataset import SampleBuilder
from asm_jepa.data.normalization import body_normalize, hand_local
from asm_jepa.data.sampling import full_clip_indices
from asm_jepa.data.schema import MASKABLE, MODEL_TO_RAW, N_JOINTS, PART_ORDER, PART_SLICES
from asm_jepa.data.synthetic import synthetic_clip
from asm_jepa.features.masking import sample_raw_mask
from asm_jepa.features.tokens import TYPE_GEOMETRY, TYPE_RELATION
from asm_jepa.models.asm_jepa import Backbone

from conftest import cfg_for

N_P = 16


def _rich_clip(min_frames=120):
    """A synthetic clip with every part valid on (almost) all patches and > 64 frames."""
    for i in range(500):
        c = synthetic_clip(i, unusable_every=0)
        if c.num_frames >= min_frames and c.observed.all(0)[MODEL_TO_RAW].all() and c.frame_valid.all():
            return c
    raise RuntimeError("no rich synthetic clip")


def _batch(s):
    return {k: torch.as_tensor(np.asarray(v))[None] for k, v in s.items()}


def _context_inputs(bb, s):
    b = _batch(s)
    with torch.no_grad():
        x, idx, valid = bb.context_sequence(b, b["context"])
    return x[valid], idx[valid]


def _donor(clip, seed, units, n_frames=64, scale=40.0, alter_scores=False):
    """Replace hidden raw coordinates (and optionally scores) of the masked (part, patch) units with donor values."""
    rng = np.random.default_rng(seed)
    idx, valid = full_clip_indices(clip.num_frames, n_frames, rng, 0.15)  # identical first draws as the builder
    out = copy.deepcopy(clip)
    drng = np.random.default_rng(1234)
    for a, p in zip(*np.nonzero(units)):
        frames = idx[p * 4:(p + 1) * 4]
        frames = frames[frames >= 0]
        raw = MODEL_TO_RAW[MASKABLE[PART_ORDER[a]]]
        for f in frames:
            out.xy[f, raw] += drng.normal(0, scale, (len(raw), 2)).astype(np.float32)
            if alter_scores:
                out.score[f, raw] = drng.uniform(1.0, 9.0, len(raw)).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------- normalisation
def test_missing_wrist_or_mcp_gives_local_invalid_and_zero_is_not_observed():
    T = 3
    xy = np.random.default_rng(0).normal(0, 1, (T, N_JOINTS, 2)).astype(np.float32)
    obs = np.ones((T, N_JOINTS), bool)
    w = PART_SLICES["lh"].start
    obs[0, w] = False  # frame 0: no wrist
    obs[1, [w + 5, w + 9, w + 13]] = False  # frame 1: one MCP left -> < 2 distances
    local, lv = hand_local(xy, obs)
    assert not lv[0, PART_SLICES["lh"]].any() and not lv[1, PART_SLICES["lh"]].any()
    assert lv[2, PART_SLICES["lh"]].all() and lv[:, PART_SLICES["rh"]].all()
    assert (local[~lv] == 0).all()
    obs2 = obs.copy()
    obs2[:, 10] = False
    nb2 = body_normalize(xy, obs2, np.ones(T, bool))
    assert (nb2["xy"][:, 10] == 0).all()  # zero filled, and the observed flag stays False


def test_hidden_hand_does_not_change_visible_hand_scale_and_shoulders_are_public():
    rng = np.random.default_rng(0)
    T = 8
    xy = rng.normal(0, 50, (T, N_JOINTS, 2)).astype(np.float32) + 300
    obs = np.ones((T, N_JOINTS), bool)
    loc1, _ = hand_local(body_normalize(xy, obs, np.ones(T, bool))["xy"], obs)
    xy2 = xy.copy()
    xy2[4:, PART_SLICES["lh"]] += rng.normal(0, 80, (4, 21, 2)).astype(np.float32)  # hidden LH frames 4..7
    loc2, _ = hand_local(body_normalize(xy2, obs, np.ones(T, bool))["xy"], obs)
    assert np.allclose(loc1[:4], loc2[:4])  # per-frame hand scale: visible LH frames unchanged
    assert np.allclose(loc1[:, PART_SLICES["rh"]], loc2[:, PART_SLICES["rh"]])
    xy3 = xy.copy()
    xy3[:, 2:4] += 30  # elbows (maskable body joints) never move the body frame
    n1, n3 = body_normalize(xy, obs, np.ones(T, bool)), body_normalize(xy3, obs, np.ones(T, bool))
    assert n1["body_scale_px"] == n3["body_scale_px"]
    assert np.allclose(n1["xy"][:, PART_SLICES["lh"]], n3["xy"][:, PART_SLICES["lh"]])


# ------------------------------------------------------------------------------------------- closure
def test_closure_lh_mask_removes_lh_relations_and_overlapping_paths(full_cfg, stats):
    b = SampleBuilder(full_cfg, stats)
    clip = _rich_clip()
    units = np.zeros((4, N_P), bool)
    units[1, 6] = True  # LH patch 6
    s = b.build(clip, np.random.default_rng(3), "ssl", force_mask=units)
    A = b.layout.arrays
    tgt = s["target"]
    rel = (A["type"] == TYPE_RELATION) & (A["patch"] == 6)
    assert tgt[rel & np.isin(A["sub"], [0, 1])].all()  # LH-RH and LH-nose at patch 6 removed
    assert not tgt[rel & (A["sub"] == 2)].any()  # RH-nose untouched
    deps = s["context"] | s["target"]
    motion = A["type"] >= TYPE_GEOMETRY
    lh_paths = motion & np.isin(A["sub"], [0, 2])
    assert tgt[motion & (A["sub"] == 1)].sum() == 0  # right-wrist paths do not read LH
    assert tgt[lh_paths].any()  # at least the centre-6 windows of LH families are removed
    assert (deps == s["token_valid"]).all()


def test_closure_face_mask_removes_hand_nose_relations(full_cfg, stats):
    b = SampleBuilder(full_cfg, stats)
    units = np.zeros((4, N_P), bool)
    units[3, 9] = True
    s = b.build(_rich_clip(), np.random.default_rng(4), "ssl", force_mask=units)
    A = b.layout.arrays
    rel9 = (A["type"] == TYPE_RELATION) & (A["patch"] == 9)
    assert s["target"][rel9 & np.isin(A["sub"], [1, 2])].all() and not s["target"][rel9 & (A["sub"] == 0)].any()
    assert not s["target"][A["type"] >= TYPE_GEOMETRY].any()  # face never enters wrist paths


# ------------------------------------------------------------------------------------------- counterfactual
CASES = {
    "lh": [(1, 5), (1, 6)],
    "rh": [(2, 10), (2, 11), (2, 12)],
    "face": [(3, 3), (3, 4)],
    "body_elbows": [(0, 7), (0, 8)],
    "boundary_patches": [(1, 0), (2, 15), (3, 15)],
    "multi_scale_both_hands": [(1, 2), (2, 2), (1, 13), (2, 13)],
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("alter_scores", [False, True])
def test_counterfactual_context_invariance(full_cfg, stats, case, alter_scores):
    cfg = full_cfg
    b = SampleBuilder(cfg, stats)
    bb = Backbone(cfg).eval()
    clip = _rich_clip()
    units = np.zeros((4, N_P), bool)
    for a, p in CASES[case]:
        units[a, p] = True
    seed = 77
    s1 = b.build(clip, np.random.default_rng(seed), "ssl", force_mask=units)
    s2 = b.build(_donor(clip, seed, units, alter_scores=alter_scores), np.random.default_rng(seed), "ssl",
                 force_mask=units)
    for k in ("context", "target", "token_valid", "token_time", "mask_units", "log_duration"):
        assert np.array_equal(s1[k], s2[k]), k
    x1, i1 = _context_inputs(bb, s1)
    x2, i2 = _context_inputs(bb, s2)
    assert torch.equal(i1, i2)
    assert torch.allclose(x1, x2, atol=1e-6, rtol=1e-5), f"context leaks hidden values ({case})"
    # the donor really changed hidden values, so teacher targets are allowed to change
    assert not np.allclose(s1["joints_clean"], s2["joints_clean"])
    t1, t2 = (Backbone.embed(bb, _batch(s), "_clean")[0][torch.as_tensor(s["target"])] for s in (s1, s2))
    assert not torch.allclose(t1, t2)


def test_disabling_closure_is_detected_as_leakage(stats):
    """Sanity check of the test itself: without closure the relation/motion context must leak."""
    cfg = cfg_for("full", "mask.dependency_closure=false")
    b = SampleBuilder(cfg, stats)
    bb = Backbone(cfg).eval()
    clip = _rich_clip()
    units = np.zeros((4, N_P), bool)
    units[1, 5] = units[1, 6] = True
    s1 = b.build(clip, np.random.default_rng(5), "ssl", force_mask=units)
    s2 = b.build(_donor(clip, 5, units), np.random.default_rng(5), "ssl", force_mask=units)
    x1, _ = _context_inputs(bb, s1)
    x2, _ = _context_inputs(bb, s2)
    assert not torch.allclose(x1, x2, atol=1e-6, rtol=1e-5)


# ------------------------------------------------------------------------------------------- mask sampler
def test_mask_sampler_ratio_and_constraints(full_cfg):
    rng = np.random.default_rng(0)
    valid = np.ones((4, N_P), bool)
    for _ in range(200):
        m, info = sample_raw_mask(valid, rng, full_cfg["mask"])
        assert m.sum() == round(0.5 * valid.sum()) or info["fallbacks"] > 0
        vis = valid & ~m
        assert vis.sum() >= 8
        for a in (1, 2):
            assert vis[a].sum() >= int(np.ceil(0.25 * valid[a].sum()))
    sparse = np.zeros((4, N_P), bool)
    sparse[1, :3] = True
    m, info = sample_raw_mask(sparse, rng, full_cfg["mask"])
    assert not (m & ~sparse).any()  # invalid units are never targets
    assert (sparse & ~m).sum() >= 1


def test_mask_sampling_never_reads_coordinates(full_cfg, stats):
    b = SampleBuilder(full_cfg, stats)
    clip = _rich_clip()
    s1 = b.build(clip, np.random.default_rng(9), "ssl")
    clip2 = copy.deepcopy(clip)
    clip2.xy[:, 91:] += 25.0
    s2 = b.build(clip2, np.random.default_rng(9), "ssl")
    assert np.array_equal(s1["mask_units"], s2["mask_units"])
