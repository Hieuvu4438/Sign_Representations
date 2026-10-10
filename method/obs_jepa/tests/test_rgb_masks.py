"""Mask invariants on real ASL Citizen windows (proposal11 §11.6 'Mask constraints')."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.clips import make_letterbox, random_start, video_meta, window_indices  # noqa: E402
from oj.data.landmarks import load_landmarks  # noqa: E402
from oj.masks.rgb_masks import RGBMaskSampler, token_ownership  # noqa: E402
from oj.paths import ASLC002_CORPUS  # noqa: E402


def windows(n=30, S=224, T=16, seed=0):
    rows = [json.loads(l) for l in open(ASLC002_CORPUS)][: n * 3: 3]
    rng = np.random.default_rng(seed)
    for r in rows:
        W, H, fps, nf = video_meta(r["video"]); lm = load_landmarks(Path(r["video"]).stem, W, H)
        idx, _, _ = window_indices(nf, fps, T, 25, random_start(nf, fps, T, 25, rng))
        cl = np.clip(idx, 0, len(lm.xy) - 1)
        lb = make_letterbox(W, H, S)
        xy = lb.apply(np.where(idx[:, None, None] >= 0, lm.xy[cl], np.nan)); ob = (idx[:, None] >= 0) & lm.observed[cl]
        yield xy, ob, np.broadcast_to(lb.token_valid()[None], (T // 2, S // 16, S // 16)).copy(), rng


def run(policy, S=224):
    sm = RGBMaskSampler(T=16, size=S, policy=policy, seed=1); out = []
    for xy, ob, valid, rng in windows(S=S):
        off = (rng.uniform(-48, 48), rng.uniform(-48, 48)) if policy == "jitter" else (0, 0)
        own, rel = token_ownership(xy, ob, 8, S // 16, offset=off); own_true, _ = token_ownership(xy, ob, 8, S // 16)
        m = sm.sample(own, rel, valid, owner_weight=own_true)
        allidx = np.concatenate([m.ctx, m.tgt, m.guard])
        assert len(np.unique(allidx)) == len(allidx), "C/M/G overlap"
        assert valid.ravel()[allidx].all(), "padding token used"
        assert len(m.tgt) + len(m.guard) == m.stats["budget"], m.stats
        assert len(m.ctx) == m.stats["Nv"] - m.stats["budget"]
        if policy in ("partial", "jitter", "generic_cov"):
            for k in ("visible_lh", "visible_rh"):
                if k in m.stats:
                    assert m.stats[k] >= 0.25 - 1e-6, (k, m.stats)
        out.append(m)
    return out


def test_all_policies():
    for S in (224, 384):
        for p in ("generic", "generic_cov", "partial", "whole", "jitter"):
            ms = run(p, S)
            if p.startswith("generic"):
                assert all(m.tgt_structured.sum() == 0 for m in ms)
            else:
                assert np.mean([m.tgt_structured.sum() > 0 for m in ms]) > 0.7


if __name__ == "__main__":
    for S in (224, 384):
        for p in ("generic", "generic_cov", "partial", "whole", "jitter"):
            ms = run(p, S)
            print(S, p, "ctx", len(ms[0].ctx), "tgt %.0f" % np.mean([len(m.tgt) for m in ms]),
                  "guard %.1f" % np.mean([len(m.guard) for m in ms]), "struct %.1f" % np.mean([m.tgt_structured.sum() for m in ms]),
                  "visLH %.2f visRH %.2f (true %.2f/%.2f)" % tuple(np.nanmean([m.stats.get(k, np.nan) for m in ms]) for k in ("visible_lh", "visible_rh", "visible_lh_true", "visible_rh_true")),
                  "fallback %.2f" % np.mean(["fallback" in m.stats for m in ms]))
    print("OK")
