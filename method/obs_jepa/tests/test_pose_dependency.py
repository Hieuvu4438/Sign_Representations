"""B dependency safety: hidden-value perturbation (proposal11 §6.6, §11.6) — with the mask fixed, changing the raw
coordinates of hidden cells must leave every student-side input and observable-anchor target unchanged, while the
teacher-side inputs do change (sanity)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.data.clips import video_meta  # noqa: E402
from oj.data.pose_dataset import PoseWindows, read_corpus  # noqa: E402
from oj.paths import ASLC002_CORPUS  # noqa: E402

STUDENT_KEYS = ["st_node", "vis", "hid"]


def test_hidden_value_perturbation(n=12):
    rows = read_corpus(ASLC002_CORPUS)[:n * 5:5]
    meta = {r["sample_id"]: video_meta(r["video"]) for r in rows}
    changed_teacher = 0
    for policy in ("core", "whole", "generic"):
        ds = PoseWindows(rows, meta, policy=policy, train=True)
        for i in range(len(rows)):
            a = ds.build(i, np.random.default_rng(i))
            lm, _ = ds._lm(i)
            hid_frames = np.repeat(a["hid"].numpy(), 2, axis=0)
            # perturb every source frame that maps to a hidden cell: rebuild index map
            from oj.data.clips import window_indices
            W, H, fps, nf = ds.meta[ds.rows[i]["sample_id"]]
            st = np.random.default_rng(i)
            from oj.data.clips import random_start
            s0 = random_start(min(nf, len(lm.xy)), fps, ds.T, ds.fps, st)
            idx, _, _ = window_indices(min(nf, len(lm.xy)), fps, ds.T, ds.fps, s0)
            xy = lm.xy.copy()
            for t, src in enumerate(idx):
                if src >= 0:
                    xy[src, hid_frames[t]] += np.random.default_rng(t).normal(0, 50, size=(int(hid_frames[t].sum()), 2)).astype(np.float32)
            orig = lm.xy; lm.xy = xy
            b = ds.build(i, np.random.default_rng(i))
            lm.xy = orig
            for k in STUDENT_KEYS:
                assert torch.equal(a[k], b[k]), (policy, i, k)
            for k in a["st_loc"]:
                assert torch.equal(a["st_loc"][k], b["st_loc"][k]), (policy, i, "st_loc", k)
            for k in a:
                if k.startswith("anc_") or k.startswith("ancm_"):
                    assert torch.equal(a[k], b[k]), (policy, i, k)
            changed_teacher += int(not torch.equal(a["te_node"], b["te_node"]))
    assert changed_teacher > 0, "perturbation did not reach the teacher side (test is vacuous)"
    return changed_teacher


if __name__ == "__main__":
    print("teacher-side changed in", test_hidden_value_perturbation(), "windows; student side invariant: OK")
