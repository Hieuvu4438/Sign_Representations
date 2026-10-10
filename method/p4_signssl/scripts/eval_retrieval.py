"""Frozen dictionary retrieval on cached per-window SignRep outputs.

Gallery = train clips (labels = gloss), queries = val clips. Reports DCG/MRR/R@K for
several pooling variants of the SAME window embeddings, so that differences come from
pooling only. Saves per-query ranks for paired comparisons.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import (activity_weights, class_max_scores, l2n, metrics_from_ranks,  # noqa: E402
                          pool_clips, ranks_from_scores)


def load(d):
    d = Path(d)
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    return {
        "clips": clips,
        "off": np.load(d / "offsets.npy"),
        "act": np.load(d / "windows_active.npy"),
        "start": np.load(d / "window_start.npy"),
        "d": d,
    }


def pooled(F, key, variant, stride_sub=1):
    win = np.load(F["d"] / f"windows_{key}.npy", mmap_mode="r")
    off, act = F["off"], F["act"]
    if stride_sub > 1:  # keep every k-th window per clip (emulates a larger stride)
        keep = np.concatenate([np.arange(off[i], off[i + 1])[::stride_sub] for i in range(len(off) - 1)])
        counts = np.array([len(range(off[i], off[i + 1])[::stride_sub]) for i in range(len(off) - 1)])
        win, act = win[keep], act[keep]
        off = np.concatenate([[0], np.cumsum(counts)])
    w = activity_weights(act) if variant == "activity" else None
    p, fb = pool_clips(np.asarray(win), off, w)
    return p, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="features:activity,features:mean,latent:mean,latent:activity,features:mean:s8")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    G, Q = load(a.gallery), load(a.query)
    assert G["clips"][0]["split"] == "train" and (Q["clips"][0]["split"] != "test" or a.allow_test)
    glosses = sorted({c["gloss"] for c in G["clips"]})
    lab = {g: i for i, g in enumerate(glosses)}
    gl = np.array([lab[c["gloss"]] for c in G["clips"]])
    ql = np.array([lab[c["gloss"]] for c in Q["clips"]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for v in a.variants.split(","):
        parts = v.split(":")
        key, var = parts[0], parts[1]
        sub = int(parts[2][1:]) // 2 if len(parts) > 2 else 1
        gp, gfb = pooled(G, key, var, sub)
        qp, qfb = pooled(Q, key, var, sub)
        s = class_max_scores(l2n(qp), l2n(gp), gl, len(glosses))
        r = ranks_from_scores(s, ql)
        m = metrics_from_ranks(r, ql)
        m.update(gallery_zero_activity_fallback=gfb, query_zero_activity_fallback=qfb)
        res[v] = m
        np.save(out / f"ranks_{v.replace(':', '_')}.npy", r)
        print(v, json.dumps(m), flush=True)
    json.dump({"results": res, "gallery": str(G["d"]), "query": str(Q["d"]), "n_classes": len(glosses)},
              open(out / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
