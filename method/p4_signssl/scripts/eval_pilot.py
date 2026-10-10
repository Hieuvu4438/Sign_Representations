"""EX02 pilot evaluation (val leave-one-signer-out), identical readout for every arm.

Arm features: val windows at stride 4 extracted with the arm's encoder. Pooling weights:
activity outputs of the ORIGINAL SignRep at the same window starts (from the stride-2
cache), so only the encoder differs between arms. B0 uses the cache itself (subsampled
to stride 4).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.phonology import load_asllex  # noqa: E402
from p4.retrieval import (activity_weights, class_max_scores, l2n, metrics_from_ranks,  # noqa: E402
                          paired_cluster_bootstrap, per_query, pool_clips, ranks_from_scores)
from diag_order import binned, seqs  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"
CACHE = RT / "feats" / "aslc_val_s2"


def teacher_cache():
    clips = [json.loads(l) for l in open(CACHE / "clips.jsonl")]
    return clips, np.load(CACHE / "offsets.npy"), np.load(CACHE / "window_start.npy"), np.load(CACHE / "windows_active.npy")


def arm_windows(name, clips, off, start, stride=4):
    """Return (windows, offsets, teacher_activity) at stride `stride` for an arm."""
    if name == "B0":
        win = np.load(CACHE / "windows_features.npy", mmap_mode="r")
        keep = np.concatenate([np.arange(off[i], off[i + 1])[start[off[i]:off[i + 1]] % stride == 0] for i in range(len(off) - 1)])
        cnt = np.array([(start[off[i]:off[i + 1]] % stride == 0).sum() for i in range(len(off) - 1)])
        return np.asarray(win[keep]), np.concatenate([[0], np.cumsum(cnt)]), keep
    d = RT / "feats" / f"val_s4_{name}"
    aclips = [json.loads(l) for l in open(d / "clips.jsonl")]
    assert [c["sample_id"] for c in aclips] == [c["sample_id"] for c in clips]
    aoff, astart = np.load(d / "offsets.npy"), np.load(d / "window_start.npy")
    # map each arm window to the cache index with the same clip and start
    keep = []
    for i in range(len(off) - 1):
        pos = {s: off[i] + k for k, s in enumerate(start[off[i]:off[i + 1]])}
        keep += [pos[s] for s in astart[aoff[i]:aoff[i + 1]]]
    return np.load(d / "windows_features.npy"), aoff, np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B0,M0,C0,A0")
    ap.add_argument("--ref", default="C0")
    ap.add_argument("--out", default=str(RT / "pilot" / "eval_ex02.json"))
    a = ap.parse_args()
    clips, off, start, act = teacher_cache()
    gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
    y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
    lex = load_asllex()
    mv = np.array([(lex.get(c["asllex"], {}).get("RepeatedMovement.2.0"), lex.get(c["asllex"], {}).get("Movement.2.0")) for c in clips], dtype=object)
    path_nonrep = np.array([(m[0] == "0" and m[1] not in (None, "None")) for m in mv])
    rep_or_none = np.array([(m[0] == "1" or m[1] == "None") for m in mv])

    def loso(F):
        r = np.zeros(len(y), int)
        for s in np.unique(sg):
            q, g = sg == s, sg != s
            r[q] = ranks_from_scores(class_max_scores(F[q], F[g], y[g], len(gl_)), y[q])
        return r

    res, ranks = {}, {}
    for arm in a.arms.split(","):
        if arm != "B0" and not (RT / "feats" / f"val_s4_{arm}" / "meta.json").exists():
            print("skip", arm); continue
        win, aoff, keep = arm_windows(arm, clips, off, start)
        w = activity_weights(act[keep])
        P, fb = pool_clips(win, aoff, w)
        r = loso(l2n(P)); ranks[arm] = r
        rb = loso(binned(seqs(aoff, win, act[keep]), 2))
        res[arm] = {"wmean": metrics_from_ranks(r, y), "bins2": metrics_from_ranks(rb, y), "zero_act_fallback": fb,
                    "R@1_path_nonrep": float((r[path_nonrep] == 1).mean()), "R@1_rep_or_none": float((r[rep_or_none] == 1).mean())}
        print(arm, json.dumps(res[arm]), flush=True)
    if a.ref in ranks:
        for arm, r in ranks.items():
            if arm == a.ref: continue
            pa, pb = per_query(ranks[a.ref]), per_query(r)
            res[f"{arm}-{a.ref}"] = {m: paired_cluster_bootstrap(pa[m], pb[m], sg) for m in ("R@1", "MRR")}
            print(arm, "-", a.ref, json.dumps(res[f"{arm}-{a.ref}"]["R@1"]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
