"""EX11 evaluation (val LOSO): H7 / C7 32-frame students vs B0 (16-frame original, stride 4).
γ32(t) = max(γ(t), γ(t+16)) from the original model's stride-2 activity cache (padded clips:
missing starts fall back to the last available start)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, pool_clips, ranks_from_scores  # noqa
from diag_order import binned, seqs  # noqa: E402
from eval_pilot import arm_windows, teacher_cache  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def main():
    clips, off, start, act = teacher_cache()
    gamma = activity_weights(act)
    gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
    y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])

    def loso(F):
        r = np.zeros(len(y), int)
        for s in np.unique(sg):
            q, g = sg == s, sg != s
            r[q] = ranks_from_scores(class_max_scores(F[q], F[g], y[g], len(gl_)), y[q])
        return r

    R = {}
    win, aoff, keep = arm_windows("B0", clips, off, start)
    R["B0_R0"] = loso(l2n(pool_clips(win, aoff, gamma[keep])[0]))
    R["B0_R1"] = loso(binned(seqs(aoff, win, act[keep]), 2))
    for arm in ("H7", "C7"):
        d = RT / "longwin" / arm / "val_feats"
        if not (d / "clips.jsonl").exists():
            print("missing", arm); continue
        W = np.load(d / "windows_features.npy"); st = np.load(d / "window_start.npy"); ao = np.load(d / "offsets.npy")
        g32 = np.zeros(len(W), np.float32); a32 = np.zeros((len(W), 2), np.float32)
        for i in range(len(clips)):
            pos = {s: off[i] + k for k, s in enumerate(start[off[i]:off[i + 1]])}
            last = max(pos)
            for j in range(ao[i], ao[i + 1]):
                p1, p2 = pos.get(int(st[j]), pos[min(int(st[j]), last)]), pos.get(int(st[j]) + 16, pos[min(int(st[j]) + 16, last)])
                g32[j] = max(gamma[p1], gamma[p2]); a32[j] = np.maximum(act[p1], act[p2])
        R[f"{arm}_R0"] = loso(l2n(pool_clips(W, ao, g32)[0]))
        R[f"{arm}_R1"] = loso(binned(seqs(ao, W, a32), 2))
    res = {k: metrics_from_ranks(v, y) for k, v in R.items()}
    for k, v in res.items():
        print(f"{k:6s} R@1 {v['R@1']:.4f} MRR {v['MRR']:.4f}", flush=True)
    comps = {}
    for a, b in (("C7_R0", "H7_R0"), ("B0_R0", "H7_R0"), ("B0_R1", "H7_R0"), ("B0_R0", "C7_R0"), ("B0_R1", "H7_R1"), ("C7_R1", "H7_R1")):
        if a in R and b in R:
            bs = paired_cluster_bootstrap(per_query(R[a])["R@1"], per_query(R[b])["R@1"], sg)
            comps[f"{b}-{a}"] = bs
            print(f"{b}-{a} {bs['diff']:+.4f} [{bs['ci95'][0]:+.4f},{bs['ci95'][1]:+.4f}] pos {bs['n_clusters_positive']}/6")
    json.dump({"metrics": res, "comparisons": comps}, open(RT / "longwin" / "eval_ex11.json", "w"), indent=1)


if __name__ == "__main__":
    main()
