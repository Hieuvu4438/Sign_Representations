"""Does the readout change conclusions about SSL objectives? EX02 arms (B0 original, M0 XWP,
C0 matched control, A0 anchor-only) on ASL Citizen val LOSO, readout R0 (paper) vs R1 (bins2)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores, per_query, paired_cluster_bootstrap
from diag_order import binned, seqs
from eval_pilot import arm_windows, teacher_cache
clips, off, start, act = teacher_cache()
gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
def loso(F):
    r = np.zeros(len(y), int)
    for s in np.unique(sg):
        q, g = sg == s, sg != s; r[q] = ranks_from_scores(class_max_scores(F[q], F[g], y[g], len(gl_)), y[q])
    return r
R = {}
for arm in ("B0", "M0", "C0", "A0"):
    win, aoff, keep = arm_windows(arm, clips, off, start)
    a = act[keep]
    R[(arm, "R0")] = loso(l2n(pool_clips(win, aoff, activity_weights(a))[0]))
    R[(arm, "R1")] = loso(binned(seqs(aoff, win, a), 2))
    print(arm, " ".join(f"{ro}: R@1 {metrics_from_ranks(R[(arm, ro)])['R@1']:.4f}" for ro in ("R0", "R1")), flush=True)
for ro in ("R0", "R1"):
    for arm in ("M0", "C0", "A0"):
        b = paired_cluster_bootstrap(per_query(R[("B0", ro)])["R@1"], per_query(R[(arm, ro)])["R@1"], sg)
        print(f"{ro}: {arm}-B0 {b['diff']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] pos {b['n_clusters_positive']}/6")
