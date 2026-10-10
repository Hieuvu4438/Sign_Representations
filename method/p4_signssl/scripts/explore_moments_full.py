"""Exploratory (val only): full-val paired comparison of training-free moment vectors vs the H008 student.
M2D2 (3072-d) and M2D2-PCA1024 (PCA fit on unlabelled train vectors, matched to H008's 1024-d)
vs H008 D2 val ranks (`runtime/h008/D2/ranks_val.npy`), signer-cluster paired bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, ranks_from_scores  # noqa: E402
from diag_order import load, seqs  # noqa: E402
from explore_moments import readouts  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def main():
    gc, goff, gw, ga = load(RT / "feats" / "aslc_train_s2"); qc, qoff, qw, qa = load(RT / "feats" / "aslc_val_s2")
    G, Q = readouts(seqs(goff, gw, ga))["M2D2"], readouts(seqs(qoff, qw, qa))["M2D2"]
    mu = G.mean(0); _, _, Vt = np.linalg.svd(G - mu, full_matrices=False)
    Gp, Qp = l2n((G - mu) @ Vt[:1024].T), l2n((Q - mu) @ Vt[:1024].T)
    gl = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl)}
    gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
    cl = np.array([c["signer"] for c in qc])
    h8 = np.load(RT / "h008" / "D2" / "ranks_val.npy"); assert len(h8) == len(qy)
    out = {"H8_D2": metrics_from_ranks(h8)["R@1"]}
    for k, (g, q) in {"M2D2": (G, Q), "M2D2_pca1024": (Gp, Qp)}.items():
        r = ranks_from_scores(class_max_scores(q, g, gy, len(gl), device="cuda"), qy)
        np.save(RT / "explore" / f"ranks_val_{k}.npy", r)
        bs = paired_cluster_bootstrap(per_query(r)["R@1"], per_query(h8)["R@1"], cl, n_boot=10000)
        out[k] = metrics_from_ranks(r)["R@1"]
        out[f"H8-{k}"] = {"diff": bs["diff"], "ci95": bs["ci95"], "clusters_positive": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
    print(json.dumps(out))
    json.dump(out, open(RT / "explore" / "moments_full_val.json", "w"), indent=1)


if __name__ == "__main__":
    main()
