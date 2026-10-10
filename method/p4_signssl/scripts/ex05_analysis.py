"""EX05-v1 confirmatory analysis (LOCKED protocol). Reads rank files produced by
eval_retrieval.py (R0 = features:activity) and diag_order.py (R1 = bins2, R2 = chamfer)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import metrics_from_ranks, paired_cluster_bootstrap, per_query  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - k) * pvals[i]))
        adj[i] = run
    return adj


def boot_p(a, b, clusters, n=10000, seed=0):
    """two-sided bootstrap p for mean(b-a)=0 via cluster resampling (percentile method)."""
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    uc = np.unique(clusters)
    idx = [np.where(clusters == c)[0] for c in uc]
    sums = np.array([(b[i] - a[i]).sum() for i in idx]); cnt = np.array([len(i) for i in idx])
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(uc), (n, len(uc)))
    bs = sums[d].sum(1) / cnt[d].sum(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return max(p, 1.0 / n), bs


def main():
    out = {}
    for ds, cl_key, e1, e2, qdir in [("aslc", "signer", "EX05_aslc_R0", "EX05_aslc_R12", "aslc_test_s2"),
                                     ("wlasl", "gloss", "EX05_wlasl_R0", "EX05_wlasl_R12", "wlasl_test_s2")]:
        clips = [json.loads(l) for l in open(RT / "feats" / qdir / "clips.jsonl")]
        assert all(c["split"] == "test" for c in clips)
        cl = np.array([c[cl_key] for c in clips])
        r = {"R0": np.load(RT / "eval" / e1 / "ranks_features_activity.npy"),
             "R1": np.load(RT / "eval" / e2 / "ranks_bins2.npy"),
             "R2": np.load(RT / "eval" / e2 / "ranks_chamfer.npy")}
        res = {k: metrics_from_ranks(v) for k, v in r.items()}
        comps = {}
        for k in ("R1", "R2"):
            a, b = per_query(r["R0"])["R@1"], per_query(r[k])["R@1"]
            bs = paired_cluster_bootstrap(a, b, cl, n_boot=10000)
            p, _ = boot_p(a, b, cl)
            comps[f"{k}-R0"] = {"diff_R@1": bs["diff"], "ci95": bs["ci95"], "p": p,
                                "clusters_positive": bs["n_clusters_positive"], "n_clusters": bs["n_clusters"],
                                "diff_MRR": float(per_query(r[k])["MRR"].mean() - per_query(r["R0"])["MRR"].mean())}
        out[ds] = {"metrics": res, "comparisons": comps, "cluster_unit": cl_key}
    adj = holm([out["aslc"]["comparisons"]["R1-R0"]["p"], out["aslc"]["comparisons"]["R2-R0"]["p"]])
    out["aslc"]["comparisons"]["R1-R0"]["p_holm"], out["aslc"]["comparisons"]["R2-R0"]["p_holm"] = adj
    for k in ("R1", "R2"):
        c = out["aslc"]["comparisons"][f"{k}-R0"]
        w = out["wlasl"]["comparisons"][f"{k}-R0"]
        if c["diff_R@1"] <= 0:
            verdict = "REFUTED"
        elif c["diff_R@1"] >= 0.02 and c["p_holm"] < 0.05 and c["ci95"][0] > 0 and w["diff_R@1"] > 0:
            verdict = "SUPPORTED_WITHIN_SCOPE"
        else:
            verdict = "INCONCLUSIVE"
        out[f"verdict_{k}"] = verdict
    json.dump(out, open(RT / "eval" / "EX05_result.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
