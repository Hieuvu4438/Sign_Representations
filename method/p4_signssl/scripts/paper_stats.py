"""R002 E3–E5: per-signer table + exact sign-flip test (ASL Citizen test), WLASL signer-clustered
CIs (official WLASL_v0.3 signer IDs), class-macro R@1 with recomputed bootstrap, and the list of
all readout variants that were scored on test (disclosure). Reads existing rank files only."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.retrieval import metrics_from_ranks, paired_cluster_bootstrap, per_query  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def clips(d):
    return [json.loads(l) for l in open(RT / "feats" / d / "clips.jsonl")]


def signflip_p(per_cluster):
    v = np.array(list(per_cluster.values()))
    obs = v.mean(); cnt = 0; tot = 0
    for s in itertools.product([1, -1], repeat=len(v)):
        tot += 1; cnt += (np.array(s) * v).mean() >= obs
    return cnt / tot


def macro_boot(r_a, r_b, y, clusters, n=2000, seed=0):
    """class-macro R@1 difference, bootstrap over clusters, macro recomputed in each resample."""
    rng = np.random.default_rng(seed); uc = np.unique(clusters); idx = [np.where(clusters == c)[0] for c in uc]
    def macro(r, ii):
        yy = y[ii]; hit = (r[ii] == 1); cl = np.unique(yy)
        return np.mean([hit[yy == c].mean() for c in cl])
    full = macro(r_b, np.arange(len(y))) - macro(r_a, np.arange(len(y)))
    bs = []
    for _ in range(n):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(uc), len(uc))])
        bs.append(macro(r_b, ii) - macro(r_a, ii))
    return float(full), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    out = {}
    # ASL Citizen test: per-signer and sign-flip
    c = clips("aslc_test_s2"); sg = np.array([x["signer"] for x in c]); y = np.array([x["gloss"] for x in c])
    r0 = np.load(RT / "eval/EX05_aslc_R0/ranks_features_activity.npy")
    rs = {"R1": np.load(RT / "eval/EX05_aslc_R12/ranks_bins2.npy"), "R2": np.load(RT / "eval/EX05_aslc_R12/ranks_chamfer.npy")}
    per = {}
    for s in np.unique(sg):
        m = sg == s
        per[s] = {"n": int(m.sum()), "R0": float((r0[m] == 1).mean()), **{f"d{k}": float((v[m] == 1).mean() - (r0[m] == 1).mean()) for k, v in rs.items()}}
    out["aslc_per_signer"] = per
    for k, v in rs.items():
        bs = paired_cluster_bootstrap(per_query(r0)["R@1"], per_query(v)["R@1"], sg, n_boot=10000)
        out[f"aslc_{k}"] = {"diff": bs["diff"], "ci95": bs["ci95"], "signflip_p_one_sided": signflip_p(bs["per_cluster"]),
                            "classmacro": macro_boot(r0, v, y, sg)}
    # WLASL: signer-clustered
    meta = json.load(open(RT / "external/wlasl/WLASL_v0.3.json"))
    sig = {i["video_id"]: i["signer_id"] for e in meta for i in e["instances"]}
    c = clips("wlasl_test_s2"); wsg = np.array([sig.get(x["sample_id"], -1) for x in c]); wy = np.array([x["gloss"] for x in c])
    w0 = np.load(RT / "eval/EX05_wlasl_R0/ranks_features_activity.npy")
    for k, f in (("R1", "ranks_bins2.npy"), ("R2", "ranks_chamfer.npy")):
        v = np.load(RT / "eval/EX05_wlasl_R12" / f)
        bs = paired_cluster_bootstrap(per_query(w0)["R@1"], per_query(v)["R@1"], wsg, n_boot=10000)
        out[f"wlasl_{k}_signer_clustered"] = {"diff": bs["diff"], "ci95": bs["ci95"], "n_signers": bs["n_clusters"],
                                              "signers_positive": bs["n_clusters_positive"], "classmacro": macro_boot(w0, v, wy, wsg)}
    # MS-ASL: class-macro (class clusters)
    c = clips("msasl_test_s2"); my = np.array([x["gloss"] for x in c])
    m0 = np.load(RT / "eval/EX07_ranks_R0.npy")
    for k in ("R1", "R2"):
        out[f"msasl_{k}_classmacro"] = macro_boot(m0, np.load(RT / f"eval/EX07_ranks_{k}.npy"), my, my)
    # disclosure: every variant scored on test
    disc = {}
    for d in ("EX05_aslc_R12", "EX05_wlasl_R12"):
        disc[d] = {k: round(100 * v["R@1"], 2) for k, v in json.load(open(RT / "eval" / d / "metrics.json")).items() if isinstance(v, dict) and "R@1" in v}
    out["test_scored_variants_R@1"] = disc
    json.dump(out, open(RT / "eval" / "paper_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "aslc_per_signer"}, indent=1))
    for s, v in sorted(per.items()):
        print(f"{s:4s} n={v['n']:4d} R0 {100*v['R0']:.1f} dR1 {100*v['dR1']:+.2f} dR2 {100*v['dR2']:+.2f}")


if __name__ == "__main__":
    main()
