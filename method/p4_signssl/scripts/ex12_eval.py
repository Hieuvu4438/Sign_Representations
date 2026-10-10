"""EX12-v1 (LOCKED): evaluate a trained H008 aggregator on a test split vs locked readout ranks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h004_train import Aggregator, embed, load_seqs  # noqa: E402
from p4.retrieval import class_max_scores, metrics_from_ranks, paired_cluster_bootstrap, per_query, ranks_from_scores  # noqa: E402
from paper_stats import signflip_p  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"
REF = {"aslc": ("EX05_aslc_R0/ranks_features_activity.npy", "EX05_aslc_R12/ranks_bins2.npy", "EX05_aslc_R12/ranks_chamfer.npy"),
       "wlasl": ("EX05_wlasl_R0/ranks_features_activity.npy", "EX05_wlasl_R12/ranks_bins2.npy", "EX05_wlasl_R12/ranks_chamfer.npy"),
       "msasl": ("EX07_ranks_R0.npy", "EX07_ranks_R1.npy", "EX07_ranks_R2.npy")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(REF))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    gc, GX, GA, GM = load_seqs(f"{a.dataset}_train_s2")
    qc, QX, QA, QM = load_seqs(f"{a.dataset}_test_s2")
    assert all(c["split"] == "test" for c in qc)
    model = Aggregator().cuda(); model.load_state_dict(torch.load(a.model, map_location="cuda")); model.eval()
    G, Q = embed(model, GX, GA, GM, "cuda"), embed(model, QX, QA, QM, "cuda")
    gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
    gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
    rh = ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy)
    if a.dataset == "aslc":
        cl = np.array([c["signer"] for c in qc])
    elif a.dataset == "wlasl":
        sig = {i["video_id"]: i["signer_id"] for e in json.load(open(RT / "external/wlasl/WLASL_v0.3.json")) for i in e["instances"]}
        cl = np.array([sig.get(c["sample_id"], -1) for c in qc])
    else:
        cl = qy
    refs = {k: np.load(RT / "eval" / f) for k, f in zip(("R0", "R1", "R2"), REF[a.dataset])}
    out = {"dataset": a.dataset, "H8": metrics_from_ranks(rh, qy), **{k: metrics_from_ranks(v, qy) for k, v in refs.items()}, "comparisons": {}}
    for k, v in refs.items():
        bs = paired_cluster_bootstrap(per_query(v)["R@1"], per_query(rh)["R@1"], cl, n_boot=10000)
        c = {"diff": bs["diff"], "ci95": bs["ci95"], "clusters_positive": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
        if a.dataset == "aslc":
            c["signflip_p_one_sided"] = signflip_p(bs["per_cluster"])
        out["comparisons"][f"H8-{k}"] = c
    if a.dataset == "aslc":
        d = out["comparisons"]["H8-R1"]
        out["verdict"] = ("SUPPORTED_WITHIN_SCOPE" if d["diff"] >= 0.01 and d["ci95"][0] > 0 else ("REFUTED" if d["diff"] <= 0 else "INCONCLUSIVE"))
    np.save(RT / "eval" / f"EX12{a.tag}_{a.dataset}_ranks_H8.npy", rh)
    json.dump(out, open(RT / "eval" / f"EX12{a.tag}_{a.dataset}.json", "w"), indent=1)
    print(json.dumps({k: (v["R@1"] if isinstance(v, dict) and "R@1" in v else v) for k, v in out.items() if k != "comparisons"}), json.dumps(out["comparisons"]))


if __name__ == "__main__":
    main()
