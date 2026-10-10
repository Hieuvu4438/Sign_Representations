"""Label-free pseudo-signer clusters for a dataset's unlabelled train windows (H012 rule, fixed in EX14-v1):
rest-segment descriptor (mean of L2-normalised windows outside the active span; fallback first+last window),
K-means with K in {20, 40, 60, 80, 120} selected by cosine silhouette on 6,000 clips (rng 1).
Signer IDs (when available) are used only to report NMI afterwards.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_order import load  # noqa: E402
from explore_restpose import rest_desc  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="e.g. wlasl_train_s2")
    a = ap.parse_args()
    clips, off, win, act = load(RT / "feats" / a.feats)
    assert all(c.get("split", "train") == "train" for c in clips)
    R, fb = rest_desc(off, win, act)
    samp = np.random.default_rng(1).choice(len(R), min(6000, len(R)), replace=False)
    res = {"n": len(R), "fallback_rate": fb / len(R), "K": {}}
    best = None
    for K in (20, 40, 60, 80, 120):
        km = KMeans(K, n_init=4, random_state=0).fit(R)
        sil = float(silhouette_score(R[samp], km.labels_[samp], metric="cosine"))
        res["K"][K] = {"silhouette": sil}
        if best is None or sil > best[1]:
            best = (K, sil, km.labels_)
    res["K_selected"] = best[0]
    sg = [c.get("signer") for c in clips]
    if a.feats.startswith("wlasl"):  # scoring only: official WLASL_v0.3 signer ids
        sig = {i["video_id"]: i["signer_id"] for e in json.load(open(RT / "external/wlasl/WLASL_v0.3.json")) for i in e["instances"]}
        sg = [sig.get(c["sample_id"]) for c in clips]
    if all(s not in (None, "NA") for s in sg):
        res["nmi_selected_vs_signer"] = float(normalized_mutual_info_score(sg, best[2]))
    np.save(RT / "explore" / f"restpose_clusters_{a.feats}.npy", best[2])
    json.dump(res, open(RT / "explore" / f"restpose_clusters_{a.feats}.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
