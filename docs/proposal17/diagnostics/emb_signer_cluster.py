"""D-ECLUS: label-free, transductive signer normalisation using clusters discovered IN THE EMBEDDING (proposal17 diagnostic G10).

The style-variable pseudo-signers (pseudo_signer.py) did not recover the signer-CORAL gain.  The embedding itself decodes signers
at ~90 %, so cluster the query set (and the gallery) in the embedding space, then apply per-cluster CORAL (+ label-free PCA whiten).
No signer id and no gloss label is used.  It is transductive (cluster statistics come from the query batch), i.e. a test-time
adaptation variant, to be reported separately from the inductive protocol.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from common import *  # noqa
import geo_headroom as G
import signer_norm as SN

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/emb_signer_cluster.json"


def main():
    d = load_baseline()
    tr, va = d["train"], d["val"]
    Xg, yg, sg = tr["X"], tr["y"], tr["signer"]
    Xq, yq, sq = va["X"], va["y"], va["signer"]
    mu_w, W = G.fit_pca_whiten(Xg, 512, 1.0)
    Zg, Zq = G.apply_lin(Xg, mu_w, W), G.apply_lin(Xq, mu_w, W)
    r1 = lambda Q, Gl: float(metrics(class_max_ranks(Q, yq, Gl, yg))["R1"])
    out = {"whiten_only": r1(Zq, Zg)}
    # reference: true signers on both sides
    Gc, Qc = SN.coral(Xg, sg), SN.coral(Xq, sq)
    m2, W2 = G.fit_pca_whiten(Gc, 512, 1.0)
    out["true_signers_both_CORAL+whiten"] = r1(G.apply_lin(Qc, m2, W2), G.apply_lin(Gc, m2, W2))
    print(out, flush=True)
    rng = np.random.default_rng(0)
    # cluster in a low-dim whitened space: signer structure is the dominant local structure
    for Kq in (6, 12, 24, 48):
        km = KMeans(Kq, n_init=4, random_state=0).fit(Zq[:, :64])
        lq = km.labels_
        out[f"Kq{Kq}_AMI_val"] = float(adjusted_mutual_info_score(sq, lq))
        Qn = SN.coral(Xq, lq)
        for gmode in ("true", "cluster70"):
            if gmode == "true":
                gl = sg
            else:
                gl = KMeans(70, n_init=2, random_state=0).fit(Zg[:, :64]).labels_
            Gn = SN.coral(Xg, gl)
            m3, W3 = G.fit_pca_whiten(Gn, 512, 1.0)
            out[f"Kq{Kq}_gallery-{gmode}_CORAL+whiten"] = r1(G.apply_lin(Qn, m3, W3), G.apply_lin(Gn, m3, W3))
        print({k: round(v, 3) for k, v in out.items() if k.startswith(f"Kq{Kq}")}, flush=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
