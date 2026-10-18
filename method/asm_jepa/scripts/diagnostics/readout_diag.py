"""Offline readout diagnostics on saved frozen embeddings (train gallery / val queries only; test is never read).

D2 per-part retrieval, D4 signer vs gloss dominance, D5 centring / whitening fitted on TRAIN embeddings only.
usage: python scripts/diagnostics/readout_diag.py <run_dir>/embeddings [--out report.json]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from asm_jepa.evaluation.retrieval import evaluate_retrieval


def load(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta_json"}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def retr(gx, g, qx, q, n_classes, name, out):
    res, _, _ = evaluate_retrieval(l2(qx), q["class_ids"], l2(gx), g["class_ids"], n_classes, query_usable=q["usable"],
                                   gallery_usable=g["usable"], device="cpu")
    m = res["full"]
    out[name] = {k: m[k] for k in ("r1", "r5", "r10", "mrr", "dcg")}
    print(f"{name:42s} R@1 {100*m['r1']:6.2f}  R@5 {100*m['r5']:6.2f}  R@10 {100*m['r10']:6.2f}  DCG {100*m['dcg']:6.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("--out")
    ap.add_argument("--brief", action="store_true", help="only concat / hands / whitened hands / signer rate")
    a = ap.parse_args()
    d = Path(a.emb_dir)
    g, q = load(d / "train.npz"), load(d / "val.npz")
    n = int(max(g["class_ids"].max(), q["class_ids"].max()) + 1)
    out = {}
    G, Q = g["embeddings"], q["embeddings"]
    retr(G, g, Q, q, n, "concat1024 (current readout)", out)
    if a.brief:
        H, HQ = G[:, 256:768], Q[:, 256:768]
        retr(H, g, HQ, q, n, "hands only (lh+rh)", out)
        mu = H[g["usable"]].mean(0); w, V = np.linalg.eigh(np.cov((H[g["usable"]] - mu).T))
        W = V / np.sqrt(w + 1e-3 * w.max())
        retr((H - mu) @ W, g, (HQ - mu) @ W, q, n, "hands only, whitened", out)
        qx = l2(HQ)[q["usable"]]; sg = q["signer_ids"][q["usable"]]
        idx = np.random.default_rng(0).choice(len(qx), min(3000, len(qx)), replace=False)
        s = qx[idx] @ qx.T; s[np.arange(len(idx)), idx] = -2
        out["hands_val_nn_same_signer"] = float((sg[s.argmax(1)] == sg[idx]).mean())
        print("hands-only val NN same signer", round(out["hands_val_nn_same_signer"], 3))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=2))
        return
    retr(g["balanced"], g, q["balanced"], q, n, "balanced256", out)
    for i, p in enumerate(("body", "lh", "rh", "face")):
        retr(G[:, i*256:(i+1)*256], g, Q[:, i*256:(i+1)*256], q, n, f"part {p}", out)
    retr(G[:, 256:768], g, Q[:, 256:768], q, n, "hands only (lh+rh)", out)
    # D5: centring / whitening fitted on train (unlabelled statistics only)
    ok = g["usable"]
    mu = G[ok].mean(0)
    retr(G - mu, g, Q - mu, q, n, "centred (train mean)", out)
    C = np.cov((G[ok] - mu).T)
    w, V = np.linalg.eigh(C)
    for eps_rel in (1e-1, 1e-2, 1e-3):
        W = V / np.sqrt(w + eps_rel * w.max())
        retr((G - mu) @ W, g, (Q - mu) @ W, q, n, f"whitened (eps={eps_rel} x max eig)", out)
    for k in (32, 64, 128):  # drop the top-k principal directions (dominant nuisance)
        P = V[:, ::-1][:, k:]
        retr((G - mu) @ P, g, (Q - mu) @ P, q, n, f"drop top-{k} PCs", out)
    # D4: within-val nearest neighbour (excluding itself): same signer vs same gloss
    qx = l2(Q)[q["usable"]]; sg = q["signer_ids"][q["usable"]]; cl = q["class_ids"][q["usable"]]
    idx = np.random.default_rng(0).choice(len(qx), min(3000, len(qx)), replace=False)
    s = qx[idx] @ qx.T
    s[np.arange(len(idx)), idx] = -2
    nn = s.argmax(1)
    out["val_nn_same_signer"] = float((sg[nn] == sg[idx]).mean())
    out["val_nn_same_gloss"] = float((cl[nn] == cl[idx]).mean())
    chance = float(np.mean([(sg == s_).mean() for s_ in sg[idx]]))
    out["val_nn_same_signer_chance"] = chance
    print(f"val NN: same signer {out['val_nn_same_signer']:.3f} (chance {chance:.3f}), same gloss {out['val_nn_same_gloss']:.4f}")
    ev = w[::-1] / w.sum()
    out["train_pca_explained_top"] = [float(ev[:k].sum()) for k in (1, 5, 10, 32, 100)]
    print("train PCA explained variance top-1/5/10/32/100:", np.round(out["train_pca_explained_top"], 3))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
