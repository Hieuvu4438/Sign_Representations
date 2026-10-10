"""Frozen evaluation shared by SignRep-reproduction baselines and proposals A/B/C (val only; no test split).

Protocols
  loso    : ASL Citizen val, leave-one-signer-out retrieval inside val (EX15/EX16 protocol); clip embedding =
            mean of L2-normalised windows after label-free PCA whitening (k=512, fit on the evaluated windows);
            also the plain mean readout. Paired signer-cluster bootstrap vs --ref.
  aslc300 : ASLC002 protocol of the SignRep reproduction (300 classes; gallery 3,000 train videos, 1,500 val
            queries, 5 signers): (i) retrieval R@1/R@5/MRR, nearest gallery video, whitening fit on gallery windows;
            (ii) TRUE linear probe (fixed mean pooling + multinomial logistic regression; L2 chosen on a signer-
            held-out split of the gallery; refit on the full gallery) -> per-class (macro) Top-1, instance Top-1.
  probes  : (loso set) body-part ridge probes (handshape, body, location/relations, non-manual, movement) and a
            signer probe, as EX15 (reusing method/jepa_targets/jt evaluation code, read-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oj.paths import REPO, RUNTIME  # noqa: E402

JT = REPO / "method" / "jepa_targets"
sys.path.insert(0, str(JT)); sys.path.insert(0, str(REPO / "method" / "p4_signssl"))
from jt.data import PriorStats, load_kp  # noqa: E402
from jt.evaluate import geometry, loso, pool, probes, signer_probe, whiten  # noqa: E402
from jt.priors import DIMS, compute  # noqa: E402
from p4.retrieval import l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, ranks_from_scores  # noqa: E402


def load(name, setname):
    d = RUNTIME / "feats" / name / setname
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    return clips, np.load(d / "windows.npy").astype(np.float32), np.load(d / "offsets.npy"), np.load(d / "src_span.npy")


def window_priors(clips, offs, spans):
    stats = PriorStats(JT / "runtime" / "prior_stats_aslc_train.npz")
    acc = {g: ([], []) for g in DIMS}
    for i, c in enumerate(clips):
        K, Sc = load_kp(c["sample_id"]); pr = compute(K, Sc, list(DIMS))
        for a, b in spans[offs[i]:offs[i + 1]]:
            for g, (v, m) in pr.items():
                vv, mm = v[a:b], m[a:b]; cnt = mm.sum(0)
                acc[g][0].append(stats(g, (vv * mm).sum(0) / np.maximum(cnt, 1)) if len(vv) else np.zeros(DIMS[g], np.float32))
                acc[g][1].append(cnt > 0 if len(vv) else np.zeros(DIMS[g], bool))
    return {g: (np.stack(v), np.stack(m)) for g, (v, m) in acc.items()}


def eval_loso(name, with_probes=True):
    clips, win, offs, spans = load(name, "val")
    gl_ = sorted({c["gloss"] for c in clips}); lab = {g: i for i, g in enumerate(gl_)}
    y = np.array([lab[c["gloss"]] for c in clips]); sg = np.array([c["signer"] for c in clips])
    r = {}
    Ww = whiten(win)
    for mode, src in (("white512", Ww), ("mean", win)):
        F = pool(src, offs, "mean"); rk = loso(F, y, sg, len(gl_)); r[mode] = metrics_from_ranks(rk, y)
        if mode == "white512":
            ranks = rk
        if mode == "mean":
            r["geometry"] = geometry(F); Fm = F
    if with_probes:
        pri = window_priors(clips, offs, spans)
        r["probe_R2"] = {k: v for k, v in probes(win, np.repeat(sg, np.diff(offs)), pri).items() if not k.endswith("alpha")}
        r["signer_probe_acc"] = signer_probe(Fm, sg, np.array([c["gloss"] for c in clips]))
    return r, ranks, sg


def eval_aslc300(name):
    clips, win, offs, _ = load(name, "aslc300")
    role = np.array([c["role"] for c in clips]); y = np.array([c["label"] for c in clips]); sg = np.array([c["signer"] for c in clips])
    gi, qi = np.flatnonzero(role == "gallery"), np.flatnonzero(role == "query")
    # whitening fit on gallery windows only (label-free)
    gw = np.concatenate([win[offs[i]:offs[i + 1]] for i in gi])
    S = gw[::3]; mu = S.mean(0); _, s, Vt = np.linalg.svd(S - mu, full_matrices=False); s = s / np.sqrt(len(S))
    k = min(512, Vt.shape[0])
    tr = lambda w: ((w - mu) @ Vt[:k].T) / (s[:k] + 1e-3 * s[0])
    F = pool(tr(win), offs, "mean"); Fm = pool(win, offs, "mean")
    res = {}
    for tag, E in (("white512", F), ("mean", Fm)):
        sim = E[qi] @ E[gi].T
        ncls = int(y.max()) + 1
        sc = np.full((len(qi), ncls), -9.0, np.float32)
        np.maximum.at(sc, (np.arange(len(qi))[:, None].repeat(len(gi), 1), np.broadcast_to(y[gi], sim.shape)), sim)
        rk = ranks_from_scores(sc, y[qi]); res[f"retrieval_{tag}"] = metrics_from_ranks(rk, y[qi])
        if tag == "white512":
            ranks = rk
    # true linear probe on fixed mean pooled features (standardised with gallery statistics)
    X = Fm; mu_x, sd_x = X[gi].mean(0), X[gi].std(0) + 1e-6; Xs = (X - mu_x) / sd_x
    sig_g = np.unique(sg[gi]); rng = np.random.default_rng(0); hold = rng.choice(sig_g, max(1, len(sig_g) // 5), replace=False)
    a, b = gi[~np.isin(sg[gi], hold)], gi[np.isin(sg[gi], hold)]
    best, scores = None, {}
    for C in (0.01, 0.1, 1.0, 10.0):
        if len(b) == 0:
            break
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xs[a], y[a])
        scores[C] = float((clf.predict(Xs[b]) == y[b]).mean())
        best = C if best is None or scores[C] > scores[best] else best
    clf = LogisticRegression(C=best or 1.0, max_iter=3000).fit(Xs[gi], y[gi])
    pred = clf.predict(Xs[qi]); yq = y[qi]
    res["linear_probe"] = {"C": best, "val_scores": scores, "instance_top1": float((pred == yq).mean()),
                           "macro_top1": float(np.mean([(pred[yq == c] == c).mean() for c in np.unique(yq)]))}
    res["_probe_correct"] = (pred == yq).astype(np.int8).tolist(); res["_probe_y"] = yq.tolist()
    return res, ranks, sg[qi]


EVAL_DIR = RUNTIME / "eval"


def evaluate_model(name, protocol="both", with_probes=True, force=False):
    """per-model results cached in runtime/eval/models/<name>.json (+ ranks / clusters .npz)."""
    EVAL_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
    js, nz = EVAL_DIR / "models" / f"{name}.json", EVAL_DIR / "models" / f"{name}.npz"
    if js.exists() and nz.exists() and not force:
        return json.load(open(js))
    res, arrays = {}, {}
    if protocol in ("loso", "both") and (RUNTIME / "feats" / name / "val").exists():
        r, rk, sg = eval_loso(name, with_probes); res["loso"] = r; arrays["loso_ranks"] = rk; arrays["loso_sg"] = sg
    if protocol in ("aslc300", "both") and (RUNTIME / "feats" / name / "aslc300").exists():
        r, rk, sg = eval_aslc300(name); arrays["a300_ranks"] = rk; arrays["a300_sg"] = sg
        arrays["a300_probe_correct"] = np.array(r.pop("_probe_correct")); arrays["a300_y"] = np.array(r.pop("_probe_y"))
        res["aslc300"] = r
    json.dump(res, open(js, "w"), indent=1, default=str); np.savez(nz, **arrays)
    return res


def macro_cluster_bootstrap(ca, cb, y, cl, n_boot=10000, seed=0):
    """paired signer-cluster bootstrap of the class-macro accuracy difference (b - a), recomputed per resample
    (vectorised: per-(cluster, class) correct counts; a resample = cluster multiplicities)."""
    rng = np.random.default_rng(seed)
    ucl, ci = np.unique(cl, return_inverse=True); uk, ki = np.unique(y, return_inverse=True)
    N = np.zeros((len(ucl), len(uk))); A = np.zeros_like(N); Bm = np.zeros_like(N)
    np.add.at(N, (ci, ki), 1); np.add.at(A, (ci, ki), ca); np.add.at(Bm, (ci, ki), cb)
    def macro(w, S):
        n = w @ N; ok = n > 0
        return ((w @ S)[ok] / n[ok]).mean()
    one = np.ones(len(ucl)); base = macro(one, Bm) - macro(one, A)
    W = rng.multinomial(len(ucl), np.ones(len(ucl)) / len(ucl), size=n_boot).astype(float)
    diffs = np.array([macro(w, Bm) - macro(w, A) for w in W])
    per = {str(c): float((cb[ci == i] - ca[ci == i]).mean()) for i, c in enumerate(ucl)}
    return {"diff": float(base), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "per_signer_instance_diff": per, "pos": f"{sum(v > 0 for v in per.values())}/{len(per)}"}


def paired(name, ref):
    a, b = np.load(EVAL_DIR / "models" / f"{ref}.npz"), np.load(EVAL_DIR / "models" / f"{name}.npz")
    out = {}
    if "loso_ranks" in a and "loso_ranks" in b:
        bs = paired_cluster_bootstrap(per_query(a["loso_ranks"])["R@1"], per_query(b["loso_ranks"])["R@1"], a["loso_sg"], n_boot=10000)
        out["loso_R@1_white512"] = {"diff": bs["diff"], "ci95": bs["ci95"], "pos": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
    if "a300_probe_correct" in a and "a300_probe_correct" in b:
        out["aslc300_linear_probe_macro"] = macro_cluster_bootstrap(a["a300_probe_correct"].astype(float), b["a300_probe_correct"].astype(float), a["a300_y"], a["a300_sg"])
        bs = paired_cluster_bootstrap(per_query(a["a300_ranks"])["R@1"], per_query(b["a300_ranks"])["R@1"], a["a300_sg"], n_boot=10000)
        out["aslc300_retrieval_R@1"] = {"diff": bs["diff"], "ci95": bs["ci95"], "pos": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="comma list of runtime/feats/<name>")
    ap.add_argument("--refs", default="", help="comma list of reference names for paired statistics")
    ap.add_argument("--protocol", default="both", choices=["loso", "aslc300", "both"])
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [n for n in a.names.split(",") if n]
    res = {"models": {}, "paired": {}}
    for name in names:
        res["models"][name] = evaluate_model(name, a.protocol, not a.no_probes, a.force)
        print(name, json.dumps({p: {k: (v["R@1"] if isinstance(v, dict) and "R@1" in v else v) for k, v in d.items() if k != "probe_R2"} for p, d in res["models"][name].items()}, default=str)[:700], flush=True)
    for ref in [r for r in a.refs.split(",") if r]:
        for name in names:
            if name != ref:
                res["paired"][f"{name}-{ref}"] = paired(name, ref)
    json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
