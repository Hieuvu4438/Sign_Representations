"""EX14-v1: evaluate signer-robust label-free arms (H011 / H012b) on a test split vs H8, M2D2, T4 and locked R0–R2.
Refuses to run on test until protocols/EX14-v1.md contains a 'LOCKED' status line. Appends to TEST_ACCESS.log.
  --arms name=path/last.pt[,name=path...]   --h8 ranks file of the H008 model (EX12/EX12b)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_order import load, pad, seqs  # noqa: E402
from ex12_eval import REF  # noqa: E402
from explore_moments import readouts, tdiff  # noqa: E402
from explore_teachers import q2g  # noqa: E402
from h004_train import Aggregator, embed, load_seqs  # noqa: E402
from p4.retrieval import class_max_scores, l2n, metrics_from_ranks, paired_cluster_bootstrap, per_query, ranks_from_scores  # noqa: E402
from paper_stats import signflip_p  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "runtime"
PROTO = Path(__file__).resolve().parents[3] / "docs/proposal4/sign-ssl-research-agent/research/protocols/EX14-v1.md"


def t4_ranks(gs, qs, GW, QW, gy, qy, ncls):
    base = class_max_scores(QW, GW, gy, ncls, device="cuda")
    P = {}
    for k, f in (("T0", lambda w: w), ("T1", tdiff)):
        X, M = pad([(f(w), None) for w, _ in gs], 48); P[k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
        X, M = pad([(f(w), None) for w, _ in qs], 48); P["q" + k] = (torch.from_numpy(X.astype(np.float16)), torch.from_numpy(M))
    G = {k: (P[k][0].cuda(), P[k][1].cuda()) for k in ("T0", "T1")}
    out = []
    for i in range(len(qs)):
        top = np.argsort(-base[i])[:20]; items = torch.from_numpy(np.where(np.isin(gy, top))[0]).cuda()
        v = 0
        for k in ("T0", "T1"):
            qx, qm = P["q" + k]
            v = v + q2g(qx[i].cuda(), qm[i].cuda(), G[k][0][items], G[k][1][items])
        v = v.float().cpu().numpy(); it = items.cpu().numpy()
        new = np.full(ncls, -1e9, np.float32); np.maximum.at(new, gy[it], v)
        out.append(ranks_from_scores(np.where(new > -1e8, 10 + new, base[i])[None], [qy[i]])[0])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(REF))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--h8", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    a = ap.parse_args()
    if a.split == "test":
        assert any(l.startswith("LOCKED") for l in open(PROTO)), "EX14-v1 is not locked: no test access"
    gname, qname = f"{a.dataset}_train_s2", f"{a.dataset}_{a.split}_s2"
    gc, GX, GA, GM = load_seqs(gname); qc, QX, QA, QM = load_seqs(qname)
    assert all(c["split"] == a.split for c in qc)
    gl_ = sorted({c["gloss"] for c in gc}); lab = {g: i for i, g in enumerate(gl_)}
    gy = np.array([lab[c["gloss"]] for c in gc]); qy = np.array([lab[c["gloss"]] for c in qc])
    if a.dataset == "aslc":
        cl = np.array([c["signer"] for c in qc])
    elif a.dataset == "wlasl":
        sig = {i["video_id"]: i["signer_id"] for e in json.load(open(RT / "external/wlasl/WLASL_v0.3.json")) for i in e["instances"]}
        cl = np.array([sig.get(c["sample_id"], -1) for c in qc])
    else:
        cl = qy
    pre = RT / "eval" / f"EX14{a.tag}_{a.dataset}_{a.split}"
    ranks = {"H8": np.load(a.h8)}
    if a.split == "test":
        ranks.update({k: np.load(RT / "eval" / f) for k, f in zip(("R0", "R1", "R2"), REF[a.dataset])})
    # training-free baselines (cached per dataset/split)
    _, goff, gw, ga = load(RT / "feats" / gname); _, qoff, qw, qa = load(RT / "feats" / qname)
    gs, qs = seqs(goff, gw, ga), seqs(qoff, qw, qa)
    for k in ("M2D2", "T4"):
        f = RT / "eval" / f"EX14_{a.dataset}_{a.split}_ranks_{k}.npy"
        if not f.exists():
            if k == "M2D2":
                r = ranks_from_scores(class_max_scores(readouts(qs)["M2D2"], readouts(gs)["M2D2"], gy, len(gl_)), qy)
            else:
                wm = lambda sq: l2n(np.stack([((g[:, None] + 1e-6) * w).sum(0) for w, g in sq]))
                r = t4_ranks(gs, qs, wm(gs), wm(qs), gy, qy, len(gl_))
            np.save(f, r)
        ranks[k] = np.load(f)
    for spec in a.arms.split(","):
        name, path = spec.split("=")
        model = Aggregator().cuda(); model.load_state_dict(torch.load(path, map_location="cuda")); model.eval()
        with torch.no_grad():
            G, Q = embed(model, GX, GA, GM, "cuda"), embed(model, QX, QA, QM, "cuda")
        ranks[name] = ranks_from_scores(class_max_scores(Q, G, gy, len(gl_)), qy)
        np.save(f"{pre}_ranks_{name}.npy", ranks[name])
    assert all(len(v) == len(qy) for v in ranks.values())
    out = {"dataset": a.dataset, "split": a.split, "R@1": {k: metrics_from_ranks(v)["R@1"] for k, v in ranks.items()}, "comparisons": {}}
    arms = [s.split("=")[0] for s in a.arms.split(",")]
    for arm in arms:
        for ref in [k for k in ranks if k not in arms]:
            bs = paired_cluster_bootstrap(per_query(ranks[ref])["R@1"], per_query(ranks[arm])["R@1"], cl, n_boot=10000)
            c = {"diff": bs["diff"], "ci95": bs["ci95"], "clusters_positive": f"{bs['n_clusters_positive']}/{bs['n_clusters']}"}
            if a.dataset == "aslc":
                c["signflip_p_one_sided"] = signflip_p(bs["per_cluster"])
            out["comparisons"][f"{arm}-{ref}"] = c
    json.dump(out, open(f"{pre}.json", "w"), indent=1, default=str)
    if a.split == "test":
        with open(RT / "logs" / "TEST_ACCESS.log", "a") as fh:
            fh.write(f"EX14-v1 {a.dataset} test evaluation tag={a.tag} arms={arms} {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n")
    print(json.dumps(out["R@1"]), json.dumps({k: (round(v["diff"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in out["comparisons"].items()}))


if __name__ == "__main__":
    main()
