"""Frozen class-ranking retrieval (adapted protocol, NOT an official benchmark metric).

Class score = max cosine similarity over that class's gallery instances; candidates
are unique classes; ties broken by ascending canonical class id; RR = 1/rank.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def class_scores(q: torch.Tensor, g: torch.Tensor, g_labels: torch.Tensor, num_classes: int, chunk: int = 1024):
    out = []
    for s in range(0, q.shape[0], chunk):
        sim = q[s : s + chunk] @ g.T  # [c,G]
        cs = torch.full((sim.shape[0], num_classes), -float("inf"), device=sim.device, dtype=sim.dtype)
        cs.scatter_reduce_(1, g_labels.expand(sim.shape[0], -1), sim, reduce="amax", include_self=True)
        out.append(cs)
    return torch.cat(out)


def true_class_rank(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """1-based rank with deterministic ties (descending score, then ascending class id)."""
    ts = scores.gather(1, truth[:, None])
    ids = torch.arange(scores.shape[1], device=scores.device)[None]
    better = (scores > ts) | ((scores == ts) & (ids < truth[:, None]))
    return better.sum(1) + 1


def metrics_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> dict:
    rr = 1.0 / ranks
    m = {"mrr": float(rr.mean()), "num_queries": int(len(ranks))}
    for k in (1, 5, 10):
        m[f"recall@{k}"] = float((ranks <= k).mean())
    classes = np.unique(labels)
    per_c_rr = np.array([rr[labels == c].mean() for c in classes])
    m["macro_mrr"] = float(per_c_rr.mean())
    for k in (1, 5, 10):
        m[f"macro_recall@{k}"] = float(np.mean([(ranks[labels == c] <= k).mean() for c in classes]))
    m["num_classes"] = int(len(classes))
    return m


def load_features(features_dir: Path, split: str, key: str = "pooled_features"):
    z = np.load(features_dir / f"features_{split}.npz", allow_pickle=False)
    ok = z["status"] == "ok"
    return {"sample_id": z["sample_id"][ok], "label": z["label_id"][ok], "signer": z["signer_id"][ok],
            "x": z[key][ok], "excluded": {str(s): int(c) for s, c in zip(*np.unique(z["status"][~ok], return_counts=True))}}


def evaluate_retrieval(gal: dict, qry: dict, num_classes: int, device="cuda") -> dict:
    if len(set(gal["sample_id"]) & set(qry["sample_id"])):
        raise ValueError("query/gallery sample overlap (self-match)")
    if len(set(qry["sample_id"])) != len(qry["sample_id"]):
        raise ValueError("duplicate query ids")
    for d in (gal, qry):
        if not np.isfinite(d["x"]).all():
            raise ValueError("non-finite embeddings")
    absent = set(np.unique(qry["label"])) - set(np.unique(gal["label"]))
    if absent:
        raise ValueError(f"{len(absent)} query classes absent from gallery (protocol requires full coverage)")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.as_tensor(gal["x"], dtype=torch.float32, device=dev)
    q = torch.as_tensor(qry["x"], dtype=torch.float32, device=dev)
    g = g / g.norm(dim=1, keepdim=True)
    q = q / q.norm(dim=1, keepdim=True)
    gl = torch.as_tensor(gal["label"], device=dev)
    ql = torch.as_tensor(qry["label"], device=dev)
    sc = class_scores(q, g, gl, num_classes)
    ranks = true_class_rank(sc, ql).cpu().numpy()
    top1 = sc.argmax(1).cpu().numpy()
    return {"ranks": ranks, "top1": top1}


def run_retrieval(ctx, features_dir: Path, gallery_split: str, query_split: str) -> None:
    res = {}
    for key in ("pooled_features", "pooled_latent"):
        gal = load_features(features_dir, gallery_split, key)
        qry = load_features(features_dir, query_split, key)
        num_classes = int(max(gal["label"].max(), qry["label"].max()) + 1)
        out = evaluate_retrieval(gal, qry, num_classes)
        m = metrics_from_ranks(out["ranks"], qry["label"])
        m.update(gallery_size=int(len(gal["label"])), gallery_excluded=gal["excluded"], query_excluded=qry["excluded"])
        res[key] = m
        with open(ctx.dir / f"per_query_{key}.jsonl", "x") as f:
            for sid, lab, sig, r, t1 in zip(qry["sample_id"], qry["label"], qry["signer"], out["ranks"], out["top1"]):
                f.write(json.dumps({"sample_id": str(sid), "label": int(lab), "signer": str(sig), "rank": int(r), "top1": int(t1)}) + "\n")
    fm = features_dir / "feature_manifest.json"
    res["protocol"] = {"id": "asl_citizen_frozen_class_retrieval_v1", "gallery_split": gallery_split,
                       "query_split": query_split, "support_per_class": "all_available",
                       "class_score": "max_cosine", "ties": "ascending_class_id",
                       "features_manifest_sha256": hashlib.sha256(fm.read_bytes()).hexdigest() if fm.exists() else None}
    ctx.write_json("retrieval_metrics.json", res)
    print(json.dumps({k: {kk: v for kk, v in res[k].items() if kk in ("mrr", "recall@1", "recall@5", "macro_mrr")} for k in ("pooled_features", "pooled_latent")}, indent=2))
