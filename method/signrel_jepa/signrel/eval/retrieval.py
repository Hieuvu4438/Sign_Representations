"""Exact frozen retrieval with per-query self/source/signer exclusions and coverage."""
import numpy as np


def normalize(z):
    z = np.asarray(z,dtype=np.float32)
    if z.ndim!=2 or not np.isfinite(z).all():
        raise ValueError("Features must be finite [N,D]")
    norms = np.linalg.norm(z,axis=1,keepdims=True)
    if (norms<=1e-12).any():
        raise ValueError("Zero embedding")
    return z/norms


def retrieval(query,gallery,q_rows,g_rows,ks=(1,5),exclude_same_signer=False,class_max=True,chunk_size=256):
    q,g = normalize(query),normalize(gallery)
    if len(q)!=len(q_rows) or len(g)!=len(g_rows):
        raise ValueError("Feature/metadata row mismatch")
    labels = np.array([r["gloss"] for r in g_rows],dtype=str)
    classes = sorted(set(labels))
    class_ids = {c:np.flatnonzero(labels==c) for c in classes}
    hits = {k:[] for k in ks}
    ap,dcg,per_class = [],[],{}
    skipped = []
    for begin in range(0,len(q),chunk_size):
        scores = q[begin:begin+chunk_size]@g.T
        for offset,score in enumerate(scores):
            i = begin+offset
            row = q_rows[i]
            eligible = np.array([r["sample_id"]!=row["sample_id"] and
                      not (r.get("source_video_id") and row.get("source_video_id")==r["source_video_id"]) and
                      not (exclude_same_signer and r.get("signer_id") is not None and r.get("signer_id")==row.get("signer_id"))
                      for r in g_rows])
            relevant = (labels==str(row["gloss"]))&eligible
            if not relevant.any():
                skipped.append(row["sample_id"])
                continue
            score[~eligible] = -np.inf
            order = np.argsort(-score,kind="stable")
            order = order[eligible[order]]
            rel = relevant[order]
            ranks = np.flatnonzero(rel)+1
            # Instance AP over all eligible gallery videos, separately from class-max Recall.
            ap.append(float(np.mean(np.arange(1,len(ranks)+1)/ranks)))
            if class_max:
                cs = np.array([score[class_ids[c]].max() for c in classes])
                co = np.argsort(-cs,kind="stable")
                class_rank = next(j+1 for j in range(len(co)) if classes[co[j]]==str(row["gloss"]))
                rank = class_rank
            else:
                rank = int(ranks[0])
            for k in ks:
                hits[k].append(float(rank<=k))
            dcg.append(float(1/np.log2(rank+1)))
            per_class.setdefault(str(row["gloss"]),[]).append(float(rank<=1))
    if not ap:
        raise ValueError("No query has an eligible relevant gallery item")
    return {**{f"Recall@{k}":float(np.mean(v)) for k,v in hits.items()},
            "mAP_instance":float(np.mean(ap)),"DCG_class" if class_max else "DCG_instance":float(np.mean(dcg)),
            "class_macro_Recall@1":float(np.mean([np.mean(v) for v in per_class.values()])),
            "queries_total":len(q),"queries_evaluated":len(ap),"queries_without_positive":len(skipped),
            "skipped_sample_ids":skipped,"gallery_size":len(g),"gallery_classes":len(classes),
            "ranking":"class_max_cosine" if class_max else "instance_cosine",
            "ap_protocol":"all eligible gallery instances; no class-max aggregation",
            "same_signer_excluded":exclude_same_signer}
