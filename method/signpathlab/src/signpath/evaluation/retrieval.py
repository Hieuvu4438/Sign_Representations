import numpy as np


def retrieval_metrics(queries, gallery, query_labels, gallery_labels, query_sources, gallery_sources, *, ks=(1, 5, 10)):
    q, g = np.array(queries, dtype=np.float64, copy=True), np.array(gallery, dtype=np.float64, copy=True)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1] or not len(q) or not len(g) or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("Retrieval needs nonempty finite compatible embedding matrices")
    if len(query_labels) != len(q) or len(query_sources) != len(q) or len(gallery_labels) != len(g) or len(gallery_sources) != len(g):
        raise ValueError("Retrieval annotation/embedding lengths differ")
    if (np.linalg.norm(q, axis=1) == 0).any() or (np.linalg.norm(g, axis=1) == 0).any():
        raise ValueError("Cosine retrieval cannot normalize zero embeddings")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    scores = q @ g.T
    reciprocal, ranks, excluded = [], [], []
    gl = np.asarray(gallery_labels)
    for i, row in enumerate(scores):
        allowed = np.asarray([source != query_sources[i] for source in gallery_sources])
        ordered = np.argsort(-row, kind="stable")
        ordered = ordered[allowed[ordered]]
        relevant = np.flatnonzero(gl[ordered] == query_labels[i])
        rank = int(relevant[0] + 1) if len(relevant) else None
        ranks.append(rank)
        reciprocal.append(1 / rank if rank else 0.0)
        excluded.append(int((~allowed).sum()))
    recalls = {f"recall_at_{k}": float(np.mean([rank is not None and rank <= k for rank in ranks])) for k in ks}
    return {**recalls, "mrr": float(np.mean(reciprocal)), "query_count": len(q),
            "queries_without_relevant_gallery": sum(rank is None for rank in ranks),
            "same_source_exclusions": sum(excluded), "tie_policy": "stable_gallery_order", "ranks": ranks}
