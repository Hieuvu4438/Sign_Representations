"""Dictionary retrieval, gloss-level max-instance protocol (§13.2).

Gallery = official train embeddings, queries = official val or test embeddings.
score(query, g) = max cosine(query, gallery videos of gloss g); all classes are ranked (classes absent from the
gallery score -inf). Query/gallery chunking keeps memory bounded; class maxima are combined with a segmented
max (scatter_reduce 'amax'). y_true is only used after scores are final, to read off the rank.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from asm_jepa.evaluation.ranking import metrics_from_ranks, ranks_from_scores


@torch.no_grad()
def gloss_max_scores(query, gallery, gallery_labels, n_classes, query_chunk=256, gallery_chunk=8192, device="cpu"):
    """-> class scores [Nq, C] float32 (numpy)."""
    q_all = torch.as_tensor(np.asarray(query, np.float32))
    g_all = torch.as_tensor(np.asarray(gallery, np.float32))
    gl_all = torch.as_tensor(np.asarray(gallery_labels, np.int64))
    q_all = F.normalize(q_all, dim=-1)  # zero vectors stay zero -> cosine 0 with everything
    g_all = F.normalize(g_all, dim=-1)
    out = np.empty((len(q_all), n_classes), np.float32)
    for qs in range(0, len(q_all), query_chunk):
        q = q_all[qs:qs + query_chunk].to(device)
        scores = torch.full((len(q), n_classes), float("-inf"), device=device)
        for gs in range(0, len(g_all), gallery_chunk):
            g = g_all[gs:gs + gallery_chunk].to(device)
            gl = gl_all[gs:gs + gallery_chunk].to(device)
            sim = q @ g.T  # [q, g]
            scores.scatter_reduce_(1, gl[None, :].expand(len(q), -1), sim, reduce="amax", include_self=True)
        out[qs:qs + len(q)] = scores.cpu().numpy()
    return out


def evaluate_retrieval(query, query_labels, gallery, gallery_labels, n_classes, ks=(1, 5, 10), query_chunk=256,
                       gallery_chunk=8192, device="cpu", query_usable=None, gallery_usable=None):
    """Unusable gallery clips (no valid pose -> zero embedding) are excluded from the per-gloss maximum so they
    cannot contribute a fake cosine of 0; unusable queries stay in the denominator."""
    n_gallery_total = len(gallery_labels)
    if gallery_usable is not None:
        gu = np.asarray(gallery_usable, bool)
        gallery, gallery_labels = np.asarray(gallery)[gu], np.asarray(gallery_labels)[gu]
    scores = gloss_max_scores(query, gallery, gallery_labels, n_classes, query_chunk, gallery_chunk, device)
    ranks = ranks_from_scores(scores, query_labels)
    res = {"full": metrics_from_ranks(ranks, ks)}
    # SignRep §7.2 / repo convention: rank = #classes with score >= true score (pessimistic ties)
    true = scores[np.arange(len(scores)), np.asarray(query_labels)]
    pess = (scores >= true[:, None]).sum(1)
    res["pessimistic_ties"] = metrics_from_ranks(pess, ks)
    res["true_score_ties"] = int((pess != ranks).sum())
    counts = np.bincount(np.asarray(gallery_labels), minlength=n_classes)
    res["gallery_class_counts"] = {"min": int(counts.min()), "max": int(counts.max()),
                                   "classes_absent": int((counts == 0).sum()), "n_gallery": int(len(gallery_labels)),
                                   "n_gallery_excluded_unusable": int(n_gallery_total - len(gallery_labels))}
    if query_usable is not None:
        u = np.asarray(query_usable, bool)
        res["coverage"] = float(u.mean()) if len(u) else 0.0
        res["failure_count"] = int((~u).sum())
        res["usable_only_diagnostic"] = metrics_from_ranks(ranks[u], ks)
    return res, ranks, scores


def dense_reference_ranks(query, query_labels, gallery, gallery_labels, n_classes):
    """O(Nq*Ng) reference implementation used in tests to check the chunked evaluator."""
    q = np.asarray(query, np.float64)
    g = np.asarray(gallery, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    sim = q @ g.T
    scores = np.full((len(q), n_classes), -np.inf)
    for c in range(n_classes):
        m = np.asarray(gallery_labels) == c
        if m.any():
            scores[:, c] = sim[:, m].max(1)
    order = np.lexsort((np.tile(np.arange(n_classes), (len(q), 1)), -scores), axis=1)
    return np.array([int(np.flatnonzero(order[i] == y)[0]) + 1 for i, y in enumerate(query_labels)])
