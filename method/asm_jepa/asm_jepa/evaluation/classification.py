"""Classifier evaluation (§13.4): Top-1/5 instance, class-macro Top-1, DCG/MRR from class ranks, per signer.

These are *classifier rankings*, not nearest-neighbour frozen retrieval.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from asm_jepa.evaluation.ranking import metrics_from_ranks, quality_bins, ranks_from_scores


def classification_metrics(logits, y_true, signers=None, usable=None, quality=None) -> dict:
    logits = np.asarray(logits, np.float32)
    y = np.asarray(y_true, np.int64)
    if usable is not None:
        usable = np.asarray(usable, bool)
        logits = np.where(usable[:, None], logits, 0.0)  # unusable queries -> uniform logits, class-id tie order
    ranks = ranks_from_scores(logits, y)
    rm = metrics_from_ranks(ranks, ks=(1, 5, 10))
    out = {"top1_instance": rm["r1"], "top5_instance": rm["r5"], "top10_instance": rm["r10"],
           "dcg": rm["dcg"], "mrr": rm["mrr"], "n": rm["n"]}
    per_class = defaultdict(list)
    for r, c in zip(ranks, y):
        per_class[int(c)].append(r <= 1)
    out["top1_class_macro"] = float(np.mean([np.mean(v) for v in per_class.values()]))
    out["n_classes_with_queries"] = len(per_class)
    for k in ("top1_instance", "top5_instance", "top1_class_macro", "dcg", "mrr"):
        out[k + "_pct"] = 100.0 * out[k]
    if signers is not None:
        ps = defaultdict(list)
        for r, s in zip(ranks, signers):
            ps[str(s)].append(r)
        out["per_signer"] = {s: {"n": len(v), "top1": float(np.mean(np.array(v) <= 1)),
                                 "top5": float(np.mean(np.array(v) <= 5))} for s, v in sorted(ps.items())}
    if quality is not None:
        out["quality_bins"] = quality_bins(ranks, quality)
    if usable is not None:
        out["coverage"] = float(usable.mean())
        out["failure_count"] = int((~usable).sum())
        out["usable_only_diagnostic"] = metrics_from_ranks(ranks[usable], ks=(1, 5, 10))
    return out, ranks
