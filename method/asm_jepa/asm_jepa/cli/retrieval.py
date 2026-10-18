"""Frozen dictionary retrieval (gloss-level max-instance, §13.2): train gallery vs val/test queries.

Reports R@1/5/10, MRR, DCG (fractions and *_pct percentages), the SignRep-style keys (DCG, Rec@1, Rec@5),
coverage/failure accounting with full-split denominators, per-signer results and bootstrap CIs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.evaluation.bootstrap import bootstrap_ci
from asm_jepa.evaluation.embeddings import check_compatible, load_embeddings
from asm_jepa.evaluation.ranking import metrics_from_ranks, quality_bins
from asm_jepa.evaluation.retrieval import evaluate_retrieval
from asm_jepa.utils import now, sha256_file, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default="configs/eval/retrieval.yaml")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--protocol", default=None, choices=("gloss_max",))
    ap.add_argument("--feature-key", choices=("state_part_concat1024_l2", "state_balanced_mean256_l2"))
    ap.add_argument("--query-chunk", type=int)
    ap.add_argument("--gallery-chunk", type=int)
    ap.add_argument("--num-classes", type=int, help="default: vocabulary size implied by gallery labels")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True)
    ap.add_argument("--save-ranks", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args, "retrieval")
    if (args.protocol or cfg["protocol"]) != "gloss_max":
        raise ValueError("only the gloss_max protocol is implemented")
    key = args.feature_key or cfg["feature_key"]
    g = load_embeddings(args.gallery, key)
    q = load_embeddings(args.queries, key)
    check_compatible(g, q)
    if g["meta"]["split"] != cfg["gallery_split"]:
        raise ValueError(f"gallery must be the official {cfg['gallery_split']} split, got {g['meta']['split']}")
    if q["meta"]["split"] not in ("val", "test") or q["meta"]["split"] == g["meta"]["split"]:
        raise ValueError(f"queries must be the official val or test split, got {q['meta']['split']}")
    if g["meta"].get("partial_limit") or q["meta"].get("partial_limit"):
        print("WARNING: partial (--limit) embeddings; results are not full-benchmark numbers")
    n_classes = args.num_classes or int(max(g["class_ids"].max(), q["class_ids"].max()) + 1)
    res, ranks, _ = evaluate_retrieval(q["features"], q["class_ids"], g["features"], g["class_ids"], n_classes,
                                       tuple(cfg["ks"]), args.query_chunk or cfg["query_chunk"],
                                       args.gallery_chunk or cfg["gallery_chunk"], args.device, q["usable"],
                                       gallery_usable=g["usable"])
    full = res["full"]
    per_signer = defaultdict(list)
    for r, s in zip(ranks, q["signer_ids"]):
        per_signer[str(s)].append(r)
    report = {"created": now(), "protocol": "gloss_max", "feature_key": key, "encoder": g["meta"]["encoder"],
              "gallery": {"path": str(Path(args.gallery).resolve()), "sha256": sha256_file(args.gallery),
                          "split": g["meta"]["split"], "n": int(len(g["class_ids"]))},
              "queries": {"path": str(Path(args.queries).resolve()), "sha256": sha256_file(args.queries),
                          "split": q["meta"]["split"], "n": int(len(q["class_ids"]))},
              "checkpoint_sha256": g["meta"]["checkpoint_sha256"], "checkpoint_step": g["meta"].get("checkpoint_step"),
              "n_classes_ranked": n_classes, "tie_break": cfg["tie_break"], "denominator": cfg["denominator"],
              "metrics": full, "signrep_style": {"DCG": res["pessimistic_ties"]["dcg"], "Rec@1": res["pessimistic_ties"]["r1"],
                                "Rec@5": res["pessimistic_ties"]["r5"], "tie_rule": "pessimistic (rank = #classes >= true)",
                                "true_score_ties": res["true_score_ties"]},
              "coverage": res.get("coverage"), "failure_count": res.get("failure_count"),
              "usable_only_diagnostic": res.get("usable_only_diagnostic"),
              "gallery_class_counts": res["gallery_class_counts"],
              "per_signer": {s: metrics_from_ranks(v) for s, v in sorted(per_signer.items())},
              "quality_bins": quality_bins(ranks, q["quality"]) if "quality" in q else None,
              "bootstrap_ci_queries": bootstrap_ci(ranks, cfg["bootstrap_samples"], cfg["bootstrap_seed"]),
              "bootstrap_ci_signers": bootstrap_ci(ranks, cfg["bootstrap_samples"], cfg["bootstrap_seed"],
                                                   cluster=q["signer_ids"]),
              "synthetic": bool(g["meta"].get("synthetic")),
              "train_gallery_seen_in_ssl": True,
              "setting": "in-domain ASL-Citizen SSL with signer-independent evaluation"}
    write_json(args.output, report)
    if args.save_ranks:
        np.save(Path(args.output).with_suffix(".ranks.npy"), ranks)
    print(f"{q['meta']['split']} R@1 {full['r1_pct']:.2f} R@5 {full['r5_pct']:.2f} R@10 {full['r10_pct']:.2f} "
          f"MRR {full['mrr_pct']:.2f} DCG {full['dcg_pct']:.2f} coverage {res.get('coverage')}")
    return 0


if __name__ == "__main__":
    run(main)
