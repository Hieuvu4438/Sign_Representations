"""Retrieval + ASL-LEX minimal-pair diagnostics for one gallery and several query dirs
(e.g. unperturbed and perturbed val queries). Evaluation-only use of ASL-LEX."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.phonology import error_attribute_sharing, gloss_codes, load_asllex, minimal_pair_accuracy, minimal_pairs  # noqa
from p4.retrieval import activity_weights, class_max_scores, l2n, metrics_from_ranks, pool_clips, ranks_from_scores  # noqa


def pooled(d, key, pool):
    d = Path(d)
    clips = [json.loads(l) for l in open(d / "clips.jsonl")]
    off = np.load(d / "offsets.npy")
    win = np.load(d / f"windows_{key}.npy")
    w = activity_weights(np.load(d / "windows_active.npy")) if pool == "activity" else None
    p, _ = pool_clips(win, off, w)
    return clips, l2n(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--queries", required=True, help="comma list name=dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="features")
    ap.add_argument("--pools", default="activity,mean")
    a = ap.parse_args()
    lex = load_asllex()
    res = {}
    for pool in a.pools.split(","):
        gc, gp = pooled(a.gallery, a.key, pool)
        glosses = sorted({c["gloss"] for c in gc})
        lab = {g: i for i, g in enumerate(glosses)}
        gl = np.array([lab[c["gloss"]] for c in gc])
        pairs, sigs = minimal_pairs(glosses, gloss_codes(gc), lex)
        res.setdefault("n_minimal_pairs", {k: len(v) for k, v in pairs.items()})
        for spec in a.queries.split(","):
            name, qd = spec.split("=")
            qc, qp = pooled(qd, a.key, pool)
            ql = np.array([lab[c["gloss"]] for c in qc])
            s = class_max_scores(qp, gp, gl, len(glosses))
            r = ranks_from_scores(s, ql)
            out = {"retrieval": metrics_from_ranks(r, ql),
                   "minimal_pairs": minimal_pair_accuracy(s, ql, pairs),
                   "error_sharing": error_attribute_sharing(s, ql, sigs)}
            res[f"{name}|{a.key}|{pool}"] = out
            Path(a.out).mkdir(parents=True, exist_ok=True)
            np.save(Path(a.out) / f"ranks_{name}_{a.key}_{pool}.npy", r)
            print(name, pool, json.dumps(out), flush=True)
    json.dump(res, open(Path(a.out) / "phonology_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
