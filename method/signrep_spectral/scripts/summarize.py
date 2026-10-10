"""Print a markdown table of per-arm dev (or test) metrics from owned run directories.

Usage: python method/signrep_spectral/scripts/summarize.py <tag> [split] [arms...]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "runtime"


def main():
    tag = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    arms = sys.argv[3:] or ["B0", "B1", "M1", "B2", "B3", "B4"]
    rows = []
    for arm in arms:
        ret = R / f"{tag}_{arm}_retrieval_{split}" / "retrieval_metrics.json"
        prb = R / f"{tag}_{arm}_probe_{split}" / "probe_metrics.json"
        trn = R / f"{tag}_{arm}_train" / "train_summary.json"
        if not ret.exists():
            continue
        m = json.load(open(ret))
        f, l = m["pooled_features"], m["pooled_latent"]
        p = json.load(open(prb)) if prb.exists() else None
        t = json.load(open(trn)) if trn.exists() else None
        rows.append([
            arm,
            f"{f['mrr']:.4f}", f"{f['recall@1']:.4f}", f"{f['recall@5']:.4f}", f"{f['macro_mrr']:.4f}",
            f"{l['mrr']:.4f}",
            f"{p[split]['top1']:.4f}" if p else "-", f"{p[split]['macro_top1']:.4f}" if p else "-",
            f"{t['lambda']:.3g}" if t else "-", f"{t['wall_seconds']/60:.1f}" if t else "-",
        ])
    hdr = ["arm", "MRR", "R@1", "R@5", "macro MRR", "MRR (latent)", "probe top1", "probe macro", "lambda", "train min"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
