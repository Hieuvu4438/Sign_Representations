"""Collect run outputs into results.csv + final_report.md (§18). Missing metrics stay empty (never 0);
literature numbers are listed separately and never in the same rows as local runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from asm_jepa.cli.common import run
from asm_jepa.utils import atomic_write_bytes, now, read_json, resolve

COLUMNS = ["run_id", "seed", "method_version", "pretrain_source", "pretrain_labels", "split_version", "pose_extractor",
           "pose_model_hash", "manifest_hash", "encoder_variant", "feature_key", "readout", "selection_policy",
           "test_exposure", "checkpoint_step", "params_m", "training_hours", "peak_vram_gb", "frozen_r1", "frozen_r5",
           "frozen_r10", "frozen_mrr", "frozen_dcg", "cls_top1_instance", "cls_top5_instance", "cls_top1_class",
           "coverage", "status"]

LITERATURE = [
    ("ASL-Citizen paper I3D (RGB, supervised)", "Top-1 63.10%", "[S1]"),
    ("ASL-Citizen paper ST-GCN (27-joint pose, supervised)", "Top-1 59.52%", "[S1]"),
    ("MS-MAE (external YT-ASL pretraining)", "Top-1 75.72%", "[S11]"),
    ("SignRep (external RGB pretraining, recognition table)", "Top-1 81.37%", "[S12]"),
]


def _load(p: Path):
    return read_json(p) if p.is_file() else None


def method_version(cfg):
    if not cfg:
        return None
    f, l = cfg.get("features", {}), cfg.get("loss", {})
    if cfg.get("train", {}).get("objective") == "mae":
        return "R1_coordinate_mae"
    if f.get("geometry_paths") or f.get("timing_paths"):
        return "ASM-v2" + ("+global" if l.get("lambda_global", 0) > 0 else "")
    if f.get("relations") and l.get("lambda_anchor", 0) > 0:
        return "ASM-v1"
    if l.get("lambda_anchor", 0) > 0:
        return "JEPA+anchor"
    return "JEPA-base"


def collect(run_dir: Path) -> list[dict]:
    rows = []
    prov = _load(run_dir / "provenance.json") or {}
    cfg_p = run_dir / "config_resolved.yaml"
    cfg = yaml.safe_load(cfg_p.read_text()) if cfg_p.is_file() else None
    metrics = run_dir / "metrics.jsonl"
    peak = None
    if metrics.is_file():
        vals = [json.loads(l).get("peak_vram_gb") for l in metrics.read_text().splitlines() if l.strip()]
        vals = [v for v in vals if v is not None]
        peak = max(vals) if vals else None
    base = {c: None for c in COLUMNS}
    base.update(run_id=run_dir.name, seed=(cfg or {}).get("seed"), method_version=method_version(cfg),
                pretrain_source="ASL-Citizen official train (in-domain SSL)" if cfg else None,
                pretrain_labels="none" if cfg else None, split_version="ASL-Citizen v1.0 official",
                manifest_hash=prov.get("manifest_sha256"), peak_vram_gb=peak,
                params_m=(prov.get("params_encoder") / 1e6) if prov.get("params_encoder") else None)
    if prov.get("synthetic"):
        base["pretrain_source"] = "SYNTHETIC (code check only)"
    for split in ("val", "test"):
        r = _load(run_dir / "reports" / f"frozen_{split}.json")
        if r:
            m = r["metrics"]
            rows.append({**base, "encoder_variant": r["encoder"], "feature_key": r["feature_key"],
                         "readout": f"ssl_frozen_retrieval_{split}", "selection_policy": "val/frozen_gloss_max/r1",
                         "test_exposure": split == "test", "checkpoint_step": r.get("checkpoint_step"),
                         "frozen_r1": m["r1"], "frozen_r5": m["r5"], "frozen_r10": m["r10"], "frozen_mrr": m["mrr"],
                         "frozen_dcg": m["dcg"], "coverage": r.get("coverage"), "status": "measured"})
    best = _load(run_dir / "reports" / "best_val.json")
    if best and "r1" in best and not any(r["readout"] == "ssl_frozen_retrieval_val" for r in rows):
        m = best["metrics"]["full"]
        rows.append({**base, "encoder_variant": (cfg or {}).get("eval", {}).get("encoder"),
                     "feature_key": "state_part_concat1024_l2", "readout": "ssl_frozen_retrieval_val (in-training)",
                     "selection_policy": "val/frozen_gloss_max/r1", "test_exposure": False,
                     "checkpoint_step": best["step"], "frozen_r1": m["r1"], "frozen_r5": m["r5"], "frozen_r10": m["r10"],
                     "frozen_mrr": m["mrr"], "frozen_dcg": m["dcg"], "coverage": best["metrics"].get("coverage"),
                     "status": "measured"})
    if best and "top1_instance" in best:
        rows.append({**base, "readout": "finetune_islr_val", "selection_policy": "val/cls/top1_instance",
                     "test_exposure": False, "checkpoint_step": best.get("step"),
                     "cls_top1_instance": best["top1_instance"], "cls_top5_instance": best["top5_instance"],
                     "cls_top1_class": best.get("top1_class_macro"), "coverage": best.get("coverage"),
                     "status": "measured"})
    lin = _load(run_dir / "reports" / "val.json")
    if lin and lin.get("mode") == "linear_probe":
        b = lin["best_val"]
        rows.append({**base, "readout": "linear_probe_val", "encoder_variant": lin.get("encoder"),
                     "selection_policy": "val/cls/top1_instance", "test_exposure": False,
                     "cls_top1_instance": b["top1_instance"], "cls_top5_instance": b["top5_instance"],
                     "cls_top1_class": b.get("top1_class_macro"), "coverage": b.get("coverage"), "status": "measured"})
    test = _load(run_dir / "reports" / "test.json")
    if test:
        m = test["metrics"]
        rows.append({**base, "readout": f"{test['kind']}_test", "selection_policy": "val/cls/top1_instance",
                     "test_exposure": True, "cls_top1_instance": m["top1_instance"],
                     "cls_top5_instance": m["top5_instance"], "cls_top1_class": m["top1_class_macro"],
                     "coverage": m.get("coverage"), "status": "measured"})
    if not rows:
        rows.append({**base, "status": "not_run_or_no_reports"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--output", required=True, help="final_report.md path; results.csv is written next to it")
    args = ap.parse_args()
    out = resolve(args.output)
    rows = [r for d in args.runs for r in collect(resolve(d))]
    csv_path = out.with_name("results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    def fmt(v):
        return "—" if v is None else (f"{100 * v:.2f}" if isinstance(v, float) and v <= 1.0 else str(v))

    lines = [f"# ASM-JEPA results ({now()})", "",
             "Setting: **in-domain ASL-Citizen SSL with signer-independent evaluation** (train gallery seen by SSL).",
             "Model selection uses val labels (`selection_uses_val_labels: true`). Empty cells = not run.", "",
             "| run | version | readout | step | R@1 | R@5 | R@10 | MRR | DCG | cls top1 | cls top5 | coverage | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k in ("run_id", "method_version", "readout", "checkpoint_step",
                                                             "frozen_r1", "frozen_r5", "frozen_r10", "frozen_mrr",
                                                             "frozen_dcg", "cls_top1_instance", "cls_top5_instance",
                                                             "coverage", "status")) + " |")
    lines += ["", "## Literature reference points (different settings; not reproduced here)", ""]
    lines += [f"- {a}: {b} {c}" for a, b, c in LITERATURE]
    atomic_write_bytes(out, ("\n".join(lines) + "\n").encode())
    print(f"wrote {out} and {csv_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    run(main)
