"""Frozen embedding extraction (§13.1): deterministic full-clip sampling, no augmentation, no masks,
teacher encoder by default. Output NPZ: embeddings [N,1024] (state_part_concat1024_l2), balanced [N,256]
(diagnostic), usable flags, sample/class/signer ids and provenance metadata."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.config import arch_diff, config_hash
from asm_jepa.data.dataset import LabeledDataset, PoseStoreSource, SampleBuilder
from asm_jepa.data.manifest import SPLITS, load_split
from asm_jepa.evaluation.embeddings import extract_embeddings, save_embeddings
from asm_jepa.features.stats import load_stats
from asm_jepa.models.asm_jepa import Backbone
from asm_jepa.training.checkpoint import load_checkpoint
from asm_jepa.utils import now, resolve, sha256_file, sha256_json


def load_backbone(ckpt: dict, cfg: dict, encoder: str) -> Backbone:
    bb = Backbone(cfg)
    if encoder == "teacher":
        if ckpt.get("teacher") is None:
            raise ValueError("checkpoint has no EMA teacher (e.g. MAE control); use --encoder online")
        state = {k[len("backbone."):]: v for k, v in ckpt["teacher"].items() if k.startswith("backbone.")}
    elif encoder == "online":
        state = {k[len("online."):]: v for k, v in ckpt["model"].items() if k.startswith("online.")}
    else:
        raise ValueError(encoder)
    bb.load_state_dict(state, strict=True)
    return bb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--output", required=True)
    ap.add_argument("--encoder", choices=("teacher", "online"), help="default: eval.encoder of the config")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="debug only; output is marked partial")
    args = ap.parse_args()
    cfg = config_from_args(args, "pretrain")
    ck = load_checkpoint(args.checkpoint)
    diffs = arch_diff(ck["config"], cfg)
    if diffs:
        raise ValueError("config and checkpoint architectures differ: " + "; ".join(diffs))
    encoder = args.encoder or cfg["eval"]["encoder"]
    root = resolve(cfg["data"]["processed_root"])
    stats_path = root / cfg["data"]["stats_path"]
    stats = load_stats(stats_path, len(cfg["features"]["interval_widths_sec"]))
    stats_sha = sha256_file(stats_path)
    if ck["provenance"]["stats_sha256"] != stats_sha:
        raise ValueError("feature statistics differ from those used in pretraining")
    rows = load_split(root, args.split)
    if args.limit:
        rows = rows[: args.limit]
    builder = SampleBuilder(cfg, stats)
    ds = LabeledDataset(PoseStoreSource(rows, root), builder, [r["class_id"] for r in rows], "eval")
    bb = load_backbone(ck, cfg, encoder).to(args.device)
    arrays = extract_embeddings(bb, ds, args.batch_size or cfg["eval"]["extract_batch_size"],
                                cfg["data"]["num_workers"] if args.workers is None else args.workers, args.device,
                                cfg["train"]["precision"], progress=True)
    meta = {"created": now(), "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_step": ck["step"],
            "stats_sha256": stats_sha, "schema": cfg["data"]["schema"], "feature_key": "state_part_concat1024_l2",
            "diagnostic_feature_key": "state_balanced_mean256_l2", "encoder": encoder,
            "sampler": f"full_clip_bins_eval_T{cfg['data']['max_frames']}", "split": args.split,
            "arch_hash": sha256_json({k: v for k, v in ck["config"].items() if k in ("data", "features", "model")}),
            "config_sha256": config_hash(cfg), "n": len(rows), "partial_limit": args.limit,
            "usable": int(arrays["usable"].sum()), "unusable": int((~arrays["usable"]).sum()),
            "synthetic": bool(ck["provenance"].get("synthetic")),
            "metadata_sha256": sha256_file(root / f"metadata_{args.split}.jsonl")}
    save_embeddings(args.output, arrays, [r["sample_id"] for r in rows], [r["signer_id"] for r in rows], meta)
    print(f"wrote {args.output}: {len(rows)} clips, {meta['unusable']} without usable pose")
    return 0


if __name__ == "__main__":
    run(main)
