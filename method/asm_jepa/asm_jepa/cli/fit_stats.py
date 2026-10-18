"""Fit fixed feature statistics on the official TRAIN split only (§7.4).

Computes mean/std of observed body-frame coordinates (114x2), hand-local coordinates, relation dx/dy/norm and
the level-2 log-signature path components of the full config (so base/state/full share one file). Deterministic
eval sampling, no augmentation, no masks. Std floor 1e-3; clipping to +-5 happens at use time.
"""
from __future__ import annotations

import argparse

import torch

from asm_jepa.cli.common import add_config_args, config_from_args, run
from asm_jepa.config import config_hash
from asm_jepa.data.dataset import PoseStoreSource, SampleBuilder, worker_init
from asm_jepa.data.manifest import load_split
from asm_jepa.data.normalization import body_normalize, hand_local
from asm_jepa.features.paths import path_tokens
from asm_jepa.features.relations import relation_raw
from asm_jepa.features.stats import StatsAccumulator, identity_stats, save_stats
from asm_jepa.utils import now, resolve, sha256_file


class RawFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, source, builder):
        self.source, self.builder = source, builder

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        b = self.builder
        clip = self.source[i]
        if clip is None:
            return {"ok": False}
        xy, ob, conf, ts, tv, _ = b._select(clip, None, False)
        norm = body_normalize(xy, ob, tv)
        if norm["unusable_pose"]:
            return {"ok": False}
        xb = norm["xy"]
        local, lv = hand_local(xb, ob)
        rel, op = relation_raw(xb, ob, tv)
        f = b.cfg["features"]
        pt = path_tokens(xb, ob, conf, tv, ts, b.P, f["interval_widths_sec"], f["centers_per_clip"],
                         f["path_min_points"], b.stats)
        return {"ok": True, "xb": xb, "ob": ob & tv[:, None], "local": local, "lv": lv & tv[:, None], "rel": rel,
                "op": op, "geom": pt["raw_geom"], "time": pt["raw_timing"], "pvalid": pt["valid"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default="configs/pretrain/full.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", help="default: <processed_root>/<data.stats_path>")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="debug only: first N train clips (output marked as partial)")
    args = ap.parse_args()
    if args.split != "train":
        raise ValueError("statistics are fitted on the official train split only")
    cfg = config_from_args(args, "pretrain")
    root = resolve(cfg["data"]["processed_root"])
    out = resolve(args.output) if args.output else root / cfg["data"]["stats_path"]
    n_scales = len(cfg["features"]["interval_widths_sec"])
    rows = load_split(root, "train")
    if args.limit:
        rows = rows[: args.limit]
    builder = SampleBuilder(cfg, identity_stats(n_scales))
    ds = RawFeatureDataset(PoseStoreSource(rows, root), builder)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=args.workers, worker_init_fn=worker_init,
                                         prefetch_factor=4 if args.workers else None)
    acc = StatsAccumulator(n_scales)
    used, skipped = 0, 0
    for i, s in enumerate(loader):
        if not s["ok"]:
            skipped += 1
            continue
        s = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in s.items()}
        acc.add("body", s["xb"], s["ob"][..., None])
        acc.add("local", s["local"], s["lv"][..., None])
        acc.add("rel", s["rel"], s["op"][..., None])
        acc.add("geom", s["geom"].transpose(1, 0, 2, 3), s["pvalid"].transpose(1, 0, 2)[..., None])
        acc.add("time", s["time"].transpose(1, 0, 2, 3), s["pvalid"].transpose(1, 0, 2)[..., None])
        used += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)
    stats = acc.finalize()
    info = {"created": now(), "split": "train", "clips_used": used, "clips_skipped_unusable_or_missing": skipped,
            "partial_limit": args.limit, "metadata_train_sha256": sha256_file(root / "metadata_train.jsonl"),
            "config_sha256": config_hash(cfg), "schema": cfg["data"]["schema"], "std_floor": 1e-3,
            "interval_widths_sec": cfg["features"]["interval_widths_sec"], "sampler": "full_clip_bins_eval",
            "counts": {k: int(stats[f"{k}_count"].sum()) for k in ("body", "local", "rel", "geom", "time")}}
    save_stats(out, stats, info)
    print(f"wrote {out}: {info['counts']}, used {used}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    run(main)
