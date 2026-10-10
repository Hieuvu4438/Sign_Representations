"""Held-out decoder reconstruction diagnostics and multi-arm paired comparisons.

Reconstruction numbers describe the training objective only; they are never
evidence of a representation benefit (spec 12.7).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TrainClipDataset
from ..data.manifest import read_jsonl, record_valid_starts
from ..data.priors import HEADS
from ..data.scaler import load_scaler, scaler_start
from ..data.video import worker_init
from ..losses.stats import aggregate_epoch
from .statistics import load_per_query, paired_bootstrap

REPO_ROOT = Path(__file__).resolve().parents[3]


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def diagnose_reconstruction(cfg, ctx, checkpoint: Path, split: str, num_clips: int) -> None:
    from ..engine.train import LossComputer, to_device
    from ..model import TemporalDecoder, TrainingModel, attention_kernel, build_empty_encoder

    device = torch.device(cfg["train"]["device"])
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    enc = build_empty_encoder(cfg["paths"]["vendor_root"])
    enc.backbone.load_state_dict(ck["encoder_state"]["backbone"], strict=True)
    enc.fc.load_state_dict(ck["encoder_state"]["fc"], strict=True)
    m = cfg["model"]
    dec = TemporalDecoder(cfg["paths"]["vendor_root"], m["heads"], m["decoder_hidden_dim"], m["decoder_latent_dim"], m["total_frames"])
    dec.load_state_dict(ck["decoder_state"], strict=True)
    model = TrainingModel(enc, dec).to(device).eval()
    scaler = load_scaler(_abs(cfg["paths"]["scaler"]))
    if scaler["_sha256"] != ck["scaler_hash"]:
        raise ValueError("scaler differs from the one used in training")
    T = cfg["data"]["clip_length"]
    recs = [r for r in read_jsonl(_abs(cfg["paths"]["manifest"])) if r["eligible"] and r["official_split"] == split]
    recs = sorted(recs, key=lambda r: r["sample_id"])[:: max(1, len(recs) // num_clips)][:num_clips]
    sched = np.array([(i, scaler_start(r["sample_id"], 4242, record_valid_starts(r, T))) for i, r in enumerate(recs)])
    ds = TrainClipDataset(recs, sched, cfg["targets"], T, cfg["data"]["max_relative_dt_deviation"])
    loader = DataLoader(ds, batch_size=32, num_workers=8, worker_init_fn=worker_init)
    lc = LossComputer(cfg, scaler, device)
    logs = []
    with torch.no_grad(), attention_kernel(m["attention_kernel"]):
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg["train"]["amp"]):
                out = model(batch["rgb"], mask_ratio=0.0)
            sc = lc.scaled(out["priors"], batch)
            rec = {}
            rec.update(lc.base(sc)[1])
            for kind in ("spectral_logmag_l1", "velocity_l1", "window_time_l1"):
                rec.update(lc.aux(sc, kind)[1])
            logs.append(rec)
    agg = aggregate_epoch(logs)
    groups = {}
    for k, v in agg.items():
        fam = k.split("/")[0]
        groups.setdefault(fam, []).append(v)
    summary = {f"{fam}_mean_of_terms": float(np.nanmean(v)) for fam, v in groups.items()}
    ctx.write_json("reconstruction.json", {"checkpoint": str(checkpoint), "split": split, "num_clips": len(recs),
                                           "clip_policy": "every k-th sorted eligible video, sha256 start (seed 4242)",
                                           "summary": summary, "terms": agg})
    print(json.dumps(summary, indent=2))


def compare_arms(ctx, arms: dict, baseline: str, file_name: str) -> None:
    """arms: {arm_id: run_dir}. Paired deltas vs `baseline` for RR / R@1 (retrieval) or accuracy (probe)."""
    value_kinds = ("acc",) if file_name.startswith("probe") else ("rr", "r1")
    out = {"baseline": baseline, "file": file_name, "comparisons": {}}
    base = {vk: load_per_query(_abs(arms[baseline]) / file_name, vk) for vk in value_kinds}
    for arm, d in arms.items():
        if arm == baseline:
            continue
        out["comparisons"][arm] = {vk: paired_bootstrap(base[vk], load_per_query(_abs(d) / file_name, vk)) for vk in value_kinds}
    ctx.write_json("comparison.json", out)
    print(json.dumps({a: {vk: {k: v[vk][k] for k in ("delta", "ci95_video", "ci95_signer_cluster", "signers_positive")} for vk in v}
                      for a, v in out["comparisons"].items()}, indent=2))
