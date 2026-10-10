"""Entry point: python -m method.signrep_spectral.cli <subcommand> ..."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from .config import ConfigError, resolved
from .engine.run import REPO_ROOT, RunContext


def _abs(p):
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _require(cfg, *keys):
    missing = [k for k in keys if cfg["paths"].get(k) in (None, "")]
    if missing:
        raise ConfigError(f"missing required paths fields: {missing} (set them in --paths-config)")


# ---------------------------------------------------------------- preflight
def cmd_preflight(args, cfg, ctx):
    import numpy as np
    import torch

    from .model import encoder_from_vendor_model
    from .vendor import load_vendor_model

    rep = {
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    }
    try:
        import cv2, av  # noqa: E401

        rep["runtime"].update(opencv=cv2.__version__, pyav=av.__version__)
    except Exception as e:
        rep["runtime"]["video_backend_error"] = repr(e)
    model, audit = load_vendor_model(cfg["paths"]["vendor_root"], cfg["paths"]["checkpoint"], cfg["checkpoint_sha256"], "cpu")
    audit.pop("projection_model_class")
    rep["vendor"] = audit
    model.eval()
    enc = encoder_from_vendor_model(model)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 16, 224, 224, generator=g)
    with torch.no_grad():
        y = model(x)
        z = enc.encode(x)
    rep["parity_original_vs_adapter"] = {
        "features_max_abs": float((y["features"] - z["features"]).abs().max()),
        "latent_max_abs": float((y["latent"] - z["latent"]).abs().max()),
        "cls_head_keys": sorted(k for k in y["cls_head"] if k not in ("latent", "post_latent")),
        "cls_active_shape": list(y["cls_active"].shape),
    }
    ctx.write_json("preflight.json", rep)
    ctx.write_json("interface_contract.json", {
        "input": "BCTHW float32 [B,3,16,224,224], ImageNet-normalized RGB, resize 224 (INTER_LINEAR)",
        "encoder_outputs": {"features": [None, 768], "latent": [None, 768]},
        "backbone_call": "backbone(rgb, mask_ratio, mask, eval=False) -> (latent, maps, cls)",
        "mask_units": enc.num_mask_units(),
        "mask_polarity": "True = keep (Hiera.get_random_mask)",
    })
    print(json.dumps(rep["parity_original_vs_adapter"], indent=2))


# ---------------------------------------------------------------- audit-data
def _audit_worker(a):
    from .data.manifest import audit_record

    rec, T, dev, thr = a
    return audit_record(rec, T, dev, thr)


def cmd_audit_data(args, cfg, ctx):
    from multiprocessing import Pool

    from .data.manifest import read_split_rows, write_jsonl

    _require(cfg, "dataset_root")
    root = _abs(cfg["paths"]["dataset_root"])
    label_map = json.load(open(root / "label_maps.json"))
    rows = read_split_rows(root, label_map)
    if args.limit:
        rows = rows[: args.limit]
    T = cfg["data"]["clip_length"]
    dev = cfg["data"]["max_relative_dt_deviation"]
    thr = cfg["targets"]["confidence_threshold"]
    out = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(_audit_worker, [(r, T, dev, thr) for r in rows], chunksize=16)):
            out.append(r)
            if i % 5000 == 0:
                print(f"audited {i}/{len(rows)}", flush=True)
    write_jsonl(ctx.dir / "manifest.jsonl", out)
    summary = {}
    for split in ("train", "val", "test"):
        rs = [r for r in out if r["official_split"] == split]
        reasons = {}
        for r in rs:
            for x in r["reasons"]:
                key = x.split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        el = [r for r in rs if r["eligible"]]
        summary[split] = {
            "records": len(rs),
            "eligible": len(el),
            "exclusion_reasons": reasons,
            "signers": len({r["signer_id"] for r in rs}),
            "classes": len({r["label_id"] for r in rs}),
            "eligible_classes": len({r["label_id"] for r in el}),
            "mean_frames": float(sum(r.get("num_frames", 0) for r in rs) / max(len(rs), 1)),
            "mean_left_present": float(sum(r.get("frac_left_present", 0) for r in el) / max(len(el), 1)),
            "mean_right_present": float(sum(r.get("frac_right_present", 0) for r in el) / max(len(el), 1)),
        }
    ctx.write_json("audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- build-targets (coverage audit)
def _coverage_worker(a):
    import numpy as np

    from .data.manifest import record_valid_starts
    from .data.pose_reader import read_raw_pose
    from .data.priors import SPECTRAL_GROUPS, build_canonical_targets
    from .data.scaler import scaler_start

    rec, cfg_t, T, windows = a
    raw = read_raw_pose(rec["pose_path"], cfg_t["confidence_threshold"])
    start = scaler_start(rec["sample_id"], cfg_t["scaler_seed"], record_valid_starts(rec, T))
    tb = build_canonical_targets(raw, start, T, rec["width"], rec["height"], cfg_t["scale_min_observations"], cfg_t["scale_floor"],
                                 cfg_t["max_relative_extent"])
    res = {"sample_id": rec["sample_id"]}
    for h, m in tb["masks"].items():
        res[f"obs_frac/{h}"] = float(m.mean())
    for g, (h, sl) in SPECTRAL_GROUPS.items():
        v = tb["masks"][h][:, sl]
        for w in windows:
            n, hop = w["n"], w["hop"]
            rows = [v[s : s + n].all(0) for s in range(0, T - n + 1, hop)]
            res[f"spec_frac/{g}/N{n}"] = float(np.mean(rows))
    return res


def cmd_build_targets(args, cfg, ctx):
    from multiprocessing import Pool

    import numpy as np

    from .data.manifest import read_jsonl

    _require(cfg, "manifest")
    recs = [r for r in read_jsonl(_abs(cfg["paths"]["manifest"])) if r["eligible"] and r["official_split"] == "train"]
    T = cfg["data"]["clip_length"]
    with Pool(args.workers) as pool:
        out = pool.map(_coverage_worker, [(r, cfg["targets"], T, cfg["loss"]["windows"]) for r in recs], chunksize=32)
    keys = [k for k in out[0] if k != "sample_id"]
    summary = {k: float(np.mean([o[k] for o in out])) for k in keys}
    summary["num_clips"] = len(out)
    summary["policy"] = "one hash-sampled clip per eligible train video (same as scaler sampling)"
    ctx.write_json("target_coverage.json", summary)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- fit-scaler
def _scaler_worker(a):
    from .data.scaler import clip_moments

    rec, seed, cfg_t, T = a
    return clip_moments(rec, seed, cfg_t, T)


def cmd_fit_scaler(args, cfg, ctx):
    import hashlib
    from multiprocessing import Pool

    from .data.manifest import read_jsonl
    from .data.scaler import combine, save_scaler

    _require(cfg, "manifest")
    mpath = _abs(cfg["paths"]["manifest"])
    recs = [r for r in read_jsonl(mpath) if r["eligible"] and r["official_split"] == cfg["data"]["train_split"]]
    T = cfg["data"]["clip_length"]
    seed = cfg["targets"]["scaler_seed"]
    with Pool(args.workers) as pool:
        parts = pool.map(_scaler_worker, [(r, seed, cfg["targets"], T) for r in recs], chunksize=32)
    stats = combine(parts, cfg["targets"]["scaler_std_floor"])
    with open(ctx.dir / "fitted_clips.jsonl", "x") as f:
        for p in parts:
            f.write(json.dumps({"sample_id": p["sample_id"], "start": p["start"]}) + "\n")
    digest = save_scaler(ctx.dir / "scaler.npz", stats, {
        "target_schema": cfg["targets"]["schema"],
        "manifest": str(mpath),
        "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
        "split": cfg["data"]["train_split"],
        "num_clips": len(parts),
        "scaler_seed": seed,
        "std_floor": cfg["targets"]["scaler_std_floor"],
        "sampling_policy": "sha256(sample_id|seed) mod len(sorted valid starts); 1 clip/video",
    })
    print("scaler sha256", digest, "clips", len(parts))
    for h, st in stats.items():
        print(h, "count min", st["count"].min(), "std range", st["std"].min(), st["std"].max(), "unsupported", int(st["unsupported"].sum()))


# ---------------------------------------------------------------- train / calibrate
def cmd_calibrate(args, cfg, ctx):
    from .engine.train import calibrate

    _require(cfg, "manifest", "scaler")
    calibrate(cfg, ctx)


def cmd_train(args, cfg, ctx):
    from .engine.train import train

    _require(cfg, "manifest", "scaler")
    train(cfg, ctx)


def cmd_export(args, cfg, ctx):
    from .engine.checkpoint import export_encoder

    export_encoder(cfg, _abs(args.checkpoint), _abs(args.output), ctx)


def cmd_extract(args, cfg, ctx):
    from .engine.extract import extract

    _require(cfg, "manifest")
    if bool(args.encoder) == bool(args.reference_checkpoint):
        raise ConfigError("exactly one of --encoder / --reference-checkpoint is required")
    extract(cfg, ctx, encoder_path=_abs(args.encoder), reference=bool(args.reference_checkpoint),
            splits=args.splits.split(","), limit=args.limit)


def cmd_evaluate_retrieval(args, cfg, ctx):
    from .evaluation.retrieval import run_retrieval

    run_retrieval(ctx, _abs(args.features_dir), args.gallery_split, args.query_split)


def cmd_evaluate_probe(args, cfg, ctx):
    from .evaluation.probe import run_probe

    run_probe(ctx, _abs(args.features_dir), args.eval_split, json.loads(args.probe_config))


def cmd_diagnose_recon(args, cfg, ctx):
    from .evaluation.diagnostics import diagnose_reconstruction

    _require(cfg, "manifest", "scaler")
    diagnose_reconstruction(cfg, ctx, _abs(args.checkpoint), args.split, args.num_clips)


def cmd_compare(args, cfg, ctx):
    from .evaluation.diagnostics import compare_arms

    arms = dict(kv.split("=", 1) for kv in args.arms.split(","))
    compare_arms(ctx, arms, args.baseline, args.file)


def cmd_verify(args, cfg, ctx):
    from .engine.checkpoint import verify_dir

    verify_dir(ctx, _abs(args.target))


COMMANDS = {
    "preflight": cmd_preflight,
    "audit-data": cmd_audit_data,
    "build-targets": cmd_build_targets,
    "fit-scaler": cmd_fit_scaler,
    "calibrate": cmd_calibrate,
    "train": cmd_train,
    "export": cmd_export,
    "extract": cmd_extract,
    "evaluate-retrieval": cmd_evaluate_retrieval,
    "evaluate-probe": cmd_evaluate_probe,
    "verify": cmd_verify,
    "diagnose-recon": cmd_diagnose_recon,
    "compare": cmd_compare,
}


def build_parser():
    p = argparse.ArgumentParser(prog="python -m method.signrep_spectral.cli")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--paths-config", default=None)
        s.add_argument("--run-dir", required=True)
        if name in ("audit-data", "build-targets", "fit-scaler"):
            s.add_argument("--workers", type=int, default=16)
        if name == "audit-data":
            s.add_argument("--limit", type=int, default=0)
        if name == "export":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--output", required=True)
        if name == "extract":
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--encoder")
            g.add_argument("--reference-checkpoint", action="store_true",
                           help="use the original SignRep checkpoint from config (arm B0)")
            s.add_argument("--splits", default="val")
            s.add_argument("--limit", type=int, default=0)
        if name in ("evaluate-retrieval", "evaluate-probe"):
            s.add_argument("--features-dir", required=True)
        if name == "evaluate-retrieval":
            s.add_argument("--gallery-split", default="train")
            s.add_argument("--query-split", default="val")
        if name == "evaluate-probe":
            s.add_argument("--eval-split", default="val")
            s.add_argument("--probe-config", default="{}")
        if name == "diagnose-recon":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--split", default="val")
            s.add_argument("--num-clips", type=int, default=1024)
        if name == "compare":
            s.add_argument("--arms", required=True, help="ARM=run_dir,ARM=run_dir,...")
            s.add_argument("--baseline", required=True)
            s.add_argument("--file", default="per_query_pooled_features.jsonl")
        if name == "verify":
            s.add_argument("--target", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        cfg = resolved(args.config, args.paths_config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    ctx = RunContext(args.run_dir, args.command, cfg)
    try:
        COMMANDS[args.command](args, cfg, ctx)
    except ConfigError as e:
        ctx.status("BLOCKED", error=str(e))
        print(f"blocked: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        ctx.status("FAILED", error=repr(e), traceback=traceback.format_exc())
        raise
    ctx.finalize_outputs()
    if json.load(open(ctx.dir / "status.json"))["state"] == "RUNNING":
        ctx.status("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
