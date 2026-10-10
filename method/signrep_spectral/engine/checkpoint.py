"""Training checkpoint (schema v1), RGB-only encoder export with parity, artifact verification."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

CKPT_SCHEMA = "signrep_spectral_checkpoint_v1"
EXPORT_SCHEMA = "signrep_spectral_encoder_v1"


def rng_state() -> dict:
    """RNG states as tensors / plain lists so weights_only=True loading works."""
    np_state = np.random.get_state()
    out = {
        "python": json.dumps(random.getstate()),
        "numpy": {"key": torch.from_numpy(np_state[1].astype(np.int64)), "pos": int(np_state[2])},
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state(state: dict) -> None:
    s = json.loads(state["python"])
    random.setstate((s[0], tuple(s[1]), s[2]))
    np.random.set_state(("MT19937", state["numpy"]["key"].numpy().astype(np.uint32), state["numpy"]["pos"]))
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_save(obj, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_training_checkpoint(path, model, opt, cfg, audit, updates, consumed, scaler_sha256) -> None:
    atomic_save({
        "schema_version": CKPT_SCHEMA,
        "mode": cfg["mode"],
        "encoder_state": {"backbone": model.encoder.backbone.state_dict(), "fc": model.encoder.fc.state_dict()},
        "decoder_state": model.decoder.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": {"kind": cfg["train"]["scheduler"], "updates": updates},
        "grad_scaler": None,
        "successful_updates": updates,
        "epoch": 0,
        "sampler_state": {"consumed_batches": consumed},
        "rng": rng_state(),
        "resolved_config": json.dumps(cfg, sort_keys=True),
        "provenance": json.dumps({k: v for k, v in audit.items() if k != "source_hashes"}, default=str),
        "target_schema": cfg["targets"]["schema"],
        "scaler_hash": scaler_sha256,
        "initial_state_hash": audit["initial_state_hash"],
    }, path)


def export_encoder(cfg, checkpoint: Path, output: Path, ctx) -> None:
    from ..model import attention_kernel, build_empty_encoder, state_hash

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("schema_version") != CKPT_SCHEMA:
        raise ValueError("not a signrep_spectral training checkpoint")
    enc = build_empty_encoder(cfg["paths"]["vendor_root"])
    enc.backbone.load_state_dict(ck["encoder_state"]["backbone"], strict=True)
    enc.fc.load_state_dict(ck["encoder_state"]["fc"], strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EXPORT_SCHEMA,
        "encoder_state": {"backbone": enc.backbone.state_dict(), "fc": enc.fc.state_dict()},
        "architecture": "mae_hiera_base_16x224 + LayerNorm/Linear(768,768,bias=False)",
        "output_keys": ["features", "latent"],
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "original_signrep_sha256": cfg["checkpoint_sha256"],
        "training_config": ck["resolved_config"],
        "target_schema": ck["target_schema"],
        "scaler_hash": ck["scaler_hash"],
        "successful_updates": ck["successful_updates"],
        "encoder_hash": state_hash(enc),
    }
    atomic_save(payload, output)
    # strict reload + parity on a fixed RGB fixture (eval, mask 0, no decoder/teacher/FFT)
    enc2 = load_exported_encoder(cfg["paths"]["vendor_root"], output)
    enc.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 16, 224, 224, generator=g)
    with torch.no_grad(), attention_kernel("vendor_default"):
        a, b = enc.encode(x), enc2.encode(x)
    parity = {k: float((a[k] - b[k]).abs().max()) for k in ("features", "latent")}
    if max(parity.values()) != 0.0:
        raise RuntimeError(f"export parity failed {parity}")
    ctx.write_json("export_report.json", {"output": str(output), "sha256": sha256_file(output), "parity_max_abs": parity,
                                          "encoder_hash": payload["encoder_hash"], "successful_updates": ck["successful_updates"]})


def load_exported_encoder(vendor_root, path: Path):
    from ..model import build_empty_encoder

    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("schema_version") != EXPORT_SCHEMA:
        raise ValueError("not an exported signrep_spectral encoder")
    enc = build_empty_encoder(vendor_root)
    enc.backbone.load_state_dict(ck["encoder_state"]["backbone"], strict=True)
    enc.fc.load_state_dict(ck["encoder_state"]["fc"], strict=True)
    enc.eval()
    return enc


def verify_dir(ctx, target: Path) -> None:
    report = {"target": str(target), "files": {}}
    status = json.load(open(target / "status.json")) if (target / "status.json").exists() else None
    report["status"] = status
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.suffix in (".pt", ".npz", ".jsonl", ".json"):
            report["files"][str(p.relative_to(target))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    ctx.write_json("verify_report.json", report)
