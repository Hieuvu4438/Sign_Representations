"""Deterministic frozen embedding extraction (§13.1) and NPZ storage (no torch pickles)."""
from __future__ import annotations

import io
import json

import numpy as np
import torch

from asm_jepa.data.dataset import LabeledDataset, worker_init
from asm_jepa.utils import atomic_write_bytes

COMPAT_KEYS = ("checkpoint_sha256", "stats_sha256", "feature_key", "schema", "arch_hash", "encoder", "sampler")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast_ctx(device, precision: str):
    dev = torch.device(device)
    if dev.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=dev.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def extract_embeddings(backbone, dataset: LabeledDataset, batch_size=128, num_workers=4, device="cuda",
                       precision="bf16", progress: bool = False) -> dict:
    was_training = backbone.training
    backbone.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                         pin_memory=torch.device(device).type == "cuda",
                                         worker_init_fn=worker_init if num_workers else None)
    feats = {"state_part_concat1024_l2": [], "state_balanced_mean256_l2": []}
    usable, idx, cls, quality = [], [], [], []
    for i, batch in enumerate(loader):
        b = to_device(batch, device)
        with autocast_ctx(device, precision):
            f, present = backbone.embed_clip(b)
        ok = b["usable"] & present.any(1)
        for k in feats:
            feats[k].append((f[k].float() * ok[:, None]).cpu().numpy())
        usable.append(ok.cpu().numpy())
        n_state = 4 * backbone.n_patches
        quality.append(b["token_weight"][:, :n_state].float().mean(1).cpu().numpy())
        idx.append(batch["index"].numpy())
        cls.append(batch["class_id"].numpy())
        if progress and (i % 50 == 0):
            print(f"  extracted {sum(len(u) for u in usable)}/{len(dataset)}", flush=True)
    backbone.train(was_training)
    out = {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}
    out["usable"] = np.concatenate(usable).astype(bool)
    out["index"] = np.concatenate(idx)
    out["class_ids"] = np.concatenate(cls)
    out["quality"] = np.concatenate(quality).astype(np.float32)
    assert np.array_equal(out["index"], np.arange(len(dataset))), "extraction order mismatch"
    return out


def save_embeddings(path, arrays: dict, sample_ids, signer_ids, meta: dict) -> None:
    buf = io.BytesIO()
    extra = {"quality": arrays["quality"]} if "quality" in arrays else {}
    np.savez(buf, embeddings=arrays["state_part_concat1024_l2"], balanced=arrays["state_balanced_mean256_l2"],
             usable=arrays["usable"], class_ids=arrays["class_ids"].astype(np.int64), **extra,
             sample_ids=np.array(sample_ids), signer_ids=np.array(signer_ids),
             meta_json=np.array(json.dumps(meta, sort_keys=True)))
    atomic_write_bytes(path, buf.getvalue())


def load_embeddings(path, feature_key: str = "state_part_concat1024_l2") -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    if out["meta"].get("feature_key") != "state_part_concat1024_l2":
        raise ValueError(f"{path}: unexpected primary feature key {out['meta'].get('feature_key')}")
    out["features"] = out["embeddings"] if feature_key == "state_part_concat1024_l2" else out["balanced"]
    return out


def check_compatible(a: dict, b: dict) -> None:
    diffs = [k for k in COMPAT_KEYS if a["meta"].get(k) != b["meta"].get(k)]
    if diffs:
        raise ValueError(f"gallery/query embeddings are not comparable, differing keys: "
                         f"{ {k: (a['meta'].get(k), b['meta'].get(k)) for k in diffs} }")
