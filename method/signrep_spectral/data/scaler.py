"""Train-only per-channel scaler (streaming float64 moments over one sampled clip/video)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .manifest import record_valid_starts
from .pose_reader import read_raw_pose
from .priors import HEADS, build_canonical_targets


def scaler_start(sample_id: str, seed: int, valid_starts: np.ndarray) -> int:
    digest = hashlib.sha256(f"{sample_id}|{seed}".encode()).digest()
    idx = int.from_bytes(digest, "big") % len(valid_starts)
    return int(np.sort(valid_starts)[idx])


def clip_moments(rec: dict, seed: int, cfg_t: dict, clip_length: int) -> dict:
    starts = record_valid_starts(rec, clip_length)
    start = scaler_start(rec["sample_id"], seed, starts)
    raw = read_raw_pose(rec["pose_path"], cfg_t["confidence_threshold"])
    tb = build_canonical_targets(raw, start, clip_length, rec["width"], rec["height"],
                                 cfg_t["scale_min_observations"], cfg_t["scale_floor"],
                                 cfg_t["max_relative_extent"])
    out = {"sample_id": rec["sample_id"], "start": start}
    for h in HEADS:
        x = tb["targets"][h].astype(np.float64)
        m = tb["masks"][h]
        out[h] = (m.sum(0), np.where(m, x, 0).sum(0), np.where(m, x * x, 0).sum(0))
    return out


def combine(parts: List[dict], std_floor: float) -> Dict[str, dict]:
    res = {}
    for h, d in HEADS.items():
        n = np.zeros(d)
        s = np.zeros(d)
        ss = np.zeros(d)
        for p in parts:
            n += p[h][0]
            s += p[h][1]
            ss += p[h][2]
        mean = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        var = np.where(n > 1, (ss - n * mean**2) / np.maximum(n - 1, 1), 1.0)
        std = np.sqrt(np.maximum(var, 0.0))
        unsupported = n < 2
        scale = np.where(unsupported, 1.0, np.maximum(std, std_floor))
        res[h] = {"count": n, "mean": mean, "std": std, "scale": scale, "unsupported": unsupported}
    return res


def save_scaler(path_npz: Path, stats: Dict[str, dict], meta: dict) -> str:
    arrays = {}
    for h, st in stats.items():
        for k, v in st.items():
            arrays[f"{h}__{k}"] = np.asarray(v)
    with open(path_npz, "xb") as f:
        np.savez(f, **arrays)
    digest = hashlib.sha256(path_npz.read_bytes()).hexdigest()
    meta = dict(meta, sha256=digest, channel_order={h: d for h, d in HEADS.items()})
    with open(path_npz.with_suffix(".json"), "x") as f:
        json.dump(meta, f, indent=2)
    return digest


def load_scaler(path_npz: Path, expected_sha256: str | None = None) -> Dict[str, dict]:
    digest = hashlib.sha256(Path(path_npz).read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"scaler hash mismatch {digest} != {expected_sha256}")
    z = np.load(path_npz, allow_pickle=False)
    out = {}
    for h, d in HEADS.items():
        out[h] = {"mean": z[f"{h}__mean"].astype(np.float32), "scale": z[f"{h}__scale"].astype(np.float32)}
        if out[h]["mean"].shape != (d,):
            raise ValueError(f"scaler channel mismatch for {h}")
    out["_sha256"] = digest
    return out
