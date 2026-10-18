"""Fixed train-only feature statistics (§7.4): container, identity fixture, accumulator, save/load."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from asm_jepa.data.schema import N_JOINTS, N_RELATIONS, PATH_FAMILIES

STD_FLOOR = 1e-3
N_SCALES_MAX = 8
KEYS = ("body", "local", "rel", "geom", "time")


def stat_shapes(n_scales: int) -> dict:
    nf = len(PATH_FAMILIES)
    return {"body": (N_JOINTS, 2), "local": (N_JOINTS, 2), "rel": (N_RELATIONS, 3),
            "geom": (nf, n_scales, 3), "time": (nf, n_scales, 7)}


def identity_stats(n_scales: int = 3) -> dict:
    """Synthetic-fixture statistics (mean 0, std 1); never used for real ASL-Citizen runs."""
    out = {}
    for k, shape in stat_shapes(n_scales).items():
        out[f"{k}_mean"] = np.zeros(shape, np.float32)
        out[f"{k}_std"] = np.ones(shape, np.float32)
    out["_info"] = {"kind": "identity_synthetic_fixture"}
    return out


class StatsAccumulator:
    """Float64 sum / sum of squares / count over observed values only."""

    def __init__(self, n_scales: int):
        self.shapes = stat_shapes(n_scales)
        self.s = {k: np.zeros(v) for k, v in self.shapes.items()}
        self.ss = {k: np.zeros(v) for k, v in self.shapes.items()}
        self.n = {k: np.zeros(v) for k, v in self.shapes.items()}

    def add(self, key: str, values: np.ndarray, valid: np.ndarray) -> None:
        """values [..., *shape]; valid broadcastable to values (bool)."""
        v = np.asarray(values, np.float64)
        m = np.broadcast_to(np.asarray(valid, bool), v.shape)
        lead = v.ndim - len(self.shapes[key])
        axes = tuple(range(lead))
        self.s[key] += np.where(m, v, 0.0).sum(axes)
        self.ss[key] += np.where(m, v * v, 0.0).sum(axes)
        self.n[key] += m.sum(axes)

    def finalize(self) -> dict:
        out = {}
        for k in self.shapes:
            n = np.maximum(self.n[k], 1.0)
            mean = np.where(self.n[k] > 0, self.s[k] / n, 0.0)
            var = np.where(self.n[k] > 1, self.ss[k] / n - mean**2, 1.0)
            std = np.maximum(np.sqrt(np.maximum(var, 0.0)), STD_FLOOR)
            out[f"{k}_mean"] = mean.astype(np.float32)
            out[f"{k}_std"] = std.astype(np.float32)
            out[f"{k}_count"] = self.n[k].astype(np.int64)
        return out


def save_stats(path, stats: dict, info: dict) -> None:
    import json

    from asm_jepa.utils import atomic_write_bytes
    import io

    buf = io.BytesIO()
    arrays = {k: v for k, v in stats.items() if not k.startswith("_")}
    np.savez(buf, **arrays, info_json=np.array(json.dumps(info, sort_keys=True)))
    atomic_write_bytes(path, buf.getvalue())


def load_stats(path, n_scales: int) -> dict:
    import json

    if not Path(path).is_file():
        raise FileNotFoundError(f"feature statistics {path} not found; run `python -m asm_jepa.cli.fit_stats` "
                                "(train split only) or use --synthetic for the fixture")
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "info_json"}
        out["_info"] = json.loads(str(z["info_json"])) if "info_json" in z.files else {}
    for k, shape in stat_shapes(n_scales).items():
        for suffix in ("mean", "std"):
            if out[f"{k}_{suffix}"].shape != shape:
                raise ValueError(f"stats {k}_{suffix} has shape {out[f'{k}_{suffix}'].shape}, expected {shape}")
    return out
