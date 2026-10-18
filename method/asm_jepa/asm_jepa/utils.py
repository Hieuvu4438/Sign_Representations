"""Small shared helpers: hashing, atomic writes, JSONL, seeds, git provenance."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # method/asm_jepa


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value, indent: int = 2) -> None:
    atomic_write_bytes(path, (json.dumps(value, indent=indent, ensure_ascii=False, default=_json_default) + "\n").encode())


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_jsonl(path, rows: Iterable[dict]) -> None:
    data = "".join(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n" for r in rows)
    atomic_write_bytes(path, data.encode())


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def sample_seed(*parts: int) -> int:
    """Deterministic 63-bit seed from integers (base seed, step, micro position, sample index...)."""
    return int(np.random.SeedSequence([int(p) % (2**63) for p in parts]).generate_state(2, np.uint32).view(np.uint64)[0] >> 1)


def git_commit() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=PACKAGE_ROOT, text=True,
                                        stderr=subprocess.DEVNULL).strip() != ""
        return {"commit": commit, "package_dirty": dirty}
    except Exception:  # noqa: BLE001 - provenance is best effort
        return {"commit": None, "package_dirty": None}


def die(message: str, code: int = 2):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def resolve(path, base=None) -> Path:
    """Resolve a config path relative to `base` (default: the package root, i.e. method/asm_jepa)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(base) if base is not None else PACKAGE_ROOT) / p
