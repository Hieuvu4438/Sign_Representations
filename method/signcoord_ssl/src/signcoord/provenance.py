from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def tensor_state_hash(state):
    """Deterministic tensor bytes independent of torch-save container metadata."""
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        descriptor={'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype)}
        digest.update(json.dumps(descriptor,sort_keys=True).encode())
        raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def execution_stamp():
    root = Path(__file__).resolve().parents[4]
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    files = sorted(Path(__file__).parent.rglob("*.py"))
    return {"executed_at": datetime.now(timezone.utc).isoformat(),
            "code_revision": r.stdout.strip(),
            "implementation_hash": object_hash({str(p.relative_to(Path(__file__).parent)): sha256_file(p) for p in files})}
