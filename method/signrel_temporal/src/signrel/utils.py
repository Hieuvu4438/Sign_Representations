"""Local provenance, stable seeds and atomic artifacts."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def stable_seed(*parts):
    return int(fingerprint(list(parts))[:15], 16)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows))
    os.replace(temporary, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def code_hash(root):
    root = Path(root)
    return fingerprint({str(p.relative_to(root)): sha256(p)
                        for p in sorted((root/'src').rglob('*.py'))})


def finite_arrays(values):
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite numeric array: {name}')
