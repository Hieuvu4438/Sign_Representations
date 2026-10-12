import json
from pathlib import Path
import numpy as np

from .hashing import file_hash
from .schema import ARRAY_DTYPES, IDENTITY_FIELDS, FeatureBundle
from ..geometry.supports import decode_supports


def read_bundle(path, expected, *, allow_fixture=False, mmap=True):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if not set(IDENTITY_FIELDS) <= expected.keys():
        raise ValueError("Reader requires expected feature identity, including weights and coordinates")
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Cache identity mismatch: {key}")
    names = {f"{name}.npy" for name in ARRAY_DTYPES} | {"raw_supports.npz"}
    if set(meta.get("array_sha256", {})) != names:
        raise ValueError("Cache checksum table is missing or incompatible")
    for name in sorted(names):
        if file_hash(path / name) != meta["array_sha256"][name]:
            raise ValueError(f"Cache content checksum mismatch: {name}")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False) for name in ARRAY_DTYPES}
    with np.load(path / "raw_supports.npz", allow_pickle=False) as encoded:
        supports = decode_supports(encoded)
    return FeatureBundle(meta, **arrays, raw_supports=supports).validate(allow_fixture=allow_fixture)
