import json
from pathlib import Path
import shutil
import tempfile
import numpy as np

from .hashing import file_hash
from .schema import ARRAY_DTYPES, FeatureBundle
from ..geometry.supports import encode_supports


def write_bundle(path, bundle: FeatureBundle, *, allow_fixture=False, compressed=False):
    bundle.validate(allow_fixture=allow_fixture)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}")
    temp = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        if compressed:
            np.savez_compressed(temp / "features.npz", **{name: getattr(bundle, name) for name in ARRAY_DTYPES})
        else:
            for name in ARRAY_DTYPES:
                np.save(temp / f"{name}.npy", getattr(bundle, name), allow_pickle=False)
        np.savez_compressed(temp / "raw_supports.npz", **encode_supports(bundle.raw_supports))
        meta = dict(bundle.meta)
        meta["storage_format"] = "npz_compressed_v1" if compressed else "npy_v1"
        meta["array_sha256"] = {file.name: file_hash(file) for file in temp.iterdir()}
        (temp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False))
        temp.rename(path)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
