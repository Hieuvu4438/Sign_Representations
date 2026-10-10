"""Exact corpus medians, with expanded FK offsets rather than per-video scales."""

import json
from pathlib import Path
import numpy as np


def corpus_bone_statistics(cache_paths, output, manifest_sha256):
    """Compute one median length per teacher bone from observed pretraining frames.

    Out-of-core arrays retain exact medians without subsampling. Reference directions
    use coordinate-wise median offsets then rescale to the median Euclidean length;
    this directional choice (for multi-axis offsets) is recorded, not author-verified.
    """
    paths = [Path(p) for p in cache_paths]
    if not paths or not manifest_sha256:
        raise ValueError("Statistics require cache paths and pretraining manifest hash")
    output = Path(output)
    work = output.parent / (output.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    result, sample_counts = {}, {}
    for part, nodes in [("body", 22), ("hand", 21)]:
        def samples(path):
            with np.load(path, allow_pickle=False) as data:
                if part == "body":
                    yield data["body_offsets_full"][data["body_detected"]]
                else:
                    for side in ["right", "left"]:
                        yield data[f"{side}_offsets_full"][data[f"{side}_detected"]]

        count = sum(len(x) for path in paths for x in samples(path))
        if not count:
            raise ValueError(f"No observed {part} bones; cannot fabricate corpus statistics")
        vector_path = work / f"{part}.bin"
        vectors = np.memmap(vector_path, dtype=np.float32, mode="w+", shape=(count, nodes, 3))
        index = 0
        for path in paths:
            for block in samples(path):
                block = np.asarray(block, dtype=np.float32).reshape(-1, nodes, 3)
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite observed offsets in {path}")
                vectors[index:index + len(block)] = block
                index += len(block)
        standard, lengths = np.zeros((nodes, 3), np.float32), np.zeros(nodes, np.float32)
        for node in range(nodes):
            bone = np.asarray(vectors[:, node]).copy()
            length = float(np.median(np.linalg.norm(bone, axis=1)))
            direction = np.median(bone, axis=0)
            norm = np.linalg.norm(direction)
            if length > 0 and norm == 0:
                raise ValueError(f"Undefined reference direction for {part} bone {node}")
            if norm > 0:
                standard[node] = direction * (length / norm)
            lengths[node] = length
        vectors.flush()
        del vectors
        vector_path.unlink()
        result[f"{part}_offsets_full"] = standard[..., None]
        result[f"{part}_median_lengths"] = lengths
        sample_counts[part] = count
    metadata = {"schema": "signrep_bones_v1", "manifest_sha256": manifest_sha256,
                "sample_counts": sample_counts, "cache_files": [str(p.resolve()) for p in paths],
                "aggregation": "exact median over observed frames, shared left/right hand",
                "direction": "median expanded-offset direction (engineering convention)",
                "historical_author_statistics": "unverified"}
    result["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    work.rmdir()
    return metadata


def load_bone_statistics(path, expected_manifest_sha256=None):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if expected_manifest_sha256 and metadata["manifest_sha256"] != expected_manifest_sha256:
            raise ValueError("Bone statistics came from a different pretraining manifest")
        output = {key: data[key].copy() for key in data.files if key != "metadata_json"}
    for part, count in [("body", 22), ("hand", 21)]:
        offsets = output[f"{part}_offsets_full"]
        if offsets.shape != (count, 3, 1) or not np.isfinite(offsets).all():
            raise ValueError(f"Invalid {part} standard skeleton")
        lengths = np.linalg.norm(offsets[..., 0], axis=-1)
        if not np.allclose(lengths, output[f"{part}_median_lengths"], rtol=1e-5, atol=1e-7):
            raise ValueError("Standardized offsets do not preserve corpus median lengths")
    return output, metadata
