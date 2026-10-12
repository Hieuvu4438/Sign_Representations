"""Seven-stream train-only hidden units; never original SHuBERT/gloss labels."""
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import numpy as np

from ..cache.hashing import canonical_hash, file_hash, is_sha256
from ..cli import write_json
from ..pose.coordinates import static_pose_streams
from ..pose.topology import POSE_GROUPS


def stream_vectors(bundle):
    pose = static_pose_streams(bundle.pose_xy_body, bundle.pose_quality, bundle.pose_coordinate_valid)
    values = [bundle.visual[:, i].astype(np.float64) for i in range(4)] + list(pose)
    valid = [bundle.visual_valid[:, i] for i in range(4)] + [bundle.pose_coordinate_valid[:, joints].all(axis=1) for joints in POSE_GROUPS]
    return values, valid


@dataclass(frozen=True)
class Codebook:
    mean: np.ndarray
    std: np.ndarray
    centroids: np.ndarray

    def assign(self, vectors):
        x = (np.asarray(vectors, dtype=np.float64) - self.mean) / self.std
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Invalid codebook input shape/finite values")
        distance = np.sum(x * x, axis=1, keepdims=True) + np.sum(self.centroids ** 2, axis=1)[None] - 2 * x @ self.centroids.T
        return distance.argmin(axis=1).astype(np.int64)


@dataclass(frozen=True)
class KMeansTargets:
    books: tuple[Codebook, ...]
    metadata: dict

    @classmethod
    def fit(cls, records, *, manifest_sha256, extractor_hash, seed=17, clusters=256, max_per_source=2048):
        from sklearn.cluster import MiniBatchKMeans
        from threadpoolctl import threadpool_limits
        if not is_sha256(manifest_sha256) or not is_sha256(extractor_hash) or clusters < 2:
            raise ValueError("K-means requires actual data/feature identities and at least two clusters")
        by_source = {}
        for record, bundle in records:
            view = record.pretrain_view()  # rejects all held-out annotation views
            if bundle.meta["extractor_hash"] != extractor_hash:
                raise ValueError("K-means pool mixes extractor identities")
            values, valid = stream_vectors(bundle)
            unique = np.zeros(len(bundle.timestamps), dtype=bool)
            _, first = np.unique(bundle.source_frame_index, return_index=True)
            unique[first] = True
            source = by_source.setdefault(view.source_group_id, [[] for _ in range(7)])
            for stream in range(7):
                source[stream].append(values[stream][valid[stream] & unique])
        if not by_source:
            raise ValueError("K-means pool has no training sources")
        pools = [[] for _ in range(7)]
        for source_id, streams in sorted(by_source.items()):
            rng = np.random.default_rng(int(canonical_hash({"seed": seed, "source": source_id})[:16], 16))
            for stream, arrays in enumerate(streams):
                x = np.concatenate(arrays)
                if len(x) > max_per_source:
                    x = x[rng.choice(len(x), max_per_source, replace=False)]
                pools[stream].append(x)
        books, counts, occupancy = [], [], []
        for stream, arrays in enumerate(pools):
            x = np.concatenate(arrays).astype(np.float64)
            if len(x) < clusters:
                raise ValueError(f"Stream{stream}: {len(x)} valid train vectors cannot fit requested K={clusters}")
            mean, std = np.zeros(x.shape[1]), np.ones(x.shape[1])
            if stream >= 4:
                # Fit coordinates/quality only; preserve binary validity indicators as indicators.
                mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
                mean[3::4], std[3::4] = 0, 1
            with threadpool_limits(limits=1):
                model = MiniBatchKMeans(n_clusters=clusters, random_state=seed + stream, n_init=3,
                    batch_size=max(256, clusters), max_iter=100, reassignment_ratio=.01)
                assigned = model.fit_predict((x - mean) / std)
            books.append(Codebook(mean, std, model.cluster_centers_.astype(np.float64)))
            counts.append(len(x))
            occupancy.append(np.bincount(assigned, minlength=clusters).tolist())
        return cls(tuple(books), {"schema_version": 1, "fitting_split": "train", "fitting_source_ids": sorted(by_source),
            "manifest_sha256": manifest_sha256, "extractor_hash": extractor_hash, "seed": seed, "clusters": clusters,
            "max_per_source": max_per_source, "sample_counts": counts, "occupancy": occupancy,
            "sklearn_version": importlib.metadata.version("scikit-learn"), "recipe": "proposed_seven_stream_cluster_control"})

    def __call__(self, bundle):
        if bundle.meta["extractor_hash"] != self.metadata["extractor_hash"]:
            raise ValueError("Codebook/cache extractor mismatch")
        values, valid = stream_vectors(bundle)
        ids = np.full((len(bundle.timestamps), 7), -100, dtype=np.int64)
        for stream, book in enumerate(self.books):
            ids[valid[stream], stream] = book.assign(values[stream][valid[stream]])
        return ids

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {f"s{i}_{name}": getattr(book, name) for i, book in enumerate(self.books) for name in ("mean", "std", "centroids")}
        np.savez(directory / "codebooks.npz", **arrays)
        metadata = {**self.metadata, "arrays_sha256": file_hash(directory / "codebooks.npz")}
        metadata["metadata_hash"] = canonical_hash(metadata)
        write_json(directory / "metadata.json", metadata)

    @classmethod
    def load(cls, directory, *, manifest_sha256, extractor_hash, clusters=256):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        digest = meta.pop("metadata_hash", None)
        if canonical_hash(meta) != digest or file_hash(directory / "codebooks.npz") != meta["arrays_sha256"]:
            raise ValueError("Codebook integrity mismatch")
        if meta["schema_version"] != 1 or meta["fitting_split"] != "train" or meta["manifest_sha256"] != manifest_sha256 or meta["extractor_hash"] != extractor_hash or meta["clusters"] != clusters:
            raise ValueError("Codebook fitting data/feature identity mismatch")
        with np.load(directory / "codebooks.npz", allow_pickle=False) as arrays:
            books = tuple(Codebook(*(arrays[f"s{i}_{name}"] for name in ("mean", "std", "centroids"))) for i in range(7))
        for book in books:
            if book.mean.ndim != 1 or book.std.shape != book.mean.shape or book.centroids.shape != (clusters, len(book.mean)) or not all(np.isfinite(x).all() for x in (book.mean, book.std, book.centroids)) or (book.std <= 0).any():
                raise ValueError("Codebook array schema mismatch")
        return cls(books, meta)
