"""Train-only standardization and fixed Gaussian projection for matched controls."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from ..cache.hashing import canonical_hash, file_hash, is_sha256
from ..cli import write_json


@dataclass(frozen=True)
class ProjectionObservation:
    source_group_id: str
    split: str
    group_scale: str
    descriptor: np.ndarray


@dataclass(frozen=True)
class Projection:
    mean: np.ndarray
    std: np.ndarray
    matrix: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    count: int

    def apply(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Projection input dimension/nonfinite mismatch")
        return ((((x - self.mean) / self.std) @ self.matrix - self.output_mean) / self.output_std).astype(np.float32)


@dataclass(frozen=True)
class TargetProjectors:
    projections: dict[str, Projection]
    metadata: dict

    def fingerprint(self):
        import hashlib
        matrices = {}
        for key, projection in sorted(self.projections.items()):
            matrices[key] = {name: hashlib.sha256(np.ascontiguousarray(getattr(projection, name)).tobytes()).hexdigest()
                             for name in ("mean", "std", "matrix", "output_mean", "output_std")}
            matrices[key]["count"] = projection.count
        meta = {k: v for k, v in self.metadata.items() if k not in {"arrays_sha256", "group_order", "metadata_hash"}}
        return canonical_hash({"metadata": meta, "transforms": matrices})

    @classmethod
    def fit_batches(cls, batches, *, allowed_train_sources, manifest_sha256, extractor_hash,
                    descriptor_config, output_dim=32, seed=991, std_floor=1e-6):
        """Exact population moments in bounded memory, including projected covariance."""
        if not is_sha256(manifest_sha256) or not is_sha256(extractor_hash) or output_dim <= 0 or std_floor <= 0:
            raise ValueError("Projection fit requires actual data/feature hashes and positive dimensions")
        moments, sources = {}, set()
        for batch in batches:
            groups = {}
            for item in batch:
                if item.split != "train" or item.source_group_id not in allowed_train_sources:
                    raise ValueError("Projection statistics must use registered training sources only")
                value = np.asarray(item.descriptor, dtype=np.float64)
                if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                    raise ValueError("Projection fit needs finite descriptor vectors")
                groups.setdefault(item.group_scale, []).append(value)
                sources.add(item.source_group_id)
            for key, values in groups.items():
                x = np.stack(values)
                n, mean = len(x), x.mean(axis=0)
                centered = x - mean
                m2 = centered.T @ centered
                if key in moments:
                    old_n, old_mean, old_m2 = moments[key]
                    delta = mean - old_mean
                    total = old_n + n
                    moments[key] = (total, old_mean + delta * (n / total),
                        old_m2 + m2 + np.outer(delta, delta) * (old_n * n / total))
                else:
                    moments[key] = n, mean, m2
        if not moments:
            raise ValueError("No valid train descriptors for projection fitting")
        result = {}
        for key, (n, mean, m2) in sorted(moments.items()):
            covariance = m2 / n
            std = np.maximum(np.sqrt(np.maximum(np.diag(covariance), 0)), std_floor)
            group_seed = int(canonical_hash({"seed": seed, "group_scale": key})[:16], 16)
            matrix = np.random.default_rng(group_seed).normal(0, 1 / np.sqrt(len(mean)), (len(mean), output_dim))
            projected_covariance = matrix.T @ (covariance / std[:, None] / std[None, :]) @ matrix
            output_std = np.maximum(np.sqrt(np.maximum(np.diag(projected_covariance), 0)), std_floor)
            result[key] = Projection(mean, std, matrix, np.zeros(output_dim), output_std, n)
        metadata = {"schema_version": 1, "fitting_split": "train", "fitting_source_ids": sorted(sources),
            "manifest_sha256": manifest_sha256, "extractor_hash": extractor_hash, "descriptor_config": descriptor_config,
            "descriptor_hash": canonical_hash(descriptor_config), "projection_kind": "fixed_gaussian", "output_dim": output_dim,
            "seed": seed, "std_floor": std_floor, "native_dimensions": {key: len(p.mean) for key, p in result.items()},
            "counts": {key: p.count for key, p in result.items()}}
        return cls(result, metadata)

    @classmethod
    def fit(cls, observations, *, allowed_train_sources, manifest_sha256, extractor_hash,
            descriptor_config, output_dim=32, seed=991, std_floor=1e-6):
        if not is_sha256(manifest_sha256) or not is_sha256(extractor_hash) or output_dim <= 0 or std_floor <= 0:
            raise ValueError("Projection fit requires actual data/feature hashes and positive dimensions")
        groups, sources = {}, set()
        for item in observations:
            if item.split != "train" or item.source_group_id not in allowed_train_sources:
                raise ValueError("Projection statistics must use registered training sources only")
            value = np.asarray(item.descriptor, dtype=np.float64)
            if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                raise ValueError("Projection fit needs finite descriptor vectors")
            groups.setdefault(item.group_scale, []).append(value)
            sources.add(item.source_group_id)
        if not groups:
            raise ValueError("No valid train descriptors for projection fitting")
        result = {}
        for key, vectors in sorted(groups.items()):
            x = np.stack(vectors)
            mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), std_floor)
            group_seed = int(canonical_hash({"seed": seed, "group_scale": key})[:16], 16)
            matrix = np.random.default_rng(group_seed).normal(0, 1 / np.sqrt(x.shape[-1]), (x.shape[-1], output_dim))
            projected = ((x - mean) / std) @ matrix
            result[key] = Projection(mean, std, matrix, projected.mean(axis=0), np.maximum(projected.std(axis=0), std_floor), len(x))
        metadata = {"schema_version": 1, "fitting_split": "train", "fitting_source_ids": sorted(sources),
            "manifest_sha256": manifest_sha256, "extractor_hash": extractor_hash, "descriptor_config": descriptor_config,
            "descriptor_hash": canonical_hash(descriptor_config), "projection_kind": "fixed_gaussian",
            "output_dim": output_dim, "seed": seed, "std_floor": std_floor,
            "native_dimensions": {key: len(p.mean) for key, p in result.items()},
            "counts": {key: p.count for key, p in result.items()}}
        return cls(result, metadata)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {}
        keys = sorted(self.projections)
        for i, key in enumerate(keys):
            p = self.projections[key]
            for name in ("mean", "std", "matrix", "output_mean", "output_std"):
                arrays[f"g{i}_{name}"] = getattr(p, name)
        path = directory / "projection.npz"
        np.savez(path, **arrays)
        metadata = {**self.metadata, "group_order": keys, "arrays_sha256": file_hash(path)}
        metadata["metadata_hash"] = canonical_hash(metadata)
        write_json(directory / "metadata.json", metadata)

    def save_files(self, statistics_path, matrices_path):
        """Separate NPZ contracts from proposal config; metadata is the commit marker."""
        import os
        import tempfile
        stats, matrices = Path(statistics_path), Path(matrices_path)
        metadata_path = stats.with_suffix(stats.suffix + ".json")
        if any(p.exists() for p in (stats, matrices, metadata_path)) or stats.resolve() == matrices.resolve():
            raise FileExistsError("Target files already exist or statistics/matrix paths coincide")
        stats.parent.mkdir(parents=True, exist_ok=True)
        matrices.parent.mkdir(parents=True, exist_ok=True)
        groups, stats_arrays, matrix_arrays = sorted(self.projections), {}, {}
        for i, key in enumerate(groups):
            projection = self.projections[key]
            for name in ("mean", "std", "output_mean", "output_std"):
                stats_arrays[f"g{i}_{name}"] = getattr(projection, name)
            matrix_arrays[f"g{i}_matrix"] = projection.matrix
        temporary = []
        try:
            for path, arrays in ((stats, stats_arrays), (matrices, matrix_arrays)):
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                    temp = Path(stream.name)
                    temporary.append(temp)
                    np.savez(stream, **arrays)
                os.link(temp, path)
            meta = {**self.metadata, "group_order": groups, "statistics_sha256": file_hash(stats),
                    "matrices_sha256": file_hash(matrices), "fingerprint": self.fingerprint()}
            meta["metadata_hash"] = canonical_hash(meta)
            write_json(metadata_path, meta)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    @classmethod
    def load_files(cls, statistics_path, matrices_path, *, expected_manifest_sha256, expected_extractor_hash, expected_descriptor_config):
        stats, matrices = Path(statistics_path), Path(matrices_path)
        meta = json.loads(stats.with_suffix(stats.suffix + ".json").read_text())
        digest = meta.pop("metadata_hash", None)
        if canonical_hash(meta) != digest:
            raise ValueError("Target metadata checksum mismatch")
        if meta["fitting_split"] != "train" or meta["manifest_sha256"] != expected_manifest_sha256 or meta["extractor_hash"] != expected_extractor_hash or meta["descriptor_hash"] != canonical_hash(expected_descriptor_config):
            raise ValueError("Target fitting/descriptor identity mismatch")
        if file_hash(stats) != meta["statistics_sha256"] or file_hash(matrices) != meta["matrices_sha256"]:
            raise ValueError("Target statistics/matrices checksum mismatch")
        result = {}
        with np.load(stats, allow_pickle=False) as statistics, np.load(matrices, allow_pickle=False) as mapping:
            for i, key in enumerate(meta["group_order"]):
                result[key] = Projection(**{name: statistics[f"g{i}_{name}"] for name in ("mean", "std", "output_mean", "output_std")},
                                        matrix=mapping[f"g{i}_matrix"], count=meta["counts"][key])
        for key, projection in result.items():
            native, out = meta["native_dimensions"][key], meta["output_dim"]
            expected = {"mean": (native,), "std": (native,), "matrix": (native, out), "output_mean": (out,), "output_std": (out,)}
            if any(getattr(projection, name).shape != shape or getattr(projection, name).dtype != np.float64
                   or not np.isfinite(getattr(projection, name)).all() for name, shape in expected.items()):
                raise ValueError("Target transform array schema mismatch")
            if (projection.std <= 0).any() or (projection.output_std <= 0).any():
                raise ValueError("Target transform scales must be positive")
        storage_keys = {"group_order", "statistics_sha256", "matrices_sha256", "fingerprint"}
        model = cls(result, {k: v for k, v in meta.items() if k not in storage_keys})
        if model.fingerprint() != meta["fingerprint"]:
            raise ValueError("Target transform fingerprint mismatch")
        return model

    @classmethod
    def load(cls, directory, *, expected_manifest_sha256, expected_extractor_hash, expected_descriptor_config):
        directory = Path(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        expected_hash = meta.pop("metadata_hash", None)
        if canonical_hash(meta) != expected_hash:
            raise ValueError("Target projection metadata checksum mismatch")
        if meta["schema_version"] != 1 or meta["fitting_split"] != "train" or meta["manifest_sha256"] != expected_manifest_sha256 or meta["extractor_hash"] != expected_extractor_hash or meta["descriptor_hash"] != canonical_hash(expected_descriptor_config):
            raise ValueError("Target projection fitting source/descriptor identity mismatch")
        if file_hash(directory / "projection.npz") != meta["arrays_sha256"]:
            raise ValueError("Target projection array checksum mismatch")
        projections = {}
        with np.load(directory / "projection.npz", allow_pickle=False) as arrays:
            for i, key in enumerate(meta["group_order"]):
                values = {name: arrays[f"g{i}_{name}"] for name in ("mean", "std", "matrix", "output_mean", "output_std")}
                native, out = meta["native_dimensions"][key], meta["output_dim"]
                shapes = {"mean": (native,), "std": (native,), "matrix": (native, out), "output_mean": (out,), "output_std": (out,)}
                if any(value.shape != shapes[name] or value.dtype != np.float64 or not np.isfinite(value).all() for name, value in values.items()):
                    raise ValueError("Target projection array schema mismatch")
                if (values["std"] <= 0).any() or (values["output_std"] <= 0).any():
                    raise ValueError("Target projection scales must be positive")
                projections[key] = Projection(**values, count=meta["counts"][key])
        return cls(projections, meta)
