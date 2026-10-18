"""A content-checked corpus index shared by extraction, training and evaluation."""
from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
from pathlib import Path

from .hashing import canonical_hash, file_hash
from .reader import read_bundle
from .schema import IDENTITY_FIELDS
from ..cli import write_json
from ..data.manifest import read_manifest


@contextmanager
def index_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".index.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def index_path(root):
    return Path(root) / "index.json"


def sample_directory(root, extractor_hash, sample_id):
    return Path(root) / extractor_hash / canonical_hash(sample_id)[:24]


def record_cache(root, directory, record, *, allow_fixture=False):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Cache sample must belong to its declared root")
    meta = json.loads((directory / "meta.json").read_text())
    identity = {key: meta[key] for key in IDENTITY_FIELDS}
    if meta["sample_id"] != record.sample_id or meta["source_sha256"] != record.source_sha256:
        raise ValueError("Manifest/cache source identity mismatch")
    # Content validation precedes publication of the index entry.
    read_bundle(directory, identity, allow_fixture=allow_fixture)
    entry = {"path": str(directory.relative_to(root)), "source_sha256": record.source_sha256,
             "meta_sha256": file_hash(directory / "meta.json"), "split": record.split,
             "start_s": record.start_s, "end_s": record.end_s}
    with index_lock(root):
        path = index_path(root)
        index = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "identity": identity, "samples": {}, "fixture_only": bool(meta.get("synthetic_correctness_only"))}
        if index["schema_version"] != 1 or index["identity"] != identity:
            raise ValueError("A cache root cannot mix foundation/preprocessing identities")
        existing = index["samples"].get(record.sample_id)
        if existing is not None and existing != entry:
            raise ValueError("Sample ID already indexes a different source/interval/cache")
        index["samples"][record.sample_id] = entry
        write_json(path, index)
    return entry


class CachedCorpus:
    def __init__(self, manifest, root, *, split, config=None, allow_fixture=False):
        self.root, self.manifest = Path(root).resolve(), Path(manifest).resolve()
        self.index = json.loads(index_path(self.root).read_text())
        self.allow_fixture = allow_fixture
        if self.index.get("fixture_only", False) and not allow_fixture:
            raise ValueError("Synthetic fixture corpus cannot substitute for scientific cached data")
        if self.index["schema_version"] != 1 or not set(IDENTITY_FIELDS) <= self.index["identity"].keys():
            raise ValueError("Incompatible cache index schema")
        self.identity = self.index["identity"]
        self.target_fps = config["data"]["target_fps"] if config is not None else None
        import threading
        self.reader_capacity = config["cache"]["reader_lru_samples"] if config is not None else 0
        self._loaded, self._reader_lock, self._sample_locks = OrderedDict(), threading.RLock(), {}
        records = read_manifest(manifest)
        if not records or any(r.split != split for r in records):
            raise ValueError(f"Manifest must contain only nonempty declared {split} records")
        if len({r.sample_id for r in records}) != len(records):
            raise ValueError("Duplicate sample IDs in corpus manifest")
        self.records = {r.sample_id: r for r in records}
        self.available = tuple(r.sample_id for r in records if r.availability == "available")
        self.available_set = frozenset(self.available)
        if not self.available:
            raise ValueError("Manifest has no actually available samples")
        for name in self.available:
            record, entry = self.records[name], self.index["samples"].get(name)
            if entry is None:
                raise ValueError(f"Available sample has no completed cache: {name}")
            if (entry["source_sha256"], entry["start_s"], entry["end_s"], entry["split"]) != (record.source_sha256, record.start_s, record.end_s, split):
                raise ValueError(f"Indexed cache belongs to a different source/interval/split: {name}")
            directory = (self.root / entry["path"]).resolve()
            if not directory.is_relative_to(self.root) or file_hash(directory / "meta.json") != entry["meta_sha256"]:
                raise ValueError(f"Cache path or metadata integrity failure: {name}")
        if config is not None:
            c = config["cache"]
            if c["kind"] != "per_frame_dino" or c["visual_dim"] != self.identity["feature_dim"] or c["visual_streams"] != self.identity["visual_stream_order"]:
                raise ValueError("Config/cache feature schema mismatch")
            if self.identity["detector_mode"] != c["detector_mode"] or self.identity["temporal_dependency_policy"] != c["temporal_dependency_policy"]:
                raise ValueError("Config/cache extraction dependency policy mismatch")
        self.fingerprint = canonical_hash({"manifest_sha256": file_hash(self.manifest), "identity": self.identity,
            "samples": {name: self.index["samples"][name] for name in self.available}})

    def load(self, sample_id):
        if sample_id not in self.available_set:
            raise ValueError("Requested sample is outside available manifest membership")
        record, entry = self.records[sample_id], self.index["samples"][sample_id]
        directory = self.root / entry["path"]
        with self._reader_lock:
            import threading
            lock = self._sample_locks.setdefault(sample_id, threading.Lock())
        with lock:
            files = [directory / "meta.json", directory / "raw_supports.npz"] + sorted(directory.glob("*.npy"))
            if (directory / "features.npz").exists():
                files.append(directory / "features.npz")
            def versions():
                return tuple((str(p.name), s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for p in files for s in [p.stat()])
            version = versions()
            with self._reader_lock:
                cached = self._loaded.get(sample_id)
                if cached is not None and cached[0] == version:
                    self._loaded.move_to_end(sample_id)
                    return replace(cached[1], meta=deepcopy(cached[1].meta))
            if file_hash(directory / "meta.json") != entry["meta_sha256"]:
                raise ValueError("Cache metadata changed since index publication")
            bundle = read_bundle(directory, self.identity, allow_fixture=self.allow_fixture)
            if version != versions():
                raise ValueError("Cache files changed during content validation")
        if self.target_fps is not None and bundle.meta.get("selection_target_fps") != self.target_fps:
            raise ValueError("Cache sampling FPS differs from the registered recipe")
        if bundle.meta["source_sha256"] != record.source_sha256 or bundle.meta["sample_id"] != sample_id:
            raise ValueError("Loaded cache sample/source differs from manifest")
        if self.reader_capacity:
            with self._reader_lock:
                self._loaded[sample_id] = (version, bundle)
                while len(self._loaded) > self.reader_capacity:
                    self._loaded.popitem(last=False)
        return replace(bundle, meta=deepcopy(bundle.meta))

    def pretrain_sources(self):
        sources = {}
        for name in self.available:
            view = self.records[name].pretrain_view()
            sources.setdefault(view.source_group_id, []).append(view.sample_id)
        return sources

    def pretrain_loader(self, workers=0, persistent=True):
        """Return a label-free callable, rather than exposing annotations to the provider."""
        allowed = frozenset(r.pretrain_view().sample_id for r in self.records.values() if r.sample_id in self.available)
        def load(name):
            if name not in allowed:
                raise ValueError("Sample outside registered pretraining view")
            return self.load(name)
        if workers:
            from concurrent.futures import ThreadPoolExecutor
            if persistent:
                pool = ThreadPoolExecutor(max_workers=workers)
                load.prefetch = lambda names: list(pool.map(load, names))
                load.close = pool.shutdown
            else:
                def prefetch(names):
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        return list(pool.map(load, names))
                load.prefetch = prefetch
        return load
