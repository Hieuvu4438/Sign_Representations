from dataclasses import replace
import json
import numpy as np
import pytest

from signpath.cache.hashing import file_hash
from signpath.cache.index import CachedCorpus, record_cache, sample_directory
from signpath.cache.writer import write_bundle
from signpath.data.fixtures import synthetic_features
from signpath.data.manifest import ManifestRecord, write_manifest
from signpath.targets.kmeans import KMeansTargets
from signpath.targets.projection import TargetProjectors
from model_helpers import example, tiny_config


def add_fixture(root, bundle, record):
    directory = sample_directory(root, bundle.meta["extractor_hash"], record.sample_id)
    write_bundle(directory, bundle, allow_fixture=True)
    record_cache(root, directory, record, allow_fixture=True)
    return directory


def test_cache_index_membership_interval_identity_and_fixture_guard(tmp_path):
    bundle = synthetic_features("clip", visual_dim=8)
    record = ManifestRecord(1, "clip", "fixture", "ase", "train", "source", "unused.mp4", availability="available", label_id=0)
    root, manifest = tmp_path / "cache", tmp_path / "train.jsonl"
    directory = add_fixture(root, bundle, record)
    write_manifest(manifest, [record])
    with pytest.raises(ValueError, match="fixture"):
        CachedCorpus(manifest, root, split="train")
    corpus = CachedCorpus(manifest, root, split="train", allow_fixture=True)
    assert corpus.pretrain_sources() == {"source": ["clip"]}
    assert corpus.pretrain_loader()("clip").meta["sample_id"] == "clip"
    write_manifest(manifest, [replace(record, start_s=.1)])
    with pytest.raises(ValueError, match="interval"):
        CachedCorpus(manifest, root, split="train", allow_fixture=True)


def test_index_metadata_mutation_rejected(tmp_path):
    bundle = synthetic_features("clip", visual_dim=8)
    record = ManifestRecord(1, "clip", "fixture", "ase", "train", "source", "unused.mp4", availability="available")
    root, manifest = tmp_path / "cache", tmp_path / "train.jsonl"
    directory = add_fixture(root, bundle, record)
    write_manifest(manifest, [record])
    (directory / "meta.json").write_text((directory / "meta.json").read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        CachedCorpus(manifest, root, split="train", allow_fixture=True)


def test_train_only_kmeans_masks_missing_and_roundtrips(tmp_path):
    bundles = [synthetic_features(f"clip{i}", visual_dim=8, seed=i, phase=i) for i in range(3)]
    records = [ManifestRecord(1, f"clip{i}", "fixture", "ase", "train", f"source{i}", "unused.mp4") for i in range(3)]
    kwargs = dict(manifest_sha256="a" * 64, extractor_hash=bundles[0].meta["extractor_hash"], clusters=4)
    model = KMeansTargets.fit(zip(records, bundles), **kwargs)
    ids = model(bundles[0])
    assert ids.shape == (48, 7) and ids.dtype == np.int64
    assert (ids[6:9, 5] == -100).all()
    model.save(tmp_path / "codebooks")
    reloaded = KMeansTargets.load(tmp_path / "codebooks", **kwargs)
    np.testing.assert_array_equal(ids, reloaded(bundles[0]))
    with pytest.raises(ValueError, match="held-out"):
        KMeansTargets.fit([(replace(records[0], split="val"), bundles[0])], **kwargs)
    with pytest.raises(ValueError, match="cannot fit"):
        KMeansTargets.fit([(records[0], bundles[0])], **{**kwargs, "clusters": 256})


def test_projection_separate_statistics_matrix_contract(tmp_path, bundle):
    original = example(bundle).projectors
    stats, matrices = tmp_path / "stats.npz", tmp_path / "matrices.npz"
    original.save_files(stats, matrices)
    reloaded = TargetProjectors.load_files(stats, matrices, expected_manifest_sha256="a" * 64,
        expected_extractor_hash=bundle.meta["extractor_hash"], expected_descriptor_config={"fixture": True})
    assert reloaded.fingerprint() == original.fingerprint()
    with matrices.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        TargetProjectors.load_files(stats, matrices, expected_manifest_sha256="a" * 64,
            expected_extractor_hash=bundle.meta["extractor_hash"], expected_descriptor_config={"fixture": True})


def test_bounded_reader_cache_reuses_verified_data_but_detects_mutation(tmp_path):
    from unittest.mock import patch
    bundle = synthetic_features("clip", visual_dim=8)
    bundle = replace(bundle, meta={**bundle.meta, "selection_target_fps": 25.})
    record = ManifestRecord(1, "clip", "fixture", "ase", "train", "source", "unused.mp4", availability="available")
    root, manifest = tmp_path / "cache", tmp_path / "train.jsonl"
    directory = add_fixture(root, bundle, record)
    write_manifest(manifest, [record])
    corpus = CachedCorpus(manifest, root, split="train", allow_fixture=True)
    corpus.reader_capacity = 1
    first = corpus.load("clip")
    first.meta["sample_id"] = "caller_edit"
    with patch("signpath.cache.index.read_bundle", side_effect=AssertionError("Should reuse verified immutable arrays")):
        second = corpus.load("clip")
        assert second.meta["sample_id"] == "clip"
        assert second.visual is first.visual
    with (directory / "visual.npy").open("ab") as stream:
        stream.write(b"mutated")
    with pytest.raises(ValueError, match="checksum"):
        corpus.load("clip")
