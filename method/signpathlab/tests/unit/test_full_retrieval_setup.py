from dataclasses import replace
import json
import numpy as np
import pytest
from signpath.cache.schema import IDENTITY_FIELDS, ARRAY_DTYPES
from signpath.cache.writer import write_bundle
from signpath.cache.reader import read_bundle
from signpath.evaluation.dictionary import dictionary_metrics
from signpath.targets.projection import ProjectionObservation, TargetProjectors
from signpath.data.sampling import SourceSegmentSampler


def test_lossless_compressed_features_and_checksum_guard(tmp_path, bundle):
    output = tmp_path / "compressed"
    write_bundle(output, bundle, allow_fixture=True, compressed=True)
    expected = {key: bundle.meta[key] for key in IDENTITY_FIELDS}
    restored = read_bundle(output, expected, allow_fixture=True)
    for name in ARRAY_DTYPES:
        np.testing.assert_array_equal(getattr(restored, name), getattr(bundle, name))
    assert restored.raw_supports == bundle.raw_supports
    with (output / "features.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(output, expected, allow_fixture=True)


def test_streaming_projection_matches_materialized_population_statistics():
    vectors = np.random.default_rng(91).normal(size=(71, 12)) + np.arange(12) * 100
    rows = [ProjectionObservation("train_source", "train", "path/0", value) for value in vectors]
    kwargs = dict(allowed_train_sources={"train_source"}, manifest_sha256="a" * 64,
                  extractor_hash="b" * 64, descriptor_config={"test": True})
    full = TargetProjectors.fit(rows, **kwargs).projections["path/0"]
    streamed = TargetProjectors.fit_batches((rows[:13], rows[13:37], rows[37:]), **kwargs).projections["path/0"]
    np.testing.assert_allclose(streamed.apply(vectors), full.apply(vectors), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(streamed.mean, full.mean, atol=1e-10)
    np.testing.assert_allclose(streamed.std, full.std, atol=1e-10)
    with pytest.raises(ValueError, match="training sources"):
        TargetProjectors.fit_batches([[replace(rows[0], split="val")]], **kwargs)


def test_dictionary_ranks_classes_with_max_exemplar_and_pessimistic_ties():
    # Two copies of the wrong gloss are one class, not two ranked candidates.
    gallery = [[1., 0], [1., 0], [0., 1], [-1., 0]]
    report = dictionary_metrics([[1., .1], [0., 0]], gallery, [1, 2], [0, 0, 1, 2], ["q0", "q1"], ["g0", "g1", "g2", "g3"], chunk_size=1)
    assert report["ranks"] == [2, 3] and report["Rec@1"] == 0 and report["Rec@5"] == 1
    assert report["DCG"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / 2)
    assert report["zero_query_embeddings"] == 1
    with pytest.raises(ValueError, match="absent"):
        dictionary_metrics([[1., 0]], gallery, [99], [0, 0, 1, 2], ["q"], ["g0", "g1", "g2", "g3"])


def test_epoch_sampler_covers_every_source_and_replays_boundary():
    sampler = SourceSegmentSampler({str(i): [f"clip{i}"] for i in range(13)}, 17, epoch_shuffle=True)
    first = [sampler.select_sample() for _ in range(13)]
    assert len(set(first)) == 13
    state = sampler.state_dict()
    expected = [sampler.select_sample() for _ in range(17)]
    sampler.load_state_dict(state)
    assert expected == [sampler.select_sample() for _ in range(17)]


def test_chunked_fingerprints_match_brute_force_cross_split_distances():
    from types import SimpleNamespace
    from signpath.data.fingerprints import cross_split_candidates
    records = [SimpleNamespace(sample_id=str(i), split="train" if i < 3 else "val", availability="available") for i in range(6)]
    rng = np.random.default_rng(9)
    fingerprints = {r.sample_id: rng.integers(0, 2, 504, dtype=np.uint8) for r in records}
    fingerprints["4"] = fingerprints["1"].copy()
    report = cross_split_candidates(records, fingerprints=fingerprints)
    assert [(c["first"], c["second"]) for c in report["candidates"]] == [("1", "4")]


def test_dino_batch_recipe_is_versioned_and_original_default_is_preserved(monkeypatch):
    from signpath import preprocessing_full
    class Adapter:
        def load(self, plan, device):
            self._provenance = {"weights": "original"}
            self.calls = []
            return self
        def provenance(self):
            return dict(self._provenance)
        def encode_crops(self, images, *, batch_size=16):
            self.calls.append(batch_size)
            return images
    monkeypatch.setattr(preprocessing_full, "DINOAdapter", Adapter)
    baseline = preprocessing_full.cache_adapter({"dino": {}}, "cpu", 16)
    changed = preprocessing_full.cache_adapter({"dino": {}}, "cpu", 32)
    baseline.encode_crops([]); changed.encode_crops([])
    assert baseline.calls == [16] and changed.calls == [32]
    assert "cache_execution_batch_size" not in baseline.provenance()
    assert changed.provenance()["cache_execution_batch_size"] == 32
