from dataclasses import replace
import numpy as np
import pytest
from signpath.targets.projection import ProjectionObservation, TargetProjectors


def inputs():
    rng = np.random.default_rng(14)
    return [ProjectionObservation("source", "train", "left/0", x) for x in rng.normal(size=(40, 12))]


def fit(observations):
    return TargetProjectors.fit(observations, allowed_train_sources={"source"}, manifest_sha256="a" * 64,
                                extractor_hash="b" * 64, descriptor_config={"descriptor": "logsig2", "windows": [.16]})


def test_projection_replay_and_train_output_normalization(tmp_path):
    observations = inputs()
    model, repeat = fit(observations), fit(observations)
    projection = model.projections["left/0"]
    output = projection.apply(np.stack([r.descriptor for r in observations]))
    assert output.shape == (40, 32)
    np.testing.assert_allclose(output.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(output.std(axis=0), 1, atol=1e-6)
    np.testing.assert_array_equal(projection.matrix, repeat.projections["left/0"].matrix)
    path = tmp_path / "targets"
    model.save(path)
    reloaded = TargetProjectors.load(path, expected_manifest_sha256="a" * 64, expected_extractor_hash="b" * 64,
                                    expected_descriptor_config={"descriptor": "logsig2", "windows": [.16]})
    np.testing.assert_array_equal(output, reloaded.projections["left/0"].apply(np.stack([r.descriptor for r in observations])))
    assert model.fingerprint() == reloaded.fingerprint()
    with pytest.raises(ValueError, match="identity"):
        TargetProjectors.load(path, expected_manifest_sha256="c" * 64, expected_extractor_hash="b" * 64, expected_descriptor_config={})


@pytest.mark.parametrize("edit", [lambda r: replace(r, split="val"), lambda r: replace(r, source_group_id="held-out")])
def test_target_fit_rejects_heldout_sources(edit):
    with pytest.raises(ValueError, match="training sources"):
        fit([edit(inputs()[0])])


def test_constant_descriptors_are_finite_and_empty_fit_rejected():
    model = fit([ProjectionObservation("source", "train", "group", np.ones(12))] * 4)
    assert np.isfinite(model.projections["group"].apply(np.ones(12))).all()
    with pytest.raises(ValueError, match="No valid"):
        fit([])
