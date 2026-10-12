from dataclasses import replace
import numpy as np
import pytest
import torch

from signpath.cache.index import CachedCorpus
from signpath.data.fixtures import synthetic_features
from signpath.data.manifest import ManifestRecord, write_manifest
from signpath.evaluation.islr import fit_probe
from signpath.evaluation.retrieval import retrieval_metrics
from signpath.inference.extract import temporal_embedding, window_starts
from signpath.models.teacher_student import ResearchModel
from test_corpus_targets import add_fixture
from model_helpers import tiny_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_retrieval_stable_ties_and_source_exclusion():
    q, g = [[1., 0], [0, 1]], [[1., 0], [1., 0], [0, 1]]
    metrics = retrieval_metrics(q, g, [10, 20], [99, 10, 20], ["same", "query2"], ["same", "gallery2", "gallery3"])
    assert metrics["recall_at_1"] == metrics["mrr"] == 1
    assert metrics["same_source_exclusions"] == 1
    ties = retrieval_metrics([[1., 0]], [[1., 0], [1., 0]], [10], [99, 10], ["q"], ["a", "b"])
    assert ties["mrr"] == .5 and ties["recall_at_1"] == 0


def test_long_clip_tiling_covers_tail_and_matches_full_when_one_window():
    assert window_starts(100, 48, 16) == [0, 32, 52]
    assert window_starts(20, 48, 16) == [0]
    config = tiny_config()
    model = ResearchModel(config).eval()
    bundle = synthetic_features("long", length=100, visual_dim=8)
    with torch.no_grad():
        result = temporal_embedding(model, bundle, config, window_samples=48, overlap_samples=16)
    assert result.dense.shape == (100, 24) and result.valid.all()
    np.testing.assert_array_equal(result.timestamps.numpy(), bundle.timestamps)


@pytest.mark.parametrize("head", ["linear", "attentive", "finetune"])
def test_train_val_probe_has_functioning_heads(head, tmp_path):
    config = tiny_config()
    config["downstream"].update(head=head, max_epochs=2, patience=2, learning_rates=[.01], batch_size=2, attentive_heads=4)
    root = tmp_path / "cache"
    manifests = {}
    for split in ["train", "val"]:
        records = []
        for label in range(2):
            name = f"{split}{label}"
            bundle = synthetic_features(name, length=20, visual_dim=8, seed=label, phase=label * 2)
            record = ManifestRecord(1, name, "fixture", "ase", split, name, "unused.mp4", availability="available", label_id=label)
            add_fixture(root, bundle, record)
            records.append(record)
        path = tmp_path / f"{split}.jsonl"
        write_manifest(path, records)
        manifests[split] = CachedCorpus(path, root, split=split, allow_fixture=True)
    model = ResearchModel(config)
    initial = {name: value.clone() for name, value in model.online.state_dict().items()}
    result = fit_probe(manifests["train"], manifests["val"], config, model, config)
    assert result.logits.shape == (2, 2) and np.isfinite(result.logits).all()
    assert result.class_ids == (0, 1)
    assert result.trials[0]["epochs_attempted"] == 2
    assert (result.model_state is not None) == (head == "finetune")
    changed = any(not torch.equal(value, initial[name]) for name, value in model.online.state_dict().items())
    assert changed == (head == "finetune")


def test_paired_source_bootstrap_constant_difference():
    from signpath.evaluation.bootstrap import paired_source_bootstrap
    result = paired_source_bootstrap([0.] * 8, [.25] * 8, ["a"] * 4 + ["b"] * 4, samples=50)
    assert result["mean_difference"] == .25 and result["ci95"] == [.25, .25]
    assert result["resampling_unit"] == "source_group"


def test_retrieval_does_not_modify_caller_arrays():
    query, gallery = np.array([[2., 0.]]), np.array([[3., 0.]])
    retrieval_metrics(query, gallery, [1], [1], ["query"], ["gallery"])
    np.testing.assert_array_equal(query, [[2., 0.]])
    np.testing.assert_array_equal(gallery, [[3., 0.]])
