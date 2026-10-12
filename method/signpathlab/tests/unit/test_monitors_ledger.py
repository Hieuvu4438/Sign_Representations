import json
from types import SimpleNamespace
import torch
from signpath.data.fixtures import synthetic_features
from signpath.models.teacher_student import ResearchModel
from signpath.training.ledger import stage_record
from signpath.training.monitor import training_diagnostics
from model_helpers import tiny_config


def test_stage_ledger_retains_both_original_and_reuse_records(tmp_path):
    first = stage_record(tmp_path, "pose_roi", "train", {"status": "completed"}, elapsed_seconds=2.)
    second = stage_record(tmp_path, "pose_roi", "train", {"status": "verified_reuse"}, elapsed_seconds=.1)
    assert first != second and first.exists() and second.exists()
    assert json.loads(first.read_text())["gpu_hours"] is None


def test_monitor_uses_independent_sources_and_restores_mode(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tiny_config()
        model = ResearchModel(config).train()
        bundles = {name: synthetic_features(name, length=20, visual_dim=8, seed=i) for i, name in enumerate(["a", "b", "c"])}
        corpus = SimpleNamespace(pretrain_sources=lambda: {name: [name] for name in bundles}, load=lambda name: bundles[name])
        report = training_diagnostics(model, corpus, config, tmp_path, 1, "cpu")
        assert report["online"]["clip_count"] == 3
        assert len(report["sources"]) == 3 and model.training and not model.teacher.training
        assert (tmp_path / "feature_spectrum.npz").is_file()
        assert (tmp_path / "diagnostics/step000000001/report.json").is_file()
    finally:
        torch.set_num_threads(previous)
