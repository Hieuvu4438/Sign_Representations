import torch
import pytest
from signpath.inference.deployment import OfflineEncoderGraph
from signpath.inference.export import export_graph, verify_parity
from signpath.models.heads import LinearHead, AttentiveHead
from signpath.models.teacher_student import ResearchModel
from model_helpers import tiny_config


@pytest.mark.parametrize("kind", ["linear", "attentive", "frozen_visual"])
def test_dynamic_export_save_load_and_parity(tmp_path, kind):
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tiny_config()
        if kind == "frozen_visual":
            graph = OfflineEncoderGraph(None, LinearHead(8, 3), frozen_visual=True)
        else:
            head = LinearHead(24, 3) if kind == "linear" else AttentiveHead(24, 3, heads=4)
            graph = OfflineEncoderGraph(ResearchModel(config).online, head, head_kind=kind)
        program = export_graph(graph, config["model"]["raw_dimensions"])
        path = tmp_path / "encoder.pt2"
        torch.export.save(program, path)
        loaded = torch.export.load(path).module()
        cases = verify_parity(graph, loaded, config["model"]["raw_dimensions"], lengths=(1, 3, 17))
        assert len(cases) == 7 and all(c.get("max_absolute_error", 0) < 2e-5 for c in cases)
    finally:
        torch.set_num_threads(prior)
