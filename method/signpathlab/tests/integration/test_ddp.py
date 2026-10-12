from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
import torch

from signpath.data.collate import collate_examples
from signpath.models.teacher_student import ResearchModel
from signpath.training.engine import Trainer
from model_helpers import example, tiny_config


def test_two_gloo_ranks_match_global_reference_with_empty_local_rank(tmp_path, bundle):
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(31)
        config = tiny_config()
        e = example(bundle)
        batch = collate_examples([e, e], config["model"]["raw_dimensions"])
        visible, targets, geo = batch.visible.clone(), batch.semantic_targets.clone(), batch.geometry_targets.clone()
        visible[0], targets[0], geo[0] = False, False, False
        batch = replace(batch, visible=visible, semantic_targets=targets, geometry_targets=geo)
        trainer = Trainer(ResearchModel(config), config)
        path = tmp_path / "input.pt"
        torch.save({"config": config, "batch": batch, "initial": trainer.core.state_dict()}, path)
        # The empty example contributes no target. Evaluate the identical contributing
        # shape on the reference to avoid Adam amplifying batched-attention roundoff
        # in mathematically zero bias gradients (~1e-10).
        expected = trainer.update([batch.select([1])])
        assert expected.succeeded
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(Path(__file__).with_name("ddp_worker.py")), "--input", str(path), "--output", str(tmp_path)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
        assert process.returncode == 0, process.stdout + process.stderr
        first = torch.load(tmp_path / "rank0.pt", weights_only=False)
        second = torch.load(tmp_path / "rank1.pt", weights_only=False)
        assert first["success"] and second["success"] and first["step"] == second["step"] == 1
        for name, parameter in trainer.core.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, first["gradients"][name], atol=2e-6, rtol=2e-4)
        for name, value in trainer.core.state_dict().items():
            torch.testing.assert_close(first["state"][name], second["state"][name], atol=0, rtol=0)
            torch.testing.assert_close(first["state"][name], value, atol=3e-5, rtol=2e-4)
    finally:
        torch.set_num_threads(prior)
