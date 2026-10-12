from dataclasses import replace
import pytest
import torch
from signpath.models.teacher_student import ResearchModel
from signpath.training.engine import Trainer
from model_helpers import batch_from, tiny_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA AMP validation requires a GPU")
def test_actual_fp16_scaler_overflow_preserves_online_teacher_and_schedule(bundle):
    config = tiny_config()
    config["training"]["precision"] = "fp16"
    trainer = Trainer(ResearchModel(config), config, "cuda:0")
    batch = batch_from(bundle)
    first = trainer.update([batch])
    assert first.succeeded
    before = {name: value.clone() for name, value in trainer.core.state_dict().items()}
    scale = trainer.scaler.get_scale()
    hook = trainer.core.online.projectors[0][0].weight.register_hook(lambda gradient: gradient * float("inf"))
    result = trainer.update([batch])
    hook.remove()
    assert not result.succeeded and result.reason == "nonfinite_loss_or_gradient"
    assert trainer.successful_step == trainer.scheduler.successful_steps == 1
    assert trainer.scaler.get_scale() == scale * trainer.scaler.get_backoff_factor()
    for name, value in trainer.core.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    # The same scaler can continue after a skipped attempt without a double step.
    assert trainer.update([batch]).succeeded
    assert trainer.successful_step == trainer.scheduler.successful_steps == 2
