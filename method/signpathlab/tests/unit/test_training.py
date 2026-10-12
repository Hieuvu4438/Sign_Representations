from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import pytest
import torch

from signpath.models.teacher_student import ResearchModel
from signpath.training.checkpoint import load_checkpoint, save_checkpoint
from signpath.training.engine import Trainer
from model_helpers import batch_from, tiny_config


@pytest.fixture(autouse=True)
def fixed_cpu_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(previous)


def clone_parameters(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def test_fixed_batch_learning_and_teacher_no_grad(bundle):
    config = tiny_config(steps=12)
    trainer = Trainer(ResearchModel(config), config)
    batch = batch_from(bundle)
    losses = [trainer.update([batch]).loss for _ in range(12)]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    assert trainer.successful_step == trainer.scheduler.successful_steps == 12
    assert all(p.grad is None for p in trainer.core.teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.core.online.parameters())


@pytest.mark.parametrize("cause", ["nonfinite", "empty"])
def test_failed_update_preserves_weights_ema_schedule_and_optimizer(bundle, cause):
    config = tiny_config()
    trainer = Trainer(ResearchModel(config), config)
    batch = batch_from(bundle)
    trainer.update([batch])
    before = clone_parameters(trainer.core)
    state_steps = [v["step"].clone() for v in trainer.optimizer.state.values()]
    if cause == "nonfinite":
        hook = trainer.core.online.projectors[0][0].weight.register_hook(lambda gradient: gradient * float("inf"))
    else:
        batch = replace(batch, semantic_targets=torch.zeros_like(batch.semantic_targets))
    result = trainer.update([batch])
    if cause == "nonfinite":
        hook.remove()
    assert not result.succeeded and trainer.successful_step == 1 and trainer.scheduler.successful_steps == 1
    assert trainer.attempted_batches == 2
    for key, value in trainer.core.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    for previous, value in zip(state_steps, trainer.optimizer.state.values()):
        torch.testing.assert_close(previous, value["step"], atol=0, rtol=0)


def test_resume_exact_ten_steps_including_rng_optimizer_teacher(tmp_path, bundle):
    config = tiny_config(steps=10)
    batch = batch_from(bundle)
    trainer = Trainer(ResearchModel(config), config)
    generator = np.random.default_rng(19)
    identity = {"manifest": "a" * 64, "protocol": "fixed_train_diagnostic"}
    for _ in range(5):
        trainer.update([batch])
    path = tmp_path / "last.pt"
    save_checkpoint(path, trainer, config, identity, mask_generator=generator, data_state={"next_batch": 5})
    expected_rng = (random.random(), np.random.rand(), torch.rand(2), generator.integers(0, 100, size=5))
    for _ in range(5):
        trainer.update([batch])
    expected = clone_parameters(trainer.core)
    resumed = Trainer(ResearchModel(config), config)
    actual_state = load_checkpoint(path, resumed, config, identity, mask_generator=generator)
    assert actual_state == {"next_batch": 5}
    actual_rng = (random.random(), np.random.rand(), torch.rand(2), generator.integers(0, 100, size=5))
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], atol=0, rtol=0)
    np.testing.assert_array_equal(expected_rng[3], actual_rng[3])
    for _ in range(5):
        resumed.update([batch])
    for key, value in resumed.core.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert resumed.scheduler.state_dict() == trainer.scheduler.state_dict()
    for first, second in zip(resumed.optimizer.state.values(), trainer.optimizer.state.values()):
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    bad = deepcopy(config)
    bad["experiment"]["evaluation_protocol"] = "changed_protocol"
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(path, resumed, bad, identity, mask_generator=generator)


def test_accumulation_matches_duplicate_microbatch_average(bundle):
    batch = batch_from(bundle)
    config = tiny_config()
    single = Trainer(ResearchModel(config), config)
    config2 = deepcopy(config)
    config2["training"]["gradient_accumulation"] = 2
    accumulated = Trainer(ResearchModel(config2), config2)
    accumulated.core.load_state_dict(single.core.state_dict())
    single.update([batch])
    accumulated.update([batch, batch])
    for name, value in single.core.state_dict().items():
        torch.testing.assert_close(value, accumulated.core.state_dict()[name], atol=1e-6, rtol=1e-5)
