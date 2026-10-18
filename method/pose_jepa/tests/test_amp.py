from copy import deepcopy
import torch
import pytest
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.distributed import initialize_distributed


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_fp16_scaler_success_overflow_and_teacher_fp32(config, batch):
    cfg = deepcopy(config); cfg['optim']['precision'] = 'fp16'
    trainer = SSLTrainer(cfg, 'cuda')
    # GradScaler may need to adapt its initial 65536 scale before a first update.
    for _ in range(16):
        report = trainer.update([batch])
        if report['successful']: break
        assert report['reason'] == 'nonfinite_gradients' and trainer.step == 0
    assert report['successful']
    assert all(p.dtype == torch.float32 for p in trainer.teacher.parameters())
    before, center, scale = deepcopy(trainer.teacher.state_dict()), trainer.criterion.center.clone(), trainer.scaler.get_scale()
    hook = trainer.system.predictor.latent.weight.register_hook(lambda g: g*float('inf'))
    report = trainer.update([batch]); hook.remove()
    assert report['reason'] == 'nonfinite_gradients' and trainer.step == 1
    assert trainer.scaler.get_scale() == scale/2
    for key, value in before.items(): torch.testing.assert_close(value, trainer.teacher.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(center, trainer.criterion.center, atol=0, rtol=0)


def test_cpu_precision_contract(config, batch):
    cfg = deepcopy(config); cfg['optim']['precision'] = 'bf16'
    with pytest.raises(ValueError, match='CPU uses fp32'):
        SSLTrainer(cfg)
    cfg['optim']['precision'] = 'bf16_if_supported_else_fp32'
    trainer = SSLTrainer(cfg)
    report = trainer.update([batch])
    assert report['successful']
    assert trainer.criterion.center.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason='bf16 CUDA unavailable')
def test_bf16_cuda_losses_and_stats_remain_float32(config, batch):
    cfg = deepcopy(config); cfg['optim']['precision'] = 'bf16'
    trainer = SSLTrainer(cfg, 'cuda')
    report = trainer.update([batch])
    assert report['successful']
    assert trainer.criterion.center.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cli_cuda_device_without_index_resolves_local_device():
    assert initialize_distributed('cuda').index is not None
