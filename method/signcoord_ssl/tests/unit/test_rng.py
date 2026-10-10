import torch

from signcoord.engine.checkpoint import rng_state,restore_rng
from signcoord.engine.distributed import TrainingContext


def test_cpu_checkpoint_rng_never_initializes_or_queries_cuda(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('CPU checkpoint must not query any CUDA RNG/device')
    for name in ('is_available','get_rng_state_all','get_rng_state'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    snapshot=rng_state('cpu')
    assert snapshot['cuda'] is None and snapshot['cuda_device_rng'] is None
    expected=torch.rand(4)
    restore_rng(snapshot)
    assert torch.equal(torch.rand(4),expected)


def test_default_cuda_device_is_resolved_before_set_device_without_allocating(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','1')
    seen=[]
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:seen.append(torch.cuda._utils._get_device_index(device)))
    context=TrainingContext('cuda')
    assert context.device==torch.device('cuda:0') and seen==[0]
    context.close()
