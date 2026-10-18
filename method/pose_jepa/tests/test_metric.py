from copy import deepcopy
import numpy as np
import torch
from signrepr.losses.supcon import supcon_terms
from signrepr.data.samplers import PKSampler
from signrepr.data.fixture import make_fixture
from signrepr.data.schema import read_manifest
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.checkpoints import save_training, load_training
from signrepr.training.metric import MetricTrainer
from signrepr.export.api import export_checkpoint, SignRepresentationModel


def test_supcon_exact_mathematical_reference_invalid_rows():
    x = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [.6, .8], [0., 0.]], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0])
    valid = torch.tensor([True, True, True, True, False])
    numerator, count = supcon_terms(x, labels, valid)
    expected = 0
    for i in range(4):
        keys = [j for j in range(4) if i != j]
        positive = i^1
        logits = torch.stack([x[i]@x[j]/.07 for j in keys])
        expected += -(x[i]@x[positive]/.07-torch.logsumexp(logits, 0))
    torch.testing.assert_close(numerator, expected)
    assert count == 4
    numerator.backward()
    assert torch.isfinite(x.grad).all() and x.grad[-1].eq(0).all()


def test_pk_distinct_clips_namespaces_and_sharding(tmp_path):
    path = make_fixture(tmp_path, 16)
    rows = read_manifest(path)
    global_sampler = PKSampler(rows, p=4, k=2)
    rank0, rank1 = PKSampler(rows, p=4, k=2, rank=0, world_size=2), PKSampler(rows, p=4, k=2, rank=1, world_size=2)
    whole, a, b = global_sampler.next(), rank0.next(), rank1.next()
    assert whole[::2] == a and whole[1::2] == b
    assert len(set(whole)) == 8 and not set(a)&set(b)
    for i in range(0, 8, 2):
        assert rows[whole[i]]['gloss_id'] == rows[whole[i+1]]['gloss_id']
        assert rows[whole[i]]['signer_id'] != rows[whole[i+1]]['signer_id']
    state = rank0.state_dict(); expected = rank0.next(); rank0.load_state_dict(state)
    assert rank0.next() == expected


def test_metric_warmup_freeze_replay_and_matched_online_export(config, batch, tmp_path):
    ssl = SSLTrainer(config); ssl.update([batch])
    save_training(ssl, tmp_path/'ssl.pt')
    cfg = deepcopy(config)
    cfg['run'].update(stage='metric', supervision='gloss_supervised_metric_adaptation')
    cfg['metric'] = {'head_warmup_steps': 1, 'joint_steps': 2, 'head_lr': 1e-4, 'encoder_lr': 2e-5,
                     'weight_decay': .01, 'ssl_replay': True, 'temperature': .07, 'lr_warmup_steps': 1}
    vocabulary = [('synthetic', 'synthetic', '0')]
    batch = deepcopy(batch); batch['labels'] = [vocabulary[0]]*2
    trainer = MetricTrainer(cfg, tmp_path/'ssl.pt', vocabulary)
    before = deepcopy(trainer.system.encoder.state_dict())
    center = trainer.criterion.center.clone()
    first = trainer.update([batch])
    assert first['phase'] == 'warmup' and not first['ssl_replay']
    for name, value in trainer.system.encoder.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    torch.testing.assert_close(trainer.criterion.center, center, atol=0, rtol=0)
    second = trainer.update([batch])
    assert second['phase'] == 'joint' and second['ssl_replay']
    assert not torch.equal(center, trainer.criterion.center)
    save_training(trainer, tmp_path/'metric.pt', metadata={'stage': 'metric'})
    info = export_checkpoint(tmp_path/'metric.pt', tmp_path/'bundle')
    assert info['selected_encoder'] == 'online' and info['metric_head_trained']
    model = SignRepresentationModel(tmp_path/'bundle')
    assert model.metric_head is not None
    uninterrupted = trainer.update([batch])
    restored = MetricTrainer(cfg, tmp_path/'ssl.pt', vocabulary)
    load_training(restored, tmp_path/'metric.pt')
    assert restored.update([batch]) == uninterrupted
    for name, value in trainer.system.state_dict().items():
        torch.testing.assert_close(value, restored.system.state_dict()[name], atol=0, rtol=0)
    no_replay_cfg = deepcopy(cfg); no_replay_cfg['metric'].update(head_warmup_steps=0, ssl_replay=False)
    no_replay = MetricTrainer(no_replay_cfg, tmp_path/'ssl.pt', vocabulary)
    before = deepcopy(no_replay.system.predictor.state_dict())
    teacher_before = deepcopy(no_replay.teacher.state_dict())
    center_before = no_replay.criterion.center.clone()
    assert no_replay.update([batch])['successful']
    for name, value in before.items():
        torch.testing.assert_close(value, no_replay.system.predictor.state_dict()[name], atol=0, rtol=0)
    for name, value in teacher_before.items():
        torch.testing.assert_close(value, no_replay.teacher.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(center_before, no_replay.criterion.center, atol=0, rtol=0)
