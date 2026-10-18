"""Execute whole Stage 2 DDP updates, including the warmup-to-replay transition."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.collate import collate_pose
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.metric import MetricTrainer
from signrepr.training.checkpoints import save_training


def worker(rank, rendezvous, output, config, initial):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=40))
    vocab = [('synthetic', 'synthetic', '0')]
    trainer = MetricTrainer(config, initial, vocab)
    batches = []
    for micro in range(2):
        variant = 'all_missing' if rank == 1 and micro == 0 else 'clean'
        batch = collate_pose([canonicalize_sample(fixture_pose(12+micro*2, seed=10+rank*3+micro, variant=variant))])
        batch['labels'] = [vocab[0]]
        batches.append(batch)
    before = deepcopy(trainer.system.encoder.state_dict())
    warmup = trainer.update(batches)
    assert warmup['successful'] and warmup['phase'] == 'warmup'
    for key, value in before.items():
        torch.testing.assert_close(value, trainer.system.encoder.state_dict()[key], atol=0, rtol=0)
    joint = trainer.update(batches)
    assert joint['successful'] and joint['phase'] == 'joint' and joint['ssl_replay']
    torch.save({'system': trainer.system.state_dict(), 'teacher': trainer.teacher.state_dict(),
                'center': trainer.criterion.center, 'step': trainer.step}, Path(output)/f'rank{rank}.pt')
    dist.destroy_process_group()


def test_stage2_actual_ddp_transition_and_zero_support_microbatch(config, tmp_path):
    ssl = SSLTrainer(config)
    initial = tmp_path/'ssl.pt'
    save_training(ssl, initial)
    cfg = deepcopy(config)
    cfg['metric'] = {'head_warmup_steps': 1, 'joint_steps': 2, 'head_lr': 1e-4,
                     'encoder_lr': 2e-5, 'weight_decay': .01, 'ssl_replay': True,
                     'temperature': .07, 'lr_warmup_steps': 1}
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'), str(tmp_path), cfg, str(initial)), nprocs=2, join=True)
    a, b = [torch.load(tmp_path/f'rank{i}.pt', weights_only=True) for i in range(2)]
    assert a['step'] == b['step'] == 2
    for component in ('system', 'teacher'):
        for name, value in a[component].items():
            assert torch.isfinite(value).all()
            torch.testing.assert_close(value, b[component][name], atol=1e-6, rtol=1e-5)
    assert a['center'].abs().sum() > 0
    torch.testing.assert_close(a['center'], b['center'], atol=0, rtol=0)
