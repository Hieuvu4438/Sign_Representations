from copy import deepcopy
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from signrepr.training.pretrain import SSLTrainer
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.collate import collate_pose
from signrepr.models.part_tokenizer import token_metadata


def masks_for(batch, count):
    valid = token_metadata(batch)['valid']
    target = torch.zeros_like(valid)
    target[:, 1:count+1] = valid[:, 1:count+1]
    return {'target': target, 'visible': valid & ~target}


def ddp_worker(rank, rendezvous, output, cfg):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    trainer = SSLTrainer(cfg)
    variant = 'clean' if rank == 0 else 'all_missing'
    batches = [collate_pose([canonicalize_sample(fixture_pose(n, seed=s, variant=variant))]) for n, s in [(5, 1), (16, 2)]]
    masks = [masks_for(batch, count) for batch, count in zip(batches, [1, 8])]
    report = trainer.update(batches, masks)
    torch.save({'gradients': [p.grad for p in trainer.system.parameters()], 'center': trainer.criterion.center,
                'report': report}, Path(output)/f'rank{rank}.pt')
    dist.destroy_process_group()


def test_actual_gloo_ddp_unequal_counts_and_zero_support_rank(config, tmp_path):
    cfg = deepcopy(config)
    cfg['optim']['gradient_clip_norm'] = 10000
    mp.spawn(ddp_worker, args=(str(tmp_path/'rendezvous'), str(tmp_path), cfg), nprocs=2, join=True)
    trainer = SSLTrainer(cfg)
    samples = [canonicalize_sample(fixture_pose(n, seed=s, variant=v)) for v in ['clean', 'all_missing'] for n, s in [(5, 1), (16, 2)]]
    batch = collate_pose(samples)
    valid = token_metadata(batch)['valid']
    target = torch.zeros_like(valid)
    target[0, 1:2] = True
    target[1, 1:9] = valid[1, 1:9]
    trainer.update([batch], [{'target': target, 'visible': valid & ~target}])
    for rank in range(2):
        state = torch.load(tmp_path/f'rank{rank}.pt', weights_only=True)
        for grad, param in zip(state['gradients'], trainer.system.parameters()):
            torch.testing.assert_close(grad, param.grad, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(state['center'], trainer.criterion.center, atol=1e-6, rtol=1e-5)
        assert state['report']['successful']
