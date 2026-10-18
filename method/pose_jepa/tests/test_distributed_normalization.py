from copy import deepcopy
import torch
from signrepr.training.pretrain import SSLTrainer
from signrepr.models.part_tokenizer import token_metadata
from signrepr.data.collate import collate_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.fixture import fixture_pose


def test_unequal_accumulation_counts_match_effective_batch(config):
    cfg = deepcopy(config)
    cfg['optim']['gradient_clip_norm'] = 10000
    samples = [canonicalize_sample(fixture_pose(5, seed=1)), canonicalize_sample(fixture_pose(16, seed=2, variant='missing'))]
    batches = [collate_pose([s]) for s in samples]
    full = collate_pose(samples)
    def mask(batch, count):
        meta = token_metadata(batch)
        m = torch.zeros_like(meta['valid'])
        m[:, 1:1+count] = meta['valid'][:, 1:1+count]
        return {'target': m, 'visible': meta['valid'] & ~m}
    masks = [mask(batches[0], 1), mask(batches[1], 8)]
    fm = torch.zeros_like(token_metadata(full)['valid'])
    for i, m in enumerate(masks):
        fm[i, :m['target'].shape[1]] = m['target'][0]
    a, b = SSLTrainer(cfg), SSLTrainer(cfg)
    a.update(batches, masks)
    b.update([full], [{'target': fm, 'visible': token_metadata(full)['valid'] & ~fm}])
    for pa, pb in zip(a.system.parameters(), b.system.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(a.criterion.center, b.criterion.center, atol=1e-6, rtol=1e-5)
