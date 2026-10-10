import torch

from signcoord.engine.train import paired_keep_mask
from signcoord.models.backbone_adapter import TinyEncoderAdapter


def test_mask_plan_ignores_module_rng_and_rank_partition():
    config = {'seed': 3407, 'model': {'mask_ratio': .8}}
    encoder = TinyEncoderAdapter(32, (32, 32), 16)
    ids = ['video_a', 'video_b', 'video_c', 'video_d']
    whole, seeds = paired_keep_mask(encoder, ids, config, 23, 'cpu')
    # Another arm's decoder initialization/dropout changes global RNG only.
    torch.rand(10000)
    left, left_seeds = paired_keep_mask(encoder, ids[:2], config, 23, 'cpu')
    right, right_seeds = paired_keep_mask(encoder, ids[2:], config, 23, 'cpu')
    assert torch.equal(whole, torch.cat([left, right]))
    assert seeds == left_seeds + right_seeds
    changed, changed_seeds = paired_keep_mask(encoder, ids, config, 24, 'cpu')
    assert seeds != changed_seeds
    assert not torch.equal(whole, changed)
