from copy import deepcopy
import torch
from signrepr.data.fixture import fixture_pose
from signrepr.data.normalization import canonicalize_sample
from signrepr.data.collate import collate_pose
from signrepr.models.pose_encoder import PoseEncoder, pool_clip
from signrepr.models.part_tokenizer import token_metadata
from signrepr.training.system import SSLSystem
from signrepr.losses.hybrid import HybridCriterion


def test_padding_does_not_change_output_or_loss(config):
    sample = canonicalize_sample(fixture_pose(5))
    small = collate_pose([sample])
    padded = collate_pose([sample, canonicalize_sample(fixture_pose(15, seed=4))])
    changed = deepcopy(padded)
    for stream in changed['parts'].values():
        for key in ('xy', 'global_xy', 'anchor', 'confidence'):
            stream[key][0, 5:] = 1234
    changed['position'][0, 5:] = 1234
    system = SSLSystem(config['model']).eval()
    teacher = deepcopy(system.encoder).eval()
    criterion = HybridCriterion(config['loss'], 16)
    with torch.no_grad():
        ref = teacher.encode_full(small)
        for current in (padded, changed):
            out = teacher.encode_full(current)
            torch.testing.assert_close(ref['features'][0], out['features'][0, :12], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(pool_clip(ref)[0][0], pool_clip(out)[0][0], atol=1e-6, rtol=1e-5)
        # Single target on first sample; second sample has no target support.
        def terms(batch):
            meta = token_metadata(batch)
            target = torch.zeros_like(meta['valid']); target[0, 1] = True
            masks = {'target': target, 'visible': meta['valid'] & ~target}
            targets, _ = criterion.targets(batch, masks, meta)
            return criterion(system(batch, masks), teacher.encode_full(batch)['features'], targets, masks, meta)[0]
        a = terms(small)
        for current in (padded, changed):
            b = terms(current)
            for k in a:
                torch.testing.assert_close(a[k][0], b[k][0], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(a[k][1], b[k][1])


def test_all_missing_is_finite_zero(config):
    batch = collate_pose([canonicalize_sample(fixture_pose(7, variant='all_missing'))])
    model = PoseEncoder(config['model'])
    out = model.encode_full(batch)
    assert out['features'].eq(0).all()
    assert not out['valid'].any()
    assert pool_clip(out)[0].eq(0).all()
