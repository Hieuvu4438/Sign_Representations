from copy import deepcopy
import torch
from signrepr.models.pose_encoder import PoseEncoder
from signrepr.models.part_tokenizer import token_metadata
from signrepr.masking.part_temporal import MaskGenerator
from signrepr.losses.hybrid import HybridCriterion


def test_mask_constraints(config, batch):
    valid = token_metadata(batch)['valid']
    make = MaskGenerator(config['mask'])
    mask = make(valid, torch.Generator().manual_seed(9))
    again = make(valid, torch.Generator().manual_seed(9))
    assert torch.equal(mask['target'], again['target'])
    m, c = mask['target'], mask['visible']
    assert not (m & c).any()
    assert torch.equal(m | c, valid)
    assert (c.sum(1) >= 4).all()
    assert (c.reshape(2, -1, 4).sum(-1) >= 1).all()
    assert (c.reshape(2, -1, 4)[:, :, 1:3].sum(1) >= 2).all()


def test_private_hidden_support_and_temporal_negative_control(config, batch):
    model = PoseEncoder(config['model']).eval()
    meta = token_metadata(batch)
    target = torch.zeros_like(meta['valid']); target[:, 4*3+1] = True
    visible = meta['valid'] & ~target
    changed = deepcopy(batch)
    changed['parts']['left_hand']['xy'][:, 6:8, 1:] += .17
    changed['parts']['left_hand']['global_xy'][:, 6:8, 1:] += .17
    with torch.no_grad():
        base_tokens, _ = model.tokenizer(batch)
        new_tokens, _ = model.tokenizer(changed)
        torch.testing.assert_close(base_tokens[visible], new_tokens[visible], atol=0, rtol=0)
        a, b = model.encode_visible(batch, visible), model.encode_visible(changed, visible)
        torch.testing.assert_close(a['features'], b['features'], atol=0, rtol=0)
        fa, fb = model.encode_full(batch), model.encode_full(changed)
        assert not torch.allclose(fa['features'][target], fb['features'][target])
        # A temporal mixing kernel creates a measurable forbidden path.
        def leaky(tokens):
            grid = tokens.reshape(2, -1, 4, tokens.shape[-1])
            return (grid + grid.roll(1, 1)).flatten(1, 2)
        assert not torch.allclose(leaky(base_tokens)[visible], leaky(new_tokens)[visible])
    criterion = HybridCriterion(config['loss'], 16)
    ta, _ = criterion.targets(batch, {'target': target}, meta)
    tb, _ = criterion.targets(changed, {'target': target}, meta)
    assert not torch.equal(ta['left_hand']['geometry'][0], tb['left_hand']['geometry'][0])
    assert not torch.equal(ta['left_hand']['motion'][0][:, 7], tb['left_hand']['motion'][0][:, 7])
