import torch
from signrel.models.backbone_adapter import TinyVideoViT

from signrel.masks import base as mask_module
from signrel.eval import extract as extract_module


def test_random_baseline_masks_complete_temporal_tubes():
    module = mask_module
    # Dataclass resolves its module for annotations during loading.
    masker = module.SignMaskGenerator(mixture={'random_multiblock':1.},hidden_ratio=.65,min_visible_ratio=.25)
    member = torch.ones(1,8,4,6,6)
    mask = masker.sample((8,6,6),member,rng=torch.Generator().manual_seed(42))
    hidden = mask.hidden_tokens.reshape(8,6,6)
    assert torch.equal(hidden,hidden[:1].expand_as(hidden))
    assert .6 <= float(hidden.float().mean()) <= .65
    assert mask.metadata[0]['complete_temporal_tubes']


def test_restored_encoder_retains_checkpoint_origin(tmp_path):
    module = extract_module
    encoder = TinyVideoViT(dim=16,heads=4,depth=1)
    origin = {**encoder.upstream_init,'run_note':'retained original construction provenance'}
    path = tmp_path/'trained.pt'
    torch.save({'config':{'model':{'backbone':{'provider':'tiny','dim':16,'heads':4,'depth':1}}},
                'student_encoder':encoder.state_dict(),'upstream_init':origin,'global_step':4,'source_hash':'source-fingerprint'},path)
    restored = module.load_encoder(path)
    assert restored.upstream_init == origin
    assert restored.restored_checkpoint['global_step'] == 4
    assert restored.restored_checkpoint['component'] == 'student_encoder'
    assert all(torch.equal(value,restored.state_dict()[name]) for name,value in encoder.state_dict().items())


def test_stronger_relation_mask_removes_both_endpoint_regions():
    module = mask_module
    member = torch.zeros(1,4,4,4,4)
    member[:,:,0,:,0] = 1
    member[:,:,1,:,3] = 1
    member[:,:,2,0,1:3] = 1
    member[:,:,3,3,:] = 1
    mask = module.SignMaskGenerator(mixture={'relation_break':1.},edges=[(0,1)],relation_hide_both=True).sample(
        (4,4,4),member,rng=torch.Generator().manual_seed(42))
    info = mask.metadata[0]
    hidden = mask.hidden_tokens.reshape(4,4,4)
    for endpoint in info['endpoints_hidden']:
        assert (hidden[info['start']:info['end']][member[0,info['start']:info['end'],endpoint] > 0]).all()
    remaining = member[0,info['start']:info['end']] > 0
    remaining &= ~hidden[info['start']:info['end'],None]
    assert remaining.flatten(1).any(1).all()
    assert mask.visible_tokens.float().mean() >= .25
