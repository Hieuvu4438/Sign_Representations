import torch
from signrel.config import load_config
from signrel.models.encoder import initialized_model
from signrel.data.synthetic import make_synthetic
from signrel.data.windows import WindowDataset,collate


def test_hidden_feature_leakage_and_layer_initialization(tmp_path):
    torch.set_num_threads(2)
    make_synthetic(tmp_path,videos=6)
    cfg=load_config('configs/smoke_a4.yaml').values
    data=WindowDataset(tmp_path/'cache_manifest.jsonl',cfg,17)
    batch=collate([data.sample(0),data.sample(1)])
    model=initialized_model(cfg['model'],123).eval()
    original=model(batch['student'])
    modified={k:v.clone() for k,v in batch['student'].items()}
    modified['appearance'][modified['app_ssl_mask']]=torch.randn_like(modified['appearance'][modified['app_ssl_mask']])*100
    changed=model(modified)
    assert torch.equal(original['context'],changed['context'])
    assert not torch.equal(model.transformer.layers[0].linear1.weight,model.transformer.layers[1].linear1.weight)
    from signrel.losses.state import hand_state_loss
    c=batch['clean'];mask=batch['student']['app_ssl_mask']
    before=hand_state_loss(original['app_prediction'],c['appearance'],c['app_valid'],c['app_quality'],mask)[0]
    target=c['appearance'].clone();target[mask]=torch.randn_like(target[mask])
    after=hand_state_loss(original['app_prediction'],target,c['app_valid'],c['app_quality'],mask)[0]
    assert before!=after


def test_same_draw_mask_plan_across_variant_and_global_rng(tmp_path):
    make_synthetic(tmp_path,videos=6)
    a=load_config('configs/smoke_a1.yaml').values;b=load_config('configs/smoke_a4.yaml').values
    first=WindowDataset(tmp_path/'cache_manifest.jsonl',a,17).sample(42)
    torch.randn(1000)
    second=WindowDataset(tmp_path/'cache_manifest.jsonl',b,17).sample(42)
    assert first['metadata']==second['metadata']
    for key in first['student']: assert torch.equal(first['student'][key],second['student'][key])
