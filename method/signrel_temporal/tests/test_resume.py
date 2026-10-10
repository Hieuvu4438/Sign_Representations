import copy
import json
from pathlib import Path
import torch
from signrel.config import load_config
from signrel.data.synthetic import make_synthetic
from signrel.train.loop import train
from signrel.train.checkpoint import tensor_hash


def setup(tmp_path):
    make_synthetic(tmp_path/'data',videos=8)
    cfg=copy.deepcopy(load_config('configs/smoke_a4.yaml').values)
    for key in ['cache_manifest','ssl_manifest','ssl_dev_manifest','protocol']:
        cfg['paths'][key]=str(tmp_path/'data'/Path(cfg['paths'][key]).name)
    cfg['train'].update(max_updates=8,checkpoint_every=4,log_every=100)
    return cfg


def test_exact_resume_optimizer_dropout_and_plan(tmp_path,monkeypatch):
    cfg=setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg['experiment']['output_dir']='runs/full'
    full=train(cfg)
    # Same resolved contract including output dir, separate filesystem snapshots.
    original=Path('runs/full')
    archived=Path('runs/full_archive')
    original.rename(archived)
    split=train(cfg,stop_after=4)
    resumed=train(cfg,resume='runs/full/checkpoints/last.pt')
    first=torch.load(archived/'checkpoints/last.pt',weights_only=False)
    second=torch.load(original/'checkpoints/last.pt',weights_only=False)
    assert tensor_hash(first['model_state'])==tensor_hash(second['model_state'])
    assert first['consumed_microbatch_index']==second['consumed_microbatch_index']==8
    assert (archived/'window_plan.jsonl').read_bytes()==(original/'window_plan.jsonl').read_bytes()
    a=[json.loads(l) for l in (archived/'metrics.jsonl').read_text().splitlines()]
    b=[json.loads(l) for l in (original/'metrics.jsonl').read_text().splitlines()]
    assert [r['microbatches'] for r in a]==[r['microbatches'] for r in b]
    assert train(cfg,resume='runs/full/checkpoints/last.pt')['status']=='already_complete'


def test_budget_stop_saves_boundary(tmp_path,monkeypatch):
    cfg=setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg['experiment']['output_dir']='runs/budget'
    cfg['budget']['max_wall_minutes']=0
    result=train(cfg)
    assert result['status']=='budget_stopped' and result['successful_updates']==0
    state=torch.load('runs/budget/checkpoints/last.pt',weights_only=False)
    assert state['successful_update']==0 and state['consumed_microbatch_index']==0
