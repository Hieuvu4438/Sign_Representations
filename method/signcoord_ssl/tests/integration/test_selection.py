import copy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from signcoord.config import load_config
from signcoord.engine.checkpoint import load_checkpoint
from signcoord.engine.distributed import TrainingContext
from signcoord.engine.selection import selection_inputs
from signcoord.engine.train import make_dataset,train
from signcoord.provenance import sha256_file,tensor_state_hash
from signcoord.targets.statistics import fit_target_statistics

PROJECT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def selected_config(tmp_path_factory):
    root=tmp_path_factory.mktemp('selection_source')/'fixture'
    subprocess.run([sys.executable,str(PROJECT/'tools/generate_synthetic_fixture.py'),'--output',str(root)],
                   check=True,capture_output=True)
    config=load_config(root/'fixture_smoke.yaml')
    fit_target_statistics(make_dataset(config),config['data']['ssl_train_manifest'],
                          config['data']['target_stats_path'],config['data']['phase_calibration_path'])
    labels=json.loads((root/'labels.json').read_text());selected_labels={}
    for split,source,count in [('train','ssl_train',8),('dev','ssl_dev',4)]:
        records=[json.loads(line) for line in (root/'manifests'/f'{source}.jsonl').read_text().splitlines()][:count]
        for record in records:
            record['split']=split;record['pose_cache_key']=None
            selected_labels[record['sample_id']]=labels[record['sample_id']]
        (root/'manifests'/f'downstream_{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root/'selection_labels.json').write_text(json.dumps(selected_labels))
    config['evaluation'].update(selection_enabled=True,
        selection_gallery_manifest=str(root/'manifests/downstream_train.jsonl'),
        selection_query_manifest=str(root/'manifests/downstream_dev.jsonl'),
        selection_labels_path=str(root/'selection_labels.json'),selection_keep_features=True)
    config['train'].update(max_updates=4,eval_every_updates=2,save_every_updates=2)
    config['model']['decoder']['dropout']=.1
    return config


def test_frozen_selection_preserves_training_rng_and_exact_resume(selected_config,tmp_path):
    torch.set_num_threads(2)
    full=copy.deepcopy(selected_config);full['run_dir']=str(tmp_path/'full')
    resumed=copy.deepcopy(selected_config);resumed['run_dir']=str(tmp_path/'resumed')
    off=copy.deepcopy(selected_config);off['run_dir']=str(tmp_path/'selection_off');off['evaluation']['selection_enabled']=False
    train(full);train(resumed,stop_after=2)
    train(resumed,resume=tmp_path/'resumed/checkpoints/last.pt')
    train(off)
    a=load_checkpoint(tmp_path/'full/checkpoints/final.pt')
    b=load_checkpoint(tmp_path/'resumed/checkpoints/final.pt')
    c=load_checkpoint(tmp_path/'selection_off/checkpoints/final.pt')
    for key in a['model_state']:
        assert torch.equal(a['model_state'][key],b['model_state'][key]),key
        assert torch.equal(a['model_state'][key],c['model_state'][key]),key
    assert torch.equal(a['rng_state']['torch'],b['rng_state']['torch'])
    assert torch.equal(a['rng_state']['torch'],c['rng_state']['torch'])
    assert (tmp_path/'full/metrics.jsonl').read_text()==(tmp_path/'resumed/metrics.jsonl').read_text()
    history=a['selection_state']['history']
    assert [r['optimizer_update'] for r in history]==[2,4]
    assert [r['metric'] for r in history]==[r['metric'] for r in b['selection_state']['history']]
    best_metric=max(r['metric'] for r in history)
    best_update=min(r['optimizer_update'] for r in history if r['metric']==best_metric)
    best=load_checkpoint(tmp_path/'full/checkpoints/best_dev.pt')
    assert a['optimizer_update']==4 and best['optimizer_update']==best_update
    assert best['selection_rule']=='dev_rec1_max_earliest_tie_secondary'
    assert a['selection_rule']=='fixed_budget_final'
    assert a['selection_state']['best_checkpoint_sha256']==sha256_file(tmp_path/'full/checkpoints/best_dev.pt')
    state={key.removeprefix('encoder.'):value for key,value in best['model_state'].items() if key.startswith('encoder.')}
    assert tensor_state_hash(state)==next(r['encoder_state_hash'] for r in history if r['optimizer_update']==best_update)
    assert a['compute_ledger']['elapsed_dev_selection_seconds']>0
    for row in history:
        report=json.loads(Path(row['evaluation_report']).read_text())
        assert not report['test_evaluation'] and not report['gallery_feature_metadata']['labels_read_by_encoder']
        assert not report['gallery_feature_metadata']['sequence_features_stored']
        assert not (Path(row['evaluation_report']).parent/'encoder_snapshot.pt').exists()


def test_resume_from_best_checkpoint_in_new_run_preserves_prior_best(selected_config,tmp_path):
    current=copy.deepcopy(selected_config);current['run_dir']=str(tmp_path/'source')
    train(current,stop_after=2)
    checkpoint=tmp_path/'source/checkpoints/best_dev.pt'
    current['run_dir']=str(tmp_path/'continuation')
    train(current,resume=checkpoint)
    saved=load_checkpoint(tmp_path/'continuation/checkpoints/final.pt')
    assert saved['optimizer_update']==4
    assert Path(saved['selection_state']['best_checkpoint_path']).exists()
    assert saved['selection_state']['best_checkpoint_sha256']==sha256_file(saved['selection_state']['best_checkpoint_path'])


def test_selection_rejects_test_split_and_extra_test_labels(selected_config,tmp_path):
    current=copy.deepcopy(selected_config)
    path=tmp_path/'test.jsonl'
    text=Path(current['evaluation']['selection_query_manifest']).read_text().replace('"split": "dev"','"split": "test"')
    path.write_text(text);current['evaluation']['selection_query_manifest']=str(path)
    with pytest.raises(ValueError,match='not allowed'):selection_inputs(current)
    current=copy.deepcopy(selected_config)
    labels=json.loads(Path(current['evaluation']['selection_labels_path']).read_text());labels['test_only_id']=0
    path=tmp_path/'labels.json';path.write_text(json.dumps(labels));current['evaluation']['selection_labels_path']=str(path)
    with pytest.raises(ValueError,match='exactly'):selection_inputs(current)


def heartbeat_worker(rank,rendezvous,output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,world_size=2,rank=rank,
                            timeout=timedelta(seconds=2))
    context=TrainingContext('cpu')
    try:
        def slow():time.sleep(4);return {'done':True}
        assert context.rank_zero_long('slow_eval',slow,heartbeat_seconds=.1)=={'done':True}
        def bad():time.sleep(.2);raise ValueError('injected frozen evaluation failure')
        try:context.rank_zero_long('failing_eval',bad,heartbeat_seconds=.05)
        except RuntimeError as error:
            assert 'injected frozen evaluation failure' in str(error)
            Path(output+str(rank)).write_text('passed')
        else:raise AssertionError('all ranks must receive evaluation failure')
    finally:dist.destroy_process_group()


def test_long_rankzero_work_uses_heartbeats_and_collective_errors(tmp_path):
    mp.spawn(heartbeat_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')),nprocs=2,join=True)
    assert all((tmp_path/f'rank{rank}').read_text()=='passed' for rank in range(2))


def selected_ddp_worker(rank,rendezvous,config,resume):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        if resume:
            train(config,stop_after=2)
            train(config,resume=Path(config['run_dir'])/'checkpoints/last.pt')
        else:train(config)
    finally:dist.destroy_process_group()


def test_ddp_selected_checkpoints_and_rank_rng_exact_resume(selected_config,tmp_path):
    checkpoints=[]
    for name,resume in [('full',False),('resumed',True)]:
        config=copy.deepcopy(selected_config);config['run_dir']=str(tmp_path/name)
        config['train'].update(microbatch_per_gpu=2,accumulation_steps=2)
        mp.spawn(selected_ddp_worker,args=(str(tmp_path/(name+'_rendezvous')),config,resume),nprocs=2,join=True)
        checkpoints.append(load_checkpoint(tmp_path/name/'checkpoints/final.pt'))
    a,b=checkpoints
    for key in a['model_state']:assert torch.equal(a['model_state'][key],b['model_state'][key]),key
    for rank in range(2):assert torch.equal(a['rng_states_by_rank'][rank]['torch'],b['rng_states_by_rank'][rank]['torch'])
    assert [r['metric'] for r in a['selection_state']['history']]==[r['metric'] for r in b['selection_state']['history']]
    assert a['selection_state']['best_checkpoint_sha256']==sha256_file(tmp_path/'full/checkpoints/best_dev.pt')
