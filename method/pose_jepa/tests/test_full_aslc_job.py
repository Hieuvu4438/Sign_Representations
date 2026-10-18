from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import pytest
import torch
import yaml
from signrepr.data.fixture import make_fixture, fixture_pose
from signrepr.data.schema import save_pose, sha256_file
from signrepr.evaluation.dictionary import dictionary_metrics
from signrepr.training import managed


def script(name):
    path = Path(__file__).parents[1]/'scripts'/f'{name}.py'
    sys.path.insert(0,str(path.parent))
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_full_dictionary_chunking_pessimistic_ties_and_fixed_denominator():
    gallery = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,0.]])
    queries = np.array([[1.,0.],[0.,1.],[0.,0.],[1.,0.]])
    result = dictionary_metrics(gallery,queries,['a','b','c','d'],['a','c','a','d'],
                                ['g0','g1','g2','g3'],['q0','q1','q2','q3'],query_chunk=1,gallery_chunk=1)
    assert [q['rank'] for q in result['queries']] == [2,1,None,None]
    assert result['metrics']['num_queries'] == 4
    assert result['metrics']['R@1'] == .25 and result['metrics']['MRR'] == .375
    assert result['gallery_coverage']['valid'] == 3
    assert result['queries'][3]['failure'] == 'missing_gallery_gloss'
    other = dictionary_metrics(gallery,queries,['a','b','c','d'],['a','c','a','d'],
                               ['g0','g1','g2','g3'],['q0','q1','q2','q3'],query_chunk=4,gallery_chunk=4)
    assert result == other


def test_dictionary_rejects_shared_ids_and_interrupts():
    with pytest.raises(ValueError,match='disjoint'):
        dictionary_metrics([[1.,0.]],[[1.,0.]],['a'],['a'],['same'],['same'])
    with pytest.raises(InterruptedError):
        dictionary_metrics([[1.,0.]],[[1.,0.]],['a'],['a'],['g'],['q'],should_stop=lambda:True)


def test_pose_reuse_uses_official_labels_without_copying_old_cache(tmp_path):
    full = script('full_aslc_job')
    old = tmp_path/'old'; new = tmp_path/'new'; (old/'receipts').mkdir(parents=True)
    source = tmp_path/'source.pkl'; source.write_bytes(b'original source')
    video = tmp_path/'video.mp4'; video.write_bytes(b'original video')
    pose = old/'pose.npz'; save_pose(pose,fixture_pose(16))
    raw = old/'scores.npz'; np.savez(raw,score=[1.,2.])
    metadata = {'sample_id':'id','split':'train','signer_id':'P1','gloss_id':'OFFICIAL',
                'video_path':str(video),'coordinate_system':'viewport_normalized','joint_layout':'coco_wholebody_133',
                'score_profile':'simcc_positive_rational_v1','timing_source':'decoded_source_frame_pts'}
    old_meta = {**metadata,'gloss_id':'LEGACY_ALIAS','signer_id':'legacy-hash'}
    identity = {'pose':full.file_identity(source),'video':full.file_identity(video),'metadata':old_meta}
    row = {**old_meta,'pose_path':str(pose),'pose_sha256':sha256_file(pose),'source_cache_sha256':sha256_file(source),
           'raw_detector_scores_path':str(raw),'raw_detector_scores_sha256':sha256_file(raw),'num_frames':16}
    token = hashlib.sha256(b'id').hexdigest()
    (old/'receipts'/f'{token}.json').write_text(json.dumps({'input_identity':identity,'row':row}))
    task = {'root':str(new),'reuse_root':str(old),'record':{'source':str(source),'metadata':metadata},
            'signer_aliases':{'legacy-hash':'P1'}}
    restored,reused,status = full.prepare_clip(task)
    assert reused and status == 'ok' and restored['gloss_id'] == 'OFFICIAL' and restored['signer_id'] == 'P1'
    assert restored['pose_path'] == str(pose) and not (new/'poses').exists()
    assert full.prepare_clip(task) == (restored,True,'ok')
    raw.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='score checksum'):
        full.prepare_clip(task)


def test_official_protocol_rejects_signer_overlap(tmp_path):
    full = script('full_aslc_job')
    (tmp_path/'splits').mkdir(); (tmp_path/'videos').mkdir(); (tmp_path/'keypoints').mkdir()
    for split,signer in [('train','P1'),('val','P2'),('test','P3')]:
        (tmp_path/'splits'/f'{split}.csv').write_text(f'Participant ID,Video file,Gloss\n{signer},{split}.mp4,WORD\n')
        (tmp_path/'videos'/f'{split}.mp4').touch(); (tmp_path/'keypoints'/f'{split}.pkl').touch()
    records,splits,signers = full.build_inputs(tmp_path,{'train':1,'val':1,'test':1})
    assert len(records) == 3 and signers['test'] == {'P3'}
    (tmp_path/'splits/val.csv').write_text('Participant ID,Video file,Gloss\nP1,val.mp4,WORD\n')
    with pytest.raises(ValueError,match='signer overlap'):
        full.build_inputs(tmp_path,{'train':1,'val':1,'test':1})


def test_full_managed_dictionary_validation_pause_and_resume(config,tmp_path,monkeypatch):
    manifest = make_fixture(tmp_path/'data')
    cfg = deepcopy(config); cfg['data'].update(train_manifest=str(manifest),validation_manifest=str(tmp_path/'data/val.jsonl'),num_workers=0)
    cfg['run']['output_dir'] = str(tmp_path/'train'); cfg['optim']['max_steps'] = 1
    cfg['evaluation'].update(selection_protocol='dictionary',checkpoint_interval_steps=1,frozen_validation_interval_steps=1)
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    original = managed.validate_dictionary
    monkeypatch.setattr(managed,'validate_dictionary',lambda *a,**kw: (_ for _ in ()).throw(InterruptedError('paused')))
    result = managed.run(path,tmp_path,device='cpu',cpu_threads=1)
    assert result['status'] == 'paused' and not (tmp_path/'train/complete.json').exists()
    before = torch.load(tmp_path/'train/last.pt',weights_only=False)
    monkeypatch.setattr(managed,'validate_dictionary',original)
    result = managed.run(path,tmp_path,device='cpu',cpu_threads=1)
    after = torch.load(tmp_path/'train/last.pt',weights_only=False)
    assert result['status'] == 'complete' and after['step'] == 1
    assert after['metadata']['selection_protocol'] == 'raw_dictionary_official_train_gallery_val_queries'
    for key,value in before['system'].items(): torch.testing.assert_close(value,after['system'][key],rtol=0,atol=0)
    metrics = json.loads((tmp_path/'train/validation_00000001.json').read_text())['metrics']
    assert metrics['num_queries'] == len((tmp_path/'data/val.jsonl').read_text().splitlines())


def test_linear_probe_interruption_replays_epoch_and_counts_failed_queries(tmp_path):
    evaluator = script('evaluate_full_aslc')
    rng = np.random.default_rng(9)
    sizes = {'train':18,'val':9,'test':9}
    rows = {s:[{'sample_id':f'{s}{i}','gloss_id':str(i%3)} for i in range(n)] for s,n in sizes.items()}
    features = {s:rng.normal(size=(n,6)).astype(np.float32) for s,n in sizes.items()}
    features['val'][-1] = 0; features['test'][-1] = 0
    direct = tmp_path/'direct'; direct.mkdir()
    result = evaluator.train_linear_probe(features,rows,direct,tmp_path,'sha',lambda:False,device='cpu',max_epochs=2)
    interrupted = tmp_path/'interrupted'; interrupted.mkdir()
    with pytest.raises(InterruptedError):
        evaluator.train_linear_probe(features,rows,interrupted,tmp_path,'sha',lambda:True,device='cpu',max_epochs=2)
    resumed = evaluator.train_linear_probe(features,rows,interrupted,tmp_path,'sha',lambda:False,device='cpu',max_epochs=2)
    assert resumed == result and result['test']['num_queries'] == 9 and result['test']['valid_queries'] == 8
    a = torch.load(direct/'linear_probe/best.pt',weights_only=False)
    b = torch.load(interrupted/'linear_probe/best.pt',weights_only=False)
    for key,value in a['model'].items(): torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)


def test_gpu_queue_waits_without_allocating_and_preserves_checkpoint_step(tmp_path):
    full = script('full_aslc_job')
    (tmp_path/'train').mkdir(); (tmp_path/'train/checkpoint.json').write_text(json.dumps({'step':123}))
    values = iter([100,8192]); sleeps = []
    assert full.wait_for_gpu(tmp_path,{'max_steps':125482},'train',lambda:False,
                              probe=lambda:next(values),sleep=lambda seconds:sleeps.append(seconds))
    state = json.loads((tmp_path/'progress.json').read_text())
    assert state['status'] == 'waiting_for_gpu' and state['step'] == 123
    assert state['required_free_vram_mib'] == 6144 and sum(sleeps) == 30
    stopped = {'value':False}
    def pause(seconds): stopped['value'] = True
    assert not full.wait_for_gpu(tmp_path,{'max_steps':125482},'train',lambda:stopped['value'],
                                  probe=lambda:100,sleep=pause)


def test_gpu_retry_classifies_only_current_attempt_cuda_oom(tmp_path):
    full = script('full_aslc_job'); log = tmp_path/'train.log'
    log.write_text('torch.OutOfMemoryError: CUDA out of memory\n')
    offset = log.stat().st_size
    with log.open('a') as file: file.write('ValueError: bad data\n')
    assert not full.attempt_was_cuda_oom(log,offset)
    with log.open('a') as file: file.write('torch.OutOfMemoryError: CUDA out of memory\n')
    assert full.attempt_was_cuda_oom(log,offset)
