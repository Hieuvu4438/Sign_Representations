import importlib.util,json,os,subprocess,sys,time,signal
from pathlib import Path
import numpy as np
import pytest
import torch
from test_core import config,batch
from fp.trainer import Trainer
from fp.data.dataset import collate,Cfg
from fp.data.stream import Windows
from fp.data.mapped import MappedCache
from fp.inference import Prepare,split_rows,encode_prepared
from fp import features as Fe
from fp.pl import features as PLFe
from fp.paths import ROOT,REPO
from fp.final_eval import linear_probe


def job_module():
    spec=importlib.util.spec_from_file_location('fusion_job',ROOT/'scripts/job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_queue_uses_free_vram_not_other_process_presence_and_frozen_setup(tmp_path):
    module=job_module()
    assert not module.ready(10239) and module.ready(10240) and module.ready(36000)
    job=tmp_path/'job';module.setup(job,ROOT/'configs/full160.json',Path(sys.executable))
    first=module.hashes(job)
    assert first==json.loads((job/'lock.json').read_text())['sha256']
    assert json.loads((job/'lock.json').read_text())['python']==str(Path(sys.executable).absolute())
    assert (job/'code/fp/model.py').is_file()
    with pytest.raises(FileExistsError):module.setup(job,ROOT/'configs/full160.json',Path(sys.executable))
    (job/'code/fp/model.py').write_text('# altered')
    assert module.hashes(job)!=first


def test_whole_inference_gate0_matches_actual_original_pretrained(tmp_path):
    # Current reviewed code must load the real7-channel checkpoint exactly.
    c=json.loads((ROOT/'configs/smoke.json').read_text());c['device']='cpu'
    tr=Trainer(c)
    p=Prepare(split_rows('val')[:1])[0]
    got=encode_prepared(tr.model.teacher,p,'cpu')['pl_gate0']
    import sys
    sys.path.insert(0,str(REPO/'method/phonojepa'))
    from pj.extract import Extractor
    old=Extractor('model',c['pl_checkpoint'],device='cpu')
    old_p={'xy':p['t_xy'],'obs':p['t_obs'],'conf':p['t_conf'],'valid':p['valid_f'],'sample_id':p['sample_id']}
    raw=old.encode(old_p)['_grp'].astype(np.float32).mean(0)
    raw=raw/np.linalg.norm(raw)
    # Original extractor stores fp16; compare within its quantization tolerance.
    np.testing.assert_allclose(got,raw,rtol=3e-3,atol=3e-5)


def test_real_workers_match_with_shared_mmap():
    ds=Windows(MappedCache(REPO/'method/phonojepa/runtime/cache/aslc_train_full.npz'),Cfg(),7)
    from fp.data.stream import Batches
    def read(workers):
        dl=torch.utils.data.DataLoader(ds,batch_sampler=Batches(len(ds),2,7,0,1),num_workers=workers,
                    collate_fn=collate,multiprocessing_context='spawn' if workers else None)
        return next(iter(dl))
    a,b=read(0),read(2)
    for side in ('near','far'):
        for k,x in a[side].items():
            if torch.is_tensor(x):torch.testing.assert_close(x,b[side][k],rtol=0,atol=0)


def test_frozen_linear_probe_with_known_labels_invalid_query_and_resume(tmp_path):
    gen=np.random.default_rng(8)
    centers=np.eye(4,dtype=np.float32)
    rows={s:[{'sample_id':f'{s}{i}','gloss':str(i%4)} for i in range(n)] for s,n in [('train',80),('val',16),('test',16)]}
    features={s:np.stack([centers[i%4]+gen.normal(0,.01,4) for i in range(len(rs))]).astype(np.float32) for s,rs in rows.items()}
    features['test'][0]=0
    value=linear_probe(features,rows,tmp_path,'test-id','cpu',max_epochs=3)
    assert value['test']['n_queries']==16 and value['test']['valid_queries']==15
    assert value['encoder_frozen'] and value['test_selection'] is False
    assert value==linear_probe(features,rows,tmp_path,'test-id','cpu',max_epochs=3)
    with pytest.raises(ValueError,match='identity'):linear_probe(features,rows,tmp_path,'changed','cpu',max_epochs=3)


def test_resume_pending_validation_before_update_subprocess(tmp_path):
    """Signal the actual training CLI inside validation, then resume committed state.
    Small CPU architecture and real clips; expensive production CUDA is separate.
    """
    c=json.loads((ROOT/'configs/smoke.json').read_text())
    # Synthetic checkpoint matches diagnostic architecture; real data/preflight remain.
    from fp.model import FusionJEPA
    model=FusionJEPA(32,2,2,16,1)
    init=tmp_path/'initial.pt';torch.save({'teacher':model.student.pl.state_dict()},init)
    c.update(d=32,pl_depth=2,graph_depth=2,dp=16,pdepth=1,device='cpu',workers=0,
             extract_workers=0,validation_limit=6,steps=2,stats_clips=2,pl_checkpoint=str(init))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c))
    job=tmp_path/'job';run=job/'train'
    cmd=[sys.executable,'-m','fp.train','--config',str(cfg),'--root',str(run)]
    env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log=tmp_path/'first.log'
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+60;sent=False
        while time.monotonic()<deadline and proc.poll() is None:
            p=job/'progress.json'
            if p.exists():
                try:state=json.loads(p.read_text())
                except json.JSONDecodeError:continue
                if state.get('phase')=='validation':proc.send_signal(signal.SIGTERM);sent=True;break
            time.sleep(.05)
        if not sent:proc.kill();proc.wait();pytest.fail(log.read_text())
        assert proc.wait(timeout=30)==75,log.read_text()
    ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert ck['step']==0 and ck['progress']['pending_validation']==0
    completed=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stdout+completed.stderr
    j=json.loads((run/'complete.json').read_text());assert j['steps']==2 and j['validated']==[0,1,2]
    ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    assert ck['progress']['pending_validation'] is None


def test_supervisor_real_process_queue_and_single_instance(tmp_path):
    module=job_module();job=tmp_path/'job'
    module.setup(job,ROOT/'configs/full160.json',Path(sys.executable))
    # Isolated short child proves launch/requeue orchestration, never full training.
    (job/'code/fp/train.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--config');a=p.parse_args()
r=Path(a.root);r.mkdir(exist_ok=True);(r/'complete.json').write_text('{}')
(r.parent/'evaluation').mkdir();(r.parent/'evaluation/complete.json').write_text('{}')
''')
    lock=json.loads((job/'lock.json').read_text());lock['sha256']=module.hashes(job);(job/'lock.json').write_text(json.dumps(lock))
    bindir=tmp_path/'bin';bindir.mkdir();vram=tmp_path/'vram';vram.write_text('100')
    smi=bindir/'nvidia-smi';smi.write_text('#!/bin/sh\ncat '+str(vram)+'\n');smi.chmod(0o755)
    command=[sys.executable,str(ROOT/'scripts/job.py'),'supervise','--job',str(job)]
    env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],PYTHONPATH=str(ROOT))
    with (tmp_path/'supervisor.log').open('w') as f:
        p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                state=job/'supervisor.json'
                if state.exists() and json.loads(state.read_text())['status']=='waiting_for_gpu':break
                if p.poll() is not None:pytest.fail((tmp_path/'supervisor.log').read_text())
                time.sleep(.1)
            assert p.poll() is None and not (job/'train').exists()
            duplicate=subprocess.run(command,env=env,capture_output=True,text=True,timeout=10)
            assert duplicate.returncode!=0 and 'Another supervisor' in duplicate.stderr
            vram.write_text('20000')
            assert p.wait(timeout=25)==0,(tmp_path/'supervisor.log').read_text()
            assert json.loads((job/'supervisor.json').read_text())['status']=='complete'
            assert (job/'train/complete.json').exists()
        finally:
            if p.poll() is None:p.terminate();p.wait(timeout=20)


def test_full_evaluation_chain_and_cached_completion(tmp_path,monkeypatch):
    """Actual ranking/whitening/LP orchestration on an explicit small fixture.
    Real pose extraction is tested separately; synthetic values establish wiring.
    """
    import fp.final_eval as E
    from fp.utils import atomic_torch
    c=config(tmp_path);c.update(smoke=False,extract_workers=0)
    tr=Trainer(c,initialize=False);proof={'fixture':True}
    job=tmp_path/'full_fixture';(job/'train').mkdir(parents=True)
    (job/'train/complete.json').write_text('{}')
    atomic_torch(job/'train/best_validation.pt',tr.state(3,proof,{'best':{'step':3}}))
    rows={s:[{'sample_id':f'{s}_{i}','gloss':str(i%4),'signer':s} for i in range(n)]
          for s,n in [('train',80),('val',16),('test',16)]}
    monkeypatch.setattr(E,'load_config',lambda _:c)
    monkeypatch.setattr(E,'verify',lambda _: (None,proof))
    monkeypatch.setattr(E,'split_rows',lambda split:rows[split])
    called=[]
    def extract(enc,records,cache,out,identity,device,**kw):
        split=records[0]['signer'];called.append(split)
        rng=np.random.default_rng(5)
        values=np.stack([np.pad(np.eye(4)[i%4],(0,92))+rng.normal(0,.01,96) for i in range(len(records))]).astype(np.float32)
        if split=='test':values[0]=0.
        return {'fusion':values,'pl_gate0':values.copy()}
    monkeypatch.setattr(E,'extract',extract)
    original_probe=E.linear_probe
    monkeypatch.setattr(E,'linear_probe',lambda *a,**kw:original_probe(*a,**kw,max_epochs=2))
    assert E.run('fixture',job)==0
    assert called==['train','val','test']
    report=json.loads((job/'evaluation/report.json').read_text())
    for mode in ('fusion','pl_gate0'):
        r=report['results'][mode]
        assert set(r)=={'raw_val','raw_test','white512_val','white512_test','linear_probe'}
        assert r['raw_test']['n_queries']==16 and r['raw_test']['query_valid']==15
        assert r['linear_probe']['test_selection'] is False
    assert E.run('fixture',job)==0 and called==['train','val','test']
