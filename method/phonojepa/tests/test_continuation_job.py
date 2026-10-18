import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]))
from pj.continuation_support import (continuation_lr,SampleStream,SeededWindows,save_last,load_latest,rng_state,restore_rng)
from pj.continuation_eval import report_metrics
from pj.train import Trainer,get_args
from pj.data.dataset import Cfg
from pj.joints import J


class MiniCache:
    def __init__(self,n=10):
        rng=np.random.default_rng(0)
        self.xy=rng.normal(200,50,size=(n,80,J,2)).astype(np.float32)
        self.conf=np.ones((n,80,J),np.float32)
    def __len__(self):return len(self.xy)
    def clip(self,index):return self.xy[index],self.conf[index],25.


def small_trainer():
    a=get_args(['--out','unused','--target','ctx','--device','cpu'])
    a.d=16;a.depth=2;a.heads=4;a.dp=8;a.pdepth=1;a.batch=2
    return Trainer(a,'cpu')


def compare(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):compare(left,right)
    else:assert a==b


def update(trainer,batch):
    trainer.opt.zero_grad(set_to_none=True);loss,_=trainer.step(batch,4000);loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.params,1.,error_if_nonfinite=True)
    trainer.opt.step();trainer.bal.tick();trainer.commit_centres();trainer.model.ema_update(.9999)
    return float(loss.detach())


def test_continuation_lr_has_separate_warmup_and_final_endpoint():
    assert continuation_lr(499,50160,3e-5)==pytest.approx(3e-5)
    assert continuation_lr(50159,50160,3e-5)==3e-6
    assert 3e-6<continuation_lr(0,50160,1e-5)<1e-5
    with pytest.raises(ValueError):continuation_lr(50160,50160,1e-5)


@pytest.mark.parametrize('workers',[0,2])
def test_prefetched_stream_restores_exact_next_views_and_shuffled_drop_last(workers):
    torch.set_num_threads(1)
    dataset=SeededWindows(MiniCache(),Cfg(),seed=0)
    a=SampleStream(dataset,4,123,workers)
    a.next();state=copy.deepcopy(a.state_dict());expected=a.next()
    assert state['cursor']==4 and len(state['order'])==8
    b=SampleStream(dataset,4,123,workers);b.load_state_dict(state)
    compare(expected,b.next());a.close();b.close()


def test_checkpoint_resume_preserves_optimizer_center_teacher_rng_and_next_loss(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(0)
    ds=SeededWindows(MiniCache(),Cfg(),seed=0)
    a=small_trainer();stream=SampleStream(ds,2,7,0)
    update(a,stream.next());meta={'global_step':1,'branch_step':1};config={'arm':'lr1','data':'hash'}
    save_last(a,stream,meta,config,tmp_path)
    expected_batch=stream.next();expected_loss=update(a,expected_batch)
    b=small_trainer();other=SampleStream(ds,2,7,0)
    restored,recovery=load_latest(b,other,config,tmp_path)
    assert restored==meta and recovery['step']==1
    actual_batch=other.next();compare(expected_batch,actual_batch)
    assert update(b,actual_batch)==expected_loss
    compare(a.model.state_dict(),b.model.state_dict());compare(a.opt.state_dict(),b.opt.state_dict())
    compare(a.cce.c,b.cce.c);compare(a.bal.state_dict(),b.bal.state_dict())
    stream.close();other.close()


def test_checkpoint_fallback_does_not_bypass_provenance_and_preserves_backup(tmp_path):
    torch.set_num_threads(1)
    ds=SeededWindows(MiniCache(),Cfg(),seed=0);stream=SampleStream(ds,2,7,0);a=small_trainer();config={'arm':'lr1'}
    save_last(a,stream,{'global_step':0,'branch_step':0},config,tmp_path)
    update(a,stream.next());save_last(a,stream,{'global_step':1,'branch_step':1},config,tmp_path)
    (tmp_path/'last.pt').write_bytes(b'corrupt')
    b=small_trainer();other=SampleStream(ds,2,7,0)
    restored,recovery=load_latest(b,other,config,tmp_path)
    assert restored['global_step']==0 and recovery['checkpoint'].endswith('previous.pt')
    with pytest.raises(ValueError,match='provenance'):load_latest(b,other,{'arm':'changed'},tmp_path)
    save_last(b,other,restored,config,tmp_path)
    assert torch.load(tmp_path/'previous.pt',weights_only=False)['step']==0
    stream.close();other.close()


def test_dictionary_metrics_keep_native_pessimistic_ties_and_query_denominator():
    g=np.array([[1.,0.],[1.,0.],[0.,1.]],np.float32);q=np.array([[1.,0.],[0.,1.],[0.,0.]],np.float32)
    gallery=[{'label':i} for i in range(3)];queries=[{'label':i} for i in [0,2,0]]
    metrics,ranks=report_metrics(g,q,gallery,queries)
    np.testing.assert_array_equal(ranks,[2,1,3])
    assert metrics['queries']==3 and metrics['Rec@1']==1/3
    assert metrics['MRR']==pytest.approx((.5+1+1/3)/3)


def test_interrupted_validation_replays_before_any_new_optimizer_update(tmp_path,monkeypatch):
    import pj.continuation_train as runner
    from pj.continuation_support import sha256
    torch.set_num_threads(1)
    base=small_trainer();base.save(tmp_path/'base.pt',200640)
    (tmp_path/'baseline/validation').mkdir(parents=True)
    (tmp_path/'baseline/validation/metrics.json').write_text(json.dumps({'metrics':{'Rec@1':.5}}))
    (tmp_path/'job_lock.json').write_text('{}')
    directory=tmp_path/'arrays';directory.mkdir();cache=MiniCache()
    np.save(directory/'xy.npy',cache.xy.reshape(-1,J,2));np.save(directory/'conf.npy',cache.conf.reshape(-1,J))
    np.save(directory/'offsets.npy',np.arange(len(cache)+1)*80);np.save(directory/'fps.npy',np.full(len(cache),25.))
    job={'base_checkpoint':str(tmp_path/'base.pt'),'base_checkpoint_sha256':sha256(tmp_path/'base.pt'),
         'training_cache_sha256':'synthetic','memmap_cache':str(directory),'sampler_seed':1,'train_workers':0,
         'cuda_memory_fraction':.16,'validation_interval':1,'patience':1}
    arm={'name':'test','peak_lr':1e-5,'additional_steps':50160}
    def interrupted(*a,**kw):raise InterruptedError('simulated interrupted validation')
    monkeypatch.setattr(runner,'evaluate_checkpoint',interrupted)
    result=runner.run(job,tmp_path,arm,'cpu')
    assert result['status']=='paused' and result['branch_step']==1
    before=torch.load(tmp_path/'arms/test/last.pt',weights_only=False)
    monkeypatch.setattr(runner,'evaluate_checkpoint',lambda *a,**kw:{'metrics':{'Rec@1':.4}})
    result=runner.run(job,tmp_path,arm,'cpu')
    after=torch.load(tmp_path/'arms/test/last.pt',weights_only=False)
    assert result['status']=='early_stopped' and result['branch_step']==1
    compare(before['student'],after['student']);compare(before['opt'],after['opt'])
