import copy,io,pickle
from pathlib import Path
import numpy as np
import pytest
import torch
from fp.data.dataset import Cfg,build_pair,collate
from fp.data.stream import Batches
from fp.data.mapped import MappedCache
from fp.model import FusionJEPA
from fp.trainer import Trainer
from fp import features as Fe
from fp.pl import features as PLFe
from fp.joints import J,ANCHORS
from fp.utils import atomic_torch
from fp import checkpoint
from fp.inference import encode_prepared
from fp.evaluation import dictionary


def clip():
    xy=np.random.default_rng(8).normal(150,20,(110,J,2)).astype(np.float32)
    xy[:,ANCHORS[0]]=[200,100];xy[:,ANCHORS[1]]=[100,100]
    return xy,np.ones((110,J),np.float32),30.


def batch(seed=0):
    x,c,fps=clip()
    return collate([{k:build_pair(x,c,fps,np.random.default_rng([seed,i]),Cfg(),k) for k in ('near','far')}|{'flipped':False} for i in (0,1)])


def config(tmp_path,**kw):
    return {'d':32,'pl_depth':2,'graph_depth':2,'dp':16,'pdepth':1,'gate_max':.3,'device':'cpu',
            'pl_checkpoint':'unused','lr':3e-4,'wd':.05,'stats_clips':2,'steps':8,'warmup_steps':1,
            'gate_ramp_steps':1,'lambda_body':.1,'lambda_relations':.1,'lambda_retention':1.,**kw}


def test_gate0_exact_frozen_PL_and_gradients(tmp_path):
    torch.manual_seed(0);tr=Trainer(config(tmp_path),initialize=False);v=batch()['near']
    tr.model.student.gate_ramp.fill_(1.)
    a=tr.model.student(v,gate_override=0)
    base=a['base'].reshape(2,32,7,32)
    torch.testing.assert_close(a['h'].reshape(2,32,8,32)[:,:,:7],base,rtol=0,atol=0)
    old=copy.deepcopy(tr.model.student.pl.state_dict())
    result=tr.update(batch(),2)
    assert np.isfinite(result['loss']) and result['near_body_queries']>0 and result['near_relation_queries']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tr.model.student.graph.parameters())
    assert tr.model.student.out.weight.grad.abs().sum()>0
    assert tr.model.student.gate.grad.abs().sum()>0
    assert all(p.grad is None for p in tr.model.student.pl.parameters())
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    for key,x in old.items():torch.testing.assert_close(tr.model.student.pl.state_dict()[key],x,rtol=0,atol=0)
    assert result['near_gate_max_actual']<=.3


@pytest.mark.parametrize('kind',['near','far'])
def test_raw_mask_graph_and_fusion_leakage(kind):
    model=FusionJEPA(32,2,2,16,1).student.eval();model.gate_ramp.fill_(1.)
    x,c,fps=clip()
    for seed in range(5):
        a=build_pair(x,c,fps,np.random.default_rng(seed),Cfg(),kind)
        changed=np.where(a['raw_scrub'][...,None],1e7,x)
        b=build_pair(changed,np.where(a['raw_scrub'],.123,c),fps,np.random.default_rng(seed),Cfg(),kind)
        def out(v):
            v=collate([{'near':v,'far':v,'flipped':False}])['near']
            return model(v),v
        oa,v=out(a);ob,_=out(b)
        for key in ['h','cls','base','graph_present']:
            torch.testing.assert_close(oa[key],ob[key],rtol=0,atol=0)


def test_checkpoint_exact_continuation_and_fail_closed(tmp_path):
    torch.manual_seed(1);c=config(tmp_path);tr=Trainer(c,initialize=False);b=batch()
    tr.update(b,2);proof={'test':'fixture'};progress={'validated':[0,2],'pending_validation':None}
    checkpoint.save(tmp_path,tr.state(3,proof,progress))
    result=tr.update(b,3)
    other=Trainer(c,initialize=False);ck,recovery=checkpoint.load(tmp_path)
    assert other.restore(ck,proof)==(3,progress)
    assert other.update(b,3)==result
    for name,x in tr.model.state_dict().items():torch.testing.assert_close(x,other.model.state_dict()[name],rtol=0,atol=0)
    for name,x in tr.aux.state_dict().items():torch.testing.assert_close(x,other.aux.state_dict()[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='mismatch'):other.restore(ck,{'changed':True})
    checkpoint.save(tmp_path,tr.state(4,proof,progress))
    (tmp_path/'last.pt').write_bytes(b'corrupt')
    ck,recovery=checkpoint.load(tmp_path)
    assert ck['step']==3 and recovery['generation']=='previous.pt' and recovery['read_errors']


def test_mapped_cache_values_and_worker_pickle():
    p=Path('method/phonojepa/runtime/cache/aslc_train_full.npz')
    c=MappedCache(p)
    assert len(c)==40154 and c.xy.shape[1:]==(103,2)
    payload=pickle.dumps(c);assert len(payload)<500
    other=pickle.loads(payload)
    for i in [0,51,40153]:
        for x,y in zip(c.clip(i),other.clip(i)):np.testing.assert_array_equal(x,y)


def test_dictionary_known_rank_invalid_missing_and_ties():
    g=np.array([[1.,0.],[0.,1.],[0.,0.]])
    q=np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.],[1.,0.]])
    gr=[{'sample_id':f'g{i}','gloss':str(i)} for i in range(3)]
    qr=[{'sample_id':f'q{i}','gloss':str(i if i<2 else 0 if i<4 else 9)} for i in range(5)]
    m,r=dictionary(g,q,gr,qr)
    np.testing.assert_array_equal(r,np.array([1.,1.,2.,np.inf,np.inf]))
    assert m['Rec@1']==.4 and m['n_queries']==5 and m['gallery_valid']==2
    with pytest.raises(ValueError,match='overlap'):dictionary(g,q,gr,[gr[0],*qr[1:]])


def test_schedule_resume_and_validation_boundary():
    from fp.train import validation_due
    rows=list(Batches(3000,32,7,0,10))
    assert rows[5:]==list(Batches(3000,32,7,5,10))
    c={'warmup_steps':2000,'validate_every':5000,'steps':200640}
    for s in [0,2000,5000,200640]:assert validation_due(s,c,[]) and not validation_due(s,c,[s])
    assert not validation_due(4999,c,[])


def test_auxiliary_dropout_not_training_target(tmp_path):
    from fp.auxiliary import geometry,loss_weights
    v=batch()['near'];ys=geometry(v['t_xy'],v['t_obs'],v['valid_f'])
    v['forbidden'].zero_()
    # Observation dropout/missingness alone must never justify a hidden target.
    v['s_obs'].zero_()
    for value in loss_weights(v,ys).values():assert not value.any()


def test_no_shoulder_fallback_leak_and_no_graph_reads_missing():
    model=FusionJEPA(32,2,2,16,1).student.eval();model.gate_ramp.fill_(1.)
    v=batch()['near']
    v['s_obs'][:,:,ANCHORS]=False
    o=model(v)
    altered=copy.deepcopy(v)
    altered['s_xy'][~v['s_obs']]=1e7
    b=model(altered)
    torch.testing.assert_close(o['h'],b['h'],rtol=0,atol=0)


def test_initialization_loads_all_original_PL_parameters_exactly():
    from fp.paths import REPO
    path=REPO/'method/phonojepa/runtime/runs/PJ_PLctx_FULL_160ep/final.pt'
    m=FusionJEPA();m.initialize(path)
    old=torch.load(path,map_location='cpu',weights_only=False)
    for name,x in old['teacher'].items():
        torch.testing.assert_close(m.student.pl.state_dict()[name],x,rtol=0,atol=0)
        torch.testing.assert_close(m.teacher.pl.state_dict()[name],x,rtol=0,atol=0)
