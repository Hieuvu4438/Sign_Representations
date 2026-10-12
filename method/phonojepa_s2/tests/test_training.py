from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from test_data import clip, pair
from s2.train import args,Trainer,Windows,Batches,input_control
from s2.data.dataset import collate,Cfg
from s2 import features as Fe
from s2.joints import NC,TORSO,ANCHORS
from s2.auxiliary import geometry,loss_weights


def config(tmp_path,**changes):
    a=args(['--lambda-body','.1','--lambda-relations','.1','--out',str(tmp_path),'--smoke','--device','cpu','--d','32','--depth','2','--dp','16','--pdepth','1',
            '--batch','2','--steps','4','--workers','0','--stats-clips','2'])
    for k,v in changes.items():setattr(a,k,v)
    return a


def batch():
    xy,c,fps=clip()
    return collate([{'near':pair(xy,c,fps,seed=i),'far':pair(xy,c,fps,'far',i),'flipped':False} for i in [2,3]])


def test_masked_input_cannot_change_visible_encoder(tmp_path):
    torch.manual_seed(0);tr=Trainer(config(tmp_path));v=batch()['near']
    f=Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'])
    ins,p=Fe.tokens(f,v['valid_f']);a=tr.model.student(ins,p,v['visible'])
    modified=[x.clone() for x in ins]
    for c in range(NC): modified[c][~v['visible'][:,c]] = 12345
    b=tr.model.student(modified,p,v['visible'])
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    visible=v['visible'].transpose(1,2).flatten(1)
    torch.testing.assert_close(a[0][visible],b[0][visible],rtol=0,atol=0)


def test_auxiliary_masks_only_hidden_supported_dimensions():
    v=batch()['near'];ys=geometry(v['t_xy'],v['t_obs'],v['valid_f']);w=loss_weights(v,ys)
    assert w['body'].sum()>0 and w['relations'].sum()>0
    assert not w['body'][:,:,[0,1,2,3,4,5,6]].any()
    assert w['body'].shape[-1]==12 # three nose/hip points; no shoulder target
    for group in w:
        assert not w[group][~v['target'].transpose(1,2)].any()
    # Completely observed endpoint data is not a missing-coordinate task.
    v['s_obs']=v['t_obs'].clone()
    for x in loss_weights(v,ys).values():assert not x.any()


def test_ema_auxiliary_gradient_and_checkpoint_exact_continuation(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(7)
    a=config(tmp_path);tr=Trainer(a);b=batch()
    teacher=[p.clone() for p in tr.model.teacher.parameters()]
    result=tr.step(b,0)
    assert result['near_body_queries'] and result['near_relation_queries']
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    assert tr.aux.heads['body'].weight.grad.abs().sum()>0
    assert tr.aux.heads['relations'].weight.grad.abs().sum()>0
    assert tr.model.student.tok.mlps[5][0].weight.grad.abs().sum()>0
    assert any(not torch.equal(a,b) for a,b in zip(teacher,tr.model.teacher.parameters()))
    proof={'test':True};path=tmp_path/'ckpt.pt';tr.save(path,1,proof)
    r1=tr.step(b,1)
    other=Trainer(a);assert other.load(path,proof)==1
    r2=other.step(b,1)
    assert r1==r2
    for x,y in zip(tr.model.parameters(),other.model.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(tr.aux.parameters(),other.aux.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    other.a.lambda_body=.2
    with pytest.raises(ValueError,match='lambda_body'):other.load(path,proof)


def test_control_no_body_evidence():
    ins=[torch.rand(2,32,d) for d in Fe.TOKEN_DIMS];pres=torch.ones(2,NC,32,dtype=torch.bool)
    x,p=input_control(ins,pres,'capacity');assert x[7].count_nonzero()==0
    x,p=input_control(ins,pres,'shuffle',torch.tensor([1,0]));torch.testing.assert_close(x[7][0],ins[7][1])


def test_batch_plan_worker_independent_and_resume():
    b=list(Batches(3000,32,7,0,10))
    assert b[5:]==list(Batches(3000,32,7,5,10))
    assert b==list(Batches(3000,32,7,0,10))
    assert all(len({i for _,_,i in row})==32 for row in b)


def test_actual_data_workers_match(tmp_path):
    from s2.data.cache import ClipCache
    from s2.train import DEFAULT_CACHE
    ds=Windows(ClipCache(DEFAULT_CACHE),Cfg(),7)
    def first(workers):
        loader=torch.utils.data.DataLoader(ds,batch_sampler=Batches(len(ds),2,7,1,2),num_workers=workers,collate_fn=collate,
            multiprocessing_context="spawn" if workers else None)
        return next(iter(loader))
    a=first(0);b=first(2)
    for view in ['near','far']:
        for key,value in a[view].items():
            if torch.is_tensor(value):torch.testing.assert_close(value,b[view][key],rtol=0,atol=0)


@pytest.mark.parametrize('mode',['real','capacity','shuffle'])
def test_input_only_controls_train_without_auxiliary_gradients(tmp_path,mode):
    tr=Trainer(config(tmp_path,input=mode,lambda_body=0.,lambda_relations=0.,lambda_motion=0.))
    result=tr.step(batch(),0)
    assert np.isfinite(result['loss'])
    assert all(p.grad is None for p in tr.aux.parameters())
