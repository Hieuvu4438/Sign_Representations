import numpy as np
import torch
import pytest
from pp.config import Config,ARMS
from pp.signature import logsig2,chen,RobustStats
from pp.paths import token_dims
from pp.data import build_pair,collate
from pp.ptokens import build,WINDOWS,SEGMENTS
from pp.features import assemble
from pp.model import PhonoPathJEPA
from pp.train import Trainer
from pp.base.joints import *

torch.set_num_threads(2)

def pose(n=90):
    rng=np.random.default_rng(7)
    x=rng.uniform(100,300,(n,103,2)).astype(np.float32)
    x[:,LSHO]=[350,120];x[:,RSHO]=[150,120]
    for base,wr,el in ((RH0,RWRI,RELB),(LH0,LWRI,LELB)):
        x[:,wr]=[220,230];x[:,el]=[160,180]
        x[:,base:base+21]=x[:,wr,None]+rng.normal(0,10,(n,21,2))
        x[:,base+9]=x[:,base]+[0,-30]
    return x,np.ones((n,103),np.float32)*.8


def fake_stats(cfg):
    from pp.paths import signature_dims
    state={}
    for c,dims in enumerate(signature_dims(cfg.time_channel)):
        for scale in (0,1,2):
            state[f'{c}/{scale}']=dict(median=torch.zeros(sum(dims)),iqr=torch.ones(sum(dims)),eps=torch.zeros(len(dims)))
    s=RobustStats(state,cfg.asinh);s.provenance=dict(split='train',source_sha256='synthetic')
    return s


def view(cfg,hidden=None):
    x,c=pose();v=build_pair(x,c,30.,np.random.default_rng(9),cfg,'near',hidden)
    b=collate([dict(near=v,far=v)])['near']
    return b


def test_signature_geometry_reparam_chen():
    x=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]],dtype=torch.float64)
    torch.testing.assert_close(logsig2(x),torch.tensor([0.,0.,1.],dtype=torch.float64))
    dense=torch.cat([x[k:k+1]+torch.arange(4,dtype=x.dtype)[:,None]/4*(x[k+1]-x[k]) for k in range(4)]+[x[-1:]])
    torch.testing.assert_close(logsig2(x),logsig2(dense),atol=1e-14,rtol=1e-14)
    seg=torch.stack([logsig2(x[:3]),logsig2(x[2:])])
    torch.testing.assert_close(chen(seg,torch.ones(2,dtype=torch.bool),2),logsig2(x))
    # A false segment contributes no hidden endpoint jump.
    torch.testing.assert_close(chen(seg,torch.tensor([True,False]),2),seg[0])
    mirror=x*torch.tensor([-1.,1.]); assert logsig2(mirror)[-1]==-logsig2(x)[-1]


def test_dims_count_global_equality():
    assert token_dims()==[19,195,19,195,37]
    cfg=Config.for_arm('A1',morph=False,noise=0,warp=0)
    b=view(cfg,np.zeros((7,32),bool));stats=fake_stats(cfg)
    p=build(b['t_xy'],b['t_obs'],b['t_conf'],b['valid_f'],b['output_hidden'],cfg,stats)
    assert sum(x.shape[1] for x in p['inputs'])==110
    for t,s in zip(p['global_t'],p['global_s']):assert torch.equal(t,s)
    for x,d in zip(p['inputs'],token_dims()):assert x.shape[-1]==d


def test_end_to_end_source_leakage():
    cfg=Config.for_arm('A1',morph=True,noise=.02,warp=.2)
    x,c=pose();h=np.zeros((7,32),bool);h[[0,1,6],9:16]=True
    v=build_pair(x,c,15.,np.random.default_rng(9),cfg,'near',h)
    altered=x.copy();altered[v['source_hidden']]=1e6
    v2=build_pair(altered,c,15.,np.random.default_rng(9),cfg,'near',h)
    stats=fake_stats(cfg)
    a=assemble(collate([dict(near=v,far=v)])['near'],cfg,stats)
    b=assemble(collate([dict(near=v2,far=v2)])['near'],cfg,stats)
    assert torch.equal(a['visible'],b['visible'])
    for ch in range(7):
        visible=v['visible'][ch]
        assert torch.equal(a['s_inputs'][ch][:,visible],b['s_inputs'][ch][:,visible])
    for ch in range(5):
        visible=a['p']['visible'][:,ch]
        assert torch.equal(a['p']['inputs'][ch][visible],b['p']['inputs'][ch][visible])
        assert torch.equal(a['global'][ch],b['global'][ch])
    model=PhonoPathJEPA(cfg).eval()
    with torch.no_grad():
        ha,ca,_=model.student(a);hb,cb,_=model.student(b)
    assert torch.equal(ca,cb);assert torch.equal(ha[a['visible']],hb[b['visible']])


@pytest.mark.parametrize('arm',list(ARMS))
def test_arms_forward_backward(arm):
    cfg=Config.for_arm(arm,d=32,depth=2,dp=16,pdepth=1,device='cpu',morph=False,noise=0)
    tr=Trainer(cfg,fake_stats(cfg) if cfg.paths else None)
    b=view(cfg);batch=dict(near=b,far=b)
    loss,gn,terms=tr.step(batch,0,10)
    assert np.isfinite(loss) and np.isfinite(gn)
    assert all(p.grad is None for p in tr.model.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tr.model.student.parameters())


def test_basepoint_local_support():
    cfg=Config.for_arm('A1',smooth=False,morph=False,noise=0)
    b=view(cfg,np.zeros((7,32),bool));stats=fake_stats(cfg)
    p=build(b['t_xy'],b['t_obs'],b['t_conf'],b['valid_f'],b['output_hidden'],cfg,stats)
    x=b['t_xy'].clone();x[:,40:]+=10000
    q=build(x,b['t_obs'],b['t_conf'],b['valid_f'],b['output_hidden'],cfg,stats)
    # Altering public anchors changes body frame: preserve them in this intervention.
    x[:,:,ANCHORS]=b['t_xy'][:,:,ANCHORS]
    q=build(x,b['t_obs'],b['t_conf'],b['valid_f'],b['output_hidden'],cfg,stats)
    for c in range(5): assert torch.equal(p['inputs'][c][:,0],q['inputs'][c][:,0])
