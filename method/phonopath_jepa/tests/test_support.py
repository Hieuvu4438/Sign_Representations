import numpy as np
import torch
from pp.config import Config
from pp.data import build_pair,collate,positions,observation
from pp.features import assemble
from pp.ptokens import fill_short_gaps,smooth_hands,build
from pp.signature import logsig2,chen
from pp.base.joints import *
from test_core import pose,fake_stats


def test_short_gaps_never_fill_long_runs_and_carry_endpoint_support():
    x=torch.arange(64.).view(1,64,1,1).expand(1,64,103,2).clone()
    ob=torch.ones(1,64,103,dtype=torch.bool);h=torch.zeros_like(ob)
    ob[:,10:12,0]=False;ob[:,20:23,0]=False;h[:,9,0]=True
    y,o,u=fill_short_gaps(x,ob,h)
    assert o[:,10:12,0].all() and not o[:,20:23,0].any()
    assert u[:,10:12,0].all();torch.testing.assert_close(y[:,10:12],x[:,10:12])


def test_shared_warp_and_fixed_25fps():
    cfg=Config.for_arm('A1',warp=.2,morph=False,noise=0)
    x,c=pose(100);hidden=np.zeros((7,32),bool)
    v=build_pair(x,c,15.,np.random.default_rng(2),cfg,'near',hidden)
    np.testing.assert_array_equal(v['t_xy'],v['s_xy'])
    np.testing.assert_array_equal(v['t_xy'],v['p_xy'])
    cfg.warp=0;cfg.speed=(1.,1.)
    i0,i1,w,valid=positions(100,15.,np.random.default_rng(2),cfg)
    assert abs(np.diff(i0+w).mean()-.6)<1e-5


def test_smoothing_expands_support_and_temporal_target_fraction():
    cfg=Config.for_arm('A1',morph=False,noise=0)
    x,c=pose(80);h=np.zeros((7,32),bool);h[0,8:10]=True
    v=build_pair(x,c,25.,np.random.default_rng(4),cfg,'far',h)
    b=collate([dict(near=v,far=v)])['near'];p=assemble(b,cfg,fake_stats(cfg))['p']
    # Any token covering the hidden hand stencil is removed, regardless of path dimension.
    assert not p['visible'][0,1,3:6].any()
    assert (p['target']&p['visible']).sum()==0
    assert not (p['drop']&p['target']).any()


def test_observation_filters_do_not_use_scores_as_thresholds():
    x,c=pose();wh=[640,480]
    a,ac=observation(x,c*0,wh);b,bc=observation(x,c,wh)
    np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b))
    x[:,RH0]=x[:,RWRI]+[200.,0]
    a,_=observation(x,c,wh);assert not np.isfinite(a[:,RH0:RH0+21]).any()


def test_perturbation_with_natural_gaps_and_no_shoulders():
    cfg=Config.for_arm('A1',morph=False,noise=0,warp=.2)
    x,c=pose();x[20:22,RH0+8]=np.nan;x[:,ANCHORS]=np.nan
    h=np.zeros((7,32),bool);h[0,6:13]=True;h[6,19:23]=True
    first=build_pair(x,c,30.,np.random.default_rng(9),cfg,'near',h)
    other=x.copy();other[first['source_hidden']]=1e6
    second=build_pair(other,c,30.,np.random.default_rng(9),cfg,'near',h)
    a=assemble(collate([dict(near=first,far=first)])['near'],cfg,fake_stats(cfg))
    b=assemble(collate([dict(near=second,far=second)])['near'],cfg,fake_stats(cfg))
    for ch in range(5):
        visible=a['p']['visible'][:,ch]
        assert torch.equal(a['p']['inputs'][ch][visible],b['p']['inputs'][ch][visible])
        assert torch.equal(a['global'][ch],b['global'][ch])


def test_a2_global_is_additive_qv_without_antisymmetric_area():
    x=torch.randn(12,3,dtype=torch.double)
    segments=torch.stack([logsig2(x[:7],moment=True),logsig2(x[6:],moment=True)])
    actual=chen(segments,torch.ones(2,dtype=torch.bool),3,moment=True)
    torch.testing.assert_close(actual,logsig2(x,moment=True))


def test_raw_observation_gate_dependency_closure():
    cfg=Config.for_arm('A1',morph=True,noise=.02,warp=.2)
    raw,c=pose();x,cc=observation(raw,c,[640,480])
    h=np.zeros((7,32),bool);h[1,7:16]=True
    first=build_pair(x,cc,30.,np.random.default_rng(9),cfg,'far',h)
    changed=raw.copy();changed[first['source_hidden']]=1e6
    xx,c2=observation(changed,c,[640,480])
    second=build_pair(xx,c2,30.,np.random.default_rng(9),cfg,'far',h)
    a=assemble(collate([dict(near=first,far=first)])['near'],cfg,fake_stats(cfg))
    b=assemble(collate([dict(near=second,far=second)])['near'],cfg,fake_stats(cfg))
    for ch in range(7):
        visible=first['visible'][ch]
        assert torch.equal(a['s_inputs'][ch][:,visible],b['s_inputs'][ch][:,visible])
    for ch in range(5):
        visible=a['p']['visible'][:,ch]
        assert torch.equal(a['p']['inputs'][ch][visible],b['p']['inputs'][ch][visible])
        assert torch.equal(a['global'][ch],b['global'][ch])


def test_hidden_state_targets_remain_predictor_queries():
    cfg=Config.for_arm('A1',morph=False,noise=0)
    x,c=pose();h=np.zeros((7,32),bool);h[0,8:16]=True
    v=build_pair(x,c,25.,np.random.default_rng(9),cfg,'far',h)
    a=assemble(collate([dict(near=v,far=v)])['near'],cfg,fake_stats(cfg))
    target=torch.from_numpy(v['target']).T.flatten()
    assert torch.equal(a['target'][0,:224],target)
    assert target.any() and not (a['visible'][0,:224]&target).any()


def test_motion_seeding_preserves_plcm_cell_budget():
    from pp.data import sample_mask
    x,_=pose(64)
    for kind in ('near','far'):
        for seed in range(10):
            a=sample_mask(32,np.random.default_rng(seed),Config.for_arm('A1'),kind,x)
            b=sample_mask(32,np.random.default_rng(seed),Config.for_arm('A10'),kind,x)
            assert a.hidden.sum()==b.hidden.sum()
            assert not (b.hidden&b.drop).any()


def test_full_global_positive_leak_control_changes_with_hidden_content():
    cfg=Config.for_arm('A5',morph=False,noise=0,warp=0)
    x,c=pose();h=np.zeros((7,32),bool);h[1,8:16]=True
    first=build_pair(x,c,25.,np.random.default_rng(9),cfg,'far',h)
    changed=x.copy();changed[first['source_hidden']]=1e6
    second=build_pair(changed,c,25.,np.random.default_rng(9),cfg,'far',h)
    a=assemble(collate([dict(near=first,far=first)])['near'],cfg,fake_stats(cfg))
    b=assemble(collate([dict(near=second,far=second)])['near'],cfg,fake_stats(cfg))
    assert any(not torch.equal(x,y) for x,y in zip(a['global'],b['global']))


def test_a3_global_follows_whole_time_subtree_closure():
    cfg=Config.for_arm('A3',morph=False,noise=0)
    x,c=pose();h=np.zeros((7,32),bool);h[[0,1]]=True
    v=build_pair(x,c,25.,np.random.default_rng(9),cfg,'far',h)
    pack=assemble(collate([dict(near=v,far=v)])['near'],cfg,fake_stats(cfg))
    assert not pack['visible'][0,-5] and not pack['visible'][0,-4]
    assert pack['visible'][0,-3] and pack['visible'][0,-2]
    assert not pack['visible'][0,-1] # REL touches the dominant hand
