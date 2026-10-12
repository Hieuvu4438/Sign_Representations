import numpy as np
import pytest
import torch
from s2.intervals import logsig2,lead_lag,motion,descriptors,layout,selectors,PATHS,P
from s2.joints import J,LSHO,RSHO,RWRI,RH0,D_LOC,D_CONF


def oracle(path):
    """Independent tensor-algebra Chen product then log(S)_2=S2-S1⊗S1/2."""
    d=path.shape[-1];s1=np.zeros(d);s2=np.zeros((d,d))
    for delta in np.diff(path,axis=0):
        s2=s2+np.outer(s1,delta)+np.outer(delta,delta)/2
        s1=s1+delta
    log2=s2-np.outer(s1,s1)/2
    return np.r_[s1,log2[np.triu_indices(d,1)]]


@pytest.mark.parametrize('dim',[2,4,6])
def test_tensor_algebra_oracle_translation_reverse_subdivision(dim):
    x=torch.tensor(np.random.default_rng(8).normal(size=(13,dim)),dtype=torch.float64)
    np.testing.assert_allclose(logsig2(x),oracle(x.numpy()),atol=1e-13)
    torch.testing.assert_close(logsig2(x+123),logsig2(x),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(logsig2(x.flip(0)),-logsig2(x))
    sub=torch.stack([x[:-1],(x[:-1]+x[1:])/2],1).flatten(0,1)
    torch.testing.assert_close(logsig2(torch.cat([sub,x[-1:]])),logsig2(x))


def test_known_geometry_and_leadlag():
    square=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    torch.testing.assert_close(logsig2(square),torch.tensor([0.,0.,1.]))
    line=torch.tensor([[0.,0.],[1.,2.],[2.,4.]])
    torch.testing.assert_close(logsig2(line),torch.tensor([2.,4.,0.]))
    lift=lead_lag(line)
    torch.testing.assert_close(lift,torch.tensor([[0.,0.,0.,0.],[1.,2.,0.,0.],
        [1.,2.,1.,2.],[2.,4.,1.,2.],[2.,4.,2.,4.]]))
    assert logsig2(lift).shape==(10,)
    np.testing.assert_allclose(logsig2(lift),oracle(lift.numpy()),atol=1e-6)
    # Area changes with orientation/scale; temporal ordering must be retained.
    A=torch.tensor([[2.,.5],[0.,3.]])
    assert logsig2(square@A.T)[-1]==torch.linalg.det(A)
    reflected=square*torch.tensor([-1.,1.])
    assert logsig2(reflected)[-1]==-1


def test_physical_time_speed_and_controls():
    t=torch.arange(16)/25
    x=torch.stack([2*t,3*t],-1)
    m,ok=motion(x,t)
    for k in (0,3,6):torch.testing.assert_close(m[k:k+2],torch.tensor([2.,3.]))
    faster,_=motion(x,t/2)
    torch.testing.assert_close(faster[:9],m[:9]*2)
    assert faster[9]==m[9] # path length, not speed
    raw,_=motion(x,t,normalization='frame')
    torch.testing.assert_close(raw[3:5],torch.tensor([8/25,12/25]))
    single,sv=motion(x,t,lags=(1,))
    assert not sv[3:9].any() and not single[3:9].any()
    _,short=motion(x[:8],t[:8]);assert not short[6:9].any()


def sample():
    torch.manual_seed(4)
    xy=torch.rand(2,64,J,2)*20
    xy[:,:,LSHO]=torch.tensor([0.,0.]);xy[:,:,RSHO]=torch.tensor([100.,0.])
    obs=torch.ones(2,64,J,dtype=torch.bool)
    return xy,obs,torch.ones_like(obs,dtype=torch.float32),torch.ones(2,64,dtype=torch.bool),torch.arange(64)[None].repeat(2,1)/25


def test_layout_contiguous_gaps_overlapping_support_and_invalid_values():
    xy,obs,c,v,t=sample();p,s,n,k=layout();assert len(p)==375
    clean=descriptors(xy,obs,c,v,t)
    assert clean['input'].shape==(2,375,27) and clean['valid'].all()
    forbidden=torch.zeros_like(obs);forbidden[:,10,RWRI]=True
    masked=descriptors(xy,obs,c,v,t,forbidden=forbidden)
    wrist=(p==0)&(s<=10)&(s+n>10)
    assert not masked['valid'][:,wrist].any()
    assert masked['valid'][:,(p==0)&~wrist].all()
    xy[:,10,RWRI]=1e8
    changed=descriptors(xy,obs,c,v,t,forbidden=forbidden)
    torch.testing.assert_close(changed['input'],masked['input'],rtol=0,atol=0)
    # Missingness independently invalidates all touching intervals, not just deltas.
    obs[:,10,RWRI]=False
    gap=descriptors(xy,obs,c,v,t)
    assert not gap['valid'][:,wrist].any()
    assert not gap['motion'][:,wrist].any()
    with pytest.raises(ValueError,match='timestamps'):
        descriptors(xy,obs,c,v,t*0)


def test_unrelated_nan_joint_does_not_poison_other_paths():
    xy,obs,c,v,t=sample()
    clean=descriptors(xy,obs,c,v,t)
    # This hand joint has no trajectory in S2; its missingness is unrelated.
    obs[:,:,RH0+20]=False;xy[:,:,RH0+20]=torch.nan
    got=descriptors(xy,obs,c,v,t)
    for key in clean:torch.testing.assert_close(got[key],clean[key],rtol=0,atol=0)


def test_query_selection_requires_same_hidden_endpoint_and_bin():
    xy,obs,c,v,t=sample();f=torch.zeros_like(obs);f[:,10,RWRI]=True
    target=torch.zeros(2,8,32,dtype=torch.bool);target[:,D_LOC,5]=True
    inp={'forbidden':f,'valid_f':v,'target':target}
    w=selectors(inp);p,s,n,_=layout()
    hit=(p==0)&(s<=10)&(s+n>10)
    assert (w[:,hit].sum(-1)>0).all() and not w[:,(p==0)&~hit].any()
    target.zero_();target[:,D_CONF,5]=True
    assert selectors(inp)[:,hit].sum()>0 # duplicate wrist owner closure
    target.zero_();target[:,D_LOC,25]=True
    assert not selectors(inp).any() # hidden elsewhere cannot justify query
