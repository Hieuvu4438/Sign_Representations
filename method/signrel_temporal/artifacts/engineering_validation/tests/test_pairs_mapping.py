import numpy as np
import torch
from signrel.losses.pairs import build_pairs
from signrel.losses.relation import relation_values,shuffle_targets
from signrel.data.windows import gather_window
from signrel.data.cache import read_cache
from signrel.data.synthetic import make_synthetic


def test_signed_canonical_boundaries_and_duplicate_sources():
    v=torch.ones(1,4,2,dtype=torch.bool)
    frames=torch.tensor([[0,0,2,3]])
    pairs=build_pairs(v,v.float(),v,frames,torch.zeros(1,4,dtype=torch.bool),[-3,-1,0,1,3],'cross_signed',100,torch.Generator().manual_seed(2))
    assert (pairs[:,2]==0).all() and (pairs[:,4]==1).all()
    assert len(torch.unique(pairs,dim=0))==len(pairs)
    assert all(0<=t<4 and 0<=u<4 for _,t,_,u,_ in pairs.tolist())
    assert all(t==u or frames[0,t]!=frames[0,u] for _,t,_,u,_ in pairs.tolist())
    q=torch.zeros(1,4,2,2);q[:,:,0,0]=1;q[0,:,1]=torch.tensor([[1.,0],[0,1],[1,0],[0,1]])
    _,values=relation_values(q,q,torch.tensor([[0,0,0,1,1],[0,1,0,0,1]]))
    assert values.tolist()==[0,1]


def test_pair_budget_stratified_and_intra_no_diagonal():
    v=torch.ones(2,16,2,dtype=torch.bool)
    pairs=build_pairs(v,v.float(),v,torch.arange(16).repeat(2,1),torch.zeros(2,16,dtype=torch.bool),[1,2,4,8],'intra',32,torch.Generator().manual_seed(4))
    assert len(pairs)==64
    assert (pairs[:,2]==pairs[:,4]).all() and (pairs[:,3]>pairs[:,1]).all()
    assert set((pairs[:,3]-pairs[:,1]).tolist())=={1,2,4,8}


def test_explicit_nonidentity_g(tmp_path):
    make_synthetic(tmp_path,videos=6,frames=40)
    arrays,_=read_cache(tmp_path/'caches/synthetic_0000.npz')
    g=np.array([0,2,3,5])
    clean,padding,time,mapping=gather_window(arrays,g,8,.5)
    assert mapping['g'][:4].tolist()==[0,2,3,5]
    assert mapping['g'][4:].tolist()==[-1]*4
    assert time[:4].tolist()==[0,80,120,200]
    assert padding.tolist()==[False]*4+[True]*4
    q=clean['appearance'][None]
    _,values=relation_values(q,q,torch.tensor([[0,0,0,3,1]]))
    expected=float(q[0,0,0]@q[0,3,1])
    assert np.isclose(values.item(),expected,atol=1e-5)


def test_shuffle_cross_window_not_state_targets():
    pairs=torch.tensor([[0,0,0,1,1],[1,0,0,1,1],[0,0,0,0,1]])
    target=torch.tensor([.1,.8,.6])
    shuffled,keep=shuffle_targets(target,pairs,torch.Generator().manual_seed(17))
    assert shuffled[:2].tolist()==target[:2].flip(0).tolist()
    assert keep.tolist()==[True,True,False]
    assert target.tolist()==torch.tensor([.1,.8,.6]).tolist()
