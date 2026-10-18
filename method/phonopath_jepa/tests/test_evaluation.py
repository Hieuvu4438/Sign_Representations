import numpy as np
import torch
import pytest
from pp.evaluate import class_max_ranks,metrics,pca_pair,holm,bootstrap_delta
from pp.downstream import select_labels,scores
from pp.signature import logsig2,chen,lead_lag
from pp.config import Config
from pp.features import assemble
from pp.model import Encoder,Predictor
from pp.base.model import Encoder as ReferenceEncoder,Predictor as ReferencePredictor
from test_core import view,fake_stats,pose
from pp.base import window as W
from pp.base.joints import *


def test_retrieval_ties_classmax_metrics():
    g=np.array([[1.,0],[1.,0],[0,1],[-1,0]],np.float32)
    q=np.array([[1.,0],[0,1]],np.float32)
    r=class_max_ranks(g,q,['a','b','c','c'],['a','c'],gallery_batch=1,query_batch=1)
    np.testing.assert_array_equal(r,[2,1]);m=metrics(r,['a','c']);assert m['R@1']==50 and m['R@5']==100
    with pytest.raises(ValueError):class_max_ranks(g,q,['a','b','c','c'],['x','c'])


def test_pca_train_only_and_holm():
    rng=np.random.default_rng(3);g=rng.normal(size=(20,7));q=rng.normal(size=(3,7))
    a,b,fit=pca_pair(g,q,4);aa,bb,fit2=pca_pair(g,q+1e6,4)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(fit['components'],fit2['components'])
    assert holm([.01,.04,.03])==[.03,.06,.06]
    c=bootstrap_delta(np.ones(50),np.ones(50)*2,np.repeat(['s1','s2'],25),np.tile(['g1','g2'],25),100)
    assert c['delta']==100 and c['ci95']==[100,100]


def test_signature_autograd_and_tree_like():
    x=torch.randn(2,9,4,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(logsig2,(x,))
    path=torch.tensor([[0.,0],[1,0],[0,0],[1,0],[0,0]])
    assert torch.equal(logsig2(path),torch.zeros(3))
    assert logsig2(lead_lag(path)).abs().sum()>0


def test_a0_matches_plctx_encoder_and_predictor():
    cfg=Config.for_arm('A0',d=32,depth=2,dp=16,pdepth=1,device='cpu')
    ref=ReferenceEncoder(32,2,4).eval();enc=Encoder(cfg).eval()
    enc.tok.load_state_dict(ref.tok.state_dict());enc.cls.data.copy_(ref.cls)
    enc.blocks.load_state_dict(ref.blocks.state_dict());enc.norm.load_state_dict(ref.norm.state_dict())
    b=view(cfg);p=assemble(b,cfg,None,True);vis=p['visible'].reshape(1,32,7).transpose(1,2)
    h,c,_=enc(p);hr,cr,_=ref(p['s_inputs'],p['s_present'],vis)
    assert torch.equal(h,hr) and torch.equal(c,cr)
    pred=Predictor(cfg);base=ReferencePredictor(32,16,1,4)
    pred.reference.load_state_dict(base.state_dict())
    targets=b['target'];out=pred(h,c,p['visible'],targets.transpose(1,2).flatten(1))
    expected=pred.head(base(h,c,vis,targets));assert torch.equal(out,expected)


def test_label_selection_fixed_glosses_and_exact_k():
    rows=[dict(gloss=f'g{i}') for i in range(12) for _ in range(8)]
    a,classes=select_labels(rows,k=5,n_glosses=4,seed=2)
    b,c2=select_labels(rows,k=5,n_glosses=4,seed=3)
    np.testing.assert_array_equal(classes,c2);assert len(a)==20
    assert not np.array_equal(a,b)


def test_mirror_involution_and_dn_swap():
    x,c=pose();mirror,mc=W.mirror(x,c);back,bc=W.mirror(mirror,mc)
    np.testing.assert_allclose(back,x,atol=3e-5);np.testing.assert_array_equal(c,bc)
    np.testing.assert_allclose(mirror[:,RH0:RH0+21,1],x[:,LH0:LH0+21,1])


def test_global_chen_full_path_includes_last_increment():
    from pp.ptokens import SEGMENTS
    x=torch.randn(2,64,4,dtype=torch.double)
    seg=torch.stack([logsig2(x[:,s:e+1]) for s,e in SEGMENTS],1)
    torch.testing.assert_close(chen(seg,torch.ones(2,len(SEGMENTS),dtype=torch.bool),4),logsig2(x),atol=1e-13,rtol=1e-13)
