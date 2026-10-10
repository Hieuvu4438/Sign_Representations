import numpy as np
import pytest
import torch
from signrel.losses.state import weighted_mean,hand_state_loss,body_state_loss
from signrel.losses.relation import relation_loss


def oracle(pred,target,valid,quality,masked,pairs,beta=.1):
    p,q = pred/np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-6),target/np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-6)
    numerator=denominator=0.
    for i,t,a,u,b in pairs:
        if not (valid[i,t,a] and valid[i,u,b] and (masked[i,t,a] or masked[i,u,b])): continue
        w=min(quality[i,t,a],quality[i,u,b])
        error=abs(float(p[i,t,a]@p[i,u,b]-q[i,t,a]@q[i,u,b]))
        numerator+=w*(.5*error**2/beta if error<beta else error-.5*beta)
        denominator+=w
    return numerator/denominator if denominator else 0.


def test_fractional_weight_denominator():
    x=torch.tensor([1.,3.],requires_grad=True)
    loss,d=weighted_mean(x,torch.tensor([.1,.2]),x)
    assert loss.item()==pytest.approx(7/3)
    assert d.item()==pytest.approx(.3)
    loss.backward()
    assert x.grad.tolist()==pytest.approx([1/3,2/3])


def test_hand_stream_balance_and_detach():
    p=torch.zeros(1,10,2,3,requires_grad=True)
    q=torch.ones_like(p,requires_grad=True)
    valid=torch.ones(1,10,2,dtype=torch.bool)
    valid[:,1:,1]=False
    loss,_,_=hand_state_loss(p,q,valid,torch.ones(1,10,2),valid)
    assert loss.item()==3
    loss.backward()
    assert q.grad is None and p.grad is not None
    assert p.grad[:,:,0].abs().sum()==pytest.approx(float(p.grad[:,:,1].abs().sum()))


def test_body_invalid_coordinate_and_clean_quality_detach():
    pred=torch.zeros(1,1,14,requires_grad=True)
    target=torch.ones_like(pred,requires_grad=True)
    target.data[:,:,0]=1e6
    coord=torch.ones(1,1,14,dtype=torch.bool); coord[:,:,0]=False
    quality=torch.full((1,1,14),.1,requires_grad=True)
    v=torch.ones(1,1,dtype=torch.bool)
    loss,_=body_state_loss(pred,target,v,coord,quality,v)
    assert loss.item()==pytest.approx(.5)
    loss.backward()
    assert target.grad is None and quality.grad is None
    assert pred.grad[0,0,0]==0


def test_relation_oracle_gradient_and_mask_or():
    rng=np.random.default_rng(42)
    p=rng.normal(size=(2,4,2,3)).astype(np.float32)
    q=rng.normal(size=p.shape).astype(np.float32)
    valid=np.ones(p.shape[:-1],bool)
    quality=np.full(valid.shape,.1,np.float32)
    mask=np.zeros(valid.shape,bool);mask[:,0,0]=True
    pairs=np.array([[0,0,0,1,1],[1,0,0,3,1],[0,1,0,2,1]])
    pred=torch.tensor(p,requires_grad=True);target=torch.tensor(q,requires_grad=True)
    loss,_,_=relation_loss(pred,target,torch.tensor(valid),torch.tensor(quality),torch.tensor(mask),torch.tensor(pairs))
    assert loss.item()==pytest.approx(oracle(p,q,valid,quality,mask,pairs),abs=1e-6)
    loss.backward()
    assert target.grad is None
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


@pytest.mark.parametrize('empty_pairs',[True,False])
def test_zero_eligible_is_differentiable(empty_pairs):
    p=torch.randn(1,4,2,3,requires_grad=True)
    v=torch.ones(1,4,2,dtype=torch.bool)
    pairs=torch.empty(0,5,dtype=torch.long) if empty_pairs else torch.tensor([[0,0,0,1,1]])
    loss,d,_=relation_loss(p,p.detach(),v,torch.full(v.shape,.2),torch.zeros_like(v),pairs)
    loss.backward()
    assert loss.item()==0 and d.item()==0 and p.grad.abs().sum()==0


def test_fail_nonfinite_even_if_weight_zero():
    p=torch.zeros(1,1,2,3);q=p.clone();q[0,0,0,0]=float('nan')
    v=torch.zeros(1,1,2,dtype=torch.bool)
    with pytest.raises(ValueError): hand_state_loss(p,q,v,torch.zeros_like(v,dtype=torch.float),v)
