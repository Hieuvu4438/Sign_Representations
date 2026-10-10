import math

import torch

from signcoord.targets.magnitude import fit_magnitude_calibration, magnitude_loss_terms
from signcoord.targets.shuffle import compatible_derangement


def circular_signals():
    t=torch.arange(32,dtype=torch.float64)*2*math.pi*3/16
    xy=torch.stack((t.sin(),t.cos()),-1)
    return xy[None,:,None,:].expand(8,-1,3,-1).clone()


def test_magnitude_delay_invariant_amplitude_sensitive_and_gate_detached():
    target=circular_signals().requires_grad_()
    rel=torch.full((8,32,3),.8,dtype=torch.float64,requires_grad=True)
    cal=fit_magnitude_calibration(target,rel,hop=16)
    shifted=target.detach().clone()
    shifted[:,:,1]=torch.roll(shifted[:,:,1],3,1)
    n,d,diag=magnitude_loss_terms(shifted,target,rel,cal,hop=16)
    assert n/d < 1e-12 and d > 0
    assert not diag['gate'].requires_grad
    student=(2*shifted).requires_grad_()
    n,d,_=magnitude_loss_terms(student,target,rel,cal,hop=16)
    assert n/d > 1
    (n/d).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
    assert target.grad is None and rel.grad is None


def test_magnitude_missing_or_static_target_gives_finite_zero_gradient():
    target=circular_signals()
    rel=torch.ones(8,32,3,dtype=torch.float64)
    cal=fit_magnitude_calibration(target,rel)
    for missing in (True,False):
        student=torch.zeros_like(target,requires_grad=True)
        n,d,_=magnitude_loss_terms(student,torch.zeros_like(target),
                                  torch.zeros_like(rel) if missing else rel,cal)
        assert n == 0 and d == 0
        n.backward()
        assert torch.isfinite(student.grad).all() and student.grad.abs().sum() == 0


def test_shuffle_deranges_only_exact_compatible_support_and_timing():
    support=torch.ones(6,31,3,dtype=torch.bool)
    support[3,-1]=False
    support[4,-1]=False
    times=torch.arange(32,dtype=torch.float64)[None].expand(6,-1).clone()/25
    times[5]*=2
    ids=['b','a','c','d','e','f']
    p,eligible,log=compatible_derangement(support,times,ids,3407,7)
    assert eligible.tolist()==[True,True,True,True,True,False]
    assert torch.equal(support[p[eligible]],support[eligible])
    assert not (p[eligible]==torch.arange(6)[eligible]).any()
    assert log['target_sample_ids'][-1] is None and log['changed_samples']==5
    q,other,_=compatible_derangement(support,times,ids,3407,7)
    assert torch.equal(p,q) and torch.equal(eligible,other)
    # A permutation of batch order retains the same sample-to-target mapping.
    order=torch.tensor([5,3,1,4,0,2])
    shuffled_ids=[ids[i] for i in order]
    q,_,_=compatible_derangement(support[order],times[order],shuffled_ids,3407,7)
    mapping={own:ids[target] for own,target in zip(ids,p.tolist())}
    assert {own:shuffled_ids[target] for own,target in zip(shuffled_ids,q.tolist())}==mapping
