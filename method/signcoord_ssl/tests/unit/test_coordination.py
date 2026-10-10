import math
import pytest
import torch

from signcoord.losses.distributed_reduce import weighted_mean
from signcoord.targets.lag import lag_loss_terms, window_correlations
from signcoord.targets.spectral import (cross_spectra, fit_energy_thresholds, pair_window_weights,
                                       phase_loss_terms, rfft_energy, window_fft, window_starts)


def signals(batch=3, n=31, dtype=torch.double):
    torch.manual_seed(8)
    return torch.randn(batch,n,3,2,dtype=dtype),torch.ones(batch,n,3,dtype=dtype)


def test_velocity_window_counts_and_bins():
    assert window_starts(31) == (0,8)
    assert len(window_starts(63)) == 6
    assert len(window_starts(63,32,16)) == 2
    x,r=signals()
    assert window_fft(x).shape == (3,2,3,2,9)
    with pytest.raises(ValueError):
        phase_loss_terms(x,x,r,1,1,1,window=32)
    with pytest.raises(ValueError):
        window_fft(x,window=15)


def test_fft_axis_peak():
    t=torch.arange(32,dtype=torch.double)
    x=torch.zeros(1,32,3,2,dtype=torch.double)
    x[0,:,1,0]=torch.cos(2*math.pi*3*t/16)
    u=window_fft(x)
    assert (u[0,:,1,0].abs().argmax(-1) == 3).all()
    assert not u[:,:,0].abs().any()


def test_exact_phase_sign_swap_and_amplitude_ambiguity():
    n,d,k=64,3,4
    t=torch.arange(n,dtype=torch.double)
    p=torch.cos(2*math.pi*k*t/n)
    q=torch.roll(p,d)
    up,uq=torch.fft.rfft(p,norm="ortho"),torch.fft.rfft(q,norm="ortho")
    cross=up[k]*uq[k].conj()
    phase=2*math.pi*k*d/n
    assert torch.allclose(cross/cross.abs(),torch.exp(torch.tensor(1j*phase,dtype=torch.cdouble)),atol=1e-10)
    assert torch.allclose(uq.abs(),up.abs(),atol=1e-10)
    assert torch.allclose(uq[k]*up[k].conj(),cross.conj(),atol=1e-10)


@pytest.mark.parametrize("n",[31,32,64])
def test_parseval_full_and_weighted_rfft(n):
    x=torch.randn(3,n,dtype=torch.double)
    time=x.square().sum(-1)
    assert torch.allclose(time,torch.fft.fft(x,norm="ortho").abs().square().sum(-1),rtol=1e-10,atol=1e-10)
    assert torch.allclose(time,rfft_energy(x),rtol=1e-10,atol=1e-10)


def test_invalid_window_and_detached_target_gate():
    x,r=signals()
    r[0,5,0]=0
    weights=pair_window_weights(r)
    assert weights[0,0,0] == 0 and weights[0,1,0] == 1
    pred=(x+.1*torch.randn_like(x)).requires_grad_()
    n,d,diag=phase_loss_terms(pred,x.requires_grad_(),r,1,1,1)
    assert diag["gate"][0,0].sum() == 0
    assert not diag["gate"].requires_grad and not diag["q_target"].requires_grad
    _,_,changed=phase_loss_terms(pred*2,x,r,1,1,1)
    assert torch.equal(diag["gate"],changed["gate"])
    (n/d).backward()
    assert x.grad is None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_hold_and_orthogonal_vector_gates():
    pred=torch.randn(2,31,3,2,requires_grad=True)
    zero=torch.zeros_like(pred); rel=torch.ones(2,31,3)
    n,d,diag=phase_loss_terms(pred,zero,rel,1,1,1)
    loss,_=weighted_mean(n,d)
    loss.backward()
    assert d == 0 and loss == 0 and torch.isfinite(pred.grad).all()
    x=zero.clone(); t=torch.arange(31.)
    x[:,:,0,0]=torch.cos(2*math.pi*3*t/16)
    x[:,:,1,1]=torch.cos(2*math.pi*3*t/16)
    _,_,diag=phase_loss_terms(pred,x,rel,1,1,1)
    assert not diag["gate"][:,0].any()


def test_phase_float64_gradcheck():
    x,r=signals(batch=1)
    pred=(x+.05*torch.randn_like(x)).requires_grad_()
    def objective(p):
        n,d,_=phase_loss_terms(p,x,r,1,1,1)
        return n/d
    assert torch.autograd.gradcheck(objective,(pred,),eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=True)


def test_positive_non_circular_lag():
    # Zero-mean random sequence and delay: positive lag aligns p[t],q[t+d].
    torch.manual_seed(99)
    p=torch.randn(32,dtype=torch.double); p-=p.mean()
    q=torch.cat((torch.zeros(3,dtype=torch.double),p[:-3]))
    x=torch.zeros(1,32,3,2,dtype=torch.double); x[0,:,0,0]=p; x[0,:,1,0]=q
    correlations,_=window_correlations(x,window=32,hop=8)
    assert int(correlations[0,0,0].argmax())-4 == 3
    assert correlations.shape[-1] == 9


def test_lag_zero_signal_gradient_and_nonzero_backward():
    x,r=signals()
    pred=(x+.1*torch.randn_like(x)).requires_grad_()
    n,d,_=lag_loss_terms(pred,x,r)
    (n/d).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    hold=torch.zeros_like(x,requires_grad=True)
    n,d,_=lag_loss_terms(hold,hold.detach(),r)
    weighted_mean(n,d)[0].backward()
    assert d == 0 and torch.isfinite(hold.grad).all()


def test_a8_keeps_basic_validity_but_removes_confidence_gate():
    x,r=signals(); r[:]=.01
    _,d,_=phase_loss_terms(x,x,r,1,1,1)
    assert d == 0
    _,d,diag=phase_loss_terms(x,x,r,1,1,1,basic_valid=torch.ones_like(r,dtype=torch.bool))
    assert d > 0 and (diag["gate"] == 1).all()
    basic=torch.ones_like(r,dtype=torch.bool); basic[:,:,0]=False
    _,_,diag=phase_loss_terms(x,x,r,1,1,1,basic_valid=basic)
    assert not diag["gate"][:,:2].any()


def test_calibration_has_counts_and_insufficient_data_fails():
    x,r=signals(batch=10)
    stats=fit_energy_thresholds(x,r,min_samples=8)
    assert stats["tau_cross"].shape == (3,7) and (stats["tau_cross_counts"] == 10).all()
    with pytest.raises(ValueError,match="calibration failure"):
        fit_energy_thresholds(x[:1],r[:1],min_samples=8)


def test_complex_vector_cancellation():
    u=torch.ones(1,2,3,2,9,dtype=torch.cdouble)
    u[:,:,1,1]=-1
    s,p,q=cross_spectra(u,torch.ones(1,2,3))
    assert not s[:,0].abs().any() and (p[:,0]>0).all() and (q[:,0]>0).all()
