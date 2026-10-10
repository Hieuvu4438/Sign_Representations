from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from signcoord.losses.distributed_reduce import weighted_mean
from signcoord.losses.regularization import variance_covariance


def worker(rank, rendezvous, results):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    theta=torch.tensor(1.2,dtype=torch.double,requires_grad=True)
    x=torch.tensor([1.,2.],dtype=torch.double) if rank == 0 else torch.tensor([3.,4.],dtype=torch.double)
    weights=torch.tensor([1.,0.],dtype=torch.double) if rank == 0 else torch.tensor([.3,.7],dtype=torch.double)
    numerator=((theta*x).square()*weights).sum()
    loss,log=weighted_mean(numerator,weights.sum())
    loss.backward()
    grad=theta.grad.detach().clone(); dist.all_reduce(grad); grad/=2
    central=(torch.tensor([1.,2.,3.,4.],dtype=torch.double).square()*torch.tensor([1.,0.,.3,.7],dtype=torch.double)).sum()
    assert torch.allclose(grad,2*theta.detach()*central/2,atol=1e-10)
    torch.manual_seed(13)
    all_x=torch.randn(4,3,dtype=torch.double)
    matrix=torch.arange(12.,dtype=torch.double).view(3,4).div(20).requires_grad_()
    z=all_x[rank*2:(rank+1)*2]@matrix
    var,cov=variance_covariance(z)
    (var+.01*cov).backward()
    gathered_grad=matrix.grad.clone(); dist.all_reduce(gathered_grad); gathered_grad/=2
    # Same centralized functional after destroying process group to avoid gather.
    dist.destroy_process_group()
    central_matrix=matrix.detach().clone().requires_grad_()
    var0,cov0=variance_covariance(all_x@central_matrix)
    (var0+.01*cov0).backward()
    assert torch.allclose(gathered_grad,central_matrix.grad,atol=1e-10,rtol=1e-10)
    torch.save({"weighted_gradient_error":abs(grad-2*theta.detach()*central/2).item(),
                "gather_gradient_error":(gathered_grad-central_matrix.grad).abs().max().item()},results+str(rank)+".pt")


def test_two_process_weighting_and_differentiable_gather(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/"rendezvous"),str(tmp_path/"rank")),nprocs=2,join=True)
    for rank in range(2):
        result=torch.load(str(tmp_path/"rank")+str(rank)+".pt",weights_only=True)
        assert result["weighted_gradient_error"] < 1e-10
        assert result["gather_gradient_error"] < 1e-10
