import torch

from signcoord.models.pooled_prior import PooledPriorDecoder


def test_pooled_prior_uses_only_valid_pooled_memory_and_backpropagates():
    decoder=PooledPriorDecoder(8,32,width=16,blocks=1,mlp_ratio=2).eval()
    memory=torch.randn(2,5,8,requires_grad=True)
    valid=torch.tensor([[True,True,False,True,False],[True,False,False,True,True]])
    coords=torch.randn(2,5,3); times=torch.arange(32)[None].expand(2,-1)/25
    prediction=decoder(memory,coords,valid,times)
    order=torch.tensor([3,1,4,0,2])
    other=decoder(memory[:,order],coords[:,order]+7,valid[:,order],times+5)
    assert prediction.shape==(2,32,119,2)
    assert torch.allclose(prediction,other,atol=1e-6)
    prediction.square().mean().backward()
    assert memory.grad[~valid].abs().sum()==0 and memory.grad[valid].abs().sum()>0
