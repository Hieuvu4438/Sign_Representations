import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from signcoord.models.attention import forward_4d


class AttentionFixture(nn.Module):
    def __init__(self,local,stride):
        super().__init__();self.num_cls_tokens=0;self.heads=4;self.head_dim=8
        self.dim_out=32;self.q_stride=stride;self.window_size=4;self.use_mask_unit_attn=local
        self.qkv=nn.Linear(16,96);self.proj=nn.Linear(32,32)

    def forward(self,x):
        b,n,_=x.shape;windows=n//(self.q_stride*self.window_size) if self.use_mask_unit_attn else 1
        q,k,v=self.qkv(x).reshape(b,-1,windows,3,4,8).permute(3,0,4,2,1,5).unbind(0)
        if self.q_stride>1:q=q.reshape(b,4,windows,self.q_stride,-1,8).max(3).values
        out=F.scaled_dot_product_attention(q,k,v)
        return self.proj(out.transpose(1,3).reshape(b,-1,32))


@pytest.mark.parametrize('local,stride',[(True,1),(True,4),(False,1)])
@pytest.mark.parametrize('constant',[False,True])
def test_4d_sdpa_preserves_output_and_pool_tie_parameter_input_gradients(local,stride,constant):
    torch.manual_seed(7);native=AttentionFixture(local,stride).double();fast=copy.deepcopy(native)
    x=(torch.zeros(2,64,16) if constant else torch.randn(2,64,16)).double().requires_grad_()
    y=x.detach().clone().requires_grad_()
    a=native(x);b=forward_4d(fast,y)
    assert torch.allclose(a,b,atol=1e-10,rtol=1e-9)
    a.square().sum().backward();b.square().sum().backward()
    assert torch.allclose(x.grad,y.grad,atol=1e-10,rtol=1e-9)
    for name,p in native.named_parameters():
        assert torch.allclose(p.grad,dict(fast.named_parameters())[name].grad,atol=1e-10,rtol=1e-9),name
