"""Instance-scoped 4D SDPA for the pinned zero-class-token Hiera architectures."""
from types import MethodType

import torch
from torch.nn import functional as F


def forward_4d(module,x):
    if getattr(module,'num_cls_tokens',0)!=0:raise ValueError('4D kernel supports zero class tokens only')
    b,n,_=x.shape
    windows=n//(module.q_stride*module.window_size) if module.use_mask_unit_attn else 1
    qkv=module.qkv(x).reshape(b,-1,windows,3,module.heads,module.head_dim).permute(3,0,4,2,1,5)
    q,k,v=qkv.unbind(0)
    if module.q_stride>1:
        q=q.reshape(b,module.heads,windows,module.q_stride,-1,module.head_dim).max(3).values
    nq,nk=q.shape[-2],k.shape[-2]
    out=F.scaled_dot_product_attention(
        q.reshape(b,module.heads*windows,nq,module.head_dim).contiguous(),
        k.reshape(b,module.heads*windows,nk,module.head_dim).contiguous(),
        v.reshape(b,module.heads*windows,nk,module.head_dim).contiguous())
    out=out.reshape(b,module.heads,windows,nq,module.head_dim).transpose(1,3).reshape(b,-1,module.dim_out)
    return module.proj(out)


def configure_attention(backbone,kernel):
    if kernel=='vendor_5d':return
    if kernel!='sdpa_4d_zero_cls':raise ValueError(f'unsupported attention kernel: {kernel}')
    for block in backbone.blocks:
        if getattr(block.attn,'num_cls_tokens',0)!=0:raise ValueError('class tokens require the vendor kernel')
        block.attn.forward=MethodType(forward_4d,block.attn)
