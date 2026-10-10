import pytest
import torch

from signcoord.models.backbone_adapter import TinyEncoderAdapter
from signcoord.models.hiera_adapter import visible_indices
from signcoord.models.representation import SignCoordModel


def test_dense_decoder_backward_and_target_independent_encoder():
    torch.manual_seed(1)
    enc=TinyEncoderAdapter()
    model=SignCoordModel(enc,{"width":32,"blocks":1,"heads":4},{"hidden":32,"out_dim":16})
    x=torch.randn(2,3,32,32,32); fv=torch.ones(2,32,dtype=torch.bool); times=torch.arange(32)[None].expand(2,-1)/25
    mask=enc.random_mask(2,.8,"cpu")
    out=model(x,mask,fv,times)
    assert out["pred_standardized"].shape == (2,32,119,2)
    out["pred_standardized"].square().mean().backward()
    assert torch.isfinite(enc.patch.weight.grad).all() and enc.patch.weight.grad.abs().sum()>0
    model.decoder.forward=lambda *args: (_ for _ in ()).throw(RuntimeError("targets/decoder unavailable"))
    model.eval()
    with torch.inference_mode():
        encoded=model.encode(x,fv,times)
    assert encoded["sequence"].shape == (2,16,32)
    assert torch.allclose(encoded["clip"].norm(dim=-1),torch.ones(2))


def test_masked_final_block_indices_different_batches():
    mask=torch.zeros(2,8*7*7,dtype=torch.bool)
    mask[0,[0,1,50,100]]=True; mask[1,[2,17,99,300]]=True
    flat,coords,grid=visible_indices(mask,(8,7,7),(1,2,2))
    assert grid == (8,14,14) and flat.shape == (2,16)
    assert torch.equal(flat,(coords[...,0]*14+coords[...,1])*14+coords[...,2])
    # The 4 child positions remain adjacent in the rerolled mask-unit block.
    assert torch.equal(flat[0,:4],torch.tensor([0,1,14,15]))
    assert not torch.equal(flat[0],flat[1])
    mask[1,10]=True
    with pytest.raises(ValueError):
        visible_indices(mask,(8,7,7),(1,2,2))
