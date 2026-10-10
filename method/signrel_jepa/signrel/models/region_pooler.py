import torch
from torch import nn

ARTICULATORS = ("left_hand","right_hand","face","body")
DEFAULT_EDGES = ((0,1),(0,2),(1,2),(0,3),(1,3))


def token_membership(boxes, confidence, grid, threshold=0.5, mode="center"):
    """Union valid raw-frame boxes per tubelet; invalid boxes never enlarge the union."""
    b,t,a,_ = boxes.shape
    tt,hh,ww = grid
    if a != 4 or t % tt:
        raise AssertionError("Raw frames/token-time alignment mismatch")
    raw = boxes.reshape(b,tt,t//tt,a,4)
    conf = confidence.reshape(b,tt,t//tt,a)
    valid = conf >= threshold
    low = raw[...,:2].masked_fill(~valid[...,None],float("inf")).amin(2)
    high = raw[...,2:].masked_fill(~valid[...,None],float("-inf")).amax(2)
    any_valid = valid.any(2)
    if mode == "center":
        yy,xx = torch.meshgrid((torch.arange(hh,device=boxes.device)+0.5)/hh,
                              (torch.arange(ww,device=boxes.device)+0.5)/ww,indexing="ij")
        m = ((xx >= low[...,0,None,None]) & (xx <= high[...,0,None,None]) &
             (yy >= low[...,1,None,None]) & (yy <= high[...,1,None,None])).float()
    elif mode == "intersection":
        yy,xx = torch.meshgrid(torch.arange(hh,device=boxes.device)/hh,
                              torch.arange(ww,device=boxes.device)/ww,indexing="ij")
        width = (torch.minimum(xx+1/ww,high[...,0,None,None])-
                 torch.maximum(xx,low[...,0,None,None])).clamp(min=0)
        height = (torch.minimum(yy+1/hh,high[...,1,None,None])-
                  torch.maximum(yy,low[...,1,None,None])).clamp(min=0)
        m = width*height*hh*ww
    else:
        raise ValueError(mode)
    return m * any_valid[...,None,None]


class RegionPooler(nn.Module):
    def forward(self, tokens, membership):
        if tokens.ndim != 5 or membership.shape[:2] != tokens.shape[:2] or membership.shape[-2:] != tokens.shape[2:4]:
            raise AssertionError("Token/region alignment mismatch")
        sums = membership.sum((-1,-2))
        z = torch.einsum("bthwd,btahw->btad",tokens,membership.to(tokens.dtype))
        z = z/sums[...,None].clamp_min(1e-6)
        valid = sums > 0
        return z.masked_fill(~valid[...,None],0),valid
