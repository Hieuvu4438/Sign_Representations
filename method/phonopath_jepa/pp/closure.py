"""Source-frame masking and conservative, temporally aware support closure."""
import numpy as np
import torch
from pp.base.joints import OWNED, DEPENDS, NC, F_HD, F_UP, F_MO,RWRI,LWRI,RH0,LH0


def source_hidden(hidden, support, n):
    """S cells -> source (frame,joint) set H. Public anchors stay untouched."""
    cells = hidden.copy()
    cells[F_UP] |= cells[F_HD]; cells[F_MO] |= cells[F_HD]
    out=np.zeros((n,103),bool)
    for c,joints in OWNED.items():
        frames={s for b in np.flatnonzero(cells[c]) for k in (2*b,2*b+1) for s in support[k]}
        if frames: out[np.ix_(sorted(frames),joints)]=True
    # The hand observation gate reads its BODY wrist. Unlike ASM's 114-joint
    # layout, PLctx owns these wrists in LOC. Include that dependency in H.
    for wr,base in ((RWRI,RH0),(LWRI,LH0)):
        out[:,base:base+21] |= out[:,wr,None]
    return out


def resampled_hidden(hidden, support):
    return np.stack([hidden[s].any(0) if s else np.zeros(hidden.shape[1],bool) for s in support])


def state_visible(hidden, output_hidden, valid, drop):
    visible=valid[None]&~hidden&~drop
    for c in range(NC):
        joints=sorted({j for d in DEPENDS[c] for j in OWNED[d]})
        bad=output_hidden[:,joints].any(-1).reshape(32,2).any(-1)
        visible[c] &= ~bad
    return visible


def path_masks(unsafe, start, size, valid):
    """unsafe B,T includes preprocessing support; target fraction is over TIME."""
    bad=unsafe[:,start:start+size]
    vf=valid[:,start:start+size]
    touched=(bad&vf).any(-1)
    frac=(bad&vf).sum(-1)/vf.sum(-1).clamp_min(1)
    return ~touched, touched & (frac>=.5), touched & (frac<.5)


def mask_report(pack):
    result={}
    for c in range(5):
        for scale,sl in ((1,slice(0,15)),(2,slice(15,22))):
            result[f'{c}/{scale}']={k:float(pack[k][:,c,sl].float().mean())
                 for k in ('visible','target','drop','present')}
            result[f'{c}/{scale}']['visible_present']=float((pack['visible'][:,c,sl]&pack['present'][:,c,sl]).float().mean())
    return result
