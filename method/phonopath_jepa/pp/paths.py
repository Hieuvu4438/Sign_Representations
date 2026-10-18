"""Phonological path families; ownership remains explicit down to each path."""
from dataclasses import dataclass
import torch
from pp.base.joints import RH0,LH0,RWRI,LWRI,RELB,LELB,RSHO,LSHO,NOSE_TIP

CHANNELS = ('D_LOC','D_CONF','N_LOC','N_CONF','REL')

@dataclass
class Path:
    value: torch.Tensor                 # B,T,d
    observed: torch.Tensor              # B,T
    confidence: torch.Tensor            # B,T
    unsafe: torch.Tensor                # B,T (reads hidden source)
    coverage: torch.Tensor              # B,T genuine observations before interpolation
    basepoint: bool = False
    angle: bool = False


def families(p, obs, conf, unsafe, original_obs=None):
    original_obs=obs if original_obs is None else original_obs
    def make(v, joints, basepoint=False, angle=False):
        return Path(v, obs[:,:,joints].all(-1), conf[:,:,joints].mean(-1),
                    unsafe[:,:,joints].any(-1),original_obs[:,:,joints].float().mean(-1),basepoint,angle)
    out = []
    for base,wr,el,sh in ((RH0,RWRI,RELB,RSHO),(LH0,LWRI,LELB,LSHO)):
        loc = [make(p[:,:,wr], [wr], True),
               make(torch.cat([p[:,:,el]-p[:,:,sh],p[:,:,wr]-p[:,:,el]],-1),[wr,el,sh])]
        h = p[:,:,base:base+21]
        axis = h[:,:,9]-h[:,:,0]
        length = axis.norm(dim=-1).clamp_min(.1)
        ex = axis / axis.norm(dim=-1).clamp_min(1e-6)[...,None]
        ey = torch.stack([-ex[...,1],ex[...,0]],-1)
        rel = (h-h[:,:,:1])/length[...,None,None]
        hand = torch.stack([(rel*ex[:,:,None]).sum(-1),(rel*ey[:,:,None]).sum(-1)],-1)
        config = []
        for finger in range(5):
            chain = [0]+list(range(1+4*finger,5+4*finger))
            edges = hand[:,:,chain[1:]]-hand[:,:,chain[:-1]]
            joints = sorted({base+j for j in chain} | {base,base+9})
            config.append(make(edges.flatten(-2),joints))
        transverse = h[:,:,5]-h[:,:,17]
        angle = torch.atan2(transverse[...,1],transverse[...,0])
        a,b = h[:,:,5]-h[:,:,0],h[:,:,17]-h[:,:,0]
        signed = (a[...,0]*b[...,1]-a[...,1]*b[...,0]) / length.square()
        config.append(make(torch.stack([angle,signed],-1),[base,base+5,base+9,base+17],angle=True))
        ap = torch.stack([(hand[:,:,4]-hand[:,:,8]).norm(dim=-1),
                          (hand[:,:,4]-hand[:,:,12]).norm(dim=-1)],-1)
        config.append(make(ap,[base,base+4,base+8,base+9,base+12]))
        out.extend([loc,config])
    n = p[:,:,NOSE_TIP]; wd,wn = p[:,:,RWRI],p[:,:,LWRI]
    mirror_n = wn * wn.new_tensor([-1.,1.])
    out.append([make(wd-n,[RWRI,NOSE_TIP],True),
                make(p[:,:,RH0+8]-n,[RH0+8,NOSE_TIP],True),
                make(wn-n,[LWRI,NOSE_TIP],True),
                make(torch.cat([wd+mirror_n,wd-mirror_n],-1),[RWRI,LWRI]),
                make(wd-wn,[RWRI,LWRI],True)])
    return out


def signature_dims(time_channel=False):
    out=[]
    for ds in ([2,4],[8]*5+[2,2],[2,4],[8]*5+[2,2],[2,2,2,4,2]):
        out.append([((d+int(time_channel))*(d+int(time_channel)+1)//2) for d in ds])
    return out


def token_dims(time_channel=False, path_means=False):
    dims=signature_dims(time_channel)
    base=[2,0,2,0,8]; counts=[2,7,2,7,5]
    means=[6,44,6,44,12]
    return [sum(d)+b+n+2+(m if path_means else 0) for d,b,n,m in zip(dims,base,counts,means)]
