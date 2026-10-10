"""Seeded RGB-only tube occlusion for frozen robustness diagnostics."""
import math
import torch


def occlude_clip(video, fraction, seed=42, max_fraction=.4):
    if video.ndim != 4 or not 0 <= fraction <= max_fraction <= 1:
        raise ValueError('Expected [C,T,H,W] video and valid occlusion fraction')
    if fraction == 0:
        return video.clone(),{'rectangle':None,'actual_area_fraction':0.}
    _,_,h,w = video.shape
    margin = math.sqrt(max_fraction)/2
    center = margin+(1-2*margin)*torch.rand(2,generator=torch.Generator().manual_seed(seed))
    bh,bw = max(1,int(round(h*math.sqrt(fraction)))),max(1,int(round(w*math.sqrt(fraction))))
    y = min(h-bh,max(0,int(round(float(center[0])*h-bh/2))))
    x = min(w-bw,max(0,int(round(float(center[1])*w-bw/2))))
    output = video.clone()
    # Zero in ImageNet-normalized coordinates is the RGB normalization mean.
    output[:,:,y:y+bh,x:x+bw] = 0
    return output,{'rectangle':[x,y,x+bw,y+bh],'actual_area_fraction':bh*bw/(h*w),'all_frames':True}
