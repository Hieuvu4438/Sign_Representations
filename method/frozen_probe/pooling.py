"""Explicit offline pooling controls; no learned corpus-level projection."""
import hashlib
import numpy as np


def valid_windows(features,centers):
    x=np.asarray(features,dtype=np.float64);t=np.asarray(centers,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<1 or x.shape[1]<1 or t.shape!=(len(x),):raise ValueError('Invalid windows/timeline shape')
    if not np.isfinite(x).all() or not np.isfinite(t).all() or (len(t)>1 and not np.all(np.diff(t)>0)):
        raise ValueError('Nonfinite or unordered window input')
    return x,t


def activity_mean(features,raw_activity):
    x=np.asarray(features,dtype=np.float64);a=np.asarray(raw_activity,dtype=np.float64)
    if x.ndim!=2 or len(x)<1 or a.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError('Invalid activity/window inputs')
    weights=np.clip(a,0,1).max(axis=1);total=float(weights.sum());fallback=total<=1e-12
    return (x.mean(axis=0) if fallback else (x*weights[:,None]).sum(axis=0)/total),{
        'fallback_mean':fallback,'weight_sum':total,'zero_weight_windows':int((weights==0).sum())}


def ordered_bins(features,centers,bins=4):
    x,t=valid_windows(features,centers)
    if type(bins)is not int or bins<1:raise ValueError('Invalid bin count')
    if len(t)==1:ids=np.zeros(1,dtype=int)
    else:ids=np.minimum(np.floor(bins*(t-t[0])/(t[-1]-t[0])).astype(int),bins-1)
    global_mean=x.mean(axis=0)
    result=np.stack([x[ids==b].mean(axis=0) if np.any(ids==b) else global_mean for b in range(bins)])
    return result,{'bin_counts':[int((ids==b).sum()) for b in range(bins)],'bin_ids':ids.tolist(),
                   'empty_bin_rule':'clip_mean','timeline_rule':'relative_window_center_range'}


def gaussian_projection(input_dim,output_dim,seed=1337):
    if min(input_dim,output_dim)<1:raise ValueError('Invalid projection dimension')
    return np.random.default_rng(seed).normal(0,1/np.sqrt(output_dim),(input_dim,output_dim))


def source_permutation(source_sha256,bins=4,seed=2718):
    if len(source_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('Expected source SHA256, no label/filename key')
    payload=f'{seed}:{source_sha256}'.encode('ascii')
    local_seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')
    return np.random.default_rng(local_seed).permutation(bins)


def pool_controls(features,centers,raw_activity,source_sha256,projection):
    x,t=valid_windows(features,centers)
    bins,metadata=ordered_bins(x,t,4)
    if projection.shape!=(4*x.shape[1],x.shape[1]) or not np.isfinite(projection).all():
        raise ValueError('Projection must preserve output dimension')
    weighted,activity_metadata=activity_mean(x,raw_activity)
    permutation=source_permutation(source_sha256)
    mean=x.mean(axis=0)
    controls={'mean':mean,'max':x.max(axis=0),'activity':weighted,'ordered':bins.reshape(-1)@projection,
              'shuffled':bins[permutation].reshape(-1)@projection,
              'mean_projected':np.tile(mean,4)@projection}
    if any(not np.isfinite(value).all() for value in controls.values()):raise ValueError('Nonfinite pooled vector')
    return controls,{'temporal':metadata,'activity':activity_metadata,'bin_permutation':permutation.tolist()}
