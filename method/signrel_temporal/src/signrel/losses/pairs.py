"""Sparse, canonical, signed-lag pair table with an independent RNG."""
import torch


def validate_pairs(pairs,shape):
    if pairs.ndim != 2 or pairs.shape[1] != 5 or pairs.dtype != torch.int64:
        raise ValueError('Pair table must be int64 [N,5]')
    if len(pairs):
        bounds = torch.tensor([shape[0],shape[1],shape[2],shape[1],shape[2]],device=pairs.device)
        if ((pairs<0)|(pairs>=bounds)).any():
            raise ValueError('Pair index out of bounds')
        if len(torch.unique(pairs,dim=0)) != len(pairs):
            raise ValueError('Duplicate pair constraints')


def build_pairs(valid,quality,masked,source_frame_index,padding,lags,mode,cap,generator,threshold=.5):
    if valid.shape != quality.shape or masked.shape != valid.shape or valid.shape[-1] != 2:
        raise ValueError('Pair observation shape mismatch')
    b,t,_ = valid.shape
    if source_frame_index.shape != (b,t) or padding.shape != (b,t):
        raise ValueError('Pair correspondence shape mismatch')
    if any(type(lag) is not int or abs(lag)>=t for lag in lags) or cap<1:
        raise ValueError('Invalid lag/cap')
    if mode == 'intra' and any(lag<=0 for lag in lags):
        raise ValueError('Intra-hand relation must avoid diagonal/transpose duplication')
    if mode == 'cross_zero' and lags != [0]:
        raise ValueError('Zero-lag control requires lag 0')
    observed = (valid&(quality>=threshold)).detach().cpu()
    masked,indices,padding = masked.detach().cpu(),source_frame_index.detach().cpu(),padding.detach().cpu()
    result = []
    for i in range(b):
        bins = {}
        for lag in sorted(lags):
            candidates = []
            streams = [(0,0),(1,1)] if mode=='intra' else [(0,1)]
            for a,c in streams:
                for left in range(t):
                    right = left+lag
                    if not 0<=right<t or padding[i,left] or padding[i,right]: continue
                    if not observed[i,left,a] or not observed[i,right,c]: continue
                    if not (masked[i,left,a] or masked[i,right,c]): continue
                    if indices[i,left]<0 or indices[i,right]<0: continue
                    if lag != 0 and indices[i,left] == indices[i,right]: continue
                    candidates.append([i,left,a,right,c])
            if candidates:
                order = torch.randperm(len(candidates),generator=generator).tolist()
                bins[lag] = [candidates[j] for j in order]
        chosen = []
        cursor = {lag:0 for lag in bins}
        while len(chosen)<cap:
            active = False
            for lag in bins:
                if cursor[lag]<len(bins[lag]) and len(chosen)<cap:
                    chosen.append(bins[lag][cursor[lag]])
                    cursor[lag]+=1
                    active=True
            if not active: break
        result.extend(chosen)
    pairs = torch.tensor(result,dtype=torch.int64,device=valid.device).reshape(-1,5)
    validate_pairs(pairs,valid.shape)
    return pairs
