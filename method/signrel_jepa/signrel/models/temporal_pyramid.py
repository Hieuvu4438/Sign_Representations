import torch


def temporal_targets(relations, valid, kernel):
    """Only complete windows with both endpoints valid contribute; no invalid averaging."""
    if kernel < 1 or kernel > relations.shape[1]:
        raise ValueError("Temporal kernel outside token clip")
    z = relations.unfold(1,kernel,1).mean(-1)
    v = valid.unfold(1,kernel,1).all(-1)
    return z.masked_fill(~v[...,None],0), v
