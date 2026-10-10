import numpy as np


def boxes_from_landmarks(xy, confidence, joint_type, threshold=0.5,
                         margins=(0.35,0.35,0.20,0.10), max_gap=3, smooth=0.6):
    t = len(xy)
    boxes = np.zeros((t,4,4),np.float32)
    conf = np.zeros((t,4),np.float32)
    for a in range(4):
        joints = np.flatnonzero(joint_type==a)
        if not len(joints):
            continue
        for frame in range(t):
            scores = confidence[frame,joints]
            valid = (scores>=threshold) & np.isfinite(xy[frame,joints]).all(-1)
            if valid.sum()<2:
                continue
            p = xy[frame,joints[valid]]
            lo,hi = p.min(0),p.max(0)
            size = np.maximum(hi-lo,0.02)
            center = (lo+hi)/2
            size *= 1+2*margins[a]
            boxes[frame,a] = np.clip(np.r_[center-size/2,center+size/2],0,1)
            conf[frame,a] = float(scores[valid].mean()*valid.mean())
        valid_indices = np.flatnonzero(conf[:,a]>=threshold)
        for left,right in zip(valid_indices[:-1],valid_indices[1:]):
            gap = right-left-1
            if 0<gap<=max_gap:
                for j in range(left+1,right):
                    alpha = (j-left)/(right-left)
                    boxes[j,a] = (1-alpha)*boxes[left,a]+alpha*boxes[right,a]
                    conf[j,a] = min(conf[left,a],conf[right,a])
        prev = None
        for frame in range(t):
            if conf[frame,a]<threshold:
                prev = None
            elif prev is not None:
                boxes[frame,a] = smooth*boxes[frame,a]+(1-smooth)*prev
                prev = boxes[frame,a].copy()
            else:
                prev = boxes[frame,a].copy()
    return boxes,conf


def motion_from_landmarks(xy, confidence, joint_type, threshold=0.5):
    out = np.zeros((len(xy),4),np.float32)
    for a in range(4):
        ids = np.flatnonzero(joint_type==a)
        ok = (confidence[1:,ids]>=threshold)&(confidence[:-1,ids]>=threshold)
        delta = np.linalg.norm(xy[1:,ids]-xy[:-1,ids],axis=-1)
        out[1:,a] = np.where(ok,delta,0).sum(-1)/np.maximum(ok.sum(-1),1)
    # Normalize by upper-body size; fallback records image-relative movement.
    body = np.flatnonzero(joint_type==3)
    if len(body)>=2:
        size = np.linalg.norm(xy[:,body[0]]-xy[:,body[1]],axis=-1)
        ok = (confidence[:,body[:2]]>=threshold).all(-1)&(size>0.05)
        out[ok] /= size[ok,None]
    return out
