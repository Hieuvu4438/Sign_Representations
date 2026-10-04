"""Deterministic low-label controls with train-only fitting."""
import numpy as np


def normalize_rows(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Expected finite nonempty matrix')
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if (norm<=1e-12).any():raise ValueError('Zero clip/prototype vector cannot be normalized')
    return x/norm


def labels_onehot(labels,classes):
    y=np.asarray(labels)
    if y.ndim!=1 or not len(y) or not np.issubdtype(y.dtype,np.integer) or classes<1 or (y<0).any() or (y>=classes).any():
        raise ValueError('Invalid class labels')
    if set(y.tolist())!=set(range(classes)):raise ValueError('Every class must have train support')
    return np.eye(classes,dtype=np.float64)[y]


def fit_ridge(x,labels,classes,alpha):
    x=np.asarray(x,dtype=np.float64);y=labels_onehot(labels,classes)
    if x.ndim!=2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('Invalid ridge inputs')
    mean_x=x.mean(axis=0);mean_y=y.mean(axis=0);xc=x-mean_x;yc=y-mean_y
    if len(x)<x.shape[1]:
        coef=xc.T@np.linalg.solve(xc@xc.T+alpha*np.eye(len(x)),yc)
    else:
        coef=np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@yc)
    intercept=mean_y-mean_x@coef
    if not np.isfinite(coef).all():raise ValueError('Nonfinite ridge solution')
    return {'coef':coef,'intercept':intercept,'alpha':alpha,'classes':classes,'fit_samples':len(x)}


def predict_ridge(model,x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not len(x) or x.shape[1]!=model['coef'].shape[0] or not np.isfinite(x).all():
        raise ValueError('Invalid query features')
    return x@model['coef']+model['intercept']


def fit_prototypes(x,labels,classes):
    x=normalize_rows(x);labels_onehot(labels,classes);y=np.asarray(labels)
    if len(x)!=len(y):raise ValueError('Support feature/label mismatch')
    return normalize_rows(np.stack([x[y==c].mean(axis=0) for c in range(classes)]))


def nearest_class_scores(support,labels,queries,classes):
    """Cosine gallery matching collapsed to one maximum score per class."""
    support=normalize_rows(support);queries=normalize_rows(queries)
    labels_onehot(labels,classes);y=np.asarray(labels)
    if len(support)!=len(y) or support.shape[1]!=queries.shape[1]:raise ValueError('Gallery/query mismatch')
    similarities=queries@support.T
    return np.stack([similarities[:,y==c].max(axis=1) for c in range(classes)],axis=1)
