"""H006 fixed-rank, known-family covariance controls; no novel method claim."""
import numpy as np
from participant_removal import leading_basis
from probes import normalize_rows

MODES=('raw','center','random','total_pca','shuffled_pca','residual_pca')


def class_residuals(features,labels):
    x=np.asarray(features,dtype=np.float64)
    if x.ndim!=2 or not len(x) or len(x)!=len(labels) or not np.isfinite(x).all():
        raise ValueError('Invalid residual data')
    residual=x.copy()
    for label in sorted(set(labels)):
        positions=[i for i,c in enumerate(labels) if c==label]
        residual[positions]-=x[positions].mean(axis=0)
    return residual


def fit_geometry(features,labels,rank,seed):
    x=np.asarray(features,dtype=np.float64)
    residual=class_residuals(x,labels)
    mean=x.mean(axis=0)
    centered=x-mean
    residual_u,residual_s=leading_basis(residual,rank)
    total_u,total_s=leading_basis(centered,rank)
    # Independent deterministic streams; no choice of a successful seed.
    shuffled_labels=np.random.default_rng(seed).permutation(labels).tolist()
    shuffled_u,shuffled_s=leading_basis(class_residuals(x,shuffled_labels),rank)
    random_u,_=np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1],rank)),mode='reduced')
    return dict(mean=mean,residual_pca=residual_u,total_pca=total_u,
                shuffled_pca=shuffled_u,random=random_u,rank=rank,fit_count=len(x),
                residual_singular=residual_s,total_singular=total_s,shuffled_singular=shuffled_s,
                shuffled_labels=shuffled_labels)


def apply_geometry(features,fitted,mode):
    if mode not in MODES:raise ValueError('Unknown geometry control')
    if mode=='raw':return normalize_rows(features)
    x=np.asarray(features,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(fitted['mean']):raise ValueError('Feature dimension differs')
    value=x-fitted['mean']
    if mode!='center':
        basis=fitted[mode]
        value-=(value@basis)@basis.T
    return normalize_rows(value)


def disjoint_cohorts(base,target):
    if (set(base['classes'])&set(target['classes'])
            or {r['sample_id'] for r in base['samples']}&{r['sample_id'] for r in target['samples']}
            or {r['source_sha256'] for r in base['samples']}&{r['source_sha256'] for r in target['samples']}):
        raise ValueError('Base and target classes/IDs/source bytes overlap')


def common_feature_recipe(base,target):
    for field in ('checkpoint_sha256','upstream_commit','packages','preprocessing_code_sha256','temporal_code_sha256'):
        if base[field]!=target[field]:raise ValueError('Feature recipe differs: '+field)
    # The only descriptor difference is the separately hash-bound timeline audit.
    expected=('complete_native_PTS_and_FPS_in_hash_bound_VID002_audit',
              'complete native PTS and FPS in hash-bound VID003 audit')
    if (base['contract'].get('temporal_metadata'),target['contract'].get('temporal_metadata'))!=expected:
        raise ValueError('Unregistered temporal audit descriptors')
    a={k:v for k,v in base['contract'].items() if k!='temporal_metadata'}
    b={k:v for k,v in target['contract'].items() if k!='temporal_metadata'}
    if a!=b:raise ValueError('Native inference contracts differ')
