"""Manual geometry, train/query joins, ties and unequal-group uncertainty oracles."""
import copy
import math
import numpy as np

from msasl_geometry_eval import frozen_geometry, score_frozen_dictionary, validate_dictionary, EVAL_MODES


def reject(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Expected invalid input rejection')


def fixture():
    items = [dict(sample_id='s0', class_id=0, split='train', signer_id=0, parent_id='s0'),
             dict(sample_id='s1', class_id=1, split='train', signer_id=0, parent_id='s1')]
    for i, label in enumerate((0, 0, 1)):
        items.append(dict(sample_id='v'+str(i), class_id=label, split='val', signer_id=1 if i < 2 else 2, parent_id='v'+str(i)))
    d=dict(class_ids=[0, 1], support_seeds=[1701, 1702, 1703], query_ids=['v0','v1','v2'],
        supports={str(s): ['s0','s1'] for s in (1701,1702,1703)})
    return items, d


def run_tests():
    v=np.array([[2., 3.], [4., 1.]], dtype=np.float64);mu=np.array([1., 1.])
    bases={'R': np.eye(2), 'T': np.array([[0.,1.],[1.,0.]])}
    spectra={'R': np.array([1.,4.]), 'T': np.array([4.,9.])}
    projected=frozen_geometry(v,mu,bases,spectra)
    expected={'RR': [[1.,1.],[3.,0.]], 'TR': [[.5,2.],[1.5,0.]],
        'RT': [[.5,2/3],[1.5,0.]], 'TT': [[1/3,1.],[1.,0.]],
        'raw': [[2.,3.],[4.,1.]], 'centered': [[1.,2.],[3.,0.]]}
    for m, rows in expected.items():
        manual=np.array([[a/math.sqrt(a*a+b*b),b/math.sqrt(a*a+b*b)] for a,b in rows])
        assert np.max(np.abs(projected[m]-manual)) < 1e-14, m
    reject(lambda: frozen_geometry(v,np.array([2.,3.]),bases,spectra))
    reject(lambda: frozen_geometry(v.astype(np.float32),mu,bases,spectra))
    broken=copy.deepcopy(bases);broken['R'][0,0]=2
    reject(lambda: frozen_geometry(v,mu,broken,spectra))
    broken=copy.deepcopy(spectra);broken['T'][1]=float('nan')
    reject(lambda: frozen_geometry(v,mu,bases,broken))
    items,d=fixture();validate_dictionary(items,d)
    for field,value in (('signer_id',0),('parent_id','s0'),('split','train')):
        broken=copy.deepcopy(items);broken[-1][field]=value
        reject(lambda: validate_dictionary(broken,d))
    broken=copy.deepcopy(d);broken['supports']['1701']=['s1','s0']
    reject(lambda: validate_dictionary(items,broken))
    reject(lambda: validate_dictionary(items+items[:1],d))
    reject(lambda: validate_dictionary(items[:-1],d))
    broken=copy.deepcopy(d);broken['query_ids'].append('v2')
    reject(lambda: validate_dictionary(items,broken))
    # All scores tie: ascending class IDs; unequal query class support separates micro/macro.
    z=np.tile(np.array([[1.,0.]]),(5,1));p={m:z.copy() for m in EVAL_MODES}
    arrays,meta,s=score_frozen_dictionary(p,items,d)
    assert arrays['ranks.npy'].shape==(54,2) and np.array_equal(arrays['ranks.npy'],np.tile([0,1],(54,1)))
    assert s['metrics']['raw']['1701']['micro']['top1']==2/3
    assert s['metrics']['raw']['1701']['macro']['top1']==.5
    assert s['numerical_diagnostics']['raw']['1701']['exact_adjacent_tie_pairs']==3
    assert s['numerical_diagnostics']['raw']['1701']['minimum_adjacent_score_gap']==0
    assert not s['primary']['passed'] and s['primary']['delta_DCG']==0
    # RR correct all; TR wrong on A only. A has two positive query deltas; B zero.
    for m in ('RR','TR'):
        p[m]=np.array([[1.,0.],[-1.,0.],[1.,0.],[1.,0.],[-1.,0.]])
    p['TR'][2:]=np.array([[-1.,0.],[-1.,0.],[-1.,0.]])
    _,_,s=score_frozen_dictionary(p,items,d)
    a=1-1/math.log2(3);expected=a*2/3
    assert abs(s['primary']['delta_DCG']-expected)<1e-14
    assert all(abs(x-expected)<1e-14 for x in s['primary']['per_gallery_delta_DCG'].values())
    assert s['primary']['passed']
    # Manual group resampling of A:[a,a], B:[0], including all clips of repeated groups.
    rng=np.random.default_rng(1701);draws=[]
    for _ in range(2000):
        chosen=rng.integers(0,2,size=2).tolist();flat=[]
        for group in chosen:flat.extend([a,a] if group==0 else [0.])
        draws.append(math.fsum(flat)/len(flat))
    assert np.max(np.abs(np.quantile(draws,[.025,.975])-s['primary']['conditional_bootstrap']['interval95_DCG']))<1e-14
    assert abs(s['primary']['delta_DCG']-a/2)>.01  # rejects average of signer means
    broken=copy.deepcopy(p);broken['raw'][0,0]=float('nan')
    reject(lambda: score_frozen_dictionary(broken,items,d))
    return dict(passed=True, groups=['manual_six_saved_geometry_projections',
        'exact_train_query_dictionary_source_signer_joins', 'tied_scores_complete_ranks_micro_macro',
        'fixed_primary_and_unequal_signer_clip_bootstrap'], GPU_hours=0,
        scope='Synthetic software oracles only; no real target features or taskquality')
