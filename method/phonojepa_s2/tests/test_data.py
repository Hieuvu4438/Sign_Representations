import sys
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2 import features as Fe
from s2.data.dataset import Cfg, build_pair, collate
from s2.data.cache import ClipCache
from s2.joints import NC, J, ANCHORS, DUPLICATES, TORSO


def clip():
    rng = np.random.default_rng(42)
    xy = rng.normal(150, 20, (100, J, 2)).astype('float32')
    xy[:, ANCHORS[0]] = [200, 100]
    xy[:, ANCHORS[1]] = [100, 100]
    return xy, np.ones((100, J), np.float32), 30.0


def pair(xy, conf, fps, kind='near', seed=1):
    return build_pair(xy, conf, fps, np.random.default_rng(seed), Cfg(), kind)


@pytest.mark.parametrize('kind', ['near', 'far'])
def test_raw_mask_perturbation(kind):
    xy, conf, fps = clip()
    for seed in range(20):
        a = pair(xy, conf, fps, kind, seed)
        assert a['hidden'][TORSO].any()
        assert not a['raw_scrub'][:, ANCHORS].any()
        for i, j in DUPLICATES:
            np.testing.assert_array_equal(a['raw_scrub'][:, i], a['raw_scrub'][:, j])
        changed = np.where(a['raw_scrub'][..., None], 1e6, xy)
        scores = np.where(a['raw_scrub'], .123, conf)
        b = pair(changed, scores, fps, kind, seed)
        for key in ['s_xy', 's_obs', 's_conf', 'visible', 'target', 'drop']:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        def inputs(v):
            v = collate([{'near':v, 'far':v, 'flipped':False}])['near']
            f = Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'],v['s_aff'])
            return Fe.tokens(f, v['valid_f'])[0]
        for aa, bb in zip(inputs(a), inputs(b)):
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)


def test_torso_dimensions_missing_and_no_cross_bin_delta():
    xy, conf, fps = clip()
    v = collate([{'near': pair(xy,conf,fps), 'far':pair(xy,conf,fps), 'flipped':False}])['near']
    feats = Fe.frame_features(v['s_xy'],v['s_obs'],v['s_conf'],v['s_jh'])
    ins, present = Fe.tokens(feats,v['valid_f'])
    assert len(ins) == NC == 8 and ins[7].shape == (1,32,50)
    assert all(torch.isfinite(x).all() for x in ins)
    # Changing a different bin must not change a torso token (anchors held fixed).
    anchor = Fe.anchors(v['s_xy'],v['s_obs'])
    before = feats[7][0][:,10:12].clone()
    x = v['s_xy'].clone(); x[:,8:10,7:9] += 99
    after = Fe.frame_features(x,v['s_obs'],v['s_conf'],v['s_jh'],anchor=anchor)[7][0][:,10:12]
    torch.testing.assert_close(before, after)


def test_no_shoulders_fallback_independent_of_hidden_points():
    xy = torch.rand(1,64,J,2); obs = torch.ones(1,64,J,dtype=torch.bool)
    obs[:,:,ANCHORS] = False
    a = Fe.anchors(xy,obs)
    xy[:,:,7:9] += 1e5
    b = Fe.anchors(xy,obs)
    for x,y in zip(a,b): torch.testing.assert_close(x,y)


def test_real_training_cache():
    p = ROOT.parent/'phonojepa/runtime/cache/aslc002_train.npz'
    assert p.exists(), 'The agreed 3,000-clip cache is required for this integration check'
    c = ClipCache(p)
    assert len(c) == 3000
    for i in [0, 33, 271, 2999]:
        xy, conf, fps = c.clip(i)
        v = pair(xy,conf,fps)
        assert np.isfinite(v['s_xy']).all()
        assert not (v['target'] & v['visible']).any()


def test_zero_weight_interpolation_does_not_read_next_missing_frame():
    from s2.data.window import resample
    x=np.array([[[1.,2.]],[[np.nan,np.nan]]],np.float32)
    c=np.array([[1.],[0.]],np.float32)
    y,score,_=resample(x,c,np.array([0]),np.array([1]),np.array([0.]),np.array([True]))
    np.testing.assert_array_equal(y, [[[1.,2.]]]);assert score[0,0]==1


@pytest.mark.parametrize('fps',[15.,25.,29.97,60.])
def test_exported_forbidden_support_and_source_time(fps):
    from s2.data.window import window_positions,resample,random_start
    xy,conf,_=clip()
    cfg=Cfg(speed=(1.17,1.17),noise=0.,hand_drop=0.)
    v=build_pair(xy,conf,fps,np.random.default_rng(6),cfg,'near')
    rng=np.random.default_rng(6);speed=rng.uniform(1.17,1.17)
    start=random_start(len(xy),fps,64,speed,rng)
    i0,i1,w,valid=window_positions(len(xy),fps,64,start,speed)
    expected=resample(xy,conf,i0,i1,w,valid,flag=v['raw_scrub'])[2]
    np.testing.assert_array_equal(v['forbidden'],expected)
    np.testing.assert_allclose(np.diff(v['times']),1.17/min(fps,25),rtol=1e-5)
    assert not v['forbidden'][~v['valid_f']].any()
