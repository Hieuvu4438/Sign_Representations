import copy
import numpy as np
import pytest
from signrel.config import load_config,validate
from signrel.data.manifest import validate_ssl
from signrel.data.synthetic import make_synthetic
from signrel.data.cache import read_cache
from signrel.data.decode import nearest_mapping
from signrel.data.crops import hand_square,in_bounds_fraction,crop_hand


@pytest.mark.parametrize('change',[{'teacher':{'frozen':False}},{'train':{'ddp':True}},{'model':{'hidden_dim':50}},{'loss':{'relation':{'lags':[64]}}},{'loss':{'relation':{'stream_pairs':[['LH','BODY']]}}},{'train':{'typo':True}}])
def test_reject_invalid_config(change):
    from signrel.config import merge
    cfg=load_config('configs/smoke_a4.yaml').values
    with pytest.raises(ValueError): validate(merge(cfg,change))


def test_semantic_ssl_manifest_rejection():
    row={'schema_version':1,'video_id':'v','relative_path':'video.mp4','source_recording_id':'r','source_group_id':'r','split':'ssl_train'}
    assert validate_ssl(row)==row
    for key in ['gloss','class_id','signer_id','caption']:
        with pytest.raises(ValueError): validate_ssl({**row,key:'hidden_label'})
    with pytest.raises(ValueError): validate_ssl({**row,'other':{'translation':'word'}})


def test_cache_provenance_and_nonpickle(tmp_path):
    make_synthetic(tmp_path,videos=6)
    path=tmp_path/'caches/synthetic_0000.npz'
    a,m=read_cache(path,{'teacher_sha256':'synthetic-no-teacher'})
    assert a['appearance'].shape[-1]==384
    with pytest.raises(ValueError): read_cache(path,{'teacher_sha256':'different'})
    with path.open('ab') as f: f.write(b'corruption')
    with pytest.raises(ValueError): read_cache(path)


def test_vfr_irregular_nearest_and_tolerance():
    indices,valid=nearest_mapping([0,33,90,250],[0,40,80,120,160,200,240])
    assert indices.tolist()==[0,1,2,2,2,3,3]
    assert valid.tolist()==[True,True,True,True,False,False,True]


def test_square_geometry_and_mean_padding():
    points=np.stack([np.linspace(0,.2,21),np.linspace(.2,.4,21)],axis=1)
    box=hand_square(points,200,100)
    assert box[2]-box[0]==box[3]-box[1]
    assert .5<in_bounds_fraction(box,200,100)<1
    crop=crop_hand(np.zeros((100,200,3),np.uint8),box)
    assert crop.shape==(3,224,224) and crop.isfinite().all()
