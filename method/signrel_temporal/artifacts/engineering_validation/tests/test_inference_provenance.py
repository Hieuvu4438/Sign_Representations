import copy
import json
import numpy as np
import pytest
import torch
from signrel.config import load_config
from signrel.data.synthetic import make_synthetic
from signrel.data.cache import read_cache
from signrel.data.windows import WindowDataset
from signrel.models.encoder import initialized_model
from signrel.eval.embed import embed_arrays, embed_cached, restore_encoder
from signrel.data.provenance import preprocessing_fingerprint, inference_expectations
from signrel.utils import read_jsonl


def inference_fixture(tmp_path):
    make_synthetic(tmp_path/'data', videos=6, frames=32)
    cfg=load_config('configs/smoke_a1.yaml').values
    model=initialized_model(cfg['model'],17).eval()
    path=tmp_path/'encoder.pt'
    state={'encoder_state':{k:v for k,v in model.state_dict().items() if not k.startswith('heads.')},
           'config_resolved':cfg,'teacher_sha256':'synthetic-no-teacher','preprocess_fingerprint':'synthetic-v1'}
    torch.save(state,path)
    return cfg,model,path,tmp_path/'data/caches/synthetic_0000.npz'


def test_dedicated_encoder_has_no_decoders_and_matches_full(tmp_path,monkeypatch):
    cfg,full,path,cache=inference_fixture(tmp_path)
    arrays,_=read_cache(cache)
    expected=embed_arrays(arrays,full)['embedding']
    def forbidden(*args,**kwargs): raise AssertionError('Inference evaluated discarded decoders')
    monkeypatch.setattr(full.heads,'forward',forbidden)
    assert np.array_equal(embed_arrays(arrays,full)['embedding'],expected)
    dedicated,_,_=restore_encoder(path)
    assert not hasattr(dedicated,'heads')
    actual=embed_cached(cache,path)
    assert actual['status']=='ok' and np.array_equal(actual['embedding'],expected)
    assert actual['decoder_prediction_norm'] is None


@pytest.mark.parametrize('field,value',[('preprocess_fingerprint','changed-crop-policy'),('teacher_sha256','changed-teacher')])
def test_public_inference_rejects_incompatible_cache_metadata(tmp_path,field,value):
    _,_,path,cache=inference_fixture(tmp_path)
    metadata_path=cache.with_suffix('.json')
    metadata=json.loads(metadata_path.read_text());metadata[field]=value
    metadata_path.write_text(json.dumps(metadata))
    # Numerical bytes and their hash remain valid; provenance alone must fail.
    read_cache(cache)
    with pytest.raises(ValueError,match='Cache fingerprint mismatch'):
        embed_cached(cache,path)


def test_training_rejects_manifest_hash_and_mixed_preprocessing(tmp_path):
    cfg,_,_,_=inference_fixture(tmp_path)
    manifest=tmp_path/'data/cache_manifest.jsonl'
    rows=read_jsonl(manifest)
    original=manifest.read_text()
    rows[0]['cache_sha256']='incorrect'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(ValueError,match='manifest byte hash'):
        WindowDataset(manifest,cfg,17)
    manifest.write_text(original)
    meta_path=__import__('pathlib').Path(rows[0]['cache_path']).with_suffix('.json')
    metadata=json.loads(meta_path.read_text());metadata['preprocess_fingerprint']='different'
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='mixes preprocessing'):
        WindowDataset(manifest,cfg,17)


def test_real_recipe_fingerprint_changes_and_legacy_lock_is_required(tmp_path):
    cfg=copy.deepcopy(load_config('configs/pilot_a4.yaml').values)
    lock={'teacher':{'checkpoint_sha256':'teacher-a'},'perception':{'pose':{'sha256':'pose-a'},'hands':{'sha256':'hands-a'}},'mediapipe_version':'pinned-version'}
    baseline=preprocessing_fingerprint(cfg,lock)
    cfg['preprocess']['hand_margin']=2
    assert preprocessing_fingerprint(cfg,lock)!=baseline
    cfg['paths']['sources_lock']=str(tmp_path/'missing.json')
    with pytest.raises(ValueError,match='Legacy checkpoint needs'):
        inference_expectations({},cfg)
