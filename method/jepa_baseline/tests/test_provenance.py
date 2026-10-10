import json
from pathlib import Path
import numpy as np
import pytest
from signrep_baseline.cache import sha256
from signrep_baseline.cli import held_out_contract, cached_feature_rows
from signrep_baseline.recognition import save_class_distribution, load_class_distribution
import torch


def test_relabeling_same_video_does_not_bypass_split_guard(tmp_path):
    gallery, query = tmp_path / 'train.jsonl', tmp_path / 'val.jsonl'
    gallery.write_text(json.dumps({'video_id':'a','video':'same.mp4','split':'train'})+'\n')
    query.write_text(json.dumps({'video_id':'b','video':'same.mp4','split':'val'})+'\n')
    with pytest.raises(ValueError, match='overlap in video'):
        held_out_contract(gallery, query)


def test_class_table_requires_training_membership_and_encoder_pin(tmp_path):
    manifest, table = tmp_path / 'train.jsonl', tmp_path / 'phi.npz'
    manifest.write_text(''.join(json.dumps({'video_id':str(i),'label':i,'split':'train'})+'\n' for i in range(3)))
    save_class_distribution(table, [[1.,.01],[1.,-.01],[1.,0.]], [0,1,2], 3, manifest,
                            encoder_checkpoint_sha256='encoder1')
    load_class_distribution(table, manifest, 'encoder1')
    with pytest.raises(ValueError, match='different initial encoder'):
        load_class_distribution(table, manifest, 'encoder2')
    manifest.write_text(manifest.read_text().replace('"train"','"test"'))
    with pytest.raises(ValueError, match='only train'):
        load_class_distribution(table, manifest)


def test_frozen_features_reject_mixed_encoder_or_changed_content(tmp_path):
    files = []
    for i in range(2):
        path = tmp_path / f'{i}.npz'
        np.savez(path, features=np.ones((1,768)), metadata_json=np.array(json.dumps({
            'schema':'signrep_features_v1','stride':2,'encoder_checkpoint_sha256':f'encoder{i}',
            'video_sha256':f'video{i}'})))
        files.append(path)
    manifest = tmp_path / 'features.jsonl'
    manifest.write_text(''.join(json.dumps({'video_id':str(i),'split':'train','features':str(p)})+'\n' for i,p in enumerate(files)))
    with pytest.raises(ValueError, match='different encoder'):
        cached_feature_rows(manifest,'train')
    manifest.write_text(json.dumps({'video_id':'0','split':'train','features':str(files[0]),
                                    'features_sha256':'changed-content'})+'\n')
    with pytest.raises(ValueError, match='cache changed'):
        cached_feature_rows(manifest,'train')
