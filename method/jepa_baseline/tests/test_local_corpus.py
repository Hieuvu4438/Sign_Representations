import json
import pytest
from signrep_baseline.data import deduplicate_local_manifest, source_identity
from signrep_baseline.cli import held_out_contract


def write(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_local_identity_is_namespaced_and_real_youtube_identity_is_preserved():
    assert source_identity({'dataset':'asl_citizen','source_id':'v1'}) == 'asl_citizen:v1'
    assert source_identity({'dataset':'other','source_id':'v1'}) != 'asl_citizen:v1'
    assert source_identity({'youtube_id':'https://youtu.be/abc'}) == 'youtube:abc'
    with pytest.raises(ValueError):
        source_identity({'video_id':'unscoped'})


def test_local_audit_accepts_training_only_and_rejects_held_out_leakage(tmp_path):
    train, exclusion, output = [tmp_path / name for name in ['train.jsonl','held_out.jsonl','filtered.jsonl']]
    row = {'video_id':'t','source_id':'t','dataset':'asl_citizen','source_split':'train',
           'split':'pretrain','video':'train.mp4','video_sha256':'train-content'}
    held = {'video_id':'v','source_id':'v','dataset':'asl_citizen','split':'val',
            'video':'val.mp4','video_sha256':'val-content'}
    reference = tmp_path/'official_train.jsonl'
    write(reference,[{**row,'split':'train'}])
    write(train,[row]);write(exclusion,[held])
    assert deduplicate_local_manifest(train,exclusion,output,reference)['audit_kind'] == 'local_train_only'
    for mutation in [{'source_id':'v'}, {'video':'val.mp4'}, {'video_sha256':'val-content'}]:
        write(train,[{**row,**mutation}])
        with pytest.raises(ValueError,match='overlaps|reference'):
            deduplicate_local_manifest(train,exclusion,output,reference)
    write(train,[{**row,'source_split':'test'}])
    with pytest.raises(ValueError,match='source_split'):
        deduplicate_local_manifest(train,exclusion,output,reference)


def test_local_validation_guard_rejects_relabelled_source(tmp_path):
    train, val = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    write(train,[{'video_id':'a','source_id':'same','dataset':'asl_citizen','video':'a.mp4','split':'train'}])
    write(val,[{'video_id':'b','source_id':'same','dataset':'asl_citizen','video':'b.mp4','split':'val'}])
    with pytest.raises(ValueError,match='overlap'):
        held_out_contract(train,val)
