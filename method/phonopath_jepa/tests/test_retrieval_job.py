import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
from types import SimpleNamespace

path=Path(__file__).resolve().parents[1]/'scripts/retrieval_job.py'
spec=importlib.util.spec_from_file_location('retrieval_job_under_test',path);J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)


def setup_shards(tmp_path):
    ids=np.array(['a','b','c']);rows=[dict(sample_id=x,class_id=i%2,gloss=str(i%2),signer_id='s') for i,x in enumerate(ids)]
    source=SimpleNamespace(ids=ids,rows=rows)
    lock=dict(checkpoint_sha256='ck',source_sha256=dict(train='source'))
    return source,lock


def test_durable_shard_resume_and_merge_identity(tmp_path):
    source,lock=setup_shards(tmp_path)
    class Source:
        ids=source.ids;rows=source.rows
        def __len__(self):return len(self.ids)
    source=Source()
    for start,stop in ((0,2),(2,3)):
        values={key:np.full((stop-start,dim),start+1,np.float16) for key,dim in J.DIMS.items()}
        J.atomic_npz(tmp_path/'shards/train'/f'{start:07d}_{stop:07d}.npz',**values,ids=source.ids[start:stop],
                     start=np.array(start),stop=np.array(stop),checkpoint=np.array('ck'),source=np.array('source'))
    # An interrupted uncommitted write never enters the resumed prefix.
    (tmp_path/'shards/train/0000003_0000004.npz.partial').write_bytes(b'partial')
    count,paths=J.completed_shards(tmp_path,'train',lock,source)
    assert count==3 and len(paths)==2
    J.merge(tmp_path,'train',lock,source,paths)
    with np.load(tmp_path/'train.npz') as z:
        np.testing.assert_array_equal(z['ids'],source.ids)
        assert z['grp'].dtype==np.float16 and z['grp'][2,0]==3
        assert json.loads(str(z['metadata']))['limited'] is False


def test_mixed_checkpoint_shard_is_rejected(tmp_path):
    source,lock=setup_shards(tmp_path)
    values={key:np.zeros((3,dim),np.float16) for key,dim in J.DIMS.items()}
    J.atomic_npz(tmp_path/'shards/train/0000000_0000003.npz',**values,ids=source.ids,start=np.array(0),stop=np.array(3),
                 checkpoint=np.array('different'),source=np.array('source'))
    with pytest.raises(ValueError,match='identity'):J.completed_shards(tmp_path,'train',lock,source)
