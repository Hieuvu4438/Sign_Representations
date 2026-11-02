import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/official_test_job.py'
spec=importlib.util.spec_from_file_location('official_test_job_test',path);J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)


def test_official_cache_identity_counts_and_split():
    lock=dict(checkpoint_sha256='final160',source_sha256={'train':'trainhash','test':'testhash'})
    def cache(split,n):
        return dict(ids=np.array([split+str(i) for i in range(n)]),gloss=np.array([str(i%2731) for i in range(n)]),
                    metadata=np.array(json.dumps(dict(split=split,checkpoint_sha256='final160',source_sha256=lock['source_sha256'][split],limited=False))))
    g,q=cache('train',40154),cache('test',32941)
    J.validate_cache_metadata(g,q,lock)
    q['metadata']=np.array(json.dumps(dict(split='test',checkpoint_sha256='epoch40',source_sha256='testhash',limited=False)))
    with pytest.raises(ValueError,match='Mixed'):J.validate_cache_metadata(g,q,lock)


def test_test_ledger_idempotent_resume_and_no_duplicate_job(tmp_path,monkeypatch):
    monkeypatch.setattr(J,'METHOD',tmp_path)
    job=tmp_path/'job';job.mkdir()
    lock=dict(checkpoint_sha256='final160',source_sha256={'test':'testhash'})
    J.B.atomic_json(job/'lock.json',lock)
    J.ledger_update(job,'scheduled');J.ledger_update(job,'complete','grp',metrics={'R@1':10.})
    J.ledger_update(job,'extracting')
    entries=json.loads((tmp_path/'runtime/test_ledger.json').read_text())
    assert len(entries)==4
    assert next(e for e in entries if e['key'][-1]=='grp')['status']=='complete'
    other=tmp_path/'other';other.mkdir();J.B.atomic_json(other/'lock.json',lock)
    with pytest.raises(ValueError,match='reserved'):J.ledger_update(other,'scheduled')
