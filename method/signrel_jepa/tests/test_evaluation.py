import numpy as np
import pytest
from signrel.eval.retrieval import retrieval
from signrel.eval.phonology import phonological_probes
from signrel.eval.signer_probe import signer_probe


def test_retrieval_analytical_oracle():
    query = np.array([[1.,0.]])
    gallery = np.array([[1.,0.],[0.9,0.1],[0.8,0.2]])
    qr = [{"sample_id":"q","gloss":"a","source_video_id":"q"}]
    gr = [{"sample_id":str(i),"source_video_id":str(i),"gloss":y} for i,y in enumerate(["a","b","a"])]
    result = retrieval(query,gallery,qr,gr)
    assert result["Recall@1"]==1.
    assert result["mAP_instance"]==pytest.approx((1+2/3)/2)


def test_source_self_and_signer_exclusion():
    qr = [{"sample_id":"q","source_video_id":"shared","gloss":"a","signer_id":"p1"}]
    gr = [{"sample_id":"a","source_video_id":"shared","gloss":"b","signer_id":"p2"},
          {"sample_id":"b","source_video_id":"other","gloss":"a","signer_id":"p2"}]
    result = retrieval([[1,0]],[[1,0],[0,1]],qr,gr,exclude_same_signer=True)
    assert result["Recall@1"]==1
    with pytest.raises(ValueError,match="No query"):
        retrieval([[1,0]],[[1,0]],qr,gr[:1])


def test_phonology_protocol_rejects_overlap():
    r=[{"sample_id":"x","signer_id":"p1","gloss":"g","asllex":"c"}]
    with pytest.raises(AssertionError,match="disjoint"):
        phonological_probes([[1,0]],r,[[1,0]],r,{})


def test_signer_probe_gloss_partition():
    rows=[]
    features=[]
    for p in range(3):
        for g in range(40):
            rows.append({"sample_id":f"{p}:{g}","signer_id":str(p),"gloss":str(g)})
            features.append([p,g%2,g%3])
    result=signer_probe(features,rows)
    assert result["status"]=="evaluated"
    assert result["gloss_disjoint"]
    assert result["known_class_coverage"]==1.
