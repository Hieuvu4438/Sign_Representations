import numpy as np
import pytest
import torch

from signcoord.eval.bootstrap import paired_group_bootstrap
from signcoord.eval.ctc import ctc_loss,edit_counts,greedy_decode,min_ctc_length
from signcoord.eval.retrieval import evaluate_retrieval


def test_instance_ranking_ties_missing_classes_and_chunk_reference():
    g=np.array([[1.,0.],[1.,0.],[0.,1.]],np.float32)
    q=np.array([[1.,0.],[0.,1.],[1.,0.]],np.float32)
    labels=[0,1,2]; qlabels=[1,2,3]
    args=(g,q,labels,qlabels,["z","a","b"],["q0","q1","q2"],["S1","S1","S2"])
    result=evaluate_retrieval(*args,chunk=1)
    assert result==evaluate_retrieval(*args,chunk=99)
    assert result["per_query"][0]["top_gallery_ids"][:2]==["a","z"]
    assert result["metrics"]["rec1"]==pytest.approx(2/3)
    assert result["queries_without_gallery_match"]==1
    assert result["per_query"][2]["rr"]==0


def test_cluster_bootstrap_uses_paired_ids():
    a=[{"query_id":str(i),"group":str(i//3),"rec1":1.} for i in range(9)]
    b=[{**r,"rec1":0.} for r in a]
    report=paired_group_bootstrap(a,b,draws=100)
    assert report["ci95"]==[1.,1.] and report["groups"]==3
    with pytest.raises(ValueError): paired_group_bootstrap(a,b[::-1])


def test_ctc_repeats_blanks_and_infeasibility():
    assert greedy_decode([1,1,0,1,2,2,0])==[1,1,2]
    assert min_ctc_length([1,1,2])==4
    logits=torch.randn(4,1,3,requires_grad=True)
    loss=ctc_loss(logits,[[1,1,2]],torch.tensor([4])); loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError,match="infeasible"):
        ctc_loss(logits,[[1,1,2]],torch.tensor([3]))
    assert edit_counts([1,2,3],[1,4,3])["substitutions"]==1
    assert edit_counts([1,2],[])=={"substitutions":0,"deletions":2,"insertions":0,"reference_tokens":2,"wer":1.}
