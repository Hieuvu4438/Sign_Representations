import numpy as np
import pytest
from signrel.eval.retrieval import score_retrieval
from signrel.eval.bootstrap import paired_gloss_bootstrap
from signrel.eval.embed import window_starts,temporal_pyramid


def row(i,gloss,signer): return {'video_id':i,'gloss_id':gloss,'signer_id':signer,'source_recording_id':i}


def test_rank_unique_gloss_and_max_exemplar():
    gallery=np.array([[1,0],[.99,.01],[0,1]])
    query=np.array([[.8,.2],[0,1]])
    gr=[row('g1','A','s1'),row('g2','A','s1'),row('g3','B','s1')]
    qr=[row('q1','B','s2'),row('q2','B','s2')]
    result=score_retrieval(query,qr,gallery,gr)
    assert [r['rank'] for r in result['per_query']]==[2,1]
    assert result['micro_r1']==.5 and result['mrr']==.75
    with pytest.raises(ValueError): score_retrieval(query,[row('q1','B','s1'),qr[1]],gallery,gr)


@pytest.mark.parametrize('length',[65,95,129])
def test_long_tail_covered(length):
    starts=window_starts(length)
    covered=set(j for i in starts for j in range(i,min(length,i+64)))
    assert covered==set(range(length))


def test_physical_pooling_padding_and_paired_bootstrap():
    values=np.array([[1.,0],[0,1],[1,0],[0,1]])
    a=temporal_pyramid([values],[np.ones(4,bool)],[0,10,20,30])
    # Explicitly exclude padding before pooling, as production canonical merge does.
    padded=np.concatenate([values,np.zeros((10,2))])
    valid=np.array([True]*4+[False]*10)
    b=temporal_pyramid([padded[valid]],[valid[valid]],[0,10,20,30])
    assert np.allclose(a,b)
    rows=[{'video_id':'q1','gloss_id':'A','r1':0},{'video_id':'q2','gloss_id':'B','r1':0}]
    second=[{**r,'r1':1} for r in rows]
    result=paired_gloss_bootstrap(rows,second)
    assert result['mean_delta']==1 and result['ci95']==[1,1]
