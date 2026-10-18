import numpy as np
import pytest
from signrepr.evaluation.retrieval import rank_dictionary


def test_gloss_max_ties_failures_and_denominator():
    gallery = np.array([[1, 0], [1, 0], [0, 1], [np.nan, 0]])
    queries = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    result = rank_dictionary(gallery, queries, ['g1', 'g2', 'g3', 'g4'], ['q1', 'q2', 'q3', 'q4'], ['b', 'a', 'c', 'd'], ['b', 'c', 'a', 'missing'])
    assert [r['rank'] for r in result['queries']] == [2, 1, None, None]
    assert result['metrics']['R@1'] == .25
    assert result['metrics']['R@5'] == .5
    assert result['metrics']['MRR'] == .375
    assert result['metrics']['DCG_single_relevant'] == pytest.approx((1+1/np.log2(3))/4)
    assert result['metrics']['num_queries'] == 4
    assert result['covered_gallery_subset']['num_queries'] == 3
    assert 'mAP' not in result['metrics']


def test_video_protocol_multiple_relevant():
    result = rank_dictionary([[1, 0], [0, 1], [.5, .5]], [[1, 0]], ['g1', 'g2', 'g3'], ['q'], ['a', 'b', 'a'], ['a'], 'video_rank_multiple_relevant')
    assert result['metrics']['mAP'] == 1
    assert result['metrics']['nDCG'] == 1
    assert 'DCG_single_relevant' not in result['metrics']
