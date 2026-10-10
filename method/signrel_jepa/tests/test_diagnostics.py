"""Behavioral checks for frozen representation diagnostics."""
import numpy as np
import pytest

from signrel.eval import diagnostics as d
def test_embedding_rank_and_identical_features():
    x = np.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=float)
    r = d.embedding_health(x)
    assert r["effective_rank"] == pytest.approx(2)
    assert r["mean_pairwise_cosine"] == pytest.approx(-1/3)
    r = d.embedding_health(np.ones((4,2)))
    assert r["effective_rank"] == 0
    assert r["mean_pairwise_cosine"] == pytest.approx(1)


def test_minimal_pair_confusion_and_exclusions():
    # X and Y match location, differ only in handshape; Z differs in both.
    metadata = {"x":{"h":"a","l":"head"}, "y":{"h":"b","l":"head"}, "z":{"h":"c","l":"body"}}
    qr = [{"sample_id":"q", "gloss":"X", "asllex":"x", "source_video_id":"s", "signer_id":"A"}]
    gr = [
        {"sample_id":"x", "gloss":"X", "asllex":"x", "source_video_id":"t", "signer_id":"B"},
        {"sample_id":"y", "gloss":"Y", "asllex":"y", "source_video_id":"u", "signer_id":"B"},
        {"sample_id":"z", "gloss":"Z", "asllex":"z", "source_video_id":"v", "signer_id":"B"},
        {"sample_id":"cheat", "gloss":"X", "asllex":"x", "source_video_id":"s", "signer_id":"C"}]
    result = d.minimal_pair_retrieval(np.array([[1,0]]),np.array([[0,1],[.9,.1],[1,0],[1,0]]),qr,gr,metadata,["h","l"])
    assert result["queries"] == 1
    assert result["Recall@1"] == 0
    assert result["per_query"][0]["predicted_gloss"] == "Y"
    assert result["per_query"][0]["candidate_classes"] == 2
    assert result["per_property"]["h"]["Recall@1"] == 0
    assert result["per_property"]["l"]["queries"] == 0
    assert len(result["lexical_pairs"]) == 1
    gr[0]["signer_id"] = "A"
    result = d.minimal_pair_retrieval(np.array([[1,0]]),np.array([[0,1],[.9,.1],[1,0],[1,0]]),qr,gr,metadata,["h","l"])
    assert result["queries"] == 0
    assert result["skipped"][0]["reason"] == "no_eligible_positive_or_distractor"


def test_ambiguous_phonology_mapping_fails():
    rows = [{"sample_id":"1", "gloss":"A", "asllex":"a"}, {"sample_id":"2", "gloss":"A", "asllex":"b"}]
    with pytest.raises(AssertionError,match="Ambiguous"):
        d.lexical_signatures(rows,{"a":{"h":"1","l":"2"},"b":{"h":"3","l":"2"}},["h","l"])
