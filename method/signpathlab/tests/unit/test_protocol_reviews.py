from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from signpath.cache.hashing import canonical_hash
from signpath.config import ResolvedConfig
from signpath.evaluation.protocols import verify_pipeline
from model_helpers import tiny_config


@pytest.mark.parametrize("review", [None, "stale", "duplicate", "distinct"])
def test_near_duplicate_review_requires_matching_fingerprint_and_distinct_reason(tmp_path, monkeypatch, review):
    config = tiny_config()
    config["cache"]["root"] = "cache"
    resolved = ResolvedConfig(config, tmp_path, (), canonical_hash(config))
    corpora = [SimpleNamespace(records={}, available=(), fingerprint=part) for part in ("train", "val")]
    monkeypatch.setattr("signpath.evaluation.protocols.development_corpora", lambda _: corpora)
    monkeypatch.setattr("signpath.evaluation.protocols.audit_development", lambda *args: {"passed": True})
    near = {"candidates": [{"first": "a", "second": "b", "hamming_fraction": .01}],
            "fingerprints_hash": "a" * 64, "status": "review_required"}
    monkeypatch.setattr("signpath.data.fingerprints.cross_split_candidates", lambda _: deepcopy(near))
    if review:
        (tmp_path / "near_duplicate_reviews.json").write_text(json.dumps({
            "fingerprints_hash": "b" * 64 if review == "stale" else near["fingerprints_hash"],
            "decisions": [{"first": "a", "second": "b", "decision": review,
                           "reason": "Reviewed frame overlays from two different recordings", "reviewer": "unit_test_reviewer"}]}))
    if review == "distinct":
        report = verify_pipeline(resolved)
        assert report["status"] == "passed" and report["near_duplicate_review"]["status"].startswith("passed_after")
    else:
        with pytest.raises(ValueError):
            verify_pipeline(resolved)
