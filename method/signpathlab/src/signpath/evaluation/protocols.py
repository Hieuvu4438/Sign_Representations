from pathlib import Path
from ..cache.hashing import canonical_hash, file_hash
from ..cache.index import CachedCorpus
from ..cli import write_json
from ..data.audit import audit_records


def development_corpora(resolved):
    c = resolved.values
    root = resolved.path(c["cache"]["root"])
    train = CachedCorpus(resolved.path(c["data"]["train_manifest"]), root, split="train", config=c)
    val = CachedCorpus(resolved.path(c["data"]["val_manifest"]), root, split="val", config=c)
    return train, val


def audit_development(resolved, train, val, *, verify_files=True):
    records = list(train.records.values()) + list(val.records.values())
    protocol = resolved.values["experiment"]["evaluation_protocol"]
    signer_disjoint = "signer_disjoint" in protocol
    audit = audit_records(records, signer_disjoint=signer_disjoint, verify_files=verify_files)
    if not audit["passed"]:
        raise ValueError(f"Development split audit failed: {audit['problems'][:3]}")
    if "asl_citizen_official" in protocol and any(r.dataset != "asl_citizen" or not r.official_split_version for r in records):
        raise ValueError("Official ASL Citizen protocol requires actual official-split provenance")
    if train.identity != val.identity:
        raise ValueError("Train/validation caches use different input identities")
    return audit


def verify_pipeline(resolved, output=None):
    import numpy as np
    from ..data.dataset import geometry_kwargs
    from ..geometry.paths import geometry_from_features
    from ..data.fingerprints import cross_split_candidates
    from ..masking.closure import ClosureResolver
    from ..masking.samplers import MaskSampler
    from ..models.tokenizer import SignTokenizer
    train, val = development_corpora(resolved)
    audit = audit_development(resolved, train, val)
    records = list(train.records.values()) + list(val.records.values())
    near = cross_split_candidates(records)
    reviews_file = resolved.path(resolved.values["cache"]["root"]).parent / "near_duplicate_reviews.json"
    if near["candidates"] and reviews_file.exists():
        import json
        reviews = json.loads(reviews_file.read_text())
        if reviews.get("fingerprints_hash") != near["fingerprints_hash"]:
            raise ValueError("Near-duplicate review refers to a different fingerprint pool")
        decisions = {tuple(sorted((item["first"], item["second"]))): item for item in reviews["decisions"]}
        accepted = []
        for candidate in near["candidates"]:
            item = decisions.get(tuple(sorted((candidate["first"], candidate["second"]))))
            if item and item.get("decision") == "distinct" and item.get("reason") and item.get("reviewer"):
                accepted.append(item)
        near["reviewed_distinct"] = accepted
        if len(accepted) == len(near["candidates"]):
            near["status"] = "passed_after_recorded_distinct_review"
    diagnostics = []
    for name in train.available:
        bundle = train.load(name)
        geometry = tuple(geometry_from_features(bundle, **geometry_kwargs(resolved.values))) if resolved.values["geometry"]["enabled"] else ()
        tokens = SignTokenizer().build(bundle, geometry)
        raw = MaskSampler().sample(tokens, np.random.default_rng(resolved.values["experiment"]["seed"]))
        plan = ClosureResolver().resolve(tokens, raw)
        if any(tokens.tokens[i].raw_support & raw.cells for i in plan.visible):
            raise ValueError(f"Support closure failed: {name}")
        if any(not tokens.geometry[i].support <= raw.cells for i in plan.geometry_targets):
            raise ValueError(f"Geometry target eligibility failed: {name}")
        diagnostics.append({"sample_id": name, **plan.diagnostics})
    identity = {"train_corpus": train.fingerprint, "val_corpus": val.fingerprint,
                "protocol": resolved.values["experiment"]["evaluation_protocol"]}
    report = {"status": "passed" if near["status"].startswith("passed") else "manual_review_required", "audit": audit,
              "near_duplicate_review": near, "identity": identity, "identity_hash": canonical_hash(identity),
              "mask_diagnostics": diagnostics, "scope": "development train/val; no test labels accessed"}
    destination = resolved.path(resolved.values["cache"]["root"]).parent / "pipeline_checks.json"
    write_json(destination, report)
    if output:
        write_json(output, report)
    if report["status"] != "passed":
        raise ValueError(f"Near-duplicate candidates need recorded review before training: {destination}")
    return report


def require_pipeline_gate(resolved, train, val):
    import json
    path = resolved.path(resolved.values["cache"]["root"]).parent / "pipeline_checks.json"
    report = json.loads(path.read_text())
    identity = {"train_corpus": train.fingerprint, "val_corpus": val.fingerprint,
                "protocol": resolved.values["experiment"]["evaluation_protocol"]}
    if report["status"] != "passed" or report["identity"] != identity or report["identity_hash"] != canonical_hash(identity):
        raise ValueError("Pipeline verification is absent, stale or failed; rerun verify pipeline")
    return file_hash(path)
