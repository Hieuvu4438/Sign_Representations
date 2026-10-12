from collections import defaultdict
from pathlib import Path

from ..cache.hashing import file_hash
from .manifest import read_manifest


def audit_records(records, *, require_hashes=True, signer_disjoint=True, verify_files=False):
    records = list(records)
    problems, missing_hashes, missing_signers = [], [], []
    if not records:
        problems.append({"kind": "empty_manifest"})
    indexes = {key: defaultdict(list) for key in ("sample_id", "source_group_id", "source_sha256", "signer_id")}
    sample_ids = set()
    counts = defaultdict(lambda: {"count": 0, "available": 0, "classes": set()})
    for record in records:
        if record.sample_id in sample_ids:
            problems.append({"kind": "duplicate_sample_id", "sample_id": record.sample_id})
        sample_ids.add(record.sample_id)
        counts[record.split]["count"] += 1
        counts[record.split]["available"] += record.availability == "available"
        if record.label_id is not None:
            counts[record.split]["classes"].add(record.label_id)
        if record.availability == "available" and record.source_sha256 is None:
            missing_hashes.append(record.sample_id)
        if record.signer_id is None:
            missing_signers.append(record.sample_id)
        if verify_files and record.availability == "available":
            path = Path(record.video_path)
            if not path.is_file() or (record.source_sha256 is not None and file_hash(path) != record.source_sha256):
                problems.append({"kind": "source_file_mismatch", "sample_id": record.sample_id})
        for key, index in indexes.items():
            value = getattr(record, key)
            if value is not None:
                index[value].append(record)
    for key, index in indexes.items():
        if key == "signer_id" and not signer_disjoint:
            continue
        for value, members in index.items():
            splits = sorted({r.split for r in members})
            if len(splits) > 1:
                problems.append({"kind": f"cross_split_{key}", "value": value, "splits": splits,
                                 "sample_ids": [r.sample_id for r in members]})
    return {"passed": not problems and not (require_hashes and missing_hashes) and not (signer_disjoint and missing_signers),
            "problems": problems, "missing_hashes": missing_hashes, "missing_signer_ids": missing_signers,
            "counts": {s: {**v, "classes": len(v["classes"])} for s, v in counts.items()},
            "near_duplicate_review": {"status": "not_run", "reason": "Perceptual fingerprints and candidate review required before research pilot"}}


def audit_directory(root):
    paths = sorted(Path(root).glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No manifests in {root}")
    report = audit_records([r for path in paths for r in read_manifest(path)], verify_files=True)
    report["manifest_sha256"] = {str(p): file_hash(p) for p in paths}
    return report
