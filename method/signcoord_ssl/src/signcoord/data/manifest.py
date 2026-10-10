from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from signcoord.provenance import object_hash

REQUIRED = ("schema_version", "sample_id", "dataset_id", "source_video_id", "source_group_id", "media_path",
            "media_sha256", "start_ms", "end_ms", "signer_id", "split", "pose_cache_key", "license_ref")


def read_manifest(path, allowed_splits=None):
    records = []
    ids = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if any(k not in r for k in REQUIRED) or r["schema_version"] != "video_manifest_v1":
            raise ValueError(f"invalid manifest schema at line {lineno}")
        if r["sample_id"] in ids or not 0 <= r["start_ms"] < r["end_ms"]:
            raise ValueError(f"duplicate ID or invalid interval at line {lineno}")
        if allowed_splits is not None and r["split"] not in allowed_splits:
            raise ValueError(f"split {r['split']} not allowed for this operation")
        if not isinstance(r["media_sha256"], str) or len(r["media_sha256"]) != 64:
            raise ValueError("manifest requires a frozen media SHA256")
        ids.add(r["sample_id"])
        records.append(r)
    if not records:
        raise ValueError("empty manifest")
    return records


def audit_overlap(ssl, evaluation, max_examples=1000):
    collisions = [];count=0
    tables=[defaultdict(set) for _ in range(5)]
    def keys(r):
        return (r['media_sha256'],(r['dataset_id'],r['source_video_id']),
                (r['dataset_id'],r['source_group_id']),r.get('duplicate_group_id'),
                (r['dataset_id'],r['signer_id']) if r['signer_id'] is not None else None)
    for index,r in enumerate(evaluation):
        for table,key in zip(tables,keys(r)):
            if key is not None:table[key].add(index)
    for a in ssl:
        possible=set()
        for table,key in zip(tables,keys(a)):
            if key is not None:possible.update(table.get(key,()))
        for index in sorted(possible):
            b=evaluation[index]
            interval = max(a["start_ms"], b["start_ms"]) < min(a["end_ms"], b["end_ms"])
            exact = a["media_sha256"] == b["media_sha256"] and interval
            source = (a.get('source_interval_verified',True) and b.get('source_interval_verified',True)
                      and a["dataset_id"] == b["dataset_id"] and a["source_video_id"] == b["source_video_id"] and interval)
            group = a["dataset_id"] == b["dataset_id"] and a["source_group_id"] == b["source_group_id"]
            dup = a.get("duplicate_group_id") is not None and a.get("duplicate_group_id") == b.get("duplicate_group_id")
            signer = a["signer_id"] is not None and a["dataset_id"] == b["dataset_id"] and a["signer_id"] == b["signer_id"]
            if exact or source or group or dup or signer:
                count+=1
                if len(collisions)<max_examples:collisions.append({"ssl_id": a["sample_id"], "eval_id": b["sample_id"],
                                   "media_interval": exact, "source_interval": source, "session": group, "duplicate": dup, "signer": signer})
    known_signers = all(r["signer_id"] is not None for r in ssl+evaluation)
    return {"exact_source_session_signer_collisions": collisions, "collision_count":count,
            "examples_truncated":count>len(collisions),"passed": count==0,
            "signer_disjoint_verified": known_signers and count==0,
            "near_duplicate_verification": "not_run", "ssl_ordered_ids_hash": object_hash([r["sample_id"] for r in ssl]),
            "evaluation_ordered_ids_hash": object_hash([r["sample_id"] for r in evaluation])}
