"""Read public split metadata; write aggregate counts, never load video/pose/model.

Run from any directory. This is provenance for the literature report, not an
evaluation program and not a train/test selection tool.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import itertools
import json
from pathlib import Path


def audit(root: Path) -> dict:
    records = {}
    result = {
        "audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "metadata-only; no video, pose, checkpoint or retrieval scoring",
        "root": str(root),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        source = root / "splits" / f"{split}.csv"
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        required = {"Participant ID", "Video file", "Gloss", "ASL-LEX Code"}
        if fields is None or not required.issubset(fields):
            raise ValueError(f"Unexpected split schema: {source}")
        records[split] = rows
        signers = collections.Counter(row["Participant ID"] for row in rows)
        glosses = collections.Counter(row["Gloss"] for row in rows)
        codes = collections.Counter(row["ASL-LEX Code"] for row in rows)
        files = collections.Counter(row["Video file"] for row in rows)
        mapping = collections.defaultdict(set)
        for row in rows:
            mapping[row["ASL-LEX Code"]].add(row["Gloss"])
        result["splits"][split] = {
            "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "columns": fields,
            "rows": len(rows),
            "n_signers": len(signers),
            "n_glosses": len(glosses),
            "n_asllex_codes": len(codes),
            "empty_asllex_codes": codes.get("", 0),
            "signer_counts": dict(sorted(signers.items())),
            "gloss_count_range": [min(glosses.values()), max(glosses.values())],
            "duplicate_video_names": sum(n - 1 for n in files.values()),
            "codes_mapping_to_multiple_glosses": {
                code: sorted(values)
                for code, values in sorted(mapping.items())
                if len(values) > 1
            },
        }
    result["overlap"] = {}
    for left, right in itertools.combinations(records, 2):
        entry = {}
        for field in ("Participant ID", "Video file", "Gloss", "ASL-LEX Code"):
            intersection = {r[field] for r in records[left]} & {
                r[field] for r in records[right]
            }
            entry[field] = {
                "n": len(intersection),
                "values_if_small": sorted(intersection) if len(intersection) <= 20 else None,
            }
        result["overlap"][f"{left}__{right}"] = entry
    candidates = {
        "aslc_keypoints": root / "keypoints",
        "aslc_videos": root / "videos",
        "wlasl": Path("/home/dongvk/datasets/WLASL2000"),
        "msasl": Path("/home/dongvk/datasets/MS-ASL"),
        "how2sign_shared": Path("/home/shared_data/sign_language/How2Sign"),
        "youtubeasl_local": Path("/home/dongvk/datasets/youtubeASL"),
    }
    result["directory_presence_only"] = {
        name: {"path": str(path), "exists": path.is_dir()}
        for name, path in candidates.items()
    }
    result["limitations"] = [
        "CSV row counts do not verify video/pose availability or detector quality.",
        "No source-video-ID/time-range deduplication was performed.",
        "Test metadata was counted; no model was scored or selected using test.",
        "ASL-LEX code grouping does not establish linguistic equivalence of variants.",
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/dongvk/datasets/ASL_Citizen"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("local_dataset_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({s: {k: result["splits"][s][k] for k in ("rows", "n_signers", "n_glosses", "n_asllex_codes")} for s in result["splits"]}, ensure_ascii=False))
