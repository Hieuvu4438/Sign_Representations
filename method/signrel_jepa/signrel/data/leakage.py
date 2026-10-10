"""Conservative source-level exclusion across IDs, URLs, hashes and paths."""
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from .manifest import sha256


def youtube_id(url):
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0] or None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]
    return None


def identities(row, hash_files=False):
    keys = set()
    for k in ("source_video_id", "source_url"):
        if row.get(k):
            keys.add((k, str(row[k])))
    yt = youtube_id(row.get("source_url"))
    if yt:
        keys.add(("source_video_id", yt))
    if row.get("video_path"):
        keys.add(("path", str(Path(row["video_path"]).resolve())))
    digest = row.get("file_sha256")
    if hash_files and row.get("video_path"):
        digest = sha256(row["video_path"])
    if digest:
        keys.add(("file_sha256", digest))
    return keys


def audit_and_filter(pretrain, heldout, hash_files=False):
    held_keys = set().union(*(identities(r, hash_files) for r in heldout))
    contaminated_sources, overlaps = set(), []
    for row in pretrain:
        hit = identities(row, hash_files) & held_keys
        if hit:
            contaminated_sources.add(row["source_video_id"])
            overlaps.append({"sample_id": row["sample_id"], "matches": sorted(hit)})
    clean = [r for r in pretrain if r["source_video_id"] not in contaminated_sources]
    return clean, {"clean": not overlaps, "pretrain_rows": len(pretrain),
                   "heldout_rows": len(heldout), "overlaps": overlaps,
                   "removed_sources": sorted(contaminated_sources), "retained_rows": len(clean),
                   "hash_files": hash_files, "perceptual_duplicates_checked": False}


def assert_disjoint(pretrain, heldout, hash_files=False):
    _, report = audit_and_filter(pretrain, heldout, hash_files)
    if not report["clean"]:
        raise AssertionError(f"Pretrain/evaluation source leakage: {report['overlaps'][:5]}")
    return report
