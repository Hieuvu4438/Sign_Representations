from dataclasses import replace
from pathlib import Path

from ...cache.hashing import file_hash
from ...cli import write_json
from ..manifest import read_manifest, write_manifest


def ingest(root, output):
    root, output = Path(root), Path(output)
    paths = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No JSONL manifests in {root}")
    by_split = {split: [] for split in ("train", "val", "test")}
    for path in paths:
        for record in read_manifest(path):
            video = Path(record.video_path)
            video = (path.parent / video).resolve() if not video.is_absolute() else video
            available = video.is_file()
            by_split[record.split].append(replace(record, video_path=str(video),
                source_sha256=file_hash(video) if available else None,
                availability="available" if available else "missing"))
    if any((output / f"local.{split}.jsonl").exists() for split in by_split):
        raise FileExistsError(f"Refusing to overwrite manifests in {output}")
    summary = {"adapter": "local_jsonl", "source_hashes": {str(p): file_hash(p) for p in paths}, "splits": {}}
    for split, records in by_split.items():
        path = output / f"local.{split}.jsonl"
        write_manifest(path, records)
        summary["splits"][split] = {"count": len(records), "available": sum(r.availability == "available" for r in records), "manifest_sha256": file_hash(path)}
    write_json(output / "import_report.json", summary)
    return summary
