"""Separate structural SSL records from evaluation annotations."""
import hashlib
import json
from pathlib import Path

FORBIDDEN_SSL_KEYS = {
    "gloss", "class_id", "translation", "caption", "text", "signer_id", "phonology_label",
    "label", "labels", "gloss_raw", "translation_raw", "lexical_id", "lexical_mapping_id", "asllex",
}
SSL_KEYS = {
    "sample_id", "video_path", "source_video_id", "source_url", "file_sha256",
    "start_sec", "end_sec", "fps", "landmark_path", "split", "signer_crop",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample_id in manifest")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ssl_record(row):
    """Fail rather than silently dropping annotations from a purported SSL manifest."""
    assert_no_labels(row)
    if row.get("split") != "ssl_train":
        raise ValueError("SSL records must explicitly declare split=ssl_train")
    for key in ("sample_id", "video_path", "source_video_id"):
        if not row.get(key):
            raise ValueError(f"Missing required SSL field: {key}")
    # dataset/language metadata may be stored in the manifest but never enter a batch.
    return {k: v for k, v in row.items() if k in SSL_KEYS}


def assert_no_labels(value):
    if isinstance(value, dict):
        illegal = FORBIDDEN_SSL_KEYS.intersection(value)
        if illegal:
            raise AssertionError(f"Semantic labels at SSL boundary: {sorted(illegal)}")
        for v in value.values():
            assert_no_labels(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            assert_no_labels(v)
