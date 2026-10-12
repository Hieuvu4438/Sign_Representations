from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from signpath.cache.hashing import file_hash
from signpath.data.adapters.asl_citizen import ingest
from signpath.data.audit import audit_records
from signpath.data.decode import nearest_observations
from signpath.data.manifest import ManifestRecord, read_manifest, write_manifest


def record(**kwargs):
    base = dict(schema_version=1, sample_id="clip", dataset="local", language="ase", split="train",
                source_group_id="source", video_path="clip.mp4", source_sha256="a" * 64, signer_id="P1", availability="available")
    base.update(kwargs)
    return ManifestRecord(**base)


def test_manifest_roundtrip_and_label_boundary(tmp_path):
    r = record(label_id=0, gloss="label", text="private", metadata={"label_alias": "hidden"})
    write_manifest(tmp_path / "manifest.jsonl", [r])
    assert read_manifest(tmp_path / "manifest.jsonl") == [r]
    view = asdict(r.pretrain_view())
    assert not {"label_id", "gloss", "text", "metadata", "signer_id"} & view.keys()
    with pytest.raises(ValueError, match="held-out"):
        replace(r, split="val").pretrain_view()


@pytest.mark.parametrize("field", ["source_group_id", "source_sha256", "signer_id"])
def test_overlap_audit(field):
    a = record()
    b = record(sample_id="other", split="val", source_group_id="other-source", source_sha256="b" * 64, signer_id="P2")
    assert audit_records([a, b])["passed"]
    b = replace(b, **{field: getattr(a, field)})
    report = audit_records([a, b])
    assert not report["passed"]
    assert any(p["kind"] == "cross_split_" + field for p in report["problems"])


def test_audit_missing_hash_and_source_mutation(tmp_path):
    path = tmp_path / "video"
    path.write_bytes(b"video")
    r = record(video_path=str(path), source_sha256=file_hash(path))
    assert audit_records([r], verify_files=True)["passed"]
    path.write_bytes(b"corrupt")
    assert not audit_records([r], verify_files=True)["passed"]
    assert not audit_records([replace(r, source_sha256=None)])["passed"]


def test_asl_citizen_actual_columns(tmp_path):
    root, output = tmp_path / "ASL_Citizen", tmp_path / "manifests"
    (root / "splits").mkdir(parents=True)
    (root / "videos").mkdir()
    for index, split in enumerate(("train", "val", "test")):
        with (root / "splits" / f"{split}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Participant ID", "Video file", "Gloss", "ASL-LEX Code"])
            writer.writerow([f"P{index}", f"{split}.mp4", "APPLE", "A_03_054"])
        if split != "test":
            (root / "videos" / f"{split}.mp4").write_bytes(split.encode())
    report = ingest(root, output)
    records = [r for p in output.glob("*.jsonl") for r in read_manifest(p)]
    assert report["splits"]["test"]["available"] == 0
    assert {r.label_id for r in records} == {0}
    assert audit_records(records)["passed"]
    assert read_manifest(output / "asl_citizen.train.jsonl")[0].source_sha256 == file_hash(root / "videos/train.mp4")
    with pytest.raises(FileExistsError):
        ingest(root, output)


def test_vfr_nearest_and_aliases():
    times = np.array([0, 0.07, 0.2], dtype=np.float64)
    selection = nearest_observations(times, np.arange(3, dtype=np.int64), 25)
    np.testing.assert_array_equal(selection.indices, [0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(selection.actual_timestamps, times[selection.indices])
    np.testing.assert_array_equal(selection.duplicate_selection, [False, True, True, True, True, True])
    with pytest.raises(ValueError):
        nearest_observations(times[::-1], np.arange(3, dtype=np.int64), 25)
