"""Split/vocab protocol: disjoint signers/videos, one train-only mapping, unknown glosses detected."""
import csv

import pytest

from asm_jepa.data.manifest import (ProtocolError, audit_splits, build_split_rows, build_vocab, read_official_csv,
                                    sample_id_for)

HEADERS = {"signer": "Participant ID", "filename": "Video file", "gloss": "Gloss"}
POS = {"signer": 0, "filename": 1, "gloss": 2}


def _csv(path, rows, header=("Participant ID", "Video file", "Gloss", "ASL-LEX Code")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_vocab_train_only_and_shared_mapping(tmp_path):
    tr, _ = read_official_csv(_csv(tmp_path / "train.csv", [["P1", "a-APPLE.mp4", "APPLE", "x"],
                                                            ["P2", "b-ZEBRA.mp4", "ZEBRA ", "x"],
                                                            ["P1", "c-APPLE.mp4", "APPLE", "x"]]), HEADERS, POS)
    va, _ = read_official_csv(_csv(tmp_path / "val.csv", [["P3", "d-ZEBRA.mp4", "ZEBRA", "x"]]), HEADERS, POS)
    vocab = build_vocab(tr)
    assert vocab == {"APPLE": 0, "ZEBRA": 1}  # sorted(unique(strip)), no case/synonym changes
    rows_tr = build_split_rows("train", tr, vocab, "videos")
    rows_va = build_split_rows("val", va, vocab, "videos")
    assert rows_va[0]["class_id"] == vocab["ZEBRA"] == rows_tr[1]["class_id"]
    assert rows_tr[0]["sample_id"] == sample_id_for("videos/a-APPLE.mp4")


def test_unknown_val_gloss_is_a_protocol_error(tmp_path):
    tr, _ = read_official_csv(_csv(tmp_path / "train.csv", [["P1", "a.mp4", "APPLE", "x"]]), HEADERS, POS)
    te, _ = read_official_csv(_csv(tmp_path / "test.csv", [["P9", "z.mp4", "NEWSIGN", "x"]]), HEADERS, POS)
    with pytest.raises(ProtocolError):
        build_split_rows("test", te, build_vocab(tr), "videos")


def test_positional_adapter_when_headers_differ(tmp_path):
    rows, info = read_official_csv(_csv(tmp_path / "t.csv", [["P1", "a.mp4", "APPLE"]], header=("u", "f", "g")),
                                   HEADERS, POS)
    assert info["mode"] == "positional" and rows[0] == {"signer": "P1", "filename": "a.mp4", "gloss": "APPLE"}


def test_audit_detects_signer_and_video_overlap():
    def r(sid, signer):
        return {"sample_id": sid, "signer_id": signer, "class_id": 0}

    ok = audit_splits({"train": [r("a", "P1")], "val": [r("b", "P2")], "test": [r("c", "P3")]})
    assert ok["disjoint"]
    bad = audit_splits({"train": [r("a", "P1")], "val": [r("b", "P1")], "test": [r("a", "P3")]})
    assert not bad["disjoint"]
    assert bad["overlaps"]["train_val"]["signers"] == ["P1"]
    assert bad["overlaps"]["train_test"]["videos"] == 1
