import json
from pathlib import Path
import numpy as np
import pytest
from signrep_baseline import preparation
from signrep_baseline.cache import sha256
from signrep_baseline.corpus_policy import validate_pose_policy


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture_sources(tmp_path):
    rows = []
    for index, ident in enumerate(("a", "short", "c")):
        video = tmp_path / (ident + ".mp4")
        video.write_bytes(ident.encode())
        cache = tmp_path / "raw" / f"{index:08d}.npz"
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, metadata_json=np.array(json.dumps({
            "schema": "signrep_teacher_v1", "video_sha256": sha256(video),
            "body_root_mode": "canonical", "decoded_frames": 11 if ident == "short" else 16})))
        rows.append({"video_id": ident, "source_id": ident, "dataset": "local",
                     "source_split": "train", "split": "pretrain", "video": str(video),
                     "label": 0, "signer_bbox_xyxy": [0, 0, 64, 48],
                     "raw_cache": str(cache), "raw_cache_sha256": sha256(cache)})
    return rows


def test_prepare_reuses_source_bound_caches_after_removing_middle_video(tmp_path, monkeypatch):
    rows = fixture_sources(tmp_path)
    source, reference, held = (tmp_path / name for name in ("source.jsonl", "train.jsonl", "held.jsonl"))
    write_rows(source, [rows[0], rows[2]])
    write_rows(reference, [{**r, "split": "train"} for r in rows])
    write_rows(held, [{"video_id": "held", "source_id": "held", "dataset": "local", "split": "val"}])
    observed = []

    class Capture:
        def __init__(self, *args):
            pass

        def get(self, key):
            return 64 if key == 3 else 48

        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", Capture)

    def bones(paths, output, manifest):
        observed.extend(map(str, paths))
        Path(output).write_bytes(b"technical_statistics")

    monkeypatch.setattr(preparation, "corpus_bone_statistics", bones)

    def normalize(python, command, weights, **paths):
        assert command == "normalize-batch", "No cache should be re-extracted"
        tasks = json.loads(Path(paths["manifest"]).read_text())
        assert [t["raw"] for t in tasks] == [rows[0]["raw_cache"], rows[2]["raw_cache"]]
        for task in tasks:
            target = Path(task["output"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(task["raw"]).read_bytes())
        return ""

    monkeypatch.setattr(preparation, "pose_command", normalize)
    output = tmp_path / "corpus"
    audit = preparation.prepare_corpus(source, None, output, "unused", "unused", held, reference)
    prepared = [json.loads(s) for s in (output / "pretrain.jsonl").read_text().splitlines()]
    assert [r["video_id"] for r in prepared] == ["a", "c"]
    assert [r["raw_cache"] for r in prepared] == observed == [rows[0]["raw_cache"], rows[2]["raw_cache"]]
    assert prepared[1]["pose_cache"].endswith("normalized/00000001.npz")
    assert audit["retained"] == 2


def policy_job(tmp_path):
    rows = fixture_sources(tmp_path)
    write_rows(tmp_path / "source.jsonl", rows)
    effective = [rows[0], rows[2]]
    write_rows(tmp_path / "source_training.jsonl", effective)
    write_rows(tmp_path / "manifests/gallery_train.jsonl", [{**r, "split": "train"} for r in effective])
    report = {"results": [{"video_id": r["video_id"], "status": "NO_VALID_WINDOW" if r["video_id"] == "short" else "PASS",
                "decoded_frames": 11 if r["video_id"] == "short" else 16,
                "valid_body_windows": 0 if r["video_id"] == "short" else 1,
                "raw": r["raw_cache"], "raw_sha256": r["raw_cache_sha256"]} for r in rows], "errors": []}
    (tmp_path / "pose_report.json").write_text(json.dumps(report))
    approval = {"approved_by_user": True, "entries": [{"video_id": "short", "decoded_frames": 11,
                "reason": "shorter_than_16_decoded_frames"}]}
    (tmp_path / "approved_exclusions.json").write_text(json.dumps(approval))
    opts = {"training_source_manifest": str(tmp_path / "source_training.jsonl"),
            "full_gallery_manifest": str(tmp_path / "manifests/gallery_train.jsonl"),
            "approved_exclusions": str(tmp_path / "approved_exclusions.json")}
    return opts, report, rows


def test_policy_accepts_only_approved_exclusion_without_rewriting_official_report(tmp_path):
    opts, report, rows = policy_job(tmp_path)
    before = sha256(tmp_path / "pose_report.json")
    result = validate_pose_policy(tmp_path, opts)
    assert result["effective_train_videos"] == result["gallery_videos"] == 2
    assert result["excluded_video_ids"] == ["short"]
    assert sha256(tmp_path / "pose_report.json") == before


def test_policy_rejects_unapproved_other_failure_and_shifted_cache(tmp_path):
    opts, report, rows = policy_job(tmp_path)
    report["results"][0]["status"] = "NO_VALID_WINDOW"
    (tmp_path / "pose_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differ from the explicit"):
        validate_pose_policy(tmp_path, opts)
    report["results"][0]["status"] = "PASS"
    (tmp_path / "pose_report.json").write_text(json.dumps(report))
    shifted = {**rows[2], "raw_cache": rows[1]["raw_cache"], "raw_cache_sha256": rows[1]["raw_cache_sha256"]}
    write_rows(tmp_path / "source_training.jsonl", [rows[0], shifted])
    with pytest.raises(ValueError, match="not bound"):
        validate_pose_policy(tmp_path, opts)


def test_policy_requires_actual_short_clip_and_user_authorization(tmp_path):
    opts, report, rows = policy_job(tmp_path)
    report["results"][1]["decoded_frames"] = 32
    (tmp_path / "pose_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="observed short clip"):
        validate_pose_policy(tmp_path, opts)
    approval = json.loads((tmp_path / "approved_exclusions.json").read_text())
    approval["approved_by_user"] = False
    (tmp_path / "approved_exclusions.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="user authorization"):
        validate_pose_policy(tmp_path, opts)
