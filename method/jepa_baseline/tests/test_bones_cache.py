import json
import numpy as np
import pytest
from signrep_baseline.bones import corpus_bone_statistics, load_bone_statistics
from signrep_baseline.cache import load_standardized


def test_exact_corpus_lengths_ignore_missing_and_provenance(tmp_path):
    # Unequal clip lengths distinguish a corpus median from median-of-video-medians.
    paths = []
    for i, scales in enumerate([[1., 2., 7.], [8.]]):
        frames = len(scales)
        body = np.zeros((frames, 22, 3, 1), np.float32)
        hand = np.zeros((frames, 21, 3, 1), np.float32)
        body[:, 1:, 1, 0] = np.array(scales)[:, None]
        hand[:, 1:, 0, 0] = np.array(scales)[:, None] * .1
        path = tmp_path / f"{i}.npz"
        np.savez(path, body_offsets_full=body, right_offsets_full=hand,
                 left_offsets_full=hand * 1000, body_detected=np.ones(frames, bool),
                 right_detected=np.ones(frames, bool), left_detected=np.zeros(frames, bool))
        paths.append(path)
    out = tmp_path / "stats.npz"
    metadata = corpus_bone_statistics(paths, out, "manifest-pin")
    stats, _ = load_bone_statistics(out, "manifest-pin")
    assert metadata["sample_counts"] == {"body": 4, "hand": 4}
    np.testing.assert_allclose(stats["body_median_lengths"][1:], 4.5)
    np.testing.assert_allclose(stats["hand_median_lengths"][1:], .45)
    assert stats["body_median_lengths"][0] == 0
    with pytest.raises(ValueError, match="different pretraining manifest"):
        load_bone_statistics(out, "other-pin")


def test_refuse_raw_and_dropped_frame_caches(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez(path, metadata_json=np.array(json.dumps({"schema": "signrep_teacher_v1"})))
    with pytest.raises(ValueError, match="corpus-standardized"):
        load_standardized(path)
    fields = {"pose_3d": np.zeros((2, 61, 3)), "confidence": np.ones((2, 61)),
              "body_angles": np.zeros((2, 22)), "right_angles": np.zeros((2, 41)),
              "left_angles": np.zeros((2, 41)), "frame_indices": np.array([0, 2]),
              "body_detected": np.ones(2, bool), "right_detected": np.ones(2, bool), "left_detected": np.ones(2, bool),
              "metadata_json": np.array(json.dumps({"schema": "signrep_standardized_v1", "decoded_frames": 2}))}
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="dropped/reordered"):
        load_standardized(path)
