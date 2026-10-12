from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from signpath.cache.hashing import canonical_hash
from signpath.config import ResolvedConfig
from signpath.data.manifest import ManifestRecord
from signpath.preprocessing import ensure_raw_coordinates
from model_helpers import tiny_config


def test_raw_coordinate_sidecar_saves_pixels_and_normalization_prerequisites(tmp_path):
    c = tiny_config()
    resolved = ResolvedConfig(c, tmp_path, (), canonical_hash(c))
    xy = np.full((3, 49, 2), .5, dtype=np.float32)
    xy[:, 1], xy[:, 2] = [.4, .5], [.6, .5]
    xy[:, 7] = [.5, .7]
    observed = np.ones((3, 49), dtype=bool)
    observed[2, 1] = False
    xy[2, 1] = np.nan
    raw = SimpleNamespace(xy=xy, observed=observed, quality=observed.astype(np.float32))
    video = SimpleNamespace(frames=np.zeros((3, 480, 640, 3), dtype=np.uint8), timestamps=np.arange(3) / 25,
                            source_frame_index=np.arange(3, dtype=np.int64))
    np.savez(tmp_path / "landmarks.npz", original_raw=xy)
    record = ManifestRecord(1, "fixture", "fixture", "ase", "train", "source", "unused.mp4", "a" * 64)
    output = ensure_raw_coordinates(tmp_path, record, resolved, raw, video)
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["xy_image_px"][0, 7], [320, 336], atol=1e-4)
        np.testing.assert_allclose(data["shoulder_center_px"][0], [320, 240], atol=1e-4)
        assert data["shoulder_scale_px"][0] == pytest.approx(128)
        assert np.isnan(data["shoulder_scale_px"][2]) and not data["pose_coordinate_valid"][2].any()
    assert ensure_raw_coordinates(tmp_path, record, resolved) == output
    with output.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="mismatch"):
        ensure_raw_coordinates(tmp_path, record, resolved)
