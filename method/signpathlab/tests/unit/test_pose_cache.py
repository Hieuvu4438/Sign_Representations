from dataclasses import replace
import json
import numpy as np
import pytest
from signpath.cache.reader import read_bundle
from signpath.cache.schema import IDENTITY_FIELDS
from signpath.cache.writer import write_bundle
from signpath.pose.coordinates import body_coordinates, static_pose_streams


def test_aspect_ratio_and_invalid_shoulders():
    raw = np.ones((2, 49, 2)) * .5
    raw[:, 1], raw[:, 2] = [.4, .5], [.6, .5]
    raw[:, 7] = [.5, .7]
    observed = np.ones((2, 49), dtype=bool)
    observed[1, 1] = False
    raw[1, 1] = np.nan
    out = body_coordinates(raw, observed, np.ones((2, 49)), [[640, 480], [640, 480]])
    np.testing.assert_allclose(out.xy[0, 7], [0, .75], atol=1e-6)
    assert not out.coordinate_valid[1].any()
    assert not out.xy[1].any() and not out.quality[1].any()
    assert np.isnan(out.scale_px[1])


def test_static_pose_keeps_shape_location(bundle):
    streams = static_pose_streams(bundle.pose_xy_body, bundle.pose_quality, bundle.pose_coordinate_valid)
    assert [s.shape[-1] for s in streams] == [28, 84, 84]
    np.testing.assert_allclose(streams[1][0].reshape(21, 4)[:, :2], bundle.pose_xy_body[0, 7:28])


def test_cache_roundtrip_and_identity(tmp_path, bundle):
    bundle.validate(allow_fixture=True)
    path = tmp_path / "sample"
    write_bundle(path, bundle, allow_fixture=True)
    expected = {key: bundle.meta[key] for key in IDENTITY_FIELDS}
    actual = read_bundle(path, expected, allow_fixture=True)
    np.testing.assert_array_equal(actual.visual, bundle.visual)
    assert actual.raw_supports == bundle.raw_supports
    assert isinstance(actual.visual, np.memmap)
    with pytest.raises(ValueError, match="Synthetic"):
        read_bundle(path, expected)
    with pytest.raises(ValueError, match="identity mismatch"):
        read_bundle(path, {**expected, "weights_sha256": "b" * 64}, allow_fixture=True)
    with pytest.raises(FileExistsError):
        write_bundle(path, bundle, allow_fixture=True)
    with (path / "visual.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(path, expected, allow_fixture=True)


def test_cache_invalid_schema(bundle):
    with pytest.raises(ValueError, match="stream order"):
        replace(bundle, meta={**bundle.meta, "visual_stream_order": ["face"]}).validate(allow_fixture=True)
    with pytest.raises(ValueError, match="expected float64"):
        replace(bundle, timestamps=bundle.timestamps.astype(np.float32)).validate(allow_fixture=True)
    broken = bundle.pose_xy_body.copy()
    broken[0, 7] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        replace(bundle, pose_xy_body=broken).validate(allow_fixture=True)
